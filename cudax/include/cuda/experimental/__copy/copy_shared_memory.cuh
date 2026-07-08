//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_COPY_SHARED_MEMORY_H
#define _CUDAX__COPY_COPY_SHARED_MEMORY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__launch/configuration.h>
#include <cuda/__launch/launch.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__mdspan/default_accessor.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/array>

#include <cuda/experimental/__lazy_jit/dispatch.cuh>
#include <cuda/experimental/__copy/tensor_iterator.cuh>
#include <cuda/experimental/__copy_bytes/types.cuh>
#include <cuda/experimental/__copy/copy_shared_memory_kernel.cuh>

#include <cuda/std/__cccl/prologue.h>

//! Shared-memory tiled transpose for arbitrary-rank tensor copies.
//!
//! The overall idea is to decompose the tensors into tiles that can fit in shared memory.
//! Each tile is assigned to a thread block. A tile can entirely represent a dimension or split the respective extent.
//! The algorithm creates tiles over dimensions that provide coalesced accesses in the source and destination tensors.
//!
//! (1) Grid decomposition
//! The tensor is partitioned into tiles whose per-dimension sizes are capped by warp size and shared-memory capacity.
//! The total number of tiles (product of ceil(extent[d] / tile_size[d]) over all dimensions) becomes the 1-D grid size.
//!
//! (2) Block processing
//! Each block handles one tile in two phases:
//!   1. *Load*: threads cooperatively read source elements into shared memory.
//!      This requires additional logic to "transpose" the source tensor into a row-major order.
//!      The mapping is determined by using the source-tile permutation obtained by sorting by |src stride|.
//!   2. *Store*: after a barrier, threads read shared memory in destination-coalesced order by using the
//!      destination-tile permutation obtained by sorting by |dst stride|.
//!
//! Boundary tiles that extend past the tensor extents fall back to a direct element-wise copy without shared memory.

namespace cuda::experimental
{

#if !_CCCL_COMPILER(NVRTC)

//! @brief Launch the shared-memory tiled transpose kernel.
//!
//! Precomputes the source/destination-coalesced permutations and tile shapes, constructs coordinate iterators, then
//! launches one block per tile.
//!
//! @pre `__src.__rank >= 2`
//!
//! @param[in]  __src          Source raw tensor descriptor
//! @param[out] __dst          Destination raw tensor descriptor
//! @param[in]  __stream       CUDA stream for asynchronous execution
//! @param[in]  __src_accessor Accessor for reading source elements
//! @param[in]  __dst_accessor Accessor for writing destination elements
template <typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut,
          typename _TpIn,
          typename _TpOut,
          ::cuda::std::size_t _MaxRank,
          typename _SrcAccessor,
          typename _DstAccessor>
_CCCL_HOST_API ::cuda::experimental::lazy_jit::dispatch_ret_type __launch_copy_shared_mem_kernel(
  const __raw_tensor<_ExtentT, _StrideTIn, _TpIn, _MaxRank>& __src,
  const __raw_tensor<_ExtentT, _StrideTOut, _TpOut, _MaxRank>& __dst,
  ::cuda::stream_ref __stream,
  const _SrcAccessor& __src_accessor = {},
  const _DstAccessor& __dst_accessor = {})
{
  namespace cudax = ::cuda::experimental;
  using ::cuda::std::size_t;
  _CCCL_ASSERT(__src.__rank >= 2, "Rank must be at least 2 for shared memory transpose");

  const auto __tiling          = cudax::__find_shared_mem_tiling<_TpIn>(__src, __dst);
  const auto __tile_sizes      = __tiling.__tile_sizes;
  const auto __rank            = __src.__rank;
  const auto __tile_total_size = __tiling.__tile_total_size;

  //--------------------------------------------------------------------------------------------------------------------
  // Find the grid size (number of blocks) and strides for block index decomposition
  ::cuda::std::array<_ExtentT, _MaxRank> __grid_tile_sizes{};
  ::cuda::std::array<_StrideTIn, _MaxRank> __grid_tile_src_strides{};
  ::cuda::std::array<_StrideTOut, _MaxRank> __grid_tile_dst_strides{};
  _ExtentT __grid_size = 1;
  for (size_t __i = 0; __i < __rank; ++__i)
  {
    __grid_tile_sizes[__i]       = ::cuda::ceil_div(__src.__extents[__i], static_cast<_ExtentT>(__tile_sizes[__i]));
    __grid_tile_src_strides[__i] = static_cast<_StrideTIn>(__tile_sizes[__i]) * __src.__strides[__i];
    __grid_tile_dst_strides[__i] = static_cast<_StrideTOut>(__tile_sizes[__i]) * __dst.__strides[__i];
    __grid_size *= __grid_tile_sizes[__i];
  }
  for (size_t __i = __rank; __i < _MaxRank; ++__i)
  {
    __grid_tile_sizes[__i] = 1;
  }

  //--------------------------------------------------------------------------------------------------------------------
  // Reordered arrays for loading src and storing dst based on coalesced permutations
  ::cuda::std::array<_StrideTIn, _MaxRank> __src_perm_src_strides{};
  ::cuda::std::array<_StrideTOut, _MaxRank> __dst_perm_dst_strides{};
  ::cuda::std::array<__tile_extent_t, _MaxRank> __tile_src_perm_sizes{};
  ::cuda::std::array<__tile_extent_t, _MaxRank> __tile_dst_perm_sizes{};
  ::cuda::std::array<__tile_extent_t, _MaxRank> __tile_src_perm_smem_strides{};
  ::cuda::std::array<__tile_extent_t, _MaxRank> __tile_dst_perm_smem_strides{};
  ::cuda::std::array<__tile_extent_t, _MaxRank> __canonical_strides{};
  __canonical_strides[0] = 1;
  for (size_t __i = 1; __i < __rank; ++__i)
  {
    __canonical_strides[__i] = __canonical_strides[__i - 1] * __tile_sizes[__i - 1];
  }
  for (size_t __i = 0; __i < __rank; ++__i)
  {
    const auto __p                    = __tiling.__src_perm[__i];
    __tile_src_perm_sizes[__i]        = __tile_sizes[__p];
    __src_perm_src_strides[__i]       = __src.__strides[__p];
    __tile_src_perm_smem_strides[__i] = __canonical_strides[__p];

    const auto __q                    = __tiling.__dst_perm[__i];
    __tile_dst_perm_sizes[__i]        = __tile_sizes[__q];
    __dst_perm_dst_strides[__i]       = __dst.__strides[__q];
    __tile_dst_perm_smem_strides[__i] = __canonical_strides[__q];
  }
  for (size_t __i = __rank; __i < _MaxRank; ++__i)
  {
    __tile_src_perm_sizes[__i] = 1;
    __tile_dst_perm_sizes[__i] = 1;
  }

  //--------------------------------------------------------------------------------------------------------------------
  // Construct coordinate iterators on the host (precomputed fast modulo/division)
  // namely, given a linear index, compute the multi-dimensional coordinates
  const __tensor_coord_iterator<_ExtentT, _MaxRank> __grid_iter{__grid_tile_sizes}; // grid tile index
  const __tensor_coord_iterator<__tile_extent_t, _MaxRank> __tile_perm_iter{__tile_src_perm_sizes}; // src -> shared
                                                                                                    // memory
  const __tensor_coord_iterator<__tile_extent_t, _MaxRank> __tile_dst_perm_iter{__tile_dst_perm_sizes}; // shared memory
                                                                                                        // -> dst

  //--------------------------------------------------------------------------------------------------------------------
  // Launch the kernel
  using __value_type            = ::cuda::std::remove_cv_t<_TpIn>;
  const int __thread_block_size = cudax::__find_thread_block_size(__tile_total_size * sizeof(__value_type));

  const auto __config = ::cuda::make_config(
    ::cuda::block_dims(__thread_block_size),
    ::cuda::grid_dims(__grid_size),
    ::cuda::dynamic_shared_memory<__value_type[]>(__tile_total_size));

  if (__tiling.__use_xor_swizzle)
  {
    using _Functor = ::cuda::experimental::__copy_shared_mem_impl<
      ::cuda::std::integral_constant<bool, true>,
      ::cuda::std::integral_constant<::cuda::std::size_t, _MaxRank>,
      _TpIn,
      _TpOut,
      _SrcAccessor,
      _DstAccessor,
      _ExtentT,
      _StrideTIn,
      _StrideTOut>;

    return ::cuda::experimental::lazy_jit::device(
      __stream,
      __config,
      _Functor{},
      __src.__data,
      __src_accessor,
      __dst.__data,
      __dst_accessor,
      __grid_iter,
      __grid_tile_src_strides,
      __grid_tile_dst_strides,
      __tile_perm_iter,
      __src_perm_src_strides,
      __tile_src_perm_smem_strides,
      __tile_dst_perm_iter,
      __dst_perm_dst_strides,
      __tile_dst_perm_smem_strides,
      __dst.__strides,
      static_cast<int>(__tile_total_size),
      __tile_sizes,
      __dst.__extents,
      __src.__strides);
  }
  else
  {
    using _Functor = ::cuda::experimental::__copy_shared_mem_impl<
      ::cuda::std::integral_constant<bool, false>,
      ::cuda::std::integral_constant<::cuda::std::size_t, _MaxRank>,
      _TpIn,
      _TpOut,
      _SrcAccessor,
      _DstAccessor,
      _ExtentT,
      _StrideTIn,
      _StrideTOut>;

    return ::cuda::experimental::lazy_jit::device(
      __stream,
      __config,
      _Functor{},
      __src.__data,
      __src_accessor,
      __dst.__data,
      __dst_accessor,
      __grid_iter,
      __grid_tile_src_strides,
      __grid_tile_dst_strides,
      __tile_perm_iter,
      __src_perm_src_strides,
      __tile_src_perm_smem_strides,
      __tile_dst_perm_iter,
      __dst_perm_dst_strides,
      __tile_dst_perm_smem_strides,
      __dst.__strides,
      static_cast<int>(__tile_total_size),
      __tile_sizes,
      __dst.__extents,
      __src.__strides);
  }
}

#endif // !_CCCL_COMPILER(NVRTC)
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_COPY_SHARED_MEMORY_H
