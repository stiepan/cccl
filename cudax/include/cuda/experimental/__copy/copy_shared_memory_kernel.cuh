//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_SHARED_MEMORY_KERNEL_H
#define _CUDAX__COPY_SHARED_MEMORY_KERNEL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// Kept intentionally minimal (and NVRTC-friendly): only what the __global__ kernel body itself needs.
// _Config is accepted purely as an opaque template parameter here; making it (and its transitive headers,
// e.g. <cuda/__launch/configuration.h>) NVRTC-friendly is out of scope for now.
#include <cuda/__hierarchy/hierarchy_dimensions.h>
#include <cuda/__hierarchy/hierarchy_levels.h>
#include <cuda/__mdspan/host_device_mdspan.h>
#include <cuda/__mdspan/traits.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__mdspan/default_accessor.h>
#include <cuda/std/array>

#include <cuda/experimental/__copy/tensor_iterator.cuh>
#include <cuda/experimental/__copy/copy_shared_memory_utils.cuh>
// Not used by this kernel's own logic -- pulled in because a template argument (_TpSrc/_TpDst) may be
// ::cuda::experimental::__vector_access<N> itself, via the generic opaque-blob repr_type fallback (see
// repr_type_opaque_blob.cuh), which needs this type visible wherever the kernel gets NVRTC-instantiated.
#include <cuda/experimental/__copy/vector_access.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Compute the shared-memory offset for the XOR swizzle.
//!
//! @param[in] __offset The offset in the shared-memory tile.
//! @return The offset in the shared-memory tile with the XOR swizzle applied.
template <bool _UseXorSwizzle>
[[nodiscard]] _CCCL_DEVICE_API __tile_extent_t __smem_offset(__tile_extent_t __offset) noexcept
{
  if constexpr (_UseXorSwizzle)
  {
    static_assert(__max_tile_size == 32, "XOR shared-memory swizzle assumes 32 banks and 32-element tile modes");
    constexpr __tile_extent_t __swizzle_tile_size = __max_tile_size * __max_tile_size;
    const auto __outer                            = __offset / __swizzle_tile_size;
    const auto __offset_tile_rounded              = __outer * __swizzle_tile_size;
    const auto __inner                            = __offset - __offset_tile_rounded;
    const auto __row                              = __inner / __max_tile_size;
    const auto __row_tile_rounded                 = __row * __max_tile_size;
    const auto __col                              = __inner - __row_tile_rounded;
    return __offset_tile_rounded + __row_tile_rounded + (__col ^ __row);
  }
  return __offset;
}

//! @brief Shared-memory tiled transpose kernel for arbitrary-rank tensors.
//!
//! Each block processes one tile. Threads cooperatively iterate over tile elements with a stride loop. Full (interior)
//! tiles use a two-phase shared-memory transpose: load source data into shared memory using source-coalesced ordering,
//! then store from shared memory to destination using destination-coalesced ordering. Partial (boundary) tiles copy
//! elements directly without shared memory.
//!
//! @param[in]  __config                 Kernel launch configuration
//! @param[in]  __src_ptr                Pointer to source data
//! @param[in]  __src_accessor           Accessor for reading source elements
//! @param[out] __dst_ptr                Pointer to destination data
//! @param[in]  __dst_accessor           Accessor for writing destination elements
//! @param[in]  __grid_iter              Coordinate iterator for grid tile decomposition
//! @param[in]  __grid_tile_src_strides  Per-dimension source strides scaled by tile sizes
//! @param[in]  __grid_tile_dst_strides  Per-dimension destination strides scaled by tile sizes
//! @param[in]  __tile_perm_iter         Coordinate iterator for src-permuted tile decomposition
//! @param[in]  __src_perm_src_strides   Src-permuted source strides for loading
//! @param[in]  __tile_src_perm_smem_strides Src-permuted shared memory strides for loading
//! @param[in]  __tile_dst_perm_iter     Coordinate iterator for dst-permuted tile decomposition
//! @param[in]  __dst_perm_dst_strides   Dst-permuted destination strides for storing
//! @param[in]  __tile_dst_smem_strides  Dst-permuted shared memory strides for storing
//! @param[in]  __dst_strides            Per-dimension destination strides for partial tiles
//! @param[in]  __tile_total_size        Total number of elements in one tile
//! @param[in]  __tile_sizes             Per-dimension tile extents
//! @param[in]  __extents                Per-dimension tensor extents (for partial-tile bounds)
//! @param[in]  __src_strides            Per-dimension source strides (for partial-tile access)
template <typename _UseXorSwizzleConstant,
          typename _MaxRankUZConstant,
          typename _TpSrc,
          typename _TpDst,
          typename _SrcAccessor,
          typename _DstAccessor,
          typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut>
struct __copy_shared_mem_impl
{
  static constexpr ::cuda::std::size_t _MaxRankUZ = _MaxRankUZConstant::value;
  static constexpr bool _UseXorSwizzle       = _UseXorSwizzleConstant::value;

#if !_CCCL_COMPILER(NVRTC)
  static constexpr char name[]     = "cuda::experimental::__copy_shared_mem_impl";
  static constexpr char includes[] = "#include <cuda/experimental/__copy/copy_shared_memory_kernel.cuh>";
#endif

  template <typename _Config>
  _CCCL_DEVICE_API void operator()(
    const _Config __config,
    const _TpSrc* _CCCL_RESTRICT __src_ptr,
    const _SrcAccessor __src_accessor,
    _TpDst* _CCCL_RESTRICT __dst_ptr,
    const _DstAccessor __dst_accessor,
    const __tensor_coord_iterator<_ExtentT, _MaxRankUZ> __grid_iter,
    const ::cuda::std::array<_StrideTIn, _MaxRankUZ> __grid_tile_src_strides,
    const ::cuda::std::array<_StrideTOut, _MaxRankUZ> __grid_tile_dst_strides,
    const __tensor_coord_iterator<__tile_extent_t, _MaxRankUZ> __tile_perm_iter,
    const ::cuda::std::array<_StrideTIn, _MaxRankUZ> __src_perm_src_strides,
    const ::cuda::std::array<__tile_extent_t, _MaxRankUZ> __tile_src_perm_smem_strides,
    const __tensor_coord_iterator<__tile_extent_t, _MaxRankUZ> __tile_dst_perm_iter,
    const ::cuda::std::array<_StrideTOut, _MaxRankUZ> __dst_perm_dst_strides,
    const ::cuda::std::array<__tile_extent_t, _MaxRankUZ> __tile_dst_perm_smem_strides,
    const ::cuda::std::array<_StrideTOut, _MaxRankUZ> __dst_strides,
    const int __tile_total_size,
    const ::cuda::std::array<__tile_extent_t, _MaxRankUZ> __tile_sizes,
    const ::cuda::std::array<_ExtentT, _MaxRankUZ> __extents,
    const ::cuda::std::array<_StrideTIn, _MaxRankUZ> __src_strides)
  {
    // See the analogous comment in __copy_optimized_impl::operator() (copy_optimized_kernel.cuh):
    // stubbed out under LAZY_JIT_DISPATCH so a plain host build never has to parse the device-only
    // APIs below; the real body is only compiled by NVRTC (which doesn't define LAZY_JIT_DISPATCH).
#ifndef LAZY_JIT_DISPATCH
    constexpr auto __max_rank = int{_MaxRankUZ};
    // Grid tile decomposition: map linearized block index to src/dst base offsets
    // __grid_coords: linear tile index -> multi-dimensional coordinates (array)
    const auto __grid_index  = ::cuda::block.index_as<_ExtentT>(::cuda::grid).x;
    const auto __grid_coords = __grid_iter(__grid_index);

    {
      _StrideTIn __src_base  = 0;
      _StrideTOut __dst_base = 0;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int __k = 0; __k < __max_rank; ++__k)
      {
        __src_base += static_cast<_StrideTIn>(__grid_coords[__k]) * __grid_tile_src_strides[__k];
        __dst_base += static_cast<_StrideTOut>(__grid_coords[__k]) * __grid_tile_dst_strides[__k];
      }
      __src_ptr += __src_base;
      __dst_ptr += __dst_base;
    }

    // Partial tile detection: is the current tile full or partial?
    bool __is_full_tile = true;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __k = 0; __k < __max_rank; ++__k)
    {
      const auto __block_start = __grid_coords[__k] * __tile_sizes[__k];
      if (__block_start + __tile_sizes[__k] > __extents[__k])
      {
        __is_full_tile = false;
        break;
      }
    }

    // Dispatch to Full-tile or Boundary case
    const auto __tid           = ::cuda::gpu_thread.rank_as<int>(::cuda::block, __config);
    const auto __block_stride  = ::cuda::gpu_thread.count_as<int>(::cuda::block, __config);
    using __partial_tensor_src = __partial_tensor<const _TpSrc, _StrideTIn, _MaxRankUZ, _SrcAccessor>;
    using __partial_tensor_dst = __partial_tensor<_TpDst, _StrideTOut, _MaxRankUZ, _DstAccessor>;

    //--------------------------------------------------------------------------------------------------------------------
    // Full-tile shared-memory transpose
    if (__is_full_tile)
    {
      using _Tp = ::cuda::std::remove_cv_t<_TpSrc>;
      using __partial_tensor_smem =
        __partial_tensor<_Tp, __tile_extent_t, _MaxRankUZ, ::cuda::std::default_accessor<_Tp>>;

      extern __shared__ char __smem_bytes[];
      auto* __smem = reinterpret_cast<_Tp*>(__smem_bytes);

      // (1) load src to shared memory by using the src/tile-permuted ordering
      const __partial_tensor_src __src_tensor{__src_ptr, __src_perm_src_strides, __src_accessor};
      const __partial_tensor_smem __smem_tensor{
        __smem, __tile_src_perm_smem_strides, ::cuda::std::default_accessor<_Tp>{}};

      for (auto __i = __tid; __i < __tile_total_size; __i += __block_stride)
      {
        const auto __coords          = __tile_perm_iter(__i);
        const auto __raw_offset      = __smem_tensor.__offset(__coords);
        const auto __swizzled_offset = ::cuda::experimental::__smem_offset<_UseXorSwizzle>(__raw_offset);
        __smem[__swizzled_offset]    = __src_tensor(__coords);
      }
      __syncthreads();

      // (2) store from shared memory to destination by using the dst/tile-permuted ordering
      const __partial_tensor_dst __dst_tensor{__dst_ptr, __dst_perm_dst_strides, __dst_accessor};
      const __partial_tensor_smem __smem_dst_tensor{
        __smem, __tile_dst_perm_smem_strides, ::cuda::std::default_accessor<_Tp>{}};

      for (auto __i = __tid; __i < __tile_total_size; __i += __block_stride)
      {
        const auto __coords          = __tile_dst_perm_iter(__i);
        const auto __raw_offset      = __smem_dst_tensor.__offset(__coords);
        const auto __swizzled_offset = ::cuda::experimental::__smem_offset<_UseXorSwizzle>(__raw_offset);
        __dst_tensor(__coords)       = __smem[__swizzled_offset];
      }
    }

    //--------------------------------------------------------------------------------------------------------------------
    // Boundary direct-copy (no shared memory)
    else
    {
      using __uextent_t = ::cuda::std::make_unsigned_t<_ExtentT>;
      const __partial_tensor_src __src_tensor{__src_ptr, __src_strides, __src_accessor};
      const __partial_tensor_dst __dst_tensor{__dst_ptr, __dst_strides, __dst_accessor};

      // Find the partial tile sizes and total number of elements
      ::cuda::std::array<__tile_extent_t, __max_rank> __partial_tile_sizes{};
      int __partial_tile_total = 1;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int __k = 0; __k < __max_rank; ++__k)
      {
        const auto __block_start  = static_cast<__uextent_t>(__grid_coords[__k] * __tile_sizes[__k]);
        const auto __diff         = static_cast<__tile_extent_t>(__extents[__k] - __block_start);
        __partial_tile_sizes[__k] = ::cuda::std::min(__tile_sizes[__k], __diff);
        __partial_tile_total *= __partial_tile_sizes[__k];
      }

      // map the linear index to the multi-dimensional coordinates and copy the elements
      for (auto __i = __tid; __i < __partial_tile_total; __i += __block_stride)
      {
        __tile_extent_t __linear = __i;
        ::cuda::std::array<__tile_extent_t, __max_rank> __coords;
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int __k = 0; __k < __max_rank; ++__k)
        {
          __coords[__k] = __linear % __partial_tile_sizes[__k];
          __linear /= __partial_tile_sizes[__k];
        }
        __dst_tensor(__coords) = __src_tensor(__coords);
      }
    }
#endif // !LAZY_JIT_DISPATCH
  }
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_SHARED_MEMORY_KERNEL_H
