//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_CONTIGUOUS_KERNEL_H
#define _CUDAX__COPY_CONTIGUOUS_KERNEL_H

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
// #include <cuda/launch>
#include <cuda/std/array>

#include <cuda/experimental/__copy/tensor_iterator.cuh>
#include <cuda/experimental/__copy/vector_access.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Tiled copy kernel for contiguous innermost dimension.
//!
//! Uses a 2D grid: blockIdx.x = tile along inner dimension, blockIdx.y = outer index.
//! Threads within a block stride over the tile, reading from the source and writing to the
//! destination via accessors. The coordinate iterator maps linear indices to multi-dimensional
//! coordinates, which are then used with per-tensor strides for the actual memory access.
//!
//! @param[in]  __config          Kernel launch configuration
//! @param[in]  __src_ptr         Pointer to source data
//! @param[in]  __src_strides     Per-dimension strides for the source tensor
//! @param[in]  __src_accessor    Accessor for reading source elements
//! @param[out] __dst_ptr         Pointer to destination data
//! @param[in]  __dst_strides     Per-dimension strides for the destination tensor
//! @param[in]  __dst_accessor    Accessor for writing destination elements
//! @param[in]  __coord_iter      Coordinate iterator for multi-dimensional index mapping
//! @param[in]  __inner_size      Extent of the contiguous innermost dimension
template <typename _TileSizeConstant,
          typename _TpSrc,
          typename _TpDst,
          typename _SrcAccessor,
          typename _DstAccessor,
          typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut,
          typename _RankConstant>
struct __copy_contiguous_impl
{
  static constexpr ::cuda::std::size_t _Rank = _RankConstant::value;
  static constexpr int _TileSize             = _TileSizeConstant::value;

#if !_CCCL_COMPILER(NVRTC)
  static constexpr char name[]     = "cuda::experimental::__copy_contiguous_impl";
  static constexpr char includes[] = "#include <cuda/experimental/__copy/copy_contiguous_kernel.cuh>";
#endif

  template <typename _Config>
  _CCCL_DEVICE_API void operator()(
    const _Config __config,
    const _TpSrc* const _CCCL_RESTRICT __src_ptr,
    const ::cuda::std::array<_StrideTIn, _Rank> __src_strides,
    const _SrcAccessor __src_accessor,
    _TpDst* const _CCCL_RESTRICT __dst_ptr,
    const ::cuda::std::array<_StrideTOut, _Rank> __dst_strides,
    const _DstAccessor __dst_accessor,
    const __tensor_coord_iterator<_ExtentT, _Rank> __coord_iter,
    const _ExtentT __inner_size) const
  {
    using __partial_tensor_src  = __partial_tensor<const _TpSrc, _StrideTIn, _Rank, _SrcAccessor>;
    using __partial_tensor_dst  = __partial_tensor<_TpDst, _StrideTOut, _Rank, _DstAccessor>;
    const auto __thread_id      = ::cuda::gpu_thread.rank_as<_ExtentT>(::cuda::block, __config);
    const auto __block_idx      = ::cuda::block.index_as<_ExtentT>(::cuda::grid);
    constexpr auto __block_size = ::cuda::gpu_thread.count_as<int>(::cuda::block, __config);
    const __partial_tensor_src __src{__src_ptr, __src_strides, __src_accessor};
    const __partial_tensor_dst __dst{__dst_ptr, __dst_strides, __dst_accessor};

    const auto __tile_offset = __block_idx.x * _TileSize;
    const auto __outer_idx   = __block_idx.y;
    const auto __remaining   = __inner_size - __tile_offset;
    const auto __base_idx    = __outer_idx * __inner_size + __tile_offset + __thread_id;

    if (__remaining >= _TileSize)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int __i = 0; __i < _TileSize; __i += __block_size)
      {
        const auto __coord = __coord_iter(__base_idx + __i);
        __dst(__coord)     = __src(__coord);
      }
    }
    else
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int __i = 0; __i < _TileSize; __i += __block_size)
      {
        if (__thread_id + __i < __remaining)
        {
          const auto __coord = __coord_iter(__base_idx + __i);
          __dst(__coord)     = __src(__coord);
        }
      }
    }
  }
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_CONTIGUOUS_KERNEL_H
