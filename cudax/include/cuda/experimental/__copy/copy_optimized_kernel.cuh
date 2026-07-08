//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_OPTIMIZED_KERNEL_H
#define _CUDAX__COPY_OPTIMIZED_KERNEL_H

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
#include <cuda/__hierarchy/hierarchy_levels.h>
#include <cuda/__hierarchy/hierarchy_dimensions.h>
#include <cuda/__mdspan/host_device_mdspan.h>
#include <cuda/__mdspan/traits.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__mdspan/default_accessor.h>
// #include <cuda/launch>
#include <cuda/std/array>

#include <cuda/experimental/__copy/tensor_iterator.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

//! @brief Element-wise copy kernel functor for strided tensor data.
//!
//! Each thread copies one element at a time using a grid-stride loop, mapping linear indices to
//! multi-dimensional coordinates via @ref __tensor_coord_iterator.
//!
//! Deliberately shaped like @c cuda::launch's own documented functor-kernel pattern (see the
//! @c cuda::launch overload taking a functor in `<cuda/__launch/launch.h>`): @c operator() is a member
//! *template*, deduced on @c _Config, rather than @c _Config being one of this class's own template
//! parameters. This keeps @c __copy_optimized_impl's own template argument list a pure list of types
//! (no @c _Config, no non-type parameters -- @c _Rank travels as an @c integral_constant), which is what
//! makes it straightforward to name in full for JIT dispatch (see @ref __jit_type_name_impl below and
//! @c lazy_jit::__functor_kernel), and sidesteps having to keep this functor's notion of
//! @c _Config in exact sync with whatever @c cuda::launch's non-JIT functor overload actually ends up
//! invoking it with (it calls @c combine_with_default on the configuration first).
//!
//! @tparam _TpSrc         Source element type
//! @tparam _TpDst         Destination element type
//! @tparam _SrcAccessor   Accessor for reading source elements
//! @tparam _DstAccessor   Accessor for writing destination elements
//! @tparam _ExtentT       Index/extent type
//! @tparam _StrideTIn     Source stride type
//! @tparam _StrideTOut    Destination stride type
//! @tparam _RankConstant  @c cuda::std::integral_constant<size_t, Rank> wrapping the tensor rank
template <typename _TpSrc,
          typename _TpDst,
          typename _SrcAccessor,
          typename _DstAccessor,
          typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut,
          typename _RankConstant>
struct __copy_optimized_impl
{
  static constexpr ::cuda::std::size_t _Rank = _RankConstant::value;
  #if !_CCCL_COMPILER(NVRTC)
  static constexpr char name[] = "cuda::experimental::__copy_optimized_impl";
  static constexpr char includes[] = "#include <cuda/experimental/__copy/copy_optimized_kernel.cuh>";
  #endif

  //! @param[in]  __config        Kernel launch configuration
  //! @param[in]  __src_ptr       Pointer to source data
  //! @param[in]  __src_strides   Per-dimension strides for the source tensor
  //! @param[in]  __src_accessor  Accessor for reading source elements
  //! @param[out] __dst_ptr       Pointer to destination data
  //! @param[in]  __dst_strides   Per-dimension strides for the destination tensor
  //! @param[in]  __dst_accessor  Accessor for writing destination elements
  //! @param[in]  __coord_iter    Coordinate iterator for multi-dimensional index mapping
  //! @param[in]  __tensor_size   Total number of elements to copy
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
    const _ExtentT __tensor_size) const
  {
    // Under LAZY_JIT_DISPATCH, the host build only needs this operator()'s *declaration* (its
    // parameter-type list, e.g. for operator_args_t) to build a KernelDesc -- it never actually
    // calls it. The real body below uses device-only APIs (rank_as/count_as, built on
    // threadIdx/blockIdx) that only a genuine CUDA compiler can parse, so it's stubbed out here and
    // only compiled for real when NVRTC compiles the extracted `code` string (which does not define
    // LAZY_JIT_DISPATCH).
#ifndef LAZY_JIT_DISPATCH
    using __partial_tensor_src = __partial_tensor<const _TpSrc, _StrideTIn, _Rank, _SrcAccessor>;
    using __partial_tensor_dst = __partial_tensor<_TpDst, _StrideTOut, _Rank, _DstAccessor>;
    const auto __idx           = ::cuda::gpu_thread.rank_as<_ExtentT>(::cuda::grid, __config);
    const auto __stride        = ::cuda::gpu_thread.count_as<_ExtentT>(::cuda::grid, __config);
    const __partial_tensor_src __src{__src_ptr, __src_strides, __src_accessor};
    const __partial_tensor_dst __dst{__dst_ptr, __dst_strides, __dst_accessor};

    for (auto __i = __idx; __i < __tensor_size; __i += __stride)
    {
      const auto __coord = __coord_iter(__i);
      __dst(__coord)     = __src(__coord);
      if constexpr (sizeof(_ExtentT) <= 4)
      {
        return;
      }
    }
#endif // !LAZY_JIT_DISPATCH
  }
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_OPTIMIZED_KERNEL_H
