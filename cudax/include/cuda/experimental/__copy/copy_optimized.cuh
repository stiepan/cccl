//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_OPTIMIZED_H
#define _CUDAX__COPY_OPTIMIZED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/launch>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__mdspan/default_accessor.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/array>
#include <cuda/std/tuple>

#include <cuda/experimental/__copy/copy_optimized_kernel.cuh>
#include <cuda/experimental/__lazy_jit/dispatch.cuh>
#include <cuda/experimental/__copy/tensor_iterator.cuh>
#include <cuda/experimental/__copy_bytes/types.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Launch a naive element-wise copy kernel for strided tensor data.
//!
//! Each thread copies one element at a time using a grid-stride loop. Coordinates are
//! computed from linear indices via @ref __tensor_coord_iterator.
//!
//! @param[in]  __src          Source raw tensor descriptor
//! @param[out] __dst          Destination raw tensor descriptor
//! @param[in]  __tensor_size  Total number of elements to copy
//! @param[in]  __stream       CUDA stream for asynchronous execution
//! @param[in]  __src_accessor Accessor for reading source elements
//! @param[in]  __dst_accessor Accessor for writing destination elements
template <typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut,
          typename _TpIn,
          typename _TpOut,
          ::cuda::std::size_t _Rank,
          typename _SrcAccessor = ::cuda::std::default_accessor<_TpIn>,
          typename _DstAccessor = ::cuda::std::default_accessor<_TpOut>>
_CCCL_HOST_API ::cuda::experimental::lazy_jit::dispatch_ret_type __copy_optimized(
  const __raw_tensor<_ExtentT, _StrideTIn, _TpIn, _Rank>& __src,
  const __raw_tensor<_ExtentT, _StrideTOut, _TpOut, _Rank>& __dst,
  _ExtentT __tensor_size,
  ::cuda::stream_ref __stream,
  const _SrcAccessor& __src_accessor = {},
  const _DstAccessor& __dst_accessor = {})
{
  constexpr int __block_size = 256;
  const __tensor_coord_iterator<_ExtentT, _Rank> __coord_iter(__src.__extents);
  const auto __grid_size = ::cuda::ceil_div(__tensor_size, _ExtentT{__block_size});
  const auto __config    = ::cuda::make_config(::cuda::block_dims<__block_size>(), ::cuda::grid_dims(__grid_size));

  // __copy_optimized_impl's operator() is templated on _Config (deduced per-call, like
  // cuda::launch's own documented functor-kernel pattern), so the functor's own template argument list
  // is a pure list of types -- no _Config, no non-type parameters (_Rank travels as an
  // integral_constant) -- which is what makes it straightforward to spell out in full for JIT dispatch.
  using _Functor = ::cuda::experimental::__copy_optimized_impl<
    _TpIn,
    _TpOut,
    _SrcAccessor,
    _DstAccessor,
    _ExtentT,
    _StrideTIn,
    _StrideTOut,
    ::cuda::std::integral_constant<::cuda::std::size_t, _Rank>>;

  return ::cuda::experimental::lazy_jit::device(
    __stream,
    __config,
    _Functor{},
    __src.__data,
    __src.__strides,
    __src_accessor,
    __dst.__data,
    __dst.__strides,
    __dst_accessor,
    __coord_iter,
    __tensor_size);
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_OPTIMIZED_H
