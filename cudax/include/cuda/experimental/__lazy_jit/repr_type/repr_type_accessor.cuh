//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAZY_JIT_REPR_TYPE_ACCESSOR_H
#define _CUDAX__LAZY_JIT_REPR_TYPE_ACCESSOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__mdspan/host_device_accessor.h>
#  include <cuda/std/__mdspan/default_accessor.h>

#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_common.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::lazy_jit
{
//! @brief default_accessor<T> composes from the element spelling.
template <typename _Tp>
struct repr_type<::cuda::std::default_accessor<_Tp>>
{
  static constexpr auto fixed_string = __fixed_string("::cuda::std::default_accessor<") + repr_type<_Tp>::fixed_string + ">";
};

//! @brief device_mdspan wraps its accessor in cuda::device_accessor<A>.
template <typename _Accessor>
struct repr_type<::cuda::__device_accessor<_Accessor>>
{
  static constexpr auto fixed_string = __fixed_string("::cuda::device_accessor<") + repr_type<_Accessor>::fixed_string + ">";
};
} // namespace cuda::experimental::lazy_jit

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX__LAZY_JIT_REPR_TYPE_ACCESSOR_H
