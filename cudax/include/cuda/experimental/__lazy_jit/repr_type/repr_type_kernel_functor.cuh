//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAZY_JIT_REPR_TYPE_KERNEL_FUNCTOR_H
#define _CUDAX__LAZY_JIT_REPR_TYPE_KERNEL_FUNCTOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_common.cuh>
#  include <cuda/experimental/__lazy_jit/functor_kernel.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::lazy_jit
{

template <typename _Functor, typename... _Args>
struct functor_args_repr_type;
template <template <typename...> class _Functor, typename... _Args>
struct functor_args_repr_type<_Functor<_Args...>>
{
  static constexpr auto fixed_string = repr_types<_Args...>::fixed_string;
};

template <typename _Functor, typename _Hierarchy>
struct repr_type<functor_kernel_impl<_Functor, _Hierarchy>>
{
  static constexpr auto fixed_string = (
    __fixed_string("::cuda::experimental::lazy_jit::__functor_kernel<")
    + _Functor::name 
    + "<" + functor_args_repr_type<_Functor>::fixed_string + ">, " 
    + repr_type<_Hierarchy>::fixed_string 
    + ">"
  );
};
} // namespace cuda::experimental::lazy_jit

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX__LAZY_JIT_REPR_TYPE_KERNEL_FUNCTOR_H
