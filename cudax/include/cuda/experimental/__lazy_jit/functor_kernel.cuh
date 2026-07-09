//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAZY_JIT_FUNCTOR_KERNEL_H
#define _CUDAX__LAZY_JIT_FUNCTOR_KERNEL_H

#include <cuda/std/detail/__config>
#include <cuda/std/tuple>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::lazy_jit
{
template <typename T>
struct args_of;

//! @brief Pointer-to-member-function specializations, extracting the parameter list as a
//! @c cuda::std::tuple<Args...>. @c operator() is conventionally declared @c const (see
//! @ref __copy_optimized_impl), hence the @c const-qualified specialization -- without it, @c args_of
//! for a const member function falls through to the (incomplete) primary template.
template <typename C, typename R, typename... Args>
struct args_of<R (C::*)(Args...)>
{
  using type = ::cuda::std::tuple<Args...>;
};

template <typename C, typename R, typename... Args>
struct args_of<R (C::*)(Args...) const>
{
  using type = ::cuda::std::tuple<Args...>;
};

//! @brief The full tuple of argument types @c T::operator()<Args...> expects -- @c _Config included, as
//! its first element -- derived straight from @c T's own (member template) @c operator(), with no
//! separate, hand-maintained type list needed anywhere.
template <typename T, typename... Args>
using operator_args_t = typename args_of<decltype(&T::template operator()<Args...>)>::type;


//! @brief Plain (non-lambda) callable that forwards its arguments straight to a default-constructed
//! @c _Functor -- used as the callable @c cuda::std::apply invokes in @ref functor_kernel_impl below.
//! An ordinary struct with an explicitly @c __device__-annotated @c operator() is used here, rather than
//! a lambda, so that its execution space is spelled out explicitly instead of relying on NVRTC's
//! lambda-execution-space inference (which otherwise requires passing @c -default-device to NVRTC).
template <typename _Functor>
struct __unpacked_caller
{
  template <typename... Args>
  __device__ void operator()(Args&&... __args) const
  {
    _Functor{}(__args...);
  }
};

template <typename _Functor, typename _Hierarchy>
struct functor_kernel_impl
{
  using args_t = operator_args_t<_Functor, _Hierarchy>;

  __device__ void operator()(const args_t __args) const
  {
    ::cuda::std::apply(__unpacked_caller<_Functor>{}, __args);
  }
};

template <typename _Functor, typename _Hierarchy>
__global__ void __functor_kernel(const _CCCL_GRID_CONSTANT operator_args_t<_Functor, _Hierarchy> __args)
{
  functor_kernel_impl<_Functor, _Hierarchy>{}(__args);
}
} // namespace cuda::experimental::lazy_jit

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__LAZY_JIT_FUNCTOR_KERNEL_H
