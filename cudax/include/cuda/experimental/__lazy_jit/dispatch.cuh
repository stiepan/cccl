//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAZY_JIT_DISPATCH_H
#define _CUDAX__LAZY_JIT_DISPATCH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#ifdef LAZY_JIT_DISPATCH
#  include <cuda/experimental/__lazy_jit/lazy_launch.cuh>
#endif // LAZY_JIT_DISPATCH

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
namespace lazy_jit
{
#ifndef LAZY_JIT_DISPATCH

using dispatch_ret_type = void;

template <typename Fn>
auto host(Fn&& fn)
{
  return fn();
}

inline void host() { return; }

template <typename... _Args>
auto device(_Args&&... __args)
{
  return ::cuda::launch(__args...);
}

#else

using dispatch_ret_type = KernelDesc;

template <typename Fn>
auto host(Fn&& fn)
{
  return KernelDesc{};
}

inline auto host() { return KernelDesc{}; }

template <typename... _Args>
auto device(_Args&&... __args)
{
  return make_kernel_desc(__args...);
}

#endif // LAZY_JIT_DISPATCH
} // namespace lazy_jit
} // namespace cuda::experimental
#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__LAZY_JIT_DISPATCH_H
