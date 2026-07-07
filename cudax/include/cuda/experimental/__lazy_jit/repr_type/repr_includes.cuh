//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAZY_JIT_REPR_TYPE_REPR_INCLUDES_H
#define _CUDAX__LAZY_JIT_REPR_TYPE_REPR_INCLUDES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__hierarchy/hierarchy_level_base.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/launch>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__mdspan/default_accessor.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/array>

#include <cuda/experimental/__lazy_jit/functor_kernel.cuh>
#include <cuda/experimental/__lazy_jit/repr_type/fixed_string.cuh>

#include <string>

#include <cxxabi.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::lazy_jit
{
template <typename _FunctorKernel>
struct repr_includes;

template <typename _Functor, typename _Hierarchy>
struct repr_includes<functor_kernel_impl<_Functor, _Hierarchy>>
{
  static constexpr auto fixed_string =
    (__fixed_string("#include <cuda/experimental/__lazy_jit/functor_kernel.cuh>\n") + _Functor::includes + "\n");
};
} // namespace cuda::experimental::lazy_jit

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__LAZY_JIT_REPR_TYPE_REPR_INCLUDES_H
