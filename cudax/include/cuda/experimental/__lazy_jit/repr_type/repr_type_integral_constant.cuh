//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAZY_JIT_REPR_TYPE_INTEGRAL_CONSTANT_H
#define _CUDAX__LAZY_JIT_REPR_TYPE_INTEGRAL_CONSTANT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__type_traits/integral_constant.h>

#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_common.cuh>
#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_integer.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::lazy_jit
{
//! @brief Spells a @c cuda::std::integral_constant as a full type -- e.g. functor-style kernels (see
//! @c __copy_optimized_impl) wrap a non-type template argument (like a @c size_t rank) in an
//! @c integral_constant so it can travel as a genuine *type* template argument, keeping the functor's own
//! template argument list a pure list of types.
template <typename _Tp, _Tp _Val>
struct repr_type<::cuda::std::integral_constant<_Tp, _Val>>
{
  static constexpr auto fixed_string = __fixed_string("::cuda::std::integral_constant<") + repr_type<_Tp>::fixed_string
                                      + ", " + repr_integer<_Tp, _Val>::fixed_string + ">";
};
} // namespace cuda::experimental::lazy_jit

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX__LAZY_JIT_REPR_TYPE_INTEGRAL_CONSTANT_H
