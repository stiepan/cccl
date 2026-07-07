//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAZY_JIT_REPR_TYPE_VECTOR_ACCESS_H
#define _CUDAX__LAZY_JIT_REPR_TYPE_VECTOR_ACCESS_H

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

#  include <cuda/experimental/__copy/vector_access.cuh>
#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_common.cuh>
#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_integer.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::lazy_jit
{
template <::cuda::std::size_t _VectorBytes>
struct repr_type<::cuda::experimental::__vector_access<_VectorBytes>>
{
  static constexpr auto fixed_string = __fixed_string("::cuda::experimental::__vector_access<")
                                      + repr_integer<::cuda::std::size_t, _VectorBytes>::fixed_string + ">";
};
} // namespace cuda::experimental::lazy_jit

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX__LAZY_JIT_REPR_TYPE_VECTOR_ACCESS_H
