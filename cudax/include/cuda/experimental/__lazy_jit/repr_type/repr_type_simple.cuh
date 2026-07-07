//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAZY_JIT_REPR_TYPE_SIMPLE_H
#define _CUDAX__LAZY_JIT_REPR_TYPE_SIMPLE_H

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

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::lazy_jit
{
//! @brief Fundamental scalar types spell as themselves.
#  define _CCCLX_JIT_NAME(_type, _spelling)                            \
    template <>                                                        \
    struct repr_type<_type>                                            \
    {                                                                  \
      static constexpr auto fixed_string = __fixed_string(_spelling); \
    }

_CCCLX_JIT_NAME(bool, "bool");
_CCCLX_JIT_NAME(char, "char");
_CCCLX_JIT_NAME(signed char, "signed char");
_CCCLX_JIT_NAME(unsigned char, "unsigned char");
_CCCLX_JIT_NAME(short, "short");
_CCCLX_JIT_NAME(unsigned short, "unsigned short");
_CCCLX_JIT_NAME(int, "int");
_CCCLX_JIT_NAME(unsigned int, "unsigned int");
_CCCLX_JIT_NAME(long, "long");
_CCCLX_JIT_NAME(unsigned long, "unsigned long");
_CCCLX_JIT_NAME(long long, "long long");
_CCCLX_JIT_NAME(unsigned long long, "unsigned long long");
_CCCLX_JIT_NAME(float, "float");
_CCCLX_JIT_NAME(double, "double");

#  undef _CCCLX_JIT_NAME
} // namespace cuda::experimental::lazy_jit

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX__LAZY_JIT_REPR_TYPE_SIMPLE_H
