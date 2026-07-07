//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAZY_JIT_REPR_TYPE_H
#define _CUDAX__LAZY_JIT_REPR_TYPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

//! @file
//! Entry point for the @c repr_type family: pulls in @c repr_type<_Tp>/@c repr_types<_Ts...> (see
//! @c repr_type_common.cuh) plus every specialization of @c repr_type for the types JIT dispatch needs to
//! spell out (fundamental scalars, accessors, integral_constant, array, hierarchy, tensor_coord_iterator,
//! extents, vector_access). Include *this* header rather than the individual @c repr_type_*.cuh ones directly.
#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_accessor.cuh>
#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_array.cuh>
#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_common.cuh>
#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_extents.cuh>
#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_hierarchy.cuh>
#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_integer.cuh>
#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_integral_constant.cuh>
#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_kernel_functor.cuh>
#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_simple.cuh>
#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_tensor_coord_iterator.cuh>
#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_vector_access.cuh>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX__LAZY_JIT_REPR_TYPE_H
