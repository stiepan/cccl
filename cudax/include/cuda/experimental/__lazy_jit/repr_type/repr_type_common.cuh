//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAZY_JIT_REPR_TYPE_COMMON_H
#define _CUDAX__LAZY_JIT_REPR_TYPE_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// Host-only: produces C++ type spellings (as compile-time __fixed_string's) for the JIT code string.
#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/experimental/__lazy_jit/repr_type/fixed_string.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::lazy_jit
{
//! @brief Maps a C++ type to its NVRTC-parsable spelling, as @c repr_type<_Tp>::fixed_string.
//!
//! Every supported @c _Tp is added via a specialization in one of the sibling @c repr_type_*.cuh headers
//! (see @ref repr_type.cuh, which pulls all of them in). By default, any other, unsupported type (e.g. a
//! user-defined accessor) leaves this primary template incomplete, so trying to use
//! @c repr_type<UnsupportedType>::fixed_string fails to compile right at the describe site, rather than
//! silently producing bogus JIT source text.
//!
//! Defining @c REPR_TYPE_ALLOW_UNKNOWN before including this header instead gives the primary template a
//! catch-all body that spells any unsupported @c _Tp as the literal text @c "unknown" -- useful e.g. while
//! exploring which types a given describe site actually needs, without having every one of them be a hard
//! compile error up front.
//!
//! Note: a generic "any trivially-copyable, alignment-equals-size type falls back to
//! @c __vector_access<sizeof(_Tp)>'s spelling" specialization was deliberately *not* added here: SFINAE'd
//! partial specializations like that one do not reliably coexist with the *other*, template-pattern-based
//! partial specializations already in this family (e.g. @c repr_type<functor_kernel_impl<_Functor,
//! _Hierarchy>> in repr_type_kernel_functor.cuh) whenever the pattern-based one happens to also match an
//! empty (`sizeof==alignof==1`) type -- GCC reports a genuine "ambiguous template instantiation" in that
//! case, not just a style nit. Callers that want a dtype-agnostic element type should instead use
//! @c cuda::experimental::__vector_access<N> directly as their mdspan's element type (see
//! repr_type_vector_access.cuh, which already spells it) rather than relying on a fallback here.
#  ifdef REPR_TYPE_ALLOW_UNKNOWN
template <typename _Tp>
struct repr_type
{
  static constexpr auto fixed_string = __fixed_string("unknown");
};
#  else
template <typename _Tp>
struct repr_type;
#  endif

//! @brief Spells a list of types as a bare, comma-separated list (e.g. @c "T0, T1, T2"), rather than
//! wrapped in @c "tuple<...>" or the like -- suitable for building an NVRTC name-expression that
//! explicitly instantiates a multi-parameter template.
template <typename... _Ts>
struct repr_types
{
  static constexpr auto fixed_string = __fixed_string("");
};

template <typename _T0, typename... _Rest>
struct repr_types<_T0, _Rest...>
{
  static constexpr auto fixed_string =
    repr_type<_T0>::fixed_string + ((__fixed_string(", ") + repr_type<_Rest>::fixed_string) + ... + __fixed_string(""));
};

//! @brief const-qualified types reuse the unqualified spelling with a @c const prefix.
template <typename _Tp>
struct repr_type<const _Tp>
{
  static constexpr auto fixed_string = __fixed_string("const ") + repr_type<_Tp>::fixed_string;
};

template <typename _Tp>
struct repr_type<_Tp*>
{
  static constexpr auto fixed_string = repr_type<_Tp>::fixed_string + "*";
};
} // namespace cuda::experimental::lazy_jit

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX__LAZY_JIT_REPR_TYPE_COMMON_H
