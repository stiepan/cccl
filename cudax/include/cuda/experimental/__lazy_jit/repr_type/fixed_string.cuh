//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAZY_JIT_REPR_TYPE_FIXED_STRING_H
#define _CUDAX__LAZY_JIT_REPR_TYPE_FIXED_STRING_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// Host-only: a compile-time string builder used while assembling the JIT code string (see
// stringify_launch.cuh) -- there's no reason for NVRTC to ever need this itself.
#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__cstddef/types.h>

#  include <string_view>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::lazy_jit
{
//! @brief A fixed-capacity, compile-time string with @b value (not pointer/view) semantics.
//!
//! Unlike @c std::string_view, which merely *views* someone else's storage, @c __fixed_string *owns* its
//! characters directly, as a plain fixed-size array data member. That's what makes it safe to build (and
//! concatenate, via @c operator+ below) inside a @c constexpr evaluation and then keep around in a
//! @c static @c constexpr variable: its buffer *is* that variable's storage, not a pointer into some
//! temporary that's gone by the time the variable is used. (That's the trap plain @c std::string falls
//! into: even a @c constexpr @c std::string's heap buffer doesn't outlive the expression that created it,
//! so a @c string_view can't safely alias it afterwards -- see @c stringify_includes_impl's docs for the
//! concrete failure.)
//!
//! @note This safety only holds once a @c __fixed_string itself is bound to a named @c static @c
//! constexpr variable -- @b that variable is what has static storage duration, not any @c __fixed_string
//! temporary that happens to flow through an expression before it. So, prefer:
//! @code
//! static constexpr auto __combined = __fixed_string("a") + "b"; // __combined has static storage duration
//! static constexpr ::std::string_view __view = __combined;      // safe: aliases __combined, not a temporary
//! @endcode
//! over converting straight from the @c operator+ result to a @c string_view in one step (@c static
//! @c constexpr ::std::string_view __view = __fixed_string("a") + "b";), which dangles exactly like the
//! @c std::string case above: the concatenated @c __fixed_string is itself just a temporary there, gone
//! by the time @c __view is used.
//!
//! @tparam _Np Size of the backing array, @b including the trailing NUL -- i.e. the same @c N a string
//! literal @c "..." of length @c N-1 would have as @c const @c char(&)[N]. Normally never spelled out
//! explicitly: deduced automatically when constructing from a string literal, via the deduction guide
//! below.
template <::cuda::std::size_t _Np>
struct __fixed_string
{
  char __data[_Np]{};

  constexpr __fixed_string() = default;

  //! @brief Implicit on purpose: lets a plain string literal be passed wherever a @c __fixed_string is
  //! expected (e.g. as an operand to @c operator+ below), with @c _Np deduced from the literal itself.
  constexpr __fixed_string(const char (&__str)[_Np])
  {
    for (::cuda::std::size_t __i = 0; __i < _Np; ++__i)
    {
      __data[__i] = __str[__i];
    }
  }

  //! @brief Length, excluding the trailing NUL (mirrors @c std::string_view::size()).
  static constexpr ::cuda::std::size_t size() noexcept
  {
    return _Np - 1;
  }

  constexpr const char* c_str() const noexcept
  {
    return __data;
  }

  constexpr operator ::std::string_view() const noexcept
  {
    return ::std::string_view(__data, size());
  }
};

//! @brief Deduces @c _Np from a string literal, so e.g. @c __fixed_string("foo") works without ever
//! having to spell out the array size.
template <::cuda::std::size_t _Np>
__fixed_string(const char (&)[_Np]) -> __fixed_string<_Np>;

//! @brief Concatenates two @c __fixed_string's into a new one, @c _Np1 + _Np2 - 1 characters long (the
//! two operands' NUL terminators collapse into the single one at the end of the result). Thanks to
//! @c __fixed_string's converting constructor above, either operand may instead be passed as a raw
//! string literal, e.g. @c __fixed_string("a") + "b" or @c "a" + __fixed_string("b").
//!
//! Chains left-to-right, so any number of pieces can be joined with repeated @c operator+, e.g.
//! @code
//! constexpr auto __msg = __fixed_string("a") + "b" + "c"; // __fixed_string<4>{"abc"}
//! @endcode
template <::cuda::std::size_t _Np1, ::cuda::std::size_t _Np2>
constexpr __fixed_string<_Np1 + _Np2 - 1>
operator+(const __fixed_string<_Np1>& __lhs, const __fixed_string<_Np2>& __rhs)
{
  __fixed_string<_Np1 + _Np2 - 1> __result{};
  for (::cuda::std::size_t __i = 0; __i < _Np1 - 1; ++__i)
  {
    __result.__data[__i] = __lhs.__data[__i];
  }
  for (::cuda::std::size_t __i = 0; __i < _Np2; ++__i) // also copies __rhs's own trailing NUL
  {
    __result.__data[_Np1 - 1 + __i] = __rhs.__data[__i];
  }
  return __result;
}

//! @brief These two overloads (raw literal on the right/left, respectively) exist because template
//! argument deduction never considers converting constructors: deducing the primary @c operator+
//! overload's @c _Np2 (or @c _Np1) against a raw @c const @c char(&)[N] argument fails outright, rather
//! than implicitly converting it to @c __fixed_string<N> first and deducing against @e that.
template <::cuda::std::size_t _Np1, ::cuda::std::size_t _Np2>
constexpr __fixed_string<_Np1 + _Np2 - 1> operator+(const __fixed_string<_Np1>& __lhs, const char (&__rhs)[_Np2])
{
  return __lhs + __fixed_string<_Np2>(__rhs);
}

template <::cuda::std::size_t _Np1, ::cuda::std::size_t _Np2>
constexpr __fixed_string<_Np1 + _Np2 - 1> operator+(const char (&__lhs)[_Np1], const __fixed_string<_Np2>& __rhs)
{
  return __fixed_string<_Np1>(__lhs) + __rhs;
}
} // namespace cuda::experimental::lazy_jit

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDAX__LAZY_JIT_REPR_TYPE_FIXED_STRING_H
