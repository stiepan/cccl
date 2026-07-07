//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAZY_JIT_REPR_TYPE_INTEGER_H
#define _CUDAX__LAZY_JIT_REPR_TYPE_INTEGER_H

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
#  include <cuda/std/__type_traits/is_signed.h>

#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_common.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::lazy_jit
{
//! @brief Number of decimal digits @p __val needs (at least 1, even for @c 0) -- used to size the
//! @c __fixed_string built by @c repr_integer below. Takes the *magnitude* only; sign (if any) is
//! accounted for separately by @c repr_integer itself.
constexpr ::cuda::std::size_t __decimal_digit_count(unsigned long long __val)
{
  ::cuda::std::size_t __n = 1;
  while (__val >= 10)
  {
    __val /= 10;
    ++__n;
  }
  return __n;
}

//! @brief Spells the non-type template argument @p _Val (of type @p _IntType, which may be any signed or
//! unsigned integral type) as an NVRTC-parsable integer literal, entirely at compile time -- e.g.
//! @c repr_integer<unsigned, 42u>::fixed_string is @c "42ull", @c repr_integer<int, -7>::fixed_string is
//! @c "-7ll". Not a @c repr_type specialization itself (there's no *type* to key off -- @p _Val is a bare
//! value), but the shared building block behind every @c repr_type specialization that needs to spell an
//! integer (@c integral_constant, @c array's extent, etc.).
//!
//! Regardless of @p _IntType's actual width, the spelling is computed through a 64-bit intermediate (so
//! callers never need to pre-widen @p _Val themselves) and always carries an explicit @c ll/@c ull suffix
//! (so the resulting literal's signedness is unambiguous wherever it gets substituted), with the sign (if
//! negative) folded into the digits themselves.
template <typename _IntType, _IntType _Val>
struct repr_integer
{
  //! @brief Whether @p _Val is negative -- only ever @c true for signed @p _IntType, and computed via a
  //! discarded @c if @c constexpr branch (rather than a plain @c _Val @c < @c 0) so an unsigned @p _IntType
  //! never triggers an always-false comparison.
  static constexpr bool __compute_negative()
  {
    if constexpr (::cuda::std::is_signed_v<_IntType>)
    {
      return _Val < 0;
    }
    else
    {
      return false;
    }
  }
  static constexpr bool __negative = __compute_negative();

  //! @brief Absolute value of @p _Val, widened to @c unsigned @c long @c long -- computed via unsigned
  //! wraparound (rather than @c -_Val) so it's well-defined even for @p _IntType's most negative value.
  static constexpr unsigned long long __magnitude =
    __negative ? (0ULL - static_cast<unsigned long long>(_Val)) : static_cast<unsigned long long>(_Val);

  static constexpr auto __make()
  {
    constexpr ::cuda::std::size_t __ndigits = __decimal_digit_count(__magnitude);
    __fixed_string<(__negative ? 1 : 0) + __ndigits + 1> __digits{};
    unsigned long long __v = __magnitude;
    for (::cuda::std::size_t __i = __ndigits; __i-- > 0;)
    {
      __digits.__data[(__negative ? 1 : 0) + __i] = static_cast<char>('0' + (__v % 10));
      __v /= 10;
    }
    if constexpr (__negative)
    {
      __digits.__data[0] = '-';
    }
    if constexpr (::cuda::std::is_signed_v<_IntType>)
    {
      return __digits + "ll";
    }
    else
    {
      return __digits + "ull";
    }
  }
  static constexpr auto fixed_string = __make();
};
} // namespace cuda::experimental::lazy_jit

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX__LAZY_JIT_REPR_TYPE_INTEGER_H
