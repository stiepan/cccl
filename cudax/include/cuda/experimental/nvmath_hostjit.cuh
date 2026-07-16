//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_NVMATH_HOSTJIT_CUH
#define _CUDA_EXPERIMENTAL_NVMATH_HOSTJIT_CUH

#include <cuda/std/detail/__config>

#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__exception/terminate.h>

#include <nv/target>

#if !_CCCL_HOSTED()

//! Freestanding stubs for the standard exception hierarchy that CCCL's copy path
//! throws. hostjit compiles CCCL freestanding, so `<stdexcept>`/`<exception>` are
//! absent and these types would otherwise be undefined. The stubs copy their
//! message into a fixed inline buffer (no heap, no host stdlib), so a thrown
//! object owns its text for the lifetime CCCL needs (through unwinding to the
//! embedder's catch). Only defined for freestanding (`!_CCCL_HOSTED()`); a hosted
//! sanity build uses the real standard library.
//!
//! Defining these in namespace `std` is technically reserved, but is the
//! established pattern for a freestanding/embedded C++ environment and matches how
//! hostjit already stubs other std facilities.
namespace std
{
class exception
{
public:
  exception() noexcept
  {
    __what_[0] = '\0';
  }
  exception(const exception&) noexcept            = default;
  exception& operator=(const exception&) noexcept = default;
  virtual ~exception() noexcept {}

  [[nodiscard]] virtual const char* what() const noexcept
  {
    return __what_;
  }

protected:
  //! Copy @p __msg into the buffer, truncating to fit and always NUL-terminating.
  void __assign(const char* __msg) noexcept
  {
    unsigned __i = 0;
    if (__msg != nullptr)
    {
      for (; __msg[__i] != '\0' && __i + 1 < sizeof(__what_); ++__i)
      {
        __what_[__i] = __msg[__i];
      }
    }
    __what_[__i] = '\0';
  }

  //! Append @p __s onto the current message (bounded, NUL-terminated).
  void __append(const char* __s) noexcept
  {
    unsigned __len = 0;
    while (__len + 1 < sizeof(__what_) && __what_[__len] != '\0')
    {
      ++__len;
    }
    if (__s != nullptr)
    {
      for (; *__s != '\0' && __len + 1 < sizeof(__what_); ++__s, ++__len)
      {
        __what_[__len] = *__s;
      }
    }
    __what_[__len] = '\0';
  }

  //! Append the decimal representation of @p __v (bounded, NUL-terminated).
  void __append_int(long __v) noexcept
  {
    char __digits[24];
    int __n             = 0;
    const bool __neg    = __v < 0;
    unsigned long __mag = __neg ? (0UL - static_cast<unsigned long>(__v)) : static_cast<unsigned long>(__v);
    if (__mag == 0)
    {
      __digits[__n++] = '0';
    }
    while (__mag != 0)
    {
      __digits[__n++] = static_cast<char>('0' + (__mag % 10));
      __mag /= 10;
    }
    char __out[26];
    int __k = 0;
    if (__neg)
    {
      __out[__k++] = '-';
    }
    while (__n != 0)
    {
      __out[__k++] = __digits[--__n];
    }
    __out[__k] = '\0';
    __append(__out);
  }

  char __what_[512];
};

class logic_error : public exception
{
public:
  explicit logic_error(const char* __msg) noexcept
  {
    __assign(__msg);
  }
};

class runtime_error : public exception
{
public:
  explicit runtime_error(const char* __msg) noexcept
  {
    __assign(__msg);
  }
};

class invalid_argument : public logic_error
{
public:
  explicit invalid_argument(const char* __msg) noexcept
      : logic_error(__msg)
  {}
};

class length_error : public logic_error
{
public:
  explicit length_error(const char* __msg) noexcept
      : logic_error(__msg)
  {}
};

class out_of_range : public logic_error
{
public:
  explicit out_of_range(const char* __msg) noexcept
      : logic_error(__msg)
  {}
};

class overflow_error : public runtime_error
{
public:
  explicit overflow_error(const char* __msg) noexcept
      : runtime_error(__msg)
  {}
};
} // namespace std

//! Freestanding stub for `::cuda::cuda_error`. The real type is hosted-only (it
//! formats via snprintf/cudaGetErrorString); here we derive from the stubbed
//! `::std::runtime_error` and fold the CUDA status / API name into `what()` so the
//! `_CCCL_TRY_CUDA_API` path (`_CCCL_THROW(::cuda::cuda_error, status, msg, api)`)
//! both compiles and yields a readable message. No conflict with the real type:
//! `cuda_error.h` defines nothing under `!_CCCL_HOSTED()`.
namespace cuda
{
class cuda_error : public ::std::runtime_error
{
public:
  cuda_error(int __status, const char* __msg, const char* __api = nullptr) noexcept
      : ::std::runtime_error(__msg)
      , __status_(__status)
  {
    if (__api != nullptr)
    {
      __append(" [");
      __append(__api);
      __append("]");
    }
    __append(" (cuda error ");
    __append_int(__status);
    __append(")");
  }

  [[nodiscard]] int status() const noexcept
  {
    return __status_;
  }

private:
  int __status_;
};
} // namespace cuda

#endif // !_CCCL_HOSTED()

// Redefine _CCCL_THROW to construct and throw the real exception type on the host
// (mirroring CCCL's own do/while + NV_IF_ELSE_TARGET form -- device code cannot
// throw, so it terminates). With the stubs above, `_TYPE(__VA_ARGS__)` names a
// defined type even in the freestanding hostjit environment.
#undef _CCCL_THROW
#define _CCCL_THROW(_TYPE, ...)                                                             \
  do                                                                                        \
  {                                                                                         \
    NV_IF_ELSE_TARGET(NV_IS_HOST, (throw _TYPE(__VA_ARGS__);), (::cuda::std::terminate();)) \
  } while (0)

#endif // _CUDA_EXPERIMENTAL_NVMATH_HOSTJIT_CUH
