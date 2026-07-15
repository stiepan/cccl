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

//! @file
//! Opt-in shim (enabled by defining `NVMATH_HOSTJIT` before inclusion) that makes
//! CCCL's error reporting propagatable across an in-process hostjit boundary.
//!
//! hostjit compiles CCCL with `-DCCCL_DISABLE_EXCEPTIONS=1`, so `_CCCL_THROW`
//! degrades to `::cuda::std::terminate()`: any precondition violation (e.g. an
//! invalid mdspan layout) or CUDA error aborts the whole host process instead of
//! surfacing an error the embedder can handle. When `NVMATH_HOSTJIT` is defined,
//! this header redefines `_CCCL_THROW` to instead throw a lightweight
//! `nvmath_hostjit_error`, which the embedder catches at its entry point and
//! converts into a host-language exception (e.g. a Python exception).
//!
//! When `NVMATH_HOSTJIT` is not defined this header is empty, so it is inert for
//! ordinary (hosted / NVRTC) builds.
//!
//! Must be included BEFORE the CCCL headers whose `_CCCL_THROW` uses should be
//! overridden (e.g. `<cuda/experimental/copy.cuh>`). It force-includes
//! `exception_macros.h` first so that header's include guard is set and our
//! redefinition survives subsequent (guard-skipped) inclusions.

#ifdef NVMATH_HOSTJIT

#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__exception/terminate.h>
#  include <cuda/std/__type_traits/is_convertible.h>

#  include <nv/target>

namespace cuda::experimental
{
//! @brief Lightweight, freestanding-friendly exception carrying a CCCL diagnostic.
//!
//! Only stores pointers to string literals (type name, file, message), which have
//! static storage duration, so instances are trivially copyable and reference no
//! heap or exception-object-owned storage.
class nvmath_hostjit_error
{
public:
  constexpr nvmath_hostjit_error(const char* __type, const char* __file, int __line, const char* __msg) noexcept
      : __type_(__type)
      , __file_(__file)
      , __line_(__line)
      , __msg_(__msg)
  {}

  [[nodiscard]] constexpr const char* what() const noexcept
  {
    return __msg_;
  }
  [[nodiscard]] constexpr const char* type_name() const noexcept
  {
    return __type_;
  }
  [[nodiscard]] constexpr const char* file() const noexcept
  {
    return __file_;
  }
  [[nodiscard]] constexpr int line() const noexcept
  {
    return __line_;
  }

private:
  const char* __type_;
  const char* __file_;
  int __line_;
  const char* __msg_;
};

//! @brief Extract the human-readable message from `_CCCL_THROW`'s trailing args.
//!
//! Call sites differ: `_CCCL_THROW(::std::invalid_argument, "msg")` puts the
//! message first, while `_CCCL_THROW(::cuda::cuda_error, status, "msg", ...)` puts
//! a status int (and possibly a source_location) around it. We return the first
//! argument convertible to `const char*`, skipping the rest.
[[nodiscard]] constexpr const char* __nvmath_pick_msg() noexcept
{
  return "unspecified error";
}

template <class _First, class... _Rest>
[[nodiscard]] constexpr const char* __nvmath_pick_msg(_First __first, _Rest... __rest) noexcept
{
  if constexpr (::cuda::std::is_convertible_v<_First, const char*>)
  {
    return static_cast<const char*>(__first);
  }
  else if constexpr (sizeof...(_Rest) > 0)
  {
    return ::cuda::experimental::__nvmath_pick_msg(__rest...);
  }
  else
  {
    return "unspecified error";
  }
}
} // namespace cuda::experimental

// Redefine _CCCL_THROW to throw our exception on the host, terminate on device
// (mirroring CCCL's own NV_IF_ELSE_TARGET pattern -- device code cannot throw).
#  undef _CCCL_THROW
#  define _CCCL_THROW(_TYPE, ...)                                                                             \
    NV_IF_ELSE_TARGET(NV_IS_HOST,                                                                             \
                      (throw ::cuda::experimental::nvmath_hostjit_error(                                      \
                         #_TYPE, __FILE__, __LINE__, ::cuda::experimental::__nvmath_pick_msg(__VA_ARGS__));), \
                      (::cuda::std::terminate();))

#endif // NVMATH_HOSTJIT

#endif // _CUDA_EXPERIMENTAL_NVMATH_HOSTJIT_CUH
