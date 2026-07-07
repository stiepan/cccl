//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAZY_JIT_DESC_H
#define _CUDAX__LAZY_JIT_DESC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__type_traits/remove_cvref.h>

#  include <memory>
#  include <string>
#  include <utility>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
namespace lazy_jit
{
//! @brief Type-erased, self-managed bundle with kernel arguments.
using kernel_args_ptr = ::std::unique_ptr<void, void (*)(void*)>;

template <class _Args>
inline void kernel_args_delete(void* p) noexcept
{
  delete static_cast<_Args*>(p);
}

//! @brief Heap-allocate an argument bundle of type @c _Args and wrap it type-erased.
template <class _ArgsTuple>
[[nodiscard]] inline kernel_args_ptr make_kernel_args(_ArgsTuple&& __args)
{
  using _Args = ::cuda::std::remove_cvref_t<_ArgsTuple>;
  return kernel_args_ptr(new _Args{__args}, &kernel_args_delete<_Args>);
}

//! @brief Result of the JIT copy dispatch (only used when @c LAZY_JIT_DISPATCH is defined).
struct KernelDesc
{
  const char* code{};
  //! @brief Bare, comma-separated template argument list (e.g. "T0, T1, T2") for the @c __global__
  //! kernel template defined/pulled in by @c code -- NOT wrapped in "kernel<...>" or "tuple<...>". The
  //! caller builds the full NVRTC name-expression by wrapping this in the kernel's qualified name, e.g.
  //! @c "::ns::my_kernel<" + functor_t + ">".
  ::std::string functor_t{};
  kernel_args_ptr args_bundle{nullptr, +[](void*) {}};
  const char* type_id_name{};
  unsigned long long type_id_hash = 0;
  unsigned grid_dim_x             = 0;
  unsigned grid_dim_y             = 1;
  unsigned grid_dim_z             = 1;
  unsigned block_dim_x            = 0;
  unsigned block_dim_y            = 1;
  unsigned block_dim_z            = 1;
  unsigned shared_mem_bytes       = 0;
};
} // namespace lazy_jit
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX__LAZY_JIT_DESC_H
