//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAZY_JIT_REPR_TYPE_EXTENTS_H
#define _CUDAX__LAZY_JIT_REPR_TYPE_EXTENTS_H

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
#  include <cuda/std/__mdspan/extents.h>

#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_common.cuh>
#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_integer.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::lazy_jit
{
//! @brief Spells a single @c cuda::std::extents extent, using the @c cuda::std::dynamic_extent marker
//! instead of its underlying magic-number value (@c size_t(-1)) for dynamic dimensions. Not a
//! @c repr_type specialization itself (there's no *type* to key off -- @p _Extent is a bare value), just
//! the per-extent building block @c repr_type<cuda::std::extents<...>> (below) folds over.
template <::cuda::std::size_t _Extent>
struct repr_extent
{
  static constexpr auto __make()
  {
    if constexpr (_Extent == ::cuda::std::dynamic_extent)
    {
      return __fixed_string("::cuda::std::dynamic_extent");
    }
    else
    {
      return repr_integer<::cuda::std::size_t, _Extent>::fixed_string;
    }
  }
  static constexpr auto fixed_string = __make();
};

template <typename _IndexType, ::cuda::std::size_t... _Extents>
struct repr_type<::cuda::std::extents<_IndexType, _Extents...>>
{
  static constexpr auto fixed_string =
    __fixed_string("::cuda::std::extents<") + repr_type<_IndexType>::fixed_string
    + ((__fixed_string(", ") + repr_extent<_Extents>::fixed_string) + ... + __fixed_string("")) + ">";
};
} // namespace cuda::experimental::lazy_jit

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX__LAZY_JIT_REPR_TYPE_EXTENTS_H
