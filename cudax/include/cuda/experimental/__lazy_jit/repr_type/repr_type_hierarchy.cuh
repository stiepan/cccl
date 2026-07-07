//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAZY_JIT_REPR_TYPE_HIERARCHY_H
#define _CUDAX__LAZY_JIT_REPR_TYPE_HIERARCHY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__hierarchy/hierarchy_dimensions.h>
#  include <cuda/__hierarchy/hierarchy_levels.h>
#  include <cuda/__hierarchy/level_dimensions.h>
#  include <cuda/std/__type_traits/always_false.h>

#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_common.cuh>
#  include <cuda/experimental/__lazy_jit/repr_type/repr_type_extents.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::lazy_jit
{
//! @brief Maps a supported hierarchy level type to its NVRTC-parsable spelling.
//!
//! JIT dispatch only supports the levels the copy kernels' launch configuration can actually be
//! instantiated with (thread_level, block_level, grid_level). Any other level (warp_level,
//! cluster_level, or a user-defined level) trips the static_assert below with a clear diagnostic,
//! rather than silently degrading to "unknown" or failing later with an obscure incomplete-type error.
template <typename _Level>
struct repr_level
{
  static_assert(::cuda::std::__always_false_v<_Level>, "JIT dispatch does not support this hierarchy level");

  static constexpr auto fixed_string = __fixed_string("unsupported-level");
};

#  define _CCCLX_JIT_LEVEL_NAME(_level)                                        \
    template <>                                                                \
    struct repr_level<::cuda::_level>                                          \
    {                                                                          \
      static constexpr auto fixed_string = __fixed_string("::cuda::" #_level); \
    }

_CCCLX_JIT_LEVEL_NAME(thread_level);
_CCCLX_JIT_LEVEL_NAME(block_level);
_CCCLX_JIT_LEVEL_NAME(grid_level);

#  undef _CCCLX_JIT_LEVEL_NAME

template <typename _Level, typename _Exts>
struct repr_type<::cuda::hierarchy_level_desc<_Level, _Exts>>
{
  static constexpr auto fixed_string = __fixed_string("::cuda::hierarchy_level_desc<") + repr_level<_Level>::fixed_string
                                      + ", " + repr_type<_Exts>::fixed_string + ">";
};

//! @brief Spells out a @c cuda::hierarchy by inspecting its actual bottom unit and levels (with their
//! actual, possibly-dynamic extents), rather than assuming a fixed shape.
template <typename _BottomUnit, typename... _LevelDescs>
struct repr_type<::cuda::hierarchy<_BottomUnit, _LevelDescs...>>
{
  static constexpr auto fixed_string =
    __fixed_string("::cuda::hierarchy<") + repr_level<_BottomUnit>::fixed_string
    + ((__fixed_string(", ") + repr_type<_LevelDescs>::fixed_string) + ... + __fixed_string("")) + ">";
};
} // namespace cuda::experimental::lazy_jit

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX__LAZY_JIT_REPR_TYPE_HIERARCHY_H
