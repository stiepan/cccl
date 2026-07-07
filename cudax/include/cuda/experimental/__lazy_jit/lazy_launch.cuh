//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAZY_JIT_LAUNCH_H
#define _CUDAX__LAZY_JIT_LAUNCH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/launch>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__mdspan/default_accessor.h>
#include <cuda/std/array>

#ifdef LAZY_JIT_DISPATCH
#  include <cuda/std/__type_traits/remove_cvref.h>
#  include <cuda/std/tuple>

#  include <cuda/experimental/__lazy_jit/desc.cuh>
#  include <cuda/experimental/__lazy_jit/functor_kernel.cuh>
#  include <cuda/experimental/__lazy_jit/repr_type/repr_includes.cuh>
#  include <cuda/experimental/__lazy_jit/repr_type/repr_type.cuh>
#  include <cuda/experimental/__copy/tensor_iterator.cuh>
#  include <cuda/experimental/__copy_bytes/types.cuh>

#  include <string>
#  include <utility>
#endif // LAZY_JIT_DISPATCH

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
namespace lazy_jit
{
#ifdef LAZY_JIT_DISPATCH
//! @brief True for @c cuda::dynamic_shared_memory_option<T> for any @c T, false otherwise.
template <typename _Tp>
inline constexpr bool __is_dynamic_shared_memory_option_v = false;

template <typename _Tp>
inline constexpr bool __is_dynamic_shared_memory_option_v<::cuda::dynamic_shared_memory_option<_Tp>> = true;

//! @brief Size in bytes of the @c dynamic_shared_memory_option present in @p __opts, or 0 if there is none.
//! @c kernel_config disallows duplicate option kinds, so at most one element of @p __opts can match.
template <typename... _Options>
[[nodiscard]] inline ::cuda::std::size_t __dynamic_shared_memory_bytes(const ::cuda::std::tuple<_Options...>& __opts)
{
  ::cuda::std::size_t __bytes = 0;
  ::cuda::std::apply(
    [&__bytes](const auto&... __opt) {
      (
        [&] {
          using _Opt = ::cuda::std::remove_cvref_t<decltype(__opt)>;
          if constexpr (__is_dynamic_shared_memory_option_v<_Opt>)
          {
            __bytes = __opt.size_bytes();
          }
        }(),
        ...);
    },
    __opts);
  return __bytes;
}

//! @brief Fill in @p __desc's grid/block dimensions and dynamic shared memory size from @p __config.
template <typename _Config>
inline void __fill_launch_dims(KernelDesc& __desc, const _Config& __config)
{
  const auto __grid_dims  = ::cuda::block.dims(::cuda::grid, __config);
  const auto __block_dims = ::cuda::gpu_thread.dims(::cuda::block, __config);
  __desc.grid_dim_x       = __grid_dims.x;
  __desc.grid_dim_y       = __grid_dims.y;
  __desc.grid_dim_z       = __grid_dims.z;
  __desc.block_dim_x      = __block_dims.x;
  __desc.block_dim_y      = __block_dims.y;
  __desc.block_dim_z      = __block_dims.z;
  __desc.shared_mem_bytes = static_cast<unsigned>(__dynamic_shared_memory_bytes(__config.options()));
}

//! @brief Same as @c make_kernel_desc, but for a *functor*-style kernel: instead of an already-fully-
//! instantiated kernel (function pointer or reference), @p _Functor is a stateless, default-constructible
//! type with a templated @c operator()(_Config, ...) (see @ref __copy_optimized_impl for an example).
//! JIT-side, it's launched through the generic @c lazy_jit::__functor_kernel<_Functor, _Config>
//! entry point -- whose own template argument list is just @c <_Functor, _Config>, with @p __args
//! bundled into a single trailing tuple parameter whose type (@c operator_args_t<_Functor, _Config>) is
//! derived from @p _Functor itself. So, unlike a bespoke multi-parameter kernel, there's no separate
//! per-kernel argument-type list to build (or keep in sync) here at all.
//!
//! @tparam _Functor The kernel functor type (never actually constructed here).
template <typename _Functor, typename _Submitter, typename _LaunchConfig, typename... _Args>
KernelDesc make_kernel_desc(_Submitter&& __submitter, const _LaunchConfig& __conf, _Args&&... __args)
{
  static_assert(::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Submitter>, ::cuda::stream_ref>,
                "Submitter must be a stream_ref");
  KernelDesc __desc{};
  const auto __hierarchy = ::cuda::__unpack_hierarchy_if_needed(__conf);
  using _Hierarchy       = decltype(__hierarchy);
  using functor_kernel_t = functor_kernel_impl<_Functor, _Hierarchy>;
  using args_t           = typename functor_kernel_t::args_t;
  auto __args_tuple      = args_t(__hierarchy, __args...);
  __desc.code            = repr_includes<functor_kernel_t>::fixed_string.c_str();
  __desc.functor_t       = repr_type<functor_kernel_t>::fixed_string.c_str();
  __desc.args_bundle     = make_kernel_args(__args_tuple);
  __fill_launch_dims(__desc, __conf);
  __desc.type_id_name = typeid(functor_kernel_t).name();
  __desc.type_id_hash = typeid(functor_kernel_t).hash_code();
  return __desc;
}
#endif // LAZY_JIT_DISPATCH
} // namespace lazy_jit
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

//! @brief Launch a kernel, or (when @c LAZY_JIT_DISPATCH is defined) stringify the launch for JIT
//! dispatch instead of actually launching it.
#ifdef LAZY_JIT_DISPATCH
//! @param functor_t A stateless kernel functor type (see @c make_kernel_desc's docs) -- @p functor_t
//!   is a single type (never a comma-separated list), so, unlike @p kernel/@p conf/@p ..., it needs no
//!   special wrapping to survive being an (outer) macro argument.
#  define LAUNCH_OR_LAZY_JIT_DISPATCH(stream, conf, functor_t, ...) \
    ::cuda::experimental::lazy_jit::make_kernel_desc<functor_t>(stream, conf, __VA_ARGS__)
#  define DISPATCH_RET_TYPE ::cuda::experimental::lazy_jit::KernelDesc
#  define DISPATCH_RET_VOID return ::cuda::experimental::lazy_jit::KernelDesc{};
#else
//! @brief Non-JIT branch: @p functor_t is launched directly via @c cuda::launch's own functor-kernel
//! overload (see the @c cuda::launch overload taking a functor in `<cuda/__launch/launch.h>`), which
//! handles picking/instantiating an appropriate @c __global__ launcher itself.
#  define LAUNCH_OR_LAZY_JIT_DISPATCH(stream, conf, functor_t, ...) \
    ::cuda::launch(stream, conf, functor_t{}, __VA_ARGS__)
#  define DISPATCH_RET_TYPE void
#  define DISPATCH_RET_VOID return;
#endif // LAZY_JIT_DISPATCH

#endif // _CUDAX__LAZY_JIT_LAUNCH_H
