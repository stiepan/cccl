//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! Minimal-dependency sibling of mdspan_copy_optimized_jit: exercises the same JIT-wired
//! dispatch branches of `cuda::experimental::copy`, but only *prints* the returned KernelDesc
//! (code string, functor template-argument list, launch config) -- it never NVRTC-compiles or
//! launches anything.
//!
//! Goal of this example: show how few link-time (and, here, *runtime*) dependencies a
//! `LAZY_JIT_DISPATCH` build actually needs.
//!
//!   * No `-lnvrtc` (we never create an nvrtcProgram).
//!   * No `-lcuda` at *link* time: any driver calls that do happen resolve their entry points via
//!     `dlopen("libcuda.so.1")` + `cuGetProcAddress` at first use (see cuda/__driver/driver_api.h)
//!     rather than through a `DT_NEEDED` link dependency. `-ldl` is technically all that's needed,
//!     and on glibc >= 2.34 that's folded into libc itself.
//!   * No driver calls at *runtime* either, in this particular example: the stream below is a bare
//!     `stream_ref` from a raw (never-created) handle rather than an owning `cuda::stream`, and the
//!     handful of device-attribute queries `copy()`'s dispatch logic would otherwise perform
//!     (max vector width, bytes-in-flight, shared-memory/SM/thread-block limits) are pessimized to
//!     conservative, minimum-supported-architecture constants under `LAZY_JIT_DISPATCH` (see
//!     vector_access.cuh, copy_contiguous.cuh, copy_shared_memory_utils.cuh) instead of querying a
//!     live device. So this example runs to completion -- including exercising the shared-memory
//!     tiling heuristic -- without ever touching the CUDA driver, even at runtime, and without any
//!     GPU present.
//!   * `-lcudart` is still required for exactly one symbol, `cudaGetErrorString`, used only to
//!     format `cuda::cuda_error`'s message on the (here, unreached) error path -- not a real
//!     functional dependency on the CUDA Runtime.
//!   * "Device" buffers are plain host heap allocations -- since nothing is ever launched, the
//!     pointers are only used for address-alignment/aliasing checks and for building the
//!     descriptor's argument bundle, never dereferenced on a GPU.
//!
//! Usage: ./mdspan_copy_optimized_jit_desc [branch]   branch in: optimized|contiguous|transpose|all (default all)

#include <cuda/mdspan>
#include <cuda/std/array>
#include <cuda/stream>

#include <cuda/experimental/__copy/vector_access.cuh>
#include <cuda/experimental/copy.cuh>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace cudax = cuda::experimental;
using KernelDesc = cudax::lazy_jit::KernelDesc;

// Print a KernelDesc without compiling or launching anything.
static void print_desc(const char* label, const KernelDesc& desc)
{
  std::printf(
    "[%s] grid=(%u,%u,%u) block=(%u,%u,%u) smem=%u\n"
    "  functor_t: %s\n"
    "  code:\n%s\n\n",
    label,
    desc.grid_dim_x,
    desc.grid_dim_y,
    desc.grid_dim_z,
    desc.block_dim_x,
    desc.block_dim_y,
    desc.block_dim_z,
    desc.shared_mem_bytes,
    desc.functor_t.c_str(),
    desc.code
  );
}

// `T` is a size-only stand-in for a concrete dtype (int, float, ...) -- copy() and mdspan never look
// at anything but its size/alignment. We reuse `cuda::experimental::__vector_access<N>` directly
// rather than inventing our own such type (see mdspan_copy_optimized.cu for why).
using T          = cudax::__vector_access<4>;
using extents1_t = cuda::std::dextents<int, 1>;
using extents2_t = cuda::std::dextents<int, 2>;
using strides1_t = cuda::dstrides<int, 1>;
using strides2_t = cuda::dstrides<int, 2>;
using relaxed_t  = cuda::layout_stride_relaxed;
template <int R>
using mapping_t = typename relaxed_t::mapping<cuda::std::dextents<int, R>>;

// Plain host-heap "device" buffer: never dereferenced since JIT mode never launches, only its
// address is used to build mdspans and the KernelDesc's argument bundle.
static T* host_buffer(int n)
{
  return new T[static_cast<std::size_t>(n)]();
}

// branch 5: rank-1 strided (stride 2) gather -> __copy_optimized
static void run_optimized(cuda::stream_ref stream)
{
  constexpr int N = 4096;
  extents1_t ext(N);
  mapping_t<1> src_map(ext, strides1_t(cuda::std::array<int, 1>{2}), 0);
  mapping_t<1> dst_map(ext, strides1_t(cuda::std::array<int, 1>{1}), 0);
  T* d_base = host_buffer(static_cast<int>(src_map.required_span_size()));
  T* d_out  = host_buffer(static_cast<int>(dst_map.required_span_size()));
  cuda::device_mdspan<T, extents1_t, relaxed_t> src(d_base, src_map);
  cuda::device_mdspan<T, extents1_t, relaxed_t> dst(d_out, dst_map);

  KernelDesc desc = cudax::copy(src, dst, stream);
  print_desc("optimized", desc);

  delete[] d_base;
  delete[] d_out;
}

// branch 2: 2-D contiguous inner, strided (padded) outer -> __copy_contiguous
static void run_contiguous(cuda::stream_ref stream)
{
  constexpr int M = 32;
  constexpr int W = 16384; // inner bytes (W*4 = 64KB) >= bytes-in-flight -> contiguous branch
  constexpr int P = W + 5; // padded row pitch -> outer stride != inner extent (no coalesce)
  extents2_t ext(M, W);
  mapping_t<2> src_map(ext, strides2_t(cuda::std::array<int, 2>{P, 1}), 0);
  mapping_t<2> dst_map(ext, strides2_t(cuda::std::array<int, 2>{W, 1}), 0);
  T* d_base = host_buffer(static_cast<int>(src_map.required_span_size()));
  T* d_out  = host_buffer(static_cast<int>(dst_map.required_span_size()));
  cuda::device_mdspan<T, extents2_t, relaxed_t> src(d_base, src_map);
  cuda::device_mdspan<T, extents2_t, relaxed_t> dst(d_out, dst_map);

  KernelDesc desc = cudax::copy(src, dst, stream);
  print_desc("contiguous", desc);

  delete[] d_base;
  delete[] d_out;
}

// branch 4: 2-D transpose, layout_left src -> layout_right dst -> __copy_shared_mem
static void run_transpose(cuda::stream_ref stream)
{
  constexpr int N = 512;
  extents2_t ext(N, N);
  mapping_t<2> src_map(ext, strides2_t(cuda::std::array<int, 2>{1, N}), 0); // column-major (transpose view)
  mapping_t<2> dst_map(ext, strides2_t(cuda::std::array<int, 2>{N, 1}), 0); // row-major
  T* d_base = host_buffer(static_cast<int>(src_map.required_span_size()));
  T* d_out  = host_buffer(static_cast<int>(dst_map.required_span_size()));
  cuda::device_mdspan<T, extents2_t, relaxed_t> src(d_base, src_map);
  cuda::device_mdspan<T, extents2_t, relaxed_t> dst(d_out, dst_map);

  KernelDesc desc = cudax::copy(src, dst, stream);
  print_desc("transpose", desc);

  delete[] d_base;
  delete[] d_out;
}

int main(int argc, char** argv)
{
  const char* which = (argc > 1) ? argv[1] : "all";
  // A bare stream_ref constructed from a raw handle is a non-owning, driver-free wrapper (just
  // stores the pointer) -- unlike cuda::stream{cuda::device_ref{0}}, it never touches the CUDA
  // driver at all. Combined with the pessimized device-capability queries in the dispatch path,
  // this lets the whole example run to completion (printing real KernelDesc contents) without ever
  // dlopen-ing libcuda.so.1 or requiring a live GPU/driver.
  cuda::stream_ref stream{cudaStream_t{nullptr}};

  const bool all = std::strcmp(which, "all") == 0;
  if (all || std::strcmp(which, "optimized") == 0)
  {
    run_optimized(stream);
  }
  if (all || std::strcmp(which, "contiguous") == 0)
  {
    run_contiguous(stream);
  }
  if (all || std::strcmp(which, "transpose") == 0)
  {
    run_transpose(stream);
  }
  return EXIT_SUCCESS;
}
