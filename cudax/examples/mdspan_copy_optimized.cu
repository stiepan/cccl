//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! Exercises each JIT-wired dispatch branch of `cuda::experimental::copy`:
//!
//!   optimized  (branch 5) : rank-1 strided (stride-2) gather -> generic element-wise kernel
//!   contiguous (branch 2) : 2-D contiguous-inner, strided-outer (padded rows) copy
//!   transpose  (branch 4) : 2-D layout_left -> layout_right (shared-memory transpose)
//!
//! Built two ways (Makefile targets run-optimized / run-optimized-jit):
//!   * WITHOUT LAZY_JIT_DISPATCH — copy() launches directly; result is verified.
//!   * WITH -DLAZY_JIT_DISPATCH  — copy() returns a KernelDesc; the example NVRTC-compiles
//!     desc.code, looks up the uniform entry `cudax_copy_kernel`, launches it with
//!     desc.args.get() and desc's grid/block/shared config, then verifies.
//!
//! Usage: ./mdspan_copy_optimized [branch]   where branch is one of: optimized|contiguous|transpose|all (default all)

#include <cuda/mdspan>
#include <cuda/std/array>
#include <cuda/stream>

#include <cuda/experimental/__copy/vector_access.cuh>
#include <cuda/experimental/copy.cuh>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

namespace cudax = cuda::experimental;
#ifdef LAZY_JIT_DISPATCH
#  include <cuda.h>
#  include <nvrtc.h>
#  include <cuda/experimental/__lazy_jit/desc.cuh>
#  include <cxxabi.h>
using KernelDesc = cudax::lazy_jit::KernelDesc;

std::string demangle(const char* name)
{
  int status           = 0;
  char* demangled_name = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  if (demangled_name != nullptr)
  {
    std::string result(demangled_name);
    std::free(demangled_name);
    return result;
  }
  return name;
}
#endif

#define CHECK(call)                                                                        \
  do                                                                                       \
  {                                                                                        \
    cudaError_t err_ = (call);                                                             \
    if (err_ != cudaSuccess)                                                               \
    {                                                                                      \
      std::fprintf(                                                                        \
        stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                                             \
    }                                                                                      \
  } while (0)

#ifdef LAZY_JIT_DISPATCH
#  define NVRTC_CHECK(call)                                                                \
    do                                                                                     \
    {                                                                                      \
      nvrtcResult r_ = (call);                                                             \
      if (r_ != NVRTC_SUCCESS)                                                             \
      {                                                                                    \
        std::fprintf(stderr, "NVRTC error %s at %d\n", nvrtcGetErrorString(r_), __LINE__); \
        std::exit(EXIT_FAILURE);                                                           \
      }                                                                                    \
    } while (0)

#  define DRV_CHECK(call)                                                     \
    do                                                                        \
    {                                                                         \
      CUresult r_ = (call);                                                   \
      if (r_ != CUDA_SUCCESS)                                                 \
      {                                                                       \
        const char* msg_ = nullptr;                                          \
        cuGetErrorString(r_, &msg_);                                         \
        std::fprintf(stderr, "CUDA driver error %s at %d\n", msg_, __LINE__); \
        std::exit(EXIT_FAILURE);                                             \
      }                                                                       \
    } while (0)

// NVRTC-compile desc.code and launch the uniform entry `cudax_copy_kernel` with desc.args.
static void jit_compile_and_launch(const KernelDesc& desc, cudaStream_t stream)
{
  nvrtcProgram prog;
  NVRTC_CHECK(nvrtcCreateProgram(&prog, desc.code, "cudax_copy_jit.cu", 0, nullptr, nullptr));

  // __functor_kernel is a template, so its actual (mangled) symbol name depends on the instantiation
  // and isn't just its plain qualified name. Registering the name expression below, before compiling,
  // is what makes NVRTC instantiate it; nvrtcGetLoweredName (after a successful compile) then recovers
  // the mangled name that cuModuleGetFunction needs to find it.
  // desc.functor_t is a bare, comma-separated template-argument list (not wrapped in
  // "tuple<...>" or the like), built host-side as <Functor, Config, Args...> to match
  // lazy_jit::__functor_kernel's own template parameter list -- the generic __global__ entry
  // point that default-constructs Functor and forwards the launch arguments to its operator().
  const std::string name_expr = desc.functor_t;
  
  printf("name_expr: %s\n", name_expr.c_str());
  NVRTC_CHECK(nvrtcAddNameExpression(prog, name_expr.c_str()));

  const char* opts[] = {
    "-I" CCCL_CUDAX_INC,
    "-I" CCCL_LIBCUDACXX_INC,
    "-I" CCCL_CUB_INC,
    "-I" CCCL_THRUST_INC,
    "--gpu-architecture=compute_80",
    "-std=c++20"
  };
  nvrtcResult rc       = nvrtcCompileProgram(prog, sizeof(opts) / sizeof(opts[0]), opts);
  std::size_t log_size = 0;
  nvrtcGetProgramLogSize(prog, &log_size);
  if (log_size > 1)
  {
    std::string log(log_size, '\0');
    nvrtcGetProgramLog(prog, log.data());
    if (std::strstr(log.c_str(), "error"))
    {
      std::fprintf(stderr, "---- NVRTC log ----\n%s\n", log.c_str());
    }
  }
  if (rc != NVRTC_SUCCESS)
  {
    std::fprintf(stderr, "NVRTC compile FAILED: %s\n", nvrtcGetErrorString(rc));
    std::exit(EXIT_FAILURE);
  }

  // The lowered name string is owned by `prog`, so it must be copied out before nvrtcDestroyProgram.
  const char* lowered_name_ptr = nullptr;
  NVRTC_CHECK(nvrtcGetLoweredName(prog, name_expr.c_str(), &lowered_name_ptr));
  std::string kernel_name = lowered_name_ptr;

  std::size_t ptx_size = 0;
  NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
  std::vector<char> ptx(ptx_size);
  NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
  NVRTC_CHECK(nvrtcDestroyProgram(&prog));

  DRV_CHECK(cuInit(0));
  CUcontext ctx = nullptr;
  DRV_CHECK(cuCtxGetCurrent(&ctx));
  CUmodule mod;
  DRV_CHECK(cuModuleLoadData(&mod, ptx.data()));
  CUfunction fn;
  DRV_CHECK(cuModuleGetFunction(&fn, mod, kernel_name.c_str()));

  // desc.args.ptrs already holds one void* per *actual* kernel parameter (built by walking the args
  // tuple element-by-element), in declaration order -- exactly what cuLaunchKernel's kernelParams wants,
  // regardless of whether the target __global__ function takes N separate parameters (as here) or a
  // single packed one.
  void* params[] = { desc.args_bundle.get() };
  DRV_CHECK(cuLaunchKernel(
    fn,
    desc.grid_dim_x,
    desc.grid_dim_y,
    desc.grid_dim_z,
    desc.block_dim_x,
    desc.block_dim_y,
    desc.block_dim_z,
    desc.shared_mem_bytes,
    stream,
    params,
    nullptr));
  DRV_CHECK(cuModuleUnload(mod));
}
#endif // LAZY_JIT_DISPATCH

// Dispatch a copy either directly (non-JIT) or via the JIT round-trip.
template <typename SrcMd, typename DstMd>
static void dispatch_copy(const char* label, SrcMd src, DstMd dst, cuda::stream& stream)
{
#ifdef LAZY_JIT_DISPATCH
  KernelDesc desc = cudax::copy(src, dst, stream);
  std::printf("[%s] JIT: grid=(%u,%u,%u) block=(%u,%u,%u) smem=%u\ncode:\n%s\n",
              label,
              desc.grid_dim_x,
              desc.grid_dim_y,
              desc.grid_dim_z,
              desc.block_dim_x,
              desc.block_dim_y,
              desc.block_dim_z,
              desc.shared_mem_bytes,
              desc.code);
  jit_compile_and_launch(desc, stream.get());
#else
  (void) label;
  cudax::copy(src, dst, stream);
#endif
  stream.sync();
}

// `T` is a size-only stand-in for a concrete dtype (int, float, ...) -- copy() and mdspan never look
// at anything but its size/alignment. Rather than inventing our own such type, we reuse
// `cuda::experimental::__vector_access<N>` directly: it's *already* exactly this ("an alignas(N)
// blob of N bytes, nothing else"), it's already a first-class citizen of the JIT-dispatch path (see
// repr_type_vector_access.cuh -- copy()'s own vectorization path reinterprets same-sized elements as
// this very type before ever reaching JIT dispatch), so there's no dedicated stringification support
// to add for it. Fill/verify below goes through `to_u64`/`from_u64` (plain memcpy helpers) rather than
// arithmetic on `T` directly, since `T` itself only supports construction/copy-assignment/equality --
// matching what copy()'s kernels actually require of any element type.
using T          = cudax::__vector_access<4>;
using extents1_t = cuda::std::dextents<int, 1>;
using extents2_t = cuda::std::dextents<int, 2>;
using strides1_t = cuda::dstrides<int, 1>;
using strides2_t = cuda::dstrides<int, 2>;
using relaxed_t  = cuda::layout_stride_relaxed;
template <int R>
using mapping_t = typename relaxed_t::mapping<cuda::std::dextents<int, R>>;

// Treated as a fully opaque byte blob throughout (never touching its internal layout/member names):
// only its size participates.
static T from_u64(unsigned long long pattern)
{
  T out{};
  std::memcpy(&out, &pattern, sizeof(out) < sizeof(pattern) ? sizeof(out) : sizeof(pattern));
  return out;
}
static unsigned long long to_u64(const T& v)
{
  unsigned long long out = 0;
  std::memcpy(&out, &v, sizeof(v) < sizeof(out) ? sizeof(v) : sizeof(out));
  return out;
}

// Allocate a device buffer of `n` elements, filled with an arange (src) or zeroed (dst).
static T* device_arange(int n)
{
  std::vector<T> h(n);
  for (int i = 0; i < n; ++i)
  {
    h[i] = from_u64(static_cast<unsigned long long>(i));
  }
  T* d = nullptr;
  CHECK(cudaMalloc(&d, n * sizeof(T)));
  CHECK(cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
static T* device_zero(int n)
{
  T* d = nullptr;
  CHECK(cudaMalloc(&d, n * sizeof(T)));
  CHECK(cudaMemset(d, 0, n * sizeof(T)));
  return d;
}

// branch 5: rank-1 strided (stride 2) gather -> __copy_optimized
static int run_optimized(cuda::stream& stream)
{
  constexpr int N = 4096;
  extents1_t ext(N);
  mapping_t<1> src_map(ext, strides1_t(cuda::std::array<int, 1>{2}), 0);
  mapping_t<1> dst_map(ext, strides1_t(cuda::std::array<int, 1>{1}), 0);
  const int src_alloc = static_cast<int>(src_map.required_span_size());
  const int dst_alloc = static_cast<int>(dst_map.required_span_size());

  T* d_base = device_arange(src_alloc);
  T* d_out  = device_zero(dst_alloc);
  cuda::device_mdspan<T, extents1_t, relaxed_t> src(d_base, src_map);
  cuda::device_mdspan<T, extents1_t, relaxed_t> dst(d_out, dst_map);

  dispatch_copy("optimized", src, dst, stream);

  std::vector<T> h(dst_alloc);
  CHECK(cudaMemcpy(h.data(), d_out, dst_alloc * sizeof(T), cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < N; ++i)
  {
    if (to_u64(h[i]) != static_cast<unsigned long long>(2 * i))
    {
      ++bad;
    }
  }
  CHECK(cudaFree(d_base));
  CHECK(cudaFree(d_out));
  std::printf("[optimized] %s (N=%d)\n", bad == 0 ? "OK" : "FAILED", N);
  return bad;
}

// branch 2: 2-D contiguous inner, strided (padded) outer -> __copy_contiguous
static int run_contiguous(cuda::stream& stream)
{
  constexpr int M = 32;
  constexpr int W = 16384; // inner bytes (W*sizeof(T) = 64KB) >= bytes-in-flight -> contiguous branch
  constexpr int P = W + 5; // padded row pitch -> outer stride != inner extent (no coalesce)
  extents2_t ext(M, W);
  mapping_t<2> src_map(ext, strides2_t(cuda::std::array<int, 2>{P, 1}), 0);
  mapping_t<2> dst_map(ext, strides2_t(cuda::std::array<int, 2>{W, 1}), 0);
  const int src_alloc = static_cast<int>(src_map.required_span_size());
  const int dst_alloc = static_cast<int>(dst_map.required_span_size());

  T* d_base = device_arange(src_alloc);
  T* d_out  = device_zero(dst_alloc);
  cuda::device_mdspan<T, extents2_t, relaxed_t> src(d_base, src_map);
  cuda::device_mdspan<T, extents2_t, relaxed_t> dst(d_out, dst_map);

  dispatch_copy("contiguous", src, dst, stream);

  std::vector<T> h(dst_alloc);
  CHECK(cudaMemcpy(h.data(), d_out, dst_alloc * sizeof(T), cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < W; ++j)
    {
      if (to_u64(h[static_cast<std::size_t>(i) * W + j]) != static_cast<unsigned long long>(i * P + j))
      {
        ++bad;
      }
    }
  }
  CHECK(cudaFree(d_base));
  CHECK(cudaFree(d_out));
  std::printf("[contiguous] %s (M=%d, W=%d)\n", bad == 0 ? "OK" : "FAILED", M, W);
  return bad;
}

// branch 4: 2-D transpose, layout_left src -> layout_right dst -> __copy_shared_mem
static int run_transpose(cuda::stream& stream)
{
  constexpr int N = 512;
  extents2_t ext(N, N);
  mapping_t<2> src_map(ext, strides2_t(cuda::std::array<int, 2>{1, N}), 0); // column-major (transpose view)
  mapping_t<2> dst_map(ext, strides2_t(cuda::std::array<int, 2>{N, 1}), 0); // row-major
  const int src_alloc = static_cast<int>(src_map.required_span_size());
  const int dst_alloc = static_cast<int>(dst_map.required_span_size());

  T* d_base = device_arange(src_alloc);
  T* d_out  = device_zero(dst_alloc);
  cuda::device_mdspan<T, extents2_t, relaxed_t> src(d_base, src_map);
  cuda::device_mdspan<T, extents2_t, relaxed_t> dst(d_out, dst_map);

  dispatch_copy("transpose", src, dst, stream);

  std::vector<T> h(dst_alloc);
  CHECK(cudaMemcpy(h.data(), d_out, dst_alloc * sizeof(T), cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      // src(i,j) == d_base[i + j*N] == i + j*N ; dst is row-major
      if (to_u64(h[static_cast<std::size_t>(i) * N + j]) != static_cast<unsigned long long>(i + j * N))
      {
        ++bad;
      }
    }
  }
  CHECK(cudaFree(d_base));
  CHECK(cudaFree(d_out));
  std::printf("[transpose] %s (N=%d)\n", bad == 0 ? "OK" : "FAILED", N);
  return bad;
}

int main(int argc, char** argv)
{
  const char* which = (argc > 1) ? argv[1] : "all";
  cuda::stream stream{cuda::device_ref{0}};

  int bad = 0;
  const bool all = std::strcmp(which, "all") == 0;
  if (all || std::strcmp(which, "optimized") == 0)
  {
    bad += run_optimized(stream);
  }
  if (all || std::strcmp(which, "contiguous") == 0)
  {
    bad += run_contiguous(stream);
  }
  if (all || std::strcmp(which, "transpose") == 0)
  {
    bad += run_transpose(stream);
  }
  return bad == 0 ? 0 : EXIT_FAILURE;
}
