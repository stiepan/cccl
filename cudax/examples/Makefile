# Basic standalone build for mdspan_copy_transpose.cu.
# This is just the nvcc command used to build the example by hand; the
# canonical build path is still CMake (target cudax.example.mdspan_copy_transpose).
#
# Usage:
#   make                 # build
#   make run             # build and run (default N=1023)
#   make clean

NVCC    ?= /usr/local/cuda/bin/nvcc
ARCH    ?= sm_80
CCCL    := ../..

INCLUDES := -I $(CCCL)/cudax/include \
            -I $(CCCL)/libcudacxx/include \
            -I $(CCCL)/cub \
            -I $(CCCL)/thrust

NVCCFLAGS := -std=c++20 -arch=$(ARCH) -Wno-deprecated-gpu-targets \
             --expt-relaxed-constexpr --extended-lambda

# mdspan_copy_transpose: mdspan_copy_transpose.cu
# 	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@

# __copy_optimized (branch 5) example, built two ways:
#   mdspan_copy_optimized      -> regular dispatch (launches + verifies)
#   mdspan_copy_optimized_jit  -> -DLAZY_JIT_DISPATCH (prints KernelDesc)
mdspan_copy_optimized: mdspan_copy_optimized.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@

# The JIT variant NVRTC-compiles the emitted code string and launches it via the
# driver API, so it needs the include paths baked in (for NVRTC) and -lnvrtc -lcuda.
JIT_INCDEFS := -DCCCL_CUDAX_INC='"$(abspath $(CCCL)/cudax/include)"' \
               -DCCCL_LIBCUDACXX_INC='"$(abspath $(CCCL)/libcudacxx/include)"' \
               -DCCCL_CUB_INC='"$(abspath $(CCCL)/cub)"' \
               -DCCCL_THRUST_INC='"$(abspath $(CCCL)/thrust)"'

mdspan_copy_optimized_jit: mdspan_copy_optimized.cu
	$(NVCC) $(NVCCFLAGS) -DLAZY_JIT_DISPATCH $(JIT_INCDEFS) $(INCLUDES) $< -o $@ -lnvrtc -lcuda

# Minimal-dependency sibling of mdspan_copy_optimized_jit: only prints the returned KernelDesc,
# never NVRTC-compiles or launches anything. Built with a *plain host compiler* (no nvcc) to show
# how few link-time (and, here, runtime) dependencies a LAZY_JIT_DISPATCH build actually needs:
#   * no -lnvrtc  (never creates an nvrtcProgram)
#   * no -lcuda at link time (any driver calls resolve their entry points via
#                dlopen("libcuda.so.1") + cuGetProcAddress at first use -- see
#                cuda/__driver/driver_api.h -- rather than a link-time dependency; -ldl is all
#                that's needed, and on glibc >= 2.34 that's folded into libc itself)
#   * no driver calls at runtime either: uses a bare stream_ref (no context/stream creation) and
#     pessimized, conservative device-capability constants instead of live queries under
#     LAZY_JIT_DISPATCH (see vector_access.cuh, copy_contiguous.cuh,
#     copy_shared_memory_utils.cuh) -- runs to completion with no GPU present at all.
#   * no -lcudart either: cuda::cuda_error's message formatter (__format_cuda_error, see
#     libcudacxx/include/cuda/std/__exception/cuda_error.h) is the only thing that would need
#     cudaGetErrorString, and it's never actually invoked on this driver-free code path -- but at
#     -O0 GCC still emits every parsed `static` function regardless of whether it's called, so the
#     unused reference lingers into the object file. -O1 (or -ffunction-sections/--gc-sections at
#     -O0) lets it prove the function is dead and drop it, taking the cudart dependency with it.
CXX ?= g++
mdspan_copy_optimized_jit_desc: mdspan_copy_optimized_jit_desc.cu
	$(CXX) -x c++ -std=c++20 -O1 -DLAZY_JIT_DISPATCH $(INCLUDES) -I /usr/local/cuda/include $< -o $@ -ldl

.PHONY: run-optimized run-optimized-jit run-optimized-jit-desc clean
# run: mdspan_copy_transpose
# 	./mdspan_copy_transpose

run-optimized: mdspan_copy_optimized
	./mdspan_copy_optimized

run-optimized-jit: mdspan_copy_optimized_jit
	./mdspan_copy_optimized_jit

run-optimized-jit-desc: mdspan_copy_optimized_jit_desc
	./mdspan_copy_optimized_jit_desc

clean:
	rm -f mdspan_copy_transpose mdspan_copy_optimized mdspan_copy_optimized_jit mdspan_copy_optimized_jit_desc
