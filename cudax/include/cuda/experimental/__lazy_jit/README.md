# Enable nvrtc (device only)-firendly jit 

* Examples are in `cudax/examples/mdspan_copy*.cu`

* The host (dispatching logic) can be precompiled, but we want to
  defer device code compilation to user/consumer JIT (via nvrtc).
  So, when calling a host function that would normally end with lanching
  a cuda kernel, the function returns "a callback for eval" back to the caller.
  Caller jits and runs the callback next.

* This proof of concept:
  * adds costum device() distpach call that replaces ::cuda::launch
  * depending on LAZY_JIT_DISPATCH flag presence it either is: 
    * a regular ::cuda::launch call, or 
    * "a lazy jit dispatch" call
  * note 1: for ease of templates manipulations, the kernel implementations
    are changed from regular template<Config, Args...> __global__ fn into
    functor scheme (supported/proposed by cuda::launch already), i.e.
    template <Args...> struct kernel {template <Config> __device__ operator()()}

  * The lazy jit dispatch, instead of lauching the provided kernel,
    returns: 
      * string with includes
      * `Config` and `Args` types as compile time static strings
      * unique_ptr<void> opaque bundle of kernel arguments
    if the includes and types are fed into nvrtc, it will compile a 
    `template<Tuple> __global__(Tuple arg_tuple) kernel` that
    calls the respective kernel functor with args unpacked from the tuple.
  * note 2: we stringify Args exactly as in the original kernel/functor template list
    (and not full types of the operator() arguments) and then infer the full arg types
    with `operator_args_t`. The point is just that resulting string is shorter
    (for argument of type cuda::something<T> we need to only print T not the full arg type).
  * note 3: we need to unpack from the Config (as in make_config) the underlying _Hierarchy
    and pass that to nvrtc - the whole Config definition does not exists in nvrtc.
    Similarily, we need to extract block/grid info + shared memory info from the config on the
    host so that the caller knows how to run the jitted kernel.
  * note 4: because the dispatch happens at every call, it's best if the source code
    string building can happen in host-compile time so that we just return a pointer
    (BONUS: equality of strings from pointers?). Thus __fixed_string helper.


## Challenges
* Type stringification - in this POC we painstakingly specialize repr_type<>
  emitter for every possible kernel argument type.
  Alternatives? PRETTY_FUNCTION macro, typedid + demnagling?
  -> Are those precise enough to get representation that compiles
  -> Can they be compile time
* CUDA dependencies.
  Ideally, it should be possible to precompile host code with plain gcc
  and no cuda-specific dependencies (so that the pre-compiled code
  is truly CTK agnostic, i.e. doesn't need -cu12, -cu13 variants).
  * TODO: this precludes/makes it tricky to e.g. query device attributes,
    in this POC I simply pessimized attributes query (assumed the smallest shm size etc)
  * Includes are still messy. Currently, at least -O1 is required for cudart symbols to be gone
    from a binary that precompiles copy(mdspan) in lazy jit mode.
* Precompiling for ndim x data types is slow - so much that I don't think it
  would be acceptable, I guess we'd need to limit branching above some ndim
  (say 8) or keep nvmath's copy as a fallback above that threshold.
* Binary size can still become an issue:
  Crucially when building Python bindings that use lazy jit dispatch, the `-g` flag must be overriden, otherwise binary blows up with debugging symbols over 10x.
  Even without it, the result of building copy_kernel_v2.pyx (64 ndims x 5 itemsizes) is 8x bigger than the baseline copy_kernel.pyx (maybe it's fine and reasonable - mdspan copy dispatch
  has just more if/else runtime selections, so more specializations per itemsize, ndim pair).
* cudax/include is not shipped in wheels, does nvmath need cudax copy/submodule.
  Would we need to ship them all (or just some small subset like __copy dir would be enough)?
