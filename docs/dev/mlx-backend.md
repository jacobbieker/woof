# The MLX (Apple GPU) backend

`WOOF_BACKEND=mlx` runs woof's GPU code on an Apple-silicon GPU. The CUDA
sources are not ported by hand: each kernel unit is the same `.cu` text,
translated to Metal Shading Language (MSL) when it is first loaded.
`import cupy` resolves to a CuPy-compatible module over `mlx.core`, so the
Python that launches the kernels runs unchanged. On the `cuda` backend none
of this is installed, and CUDA runs exactly as before.

```
pixi run -e mlx woof doctor          # the backend's probe: translate, compile, run, check words
pixi run -e mlx pytest tests/test_kessler_wrf461_parity.py
```

The pixi `mlx` environment (osx-arm64, macOS ≥ 14) sets `WOOF_BACKEND=mlx`.
A Mac with MLX installed and no CuPy also selects mlx on its own.

## Layout

| Path | What it is |
|---|---|
| `woof/backend/__init__.py` | Backend selection. `install()` puts an import hook in place that serves the facade for `import cupy`. |
| `woof/backend/facade/` | CuPy's API over MLX: arrays (`_array.py`), operations (`_ops.py`), `cupy.cuda` (`cuda.py`), `ElementwiseKernel` and `ReductionKernel` (`_elementwise.py`). |
| `woof/backend/translate/` | The CUDA→MSL translator, which uses libclang. `spaces.py` infers address spaces; `cuda_decls.h` declares CUDA's builtins for the parser; `vendor_libm64.cuh` provides binary64 libm. |
| `woof/backend/metal/prelude.h` | The CUDA intrinsics, atomics and math every translated unit starts with. |
| `woof/backend/metal/f64_soft.h` | Bit-exact IEEE binary64 in integer arithmetic. Apple GPUs have no fp64. |
| `woof/backend/metal/f32s.h` | Binary32 that keeps subnormals, for `--ftz=false` units. |
| `woof/backend/metal/ext/launch.cpp` | An MLX primitive that launches a translated kernel *in place* on its argument buffers. |
| `woof/backend/metal/loader.py` | Turns a unit into a `MetalModule`, and binds arguments with CuPy's calling convention. |
| `woof/backend/host/` | The host tier: `cuda_host_shim.h` builds CUDA sources as host C++, and `libm32` provides glibc's binary32 functions from woof's own ports. |

## Numerics: what each CUDA arithmetic rule becomes

| CUDA | Metal translation |
|---|---|
| `double` | `f64`, soft-float binary64. Every operation is IEEE round-to-nearest, word-exact (`tools/f64_soft_grade.py`). |
| binary32 `+ - * /`, `fma` | Metal's own: IEEE in safe math mode (`tools/metal_arith_receipt.py`). |
| `--fmad=true` / `false` | `#pragma METAL fp contract(fast)` / `(off)`. In strict numerics nothing is contracted, because NVRTC's choices cannot be reproduced. |
| `-ftz=true` (CuPy's default) | Metal flushes on arithmetic. CUDA also flushes in places where Metal keeps the value, and the prelude adds those flushes: the result of binary64→binary32 conversion, and the operands of `fabsf`, negation, `copysignf`, `fminf` and `fmaxf`. |
| `--ftz=false` units | `float` becomes `f32s`: a hardware fast path, with an exact slow path when an operand or result is subnormal. |
| inline PTX without `.ftz` (`add.rn.f32` …) | `__woof_ieee_*`, which keeps subnormals in any unit, as CUDA does. |
| `fminf` / `fmaxf`, and `a > b ? a : b` | −0 orders below +0, as `min.f32` / `max.f32` do. Metal returns the second operand for equal zeros. |
| libdevice binary32 (`expf` …) | Metal's `precise::` functions. These are a recorded vendor-libm difference; units built on woof's glibc ports are exact. |
| binary64 libm | `vendor_libm64.cuh`: correctly rounded exp, log, log10, sin, cos, atan and pow (f64cr); Rust-libm transcriptions for the rest (`tools/vendor_libm64_grade.py`). |

## Launch ordering

CUDA kernels write their argument buffers in place. MLX is functional and never
does, so two orderings that CUDA's stream gives for free have to be built:

* **Earlier readers before an in-place write.** MLX fences one encoder after
  another only when the later one reads the earlier one's outputs. A
  translated launch is therefore its own encoder pass, ended with
  `wait_in_flight`, which waits on every in-flight encoder.
* **Lazy values before an in-place write.** A facade value that is still lazy
  (for example `x.copy()`) is scheduled before the next in-place launch
  (`_array.PENDING`).

`tests/test_metal_launch_ordering.py` guards both. When a result is
nondeterministic, three diagnostic variables help:

- `WOOF_METAL_SERIAL=1` (or `before` / `after`) serialises launches;
- `WOOF_METAL_SERIAL_ONLY=<substrings>` limits that to matching kernels;
- `WOOF_METAL_TRACE=<file>` logs every launch.

## Known Metal compiler behaviour

* **Every device function of a unit is `always_inline`**, except where the
  source says `__noinline__`. This is CUDA's own default, and the only policy
  under which Metal has given CUDA's words in every case tried. The
  alternatives each failed somewhere:
  - left to the compiler's heuristics, Noah-MP's water kernel gave wrong
    values that varied with the threadgroup size and vanished under shader
    validation;
  - with small functions inlined into out-of-line ones, Morrison's
    `mo_process` gave different words on repeated launches of the same
    inputs;
  - with everything out of line, Morrison was right alone but wrong after
    another test file had run in the process.

  The cost is cold compile time on the largest units: CLM lake's step
  kernel alone takes about 5 minutes. `WOOF_INLINE_POLICY=size|none` and
  `WOOF_KEEP_CALL=f,g` select the alternatives, for diagnosis.
  `tools/kernel_probe.py --repeat N` shows whether a kernel's words vary
  across replays.
* **goddard compiles with out-of-line calls** (`INLINE_POLICY_BY_UNIT`).
  Fully inlined, its radiation kernel crashes Metal's compiler service
  (`XPC_ERROR_CONNECTION_INTERRUPTED`). Its device oracle holds the result to
  WRF's words in both builds, on repeated runs.
* **TEMPO compiles with out-of-line calls** too (`INLINE_POLICY_BY_KERNEL`,
  keyed on its `tempo_columns` kernel because a `RawModule`'s unit name is
  a hash). Fully inlined, the column kernel had not finished compiling after
  35 minutes; out of line it takes a few. Given CUDA's init blob, three
  steps of eight columns are CUDA's words in every field and diagnostic.
  The blob itself is built by `rw-tempo` with the platform C library's
  `tgamma`, so on macOS it differs from Linux's in 2772 of 122 million words,
  and TEMPO on a Mac is not CUDA's words with its own blob.
* **Preprocessing is NVRTC's.** The translator defines `__CUDA_ARCH__`
  (860, the comparison machine's) because NVRTC compiles device code only:
  TEMPO's `__CUDA_ARCH__` branch uses woof's glibc ports where its host
  branch calls the C library. The unit's `-I` options are include
  directories, and the binary64 libm is prepended when a header the unit
  includes calls it, not only the unit's own text.
* **A `float` literal argument in an `--ftz=false` unit** becomes `f32s(...)`
  when clang resolved its parameter as `float`. Otherwise `g(f64, float)`
  beside `g(f64, f64)` is ambiguous in Metal, because both need a
  user-defined conversion.
* **Static constexpr members.** MSL is C++14, so an out-of-line call that
  binds a `static constexpr` data member to a `const&` parameter odr-uses
  it. The translator therefore emits the member's out-of-class definition.
* **Pipelines** are built by woof, not MLX, keeping `maxCallStackDepth` at 1.
  A deeper declared depth reserves stack per thread for every level, and
  `fast_sbm` then fails to build: "exceeds available stack space".

## Compile time

A unit's Metal library is built when it is loaded. Each kernel's pipeline
is built at its first `get_function`, and pipelines are most of the cold
cost (Morrison: library 4.7 s, `mo_process` 18 s).

Built pipelines persist as Metal binary archives in
`~/.cache/woof/metal-pipelines`. `WOOF_METAL_PIPELINE_DIR` moves the
directory, and `off` disables the archives. The OS shader cache does not
keep the largest pipelines: CLM lake's step kernel compiled again from
scratch hours after its first build. `WOOF_METAL_COMPILE_LOG=1` prints each
library and pipeline build's time, and whether it came from an archive.

## Address spaces

Metal pointers carry an address space, and CUDA pointers do not. The translator
infers the space of each pointer from its call sites, and emits one copy of a
function per combination of spaces. Two cases need runtime help:

* **A pointer reassigned across spaces and then indexed** (for example
  `zcd = tile` in RRTMG SW) becomes `__woof_vptr<T>`: one pointer per space
  and a tag.
* **A struct pointer field given pointers in two spaces** (goddard's `GArray`
  wraps kernel buffers and local arrays) makes the struct *space-generic*. A
  probe of the analysis finds such fields. The class template then gains an
  `int __WS` parameter, and its field and reference returns become
  `__woof_sptr<T, __WS>` / `__woof_sref<T, __WS>`. A parameter of the struct
  type is a slot like a pointer: one overload per space combination. Each
  local declaration takes its initialiser's space.
* **Program-scope variables.** A mutable `__device__` global is a module
  buffer in `device` memory, and the analysis says so. Only `const` and
  `__constant__` data is `constant`.

## Memory order

Facade arrays carry NumPy strides, and CuPy's memory-order rules hold:
`reshape`/`ravel`/`flatten` with `order="F"` read indices column-major,
and `astype`, `copy(order="K")`, `array(order=...)` and the `*_like`
constructors keep a column-major prototype's layout. Kernels index these
buffers directly, so a dropped layout scrambles fields rather than failing
(SATMEDMF's column packing did). `tests/test_metal_memory_order.py` guards
this. Elementwise results are row-major whatever their operands' layout;
make a column-major kernel argument explicit with `asfortranarray`.

## Threads

CuPy arrays and CUDA's null stream are shared by every host thread. MLX's
streams belong to the thread that made them, so an array computed lazily in
one thread could not be evaluated in another ("There is no Stream(gpu, N) in
current thread"). The facade therefore puts all work on one
`new_thread_unsafe_stream`, made the default in every thread as it starts
(`loader.share_one_stream`). That stream is not thread-safe, and two threads
evaluating at once abort the process, so `mx.eval`, `mx.async_eval` and
`mx.synchronize` take one lock. Threads still build graphs concurrently, but
their GPU work is serialised.

## NumPy protocols and the CUDA runtime surface

* `np.maximum(0, x)`, `np.where(m, x, y)` and other NumPy calls on facade
  arrays dispatch to the facade (`__array_ufunc__`, `__array_function__`),
  as they do to CuPy. Binary ufuncs have `reduce`, `reduceat`, `accumulate`
  and `outer`, which NumPy computes on the host words.
* A same-width view of another kind (`a.view(cupy.uint32)` of float32) reads
  the bits. A boolean mask over several axes indexes as NumPy's `nonzero`.
* `cupy.cuda.runtime.memcpy`/`memcpyAsync`/`memset` work on raw addresses: a
  facade storage's GPU address on the device side, any process address on
  the host side. `alloc_pinned_memory` returns a `hostAlloc` block.
* `cupy.cuda.nvrtc`'s program API and `compile_using_nvrtc` translate and
  build the unit when they are called, so a unit that does not compile fails
  there, as it does under NVRTC. `cupy.cuda.compiler`/`runtime`/`driver`/
  `nvrtc` and `cupy_backends.cuda.{api,libs}` import as modules.

## Mapped host memory

`cudaHostAlloc` returns an evaluated MLX buffer's host address. Apple
memory is unified, so `cudaPointerGetAttributes` gives that buffer's GPU
address as the device pointer. `UnownedMemory` over the address is the
same buffer, as an ndarray of whatever dtype it is viewed with
(`Storage.alias`). Kernels and facade writes change the host bytes in place,
as CUDA's mapped page-locked memory does. `tests/test_metal_mapped_host_memory.py`
guards this.

## Measured-word pins per backend

Many parity tests pin the words a GPU produced, and their distance from
compiled WRF, rather than only asserting bit-identity. CUDA's pins carry
NVRTC's multiply-add contraction, which fast numerics and the oracle tools'
default options leave on. The Metal translation never contracts, so on MLX
the same measurements give CUDA's `--fmad=false` words. Those words are
usually closer to WRF; momentum's `rw_t`, for example, is exact in every
case.

The MLX backend's pins are recorded beside CUDA's:

* `<pin>-metal.json` files, chosen by `woof.verify.platform_pins.pin_path`
  and written by `tools/record_metal_pins.py`;
* rows keyed `metal` in per-compiler tables (`nvrtc_build()` is `"metal"`
  on MLX, and the facade's compute capability is `metal`);
* a `deformation-gpu-metal-receipt.json`, written by the receipt tool
  itself.

Record a family only after showing that its Metal words are CUDA's words
with contraction off. Run `tools/launch_trace.py` on both machines, with
CUDA's compiles forced to `--fmad=false` (`WOOF_FMAD_OFF=1`, applied at
`woof.numerics.effective_options`), and compare the traces with
`tools/backend_receipts.py`. A remaining difference is then a vendor-libm
or NaN-payload delta, and the pin file says so (for example the `known_deltas`
in `vertical-native-metal.json`).

## Finding a difference

1. `tools/launch_trace.py`: run the same scenario on CUDA (`ssh pb`) and on
   MLX, and report the first launch whose array words differ. Each array
   argument records its digest before and after the launch, and an argument
   counts only if both backends gave it the same input words. An
   uninitialised `cupy.empty` buffer holds old pool data on CUDA and zeros
   on MLX, and that is not a divergence.
2. `tools/kernel_probe.py`: replay that launch on the Metal translation and on
   the host C++ build of the same source (with the unit's `-D` options),
   with probe edits, and diff every word. `--grid X,Y,Z --block X,Y,Z`
   gives a 3-D launch.
3. `tools/metal_unit_errors.py UNIT`: show each Metal compile error beside its
   MSL line.
4. `tools/metal_translation_census.py`: which units translate and compile.
5. `tools/cupy_surface_census.py --gate`: which CuPy paths the facade lacks.
