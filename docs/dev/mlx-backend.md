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
| `double`, strict | `f64`, soft-float binary64. Every operation is IEEE round-to-nearest, word-exact (`tools/f64_soft_grade.py`); division and square root are the same words from `f64_df64.h`'s df64-seeded exact path (`tools/df64_grade.py strict`). |
| `double`, fast | `f64` too, the same binary64 words in memory. `+ - *`, fma, conversions and comparisons are the soft-float words; `/` and `sqrt` are the correctly rounded words from a df64 estimate; CUDA's binary64 libm names are df64, about 2^-46 relative ("Binary64: soft-float and df64" below). |
| binary32 `+ - * /`, `fma` | Metal's own: IEEE in safe math mode (`tools/metal_arith_receipt.py`). |
| `--fmad=true` / `false` | `#pragma METAL fp contract(fast)` / `(off)`. In strict numerics nothing is contracted, because NVRTC's choices cannot be reproduced. |
| `-ftz=true` (CuPy's default) | Metal flushes on arithmetic. CUDA also flushes in places where Metal keeps the value, and the prelude adds those flushes: the result of binary64→binary32 conversion, and the operands of `fabsf`, negation, `copysignf`, `fminf` and `fmaxf`. |
| `--ftz=false` units | `float` becomes `f32s`: a hardware fast path, with an exact slow path when an operand or result is subnormal. |
| inline PTX without `.ftz` (`add.rn.f32` …) | `__woof_ieee_*`, which keeps subnormals in any unit, as CUDA does. |
| `fminf` / `fmaxf`, and `a > b ? a : b` | −0 orders below +0, as `min.f32` / `max.f32` do. Metal returns the second operand for equal zeros. |
| libdevice binary32 (`expf` …) | Metal's `precise::` functions. These are a recorded vendor-libm difference; units built on woof's glibc ports are exact. |
| binary64 libm, strict | `vendor_libm64.cuh`: correctly rounded exp, log, log10, sin, cos, atan and pow (f64cr); Rust-libm transcriptions for the rest (`tools/vendor_libm64_grade.py`). |
| binary64 libm, fast | `f64_df64.h` for exp, expm1, log, log2, log10, log1p, pow, sin, cos, tan, atan, atan2, tanh, hypot and cbrt; `vendor_libm64.cuh` for tgamma, lgamma, erf, erfc (and the binary32 log1pf, cbrtf, tgammaf). woof's own glibc ports (`g64_exp::exp` and kin) are unchanged in both modes. |
| `cupy.power` on binary32, strict | `strict_powf.cuh`: woof's own correctly rounded powf (constants from `tools/strict_powf/constants.py`). CuPy's powf is not correctly rounded: it differs at 273 of the NSSL-2 oracle's 4800 Exner words, the parity ledger's `independent-powf` gap. `WOOF_STRICT_POWF=cuda`, in a source checkout only, selects the opt-in transcription of CUDA's powf (`tools/cuda_powf/`, NOTICE), which gives CUDA's words. |

## Binary64: soft-float and df64

Apple GPUs have no binary64. `double` is always `f64`, the binary64 word in
a `ulong`, so buffers, structs, literals, unions and bit casts hold the
same bytes in either mode; what differs is how its operations compute.

* **Strict numerics** (`GPUWM_WRF_EXACT=1`): every operation gives
  `f64_soft.h`'s word, and the binary64 libm is the vendored correctly
  rounded one. Division and square root take `f64_df64.h`'s exact
  df64-seeded path (below), which gives the soft-float words: the strict
  leg of `tools/df64_grade.py` compares the two on the GPU, NaN payloads
  included, over every special-value pair, soft-float's grading mix and
  quotients landing on gradual underflow and overflow (see "Compile time"
  for the sample size). The df64 libm is left out (`WOOF_DF64_NO_LIBM`).
* **Fast numerics** (the default): `woof/backend/metal/f64_df64.h`, through
  the same translated body with another prelude
  (`translate(..., f64="df64")`, chosen by
  `woof.backend.metal.loader.f64_mode`, so the unit's MSL and every cache
  key built from it differ between the modes):
  - `+ - *`, fma, conversions and comparisons stay the soft-float words. A
    df64 round trip through the binary64 word costs about as much, and it
    would lose the operands' low five bits (df64 carries 48), which breaks
    exact subtraction and the exact-scaling tricks binary64 code relies on
    (`rint`'s 2^52 shift, `ldexp`, Dekker splits);
  - `/` and `sqrt` are the **correctly rounded words**, from a df64
    estimate fixed by the exact integer remainder (`f64_df_div`,
    `f64_df_sqrt`), in place of the soft-float digit loops (63 iterations);
  - exp, expm1, log, log2, log10, log1p, pow, sin, cos, tan, atan, atan2,
    tanh, hypot and cbrt are df64 (`df_*_word`), each with its exponent
    carried apart in an integer, so the whole binary64 range is covered:
    nothing overflows or flushes in binary32, and subnormal arguments and
    results are kept. Arguments are reduced exactly (exp: the word split
    into three binary32 values; sin/cos/tan: Payne-Hanek on the significand
    with 1280 bits of 2/pi). tgamma, lgamma, erf and erfc stay the vendored
    transcriptions (on the exact arithmetic above);
  - woof's own glibc ports (`g64_exp::exp`, `wcr_*` and kin, which the
    WRF word-for-word units call) are unchanged: they run on the exact
    arithmetic, so those units keep their words in fast numerics too
    (CLM lake, MPAS-Seaice EVP at RKIND 8 and NSSL-2's oracle tests pass
    unchanged).

  A unit keeps soft-float in fast numerics when its options carry
  `-DGPUWM_PREPARATION_IDENTITY=1` or it is listed in
  `loader.SOFT_F64_UNITS` (empty); `WOOF_METAL_F64=soft` keeps soft-float
  everywhere, for diagnosis.

**Accuracy.** `tools/df64_grade.py host` (x86-64, x87 `long double`
references, 4 million operands per function over random words, each
function's domain and the special values) gives the largest error in
binary64 ulps of the reference: division and square root 0 (every word
correctly rounded); exp 52, log 81, log2 103, log10 112, log1p 77,
expm1 182, sin 103, cos 108, tan 135, atan 61, atan2 77, tanh 242,
hypot 68, cbrt 54; pow 102 ulps per unit of 1 + |y log x| (its argument to
exp carries df64's relative error, so the error grows with that
magnitude). The leg fails above 512 ulps (2^-44 relative). The df64 core
itself (df64 operands): add 40, mul 79, div 39, sqrt 53 ulps. Special
operands follow C99 Annex F. A result within df64's error of the overflow
threshold saturates to DBL_MAX rather than becoming an infinity df64 alone
could have made. `tools/df64_grade.py gpu` checks that Metal computes the
host build's words for every operation (it does, for all 21).

**Cost.** Cold compile and run times, soft-float against df64, are in
"Compile time" below.

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
* **FAST_SBM inlines its `__noinline__` helpers too**
  (`INLINE_NOINLINE_BY_KERNEL`, keyed on `fsbm_tile_*`). Out of line, its
  tile kernel gave wrong words in 3 of 49 columns of the kc2 replay; fully
  inlined it is CUDA's `--fmad=false` words but for 1-ulp deltas in a few
  liquid-bin words. The other units with `__noinline__` give the same
  results either way and keep it (fully inlined, sfclay compiled for 46
  minutes cold). `WOOF_INLINE_NOINLINE=1|0` forces either, for diagnosis.
* **goddard compiles with out-of-line calls** (`INLINE_POLICY_BY_UNIT`).
  Fully inlined, its radiation kernel crashes Metal's compiler service
  (`XPC_ERROR_CONNECTION_INTERRUPTED`). Its device oracle holds the result to
  WRF's words in both builds, on repeated runs.
* **TEMPO compiles with out-of-line calls** too (`INLINE_POLICY_BY_KERNEL`,
  keyed on its `tempo_columns` kernel because a `RawModule`'s unit name is
  a hash). Fully inlined, the column kernel had not finished compiling after
  35 minutes; out of line it takes a few. Given CUDA's init blob, three
  steps of eight columns are CUDA's words in every field and diagnostic.
  The blob is built by `rw-tempo`, whose reference is gfortran on glibc.
  The platform libm moved 2772 of its 122 million words on macOS: `rw-tempo`
  answers `tgamma` from glibc's own words for every argument it passes
  (`glibc_tgamma.rs`), and the six tables computed with exp/log/pow/atan
  (t_efrw, t_efsw, tpc_wev, tnc_wev, tps_iaus, tni_iaus; the same for every
  init option) are overlaid from `woof/data/tempo/glibc-libm-fields.npz`
  when they differ (`woof.core.tempo_rust`). The Mac's blob is now the
  Linux blob byte for byte (sha256 08fc1095…), which itself is unchanged.
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
* **Float vectors in an `--ftz=false` unit** keep `float` components (Metal
  has no vector of `f32s`). A component read as a value is `f32s(v.x)`, an
  assignment stores `float(e)`, `v.x op= e` goes through `f32s`, and
  `make_floatN`/`{...}` take `float(e)` elements. A `volatile float` value
  loses its qualifier there (f32s arithmetic is calls, never contracted),
  and in a unit with base classes C++'s implicit float→integer conversions
  are spelled as casts before the bases are flattened into templates.
* **A kernel-local `blockIdx`/`gridDim`** (the ensemble batch adapter's
  virtual block coordinates) hides the builtin, as in CUDA.
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

**Soft-float against df64.** `tools/df64_compile_bench.py` builds a unit's
library and every pipeline cold (archives off, a nonce in the MSL so no
cache serves it), in each binary64 mode. On an M3 Max shared with other
jobs (2026-11-08), library plus pipelines, seconds:

| unit | soft-float | df64 | |
|---|---|---|---|
| `sf_gfs` | 114.4 | 0.95 | 120x |
| `gfs_sas` | 55.3 | 1.88 | 29x |
| `nssl2_qvexcess` | 5.18 | 0.10 | 52x |
| `ice_accretion` | 1.95 | 0.15 | 13x |
| `ensemble_stochastic` | 1.85 | 0.15 | 12x |
| `wind_farm` | 1.13 | 0.10 | 11x |
| `fast_sbm` | 402 | 253 | 1.6x |
| `lake` (CLM lake) | 1756 | 312 | 5.6x |

Most of the gain is the binary64 libm: a unit calling CUDA's `exp`/`log`/
`pow`/`sin` in binary64 inlined the soft-float correctly rounded libm at
every call; df64's functions are small and out of line. Units built on
woof's own glibc ports (CLM lake, the MPAS WSM6 unit of the water-budget
gates, the WRF word-for-word units) keep those ports in fast numerics;
for them df64 changes only division and square root, which replace the
soft-float digit loops (CLM lake still compiles 5.6x sooner, and its
words are unchanged). The WSM6 unit was not re-measured.

Strict numerics takes only the exact division and square root (the
soft-float words). Its cold compile, main's strict prelude against this
one, seconds: CLM lake 3087 -> 716 (4.3x); units whose cost is the
vendored libm (`gfs_sas`, `sf_gfs`, `ice_accretion`, `wind_farm`) are
unchanged within noise. The strict leg of `tools/df64_grade.py` compared
the two paths over 2^30 GPU operands: 0 mismatches in either operation.
Strict run time, the kernel below: division 1.7x, square root 5.6x
faster; the libm calls unchanged.

The MPAS WSM6 unit of the water-budget gates (hex, 55 levels, fast
numerics, df64 division and square root, its glibc ports unchanged):
cold compile 224 s, `wsm6_columns` 223 s of it (it had not finished after
1 h 45 min with soft-float division and square root).

Run time, `tools/df64_grade.py bench` (2^20 threads, 64 dependent
operations each) and a translated kernel calling each function 16 times
per thread over 2^20 threads: division 3.6x faster, square root 11x;
exp 19x, log 21x, pow 35x, sin 156x. Addition and multiplication through
df64 would gain nothing (1.0x), and fma 1.7x, which is why those stay the
exact soft-float words.

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
this. `reshape` of a strided view is a view wherever NumPy's would be
(a padded member slab keeps its backing), and `x.data.mem` is the whole
allocation, one object per buffer, as CuPy's is. Elementwise results are row-major whatever their operands' layout;
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

The partial-depth nest's D4 unit (`kernels/partial_nest_d4.cu`) writes
every operation as an explicit round-to-nearest intrinsic, like D3's.
Traced on both machines, its word tests, D3's and the WRF oracle tests agree
launch for launch (`tests/data/receipts/partial-nest-d4/backend-launch-words.txt`).
The envelope receipts per backend are in the same directory.

## The parity ledger

`woof/verify/backend_parity_ledger.json` records each test of a strict sweep
(the WRF 4.6.1 parity and column-exact tests, NSSL-2, the partial-nest
kernel words and the projection tests). For each test it gives the outcome
on both backends and whether the words of every kernel launch agree.
Each open gap names its cause. The causes so far are:

- NaN payloads (`0x7FC00000` on Metal, `0x7FFFFFFF` on CUDA);
- the vendored binary64 `tgamma`;
- the urban kernel's device-address tables;
- a CUDA-only test;
- the projection's host libm (macOS against glibc);
- the independent powf: Metal's correctly rounded strict power against
  CuPy's powf. The `mlx-cuda-powf` receipt, swept with
  `WOOF_STRICT_POWF=cuda`, shows the opt-in closing it.

The per-backend receipts are in `tests/data/backend_receipts`.
`tools/backend_parity_ledger.py` documents the sweep and regenerates both.
`tests/test_backend_parity_ledger.py` holds the ledger to the receipts.
When a gap closes, regenerate the receipts and the ledger, then drop the
gap's cause.

## hex (MPAS) on Metal

The hex lane resolves kernels through `woof.hex.cuda_backend.KernelCache`. Its
admission asks CUDA questions: the runtime version, the compute capability,
NVRTC's architectures and NVRTC's build identity. The facade answers these
with zeros. On the `mlx` backend `woof/hex/metal_lane.py` answers them for
the Metal translation instead, on an Apple GPU (`applegpu_*`; a virtual
Mac's paravirtual device computes wrong answers, so hex refuses there).
`WOOF_HEX_METAL=0` opts out, and then hex refuses on a Mac. The lane:

- stands in for compute 12.0;
- names the translator in its compile fingerprint, never NVRTC;
- needs nothing for the regional unit's direct NVRTC compile: the facade's
  `compile_using_nvrtc` translates it and `function.Module().load` loads
  the translation;
- translates the MPAS_to_physics preparation (`cuda_physics_prep_v841`)
  without contraction in fast numerics too. `validate_prep_v841_f32`
  recomputes the prepared fields and compares words, and Metal's
  `fp contract(fast)` may fuse the producer and the checker differently
  (it did on a real HRRR start, and every fast run refused at step 0);
- runs the dynamics subcycle eager: the forecast attaches no subcycle
  graph, because the facade's stream capture records no CUDA graph and a
  replay would launch nothing.

Each answer is the Metal one only while `cupy` is woof's facade, so a test
that stubs `cupy` gets the hex runtime's own answers.

No kernel source changes.

### What is measured, and what differs

**End to end.** `woof hex regional` on a 1,607-cell generated limited-area
mesh (15 km core, 60 km background, 55 levels), HRRR initial and boundary
states, `arwen_mpas_column` physics, 40 steps of 90 s, on an M3 Max and on
an RTX A6000:

- Fast numerics (the default) runs to the end on Metal. At 1 h Metal and
  CUDA differ by 0.29 K in theta (rms 5.3e-3 K). CUDA against itself with
  `--fmad=false` differs by 0.51 K (rms 5.5e-3 K).
- With `WOOF_FMAD_OFF=1` on both, the initial frame is word-identical except
  `hfx` (567 of 1,607 cells, up to 2.3 W/m²).

**Where the first step parts.** `tools/launch_trace.py` traced the first
step on both machines with contraction off: 1,709 launches, aligned by
kernel name. Each Metal launch was then replayed on CUDA's input words and
compared with CUDA's output words. 1,627 replay word for word, including
every surface-layer, Noah-MP, YSU and dynamics kernel. 29 were not
replayed (device pointer tables, or a validation kernel whose launch order
differs). The 53 that differ:

| kernels | cause |
|---|---|
| `reconstruct_winds_v841_f32` (4) | `sinf`/`cosf` of latitude and longitude: libdevice and `precise::` differ in 444 to 917 of 1,608 words. Each side reproduces its own u and v from its own sines with uncontracted binary32 arithmetic. |
| `prep_interfaces_v841_f32` (5) | `logf`/`expf` of the top-interface extrapolation. Each side reproduces its own words from its own `logf` and `expf`. The hydrostatic integration then carries them down. |
| `mixing_scaling_f32` (4), `recover_pressure_f32` (1) | `powf`, the only non-IEEE operation in each. |
| `bl_ysu_gwdo_v841_f32` (1) | `atan2f` and three `powf`. Taken through binary64 on both machines, the replay differs in 0 words. |
| `momentum_filter_seed`/`lap4_v841_f32` (8) | NaN payload on the padding edge (`dc_edge` = 0): CUDA writes 0x7fffffff, Metal 0x7fc00000. |
| `rp_probe` (1) | RRTMG's signed-zero probe (`max(+0, -0)`), the translator's `__woof_selmax`. The probe is compared on each host, not carried. |
| `pv_apvm_v841_f32` (12), `noahmp_scatter_fields` (1), `solve_region_validate`/`envelope_v841` (16) | The replay itself: aliased outputs, strides baked into a generated source, device-address tables. These launches agree in the real runs. |

So `hfx`'s first difference is vendor binary32 libm in the prepared winds
and pressures. The surface layer reproduces CUDA's words on CUDA's inputs.

**Hex GPU gates** (`components/hex/tools/battery/gpu_gates.txt`,
`-m "not assets"`, `-n 3`), test by test against CUDA (`WOOF_HEX_AOT=off`):
see the PR description of jacobbieker/woof#142 for the table. Every test
that passes on CUDA passes on Metal or skips by name:

- graph-replay tests (`test_cuda_subcycle_graphs*`): the facade captures no
  CUDA graph;
- `test_hex_aot_gpu`: the kernel pack holds NVRTC images;
- `test_moving_check_gpu` (LTS executor): its reserve census adds binary64
  and 64-bit counters atomically, and Apple GPUs have 64-bit atomic
  min/max only;
- `test_stochastic`'s host regressions and `test_partial_nest_kernels_v841`'s
  local-frame test name Linux g++ and NVRTC.

The MPAS WSM6 water-budget gates (`mpas_wb3_v842`, `mpas_wb5_v842`) run
apart: their WSM6 unit carries binary64 (soft-float on Metal) and Thrust's
`complex<double>`. The Kessler and Thompson writebacks and the referee pass
(5 of 10). The WSM6 unit had not finished its cold Metal compile after
1 h 45 min of compiler time on an M3 Max with soft-float division and
square root; with the exact df64-seeded ones it compiles in 224 s ("Compile
time"), and its 5 tests have not been re-run.

The partial-depth band (Phase D5) keeps its own measurements:

- `components/hex/tests/test_partial_nest_oracle_v841.py`: the device
  operators are bitwise to compiled MPAS v8.4.1 on Metal;
- the envelope receipt `tests/data/receipts/partial-nest-d5/envelope-mlx.json`.

### Translation

`tools/metal_translation_census.py --hex --translate-only` translates every
unit the hex kernel pack holds (`woof.hex.cuda_backend.aot_build.pack_inventory`:
the KernelCache declarations and the recorded direct and raw NVRTC compiles;
CuPy's generated kernels are the facade's). The inventory's sources and
options come from both numerics modes, but the census translates every row
under the process's own numerics: fast by default, where a strict row's
`--fmad=false` is dropped as a fast compile drops it (rows marked
`-DGPUWM_PREPARATION_IDENTITY=1` keep it); `--retain` translates as
`GPUWM_WRF_EXACT=1`, strict. Translation
needs no Metal device and no MLX: only libclang's Python bindings and a
`clang` or `clang++` on `PATH` (the preprocessor). On Linux the `mlx` pixi
environment does not exist (it is osx-arm64 only), so put the bindings of
the `libclang` 18.1.1 wheel the `mlx` environment pins on `PYTHONPATH` in
the `default` environment:

```
pip download libclang==18.1.1 --no-deps --platform manylinux2010_x86_64 -d /tmp/lc
python -m zipfile -e /tmp/lc/libclang-18.1.1-*.whl /tmp/lc/site
PYTHONPATH=/tmp/lc/site/libclang-18.1.1.data/platlib pixi run -e default \
    python tools/metal_translation_census.py --hex --translate-only --out /tmp/hex.jsonl
```

With `--translate-only` the census leaves `WOOF_BACKEND` as the host has it.
353 of the 361 units translate. The rest are refused by name:

- RRFS smoke (4 units) includes `<cuda/barrier>` for a block-scope
  `cuda::barrier`;
- RRFS NSSL (2) aggregate-initialises a derived struct through its base
  (`NsslColumn column{{}, arena}`) and slices it back to the base, which
  the flattened struct cannot take;
- Grell-Freitas (2) has two retry loops that cross and share a label.

The hex units brought in these translator constructs:

- crossing forward gotos nest;
- a forward goto into a retry loop guards the loop's head on the first pass;
- PTX bit operations are translated (`xor`/`and`/`mov`, `setp`/`selp` without
  `.ftz`);
- a string literal is `constant` data;
- `static const` arrays are recognised as constant;
- `__ldg`, `__maxnreg__`, `clock64` and `__reduce_*_sync` are provided;
- a `cupy/complex.cuh` stand-in carries Thrust's division, `csqrtf`, and
  binary64 `log` (FreeBSD's `clog`), `sqrt` and `abs`;
- a type trait's explicit specializations keep their static members folded;
- a struct field given a string literal anywhere is `constant` data, also
  when a free function assigns it and passes it on (WSM6's `last_message`);
- `static_cast<volatile T&>(x)` read as a value is `__woof_vload(x)`, which
  finds x's address space by overload;
- CUDA's `__expf`, `__logf` and `__fdividef` are Metal's `fast::`
  functions.

`tests/test_metal_hex_translation.py` checks on the device the gotos, the
PTX bit operations, string literals in a struct, the Thrust complex
functions and the WSM6 and RRFS shapes above. The `static const`
arrays, `__ldg`, `__maxnreg__`, `clock64` and `__reduce_*_sync` are
checked only by the census, which translates the MYNN, TEMPO and RRFS
units that use them.

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
4. `tools/metal_translation_census.py`: which units translate and compile
   (`--hex`: the hex lane's units). Cached
   results are keyed on a fingerprint of the translator, the Metal headers
   and the kernel sources, so a stale result is never reported.
5. `tools/cupy_surface_census.py --gate`: which CuPy paths the facade lacks.
