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

* **Out-of-line calls between a unit's own functions** gave wrong values in
  Noah-MP's water kernel. The result depended on the threadgroup size and
  disappeared under shader validation. Every device function of a unit is
  therefore `always_inline`, except where the source says `__noinline__`.
  This costs compile time: CLM lake takes about 20 minutes cold. Metal's
  shader cache makes later builds fast.
* **Pipelines** are built by woof, not MLX, keeping `maxCallStackDepth` at 1.
  A deeper declared depth reserves stack per thread for every level, and
  `fast_sbm` then fails to build: "exceeds available stack space".

## Address spaces

Metal pointers carry an address space, and CUDA pointers do not. The translator
infers the space of each pointer from its call sites, and emits one copy of a
function per combination of spaces. Two cases need runtime help:

* **A pointer reassigned across spaces and then indexed** (for example
  `zcd = tile` in RRTMG SW) becomes `__woof_vptr<T>`: one pointer per space
  and a tag.
* **A struct pointer field given pointers in two spaces** (goddard's `GArray`)
  is refused by name until structs get the same treatment.

## Finding a difference

1. `tools/launch_trace.py`: run the same scenario on CUDA (`ssh pb`) and on
   MLX, and report the first launch whose array words differ.
2. `tools/kernel_probe.py`: replay that launch on the Metal translation and on
   the host C++ build of the same source, with probe edits, and diff every
   word.
3. `tools/metal_unit_errors.py UNIT`: show each Metal compile error beside its
   MSL line.
4. `tools/metal_translation_census.py`: which units translate and compile.
5. `tools/cupy_surface_census.py --gate`: which CuPy paths the facade lacks.
