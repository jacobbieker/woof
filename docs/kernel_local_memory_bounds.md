# Two compile-time array bounds, 9.4 GiB of device memory

`woof/core/kernels/kf.cu` declared ~47 per-thread column arrays at
`KF_KMAX = 128`, and `woof/core/kernels/refl.cu` another ~14 at
`REFL_KMAX = 256`. The cases this project runs have `nz = 49`. Neither
bound is read as a value anywhere in either kernel (every loop runs to the
runtime `nz`) so both were pure allocation size, 2.6x and 5.2x larger than
the vertical grid in use.

On this card that is not a style question. When a kernel's per-thread local
frame exceeds the context's default stack limit, the driver answers the
kernel's **first launch** by allocating a backing store for the device's
whole resident-thread capacity (not for the launched grid) and holds it
for the life of the process. One allocation serves the process, sized by
the largest frame launched so far. The CuPy pool never sees it.

```
reservation_bytes = (max_local_size_bytes - default_stack_limit_bytes)
                    * max_threads_per_multiprocessor * multiprocessor_count
                  = (frame - 1024) * 1536 * 170          [RTX 5090, 610.74]
```

Both bounds are now `#ifndef`-guarded and both launchers compile them to the
field's own level count through `woof.core.kernels.get_kernel_int_defines`.
This document is the measurement record.

---

## 1. The frames, and what the driver reserved for them

Every row measured on the run host: a fresh process, a bare CUDA context,
one small launch of one kernel symbol bracketed by `cudaMemGetInfo` with the
CuPy pool's own delta subtracted. The frame is
`CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES` read back off the driver.

| kernel | bound | frame/thread | **measured reservation** | law |
|---|---|---|---|---|
| `kf_column` | `KF_KMAX 128` | 24,064 B | **5,738.0 MiB** | 5,737.5 |
| `kf_column` | `KF_KMAX 49` | 9,216 B | **2,040.0 MiB** | 2,040.0 |
| `kf_column` | `KF_KMAX 30` | 5,640 B | **1,152.0 MiB** | 1,149.5 |
| `refl10cm_morrison_column` | `REFL_KMAX 256` | 18,432 B | **4,334.0 MiB** | 4,335.0 |
| `refl10cm_morrison_column` | `REFL_KMAX 49` | 3,528 B | **626.0 MiB** | 623.6 |
| `refl10cm_morrison_column` | `REFL_KMAX 30` | 2,160 B | **286.0 MiB** | 282.9 |
| `refl10cm_thompson_column` | `REFL_KMAX 256` | 16,128 B | **3,760.0 MiB** | 3,761.2 |
| `refl10cm_thompson_column` | `REFL_KMAX 49` | 3,088 B | **514.0 MiB** | 514.0 |
| `refl10cm_wsm6_column` | `REFL_KMAX 256` | 14,080 B | **3,250.0 MiB** | 3,251.2 |
| `refl10cm_wsm6_column` | `REFL_KMAX 49` | 2,696 B | **420.0 MiB** | 416.4 |
| `refl10cm_kessler_cell` | either | 0 B | 0 | 0 |

Ten measurements, worst disagreement with the law 3.6 MiB (0.6%). The frame
itself is linear in the bound and rounds up to the local frame's 8-byte
granularity:

| module | bytes/level | check |
|---|---|---|
| `kf` | 188 | 188 x 128 = 24,064; `align8(188 x 49)` = 9,216 |
| `refl` (Morrison) | 72 | 72 x 256 = 18,432; 72 x 49 = 3,528 |
| `refl` (Thompson) | 63 | 63 x 256 = 16,128; `align8(63 x 49)` = 3,088 |
| `refl` (WSM6) | 55 | 55 x 256 = 14,080; `align8(55 x 49)` = 2,696 |

**Saved at `nz = 49`: 3,698 MiB on `kf_column`, 3,708 MiB on
`refl10cm_morrison_column`.** They are not additive (the reservation is a
maximum over launched kernels, not a sum) but they were the two largest
frames in the tree by a factor of three, so between them they set it.

> `kf_column`'s remaining 840 MiB was removed on 2026-08-21 by taking its
> column arrays off the stack entirely; see §9. The 188 B/level row above
> describes the source as it stood before that.

## 2. No bit moved

The bound is an allocation size. Nothing reads it as a value, and the
highest index any loop in either file forms is `nz - 1` (the reflectivity
melting scan reads `k + 1` from `k <= nz - 2`; KF's downdraft recursions
read `nd + 1` from `nd <= lfs - 1 <= nz - 2`). Two independent checks were
run rather than resting on that reading.

**Before/after digests through the production launchers.** SHA-256 over the
raw output bytes of `woof.core.kf.launch_kf` (the five-column parity batch:
synthetic unstable, the two extracted 12Z soundings, shallow, guarded, all
four `KFPhaseMode` values, 17 output arrays each) and of all four
`launch_refl10cm_*` entry points (the mixed-regime stress batch: warm rain
low, snow/graupel aloft, melting layers, empty columns, Morrison at both
`morr_rimed_ice`, WSM6 at both `hail_opt`, Thompson, Kessler). Captured at
`92216d0` with the unspecialized bounds, and again after the change with the
launchers taking the specialized path.

**74 output digests, plus the 5 recorded driver frames and 2 level counts:
81 entries compared, 0 differing.**

**Side-by-side in one process**, as a permanent gate:
`tests/test_kernel_local_bounds.py` launches the specialized kernel and the
unspecialized module (no defines injected, i.e. exactly the binary that ran
before this change) on the same device buffers and compares raw bytes. 68
KF arrays over four phase modes, and every `refl.cu` kernel over every
scheme option. A stray index past `nz` would be absorbed by the slack at the
old bound and would corrupt a neighbouring column array at the new one, so
that comparison is also the out-of-bounds check.

The same file re-measures every frame in §1 against the driver, so a
compiler or source change that breaks the linear model fails there instead
of silently mispricing a rail gate.

## 3. Preflight

`woof/core/preflight.py` priced the local-memory reservation from a single
per-module constant. Two of those modules no longer launch at the bound they
compile to by default, so the module table is now explicitly the **ceiling**
and `LEVEL_SPECIALIZED_KERNEL_FRAMES` carries the per-level model:

```python
LEVEL_SPECIALIZED_KERNEL_FRAMES = {
    "kf":   LevelSpecializedFrame("kf",   "KF_KMAX",   128, 188),
    "refl": LevelSpecializedFrame("refl", "REFL_KMAX", 256,  72),
}
```

> **Superseded for `kf` by §9 (2026-08-21).** Its column arrays left the
> stack for a global workspace, so its frame is a flat 512 B, it no longer
> follows `nz`, and its row is gone from this table. `refl` is unchanged.
> The 188 B/level model above is still what the historical runs in §4 are
> priced against; `tests/test_preflight.py` keeps it as
> `KF_AS_BUILT_FRAME` for exactly that.

An import-time assertion ties each row's unspecialized frame to the
driver-measured `KERNEL_MAX_LOCAL_SIZE_BYTES` entry, so the two tables
cannot drift apart. `kernel_local_frame_bytes(exp)` prices every launched
module, taking `kf`/`refl` at the deepest domain that launches them;
`kernel_local_memory_bytes` is the maximum over that, as before.

`KERNEL_MAX_LOCAL_SIZE_BYTES` was regenerated against the driver after the
change (`tests/test_preflight.py::test_the_recorded_local_frames_match_the_driver`,
plus an independent dump over all 51 modules): **unchanged, every row**,
including the three that still fail NVRTC. The `#ifndef` guards move
nothing when nothing overrides them.

Two historical measurements are preserved rather than re-pointed. The two
four-domain runs of 2026-07-26 were made by the as-built binary at the
unspecialized bounds, so `_as_built_overhead` in the tests prices them at
the frames those runs actually compiled to; the bracketing test still holds
them to within 4%.

`woof check configs/real74_4dom_mynn_norad.toml --rail-mib 29500`:

| | before | after |
|---|---|---|
| local-memory backing store | 5,738 MiB (5.60 GiB) | **1,990 MiB (1.99 GiB)** |
| reserve | 7.12 GiB (0.60 + 6.02 + 0.50) | **3.52 GiB (0.60 + 2.41 + 0.50)** |
| estimate vs budget | 20.13 vs 19.16 GiB | **20.13 vs 22.76 GiB** |
| `alloc_estimate_le_wddm_budget` | **FAIL** | **PASS** |

The projection tracks the fix in the other direction too: the same check on
the 60 s-history variant selects **24** kernel modules instead of 23
(`refl` is now reachable and priced) and the local-memory term does not
move. That is the time bomb, gone from the estimate as well as from the
device.

## 4. The runs

Five forecasts, all `status: complete`, all through the supervised
`python -m woof.cli run` path. Device-wide VRAM sampled at 1 Hz by
`nvidia-smi` from outside the process: the whole-machine rail is the bar,
so the process's own view is not the measurement.

| | run A | run B | run C | run D | run E |
|---|---|---|---|---|---|
| config | `real74_4dom_mynn_norad.toml` | + 60 s history | + 60 s history | + 60 s history | `real74_3dom_mynn_norad.toml` |
| domains / columns | 4 / 861,001 | 4 | 4 | 4 | 3 / 501,001 |
| Kain-Fritsch | **on** (d01) | **on** | **on** | **on** | **on** |
| `refl10cm_*` launched | no | **yes, all 4** | **yes** | **yes** | no |
| in-worker instrument | - | - | yes | yes | yes |
| **device-wide peak** | **27,216 MiB** | **27,277 MiB** | **27,317 MiB** | **26,740 MiB** | **19,574 MiB** |
| card before the run | 2,679 MiB | 2,541 | 2,596 | 2,595 | 2,647 |
| **under the 29,500 rail** | **2,284 MiB** | **2,223** | **2,183** | **2,760** | **9,926** |
| wall | 123 s | 118 s | 124 s | 119 s | 69 s |

**Four domains with Kain-Fritsch fits.** Run A is
`configs/real74_4dom_mynn_norad.toml` with nothing changed. The same
configuration measured **31,130 MiB** on 2026-07-26 (1,630 MiB *over* the
rail) and `configs/real74_4dom_mynn_norad_nocu.toml` exists only because of
that. It no longer has to.

### `refl10cm_*` was launched, and it cost nothing

The one change in runs B-D is `history_interval_s = 60.0` on every domain
(a whole number of each domain's steps), so the 60 s forecast reaches a
history frame and the microphysics drivers' `refl_10cm_due` branch fires.
Both traced probes behind the original diagnosis wrote their t=0 frames
against a 900 s interval and never launched a reflectivity kernel.

The output is the proof. `woof/runtime.py:1375` emits `REFL_10CM` only for
`ticks != 0`, and only by consuming the stash the kernel wrote:

```
wrfout_d01_1974-04-03_12_00_00   REFL_10CM ABSENT   (86 variables)
wrfout_d01_1974-04-03_12_01_00   REFL_10CM present  (1,49,200,250) float32
                                 min -35.000  max 32.437 dBZ  all finite
wrfout_d04_1974-04-03_12_00_00   REFL_10CM ABSENT   (85 variables)
wrfout_d04_1974-04-03_12_01_00   REFL_10CM present  (1,49,600,600) float32
                                 min -35.000  max -3.659 dBZ  all finite
```

The in-worker instrument caught its first launch:
`refl10cm_morrison_column`, `local_size_bytes = 3528`, **non-pool step
0.0 MiB**. Mid-flight, at the first history frame of a four-domain
forecast, the reservation does not move. As built it would have gone from
64 MiB to 4,335 MiB at that instant, past the gate that let the run start.

### The in-run reservation, same instrument as the diagnosis

Run E is the *same three-domain configuration* whose as-built trace measured
`kf_column`'s first launch at **5,738.0 MiB**, non-pool, in-process:

| | as-built (2026-07-26) | specialized (run E) |
|---|---|---|
| `kf_column` frame | 24,064 B | **9,216 B** |
| first-launch non-pool step | **5,738.0 MiB** | **2,040.0 MiB** |
| non-pool at the device peak | 6,573 MiB | **4,052 MiB** |
| device-wide peak | 23,799 MiB | **19,574 MiB** |

**3,698.0 MiB returned, in a real forecast, measured by the instrument that
found the problem.**

### One thing measured and not explained

At *four* domains the same bracket reports **3,645.9 MiB** (run D) and
**3,819.5 MiB** (run C) across `kf_column`'s first launch, for the same
9,216 B frame that steps exactly 2,040.0 MiB at three domains and in every
fresh-process probe. It reproduces, so it is not another process on the
card. Two candidate explanations were tested and both are dead:

- **Grid dependence.** A fresh process launching `kf_column` at 1, 1,563,
  8,160 and 16,320 blocks reserves 2,040.0 MiB every time; at
  `KF_KMAX = 128` and 1,563 blocks it reserves 5,738.0 MiB. The reservation
  does not see the grid.
- **Sized by the widest LOADED frame rather than the widest launched one.**
  3,606 MiB is what `nssl2`'s 15,504 B frame would reserve, and the run
  compiles that module without launching it. Resolving every `nssl2` symbol
  in a fresh process and then launching `kf_column` still steps 2,040.0 MiB.
  Preflight's "maximum over launched kernels" survives.

Whatever the four-domain residual is, it is not this array bound: the
identical kernel at the identical frame reserves exactly the law's 2,040 MiB
at three domains, in seven fresh-process probes, and its own driver-reported
frame inside the four-domain worker is 9,216 B. It belongs to the in-run
non-pool accounting at the four-domain shape and is recorded here for
whoever owns that model. The rail measurement is unaffected: run A is
device-wide NVML, taken from outside the process, and it is 2,284 MiB under.


## 5. What this does not do

- **The bound is still a ceiling, not a clamp.** `nz` past 128 (KF; 256
  since 2026-10-12, §9) or 256 (reflectivity) raises, as before. Specializing downward must never turn a
  refusal into a silent truncation.
- **One extra NVRTC compilation per distinct `nz`.** Configurations whose
  domains differ in level count compile `kf.cu`/`refl.cu` once per level
  count. CuPy's on-disk cache absorbs this across runs; the reservation is
  still one allocation, sized by the deepest.
- **Nothing else in the tree was specialized.** `ysu` (9,232 B) is now the
  widest frame in the four-domain YSU reference configuration, ahead of
  KF's specialized 9,216 B; `thompson` (11,264 B) and `nssl2` (15,504 B) are
  wider still where they are selected. Their bounds were not examined here.
- **The 3% pool-retention term and the 432 MiB context term are unchanged**
  and carry their existing provenance.
- **The four-domain in-run bracket residual of §4 is open**, and belongs to
  whoever owns the non-pool model rather than to this change.

## 6. Reproducing

```
# the bounds, the frames, and that no bit moves (device)
pytest tests/test_kernel_local_bounds.py
pytest tests/test_preflight.py::test_the_recorded_local_frames_match_the_driver

# the pricing model and the gates (CPU)
GPUWM_NO_LOCAL_GPU=1 pytest tests/test_preflight.py

# the gate that used to refuse the cumulus configuration
GPUWM_NO_LOCAL_GPU=1 python -m woof.cli check \
    configs/real74_4dom_mynn_norad.toml --rail-mib 29500     # exit 0

# run A: four domains with Kain-Fritsch, nothing else changed
python -m woof.cli run configs/real74_4dom_mynn_norad.toml --outdir OUT
```

Runs B-E used a copy of that config with `history_interval_s = 60.0` on
every domain (runs B-D) and `real74_3dom_mynn_norad.toml` (run E). Device-
wide VRAM was sampled at 1 Hz with
`nvidia-smi --query-gpu=memory.used`, outside the process; runs C-E
additionally put a `sitecustomize.py` on `PYTHONPATH` so the supervisor's
worker subprocess carried the launch-bracket instrument.

---

## 7. Grell-Freitas: the frame that could not be specialized away

`kf` and `refl` above were fixed by making the compile-time bound follow
`nz`. `woof/core/kernels/gf.cu` could not be fixed that way: its arrays
already sized to the level count, and there were 114 of them. One thread
owns one whole GFDRV column, so the deep and shallow column routines'
working set IS the frame -- 22,416 B at the shipped `GF_KMAX = 40` tier,
rising 456 B per level.

The arrays therefore moved OFF the stack, into a global workspace the
launcher sizes to the columns it keeps IN FLIGHT. That ratio is the whole
saving. The driver charges the local frame at
`multiProcessorCount x maxThreadsPerMultiProcessor`; the kernel only ever
had `70 x 384` threads resident on this card, a 4.0x gap that the
reservation paid for and nothing used.

### The reservation, measured

Fresh process per row, zero-work launch (`n = 0`), `cudaMemGetInfo` either
side. **a development machine, NVIDIA GeForce RTX 5070 Ti, 70 SMs x 1,536, sm_120,
NVRTC 13.3, driver 610.43.02**, card verified idle.

| tier | frame before | reserved before | frame after | reserved after | free after, before -> after |
|---|---|---|---|---|---|
| `nz <= 40` | 22,416 B | **2,200.0 MiB** | 72 B | **4.0 MiB** | 13,214.1 -> 15,410.1 MiB |
| `nz = 49` | 26,520 B | **2,622.0 MiB** | 72 B | **6.0 MiB** | 12,792.1 -> 15,408.1 MiB |
| `nz = 55` | 29,264 B | **2,902.0 MiB** | 72 B | **6.0 MiB** | 12,512.1 -> 15,408.1 MiB |
| `nz = 64` | 33,360 B | **3,322.0 MiB** | 72 B | **6.0 MiB** | 12,092.1 -> 15,408.1 MiB |

The 4-6 MiB the cut arm still steps is module load, not a backing store:
72 B is under the 1,024 B default stack, so the law's reservation term is
exactly zero. The frame also stopped moving with `nz`, which retires the
level-specialization gap this module used to carry -- the row is the same
72 B at every tier above.

### What the workspace costs, and why the tile was 4 blocks/SM

The workspace is real device memory, so the cut is a trade and the tile is
where the trade is set. 100,000 columns at `nz = 40`, same process, same
inputs (the 216-column WRF v4.6.1 oracle fixture tiled up), median of 7:

| arm | tile | wall | workspace | reservation | device total |
|---|---|---|---|---|---|
| pre-cut | -- | 19.35 ms | 0 | 2,193.5 MiB | 2,193.5 MiB |
| cut, 2 blocks/SM | 8,960 | 31.03 ms | 211.0 MiB | 0 | 211.0 MiB |
| **cut, 4 blocks/SM** | **17,920** | **24.18 ms** | **422.1 MiB** | **0** | **422.1 MiB** |
| cut, 6 blocks/SM | 26,880 | 24.27 ms | 633.1 MiB | 0 | 633.1 MiB |
| cut, 8 blocks/SM | 35,840 | 25.16 ms | 844.1 MiB | 0 | 844.1 MiB |
| cut, 12 blocks/SM | 53,760 | 26.63 ms | 1,266.2 MiB | 0 | 1,266.2 MiB |

4 blocks/SM is the plateau and the cheapest point on it. The kernel's
hardware occupancy at block 64 is 12 blocks/SM, so this is a throughput
choice; `woof/core/gf.py` takes the smaller of the two.

Since 2026-09-30 the driver carries `__launch_bounds__(64, 8)` (168 to 128
registers, the same 88 B frame) and the tile is 8 blocks/SM: with six
blocks resident the sweep above had no seventh or eighth to fill. On an
82-SM RTX PRO 4500 at 50,000 columns and `nz = 50` a GF call took 10.85 ms
against 11.53 ms at the old bound and tile, and the same at 16,384 columns
(3.92 ms both); every output word was unchanged. The workspace doubles
with the tile, and preflight prices it through the same constant.

The residual 1.25x on the kernel is the price of global addressing over
local. `__restrict__` on the column views does not close it (24.12 ms
against 24.18, inside the noise) and is not shipped.

### The layout is not a detail

A first cut gave every thread a CONTIGUOUS slab. Same frames, same bytes,
and it ran 2.18x to 5.86x slower -- 42.13 ms at the smallest tile and
113.28 ms at the largest, i.e. WORSE with more threads, which is the
signature of 32-way scatter rather than of arithmetic. CUDA interleaves
local memory across a warp, so the workspace has to be interleaved the
same way: element `k` of slot `s` for lane `t` lives at
`block_base + (s * GF_KP + k) * GFWS_LANES + t`, and a warp reading
`arr[k]` touches 32 consecutive floats. That change alone took 5.86x to
1.25x.

### The capture slabs, and why a null sink beats a clever compiler

`gf_gfdrv_stage` captures nothing, and used to pass the column routines a
local `[GF_NLEV * GF_KMAX]` dummy for the per-stage capture -- 13,280 B at
`nz = 40` that survived only because the compiler dead-store-eliminated it.
It does not always. MEASURED on the same source, same box, same card:
NVRTC 13.3 eliminated it (72 B frame) and NVRTC 13.0.48 did not
(**13,824 B**, a 1,252 MiB reservation). Absence is now spelled as a null
sink whose writes compile away by construction, and both compilers agree:
72 B on 13.3, 88 B on 13.0.48.

### Bit-identity

`gf.cu` does all its arithmetic through `__fadd_rn` / `__fsub_rn` /
`__fmul_rn` / `__fdiv_rn` / `__fsqrt_rn` and their double siblings -- there
is not one raw `*` between floats in the file -- so ptxas has no
contraction or reassociation decision to make, and moving declarations
cannot move a bit by that route. Measured rather than argued:

* the three byte-frozen oracle suites (`tests/test_gf_deep_cuda.py`,
  `tests/test_gf_shallow_cuda.py`, `tests/test_gf_gfdrv_cuda.py`), 366
  assertions at max_ulp 0 against the WRF v4.6.1 capture, green;
* an interleaved A/B in ONE process, pre-cut and cut arms alternating over
  four rounds on the same device buffers: **141,480 graded words, 0
  differ**, all eight round digests equal;
* the same A/B at a 64-column tile, so the workspace is REUSED across four
  sequential launches: 0 words differ;
* perturbed-input control (one T word, one ULP): **217 words move**;
* perturbed-kernel control (`ENTR_BASE` in the `__constant__` table, one
  ULP): **6,324 words move**.

Both controls fire, so the null result is a measurement and not a
harness that never ran.

### Reproducing

```
pytest tests/test_gf_workspace.py
pytest tests/test_gf_deep_cuda.py tests/test_gf_shallow_cuda.py        tests/test_gf_gfdrv_cuda.py
```

`tests/test_gf_workspace.py::test_the_gf_frame_stays_under_the_default_stack`
is the platform-independent gate: it asserts the property that matters --
frame at or under the 1,024 B default stack, so the reservation is zero --
on whatever box runs it, including boxes with no recorded frame row.

---

## 8. YSU: the frame every default run was paying for

Sections 1 and 7 cut kernels a user has to *select*. This one is different:
`bl_pbl_physics = 1` is the wizard's default (`woof/domain_wizard.py:714`),
YSU is in the shipped HRRR and GFS profiles, and it ran on all four domains
of the deep nest tree and on the 3 km CONUS run. Its frame was the widest a
**bare default run** launched, so every default run paid it.

All measurements below: **a development machine, RTX 5070 Ti, 70 SM x 1,536
threads/SM, sm_120**, CuPy 14.0.1 (NVRTC 13.0) and 14.2.0 (NVRTC 13.3),
2026-08-21. Fresh process per arm, real launcher, `memGetInfo` bracket.

### The instrument, validated first

Against the two known controls, before any YSU number was taken:

| frame/thread | reserved | law `(frame-1024) x 70 x 1536` |
|---|---|---|
| 0 B | **0.0 MiB** | 0.0 |
| 16,384 B | **1,574.0 MiB** | 1,575.0 |

### The reservation is taken at LAUNCH, not at module load

A two-kernel module was compiled holding a 0 B kernel and a 16,384 B
kernel. Compiling it reserved **0.0 MiB**. Launching the small kernel
reserved **0.0 MiB**. Only launching the big one took the 1,574.0 MiB.

This is why the ceiling belongs to the kernels a configuration *runs*.
`thompson`'s row is 11,264 B, but that is its `KMAX = 256` template, and a
run with `nz <= 64` dispatches only the `_64` variants: measured 2,816 B.
`preflight` prices the row, which is the safe direction and is not what the
driver charges.

### Before and after

| | frame | reserved | wall @ 102,400 col |
|---|---|---|---|
| pre-cut | 9,232 B | **842.0 MiB** | 3.91 ms |
| cut | **0 B** (both NVRTC 13.0 and 13.3) | **none** | 3.98 ms |

The workspace that replaces it is **123.0 MiB** at `nz = 49`, and unlike the
frame it follows `nz`: the extent is a runtime argument, so a 49-level run
holds 50 levels of arrays where the frame had to hold 128 whatever the grid.

### Why the tile is 16 blocks/SM

Measured at 102,400 columns, `nz = 49`, block 32:

| blocks/SM | tile | workspace | wall |
|---|---|---|---|
| 2 | 4,480 | 15.4 MiB | 9.57 ms |
| 4 | 8,960 | 30.8 MiB | 6.34 ms |
| 8 | 17,920 | 61.5 MiB | 4.62 ms |
| 12 | 26,880 | 92.3 MiB | 4.14 ms |
| **16** | **35,840** | **123.0 MiB** | **3.98 ms** |
| 24 | 53,760 | 184.6 MiB | 4.03 ms |
| 32 | 71,680 | 246.1 MiB | 3.87 ms |

The kernel's hardware occupancy at block 32 is **16 blocks/SM**, so 16 is
both the plateau and the point where the tile exactly fills the card;
`ysu_tile_columns` queries occupancy and only ever lowers it. Rows past 16
buy nothing (they cannot be resident) and cost workspace linearly.

### The floor: what a bare default run actually gains, and why

The pool is one per-context store sized by the widest frame the process
LAUNCHES, so a cut is worth nothing until it clears the next kernel down.
Measured in one process, the real launchers, in the order a default run
reaches them (`bl_pbl_physics = 1`, `cu_physics = 1`, `nz = 49`, a development machine):

| | `ysu_column` takes | then `kf_column` takes | context total |
|---|---|---|---|
| pre-cut | 844.0 MiB | **0.0 MiB** | **844.0 MiB** |
| YSU cut | 2.0 MiB | **840.0 MiB** | **842.0 MiB** |

Pre-cut, KF costs *nothing extra*: its 9,216 B frame fits inside a store
already sized for YSU's 9,232 B. Cut YSU and KF inherits the bill almost
whole.

**So the YSU cut on its own is worth 2.0 MiB to a bare default run.** Not
842. `kf_column` sits 16 bytes per thread below YSU and becomes the ceiling
the instant YSU leaves. Naming that is the point of this section: the next
kernel ate the saving, and a claim of 842 MiB from this cut alone would be
false.

The ceiling after **both** YSU and KF is the next kernel a default run
launches: `rrtmgp_sw_2stream` at 3,600 B, which the law prices at 264.1
MiB. Cutting the pair is therefore worth about 580 MiB of reservation
against roughly 123 MiB of YSU workspace at `nz = 49`.

### On a real forecast: 678 MiB saved with cumulus OFF, 102 MiB spent with it ON

> **Superseded in part by section 9.** The `cu_physics = 1` column below
> is what YSU's cut was worth ON ITS OWN, before Kain-Fritsch was cut.
> With both, that case is **8,740 -> 8,156 MiB, 686 MiB saved**. The
> `cu_physics = 0` column is unchanged and still current, because a run
> with no cumulus scheme never launches `kf_column` at all. Kept as
> measured because it is what established that the two had to land
> together.


The answer depends entirely on whether a cumulus scheme is in the run,
because that is what decides whether YSU is the widest LAUNCHED frame.
Both arms below are real forecasts, re-prepared on a development machine at the current
version from the surviving GFS GRIBs and `WPS_GEOG`, `nvidia-smi` sampled
at 10 Hz, every run `status: PASS`, wall clock unchanged. Physics:
`mp_physics 10`, RRTMGP LW+SW, `bl_pbl_physics 1`, Noah, `nz 49`, d01
466x374 + d02 198x198. Only `ysu.cu`, `ysu.py`, `preflight.py` and
`kernel_frame_recordings.py` differ between arms.

| `cu_physics` on d01 | pre-cut peak | cut peak | change |
|---|---|---|---|
| **0** (convection-permitting practice) | 8,190 MiB | **7,512 MiB** | **-678 MiB** |
| 1 (Kain-Fritsch) | 8,740 MiB | 8,842 MiB | +102 MiB |

Three runs each at `cu_physics = 0` (8,190 / 8,190 / 8,190 against 7,512 /
7,510 / 7,512) and eight each at `cu_physics = 1`.

**With cumulus off, YSU held the ceiling and the cut takes it away: 678
MiB back on a 16 GiB card.** That is the case the config comment in the
shipped experiment calls operational practice ("convection-permitting
practice runs `cu_physics = 0` below about 4 km and lets the resolved
dynamics convect") and it is what `configs/hrrr_native_3km_demo.toml`
selects.

**With Kain-Fritsch on, the cut costs 102 MiB**, because `kf_column` still
pins the reservation at 842 MiB so YSU's frame frees nothing, while the
workspace replacing it is real allocation. Cutting KF is what makes that
column positive too.

### The cumulus-on case in detail

Eight runs of each arm: the pre-cut peak read 8,740 MiB on every one, and
the cut arm read 8,842 MiB on seven of eight with a single 8,830. So the
arms are separated by about 100 MiB against a run-to-run spread of at most
12: the sign and the size are solid, and quoting it as exact to the MiB
would not be.

The sign is not a mistake and it is not a defect in the cut. It is the
floor: `kf_column` still pins the reservation at 842 MiB, so removing
YSU's frame frees *nothing*, while the workspace that replaces it is 123
MiB of real allocation. Until Kain-Fritsch is cut too, a cumulus run
trades a reservation nobody stops paying for an allocation somebody does.

The tile is the only lever on that overhead, and at forecast scale it is
nearly free to lower:

| blocks/SM | workspace | forecast peak | wall |
|---|---|---|---|
| 4 | 30.8 MiB | 8,778 MiB | 8.459 s |
| 8 | 61.5 | 8,780 | 8.451 |
| 12 | 92.3 | 8,812 | 8.504 |
| 16 (shipped) | 123.0 | 8,842 | 8.47 |

The shipped 16 is kept because it is the KERNEL-level plateau and the
occupancy cap, and this 60-second forecast calls YSU too few times to be a
throughput test: tuning the default down on this evidence would be
over-fitting a short run. The row is here so the knob's real cost is on
record.

**The conclusion this forces: the YSU and KF cuts should land together.**
Shipped alone, YSU is worth 678 MiB to a convection-permitting run and
costs 102 MiB to a Kain-Fritsch one. Cutting KF removes the second half of
that trade and makes the change positive everywhere.

### The ceiling map, so the next claim is not made against a stale one

Frames of kernels that are actually LAUNCHED, read off the driver
2026-08-21 (sm_86 box for this table; sm_120 differs by a few hundred bytes
on some rows and not at all on others).

After YSU and KF, a bare default run's ceiling is:

| kernel | frame | reservation on 70 SM x 1,536 |
|---|---|---|
| `rrtmgp_sw_2stream` | 3,600 B | 264.1 MiB |
| `refl10cm_morrison_column` (`REFL_KMAX 49`) | 3,528 B | 256.8 |
| `thompson_*_sediment_64` | 2,816 B | 183.7 |
| `vert_interp` | 768 B | 0 |
| `acoustic` | 544 B | 0 |

Selectors that would put the ceiling straight back up, each a single
kernel holding a whole column exactly as YSU did: these are the next
targets, in order:

| kernel | frame | selector |
|---|---|---|
| `shinhong_column` | 13,000 B on NVRTC 13.0 (17,160 on 13.3 and 13.4; sm_120, read 2026-09-28) | `bl_pbl_physics = 11` |
| ~~`myjpbl_column`~~ | ~~9,232 B~~ -> **0 B**, done 2026-10-12 (section 10); `myjurb_column` (10,256-12,304 B) with it | `bl_pbl_physics = 2` |
| ~~`wsm6_column`~~ | ~~7,216 B~~ -> 0 B, done 2026-10-12 (section 10; `wdm6_column` with it) | `mp_physics = 6` |
| `sase_*` | 6,272 B at the 128 tier; 49 B x the per-run `SASE_KMAX` tier (1,568 / 3,136 / 6,272 / 12,544 B at 32 / 64 / 128 / 256) | SASE |

`sase_*` is already specialized: `woof/core/sase.py` compiles `SASE_KMAX`
per run at the smallest power-of-two tier from 32 that holds `nz`
(`woof.core.sase_limits.kmax_tier`), and preflight charges that tier's
frame (`preflight.SASE_TIER_FRAME`, driver-read at all four rungs on
sm_86, NVRTC 12.9.86). A 49-level run holds 3,136 B, a 256-level run
12,544 B -- about 2.8 GiB of reservation on a 170-SM card, which is the
price of the deepest column, not of selecting SASE. The tier only sizes
per-thread arrays, so it moves no output bit
(`tests/test_sase_gpu.py::test_sase_kmax_tier_is_bitwise_inert`).

`myjpbl_column` deserved the emphasis: it was the *same* 9,232 B YSU
carried, in one kernel with no template variants, so a MYJ run still paid
the full ~842 MiB that YSU used to. The technique in this section applied
to it unchanged, and section 10 records the cut.

What is NOT a target, and why it looks like one: `thompson` (11,264 B) and
`morrison` (5,120 B) top out in their `KMAX = 256` template instantiations,
which a run with `nz <= 64` never launches, measured 2,816 B and 1,280 B
for the `_64` variants those runs actually dispatch. Cutting them would
buy a 49-level run nothing.

### Bit-identity: a declared divergence, contraction only

Unlike `gf.cu`, `ysu.cu` spells its arithmetic with plain `+`/`*` rather
than pinned `__fadd_rn`/`__fmul_rn` intrinsics, so it is exposed to ptxas
re-deciding FP contraction when operands move from local to global memory.
It did.

* **Cause, proven:** compiled `--fmad=false`, the pre-cut and cut kernels
  produce the **identical digest, 0 differing words of 903,168**. The
  arithmetic the source specifies is unchanged; what moved is the
  contraction choice.
* **Extent, with contraction on:** on a randomized 2,304-column adversarial
  set, **724 words of 903,168 differ (0.08%)**, all at cancellation points.
  Max relative difference **9.2e-04** on `dtheta` (2 words of 112,896);
  `exch_h`/`exch_m` within **3 ULP**; `hpbl`, `kpbl`, `wstar`, `delta`,
  `topdown_radsum`, `wstar3_2` and `cloudflg` **bit-identical everywhere**.
  Every discrete output and every PBL diagnostic is unmoved.
* **The verification of record is unmoved.**
  `tests/test_ysu_wrf461_parity.py` pins the ULP distances to WRF v4.6.1's
  own `bl_ysu_run` words *for equality*, not as bounds. All ten pass
  unchanged (`du` 1457, `dv` 23302) on the cut kernel. That suite was
  validated as an instrument first: perturbing one scheme constant makes it
  fail at `du 559241 != 1457`.
* **Level coverage.** The same comparison at `nz` = 20, 64, 100 and 128
  diverges the same way and by the same kind of amount: nothing grows with
  the level count. At `nz = 128`, the deepest the kernel accepts: 1 column
  of 2,304 has a `dtheta` difference, max relative 1.7e-03, `exch_h`/
  `exch_m` within 5 ULP, and every 2-D diagnostic still bit-identical.
* **The divergence is REMOVABLE, and the price is measured.** The
  Kain-Fritsch cut (section 9) found the technique: bisect over the
  ARRAYS, one hybrid kernel per array with that array in the workspace and
  every other left on the stack, and see which placements move a bit.
  Applied to YSU on a development machine, with the method validated first (an all-local
  hybrid with templated helpers reproduces the untouched pre-cut kernel's
  digest exactly, so templating alone moves nothing):

  | array | verdict |
  |---|---|
  | `zq`, `dza`, `lower`, `diag`, `upper`, `rhs` | **move bits** |
  | the other twelve | clean |

  All six produce the *same* digest individually, which is the signature of
  one fused chain: they are the arrays feeding the Thomas solve and the
  `rdz`/`dsdz2` terms that consume it. Keeping those six on the stack and
  the other twelve in the workspace is **BIT-EXACT** against the pre-cut
  kernel.

  What it costs, measured on a development machine:

  | variant | frame | reservation | workspace | total |
  |---|---|---|---|---|
  | all-workspace (**shipped**) | 0 B | 0.0 MiB | 123.0 MiB | **123.0 MiB** |
  | six movers local (bit-exact) | 3,088 B | 211.6 MiB | 82.0 MiB | **293.6 MiB** |

  So bit-exactness is available for **170.6 MiB**, which would turn the
  678 MiB saved on a convection-permitting forecast into about 507. That
  is a trade between footprint and bit-identity rather than a defect with
  an obvious fix, so it is recorded here as a lever and NOT taken
  unilaterally. Level-specializing just those six to `nz` would shrink the
  frame further (about 1,180 B at nz=49), at the cost of compiling the
  module per level count -- which is exactly what section 9 retired for
  `kf`.
* `__restrict__` on the workspace pointer and on the column views was tried
  and does **not** restore contraction parity.
* Both A/B controls fire: perturbed input and perturbed kernel each move the
  digest, and every arm clears a work proof (tendencies non-constant and
  finite, `hpbl > 0` on 100% of columns, `kpbl > 1` on 96.5%), so the
  comparison is a measurement and not a harness that never ran.

### Reproducing

```
pytest tests/test_ysu_workspace.py
pytest tests/test_ysu.py tests/test_ysu_wrf461_parity.py
```

`tests/test_ysu_workspace.py::test_the_ysu_frame_stays_under_the_default_stack`
is the platform-independent gate. Run against the pre-cut kernel, 5 of the
7 nodes in that file fail, including that one and the tiling gate.

### 2026-10-12: the ceiling went to 256, and it cost nothing

Because the workspace extent is the runtime `nz`, `YSU_KMAX` stopped sizing
anything when this cut landed: it was left as a 128-level refusal guard
(`if (col >= st || nz > YSU_KMAX) return;`) over arrays that would have held
any depth. It now sits at the dynamical core's own 256
(`ACOUSTIC_VERTICAL_LEVEL_BOUNDS`), in `woof/core/kernels/ysu.cu`, the
global port's `woof/globe/core/kernels/ysu.cu` (whose launcher reads the
contract's `YSU_VERTICAL_LEVEL_BOUNDS` instead of its own literal), and the
ensemble batch planner (`woof/ensemble/batch_physics.py`, which had its own
`4 <= nz <= 128` and now imports the contract). Shin-Hong took the same
route on 2026-09-30 and went the same way: `SHINHONG_KMAX` 256.

* **Frames:** `ysu_column`, `ysu_column_bep`, `ysu_column_topo`, the global
  `ysu_column`, `shinhong_column` all read **0 B** before and after (sm_86,
  NVRTC 12.9.86). Nothing to re-price.
* **Bit-identity at `nz <= 128`:** every output word of the shipped kernel
  equals the same source compiled with `#define YSU_KMAX 128` (resp.
  `SHINHONG_KMAX`), at `nz` 49 and 128, regional and global
  (`tests/test_physics_256_levels_gpu.py`). That old-ceiling arm is the
  pre-raise source to the byte once comments are stripped.
* **256 levels to a 1 hPa top** run through the shipped launchers and come
  back finite, with `hpbl > 0` and `1 <= kpbl < 256` in every column (same
  file).

---

## 9. Kain-Fritsch: the other half of the pair, and how it stayed bit-exact

Section 1 fixed `kf_column` by making its compile-time bound follow `nz`,
which took the frame from 24,064 B to 9,216 B and the reservation from
5,738 MiB to 840 MiB. This section removes the remaining 840 MiB with the
technique of §7 and §8: the column arrays leave the per-thread frame for a
global workspace the launcher sizes to the columns actually IN FLIGHT.

It is the second half of a pair, and **neither cut is worth anything
alone.** The driver's backing store is a maximum over launched frames, so
with `ysu_column` at 9,232 B and `kf_column` at 9,216 B, cutting either one
leaves the other setting the same ceiling: §8 measured the YSU cut on its
own as a 102 MiB *regression*, because it added a workspace and freed
nothing.

### The reservation, measured

Fresh process per row, zero-work launch (`ny*nx = 0`), `cudaMemGetInfo`
either side, with the 0 B and 16,384 B controls validating the instrument
first on each box (0.0 MiB, and 1,574.0 MiB against the law's 1,575.0).

| platform | frame before | reserved before | frame after | reserved after |
|---|---|---|---|---|
| a development machine, RTX 5070 Ti, 70 SM x 1,536, sm_120, NVRTC 13.0.48 | 9,216 B | **840.0 MiB** | 512 B | **0.0 MiB** |
| a development machine, same card, NVRTC 13.3.33 | 9,216 B | **840.0 MiB** | 512 B | **0.0 MiB** |
| a development machine, RTX 3080, 68 SM x 1,536, sm_86, NVRTC 13.0.48 | 9,216 B | **816.0 MiB** | 512 B | **0.0 MiB** |

Every "before" row is the law exactly. Every "after" row is zero because
512 B is under the 1,024 B default stack.

### On a real forecast, the pair is worth 686 MiB

The same rig §8 used: a bare-default 2-domain forecast (mp_physics 10,
RRTMGP LW+SW, Kain-Fritsch, YSU, Noah, nz 49, d01 466x374 + d02 198x198),
device memory sampled at 10 Hz by `nvidia-smi` from outside the process,
card verified idle at 2 MiB, **the YSU cut present in BOTH arms** so the
only thing that differs is KF. Three runs per arm.

| | peak device memory | wall |
|---|---|---|
| YSU cut only (§8's shipped arm) | 8,842 / 8,842 / 8,842 MiB | 8.44 / 8.46 / 8.46 s |
| YSU + KF cut | 8,260 / **8,156** / **8,156** MiB | 9.70 / 8.38 / 8.43 s |

**686 MiB on a bare default run, with no flag to set, and the wall clock
unchanged** (8.43 s median against 8.46). The first cut run's 8,260 MiB and
9.70 s are a cold NVRTC cache compiling the new `kf.cu`; both settle from
the second run on. Against §8's measured 8,740 MiB for the *pre-cut*
baseline, the pair lands 584 MiB below where the tree started, and the
102 MiB the YSU cut cost on its own is repaid six times over.

### 512 B, not 0: the two arrays that stay on the stack

`kf_column` declares 54 column arrays. Fifty-two move. `tv_env` and
`positive_energy` stay, and that is the one part of this cut that is not a
pure placement change.

They were found by measurement, not by reading: build 54 hybrid kernels,
each with ONE array in the workspace and the other 53 left local, and diff
the outputs against the pre-cut kernel. Fifty-two produce zero differing
words. These two produce 55,360 and 34,560 of a 410,624-word grade.

The reason is visible in the frame all along. Those 54 arrays account for
188 B per level, which is 47 arrays' worth and not 54: the compiler was
already ELIMINATING seven of them, and these two are among them. `tv_env`
is rematerialised from `temperature` and `qenv` at its use sites;
`positive_energy` is written and read inside one loop iteration. Eliminated,
their defining expressions FUSE into the expressions that consume them
(`cape += positive_energy[nk1]` becomes a single `fma` with `dilbe*9.81f`)
and the CAPE-removal closure amplifies the difference into thousands of
moved words. Force them into memory and the store breaks the fusion.

So they keep their `float[KF_KMAX]` declarations. At the shipped
`KF_KMAX = 128` that is 512 B, half the default stack, and the driver's
reservation term is still exactly zero. (At the 256 it was raised to on
2026-10-12 it is 1,024 B: AT the default stack, still zero -- see the end of
this section.)

### Bit-identity, and the method that got there

`kf.cu` uses plain `+`/`*`, not `gf.cu`'s pinned intrinsics, so it is
exposed to ptxas re-deciding FP contraction exactly as `ysu.cu` was. The
naive cut (all 54 arrays into the workspace) **was divergent**:
1,746,944 of 13,139,968 words, up to 4,660 ULP. Compiled `--fmad=false`,
both arms were identical, which names the cause as contraction and not the
transformation.

What did NOT work, recorded because it looks like the obvious next move:
attributing `fma.rn.f32` instructions to source lines through the PTX
`.loc` markers. Loop-unroll factors dominate the per-line counts (31
against 15 for the same line), and pinning the four lines that genuinely
gained or lost an FMA changed the delta by **exactly zero**.

What worked was the 54-hybrid bisect above. It is mechanical, it runs in
one process in about four minutes, and it answers the question directly
instead of inferring it from the compiler's output. **Any kernel whose cut
comes out divergent under contraction can be asked which of its arrays is
responsible, one array at a time**, including `ysu.cu`, whose divergence
§8 declared rather than removed.

The result, MEASURED rather than argued: interleaved pre-cut and cut arms
in ONE process, four rounds, four `KFPhaseMode` values, 2,048 columns of
eight interleaved soundings (so neighbouring lanes take different branches
and warp divergence is exercised), compared as `uint32` bit patterns and
never as tolerances:

* **13,139,968 graded words per arm, 0 differ**, on NVRTC 13.0.48, on
  NVRTC 13.3.33, and on sm_86. All twelve arm-runs produce one digest.
* The same A/B at a **32-column tile**, so the workspace is REUSED across
  64 sequential launches: 0 words differ.
* Perturbed-input control (one temperature word, one ULP): **666 words
  move**.
* Perturbed-kernel control (one constant, one ULP): **652,032 words move**.
* The verdict is SUPPRESSED unless both arms show triggered columns and
  non-zero output, so a null result cannot come from a harness that never
  ran.
* `tests/test_kf.py`, the WRF v4.6.1 oracle parity suite, is green
  unchanged.

Unlike §8's YSU result, this is bit-identity and not a declared divergence.

### The tile is 8 blocks/SM, which is where the card fills

100,000 columns, median of 7, in two independent sweeps agreeing to
0.05 ms:

| blocks/SM | tile | wall | workspace |
|---|---|---|---|
| 1 | 2,240 | 234.8 ms | 21.8 MiB |
| 2 | 4,480 | 177.9 ms | 43.5 MiB |
| 4 | 8,960 | 109.8 ms | 87.1 MiB |
| 6 | 13,440 | 101.0 ms | 130.6 MiB |
| **8** | **17,920** | **98.3 ms** | **174.2 MiB** |
| 12 | 26,880 | 103.5 ms | 261.3 MiB |
| 16 | 35,840 | 95.6 ms | 348.4 MiB |
| 24 | 53,760 | 93.9 ms | 522.5 MiB |
| pre-cut (local frame) | -- | 90.7 ms | 0, plus 840.0 MiB reserved |

8 blocks/SM is the kernel's MEASURED hardware residency at block 32:
`occupancyMaxActiveBlocksPerMultiprocessor` returns 8, because the kernel
uses 227 registers and 65,536/227 caps it at 288 threads per SM. Past that
the workspace grows for threads the card cannot hold resident: 24
blocks/SM buys 4.5% of kernel wall for 3x the memory. `kf_tile_columns`
takes the smaller of the constant and the live card's own query, so it only
ever goes down.

The residual 1.08x on the kernel is the price of global addressing over
local, and it does not reach the forecast: the whole-run wall above is
unchanged.

A block width of 64 was measured and is not shipped: identical wall at the
same tile (98.27 ms at 17,920 columns) with coarser tile granularity. 32 is
the warp width and the launcher's existing block, so the cut changes no
launch shape at all.

### What changed around it

* `KF_KMAX` is a REFUSAL CEILING and nothing else. It sizes only the two
  stack-resident arrays, so `woof/core/kf.py` compiles the module ONCE
  instead of once per distinct level count. The `nz` outside [8, 128]
  refusal (now [8, 256], below) is unchanged and is carried by
  `woof.physics_compat.validate_resolved_physics_vertical_levels`, which
  names Kain-Fritsch, plus the guard in `kf_column` itself.
* `kf` left `LEVEL_SPECIALIZED_KERNEL_FRAMES`. Its 188 B/level model is
  wrong by a factor of 47 now, and the frame does not follow `nz` at all.
  The retired model survives in `tests/test_preflight.py` as
  `KF_AS_BUILT_FRAME`, which is what the 2026-07-26 historical runs and the
  RTX 4080 fleet table are priced against: pricing them at 512 B would
  claim those runs carried no cumulus reservation, which is the one thing
  they are evidence against.
* `preflight.kf_column_workspace_bytes` prices the workspace beside the
  context and the backing store, and `column_workspace_bytes` SUMS the
  three launcher-owned workspaces (GF, KF, YSU). A sum and not a maximum:
  they are ordinary allocations owned by different launchers, not one
  per-context store the driver sizes to the widest frame.
* The workspace extent is the RUNTIME `nz`, taken from the kernel's
  existing `nz` argument. A compile-time frame could only ever be sized by
  a compile-time bound; this is the part of the saving §1 could not reach.
* The columns launch in TILES offset by a new `col0` argument rather than
  by slicing, because KF's state arrays are `(nz, ny, nx)` and a tile of
  columns is a stride-`ncol` scatter, not a contiguous slice. (GF could
  slice; its layout is `(ncol, ...)`.)

### Reproducing

```
pytest tests/test_kf_workspace.py
pytest tests/test_kf.py tests/test_kernel_local_bounds.py
GPUWM_NO_LOCAL_GPU=1 pytest tests/test_preflight.py
```

`tests/test_kf_workspace.py::test_the_kf_frame_stays_under_the_default_stack`
is the platform-independent gate: it asserts the property that matters --
frame at or under the 1,024 B default stack, so the reservation is zero --
on whatever box runs it, including boxes with no recorded frame row.
`test_the_two_stack_arrays_are_the_measured_pair` is the one that stops a
later pass from "finishing the job" on `tv_env` or `positive_energy`
without reading this section first.

## 10. Legacy RRTMG LW: `RLW_MAXLAY` became a per-run tier (2026-10)

`rlw_rtrn_march` (and its coalesced twin) is the one legacy-LW kernel
with a frame: four per-thread float columns of `RLW_MAXLAY` entries,
2,048 B at the old fixed 128 -- which was also the reason a legacy run
could not carry more than 128 longwave layers. `RLW_MAXLAY` is now an
`#ifndef` default the host overrides per run at
`legacy_longwave_layer_tier(nlayers)` (128 for any column of at most 128
layers, else rounded up to a multiple of 64), so the frame follows the
run's radiation column instead of bounding it.

Read through the production loader
(`woof.core.rrtmg_lw.gpu_local_frame_bytes(maxlay)`) on sm_86 (RTX
A6000), widest kernel of the unit:

| `RLW_MAXLAY` | NVRTC 12.9 | NVRTC 13.4 | reservation, 84 SM x 1,536 |
|---|---|---|---|
| 64  | 1,024 B | 1,024 B | 0 |
| 100 | 1,600 B | -- | 70.9 MiB |
| 128 (shipped) | 2,048 B | 2,048 B | 126.0 MiB |
| 129 | 2,064 B | 2,064 B | 128.0 MiB |
| 192 | 3,072 B | 3,072 B | 252.0 MiB |
| 256 | 4,096 B | 4,096 B | 378.0 MiB |
| 269 | 4,304 B | 4,304 B | 403.6 MiB |
| 320 | 5,120 B | 5,120 B | 504.0 MiB |
| 384 | 6,144 B | 6,144 B | 630.0 MiB |

Exactly 16 B x `RLW_MAXLAY` at every bound on both compilers;
`rlw_cldprmc` stays 64 B and every other kernel 0 B. Preflight prices
the unit per domain with that law
(`woof.core.preflight.rrtmg_lw_legacy_chain_frame_bytes`, from the run's
`nz` and `p_top`), anchored on the shipped-tier row
`CHAINED_TRANSLATION_UNIT_FRAMES["rrtmg_lw_legacy_chain"] = 2048` (which
an import-time check holds to 16 x 128). sm_120 and sm_89 carry only the
shipped-tier reading; their deeper tiers are priced by the same law and
were not re-measured. On a 170 SM x 1,536 card a 256-level column at a
1 hPa top (tier 256) reserves (4,096 - 1,024) x 261,120 B = 765 MiB,
against 255 MiB at the shipped tier.

Bit-identity: a column of at most 128 layers compiles the unchanged
unit (no define is prepended), whose PTX is byte-identical to the
pre-tier source on sm_86 at NVRTC 12.9 and 13.4. Deeper tiers are held
to the NumPy port word for word by `tests/test_rrtmg_lw_deep_columns.py`,
whose frame leg also re-reads 128/192/256/320 on whatever card runs it
and fails if the driver reports more than preflight charges.

### 2026-10-12: `KF_KMAX` 256

`KF_KMAX` went 128 -> 256, the dynamical core's own ceiling
(`ACOUSTIC_VERTICAL_LEVEL_BOUNDS`), with the floor unchanged at 8. Because
it still sizes `tv_env` and `positive_energy`, this is not a guard-only
change, and both properties this section rests on were re-measured rather
than argued:

* **Frame, sm_86, NVRTC 12.9.86 (RTX A6000):** `KF_KMAX` = 49 / 128 / 256
  read 200 / 512 / **1,024 B**, identically from the pre- and post-raise
  source. 1,024 B is exactly the default stack, so the reservation term is
  still zero and `test_the_kf_frame_stays_under_the_default_stack` (frame
  `<=` 1,024) holds with no margin left: the next array to land on the
  stack will be caught. The `kf` row of `KERNEL_MAX_LOCAL_SIZE_BYTES` and
  the sm_86 / 12.9.86 recording are 1,024. The sm_120 and older sm_86
  readings (512 B) predate the raise and could not be re-read here; the
  sm_120 / 13.4.92 recording's `kf: 0` is very likely a symbol-regex
  artefact (`extern "C" __global__ __launch_bounds__(256) void kf_column`
  did not match the census regex, so the module read as having no kernel);
  `tools/vram_reserve_probe.py` and the driver gate in
  `tests/test_preflight.py` now accept the `__launch_bounds__` spelling.
* **Bit-identity at `nz <= 128`:** every output word, all four
  `KFPhaseMode`s, at `nz` 49 and 128, equals the kernel compiled with
  `KF_KMAX = 128` (`tests/test_physics_256_levels_gpu.py`, which asserts
  the two frames differ first so the A/B cannot compare an image with
  itself, and that columns convected). The existing `KF_KMAX = nz` A/B in
  `tests/test_kernel_local_bounds.py` and the WRF parity suite
  `tests/test_kf.py` are unchanged and green.
* **256 levels to a 1 hPa top:** a 64-column batch through `launch_kf`
  triggers on 23 columns (3 deep, 20 shallow), every output finite.

The same raise went to terrain drag (`TD_KMAX`, `gwd_opt` 1 and 3), whose
per-thread stress profiles `taup[TD_KMAX + 1]` (and GSL's `taufb`) are real
per-thread arrays: KIM 640 -> **1,152 B**, GSL 1,184 -> **2,208 B** on sm_86
/ 12.9.86, so `gwd_opt = 3` reserves 1,184 B per resident thread (about
146 MiB on an 84-SM x 1,536 card), priced through the
`terrain_drag_composed` chained-unit frame. Terrain drag also gained a row
in the vertical contract (`TERRAIN_DRAG_VERTICAL_LEVEL_BOUNDS`) read by
`validate_resolved_physics_vertical_levels` whenever `gwd_opt` is 1 or 3:
until then a too-deep drag run passed `woof check` and failed at its first
PBL call. Bit-identity at `nz <= 128` is shown the same way, and the WRF
v4.7.1 parity suite is unchanged and green.

---

## 10. WSM6 and WDM6: the hex model's microphysics, and the 80-level wall

`wsm6_column` is the hex (MPAS) model's default microphysics and a
regional default in several shipped profiles; `wdm6_column` is its
double-moment sibling. Both kept ~30 (WSM6) and ~37 (WDM6) whole-column
arrays per thread, sized by a compile-time `*_KMAX` tier the launchers
chose from `nz` -- 64 or 80 -- so the two schemes had two costs:

* the frame: WSM6 7,216 B at 64 and 9,008 B at 80 (+112 B/level), WDM6
  9,264 B and 11,568 B (+144 B/level), which the driver reserves times the
  card's whole resident-thread capacity;
* a hard 80-level ceiling. 256 levels would have been ~28.7 KB (WSM6) and
  ~36.9 KB (WDM6) per thread -- several GiB of reservation -- so the tier
  ladder simply stopped at 80, and with it every WSM6/WDM6 column, including
  the hex model's.

Both now keep their column arrays in a global workspace their launchers
(`woof/core/wsm6.py`, `woof/core/wdm6.py`) allocate per call, sized to the
columns IN FLIGHT and to the RUNTIME `nz`, exactly the YSU/KF route
(sections 8-9). The `*_KMAX` define is a 256-level refusal ceiling that sizes
nothing; the tier ladders (`WSM6_TIER_FRAME`, `WDM6_TIER_FRAME`,
`*_KERNEL_LEVEL_TIERS`, `*_level_tier`) are retired; one binary serves every
level count from 2 to 256.

All measurements below: **RTX A6000, 84 SMs x 1,536, sm_86**, CuPy 14.2.0,
NVRTC 12.9.86 and 13.4.92 (both read; they agree everywhere), 2026-10-12.

### Layout

Slot `s`, level `k`, lane `t` of a block lives at
`block_base + (s * wskp + k) * 32 + t` with `wskp = nz + 1` (the PLM
interface arrays index `nz`), one region of `SLOTS x wskp x 32` floats per
one-warp block. WSM6 has 30 slots (15 in the column routine, 12 in
`sediment_column`, 1 in `plm_arrival`, 2 in `plm_regular_remap`); WDM6 has
37 (23 + 11 in the snow/graupel fallout + 1 + 2; the cloud-ice fallout runs
after the snow/graupel fallout returns and reuses its slots). Slot ids are
literals; `tests/test_wsm6_workspace.py` / `tests/test_wdm6_workspace.py`
read them out of the source and fail on a duplicate, an overflow or a gap.

The columns launch in tiles of `SMs x 16 x 32` offset by a `col0`
argument (the `(nz, ny, nx)` fields make a column tile a strided scatter),
with the device's occupancy query allowed only to lower the 16.

### The register file decided the view type

The obvious view -- a struct holding a 64-bit pointer per array, as
`ysu.cu` does -- compiled to **160 registers**, which on sm_86 holds 12
one-warp blocks per SM where the pre-cut kernel held 16 (the per-SM block
ceiling). 65,536 columns, `nz = 55`, median of 7:

| WSM6 variant | regs | frame | blocks/SM | wall |
|---|---|---|---|---|
| pre-cut (local arrays, 64 tier) | 96 | 7,216 B | 16 | 7.25 ms |
| 64-bit pointer views | 160 | 0 B | 12 | 9.87 ms |
| same, `__launch_bounds__(32, 16)` | 128 | **72 B** spill | 16 | 8.43 ms |
| **one base pointer + 32-bit slot offsets (shipped)** | **116** | **0 B** | 16 | **8.32-8.44 ms** |

WDM6 the same way: 160 regs / 0 B, 128 regs / 88 B with the bound, **116
regs / 0 B** with offsets. The offsets need no launch bound, so none is
shipped.

### Before and after

Frames, through the production loader and `local_size_bytes`:

| kernel | pre-cut 64 tier | pre-cut 80 tier | cut, any nz <= 256 |
|---|---|---|---|
| `wsm6_column` | 7,216 B | 9,008 B | **0 B** |
| `wdm6_column` | 9,264 B | 11,568 B | **0 B** |

The 0 B was also read compile-only (NVRTC -> cubin, `ptxas -v`) for sm_89
and sm_120 at both builds: 0 B stack frame, 0 B spill, everywhere.
`kernel_frame_recordings.py` carries those readings and DROPS the old
9,264 / 7,216 B rows from the recordings whose NVRTC builds were not
available to re-read (sm_120 13.0.48, 13.0.88, 13.3.33; sm_86 13.0.48)
rather than carry rows describing a source that no longer exists.

What the workspace holds, at the shipped 16 blocks/SM (`tile = 43,008`
columns on this card):

| `nz` | WSM6 | WDM6 | pre-cut WSM6 reservation, by the law |
|---|---|---|---|
| 55 | 275.6 MiB | 339.9 MiB | 761.9 MiB (64 tier) |
| 80 | 398.7 MiB | 491.7 MiB | 982.4 MiB (80 tier) |
| 256 | 1,264.9 MiB | 1,560.1 MiB | not compilable |

Wall clock, 65,536 columns (43,008 at `nz = 256`), dt 60 s, median of 7:

| case | pre-cut | cut, 16 blocks/SM |
|---|---|---|
| WSM6 `nz = 55` | 7.25 ms | 8.32 ms (1.15x) |
| WSM6 `nz = 80` | 10.88 ms | 12.57 ms (1.16x) |
| WDM6 `nz = 55` | 11.90 ms | 13.12 ms (1.10x) |
| WDM6 `nz = 80` | 20.72 ms | 22.23 ms (1.07x) |
| WSM6 `nz = 256` | -- | 21.69 ms (32.12 at 8 blocks/SM, 58.24 at 4) |
| WDM6 `nz = 256` | -- | 168.66 ms |

16 is the tile, not 8: one thread is one column, so a ragged second tile
costs nearly a whole column's latency, and a tile at full residency ran as
fast as one launch over every column (8.44 ms vs 8.37 ms at `nz = 55`,
balanced or not). Halving it halves the workspace and, at 256 levels,
costs 1.5x on the kernel. WDM6's 256-level wall is its explicit rain
sub-stepping on thin layers, not the workspace.

### Bit-identity: measured, both schemes, both compilers

`wsm6.cu` and `wdm6.cu` use plain `+`/`*`, so they are exposed to
contraction re-decisions the way `ysu.cu` was. They did not move. Pre-cut
kernel compiled from git (`7950e417`) at the tier its launcher chose,
shipped kernel through the shipped launcher, ONE process, compared as
`uint32` bit patterns:

* WSM6: `nz` 40, 64 and 80 x `hail_opt` 0/1 x {two 20 s calls, one 300 s
  call (three minor loops)}, 1,024-2,048 columns of varied soundings with
  every hydrometeor present: **21,274,624 graded words, 0 differ**, under
  NVRTC 12.9.86 and 13.4.92;
* WDM6, the same grid plus `nn`/`nc`/`nr`, a numberless-rain sprinkle and a
  land/sea mix: **25,821,184 words, 0 differ**, both builds;
* controls fire: one constant moved one ULP in the pre-cut kernel moves
  64,871 (WSM6) / 63,787 (WDM6) words; one `theta` word moved one ULP moves
  28 / 25;
* the residue gate (workspace pre-filled with NaN vs 1e30) and the tiled
  gate (five ragged 64-column tiles vs one launch vs the shipped launcher)
  are bit-identical;
* the WDM6 CPU-mirror suites (`test_wdm6_sedimentation.py`,
  `test_wdm6_count_safety.py`, `test_wdm6_numberless_rain.py`) pass
  unchanged: the mirror keeps the pre-workspace `wdm6_column_impl` entry
  point and the array helpers are templates over the array type.

### What changed around it

* `preflight.wsm6_column_workspace_bytes` / `wdm6_column_workspace_bytes`
  price the workspaces (the worst selecting domain, bounded by the tile) and
  join the `column_workspace_bytes` sum; the `wsm6`/`wdm6` frame rows are 0.
* `woof.physics_vertical_contract` IMPORTS both bounds, `(2, 256)`, from
  the CuPy-free constants modules instead of transcribing WSM6's.
* The hex device admission charges `WSM6_WORKSPACE` as a tiled workspace
  (30 slots, 32 lanes, 16 blocks/SM, `nz + 1`), its 80-tier
  `local_store_extra_bytes` is zero, and its WSM6 ceiling is 256. Its
  measured cores still carry the retired 7,216 B frame's reservation (no
  ledger arm has been run at the new tip), so the workspace is an over-count
  on top of them -- the safe direction -- until one is.

---

## 10. MYJ and MYJURB: the frame that stood between MYJ and 256 levels

`myjpbl_column` (`bl_pbl_physics = 2`) and `myjurb_column` (MYJ under
`sf_urban_physics = 2/3`) were the last PBL kernels holding a whole column
in per-thread local arrays sized by a compile-time bound, `MYJ_KMAX = 128`:
19 and 24 arrays, 9,232 B and 10,256 B frames (12,304 B for `myjurb` on
sm_120 under NVRTC 12.9.86). The 256-level target (nz = 256 at
p_top = 1 hPa) would have needed `MYJ_KMAX = 256`, about 18.4 KB and
20.5 KB per thread, roughly 2.1 GiB and 2.3 GiB of reservation on an
84-SM card. Raising the define was not a real option.

Both kernels now take the section 8 route. Every column array lives in a
global workspace that `woof/core/myjpbl.py` / `woof/core/myjurb.py`
allocate per call. The workspace is sized to the columns IN FLIGHT and
to the RUNTIME level count: `blocks x slots x (nz + 1) x 32` floats,
lane-interleaved (`block_base + (s * wskp + k) * 32 + t`). The
launch block is 32. Columns go in tiles offset by a new `col0`
argument, because the state arrays are `(nz, ny, nx)`. `MYJ_KMAX`
is now a refusal guard and sizes nothing. It is 256 in both kernels, and
the contract (`MYJ_VERTICAL_LEVEL_BOUNDS = (4, 256)`) and both launchers
match it. The arrays keep their names, extents and access order. Only
`real x[MYJ_KMAX]` became a `MyjCol x = MYJWS_AT(wsb, slot, wskp)` view,
and the two `real* var` species/component selectors became views.

### The reservation, measured

RTX A6000, 84 SMs x 1,536, sm_86, NVRTC 12.9.86 (CuPy 14.2.0). Fresh
process per row, zero-work launch, `cudaMemGetInfo` on either side. The
pre-cut kernel is the source at 7950e4171 compiled beside the new one.

| kernel | frame before | reserved before | law | frame after | reserved after |
|---|---|---|---|---|---|
| `myjpbl_column` | 9,232 B | **1,012.0 MiB** | 1,010.0 | 0 B | **2.0 MiB** (module load) |
| `myjurb_column` | 10,256 B | **1,138.0 MiB** | 1,136.0 | 0 B | **2.0 MiB** |

Compile-only readings were taken with NVRTC through the production source
assembly and options, then `cuobjdump --dump-resource-usage`. The same
reader reproduces every pre-cut row the recordings carried (9,232 /
10,256 / 12,304 B). After the cut, every compiler that could be loaded
here reads **0 B** for both kernels: NVRTC 12.9.86, 13.0.88, 13.3.33 and
13.4.92 on sm_86, sm_89 and sm_120. `kernel_frame_recordings` was updated
from these readings. The sm_86 12.9.86 rows are device reads through the
production loader. The sm_120 12.9.86 / 13.0.88 / 13.3.33 / 13.4.92 and
sm_89 13.4.92 rows are compile-only reads at each recording's own build.
The two NVRTC 13.0.48 recordings had their `myjpbl` row DROPPED rather
than back-filled, because that compiler was not in reach. The shipped
ceiling `KERNEL_MAX_LOCAL_SIZE_BYTES["myjpbl"/"myjurb"]` is 0.

### Bit-identity

`myjpbl.cu` uses plain `+`/`*`, so it had the same contraction exposure
that cost `ysu.cu` its bit-identity (section 8). Measured, not argued, by
`tests/test_myj_workspace.py`, which compiles the pre-cut source from git
beside the shipped one and compares every output word as a `uint32`
pattern: the eight tendencies/EL/EXCH, PBLH/KPBL/MIXHT, the carried TKE
and the THZ0/QZ0/QSFC/CT INOUT state. It uses 333 randomized columns per
case, mixing stable, convective, cloudy, iced, water, sea-ice and snow
columns so neighbouring lanes take different branches.

| kernel | arms | nz = 49 | nz = 100 | nz = 128 |
|---|---|---|---|---|
| `myjpbl_column` | `flqi` on / off | 149,184 words, **0 differ** | 302,031, **0** | 385,947, **0** |
| `myjurb_column` | BEP on / rural / `idiff = 1` | 149,184, **0** | 302,031, **0** | 385,947, **0** |

* Perturbed-input control (one `th` word, one ULP): the outputs move, so
  the harness can see a single ULP.
* Tiling control (forced 32-column tiles, so the workspace is reused
  across many launches and the last block is partial) at nz = 4, 49 and
  256: 0 words move.
* Residue control (workspace poisoned with -7e30 and then 3.5e30): 0 words
  move, so no slot is read before it is written.
* `tests/test_myjurb_wrf471_parity.py`, the WRF v4.7.1 word-for-word oracle,
  is green unchanged. So is the CPU-authority conformance in
  `tests/test_myj_port.py`.

Unlike YSU, no bisect was needed. On this toolchain NVRTC made the same
contraction choices with the arrays in global memory as with them on the
stack. `myjurb.cu` pins its multiplies with `FMUL` (`__fmul_rn`), so it
could not have diverged that way at all.

### 256 levels

`test_a_256_level_column_completes_with_sane_tke_and_fluxes` runs 126
columns at nz = 256 with a 48 km top (p_top 1.6 hPa), on a geometric
stretch from a 15-60 m first layer, through both kernels. Results:

* every output is finite;
* TKE is 0.1 (the EPSQ2/2 floor) to 3.5 / 3.6 m2/s2 for MYJ / MYJURB;
* PBLH is 154-2,506 m and KPBL is 4-73, so the PBL never reaches the
  stretched upper column;
* EXCH_H in the upper half of the column is 1e-4 of its 867 m2/s maximum;
* the heating tendency peaks at 0.017 K/s (MYJ) and 0.005 K/s (MYJURB),
  and the momentum tendency at 0.015 m/s2.

`test_a_256_level_column_agrees_with_the_cpu_authority` holds one
256-level column to the float32 CPU authority `woof.verify.myj_ref`
under the same 5e-3 relative bar the 30-level conformance test uses.
KPBL and PBLH are exact.

### The tile is 12 blocks/SM

51,200 columns, median of 7, A6000. Kernel residency at block 32 is 12
blocks/SM for `myjpbl` (136 registers) and 16 for `myjurb` (104 registers).

| blocks/SM | tile | ws @ nz=49 | wall @ 49 | wall @ 128 | wall @ 256 | ws @ 256 |
|---|---|---|---|---|---|---|
| pre-cut (local frame) | -- | 0, plus 1,012 MiB reserved | 3.10 ms | 8.02 ms | -- | -- |
| 2 | 5,376 | 19.5 MiB | 7.19 | 29.67 | 62.82 | 100.1 MiB |
| 4 | 10,752 | 39.0 | 6.02 | 16.71 | 34.32 | 200.3 |
| 8 | 21,504 | 77.9 | 4.50 | 11.91 | 24.26 | 400.6 |
| **12** | **32,256** | **116.9** | **3.74** | **9.71** | **19.60** | **600.8** |
| 16 | 43,008 | 155.9 | 4.65 | 12.08 | 24.42 | 801.1 |

`MYJ_TILE_BLOCKS_PER_SM = 12` (`woof/core/physics_inventory.py`) is both
the residency and the plateau. `myj_tile_columns` takes the smaller of
that value and the live occupancy query, so it can only go down. The
residual 1.2x on the kernel (3.74 against 3.10 ms at nz = 49) is the cost
of global addressing over local, about what GF and KF measured.
`myjurb` uses the same tile.

### Pricing

`preflight.myj_column_workspace_bytes` prices the worst MYJ domain at
`min(nx*ny, SMs x 12 x 32)` columns and the runtime nz. It uses 24 slots
when `sf_urban_physics` is 2 or 3 (MYJURB replaces the MYJPBL call there)
and 19 otherwise. `column_workspace_bytes` now sums it with the GF / KF /
YSU / Shin-Hong / New Tiedtke / UW workspaces, and the `woof check`
non-pool line prints it. On a 170-SM card at nz = 256 that is 1.19 GiB
(`myjpbl`) or 1.50 GiB (`myjurb`) of ordinary allocation. The old
reservation was 2.0 GiB at nz <= 128 and would have been about 4.2 /
5.7 GiB at a 256-level frame bound. On an 84-SM A6000 the numbers are
600.8 / 758.9 MiB.

## 10. RTE+RRTMGP: a frame that follows the radiation column, to 256 levels

Until 2026-10 RTE+RRTMGP refused any radiation column over 128 layers,
cap included. That ceiling was owned by one array:
`rrtmgp_planck_sources` (`woof/core/kernels/rrtmgp_gas.cu`, and the global
model's copy under `woof/globe/core/kernels/`) held a per-thread
`float pfrac[128]`, 512 B of frame. The RTE solvers already compiled
`RRTMGP_MAX_LAYERS` to each call's own layer count (`_rte_kernel`).

**The Planck array is gone, not resized.** A level source reads only the
two layers on either side of it, so the kernel now walks a one-layer
window (`pfrac_prev`) and keeps the surface layer's fraction (`pfrac_sfc`).
The expressions and their operands are unchanged, so every output word is
the old kernel's: on an RTX A6000 (sm_86, NVRTC 12.9.86) the source arrays
are byte-identical to the base kernel's at 1, 2, 30, 74, 127 and 128 layers
in both storage orders, for both copies, and so are the whole driver's
fluxes and heating rates at nz 49, 64, 115 and 127. The frame went from
512 B to 0 B (63 registers, up from 56). The other recordings keep their
unread 512 B, so the shipped `rrtmgp_gas` ceiling stays 512 B. That is
under the 1,024 B default stack, so it reserves nothing.

The bound is now on **model levels**:
`physics_vertical_contract.MAX_RRTMGP_MODEL_LEVELS = 256`, the dynamical
core's ceiling. WRF's cap layers ride on top at any model top. At
`p_top = 100 Pa` that is 256 LW layers (nint(100/400) = 0) and 257 SW. At
`p_top = 10 kPa` it is 281 LW layers and 257 SW.

**The solver frame is priced per layer above 128.** Measured through
`load_module_int_defines` on the same card, compile-only, at 1 to 300
layers:

| layers | `rrtmgp_sw_2stream` | `rrtmgp_lw_noscat` |
|---|---|---|
| 64 | 1,808 B | 512 B |
| 128 (source default, the shipped row) | 3,600 B | 1,024 B |
| 129 | 3,632 B | 1,032 B |
| 160 | 4,496 B | 1,280 B |
| 192 | 5,392 B | 1,536 B |
| 256 | 7,184 B | 2,048 B |
| 257 (256 levels, 1 hPa top) | 7,216 B | 2,056 B |
| 260 | 7,296 B | 2,080 B |
| 269 | 7,552 B | 2,152 B |
| 281 | 7,888 B | 2,248 B |

The two-stream is `28 n + 12` rounded up to 16 B. Its arrays are
`float4 optics[n]`, `float direct[n+1]` and `float2 adding[n+1]`. The LW
solver is `8 n` (`trans`, `src_up`). Both hold at every count from 2 up.
`preflight.RRTMGP_RTE_SW_TIER_FRAME` (3,600 B at the 128 tier plus 28 B per
layer, 16 B alignment) reproduces the two-stream exactly wherever `28 n + 12`
is not a multiple of 16. Where it is (n = 3 mod 4), it over-prices by 16 B.
`preflight.rrtmgp_rte_frame_bytes(nz, p_top)` takes the wider of the SW
tier at `nz + sw_upper` and the LW line at `nz + lw_upper`, for the spectra
a domain selects, and never goes below the flat 3,600 B row. Every column
of 128 radiation layers or fewer is therefore priced exactly as before.

What this charges at depth, by the law, on the RTX 5090 reference profile
(170 SMs x 1,536 threads):

| column | frame | reservation |
|---|---|---|
| any column of 128 layers or fewer (unchanged) | 3,600 B | 641.5 MiB |
| 256 levels, `p_top = 0` (no cap known) | 7,184 B | 1,534.0 MiB |
| 256 levels, 1 hPa top (SW 257) | 7,216 B | 1,542.0 MiB |

Before this change, the flat row under-priced the 1 hPa case by 900.5 MiB.

Only the solver frames grow; `_rte_gpt_tile` reads the frame off the
per-run compiled kernel (`local_size_bytes`), so the non-folding tile rule
already sees the deeper frame. The production folds use fixed widths.

Other architectures and NVRTC builds were not re-read at depth. The slope
is the byte count of the arrays, which no register allocation can shrink,
and every recorded platform reads the same 3,600 B at the 128 default.
`tests/test_kernel_local_bounds.py::test_the_rrtmgp_rte_layer_frames_are_what_preflight_prices`
re-reads the depths on whatever card runs it.

### Reproducing

```
pytest tests/test_wsm6_workspace.py tests/test_wdm6_workspace.py
pytest tests/test_kernel_local_bounds.py -k workspace_kernels
GPUWM_NO_LOCAL_GPU=1 pytest tests/test_preflight.py -k "wsm6 or wdm6"
```

pytest tests/test_myj_workspace.py tests/test_myj_port.py \
       tests/test_myjurb_wrf471_parity.py tests/test_kernel_local_bounds.py
GPUWM_NO_LOCAL_GPU=1 pytest tests/test_pbl_vertical_bounds.py \
       tests/test_physics_vertical_preflight.py tests/test_kernel_frame_recordings.py
```

`test_the_frame_stays_under_the_default_stack` is the gate that does not
depend on the platform. `test_no_column_array_is_left_on_the_stack` is the
gate that reads the source and needs no GPU.

pytest tests/test_rrtmgp.py -k "retired_ceiling or deep_column or 256_level or in_solver_planck or 257_model"
pytest tests/test_kernel_local_bounds.py -k rrtmgp
GPUWM_NO_LOCAL_GPU=1 pytest tests/test_preflight.py -k rrtmgp_rte_is_priced
```
