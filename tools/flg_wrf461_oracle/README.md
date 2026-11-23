# WRF Fu-Liou-Gu oracle and CUDA forecast adapter

`woof.core.flg.FLGColumn` executes the complete stock `rad_all` call.
`woof.core.flg_radiation.FLGRadiation` owns the CUDA `RAD_FLG` packing,
solar geometry, ozone interpolation, column and history publication.
The forecast selects `ra_lw_physics=7, ra_sw_physics=7`; its default is
fast numerics. Strict proof opts in with `GPUWM_WRF_EXACT=1`.

The public WRF 4.6.1 source pin is
`d66e442fccc04111067e29274c9f9eaccc3cef28`. Its FLG file is byte-identical
to WRF 4.7.1. All 198 coefficient entries come from that public source;
see `woof/data/wrf_radiation/flg/PROVENANCE.md` for licence and hashes.

Generate the CUDA column and warp-interleaved global workspace. The
workspace generator also applies storage-only passes (thread-local small
scratch, no storage for aerosol-only arrays, the register-window `qcfel`
band solve, the fused `qftisf` angle sweeps and band-level `comscp` phase
moments); none changes an operation or its order:

```sh
python tools/flg_wrf461_oracle/generate_column_kernel.py \
  WRF/phys/module_ra_flg.F woof/data/wrf_radiation/flg/manifest.json \
  woof/core/kernels/flg_column_generated.cuh
python tools/flg_wrf461_oracle/generate_workspace_kernel.py \
  woof/core/kernels woof/core/kernels/flg_workspace_generated.cuh \
  --receipt workspace-layout.json
```

For a real native replay, instrument the native `rad_all` caller to record
its already-packed inputs immediately before the call and its eleven
output vectors immediately after. The binary record is int32 `nv,nv1`,
then float32 albedo[6], cosine, solar constant, skin temperature,
emissivity[12], four interface profiles in `FLGColumn.profile_names`
order, eight layer profiles in `layer_names` order, and the eleven outputs
in `output_names` order. No inputs are synthesized by the replay reader.

```sh
GPUWM_WRF_EXACT=1 python tools/flg_wrf461_oracle/check_column.py \
  unused-library replay.json --capture flg-real-columns.bin
GPUWM_WRF_EXACT=1 python tools/flg_wrf461_oracle/check_adapter.py \
  libflg_oracle.so adapter.json
```

The real 32-layer deck captured twelve columns from a successful strict
native WRF run, including four night, four day and four cloudy day columns.
All 4,320 words across the eleven outputs match exactly. The separate
synthetic adapter check compares nine published fields with pristine
`RAD_FLG`, including heating, surface and direct-normal fluxes; it also
trips any per-call `cupy.asnumpy` transfer after initialization.

Build the pristine standalone reference, repeat deck and mutation control:

```sh
bash tools/flg_wrf461_oracle/build.sh WRF ORACLE_BUILD
GPUWM_WRF_EXACT=1 python -m pytest -q \
  tests/test_flg_columns.py tests/test_flg_cuda.py
python -m pytest -q tests/test_flg_wiring.py tests/test_flg_integration_gpu.py
```

Run all CUDA commands through the machine's shared GPU helper. The default
arm is fast arithmetic and is not asserted bitwise. The primitive,
cloud-optics, gas and stream tests remain useful component regressions;
they do not substitute for the captured native replay or a real forecast.

The full forecast proof uses stock inputs with `woof go INPUT_DIRECTORY`.
Its three-hour native and fast GPU histories were rendered through
`woof.rustwx` and `rw_wrfbatch` at 13Z, 14Z and 15Z with station overlays.
A missing cloud-cover selector was recorded; temperature and MSLP/wind
maps rendered successfully. Computed OLR is published by this adapter;
stock WRF excludes FLG from its generic OLR history assignment. Computed
per-column TSWUP also avoids native RAD_FLG's whole-array USWTOP assignment
inside its column loop. Neither diagnostic is claimed as bitwise stock
history proof. General skill and mixed-pair native forecasts remain
unqualified.

## 101 to 256 layers: the patched-nvx oracle

Stock WRF's `PARA_FILE` fixes `nvx = 100`, so stock FLG has no column
deeper than 100 layers to compare with. The CUDA sources carry that bound
as `FLG_NVX` (the generators emit the macro, not the literal; its `#ifndef`
default is the stock 100, so every column of at most 100 layers compiles
stock WRF's bound and the default-tier sources preprocess to the previous
words). A 101 to 256 layer column compiles with `-D FLG_NVX=256` and is
verified against `module_ra_flg.F` with only `nvx` raised to 256:
`nvx256.patch`, SHA256
`b1e28863f9d089db2574252f268a2b8c507c8a8f8df3550b0d00117d2ca0d788`, recorded
with its reason in `woof/data/wrf_radiation/flg/PROVENANCE.md`. `build.sh`
still refuses an edited source; `build_nvx256.sh` copies the pristine file,
applies the recorded patch to the copy and refuses any other source or
patch bytes. It builds the pristine and patched libraries side by side.

```sh
bash tools/flg_wrf461_oracle/build_nvx256.sh WRF NVX_BUILD
python tools/flg_wrf461_oracle/make_nvx256_fixture.py NVX_BUILD \
  tests/fixtures/flg/nvx256-columns.npz
GPUWM_WRF_EXACT=1 python -m pytest -q tests/test_flg_256_levels.py
```

The fixture script first proves the patch inert where stock WRF runs: both
libraries return the same words for every output of a 24-layer and a
100-layer deck. The committed fixture was built with gfortran 13.3 on glibc
2.39 (the strict numerics' default referee libc), from a synthetic six-column
256-layer deck to 1 hPa (clear, overcast, broken and mixed-phase cloud; four
day and two night columns): 16,944 output words.
