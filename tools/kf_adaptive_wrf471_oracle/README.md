# Kain-Fritsch under a varying step, WRF v4.7.1

Grades woof's KF driver seam (the cumulus cadence, the NCA hold, the
per-step RAINC/NCA clock) against WRF v4.7.1 when `dt` changes from step to
step, as it does under `use_adaptive_time_step`.

## What runs

`run_kf_adaptive_oracle.F90` steps a frozen three-column state through the
`dt` sequence in `case.py` (110 steps, 30-96 s, one CFL cut). Per step:

1. the adaptive `run_param` test of `phys/module_cumulus_driver.F:803-857`:
   the scheme runs on step 1 and whenever `curr_secs >= cudtacttime`;
2. on those steps, `KF_eta_CPS` from the byte-unmodified
   `phys/module_cu_kfeta.F` (SHA-256 `e6376c2d...`, identical in v4.6.1);
3. `advance_ppt` (`phys/module_physics_addtendc.F:2289-2293, 2346-2380`):
   `RAINC += PRATEC*DT`, the KFETASCHEME expiry (`NINT(NCA/DT) .LE. 1`
   zeroes the held tendencies) and `NCA -= DT`.

Steps 1 and 3 are transcribed, because the routines that hold them need the
whole WRF registry to compile. `build.sh` pins all three source files.

woof's side (`case.run_woof`) is its real `PhysicsDriver`: the production
`KainFritsch` adapter, `_run_cumulus` and `_advance_cumulus_clock`. The KF
inputs are pinned to the case's float32 words through the driver's
`prepared_physics_atmosphere` seam, and the scheme is called on WRF's
`run_param` steps through `cumulus_due_override`, as the adaptive clock
does.

## Result

With `GPUWM_WRF_EXACT=1`, NCA, RAINC, PRATEC and RAINCV match WRF word for
word on all 110 steps. The held tendencies are within 1 ulp (the mass
couple/decouple round trip). The default build matches NCA exactly and the
rest to 6e-6 relative.

Before the fix, the driver skipped the per-step clock on almost every
adaptive step. On this case that gave 0.26 mm of RAINC where WRF has 76 mm.

## Rerun

```bash
# WRF v4.7.1 source; gfortran (any recent version: conda-forge gcc 16.2 made the fixture)
pixi run -e default python -m tools.kf_adaptive_wrf471_oracle.case case.bin
bash tools/kf_adaptive_wrf471_oracle/build.sh WRF-4.7.1 build case.bin oracle-O0.out
cmp case.bin tests/data/oracles/kf_adaptive/case.bin
cmp oracle-O0.out tests/data/oracles/kf_adaptive/oracle-O0.out
WRF_SOURCE_ROOT=WRF-4.7.1 pixi run -e default python -m pytest tests/test_kf_adaptive_wrf471_oracle.py
```

`case.bin` is written from the default build's `_prepare_atmosphere`, so
regenerate it under `GPUWM_WRF_EXACT=0`.
