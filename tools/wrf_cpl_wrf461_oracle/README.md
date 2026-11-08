# WRF ocean-coupler exchange oracle, WRF v4.6.1

```
bash tools/wrf_cpl_wrf461_oracle/build.sh <WRF_SOURCE_ROOT> <BUILD_DIR>
```

Builds `cpl_snd2` and `cpl_rcv`, cut byte for byte from the pinned
`frame/module_cpl.F`, inside `cpl_harness.F90`, drives them over 64 columns
and two external ocean domains (`run_cpl.F90`), and writes `wrf-cpl.csv`.
The fixture lives in `tests/data/oracles/wrf_cpl/` (see `PROVENANCE.md`
there); the gate is `tests/test_wrf_cpl_wrf461_parity.py`, which measures
`woof/ocean/wrf_cpl.cu` against it.

These two routines are the whole of WRF's own side of its NEMO coupling: the
fields it sends every step and how it merges what the ocean sends back.  The
OASIS3-MCT library between them (time averaging, remapping) and NEMO itself
are not part of this oracle; see `docs/coupling/WRF-HYDRO-NEMO.md`.
