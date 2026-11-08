# Noah-MP WATER under WRF_HYDRO oracle -- provenance

`noahmp-water-hydro.csv` is `tools/noahmp_wrf461_oracle/build_water_hydro.sh`:
WRF 4.6.1's `module_sf_noahmplsm.F` (phys/noahmp submodule 848f54ad,
sha256 `bd592a5b...`, the file `PROVENANCE-water.md` pins) compiled at `-O0`
with `-DWRF_HYDRO`, which adds `QINSUR = QINSUR+sfcheadrt/DT*0.001` at
:6173-6175 and the `sfcheadrt`/`WATBLED` arguments.  The harness is
`run_water.F90`, unchanged, plus the edits `patch_run_water_hydro.py`
applies at build time: the two arguments in the state and in the WATER call,
a ponded head per case derived from the case name (0 to 5.5 mm), and their
rows in the fixture.  `WATBLED` is passed as 0.0 and unread (opt_tdrn = 0).
Same accessibility lift and libmvec refusal as `build_water.sh`.  The
`noahmp-water-hydro-sha256sums.txt` beside it pins the source, the harness,
the patcher, the build script and the fixture.
