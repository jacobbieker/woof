# MPAS-Seaice v8.4.2 column-package oracle, patched: the vertical salinity solved (`zsal_solve`)

Written by `WOOF_ORACLE_PATCH=zsal_solve bash tools/mpas_seaice_oracle/build.sh`
and `make_column_fixture.py --cases zsal_solve` on pb (Linux 6.12 x86_64,
glibc 2.44), 2026-11-21.

This is not v8.4.2's behaviour: it is the reference for the opt-in
correction `zsal_solve` (`woof.hex.mpas_seaice.column_v842.CORRECTIONS`).
v8.4.2 never solves the vertical salinity (`../mpas_seaice_bgc_zsal/PROVENANCE.md`);
its default words are that fixture's.

- Source: MPAS-Model v8.4.2 (`f34984b2`), checked against
  `SOURCES.sha256`, then copied into the build directory and patched with
  `tools/mpas_seaice_oracle/zsal_solve.patch`, which only
  - assigns `solve_zsal = solve_zsal_in` in `colpkg_init_parameters`
    (`ice_colpkg.F90`), and
  - copies `verticalSalinity` and `verticalSalinityCell` to and from the
    tracer arrays when the vertical salinity is on without the vertical
    tracers (the four `set/get_cice_biogeochemistry_tracer_array_*`
    routines of `mpas_seaice_column.F`), as they do with them.
  Everything else is compiled verbatim, as for the unpatched oracle.
- Cases: `BGC_ZSAL_SOLVE_CASES` (`woof_v842_corrections = zsal_solve`):
  - the init (`sss*salt_loss` on every owned cell and category, no Rayleigh
    criterion), its restart, three layers and four categories;
  - `column_biogeochemistry` (`_bgc_state`, `_zsal_state`): new ice, a brine
    height above `maxhbr` times the ice (cell 8), thin ice with and without
    the Rayleigh criterion (cells 1 and 3), runoff and flooding; at half an
    hour with a molecular sublayer and other drainage scales; at 200 s (one
    `dts_b` substep); four layers and three categories;
  - `bgc_zsal_solve_abort_remap`: `compute_microS`'s remap stop is checked
    (`ice_colpkg.F90:5118`), so cell 5's middle category aborts at
    `indexToCellID` 5;
  - `bgc_zsal_solve_abort_max_salinity`: cold, salty, old ice in cell 1's
    first category, where `solve_S_dt`'s `bSin(k) > max_salin` stops;
  - BL99's vertical thermodynamics, winter and melting, with cell 5 inside
    the absorbed-shortwave band `dTemp = 0.1` moves.
- Kept as the column package writes them: `rayleighCriteriaReal` is only
  ever zeroed (no effect: `solve_S_dt` forces the criterion where its
  convective branch runs, `hbrin > thinS = Ra_c`), and
  `totalVerticalSalinity` is summed into, never reset.
- Files and libm as in `../mpas_seaice_bgc_zsal/PROVENANCE.md`; `x**2`,
  `x**3` and `**exp_h` are libgcc's `__powidf2` at -O0, transcribed as
  `x*x` and `x*(x*x)`.
