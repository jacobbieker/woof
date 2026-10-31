# Partial-depth nest coupling operators: WRF v4.7.1 oracle

Gate (b) of plans/sorted-mapping-hummingbird.md Phase D3: the operators the
partial-depth nest (woof/core/partial_nest_run.py) couples its child with
are held, word for word, to WRF v4.7.1's own statements compiled with
gfortran.

## What is compiled

`extract.py` copies, after checking each file's and each block's sha256
(SOURCES.sha256; WRF tag v4.7.1, commit
f52c197ed39d12e087d02c50f412d90d418f6186, public domain):

| block | source | woof counterpart |
|---|---|---|
| `SINT`, `SINTB` | share/sint.F (whole file) | `nest_interp.sint` (nest.cu `sint_point`) |
| `interp_fcn_sint` | share/interp_fcn.F:874-993 | `nest_interp.sint` with `register_nest(..., wrapper="interp")` |
| `copy_fcn` | share/interp_fcn.F:1397-1742 | `nest_interp.copy_fcn` |
| `copy_fcnm` | share/interp_fcn.F:1747-1824 | `nest_interp.copy_fcnm` |
| `copy_fcni` | share/interp_fcn.F:1829-1906 | `nest_interp.copy_fcni` |
| `smoother`, `sm121`, `smdsm` | share/interp_fcn.F:3794-4014 | `nest_interp.smoother` |
| `relax_bdytend_core` | share/module_bc.F:1221-1427 | `partial_nest.cu` `nest_top_relaxation` (relaxation rows) |
| `spec_bdytend` | share/module_bc.F:1430-1549 | `partial_nest.cu` `nest_top_relaxation` (specified rows) |

No statement is edited.  `stubs.F90` supplies what the blocks `USE`:
`module_configure` (`nl_get_spec_zone`, `nl_get_feedback`,
`nl_get_smooth_option`, set by the driver; `grid_config_rec_type` with
`periodic_x`) and `module_wrf_error`.  The two module_bc.F routines are
wrapped in `MODULE wrf_bc`; the interp_fcn.F routines stay external.

Build: `bash build.sh WRF_SOURCE_ROOT BUILD_DIR` with
`pixi exec -s gfortran gfortran` (GNU Fortran 16.2.0, conda-forge),
`-O0 -ffp-contract=off -ffree-form`, REAL = real(4).  The operators call no
libm function.  The libmvec guard is the vert_nest oracle's: an -O0 object
may not reference glibc's `_ZGV*` vector libm, proven non-vacuous by an
-Ofast positive control.  On macOS (no glibc, no libmvec) the positive
control cannot produce a `_ZGV` symbol; build.sh records that in
libmvec-report.txt and fails the vacuous guard only on Linux.

## The cases (run_partial_nest.F90)

Inputs are generated in the driver (smooth fields plus a Park-Miller
perturbation, at coupled-field magnitudes) and written with WRF's outputs
in WRF's (i, k, j) order through ../urban_wrf471_oracle/oracle_io.F90.
Parent 24 x 20 x 4, child 3:1 at i/j_parent_start 7/6, 24 x 18 mass cells.

* `sint_{m,x,y}` -- interp_fcn_sint, imask 1, perturbation 60% of the
  structure so the SINTB limiter acts.  At ratio 3 the bdy wrapper woof
  registers the partial nest's planes and strips with has the interp
  wrapper's stagger offset (`max((3-1)/2, 1) = (3-1)/2`).
* `copy_fcn_{m,x,y}`, `copy_fcnm`, `copy_fcni` -- spec_zone 1.
* `smoother{1,2}_{m,x,y}` -- feedback 1, smooth_option 1 (sm121) and 2
  (smdsm).
* `strip_full_phi`, `strip_half_u`, `strip_half_theta` -- the TOP STRIP on
  WRF's south side, `periodic_x = .TRUE.` (b_limit 0, every i processed),
  spec_zone 1, relax_zone 6, spec_bdy_width 7.  WRF's i is woof's x, WRF's
  carried k is woof's y (kde = ny + 1, a 't' variable: ktf = ny), WRF's row
  j is strip row d + 1, row 0 the interface.  fcx/gcx are WRF's nested
  lbc_fcx_gcx order (default REAL, left to right) and equal
  `lateral_bc._weights(7, 1, 6, dt, 0, wrf_real=True)` word for word.
  For a half-level field (half = 1) row 0 is the parent's level above the
  interface, which the child does not carry: WRF's field row 1 is set to
  its own target (top_tend row 0 = 0), the relaxation's outer residual is
  then exactly zero, and woof compares rows 1..5 only.  The north side's
  rows (j >= 15 of 20) never reach the compared rows.

## Result

tests/test_partial_nest_wrf471_oracle.py, Metal (Apple, pixi env mlx),
2026-10-30: every operator equals WRF word for word -- sint (3 staggerings),
copy_fcn (3), copy_fcnm, copy_fcni, smoother (2 options x 3 staggerings),
and the top strip (full and half levels).  No ulp tolerance is used
anywhere.

Publish: copy BUILD_DIR/fixtures to tests/data/oracles/partial_nest/.

## D4: the interface face's vertical flux

`dyn_em/module_advect_em.F:4337-4350` (advect_scalar, `vert_order == 2`,
sha256 d7507c42...), wrapped in `wrf_vflux2` with no statement edited: the
second-order face flux `vflux(i,k) = rom*(fzm(k)*field(k) + fzp(k)*field(k-1))`
and `tendency(k) = tendency(k) - rdzw(k)*(vflux(k+1) - vflux(k))`.  Three
cases (`vflux2_theta`, `vflux2_qv`, `vflux2_u`) on a two-level column per
point -- level 1 the child's top half level, level 2 the parent's half level
above, face 2 the interface, Omega of either sign.  woof's
`nest_top_face_flux_all` (centered) and `nest_top_scalar_face_flux`
(kernels/partial_nest_d4.cu) equal WRF's vflux and tendency word for word;
the scalar entry point's outgoing-face cap (a donor never gives more than
its own value) is woof's, compared where it binds against `rom*q_top`.
The D3 fixtures are unchanged (the new cases are generated after them).
Metal, 2026-10-30: 24 tests pass, no tolerance.
