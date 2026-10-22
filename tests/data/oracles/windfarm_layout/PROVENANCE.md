# WRF v4.7.1 wind-farm layout oracle provenance

These fixtures pin how WRF's Fitch scheme places a turbine, reads its
table and interpolates the table. The woof functions under test are
`woof.windfarm.layout`, `woof.windfarm.wrf_llij` and
`woof.windfarm.fitch_tables`. WRF ran on four synthetic farms that woof laid
out and wrote as WRF inputs. `tests/test_windfarm_layout_wrf471_parity.py`
rebuilds each layout from its `deck/deck.json` and compares it against the
fixtures by raw float32 words. Max ULP is 0, CPU against WRF and GPU against
WRF and the CPU. Integer outputs must match exactly. No output is held out.

Upstream tag: `v4.7.1`.
Upstream commit: `f52c197ed39d12e087d02c50f412d90d418f6186`.
Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`pixi exec -s gfortran`).
C library: GNU libc 2.44 (the host's; the oracle's `cosf`, `sinf`, `tanf`,
`log10f` and `powf` are this library's).
Host: Linux 6.12 x86-64, little endian.

## What was compiled

Every upstream file below was used read-only and byte-unmodified, and
`build.sh` checks each hash before compiling:

```
0c777e66ffbb0602479cadcc0ab484769a140c4cb2b3e9a8f757597086f0c76b  share/module_llxy.F
83247add74780608e798bb62424533b18abf31f3da5aba6374de6fe2550060c3  phys/module_wind_fitch.F
5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062  share/module_model_constants.F
```

`share/module_llxy.F` is the same file as the v4.6.1 copy pinned by
`tools/llxy_wrf461_oracle`. `module_llxy.F` and `module_model_constants.F`
are compiled whole. `module_wind_fitch.F` cannot be built outside WRF
because it USEs module_configure and module_dm. `extract.py` therefore cuts
five contiguous line ranges from it, checking the first and last line of
each, and compiles them inside the harness's scaffolding:

| range | contents |
|---|---|
| 54-63 | the module data (`MAXVALS`, the arrays `nt`, `NKIND`, `ival`, `jval`, `hubheight`, `diameter`, `stc`, `stc2`, `cutin`, `cutout`, `npower`, `turbws`, `turbtc` and `turbpw`, and `correction_factor`) |
| 277-334 | `SUBROUTINE dragcof` |
| 399-477 | the body of `init_module_wind_fitch`'s per-turbine loop: read `lat lon nkind`, `map_init`, `map_set` anchored at `XLAT`/`XLONG(its,jts)`, `latlon_to_ij`, `nint` |
| 491-510 | the table read |
| 190-191 | `dragforce`'s `area` and `power1` |

The driver only opens files, counts the turbines (as lines 369-377 do) and
allocates (as lines 383-386 do). It calls `dragcof` with `dragforce`'s
argument list (lines 184-186) and writes the arrays. The tile is
`its = jts = 1`, a serial run, with `XLAT`/`XLONG(1,1)` taken from
`grid.txt`.

Build: `bash tools/windfarm_layout_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR`,
after `fetch_sources.sh WRF_SOURCE_ROOT`. Flags: `-O0 -cpp -ffree-form
-ffree-line-length-none`. Default REAL is real(4), as in WRF's `-DRWORDSIZE=4`
build. `libmvec-report.txt` shows no `_ZGV*` symbol in any `-O0` object, and
a positive control shows the check can see one. The objects call scalar
`powf`, `cosf`, `sinf`, `tanf` and `log10f`.

The harness files are hashed in `HARNESS.sha256`, the sources in
`SOURCES.sha256` and every fixture in `FIXTURES.sha256`. The test checks all
three.

## The decks

`make_deck.py` builds four Lambert grids: NH secant at 12 km and 1 km, SH
secant at 3 km, and tangent at 4 km. Each has 431 turbines:

- 400 at seeded random interior positions;
- 24 on exact half-cell projection coordinates, the edge where `nint`
  rounds;
- one cell holding two machines, with hub heights spread within the
  grouping tolerance.

Between them the decks use all 201 curves of the bundled turbine library.
That gives 69, 69, 69 and 5 WRF types. Each type gets dragcof queries at 0,
0.5, 30 and 40 m/s, at every table row and every row midpoint, one float32
ulp either side of cut-in and cut-out, and at 16 random speeds. The
`correction_factor` values are 1.0, 0.25, 1.0 and 0.5.

## Words compared (all four cases; max ULP in every column is 0)

| output | words | CPU vs WRF | GPU vs WRF | CPU vs GPU |
|---|---:|---:|---:|---:|
| lat, lon read (each) | 1,724 | 0 | (host only) | (host only) |
| XLAT/XLONG(1,1) | 8 | 0 | (host only) | (host only) |
| proj_info: cone, rebydx, rsw, polei, polej, hemi | 24 | 0 | (host only) | (host only) |
| ts_rx, ts_ry (each) | 1,724 | 0 | (host only) | (host only) |
| ival, jval, nkind (each, exact) | 1,724 | exact | (host only) | (host only) |
| hubheight, diameter, stc, stc2, npower, cutin, cutout (each) | 1,724 | 0 | upload exact | upload exact |
| turbws, turbtc, turbpw (each) | 40,000 | 0 | upload exact | upload exact |
| thrcof | 22,582 | 0 | 0 | 0 |
| powcof | 22,582 | 0 | 0 | 0 |
| tkecof | 22,582 | 0 | 0 | 0 |
| power1 | 22,582 | 0 | 0 | 0 |

In total, 234,496 words are compared against WRF. Placement and the table
read are one-off CPU setup and have no GPU version. On the GPU the layout's
arrays are a copy, and the test checks that the copy is raw-word equal to
the host for every array. `dragcof` and `power1` run on both CPU and GPU.

## Negative controls

- Placement through woof's binary64 WPS-convention transform, rounded to
  float32, differs from WRF's `ts_rx`.
- The thrust interpolation evaluated in binary64 and rounded once differs
  from WRF's real(4) answer.

## Deliberate departures (not values)

- WRF's outside-domain test `ival .lt. ids .and. ival .gt. ide` can never be
  true. woof instead refuses, by name, any turbine outside the mass grid or
  in its outermost row or column.
- WRF anchors each MPI patch at its own corner. woof places turbines once,
  as a serial run (`its = jts = 1`) does.
- The anchor `XLAT`/`XLONG(1,1)` is an input to the placement, not
  something this oracle grades. It is the float32 word woof itself writes
  as `XLAT`/`XLONG`: `woof.static.lambert`'s binary64 transform, rounded
  once. `grid.txt` gives WRF that same word, so the oracle grades
  `map_set`, `llij_lc` and `nint` given woof's grid. A geogrid-made
  `XLAT` can differ from woof's by a few float32 ULP (the float32-storage
  bound `woof.static.lambert` documents). A WRF run fed geogrid's
  `geo_em` instead of woof's `XLAT` would therefore start from a different
  anchor.
- On the CPU, `dragcof` raises numbers to powers with the pinned glibc
  2.39 `powf` (`woof.core.noahmp_libm`). The GPU uses that function's
  device transcription, `gfk_pow`. Both match the oracle's host glibc 2.44
  `powf` on every word here. Placement uses the host C library through
  ctypes, the same library the oracle links.
- WRF reads lat/lon decimal text straight into real(4). woof rounds its
  binary64 table values to float32 and writes them as shortest round-trip
  decimals, so WRF reads exactly woof's words. The `lat` and `lon` rows
  above check this.
