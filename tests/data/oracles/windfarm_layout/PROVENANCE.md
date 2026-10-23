# WRF v4.7.1 wind-farm layout oracle provenance

These fixtures pin how WRF's Fitch scheme places a turbine, reads its
table and interpolates the table. The woof functions under test are
`woof.windfarm.layout`, `woof.windfarm.wrf_llij` and
`woof.windfarm.fitch_tables`. WRF ran on eleven synthetic farms that woof laid
out and wrote as WRF inputs. The farms cover Lambert, Mercator and polar
stereographic domains, including nests up to three levels deep. `tests/test_windfarm_layout_wrf471_parity.py`
rebuilds each layout from its `deck/deck.json` and compares it against the
fixtures by raw float32 words. Max ULP is 0, CPU against WRF and GPU against
WRF and the CPU. Integer outputs must match exactly. No output is held out.

Upstream tag: `v4.7.1`.
Upstream commit: `f52c197ed39d12e087d02c50f412d90d418f6186`.
Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`pixi exec -s gfortran`).
C library: GNU libc 2.44 (the host's; the oracle's `cosf`, `sinf`, `tanf`,
`logf`, `log10f` and `powf` are this library's).
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
are compiled whole, so `map_set`, `set_lc`, `set_ps` and `set_merc`, and
`latlon_to_ij` with `llij_lc`, `llij_ps` and `llij_merc`, are WRF's own
object code, not cuts. `module_wind_fitch.F` cannot be built outside WRF
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
`powf`, `cosf`, `sinf`, `tanf`, `logf` and `log10f`.

The harness files are hashed in `HARNESS.sha256`, the sources in
`SOURCES.sha256` and every fixture in `FIXTURES.sha256`. The test checks all
three.

## The decks

`make_deck.py` builds 11 domains, one case each. Every domain is placed from
its own corner and its own `dx`, as WRF places a nest:

| case | projection | where | nest chain |
|---|---|---|---|
| `lc_nh_12km` | Lambert, secant | 33-38N, both sides of stand_lon -97.5 | root |
| `lc_ladder_d02_3km` | Lambert, secant | 34-36N | 12 km root, then 3 km (ratio 4) |
| `lc_ladder_d03_1km` | Lambert, secant | 35N | 12, then 3, then 1 km (ratios 4, 3) |
| `lc_sh_3km` | Lambert, secant, SH | 29-31S | root |
| `lc_tangent_4km` | Lambert, tangent | 44-46N | root |
| `merc_tropics_9km` | Mercator | 3-7N | root |
| `merc_nest_d02_3km` | Mercator | 4-6N | 9 km root, then 3 km (ratio 3) |
| `merc_dateline_6km` | Mercator, SH | 16-19S, across 180 | root |
| `ps_nh_6km` | polar stereographic | 84.0-88.6N, lon -98 to 14 about stand_lon -40 | root |
| `ps_nh_nest_d02_2km` | polar stereographic | 85.7-87.1N | 6 km root, then 2 km (ratio 3) |
| `ps_sh_dateline_6km` | polar stereographic, SH | 75-80S, across 180 | root |

Each case has 331 turbines:

- 300 at seeded random interior positions;
- 24 on exact half-cell projection coordinates, where `nint` rounds;
- one cell holding two machines, with hub heights 0.2 m apart (each is
  its own WRF type and its own group).

The 201 curves of the bundled turbine library are dealt across the cases,
22 or 23 WRF types per case. Each type gets dragcof queries at:

- 0, 0.5, 30 and 40 m/s;
- every table row and every row midpoint;
- one float32 ulp either side of cut-in and of cut-out;
- 16 random speeds.

The `correction_factor` (WRF's `windfarm_tke_factor`) is 0.25, 0.5 or 1.0,
depending on the case.

Regular and rotated latitude-longitude (`map_proj = 6`) are not covered,
because neither side can place on them. WRF v4.7.1's
`init_module_wind_fitch` sets up only `PROJ_MERC`, `PROJ_LC` and `PROJ_PS`
(lines 408-438); its Cassini and rotated-lat-lon arms are commented out,
so `latlon_to_ij` would stop on an unset projection. woof builds no
latitude-longitude domain. woof refuses such a grid by name.

## Words compared (all 11 cases; max ULP in every column is 0)

| output | words | CPU vs WRF | GPU vs WRF | CPU vs GPU |
|---|---:|---:|---:|---:|
| lat, lon read (each) | 3,641 | 0 | host only | host only |
| XLAT/XLONG(1,1) | 22 | 0 | host only | host only |
| proj_info: cone, rebydx, rsw, polei, polej, hemi, dlon, stdlon, truelat1, truelat2, lat1, lon1, knowni, knownj, dx | 165 | 0 | host only | host only |
| ts_rx, ts_ry (each) | 3,641 | 0 | host only | host only |
| ival, jval, nkind (each, exact) | 3,641 | exact | host only | host only |
| hubheight, diameter, stc, stc2, npower, cutin, cutout (each) | 3,641 | 0 | upload exact | upload exact |
| turbws, turbtc, turbpw (each, the types used) | 24,500 | 0 | upload exact | upload exact |
| thrcof | 26,279 | 0 | 0 | 0 |
| powcof | 26,279 | 0 | 0 | 0 |
| tkecof | 26,279 | 0 | 0 | 0 |
| power1 | 26,279 | 0 | 0 | 0 |

In total, 229,777 words are compared against WRF. `proj_info` members that a
projection does not set keep `map_init`'s -999.9. `map_init` leaves `dlon`
undefined, so it is written and compared only on Mercator, and as 0 on the
other projections.

Placement and the table read are one-off CPU setup and have no GPU version.
On the GPU the layout's arrays are a copy, and the test checks that the copy
is raw-word equal to the host for every array. `dragcof` and `power1` run on
both CPU and GPU.

The test also walks WRF's own turbines `k = 1..nt` through the dumped
`ival`, `jval` and `nkind`. In every cell, the runs of consecutive equal
types must be the layout's groups, in the same order and with the same
counts; 3,616 groups match. This is the order a Fitch kernel must add them
in to reproduce WRF's words.

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
