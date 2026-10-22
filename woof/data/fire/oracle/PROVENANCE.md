# WRF v4.7.1 fire-mesh oracle provenance

These fixtures hold woof's fire-mesh terrain (`ZSF`) and slope (`DZDXF`,
`DZDYF`) to WRF v4.7.1's own fire code, word for word. They also hold the
LANDFIRE pixel each fire cell takes to WRF's own map projection.
`tests/test_fire_mesh_wrf471_parity.py` asserts max ULP 0, which is raw-word
equality, for the CPU body (`woof/static/fuel.py`) and the GPU body
(`woof/core/kernels/fire_mesh.cu`) against the oracle and against each other.
It asserts exact equality for every pixel index. No word is held out.

Upstream tag: `v4.7.1`.
Upstream commit: `f52c197ed39d12e087d02c50f412d90d418f6186`.
Compiler: GNU Fortran (conda-forge gcc 16.2.0-7) 16.2.0 (`TOOLCHAIN.txt`).
C library: GNU libc 2.44.
Host: Linux x86-64, little endian.

## What is compiled

The sources are byte-unmodified. Their SHA-256 pins are in
`tools/fire_mesh_wrf471_oracle/SOURCES.sha256` and are copied here as
`SOURCES.sha256`. They are the three fire files below and
`share/module_llxy.F` (see the next section). `build.sh` verifies them
before anything compiles.

- `phys/module_fr_fire_util.F` is compiled whole. It provides
  `interpolate_z2fire`, `continue_at_boundary` and `interpolate_2d`.
  - The only stub is `wrf_error_stub.F90`, which supplies the two
    `module_wrf_error` entries and the two process-count queries the
    module's array writers name. None of these is reached.
- `phys/module_fr_fire_model.F:194-200` and `:207-208` are cut verbatim by
  `extract.py`, which checks the sha256 of each cut.
  - Lines 194-200 are the `continue_at_boundary` call that continues ZSF one
    node past the mesh.
  - Lines 207-208 are the centred differences
    `(fp%zsf(i+1,j)-fp%zsf(i-1,j))/(2.*fdx)` and their y twin.
  - In WRF v4.7.1, `DZDXF`/`DZDYF` are geo_em fields, and these expressions
    exist to check them. They are WRF's only statement of the fire-mesh
    slope, so the oracle evaluates them as the slope itself.
- `phys/module_fr_fire_driver.F:566-567` is cut the same way: the fire-mesh
  spacing `dxf=dx/ir`.

Build command:

```sh
bash tools/fire_mesh_wrf471_oracle/fetch_sources.sh WRF_SOURCE_ROOT
bash tools/fire_mesh_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
cp -r BUILD_DIR/fixtures/* woof/data/fire/oracle/
```

The flags are `-O0 -ffree-form -ffree-line-length-none`, with default
REAL(4) (WRF's `RWORDSIZE=4`) and no OpenMP. The routines call no libm. No
-O0 object references a vector libm symbol, and the positive control emitted
`_ZGVbN4v_expf` (`libmvec-report.txt`). The fixture and harness hashes are in
`FIXTURES.sha256` and `HARNESS.sha256`.

## Cases

`run_fire_mesh.F90` calls `interpolate_z2fire` the way WRF-SFIRE calls it for
every atmosphere-to-fire field (`flag_z0 = 0`, no clamp). It runs on one tile
that is the whole domain, with the fire memory one node past each end.

| case | atm grid | ratio | dx, dy (m) | what it reaches |
|---|---|---|---|---|
| `hills_even` | 7 x 5 | 4, 4 | 1000, 1000 | even ratio: no node on a cell edge |
| `odd_ratio` | 6 x 4 | 3, 3 | 1500, 1500 | odd ratio: every edge node written twice, the later write kept |
| `coast` | 8 x 6 | 5, 2 | 333.3, 500 | unequal ratios and spacings, sea inside the domain |
| `unit_ratio` | 5 x 6 | 1, 1 | 3000, 3000 | the fire mesh is the atmospheric grid |
| `fine_ratio` | 4 x 3 | 10, 10 | 1000, 1000 | WRF-SFIRE's usual refinement |
| `below_sea` | 3 x 2 | 2, 7 | 250, 125 | terrain below sea level, the smallest domain |
| `coast_edge` | 6 x 5 | 3, 4 | 750, 900 | a shore on the domain edge |

Each case directory holds `zs` (the input), `zsf`, `dzdxf`, `dzdyf`, `ir`,
`jr`, `dx`, `dy`, `dxf` and `dyf`, in Fortran index order.

## Result

| field | words | CPU vs GPU | CPU vs oracle | GPU vs oracle |
|---|---:|---:|---:|---:|
| ZSF | 2930 | 0 ULP | 0 ULP | 0 ULP |
| DZDXF | 2930 | 0 ULP | 0 ULP | 0 ULP |
| DZDYF | 2930 | 0 ULP | 0 ULP | 0 ULP |

The GPU result was measured on an RTX A6000, CuPy 14.2.0, with NVRTC options
`-std=c++17 -fmad=false --ftz=false`, compiled through
`compile_using_nvrtc` (not `cupy.RawModule`, which appends `-ftz=true`).

What the oracle settles:

- **The edge band.** WRF continues the atmospheric terrain one cell past
  the domain linearly, `EX = 2a - b` with bias 0. It then interpolates the
  outer half-cell band between the continued and edge values. The corners
  come from the diagonal.
- **Terrain below zero at a shore.** Terrain that rises from a shore on the
  domain edge therefore continues below zero past it: `coast_edge` reaches
  -28.125 m. WRF clamps only the roughness call (`flag_z0 = 1`), and woof
  keeps WRF's value.
- **Which write wins.** For odd ratios a node on a cell edge is written by
  both cells, and the later write in WRF's loop order is kept, which is the
  highest covering cell. `wrf_refinement_axis` does the same.

## LANDFIRE pixel placement (`llxy_fire_mesh/`)

The fuel reader places each fire-cell centre through the grid's own
`ij_to_latlon` (woof's transcription of WRF's module_llxy). It then goes from
geographic coordinates into the raster's CRS through PROJ, and takes the
pixel it falls in. This is how WPS geogrid samples a source, and it serves
every projection woof grids have and every nest.

`tools/fire_mesh_wrf471_oracle/build_llxy_fire.py` places the same centres
through WRF v4.7.1 `share/module_llxy.F` `ij_to_latlon`.
- That file is byte-identical in v4.6.1 and v4.7.1 (sha256
  `0c777e66ffbb0602479cadcc0ab484769a140c4cb2b3e9a8f757597086f0c76b`), and
  `tools/llxy_wrf461_oracle` builds it. It is pinned in this oracle's
  `SOURCES.sha256` beside the fire sources.
- `llxy_fire_mesh/FIXTURES.sha256` and `llxy_fire_mesh/HARNESS.sha256` hold
  the fixture's hashes and those of the scripts that wrote it, and the parity
  test checks both.
- This oracle uses the same stubs, flags and driver (binary64,
  `-O0 -ffp-contract=off`).
- A nest is placed the way WPS geogrid and `ProjectedGrid.nest` place it.
  The parent's `ij_to_latlon` at the child's (1, 1) mass point becomes the
  child's known point, and the oracle computes that point itself in a first
  pass.

| case | projection | latitude | atm grid | sr | cells |
|---|---|---:|---|---|---:|
| `lc_nh_conus` | Lambert secant | 39.7 | 6 x 5 | 3 x 3 | 270 |
| `lc_nh_boreal` | Lambert secant | 62.5 | 5 x 4 | 4 x 2 | 160 |
| `lc_sh_secant` | Lambert secant | -33.9 | 5 x 5 | 3 x 3 | 225 |
| `lc_sh_tangent` | Lambert tangent | -42.0 | 4 x 4 | 2 x 2 | 64 |
| `merc_nh` | Mercator | 19.6 | 4 x 4 | 5 x 5 | 400 |
| `merc_sh` | Mercator | -8.5 | 5 x 4 | 3 x 3 | 180 |
| `merc_equator` | Mercator | 0.5 | 4 x 5 | 2 x 2 | 80 |
| `ps_nh` | polar stereographic | 64.8 | 5 x 5 | 3 x 3 | 225 |
| `ps_sh` | polar stereographic | -77.85 | 4 x 4 | 2 x 2 | 64 |
| `nest_lc_nh` | Lambert, ratio-3 nest of `lc_nh_conus` | 39.7 | 6 x 6 | 2 x 2 | 144 |
| `nest_ps_sh` | polar, ratio-2 nest of `ps_sh` | -77.9 | 4 x 4 | 3 x 3 | 144 |

Woof grids are Lambert, Mercator and polar stereographic
(`woof.static.projection.implemented_projections`). Regular and rotated
latitude-longitude are not woof domain projections (`latlon_blocker`), so
there is nothing for the fuel reader to place on them.

Results, all 1956 cells:

- **Latitude and longitude.** woof's words equal module_llxy's at 0 ULP
  (binary64), 1956 + 1956 words. This holds with the CPU preprocessing
  bridge's native libm and with the scalar fallback.
- **Pixel indices, geographic raster.** The pixel WRF's words fall in equals
  the pixel `landfire_pixel_indices` takes, in every case.
- **Pixel indices, the `[static.highres]` overlay.** On a geographic raster
  whose words are their pixel's index, the overlay's default reprojection
  takes the same pixel as the fuel reader. That reprojection is the native
  static-fields library, with its own projection code, nearest neighbour
  onto a grid whose mass points are the fire cells. The comparison covers
  the 1796 cells of the ten isotropic cases (the anisotropic fire mesh is no
  woof grid). It holds for all but one cell, which is held out and pinned
  (`OVERLAY_HELD_OUT`):
  - The cell is `lc_nh_conus` (10, 10). Its WRF position is 2.5e-4 pixel
    (5 mm) from a row edge, and the overlay, whose placement differs from
    module_llxy by millimetres, takes the next row.
  - The test requires every held-out cell to lie within 1e-3 pixel of an
    edge and to be one pixel off.
- **The GDAL warper.** The overlay's `WOOF_STATIC_PYTHON=1` workaround route
  disagrees at pixel edges more often (7 of 270 cells on an Albers raster,
  with or without `tolerance=0`). It is a reported workaround and not
  compared.
