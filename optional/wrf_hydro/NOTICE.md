# WRF-Hydro opt-in: attribution and provenance

Everything under `optional/wrf_hydro/` is an **opt-in**: off by default, in no
wheel (`metal-woof`, `recast-woof`), in no default build, and excluded from the
release snapshot (`RELEASE-EXCLUDE.txt`).  It loads only through
`woof.optin.load("wrf_hydro")` with `WOOF_ENABLE_WRF_HYDRO=1` in a source
checkout.  The project owner approved adding it on those terms on 2026-11-07.

## Licence

The code here is derived from WRF-Hydro and is distributed under UCAR's
WRF-Hydro licence, whose full text is `LICENSE` in this directory (copied
verbatim from NCAR/wrf_hydro_nwm_public `v5.4.0`,
`49b0fba3e859ef5288b06ec430621be03193f49e`, `LICENSE.txt` sha256
`434cc6a8692405f600a3b7dd45d6c3d1af0848543d0ffd2e9cb886037b6be41b`).  Note
its clause 1: **you may not sell, license or transfer for a fee the Software,
or any work that in any manner contains the Software.**  The rest of WOOF is
Apache-2.0; this directory is not.

Notice required by clause 9:

> The source of this material is the Research Applications Laboratory at the
> National Center for Atmospheric Research, a program of the University
> Corporation for Atmospheric Research (UCAR) pursuant to a Cooperative
> Agreement with the National Science Foundation; ©2007 University
> Corporation for Atmospheric Research. All Rights Reserved.

Scholarly works derived from it must display:

> The WRF-Hydro modeling system was developed at the National Center for
> Atmospheric Research (NCAR) through grants from the National Aeronautics
> and Space Administration (NASA) and the National Oceanic and Atmospheric
> Administration (NOAA). NCAR is sponsored by the United States National
> Science Foundation.

No endorsement by UCAR or NCAR is implied (clause 5).  This is a modified
work: a CUDA transcription with a Python launcher, not the original source.

## Source

The WRF-Hydro tree WRF 4.6.1 ships under `hydro/` (WRF pinned at
`d66e442fccc04111067e29274c9f9eaccc3cef28`), the reference WOOF already
holds its WRF physics to.

| WOOF file | WRF-Hydro source |
| --- | --- |
| `woof_wrf_hydro/subsurface.cu`, `subsurface.py` | `hydro/Routing/Noah_distr_routing_subsurface.F90` (subsurfaceRouting, FINDZWAT, ROUTE_SUBSURFACE1, SUBSFC_RTNG); `hydro/Routing/Noah_distr_routing.F90` (GETSUB8, GETSUB8DIR) |
| `woof_wrf_hydro/grid_transform.cu`, `grid_transform.py` | `hydro/Routing/Noah_distr_routing.F90` (disaggregateDomain); `hydro/HYDRO_drv/module_HYDRO_drv.F90` (aggregateDomain) |
| `woof_wrf_hydro/overland.cu`, `overland.py` | `hydro/Routing/Noah_distr_routing_overland.F90` (OverlandRouting, OV_RTNG, ROUTE_OVERLAND1); `hydro/Routing/Noah_distr_routing.F90` (GETMAX8DIR, GET8DIR) |
| `oracle/` | build harnesses and drivers (`build.sh`, `build_grid_transform.sh`, `build_overland.sh`); they compile the pinned files at build time and copy none of them |

`subsurface.cu` is preceded at load time by WOOF's
`woof/core/kernels/glibc_flt32.cuh` (glibc's powf words, MIT-licensed Arm
code, see that file); nothing of it is copied here.

## Oracle

`oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR` compiles the pinned routines
unchanged (`-O0`, no MPP_LAND), drives six routing calls over a 20 x 16 grid
with hashed inputs, a valley, a saturated sink block and a dry column, and
writes `oracle/fixtures/`.  Toolchain: gfortran 13.3.0, glibc 2.39 (Ubuntu
24.04 container), 2026-11-08.  The gate is
`tests/test_wrf_hydro_subsurface_parity.py`.
