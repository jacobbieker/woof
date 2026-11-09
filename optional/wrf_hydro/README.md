# WRF-Hydro routing (opt-in)

UCAR-licensed; read `NOTICE.md` and `LICENSE` before using or redistributing.

```
WOOF_ENABLE_WRF_HYDRO=1 pixi run -e default python -m pytest tests/test_wrf_hydro_subsurface_parity.py
```

```python
from woof import optin
hydro = optin.load("wrf_hydro")          # refuses by name unless enabled
routing = hydro.SubsurfaceRouting(...)
```

Numerics: the opt-in units follow the rest of WOOF (`woof/numerics.py`).
In the default `fast` mode the compile boundary drops their `--fmad=false`, so
NVRTC fuses multiply-adds and results can sit a few ULP from WRF-Hydro (the
diffusive channel's DIFFUSION does).  `strict` (`GPUWM_NUMERICS=strict` or
`GPUWM_WRF_EXACT=1`, which the test battery's conftest sets) keeps the
no-contraction request, and every WRF-Hydro and NEMO parity test's bitwise
claim holds there.

Ported: subsurface and overland routing (`rt_option = 1`); the LSM <-> routing
grid disaggregation and aggregation; reach channel routing (Muskingum and
Muskingum-Cunge, channel_option 1 and 2); the diffusive wave on nodes
(channel_option 3) with level-pool lakes; the groundwater bucket
(GWBASESWCRT 1 and 2).  The coupled driver (`woof_wrf_hydro.coupled`:
HYDRO_ini from the domain files and wrf_cpl_HYDRO/HYDRO_exe, reached from a
forecast with `--wrf-hydro HYDRO.NAMELIST`, see `woof/wrf_hydro_route.py`).
lake_option 3's persistence/level-pool hybrid and RFC-forecast reservoirs
(`woof_wrf_hydro.reservoir_da`).  Not yet: gw2d in the coupled driver;
Route_Link reaches in the coupled driver.  `woof_wrf_hydro.namelist.read("hydro.namelist")` maps
WRF-Hydro's namelist to these pieces and refuses unported options by name
(docs/coupling/WRF-HYDRO-NEMO.md).
