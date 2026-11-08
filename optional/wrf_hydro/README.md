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

Ported: subsurface and overland routing (`rt_option = 1`); the LSM <-> routing
grid disaggregation and aggregation; reach channel routing (Muskingum and
Muskingum-Cunge, channel_option 1 and 2); the diffusive wave on nodes
(channel_option 3) without lakes; the groundwater bucket (GWBASESWCRT 1 and
2).  Not yet: reservoirs, gw2d.  `woof_wrf_hydro.namelist.read("hydro.namelist")` maps
WRF-Hydro's namelist to these pieces and refuses unported options by name
(docs/coupling/WRF-HYDRO-NEMO.md).
