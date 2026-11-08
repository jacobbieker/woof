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

Ported: subsurface routing (`rt_option = 1`).  Not yet: the LSM/routing grid
aggregation and disaggregation, overland flow, channel routing, baseflow
(docs/coupling/WRF-HYDRO-NEMO.md).
