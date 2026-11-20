"""GR-V3: one real case of clear-sky ABI radiances with VarBC in woof's 3D-Var (plan section 13.8).

The case is supplied by the caller and nothing of it is in the repository.
``WOOF_GRV3_DIR`` names a directory holding:

- ``be.dat``: a cv5 background-error file whose levels match the first guess
  (WRFDA's tutorial ``be.dat`` on the tutorial grid);
- ``fg``: a WRF first guess (woof's own GFS initialisation);
- ``ob.ascii``: OBSPROC ``obs_gts`` from ``woof obs wis2 fetch --obs-gts``;
- ``scene``: one cached imager scene, the entry directory
  ``<cache>/goes<N>/<YYYYmmddTHHMM>/<key>`` that ``woof obs geo fetch --cache``
  writes (GOES ABI from ``bkr/geo/virtualized`` only).

CRTM comes from ``WOOF_CRTM_LIB``/``WOOF_CRTM_COEFFS``.  The test runs two
analyses on the reference tier, conventional only and conventional plus
radiances (VarBC cold start), and asserts:

- the water-vapour channels' O-A is below their O-B;
- the radiosonde temperature, humidity and wind O-A is no worse than in the
  conventional-only analysis (within 1 %).

The numbers are printed as one ``GRV3`` JSON line.  docs/dev/WRFDA-3DVAR-PLAN.md
section 13.8 records a run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

NEED = ("be.dat", "fg", "ob.ascii", "scene")
WV = (8, 9, 10)
SONDE = ("sound:u", "sound:v", "sound:t", "sound:q")


def _case() -> Path:
    d = os.environ.get("WOOF_GRV3_DIR")
    if not d or not all((Path(d) / n).exists() for n in NEED):
        pytest.skip(f"WOOF_GRV3_DIR with {', '.join(NEED)} is not set")
    if not os.environ.get("WOOF_CRTM_LIB") or not os.environ.get("WOOF_CRTM_COEFFS"):
        pytest.skip("WOOF_CRTM_LIB and WOOF_CRTM_COEFFS are not set")
    return Path(d)


def test_abi_radiances_fit_without_hurting_the_sondes():
    d = _case()
    from woof.da.var import radiance as rad
    from woof.da.var import varbc as vbc
    from woof.da.var.analysis import analyse
    from woof.obs.imager_scene import read_scene_netcdf

    common = dict(tier="reference", ntmax=int(os.environ.get("WOOF_GRV3_NTMAX", "50")), eps=0.01)
    base = analyse(d / "be.dat", d / "ob.ascii", d / "fg", report_fit=True, **common)
    records = rad.from_imager_scene(read_scene_netcdf(d / "scene"))
    vb = vbc.VarBC({r.sensor: vbc.template(r.sensor, list(r.channels)) for r in records})
    res = analyse(d / "be.dat", d / "ob.ascii", d / "fg", radiances=records, varbc=vb, **common)
    sensors = sorted(vb.instruments)
    out = {"base": base.fit, "radiance": res.fit, "report": res.radiance,
           "iterations": [[r.iterations for r in base.cg], [r.iterations for r in res.cg]],
           "varbc": {s: {c.channel: [round(float(p), 4) for p in c.param] for c in vb.instruments[s].channels
                         if c.nobs} for s in sensors}}
    print("GRV3 " + json.dumps(out, default=str))
    keys = [k for k in res.fit if k.split(":")[0] in sensors and int(k.split(":")[1]) in WV]
    assert keys, "no water-vapour channel was assimilated"
    for k in keys:
        f = res.fit[k]
        assert f["n"] > 0 and f["oma_rms"] < f["omb_rms"], (k, f)
    for k in SONDE:
        if k in base.fit:
            assert res.fit[k]["n"] == base.fit[k]["n"]
            assert res.fit[k]["oma_rms"] <= 1.01 * base.fit[k]["oma_rms"], (k, res.fit[k], base.fit[k])
