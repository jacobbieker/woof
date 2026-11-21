"""spacewx_ionosphere_opt on hex: the climatological ionosphere's rows, one value per cell (woof.hex.spacewx_history).

* The generic column seam (tests/test_hex_rrtmgk_gpu.py's cells on a polar cap, 3 km to 20 km cell areas):
  ``--physics spacewx_ionosphere_opt=1`` reaches the seam's own PhysicsDriver, whose rows at each frame's valid
  time are word for word the WRF route's Ionosphere on the same cell centres.
* The MPAS v8.4.2 suite route: the door's spacewx keys build the same rows on the mesh's cell centres.

The CCIR maps are tests/test_spacewx_ionosphere.py's synthetic ones and the field its dipole, so no fetch runs.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from test_hex_rrtmgk_gpu import NCOL, START, _build, _cap, _physics  # noqa: E402
from test_spacewx_ionosphere import _dipole, _loader  # noqa: E402
from woof.hex import spacewx_history as sh  # noqa: E402
from woof.spacewx import history as hist  # noqa: E402
from woof.spacewx import igrf  # noqa: E402
from woof.spacewx import ionosphere as ion  # noqa: E402

pytestmark = pytest.mark.gpu
NAMES = tuple(hist.HISTORY_FIELDS)


@pytest.fixture
def synthetic(monkeypatch):
    read_maps = ion.read_maps
    monkeypatch.setattr(ion, "read_maps", lambda month, family="ccir", **kw: read_maps(month, family,
                                                                                       loader=_loader()))
    monkeypatch.setattr(igrf, "load", lambda path=None: _dipole())


def _want(lat, lon, when, f107=150.0):
    out = ion.Ionosphere(np.asarray(lat, np.float64).reshape(1, -1), np.asarray(lon, np.float64).reshape(1, -1),
                         when=START, igrf_model=_dipole(), map_loader=_loader()).evaluate(when, f107)
    return {name: cp.asnumpy(out[key]).reshape(-1).astype(np.float32)
            for name, (key, _d, _u) in hist.HISTORY_FIELDS.items()}


def test_the_generic_seam_writes_the_wrf_routes_rows_per_cell(synthetic):
    seam = _build(dict(_physics(), spacewx_ionosphere_opt=1, spacewx_f107_source="150"))
    holders = sh.seam_holders(seam)
    assert len(holders) == 1 and holders[0].start_time == START
    lat, lon = _cap(NCOL)
    frames = {}
    for hours in (0, 3):
        when = START + timedelta(hours=hours)
        got = sh.cell_fields(holders, when.strftime("%Y-%m-%d_%H:%M:%S"), NCOL)
        assert set(got) == set(NAMES)
        want = _want(lat, lon, when)
        for name, (values, dims, units) in got.items():
            assert dims == ("nCells",) and values.dtype == np.float32 and values.shape == (NCOL,)
            assert units == hist.HISTORY_FIELDS[name][2]
            np.testing.assert_array_equal(values, want[name], name)
        frames[hours] = got
    assert float(frames[0]["IONO_TEC"][0].min()) > 0.0
    assert not np.array_equal(frames[0]["IONO_TEC"][0], frames[3]["IONO_TEC"][0])
    # A regional mesh's padded physics column is not a history cell.
    short = sh.cell_fields(holders, START, NCOL - 1)
    np.testing.assert_array_equal(short["IONO_FOF2"][0], frames[0]["IONO_FOF2"][0][:-1])
    assert sh.seam_holders(_build(_physics())) == ()


def test_the_suite_route_builds_the_same_rows_from_the_door_keys(synthetic):
    settings = sh.suite_settings({"spacewx_ionosphere_opt": 1, "spacewx_f107_source": "150",
                                  "config_microp_scheme": "mp_wsm6"})
    assert settings == {"spacewx_ionosphere_opt": 1, "spacewx_f107_source": "150",
                        "spacewx_f107_mode": "81day", "spacewx_ionosphere_maps": "ccir"}
    lat, lon = _cap(NCOL)
    holders = sh.suite_holders(settings, latitude_deg=lat, longitude_deg=lon, start_time=START,
                               run_seconds=6 * 3600.0)
    when = START + timedelta(hours=5)
    got = sh.cell_fields(holders, when, NCOL)
    want = _want(lat, lon, when)
    for name in NAMES:
        np.testing.assert_array_equal(got[name][0], want[name], name)
    assert sh.suite_holders(sh.suite_settings({}), latitude_deg=lat, longitude_deg=lon, start_time=START,
                            run_seconds=0.0) == ()


@pytest.mark.parametrize("given, match", [
    ({"spacewx_ionosphere_opt": 2}, "must be 0"),
    ({"spacewx_f107_source": "150"}, "would do nothing"),
    ({"spacewx_ionosphere_opt": 1, "spacewx_f107_mode": "weekly"}, "81day"),
    ({"spacewx_ionosphere_opt": 1, "spacewx_ionosphere_maps": "iri"}, "ccir"),
])
def test_the_suite_keys_are_judged_as_the_wrf_route_judges_them(given, match):
    with pytest.raises(ValueError, match=match):
        sh.suite_settings(given)
