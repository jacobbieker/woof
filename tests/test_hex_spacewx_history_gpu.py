"""spacewx_ionosphere_opt and spacewx_aurora_opt on hex's generic column seam: one value per cell
(woof.hex.spacewx_history).

On tests/test_hex_rrtmgk_gpu.py's cells (a spherical cap, cell areas 3 km to 20 km across: a variable-resolution
set), ``--physics spacewx_ionosphere_opt=1`` / ``spacewx_aurora_opt=1`` reach the seam's own PhysicsDriver,
and the rows hex writes at each frame's valid time are word for word the WRF route's on the same cell centres
(the radiation geometry's float32 latitudes and longitudes, as on a WRF domain).

The CCIR maps are tests/test_spacewx_ionosphere.py's synthetic ones and the field its dipole, so no fetch runs;
the aurora is driven by a fixed Kp.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from test_hex_rrtmgk_gpu import NCOL, START, _build, _physics  # noqa: E402
from test_spacewx_ionosphere import _dipole, _loader  # noqa: E402
from woof.hex import spacewx_history as sh  # noqa: E402
from woof.spacewx import aurora_history as ah  # noqa: E402
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


def _geometry(seam):
    radiation = seam._driver.radiation_callable
    return (cp.asnumpy(radiation.latitude_deg).astype(np.float64).reshape(-1),
            cp.asnumpy(radiation.longitude_deg).astype(np.float64).reshape(-1))


def test_the_generic_seam_writes_the_wrf_routes_rows_per_cell(synthetic):
    seam = _build(dict(_physics(), spacewx_ionosphere_opt=1, spacewx_f107_source="150"))
    holders = sh.seam_holders(seam)
    assert len(holders) == 1 and holders[0].start_time == START
    lat, lon = _geometry(seam)
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


def test_the_aurora_rows_are_the_wrf_routes_per_cell(synthetic):
    from types import SimpleNamespace
    physics = dict(_physics(), spacewx_aurora_opt=1, spacewx_kp_source="5")
    seam = _build(physics)
    holders = sh.seam_holders(seam)
    assert [type(h).__name__ for h in holders] == ["AuroraHistory"]
    lat, lon = _geometry(seam)
    # The WRF route's own rows, from the same configuration on a (1, ncells) radiation geometry.
    geometry = SimpleNamespace(start_time=START, latitude_deg=lat.reshape(1, -1), longitude_deg=lon.reshape(1, -1))
    route = ah.AuroraHistory.for_run(seam._cfg, geometry, igrf_model=_dipole())
    when = START + timedelta(hours=2)
    got = sh.cell_fields(holders, when, NCOL)
    want = route.history_fields((when - START).total_seconds())
    assert set(got) == set(ah.HISTORY_FIELDS) == set(want)
    for name, (values, dims, units) in got.items():
        assert dims == ("nCells",) and units == ah.HISTORY_FIELDS[name][2]
        np.testing.assert_array_equal(values, cp.asnumpy(want[name]).reshape(-1), name)
    # The cap sits around 56N: Kp 5 puts its cells in more than one zone.
    zone = got["AURORA_ZONE"][0]
    assert zone.min() >= 0 and zone.max() <= 3
    assert np.all(np.abs(got["MAG_LAT"][0]) <= 90.0)


def test_both_options_together_write_both_row_sets(synthetic):
    seam = _build(dict(_physics(), spacewx_ionosphere_opt=1, spacewx_f107_source="150",
                       spacewx_aurora_opt=1, spacewx_kp_source="5"))
    got = sh.cell_fields(sh.seam_holders(seam), START, NCOL)
    assert set(got) == set(NAMES) | set(ah.HISTORY_FIELDS)


def test_the_generic_seam_admits_the_keys():
    from woof.hex.physics_suites import resolve_physics
    physics = resolve_physics("arwen_mpas_column", {"spacewx_ionosphere_opt": 1, "spacewx_f107_source": "150",
                                                    "spacewx_aurora_opt": 1, "spacewx_kp_source": "5"},
                              mesh="x1.40962")
    assert physics.engine_physics["spacewx_ionosphere_opt"] == 1
    assert physics.engine_physics["spacewx_aurora_opt"] == 1
