"""The hex history writer's spacewx rows (woof.hex.spacewx_history.cell_fields -> write_snapshot_netcdf):
``(Time, nCells)`` float32 under the wrfout name, with the WRF route's units and long name."""
from __future__ import annotations

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from woof.hex.drivers import run_cuda_v841_full_physics_x4 as x4  # noqa: E402

N = 5


def _static():
    return {"indexToCellID": np.arange(1, N + 1, dtype=np.int32), "latCell": np.linspace(0.1, 0.5, N),
            "lonCell": np.linspace(-0.2, 0.2, N), "ter": np.zeros(N),
            "indexToEdgeID": np.arange(1, 4, dtype=np.int32), "latEdge": np.zeros(3), "lonEdge": np.zeros(3)}


@pytest.fixture
def mesh(monkeypatch):
    for name, value in (("N_CELLS", N), ("N_LEVELS", 2), ("N_EDGES", 3), ("N_INTERFACES", 3),
                        ("N_SOIL_LEVELS", 0)):
        monkeypatch.setattr(x4, name, value)


def test_the_rows_are_written_per_cell_with_their_units(tmp_path, mesh):
    tec = np.linspace(5.0, 45.0, N, dtype=np.float32)
    zone = np.array([0, 1, 2, 3, 2], dtype=np.float32)
    snapshot = {"arrays": {}, "valid_time": "2026-03-20_06:00:00",
                "spacewx_fields": {"IONO_TEC": (tec, ("nCells",), "TECU"),
                                   "AURORA_ZONE": (zone, ("nCells",), "-")}}
    path = tmp_path / "frame.nc"
    x4.write_snapshot_netcdf(path, snapshot, _static())
    with netCDF4.Dataset(path) as data:
        for name, want, units in (("IONO_TEC", tec, "TECU"), ("AURORA_ZONE", zone, "-")):
            var = data.variables[name]
            assert var.dimensions == ("Time", "nCells") and var.dtype == np.float32
            assert var.units == units and var.long_name
            np.testing.assert_array_equal(var[0, :], want)
        assert "CLIMATOLOGICAL IONOSPHERE" in data.variables["IONO_TEC"].long_name


def test_a_row_that_is_not_one_value_per_cell_is_refused(tmp_path, mesh):
    snapshot = {"arrays": {}, "valid_time": "2026-03-20_06:00:00",
                "spacewx_fields": {"IONO_TEC": (np.zeros(N + 1, np.float32), ("nCells",), "TECU")}}
    with pytest.raises(ValueError, match="one value per history cell"):
        x4.write_snapshot_netcdf(tmp_path / "frame.nc", snapshot, _static())


def _backend(ionosphere=None):
    from types import SimpleNamespace
    return SimpleNamespace(_seam=SimpleNamespace(_driver=SimpleNamespace(spacewx_ionosphere=ionosphere,
                                                                         spacewx_aurora=None)))


def test_launch_modes_without_the_rows_refuse_them():
    from woof.hex import spacewx_history as sh
    sh.refuse_unwired(_backend(), "a same-host rank run")            # off: nothing to refuse
    with pytest.raises(NotImplementedError, match="same-host rank run.*does not carry"):
        sh.refuse_unwired(_backend(object()), "a same-host rank run")


def test_the_frames_maps_load_before_the_time_loop():
    from types import SimpleNamespace
    from woof.hex import spacewx_history as sh
    from woof.spacewx.ionosphere import month_weights
    loaded = []
    holder = SimpleNamespace(ionosphere=SimpleNamespace(maps=loaded.append))
    times = ["2026-03-30_00:00:00", "2026-04-02_12:00:00", "2026-04-20_00:00:00"]
    sh.preload([holder, SimpleNamespace()], times)
    want = sorted({m for t in times for m, _w in month_weights(sh._when(t))})
    assert loaded == want and len(want) >= 2


@pytest.mark.parametrize("value", [5, 5.0, "5", " '5' "])
def test_a_runconfig_string_keeps_its_text_whatever_the_spelling(value):
    from woof.hex.engine_registry import setting_value
    assert setting_value("spacewx_kp_source", value) == ("5.0" if value == 5.0 and not isinstance(value, int)
                                                        else "5")
    assert setting_value("spacewx_aurora_opt", "2") == 2
