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
