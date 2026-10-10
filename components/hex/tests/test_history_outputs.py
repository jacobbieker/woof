"""Hex history outputs: radiation fields, zgrid, xtime and variable selection.

THE BREAKAGE THIS PREVENTS: the energy sampler (woof.energy.sample_hex)
reads SWDOWN/SWDDNI/SWDDIF/COSZEN, places profiles with zgrid and dates
frames with xtime.  The CUDA history carried swdown only, no zgrid and no
xtime, so a hex energy forecast could not answer solar questions and could
not be sampled without the culled init beside it.  These tests go red when
any of those leaves the forecast lane's frame, when a selection silently
drops an explicitly requested name, or when the proof lane's frame (written
without the new keywords) starts carrying them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from _layout import PACKAGE_DIR

netCDF4 = pytest.importorskip("netCDF4")

from woof.hex import history_selection as selection  # noqa: E402
from woof.hex.history_selection import HistorySelectionRefusal as ConfigurationRefusal  # noqa: E402

RUNNER_PATH = PACKAGE_DIR / "drivers" / "run_cuda_v841_full_physics_x4.py"


def _load_runner():
    name = "_test_history_outputs_runner"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    module = _load_runner()
    # A small mesh: the writer sizes every dimension from these constants,
    # exactly as a forecast's bind rebinds them.
    module.N_CELLS, module.N_EDGES = 12, 30
    return module


def _frame(runner):
    n, e, k = runner.N_CELLS, runner.N_EDGES, runner.N_LEVELS
    rng = np.random.default_rng(16)
    arrays = {
        "u_zonal": rng.normal(size=(k, n)).astype(np.float32),
        "theta": (290.0 + rng.normal(size=(k, n))).astype(np.float32),
        "w": rng.normal(size=(runner.N_INTERFACES, n)).astype(np.float32),
        "normal_u": rng.normal(size=(k, e)).astype(np.float32),
        "t2": rng.uniform(270, 300, size=n).astype(np.float32),
        "swdown": rng.uniform(0, 900, size=n).astype(np.float32),
        "swddni": rng.uniform(0, 900, size=n).astype(np.float32),
        "swddif": rng.uniform(0, 300, size=n).astype(np.float32),
        "coszr": rng.uniform(0, 1, size=n).astype(np.float32),
    }
    static = {
        "indexToCellID": np.arange(1, n + 1, dtype=np.int32),
        "latCell": np.linspace(0.9, 0.91, n).astype(np.float32),
        "lonCell": np.linspace(-0.07, -0.06, n).astype(np.float32),
        "ter": np.full(n, 120.0, dtype=np.float32),
        "indexToEdgeID": np.arange(1, e + 1, dtype=np.int32),
        "latEdge": np.zeros(e, dtype=np.float32),
        "lonEdge": np.zeros(e, dtype=np.float32),
    }
    zgrid = (120.0 + np.cumsum(np.full((n, runner.N_INTERFACES), 20.0), axis=1)
             - 20.0).astype(np.float32)
    return {"arrays": arrays, "receipt": {}}, static, zgrid


def test_the_forecast_frame_carries_radiation_zgrid_and_xtime(runner, tmp_path):
    snapshot, static, zgrid = _frame(runner)
    path = tmp_path / "cuda-history.2026-10-10_06.00.00.nc"
    runner.write_snapshot_netcdf(
        path, snapshot, static, xtime="2026-10-10_06:00:00", zgrid=zgrid
    )
    with netCDF4.Dataset(path) as ds:
        for name, units in (("swdown", "W m^{-2}"), ("swddni", "W m^{-2}"),
                            ("swddif", "W m^{-2}"), ("coszr", "1")):
            variable = ds.variables[name]
            assert variable.dimensions == ("Time", "nCells")
            assert variable.getncattr("units") == units
            np.testing.assert_array_equal(variable[0], snapshot["arrays"][name])
        z = ds.variables["zgrid"]
        assert z.dimensions == ("nCells", "nVertLevelsP1")
        assert z.getncattr("units") == "m"
        np.testing.assert_array_equal(z[:], zgrid)
        assert ds.getncattr("zgrid_sha256") == runner.array_sha256(zgrid)
        xtime = ds.variables["xtime"]
        assert xtime.dimensions == ("Time", "StrLen")
        text = netCDF4.chartostring(np.asarray(xtime[:]))[0]
        assert str(text).strip() == "2026-10-10_06:00:00"


def test_the_proof_frame_is_unchanged_without_the_keywords(runner, tmp_path):
    snapshot, static, _ = _frame(runner)
    path = tmp_path / "proof.nc"
    runner.write_snapshot_netcdf(path, snapshot, static)
    with netCDF4.Dataset(path) as ds:
        assert "xtime" not in ds.variables
        assert "zgrid" not in ds.variables
        assert "StrLen" not in ds.dimensions
        assert "zgrid_sha256" not in ds.ncattrs()
        assert set(snapshot["arrays"]) <= set(ds.variables)


def test_the_writer_publishes_only_the_selected_arrays(runner, tmp_path):
    snapshot, static, zgrid = _frame(runner)
    path = tmp_path / "selected.nc"
    runner.write_snapshot_netcdf(
        path, snapshot, static, variables=("t2", "swddni"),
        xtime="2026-10-10_06:00:00", zgrid=None,
    )
    with netCDF4.Dataset(path) as ds:
        names = set(ds.variables)
    assert {"t2", "swddni", "xtime", "latCell", "lonCell"} <= names
    assert not names & {"u_zonal", "theta", "w", "normal_u", "swdown", "zgrid"}


def test_a_wrong_zgrid_or_time_is_refused(runner, tmp_path):
    snapshot, static, zgrid = _frame(runner)
    with pytest.raises(ValueError, match="zgrid has shape"):
        runner.write_snapshot_netcdf(tmp_path / "a.nc", snapshot, static,
                                     zgrid=zgrid[:, :-1])
    with pytest.raises(ValueError):
        runner.write_snapshot_netcdf(tmp_path / "b.nc", snapshot, static,
                                     xtime="2026-10-10 06:00")
    assert not (tmp_path / "a.nc").exists()
    assert not (tmp_path / "b.nc").exists()


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def test_the_default_is_the_full_set():
    chosen = selection.resolve_history_selection(None, None)
    assert chosen["variables"] is None and chosen["source"] == "default"
    plan = selection.plan_frame(chosen, {"t2", "u_zonal"}, zgrid_available=True)
    assert plan == {"arrays": None, "zgrid": True, "missing": ()}
    assert selection.selection_argv(chosen) == []
    full = selection.resolve_history_selection(None, "full")
    assert full["variables"] is None
    assert selection.selection_argv(full) == []


def test_the_energy_preset_mirrors_the_wrf_side():
    chosen = selection.resolve_history_selection(None, "energy")
    assert chosen["variables"] == (
        "u_zonal", "v_meridional", "w", "theta", "pressure", "rho",
        "qv", "qc", "qr", "qi", "qs", "qg",
        "t2", "q2", "u10", "v10", "surface_pressure",
        "swdown", "swddni", "swddif", "coszr",
        "rainnc", "rainc", "zgrid", "xtime",
    )
    assert selection.selection_argv(chosen) == ["--history-preset", "energy"]


def test_a_preset_member_the_run_lacks_is_recorded_not_refused():
    chosen = selection.resolve_history_selection(None, "energy")
    carried = set(chosen["variables"]) - {"swddni", "swddif", "zgrid", "xtime"}
    carried |= {"refl10cm", "normal_u"}
    plan = selection.plan_frame(chosen, carried, zgrid_available=True)
    assert plan["missing"] == ("swddni", "swddif")
    assert plan["zgrid"] is True
    assert "refl10cm" not in plan["arrays"]
    assert "swddni" not in plan["arrays"]


def test_an_explicit_name_the_run_lacks_refuses():
    chosen = selection.resolve_history_selection("t2,swddni", None)
    assert chosen["strict"] is True
    assert selection.selection_argv(chosen) == ["--history-vars", "t2,swddni"]
    with pytest.raises(ConfigurationRefusal, match=r"\['swddni'\]"):
        selection.plan_frame(chosen, {"t2"}, zgrid_available=False)
    zg = selection.resolve_history_selection("t2,zgrid,xtime", None)
    with pytest.raises(ConfigurationRefusal, match="zgrid"):
        selection.plan_frame(zg, {"t2"}, zgrid_available=False)
    plan = selection.plan_frame(zg, {"t2"}, zgrid_available=True)
    assert plan == {"arrays": ("t2",), "zgrid": True, "missing": ()}


@pytest.mark.parametrize(
    "text, match",
    [("t2,,u10", "empty name"), ("t2,2bad", "not variable names"),
     ("t2,t2", "repeats")],
)
def test_a_malformed_list_refuses(text, match):
    with pytest.raises(ConfigurationRefusal, match=match):
        selection.resolve_history_selection(text, None)


def test_a_list_and_a_preset_together_refuse():
    with pytest.raises(ConfigurationRefusal, match="exclusive"):
        selection.resolve_history_selection("t2", "energy")
    with pytest.raises(ConfigurationRefusal, match="not a history preset"):
        selection.resolve_history_selection(None, "wind")


def test_the_driver_parses_the_selection_and_refuses_both(tmp_path):
    from woof.hex.drivers import run_cuda_v841_forecast as forecast

    base = ["--init", str(tmp_path / "i.nc"), "--init-source", "test",
            "--hours", "1", "--history-every-minutes", "30",
            "--preflight-only"]
    args = forecast.parse_args(base + ["--history-preset", "energy"])
    assert args.history_selection["preset"] == "energy"
    args = forecast.parse_args(base)
    assert args.history_selection["variables"] is None
    with pytest.raises(SystemExit):
        forecast.parse_args(base + ["--history-preset", "energy",
                                    "--history-vars", "t2"])


def test_the_adapter_reads_the_radiation_buffers_the_seam_exports():
    """The export keys are the restart manifest's names for the buffers.

    ``woof.io.restart._driver_manifest`` writes each checkpoint-only driver
    attribute as ``diag/<attribute>``; a rename on either side would make
    the optional fields silently absent from every frame.
    """

    from woof.hex import cuda_arwen_physics_v841 as adapter
    from woof.io.restart import DRIVER_CHECKPOINT_ONLY_ATTRS

    for name in ("swddni", "swddif", "coszr"):
        assert name in adapter._OPTIONAL_ARWEN_EXPORT_FIELDS
        key = adapter._ARWEN_EXPORT_KEYS[name]
        prefix, attribute = key.split("/")
        assert prefix == "diag"
        assert attribute in DRIVER_CHECKPOINT_ONLY_ATTRS
        assert name not in adapter._REQUIRED_ARWEN_EXPORT_FIELDS


def test_the_energy_sampler_reads_a_written_frame_without_the_init(
    runner, tmp_path
):
    """Writer to sampler: radiation keys, history zgrid and xtime.

    No ``mesh_path``: coordinates and zgrid come from the frame itself, and
    the file name carries no label, so the valid time can only be xtime.
    """

    from woof.energy.sample_hex import sample_mpas

    snapshot, static, zgrid = _frame(runner)
    path = tmp_path / "frame.nc"
    runner.write_snapshot_netcdf(
        path, snapshot, static, xtime="2026-10-10_12:00:00", zgrid=zgrid
    )
    cell = 5
    lat = np.degrees(float(static["latCell"][cell]))
    lon = np.degrees(float(static["lonCell"][cell]))
    result = sample_mpas([path], [lat], [lon], [30.0],
                         profile_vars=("THETA",),
                         surface_vars=("SWDOWN", "SWDDNI", "SWDDIF", "COSZEN"))
    assert result.times.tolist() == [np.datetime64("2026-10-10T12:00:00")]
    assert result.inside.tolist() == [True]
    for key, name in (("SWDOWN", "swdown"), ("SWDDNI", "swddni"),
                      ("SWDDIF", "swddif"), ("COSZEN", "coszr")):
        assert result.surface[key][0, 0] == pytest.approx(
            float(snapshot["arrays"][name][cell]), rel=1e-6)
    # Mass levels sit at 10, 30, 50 m above the 120 m terrain: 30 m is
    # level 1 exactly.
    assert result.profile["THETA"][0, 0, 0] == pytest.approx(
        float(snapshot["arrays"]["theta"][1, cell]), rel=1e-6)
    assert result.terrain_m[0] == pytest.approx(120.0)
    assert any("zgrid read from the history" in n for n in result.notes)


def test_a_name_no_frame_can_carry_refuses_before_the_run():
    chosen = selection.resolve_history_selection("t2m,u_zonal", None)
    possible = (*selection.FRAME_NAMES, "qv", "qc", "rainnc")
    with pytest.raises(ConfigurationRefusal, match=r"\['t2m'\]"):
        selection.precheck_names(chosen, possible)
    selection.precheck_names(
        selection.resolve_history_selection("t2,qv,zgrid,xtime", None), possible
    )
    # Presets are never prechecked: their absent members are recorded.
    selection.precheck_names(
        selection.resolve_history_selection(None, "energy"), ("t2",)
    )
    assert selection.wants_zgrid(selection.resolve_history_selection(None, None))
    assert not selection.wants_zgrid(
        selection.resolve_history_selection("t2", None))


def test_a_selection_refusal_is_not_a_model_refusal():
    """The forecast loop reads RuntimeError from a capture as the port
    refusing the model state; a selection mistake must not look like one."""

    assert issubclass(selection.HistorySelectionRefusal, ValueError)
    assert not issubclass(selection.HistorySelectionRefusal, RuntimeError)


def test_frame_names_cover_every_adapter_export():
    from woof.hex import cuda_arwen_physics_v841 as adapter

    for name in (*adapter._REQUIRED_ARWEN_EXPORT_FIELDS,
                 *adapter._OPTIONAL_ARWEN_EXPORT_FIELDS,
                 *adapter._GWDO_DIAGNOSTIC_FIELDS):
        assert name in selection.FRAME_NAMES, name


def test_a_selected_array_without_a_layout_is_refused(runner, tmp_path):
    snapshot, static, _ = _frame(runner)
    snapshot["arrays"]["odd"] = np.zeros((3, 3), np.float32)
    with pytest.raises(ValueError, match="odd"):
        runner.write_snapshot_netcdf(tmp_path / "odd.nc", snapshot, static,
                                     variables=("t2", "odd"))
    assert not (tmp_path / "odd.nc").exists()
    # The proof lane (no selection) still skips it as it always did.
    runner.write_snapshot_netcdf(tmp_path / "all.nc", snapshot, static)
    with netCDF4.Dataset(tmp_path / "all.nc") as ds:
        assert "odd" not in ds.variables
