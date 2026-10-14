"""Hub-height wind fields and per-history-interval statistics.

CPU tier: the NumPy reference (vertical interpolation in z-AGL,
destaggering, rotation), the interval window (time weighting, reset,
cold frames), site sampling and its sidecar, the ``[output]`` parsing and
refusals, the writer's ``hub_height`` axis, the registry and its hooks.

The GPU tier (the inline kernel against this reference, and real dycore
steps) is tests/test_hub_wind_diag_gpu.py: a module that imports cupy is
marked ``gpu`` whole by tests/conftest.py.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.core import interval_diagnostics
from woof.core import hub_wind_diag as hw
from woof.io.history_selection import (
    DEFAULT_HUB_HEIGHTS_M, HISTORY_PRESETS, HISTORY_VOCABULARY,
    HistorySelection, HubWindConfigError, HubWindRequest)
from woof.io.wrf_output_schema import (HUB_HEIGHT_DIMENSIONED_FIELDS,
                                        HUB_WIND_OUTPUT_FIELDS)

REPO = Path(__file__).resolve().parents[1]
G = np.float32(9.81)

#: The inline kernel's bytes.  Changing them is a decision, not an edit:
#: re-run the GPU tier below and re-pin (the kernels/*.cu freeze discipline,
#: tests/test_kernel_source_freeze_per_module.py, applied to an inline unit).
KERNEL_SOURCE_SHA256 = (
    "19faada0c001a5e05e7f1b7c65fc00571e71ecd65d2e95096eb39aca05d1d045")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _column_state(nz=12, ny=4, nx=5, *, dz=40.0, hgt=100.0, u_of_z=None,
                  v_of_z=None, phb_1d=True, sina=0.0, cosa=1.0):
    """A host state whose mass levels sit at known heights above ground.

    w levels at ``hgt + k*dz`` (so mass levels at ``(k+0.5)*dz`` AGL);
    PH carries the terrain, PHB the column (flat 1-D by default).
    """
    zw = (np.arange(nz + 1, dtype=np.float64) * dz)
    phb = (zw * float(G)).astype(np.float32)
    ph = np.full((nz + 1, ny, nx), hgt * float(G), dtype=np.float32)
    if not phb_1d:
        phb = np.ascontiguousarray(np.broadcast_to(phb[:, None, None],
                                                   (nz + 1, ny, nx)))
    zm = (np.arange(nz) + 0.5) * dz
    u_of_z = u_of_z or (lambda z: 0.0 * z)
    v_of_z = v_of_z or (lambda z: 0.0 * z)
    u = np.broadcast_to(u_of_z(zm)[:, None, None], (nz, ny, nx + 1))
    v = np.broadcast_to(v_of_z(zm)[:, None, None], (nz, ny + 1, nx))
    return SimpleNamespace(
        u=np.ascontiguousarray(u, dtype=np.float32),
        v=np.ascontiguousarray(v, dtype=np.float32),
        php=ph, phb=phb, ht=np.full((ny, nx), hgt, dtype=np.float32),
        sina=np.full((ny, nx), sina, dtype=np.float32),
        cosa=np.full((ny, nx), cosa, dtype=np.float32))


def _diag(state, heights=(40.0, 80.0, 100.0), **kw):
    ny, nx = state.ht.shape
    request = HubWindRequest(heights_m=tuple(heights), **kw)
    return hw.HubWindDiagnostic(request, ny=ny, nx=nx)


# ---------------------------------------------------------------------------
# the reference arithmetic
# ---------------------------------------------------------------------------

def test_mass_heights_are_agl_midpoints_of_the_w_levels():
    state = _column_state(nz=6, dz=50.0, hgt=321.0)
    z = hw.mass_heights_agl_np(state.php, state.phb, state.ht)
    assert z.shape == (6, 4, 5) and z.dtype == np.float32
    np.testing.assert_allclose(z[:, 0, 0], (np.arange(6) + 0.5) * 50.0,
                               atol=2e-3)
    # 1-D and 3-D PHB give identical bytes.
    z3 = hw.mass_heights_agl_np(
        state.php, np.broadcast_to(state.phb[:, None, None], state.php.shape),
        state.ht)
    np.testing.assert_array_equal(z, z3)


def test_destaggering_averages_the_two_faces():
    u = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    v = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3) * 2
    um, vm = hw.destagger_np(u, v)
    assert um.shape == vm.shape == (2, 3, 3)
    np.testing.assert_array_equal(um, 0.5 * (u[:, :, :-1] + u[:, :, 1:]))
    np.testing.assert_array_equal(vm, 0.5 * (v[:, :-1, :] + v[:, 1:, :]))
    # A non-uniform face pair: the mass value is their mean, not either face.
    u2 = np.zeros((1, 1, 2), np.float32)
    u2[0, 0, 1] = 10.0
    assert hw.destagger_np(u2, np.zeros((1, 2, 1), np.float32))[0][0, 0, 0] == 5.0


def test_interpolation_is_linear_in_z_agl_and_holds_outside_the_column():
    # u = 2 + 0.05 z: linear interpolation must recover it exactly-ish.
    state = _column_state(nz=10, dz=40.0, u_of_z=lambda z: 2.0 + 0.05 * z)
    heights = (10.0, 20.0, 41.0, 100.0, 333.0, 380.0, 1000.0)
    speed, direction = hw.hub_wind_np(
        state.u, state.v, state.php, state.phb, state.ht,
        state.sina, state.cosa, heights)
    zm = (np.arange(10) + 0.5) * 40.0          # 20 .. 380 m
    want = []
    for h in heights:
        clipped = min(max(h, zm[0]), zm[-1])   # below/above: hold
        want.append(2.0 + 0.05 * clipped)
    np.testing.assert_allclose(speed[:, 0, 0], want, rtol=1e-5)
    # Exactly on a mass level the interpolant is that level's value.
    assert speed[1, 0, 0] == np.float32(2.0 + 0.05 * 20.0)
    # A pure +u (westerly) wind blows FROM 270.
    np.testing.assert_allclose(direction, 270.0, atol=1e-4)


def test_interpolation_brackets_by_the_columns_own_heights():
    # Two columns with different terrain-following thickness: the same
    # target height lands between different level pairs.
    nz, ny, nx = 8, 1, 2
    zw = np.stack([np.arange(nz + 1) * 30.0, np.arange(nz + 1) * 60.0], -1)
    ph = (zw[:, None, :] * G).astype(np.float32)
    phb = np.zeros(nz + 1, np.float32)
    ht = np.zeros((ny, nx), np.float32)
    zm = 0.5 * (zw[:-1] + zw[1:])                         # (nz, nx)
    um = (1.0 + zm / 10.0).astype(np.float32)             # u = 1 + z/10
    u = np.zeros((nz, ny, nx + 1), np.float32)
    # Faces chosen so each mass value destaggers to um.
    u[:, 0, 0] = um[:, 0]
    u[:, 0, 1] = um[:, 0]
    u[:, 0, 2] = 2 * um[:, 1] - um[:, 0]
    v = np.zeros((nz, ny + 1, nx), np.float32)
    speed, _ = hw.hub_wind_np(u, v, ph, phb, ht, np.zeros_like(ht),
                              np.ones_like(ht), (100.0,))
    np.testing.assert_allclose(speed[0, 0], [11.0, 11.0], rtol=1e-6)


@pytest.mark.parametrize("alpha_deg, want", [
    (0.0, 270.0),     # grid +u is earth eastward: from the west
    (90.0, 180.0),    # grid x points north: +u blows toward north, from south
    (-90.0, 0.0),     # grid x points south: from the north
])
def test_rotation_gives_earth_relative_from_direction(alpha_deg, want):
    alpha = np.radians(alpha_deg)
    state = _column_state(u_of_z=lambda z: 7.0 + 0.0 * z,
                          sina=np.sin(alpha), cosa=np.cos(alpha))
    speed, direction = hw.hub_wind_np(
        state.u, state.v, state.php, state.phb, state.ht,
        state.sina, state.cosa, (100.0,))
    np.testing.assert_allclose(speed, 7.0, rtol=1e-6)
    got = float(direction[0, 0, 0])
    assert min(abs(got - want), 360.0 - abs(got - want)) < 1e-3
    assert 0.0 <= got < 360.0


def test_speed_ignores_the_rotation_and_a_calm_reports_zero():
    uh = np.array([[3.0, 0.0]], np.float32)
    vh = np.array([[4.0, 0.0]], np.float32)
    for angle in (0.0, 0.7, 2.1):
        s, d = hw.speed_direction_np(uh, vh, np.full_like(uh, np.sin(angle)),
                                     np.full_like(uh, np.cos(angle)))
        assert s[0, 0] == np.float32(5.0)
        assert s[0, 1] == 0.0 and d[0, 1] == 0.0


# ---------------------------------------------------------------------------
# the interval window
# ---------------------------------------------------------------------------

def _set_speed(state, value):
    state.u[...] = np.float32(value)
    state.v[...] = 0.0


def test_window_is_time_weighted_and_restarts_after_each_frame(tmp_path):
    state = _column_state()
    diag = _diag(state)
    for speed, dt in ((5.0, 10.0), (15.0, 30.0), (10.0, 20.0)):
        _set_speed(state, speed)
        diag.on_step(state, dt)
    fields = diag.history_fields(state)
    assert tuple(fields) == HUB_WIND_OUTPUT_FIELDS
    assert float(fields["HUB_WINDOW_S"]) == 60.0
    np.testing.assert_array_equal(fields["WSPD_HUB_MAX"], 15.0)
    np.testing.assert_array_equal(fields["WSPD_HUB_MIN"], 5.0)
    np.testing.assert_allclose(fields["WSPD_HUB_MEAN"],
                               (5 * 10 + 15 * 30 + 10 * 20) / 60.0, rtol=1e-6)
    np.testing.assert_array_equal(fields["WSPD_HUB"], 10.0)
    np.testing.assert_array_equal(fields["HUB_HEIGHT"], [40.0, 80.0, 100.0])
    # A plain mean would be 10; time weighting must not be it.
    assert abs(float(fields["WSPD_HUB_MEAN"][0, 0, 0]) - 10.0) > 1.0

    diag.close_interval(state, valid_time=datetime(2025, 1, 1, 0, 15),
                        output_dir=tmp_path, grid_id=1)
    assert diag.window_s == 0.0 and diag.window_steps == 0
    # The next window forgets the 15 m/s maximum entirely.
    _set_speed(state, 3.0)
    diag.on_step(state, 10.0)
    _set_speed(state, 4.0)
    diag.on_step(state, 10.0)
    fields = diag.history_fields(state)
    np.testing.assert_array_equal(fields["WSPD_HUB_MAX"], 4.0)
    np.testing.assert_array_equal(fields["WSPD_HUB_MIN"], 3.0)
    np.testing.assert_allclose(fields["WSPD_HUB_MEAN"], 3.5, rtol=1e-6)
    assert float(fields["HUB_WINDOW_S"]) == 20.0
    # No sites, no sidecar.
    assert not list(tmp_path.iterdir())


def test_a_frame_with_no_step_reports_the_instant_as_its_window():
    state = _column_state()
    _set_speed(state, 6.0)
    fields = _diag(state).history_fields(state)
    assert float(fields["HUB_WINDOW_S"]) == 0.0
    for name in ("WSPD_HUB_MAX", "WSPD_HUB_MIN", "WSPD_HUB_MEAN"):
        np.testing.assert_array_equal(fields[name], fields["WSPD_HUB"])
    np.testing.assert_array_equal(fields["WSPD_HUB"], 6.0)


def test_the_window_reads_state_and_writes_nothing_of_it():
    state = _column_state(u_of_z=lambda z: 1.0 + 0.01 * z,
                          v_of_z=lambda z: -2.0 + 0.0 * z)
    before = {name: np.array(getattr(state, name), copy=True)
              for name in ("u", "v", "php", "phb", "ht", "sina", "cosa")}
    diag = _diag(state)
    diag.on_step(state, 5.0)
    diag.history_fields(state)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(state, name), value,
                                      err_msg=f"the window wrote {name}")


# ---------------------------------------------------------------------------
# registry and hooks
# ---------------------------------------------------------------------------

def test_registry_dispatches_and_refuses_a_second_window_of_one_name():
    state = _column_state()
    diag = _diag(state)
    interval_diagnostics.attach(state, diag)
    with pytest.raises(interval_diagnostics.IntervalDiagnosticError):
        interval_diagnostics.attach(state, _diag(state))
    _set_speed(state, 9.0)
    interval_diagnostics.step(state, SimpleNamespace(dt=12.0))
    assert diag.window_s == 12.0
    fields = interval_diagnostics.history_fields(state)
    assert set(fields) == set(HUB_WIND_OUTPUT_FIELDS)
    with pytest.raises(hw.HubWindUnsupportedRoad, match="streamed"):
        interval_diagnostics.refuse_unsupported_publisher(
            state, "the domain runs [tiles]-streamed")
    # A state with nothing attached: every hook is a no-op.
    bare = SimpleNamespace()
    interval_diagnostics.step(bare, SimpleNamespace(dt=1.0))
    assert interval_diagnostics.history_fields(bare) == {}
    interval_diagnostics.refuse_unsupported_publisher(bare, "anything")
    interval_diagnostics.close_interval(bare, valid_time=None,
                                        output_dir=None, grid_id=1)


def test_the_shared_frame_builder_carries_the_attached_window():
    from woof.io.history_layout import live_state_history_fields

    state = _column_state()
    assert not set(live_state_history_fields(state)) & set(HUB_WIND_OUTPUT_FIELDS)
    interval_diagnostics.attach(state, _diag(state))
    assert set(HUB_WIND_OUTPUT_FIELDS) <= set(live_state_history_fields(state))


def test_attach_is_off_unless_asked_and_refuses_a_streamed_domain():
    state = _column_state()
    ny, nx = state.ht.shape
    assert hw.attach_hub_wind(state, HistorySelection(), nx=nx, ny=ny,
                              dx=100.0, dy=100.0, grid_id=2) is None
    assert interval_diagnostics.attached(state) == ()
    selection = HistorySelection.from_mapping({"hub_wind": True})
    with pytest.raises(hw.HubWindUnsupportedRoad, match="streamed"):
        hw.attach_hub_wind(state, selection, nx=nx, ny=ny, dx=100.0,
                           dy=100.0, grid_id=2, streamed=True)
    diag = hw.attach_hub_wind(state, selection, nx=nx, ny=ny, dx=100.0,
                              dy=100.0, grid_id=2)
    assert diag.heights_m.tolist() == list(DEFAULT_HUB_HEIGHTS_M)
    # Re-attaching the same request (a writer rebuilt on resume) is idempotent.
    assert hw.attach_hub_wind(state, selection, nx=nx, ny=ny, dx=100.0,
                              dy=100.0, grid_id=2) is diag
    sited = HistorySelection.from_mapping(
        {"hub_sites": [{"site_id": "a", "lat": 0.0, "lon": 0.0}]})
    with pytest.raises(hw.HubWindError, match="latitude/longitude"):
        hw.attach_hub_wind(_column_state(), sited, nx=nx, ny=ny, dx=100.0,
                           dy=100.0, grid_id=2)


def test_every_hook_is_wired_where_the_window_needs_it():
    """The fold, the frame and the restart each have exactly one owner."""
    dycore = (REPO / "woof/core/dycore.py").read_text(encoding="utf-8")
    assert "interval_diagnostics.step(state, cfg)" in dycore
    wrfout = (REPO / "woof/io/wrfout.py").read_text(encoding="utf-8")
    assert "interval_diagnostics.close_interval(" in wrfout
    assert "_attach_hub_wind(node, case" in wrfout
    runtime = (REPO / "woof/runtime.py").read_text(encoding="utf-8")
    assert "interval_diagnostics.close_interval(" in runtime
    assert "attach_hub_wind(" in runtime
    child = (REPO / "woof/offline_child_run.py").read_text(encoding="utf-8")
    assert "interval_diagnostics.close_interval(" in child
    assert "attach_hub_wind(" in child


def test_the_inline_kernel_source_is_pinned():
    digest = hashlib.sha256(hw.KERNEL_SOURCE.encode("utf-8")).hexdigest()
    assert digest == hw.KERNEL_SOURCE_SHA256 == KERNEL_SOURCE_SHA256
    assert hw.KERNEL_OPTIONS == ("--fmad=false",)


# ---------------------------------------------------------------------------
# sites and the sidecar
# ---------------------------------------------------------------------------

def _latlon(ny=6, nx=7, lat0=55.0, lon0=2.0, step=0.01):
    lat = lat0 + step * np.arange(ny)[:, None] + 0.0 * np.arange(nx)[None]
    lon = lon0 + step * np.arange(nx)[None, :] + 0.0 * np.arange(ny)[:, None]
    return lat.astype(np.float32), lon.astype(np.float32)


def test_sites_map_to_their_nearest_interior_cell_or_are_outside():
    lat, lon = _latlon()
    sites = (("inside", 55.0302, 2.0398),      # nearest (3, 4)
             ("edge", 55.0, 2.03),             # nearest is on row 0
             ("far", 56.0, 2.03))              # nowhere near
    located = hw.locate_sites(sites, lat, lon, dx=1000.0, dy=1000.0)
    assert [row[0] for row in located["inside"]] == ["inside"]
    site_id, _, _, j, i, cell_lat, cell_lon, dist = located["inside"][0]
    assert (j, i) == (3, 4)
    assert dist < 400.0
    assert located["outside"] == ["edge", "far"]


def test_site_series_is_sampled_on_cadence_and_flushed_per_frame(tmp_path):
    import netCDF4

    lat, lon = _latlon()
    state = _column_state(ny=6, nx=7)
    selection = HistorySelection.from_mapping({
        "hub_heights_m": [60.0, 100.0], "hub_sample_interval_s": 60.0,
        "hub_sites": [{"site_id": "T1", "lat": 55.0302, "lon": 2.0398},
                      {"site_id": "T2", "lat": 55.0199, "lon": 2.0201},
                      {"site_id": "away", "lat": 50.0, "lon": 2.0}]})
    diag = hw.attach_hub_wind(state, selection, nx=7, ny=6, dx=1000.0,
                              dy=1000.0, grid_id=3, latlon=lambda: (lat, lon))
    start = datetime(2025, 3, 1, 12, 0, 0)
    # Cold-start frame: one sample, at the frame's own instant.
    _set_speed(state, 2.0)
    path0 = diag.close_interval(state, valid_time=start, output_dir=tmp_path,
                                grid_id=3)
    assert path0.name == "hubwind_d03_2025-03-01_12:00:00.nc"
    # A 15-minute window of 20 s steps whose speed ramps 1 m/s per step.
    for n in range(45):
        _set_speed(state, 3.0 + n)
        diag.on_step(state, 20.0)
    end = datetime(2025, 3, 1, 12, 15, 0)
    path = diag.close_interval(state, valid_time=end, output_dir=tmp_path,
                               grid_id=3)
    assert path == tmp_path / "hubwind_d03_2025-03-01_12:15:00.nc"
    with netCDF4.Dataset(path) as ds:
        assert ds.variables["WSPD"].dimensions == ("time", "site", "height")
        assert ds.variables["WSPD"].shape == (15, 2, 2)
        assert list(ds.variables["site_id"][:]) == ["T1", "T2"]
        assert ds.sites_outside_domain == "away"
        times = ds.variables["time"][:]
        epoch = (start - datetime(1970, 1, 1)).total_seconds()
        np.testing.assert_allclose(times - epoch, 60.0 * np.arange(1, 16))
        # Sample k is taken at the end of step 3k-1 (speed 3 + 3k - 1).
        want = 3.0 + (3 * np.arange(1, 16) - 1)
        np.testing.assert_allclose(ds.variables["WSPD"][:, 0, 0], want)
        np.testing.assert_allclose(ds.variables["WDIR"][:], 270.0, atol=1e-4)
        assert ds.window_seconds == 900.0
    with netCDF4.Dataset(path0) as ds:
        assert ds.variables["WSPD"].shape == (1, 2, 2)
        np.testing.assert_allclose(ds.variables["WSPD"][:], 2.0)
    # Nothing staged is left behind.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "hubwind_d03_2025-03-01_12:00:00.nc",
        "hubwind_d03_2025-03-01_12:15:00.nc"]


# ---------------------------------------------------------------------------
# [output] parsing
# ---------------------------------------------------------------------------

def test_hub_wind_is_off_by_default_and_full_stays_full():
    selection = HistorySelection.from_mapping({"preset": "full"})
    assert selection.hub_wind is None
    assert HistorySelection.from_mapping(None).hub_wind is None
    assert "hub" not in selection.spelling()


def test_hub_wind_true_takes_the_default_heights():
    selection = HistorySelection.from_mapping({"hub_wind": True})
    assert selection.hub_wind.heights_m == DEFAULT_HUB_HEIGHTS_M == (
        40.0, 80.0, 100.0, 120.0, 150.0, 200.0)
    assert selection.hub_wind.sites == ()
    assert selection.hub_wind.sample_interval_s is None
    assert selection.writes_everything
    assert "hub_heights_m=40,80,100,120,150,200" in selection.spelling()


def test_heights_alone_turn_it_on_and_sites_default_to_a_minute():
    selection = HistorySelection.from_mapping({
        "hub_heights_m": [90, 140.5],
        "hub_sites": [{"site_id": "x", "lat": 10.0, "lon": 350.0}]})
    assert selection.hub_wind.heights_m == (90.0, 140.5)
    assert selection.hub_wind.sample_interval_s == 60.0
    assert selection.hub_wind.sites == (("x", 10.0, -10.0),)


@pytest.mark.parametrize("table, match", [
    ({"hub_heights_m": [100, 80]}, "strictly increasing"),
    ({"hub_heights_m": [100, 100]}, "strictly increasing"),
    ({"hub_heights_m": [0, 80]}, "above"),
    ({"hub_heights_m": [-5]}, "above"),
    ({"hub_heights_m": [2500]}, "above"),
    ({"hub_heights_m": []}, "empty"),
    ({"hub_heights_m": "100"}, "LIST"),
    ({"hub_heights_m": [True]}, "number"),
    ({"hub_heights_m": [float("nan")]}, "finite"),
    ({"hub_heights_m": list(range(1, 40))}, "at most"),
    ({"hub_wind": "yes"}, "true or false"),
    ({"hub_wind": False, "hub_heights_m": [80]}, "hub_wind = false"),
    ({"hub_sample_interval_s": 60}, "without hub_sites"),
    ({"hub_sites": [{"site_id": "a", "lat": 0, "lon": 0}],
      "hub_sample_interval_s": 0}, "positive"),
    ({"hub_sites": []}, "no sites"),
    ({"hub_sites": [{"site_id": "a", "lat": 0}]}, "exactly site_id"),
    ({"hub_sites": [{"site_id": "a", "lat": 0, "lon": 0},
                    {"site_id": "a", "lat": 1, "lon": 1}]}, "repeats"),
    ({"hub_sites": [{"site_id": "a", "lat": 95, "lon": 0}]}, "latitude"),
    ({"hub_sites": "no/such/sites.json"}, "not a file"),
    ({"hub_wind": True, "history_vars": ["T2"]}, "keeps none"),
    ({"hub_wind": True, "history_drop": ["WSPD_HUB", "WDIR_HUB",
                                         "WSPD_HUB_MAX", "WSPD_HUB_MIN",
                                         "WSPD_HUB_MEAN"]}, "keeps none"),
])
def test_contradictions_and_typos_are_refused_by_name(table, match):
    with pytest.raises(HubWindConfigError, match=match):
        HistorySelection.from_mapping(table, source="probe.toml")


def test_unknown_hub_spelling_is_still_an_unknown_key():
    with pytest.raises(ValueError, match="hub_heights_m"):
        HistorySelection.from_mapping({"hub_height_m": [80]})


def test_a_sites_v1_document_is_read_relative_to_the_config(tmp_path):
    from woof.energy.contracts import Site, SiteSet, dump_sites

    sites = SiteSet(sites=[
        Site(site_id="wtg-1", asset_id="farm", kind="turbine",
             lat=53.9, lon=1.9, hub_height_m=110.0),
        Site(site_id="wtg-2", asset_id="farm", kind="turbine",
             lat=53.91, lon=1.92)], heights_m=(110.0,))
    dump_sites(sites, tmp_path / "sites.json")
    selection = HistorySelection.from_mapping(
        {"hub_wind": True, "hub_sites": "sites.json"}, base_dir=tmp_path)
    assert selection.hub_wind.sites == (("wtg-1", 53.9, 1.9),
                                        ("wtg-2", 53.91, 1.92))
    assert selection.hub_wind.sites_source == str(
        (tmp_path / "sites.json").resolve())
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(HubWindConfigError, match="sites.v1"):
        HistorySelection.from_mapping({"hub_sites": "bad.json"},
                                      base_dir=tmp_path)


def test_the_hub_names_join_the_vocabulary_and_the_energy_presets():
    assert set(HUB_WIND_OUTPUT_FIELDS) <= HISTORY_VOCABULARY
    for preset in ("energy", "minimal", "severe"):
        assert set(HUB_WIND_OUTPUT_FIELDS) <= HISTORY_PRESETS[preset], preset
    selection = HistorySelection.from_mapping(
        {"preset": "energy", "hub_wind": True})
    assert all(selection.keeps(name) for name in HUB_WIND_OUTPUT_FIELDS)
    explicit = HistorySelection.from_mapping(
        {"history_vars": ["WSPD_HUB_MAX"], "hub_wind": True})
    assert explicit.keeps("WSPD_HUB_MAX") and not explicit.keeps("WSPD_HUB")


def test_the_experiment_door_accepts_and_validates_the_keys(tmp_path):
    from woof.config import load_history_selection

    path = tmp_path / "case.toml"
    path.write_text('[output]\npreset = "energy"\nhub_heights_m = [100, 150]\n'
                    'hub_sample_interval_s = 30\n'
                    'hub_sites = [{ site_id = "a", lat = 1.0, lon = 2.0 }]\n')
    selection = load_history_selection(path)
    assert selection.hub_wind.heights_m == (100.0, 150.0)
    assert selection.hub_wind.sample_interval_s == 30.0


# ---------------------------------------------------------------------------
# the writer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_hub", [3, 4])
def test_the_writer_puts_the_hub_fields_on_their_own_axis(tmp_path, n_hub):
    import netCDF4
    from woof.io.wrfout import WrfoutWriter

    nz, ny, nx = 4, 5, 6           # n_hub == nz must NOT land on bottom_top
    heights = tuple(float(h) for h in (60, 90, 120, 150)[:n_hub])
    state = _column_state(nz=nz, ny=ny, nx=nx,
                          u_of_z=lambda z: 4.0 + 0.02 * z)
    diag = _diag(state, heights=heights)
    diag.on_step(state, 10.0)
    frame = {"T": np.zeros((nz, ny, nx), np.float32),
             **diag.history_fields(state)}
    path = tmp_path / "wrfout_d01_hub"
    with WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=100.0, dy=100.0,
                      field_schema=frame, engine="python") as writer:
        writer.write_frame("2025-01-01_00:00:00", frame)
    with netCDF4.Dataset(path) as ds:
        assert len(ds.dimensions["hub_height"]) == n_hub
        for name in HUB_HEIGHT_DIMENSIONED_FIELDS:
            var = ds.variables[name]
            assert var.dimensions == ("Time", "hub_height", "south_north",
                                      "west_east"), name
            assert var.MemoryOrder == "XYZ"
        assert ds.variables["HUB_HEIGHT"].dimensions == ("Time", "hub_height")
        assert ds.variables["HUB_WINDOW_S"].dimensions == ("Time",)
        assert ds.variables["WSPD_HUB"].units == "m s-1"
        np.testing.assert_array_equal(ds.variables["HUB_HEIGHT"][0], heights)
        np.testing.assert_array_equal(ds.variables["WSPD_HUB"][0],
                                      frame["WSPD_HUB"])


def test_a_run_without_the_diagnostic_declares_no_hub_axis(tmp_path):
    import netCDF4
    from woof.io.wrfout import WrfoutWriter

    frame = {"T": np.zeros((2, 3, 4), np.float32)}
    path = tmp_path / "wrfout_plain"
    with WrfoutWriter(path, nx=4, ny=3, nz=2, dx=1.0, dy=1.0,
                      field_schema=frame, engine="python") as writer:
        writer.write_frame("2025-01-01_00:00:00", frame)
    with netCDF4.Dataset(path) as ds:
        assert "hub_height" not in ds.dimensions


def test_disk_pricing_counts_the_hub_planes():
    from woof.config import RunConfig
    from woof.io.history_layout import history_frame_bytes

    cfg = RunConfig(nx=20, ny=10, nz=8, dx=1000.0, dy=1000.0, ztop=10000.0,
                    dt=6.0, run_seconds=60.0)
    plain = history_frame_bytes(cfg, HistorySelection())
    hub = history_frame_bytes(
        cfg, HistorySelection.from_mapping({"hub_heights_m": [80, 120]}))
    planes = 5 * 2 * 10 * 20 + 2 + 1
    assert hub - plain == 4 * planes + 256 * len(HUB_WIND_OUTPUT_FIELDS)



# ---------------------------------------------------------------------------
# the writer set's ownership of the window, and the public documents
# ---------------------------------------------------------------------------

def _writer_set(selection, state):
    from woof.io.wrfout import PerDomainWrfoutWriters

    ny, nx = state.ht.shape
    writers = PerDomainWrfoutWriters.__new__(PerDomainWrfoutWriters)
    writers.history_selection = selection
    node = SimpleNamespace(
        state=state, grid=SimpleNamespace(latlon_mass=None),
        cfg=SimpleNamespace(grid_id=2, output=None,
                            run=SimpleNamespace(nx=nx, ny=ny, dx=100.0,
                                                dy=100.0)))
    writers.model = SimpleNamespace(_prepared_by_grid_id={2: SimpleNamespace()},
                                    node=lambda gid: node)
    return writers, node


def test_a_replaced_state_is_reattached_only_when_nothing_was_folded():
    selection = HistorySelection.from_mapping({"hub_wind": True})
    writers, node = _writer_set(selection, _column_state())
    writers._ensure_hub_wind(node)                 # first frame: attach
    first = interval_diagnostics.get(node.state, hw.DIAGNOSTIC_NAME)
    assert first is not None
    # A delayed-start nest rebuilt at activation: nothing folded yet.
    node.state = _column_state()
    writers._ensure_hub_wind(node)
    second = interval_diagnostics.get(node.state, hw.DIAGNOSTIC_NAME)
    assert second is not None and second is not first
    # A state replaced mid-window must not publish a silently short window.
    second.on_step(node.state, 10.0)
    node.state = _column_state()
    with pytest.raises(hw.HubWindError, match="replaced"):
        writers._ensure_hub_wind(node)
    # Off: never touches the state.
    off, bare = _writer_set(HistorySelection(), _column_state())
    off._ensure_hub_wind(bare)
    assert interval_diagnostics.attached(bare.state) == ()


def test_relocation_is_refused_from_the_selection_not_the_state():
    selection = HistorySelection.from_mapping({"hub_wind": True})
    writers, node = _writer_set(selection, _column_state())
    # The mover has already swapped in a fresh state carrying no window.
    with pytest.raises(hw.HubWindUnsupportedRoad, match="relocated"):
        writers.refresh_domain(2, grid=None, static_fields=None)


def test_the_output_key_table_is_declared_whole():
    from woof.config import declared_key_rows
    from woof.io.history_selection import OUTPUT_KEYS

    rows = declared_key_rows()
    assert set(rows["output"]) == set(OUTPUT_KEYS)
    assert set(rows["domain.output"]) == set(OUTPUT_KEYS)


def test_an_output_block_that_does_not_ask_keeps_its_public_bytes():
    from woof.experiment import _public_config_value

    assert "hub_wind" not in _public_config_value(HistorySelection())
    on = _public_config_value(HistorySelection.from_mapping({"hub_wind": True}))
    assert tuple(on["hub_wind"]["heights_m"]) == DEFAULT_HUB_HEIGHTS_M
