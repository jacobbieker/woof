"""WSPD10MAX, U10MAX/V10MAX, W_UP_MAX/W_DN_MAX and GUST (woof/core/wind_gust_diag.py).

Under test: the WRF ``nwp_diagnostics`` semantics of the wind members (per-
step running maxima over the history interval, reset after each history
write, restart-carried), the UPP CALGUST gust method on hand-computed
columns, emission and the ``energy`` preset, every history publisher's
reset, and -- GPU tier -- the CUDA kernel bit for bit against the NumPy
mirror, alone and over a tiny dycore run with history resets.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core import wind_gust_diag as wg
from woof.core.interval_maxima import (HISTORY_INTERVAL_SLOTS,
                                       history_interval_slots,
                                       reset_history_interval_maxima)
from woof.io import restart

from test_restart import _cfg, _fill_setup, _shim_state

REPO = Path(__file__).resolve().parents[1]
G32 = np.float32(wg.G)
F32 = np.float32


def _column(nz_heights, *, ny=1, nx=1):
    """Flat ph/phb whose full-level heights are ``nz_heights`` metres."""
    zw = np.asarray(nz_heights, dtype=np.float64)
    phb = (zw * float(G32)).astype(F32)
    ph = np.zeros((zw.size, ny, nx), dtype=F32)
    return ph, phb


def _maxima(ny=1, nx=1):
    return {slot: np.zeros((ny, nx), dtype=F32) for slot in wg.WIND_MAXIMA_SLOTS}


# ---------------------------------------------------------------------------
# The NumPy reference, member by member, against hand-computed columns
# ---------------------------------------------------------------------------

def test_gust_is_upp_calgust_over_the_pbl_plus_one_level():
    """SFCWIND + (|V_k| - SFCWIND)*(1 - min(0.5, dz/2000)), k = 1..kpbl+1."""
    ph, phb = _column([0.0, 100.0, 300.0, 600.0, 1000.0])  # zm 50/200/450/800
    nz = 4
    u = np.zeros((nz, 1, 2), dtype=F32)
    for k, speed in enumerate((1.0, 10.0, 20.0, 30.0)):
        u[k] = speed
    v = np.zeros((nz, 2, 1), dtype=F32)
    u10, v10 = np.full((1, 1), 3.0, F32), np.full((1, 1), 4.0, F32)  # |V10|=5

    def gust(pblh):
        return float(wg.gust_np(u, v, ph, phb, u10, v10,
                                np.full((1, 1), pblh, F32))[0, 0])

    # PBLH 250 m: kpbl = 2 (450 m), levels 1..3 -> 9.5, 16.625, 20.0.
    assert gust(250.0) == pytest.approx(20.0, rel=1e-6)
    # PBLH 100 m: kpbl = 1 (200 m), levels 1..2.
    assert gust(100.0) == pytest.approx(16.625, rel=1e-6)
    # PBLH 0: kpbl = 0, still level 1 (UPP always reads LM-1).
    assert gust(0.0) == pytest.approx(9.5, rel=1e-6)
    # Above the column: every level but the lowest.
    assert gust(5000.0) == pytest.approx(20.0, rel=1e-6)
    # The reduction caps at 0.5 above 1 km and the gust never falls
    # below the 10 m wind when aloft is calmer.
    u[1:] = 0.0
    assert gust(5000.0) == pytest.approx(5.0, rel=1e-6)


def test_pbl_top_level_is_the_lowest_mass_level_at_or_above_pblh():
    zagl = np.asarray([50.0, 200.0, 450.0, 800.0], dtype=F32)[:, None, None]
    got = [int(wg.pbl_top_level_np(zagl, np.full((1, 1), h, F32))[0, 0])
           for h in (0.0, 50.0, 51.0, 200.0, 799.0, 800.0, 801.0)]
    assert got == [0, 0, 1, 1, 3, 3, 3]


def test_p8w_is_phy_prep_and_w_extrema_take_only_levels_below_400_hpa():
    ph, phb = _column([0.0, 100.0, 300.0, 600.0, 1000.0])
    p = np.asarray([90000.0, 70000.0, 45000.0, 30000.0],
                   dtype=F32)[:, None, None]
    fnm = np.full(4, 0.5, F32)
    fnp = np.full(4, 0.5, F32)
    p8w = wg.p8w_np(p, ph, phb, fnm, fnp)[:, 0, 0]
    # Interior fnm*p(k) + fnp*p(k-1).
    assert p8w[1] == F32(80000.0) and p8w[2] == F32(57500.0)
    assert p8w[3] == F32(37500.0)
    # Bottom: linear extrapolation in height, w1 = (0-200)/(50-200).
    assert p8w[0] == pytest.approx(4.0 / 3.0 * 90000.0 - 1.0 / 3.0 * 70000.0,
                                   rel=1e-6)
    # Top: log-linear, w1 = (1000-450)/(800-450).
    w1 = 550.0 / 350.0
    assert p8w[4] == pytest.approx(
        np.exp(w1 * np.log(30000.0) + (1 - w1) * np.log(45000.0)), rel=1e-5)

    w = np.asarray([0.0, 5.0, -3.0, 12.0, -20.0], dtype=F32)[:, None, None]
    up = np.zeros((1, 1), F32)
    dn = np.zeros((1, 1), F32)
    wg.fold_w_extrema_np(w, wg.p8w_np(p, ph, phb, fnm, fnp), up, dn)
    assert (up[0, 0], dn[0, 0]) == (F32(5.0), F32(-3.0))
    # Running: a weaker later step leaves both extremes alone.
    wg.fold_w_extrema_np(np.float32(0.1) * w, wg.p8w_np(p, ph, phb, fnm, fnp),
                         up, dn)
    assert (up[0, 0], dn[0, 0]) == (F32(5.0), F32(-3.0))


def test_u10max_v10max_are_the_components_at_the_speed_maximum():
    maxima = _maxima(1, 2)
    steps = [((3.0, -1.0), (4.0, 0.5)),     # speeds 5, ~1.118
             ((-6.0, 1.0), (0.0, 0.0)),     # speeds 6, 1
             ((1.0, -2.0), (1.0, 0.0))]     # speeds ~1.41, 2
    for (u10, v10) in steps:
        wg.fold_wind10_np(np.asarray([u10], F32), np.asarray([v10], F32),
                          maxima["wspd10max"], maxima["u10max"],
                          maxima["v10max"])
    np.testing.assert_allclose(maxima["wspd10max"], [[6.0, 2.0]], rtol=1e-7)
    np.testing.assert_array_equal(maxima["u10max"], [[-6.0, -2.0]])
    np.testing.assert_array_equal(maxima["v10max"], [[0.0, 0.0]])


# ---------------------------------------------------------------------------
# Lifecycle on real DomainState objects (NumPy-backed)
# ---------------------------------------------------------------------------

def test_state_allocates_every_slot_eagerly_under_the_knob(monkeypatch):
    on = _shim_state(_cfg(nwp_diagnostics=1), monkeypatch)
    for slot in wg.WIND_MAXIMA_SLOTS:
        assert on.existing_scratch(slot).shape == (4, 6), slot
    off = _shim_state(_cfg(), monkeypatch)
    for slot in wg.WIND_MAXIMA_SLOTS:
        assert off.existing_scratch(slot) is None, slot


def _loaded_numpy_state(monkeypatch, **overrides):
    from woof.core.grid import make_base_state, make_vertical_coord

    cfg = _cfg(nx=7, ny=5, nz=12, ztop=12000.0, specified=True,
               nwp_diagnostics=1, **overrides)
    state = _shim_state(cfg, monkeypatch)
    vc = make_vertical_coord(cfg.nz)
    base = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state.load_base(vc, base)
    pb = np.asarray(state.pb, dtype=F32)
    state.p[...] = (pb if pb.ndim == 3 else pb[:, None, None])
    return state, cfg


def _surface(ny, nx, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(fields={
        "u10": (scale * rng.uniform(-12, 12, (ny, nx))).astype(F32),
        "v10": (scale * rng.uniform(-12, 12, (ny, nx))).astype(F32),
        "pblh": rng.uniform(0, 2500, (ny, nx)).astype(F32)})


def _winds(state, cfg, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    state.u[...] = (scale * rng.uniform(-25, 25, state.u.shape)).astype(F32)
    state.v[...] = (scale * rng.uniform(-25, 25, state.v.shape)).astype(F32)
    state.w[...] = (scale * rng.uniform(-6, 6, state.w.shape)).astype(F32)


def _mirror(state, surface, maxima):
    fields = None if surface is None else surface.fields
    wg.mirror_wind_maxima_step_np(
        state.u, state.v, state.w, state.p, state.php, state.phb,
        state.fnm, state.fnp,
        None if fields is None else fields["u10"],
        None if fields is None else fields["v10"],
        None if fields is None else fields["pblh"], maxima)


def test_update_accumulates_resets_and_matches_the_mirror(monkeypatch):
    state, cfg = _loaded_numpy_state(monkeypatch)
    ny, nx = cfg.ny, cfg.nx
    mirror = _maxima(ny, nx)
    state.physics = _surface(ny, nx, 1)
    _winds(state, cfg, 2)
    wg.update_wind_maxima(state, cfg)
    _mirror(state, state.physics, mirror)
    first = {s: state.existing_scratch(s).copy() for s in wg.WIND_MAXIMA_SLOTS}
    for slot in wg.WIND_MAXIMA_SLOTS:
        np.testing.assert_array_equal(first[slot], mirror[slot], err_msg=slot)
    assert first["gust_max"].min() >= first["wspd10max"].min()
    assert (first["gust_max"] >= first["wspd10max"]).all()
    assert first["w_up_max"].max() > 0.0 and first["w_dn_max"].min() < 0.0

    # A weaker step: every running max holds.
    state.physics = _surface(ny, nx, 1, scale=0.25)
    _winds(state, cfg, 2, scale=0.25)
    wg.update_wind_maxima(state, cfg)
    for slot in wg.WIND_MAXIMA_SLOTS:
        np.testing.assert_array_equal(state.existing_scratch(slot),
                                      first[slot], err_msg=slot)

    # The history reset opens a new window: the weak step alone is smaller.
    reset_history_interval_maxima(state)
    for slot in wg.WIND_MAXIMA_SLOTS:
        assert (state.existing_scratch(slot) == 0.0).all(), slot
    wg.update_wind_maxima(state, cfg)
    assert (state.existing_scratch("wspd10max")
            < first["wspd10max"]).all()
    assert state.existing_scratch("gust_max").max() < first["gust_max"].max()


def test_without_a_surface_layer_only_the_w_extrema_fold(monkeypatch):
    """WRF's u10 is zero without sf_sfclay_physics; woof folds nothing."""
    state, cfg = _loaded_numpy_state(monkeypatch)
    assert state.physics is None
    _winds(state, cfg, 5)
    wg.update_wind_maxima(state, cfg)
    for slot in ("wspd10max", "u10max", "v10max", "gust_max"):
        assert (state.existing_scratch(slot) == 0.0).all(), slot
    assert state.existing_scratch("w_up_max").max() > 0.0


def test_a_missing_pbl_height_skips_only_the_gust(monkeypatch):
    state, cfg = _loaded_numpy_state(monkeypatch)
    state.physics = _surface(cfg.ny, cfg.nx, 8)
    del state.physics.fields["pblh"]
    _winds(state, cfg, 9)
    wg.update_wind_maxima(state, cfg)
    assert (state.existing_scratch("wspd10max") > 0.0).all()
    assert (state.existing_scratch("gust_max") == 0.0).all()


def test_the_history_reset_leaves_up_heli_max_and_tracker_windows_alone(
        monkeypatch):
    state = _shim_state(_cfg(nwp_diagnostics=1), monkeypatch)
    for slot in ("up_heli_max", "uh_follow_window", *wg.WIND_MAXIMA_SLOTS):
        state.existing_scratch(slot)[...] = 9.0
    reset_history_interval_maxima(state)
    for slot in wg.WIND_MAXIMA_SLOTS:
        assert (state.existing_scratch(slot) == 0.0).all()
    assert (state.existing_scratch("up_heli_max") == 9.0).all()
    assert (state.existing_scratch("uh_follow_window") == 9.0).all()
    # Knob off: a clean no-op.
    reset_history_interval_maxima(_shim_state(_cfg(), monkeypatch))
    # And the registry carries exactly this module's slots in this order.
    assert HISTORY_INTERVAL_SLOTS["woof.core.wind_gust_diag"] \
        == wg.WIND_MAXIMA_SLOTS
    assert set(wg.WIND_MAXIMA_SLOTS) <= set(history_interval_slots())


def test_streamed_store_copy_is_zeroed_too():
    from woof.core.interval_maxima import reset_streamed_history_interval_maxima

    store = {f"scratch/{slot}": np.full((2, 2), 4.0, F32)
             for slot in wg.WIND_MAXIMA_SLOTS}
    store["scratch/up_heli_max"] = np.full((2, 2), 4.0, F32)
    reset_streamed_history_interval_maxima(SimpleNamespace(store=store))
    for slot in wg.WIND_MAXIMA_SLOTS:
        assert (store[f"scratch/{slot}"] == 0.0).all()
    assert (store["scratch/up_heli_max"] == 4.0).all()
    reset_streamed_history_interval_maxima(None)


def test_tree_history_handler_snapshots_then_resets(monkeypatch):
    from woof.runtime import _submit_tree_history_frame

    state = _shim_state(_cfg(nwp_diagnostics=1), monkeypatch)
    state.existing_scratch("gust_max")[...] = 31.0
    seen = {}

    class _Writers:
        def submit(self, node, ticks, *, refl_field=None):
            seen["gust"] = node.state.existing_scratch("gust_max").copy()

    _submit_tree_history_frame(_Writers(), SimpleNamespace(state=state), 0)
    assert (seen["gust"] == 31.0).all()
    assert (state.existing_scratch("gust_max") == 0.0).all()


def test_every_history_publisher_resets_the_wind_maxima():
    """Derived like test_uh_lifecycle's: who builds PerDomainWrfoutWriters."""
    from test_uh_lifecycle import _RESET_DELEGATES, _history_publishing_modules

    modules = _history_publishing_modules()
    assert "woof/prepared_single_domain_forecast.py" in modules
    delinquent = [
        name for name, text in modules.items()
        if "reset_history_interval_maxima" not in text
        and not any(delegate in text for delegate in _RESET_DELEGATES)]
    assert delinquent == []
    runtime = (REPO / "woof/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(runtime)
    submit = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_submit_tree_history_frame")
    assert "reset_history_interval_maxima" in ast.unparse(submit)
    # Every other publisher that resets UP_HELI_MAX resets these too.
    for rel in ("woof/runtime.py", "woof/offline_child_run.py",
                "woof/ensemble/native_forecast.py",
                "woof/prepared_single_domain_forecast.py"):
        calls = [node.func for node in ast.walk(ast.parse(
                     (REPO / rel).read_text(encoding="utf-8")))
                 if isinstance(node, ast.Call)]
        names = [getattr(f, "id", getattr(f, "attr", None)) for f in calls]
        assert (names.count("reset_up_heli_max")
                <= names.count("reset_history_interval_maxima")), rel


def test_no_model_reader_of_the_wind_maxima():
    """The accumulators feed output only: no module outside the sanctioned
    owner/allocation/emission/restore/pricing sites names a slot through the
    scratch API."""
    sanctioned = {"woof/core/wind_gust_diag.py", "woof/core/state.py",
                  "woof/io/history_layout.py", "woof/io/restart.py"}
    for path in (REPO / "woof").rglob("*.py"):
        rel = path.relative_to(REPO).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("scratch", "existing_scratch")
                    and any(isinstance(arg, ast.Constant)
                            and arg.value in wg.WIND_MAXIMA_SLOTS
                            for arg in node.args)):
                assert rel in sanctioned, f"unsanctioned slot access in {rel}"


# ---------------------------------------------------------------------------
# Emission, schema, presets, restart
# ---------------------------------------------------------------------------

def test_history_fields_schema_and_planner_carry_the_six(monkeypatch):
    from woof.io.history_layout import produced_history_shapes
    from woof.io.wrfout import _VAR_META, _live_state_history_fields

    names = [name for name, _ in wg.WIND_MAXIMA_FIELDS]
    assert names == ["WSPD10MAX", "U10MAX", "V10MAX", "W_UP_MAX",
                     "W_DN_MAX", "GUST"]
    # WRF Registry.EM_COMMON:2079-2081 attribute text, verbatim.
    assert _VAR_META["WSPD10MAX"] == ("WIND SPD MAX 10 M", "m s-1")
    assert _VAR_META["W_UP_MAX"] == ("MAX Z-WIND UPDRAFT", "m s-1")
    assert _VAR_META["W_DN_MAX"] == ("MAX Z-WIND DOWNDRAFT", "m s-1")
    for name in names:
        assert _VAR_META[name][1] == "m s-1"
    state = _shim_state(_cfg(nwp_diagnostics=1), monkeypatch)
    fields = _live_state_history_fields(state)
    for name, slot in wg.WIND_MAXIMA_FIELDS:
        assert fields[name] is state.existing_scratch(slot)
    off = _live_state_history_fields(_shim_state(_cfg(), monkeypatch))
    assert not set(names) & set(off)

    on_shapes = produced_history_shapes(_cfg(nwp_diagnostics=1))
    off_shapes = produced_history_shapes(_cfg())
    assert set(on_shapes) - set(off_shapes) == {"UP_HELI_MAX", *names}
    for name in names:
        assert on_shapes[name] == (4, 6)


def test_energy_preset_and_vocabulary_name_the_six():
    from woof.io.history_selection import (HISTORY_PRESETS, HISTORY_VOCABULARY,
                                           HistorySelection)

    names = {name for name, _ in wg.WIND_MAXIMA_FIELDS}
    assert names <= HISTORY_VOCABULARY
    assert names <= HISTORY_PRESETS["energy"]
    assert HISTORY_PRESETS["full"] is None     # full keeps them when produced
    HistorySelection(history_vars=tuple(sorted(names)))   # all accepted


def test_restart_roundtrip_carries_every_running_max(monkeypatch, tmp_path):
    cfg = _cfg(nwp_diagnostics=1)
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    patterns = {}
    for index, slot in enumerate(wg.WIND_MAXIMA_SLOTS):
        patterns[slot] = (np.arange(24, dtype=F32).reshape(4, 6)
                          + F32(100 * index))
        source.existing_scratch(slot)[...] = patterns[slot]
    path = restart.write_restart(tmp_path / "wind.npz", source, cfg)
    with np.load(path, allow_pickle=False) as archive:
        for slot in wg.WIND_MAXIMA_SLOTS:
            assert f"scratch/{slot}" in archive.files
    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)
    for slot in wg.WIND_MAXIMA_SLOTS:
        np.testing.assert_array_equal(fresh.existing_scratch(slot),
                                      patterns[slot])
    for slot in wg.WIND_MAXIMA_SLOTS:
        assert restart.classify_scratch_slot(slot) == "serialize"


def test_old_checkpoint_without_the_slots_restores_zeroed_with_a_note(
        monkeypatch, tmp_path, capsys):
    cfg = _cfg(nwp_diagnostics=1)
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    path = restart.write_restart(tmp_path / "new.npz", source, cfg)
    # Age the file: drop the six slots a pre-diagnostic build never wrote,
    # from the payload and from the header's own array manifest.
    import json

    dropped = {f"scratch/{slot}" for slot in wg.WIND_MAXIMA_SLOTS}
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files
                   if name not in dropped}
    header = json.loads(bytes(bytearray(
        payload[restart._HEADER_KEY])).decode("utf-8"))
    for key in dropped:
        header["array_manifest"].pop(key)
    payload[restart._HEADER_KEY] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)
    aged = tmp_path / "aged.npz"
    with aged.open("wb") as stream:
        np.savez(stream, **payload)
    live = _shim_state(cfg, monkeypatch)
    _fill_setup(live)
    for slot in wg.WIND_MAXIMA_SLOTS:
        live.existing_scratch(slot)[...] = 77.0
    restart.restore_restart(aged, live, cfg)
    for slot in wg.WIND_MAXIMA_SLOTS:
        assert (live.existing_scratch(slot) == 0.0).all(), slot
    out = capsys.readouterr().out
    assert "note:" in out and "gust_max" in out


def test_diagnostics_off_resume_drops_the_stored_maxima_with_a_note(
        monkeypatch, tmp_path, capsys):
    on_cfg = _cfg(nwp_diagnostics=1)
    source = _shim_state(on_cfg, monkeypatch)
    _fill_setup(source)
    source.existing_scratch("gust_max")[...] = 12.5
    path = restart.write_restart(tmp_path / "on.npz", source, on_cfg)
    off_cfg = _cfg()
    live = _shim_state(off_cfg, monkeypatch)
    _fill_setup(live)
    restart.restore_restart(path, live, off_cfg)
    for slot in wg.WIND_MAXIMA_SLOTS:
        assert live.existing_scratch(slot) is None
    out = capsys.readouterr().out
    assert "note:" in out and "'gust_max'" in out and "dropped" in out


def test_namelist_and_config_accept_the_knob_and_name_the_members():
    from woof.config import RunConfig, validate_run_config

    with pytest.raises(ValueError, match="WSPD10MAX"):
        validate_run_config(RunConfig(nx=6, ny=4, nz=5, dx=2000.0, dy=2000.0,
                                  ztop=8000.0, dt=10.0, run_seconds=0.0,
                                  nwp_diagnostics=2))


# ---------------------------------------------------------------------------
# GPU tier
# ---------------------------------------------------------------------------

def _device_numpy_pair(terrain: bool, cp):
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import DomainState

    cfg = _cfg(nx=13, ny=11, nz=20, ztop=15000.0, specified=True,
               nwp_diagnostics=1)
    host = DomainState(cfg, array_module=np)
    device = DomainState(cfg)
    vc = make_vertical_coord(cfg.nz)
    base = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    host.load_base(vc, base)
    device.load_base(vc, base)
    rng = np.random.default_rng(11)
    pb = np.asarray(host.pb, dtype=F32)
    pb3 = pb if pb.ndim == 3 else np.broadcast_to(pb[:, None, None],
                                                  host.p.shape)
    arrays = {
        "u": rng.uniform(-30, 30, host.u.shape).astype(F32),
        "v": rng.uniform(-30, 30, host.v.shape).astype(F32),
        "w": rng.uniform(-8, 8, host.w.shape).astype(F32),
        "p": (pb3 + rng.uniform(-300, 300, host.p.shape)).astype(F32),
        "php": rng.uniform(-50, 50, host.php.shape).astype(F32),
    }
    if terrain:
        phb = np.asarray(host.phb, dtype=F32)
        phb3 = (np.broadcast_to(phb[:, None, None], host.php.shape)
                + rng.uniform(0, 3000, (1, cfg.ny, cfg.nx)).astype(F32))
        host.phb = np.ascontiguousarray(phb3, dtype=F32)
        device.phb = cp.asarray(host.phb)
    for name, value in arrays.items():
        getattr(host, name)[...] = value
        getattr(device, name)[...] = cp.asarray(value)
    return cfg, host, device


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("terrain", [False, True])
@pytest.mark.parametrize("with_surface", ["full", "no_pblh", "none"])
def test_gpu_kernel_matches_the_numpy_mirror_bitwise(terrain, with_surface):
    import cupy as cp

    cfg, host, device = _device_numpy_pair(terrain, cp)
    if with_surface != "none":
        surface = _surface(cfg.ny, cfg.nx, 3)
        if with_surface == "no_pblh":
            del surface.fields["pblh"]
        host.physics = surface
        device.physics = SimpleNamespace(fields={
            k: cp.asarray(v) for k, v in surface.fields.items()})
    for _ in range(2):
        wg.update_wind_maxima(host, cfg)
        wg.update_wind_maxima(device, cfg)
    for slot in wg.WIND_MAXIMA_SLOTS:
        np.testing.assert_array_equal(
            cp.asnumpy(device.existing_scratch(slot)),
            host.existing_scratch(slot), err_msg=slot)
    assert float(host.existing_scratch("w_up_max").max()) > 0.0
    if with_surface == "full":
        assert float(host.existing_scratch("gust_max").min()) > 0.0
    if with_surface == "no_pblh":
        assert float(host.existing_scratch("wspd10max").min()) > 0.0
        assert float(host.existing_scratch("gust_max").max()) == 0.0


@pytest.mark.gpu
@requires_gpu
def test_gpu_tiny_run_resets_per_history_interval_and_matches_numpy(tmp_path):
    """60 model seconds on a 9x8x30 dycore domain, history every 15 s.

    The surface layer is a deterministic stand-in (10 m wind decaying
    through the run, PBLH rising), so each interval's maxima are known to
    shrink after the reset; the NumPy mirror is folded from host copies of
    every completed step and compared bit for bit at each history time.
    """
    import cupy as cp

    from woof.core.dycore import step
    from woof.io.wrfout import _live_state_history_fields
    from test_mp_spec_zone_ring import _square_moist_state

    state, cfg = _square_moist_state(nz=30, mp=1, dt=5.0, open_x=True,
                                     open_y=True, nwp_diagnostics=1)
    ny, nx = cfg.ny, cfg.nx
    # A warm bubble's worth of vertical motion so W_UP/W_DN are not trivial.
    rng = np.random.default_rng(7)
    state.w[1:-1] += cp.asarray(
        rng.uniform(-1.5, 1.5, state.w[1:-1].shape).astype(F32))
    state.u[...] += cp.asarray(
        rng.uniform(5, 15, state.u.shape).astype(F32))

    class _Surface:
        fields = {}

        def accept_microphysics(self, *args, **kwargs):
            return None

    surface = _Surface()
    state.physics = surface
    host_max = _maxima(ny, nx)
    history_every = 3       # steps: 15 s at dt = 5 s
    previous = None
    base = rng.uniform(4, 10, (ny, nx)).astype(F32)
    for n in range(1, 13):  # 60 s
        decay = F32(1.0 - 0.06 * n)
        surface.fields = {
            "u10": cp.asarray((decay * base).astype(F32)),
            "v10": cp.asarray((decay * F32(0.5) * base).astype(F32)),
            "pblh": cp.asarray(np.full((ny, nx), 200.0 + 40.0 * n, F32)),
        }
        step(state, cfg)
        host_surface = SimpleNamespace(fields={
            k: cp.asnumpy(v) for k, v in surface.fields.items()})
        snapshot = SimpleNamespace(**{
            name: cp.asnumpy(getattr(state, name))
            for name in ("u", "v", "w", "p", "php", "phb", "fnm", "fnp")})
        _mirror(snapshot, host_surface, host_max)
        if n % history_every:
            continue
        frame = _live_state_history_fields(state)
        got = {slot: cp.asnumpy(frame[name])
               for name, slot in wg.WIND_MAXIMA_FIELDS}
        for slot in wg.WIND_MAXIMA_SLOTS:
            np.testing.assert_array_equal(got[slot], host_max[slot],
                                          err_msg=f"{slot} at step {n}")
        assert got["gust_max"].min() >= got["wspd10max"].min()
        assert float(got["w_up_max"].max()) > 0.0
        if previous is not None:
            # A decaying 10 m wind: every column's interval max is lower
            # than the previous interval's, which only a reset allows.
            assert (got["wspd10max"] < previous["wspd10max"]).all()
        previous = got
        reset_history_interval_maxima(state)
        for slot in wg.WIND_MAXIMA_SLOTS:
            host_max[slot][...] = 0.0
            assert float(cp.abs(state.existing_scratch(slot)).max()) == 0.0


def test_registry_shaped_adapter_folds_emits_and_resets(monkeypatch):
    """WindMaximaIntervalDiagnostic == the direct hooks, for registry use."""
    state, cfg = _loaded_numpy_state(monkeypatch)
    twin, _ = _loaded_numpy_state(monkeypatch)
    for target in (state, twin):
        target.physics = _surface(cfg.ny, cfg.nx, 4)
        _winds(target, cfg, 6)
    member = wg.WindMaximaIntervalDiagnostic()
    member.on_step(state, cfg.dt)
    wg.update_wind_maxima(twin, cfg)
    fields = member.history_fields(state)
    assert list(fields) == [name for name, _ in wg.WIND_MAXIMA_FIELDS]
    for name, slot in wg.WIND_MAXIMA_FIELDS:
        np.testing.assert_array_equal(fields[name],
                                      twin.existing_scratch(slot))
    member.refuse_publisher("any publisher")          # never refuses
    member.close_interval(state, valid_time=None, output_dir=None, grid_id=1)
    for slot in wg.WIND_MAXIMA_SLOTS:
        assert (state.existing_scratch(slot) == 0.0).all()
    # Knob off: nothing allocated, nothing folded, nothing emitted.
    off = _shim_state(_cfg(), monkeypatch)
    member.on_step(off, 10.0)
    assert member.history_fields(off) == {}
