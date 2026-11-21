"""spacewx_ionosphere_opt: the climatological ionosphere in every history frame.

Admission, the checkpoint echo, the history inventory and the refusals (hex,
streamed) run on the CPU.  The card tests run the real ``initialize_physics``
and check that the published rows are the device kernel's output at each
frame's valid time on the radiation columns, that they are held while the
valid time does not move, and that the atmosphere is untouched.  Every input
is synthetic (CCIR-format maps, an axial-dipole IGRF table, a GFZ-format
index file or a fixed flux), so nothing is fetched.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from woof.config import ConfigurationRefused, RunConfig, validate_run_config
from woof.spacewx import history as hist
from woof.spacewx import igrf
from woof.spacewx import ionosphere as ion

from conftest import requires_gpu  # noqa: E402
from test_spacewx_ionosphere import _dipole, _loader  # noqa: E402

START = datetime(2024, 5, 10, 18, 0)
NAMES = ("IONO_TEC", "IONO_FOF2", "IONO_HMF2", "IONO_NMF2", "IONO_FOE")


def _cfg(**overrides):
    values = dict(nx=6, ny=4, nz=5, dx=3000.0, dy=3000.0, ztop=20000.0,
                  dt=20.0, run_seconds=3600.0, ra_physics=4)
    values.update(overrides)
    return RunConfig(**values)


def _gfz(tmp_path, f107=150.0):
    """A GFZ-format index file: 90 days ending 2024-05-12, constant indices."""
    from datetime import date, timedelta
    rows = ["# synthetic, GFZ Kp_ap_Ap_SN_F107 layout"]
    day = date(2024, 2, 13)
    for _ in range(90):
        kp = " ".join(["2.000"] * 8)
        ap = " ".join(["7"] * 8)
        rows.append(f"{day.year} {day.month:02d} {day.day:02d} 0.0 0.0 0 0 {kp} {ap} 7 100 "
                    f"{f107:.1f} {f107:.1f} 2")
        day += timedelta(days=1)
    path = tmp_path / "indices.txt"
    path.write_text("\n".join(rows) + "\n", encoding="ascii")
    return path


def _radiation(ny=4, nx=6):
    lat, lon = np.meshgrid(np.linspace(-40.0, 50.0, ny), np.linspace(-120.0, 100.0, nx), indexing="ij")
    return SimpleNamespace(start_time=START, latitude_deg=lat.astype(np.float32),
                           longitude_deg=lon.astype(np.float32))


# --- admission ----------------------------------------------------------------------

def test_defaults_are_off_and_admitted():
    cfg = _cfg()
    assert (cfg.spacewx_ionosphere_opt, cfg.spacewx_f107_source, cfg.spacewx_f107_mode,
            cfg.spacewx_ionosphere_maps) == (0, "", "81day", "ccir")
    validate_run_config(cfg)
    assert hist.history_shapes(cfg) == {} and hist.device_shapes(cfg) == {}


@pytest.mark.parametrize("overrides, match", [
    (dict(spacewx_ionosphere_opt=3), "spacewx_ionosphere_opt"),
    (dict(spacewx_ionosphere_opt=2, spacewx_f107_source="150"), "reads ap from the GFZ index file"),
    (dict(spacewx_ionosphere_opt=2, ra_physics=0, ra_sw_physics=0, ra_lw_physics=0),
     "spacewx_ionosphere_opt = 2 places"),
    (dict(spacewx_ionosphere_opt=True), "spacewx_ionosphere_opt"),
    (dict(spacewx_ionosphere_opt=1, spacewx_f107_mode="27day"), "spacewx_f107_mode"),
    (dict(spacewx_ionosphere_opt=1, spacewx_ionosphere_maps="iri"), "spacewx_ionosphere_maps"),
    (dict(spacewx_ionosphere_opt=1, spacewx_f107_source="-5"), "not a solar flux"),
    (dict(spacewx_ionosphere_opt=1, spacewx_f107_source="nan"), "not a solar flux"),
    (dict(spacewx_f107_source="gfz"), "would do nothing"),
    (dict(spacewx_f107_mode="daily"), "would do nothing"),
    (dict(spacewx_ionosphere_maps="ursi"), "would do nothing"),
    (dict(spacewx_ionosphere_opt=1, ra_physics=0, ra_sw_physics=0, ra_lw_physics=0),
     "no radiation scheme"),
])
def test_refusals_name_the_breakage(overrides, match):
    with pytest.raises(ConfigurationRefused, match=match):
        validate_run_config(_cfg(**overrides))


@pytest.mark.parametrize("source", ["", "gfz", "152.5", "/data/indices.txt"])
def test_an_on_config_is_admitted(source):
    validate_run_config(_cfg(spacewx_ionosphere_opt=1, spacewx_f107_source=source,
                             spacewx_f107_mode="daily", spacewx_ionosphere_maps="ursi"))


@pytest.mark.parametrize("source", ["", "gfz", "/data/indices.txt"])
def test_the_storm_correction_is_admitted_with_an_index_file(source):
    validate_run_config(_cfg(spacewx_ionosphere_opt=2, spacewx_f107_source=source))


def test_f107_source_forms():
    assert hist.parse_f107_source("") == ("gfz", None)
    assert hist.parse_f107_source(" gfz ") == ("gfz", None)
    assert hist.parse_f107_source("152.5") == ("value", 152.5)
    assert hist.parse_f107_source("a/b.txt") == ("file", "a/b.txt")


def test_checkpoint_echo_omits_the_off_keys_and_keeps_the_on_ones():
    from woof.io import restart
    values = dataclasses.asdict(_cfg())
    restart._drop_default_off_run_keys(values)
    assert not {"spacewx_ionosphere_opt", "spacewx_f107_source", "spacewx_f107_mode",
                "spacewx_ionosphere_maps"} & set(values)
    values = dataclasses.asdict(_cfg(spacewx_ionosphere_opt=1, spacewx_f107_source="140"))
    restart._drop_default_off_run_keys(values)
    assert values["spacewx_ionosphere_opt"] == 1 and values["spacewx_f107_source"] == "140"
    assert "spacewx_ionosphere" in restart.DRIVER_REBUILT_ATTRS


# --- the history inventory ----------------------------------------------------------

def test_produced_history_shapes_carry_the_rows_only_when_on():
    from woof.io.history_layout import produced_history_shapes
    from woof.io.wrf_output_schema import REGISTRY_VAR_META
    off = produced_history_shapes(_cfg())
    assert not set(NAMES) & set(off)
    on = produced_history_shapes(_cfg(spacewx_ionosphere_opt=1, spacewx_f107_source="150"))
    assert {name: on[name] for name in NAMES} == {name: (4, 6) for name in NAMES}
    assert set(on) - set(off) == set(NAMES)
    assert tuple(hist.HISTORY_FIELDS) == NAMES
    for name, (_key, desc, units) in hist.HISTORY_FIELDS.items():
        assert REGISTRY_VAR_META[name] == (desc, units)
    storm_on = produced_history_shapes(_cfg(spacewx_ionosphere_opt=2))
    assert set(storm_on) - set(off) == set(NAMES) | {"IONO_STORM"}
    for name, (desc, units) in hist.STORM_HISTORY_FIELDS.items():
        assert REGISTRY_VAR_META[name] == (desc, units)


def test_preflight_prices_the_held_arrays_and_the_kernel():
    from woof.core import preflight
    cfg = _cfg(spacewx_ionosphere_opt=1, spacewx_f107_source="150")
    shapes = hist.device_shapes(cfg)
    assert shapes["spacewx_ionosphere/history"] == (5, 4, 6)
    storm_shapes = hist.device_shapes(_cfg(spacewx_ionosphere_opt=2))
    assert storm_shapes["spacewx_ionosphere/history"] == (6, 4, 6)
    assert storm_shapes["spacewx_ionosphere/fo_scale"] == (24,)
    assert sum(1 for name in shapes if shapes[name] == (24,)) == 4 + len(ion.OUTPUTS)
    assert "spacewx_ionosphere" in preflight.KERNEL_MAX_LOCAL_SIZE_BYTES
    src = inspect.getsource(preflight)
    assert 'modules.add("spacewx_ionosphere")' in src
    assert "items += _items(\"physics\", spacewx_shapes(run))" in src


def test_streamed_domains_refuse_by_name():
    from woof.core import streaming
    src = inspect.getsource(streaming)
    assert "spacewx_ionosphere_opt = {int(cfg.spacewx_ionosphere_opt)}, whose history rows are computed on the" in src


def test_hex_refuses_the_option_and_admits_off():
    from woof.hex.physics_suites import PhysicsSelectionRefusal, resolve_physics
    with pytest.raises(PhysicsSelectionRefusal, match="spacewx_ionosphere_opt"):
        resolve_physics("arwen_mpas_column", {"spacewx_ionosphere_opt": 1}, mesh="x1.40962")
    resolve_physics("arwen_mpas_column", {"spacewx_ionosphere_opt": 0}, mesh="x1.40962")


def test_the_default_driver_path_imports_nothing_new():
    from woof.core import physics
    src = inspect.getsource(physics.PhysicsDriver)
    assert 'if int(getattr(cfg, "spacewx_ionosphere_opt", 0) or 0) and radiation is not None:' in src
    assert "import woof.spacewx" not in inspect.getsource(physics).split("class PhysicsDriver")[0]


# --- the rows -----------------------------------------------------------------------

def _history(cfg, **kw):
    kw.setdefault("igrf_model", _dipole())
    kw.setdefault("map_loader", _loader())
    return hist.IonosphereHistory.for_run(cfg, _radiation(), **kw)


def test_for_run_holds_the_start_flux_from_the_index_file(tmp_path):
    from woof.spacewx import indices
    path = _gfz(tmp_path, f107=163.0)
    cfg = _cfg(spacewx_ionosphere_opt=1, spacewx_f107_source=str(path), spacewx_f107_mode="daily")
    h = _history(cfg)
    assert h.f107 == pytest.approx(163.0) and h.shape == (4, 6)
    assert h.receipt()["spacewx_f107_sha256"] == indices.read(path).sha256
    assert h.receipt()["spacewx_f107_sfu"] == pytest.approx(163.0)
    assert hist.IonosphereHistory.for_run(_cfg(), _radiation()) is None
    with pytest.raises(hist.IonosphereHistoryError, match="radiation"):
        hist.IonosphereHistory.for_run(cfg, SimpleNamespace(start_time=START))


def test_gfz_coverage_is_asked_only_through_the_day_before_the_start(tmp_path):
    from woof.spacewx import indices
    path = _gfz(tmp_path, f107=150.0)
    asked = []

    def loader(source, needed):
        asked.append((source, needed))
        return indices.read(path)

    h = _history(_cfg(spacewx_ionosphere_opt=1, spacewx_f107_source="gfz"), indices_loader=loader)
    assert asked == [("gfz", START - timedelta(days=1))]
    assert h.f107 == pytest.approx(150.0)


def test_for_run_loads_every_month_the_run_blends_up_front():
    seen = []
    inner = _loader()

    def loader(asset):
        seen.append(int(asset.name[4:6]) - 10)
        return inner(asset)

    cfg = _cfg(spacewx_ionosphere_opt=1, spacewx_f107_source="150", run_seconds=10 * 86400.0)
    h = _history(cfg, map_loader=loader)
    assert sorted(set(seen)) == [4, 5, 6]       # 10-20 May blends April, May and June
    assert sorted(h.ionosphere._maps) == [4, 5, 6]


def test_an_unreleased_flux_is_fetched_again_then_refused_by_name(tmp_path):
    from woof.spacewx import indices
    path = _gfz(tmp_path, f107=150.0)
    text = path.read_text().replace(" 150.0 150.0 2", " -1.0 -1.0 2")
    stale = tmp_path / "stale.txt"
    stale.write_text(text)
    asked = []

    def loader(source, needed):
        asked.append(needed)
        return indices.read(stale)

    cfg = _cfg(spacewx_ionosphere_opt=1, spacewx_f107_source="gfz", spacewx_f107_mode="daily")
    with pytest.raises(hist.IonosphereHistoryError, match="has not released the flux"):
        _history(cfg, indices_loader=loader)
    assert asked == [START - timedelta(days=1), START]
    with pytest.raises(hist.IonosphereHistoryError, match="observed F10.7"):
        _history(_cfg(spacewx_ionosphere_opt=1, spacewx_f107_source=str(stale),
                      spacewx_f107_mode="daily"))


@pytest.mark.gpu
@requires_gpu
def test_rows_are_the_kernel_at_each_valid_time_and_are_held_between():
    import cupy as cp
    cfg = _cfg(spacewx_ionosphere_opt=1, spacewx_f107_source="150")
    h = _history(cfg)
    first = h.history_fields(0.0)
    assert tuple(first) == NAMES
    assert all(a.dtype == cp.float32 and a.shape == (4, 6) for a in first.values())
    want = h.ionosphere.evaluate(START, 150.0)
    for name, (key, _d, _u) in hist.HISTORY_FIELDS.items():
        assert np.array_equal(cp.asnumpy(first[name]), cp.asnumpy(want[key]).reshape(4, 6)), name
    ref = h.ionosphere.reference(START, 150.0)
    np.testing.assert_allclose(cp.asnumpy(first["IONO_TEC"]).ravel(), ref["tec"], rtol=1e-3)
    held = {k: cp.asnumpy(v) for k, v in first.items()}
    again = h.history_fields(0.0)
    assert all(again[k] is first[k] for k in NAMES)
    later = h.history_fields(6 * 3600.0)
    assert all(later[k] is first[k] for k in NAMES)            # one stable allocation
    assert not np.array_equal(cp.asnumpy(later["IONO_TEC"]), held["IONO_TEC"])
    want = h.ionosphere.evaluate(datetime(2024, 5, 11, 0, 0), 150.0)
    assert np.array_equal(cp.asnumpy(later["IONO_FOF2"]), cp.asnumpy(want["fof2"]).reshape(4, 6))


def _driver(**overrides):
    from test_physics_driver import _base_config
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics

    cfg = _base_config(ra_physics=0, ra_lw_physics=4, ra_sw_physics=4,
                       sf_surface_physics=2, sf_sfclay_physics=91, bl_pbl_physics=1,
                       run_seconds=600.0, **overrides)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 298.0 + 0.004 * np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base,
                                lambda z: 0.010 * np.exp(-np.asarray(z, np.float64) / 2400.0))
    lat = np.linspace(30.0, 34.0, cfg.ny)[:, None] * np.ones((1, cfg.nx))
    lon = np.linspace(-90.0, -86.0, cfg.nx)[None, :] * np.ones((cfg.ny, 1))
    driver = initialize_physics(state, cfg, radiation_start_time=START,
                                radiation_latitude=lat, radiation_longitude=lon)
    return driver, state, cfg


@pytest.mark.gpu
@requires_gpu
def test_driver_publishes_the_rows_and_leaves_the_atmosphere_alone(monkeypatch):
    import cupy as cp
    from woof.core.physics import _prepare_atmosphere

    read_maps = ion.read_maps
    monkeypatch.setattr(ion, "read_maps", lambda month, family="ccir", **kw:
                        read_maps(month, family, loader=_loader()))
    monkeypatch.setattr(igrf, "load", lambda path=None: _dipole())
    off, state, cfg = _driver()
    on, state_on, cfg_on = _driver(spacewx_ionosphere_opt=1, spacewx_f107_source="150")
    assert off.spacewx_ionosphere is None and on.spacewx_ionosphere is not None
    assert not set(NAMES) & set(off.output_fields())
    for driver, st, c in ((off, state, cfg), (on, state_on, cfg_on)):
        driver._run_radiation(_prepare_atmosphere(st), st, c)
    fields = on.output_fields()
    lat = cp.asnumpy(on.radiation_callable.latitude_deg).astype(np.float64)
    lon = cp.asnumpy(on.radiation_callable.longitude_deg).astype(np.float64)
    want = ion.Ionosphere(lat, lon, when=START, igrf_model=_dipole(),
                          map_loader=_loader()).evaluate(START, 150.0)
    for name, (key, _d, _u) in hist.HISTORY_FIELDS.items():
        assert np.array_equal(cp.asnumpy(fields[name]), cp.asnumpy(want[key]).reshape(lat.shape)), name
    assert float(cp.asnumpy(fields["IONO_TEC"]).min()) > 0.0
    for name in ("rthratensw", "rthratenlw"):
        assert np.array_equal(cp.asnumpy(getattr(on, name)), cp.asnumpy(getattr(off, name)))
    for name in ("swdown", "glw"):
        assert np.array_equal(cp.asnumpy(on.fields[name]), cp.asnumpy(off.fields[name]))


# --- the storm-time correction (spacewx_ionosphere_opt = 2) ---------------------------

def _storm_gfz(tmp_path, storm_ap=300.0):
    """_gfz with a storm: ap = ``storm_ap`` in 2024-05-10's 00-18 UT intervals."""
    path = _gfz(tmp_path)
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("2024 05 10 "):
            w = line.split()
            w[15:21] = [f"{storm_ap:.0f}"] * 6               # ap of 00, 03, ..., 15 UT
            lines[i] = " ".join(w)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


@pytest.fixture
def storm_fit(tmp_path):
    import json
    from woof.spacewx import storm
    from test_spacewx_storm import _fit_doc
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(_fit_doc()))
    return storm.load_fit(path)


def test_persisted_ap_tau_is_the_clis_at_the_start_and_holds_the_latest_ap_after():
    from woof.spacewx import storm
    from test_spacewx_storm import _record
    ap = np.full(10 * 8, 7.0)
    ap[9 * 8 + 4:9 * 8 + 6] = 300.0                       # 2024-05-10 12-18 UT
    rec = _record(ap)
    tau, nlag = 0.9, 24
    series = storm.ap_series(rec, START, nlag)
    assert storm.persisted_ap_tau(series, START, START, tau) == storm.filtered_ap(rec, START, tau, nlag)
    assert storm.persisted_ap_tau(series, START, START + timedelta(hours=2, minutes=59), tau) == (
        storm.filtered_ap(rec, START, tau, nlag))       # no interval has ended since the start
    # Six hours on: two more intervals ended, each persisting the latest ap (300).
    later = START + timedelta(hours=6)
    future = ap.copy()
    future[9 * 8 + 6:9 * 8 + 8] = 300.0
    assert storm.persisted_ap_tau(series, START, later, tau) == pytest.approx(
        storm.filtered_ap(_record(future), later, tau, nlag), rel=1e-12)
    # Far on, the filter has forgotten the record: ap_tau is the persisted ap.
    far = storm.persisted_ap_tau(series, START, START + timedelta(days=20), tau)
    assert far == pytest.approx(300.0)
    with pytest.raises(storm.StormError, match="before the start"):
        storm.persisted_ap_tau(series, START, START - timedelta(hours=3), tau)
    gap = series.copy()
    gap[0] = np.nan
    with pytest.raises(storm.StormError, match="no ap yet"):
        storm.persisted_ap_tau(gap, START, START + timedelta(hours=9), tau)


def test_storm_for_run_reads_ap_through_the_latest_ended_interval(tmp_path, storm_fit):
    from woof.spacewx import indices, storm
    path = _storm_gfz(tmp_path)
    asked = []

    def loader(source, needed, kp=False):
        asked.append((needed, kp))
        return indices.read(path)

    cfg = _cfg(spacewx_ionosphere_opt=2, spacewx_f107_source=str(path))
    h = _history(cfg, indices_loader=loader, storm_fit=storm_fit)
    assert (START - timedelta(hours=3), True) in asked            # the 15-18 UT interval, with ap
    want = storm.filtered_ap(indices.read(path), START, storm_fit.tau, storm_fit.nlag)
    receipt = h.receipt()
    assert receipt["spacewx_storm_ap_tau_start"] == pytest.approx(want) and want > 50.0
    assert receipt["spacewx_storm_ap_latest"] == 300.0
    assert receipt["spacewx_storm_ap_interval"] == "2024-05-10T15:00Z"
    assert receipt["spacewx_storm_fit_sha256"] == storm_fit.sha256
    assert receipt["spacewx_storm_ap_sha256"] == indices.read(path).sha256
    assert h.names == NAMES + ("IONO_STORM",)
    ap_tau, scale = h.storm_scale(START)
    assert ap_tau == pytest.approx(want)
    np.testing.assert_array_equal(scale, storm.fof2_scale(
        storm_fit, *(a.astype(np.float64) for a in (_radiation().latitude_deg, _radiation().longitude_deg)),
        START, want, model=_dipole()))
    later_tau, _ = h.storm_scale(START + timedelta(hours=12))
    assert later_tau > ap_tau                                      # the persisted 300 keeps raising it
    # Option 1 on the same file reads no ap and has no storm state.
    plain = _history(_cfg(spacewx_ionosphere_opt=1, spacewx_f107_source=str(path)))
    assert plain.storm is None and plain.names == NAMES
    assert not any(k.startswith("spacewx_storm") for k in plain.receipt())


def test_storm_refusals_name_the_breakage(tmp_path, storm_fit):
    path = _gfz(tmp_path)
    late = tmp_path / "late.txt"
    late.write_text("\n".join(line for line in path.read_text().splitlines()
                              if not line.startswith(("2024 05 10", "2024 05 11", "2024 05 12"))) + "\n")
    with pytest.raises(hist.IonosphereHistoryError, match="no ap yet"):
        _history(_cfg(spacewx_ionosphere_opt=2, spacewx_f107_source=str(late), spacewx_f107_mode="81day"),
                 storm_fit=storm_fit)
    with pytest.raises(hist.IonosphereHistoryError, match="not a number"):
        _history(_cfg(spacewx_ionosphere_opt=2, spacewx_f107_source="150"), storm_fit=storm_fit)


def test_the_held_ap_is_reused_for_the_start(tmp_path):
    from woof.spacewx import indices
    path = _storm_gfz(tmp_path)
    memo = tmp_path / "memo"
    calls = []

    def loader(source, needed, kp=False):
        calls.append(source)
        return indices.read(path)

    ap, record = hist._held_ap(loader, "gfz", START, 40, memo)
    again, held = hist._held_ap(lambda *a, **k: 1 / 0, "gfz", START, 40, memo)
    assert calls == ["gfz"] and np.array_equal(ap, again, equal_nan=True)
    assert held["spacewx_storm_ap_sha256"] == record["spacewx_storm_ap_sha256"]
    assert "spacewx_storm_ap_held_from" in held
    hist._held_ap(loader, "gfz", START + timedelta(hours=3), 40, memo)       # another start: read again
    assert calls == ["gfz", "gfz"]
    for p in memo.iterdir():
        p.write_text("{")                                                  # damaged: read again
    hist._held_ap(loader, "gfz", START, 40, memo)
    assert len(calls) == 3


def test_an_incomplete_or_offline_ap_series_is_not_held(tmp_path):
    from woof.spacewx import indices
    good = _storm_gfz(tmp_path)
    late = tmp_path / "late.txt"
    late.write_text("\n".join(line for line in good.read_text().splitlines()
                              if not line.startswith(("2024 05 10", "2024 05 11", "2024 05 12"))) + "\n")
    memo = tmp_path / "memo"
    ap, record = hist._held_ap(lambda s, n, kp=False: indices.read(late), "gfz", START, 40, memo)
    assert np.isnan(ap[0]) and not memo.exists()                       # GFZ had not posted it yet
    ap, _ = hist._held_ap(lambda s, n, kp=False: indices.read(good), "gfz", START, 40, memo)
    assert ap[0] == 300.0 and len(list(memo.iterdir())) == 1          # read again once posted

    def offline(source, needed, kp=False):
        if needed is not None:
            raise OSError("network down")
        return indices.read(good)

    other = START + timedelta(hours=3)
    _ap, record = hist._held_ap(offline, "gfz", other, 40, memo)
    assert record["spacewx_storm_ap_offline_cache"] is True
    assert len(list(memo.iterdir())) == 1                               # the offline read is not held


def test_the_blocked_basis_gives_the_unblocked_ratio(monkeypatch):
    from woof.spacewx import storm
    fit = storm.load_fit()
    rng = np.random.default_rng(3)
    mlat, lt, season = rng.uniform(-80, 80, 1000), rng.uniform(0, 24, 1000), rng.uniform(-1, 1, 1000)
    whole = storm.log_ratio(fit, mlat, lt, season, 55.0)
    monkeypatch.setattr(storm, "BASIS_BLOCK", 97)
    np.testing.assert_array_equal(storm.log_ratio(fit, mlat, lt, season, 55.0), whole)


def test_the_shipped_storm_fit_has_the_published_seasonal_pattern():
    """Physical sanity of the shipped fit (woof/data/spacewx/storm_fit_v1.json)
    at a strong storm: the F2 negative phase is strongest in local summer at
    middle and high magnetic latitudes and weakest in winter, where low
    latitudes show a positive phase (Prolss 1995, Handbook of Atmospheric
    Electrodynamics II, ch. 8; Fuller-Rowell et al. 1994, JGR 99, 3893;
    the STORM model, Araujo-Pradere et al. 2002, Space Weather/Radio Sci.
    37, 1071, has summer mid-latitude NmF2 ratios near 0.5 and below)."""
    from woof.spacewx import storm
    fit = storm.load_fit()
    lt = np.arange(0.0, 24.0, 2.0)

    def mean_r(mlat, season):
        return float(np.mean(storm.nmf2_factor(fit, np.full(lt.size, mlat), lt, np.full(lt.size, season), 150.0)))

    mid = (35.0, 50.0, 62.5)
    summer = np.mean([mean_r(m, 1.0) for m in mid])
    equinox = np.mean([mean_r(m, 0.0) for m in mid])
    winter = np.mean([mean_r(m, -1.0) for m in mid])
    assert summer < equinox < winter
    assert 0.15 <= summer <= 0.7
    assert max(storm.nmf2_factor(fit, np.full(lt.size, 10.0), lt, np.full(lt.size, -1.0), 150.0)) > 1.2
    assert storm.nmf2_factor(fit, [50.0], [12.0], [1.0], 0.0)[0] == pytest.approx(1.0)


@pytest.mark.gpu
@requires_gpu
def test_storm_rows_are_the_scaled_kernel_and_r_at_each_valid_time(tmp_path, storm_fit):
    import cupy as cp
    from woof.spacewx import indices
    path = _storm_gfz(tmp_path)
    cfg = _cfg(spacewx_ionosphere_opt=2, spacewx_f107_source=str(path))
    h = _history(cfg, indices_loader=lambda s, n, kp=False: indices.read(path), storm_fit=storm_fit)
    plain = _history(_cfg(spacewx_ionosphere_opt=1, spacewx_f107_source=str(path)))
    for elapsed in (0.0, 6 * 3600.0):
        when = START + timedelta(seconds=elapsed)
        rows = h.history_fields(elapsed)
        assert tuple(rows) == NAMES + ("IONO_STORM",)
        _ap_tau, scale = h.storm_scale(when)
        want = h.ionosphere.evaluate(when, h.f107, fo_scale=scale)
        for name, (key, _d, _u) in hist.HISTORY_FIELDS.items():
            assert np.array_equal(cp.asnumpy(rows[name]), cp.asnumpy(want[key]).reshape(4, 6)), name
        r = cp.asnumpy(rows["IONO_STORM"])
        np.testing.assert_array_equal(r, (scale * scale).astype(np.float32).reshape(4, 6))
        assert np.ptp(r) > 0.05
        base = plain.history_fields(elapsed)
        np.testing.assert_allclose(cp.asnumpy(rows["IONO_FOF2"]), cp.asnumpy(base["IONO_FOF2"]) * np.sqrt(r),
                                   rtol=1e-5)
