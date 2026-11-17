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
from datetime import datetime
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
    (dict(spacewx_ionosphere_opt=2), "spacewx_ionosphere_opt"),
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


def test_preflight_prices_the_held_arrays_and_the_kernel():
    from woof.core import preflight
    cfg = _cfg(spacewx_ionosphere_opt=1, spacewx_f107_source="150")
    shapes = hist.device_shapes(cfg)
    assert shapes["spacewx_ionosphere/history"] == (5, 4, 6)
    assert sum(1 for name in shapes if shapes[name] == (24,)) == 4 + len(ion.OUTPUTS)
    assert "spacewx_ionosphere" in preflight.KERNEL_MAX_LOCAL_SIZE_BYTES
    src = inspect.getsource(preflight)
    assert 'modules.add("spacewx_ionosphere")' in src
    assert "items += _items(\"physics\", spacewx_shapes(run))" in src


def test_streamed_domains_refuse_by_name():
    from woof.core import streaming
    src = inspect.getsource(streaming)
    assert "spacewx_ionosphere_opt = 1, whose history rows are computed on the" in src


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
