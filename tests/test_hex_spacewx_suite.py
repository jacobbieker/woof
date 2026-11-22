"""spacewx_ionosphere_opt / spacewx_aurora_opt on hex's MPAS v8.4.2 suite route (woof.hex.mpasdiag.spacewx_diag_v842).

* ``--physics`` takes the WRF keys: the route moves only them to its own rows (recorded in the run log and the
  driver receipt) and keeps refusing every other WRF key by name; the keys are judged as the WRF route judges them.
* The Registry, the default streams and the receipt carry the fields only with the keys (the off state is pinned by
  the digests test_mpas_lightning_diagnostics pins).
* On a variable-resolution set of cells (spacing 3 km to 60 km), every published word is the WRF route's row on the
  same cell centres: woof.spacewx.ionosphere.Ionosphere and woof.spacewx.aurora_history.AuroraHistory.  A checkpoint
  continuation (local steps plus the straight run's offset) writes the straight run's words.

The CCIR maps are tests/test_spacewx_ionosphere.py's synthetic ones and the field its dipole, so no fetch runs; F10.7
and Kp are fixed values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasdiag import spacewx_diag_v842 as sw
from woof.hex.mpasphys import resolve_namelist

SUITE = "convection_permitting"
START = datetime(2026, 3, 20, 0, 0, 0)
DT = 30.0
ION = ["spacewx_ionosphere_opt=1", "spacewx_f107_source=150"]
AURORA = ["spacewx_aurora_opt=1", "spacewx_kp_source='5'"]


def _route(physics):
    from woof.hex.mpasphys.forecast_v842 import native_route
    request = argparse.Namespace(physics_namelist=None, physics_suite=SUITE, physics=list(physics) or None,
                                 physics_mode=None, physics_correct_skip=None, physics_default_backend=None)
    return native_route(request)[0]


def test_the_suite_route_takes_the_keys_from_physics():
    config = _route([*ION, *AURORA])
    assert config.spacewx_diagnostics.receipt() == {
        "spacewx_ionosphere_opt": 1, "spacewx_f107_source": "150", "spacewx_f107_mode": "81day",
        "spacewx_ionosphere_maps": "ccir", "spacewx_aurora_opt": 1, "spacewx_kp_source": "5"}
    assert config.receipt()["spacewx_diagnostics"]["spacewx_aurora_opt"] == 1
    assert sw.packages(config) == {"spacewx_ionosphere", "spacewx_aurora"}
    assert not any(key.startswith("spacewx") for key in config.options)
    assert sw.packages(_route(["spacewx_aurora_opt=2", "spacewx_kp_source=omni"])) == {
        "spacewx_aurora", "spacewx_aurora_energy"}
    assert sw.packages(_route(["spacewx_aurora_opt=3", "spacewx_kp_source=omni"])) == {
        "spacewx_aurora", "spacewx_aurora_energy", "spacewx_joule"}
    # The off state: no key, no rows, the receipt unchanged.
    plain = _route([])
    assert plain.spacewx_diagnostics is None and "spacewx_diagnostics" not in plain.receipt()
    assert _route(["spacewx_ionosphere_opt=0"]).spacewx_diagnostics is None
    # Option 2, the storm-time correction, adds its ratio row (IONO_STORM).
    storm = _route(["spacewx_ionosphere_opt=2", "spacewx_f107_source=gfz"])
    assert sw.packages(storm) == {"spacewx_ionosphere", "spacewx_storm"}
    assert sw.field_definition("iono_storm")["gates"] == ["spacewx_storm"]
    assert sw.field_definition("iono_storm")["attrs"]["wrf_name"] == "IONO_STORM"


@pytest.mark.parametrize("physics, message", [
    (["mp_physics=8"], "Registry has no physics option"),
    (["spacewx_epp_opt=1"], "Registry has no physics option"),
    (["spacewx_ionosphere_opt=3"], "must be 0"),
    (["spacewx_ionosphere_opt=2", "spacewx_f107_source=150"], "not a number"),
    (["spacewx_ionosphere_opt=one"], "an integer"),
    (["spacewx_f107_source=150"], "would do nothing"),
    (["spacewx_ionosphere_opt=1", "spacewx_f107_mode=weekly"], "81day"),
    (["spacewx_ionosphere_opt=1", "spacewx_ionosphere_maps=iri"], "ccir"),
    (["spacewx_aurora_opt=2", "spacewx_kp_source=gfz"], "'omni' or 'swpc'"),
    (["spacewx_aurora_opt=3", "spacewx_kp_source=swpc"], "must be 'omni'"),
    (["spacewx_aurora_opt=4"], "must be 0"),
    (["spacewx_kp_source=5"], "would do nothing"),
])
def test_other_wrf_keys_and_bad_values_are_refused_by_name(physics, message):
    with pytest.raises(ValueError, match=message):
        _route(physics)


def _layout(physics=()):
    from woof.hex.mpasphys.framework_backend_v842 import registry_source
    from woof.hex.mpasphys.pool_v842 import RegistryLayout
    text = f"&physics\n config_physics_suite='{SUITE}'\n/\n"
    config = sw.configure(resolve_namelist(text, default_backend="hex-cuda"),
                          dict(item.split("=", 1) for item in physics))
    dims = dict(nCells=10, nEdges=30, nVertices=20, nVertLevels=12, maxEdges=6, maxEdges2=12,
                vertexDegree=3, nSoilLevels=4, nOznLevels=59, nAerLevels=1, nMonths=12)
    return RegistryLayout(registry_source(), config, text, dims)


def test_pools_and_streams_only_with_the_keys(tmp_path):
    from woof.hex.mpasphys.forecast_v842 import read_streams, write_default_streams
    from woof.hex.streams_v842 import _all_packages, field_definition
    off, on = _layout(), _layout(["spacewx_ionosphere_opt=2", "spacewx_f107_source=gfz",
                                  "spacewx_aurora_opt=3", "spacewx_kp_source=omni"])
    rows = lambda layout: {r["name"]: r for r in layout.fields if r["pool"] == "diag"}
    assert not set(sw.NAMES) & set(rows(off))
    assert set(sw.NAMES) <= set(rows(on))
    assert [r for r in on.fields if r["name"] not in sw.NAMES] == list(off.fields)
    assert set(sw.PACKAGES) <= set(on.packages) and not set(sw.PACKAGES) & set(off.packages)
    streams = {}
    for label, layout in (("off", off), ("on", on)):
        write_default_streams(layout, tmp_path / f"streams.{label}")
        streams[label] = {s.name: s for s in read_streams(tmp_path / f"streams.{label}", layout)}
    assert set(sw.NAMES) <= set(streams["on"]["diagnostics"].fields)
    assert [f for f in streams["on"]["diagnostics"].fields if f not in sw.NAMES] == \
        list(streams["off"]["diagnostics"].fields)
    tec = field_definition("iono_tec")
    assert tec["gates"] == ["spacewx_ionosphere"] and tec["attrs"]["units"] == "TECU"
    assert tec["attrs"]["wrf_name"] == "IONO_TEC"
    assert field_definition("aurora_eflux")["gates"] == ["spacewx_aurora_energy"]
    assert field_definition("joule_power")["attrs"]["units"] == "GW"
    assert set(sw.PACKAGES) <= _all_packages()
    # Only the ionosphere: no aurora rows, and option 1 has no storm row.
    assert not {n for n in sw.NAMES if not n.startswith("iono")} & set(rows(_layout(ION)))
    assert "iono_storm" not in rows(_layout(ION))


def test_the_off_state_registry_streams_and_receipt_are_unchanged():
    """test_mpas_lightning_diagnostics's off-state digests."""
    layout = _layout()
    digest = hashlib.sha256(json.dumps([layout.fields, sorted(layout.packages), sorted(layout.dimensions.items())],
                                       sort_keys=True, default=str).encode()).hexdigest()[:16]
    assert digest == "91afd1d2f4968e3f"
    with tempfile.TemporaryDirectory() as directory:
        from woof.hex.mpasphys.forecast_v842 import write_default_streams
        write_default_streams(layout, Path(directory) / "streams")
        assert hashlib.sha256((Path(directory) / "streams").read_bytes()).hexdigest()[:16] == "b868bf60f2a107f7"
    config = resolve_namelist(f"&physics\n config_physics_suite='{SUITE}'\n/\n", default_backend="hex-cuda")
    assert hashlib.sha256(json.dumps(config.receipt(), sort_keys=True, default=str).encode()).hexdigest()[:16] == \
        "bb186b77c0898a2c"


# ---------------------------------------------------------------------------
# the published words, on a variable-resolution set of cells
# ---------------------------------------------------------------------------
NC = 24


def _cells():
    """Cell centres (radians, float32) packed tightly near 62N 20W and spreading out: neighbour spacing from ~3 km
    to ~60 km, as on a variable-resolution mesh; the set straddles the auroral oval."""
    step = np.geomspace(0.027, 0.54, NC)                     # degrees between neighbours
    lat = 62.0 + np.cumsum(step) * np.where(np.arange(NC) % 2, 1.0, -0.4)
    lon = -20.0 + np.cumsum(step[::-1]) * 0.8
    return np.deg2rad(lat).astype(np.float32), np.deg2rad(lon).astype(np.float32)


@pytest.fixture
def cp(monkeypatch):
    cp = pytest.importorskip("cupy")
    from test_spacewx_ionosphere import _dipole, _loader
    from woof.spacewx import igrf
    from woof.spacewx import ionosphere as ion
    read_maps = ion.read_maps
    monkeypatch.setattr(ion, "read_maps", lambda month, family="ccir", **kw: read_maps(month, family,
                                                                                       loader=_loader()))
    monkeypatch.setattr(igrf, "load", lambda path=None: _dipole())
    return cp


class _Utils:
    def __init__(self, written):
        self.written = set(written)

    def stream_inclusion_count(self, name, direction=None, **kw):
        return 1

    def field_will_be_written(self, name):
        return name in self.written


def _owner(cp, step_offset=0, hours=12, segment_start=START):
    lat, lon = _cells()
    pad = lambda a: cp.asarray(np.concatenate([a, a[:1]]))            # one halo word past nCellsSolve
    fields = {("mesh", "latCell"): pad(lat), ("mesh", "lonCell"): pad(lon)}
    buffers = {}

    def buffer(pool, name):
        return buffers.setdefault((pool, name), cp.full(NC + 1, -9.0, dtype=cp.float32))

    return SimpleNamespace(cp=cp, session=SimpleNamespace(layout=SimpleNamespace(dimensions={"nCellsSolve": NC})),
                           authority=lambda pool, name: fields[pool, name], buffer=buffer, buffers=buffers,
                           dycore=SimpleNamespace(config=SimpleNamespace(config_dt=DT)), step_offset=step_offset,
                           writer=SimpleNamespace(start=segment_start, final=START + timedelta(hours=hours)),
                           halo=None)


def _module(cp, physics, written, step_offset=0, segment_start=START):
    owner = _owner(cp, step_offset, segment_start=segment_start)
    module = sw.SpacewxModule(owner, sw.resolve(dict(item.split("=", 1) for item in physics)))
    module.setup(_Utils(written))
    return owner, module


def _publish(owner, module, step, when):
    module.begin(step, when)
    module.compute()
    return {name: owner.cp.asnumpy(value) for (_pool, name), value in owner.buffers.items()}


@pytest.mark.gpu
def test_every_word_is_the_wrf_routes_ionosphere_row(cp, capsys):
    from test_spacewx_ionosphere import _dipole, _loader
    from woof.spacewx import history as hist
    from woof.spacewx.ionosphere import Ionosphere
    owner, module = _module(cp, ION, written={"iono_tec", "iono_fof2", "iono_hmf2"})
    lat, lon = (np.degrees(v.astype(np.float64)).reshape(1, -1) for v in _cells())
    want_model = Ionosphere(lat, lon, when=START, igrf_model=_dipole(), map_loader=_loader())
    frames = {}
    for step in (0, 240, 480):                                  # 0 h, 2 h, 4 h
        when = START + timedelta(seconds=step * DT)
        got = _publish(owner, module, step, when.strftime("%Y-%m-%d_%H:%M:%S") if step == 240 else when)
        out = want_model.evaluate(when, 150.0)
        for wrf, (key, _text, _units) in hist.HISTORY_FIELDS.items():
            name = wrf.lower()
            if name not in module.utils.written:
                assert ("diag", name) not in owner.buffers
                continue
            np.testing.assert_array_equal(got[name][:NC], cp.asnumpy(out[key]).reshape(-1).astype(np.float32), name)
            assert got[name][NC] == -9.0                       # the halo word is not a solve cell
        frames[step] = got["iono_tec"][:NC].copy()
    assert module.start == START and frames[0].min() > 0.0
    log = capsys.readouterr().out
    assert log.count("[spacewx suite]") == 1 and '"spacewx_f107_sfu": 150.0' in log and "2026-03-20T00:00:00" in log
    assert not np.array_equal(frames[0], frames[240])


@pytest.mark.gpu
@pytest.mark.parametrize("opt, source", [(1, "5"), (1, "7.3")])
def test_every_word_is_the_wrf_routes_aurora_row(cp, opt, source):
    from woof.spacewx.aurora_history import AuroraHistory, HISTORY_FIELDS
    physics = [f"spacewx_aurora_opt={opt}", f"spacewx_kp_source={source}"]
    owner, module = _module(cp, physics, written={n.lower() for n in HISTORY_FIELDS})
    lat, lon = (np.degrees(v.astype(np.float64)).reshape(1, -1) for v in _cells())
    cfg = SimpleNamespace(**sw.resolve(dict(p.split("=", 1) for p in physics)).settings, run_seconds=0.0)
    route = AuroraHistory.for_run(cfg, SimpleNamespace(start_time=START, latitude_deg=lat, longitude_deg=lon))
    when = START + timedelta(hours=3)
    got = _publish(owner, module, int(3 * 3600 / DT), when)
    want = route.history_fields((when - START).total_seconds())
    for wrf in HISTORY_FIELDS:
        np.testing.assert_array_equal(got[wrf.lower()][:NC], cp.asnumpy(want[wrf]).reshape(-1), wrf)
    zone = got["aurora_zone"][:NC]
    assert zone.min() >= 0 and zone.max() <= 3 and len(set(zone.tolist())) > 1


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["correct", "exact"])
def test_a_continuation_writes_the_straight_runs_words(cp, mode):
    """Correct mode hands the continuation the straight run's step offset; exact mode does not, and the rows' start
    rides the checkpoint (state()/resume())."""
    physics = [*ION, *AURORA]
    written = set(sw.NAMES) - {"iono_storm", "aurora_eflux", "aurora_emean", "aurora_iflux", "joule_power"}
    when = START + timedelta(hours=5)
    straight_owner, straight = _module(cp, physics, written)
    want = _publish(straight_owner, straight, int(5 * 3600 / DT), when)
    offset = int(2 * 3600 / DT)                              # the continuation started at +2 h
    segment = START + timedelta(hours=2)
    owner, module = _module(cp, physics, written, step_offset=offset if mode == "correct" else 0,
                            segment_start=segment)
    assert module.start == (START if mode == "correct" else segment)
    module.resume(straight.state())
    got = _publish(owner, module, int(5 * 3600 / DT) - offset, when)
    assert module.start == straight.start == START
    assert set(got) == set(written) == set(want)
    for name in written:
        np.testing.assert_array_equal(got[name], want[name], name)


def test_the_active_packages_fields_are_produced_before_the_writer_is_built():
    """streams_output_v842.PhysicsStreams checks the enabled streams' fields against ``produced`` before it binds
    the diagnostics (test_output_diagnostics_produced)."""
    from woof.hex.mpasphys import resident_output_diagnostics_v842 as rod

    def produced(packages):
        session = SimpleNamespace(layout=SimpleNamespace(packages=set(packages), fields=[]),
                                  device_pools=SimpleNamespace(cp=None))
        return rod.ResidentOutputDiagnostics(session, dycore=None).produced

    assert produced(()) == rod.PRODUCED
    ion = produced({"spacewx_ionosphere"})
    assert {n for n in sw.NAMES if n.startswith("iono_")} - {"iono_storm"} <= ion
    assert not {"iono_storm", "aurora_zone", "joule_power"} & ion
    assert "iono_storm" in produced({"spacewx_ionosphere", "spacewx_storm"})
    assert set(sw.NAMES) <= produced(set(sw.PACKAGES))
