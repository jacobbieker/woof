"""WRF's CLWRF, RASM mean_diag/diurnal_diag and surface_energy_diag on the MPAS suite route
(woof.hex.mpasdiag.reduction_diag_v842).

* the options resolve (off by default), inconsistent ones and the RRFS fork route are refused, and the Registry, default
  streams and driver receipt carry the fields only with the options (the off state is pinned by digest);
* CLWRF: the module's words equal WRF 4.7.1's module_diag_cl.F words (the committed gfortran-built fixture
  tests/data/oracles/clwrf: 41 calls, every output word and counter) on mesh columns, bitwise;
* RASM: finished means equal the sequential float32 sum over the window divided by the count, the three-hour diurnal bins
  take the window means at WRF's clock, PMSL equals the kernel on the same words; surface-energy sums equal
  ``acc + hfx * dt`` in float32 step by step;
* the t = 0 frame is zero, a window that is not whole steps is refused, accumulators ride the checkpoint.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasdiag import reduction_diag_v842 as red
from woof.hex.mpasphys import resolve_namelist

ROOT = Path(__file__).resolve().parents[3]


def _namelist(extra=None):
    lines = ["&physics", " config_physics_suite='convection_permitting'", "/"]
    if extra is not None:
        lines += ["&diagnostics", *[f" {k}={v}" for k, v in extra.items()], "/"]
    return "\n".join(lines) + "\n"


def _resolve(extra=None):
    return resolve_namelist(_namelist(extra), default_backend="hex-cuda")


ALL = {"config_output_diagnostics": ".true.", "config_auxhist3_interval": 60, "config_mean_diag": ".true.",
       "config_mean_freq": 3, "config_mean_interval": 1, "config_diurnal_diag": ".true.",
       "config_surface_energy_diag": ".true."}


def test_options_default_off_and_resolve():
    assert _resolve().reduction_diagnostics is None and "reduction_diagnostics" not in _resolve().receipt()
    config = _resolve(ALL)
    assert config.reduction_diagnostics == red.ReductionConfig(60, True, 3, 1, True, True)
    assert red.packages(config) == set(red.PACKAGES)
    assert config.receipt()["reduction_diagnostics"]["clwrf_interval"] == 60
    assert red.packages(_resolve({"config_surface_energy_diag": ".true."})) == {red.ENERGY_PACKAGE}
    assert red.packages(_resolve({"config_mean_diag": ".true.", "config_mean_freq": 5})) == {red.MEAN_PACKAGE}
    for key in red.OPTIONS:
        assert key not in config.options
    mixed = _resolve({"config_nwp_diagnostics": ".true.", "config_diurnal_diag": ".true."})
    assert mixed.nwp_diagnostics and mixed.reduction_diagnostics.diurnal_diag


@pytest.mark.parametrize("extra, message", [
    ({"config_output_diagnostics": ".true."}, "needs config_auxhist3_interval"),
    ({"config_auxhist3_interval": 30}, "read by nothing"),
    ({"config_mean_diag": ".true."}, "config_mean_freq 1..5"),
    ({"config_mean_diag": ".true.", "config_mean_freq": 3}, "positive"),
    ({"config_mean_freq": 3}, "read"),
    ({"config_diurnal_diag": 1}, "logical"),
    ({"config_auxhist3_interval": -5}, "integer >= 0"),
])
def test_inconsistent_options_are_refused(extra, message):
    with pytest.raises(ValueError, match=message):
        _resolve(extra)


def test_rrfs_fork_route_refuses_the_options():
    from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist as resolve_rrfs
    text = "&physics\n config_physics_suite='hrrrv5'\n/\n&diagnostics\n config_mean_diag=.true.\n/\n"
    with pytest.raises(ValueError, match="not available on the RRFS fork route"):
        resolve_rrfs(text)


def _layout(extra):
    from woof.hex.mpasphys.framework_backend_v842 import registry_source
    from woof.hex.mpasphys.pool_v842 import RegistryLayout
    text = _namelist(extra)
    config = resolve_namelist(text, default_backend="hex-cuda")
    dims = dict(nCells=10, nEdges=30, nVertices=20, nVertLevels=12, maxEdges=6, maxEdges2=12,
                vertexDegree=3, nSoilLevels=4, nOznLevels=59, nAerLevels=1, nMonths=12)
    return RegistryLayout(registry_source(), config, text, dims)


def test_pools_and_streams_only_with_the_options(tmp_path):
    from woof.hex.mpasphys.forecast_v842 import read_streams, write_default_streams
    from woof.hex.streams_v842 import field_definition, _all_packages
    off, on, energy = _layout(None), _layout(ALL), _layout({"config_surface_energy_diag": ".true."})
    rows = lambda layout: {r["name"]: r for r in layout.fields if r["pool"] == "diag"}
    assert not set(red.NAMES) & set(rows(off))
    assert set(red.NAMES) <= set(rows(on))
    assert set(red.ENERGY_NAMES) <= set(rows(energy)) and not set(red.CLWRF_NAMES) & set(rows(energy))
    assert [r for r in on.fields if r["name"] not in red.NAMES] == list(off.fields)
    assert set(red.PACKAGES) <= on.packages and not set(red.PACKAGES) & off.packages
    assert rows(on)["rasm_t2_diurn"]["dimensions"] == ["nDiurnalBins", "nCells"] and rows(on)["rasm_t2_diurn"]["shape"] == [8, 11]
    assert rows(on)["clwrf_t2min"]["shape"] == [11]
    streams = {}
    for label, layout in (("off", off), ("on", on)):
        write_default_streams(layout, tmp_path / f"streams.{label}")
        streams[label] = {s.name: s for s in read_streams(tmp_path / f"streams.{label}", layout)}
    assert set(red.NAMES) <= set(streams["on"]["diagnostics"].fields) and "gsw" in streams["on"]["diagnostics"].fields
    assert not set(red.NAMES) & set(streams["off"]["diagnostics"].fields) and "gsw" not in streams["off"]["diagnostics"].fields
    assert {n: s.fields for n, s in streams["on"].items() if n != "diagnostics"} == \
           {n: s.fields for n, s in streams["off"].items() if n != "diagnostics"}
    assert [f for f in streams["on"]["diagnostics"].fields if f not in (*red.NAMES, "gsw")] == \
           list(streams["off"]["diagnostics"].fields)
    d = field_definition("clwrf_t2min")
    assert d["attrs"]["wrf_name"] == "T2MIN" and d["attrs"]["units"] == "K"
    assert field_definition("achfx")["attrs"]["units"] == "J m-2"
    assert set(red.PACKAGES) <= _all_packages()


def test_off_state_digests_are_unchanged():
    import hashlib
    import json
    import tempfile
    from woof.hex.mpasphys.forecast_v842 import write_default_streams
    layout = _layout(None)
    digest = hashlib.sha256(json.dumps([layout.fields, sorted(layout.packages), sorted(layout.dimensions.items())],
                                       sort_keys=True, default=str).encode()).hexdigest()[:16]
    assert digest == "91afd1d2f4968e3f"
    with tempfile.TemporaryDirectory() as directory:
        write_default_streams(layout, Path(directory) / "streams")
        assert hashlib.sha256((Path(directory) / "streams").read_bytes()).hexdigest()[:16] == "b868bf60f2a107f7"
    assert hashlib.sha256(json.dumps(_resolve().receipt(), sort_keys=True, default=str).encode()).hexdigest()[:16] == "bb186b77c0898a2c"


def test_the_field_rows_are_the_wrf_routes_rows():
    from woof.core.output_diagnostics import HISTORY
    for name, slot in red.CLWRF_SLOT.items():
        assert red.FIELDS[name]["attrs"]["wrf_name"] == HISTORY[slot][0]
    for name in (*red.MEAN_NAMES, *red.DIURNAL_NAMES):
        a = red.FIELDS[name]["attrs"]
        assert (a["wrf_name"], a["units"], a["long_name"]) == HISTORY[name]


# ---------------------------------------------------------------------------
# stand-in owner
# ---------------------------------------------------------------------------
@pytest.fixture
def cp():
    return pytest.importorskip("cupy")


class _Utils:
    def __init__(self, written=()):
        self.written = set(written)

    def stream_inclusion_count(self, name, direction=None, **kw):
        return 1

    def field_will_be_written(self, name):
        return name in self.written


SURF = ("t2m", "th2m", "q2", "u10", "v10", "skintemp", "hfx", "lh", "grdflx", "swdnb", "glw", "lwupb", "swupb", "swupt",
        "swdnt", "lwupt", "lwdnt", "raincv", "rainncv")


def _owner(cp, nc, nl=6, start=dt.datetime(2026, 1, 1)):
    fields, buffers = {}, {}
    layout = [dict(pool="diag", pool_name="surface_pressure")] + [dict(pool="diag_physics", pool_name=n) for n in SURF]
    rng = np.random.default_rng(3)
    f = np.float32
    pad = lambda a: cp.asarray(np.concatenate([a, a[..., :1]], axis=-1))
    zgrid = np.cumsum(np.full((nl + 1, nc), 400.0, f), axis=0) - f(400.0) + rng.uniform(0, 500, nc).astype(f)
    fields["mesh", "zgrid"] = pad(zgrid)
    theta = rng.uniform(285, 300, (nl, nc)).astype(f)
    fields["diag", "theta"] = pad(theta)
    fields["diag", "pressure_base"] = pad(np.full((nl, nc), 90000.0, f))
    fields["diag", "pressure_p"] = pad(rng.uniform(-300, 300, (nl, nc)).astype(f))
    fields["state", "scalars"] = cp.asarray(np.stack([pad(rng.uniform(0, 0.01, (nl, nc)).astype(f))]))
    session = SimpleNamespace(layout=SimpleNamespace(dimensions=dict(nCellsSolve=nc, nVertLevels=nl), fields=layout,
                                                     indices={("state", "index_qv"): 1}))

    def buffer(pool, name):
        shape = (8, nc + 1) if name.endswith("_diurn") else (nc + 1,)
        return buffers.setdefault((pool, name), cp.zeros(shape, dtype=cp.float32))

    owner = SimpleNamespace(cp=cp, session=session, authority=lambda pool, name: fields[pool, name], buffer=buffer,
                            fields=fields, buffers=buffers, writer=SimpleNamespace(start=start), start=start, pad=pad)
    return owner


def _feed(owner, nc, rng):
    f = np.float32
    ranges = dict(t2m=(250, 305), th2m=(250, 310), q2=(0, 0.02), u10=(-15, 15), v10=(-15, 15), skintemp=(250, 310),
                  hfx=(-50, 400), lh=(-10, 500), grdflx=(-80, 120), swdnb=(0, 900), glw=(200, 420), lwupb=(300, 500),
                  swupb=(0, 300), swupt=(0, 400), swdnt=(0, 1300), lwupt=(150, 320), lwdnt=(0, 5),
                  raincv=(0, 0.5), rainncv=(0, 1.0))
    values = {k: rng.uniform(*ranges[k], nc).astype(f) for k in SURF}
    values["surface_pressure"] = rng.uniform(85000, 101000, nc).astype(f)
    for k, v in values.items():
        owner.fields["diag" if k == "surface_pressure" else "diag_physics", k] = owner.pad(v)
    return values


def _module(cp, nc, config, dt_s, written=red.NAMES, start=dt.datetime(2026, 1, 1)):
    owner = _owner(cp, nc, start=start)
    module = red.ReductionModule(owner, config)
    utils = _Utils(written)
    module.setup(utils, dt_s)
    return module, owner, utils


def _step(module, owner, step):
    module.begin(step, owner.start + dt.timedelta(seconds=step * module.dt))
    module.update()


def _pub(cp, owner, name, nc, bins=False):
    arr = cp.asnumpy(owner.buffers["diag", name])
    return arr[:8, :nc] if bins else arr[:nc]


def test_clwrf_equals_wrf_module_diag_cl_on_mesh_columns(cp):
    from woof.core.clwrf import HISTORY, INPUTS
    with np.load(ROOT / "tests/data/oracles/clwrf/vectors.npz") as v:
        inputs, clock, expected, counters = v["inputs"], v["clock"], v["expected"], v["counters"]
    nc = 6
    module, owner, utils = _module(cp, nc, red.ReductionConfig(clwrf_interval=1), 20.0, red.CLWRF_NAMES)
    source = {"t2": "t2m", "q2": "q2", "tsk": "skintemp", "u10": "u10", "v10": "v10", "raincv": "raincv", "rainncv": "rainncv"}
    from woof.core.clwrf import HISTORY as _H
    for slot in _H:                       # the fixture starts every word at a sentinel (untouched wind maxima on reset)
        module.words[slot][...] = np.float32(-123.25)
    for index in range(len(clock)):
        dt_s, xtime, seconds, restart, interval = (float(x) for x in clock[index])
        module.dt = dt_s
        module.config = red.ReductionConfig(clwrf_interval=int(interval))
        for name, value in zip(INPUTS, inputs[index]):
            owner.fields["diag_physics", source[name]] = owner.pad(value.reshape(nc))
        module._seconds, module._now = seconds, owner.start
        _xt = np.float32(xtime)
        from woof.core import clwrf
        clwrf.launch({n: module._plane(*red.CLWRF_SOURCE[n]) for n in INPUTS},
                     {s: module.words[s] for s in HISTORY}, module.words["clwrf_nsteps"],
                     dt=dt_s, xtime=_xt, seconds=seconds, interval=int(interval))
        module.compute()
        for k, slot in enumerate(HISTORY):
            name = "clwrf_" + HISTORY[slot][0].lower()
            got = _pub(cp, owner, name, nc).view(np.uint32)
            np.testing.assert_array_equal(got, expected[index][k].reshape(nc), err_msg=f"call {index} {name}")


def test_clwrf_module_update_drives_the_same_kernel(cp):
    """update() itself (not the direct launch above): its clock is the step's start time, xtime = seconds / 60."""
    nc = 5
    module, owner, utils = _module(cp, nc, red.ReductionConfig(clwrf_interval=1), 20.0, red.CLWRF_NAMES)
    rng = np.random.default_rng(5)
    t2 = []
    for step in (1, 2, 3):
        t2.append(_feed(owner, nc, rng)["t2m"])
        _step(module, owner, step)
    module.compute()
    # window = 3 steps (60 s / 20 s): reset at the first step, finish at the third
    celsius = np.stack(t2) - np.float32(273)
    np.testing.assert_allclose(_pub(cp, owner, "clwrf_t2max", nc), celsius.max(0) + 273, atol=1e-3)
    np.testing.assert_allclose(_pub(cp, owner, "clwrf_t2min", nc), celsius.min(0) + 273, atol=1e-3)
    np.testing.assert_allclose(_pub(cp, owner, "clwrf_t2mean", nc), celsius.mean(0) + 273, atol=1e-3)


def test_rasm_mean_and_energy_sums_follow_the_step_sequence(cp):
    nc, dt_s = 7, 600.0
    config = red.ReductionConfig(0, True, 2, 30, False, True)          # 30 min means: 3 steps
    module, owner, utils = _module(cp, nc, config, dt_s)
    rng = np.random.default_rng(9)
    f = np.float32
    window = {k: [] for k in ("t2m", "swdnb", "lwupt")}
    acc = {n: np.zeros(nc, f) for n in red.ENERGY_NAMES}
    for step in (1, 2, 3, 4, 5, 6):
        values = _feed(owner, nc, rng)
        _step(module, owner, step)
        for name, (source, _, _) in red.ENERGY.items():
            acc[name] = (acc[name] + values[source] * f(dt_s)).astype(f)
            np.testing.assert_array_equal(cp.asnumpy(module.words[name]).view(np.uint32), acc[name].view(np.uint32))
        if step in (4, 5, 6):
            for k in window:
                window[k].append(values[k])
        if step == 3:
            module.compute()
            first = {k: _pub(cp, owner, f"rasm_{ {'t2m': 't2', 'swdnb': 'swdnb', 'lwupt': 'lwupt'}[k] }_mean", nc).copy()
                     for k in window}
    module.compute()
    for k, short in (("t2m", "t2"), ("swdnb", "swdnb"), ("lwupt", "lwupt")):
        total = np.zeros(nc, f)
        for v in window[k]:
            total = (total + v).astype(f)
        np.testing.assert_array_equal(_pub(cp, owner, f"rasm_{short}_mean", nc).view(np.uint32),
                                      (total / f(3)).astype(f).view(np.uint32), err_msg=k)
        assert not np.array_equal(first[k], _pub(cp, owner, f"rasm_{short}_mean", nc))
    pmsl = _pub(cp, owner, "rasm_pmsl_mean", nc)
    assert (pmsl > 50000).all() and (pmsl < 120000).all()


def test_rasm_diurnal_bins_take_the_three_hour_window_means(cp):
    nc, dt_s = 4, 3600.0
    module, owner, utils = _module(cp, nc, red.ReductionConfig(0, False, 0, 0, True, False), dt_s)
    rng = np.random.default_rng(13)
    t2 = []
    for step in range(1, 10):
        t2.append(_feed(owner, nc, rng)["t2m"])
        _step(module, owner, step)
    module.compute()
    bins = _pub(cp, owner, "rasm_t2_diurn", nc, bins=True)
    f = np.float32
    for window, cycle in ((slice(0, 3), 0), (slice(3, 6), 1), (slice(6, 9), 2)):
        total = np.zeros(nc, f)
        for v in t2[window]:
            total = (total + v).astype(f)
        np.testing.assert_array_equal(bins[cycle].view(np.uint32), (total / f(3)).astype(f).view(np.uint32),
                                      err_msg=f"bin {cycle}")
    assert not bins[3:].any()


def test_t0_frame_is_zero_windows_must_be_whole_steps_and_a_checkpoint_resumes(cp):
    nc = 4
    module, owner, utils = _module(cp, nc, red.ReductionConfig(30, True, 2, 30, False, True), 600.0)
    _feed(owner, nc, np.random.default_rng(1))
    module.begin(0, owner.start)
    module.update()
    module.compute()
    assert not any(cp.asnumpy(owner.buffers["diag", n]).any() for n in red.NAMES if ("diag", n) in owner.buffers)
    rng = np.random.default_rng(2)
    for step in (1, 2):
        _feed(owner, nc, rng)
        _step(module, owner, step)
    state = module.state()
    fresh, f_owner, _ = _module(cp, nc, red.ReductionConfig(30, True, 2, 30, False, True), 600.0)
    fresh.resume(state)
    for k in module.words:
        np.testing.assert_array_equal(cp.asnumpy(fresh.words[k]), cp.asnumpy(module.words[k]), err_msg=k)
    owner2 = _owner(cp, nc)
    with pytest.raises(ValueError, match="whole number"):
        red.ReductionModule(owner2, red.ReductionConfig(7, False, 0, 0, False, False)).setup(_Utils(red.NAMES), 600.0)
    with pytest.raises(ValueError, match="divides three hours"):
        red.ReductionModule(owner2, red.ReductionConfig(0, False, 0, 0, True, False)).setup(_Utils(red.NAMES), 700.0)


def test_resident_provider_folds_publishes_and_carries_the_module(cp):
    from woof.hex.mpasdiag.convective_v842 import MAXIMA
    from woof.hex.mpasphys.resident_output_diagnostics_v842 import PRODUCED, ResidentOutputDiagnostics as R
    nc = 4
    module, owner, utils = _module(cp, nc, red.ReductionConfig(0, False, 0, 0, False, True), 600.0)
    provider = R.__new__(R)
    provider.cp, provider.reduction, provider.afwa, provider.accdiag, provider.nwp, provider.tslist = cp, module, None, None, None, None
    provider.buffers, provider._event_views, provider._resume, provider.updates, provider.acoustic = {}, None, None, 0, {}
    provider.convective_maxima = MAXIMA
    provider.convective = SimpleNamespace(update=lambda: None, reset=lambda: None)
    assert R.produced is PRODUCED and not set(red.NAMES) & set(PRODUCED)
    _feed(owner, nc, np.random.default_rng(4))
    module.begin(1, owner.start + dt.timedelta(seconds=600))
    provider._update_convective()
    assert cp.asnumpy(module.words["achfx"]).any()
    state = provider.accumulator_state()
    assert {red.STATE_PREFIX + n for n in red.ENERGY_NAMES} <= set(state)
    provider.resume_accumulators(state)
    assert set(provider._resume) == set(state)
