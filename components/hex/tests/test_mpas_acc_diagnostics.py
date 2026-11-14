"""WRF's prec_acc_dt windows and acc_phy_tend accumulators on the MPAS suite route (woof.hex.mpasdiag.acc_diag_v842).

&diagnostics config_prec_acc_dt / config_acc_phy_tend (+ config_acc_phy_tend_interval) add PREC_ACC_C, PREC_ACC_NC,
SNOW_ACC_NC and the sixteen A*TEN tendency accumulators as ``diag`` fields.  Here:

* the options resolve (off by default), are refused where they are inconsistent and on the RRFS fork route, and the
  Registry, default streams and driver receipt carry them only with the options (the off state is pinned by digest);
* every accumulator, after every step, equals WRF 4.7.1's own phys/module_diag_misc.F blocks run on the same
  columns (components/hex/tools/acc_wrf471_oracle: the blocks are cut from the byte-unmodified file and compiled with
  gfortran), word for word, across window resets, the MAX(., 0) clamps and the t2 < 273.15 snow branch;
* the window clock: the t = 0 frame holds zero, the frame at a window's end holds the steps that started in it, a
  window that is not a whole number of steps is refused, and a checkpoint's accumulators resume.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasdiag import acc_diag_v842 as acc
from woof.hex.mpasphys import resolve_namelist

HERE = Path(__file__).resolve().parent
ORACLE = HERE / "acc_oracle"


def _namelist(extra=None):
    lines = ["&physics", " config_physics_suite='convection_permitting'", "/"]
    if extra is not None:
        lines += ["&diagnostics", *[f" {k}={v}" for k, v in extra.items()], "/"]
    return "\n".join(lines) + "\n"


def _resolve(extra=None):
    return resolve_namelist(_namelist(extra), default_backend="hex-cuda")


def test_options_default_off_and_resolve():
    assert _resolve().acc_diagnostics is None and "acc_diagnostics" not in _resolve().receipt()
    assert _resolve({"config_prec_acc_dt": 0.0}).acc_diagnostics is None
    config = _resolve({"config_prec_acc_dt": 30.0})
    assert config.acc_diagnostics == acc.AccConfig(30.0, False, 0.0)
    assert acc.packages(config) == {acc.PREC_PACKAGE}
    assert config.receipt()["acc_diagnostics"] == dict(prec_acc_dt=30.0, acc_phy_tend=False, acc_phy_tend_interval=0.0)
    config = _resolve({"config_acc_phy_tend": ".true.", "config_acc_phy_tend_interval": 60})
    assert acc.packages(config) == {acc.TEND_PACKAGE}
    both = _resolve({"config_prec_acc_dt": 15, "config_acc_phy_tend": ".true.", "config_acc_phy_tend_interval": 60.0})
    assert acc.packages(both) == {acc.PREC_PACKAGE, acc.TEND_PACKAGE}
    for key in acc.OPTIONS:
        assert key not in config.options
    # the options live in the same &diagnostics group as the other WOOF options
    mixed = _resolve({"config_nwp_diagnostics": ".true.", "config_prec_acc_dt": 60})
    assert mixed.nwp_diagnostics and mixed.acc_diagnostics.prec_acc_dt == 60.0


@pytest.mark.parametrize("extra, message", [
    ({"config_prec_acc_dt": -1.0}, "nonnegative"),
    ({"config_prec_acc_dt": "'60'"}, "is real"),
    ({"config_acc_phy_tend": ".true."}, "needs config_acc_phy_tend_interval"),
    ({"config_acc_phy_tend": ".true.", "config_acc_phy_tend_interval": 90.5}, "positive whole minutes"),
    ({"config_acc_phy_tend_interval": 60}, "read by nothing"),
    ({"config_acc_phy_tend": 1, "config_acc_phy_tend_interval": 60}, "logical"),
])
def test_inconsistent_options_are_refused(extra, message):
    with pytest.raises(ValueError, match=message):
        _resolve(extra)


def test_rrfs_fork_route_refuses_the_options():
    from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist as resolve_rrfs
    for body in (" config_prec_acc_dt=60.0", " config_acc_phy_tend=.true.\n config_acc_phy_tend_interval=60"):
        text = f"&physics\n config_physics_suite='hrrrv5'\n/\n&diagnostics\n{body}\n/\n"
        with pytest.raises(ValueError, match="not available on the RRFS fork"):
            resolve_rrfs(text)


def _layout(extra):
    from woof.hex.mpasphys.framework_backend_v842 import registry_source
    from woof.hex.mpasphys.pool_v842 import RegistryLayout
    text = _namelist(extra)
    config = resolve_namelist(text, default_backend="hex-cuda")
    dims = dict(nCells=10, nEdges=30, nVertices=20, nVertLevels=12, maxEdges=6, maxEdges2=12,
                vertexDegree=3, nSoilLevels=4, nOznLevels=59, nAerLevels=1, nMonths=12)
    return RegistryLayout(registry_source(), config, text, dims)


ON = {"config_prec_acc_dt": 30.0, "config_acc_phy_tend": ".true.", "config_acc_phy_tend_interval": 60}


def test_pools_and_streams_only_with_the_options(tmp_path):
    from woof.hex.mpasphys.forecast_v842 import read_streams, write_default_streams
    from woof.hex.streams_v842 import field_definition, _all_packages
    off, on, prec = _layout(None), _layout(ON), _layout({"config_prec_acc_dt": 30.0})
    rows = lambda layout: {r["name"]: r for r in layout.fields if r["pool"] == "diag"}
    assert not set(acc.NAMES) & set(rows(off))
    assert set(acc.NAMES) <= set(rows(on))
    assert set(acc.PREC_NAMES) <= set(rows(prec)) and not set(acc.TEND_NAMES) & set(rows(prec))
    assert [r for r in on.fields if r["name"] not in acc.NAMES] == list(off.fields)
    assert {acc.PREC_PACKAGE, acc.TEND_PACKAGE} <= on.packages and not {acc.PREC_PACKAGE, acc.TEND_PACKAGE} & off.packages
    for name in acc.PREC_NAMES:
        assert rows(on)[name]["dimensions"] == ["nCells"] and rows(on)[name]["shape"] == [11]
    for name in acc.TEND_NAMES:
        assert rows(on)[name]["dimensions"] == ["nVertLevels", "nCells"] and rows(on)[name]["shape"] == [12, 11]
    streams = {}
    for label, layout in (("off", off), ("on", on)):
        write_default_streams(layout, tmp_path / f"streams.{label}")
        streams[label] = {s.name: s for s in read_streams(tmp_path / f"streams.{label}", layout)}
    assert set(acc.NAMES) <= set(streams["on"]["diagnostics"].fields)
    assert not set(acc.NAMES) & set(streams["off"]["diagnostics"].fields)
    assert {n: s.fields for n, s in streams["on"].items() if n != "diagnostics"} == \
           {n: s.fields for n, s in streams["off"].items() if n != "diagnostics"}
    assert [f for f in streams["on"]["diagnostics"].fields if f not in acc.NAMES] == \
           list(streams["off"]["diagnostics"].fields)
    d = field_definition("prec_acc_nc")
    assert d["attrs"]["units"] == "mm" and d["attrs"]["wrf_name"] == "PREC_ACC_NC"
    d = field_definition("athblten")
    assert d["attrs"]["units"] == "K" and d["attrs"]["wrf_name"] == "ATHBLTEN" and d["dimensions"][0] == "nVertLevels"
    assert {acc.PREC_PACKAGE, acc.TEND_PACKAGE} <= _all_packages()


def test_the_field_rows_are_the_wrf_routes_rows():
    from woof.core.output_diagnostics import HISTORY, PRECIP_HISTORY
    for name in acc.NAMES:
        wrf, units, text = (PRECIP_HISTORY.get(name) or HISTORY[name])
        assert FIELDS_ROW(name) == (wrf, units.strip(), text)


def FIELDS_ROW(name):
    a = acc.FIELDS[name]["attrs"]
    return a["wrf_name"], a["units"], a["long_name"]


def test_the_off_state_registry_streams_and_receipt_are_unchanged():
    """Digests recorded on origin/main (fe554cdb2), before these options existed (the p_lev / z_lev test's values)."""
    import hashlib
    import json
    import tempfile
    from woof.hex.mpasphys.forecast_v842 import write_default_streams
    layout = _layout(None)
    digest = hashlib.sha256(json.dumps([layout.fields, sorted(layout.packages), sorted(layout.dimensions.items())],
                                       sort_keys=True, default=str).encode()).hexdigest()[:16]
    assert digest == OFF_LAYOUT
    with tempfile.TemporaryDirectory() as directory:
        write_default_streams(layout, Path(directory) / "streams")
        assert hashlib.sha256((Path(directory) / "streams").read_bytes()).hexdigest()[:16] == OFF_STREAMS
    assert hashlib.sha256(json.dumps(_resolve().receipt(), sort_keys=True, default=str).encode()).hexdigest()[:16] == OFF_RECEIPT


OFF_LAYOUT, OFF_STREAMS, OFF_RECEIPT = "91afd1d2f4968e3f", "b868bf60f2a107f7", "bb186b77c0898a2c"


# ---------------------------------------------------------------------------
# the WRF oracle: module_diag_misc.F's own blocks, per step, on the same inputs
# ---------------------------------------------------------------------------
def _load_oracle():
    raw = (ORACLE / "inputs.bin").read_bytes()
    nc, nl, ns, cu_used = (int(v) for v in np.frombuffer(raw, "<i4", 4))
    dt_s, prec_min = (float(v) for v in np.frombuffer(raw, "<f4", 2, 16))
    hist_min = int(np.frombuffer(raw, "<i4", 1, 24)[0])
    body = np.frombuffer(raw, "<f4", offset=28)
    per = 4 * nc + 12 * nl * nc
    steps = []
    for s in range(ns):
        b = body[s * per:(s + 1) * per]
        steps.append(dict(
            raincv=b[:nc], rainncv=b[nc:2 * nc], snowncv=b[2 * nc:3 * nc], t2=b[3 * nc:4 * nc],
            rates=b[4 * nc:].reshape(12, nl, nc)))
    out = np.frombuffer((ORACLE / "oracle.bin").read_bytes(), "<f4")
    per_out = 3 * nc + 16 * nl * nc
    expected = []
    for s in range(ns):
        b = out[s * per_out:(s + 1) * per_out]
        expected.append(dict(prec=b[:3 * nc].reshape(3, nc), tend=b[3 * nc:].reshape(16, nl, nc)))
    return dict(nc=nc, nl=nl, ns=ns, cu=cu_used, dt=dt_s, prec_min=prec_min, hist_min=hist_min,
                steps=steps, expected=expected)


class _Utils:
    def __init__(self, written=()):
        self.written = set(written)

    def stream_inclusion_count(self, name, direction=None, **kw):
        return 1

    def field_will_be_written(self, name):
        return name in self.written


#: the module's WRF rate slots, in the oracle file's order, and the tend_physics field each one is
_ORACLE_RATES = ("rthmpten", "rqvmpten", "rthcuten", "rqvcuten", "rucuten", "rvcuten",
                 "rthblten", "rqvblten", "rublten", "rvblten", "rthratenlw", "rthratensw")


def _words_equal(got, want, err_msg):
    """Bitwise, as WRF's gfortran -ffp-contract=off words.  The one exception is the default CUDA build, whose NVRTC
    contracts ``acc + rate * dt`` into one fma (GPUWM_WRF_EXACT=1 compiles with --fmad=false and is bitwise): there the
    words are held to 4 eps of the field's largest word, the same arithmetic with one rounding fewer per fold."""
    from woof import wrf_exact
    from woof.backend import active
    if active() != "cuda" or wrf_exact.ENABLED:
        np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32), err_msg=err_msg)
        return
    # the fold of two terms of either sign: the error is in units of the larger term, not of a cancelled sum
    bound = 4 * np.finfo(np.float32).eps * max(float(np.abs(want).max()), 1e-30)
    assert np.abs(got.astype(np.float64) - want).max() <= bound, err_msg


@pytest.fixture
def cp():
    return pytest.importorskip("cupy")


def _owner(cp, nc, nl, with_cu=True, precip=("raincv", "rainncv", "snowncv")):
    fields, buffers = {}, {}
    rates = [r for r in _ORACLE_RATES if with_cu or "cuten" not in r]
    layout_fields = [dict(pool="tend_physics", pool_name=r) for r in rates]
    layout_fields += [dict(pool="diag_physics", pool_name=n) for n in (*precip, "t2m")]
    start = dt.datetime(2026, 1, 1)
    session = SimpleNamespace(layout=SimpleNamespace(dimensions=dict(nCellsSolve=nc, nVertLevels=nl), fields=layout_fields,
                                                     indices={}))

    def buffer(pool, name):
        shape = (nl + 1, nc + 1) if name in acc.TEND_NAMES else (nc + 1,)
        return buffers.setdefault((pool, name), cp.zeros(shape, dtype=cp.float32))

    return SimpleNamespace(cp=cp, session=session, authority=lambda pool, name: fields[pool, name], buffer=buffer,
                           fields=fields, buffers=buffers, writer=SimpleNamespace(start=start), start=start)


def _feed(owner, case, step):
    cp = owner.cp
    pad = lambda a: cp.asarray(np.concatenate([a, a[..., :1]], axis=-1))
    s = case["steps"][step]
    owner.fields["diag_physics", "raincv"] = pad(s["raincv"])
    owner.fields["diag_physics", "rainncv"] = pad(s["rainncv"])
    owner.fields["diag_physics", "snowncv"] = pad(s["snowncv"])
    owner.fields["diag_physics", "t2m"] = pad(s["t2"])
    for name, rate in zip(_ORACLE_RATES, s["rates"]):
        if ("tend_physics", name) in [(r["pool"], r["pool_name"]) for r in owner.session.layout.fields]:
            owner.fields["tend_physics", name] = pad(rate)


def _module(cp, case, *, prec=True, tend=True, written=acc.NAMES, **kw):
    owner = _owner(cp, case["nc"], case["nl"], **kw)
    config = acc.AccConfig(case["prec_min"] if prec else 0.0, tend, float(case["hist_min"]) if tend else 0.0)
    module = acc.AccumulatorModule(owner, config)
    utils = _Utils(written)
    module.setup(utils, case["dt"])
    return module, owner, utils


def _step(module, owner, case, step):
    """One hex step: its state is fed, the hooks carry the step's END time, then the fold."""
    _feed(owner, case, step)
    module.begin(step + 1, owner.start + dt.timedelta(seconds=(step + 1) * case["dt"]))
    module.update()


def test_every_accumulator_equals_wrf_module_diag_misc_after_every_step(cp):
    case = _load_oracle()
    module, owner, utils = _module(cp, case)
    for s in range(case["ns"]):
        _step(module, owner, case, s)
        module.compute()
        want = case["expected"][s]
        for i, name in enumerate(acc.PREC_NAMES):
            got = cp.asnumpy(owner.buffers["diag", name])[:case["nc"]]
            _words_equal(got, want["prec"][i], f"{name} step {s + 1}")
        for i, name in enumerate(acc.TEND_NAMES):
            got = cp.asnumpy(owner.buffers["diag", name])[:case["nl"], :case["nc"]]
            _words_equal(got, want["tend"][i], f"{name} step {s + 1}")
    # the file exercises both windows (3 min prec, 4 min tend at 60 s): there are resets inside it
    assert case["ns"] > 2 * case["hist_min"] and case["prec_min"] * 60 == 3 * case["dt"]
    assert (case["expected"][case["ns"] - 1]["tend"][0] != 0).any() and (case["expected"][case["ns"] - 1]["prec"][1] != 0).any()


def test_the_oracle_is_load_bearing():
    """A window that never reset (or reset one step off) would not reproduce the oracle's words."""
    case = _load_oracle()
    pc = np.zeros(case["nc"], np.float32)
    for s, step in enumerate(case["steps"]):
        pc = np.maximum(pc + step["raincv"], np.float32(0))        # no reset at all
    assert not np.array_equal(pc, case["expected"][-1]["prec"][0])
    # a clock one step late (reset after the window's last fold, not before its first) empties the frame at the
    # window's end, where WRF's holds the window
    end = 3 * 1 - 1
    assert case["expected"][end]["prec"][0].any()
    pc = np.zeros(case["nc"], np.float32)
    for step in case["steps"][:end + 1]:
        pc = np.maximum(pc + step["raincv"], np.float32(0))
    pc[:] = 0
    assert not np.array_equal(pc, case["expected"][end]["prec"][0])


def test_t0_frame_is_zero_and_an_unallocated_input_is_zero(cp):
    case = _load_oracle()
    module, owner, utils = _module(cp, case, precip=("raincv", "t2m"), with_cu=False)
    module.begin(0, owner.start)
    module.update()                                   # no step has run: nothing is folded
    module.compute()
    assert not any(cp.asnumpy(owner.buffers["diag", n]).any() for n in acc.NAMES)
    for s in range(3):
        _step(module, owner, case, s)
    assert not cp.asnumpy(module.acc["prec_acc_nc"]).any()          # rainncv not allocated: WRF's amount() of no scheme
    assert cp.asnumpy(module.acc["prec_acc_c"]).any()
    assert not cp.asnumpy(module.tend["athcuten"]).any()            # no cumulus package: the group is untouched
    assert cp.asnumpy(module.tend["athblten"]).any()
    assert not cp.asnumpy(module.tend["athshten"]).any()            # MPAS has no shallow-cumulus pool


def test_a_window_that_is_not_whole_steps_is_refused(cp):
    case = _load_oracle()
    owner = _owner(cp, case["nc"], case["nl"])
    with pytest.raises(ValueError, match="whole number"):
        acc.AccumulatorModule(owner, acc.AccConfig(2.5, False, 0.0)).setup(_Utils(acc.NAMES), 60.0)
    with pytest.raises(ValueError, match="whole number"):
        acc.AccumulatorModule(owner, acc.AccConfig(0.0, True, 5.0)).setup(_Utils(acc.NAMES), 70.0)


def test_only_carried_fields_accumulate_and_a_checkpoint_resumes(cp):
    case = _load_oracle()
    module, owner, utils = _module(cp, case)
    for s in range(5):
        _step(module, owner, case, s)
    state = module.state()
    assert set(state) == {acc.STATE_PREFIX + n for n in acc.NAMES + ("graup_acc_nc",)}
    fresh, f_owner, _ = _module(cp, case)
    fresh.resume(state)
    for s in range(5, case["ns"]):
        _step(fresh, f_owner, case, s)
        fresh.compute()
        want = case["expected"][s]
        for i, name in enumerate(acc.PREC_NAMES):
            np.testing.assert_array_equal(cp.asnumpy(f_owner.buffers["diag", name])[:case["nc"]].view(np.uint32),
                                          want["prec"][i].view(np.uint32))
    # a stream that carries none of them: nothing is allocated or folded
    class Nothing(_Utils):
        def stream_inclusion_count(self, name, direction=None, **kw):
            return 0
    owner2 = _owner(cp, case["nc"], case["nl"])
    idle = acc.AccumulatorModule(owner2, acc.AccConfig(case["prec_min"], True, float(case["hist_min"])))
    idle.setup(Nothing(), case["dt"])
    assert not idle.accumulates
    idle.begin(1, owner2.start + dt.timedelta(seconds=60))
    idle.update()
    idle.compute()
    assert not owner2.buffers


def test_resident_provider_folds_publishes_and_carries_the_module(cp):
    from woof.hex.mpasdiag.convective_v842 import MAXIMA
    from woof.hex.mpasphys.resident_output_diagnostics_v842 import PRODUCED, ResidentOutputDiagnostics as R
    case = _load_oracle()
    module, owner, utils = _module(cp, case)
    provider = R.__new__(R)
    provider.cp, provider.accdiag, provider.nwp, provider.tslist, provider.buffers = cp, module, None, None, {}
    provider._event_views, provider._resume, provider.updates, provider.acoustic = None, None, 0, {}
    provider.convective_maxima = MAXIMA
    calls = []
    provider.convective = SimpleNamespace(update=lambda: calls.append("convective"), reset=lambda: calls.append("reset"))
    assert R.produced is PRODUCED and not set(acc.NAMES) & set(PRODUCED)
    _feed(owner, case, 0)
    module.begin(1, owner.start + dt.timedelta(seconds=60))
    provider._update_convective()
    assert calls == ["convective"] and cp.asnumpy(module.acc["prec_acc_c"]).any()
    state = provider.accumulator_state()
    assert {acc.STATE_PREFIX + n for n in acc.NAMES} <= set(state)
    provider.resume_accumulators(state)
    assert set(provider._resume) == {acc.STATE_PREFIX + n for n in (*acc.NAMES, "graup_acc_nc")}
    provider.after_snapshot()           # clock windows: nothing is reset by a write
    assert cp.asnumpy(module.acc["prec_acc_c"]).any()
