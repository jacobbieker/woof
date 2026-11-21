"""The E3SM coupled step on the coupling spine, word for word.

:func:`woof.hex.coupled.run_e3sm.run` drives a coupler interval over a
:class:`~woof.hex.coupled.clock.CoupledClock`, with the ocean's and sea
ice's run alarms cut from their coupling periods and the components called
where ``cime_comp_mod``'s RASM_OPTION1 loop runs them. The reference is
the chained Fortran oracle of ``tests/test_e3sm_coupler_interval_parity.py``
(E3SM v3.2.0's coupler routines cut verbatim and linked): its cases, driven
through the spine with components scripted from the cases, must give its
words after every step.

THE BREAKAGES THESE PREVENT
---------------------------
1. A spine whose alarms or call order differ from the driver's: every
   non-stopping case is graded word for word after every step, in fast and
   strict children, and alarms one step off are caught.
2. A hand-over that drops or changes words: a component that holds a
   different import than the coupler sent is refused by the channel's name.
3. A stop swallowed by the spine: seq_frac_set's stop comes through.
4. The default changed: nothing in woof/ calls the spine, and importing it
   loads none of the coupler routines.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _numerics_child import MODES, run as run_child
from conftest import requires_gpu
from woof.hex.coupled import flux_atmocn_e3sm as fxe
from woof.hex.coupled import run_e3sm as re3
from woof.hex.coupled import seq_frac_e3sm as sfe
from woof.hex.coupled.clock import CoupledClock
from woof.hex.errors import ConfigurationRefusal
from woof.verify import e3sm_coupler_interval_oracle as ci

ROOT = Path(__file__).resolve().parents[1]
DT = 1800.0


@pytest.fixture
def glibc239(monkeypatch):
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "1")
    fxe._module_on.cache_clear()
    yield
    fxe._module_on.cache_clear()


class Scripted:
    """The components, scripted from an oracle case."""

    def __init__(self, case, *, tamper=None):
        self.case, self.tamper, self.x2o = case, tamper, None

    def _step(self, s):
        return self.case["steps"][s]

    def atmosphere(self, s):
        import cupy as cp
        return cp.asarray(self._step(s)["a2x"])

    def inputs(self, s):
        import cupy as cp
        st = self._step(s)
        out = dict(nextsw_cday=st["nextsw_cday"], flux_epbalfact=st["flux_epbalfact"])
        if st["side"] is not None:
            out.update({k: cp.asarray(v) for k, v in st["side"].items()})
        return out

    def ocean_start(self, s, x2o):
        self.x2o = x2o
        if self.tamper == "x2o":
            x2o = x2o * 1.0000000000000002
        return x2o

    # (alarms other than the case's ask for exports it has none for: the starting ones)
    def ocean_finish(self, s):
        import cupy as cp
        o2x = self._step(s)["o2x"]
        return cp.asarray(self.case["values"]["o2x"] if o2x is None else o2x)

    def seaice(self, s, x2i):
        import cupy as cp
        i2x = self._step(s)["i2x"]
        return cp.asarray(self.case["values"]["i2x"] if i2x is None else i2x), x2i


def _clock(case):
    return CoupledClock(start=0.0, end=case["spec"]["steps"] * DT, cadence=DT,
                        dts={"atmosphere": DT, "ocean": DT, "seaice": DT})


def _alarms(case, clock):
    s = case["spec"]
    return re3.Alarms.from_clock(clock, ocean_period=s["ocn_every"] * DT, seaice_period=s["ice_every"] * DT)


def _run(case, *, alarms=None, components=None):
    import cupy as cp
    clock = _clock(case)
    out, counts = {}, []

    def record(s, interval):
        counts.append(interval.accumulator.count)
        for key in ci.OUTPUTS:
            out[f"step{s}/{key}"] = cp.asnumpy(interval.vector(key))

    n = re3.run(clock, ci.interval_for(case), components or Scripted(case), alarms=alarms or _alarms(case, clock),
                on_step=record)
    assert n == case["spec"]["steps"]
    out["counts"] = np.array(counts, dtype=np.int32)
    return out


def _mismatches(got, want):
    return {k: (int((got[k] != want[k]).sum()) if k == "counts" else ci.word_mismatches(got[k], want[k]))
            for k in want}


def test_the_spine_alarms_are_the_oracle_alarms():
    for case in ci.cases():
        clock = _clock(case)
        alarms = _alarms(case, clock)
        assert [alarms.at(w.index) for w in clock.intervals()] == ci.alarms(case["spec"]), case["name"]


def test_a_coupling_period_off_the_coupler_step_is_refused():
    clock = CoupledClock(start=0.0, end=4 * DT, cadence=DT, dts={"ocean": DT})
    with pytest.raises(ValueError, match="ocean coupling period"):
        re3.Alarms.from_clock(clock, ocean_period=1.5 * DT, seaice_period=DT)
    with pytest.raises(ValueError, match="seaice coupling period"):
        re3.Alarms.from_clock(clock, ocean_period=DT, seaice_period=0.0)


def test_nothing_on_the_default_path_calls_the_spine():
    users = [p for p in (ROOT / "woof").rglob("*.py")
             if "run_e3sm" in p.read_text(encoding="utf-8") and p.relative_to(ROOT).as_posix() !=
             "woof/hex/coupled/run_e3sm.py"]
    assert users == []
    code = ("import sys, woof.hex.coupled.run_e3sm\n"
            "assert not [m for m in sys.modules if m.startswith('woof.hex.coupled.') and m.endswith('_e3sm')\n"
            "            and m != 'woof.hex.coupled.run_e3sm'], 'loaded'\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


@requires_gpu
@pytest.mark.usefixtures("glibc239")
def test_the_spine_matches_the_e3sm_oracle_word_for_word():
    outputs, _ = ci.load_fixture()
    bad = {}
    for case in ci.cases():
        if case["aborts"]:
            continue
        report = _mismatches(_run(case), outputs[case["name"]])
        bad[case["name"]] = {k: v for k, v in report.items() if v}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_the_spine_matches_the_e3sm_oracle_word_for_word"], numerics, exact,
              env={"WOOF_GLIBC239_LIBM64": "1"})


@requires_gpu
def test_alarms_one_step_off_are_caught(glibc239):
    outputs, _ = ci.load_fixture()
    case = next(c for c in ci.cases() if c["name"] == "ocean_every_third")
    got = _run(case, alarms=re3.Alarms(ocean=4, seaice=1))
    assert any(_mismatches(got, outputs[case["name"]]).values())
    case = next(c for c in ci.cases() if c["name"] == "albav_ice_every_other_epbal")
    got = _run(case, alarms=re3.Alarms(ocean=2, seaice=1))
    assert any(_mismatches(got, outputs[case["name"]]).values())


@requires_gpu
def test_a_changed_hand_over_is_refused_by_the_channel_name(glibc239):
    case = next(c for c in ci.cases() if c["name"] == "ocean_every_third")
    with pytest.raises(ConfigurationRefusal, match="cpl2ocn_x2o"):
        _run(case, components=Scripted(case, tamper="x2o"))


@requires_gpu
def test_the_stop_comes_through_the_spine(glibc239):
    _, aborts = ci.load_fixture()
    case = next(c for c in ci.cases() if c["name"] == "frac_abort_mid_run")
    with pytest.raises(sfe.FracAbort) as stopped:
        _run(case)
    assert str(stopped.value) == aborts[case["name"]]


@requires_gpu
def test_bad_arguments_are_refused(glibc239):
    case = ci.cases()[0]
    clock = _clock(case)
    with pytest.raises(TypeError, match="CouplerInterval"):
        re3.run(clock, object(), Scripted(case), alarms=_alarms(case, clock))
    with pytest.raises(TypeError, match="Alarms"):
        re3.run(clock, ci.interval_for(case), Scripted(case), alarms=(3, 1))
