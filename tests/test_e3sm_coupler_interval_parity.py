"""One E3SM v3.2.0 coupler interval, chained, word for word.

The reference is E3SM v3.2.0's own coupler routines, each cut verbatim as
its single-routine oracle cuts it, linked together and driven over several
coupler steps in ``cime_comp_mod``'s RASM_OPTION1 order (aoflux_grid
'ocn'): ``prep_ocn_calc_i2x_ox``, ``seq_flux_atmocn_mct``, ``prep_ocn_mrg``,
``prep_ocn_accum``, ``seq_flux_ocnalb_mct``, ``prep_ocn_accum_avg``,
``prep_ice_calc_o2x_ix``, ``prep_ice_calc_a2x_ix``, ``prep_ice_mrg`` and
``seq_frac_set``, on the pinned MCT, one task, gfortran -O0
-ffp-contract=off (``tools/e3sm_coupler_oracle/run_coupler_interval.F90``).
The outputs are committed (tests/data/oracles/e3sm_coupler_interval,
PROVENANCE.md). When ``GFORTRAN`` (or gfortran on PATH),
``WOOF_E3SM_V320_SOURCE`` and ``WOOF_MCT_SOURCE`` are set, the fixture
itself is rebuilt and compared.

The flux calls glibc's binary64 ``atan`` (LGPL): an interval runs only with
the opt-in transcription (``WOOF_GLIBC239_LIBM64=1``) and is refused by
name without it.

THE BREAKAGES THESE PREVENT
---------------------------
1. A chain that agrees with E3SM only to a tolerance, or only under strict
   numerics: the word-exact gate also runs in fast and strict child
   processes, over xao, the accumulator and its count, x2o, x2i and the
   three fractions after every step.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: accumulation windows of 1, 2, 3 and 4
   samples cut by the ocean's alarms, the sea ice every step and every
   other, the albedos averaged and computed, the implicit stress and
   gustiness, dead models, an ocean or sea ice that is not prognostic, the
   component maps switched off, an atmosphere mapper as a matrix, a copy
   and a rearrangement.
4. A gate that cannot fail: phases out of order, the flux's mask taken from
   the fractions of the moment, a map taken when its switch is off, the
   accumulation before the merge, a copy where E3SM keeps the earlier
   value.
5. The default path running LGPL code: without the opt-in an interval is
   refused by name.
6. The default changed: nothing in woof/ calls this.

QUIRKS (E3SM v3.2.0, cime_comp_mod.F90)
---------------------------------------
- ``seq_flux_init_mct`` takes the flux grid's mask once, from the
  fractions at initialisation; later fractions (seq_frac_set's) do not
  change which points the flux computes (kept).
- With ``ice_c2_ocn``, ``ocn_c2_ice`` or ``atm_c2_ice`` off the mapped
  vector keeps prep_ocn_init's or prep_ice_init's zeros, and the merges
  read them (kept).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _numerics_child import MODES, run as run_child
from conftest import requires_gpu
from woof.hex.coupled import flux_atmocn_e3sm as fxe
from woof.hex.coupled import interval_e3sm as ie
from woof.hex.coupled import ocn_accum_e3sm as oae
from woof.verify import e3sm_coupler_interval_oracle as ci

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def glibc239(monkeypatch):
    """The opt-in glibc 2.39 binary64 atan transcription (LGPL, source checkout only)."""
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "1")
    fxe._module_on.cache_clear()
    yield
    fxe._module_on.cache_clear()


def _case(name):
    return next(c for c in ci.cases() if c["name"] == name)


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (ci.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {case["name"]: ci.input_digest(case) for case in ci.cases()}


def test_fixture_covers_the_branches():
    outputs, aborts = ci.load_fixture()
    cases = {c["name"]: c for c in ci.cases()}
    assert aborts == {} and set(outputs) == set(cases)
    specs = [c["spec"] for c in cases.values()]
    for key in ("implicit", "gust", "dead", "albav"):
        assert {s[key] for s in specs} == {True, False}, key
    for key in ("ocn_prognostic", "ice_prognostic", "ice_c2_ocn", "ocn_c2_ice", "atm_c2_ice"):
        assert {s[key] for s in specs} == {True, False}, key
    assert {s["i2a"] for s in specs} == {"matrix", "copy", "rearrange"}
    assert {s["ice_every"] for s in specs} == {1, 2}
    # the accumulation windows the ocean's alarms cut: 1, 3, 3 samples
    assert outputs["ocean_every_third"]["counts"].tolist() == [0, 1, 2, 0, 1, 2, 0]
    assert outputs["implicit_gust"]["counts"].tolist() == [0, 1, 2, 3]
    # after an average x2o is the accumulator scaled, before the next merge overwrites it
    o = outputs["ocean_every_third"]
    assert o["step3/x2o"].tobytes() == o["step3/acc"].tobytes()
    assert o["step3/x2o"].tobytes() != o["step2/x2o"].tobytes()
    # not prognostic: x2o and the accumulator stay as they were set up
    c, n = cases["ocean_not_prognostic"], outputs["ocean_not_prognostic"]
    for k in range(c["spec"]["steps"]):
        assert n[f"step{k}/x2o"].tobytes() == c["values"]["x2o"].tobytes()
        assert not n[f"step{k}/acc"].any()
    c, n = cases["ice_not_prognostic"], outputs["ice_not_prognostic"]
    assert n["step2/x2i"].tobytes() == c["values"]["x2i"].tobytes()
    # the fractions follow the sea ice between steps
    o = outputs["albav_ice_every_other_epbal"]
    ifrac = list(ci.LISTS["fractions_o"]).index("ifrac")
    assert o["step0/fractions_o"][:, ifrac].tobytes() != o["step2/fractions_o"][:, ifrac].tobytes()
    assert o["step0/fractions_o"][:, ifrac].tobytes() == o["step1/fractions_o"][:, ifrac].tobytes()
    # ... and with flux_albav the albedos' radiative fractions follow them a step later
    ifrad = list(ci.LISTS["fractions_o"]).index("ifrad")
    assert o["step1/fractions_o"][:, ifrad].tobytes() == o["step0/fractions_o"][:, ifrac].tobytes()


def test_the_default_path_does_not_call_the_interval():
    users = [p for p in (ROOT / "woof").rglob("*.py")
             if "interval_e3sm" in p.read_text(encoding="utf-8")
             and p.name not in ("interval_e3sm.py", "e3sm_coupler_interval_oracle.py")]
    assert users == []
    # importing it loads none of the routines it chains
    code = ("import sys, woof.hex.coupled.interval_e3sm\n"
            "assert not [m for m in sys.modules if m.startswith('woof.hex.coupled.') and m.endswith('_e3sm')\n"
            "            and m != 'woof.hex.coupled.interval_e3sm'], 'loaded'\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


@requires_gpu
def test_without_the_opt_in_an_interval_is_refused(monkeypatch):
    import cupy as cp
    monkeypatch.delenv("WOOF_GLIBC239_LIBM64", raising=False)
    fxe._module_on.cache_clear()
    case = _case("ocean_every_third")
    interval = ci.interval_for(case)
    with pytest.raises(fxe.FluxRefusal, match="glibc|LGPL|WOOF_GLIBC239_LIBM64"):
        interval.step(cp.asarray(case["steps"][0]["a2x"]), ocn_run=True, ice_run=False, ocn_next=False)


@requires_gpu
@pytest.mark.usefixtures("glibc239")
def test_gpu_matches_the_e3sm_oracle_word_for_word():
    report = ci.measure()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_e3sm_oracle_word_for_word"], numerics, exact,
              env={"WOOF_GLIBC239_LIBM64": "1"})


def _caught(case_name):
    outputs, _ = ci.load_fixture()
    got, want = ci.run_gpu(_case(case_name)), outputs[case_name]
    return any((got[k] != want[k]).any() if k == "counts" else ci.word_mismatches(got[k], want[k]) for k in want)


@requires_gpu
def test_a_reordered_or_rewired_chain_is_caught(glibc239, monkeypatch):
    real = {name: getattr(ie.CouplerInterval, name) for name in
            ("atmocn_setup", "ice_setup_send", "update_fractions", "step", "__init__")}

    def restore():
        for name, fn in real.items():
            monkeypatch.setattr(ie.CouplerInterval, name, fn)

    # the fractions updated before the sea ice's export arrives
    def fractions_first(self, a2x, *, ocn_run, ice_run, ocn_next, i2x=None, o2x=None):
        self.atmocn_setup(a2x)
        if ocn_run:
            self.ocn_setup_send()
        self.update_fractions()
        if ice_run:
            self.ice_setup_send()
            self.ice_recv_post(i2x)
        if ocn_next:
            self.ocn_recv_post(o2x)
    monkeypatch.setattr(ie.CouplerInterval, "step", fractions_first)
    assert _caught("ocean_every_third")
    restore()

    # the ocean's new export taken before the sea ice's merge reads o2x
    def ocean_first(self, a2x, *, ocn_run, ice_run, ocn_next, i2x=None, o2x=None):
        self.atmocn_setup(a2x)
        if ocn_run:
            self.ocn_setup_send()
        if ocn_next:
            self.ocn_recv_post(o2x)
        if ice_run:
            self.ice_setup_send()
            self.ice_recv_post(i2x)
        self.update_fractions()
    monkeypatch.setattr(ie.CouplerInterval, "step", ocean_first)
    assert _caught("ocean_every_third")
    restore()

    # the flux's mask taken from the fractions of the moment
    def fresh_mask(self, a2x):
        self.mask = fxe.flux_mask(dom=self._v["dom"], dom_fields=self.lists["dom"],
                                  fractions=self._v["fractions_o"], fractions_fields=self.lists["fractions_o"])
        real["atmocn_setup"](self, a2x)
    monkeypatch.setattr(ie.CouplerInterval, "atmocn_setup", fresh_mask)
    assert _caught("ocean_every_third")
    restore()

    # the component maps taken whatever their switches say
    def always_map(self, a2x):
        self._v["i2x_ox"][...] = self._v["i2x"]
        real["atmocn_setup"](self, a2x)
    monkeypatch.setattr(ie.CouplerInterval, "atmocn_setup", always_map)
    assert _caught("no_component_maps")
    restore()

    def always_map_ice(self):
        self._v["o2x_ix"][...] = self._v["o2x"]
        self._v["a2x_ix"][...] = self._v["a2x"]
        real["ice_setup_send"](self)
    monkeypatch.setattr(ie.CouplerInterval, "ice_setup_send", always_map_ice)
    assert _caught("no_component_maps")
    restore()

    # the sea ice's merge reading the ocean's export itself, not the copy the coupler took
    def no_copy(self):
        self._v["o2x_ix"] = self._v["o2x"]
        real["ice_setup_send"](self)
    monkeypatch.setattr(ie.CouplerInterval, "ice_setup_send", no_copy)
    assert _caught("no_component_maps")
    restore()


@requires_gpu
def test_the_accumulation_before_the_merge_is_caught(glibc239, monkeypatch):
    real_accum = oae.OceanAccumulator.accum
    pending = {}

    # accumulate the previous merge's x2o, then merge
    def late(self, x2o):
        prev = pending.get(id(self))
        pending[id(self)] = x2o.copy()
        real_accum(self, x2o if prev is None else prev)
    monkeypatch.setattr(oae.OceanAccumulator, "accum", late)
    assert _caught("ocean_every_third")


@requires_gpu
def test_what_is_not_ported_is_refused_by_name(glibc239):
    case = _case("ocean_every_third")
    with pytest.raises(ie.IntervalRefusal, match="wav_ocn_coup"):
        ie.Switches(wav_ocn_coup="twoway").check()
    with pytest.raises(ie.IntervalRefusal, match="wav_ice_coup"):
        ie.Switches(wav_ice_coup="twoway").check()
    flux = fxe.FluxOptions(wav_atm_coup="twoway")
    interval = ci.interval_for(case)
    with pytest.raises(ie.IntervalRefusal, match="wav_atm_coup"):
        ie.CouplerInterval(case["lists"], case["values"], n=interval.n, n_a=interval.n_a,
                           mapper_i2a=interval.mapper_i2a, flux=flux)


@requires_gpu
def test_bad_set_ups_are_refused(glibc239):
    import cupy as cp
    case = _case("ocean_every_third")
    interval = ci.interval_for(case)
    kw = dict(n=interval.n, n_a=interval.n_a, mapper_i2a=interval.mapper_i2a, flux=interval.flux)
    with pytest.raises(TypeError, match="Switches"):
        ie.CouplerInterval(case["lists"], case["values"], switches={}, **kw)
    with pytest.raises(TypeError, match="ocn_prognostic"):
        ie.CouplerInterval(case["lists"], case["values"], switches=ie.Switches(ocn_prognostic=1), **kw)
    with pytest.raises(TypeError, match="flux_epbalfact"):
        ie.CouplerInterval(case["lists"], case["values"], switches=ie.Switches(flux_epbalfact=True), **kw)
    with pytest.raises(ValueError, match="lists has no x2i"):
        ie.CouplerInterval({k: v for k, v in case["lists"].items() if k != "x2i"}, case["values"], **kw)
    with pytest.raises(ValueError, match="values has no xao"):
        ie.CouplerInterval(case["lists"], {k: v for k, v in case["values"].items() if k != "xao"}, **kw)
    with pytest.raises(ValueError, match="x2o must be float64"):
        ie.CouplerInterval(case["lists"], dict(case["values"], x2o=case["values"]["x2o"][:-1]), **kw)
    with pytest.raises(ValueError, match="repeats"):
        ie.CouplerInterval(dict(case["lists"], x2o=case["lists"]["x2o"] + case["lists"]["x2o"][:1]),
                           case["values"], **kw)
    with pytest.raises(TypeError, match="Mapper"):
        ie.CouplerInterval(case["lists"], case["values"], **dict(kw, mapper_i2a="copy"))
    with pytest.raises(ValueError, match="n_a"):
        ie.CouplerInterval(case["lists"], case["values"], **dict(kw, n_a=0))
    a2x = cp.asarray(case["steps"][0]["a2x"])
    with pytest.raises(ValueError, match="new export"):
        interval.step(a2x, ocn_run=True, ice_run=True, ocn_next=False)
    with pytest.raises(TypeError, match="ice_run"):
        interval.step(a2x, ocn_run=True, ice_run=1, ocn_next=False)
    with pytest.raises(ValueError, match="a2x must be float64"):
        interval.atmocn_setup(a2x[:-1])


def _oracle_toolchain():
    e3sm, mct = os.environ.get("WOOF_E3SM_V320_SOURCE"), os.environ.get("WOOF_MCT_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not e3sm or not mct or not gfortran or not Path(e3sm, "driver-mct", "main", "prep_ocn_mod.F90").is_file():
        return None
    return gfortran, Path(e3sm), Path(mct)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the E3SM coupler oracle build needs gfortran (pixi -e mpas-oracle), "
                           "WOOF_E3SM_V320_SOURCE (an E3SM v3.2.0 tree with driver-mct and share/util) and "
                           "WOOF_MCT_SOURCE (MCT at 82b0071)")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, e3sm, mct = _oracle_toolchain()
    env = dict(os.environ, GFORTRAN=gfortran)
    tools = ROOT / "tools" / "e3sm_coupler_oracle"
    subprocess.run(["bash", str(tools / "build.sh"), str(e3sm), str(mct), str(tmp_path / "build")], check=True,
                   env=env, capture_output=True)
    subprocess.run([sys.executable, str(tools / "make_fixture.py"), str(tmp_path / "build"), "--oracle", "interval",
                    "--out", str(tmp_path / "out")], check=True, cwd=ROOT)
    with np.load(tmp_path / "out" / "oracle-r8.npz") as got, np.load(ci.FIXTURE_DIR / "oracle-r8.npz") as want:
        assert sorted(got.files) == sorted(want.files)
        for k in want.files:
            assert got[k].tobytes() == want[k].tobytes(), k
    for name in ("aborts.json", "inputs.sha256"):
        assert (tmp_path / "out" / name).read_bytes() == (ci.FIXTURE_DIR / name).read_bytes(), name
