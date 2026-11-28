"""E3SM v3.2.0's ocean-input accumulation (prep_ocn_accum, prep_ocn_accum_avg), word for word.

The reference is E3SM v3.2.0's own ``prep_ocn_accum`` and
``prep_ocn_accum_avg`` (cut verbatim out of
``driver-mct/main/prep_ocn_mod.F90``) over ``mct_avect_accum`` and
``mct_aVect_avg`` (cut verbatim out of ``share/util/mct_mod.F90``) and the
pinned MCT on one task, built with gfortran -O0 -ffp-contract=off and driven
by ``tools/e3sm_coupler_oracle/run_ocn_accum.F90``. The outputs are
committed (tests/data/oracles/e3sm_ocn_accum, PROVENANCE.md). When
``GFORTRAN`` (or gfortran on PATH), ``WOOF_E3SM_V320_SOURCE`` and
``WOOF_MCT_SOURCE`` are set, the fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with E3SM only to a tolerance, or only under strict
   numerics: the word-exact gate also runs in fast and strict child
   processes, over the count, the accumulator and x2o after every operation.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: windows of 1, 2, 3, 5 and 7, an average
   with nothing accumulated (twice), a window's first sample over the last
   average, signed zeros, subnormals and overflow.
4. A gate that cannot fail: a division for the reciprocal, an average at a
   count of 1 skipped or not, the first sample added, the count kept.
5. The default changed: nothing in woof/ calls this.

QUIRKS (E3SM v3.2.0, mct_mod.F90 / prep_ocn_mod.F90)
----------------------------------------------------
- The average multiplies by ``1/count`` rounded once (``mct_aVect_avg``'s
  ``ravg``), which is not ``x / count`` for counts that are not powers of 2
  (kept; a rounding choice, not a physical error, so no corrected option).
- An average with nothing accumulated copies the accumulator as it is (its
  zeros, or the last average again) into x2o (kept).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _numerics_child import MODES, run as run_child
from conftest import requires_gpu
import _oracle_toolchains
from woof.hex.coupled import ocn_accum_e3sm as oae
from woof.verify import e3sm_ocn_accum_oracle as oa

ROOT = Path(__file__).resolve().parents[1]


def _case(name):
    return next(c for c in oa.cases() if c["name"] == name)


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oa.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {case["name"]: oa.input_digest(case) for case in oa.cases()}


def test_fixture_covers_the_branches():
    outputs, aborts = oa.load_fixture()
    assert aborts == {} and set(outputs) == {c["name"] for c in oa.cases()}
    assert outputs["average_without_samples"]["counts"].tolist() == [0, 1, 2, 0, 0, 1, 0]
    assert max(outputs["windows_of_seven_and_five"]["counts"]) == 7
    # nothing accumulated: x2o becomes the zeroed accumulator, then the last average again
    o = outputs["average_without_samples"]
    assert (o["op0/x2o"] == 0).all() and o["op4/x2o"].tobytes() == o["op3/x2o"].tobytes()
    # the first sample of a window is copied over the last average
    c = _case("average_without_samples")
    assert o["op5/acc"].tobytes() == c["samples"][5].tobytes()
    # the reciprocal, not a division, and it shows
    w, c3 = outputs["window_of_three"], _case("window_of_three")
    s = (c3["samples"][0] + c3["samples"][1]) + c3["samples"][2]
    assert w["op3/x2o"].tobytes() == (s * (1.0 / 3.0)).tobytes()
    assert (s * (1.0 / 3.0) != s / 3.0).any()
    # a window of one is not scaled
    one, c1 = outputs["window_of_one"], _case("window_of_one")
    assert one["op1/x2o"].tobytes() == c1["samples"][0].tobytes()
    sp = outputs["signed_zeros_subnormals_overflow"]["op3/x2o"]
    assert np.signbit(sp[0:4]).all() and np.isinf(sp[10:12]).all()
    assert (sp[13] == np.nextafter(0.0, 1.0)).all() and (sp[6:10] != 0).any()


def test_the_default_path_does_not_call_the_accumulation():
    users = [p for p in (ROOT / "woof").rglob("*.py")
             if "ocn_accum_e3sm" in p.read_text(encoding="utf-8")
             and p.name not in ("ocn_accum_e3sm.py", "e3sm_ocn_accum_oracle.py")
             and p.relative_to(ROOT).as_posix() not in ("woof/hex/coupled/interval_e3sm.py",
                                                    "woof/verify/e3sm_coupler_interval_oracle.py")]
    assert users == []


@requires_gpu
def test_gpu_matches_the_e3sm_oracle_word_for_word():
    report = oa.measure()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_e3sm_oracle_word_for_word"], numerics, exact)


@requires_gpu
def test_bad_arrays_are_refused(monkeypatch):
    import cupy as cp
    acc = oae.OceanAccumulator(4, ("a", "b"))
    good = cp.zeros((4, 2), dtype=cp.float64)  # the mlx facade's zeros default to float32
    with pytest.raises(ValueError, match="once"):
        oae.OceanAccumulator(4, ("a", "a"))
    with pytest.raises(ValueError, match="lsize"):
        oae.OceanAccumulator(-1, ("a",))
    with pytest.raises(TypeError, match="cupy"):
        acc.accum(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="shape"):
        acc.accum(cp.zeros((4, 3), dtype=cp.float64))
    with pytest.raises(ValueError, match="shape"):
        acc.accum(cp.asfortranarray(cp.zeros((4, 2), dtype=cp.float64)))
    with pytest.raises(ValueError, match="accumulator"):
        acc.accum_avg(acc.acc)
    monkeypatch.setattr(oae, "_device_of", lambda a: 99)
    with pytest.raises(ValueError, match="device 99"):
        acc.accum(good)
    assert acc.count == 0


def _caught(case_name):
    outputs, _ = oa.load_fixture()
    got, want = oa.run_gpu(_case(case_name)), outputs[case_name]
    return any((got[k] != want[k]).any() if k == "counts" else oa.word_mismatches(got[k], want[k]) for k in want)


@requires_gpu
@pytest.mark.parametrize("original, mutated, case", [
    ("acc[i] = __dmul_rn(acc[i], ravg);", "acc[i] = __ddiv_rn(acc[i], 1.0 / ravg);", "window_of_three"),
    ("acc[i] = __dadd_rn(acc[i], x2o[i]);", "acc[i] = __dadd_rn(acc[i], __dmul_rn(x2o[i], 1.0000000000000002));",
     "window_of_two"),
])
def test_a_mutated_kernel_is_caught(monkeypatch, original, mutated, case):
    real = oae.module_source
    assert real().count(original) == 1, original
    monkeypatch.setattr(oae, "module_source", lambda: real().replace(original, mutated))
    oae._module_on.cache_clear()
    try:
        assert _caught(case)
    finally:
        oae._module_on.cache_clear()


@requires_gpu
def test_mutated_bookkeeping_is_caught(monkeypatch):
    real_accum, real_avg = oae.OceanAccumulator.accum, oae.OceanAccumulator.accum_avg

    def add_first(self, x2o):          # the first sample added to the last average
        self._check(x2o)
        self._launch("ocn_accum_add", (x2o,))
        self.count += 1
    monkeypatch.setattr(oae.OceanAccumulator, "accum", add_first)
    assert _caught("average_without_samples")
    monkeypatch.setattr(oae.OceanAccumulator, "accum", real_accum)

    def keep_count(self, x2o):         # the count not reset
        real_avg(self, x2o)
        self.count = 1
    monkeypatch.setattr(oae.OceanAccumulator, "accum_avg", keep_count)
    assert _caught("window_of_three")
    monkeypatch.setattr(oae.OceanAccumulator, "accum_avg", real_avg)
    monkeypatch.setattr(oae, "mct_avg_factor", lambda n: 1.0 / (n + 1))
    assert _caught("window_of_two")
    monkeypatch.setattr(oae, "mct_avg_factor", lambda n: 1.0 / n)

    def no_copy(self, x2o):            # x2o left as it was
        self._check(x2o)
        self.count = 0
    monkeypatch.setattr(oae.OceanAccumulator, "accum_avg", no_copy)
    assert _caught("average_without_samples")


def _oracle_toolchain():
    return _oracle_toolchains.e3sm_v320("driver-mct", "main", "prep_ocn_mod.F90")


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
    subprocess.run([sys.executable, str(tools / "make_fixture.py"), str(tmp_path / "build"), "--oracle", "accum",
                    "--out", str(tmp_path / "out")], check=True, cwd=ROOT)
    with np.load(tmp_path / "out" / "oracle-r8.npz") as got, np.load(oa.FIXTURE_DIR / "oracle-r8.npz") as want:
        assert sorted(got.files) == sorted(want.files)
        for k in want.files:
            assert got[k].tobytes() == want[k].tobytes(), k
    for name in ("aborts.json", "inputs.sha256"):
        assert (tmp_path / "out" / name).read_bytes() == (oa.FIXTURE_DIR / name).read_bytes(), name
