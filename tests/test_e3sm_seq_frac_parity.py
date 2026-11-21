"""E3SM v3.2.0's coupler surface fractions update (seq_frac_set), word for word.

The reference is E3SM v3.2.0's own ``seq_frac_set`` and ``seq_frac_check``
(cut verbatim out of ``driver-mct/main/seq_frac_mct.F90``) with the maps
they make, ``seq_map_map`` and ``seq_map_avNormArr`` (cut out of
``seq_map_mod.F90``), over the pinned MCT on one task, built with gfortran
-O0 -ffp-contract=off and driven by
``tools/e3sm_coupler_oracle/run_seq_frac.F90``. The outputs are committed
(tests/data/oracles/e3sm_seq_frac, PROVENANCE.md). When ``GFORTRAN`` (or
gfortran on PATH), ``WOOF_E3SM_V320_SOURCE`` and ``WOOF_MCT_SOURCE`` are
set, the fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with E3SM only to a tolerance, or only under strict
   numerics: the word-exact gate also runs in fast and strict child
   processes, over the three fraction vectors.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: each mapper kind on each map, each
   component absent, a sparse matrix in an unsorted file order with
   repeated elements and empty rows, ice fractions of 0, -0, 1, tiny and at
   the bound, land and coastal points, each check stopping the coupler
   (every checked fraction's bound on its own), NaN passing as in E3SM and
   not hiding a bad value, and the dead models reporting.
4. A check that drifts from E3SM's: the stops (stage and MCT's message), the
   reports of the dead models, the single-precision eps.
5. A gate that cannot fail: mutated kernels, constants and element orders
   must mismatch or change the outcome.
6. The default changed: nothing in woof/ calls this.

QUIRKS (E3SM v3.2.0, seq_frac_mct.F90 / MCT)
--------------------------------------------
- eps_fracsum and eps_fracval are real(r8) parameters set from the
  default-real literal 1.0e-02: float32(0.01) widened, 0.009999999776482582
  (kept; abort_ice_float32_eps and abort_atm_sum_float32_eps sit between
  that and a double 0.01).
- seq_frac_check's zfrac report prints kminval/kmaxval (lfrin's), not
  zminval/zmaxval; only the log, the test is right (not ported: the log).
- seq_frac_set looks up the domain's "frac" with perrWith: a missing field
  is reported and the index is 0, and the product reads rAttr(0,:), outside
  the vector (refused).
- MCT's sparse matrix multiply sums each row in the order its init leaves
  the elements: sorted by row, then column, with repeated elements in the
  mapping file's order, not the file's order (kept; the matrices here are
  shuffled and repeat elements).
- A -0 ice fraction stays -0 through a copy or rearrangement and becomes +0
  through the matrix (the row's sum starts from 0).
- NaN fractions pass seq_frac_check: gfortran's minval/maxval skip NaN and
  a comparison with NaN is false (kept; a NaN's payload is the processor's,
  so NaN words compare as NaN).
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
from woof.hex.coupled import seq_frac_e3sm as sfe
from woof.verify import e3sm_seq_frac_oracle as sf

ROOT = Path(__file__).resolve().parents[1]


def _case(name):
    return next(c for c in sf.cases() if c["name"] == name)


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (sf.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {case["name"]: sf.input_digest(case) for case in sf.cases()}


def test_fixture_covers_the_branches():
    outputs, aborts = sf.load_fixture()
    cases = {c["name"]: c for c in sf.cases()}
    assert set(outputs) | set(aborts) == set(cases)
    assert set(aborts) == {n for n, c in cases.items() if c["aborts"]}
    run = [c["spec"] for c in cases.values() if not c["aborts"]]
    for key in ("i2o", "i2a"):
        assert {s[key] for s in run} == set(sfe.KINDS), key
    for key in ("atm", "ice", "ocn", "dead"):
        assert {s[key] for s in run} == {True, False}, key
    # the stops: each stage, both sides of the single-precision eps, MCT's
    assert {line.split("]")[0] for line in aborts.values() if line.startswith("(seq_frac_check)")} == {
        f"(seq_frac_check) [{s}" for s in sfe.STAGES}
    assert any("FATAL" in line for line in aborts.values())
    # the dead models report every stage and stop at none
    assert list(outputs["dead_reports_all"]["reported"]) == list(sfe.STAGES)
    # the matrices: unsorted, repeated elements, empty rows
    m = _case("mpas_default")["matrices"]["i2a"]
    rows, cols = m[0], m[1]
    pairs = list(zip(rows.tolist(), cols.tolist()))
    assert pairs != sorted(pairs) and len(set(pairs)) < len(pairs)
    assert len(set(rows.tolist())) < _case("mpas_default")["spec"]["na"]
    # -0 through the rearrangement stays -0; through the matrix it is +0
    d = outputs["mpas_default"]
    assert d["fractions_i"][1, 1] == 0 and np.signbit(d["fractions_i"][1, 1])
    assert np.signbit(d["fractions_o"][1, 1])
    assert 2 in cols.tolist()  # the -0 point feeds an atmosphere row
    a = d["fractions_a"][:, 1:3]
    assert not ((a == 0) & np.signbit(a)).any()
    # the lower bound exactly: ifrac -eps, ofrac 1 + eps, both kept
    assert d["fractions_i"][4, 1] == -sf.EPS and d["fractions_i"][4, 2] == 1.0 + sf.EPS
    # every checked fraction's bound stops the coupler on its own grid
    poked = {what.split(":")[1] for c in cases.values() if c["aborts"] for what, _, _ in c["spec"]["poke"]
             if ":" in what}
    assert {"gfrac", "rfrac", "wfrac", "zfrac", "lfrin"} <= poked
    assert set(sfe.CHECKED) <= set().union(*(c["spec"][k] for c in cases.values()
                                             for k in ("fi_fields", "fo_fields", "fa_fields")))
    # NaN passes (gfortran's minval/maxval skip it) and does not hide a bad value
    assert all(np.isnan(outputs["nan_passes"][k]).any() for k in ("fractions_i", "fractions_o", "fractions_a"))
    assert aborts["abort_nan_beside_bound"].startswith("(seq_frac_check) [ice set]")
    # absent components leave their vectors alone
    for name, key in (("atm_absent", "fractions_a"), ("ocn_absent", "fractions_o")):
        assert sf.word_mismatches(outputs[name][key], _case(name)[key]) == 0
    for key in ("fractions_i", "fractions_o", "fractions_a"):
        assert sf.word_mismatches(outputs["ice_absent"][key], _case("ice_absent")[key]) == 0
    # only ofrac and ifrac are written on the ocean and atmosphere grids
    for name in ("mpas_default", "ocean_matrix_atm_copy"):
        c = _case(name)
        assert sf.word_mismatches(outputs[name]["fractions_o"][:, [0, 3, 4]], c["fractions_o"][:, [0, 3, 4]]) == 0
        assert sf.word_mismatches(outputs[name]["fractions_a"][:, [0, 3, 4]], c["fractions_a"][:, [0, 3, 4]]) == 0


def test_the_eps_is_single_precision():
    assert sfe.EPS_FRACVAL == sfe.EPS_FRACSUM == float(np.float32(0.01)) == sf.EPS
    assert sfe.EPS_FRACVAL != 0.01


def test_the_default_path_does_not_call_seq_frac():
    users = [p for p in (ROOT / "woof").rglob("*.py")
             if "seq_frac_e3sm" in p.read_text(encoding="utf-8")
             and p.name not in ("seq_frac_e3sm.py", "e3sm_seq_frac_oracle.py")
             and p.relative_to(ROOT).as_posix() not in ("woof/hex/coupled/interval_e3sm.py",
                                                    "woof/verify/e3sm_coupler_interval_oracle.py")]
    assert users == []
    siblings = sorted(p.stem for p in (ROOT / "woof" / "hex" / "coupled").glob("*.py")
                      if p.stem not in ("__init__", "seq_frac_e3sm"))
    code = ("import sys, importlib, woof, woof.hex, woof.hex.coupled\n"
            f"for m in {siblings!r}: importlib.import_module('woof.hex.coupled.' + m)\n"
            "assert 'woof.hex.coupled.seq_frac_e3sm' not in sys.modules, 'loaded'\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_the_mapper_orders_rows_as_mct_does():
    m = sfe.Mapper("matrix", rows=(2, 1, 2, 2, 1), cols=(3, 2, 1, 3, 1), weights=(0.1, 0.2, 0.3, 0.4, 0.5),
                   n_dst=3, n_src=3)
    start, cols, w = m._csr
    assert start.tolist() == [0, 2, 5, 5]
    assert cols.tolist() == [0, 1, 0, 2, 2] and w.tolist() == [0.5, 0.2, 0.3, 0.1, 0.4]


@pytest.mark.parametrize("kwargs, match", [
    (dict(kind="sparse"), "unknown mapper kind"),
    (dict(kind="matrix", rows=(1, 4), cols=(1, 1), weights=(1.0, 1.0), n_dst=3, n_src=3), "outside the grids"),
    (dict(kind="matrix", rows=(1,), cols=(0,), weights=(1.0,), n_dst=3, n_src=3), "outside the grids"),
    (dict(kind="matrix", rows=(1, 2), cols=(1,), weights=(1.0,), n_dst=3, n_src=3), "same length"),
    (dict(kind="matrix", n_dst=-1, n_src=3), "negative"),
])
def test_bad_mappers_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sfe.Mapper(**kwargs)


def _device(case, **override):
    import cupy as cp
    dev = {k: cp.asarray(np.ascontiguousarray(case[k], dtype=np.float64))
           for k in ("i2x", "dom", "fractions_i", "fractions_o", "fractions_a")}
    i2o, i2a = sf.mappers_for(case)
    s = case["spec"]
    kw = dict(i2x=dev["i2x"], i2x_fields=case["i2x_fields"], dom=dev["dom"], dom_fields=case["dom_fields"],
              fractions_i=dev["fractions_i"], fi_fields=s["fi_fields"], fractions_o=dev["fractions_o"],
              fo_fields=s["fo_fields"], fractions_a=dev["fractions_a"], fa_fields=s["fa_fields"], mapper_i2o=i2o,
              mapper_i2a=i2a, atm_present=s["atm"], ice_present=s["ice"], ocn_present=s["ocn"], dead=s["dead"])
    kw.update(override)
    return kw


@requires_gpu
def test_gpu_matches_the_e3sm_oracle_word_for_word():
    report = sf.measure()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_e3sm_oracle_word_for_word"], numerics, exact)


@requires_gpu
def test_the_port_stops_where_e3sm_does():
    _, aborts = sf.load_fixture()
    for case in sf.cases():
        if not case["aborts"]:
            continue
        with pytest.raises((sfe.FracAbort, sfe.CouplingAbort)) as stopped:
            sfe.seq_frac_set(**_device(case))
        assert str(stopped.value) == aborts[case["name"]], case["name"]


@requires_gpu
def test_what_e3sm_gets_wrong_or_mct_stops_on_is_refused():
    import cupy as cp
    case = _case("mpas_default")
    # a domain without frac: E3SM reads rAttr(0,:)
    kw = _device(case)
    k = case["dom_fields"].index("frac")
    keep = [j for j in range(len(case["dom_fields"])) if j != k]
    kw.update(dom=cp.ascontiguousarray(kw["dom"][:, keep]), dom_fields=tuple(case["dom_fields"][j] for j in keep))
    with pytest.raises(ValueError, match="rAttr\\(0,:\\)"):
        sfe.seq_frac_set(**kw)
    # fractions_o without ofrac: MCT's indexRA stops
    kw = _device(case)
    kw.update(fractions_o=cp.ascontiguousarray(kw["fractions_o"][:, [0, 1, 3, 4]]),
              fo_fields=("afrac", "ifrac", "ifrad", "ofrad"))
    with pytest.raises(sfe.CouplingAbort, match='"ofrac"'):
        sfe.seq_frac_set(**kw)
    # a rearrangement between grids of different sizes
    kw = _device(_case("ocean_matrix_atm_copy"), mapper_i2o=sfe.Mapper("rearrange"))
    with pytest.raises(ValueError, match="one size"):
        sfe.seq_frac_set(**kw)
    # a matrix for other grids
    kw = _device(case, mapper_i2a=sfe.Mapper("matrix", n_dst=5, n_src=case["spec"]["ni"]))
    with pytest.raises(ValueError, match="maps"):
        sfe.seq_frac_set(**kw)
    # host arrays, bad shapes, repeated fields
    kw = _device(case, fractions_a=case["fractions_a"])
    with pytest.raises(TypeError, match="fractions_a must be a cupy"):
        sfe.seq_frac_set(**kw)
    kw = _device(case)
    kw.update(fractions_i=cp.asfortranarray(kw["fractions_i"]))
    with pytest.raises(ValueError, match="C-contiguous"):
        sfe.seq_frac_set(**kw)
    kw = _device(case, fi_fields=("afrac", "ifrac", "ifrac"))
    with pytest.raises(ValueError, match="each once"):
        sfe.seq_frac_set(**kw)
    kw = _device(case)
    kw.update(dom=cp.ascontiguousarray(kw["dom"][:-1]))
    with pytest.raises(ValueError, match="points"):
        sfe.seq_frac_set(**kw)
    with pytest.raises(TypeError, match="Mapper"):
        sfe.seq_frac_set(**_device(case, mapper_i2a="matrix"))
    # a fraction vector sharing memory with another array
    kw = _device(case)
    kw.update(fractions_o=kw["fractions_i"], fo_fields=sf.FI_FIELDS)
    with pytest.raises(ValueError, match="share memory"):
        sfe.seq_frac_set(**kw)
    kw = _device(case)
    with pytest.raises(ValueError, match="share memory"):
        sfe.seq_frac_set(**dict(kw, fractions_a=kw["i2x"], fa_fields=case["i2x_fields"]))


@requires_gpu
def test_an_array_on_another_device_is_refused(monkeypatch):
    monkeypatch.setattr(sfe, "_device_of", lambda a: 99)
    with pytest.raises(ValueError, match="device 99"):
        sfe.seq_frac_set(**_device(_case("mpas_default")))


def test_the_mapper_takes_arrays_and_keeps_them_read_only():
    rows, cols, w = np.array([2, 1, 2]), np.array([1, 2, 2]), np.array([0.25, 0.5, 0.75])
    a = sfe.Mapper("matrix", rows, cols, w, n_dst=2, n_src=2)
    b = sfe.Mapper("matrix", tuple(rows.tolist()), tuple(cols.tolist()), tuple(w.tolist()), n_dst=2, n_src=2)
    for x, y in zip(a._csr, b._csr):
        assert np.array_equal(x, y)
    rows[0] = 1  # the mapper holds its own copy
    assert a.rows[0] == 2 and not a.rows.flags.writeable and not a.weights.flags.writeable


def _outcome(case):
    try:
        return sf.run_gpu(case)
    except (sfe.FracAbort, sfe.CouplingAbort) as stop:
        return str(stop)


def _caught(case_name):
    """The mutated port differs from the oracle: other words, another
    report, or a stop where E3SM has none (or none where it stops)."""
    outputs, aborts = sf.load_fixture()
    got = _outcome(_case(case_name))
    if case_name in aborts:
        return not isinstance(got, str)
    if isinstance(got, str):
        return True
    want = outputs[case_name]
    return any(list(got[k]) != list(want[k]) if k == "reported" else sf.word_mismatches(got[k], want[k]) > 0
               for k in want)


@requires_gpu
@pytest.mark.parametrize("original, mutated, case", [
    # ifrac not scaled by the domain's fraction
    ("M(i2x[(size_t)i * ni2x + kif], frac)", "i2x[(size_t)i * ni2x + kif]", "mpas_default"),
    # ofrac as 1 - ifrac
    ("S(frac, ifrac)", "S(1.0, ifrac)", "mpas_default"),
    # the copy's fields swapped
    ("dst[(size_t)i * nd + dko] = src[(size_t)i * ns + sko];", "dst[(size_t)i * nd + dko] = src[(size_t)i * ns + ski];",
     "ocean_copy_atm_rearrange"),
    # a row summed backwards
    ("for (int e = row_start[i]; e < row_start[i + 1]; ++e) {",
     "for (int e = row_start[i + 1] - 1; e >= row_start[i]; --e) {", "mpas_default"),
    # the weight dropped
    ("yi = A(yi, M(w[e], x[ski]));", "yi = A(yi, x[ski]);", "ocean_matrix_atm_copy"),
    # the bounds inclusive
    ("(a[k[j]] < lo || a[k[j]] > hi)", "(a[k[j]] <= lo || a[k[j]] >= hi)", "mpas_default"),
    # the bounds not checked
    ("(a[k[j]] < lo || a[k[j]] > hi)", "(0)", "abort_ice_bound"),
    # the sum not checked
    ("if (fabs(S(1.0, sum)) > eps_fracsum) e = 1;", "", "abort_atm_sum"),
])
def test_a_mutated_kernel_is_caught(monkeypatch, original, mutated, case):
    real = sfe.module_source
    assert real().count(original) == 1
    monkeypatch.setattr(sfe, "module_source", lambda: real().replace(original, mutated))
    sfe._module_on.cache_clear()
    try:
        assert _caught(case)
    finally:
        sfe._module_on.cache_clear()


@requires_gpu
@pytest.mark.parametrize("name, value, case", [
    ("EPS_FRACVAL", 0.01, "abort_ice_float32_eps"),
    ("EPS_FRACSUM", 0.01, "abort_atm_sum_float32_eps"),
    ("EPS_FRACVAL", 0.02, "abort_ocn_bound"),
])
def test_a_mutated_constant_is_caught(monkeypatch, name, value, case):
    monkeypatch.setattr(sfe, name, value)
    assert _caught(case)


@requires_gpu
def test_the_file_order_is_not_the_summation_order(monkeypatch):
    """Summing each row in the mapping file's order (not MCT's sorted one)
    changes words."""
    monkeypatch.setattr(sfe, "_sort", lambda rows, cols: np.argsort(rows, kind="stable"))
    assert _caught("mpas_default")


def _oracle_toolchain():
    e3sm, mct = os.environ.get("WOOF_E3SM_V320_SOURCE"), os.environ.get("WOOF_MCT_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not e3sm or not mct or not gfortran or not Path(e3sm, "driver-mct", "main", "seq_frac_mct.F90").is_file():
        return None
    return gfortran, Path(e3sm), Path(mct)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the E3SM coupler oracle build needs gfortran (pixi -e mpas-oracle), "
                           "WOOF_E3SM_V320_SOURCE (an E3SM v3.2.0 tree with driver-mct) and WOOF_MCT_SOURCE "
                           "(MCT at 82b0071)")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, e3sm, mct = _oracle_toolchain()
    env = dict(os.environ, GFORTRAN=gfortran)
    tools = ROOT / "tools" / "e3sm_coupler_oracle"
    subprocess.run(["bash", str(tools / "build.sh"), str(e3sm), str(mct), str(tmp_path / "build")], check=True,
                   env=env, capture_output=True)
    subprocess.run([sys.executable, str(tools / "make_fixture.py"), str(tmp_path / "build"), "--oracle", "frac",
                    "--out", str(tmp_path / "out")], check=True, cwd=ROOT)
    for name in ("oracle-r8.npz", "aborts.json", "inputs.sha256"):
        if name == "oracle-r8.npz":
            with np.load(tmp_path / "out" / name) as got, np.load(sf.FIXTURE_DIR / name) as want:
                assert sorted(got.files) == sorted(want.files)
                for k in want.files:
                    assert got[k].tobytes() == want[k].tobytes(), k
        else:
            assert (tmp_path / "out" / name).read_bytes() == (sf.FIXTURE_DIR / name).read_bytes(), name
