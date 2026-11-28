"""E3SM v3.2.0's ocean albedos (seq_flux_ocnalb_mct), word for word.

The reference is E3SM v3.2.0's own ``seq_flux_ocnalb_mct`` (cut verbatim out
of ``driver-mct/main/seq_flux_mct.F90``) over the pinned MCT on one task,
built with gfortran -O0 -ffp-contract=off and driven by
``tools/e3sm_coupler_oracle/run_flux_ocnalb.F90``. The outputs are committed
(tests/data/oracles/e3sm_ocnalb, PROVENANCE.md). When ``GFORTRAN`` (or
gfortran on PATH), ``WOOF_E3SM_V320_SOURCE`` and ``WOOF_MCT_SOURCE`` are
set, the fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with E3SM only to a tolerance, or only under strict
   numerics: the word-exact gate also runs in fast and strict child
   processes, over xao and the fractions after every call.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: flux_albav, the computed net shortwave
   with the dark side and beside it, signed zeros, nextsw_cday just below
   -0.5, albedos of +0 and -0, and every stop (E3SM's and MCT's).
4. The orbital albedo (glibc's sin, cos and asin) run instead of refused.
5. A gate that cannot fail: mutated kernels must mismatch.
6. The default changed: nothing in woof/ calls this.

QUIRKS (E3SM v3.2.0, seq_flux_mct.F90)
--------------------------------------
- The namelist's albedos are checked with ``< 0``: -0 passes (kept).
- Without flux_albav and the orbital branch, ``update_alb`` stays false:
  ``ifrad``/``ofrad`` are left as they were and never looked up, so a
  fractions vector without them does not stop (kept).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _numerics_child import MODES, run as run_child
from conftest import requires_gpu
import _oracle_toolchains
from woof.hex.coupled import ocnalb_e3sm as oae
from woof.verify import e3sm_ocnalb_oracle as oa

ROOT = Path(__file__).resolve().parents[1]


def _case(name):
    return next(c for c in oa.cases() if c["name"] == name)


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oa.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {case["name"]: oa.input_digest(case) for case in oa.cases()}


def test_fixture_covers_the_branches():
    outputs, aborts = oa.load_fixture()
    cases = {c["name"]: c for c in oa.cases()}
    assert set(outputs) | set(aborts) == set(cases)
    assert set(aborts) == {n for n, c in cases.items() if c["aborts"]}
    run = [c["spec"] for c in cases.values() if not c["aborts"]]
    assert {s["albav"] for s in run} == {True, False}
    assert any(len(s["cdays"]) > 1 for s in run if s["albav"]) and any(len(s["cdays"]) > 2 for s in run)
    assert -0.5 > oa.CDAY_BELOW > -0.5000000000000002
    # flux_albav: the namelist's albedos everywhere, ifrad = ifrac, ofrad = ofrac
    c, o = cases["albav"], outputs["albav"]
    x, f = o["call0/xao"], o["call0/fractions"]
    for name, value in (("So_avsdr", 0.07), ("So_anidr", 0.07), ("So_avsdf", 0.06), ("So_anidf", 0.06)):
        assert (x[:, c["xao_fields"].index(name)] == value).all(), name
    for a, b in (("ifrad", "ifrac"), ("ofrad", "ofrac")):
        assert f[:, c["fr_fields"].index(a)].tobytes() == f[:, c["fr_fields"].index(b)].tobytes()
    # -0 albedos written as -0 (albdir) and +0 (albdif)
    z, cz = outputs["albav_zero_albedos"]["call0/xao"], cases["albav_zero_albedos"]
    assert np.signbit(z[:, cz["xao_fields"].index("So_anidr")]).all()
    assert not np.signbit(z[:, cz["xao_fields"].index("So_anidf")]).any()
    # computed: the dark side 0, beside it not; the fractions untouched
    c, o = cases["computed"], outputs["computed"]
    x = o["call0/xao"]
    dn, up = c["xao_fields"].index("Faox_swdn"), c["xao_fields"].index("Faox_swup")
    assert (x[[5, 9], dn] == 0).all() and (x[[5, 9], up] == 0).all()
    assert (x[[6, 7], dn] > 0).all() and (x[[6, 7], up] < 0).all()
    assert np.signbit(x[2:4, dn]).all()
    for i in range(3):
        assert o[f"call{i}/fractions"].tobytes() == c["calls"][i]["fr"].tobytes()
    assert outputs["computed_without_ifrad"]["call0/fractions"].shape[1] == 4
    assert set(aborts.values()) == {
        "(seq_flux_ocnalb_mct) ERROR seq_flux_mct_inparm namelist not initialized"} | {
        f'MCT::m_AttrVect::indexRA_:: FATAL--attribute not found: "{f}" Traceback:'
        for f in ("So_anidr", "Faox_swup", "Faxa_swvdf", "ofrad")}


def test_the_default_path_does_not_call_the_albedos():
    users = [p for p in (ROOT / "woof").rglob("*.py")
             if "ocnalb_e3sm" in p.read_text(encoding="utf-8")
             and p.name not in ("ocnalb_e3sm.py", "e3sm_ocnalb_oracle.py")
             and p.relative_to(ROOT).as_posix() not in ("woof/hex/coupled/interval_e3sm.py",
                                                    "woof/verify/e3sm_coupler_interval_oracle.py")]
    assert users == []
    siblings = sorted(p.stem for p in (ROOT / "woof" / "hex" / "coupled").glob("*.py")
                      if p.stem not in ("__init__", "ocnalb_e3sm"))
    code = ("import sys, importlib, woof, woof.hex, woof.hex.coupled\n"
            f"for m in {siblings!r}: importlib.import_module('woof.hex.coupled.' + m)\n"
            "assert 'woof.hex.coupled.ocnalb_e3sm' not in sys.modules, 'loaded'\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_the_kernel_has_no_libm():
    src = oae.module_source()
    for name in ("sin(", "cos(", "asin(", "pow(", "exp(", "log("):
        assert name not in src, name


@requires_gpu
def test_gpu_matches_the_e3sm_oracle_word_for_word():
    report = oa.measure()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_e3sm_oracle_word_for_word"], numerics, exact)


_QUOTED = re.compile(r'"([^"]+)"')


def _same_stop(port: str, oracle: str) -> bool:
    """E3SM's line as it is; MCT's: the same missing attribute."""
    if "FATAL" in oracle:
        return "FATAL--attribute not found" in port and _QUOTED.findall(port) == _QUOTED.findall(oracle)
    return port == oracle


@requires_gpu
def test_the_port_stops_where_e3sm_does():
    _, aborts = oa.load_fixture()
    for case in oa.cases():
        if not case["aborts"]:
            continue
        with pytest.raises((oae.AlbedoAbort, oae.CouplingAbort)) as stopped:
            oa.run_gpu(case)
        assert _same_stop(str(stopped.value), aborts[case["name"]]), (case["name"], str(stopped.value))


def _with_cday(case, cday):
    return dict(case, calls=[dict(c, cday=cday) for c in case["calls"]])


@requires_gpu
def test_the_orbital_albedo_is_refused():
    with pytest.raises(oae.AlbedoRefusal, match="orbital.*asin"):
        oa.run_gpu(_with_cday(_case("computed"), -0.5))
    # flux_albav never reaches it
    oa.run_gpu(_with_cday(_case("albav"), 10.0))


@requires_gpu
def test_bad_arrays_are_refused(monkeypatch):
    import cupy as cp
    case = _case("computed")
    c = case["calls"][0]
    kw = dict(a2x=cp.asarray(c["a2x"]), a2x_fields=case["a2x_fields"], fractions=cp.asarray(c["fr"]),
              fractions_fields=case["fr_fields"], xao=cp.asarray(c["xao"]), xao_fields=case["xao_fields"],
              flux_albav=False, albdif=0.06, albdir=0.07, nextsw_cday=-1.0)
    with pytest.raises(TypeError, match="cupy"):
        oae.seq_flux_ocnalb(**dict(kw, xao=c["xao"]))
    with pytest.raises(TypeError, match="bool"):
        oae.seq_flux_ocnalb(**dict(kw, flux_albav=1))
    with pytest.raises(ValueError, match="C-contiguous"):
        oae.seq_flux_ocnalb(**dict(kw, a2x=cp.asfortranarray(kw["a2x"])))
    with pytest.raises(ValueError, match="points"):
        oae.seq_flux_ocnalb(**dict(kw, a2x=kw["a2x"][:-1].copy()))
    with pytest.raises(ValueError, match="share memory"):
        oae.seq_flux_ocnalb(**dict(kw, a2x=kw["xao"], a2x_fields=case["xao_fields"]))
    with pytest.raises(ValueError, match="once"):
        oae.seq_flux_ocnalb(**dict(kw, a2x_fields=case["a2x_fields"][:-1] + case["a2x_fields"][:1]))
    monkeypatch.setattr(oae, "_device_of", lambda a: 99)
    with pytest.raises(ValueError, match="device 99"):
        oae.seq_flux_ocnalb(**kw)


def _caught(case_name):
    outputs, _ = oa.load_fixture()
    got, want = oa.run_gpu(_case(case_name)), outputs[case_name]
    return any(oa.word_mismatches(got[k], want[k]) for k in want)


@requires_gpu
@pytest.mark.parametrize("original, mutated, case", [
    # the dark side's test
    ("if (anidr == 1.0) {", "if (anidr >= 1.0) {", "computed"),
    ("        swupc = 0.0;\n", "", "computed"),
    # the sum's order
    ("A(A(A(a[ka[A_SWNDR]], a[ka[A_SWNDF]]), a[ka[A_SWVDR]]), a[ka[A_SWVDF]])",
     "A(A(a[ka[A_SWNDR]], a[ka[A_SWNDF]]), A(a[ka[A_SWVDR]], a[ka[A_SWVDF]]))", "computed"),
    # the diffuse near-infrared band with the direct albedo
    ("M(a[ka[A_SWNDF]], -anidf)", "M(a[ka[A_SWNDF]], -anidr)", "computed"),
    # swup summed in another order
    ("A(A(A(M(a[ka[A_SWNDR]], -anidr), M(a[ka[A_SWNDF]], -anidf)), M(a[ka[A_SWVDR]], -avsdr)),",
     "A(A(M(a[ka[A_SWNDR]], -anidr), A(M(a[ka[A_SWNDF]], -anidf), M(a[ka[A_SWVDR]], -avsdr))),", "computed"),
    # albdif and albdir swapped
    ("anidr = albdir, avsdr = albdir, anidf = albdif", "anidr = albdif, avsdr = albdir, anidf = albdif", "albav"),
    # ofrad not copied
    ("f[kf[F_OFRAD]] = f[kf[F_OFRAC]];", ";", "albav"),
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


def _oracle_toolchain():
    return _oracle_toolchains.e3sm_v320("driver-mct", "main", "seq_flux_mct.F90")


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
    subprocess.run([sys.executable, str(tools / "make_fixture.py"), str(tmp_path / "build"), "--oracle", "ocnalb",
                    "--out", str(tmp_path / "out")], check=True, cwd=ROOT)
    with np.load(tmp_path / "out" / "oracle-r8.npz") as got, np.load(oa.FIXTURE_DIR / "oracle-r8.npz") as want:
        assert sorted(got.files) == sorted(want.files)
        for k in want.files:
            assert got[k].tobytes() == want[k].tobytes(), k
    for name in ("aborts.json", "inputs.sha256"):
        assert (tmp_path / "out" / name).read_bytes() == (oa.FIXTURE_DIR / name).read_bytes(), name
    # the orbital branch reaches the stand-in (status 9), which the port refuses by name
    src = tmp_path / "orbital.bin"
    src.write_bytes(oa.input_bytes(_with_cday(_case("computed"), -0.5)))
    done = subprocess.run([str(tmp_path / "build" / "run_flux_ocnalb"), str(src), str(tmp_path / "orbital.out")],
                          capture_output=True, text=True)
    assert done.returncode == 9, done.stderr
