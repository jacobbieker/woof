"""E3SM v3.2.0's atmosphere-ocean flux (seq_flux_atmocn_mct), word for word.

The reference is E3SM v3.2.0's own ``seq_flux_init_mct`` and
``seq_flux_atmocn_mct`` (cut verbatim out of
``driver-mct/main/seq_flux_mct.F90``) over ``shr_flux_mod`` (compiled
verbatim) and the pinned MCT on one task, built with gfortran -O0
-ffp-contract=off and driven by ``tools/e3sm_coupler_oracle/run_flux_atmocn.F90``.
The outputs are committed (tests/data/oracles/e3sm_flux_atmocn,
PROVENANCE.md). When ``GFORTRAN`` (or gfortran on PATH),
``WOOF_E3SM_V320_SOURCE`` and ``WOOF_MCT_SOURCE`` are set, the fixture
itself is rebuilt and compared.

``psimhu`` calls glibc's binary64 ``atan`` (LGPL): the port runs only with
the opt-in transcription (``WOOF_GLIBC239_LIBM64=1``, the ``glibc239``
fixture here) and is refused by name without it.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with E3SM only to a tolerance, or only under strict
   numerics: the word-exact gate also runs in fast and strict child
   processes, over every xao field.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: the explicit and implicit stress,
   gustiness, the cold-air outbreak, a convergence tolerance, one
   iteration, the minimum wind, dead models, the mask and emask on both
   sides of their bounds, calm, stable, unstable and neutral points, and
   every stop (E3SM's and MCT's).
4. The default path running LGPL code: without the opt-in the port refuses
   by name and the source holds no kernel.
5. A gate that cannot fail: mutated kernels, constants and switches must
   mismatch or change the outcome.
6. The default changed: nothing in woof/ calls this.

QUIRKS (E3SM v3.2.0, seq_flux_mct.F90 / shr_flux_mod.F90)
---------------------------------------------------------
- emask compares the density with the default-real literal ``1.0e-12``:
  float32(1e-12) widened (kept; the case's point 51 sits between that and
  a double 1e-12).
- With dead models seq_flux_atmocn_mct sets the module's mask to 1 but
  passes emask as it was: points the flux grid's mask dropped are written
  with shr_const_spval (kept).
- A ``flux_convergence`` of 1 or more stops the coupler: the iteration's
  first test, |(u* - 2u*)/u*| = 1, is not above it, so no point iterates
  and shr_flux_atmOcn stops with "No iterations performed" (kept).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _numerics_child import MODES, run as run_child
from conftest import requires_gpu
from woof.hex.coupled import flux_atmocn_e3sm as fxe
from woof.verify import e3sm_flux_atmocn_oracle as fx

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def glibc239(monkeypatch):
    """The opt-in glibc 2.39 binary64 atan transcription (LGPL, source checkout only)."""
    monkeypatch.setenv("WOOF_GLIBC239_LIBM64", "1")
    fxe._module_on.cache_clear()
    yield
    fxe._module_on.cache_clear()


def _case(name):
    return next(c for c in fx.cases() if c["name"] == name)


def _col(case, field):
    return case["xao_fields"].index(field)


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (fx.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {case["name"]: fx.input_digest(case) for case in fx.cases()}


def test_fixture_covers_the_branches():
    outputs, aborts = fx.load_fixture()
    cases = {c["name"]: c for c in fx.cases()}
    assert set(outputs) | set(aborts) == set(cases)
    assert set(aborts) == {n for n, c in cases.items() if c["aborts"]}
    run = [c["spec"] for c in cases.values() if not c["aborts"]]
    for key in ("dead", "implicit", "gust", "coldair"):
        assert {s[key] for s in run} == {True, False}, key
    assert {s["maxit"] for s in run} >= {-1, 1, 3, 10} and {s["tol"] > 0 for s in run} == {True, False}
    d = outputs["explicit_default"]["xao"]
    c = cases["explicit_default"]
    sen = _col(c, "Faox_sen")
    # the mask: the domain's below 0.5, no ocean or ice (+0 + -0 too) leave xao as it was
    for k in (3, 17, 22, 5, 30, 31):
        assert d[k].tobytes() == c["xao"][k].tobytes(), k
    # kept: a domain mask of exactly 0.5 and a tiny fraction
    for k in (21, 32):
        assert d[k, sen] != c["xao"][k, sen] and d[k, sen] != fxe.SHR_CONST_SPVAL, k
    # emask: a density of 0 and an ocean below 1 K get spval; float32(1e-12) < 51's density, 1 K kept
    for k in (50, 52):
        assert d[k, sen] == fxe.SHR_CONST_SPVAL, k
    for k in (51, 53):
        assert d[k, sen] != fxe.SHR_CONST_SPVAL, k
    assert fx.DENS_LIMIT < fx.DENS_BETWEEN < 1.0e-12
    # dead models: every point written, the diurnal fields 0, the dropped ones spval
    dead, cd = outputs["dead_models"]["xao"], cases["dead_models"]
    assert (dead[:, _col(cd, "So_warm_diurn")] == 0).all()
    assert dead[3, _col(cd, "Faox_sen")] == fxe.SHR_CONST_SPVAL
    # the stops: E3SM's (each) and MCT's
    assert {line for line in aborts.values() if "FATAL" not in line} == {
        "No iterations performed ERROR in shr_flux_mod.F90 at line 476",
        "(seq_flux_atmocn_mct) ERROR nloc sizes do not match",
        "(shr_flux_atmOcn)  subroutine shr_flux_atmOcn requires ocn_surface_flux_scheme = 0 or 1",
        "(seq_flux_atmocn_mct) ERROR cannot use flux_diurnal with UA flux scheme"}
    assert sum("FATAL" in line for line in aborts.values()) == 3


def test_the_constants_are_e3sms():
    assert fxe.folded_al2() == 1.6094379124341003
    assert fxe.SHR_CONST_STEBOL == 5.670374419e-8
    assert fxe.FluxOptions().max_iteration() == 2
    assert fxe.FluxOptions(atm_flux_method="implicit_stress").max_iteration() == 30


def test_the_default_path_does_not_call_the_flux():
    users = [p for p in (ROOT / "woof").rglob("*.py")
             if "flux_atmocn_e3sm" in p.read_text(encoding="utf-8")
             and p.name not in ("flux_atmocn_e3sm.py", "e3sm_flux_atmocn_oracle.py")]
    assert users == []
    siblings = sorted(p.stem for p in (ROOT / "woof" / "hex" / "coupled").glob("*.py")
                      if p.stem not in ("__init__", "flux_atmocn_e3sm"))
    code = ("import sys, importlib, woof, woof.hex, woof.hex.coupled\n"
            f"for m in {siblings!r}: importlib.import_module('woof.hex.coupled.' + m)\n"
            "assert 'woof.hex.coupled.flux_atmocn_e3sm' not in sys.modules, 'loaded'\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_without_the_opt_in_no_lgpl_code_is_compiled(monkeypatch):
    monkeypatch.delenv("WOOF_GLIBC239_LIBM64", raising=False)
    src = fxe.module_source()
    assert "g239_atan(" not in src.split("#ifdef WOOF_GLIBC239_TRIG64")[0]
    assert "#define WOOF_GLIBC239_TRIG64" not in src


@requires_gpu
def test_the_default_refuses_by_name(monkeypatch):
    monkeypatch.delenv("WOOF_GLIBC239_LIBM64", raising=False)
    fxe._module_on.cache_clear()
    with pytest.raises(fxe.FluxRefusal, match="atan.*WOOF_GLIBC239_LIBM64=1"):
        fx.run_gpu(_case("explicit_default"))


@requires_gpu
@pytest.mark.usefixtures("glibc239")
def test_gpu_matches_the_e3sm_oracle_word_for_word():
    report = fx.measure()
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_e3sm_oracle_word_for_word"], numerics, exact,
              env={"WOOF_GLIBC239_LIBM64": "1"})


_QUOTED = re.compile(r'"([^"]+)"')


def _same_stop(port: str, oracle: str) -> bool:
    """E3SM's line as it is; MCT's: the same missing attribute."""
    if "FATAL" in oracle:
        return "FATAL--attribute not found" in port and _QUOTED.findall(port) == _QUOTED.findall(oracle)
    return port == oracle


@requires_gpu
def test_the_port_stops_where_e3sm_does(glibc239):
    import cupy as cp
    _, aborts = fx.load_fixture()
    for case in fx.cases():
        if not case["aborts"]:
            continue
        with pytest.raises((fxe.FluxAbort, fxe.CouplingAbort)) as stopped:
            fx.run_gpu(case)
        assert _same_stop(str(stopped.value), aborts[case["name"]]), (case["name"], str(stopped.value))
    # a stop leaves xao and the mask as they were
    case = _case("abort_no_iteration")
    dev = {k: cp.asarray(case[k]) for k in ("dom", "fractions", "a2x", "o2x", "xao")}
    mask = fxe.flux_mask(dom=dev["dom"], dom_fields=case["dom_fields"], fractions=dev["fractions"],
                         fractions_fields=fx.FR_FIELDS)
    before = mask.copy()
    with pytest.raises(fxe.FluxAbort):
        fxe.seq_flux_atmocn(a2x=dev["a2x"], a2x_fields=case["a2x_fields"], o2x=dev["o2x"],
                            o2x_fields=case["o2x_fields"], xao=dev["xao"], xao_fields=case["xao_fields"], mask=mask,
                            options=fx.options_for(case))
    assert cp.asnumpy(dev["xao"]).tobytes() == case["xao"].tobytes()
    assert (mask == before).all()


def _with(case, **spec):
    return dict(case, spec=dict(case["spec"], **spec))


@requires_gpu
@pytest.mark.parametrize("change, match", [
    (dict(scheme=1), "COARE"),
    (dict(scheme=2), "UA"),
    (dict(diurnal=True), "shr_flux_atmOcn_diurnal"),
])
def test_what_is_not_ported_is_refused(glibc239, change, match):
    with pytest.raises(fxe.FluxRefusal, match=match):
        fx.run_gpu(_with(_case("explicit_default"), **change))


@requires_gpu
def test_waves_isotopes_and_bad_arrays_are_refused(glibc239, monkeypatch):
    import cupy as cp
    case = _case("explicit_default")
    real = fx.options_for
    monkeypatch.setattr(fx, "options_for", lambda c: fxe.FluxOptions(**{**real(c).__dict__, "wav_atm_coup": "twoway"}))
    with pytest.raises(fxe.FluxRefusal, match="wave"):
        fx.run_gpu(case)
    monkeypatch.setattr(fx, "options_for", real)
    iso = dict(case, xao_fields=case["xao_fields"] + ("Faox_evap_HDO",),
               xao=np.column_stack([case["xao"], np.zeros(case["spec"]["n"])]))
    with pytest.raises(fxe.FluxRefusal, match="isotopes"):
        fx.run_gpu(iso)
    a2x = cp.asarray(case["a2x"])
    mask = cp.ones(case["spec"]["n"], dtype=np.int32)
    kw = dict(a2x=a2x, a2x_fields=case["a2x_fields"], o2x=cp.asarray(case["o2x"]), o2x_fields=case["o2x_fields"],
              xao=cp.asarray(case["xao"]), xao_fields=case["xao_fields"], mask=mask, options=fxe.FluxOptions())
    with pytest.raises(TypeError, match="cupy"):
        fxe.seq_flux_atmocn(**dict(kw, xao=case["xao"]))
    with pytest.raises(ValueError, match="C-contiguous"):
        fxe.seq_flux_atmocn(**dict(kw, a2x=cp.asfortranarray(a2x)))
    with pytest.raises(ValueError, match="share memory"):
        fxe.seq_flux_atmocn(**dict(kw, xao=a2x, xao_fields=case["a2x_fields"]))
    with pytest.raises(ValueError, match="int32"):
        fxe.seq_flux_atmocn(**dict(kw, mask=mask.astype(np.int64)))
    with pytest.raises(ValueError, match="atm_flux_method"):
        fxe.seq_flux_atmocn(**dict(kw, options=fxe.FluxOptions(atm_flux_method="implicit")))
    monkeypatch.setattr(fxe, "_device_of", lambda a: 99)
    with pytest.raises(ValueError, match="device 99"):
        fx.run_gpu(case)


def _outcome(case):
    try:
        return fx.run_gpu(case)
    except (fxe.FluxAbort, fxe.CouplingAbort) as stop:
        return str(stop)


def _caught(case_name):
    outputs, aborts = fx.load_fixture()
    got = _outcome(_case(case_name))
    if case_name in aborts:
        return not isinstance(got, str)
    if isinstance(got, str):
        return True
    return fx.word_mismatches(got["xao"], outputs[case_name]["xao"]) > 0


@requires_gpu
@pytest.mark.parametrize("original, mutated, case", [
    # emask's density limit a double
    ("rbot < (r8)1.0e-12f", "rbot < 1.0e-12", "explicit_default"),
    # emask's temperature inclusive
    ("ts < 1.0) emask = 0", "ts <= 1.0) emask = 0", "explicit_default"),
    # the domain's mask inclusive, the fractions' strict
    ("dom[(size_t)i * ndom + kmask] < 0.5", "dom[(size_t)i * ndom + kmask] <= 0.5", "explicit_default"),
    ("fr[(size_t)i * nfr + ki]) <= 0.0", "fr[(size_t)i * nfr + ki]) < 0.0", "explicit_default"),
    # psimhu's pi/2 exact
    ("M(2.0, g239::g239_atan(xd))), 1.571)", "M(2.0, g239::g239_atan(xd))), 1.5707963267948966)",
     "explicit_default"),
    # |z/L| clamped at 20
    ("min_first(fabs(hol), 10.0)", "min_first(fabs(hol), 20.0)", "explicit_default"),
    # one iteration more
    ("&& iter < c.max_iter)", "&& iter <= c.max_iter)", "converges_early"),
    # the stress's damping
    ("? 0.6 : 0.95", "? 0.95 : 0.95", "implicit_stress"),
    # the gust not added
    ("sqrt(A(M(v, v), M(ugust, ugust)))", "v", "gustiness"),
    # the cold-air scaling
    ("#define ALPHA 1.4", "#define ALPHA 1.5", "coldair_outbreak"),
    # the dead models' diurnal fields kept
    ("x[kdiurnal[k]] = 0.0;", ";", "dead_models"),
    # the dead models' mask not set
    ("        mask[i] = 1;\n", "", "dead_models"),
    # u10withgusts from u10
    ("x[kx[X_U10G]] = sqrt(w[W_DUU10N]);", "x[kx[X_U10G]] = w[W_U10RES];", "gustiness"),
    # the minimum wind not applied
    ("return max_first(minwind, v);", "return v;", "high_minwind"),
])
def test_a_mutated_kernel_is_caught(glibc239, monkeypatch, original, mutated, case):
    real = fxe.module_source
    assert real().count(original) == 1, original
    monkeypatch.setattr(fxe, "module_source", lambda: real().replace(original, mutated))
    fxe._module_on.cache_clear()
    try:
        assert _caught(case)
    finally:
        fxe._module_on.cache_clear()


@requires_gpu
@pytest.mark.parametrize("name, value, case", [
    ("SHR_CONST_STEBOL", 5.67e-8, "explicit_default"),
    ("SHR_CONST_KARMAN", 0.41, "explicit_default"),
    ("SHR_CONST_ZVIR", 0.61, "explicit_default"),
])
def test_a_mutated_constant_is_caught(glibc239, monkeypatch, name, value, case):
    monkeypatch.setattr(fxe, name, value)
    assert _caught(case)


@requires_gpu
def test_the_iteration_count_default_is_caught(glibc239, monkeypatch):
    monkeypatch.setattr(fxe.FluxOptions, "max_iteration", lambda self: 30)
    assert _caught("explicit_default")


def _oracle_toolchain():
    e3sm, mct = os.environ.get("WOOF_E3SM_V320_SOURCE"), os.environ.get("WOOF_MCT_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not e3sm or not mct or not gfortran or not Path(e3sm, "share", "util", "shr_flux_mod.F90").is_file():
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
    subprocess.run([sys.executable, str(tools / "make_fixture.py"), str(tmp_path / "build"), "--oracle", "flux",
                    "--out", str(tmp_path / "out")], check=True, cwd=ROOT)
    for name in ("oracle-r8.npz", "aborts.json", "inputs.sha256"):
        if name == "oracle-r8.npz":
            with np.load(tmp_path / "out" / name) as got, np.load(fx.FIXTURE_DIR / name) as want:
                assert sorted(got.files) == sorted(want.files)
                for k in want.files:
                    assert got[k].tobytes() == want[k].tobytes(), k
        else:
            assert (tmp_path / "out" / name).read_bytes() == (fx.FIXTURE_DIR / name).read_bytes(), name
