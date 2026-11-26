"""The Kain-Fritsch capped-downdraft rain floor (quirk pass, correct-mode FQ11).

THE QUIRK
---------
WRF's Kain-Fritsch family caps the downdraft when it would evaporate more
than the updraft rains: DDINC = TRPPT/TDER, TDER = TDER*DDINC, then
PPTFLX = TRPPT - TDER with no floor (module_cu_kfeta.F:1810-1821,
module_cu_kfcup.F:2711-2721, module_cu_mskf.F:5840 of WRF 4.7.1; the same
lines in 4.8.1).  The float64 answer is 0, but in binary32 TDER*DDINC can
round above TRPPT, and the flux is then a few ulp negative: RAINCV is
negative for the whole NCA window and RAINC accumulates below zero.
RunConfig.kf_capped_downdraft_rain_floor floors the flux at zero.

WHAT THIS FILE HOLDS
--------------------
* The key is correct-mode finding FQ11: on by default, off in exact mode,
  read only under cu_physics = 1 (kf.cu), 10 (kfcup.cu) and 11 (mskf.cu).
* Exact mode is WRF's program: each unit carries the floor only inside one
  ``#ifdef KF_CAPPED_DOWNDRAFT_RAIN_FLOOR`` block after WRF's bare
  subtraction, and each unit with the block removed is byte-identical to
  its source with the earlier unconditional floor taken out (kfcup.cu: the
  file at a70e40753, before the floor; mskf.cu and kf.cu: the files at
  a70e40753 with WRF's ``PPTFLX = TRPPT - TDER`` back in place of the
  floored statement, as before 695ddf87a and d547dc8c3).
* Every launcher compiles the define exactly when the key reaches the run;
  a direct launcher call is WRF's arithmetic unless it asks.

The device columns are in tests/test_kfcup_capped_downdraft_gpu.py (KF-CuP)
and tests/test_kf.py (the seven WRF-graded KF-eta capped columns).
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from woof import correct_mode
from woof.config import RunConfig, validate_run_config

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "woof" / "core" / "kernels"
KEY = "kf_capped_downdraft_rain_floor"
DEFINE = "KF_CAPPED_DOWNDRAFT_RAIN_FLOOR"

#: Each unit: (WRF's bare statement, the floor in the block, sha256 of the
#: unit with the block removed).
UNITS = {
    "kf.cu": ("pptflx = trppt - tder;", "pptflx = fmaxf(pptflx, 0.0f);",
              "cec65b1c6d537c0960b910283e49623b2ff1fce12c7776ccad38d6787b168093"),
    "kfcup.cu": ("PPTFLX = TRPPT - TDER;", "PPTFLX = fmaxf(PPTFLX, 0.0f);",
                 "2a77ac6f1bbaf0f3704c7a4d6813774cfe7b19525649c528cdae21134d702dae"),
    "mskf.cu": ("PPTFLX = TRPPT - TDER;", "PPTFLX = fmaxf(PPTFLX, 0.0f);",
                "4881ebc1510695b0dd8de0414403a974c359af53ec612839c632c663a4bf9df1"),
}


def _split_blocks(text):
    """(block bodies, text with the blocks removed)."""
    bodies, kept, current = [], [], None
    for line in text.splitlines(keepends=True):
        word = line.strip()
        if current is None:
            if word == f"#ifdef {DEFINE}":
                current = []
            else:
                kept.append(line)
            continue
        if word.startswith("#endif"):
            bodies.append("".join(current))
            current = None
            continue
        assert not word.startswith("#"), "the block holds no preprocessor arm"
        current.append(line)
    assert current is None, "unterminated floor block"
    return bodies, "".join(kept)


def _run(**kw):
    kw.setdefault("moist", True)
    return validate_run_config(RunConfig(
        nx=8, ny=8, nz=12, dx=3000.0, dy=3000.0, ztop=12000.0, dt=10.0,
        run_seconds=60.0, **kw))


def test_the_key_is_correct_mode_finding_fq11():
    assert correct_mode.FINDING_BY_KEY[KEY] == "FQ11"
    assert correct_mode.KEY_SCOPE[KEY] == ("cu_physics", (1, 10, 11))
    assert RunConfig.__dataclass_fields__[KEY].default is correct_mode.CORRECT_KEYS_DEFAULT
    assert correct_mode.resolve("exact").settings[KEY] is False
    assert correct_mode.resolve("correct").settings[KEY] is True


@pytest.mark.parametrize("cu", [1, 10, 11])
def test_the_launchers_read_it_where_the_kf_family_runs(cu):
    from woof.core.kf import capped_downdraft_rain_floor_due
    extra = dict(bl_pbl_physics=1, sf_sfclay_physics=1) if cu == 10 else {}
    assert capped_downdraft_rain_floor_due(_run(cu_physics=cu, **extra)) \
        is correct_mode.CORRECT_KEYS_DEFAULT
    assert capped_downdraft_rain_floor_due(_run(cu_physics=cu, **extra, **{KEY: True})) is True
    assert capped_downdraft_rain_floor_due(_run(cu_physics=cu, **extra, **{KEY: False})) is False
    exact = _run(cu_physics=cu, **extra, **correct_mode.resolve("exact").settings)
    assert capped_downdraft_rain_floor_due(exact) is False


@pytest.mark.parametrize("cu", [0, 99])
def test_it_is_inert_where_the_kf_family_does_not_run(cu):
    """The old KF (99, module_cu_kf.F) forms no capped TRPPT - TDER."""
    from woof.core.kf import capped_downdraft_rain_floor_due
    assert capped_downdraft_rain_floor_due(_run(cu_physics=cu, **{KEY: True})) is False


@pytest.mark.parametrize("unit", sorted(UNITS))
def test_without_the_define_every_unit_is_wrfs_subtraction(unit):
    text = (KERNELS / unit).read_text(encoding="utf-8")
    bare, floor, digest = UNITS[unit]
    bodies, stripped = _split_blocks(text)
    assert len(bodies) == 1, unit
    # The block floors the flux WRF's statement just formed, and nothing else.
    statements = [line.strip() for line in bodies[0].splitlines()
                  if line.strip() and not line.strip().startswith(("/*", "*"))]
    assert statements == [floor], (unit, statements)
    assert text.count(bare) == 1, unit
    head = text[:text.index(f"#ifdef {DEFINE}")]
    assert head.rstrip().endswith(bare), unit
    assert "fmaxf(trppt - tder" not in stripped.lower(), unit
    assert hashlib.sha256(stripped.encode("utf-8")).hexdigest() == digest, unit


def test_the_default_compile_never_sets_the_define():
    from woof.core.kf import CAPPED_DOWNDRAFT_RAIN_FLOOR_DEFINES
    assert CAPPED_DOWNDRAFT_RAIN_FLOOR_DEFINES == ((DEFINE, 1),)
    from woof.core import kf, mskf
    # A direct launcher call is WRF's arithmetic unless it asks.
    for launcher in (kf.launch_kf, kf.launch_kfcup_columns, mskf.launch_mskf):
        assert inspect.signature(launcher).parameters["rain_floor"].default is False


def test_every_production_adapter_passes_the_key():
    from woof.core import column_cumulus, kf
    for source in (inspect.getsource(kf.KainFritsch.__call__),
                   inspect.getsource(kf.KFCumulusPotential.__call__),
                   inspect.getsource(column_cumulus.MskfColumnCumulus.__call__)):
        assert "rain_floor=capped_downdraft_rain_floor_due(cfg)" in source

