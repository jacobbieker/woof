"""The opt-in MYNN surface-layer 2 m / 10 m heat profile from zt/L.

THE WRF QUIRK.  WRF 4.7.1's MYNN surface layer (``module_sf_mynn.F``)
pairs the 2 m heat log term ``gz2ozt = ln((2+z0)/zt)`` (and the first-level
``psih``, integrated from ``zt/L``) with a stability correction integrated
from the momentum roughness: ``psih2 = psih(zol2) - psih(z0/L)`` and the same
for ``psih10`` (:826-827, :839-840 stable; :910-911, :921-922 unstable).
The 2 m and first-level profiles therefore start at different lower limits,
and T2/Q2 come from neither.

THE OPT-IN.  ``mynn_sfclay_psih_zolzt = true`` compiles kernels/
mynn_surface.cu with ``MYNN_SL_PSIH_ZOLZT``: psih2/psih10 start at zt/L.
Only the 2 m / 10 m diagnostics (TH2, T2, Q2, CHS2, CQS2, CK, CKA) can
move.  Off (the default) the unit is WRF's word for word.

WHAT THIS FILE PINS.
* The default program: the source without the opt-in blocks is byte for
  byte the unit pinned before the key existed.
* Configuration: off by default, MYNN surface layer and wrf_461 only.
* The corrected CPU reference: nothing but the 2 m / 10 m diagnostics move,
  and they move with the sign the profile requires (stable: T2 nearer the
  air, unstable: nearer the skin), with T2 between skin and air.
* The GPU: the default launch is unchanged and the corrected launch agrees
  with the corrected reference.
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

REPO = Path(__file__).resolve().parents[1]
KERNEL = REPO / "woof" / "core" / "kernels" / "mynn_surface.cu"
ORACLE = REPO / "tests" / "data" / "oracles" / "mynn" / "surface-layer.csv"

#: kernels/mynn_surface.cu as pinned before the opt-in existed
#: (tests/test_mp8_frozen.py, previously a02979c2).
PRE_OPT_IN_SHA256 = "a02979c2ef27ce361f092675cef0d55705be6ff3f298a04b7e969b1c9694bdd9"

_BLOCK = re.compile(r"#ifdef MYNN_SL_PSIH_ZOLZT\n.*?#endif\n", re.S)
#: the later FQ10 block is not part of the unit pinned before this key
_Q2_BLOCK = re.compile(r"#ifdef MYNN_SL_Q2_LAND_CAP\n.*?#endif\n", re.S)

#: Outputs psih2/psih10 reach: the 2 m / 10 m diagnostics only.
MOVED = frozenset({"th2", "t2", "q2", "chs2", "cqs2", "ck", "cka"})

ALIASES = {"hfx": "hfx_input", "qfx": "qfx_input", "znt": "znt_input",
           "qsfc": "qsfc_input", "ust": "ust_input"}
NAMES = ("u1", "v1", "t1", "qv1", "p1", "rho1", "dz1", "u2", "v2", "dz2",
         "psfc", "tsk", "pblh", "mavail", "hfx", "qfx", "znt", "qsfc", "ust",
         "xland", "snowh")


def _cases():
    with ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    inputs = {name: np.asarray([float(row[ALIASES.get(name, name)]) for row in rows],
                               np.float32) for name in NAMES}
    return [row["case"] for row in rows], inputs


def test_the_default_program_is_the_pre_opt_in_unit():
    source = KERNEL.read_text(encoding="utf-8")
    blocks = _BLOCK.findall(source)
    assert len(blocks) == 2
    assert all("zolzt);" in block for block in blocks)
    stripped = _BLOCK.sub("", _Q2_BLOCK.sub("", source))
    assert hashlib.sha256(stripped.encode("utf-8")).hexdigest() == PRE_OPT_IN_SHA256


def _base(**kw):
    from woof.config import RunConfig
    return RunConfig(nx=8, ny=8, nz=12, dx=3000.0, dy=3000.0, ztop=12000.0,
                     dt=10.0, run_seconds=60.0, **kw)


def test_the_key_follows_the_mode_and_is_read_by_mynn_wrf461_only():
    from woof import correct_mode
    from woof.config import ConfigurationRefused, RunConfig, validate_run_config
    from woof.quirk_corrections import run_reads
    field = RunConfig.__dataclass_fields__["mynn_sfclay_psih_zolzt"]
    # correct-mode finding FQ3: on in a default process, off in exact mode
    assert field.default is correct_mode.CORRECT_KEYS_DEFAULT and field.kw_only
    assert correct_mode.resolve("exact").settings["mynn_sfclay_psih_zolzt"] is False
    assert run_reads(validate_run_config(_base(bl_pbl_physics=5, sf_sfclay_physics=5,
                                               mynn_sfclay_psih_zolzt=True)), "mynn_sfclay_psih_zolzt")
    assert not run_reads(validate_run_config(_base(bl_pbl_physics=1, sf_sfclay_physics=1,
                                                   mynn_sfclay_psih_zolzt=True)), "mynn_sfclay_psih_zolzt")
    # the gsl_wrf39 fork integrates from z0/L throughout: nothing to correct
    gsl = validate_run_config(_base(bl_pbl_physics=5, sf_sfclay_physics=5,
                                    mynn_sfclay_variant="gsl_wrf39",
                                    mynn_sfclay_psih_zolzt=True))
    assert not run_reads(gsl, "mynn_sfclay_psih_zolzt")
    assert "FQ3" not in correct_mode.uncorrected(gsl)
    with pytest.raises(ConfigurationRefused, match="true or false"):
        validate_run_config(_base(bl_pbl_physics=5, sf_sfclay_physics=5,
                                  mynn_sfclay_psih_zolzt=1))


def test_the_corrected_reference_moves_only_the_2m_profile_with_its_sign():
    from woof.core.mynn_surface import ROVCP, mynn_surface_layer_default
    cases, inputs = _cases()
    default = mynn_surface_layer_default(inputs)
    named_off = mynn_surface_layer_default(inputs, psih_zolzt=False)
    corrected = mynn_surface_layer_default(inputs, psih_zolzt=True)
    for name, value in default.items():
        assert value.tobytes() == named_off[name].tobytes(), name
        if name not in MOVED:
            assert value.tobytes() == corrected[name].tobytes(), name
    regime = default["regime"]
    neutral = regime == 3.0
    np.testing.assert_array_equal(corrected["th2"][neutral], default["th2"][neutral])
    thgb = (inputs["tsk"].astype(np.float64)
            * (100.0 / (inputs["psfc"].astype(np.float64) / 1000.0)) ** float(ROVCP))
    th1 = (inputs["t1"].astype(np.float64)
           * (100.0 / (inputs["p1"].astype(np.float64) / 1000.0)) ** float(ROVCP))
    za = 0.5 * inputs["dz1"].astype(np.float64)
    fallback = thgb + 2.0 * (th1 - thgb) / za
    moved = 0
    for i, case in enumerate(cases):
        if neutral[i]:
            continue
        lo, hi = sorted((thgb[i], th1[i]))
        th2 = float(corrected["th2"][i])
        # Between skin and air, by the profile (not the 2/za fallback).
        assert lo - 1e-3 <= th2 <= hi + 1e-3, case
        assert abs(th2 - fallback[i]) > 1e-4, case
        frac_default = (float(default["th2"][i]) - thgb[i]) / (th1[i] - thgb[i])
        frac_corrected = (th2 - thgb[i]) / (th1[i] - thgb[i])
        if frac_corrected == frac_default:
            continue
        moved += 1
        # zt < z0: stable, psih(zt/L) > psih(z0/L) so psit2 grows and T2
        # moves toward the air; unstable, the reverse.
        if regime[i] in (1.0, 2.0):
            assert frac_corrected > frac_default, case
        else:
            assert frac_corrected < frac_default, case
    assert moved >= 2


def _gpu(inputs, **kw):
    import cupy as cp
    from woof.core.mynn_sfclay import mynn_surface_layer
    values = {name: cp.asarray(value.reshape(1, -1)) for name, value in inputs.items()}
    result = mynn_surface_layer(values, **kw)
    return {name: cp.asnumpy(value).ravel() for name, value in vars(result).items()}


@pytest.mark.gpu
@requires_gpu
def test_the_gpu_default_is_unchanged_and_the_correction_matches_the_reference():
    from woof.core.mynn_surface import mynn_surface_layer_default
    _, inputs = _cases()
    default = _gpu(inputs)
    named_off = _gpu(inputs, psih_zolzt=False)
    corrected = _gpu(inputs, psih_zolzt=True)
    reference = mynn_surface_layer_default(inputs, psih_zolzt=True)
    for name, value in default.items():
        assert value.tobytes() == named_off[name].tobytes(), name
        if name not in MOVED:
            assert value.tobytes() == corrected[name].tobytes(), name
    for name in ("th2", "t2", "q2", "chs2", "cqs2"):
        np.testing.assert_allclose(corrected[name], reference[name], rtol=2e-5, err_msg=name)
    assert default["th2"].tobytes() != corrected["th2"].tobytes()


def test_the_gsl_variant_refuses_the_correction():
    from woof.core.mynn_sfclay import mynn_surface_defines
    assert mynn_surface_defines("wrf_461", False) == ()
    assert mynn_surface_defines("wrf_461", True) == (("MYNN_SL_PSIH_ZOLZT", 1),)
    with pytest.raises(ValueError, match="gsl_wrf39"):
        mynn_surface_defines("gsl_wrf39", True)
    # the reference refuses what the device path refuses
    from woof.core.mynn_surface import mynn_surface_layer_default
    _, inputs = _cases()
    with pytest.raises(ValueError, match="gsl_wrf39"):
        mynn_surface_layer_default(inputs, variant="gsl_wrf39", psih_zolzt=True)


def test_the_umbrella_turns_it_on_for_the_wrf461_surface_layer():
    from woof.config import ConfigurationRefused, validate_run_config
    from woof.quirk_corrections import apply_run_corrections
    kw = apply_run_corrections(dict(woof_quirk_corrections="all", bl_pbl_physics=5, sf_sfclay_physics=5))
    assert kw["mynn_sfclay_psih_zolzt"] is True
    assert validate_run_config(_base(**kw)).mynn_sfclay_psih_zolzt is True
    # the gsl_wrf39 fork has nothing to correct: the key is not read there
    from woof.quirk_corrections import run_reads
    kw = apply_run_corrections(dict(woof_quirk_corrections="all", bl_pbl_physics=5, sf_sfclay_physics=5,
                                    mynn_sfclay_variant="gsl_wrf39"))
    assert not run_reads(validate_run_config(_base(**kw)), "mynn_sfclay_psih_zolzt")
    with pytest.raises(ConfigurationRefused, match="mynn_sfclay_psih_zolzt"):
        apply_run_corrections(dict(woof_quirk_corrections="all", bl_pbl_physics=5, sf_sfclay_physics=5,
                                   mynn_sfclay_psih_zolzt=False))
