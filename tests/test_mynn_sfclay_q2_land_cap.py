"""Correct-mode FQ10: the MYNN surface layer caps Q2 over land only.

THE WRF QUIRK.  WRF 4.7.1's ``module_sf_mynn.F`` ends its 2 m diagnostics
with ``Q2(I)= MIN(Q2(I), 1.05*QV1D(I))`` (:1149), at every point, water
included.  ``module_surface_driver.F:4452-4456`` applies the same limit
after the schemes, and only where ``XLAND < 1.5``: its comment gives the
reason ("Q2 diagnostic ... because calculation is based on surface flux only
... transition periods and weak winds and vegetation source"), which is a
land problem (the LSM owns land Q2).  Over water the 2 m mixing ratio is the
Monin-Obukhov interpolation between the saturated surface and the lowest
level, and the cap truncates it: over a warm sea under cooler, drier air Q2
cannot exceed 5 per cent of the air value however moist the surface is.

THE CORRECTION.  ``mynn_sfclay_q2_land_cap = true`` compiles
kernels/mynn_surface.cu with ``MYNN_SL_Q2_LAND_CAP``: the cap runs where
``xland < 1.5`` only.  Q2 is a diagnostic: nothing else the kernel writes
moves.  Off (exact mode) the unit is WRF's word for word.

WHAT THIS FILE PINS: the default program (both opt-in blocks stripped is the
unit pinned before FQ3), the configuration (off in exact mode, read at
sf_sfclay_physics = 5 / wrf_461), the corrected reference (a water column
whose Q2 exceeded 1.05*qv1 before the cap now keeps it, between the surface
and the air value; a land column is unchanged; only Q2 moves) and the GPU
kernel against that reference.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mynn_sfclay_psih_zolzt import (KERNEL, PRE_OPT_IN_SHA256, _BLOCK,  # noqa: E402
                                         _base, _cases, _gpu)

_Q2_BLOCK = re.compile(r"#ifdef MYNN_SL_Q2_LAND_CAP\n.*?#endif\n", re.S)


def _land_moist(inputs):
    """The oracle columns on land with a moist skin (qsfc above the air)."""
    land = {name: value.copy() for name, value in inputs.items()}
    land["xland"][:] = 1.0
    land["qsfc"][:] = np.float32(1.6) * land["qv1"]
    land["mavail"][:] = 1.0
    return land


def test_the_default_program_is_the_pre_opt_in_unit():
    source = KERNEL.read_text(encoding="utf-8")
    blocks = _Q2_BLOCK.findall(source)
    assert len(blocks) == 1
    assert "xland < 1.5f" in blocks[0] and "module_sf_mynn.F:1149" in blocks[0]
    # the block is a pure addition: the guarded statement is the original line
    assert "#endif\n    q2 = fminf(q2, 1.05f * qv1);\n" in source
    stripped = _BLOCK.sub("", _Q2_BLOCK.sub("", source))
    assert hashlib.sha256(stripped.encode("utf-8")).hexdigest() == PRE_OPT_IN_SHA256
    # stripping only this block leaves the unguarded WRF cap
    assert "    q2 = fminf(q2, 1.05f * qv1);\n" in _Q2_BLOCK.sub("", source)


def test_the_key_follows_the_mode_and_is_read_by_mynn_wrf461_only():
    from woof import correct_mode
    from woof.config import ConfigurationRefused, RunConfig, validate_run_config
    from woof.quirk_corrections import run_reads
    key = "mynn_sfclay_q2_land_cap"
    field = RunConfig.__dataclass_fields__[key]
    assert field.default is correct_mode.CORRECT_KEYS_DEFAULT and field.kw_only
    assert correct_mode.resolve("exact").settings[key] is False
    assert correct_mode.KEY_SCOPE[key] == ("sf_sfclay_physics", (5,))
    assert run_reads(validate_run_config(_base(bl_pbl_physics=5, sf_sfclay_physics=5,
                                               **{key: True})), key)
    assert not run_reads(validate_run_config(_base(bl_pbl_physics=1, sf_sfclay_physics=1,
                                                   **{key: True})), key)
    gsl = validate_run_config(_base(bl_pbl_physics=5, sf_sfclay_physics=5,
                                    mynn_sfclay_variant="gsl_wrf39", **{key: True}))
    assert not run_reads(gsl, key)
    assert "FQ10" not in correct_mode.uncorrected(gsl)
    with pytest.raises(ConfigurationRefused, match="true or false"):
        validate_run_config(_base(bl_pbl_physics=5, sf_sfclay_physics=5, **{key: 1}))


def test_water_keeps_its_q2_above_the_cap_land_and_every_other_output_do_not_move():
    from woof.core.mynn_surface import mynn_surface_layer_default
    cases, inputs = _cases()
    default = mynn_surface_layer_default(inputs)
    named_off = mynn_surface_layer_default(inputs, q2_land_cap=False)
    corrected = mynn_surface_layer_default(inputs, q2_land_cap=True)
    water = inputs["xland"] >= 1.5
    cap = np.float32(1.05) * inputs["qv1"]
    assert water.any() and (~water).any()
    for name, value in default.items():
        assert value.tobytes() == named_off[name].tobytes(), name
        if name != "q2":
            assert value.tobytes() == corrected[name].tobytes(), name
    # WRF: every water column here is held at 1.05*qv1 ...
    np.testing.assert_array_equal(default["q2"][water], cap[water])
    # ... and the corrected Q2 is the uncapped profile value, above the cap and
    # between the air and the surface mixing ratio (qsfc is a specific
    # humidity; qsfcmr = qsfc/(1-qsfc))
    qsfcmr = default["qsfc"] / (np.float32(1.0) - default["qsfc"])
    assert (corrected["q2"][water] > cap[water]).all()
    assert (corrected["q2"][water] != default["q2"][water]).all()
    assert (corrected["q2"][water] <= qsfcmr[water] * (1 + 1e-6)).all()
    assert (corrected["q2"][water] >= inputs["qv1"][water]).all()
    # land is the same words
    assert corrected["q2"][~water].tobytes() == default["q2"][~water].tobytes()
    # a land column whose Q2 would exceed the cap is still capped
    land = _land_moist(inputs)
    land_default = mynn_surface_layer_default(land)
    land_corrected = mynn_surface_layer_default(land, q2_land_cap=True)
    land_cap = np.float32(1.05) * land["qv1"]
    np.testing.assert_array_equal(land_default["q2"], land_cap)
    for name, value in land_default.items():
        assert value.tobytes() == land_corrected[name].tobytes(), name


def test_the_define_and_the_gsl_refusal():
    from woof.core.mynn_sfclay import mynn_surface_defines
    assert mynn_surface_defines("wrf_461", False, False) == ()
    assert mynn_surface_defines("wrf_461", False, True) == (("MYNN_SL_Q2_LAND_CAP", 1),)
    assert mynn_surface_defines("wrf_461", True, True) == (("MYNN_SL_PSIH_ZOLZT", 1),
                                                           ("MYNN_SL_Q2_LAND_CAP", 1))
    with pytest.raises(ValueError, match="gsl_wrf39"):
        mynn_surface_defines("gsl_wrf39", False, True)
    from woof.core.mynn_surface import mynn_surface_layer_default
    _, inputs = _cases()
    with pytest.raises(ValueError, match="gsl_wrf39"):
        mynn_surface_layer_default(inputs, variant="gsl_wrf39", q2_land_cap=True)


@pytest.mark.gpu
@requires_gpu
def test_the_gpu_default_is_unchanged_and_the_correction_matches_the_reference():
    from woof.core.mynn_surface import mynn_surface_layer_default
    _, inputs = _cases()
    water = inputs["xland"] >= 1.5
    default = _gpu(inputs)
    named_off = _gpu(inputs, q2_land_cap=False)
    corrected = _gpu(inputs, q2_land_cap=True)
    reference = mynn_surface_layer_default(inputs, q2_land_cap=True)
    for name, value in default.items():
        assert value.tobytes() == named_off[name].tobytes(), name
        if name != "q2":
            assert value.tobytes() == corrected[name].tobytes(), name
    np.testing.assert_allclose(corrected["q2"], reference["q2"], rtol=2e-5)
    assert (corrected["q2"][water] > default["q2"][water]).all()
    assert corrected["q2"][~water].tobytes() == default["q2"][~water].tobytes()
    land = _land_moist(inputs)
    assert _gpu(land)["q2"].tobytes() == _gpu(land, q2_land_cap=True)["q2"].tobytes()


def test_the_umbrella_turns_it_on_for_the_wrf461_surface_layer():
    from woof.config import ConfigurationRefused, validate_run_config
    from woof.quirk_corrections import apply_run_corrections, run_reads
    key = "mynn_sfclay_q2_land_cap"
    kw = apply_run_corrections(dict(woof_quirk_corrections="all", bl_pbl_physics=5, sf_sfclay_physics=5))
    assert kw[key] is True
    assert validate_run_config(_base(**kw)).mynn_sfclay_q2_land_cap is True
    kw = apply_run_corrections(dict(woof_quirk_corrections="all", bl_pbl_physics=5, sf_sfclay_physics=5,
                                    mynn_sfclay_variant="gsl_wrf39"))
    assert not run_reads(validate_run_config(_base(**kw)), key)
    with pytest.raises(ConfigurationRefused, match=key):
        apply_run_corrections(dict(woof_quirk_corrections="all", bl_pbl_physics=5, sf_sfclay_physics=5,
                                   **{key: False}))
