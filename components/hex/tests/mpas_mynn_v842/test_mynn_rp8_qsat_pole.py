"""RP8 (correct mode): MYNN's saturation mixing ratio no longer has a pole at the top of a high lid.

THE BREAKAGE: qsat_blend (mynn_shared.F90:90-98) is 0.622*e/(p-e) with no guard.  Under MPAS's 30 km lid
p - e stays far from zero.  Under the 80-level, 50 km lid (s2s80-v1) the levels above about 35 km have
p of 50-300 Pa and a saturation vapour pressure of the same size or larger, so qsat runs through the
pole to large negative values; at a step where p - e lands next to zero in one column, MYNN's
tendencies, variances and mixing go NaN there and the dycore refuses the next step with thousands of
columns non-finite (x1.40962 120 km, 2026-11-05T00 at step 51).

RP8 takes the guards the NOAA-GSL MYNN-EDMF carries (module_bl_mynnedmf.F90:7650-7662): e at most
0.15 p, p - e at least 1e-5.  Exact mode keeps native MPAS v8.4.2's text byte for byte.
"""
import numpy as np

from woof.hex.mpasphys import bl_mynn_v842 as m
from woof.hex.mpasphys.driver_v842 import CORRECTIONS_V842, Backend, correct_source, table_declarations

ROWS = m.MYNN_TEXT_CORRECTIONS["RP8"]


def test_rp8_text_is_substituted_once_and_only_in_correct_mode():
    assert len(ROWS) == 4
    assert all(m.MYNN_MODULE_CUDA.count(exact) == 1 and count == 1 for exact, _, count in ROWS)
    assert CORRECTIONS_V842["RP8"]["slots"] == {"pbl": ("bl_mynn",)}
    assert "RP8" in table_declarations("pbl", "bl_mynn", Backend.HEX_CUDA)
    corrected = correct_source(m.MYNN_MODULE_CUDA, m.MYNN_TEXT_CORRECTIONS, {"RP1", "RP7", "RP8"}, "test")
    for exact, fixed, _ in ROWS:
        assert exact not in corrected and corrected.count(fixed) == 1
        assert "mpas_mul(p, 0.15f)" in fixed and "1e-5f" in fixed
    # RP7 and RP8 touch different statements, so either can be skipped alone.
    alone = correct_source(m.MYNN_MODULE_CUDA, m.MYNN_TEXT_CORRECTIONS, {"RP8"}, "test")
    assert m.MYNN_TEXT_CORRECTIONS["RP7"][0][0] in alone
    assert correct_source(m.MYNN_MODULE_CUDA, m.MYNN_TEXT_CORRECTIONS, (), "test") == m.MYNN_MODULE_CUDA


def _poly(coefficients, x):
    value = np.float32(coefficients[-1])
    for c in coefficients[-2::-1]:
        value = np.float32(np.float32(c) + np.float32(x * value))
    return value


J = (.611583699e03, .444606896e02, .143177157e01, .264224321e-1, .299291081e-3, .203154182e-5,
     .702620698e-8, .379534310e-11, -.321582393e-13)


def test_the_pole_is_where_the_s2s80_top_sits():
    """A 50 km lid's top level (about 80 Pa, 248 K) is next to the liquid branch's pole; 30 km's is not."""
    f = np.float32
    t = f(248.30)
    esl = _poly(J, f(t - f(273.15)))
    top_50km, top_30km = f(81.5), f(1200.0)
    exact = f(0.622) * esl / (top_50km - esl)
    assert exact < 0                                        # through the pole
    capped = min(esl, top_50km * f(0.15))
    guarded = f(0.622) * capped / max(top_50km - capped, f(1e-5))
    assert 0 < guarded < 1
    # Under the 30 km lid the cap does not bind, so RP8 leaves the word unchanged.
    assert esl < top_30km * f(0.15)
    assert f(0.622) * esl / (top_30km - esl) == f(0.622) * min(esl, top_30km * f(0.15)) / max(
        top_30km - min(esl, top_30km * f(0.15)), f(1e-5))
