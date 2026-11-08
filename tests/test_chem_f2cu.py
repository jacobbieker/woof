"""tools/chem_f2cu: the translator's exactness rules, without a GPU.

The kernels it builds are graded against Fortran oracles
(test_chem_sorgam_wrf461_parity.py); these pin the rules those grades rest
on: constant folding is IEEE per operation with correctly rounded
elementary functions (gfortran's MPFR folding), integer powers follow
gfortran's powi_table chains, and the subset it does not understand is
refused rather than guessed.
"""
from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "chem_f2cu"))
import f2cu  # noqa: E402


def _f32(x):
    return Fraction(float(np.float32(x)))


def test_literals_round_directly_to_their_kind():
    assert f2cu.parse_literal("0.1", "float") == _f32(0.1)
    assert f2cu.parse_literal("1.E-16", "float") == _f32(1e-16)
    assert f2cu.parse_literal("2.D0", "double") == 2
    # ties to even at binary32
    assert f2cu.round_to(Fraction(2**24 + 1), "float") == 2**24
    assert f2cu.round_to(Fraction(2**24 + 3), "float") == 2**24 + 4


def test_folding_is_ieee_per_operation():
    tr = f2cu.Translator([])
    scope = f2cu.Scope(tr.module)
    val, kind = tr.const(f2cu.parse_expr("1.0/3.0*3.0"), scope)
    assert kind == "float"
    third = np.float32(1.0) / np.float32(3.0)
    assert val == Fraction(float(third * np.float32(3.0)))
    val, kind = tr.const(f2cu.parse_expr("7/2"), scope)
    assert (val, kind) == (3, "int")
    val, kind = tr.const(f2cu.parse_expr("-7/2"), scope)
    assert val == -3  # Fortran truncates toward zero


def test_constant_intrinsics_fold_correctly_rounded():
    tr = f2cu.Translator([])
    scope = f2cu.Scope(tr.module)
    val, kind = tr.const(f2cu.parse_expr("log(2.0D0)"), scope)
    assert kind == "double" and float(val) == math.log(2.0)
    val, _ = tr.const(f2cu.parse_expr("sqrt(2.0)"), scope)
    assert val == _f32(np.sqrt(np.float32(2.0)))


def test_integer_powers_follow_gfortrans_chain():
    tr = f2cu.Translator([])
    scope = f2cu.Scope(tr.module, implicit=True)
    r = f2cu.Routine("t", [], scope=scope)
    em = f2cu.Emitter(tr, scope, r)
    scope.ensure("x")
    assert em.emit(f2cu.parse_expr("x**2"))[0] == "FMUL(x,x)"
    assert em.emit(f2cu.parse_expr("x**3"))[0] == "FMUL(x,FMUL(x,x))"
    # x**6 = x**3 * x**3 (powi_table[6] = 3), not (x**2)**3
    assert em.emit(f2cu.parse_expr("x**6"))[0] == "FMUL(FMUL(x,FMUL(x,x)),FMUL(x,FMUL(x,x)))"
    assert em.emit(f2cu.parse_expr("x**(-2)"))[0] == "FDIV(1.0f,FMUL(x,x))"
    scope.ensure("n")
    assert em.emit(f2cu.parse_expr("x**n"))[0] == "f2cu_powi_f(x,n)"
    scope.ensure("y")
    assert em.emit(f2cu.parse_expr("x**y"))[0] == "gfk_pow(x,y)"


def test_a_routine_translates_and_the_unknown_is_refused():
    src = """      SUBROUTINE demo(n, a, s)
      INTEGER n, i
      REAL a(n), s
      REAL*8 acc
      REAL c(2)
      DATA c/0.5, 2.0/
      acc = 0.D0
      DO i = 1, n
        IF (a(i) > 0.) THEN
          acc = acc + a(i)*c(1)
        ELSE IF (a(i) == 0.) THEN
          acc = acc + 1.D0
        END IF
      END DO
      s = acc
      END SUBROUTINE demo
"""
    tr = f2cu.Translator([], implicit=False)
    tr.add_source(src, ["demo"])
    text = "\n".join(tr.emit_routine(tr.routines["demo"]))
    assert "__device__ void demo(F2cuModule &mod, int n, const float *a, float &s)" in text
    assert "acc = DADD(acc,((double)(FMUL(a[(i-1)],demo_c[0]))));" in text
    assert tr.constant_tables == ["__constant__ float demo_c[2] = {0.5f, 2.0f};"]
    assert "s = ((float)(acc));" in text
    bad = src.replace("acc = acc + 1.D0", "GO TO 99")
    tr2 = f2cu.Translator([], implicit=False)
    tr2.add_source(bad, ["demo"])
    with pytest.raises(NotImplementedError):
        tr2.emit_routine(tr2.routines["demo"])
