"""tools/chem_f2cu: the translator's exactness rules, without a GPU.

The kernels it builds are graded against Fortran oracles
(test_chem_sorgam_wrf471_parity.py); these pin the rules those grades rest
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


def _routine(src, name, **kw):
    tr = f2cu.Translator(kw.pop("modules", []), implicit=False, kind_names={"dp": 8}, **kw)
    tr.add_source(src, [name])
    return tr, "\n".join(tr.emit_routine(tr.routines[name]))


def test_named_real_kinds_and_declarations_with_a_kind_number():
    # real(dp) and 1._dp spell binary64; "real(8), parameter" must not be
    # read as a default REAL (a kind spec ends in ")", not a word boundary)
    src = """      subroutine k(x)
      implicit none
      integer, parameter :: dp = selected_real_kind(14,300)
      real(dp), intent(inout) :: x
      real(dp), parameter :: big = huge(1.0_dp)
      x = min(x, big) + real(3, kind=dp)
      end subroutine k
"""
    tr, text = _routine(src, "k")
    assert tr.routines["k"].scope.vars["x"].kind == "double"
    assert tr.routines["k"].scope.vars["big"].param == (f2cu._KIND_CONST["huge"]["double"], "double")
    assert "x = DADD(f2cu_mind(x,1.7976931348623157e+308),((double)(3)));" in text


def test_sections_scalarise_over_the_left_side_first_dimension_innermost():
    src = """      subroutine s(n, a, b)
      implicit none
      integer, intent(in) :: n
      real(dp), intent(inout) :: a(n,3)
      real(dp), intent(in) :: b(2*n)
      a(:,2) = a(:,1) + b(1:2*n:2)
      a(2:n,3) = 0._dp
      end subroutine s
"""
    _, text = _routine(src, "s", modules=["      integer, parameter :: dp = 8\n"])
    lines = [ln.strip() for ln in text.splitlines() if "//" in ln and "for (int" in ln]
    # one loop over a(:,2)'s extent; b is read with its own lower bound and stride
    assert "a[((1+f2cu_i" in lines[0] and "b[((1+(f2cu_i" in lines[0] and "*2))-1)]" in lines[0]
    assert lines[0].startswith("for (int f2cu_i1 = 0; f2cu_i1 < ((n-1)+1); ++f2cu_i1)")
    assert lines[1].startswith("for (int f2cu_i2 = 0; f2cu_i2 < ((n-2)+1); ++f2cu_i2)")


def test_overlapping_sections_are_refused_and_disjoint_columns_taken():
    ok = """      subroutine t(n, m, x)
      implicit none
      integer, intent(in) :: n, m
      real(dp), intent(inout) :: x(n,4)
      x(:,m) = x(:,m) - x(:,m-1)
      end subroutine t
"""
    mods = ["      integer, parameter :: dp = 8\n"]
    _routine(ok, "t", modules=mods)
    bad = ok.replace("x(:,m) = x(:,m) - x(:,m-1)", "x(2:n,m) = x(1:n-1,m)")
    tr = f2cu.Translator(mods, implicit=False)
    tr.add_source(bad, ["t"])
    with pytest.raises(NotImplementedError, match="overlapping"):
        tr.emit_routine(tr.routines["t"])


def test_sum_and_dot_product_are_running_sums_from_zero():
    src = """      subroutine r(n, a, b, s)
      implicit none
      integer, intent(in) :: n
      real(dp), intent(in) :: a(n), b(n)
      real(dp), intent(out) :: s
      s = sum(a(:)) + dot_product(a(1:n), b(1:n))
      end subroutine r
"""
    _, text = _routine(src, "r", modules=["      integer, parameter :: dp = 8\n"])
    assert "double f2cu_r2 = 0.0;" in text
    assert "f2cu_r2 = DADD(f2cu_r2,a[((1+f2cu_q1)-1)]);" in text
    assert "f2cu_r4 = DADD(f2cu_r4,DMUL(a[((1+f2cu_q3)-1)],b[((1+f2cu_q3)-1)]));" in text
    assert "s = DADD(f2cu_r2,f2cu_r4);" in text


def test_where_select_case_exit_cycle_and_construct_names():
    src = """      subroutine w(n, a, k)
      implicit none
      integer, intent(in) :: n, k
      real(dp), intent(inout) :: a(n)
      integer :: i
      where( a(:) < 0._dp )
         a(:) = 0._dp
      endwhere
      select case( k )
         case( 1, 2 )
            a(1) = 1._dp
         case default
            a(1) = 2._dp
      end select
outer : &
      do i = 1, n
         if( a(i) > 5._dp ) exit
         if( a(i) > 4._dp ) cycle
         a(i) = a(i) + 1._dp
      end do outer
      do
         exit
      end do
      end subroutine w
"""
    _, text = _routine(src, "w", modules=["      integer, parameter :: dp = 8\n"])
    assert "if ((a[((1+f2cu_i1)-1)] < 0.0)) {" in text
    assert "if ((k == 1) || (k == 2)) {" in text and "} else {" in text
    assert "break;" in text and "continue;" in text and "for (;;) {" in text


def test_automatic_arrays_live_on_the_workspace_stack():
    src = """      subroutine a(n, x)
      implicit none
      integer, intent(in) :: n
      real(dp), intent(out) :: x
      real(dp) :: tmp(n)
      tmp(:) = 1._dp
      x = tmp(n)
      return
      end subroutine a
"""
    tr = f2cu.Translator(["      integer, parameter :: dp = 8\n"], implicit=False, kind_names={"dp": 8})
    tr.workspace = {"threshold": 64}
    tr.add_source(src, ["a"])
    text = "\n".join(tr.emit_routine(tr.routines["a"]))
    assert "double *tmp = mod.ws_d; mod.ws_d += (((n-1)+1));" in text
    assert "mod.ws_d = f2cu_wsd0; mod.ws_i = f2cu_wsi0; return;" in text
    assert tr.workspace_arrays == [("a", "tmp", "double", tr.routines["a"].scope.vars["tmp"].dims)]


def test_data_implied_do_and_reserved_names():
    mod = """      integer, parameter :: dp = 8
      real(dp) :: tab(2,2)
      integer :: ii, jj
      data ((tab(ii,jj),ii=1,2),jj=1,2) / 1., 2., 3., 4. /
"""
    src = """      subroutine z(long, y)
      implicit none
      real(dp), intent(in) :: long
      real(dp), intent(out) :: y
      y = long * tab(2,1)
      end subroutine z
"""
    tr = f2cu.Translator([mod], implicit=False, kind_names={"dp": 8})
    tr.module_data_constants = True
    tr.add_source(src, ["z"])
    text = "\n".join(tr.emit_routine(tr.routines["z"]))
    assert tr.module.vars["tab"].data == [1, 2, 3, 4]
    assert "double long_" in text and "y = DMUL(long_,m_tab[" in text
    assert tr.module_tables["tab"] == "__constant__ double m_tab[4] = {1.0, 2.0, 3.0, 4.0};"
