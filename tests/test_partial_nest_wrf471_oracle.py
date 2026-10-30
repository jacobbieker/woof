"""Partial-depth nest (Phase D3) coupling operators held to compiled WRF v4.7.1.

The fixture (tests/data/oracles/partial_nest/, built by
tools/partial_nest_wrf471_oracle/build.sh: WRF's own statements cut by
extract.py, gfortran -O0 -ffp-contract=off, REAL = real(4)) carries every
input and WRF's output word.  Each test reruns woof's operator on the same
input words and requires the same output words:

* ``sint`` -- ``interp_fcn_sint`` + ``SINT``, ratio 3, mass and both
  staggerings (woof's ``wrapper="interp"`` registration; at ratio 3 the
  bdy wrapper the partial nest uses has the same stagger offset,
  ``max((3-1)//2, 1) == (3-1)//2``, so the same words);
* ``copy_fcn`` (mass, u, v), ``copy_fcnm``, ``copy_fcni``, spec_zone 1;
* ``smoother`` -- ``sm121`` (smooth_option 1) and ``smdsm`` (2), feedback 1;
* the TOP STRIP: ``relax_bdytend_core`` + ``spec_bdytend`` on WRF's south
  side with ``periodic_x`` (every i processed), the strip transposed onto
  it -- WRF's i is woof's x, WRF's carried k is woof's y, WRF's row j is
  strip row d + 1 (row 0 the interface) -- against woof's
  ``nest_top_relaxation`` (kernels/partial_nest.cu) for a full-level field
  (half = 0, phi) and two half-level fields (half = 1, u and theta, whose
  row 0 is the parent's level above the interface, read as its own target).

Plus the strip tables: ``nest_top_strip_tables`` over a SINT plane equals
``bdy_interp1``'s words wherever the strip overlaps the lateral frame.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_gpu

FIXTURE = Path(__file__).resolve().parent / "data" / "oracles" / "partial_nest"
IPOS, JPOS, RATIO = 7, 6, 3
NPX, NPY, NCX, NCY = 24, 20, 24, 18


def _case(name):
    from woof.verify.chem_oracle import load_case
    return load_case(FIXTURE / name)


def _woof(a):
    """WRF (i, k, j) -> woof (k, j, i), C-contiguous."""
    return np.ascontiguousarray(np.asarray(a).transpose(1, 2, 0))


def _same(got, want, what):
    g = np.ascontiguousarray(got).view(np.uint32)
    w = np.ascontiguousarray(want).view(np.uint32)
    bad = np.flatnonzero(g.ravel() != w.ravel())
    assert bad.size == 0, (f"{what}: {bad.size} of {g.size} words differ, first at "
                           f"{np.unravel_index(bad[0], g.shape)}: "
                           f"{np.asarray(got).ravel()[bad[0]]!r} vs "
                           f"{np.asarray(want).ravel()[bad[0]]!r}")


def _reg(stagger, wrapper="interp"):
    from woof.core.nest_interp import register_nest
    return register_nest(nri=RATIO, nrj=RATIO, i_parent_start=IPOS,
                         j_parent_start=JPOS, child_nx=NCX, child_ny=NCY,
                         parent_nx=NPX, parent_ny=NPY, stagger=stagger,
                         wrapper=wrapper)


_STAG = {"m": "", "x": "x", "y": "y"}


def test_fixture_is_the_published_one():
    lines = (FIXTURE / "FIXTURES.sha256").read_text().split("\n")
    rows = [line.split() for line in lines if line.strip()]
    assert len(rows) > 60
    for digest, path in rows:
        assert hashlib.sha256((FIXTURE / path).read_bytes()).hexdigest() == digest, path


def test_bdy_and_interp_wrappers_agree_at_ratio_three():
    """The partial nest registers its planes and strips with the bdy
    wrapper; at ratio 3 its donor maps are the interp wrapper's."""
    for s in ("", "x", "y"):
        a, b = _reg(s, "interp"), _reg(s, "bdy")
        assert (a.ioff, a.joff) == (b.ioff, b.joff)
        np.testing.assert_array_equal(a.ci, b.ci)
        np.testing.assert_array_equal(a.cj, b.cj)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("stag", ["m", "x", "y"])
def test_sint_matches_wrf(stag):
    import cupy as cp
    from woof.core.nest_interp import sint
    c = _case(f"sint_{stag}")
    got = cp.asnumpy(sint(cp.asarray(_woof(c["cfld"])), _reg(_STAG[stag])))
    _same(got, _woof(c["nfld"]), f"sint {stag}")


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("stag", ["m", "x", "y"])
def test_copy_fcn_matches_wrf(stag):
    import cupy as cp
    from woof.core.nest_interp import copy_fcn
    c = _case(f"copy_fcn_{stag}")
    parent = cp.asarray(_woof(c["cfld_in"]))
    copy_fcn(parent, cp.asarray(_woof(c["nfld"])), _reg(_STAG[stag]), spec_zone=1)
    got = cp.asnumpy(parent)
    _same(got, _woof(c["cfld"]), f"copy_fcn {stag}")
    assert np.any(got != _woof(c["cfld_in"])), "the restriction must write"


@requires_gpu
@pytest.mark.gpu
def test_copy_fcnm_matches_wrf():
    import cupy as cp
    from woof.core.nest_interp import copy_fcnm
    c = _case("copy_fcnm")
    parent = cp.asarray(_woof(c["cfld_in"]))
    copy_fcnm(parent, cp.asarray(_woof(c["nfld"])), _reg(""), spec_zone=1)
    _same(cp.asnumpy(parent), _woof(c["cfld"]), "copy_fcnm")


@requires_gpu
@pytest.mark.gpu
def test_copy_fcni_matches_wrf():
    import cupy as cp
    from woof.core.nest_interp import copy_fcni
    c = _case("copy_fcni")
    parent = cp.asarray(_woof(c["cfld_in"]).astype(np.int32))
    copy_fcni(parent, cp.asarray(_woof(c["nfld"]).astype(np.int32)), _reg(""),
              spec_zone=1)
    got = cp.asnumpy(parent)
    np.testing.assert_array_equal(got, _woof(c["cfld"]))
    assert np.any(got != _woof(c["cfld_in"]))


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("option", [1, 2])
@pytest.mark.parametrize("stag", ["m", "x", "y"])
def test_smoother_matches_wrf(option, stag):
    import cupy as cp
    from woof.core.nest_interp import smoother
    c = _case(f"smoother{option}_{stag}")
    field = cp.asarray(_woof(c["cfld_in"]))
    smoother(field, _reg(_STAG[stag]), smooth_option=option,
             scratch=cp.zeros(field.size, dtype=cp.float32))
    got = cp.asnumpy(field)
    _same(got, _woof(c["cfld"]), f"smoother {option} {stag}")
    assert np.any(got != _woof(c["cfld_in"]))


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("case", ["strip_full_phi", "strip_half_u", "strip_half_theta"])
def test_top_strip_relaxation_matches_wrf(case):
    """woof's top zone on the transposed strip == WRF's south side."""
    import cupy as cp
    from woof.core.kernels import get_kernel
    from woof.ingest.lateral_bc import _weights
    c = _case(case)
    half, nzf = int(c["half"]), int(c["nzf"])
    dtbc, dt = np.float32(c["dtbc"]), float(c["dt"])
    width, spec, relax = 7, 1, 6
    fcx, gcx = _weights(width, spec, relax, dt, 0.0, wrf_real=True)
    _same(fcx, c["fcx"], "fcx")
    _same(gcx, c["gcx"], "gcx")
    rows = c["field"]                      # (nx, ny, width): [i, k=y, j=row]
    nxf, nyf = rows.shape[0], rows.shape[1]
    rng = np.random.default_rng(7)
    field = rng.standard_normal((nzf, nyf, nxf)).astype(np.float32)
    tend = rng.standard_normal((nzf, nyf, nxf)).astype(np.float32)
    level = {d: nzf - 1 + half - d for d in range(width)}
    for d, k in level.items():
        if 0 <= k < nzf:
            field[k] = rows[:, :, d].T
            tend[k] = c["tend_in"][:, :, d].T
    tend0 = tend.copy()
    val = np.ascontiguousarray(c["top_val"].transpose(2, 1, 0))
    ttd = np.ascontiguousarray(c["top_tend"].transpose(2, 1, 0))
    d_tend = cp.asarray(tend)
    n = relax * nyf * nxf
    get_kernel("partial_nest", "nest_top_relaxation")(
        ((n + 255) // 256,), (256,),
        (cp.asarray(field), d_tend, cp.asarray(val), cp.asarray(ttd),
         cp.asarray(fcx), cp.asarray(gcx), dtbc, np.int32(width), np.int32(spec),
         np.int32(relax), np.int32(0), np.int32(half),
         np.int32(nzf), np.int32(nyf), np.int32(nxf)))
    got = cp.asnumpy(d_tend)
    compared = 0
    for d in range(relax):
        k = level[d]
        if not 1 <= k < nzf:
            continue
        _same(got[k], c["tend"][:, :, d].T, f"{case} row {d} (level {k})")
        compared += 1
    assert compared == (relax if half == 0 else relax - 1)
    untouched = [k for k in range(nzf) if k not in
                 {level[d] for d in range(relax)} or k < 1]
    _same(got[untouched], tend0[untouched], "levels outside the zone")
    # non-vacuity: the relaxation moved the relaxed rows
    assert not np.array_equal(got[level[2]], tend0[level[2]])


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("stag", ["m", "x"])
def test_strip_tables_equal_bdy_interp1_on_the_frame(stag):
    """The top strip's tables, built from a SINT plane with the bdy-wrapper
    registration, are bdy_interp1's words wherever the strip overlaps the
    lateral frame (west and south side tables)."""
    import cupy as cp
    from woof.core.kernels import get_kernel
    from woof.core.nest_interp import bdy_interp1, sint
    reg = _reg(_STAG[stag], "bdy")
    c = _case(f"sint_{stag}")
    parent = _woof(c["cfld"])
    rng = np.random.default_rng(11)
    child = (_woof(c["nfld"]) * (1.0 + 1e-3 * rng.standard_normal(
        _woof(c["nfld"]).shape))).astype(np.float32)
    cdt = np.float32(12.0)
    tables = bdy_interp1(cp.asarray(parent), cp.asarray(child), reg,
                         parent_dt_fp32=cdt, spec_zone=1, relax_zone=4,
                         spec_bdy_width=5)
    psca = sint(cp.asarray(parent), reg)
    val = cp.zeros_like(psca)
    tend = cp.zeros_like(psca)
    n = int(psca.size)
    get_kernel("partial_nest", "nest_top_strip_tables")(
        ((n + 255) // 256,), (256,), (psca, cp.asarray(child), val, tend, cdt, np.int32(n)))
    val, tend = cp.asnumpy(val), cp.asnumpy(tend)
    wv, wt = (cp.asnumpy(a) for a in tables["west"])       # (nz, ny, sz)
    sv, st = (cp.asnumpy(a) for a in tables["south"])      # (nz, sz, nx)
    sz = wv.shape[-1]
    _same(val[:, :, :sz], wv, "west value")
    _same(tend[:, :, :sz], wt, "west tendency")
    _same(val[:, :sz, :], sv, "south value")
    _same(tend[:, :sz, :], st, "south tendency")
