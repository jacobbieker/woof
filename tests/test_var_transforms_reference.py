"""woof.da.var.reference: the binary64 U and U^T (plan V0.2-0.4, gates O2/O3).

Synthetic tests (run everywhere): each stage's adjoint satisfies
<A x, y> = <x, A^T y> to 1e-13 relative, as WRFDA's da_check_vtox_adjoint
defines it.  The O2 gate compares every stage with WRFDA's own words, from
the oracle build's ``woof_u.bin`` / ``woof_xb.bin`` / ``woof_be.bin`` dumps
in ``WOOF_WRFDA471_DIR`` (a test_transforms run of the dump build).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from woof.da.var import be_dat as bd
from woof.da.var import reference as ref

from test_var_be_dat import _synthetic


def _dot(a, b):
    return float(np.sum(np.asarray(a) * np.asarray(b)))


def _xb(rng, nx, ny, nz):
    q = rng.uniform(1e-4, 1.5e-2, (nx, ny, nz))
    return {
        "grid_box_area": rng.uniform(0.8, 1.2, (nx, ny)) * 9e6,
        "coefx": rng.uniform(0.9, 1.1, (nx, ny)) / 6000.0,
        "coefy": rng.uniform(0.9, 1.1, (nx, ny)) / 6000.0,
        "qs": rng.uniform(1e-3, 2e-2, (nx, ny, nz)),
        "q": q,
        "psac": rng.uniform(7e4, 9.5e4, (nx, ny)),
        "dnw": -np.full(nz, 1.0 / nz),
        "rho": rng.uniform(0.3, 1.2, (nx, ny, nz)),
        "p": rng.uniform(2e4, 1e5, (nx, ny, nz)),
        "t": rng.uniform(220, 300, (nx, ny, nz)),
        "c1h": np.linspace(1.0, 0.1, nz),
        "c2h": np.linspace(0.0, 5000.0, nz),
    }


@pytest.fixture(scope="module")
def setup():
    be = _synthetic(ni=9, nj=7, nk=6, nb2d=1)
    b = bd.setup_be_regional(be, ix=9, jy=7, kz=6, ds=3000.0)
    rng = np.random.default_rng(11)
    return b, _xb(rng, 9, 7, 6), rng


def _xa(rng, nx, ny, nz):
    return {
        "u": rng.standard_normal((nx, ny, nz)),
        "v": rng.standard_normal((nx, ny, nz)),
        "t": rng.standard_normal((nx, ny, nz)),
        "q": rng.standard_normal((nx, ny, nz)) * 1e-3,
        "psfc": rng.standard_normal((nx, ny)) * 100,
        "p": rng.standard_normal((nx, ny, nz)) * 100,
        "rho": rng.standard_normal((nx, ny, nz)) * 1e-3,
    }


def test_rf_2d_adjoint(setup):
    b, xb, rng = setup
    s = b.v[0]
    x = rng.standard_normal((9, 7, s.mz))
    y = rng.standard_normal((9, 7, s.mz))
    lhs = _dot(ref.transform_through_rf(x, s.rf_alpha, xb["grid_box_area"], 6), y)
    rhs = _dot(x, ref.transform_through_rf_adj(y, s.rf_alpha, xb["grid_box_area"], 6))
    assert lhs == pytest.approx(rhs, rel=1e-13)


def test_eof_adjoint(setup):
    b, _, rng = setup
    s = b.v[2]
    x = rng.standard_normal((9, 7, s.mz))
    y = rng.standard_normal((9, 7, 6))
    assert _dot(ref.vvtovp(s.evec, s.val, x, 6), y) == pytest.approx(
        _dot(x, ref.vvtovp_adj(s.evec, s.val, y, s.mz)), rel=1e-13)


def test_psichi_adjoint(setup):
    _, xb, rng = setup
    psi, chi = rng.standard_normal((2, 9, 7, 6))
    u, v = rng.standard_normal((2, 9, 7, 6))
    fu, fv = ref.psichi_to_uv(psi, chi, xb["coefx"], xb["coefy"])
    ap, ac = ref.psichi_to_uv_adj(u, v, xb["coefx"], xb["coefy"])
    assert _dot(fu, u) + _dot(fv, v) == pytest.approx(_dot(ap, psi) + _dot(ac, chi), rel=1e-13)


@pytest.mark.parametrize("cv_options", [5, 7])
def test_u_adjoint(cv_options):
    be = _synthetic(ni=9, nj=7, nk=6, nb2d=1, cv_options=cv_options)
    b = bd.setup_be_regional(be, ix=9, jy=7, kz=6, ds=3000.0, config=bd.BeConfig(cv_options=cv_options))
    rng = np.random.default_rng(5)
    xb = _xb(rng, 9, 7, 6)
    n = 9 * 7 * sum(b.cv_mz)
    v = rng.standard_normal(n)
    xa = _xa(rng, 9, 7, 6)
    fwd = ref.transform_u(v, b, xb)["xa"]
    adj = ref.transform_u_adj(xa, b, xb)["cv"]
    lhs = sum(_dot(fwd[k], xa[k]) for k in ("u", "v", "t", "q", "psfc", "p", "rho"))
    assert lhs == pytest.approx(_dot(v, adj), rel=1e-13)


# ---------------------------------------------------------------- gate O2

def _oracle():
    d = os.environ.get("WOOF_WRFDA471_DIR")
    if not d or not all((Path(d) / f).is_file() for f in ("be.dat", "woof_u.bin", "woof_xb.bin")):
        pytest.skip("WOOF_WRFDA471_DIR with be.dat, woof_u.bin and woof_xb.bin is not set")
    from woof.da.var.oracle import read_dump

    d = Path(d)
    dx = read_dump(d / "woof_xb.bin")
    du = read_dump(d / "woof_u.bin")
    nx, ny = dx["lat"].shape
    nz = dx["dnw"].size
    be = bd.read_be_dat(d / "be.dat")
    b = bd.setup_be_regional(be, ix=nx, jy=ny, kz=nz, ds=float(dx["ds"]))
    xb = {k: dx[k] for k in ("grid_box_area", "coefx", "coefy", "qs", "q", "psac", "dnw", "rho", "p", "t",
                             "c1h", "c2h")}
    return b, xb, du


def _rel(a, b, stage="?"):
    """Relative residual (diagnostic); inf unless the binary64 bit patterns match
    when O2_TOL == 0 (so +0.0 vs -0.0 counts).  Checked 2026-11-19: the dump and
    woof agree bitwise at every stage, signed zeros included."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise AssertionError(f"{stage}: shape mismatch {a.shape} vs {b.shape}")
    scale = max(float(np.max(np.abs(b))), 1e-300)
    r = float(np.max(np.abs(a - b))) / scale
    if O2_TOL == 0.0:
        a64 = np.ascontiguousarray(a, dtype=np.float64)
        b64 = np.ascontiguousarray(b, dtype=np.float64)
        if not np.array_equal(a64.view(np.uint64), b64.view(np.uint64)):
            return r if r > 0.0 else float("inf")
    return r


# The plan's O2 tolerance is 1e-13 relative; the binary64 reference follows
# WRFDA's operation order and agrees with the serial gfortran build word for
# word at every stage (measured 2026-11-19), so the gate holds it to that.
O2_TOL = 0.0


def test_o2_forward_stages():
    b, xb, du = _oracle()
    st = ref.transform_u(du["cv1"], b, xb)
    report = {}
    for stage, tag in (("vv", "vv"), ("vp", "vp"), ("vp_bal", "vp_bal")):
        for n in range(1, 6):
            mz = st[stage][f"v{n}"].shape[2]
            report[f"{tag}%v{n}"] = _rel(st[stage][f"v{n}"], du[f"{tag}%v{n}"][:, :, :mz], f"{tag}%v{n}")
    for k in ("u", "v", "t", "q", "psfc", "p", "rho"):
        report[f"xa%{k}"] = _rel(st["xa"][k], du[f"xa%{k}"], f"xa%{k}")
    bad = {k: r for k, r in report.items() if not r <= O2_TOL}
    assert not bad, bad


def test_o2_adjoint_stages():
    b, xb, du = _oracle()
    xa_in = {k: du[f"xa%{k}#2"] for k in ("u", "v", "t", "q", "psfc", "p", "rho")}
    st = ref.transform_u_adj(xa_in, b, xb)
    report = {}
    for k in ("u", "v", "t", "q", "psfc", "p", "rho"):
        report[f"xa_adj%{k}"] = _rel(st["xa"][k], du[f"xa%{k}#3"], f"xa_adj%{k}")
    for n in range(1, 6):
        mz = st["vv"][f"v{n}"].shape[2]
        report[f"vp_adj%v{n}"] = _rel(st["vp"][f"v{n}"], du[f"vp_adj%v{n}"][:, :, : st["vp"][f"v{n}"].shape[2]], f"vp_adj%v{n}")
        report[f"vv_adj%v{n}"] = _rel(st["vv_eof"][f"v{n}"], du[f"vv_adj%v{n}"][:, :, :mz], f"vv_adj%v{n}")
    report["cv2"] = _rel(st["cv"], du["cv2"], "cv2")
    bad = {k: r for k, r in report.items() if not r <= O2_TOL}
    assert not bad, bad


def test_vptox_adj_refuses_cv6():
    z = np.zeros((2, 2, 2))
    xa = {k: z for k in ("u", "v", "t", "q")} | {"psfc": np.zeros((2, 2))}
    with pytest.raises(NotImplementedError):
        ref.vptox_adj(xa, {}, 6, z, z, z)
