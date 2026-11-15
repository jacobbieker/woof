"""advance_w_phi_msf_kpar (kernels/acoustic.cu): the level-parallel w''-phi''
solve against the one-thread-per-column kernel it replaces on the fast
mapped path, word for word, over the moist pressure term, the rigid lid
and the open top, the implicit damper, a specified frame and a 3-D base
state; and the launcher's selection (acoustic._w_phi_launch).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu


def _column_inputs(rng, nz, ny, nx, *, base3d):
    import cupy as cp

    f = lambda *shape: cp.asarray(rng.normal(0.0, 1.0, shape).astype(np.float32))
    u = lambda lo, hi, *shape: cp.asarray(rng.uniform(lo, hi, shape).astype(np.float32))
    full, half, plane = (nz + 1, ny, nx), (nz, ny, nx), (ny, nx)
    eta_h = np.linspace(1.0, 0.0, nz + 1, dtype=np.float32)
    c1f = eta_h.copy(); c2f = ((1.0 - eta_h) * 5000.0).astype(np.float32)
    c1h = (0.5 * (eta_h[1:] + eta_h[:-1])).astype(np.float32)
    c2h = ((1.0 - c1h) * 5000.0).astype(np.float32)
    # WRF's stretched eta: negative dnw/rdnw and interpolation weights
    # that are NOT 0.5 (a product by 0.5 is exact, which would hide a
    # contraction choice at fnm*wd_hi + fnp*wd_lo).
    eta_w = (1.0 - (np.arange(nz + 1) / nz) ** 1.4).astype(np.float32)
    dnw = np.diff(eta_w).astype(np.float32)
    rdnw = (1.0 / dnw).astype(np.float32)
    dn = np.concatenate([[dnw[0]], 0.5 * (dnw[1:] + dnw[:-1])]).astype(np.float32)
    rdn = (1.0 / dn).astype(np.float32)
    fnm = np.concatenate([[0.5], 0.5 * dnw[1:] / dn[1:]]).astype(np.float32)
    fnp = np.concatenate([[0.5], 0.5 * dnw[:-1] / dn[1:]]).astype(np.float32)
    z = np.cumsum(np.full(nz + 1, 400.0, np.float32)) - 400.0
    phb = (z * 9.81).astype(np.float32)
    phb3 = np.broadcast_to(phb[:, None, None], full).copy()
    thb = np.linspace(290.0, 320.0, nz, dtype=np.float32)
    thb3 = np.broadcast_to(thb[:, None, None], half).copy()
    arrays = dict(
        w_pp=f(*full) * 0.1, ph_pp=f(*full) * 5.0, rw_t=f(*full) * 1e-3,
        rph_t=f(*full) * 1e-2, ww_pp=f(*full) * 1e-2,
        mu_pp=f(*plane) * 10.0, mu_pp_old=f(*plane) * 10.0,
        th_pp=f(*half) * 0.1, th_pp_old=f(*half) * 0.1, thp=f(*half) * 2.0,
        thb=cp.asarray(thb3 if base3d else thb),
        php=f(*full) * 20.0, phb=cp.asarray(phb3 if base3d else phb),
        alt=u(0.8, 1.2, *half), c2a=u(8e4, 1.2e5, *half),
        a=f(*full) * 0.05, alpha=u(0.8, 1.2, *full), gam=f(*full) * 0.05,
        mup=f(*plane) * 100.0, u_pp=f(nz, ny, nx + 1) * 0.1,
        v_pp=f(nz, ny + 1, nx) * 0.1, w_ref=f(*full) * 0.1,
        ht=u(0.0, 500.0, *plane), rdn=cp.asarray(rdn), rdnw=cp.asarray(rdnw),
        fnm=cp.asarray(fnm), fnp=cp.asarray(fnp), c1h=cp.asarray(c1h),
        c2h=cp.asarray(c2h), c1f=cp.asarray(c1f), c2f=cp.asarray(c2f),
        mub2d=u(9e4, 1.1e5, *plane), cqw=u(0.9, 1.0, *full),
        msft=u(0.95, 1.05, *plane), p_pp=f(*half) * 10.0, al_pp=f(*half) * 1e-3)
    return arrays


_WRITTEN = ("w_pp", "ph_pp", "p_pp", "al_pp")


def _run(kernel, grid, block, arrays, *, moist_cq, top_lid, dampmag, spec_zone,
         base3d, boundary, nz, ny, nx):
    import cupy as cp

    a = {k: v.copy() for k, v in arrays.items()}
    args = (a["w_pp"], a["ph_pp"], a["rw_t"], a["rph_t"], a["ww_pp"],
            a["mu_pp"], a["mu_pp_old"], a["th_pp"], a["th_pp_old"],
            a["thp"], a["thb"], a["php"], a["phb"], a["alt"], a["c2a"],
            a["a"], a["alpha"], a["gam"], a["mup"], a["u_pp"], a["v_pp"],
            a["w_ref"], a["ht"], a["rdn"], a["rdnw"], a["fnm"], a["fnp"],
            a["c1h"], a["c2h"], a["c1f"], a["c2f"], a["mub2d"],
            a["cqw"], np.int32(moist_cq), a["msft"], a["p_pp"], a["al_pp"],
            np.float32(1.5), np.float32(-0.5), np.float32(0.0),
            np.float32(1.0 / 3000.0), np.float32(1.0 / 3000.0),
            np.float32(2.0), np.float32(0.1), np.float32(dampmag),
            np.float32(5000.0), np.int32(boundary), np.int32(boundary),
            np.int32(spec_zone), np.int32(base3d), np.int32(top_lid),
            np.int32(nz), np.int32(ny), np.int32(nx))
    kernel(grid, block, args)
    return {k: cp.asnumpy(a[k]).view(np.uint32) for k in _WRITTEN}


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("moist_cq,top_lid,dampmag,spec_zone,base3d,boundary", [
    (1, 1, 0.1, 1, 0, 1), (0, 0, 0.0, 0, 0, 0), (1, 0, 0.1, 1, 1, 1),
    (0, 1, 0.0, 1, 1, 0),
])
@pytest.mark.parametrize("nz,ny,nx", [(20, 7, 9), (49, 30, 37)])
def test_level_parallel_solve_matches_the_column_kernel(
        moist_cq, top_lid, dampmag, spec_zone, base3d, boundary, nz, ny, nx):
    from woof.core.acoustic import _W_PHI_KPAR_BLOCK, _W_PHI_MSF_THREADS, _w_phi_kernel
    from woof.wrf_exact import ENABLED
    if ENABLED:
        pytest.skip("the level-parallel entry is the fast mapped path only")

    rng = np.random.default_rng(17 + nz)
    arrays = _column_inputs(rng, nz, ny, nx, base3d=bool(base3d))
    common = dict(moist_cq=moist_cq, top_lid=top_lid, dampmag=dampmag,
                  spec_zone=spec_zone, base3d=base3d, boundary=boundary,
                  nz=nz, ny=ny, nx=nx)
    # Both as the forecast compiles them (shape-specialized): the contract
    # is the column kernel the forecast runs.
    column = _run(_w_phi_kernel("advance_w_phi_msf", nz, ny, nx),
                  ((ny * nx + _W_PHI_MSF_THREADS - 1) // _W_PHI_MSF_THREADS,),
                  (_W_PHI_MSF_THREADS,), arrays, **common)
    kpar = _run(_w_phi_kernel("advance_w_phi_msf_kpar", nz, ny, nx),
                ((ny * nx + 31) // 32,), _W_PHI_KPAR_BLOCK, arrays, **common)
    before = {k: arrays[k].get().view(np.uint32) for k in _WRITTEN}
    for key in _WRITTEN:
        assert np.array_equal(column[key], kpar[key]), key
        assert not np.array_equal(column[key], before[key]), key   # it did something


def test_launcher_selects_the_level_parallel_entry_on_the_fast_mapped_path(monkeypatch):
    from types import SimpleNamespace

    import woof.core.acoustic as ac
    from woof.wrf_exact import ENABLED

    seen = []
    monkeypatch.setattr(ac, "_w_phi_kernel", lambda name, nz, ny=0, nx=0: seen.append(name) or name)
    mapped = SimpleNamespace(has_msf=True)
    monkeypatch.setattr(ac, "ACOUSTIC_KPAR_ENABLED", True)
    kernel, grid, block = ac._w_phi_launch(mapped, 49, 30, 37)
    if ENABLED:
        assert kernel == "advance_w_phi_msf" and block == (ac._THREADS,)
    else:
        assert kernel == "advance_w_phi_msf_kpar"
        assert grid == ((30 * 37 + 31) // 32,) and block == (32, 8)
    monkeypatch.setattr(ac, "ACOUSTIC_KPAR_ENABLED", False)
    kernel, grid, block = ac._w_phi_launch(mapped, 49, 30, 37)
    assert kernel == "advance_w_phi_msf"
    assert block == ((ac._THREADS,) if ENABLED else (ac._W_PHI_MSF_THREADS,))
    kernel, _grid, block = ac._w_phi_launch(SimpleNamespace(has_msf=False), 49, 30, 37)
    assert kernel == "advance_w_phi" and block == (ac._THREADS,)
    # the top tier's shared rows exceed 32 KB: the column kernel keeps it
    monkeypatch.setattr(ac, "ACOUSTIC_KPAR_ENABLED", True)
    kernel, _grid, block = ac._w_phi_launch(mapped, 200, 30, 37)
    assert kernel == "advance_w_phi_msf"
    kernel, _grid, _block = ac._w_phi_launch(mapped, 192, 30, 37)
    assert kernel == ("advance_w_phi_msf" if ENABLED else "advance_w_phi_msf_kpar")
