"""Every partial-nest D4 kernel (kernels/partial_nest_d4.cu) on host-made inputs.

Each entry point is launched on deterministic host words and held, word for
word, to a NumPy statement of its arithmetic or to the D2/D3 entry points it
fuses.  Traced on CUDA (--fmad=false) and on Metal (WOOF_LAUNCH_TRACE_DIR,
tools/launch_trace.py, tools/backend_receipts.py) these tests are the
backend receipt for D4's kernels.
"""

import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [pytest.mark.gpu]

F = np.float32


def _skip_strict():
    from woof import wrf_exact
    if wrf_exact.ENABLED:
        pytest.skip("the partial-depth nest runs in the default build only (refused by name)")


def _k(name, module="partial_nest_d4"):
    from woof.core.kernels import get_kernel
    return get_kernel(module, name)


def _eq(got, want, what):
    import cupy as cp
    g = np.ascontiguousarray(cp.asnumpy(got)).view(np.uint32)
    w = np.ascontiguousarray(cp.asnumpy(want) if hasattr(want, "get")
                             else np.asarray(want, dtype=F)).view(np.uint32)
    bad = np.flatnonzero(g.ravel() != w.ravel())
    assert bad.size == 0, f"{what}: {bad.size} of {g.size} words differ"


def _lerp(a, b, f):
    return (a + (f * (b - a).astype(F)).astype(F)).astype(F)


@requires_gpu
def test_fused_stage_forcing_is_d3s_four_launches():
    _skip_strict()
    import cupy as cp
    rng = np.random.default_rng(31)
    ny, nx, nz = 5, 7, 6
    n = ny * nx
    r = lambda *s: (rng.standard_normal(s) * 3).astype(F)
    om_a, om_b, adj_a, adj_b = r(n), r(n), r(n) * F(1e-3), r(n) * F(1e-3)
    th_a, th_b = r(n) + F(330), r(n) + F(330)
    ph_a, ph_b = r(n) * F(10) + F(6e4), r(n) * F(10) + F(6e4)
    u_a, u_b = r(ny, nx + 1) + F(20), r(ny, nx + 1) + F(20)
    v_a, v_b = r(ny + 1, nx), r(ny + 1, nx)
    w, php = r(nz + 1, ny, nx) * F(0.1), r(nz + 1, ny, nx) * F(20)
    phb = np.linspace(0, 6e4, nz + 1).astype(F)
    thp, thb = r(nz, ny, nx), np.linspace(300, 330, nz).astype(F)
    fs = F(0.4166667)
    d = {k: cp.asarray(v) for k, v in dict(
        om_a=om_a, om_b=om_b, adj_a=adj_a, adj_b=adj_b, th_a=th_a, th_b=th_b,
        ph_a=ph_a, ph_b=ph_b, u_a=u_a, u_b=u_b, v_a=v_a, v_b=v_b, w=w, php=php,
        phb=phb, thp=thp, thb=thb).items()}
    outs = [cp.zeros(n, dtype=cp.float32) for _ in range(6)]
    u_ab, v_ab = cp.zeros((ny, nx + 1), cp.float32), cp.zeros((ny + 1, nx), cp.float32)
    theta = cp.zeros((nz, ny, nx), cp.float32)
    _k("nest_top_stage_forcing_fused")((1,), (64,), (
        d["om_a"], d["om_b"], d["adj_a"], d["adj_b"], d["th_a"], d["th_b"],
        d["ph_a"], d["ph_b"], d["u_a"], d["u_b"], d["v_a"], d["v_b"],
        d["w"], d["php"], d["phb"], d["thp"], d["thb"], np.int32(0), np.int32(nz),
        np.int32(ny), np.int32(nx), fs, *outs, u_ab, v_ab, theta))
    # D3's own entry point on the same words
    ref = [cp.zeros(n, dtype=cp.float32) for _ in range(6)]
    _k("nest_top_stage_forcing", "partial_nest")((1,), (64,), (
        d["om_a"], d["om_b"], d["adj_a"], d["adj_b"], d["th_a"], d["th_b"],
        d["ph_a"], d["ph_b"], d["w"], d["php"], d["phb"], np.int32(0), np.int32(nz),
        fs, *ref, np.int32(n)))
    for got, want, name in zip(outs, ref, ("ww_rf", "adj_rf", "t_above", "pht_above",
                                           "w_star", "ph_star")):
        _eq(got, want, name)
    _eq(u_ab, _lerp(u_a, u_b, fs), "u_above")
    _eq(v_ab, _lerp(v_a, v_b, fs), "v_above")
    _eq(theta[nz - 1], (thp[nz - 1] + thb[nz - 1]).astype(F), "theta_top")
    assert not cp.asnumpy(theta[:nz - 1]).any()


@requires_gpu
@pytest.mark.parametrize("frame", [0, 2])
def test_fused_held_is_the_single_field_entry_point(frame):
    _skip_strict()
    import cupy as cp
    rng = np.random.default_rng(40 + frame)
    nz, ny, nx, spec, relax = 9, 8, 10, 1, 6
    shapes = {"u": (nz, ny, nx + 1, 1), "v": (nz, ny + 1, nx, 1), "t": (nz, ny, nx, 1),
              "ph": (nz + 1, ny, nx, 0), "w": (nz + 1, ny, nx, 0)}
    tend0 = {k: (rng.standard_normal(s[:3]) * 50).astype(F) for k, s in shapes.items()}
    held = {k: cp.asarray((rng.standard_normal(s[:3]) * 5).astype(F))
            for k, s in shapes.items()}
    fused = {k: cp.asarray(v) for k, v in tend0.items()}
    _k("nest_top_apply_held_all")((relax * ny * (nx + 1) // 64 + 2,), (64,), (
        fused["u"], held["u"], fused["v"], held["v"], fused["t"], held["t"],
        fused["ph"], held["ph"], fused["w"], held["w"], np.int32(spec),
        np.int32(relax), np.int32(frame), np.int32(nz), np.int32(ny), np.int32(nx)))
    for kind, (nzf, nyf, nxf, half) in shapes.items():
        single = cp.asarray(tend0[kind])
        _k("nest_top_apply_held", "partial_nest")((relax * nyf * nxf // 64 + 1,), (64,), (
            single, held[kind], np.int32(spec), np.int32(relax), np.int32(frame),
            np.int32(half), np.int32(nzf), np.int32(nyf), np.int32(nxf)))
        _eq(fused[kind], single, kind)


@requires_gpu
@pytest.mark.parametrize("centered", [0, 1])
def test_fused_face_flux_words(centered):
    _skip_strict()
    import cupy as cp
    rng = np.random.default_rng(50 + centered)
    nz, ny, nx = 6, 5, 7
    r = lambda *s: (rng.standard_normal(s)).astype(F)
    ww = r(ny, nx) * F(30)
    rdnw = (-1.0 / np.linspace(0.02, 0.05, nz)).astype(F)
    fnm, fnp = F(0.48), F(0.52)
    q = {"t": r(nz, ny, nx) + F(320), "u": r(nz, ny, nx + 1) * F(10), "v": r(nz, ny + 1, nx)}
    above = {"t": r(ny, nx) + F(321), "u": r(ny, nx + 1) * F(10), "v": r(ny + 1, nx)}
    tend = {k: r(*v.shape) * F(5) for k, v in q.items()}
    dev_t = {k: cp.asarray(v) for k, v in tend.items()}
    flux = {k: cp.zeros(v.shape, cp.float32) for k, v in above.items()}
    _k("nest_top_face_flux_all")((1,), (64,), (
        dev_t["t"], cp.asarray(q["t"]), cp.asarray(above["t"]), flux["t"],
        dev_t["u"], cp.asarray(q["u"]), cp.asarray(above["u"]), flux["u"],
        dev_t["v"], cp.asarray(q["v"]), cp.asarray(above["v"]), flux["v"],
        cp.asarray(ww), cp.asarray(rdnw), fnm, fnp, np.int32(centered),
        np.int32(1), np.int32(1), np.int32(nz), np.int32(ny), np.int32(nx)))
    for kind in ("t", "u", "v"):
        a, qq = above[kind], q[kind][nz - 1]
        if kind == "t":
            vel = ww
        elif kind == "u":
            ia = np.minimum(np.arange(nx + 1), nx - 1)
            ib = np.maximum(np.arange(nx + 1) - 1, 0)
            vel = (F(0.5) * (ww[:, ib] + ww[:, ia]).astype(F)).astype(F)
        else:
            ja = np.minimum(np.arange(ny + 1), ny - 1)
            jb = np.maximum(np.arange(ny + 1) - 1, 0)
            vel = (F(0.5) * (ww[jb, :] + ww[ja, :]).astype(F)).astype(F)
        if centered:
            qf = ((fnm * a).astype(F) + (fnp * qq).astype(F)).astype(F)
        else:
            qf = np.where(vel > 0, a, qq).astype(F)
        fl = (vel * qf).astype(F)
        want = tend[kind].copy()
        want[nz - 1] = (want[nz - 1] - (rdnw[nz - 1] * fl).astype(F)).astype(F)
        _eq(flux[kind], fl, kind + " flux")
        _eq(dev_t[kind], want, kind + " tendency")


@requires_gpu
def test_scalar_interface_flux_and_update_words():
    _skip_strict()
    import cupy as cp
    rng = np.random.default_rng(60)
    nz, ny, nx = 5, 4, 6
    n = ny * nx
    q = np.abs(rng.standard_normal((nz, ny, nx)) * 4e-3).astype(F)
    q_above = np.abs(rng.standard_normal(n) * 4e-3).astype(F)
    ww_m = (rng.standard_normal((nz + 1, ny, nx)) * 20).astype(F)
    fnm, fnp = F(0.47), F(0.53)
    flux = cp.zeros(n, cp.float32)
    _k("nest_top_scalar_face_flux")((1,), (64,), (
        cp.asarray(q), cp.asarray(q_above), cp.asarray(ww_m), fnm, fnp, flux,
        np.int32(nz), np.int32(n)))
    qt = q[nz - 1].ravel()
    vel = ww_m[nz].ravel()
    qf = ((fnm * q_above).astype(F) + (fnp * qt).astype(F)).astype(F)
    qf = np.where((vel < 0) & (qf > qt), qt, qf).astype(F)
    want_flux = (vel * qf).astype(F)
    _eq(flux, want_flux, "flux")
    mub = (np.full(n, 6e4) + rng.standard_normal(n) * 100).astype(F)
    mup = (rng.standard_normal(n) * 30).astype(F)
    c1h = np.linspace(0.9, 0.0, nz).astype(F)
    c2h = np.linspace(1e3, 4e4, nz).astype(F)
    rdnw = (-1.0 / np.linspace(0.02, 0.05, nz)).astype(F)
    dt = F(1.3333334)
    qd = cp.asarray(q)
    _k("nest_top_scalar_apply")((1,), (64,), (
        qd, flux, cp.asarray(mub), cp.asarray(mup), cp.asarray(c1h), cp.asarray(c2h),
        cp.asarray(rdnw), dt, np.int32(nz), np.int32(n)))
    k = nz - 1
    mass = ((c1h[k] * (mub + mup).astype(F)).astype(F) + c2h[k]).astype(F)
    tend = ((F(0.0) - rdnw[k]).astype(F) * want_flux).astype(F)
    v = (qt + ((dt * tend).astype(F) / mass).astype(F)).astype(F)
    want = q.copy()
    want[k] = np.where(v > 0, v, F(0.0)).reshape(ny, nx)
    _eq(qd, want, "q")


@requires_gpu
@pytest.mark.parametrize("full", [0, 1])
def test_extended_column_fill_words(full):
    _skip_strict()
    import cupy as cp
    rng = np.random.default_rng(70 + full)
    nz, k, ny, nx = 10, 6, 3, 4
    n = ny * nx
    nout = nz + full
    nb = k + full
    below = rng.standard_normal((nb + 2, ny, nx)).astype(F)
    above = rng.standard_normal((nout - k, ny, nx)).astype(F)
    out = cp.zeros((nout, ny, nx), cp.float32)
    _k("nest_ext_column_fill")((nout * n // 64 + 1,), (64,), (
        cp.asarray(below[:nb]), cp.asarray(above), out, np.int32(nb), np.int32(nout),
        np.int32(full), np.int32(n)))
    want = np.concatenate([below[:nb], above[full:]])
    _eq(out, want, "column")
