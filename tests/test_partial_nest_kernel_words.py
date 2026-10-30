"""Every partial-nest D3 kernel (kernels/partial_nest.cu) on host-made inputs.

Each entry point is launched on deterministic inputs built on the host and
held to a NumPy statement of its arithmetic, word for word.  Because the
inputs are host words, the same tests traced on CUDA and on Metal
(WOOF_LAUNCH_TRACE_DIR, tools/launch_trace.py, tools/backend_receipts.py)
compare every launch of the new unit on equal inputs: the backend receipt
for D3's kernels.  (The run tests step the dycore, whose EOS powf is a
vendor-libm word, so their traces part at the first diagnostics launch.)
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


def _k(name):
    from woof.core.kernels import get_kernel
    return get_kernel("partial_nest", name)


def _eq(got, want, what):
    import cupy as cp
    g = np.ascontiguousarray(cp.asnumpy(got)).view(np.uint32)
    w = np.ascontiguousarray(np.asarray(want, dtype=F)).view(np.uint32)
    bad = np.flatnonzero(g.ravel() != w.ravel())
    assert bad.size == 0, f"{what}: {bad.size} of {g.size} words differ"


def _lerp(a, b, f):
    return (a + (f * (b - a).astype(F)).astype(F)).astype(F)


@requires_gpu
def test_stage_forcing_words():
    _skip_strict()
    import cupy as cp
    rng = np.random.default_rng(11)
    ny, nx, nz = 6, 9, 5
    n = ny * nx
    pl = {k: (rng.standard_normal(n) * s).astype(F) for k, s in
          (("om_a", 2), ("om_b", 2), ("adj_a", 1e-3), ("adj_b", 1e-3), ("th_a", 3),
           ("th_b", 3), ("ph_a", 50), ("ph_b", 50))}
    pl["th_a"] += F(330.0)
    pl["th_b"] += F(330.0)
    pl["ph_a"] += F(6.0e4)
    pl["ph_b"] += F(6.0e4)
    w = (rng.standard_normal((nz + 1, ny, nx)) * 0.1).astype(F)
    php = (rng.standard_normal((nz + 1, ny, nx)) * 20).astype(F)
    phb = np.linspace(0, 6e4, nz + 1).astype(F)
    fs = F(0.37)
    outs = [cp.zeros(n, dtype=cp.float32) for _ in range(6)]
    d = {k: cp.asarray(v) for k, v in pl.items()}
    _k("nest_top_stage_forcing")((1,), (64,), (
        d["om_a"], d["om_b"], d["adj_a"], d["adj_b"], d["th_a"], d["th_b"],
        d["ph_a"], d["ph_b"], cp.asarray(w), cp.asarray(php), cp.asarray(phb),
        np.int32(0), np.int32(nz), fs, *outs, np.int32(n)))
    want = [_lerp(pl["om_a"], pl["om_b"], fs), _lerp(pl["adj_a"], pl["adj_b"], fs),
            _lerp(pl["th_a"], pl["th_b"], fs), _lerp(pl["ph_a"], pl["ph_b"], fs),
            w[nz].ravel(), (phb[nz] + php[nz]).astype(F).ravel()]
    for got, ref, name in zip(outs, want, ("ww_rf", "adj_rf", "t_above", "pht_above",
                                           "w_star", "ph_star")):
        _eq(got, ref, name)


@requires_gpu
def test_budget_residual_words():
    _skip_strict()
    import cupy as cp
    rng = np.random.default_rng(12)
    nyp, nxp, r, ipos, jpos, nyc, nxc = 10, 14, 3, 4, 3, 12, 18
    parent = rng.standard_normal((nyp, nxp)).astype(F)
    child = rng.standard_normal((nyc, nxc)).astype(F)
    res = cp.zeros(1, dtype=cp.float32)
    _k("nest_top_budget_residual")((1,), (1,), (
        cp.asarray(parent), cp.asarray(child), res, np.int32(ipos), np.int32(jpos),
        np.int32(r), np.int32(r), np.int32(nyp), np.int32(nxp), np.int32(nyc), np.int32(nxc)))
    sp = 0.0
    for pj in range(nyc // r):
        for pi in range(nxc // r):
            v = float(parent[jpos - 1 + pj, ipos - 1 + pi])
            for _ in range(r * r):
                sp += v
    sc = 0.0
    for v in child.ravel():
        sc += float(v)
    _eq(res, [np.float32((sp - sc) / (nyc * nxc))], "residual")


@requires_gpu
def test_strip_gather_and_tables_words():
    _skip_strict()
    import cupy as cp
    rng = np.random.default_rng(13)
    width, nzf, ny, nx = 4, 7, 5, 6
    for half in (0, 1):
        field = (rng.standard_normal((nzf, ny, nx)) * 7e4).astype(F)
        psca = (rng.standard_normal((width, ny, nx)) * 7e4).astype(F)
        strip = cp.zeros((width, ny, nx), dtype=cp.float32)
        n = width * ny * nx
        _k("nest_top_strip_gather")((1,), (256,), (
            cp.asarray(field), cp.asarray(psca), strip, np.int32(width), np.int32(half),
            np.int32(nzf), np.int32(ny), np.int32(nx)))
        want = np.stack([psca[d] if nzf - 1 + half - d >= nzf else field[nzf - 1 + half - d]
                         for d in range(width)])
        _eq(strip, want, f"gather half={half}")
        val = cp.zeros_like(strip)
        tend = cp.zeros_like(strip)
        cdt = F(12.0)
        _k("nest_top_strip_tables")((1,), (256,), (
            cp.asarray(psca), strip, val, tend, cdt, np.int32(n)))
        rdt = 1.0 / np.float64(cdt)
        want_t = (rdt * (psca - want).astype(F).astype(np.float64)).astype(F)
        _eq(val, want, "value")
        _eq(tend, want_t, "tendency")
        if half:
            assert not cp.asnumpy(tend[0]).any()      # the virtual row: its own target


@requires_gpu
@pytest.mark.parametrize("half,frame", [(0, 0), (1, 0), (0, 2), (1, 2)])
def test_relaxation_and_held_words(half, frame):
    _skip_strict()
    import cupy as cp
    from woof.ingest.lateral_bc import _weights
    rng = np.random.default_rng(14 + 2 * half + frame)
    spec, relax, width = 1, 6, 7
    nzf, ny, nx = 9, 8, 10
    field = (rng.standard_normal((nzf, ny, nx)) * 7e4).astype(F)
    tend0 = (rng.standard_normal((nzf, ny, nx)) * 50).astype(F)
    val = (field[::-1][:width] + rng.standard_normal((width, ny, nx)) * 300).astype(F) \
        if not half else (rng.standard_normal((width, ny, nx)) * 7e4).astype(F)
    tt = (rng.standard_normal((width, ny, nx)) * 30).astype(F)
    if half:
        tt[0] = 0.0
    fcx, gcx = _weights(width, spec, relax, 4.0, 0.0, wrf_real=True)
    dtbc = F(2.0)
    tend = cp.asarray(tend0)
    _k("nest_top_relaxation")((relax * ny * nx // 256 + 1,), (256,), (
        cp.asarray(field), tend, cp.asarray(val), cp.asarray(tt), cp.asarray(fcx),
        cp.asarray(gcx), dtbc, np.int32(width), np.int32(spec), np.int32(relax),
        np.int32(frame), np.int32(half), np.int32(nzf), np.int32(ny), np.int32(nx)))
    want = tend0.copy()
    top = nzf - 1 + half

    def bdy(d, j, i):
        return F(F(val[d, j, i]) + F(dtbc * tt[d, j, i]))

    for d in range(relax):
        k = top - d
        if k >= nzf or k < 1:
            continue
        for j in range(frame, ny - frame):
            for i in range(frame, nx - frame):
                if d < spec:
                    want[k, j, i] = tt[d, j, i]
                    continue
                im, ip = max(i - 1, 0), min(i + 1, nx - 1)
                outer = field[k + 1, j, i] if k + 1 < nzf else val[d - 1, j, i]
                f0 = F(bdy(d, j, i) - field[k, j, i])
                f1 = F(bdy(d, j, im) - field[k, j, im])
                f2 = F(bdy(d, j, ip) - field[k, j, ip])
                f3 = F(bdy(d - 1, j, i) - outer)
                f4 = F(bdy(d + 1, j, i) - field[k - 1, j, i])
                lap = F(F(F(F(f1 + f2) + f3) + f4) - F(F(4.0) * f0))
                want[k, j, i] = F(F(want[k, j, i] + F(fcx[d] * f0)) - F(gcx[d] * lap))
    _eq(tend, want, "relaxation")
    # nest_top_apply_held: specified rows replaced, relaxation rows added.
    held = cp.asarray(want - tend0)
    out = cp.asarray(tend0)
    _k("nest_top_apply_held")((relax * ny * nx // 256 + 1,), (256,), (
        out, held, np.int32(spec), np.int32(relax), np.int32(frame), np.int32(half),
        np.int32(nzf), np.int32(ny), np.int32(nx)))
    h = cp.asnumpy(held)
    want2 = tend0.copy()
    for d in range(relax):
        k = top - d
        if k >= nzf or k < 1:
            continue
        sl = (k, slice(frame, ny - frame), slice(frame, nx - frame))
        want2[sl] = h[sl] if d < spec else (tend0[sl] + h[sl]).astype(F)
    _eq(out, want2, "apply_held")


@requires_gpu
def test_specified_predictor_words():
    _skip_strict()
    import cupy as cp
    from woof.core import constants as c
    rng = np.random.default_rng(15)
    ny, nx, nz = 4, 7, 6
    n = ny * nx
    w_top = (rng.standard_normal(n) * 30).astype(F)
    ph_top = (rng.standard_normal(n) * 2).astype(F)
    mub = (np.full(n, 7.0e4) + rng.standard_normal(n) * 100).astype(F)
    mup = (rng.standard_normal(n) * 50).astype(F)
    mupp = (rng.standard_normal(n) * 5).astype(F)
    c1f = np.zeros(nz + 1, dtype=F)
    c1f[nz] = F(0.0)
    c2f = np.linspace(9e4, 6e4, nz + 1).astype(F)
    rhs = cp.zeros(n, dtype=cp.float32)
    dtau, eps = F(1.3333334), F(0.1)
    _k("nest_top_specified_predictor")((1,), (64,), (
        rhs, cp.asarray(w_top), cp.asarray(ph_top), cp.asarray(mub), cp.asarray(mup),
        cp.asarray(mupp), cp.asarray(c1f), cp.asarray(c2f), dtau, eps,
        np.int32(nz), np.int32(n)))
    muts = F(F(mub + mup) + mupp)
    mass = F(F(c1f[nz] * muts) + c2f[nz])
    coef = F(F(F(F(0.5) * dtau) * F(c.G)) * F(F(1.0) + eps))
    _eq(rhs, F(ph_top - F(F(coef * w_top) / mass)), "rhs_top")
