"""Partial-depth nest, Phase D2: the child's dynamics with a top flux.

kernels/nest_top.cu + woof/core/partial_nest_dyn.py.  The gates:

* self-consistency: a full column runs the EXISTING acoustic kernels; the
  child, cut at a pure-pressure level k_top and driven at its top by the
  full column's own interface values every small step, reproduces the full
  column below k_top word for word.  One reassociation is unavoidable and
  is stated exactly: the full column adds its divergence above k_top level
  by level into dmdt, the child adds the interface flux once, so the child
  is given the column-mass-tendency adjustment ``fl(D - fl(S + ww_top))``
  that turns its sum into the full column's word (and the adjustment's size
  in ulps of dmdt is asserted).  Driven by the recorded interface flux
  alone (no adjustment), the departure is bounded and asserted.
* degenerate: with a zero interface flux and lid-like top values the new
  entry points equal the existing kernels word for word.
* calc_ww with a top flux (big step), the upwinded top-face flux, the time
  interpolation and the sint interface interpolation.
"""

import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = pytest.mark.gpu

K_TOP = 14          # nz = 16, etac 0.2: full levels 13..16 are pure pressure
NZ, NY, NX = 16, 5, 12
DTAU = 0.5


def _skip_strict():
    from woof import wrf_exact
    if wrf_exact.ENABLED:
        pytest.skip("D2 top entry points are default-build only (refused by name)")


def _parent(seed, **kw):
    import cupy as cp
    from woof.verify.npref import random_acoustic_state
    s, cfg = random_acoustic_state(seed=seed, nz=NZ, ny=NY, nx=NX, hybrid_opt=2, **kw)
    # The interior of a nest: no large-step mass forcing (WRF's mu_tend is a
    # lateral-boundary term).
    s.rmu_t[...] = cp.float32(0.0)
    return s, cfg


def _words(a):
    import cupy as cp
    return np.ascontiguousarray(cp.asnumpy(a)).view(np.uint32)


def _same(got, want, what):
    g, w = _words(got), _words(want)
    bad = np.flatnonzero(g.ravel() != w.ravel())
    assert bad.size == 0, (f"{what}: {bad.size} of {g.size} words differ, first at "
                           f"{np.unravel_index(bad[0], g.shape)}")


def _adjustment(s, w, d):
    """The float32 ``adj`` with ``fl(fl(s + w) + adj) == d`` (s, w, d float32
    arrays), nearest to ``fl(d - fl(s + w))``."""
    t = (s + w).astype(np.float32)
    adj = (d - t).astype(np.float32)
    ok = (t + adj).astype(np.float32) == d
    for step in range(1, 9):
        if ok.all():
            break
        for sign in (1, -1):
            cand = adj.copy()
            for _ in range(step):
                cand = np.nextafter(cand, np.float32(sign * np.inf), dtype=np.float32)
            hit = ~ok & ((t + cand).astype(np.float32) == d)
            adj[hit] = cand[hit]
            ok |= hit
    assert ok.all(), "no float32 adjustment reproduces the column sum"
    return adj


def _ulps(x, *terms):
    """|x| in ulps of the largest of ``terms`` per column (the sums cancel,
    so the rounding scales with their terms, not with the result)."""
    scale = np.max([np.abs(t) for t in terms], axis=0).astype(np.float32)
    return np.abs(x.astype(np.float64)) / np.spacing(scale).astype(np.float64)


def _gross_pp(state, cfg, k, d_pp):
    """Sum over the levels below ``k`` of the magnitudes of the Omega''
    recurrence's terms, |dnw*div''| + |dnw*c1h*dmdt|: the scale its
    rounding grows with (float64, host)."""
    import cupy as cp
    u = cp.asnumpy(state.u_pp).astype(np.float64)[:k]
    v = cp.asnumpy(state.v_pp).astype(np.float64)[:k]
    dnw = cp.asnumpy(state.dnw).astype(np.float64)[:k]
    c1h = cp.asnumpy(state.c1h).astype(np.float64)[:k]
    div = (u[:, :, 1:] - u[:, :, :-1]) / cfg.dx + (v[:, 1:, :] - v[:, :-1, :]) / cfg.dy
    return (np.abs(dnw[:, None, None] * div).sum(axis=0)
            + np.abs(dnw * c1h).sum() * np.abs(d_pp.astype(np.float64)))


def _compare_below(child, parent, k):
    for name in ("u_pp", "v_pp", "th_pp", "p_pp", "al_pp", "p_pp_old"):
        _same(getattr(child, name), getattr(parent, name)[:k], name)
    for name in ("w_pp", "ph_pp", "ww_pp"):
        _same(getattr(child, name), getattr(parent, name)[:k + 1], name)
    _same(child.mu_pp, parent.mu_pp, "mu_pp")


def _drive(parent, cfg, n_sub, *, exact, theta_upwind=False):
    """Run ``n_sub`` existing-kernel substeps on the parent and the child's
    top-flux substeps alongside, the child driven by the parent's interface
    values.  ``exact``: also hand the child the column-sum adjustment.
    Returns (child, adjustment in eps*gross units, per-substep max |child - parent|)."""
    import cupy as cp
    from woof.core.acoustic import acoustic_substep, prepare_acoustic_coefficients
    from woof.core.partial_nest_dyn import (Interface, TopForcing, column_sums,
                                            prepare_top_substep, truncate_state)
    k = K_TOP
    coef_p = prepare_acoustic_coefficients(parent, cfg, DTAU)
    child, ccfg = truncate_state(parent, cfg, k)
    coef_c = prepare_acoustic_coefficients(child, ccfg, DTAU)
    for full, cut in zip(coef_p[:4], coef_c[:4]):          # rows below the interface
        _same(cut[:k], full[:k], "calc_coefs rows")
    iface = Interface.from_parent(parent, k)
    top = TopForcing.zeros(cfg.ny, cfg.nx)
    top.t_above[...] = parent.thb[k] + parent.thp[k]
    top.pht_above[...] = parent.phb[k + 1] + parent.php[k + 1]
    mudf_p = cp.zeros((cfg.ny, cfg.nx), dtype=cp.float32)
    mudf_c = cp.zeros((cfg.ny, cfg.nx), dtype=cp.float32)
    launch = prepare_top_substep(child, ccfg, DTAU, coef_c, iface, top, mudf=mudf_c,
                                 theta_upwind=theta_upwind)
    ulps, departures = [], []
    for n in range(n_sub):
        first = n == 0
        top.p_above[...] = parent.p_pp[k]
        top.p_above_old[...] = parent.p_pp_old[k]
        acoustic_substep(parent, cfg, DTAU, first, coef_p, mudf=mudf_p)
        top.ww_pp[...] = parent.ww_pp[k]
        top.ww_above[...] = parent.ww_pp[k + 1]
        top.w_top[...] = parent.w_pp[k]
        top.ph_top[...] = parent.ph_pp[k]
        sums = cp.asnumpy(column_sums(parent, cfg, k))
        s_pp, s_rf, d_pp, d_rf = sums[0], sums[1], sums[2], sums[3]
        # The reference flux through the interface: the full column's
        # reference divergence above it, D_rf - S_rf.
        ww_rf = (d_rf - s_rf).astype(np.float32)
        top.ww_rf[...] = cp.asarray(ww_rf)
        if exact:
            ww_pp = cp.asnumpy(top.ww_pp)
            adj_pp = _adjustment(s_pp, ww_pp, d_pp)
            adj_rf = _adjustment(s_rf, ww_rf, d_rf)
            top.adj_pp[...] = cp.asarray(adj_pp)
            top.adj_rf[...] = cp.asarray(adj_rf)
            eps = float(np.finfo(np.float32).eps)
            ulps.append(max(
                (np.abs(adj_pp) / (eps * _gross_pp(parent, cfg, k, d_pp))).max(),
                _ulps(adj_rf, s_rf, ww_rf, d_rf).max()))
        launch(first=first)
        if exact:
            _same(mudf_c, mudf_p, f"mudf (substep {n})")
            _compare_below(child, parent, k)
        else:
            departures.append({
                name: float(cp.max(cp.abs(getattr(child, name)
                                          - getattr(parent, name)[:getattr(child, name).shape[0]]))
                            / (cp.max(cp.abs(getattr(parent, name))) + 1e-30))
                for name in ("u_pp", "v_pp", "w_pp", "ph_pp", "th_pp", "mu_pp", "p_pp")})
    return child, ulps, departures


@requires_gpu
@pytest.mark.parametrize("seed", [3, 17])
def test_child_reproduces_the_full_column_below_the_interface_word_for_word(seed):
    _skip_strict()
    parent, cfg = _parent(seed)
    _, ulps, _ = _drive(parent, cfg, 12, exact=True)
    # The column-sum reassociation is a rounding, not a flux: a few ulps of
    # dmdt (each sum boundary rounds once; the interface flux is one term).
    # The adjustment is the rounding of the two association orders: in units
    # of eps times the recurrence's gross column flux, at most one per
    # rounded level in theory (k_top), about one measured.
    assert max(ulps) <= K_TOP, ulps


@requires_gpu
def test_child_reproduces_a_terrain_and_moist_column():
    """Terrain (3-D base profiles, kinematic lower boundary) and moist cq."""
    _skip_strict()
    import cupy as cp
    parent, cfg = _parent(29, hill_height=400.0, moist=True, mp_physics=1)
    rng = np.random.default_rng(29)
    for name, amp in (("qv", 0.012), ("qc", 0.001), ("qr", 0.0005)):
        getattr(parent, name)[...] = cp.asarray(
            amp * rng.random(parent.qv.shape), dtype=cp.float32)
    _drive(parent, cfg, 8, exact=True)


def _warm_bubble():
    """A small 3-D warm bubble in a uniform 10 m/s flow on WRF's hybrid
    coordinate (etac 0.2), stepped by the existing dycore."""
    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_theta_perturbation
    cfg = RunConfig(nx=24, ny=6, nz=20, dx=1000.0, dy=1000.0, ztop=12000.0,
                    dt=6.0, run_seconds=0.0, hybrid_opt=2)
    coord = make_vertical_coord(cfg.nz, hybrid_opt=2, etac=cfg.etac)
    base = make_base_state(coord, lambda z: 300.0 * np.exp(1e-4 * np.asarray(z) / 9.81),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    y = (np.arange(cfg.ny) + 0.5) * cfg.dy - 0.5 * cfg.ny * cfg.dy

    def thp_func(x, z):
        r = np.sqrt((x[None, None, :] / 4000.0) ** 2 + (y[None, :, None] / 4000.0) ** 2
                    + ((z[:, None, None] - 2000.0) / 1500.0) ** 2)
        return np.where(r < 1.0, 2.0 * np.cos(0.5 * np.pi * r) ** 2, 0.0)

    state = init_theta_perturbation(cfg, coord, base, thp_func)
    state.u[...] = 10.0
    return state, cfg, coord


@requires_gpu
def test_child_reproduces_a_dycore_warm_bubble_stage(monkeypatch):
    """The full column is the existing dycore (woof.core.dycore.step, its
    own acoustic launches, fused or separate) on a rising warm bubble; the
    third RK stage of the fourth step is recorded at the interface, and the
    child, cut there and built from the stage's opening state, reproduces
    every substep below the interface word for word."""
    _skip_strict()
    import copy
    import cupy as cp
    import woof.core.dycore as dycore
    from woof.core.acoustic import prepare_acoustic_coefficients
    from woof.core.partial_nest_dyn import (Interface, TopForcing, column_sums,
                                            prepare_top_substep, truncate_state)
    state, cfg, coord = _warm_bubble()
    k = int(np.flatnonzero(coord.c3f == 0.0)[0]) + 1
    assert k <= cfg.nz - 2
    original = dycore.prepare_acoustic_substep_launch
    target = 3 * 3 + 3                       # step 4, stage 3
    calls, rec = [0], {"subs": []}

    def hooked(st, cf, dtau, coefficients, **kw):
        launch = original(st, cf, dtau, coefficients, **kw)
        calls[0] += 1
        if calls[0] != target:
            return launch

        def recorded(*, first):
            if first:
                snap = copy.copy(st)
                for name, value in vars(st).items():
                    if isinstance(value, cp.ndarray):
                        setattr(snap, name, value.copy())
                rec["snap"], rec["dtau"] = snap, dtau
                rec["coef"] = tuple(c.copy() for c in coefficients[:4])
            sub = {"p_above": st.p_pp[k].copy(), "p_above_old": st.p_pp_old[k].copy()}
            launch(first=first)
            sub.update(ww_pp=st.ww_pp[k].copy(), ww_above=st.ww_pp[k + 1].copy(),
                       w_top=st.w_pp[k].copy(), ph_top=st.ph_pp[k].copy(),
                       sums=cp.asnumpy(column_sums(st, cf, k)),
                       below={n: getattr(st, n)[:k + (n in ("w_pp", "ph_pp", "ww_pp"))].copy()
                              for n in ("u_pp", "v_pp", "th_pp", "p_pp", "al_pp",
                                        "w_pp", "ph_pp", "ww_pp", "mu_pp")})
            sub["below"]["mu_pp"] = st.mu_pp.copy()
            rec["subs"].append(sub)

        for name in ("absorbs", "joined", "joined_written"):
            setattr(recorded, name, getattr(launch, name))
        return recorded

    monkeypatch.setattr(dycore, "prepare_acoustic_substep_launch", hooked)
    dycore.run_steps(state, cfg, 4)
    monkeypatch.undo()
    subs = rec["subs"]
    assert len(subs) >= 4
    parent = rec["snap"]
    flux = max(float(cp.max(cp.abs(s["ww_pp"]))) for s in subs)
    assert flux > 0.0, "the bubble must drive a flux through the interface"

    child, ccfg = truncate_state(parent, cfg, k)
    coef = prepare_acoustic_coefficients(child, ccfg, rec["dtau"])
    for full, cut in zip(rec["coef"], coef[:4]):
        _same(cut[:k], full[:k], "calc_coefs rows")
    iface = Interface.from_parent(parent, k)
    top = TopForcing.zeros(cfg.ny, cfg.nx)
    top.t_above[...] = parent.thb[k] + parent.thp[k]
    top.pht_above[...] = parent.phb[k + 1] + parent.php[k + 1]
    launch = prepare_top_substep(child, ccfg, rec["dtau"], coef, iface, top)
    for n, sub in enumerate(subs):
        for name in ("p_above", "p_above_old", "ww_pp", "ww_above", "w_top", "ph_top"):
            getattr(top, name)[...] = sub[name]
        s_pp, s_rf, d_pp, d_rf = sub["sums"][:4]
        ww_rf = (d_rf - s_rf).astype(np.float32)
        top.ww_rf[...] = cp.asarray(ww_rf)
        top.adj_pp[...] = cp.asarray(_adjustment(s_pp, cp.asnumpy(sub["ww_pp"]), d_pp))
        top.adj_rf[...] = cp.asarray(_adjustment(s_rf, ww_rf, d_rf))
        launch(first=n == 0)
        for name, want in sub["below"].items():
            _same(getattr(child, name), want, f"{name} (substep {n})")


@requires_gpu
def test_recorded_interface_flux_alone_departs_by_rounding_only():
    """Without the adjustment the child sees the physical interface flux; the
    only difference from the full column is the reassociated column sum,
    which stays at float32 rounding over a stage of substeps."""
    _skip_strict()
    parent, cfg = _parent(5)
    _, _, departures = _drive(parent, cfg, 12, exact=False)
    worst = {k: max(d[k] for d in departures) for k in departures[0]}
    # Measured (seed 5, 12 substeps): mu'' 7e-8, w'' 3e-7, u''/v'' 3.5e-6,
    # phi'' 6e-6, p'' 1e-5, (mu theta)'' 5e-5 of each field's magnitude --
    # (mu theta)'' carries the Omega'' difference times theta_t* (~300 K).
    assert worst["mu_pp"] < 1e-6, worst
    assert all(v < 2e-4 for v in worst.values()), worst
    assert any(v > 0.0 for v in worst.values()), "the reassociation should be visible"


@requires_gpu
def test_a_wrong_sign_interface_flux_is_seen():
    """Non-vacuity: flipping the interface flux's sign breaks the child."""
    _skip_strict()
    import cupy as cp
    from woof.core.acoustic import acoustic_substep, prepare_acoustic_coefficients
    from woof.core.partial_nest_dyn import (Interface, TopForcing,
                                            prepare_top_substep, truncate_state)
    parent, cfg = _parent(3)
    k = K_TOP
    coef_p = prepare_acoustic_coefficients(parent, cfg, DTAU)
    child, ccfg = truncate_state(parent, cfg, k)
    coef_c = prepare_acoustic_coefficients(child, ccfg, DTAU)
    iface = Interface.from_parent(parent, k)
    top = TopForcing.zeros(cfg.ny, cfg.nx)
    top.p_above[...] = parent.p_pp[k]
    top.p_above_old[...] = parent.p_pp_old[k]
    top.t_above[...] = parent.thb[k] + parent.thp[k]
    acoustic_substep(parent, cfg, DTAU, True, coef_p)
    top.ww_pp[...] = -parent.ww_pp[k]
    top.w_top[...] = parent.w_pp[k]
    top.ph_top[...] = parent.ph_pp[k]
    prepare_top_substep(child, ccfg, DTAU, coef_c, iface, top)(first=True)
    err = float(cp.max(cp.abs(child.mu_pp - parent.mu_pp)))
    assert err > 1e-3 * float(cp.max(cp.abs(parent.ww_pp[k]))) * DTAU


@requires_gpu
def test_zero_flux_and_a_lid_reduce_to_the_existing_kernels_word_for_word():
    """The degenerate gate: the new entry points with nothing through the top
    are the existing kernels.  advance_uv_top with fnm_top = fnp_top = 0 is
    advance_uv's open top (dpn = 0 at the top); advance_mu_th_top with zero
    flux and adjustment is advance_mu_th; advance_w_phi_top with a zero
    predictor, w_top = ph_top = 0 is advance_w_phi under top_lid = 1."""
    _skip_strict()
    import dataclasses
    import cupy as cp
    from woof.core.acoustic import (acoustic_substep_explicit,
                                    prepare_acoustic_coefficients)
    from woof.core.kernels import get_kernel
    from woof.core.partial_nest_dyn import (Interface, TopForcing,
                                            prepare_top_substep)
    from woof.verify.npref import random_acoustic_state

    a, cfg = random_acoustic_state(seed=41, nz=NZ, ny=NY, nx=NX, hybrid_opt=2)
    b, _ = random_acoustic_state(seed=41, nz=NZ, ny=NY, nx=NX, hybrid_opt=2)
    open_cfg = dataclasses.replace(cfg, top_lid=0)
    lid_cfg = dataclasses.replace(cfg, top_lid=1)
    # a: existing kernels.  The explicit part on the open top (uv + mu),
    # then the lid's coefficients and solve.
    mudf_a = cp.zeros((NY, NX), dtype=cp.float32)
    mudf_b = cp.zeros((NY, NX), dtype=cp.float32)
    for first in (True, False):
        coef_a = prepare_acoustic_coefficients(a, lid_cfg, DTAU)
        acoustic_substep_explicit(a, open_cfg, DTAU, first, mudf=mudf_a)
        mu_old = a.scratch((NY, NX), "acoustic_mu_pp_old")
        th_old = a.scratch((NZ, NY, NX), "acoustic_th_pp_old")
        c2a, ca, calpha, cgam = coef_a
        get_kernel("acoustic", "advance_w_phi")(
            ((NY * NX + 255) // 256,), (256,),
            (a.w_pp, a.ph_pp, a.rw_t, a.rph_t, a.ww_pp, a.mu_pp, mu_old,
             a.th_pp, th_old, a.thp, a.thb, a.php, a.phb, a.alt, c2a,
             ca, calpha, cgam, a.mup, a.u_pp, a.v_pp, a.w, a.ht,
             a.rdn, a.rdnw, a.fnm, a.fnp, a.c1h, a.c2h, a.c1f, a.c2f, a.mub2d,
             a.p, np.int32(0), a.p_pp, a.al_pp, a.cf1, a.cf2, a.cf3,
             np.float32(1.0 / cfg.dx), np.float32(1.0 / cfg.dy), np.float32(DTAU),
             np.float32(cfg.epssm), np.float32(0.0), np.float32(cfg.zdamp),
             np.int32(0), np.int32(0), np.int32(0), np.int32(0), np.int32(1),
             np.int32(NZ), np.int32(NY), np.int32(NX)))
        # b: the new entry points with nothing through the top.
        coef_b = prepare_acoustic_coefficients(b, lid_cfg, DTAU)
        iface = Interface(NZ, np.float32(0.0), np.float32(0.0), np.float32(0.0))
        top = TopForcing.zeros(NY, NX)
        prepare_top_substep(b, open_cfg, DTAU, coef_b, iface, top, mudf=mudf_b,
                            rigid_predictor=True)(first=first)
        for name in ("u_pp", "v_pp", "mu_pp", "th_pp", "ww_pp", "w_pp", "ph_pp",
                     "p_pp", "al_pp", "p_pp_old"):
            _same(getattr(b, name), getattr(a, name), f"{name} (first={first})")
        _same(mudf_b, mudf_a, "mudf")


def _omega_parent(seed):
    import cupy as cp
    from woof.core.dycore import stage_fluxes
    parent, cfg = _parent(seed)
    ru, rv, ww = stage_fluxes(parent, cfg)
    return parent, cfg, ru.copy(), rv.copy(), ww.copy()


@requires_gpu
def test_calc_ww_with_a_top_flux_reproduces_the_full_column():
    """Big step: WRF calc_ww_cp on the child with the full column's interface
    Omega (and the column-sum adjustment) is the full column's Omega below
    the interface, word for word; without the adjustment it departs by
    rounding; with a zero flux it is the existing kernel on a lid column."""
    _skip_strict()
    import cupy as cp
    from woof.core.partial_nest_dyn import calc_ww_top, column_sums, truncate_state
    k = K_TOP
    parent, cfg, ru, rv, ww = _omega_parent(11)
    child, ccfg = truncate_state(parent, cfg, k)
    sums = cp.asnumpy(column_sums(parent, cfg, k, ru, rv))
    ww_k = cp.asnumpy(ww[k])
    adj = _adjustment(sums[4], ww_k, sums[5])
    assert _ulps(adj, sums[4], ww_k, sums[5]).max() <= 4.0
    got = calc_ww_top(child, ccfg, ru[:k].copy(), rv[:k].copy(), ww[k].copy(),
                      cp.asarray(adj))
    _same(got, ww[:k + 1], "ww")
    plain = calc_ww_top(child, ccfg, ru[:k].copy(), rv[:k].copy(), ww[k].copy()).copy()
    rel = float(cp.max(cp.abs(plain - ww[:k + 1])) / cp.max(cp.abs(ww[:k + 1])))
    assert rel < 1e-5
    # zero flux: the existing kernel's column on the child's own levels.
    from woof.core.dycore import _omega_ref
    want = _omega_ref(child, ccfg, ru[:k].copy(), rv[:k].copy()).copy()
    zero = calc_ww_top(child, ccfg, ru[:k].copy(), rv[:k].copy(),
                       cp.zeros((NY, NX), dtype=cp.float32))
    _same(zero, want, "zero-flux ww")


@requires_gpu
@pytest.mark.parametrize("stagger", ["", "x", "y"])
def test_upwinded_top_face_flux(stagger):
    """The top-face flux enters the top half level's tendency upwinded on
    the face Omega: the parent's value above for inflow (Omega > 0, down),
    the child's own top value for outflow; a zero flux leaves the tendency's
    words alone; the column-integrated tendency changes by exactly the face
    flux (rdnw * dnw = 1)."""
    _skip_strict()
    import cupy as cp
    from woof.core.partial_nest_dyn import Interface, add_top_face_flux, truncate_state
    parent, cfg = _parent(13)
    k = K_TOP
    child, ccfg = truncate_state(parent, cfg, k)
    iface = Interface.from_parent(parent, k)
    rng = np.random.default_rng(1)
    shape = {"": (k, NY, NX), "x": (k, NY, NX + 1), "y": (k, NY + 1, NX)}[stagger]
    q = cp.asarray(rng.standard_normal(shape), dtype=cp.float32)
    q_above = cp.asarray(rng.standard_normal(shape[1:]), dtype=cp.float32)
    ww_top = cp.asarray(rng.standard_normal((NY, NX)), dtype=cp.float32)
    tend0 = cp.asarray(rng.standard_normal(shape), dtype=cp.float32)

    tend = tend0.copy()
    add_top_face_flux(tend, q, q_above, cp.zeros_like(ww_top), child, ccfg, iface, stagger)
    _same(tend, tend0, "zero-flux tendency")

    tend = tend0.copy()
    flux = cp.asnumpy(add_top_face_flux(tend, q, q_above, ww_top, child, ccfg, iface, stagger))
    w = cp.asnumpy(ww_top).astype(np.float64)
    if stagger == "x":
        vel = 0.5 * (np.roll(w, 1, axis=1) + w)
        vel = np.concatenate([vel, vel[:, :1]], axis=1)
    elif stagger == "y":
        vel = 0.5 * (np.roll(w, 1, axis=0) + w)
        vel = np.concatenate([vel, vel[:1]], axis=0)
    else:
        vel = w
    qa, qt = cp.asnumpy(q_above).astype(np.float64), cp.asnumpy(q[-1]).astype(np.float64)
    want_flux = vel * np.where(vel > 0, qa, qt)
    np.testing.assert_allclose(flux, want_flux, rtol=2e-6, atol=1e-7)
    assert (vel > 0).any() and (vel < 0).any()
    rdnw = float(cp.asnumpy(child.rdnw)[k - 1])
    got = cp.asnumpy(tend - tend0)
    np.testing.assert_allclose(got[-1], -rdnw * flux, rtol=1e-5, atol=1e-5)
    assert not got[:-1].any()


@requires_gpu
def test_upwinded_theta_flux_switches_on_the_total_interface_flux():
    """advance_mu_th_top's upwind option: the top half level's theta flux
    takes theta above for a downward total interface flux and its own for
    an upward one; the centered option is the parent's face value."""
    _skip_strict()
    import cupy as cp
    from woof.core.acoustic import prepare_acoustic_coefficients
    from woof.core.partial_nest_dyn import (Interface, TopForcing,
                                            prepare_top_substep, truncate_state)
    parent, cfg = _parent(19)
    k = K_TOP
    results = {}
    for upwind in (False, True):
        for sign in (1.0, -1.0):
            child, ccfg = truncate_state(parent, cfg, k)
            coef = prepare_acoustic_coefficients(child, ccfg, DTAU)
            top = TopForcing.zeros(NY, NX)
            top.ww_rf[...] = cp.float32(sign * 50.0)
            top.ww_pp[...] = cp.float32(sign * 0.5)
            top.t_above[...] = cp.float32(500.0)        # far from the child's theta
            th0 = child.th_pp.copy()
            launch = prepare_top_substep(child, ccfg, DTAU, coef,
                                         Interface.from_parent(parent, k), top,
                                         theta_upwind=upwind)
            launch(first=True)
            results[(upwind, sign)] = cp.asnumpy(child.th_pp[k - 1] - th0[k - 1])
    # downward total flux, upwind: theta_above (500) enters, more than the
    # centered average does; upward: the child's own theta leaves.
    rdnw = float(cp.asnumpy(parent.rdnw)[k - 1])
    t_dn = cp.asnumpy(parent.thb[k - 1] + parent.thp[k - 1])
    fnm, fnp = (float(cp.asnumpy(parent.fnm)[k]), float(cp.asnumpy(parent.fnp)[k]))
    d_cent = results[(True, 1.0)] - results[(False, 1.0)]
    want = -DTAU * rdnw * 0.5 * (500.0 - (fnm * 500.0 + fnp * t_dn))
    np.testing.assert_allclose(d_cent, want, rtol=2e-3, atol=2e-3)
    d_up = results[(True, -1.0)] - results[(False, -1.0)]
    want_up = -DTAU * rdnw * (-0.5) * (t_dn - (fnm * 500.0 + fnp * t_dn))
    np.testing.assert_allclose(d_up, want_up, rtol=2e-3, atol=5e-3)


@requires_gpu
def test_time_interpolation_across_the_parent_step():
    _skip_strict()
    import cupy as cp
    from woof.core.partial_nest_dyn import time_interpolate
    rng = np.random.default_rng(7)
    a = rng.standard_normal((NY, NX)).astype(np.float32)
    b = rng.standard_normal((NY, NX)).astype(np.float32)
    for frac in (0.0, 0.25, 1.0 / 3.0, 1.0):
        got = cp.asnumpy(time_interpolate(cp.asarray(a), cp.asarray(b), frac))
        f = np.float32(frac)
        want = (a + (f * (b - a).astype(np.float32)).astype(np.float32)).astype(np.float32)
        np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


@requires_gpu
def test_interface_flux_interpolates_with_sint():
    """A uniform parent interface flux stays uniform on the child (sint is
    exact on constants), and the child field is the nest sint operator's."""
    _skip_strict()
    import cupy as cp
    from woof.core.nest_interp import register_nest, sint
    from woof.core.partial_nest_dyn import interpolate_top_field
    reg = register_nest(nri=3, nrj=3, i_parent_start=4, j_parent_start=4,
                        child_nx=15, child_ny=12, parent_nx=16, parent_ny=14,
                        stagger="")
    parent = cp.full((14, 16), 7.25, dtype=cp.float32)
    got = cp.asnumpy(interpolate_top_field(parent, reg))
    assert got.shape == (12, 15) and np.all(got == np.float32(7.25))
    rng = np.random.default_rng(3)
    field = cp.asarray(rng.standard_normal((14, 16)), dtype=cp.float32)
    _same(interpolate_top_field(field, reg), sint(field, reg), "sint")


@requires_gpu
def test_shifted_child_coordinate_departs_by_rounding_only():
    """D1's truncated coordinate (mu' = mu - (p_i - p_top), re-expressed c2)
    is the same column in other variables: driven by the same interface
    values its substeps differ from the parent's words only by the rounding
    of the re-expressed mass coupling (bounded here); the word-exact oracle
    above therefore runs the child in the parent's own variables."""
    _skip_strict()
    import cupy as cp
    from woof.core.acoustic import acoustic_substep, prepare_acoustic_coefficients
    from woof.core.grid import finalize_vertical_coord, make_vertical_coord
    from woof.core.partial_nest_dyn import (Interface, TopForcing, column_sums,
                                            prepare_top_substep, truncate_state)
    parent, cfg = _parent(23)
    k = K_TOP
    coord = make_vertical_coord(NZ, hybrid_opt=2, etac=cfg.etac)
    finalize_vertical_coord(coord, float(parent.p_top))
    coef_p = prepare_acoustic_coefficients(parent, cfg, DTAU)
    child, ccfg = truncate_state(parent, cfg, k, coord=coord)
    assert float(child.p_top) > float(parent.p_top)
    coef_c = prepare_acoustic_coefficients(child, ccfg, DTAU)
    iface = Interface.from_parent(parent, k)
    top = TopForcing.zeros(NY, NX)
    top.t_above[...] = parent.thb[k] + parent.thp[k]
    top.pht_above[...] = parent.phb[k + 1] + parent.php[k + 1]
    launch = prepare_top_substep(child, ccfg, DTAU, coef_c, iface, top)
    for n in range(8):
        top.p_above[...] = parent.p_pp[k]
        top.p_above_old[...] = parent.p_pp_old[k]
        acoustic_substep(parent, cfg, DTAU, n == 0, coef_p)
        top.ww_pp[...] = parent.ww_pp[k]
        top.ww_above[...] = parent.ww_pp[k + 1]
        top.w_top[...] = parent.w_pp[k]
        top.ph_top[...] = parent.ph_pp[k]
        sums = column_sums(parent, cfg, k)
        top.ww_rf[...] = sums[3] - sums[1]
        launch(first=n == 0)
    for name in ("u_pp", "w_pp", "ph_pp", "th_pp", "mu_pp", "p_pp"):
        got = getattr(child, name)
        want = getattr(parent, name)[:got.shape[0]]
        rel = float(cp.max(cp.abs(got - want)) / cp.max(cp.abs(want)))
        assert rel < 1e-4, (name, rel)


def test_strict_and_unsupported_configurations_are_refused_by_name():
    import dataclasses
    from woof.config import RunConfig
    from woof.core.partial_nest import PartialNestRefused
    from woof.core.partial_nest_dyn import refuse_unsupported
    cfg = RunConfig(nx=8, ny=4, nz=10, dx=500.0, dy=500.0, ztop=8000.0, dt=3.0,
                    run_seconds=0.0, hybrid_opt=2)
    with pytest.raises(PartialNestRefused, match="damp_opt=3"):
        refuse_unsupported(dataclasses.replace(cfg, damp_opt=3))
    with pytest.raises(PartialNestRefused, match="lateral"):
        refuse_unsupported(dataclasses.replace(cfg, open_x=True))


def test_the_acoustic_unit_is_untouched_and_composed_first():
    """The new module is acoustic.cu's assembled text followed by
    nest_top.cu: the existing unit (and so every existing launch) keeps its
    bytes, and the new entry points call its device helpers verbatim."""
    from pathlib import Path
    from woof.core.kernels import module_source
    nest = module_source("nest_top")
    own = (Path(__file__).resolve().parents[1] / "woof/core/kernels/nest_top.cu").read_text(
        encoding="utf-8")
    assert nest == module_source("acoustic") + own
