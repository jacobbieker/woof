"""Two-way feedback from a nest on its own eta ladder, below a height.

WRF v4.7.1 refuses feedback with vertical refinement
(share/module_check_a_mundo.F:2719-2737), so there is no oracle to hold
woof/core/nest_feedback.py to.  What is held instead:

* the device kernels to their NumPy twins at max ULP 0;
* the ladder remap to conservation, column by column, and the whole
  transaction to it over the restriction rectangle (layer dry mass and
  water of the parent == the child's over the same layer, to rounding);
* identity: a nest on the parent's own levels with a height limit above
  the model top is today's WRF feedback, word for word;
* the height limit: above it the parent's words are not touched;
* restart continuation, sha-identical;
* the configuration, its refusals and its absence from the identities.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

from woof.core import nest_feedback as fb
from woof.core.model import FeedbackScratch

F = np.float32


def _ladder(nz, rng, *, etac=0.2, span=95000.0):
    inner = np.sort(rng.uniform(0.02, 0.98, nz - 1))[::-1]
    eta = np.concatenate([[1.0], inner, [0.0]])
    b = np.where(eta < etac, 0.0, ((eta - etac) / (1.0 - etac)) ** 2)
    c3 = b.astype(F)
    c4 = ((eta - b) * span).astype(F)
    c3[0], c4[0], c3[-1], c4[-1] = 1.0, 0.0, 0.0, 0.0
    return c3, c4


# -- the operators ----------------------------------------------------------


def test_layer_remap_conserves_each_column():
    rng = np.random.default_rng(3)
    child, parent = _ladder(61, rng), _ladder(29, rng)
    ny, nx = 7, 9
    mub = rng.uniform(70000.0, 95000.0, (ny, nx)).astype(F)
    mup = rng.uniform(-800.0, 800.0, (ny, nx)).astype(F)
    q = rng.uniform(0.0, 0.02, (61, ny, nx)).astype(F)
    content = fb.remap_layers_cpu(q, mub, mup, child, parent)
    mu = mub.astype(np.float64) + mup
    edges = [fb._edge_cpu(*child, j, 61, mu) for j in range(62)]
    want = sum((edges[j] - edges[j + 1]) * q[j] for j in range(61))
    got = content.astype(np.float64).sum(axis=0)
    np.testing.assert_allclose(got, want, rtol=2e-7)
    # And a constant is carried exactly as the parent layer's own mass
    # times it: the overlaps of a parent layer sum to its mass.
    ones = fb.remap_layers_cpu(np.ones_like(q), mub, mup, child, parent)
    pedges = [fb._edge_cpu(*parent, k, 29, mu) for k in range(30)]
    dm = np.stack([pedges[k] - pedges[k + 1] for k in range(29)])
    np.testing.assert_allclose(ones, dm, rtol=1e-7)


def test_remap_onto_the_same_ladder_is_the_identity_on_interfaces():
    rng = np.random.default_rng(4)
    ladder = _ladder(20, rng)
    mub = rng.uniform(70000.0, 95000.0, (5, 6)).astype(F)
    mup = np.zeros_like(mub)
    w = rng.normal(size=(21, 5, 6)).astype(F)
    out = fb.remap_layers_cpu(w, mub, mup, ladder, ladder, interface=True)
    np.testing.assert_array_equal(out, w)


def test_level_weights_taper_and_full_column():
    w, wi = fb.level_weights(10, 6, 3)
    np.testing.assert_array_equal(
        w, F([1, 1, 1, 0.75, 0.5, 0.25, 0, 0, 0, 0]))
    assert wi[0] == 1 and wi[10] == 0 and wi[4] == F(0.625)
    w, wi = fb.level_weights(10, 10, 3)
    assert (w == 1).all() and (wi == 1).all()
    w, _ = fb.level_weights(10, 2, 3)
    np.testing.assert_array_equal(
        w, F([2 / 3, 1 / 3, 0, 0, 0, 0, 0, 0, 0, 0]))
    w, _ = fb.level_weights(10, 4, 0)
    np.testing.assert_array_equal(w, F([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]))


def test_levels_below_reads_every_column_above_its_own_ground():
    phb = np.zeros((6, 3, 3))
    for k in range(6):
        phb[k] = 9.81 * (100.0 * k)
    phb[:, 1, 1] += 9.81 * 500.0            # a hill: AGL is unchanged
    assert fb.levels_below(phb, 250.0, (0, 0, 3, 3)) == 2
    phb[2, 2, 2] = 9.81 * 260.0             # one deeper column decides
    assert fb.levels_below(phb, 250.0, (0, 0, 3, 3)) == 1
    assert fb.levels_below(phb, 250.0, (0, 0, 2, 2)) == 2
    assert fb.levels_below(phb[:, 0, 0], 1e6, None) == 5


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("stag", ["mass", "x", "y", "interface"])
def test_device_kernels_match_their_twins_bit_for_bit(stag):
    import cupy as cp

    rng = np.random.default_rng(7)
    child, parent = _ladder(57, rng), _ladder(33, rng)
    ny, nx = 13, 17
    mub = rng.uniform(70000.0, 95000.0, (ny, nx)).astype(F)
    mup = rng.uniform(-500.0, 500.0, (ny, nx)).astype(F)
    xs, ys = int(stag == "x"), int(stag == "y")
    interface = stag == "interface"
    src = rng.normal(size=(57 + interface, ny + ys, nx + xs)).astype(F)
    host = fb.remap_layers_cpu(src, mub, mup, child, parent, xstag=xs,
                               ystag=ys, interface=interface)
    dev = fb.remap_layers(cp.asarray(src), cp.asarray(mub),
                          cp.asarray(mup), child, parent, xstag=xs,
                          ystag=ys, interface=interface)
    np.testing.assert_array_equal(host.view(np.int32),
                                  cp.asnumpy(dev).view(np.int32))
    if not interface:
        mass = fb.remap_layers_cpu(src, mub, mup, child, parent, xstag=xs,
                                   ystag=ys, mass=True)
        dmass = fb.remap_layers(cp.asarray(src), cp.asarray(mub),
                                cp.asarray(mup), child, parent, xstag=xs,
                                ystag=ys, mass=True)
        np.testing.assert_array_equal(mass.view(np.int32),
                                      cp.asnumpy(dmass).view(np.int32))
    window = (2, 1, nx - 4, ny - 3)
    weights, wi = fb.level_weights(33, 20, 3)
    weights = wi if interface else weights
    base = rng.normal(size=host.shape).astype(F)
    dbase = cp.asarray(base)
    fb.blend_cpu(base, host, weights, window)
    fb.blend(dbase, dev, weights, window)
    np.testing.assert_array_equal(base.view(np.int32),
                                  cp.asnumpy(dbase).view(np.int32))


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("flat", [False, True])
def test_ph_column_matches_its_twin_bit_for_bit(flat):
    import cupy as cp

    rng = np.random.default_rng(9)
    ny, nx = 11, 13
    ph = rng.normal(0.0, 50.0, (31, ny, nx)).astype(F)
    fed = (ph + rng.normal(0.0, 5.0, ph.shape)).astype(F)
    phb = (np.linspace(0.0, 2.0e5, 31) if flat else
           np.linspace(0.0, 2.0e5, 31)[:, None, None]
           + rng.uniform(0, 3000.0, (1, ny, nx))).astype(F)
    w, wi = fb.level_weights(30, 12, 3)
    window = (1, 2, nx - 3, ny - 4)
    host = ph.copy()
    fb.ph_column_cpu(host, fed, phb, w, wi, window)
    dev = cp.asarray(ph)
    fb.ph_column(dev, cp.asarray(fed), cp.asarray(phb), w, wi, window)
    np.testing.assert_array_equal(host.view(np.int32),
                                  cp.asnumpy(dev).view(np.int32))
    assert not np.array_equal(host, ph)


# -- the transaction --------------------------------------------------------


def _domain(grid_id, parent_id, *, nx, ny, nz, ratio=1, smooth=False):
    from woof.config import RunConfig
    from woof.experiment import DomainConfig

    root = parent_id == 0
    run = RunConfig(
        nx=nx, ny=ny, nz=nz, dx=3000.0 if root else 3000.0 / ratio,
        dy=3000.0 if root else 3000.0 / ratio, ztop=12000.0,
        dt=30.0 if root else 30.0 / ratio, run_seconds=30.0,
        output_interval_s=30.0, grid_id=grid_id, specified=root,
        nested=not root, moist=True, mp_physics=0, spec_bdy_width=5,
        spec_zone=1, relax_zone=4)
    return DomainConfig(
        grid_id=grid_id, parent_id=parent_id,
        i_parent_start=1 if root else 4, j_parent_start=1 if root else 4,
        parent_grid_ratio=1 if root else ratio,
        parent_time_step_ratio=1 if root else ratio,
        history_interval_s=30.0, run=run, time_step=30 if root else None)


def _seed(state, nz, *, child, rng):
    import cupy as cp

    eta = np.linspace(1.0, 0.0, nz + 1).astype(F)
    state.c3f[...] = cp.asarray(eta)
    state.c4f[...] = cp.float32(0.0)
    state.c1h[...] = cp.float32(1.0)
    state.c1f[...] = cp.float32(1.0)
    state.mub2d[...] = cp.float32(90000.0)
    state.thb[...] = cp.float32(300.0)
    state.p_top = np.float32(10000.0)
    for k in range(nz + 1):
        state.phb[k, ...] = cp.float32(k * 9.81 * 1000.0)
    shape = state.mup.shape
    sign = 1.0 if child else -1.0
    state.mup[...] = cp.asarray(rng.uniform(-300.0, 300.0, shape)
                                .astype(F)) if child else cp.float32(0.0)
    for name in ("u", "v", "w", "php", "thp"):
        arr = getattr(state, name)
        arr[...] = cp.asarray((sign * rng.uniform(1.0, 5.0, arr.shape))
                              .astype(F))
    for name, scale in (("qv", 0.01), ("qc", 0.001), ("qr", 0.002)):
        arr = getattr(state, name)
        arr[...] = cp.asarray(rng.uniform(0.0, scale, arr.shape).astype(F))


def _edge(*, nz_parent=6, nz_child=6, feedback_top=None, smooth=0,
          ratio=3, seed=11, child_mub=90000.0):
    from woof.core.nest import NestCoupler
    from woof.core.state import DomainState

    rng = np.random.default_rng(seed)
    parent_cfg = _domain(1, 0, nx=20, ny=20, nz=nz_parent)
    child_cfg = _domain(2, 1, nx=8 * ratio, ny=8 * ratio, nz=nz_child,
                        ratio=ratio)
    child_cfg = replace(child_cfg, feedback_top=feedback_top)
    pstate, cstate = DomainState(parent_cfg.run), DomainState(child_cfg.run)
    _seed(pstate, nz_parent, child=False, rng=rng)
    _seed(cstate, nz_child, child=True, rng=rng)
    cstate.mub2d[...] = _cp().float32(child_mub)
    clock = SimpleNamespace(ticks=0)
    parent = SimpleNamespace(cfg=parent_cfg, state=pstate, clock=clock)
    child = SimpleNamespace(cfg=child_cfg, state=cstate, clock=clock,
                            parent=parent)
    coupler = NestCoupler(child, feedback=1, smooth_option=smooth)
    return coupler, child, pstate, cstate


def _cp():
    import cupy

    return cupy


def _feed(coupler, child):
    scratch = FeedbackScratch()
    coupler.feedback_prepare(child, scratch)
    coupler.feedback_commit(child)
    coupler.feedback_finalize(child)


_FIELDS = ("mup", "u", "v", "w", "php", "thp", "qv", "qc", "qr")


def _snapshot(state):
    import cupy as cp

    return {name: cp.asnumpy(getattr(state, name)).copy()
            for name in _FIELDS}


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("smooth", [0, 1, 2])
def test_a_limit_above_the_model_top_is_todays_feedback_word_for_word(
        smooth):
    from woof.experiment import FeedbackTop

    from woof.core.nest_interp import smoother_parent_window

    coupler, child, pstate, _ = _edge(smooth=smooth)
    assert not coupler.partial_feedback
    assert smoother_parent_window(coupler.registrations["m"])[2] > 0
    _feed(coupler, child)
    reference = _snapshot(pstate)

    coupler, child, pstate, _ = _edge(
        smooth=smooth, feedback_top=FeedbackTop(1.0e6, 3))
    assert coupler.partial_feedback and not coupler.crosses_ladders
    _feed(coupler, child)
    assert coupler.feedback_levels_below == 6
    got = _snapshot(pstate)
    for name in _FIELDS:
        np.testing.assert_array_equal(got[name].view(np.int32),
                                      reference[name].view(np.int32),
                                      err_msg=name)


@requires_gpu
@pytest.mark.gpu
def test_above_the_limit_the_parent_is_untouched_and_below_it_is_fed():
    from woof.experiment import FeedbackTop

    coupler, child, pstate, _ = _edge(smooth=2)
    _feed(coupler, child)
    full = _snapshot(pstate)

    # Layer tops at 1000 m steps AGL: 3500 m keeps layers 0..2, the top
    # one of them tapered at 1/2.
    coupler, child, pstate, _ = _edge(
        smooth=2, feedback_top=FeedbackTop(3500.0, 1))
    before = _snapshot(pstate)
    _feed(coupler, child)
    assert coupler.feedback_levels_below == 3
    after = _snapshot(pstate)
    # Below the model top the parent keeps its own mass.
    np.testing.assert_array_equal(after["mup"], before["mup"])
    for name in ("u", "v", "thp", "qv", "qc", "qr"):
        np.testing.assert_array_equal(after[name][3:], before[name][3:],
                                      err_msg=name)
        np.testing.assert_array_equal(after[name][:2], full[name][:2],
                                      err_msg=name)
        changed = after[name][2] != before[name][2]
        assert changed.any(), name
        half = before[name][2] + F(0.5) * (full[name][2] - before[name][2])
        np.testing.assert_array_equal(after[name][2], half, err_msg=name)
    # Interfaces take the mean of the layers beside them: 1, 1, 0.75,
    # 0.25, then 0.
    np.testing.assert_array_equal(after["w"][4:], before["w"][4:])
    np.testing.assert_array_equal(after["w"][:2], full["w"][:2])
    # ph is rebuilt as a column: above the limit each layer keeps its
    # inverse density, its thickness scaled by its new layer mass.
    _assert_untouched_layers_keep_their_density(
        coupler, pstate, before, after, first_untouched=3)


def _assert_untouched_layers_keep_their_density(coupler, pstate, before,
                                                after, *, first_untouched):
    import cupy as cp

    from woof.core.nest_interp import feedback_parent_bounds

    i_lo, i_hi, j_lo, j_hi = feedback_parent_bounds(
        coupler.registrations["m"], spec_zone=1)
    cells = (slice(None), slice(j_lo, j_hi + 1), slice(i_lo, i_hi + 1))
    phb = cp.asnumpy(pstate.phb).astype(np.float64)
    phb = phb[cells] if phb.ndim == 3 else phb[:, None, None]
    mub = cp.asnumpy(pstate.mub2d).astype(np.float64)[cells[1:]]
    ladder = fb._ladder(pstate)
    nz = ladder[0].size - 1

    def dm(mup):
        mu = mub + mup[cells[1:]].astype(np.float64)
        e = np.stack([fb._edge_cpu(*ladder, k, nz, mu)
                      for k in range(nz + 1)])
        return e[:-1] - e[1:]

    old = np.diff(phb + before["php"][cells], axis=0) / dm(before["mup"])
    new = np.diff(phb + after["php"][cells], axis=0) / dm(after["mup"])
    np.testing.assert_allclose(new[first_untouched:], old[first_untouched:],
                               rtol=2e-5)
    assert not np.array_equal(after["php"][cells][first_untouched + 1:],
                              before["php"][cells][first_untouched + 1:])


@requires_gpu
@pytest.mark.gpu
def test_a_limit_below_the_lowest_layer_is_refused_by_name():
    from woof.experiment import FeedbackTop

    coupler, child, _, _ = _edge(feedback_top=FeedbackTop(50.0, 3))
    with pytest.raises(ValueError, match="lowest layer"):
        _feed(coupler, child)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("ratio", [3, 2])
@pytest.mark.parametrize("top_m", [None, 4500.0])
def test_cross_ladder_feedback_conserves_dry_mass_and_water(ratio, top_m):
    """The parent's layer water over the restriction rectangle equals the
    child's over the same pressure layers, below the limit; with the whole
    column fed back (mu too) so does the layer dry mass."""
    import cupy as cp

    from woof.core.nest_interp import feedback_parent_bounds
    from woof.experiment import FeedbackTop

    nzp, nzc = 6, 11
    coupler, child, pstate, cstate = _edge(
        nz_parent=nzp, nz_child=nzc, ratio=ratio,
        feedback_top=None if top_m is None else FeedbackTop(top_m, 0))
    assert coupler.crosses_ladders
    child_q = {n: cp.asnumpy(getattr(cstate, n)).copy()
               for n in ("qv", "qc", "qr")}
    child_mup = cp.asnumpy(cstate.mup).copy()
    child_mub = cp.asnumpy(cstate.mub2d).copy()
    parent_mup = cp.asnumpy(pstate.mup).copy()
    _feed(coupler, child)
    below = coupler.feedback_levels_below
    assert below == (nzp if top_m is None else 4)
    if top_m is not None:
        # Below the model top the parent keeps its own mass.
        np.testing.assert_array_equal(cp.asnumpy(pstate.mup), parent_mup)
    reg = coupler.registrations["m"]
    i_lo, i_hi, j_lo, j_hi = feedback_parent_bounds(reg, spec_zone=1)
    pl = fb._ladder(pstate)
    cl = fb._ladder(cstate)
    p_mu = (cp.asnumpy(pstate.mub2d).astype(np.float64)
            + cp.asnumpy(pstate.mup))
    p_edges = np.stack([fb._edge_cpu(*pl, k, nzp, p_mu)
                        for k in range(nzp + 1)])
    p_dm = p_edges[:-1] - p_edges[1:]
    c_mu = child_mub.astype(np.float64) + child_mup
    c_on_p = np.stack([fb._edge_cpu(*pl, k, nzp, c_mu)
                       for k in range(nzp + 1)])
    c_dm = c_on_p[:-1] - c_on_p[1:]
    ip, jp = coupler.child_node.cfg.i_parent_start - 1, \
        coupler.child_node.cfg.j_parent_start - 1
    for J in range(j_lo, j_hi + 1):
        for I in range(i_lo, i_hi + 1):
            block = (slice((J - jp) * ratio, (J - jp + 1) * ratio),
                     slice((I - ip) * ratio, (I - ip + 1) * ratio))
            if top_m is None:
                # dry mass of the fed-back layers
                np.testing.assert_allclose(
                    p_dm[:below, J, I].sum(),
                    c_dm[:below][(slice(None),) + block].mean(
                        axis=(1, 2)).sum(), rtol=1e-6)
            for name, q in child_q.items():
                content = fb.remap_layers_cpu(
                    q, child_mub, child_mup, cl, pl)[:below]
                want = content[(slice(None),) + block].astype(
                    np.float64).mean(axis=(1, 2))
                parent_q = cp.asnumpy(getattr(pstate, name))[:below, J, I]
                np.testing.assert_allclose(
                    p_dm[:below, J, I] * parent_q, want, rtol=2e-6,
                    err_msg=name)
                np.testing.assert_allclose(
                    (p_dm[:below, J, I] * parent_q).sum(), want.sum(),
                    rtol=1e-6, err_msg=name)


@requires_gpu
@pytest.mark.gpu
def test_cross_ladder_constant_fields_feed_back_unchanged():
    """A child holding one value at every level feeds that value back on
    every parent level (theta in the parent's own base frame)."""
    import cupy as cp

    from woof.core.nest_interp import feedback_parent_bounds

    # The child's ground 3% lower in pressure than the parent's: the
    # restricted content is divided by the restricted child mass, so a
    # constant comes back as itself regardless.
    coupler, child, pstate, cstate = _edge(nz_parent=6, nz_child=9,
                                           child_mub=87300.0)
    for name, value in (("u", 3.0), ("v", -2.0), ("w", 0.5), ("php", 7.0),
                        ("thp", 4.0), ("qv", 0.008)):
        getattr(cstate, name)[...] = cp.float32(value)
    _feed(coupler, child)
    i_lo, i_hi, j_lo, j_hi = feedback_parent_bounds(
        coupler.registrations["m"], spec_zone=1)
    win = (slice(None), slice(j_lo, j_hi + 1), slice(i_lo, i_hi + 1))
    for name, value in (("w", 0.5), ("thp", 4.0)):
        np.testing.assert_allclose(cp.asnumpy(getattr(pstate, name))[win],
                                   value, rtol=2e-6, err_msg=name)
    # Water keeps its content instead: on these sigma ladders a layer's
    # mass is proportional to MU, so the mixing ratio scales by the ratio
    # of the child's column mass to the parent's.
    c_mu = (cp.asnumpy(cstate.mub2d).astype(np.float64)
            + cp.asnumpy(cstate.mup))
    p_mu = (cp.asnumpy(pstate.mub2d).astype(np.float64)
            + cp.asnumpy(pstate.mup))[win[1:]]
    ratio = 3
    blocks = c_mu.reshape(c_mu.shape[0] // ratio, ratio,
                          c_mu.shape[1] // ratio, ratio).mean(axis=(1, 3))
    jp = coupler.child_node.cfg.j_parent_start - 1
    ip = coupler.child_node.cfg.i_parent_start - 1
    want = 0.008 * blocks[j_lo - jp:j_hi - jp + 1,
                          i_lo - ip:i_hi - ip + 1] / p_mu
    np.testing.assert_allclose(cp.asnumpy(pstate.qv)[win],
                               np.broadcast_to(want, pstate.qv[win].shape),
                               rtol=2e-6)


# -- restart -----------------------------------------------------------------


@requires_gpu
@pytest.mark.gpu
def test_height_limited_feedback_restart_is_sha_identical(monkeypatch,
                                                          tmp_path):
    from datetime import timedelta

    import test_feedback as tf

    from woof.core.model import execute_experiment
    from woof.core.preflight import nest_field_kinds
    from woof.experiment import FeedbackTop
    from woof.io.restart import restore_tree_restart, write_tree_restart

    def deterministic_step(state, cfg, *, refl_10cm_due=False):
        del refl_10cm_due
        for index, kind in enumerate(nest_field_kinds(cfg)):
            name = {"t": "thp", "ph": "php"}.get(kind, kind)
            target = state.mup if kind == "mu" else getattr(state, name)
            target += np.float32(cfg.grid_id * 0.01 + index * 0.001)

    def experiment(seconds):
        exp = tf._feedback_experiment(seconds)
        root, nest = exp.domains
        # Layer tops at 102 m, 204 m AGL (phb = k * 1000): only the first.
        nest = replace(nest, feedback_top=FeedbackTop(150.0, 0))
        return replace(exp, domains=(root, nest))

    monkeypatch.setattr("woof.core.dycore.step", deterministic_step)
    full = experiment(60.0)
    unbroken = tf._feedback_model(full)
    assert unbroken.node(2).coupler.partial_feedback
    execute_experiment(unbroken, validate_state=False,
                       pool_trim_per_period=False)
    unbroken_sha = tf._feedback_state_sha256(unbroken)

    first = tf._feedback_model(experiment(30.0))
    execute_experiment(first, validate_state=False,
                       pool_trim_per_period=False)
    path = write_tree_restart(tmp_path, first,
                              full.start_time + timedelta(seconds=30))
    resumed = tf._feedback_model(full)
    restore_tree_restart(path, resumed)
    execute_experiment(resumed, validate_state=False,
                       pool_trim_per_period=False)
    assert tf._feedback_state_sha256(resumed) == unbroken_sha
    assert resumed.node(2).coupler.feedback_levels_below == 1


# -- parent->child carriers on an own-ladder nest ------------------------------


def _ozone_edge(xp, *, nzp=8, nzc=15, ratio=3):
    """A CAM-ozone (o3input = 2) parent->child edge with the child on its
    own ladder: real registration and plan, stub physics drivers."""
    from test_vertical_nesting_config import CHILD_ETA, SHARED_ETA, _plan

    from woof.core.cam_ozone import ROUTING_PARENT_INTERPOLATED
    from woof.core.nest_interp import register_nest

    assert len(SHARED_ETA) - 1 == nzp and len(CHILD_ETA) - 1 == nzc
    plan = _plan(SHARED_ETA, CHILD_ETA)
    parent_cfg = _domain(1, 0, nx=20, ny=20, nz=nzp)
    child_cfg = _domain(2, 1, nx=8 * ratio, ny=8 * ratio, nz=nzc,
                        ratio=ratio)
    reg = register_nest(
        nri=ratio, nrj=ratio, i_parent_start=child_cfg.i_parent_start,
        j_parent_start=child_cfg.j_parent_start, child_nx=child_cfg.run.nx,
        child_ny=child_cfg.run.ny, parent_nx=20, parent_ny=20, stagger="",
        wrapper="interp")
    rng = np.random.default_rng(5)
    parent_o3 = rng.uniform(1e-8, 1e-5, (nzp, 20, 20)).astype(F)
    parent = SimpleNamespace(
        state=SimpleNamespace(physics=SimpleNamespace(
            o3rad=xp.asarray(parent_o3), call_counts={"cam_ozone": 1})))
    child_driver = SimpleNamespace(
        cam_ozone=SimpleNamespace(mode=ROUTING_PARENT_INTERPOLATED),
        o3rad=xp.zeros((nzc, child_cfg.run.ny, child_cfg.run.nx),
                       dtype=xp.float32),
        call_counts={})
    node = SimpleNamespace(
        cfg=child_cfg, parent=parent,
        state=SimpleNamespace(physics=child_driver),
        coupler=SimpleNamespace(crosses_ladders=True,
                                vertical_plan=lambda: plan))
    return node, reg, plan, parent_o3


def test_cam_ozone_reaches_an_own_ladder_nest_on_its_levels_cpu():
    """o3input = 2 on an own-ladder nest: the parent's ozone (8 levels)
    reaches the child (15) through the vertical interpolation, then SINT,
    instead of SINT writing parent levels into a child-shaped buffer."""
    from woof.core.cam_ozone import transfer_parent_ozone
    from woof.core.nest_interp import sint, window_registration

    node, reg, plan, parent_o3 = _ozone_edge(np)
    transfer_parent_ozone(node, reg)
    got = node.state.physics.o3rad
    cropped, donor = window_registration(reg, (slice(0, reg.nyc),
                                               slice(0, reg.nxc)))
    want = sint(plan.interp_cpu(np.ascontiguousarray(
        parent_o3[(slice(None),) + donor])), cropped)
    np.testing.assert_array_equal(got, want)
    assert node.state.physics.call_counts["cam_ozone"] == 1


@requires_gpu
@pytest.mark.gpu
def test_cam_ozone_own_ladder_transfer_cpu_and_gpu_agree_bit_for_bit():
    import cupy as cp

    from woof.core.cam_ozone import transfer_parent_ozone

    host_node, reg, _, _ = _ozone_edge(np)
    transfer_parent_ozone(host_node, reg)
    dev_node, reg, _, _ = _ozone_edge(cp)
    transfer_parent_ozone(dev_node, reg)
    np.testing.assert_array_equal(
        host_node.state.physics.o3rad.view(np.int32),
        cp.asnumpy(dev_node.state.physics.o3rad).view(np.int32))


@requires_gpu
@pytest.mark.gpu
def test_legacy_rrtmg_parent_ozone_provider_uses_the_child_ladder():
    """The legacy-RRTMG route (ParentOzoneProvider) carries o33d the same
    way, and agrees with the CPU path bit for bit."""
    from woof.core.nest_interp import sint, window_registration
    from woof.core.rrtmg_legacy import ParentOzoneProvider

    node, reg, plan, parent_o3 = _ozone_edge(np)
    provider = ParentOzoneProvider(SimpleNamespace(_o33d_grid=parent_o3),
                                   reg, vertical_plan=plan)
    columns = provider()
    cropped, donor = window_registration(reg, (slice(0, reg.nyc),
                                               slice(0, reg.nxc)))
    want = sint(plan.interp_cpu(np.ascontiguousarray(
        parent_o3[(slice(None),) + donor])), cropped)
    nz = want.shape[0]
    np.testing.assert_array_equal(
        columns, want.transpose(1, 2, 0).reshape(-1, nz))
