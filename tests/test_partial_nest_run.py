"""Partial-depth nest, Phase D3: a running two-way child below an interface.

woof/core/partial_nest_run.py + kernels/partial_nest.cu.  The gates:

* D0 config: one band, refused by name for anything unsupported (interface
  not pure pressure or at etac itself, deep cumulus below ~12 km, damp_opt=3,
  map factors, more than one band, a strict build), and no partial nest
  touches nothing;
* the parent-to-child substep forcing is the stated arithmetic word for word
  (time interpolation relative to the child's stage reference);
* the top strip's tables are bdy_interp1's arithmetic and, wherever the strip
  overlaps the lateral frame, the lateral tables' own words (the corners
  agree), and the frame owns its columns at every level;
* the interface budget: the child's interface flux sums to the parent's over
  the footprint;
* feedback is WRF copy_fcn below the interface (mu, u, v, theta), word for word, the interface
  w/phi (and w/phi everywhere) untouched, and the column mass carries the constant
  offset back (parent total = R(child total) + p_i - p_top);
* scheduling through clock.execute_schedule: parent first, the child
  parent_time_step_ratio times, feedback last;
* a resting atmosphere stays at rest over the measured horizon.
"""

import dataclasses

import numpy as np
import pytest
from conftest import requires_gpu


def _skip_strict():
    from woof import wrf_exact
    if wrf_exact.ENABLED:
        pytest.skip("the partial-depth nest runs in the default build only (refused by name)")


def _coord(nz=30, etac=0.45):
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.verify.partial_nest_envelope import G
    coord = make_vertical_coord(nz, hybrid_opt=2, etac=etac)
    make_base_state(coord, lambda z: 300.0 * np.exp(1e-4 * np.asarray(z) / G),
                    p_surf=1.0e5, ztop=10000.0)
    return coord


def _band(**kw):
    from woof.core.partial_nest import PartialNestBand
    values = dict(k_top=21, parent_grid_ratio=3, parent_time_step_ratio=3,
                  i_parent_start=16, j_parent_start=5, nx=90, ny=12)
    values.update(kw)
    return PartialNestBand(**values)


def _pcfg(**kw):
    from woof.config import RunConfig
    values = dict(nx=60, ny=12, nz=30, dx=3000.0, dy=3000.0, ztop=10000.0, dt=12.0,
                  run_seconds=0.0, hybrid_opt=2, etac=0.45)
    values.update(kw)
    return RunConfig(**values)


# ---------------------------------------------------------------------------
# D0: config and refusals (no device)
# ---------------------------------------------------------------------------

def test_no_partial_nest_is_inert():
    from woof.core.partial_nest import partial_nest_from_mapping, validate_partial_nest
    assert partial_nest_from_mapping(None) is None
    assert validate_partial_nest(None, _pcfg(), _coord()) is None


def test_config_entry_round_trip_and_unknown_keys():
    from woof.core.partial_nest import (PartialNestRefused, partial_nest_from_mapping)
    band = partial_nest_from_mapping({"k_top": 21, "parent_grid_ratio": 3,
                                      "i_parent_start": 16, "j_parent_start": 5,
                                      "nx": 90, "ny": 12})
    # D4: the entry is a chain (one band here).
    assert band == (_band(),)
    band = band[0]
    assert band.relax_zone == 6 and band.strip_width == 7
    with pytest.raises(PartialNestRefused, match="unknown"):
        partial_nest_from_mapping({**dataclasses.asdict(band), "k_tpo": 3})
    assert partial_nest_from_mapping([dataclasses.asdict(band)] * 2) == (band, band)


def test_unsupported_configurations_are_refused_by_name(monkeypatch):
    from woof import wrf_exact
    from woof.core.partial_nest import PartialNestRefused, validate_partial_nest
    coord, cfg = _coord(), _pcfg()
    if wrf_exact.ENABLED:
        with pytest.raises(PartialNestRefused, match="GPUWM_WRF_EXACT=1"):
            validate_partial_nest(_band(), cfg, coord)
        return
    assert validate_partial_nest(_band(), cfg, coord) == _band()
    first_pure = int(np.flatnonzero(np.asarray(coord.c3f) == 0.0)[0])
    with pytest.raises(PartialNestRefused, match="terrain-following"):
        validate_partial_nest(_band(k_top=first_pure - 1), cfg, coord)
    if float(coord.c1f[first_pure]) != 0.0:
        with pytest.raises(PartialNestRefused, match="etac level itself"):
            validate_partial_nest(_band(k_top=first_pure), cfg, coord)
    with pytest.raises(PartialNestRefused, match="deep cumulus"):
        validate_partial_nest(_band(), dataclasses.replace(cfg, cu_physics=1), coord,
                              interface_height_m=5900.0)
    with pytest.raises(PartialNestRefused, match="interface height is unknown"):
        validate_partial_nest(_band(), dataclasses.replace(cfg, cu_physics=1), coord)
    assert validate_partial_nest(_band(), dataclasses.replace(cfg, cu_physics=1), coord,
                                 interface_height_m=12500.0) == _band()
    with pytest.raises(PartialNestRefused, match="damp_opt=3"):
        validate_partial_nest(_band(), dataclasses.replace(cfg, damp_opt=3), coord)
    with pytest.raises(PartialNestRefused, match="map factors"):
        validate_partial_nest(_band(), cfg, coord, has_msf=True)
    with pytest.raises(PartialNestRefused, match="validate_band_chain"):
        validate_partial_nest([_band(), _band()], cfg, coord)
    with pytest.raises(PartialNestRefused, match="phi is not fed back"):
        validate_partial_nest(_band(feedback_fields=("t", "ph")), cfg, coord)
    with pytest.raises(PartialNestRefused, match="feedback_fields"):
        validate_partial_nest(_band(feedback_fields=("t", "qv")), cfg, coord)
    with pytest.raises(PartialNestRefused, match="whole parent cells"):
        validate_partial_nest(_band(nx=89), cfg, coord)
    with pytest.raises(PartialNestRefused, match="does not fit"):
        validate_partial_nest(_band(relax_levels=30), cfg, coord)
    monkeypatch.setattr(wrf_exact, "ENABLED", True)
    with pytest.raises(PartialNestRefused, match="GPUWM_WRF_EXACT=1"):
        validate_partial_nest(_band(), cfg, coord)


# ---------------------------------------------------------------------------
# device gates
# ---------------------------------------------------------------------------

def _words(a):
    import cupy as cp
    return np.ascontiguousarray(cp.asnumpy(a)).view(np.uint32)


def _same(got, want, what):
    g, w = _words(got), _words(want)
    bad = np.flatnonzero(g.ravel() != w.ravel())
    assert bad.size == 0, (f"{what}: {bad.size} of {g.size} words differ, first at "
                           f"{np.unravel_index(bad[0], g.shape)}")


def _nest(case="gravity_wave", **band):
    from woof.verify.partial_nest_envelope import build
    nest, _, _ = build(case, band_overrides=band or None)
    return nest


def _forced(case="gravity_wave", **band):
    """A nest one parent step in, forced, before its child steps."""
    from woof.core import dycore
    from woof.core.clock import build_schedule
    from woof.core.partial_nest_run import two_domain_experiment
    from woof.verify.partial_nest_envelope import GEOMETRY
    nest = _nest(case, **band)
    exp = two_domain_experiment(nest.pcfg, nest.ccfg, nest.band,
                                run_seconds=GEOMETRY["dt"], time_step=int(GEOMETRY["dt"]))
    clocks = build_schedule(exp).clock.clocks()
    pc, cc = clocks[1], clocks[2]
    nest.snapshot(nest.a)
    pc.prepare_step()
    dycore.step(nest.parent, nest.pcfg)
    pc.advance()
    nest.force(cc, pc)
    cc.mark_force()
    return nest, pc, cc


@requires_gpu
@pytest.mark.gpu
def test_substep_forcing_is_the_stated_arithmetic():
    """nest_top_substep_forcing: a + f*(b - a) (nest_top_time_interp's
    words) relative to the child's stage reference, each operation rounded
    once, for random planes and fractions."""
    _skip_strict()
    import cupy as cp
    from woof.core.kernels import get_kernel
    rng = np.random.default_rng(5)
    n = 37
    f32 = np.float32
    arr = {k: rng.standard_normal(n).astype(f32) * f32(s) for k, s in
           (("om_a", 2), ("om_b", 2), ("w_a", .1), ("w_b", .1), ("ph_a", 5e4),
            ("ph_b", 5e4), ("omu_a", 2), ("omu_b", 2), ("p_a", 4e4), ("p_b", 4e4),
            ("adj_a", 1e-3), ("adj_b", 1e-3), ("ww_rf", 2), ("adj_rf", 1e-3),
            ("w_star", .1), ("ph_star", 5e4))}
    c_top, fr, fo, fs = f32(7.3e4), f32(0.4), f32(0.3), f32(1.0 / 6.0)
    outs = [cp.zeros(n, dtype=cp.float32) for _ in range(7)]
    dev = {k: cp.asarray(v) for k, v in arr.items()}
    get_kernel("partial_nest", "nest_top_substep_forcing")(
        (1,), (64,), tuple(dev[k] for k in (
            "om_a", "om_b", "w_a", "w_b", "ph_a", "ph_b", "omu_a", "omu_b", "p_a", "p_b",
            "adj_a", "adj_b", "ww_rf", "adj_rf", "w_star", "ph_star"))
        + (c_top, fr, fo, fs) + tuple(outs) + (np.int32(n),))

    def lerp(a, b, f):
        return (a + (f * (b - a).astype(f32)).astype(f32)).astype(f32)

    a = arr
    want = [
        (lerp(a["om_a"], a["om_b"], fr) - a["ww_rf"]).astype(f32),
        (c_top * (lerp(a["w_a"], a["w_b"], fr) - a["w_star"]).astype(f32)).astype(f32),
        (lerp(a["ph_a"], a["ph_b"], fr) - a["ph_star"]).astype(f32),
        (lerp(a["omu_a"], a["omu_b"], fr) - lerp(a["omu_a"], a["omu_b"], fs)).astype(f32),
        (lerp(a["p_a"], a["p_b"], fr) - lerp(a["p_a"], a["p_b"], fs)).astype(f32),
        (lerp(a["p_a"], a["p_b"], fo) - lerp(a["p_a"], a["p_b"], fs)).astype(f32),
        (lerp(a["adj_a"], a["adj_b"], fr) - a["adj_rf"]).astype(f32),
    ]
    for got, w, name in zip(outs, want, ("ww_pp", "w_top", "ph_top", "ww_above",
                                         "p_above", "p_above_old", "adj_pp")):
        np.testing.assert_array_equal(cp.asnumpy(got).view(np.uint32), w.view(np.uint32),
                                      err_msg=name)


@requires_gpu
@pytest.mark.gpu
def test_top_strip_and_lateral_tables_agree_at_the_corners():
    """Where the top strip overlaps the lateral frame, its VALUE and
    TENDENCY words are the lateral bdy_interp1 tables' words for the same
    point: one coupled parent field, one SINT, one arithmetic."""
    _skip_strict()
    import cupy as cp
    nest, _, _ = _forced()
    k = nest.k
    width = nest.band.strip_width
    sz = nest.frame_width
    checked = 0
    for kind in ("u", "v", "t", "ph", "w"):
        val, tend = nest.strip[kind]
        full = kind in ("w", "ph")
        lat = nest.lateral[kind]
        for d in range(width):
            lev = k - d               # strip row d is level k - d (header of the .cu)
            if lev >= k + full:       # a half-level field's row 0: the parent's
                continue
            for side, (tv, tt) in lat.items():
                tv, tt = cp.asnumpy(tv[lev]), cp.asnumpy(tt[lev])
                sv, st = cp.asnumpy(val[d]), cp.asnumpy(tend[d])
                if side == "west":
                    s_v, s_t = sv[:, :sz], st[:, :sz]
                elif side == "east":
                    s_v, s_t = sv[:, ::-1][:, :sz], st[:, ::-1][:, :sz]
                elif side == "south":
                    s_v, s_t = sv[:sz], st[:sz]
                else:
                    s_v, s_t = sv[::-1][:sz], st[::-1][:sz]
                np.testing.assert_array_equal(s_v.view(np.uint32), tv.view(np.uint32),
                                              err_msg=f"{kind} {side} value row {d}")
                np.testing.assert_array_equal(s_t.view(np.uint32), tt.view(np.uint32),
                                              err_msg=f"{kind} {side} tendency row {d}")
                checked += tv.size
    assert checked > 10000


@requires_gpu
@pytest.mark.gpu
def test_top_zone_leaves_the_frame_columns_to_the_frame():
    """The relaxation kernel writes nothing in the lateral frame's columns
    (the frame owns them at every level) and only the top zone's rows."""
    _skip_strict()
    import cupy as cp
    from woof.core.partial_nest_run import _blocks, _kernel
    nest, _, _ = _forced()
    band, fr = nest.band, nest.frame_width
    rng = np.random.default_rng(2)
    for kind, half in (("u", 1), ("ph", 0)):
        shp = nest._child_shape(kind)
        field = cp.asarray(rng.standard_normal(shp), dtype=cp.float32)
        tend0 = cp.asarray(rng.standard_normal(shp), dtype=cp.float32)
        tend = tend0.copy()
        val, tt = nest.strip[kind]
        n = shp[1] * shp[2]
        _kernel("nest_top_relaxation")(
            _blocks(band.relax_zone * n), (256,),
            (field, tend, val, tt, nest.fcx, nest.gcx, np.float32(2.0),
             np.int32(band.strip_width), np.int32(band.spec_levels),
             np.int32(band.relax_zone), np.int32(fr), np.int32(half),
             np.int32(shp[0]), np.int32(shp[1]), np.int32(shp[2])))
        changed = cp.asnumpy(tend != tend0)
        assert not changed[:, :fr].any() and not changed[:, -fr:].any()
        assert not changed[:, :, :fr].any() and not changed[:, :, -fr:].any()
        top = shp[0] - 1 + half
        rows = [top - d for d in range(band.relax_zone) if top - d < shp[0]]
        assert changed[rows, fr:-fr, fr:-fr].all(), kind
        assert not changed[:min(rows)].any()


@requires_gpu
@pytest.mark.gpu
def test_interface_budget_matches_the_parent_over_the_footprint():
    """With mass_correction the child's interface flux, summed over its
    cells, is the parent's over the footprint times the cell-area ratio."""
    _skip_strict()
    import cupy as cp
    from woof.core.dycore import stage_fluxes
    nest, _, _ = _forced("bubble")
    k, band = nest.k, nest.band
    r = band.parent_grid_ratio
    _, _, ww = stage_fluxes(nest.parent, nest.pcfg)
    pw = cp.asnumpy(ww[k]).astype(np.float64)
    j0, i0 = band.j_parent_start - 1, band.i_parent_start - 1
    parent_sum = r * r * pw[j0:j0 + band.ny // r, i0:i0 + band.nx // r].sum()
    scale = np.abs(pw).max() * band.nx * band.ny
    assert nest.max_abs_mass_residual > 0.0                   # SINT alone is not conservative
    # mass_route "face" (default): the residual folded into the flux itself.
    assert nest.mass_route == "face"
    face_sum = cp.asnumpy(nest.b.om_k).astype(np.float64).sum()
    assert not cp.asnumpy(nest.b.adj).any()
    assert abs(face_sum - parent_sum) < 1e-6 * scale, (face_sum, parent_sum)
    # mass_route "adj": the SINT flux plus the uniform residual in dmdt_adj.
    nest.mass_route = "adj"
    nest.snapshot(nest.b)
    sint_sum = cp.asnumpy(nest.b.om_k).astype(np.float64).sum()
    adj = cp.asnumpy(nest.b.adj).astype(np.float64)
    assert np.all(adj == adj.flat[0]) and adj.flat[0] != 0.0
    assert abs(sint_sum + adj.sum() - parent_sum) < 1e-6 * scale


@requires_gpu
@pytest.mark.gpu
def test_feedback_is_copy_fcn_below_the_interface():
    """copy_fcn's words on the parent below the interface for mu, u, v and
    theta (WRF's t_2 = theta - 300, re-expressed on the parent's base) when
    a band selects them (feedback_fields; D4's default is theta alone);
    everything above the interface, and w and phi everywhere, untouched;
    mu' restricted, which is the total mass with the constant offset
    p_i - p_top added back."""
    _skip_strict()
    import cupy as cp
    from woof.core.nest_interp import copy_fcn, feedback_parent_bounds
    from woof.core.partial_nest import DEFAULT_FEEDBACK_FIELDS
    assert DEFAULT_FEEDBACK_FIELDS == ("t",)
    nest, _, _ = _forced("bubble", feedback_fields=("mu", "u", "v", "t"))
    p, c, k = nest.parent, nest.child, nest.k
    before = {n: getattr(p, n).copy() for n in ("u", "v", "w", "thp", "php", "mup")}
    want = {}
    for kind, name in (("u", "u"), ("v", "v")):
        reg = nest.regs[{"u": "x", "v": "y"}[kind]]
        tgt = getattr(p, name)[:k].copy()
        copy_fcn(tgt, cp.ascontiguousarray(getattr(c, name)[:k]), reg, spec_zone=1)
        want[name] = tgt
    reg = nest.regs[""]
    tgt = p.thp[:k].copy()
    copy_fcn(tgt, (c.thp + c.thb[:, None, None]) - np.float32(300.0), reg, spec_zone=1)
    i_lo, i_hi, j_lo, j_hi = feedback_parent_bounds(reg, spec_zone=1)
    win = tgt[:, j_lo:j_hi + 1, i_lo:i_hi + 1]
    win += np.float32(300.0)
    win -= p.thb[:k, None, None]
    tgt[:, j_lo:j_hi + 1, i_lo:i_hi + 1] = win
    want["thp"] = tgt
    mu = p.mup[None].copy()
    copy_fcn(mu, c.mup[None], nest.regs[""], spec_zone=1)
    nest.feedback()
    for name, w in want.items():
        _same(getattr(p, name)[:k], w, name)
        _same(getattr(p, name)[k:], before[name][k:], name + " above")
    for name in ("w", "php"):
        _same(getattr(p, name), before[name], name + " (not fed back)")
    _same(p.mup, mu[0], "mup")
    i_lo, i_hi, j_lo, j_hi = feedback_parent_bounds(nest.regs[""], spec_zone=1)
    r = nest.band.parent_grid_ratio
    ptot = (cp.asnumpy(p.mub2d).astype(np.float64)
            + cp.asnumpy(p.mup).astype(np.float64))[j_lo:j_hi + 1, i_lo:i_hi + 1]
    ctot = cp.asnumpy(c.mub2d).astype(np.float64) + cp.asnumpy(c.mup).astype(np.float64)
    js = slice((j_lo - nest.band.j_parent_start + 1) * r, (j_hi - nest.band.j_parent_start + 2) * r)
    is_ = slice((i_lo - nest.band.i_parent_start + 1) * r, (i_hi - nest.band.i_parent_start + 2) * r)
    blk = ctot[js, is_]
    rmean = blk.reshape(blk.shape[0] // r, r, blk.shape[1] // r, r).mean(axis=(1, 3))
    assert np.abs(ptot - (rmean + nest.shift)).max() < 4 * np.spacing(np.float32(1.0e5))


@requires_gpu
@pytest.mark.gpu
def test_schedule_parent_first_child_ratio_times_feedback_last():
    _skip_strict()
    from woof.core.partial_nest_run import run
    from woof.verify.partial_nest_envelope import GEOMETRY
    nest = _nest("rest")
    events = []
    force, feedback = nest.force, nest.feedback

    def rec_force(*a):
        events.append("F")
        force(*a)

    def rec_feedback():
        events.append("B")
        feedback()

    nest.force, nest.feedback = rec_force, rec_feedback
    report = run(nest, run_seconds=3 * GEOMETRY["dt"], time_step=int(GEOMETRY["dt"]),
                 on_parent_step=lambda n, c: events.append("P"),
                 on_child_step=lambda n, c: events.append("C"))
    assert "".join(events) == "PFCCCB" * 3
    assert (report.steps, report.forces, report.feedback_calls) == (12, 3, 3)
    assert nest.stats["child_steps"] == 9


@requires_gpu
@pytest.mark.gpu
def test_a_resting_atmosphere_stays_at_rest():
    """Short form of the envelope's rest case (tests/test_partial_nest_envelope.py
    holds the measured long-horizon bounds)."""
    _skip_strict()
    from woof.verify.partial_nest_envelope import run_case
    out = run_case("rest", seconds=240.0, references=False)
    assert out["finite"]
    assert out["child_w_max_abs"] < 2e-3, out
    assert out["interface_w_max_abs"] < 1e-3, out
    assert abs(out["child_dry_mass_drift_rel"]) < 1e-7, out
