"""state_relaxation_multi: one RK stage's state relaxation rows in one
launch, word for word the per-row path (kernels/lbc_state.cu,
lateral_bc._launch_relaxation_multi vs _launch_state_relaxation plus the
interior ufuncs of _apply_legacy_held_interior).

Every row form the stage code builds is checked: the specified domain's
held rows (stage 1 computes and stores the held row, later stages read
it), the nested form (held in a register, time-t sources and mass), the
strict form (specified-only rows, a mu row relaxing straight into its
tendency), a seamed tile (relax_sides != 15), the map-factor divide, and
more rows than one table holds.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from conftest import requires_gpu


def _setup(*, nx=40, ny=36, relax_zone=4, map_factors=False):
    import cupy as cp

    import woof.ingest.lateral_bc as lbc
    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest

    cfg = RunConfig(nx=nx, ny=ny, nz=6, dx=250.0, dy=250.0, ztop=12000.0,
                    dt=1.25, run_seconds=60.0, moist=False, specified=True,
                    spec_zone=1, relax_zone=relax_zone,
                    spec_bdy_width=relax_zone + 1, relax_timescale_s=150.0,
                    relax_w=True)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.0),
                           cfg.p_surf, cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    forcing = [init_at_rest(cfg, coord, base) for _ in range(2)]
    rng = np.random.default_rng(2609)
    for name in ("u", "v", "w", "thp", "php"):
        values = rng.normal(0.0, 1.0, getattr(state, name).shape)
        getattr(state, name)[...] = cp.asarray(values.astype(np.float32))
    state.w[0] = 0.0
    state.mup[...] = cp.asarray(
        rng.normal(0.0, 50.0, state.mup.shape).astype(np.float32))
    if map_factors:
        for name in ("msft", "msfu", "msfv"):
            m = getattr(state, name)
            m[...] = cp.asarray(
                (1.0 + 0.05 * rng.normal(0.0, 1.0, m.shape)).astype(np.float32))
        state.msfv_inv[...] = np.float32(1.0) / state.msfv
        state.has_msf = True
    snapshots = []
    for n, f in enumerate(forcing):
        f.u += cp.float32(2.0 + n)
        f.thp += cp.float32(3.0 + n)
        f.w[1:] = cp.float32(0.5 + n)
        snapshot = dict(lbc.domain_boundary_snapshot(f))
        mu = f.total_mu()
        chf = f.c1f[:, None, None] * mu[None] + f.c2f[:, None, None]
        snapshot["w"] = lbc._host(chf * f.w)
        snapshots.append(snapshot)
    boundaries = lbc.build_lateral_boundaries(
        snapshots, [0.0, 60.0], spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    lbc.attach_lateral_boundaries(state, boundaries)
    return cfg, state, boundaries


_TENDENCIES = ("ru_t", "rv_t", "rw_t", "rth_t", "rph_t", "rmu_t")
_HELD = ("lbc_relax_u", "lbc_relax_v", "lbc_relax_w", "lbc_relax_theta",
         "lbc_relax_phi")


def _seed(state, rng):
    """Random words in every tendency and held row the stage touches, so
    the check sees the incoming values carried, not zeros."""
    import cupy as cp

    for name in _TENDENCIES:
        a = getattr(state, name)
        a[...] = cp.asarray(rng.normal(0.0, 1e-3, a.shape).astype(np.float32))
    for slot, name in zip(_HELD, ("u", "v", "w", "thp", "php")):
        held = state.scratch(getattr(state, name).shape, slot)
        held[...] = cp.asarray(
            rng.normal(0.0, 1e-3, held.shape).astype(np.float32))


def _words(state):
    import cupy as cp

    out = {name: cp.asnumpy(getattr(state, name)).view(np.uint32)
           for name in _TENDENCIES}
    for slot, name in zip(_HELD, ("u", "v", "w", "thp", "php")):
        out[slot] = cp.asnumpy(
            state.scratch(getattr(state, name).shape, slot)).view(np.uint32)
    return out


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert np.array_equal(a[key], b[key]), key


def _fused_counter(monkeypatch):
    """Count the fused launches while keeping the real implementation, so
    an enabled run is known to have taken the fused path (a silent
    fallback to the per-row launches would otherwise pass every check)."""
    import woof.ingest.lateral_bc as lbc

    counts = {"calls": 0, "launches": 0}
    real_launch = lbc._launch_relaxation_multi
    real_get = lbc.get_kernel

    def get_kernel(name, func, shape=()):
        kernel = real_get(name, func, shape)
        if func != "state_relaxation_multi":
            return kernel

        def launch(*args, **kwargs):
            counts["launches"] += 1
            return kernel(*args, **kwargs)
        return launch

    def wrapped(*args, **kwargs):
        counts["calls"] += 1
        return real_launch(*args, **kwargs)

    monkeypatch.setattr(lbc, "_launch_relaxation_multi", wrapped)
    monkeypatch.setattr(lbc, "get_kernel", get_kernel)
    return counts


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("map_factors", [False, True])
def test_stage_rows_match_the_per_row_path(monkeypatch, map_factors):
    """apply_state_lateral_boundaries over the three stages, fused vs
    per-row, from the same seeded tendencies and held rows."""
    import woof.ingest.lateral_bc as lbc

    cfg, state, _ = _setup(map_factors=map_factors)
    counts = _fused_counter(monkeypatch)
    results = {}
    for enabled in (False, True):
        monkeypatch.setattr(lbc, "LBC_RELAX_MULTI_ENABLED", enabled)
        rng = np.random.default_rng(5)
        _seed(state, rng)
        stages = []
        for stage in (0, 1, 2):
            lbc.apply_state_lateral_boundaries(state, cfg, rk_stage=stage)
            stages.append(_words(state))
        results[enabled] = stages
        # one fused launch per stage, none on the per-row path
        assert counts["launches"] == (3 if enabled else 0)
    for per_row, fused in zip(results[False], results[True]):
        _assert_same(per_row, fused)
    # The stage changed the rows: the check is not vacuous.
    rng = np.random.default_rng(5)
    _seed(state, rng)
    before = _words(state)
    assert any(not np.array_equal(before[k], results[True][0][k])
               for k in before)


def _nested_plan(state, device, *, store):
    """The nested path's rows: time-t sources and mass, held in a register
    (store=False) or in the row's own buffer (the strict nested phi form)."""
    import woof.ingest.lateral_bc as lbc

    sources = {"u": state.u0, "v": state.v0, "w": state.w0,
               "theta": state.thp0, "phi": state.php0}
    plan = []
    for name, tendency in (("u", state.ru_t), ("v", state.rv_t),
                           ("theta", state.rth_t), ("phi", state.rph_t),
                           ("w", state.rw_t)):
        held = state.scratch(tendency.shape, "lbc_relax_" + name)
        plan.append(lbc._RelaxRow(
            name, device.fields[name], target=tendency, held=held,
            held_mode="compute", held_store=store,
            source_field=sources[name], source_mup=state.mup0,
            divide_msf=(state.has_msf and name in ("theta", "phi"))))
    plan.append(lbc._RelaxRow("mu", device.fields["mu"],
                              target=state.rmu_t[None], apply_relax=True))
    return plan


def _strict_plan(state, device, stage):
    """The strict path's rows: specified-only u/v/theta/w, phi held, mu
    relaxing straight into its tendency on stage 1."""
    import woof.ingest.lateral_bc as lbc

    plan = [lbc._RelaxRow(name, device.fields[name], target=tendency)
            for name, tendency in (("u", state.ru_t), ("v", state.rv_t),
                                   ("theta", state.rth_t), ("w", state.rw_t))]
    held = state.scratch(state.rph_t.shape, "lbc_relax_phi")
    if stage == 0:
        plan.append(lbc._RelaxRow(
            "phi", device.fields["phi"], target=state.rph_t, held=held,
            held_mode="compute", divide_msf=state.has_msf))
    else:
        plan.append(lbc._RelaxRow(
            "phi", device.fields["phi"], target=state.rph_t, held=held,
            held_mode="read"))
    plan.append(lbc._RelaxRow("mu", device.fields["mu"],
                              target=state.rmu_t[None],
                              apply_relax=(stage == 0)))
    return plan


def _capture_plan(state, device):
    """capture_folded_relaxation's rows: held only, no target."""
    import woof.ingest.lateral_bc as lbc

    sources = {"u": state.u0, "v": state.v0, "w": state.w0,
               "theta": state.thp0}
    return [lbc._RelaxRow(
        name, device.fields[name],
        held=state.scratch(getattr(state, attr).shape, "lbc_relax_" + name),
        held_mode="compute", source_field=sources[name],
        source_mup=state.mup0)
        for name, attr in (("u", "u"), ("v", "v"), ("theta", "thp"),
                           ("w", "w"))]


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("form", ["nested", "nested_store", "strict",
                                  "capture", "seam"])
def test_every_row_form_matches_the_per_row_path(monkeypatch, form):
    import cupy as cp

    import woof.ingest.lateral_bc as lbc

    cfg, state, boundaries = _setup(map_factors=True)
    rng = np.random.default_rng(9)
    for name in ("u", "v", "w", "thp", "php", "mup"):
        src = getattr(state, name)
        copy = getattr(state, name + "0")
        copy[...] = src + cp.asarray(
            rng.normal(0.0, 0.1, src.shape).astype(np.float32))
    if form == "seam":
        state.lateral_boundaries = dataclasses.replace(
            boundaries, seam_sides=("east",))
    device, common, weights = lbc._relaxation_context(state, cfg)

    def plans():
        if form in ("nested", "nested_store"):
            return [_nested_plan(state, device, store=form == "nested_store")
                    for _ in range(2)]
        if form == "strict":
            return [_strict_plan(state, device, stage) for stage in (0, 1)]
        if form == "capture":
            return [_capture_plan(state, device)]
        return [_strict_plan(state, device, 0),
                _nested_plan(state, device, store=False)]

    # A register-held row (held_store=False) leaves its buffer alone, where
    # the per-row path writes its shared temporary: only the tendencies
    # are the contract there.
    compare_held = form in ("nested_store", "strict", "capture")
    counts = _fused_counter(monkeypatch)
    results = {}
    for enabled in (False, True):
        monkeypatch.setattr(lbc, "LBC_RELAX_MULTI_ENABLED", enabled)
        _seed(state, np.random.default_rng(3))
        stages = []
        launched = 0
        for plan in plans():
            lbc._run_relaxation_rows(state, plan, common, weights)
            launched += 1
            words = _words(state)
            if not compare_held:
                words = {k: v for k, v in words.items() if k in _TENDENCIES}
            stages.append(words)
        results[enabled] = stages
        assert counts["launches"] == (launched if enabled else 0)
    for per_row, fused in zip(results[False], results[True]):
        _assert_same(per_row, fused)


@requires_gpu
@pytest.mark.gpu
def test_more_rows_than_one_table_take_further_launches(monkeypatch):
    import cupy as cp

    import woof.ingest.lateral_bc as lbc

    cfg, state, _ = _setup(map_factors=True)
    device, common, weights = lbc._relaxation_context(state, cfg)
    nrows = lbc.LBC_RELAX_MULTI_MAX_ROWS + 3
    rng = np.random.default_rng(13)
    seeds = [rng.normal(0.0, 1e-3, state.thp.shape).astype(np.float32)
             for _ in range(nrows)]
    counts = _fused_counter(monkeypatch)
    results = {}
    for enabled in (False, True):
        monkeypatch.setattr(lbc, "LBC_RELAX_MULTI_ENABLED", enabled)
        targets = [cp.asarray(s) for s in seeds]
        plan = [lbc._RelaxRow("theta", device.fields["theta"], target=t,
                              apply_relax=True) for t in targets]
        lbc._run_relaxation_rows(state, plan, common, weights)
        results[enabled] = [cp.asnumpy(t).view(np.uint32) for t in targets]
    # one call, two tables (12 + 3 rows)
    assert counts["calls"] == 1 and counts["launches"] == 2
    for per_row, fused, seed in zip(results[False], results[True], seeds):
        assert np.array_equal(per_row, fused)
        assert not np.array_equal(fused, seed.view(np.uint32))


def test_rows_sharing_a_target_fall_back_to_the_per_row_path():
    """Two rows accumulating into one buffer must run in order."""
    import woof.ingest.lateral_bc as lbc

    class Side:
        value = tendency = None

    class Boundary:
        west = east = south = north = Side()

    class State:
        pass

    target = np.zeros((2, 3, 4), np.float32)
    rows = [lbc._RelaxRow("u", Boundary(), target=target),
            lbc._RelaxRow("v", Boundary(), target=target)]
    # Host arrays are refused before any device address is read.
    assert lbc._relaxation_multi_rows(State(), rows) is False
    assert lbc._relaxation_multi_rows(State(), []) is False


@requires_gpu
@pytest.mark.gpu
def test_overlapping_device_targets_fall_back_to_the_per_row_path():
    """Two views of one backing that share cells at different offsets."""
    import cupy as cp

    import woof.ingest.lateral_bc as lbc

    cfg, state, _ = _setup()
    device, _common, _weights = lbc._relaxation_context(state, cfg)
    n = state.thp.size
    backing = cp.zeros((n + 1,), dtype=cp.float32)
    first = backing[:n].reshape(state.thp.shape)
    second = backing[1:n + 1].reshape(state.thp.shape)
    rows = [lbc._RelaxRow("theta", device.fields["theta"], target=first,
                          apply_relax=True),
            lbc._RelaxRow("theta", device.fields["theta"], target=second,
                          apply_relax=True)]
    assert lbc._relaxation_multi_rows(state, rows) is False
    apart = cp.zeros_like(state.thp)
    rows[1] = lbc._RelaxRow("theta", device.fields["theta"], target=apart,
                            apply_relax=True)
    assert lbc._relaxation_multi_rows(state, rows) is True
