"""A live nest on its own eta ladder (WRF vert_refine_method = 2).

The configuration surface, the refusals named for what the tree cannot
carry yet, and the host-side operators that birth and force such a nest
(their WRF parity is tests/test_nest_vertical_wrf471_parity.py).
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest
from conftest import requires_gpu

from woof.experiment import (domain_vertical, has_own_ladder,
                             load_experiment, vertically_nested_ids)

SHARED_ETA = [1.0, 0.95, 0.88, 0.78, 0.65, 0.5, 0.35, 0.2, 0.0]
CHILD_ETA = [1.0, 0.98, 0.96, 0.93, 0.9, 0.86, 0.82, 0.77, 0.71, 0.64,
             0.56, 0.47, 0.37, 0.26, 0.14, 0.0]

TEMPLATE = """\
[experiment]
name = "synth"
start_time = 1974-04-03T12:00:00
run_seconds = 3600.0
restart_interval_s = 0.0
{experiment}

[shared]
nz = 8
ztop = 12000.0
p_top = 5000.0
hybrid_opt = 2
etac = 0.2
eta_levels = {shared_eta}
{shared}

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 100
ny = 80
time_step = 60
dx = 12000.0
history_interval_s = 3600.0

[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 40
j_parent_start = 30
parent_grid_ratio = 3
parent_time_step_ratio = 3
e_we = 61
e_sn = 61
history_interval_s = 900.0
{d02}
"""


def _write(tmp_path, *, d02="", shared="", experiment=""):
    path = tmp_path / "exp.toml"
    path.write_text(textwrap.dedent(TEMPLATE.format(
        experiment=experiment, shared=shared, d02=d02,
        shared_eta=SHARED_ETA)))
    return path


def test_a_nest_carries_its_own_ladder(tmp_path):
    exp = load_experiment(_write(tmp_path, d02=f"eta_levels = {CHILD_ETA}"))
    child = exp.domain(2)
    assert child.run.nz == len(CHILD_ETA) - 1
    assert tuple(child.run.eta_levels) == tuple(CHILD_ETA)
    assert has_own_ladder(child) and not has_own_ladder(exp.root)
    assert vertically_nested_ids(exp) == (2,)
    vertical = domain_vertical(exp.vertical, child)
    assert tuple(vertical.eta_levels) == tuple(CHILD_ETA)
    assert vertical.p_top == exp.vertical.p_top
    assert domain_vertical(exp.vertical, exp.root) is exp.vertical


def test_level_counts_beside_the_ladder_must_agree(tmp_path):
    nz = len(CHILD_ETA) - 1
    exp = load_experiment(_write(
        tmp_path, d02=f"eta_levels = {CHILD_ETA}\nnz = {nz}\ne_vert = {nz + 1}"))
    assert exp.domain(2).run.nz == nz
    with pytest.raises(ValueError, match="nz = 9"):
        load_experiment(_write(tmp_path,
                               d02=f"eta_levels = {CHILD_ETA}\nnz = 9"))
    with pytest.raises(ValueError, match="e_vert"):
        load_experiment(_write(tmp_path,
                               d02=f"eta_levels = {CHILD_ETA}\ne_vert = 9"))


def test_shareds_own_ladder_restated_is_no_ladder_of_its_own(tmp_path):
    """NEGATIVE CONTROL: a nest naming [shared]'s ladder builds the same
    RunConfig as one naming none, so nothing about it changes."""
    plain = load_experiment(_write(tmp_path))
    restated = load_experiment(_write(tmp_path,
                                      d02=f"eta_levels = {SHARED_ETA}"))
    assert restated.domain(2).run == plain.domain(2).run
    assert restated.domain(2).run.eta_levels is None
    assert vertically_nested_ids(restated) == ()


def test_vert_refine_method_two_is_accepted_and_carries_nothing(tmp_path):
    plain = load_experiment(_write(tmp_path))
    declared = load_experiment(_write(tmp_path,
                                      shared="vert_refine_method = 2"))
    assert declared.domain(2).run == plain.domain(2).run


@pytest.mark.parametrize("ladder, words", [
    ([1.0, 0.5, 0.0], "4..256"),
    ([1.0, 0.9, 0.95, 0.5, 0.3, 0.0], "decreasing"),
    ([0.99, 0.8, 0.6, 0.4, 0.2, 0.0], "1.0"),
])
def test_a_malformed_nest_ladder_is_refused(tmp_path, ladder, words):
    with pytest.raises(ValueError, match=words):
        load_experiment(_write(tmp_path, d02=f"eta_levels = {ladder}"))


def test_feedback_across_two_ladders_is_admitted(tmp_path):
    exp = load_experiment(_write(tmp_path, d02=f"eta_levels = {CHILD_ETA}",
                                 experiment="feedback = 1"))
    assert exp.feedback == 1 and exp.domain(2).feedback_top is None


def test_a_feedback_height_limit_is_parsed(tmp_path):
    from woof.experiment import FeedbackTop

    exp = load_experiment(_write(
        tmp_path, d02=f"eta_levels = {CHILD_ETA}\nfeedback_top_m = 4000.0\n"
                      "feedback_taper_levels = 2",
        experiment="feedback = 1"))
    assert exp.domain(2).feedback_top == FeedbackTop(4000.0, 2)
    exp = load_experiment(_write(tmp_path, d02="feedback_top_m = 3000",
                                 experiment="feedback = 1"))
    assert exp.domain(2).feedback_top == FeedbackTop(3000, 3)
    assert exp.root.feedback_top is None


@pytest.mark.parametrize("d02, experiment, words", [
    ("feedback_top_m = 3000.0", "", "feedback = 0"),
    ("feedback_taper_levels = 2", "feedback = 1", "without feedback_top_m"),
    ("feedback_top_m = -5.0", "feedback = 1", "finite positive"),
    ("feedback_top_m = 3000.0\nfeedback_taper_levels = 40",
     "feedback = 1", "0..20"),
])
def test_a_malformed_feedback_height_limit_is_refused(tmp_path, d02,
                                                      experiment, words):
    with pytest.raises(ValueError, match=words):
        load_experiment(_write(tmp_path, d02=d02, experiment=experiment))


def test_a_feedback_height_limit_on_the_root_is_refused(tmp_path):
    path = _write(tmp_path, experiment="feedback = 1")
    text = path.read_text().replace(
        "dx = 12000.0\n", "dx = 12000.0\nfeedback_top_m = 3000.0\n", 1)
    path.write_text(text)
    with pytest.raises(ValueError, match="root"):
        load_experiment(path)


def test_a_height_limited_feedback_refuses_moving_and_streamed_edges(
        tmp_path):
    import dataclasses
    from types import SimpleNamespace

    from woof.experiment import _refuse_unsupported_partial_feedback

    exp = load_experiment(_write(tmp_path, d02="feedback_top_m = 3000.0",
                                 experiment="feedback = 1"))
    _refuse_unsupported_partial_feedback(exp)
    spawned = dataclasses.replace(exp.domain(2), spawn=object())
    tree = SimpleNamespace(
        domains=(exp.root, spawned), vertical=exp.vertical, feedback=1,
        relocation=exp.relocation, devices=exp.devices, tiles=exp.tiles)
    with pytest.raises(ValueError, match="moves, follows or spawns"):
        _refuse_unsupported_partial_feedback(tree)
    streamed = SimpleNamespace(enabled=True, mode="on")
    tiled = dataclasses.replace(exp.domain(2), tiles=streamed)
    tree = SimpleNamespace(
        domains=(exp.root, tiled), vertical=exp.vertical, feedback=1,
        relocation=exp.relocation, devices=exp.devices, tiles=exp.tiles)
    with pytest.raises(ValueError, match="streams d02"):
        _refuse_unsupported_partial_feedback(tree)


def test_an_absent_height_limit_leaves_the_identities_alone(tmp_path):
    """OFF contract: the field is dropped from the restart identity when
    None, so a tree that never names it fingerprints as it always did; a
    declared limit binds."""
    import json

    from woof.core.model import restart_identity_payload

    exp = load_experiment(_write(tmp_path, experiment="feedback = 1"))
    assert "feedback_top" not in json.dumps(restart_identity_payload(exp))
    limited = load_experiment(_write(tmp_path, d02="feedback_top_m = 3000.0",
                                     experiment="feedback = 1"))
    assert "feedback_top" in json.dumps(restart_identity_payload(limited))


def test_a_spawned_nest_on_its_own_ladder_is_refused_by_name(tmp_path):
    import dataclasses
    from types import SimpleNamespace

    from woof.experiment import _refuse_unsupported_vertical_nesting

    exp = load_experiment(_write(tmp_path, d02=f"eta_levels = {CHILD_ETA}"))
    spawned = dataclasses.replace(exp.domain(2), spawn=object())
    tree = SimpleNamespace(
        domains=(exp.root, spawned), vertical=exp.vertical, feedback=0,
        relocation=exp.relocation, devices=exp.devices, tiles=exp.tiles)
    with pytest.raises(ValueError, match="moves, follows, spawns"):
        _refuse_unsupported_vertical_nesting(tree)


def test_a_default_tree_builds_exactly_the_runconfig_it_always_did(tmp_path):
    """The off state: no nest names a ladder, so no RunConfig gains one and
    no edge crosses ladders."""
    exp = load_experiment(_write(tmp_path))
    assert all(dc.run.eta_levels is None for dc in exp.domains)
    assert vertically_nested_ids(exp) == ()


# -- the host operators ------------------------------------------------------


def _coord(eta):
    from woof.core.grid import finalize_vertical_coord, make_vertical_coord

    coord = make_vertical_coord(len(eta) - 1, hybrid_opt=2, etac=0.2,
                                eta_levels=np.asarray(eta, dtype=np.float64))
    finalize_vertical_coord(coord, 5000.0)
    return coord


def _plan(parent_eta, child_eta):
    from woof.core.nest_vertical import VerticalNestPlan
    from woof.ingest.nest_init import _ladder_arrays

    class _State:
        pass

    parent = _coord(parent_eta)
    state = _State()
    for name in ("c1h", "c2h", "c3h", "c4h", "c1f", "c2f", "c3f", "c4f",
                 "dnw", "rdnw", "rdn"):
        setattr(state, name, np.asarray(getattr(parent, name),
                                        dtype=np.float32))
    dn, dnw, fnp, fnm = parent.dn, parent.dnw, parent.fnp, parent.fnm
    cof1 = (2.0 * dn[1] + dn[2]) / (dn[1] + dn[2]) * dnw[0] / dn[1]
    cof2 = dn[1] / (dn[1] + dn[2]) * dnw[0] / dn[2]
    state.cf1, state.cf2, state.cf3 = (np.float32(fnp[1] + cof1),
                                       np.float32(fnm[1] - cof1 - cof2),
                                       np.float32(cof2))
    state.cfn, state.cfn1 = np.float32(1.0 + fnp[-1]), np.float32(-fnp[-1])
    return VerticalNestPlan.from_coefficients(
        parent=_ladder_arrays(state, with_cf=True),
        child=_ladder_arrays(_coord(child_eta), with_cf=False),
        p_top=5000.0, hypsometric_opt=2, base_temp=290.0)


def test_one_ladder_onto_itself_is_the_identity_word_for_word():
    plan = _plan(CHILD_ETA, CHILD_ETA)
    rng = np.random.default_rng(3)
    half = rng.standard_normal((len(CHILD_ETA) - 1, 4, 5)).astype(np.float32)
    full = rng.standard_normal((len(CHILD_ETA), 4, 5)).astype(np.float32)
    assert np.array_equal(plan.interp_cpu(half).view(np.uint32),
                          half.view(np.uint32))
    assert np.array_equal(plan.interp_cpu(full).view(np.uint32),
                          full.view(np.uint32))


def test_a_linear_profile_in_the_reference_height_is_kept():
    """Monotone cubic through points on a line in alt: exactly that line
    (to float32 rounding) at the child's points."""
    plan = _plan(SHARED_ETA, CHILD_ETA)
    alt_c, alt_n = plan.alt_w_c, plan.alt_w_n
    field = (3.0 * alt_c + 1.0).astype(np.float32)[:, None, None]
    out = plan.interp_cpu(np.broadcast_to(field, (alt_c.size, 2, 2)).copy())
    np.testing.assert_allclose(out[:, 0, 0], 3.0 * alt_n + 1.0, rtol=2e-6,
                               atol=2e-6)


def test_the_parent_view_hands_down_theta_as_wrf_holds_it():
    from woof.core.nest_vertical import VerticallyNestedParent

    plan = _plan(SHARED_ETA, CHILD_ETA)
    nzp = len(SHARED_ETA) - 1

    class _Parent:
        pass

    parent = _Parent()
    parent.thb = np.linspace(300.0, 330.0, nzp).astype(np.float32)
    parent.thp = np.full((nzp, 3, 4), 1.5, dtype=np.float32)
    parent.qv = np.linspace(0.01, 0.0, nzp, dtype=np.float32)[:, None, None] \
        * np.ones((1, 3, 4), dtype=np.float32)
    parent.mup = np.ones((3, 4), dtype=np.float32)
    view = VerticallyNestedParent(parent, plan)
    assert view.mup is parent.mup                      # 2-D: untouched
    assert view.qv.shape == (len(CHILD_ETA) - 1, 3, 4)
    assert view.thb.ndim == 1                          # as the parent's
    total = view.thp + view.thb[:, None, None]
    expected = plan.interp_cpu(
        (parent.thp + parent.thb[:, None, None]) - np.float32(300.0)
    ) + np.float32(300.0)
    np.testing.assert_allclose(total, expected, rtol=0, atol=1e-4)
    with pytest.raises(AttributeError):
        view.thp = parent.thp
    parent.landusef = np.zeros((nzp, 3, 4), dtype=np.float32)
    assert view.landusef is parent.landusef            # not a vertical field


def test_ladders_that_do_not_share_their_ends_are_refused():
    from woof.core.nest_vertical import VerticalNestRefused, _segments

    with pytest.raises(VerticalNestRefused, match="end"):
        _segments(np.array([0.0, 1.0, 2.0], dtype=np.float32),
                  np.array([0.0, 2.5, 3.0], dtype=np.float32))


# -- WRF namelist import ------------------------------------------------------


def _namelist_texts():
    """The shared WRF pair of tests/test_namelist_import.py, read without
    importing that module (it imports a verification bundle this tree may
    not carry)."""
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parent
              / "test_namelist_import.py").read_text(encoding="utf-8")
    values = {}
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("WPS_TEXT", "INPUT_TEXT")):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    return values["WPS_TEXT"], values["INPUT_TEXT"]


def _pair(tmp_path, wps, inp):
    wps_path = tmp_path / "namelist.wps"
    inp_path = tmp_path / "namelist.input"
    wps_path.write_text(wps)
    inp_path.write_text(inp)
    return wps_path, inp_path


def test_vert_refine_method_two_imports_each_nests_own_ladder(tmp_path):
    """WRF lists every domain's ladder in turn (nest_init_utils.F:315-322);
    the nest's lands on its [[domain]] and d01's on [shared]."""
    import tomllib

    from woof.namelist_import import import_namelists

    WPS_TEXT, INPUT_TEXT = _namelist_texts()
    child = "1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.45, 0.3, 0.15, 0.0"
    start = INPUT_TEXT.index(" eta_levels")
    end = INPUT_TEXT.index("\n", INPUT_TEXT.index("0.0", start))
    shared = INPUT_TEXT[start:end].split("=", 1)[1].strip().rstrip(",")
    shared = " ".join(shared.split())
    inp = (INPUT_TEXT[:start]
           + f" eta_levels = {shared}, {child},"
           + INPUT_TEXT[end:]).replace(
        "e_vert = 9, 9", "e_vert = 9, 11\n vert_refine_method = 0, 2,")
    toml_text, _ = import_namelists(*_pair(tmp_path, WPS_TEXT, inp))
    parsed = tomllib.loads(toml_text)
    assert len(parsed["shared"]["eta_levels"]) == 9
    domains = parsed["domain"]
    assert "eta_levels" not in domains[0]
    assert domains[1]["eta_levels"] == [float(v) for v in child.split(",")]
    short = inp.replace(f", {child},", ",")
    with pytest.raises(ValueError, match="every domain's ladder"):
        import_namelists(*_pair(tmp_path, WPS_TEXT, short))
    one = INPUT_TEXT.replace(" time_step = 60,",
                             " time_step = 60,\n vert_refine_method = 1,")
    with pytest.raises(ValueError, match="vert_refine_method"):
        import_namelists(*_pair(tmp_path, WPS_TEXT, one))


# -- the live forcing wiring --------------------------------------------------


def _ladder_state(nx, ny, eta, rng, *, cp):
    """A device stand-in for a DomainState on its own ladder, carrying what
    the coupler and its vertical step read."""
    from types import SimpleNamespace

    coord = _coord(eta)
    nz = len(eta) - 1
    f32 = lambda a: cp.asarray(np.asarray(a, dtype=np.float32))  # noqa: E731
    state = SimpleNamespace()
    for name in ("c1h", "c2h", "c3h", "c4h", "c1f", "c2f", "c3f", "c4f",
                 "dnw", "rdnw", "rdn", "znw", "znu"):
        setattr(state, name, f32(getattr(coord, name)))
    dn, dnw, fnp, fnm = coord.dn, coord.dnw, coord.fnp, coord.fnm
    cof1 = (2.0 * dn[1] + dn[2]) / (dn[1] + dn[2]) * dnw[0] / dn[1]
    cof2 = dn[1] / (dn[1] + dn[2]) * dnw[0] / dn[2]
    state.cf1, state.cf2 = np.float32(fnp[1] + cof1), np.float32(fnm[1] - cof1 - cof2)
    state.cf3 = np.float32(cof2)
    state.cfn, state.cfn1 = np.float32(1.0 + fnp[-1]), np.float32(-fnp[-1])
    state.p_top = np.float32(5000.0)
    state.mub2d = f32(90000.0 + 500.0 * rng.standard_normal((ny, nx)))
    state.mup = f32(50.0 * rng.standard_normal((ny, nx)))
    state.ht = f32(np.abs(800.0 * rng.standard_normal((ny, nx))))
    state.thb = f32(np.linspace(295.0, 330.0, nz))
    z = np.linspace(0.0, 1.0, nz)[:, None, None]
    state.u = f32(5 + 20 * z + rng.standard_normal((nz, ny, nx + 1)))
    state.v = f32(-2 + 10 * z + rng.standard_normal((nz, ny + 1, nx)))
    state.w = f32(0.1 * rng.standard_normal((nz + 1, ny, nx)))
    state.thp = f32(rng.standard_normal((nz, ny, nx)))
    state.php = f32(10.0 * rng.standard_normal((nz + 1, ny, nx)))
    state.qv = f32(np.abs(0.01 * (1 - z) + 1e-4 * rng.standard_normal((nz, ny, nx))))
    for name in ("qc", "qr", "qi", "qs", "qg", "nr", "ni", "ns", "ng"):
        setattr(state, name, cp.zeros((nz, ny, nx), dtype=cp.float32))
    state.msft = cp.ones((ny, nx), dtype=cp.float32)
    state.msfu = cp.ones((ny, nx + 1), dtype=cp.float32)
    state.msfv = cp.ones((ny + 1, nx), dtype=cp.float32)
    state.has_msf = True
    state.lateral_boundaries = None
    store = {}

    def scratch(shape, slot, dtype=None):
        dtype = cp.dtype(cp.float32 if dtype is None else dtype)
        if slot not in store:
            store[slot] = cp.zeros(tuple(shape), dtype=dtype)
        return store[slot]

    state.scratch = scratch
    return state


@requires_gpu
@pytest.mark.gpu
def test_the_force_corridor_runs_the_vertical_step_on_the_sint_donors():
    """NestCoupler.force on an edge crossing two ladders, against the same
    operators assembled by hand from the CPU twin: the parent coupled on
    its donor columns, interpolated, t/QV/ph rebalanced on the parent's
    mub/mu/ht there, then bdy_interp1 on the cropped registration.  Every
    table word must match."""
    import cupy as cp
    from types import SimpleNamespace

    from woof.config import RunConfig
    from woof.core.clock import DomainClock, DomainTicks
    from woof.core.nest import NestCoupler
    from woof.core.nest_interp import bdy_interp1, window_registration
    from woof.ingest.lateral_bc import couple_nest_field

    rng = np.random.default_rng(11)
    prun = RunConfig(nx=16, ny=16, nz=len(SHARED_ETA) - 1, dx=3000.0,
                     dy=3000.0, ztop=20000.0, dt=9.0, run_seconds=90.0,
                     nested=False, specified=True, grid_id=1,
                     spec_bdy_width=5, spec_zone=1, relax_zone=4,
                     moist=True, mp_physics=8)
    crun = RunConfig(nx=30, ny=30, nz=len(CHILD_ETA) - 1, dx=1000.0,
                     dy=1000.0, ztop=20000.0, dt=3.0, run_seconds=90.0,
                     nested=True, specified=False, grid_id=2,
                     spec_bdy_width=5, spec_zone=1, relax_zone=4,
                     moist=True, mp_physics=8, eta_levels=tuple(CHILD_ETA))

    def clock(grid_id, parent_id, step_ticks, dt, advanced=False):
        spec = DomainTicks(
            grid_id=grid_id, parent_id=parent_id, parent_time_step_ratio=3,
            step_ticks=step_ticks, dt_fp32=np.float32(dt), history_ticks=100,
            restart_ticks=None, radt_ticks=None, stepra=None,
            cudt_ticks=None, stepcu=None, bldt_ticks=None, stepbl=None)
        out = DomainClock(spec, tick_den=1, run_ticks=1000)
        if advanced:
            out.advance()
        return out

    parent = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1, parent_id=0, parent_grid_ratio=1,
                            i_parent_start=1, j_parent_start=1, run=prun),
        state=_ladder_state(16, 16, SHARED_ETA, rng, cp=cp),
        clock=clock(1, 0, 3, 9.0, advanced=True))
    child_clock = clock(2, 1, 1, 3.0)
    child_clock.prepare_step()
    child = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=2, parent_id=1, parent_grid_ratio=3,
                            i_parent_start=4, j_parent_start=4, run=crun),
        state=_ladder_state(30, 30, CHILD_ETA, rng, cp=cp), parent=parent,
        clock=child_clock)
    coupler = NestCoupler(child)
    assert coupler.crosses_ladders
    plan = coupler.vertical_plan()
    coupler.force(child)

    stagger_of = {"u": "x", "v": "y"}
    hand = {}
    for kind in ("u", "v", "w", "t", "ph", "qv", "qc"):
        reg = coupler.registrations[stagger_of.get(kind, "m")]
        cropped, donor = window_registration(
            reg, (slice(0, reg.nyc), slice(0, reg.nxc)))
        name = {"t": "thp", "ph": "php"}.get(kind, kind)
        full = cp.zeros(getattr(parent.state, name).shape, dtype=cp.float32)
        couple_nest_field(parent.state, kind, out=full)
        window = cp.asnumpy(full[(slice(None),) + donor])
        hand[kind] = [plan.interp_cpu(window), cropped]
    cropped, donor = window_registration(
        coupler.registrations["m"],
        (slice(0, coupler.registrations["m"].nyc),
         slice(0, coupler.registrations["m"].nxc)))
    t, qv, ph = plan.rebalance_cpu(
        hand["t"][0], hand["qv"][0], hand["ph"][0],
        cp.asnumpy(parent.state.mub2d[donor]),
        cp.asnumpy(parent.state.mup[donor]),
        cp.asnumpy(parent.state.ht[donor]))
    hand["t"][0], hand["qv"][0], hand["ph"][0] = t, qv, ph
    application = {"t": "theta", "ph": "phi"}
    for kind, (field, reg) in hand.items():
        child_name = {"t": "thp", "ph": "php"}.get(kind, kind)
        child_full = cp.zeros(getattr(child.state, child_name).shape,
                              dtype=cp.float32)
        couple_nest_field(child.state, kind, out=child_full)
        expected = bdy_interp1(
            cp.asarray(field), child_full, reg,
            parent_dt_fp32=parent.clock.dt_fp32,
            parent_interval_ticks=parent.clock.step_ticks,
            spec_zone=1, relax_zone=4, spec_bdy_width=5)
        got = coupler._last_tables[application.get(kind, kind)]
        for side in ("west", "east", "south", "north"):
            for a, b in zip(got[side], expected[side]):
                assert np.array_equal(cp.asnumpy(a).view(np.uint32),
                                      cp.asnumpy(b).view(np.uint32)), \
                    (kind, side)


def test_a_height_limited_nest_under_a_moving_one_is_refused(tmp_path):
    import dataclasses
    from types import SimpleNamespace

    from woof.experiment import _refuse_unsupported_partial_feedback

    exp = load_experiment(_write(tmp_path, d02="feedback_top_m = 3000.0",
                                 experiment="feedback = 1"))
    d02 = exp.domain(2)
    d03 = dataclasses.replace(d02, grid_id=3, parent_id=2,
                              feedback_top=d02.feedback_top)
    mover = dataclasses.replace(d02, feedback_top=None, spawn=object())
    tree = SimpleNamespace(
        domains=(exp.root, mover, d03), vertical=exp.vertical, feedback=1,
        relocation=exp.relocation, devices=exp.devices, tiles=exp.tiles)
    with pytest.raises(ValueError, match="d03.*under d02"):
        _refuse_unsupported_partial_feedback(tree)


def test_an_absent_height_limit_is_absent_from_the_public_documents(
        tmp_path):
    from woof.experiment import domain_config_document
    from woof.ingest.prepared_cache import effective_prepared_domain_config

    exp = load_experiment(_write(tmp_path, experiment="feedback = 1"))
    assert "feedback_top" not in domain_config_document(exp.domain(2))
    limited = load_experiment(_write(tmp_path, d02="feedback_top_m = 3000.0",
                                     experiment="feedback = 1"))
    document = domain_config_document(limited.domain(2))
    assert document["feedback_top"]["top_m"] == 3000.0
    # Preparation never reads it: one prepared bundle serves any limit.
    assert effective_prepared_domain_config(document) == \
        effective_prepared_domain_config(
            domain_config_document(exp.domain(2)))
