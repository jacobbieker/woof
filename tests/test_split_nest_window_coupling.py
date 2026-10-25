"""The nest coupler reads and writes a split grid's windows on its slabs.

A split ([devices]) grid used to be coupled through its whole host store:
every parent step drained every carrier of every slab to the host and the
next sweep copied the store back.  ``NestWindowSource`` now serves a split
grid through :class:`woof.core.nest_operands.SlabField`, which moves only the
coupler's rectangles (:meth:`tilestream.ranks.RankedRun.read_window` and
``write_window``).  ``WOOF_SPLIT_NEST_DRAIN=1`` keeps the drain road for
comparison.

The CPU half holds the routing (no drain is ever asked for) and the window
geometry against the drain road's own scatter and gather plans.  The GPU half
runs a two-slab tree both ways and requires the same bytes, no drain and no
copy back.
"""
from __future__ import annotations

import gc
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core import nest_operands, streaming
from woof.core.devices import DeviceOptions
from woof.core.nest_operands import NestWindowSource, SlabField, slab_run
from tilestream import harness, multigpu
from tilestream.ranks import RankedRun, RankedRunError


class _DrainForbidden(AssertionError):
    pass


class _FakeRun:
    """A split grid's run: windows from a truth array, and no drain."""

    ranked = True

    def __init__(self, members):
        self.raw_store = members
        self.reads, self.writes = [], []

    @property
    def store(self):
        raise _DrainForbidden("the coupler drained a split grid")

    def store_keys(self):
        return self.raw_store.keys()

    def read_window(self, name, window):
        self.reads.append((name, window))
        return self.raw_store[name][(Ellipsis,) + window].copy()

    def write_window(self, name, window, value):
        self.writes.append((name, window))
        self.raw_store[name][(Ellipsis,) + window] = value


class _FakeOwner:
    """``StreamedDomain`` as the coupler sees a split grid's owner."""

    ranked = True
    _geography = {}
    template_state = None

    def __init__(self, run):
        self._run = run

    @property
    def store(self):
        return self._run.store


def _split_state(members):
    run = _FakeRun(members)
    return SimpleNamespace(_streamed_domain=_FakeOwner(run)), run


def test_slab_run_selects_split_grids_and_honours_the_drain_flag(monkeypatch):
    monkeypatch.delenv(nest_operands.SPLIT_DRAIN_ENV, raising=False)
    state, run = _split_state({})
    assert slab_run(state._streamed_domain) is run
    # A [tiles] store and a resident domain keep their roads.
    assert slab_run(SimpleNamespace(ranked=False, _run=run)) is None
    assert slab_run(None) is None
    monkeypatch.setenv(nest_operands.SPLIT_DRAIN_ENV, "1")
    assert slab_run(state._streamed_domain) is None
    monkeypatch.setenv(nest_operands.SPLIT_DRAIN_ENV, "0")
    assert slab_run(state._streamed_domain) is run


def test_drain_flag_restores_the_whole_store_road(monkeypatch):
    monkeypatch.setenv(nest_operands.SPLIT_DRAIN_ENV, "1")
    state, _ = _split_state({"state/mup": np.zeros((4, 5), np.float32)})
    with pytest.raises(_DrainForbidden):
        NestWindowSource(state)


def test_a_split_grid_is_served_by_window_and_never_drained(monkeypatch):
    monkeypatch.delenv(nest_operands.SPLIT_DRAIN_ENV, raising=False)
    mup = np.arange(20, dtype=np.float32).reshape(4, 5)
    u = np.arange(3 * 4 * 6, dtype=np.float32).reshape(3, 4, 6)
    state, run = _split_state({"state/mup": mup, "state/u": u})
    source = NestWindowSource(state)
    assert "state/u" in source.store and "state/qv" not in source.store
    field = source.array("u")
    assert isinstance(field, SlabField)
    assert (field.shape, field.ndim, field.dtype) == (u.shape, 3, u.dtype)
    assert run.reads == []          # shape and membership read nothing
    window = (slice(1, 3), slice(2, 6))
    np.testing.assert_array_equal(field[(...,) + window], u[..., 1:3, 2:6])
    np.testing.assert_array_equal(field[:, 1:3, 2:6], u[..., 1:3, 2:6])
    assert run.reads == [("state/u", window)] * 2
    for bad in (0, (slice(0, 2), slice(1, 3), slice(0, 4)), (..., 1, slice(0, 2))):
        with pytest.raises(TypeError):
            field[bad]
    value = np.full((2, 3), -1, np.float32)
    source.write("mup", (slice(0, 2), slice(1, 4)), value)
    assert run.writes == [("state/mup", (slice(0, 2), slice(1, 4)))]
    np.testing.assert_array_equal(mup[0:2, 1:4], value)
    assert source.store.get("state/qv") is None
    # A member with no horizontal extent is served as the store holds it.
    run.raw_store["state/p_top"] = np.float32(5000.)
    assert source.array("p_top") == np.float32(5000.)


def test_restriction_never_indexes_the_parent_for_a_shape():
    """``_restrict_windowed`` takes the window's shape off its slices; an
    index into a split parent's field would copy the rectangle off its slabs
    for nothing."""
    import inspect
    from woof.core.nest import NestCoupler
    source = inspect.getsource(NestCoupler._restrict_windowed)
    assert "field[(...,) + window]" not in source


def _ranked_geometry(nx, ny, nz, grid, *, halo=6):
    cfg = multigpu.forced_config(nx, ny, nz)
    options = DeviceOptions(count=grid[0] * grid[1], grid=grid,
                            ids=(0,) * (grid[0] * grid[1]))
    run = RankedRun.__new__(RankedRun)
    run.nz, run.ny, run.nx = nz, ny, nx
    run.specs = streaming.ranked_specs(cfg, options, halo=halo)
    shapes = {"mass": (nz, ny, nx), "u": (nz, ny, nx + 1),
              "v": (nz, ny + 1, nx), "w": (nz + 1, ny, nx), "plane": (ny, nx)}
    rng = np.random.default_rng(7)
    run._home = {name: rng.standard_normal(shape).astype(np.float32)
                 for name, shape in shapes.items()}
    run.arrays = []
    for s in run.specs:
        tile = {}
        for name, shape in shapes.items():
            lead = shape[:-2]
            tile[name] = np.full(lead + (s.cny + (name == "v"), s.cnx + (name == "u")),
                                 np.nan, np.float32)
        run.arrays.append(tile)
    run._window_plans = {}
    return run


def _gather_all(run, home):
    """The full gather a sweep runs after a drain, on numpy slabs."""
    for name, full in home.items():
        window = tuple(slice(0, n) for n in full.shape[-2:])
        for rank, slab, part, _ in run._window_pieces(name, window, "gather"):
            run.arrays[rank][name][slab] = full[part]


def _scatter_all(run):
    """The drain: every slab's owned columns into one host array."""
    out = {}
    for name, full in run._home.items():
        landed = np.full(full.shape, np.nan, np.float32)
        count = np.zeros(full.shape[-2:], int)
        window = tuple(slice(0, n) for n in full.shape[-2:])
        for rank, slab, part, _ in run._window_pieces(name, window, "scatter"):
            landed[part] = run.arrays[rank][name][slab]
            count[part[1:]] += 1
        assert (count == 1).all(), name
        out[name] = landed
    return out


@pytest.mark.parametrize("grid", [(1, 2), (2, 1), (2, 2)])
def test_windows_are_the_drain_and_the_copy_back_restricted(grid):
    """Window reads land what a drain lands; window writes leave the slabs
    exactly as a host write plus the full copy back would."""
    run = _ranked_geometry(40, 36, 3, grid)
    _gather_all(run, run._home)
    assert all(not np.isnan(a).any() for tile in run.arrays for a in tile.values())
    drained = _scatter_all(run)
    for name in run._home:
        np.testing.assert_array_equal(drained[name], run._home[name])
    # Windows that cross every seam, touch a domain edge, and sit inside one
    # slab, on every stagger.
    windows = [(slice(10, 26), slice(14, 27)), (slice(0, 5), slice(0, 41)),
               (slice(30, 36), slice(2, 9)), (slice(17, 19), slice(19, 22))]
    for name in list(run._home):
        for y, x in windows:
            full = run._home[name]
            y = slice(y.start, min(y.stop, full.shape[-2]))
            x = slice(x.start, min(x.stop, full.shape[-1]))
            columns = 0
            read = np.full(full.shape[:-2] + (y.stop - y.start, x.stop - x.start),
                           np.nan, np.float32)
            for rank, slab, part, n in run._window_pieces(name, (y, x), "scatter"):
                read[part] = run.arrays[rank][name][slab]
                columns += n
            assert columns == read.shape[-1] * read.shape[-2]
            np.testing.assert_array_equal(read, full[..., y, x])
            # The write: every slab cell the gather maps the window to.
            value = np.random.default_rng(columns).standard_normal(read.shape).astype(np.float32)
            reference = {k: v.copy() for k, v in run._home.items()}
            reference[name][..., y, x] = value
            for rank, slab, part, _ in run._window_pieces(name, (y, x), "gather"):
                run.arrays[rank][name][slab] = value[part]
            expected = _ranked_geometry(40, 36, 3, grid)
            _gather_all(expected, reference)
            for rank, tile in enumerate(run.arrays):
                for member, array in tile.items():
                    np.testing.assert_array_equal(
                        array, expected.arrays[rank][member], err_msg=f"{name} {member} {rank}")
            run._home = reference


def test_window_bounds_refuse_what_no_slab_holds():
    run = _ranked_geometry(40, 36, 3, (1, 2))
    with pytest.raises(RankedRunError, match="non-empty"):
        run._window_bounds("mass", (slice(5, 5), slice(0, 4)))
    with pytest.raises(RankedRunError, match="two horizontal"):
        run._window_bounds("mass", (slice(0, 4),))


# -- the GPU half -------------------------------------------------------------


def _tree(cp, monkeypatch, *, mode, feedback, road, steps=4):
    """A two-slab tree (or a resident parent over a two-slab nest), stepped
    through the executor; the slab windows unless ``road == "drain"``.

    ``cp`` is handed in by the test, so the device import stays in the test
    function the gpu marker reads.
    """
    from woof.core.model import execute_experiment
    from woof.ingest.lateral_bc import bind_lateral_boundary_clock
    from woof.state_digest import canonical_state_digest
    from tilestream import driver, gather, physics_inventory
    from tilestream import test_moving_nest as moving
    from tilestream import test_nest_executor as executor

    if road == "drain":
        monkeypatch.setenv(nest_operands.SPLIT_DRAIN_ENV, "1")
    else:
        monkeypatch.delenv(nest_operands.SPLIT_DRAIN_ENV, raising=False)
    monkeypatch.setattr(executor, "I_PARENT_START", 41)
    monkeypatch.setattr(executor, "J_PARENT_START", 41)
    monkeypatch.setattr(executor, "PARENT_DT", 6.)
    cfg = moving.parent_cfg(nx=96, ny=96, nz=24, dt=6., run_seconds=steps*6.,
                            radt_minutes=.5, cudt_minutes=.5)
    child_cfg = moving.child_cfg(cfg, nx=48, ny=48)
    boundaries = moving.domain_boundaries(cfg, seconds=21600.)
    state, geography = moving.build_parent(cfg, boundaries=boundaries)
    model = executor.assemble(executor.experiment(cfg, child_cfg, steps=steps),
                              state, geography.grid, feedback=feedback)
    bind_lateral_boundary_clock(model.root.state, model.root.clock)
    if feedback:
        executor.initial_feedback(model)
    cards = tuple(range(min(2, cp.cuda.runtime.getDeviceCount())))
    options = DeviceOptions(count=2, grid=(1, 2), ids=(cards[0], cards[-1]))
    steppers = {}
    for node in model.walk_parent_first():
        streaming.prime_lazy_carriers(node.state, node.cfg.run)
        gid = int(node.cfg.grid_id)
        if mode == "nest" and gid == 1:
            continue
        store = {}
        for name, value in streaming.streamed_store_inventory()(node.state).items():
            host = cp.asnumpy(value)
            store[name] = gather.pinned_empty(host.shape, host.dtype)
            store[name][...] = host
        bundle = SimpleNamespace(
            store=store, scalars=physics_inventory.carrier_scalars(node.state),
            geography=driver.geography_store(node.state, host=True),
            boundaries=boundaries if gid == 1 else None, template=node.state)
        owner = streaming.ranked_domain_builder(
            bundle, clock=node.clock, options=options, node=node,
            check_geography=False)(None, node.cfg.run,
                                   streaming.ranked_decision(node.cfg.run, options))
        owner._state = node.state
        node.state._streamed_domain = owner
        steppers[gid] = owner
    try:
        result = execute_experiment(model, steppers=steppers, validate_state=True,
                                    pool_trim_per_period=False)
        # Counted before the digests below drain the stores on purpose.
        reports = {gid: dict(owner._run.output_report)
                   for gid, owner in steppers.items()}
        domains = {}
        for node in model.walk_parent_first():
            owner = steppers.get(int(node.cfg.grid_id))
            digest, fields = moving.carrier_digest(
                node.state if owner is None else owner.store)
            canonical = (canonical_state_digest(node.state, node.clock)
                         if owner is None else owner.canonical_digest(node.clock))
            domains[int(node.cfg.grid_id)] = dict(
                digest=digest, fields=fields, canonical=canonical["sha256"])
        return dict(domains=domains, reports=reports, forces=result.forces,
                    feedback=model.node(2).coupler.feedback_count)
    finally:
        for owner in steppers.values():
            owner.tiled_run.close()
        del model, state, steppers
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("mode, feedback", [("nest", 0), ("nest", 1), ("both", 1)])
def test_slab_windows_match_the_drain_road_with_no_drain(monkeypatch, mode, feedback):
    import cupy as cp
    with monkeypatch.context() as arm:
        drained = _tree(cp, arm, mode=mode, feedback=feedback, road="drain")
    with monkeypatch.context() as arm:
        windowed = _tree(cp, arm, mode=mode, feedback=feedback, road="windows")
    for gid in (1, 2):
        old, new = drained["domains"][gid], windowed["domains"][gid]
        differing = sorted(key for key in old["fields"].keys() | new["fields"].keys()
                           if old["fields"].get(key) != new["fields"].get(key))
        assert not differing, (mode, feedback, gid, differing)
        assert old["canonical"] == new["canonical"]
    assert drained["forces"] == windowed["forces"] == 4
    assert drained["feedback"] == windowed["feedback"]
    split = (2,) if mode == "nest" else (1, 2)
    assert sorted(windowed["reports"]) == sorted(drained["reports"]) == list(split)
    for gid in split:
        old, new = drained["reports"][gid], windowed["reports"][gid]
        # The control: the drain road still drains and copies back.
        assert old["full_drains"] >= 3 and old["full_gathers"] >= 3, (gid, old)
        assert old["window_reads"] == old["window_writes"] == 0
        # The one copy back allowed is the health validator's: it is built
        # off the store before the first step (a drain with nothing to land,
        # so no drain is counted), and the first sweep copies the store back.
        # And one drain is the stop boundary's mandatory health gate: since
        # upstream 1.1.0 a streamed domain's health rules read the store
        # through a drain (woof.core.health._AttachedStoreBundle), so the
        # final full-state check scans the slabs, not a stale mirror.  No
        # nest force drains.
        assert new["full_drains"] <= 1 and new["full_gathers"] <= 1, (gid, new)
        assert new["full_drains"] < old["full_drains"], (gid, old, new)
        assert new["window_reads"] > 0
    # Only a split parent under feedback is written.
    assert (windowed["reports"].get(1, {}).get("window_writes", 0) > 0) == (
        mode == "both" and feedback == 1)
    assert windowed["reports"][2]["window_writes"] == 0
