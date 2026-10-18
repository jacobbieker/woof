"""A delayed nest born from its live parent on the prepared tree (`woof go`).

``[[domain]] start_from = "parent"`` births a delayed nest by interpolating
its LIVE parent onto its own grid and terrain at its start_time, instead of
restoring a cache prepared from the analysis at that instant.  These pin the
config contract (parse, refusals, identity), the preparation's choice of
snapshot and stamp, the prepared tree's admission of a root-dated cache, the
birth's wiring onto the spawn operator, and the resume rule.  The GPU birth
itself is exercised end to end by a real run, not here.
"""
from dataclasses import replace
from datetime import datetime, timedelta
import json
from types import SimpleNamespace

import numpy as np
import pytest

from woof import prepared_domain_tree_forecast as runner
from woof.experiment import (load_experiment, parent_born_domain_ids,
                              refuse_delayed_activation,
                              refuse_parent_born_activation,
                              starts_from_parent, validate_boundary_timing)
from test_experiment import _write

START = datetime(1974, 4, 3, 12)


def _parent_born(tmp_path, *, at="1974-04-03T12:10:00"):
    return load_experiment(_write(
        tmp_path, d02=f"start_time = {at}\nstart_from = \"parent\""))


# --- the config contract -------------------------------------------------

def test_a_parent_born_nest_loads(tmp_path):
    exp = _parent_born(tmp_path)
    assert exp.domain(2).start_from == "parent"
    assert starts_from_parent(exp.domain(2))
    assert not starts_from_parent(exp.domain(1))
    assert parent_born_domain_ids(exp) == (2,)
    assert exp.domain_start_offset_exact(2) == 600


@pytest.mark.parametrize("spelled", ["", "start_from = \"analysis\""])
def test_the_analysis_birth_is_the_off_contract(tmp_path, spelled):
    exp = load_experiment(_write(
        tmp_path, d02=f"start_time = 1974-04-03T12:10:00\n{spelled}"))
    # Spelled or not, the default is None: the state every experiment
    # written before the key existed describes.
    assert exp.domain(2).start_from is None
    assert parent_born_domain_ids(exp) == ()


def test_an_unknown_birth_source_refuses(tmp_path):
    with pytest.raises(ValueError, match="not a birth source"):
        load_experiment(_write(
            tmp_path, d02="start_time = 1974-04-03T12:10:00\n"
                          "start_from = \"wrfinput\""))


def test_a_nest_starting_with_the_experiment_cannot_be_parent_born(tmp_path):
    with pytest.raises(ValueError, match="needs a delayed nest"):
        load_experiment(_write(tmp_path, d02="start_from = \"parent\""))


def test_the_root_cannot_be_parent_born(tmp_path):
    with pytest.raises(ValueError, match="root has no parent"):
        load_experiment(_write(tmp_path, d01="start_from = \"parent\""))


def test_a_parent_birth_needs_no_forcing_seam(tmp_path):
    """Ten minutes in is between two hourly leads: the analysis birth
    refuses it, the parent birth reads no snapshot there and accepts it.
    The parent-step alignment is not relaxed."""
    born = _parent_born(tmp_path)
    validate_boundary_timing(born, 3600)
    analysis = replace(born, domains=tuple(
        replace(dc, start_from=None) for dc in born.domains))
    with pytest.raises(ValueError, match="boundary-forcing cadence"):
        validate_boundary_timing(analysis, 3600)
    off_step = replace(born, domains=tuple(
        replace(dc, start_time=START + timedelta(seconds=630))
        if dc.grid_id == 2 else dc for dc in born.domains))
    with pytest.raises(ValueError, match="parent step boundary"):
        validate_boundary_timing(off_step, 3600)


def test_the_off_contract_leaves_every_identity_byte_alone(tmp_path):
    from woof.core.model import restart_identity_payload
    from woof.experiment import (domain_config_document,
                                  experiment_config_document)

    born = _parent_born(tmp_path)
    analysis = replace(born, domains=tuple(
        replace(dc, start_from=None) for dc in born.domains))
    for doc in (domain_config_document(analysis.domain(2)),
                experiment_config_document(analysis)["domains"][1]):
        assert "start_from" not in doc
    assert "start_from" not in json.dumps(restart_identity_payload(analysis))
    # A declared parent birth binds: it is another initial state.
    assert domain_config_document(born.domain(2))["start_from"] == "parent"
    assert (restart_identity_payload(born)
            != restart_identity_payload(analysis))


def test_routes_without_a_parent_birth_refuse_it_by_name(tmp_path):
    born = _parent_born(tmp_path)
    with pytest.raises(ValueError) as caught:
        refuse_parent_born_activation(born, "woof run (experiment-tree)")
    message = str(caught.value)
    assert "does not birth a nest from its parent" in message
    assert "d02" in message and "`woof go`" in message
    refuse_parent_born_activation(replace(born, domains=tuple(
        replace(dc, start_from=None) for dc in born.domains)), "woof run")
    # The routes with no activation callback at all still refuse any
    # delayed start, and now name both doors that activate one.
    with pytest.raises(ValueError) as caught:
        refuse_delayed_activation(born, "woof stream")
    assert "`woof go`" in str(caught.value)
    assert "`woof run`" in str(caught.value)


def test_woof_run_refuses_a_parent_birth_before_building(tmp_path):
    from woof.core.model import build_experiment

    with pytest.raises(ValueError, match="does not birth a nest"):
        build_experiment(_parent_born(tmp_path), case_data=None)


# --- preparation: the root's snapshot and stamp --------------------------

def test_preparation_reads_the_roots_snapshot_for_a_parent_born_nest(
        tmp_path):
    from woof.ingest.nest_init import _child_snapshot_time, _initial_snapshot

    born = _parent_born(tmp_path, at="1974-04-03T12:30:00").domain(2)
    later = datetime(1974, 4, 3, 12, 30)
    assert _child_snapshot_time(born) is None
    assert _child_snapshot_time(replace(born, start_from=None)) == later
    catalog = SimpleNamespace(snapshots=("t0", "t1"),
                              valid_times=(START, later))
    assert _initial_snapshot(catalog, _child_snapshot_time(born)) == "t0"


def test_a_parent_born_cache_is_stamped_with_the_root(tmp_path):
    from woof.native_domain_artifacts import _domain_valid_time

    exp = _parent_born(tmp_path)
    assert _domain_valid_time(exp, exp.domain(2), START) == START
    dated = replace(exp.domain(2), start_from=None)
    assert _domain_valid_time(exp, dated, START) == START + timedelta(
        minutes=10)


# --- the prepared tree admits a root-dated cache -------------------------

def _reader(stamp):
    return SimpleNamespace(header={"metadata": {"user": {
        "initial_valid_time": stamp}}})


def test_the_prepared_tree_holds_a_parent_born_cache_to_the_root(tmp_path):
    exp = _parent_born(tmp_path)
    child = exp.domain(2)
    assert runner._prepared_valid_time(exp, child) == START
    root = START.isoformat()
    runner._validate_delayed_prepared_time(
        exp, child, _reader(root), {"valid_time": root})
    later = (START + timedelta(minutes=10)).isoformat()
    with pytest.raises(ValueError, match="initial_valid_time"):
        runner._validate_delayed_prepared_time(
            exp, child, _reader(later), {"valid_time": later})
    analysis = replace(child, start_from=None)
    assert runner._prepared_valid_time(exp, analysis) == START + timedelta(
        minutes=10)


def test_a_parent_born_identity_matches_both_preparations(tmp_path):
    """The GFS/mapped hierarchies carry the nest's own start into its
    cache document; the HRRR hierarchy imports its tree from a namelist
    that drops it.  Both prepared the same root-dated child cache, and a
    parent-born live config admits either.  A genuinely different field
    still refuses."""
    from woof.ingest.prepared_cache import (
        compare_prepared_domain_config, prepared_domain_config_identity,
        undelayed_identity_defaults)

    exp = _parent_born(tmp_path)
    child = exp.domain(2)
    live = runner._parent_born_identity(
        prepared_domain_config_identity(child), exp)
    for cached in (prepared_domain_config_identity(child),
                   prepared_domain_config_identity(
                       replace(child, start_time=START, start_from=None))):
        tolerated, differing = compare_prepared_domain_config(
            runner._parent_born_identity(cached, exp), live,
            not_in_use=undelayed_identity_defaults(exp))
        assert differing == []
    moved = prepared_domain_config_identity(
        replace(child, i_parent_start=child.i_parent_start + 1))
    _, differing = compare_prepared_domain_config(
        runner._parent_born_identity(moved, exp), live,
        not_in_use=undelayed_identity_defaults(exp))
    assert differing == ["i_parent_start"]


def test_a_moving_parent_born_nest_refuses(tmp_path, monkeypatch):
    exp = _parent_born(tmp_path)
    monkeypatch.setattr("woof.static.corridor.moving_grid_ids",
                        lambda exp: frozenset({2}))
    with pytest.raises(ValueError, match="relocation target"):
        runner._validate_delayed_prepared_geometry(exp)
    analysis = replace(exp, domains=tuple(
        replace(dc, start_from=None) for dc in exp.domains))
    runner._validate_delayed_prepared_geometry(analysis)


def test_a_nest_born_with_its_delayed_parent_cannot_be_parent_born(
        tmp_path):
    from test_experiment import BASE

    text = BASE.format(experiment="restart_interval_s = 0.0", shared="",
                       d01="", d02="start_time = 1974-04-03T12:30:00")
    text += """
[[domain]]
grid_id = 3
parent_id = 2
i_parent_start = 20
j_parent_start = 20
parent_grid_ratio = 3
parent_time_step_ratio = 3
e_we = 31
e_sn = 31
history_interval_s = 900.0
start_time = 1974-04-03T12:30:00
start_from = "parent"
"""
    with pytest.raises(ValueError, match="starts with its parent d02"):
        load_experiment(_write(tmp_path, text=text))
    later = text.replace('start_time = 1974-04-03T12:30:00\nstart_from',
                         'start_time = 1974-04-03T12:40:00\nstart_from')
    assert parent_born_domain_ids(load_experiment(
        _write(tmp_path, text=later))) == (3,)


def test_a_parent_born_nest_posts_no_start_need(tmp_path):
    from woof.source_posting import nest_start_needs

    exp = _parent_born(tmp_path)
    assert nest_start_needs(exp, source="hrrr") == []


def test_a_streamed_parent_or_child_refuses_a_parent_birth(tmp_path):
    exp = _parent_born(tmp_path)
    runner.refuse_streamed_parent_birth(exp, set())
    for streamed in ({1}, {2}):
        with pytest.raises(ValueError, match="resident on the card"):
            runner.refuse_streamed_parent_birth(exp, streamed)
    analysis = replace(exp, domains=tuple(
        replace(dc, start_from=None) for dc in exp.domains))
    runner.refuse_streamed_parent_birth(analysis, {1, 2})


# --- the birth's wiring ----------------------------------------------------

def test_the_birth_runs_the_spawn_operator_on_the_prepared_ground(
        tmp_path, monkeypatch):
    """The prepared statics are the adopted ground, the prepared grid is
    the child grid, the land comes from the parent against the parent's
    statics under the native land-use identity, the driver is dated at
    the child's start (not the parent's live clock), and the receipt
    lands in the ledger without its live child."""
    import woof.ingest.nest_spawn_init as spawn_init
    import woof.runtime as runtime
    import woof.core.physics_continuation as continuation
    import woof.core.radiation_composition as composition
    from woof.native_wrf_contract import NATIVE_LANDUSE_IDENTITY

    exp = _parent_born(tmp_path)
    parent = SimpleNamespace(cfg=exp.domain(1), clock=SimpleNamespace(
        elapsed_seconds=99999.0), state=SimpleNamespace())
    grid = object()
    node = SimpleNamespace(cfg=exp.domain(2), parent=parent, grid=grid)
    child_static = {"HGT_M": np.zeros((2, 2))}
    parent_static = {"LU_INDEX": np.ones((2, 2))}
    seen = {}
    radiation_callable = object()

    def land(child_dc, parent_node, *, static_fields, parent_static_fields,
             landuse_attrs):
        seen["land"] = (static_fields, parent_static_fields,
                        dict(landuse_attrs))
        return {"fields": {"tsk": np.ones((2, 2))}, "receipt": {"ok": 1}}

    def rebuild(*, exp, data, model, initialized, child_dc, parent_node,
                land, landuse_attrs, radiation_factory):
        seen["date_seconds"] = parent_node.clock.elapsed_seconds
        seen["rebuild_landuse"] = landuse_attrs
        seen["radiation"] = radiation_factory(child_dc, None, 0.0, 0.0)
        initialized.state.physics = SimpleNamespace(
            radiation_callable=radiation_callable)
        return 0.5

    def spawn(child_dc, parent_node, *, static_fields, blend_width,
              scratch_arena, dycore_state_workspace, on_child_built, grid,
              state_digest, trigger_receipt):
        seen["spawn"] = (static_fields, grid, trigger_receipt,
                         scratch_arena, dycore_state_workspace)
        seen["digest"] = state_digest
        child = SimpleNamespace(state=SimpleNamespace(),
                                static_fields=static_fields["static_fields"])
        on_child_built(child, child_dc, parent_node)
        return {"contract": spawn_init.SPAWN_INIT_CONTRACT,
                "trigger": trigger_receipt,
                "land_surface": on_child_built.last_receipt,
                "terrain": {"terrain_max_abs_shift_m": 12.5},
                "child_result": child}

    monkeypatch.setattr(spawn_init, "spawn_child_from_parent", spawn)
    monkeypatch.setattr(spawn_init, "spawn_land_state_from_parent", land)
    monkeypatch.setattr(runtime, "rebuild_child_driver_from_land_state",
                        rebuild)
    monkeypatch.setattr(continuation, "seed_birth_accumulations",
                        lambda state, parent_node, child_dc: {"seeded": 1})
    attached = []
    monkeypatch.setattr(composition, "attach_modern_workspace",
                        lambda radiation, workspace: attached.append(
                            (radiation, workspace)))
    ledger = tmp_path / runner.PARENT_BIRTH_RECEIPTS_NAME
    child, receipt = runner.birth_child_from_parent(
        exp, node, static_fields=child_static,
        parent_static_fields=parent_static, scratch_arena="arena",
        dycore_state_workspace="rebuilt",
        radiation_factory=lambda *_: "radiation",
        radiation_workspace="workspace", receipts_path=ledger)

    statics, used_grid, trigger, arena, workspace = seen["spawn"]
    assert statics["static_fields"] is child_static and used_grid is grid
    assert (arena, workspace) == ("arena", "rebuilt")
    assert trigger["kind"] == "start_time" and trigger["start_from"] == "parent"
    assert trigger["start_time"] == "1974-04-03T12:10:00"
    assert seen["land"][0] is child_static and seen["land"][1] is parent_static
    assert seen["land"][2] == dict(NATIVE_LANDUSE_IDENTITY)
    assert seen["rebuild_landuse"] == dict(NATIVE_LANDUSE_IDENTITY)
    assert seen["date_seconds"] == 600.0
    assert seen["radiation"] == "radiation"
    assert attached == [(radiation_callable, "workspace")]
    assert child.static_fields is child_static
    assert "child_result" not in receipt
    assert receipt["land_surface"]["precipitation_accumulations"] == {
        "seeded": 1}
    row = json.loads(ledger.read_text().splitlines()[0])
    assert row["trigger"]["resumed"] is False
    assert row["terrain"]["terrain_max_abs_shift_m"] == 12.5
    # The real birth keeps the spawn path's parent digests.
    assert seen["digest"] is None
    # A resume's second birth is not a birth: no ledger row, and no
    # whole-parent digest of the startup parent.
    runner.birth_child_from_parent(
        exp, node, static_fields=child_static,
        parent_static_fields=parent_static, resumed=True,
        receipts_path=ledger)
    assert len(ledger.read_text().splitlines()) == 1
    assert seen["spawn"][2]["resumed"] is True
    assert seen["digest"]("any state") is None


def test_a_resume_births_again_only_the_children_alive_in_the_checkpoint(
        tmp_path, monkeypatch):
    import woof.io.restart as restart

    exp = _parent_born(tmp_path)
    headers = {1: {"domain_lifecycle": "STARTED"},
               2: {"domain_lifecycle": "NOT_STARTED"}}
    monkeypatch.setattr(restart, "tree_restart_members",
                        lambda path: {gid: gid for gid in headers})
    monkeypatch.setattr(restart, "read_restart_header",
                        lambda member: headers[member])
    assert runner.parent_births_in_checkpoint("ckpt", exp) == ()
    headers[2] = {"domain_lifecycle": "STARTED"}
    assert runner.parent_births_in_checkpoint("ckpt", exp) == (2,)
    analysis = replace(exp, domains=tuple(
        replace(dc, start_from=None) for dc in exp.domains))
    assert runner.parent_births_in_checkpoint("ckpt", analysis) == ()


def test_the_executor_hands_a_parent_born_child_to_the_route(monkeypatch):
    """The executor contract the prepared tree's initializer stands on: the
    route's initializer is asked once, at the child's own start tick with
    its parent at the same tick, and the child writes nothing before."""
    from woof.core.model import execute_experiment
    from woof.runtime import _submit_tree_history_frame
    from test_delayed_nest_activation import (
        DELAY_SECONDS, _Writers, _stashing_step, _tree)
    from test_model import _Coupler, _HistoryState

    exp, model = _tree(delay_s=DELAY_SECONDS)
    exp = replace(exp, domains=tuple(
        replace(dc, start_from="parent") if dc.grid_id == 2 else dc
        for dc in exp.domains))
    model._activation_context = {"experiment": exp}
    monkeypatch.setattr("woof.core.nest.NestCoupler", _Coupler)
    monkeypatch.setattr("woof.core.dycore.step", _stashing_step)
    born = _HistoryState()
    asked = []

    def initialize(node, clock):
        asked.append((clock.ticks, node.parent.clock.ticks, node._started))
        return (SimpleNamespace(grid=node.grid, state=born),
                SimpleNamespace(initial_result=SimpleNamespace(state=born)))

    writers = _Writers()
    model._io_manager = writers
    execute_experiment(
        model, validate_state=False, delayed_child_initializer=initialize,
        history_handler=lambda _tree, node, ticks: _submit_tree_history_frame(
            writers, node, ticks))
    assert asked == [(DELAY_SECONDS, DELAY_SECONDS, False)]
    assert model.node(2).state is born
    child_ticks = [ticks for gid, ticks, _ in writers.frames if gid == 2]
    assert child_ticks and min(child_ticks) == DELAY_SECONDS
