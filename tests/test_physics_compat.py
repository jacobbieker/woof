"""Fail-closed gates for the Thompson + MYNN + RUC port lane."""

from __future__ import annotations

from dataclasses import replace

import pytest

from woof.config import RunConfig, validate_run_config
from woof.physics_compat import (
    IMPLICIT_RUNTIME_SWITCHES,
    PhysicsCapabilityError,
    SINGLE_DOMAIN_PHYSICS_PROFILES,
    UnsupportedPhysicsSuiteError,
    WSM6_PROFILE_ID,
    WRF_RRTMG_TO_RTE_RRTMGP,
    implicit_runtime_switches,
    pending_wrf_physics_components,
    single_domain_runtime_switches,
    validate_physics_capabilities,
)


def _cfg(**overrides) -> RunConfig:
    values = dict(nx=4, ny=3, nz=4, dx=1000.0, dy=1000.0,
                  ztop=10000.0, dt=5.0, run_seconds=10.0)
    values.update(overrides)
    return RunConfig(**values)



@pytest.fixture
def captured_warnings():
    """Every :func:`woof.explain.warn` record raised inside the block.

    ``woof.explain.warn`` prints to stderr and feeds registered observers;
    it does NOT go through Python's ``warnings`` module, so ``pytest.warns``
    cannot see it.  The observer hook is the tree's own machine-facing
    surface for exactly this, and it gives the two halves as fields rather
    than a line to pattern-match.
    """
    from woof.explain import add_warning_observer, remove_warning_observer

    records: list[dict] = []
    add_warning_observer(records.append)
    try:
        yield records
    finally:
        remove_warning_observer(records.append)

def test_full_target_mynn_ruc_pair_has_no_remaining_blocker(captured_warnings):
    """The target suite's receipt after the ownership port.

    MYNN no longer appears because 5/5 runs, "RUC land-surface model" no
    longer appears because RUC runs, and "Thompson microphysics" no longer
    appears because mp8 is first-class now that the canonical classic
    tables ship as package data (product decision, product/v1 packaging
    lane 2026-07-28 -- the byte validation moved to table load, it did not
    disappear).  The MYNN/RUC pair is now admitted because WRF's ordered
    RUC write-back and SFCDIAGS_RUCLSM ownership are explicit.
    """
    blockers = pending_wrf_physics_components(
        mp_physics=8, sf_sfclay_physics=5, bl_pbl_physics=5,
        sf_surface_physics=3, num_soil_layers=9)
    assert blockers == ()

    # SIX LEVELS IS NO LONGER BLOCKED.  The 'RUC forecast column soil
    # geometry' blocker was retired when the column stopped being pinned to
    # nine: ruc.cu sizes every soil scratch from RUC_NZS, and the host and
    # device lanes resolve the count from the profile.  What replaced it is a
    # WARNING, because what six levels lacks is a WRF forecast oracle, not a
    # code path -- the warn-not-block ruling this file already applies to
    # Noah-MP's column width.
    six = pending_wrf_physics_components(
        mp_physics=6, sf_sfclay_physics=91, bl_pbl_physics=1,
        sf_surface_physics=3, num_soil_layers=6)
    assert six == ()
    assert any("no WRF forecast oracle" in record["action"]
               for record in captured_warnings), captured_warnings

    # WRF's OWN refusal survives untouched: a count init_soil_depth_3 does
    # not tabulate has no zs at all, and that is a blocker at 4 and at 5.
    for count in (4, 5):
        refused = pending_wrf_physics_components(
            mp_physics=6, sf_sfclay_physics=91, bl_pbl_physics=1,
            sf_surface_physics=3, num_soil_layers=count)
        assert [item.component for item in refused] == ["RUC soil geometry"]
        assert refused[0].selectors == (
            ("sf_surface_physics", 3), ("num_soil_layers", count))

    # And the admitted combination reports nothing at all, which is what makes
    # every refusal above a statement rather than a blanket.
    assert pending_wrf_physics_components(
        mp_physics=6, sf_sfclay_physics=91, bl_pbl_physics=1,
        sf_surface_physics=3, num_soil_layers=9) == ()


def test_every_shipped_profile_answers_its_own_implicit_switches():
    """The drift guard for the one authority on moist_cq/top_lid.

    Three places used to decide these: the shipped profiles (read by the
    HRRR root preparer AND the domain wizard) and the WRF namelist
    importer, which invented ``moist_cq = mp_physics > 0`` and WRF's
    open-top Registry default.  For every WSM6-family suite the two
    answers were opposite, which is why a public root prepared from a
    profile could not bind a public hierarchy imported from the same
    namelist.  If a new profile ever disagrees with what this lookup
    returns for its own selectors, that separation is back, and this
    fails.
    """

    for profile in SINGLE_DOMAIN_PHYSICS_PROFILES:
        switches = single_domain_runtime_switches(profile)
        assert switches["moist_cq"] is True, profile
        resolved = implicit_runtime_switches(**switches)
        assert profile in resolved["profiles"], profile
        for name in IMPLICIT_RUNTIME_SWITCHES:
            assert resolved[name] == switches[name], (profile, name)
        assert profile in resolved["source"]

    # An unmatched moist suite still uses WRF's moisture pressure
    # correction; unrelated implicit switches retain their defaults.
    defaults = _cfg()
    unknown = implicit_runtime_switches(
        mp_physics=6, sf_sfclay_physics=1, sf_surface_physics=0,
        bl_pbl_physics=0, cu_physics=0, num_soil_layers=4,
        ra_lw_physics=1, ra_sw_physics=1)
    assert unknown["profiles"] == ()
    assert unknown["moist_cq"] is True
    assert unknown["top_lid"] == defaults.top_lid
    assert "not one of the shipped" in unknown["source"]

    # An empty selection is not a match for everything.
    empty = implicit_runtime_switches()
    assert empty["profiles"] == ()
    assert empty["moist_cq"] is True


@pytest.mark.parametrize("mp,sfclay,expected_cq,matched", [
    (8, 1, True, False),
    (8, 91, True, True),
    (0, 1, True, False),
])
def test_surface_scheme_selection_keeps_the_moist_pressure_correction(
        mp, sfclay, expected_cq, matched):
    resolved = implicit_runtime_switches(
        mp_physics=mp, sf_sfclay_physics=sfclay, sf_surface_physics=2,
        bl_pbl_physics=1, cu_physics=0, num_soil_layers=4,
        ra_lw_physics=4, ra_sw_physics=4)
    assert resolved["moist_cq"] is expected_cq
    assert bool(resolved["profiles"]) is matched
    if not matched:
        assert "WRF moisture pressure correction" in resolved["source"]


def test_no_shipped_selector_reachable_option_is_unimplemented():
    """Why the refusal below is driven against a PERTURBED registry.

    This case used to run the 1/1 WRF RRTM + Dudhia pair through the front
    door and expect a refusal, because ``radiation/wrf-rrtm-dudhia`` was
    registered-but-unimplemented.  814a68ffc ("wire(rrtm): the 1/1 pair
    becomes selectable, and the ladder loses a rung") WIRED that pair --
    woof/config.py stopped raising for ra_lw_physics=1,
    woof/core/physics.py bound (1,1) to RRTMDudhiaRadiation, and the
    registry row flipped to implemented=true.  That lane repointed the
    sibling case in tests/test_physics_registry.py and missed this one,
    so it went on asserting a refusal that had correctly stopped firing.

    THE GUARD DID NOT BREAK.  The branch is still in
    woof/physics_compat.py::_resolve_physics_component_options and still
    refuses; what it lost is a witness.  The one registered-but-
    unimplemented option left in the whole registry is
    ``microphysics/sase``, and it carries NO selectors, so no
    configuration can name it and the selector resolver can never make it
    a candidate.  This test states that fact -- and fails the day it stops
    being true, which is the day a real witness exists and the perturbed
    case below should be replaced by it.
    """

    from woof.physics_registry import physics_registry

    registry = physics_registry()
    unwitnessed = [
        (component_id, option_id)
        for component_id, component in registry["components"].items()
        if component.get("selector_keys")
        for option_id, option in component["options"].items()
        if option.get("implemented") is not True and option.get("selectors")
    ]
    assert unwitnessed == [], (
        "these options are selector-reachable AND unimplemented, so the "
        "front-door refusal has a real witness again: drive "
        "test_front_door_capability_refusal_cites_unimplemented_registry_"
        f"option from {unwitnessed} instead of from a perturbed registry")


def test_front_door_capability_refusal_cites_unimplemented_registry_option(
        monkeypatch):
    """The refusal names the pointer, the blocker and the selectors.

    Driven against a deep copy with ``radiation/wrf-rrtm-dudhia`` flipped
    back to unimplemented -- the exact shape the shipped registry carried
    until 814a68ffc -- because the shipped registry no longer contains a
    selector-reachable unimplemented option (see the test above).  The
    perturbation is on a copy; the shipped registry is not touched.
    """

    from copy import deepcopy

    import woof.physics_registry as registry_module

    blocker = "Not implemented: the 1/1 WRF RRTM+Dudhia pair"
    perturbed = deepcopy(registry_module.physics_registry())
    option = perturbed["components"]["radiation"]["options"][
        "wrf-rrtm-dudhia"]
    assert option["selectors"] == {"ra_lw_physics": 1, "ra_sw_physics": 1}
    option["implemented"] = False
    option["reachability"] = {"state": "unreachable", "blocker": blocker}
    monkeypatch.setattr(registry_module, "physics_registry",
                        lambda *a, **k: perturbed)

    selected = single_domain_runtime_switches(WSM6_PROFILE_ID)
    selected.update(ra_lw_physics=1, ra_sw_physics=1)

    with pytest.raises(PhysicsCapabilityError) as caught:
        validate_physics_capabilities(selected)

    message = str(caught.value)
    assert (
        "woof/physics_registry_v2.json#/components/radiation/options/"
        "wrf-rrtm-dudhia"
    ) in message
    assert blocker in message
    assert "selectors {'ra_lw_physics': 1, 'ra_sw_physics': 1}" in message


def test_the_1_1_pair_that_814a68ffc_wired_is_admitted():
    """The other half of the widening: the pair now RUNS.

    A refusal that stopped firing is only correct if the thing it refused
    works.  This is the positive statement 814a68ffc earned, and it is
    what stops the case above from being restored by reflex.
    """

    selected = single_domain_runtime_switches(WSM6_PROFILE_ID)
    selected.update(ra_lw_physics=1, ra_sw_physics=1)
    resolved = validate_physics_capabilities(selected)
    assert resolved["radiation"] == "wrf-rrtm-dudhia"


def test_mixed_mynn_pairings_follow_wrf_v461_instead_of_a_half_suite_gate():
    """WRF admits three mixed pairings and fatals the other three."""

    for surface, pbl in (
            (5, 5), (1, 5), (91, 5), (5, 0), (1, 0), (91, 0)):
        assert pending_wrf_physics_components(
            mp_physics=6, sf_sfclay_physics=surface,
            bl_pbl_physics=pbl, sf_surface_physics=2,
            num_soil_layers=4) == ()

    for surface, pbl in ((5, 1), (0, 1), (0, 5)):
        blockers = pending_wrf_physics_components(
            mp_physics=6, sf_sfclay_physics=surface,
            bl_pbl_physics=pbl, sf_surface_physics=2,
            num_soil_layers=4)
        assert [item.component for item in blockers] == [
            "WRF v4.6.1 PBL/surface-layer compatibility"]
        assert blockers[0].selectors == (
            ("sf_sfclay_physics", surface), ("bl_pbl_physics", pbl))
        assert "phys/module_physics_init.F:" in blockers[0].missing[1]


def test_target_runconfig_is_admitted_without_substitution(captured_warnings):
    """The target MYNN/RUC suite validates directly.

    RUC at nine layers is admitted, mp8 is first-class (packaged tables;
    the promotion is the same product decision documented in
    ``test_full_target_mynn_ruc_pair_has_no_remaining_blocker``), and the
    surface-driver ownership port makes the MYNN/RUC pair first-class.
    """
    cfg = _cfg(nz=5, moist=True, mp_physics=8, sf_sfclay_physics=5,
               bl_pbl_physics=5, sf_surface_physics=3,
               num_soil_layers=9)
    assert validate_run_config(cfg) is cfg

    # SIX IS ADMITTED NOW, and admitted is not the same as validated: it
    # passes validation and warns, naming exactly what has not been measured.
    six = _cfg(moist=True, mp_physics=6, sf_sfclay_physics=91,
               bl_pbl_physics=1, sf_surface_physics=3, num_soil_layers=6)
    assert validate_run_config(six) is six
    assert any("no WRF forecast oracle" in record["action"]
               for record in captured_warnings), captured_warnings

    # A geometry WRF itself does not tabulate is still refused outright.  On
    # THIS path the refusal comes from woof.config's soil-layer resolver,
    # which runs before the physics-suite check ever sees the config, so it
    # is a ValueError rather than an UnsupportedPhysicsSuiteError.  The
    # physics-suite half of the same refusal is asserted directly against
    # pending_wrf_physics_components in
    # test_full_target_mynn_ruc_pair_has_no_remaining_blocker.
    for count in (4, 5):
        bad = _cfg(moist=True, mp_physics=6, sf_sfclay_physics=91,
                   bl_pbl_physics=1, sf_surface_physics=3,
                   num_soil_layers=count)
        with pytest.raises(ValueError, match="must be 6 or 9"):
            validate_run_config(bad)

    # And the admitted combination validates, which is what makes the two
    # refusals above meaningful rather than a blanket "RUC is unsupported".
    validate_run_config(_cfg(moist=True, mp_physics=6, sf_sfclay_physics=91,
                             bl_pbl_physics=1, sf_surface_physics=3,
                             num_soil_layers=9))


def test_thompson_runconfig_is_admitted_without_environment(monkeypatch):
    """mp8 validates with NO Thompson environment set (first-class contract).

    Table bytes are not this gate's question: the packaged root (or an
    override) is byte-validated at load, where a wrong root fails closed.
    """
    monkeypatch.delenv("WOOF_EXPERIMENTAL_THOMPSON_MP8", raising=False)
    monkeypatch.delenv("WOOF_THOMPSON_TABLE_ROOT", raising=False)
    validate_run_config(_cfg(moist=True, mp_physics=8))


def test_rrtmg_compatibility_token_is_explicit_and_pair_bound():
    native = validate_run_config(_cfg(ra_physics=4))
    assert native.wrf_rrtmg_compatibility == "none"

    imported = validate_run_config(replace(
        native, wrf_rrtmg_compatibility=WRF_RRTMG_TO_RTE_RRTMGP))
    assert imported.wrf_rrtmg_compatibility == WRF_RRTMG_TO_RTE_RRTMGP

    with pytest.raises(ValueError,
                       match="the token is the receipt of the RRTMG 4/4"):
        validate_run_config(_cfg(
            wrf_rrtmg_compatibility=WRF_RRTMG_TO_RTE_RRTMGP))
    with pytest.raises(ValueError, match="wrf_rrtmg_compatibility must"):
        validate_run_config(_cfg(wrf_rrtmg_compatibility="approximate"))


def test_rrtmg_adapter_rejects_cloud_coupling_off():
    """The refusal stands, and it names an EVIDENCE gap, not a physics one.

    The old assertion matched "always on", the phrase that made this an
    option-envelope policy: icloud is WRF's clear-sky switch, woof
    honours it on the 1/1 pair, and the legacy preparation carries both
    of its arms.  What the 4/4 pair lacks is a recorded oracle case at
    icloud=0, so the message has to say which kind of gap it is and where
    the switch does work -- otherwise the next reader reads "not
    implemented" and deletes a refusal that is protecting a receipt.
    """

    with pytest.raises(ValueError) as caught:
        validate_run_config(_cfg(ra_physics=4, icloud=0))
    message = str(caught.value)
    assert "evidence gap" in message
    assert "oracle case at icloud=0" in message
    assert "ra_lw_physics=1" in message and "ra_sw_physics=1" in message


def test_current_noah_contract_rejects_non_four_layer_state():
    with pytest.raises(ValueError, match="must be 4"):
        validate_run_config(_cfg(sf_surface_physics=2, num_soil_layers=9))


def test_a_half_named_component_is_told_which_keys_it_left_out():
    """The message stops printing a value nobody wrote.

    A component is selected by its whole selector tuple, and a mapping
    that names SOME of a component's keys used to be refused with
    ``{'ra_lw_physics': 4, 'ra_sw_physics': None}`` -- the None is this
    resolver's own filler, read back to the user as if they had typed it.
    Selector keys carry no registry default on purpose (filling one in
    would substitute a scheme nobody asked for), so the fix is a distinct
    message naming the keys that carry no value, not a substitution.
    """

    from woof.physics_compat import _resolve_physics_component_options

    with pytest.raises(PhysicsCapabilityError) as caught:
        _resolve_physics_component_options({"ra_lw_physics": 4})
    message = str(caught.value)
    assert "'ra_sw_physics'" in message
    assert "TOGETHER" in message
    assert "None" not in message


def test_an_unimplemented_selector_value_is_told_what_is_implemented():
    """A refusal that lists the values that DO resolve.

    The tuple is refused because no implemented option carries it; the
    reader's next question is what does, and the registry knows.
    """

    from woof.physics_compat import _resolve_physics_component_options

    with pytest.raises(PhysicsCapabilityError) as caught:
        _resolve_physics_component_options(
            {"ra_lw_physics": 7, "ra_sw_physics": 7})
    message = str(caught.value)
    assert "Implemented selector tuples for this component" in message
    assert "'ra_lw_physics': 4" in message


# ---------------------------------------------------------------------------
# The LES-nest tree suite: a mesoscale root over large-eddy nests.
# ---------------------------------------------------------------------------

def _les_tree():
    from woof.physics_compat import (LES_NEST_TREE_PROFILE_ID,
                                     TREE_PHYSICS_PROFILES)

    return TREE_PHYSICS_PROFILES[LES_NEST_TREE_PROFILE_ID]


def _les_tree_text(*, p_top: float = 1000.0, relocation: str = "",
                   nest_extra: str = "") -> str:
    """The tree `woof domain --physics-profile <tree>` writes, rendered.

    The same renderer the wizard's fit and final emission call, with the
    nest columns the door hands it, so a test of this text is a test of
    the emitted file rather than of a hand-copied one.
    """
    import re
    from datetime import datetime

    from woof.domain_wizard import _projection_entries, render_config

    tree = _les_tree()
    text = render_config(
        name="les-tree", start_time=datetime(2026, 4, 8, 4), hours=1,
        projection=_projection_entries(40.0, -100.0, "auto"),
        dims=[(40, 40), (45, 45), (48, 48)], ratios=(5, 3),
        fetch_hints=None, case_data=None, root_dx_m=1500.0,
        profile=tree.root, cumulus_requested=True, nz=256,
        history_interval_s=120.0, nest_history_interval_s=120.0,
        nest_physics=tree.nest_overrides())
    # The default ladder is eta-normalized, so p_top is the one number a
    # deeper column moves (woof domain --p-top-pa writes it the same way).
    text = re.sub(r"^p_top = .*$", f"p_top = {p_top!r}", text, flags=re.M)
    if nest_extra:
        # d03's own table: a key it already states is replaced, any other
        # is added.
        head, d03 = text.split("grid_id = 3\n")
        key, value = (part.strip() for part in nest_extra.split("="))
        d03, count = re.subn(rf"^{key} = .*$", f"{key} = {value}", d03,
                             count=1, flags=re.M)
        if not count:
            d03 = f"{key} = {value}\n" + d03
        text = head + "grid_id = 3\n" + d03
    return text + relocation


def _load(tmp_path, text):
    from woof.experiment import load_experiment

    path = tmp_path / "les_tree.toml"
    path.write_text(text)
    return load_experiment(path)


def test_the_les_tree_suite_pairs_two_registered_templates():
    from woof.physics_registry import physics_registry

    tree = _les_tree()
    templates = physics_registry()["templates"]
    assert templates[tree.root]["components"] == {
        "cumulus": "off", "land_surface": "noah-mp",
        "microphysics": "thompson-mp8", "pbl": "mynn",
        "radiation": "rte-rrtmgp", "surface_layer": "mynn",
        "turbulence": "smagorinsky-2d", "urban": "none"}
    # The nest differs from the root in the PBL and the closure ONLY, so a
    # paired run isolates the closure.
    nest = dict(templates[tree.nest]["components"])
    assert nest.pop("pbl") == "off"
    assert nest.pop("turbulence") == "tke-1.5-order"
    root = dict(templates[tree.root]["components"])
    del root["pbl"], root["turbulence"]
    assert nest == root
    assert tree.root in SINGLE_DOMAIN_PHYSICS_PROFILES
    assert tree.nest in SINGLE_DOMAIN_PHYSICS_PROFILES

    switches = single_domain_runtime_switches(tree.root)
    assert (switches["mp_physics"], switches["bl_pbl_physics"],
            switches["sf_sfclay_physics"], switches["sf_surface_physics"],
            switches["cu_physics"]) == (8, 5, 5, 4, 0)
    assert (switches["ra_lw_physics"], switches["ra_sw_physics"]) == (4, 4)
    assert switches["ra_rrtmg_variant"] == "rte-rrtmgp"
    assert switches["radt"] == 10.0


def test_the_les_tree_nest_columns_are_the_les_closure_and_per_domain():
    from woof.experiment import _DOMAIN_RUN_OVERRIDES

    overrides = _les_tree().nest_overrides()
    assert {key: overrides[key] for key in (
        "bl_pbl_physics", "km_opt", "diff_opt", "isfflx", "mix_isotropic",
        "radt")} == {"bl_pbl_physics": 0, "km_opt": 2, "diff_opt": 2,
                     "isfflx": 1, "mix_isotropic": 1, "radt": 5.0}
    # Every column is one a [[domain]] table may state; a key the loader
    # keeps tree-wide would be refused on the nest the wizard writes it to.
    assert set(overrides) <= set(_DOMAIN_RUN_OVERRIDES)
    # The microphysics, land surface, surface layer and radiation engine
    # stay the root's ([shared]), so no nest column restates them.
    assert not {"mp_physics", "sf_surface_physics", "sf_sfclay_physics",
                "ra_lw_physics", "ra_sw_physics",
                "ra_rrtmg_variant"} & set(overrides)


def test_the_les_tree_is_registry_reachable_on_every_domain(
        tmp_path, captured_warnings):
    """Governed with NO acknowledgement, named or unnamed."""
    from woof.physics_compat import multi_domain_physics_selection

    exp = _load(tmp_path, _les_tree_text())
    settings = {int(domain.grid_id): domain.run for domain in exp.domains}
    tree = _les_tree()
    for profile in (None, tree.tree_id):
        captured_warnings.clear()
        selection = multi_domain_physics_selection(settings, profile=profile)
        for grid_id, domain in selection["domains"].items():
            assert domain["governance"] == {
                "state": "registry-reachable",
                "required_acknowledgement": None,
                "acknowledged": True}, grid_id
        assert selection["acknowledgements"] == []
        assert not [record for record in captured_warnings
                    if "physics tuple" in str(record)
                    or "expert profile" in str(record)]
    assert selection["profile"] == tree.tree_id
    # Both halves are bound, part for part.
    parts = selection["registry_physics"]["parts"]
    assert f"templates.{tree.root}" in parts
    assert f"templates.{tree.nest}" in parts


def test_a_bound_les_tree_refuses_a_nest_running_another_closure(tmp_path):
    from woof.physics_compat import multi_domain_physics_selection

    exp = _load(tmp_path, _les_tree_text(nest_extra="km_opt = 3"))
    settings = {int(domain.grid_id): domain.run for domain in exp.domains}
    with pytest.raises(ValueError, match="d03 does not run the nest half"):
        multi_domain_physics_selection(settings,
                                       profile=_les_tree().tree_id)


def test_a_tree_suite_is_not_a_single_domain_profile():
    from woof.physics_compat import validate_single_domain_physics_profile

    tree = _les_tree()
    with pytest.raises(ValueError, match="is a tree suite") as caught:
        validate_single_domain_physics_profile(tree.tree_id)
    assert tree.root in str(caught.value)


_RELOCATION = """
[relocation]
enabled = true
grid_id = 3
max_move_parent_cells = 3
cadence_seconds = 120.0

[[relocation.move]]
at_seconds = 120.0
di_parent_cells = 1
dj_parent_cells = 0

[relocation.containment]
grid_id = 2
deadband_cells = 4
max_move_parent_cells = 2
"""


@pytest.mark.parametrize("p_top", (270.0, 450.0, 1000.0, 4000.0))
def test_the_les_tree_loads_at_256_levels_under_a_relocating_nest(
        tmp_path, p_top):
    """The balloon tree: 256 levels, a deep top, d03 moving, d02 sliding.

    The loader runs every relocation refusal (slope radiation, shortwave
    interpolation, Noah mosaic, lake) and the vertical contract with the
    authoritative p_top, so a load is the receipt.
    """
    exp = _load(tmp_path, _les_tree_text(p_top=p_top,
                                         relocation=_RELOCATION))
    assert exp.relocation.enabled and exp.relocation.grid_id == 3
    assert exp.relocation.containment.grid_id == 2
    assert exp.vertical.p_top == p_top
    root, *nests = exp.domains
    assert (root.run.nz, root.run.bl_pbl_physics, root.run.km_opt,
            root.run.radt) == (256, 5, 4, 10.0)
    for nest in nests:
        run = nest.run
        assert (run.nz, run.bl_pbl_physics, run.km_opt, run.diff_opt,
                run.isfflx, run.mix_isotropic, run.radt) == (
                    256, 0, 2, 2, 1, 1, 5.0)
        assert (run.mp_physics, run.sf_surface_physics,
                run.sf_sfclay_physics, run.ra_lw_physics,
                run.ra_sw_physics) == (8, 4, 5, 4, 4)
        assert (run.slope_rad, run.swint_opt, run.sf_surface_mosaic,
                run.sf_lake_physics) == (0, 0, 0, 0)


def test_the_relocation_refusals_still_fire_on_the_les_tree(tmp_path):
    """The control: the load above passed because the refusals had nothing
    to refuse, not because they were skipped."""
    with pytest.raises(ValueError, match="slope_rad = 1"):
        _load(tmp_path, _les_tree_text(relocation=_RELOCATION,
                                       nest_extra="slope_rad = 1"))


@pytest.mark.parametrize("root_dx_m, ratios, dims", (
    (1800.0, (3,), [(40, 40), (45, 45)]),
    (1800.0, (5, 3), [(40, 40), (45, 45), (48, 48)]),
    (900.0, (5,), [(40, 40), (45, 45)]),
))
def test_the_wizard_clock_lands_the_nest_radt_on_whole_nest_steps(
        tmp_path, root_dx_m, ratios, dims):
    """The derived root clock is chosen against the nests' own radt.

    At 1.8 km the spacing-derived clock is 200/67 s, on which a 300 s
    nest radt is not a whole number of nest steps; the loader refuses
    that, so the wizard must pick a clock it accepts.
    """
    from datetime import datetime

    from woof.domain_wizard import _projection_entries, render_config

    tree = _les_tree()
    text = render_config(
        name="les-tree", start_time=datetime(2026, 4, 8, 4), hours=1,
        projection=_projection_entries(40.0, -100.0, "auto"),
        dims=dims, ratios=ratios, fetch_hints=None, case_data=None,
        root_dx_m=root_dx_m, profile=tree.root, cumulus_requested=True,
        nz=60, nest_history_interval_s=600.0,
        nest_physics=tree.nest_overrides())
    exp = _load(tmp_path, text)
    assert all(domain.run.radt == 5.0 for domain in exp.domains[1:])


def test_the_tree_runner_asserts_a_tree_suite_half_by_half(tmp_path):
    from woof.prepared_domain_tree_forecast import validate_physics_profile

    tree = _les_tree()
    exp = _load(tmp_path, _les_tree_text())
    receipt = validate_physics_profile(exp, source="gfs",
                                       profile=tree.tree_id)
    assert receipt["profile"] == tree.tree_id
    assert (receipt["tree"]["root"], receipt["tree"]["nest"]) == (
        tree.root, tree.nest)
    assert [row["grid_id"] for row in receipt["validated_domains"]] == [
        1, 2, 3]
    drifted = _load(tmp_path, _les_tree_text(nest_extra="radt = 10.0"))
    with pytest.raises(ValueError, match="d03"):
        validate_physics_profile(drifted, source="gfs",
                                 profile=tree.tree_id)


def test_the_wizard_offers_the_tree_and_writes_its_nest_columns():
    import argparse

    from woof.domain_wizard import _domain_tables, register_cli

    tree = _les_tree()
    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers())
    domain = next(action for action in parser._actions
                  if isinstance(action, argparse._SubParsersAction)
                  ).choices["domain"]
    profile = next(action for action in domain._actions
                   if "--physics-profile" in action.option_strings)
    assert tree.tree_id in profile.choices
    tables = _domain_tables([(40, 40), (45, 45), (48, 48)], (5, 3),
                            root_dx_m=1500.0, profile=tree.root,
                            cumulus_requested=True,
                            nest_physics=tree.nest_overrides())
    assert tables[0]["radt"] == 10.0 and "km_opt" not in tables[0]
    for table in tables[1:]:
        assert {key: table[key] for key in tree.nest_overrides()} == (
            tree.nest_overrides())
        assert table["cu_physics"] == 0
