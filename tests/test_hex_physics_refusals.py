"""Hex R1 refuses, by name, every option it accepts but cannot honour.

Host only, deterministic (no GPU).  Each setting below passes the engine's
own validator once its companions are given (ifire=2 with sr_x/sr_y, MAV with
windfarm_ij, irrigation with irr_num_hours ...) and then nothing in hex reads
it, so the table's ``refused_settings`` block refuses it before that
validator, with the reason and the closest working alternative.  The options
that are admitted resolve to the same ``woof hex physics resolve`` output as
before (pinned by hash, recorded on the commit before this table row).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json

import pytest

from woof.hex import physics_suites as ps
from woof.hex.cli import main

# (overrides, knob, words of the reason, words of the alternative)
REFUSED = [
    ({"ifire": 1}, "ifire", "fire grid", "biomass_burn_opt"),
    ({"ifire": 2, "sr_x": 2, "sr_y": 2}, "ifire", "fire grid", "biomass_burn_opt"),
    ({"slope_rad": 1}, "slope_rad", "neighbour terrain", "slope_rad=0"),
    ({"topo_shading": 1}, "topo_shading", "shadow", "topo_shading=0"),
    ({"shadlen": 25000.0}, "shadlen", "topo_shading", "topo_shading=0"),
    ({"wind_farm_scheme": "mav", "wind_farms": "farm.csv"}, "wind_farm_scheme",
     "CellPlacement", "fitch_wrf"),
    ({"wind_farm_scheme": "fitch_legacy"}, "wind_farm_scheme", "CellPlacement", "fitch_wrf"),
    ({"windturbines_spec": "windturbines.txt"}, "windturbines_spec", "legacy turbine list",
     "fitch_wrf"),
    ({"windfarm_opt": 1}, "windfarm_opt", "legacy turbine list", "fitch_wrf"),
    ({"sf_surf_irr_scheme": 2, "irr_num_hours": 4, "irr_ph": 1}, "irr_ph",
     "grid index", "irr_ph=0"),
    ({"sf_surf_irr_scheme": 2, "irr_num_hours": 4, "irr_ph": 2}, "irr_ph",
     "RANDOM_NUMBER", "irr_ph=0"),
    ({"lightning_option": 1, "do_radar_ref": 1}, "lightning_option", "discarded",
     "lightning_option=0"),
]


@pytest.mark.parametrize("overrides,knob,reason,alternative", REFUSED,
                         ids=[f"{item[1]}-{i}" for i, item in enumerate(REFUSED)])
def test_each_silent_no_op_is_refused_by_name(overrides, knob, reason, alternative):
    with pytest.raises(ps.PhysicsSelectionRefusal) as caught:
        ps.resolve_physics("rrfs_v2_dev", overrides)
    mine = [item for item in caught.value.refusals if item.knob == knob]
    assert mine, [item.line() for item in caught.value.refusals]
    assert reason in mine[0].reason
    assert alternative in mine[0].follow_up
    # the table's reason, not the engine validator's incidental one
    assert "WOOF itself refuses" not in mine[0].reason


def test_off_values_are_not_refused():
    for overrides in ({"ifire": 0}, {"slope_rad": 0, "topo_shading": 0}, {"lightning_option": 0},
                      {"sf_surf_irr_scheme": 0}, {"irr_ph": 0},
                      {"windturbines_spec": "none"},
                      {"windfarm_opt": 0}, {"wind_farm_scheme": "off"},
                      {"wind_farm_scheme": "fitch_wrf", "wind_farms": "off"}):
        try:
            ps.resolve_physics(None, overrides)
        except ps.PhysicsSelectionRefusal as refusal:
            assert not any(item.knob in ps.load_table()["refused_settings"]["settings"]
                           for item in refusal.refusals), overrides


def test_cld_fraction_thompson_is_admitted_as_icloud_3():
    resolved = ps.resolve_physics(None, {"config_radt_cld_scheme": "cld_fraction_thompson"})
    assert resolved.seam == "engine"
    assert resolved.engine_physics["icloud"] == 3
    assert ps.resolve_physics(None, {"icloud": 3}).engine_physics == resolved.engine_physics
    # no stale text survives in the table
    assert "admits icloud 0 or 1" not in json.dumps(ps.load_table().get("options", {}), default=str)


def test_cld_fraction_thompson_stays_paired_off_mynn():
    with pytest.raises(ps.PhysicsSelectionRefusal) as caught:
        ps.resolve_physics(None, {"config_pbl_scheme": "bl_mynn",
                                  "config_radt_cld_scheme": "cld_fraction_thompson"})
    assert any(item.knob == "config_radt_cld_scheme" for item in caught.value.refusals)


def _resolve_output(args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert main(["physics", "resolve", *args]) == 0
    return hashlib.sha256(out.getvalue().encode()).hexdigest()[:16]


#: sha256[:16] of ``woof hex physics resolve ARGS`` stdout recorded on origin/main
#: (2bd55d9a2), before refused_settings existed.
ADMITTED = [
    ([], "214ec49ca59cf16d"),
    (["--physics-suite", "rrfs_v2_dev"], "b2fa7a27c16f90d8"),
    (["--physics", "ifire=0"], "761f1b8678879127"),
    (["--physics", "slope_rad=0", "--physics", "topo_shading=0"], "03e7ae03a4e6387d"),
    (["--physics", "lightning_option=0"], "816e254d88bc4052"),
    (["--physics", "sf_surf_irr_scheme=0"], "de10baa119b675ce"),
    (["--physics", "windturbines_spec=none"], "06e6d38eb1693288"),
    (["--physics", "config_microp_scheme=mp_thompson"], "72595be57071adb4"),
    (["--physics", "wind_farm_scheme=off"], "ef15919734737cd9"),
]


@pytest.mark.parametrize("args,digest", ADMITTED, ids=[" ".join(a) or "default" for a, _ in ADMITTED])
def test_admitted_options_resolve_unchanged(args, digest):
    assert _resolve_output(args) == digest


def test_the_cli_refuses_a_named_setting(capsys):
    code = main(["physics", "resolve", "--physics", "slope_rad=1"])
    captured = capsys.readouterr()
    assert code != 0
    report = json.loads(captured.out)
    assert report["resolved"] is False
    assert [(item["option"], item["follow_up"]) for item in report["refusals"]] == [
        ("slope_rad", "slope_rad=0")]


def test_an_explicit_off_windfarm_opt_resolves():
    # CodeRabbit: windfarm_opt has no RunConfig field, so its off value must not
    # reach the engine path, which rejects unknown names.
    resolved = ps.resolve_physics(None, {"windfarm_opt": 0})
    assert resolved.seam in ("engine", "frozen")
    assert "windfarm_opt" not in resolved.settings
