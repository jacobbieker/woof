"""The wind-energy model column (woof.energy.vertical) and its planner doors.

Pinned here: the standard-atmosphere arithmetic, the 20 km model-top
ceiling (5500 Pa admitted, WRF's 50 hPa refused), the rotor-layer floor,
the 1 -> 0 strictly-decreasing ladder, the physics/dycore level caps, the
source-reach refusal, the CLI wiring (and its hex-swath refusal), and that
both WRF planners write the column verbatim into every TOML they emit.
Generated-column cases need WRF's compute_eta bridge and say so; explicit
``--eta-levels`` cases run anywhere.  CPU only.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path
import tomllib

import numpy as np
import pytest

from conftest import requires_wrf_eta_bridge
from woof.energy import vertical as v
from woof.energy.contracts import load_plan, load_sites

FIXTURES = Path(__file__).parent / "fixtures" / "energy"
SITES_WALES = FIXTURES / "sites_wales.json"
START = "2026-10-01T00"


def rotor_ladder(*, p_top: float = 5500.0, nz: int = 40, dz0: float = 15.0,
                 stretch: float = 1.1, fine: int = 12) -> tuple[float, ...]:
    """An explicit ladder with ``fine`` geometrically stretched layers from
    ``dz0`` in the reference column, then uniform eta to the top."""

    surface = v.isa_height_m(v.REFERENCE_SURFACE_PA)
    heights, dz = [0.0], dz0
    while len(heights) <= fine:
        heights.append(heights[-1] + dz)
        dz *= stretch
    eta = [(v.isa_pressure_pa(z + surface) - p_top)
           / (v.REFERENCE_SURFACE_PA - p_top) for z in heights]
    eta = [round(level, 7) for level in eta]
    remaining = nz - (len(eta) - 1)
    last = eta[-1]
    eta += [round(last * (1.0 - k / remaining), 7)
            for k in range(1, remaining + 1)]
    eta[0], eta[-1] = 1.0, 0.0
    return tuple(eta)


# --------------------------------------------------------------------------
# standard atmosphere


def test_standard_atmosphere_layer_bases():
    assert v.isa_pressure_pa(0.0) == pytest.approx(101325.0)
    assert v.isa_pressure_pa(11000.0) == pytest.approx(22632.06, abs=0.05)
    assert v.isa_pressure_pa(20000.0) == pytest.approx(5474.89, abs=0.05)
    assert v.isa_pressure_pa(32000.0) == pytest.approx(868.02, abs=0.05)
    for z in (0.0, 500.0, 11000.0, 15000.0, 19970.0, 25000.0, 40000.0):
        assert v.isa_height_m(v.isa_pressure_pa(z)) == pytest.approx(
            z, abs=1e-6)


def test_model_top_ceiling_is_20_km():
    assert v.MIN_P_TOP_PA == pytest.approx(5474.88, abs=0.01)
    assert v.isa_height_m(v.DEFAULT_P_TOP_PA) == pytest.approx(19971.0,
                                                               abs=1.0)
    assert v.model_top_refusal(5500.0) is None
    assert v.model_top_refusal(5475.0) is None
    refusal = v.model_top_refusal(5000.0)
    assert refusal is not None and "20.58 km" in refusal
    assert "--p-top-pa >= 5475" in refusal


def test_reference_column_heights():
    eta = (1.0, 0.5, 0.0)
    h = v.column_heights(eta, 5500.0)
    assert h["full_level_m"][0] == 0.0
    assert h["full_level_m"][-1] == pytest.approx(
        v.isa_height_m(5500.0) - v.isa_height_m(1e5))
    assert len(h["mass_level_m"]) == 2
    assert 0.0 < h["mass_level_m"][0] < h["full_level_m"][1] \
        < h["mass_level_m"][1] < h["full_level_m"][2]
    assert v.rotor_layers([0.0, 100.0, 250.0, 260.0]) == 2


# --------------------------------------------------------------------------
# the preset


def test_a_top_above_20_km_is_refused_before_anything_is_generated():
    with pytest.raises(v.VerticalRefused, match="20 km ceiling"):
        v.wind_energy_column(p_top_pa=5000.0)
    with pytest.raises(v.VerticalRefused, match="20 km ceiling"):
        v.wind_energy_column(p_top_pa=5000.0,
                             eta_levels=rotor_ladder(p_top=5000.0))


def test_explicit_ladder_is_used_verbatim_and_reported():
    eta = rotor_ladder()
    column = v.wind_energy_column(eta_levels=eta)
    assert column.eta_levels == eta and column.nz == len(eta) - 1
    assert column.origin == "explicit" and column.controls == {}
    assert column.p_top_pa == v.DEFAULT_P_TOP_PA
    assert column.rotor_layers == 10
    assert column.full_level_m[1] == pytest.approx(15.0, abs=0.2)
    record = column.record()
    assert record["schema"] == "woof-energy.vertical.v1"
    assert record["eta_levels"] == list(eta)
    assert record["model_top_m"] <= v.MAX_MODEL_TOP_M
    json.dumps(record)


def test_too_few_rotor_layers_is_refused():
    from woof.domain_wizard import _ETA_LEVELS

    # The 49-level default ladder resolves 7 layers below 250 m.
    with pytest.raises(v.VerticalRefused, match="7 whole layer"):
        v.wind_energy_column(eta_levels=_ETA_LEVELS)
    assert v.wind_energy_column(eta_levels=_ETA_LEVELS,
                                min_rotor_layers=7).rotor_layers == 7


@pytest.mark.parametrize("ladder, needle", [
    ((1.0, 0.9, 0.9, 0.5, 0.2, 0.0), "decrease strictly"),
    ((1.0, 0.9, 0.95, 0.5, 0.2, 0.0), "decrease strictly"),
    ((0.99, 0.9, 0.8, 0.5, 0.2, 0.0), "exactly 1.0"),
    ((1.0, 0.9, 0.8, 0.5, 0.2, 0.01), "exactly 0.0"),
    ((1.0, 0.5, 0.2, 0.0), "at least 5"),
    ((1.0, 0.9, float("nan"), 0.5, 0.2, 0.0), "not finite"),
    (tuple(np.linspace(1.0, 0.0, 258)), "nz <= 256"),
])
def test_malformed_ladders_are_refused(ladder, needle):
    with pytest.raises(v.VerticalRefused, match=needle):
        v.wind_energy_column(eta_levels=ladder, min_rotor_layers=0)


def test_ladder_and_generator_controls_are_exclusive():
    with pytest.raises(v.VerticalRefused, match="one or the other"):
        v.wind_energy_column(eta_levels=rotor_ladder(), dzbot_m=12.0)


def test_nz_must_agree_with_an_explicit_ladder():
    with pytest.raises(v.VerticalRefused, match="disagrees"):
        v.wind_energy_column(nz=55, eta_levels=rotor_ladder(nz=40))


def test_a_top_the_source_cannot_reach_is_refused():
    # ICON-D2's certified ladder stops at 60 hPa and its fetch cannot extend.
    with pytest.raises(v.VerticalRefused,
                       match=r"preset top 5500 Pa \(no --p-top-pa given\)"):
        v.wind_energy_column(eta_levels=rotor_ladder(), source="icon-d2")
    with pytest.raises(v.VerticalRefused, match="icon-d2"):
        v.wind_energy_column(eta_levels=rotor_ladder(), p_top_pa=5500.0,
                             source="icon-d2")
    column = v.wind_energy_column(eta_levels=rotor_ladder(p_top=6000.0),
                                  p_top_pa=6000.0, source="icon-d2")
    assert column.model_top_m < v.MAX_MODEL_TOP_M
    # GFS extends its fetch upward, so the preset top is servable.
    v.wind_energy_column(eta_levels=rotor_ladder(), source="gfs")


def test_physics_level_caps_are_enforced():
    with pytest.raises(v.VerticalRefused, match="YSU PBL requires"):
        v.check_physics_levels({"nz": 300, "bl_pbl_physics": 1,
                                "mp_physics": 8}, p_top_pa=5500.0,
                               what="d01")
    v.check_physics_levels({"nz": 70, "bl_pbl_physics": 1, "mp_physics": 8},
                           p_top_pa=5500.0, what="d01")


@requires_wrf_eta_bridge
def test_preset_generates_a_rotor_resolving_column():
    column = v.wind_energy_column()
    eta = column.eta_levels
    assert column.nz == v.DEFAULT_NZ == len(eta) - 1
    assert eta[0] == 1.0 and eta[-1] == 0.0
    assert all(a > b for a, b in zip(eta, eta[1:]))
    assert column.origin == "wrf-compute-eta"
    assert column.controls["dzbot"] == v.DEFAULT_DZBOT_M
    assert column.rotor_layers >= v.DEFAULT_MIN_ROTOR_LAYERS
    assert column.full_level_m[1] == pytest.approx(v.DEFAULT_DZBOT_M, rel=0.1)
    assert column.model_top_m == pytest.approx(19971.0, abs=1.0)
    assert column.receipt["algorithm"].startswith("WRF-4.6.1-compute_eta")


@requires_wrf_eta_bridge
def test_wrf_default_surface_stretch_is_not_rotor_resolving():
    with pytest.raises(v.VerticalRefused, match="7 whole layer"):
        v.wind_energy_column(dzstretch_s=1.3)


# --------------------------------------------------------------------------
# CLI


def _parse(*argv):
    from woof.cli import build_parser

    return build_parser().parse_args(list(argv))


def test_plan_parser_reads_the_column_flags(tmp_path):
    eta_file = tmp_path / "eta.txt"
    eta_file.write_text("# a column\n1.0, 0.99\n0.5 0.2\n0.1\n0.0\n")
    args = _parse("energy", "plan", "s.json", "--topology", "wrf-tiles",
                  "--p-top-pa", "5500", "--dzbot", "12", "--dzstretch-s",
                  "1.12", "--dzstretch-u", "1.08", "--max-dz", "900",
                  "-o", "out")
    assert (args.p_top_pa, args.dzbot, args.dzstretch_s, args.dzstretch_u,
            args.max_dz) == (5500.0, 12.0, 1.12, 1.08, 900.0)
    assert v.column_flags_given(args) == [
        "--p-top-pa", "--dzbot", "--dzstretch-s", "--dzstretch-u",
        "--max-dz"]
    args = _parse("energy", "plan", "s.json", "--topology", "wrf-nests",
                  "--eta-levels", str(eta_file), "-o", "out")
    assert args.eta_levels == (1.0, 0.99, 0.5, 0.2, 0.1, 0.0)
    args = _parse("energy", "plan", "s.json", "--topology", "wrf-nests",
                  "--eta-levels", "1,0.5,0", "-o", "out")
    assert args.eta_levels == (1.0, 0.5, 0.0)
    bare = _parse("energy", "plan", "s.json", "--topology", "wrf-nests",
                  "--nz", "60", "-o", "out")
    assert v.column_flags_given(bare) == []
    assert v.column_from_values(**v.column_request(bare), nz=60) is None
    # A long comma list is never mistaken for (or stat()ed as) a file name.
    long_list = ",".join(f"{level:.7f}" for level in rotor_ladder(nz=70))
    assert len(long_list) > 255
    assert _parse("energy", "plan", "s.json", "--topology", "wrf-nests",
                  "--eta-levels", long_list, "-o", "out").eta_levels \
        == rotor_ladder(nz=70)


def test_hex_swath_refuses_the_column_flags(tmp_path, capsys):
    args = _parse("energy", "plan", str(SITES_WALES), "--topology",
                  "hex-swath", "--p-top-pa", "5500", "-o", str(tmp_path))
    assert args.func(args) == 2
    assert "--p-top-pa applies only to --topology wrf-nests and wrf-tiles" \
        in capsys.readouterr().err


def test_the_wrf_planners_refuse_a_top_above_20_km(tmp_path):
    from woof.energy import plan_wrf_nests as nests
    from woof.energy import plan_wrf_tiles as tiles

    sites = load_sites(SITES_WALES)
    with pytest.raises(v.VerticalRefused, match="20 km ceiling"):
        tiles.build_plan(sites, outdir=tmp_path / "t", start=START,
                         hours=6, source="gfs", vram_gib=80.0,
                         column_request={"p_top_pa": 5000.0})
    assert not (tmp_path / "t" / "plan.json").exists()
    with pytest.raises(v.VerticalRefused, match="20 km ceiling"):
        nests.build_plan(sites, outdir=tmp_path / "n", start=START,
                         hours=6, vram_gib=80.0,
                         column_request={"p_top_pa": 5000.0})
    assert not (tmp_path / "n" / "plan.json").exists()


# --------------------------------------------------------------------------
# the planners write the column


def _toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_wrf_tiles_carry_the_column_to_parent_and_leaves(tmp_path):
    eta = rotor_ladder(nz=50)
    eta_file = tmp_path / "eta.txt"
    eta_file.write_text(",".join(repr(level) for level in eta))
    outdir = tmp_path / "plan-tiles"
    args = _parse("energy", "plan", str(SITES_WALES), "--topology",
                  "wrf-tiles", "--dx-m", "100", "--start", START, "--hours",
                  "6", "--source", "gfs", "--vram-gib", "80", "--p-top-pa",
                  "5500", "--eta-levels", str(eta_file), "-o", str(outdir))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert args.func(args) == 0
    record = json.loads(out.getvalue())
    assert record["vertical"]["nz"] == 50
    assert record["vertical"]["p_top_pa"] == 5500.0
    assert record["vertical"]["rotor_layers"] == 10
    plan = load_plan(outdir / "plan.json")
    parent = plan.domain("parent")
    assert parent.extra["vertical"]["record"] == "vertical.json"
    raw = _toml(outdir / parent.config)
    assert raw["shared"]["p_top"] == 5500.0
    assert raw["shared"]["nz"] == 50
    assert tuple(raw["shared"]["eta_levels"]) == eta
    leaves = [d for d in plan.domains if d.role == "child"]
    assert leaves
    for leaf in leaves:
        spec = json.loads((outdir / leaf.config).read_text())
        child = _toml(outdir / spec["child_config"]["path"])
        assert tuple(child["run"]["eta_levels"]) == eta
        assert spec["child_levels"]["nz"] == 50
        assert spec["child_levels"]["stretch"] is None
        assert spec["child_levels"]["column"] == "vertical.json"
    document = json.loads((outdir / "vertical.json").read_text())
    assert document["schema"] == "woof-energy.vertical.v1"
    assert document["eta_levels"] == list(eta)
    assert any("WIND-ENERGY COLUMN" in note for note in plan.notes)


def test_wrf_nests_carry_the_column_in_the_one_config(tmp_path):
    from woof.energy import plan_wrf_nests as nests

    eta = rotor_ladder(p_top=6000.0, nz=45)
    outdir = tmp_path / "plan-nests"
    plan = nests.build_plan(
        load_sites(SITES_WALES), outdir=outdir, dx_m=100.0, start=START,
        hours=6, vram_gib=80.0,
        column_request={"p_top_pa": 6000.0, "eta_levels": eta})
    raw = _toml(outdir / plan.domains[0].config)
    assert raw["shared"]["p_top"] == 6000.0
    assert raw["shared"]["nz"] == 45
    assert tuple(raw["shared"]["eta_levels"]) == eta
    assert plan.domains[0].extra["vertical"]["nz"] == 45
    assert math.isclose(plan.domains[0].extra["vertical"]["p_top_pa"], 6000.0)
    assert (outdir / "vertical.json").is_file()
