"""tools/hex_varres.py: the variable-resolution case recipes and the transition diagnostics (host only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import hex_varres as vr  # noqa: E402


def test_every_case_parses_and_its_references_exist():
    data = vr.load_cases()
    ids = {case["id"] for case in data["cases"]}
    assert {"edi-vr", "edi-u3", "edi-u12", "pdx-vr", "haw-terrain", "haw-ladder", "glob-vr"} <= ids
    kinds = {case["kind"] for case in data["cases"]}
    assert kinds == {"regional", "global"}


def test_regional_arguments_parse_on_the_regional_door(tmp_path):
    from woof.hex.cli import build_parser

    parser = build_parser()
    for case in vr.load_cases()["cases"]:
        if case["kind"] != "regional":
            continue
        argv = vr.case_arguments(case, tmp_path)
        parsed = parser.parse_args(["regional", *argv, "--work-dir", str(tmp_path)])
        assert parsed.fine_dx_m is not None
        if case.get("density"):
            assert parsed.density_raster == [tmp_path / case["density"]["raster"]]


def test_a_density_case_builds_its_raster_before_the_door(tmp_path):
    case = vr.case_row("haw-terrain")
    commands = vr.prep_commands(case, work_dir=tmp_path, cycle="2026-11-07T00", source="gfs", hours=6)
    assert commands[0][:2] == ["hex", "density"] and commands[0][-1] == str(tmp_path / "hawaii-terrain-3km.nc")
    assert commands[1][:2] == ["hex", "regional"]
    assert commands[1][commands[1].index("--density-raster") + 1] == str(tmp_path / "hawaii-terrain-3km.nc")
    assert commands[1][-4:] == ["--skip", "forecast", "--skip", "products"]
    glob_case = vr.prep_commands(vr.case_row("glob-vr"), work_dir=tmp_path, cycle="2026-11-07T00",
                                 source="gfs", hours=6)
    assert glob_case[0][:2] == ["hex", "mesh-plan"] and "--global-mesh" in glob_case[0]
    with pytest.raises(SystemExit, match="--work-dir"):
        vr.case_arguments(case, None)


def _line_mesh(n: int, h: np.ndarray):
    """Cells on a ring (two neighbours each), spacing ``h``."""
    index = np.stack([(np.arange(n) - 1) % n, (np.arange(n) + 1) % n], axis=1)
    return index, np.ones_like(index, dtype=bool)


def test_gradient_and_two_dx_noise_on_a_ring():
    h = np.array([3.0, 3.0, 3.0, 6.0, 6.0, 6.0])
    index, valid = _line_mesh(6, h)
    g = vr.spacing_gradient(h, index, valid)
    assert g.tolist() == pytest.approx([2 / 3, 0, 2 / 3, 2 / 3, 0, 2 / 3])
    field = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    assert vr.two_dx_noise(field, index, valid).tolist() == pytest.approx([2, -2, 2, -2, 2, -2])
    smooth = np.full((6, 3), 4.0)
    assert np.all(vr.two_dx_noise(smooth, index, valid) == 0)


def test_the_screen_flags_noise_made_in_a_transition_band_and_not_a_smooth_ramp():
    n = 600
    h = np.concatenate([np.full(200, 3.0), np.linspace(3.0, 12.0, 200), np.full(200, 12.0)])
    index, valid = _line_mesh(n, h)
    g = vr.spacing_gradient(h, index, valid)
    g[200:400] = 0.08  # the band's per-cell gradient as the generator would grade it
    rng = np.random.default_rng(5)
    w = rng.normal(0.0, 0.05, (n, 4))
    key = np.arange(n) // 100
    mask = np.ones_like(w, dtype=bool)
    use = np.ones(n, dtype=bool)

    def screen(field):
        rows = vr.bin_statistics(key, use, h=h, gradient=g, w=field, w_mask=mask,
                                 noise=vr.two_dx_noise(field, index, valid))
        return vr.imprint_screen(rows)

    clean = screen(w)
    assert clean["flagged"] == [] and clean["worst_ratio"] < 1.5
    imprinted = w.copy()
    imprinted[300:400] += 0.5 * (-1.0) ** np.arange(100)[:, None]  # grid-scale noise in the band
    hit = screen(imprinted)
    assert [row["key"] for row in hit["flagged"]] == [3]
    assert hit["flagged"][0]["noise_ratio_to_neighbours"] > vr.IMPRINT_FACTOR
