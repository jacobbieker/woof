"""``woof domain --p-top-pa / --eta-levels / --dzbot ...``: the explicit column.

Absent every flag the emission is unchanged (resolve_vertical_column says
None, and the default ladder and model top are what the existing wizard
tests pin).  Present, ``[shared]`` carries exactly the column named, the
header says where it came from, the file loads, and a column that is not
1 -> 0 strictly decreasing, that a source cannot cover, or that mixes an
explicit ladder with generator controls is refused before any fit.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from conftest import requires_wrf_eta_bridge
from woof.cli import main as cli_main
from woof.domain_wizard import (DEFAULT_MODEL_TOP_PA, _ETA_LEVELS,
                                VerticalColumnRefused, experiment_from_text,
                                resolve_vertical_column, validate_eta_ladder)

LADDER = (1.0, 0.998, 0.995, 0.99, 0.98, 0.96, 0.93, 0.89, 0.84, 0.78,
          0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.12, 0.06, 0.02, 0.0)


def _wizard(tmp_path, *extra):
    out = tmp_path / "column.toml"
    rc = cli_main(["domain", "--point=55.4,-3.2", "--root-dx", "12",
                   "--card", "24gb", "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--out", str(out), *extra])
    return rc, out


def test_no_flag_means_no_explicit_column():
    assert resolve_vertical_column(source="gfs") is None
    assert resolve_vertical_column(source="gfs", nz=60) is None


def test_p_top_alone_keeps_the_default_ladder():
    column = resolve_vertical_column(source="gfs", p_top_pa=5500.0)
    assert column["eta_levels"] == tuple(_ETA_LEVELS)
    assert column["p_top"] == 5500.0 and column["origin"] == "default-ladder"
    resampled = resolve_vertical_column(source="gfs", nz=60, p_top_pa=6000.0)
    assert resampled["nz"] == 60 and len(resampled["eta_levels"]) == 61


def test_explicit_ladder_and_p_top_are_emitted_verbatim(tmp_path, capsys):
    eta_file = tmp_path / "eta.txt"
    eta_file.write_text("# custom\n" + "\n".join(map(repr, LADDER)) + "\n")
    rc, out = _wizard(tmp_path, "--p-top-pa", "5500",
                      "--eta-levels", str(eta_file))
    assert rc == 0, capsys.readouterr().err
    text = out.read_text(encoding="utf-8")
    shared = tomllib.loads(text)["shared"]
    assert shared["p_top"] == 5500.0
    assert shared["nz"] == len(LADDER) - 1
    assert tuple(shared["eta_levels"]) == LADDER
    assert "# VERTICAL COLUMN: nz = 19 mass levels, p_top = 5500 Pa" in text
    exp = experiment_from_text(text, source=str(out))
    assert exp.vertical.p_top == 5500.0
    assert tuple(exp.vertical.eta_levels) == LADDER
    assert "domain: column nz=19, p_top 5500 Pa (~19.97 km" \
        in capsys.readouterr().out


def test_without_flags_the_emission_keeps_the_default_column(tmp_path):
    rc, out = _wizard(tmp_path)
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    shared = tomllib.loads(text)["shared"]
    assert shared["p_top"] == DEFAULT_MODEL_TOP_PA
    assert tuple(shared["eta_levels"]) == tuple(_ETA_LEVELS)
    assert "VERTICAL COLUMN" not in text


@pytest.mark.parametrize("extra, needle", [
    (("--eta-levels", "1,0.9,0.95,0.5,0.2,0"), "decrease strictly"),
    (("--eta-levels", "1,0.9,0.7,0.5,0.2,0.1"), "exactly 0.0"),
    (("--eta-levels", ",".join(map(repr, LADDER)), "--dzbot", "12"),
     "one or the other"),
    (("--eta-levels", ",".join(map(repr, LADDER)), "--nz", "30"),
     "disagrees"),
])
def test_bad_columns_are_refused_at_the_door(tmp_path, capsys, extra, needle):
    rc, out = _wizard(tmp_path, *extra)
    assert rc == 2
    err = capsys.readouterr().err
    assert needle in err and "Traceback" not in err
    assert not out.exists()


def test_a_top_above_the_source_reach_is_refused():
    with pytest.raises(VerticalColumnRefused, match="icon-d2"):
        resolve_vertical_column(source="icon-d2", p_top_pa=5500.0)
    assert resolve_vertical_column(source="icon-d2", p_top_pa=6000.0)


def test_ladder_validation_is_strict():
    assert validate_eta_ladder(LADDER) == LADDER
    with pytest.raises(VerticalColumnRefused, match="nz <= 256"):
        validate_eta_ladder([1.0 - k / 300 for k in range(301)])


@requires_wrf_eta_bridge
def test_generator_controls_build_the_column_with_wrf_compute_eta(
        tmp_path, capsys):
    rc, out = _wizard(tmp_path, "--p-top-pa", "5500", "--dzbot", "12",
                      "--nz", "70")
    assert rc == 0, capsys.readouterr().err
    text = out.read_text(encoding="utf-8")
    shared = tomllib.loads(text)["shared"]
    eta = shared["eta_levels"]
    assert shared["nz"] == 70 and len(eta) == 71
    assert eta[0] == 1.0 and eta[-1] == 0.0
    assert all(a > b for a, b in zip(eta, eta[1:]))
    assert "WRF 4.6.1 compute_eta controls: auto_levels_opt = 2" in text
    assert "dzbot = 12" in text
