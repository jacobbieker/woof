"""The energy guide agrees with the code it documents.

Two bindings, in the two kinds ``tests/doc_command_parity.py`` names:

* an ENUMERATION binding: the guide's ``forecast.v1`` tables list exactly
  :data:`FORECAST_VARIABLES` and :data:`FORECAST_COORDINATES`, with the
  same dimensions, units and meanings;
* a COMMAND binding, stricter than the repository-wide membership check:
  every ``woof energy`` line the guide and the example README print is
  handed to the parser that would receive it, so a missing required
  argument or a bad value fails here and not in a reader's shell.
"""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

import pytest

from woof.energy.contracts import FORECAST_COORDINATES, FORECAST_VARIABLES

REPO = Path(__file__).resolve().parents[1]
GUIDE = REPO / "docs" / "energy-forecasts.md"
EXAMPLE = REPO / "configs" / "energy" / "README.md"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from doc_command_parity import code_fragments, doors, resolve_door  # noqa: E402


def _table_rows(text: str, header: str) -> dict[str, list[str]]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    rows: dict[str, list[str]] = {}
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows[cells[0].strip("`")] = cells[1:]
    return rows


def _dims(dims: tuple[str, ...]) -> str:
    return ", ".join(dims)


def test_the_guide_lists_every_forecast_variable_exactly():
    rows = _table_rows(GUIDE.read_text(encoding="utf-8"),
                       "| Variable | Dimensions |")
    assert list(rows) == list(FORECAST_VARIABLES)
    for name, (dims, units, long_name) in FORECAST_VARIABLES.items():
        assert rows[name] == [_dims(dims), units, long_name], name


def test_the_guide_lists_every_forecast_coordinate_exactly():
    rows = _table_rows(GUIDE.read_text(encoding="utf-8"),
                       "| Coordinate | Dimensions |")
    assert list(rows) == list(FORECAST_COORDINATES)
    for name, (dims, units, long_name) in FORECAST_COORDINATES.items():
        assert rows[name] == [_dims(dims), units or "-", long_name], name


def _energy_commands() -> list[tuple[str, int, str]]:
    """``woof energy`` lines inside ```` ```bash ```` fences: the commands a
    reader pastes.  Inline spans and the ``text`` pipeline diagram name
    doors and flags (the membership check covers those) without being
    complete commands."""

    found = []
    for path in (GUIDE, EXAMPLE):
        inside = False
        for lineno, line in enumerate(path.read_text("utf-8").splitlines(),
                                      start=1):
            fence = line.strip()
            if fence.startswith("```"):
                inside = (not inside) and fence == "```bash"
                continue
            body = line.strip()
            if inside and body.startswith("woof energy "):
                found.append((path.name, lineno, body))
    return found


def test_the_command_reader_reads_the_guide():
    assert len(_energy_commands()) >= 12
    assert any(code_fragments(GUIDE.read_text("utf-8")))


def test_the_guide_prints_every_stage():
    stages = {body.split()[2] for _, _, body in _energy_commands()}
    assert {"fetch", "import", "sites", "plan", "run", "extract",
            "rating"} <= stages
    topologies = {match.group(1) for _, _, body in _energy_commands()
                  for match in re.finditer(r"--topology (\S+)", body)}
    assert topologies >= {"wrf-nests", "wrf-tiles", "hex-swath"}


@pytest.mark.parametrize("where,lineno,body", _energy_commands())
def test_every_energy_command_parses(where, lineno, body):
    known = doors()
    door, _ = resolve_door(body, known)
    assert door in known, f"{where}:{lineno}: no door {door!r}"
    argv = shlex.split(body)[len(door.split()):]
    if not argv:
        return
    try:
        known[door].parse_args(argv)
    except SystemExit as error:
        pytest.fail(f"{where}:{lineno}: `{body}` exits {error.code} "
                    f"under the {door} parser")
