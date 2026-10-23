"""woof's WRF turbine-list and table readers against WRF v4.7.1's own READs.

The fixtures (tests/data/oracles/windfarm_reader, PROVENANCE.md there) are
module_wind_fitch.F's list-directed READs, compiled byte for byte, run on
two synthetic run directories.  Every REAL word woof reads from those
files, and every table row it hands the Fitch kernel, must equal WRF's:
max ULP 0, raw-word equality.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from woof.core import windfarm_coupling as wc
from woof.verify.urban_oracle import load_case

REPO = Path(__file__).resolve().parents[1]
ORACLE = REPO / "tests" / "data" / "oracles" / "windfarm_reader"
TOOL = REPO / "tools" / "windfarm_reader_wrf471_oracle"
DECKS = ORACLE / "decks"


def _words(a):
    a = np.ascontiguousarray(a)
    return a.view(np.uint32 if a.dtype == np.float32 else np.int32)


def _case(name):
    return load_case(ORACLE / name)


def test_the_fixtures_and_harness_are_the_pinned_bytes():
    for line in (ORACLE / "FIXTURES.sha256").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        got = hashlib.sha256((ORACLE / path).read_bytes()).hexdigest()
        assert got == digest, path
    for line in (ORACLE / "HARNESS.sha256").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        got = hashlib.sha256((TOOL / path).read_bytes()).hexdigest()
        assert got == digest, path
    assert (ORACLE / "SOURCES.sha256").read_text() == "".join(
        line + "\n" for line in (TOOL / "SOURCES.sha256").read_text()
        .splitlines() if not line.startswith("#"))


@pytest.mark.parametrize("deck, ij", [("ij", 1), ("latlon", 0)])
def test_turbine_rows_are_wrfs_words(deck, ij):
    oracle = _case(deck)
    name = "windturbines-ij.txt" if ij else "windturbines.txt"
    rows = wc._wrf_rows(DECKS / deck / name, ij)
    assert len(rows) == int(oracle["nt"])
    assert [r[2] for r in rows] == oracle["nkind"].tolist()
    if ij:
        assert [r[0] for r in rows] == oracle["ival"].tolist()
        assert [r[1] for r in rows] == oracle["jval"].tolist()
    else:
        lat = np.array([r[0] for r in rows], np.float32)
        lon = np.array([r[1] for r in rows], np.float32)
        assert np.array_equal(_words(lat), _words(oracle["lat"]))
        assert np.array_equal(_words(lon), _words(oracle["lon"]))


@pytest.mark.parametrize("deck, ij", [("ij", 1), ("latlon", 0)])
def test_tables_are_wrfs_words(deck, ij):
    oracle = _case(deck)
    name = "windturbines-ij.txt" if ij else "windturbines.txt"
    rows = wc._wrf_rows(DECKS / deck / name, ij)
    for n, (*_, kind) in enumerate(rows):
        t = wc.read_wrf_table(DECKS / deck / f"wind-turbine-{kind}.tbl", kind)
        for field, ours in (("hubheight", t.hub_height_m),
                            ("diameter", t.diameter_m),
                            ("stc", t.standing_thrust),
                            ("npower", t.nominal_power_mw),
                            ("cutin", t.speeds_ms[0]),
                            ("cutout", t.speeds_ms[-1]),
                            ("stc2", t.thrust_coefficient[-1])):
            assert _words(np.float32(ours).reshape(1)) == \
                _words(oracle[field][n:n + 1]), (field, n)
        width = wc.MAXVALS2
        for field, ours in (("turbws", t.speeds_ms),
                            ("turbtc", t.thrust_coefficient),
                            ("turbpw", t.power_kw)):
            padded = np.zeros(width, np.float32)
            padded[:ours.size] = ours
            assert np.array_equal(_words(padded),
                                  _words(oracle[field][kind - 1, :])), field


def test_the_kernel_layout_carries_wrfs_table_rows():
    """What wrf_fitch_layout hands the kernel, row for row."""
    oracle = _case("ij")
    layout, _ = wc.wrf_fitch_layout(DECKS / "ij" / "windturbines-ij.txt",
                                    ij=1, shape=(20, 20))
    kinds = sorted(set(oracle["nkind"].tolist()))
    for c, kind in enumerate(kinds):
        n = int(layout.nval[c])
        for field, ours in (("turbws", layout.speeds_ms),
                            ("turbtc", layout.thrust_coefficient),
                            ("turbpw", layout.power_kw)):
            assert np.array_equal(_words(ours[c, :n]),
                                  _words(oracle[field][kind - 1, :n])), field
            assert not oracle[field][kind - 1, n:].any()


def test_the_decks_exercise_a_word_double_rounding_gets_wrong():
    """Rounding the decimals through binary64 would fail this oracle."""
    texts = []
    for path in sorted(DECKS.rglob("*")):
        if path.is_file():
            for line in path.read_text().splitlines():
                texts += line.replace(",", " ").split()
    wrong = 0
    for token in texts:
        try:
            value = float(token.replace("d", "e").replace("D", "e"))
        except ValueError:
            continue
        if np.float32(value) != wc._f32_decimal(token, "deck"):
            wrong += 1
    assert wrong >= 1
