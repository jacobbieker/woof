"""Frame pairing for ``woof render --diff``: two runs matched by valid time.

The renderer checks the valid time and the grid again from the files'
contents; these tests hold the door's half: which frames it hands over as
pairs, which it reports as unpaired, and which it refuses by name.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from woof import render_difference, rustwx


def _touch(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"")
    return path


def test_frames_pair_by_valid_time_whatever_each_run_started_from(tmp_path):
    # Run A started at 12Z, run B at 18Z: they share 18Z and 19Z only.
    a = [_touch(tmp_path / "a", f"wrfout_d01_2025-03-15_{h:02d}_00_00")
         for h in range(12, 20)]
    b = [_touch(tmp_path / "b", f"wrfout_d01_2025-03-15_{h:02d}_00_00")
         for h in (18, 19, 20)]
    pairing = render_difference.pair_frames_by_valid_time(a, b)
    assert [(domain, valid.hour) for domain, valid, _, _ in pairing.pairs] == [
        ("d01", 18), ("d01", 19)]
    for _, valid, a_file, b_file in pairing.pairs:
        assert render_difference.wrfout_valid_time(a_file)[1] == valid
        assert render_difference.wrfout_valid_time(b_file)[1] == valid
    assert len(pairing.only_a) == 6
    assert [p.name for p in pairing.only_b] == ["wrfout_d01_2025-03-15_20_00_00"]
    notice = render_difference.unpaired_notice(pairing)
    assert "only run B has d01 03/15 20:00Z" in notice


def test_two_nests_pair_within_their_own_domain(tmp_path):
    a = [_touch(tmp_path / "a", "wrfout_d01_2025-03-16_00_00_00"),
         _touch(tmp_path / "a", "wrfout_d02_2025-03-16_00_00_00")]
    b = [_touch(tmp_path / "b", "wrfout_d02_2025-03-16_00_00_00")]
    pairing = render_difference.pair_frames_by_valid_time(a, b)
    assert [(d, a_file.name) for d, _, a_file, _ in pairing.pairs] == [
        ("d02", "wrfout_d02_2025-03-16_00_00_00")]
    assert [p.name for p in pairing.only_a] == ["wrfout_d01_2025-03-16_00_00_00"]
    assert render_difference.unpaired_notice(
        render_difference.pair_frames_by_valid_time(b, b)) is None


def test_a_frame_with_no_time_in_its_name_is_refused_by_name(tmp_path):
    a = [_touch(tmp_path, "case6-ifs-3km-scoring.nc")]
    b = [_touch(tmp_path, "wrfout_d01_2025-03-16_00_00_00")]
    with pytest.raises(ValueError, match="case6-ifs-3km-scoring.nc does not carry a valid time"):
        render_difference.pair_frames_by_valid_time(a, b)


def test_one_run_with_two_frames_at_one_valid_time_is_refused(tmp_path):
    a = [_touch(tmp_path / "x", "wrfout_d01_2025-03-16_00_00_00"),
         _touch(tmp_path / "y", "wrfout_d01_2025-03-16_00_00_00")]
    with pytest.raises(ValueError, match="one frame per domain and valid time"):
        render_difference.pair_frames_by_valid_time(a, a[:1])


def test_a_run_folder_lists_its_history_frames(tmp_path):
    for h in (1, 0):
        _touch(tmp_path, f"wrfout_d01_2025-03-16_{h:02d}_00_00")
    _touch(tmp_path, "wrfout_d01_2025-03-16_02_00_00.tmp1")
    _touch(tmp_path, "namelist.input")
    frames = render_difference.run_frames(tmp_path)
    assert [p.name for p in frames] == ["wrfout_d01_2025-03-16_00_00_00",
                                        "wrfout_d01_2025-03-16_01_00_00"]
    assert render_difference.wrfout_valid_time(frames[0]) == (
        "d01", dt.datetime(2025, 3, 16, 0, 0, 0))
    # The colon spelling a Linux history writer uses names the same time.
    assert render_difference.wrfout_valid_time(
        Path("wrfout_d01_2025-03-16_00:00:00")) == (
        "d01", dt.datetime(2025, 3, 16, 0, 0, 0))


def test_the_renderer_difference_line_parses_and_the_contract_names_it():
    row = rustwx.parse_difference_line(
        "DIFFERENCE d01-3km_2m_temperature units=°F half_range=10 step=2 "
        "rule=table defined_cells=1835008 max_abs=14.2")
    assert row == {"key": "d01-3km_2m_temperature", "units": "°F",
                   "half_range": "10", "step": "2", "rule": "table",
                   "defined_cells": "1835008", "max_abs": "14.2"}
    # The run-level line names no product.
    assert rustwx.parse_difference_line(
        'DIFFERENCE valid=2025-03-16 00:00Z a_frame=0 b_frame=0') is None
    assert "\t--diff-against\t" in rustwx.RENDERER_ABI_MARKER
    assert "\tDIFFERENCE" in rustwx.RENDERER_ABI_MARKER
