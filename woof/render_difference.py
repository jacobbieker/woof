"""Run differences: every requested product drawn as run A minus run B.

``woof render --diff A B`` pairs the two runs' history frames by valid
time and hands each pair to the Rust renderer (``rw_wrfbatch
--diff-against``), which imports both, draws each product from each run's
own inputs, subtracts, and draws the difference on a zero-centred bar.
This module is orchestration only: it reads file NAMES, never file
contents.  The renderer reads the valid time out of each file again and
refuses a pair whose times or grids do not match, so a misnamed file is
caught there by name rather than drawn.

A run is a folder of ``wrfout_dNN_YYYY-MM-DD_HH:MM:SS`` frames (the colon
or the Windows underscore spelling), or a list of such files.  Frames are
paired on ``(domain, valid time)``; a frame with no partner in the other
run is reported, not dropped silently.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

#: The valid time a WRF history file's own name carries.
_WRFOUT_TIME = re.compile(
    r"^wrfout_(?P<grid>d\d{2})_(?P<date>\d{4}-\d{2}-\d{2})[_T]"
    r"(?P<h>\d{2})[:_](?P<m>\d{2})[:_](?P<s>\d{2})")


def wrfout_valid_time(path) -> tuple[str, _dt.datetime] | None:
    """``(domain, valid time)`` from a wrfout NAME, or ``None``."""

    match = _WRFOUT_TIME.match(Path(path).name)
    if match is None:
        return None
    try:
        valid = _dt.datetime.strptime(
            f"{match.group('date')} {match.group('h')}:{match.group('m')}:"
            f"{match.group('s')}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return match.group("grid"), valid


def run_frames(run) -> list[Path]:
    """The history frames of one run: a folder's ``wrfout_d*`` files (in
    time order, lifecycle episodes included) or the files given."""

    if isinstance(run, (str, Path)):
        path = Path(run)
        if path.is_dir():
            from woof import render_layout

            return [frame.path for frame in render_layout.history_frames(path)]
        return [path]
    return [Path(item) for item in run]


@dataclass
class FramePairing:
    """Run A's and run B's frames matched on (domain, valid time)."""

    pairs: list[tuple[str, _dt.datetime, Path, Path]] = field(default_factory=list)
    only_a: list[Path] = field(default_factory=list)
    only_b: list[Path] = field(default_factory=list)


def _index(frames: Iterable[Path], run: str) -> dict:
    index: dict = {}
    for path in frames:
        key = wrfout_valid_time(path)
        if key is None:
            raise ValueError(
                f"run {run}: {path} does not carry a valid time in its name "
                "(wrfout_dNN_YYYY-MM-DD_HH:MM:SS); pass files named that way "
                "or pair the frames yourself")
        if key in index:
            raise ValueError(
                f"run {run}: {index[key]} and {path} are both {key[0]} valid "
                f"{key[1]:%Y-%m-%d %H:%MZ}; one run holds one frame per "
                "domain and valid time")
        index[key] = path
    return index


def pair_frames_by_valid_time(a_frames: Sequence[Path],
                              b_frames: Sequence[Path]) -> FramePairing:
    """Pair two runs' frames on (domain, valid time), in time order.

    Refuses by name a frame whose name carries no valid time and a run
    holding two frames of one domain at one valid time.  Frames with no
    partner are returned in ``only_a`` / ``only_b`` so the caller can say
    which valid times were not differenced.
    """

    a_index = _index(a_frames, "A")
    b_index = _index(b_frames, "B")
    pairing = FramePairing()
    for key in sorted(a_index, key=lambda item: (item[1], item[0])):
        if key in b_index:
            pairing.pairs.append((key[0], key[1], a_index[key], b_index[key]))
        else:
            pairing.only_a.append(a_index[key])
    pairing.only_b = [b_index[key] for key in sorted(
        b_index, key=lambda item: (item[1], item[0])) if key not in a_index]
    return pairing


def unpaired_notice(pairing: FramePairing) -> str | None:
    """One sentence naming the valid times only one run has, or ``None``."""

    parts = []
    for run, frames in (("A", pairing.only_a), ("B", pairing.only_b)):
        if frames:
            times = ", ".join(
                f"{wrfout_valid_time(p)[0]} {wrfout_valid_time(p)[1]:%m/%d %H:%MZ}"
                for p in frames)
            parts.append(f"only run {run} has {times}")
    if not parts:
        return None
    return ("not differenced, because a difference needs both runs at one "
            "valid time: " + "; ".join(parts))


__all__ = ["FramePairing", "pair_frames_by_valid_time", "run_frames",
           "unpaired_notice", "wrfout_valid_time"]
