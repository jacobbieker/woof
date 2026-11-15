"""Synthetic WRF OBS_DOMAIN files for the hex observation-nudging tests (generic stations, no real data).

The layout is WRF v4.7.1 share/wrf_fddaobs_in.F formats 101/102/1021/103/104/105, as
tools/rustwx/crates/rw-obs/src/fddaobs.rs decodes them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

MISSING = -888888.0


@dataclass
class Surface:
    """A surface report: temperature (K), u, v (m/s, earth-relative), RH (%), surface pressure (Pa)."""

    time: str
    lat: float
    lon: float
    elevation: float
    temperature: float = MISSING
    u: float = MISSING
    v: float = MISSING
    rh: float = MISSING
    psfc: float = MISSING
    name: str = "SYNTHETIC"


@dataclass
class Sounding:
    """A sounding: levels of (pressure Pa, height m, temperature K, u, v, RH %)."""

    time: str
    lat: float
    lon: float
    elevation: float
    levels: Sequence[tuple[float, float, float, float, float, float]] = field(default_factory=list)
    name: str = "SYNTHETIC"


def _qc(value: float) -> float:
    return MISSING if value == MISSING else 0.0


def _row(values: Sequence[float], wind: Sequence[int] = ()) -> str:
    text = " "
    for index, value in enumerate(values):
        qc = 129.0 if index in wind and value != MISSING else _qc(value)
        text += f"{value:11.3f} {qc:11.3f} "
    return text + "\n"


def _header(report, platform: str, sounding: bool, count: int) -> str:
    return (f" {report.time}\n"
            f"  {report.lat:9.4f} {report.lon:9.4f} \n"
            f"  {report.name:40}   {report.name:40}\n"
            f"  {platform:16}  {'synthetic':16}  {report.elevation:8.0f}  {'T' if sounding else 'F':>4}  "
            f"{'F':>4}  {count:5d}\n")


def write(path: str | Path, reports) -> Path:
    """Write the reports (chronological) to an OBS_DOMAIN file."""
    path = Path(path)
    text = ""
    for report in reports:
        if isinstance(report, Surface):
            text += _header(report, "FM-12 SYNOP", False, 1)
            text += _row([MISSING, MISSING, report.elevation, report.temperature, report.u, report.v, report.rh,
                          report.psfc, MISSING], wind=(4, 5))
        else:
            text += _header(report, "FM-35 TEMP", True, len(report.levels))
            for level in report.levels:
                text += _row(list(level), wind=(3, 4))
    path.write_text(text)
    return path
