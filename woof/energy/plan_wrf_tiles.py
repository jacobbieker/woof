"""``woof energy plan --topology wrf-tiles`` (unit 6, stub)."""

from __future__ import annotations

from pathlib import Path

from woof.energy.contracts import EnergyNotImplemented, Plan, SiteSet

TOPOLOGY = "wrf-tiles"


def build_plan(sites: SiteSet, *, outdir: Path, dx_m: float = 100.0,
               corridor_km: float = 2.0, parent_dx_m: float | None = None,
               start: str | None = None, hours: float = 24.0,
               source: str | None = None, card: str | None = None,
               vram_gib: float | None = None, max_domains: int | None = None,
               nz: int | None = None) -> Plan:
    """Emit the domains for ``sites`` under ``outdir`` and return the plan
    (already written to ``outdir/plan.json``)."""

    raise EnergyNotImplemented("woof energy plan --topology wrf-tiles")


def main(args) -> int:
    raise EnergyNotImplemented("woof energy plan --topology wrf-tiles")
