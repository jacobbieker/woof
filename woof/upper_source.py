"""The upper source a regional run fetches above its own column.

:mod:`woof.ingest.upper_splice` joins two mapped columns; this module is
the route glue around it: WHEN a run needs an upper source (its model top
lies above the lower source's certified top and its registry row declares
one -- :func:`woof.source_adapters.upper_source_for`), WHAT to fetch for
it, and the DONOR a preparation process holds to decode that fetch once
and splice it into every forcing time it maps.

The time choice.  HRRR forcing is hourly while the GFS registry row's
cadence is 3 h, so the upper source is fetched on GFS's own HOURLY leads
(pgrb2.0p25 publishes f000..f120 every hour) from ONE cycle: the newest
6-hourly cycle at or before the run's start, leads ``start - cycle``
through ``start - cycle + hours``.  Every forcing time of the run is then
a lead GFS actually published for that instant -- nothing is interpolated
in time -- and the whole window comes from one forecast, as the lower
source's own window does.  The first lead is at most f005, so the upper
column is never more than five hours from an analysis (f011 for a run so
near real time that the newest cycle is not yet usually posted, which is
walked back one cycle rather than waited for).
"""
from __future__ import annotations

import json
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping

import numpy as np


#: Where a run's upper-source fetch lands beside the lower source's own.
UPPER_FETCH_DIRNAME = "upper-source"
#: The upper source's 6-hourly cycles.
_UPPER_CYCLE_HOURS = 6


def _fetch_table_source(fetch_table: Mapping[str, object]) -> str | None:
    source = fetch_table.get("source") if isinstance(fetch_table, Mapping) else None
    return None if source is None else str(source)


def experiment_area(exp) -> str:
    """The GFS crop for an experiment's root domain, as the wizard spells it."""

    from dataclasses import asdict
    from woof.domain_wizard import fetch_area_hint

    run = exp.root.run
    return fetch_area_hint(asdict(exp.projection), int(run.nx), int(run.ny),
                           source="gfs", root_dx_m=float(run.dx))


def upper_fetch_plan(fetch_table: Mapping[str, object], *, start_time: datetime,
                     p_top_pa, exp=None,
                     now: datetime | None = None) -> dict[str, object] | None:
    """The ``woof fetch`` request for a run's upper source, or ``None``.

    ``fetch_table`` is the run's ``[fetch]`` table (lower source, window,
    crop, and the optional ``upper_source`` opt-out); ``start_time`` the
    experiment's start; ``p_top_pa`` its ``[shared].p_top``.  The window
    is the lower fetch's own: the same number of hours, hourly, starting
    at the same valid time, with the same crop -- or, for a lower source
    fetched whole (a table route takes no crop), the root domain's own
    GFS crop from ``exp``.
    """

    from woof.source_adapters import upper_source_for

    source = _fetch_table_source(fetch_table)
    upper = upper_source_for(source, p_top_pa, fetch_table.get("upper_source"))
    if upper is None:
        return None
    if start_time.minute or start_time.second or start_time.microsecond:
        raise ValueError(
            f"the upper source ({upper}) is fetched on whole-hour leads, and "
            f"this run starts at {start_time:%H:%M:%S}")
    cycle = start_time.replace(hour=start_time.hour
                               - start_time.hour % _UPPER_CYCLE_HOURS)
    # A run near real time cannot wait for a cycle still being produced:
    # walk back to the newest cycle the registry row says is usually
    # posted by now (its usual_delay_hours), so the upper fetch never
    # holds the lower one behind it.  An archived window is unaffected.
    from woof.source_adapters import get_source_adapter
    grid = get_source_adapter(upper).cycle_grid
    delay = timedelta(hours=float(getattr(grid, "usual_delay_hours", 0.0) or 0.0))
    current = datetime.utcnow() if now is None else now
    while cycle + delay > current:
        cycle -= timedelta(hours=_UPPER_CYCLE_HOURS)
    lead = int((start_time - cycle).total_seconds() // 3600)
    hours = fetch_table.get("hours")
    if isinstance(hours, bool) or not isinstance(hours, int) or hours < 1:
        raise ValueError(
            "[fetch] hours must name the lower window so the upper source "
            "can be fetched for the same valid times")
    plan: dict[str, object] = {
        "source": upper, "lower_source": source,
        "cycle": cycle.strftime("%Y-%m-%dT%H"), "forecast_start_hour": lead,
        "hours": int(hours), "cadence": 1, "p_top_pa": float(p_top_pa),
    }
    for key in ("area", "point", "radius_km"):
        if fetch_table.get(key) is not None:
            plan[key] = fetch_table[key]
    if "area" not in plan and "point" not in plan and exp is not None:
        plan["area"] = experiment_area(exp)
    if "area" not in plan and "point" not in plan:
        raise ValueError(
            "[fetch] carries no crop (area, or point and radius_km) for the "
            f"upper source ({upper}) to be fetched over")
    return plan


def upper_fetch_arguments(plan: Mapping[str, object], *, out: Path) -> list[str]:
    """``woof fetch`` flags (after ``fetch``) for :func:`upper_fetch_plan`."""

    arguments = ["--source", str(plan["source"]), "--cycle", str(plan["cycle"]),
                 "--hours", str(plan["hours"]), "--cadence", str(plan["cadence"]),
                 "--p-top-pa", f"{float(plan['p_top_pa']):g}"]
    if plan.get("forecast_start_hour"):
        arguments += ["--forecast-start-hour", str(plan["forecast_start_hour"])]
    if plan.get("area") is not None:
        arguments += ["--area", str(plan["area"])]
    else:
        arguments += ["--point", str(plan["point"]),
                      "--radius-km", str(plan["radius_km"])]
    from woof import fetch_endpoints
    from woof.fetch import GFS_CONTAINER_SOURCES
    if (fetch_endpoints.policy_uses_aws(str(plan["source"]))
            and plan["source"] in GFS_CONTAINER_SOURCES):
        arguments += ["--mode", "full-file"]
    return arguments + ["--out", str(out)]


def plan_sentence(plan: Mapping[str, object], *, lower_top_pa: float | None) -> str:
    """The run plan's line for the upper source."""

    top = "" if lower_top_pa is None else f" above its {lower_top_pa:g} Pa top"
    first = int(plan.get("forecast_start_hour") or 0)
    return (f"upper source: {plan['source']} {plan['cycle']}Z "
            f"f{first:03d}-f{first + int(plan['hours']):03d} (hourly) carries "
            f"the {plan['lower_source']} column{top} to p_top "
            f"{float(plan['p_top_pa']):g} Pa; opt out with [fetch] "
            "upper_source = \"none\"")


def fetch_root_manifest_sha256(root: Path) -> str:
    """The digest that binds an upper-source fetch (its SHA256SUMS)."""

    from woof.fetch import sha256_file
    return sha256_file(Path(root) / "SHA256SUMS")


def _unwrapped(longitude, centre):
    return centre + ((np.asarray(longitude) - centre + 180.0) % 360.0 - 180.0)


class UpperSourceDonor:
    """A GFS upper-source fetch, decoded once, spliced into each forcing time.

    ``root`` is a ``woof fetch --source gfs`` directory (its series file,
    fetch manifest and SHA256SUMS).  The bridge decodes the whole window
    on first use; :meth:`splice` then maps the lead valid at the lower
    snapshot's time onto the caller's target points and joins it above
    the lower column.  Thread-safe: boundary hours may be mapped
    concurrently.
    """

    def __init__(self, root: Path, *, bridge: Path | None = None,
                 scratch_parent: Path | None = None) -> None:
        self.root = Path(root).resolve()
        manifest = json.loads((self.root / "fetch-manifest.json").read_text(
            encoding="utf-8"))
        self.source = str(manifest.get("source", "gfs"))
        self.cycle = datetime.strptime(str(manifest["cycle"]),
                                       "%Y-%m-%dT%H:%M:%SZ")
        self.levels_hpa = tuple(float(level) for level in
                                manifest["pressure_levels_hpa"])
        self.manifest_sha256 = fetch_root_manifest_sha256(self.root)
        #: Every lead's GRIB digest as the fetch verified it; the decode
        #: is held to these bytes, so the bound digest is the data used.
        self._source_sha256 = {
            int(row["forecast_hour"]): str(row["sha256"])
            for row in manifest.get("files", ())
            if isinstance(row, dict) and row.get("forecast_hour") is not None
            and "sha256" in row}
        #: The top of the composite column this fetch makes, Pa.
        self.top_pa = float(min(self.levels_hpa)) * 100.0
        self._bridge = bridge
        self._scratch_parent = scratch_parent
        self._lock = threading.Lock()
        self._snapshots = None
        self._by_time: dict[datetime, int] = {}
        self._temporary = None
        self.receipts: list[dict[str, object]] = []

    # -- decode ---------------------------------------------------------
    def _decoded(self):
        if self._snapshots is not None:
            return self._snapshots
        from woof import bridges
        from woof.gfs_direct import (GfsRouteError, _decode_series_bounded,
                                     _load_bridge_snapshots, _read_series)
        from woof.bridges import decode_failure_message

        series = self.root / f"{self.source}-series.tsv"
        records = _read_series(series)
        bridge = Path(self._bridge) if self._bridge is not None else \
            bridges.resolve_source_decoder("gfs")
        self._temporary = tempfile.TemporaryDirectory(
            prefix="woof-upper-source-", dir=self._scratch_parent,
            ignore_cleanup_errors=True)
        scratch = Path(self._temporary.name)
        decoded = scratch / "decoded"
        cycle_text = self.cycle.strftime("%Y-%m-%d %H:%M:%S")
        ladder = ",".join(format(level * 100.0, "g") for level in self.levels_hpa)
        command = [str(bridge), "--series", str(series), str(decoded),
                   cycle_text, "--pressure-levels-pa", ladder]
        completed = _decode_series_bounded(
            command, series=series, records=records, decoded=decoded,
            scratch=scratch, bridge=bridge, cycle_time=self.cycle,
            levels_pa_csv=ladder, environment=None)
        if completed.returncode != 0:
            raise GfsRouteError(decode_failure_message(
                "GFS Rust bridge (upper source)", completed.stderr))
        missing = [hour for hour, _ in records if hour not in self._source_sha256]
        if missing:
            raise ValueError(
                f"the upper-source fetch manifest in {self.root} records no "
                f"digest for lead(s) {missing}; fetch the folder again")
        self._snapshots = _load_bridge_snapshots(
            decoded, self.cycle, records,
            expected_source_sha256={hour: self._source_sha256[hour]
                                    for hour, _ in records},
            expected_levels_hpa=self.levels_hpa)
        self._by_time = {self.cycle + timedelta(hours=hour): index
                         for index, (hour, _) in enumerate(records)}
        return self._snapshots

    def source_snapshot(self, valid_time: datetime, grid, *, seam_bottom_pa: float):
        """The decoded lead at ``valid_time``, cut to the levels and box a splice reads."""

        from woof.ingest.grib import Era5Snapshot
        from woof.ingest.horiz import orient_global_source_longitudes
        from woof.ingest.upper_splice import UpperSpliceRefusal

        with self._lock:
            snapshots = self._decoded()
            index = self._by_time.get(valid_time)
            if index is None:
                raise UpperSpliceRefusal(
                    f"the upper-source fetch at {self.root} holds no lead valid "
                    f"at {valid_time:%Y-%m-%dT%H:%MZ} (it holds "
                    + ", ".join(f"{t:%Y-%m-%dT%HZ}" for t in sorted(self._by_time))
                    + ")")
            snapshot = snapshots[index]
        target = tuple(pair for pair in (grid.latlon_mass(), grid.latlon_u(),
                                         grid.latlon_v()))
        snapshot = orient_global_source_longitudes(
            snapshot, *(pair[1] for pair in target))
        levels = np.asarray(snapshot.levels_hpa, dtype=np.float64)
        # Every level from the seam bottom up, plus the one beneath it.
        keep = levels * 100.0 <= seam_bottom_pa * (1.0 + 1e-9)
        deeper = np.flatnonzero(~keep)
        if deeper.size:
            keep[deeper[np.argmin(levels[deeper])]] = True
        lat = np.asarray(snapshot.latitude)
        lon = np.asarray(snapshot.longitude)
        centre = float(lon[lon.size // 2])
        t_lat = np.concatenate([np.ravel(pair[0]) for pair in target])
        t_lon = _unwrapped(np.concatenate([np.ravel(pair[1]) for pair in target]),
                           centre)
        margin = 2.0
        rows = np.flatnonzero((lat >= t_lat.min() - margin)
                              & (lat <= t_lat.max() + margin))
        cols = np.flatnonzero((lon >= t_lon.min() - margin)
                              & (lon <= t_lon.max() + margin))
        if rows.size < 4 or cols.size < 4:
            raise UpperSpliceRefusal(
                f"the upper-source fetch at {self.root} does not cover the "
                "target domain; fetch it over the run's own [fetch] area")
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        fields = {name: np.asarray(snapshot.fields[name])[keep][:, r0:r1, c0:c1]
                  for name in ("GHT", "T", "RH", "U", "V")}
        return Era5Snapshot(valid_time=snapshot.valid_time,
                            levels_hpa=levels[keep].copy(),
                            latitude=lat[r0:r1].copy(),
                            longitude=lon[c0:c1].copy(), fields=fields)

    # -- splice ---------------------------------------------------------
    def splice(self, lower, grid, *, backend="cuda", target_name: str = "domain"):
        """``lower`` (mapped onto ``grid``) with this upper source joined above it."""

        from woof.ingest.horiz import interpolate_era5_to_lambert
        from woof.ingest.upper_splice import (DEFAULT_SEAM_BOTTOM_PA,
                                              splice_upper_snapshot)

        source = self.source_snapshot(lower.valid_time, grid,
                                      seam_bottom_pa=DEFAULT_SEAM_BOTTOM_PA)
        upper = interpolate_era5_to_lambert(
            source, grid, relative_humidity_convention="water", backend=backend)
        lead = int((lower.valid_time - self.cycle).total_seconds() // 3600)
        spliced, receipt = splice_upper_snapshot(
            lower, upper, upper_label=self.source,
            upper_receipt={"cycle": self.cycle.isoformat(), "lead_hours": lead,
                           "fetch_manifest_sha256": self.manifest_sha256,
                           "target": target_name})
        with self._lock:
            self.receipts.append(receipt)
        return spliced

    def report(self) -> dict[str, object]:
        """The preparation report's ``upper_source`` entry."""

        with self._lock:
            receipts = list(self.receipts)
        worst = {}
        for receipt in receipts:
            for key, value in receipt.get("seam_statistics", {}).items():
                if key.startswith("max_abs_"):
                    worst[key] = max(worst.get(key, 0.0), float(value))
        # The initial state's own splice describes the run; a boundary
        # strip's column top (and so its appended ladder) may differ.
        ordered = sorted(receipts, key=lambda receipt: (
            receipt["valid_time"],
            (receipt.get("upper") or {}).get("target") != "domain 1"))
        first = ordered[0] if ordered else {}
        return {
            "schema": "woof-upper-source-report-v1",
            "source": self.source, "root": str(self.root),
            "cycle": self.cycle.isoformat(),
            "fetch_manifest_sha256": self.manifest_sha256,
            "pressure_levels_hpa": list(self.levels_hpa),
            "splices": len(receipts),
            "valid_times": sorted({receipt["valid_time"] for receipt in receipts}),
            "seam_pa": first.get("seam_pa"),
            "appended_levels_hpa": first.get("appended_levels_hpa"),
            "composite_top_pa": first.get("composite_top_pa"),
            "moisture": first.get("moisture"),
            "worst_seam_statistics": worst,
        }

    def close(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self._snapshots = None


__all__ = ["UPPER_FETCH_DIRNAME", "UpperSourceDonor", "plan_sentence",
           "upper_fetch_arguments", "upper_fetch_plan",
           "fetch_root_manifest_sha256"]
