"""Freeze the Dynamical.org ASOS station table that ships with woof.

``woof/obs/data/dynamical_asos_stations.json`` is what
:func:`woof.obs.dynamical_asos.stations_in_bbox` answers from, without the
network and without pyarrow.  This tool rebuilds it from one year file of
the archive (default: the latest full year), reading only the ``station``,
``latitude``, ``longitude``, ``elevation``, ``name`` and ``country``
columns by HTTP range request -- a few MB of a ~650 MB file.  Each
station's row is its last report of the year, so a station that moved
during the year is placed where it ended up.

    python -m tools.freeze_dynamical_asos_stations [--year 2025] [--out PATH]

Needs pyarrow (``pip install 'recast-woof[obs]'``).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

DEFAULT_OUT = (Path(__file__).resolve().parents[1] / "woof" / "obs" / "data"
               / "dynamical_asos_stations.json")


def build(year: int, *, refresh: bool = False, timeout: float = 3600.0) -> dict:
    from woof.obs import dynamical_asos

    latest, version, stats = dynamical_asos.read_station_columns(
        year, timeout=timeout, refresh=refresh)
    _finite = dynamical_asos._number
    stations = []
    skipped = []
    for station_id in sorted(latest):
        row = latest[station_id]
        latitude = _finite(row.get("latitude"))
        longitude = _finite(row.get("longitude"))
        elevation = _finite(row.get("elevation"))
        if (not str(station_id).strip() or latitude is None or longitude is None
                or elevation is None or not -90.0 <= latitude <= 90.0):
            skipped.append(station_id)
            continue
        stations.append({
            "station_id": str(station_id),
            "name": str(row.get("name") or ""),
            "latitude": round(latitude, 6),
            "longitude": round(((longitude + 180.0) % 360.0) - 180.0, 6),
            "elevation_m": round(elevation, 3),
            "country": str(row.get("country") or ""),
        })
    return {
        "schema": dynamical_asos.STATION_TABLE_SCHEMA,
        "source": dynamical_asos.SOURCE,
        "source_url": version.url,
        "source_etag": version.etag or version.last_modified,
        "year": int(year),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "attribution": dynamical_asos.ATTRIBUTION,
        "station_count": len(stations),
        "skipped_without_position": skipped,
        "content_sha256": dynamical_asos.station_rows_digest(stations),
        "download_bytes": stats.bytes_downloaded,
        "stations": stations,
    }


def render(record: dict) -> str:
    """One station per line: diffable, and a third the size of indent=2."""

    head = {key: value for key, value in record.items() if key != "stations"}
    lines = json.dumps(head, indent=1)[:-2].rstrip()
    rows = ",\n".join("  " + json.dumps(row, ensure_ascii=False) for row in record["stations"])
    return f"{lines},\n \"stations\": [\n{rows}\n ]\n}}\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int,
                        default=datetime.now(timezone.utc).year - 1,
                        help="the year file to read (default: the latest full year)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="where to write the table (default: the packaged one)")
    parser.add_argument("--refresh", action="store_true",
                        help="revalidate the year file with the host first")
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="the whole read's time budget in seconds")
    args = parser.parse_args(argv)
    record = build(args.year, refresh=args.refresh, timeout=args.timeout)
    from woof.fetch_guard import atomic_write_text
    atomic_write_text(args.out, render(record))
    json.loads(args.out.read_text(encoding="utf-8"))
    print(json.dumps({"out": str(args.out), "year": record["year"],
                      "stations": record["station_count"],
                      "skipped": len(record["skipped_without_position"]),
                      "download_bytes": record["download_bytes"],
                      "size_bytes": args.out.stat().st_size}), file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
