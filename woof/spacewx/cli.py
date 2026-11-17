"""``woof spacewx``: the standalone space-weather doors.

``woof spacewx ionosphere`` evaluates the L3a climatological ionosphere
(:mod:`woof.spacewx.ionosphere`) on a global latitude/longitude grid on the
GPU and writes foF2, hmF2, NmF2, foE, foF1, hmF1, M(3000)F2, the F2
thicknesses and vertical TEC to an ``.npz`` with a JSON receipt.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import numpy as np


def _utc(text: str) -> datetime:
    try:
        value = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(f"woof spacewx: --time {text!r} is not an ISO time (2024-05-10T18:00Z)") from exc
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def ionosphere_main(args) -> int:
    from woof.spacewx import indices
    from woof.spacewx.ionosphere import Ionosphere, IonosphereError, OUTPUTS, grid, receipt

    start = _utc(args.time)
    times = [start + timedelta(hours=i * args.step_hours) for i in range(max(1, args.count))]
    if args.f107 is not None:
        f107_source, record = f"fixed {args.f107}", None
    else:
        record = indices.load(args.indices, needed=times[-1] - timedelta(days=1))
        f107_source = f"{record.source} sha256:{record.sha256}"
    lat, lon = grid(args.dlat, args.dlon)
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    try:
        iono = Ionosphere(lat2, lon2, when=start, family=args.maps,
                          alt_lo=args.alt_min, alt_hi=args.alt_max, nalt=args.cells)
    except IonosphereError as exc:
        raise SystemExit(f"woof spacewx ionosphere: {exc}") from exc
    import cupy as cp
    fields = {k: [] for k in OUTPUTS}
    drivers = []
    for when in times:
        f107 = float(args.f107) if record is None else record.f107_driver(when, args.f107_mode)
        out = iono.evaluate(when, f107)
        for key in OUTPUTS:
            fields[key].append(cp.asnumpy(out[key]).reshape(lat2.shape))
        drivers.append({"time": when.isoformat() + "Z", "f107": f107})
        tec = fields["tec"][-1]
        weights = np.cos(np.deg2rad(lat2))
        print(f"{when:%Y-%m-%dT%H:%MZ}  F10.7 {f107:6.1f}  TEC mean {np.average(tec, weights=weights):6.2f} "
              f"max {tec.max():6.2f} TECU  foF2 max {fields['fof2'][-1].max():5.2f} MHz")
    doc = receipt(iono, f107_source, "fixed" if record is None else args.f107_mode)
    doc["times"] = drivers
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, lat=lat, lon=lon, **{k: np.stack(v) for k, v in fields.items()})
        args.out.with_suffix(".json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.out} and {args.out.with_suffix('.json')}")
    return 0


def register_cli(sub):
    parser = sub.add_parser("spacewx", help="space weather: the opt-in ionosphere and solar layers")
    commands = parser.add_subparsers(dest="spacewx_command", required=True)
    iono = commands.add_parser(
        "ionosphere", help="climatological ionosphere (foF2, hmF2, TEC) on a global grid, on the GPU")
    iono.add_argument("--time", required=True, help="first valid time, ISO UTC (2024-05-10T18:00Z)")
    iono.add_argument("--count", type=int, default=1, help="number of times (default 1)")
    iono.add_argument("--step-hours", type=float, default=1.0, help="hours between times (default 1)")
    source = iono.add_mutually_exclusive_group()
    source.add_argument("--f107", type=float, help="a fixed F10.7 (sfu) instead of the index record")
    source.add_argument("--indices", default="gfz",
                        help="'gfz' (fetched, CC BY 4.0) or a file in GFZ's Kp/ap/F10.7 format")
    iono.add_argument("--f107-mode", choices=("daily", "81day"), default="81day",
                      help="previous day's F10.7obs, or its trailing 81-day mean (default)")
    iono.add_argument("--maps", choices=("ccir", "ursi"), default="ccir", help="foF2 map family (default ccir)")
    iono.add_argument("--dlat", type=float, default=2.5, help="grid spacing in latitude, deg (default 2.5)")
    iono.add_argument("--dlon", type=float, default=5.0, help="grid spacing in longitude, deg (default 5)")
    iono.add_argument("--alt-min", type=float, default=60.0, help="TEC integration floor, km (default 60)")
    iono.add_argument("--alt-max", type=float, default=2000.0, help="TEC integration top, km (default 2000)")
    iono.add_argument("--cells", type=int, default=1940, help="TEC integration cells (default 1940)")
    iono.add_argument("--out", type=Path, help="write the fields to this .npz (receipt beside it)")
    iono.set_defaults(func=ionosphere_main)
    return parser


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="woof")
    register_cli(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
