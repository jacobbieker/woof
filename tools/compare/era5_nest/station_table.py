#!/usr/bin/env python3
"""Station-verification table of one woof run, from its verify-visuals receipts.

``woof go`` (or ``woof verify-visuals RUN_DIR``) writes one
``gpuwm.verify-visuals.receipt.v1`` per verified hour at
``<run>/d01/verification/<date>/verification_<HHMMSS>.json``.  Each carries a
``stations`` list of per-quantity scores against the ASOS reports (forecast
minus observation).  This tool reads the receipts and prints them as one
markdown table.

    python tools/compare/era5_nest/station_table.py RUN_DIR [--domain d01]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RECEIPT_SCHEMA = "gpuwm.verify-visuals.receipt.v1"


def load_receipts(run: Path, domain: str = "d01") -> list[tuple[str, dict]]:
    """``(valid time label, receipt)`` for every verified hour, in time order."""

    found = []
    for path in sorted((run / domain / "verification").glob("*/verification_*.json")):
        if path.name.endswith(".request.json"):
            continue
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("schema") != RECEIPT_SCHEMA:
            continue
        stamp = path.stem.removeprefix("verification_")
        found.append((f"{path.parent.name} {stamp[:2]}:{stamp[2:4]}Z", receipt))
    return found


def rows(receipts: list[tuple[str, dict]], arm: str = "WOOF") -> list[dict]:
    out = []
    for label, receipt in receipts:
        for score in receipt.get("stations", []):
            for entry in score.get("arms", []):
                if entry.get("label") != arm:
                    continue
                out.append({
                    "valid": label,
                    "quantity": score["quantity"],
                    "units": score.get("units", ""),
                    "count": entry.get("count"),
                    "bias": entry.get("bias"),
                    "rmse": entry.get("rmse"),
                    "source": receipt.get("station_source", ""),
                })
    return out


def markdown(table: list[dict]) -> str:
    lines = ["| valid (UTC) | quantity | stations | bias (F-O) | RMSE |",
             "|---|---|---|---|---|"]
    for row in table:
        bias = "--" if row["bias"] is None else f"{row['bias']:+.2f} {row['units']}"
        rmse = "--" if row["rmse"] is None else f"{row['rmse']:.2f} {row['units']}"
        lines.append(f"| {row['valid']} | {row['quantity']} | {row['count']} | "
                     f"{bias} | {rmse} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--domain", default="d01")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    table = rows(load_receipts(args.run, args.domain))
    if not table:
        print(f"no verification receipts with station scores under {args.run}",
              file=sys.stderr)
        return 1
    print(json.dumps(table, indent=2) if args.json else markdown(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
