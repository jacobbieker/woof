"""Generate a core's namelist-options inventory from its MPAS Registry.xml.

The inventory is the "every option" surface a port must support: each
``nml_option`` the core's Registry declares, with its type, default, units and
description. It is pinned to the Registry.xml it was parsed from (SHA-256) and
to the oracle commit, so a changed Registry is a changed receipt, not a silent
drift. Run it against the pinned MPAS-Model reference checkout.

Usage:
    python tools/mpas_coupled_options.py \
        --source /path/to/MPAS-Model \
        --core core_seaice \
        --pin f34984b2c89353cf31c235b08d3b4acd96b37bb3 \
        --out woof/hex/mpas_seaice/data/registry-options.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as etree
from pathlib import Path


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_options(registry: Path) -> list[dict[str, str]]:
    tree = etree.parse(registry)
    root = tree.getroot()
    options: list[dict[str, str]] = []
    for node in root.iter("nml_option"):
        options.append(
            {
                "name": node.get("name", ""),
                "type": node.get("type", ""),
                "default": node.get("default_value", ""),
                "units": node.get("units", ""),
                "description": (node.get("description") or "").strip(),
            }
        )
    return options


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="the MPAS-Model reference checkout")
    parser.add_argument("--core", required=True, help="the core directory, e.g. core_seaice")
    parser.add_argument("--pin", required=True, help="the oracle commit (the v8.4.2 SHA)")
    parser.add_argument("--out", type=Path, required=True, help="where to write the inventory JSON")
    args = parser.parse_args(argv)

    registry = args.source / "src" / args.core / "Registry.xml"
    if not registry.is_file():
        print(f"mpas_coupled_options: no Registry.xml at {registry}", file=sys.stderr)
        return 2

    options = parse_options(registry)
    inventory = {
        "schema": "gpuwm-hex.mpas-core-options/v1",
        "core": args.core,
        "pin": args.pin,
        "registry": {
            "path": f"src/{args.core}/Registry.xml",
            "sha256": sha256_of(registry),
        },
        "option_count": len(options),
        "options": options,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(
        f"mpas_coupled_options: {args.core}: {len(options)} options "
        f"from {registry} (sha256 {inventory['registry']['sha256'][:12]}...)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
