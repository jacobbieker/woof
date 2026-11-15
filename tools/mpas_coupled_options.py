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

``--fields-out`` also writes the Registry's field inventory: every dimension
(name, definition), package and ``var_struct`` (name, time levels, packages)
with its ``var`` rows (name, type, dimensions, packages), which the column
package's oracle stream is built from:

    python tools/mpas_coupled_options.py --source ... --core core_seaice \
        --pin f34984b2c89353cf31c235b08d3b4acd96b37bb3 \
        --out woof/hex/mpas_seaice/data/registry-options.json \
        --fields-out woof/hex/mpas_seaice/data/registry-fields.json
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


def parse_fields(registry: Path) -> dict:
    """The Registry's dimensions, packages and var_structs with their vars, in file order."""
    root = etree.parse(registry).getroot()
    dims = [{"name": d.get("name", ""), "definition": d.get("definition", "")} for d in root.iter("dim")]
    packages = [p.get("name", "") for p in root.iter("package")]
    structs = []
    for struct in root.iter("var_struct"):
        # a var is [name, type, dimensions, packages] (space- and semicolon-separated as in the Registry)
        variables = [[v.get("name", ""), v.get("type", ""), " ".join((v.get("dimensions") or "").split()),
                      v.get("packages") or ""] for v in struct.iter("var")]
        structs.append({"name": struct.get("name", ""), "time_levs": int(struct.get("time_levs", "1")),
                        "packages": [p for p in (struct.get("packages") or "").split(";") if p],
                        "vars": variables})
    return {"dims": dims, "packages": packages, "var_structs": structs}


def _one_var_per_line(text: str) -> str:
    """json.dumps(indent=1) with each innermost list of strings folded onto its own line."""
    import re
    return re.sub(r"\[\n\s*(\"[^\[\]{}]*?\")\n\s*\]",
                  lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",\n")) + "]", text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="the MPAS-Model reference checkout")
    parser.add_argument("--core", required=True, help="the core directory, e.g. core_seaice")
    parser.add_argument("--pin", required=True, help="the oracle commit (the v8.4.2 SHA)")
    parser.add_argument("--out", type=Path, required=True, help="where to write the inventory JSON")
    parser.add_argument("--fields-out", type=Path, help="where to write the field inventory JSON (optional)")
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
    if args.fields_out is not None:
        fields = {"schema": "gpuwm-hex.mpas-core-fields/v1", "core": args.core, "pin": args.pin,
                  "registry": inventory["registry"], **parse_fields(registry)}
        args.fields_out.parent.mkdir(parents=True, exist_ok=True)
        # one var per line
        text = json.dumps(fields, indent=1)
        args.fields_out.write_text(_one_var_per_line(text) + "\n", encoding="utf-8")
    print(
        f"mpas_coupled_options: {args.core}: {len(options)} options "
        f"from {registry} (sha256 {inventory['registry']['sha256'][:12]}...)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
