"""The shared door for an MPAS component model (ocean, sea-ice, land-ice).

The three lanes share one door shape: report the port status and the oracle it
is pinned to, list the options the core's Registry declares (the "every option"
surface the port must support), and expose a verify gate that refuses by name
until the core is ported. A lane supplies its row -- its word, its registry
name, and the package that carries its options inventory -- and the door is the
same either way.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from .coupled import oracle, registry


def _load_options(row: dict) -> dict:
    module = importlib.import_module(row["module"])
    path = Path(module.__file__).resolve().parent / "data" / row["attr"]
    if not path.is_file():
        raise ValueError(
            f"the {row['word']} options inventory is missing ({path}); regenerate it with "
            "tools/mpas_coupled_options.py against the pinned oracle"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _status(row: dict):
    def handler(_: argparse.Namespace) -> int:
        info = registry.component(row["name"])
        inventory = _load_options(row)
        print(
            json.dumps(
                {
                    "model": row["word"],
                    "core": info.core_dir,
                    "status": info.status,
                    "branch": info.branch,
                    "oracle": {
                        "repo": oracle.ORACLE_REPO,
                        "tag": oracle.ORACLE_TAG,
                        "pin": oracle.ORACLE_PIN,
                    },
                    "strict_recipe": oracle.STRICT_RECIPE,
                    "options": inventory["option_count"],
                    "registry_sha256": inventory["registry"]["sha256"],
                },
                indent=2,
            )
        )
        return 0

    return handler


def _options(row: dict):
    def handler(args: argparse.Namespace) -> int:
        inventory = _load_options(row)
        options = inventory["options"]
        if args.match:
            options = [o for o in options if args.match in o["name"]]
        if args.json:
            print(json.dumps(options, indent=2))
        else:
            for option in options:
                print(f"{option['name']}  ({option['type']}, default {option['default']})")
        return 0

    return handler


def _verify(row: dict):
    def handler(_: argparse.Namespace) -> int:
        target = row.get("verify")
        if target is None:
            registry.require_ported(row["name"])  # refuses a stub by name
            return 0
        # the ported pieces, each against its oracle; the core stays a stub
        # until every piece is ported
        module_name, _, attr = target.partition(":")
        report = getattr(importlib.import_module(module_name), attr)()
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1

    return handler


def add_model_parser(
    commands: argparse._SubParsersAction,
    *,
    word: str,
    help_text: str,
    row: dict,
) -> None:
    parser = commands.add_parser(word, help=help_text)
    sub = parser.add_subparsers(dest="model_command", required=True)

    status = sub.add_parser("status", help="report the port status and the pinned oracle")
    status.set_defaults(handler=_status(row))

    options = sub.add_parser("options", help="list the options the core's Registry declares")
    options.add_argument("--match", default=None, help="only options whose name contains this substring")
    options.add_argument("--json", action="store_true", help="emit the options as JSON")
    options.set_defaults(handler=_options(row))

    verify = sub.add_parser(
        "verify",
        help="the oracle gate: grade each ported piece word for word (refused by name while none is ported)",
    )
    verify.set_defaults(handler=_verify(row))
