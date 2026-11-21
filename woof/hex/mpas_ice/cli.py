"""The ``woof hex ice`` door: the MALI (MPAS land-ice) port, and ``run``, its forward mode on a mesh."""

from __future__ import annotations

from woof.hex.coupled import registry
from woof.hex.mpas_model_door import add_model_parser

_ROW = {
    "word": "ice",
    "name": "landice",
    "module": "woof.hex.mpas_ice",
    "attr": "registry-options.json",
    "verify": "woof.hex.mpas_ice.verify:verify",
}


def add_ice_parser(commands) -> None:
    info = registry.component(_ROW["name"])
    sub = add_model_parser(
        commands,
        word="ice",
        help_text=f"the MPAS-ICE (land-ice) port (core {info.core_dir}; status: {info.status})",
        row=_ROW,
    )
    run = sub.add_parser("run", help="MALI v8.4.2's forward mode on an MPAS land-ice mesh "
                                     "(woof.hex.mpas_ice.run_v842)")
    run.add_argument("--namelist", required=True, help="the MALI namelist (namelist.landice)")
    run.add_argument("--input", default=None, help="the MPAS land-ice input file (a cold start)")
    run.add_argument("--restart", default=None, help="a restart.npz a run wrote (continue it)")
    run.add_argument("--out", required=True, help="the folder for output.nc and restart.npz")
    run.add_argument("--precision", choices=("r4", "r8"), default=None,
                     help="MALI's RKIND: r4 (single, MALI's default build; a cold start's default) or r8; a "
                          "restart runs at its stored precision")
    run.add_argument("--steps", type=int, default=None, help="stop after this many steps (default: the stop time)")
    run.set_defaults(handler=_run)


def _run(args) -> int:
    from woof.hex.mpas_ice import run_v842
    if (args.input is None) == (args.restart is None):
        raise SystemExit("woof hex ice run: give exactly one of --input (a cold start) or --restart")
    return run_v842.main(args)
