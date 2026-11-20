"""``woof var``: WRFDA 3D-Var (the experimental port in :mod:`woof.da.var`) to an analysis bundle.

The one door to :mod:`woof.da.var` (``tests/test_var_import_graph.py``):
WRFDA's ``be.dat``, OBSPROC's ``ob.ascii`` and a WRF first guess in, a
``wrf-analysis-bundle/1`` out, which ``woof run --analysis BUNDLE`` (and the
other forecast doors' ``--analysis``) start from.  It prints WRFDA's
minimiser log (``da_minimise_cg``'s lines) as the iterations run.

Scope is the WRFDA 4.7.1 regional 3D-Var the port gates on (cv5, the
conventional observation types, GPS PW/refractivity and SATEM); see
docs/var.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from woof.cli_numbers import positive_float, positive_int

TIERS = ("reference", "oracle", "production")


def register_cli(subparsers):
    p = subparsers.add_parser(
        "var",
        help="3D-Var analysis (experimental WRFDA 4.7.1 port): be.dat + ob.ascii + first guess -> analysis bundle",
        description=(
            "Run WRFDA 4.7.1's regional 3D-Var (cv5) on a WRF first guess with OBSPROC "
            "observations and WRFDA's be.dat, print WRFDA's minimiser log, and write the "
            "analysis as a wrf-analysis-bundle/1 for `woof run --analysis BUNDLE`.  Experimental."))
    p.add_argument("--be", required=True, type=Path, help="WRFDA background-error file (be.dat, cv_options = 5)")
    p.add_argument("--obs", required=True, type=Path, help="OBSPROC observations (ob.ascii)")
    p.add_argument("--first-guess", required=True, type=Path,
                   help="WRF first guess (wrfinput_d01 or a wrfout), WRFDA's fg")
    p.add_argument("--out", required=True, type=Path, help="analysis bundle directory to write (must not exist)")
    p.add_argument("--tier", choices=TIERS, default="reference",
                   help="reference: NumPy binary64 (default); oracle: GPU binary64 with WRFDA's serial dot "
                        "product; production: GPU binary32")
    p.add_argument("--ntmax", type=positive_int, default=50, help="inner iterations per outer loop (WRFDA ntmax; default 50)")
    p.add_argument("--eps", type=positive_float, default=0.01,
                   help="gradient reduction target (WRFDA eps; default 0.01)")
    p.add_argument("--max-ext-its", type=positive_int, default=1, help="outer loops (WRFDA max_ext_its; default 1)")
    p.add_argument("--no-orthonorm-gradient", dest="orthonorm_gradient", action="store_false",
                   help="turn off WRFDA's orthonorm_gradient (on by default, as in the tutorial)")
    p.add_argument("--check-rh", type=int, choices=(0, 1), default=0,
                   help="WRFDA check_rh in the analysis transfer (0 or 1; default 0)")
    p.add_argument("--window-minutes", type=positive_float, default=60.0,
                   help="observation window half-width around the first guess's time (default 60)")
    p.add_argument("--landuse-table", type=Path,
                   help="LANDUSE.TBL for the roughness (default: woof's packaged WRF table)")
    p.add_argument("--grid-id", default=None,
                   help="the bundle's grid name (default: the first guess's file name)")
    p.add_argument("--ph", choices=("analysis", "balance"), default="analysis",
                   help="bundle PH: WRFDA's hydrostatic PH (analysis, default) or rebuilt by the bundle (balance)")
    p.add_argument("--iau-window-s", type=positive_float, default=None,
                   help="record an IAU window in the bundle (seconds); default: insert at once")
    p.add_argument("--log", type=Path, default=None, help="also write the minimiser log to this file")
    p.set_defaults(func=main)
    return p


def main(args) -> int:
    from woof.da.var.analysis import analyse, write_bundle

    for flag, path in (("--be", args.be), ("--obs", args.obs), ("--first-guess", args.first_guess)):
        if not Path(path).is_file():
            print(f"var: {flag} {path} is not a file", file=sys.stderr)
            return 2
    if Path(args.out).exists():
        print(f"var: --out {args.out} exists; name a new bundle directory", file=sys.stderr)
        return 2

    def write(line: str) -> None:
        print(line, flush=True)

    result = analyse(args.be, args.obs, args.first_guess, tier=args.tier, ntmax=args.ntmax, eps=args.eps,
                     max_ext_its=args.max_ext_its, orthonorm_gradient=args.orthonorm_gradient,
                     check_rh=args.check_rh, window_minutes=args.window_minutes,
                     landuse_table=args.landuse_table, write=write)
    if args.log is not None:
        Path(args.log).write_text("\n".join(line for r in result.cg for line in r.lines) + "\n")
    recipe = {"door": "woof var", "tier": args.tier, "ntmax": args.ntmax, "eps": args.eps,
              "max_ext_its": args.max_ext_its, "orthonorm_gradient": args.orthonorm_gradient,
              "check_rh": args.check_rh, "window_minutes": args.window_minutes,
              "be": Path(args.be).name, "obs": Path(args.obs).name}
    notes = {"var": {"nobs": result.nobs, "iterations": [r.iterations for r in result.cg],
                     "j_final": [r.j_final for r in result.cg], "timings_s": result.timings}}
    bundle, _ = write_bundle(args.out, args.first_guess, result.wrf, recipe=recipe,
                             grid_id=args.grid_id or Path(args.first_guess).name, ph=args.ph,
                             iau_window_s=args.iau_window_s, notes=notes)
    print(f"var: {result.nobs} observations, {sum(r.iterations for r in result.cg)} iterations, "
          f"minimise {result.timings['minimise_s']:.1f} s; bundle {args.out}")
    print(json.dumps({"bundle": str(args.out), "valid_time": bundle.manifest.get("valid_time")}))
    return 0
