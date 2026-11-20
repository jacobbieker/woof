"""``woof var``: WRFDA 3D-Var (the experimental port in :mod:`woof.da.var`) to an analysis bundle.

The one door to :mod:`woof.da.var` (``tests/test_var_import_graph.py``):
WRFDA's ``be.dat``, OBSPROC's ``ob.ascii`` and a WRF first guess in, a
``wrf-analysis-bundle/1`` out, which ``woof run --analysis BUNDLE`` (and the
other forecast doors' ``--analysis``) start from.  It prints WRFDA's
minimiser log (``da_minimise_cg``'s lines) as the iterations run.

Scope is the WRFDA 4.7.1 regional 3D-Var the port gates on (cv5, the
conventional observation types, GPS PW/refractivity and SATEM); see
docs/var.md.  Clear-sky geostationary radiances with VarBC are opt-in
(``--radiance-scene``; CRTM from ``WOOF_CRTM_LIB``/``WOOF_CRTM_COEFFS``).
Hybrid 3DEnVar is opt-in: ``--ensemble-member`` (members' history files,
``gen_be_ep2``'s perturbations) or ``--ep-dir`` (WRFDA ``ep`` files) with
``--je-factor``.
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
    r = p.add_argument_group("radiances (opt-in; plan section 13)")
    r.add_argument("--radiance-scene", type=Path, action="append", default=[],
                   help="a cached imager scene directory (woof.obs.imager_scene: SatWeave, GOES ABI, AHI, SEVIRI); "
                        "repeat for several")
    r.add_argument("--radiance-bands", default="6200,6900,7330,11200",
                   help="imager bands in nm (default: the three WV bands and the 11.2 um window)")
    r.add_argument("--thinning-km", type=positive_float, default=60.0,
                   help="GSI thinning mesh (WRFDA thinning_mesh; default 60 km)")
    r.add_argument("--varbc-in", type=Path, default=None,
                   help="WRFDA VARBC.in; default: a cold start for every sensor")
    r.add_argument("--varbc-out", type=Path, default=None, help="write the updated coefficients (VARBC.out)")
    r.add_argument("--no-varbc", dest="varbc", action="store_false", help="radiances without bias correction")
    h = p.add_argument_group("hybrid 3DEnVar (opt-in; WRFDA's alpha control variable)")
    h.add_argument("--ensemble-member", type=Path, action="append", default=[],
                   help="a member's history file (wrfout) valid at the analysis time; repeat for each member "
                        "(perturbations as WRFDA's gen_be_ep2)")
    h.add_argument("--ep-dir", type=Path, default=None,
                   help="WRFDA ensemble perturbation files (u.e001 ... as gen_be_ep2 writes them) instead of members")
    h.add_argument("--ensdim-alpha", type=positive_int, default=None,
                   help="members to read from --ep-dir (WRFDA ensdim_alpha)")
    h.add_argument("--ep-format", type=int, choices=(1, 11), default=1, help="--ep-dir's ep_format (default 1)")
    h.add_argument("--je-factor", type=positive_float, default=None,
                   help="WRFDA je_factor (required with an ensemble): Jb is weighted je/(je-1); <= 1 is "
                        "full-ensemble mode (no static B)")
    h.add_argument("--alpha-corr-scale", type=positive_float, default=200.0,
                   help="horizontal localisation, km (WRFDA alpha_corr_scale; default 200)")
    h.add_argument("--alpha-vertloc-opt", type=int, choices=(0, 1, 2), default=2,
                   help="vertical localisation: 0 none, 1 --be-vertloc file, 2 log-p (default)")
    h.add_argument("--be-vertloc", type=Path, default=None, help="WRFDA be.vertloc.dat (--alpha-vertloc-opt 1)")
    h.add_argument("--max-vert-var-alpha", type=positive_float, default=99.0,
                   help="percent of the localisation's variance kept (WRFDA max_vert_var_alpha; default 99)")
    h.add_argument("--alpha-std-dev", type=positive_float, default=1.0, help="WRFDA alpha_std_dev (default 1)")
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

    rad_kw = _radiance_kwargs(args) if args.radiance_scene else {}
    try:
        hyb_kw = _hybrid_kwargs(args)
    except ValueError as exc:
        print(f"var: {exc}", file=sys.stderr)
        return 2
    rad_kw.update(hyb_kw)
    result = analyse(args.be, args.obs, args.first_guess, tier=args.tier, ntmax=args.ntmax, eps=args.eps,
                     max_ext_its=args.max_ext_its, orthonorm_gradient=args.orthonorm_gradient,
                     check_rh=args.check_rh, window_minutes=args.window_minutes,
                     landuse_table=args.landuse_table, write=write, **rad_kw)
    if rad_kw.get("varbc") is not None and args.varbc_out is not None:
        rad_kw["varbc"].write(args.varbc_out)
    if args.log is not None:
        Path(args.log).write_text("\n".join(line for r in result.cg for line in r.lines) + "\n")
    recipe = {"door": "woof var", "tier": args.tier, "ntmax": args.ntmax, "eps": args.eps,
              "max_ext_its": args.max_ext_its, "orthonorm_gradient": args.orthonorm_gradient,
              "check_rh": args.check_rh, "window_minutes": args.window_minutes,
              "be": Path(args.be).name, "obs": Path(args.obs).name}
    notes = {"var": {"nobs": result.nobs, "iterations": [r.iterations for r in result.cg],
                     "j_final": [r.j_final for r in result.cg], "timings_s": result.timings}}
    if hyb_kw:
        cfg = hyb_kw["hybrid"]
        recipe["hybrid"] = {"ensdim_alpha": hyb_kw["ensemble"].ne, "je_factor": cfg.je_factor,
                            "alpha_corr_scale": cfg.alpha_corr_scale, "alpha_vertloc_opt": cfg.alpha_vertloc_opt,
                            "max_vert_var_alpha": cfg.max_vert_var_alpha, "alpha_std_dev": cfg.alpha_std_dev,
                            "members": [Path(m).name for m in args.ensemble_member] or None,
                            "ep_dir": None if args.ep_dir is None else Path(args.ep_dir).name}
        notes["var"]["hybrid"] = recipe["hybrid"]
    if result.radiance is not None:
        recipe["radiance_scenes"] = [Path(d).name for d in args.radiance_scene]
        recipe["varbc"] = bool(args.varbc)
        notes["var"]["radiance"] = result.radiance
    bundle, _ = write_bundle(args.out, args.first_guess, result.wrf, recipe=recipe,
                             grid_id=args.grid_id or Path(args.first_guess).name, ph=args.ph,
                             iau_window_s=args.iau_window_s, notes=notes)
    print(f"var: {result.nobs} observations, {sum(r.iterations for r in result.cg)} iterations, "
          f"minimise {result.timings['minimise_s']:.1f} s; bundle {args.out}")
    print(json.dumps({"bundle": str(args.out), "valid_time": bundle.manifest.get("valid_time")}))
    return 0


def _hybrid_kwargs(args) -> dict:
    """``analyse``'s ``ensemble``/``hybrid``/``vertloc`` from the hybrid flags ({} without an ensemble)."""
    members, ep_dir = list(args.ensemble_member), args.ep_dir
    if not members and ep_dir is None:
        if args.je_factor is not None or args.be_vertloc is not None or args.ensdim_alpha is not None:
            raise ValueError("--je-factor, --be-vertloc and --ensdim-alpha need --ensemble-member or --ep-dir")
        return {}
    if members and ep_dir is not None:
        raise ValueError("give --ensemble-member or --ep-dir, not both")
    if args.je_factor is None:
        raise ValueError("an ensemble needs --je-factor (WRFDA's default 1.0 is full-ensemble mode; "
                         "2 weights the static and ensemble B equally)")
    from woof.da.var import hybrid as hy

    if members:
        for m in members:
            if not Path(m).is_file():
                raise ValueError(f"--ensemble-member {m} is not a file")
        perts = hy.ep_from_wrf(members)
    else:
        if args.ensdim_alpha is None:
            raise ValueError("--ep-dir needs --ensdim-alpha")
        perts = hy.read_ep(ep_dir, args.ensdim_alpha, ep_format=args.ep_format)
    vertloc = None
    if args.alpha_vertloc_opt == 1:
        if args.be_vertloc is None:
            raise ValueError("--alpha-vertloc-opt 1 needs --be-vertloc")
        vertloc = hy.read_vertloc(args.be_vertloc)
    elif args.be_vertloc is not None:
        raise ValueError("--be-vertloc is read only with --alpha-vertloc-opt 1")
    cfg = hy.HybridConfig(je_factor=args.je_factor, alpha_corr_scale=args.alpha_corr_scale,
                          alpha_std_dev=args.alpha_std_dev, alpha_vertloc_opt=args.alpha_vertloc_opt,
                          max_vert_var_alpha=args.max_vert_var_alpha)
    cfg.check()
    return {"ensemble": hy.Ensemble.from_perturbations(perts), "hybrid": cfg, "vertloc": vertloc}


def _radiance_kwargs(args) -> dict:
    from woof.da.var import radiance as rad
    from woof.da.var import varbc as vbc
    from woof.obs.imager_scene import read_scene_netcdf

    bands = tuple(int(b) for b in str(args.radiance_bands).split(","))
    records = []
    for d in args.radiance_scene:
        records += rad.from_imager_scene(read_scene_netcdf(d), bands)
    kw = {"radiances": records, "radiance_qc": rad.RadianceQC(thinning_mesh_km=args.thinning_km)}
    if args.varbc:
        sensors = sorted({r.sensor for r in records})
        if args.varbc_in is not None:
            vb = vbc.VarBC.from_file(args.varbc_in, sensors)
        else:
            vb = vbc.VarBC({})
        for r in records:  # a sensor VARBC.in lacks starts cold, with the channels it has
            if r.sensor not in vb.instruments and r.sensor in vbc.VARBC_IDS:
                vb.instruments[r.sensor] = vbc.template(r.sensor, list(r.channels))
        kw["varbc"] = vb
    return kw
