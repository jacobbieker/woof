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
``--je-factor``.  FGAT is opt-in: one ``--fgat-obs`` and one
``--fgat-first-guess`` per time slot (WRFDA's ``obNN.ascii`` and ``fgNN``).
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
    p.add_argument("--obs", type=Path, default=None,
                   help="OBSPROC observations (ob.ascii); with FGAT give --fgat-obs per slot instead")
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
                   help="observation window half-width around the first guess's time (default 60; WRFDA's "
                        "time_window_min/max)")
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
    h.add_argument("--ep-format", type=int, choices=(1, 11), default=None, help="--ep-dir's ep_format (default 1)")
    h.add_argument("--je-factor", type=positive_float, default=None,
                   help="WRFDA je_factor (required with an ensemble): Jb is weighted je/(je-1); <= 1 is "
                        "full-ensemble mode (no static B)")
    h.add_argument("--alpha-corr-scale", type=positive_float, default=None,
                   help="horizontal localisation, km (WRFDA alpha_corr_scale; default 200)")
    h.add_argument("--alpha-vertloc-opt", type=int, choices=(0, 1, 2), default=None,
                   help="vertical localisation: 0 none, 1 --be-vertloc file, 2 log-p (default)")
    h.add_argument("--be-vertloc", type=Path, default=None, help="WRFDA be.vertloc.dat (--alpha-vertloc-opt 1)")
    h.add_argument("--max-vert-var-alpha", type=positive_float, default=None,
                   help="percent of the localisation's variance kept (WRFDA max_vert_var_alpha; default 99)")
    h.add_argument("--alpha-std-dev", type=positive_float, default=None, help="WRFDA alpha_std_dev (default 1)")
    f = p.add_argument_group("FGAT (opt-in; WRFDA num_fgat_time > 1)")
    f.add_argument("--fgat-obs", type=Path, action="append", default=[],
                   help="one time slot's OBSPROC file (ob01.ascii, ob02.ascii, ...), in slot order; repeat for "
                        "each slot (instead of --obs)")
    f.add_argument("--fgat-first-guess", type=Path, action="append", default=[],
                   help="one time slot's first guess (fg01, fg02, ...), in slot order, one per --fgat-obs: the "
                        "window's start, then evenly spaced, the last at its end.  --first-guess stays the "
                        "analysis-time background")
    p.set_defaults(func=main)
    return p


def main(args) -> int:
    from woof.da.var.analysis import analyse, write_bundle

    for flag, path in (("--be", args.be), ("--obs", args.obs), ("--first-guess", args.first_guess)):
        if path is not None and not Path(path).is_file():
            print(f"var: {flag} {path} is not a file", file=sys.stderr)
            return 2
    if Path(args.out).exists():
        print(f"var: --out {args.out} exists; name a new bundle directory", file=sys.stderr)
        return 2

    def write(line: str) -> None:
        print(line, flush=True)

    rad_kw = _radiance_kwargs(args) if args.radiance_scene else {}
    try:
        fgat_kw, fgat_recipe = _fgat_kwargs(args)
        if fgat_kw and (args.ensemble_member or args.ep_dir):
            raise ValueError("FGAT with the hybrid (--ensemble-member/--ep-dir) is not ported: WRFDA bins "
                             "the ensemble by time slot only in 4DEnVar (use_4denvar)")
        hyb_kw, hyb_recipe = _hybrid_kwargs(args)
    except ValueError as exc:
        print(f"var: {exc}", file=sys.stderr)
        return 2
    rad_kw.update(hyb_kw)
    rad_kw.update(fgat_kw)
    obs = args.fgat_obs if fgat_kw else args.obs
    result = analyse(args.be, obs, args.first_guess, tier=args.tier, ntmax=args.ntmax, eps=args.eps,
                     max_ext_its=args.max_ext_its, orthonorm_gradient=args.orthonorm_gradient,
                     check_rh=args.check_rh, window_minutes=args.window_minutes,
                     landuse_table=args.landuse_table, write=write, **rad_kw)
    if rad_kw.get("varbc") is not None and args.varbc_out is not None:
        rad_kw["varbc"].write(args.varbc_out)
    if args.log is not None:
        Path(args.log).write_text("\n".join(line for r in result.cg for line in r.lines) + "\n")
    recipe = _recipe(args, hyb_recipe)
    notes = {"var": {"nobs": result.nobs, "iterations": [r.iterations for r in result.cg],
                     "j_final": [r.j_final for r in result.cg], "timings_s": result.timings}}
    if hyb_recipe:
        notes["var"]["hybrid"] = hyb_recipe
    if fgat_recipe:
        recipe["fgat"] = fgat_recipe
        notes["var"]["fgat"] = fgat_recipe
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


def _recipe(args, hybrid: dict | None) -> dict:
    """The door's settings digested into the bundle's ``recipe_sha256`` (radiances are added by ``main``)."""
    recipe = {"door": "woof var", "tier": args.tier, "ntmax": args.ntmax, "eps": args.eps,
              "max_ext_its": args.max_ext_its, "orthonorm_gradient": args.orthonorm_gradient,
              "check_rh": args.check_rh, "window_minutes": args.window_minutes,
              "be": Path(args.be).name, "obs": Path(args.obs).name if args.obs is not None else None}
    if hybrid:
        recipe["hybrid"] = hybrid
    return recipe


def _sha256(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fgat_kwargs(args) -> tuple[dict, dict | None]:
    """``analyse``'s ``fgat`` from --fgat-obs/--fgat-first-guess, and the recipe entry
    (each slot's file names and sha256).  ``({}, None)`` for 3D-Var; every
    refusal is a ValueError the door prints as ``var: ...``."""
    import types

    obs, fgs = list(args.fgat_obs), list(args.fgat_first_guess)
    if not obs and not fgs:
        if args.obs is None:
            raise ValueError("give --obs (3D-Var) or one --fgat-obs per time slot (FGAT)")
        return {}, None
    if args.obs is not None:
        raise ValueError("give --obs or --fgat-obs, not both")
    if not obs or not fgs:
        raise ValueError("FGAT needs --fgat-obs and --fgat-first-guess for every time slot")
    for flag, paths in (("--fgat-obs", obs), ("--fgat-first-guess", fgs)):
        for path in paths:
            if not Path(path).is_file():
                raise ValueError(f"{flag} {path} is not a file")
    if args.radiance_scene:
        raise ValueError("FGAT with --radiance-scene is not ported (WRFDA bins radiances by time slot)")
    from woof.da.var import fgat as fm
    from woof.da.var.analysis import obs_window

    window = obs_window(types.SimpleNamespace(times=fm.first_guess_time(args.first_guess)), args.window_minutes)
    try:
        fm.check_fgat(len(obs), [fm.first_guess_time(f) for f in fgs], time_window_min=window["time_window_min"],
                      time_window_max=window["time_window_max"], max_ext_its=args.max_ext_its,
                      analysis_date=window["analysis_date"])
    except NotImplementedError as exc:
        raise ValueError(str(exc)) from None
    recipe = {"obs": [{"name": Path(p).name, "sha256": _sha256(p)} for p in obs],
              "first_guesses": [{"name": Path(p).name, "sha256": _sha256(p)} for p in fgs]}
    return {"fgat": fgs}, recipe


#: Hybrid flags that mean nothing without an ensemble, and the ones that belong to --ep-dir.
_ALPHA_FLAGS = (("--je-factor", "je_factor"), ("--be-vertloc", "be_vertloc"), ("--ensdim-alpha", "ensdim_alpha"),
                ("--ep-format", "ep_format"), ("--alpha-corr-scale", "alpha_corr_scale"),
                ("--alpha-vertloc-opt", "alpha_vertloc_opt"), ("--max-vert-var-alpha", "max_vert_var_alpha"),
                ("--alpha-std-dev", "alpha_std_dev"))
_EP_DIR_FLAGS = (("--ensdim-alpha", "ensdim_alpha"), ("--ep-format", "ep_format"))


def _hybrid_kwargs(args) -> tuple[dict, dict | None]:
    """``analyse``'s ``ensemble``/``hybrid``/``vertloc`` from the hybrid flags, and their recipe entry.

    ``({}, None)`` without an ensemble.  Every refusal is a ValueError the
    door prints as ``var: ...``.  The recipe entry digests what the
    analysis uses: the perturbations' words (so a changed member or ep file
    changes it) and ``be.vertloc.dat``'s bytes.
    """
    import hashlib

    members, ep_dir = list(args.ensemble_member), args.ep_dir
    if not members and ep_dir is None:
        given = [flag for flag, key in _ALPHA_FLAGS if getattr(args, key) is not None]
        if given:
            raise ValueError(f"{', '.join(given)} need an ensemble (--ensemble-member or --ep-dir)")
        return {}, None
    if members and ep_dir is not None:
        raise ValueError("give --ensemble-member or --ep-dir, not both")
    if members:
        given = [flag for flag, key in _EP_DIR_FLAGS if getattr(args, key) is not None]
        if given:
            raise ValueError(f"{', '.join(given)} belong to --ep-dir; with --ensemble-member every member is used")
    if args.je_factor is None:
        raise ValueError("an ensemble needs --je-factor (WRFDA's default 1.0 is full-ensemble mode; "
                         "2 weights the static and ensemble B equally)")
    from woof.da.var import hybrid as hy

    fg_times = hy.wrf_times(args.first_guess)
    fg_shape = _mass_shape(args.first_guess)
    recipe = {}
    if members:
        indices = []
        for m in members:
            if not Path(m).is_file():
                raise ValueError(f"--ensemble-member {m} is not a file")
            times = hy.wrf_times(m)
            if fg_times[0] not in times:
                raise ValueError(f"--ensemble-member {m} has no record at the first guess's time {fg_times[0]} "
                                 f"(its times: {', '.join(times)})")
            indices.append(times.index(fg_times[0]))
        perts = hy.ep_from_wrf(members, time_index=indices)
        recipe["members"] = [Path(m).name for m in members]
        recipe["member_time"] = fg_times[0]
    else:
        if args.ensdim_alpha is None:
            raise ValueError("--ep-dir needs --ensdim-alpha")
        ep_format = 1 if args.ep_format is None else args.ep_format
        hy.ep_paths(ep_dir, args.ensdim_alpha, ep_format)
        perts = hy.read_ep(ep_dir, args.ensdim_alpha, ep_format=ep_format, shape=fg_shape)
        recipe["ep_dir"] = Path(ep_dir).name
        recipe["ep_format"] = ep_format
    if perts["t"].shape[1:] != fg_shape:
        raise ValueError(f"the ensemble is on {perts['t'].shape[1:]}, the first guess on {fg_shape}")
    cfg_kw = {k: getattr(args, k) for k in ("alpha_corr_scale", "alpha_std_dev", "alpha_vertloc_opt",
                                             "max_vert_var_alpha") if getattr(args, k) is not None}
    cfg = hy.HybridConfig(je_factor=args.je_factor, **cfg_kw)
    cfg.check()
    vertloc = None
    if cfg.alpha_vertloc_opt == 1:
        if args.be_vertloc is None:
            raise ValueError("--alpha-vertloc-opt 1 needs --be-vertloc")
        if not Path(args.be_vertloc).is_file():
            raise ValueError(f"--be-vertloc {args.be_vertloc} is not a file")
        vertloc = hy.read_vertloc(args.be_vertloc)
        recipe["be_vertloc_sha256"] = hashlib.sha256(Path(args.be_vertloc).read_bytes()).hexdigest()
    elif args.be_vertloc is not None:
        raise ValueError("--be-vertloc is read only with --alpha-vertloc-opt 1")
    ensemble = hy.Ensemble.from_perturbations(perts)
    recipe.update(ensdim_alpha=ensemble.ne, perturbations_sha256=hy.perturbations_sha256(perts),
                  je_factor=cfg.je_factor, alpha_corr_scale=cfg.alpha_corr_scale,
                  alpha_vertloc_opt=cfg.alpha_vertloc_opt, max_vert_var_alpha=cfg.max_vert_var_alpha,
                  alpha_std_dev=cfg.alpha_std_dev)
    return {"ensemble": ensemble, "hybrid": cfg, "vertloc": vertloc}, recipe


def _mass_shape(path) -> tuple:
    """A WRF file's mass-point ``(west_east, south_north, bottom_top)``."""
    import netCDF4

    with netCDF4.Dataset(Path(path)) as nc:
        return tuple(len(nc.dimensions[d]) for d in ("west_east", "south_north", "bottom_top"))


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
