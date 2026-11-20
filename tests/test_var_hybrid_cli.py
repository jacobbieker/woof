"""Hybrid 3DEnVar end to end: ``woof var``'s flags, ``analyse`` and gen_be_ep2's perturbations.

Everywhere, on tiny synthetic WRF files: every refused flag combination
(each a ``var: ...`` exit 2, never a traceback), the members' record chosen
by the first guess's time, the recipe digest following the members'
contents, and :func:`~woof.da.var.hybrid.ep_from_wrf` against hand-computed
two-member perturbations (the destaggering, ``t`` from theta and pressure,
``q`` from the mixing ratio, the mean).

On the WRFDA 4.7.1 oracle (``WOOF_WRFDA471_DIR`` as for test_var_analysis,
``WOOF_WRFDA471_HYBRID_DIR`` as for test_var_hybrid_o6, which also holds
``wrfvar_output``; ``WOOF_GENBE_DIR`` for the members):

- ``ep_from_wrf`` on the ten member forecasts the manifest lists is
  gen_be_ep2's ``ep`` files: ``u``, ``v``, ``q`` and ``ps`` word for word,
  ``t`` within 1e-12 K (measured 2.8e-13; ``**kappa`` is a pow ulp away).
- **O7 for the hybrid**: :func:`analyse` with WRFDA's ``ep`` files and
  ``be.vertloc.dat`` (reference tier, the oracle's namelist) takes WRFDA's 50
  iterations, and per WRF field its increment against WRFDA's
  ``wrfvar_output`` minus ``fg`` meets O7's tolerances (max |difference|
  under 1% of the RMS, correlation >= 0.9999).
- ``woof var --ep-dir ... --je-factor 2`` (three iterations) prints
  WRFDA's hybrid ``minimize_cg`` lines to 1e-12 and writes a bundle.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from woof.cli import build_parser, main

MANIFEST = Path(__file__).parent / "data" / "wrfda471" / "manifest.json"
FIELDS = ("U", "V", "T", "QVAPOR", "MU", "P", "PH", "PSFC")


FG_TIME = "2008-02-05_12:00:00"
NX, NY, NZ = 4, 3, 2


def _wrf(path: Path, records: list[dict], times: list[str], nx=NX, ny=NY, nz=NZ) -> Path:
    """A WRF-like history file with gen_be's fields; each record's missing fields are filled with zeros."""
    import netCDF4

    shapes = {"U": (nz, ny, nx + 1), "V": (nz, ny + 1, nx), "T": (nz, ny, nx), "P": (nz, ny, nx),
              "PB": (nz, ny, nx), "QVAPOR": (nz, ny, nx), "PH": (nz + 1, ny, nx), "PHB": (nz + 1, ny, nx),
              "PSFC": (ny, nx), "XLAT": (ny, nx), "MAPFAC_M": (ny, nx), "MAPFAC_U": (ny, nx + 1),
              "MAPFAC_V": (ny + 1, nx)}
    dims = {"U": ("bottom_top", "south_north", "west_east_stag"), "V": ("bottom_top", "south_north_stag", "west_east"),
            "PH": ("bottom_top_stag", "south_north", "west_east"), "PHB": ("bottom_top_stag", "south_north", "west_east"),
            "MAPFAC_U": ("south_north", "west_east_stag"), "MAPFAC_V": ("south_north_stag", "west_east")}
    with netCDF4.Dataset(path, "w") as nc:
        for name, n in (("Time", None), ("DateStrLen", 19), ("west_east", nx), ("south_north", ny),
                        ("bottom_top", nz), ("west_east_stag", nx + 1), ("south_north_stag", ny + 1),
                        ("bottom_top_stag", nz + 1)):
            nc.createDimension(name, n)
        nc.DX = np.float32(30000.0)
        nc.START_DATE = times[0]
        t = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        for i, stamp in enumerate(times):
            t[i] = np.array(list(stamp), dtype="S1")
        for name, shape in shapes.items():
            d = dims.get(name, ("bottom_top", "south_north", "west_east") if len(shape) == 3
                         else ("south_north", "west_east"))
            v = nc.createVariable(name, "f4", ("Time", *d))
            for i, rec in enumerate(records):
                # Fortran (i, j, k) in the record, the file's (k, j, i) on disk.
                v[i] = np.asarray(rec.get(name, np.zeros(shape[::-1])), dtype=np.float32).T
    return path


def _member_fields(u, t, q, ps, nx=NX, ny=NY, nz=NZ):
    """Fortran-order fields: U ramps in i (so destaggering is checkable), the rest constant."""
    big_u = np.empty((nx + 1, ny, nz))
    big_u[:] = (np.arange(nx + 1, dtype=float) + u)[:, None, None]
    return {"U": big_u, "V": np.full((nx, ny + 1, nz), 2.0 * u), "T": np.full((nx, ny, nz), t),
            "PB": np.full((nx, ny, nz), 1.0e5), "P": np.zeros((nx, ny, nz)), "QVAPOR": np.full((nx, ny, nz), q),
            "PSFC": np.full((nx, ny), ps)}


@pytest.fixture()
def case(tmp_path):
    """A first guess, two members (one with an extra earlier record) and an ep directory, all tiny."""
    fg = _wrf(tmp_path / "fg", [{}], [FG_TIME])
    m1 = _wrf(tmp_path / "m1", [_member_fields(0.0, 2.0, 1.0, 90000.0)], [FG_TIME])
    m2 = _wrf(tmp_path / "m2", [_member_fields(5.0, 9.0, 0.0, 0.0), _member_fields(2.0, 4.0, 3.0, 90100.0)],
              ["2008-02-05_06:00:00", FG_TIME])
    from woof.da.var import hybrid as hy

    perts = {"u": np.zeros((3, NX, NY, NZ)), "v": np.zeros((3, NX, NY, NZ)), "t": np.zeros((3, NX, NY, NZ)),
             "q": np.zeros((3, NX, NY, NZ)), "ps": np.zeros((3, NX, NY))}
    for k in perts:
        perts[k][0] += 1.0
        perts[k][1] -= 1.0
    hy.write_ep(tmp_path / "ep", perts)
    return tmp_path


def _argv(tmp_path, *extra):
    return ["var", "--be", str(tmp_path / "be.dat"), "--obs", str(tmp_path / "ob.ascii"), "--first-guess",
            str(tmp_path / "fg"), "--out", str(tmp_path / "b"), *extra]


def _kwargs(tmp_path, *extra):
    from woof.var_cli import _hybrid_kwargs

    return _hybrid_kwargs(build_parser().parse_args(_argv(tmp_path, *extra)))


def test_hybrid_flags_default_to_off(tmp_path):
    args = build_parser().parse_args(_argv(tmp_path))
    assert (args.ensemble_member, args.ep_dir, args.je_factor, args.alpha_corr_scale, args.alpha_vertloc_opt,
            args.max_vert_var_alpha, args.alpha_std_dev, args.ep_format, args.ensdim_alpha,
            args.be_vertloc) == ([], None, None, None, None, None, None, None, None, None)
    assert _kwargs(tmp_path) == ({}, None)


@pytest.mark.parametrize("flag, value", [
    ("--je-factor", "2"), ("--be-vertloc", "vl.dat"), ("--ensdim-alpha", "3"), ("--ep-format", "11"),
    ("--alpha-corr-scale", "150"), ("--alpha-vertloc-opt", "0"), ("--max-vert-var-alpha", "90"),
    ("--alpha-std-dev", "2"),
])
def test_an_alpha_flag_without_an_ensemble_is_refused(tmp_path, flag, value):
    with pytest.raises(ValueError, match=f"{flag} need an ensemble"):
        _kwargs(tmp_path, flag, value)


@pytest.mark.parametrize("extra, message", [
    (("--ep-dir", "{ep}", "--ensemble-member", "{m1}", "--je-factor", "2"), "not both"),
    (("--ensemble-member", "{m1}", "--ensemble-member", "{m2}", "--ensdim-alpha", "2", "--je-factor", "2"),
     "--ensdim-alpha belong to --ep-dir"),
    (("--ensemble-member", "{m1}", "--ensemble-member", "{m2}", "--ep-format", "1", "--je-factor", "2"),
     "--ep-format belong to --ep-dir"),
    (("--ep-dir", "{ep}", "--ensdim-alpha", "3"), "needs --je-factor"),
    (("--ep-dir", "{ep}", "--je-factor", "2"), "--ep-dir needs --ensdim-alpha"),
    (("--ep-dir", "{m1}", "--ensdim-alpha", "3", "--je-factor", "2"), "is not a directory"),
    (("--ep-dir", "{ep}", "--ensdim-alpha", "4", "--je-factor", "2"), "first missing is u.e004"),
    (("--ep-dir", "{empty}", "--ensdim-alpha", "3", "--ep-format", "11", "--je-factor", "2"),
     "has 0 of the 15 ep files .* first missing is U.e001"),
    (("--ensemble-member", "{m1}", "--ensemble-member", "nowhere", "--je-factor", "2"), "nowhere is not a file"),
    (("--ensemble-member", "{m1}", "--ensemble-member", "{late}", "--je-factor", "2"),
     "no record at the first guess's time"),
    (("--ensemble-member", "{m1}", "--ensemble-member", "{small}", "--je-factor", "2"), "the first member"),
    (("--ep-dir", "{ep}", "--ensdim-alpha", "3", "--je-factor", "2", "--alpha-vertloc-opt", "1"),
     "needs --be-vertloc"),
    (("--ep-dir", "{ep}", "--ensdim-alpha", "3", "--je-factor", "2", "--alpha-vertloc-opt", "1", "--be-vertloc",
      "nowhere.dat"), "nowhere.dat is not a file"),
    (("--ep-dir", "{ep}", "--ensdim-alpha", "3", "--je-factor", "2", "--be-vertloc", "{m1}"),
     "only with --alpha-vertloc-opt 1"),
])
def test_hybrid_flag_combinations_are_refused(case, extra, message):
    _wrf(case / "late", [_member_fields(1.0, 1.0, 0.1, 1.0)], ["2008-02-05_18:00:00"])
    _wrf(case / "small", [_member_fields(1.0, 1.0, 0.1, 1.0, nx=3)], [FG_TIME], nx=3)
    (case / "empty").mkdir()
    names = {k: str(case / k) for k in ("ep", "m1", "m2", "late", "small", "empty")}
    with pytest.raises(ValueError, match=message):
        _kwargs(case, *(x.format(**names) for x in extra))


def test_an_ensemble_on_another_grid_is_refused(case):
    small = _wrf(case / "fg_small", [{}], [FG_TIME], nx=3)
    from woof.var_cli import _hybrid_kwargs

    args = build_parser().parse_args(_argv(case, "--ep-dir", str(case / "ep"), "--ensdim-alpha", "3",
                                           "--je-factor", "2"))
    args.first_guess = small
    with pytest.raises(ValueError, match="the domain 3x3"):
        _hybrid_kwargs(args)


def test_main_reports_a_hybrid_refusal_without_a_traceback(case, capsys):
    for name in ("be.dat", "ob.ascii"):
        (case / name).write_text("x")
    argv = _argv(case, "--ep-dir", str(case / "ep"), "--ensdim-alpha", "5", "--je-factor", "2")
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert any(line.startswith("var: ") and "first missing is u.e004" in line for line in err.splitlines()), err


def test_members_are_read_at_the_first_guess_time_and_digested(case):
    from woof.da.var import hybrid as hy

    argv = ("--ensemble-member", str(case / "m1"), "--ensemble-member", str(case / "m2"), "--je-factor", "2",
            "--alpha-corr-scale", "150")
    kw, recipe = _kwargs(case, *argv)
    assert kw["ensemble"].ne == 2 and kw["vertloc"] is None
    assert (kw["hybrid"].je_factor, kw["hybrid"].alpha_corr_scale, kw["hybrid"].alpha_vertloc_opt) == (2.0, 150.0, 2)
    assert recipe["member_time"] == FG_TIME and recipe["members"] == ["m1", "m2"]
    # m2's second record (the first guess's time), not its first.
    perts = hy.ep_from_wrf([case / "m1", case / "m2"], time_index=[0, 1])
    assert recipe["perturbations_sha256"] == hy.perturbations_sha256(perts)
    assert np.all(perts["t"][1] > 0)
    # The same names with one changed word give another digest.
    _wrf(case / "m2", [_member_fields(5.0, 9.0, 0.0, 0.0), _member_fields(2.0, 4.5, 3.0, 90100.0)],
         ["2008-02-05_06:00:00", FG_TIME])
    _, changed = _kwargs(case, *argv)
    assert changed["members"] == recipe["members"]
    assert changed["perturbations_sha256"] != recipe["perturbations_sha256"]


def test_ep_files_and_vertloc_are_digested(case):
    from woof.da.var import hybrid as hy

    vl = case / "be.vertloc.dat"
    hy.write_vertloc(vl, np.ones(NZ), np.eye(NZ))
    argv = ("--ep-dir", str(case / "ep"), "--ensdim-alpha", "3", "--je-factor", "2", "--alpha-vertloc-opt", "1",
            "--be-vertloc", str(vl))
    kw, recipe = _kwargs(case, *argv)
    assert kw["vertloc"] is not None and recipe["ep_format"] == 1
    assert recipe["be_vertloc_sha256"] == hashlib.sha256(vl.read_bytes()).hexdigest()
    perts = hy.read_ep(case / "ep", 3)
    perts["t"][2] += 0.5
    hy.write_ep(case / "ep", perts)
    _, changed = _kwargs(case, *argv)
    assert changed["perturbations_sha256"] != recipe["perturbations_sha256"]
    assert changed["be_vertloc_sha256"] == recipe["be_vertloc_sha256"]


def test_ep_from_wrf_on_two_members_by_hand(case):
    """Exact words: two members, every value a binary fraction, ``p = PB = 1e5`` so ``(p/p0)**kappa = 1``."""
    from woof.da.var import hybrid as hy

    p = hy.ep_from_wrf([case / "m1", case / "m2"], time_index=[0, 1])
    # m1: U ramps 0..4 (u = i + 0.5), T 2, QVAPOR 1, PSFC 90000; m2: U ramps 2..6, T 4, QVAPOR 3, PSFC 90100.
    # mean u = i + 1.5, so u' = -1 and +1; V is twice the U offset: 0 and 4, mean 2; t = 302 and 304;
    # q = 1/2 and 3/4; ps = 90000 and 90100.
    expect = {"u": (-1.0, 1.0), "v": (-2.0, 2.0), "t": (-1.0, 1.0), "q": (-0.125, 0.125), "ps": (-50.0, 50.0)}
    for k, (a, b) in expect.items():
        assert p[k].shape == ((2, NX, NY) if k == "ps" else (2, NX, NY, NZ)), k
        np.testing.assert_array_equal(p[k][0], np.full(p[k].shape[1:], a), err_msg=k)
        np.testing.assert_array_equal(p[k][1], np.full(p[k].shape[1:], b), err_msg=k)
    with pytest.raises(ValueError, match="at least 2 members"):
        hy.ep_from_wrf([case / "m1"])
    with pytest.raises(ValueError, match="2 time indices for 3 members"):
        hy.ep_from_wrf([case / "m1", case / "m2", case / "m1"], time_index=[0, 1])


def _man():
    return json.loads(MANIFEST.read_text())["hybrid_o6"]


def _hybrid_dir() -> Path:
    h = os.environ.get("WOOF_WRFDA471_HYBRID_DIR")
    if not h or not (Path(h) / "ep").is_dir():
        pytest.skip("WOOF_WRFDA471_HYBRID_DIR (run_hybrid.sh's run directory) is not set")
    return Path(h)


def test_ep_from_wrf_is_gen_be_ep2s_ep_files():
    from woof.da.var import hybrid as hy

    h = _hybrid_dir()
    g = os.environ.get("WOOF_GENBE_DIR")
    if not g or not (Path(g) / "gen_be_fc").is_dir():
        pytest.skip("WOOF_GENBE_DIR with gen_be_fc/ is not set")
    man = _man()
    ours = hy.ep_from_wrf([Path(g) / "gen_be_fc" / m for m in man["members"]])
    nx, ny = ours["ps"].shape[1:]
    theirs = hy.read_ep(h / "ep", man["namelist"]["ensdim_alpha"], shape=(nx, ny))
    for k in hy.FIELDS:
        assert ours[k].shape == theirs[k].shape, k
        d = float(np.max(np.abs(ours[k] - theirs[k])))
        print(f"ep {k}: max |ours - gen_be_ep2| {d:.2e}")
        if k == "t":
            assert d <= 1e-12, d
        else:
            np.testing.assert_array_equal(ours[k], theirs[k], err_msg=k)


def _oracle():
    d = os.environ.get("WOOF_WRFDA471_DIR")
    if not d or not all((Path(d) / n).is_file() for n in ("be.dat", "ob.ascii", "fg")):
        pytest.skip("WOOF_WRFDA471_DIR with be.dat, ob.ascii and fg is not set")
    h = _hybrid_dir()
    for n in ("be.vertloc.dat", "wrfvar_output", "fg"):
        if not (h / n).is_file():
            pytest.skip(f"WOOF_WRFDA471_HYBRID_DIR has no {n}")
    d = Path(d)
    sha = [hashlib.sha256((x / "fg").read_bytes()).hexdigest() for x in (d, h)]
    assert sha[0] == sha[1], "the hybrid run's fg is not the tutorial's"
    return d, h


def _file(path: Path, name: str) -> np.ndarray:
    import netCDF4

    with netCDF4.Dataset(path) as nc:
        nc.set_auto_mask(False)
        return np.asarray(nc.variables[name][0]).T


def test_hybrid_o7_increments_against_wrfvar_output():
    from woof.da.var import hybrid as hy
    from woof.da.var.analysis import analyse, increment_stats

    d, h = _oracle()
    nl = _man()["namelist"]
    fg = _file(d / "fg", "T")
    nx, ny = fg.shape[:2]
    ens = hy.Ensemble.from_perturbations(hy.read_ep(h / "ep", nl["ensdim_alpha"], shape=(nx, ny)))
    cfg = hy.HybridConfig(je_factor=nl["je_factor"], alpha_corr_scale=nl["alpha_corr_scale"],
                          alpha_vertloc_opt=nl["alpha_vertloc_opt"])
    r = analyse(d / "be.dat", d / "ob.ascii", d / "fg", tier="reference", ntmax=nl["ntmax"], eps=nl["eps"],
                orthonorm_gradient=nl["orthonorm_gradient"], ensemble=ens, hybrid=cfg,
                vertloc=hy.read_vertloc(h / "be.vertloc.dat"))
    assert r.cg[-1].iterations == _man()["iterations"]
    rows = []
    for name in FIELDS:
        first = _file(d / "fg", name).astype(np.float64)
        theirs = _file(h / "wrfvar_output", name).astype(np.float64)
        s = increment_stats(r.wrf[name] - first, theirs - first)
        rows.append(f"{name:7s} rms {s['rms']:.4e}  max|d|/rms {s['max_over_rms']:.2e}  corr {s['corr']:.12f}")
        assert s["max_over_rms"] < 0.01, (name, s)
        assert s["corr"] >= 0.9999, (name, s)
    print("\n".join(["hybrid O7 (analysis - fg vs wrfvar_output - fg):"] + rows))


def test_var_door_runs_the_hybrid(tmp_path, capsys):
    d, h = _oracle()
    if not (d / "wrfbdy_d01").is_file():
        pytest.skip("WOOF_WRFDA471_DIR has no wrfbdy_d01")
    from conftest import cpu_preprocess_gap, netcdf_bridge_gap, netcdf_writer_gap

    for gap in (netcdf_bridge_gap(), netcdf_writer_gap(), cpu_preprocess_gap("gpuwm_analysis_binary_f64")):
        if gap:
            pytest.skip(gap)
    from woof.da.var import hybrid as hy
    from woof.var_cli import _hybrid_kwargs, _recipe

    nl = _man()["namelist"]
    out = tmp_path / "bundle"
    argv = ["var", "--be", str(d / "be.dat"), "--obs", str(d / "ob.ascii"), "--first-guess", str(d / "fg"),
            "--out", str(out), "--ntmax", "3", "--ep-dir", str(h / "ep"),
            "--ensdim-alpha", str(nl["ensdim_alpha"]), "--je-factor", str(nl["je_factor"]),
            "--alpha-vertloc-opt", "1", "--be-vertloc", str(h / "be.vertloc.dat")]
    assert main(argv) == 0
    ours = [s for s in capsys.readouterr().out.splitlines() if s.startswith(" minimize_cg ")]
    theirs = _man()["log"]
    assert len(ours) == 4
    for a, b in zip(ours, theirs):
        for x, y in zip(a.split()[3:5], b.split()[3:5]):
            assert abs(float(x.replace("D", "E")) / float(y.replace("D", "E")) - 1.0) <= 1e-12, (a, b)
    # The bundle's recipe digest is the door's recipe with the hybrid's inputs digested by content:
    # the ep files' words (independently re-read here) and be.vertloc.dat's bytes (the manifest's).
    manifest = json.loads((out / "manifest.json").read_text())
    hybrid = manifest["notes"]["var"]["hybrid"]
    nx, ny = _file(d / "fg", "T").shape[:2]
    assert hybrid["perturbations_sha256"] == hy.perturbations_sha256(
        hy.read_ep(h / "ep", nl["ensdim_alpha"], shape=(nx, ny)))
    assert hybrid["be_vertloc_sha256"] == _man()["vertloc_sha256"]
    args = build_parser().parse_args(argv)
    _, recipe_hybrid = _hybrid_kwargs(args)
    assert recipe_hybrid == hybrid
    recipe = json.dumps(_recipe(args, recipe_hybrid), sort_keys=True).encode()
    assert manifest["analysis"]["recipe_sha256"] == hashlib.sha256(recipe).hexdigest()
