"""Hybrid 3DEnVar end to end: ``woof var``'s flags, ``analyse`` and gen_be_ep2's perturbations.

Everywhere: the hybrid flags' refusals, and :func:`~woof.da.var.hybrid.ep_from_wrf`
on synthetic members (the destaggering, ``t`` from theta and pressure, the
running mean and the guards).

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


def _argv(tmp_path, *extra):
    return ["var", "--be", "be.dat", "--obs", "ob.ascii", "--first-guess", "fg", "--out",
            str(tmp_path / "b"), *extra]


def _kwargs(tmp_path, *extra):
    from woof.var_cli import _hybrid_kwargs

    return _hybrid_kwargs(build_parser().parse_args(_argv(tmp_path, *extra)))


def test_hybrid_flags_default_to_off_and_wrfdas_namelist(tmp_path):
    args = build_parser().parse_args(_argv(tmp_path))
    assert (args.ensemble_member, args.ep_dir, args.je_factor, args.alpha_corr_scale, args.alpha_vertloc_opt,
            args.max_vert_var_alpha, args.alpha_std_dev, args.ep_format) == ([], None, None, 200.0, 2, 99.0, 1.0, 1)
    assert _kwargs(tmp_path) == {}


@pytest.mark.parametrize("extra, message", [
    (("--je-factor", "2"), "need --ensemble-member or --ep-dir"),
    (("--ep-dir", "ep"), "needs --je-factor"),
    (("--ep-dir", "ep", "--je-factor", "2"), "--ep-dir needs --ensdim-alpha"),
    (("--ep-dir", "ep", "--ensemble-member", "m1", "--je-factor", "2"), "not both"),
    (("--ensemble-member", "missing.nc", "--je-factor", "2"), "is not a file"),
])
def test_hybrid_flags_refuse(tmp_path, extra, message):
    with pytest.raises(ValueError, match=message):
        _kwargs(tmp_path, *extra)


def test_vertloc_file_goes_with_option_1(tmp_path, monkeypatch):
    from woof.da.var import hybrid as hy

    perts = _synthetic_perts()
    monkeypatch.setattr(hy, "read_ep", lambda *a, **k: perts)
    base = ("--ep-dir", "ep", "--ensdim-alpha", "3", "--je-factor", "2")
    with pytest.raises(ValueError, match="needs --be-vertloc"):
        _kwargs(tmp_path, *base, "--alpha-vertloc-opt", "1")
    with pytest.raises(ValueError, match="only with --alpha-vertloc-opt 1"):
        _kwargs(tmp_path, *base, "--be-vertloc", "vl.dat")
    kw = _kwargs(tmp_path, *base, "--alpha-corr-scale", "150")
    assert kw["ensemble"].ne == 3 and kw["vertloc"] is None
    assert (kw["hybrid"].je_factor, kw["hybrid"].alpha_corr_scale) == (2.0, 150.0)


def test_main_reports_a_hybrid_refusal(tmp_path, capsys):
    for name in ("be.dat", "ob.ascii", "fg"):
        (tmp_path / name).write_text("x")
    argv = ["var", "--be", str(tmp_path / "be.dat"), "--obs", str(tmp_path / "ob.ascii"),
            "--first-guess", str(tmp_path / "fg"), "--out", str(tmp_path / "b"), "--ep-dir", str(tmp_path)]
    assert main(argv) == 2
    assert "var: an ensemble needs --je-factor" in capsys.readouterr().err


def _synthetic_perts(ne=3, shape=(4, 3, 2)):
    rng = np.random.default_rng(0)
    out = {k: rng.standard_normal((ne, *shape)) for k in ("u", "v", "t", "q")}
    out["ps"] = rng.standard_normal((ne, *shape[:2]))
    return out


def _member(rng, nx=4, ny=3, nz=2):
    return {"U": rng.uniform(-10, 10, (nx + 1, ny, nz)), "V": rng.uniform(-10, 10, (nx, ny + 1, nz)),
            "T": rng.uniform(-20, 20, (nx, ny, nz)), "PB": rng.uniform(5e4, 1e5, (nx, ny, nz)),
            "P": rng.uniform(-500, 500, (nx, ny, nz)), "QVAPOR": rng.uniform(0, 0.02, (nx, ny, nz)),
            "PSFC": rng.uniform(9e4, 1e5, (nx, ny))}


def test_ep_from_wrf_on_synthetic_members(monkeypatch):
    from woof.da.var import gen_be as gb
    from woof.da.var import hybrid as hy

    rng = np.random.default_rng(1)
    files = {f"m{i}": _member(rng) for i in range(4)}
    monkeypatch.setattr(gb, "read_wrf_history", lambda path, time_index=0: files[Path(path).name])
    p = hy.ep_from_wrf(list(files))
    full = {}
    for k, f in files.items():
        full.setdefault("u", []).append(0.5 * (f["U"][:-1] + f["U"][1:]))
        full.setdefault("v", []).append(0.5 * (f["V"][:, :-1] + f["V"][:, 1:]))
        full.setdefault("t", []).append((300.0 + f["T"]) * ((f["PB"] + f["P"]) / 1e5) ** gb.KAPPA)
        full.setdefault("q", []).append(f["QVAPOR"] / (1.0 + f["QVAPOR"]))
        full.setdefault("ps", []).append(f["PSFC"])
    for k, v in full.items():
        v = np.stack(v)
        assert p[k].shape == v.shape, k
        np.testing.assert_allclose(p[k], v - v.mean(axis=0), rtol=0, atol=1e-12 * np.max(np.abs(v)), err_msg=k)
    ens = hy.Ensemble.from_perturbations(p)
    assert ens.ne == 4 and ens.shape == (4, 3, 2)
    with pytest.raises(ValueError, match="at least 2 members"):
        hy.ep_from_wrf(["m0"])
    files["odd"] = _member(rng, nx=5)
    with pytest.raises(ValueError, match="the first member"):
        hy.ep_from_wrf(["m0", "odd"])


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
    nl = _man()["namelist"]
    out = tmp_path / "bundle"
    rc = main(["var", "--be", str(d / "be.dat"), "--obs", str(d / "ob.ascii"), "--first-guess", str(d / "fg"),
               "--out", str(out), "--ntmax", "3", "--ep-dir", str(h / "ep"),
               "--ensdim-alpha", str(nl["ensdim_alpha"]), "--je-factor", str(nl["je_factor"]),
               "--alpha-vertloc-opt", "1", "--be-vertloc", str(h / "be.vertloc.dat")])
    assert rc == 0
    ours = [s for s in capsys.readouterr().out.splitlines() if s.startswith(" minimize_cg ")]
    theirs = _man()["log"]
    assert len(ours) == 4
    for a, b in zip(ours, theirs):
        for x, y in zip(a.split()[3:5], b.split()[3:5]):
            assert abs(float(x.replace("D", "E")) / float(y.replace("D", "E")) - 1.0) <= 1e-12, (a, b)
    # The bundle lists the hybrid settings in its notes (and digests them into the recipe).
    assert any('"hybrid"' in f.read_text() for f in out.rglob("*.json"))
