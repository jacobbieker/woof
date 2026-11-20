"""woof.da.var.fgat: WRFDA's FGAT in the 3D-Var port (plan V5.1, gate GV5).

The synthetic tests run everywhere.  The oracle tests need
``WOOF_WRFDA471_FGAT_DIR``: the ``da`` directory of
``tools/wrfda_oracle/run_fgat.sh`` (WRFDA 4.7.1, ``num_fgat_time = 7`` on the
testdata's 12-18Z hourly observations, first guesses from a WRF 4.7.1
forecast, analysis at 15Z) with ``be.dat``, ``fg``, ``fg01``-``fg07``,
``ob01.ascii``-``ob07.ascii``, ``woof_iv.bin`` and ``wrfvar_output``; its
files are pinned in ``tests/data/wrfda471/manifest.json`` (``fgat``).

GV5, measured:

- the innovations of every slot against its own first guess: qc flags equal
  WRFDA's everywhere, ``inv`` to 1e-10 relative (2.3e-10 for metar ``p``,
  whose O-B is small; the crmath-against-glibc ulps of O5);
- ``Jo`` at the background is WRFDA's ``J_0`` to 1e-13;
- the minimisation takes WRFDA's 31 iterations to its final ``J`` (1e-12),
  and the increments meet O7's criteria against ``wrfvar_output``.
"""

from __future__ import annotations

import json
import os
import types
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import _var_wrfda_synthetic as syn
from woof.da.var import fgat as fm
from woof.da.var import innov_wrfda as iw
from woof.da.var import obs_wrfda as ow

MANIFEST = Path(__file__).parent / "data" / "wrfda471" / "manifest.json"
M = ow.MISSING_R
W = ("2008-02-05_12:00:00", "2008-02-05_15:00:00", "2008-02-05_18:00:00")


# --------------------------------------------------------------------------- synthetic
def test_time_slots_are_wrfdas():
    """``da_get_time_slots``: half-width first and last slots, ``itime_ana`` by fg index."""
    ts, ita = fm.time_slots(7, *W)
    assert list(ts - ts[0]) == [0.0, 30.0, 90.0, 150.0, 210.0, 270.0, 330.0, 360.0]
    assert ita == 4
    assert fm.time_slots(7, W[0], W[0], W[2])[1] == 1
    assert fm.time_slots(7, W[0], W[2], W[2])[1] == 7
    assert fm.time_slots(7, W[0], "2008-02-05_14:30:00", W[2])[1] is None
    ts1, ita1 = fm.time_slots(1, *W)
    assert list(ts1 - ts1[0]) == [0.0, 360.0] and ita1 == 1
    assert list(fm.slot_minutes(7, W[0], W[2]) - ts[0]) == [0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0]


def _sound(p, t=280.0):
    return syn.obs_type("sound", [dict(
        i=3, j=4, dx=0.25, dy=0.5, h=[M], p=[p],
        fields={"u": [(5.0, 0, 2.0)], "v": [(-2.0, 0, 2.0)], "t": [(t, 0, 3.0)], "q": [(0.005, 0, 0.001)]})])


def _two_slots(xb):
    col = iw._hcol(xb["p"], 2, 3, 0.25, 0.5, 0.75, 0.5)
    p = 0.5 * (col[4] + col[5])
    return fm.concat([syn.obs({"sound": _sound(p)}), syn.obs({"sound": _sound(p, 281.0)})])


def test_each_slot_is_compared_with_its_own_first_guess():
    xb1 = syn.background()
    xb2 = dict(xb1, t=xb1["t"] + 1.0)
    obs = _two_slots(xb1)
    assert list(obs.plocal["sound"]) == [0, 1, 2]
    iv = fm.innovations_fgat(obs, [xb1, xb2])
    one = iw.innovations(fm.slot(obs, 0), xb1)["sound"]
    two = iw.innovations(fm.slot(obs, 1), xb2)["sound"]
    s = iv["sound"]
    for v in ("u", "v", "t", "q"):
        assert np.array_equal(s.fields[v].inv, np.concatenate([one.fields[v].inv, two.fields[v].inv])), v
        assert np.array_equal(s.fields[v].qc, np.concatenate([one.fields[v].qc, two.fields[v].qc])), v
    # slot 2's observation is 1 K warmer and so is its background: the same O-B as slot 1
    assert s.fields["t"].inv[1] == pytest.approx(s.fields["t"].inv[0], abs=1e-9)
    # against the analysis-time background it would be 1 K larger
    assert iw.innovations(fm.slot(obs, 1), xb1)["sound"].fields["t"].inv[0] == pytest.approx(
        s.fields["t"].inv[0] + 1.0, abs=1e-9)
    geo = s.extra["geometry"]
    assert geo.nrep == 2 and np.array_equal(geo.zk, s.extra["zk"])
    assert np.array_equal(iw.interp_lin_3d(xb1["t"], geo),
                          np.concatenate([iw.interp_lin_3d(xb1["t"], one.extra["geometry"]),
                                          iw.interp_lin_3d(xb1["t"], two.extra["geometry"])]))


def test_slot_and_concat_round_trip():
    obs = _two_slots(syn.background())
    back = fm.concat([fm.slot(obs, 0), fm.slot(obs, 1)])
    t, u = obs["sound"], back["sound"]
    for k in t.info:
        assert np.array_equal(t.info[k], u.info[k]), k
    for k in t.fields:
        assert np.array_equal(t.fields[k].inv, u.fields[k].inv) and np.array_equal(t.fields[k].qc, u.fields[k].qc)
    assert list(back.plocal["sound"]) == [0, 1, 2]
    with pytest.raises(ValueError, match="plocal is None"):
        fm.slot(syn.obs({"sound": t}), 0)
    with pytest.raises(ValueError, match="2 observation slots but 1 first guesses"):
        fm.innovations_fgat(obs, [syn.background()])


def test_unknown_extra_is_refused_not_guessed():
    t = _sound(50000.0)
    t.extra["mystery"] = np.zeros(1)
    with pytest.raises(ValueError, match="mystery"):
        fm.concat([syn.obs({"sound": t})])


def test_check_fgat_refusals():
    times = [f"2008-02-05_{h}:00:00" for h in range(12, 19)]
    kw = dict(time_window_min=W[0], time_window_max=W[2])
    fm.check_fgat(7, times, **kw)
    with pytest.raises(ValueError, match="at least two"):
        fm.check_fgat(1, times[:1], **kw)
    with pytest.raises(ValueError, match="6 observation files for 7 first guesses"):
        fm.check_fgat(6, times, **kw)
    bad = list(times)
    bad[2] = times[3]
    with pytest.raises(ValueError, match="slot 3: the first guess is at 2008-02-05_15:00:00"):
        fm.check_fgat(7, bad, **kw)
    with pytest.raises(NotImplementedError, match="max_ext_its"):
        fm.check_fgat(7, times, max_ext_its=2, **kw)


def _cli_args(**kw):
    a = dict(obs=None, fgat_obs=[], fgat_first_guess=[], radiance_scene=[], first_guess=None,
             window_minutes=180.0, max_ext_its=1)
    a.update(kw)
    return types.SimpleNamespace(**a)


def _wrf_file(path: Path, stamp: str) -> Path:
    import netCDF4

    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("Time", None)
        nc.createDimension("DateStrLen", 19)
        nc.START_DATE = stamp
        t = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[0] = np.array(list(stamp), dtype="S1")
    return path


def test_cli_fgat_flags(tmp_path):
    from woof.var_cli import _fgat_kwargs

    fgs = [_wrf_file(tmp_path / f"fg0{n}", f"2008-02-05_{11 + n}:00:00") for n in range(1, 8)]
    obs = []
    for n in range(1, 8):
        (tmp_path / f"ob0{n}.ascii").write_text(f"slot {n}\n")
        obs.append(tmp_path / f"ob0{n}.ascii")
    fg = fgs[3]
    assert _fgat_kwargs(_cli_args(obs=tmp_path / "ob01.ascii", first_guess=fg)) == ({}, None)
    kw, recipe = _fgat_kwargs(_cli_args(fgat_obs=obs, fgat_first_guess=fgs, first_guess=fg))
    assert kw == {"fgat": fgs}
    assert [o["name"] for o in recipe["obs"]] == [p.name for p in obs]
    assert len({o["sha256"] for o in recipe["obs"]}) == 7 and len(recipe["first_guesses"]) == 7
    for bad, match in (
            (dict(first_guess=fg), "give --obs .* or one --fgat-obs"),
            (dict(obs=obs[0], fgat_obs=obs, fgat_first_guess=fgs, first_guess=fg), "not both"),
            (dict(fgat_obs=obs, first_guess=fg), "for every time slot"),
            (dict(fgat_obs=obs[:6], fgat_first_guess=fgs, first_guess=fg), "6 observation files for 7"),
            (dict(fgat_obs=obs, fgat_first_guess=fgs, first_guess=fg, window_minutes=120.0), "slot 1"),
            (dict(fgat_obs=obs, fgat_first_guess=fgs, first_guess=fg, max_ext_its=2), "max_ext_its"),
            (dict(fgat_obs=obs, fgat_first_guess=fgs, first_guess=fg, radiance_scene=[tmp_path]), "radiance"),
            (dict(fgat_obs=obs[:6] + [tmp_path / "nope"], fgat_first_guess=fgs, first_guess=fg), "not a file")):
        with pytest.raises(ValueError, match=match):
            _fgat_kwargs(_cli_args(**bad))


# --------------------------------------------------------------------------- oracle (GV5)
def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())["fgat"]


@lru_cache(maxsize=1)
def _dir() -> Path:
    d = os.environ.get("WOOF_WRFDA471_FGAT_DIR")
    man = _manifest()
    need = list(man["files_sha256"]) + ["be.dat", "woof_iv.bin", "wrfvar_output"]
    if not d or not all((Path(d) / n).is_file() for n in need):
        pytest.skip("WOOF_WRFDA471_FGAT_DIR (tools/wrfda_oracle/run_fgat.sh's da directory) is not set")
    import hashlib

    d = Path(d)
    for name, want in man["files_sha256"].items():
        got = hashlib.sha256((d / name).read_bytes()).hexdigest()
        assert got == want, f"{d / name} has sha256 {got[:12]}..., not the manifest's {want[:12]}...: another run"
    return d


def _slots(d: Path):
    return [d / f"ob0{n}.ascii" for n in range(1, 8)], [d / f"fg0{n}" for n in range(1, 8)]


@lru_cache(maxsize=1)
def _innovations():
    from woof.da.var.analysis import read_first_guess, wrftoxb
    from woof.da.var.oracle import read_dump

    d = _dir()
    obs_paths, fg_paths = _slots(d)
    fg = read_first_guess(d / "fg")
    proj = ow.LambertWRFDA.from_attributes(fg.attrs)
    obs = fm.read_obs_ascii_fgat(obs_paths, proj, **_manifest()["window"])
    xbs = [wrftoxb(read_first_guess(p))[0] for p in fg_paths]
    return obs, xbs, wrftoxb(fg)[0], proj, read_dump(d / "woof_iv.bin"), fm.innovations_fgat(obs, xbs)


#: GV5's inv tolerance (relative to the oracle's |inv|); metar p's O-B is small, so its ulps weigh more.
INV_RTOL = {("metar", "p"): 3e-10}


@pytest.mark.parametrize("name", [n for n in ow.TYPES if n in iw.IMPLEMENTED])
def test_gv5_innovations_against_wrfda_fgat(name):
    *_, dump, iv = _innovations()
    t = iv[name]
    tag = "gpsrf" if name == "gpsref" else name
    if f"{tag}%levels" not in dump:
        assert t.nlocal == 0
        return
    for v, f in t.fields.items():
        inv, qc = dump[f"{name}%{v}%inv"], dump[f"{name}%{v}%qc"]
        assert np.array_equal(f.qc, qc), (name, v, np.flatnonzero(f.qc != qc)[:5])
        rel = np.abs(f.inv - inv) / np.maximum(np.abs(inv), 1e-300)
        rel[(inv == 0.0) & (f.inv == 0.0)] = 0.0
        assert rel.max(initial=0.0) <= INV_RTOL.get((name, v), 1e-10), (name, v, rel.max())


def test_gv5_jo_at_the_background_is_wrfdas_initial_cost():
    *_, iv = _innovations()
    want = float(_manifest()["j0"].replace("D", "e"))
    assert iw.jo_total(iw.jo(iv)) == pytest.approx(want, rel=1e-13)


def test_the_analysis_time_slot_is_3dvar():
    """fg04 is fg: slot 4's innovations are 3D-Var's for ob04.ascii, word for word,
    while the other slots differ from what the analysis-time background gives."""
    obs, xbs, xb, proj, _, iv = _innovations()
    d = _dir()
    one = iw.innovations(ow.read_obs_ascii(d / "ob04.ascii", proj, **_manifest()["window"]), xb)
    four = fm.slot(iv, 3)
    for name in ("sound", "synop", "geoamv"):
        for v, f in four[name].fields.items():
            assert np.array_equal(f.inv, one[name].fields[v].inv), (name, v)
    other = iw.innovations(fm.slot(obs, 0), xb)
    assert not np.array_equal(other["sound"].fields["t"].inv, fm.slot(iv, 0)["sound"].fields["t"].inv)


@pytest.fixture(scope="module")
def analysis():
    from woof.da.var.analysis import analyse

    d = _dir()
    obs_paths, fg_paths = _slots(d)
    lines = []
    w = _manifest()["window"]
    r = analyse(d / "be.dat", obs_paths, d / "fg", fgat=fg_paths, time_window=(w["time_window_min"],
                w["time_window_max"]), write=lines.append)
    return r, lines


def test_gv5_minimisation_follows_wrfda(analysis):
    r, lines = analysis
    man = _manifest()
    assert r.cg[-1].iterations == man["iterations"]
    assert r.nobs == man["nobs"]
    assert r.cg[-1].j_final == pytest.approx(float(man["j_final"].replace("D", "e")), rel=1e-12)


def test_gv5_increments_against_wrfvar_output(analysis):
    import netCDF4

    from woof.da.var.analysis import increment_stats

    r, _ = analysis
    d = _dir()

    def field(path, name):
        with netCDF4.Dataset(path) as nc:
            nc.set_auto_mask(False)
            return np.asarray(nc.variables[name][0]).T.astype(np.float64)

    rows = []
    for name in ("U", "V", "T", "QVAPOR", "MU", "P", "PH", "PSFC"):
        first = field(d / "fg", name)
        s = increment_stats(r.wrf[name] - first, field(d / "wrfvar_output", name) - first)
        rows.append(f"{name:7s} rms {s['rms']:.4e}  max|d|/rms {s['max_over_rms']:.2e}  corr {s['corr']:.12f}")
        assert s["max_over_rms"] < 0.01, (name, s)
        assert s["corr"] >= 0.9999, (name, s)
    print("\n".join(["GV5 (analysis - fg vs wrfvar_output - fg):"] + rows))


def test_fgat_refuses_what_it_does_not_port(tmp_path):
    from woof.da.var.analysis import analyse

    d = _dir()
    obs_paths, fg_paths = _slots(d)
    with pytest.raises(NotImplementedError, match="max_ext_its"):
        analyse(d / "be.dat", obs_paths, d / "fg", fgat=fg_paths, window_minutes=180.0, max_ext_its=2)
    with pytest.raises(ValueError, match="slot 2"):
        analyse(d / "be.dat", obs_paths, d / "fg", fgat=[fg_paths[0]] + fg_paths[:-1], window_minutes=180.0)


def test_var_door_runs_fgat(tmp_path, capsys):
    """``woof var --fgat-obs ... --fgat-first-guess ...``: WRFDA's first CG lines, and the
    bundle's recipe digest carries every slot's files by content.

    The 15Z first guess is a file WRF wrote mid-forecast, so its ``START_DATE`` is the
    forecast's 12Z start, and the bundle refuses a wrfinput-form analysis whose
    ``START_DATE`` is not its valid time (WRFDA's ``da_update_firstguess`` restamps it;
    woof's patched copy does not).  The door gets a copy of ``fg`` stamped 15Z, which
    leaves every field, and so the analysis, unchanged.
    """
    import hashlib
    import shutil

    import netCDF4

    d = _dir()
    from conftest import cpu_preprocess_gap, netcdf_bridge_gap, netcdf_writer_gap

    for gap in (netcdf_bridge_gap(), netcdf_writer_gap(), cpu_preprocess_gap("gpuwm_analysis_binary_f64")):
        if gap:
            pytest.skip(gap)
    from woof.cli import build_parser, main
    from woof.var_cli import _fgat_kwargs, _recipe

    obs_paths, fg_paths = _slots(d)
    fg = shutil.copyfile(d / "fg", tmp_path / "fg")
    with netCDF4.Dataset(fg, "a") as nc:
        assert nc.START_DATE == "2008-02-05_12:00:00" and nc.data_model == "NETCDF3_64BIT_OFFSET"
        nc.START_DATE = nc.SIMULATION_START_DATE = _manifest()["window"]["analysis_date"]
    out = tmp_path / "bundle"
    argv = ["var", "--be", str(d / "be.dat"), "--first-guess", str(fg), "--out", str(out),
            "--ntmax", "2", "--window-minutes", "180"]
    for o, f in zip(obs_paths, fg_paths):
        argv += ["--fgat-obs", str(o), "--fgat-first-guess", str(f)]
    assert main(argv) == 0
    first = [s for s in capsys.readouterr().out.splitlines() if s.startswith(" minimize_cg ")][0].split()
    man = _manifest()
    for got, want in zip(first[3:5], (man["j0"], man["g0"])):
        assert float(got.replace("D", "e")) == pytest.approx(float(want.replace("D", "e")), rel=1e-12)
    manifest = json.loads((out / "manifest.json").read_text())
    fgat = manifest["notes"]["var"]["fgat"]
    assert [o["sha256"] for o in fgat["obs"]] == [man["files_sha256"][p.name] for p in obs_paths]
    assert [o["sha256"] for o in fgat["first_guesses"]] == [man["files_sha256"][p.name] for p in fg_paths]
    args = build_parser().parse_args(argv)
    recipe = _recipe(args, None)
    recipe["fgat"] = _fgat_kwargs(args)[1]
    assert manifest["analysis"]["recipe_sha256"] == hashlib.sha256(
        json.dumps(recipe, sort_keys=True).encode()).hexdigest()
