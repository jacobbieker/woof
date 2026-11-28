"""``woof var --4dvar`` (PR 4D.4): the door's 4D-Var flags and refusals.

Breakage this prevents:

- 3D-Var's (and FGAT's) defaults changing because the 4D-Var flags exist;
- a WRFDA option that is not ported (``var4d_lbc``, ``jcdfi_use``,
  ``multi_inc``), a window WRFDA refuses (``var4d_bin`` not dividing it, the
  time step not dividing the bin), a first guess off the window's start,
  outer loops (the door has no nonlinear model to rerun) or a device tier
  running instead of being refused by name, before any field is read;
- the bin and time step not reaching ``analyse`` (``Var4D``) or the bundle's
  recipe.

On the tutorial (``WOOF_WRFDA471_FGAT_DIR``) the door runs end to end and its
bundle records the 4D-Var settings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from woof.cli import build_parser, main

W = ("2008-02-05_12:00:00", "2008-02-05_18:00:00")


def _wrf_file(path: Path, stamp: str, dt: float | None = 360.0) -> Path:
    import netCDF4

    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("Time", None)
        nc.createDimension("DateStrLen", 19)
        nc.START_DATE = stamp
        if dt is not None:
            nc.DT = np.float32(dt)
        t = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[0] = np.array(list(stamp), dtype="S1")
    return path


def _files(tmp_path, dt=360.0):
    fgs = [_wrf_file(tmp_path / f"fg0{n}", f"2008-02-05_{11 + n}:00:00", dt) for n in range(1, 8)]
    obs = []
    for n in range(1, 8):
        (tmp_path / f"ob0{n}.ascii").write_text(f"slot {n}\n")
        obs.append(tmp_path / f"ob0{n}.ascii")
    (tmp_path / "be.dat").write_bytes(b"")
    return obs, fgs


def _argv(tmp_path, obs, fgs, *extra, first=None):
    argv = ["var", "--be", str(tmp_path / "be.dat"), "--first-guess", str(first or fgs[0]),
            "--out", str(tmp_path / "out"), *extra]
    for o, f in zip(obs, fgs):
        argv += ["--fgat-obs", str(o), "--fgat-first-guess", str(f)]
    return argv


def test_parser_defaults_leave_3dvar_alone(tmp_path):
    args = build_parser().parse_args(["var", "--be", "be.dat", "--obs", "ob.ascii", "--first-guess", "fg",
                                      "--out", str(tmp_path / "b")])
    assert (args.var4d, args.var4d_bin, args.time_step, args.time_window_min, args.time_window_max,
            args.var4d_press, args.var4d_ph, args.var4d_lbc, args.jcdfi_use, args.multi_inc) == (
        False, None, None, None, None, "wrfda", "zero", False, False, 0)
    from woof.var_cli import _var4d_kwargs

    assert _var4d_kwargs(args) == ({}, None)


def test_var4d_kwargs_from_the_flags(tmp_path):
    from woof.da.var.fourd import IdentityModel
    from woof.var_cli import _var4d_kwargs

    obs, fgs = _files(tmp_path)
    kw, recipe = _var4d_kwargs(build_parser().parse_args(_argv(tmp_path, obs, fgs, "--4dvar")))
    v = kw["var4d"]
    assert kw["time_window"] == W
    assert (v.var4d_bin, v.time_step, v.press, v.ph, v.nl_model) == (3600, 360, "wrfda", "zero", None)
    assert isinstance(v.model, IdentityModel)
    assert recipe == {"time_window": list(W), "var4d_bin": 3600, "time_step": 360, "num_fgat_time": 7,
                      "model": "identity", "press": "wrfda", "ph": "zero"}
    kw, recipe = _var4d_kwargs(build_parser().parse_args(_argv(
        tmp_path, obs, fgs, "--4dvar", "--var4d-bin", "3600", "--time-step", "300", "--var4d-press", "dnw",
        "--var4d-ph", "hydrostatic", "--time-window-max", W[1])))
    assert (kw["var4d"].time_step, kw["var4d"].press, kw["var4d"].ph) == (300, "dnw", "hydrostatic")


@pytest.mark.parametrize("extra, match", [
    (("--4dvar", "--var4d-lbc"), "WRFDA's var4d_lbc is not ported"),
    (("--jcdfi-use",), "WRFDA's jcdfi_use is not ported"),
    (("--4dvar", "--multi-inc", "1"), "multi_inc"),
    (("--4dvar", "--multi-inc", "2"), "multi_inc"),
    (("--var4d-bin", "3600"), "--var4d-bin need --4dvar"),
    (("--time-step", "60", "--var4d-ph", "hydrostatic"), "--time-step, --var4d-ph need --4dvar"),
    (("--4dvar", "--var4d-bin", "5000"), "4DVAR assimilation window must be evenly divisible by var4d_bin"),
    (("--4dvar", "--time-step", "7"), "var4d_bin must be evenly divisible by time_step"),
    (("--4dvar", "--var4d-bin", "1800"), "makes 13 slots"),
    (("--4dvar", "--time-window-min", "2008-02-05_11:00:00"), "not at the window's start"),
    (("--4dvar", "--time-window-max", "2008-02-05 18"), "not YYYY-MM-DD_hh:mm:ss"),
    (("--4dvar", "--max-ext-its", "2"), "no model to rerun"),
    (("--4dvar", "--tier", "oracle"), "use --tier reference"),
])
def test_door_refuses_by_name_before_reading_fields(tmp_path, capsys, extra, match):
    obs, fgs = _files(tmp_path)
    assert main(_argv(tmp_path, obs, fgs, *extra)) == 2
    assert match in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_door_refusals_that_need_other_inputs(tmp_path, capsys):
    obs, fgs = _files(tmp_path, dt=None)
    assert main(_argv(tmp_path, obs, fgs, "--4dvar")) == 2
    assert "has no DT attribute; give --time-step" in capsys.readouterr().err
    obs, fgs = _files(tmp_path, dt=37.5)
    assert main(_argv(tmp_path, obs, fgs, "--4dvar")) == 2
    assert "give --time-step" in capsys.readouterr().err
    obs, fgs = _files(tmp_path)
    assert main(_argv(tmp_path, [], [], "--4dvar", "--obs", str(obs[0]), first=fgs[0])) == 2
    assert "--4dvar needs the trajectory" in capsys.readouterr().err
    # the analysis at the window's start: a mid-window --first-guess is refused
    assert main(_argv(tmp_path, obs, fgs, "--4dvar", "--time-window-min", "2008-02-05_12:00:00",
                      first=fgs[3])) == 2
    assert "not at the window's start" in capsys.readouterr().err


def test_door_runs_4dvar_on_the_tutorial(tmp_path, capsys):
    """``woof var --4dvar`` with the FGAT lane's slots as the trajectory (identity TL):
    WRFDA's first CG lines, and the bundle records the 4D-Var settings."""
    d = os.environ.get("WOOF_WRFDA471_FGAT_DIR")
    if not d:
        pytest.skip("WOOF_WRFDA471_FGAT_DIR is not set")
    from conftest import cpu_preprocess_gap, netcdf_bridge_gap, netcdf_writer_gap

    for gap in (netcdf_bridge_gap(), netcdf_writer_gap(), cpu_preprocess_gap("gpuwm_analysis_binary_f64")):
        if gap:
            pytest.skip(gap)
    d = Path(d)
    out = tmp_path / "bundle"
    argv = ["var", "--be", str(d / "be.dat"), "--first-guess", str(d / "fg01"), "--out", str(out), "--ntmax", "2",
            "--4dvar"]
    for n in range(1, 8):
        argv += ["--fgat-obs", str(d / f"ob0{n}.ascii"), "--fgat-first-guess", str(d / f"fg0{n}")]
    assert main(argv) == 0
    lines = [s for s in capsys.readouterr().out.splitlines() if s.startswith(" minimize_cg ")]
    assert len(lines) == 3
    manifest = json.loads((out / "manifest.json").read_text())
    v4 = manifest["notes"]["var"]["var4d"]
    assert (v4["var4d_bin"], v4["time_step"], v4["num_fgat_time"], v4["model"]) == (3600, 360, 7, "identity")
    assert manifest["valid_time"].startswith("2008-02-05T12") or "12:00" in manifest["valid_time"]
