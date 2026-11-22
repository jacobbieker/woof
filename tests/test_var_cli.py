"""``woof var``: the door to the WRFDA 3D-Var port (plan V1.5).

Parser refusals run everywhere; the door run needs the WRFDA 4.7.1 tutorial
(``WOOF_WRFDA471_DIR``, see tests/test_var_o6.py) and the bundle writer's
bridges.  The run is short (``--ntmax 3``): its log is WRFDA's first lines
and its bundle is accepted by ``woof run --analysis``'s door.  The full
50-iteration minimisation is tests/test_var_o6.py and test_var_analysis.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from woof.cli import build_parser, main

MANIFEST = Path(__file__).parent / "data" / "wrfda471" / "manifest.json"


def _base(tmp_path):
    return ["var", "--be", "be.dat", "--obs", "ob.ascii", "--first-guess", "fg", "--out", str(tmp_path / "b")]


def test_parser_defaults_are_the_tutorial_namelist(tmp_path):
    args = build_parser().parse_args(_base(tmp_path))
    assert (args.tier, args.ntmax, args.eps, args.max_ext_its, args.orthonorm_gradient, args.check_rh,
            args.window_minutes, args.ph) == ("reference", 50, 0.01, 1, True, 0, 60.0, "analysis")
    assert callable(args.func)


@pytest.mark.parametrize("option, value", [("--ntmax", "0"), ("--eps", "nan"), ("--max-ext-its", "-1"),
                                           ("--window-minutes", "0"), ("--tier", "fast"), ("--check-rh", "2")])
def test_parser_refuses_by_option_name(tmp_path, capsys, option, value):
    with pytest.raises(SystemExit) as stopped:
        build_parser().parse_args([*_base(tmp_path), f"{option}={value}"])
    assert stopped.value.code == 2
    assert f"argument {option}" in capsys.readouterr().err


def test_missing_inputs_and_existing_out_are_refused(tmp_path, capsys):
    assert main(_base(tmp_path)) == 2
    assert "--be be.dat is not a file" in capsys.readouterr().err
    for name in ("be.dat", "ob.ascii", "fg"):
        (tmp_path / name).write_text("x")
    (tmp_path / "b").mkdir()
    argv = ["var", "--be", str(tmp_path / "be.dat"), "--obs", str(tmp_path / "ob.ascii"),
            "--first-guess", str(tmp_path / "fg"), "--out", str(tmp_path / "b")]
    assert main(argv) == 2
    assert "exists" in capsys.readouterr().err


def test_door_prints_wrfdas_log_and_writes_a_bundle_the_run_door_takes(tmp_path, capsys):
    d = os.environ.get("WOOF_WRFDA471_DIR")
    need = ("be.dat", "ob.ascii", "fg", "wrfbdy_d01")
    if not d or not all((Path(d) / n).is_file() for n in need):
        pytest.skip("WOOF_WRFDA471_DIR with be.dat, ob.ascii, fg and wrfbdy_d01 is not set")
    from conftest import cpu_preprocess_gap, netcdf_bridge_gap, netcdf_writer_gap

    for gap in (netcdf_bridge_gap(), netcdf_writer_gap(), cpu_preprocess_gap("gpuwm_analysis_binary_f64")):
        if gap:
            pytest.skip(gap)
    d = Path(d)
    out = tmp_path / "bundle"
    rc = main(["var", "--be", str(d / "be.dat"), "--obs", str(d / "ob.ascii"), "--first-guess", str(d / "fg"),
               "--out", str(out), "--ntmax", "3", "--log", str(tmp_path / "var.log")])
    assert rc == 0
    printed = capsys.readouterr().out.splitlines()
    o6 = json.loads(MANIFEST.read_text())["o6_minimise"]
    ours = [s for s in printed if s.startswith(" minimize_cg ")]
    theirs = [s for s in o6["log"] if s.startswith(" minimize_cg ")]
    assert len(ours) == 4
    for a, b in zip(ours, theirs):  # wrftoxb off glibc is an ulp away: J and |g| to 1e-12
        for x, y in zip(a.split()[3:5], b.split()[3:5]):
            assert abs(float(x.replace("D", "E")) / float(y.replace("D", "E")) - 1.0) <= 1e-12, (a, b)
    starting = [s for s in printed if s.startswith("Starting cost function")]
    assert len(starting) == 1 and starting[0].split(",")[0] == o6["log"][3].split(",")[0]
    assert (tmp_path / "var.log").read_text().count(" minimize_cg ") == 4
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["analysis"]["recipe_id"] == "woof-var-3dvar"
    assert manifest["notes"]["var"]["iterations"] == [3]

    from woof.da.analysis_door import resolve

    run = tmp_path / "run"
    run.mkdir()
    (run / "wrfinput_d01").symlink_to(d / "fg")
    (run / "wrfbdy_d01").symlink_to(d / "wrfbdy_d01")
    (run / "namelist.input").write_text(
        "&time_control\n run_hours = 6,\n/\n&domains\n max_dom = 1,\n/\n&dynamics\n hypsometric_opt = 2,\n/\n")
    got = resolve({"analysis": str(out)}, directory=run, outdir=tmp_path / "out", run_seconds=21600.0)
    assert Path(got["directory"]).is_dir()
