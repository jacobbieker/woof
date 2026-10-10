"""``woof hex remap``: the door's argument checks, its receipt, and one real run.

The engine-free tests drive the door with a fake ``rw_mpas_remap`` (a small
Python script) so every refusal and the receipt shape are pinned without a
Rust build.  The last test runs the real binary end to end on two tiny
generated meshes when this checkout has built it, and is skipped otherwise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import textwrap

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import remap_door  # noqa: E402
from woof.hex.cli import build_parser  # noqa: E402
from woof.hex.remap_door import RemapDoorRefusal  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
CRATE = REPO / "tools" / "rustwx" / "crates" / "rw-mpas"


def _nc(path: Path, variables: dict[str, np.ndarray] | None = None) -> Path:
    from netCDF4 import Dataset

    with Dataset(path, "w", format="NETCDF3_64BIT_DATA") as d:
        d.createDimension("nCells", 3)
        for name, value in (variables or {}).items():
            v = d.createVariable(name, np.float32, ("nCells",))
            v[:] = value
    return path


def _fake_engine(tmp_path: Path, *, exit_code: int = 0, refusal: str | None = None) -> Path:
    script = tmp_path / "rw_mpas_remap"
    receipt = {
        "status": "refused" if refusal else "remapped",
        "refusal": refusal,
        "coverage": {"min_coverage": 0.25 if refusal else 1.0, "cells_below_required": 7 if refusal else 0},
        "budgets": {"horizontal_stage_dry_air_relative_change": 1e-12, "dry_air_relative_change": 2e-6,
                    "water_vapour_relative_change": 3e-6, "kinetic_energy_relative_change": -1e-3},
    }
    script.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import json, sys
        argv = sys.argv[1:]
        args = dict(zip(argv[::2], argv[1::2]))
        open(args["--receipt"], "w").write(json.dumps({receipt!r}))
        open(args["--out"] + ".argv", "w").write(json.dumps(argv))
        if {exit_code} == 0:
            open(args["--out"], "wb").write(b"CDF")
        else:
            sys.stderr.write("rw_mpas_remap: refused\\n")
        sys.exit({exit_code})
        """), encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _args(tmp_path: Path, *extra: str):
    grid_a = _nc(tmp_path / "A.grid.nc")
    state_a = _nc(tmp_path / "A.init.nc")
    grid_b = _nc(tmp_path / "B.init.nc", {"zgrid": np.zeros(3)})
    argv = ["remap", "--from-grid", str(grid_a), "--from-state", str(state_a),
            "--to-grid", str(grid_b), "-o", str(tmp_path / "B.state.nc"), *extra]
    return build_parser().parse_args(argv)


def test_the_remap_door_is_a_hex_subcommand_with_the_contract_flags(tmp_path: Path) -> None:
    arguments = _args(tmp_path)
    assert arguments.handler is remap_door.run_remap
    assert arguments.balance == "hydrostatic"
    assert arguments.virtual_factor == "reproduce-fortran"
    assert arguments.min_coverage == pytest.approx(1.0 - 1.0e-6)
    assert arguments.to_static is None and arguments.to_vertical is None


def test_the_engine_argv_carries_every_input_and_the_switches(tmp_path: Path) -> None:
    static = _nc(tmp_path / "B.static.nc")
    vertical = _nc(tmp_path / "B.vertical.nc", {"zgrid": np.zeros(3)})
    arguments = _args(tmp_path, "--to-static", str(static), "--to-vertical", str(vertical),
                      "--balance", "carry", "--min-coverage", "0.99")
    argv, inputs, receipt, engine_receipt = remap_door.build_argv(arguments, Path("/bin/true"))
    assert argv[0] == "/bin/true"
    pairs = dict(zip(argv[1::2], argv[2::2]))
    assert pairs["--to-static"] == str(static)
    assert pairs["--to-vertical"] == str(vertical)
    assert pairs["--balance"] == "carry"
    assert pairs["--virtual-factor"] == "reproduce-fortran"
    assert float(pairs["--min-coverage"]) == pytest.approx(0.99)
    assert receipt.name == "B.state.nc.receipt.json"
    assert engine_receipt.name == "B.state.nc.receipt.engine.json"
    assert inputs["to_static"] == static


def test_a_target_without_a_vertical_grid_is_refused_with_the_command_that_builds_one(tmp_path: Path) -> None:
    arguments = _args(tmp_path)
    _nc(Path(arguments.to_grid))  # rewrite B with no zgrid
    with pytest.raises(RemapDoorRefusal) as caught:
        remap_door.build_argv(arguments, Path("/bin/true"))
    assert "no zgrid" in str(caught.value)
    assert "woof hex vertical" in str(caught.value)


def test_an_output_that_names_an_input_is_refused(tmp_path: Path) -> None:
    arguments = _args(tmp_path)
    arguments.out = arguments.from_state
    with pytest.raises(RemapDoorRefusal, match="never overwrites"):
        remap_door.build_argv(arguments, Path("/bin/true"))


def test_an_existing_output_needs_clobber(tmp_path: Path) -> None:
    arguments = _args(tmp_path)
    Path(arguments.out).write_bytes(b"old")
    with pytest.raises(RemapDoorRefusal, match="--clobber"):
        remap_door.build_argv(arguments, Path("/bin/true"))
    arguments.clobber = True
    remap_door.build_argv(arguments, Path("/bin/true"))


@pytest.mark.parametrize("value", ["0", "1.5", "-0.1"])
def test_a_coverage_outside_the_unit_interval_is_refused(tmp_path: Path, value: str) -> None:
    arguments = _args(tmp_path, "--min-coverage", value)
    with pytest.raises(RemapDoorRefusal, match="outside"):
        remap_door.build_argv(arguments, Path("/bin/true"))


def test_a_missing_input_is_refused_by_flag(tmp_path: Path) -> None:
    arguments = _args(tmp_path)
    arguments.from_state = tmp_path / "missing.nc"
    with pytest.raises(RemapDoorRefusal, match="--from-state"):
        remap_door.build_argv(arguments, Path("/bin/true"))


def test_a_successful_run_writes_a_door_receipt_with_the_engine_receipt_inside(tmp_path: Path, capsys) -> None:
    engine = _fake_engine(tmp_path)
    arguments = _args(tmp_path, "--remap-exe", str(engine))
    assert remap_door.run_remap(arguments) == 0
    receipt = json.loads((tmp_path / "B.state.nc.receipt.json").read_text())
    assert receipt["schema"] == remap_door.RECEIPT_SCHEMA
    assert receipt["exit_code"] == 0
    assert receipt["engine_receipt"]["status"] == "remapped"
    assert receipt["summary"]["dry_air_relative_change"] == pytest.approx(2e-6)
    assert receipt["output"]["sha256"]
    assert set(receipt["inputs"]) == {"from_grid", "from_state", "to_grid", "to_static", "to_vertical"}
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "remapped"


def test_an_engine_refusal_carries_the_coverage_report_and_still_writes_the_receipt(tmp_path: Path) -> None:
    engine = _fake_engine(tmp_path, exit_code=1, refusal="7 of 9 target cell(s) are not covered by the source")
    arguments = _args(tmp_path, "--remap-exe", str(engine))
    with pytest.raises(RemapDoorRefusal) as caught:
        remap_door.run_remap(arguments)
    assert "not covered by the source" in str(caught.value)
    receipt = json.loads((tmp_path / "B.state.nc.receipt.json").read_text())
    assert receipt["exit_code"] == 1
    assert receipt["output"] is None
    assert receipt["summary"]["cells_below_required"] == 7


def test_a_named_but_missing_engine_is_refused(tmp_path: Path) -> None:
    arguments = _args(tmp_path, "--remap-exe", str(tmp_path / "nope"))
    with pytest.raises(RemapDoorRefusal):
        remap_door.run_remap(arguments)


def test_the_engine_is_a_ladder_row_and_its_marker_is_bound_to_the_crate() -> None:
    from woof import bridges, mpas_mesh
    from woof.hex import engines

    assert engines.REMAP in engines.ENGINES
    assert mpas_mesh.BRIDGES["rw_mpas_remap"].env_var == "WOOF_RW_MPAS_REMAP"
    marker = mpas_mesh.REMAP_ABI_MARKER
    assert bridges.BRIDGE_ABI_MARKERS["rw_mpas_remap"] == marker.encode("utf-8")
    source = CRATE / "src" / "bin" / "rw_mpas_remap.rs"
    if not source.is_file():  # pragma: no cover - wheel
        pytest.skip("no checkout crate in this install")
    text = source.read_text(encoding="utf-8")
    literal = text.split('pub const ABI_MARKER: &str = "', 1)[1].split('"', 1)[0]
    literal = literal.replace("\\\n", "")
    assert literal.startswith(marker)


# ---------------------------------------------------------------------------
# one real run on two tiny generated meshes
# ---------------------------------------------------------------------------
def _built(name: str) -> Path | None:
    for profile in ("release", "debug"):
        path = REPO / "tools" / "rustwx" / "target" / profile / name
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


COORDS = {f"{p}{e}" for e in ("Cell", "Edge", "Vertex") for p in ("lat", "lon", "x", "y", "z")}
NZ = 8
ZTOP = 20000.0


def _template(grid: Path, out: Path, *, fill: bool) -> None:
    from netCDF4 import Dataset

    with Dataset(grid) as g, Dataset(out, "w", format="NETCDF3_64BIT_DATA") as d:
        for name, dim in g.dimensions.items():
            d.createDimension(name, len(dim))
        d.createDimension("Time", None)
        d.createDimension("nVertLevels", NZ)
        d.createDimension("nVertLevelsP1", NZ + 1)
        for name, var in g.variables.items():
            if var.dtype == np.dtype("S1"):
                continue
            dtype = np.float32 if var.dtype == np.float64 and name not in COORDS else var.dtype
            d.createVariable(name, dtype, var.dimensions)[...] = var[...]
        lat = np.asarray(g["latCell"][:], dtype=float)
        lon = np.asarray(g["lonCell"][:], dtype=float)
        h = 1500.0 * np.exp(-((lat - 0.8) ** 2 + (lon - 0.3) ** 2) / 0.05)
        zeta = np.linspace(0.0, ZTOP, NZ + 1)
        zgrid = zeta[None, :] + h[:, None] * (1.0 - zeta[None, :] / ZTOP)
        d.createVariable("zgrid", np.float32, ("nCells", "nVertLevelsP1"))[...] = zgrid
        zmid = 0.5 * (zgrid[:, 1:] + zgrid[:, :-1])
        n_edges = len(g.dimensions["nEdges"])
        for name, value in (("theta", 300.0 + 0.0 * zmid), ("rho", 1.2 * np.exp(-zmid / 8000.0)),
                            ("qv", 0.01 * np.exp(-zmid / 2500.0))):
            d.createVariable(name, np.float32, ("Time", "nCells", "nVertLevels"))[0, ...] = value if fill else 0 * value
        d.createVariable("u", np.float32, ("Time", "nEdges", "nVertLevels"))[0, ...] = np.zeros((n_edges, NZ))


@pytest.mark.skipif(_built("rw_mpas_remap") is None or _built("rw_mpas_mesh") is None,
                    reason="rw_mpas_remap / rw_mpas_mesh are not built in this checkout")
def test_a_real_remap_between_two_generated_meshes_conserves_and_preserves_a_constant(tmp_path: Path) -> None:
    pytest.importorskip("netCDF4")
    from netCDF4 import Dataset

    mesh = _built("rw_mpas_mesh")
    for name, km in (("A", "1500"), ("B", "1100")):
        subprocess.run([str(mesh), "--out", str(tmp_path / f"{name}.grid.nc"), "--background-km", km],
                       check=True, capture_output=True)
    _template(tmp_path / "A.grid.nc", tmp_path / "A.state.nc", fill=True)
    _template(tmp_path / "B.grid.nc", tmp_path / "B.template.nc", fill=False)
    arguments = build_parser().parse_args([
        "remap", "--from-grid", str(tmp_path / "A.grid.nc"), "--from-state", str(tmp_path / "A.state.nc"),
        "--to-grid", str(tmp_path / "B.template.nc"), "-o", str(tmp_path / "B.state.nc"),
        "--balance", "carry", "--remap-exe", str(_built("rw_mpas_remap")),
    ])
    assert remap_door.run_remap(arguments) == 0
    receipt = json.loads((tmp_path / "B.state.nc.receipt.json").read_text())["engine_receipt"]
    assert receipt["coverage"]["cells_below_required"] == 0
    assert abs(receipt["budgets"]["horizontal_stage_dry_air_relative_change"]) < 1e-9
    assert abs(receipt["budgets"]["horizontal_stage_water_vapour_relative_change"]) < 1e-9
    with Dataset(tmp_path / "B.state.nc") as d:
        theta = np.asarray(d["theta"][0])
    assert np.allclose(theta, 300.0, rtol=2e-6)
