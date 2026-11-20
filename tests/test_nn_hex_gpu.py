"""woof.nn P2.4: the hex NN column seat in real forecasts (opt-in, slow, CUDA or Metal).

Short ``woof hex regional`` forecasts on the MPAS route of a uniform and a
variable-resolution mesh, each run in its own child process, held to:

* a tapped forecast is byte-identical to the plain one (every variable of
  every history frame), under fast and under strict numerics, and its
  record holds interior columns only;
* a residual model that returns zero leaves the forecast byte-identical;
* a constant dry-theta residual heats the interior's model levels it spans
  by about rate x time, and leaves the levels above it near the plain run;
* a run with analysis nudging that taps the nudging's increment
  (``tap_target = "nudging"``) is byte-identical to the nudged run without
  ``--nn``, under fast and under strict numerics (the analyses are the
  case's own initial state, valid at the start and at the end).

Opt in with ``WOOF_HEX_VARRES_CASE=<work dir>`` naming cases prepared by
``python tools/hex_varres.py prep NAME --work-dir DIR`` (mesh and
boundaries are generated there, never committed).  ``WOOF_NN_HEX_CASES``
names them (default ``edi-u12,edi-vr``: one uniform, one variable-resolution),
``WOOF_NN_HEX_HOURS`` the length (default 1).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import warnings

import numpy as np
import pytest

ROOT = os.environ.get("WOOF_HEX_VARRES_CASE")
if not ROOT:
    pytest.skip("opt-in: set WOOF_HEX_VARRES_CASE to a work dir prepared by tools/hex_varres.py prep",
                allow_module_level=True)
pytest.importorskip("zarr")
netCDF4 = pytest.importorskip("netCDF4")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import hex_varres as vr  # noqa: E402

pytestmark = pytest.mark.slow
CASES = tuple(os.environ.get("WOOF_NN_HEX_CASES", "edi-u12,edi-vr").split(","))
HOURS = float(os.environ.get("WOOF_NN_HEX_HOURS", "1"))
LEVELS = 10
RATE = 1.0e-4  # K s-1
#: The prepared cases carry no aerosol init fields, so the GFS cold start's
#: aerosol profile is not the default here; the suite is MPAS's own.
SUITE = "--forecast-arg=--physics-suite=convection_permitting"
_RUNS: dict = {}


def _artefact(path: Path, out_mean) -> Path:
    from woof.nn.hex_seat import OUTPUTS, VOCABULARY
    from woof.nn.manifest import FieldSpec, write_mlp_artefact

    inputs = ("theta", "qv", "p", "z", "hfx")
    ins = tuple(FieldSpec(n, VOCABULARY[n][0], LEVELS if VOCABULARY[n][1] == "3d" else 1) for n in inputs)
    outs = tuple(FieldSpec(n, OUTPUTS[n], LEVELS) for n in ("dtheta", "dqv"))
    n_in, n_out = sum(f.levels for f in ins), 2 * LEVELS
    write_mlp_artefact(path, name=path.name, inputs=ins, outputs=outs,
                       layers=[(np.zeros((4, n_in)), np.zeros(4)), (np.zeros((n_out, 4)), np.zeros(n_out))],
                       activation="tanh", input_mean=np.zeros(n_in), input_std=np.ones(n_in),
                       output_mean=np.asarray(out_mean, dtype=np.float64), output_std=np.ones(n_out),
                       input_min=np.full(n_in, -1e30), input_max=np.full(n_in, 1e30))
    return path


def _toml(path: Path, body: str) -> Path:
    path.write_text("[nn]\n" + body, encoding="utf-8")
    return path


def _run(name: str, tag: str, scratch: Path, *, nn: Path | None = None, numerics: str = "fast",
         extra: tuple = ()) -> Path:
    key = (name, tag, numerics)
    if key in _RUNS:
        return _RUNS[key]
    data = vr.load_cases()
    case = vr.case_row(name, data)
    source = Path(ROOT) / name
    work = scratch / f"{name}-{tag}-{numerics}"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(source, work, ignore=shutil.ignore_patterns("forecast", "products", "grib", "met"))
    for path in work.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        if str(source) in text:
            path.write_text(text.replace(str(source), str(work)), encoding="utf-8")
    argv = [sys.executable, "-m", "woof", "hex", "regional", *vr.case_arguments(case, Path(ROOT)),
            "--cycle", data["cycle_default"], "--source", data["source_default"], "--hours", f"{HOURS:g}",
            "--history-every-minutes", str(int(HOURS * 30)), "--work-dir", str(work),
            "--skip", "mesh", "--skip", "intermediate", "--skip", "init", "--skip", "lbc", "--skip", "contract",
            SUITE, "--no-products"]
    if nn is not None:
        argv.append(f"--forecast-arg=--nn={nn}")
    argv += [f"--forecast-arg={value}" for value in extra]
    env = {k: v for k, v in os.environ.items() if k not in ("WOOF_NN_CONFIG", "GPUWM_WRF_EXACT")}
    env["WOOF_HEX_NUMERICS"] = numerics
    done = subprocess.run(argv, cwd=str(REPO), capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    _RUNS[key] = work
    return work


def _frames(work: Path) -> dict:
    files = sorted((work / "forecast").glob("history.*.nc"))
    assert len(files) >= 2, f"no history frames in {work / 'forecast'}"
    out = {}
    for path in files:
        with netCDF4.Dataset(str(path)) as ds:
            ds.set_auto_maskandscale(False)
            for var, value in ds.variables.items():
                out[f"{path.name}/{var}"] = np.asarray(value[...])
    return out


def _assert_identical(a: Path, b: Path, what: str) -> None:
    fa, fb = _frames(a), _frames(b)
    assert sorted(fa) == sorted(fb), what
    differ = [k for k in fa if fa[k].tobytes() != fb[k].tobytes()]
    assert not differ, f"{what}: {len(differ)} variables differ, e.g. {differ[:6]}"


def _seat_receipt(work: Path) -> dict:
    def find(node):
        if isinstance(node, dict):
            if "nn_column_seat" in node:
                return node["nn_column_seat"]
            for value in node.values():
                found = find(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = find(value)
                if found is not None:
                    return found
        return None

    receipt = None
    for name in ("native-physics-receipt.json", "forecast-receipt.json"):
        path = work / "forecast" / name
        if receipt is None and path.exists():
            receipt = find(json.loads(path.read_text(encoding="utf-8")))
    assert receipt is not None, "the forecast receipt carries no nn_column_seat"
    return receipt


def _bdy(work: Path) -> np.ndarray:
    grid = next((work / "mesh").glob("*-cull.grid.nc"))
    with netCDF4.Dataset(str(grid)) as g:
        return np.asarray(g["bdyMaskCell"][:]) if "bdyMaskCell" in g.variables else np.zeros(g.dimensions["nCells"].size)


@pytest.fixture(scope="module")
def scratch(tmp_path_factory):
    return tmp_path_factory.mktemp("nn-hex")


@pytest.mark.parametrize("numerics", ["fast", "strict"])
@pytest.mark.parametrize("name", CASES)
def test_a_tapped_forecast_is_byte_identical(name, numerics, scratch):
    from woof.nn.tap import read_tap

    base = _run(name, "base", scratch, numerics=numerics)
    store = scratch / f"{name}-{numerics}.zarr"
    nn = _toml(scratch / f"tap-{name}-{numerics}.toml", f'mode = "tap"\nscheme = "total"\ntap = "{store}"\n'
                                                         "tap_columns = 64\n")
    tapped = _run(name, "tap", scratch, nn=nn, numerics=numerics)
    _assert_identical(base, tapped, f"{name} {numerics}: the tap changed the forecast")

    summary = _seat_receipt(tapped)
    assert summary["mpas_identical"] and summary["samples"] == summary["calls"] > 0
    record = read_tap(store)
    assert record["attrs"]["route"] == "hex" and record["inputs"].shape[0] == summary["samples"]
    assert np.isfinite(record["inputs"]).all() and np.isfinite(record["targets"]).all()
    assert np.abs(record["targets"]).max() > 0.0, "the schemes held no physics source"
    # The surface inputs are read live: a stale host read once recorded
    # hfx/qfx/ust at their initial values for the whole run.
    offset = 0
    for spec in record["inputs_spec"]:
        rows = record["inputs"][:, offset:offset + spec.levels]
        offset += spec.levels
        if spec.name in ("ust", "hfx", "p", "theta"):
            # The same cell drawn at two times: its value must change somewhere.
            seen, changed, revisited = {}, False, False
            for sample in range(rows.shape[0]):
                for column, cell in enumerate(np.asarray(record["column_index"])[sample]):
                    value = float(rows[sample, 0, column])
                    revisited |= int(cell) in seen
                    changed |= int(cell) in seen and seen[int(cell)] != value
                    seen[int(cell)] = value
            if not revisited:  # a run too short to draw any cell twice cannot show the change
                warnings.warn(f"{name} {numerics}: no recorded cell repeats; the live-read check of "
                              f"{spec.name} was skipped")
                continue
            assert changed, f"{spec.name} never changes in time at any recorded cell"
    bdy = _bdy(tapped)
    cells = np.asarray(record["column_index"]).ravel()
    assert cells.max() < bdy.size and np.all(bdy[cells] == 0), "a tap column left the interior"


@pytest.mark.parametrize("name", CASES)
def test_residual_models_on_the_held_source(name, scratch):
    base = _run(name, "base", scratch)
    zero = _toml(scratch / f"zero-{name}.toml",
                 f'mode = "residual"\nscheme = "total"\nmodel = "{_artefact(scratch / "zero", np.zeros(2 * LEVELS))}"\n'
                 'backend = "numpy"\n')
    _assert_identical(base, _run(name, "zero", scratch, nn=zero), f"{name}: a zero residual changed the forecast")

    heat = _artefact(scratch / "heat", np.r_[np.full(LEVELS, RATE), np.zeros(LEVELS)])
    const = _toml(scratch / f"heat-{name}.toml",
                  f'mode = "residual"\nscheme = "total"\nmodel = "{heat}"\nbackend = "numpy"\n'
                  "limit_outputs = false\npositivity = false\n")
    heated = _run(name, "heat", scratch, nn=const)
    summary = _seat_receipt(heated)
    assert not summary["mpas_identical"] and summary["columns_out_of_range"] == 0
    last = sorted((base / "forecast").glob("history.*.nc"))[-1].name
    with netCDF4.Dataset(str(base / "forecast" / last)) as a, \
            netCDF4.Dataset(str(heated / "forecast" / last)) as b:
        theta_a = np.asarray(a["theta"][-1], dtype=np.float64)   # [cell, level]
        theta_b = np.asarray(b["theta"][-1], dtype=np.float64)
    interior = _bdy(heated)[:theta_a.shape[0]] == 0
    diff = (theta_b - theta_a)[interior]
    expected = RATE * HOURS * 3600.0
    low = diff[:, 1:LEVELS - 1].mean()
    high = np.abs(diff[:, LEVELS + 4:]).mean()
    print(f"{name}: mean interior theta change {low:.4f} K on the model's levels (expected {expected:.4f}), "
          f"{high:.4f} K above")
    assert 0.6 * expected < low < 1.4 * expected
    assert high < 0.25 * expected


def _nudging(name: str, scratch: Path) -> tuple:
    """--nudging arguments: grid nudging over the whole run toward the case's
    initial state, given as analyses valid at the start and at the end."""
    from datetime import datetime, timedelta

    init = Path(ROOT) / name / "init" / "init.nc"
    folder = scratch / f"analyses-{name}"
    folder.mkdir(exist_ok=True)
    start = folder / "a0.nc"
    end = folder / "a1.nc"
    shutil.copyfile(init, start)
    shutil.copyfile(init, end)
    with netCDF4.Dataset(str(end), "r+") as d:
        raw = np.asarray(d["xtime"][0]).tobytes().decode("ascii").strip("\x00 ")
        valid = datetime.strptime(raw, "%Y-%m-%d_%H:%M:%S") + timedelta(hours=HOURS)
        xtime = d["xtime"]
        xtime.set_auto_chartostring(False)
        text = valid.strftime("%Y-%m-%d_%H:%M:%S").encode("ascii").ljust(xtime.shape[1])
        xtime[0, :] = np.frombuffer(text, dtype="S1")
    config = folder / "nudging.toml"
    config.write_text(f"[nudging]\nmode = 'grid'\nguv = 0.0003\ngt = 0.0003\ngq = 0.0003\n"
                      f"gfdda_interval_m = {HOURS * 60.0:g}\ngfdda_end_h = {HOURS:g}\n", encoding="utf-8")
    return (f"--nudging={config}", f"--nudging-analysis={start}", f"--nudging-analysis={end}")


@pytest.mark.parametrize("numerics", ["fast", "strict"])
@pytest.mark.parametrize("name", CASES)
def test_tapping_the_nudging_increment_is_byte_identical(name, numerics, scratch):
    from woof.nn.tap import read_tap

    extra = _nudging(name, scratch)
    base = _run(name, "nudged", scratch, numerics=numerics, extra=extra)
    store = scratch / f"{name}-{numerics}-nudging.zarr"
    nn = _toml(scratch / f"ntap-{name}-{numerics}.toml",
               f'mode = "tap"\nscheme = "total"\ntap_target = "nudging"\ntap = "{store}"\n'
               'tap_columns = 64\noutputs = ["dtheta", "dqv"]\n')
    tapped = _run(name, "nudged-tap", scratch, nn=nn, numerics=numerics, extra=extra)
    _assert_identical(base, tapped, f"{name} {numerics}: tapping the nudging increment changed the forecast")
    record = read_tap(store)
    assert record["attrs"]["target"] == "nudging" and record["inputs"].shape[0] > 0
    assert np.isfinite(record["targets"]).all() and np.abs(record["targets"]).max() > 0.0, \
        "the nudging added no increment"
