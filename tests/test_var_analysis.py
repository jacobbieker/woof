"""The analysis step of the WRFDA 3D-Var port (plan V1.5, gate O7) and its bundle.

On the WRFDA 4.7.1 tutorial (``WOOF_WRFDA471_DIR`` with ``be.dat``,
``ob.ascii``, ``woof_xb.bin``, the first guess ``fg`` and WRFDA's
``wrfvar_output``; see docs/dev/WRFDA-3DVAR-PLAN.md section 8):

- :func:`wrftoxb` from ``fg`` is the oracle's ``xb`` dump to 1e-12 relative
  (exact for most fields; an ulp in the fields through real powers, ``log``
  and ``exp`` off glibc).
- **O7**: :func:`analyse` from the files (reference tier, the tutorial's
  namelist) then :func:`xatowrf`; per WRF field the increment (analysis
  minus ``fg``) against WRFDA's (``wrfvar_output`` minus ``fg``) has
  max |difference| under 1% of its RMS and correlation >= 0.9999.  Measured:
  max |d|/RMS <= 1.3e-4 (PSFC; the rest <= 1.2e-5), correlation >= 0.999999997.
- From the oracle's own ``xhat`` (``woof_final.bin``), rounded to the file's
  float32, all eight fields are ``wrfvar_output``'s words.
- The bundle: written through :mod:`woof.io.analysis_bundle`, its
  ``analysis.nc`` carries those words, and ``woof run --analysis``'s door
  (:func:`woof.da.analysis_door.resolve`) accepts it for a run on ``fg``.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

MANIFEST = Path(__file__).parent / "data" / "wrfda471" / "manifest.json"
FIELDS = ("U", "V", "T", "QVAPOR", "MU", "P", "PH", "PSFC")


def _dir() -> Path:
    d = os.environ.get("WOOF_WRFDA471_DIR")
    need = ("be.dat", "ob.ascii", "woof_xb.bin", "fg", "wrfvar_output")
    if not d or not all((Path(d) / n).is_file() for n in need):
        pytest.skip("WOOF_WRFDA471_DIR with be.dat, ob.ascii, woof_xb.bin, fg and wrfvar_output is not set")
    return Path(d)


@lru_cache(maxsize=1)
def _xb(d: str):
    from woof.da.var.analysis import read_first_guess, wrftoxb

    fg = read_first_guess(Path(d) / "fg")
    xb, grid = wrftoxb(fg)
    return fg, xb, grid


def _file(path: Path, name: str) -> np.ndarray:
    import netCDF4

    with netCDF4.Dataset(path) as nc:
        nc.set_auto_mask(False)
        return np.asarray(nc.variables[name][0]).T


def test_increment_stats_definition():
    from woof.da.var.analysis import increment_stats

    theirs = np.array([1.0, -1.0, 2.0, -2.0])
    s = increment_stats(theirs + np.array([0.0, 0.0, 0.0, 0.01]), theirs)
    assert s["rms"] == pytest.approx(np.sqrt(2.5))
    assert s["max_abs_diff"] == pytest.approx(0.01)
    assert s["max_over_rms"] == pytest.approx(0.01 / np.sqrt(2.5))
    assert 0.9999 < s["corr"] < 1.0
    assert increment_stats(np.zeros(3), np.zeros(3))["corr"] == 1.0


def test_wrftoxb_against_the_oracles_xb():
    d = _dir()
    from woof.da.var.oracle import read_dump

    _, xb, _ = _xb(str(d))
    dump = read_dump(d / "woof_xb.bin")
    worst = {}
    for k, o in dump.items():
        if k not in xb or k == "rh":
            continue
        o = np.asarray(o, dtype=np.float64)
        a = np.asarray(xb[k], dtype=np.float64)
        assert a.shape == o.shape, k
        scale = float(np.max(np.abs(o))) or 1.0
        worst[k] = float(np.max(np.abs(a - o))) / scale
    print("wrftoxb worst:", max(worst.items(), key=lambda kv: kv[1]))
    assert len(worst) > 20
    assert max(worst.values()) <= 1e-12, {k: v for k, v in worst.items() if v > 1e-12}


def test_zero_increment_is_the_first_guess():
    d = _dir()
    from woof.da.var.analysis import QLIMIT, xatowrf

    fg, xb, grid = _xb(str(d))
    xa = {k: np.zeros_like(xb["t"]) for k in ("u", "v", "w", "t", "p", "q", "rho", "geoh")}
    xa["psfc"] = np.zeros_like(xb["psfc"])
    w = xatowrf(grid, xb, xa)
    for name in FIELDS:
        first = _file(d / "fg", name).astype(np.float64)
        if name == "QVAPOR":
            first = np.maximum(first, QLIMIT)
        if name in ("PH", "P"):
            # Rebuilt hydrostatically / from the recomputed base state: close, not the file's words.
            assert np.max(np.abs(w[name] - first)) <= 1e-3 * np.max(np.abs(first)), name
            continue
        np.testing.assert_allclose(w[name], first, rtol=0, atol=1e-6 * float(np.max(np.abs(first))), err_msg=name)


@pytest.fixture(scope="module")
def analysis():
    d = _dir()
    from woof.da.var.analysis import analyse

    man = json.loads(MANIFEST.read_text())["o6_minimise"]
    lines = []
    r = analyse(d / "be.dat", d / "ob.ascii", d / "fg", tier="reference", ntmax=50, eps=0.01,
                orthonorm_gradient=True, write=lines.append)
    assert r.cg[-1].iterations == man["iterations"]
    return r


def test_o7_increments_against_wrfvar_output(analysis):
    d = _dir()
    from woof.da.var.analysis import increment_stats

    rows = []
    for name in FIELDS:
        first = _file(d / "fg", name).astype(np.float64)
        theirs = _file(d / "wrfvar_output", name).astype(np.float64)
        s = increment_stats(analysis.wrf[name] - first, theirs - first)
        rows.append(f"{name:7s} rms {s['rms']:.4e}  max|d|/rms {s['max_over_rms']:.2e}  corr {s['corr']:.12f}")
        assert s["max_over_rms"] < 0.01, (name, s)
        assert s["corr"] >= 0.9999, (name, s)
    print("\n".join(["O7 (analysis - fg vs wrfvar_output - fg):"] + rows))


def _from_oracle_xhat(d: Path):
    from woof.da.var import be_dat as bd
    from woof.da.var.analysis import xa_from_cv, xatowrf
    from woof.da.var.oracle import read_dump

    if not (d / "woof_final.bin").is_file():
        pytest.skip("woof_final.bin (the oracle's xhat) is not in WOOF_WRFDA471_DIR")
    fg, xb, grid = _xb(str(d))
    nx, ny, nz = xb["t"].shape
    b = bd.setup_be_regional(bd.read_be_dat(d / "be.dat"), ix=nx, jy=ny, kz=nz, ds=float(xb["ds"]))
    return fg, xatowrf(grid, xb, xa_from_cv(b, xb, read_dump(d / "woof_final.bin")["xhat"]))


def test_o7_float32_words_from_the_oracles_xhat():
    d = _dir()
    _, w = _from_oracle_xhat(d)
    for name in FIELDS:
        theirs = _file(d / "wrfvar_output", name)
        np.testing.assert_array_equal(np.asarray(w[name], dtype=theirs.dtype), theirs, err_msg=name)


def _bridges():
    from conftest import cpu_preprocess_gap, netcdf_bridge_gap, netcdf_writer_gap

    for gap in (netcdf_bridge_gap(), netcdf_writer_gap(), cpu_preprocess_gap("gpuwm_analysis_binary_f64")):
        if gap:
            pytest.skip(gap)


def test_bundle_carries_wrfdas_words_and_the_door_accepts_it(tmp_path):
    d = _dir()
    _bridges()
    import netCDF4

    from woof.da.analysis_door import resolve
    from woof.da.var.analysis import write_bundle

    if not (d / "wrfbdy_d01").is_file():
        pytest.skip("wrfbdy_d01 (the tutorial's boundaries) is not in WOOF_WRFDA471_DIR")
    fg, w = _from_oracle_xhat(d)
    bundle, _ = write_bundle(tmp_path / "bundle", fg, w, recipe={"test": "o7"}, grid_id="wrfda-tutorial")
    assert set(bundle.manifest["files"]) == {"analysis.nc"}
    with netCDF4.Dataset(tmp_path / "bundle" / "analysis.nc") as a, netCDF4.Dataset(d / "wrfvar_output") as o:
        for name in FIELDS + ("W", "MUB", "PHB"):
            np.testing.assert_array_equal(a[name][:], o[name][:], err_msg=name)
    run = tmp_path / "run"
    run.mkdir()
    (run / "wrfinput_d01").symlink_to(d / "fg")
    (run / "wrfbdy_d01").symlink_to(d / "wrfbdy_d01")
    (run / "namelist.input").write_text(
        "&time_control\n run_hours = 6,\n/\n&domains\n max_dom = 1,\n/\n&dynamics\n hypsometric_opt = 2,\n/\n")
    got = resolve({"analysis": str(tmp_path / "bundle")}, directory=run, outdir=tmp_path / "out",
                  run_seconds=21600.0)
    assert got is not None and Path(got["directory"]).is_dir()
