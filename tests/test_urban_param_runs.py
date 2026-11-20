"""urban_param_file through the physics driver: the SLUCM and BEP read the
gridded morphology, and a run without it is the run it was.

* With a file, the driver's ``*_urb2d`` morphology is real.exe's split
  after urban_var_init's keep/zero, and the urban columns' answer moves;
  the rural columns' does not.
* Without one, or with one whose every urban column has ``HGT_URB2D = 0``
  (urban_var_init then zeroes it all, WRF's "USING DEFAULT URBAN
  MORPHOLOGY"), the run is byte-identical -- under ``GPUWM_NUMERICS``
  fast and strict and ``GPUWM_WRF_EXACT`` 0 and 1, each in its own child
  process (the kernels compile per process), with the UCM's unused
  morphology pointer table poisoned so a read of it would fault.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
URBAN_COLUMNS = 2          # ivgtyp 13 (MODIS ISURBAN) on the first two x


def _write_param(path, ny, nx, *, hgt=12.0, dx=None):
    import netCDF4

    block = np.zeros((132, ny, nx), np.float32)
    block[91 - 1] = 0.35                 # LP
    block[92 - 1] = hgt                  # MH
    block[93 - 1] = 5.0                  # STDH
    block[94 - 1] = hgt                  # HGT
    block[95 - 1] = 1.4                  # LB
    for k in range(4):
        block[96 - 1 + k] = 0.25 + 0.05 * k
    hi = np.zeros(15, np.float32)
    hi[1:4] = (0.3, 0.5, 0.2)
    block[118 - 1:] = hi[:, None, None]
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("num_urb_params", 132)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        ds.createVariable("URB_PARAM", "f4", ("Time", "num_urb_params",
                                              "south_north", "west_east"))[:] = block[None]
        if dx is not None:
            ds.DX = np.float32(dx)
            ds.DY = np.float32(dx)
    return str(path)


def _driver(option, param_file="", **kw):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_urban_default_off_identity import _small_driver

    return _small_driver(sf_urban_physics=option, mp_physics=8,
                         urban_param_file=param_file, **kw)


def _run(option, param_file="", steps=3, poison=False):
    import cupy as cp

    state, cfg, driver = _driver(option, param_file)
    if poison and option == 1:
        nudapt, table, _ = driver.urban._ucm_morph
        assert nudapt == 0
        table[...] = cp.uint64(0xFFFF_FFFF_FFFF_0000)
    for _ in range(steps):
        driver.compute(state, cfg)
    out = {}
    for name in sorted(driver.fields):
        value = driver.fields[name]
        if hasattr(value, "__cuda_array_interface__") or isinstance(value, np.ndarray):
            out[name] = cp.asnumpy(cp.asarray(value))
    for name in ("u", "v", "w", "theta", "qv"):
        value = getattr(state, name, None)
        if value is not None:
            out[f"state.{name}"] = cp.asnumpy(value)
    return out, driver


def _digest(arrays) -> dict[str, str]:
    return {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
            for k, v in arrays.items()}


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("option", (1, 2))
def test_the_file_reaches_the_scheme_and_moves_only_urban_columns(
        option, tmp_path):
    import cupy as cp

    state, cfg, driver = _driver(option)
    ny, nx = driver.fields["tsk"].shape
    path = _write_param(tmp_path / "geo_em.d01.nc", ny, nx, dx=cfg.dx)
    base, _ = _run(option)
    with_file, driver = _run(option, path)
    f = driver.fields
    urban = np.zeros((ny, nx), bool)
    urban[:, :URBAN_COLUMNS] = True
    host = lambda name: cp.asnumpy(f[name])
    assert np.all(host("lp_urb2d")[urban] == np.float32(0.35))
    assert np.all(host("lp_urb2d")[~urban] == 0)        # keep/zero off urban
    assert np.all(host("hgt_urb2d")[urban] == np.float32(12.0))
    if option == 1:
        assert np.all(host("mh_urb2d")[urban] == np.float32(12.0))
        assert driver.urban._ucm_morph[0] == 1
        assert not host("hi_urb2d").any() if "hi_urb2d" in f else True
    else:
        hi = host("hi_urb2d")
        assert np.all(hi[2][urban] == np.float32(0.5)) and not hi[:, ~urban].any()
    moved = with_file["tsk"].view(np.uint32) != base["tsk"].view(np.uint32)
    assert moved[urban].all() and not moved[~urban].any()


@pytest.mark.gpu
@requires_gpu
def test_a_file_on_another_grid_is_refused(tmp_path):
    from woof.core.urban_param import UrbanParamError

    state, cfg, driver = _driver(1)
    ny, nx = driver.fields["tsk"].shape
    with pytest.raises(UrbanParamError, match="mass grid"):
        _driver(1, _write_param(tmp_path / "a.nc", ny + 1, nx))
    with pytest.raises(UrbanParamError, match="DX"):
        _driver(1, _write_param(tmp_path / "b.nc", ny, nx, dx=3 * cfg.dx))


# ---------------------------------------------------------------------------
# Default byte-identity, one child process per numerics mode
# ---------------------------------------------------------------------------

def _child(option: int, zeroed_file: str) -> None:
    """Print {case: {array: sha256}} for no file, a zeroed file and the
    no-file run with the morphology table poisoned."""
    cases = {"none": _digest(_run(option)[0]),
             "zeroed": _digest(_run(option, zeroed_file)[0]),
             "poisoned": _digest(_run(option, poison=True)[0])}
    print("URBPARAM-DIGEST " + json.dumps(cases, sort_keys=True))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("numerics", ("fast", "strict"))
@pytest.mark.parametrize("exact", ("0", "1"))
@pytest.mark.parametrize("option", (1, 2))
def test_no_file_is_byte_identical_in_every_numerics_mode(
        option, numerics, exact, tmp_path):
    state, cfg, driver = _driver(option)
    ny, nx = driver.fields["tsk"].shape
    zeroed = _write_param(tmp_path / "zero.nc", ny, nx, hgt=0.0)
    env = dict(os.environ, GPUWM_NUMERICS=numerics, GPUWM_WRF_EXACT=exact,
               PYTHONPATH=os.pathsep.join(
                   [str(ROOT), str(ROOT / "tests"),
                    os.environ.get("PYTHONPATH", "")]))
    code = ("import sys; sys.path.insert(0, %r); "
            "import test_urban_param_runs as t; t._child(%d, %r)"
            % (str(ROOT / "tests"), option, zeroed))
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = [l for l in proc.stdout.splitlines()
            if l.startswith("URBPARAM-DIGEST ")][-1]
    cases = json.loads(line.split(" ", 1)[1])
    none = cases["none"]
    assert len(none) > 20
    for case in ("zeroed", "poisoned"):
        differ = sorted(k for k in none if cases[case].get(k) != none[k])
        assert not differ, f"{case}: {differ}"
        assert set(cases[case]) == set(none), case
