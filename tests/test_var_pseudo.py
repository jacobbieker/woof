"""WRFDA's single-observation (pseudo) tests against the WRFDA 4.7.1 oracle (plan V1, gate GV1).

``WOOF_WRFDA471_PSEUDO_DIR`` holds the tutorial's ``be.dat`` and first guess ``fg`` and,
per case ``u``, ``v``, ``t``, ``q``, WRFDA's ``wrfvar_output`` and ``wrfda.out`` from
``tools/wrfda_oracle/run_tutorial.sh`` with ``WRFVAR15='num_pseudo=1,pseudo_x=40.4,
pseudo_y=30.6,pseudo_z=15.3,pseudo_val=V,pseudo_err=E,'`` and ``WRFVAR19='pseudo_var="X",'``
(V = E = 1 for u, v, t; q has 0.001 and 0.0005). The tutorial ``be.dat`` (cv5), no other observation.

Asserted per case: ``J_0``, ``|g_0|``, the final ``J`` and the iteration count against WRFDA's log
(reference tier: 1e-12 relative; the binary32 production tier: 1e-6), and the analysis increment of
all eight WRF fields against ``wrfvar_output``'s to one float32 ulp of the field's largest first-guess
value (the file's own precision; both tiers measure at most 0.5 ulp, and the fields a case leaves
untouched are bit-exact).
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

FIELDS = ("U", "V", "T", "QVAPOR", "MU", "P", "PH", "PSFC")
POSITION = dict(x=40.4, y=30.6, z=15.3)
CASES = {"u": (1.0, 1.0), "v": (1.0, 1.0), "t": (1.0, 1.0), "q": (0.001, 0.0005)}


def _dir() -> Path:
    d = os.environ.get("WOOF_WRFDA471_PSEUDO_DIR")
    if not d or not (Path(d) / "be.dat").is_file() or not (Path(d) / "fg").is_file():
        pytest.skip("WOOF_WRFDA471_PSEUDO_DIR with be.dat, fg and per-case wrfvar_output/wrfda.out is not set")
    return Path(d)


def _d(s: str) -> float:
    return float(s.replace("D", "E"))


def _oracle_log(path: Path) -> dict:
    text = path.read_text()
    m = re.search(r"Starting cost function:\s*(\S+), Gradient=\s*(\S+)", text)
    f = re.search(r"Final:\s+(\d+) iter, J=\s*(\S+), g=\s*(\S+)", text)
    return {"j0": _d(m.group(1)), "g0": _d(m.group(2)), "iterations": int(f.group(1)), "j_final": _d(f.group(2))}


def _file(path: Path, name: str) -> np.ndarray:
    import netCDF4

    with netCDF4.Dataset(path) as nc:
        nc.set_auto_mask(False)
        return np.asarray(nc.variables[name][0]).T.astype(np.float64)


@lru_cache(maxsize=None)
def _analysis(d: str, var: str, tier: str):
    from woof.da.var.analysis import analyse
    from woof.da.var.pseudo_wrfda import pseudo_obs

    val, err = CASES[var]
    return analyse(Path(d) / "be.dat", pseudo_obs(var, val=val, err=err, **POSITION), Path(d) / "fg", tier=tier,
                   ntmax=50, eps=0.01, orthonorm_gradient=True)


def test_pseudo_obs_geometry_is_da_setup_pseudo_obs():
    from woof.da.var.pseudo_wrfda import pseudo_obs

    t = pseudo_obs("t", 40.4, 30.6, 15.3, 1.0, 2.0)["pseudo"]
    assert (t.info["i"][0], t.info["j"][0]) == (40, 30)
    assert t.info["dx"][0] == 40.4 - 40.0 and t.info["dxm"][0] == 1.0 - (40.4 - 40.0)
    assert t.extra["zk"][0] == 15.3
    assert t.fields["t"].qc[0] == 0 and t.fields["t"].inv[0] == 1.0 and t.fields["t"].error[0] == 2.0
    for v in ("u", "v", "p", "q"):
        assert t.fields[v].qc[0] < 0
    with pytest.raises(ValueError):
        pseudo_obs("tpw", 1.0, 1.0, 1.0)


def _check(var: str, tier: str, rtol: float):
    d = _dir()
    from woof.da.var.analysis import increment_stats

    o = _oracle_log(d / var / "wrfda.out")
    r = _analysis(str(d), var, tier)
    c = r.cg[-1]
    assert c.iterations == o["iterations"]
    assert abs(c.j[0] - o["j0"]) <= rtol * o["j0"]
    assert abs(c.g[0] - o["g0"]) <= rtol * o["g0"]
    assert abs(c.j_final - o["j_final"]) <= rtol * o["j_final"]
    ulp = float(np.finfo(np.float32).eps)
    rows = []
    for name in FIELDS:
        first = _file(d / "fg", name)
        theirs = _file(d / var / "wrfvar_output", name)
        s = increment_stats(r.wrf[name] - first, theirs - first)
        worst = float(np.max(np.abs(r.wrf[name] - theirs))) / (ulp * float(np.max(np.abs(first))))
        rows.append(f"{name:7s} rms {s['rms']:.3e}  max|d| {worst:.2f} float32 ulp of max|fg|")
        assert worst <= 1.0, (name, worst)
    print(f"pseudo {var} ({tier}): J_0 {c.j[0]:.15e} final J {c.j_final:.15e} iterations {c.iterations}\n" + "\n".join(rows))


@pytest.mark.parametrize("var", sorted(CASES))
def test_pseudo_reference_tier_against_wrfda(var):
    _check(var, "reference", 1e-12)


@requires_gpu
@pytest.mark.parametrize("var", sorted(CASES))
def test_pseudo_production_tier_against_wrfda(var):
    _check(var, "production", 1e-6)
