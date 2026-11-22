"""GSFC shortwave (ra_sw_physics=2) on 256-level columns.

``kernels/gsfcsw.cu`` and ``gsfcsw_fast.cuh`` size every per-level array of
their per-column arena from ``GSFC_KMAX``.  The source's own ``#ifndef``
default (128) is the shipped tier, so a column of at most 128 layers
compiles the unchanged text; a 129 to 256 layer column adds
``-D GSFC_KMAX=256`` (woof.core.gsfcsw_levels).  Claims:

1. The ladder, the contract bound, the arena formula and the kernel source's
   own ``GSFC_WORKSPACE_WORDS`` expression agree (no GPU).
2. A 256-level column to a 1 hPa top runs through the shipped door: finite,
   physically bounded heating and fluxes, and the column's net top minus
   net surface flux equals its absorption.  WRF's oracle harness is not
   available at 256 levels, so this is conservation, not a word comparison.
3. At nz in {49, 128} the shipped door takes the unchanged-source tier, and
   forcing the 256 tier onto the same column reproduces every output word
   (raw bits).
4. Optional CUDA-vs-Metal agreement: with ``WOOF_GSFC256_XBACKEND_DIR`` set
   each backend's 256-level outputs are written there and compared with the
   other backend's file when present.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_physics_256_levels_gpu import _assert_bitwise, _atmosphere  # noqa: E402

DEEP = 256
SHIPPED = 128
NCOL = 8
SOLCON = 1361.0
CP, G = 1004.5, 9.81
_KERNELS = Path(_HERE).parent / "woof/core/kernels"


def test_the_tier_ladder_contract_and_arena_formula_agree():
    from woof.core import gsfcsw_levels as lv
    from woof.physics_vertical_contract import GSFCSW_VERTICAL_LEVEL_BOUNDS

    assert [lv.kmax_tier(n) for n in (2, 49, 128, 129, 256)] == [
        128, 128, 128, 256, 256]
    with pytest.raises(ValueError):
        lv.kmax_tier(257)
    assert GSFCSW_VERTICAL_LEVEL_BOUNDS == (2, DEEP)
    assert lv.workspace_words(128) == 32677
    assert lv.workspace_words(129) == lv.workspace_words(256) == 62885
    for name in ("gsfcsw.cu", "gsfcsw_fast.cuh"):
        text = (_KERNELS / name).read_text()
        assert "#ifndef GSFC_KMAX\n#define GSFC_KMAX 128\n#endif" in text
        words = re.search(r"#define GSFC_WORKSPACE_WORDS \((\d+)\*GSFC_NP\+(\d+)\)",
                          text)
        assert (int(words[1]), int(words[2])) == (
            lv.WORKSPACE_WORDS_PER_INTERFACE, lv.WORKSPACE_WORDS_FIXED)
        assert "float zero[GSFC_NP]" in text


def test_the_door_bound_and_the_priced_arena_are_the_tier_arena():
    from woof.core.gsfcsw import MAX_LAYERS, workspace_bytes

    assert MAX_LAYERS == DEEP
    assert workspace_bytes(4096, 128) == 4096 * 32677 * 4
    assert workspace_bytes(4096, 256) == 4096 * 62885 * 4


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

def _inputs(nz):
    atm = _atmosphere(nz, NCOL)
    rng = np.random.default_rng(300 + nz)

    def lev(name):
        return np.ascontiguousarray(atm[name][:, 0, :].T)

    p = lev("pressure")
    qc = np.where((p < 85000.0) & (p > 70000.0), 2.0e-4, 0.0)
    qi = np.where((p < 35000.0) & (p > 25000.0), 5.0e-5, 0.0)
    cloud = ((qc > 0) | (qi > 0)).astype(np.float64)
    coszen = np.array([0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0, 0.6])
    albedo = rng.uniform(0.05, 0.6, NCOL)
    data = dict(temperature=lev("temperature"), pressure=p,
                pressure_interface=np.ascontiguousarray(
                    atm["p_interface"][:, 0, :].T),
                exner=lev("exner"), qv=lev("qv"), qc=qc, qi=qi,
                cloud_fraction=cloud, coszen=coszen, albedo=albedo)
    return {k: np.asarray(v, np.float32) for k, v in data.items()}


def _run(nz):
    import cupy as cp

    from woof.core.gsfcsw import gsfcsw_columns

    data = _inputs(nz)
    args = {k: cp.asarray(v) for k, v in data.items()}
    out = gsfcsw_columns(
        args.pop("temperature"), args.pop("pressure"),
        args.pop("pressure_interface"), args.pop("exner"), args.pop("qv"),
        args.pop("qc"), args.pop("qi"), args.pop("cloud_fraction"),
        args.pop("coszen"), args.pop("albedo"), warm_rain=False, f_qi=True,
        julday=172, solcon=SOLCON, center_lat=30.0)
    result = {k: np.ascontiguousarray(cp.asnumpy(v)).copy()
              for k, v in out.items()}
    return data, result


def _check_xbackend(label, result):
    directory = os.environ.get("WOOF_GSFC256_XBACKEND_DIR")
    if not directory:
        return
    from woof import backend

    os.makedirs(directory, exist_ok=True)
    mine = backend.active()
    np.savez(os.path.join(directory, f"{label}.{mine}.npz"), **result)
    for other in sorted(n for n in os.listdir(directory)
                        if n.startswith(f"{label}.") and n.endswith(".npz")
                        and n != f"{label}.{mine}.npz"):
        with np.load(os.path.join(directory, other)) as theirs:
            for name in sorted(result):
                a, b = result[name], theirs[name]
                scale = max(float(np.abs(a).max()), float(np.abs(b).max()),
                            1e-30)
                err = float(np.abs(a.astype(np.float64) - b).max()) / scale
                assert err < 5e-3, f"{label}/{name} vs {other}: {err:.3e}"


@pytest.mark.gpu
def test_a_256_level_column_to_1_hpa_is_finite_bounded_and_conserves_energy():
    data, out = _run(DEEP)
    for name, value in out.items():
        assert np.isfinite(value).all(), name
    assert out["rthraten"].shape == (NCOL, DEEP)
    mu = data["coszen"].astype(np.float64)
    day = mu > 0
    incoming = SOLCON * mu
    # Night column: no output.
    assert not out["rthraten"][~day].any() and out["gsw"][~day].all() == 0
    # Heating (K/s) is non-negative (the scheme clamps it) and bounded; the
    # stratosphere above ~10 hPa absorbs in the ozone and O2 bands.
    heating = out["rthraten"].astype(np.float64) * data["exner"]
    assert (heating >= 0).all()
    assert (heating[day] * 86400.0 < 300.0).all()
    assert (heating[day] > 0).any(axis=1).all()
    assert (heating[day][:, -40:] > 0).any(), "no absorption near the top"
    # Fluxes: absorbed at the surface within (0, incoming); reflected at the
    # top (RSWTOA is minus the upward flux) within (-incoming, 0).
    gsw, rswtoa = out["gsw"].astype(np.float64), out["rswtoa"].astype(np.float64)
    assert (gsw[day] > 0).all() and (gsw[day] < incoming[day]).all()
    assert (rswtoa[day] < 0).all() and (rswtoa[day] > -incoming[day]).all()
    # Energy closure: net flux into the column's top minus net flux out of
    # its bottom is what the layers absorb.  The scheme's own residual on
    # this column set is 0.04-0.4% of the absorption at nz=49, up to 0.6% at
    # 128 and 0.9% at 256 (same order at every depth), so 1.5% bounds a
    # closure error without asking more than the scheme delivers at the
    # shipped depths.
    dp = -np.diff(data["pressure_interface"].astype(np.float64), axis=1)
    absorbed = (heating * CP / G * dp).sum(axis=1)
    net_top = incoming + rswtoa
    np.testing.assert_allclose((net_top - gsw)[day], absorbed[day], rtol=1.5e-2,
                               atol=1e-3)
    _check_xbackend("gsfcsw256", out)


@pytest.mark.gpu
@pytest.mark.parametrize("nz", [49, SHIPPED])
def test_the_256_tier_reproduces_the_unchanged_kernel(nz, monkeypatch):
    from woof.core import gsfcsw

    calls = []
    real = gsfcsw.kmax_tier
    monkeypatch.setattr(gsfcsw, "kmax_tier",
                        lambda n: calls.append(real(n)) or real(n))
    _, shipped = _run(nz)
    assert set(calls) == {SHIPPED}, "nz <= 128 must take the unchanged tier"
    assert np.any(shipped["rthraten"] != 0)
    monkeypatch.setattr(gsfcsw, "kmax_tier", lambda n: DEEP)
    _, deep = _run(nz)
    _assert_bitwise(deep, shipped, f"GSFC_KMAX 128 -> 256 at nz={nz}")


@pytest.mark.gpu
def test_the_deep_tier_has_no_larger_local_frame_than_priced():
    from woof.core import preflight as pf
    from woof.core.gsfcsw import gsfcsw_kernel
    from woof.core.kernels import kernel_local_size_bytes

    frame = kernel_local_size_bytes(gsfcsw_kernel(DEEP))
    assert frame <= pf.KERNEL_MAX_LOCAL_SIZE_BYTES["gsfcsw"]


def test_the_config_admits_256_layers_and_refuses_257():
    from woof.config import ConfigurationRefused, RunConfig, validate_run_config

    def cfg(nz):
        return RunConfig(nx=8, ny=8, nz=nz, dx=1000.0, dy=1000.0,
                         ztop=30000.0, dt=1.0, run_seconds=10.0,
                         ra_physics=0, ra_lw_physics=0, ra_sw_physics=2)

    try:
        validate_run_config(cfg(DEEP))
    except ConfigurationRefused as error:
        assert "GSFC" not in str(error), error
    with pytest.raises(ConfigurationRefused, match="GSFC shortwave requires 2 to 256"):
        validate_run_config(cfg(DEEP + 1))
