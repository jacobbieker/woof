"""MYJ / MYJURB column workspace: geometry, bit-identity and 256 levels.

``kernels/myjpbl.cu`` (19 column arrays) and ``kernels/myjurb.cu`` (24)
used to hold their columns in per-thread local arrays sized to a
compile-time ``MYJ_KMAX = 128``: frames of 9,232 B and 10,256-12,304 B,
each a launch-time reservation of ``(frame - 1,024) x SMs x threads/SM``,
and twice that at a 256-level bound.  Both now keep those arrays in a
global workspace their launchers size to the runtime ``nz`` and to the
columns in flight (the route ysu.cu, shinhong.cu and kf.cu took;
docs/kernel_local_memory_bounds.md, section 12).

What this file asserts, beyond the geometry gates:

* BIT-IDENTITY against the pre-cut kernels.  The pre-cut ``.cu`` is read
  from git (:data:`PRE_CUT_COMMIT`), assembled with the production
  preamble and headers and compiled beside the shipped one; every output
  word -- tendencies, TKE, EL, EXCH, PBLH/KPBL/MIXHT and the INOUT surface
  state -- is compared as a bit pattern at nz = 49, 100 and 128.
* tiling and workspace reuse do not move a word, and no slot is read
  before it is written (poisoned-workspace control);
* a 256-level column -- the end-to-end target, p_top = 1 hPa -- completes
  with finite, bounded TKE and fluxes and agrees with the float32 CPU
  authority ``woof.verify.myj_ref`` under the same bars the 30-level
  conformance test uses.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from woof.core.physics_inventory import (                       # noqa: E402
    MYJ_BLOCK, MYJURBWS_SLOTS, MYJWS_SLOTS, myj_workspace_floats,
    myjurb_workspace_floats,
)
from woof.physics_vertical_contract import (                    # noqa: E402
    MYJ_VERTICAL_LEVEL_BOUNDS,
)

_KDIR = os.path.join(_ROOT, "woof", "core", "kernels")

#: The last commit whose myjpbl.cu / myjurb.cu kept the column in the
#: per-thread frame.  The A/B gate compiles THAT source beside the shipped
#: one, so it needs the object in the local repository.
PRE_CUT_COMMIT = "7950e4171af94587f25ec3da2e279acfea0bf51e"

#: The CUDA default per-thread stack: a frame at or under it reserves
#: nothing.
DEFAULT_STACK_BYTES = 1024

F = np.float32


def _source(name: str) -> str:
    with open(os.path.join(_KDIR, f"{name}.cu"), encoding="utf-8") as fh:
        return fh.read()


def _define(name: str, macro: str) -> int:
    m = re.search(rf"^#define {macro} (\d+)$", _source(name), re.M)
    assert m, f"{name}.cu has no #define {macro}"
    return int(m.group(1))


# ---------------------------------------------------------------------------
# source-only gates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,slots_macro,lanes_macro,slots", [
    ("myjpbl", "MYJWS_SLOTS", "MYJWS_LANES", MYJWS_SLOTS),
    ("myjurb", "MYJURBWS_SLOTS", "MYJURBWS_LANES", MYJURBWS_SLOTS),
])
def test_the_python_side_mirrors_the_kernels_workspace_geometry(
        name, slots_macro, lanes_macro, slots):
    """Launcher and kernel must agree, or the kernel writes out of bounds."""
    assert _define(name, slots_macro) == slots
    assert _define(name, lanes_macro) == MYJ_BLOCK, (
        "the lane width is the launch block; the launcher must launch at "
        "exactly that block or the lanes alias")


@pytest.mark.parametrize("name,at,slots", [
    ("myjpbl", "MYJWS_AT", MYJWS_SLOTS),
    ("myjurb", "MYJURBWS_AT", MYJURBWS_SLOTS),
])
def test_every_workspace_slot_id_is_used_once(name, at, slots):
    ids = [int(m) for m in re.findall(rf"{at}\(wsb,\s*(\d+),", _source(name))]
    assert sorted(ids) == list(range(slots)), (
        f"{name}.cu must bind slots 0..{slots - 1} exactly once, got "
        f"{sorted(ids)}")


@pytest.mark.parametrize("name", ["myjpbl", "myjurb"])
def test_no_column_array_is_left_on_the_stack(name):
    """A ``real name[MYJ_KMAX]`` in the kernel body is the regression."""
    left = re.findall(r"\breal\s+\w+\s*\[\s*MYJ_KMAX", _source(name))
    assert not left, (
        f"{name}.cu: {left} are back in the local frame, which re-arms the "
        "per-context local-memory reservation the workspace removed")


def test_the_workspace_follows_the_runtime_level_count():
    for fn, slots in ((myj_workspace_floats, MYJWS_SLOTS),
                      (myjurb_workspace_floats, MYJURBWS_SLOTS)):
        assert fn(49, MYJ_BLOCK) == slots * 50 * MYJ_BLOCK
        assert fn(256, MYJ_BLOCK) == slots * 257 * MYJ_BLOCK
        # Whole blocks: one column more than a block costs a second block.
        assert fn(49, MYJ_BLOCK + 1) == 2 * fn(49, MYJ_BLOCK)


def test_the_kernel_guards_and_launcher_ceilings_are_the_contract():
    from woof.core import myjpbl, myjurb

    assert _define("myjpbl", "MYJ_KMAX") == MYJ_VERTICAL_LEVEL_BOUNDS[1]
    assert _define("myjurb", "MYJ_KMAX") == MYJ_VERTICAL_LEVEL_BOUNDS[1]
    assert MYJ_VERTICAL_LEVEL_BOUNDS == (4, 256)
    assert (myjurb.MYJURB_MIN_COLUMN_LEVELS,
            myjurb.MYJURB_MAX_COLUMN_LEVELS) == MYJ_VERTICAL_LEVEL_BOUNDS
    assert (myjpbl.MYJ_MIN_COLUMN_LEVELS,
            myjpbl.MYJ_MAX_COLUMN_LEVELS) == MYJ_VERTICAL_LEVEL_BOUNDS


def test_preflight_prices_the_workspace_instead_of_the_frame():
    """The workspace is charged when MYJ is selected, at the tile and the
    runtime nz, and MYJURB's 24 slots when sf_urban_physics is 2 or 3; the
    retired frames are no longer charged at all."""
    import dataclasses
    from datetime import datetime

    from woof.config import RunConfig
    from woof.core import preflight as pf
    from woof.core.physics_inventory import MYJ_TILE_BLOCKS_PER_SM
    from woof.experiment import experiment_from_run_config

    assert pf.KERNEL_MAX_LOCAL_SIZE_BYTES["myjpbl"] == 0
    assert pf.KERNEL_MAX_LOCAL_SIZE_BYTES["myjurb"] == 0
    profile = pf.DeviceLocalMemoryProfile(
        name="myj-workspace-test", multiprocessor_count=2,
        max_threads_per_multiprocessor=1536)
    tile = 2 * MYJ_TILE_BLOCKS_PER_SM * MYJ_BLOCK
    cfg = RunConfig(nx=257, ny=10, nz=256, dx=1000.0, dy=1000.0,
                    ztop=48000.0, dt=1.0, run_seconds=10.0,
                    bl_pbl_physics=2, sf_sfclay_physics=2)
    start = datetime(2000, 1, 1)
    exp = experiment_from_run_config(cfg, start)
    assert pf.myj_column_workspace_bytes(exp, profile=profile) == (
        tile * MYJWS_SLOTS * 257 * 4)
    assert pf.column_workspace_bytes(exp, profile=profile) >= (
        tile * MYJWS_SLOTS * 257 * 4)
    # columns below the tile: priced at whole blocks of the columns present
    small = experiment_from_run_config(
        dataclasses.replace(cfg, nx=10, ny=5), start)
    assert pf.myj_column_workspace_bytes(small, profile=profile) == (
        2 * MYJ_BLOCK * MYJWS_SLOTS * 257 * 4)
    other = experiment_from_run_config(
        dataclasses.replace(cfg, bl_pbl_physics=1, sf_sfclay_physics=1),
        start)
    assert pf.myj_column_workspace_bytes(other, profile=profile) == 0
    # MYJURB replaces the MYJPBL call under BEP / BEP+BEM.
    run = dataclasses.replace(cfg, sf_urban_physics=2)
    urban = dataclasses.replace(exp, domains=tuple(
        dataclasses.replace(dc, run=run) for dc in exp.domains))
    assert pf.myj_column_workspace_bytes(urban, profile=profile) == (
        tile * MYJURBWS_SLOTS * 257 * 4)


# ---------------------------------------------------------------------------
# column builders
# ---------------------------------------------------------------------------
def _columns(nz, ny, nx, *, seed, ztop=None):
    """Randomised but physical bottom-up columns, one regime per column.

    Neighbouring lanes take different branches (stable / convective /
    cloudy / iced / water / snow), so warp divergence and every MIXLEN
    and PRODQ2 arm are exercised.  ``ztop`` sets the model top; by default
    it follows nz the way a real stretched grid does (~20 km at 49 levels,
    ~48 km -- p_top ~ 1 hPa -- at 256).
    """
    rng = np.random.default_rng(seed)
    n = ny * nx
    if ztop is None:
        ztop = 20000.0 if nz <= 64 else (30000.0 if nz <= 128 else 48000.0)
    dz0 = rng.uniform(15.0, 60.0, n)
    # geometric stretching that sums to ztop for every column
    ratio = np.empty(n)
    for i in range(n):
        lo, hi = 1.0, 1.2
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            total = dz0[i] * (mid ** nz - 1.0) / (mid - 1.0)
            lo, hi = (mid, hi) if total < ztop else (lo, mid)
        ratio[i] = 0.5 * (lo + hi)
    k = np.arange(nz)[:, None]
    dz = dz0[None, :] * ratio[None, :] ** k
    z = np.cumsum(dz, axis=0) - 0.5 * dz
    hpbl = rng.uniform(150.0, 2500.0, n)
    gamma_pbl = rng.choice([-0.003, 0.0, 0.002, 0.008], n) \
        + rng.uniform(-5e-4, 5e-4, n)
    th0 = rng.uniform(270.0, 305.0, n)
    th = np.where(z < hpbl, th0 + gamma_pbl * z,
                  th0 + gamma_pbl * hpbl + 0.004 * (z - hpbl))
    th = th + np.where(z > 12000.0, 0.02 * (z - 12000.0), 0.0)
    th = th + rng.normal(0.0, 0.05, th.shape)
    p = 1.0e5 * np.exp(-z / 7400.0)
    exner = (p / 1.0e5) ** (287.0 / 1004.5)
    t = th * exner
    qv = rng.uniform(2e-3, 1.6e-2, n) * np.exp(-z / 2500.0)
    qv = np.maximum(qv, 2e-6)
    cloud = (rng.uniform(0, 1, (nz, n)) < 0.08) & (z < 8000.0)
    qc = np.where(cloud, rng.uniform(1e-5, 5e-4, (nz, n)), 0.0)
    ice = (rng.uniform(0, 1, (nz, n)) < 0.08) & (z > 3000.0) & (z < 12000.0)
    qi = np.where(ice, rng.uniform(1e-6, 2e-4, (nz, n)), 0.0)
    ushear = rng.uniform(-3e-3, 6e-3, n)
    u = rng.uniform(-10.0, 15.0, n) + ushear * np.minimum(z, 12000.0)
    u = u * np.minimum(1.0, np.log1p(z / 0.1) / np.log1p(200.0 / 0.1))
    v = rng.uniform(-8.0, 8.0, n) + rng.normal(0.0, 0.5, (nz, n))
    tke = np.where(z < hpbl, rng.uniform(0.1, 2.5, (nz, n)), 0.1)
    psfc = 1.0e5 * np.exp(rng.uniform(-0.01, 0.01, n))
    xland = np.where(rng.uniform(0, 1, n) < 0.3, 2.0, 1.0)

    def c3(a):
        return np.ascontiguousarray(np.asarray(a, F).reshape(nz, ny, nx))

    def c2(a):
        return np.ascontiguousarray(np.asarray(a, F).reshape(ny, nx))

    columns = {"dz": c3(dz), "u": c3(u), "v": c3(v), "t": c3(t),
               "th": c3(th), "exner": c3(exner), "qv": c3(qv),
               "qc": c3(qc), "qi": c3(qi), "p": c3(p)}
    surface = {
        "psfc": c2(psfc), "ust": c2(rng.uniform(0.03, 0.9, n)),
        "tsk": c2(t[0] + rng.uniform(-6.0, 12.0, n)),
        "chklowq": c2(rng.choice([0.0, 0.4, 1.0], n)),
        "xland": c2(xland),
        "sice": c2(np.where(rng.uniform(0, 1, n) < 0.05, 1.0, 0.0)),
        "snow": c2(np.where(rng.uniform(0, 1, n) < 0.1,
                            rng.uniform(0.0, 0.05, n), 0.0)),
        "akhs": c2(rng.uniform(1e-3, 5e-2, n)),
        "akms": c2(rng.uniform(1e-3, 5e-2, n)),
        "elflx": c2(rng.uniform(-50.0, 450.0, n)),
        "uz0": c2(rng.uniform(-0.5, 0.5, n)),
        "vz0": c2(rng.uniform(-0.5, 0.5, n)),
    }
    state = {"thz0": c2(th[0] + rng.uniform(-2.0, 2.0, n)),
             "qz0": c2(qv[0] * rng.uniform(0.8, 1.2, n)),
             "qsfc": c2(qv[0] * rng.uniform(0.8, 1.3, n)),
             "ct": c2(np.zeros(n))}
    return columns, surface, state, c3(tke)


def _bep(host, *, seed):
    """BEP implicit/explicit coefficients in the form BEP hands MYJURB.

    Each tendency is ``a * X + b`` (urban_bep: a <= 0, the exchange with
    the walls and roofs; b = -a * X_surface), so ``b`` relaxes X toward
    a surface value near the air's own, not toward zero.
    """
    columns = host[0]
    nz, ny, nx = columns["dz"].shape
    rng = np.random.default_rng(seed + 7)
    shape = (nz, ny, nx)

    def f(lo, hi, extra=0):
        return np.ascontiguousarray(
            rng.uniform(lo, hi, (shape[0] + extra,) + shape[1:]).astype(F))

    low = (np.arange(nz) < 6)[:, None, None]
    bep = {name: f(-1e-3, 0.0) * low for name in
           ("a_u_bep", "a_v_bep", "a_t_bep", "a_q_bep", "a_e_bep")}
    bep["b_u_bep"] = f(-1e-4, 1e-4) * low
    bep["b_v_bep"] = f(-1e-4, 1e-4) * low
    bep["b_t_bep"] = -bep["a_t_bep"] * (columns["th"] + f(-3.0, 6.0))
    bep["b_q_bep"] = -bep["a_q_bep"] * (columns["qv"] * f(0.8, 1.2))
    bep["b_e_bep"] = f(0.0, 2e-3) * low
    bep["dlg_bep"] = f(0.0, 50.0)
    bep["dl_u_bep"] = f(0.0, 40.0) * low
    bep["vl_bep"] = np.where(low, f(0.6, 1.0), F(1.0)).astype(F)
    bep["sf_bep"] = np.where(np.arange(nz + 1)[:, None, None] < 6,
                             f(0.6, 1.0, 1), F(1.0)).astype(F)
    bep = {k: np.ascontiguousarray(v.astype(F)) for k, v in bep.items()}
    frc = np.ascontiguousarray(rng.uniform(0.0, 1.0, shape[1:]).astype(F))
    ht = np.ascontiguousarray(rng.uniform(0.0, 600.0, shape[1:]).astype(F))
    return bep, frc, ht


# ---------------------------------------------------------------------------
# the pre-cut kernels, compiled beside the shipped ones
# ---------------------------------------------------------------------------
def _pre_cut_function(name: str, func: str):
    import cupy as cp

    from woof.core.kernels import _extra_header_text, _preamble
    from woof.physics_params import edit_kernel_source

    try:
        text = subprocess.run(
            ["git", "-C", _ROOT, "show",
             f"{PRE_CUT_COMMIT}:woof/core/kernels/{name}.cu"],
            check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"pre-cut {name}.cu not readable from git: {exc}")
    assert "real zh[MYJ_KMAX + 1];" in text, (
        "PRE_CUT_COMMIT no longer names the local-frame kernel")
    src = _preamble() + _extra_header_text(name) + edit_kernel_source(
        name, text)
    mod = cp.RawModule(code=src, options=("-std=c++17",))
    return mod.get_function(func)


def _bits(arrays):
    import cupy as cp

    out = {}
    for name, value in arrays.items():
        host = cp.asnumpy(value)
        out[name] = host.view(np.uint32) if host.dtype == np.float32 else host
    return out


def _assert_bitwise(a, b, why):
    moved = {name: int(np.count_nonzero(a[name] != b[name])) for name in a}
    moved = {k: v for k, v in moved.items() if v}
    assert not moved, f"{why}: words moved {moved}"


def _run_myjpbl_new(host, *, flqi, dtturbl=60.0):
    import cupy as cp

    from woof.core.myjpbl import myj_pbl_step

    columns, surface, state, tke = host
    d_cols = {k: cp.asarray(v) for k, v in columns.items()}
    d_sfc = {k: cp.asarray(v) for k, v in surface.items()}
    d_state = {k: cp.asarray(v) for k, v in state.items()}
    d_tke = cp.asarray(tke)
    out = myj_pbl_step(d_cols, d_sfc, d_state, d_tke,
                       dtturbl=dtturbl, flqi=flqi)
    result = {k: out[k] for k in ("rublten", "rvblten", "rthblten",
                                  "rqvblten", "rqcblten", "rqiblten",
                                  "el_myj", "exch_h", "pblh", "kpbl",
                                  "mixht")}
    result.update(tke=d_tke, **d_state)
    return result


def _run_myjpbl_old(fn, host, *, flqi, dtturbl=60.0):
    import cupy as cp

    from woof.core.myjpbl import (_COLUMN_INPUTS, _SURFACE_INPUTS,
                                  MYJ_PBL_COLUMN_OUTPUTS, MYJ_PBL_INOUT,
                                  MYJ_PBL_SURFACE_OUTPUTS)

    columns, surface, state, tke = host
    nz, ny, nx = columns["dz"].shape
    n = ny * nx
    d_cols = [cp.asarray(columns[k]) for k in _COLUMN_INPUTS]
    d_tke = cp.asarray(tke)
    d_state = {k: cp.asarray(v) for k, v in state.items()}
    out = {k: cp.empty((nz, ny, nx), dtype=cp.float32)
           for k in MYJ_PBL_COLUMN_OUTPUTS}
    out.update(pblh=cp.empty((ny, nx), cp.float32),
               mixht=cp.empty((ny, nx), cp.float32),
               kpbl=cp.empty((ny, nx), cp.int32))
    fn(((n + 127) // 128,), (128,),
       tuple(d_cols) + (d_tke,)
       + tuple(cp.asarray(surface[k]) for k in _SURFACE_INPUTS)
       + tuple(d_state[k] for k in MYJ_PBL_INOUT)
       + tuple(out[k] for k in MYJ_PBL_COLUMN_OUTPUTS)
       + tuple(out[k] for k in MYJ_PBL_SURFACE_OUTPUTS)
       + (F(dtturbl), np.int32(bool(flqi)), np.int32(nz), np.int32(n)))
    out.update(tke=d_tke, **d_state)
    return out


def _run_myjurb_new(host, bep_host, *, flag_bep, idiff=0, dtturbl=60.0):
    import cupy as cp

    from woof.core.myjurb import MYJURB_COLUMN_OUTPUTS, launch_myjurb

    columns, surface, state, tke = host
    bep, frc, ht = bep_host
    nz, ny, nx = columns["dz"].shape
    d_cols = {k: cp.asarray(v) for k, v in columns.items() if k != "qi"}
    d_state = {k: cp.asarray(v) for k, v in state.items()}
    d_tke = cp.asarray(tke)
    outputs = {k: cp.zeros((nz, ny, nx), cp.float32)
               for k in MYJURB_COLUMN_OUTPUTS}
    outputs.update(pblh=cp.zeros((ny, nx), cp.float32),
                   mixht=cp.zeros((ny, nx), cp.float32),
                   kpbl=cp.zeros((ny, nx), cp.int32))
    launch_myjurb(d_cols, {k: cp.asarray(v) for k, v in surface.items()},
                  d_state, d_tke, dtturbl=dtturbl,
                  bep={k: cp.asarray(v) for k, v in bep.items()},
                  frc_urb2d=cp.asarray(frc), ht=cp.asarray(ht),
                  flag_bep=flag_bep, idiff=idiff, outputs=outputs)
    result = {k: outputs[k] for k in (*MYJURB_COLUMN_OUTPUTS, "pblh",
                                      "kpbl", "mixht")}
    result.update(tke=d_tke, **d_state)
    return result


def _run_myjurb_old(fn, host, bep_host, *, flag_bep, idiff=0, dtturbl=60.0):
    import cupy as cp

    from woof.core.myjurb import (_COLUMN_INPUTS, _SURFACE_INPUTS,
                                  BEP_INPUTS, MYJURB_COLUMN_OUTPUTS,
                                  MYJURB_INOUT, MYJURB_SURFACE_OUTPUTS)

    columns, surface, state, tke = host
    bep, frc, ht = bep_host
    nz, ny, nx = columns["dz"].shape
    n = ny * nx
    d_state = {k: cp.asarray(v) for k, v in state.items()}
    d_tke = cp.asarray(tke)
    out = {k: cp.zeros((nz, ny, nx), cp.float32)
           for k in MYJURB_COLUMN_OUTPUTS}
    out.update(pblh=cp.zeros((ny, nx), cp.float32),
               mixht=cp.zeros((ny, nx), cp.float32),
               kpbl=cp.zeros((ny, nx), cp.int32))
    fn(((n + 127) // 128,), (128,),
       tuple(cp.asarray(columns[k]) for k in _COLUMN_INPUTS) + (d_tke,)
       + tuple(cp.asarray(surface[k]) for k in _SURFACE_INPUTS)
       + tuple(d_state[k] for k in MYJURB_INOUT)
       + tuple(out[k] for k in MYJURB_COLUMN_OUTPUTS)
       + tuple(out[k] for k in MYJURB_SURFACE_OUTPUTS)
       + tuple(cp.asarray(bep[k]) for k in BEP_INPUTS)
       + (cp.asarray(frc), cp.asarray(ht))
       + (F(dtturbl), np.int32(bool(flag_bep)), np.int32(idiff),
          np.int32(nz), np.int32(n)))
    out.update(tke=d_tke, **d_state)
    return out


def _work_proof(result, nz):
    """The comparison must be over columns that actually did something."""
    th = np.asarray(result["rthblten"].view(np.float32))
    assert np.isfinite(th).all()
    assert np.count_nonzero(th) > th.size // 10
    kpbl = np.asarray(result["kpbl"])
    assert (kpbl >= 1).all() and (kpbl <= nz).all()
    assert len(np.unique(kpbl)) > 3, "every column took the same PBL depth"


# ---------------------------------------------------------------------------
# device gates
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.parametrize("name,func", [("myjpbl", "myjpbl_column"),
                                       ("myjurb", "myjurb_column")])
def test_the_frame_stays_under_the_default_stack(name, func):
    """The property that matters, on whatever box runs it."""
    import cupy  # noqa: F401

    from woof.core.kernels import load_module

    frame = load_module(name).get_function(func).local_size_bytes
    assert frame <= DEFAULT_STACK_BYTES, frame


@pytest.mark.gpu
@pytest.mark.parametrize("nz", [49, 100, 128])
@pytest.mark.parametrize("flqi", [True, False])
def test_myjpbl_is_bit_identical_to_the_pre_cut_kernel(nz, flqi):
    old = _pre_cut_function("myjpbl", "myjpbl_column")
    host = _columns(nz, 9, 37, seed=nz)
    a = _bits(_run_myjpbl_old(old, host, flqi=flqi))
    b = _bits(_run_myjpbl_new(host, flqi=flqi))
    _work_proof(a, nz)
    assert set(a) == set(b)
    _assert_bitwise(a, b, f"myjpbl nz={nz} flqi={flqi} moved against the "
                          "pre-cut kernel")
    # perturbed-input control: the comparison can see one ULP.
    columns, surface, state, tke = host
    th = columns["th"].copy()
    th.view(np.uint32)[nz // 3, 4, 5] += 1
    c = _bits(_run_myjpbl_new(({**columns, "th": th}, surface, state, tke),
                              flqi=flqi))
    assert any(np.count_nonzero(b[k] != c[k]) for k in b), (
        "a one-ULP input change moved nothing: the A/B harness is blind")


@pytest.mark.gpu
@pytest.mark.parametrize("nz", [49, 100, 128])
@pytest.mark.parametrize("flag_bep,idiff", [(True, 0), (False, 0), (True, 1)])
def test_myjurb_is_bit_identical_to_the_pre_cut_kernel(nz, flag_bep, idiff):
    old = _pre_cut_function("myjurb", "myjurb_column")
    host = _columns(nz, 9, 37, seed=1000 + nz)
    bep = _bep(host, seed=nz)
    a = _bits(_run_myjurb_old(old, host, bep, flag_bep=flag_bep,
                              idiff=idiff))
    b = _bits(_run_myjurb_new(host, bep, flag_bep=flag_bep, idiff=idiff))
    if idiff == 0:
        _work_proof(a, nz)
    assert set(a) == set(b)
    _assert_bitwise(a, b, f"myjurb nz={nz} flag_bep={flag_bep} "
                          f"idiff={idiff} moved against the pre-cut kernel")


@pytest.mark.gpu
@pytest.mark.parametrize("nz", [4, 49, 256])
def test_tiling_does_not_change_the_answer(nz, monkeypatch):
    """Many tiles, a partial last block, the workspace reused across them."""
    from woof.core import myjpbl, myjurb

    host = _columns(nz, 7, 23, seed=5)
    bep = _bep(host, seed=5)
    whole_pbl = _bits(_run_myjpbl_new(host, flqi=True))
    whole_urb = _bits(_run_myjurb_new(host, bep, flag_bep=True))
    monkeypatch.setattr(myjpbl, "myj_tile_columns",
                        lambda fn, ncol: MYJ_BLOCK)
    monkeypatch.setattr(myjurb, "myj_tile_columns",
                        lambda fn, ncol: MYJ_BLOCK)
    _assert_bitwise(whole_pbl, _bits(_run_myjpbl_new(host, flqi=True)),
                    "tiling moved myjpbl")
    _assert_bitwise(whole_urb, _bits(_run_myjurb_new(host, bep,
                                                     flag_bep=True)),
                    "tiling moved myjurb")


@pytest.mark.gpu
def test_the_workspace_is_free_of_residue(monkeypatch):
    """No column array may be read before it is written."""
    import cupy as cp

    host = _columns(64, 5, 19, seed=11)
    bep = _bep(host, seed=11)
    real_empty = cp.empty
    seen = {}
    for value in (-7.0e30, 3.5e30):
        def poisoned(shape, dtype=float, *a, _v=value, **kw):
            arr = real_empty(shape, dtype=dtype, *a, **kw)
            if np.dtype(dtype) == np.float32:
                arr.fill(_v)
            return arr
        monkeypatch.setattr(cp, "empty", poisoned)
        seen[value] = (_bits(_run_myjpbl_new(host, flqi=True)),
                       _bits(_run_myjurb_new(host, bep, flag_bep=True)))
        monkeypatch.setattr(cp, "empty", real_empty)
    (pa, ua), (pb, ub) = seen.values()
    _assert_bitwise(pa, pb, "myjpbl reads a workspace slot before writing it")
    _assert_bitwise(ua, ub, "myjurb reads a workspace slot before writing it")


@pytest.mark.gpu
def test_a_256_level_column_completes_with_sane_tke_and_fluxes():
    """The end-to-end target: nz = 256 up to p_top ~ 1 hPa."""
    nz = 256
    host = _columns(nz, 6, 21, seed=256)
    columns = host[0]
    assert float(columns["p"][-1].max()) < 200.0          # ~1-2 hPa top
    for name, result in (
            ("myjpbl", _run_myjpbl_new(host, flqi=True)),
            ("myjurb", _run_myjurb_new(host, _bep(host, seed=256),
                                       flag_bep=True))):
        r = {k: np.asarray(v.get()) for k, v in result.items()}
        for k, v in r.items():
            assert np.isfinite(v).all(), f"{name}: {k} non-finite at 256"
        assert (r["tke"] >= 0.1 - 1e-6).all(), f"{name}: TKE under EPSQ2/2"
        assert r["tke"].max() < 100.0, f"{name}: TKE {r['tke'].max()}"
        assert (r["pblh"] > 0.0).all() and (r["pblh"] < 6000.0).all()
        assert ((r["kpbl"] >= 1) & (r["kpbl"] <= nz)).all()
        # turbulence lives in the PBL: the top half of the column carries
        # no meaningful exchange or tendency.
        assert np.abs(r["exch_h"][nz // 2:]).max() < 1.0e-2 * max(
            np.abs(r["exch_h"]).max(), 1e-30)
        assert np.abs(r["rthblten"]).max() < 0.05          # K/s
        assert np.abs(r["rublten"]).max() < 0.1             # m/s^2
        assert np.abs(r["rthblten"][:nz // 4]).max() > 1e-6


@pytest.mark.gpu
def test_a_256_level_column_agrees_with_the_cpu_authority():
    """The 30-level conformance bar, applied to one 256-level column."""
    import cupy as cp

    from woof.verify import myj_ref

    nz = 256
    columns, surface, state, tke = _columns(nz, 1, 1, seed=4)
    result = _run_myjpbl_new((columns, surface, state, tke), flqi=True)

    def c(name):
        return columns[name].reshape(nz)

    def s(src, name):
        return F(src[name].ravel()[0])

    ref = myj_ref.np_myjpbl_column(
        dz=c("dz"), u=c("u"), v=c("v"), t=c("t"), th=c("th"),
        exner=c("exner"), qv=c("qv"), qc=c("qc"), qi=c("qi"), p=c("p"),
        psfc=s(surface, "psfc"), tke=tke.reshape(nz),
        ust=s(surface, "ust"), tsk=s(surface, "tsk"),
        qsfc=s(state, "qsfc"), chklowq=s(surface, "chklowq"),
        thz0=s(state, "thz0"), qz0=s(state, "qz0"), uz0=s(surface, "uz0"),
        vz0=s(surface, "vz0"), xland=s(surface, "xland"),
        sice=s(surface, "sice"), snow=s(surface, "snow"),
        akhs=s(surface, "akhs"), akms=s(surface, "akms"),
        elflx=s(surface, "elflx"), ct=s(state, "ct"), dtturbl=F(60.0))
    for name in ("rublten", "rvblten", "rthblten", "rqvblten", "el_myj",
                 "exch_h"):
        device = cp.asnumpy(result[name]).reshape(nz).astype(np.float64)
        host = np.asarray(ref[{"el_myj": "el"}.get(name, name)], np.float64)
        scale = max(float(np.max(np.abs(host))), 1.0e-30)
        worst = float(np.max(np.abs(device - host)))
        assert worst <= 5.0e-3 * scale, (name, worst, scale)
    assert int(cp.asnumpy(result["kpbl"]).ravel()[0]) == int(ref["kpbl"])
    assert float(cp.asnumpy(result["pblh"]).ravel()[0]) == pytest.approx(
        float(ref["pblh"]), rel=1.0e-5)
