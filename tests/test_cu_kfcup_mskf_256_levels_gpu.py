"""KF-CuP (cu_physics=10) and Multiscale KF (cu_physics=11) on 256 levels.

The refusal ceilings went 128 -> 256 (KF-CuP) and 160 -> 256 (MSKF).  Neither
kernel sizes anything by the level count: every column array lives in a
runtime-``nz`` global workspace (``KFCUP_WS_SLOTS`` / ``MSKF_*SLOTS`` slots of
``nz + 1`` / ``nz + 2`` words per lane, size_t indexing), so the bound is a
guard only.  Three claims:

1. A 256-level column built from the WRF oracle fixtures (every profile
   stretched in index space, layer depths rescaled so the column keeps its
   height) runs through the shipped launcher, is finite and non-trivial, and
   the column water budget closes.
2. Nothing moved at the existing level counts: at nz = 49 the fixture words
   still equal WRF's, and for MSKF the shipped kernel at nz in {49, 160}
   equals the same source with the old ``#define MSKF_KMAX 160`` as raw bits
   (and the old image refuses 256 with status bit 64).  KF-CuP has no
   compile-time ceiling at all, so its identity arm is the WRF oracle grade
   at the fixture's own nz.
3. Optional CUDA-vs-Metal agreement: with ``WOOF_CU256_XBACKEND_DIR`` set each
   backend's 256-level outputs are written there and compared when the other
   backend's file is present.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

pytestmark = pytest.mark.gpu

DEEP = 256
OLD_MSKF_KMAX = 160
NCOL = 8


def _stretch(a, n, axis=0):
    """Linear interpolation of ``a`` along ``axis`` onto ``n`` samples."""
    a = np.asarray(a, np.float64)
    m = a.shape[axis]
    x = np.linspace(0.0, m - 1.0, n)
    out = np.empty(a.shape[:axis] + (n,) + a.shape[axis + 1:])
    moved = np.moveaxis(a, axis, 0)
    flat = moved.reshape(m, -1)
    res = np.stack([np.interp(x, np.arange(m), flat[:, i])
                    for i in range(flat.shape[1])], axis=1)
    out = np.moveaxis(res.reshape((n,) + moved.shape[1:]), 0, axis)
    return out.astype(np.float32)


def _check_xbackend(label, result):
    directory = os.environ.get("WOOF_CU256_XBACKEND_DIR")
    if not directory:
        return
    from woof import backend

    os.makedirs(directory, exist_ok=True)
    mine = backend.active()
    np.savez(os.path.join(directory, f"{label}.{mine}.npz"), **result)
    other = "cuda" if mine != "cuda" else "metal"
    path = os.path.join(directory, f"{label}.{other}.npz")
    if not os.path.exists(path):
        return
    with np.load(path) as theirs:
        for name in sorted(result):
            a, b = result[name], theirs[name]
            if a.dtype.kind != "f":
                continue
            scale = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-30)
            err = float(np.abs(a.astype(np.float64) - b).max()) / scale
            assert err < 5e-3, f"{label}/{name}: CUDA vs Metal {err:.3e}"


def _finite(out):
    for name, value in out.items():
        if value.dtype.kind == "f":
            assert np.isfinite(value).all(), f"{name} has non-finite words"


# ---------------------------------------------------------------------------
# Multiscale Kain-Fritsch
# ---------------------------------------------------------------------------

def _mskf_case(rank=0):
    """The ``rank``-th fixture case by convective heating, aerosol off."""
    from woof.verify.mskf_oracle import FIELDS_3D, load_fixture
    fx = load_fixture()
    nz = fx.nz
    ih = FIELDS_3D.index("rthcuten")
    cases = [i for i, c in enumerate(fx.meta["cases"]) if c["aercu_opt"] == 0]
    cases.sort(key=lambda i: -float(np.abs(fx.out[i, ih * nz:(ih + 1) * nz]).sum()))
    return fx, cases[rank]


def _mskf_run(nz, ncol=NCOL, rank=0):
    """The fixture column stretched to ``nz`` levels, through launch_mskf."""
    import cupy as cp
    from woof.core import mskf
    from woof.verify.mskf_oracle import SENTINEL, case_columns

    fx, c = _mskf_case(rank)
    case = case_columns(fx, c)
    cols, sc, ci = case["cols"], case["scalars"], case["ints"]
    n0 = fx.nz
    ratio = n0 / nz

    def prof(name):
        a = _stretch(cols[name], nz)
        if name == "dz8w":
            a = (a * np.float32(ratio)).astype(np.float32)
        return a

    def dev3(a):
        return cp.asarray(np.ascontiguousarray(
            np.broadcast_to(a[:, None, None], (nz, 1, ncol)), dtype=np.float32))

    inputs = {n: dev3(prof(n)) for n in
              ("u", "v", "t", "qv", "p", "rho", "pi", "dz8w", "w0avg")}
    for n in ("hfx", "ust", "pblh", "zol"):
        inputs[n] = cp.full((1, ncol), case["scalars_f32"][n], dtype=cp.float32)
    w = cp.asarray(np.ascontiguousarray(np.broadcast_to(
        _stretch(cols["w"], nz + 1)[:, None, None], (nz + 1, 1, ncol)),
        dtype=np.float32))
    mskf.update_w0avg(inputs["w0avg"], w, dt=sc["dt"], cudt_minutes=sc["cudt"],
                      stepcu=ci["stepcu"], adaptive=bool(ci["adapt_step"]))
    outputs = {n: cp.full((nz, 1, ncol), SENTINEL, dtype=cp.float32)
               for n in mskf.OUTPUT_3D}
    outputs.update({n: cp.full((1, ncol), SENTINEL, dtype=cp.float32)
                    for n in mskf.OUTPUT_2D})
    outputs["nca"][...] = np.float32(sc["nca"])
    outputs["cu_act"] = cp.zeros((1, ncol), dtype=cp.int32)
    phase = mskf.MskfPhase(bool(ci["warm_rain"]), bool(ci["f_qr"]),
                           bool(ci["f_qi"]), bool(ci["f_qs"]))
    status = mskf.launch_mskf(inputs, outputs, dx=sc["dx"], dt=sc["dt"],
                              cudt_minutes=sc["cudt"], phase=phase,
                              kf_edrates=ci["kf_edrates"])
    res = {n: cp.asnumpy(v).copy() for n, v in outputs.items()}
    res["w0avg"] = cp.asnumpy(inputs["w0avg"]).copy()
    res["rho"] = cp.asnumpy(inputs["rho"]).copy()
    res["dz8w"] = cp.asnumpy(inputs["dz8w"]).copy()
    return status, res


def test_mskf_contract_and_message_agree():
    from woof.core import mskf
    from woof.physics_vertical_contract import MSKF_VERTICAL_LEVEL_BOUNDS
    assert MSKF_VERTICAL_LEVEL_BOUNDS == (8, DEEP)
    assert mskf.VERTICAL_LEVEL_BOUNDS == (8, DEEP)
    assert "256" in mskf.MSKF_STATUS_MESSAGES[64]
    src = open(os.path.join(_ROOT, "woof/core/kernels/mskf.cu")).read()
    assert f"#define MSKF_KMAX {DEEP}\n" in src


def _mskf_triggering_rank():
    """The first fixture case whose column triggers (cu_act) at the
    fixture's own nz.  The two strongest-heating cases carry their
    tendencies from an earlier trigger (NCA still running) and do not
    trigger on this call at any level count, so they cannot show that a
    256-level column convects."""
    from woof.verify.mskf_oracle import load_fixture
    nz = load_fixture().nz
    for rank in range(8):
        _, out = _mskf_run(nz, rank=rank)
        if out["cu_act"].sum() > 0:
            return rank
    pytest.fail("no MSKF fixture case triggers at the fixture's nz")


def test_mskf_runs_a_256_level_column():
    status, out = _mskf_run(DEEP, rank=_mskf_triggering_rank())
    assert status == 0
    _finite(out)
    assert out["cu_act"].sum() > 0
    assert np.any(out["rthcuten"] != 0) and np.any(out["rqvcuten"] != 0)
    # column water: sum(rho dz (dqv + dqc + dqr + dqi + dqs)) + rain == 0
    rhodz = out["rho"] * out["dz8w"]
    net = sum((rhodz * out[n]).sum(axis=0) for n in
              ("rqvcuten", "rqccuten", "rqrcuten", "rqicuten", "rqscuten"))
    gross = sum((rhodz * np.abs(out[n])).sum(axis=0) for n in
                ("rqvcuten", "rqccuten", "rqrcuten", "rqicuten", "rqscuten"))
    resid = np.abs(net + out["pratec"]).max()
    assert resid <= 1e-4 * max(float(gross.max()), 1e-30), (resid, gross.max())
    _check_xbackend("mskf", out)


@pytest.mark.parametrize("nz", [49, OLD_MSKF_KMAX])
def test_mskf_is_bit_identical_to_the_160_level_kernel(nz, monkeypatch):
    from woof.core import kernels
    from woof.core.kernels import no_ftz
    import cupy as cp

    shipped_status, shipped = _mskf_run(nz)
    source, options = no_ftz.module_recipe_no_ftz("mskf", (("GPUWM_MSKF_NO_AEROSOL", 1),))
    new = f"#define MSKF_KMAX {DEEP}"
    assert source.count(new) == 1
    old = cp.RawModule(code=source.replace(new, f"#define MSKF_KMAX {OLD_MSKF_KMAX}"),
                       options=tuple(options))
    served = []

    def get_kernel(name, symbol, defines=()):
        served.append(symbol)
        return old.get_function(symbol)

    monkeypatch.setattr(no_ftz, "get_kernel_no_ftz", get_kernel)
    before_status, before = _mskf_run(nz)
    assert served and shipped_status == before_status == 0
    assert np.any(shipped["rthcuten"] != 0)
    for name in sorted(shipped):
        assert shipped[name].tobytes() == before[name].tobytes(), \
            f"{name} moved: MSKF_KMAX 160 -> 256 at nz={nz}"


def test_mskf_old_image_refuses_256(monkeypatch):
    from woof.core.kernels import no_ftz
    import cupy as cp

    source, options = no_ftz.module_recipe_no_ftz("mskf", (("GPUWM_MSKF_NO_AEROSOL", 1),))
    old = cp.RawModule(code=source.replace(f"#define MSKF_KMAX {DEEP}",
                                           f"#define MSKF_KMAX {OLD_MSKF_KMAX}"),
                       options=tuple(options))
    monkeypatch.setattr(no_ftz, "get_kernel_no_ftz",
                        lambda name, symbol, defines=(): old.get_function(symbol))
    # launch_mskf returns the OR of the per-column status bits; bit 64 is
    # the MSKF_KMAX refusal, whose message names the new ceiling.
    status, _ = _mskf_run(DEEP)
    assert status & 64


# ---------------------------------------------------------------------------
# KF-CuP
# ---------------------------------------------------------------------------

def _cup_run(nz, cfg_index=None, ncol=NCOL):
    import cupy as cp
    from woof.core.kf import launch_kfcup_columns
    from woof.verify.kfcup_oracle import (configs, load_capture,
                                          pack_column_inputs, unpack_outputs)

    cfg = configs()[0] if cfg_index is None else next(
        c for c in configs() if c.index == cfg_index)
    cap = load_capture(cfg.index)
    b = pack_column_inputs(cap)
    n0 = cap.nz
    if nz != n0:
        lev = _stretch(b["lev_in"], nz, axis=1)
        names = __import__("woof.core.kf", fromlist=["x"]).KFCUP_LEVEL_INPUTS
        for i, name in enumerate(names):
            if name == "dz8w":
                lev[i] = lev[i] * np.float32(n0 / nz)
        b["lev_in"] = np.ascontiguousarray(lev)
        b["kpbl"] = np.maximum(
            1, np.rint(b["kpbl"] * (nz / n0))).astype(np.int32)
        b["lev_out"] = np.ascontiguousarray(
            _stretch(b["lev_out"], nz, axis=1))
        b["lev_out"][...] = b["lev_out"][:, :1, :1]  # constant WRF seed
    pick = np.arange(ncol) % cap.ncol
    for key in ("lev_in", "lev_out"):
        b[key] = np.ascontiguousarray(b[key][:, :, pick])
    for key in ("sca_in", "held", "iheld"):
        b[key] = np.ascontiguousarray(b[key][:, pick])
    b["kpbl"] = np.ascontiguousarray(b["kpbl"][pick])
    dev = {k: cp.asarray(v) for k, v in b.items()}
    launch_kfcup_columns(
        dev, dt=cfg.dt, dx=cfg.dx, ntst=cfg.stepcu, numbins=cfg.numbins,
        thbinsize=cfg.thbinsize, rbinsize=cfg.rbinsize,
        mindeepfreq=cfg.mindeepfreq, minshallowfreq=cfg.minshallowfreq,
        flags=cfg.flags)
    got = unpack_outputs({k: v.get() for k, v in dev.items()})
    return cap, b, got


def test_kfcup_contract_agrees():
    from woof.physics_vertical_contract import KFCUP_VERTICAL_LEVEL_BOUNDS
    assert KFCUP_VERTICAL_LEVEL_BOUNDS == (8, DEEP)


def _cup_budget(cfg, b, got):
    """Per-column water budget in the packed words: dz-weighted vapour and
    condensate tendency plus the surface rain rate (kg m-2 s-1), with the
    column's gross tendency for scale.  WRF's KF-CuP sets RAINCV = DT*PRATEC
    with DT the model step (module_cu_kfcup.F; woof.core.kf's driver hands
    back pratec = raincv / dt), so the rate is raincv / dt."""
    names = __import__("woof.core.kf", fromlist=["x"]).KFCUP_LEVEL_INPUTS
    rho = b["lev_in"][names.index("rho")].astype(np.float64)
    dz = b["lev_in"][names.index("dz8w")].astype(np.float64)
    q = [n for n in ("rqvcuten", "rqccuten", "rqrcuten", "rqicuten", "rqscuten")
         if n in got]
    net = sum((rho * dz * got[n]).sum(axis=0) for n in q)
    gross = sum((rho * dz * np.abs(got[n])).sum(axis=0) for n in q)
    return net + got["raincv"] / np.float64(cfg.dt), gross


def test_kfcup_runs_a_256_level_column():
    """Every configuration runs finite at 256 levels; every one that
    convects closes its water budget as well as it does at the fixture's
    own nz (some configurations carry a small budget term of their own,
    the same at both level counts, so the 256-level residual is held to
    the fixture's plus 1e-3 of the column's gross tendency)."""
    from woof.verify.kfcup_oracle import configs
    ran = 0
    for cfg in configs():
        cap, b, got = _cup_run(DEEP, cfg.index)
        _finite(got)
        if not np.any(got["rthcuten"] != 0):
            continue
        ran += 1
        resid, gross = _cup_budget(cfg, b, got)
        _, b0, got0 = _cup_run(cap.nz, cfg.index)
        resid0, gross0 = _cup_budget(cfg, b0, got0)
        live = gross > 0
        bound = np.abs(resid0) * 1.05 + 1e-3 * np.maximum(gross, gross0)
        assert (np.abs(resid) <= bound)[live].all(), (cfg.name, resid, resid0, gross)
        _check_xbackend(f"kfcup{cfg.index}", got)
    assert ran, "no KF-CuP configuration convected on the stretched column"


def test_kfcup_fixture_nz_still_equals_wrf():
    if os.environ.get("GPUWM_WRF_EXACT", "0") != "1":
        # The WRF word grade is the strict-replay claim, as in
        # tests/test_kfcup_runtime.py: production numerics differ by rounding.
        pytest.skip("word-grade proof selects GPUWM_WRF_EXACT=1")
    from woof.verify.kfcup_oracle import differing_words, expected_outputs
    # the fixture's own nz, every column, against WRF's words
    import cupy as cp
    from woof.core.kf import launch_kfcup_columns
    from woof.verify.kfcup_oracle import (configs, load_capture,
                                          pack_column_inputs, unpack_outputs)
    cfg = configs()[0]
    cap = load_capture(cfg.index)
    b = {k: cp.asarray(v) for k, v in pack_column_inputs(cap).items()}
    launch_kfcup_columns(b, dt=cfg.dt, dx=cfg.dx, ntst=cfg.stepcu,
                         numbins=cfg.numbins, thbinsize=cfg.thbinsize,
                         rbinsize=cfg.rbinsize, mindeepfreq=cfg.mindeepfreq,
                         minshallowfreq=cfg.minshallowfreq, flags=cfg.flags)
    diff = differing_words(unpack_outputs({k: v.get() for k, v in b.items()}),
                           expected_outputs(cap))
    assert not any(diff.values()), diff
