"""The use_theta_m = 1 / do_avgflx_em = 1 port against compiled WRF v4.7.1.

The record (tests/data/wrf471_thetam, made by
tools/bigstep_wrf471_oracle/thetam_build.sh) holds WRF's words for every
``use_theta_m`` statement woof transcribes, on a real WRF patch, and for
upd_avgflx/zero_avgflx.  :func:`port_outputs` runs woof's own kernels and
launchers on the record's inputs; :func:`compare` counts identical words.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ORACLE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "wrf471_thetam"

#: thetam_run.F90's output records, in order.  "m" = mass (nz, ny, nx);
#: "u"/"v"/"w" = the staggered avgflx fields.
RECORDS = (
    ("init_t_m", "m"),
    ("eos_hypso1_al", "m"), ("eos_hypso1_p", "m"),
    ("eos_hypso2_al", "m"), ("eos_hypso2_p", "m"),
    ("eos_dry_al", "m"), ("eos_dry_p", "m"),
    ("phy_prep_th_phy", "m"), ("phy_prep_th_phy_m_t0", "m"),
    ("phy_prep_t_phy", "m"), ("phy_prep_rho", "m"),
    ("mp_prep_th_phy", "m"), ("mp_prep_h_diabatic", "m"),
    ("mp_prep_qv_diabatic", "m"),
    ("finish_a_in_th_phy", "m"), ("finish_a_in_h_diabatic", "m"),
    ("finish_a_t_new", "m"), ("finish_a_th_phy_m_t0", "m"),
    ("finish_a_h_diabatic", "m"),
    ("finish_b_in_th_phy", "m"), ("finish_b_in_h_diabatic", "m"),
    ("finish_b_t_new", "m"), ("finish_b_th_phy_m_t0", "m"),
    ("finish_b_h_diabatic", "m"),
    ("conv_t_tendf", "m"),
    *((f"avgflx_{c}_{f}", kind) for c in ("reset", "c1", "c5", "c359")
      for f, kind in (("rum", "u"), ("rvm", "v"), ("wwm", "w"))),
)
AVGFLX_COUNTS = {"reset": -1, "c1": 1, "c5": 5, "c359": 359}
LIM = {"a": 10.0, "b": 0.07}
DT = 12.0


def load_inputs(directory: Path = ORACLE_DIR) -> dict[str, np.ndarray]:
    with np.load(directory / "thetam_inputs.npz") as z:
        return {name: z[name] for name in z.files}


def load_outputs(directory: Path = ORACLE_DIR,
                 inputs: dict | None = None) -> dict[str, np.ndarray]:
    inputs = load_inputs(directory) if inputs is None else inputs
    nz, ny, nx = inputs["t_dry"].shape
    shape = (nx + 1, nz + 1, ny + 1)
    raw = np.fromfile(directory / "thetam_outputs.bin", "<f4")
    size = int(np.prod(shape))
    if raw.size != size * len(RECORDS):
        raise ValueError(f"record holds {raw.size} words, expected "
                         f"{size * len(RECORDS)}")
    out = {}
    for n, (name, kind) in enumerate(RECORDS):
        mem = raw[n * size:(n + 1) * size].reshape(shape, order="F")
        cut = {"m": (nx, nz, ny), "u": (nx + 1, nz, ny),
               "v": (nx, nz, ny + 1), "w": (nx, nz + 1, ny)}[kind]
        out[name] = np.ascontiguousarray(
            np.transpose(mem[:cut[0], :cut[1], :cut[2]], (1, 2, 0)))
    return out


def _hybrid_differences(c3f, c4f):
    c3 = np.asarray(c3f, np.float64)
    c4 = np.asarray(c4f, np.float64)
    return ((c3[:-1] - c3[1:]).astype(np.float32),
            (c4[:-1] - c4[1:]).astype(np.float32))


def _eos(cp, inputs, t, *, theta_m: bool, hypso: int):
    from woof.core.diagnostics import update_diagnostics
    from woof.wrf_exact import DIAGNOSTICS_ENABLED
    f32 = np.float32
    nz, ny, nx = t.shape
    dc3f, dc4f = _hybrid_differences(inputs["c3f"], inputs["c4f"])
    dev = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=f32))
    state = SimpleNamespace(
        thp=dev(t), php=dev(inputs["ph"]), mup=dev(inputs["mu"]),
        thb=dev(np.full((nz, ny, nx), 300.0, f32)), phb=dev(inputs["phb"]),
        dphb_resid=dev(np.zeros((nz, ny, nx), f32)),
        alb=dev(inputs["alb"]), pb=dev(inputs["pb"]),
        rdnw=dev(inputs["rdnw"]), c1h=dev(inputs["c1h"]),
        c2h=dev(inputs["c2h"]), c3h=dev(inputs["c3h"]),
        c4h=dev(inputs["c4h"]), c3f=dev(inputs["c3f"]),
        c4f=dev(inputs["c4f"]), dc3f=dev(dc3f), dc4f=dev(dc4f),
        mub2d=dev(inputs["mub"]), qv=dev(inputs["qv"]), p_top=f32(5000.0),
        p=cp.zeros((nz, ny, nx), f32), al=cp.zeros((nz, ny, nx), f32),
        alt=cp.zeros((nz, ny, nx), f32),
        p_perturbation=cp.zeros((nz, ny, nx), f32), theta_m=theta_m)
    update_diagnostics(state, hypso)
    pressure = (state.p_perturbation if DIAGNOSTICS_ENABLED
                else state.p - state.pb)
    return cp.asnumpy(state.al), cp.asnumpy(pressure)


def port_outputs(inputs: dict, record: dict) -> dict[str, np.ndarray]:
    """woof's words for every record, from the record's own inputs (each
    stage reads WRF's output of the stage before it, so a difference is
    the stage's own)."""
    import cupy as cp
    from woof.core.kernels import get_kernel
    f32 = np.float32
    nz, ny, nx = inputs["t_dry"].shape
    n = nz * ny * nx
    tpb = 256
    blocks = ((n + tpb - 1) // tpb,)
    dev = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=f32))
    thb = dev(np.full(nz, 300.0, f32))
    base3d = np.int32(0)
    out: dict[str, np.ndarray] = {}

    # A. real.exe's theta -> theta_m.
    thp = dev(inputs["t_dry"])
    get_kernel("theta_m", "theta_m_from_dry")(
        blocks, (tpb,), (thp, dev(inputs["qv"]), thb, base3d,
                         np.int32(nz), np.int32(ny * nx)))
    out["init_t_m"] = cp.asnumpy(thp)
    t_m = record["init_t_m"]

    # B. calc_p_rho_phi (theta_m: moist = 0; the dry control: moist = 1).
    for hypso in (1, 2):
        al, p = _eos(cp, inputs, t_m, theta_m=True, hypso=hypso)
        out[f"eos_hypso{hypso}_al"], out[f"eos_hypso{hypso}_p"] = al, p
    al, p = _eos(cp, inputs, inputs["t_dry"], theta_m=False, hypso=1)
    out["eos_dry_al"], out["eos_dry_p"] = al, p

    # C/D. phy_prep / moist_physics_prep_em's dry theta.
    dry = cp.empty((nz, ny, nx), f32)
    get_kernel("theta_m", "theta_dry_from_moist")(
        blocks, (tpb,), (dev(t_m), dev(inputs["qv"]), thb, base3d,
                         np.int32(nz), np.int32(ny * nx), dry))
    dry_h = cp.asnumpy(dry)
    # woof's physics reads thb + thp, its save_pre_mp_theta the same sum.
    full = (f32(300.0) + dry_h).astype(f32)
    out["phy_prep_th_phy_m_t0"] = dry_h
    out["phy_prep_th_phy"] = full
    out["mp_prep_th_phy"] = full
    out["mp_prep_h_diabatic"] = full
    out["mp_prep_qv_diabatic"] = inputs["qv"].astype(f32)

    # E. moist_physics_finish_em, theta_m branch.
    for case, lim in LIM.items():
        h = dev(record[f"finish_{case}_in_h_diabatic"])
        t_new = cp.empty((nz, ny, nx), f32)
        th0 = cp.empty((nz, ny, nx), f32)
        lim32 = f32(f32(lim) * f32(DT))
        get_kernel("theta_m", "moist_physics_finish_theta_m")(
            blocks, (tpb,),
            (dev(record[f"finish_{case}_in_th_phy"]), h, dev(inputs["qv"]),
             dev(inputs["qv_post"]), thb, base3d, lim32, f32(DT),
             np.int32(nz), np.int32(ny * nx), t_new, th0))
        out[f"finish_{case}_t_new"] = cp.asnumpy(t_new)
        out[f"finish_{case}_th_phy_m_t0"] = cp.asnumpy(th0)
        out[f"finish_{case}_h_diabatic"] = cp.asnumpy(h)

    # F. conv_t_tendf_to_moist.
    rt = cp.empty((nz, ny, nx), f32)
    get_kernel("theta_m", "conv_t_tendf_to_moist")(
        blocks, (tpb,),
        (dev(inputs["t_tendf"]), dev(inputs["moist_tend"]),
         dev(inputs["qv"]), dev(t_m), thb, base3d, dev(inputs["t_tendf"]),
         np.int32(0), np.int32(nz), np.int32(ny * nx), rt))
    out["conv_t_tendf"] = cp.asnumpy(rt)

    # G. upd_avgflx (the reset is woof's pending count -1).
    kernel = get_kernel("avgflx", "upd_avgflx")
    for case, count in AVGFLX_COUNTS.items():
        for field, flux in (("rum", "ru_m"), ("rvm", "rv_m"), ("wwm", "ww_m")):
            avg = dev(inputs["avg_" + field])
            c = cp.asarray(np.array([count], np.int32))
            size = avg.size
            kernel(((size + tpb - 1) // tpb,), (tpb,),
                   (avg, dev(inputs[flux]), c, np.int64(size)))
            out[f"avgflx_{case}_{field}"] = cp.asnumpy(avg)
    return out


def compare(port: dict, record: dict) -> dict[str, dict[str, int]]:
    """Per record: words compared and words not bit-identical."""
    result = {}
    for name, _kind in RECORDS:
        if name not in port or name.startswith("finish_") and "_in_" in name:
            continue
        a = np.ascontiguousarray(port[name], np.float32).view(np.uint32)
        b = np.ascontiguousarray(record[name], np.float32).view(np.uint32)
        if a.shape != b.shape:
            raise ValueError(f"{name}: {a.shape} vs {b.shape}")
        result[name] = {"words": int(a.size), "differ": int((a != b).sum())}
    return result


def record_sha256(directory: Path = ORACLE_DIR) -> dict[str, str]:
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("thetam_inputs.npz", "thetam_outputs.bin")}
