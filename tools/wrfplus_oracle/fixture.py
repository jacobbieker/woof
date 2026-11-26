"""Make a WRFPLUS 4.7.1 oracle fixture for one TL/AD routine.

    python tools/wrfplus_oracle/fixture.py advance_uv|advance_w|calc_coef_w|advance_mu_t|small_step|rk_stage|advect|bigstep|rhsph|coriolis|w_damp|lbc|lbc_final|diffusion|hdiff|diffusion2|vdiffusion2|kcoef2 \
        --exe /mnt/NVME2/.../wrfplus_advance_uv \
        --runner "docker run --rm -v /mnt/NVME2/woof-work:/mnt/NVME2/woof-work wrf-oracle:u2404" \
        --out tests/data/wrfplus/advance_uv.npz

The inputs are woof.grad.wrfplus.cases' seeded synthetic states (no
observation or forecast data).  This script converts them to WRF's
(i, k, j) memory layout and WRF's precomputed arguments, runs the
Fortran driver (tools/wrfplus_oracle/<routine>.F90, built by build.sh),
and stores the WRFPLUS outputs mapped back to woof's layout, with masks
of the points where WRF and woof update the same words.  The tests then
need no Fortran.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from woof.grad.wrfplus import cases  # noqa: E402
from woof.grad.wrfplus.acoustic import (CW_STATE, MT_STATE, SF_ACTIVE,  # noqa: E402
                                       SF_STATE, SI_ACTIVE, SI_STATE,
                                       UV_ACTIVE, W_STATE)

#: The configurations each advance_uv fixture holds (WRFPLUS needs the
#: terrain-following coordinate and a specified boundary, spec_zone >= 1).
UV_CASES = {
    "plain": dict(spec_zone=1),
    "lid_moist_emdiv": dict(spec_zone=1, top_lid=1, moist_cq=1, use_emdiv=1,
                            has_msf=1),
}
EMDIV = 0.01


class Layout:
    """woof (k, j, i) <-> WRF (ims:ime, kms:kme, jms:jme), ims = jms = 0,
    kms = 1: woof index i is Fortran i+1, numpy index i+1."""

    def __init__(self, nz, ny, nx):
        self.nz, self.ny, self.nx = nz, ny, nx
        self.s3 = (nx + 3, nz + 1, ny + 3)
        self.s2 = (nx + 3, ny + 3)

    def w3(self, a):
        out = np.zeros(self.s3, order="F")
        nk, nj, ni = a.shape
        out[1:1 + ni, :nk, 1:1 + nj] = np.transpose(a, (2, 0, 1))
        return out

    def w2(self, a):
        out = np.zeros(self.s2, order="F")
        nj, ni = a.shape
        out[1:1 + ni, 1:1 + nj] = a.T
        return out

    def from3(self, x, shape):
        nk, nj, ni = shape
        return np.ascontiguousarray(np.transpose(x[1:1 + ni, :nk, 1:1 + nj], (1, 2, 0)))

    def from2(self, x, shape):
        nj, ni = shape
        return np.ascontiguousarray(x[1:1 + ni, 1:1 + nj].T)


def _uv_record(geom, tr, pe, au, av, emdiv):
    """The driver's input stream (advance_uv.F90 read order)."""
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    L = Layout(nz, ny, nx)
    sm = geom.smdiv
    P = tr["p_pp"] + sm * (tr["p_pp"] - tr["p_pp_old"])
    Pd = pe["p_pp"] + sm * (pe["p_pp"] - pe["p_pp_old"])
    phb = tr["phb"] if geom.base3d else np.broadcast_to(
        tr["phb"][:, None, None], tr["php"].shape)
    php = 0.5 * (phb[:-1] + phb[1:] + tr["php"][:-1] + tr["php"][1:])
    phpd = 0.5 * (pe["php"][:-1] + pe["php"][1:])
    mut = tr["mup"] + tr["mub2d"]
    muu = 0.5 * (mut + np.roll(mut, 1, axis=1))
    muu = np.concatenate([muu, muu[:, :1]], axis=1)            # face nx = face 0
    muud = 0.5 * (pe["mup"] + np.roll(pe["mup"], 1, axis=1))
    muud = np.concatenate([muud, muud[:, :1]], axis=1)
    muv = 0.5 * (mut + np.roll(mut, 1, axis=0))
    muv = np.concatenate([muv, muv[:1]], axis=0)
    muvd = 0.5 * (pe["mup"] + np.roll(pe["mup"], 1, axis=0))
    muvd = np.concatenate([muvd, muvd[:1]], axis=0)
    one_u, one_v = np.ones_like(tr["cqu"]), np.ones_like(tr["cqv"])
    cqu = tr["cqu"] if geom.moist_cq else one_u
    cqv = tr["cqv"] if geom.moist_cq else one_v
    cqud = pe["cqu"] if geom.moist_cq else 0 * one_u
    cqvd = pe["cqv"] if geom.moist_cq else 0 * one_v
    msfu = tr["msfu"] if geom.has_msf else np.ones_like(tr["msfu"])
    msfv_inv = tr["msfv"] if geom.has_msf else np.ones_like(tr["msfv"])
    k1 = lambda a: np.concatenate([a, np.zeros(nz + 1 - a.size)])  # noqa: E731
    head = np.array([nx, ny, nz, geom.spec_zone, geom.top_lid], dtype=np.int32)
    reals = np.array([geom.rdx, geom.rdy, geom.dtau, geom.cf1, geom.cf2, geom.cf3,
                      emdiv if geom.use_emdiv else 0.0])
    a3 = [tr["u_pp"], pe["u_pp"], tr["ru_t"], pe["ru_t"], tr["v_pp"], pe["v_pp"],
          tr["rv_t"], pe["rv_t"], P, Pd, tr["pb"], tr["ph_pp"], pe["ph_pp"], php, phpd,
          tr["alt"], pe["alt"], tr["al_pp"], pe["al_pp"], cqu, cqud, cqv, cqvd, au, av]
    a2 = [tr["mu_pp"], pe["mu_pp"], muu, muud, muv, muvd, tr["mudf"], pe["mudf"],
          msfu, msfu, 1.0 / msfv_inv, msfv_inv, 1.0 / msfv_inv]
    blob = head.tobytes() + reals.astype("<f8").tobytes()
    for a in a3:
        blob += L.w3(a).astype("<f8").tobytes(order="F")
    for a in a2:
        blob += L.w2(a).astype("<f8").tobytes(order="F")
    for a in (tr["fnm"], tr["fnp"], k1(tr["rdnw"])):
        blob += np.asarray(a, dtype="<f8").tobytes()
    return blob, L


def _uv_masks(geom):
    nz, ny, nx, sz = geom.nz, geom.ny, geom.nx, geom.spec_zone
    mu = np.zeros((nz, ny, nx + 1), bool)
    mu[:, sz:ny - sz, sz:nx - sz + 1] = True
    mv = np.zeros((nz, ny + 1, nx), bool)
    mv[:, sz:ny - sz + 1, sz:nx - sz] = True
    return mu, mv


def _uv_expected(geom, out, L, tr):
    """Map the driver's outputs to woof's layout and woof's argument set."""
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    n3, n2 = int(np.prod(L.s3)), int(np.prod(L.s2))
    vals = np.frombuffer(out, dtype="<f8")
    off = [0]

    def take3():
        x = vals[off[0]:off[0] + n3].reshape(L.s3, order="F")
        off[0] += n3
        return x

    def take2():
        x = vals[off[0]:off[0] + n2].reshape(L.s2, order="F")
        off[0] += n2
        return x

    su, sv = (nz, ny, nx + 1), (nz, ny + 1, nx)
    mass, wlev, col = (nz, ny, nx), (nz + 1, ny, nx), (ny, nx)
    e = {}
    e["tl_u_pp"], e["tl_u_d"] = L.from3(take3(), su), L.from3(take3(), su)
    e["tl_v_pp"], e["tl_v_d"] = L.from3(take3(), sv), L.from3(take3(), sv)
    take3(), take3()                                              # au, av (unchanged)
    a_ru, a_rv = L.from3(take3(), su), L.from3(take3(), sv)
    a_p = L.from3(take3(), mass)
    a_ph = L.from3(take3(), wlev)
    a_php = L.from3(take3(), mass)
    a_alt, a_al = L.from3(take3(), mass), L.from3(take3(), mass)
    a_cqu, a_cqv = L.from3(take3(), su), L.from3(take3(), sv)
    a_mu, a_muu, a_muv, a_mudf = (L.from2(take2(), col), L.from2(take2(), (ny, nx + 1)),
                                  L.from2(take2(), (ny + 1, nx)), L.from2(take2(), col))
    assert off[0] == vals.size, (off[0], vals.size)
    sm = geom.smdiv
    e["ad_ru_t"], e["ad_rv_t"] = a_ru, a_rv
    e["ad_p_pp"], e["ad_p_pp_old"] = (1.0 + sm) * a_p, -sm * a_p
    e["ad_ph_pp"] = a_ph
    php_a = np.zeros(wlev)
    php_a[:-1] += 0.5 * a_php
    php_a[1:] += 0.5 * a_php
    e["ad_php"] = php_a
    e["ad_alt"], e["ad_al_pp"] = a_alt, a_al
    e["ad_cqu"] = a_cqu if geom.moist_cq else np.zeros(su)
    e["ad_cqv"] = a_cqv if geom.moist_cq else np.zeros(sv)
    e["ad_mu_pp"] = a_mu
    mup_a = np.zeros(col)
    for fi in range(nx + 1):                    # face fi reads columns fi%nx, fi-1
        mup_a[:, fi % nx] += 0.5 * a_muu[:, fi]
        mup_a[:, (fi - 1) % nx] += 0.5 * a_muu[:, fi]
    for fj in range(ny + 1):
        mup_a[fj % ny] += 0.5 * a_muv[fj]
        mup_a[(fj - 1) % ny] += 0.5 * a_muv[fj]
    e["ad_mup"] = mup_a
    e["ad_mudf"] = a_mudf if geom.use_emdiv else np.zeros(col)
    return e


def advance_uv(exe: str, runner: list[str], seed: int = 20261119) -> dict:
    out = {}
    for name, over in UV_CASES.items():
        geom = cases.uv_geometry(**over)
        if geom.use_emdiv:
            geom = cases.uv_geometry(**over, xscale=-EMDIV / geom.rdx,
                                     yscale=-EMDIV / geom.rdy)
        tr = cases.uv_trajectory(geom, seed)
        pe = cases.uv_perturbation(geom, seed + 1)
        seeds = cases.uv_perturbation(geom, seed + 2)
        blob, L = _uv_record(geom, tr, pe, seeds["u_pp"], seeds["v_pp"], EMDIV)
        with tempfile.TemporaryDirectory(dir=Path(exe).parent) as tmp:
            fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            fin.write_bytes(blob)
            subprocess.run(runner + [exe, str(fin), str(fout)], check=True)
            result = fout.read_bytes()
        exp = _uv_expected(geom, result, L, tr)
        mu, mv = _uv_masks(geom)
        prefix = f"{name}/"
        out[prefix + "geometry"] = np.array(json.dumps(dataclasses.asdict(geom)))
        for k, v in tr.items():
            out[prefix + "traj/" + k] = v
        for k, v in pe.items():
            out[prefix + "pert/" + k] = v
        out[prefix + "seed/u_pp"], out[prefix + "seed/v_pp"] = seeds["u_pp"], seeds["v_pp"]
        out[prefix + "mask/u"], out[prefix + "mask/v"] = mu, mv
        for k, v in exp.items():
            out[prefix + "wrfplus/" + k] = v
    return out


#: advance_w's fixture configurations (spec_zone >= 1, as above; c1 = 1,
#: c2 = 0; no damper).  The mapped cases pass msft as msftx and msfty.
W_CASES = {
    "plain": dict(spec_zone=1),
    "lid_moist": dict(spec_zone=1, top_lid=1, moist_cq=1, base3d=0),
    "deep_moist": dict(nz=12, ny=7, nx=6, spec_zone=1, moist_cq=1, epssm=0.3),
    "mapped": dict(spec_zone=1, has_msf=1),
    "mapped_lid_moist": dict(nz=10, ny=7, nx=6, spec_zone=1, top_lid=1, moist_cq=1,
                             base3d=0, epssm=0.3, has_msf=1),
}

#: The driver's 3-D adjoint outputs after the state's four, in order.
W_AD3 = ("rw_t", "ww_pp", "th_pp", "th_pp_old", "thp", "php", "rph_t", "c2a",
         "cqw", "alt", "a", "alpha", "gam", "u_pp", "v_pp")


def _w_record(geom, tr, pe, seeds):
    """The driver's input stream (advance_w.F90 read order)."""
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    L = Layout(nz, ny, nx)
    e = geom.epssm
    phb = tr["phb"] if geom.base3d else np.broadcast_to(
        tr["phb"][:, None, None], tr["php"].shape)
    mut = tr["mub2d"] + tr["mup"]
    muts = mut + tr["mu_pp"]
    muave = 0.5 * ((1 + e) * tr["mu_pp"] + (1 - e) * tr["mu_pp_old"])
    mutd = pe["mup"]
    mutsd = pe["mup"] + pe["mu_pp"]
    muaved = 0.5 * ((1 + e) * pe["mu_pp"] + (1 - e) * pe["mu_pp_old"])
    one = np.ones_like(tr["cqw"])
    cqw = tr["cqw"] if geom.moist_cq else one
    cqwd = pe["cqw"] if geom.moist_cq else 0 * one
    k1 = lambda a: np.concatenate([a, np.zeros(nz + 1 - a.size)])  # noqa: E731
    head = np.array([nx, ny, nz, geom.top_lid], dtype=np.int32)
    reals = np.array([geom.rdx, geom.rdy, geom.dtau, geom.cf1, geom.cf2, geom.cf3,
                      geom.epssm])
    pair = lambda n: [tr[n], pe[n]]  # noqa: E731
    a3 = (pair("w_pp") + pair("ph_pp") + pair("rw_t") + pair("ww_pp")
          + pair("th_pp") + pair("th_pp_old") + pair("thp") + pair("php") + [phb]
          + pair("rph_t") + pair("c2a") + [cqw, cqwd] + pair("alt") + pair("a")
          + pair("alpha") + pair("gam") + pair("u_pp") + pair("v_pp")
          + [seeds[n] for n in W_STATE])
    a2 = [mut, mutd, muts, mutsd, muave, muaved, tr["mu_pp"], pe["mu_pp"], tr["ht"],
          tr["msft"]]
    blob = head.tobytes() + reals.astype("<f8").tobytes()
    for a in a3:
        blob += L.w3(a).astype("<f8").tobytes(order="F")
    for a in a2:
        blob += L.w2(a).astype("<f8").tobytes(order="F")
    for a in (tr["fnm"], tr["fnp"], k1(tr["rdnw"]), k1(tr["rdn"])):
        blob += np.asarray(a, dtype="<f8").tobytes()
    return blob, L


def _w_expected(geom, out, L):
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    n3, n2 = int(np.prod(L.s3)), int(np.prod(L.s2))
    vals = np.frombuffer(out, dtype="<f8")
    off = [0]

    def take(n, s):
        x = vals[off[0]:off[0] + n].reshape(s, order="F")
        off[0] += n
        return x

    shapes = cases.w_shapes(geom)
    t3 = lambda name: L.from3(take(n3, L.s3), shapes[name])  # noqa: E731
    t2 = lambda: L.from2(take(n2, L.s2), (ny, nx))  # noqa: E731
    e = {}
    for n in W_STATE:
        e["tl_" + n] = t3(n)
        e["tl_d_" + n] = t3(n)
    for n in W_STATE:
        e["ad_" + n] = t3(n)
    for n in W_AD3:
        e["ad_" + n] = t3(n)
    a_mut, a_muts, a_muave, a_mu, _ = t2(), t2(), t2(), t2(), t2()
    assert off[0] == vals.size, (off[0], vals.size)
    ep = geom.epssm
    e["ad_mup"] = a_mut + a_muts
    e["ad_mu_pp"] = a_muts + 0.5 * (1 + ep) * a_muave + a_mu
    e["ad_mu_pp_old"] = 0.5 * (1 - ep) * a_muave
    if not geom.moist_cq:
        e["ad_cqw"] = np.zeros(shapes["cqw"])
    return e


def advance_w(exe: str, runner: list[str], seed: int = 20261120) -> dict:
    out = {}
    for name, over in W_CASES.items():
        geom = cases.w_geometry(**over)
        tr = cases.w_trajectory(geom, seed)
        pe = cases.w_perturbation(geom, seed + 1)
        seeds = cases.w_perturbation(geom, seed + 2)
        blob, L = _w_record(geom, tr, pe, seeds)
        with tempfile.TemporaryDirectory(dir=Path(exe).parent) as tmp:
            fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            fin.write_bytes(blob)
            subprocess.run(runner + [exe, str(fin), str(fout)], check=True)
            result = fout.read_bytes()
        exp = _w_expected(geom, result, L)
        sz = geom.spec_zone
        col = np.zeros((geom.ny, geom.nx), bool)
        col[sz:geom.ny - sz, sz:geom.nx - sz] = True
        prefix = f"{name}/"
        out[prefix + "geometry"] = np.array(json.dumps(dataclasses.asdict(geom)))
        for k, v in tr.items():
            out[prefix + "traj/" + k] = v
        for k, v in pe.items():
            out[prefix + "pert/" + k] = v
        for n in W_STATE:
            out[prefix + "seed/" + n] = seeds[n]
        out[prefix + "mask/col"] = col
        # The bottom condition's faces: every face, since both codes add
        # only the solved columns' terms into a zero start.
        out[prefix + "mask/ad_u_pp"] = np.ones(exp["ad_u_pp"].shape, bool)
        out[prefix + "mask/ad_v_pp"] = np.ones(exp["ad_v_pp"].shape, bool)
        for k, v in exp.items():
            out[prefix + "wrfplus/" + k] = v
    return out


#: calc_coef_w's fixture configurations (c1 = 1, c2 = 0; every column).
CW_CASES = {
    "plain": dict(),
    "lid_moist": dict(top_lid=1, moist_cq=1, epssm=0.3),
    "deep_moist": dict(nz=12, ny=5, nx=6, moist_cq=1),
}
GAMMA = 1.4


def _cw_record(geom, tr, pe, seeds):
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    L = Layout(nz, ny, nx)
    c2a = GAMMA * tr["p"] / tr["alt"]
    c2ad = GAMMA * (pe["p"] - tr["p"] / tr["alt"] * pe["alt"]) / tr["alt"]
    one = np.ones_like(tr["cqw"])
    cqw = tr["cqw"] if geom.moist_cq else one
    cqwd = pe["cqw"] if geom.moist_cq else 0 * one
    k1 = lambda a: np.concatenate([a, np.zeros(nz + 1 - a.size)])  # noqa: E731
    head = np.array([nx, ny, nz, geom.top_lid], dtype=np.int32)
    reals = np.array([geom.dtau, geom.epssm])
    a3 = [tr["a"], pe["a"], tr["alpha"], pe["alpha"], tr["gam"], pe["gam"], cqw, cqwd,
          c2a, c2ad, seeds["a"], seeds["alpha"], seeds["gam"]]
    a2 = [tr["mub2d"] + tr["mup"], pe["mup"]]
    blob = head.tobytes() + reals.astype("<f8").tobytes()
    for a in a3:
        blob += L.w3(a).astype("<f8").tobytes(order="F")
    for a in a2:
        blob += L.w2(a).astype("<f8").tobytes(order="F")
    for a in (k1(tr["rdn"]), k1(tr["rdnw"])):
        blob += np.asarray(a, dtype="<f8").tobytes()
    return blob, L, c2a, c2ad


def _cw_expected(geom, out, L, tr, seeds, c2a, c2ad):
    n3, n2 = int(np.prod(L.s3)), int(np.prod(L.s2))
    vals = np.frombuffer(out, dtype="<f8")
    off = [0]

    def take(n, s):
        x = vals[off[0]:off[0] + n].reshape(s, order="F")
        off[0] += n
        return x

    shapes = cases.cw_shapes(geom)
    t3 = lambda name: L.from3(take(n3, L.s3), shapes[name])  # noqa: E731
    e = {"tl_c2a": c2a, "tl_d_c2a": c2ad}
    for n in ("a", "alpha", "gam"):
        e["tl_" + n] = t3(n)
        e["tl_d_" + n] = t3(n)
    for n in ("a", "alpha", "gam"):
        e["ad_" + n] = t3(n)
    a_c2a = seeds["c2a"] + t3("c2a")
    a_cqw = t3("cqw")
    a_mut = L.from2(take(n2, L.s2), shapes["mup"])
    assert off[0] == vals.size, (off[0], vals.size)
    e["ad_c2a"] = np.zeros(shapes["c2a"])
    e["ad_p"] = a_c2a * GAMMA / tr["alt"]
    e["ad_alt"] = -a_c2a * c2a / tr["alt"]
    e["ad_mup"] = a_mut
    e["ad_cqw"] = a_cqw if geom.moist_cq else np.zeros(shapes["cqw"])
    return e


def calc_coef_w(exe: str, runner: list[str], seed: int = 20261121) -> dict:
    out = {}
    for name, over in CW_CASES.items():
        geom = cases.cw_geometry(**over)
        tr = cases.cw_trajectory(geom, seed)
        pe = cases.cw_perturbation(geom, seed + 1)
        seeds = cases.cw_perturbation(geom, seed + 2)
        blob, L, c2a, c2ad = _cw_record(geom, tr, pe, seeds)
        with tempfile.TemporaryDirectory(dir=Path(exe).parent) as tmp:
            fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            fin.write_bytes(blob)
            subprocess.run(runner + [exe, str(fin), str(fout)], check=True)
            result = fout.read_bytes()
        exp = _cw_expected(geom, result, L, tr, seeds, c2a, c2ad)
        prefix = f"{name}/"
        out[prefix + "geometry"] = np.array(json.dumps(dataclasses.asdict(geom)))
        for k, v in tr.items():
            out[prefix + "traj/" + k] = v
        for k, v in pe.items():
            out[prefix + "pert/" + k] = v
        for n in CW_STATE:
            out[prefix + "seed/" + n] = seeds[n]
        out[prefix + "mask/col"] = np.ones((geom.ny, geom.nx), bool)
        for k, v in exp.items():
            out[prefix + "wrfplus/" + k] = v
    return out


#: advance_mu_t fixtures: WRFPLUS's specified boundary (spec_zone 1, open
#: lateral sides, the exact frame), c1 = 1, with and without map factors.
MT_CASES = {
    "plain": dict(spec_zone=1, open_x=1, open_y=1),
    "mapped": dict(spec_zone=1, open_x=1, open_y=1, mapped=1, nz=12),
    "small_mapped": dict(nz=3, ny=4, nx=5, spec_zone=1, open_x=1, open_y=1,
                         mapped=1),
}


def _mt_frame(geom):
    j, i = np.mgrid[0:geom.ny, 0:geom.nx]
    z = geom.spec_zone
    return (i < z) | (i >= geom.nx - z) | (j < z) | (j >= geom.ny - z)


def _mt_faces(geom, mut, axis):
    """WRF muu (axis 1) / muv (axis 0) from the cell column mass: the
    interior faces' two-cell mean, zero on the outer faces (unread)."""
    ny, nx = mut.shape
    if axis == 1:
        out = np.zeros((ny, nx + 1))
        out[:, 1:nx] = 0.5 * (mut[:, 1:] + mut[:, :-1])
    else:
        out = np.zeros((ny + 1, nx))
        out[1:ny] = 0.5 * (mut[1:] + mut[:-1])
    return out


def _mt_record(geom, tr, pe, seeds):
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    L = Layout(nz, ny, nx)
    k1 = lambda a: np.concatenate([a, np.zeros(nz + 1 - a.size)])  # noqa: E731
    head = np.array([nx, ny, nz, geom.spec_zone], dtype=np.int32)
    reals = np.array([geom.rdx, geom.rdy, geom.dtau, 0.1])
    a3 = [tr["ww_pp"], pe["ww_pp"], tr["ww_ref"], pe["ww_ref"], tr["u_pp"], pe["u_pp"],
          tr["u"], pe["u"], tr["v_pp"], pe["v_pp"], tr["v"], pe["v"], tr["th_pp"],
          pe["th_pp"], tr["thp"], pe["thp"], tr["rth_t"], pe["rth_t"], seeds["ww_pp"],
          seeds["th_pp"], seeds["th_old"]]
    mut = tr["mub2d"] + tr["mup"]
    a2 = [tr["mu_pp"], pe["mu_pp"], mut, pe["mup"], _mt_faces(geom, mut, 1),
          _mt_faces(geom, pe["mup"], 1), _mt_faces(geom, mut, 0),
          _mt_faces(geom, pe["mup"], 0), tr["muts"], pe["muts"], tr["rmu_t"],
          pe["rmu_t"], tr["msft"], tr["msfu"], tr["msfv"], seeds["mu_pp"],
          seeds["muts"], seeds["mudf"]]
    blob = head.tobytes() + reals.astype("<f8").tobytes()
    for a in a3:
        blob += L.w3(a).astype("<f8").tobytes(order="F")
    for a in a2:
        blob += L.w2(a).astype("<f8").tobytes(order="F")
    for a in (k1(tr["fnm"][:nz]), k1(tr["fnp"][:nz]), k1(tr["dnw"]), k1(tr["rdnw"])):
        blob += np.asarray(a, dtype="<f8").tobytes()
    return blob, L


def _mt_expected(geom, out, L, tr, pe, seeds):
    n3, n2 = int(np.prod(L.s3)), int(np.prod(L.s2))
    vals = np.frombuffer(out, dtype="<f8")
    off = [0]

    def take(n, s):
        x = vals[off[0]:off[0] + n].reshape(s, order="F")
        off[0] += n
        return x

    sh = cases.mt_shapes(geom)
    t3 = lambda name: L.from3(take(n3, L.s3), sh[name])  # noqa: E731
    t2 = lambda name: L.from2(take(n2, L.s2), sh[name])  # noqa: E731
    frame = _mt_frame(geom)
    e = {}
    for n in ("ww_pp", "th_pp", "th_old"):
        e["tl_" + n] = t3(n)
        e["tl_d_" + n] = t3(n)
    for n in ("mu_pp", "muts", "mudf"):
        e["tl_" + n] = t2(n)
        e["tl_d_" + n] = t2(n)
    e["tl_mu_old"], e["tl_d_mu_old"] = tr["mu_pp"], pe["mu_pp"]
    e["tl_p_old"], e["tl_d_p_old"] = tr["p_pp"], pe["p_pp"]
    a_ww = t3("ww_pp")
    e["ad_ww_ref"] = t3("ww_ref")
    e["ad_u_pp"] = t3("u_pp")
    e["ad_u"] = t3("u")
    e["ad_v_pp"] = t3("v_pp")
    e["ad_v"] = t3("v")
    e["ad_th_pp"] = t3("th_pp")
    e["ad_thp"] = t3("thp")
    e["ad_th_old"] = t3("th_old")
    e["ad_rth_t"] = t3("rth_t")
    a_mu = t2("mu_pp")
    a_mut = t2("mup")
    a_muave = t2("mup")
    a_muts = t2("muts")
    a_muu = L.from2(take(n2, L.s2), (geom.ny, geom.nx + 1))
    a_muv = L.from2(take(n2, L.s2), (geom.ny + 1, geom.nx))
    a_mudf = t2("mudf")
    e["ad_rmu_t"] = t2("rmu_t")
    assert off[0] == vals.size, (off[0], vals.size)
    assert not np.any(a_muave), "muave's adjoint was seeded with zero"
    # Woof consumes Omega'' on the solved columns (all levels; WRF passes
    # its untouched surface and top through) and leaves the frame alone.
    e["ad_ww_pp"] = np.where(frame[None], seeds["ww_pp"], 0.0)
    assert np.array_equal(np.where(frame[None], a_ww, 0.0)[:, :, :],
                          e["ad_ww_pp"]), "WRFPLUS changed frame Omega adjoints"
    # WRF saves t_ave on the solved columns only; woof's th_old copy
    # covers the frame too (its adjoint consumed there as well).
    f3 = frame[None]
    assert np.allclose(e["tl_th_old"][:, ~frame], tr["th_pp"][:, ~frame], rtol=0, atol=0)
    e["tl_th_old"], e["tl_d_th_old"] = tr["th_pp"], pe["th_pp"]
    e["ad_th_pp"] = e["ad_th_pp"] + np.where(f3, seeds["th_old"], 0.0)
    e["ad_th_old"] = np.zeros(sh["th_old"])
    e["ad_mu_pp"] = a_mu + seeds["mu_old"]
    e["ad_mu_old"] = np.zeros(sh["mu_old"])
    e["ad_muts"] = a_muts
    e["ad_mudf"] = np.where(frame, 0.0, a_mudf)       # woof writes 0 on the frame
    e["ad_p_old"] = np.zeros(sh["p_old"])
    e["ad_p_pp"] = seeds["p_old"].copy()
    e["ad_mup"] = (a_mut + 0.5 * (a_muu[:, 1:] + a_muu[:, :-1])
                   + 0.5 * (a_muv[1:] + a_muv[:-1]))
    return e


def advance_mu_t(exe: str, runner: list[str], seed: int = 20261122) -> dict:
    out = {}
    for name, over in MT_CASES.items():
        geom = cases.mt_geometry(**over)
        tr = cases.mt_trajectory(geom, seed)
        pe = cases.mt_perturbation(geom, seed + 1)
        seeds = cases.mt_perturbation(geom, seed + 2)
        # WRF's Omega surface and top enter as zero (its ww(1) update
        # subtracts ww_1 from the input; ww(kde) is not written).
        for d in (tr, pe):
            d["ww_pp"][0] = 0.0
            d["ww_pp"][-1] = 0.0
        blob, L = _mt_record(geom, tr, pe, seeds)
        with tempfile.TemporaryDirectory(dir=Path(exe).parent) as tmp:
            fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            fin.write_bytes(blob)
            subprocess.run(runner + [exe, str(fin), str(fout)], check=True)
            result = fout.read_bytes()
        exp = _mt_expected(geom, result, L, tr, pe, seeds)
        prefix = f"{name}/"
        out[prefix + "geometry"] = np.array(json.dumps(dataclasses.asdict(geom)))
        for k, v in tr.items():
            out[prefix + "traj/" + k] = v
        for k, v in pe.items():
            out[prefix + "pert/" + k] = v
        for n in MT_STATE:
            out[prefix + "seed/" + n] = seeds[n]
        interior = ~_mt_frame(geom)
        out[prefix + "mask/tl_mudf"] = interior
        out[prefix + "mask/tl_d_mudf"] = interior
        for k, v in exp.items():
            out[prefix + "wrfplus/" + k] = v
    return out


#: small_step cases.  WRFPLUS has c1 = 1, c2 = 0; small_step_finish drops
#: h_diabatic on the last RK stage only, so remove_hdiab iff rk_step 3.
#: The halo columns carry woof's clamped (boundary) or wrapped
#: (periodic) neighbours, so both face-mass rules are checked.  There is
#: no rk_step 1 case: WRF's small_step_prep copies the _2 state into _1
#: at the first stage, so every init output and adjoint is zero there
#: (checked: a first-stage case came back all zeros), and woof's rk_step
#: 1 rule (mtf = msf) is checked by the dot and FD tests instead.
SS_CASES = {
    "clamped": dict(rk_step=2),
    "periodic_msf_last": dict(rk_step=3, has_msf=1, boundary_x=0, boundary_y=0,
                              remove_hdiab=1, hdiab_dt=6.0, nz=12),
    "small_mixed": dict(nz=3, ny=2, nx=3, rk_step=2, has_msf=1, boundary_x=0),
}
RK_ORDER = 3


def _ss_cells(f, n, bnd):
    """Cells a (right) and b (left) of faces f on an n-cell axis."""
    if bnd:
        return np.minimum(f, n - 1), np.maximum(f - 1, 0)
    return f % n, (f - 1) % n


def _ss_faces(geom, m, axis):
    """Face means of the cell field m on u (axis 1) or v (axis 0) faces."""
    if axis == 1:
        a, b = _ss_cells(np.arange(geom.nx + 1), geom.nx, geom.boundary_x)
        return 0.5 * (m[:, a] + m[:, b])
    a, b = _ss_cells(np.arange(geom.ny + 1), geom.ny, geom.boundary_y)
    return 0.5 * (m[a] + m[b])


def _ss_faces_t(geom, f, axis):
    """Transpose of :func:`_ss_faces`."""
    out = np.zeros((geom.ny, geom.nx))
    if axis == 1:
        a, b = _ss_cells(np.arange(geom.nx + 1), geom.nx, geom.boundary_x)
        np.add.at(out, (slice(None), a), 0.5 * f)
        np.add.at(out, (slice(None), b), 0.5 * f)
    else:
        a, b = _ss_cells(np.arange(geom.ny + 1), geom.ny, geom.boundary_y)
        np.add.at(out, a, 0.5 * f)
        np.add.at(out, b, 0.5 * f)
    return out


def _ss_halo_cells(geom):
    """woof cells behind the Fortran halo columns 0 and n+1 of each axis."""
    nx, ny = geom.nx, geom.ny
    ix = (0, nx - 1) if geom.boundary_x else (nx - 1, 0)
    jy = (0, ny - 1) if geom.boundary_y else (ny - 1, 0)
    return ix, jy


def _ss_halo(geom, L, m):
    out = L.w2(m)
    (il, ir), (jl, jr) = _ss_halo_cells(geom)
    nx, ny = geom.nx, geom.ny
    out[0, 1:1 + ny] = m[:, il]
    out[nx + 1, 1:1 + ny] = m[:, ir]
    out[1:1 + nx, 0] = m[jl, :]
    out[1:1 + nx, ny + 1] = m[jr, :]
    return out


def _ss_fold(geom, x):
    """woof cell adjoint of a Fortran column array with halo entries."""
    nx, ny = geom.nx, geom.ny
    out = x[1:1 + nx, 1:1 + ny].T.copy()
    (il, ir), (jl, jr) = _ss_halo_cells(geom)
    out[:, il] += x[0, 1:1 + ny]
    out[:, ir] += x[nx + 1, 1:1 + ny]
    out[jl, :] += x[1:1 + nx, 0]
    out[jr, :] += x[1:1 + nx, ny + 1]
    return out


def _ss_record(geom, tr, pe, sd):
    nz = geom.nz
    L = Layout(nz, geom.ny, geom.nx)
    head = np.array([geom.nx, geom.ny, nz, geom.rk_step, RK_ORDER], dtype=np.int32)
    reals = np.array([geom.hdiab_dt])
    mus = tr["mub2d"] + tr["mup"]
    a3 = []
    for n1, n2 in (("u0", "u"), ("v0", "v"), ("w0", "w"), ("thp0", "thp"),
                   ("php0", "php")):
        a3 += [tr[n1], pe[n1], tr[n2], pe[n2]]
    a3 += [tr["p"], pe["p"], tr["alt"], pe["alt"]]
    a3 += [sd[n] for n in ("u_pp", "v_pp", "w_pp", "th_pp", "ph_pp", "al_pp", "p_pp",
                           "p_pp_old")]
    for n in ("u_pp", "v_pp", "w_pp", "th_pp", "ph_pp", "u", "v", "w", "thp", "php",
              "h_diabatic"):
        a3 += [tr[n], pe[n]]
    a3 += [sd[n] for n in ("u", "v", "w", "thp", "php")]
    blob = head.tobytes() + reals.astype("<f8").tobytes()
    for a in a3:
        blob += L.w3(a).astype("<f8").tobytes(order="F")
    a2 = [_ss_halo(geom, L, tr["mub2d"]), _ss_halo(geom, L, tr["mup0"]),
          _ss_halo(geom, L, pe["mup0"]), _ss_halo(geom, L, tr["mup"]),
          _ss_halo(geom, L, pe["mup"]), L.w2(_ss_faces(geom, mus, 1)),
          L.w2(_ss_faces(geom, pe["mup"], 1)), L.w2(_ss_faces(geom, mus, 0)),
          L.w2(_ss_faces(geom, pe["mup"], 0)), L.w2(mus), L.w2(pe["mup"]),
          L.w2(tr["msfu"]), L.w2(tr["msfv"]), L.w2(tr["msft"]), L.w2(sd["mu_pp"])]
    a2 += [L.w2(tr["mu_pp"]), L.w2(pe["mu_pp"]), L.w2(tr["mup"]), L.w2(pe["mup"]),
           L.w2(mus), L.w2(pe["mup"]), L.w2(tr["muts"]), L.w2(pe["muts"]),
           L.w2(_ss_faces(geom, mus, 1)), L.w2(_ss_faces(geom, pe["mup"], 1)),
           L.w2(_ss_faces(geom, tr["muts"], 1)), L.w2(_ss_faces(geom, pe["muts"], 1)),
           L.w2(_ss_faces(geom, mus, 0)), L.w2(_ss_faces(geom, pe["mup"], 0)),
           L.w2(_ss_faces(geom, tr["muts"], 0)), L.w2(_ss_faces(geom, pe["muts"], 0)),
           L.w2(tr["msfv"]), L.w2(sd["mup"])]
    for a in a2:
        blob += np.asarray(a, dtype="<f8").tobytes(order="F")
    k1 = np.concatenate([tr["rdnw"], [0.0]])
    blob += k1.astype("<f8").tobytes()
    blob += np.concatenate([1.0 / tr["rdnw"], [0.0]]).astype("<f8").tobytes()
    return blob, L


def _ss_expected(geom, out, L, sh):
    """WRFPLUS outputs mapped to woof names: init/<tl_|tl_d_|ad_>name and
    finish/<...>name.  The halo adjoints fold onto woof's neighbours and
    the face-mass adjoints onto the cells (the transpose of the faces)."""
    n3, n2 = int(np.prod(L.s3)), int(np.prod(L.s2))
    vals = np.frombuffer(out, dtype="<f8")
    off = [0]

    def take(n, s):
        x = vals[off[0]:off[0] + n].reshape(s, order="F")
        off[0] += n
        return x

    t3 = lambda name: L.from3(take(n3, L.s3), sh[name])  # noqa: E731
    r2 = lambda: take(n2, L.s2)  # noqa: E731
    uf = lambda x: _ss_faces_t(geom, L.from2(x, (geom.ny, geom.nx + 1)), 1)  # noqa: E731
    vf = lambda x: _ss_faces_t(geom, L.from2(x, (geom.ny + 1, geom.nx)), 0)  # noqa: E731
    e = {}
    for n in ("u_pp", "v_pp", "w_pp", "th_pp", "ph_pp", "al_pp", "p_pp", "p_pp_old"):
        e["init/tl_" + n] = t3(n)
        e["init/tl_d_" + n] = t3(n)
    e["init/tl_mu_pp"] = L.from2(r2(), sh["mu_pp"])
    e["init/tl_d_mu_pp"] = L.from2(r2(), sh["mu_pp"])
    for n in ("u0", "u", "v0", "v", "w0", "w", "thp0", "thp", "php0", "php", "p", "alt"):
        e["init/ad_" + n] = t3(n)
    a_mu1, a_mu2, a_muu, a_muv, a_mut = r2(), r2(), r2(), r2(), r2()
    e["init/ad_mup0"] = _ss_fold(geom, a_mu1)
    e["init/ad_mup"] = (_ss_fold(geom, a_mu2) + L.from2(a_mut, sh["mup"])
                        + uf(a_muu) + vf(a_muv))
    for n in SI_STATE:
        e["init/ad_" + n] = np.zeros(sh[n])
    for n in ("u", "v", "w", "thp", "php"):
        e["finish/tl_" + n] = t3(n)
        e["finish/tl_d_" + n] = t3(n)
    e["finish/tl_mup"] = L.from2(r2(), sh["mup"])
    e["finish/tl_d_mup"] = L.from2(r2(), sh["mup"])
    for pp, n in (("u_pp", "u"), ("v_pp", "v"), ("w_pp", "w"), ("th_pp", "thp"),
                  ("ph_pp", "php")):
        e["finish/ad_" + pp] = t3(pp)
        e["finish/ad_" + n] = t3(n)
    e["finish/ad_h_diabatic"] = t3("h_diabatic")
    b_mu2, b_mus, b_mut, b_muts = r2(), r2(), r2(), r2()
    b_muu, b_muus, b_muv, b_muvs = r2(), r2(), r2(), r2()
    e["finish/ad_mu_pp"] = L.from2(b_mu2, sh["mu_pp"])
    e["finish/ad_mup"] = (L.from2(b_mus, sh["mup"]) + L.from2(b_mut, sh["mup"])
                          + uf(b_muu) + vf(b_muv))
    e["finish/ad_muts"] = L.from2(b_muts, sh["muts"]) + uf(b_muus) + vf(b_muvs)
    assert off[0] == vals.size, (off[0], vals.size)
    return e


def small_step(exe: str, runner: list[str], seed: int = 20261123) -> dict:
    out = {}
    for name, over in SS_CASES.items():
        geom = cases.ss_geometry(**over)
        assert bool(geom.remove_hdiab) == (geom.rk_step == RK_ORDER)
        tr = cases.ss_trajectory(geom, seed)
        pe = cases.ss_perturbation(geom, seed + 1)
        sd = cases.ss_perturbation(geom, seed + 2)
        blob, L = _ss_record(geom, tr, pe, sd)
        with tempfile.TemporaryDirectory(dir=Path(exe).parent) as tmp:
            fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            fin.write_bytes(blob)
            subprocess.run(runner + [exe, str(fin), str(fout)], check=True)
            result = fout.read_bytes()
        exp = _ss_expected(geom, result, L, cases.ss_shapes(geom))
        prefix = f"{name}/"
        out[prefix + "geometry"] = np.array(json.dumps(dataclasses.asdict(geom)))
        for k, v in tr.items():
            out[prefix + "traj/" + k] = v
        for k, v in pe.items():
            out[prefix + "pert/" + k] = v
        for k, v in sd.items():
            out[prefix + "seed/" + k] = v
        for k, v in exp.items():
            out[prefix + "wrfplus/" + k] = v
    return out


#: rk_stage: one geometry per case for the four glue operators; WRFPLUS
#: has no hybrid coordinate (c1 = 1, c2 = 0) and a 3-D base.
RK_CASES = {
    "clamped_h1": dict(nz=7, ny=5, nx=6, boundary_x=1, boundary_y=1, has_msf=0,
                       hypso=1, carried=1),
    "periodic_msf_h2": dict(nz=12, ny=4, nx=5, boundary_x=0, boundary_y=0,
                            has_msf=1, hypso=2, carried=1),
    "small_mixed": dict(nz=3, ny=2, nx=3, boundary_x=0, boundary_y=1, has_msf=1,
                        hypso=1, carried=0),
    "mixed_h2_formed": dict(nz=9, ny=3, nx=4, boundary_x=1, boundary_y=0,
                            has_msf=0, hypso=2, carried=0),
}
RK_DX, RK_DY = 3000.0, 2500.0
AT_NAMES = ("ru", "rv", "rw", "rph", "rth")


def _rk_geometries(c):
    nz, ny, nx = c["nz"], c["ny"], c["nx"]
    bx, by, msf = c["boundary_x"], c["boundary_y"], c["has_msf"]
    return {
        "prp": cases.prp_geometry(nz, ny, nx, hypso=c["hypso"], carried=c["carried"]),
        "rsp": cases.rsp_geometry(nz, ny, nx, has_msf=msf, boundary_x=bx,
                                  boundary_y=by, rdx=1.0 / RK_DX, rdy=1.0 / RK_DY),
        "at": cases.at_geometry(nz, ny, nx, has_msf=msf),
        "sw": cases.sw_geometry(nz, ny, nx, mapped=msf, boundary_x=bx,
                                boundary_y=by, half_rdx=0.5 / RK_DX,
                                half_rdy=0.5 / RK_DY),
    }


def _rk_at_factor(n, tr):
    """WRF's held tendf is woof's times the map factor woof divided by
    (u/v faces: rk_step_prep's msfu and msfv_inv of the same case)."""
    if n == "ru":
        return tr["rsp"]["msfu"]
    if n == "rv":
        return 1.0 / tr["rsp"]["msfv"]
    return tr["at"]["msft"]


def _rk_record(c, G, tr, pe, sd):
    nz, ny, nx = c["nz"], c["ny"], c["nx"]
    L = Layout(nz, ny, nx)
    sg = G["rsp"]                       # boundary flags for the halos
    halo = lambda m: _ss_halo(sg, L, m)  # noqa: E731
    head = np.array([nx, ny, nz, c["hypso"], 1 - c["boundary_x"],
                     1 - c["boundary_y"]], dtype=np.int32)
    swg = G["sw"]
    reals = np.array([1.0 / RK_DX, 1.0 / RK_DY, G["prp"].p_top, swg.cf1, swg.cf2,
                      swg.cf3])
    blob = head.tobytes() + reals.astype("<f8").tobytes()
    pad = lambda v: np.concatenate([v, np.zeros(nz + 1 - v.size)])  # noqa: E731
    r, q, a, w = tr["rsp"], tr["prp"], tr["at"], tr["sw"]
    for v in (pad(r["dnw"]), pad(q["rdnw"]), pad(q["c3h"]), q["c3f"]):
        blob += np.asarray(v, dtype="<f8").tobytes()
    a3, a2 = [], []
    # rk_step_prep
    rp, rs = pe["rsp"], sd["rsp"]
    a3 = [r["u"], rp["u"], r["v"], rp["v"], rs["ru"], rs["rv"], rs["ww"]]
    a2 = [halo(r["mup"]), halo(rp["mup"]), halo(r["mub"]), L.w2(r["msfu"]),
          L.w2(r["msfv"]), L.w2(r["msft"])]
    blob += _rk_pack(L, a3, a2)
    # calc_p_rho_phi
    qp, qs = pe["prp"], sd["prp"]
    if c["carried"]:
        muts, g_muts = q["muts"], qp["muts"]
    else:
        muts, g_muts = q["mub"] + q["mup"], qp["mup"]
    a3 = [q["thp"], qp["thp"], q["php"], qp["php"], q["phb"], q["alb"], q["pb"],
          qs["al"] + qs["alt"], qs["p"] + qs["p_pp"]]
    a2 = [L.w2(q["mup"]), L.w2(qp["mup"]), L.w2(muts), L.w2(g_muts)]
    blob += _rk_pack(L, a3, a2)
    # rk_addtend_dry (rk_step 2)
    ap, as_ = pe["at"], sd["at"]
    a3 = [a[n + "_t"] for n in AT_NAMES] + [ap[n + "_t"] for n in AT_NAMES]
    a3 += [a[n + "_tf"] * _rk_at_factor(n, tr) for n in AT_NAMES]
    a3 += [ap[n + "_tf"] * _rk_at_factor(n, tr) for n in AT_NAMES]
    a3 += [a["h_diabatic"], ap["h_diabatic"]] + [as_[n + "_t"] for n in AT_NAMES]
    a2 = [L.w2(a["rmu_t"]), L.w2(ap["rmu_t"]), L.w2(a["rmu_tf"]), L.w2(ap["rmu_tf"]),
          L.w2(as_["rmu_t"]), L.w2(a["mub"] + a["mup"]), L.w2(ap["mup"]),
          L.w2(r["msfu"]), L.w2(r["msfv"]), L.w2(a["msft"])]
    blob += _rk_pack(L, a3, a2)
    # set_w_surface
    wp, ws = pe["sw"], sd["sw"]
    a3 = [w["w"], wp["w"], w["u"], wp["u"], w["v"], wp["v"], ws["w"]]
    a2 = [halo(w["ht"]), L.w2(w["msft"])]
    blob += _rk_pack(L, a3, a2)
    return blob, L


def _rk_pack(L, a3, a2):
    blob = b""
    for x in a3:
        blob += L.w3(x).astype("<f8").tobytes(order="F")
    for x in a2:
        blob += np.asarray(x, dtype="<f8").tobytes(order="F")
    return blob


def _rk_expected(c, G, out, L, tr):
    """WRFPLUS outputs mapped to woof names: <op>/<tl_|tl_d_|ad_>name.
    The halo adjoints of mu fold onto woof's neighbours; WRF's p and al
    adjoints stand for woof's p + p_pp and al + alt."""
    n3, n2 = int(np.prod(L.s3)), int(np.prod(L.s2))
    vals = np.frombuffer(out, dtype="<f8")
    off = [0]

    def take(n, s):
        x = vals[off[0]:off[0] + n].reshape(s, order="F")
        off[0] += n
        return x

    sh = {op: f(G[op]) for op, f in (("rsp", cases.rsp_shapes),
                                     ("prp", cases.prp_shapes),
                                     ("at", cases.at_shapes),
                                     ("sw", cases.sw_shapes))}
    t3 = lambda op, n: L.from3(take(n3, L.s3), sh[op][n])  # noqa: E731
    r2 = lambda: take(n2, L.s2)  # noqa: E731
    e = {}
    for n in ("ru", "rv", "ww"):
        e["rsp/tl_" + n] = t3("rsp", n)
        e["rsp/tl_d_" + n] = t3("rsp", n)
    for n in ("u", "v", "ru", "rv", "ww"):
        e["rsp/ad_" + n] = t3("rsp", n)
    e["rsp/ad_mup"] = _ss_fold(G["rsp"], r2())
    al, gal, p, gp = (t3("prp", "al") for _ in range(4))
    e["prp/tl_al"], e["prp/tl_d_al"], e["prp/tl_d_alt"] = al, gal, gal
    e["prp/tl_p_pp"], e["prp/tl_d_p_pp"], e["prp/tl_d_p"] = p, gp, gp
    e["prp/ad_thp"] = t3("prp", "thp")
    e["prp/ad_php"] = t3("prp", "php")
    sal, sp = t3("prp", "al"), t3("prp", "p")
    for n, v in (("al", sal), ("alt", sal), ("p", sp), ("p_pp", sp)):
        e["prp/ad_" + n] = v
    a_mu, a_muts = L.from2(r2(), (L.ny, L.nx)), L.from2(r2(), (L.ny, L.nx))
    if c["carried"]:
        e["prp/ad_mup"], e["prp/ad_muts"] = a_mu, a_muts
    else:
        e["prp/ad_mup"], e["prp/ad_muts"] = a_mu + a_muts, np.zeros_like(a_mu)
    for n in AT_NAMES:
        e[f"at/tl_{n}_t"] = t3("at", n + "_t")
        e[f"at/tl_d_{n}_t"] = t3("at", n + "_t")
    e["at/tl_rmu_t"] = L.from2(r2(), (L.ny, L.nx))
    e["at/tl_d_rmu_t"] = L.from2(r2(), (L.ny, L.nx))
    for n in AT_NAMES:
        e[f"at/ad_{n}_tf"] = t3("at", n + "_tf") * _rk_at_factor(n, tr)
    e["at/ad_h_diabatic"] = t3("at", "h_diabatic")
    for n in AT_NAMES:
        e[f"at/ad_{n}_t"] = t3("at", n + "_t")
    e["at/ad_rmu_tf"] = L.from2(r2(), (L.ny, L.nx))
    e["at/ad_rmu_t"] = L.from2(r2(), (L.ny, L.nx))
    e["at/ad_mup"] = L.from2(r2(), (L.ny, L.nx))
    e["sw/tl_w"] = t3("sw", "w")
    e["sw/tl_d_w"] = t3("sw", "w")
    for n in ("u", "v", "w"):
        e["sw/ad_" + n] = t3("sw", n)
    assert off[0] == vals.size, (off[0], vals.size)
    return e


def rk_stage(exe: str, runner: list[str], seed: int = 20261124) -> dict:
    out = {}
    for name, c in RK_CASES.items():
        G = _rk_geometries(c)
        tr = {op: getattr(cases, op + "_trajectory")(G[op], seed + 10 * k)
              for k, op in enumerate(G)}
        pe = {op: getattr(cases, op + "_perturbation")(G[op], seed + 10 * k + 1)
              for k, op in enumerate(G)}
        sd = {op: getattr(cases, op + "_perturbation")(G[op], seed + 10 * k + 2)
              for k, op in enumerate(G)}
        blob, L = _rk_record(c, G, tr, pe, sd)
        with tempfile.TemporaryDirectory(dir=Path(exe).parent) as tmp:
            fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            fin.write_bytes(blob)
            subprocess.run(runner + [exe, str(fin), str(fout)], check=True)
            result = fout.read_bytes()
        exp = _rk_expected(c, G, result, L, tr)
        prefix = f"{name}/"
        out[prefix + "case"] = np.array(json.dumps(c))
        for op in G:
            out[prefix + op + "/geometry"] = np.array(
                json.dumps(dataclasses.asdict(G[op])))
            for k, v in tr[op].items():
                out[prefix + op + "/traj/" + k] = v
            for k, v in pe[op].items():
                out[prefix + op + "/pert/" + k] = v
            for k, v in sd[op].items():
                out[prefix + op + "/seed/" + k] = v
        for k, v in exp.items():
            op, rest = k.split("/", 1)
            out[prefix + op + "/wrfplus/" + rest] = v
    return out


#: advect: one case per kind and lateral boundary, with the h/v orders
#: spread over them.  open_* without spec are WRF's open_xs/xe/ys/ye.
ADV_CASES = {
    "scalar_periodic": dict(kind="scalar"),
    "scalar_open_v5": dict(kind="scalar", open_x=1, open_y=1, vorder=5),
    "scalar_spec_h6_v4": dict(kind="scalar", open_x=1, open_y=1, spec=1, horder=6, vorder=4),
    "scalar_mixed_h3_v6": dict(kind="scalar", open_y=1, spec=1, horder=3, vorder=6),
    "u_periodic_v5": dict(kind="u", vorder=5),
    "u_open": dict(kind="u", open_x=1, open_y=1),
    "u_spec_h4_v2": dict(kind="u", open_x=1, open_y=1, spec=1, horder=4, vorder=2),
    "u_mixed_open_x_h6": dict(kind="u", open_x=1, horder=6, vorder=6),
    "v_periodic_h3": dict(kind="v", horder=3, vorder=4),
    "v_open_v5": dict(kind="v", open_x=1, open_y=1, vorder=5),
    "v_spec": dict(kind="v", open_x=1, open_y=1, spec=1),
    "v_mixed_open_y_h4": dict(kind="v", open_y=1, horder=4, vorder=2),
    "w_periodic": dict(kind="w"),
    "w_open_v5": dict(kind="w", open_x=1, open_y=1, vorder=5),
    "w_spec_h6_v6": dict(kind="w", open_x=1, open_y=1, spec=1, horder=6, vorder=6),
    "w_mixed_h4_v2": dict(kind="w", open_x=1, spec=1, horder=4, vorder=2),
}
ADV_MODE = {"scalar": 1, "u": 2, "v": 3, "w": 4}
ADV_DIMS = dict(nz=8, ny=9, nx=10)


def _adv_index(n, length, open_, end=True):
    """woof index behind each Fortran memory column -3..n+4 (i.e. woof
    -4..n+3) of an axis with n cells and ``length`` points: periodic axes
    wrap the cells, open and specified axes copy the edge (WRF's
    zero-gradient halo).  A periodic staggered end face keeps index n when
    ``end`` (the mass fluxes, which woof reads there); the advected u (v)
    wraps it to 0, as woof's flux_div_u (_v) reads it."""
    idx = np.arange(-4, n + 4)
    if open_:
        return np.clip(idx, 0, length - 1)
    out = idx % n
    if length == n + 1 and end:
        out[idx == n] = n
    return out


class AdvLayout:
    """woof (k, j, i) <-> WRF (-3:nx+4, 1:nz+1, -3:ny+4)."""

    def __init__(self, geom):
        self.g = geom
        self.s3 = (geom.nx + 8, geom.nz + 1, geom.ny + 8)
        self.s2 = (geom.nx + 8, geom.ny + 8)

    def _maps(self, shape, end=True):
        g = self.g
        nk, nj, ni = shape
        return (np.clip(np.arange(g.nz + 1), 0, nk - 1),
                _adv_index(g.ny, nj, g.open_y, end), _adv_index(g.nx, ni, g.open_x, end))

    def pad3(self, a, end=True):
        kz, yi, xi = self._maps(a.shape, end)
        return np.transpose(a[np.ix_(kz, yi, xi)], (2, 0, 1))

    def fold3(self, x, shape, end=True):
        kz, yi, xi = self._maps(shape, end)
        out = np.zeros(shape)
        np.add.at(out, np.ix_(kz, yi, xi), np.transpose(x, (1, 2, 0)))
        return out

    def place3(self, a):
        out = np.zeros(self.s3)
        nk, nj, ni = a.shape
        out[4:4 + ni, :nk, 4:4 + nj] = np.transpose(a, (2, 0, 1))
        return out

    def take3(self, x, shape):
        nk, nj, ni = shape
        return np.ascontiguousarray(np.transpose(x[4:4 + ni, :nk, 4:4 + nj], (1, 2, 0)))

    def pad2(self, a):
        _, yi, xi = self._maps((1,) + a.shape)
        return a[np.ix_(yi, xi)].T


def adv_mask(geom):
    """woof tendency points WRF's advect_* treat as woof does: open
    (non-specified) u/v boundary-normal faces carry WRF's cb terms, which
    woof applies in its separate radiative boundary launch."""
    from woof.grad.wrfplus.advect import adv_shapes
    m = np.ones(adv_shapes(geom)["tend"], dtype=bool)
    if not geom.spec:
        if geom.kind == "u" and geom.open_x:
            m[:, :, 0] = m[:, :, -1] = False
        if geom.kind == "v" and geom.open_y:
            m[:, 0, :] = m[:, -1, :] = False
        # WRF's advect_w applies its open non-cb terms on the lid too;
        # woof's strict flux_div_w forms them there but does not apply them.
        if geom.kind == "w" and geom.open_x:
            m[-1, :, 0] = m[-1, :, -1] = False
        if geom.kind == "w" and geom.open_y:
            m[-1, 0, :] = m[-1, -1, :] = False
    # A periodic staggered end point repeats point 0 (woof reads the
    # wrapped field there, WRF its own halo column).
    if geom.kind == "u" and not geom.open_x:
        m[:, :, -1] = False
    if geom.kind == "v" and not geom.open_y:
        m[:, -1, :] = False
    return m


def _adv_record(geom, tr, pe, seed, maps):
    L = AdvLayout(geom)
    sca, mom = geom.kind in ("scalar", "w"), geom.kind in ("u", "v")
    flags = int(not geom.open_x) | (int(not geom.open_y) << 1) | (int(geom.spec) << 2)
    if geom.open_x and not geom.spec:
        flags |= (1 << 4) | (1 << 5)
    if geom.open_y and not geom.spec:
        flags |= (1 << 6) | (1 << 7)
    h, v = geom.horder, geom.vorder
    head = np.array([ADV_MODE[geom.kind], geom.nx, geom.ny, geom.nz, flags,
                     h if mom else 5, v if mom else 3, h if sca else 5,
                     v if sca else 3], dtype=np.int32)
    blob = head.tobytes() + np.array([geom.rdx, geom.rdy]).astype("<f8").tobytes()
    a3 = [L.pad3(tr["q"], False), L.pad3(pe["q"], False), L.pad3(tr["ru"]), L.pad3(pe["ru"]),
          L.pad3(tr["rv"]), L.pad3(pe["rv"]), L.pad3(tr["rw"]), L.pad3(pe["rw"]),
          L.place3(tr["tend"]), L.place3(seed)]
    for a in a3:
        blob += np.asarray(a, dtype="<f8").tobytes(order="F")
    a2 = [np.full(L.s2, 9.0e4), L.pad2(maps["u"]), L.pad2(maps["u"]), L.pad2(maps["v"]),
          L.pad2(maps["v"]), L.pad2(maps["t"]), L.pad2(maps["t"])]
    for a in a2:
        blob += np.asarray(a, dtype="<f8").tobytes(order="F")
    for x in (tr["fnm"], tr["fnp"], tr["rdz"]):
        blob += np.concatenate([x, [0.0]]).astype("<f8").tobytes()
    return blob, L


def _adv_expected(geom, out, L):
    from woof.grad.wrfplus.advect import adv_shapes
    sh = adv_shapes(geom)
    n3 = int(np.prod(L.s3))
    vals = np.frombuffer(out, dtype="<f8")
    assert vals.size == 6 * n3, (vals.size, n3)
    x = [vals[i * n3:(i + 1) * n3].reshape(L.s3, order="F") for i in range(6)]
    return {"tl_tend": L.take3(x[0], sh["tend"]), "tl_d_tend": L.take3(x[1], sh["tend"]),
            "ad_q": L.fold3(x[2], sh["q"], False), "ad_ru": L.fold3(x[3], sh["ru"]),
            "ad_rv": L.fold3(x[4], sh["rv"]), "ad_rw": L.fold3(x[5], sh["rw"])}


def advect(exe: str, runner: list[str], seed: int = 20261125) -> dict:
    out = {}
    for n, (name, over) in enumerate(ADV_CASES.items()):
        over = dict(over)
        kind = over.pop("kind")
        geom = cases.adv_geometry(kind, **ADV_DIMS, **over)
        geom.validate()
        tr = cases.adv_trajectory(geom, seed + 10 * n)
        pe = cases.adv_perturbation(geom, seed + 10 * n + 1)
        rng = np.random.default_rng(seed + 10 * n + 2)
        nz, ny, nx = geom.nz, geom.ny, geom.nx
        maps = {"u": 1.0 + 0.02 * rng.standard_normal((ny, nx + 1)),
                "v": 1.0 + 0.02 * rng.standard_normal((ny + 1, nx)),
                "t": 1.0 + 0.02 * rng.standard_normal((ny, nx))}
        tr["msf"] = maps[{"scalar": "t", "w": "t", "u": "u", "v": "v"}[kind]]
        # A periodic u (v) reads the mass flux at the end face through both
        # copies (woof's column n, WRF's halo column 0): make them one
        # value; the test compares their adjoints' sum.
        for d in (tr, pe):
            if kind == "u" and not geom.open_x:
                d["ru"][:, :, -1] = d["ru"][:, :, 0]
            if kind == "v" and not geom.open_y:
                d["rv"][:, -1, :] = d["rv"][:, 0, :]
        mask = adv_mask(geom)
        at = np.where(mask, rng.standard_normal(mask.shape), 0.0)
        blob, L = _adv_record(geom, tr, pe, at, maps)
        with tempfile.TemporaryDirectory(dir=Path(exe).parent) as tmp:
            fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            fin.write_bytes(blob)
            subprocess.run(runner + [exe, str(fin), str(fout)], check=True)
            result = fout.read_bytes()
        exp = _adv_expected(geom, result, L)
        prefix = f"{name}/"
        out[prefix + "geometry"] = np.array(json.dumps(dataclasses.asdict(geom)))
        out[prefix + "mask"] = mask
        out[prefix + "seed/tend"] = at
        for k, v in tr.items():
            out[prefix + "traj/" + k] = v
        for k, v in pe.items():
            out[prefix + "pert/" + k] = v
        for k, v in exp.items():
            out[prefix + "wrfplus/" + k] = v
    return out


#: bigstep: horizontal_pressure_gradient on periodic, specified and open
#: axes, with and without the lid extrapolation, and pg_buoy_w with and
#: without map factors.  WRFPLUS's versions are not hybrid (c1 = 1, c2 = 0).
BS_CASES = {
    "pgf_periodic": dict(op="pgf"),
    "pgf_spec_lid": dict(op="pgf", boundary_x=1, boundary_y=1, top_lid=1, spec=1),
    "pgf_open_x": dict(op="pgf", boundary_x=1),
    "pgf_open_y_lid": dict(op="pgf", boundary_y=1, top_lid=1),
    "buoy": dict(op="buoy"),
    "buoy_msf": dict(op="buoy", has_msf=1),
}
BS_DIMS = dict(nz=8, ny=9, nx=10)


def _bs_faces(m, axis, n):
    """calc_mu_uv's face values 0.5 * (m_a + m_b) on the periodic face
    index (faces 0..n along ``axis`` of a (ny, nx) column field)."""
    f = np.arange(n + 1)
    a, b = f % n, (f - 1) % n
    return 0.5 * (np.take(m, a, axis=axis) + np.take(m, b, axis=axis))


def _bs_faces_t(af, axis, n):
    out = np.zeros(af.shape[:axis] + (n,) + af.shape[axis + 1:])
    f = np.arange(n + 1)
    for idx, cells in ((f, f % n), (f, (f - 1) % n)):
        for q, c in zip(idx, cells):
            sl = [slice(None)] * af.ndim
            so = [slice(None)] * af.ndim
            sl[axis], so[axis] = q, c
            out[tuple(so)] += 0.5 * af[tuple(sl)]
    return out


def _bs_fold2(L, x, shape):
    _, yi, xi = L._maps((1,) + shape)
    out = np.zeros(shape)
    np.add.at(out, np.ix_(yi, xi), x.T)
    return out


def _bs_take2(L, x, shape):
    # bigstep.F90 dimensions 2-D fields from ims = jms = -3, so woof's
    # (0, 0) sits at offset 4 on both axes.
    nj, ni = shape
    return np.ascontiguousarray(x[4:4 + ni, 4:4 + nj].T)


def _bs_record(op, geom, tr, pe, seeds, spec):
    from types import SimpleNamespace as NS
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    if op == "pgf":
        # WRFPLUS's pressure gradient is not hybrid: it has no c1h/c2h.
        assert np.all(tr["c1h"] == 1) and not np.any(tr["c2h"]), "oracle needs c1h=1, c2h=0"
    bx, by = (geom.boundary_x, geom.boundary_y) if op == "pgf" else (0, 0)
    L = AdvLayout(NS(nz=nz, ny=ny, nx=nx, open_x=bx, open_y=by))
    flags = int(not bx) | (int(not by) << 1) | (int(bool(spec)) << 2)
    if bx and not spec:
        flags |= (1 << 4) | (1 << 5)
    if by and not spec:
        flags |= (1 << 6) | (1 << 7)
    lid = getattr(geom, "top_lid", 0)
    head = np.array([1 if op == "pgf" else 2, nx, ny, nz, flags, lid], dtype=np.int32)
    g = geom if op == "pgf" else None
    sc = [g.rdx, g.rdy, g.cf1, g.cf2, g.cf3, g.cfn, g.cfn1] if g else [0.0] * 7
    blob = head.tobytes() + np.array(sc).astype("<f8").tobytes()
    z3 = np.zeros((nz, ny, nx))
    zw = np.zeros((nz + 1, ny, nx))
    if op == "pgf":
        half = 0.5 * (tr["phb"][:-1] + tr["phb"][1:] + tr["php"][:-1] + tr["php"][1:])
        halfd = 0.5 * (pe["php"][:-1] + pe["php"][1:])
        mut = tr["mup"] + tr["mub"]
        muu, muud = _bs_faces(mut, 1, nx), _bs_faces(pe["mup"], 1, nx)
        muv, muvd = _bs_faces(mut, 0, ny), _bs_faces(pe["mup"], 0, ny)
        a3 = [tr["php"], pe["php"], tr["alt"], pe["alt"], tr["p"], pe["p"], tr["pb"],
              tr["al"], pe["al"], half, halfd]
        a3 = [L.pad3(x) for x in a3]
        a3 += [L.place3(tr["ru_t"]), L.place3(seeds["ru_t"]), L.place3(tr["rv_t"]),
               L.place3(seeds["rv_t"]), L.place3(zw), L.place3(zw)]
        a2 = [L.pad2(muu), L.pad2(muud), L.pad2(muv), L.pad2(muvd), L.pad2(tr["mup"]),
              L.pad2(pe["mup"]), L.pad2(tr["mub"]), np.ones(L.s2)]
        lev = [tr["fnm"], tr["fnp"], tr["rdnw"], np.zeros(nz)]
    else:
        a3 = [L.pad3(zw), L.pad3(zw), L.pad3(z3), L.pad3(z3), L.pad3(tr["p"]), L.pad3(pe["p"]),
              L.pad3(z3), L.pad3(z3), L.pad3(z3), L.pad3(z3), L.pad3(z3)]
        a3 += [L.place3(np.zeros((nz, ny, nx + 1))), L.place3(np.zeros((nz, ny, nx + 1))),
               L.place3(np.zeros((nz, ny + 1, nx))), L.place3(np.zeros((nz, ny + 1, nx))),
               L.place3(tr["rw_t"]), L.place3(seeds["rw_t"])]
        msft = tr["msft"] if geom.has_msf else np.ones((ny, nx))
        a2 = [np.zeros(L.s2)] * 4 + [L.pad2(tr["mup"]), L.pad2(pe["mup"]), np.zeros(L.s2),
                                     L.pad2(msft)]
        lev = [np.zeros(nz), np.zeros(nz), tr["rdnw"], tr["rdn"]]
    for a in a3 + a2:
        blob += np.asarray(a, dtype="<f8").tobytes(order="F")
    for x in lev:
        blob += np.concatenate([x, [0.0]]).astype("<f8").tobytes()
    return blob, L


def _bs_expected(op, geom, out, L):
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    n3, n2 = int(np.prod(L.s3)), int(np.prod(L.s2))
    vals = np.frombuffer(out, dtype="<f8")
    assert vals.size == 11 * n3 + 3 * n2, (vals.size, n3, n2)
    x = [vals[i * n3:(i + 1) * n3].reshape(L.s3, order="F") for i in range(11)]
    y = [vals[11 * n3 + i * n2:11 * n3 + (i + 1) * n2].reshape(L.s2, order="F")
         for i in range(3)]
    mass, wlev = (nz, ny, nx), (nz + 1, ny, nx)
    if op == "buoy":
        return {"tl_rw_t": L.take3(x[4], wlev), "tl_d_rw_t": L.take3(x[5], wlev),
                "ad_p": L.fold3(x[8], mass), "ad_mup": _bs_fold2(L, y[2], (ny, nx))}
    aphp = L.fold3(x[6], wlev)
    ahalf = L.fold3(x[10], mass)
    aphp[:-1] += 0.5 * ahalf
    aphp[1:] += 0.5 * ahalf
    amup = (_bs_fold2(L, y[2], (ny, nx))
            + _bs_faces_t(_bs_take2(L, y[0], (ny, nx + 1)), 1, nx)
            + _bs_faces_t(_bs_take2(L, y[1], (ny + 1, nx)), 0, ny))
    return {"tl_ru_t": L.take3(x[0], (nz, ny, nx + 1)),
            "tl_d_ru_t": L.take3(x[1], (nz, ny, nx + 1)),
            "tl_rv_t": L.take3(x[2], (nz, ny + 1, nx)),
            "tl_d_rv_t": L.take3(x[3], (nz, ny + 1, nx)),
            "ad_php": aphp, "ad_alt": L.fold3(x[7], mass), "ad_p": L.fold3(x[8], mass),
            "ad_al": L.fold3(x[9], mass), "ad_mup": amup}


def bigstep(exe: str, runner: list[str], seed: int = 20261126) -> dict:
    from woof.grad.wrfplus import bigstep as B
    out = {}
    for n, (name, over) in enumerate(BS_CASES.items()):
        over = dict(over)
        op, spec = over.pop("op"), over.pop("spec", 0)
        if op == "pgf":
            geom = cases.pgf_geometry(**BS_DIMS, **over)
            tr = cases.pgf_trajectory(geom, seed + 10 * n)
            pe = cases.pgf_perturbation(geom, seed + 10 * n + 1)
            sh, states = B.pgf_shapes(geom), B.PGF_STATE
        else:
            geom = cases.buoy_geometry(**BS_DIMS, **over)
            tr = cases.buoy_trajectory(geom, seed + 10 * n)
            pe = cases.buoy_perturbation(geom, seed + 10 * n + 1)
            sh, states = B.buoy_shapes(geom), B.BUOY_STATE
        geom.validate()
        rng = np.random.default_rng(seed + 10 * n + 2)
        seeds = {s: rng.standard_normal(sh[s]) for s in states}
        blob, L = _bs_record(op, geom, tr, pe, seeds, spec)
        with tempfile.TemporaryDirectory(dir=Path(exe).parent) as tmp:
            fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            fin.write_bytes(blob)
            subprocess.run(runner + [exe, str(fin), str(fout)], check=True)
            result = fout.read_bytes()
        exp = _bs_expected(op, geom, result, L)
        prefix = f"{name}/"
        out[prefix + "op"] = np.array(op)
        out[prefix + "geometry"] = np.array(json.dumps(dataclasses.asdict(geom)))
        for k, v in seeds.items():
            out[prefix + "seed/" + k] = v
        for k, v in tr.items():
            out[prefix + "traj/" + k] = v
        for k, v in pe.items():
            out[prefix + "pert/" + k] = v
        for k, v in exp.items():
            out[prefix + "wrfplus/" + k] = v
    return out


#: rhsph: rhs_ph for each h_sca_adv_order on periodic, specified and open
#: axes, with and without map factors.  WRFPLUS's rhs_ph is not hybrid
#: (c1f = 1, c2f = 0).  The order-3/4 cases on a periodic y or specified
#: x axis are TL only: WRFPLUS's a_rhs_ph overruns its trajectory arrays
#: there (rhsph.F90's header).
RP_CASES = {
    "geo_periodic": dict(),
    "geo_spec_msf": dict(specified=1, has_msf=1),
    "geo_open_x": dict(open_x=1),
    "geo_open_y_msf": dict(open_y=1, has_msf=1),
    "geo_order6": dict(order=6),
    "geo_order3": dict(order=3, tl_only=1),
    "geo_order4_open": dict(order=4, open_x=1, open_y=1, has_msf=1),
    "geo_order2_open": dict(order=2, open_x=1, open_y=1),
    "geo_spec_o3": dict(order=3, specified=1, tl_only=1),
    "geo_spec_o4_msf": dict(order=4, specified=1, has_msf=1, tl_only=1),
}
RP_DIMS = dict(nz=8, ny=9, nx=10)


def rp_mask(geom):
    """rph_t points WRFPLUS's g_/a_rhs_ph treat as woof (and WRF's own
    rhs_ph) do.  For h_sca_adv_order <= 4 WRFPLUS ends the interior loop of
    a bounded axis at min(ite, ide-1) - 2 (order 2) or - 3 (orders 3-4)
    where rhs_ph ends it at ite - 2 or ite - 3, one point earlier with the
    tile reaching ide: it drops the term at woof's pos n-2 (order 2) or
    n-3 (orders 3-4).  For orders 3-4 the x loop does this on open_xe
    only, as both versions move the x end only there."""
    from woof.grad.wrfplus.bigstep import geo_shapes
    m = np.ones(geo_shapes(geom)["rph_t"], dtype=bool)
    if geom.order <= 4:
        back = 2 if geom.order <= 2 else 3
        if geom.boundary_y:
            m[:, geom.ny - back, :] = False
        if geom.boundary_x and (geom.order <= 2 or not geom.specified):
            m[:, :, geom.nx - back] = False
    return m


def _rp_pad2(L, a, end):
    _, yi, xi = L._maps((1,) + a.shape, end)
    return a[np.ix_(yi, xi)].T


def _rp_fold2(L, x, shape, end):
    _, yi, xi = L._maps((1,) + shape, end)
    out = np.zeros(shape)
    np.add.at(out, np.ix_(yi, xi), x.T)
    return out


def _rp_record(geom, tr, pe, seed, tl_only=0):
    """rhsph.F90's input.  Every halo wraps the periodic woof field, open
    and specified axes included: woof's strict slow_geopotential reads the
    wrapped field wherever a stencil leaves the domain (only the specified
    order-3/4 x edges do), so WRF then reads the same values.  The u (v)
    end face keeps woof's own index n on a bounded axis, where woof reads
    it, and wraps to 0 on a periodic one."""
    from types import SimpleNamespace as NS
    assert np.all(tr["c1f"] == 1) and not np.any(tr["c2f"]), "oracle needs c1f=1, c2f=0"
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    bx, by = geom.boundary_x, geom.boundary_y
    L = AdvLayout(NS(nz=nz, ny=ny, nx=nx, open_x=0, open_y=0))
    flags = int(not bx) | (int(not by) << 1) | (int(bool(geom.specified)) << 2)
    if geom.open_x and not geom.specified:
        flags |= (1 << 4) | (1 << 5)
    if geom.open_y and not geom.specified:
        flags |= (1 << 6) | (1 << 7)
    flags |= int(bool(tl_only)) << 8
    head = np.array([nx, ny, nz, flags, geom.order], dtype=np.int32)
    sc = [4.0 * geom.quarter_rdx, 4.0 * geom.quarter_rdy, geom.cfn, geom.cfn1]
    blob = head.tobytes() + np.array(sc).astype("<f8").tobytes()
    mut = tr["mup"] + tr["mub"]
    ones = lambda n: np.ones(tr[n].shape)
    msf = {n: (tr[n] if geom.has_msf else ones(n)) for n in ("msft", "msfu", "msfv")}
    a3 = [L.place3(tr["rph_t"]), L.place3(seed["rph_t"]),
          L.pad3(tr["u"], end=bx), L.pad3(pe["u"], end=bx),
          L.pad3(tr["v"], end=by), L.pad3(pe["v"], end=by),
          L.pad3(tr["ww"]), L.pad3(pe["ww"]), L.pad3(tr["php"]), L.pad3(pe["php"]),
          L.pad3(tr["phb"]), L.pad3(tr["w"]), L.pad3(pe["w"])]
    a2 = [L.pad2(mut), L.pad2(pe["mup"]),
          _rp_pad2(L, _bs_faces(mut, 1, nx), bx), _rp_pad2(L, _bs_faces(pe["mup"], 1, nx), bx),
          _rp_pad2(L, _bs_faces(mut, 0, ny), by), _rp_pad2(L, _bs_faces(pe["mup"], 0, ny), by),
          _rp_pad2(L, msf["msfu"], bx), _rp_pad2(L, msf["msfv"], by), L.pad2(msf["msft"])]
    for a in a3 + a2:
        blob += np.asarray(a, dtype="<f8").tobytes(order="F")
    for n in ("fnm", "fnp", "rdnw"):
        blob += np.concatenate([tr[n], [0.0]]).astype("<f8").tobytes()
    return blob, L


def _rp_expected(geom, out, L, tl_only=0):
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    bx, by = geom.boundary_x, geom.boundary_y
    n3, n2 = int(np.prod(L.s3)), int(np.prod(L.s2))
    vals = np.frombuffer(out, dtype="<f8")
    assert vals.size == 9 * n3 + 3 * n2, (vals.size, n3, n2)
    x = [vals[i * n3:(i + 1) * n3].reshape(L.s3, order="F") for i in range(9)]
    y = [vals[9 * n3 + i * n2:9 * n3 + (i + 1) * n2].reshape(L.s2, order="F")
         for i in range(3)]
    wlev = (nz + 1, ny, nx)
    amup = (_rp_fold2(L, y[0], (ny, nx), True)
            + _bs_faces_t(_rp_fold2(L, y[1], (ny, nx + 1), bx), 1, nx)
            + _bs_faces_t(_rp_fold2(L, y[2], (ny + 1, nx), by), 0, ny))
    tl = {"tl_rph_t": L.take3(x[0], wlev), "tl_d_rph_t": L.take3(x[1], wlev)}
    if tl_only:
        return tl
    return {**tl, "adst_rph_t": L.take3(x[2], wlev),
            "ad_u": L.fold3(x[3], (nz, ny, nx + 1), end=bx),
            "ad_v": L.fold3(x[4], (nz, ny + 1, nx), end=by),
            "ad_ww": L.fold3(x[5], wlev), "ad_php": L.fold3(x[6], wlev) + L.fold3(x[7], wlev),
            "ad_w": L.fold3(x[8], wlev), "ad_mup": amup}


def rhsph(exe: str, runner: list[str], seed: int = 20261127) -> dict:
    from woof.grad.wrfplus import bigstep as B
    out = {}
    for n, (name, over) in enumerate(RP_CASES.items()):
        over = dict(over)
        tl_only = over.pop("tl_only", 0)
        geom = cases.geo_geometry(**RP_DIMS, **over)
        geom.validate()
        tr = cases.geo_trajectory(geom, seed + 10 * n)
        pe = cases.geo_perturbation(geom, seed + 10 * n + 1)
        rng = np.random.default_rng(seed + 10 * n + 2)
        mask = rp_mask(geom)
        # Seeds vanish where WRFPLUS drops a term, so both adjoints see
        # the same operator; the TL is compared on the mask.
        seeds = {"rph_t": rng.standard_normal(B.geo_shapes(geom)["rph_t"]) * mask}
        blob, L = _rp_record(geom, tr, pe, seeds, tl_only)
        with tempfile.TemporaryDirectory(dir=Path(exe).parent) as tmp:
            fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            fin.write_bytes(blob)
            subprocess.run(runner + [exe, str(fin), str(fout)], check=True)
            result = fout.read_bytes()
        exp = _rp_expected(geom, result, L, tl_only)
        prefix = f"{name}/"
        out[prefix + "op"] = np.array("geo")
        out[prefix + "mask"] = mask
        out[prefix + "geometry"] = np.array(json.dumps(dataclasses.asdict(geom)))
        for k, v in seeds.items():
            out[prefix + "seed/" + k] = v
        for k, v in tr.items():
            out[prefix + "traj/" + k] = v
        for k, v in pe.items():
            out[prefix + "pert/" + k] = v
        for k, v in exp.items():
            out[prefix + "wrfplus/" + k] = v
    return out


#: coriolis: WRF's coriolis then curvature (map_proj 1) on periodic, open
#: and specified axes.  WRFPLUS takes the coupled rw; the fixture forms it
#: from w and mut (hybrid c1f/c2f), with its tangent and adjoint.
CC_CASES = {
    "cc_periodic": dict(hybrid=1),
    "cc_open_x": dict(boundary_x=1),
    "cc_open_y": dict(boundary_y=1, hybrid=1),
    "cc_open_xy": dict(boundary_x=1, boundary_y=1),
    "cc_spec": dict(boundary_x=1, boundary_y=1, spec=1, hybrid=1),
    # f = e = 0: curvature alone, so binary32 (Metal) is held on its scale
    "cc_curvature": dict(nocor=1),
}
CC_DIMS = dict(nz=8, ny=9, nx=10)


def cc_periodic_ends(geom, d):
    """A periodic axis's end face repeats face 0 (WRF reads its halo copy
    of face 0 where woof reads its own face n): make them one value."""
    for n in ("ru", "u", "msfu"):
        if n in d and not geom.boundary_x:
            d[n][..., -1] = d[n][..., 0]
    for n in ("rv", "v", "msfv"):
        if n in d and not geom.boundary_y:
            d[n][..., -1, :] = d[n][..., 0, :]


def _cc_rw(tr, pe=None):
    c1, c2 = tr["c1f"][:, None, None], tr["c2f"][:, None, None]
    mass = c1 * tr["mut"] + c2
    if pe is None:
        return mass * tr["w"] / tr["msft"]
    return (c1 * pe["mut"] * tr["w"] + mass * pe["w"]) / tr["msft"]


def _cc_record(geom, tr, pe, seeds, spec):
    from types import SimpleNamespace as NS
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    bx, by = geom.boundary_x, geom.boundary_y
    L = AdvLayout(NS(nz=nz, ny=ny, nx=nx, open_x=0, open_y=0))
    flags = int(not bx) | (int(not by) << 1) | (int(bool(spec)) << 2)
    if bx and not spec:
        flags |= (1 << 4) | (1 << 5)
    if by and not spec:
        flags |= (1 << 6) | (1 << 7)
    head = np.array([nx, ny, nz, flags], dtype=np.int32)
    blob = head.tobytes() + np.array([geom.rdx, geom.rdy]).astype("<f8").tobytes()
    a3 = [L.place3(tr["ru_t"]), L.place3(seeds["ru_t"]), L.place3(tr["rv_t"]),
          L.place3(seeds["rv_t"]), L.place3(tr["rw_t"]), L.place3(seeds["rw_t"]),
          L.pad3(tr["ru"], end=bx), L.pad3(pe["ru"], end=bx),
          L.pad3(tr["rv"], end=by), L.pad3(pe["rv"], end=by),
          L.pad3(_cc_rw(tr)), L.pad3(_cc_rw(tr, pe)),
          L.pad3(tr["u"], end=bx), L.pad3(pe["u"], end=bx),
          L.pad3(tr["v"], end=by), L.pad3(pe["v"], end=by)]
    a2 = [_rp_pad2(L, tr["msfu"], bx), _rp_pad2(L, tr["msfv"], by)] + [
        L.pad2(tr[n]) for n in ("msft", "f", "e", "sina", "cosa")]
    for a in a3 + a2:
        blob += np.asarray(a, dtype="<f8").tobytes(order="F")
    for n in ("fnm", "fnp"):
        blob += np.concatenate([tr[n], [0.0]]).astype("<f8").tobytes()
    return blob, L


def _cc_expected(geom, out, L, tr):
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    bx, by = geom.boundary_x, geom.boundary_y
    n3 = int(np.prod(L.s3))
    vals = np.frombuffer(out, dtype="<f8")
    assert vals.size == 14 * n3, (vals.size, n3)
    x = [vals[i * n3:(i + 1) * n3].reshape(L.s3, order="F") for i in range(14)]
    us, vs, wl = (nz, ny, nx + 1), (nz, ny + 1, nx), (nz + 1, ny, nx)
    arw = L.fold3(x[11], wl)
    c1, c2 = tr["c1f"][:, None, None], tr["c2f"][:, None, None]
    return {"tl_ru_t": L.take3(x[0], us), "tl_d_ru_t": L.take3(x[1], us),
            "tl_rv_t": L.take3(x[2], vs), "tl_d_rv_t": L.take3(x[3], vs),
            "tl_rw_t": L.take3(x[4], wl), "tl_d_rw_t": L.take3(x[5], wl),
            "adst_ru_t": L.take3(x[6], us), "adst_rv_t": L.take3(x[7], vs),
            "adst_rw_t": L.take3(x[8], wl),
            # ru, rv: a periodic end face keeps WRF's column ide (jde), read
            # where woof reads face nx (ny), so faces 0 and n are compared
            # apart.  u, v: WRF and woof split their reads of the repeated
            # face differently between 0 and n, so face n folds onto 0 and
            # the test sums woof's two faces.
            "ad_ru": L.fold3(x[9], us), "ad_rv": L.fold3(x[10], vs),
            "ad_u": L.fold3(x[12], us, end=bx), "ad_v": L.fold3(x[13], vs, end=by),
            "ad_w": (c1 * tr["mut"] + c2) * arw / tr["msft"],
            "ad_mut": (c1 * tr["w"] * arw).sum(axis=0) / tr["msft"]}


def coriolis(exe: str, runner: list[str], seed: int = 20261128) -> dict:
    from woof.grad.wrfplus import bigstep as B
    out = {}
    for n, (name, over) in enumerate(CC_CASES.items()):
        over = dict(over)
        spec, hybrid = over.pop("spec", 0), over.pop("hybrid", 0)
        nocor = over.pop("nocor", 0)
        geom = cases.cc_geometry(**CC_DIMS, **over)
        geom.validate()
        tr = cases.cc_trajectory(geom, seed + 10 * n, hybrid=bool(hybrid))
        if nocor:
            tr["f"][...] = 0.0
            tr["e"][...] = 0.0
        pe = cases.cc_perturbation(geom, seed + 10 * n + 1)
        cc_periodic_ends(geom, tr)
        cc_periodic_ends(geom, pe)
        rng = np.random.default_rng(seed + 10 * n + 2)
        sh = B.cc_shapes(geom)
        seeds = {s: rng.standard_normal(sh[s]) for s in B.CC_STATE}
        blob, L = _cc_record(geom, tr, pe, seeds, spec)
        with tempfile.TemporaryDirectory(dir=Path(exe).parent) as tmp:
            fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            fin.write_bytes(blob)
            subprocess.run(runner + [exe, str(fin), str(fout)], check=True)
            result = fout.read_bytes()
        exp = _cc_expected(geom, result, L, tr)
        prefix = f"{name}/"
        out[prefix + "op"] = np.array("cc")
        out[prefix + "geometry"] = np.array(json.dumps(dataclasses.asdict(geom)))
        for k, v in seeds.items():
            out[prefix + "seed/" + k] = v
        for k, v in tr.items():
            out[prefix + "traj/" + k] = v
        for k, v in pe.items():
            out[prefix + "pert/" + k] = v
        for k, v in exp.items():
            out[prefix + "wrfplus/" + k] = v
    return out


#: w_damp: WRF's w_damping = 1 limiter.  WRFPLUS couples with the column
#: mass mut and uses w_beta = 1 as onset and excess reference, so the
#: cases use c1f = 1, c2f = 0 and w_damp_on = w_crit_cfl = 1.
WD_CASES = {
    "wd_a": dict(),
    "wd_b": dict(dt=30.0),
    "wd_c": dict(nz=3, ny=5, nx=4, dt=12.0),
}
WD_DIMS = dict(nz=8, ny=9, nx=10)


def _wd_record(geom, tr, pe, seeds):
    from types import SimpleNamespace as NS
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    L = AdvLayout(NS(nz=nz, ny=ny, nx=nx, open_x=0, open_y=0))
    blob = np.array([nx, ny, nz], dtype=np.int32).tobytes()
    blob += np.array([geom.dt]).astype("<f8").tobytes()
    mut = tr["mub"] + tr["mup"]
    a3 = [L.place3(tr["rw_t"]), L.place3(seeds["rw_t"]), L.place3(tr["ww"]),
          L.place3(pe["ww"]), L.place3(tr["w"])]
    for a in a3 + [L.pad2(mut), L.pad2(pe["mup"])]:
        blob += np.asarray(a, dtype="<f8").tobytes(order="F")
    blob += np.concatenate([tr["rdnw"], [0.0]]).astype("<f8").tobytes()
    return blob, L


def _wd_expected(geom, out, L):
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    n3, n2 = int(np.prod(L.s3)), int(np.prod(L.s2))
    vals = np.frombuffer(out, dtype="<f8")
    assert vals.size == 4 * n3 + n2, (vals.size, n3, n2)
    x = [vals[i * n3:(i + 1) * n3].reshape(L.s3, order="F") for i in range(4)]
    m2 = vals[4 * n3:].reshape(L.s2, order="F")
    wl = (nz + 1, ny, nx)
    return {"tl_rw_t": L.take3(x[0], wl), "tl_d_rw_t": L.take3(x[1], wl),
            "adst_rw_t": L.take3(x[2], wl), "ad_ww": L.take3(x[3], wl),
            "ad_mup": np.ascontiguousarray(m2[4:4 + nx, 4:4 + ny].T)}


def w_damp(exe: str, runner: list[str], seed: int = 20261129) -> dict:
    from woof.grad.wrfplus import bigstep as B
    out = {}
    for n, (name, over) in enumerate(WD_CASES.items()):
        dims = dict(WD_DIMS)
        dims.update(over)
        geom = cases.wd_geometry(**dims)
        geom.validate()
        tr = cases.wd_trajectory(geom, seed + 10 * n)
        pe = cases.wd_perturbation(geom, seed + 10 * n + 1)
        rng = np.random.default_rng(seed + 10 * n + 2)
        seeds = {"rw_t": rng.standard_normal(B.wd_shapes(geom)["rw_t"])}
        blob, L = _wd_record(geom, tr, pe, seeds)
        with tempfile.TemporaryDirectory(dir=Path(exe).parent) as tmp:
            fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            fin.write_bytes(blob)
            subprocess.run(runner + [exe, str(fin), str(fout)], check=True)
            result = fout.read_bytes()
        exp = _wd_expected(geom, result, L)
        prefix = f"{name}/"
        out[prefix + "op"] = np.array("wd")
        out[prefix + "geometry"] = np.array(json.dumps(dataclasses.asdict(geom)))
        for k, v in seeds.items():
            out[prefix + "seed/" + k] = v
        for k, v in tr.items():
            out[prefix + "traj/" + k] = v
        for k, v in pe.items():
            out[prefix + "pert/" + k] = v
        for k, v in exp.items():
            out[prefix + "wrfplus/" + k] = v
    return out


#: lbc: relax_bdytend then spec_bdytend on one coupled field, as woof's
#: direct-form row (apply_relax + target).  WRFPLUS takes the coupled
#: field; the fixture forms it from the row's field and mu' (hybrid
#: levels, map factors) with its tangent and adjoint.
LB_CASES = {
    "lb_u": dict(kind="u", has_msf=1),
    "lb_u_nomsf": dict(kind="u", hybrid=1),
    "lb_v": dict(kind="v", has_msf=1, hybrid=1),
    "lb_theta": dict(kind="theta", hybrid=1),
    "lb_phi": dict(kind="phi"),
    "lb_w": dict(kind="w", hybrid=1),
    "lb_mu": dict(kind="mu"),
    "lb_wide": dict(kind="theta", spec_zone=2, relax_zone=5, width=7, ny=14, nx=15),
}
LB_DIMS = dict(nz=4, ny=12, nx=13)
_LB_VAR = {"u": 0, "v": 1, "theta": 2, "phi": 3, "w": 3, "mu": 4}


def _lb_faces(geom, tr):
    """(ia, ib) mass columns of each u or v face (WRF calc_mu_uv, edge
    columns doubled)."""
    ny, nx = geom.ny, geom.nx
    if geom.kind == "u":
        i = np.arange(nx + 1)
        jj = np.arange(ny)[:, None]
        return jj * nx + np.minimum(i, nx - 1)[None], jj * nx + np.maximum(i - 1, 0)[None]
    j = np.arange(ny + 1)
    ii = np.arange(nx)[None]
    return np.minimum(j, ny - 1)[:, None] * nx + ii, np.maximum(j - 1, 0)[:, None] * nx + ii


def _lb_coupled(geom, tr, pe=None, adj=None):
    """woof's strict coupled_current: the value, its tangent (``pe``) or
    the (src, mup) adjoint of a coupled-field adjoint ``adj``."""
    kind = geom.kind
    mub, mup = tr["mub"].ravel(), tr["mup"].ravel()
    src = tr["src"]
    if kind == "mu":
        if adj is not None:
            return np.zeros_like(src), adj[0].copy()
        return (pe["mup"] if pe is not None else tr["mup"])[None].copy()
    if kind in ("u", "v"):
        a, b = _lb_faces(geom, tr)
        mass = 0.5 * (((mup[a] + mup[b]) + mub[a]) + mub[b])
        c1, c2 = tr["c1h"][:, None, None], tr["c2h"][:, None, None]
        scale = 1.0
        if geom.has_msf:
            scale = 1.0 / tr["msfu"] if kind == "u" else tr["msfv"]
        ch = c1 * mass + c2
        if adj is not None:
            t = adj * scale
            amass = (c1 * src * t).sum(axis=0)
            amup = np.zeros(mup.size)
            np.add.at(amup, a.ravel(), 0.5 * amass.ravel())
            np.add.at(amup, b.ravel(), 0.5 * amass.ravel())
            return ch * t, amup.reshape(tr["mup"].shape)
        if pe is None:
            return ch * src * scale
        dmass = 0.5 * (pe["mup"].ravel()[a] + pe["mup"].ravel()[b])
        return (c1 * dmass * src + ch * pe["src"]) * scale
    full = kind in ("phi", "w")
    c1 = (tr["c1f"] if full else tr["c1h"])[:, None, None]
    c2 = (tr["c2f"] if full else tr["c2h"])[:, None, None]
    mass = tr["mub"] + tr["mup"]
    ch = c1 * mass + c2
    if adj is not None:
        return ch * adj, (c1 * src * adj).sum(axis=0)
    if pe is None:
        return ch * src
    return c1 * pe["mup"] * src + ch * pe["src"]


def _lb_record(geom, tr, pe, seeds):
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    fz, fy, fx = geom.field
    nk = 1 if geom.kind == "mu" else nz + 1
    w = geom.width
    head = np.array([nx, ny, nz, _LB_VAR[geom.kind], w, geom.spec_zone, geom.relax_zone, nk],
                    dtype=np.int32)
    blob = head.tobytes() + np.array([geom.dtbc]).astype("<f8").tobytes()
    blob += np.asarray(tr["fcx"], "<f8").tobytes() + np.asarray(tr["gcx"], "<f8").tobytes()

    def f3(a):
        out = np.zeros((nx + 8, nk, ny + 8))
        out[4:4 + fx, :fz, 4:4 + fy] = np.transpose(a, (2, 0, 1))
        return out

    def tx(a):  # (fz, fy, w) -> (jms:jme, nk, w)
        out = np.zeros((ny + 8, nk, w))
        out[4:4 + fy, :fz, :] = np.transpose(a, (1, 0, 2))
        return out

    def ty(a):  # (fz, w, fx) -> (ims:ime, nk, w)
        out = np.zeros((nx + 8, nk, w))
        out[4:4 + fx, :fz, :] = np.transpose(a, (2, 0, 1))
        return out

    arrs = [f3(_lb_coupled(geom, tr)), f3(_lb_coupled(geom, tr, pe)), f3(tr["tend"]),
            f3(seeds["tend"])]
    for d in (tr, pe):
        arrs += [tx(d["west"]), tx(d["east"]), ty(d["south"]), ty(d["north"]),
                 tx(d["west_t"]), tx(d["east_t"]), ty(d["south_t"]), ty(d["north_t"])]
    for a in arrs:
        blob += np.asarray(a, dtype="<f8").tobytes(order="F")
    return blob, (f3, tx, ty, nk)


def _lb_expected(geom, out, L, tr, seeds):
    f3, tx, ty, nk = L
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    fz, fy, fx = geom.field
    w = geom.width
    vals = np.frombuffer(out, dtype="<f8")
    n3, nxs, nys = (nx + 8) * nk * (ny + 8), (ny + 8) * nk * w, (nx + 8) * nk * w
    assert vals.size == 4 * n3 + 4 * nxs + 4 * nys, vals.size
    at = 0
    x3 = []
    for _ in range(4):
        x3.append(vals[at:at + n3].reshape((nx + 8, nk, ny + 8), order="F"))
        at += n3
    tabs = []
    for shape in [(ny + 8, nk, w)] * 2 + [(nx + 8, nk, w)] * 2 + [(ny + 8, nk, w)] * 2 + [
            (nx + 8, nk, w)] * 2:
        size = int(np.prod(shape))
        tabs.append(vals[at:at + size].reshape(shape, order="F"))
        at += size

    def b3(x):
        return np.ascontiguousarray(np.transpose(x[4:4 + fx, :fz, 4:4 + fy], (1, 2, 0)))

    def bx(x):
        return np.ascontiguousarray(np.transpose(x[4:4 + fy, :fz, :], (1, 0, 2)))

    def by(x):
        return np.ascontiguousarray(np.transpose(x[4:4 + fx, :fz, :], (1, 2, 0)))

    asrc, amup = _lb_coupled(geom, tr, adj=b3(x3[3]))
    exp = {"tl_tend": b3(x3[0]), "tl_d_tend": b3(x3[1]), "adst_tend": b3(x3[2]),
           "tl_held": tr["held"], "tl_d_held": np.zeros_like(tr["held"]),
           "adst_held": seeds["held"], "ad_src": asrc, "ad_mup": amup}
    names = ("west", "east", "south", "north", "west_t", "east_t", "south_t", "north_t")
    for name, x in zip(names, tabs):
        exp["ad_" + name] = bx(x) if name.startswith(("west", "east")) else by(x)
    return exp


def lbc(exe: str, runner: list[str], seed: int = 20261130) -> dict:
    out = {}
    for n, (name, over) in enumerate(LB_CASES.items()):
        over = dict(over)
        hybrid = over.pop("hybrid", 0)
        dims = dict(LB_DIMS)
        dims.update({k: over.pop(k) for k in ("ny", "nx") if k in over})
        geom = cases.lbc_geometry(**dims, apply_relax=1, **over)
        geom.validate()
        tr = cases.lbc_trajectory(geom, seed + 10 * n, hybrid=bool(hybrid))
        pe = cases.lbc_perturbation(geom, seed + 10 * n + 1)
        rng = np.random.default_rng(seed + 10 * n + 2)
        seeds = {s: rng.standard_normal(geom.field) for s in ("tend", "held")}
        blob, L = _lb_record(geom, tr, pe, seeds)
        with tempfile.TemporaryDirectory(dir=Path(exe).parent) as tmp:
            fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            fin.write_bytes(blob)
            subprocess.run(runner + [exe, str(fin), str(fout)], check=True)
            result = fout.read_bytes()
        exp = _lb_expected(geom, result, L, tr, seeds)
        prefix = f"{name}/"
        out[prefix + "op"] = np.array("lbc")
        out[prefix + "geometry"] = np.array(json.dumps(dataclasses.asdict(geom)))
        for k, v in seeds.items():
            out[prefix + "seed/" + k] = v
        for k, v in tr.items():
            out[prefix + "traj/" + k] = v
        for k, v in pe.items():
            out[prefix + "pert/" + k] = v
        for k, v in exp.items():
            out[prefix + "wrfplus/" + k] = v
    return out


#: lbc_final: spec_bdy_final on one field (mode 0) and spec_bdyupdate_ph +
#: zero_grad_bdy(w) + calc_p_rho (mode 1).  WRFPLUS's versions are not
#: hybrid and spec_bdy_final takes the coupling mass as given, so the
#: cases are not hybrid and the fixture forms the u/v face masses.
LF_CASES = {
    "lf_u_msf": dict(kind="u", has_msf=1),
    "lf_v_msf": dict(kind="v", has_msf=1),
    "lf_theta": dict(kind="theta"),
    "lf_phi": dict(kind="phi"),
    "lf_w_msf": dict(kind="w", has_msf=1),
    "lf_mu": dict(kind="mu"),
    "lf_wide": dict(kind="theta", spec_zone=2),
    "lf_u_wide": dict(kind="u", has_msf=1, spec_zone=2),
    "lf_v_wide": dict(kind="v", has_msf=1, spec_zone=2),
}
SP_CASES = {"sp_a": dict(), "sp_wide": dict(spec_zone=2, ny=9, nx=10)}
_LF_VAR = {"u": 0, "v": 1, "theta": 2, "phi": 3, "mu": 4, "w": 5}


def _lf_faces(geom, mu):
    """WRF calc_mu_uv's face mass of a (ny, nx) mass field, edges doubled."""
    if geom.kind == "u":
        i = np.arange(geom.nx + 1)
        return 0.5 * (mu[:, np.minimum(i, geom.nx - 1)] + mu[:, np.maximum(i - 1, 0)])
    if geom.kind == "v":
        j = np.arange(geom.ny + 1)
        return 0.5 * (mu[np.minimum(j, geom.ny - 1)] + mu[np.maximum(j - 1, 0)])
    return mu


def _lf_faces_ad(geom, a):
    out = np.zeros((geom.ny, geom.nx))
    if geom.kind == "u":
        i = np.arange(geom.nx + 1)
        np.add.at(out, (slice(None), np.minimum(i, geom.nx - 1)), 0.5 * a)
        np.add.at(out, (slice(None), np.maximum(i - 1, 0)), 0.5 * a)
        return out
    if geom.kind == "v":
        j = np.arange(geom.ny + 1)
        np.add.at(out, np.minimum(j, geom.ny - 1), 0.5 * a)
        np.add.at(out, np.maximum(j - 1, 0), 0.5 * a)
        return out
    return a.copy()


def _lf_pads(nx, ny, nk, w):
    def f3(a):
        out = np.zeros((nx + 8, nk, ny + 8))
        out[4:4 + a.shape[2], :a.shape[0], 4:4 + a.shape[1]] = np.transpose(a, (2, 0, 1))
        return out

    def f2(a):
        out = np.zeros((nx + 8, ny + 8))
        out[4:4 + a.shape[1], 4:4 + a.shape[0]] = a.T
        return out

    def tx(a):  # (fz, fy, w) -> (jms:jme, nk, w)
        out = np.zeros((ny + 8, nk, w))
        out[4:4 + a.shape[1], :a.shape[0], :] = np.transpose(a, (1, 0, 2))
        return out

    def ty(a):  # (fz, w, fx) -> (ims:ime, nk, w)
        out = np.zeros((nx + 8, nk, w))
        out[4:4 + a.shape[2], :a.shape[0], :] = np.transpose(a, (2, 0, 1))
        return out

    return f3, f2, tx, ty


def _lf_unpad(vals, nx, ny, nk, shapes):
    """Split a Fortran-order output stream into arrays of ``shapes``."""
    out, at = [], 0
    for shape in shapes:
        size = int(np.prod(shape))
        out.append(vals[at:at + size].reshape(shape, order="F"))
        at += size
    assert at == vals.size, (at, vals.size)
    return out


def _lf_one(geom, tr, pe, seeds, spert):
    nz, ny, nx, w = geom.nz, geom.ny, geom.nx, geom.width
    fz, fy, fx = geom.field
    nk = 1 if geom.kind == "mu" else nz + 1
    f3, f2, tx, ty = _lf_pads(nx, ny, nk, w)
    head = np.array([0, nx, ny, nz, _LF_VAR[geom.kind], w, geom.spec_zone, nk], np.int32)
    blob = head.tobytes() + np.array([geom.dtbc], "<f8").tobytes()
    msf = np.ones((fy, fx))
    if geom.has_msf and geom.kind in ("u", "v", "w"):
        msf = tr[{"u": "msfu", "v": "msfv", "w": "msft"}[geom.kind]]
    arrs = [f3(tr["target"]), f3(spert["target"]), f3(seeds["target"]),
            f2(_lf_faces(geom, tr["muts"])), f2(_lf_faces(geom, pe["muts"])), f2(msf)]
    for d in (tr, pe):
        arrs += [tx(d["west"]), tx(d["east"]), ty(d["south"]), ty(d["north"])]
        arrs += [tx(d["west_t"]), tx(d["east_t"]), ty(d["south_t"]), ty(d["north_t"])]
    for a in arrs:
        blob += np.asarray(a, "<f8").tobytes(order="F")

    def expected(out):
        s3, s2 = (nx + 8, nk, ny + 8), (nx + 8, ny + 8)
        sx, sy = (ny + 8, nk, w), (nx + 8, nk, w)
        x = _lf_unpad(np.frombuffer(out, "<f8"), nx, ny, nk,
                      [s3] * 3 + [s2] + [sx, sx, sy, sy] * 2)

        def b3(a):
            return np.ascontiguousarray(np.transpose(a[4:4 + fx, :fz, 4:4 + fy], (1, 2, 0)))

        def bx(a):
            return np.ascontiguousarray(np.transpose(a[4:4 + fy, :fz, :], (1, 0, 2)))

        def by(a):
            return np.ascontiguousarray(np.transpose(a[4:4 + fx, :fz, :], (1, 2, 0)))

        mub = np.ascontiguousarray(x[3][4:4 + fx, 4:4 + fy].T)
        exp = {"tl_target": b3(x[0]), "tl_d_target": b3(x[1]), "adst_target": b3(x[2]),
               "ad_muts": (np.zeros((ny, nx)) if geom.kind == "mu"
                           else _lf_faces_ad(geom, mub))}
        names = ("west", "east", "south", "north", "west_t", "east_t", "south_t", "north_t")
        for name, a in zip(names, x[4:]):
            exp["ad_" + name] = bx(a) if name.startswith(("west", "east")) else by(a)
        return exp

    return blob, expected


def _sp_one(geom, tr, pe, seeds, spert):
    from woof.grad.wrfplus.lbc import sp_shapes
    nz, ny, nx, sz = geom.nz, geom.ny, geom.nx, geom.spec_zone
    nk = nz + 1
    f3, f2, _, _ = _lf_pads(nx, ny, nk, 1)
    head = np.array([1, nx, ny, nz, 0, 1, sz, nk], np.int32)
    blob = head.tobytes() + np.array([geom.dtau], "<f8").tobytes()
    three = ("ph_pp", "w_pp", "rph_t", "php", "th_pp", "thp", "alt", "c2a")
    sd = dict(spert)
    arrs = [f3(tr[n]) for n in three]
    arrs += [f3(sd[n] if n in sd else pe[n]) for n in three]
    arrs += [f3(seeds[n]) for n in ("ph_pp", "w_pp", "p_pp", "al_pp")]
    arrs += [f2(tr[n]) for n in ("rmu_t", "muts", "mu_pp")]
    arrs += [f2(pe[n]) for n in ("rmu_t", "muts", "mu_pp")]
    for a in arrs:
        blob += np.asarray(a, "<f8").tobytes(order="F")
    blob += np.concatenate([tr["rdnw"], np.zeros(nz), np.zeros(nz)]).astype("<f8").tobytes()
    sh = sp_shapes(geom)

    def expected(out):
        s3, s2 = (nx + 8, nk, ny + 8), (nx + 8, ny + 8)
        x = _lf_unpad(np.frombuffer(out, "<f8"), nx, ny, nk, [s3] * 18 + [s2] * 3)

        def b3(a, n):
            return np.ascontiguousarray(np.transpose(a[4:4 + nx, :sh[n][0], 4:4 + ny],
                                                     (1, 2, 0)))

        def b2(a):
            return np.ascontiguousarray(a[4:4 + nx, 4:4 + ny].T)

        exp = {}
        for m, n in enumerate(("ph_pp", "w_pp", "p_pp", "al_pp")):
            exp["tl_" + n], exp["tl_d_" + n] = b3(x[2 * m], n), b3(x[2 * m + 1], n)
            exp["adst_" + n] = b3(x[8 + m], n)
        for m, n in enumerate(("rph_t", "php", "th_pp", "thp", "alt", "c2a")):
            exp["ad_" + n] = b3(x[12 + m], n)
        for m, n in enumerate(("rmu_t", "muts", "mu_pp")):
            exp["ad_" + n] = b2(x[18 + m])
        return exp

    return blob, expected


def lbc_final(exe: str, runner: list[str], seed: int = 20261201) -> dict:
    from woof.grad.wrfplus.lbc import LF_STATE, SP_STATE
    out = {}
    jobs = [("lf", k, v) for k, v in LF_CASES.items()] + [("sp", k, v)
                                                          for k, v in SP_CASES.items()]
    for n, (op, name, over) in enumerate(jobs):
        rng = np.random.default_rng(seed + 10 * n + 2)
        if op == "lf":
            geom = cases.lf_geometry(**over)
            geom.validate()
            tr = cases.lf_trajectory(geom, seed + 10 * n)
            pe = cases.lf_perturbation(geom, seed + 10 * n + 1)
            seeds = {s: rng.standard_normal(tr[s].shape) for s in LF_STATE}
            spert = {s: 0.1 * np.abs(tr[s]).max() * rng.standard_normal(tr[s].shape)
                     for s in LF_STATE}
            blob, expected = _lf_one(geom, tr, pe, seeds, spert)
        else:
            geom = cases.sp_geometry(**over)
            geom.validate()
            tr = cases.sp_trajectory(geom, seed + 10 * n)
            pe = cases.sp_perturbation(geom, seed + 10 * n + 1)
            seeds = {s: rng.standard_normal(tr[s].shape) for s in SP_STATE}
            spert = {s: 0.1 * np.abs(tr[s]).max() * rng.standard_normal(tr[s].shape)
                     for s in SP_STATE}
            ny, nx, sz = geom.ny, geom.nx, geom.spec_zone
            jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
            mask = np.minimum(np.minimum(jj, ny - 1 - jj), np.minimum(ii, nx - 1 - ii)) < sz
            for s in ("p_pp", "al_pp"):  # calc_p_rho runs on the frame only here
                seeds[s] = np.where(mask[None], seeds[s], 0.0)
            out[f"{name}/mask"] = mask
            blob, expected = _sp_one(geom, tr, pe, seeds, spert)
        for k in ("c2h", "c2f"):  # WRFPLUS's routines are not hybrid
            assert not np.any(tr[k]), k
        for k in ("c1h", "c1f"):
            assert np.all(tr[k] == 1.0), k
        with tempfile.TemporaryDirectory(dir=Path(exe).parent) as tmp:
            fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            fin.write_bytes(blob)
            subprocess.run(runner + [exe, str(fin), str(fout)], check=True)
            result = fout.read_bytes()
        prefix = f"{name}/"
        out[prefix + "op"] = np.array(op)
        out[prefix + "geometry"] = np.array(json.dumps(dataclasses.asdict(geom)))
        for kind, d in (("seed", seeds), ("spert", spert), ("traj", tr), ("pert", pe)):
            for k, v in d.items():
                out[prefix + kind + "/" + k] = v
        for k, v in expected(result).items():
            out[prefix + "wrfplus/" + k] = v
    return out


#: diffusion: the diff_opt 1 / km_opt 4 coefficients (diffusion.F90).
#: WRFPLUS's ``specified`` is one flag for every side; a periodic x with
#: specified y sides is WRF's channel (periodic_x plus specified).
#: a_cal_deform_and_div sizes its y tapes (Tmpv400-403) from
#: max(jds+1, jts), so on a periodic y axis (j_start = jts) it writes past
#: them and the driver crashes: the periodic-y cases are TL only.
#: (A bounded x with a periodic y is not a WRF configuration: specified
#: bounds y in cal_deform_and_div unless periodic_x.)
DIFF_CASES = {
    "df_periodic": dict(op="df", tl_only=1),
    "df_spec": dict(op="df", boundary_x=1, boundary_y=1),
    "df_spec_y": dict(op="df", boundary_y=1),
    "df_small": dict(op="df", nz=4, ny=6, nx=5, boundary_x=1, boundary_y=1),
    "km_periodic": dict(op="km"),
    "km_spec": dict(op="km", boundary_x=1, boundary_y=1),
    "km_spec_y": dict(op="km", boundary_y=1),
}
DIFF_DIMS = dict(nz=7, ny=8, nx=9)


def _df_layout(geom):
    from types import SimpleNamespace as NS
    return AdvLayout(NS(nz=geom.nz, ny=geom.ny, nx=geom.nx, open_x=geom.boundary_x,
                        open_y=geom.boundary_y))


def _df_head(geom, mode, sc, tl_only=0):
    bx, by = geom.boundary_x, geom.boundary_y
    flags = (int(not bx) | (int(not by) << 1) | (int(bool(bx or by)) << 2)
             | (int(bool(tl_only)) << 3))
    head = np.array([geom.nx, geom.ny, geom.nz, flags, mode], dtype=np.int32)
    return head.tobytes() + np.asarray(sc, dtype="<f8").tobytes()


def _df_record(geom, tr, pe, seeds, tl_only=0):
    """diffusion.F90 mode 1.  Halos wrap a periodic axis (a staggered end
    face takes face 0, which woof reads there) and copy the edge of a
    specified one (woof clamps)."""
    L = _df_layout(geom)
    blob = _df_head(geom, 1, [geom.rdx, geom.rdy, geom.cf1, geom.cf2, geom.cf3, 0.0, 0.0,
                              0.0], tl_only)
    rng = np.random.default_rng(7)
    w = rng.standard_normal((geom.nz + 1, geom.ny, geom.nx))
    a3 = [L.pad3(tr["u"], end=False), L.pad3(pe["u"], end=False),
          L.pad3(tr["v"], end=False), L.pad3(pe["v"], end=False),
          L.pad3(tr["php"]), L.pad3(pe["php"]), L.pad3(tr["phb"]), L.pad3(w)]
    a3 += [L.place3(seeds[n]) for n in ("defor11", "defor22", "defor12")]
    a2 = [_rp_pad2(L, tr["msfu"], False), _rp_pad2(L, tr["msfv"], False), L.pad2(tr["msft"])]
    for a in a3 + a2:
        blob += np.asarray(a, dtype="<f8").tobytes(order="F")
    for n in ("fnm", "fnp", "dn", "dnw"):
        blob += np.concatenate([tr[n], [0.0]]).astype("<f8").tobytes()
    return blob, L


def _df_expected(geom, out, L):
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    n3 = int(np.prod(L.s3))
    vals = np.frombuffer(out, dtype="<f8")
    assert vals.size == 12 * n3, (vals.size, n3)
    x = [vals[i * n3:(i + 1) * n3].reshape(L.s3, order="F") for i in range(12)]
    m = (nz, ny, nx)
    exp = {}
    for i, n in enumerate(("defor11", "defor22", "defor12")):
        exp["tl_" + n] = L.take3(x[2 * i], m)
        exp["tl_d_" + n] = L.take3(x[2 * i + 1], m)
        exp["adst_" + n] = L.take3(x[6 + i], m)
    # a periodic end face folds onto face 0 (the test sums woof's two)
    exp["ad_u"] = L.fold3(x[9], (nz, ny, nx + 1), end=False)
    exp["ad_v"] = L.fold3(x[10], (nz, ny + 1, nx), end=False)
    exp["ad_php"] = L.fold3(x[11], (nz + 1, ny, nx))
    return exp


def _km_record(geom, tr, pe, seeds, tl_only=0):
    L = _df_layout(geom)
    blob = _df_head(geom, 2, [0.0, 0.0, 0.0, 0.0, 0.0, geom.dx, geom.dy, geom.c_s], tl_only)
    a3 = []
    for n in ("defor11", "defor22", "defor12"):
        a3 += [L.pad3(tr[n]), L.pad3(pe[n])]
    a3 += [L.place3(seeds[n]) for n in ("xkmh", "xkhh")]
    for a in a3 + [L.pad2(tr["msft"])]:
        blob += np.asarray(a, dtype="<f8").tobytes(order="F")
    return blob, L


def _km_expected(geom, out, L):
    n3 = int(np.prod(L.s3))
    vals = np.frombuffer(out, dtype="<f8")
    assert vals.size == 9 * n3, (vals.size, n3)
    x = [vals[i * n3:(i + 1) * n3].reshape(L.s3, order="F") for i in range(9)]
    m = (geom.nz, geom.ny, geom.nx)
    exp = {"tl_xkmh": L.take3(x[0], m), "tl_d_xkmh": L.take3(x[1], m),
           "tl_xkhh": L.take3(x[2], m), "tl_d_xkhh": L.take3(x[3], m),
           "adst_xkmh": L.take3(x[4], m), "adst_xkhh": L.take3(x[5], m)}
    for i, n in enumerate(("defor11", "defor22", "defor12")):
        exp["ad_" + n] = L.fold3(x[6 + i], m)
    return exp


def diffusion(exe: str, runner: list[str], seed: int = 20261130) -> dict:
    from woof.grad.wrfplus import diffusion as D
    out = {}
    for n, (name, over) in enumerate(DIFF_CASES.items()):
        over = dict(over)
        op = over.pop("op")
        tl_only = over.pop("tl_only", 0)
        dims = dict(DIFF_DIMS)
        dims.update(over)
        rng = np.random.default_rng(seed + 10 * n + 2)
        if op == "df":
            # WRF's compute_diff_metrics keeps a periodic seam's top-level
            # slope at its cold zero (woof's seam_cold, RunConfig.
            # diff_metrics_seam_slope off), so every word is compared
            geom = cases.df_geometry(**dims, seam_cold=1)
            geom.validate()
            tr = cases.df_trajectory(geom, seed + 10 * n)
            pe = cases.df_perturbation(geom, seed + 10 * n + 1)
            sh = D.df_shapes(geom)
            seeds = {s: rng.standard_normal(sh[s]) for s in D.DF_STATE}
            blob, L = _df_record(geom, tr, pe, seeds, tl_only)
            expected = lambda r, g=geom, L=L: _df_expected(g, r, L)  # noqa: E731
        else:
            geom = cases.km_geometry(**dims)
            geom.validate()
            tr = cases.km_trajectory(geom, seed + 10 * n)
            pe = cases.km_perturbation(geom, seed + 10 * n + 1)
            sh = D.km_shapes(geom)
            seeds = {s: rng.standard_normal(sh[s]) for s in D.KM_STATE}
            # WRF never writes a specified axis's outer coefficient row (it
            # keeps its cold zero), so its adjoint is not consumed there;
            # woof overwrites the row with 0.  The seed is 0 there.
            for s in D.KM_STATE:
                if geom.boundary_x:
                    seeds[s][..., 0] = seeds[s][..., -1] = 0.0
                if geom.boundary_y:
                    seeds[s][..., 0, :] = seeds[s][..., -1, :] = 0.0
            blob, L = _km_record(geom, tr, pe, seeds, tl_only)
            expected = lambda r, g=geom, L=L: _km_expected(g, r, L)  # noqa: E731
        with tempfile.TemporaryDirectory(dir=Path(exe).parent) as tmp:
            fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            fin.write_bytes(blob)
            subprocess.run(runner + [exe, str(fin), str(fout)], check=True)
            result = fout.read_bytes()
        prefix = f"{name}/"
        out[prefix + "op"] = np.array(op)
        out[prefix + "geometry"] = np.array(json.dumps(dataclasses.asdict(geom)))
        for kind, d in (("seed", seeds), ("traj", tr), ("pert", pe)):
            for k, v in d.items():
                out[prefix + kind + "/" + k] = v
        for k, v in expected(result).items():
            if not (tl_only and not k.startswith("tl_")):
                out[prefix + "wrfplus/" + k] = v
    return out


#: hdiff: the diff_opt 1 horizontal-diffusion rows (hdiff.F90), every
#: row on periodic, specified and channel (periodic x, specified y) axes,
#: plus short specified cases; c1 = 1, c2 = 0 (WRFPLUS's rows take mu as
#: the mass).
HDIFF_ROW_CODE = {"u": 1, "v": 2, "w": 3, "t": 4, "s": 5}
HDIFF_CASES = {f"hd_{row}_{kind}": dict(row=row, **over)
               for row in HDIFF_ROW_CODE
               for kind, over in (("periodic", {}),
                                  ("spec", dict(boundary_x=1, boundary_y=1)),
                                  ("spec_y", dict(boundary_y=1)))}
HDIFF_CASES.update({f"hd_{row}_small": dict(row=row, nz=3, ny=5, nx=4, boundary_x=1,
                                            boundary_y=1) for row in ("w", "t")})


def _hd_record(geom, tr, pe, seed):
    """hdiff.F90's input.  Halos wrap a periodic axis (a staggered end
    face takes face 0, which woof reads there) and copy the edge of a
    specified one."""
    L = _df_layout(geom)
    bx, by = geom.boundary_x, geom.boundary_y
    flags = (int(not bx) | (int(not by) << 1) | (int(bool(bx or by)) << 2)
             | (HDIFF_ROW_CODE[geom.row] << 4))
    blob = np.array([geom.nx, geom.ny, geom.nz, flags, 0], dtype=np.int32).tobytes()
    blob += np.asarray([geom.rdx, geom.rdy], dtype="<f8").tobytes()
    base = tr["base"] if geom.perturb else np.zeros((geom.nz, geom.ny, geom.nx))
    a3 = [L.pad3(tr["field"], end=False), L.pad3(pe["field"], end=False),
          L.pad3(tr["xk"]), L.pad3(pe["xk"]), L.pad3(base), L.place3(tr["tend"]),
          np.zeros(L.s3), L.place3(seed["tend"])]
    a2 = [L.pad2(tr["mut"]), L.pad2(pe["mut"]), _rp_pad2(L, tr["msfu"], False),
          _rp_pad2(L, tr["msfv"], False), L.pad2(tr["msft"])]
    for a in a3 + a2:
        blob += np.asarray(a, dtype="<f8").tobytes(order="F")
    return blob, L


def _hd_expected(geom, out, L):
    from woof.grad.wrfplus.diffusion import hd_shapes
    sh = hd_shapes(geom)
    n3, n2 = int(np.prod(L.s3)), int(np.prod(L.s2))
    vals = np.frombuffer(out, dtype="<f8")
    assert vals.size == 5 * n3 + n2, (vals.size, n3, n2)
    x = [vals[i * n3:(i + 1) * n3].reshape(L.s3, order="F") for i in range(5)]
    amu = vals[5 * n3:].reshape(L.s2, order="F")
    return {"tl_tend": L.take3(x[0], sh["tend"]), "tl_d_tend": L.take3(x[1], sh["tend"]),
            "adst_tend": L.take3(x[2], sh["tend"]),
            # a periodic end face folds onto face 0 (the test sums woof's two)
            "ad_field": L.fold3(x[3], sh["field"], end=False),
            "ad_xk": L.fold3(x[4], sh["xk"]), "ad_mut": _rp_fold2(L, amu, sh["mut"], True)}


def hdiff(exe: str, runner: list[str], seed: int = 20261201) -> dict:
    from woof.grad.wrfplus import diffusion as D
    out = {}
    for n, (name, over) in enumerate(HDIFF_CASES.items()):
        dims = dict(DIFF_DIMS)
        dims.update(over)
        geom = cases.hd_geometry(**dims)
        geom.validate()
        tr = cases.hd_trajectory(geom, seed + 10 * n)
        pe = cases.hd_perturbation(geom, seed + 10 * n + 1)
        rng = np.random.default_rng(seed + 10 * n + 2)
        seeds = {"tend": rng.standard_normal(D.hd_shapes(geom)["tend"])}
        blob, L = _hd_record(geom, tr, pe, seeds)
        with tempfile.TemporaryDirectory(dir=Path(exe).parent) as tmp:
            fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            fin.write_bytes(blob)
            subprocess.run(runner + [exe, str(fin), str(fout)], check=True)
            result = fout.read_bytes()
        prefix = f"{name}/"
        out[prefix + "geometry"] = np.array(json.dumps(dataclasses.asdict(geom)))
        for kind, d in (("seed", seeds), ("traj", tr), ("pert", pe)):
            for k, v in d.items():
                out[prefix + kind + "/" + k] = v
        for k, v in _hd_expected(geom, result, L).items():
            out[prefix + "wrfplus/" + k] = v
    return out


#: diffusion2: the diff_opt 2 horizontal-diffusion rows (diffusion2.F90,
#: WRF 4.7.1's nonlinear horizontal_diffusion_u_2/_v_2/_w_2/_s with
#: compute_diff_metrics, rho and cal_deform_and_div), every row on
#: periodic, specified and channel (periodic x, specified y) axes plus a
#: moist case per row.  The TL and AD are fourth-order central
#: differences of the WRF rows over every active woof word (its halo
#: images move with it), so they hold to about 1e-9 of the scale.
H2_ROW_CODE = {"u": 1, "v": 2, "w": 3, "t": 5, "s": 5, "vu": 6, "vv": 6, "vw": 6, "vt": 6,
               "vs": 6}
H2_FIELD_IDS = {"php": 1, "alt": 2, "qv": 3, "xk": 4, "defor11": 5, "defor22": 6,
                "defor12": 7, "u": 8, "v": 9, "w": 10, "field": 11}
H2_DIMS = dict(nz=4, ny=4, nx=5)
DIFF2_CASES = {f"h2_{row}_{kind}": dict(row=row, **over)
               for row in ("u", "v", "w", "s")
               for kind, over in (("periodic", {}),
                                  ("spec", dict(boundary_x=1, boundary_y=1)),
                                  ("spec_y", dict(boundary_y=1)),
                                  ("moist", dict(boundary_x=1, moist=1)))}
DIFF2_CASES["h2_t_periodic"] = dict(row="t")

#: vdiffusion2: vertical_diffusion_2 (diffusion2.F90 row 6) with its
#: isfflx surface fluxes, every row on periodic, specified and channel
#: axes and moist, with the surface options of each row (USTM, cd0; the
#: prescribed heat flux, HFX, mix_full_fields; the qv row's QFX).
VDIFF2_CASES = {}
for _row, _kinds in (
        ("vu", (("periodic", {}), ("spec_ustm", dict(boundary_x=1, boundary_y=1, surface="ustm")),
                ("spec_y_cd0", dict(boundary_y=1, surface="cd0", cd0=2e-3)),
                ("moist_ustm", dict(boundary_x=1, moist=1, surface="ustm")))),
        ("vv", (("periodic", {}), ("spec_ustm", dict(boundary_x=1, boundary_y=1, surface="ustm")),
                ("spec_y_cd0", dict(boundary_y=1, surface="cd0", cd0=2e-3)),
                ("moist_ustm", dict(boundary_x=1, moist=1, surface="ustm")))),
        ("vw", (("periodic", {}), ("spec", dict(boundary_x=1, boundary_y=1)),
                ("spec_y", dict(boundary_y=1)), ("moist", dict(boundary_x=1, moist=1)))),
        ("vt", (("periodic", {}), ("spec_const", dict(boundary_x=1, boundary_y=1, heat="const",
                                                      heat_flux=0.1)),
                ("spec_y_hfx", dict(boundary_y=1, heat="hfx")),
                ("moist_hfx", dict(boundary_x=1, moist=1, heat="hfx")),
                ("periodic_full", dict(heat="hfx", full_theta=1)))),
        ("vs", (("periodic", {}), ("spec", dict(boundary_x=1, boundary_y=1)),
                ("moist_qfx", dict(boundary_x=1, moist=1, qflux=1)),
                ("spec_y_qfx", dict(boundary_y=1, moist=1, qflux=1))))):
    for _kind, _over in _kinds:
        VDIFF2_CASES[f"h2_{_row}_{_kind}"] = dict(row=_row, **_over)


def _h2_end(name):
    # woof reads a periodic staggered end face as face 0
    return name not in ("u", "v", "msfu", "msfv")


def _h2_record_head(geom, tr, nrec):
    L = _df_layout(geom)
    bx, by = geom.boundary_x, geom.boundary_y
    flags = (int(not bx) | (int(not by) << 1) | (int(bool(bx or by)) << 2)
             | (int(bool(geom.moist)) << 3) | (H2_ROW_CODE[geom.row] << 4))
    blob = np.array([geom.nx, geom.ny, geom.nz, flags, nrec], dtype=np.int32).tobytes()
    blob += np.asarray([geom.rdx, geom.rdy, geom.cf1, geom.cf2, geom.cf3], dtype="<f8").tobytes()
    for n in ("fnm", "fnp", "dn", "dnw"):
        blob += np.concatenate([tr[n], [0.0]]).astype("<f8").tobytes()
    for n in ("msfu", "msfv"):
        blob += np.asarray(_rp_pad2(L, tr[n], False), dtype="<f8").tobytes(order="F")
    blob += np.asarray(L.pad2(tr["msft"]), dtype="<f8").tobytes(order="F")
    blob += np.asarray(L.pad3(tr["phb"]), dtype="<f8").tobytes(order="F")
    act = np.zeros(L.s3 + (len(H2_FIELD_IDS),), order="F")
    act[..., H2_FIELD_IDS["alt"] - 1] = 1.0
    for n, f in H2_FIELD_IDS.items():
        if n in tr:
            act[..., f - 1] = L.pad3(tr[n], end=_h2_end(n))
    blob += np.asarray(act, dtype="<f8").tobytes(order="F")
    # -0.0 marks the words the row leaves alone (-0.0 + x is x)
    blob += np.full(L.s3, -0.0, dtype="<f8").tobytes(order="F")
    if geom.vertical:
        vrow = {"vu": 1, "vv": 2, "vw": 3, "vt": 4}.get(geom.row, 5 if geom.qflux else 6)
        isfflx = 0 if geom.surface == "cd0" or geom.heat == "const" else 1
        blob += np.array([vrow, isfflx, int(geom.full_theta)], dtype=np.int32).tobytes()
        blob += np.asarray([geom.cd0, geom.heat_flux], dtype="<f8").tobytes()
        zero2 = np.zeros((geom.ny, geom.nx))
        use = {"ustm": geom.surface == "ustm", "hfx": geom.heat == "hfx", "qfx": bool(geom.qflux)}
        for n in ("ustm", "hfx", "qfx"):
            blob += np.asarray(L.pad2(tr[n] if use[n] else zero2), dtype="<f8").tobytes(order="F")
        thb = tr["thb"] if geom.full_theta else np.zeros((geom.nz, geom.ny, geom.nx))
        blob += np.asarray(L.pad3(thb), dtype="<f8").tobytes(order="F")
    return blob, L


def _h2_images(L, shape, end):
    """For each woof word of ``shape``, its 1-based linear Fortran words."""
    kz, yi, xi = L._maps(shape, end)
    word = np.ravel_multi_index(np.ix_(kz, yi, xi), shape)          # (K, J, I) woof word
    lin = np.arange(int(np.prod(L.s3))).reshape(L.s3, order="F")     # (I, K, J)
    lin = np.transpose(lin, (1, 2, 0))                                # (K, J, I)
    out = [[] for _ in range(int(np.prod(shape)))]
    for w, m in zip(word.ravel(), lin.ravel()):
        out[w].append(int(m) + 1)
    return out


def diffusion2(exe: str, runner: list[str], seed: int = 20261210) -> dict:
    return _h2_fixture(exe, runner, seed, DIFF2_CASES)


def vdiffusion2(exe: str, runner: list[str], seed: int = 20261220) -> dict:
    return _h2_fixture(exe, runner, seed, VDIFF2_CASES)


def _h2_fixture(exe, runner, seed, table) -> dict:
    from woof.grad.wrfplus import diffusion as D
    out = {}
    for n, (name, over) in enumerate(table.items()):
        dims = dict(H2_DIMS)
        dims.update(over)
        geom = cases.h2_geometry(**dims)
        geom.validate()
        tr = cases.h2_trajectory(geom, seed + 10 * n)
        pe = cases.h2_perturbation(geom, seed + 10 * n + 1)
        rng = np.random.default_rng(seed + 10 * n + 2)
        sh = D.h2_shapes(geom)
        active = D.h2_active(geom)
        recs, cols = [], []
        for a in active:
            imgs = _h2_images(_df_layout(geom), sh[a], _h2_end(a))
            x = tr[a].ravel()
            scale = float(np.sqrt(np.mean(tr[a] ** 2)))
            for w, mem in enumerate(imgs):
                h = 1e-3 * max(abs(float(x[w])), scale)
                cols.append((a, w, h))
                for m in (1, -1, 2, -2):
                    recs.append((m * h, [H2_FIELD_IDS[a]] * len(mem), mem))
        head, L = _h2_record_head(geom, tr, len(recs) + 1)
        body = [np.array([0], np.int32).tobytes() + np.array([0.0], "<f8").tobytes()]
        for h, ids, mem in recs:
            body.append(np.array([len(ids)], np.int32).tobytes()
                        + np.array([h], "<f8").tobytes()
                        + np.array(ids, np.int32).tobytes() + np.array(mem, np.int32).tobytes())
        with tempfile.TemporaryDirectory(dir=Path(exe).parent) as tmp:
            fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
            fin.write_bytes(head + b"".join(body))
            subprocess.run(runner + [exe, str(fin), str(fout)], check=True)
            result = fout.read_bytes()
        n3 = int(np.prod(L.s3))
        vals = np.frombuffer(result, dtype="<f8").reshape(len(recs) + 1, n3)
        base = L.take3(vals[0].reshape(L.s3, order="F"), sh["tend"])
        touched = ~((base == 0.0) & np.signbit(base))
        if geom.code in (1, 4) and not geom.boundary_x:
            touched[:, :, -1] = False
        if geom.code in (2, 5) and not geom.boundary_y:
            touched[:, -1, :] = False
        if geom.vertical:
            # vertical_diffusion_2's surface loops run on every column, its
            # interior rows on WRF's loop region: grade the columns the
            # interior reaches (level 1, interior for every row)
            touched &= touched[1:2]
        f = lambda r: L.take3(vals[r].reshape(L.s3, order="F"), sh["tend"])[touched]  # noqa: E731
        jac = np.empty((int(touched.sum()), len(cols)))
        for c, (_, _, h) in enumerate(cols):
            r = 1 + 4 * c
            jac[:, c] = (8.0 * (f(r) - f(r + 1)) - (f(r + 2) - f(r + 3))) / (12.0 * h)
        dx = np.concatenate([pe[a].ravel() for a in active])
        seeds = {"tend": rng.standard_normal(sh["tend"]) * touched}
        tl = np.zeros(sh["tend"])
        tl[touched] = jac @ dx
        adj = jac.T @ seeds["tend"][touched]
        prefix = f"{name}/"
        out[prefix + "geometry"] = np.array(json.dumps(dataclasses.asdict(geom)))
        for kind, d in (("seed", seeds), ("traj", tr), ("pert", pe)):
            for k, v in d.items():
                out[prefix + kind + "/" + k] = v
        nl = tr["tend"].copy()
        nl[touched] = nl[touched] + base[touched]
        out[prefix + "wrfplus/mask"] = touched
        out[prefix + "wrfplus/nl_tend"] = nl
        out[prefix + "wrfplus/tl_d_tend"] = tl
        at = 0
        for a in active:
            size = int(np.prod(sh[a]))
            out[prefix + "wrfplus/ad_" + a] = adj[at:at + size].reshape(sh[a])
            at += size
    return out


#: kcoef2: the diff_opt 2 coefficient routines (diffusion2.F90 row 7):
#: smag2d_km (km_opt 4 with its slope limiter), calculate_N2 (dry) and
#: smag_km (km_opt 3, anisotropic and isotropic), each on periodic,
#: specified and channel axes; smag2d_km also on steep slopes (alpha > 1).
#: The TL and AD are fourth-order central differences over every active
#: woof word, as diffusion2's.
KC_OPS = {"k2": (1, ("xkmh", "xkhh")), "n2": (2, ("bn2",)),
          "k3": (3, ("xkmh", "xkhh", "xkmv", "xkhv"))}
KC_OUT = {"xkmh": 1, "xkhh": 2, "xkmv": 3, "xkhv": 4, "bn2": 5}
KC_ACTIVE = {"k2": ("defor11", "defor22", "defor12", "php"), "n2": ("thp", "php"),
             "k3": ("defor11", "defor22", "defor12", "u", "v", "w", "php", "bn2")}
KC_FIELD_IDS = {**H2_FIELD_IDS, "thp": 11, "bn2": 12}
KCOEF2_CASES = {}
for _op in KC_OPS:
    for _kind, _over in (("periodic", {}), ("spec", dict(boundary_x=1, boundary_y=1)),
                         ("spec_y", dict(boundary_y=1))):
        KCOEF2_CASES[f"kc_{_op}_{_kind}"] = dict(op=_op, **_over)
KCOEF2_CASES["kc_k2_steep"] = dict(op="k2", steep=True)
KCOEF2_CASES["kc_k2_spec_steep"] = dict(op="k2", steep=True, boundary_x=1, boundary_y=1)
KCOEF2_CASES["kc_k3_iso"] = dict(op="k3", isotropic=1)
KCOEF2_CASES["kc_k3_spec_iso"] = dict(op="k3", isotropic=1, boundary_x=1)


def _kc_record_head(geom, tr, vert, op, out_name, nrec):
    L = _df_layout(geom)
    bx, by = geom.boundary_x, geom.boundary_y
    flags = (int(not bx) | (int(not by) << 1) | (int(bool(bx or by)) << 2) | (7 << 4))
    blob = np.array([geom.nx, geom.ny, geom.nz, flags, nrec], dtype=np.int32).tobytes()
    blob += np.asarray([geom.rdx, geom.rdy, 1.6, -0.8, 0.2], dtype="<f8").tobytes()
    for n in ("fnm", "fnp", "dn", "dnw"):
        blob += np.concatenate([vert[n], [0.0]]).astype("<f8").tobytes()
    for n in ("msfu", "msfv"):
        blob += np.asarray(_rp_pad2(L, tr[n], False), dtype="<f8").tobytes(order="F")
    blob += np.asarray(L.pad2(tr["msft"]), dtype="<f8").tobytes(order="F")
    blob += np.asarray(L.pad3(tr["phb"]), dtype="<f8").tobytes(order="F")
    act = np.zeros(L.s3 + (11,), order="F")
    act[..., H2_FIELD_IDS["alt"] - 1] = 1.0
    for n, f in KC_FIELD_IDS.items():
        if n in tr and f <= 11:
            act[..., f - 1] = L.pad3(tr[n], end=_h2_end(n))
    blob += np.asarray(act, dtype="<f8").tobytes(order="F")
    blob += np.full(L.s3, -0.0, dtype="<f8").tobytes(order="F")
    blob += np.array([KC_OPS[op][0], KC_OUT[out_name], geom.isotropic],
                     dtype=np.int32).tobytes()
    blob += np.asarray([geom.dx, geom.dy, geom.dt, geom.mix_upper_bound, geom.c_s],
                       dtype="<f8").tobytes()
    blob += np.asarray(L.pad3(tr["thb"]), dtype="<f8").tobytes(order="F")
    blob += np.asarray(L.pad3(tr["bn2"]), dtype="<f8").tobytes(order="F")
    return blob, L


def kcoef2(exe: str, runner: list[str], seed: int = 20261230) -> dict:
    from woof.grad.wrfplus import diffusion as D
    out = {}
    for n, (name, over) in enumerate(KCOEF2_CASES.items()):
        over = dict(over)
        op, steep = over.pop("op"), over.pop("steep", False)
        dims = dict(H2_DIMS)
        dims.update(over)
        geom = cases.kc_geometry(**dims)
        geom.validate()
        tr = cases.kc_trajectory(geom, seed + 10 * n, steep=steep)
        pe = cases.kc_perturbation(geom, seed + 10 * n + 1)
        rng = np.random.default_rng(seed + 10 * n + 2)
        vrng = np.random.default_rng(seed + 10 * n + 3)
        v = cases._vertical(geom.nz, vrng, False)
        dn = np.zeros(geom.nz)
        dn[1:] = np.diff(v["znu"])
        fnm, fnp = np.zeros(geom.nz), np.zeros(geom.nz)
        fnm[1:] = 0.5 * v["dnw"][:-1] / dn[1:]
        fnp[1:] = 0.5 * v["dnw"][1:] / dn[1:]
        vert = dict(dn=dn, dnw=v["dnw"], fnm=fnm, fnp=fnp)
        sh = D.kc_shapes(geom)
        active = KC_ACTIVE[op]
        recs, cols = [], []
        for a in active:
            imgs = _h2_images(_df_layout(geom), sh[a], _h2_end(a))
            x = tr[a].ravel()
            scale = float(np.sqrt(np.mean(tr[a] ** 2)))
            for w, mem in enumerate(imgs):
                # relative to the word itself (deformations span 1e-4 to
                # 0.1 /s: a step of the field's scale would swamp a small one)
                h = 1e-3 * max(abs(float(x[w])), 1e-3 * scale)
                cols.append((a, w, h))
                for m in (1, -1, 2, -2):
                    recs.append((m * h, [KC_FIELD_IDS[a]] * len(mem), mem))
        body = [np.array([0], np.int32).tobytes() + np.array([0.0], "<f8").tobytes()]
        for h, ids, mem in recs:
            body.append(np.array([len(ids)], np.int32).tobytes()
                        + np.array([h], "<f8").tobytes()
                        + np.array(ids, np.int32).tobytes() + np.array(mem, np.int32).tobytes())
        prefix = f"{name}/"
        seeds, adj = {}, 0.0
        dx = np.concatenate([pe[a].ravel() for a in active])
        mask = None
        for o in KC_OPS[op][1]:
            head, L = _kc_record_head(geom, tr, vert, op, o, len(recs) + 1)
            with tempfile.TemporaryDirectory(dir=Path(exe).parent) as tmp:
                fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
                fin.write_bytes(head + b"".join(body))
                subprocess.run(runner + [exe, str(fin), str(fout)], check=True)
                result = fout.read_bytes()
            n3 = int(np.prod(L.s3))
            vals = np.frombuffer(result, dtype="<f8").reshape(len(recs) + 1, n3)
            base = L.take3(vals[0].reshape(L.s3, order="F"), sh[o])
            touched = ~((base == 0.0) & np.signbit(base))
            mask = touched if mask is None else mask
            f = lambda r: L.take3(vals[r].reshape(L.s3, order="F"), sh[o])[touched]  # noqa: E731
            jac = np.empty((int(touched.sum()), len(cols)))
            for c, (_, _, h) in enumerate(cols):
                r = 1 + 4 * c
                jac[:, c] = (8.0 * (f(r) - f(r + 1)) - (f(r + 2) - f(r + 3))) / (12.0 * h)
            seeds[o] = rng.standard_normal(sh[o]) * touched
            tl = np.zeros(sh[o])
            tl[touched] = jac @ dx
            adj = adj + jac.T @ seeds[o][touched]
            nl = np.zeros(sh[o])
            nl[touched] = base[touched]
            out[prefix + "wrfplus/nl_" + o] = nl
            out[prefix + "wrfplus/tl_d_" + o] = tl
        out[prefix + "op"] = np.array(op)
        out[prefix + "geometry"] = np.array(json.dumps(dataclasses.asdict(geom)))
        out[prefix + "wrfplus/mask"] = mask
        for kind, d in (("seed", seeds), ("traj", tr), ("pert", pe)):
            for k, val in d.items():
                out[prefix + kind + "/" + k] = val
        at = 0
        for a in active:
            size = int(np.prod(sh[a]))
            out[prefix + "wrfplus/ad_" + a] = adj[at:at + size].reshape(sh[a])
            at += size
    return out


ROUTINES = {"advance_uv": advance_uv, "advance_w": advance_w,
            "calc_coef_w": calc_coef_w, "advance_mu_t": advance_mu_t,
            "small_step": small_step, "rk_stage": rk_stage,
            "advect": advect, "bigstep": bigstep, "rhsph": rhsph,
            "coriolis": coriolis, "w_damp": w_damp,
            "lbc": lbc, "lbc_final": lbc_final, "diffusion": diffusion,
            "hdiff": hdiff, "diffusion2": diffusion2, "vdiffusion2": vdiffusion2,
            "kcoef2": kcoef2}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("routine", choices=sorted(ROUTINES))
    ap.add_argument("--exe", required=True)
    ap.add_argument("--runner", default="")
    ap.add_argument("--wrfplus-digest", default="",
                    help="sha256 of the WRFPLUS source archive the driver links")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    data = ROUTINES[args.routine](args.exe, shlex.split(args.runner))
    # vdiffusion2 runs diffusion2.F90's vertical_diffusion_2 row
    stem = {"vdiffusion2": "diffusion2", "kcoef2": "diffusion2"}.get(args.routine,
                                                                   args.routine)
    driver = ROOT / "tools/wrfplus_oracle" / f"{stem}.F90"
    data["provenance"] = np.array(
        f"WRFPLUS 4.7.1 (configure wrfplus 17, real*8, -O0 -ffp-contract=off); "
        f"source sha256 {args.wrfplus_digest}; driver sha256 "
        f"{hashlib.sha256(driver.read_bytes()).hexdigest()}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes, {len(data)} arrays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
