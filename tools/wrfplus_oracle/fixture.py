"""Make a WRFPLUS 4.7.1 oracle fixture for one TL/AD routine.

    python tools/wrfplus_oracle/fixture.py advance_uv|advance_w \
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
from woof.grad.wrfplus.acoustic import UV_ACTIVE, W_STATE  # noqa: E402

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
#: c2 = 0; no damper).
W_CASES = {
    "plain": dict(spec_zone=1),
    "lid_moist": dict(spec_zone=1, top_lid=1, moist_cq=1, base3d=0),
    "deep_moist": dict(nz=12, ny=7, nx=6, spec_zone=1, moist_cq=1, epssm=0.3),
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
    a2 = [mut, mutd, muts, mutsd, muave, muaved, tr["mu_pp"], pe["mu_pp"], tr["ht"]]
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


ROUTINES = {"advance_uv": advance_uv, "advance_w": advance_w}


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
    driver = ROOT / "tools/wrfplus_oracle" / f"{args.routine}.F90"
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
