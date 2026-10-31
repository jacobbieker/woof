"""km_opt = 5 (SMS-3DTKE) against compiled WRF v4.7.1, word for word.

tests/data/oracles/sms3dtke is tools/sms3dtke_wrf471_oracle's record: the
unmodified WRF routines (phy_prep, compute_diff_metrics, cal_deform_and_div,
nonlocal_flux, free_atmos_length, meso_length_scale, calculate_km_kh,
phy_bc, tke_rhs, vertical_diffusion_implicit) at gfortran -O0
-ffp-contract=off against glibc 2.43, driven over 3-D tiles: convective,
neutral and stable boundary layers and edge columns, flat ground and a
hill, periodic and specified boundaries, isfflx 0/1/2, dx from 50 m to
9 km.  Each woof kernel is fed WRF's own operands for its stage and every
output word must be equal, under both arithmetic selections.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "oracles" / "sms3dtke"
sys.path.insert(0, str(ROOT / "tools" / "sms3dtke_wrf471_oracle"))
from records import read_cases  # noqa: E402

from woof.core import sms3dtke  # noqa: E402


def _cases():
    receipt = (DATA / "receipt.txt").read_text()
    for name in ("cases.bin", "oracle.bin"):
        digest = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        assert f"{digest}  {name}" in receipt, name
    assert "glibc 2.43" in receipt or "GLIBC 2.43" in receipt
    return read_cases(DATA / "cases.bin", DATA / "oracle.bin")


@pytest.fixture(scope="module")
def cases():
    return _cases()


def _d(a, dtype=np.float32):
    return cp.asarray(np.ascontiguousarray(a, dtype=dtype))


def run_case(c):
    """Every km_opt = 5 kernel on one case, each fed WRF's operands.

    Returns {field: (woof words, WRF words)}.
    """
    W = c["wrf"]
    nx, ny, nz = c["nx"], c["ny"], c["nz"]
    bx = by = int(c["bnd"])
    dx, dy, dt = c["dx"], c["dy"], c["dt"]
    isfflx, heat = c["isfflx"], c["tke_heat_flux"]
    out = {}
    msft = _d(c["msf"])
    qv = _d(c["qv"])

    # phy_prep / compute_diff_metrics operands from the raw state.
    prep = {name: cp.zeros((nz, ny, nx), cp.float32)
            for name in ("th", "t2", "rho", "u_phy", "v_phy", "rdzw", "rdz")}
    prep["zx"] = cp.zeros((nz + 1, ny, nx + 1), cp.float32)
    prep["zy"] = cp.zeros((nz + 1, ny + 1, nx), cp.float32)
    for name in ("p8w1", "t8w1", "p8wt", "t8wt"):
        prep[name] = cp.zeros((ny, nx), cp.float32)
    sms3dtke.prepare(
        u=_d(c["u"]), v=_d(c["v"]), php=_d(c["ph"]), phb=_d(c["phb"]),
        thp=_d(c["t"]), thb=_d(np.full(nz, 300.0)), alt=_d(c["alt"]), qv=qv,
        p=_d(c["p"] + c["pb"]), rdx=np.float32(1.0) / np.float32(dx),
        rdy=np.float32(1.0) / np.float32(dy), bx=bx, by=by, out=prep)
    for name in ("th", "rho", "u_phy", "v_phy", "rdzw", "rdz", "zx", "zy",
                 "p8w1", "t8w1", "p8wt", "t8wt"):
        out["prep:" + name] = (prep[name], W[name])

    # WRF's own operands for every later stage.
    th, rho, rdzw, rdz = (_d(W[n]) for n in ("th", "rho", "rdzw", "rdz"))
    p8w = {n: _d(W[n]) for n in ("p8w1", "t8w1", "p8wt", "t8wt")}
    tke = _d(c["tke"])

    nlflux = cp.zeros((nz + 1, ny, nx), cp.float32)
    gamu, gamv = _d(c["gamu"]), _d(c["gamv"])
    hpbl = _d(c["pblh"])
    kpbl = cp.zeros((ny, nx), cp.int32)
    hfx = _d(c["hfx"])
    sms3dtke.nonlocal_flux(
        rdzw=rdzw, th=th, qv=qv, u_phy=_d(W["u_phy"]), v_phy=_d(W["v_phy"]),
        rho=rho, ust=_d(c["ust"]), br=_d(c["br"]), wspd=_d(c["wspd"]),
        msft=msft, hfx=hfx, dx=dx, dy=dy, isfflx=isfflx, tke_heat_flux=heat,
        nlflux=nlflux, gamu=gamu, gamv=gamv, hpbl=hpbl, kpbl=kpbl, bx=bx, by=by)
    for name, got in (("nlflux", nlflux), ("gamu", gamu), ("gamv", gamv),
                      ("pblh", hpbl), ("kpbl", kpbl), ("hfx_nonlocal", hfx)):
        out["nonlocal:" + name] = (got, W[name])

    elmin = cp.zeros((nz, ny, nx), cp.float32)
    sms3dtke.free_atmos_length(rdzw=rdzw, tke=tke, th=th, elmin=elmin,
                               bx=bx, by=by)
    out["free_atmos:elmin"] = (elmin, W["elmin"])

    dlk = cp.zeros((nz, ny, nx), cp.float32)
    hfx2 = _d(W["hfx_nonlocal"])
    sms3dtke.meso_length_scale(
        rdzw=rdzw, rdz=rdz, tke=tke, th=th, **p8w, hpbl=_d(W["pblh"]),
        elmin=_d(W["elmin"]), rmol=_d(c["rmol"]), rho=rho, hfx=hfx2, qv=qv,
        isfflx=isfflx, tke_heat_flux=heat, dlk=dlk, bx=bx, by=by)
    out["meso:dlk"] = (dlk, W["dlk"])
    out["meso:hfx"] = (hfx2, W["hfx_meso"])

    ks = {n: cp.zeros((nz, ny, nx), cp.float32)
          for n in ("xkmh", "xkmv", "xkhh", "xkhv", "xkmv_meso")}
    seed = 0.0
    if isfflx == 0 and c["tke_drag_coefficient"] < 1e-10 and heat < 1e-10:
        seed = 1.0e-6
    sms3dtke.tke_km(
        tke=tke, th=th, rdz=rdz, rdzw=rdzw, **p8w, d11=_d(W["d11"]),
        d22=_d(W["d22"]), d12=_d(W["d12"]), zx=_d(W["zx"]), zy=_d(W["zy"]),
        msft=msft, hpbl=_d(W["pblh"]), dlk=_d(W["dlk"]), dx=dx, dy=dy,
        c_s=c["c_s"], c_k=c["c_k"], tke_seed=seed, bx=bx, by=by, **ks)
    for name in ks:
        out["tke_km:" + name] = (ks[name], W[name])

    mut, c1h, c2h = _d(c["mut"]), _d(c["c1h"][:nz]), _d(c["c2h"][:nz])
    tend = _d(W["shear"])
    l_diss = cp.zeros((nz, ny, nx), cp.float32)
    sms3dtke.tke_rhs(
        tend=tend, bn2=_d(W["bn2"]), xkhv=_d(W["xkhv"]), nlflux=_d(W["nlflux"]),
        th=th, tke=tke, mut=mut, c1h=c1h, c2h=c2h, rdzw=rdzw, msft=msft,
        hpbl=_d(W["pblh"]), dlk=_d(W["dlk"]), qv=qv, rho=rho,
        hfx=_d(W["hfx_meso"]), isfflx=isfflx, tke_heat_flux=heat,
        c_k=c["c_k"], dt=dt, dx=dx, dy=dy, l_diss=l_diss, bx=bx, by=by)
    out["tke_rhs:tend"] = (tend, W["tke_rhs"])
    out["tke_rhs:l_diss"] = (l_diss, W["l_diss"])

    ws = cp.zeros(sms3dtke.workspace_shape(nz, ny, nx), cp.float32)
    fnm, fnp = _d(c["fnm"][:nz]), _d(c["fnp"][:nz])
    dnw, dn = _d(c["dnw"][:nz]), _d(c["dn"][:nz])
    c1f, c2f = _d(c["c1f"]), _d(c["c2f"])
    u, v, w = _d(c["u"]), _d(c["v"]), _d(c["w"])
    xkmv, xkhv, xkmh = _d(W["xkmv"]), _d(W["xkhv"]), _d(W["xkmh"])
    meso = _d(W["xkmv_meso"])
    ustm = _d(c["ustm"])
    common = dict(fnm=fnm, fnp=fnp, dnw=dnw, c1h=c1h, c2h=c2h, mut=mut,
                  dt=dt, ws=ws, bx=bx, by=by)
    ru = cp.zeros((nz, ny, nx + 1), cp.float32)
    rv = cp.zeros((nz, ny + 1, nx), cp.float32)
    rw = cp.zeros((nz + 1, ny, nx), cp.float32)
    rt = cp.zeros((nz, ny, nx), cp.float32)
    sms3dtke.vdiff_u(ru=ru, u=u, v=v, rho=rho, xkmv=xkmv, xkmv_meso=meso,
                     gamu=_d(W["gamu"]), rdz=rdz, ustm=ustm, **common)
    sms3dtke.vdiff_v(rv=rv, u=u, v=v, rho=rho, xkmv=xkmv, xkmv_meso=meso,
                     gamv=_d(W["gamv"]), rdz=rdz, ustm=ustm, **common)
    sms3dtke.vdiff_w(rw=rw, w=w, rho=rho, xkmh=xkmh, rdzw=rdzw, dn=dn,
                     c1f=c1f, c2f=c2f, mut=mut, dt=dt, ws=ws, bx=bx, by=by)
    hfx3 = _d(W["hfx_meso"])
    t2 = _d(c["t"])
    var = t2 if c["mix_full_fields"] else t2 - _d(c["t_base"][:nz])[:, None, None]
    sms3dtke.vdiff_th(rt=rt, var=var, th=th, rho=rho, xkhv=xkhv,
                      nlflux=_d(W["nlflux"]), rdz=rdz, hfx=hfx3,
                      qfx=_d(c["qfx"]), qv=qv, use_theta_m=c["use_theta_m"],
                      isfflx=isfflx, tke_heat_flux=heat, **common)
    rtke = _d(W["tke_rhs"])
    sms3dtke.vdiff_tke(rtke=rtke, tke=tke, rho=rho, xkmv=xkmv,
                       l_diss=_d(W["l_diss"]), rdz=rdz, **common)
    rq = {}
    for name in ("qv", "qc", "qi"):
        rq[name] = cp.zeros((nz, ny, nx), cp.float32)
        base = None
        if name == "qv" and not c["mix_full_fields"]:
            base = _d(c["qv_base"][:nz])
        sms3dtke.vdiff_scalar(
            tend=rq[name], field=_d(c[name]), base=base, rho=rho, xkhv=xkhv,
            rdz=rdz, flux=_d(c["qfx"]) if name == "qv" else None, **common)
    for name, got in (("ru", ru), ("rv", rv), ("rw", rw), ("rt", rt),
                      ("tke_tend", rtke), ("rqv", rq["qv"]),
                      ("rqc", rq["qc"]), ("rqi", rq["qi"]),
                      ("hfx_final", hfx3)):
        out["implicit:" + name] = (got, W[name])
    return out


def compare(cases):
    """{field: (words compared, words different)} over every case."""
    tally = {}
    for c in cases:
        for key, (got, want) in run_case(c).items():
            got = cp.asnumpy(got)
            bad = int((got.view(np.uint32 if got.dtype == np.float32 else np.int32)
                       != want.view(np.uint32 if want.dtype == np.float32
                                    else np.int32)).sum())
            n, b = tally.get(key, (0, 0))
            tally[key] = (n + want.size, b + bad)
    return tally


@pytest.mark.parametrize("exact", ["0", "1"])
def test_sms3dtke_every_word(cases, monkeypatch, exact):
    monkeypatch.setenv("GPUWM_WRF_EXACT", exact)
    tally = compare(cases)
    bad = {k: v for k, v in tally.items() if v[1]}
    assert not bad, bad
    assert sum(n for n, _ in tally.values()) > 1_000_000


def test_fixture_exercises_the_scale_adaptive_range(cases):
    """The record spans LES to mesoscale and reaches the branches it names."""
    dxs = {float(c["dx"]) for c in cases}
    assert min(dxs) <= 100.0 and max(dxs) >= 9000.0
    assert {c["bnd"] for c in cases} == {0, 1}
    assert {c["isfflx"] for c in cases} == {0, 1, 2}
    convective = written = held = 0
    for c in cases:
        W = c["wrf"]
        changed = (W["gamu"] != c["gamu"])
        written += int(changed.sum())
        held += int((~changed).sum())
        convective += int((W["nlflux"] != 0).any(axis=0).sum())
    assert convective > 200 and written > 200 and held > 200
    pblh = np.concatenate([c["wrf"]["pblh"].ravel() for c in cases])
    assert (pblh == 0).sum() > 5
