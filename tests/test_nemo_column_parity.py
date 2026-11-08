"""The coupled NEMO tracer column (the opt-in), word for word against NEMO 5.0.2.

``woof_nemo.column.NemoColumnOcean`` steps the ported pieces in stpmlf.F90's
order behind WRF's coupler.  The fixture is
``optional/nemo/oracle/build_column.sh``: eosbn2, traqsr, zdfsh2, zdfdrg,
zdftke, trazdf, trasbc and traatf_qco compiled whole and unchanged, with
sbccpl.F90's sbc_cpl_rcv cut byte for byte, run for six steps of 900 s with
WRF's six fields delivered every two steps, over 5 x 4 columns with a shelf,
a land point and statically unstable layers.  Two configurations: TEOS-10
with non-linear bottom drag and 2-band penetrative light; EOS-80 with the
log-layer drag, no light penetration and TKE nn_mxl = 2, nn_etau = 0.  Every
test skips, naming the opt-in rule, unless ``WOOF_ENABLE_NEMO=1``.
"""

from __future__ import annotations

import csv
import hashlib

import numpy as np
import pytest

from conftest import requires_gpu
from woof import optin

REASON = optin.unavailable_reason("nemo")
pytestmark = pytest.mark.skipif(REASON is not None, reason=f"opt-in rule: {REASON}")

FIX = optin.optional_root() / "nemo" / "oracle" / "fixtures" / "column"
NX, NY, NZ, IS0, JS0 = 5, 4, 21, 3, 3     # interior columns; NEMO's first interior index
CONFIGS = {
    "teos10": dict(eos="teos10", traqsr=True, drag="non_lin"),
    "eos80": dict(eos="eos80", traqsr=False, drag="loglayer", tke=dict(nn_mxl=2, nn_etau=0)),
}
STATE = ("tb", "sb", "tn", "sn", "en", "avt", "avm_k", "rn2")
SURFACE = ("sst_m", "sss_m", "utau", "vtau", "taum", "wndm", "qns", "qsr", "emp",
           "sbc_tsc_t", "sbc_tsc_s", "rcdu_bot")


def _rows(name):
    with (FIX / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def _ij(r):
    return int(r["jj"]) - JS0, int(r["ji"]) - IS0


def _grid(cfg):
    rows = _rows(f"column-{cfg}-grid.csv")
    lev = {k: np.zeros(NZ) for k in ("e3t", "e3w", "gdept", "gdepw")}
    mbkt = np.zeros((NY, NX), np.int32)
    lat = np.zeros((NY, NX))
    t0, s0 = np.zeros((NZ, NY, NX)), np.zeros((NZ, NY, NX))
    for r in rows:
        j, i = _ij(r)
        k = int(r["jk"]) - 1
        mbkt[j, i] = int(r["mbkt"])
        lat[j, i] = float(r["gphit"])
        for key in lev:
            lev[key][k] = float(r[key])
        t0[k, j, i], s0[k, j, i] = float(r["t0"]), float(r["s0"])
    return dict(mbkt=mbkt, latitude=lat, temperature=t0, salinity=s0, **lev)


def _forcing(cfg):
    out = {}
    for r in _rows(f"column-{cfg}-forcing.csv"):
        w = int(r["window"])
        planes = out.setdefault(w, {k: np.zeros((NY, NX), np.float32) for k in (
            "evap_precip", "surf_net_solar", "surf_net_non_solar", "taux", "tauy", "taumod")})
        j, i = _ij(r)
        for k in planes:
            planes[k][j, i] = np.float32(r[k])
    return out


def _by_step(name, keys, three_d):
    out = {}
    for r in _rows(name):
        kt = int(r["kt"])
        d = out.setdefault(kt, {k: np.zeros((NZ, NY, NX) if three_d else (NY, NX)) for k in keys})
        j, i = _ij(r)
        for k in keys:
            if three_d:
                d[k][int(r["jk"]) - 1, j, i] = float(r[k])
            else:
                d[k][j, i] = float(r[k])
    return out


def _sst_out(cfg):
    out = {}
    for r in _rows(f"column-{cfg}-surface.csv"):
        a = out.setdefault(int(r["kt"]), np.zeros((NY, NX), np.float32))
        a[_ij(r)] = np.float32(r["sst_out"])
    return out


def _same(name, got, want):
    got, want = np.asarray(got), np.asarray(want)
    bad = got.view(np.int64 if got.dtype == np.float64 else np.int32) != \
        want.view(np.int64 if want.dtype == np.float64 else np.int32)
    assert not bad.any(), (f"{name}: {int(bad.sum())} of {bad.size} words differ; first at "
                           f"{np.argwhere(bad)[0].tolist()}: {got[bad][0]!r} != {want[bad][0]!r}")


def test_fixtures_are_the_ones_the_build_hashed():
    rows = (FIX / "column-sha256sums.txt").read_text().splitlines()
    pinned = {line.split()[1]: line.split()[0] for line in rows}
    for cfg in CONFIGS:
        for part in ("grid", "forcing", "steps", "surface"):
            name = f"column-{cfg}-{part}.csv"
            assert pinned[name] == hashlib.sha256((FIX / name).read_bytes()).hexdigest()


@pytest.mark.parametrize("kw,word", [
    (dict(nn_ice=2), "nn_ice"), (dict(ln_rnf=True), "ln_rnf"), (dict(ln_rstart=True), "ln_rstart"),
    (dict(ln_mixcpl=True), "ln_mixcpl"), (dict(ln_dm2dc=True), "ln_dm2dc"),
    (dict(ln_zdfevd=True), "ln_zdfevd"), (dict(ln_zdfddm=True), "ln_zdfddm"),
    (dict(ln_boost=True), "ln_boost"), (dict(nn_avb=1), "nn_avb"), (dict(nn_havtb=1), "nn_havtb"),
    (dict(vertical_coordinate="zps"), "ln_zps"), (dict(sst_loctrans="AVERAGE"), "LOCTRANS"),
    (dict(emp_receive="conservative"), "sn_rcv_emp"),
    (dict(stress_orientation="eastward-northward"), "sn_rcv_tau"),
])
def test_unported_column_options_are_refused_by_name(kw, word):
    mod = optin.load("nemo")
    with pytest.raises(NotImplementedError, match=word):
        mod.NemoColumnOcean(rn_Dt=900.0, coupling_interval=1800.0, **_grid("teos10"), **kw)


def _ocean(cfg):
    mod = optin.load("nemo")
    return mod.NemoColumnOcean(rn_Dt=900.0, coupling_interval=1800.0, **_grid(cfg),
                               **CONFIGS[cfg])


def _fluxes(planes):
    import cupy as cp

    from woof.ocean.wrf_cpl import AtmosphereToOcean

    return AtmosphereToOcean(**{k: cp.asarray(v) for k, v in planes.items()})


def _inner(a):
    return a[..., 1:-1, 1:-1].get()


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("cfg", sorted(CONFIGS))
def test_every_nemo_step_is_word_exact(cfg):
    forcing = _forcing(cfg)
    state = _by_step(f"column-{cfg}-steps.csv", STATE, True)
    surf = _by_step(f"column-{cfg}-surface.csv", SURFACE, False)
    sst = _sst_out(cfg)
    oc = _ocean(cfg)
    for kt in range(1, 7):
        if (kt - 1) % 2 == 0:
            oc.set_fluxes(_fluxes(forcing[(kt - 1) // 2 + 1]))
        oc.nemo_step()
        got = dict(tb=oc.ts[oc.nbb, 0], sb=oc.ts[oc.nbb, 1], tn=oc.ts[oc.nnn, 0],
                   sn=oc.ts[oc.nnn, 1], en=oc.en, avt=oc.avt, avm_k=oc.avm_k, rn2=oc.rn2)
        for k in STATE:
            _same(f"step {kt} {k}", _inner(got[k]), state[kt][k])
        gs = dict(sst_m=oc.sst_m, sss_m=oc.sss_m, utau=oc.utau, vtau=oc.vtau, taum=oc.taum,
                  wndm=oc.wndm, qns=oc.qns, qsr=oc.qsr, emp=oc.emp, sbc_tsc_t=oc.sbc_tsc[0],
                  sbc_tsc_s=oc.sbc_tsc[1], rcdu_bot=oc.rCdU_bot)
        for k in SURFACE:
            _same(f"step {kt} {k}", _inner(gs[k]), surf[kt][k])
        _same(f"step {kt} sst sent", oc.surface.sst.get(), sst[kt])


@pytest.mark.gpu
@requires_gpu
def test_external_ocean_interface_steps_whole_coupling_intervals():
    from woof.ocean.interface import ExternalOcean

    cfg = "teos10"
    forcing = _forcing(cfg)
    state = _by_step(f"column-{cfg}-steps.csv", STATE, True)
    sst = _sst_out(cfg)
    oc = _ocean(cfg)
    assert isinstance(oc, ExternalOcean)
    with pytest.raises(RuntimeError, match="set_fluxes"):
        oc.step(1800.0)
    with pytest.raises(ValueError, match="coupling interval"):
        oc.step(900.0)
    for w in (1, 2, 3):
        oc.set_fluxes(_fluxes(forcing[w]))
        oc.step(1800.0)
        _same(f"window {w} sst", oc.surface.sst.get(), sst[2 * w])
        _same(f"window {w} tn", _inner(oc.ts[oc.nnn, 0]), state[2 * w]["tn"])
    assert float(abs(oc.surface.uoce).max()) == 0.0
    assert float(abs(oc.surface.voce).max()) == 0.0
