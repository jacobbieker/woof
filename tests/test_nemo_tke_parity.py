"""NEMO TKE vertical mixing (the opt-in), bitwise against NEMO 5.0.2.

zdf_sh2 and zdf_tke run by ``woof_nemo.tke``.  The fixture is
``optional/nemo/oracle/build_tke.sh``: zdfsh2.F90 and zdftke.F90 compiled
whole and unchanged against NEMO-named service modules, zdf_tke_init reading
namelist_ref's &namzdf_tke plus four overrides (the reference; nn_mxl = 2
without Prandtl number, ln_mxl0, Langmuir cells or penetration; nn_mxl = 0
with constant htau; nn_mxl = 1), three steps each over a 9 x 7 x 21 grid
with a halo, land, a shelf and unstable layers.  Every test skips, naming
the opt-in rule, unless ``WOOF_ENABLE_NEMO=1``.
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

FIX = optin.optional_root() / "nemo" / "oracle" / "fixtures" / "tke"
NX, NY, NZ, NSTEP = 9, 7, 21, 3
CONFIGS = {
    "ref": {},
    "mxl2": dict(nn_mxl=2, nn_pdl=0, ln_mxl0=False, ln_lc=False, nn_etau=0),
    "mxl0": dict(nn_mxl=0, nn_htau=0),
    "mxl1": dict(nn_mxl=1),
}


def _rows(name):
    with (FIX / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def _f3(rows, key):
    a = np.zeros((NZ, NY, NX))
    for r in rows:
        a[int(r["jk"]) - 1, int(r["jj"]) - 1, int(r["ji"]) - 1] = float(r[key])
    return a


def _f2(rows, key, dtype=float):
    a = np.zeros((NY, NX), np.int32 if dtype is int else np.float64)
    for r in rows:
        if r["jk"] == "1":
            a[int(r["jj"]) - 1, int(r["ji"]) - 1] = dtype(r[key])
    return a


def test_fixtures_are_the_ones_the_build_hashed():
    rows = (FIX / "tke-sha256sums.txt").read_text().splitlines()
    pinned = {line.split()[1]: line.split()[0] for line in rows}
    for name in ("tke-grid.csv", *(f"tke-{c}.csv" for c in CONFIGS)):
        assert pinned[name] == hashlib.sha256((FIX / name).read_bytes()).hexdigest()


@pytest.mark.parametrize("kwargs,word", [
    (dict(nn_ice=1), "sea-ice model"), (dict(ln_isfcav=True), "ln_isfcav"),
    (dict(ln_zdfiwm=True), "ln_zdfiwm"), (dict(wave_coupling=True), "wave coupling"),
    (dict(ln_mxhsw=True), "ln_mxhsw"), (dict(nn_etau=2), "nn_etau = 2"),
    (dict(ln_drg_off=True), "ln_drg_OFF")])
def test_unported_tke_options_are_refused_by_name(kwargs, word):
    mod = optin.load("nemo")
    with pytest.raises(NotImplementedError, match=word):
        mod.TkeMixing(rn_Dt=900.0, gphit=None, **kwargs)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("config", list(CONFIGS))
def test_tke_is_bitwise_against_nemo(config):
    import cupy as cp

    mod = optin.load("nemo")
    g = _rows("tke-grid.csv")
    d = lambda a: cp.asarray(np.ascontiguousarray(a))  # noqa: E731
    f = {k: d(_f3(g, k)) for k in ("tmask", "umask", "vmask", "wmask", "wumask", "wvmask", "e3t",
                                    "e3w", "e3uw", "e3vw", "gdepw", "u1", "u2", "v1", "v2",
                                    "rn2", "rn2b")}
    gphit = d(_f2(g, "gphit"))
    tke = mod.TkeMixing(rn_Dt=900.0, gphit=gphit, **CONFIGS[config])
    en, dissl = tke.start_from_rest(f["wmask"])
    avmb = cp.full(NZ, 1.2e-4, cp.float64)
    avtb = cp.full(NZ, 1.2e-5, cp.float64)
    avm = cp.ascontiguousarray(avmb[:, None, None] * f["wmask"])
    avt = cp.ascontiguousarray(avtb[:, None, None] * f["wmask"])
    sh2 = cp.zeros((NZ, NY, NX), cp.float64)
    want = _rows(f"tke-{config}.csv")
    for kt in range(1, NSTEP + 1):
        tke.sh2(avm=avm, uu_b=f["u1"], uu_n=f["u2"], vv_b=f["v1"], vv_n=f["v2"], e3uw=f["e3uw"],
                e3vw=f["e3vw"], wumask=f["wumask"], wvmask=f["wvmask"], umask=f["umask"],
                vmask=f["vmask"], sh2=sh2)
        tke.step(sh2=sh2, avm=avm, avt=avt, en=en, dissl=dissl, rn2=f["rn2"], rn2b=f["rn2b"],
                 taum=d(_f2(g, "taum")), rCdU_bot=d(_f2(g, "rcdu_bot")), uu_b=f["u1"], vv_b=f["v1"],
                 umask=f["umask"], vmask=f["vmask"], tmask=f["tmask"], wmask=f["wmask"],
                 ssmask=d(_f2(g, "ssmask")), mbkt=d(_f2(g, "mbkt", int)), e3t=f["e3t"], e3w=f["e3w"],
                 gdepw=f["gdepw"], avmb=avmb, avtb=avtb, avtb_2d=cp.ones((NY, NX), cp.float64))
        rows = [r for r in want if r["kt"] == str(kt)]
        inner = (slice(None), slice(1, NY - 1), slice(1, NX - 1))
        for key, arr in (("sh2", sh2), ("en", en), ("avm", avm), ("avt", avt)):
            got = cp.asnumpy(arr)[inner]
            ref = _f3(rows, key)[inner]
            assert np.array_equal(got.view(np.uint64), ref.view(np.uint64)), (kt, key)
