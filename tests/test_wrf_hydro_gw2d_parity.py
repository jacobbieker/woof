"""WRF-Hydro's 2-D groundwater model (the opt-in), bitwise against WRF 4.7.1's hydro/.

``woof_wrf_hydro.gw2d``: gw2d_ini (two pre-cycles), then HYDRO_drv's
GWBASESWCRT = 3 arm four times, three with the soil coupling and one
without, on a 12 x 14 routing grid over a 6 x 7 LSM grid.  The fixture is
``optional/wrf_hydro/oracle/build_gw2d.sh``: gwstep, gwSoilFlux,
darcyGwSoil, aggregateQsgw, sub_tri_solv and the driver arm cut byte for
byte from the pinned source and compiled as WRF-Hydro is, with MPP_LAND, on
one process.  Heads start in the aquifer, at its bottom and above the
surface, with ocean cells and inactive cells.  Every test skips, naming the
opt-in rule, unless ``WOOF_ENABLE_WRF_HYDRO=1``.
"""

from __future__ import annotations

import csv
import hashlib

import numpy as np
import pytest

from conftest import requires_gpu
from woof import optin

REASON = optin.unavailable_reason("wrf_hydro")
pytestmark = pytest.mark.skipif(REASON is not None, reason=f"opt-in rule: {REASON}")

FIX = optin.optional_root() / "wrf_hydro" / "oracle" / "fixtures" / "gw2d"
IX, JX, AGG, NSOIL, DX, DT = 12, 14, 2, 4, 250.0, 3600.0
SLDPTH = (0.10, 0.30, 0.60, 1.00)
ZSOIL = (-0.10, -0.40, -1.00, -2.00)


def _rows(name):
    with (FIX / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def _bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def _plane(rows, key, dtype=np.float32):
    out = np.zeros((JX, IX), dtype)
    for r in rows:
        out[int(r["j"]) - 1, int(r["i"]) - 1] = dtype(float(r[key])) if dtype is np.float32 else int(r[key])
    return out


def test_fixtures_are_the_ones_the_build_hashed():
    rows = (FIX / "gw2d-sha256sums.txt").read_text().splitlines()
    pinned = {line.split()[1]: line.split()[0] for line in rows}
    for name in ("gw2d-static.csv", "gw2d-steps.csv", "gw2d-totals.csv", "gw2d-qsgw.csv"):
        assert pinned[name] == hashlib.sha256((FIX / name).read_bytes()).hexdigest()


@pytest.mark.gpu
@requires_gpu
def test_gw2d_is_bitwise_against_wrf_hydro():
    import cupy as cp

    mod = optin.load("wrf_hydro")
    static = _rows("gw2d-static.csv")
    steps = {}
    for r in _rows("gw2d-steps.csv"):
        steps.setdefault(int(r["step"]), []).append(r)
    totals = {int(r["step"]): r for r in _rows("gw2d-totals.csv")}
    qsgw = {}
    for r in _rows("gw2d-qsgw.csv"):
        qsgw.setdefault(int(r["step"]), []).append(r)
    elev = _plane(static, "elev")
    gw = mod.Gw2d(ltype=_plane(static, "ltype", np.int32), elev=elev, bot=_plane(static, "bot"),
                  hycond=_plane(static, "hycond"), poros=_plane(static, "poros"),
                  ho=_plane(static, "ho0"), dx=DX, dt=DT, aggfactrt=AGG)
    smc = cp.asarray(np.stack([_plane(steps[0], f"smc{k}") for k in range(1, NSOIL + 1)]))
    smcmax = cp.asarray(np.stack([_plane(static, f"smcmax{k}") for k in range(1, NSOIL + 1)]))
    soiltyp = _plane(static, "soiltyp", np.int32)
    head = cp.zeros((JX, IX), cp.float32)
    gw.init(gw_pre_cycles=2)

    def check(n):
        want = steps[n]
        for name, got in (("ho", gw.ho), ("h", gw.h), ("convgw", gw.convgw),
                          ("excess", gw.excess), ("qsgwrt", gw.qsgwrt),
                          ("qdarcyrt", gw.qdarcyrt), ("sfchead", head), ("hycond", gw.hycond)):
            bad = _bits(got.get()) != _bits(_plane(want, name))
            assert not bad.any(), f"step {n} {name}: {int(bad.sum())} cells differ"
        for k in range(NSOIL):
            bad = _bits(smc[k].get()) != _bits(_plane(want, f"smc{k + 1}"))
            assert not bad.any(), f"step {n} smc{k + 1}: {int(bad.sum())} cells differ"
        t = totals[n]
        assert gw.istep == int(t["istep"]), n
        assert _bits(gw.e.get()).tolist() == _bits([np.float32(t["ebot"]), np.float32(t["eocn"])]).tolist(), n

    check(0)
    for n in range(1, 5):
        gw.hydro_step(surface_head=head, gwsoilcpl=1 if n <= 3 else 0, elrt=cp.asarray(elev),
                      soiltyp=soiltyp, smcrt=smc, smcmaxrt=smcmax, sldpth=SLDPTH, zsoil=ZSOIL)
        check(n)
        want = np.zeros((JX // AGG, IX // AGG), np.float32)
        for r in qsgw[n]:
            want[int(r["j"]) - 1, int(r["i"]) - 1] = np.float32(r["qsgw"])
        assert not (_bits(gw.qsgw.get()) != _bits(want)).any(), f"step {n} qsgw"


def test_unported_gw2d_shapes_are_refused_by_name():
    mod = optin.load("wrf_hydro")
    z = np.zeros((2, 4), np.float32)
    with pytest.raises(NotImplementedError, match="zfac"):
        mod.Gw2d(ltype=np.ones((2, 4), np.int32), elev=z, bot=z, hycond=z, poros=z, ho=z,
                 dx=1.0, dt=1.0, aggfactrt=2)
