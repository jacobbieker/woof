"""icloud = 3 reaches both RRTMG engines as WRF's driver hands it over.

WRF's radiation driver (module_radiation_driver.F:1462-1510) runs
cal_cldfra3 on copies of each column, hands the schemes its CLDFRA and its
qc/qi (with the made-up condensate of partial layers) and restores the
prognostic qc/qi afterwards (:3137-3155).  The kernel itself is graded word
for word by tests/test_cldfra3_wrf471_oracle.py; this file holds the
wiring: the cloud fraction each engine publishes (grid%cldfra) is the
routine's, the condensate the legacy wrapper preparation reads is the
routine's, the model state is not touched, and the coupling changes the
radiation.
"""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from test_rrtmgp_subgrid_cloud_radii import _ETA, _P_TOP, _hot_afternoon_column

F = np.float32


def _columns():
    """Six land/water columns of the afternoon sounding, moistened into
    near-saturated layers, with a resolved liquid deck and a cirrus deck."""
    plev, play, t, qv, dz = _hot_afternoon_column()
    nz, nx = play.size, 6
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    sat = 0.622 * es / (play - es)
    shape = (nz, 1, nx)
    qv3 = np.empty(shape, F)
    qc3 = np.zeros(shape, F)
    qi3 = np.zeros(shape, F)
    qs3 = np.zeros(shape, F)
    rng = np.random.default_rng(3)
    for i in range(nx):
        rh = np.clip(0.55 + 0.4 * np.sin(np.arange(nz) / (2.0 + i)), 0.05, 0.99)
        qv3[:, 0, i] = np.where(play < 10000.0, 3e-6, rh * sat)
    k700 = int(np.argmin(np.abs(play - 70000.0)))
    k300 = int(np.argmin(np.abs(play - 30000.0)))
    qc3[k700:k700 + 2, 0, 1::2] = 2e-4
    qc3[k700 + 3, 0, :] = 4e-7          # the log10 arm
    qi3[k300:k300 + 2, 0, ::3] = 3e-6
    qs3[k300 - 2:k300, 0, 2] = 5e-6
    xland = np.array([1, 2, 1, 2, 1, 2], F).reshape(1, nx)
    return plev, play, t, dz, qv3, qc3, qi3, qs3, xland, rng


def _call(engine, icloud, monkeypatch=None, spy=None):
    plev, play, t, dz, qv3, qc3, qi3, qs3, xland, _ = _columns()
    nz, ny, nx = qv3.shape
    shape = qv3.shape

    def expand(x, nk=nz):
        return np.broadcast_to(np.asarray(x, F)[:, None, None], (nk, ny, nx)).copy()

    exner = (play / 1.0e5) ** (287.0 / 1004.0)
    z_at_w = np.concatenate([[0.0], np.cumsum(dz)])
    dnw = np.diff(_ETA.astype(np.float64))
    fnm, fnp, dn = np.zeros(nz), np.zeros(nz), np.zeros(nz)
    dn[1:] = 0.5 * (dnw[1:] + dnw[:-1])
    fnp[1:] = 0.5 * dnw[1:] / dn[1:]
    fnm[1:] = 0.5 * dnw[:-1] / dn[1:]
    D = cp.asarray
    zeros = cp.zeros(shape, cp.float32)
    state = SimpleNamespace(
        elapsed_seconds=600.0, qv=D(qv3), qc=D(qc3), qi=D(qi3), qr=zeros,
        qs=D(qs3), qg=zeros, effc=cp.full(shape, 2.49, cp.float32),
        effi=cp.full(shape, 4.99, cp.float32), effs=cp.full(shape, 9.99, cp.float32),
        fnm=D(fnm.astype(F)), fnp=D(fnp.astype(F)), p_top=np.float32(_P_TOP),
        physics=None)
    atmosphere = {
        "pressure": D(expand(play)), "p_interface": D(expand(plev, nz + 1)),
        "temperature": D(expand(t)), "theta": D(expand(t / exner)),
        "exner": D(expand(exner)), "dz": D(expand(dz)),
        "z_interface": D(expand(z_at_w, nz + 1)),
        "qv": state.qv, "qc": state.qc, "qi": state.qi,
    }
    fields = {
        "tsk": cp.full((ny, nx), 305.0, cp.float32),
        "albedo": cp.full((ny, nx), 0.17, cp.float32),
        "emiss": cp.full((ny, nx), 0.95, cp.float32),
        "xland": D(xland), "xice": cp.zeros((ny, nx), cp.float32),
        "snow": cp.zeros((ny, nx), cp.float32),
    }
    start = datetime(2026, 8, 27, 20)
    lat = cp.full((ny, nx), 34.21, cp.float32)
    lon = cp.full((ny, nx), -118.49, cp.float32)
    if engine == "rte":
        from woof.core.rrtmgp import RRTMGPRadiation
        adapter = RRTMGPRadiation(start, lat, lon)
    else:
        from woof.core.rrtmg_legacy import RRTMGLegacyRadiation
        adapter = RRTMGLegacyRadiation(start, lat, lon, p_top=_P_TOP)
    before = (state.qc.copy(), state.qi.copy())
    cfg = SimpleNamespace(mp_physics=8, dt=60.0, radt=12.0, radt_minutes=12.0,
                          bl_pbl_physics=0, icloud_bl=0, icloud=icloud,
                          sf_surface_physics=2, dx=3000.0, dy=3000.0)
    result = adapter(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    assert cp.array_equal(state.qc, before[0]) and cp.array_equal(state.qi, before[1]), \
        "the radiation wrote the prognostic qc/qi"
    inputs = dict(qv=qv3, qc=qc3, qi=qi3, qs=qs3, dz=expand(dz), p=expand(play),
                  t=expand(t), xland=xland)
    return result, inputs


def _cols(x):
    x = np.asarray(x)
    return np.ascontiguousarray(x.transpose(1, 2, 0).reshape(-1, x.shape[0]))


def _direct(inputs):
    from woof.core.cldfra3 import cal_cldfra3, cldfra3_gridkm
    D = lambda a: cp.asarray(_cols(a))
    cf, qc, qi = cal_cldfra3(D(inputs["qv"]), D(inputs["qc"]), D(inputs["qi"]),
                             D(inputs["qs"]), D(inputs["dz"]), D(inputs["p"]),
                             D(inputs["t"]), cp.asarray(inputs["xland"].reshape(-1)),
                             cldfra3_gridkm(3000.0, 3000.0))
    return cp.asnumpy(cf), cp.asnumpy(qc), cp.asnumpy(qi)


@pytest.mark.parametrize("engine", ("rte", "legacy"))
def test_engine_publishes_cal_cldfra3(engine):
    result, inputs = _call(engine, 3)
    cf, qc, qi = _direct(inputs)
    got = _cols(cp.asnumpy(result.cldfra))
    np.testing.assert_array_equal(got.view("u4"), cf.view("u4"))
    # the column set reaches the arms: overcast, partial and clear layers,
    # and condensate the routine made up in partial layers
    assert (cf == 1).any() and ((cf > 0) & (cf < 1)).sum() > 20 and (cf == 0).any()
    assert (qc > _cols(inputs["qc"])).any() and (qi > _cols(inputs["qi"])).any()
    base, _ = _call(engine, 1)
    assert not np.array_equal(cp.asnumpy(base.cldfra), cp.asnumpy(result.cldfra))
    assert not np.array_equal(cp.asnumpy(base.swdown), cp.asnumpy(result.swdown))


def test_legacy_wrapper_reads_the_routines_condensate(monkeypatch):
    from woof.core import rrtmg_legacy_device as dev
    seen = []
    real = dev.lwrad_prep_batch_device

    def spy(**kw):
        seen.append({k: cp.asnumpy(kw[k]) for k in ("qc3d", "qi3d", "cldfra3d")})
        return real(**kw)

    monkeypatch.setattr(dev, "lwrad_prep_batch_device", spy)
    _, inputs = _call("legacy", 3)
    cf, qc, qi = _direct(inputs)
    assert seen
    got = {k: np.concatenate([s[k] for s in seen]) for k in seen[0]}
    np.testing.assert_array_equal(got["qc3d"].view("u4"), qc.view("u4"))
    np.testing.assert_array_equal(got["qi3d"].view("u4"), qi.view("u4"))
    np.testing.assert_array_equal(got["cldfra3d"].view("u4"), cf.view("u4"))

