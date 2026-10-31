"""cu_rad_feedback and Deng's cloud reach the RRTMG adapter, on a card.

The merges themselves are held to WRF's lines by
tests/test_cu_rad_feedback_oracle.py.  This file holds the WIRING: on a real
driver radiation call, the cloud fraction and condensate the legacy RRTMG
prep receives are exactly the merges applied, in WRF's order, to what the
same call receives with the cumulus fields zero -- KF's ICLOUD_CU = 2 merge
right after cal_cldfra1, then Deng's -- from the driver fields the schemes
leave (cldfra_dp, cldfra_sh, qc_cu, qi_cu, cw_rad).  And the schemes that
fill those fields run: KF with cu_rad_feedback, Grell-Freitas with
cu_diag = 1 (convtrans_prep).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import RunConfig, validate_run_config

cp = pytest.importorskip("cupy")
pytestmark = [pytest.mark.gpu, requires_gpu]


def _cfg(**overrides):
    base = dict(nx=8, ny=8, nz=40, dx=3000.0, dy=3000.0, ztop=18000.0,
                dt=20.0, run_seconds=0.0, moist=True, mp_physics=8,
                bl_pbl_physics=2, sf_sfclay_physics=2, sf_surface_physics=2,
                num_soil_layers=4, bldt=0.0, ra_physics=4,
                ra_rrtmg_variant="rrtmg_legacy", radt_minutes=1.0,
                shcu_physics=5, cu_physics=1, cudt_minutes=5.0,
                cu_rad_feedback=True)
    base.update(overrides)
    return validate_run_config(RunConfig(**base))


def _capture(driver, state, cfg, monkeypatch):
    """The keywords of the first legacy LW prep call of one radiation call."""
    from woof.core import physics as physics_module
    from woof.core import rrtmg_legacy_device

    seen = []

    class _Stop(Exception):
        pass

    def capture(**kw):
        seen.append({k: cp.asnumpy(v) for k, v in kw.items()
                     if k in ("cldfra3d", "qc3d", "qi3d")})
        raise _Stop

    monkeypatch.setattr(rrtmg_legacy_device, "lwrad_prep_batch_device",
                        capture)
    with pytest.raises(_Stop):
        driver._run_radiation(physics_module._prepare_atmosphere(state),
                              state, cfg)
    monkeypatch.undo()
    return seen[0]


def _columns(field, ncol):
    return cp.asnumpy(field).reshape(field.shape[0], ncol).T


def test_the_adapter_hands_rrtmg_the_merged_cloud(monkeypatch):
    from test_shcu_deng_driver_gpu import _build
    from woof.core import cumulus_radiation as curad

    cfg = _cfg()
    state, driver = _build(cfg)
    f = driver.fields
    nz, ny, nx = f["cldfra_sh"].shape
    ncol = ny * nx
    for name in ("cldfra_dp", "cldfra_sh", "qc_cu", "qi_cu", "cw_rad"):
        f[name][...] = 0.0
    base = _capture(driver, state, cfg, monkeypatch)
    rng = np.random.default_rng(7)
    seeded = {
        "cldfra_dp": rng.uniform(0.0, 0.6, (nz, ny, nx)) * (rng.random((nz, ny, nx)) < 0.4),
        "cldfra_sh": rng.uniform(0.0, 0.9, (nz, ny, nx)) * (rng.random((nz, ny, nx)) < 0.5),
        "qc_cu": rng.uniform(0.0, 8e-4, (nz, ny, nx)),
        "qi_cu": rng.uniform(0.0, 3e-4, (nz, ny, nx)),
        "cw_rad": rng.uniform(0.0, 5e-4, (nz, ny, nx)) * (rng.random((nz, ny, nx)) < 0.5),
    }
    for name, value in seeded.items():
        f[name][...] = cp.asarray(value.astype(np.float32))
    got = _capture(driver, state, cfg, monkeypatch)
    cols = {k: cp.asarray(_columns(f[k], ncol)) for k in seeded}
    cldfra, qc, qi = curad.merge_kf_cloud(
        cp.asarray(base["cldfra3d"]), cp.asarray(base["qc3d"]),
        cp.asarray(base["qi3d"]), cldfra_dp=cols["cldfra_dp"],
        cldfra_sh=cols["cldfra_sh"], qc_cu=cols["qc_cu"], qi_cu=cols["qi_cu"])
    cldfra, qc = curad.merge_deng_cloud(cldfra, qc, cldfra_sh=cols["cldfra_sh"],
                                        cw_rad=cols["cw_rad"])
    for name, want in (("cldfra3d", cldfra), ("qc3d", qc), ("qi3d", qi)):
        assert np.array_equal(got[name].view(np.uint32),
                              cp.asnumpy(want).view(np.uint32)), name
    assert not np.array_equal(got["cldfra3d"], base["cldfra3d"])


def test_kf_and_gf_fill_the_fields_the_radiation_reads():
    from test_shcu_deng_driver_gpu import _build
    from tilestream import harness

    cfg = _cfg(shcu_physics=0)
    state, driver = _build(cfg)
    harness.run_steps(state, cfg, 2)
    for name in ("cldfra_dp", "cldfra_sh", "qc_cu", "qi_cu"):
        assert name in driver.fields
        assert np.isfinite(cp.asnumpy(driver.fields[name])).all()
    cfg = _cfg(shcu_physics=0, cu_physics=3, cudt_minutes=0.0, cu_diag=1,
               bl_pbl_physics=1, sf_sfclay_physics=91)
    state, driver = _build(cfg)
    harness.run_steps(state, cfg, 2)
    clock = cp.asnumpy(driver.fields["g3_diag_clock"])
    assert int(clock[0]) == 2       # convtrans_prep ran on both steps
    for name in ("qc_cu", "qi_cu", "gd_cloud_a"):
        assert np.isfinite(cp.asnumpy(driver.fields[name])).all()


def test_farms_reads_the_same_merged_cloud():
    """swint_opt = 2: WRF's Solar_step forms the convective cloud before
    farms_driver reads QC/QI/CLDFRA (module_radiation_driver.F:3064), so
    FARMS's per-step cloud is the same merges of its zero-field cloud."""
    from test_shcu_deng_driver_gpu import _build
    from woof.core import cumulus_radiation as curad
    from woof.core import physics as physics_module

    cfg = _cfg(swint_opt=2)
    state, driver = _build(cfg)
    assert driver.farms is not None
    assert driver.farms.icloud_cu == 2 and driver.farms.deng_cloud
    f = driver.fields
    atmosphere = physics_module._prepare_atmosphere(state)
    names = ("cldfra_dp", "cldfra_sh", "qc_cu", "qi_cu", "cw_rad")
    for name in names:
        f[name][...] = 0.0
    base = [cp.asnumpy(a) for a in
            driver.farms.cloud_fraction(atmosphere, f, itimestep=2)]
    rng = np.random.default_rng(11)
    for name in names:
        f[name][...] = cp.asarray(rng.uniform(0.0, 0.5 if "cldfra" in name
                                               else 5e-4, f[name].shape)
                                  .astype(np.float32))
    got = [cp.asnumpy(a) for a in
           driver.farms.cloud_fraction(atmosphere, f, itimestep=2)]
    cldfra, qc, qi = curad.merge_kf_cloud(
        cp.asarray(base[0]), cp.asarray(base[1]), cp.asarray(base[2]),
        cldfra_dp=f["cldfra_dp"], cldfra_sh=f["cldfra_sh"],
        qc_cu=f["qc_cu"], qi_cu=f["qi_cu"])
    cldfra, qc = curad.merge_deng_cloud(cldfra, qc, cldfra_sh=f["cldfra_sh"],
                                        cw_rad=f["cw_rad"])
    for g, want in zip(got, (cldfra, qc, qi)):
        assert np.array_equal(g.view(np.uint32), cp.asnumpy(want).view(np.uint32))
