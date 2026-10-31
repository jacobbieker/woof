"""km_opt = 5 (SMS-3DTKE) in the whole model: admission, a short gray-zone
run against km_opt = 2, and restart bit-identity.

The closure's words are held to compiled WRF v4.7.1 by
tests/test_sms3dtke_wrf471_parity.py; this file checks the wiring: the
refusals WRF's check_a_mundo makes (and woof's own), that a convective
1 km run stays finite and bounded beside the same run on km_opt = 2, that
the scale-adaptive state (PBLH, KPBL, GAMU/GAMV) is written, and that a
run interrupted by a restart is the uninterrupted run to the bit.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from woof.config import ConfigurationRefused, RunConfig, validate_run_config
from woof.io import restart

BASE = dict(nx=10, ny=10, nz=24, dx=1000.0, dy=1000.0, ztop=6000.0,
            dt=5.0, run_seconds=0.0, time_step_sound=4, moist=True,
            mp_physics=1, sf_sfclay_physics=1, sf_surface_physics=2,
            bl_pbl_physics=0, km_opt=5, diff_opt=2, isfflx=1, c_k=0.15,
            ra_physics=0, cu_physics=0, bldt=0.0)


def _cfg(**kw):
    values = dict(BASE)
    values.update(kw)
    return RunConfig(**values)


@pytest.mark.parametrize("override,needle", [
    (dict(bl_pbl_physics=1), "bl_pbl_physics=0"),
    (dict(diff_opt=1), "diff_opt"),
    (dict(sf_sfclay_physics=2), "sf_sfclay_physics = 0, 1, 5"),
    (dict(open_x=True), "radiative open"),
    (dict(mix_isotropic=1), "mix_isotropic"),
    (dict(tke_budget=1), "tke_budget"),
    (dict(sf_sfclay_physics=0, sf_surface_physics=0, isfflx=0,
          tke_heat_flux=0.24), "0/0"),
])
def test_refusals(override, needle):
    from woof.config import validate_sms3dtke
    with pytest.raises(ConfigurationRefused, match=needle):
        validate_sms3dtke(_cfg(**override))
    with pytest.raises(ConfigurationRefused):
        validate_run_config(_cfg(**override))


def test_admitted_with_each_wrf_surface_layer():
    for sfclay in (1, 5, 91):
        validate_run_config(_cfg(sf_sfclay_physics=sfclay))


def _state(cp, km_opt):
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics
    cfg = _cfg(km_opt=km_opt)
    validate_run_config(cfg)
    coord = make_vertical_coord(cfg.nz)

    def theta(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1000.0, 299.0, 299.0 + 0.005 * (z - 1000.0))
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(
        cfg, coord, base,
        lambda z: 0.010 * np.exp(-np.asarray(z, np.float64) / 2500.0))
    state.u[...] = cp.float32(5.0)
    state.v[...] = cp.float32(1.0)
    rng = np.random.default_rng(3)
    state.thp[:3] += cp.asarray(
        0.2 * rng.standard_normal((3, cfg.ny, cfg.nx)), dtype=cp.float32)
    shape = (cfg.ny, cfg.nx)
    tsk = np.full(shape, 306.0)
    soil = np.full((4,) + shape, 0.25)
    initialize_physics(
        state, cfg, landmask=np.ones(shape), tsk=tsk,
        soil_temperature=np.stack([tsk - 2, tsk - 4, tsk - 6, tsk - 8]),
        soil_moisture=soil, liquid_moisture=soil,
        ivgtyp=np.full(shape, 10), isltyp=np.full(shape, 6), vegfra=40.0,
        tmn=290.0, swdown=700.0, glw=350.0, pblh=500.0,
        radiation_start_time=datetime(2020, 7, 1, 18),
        radiation_latitude=np.full(shape, 39.0),
        radiation_longitude=np.full(shape, -97.0))
    return state, cfg


@pytest.mark.gpu
def test_convective_run_beside_km_opt_2():
    """A 1 km convective boundary layer, 30 steps: km_opt = 5 stays finite
    and bounded, warms and mixes the surface layer as km_opt = 2 does,
    diagnoses its own PBL height and writes the nonlocal gammas."""
    cp = pytest.importorskip("cupy")
    from woof.core.dycore import run_steps, stability_report
    out = {}
    for km in (2, 5):
        state, cfg = _state(cp, km)
        th0 = float(state.thp[0].mean())
        run_steps(state, cfg, 30)
        assert not stability_report(state, cfg)["nan"]
        for name in ("u", "v", "w", "thp", "tke"):
            assert bool(cp.isfinite(getattr(state, name)).all()), (km, name)
        f = state.physics.fields
        out[km] = dict(
            dth=float(state.thp[0].mean()) - th0,
            tke=float(state.tke.max()), tke_min=float(state.tke.min()),
            umax=float(cp.abs(state.u).max()),
            wmax=float(cp.abs(state.w).max()),
            pblh=cp.asnumpy(f["pblh"]), kpbl=cp.asnumpy(f["kpbl"]),
            gamu=cp.asnumpy(getattr(state, "gamu", cp.zeros(1))))
    k2, k5 = out[2], out[5]
    assert k5["tke_min"] >= 0.0 and 1e-4 < k5["tke"] < cfg_tke_bound()
    assert k5["dth"] > 0.0 and 0.2 < k5["dth"] / k2["dth"] < 5.0
    assert k5["umax"] < 15.0 and k5["wmax"] < 5.0
    # nonlocal_flux's theta-method PBL height replaced the 500 m seed.
    assert (k5["pblh"] != 500.0).all() and (k5["kpbl"] > 1).all()
    assert np.abs(k5["gamu"]).max() > 0.0
    # km_opt = 2 leaves PBLH alone (no PBL scheme writes it).
    assert (k2["pblh"] == 500.0).all()


def cfg_tke_bound() -> float:
    return float(RunConfig.__dataclass_fields__["tke_upper_bound"].default)


@pytest.mark.gpu
def test_restart_is_bit_identical(tmp_path):
    """20 steps == 10 steps + restart + 10 steps, on every serialized
    word (TKE, GAMU/GAMV, the surface fields km_opt = 5 writes)."""
    cp = pytest.importorskip("cupy")
    from woof.core.dycore import run_steps
    from test_restart import _assert_restart_equal

    state_a, cfg = _state(cp, 5)
    run_steps(state_a, cfg, 20)
    reference = restart.write_restart(tmp_path / "ref.npz", state_a, cfg)

    state_b, cfg_b = _state(cp, 5)
    run_steps(state_b, cfg_b, 10)
    mid = restart.write_restart(tmp_path / "mid.npz", state_b, cfg_b)
    keys = set(restart.read_restart_header(mid)["array_manifest"])
    assert {"state/tke", "state/gamu", "state/gamv"} <= keys
    assert not any(k.startswith("scratch/sms_") for k in keys)

    state_c, cfg_c = _state(cp, 5)
    restart.restore_restart(mid, state_c, cfg_c)
    run_steps(state_c, cfg_c, 10)
    resumed = restart.write_restart(tmp_path / "resumed.npz", state_c, cfg_c)
    _assert_restart_equal(resumed, reference)
    assert float(cp.abs(state_c.gamu).max()) > 0.0
