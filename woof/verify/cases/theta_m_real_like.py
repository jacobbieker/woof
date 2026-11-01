"""A real-data-like use_theta_m case: Thompson + RRTMG + YSU/MYNN + Grell-Freitas.

A 12 km periodic patch over land with a warm, moist, conditionally unstable
column (surface theta 300 K, 3 K/km aloft; qv 16 g/kg at the ground with a
2.2 km scale height), a small seeded theta perturbation to break the
symmetry, the RRTMG radiation pair, a surface layer and PBL (YSU with the
revised MM5 layer, or MYNN with its own layer), Grell-Freitas on the model
step and classic Thompson.  The initial state is init_moist_balanced's, which
under use_theta_m = 1 ends with ideal.exe's conversion to theta_m
(module_initialize_ideal.F:1726), and the physics driver is
initialize_physics's.  tests/test_theta_m_whole_model.py runs it in both
theta modes.
"""

from __future__ import annotations

import dataclasses

import numpy as np

PBL = {"ysu": dict(bl_pbl_physics=1, sf_sfclay_physics=1),
       "mynn": dict(bl_pbl_physics=5, sf_sfclay_physics=5)}


def config(*, use_theta_m: int, pbl: str = "ysu", nx: int = 16, ny: int = 16,
           nz: int = 30, dt: float = 60.0, **changes):
    from woof.config import RunConfig, validate_run_config
    base = RunConfig(
        nx=nx, ny=ny, nz=nz, dx=12000.0, dy=12000.0, ztop=16000.0,
        dt=dt, run_seconds=0.0, moist=True, mp_physics=8,
        ra_lw_physics=4, ra_sw_physics=4, radt_minutes=5.0,
        cu_physics=3, cudt_minutes=0.0, ishallow=0, bldt=0.0,
        diff_opt=1, km_opt=4, **PBL[pbl])
    return validate_run_config(dataclasses.replace(
        base, use_theta_m=use_theta_m, **changes))


def build(cfg, *, seed: int = 4713):
    import cupy as cp
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics

    coord = make_vertical_coord(cfg.nz)

    def theta(z):
        z = np.asarray(z, dtype=np.float64)
        return 300.0 + 0.003 * z + 0.008 * np.maximum(z - 12000.0, 0.0)

    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    rng = np.random.Generator(np.random.PCG64(seed))
    noise = rng.normal(0.0, 0.3, (cfg.ny, cfg.nx))
    state = init_moist_balanced(
        cfg, coord, base,
        lambda z: 0.016 * np.exp(-np.asarray(z) / 2200.0))
    if state.theta_m:
        # The seed is a dry-theta perturbation of the lowest level; the
        # theta_m run takes it through theta_m's factor (1 + Rv/Rd qv).
        from woof.core import constants as c
        qv0 = state.qv[0]
        factor = cp.float32(1.0) + cp.float32(c.RVOVRD) * qv0
        state.thp[0] += cp.asarray(noise, dtype=cp.float32) * factor
    else:
        state.thp[0] += cp.asarray(noise, dtype=cp.float32)
    from datetime import datetime
    start = datetime(2025, 6, 1, 18)
    lat = np.full((cfg.ny, cfg.nx), 35.0, np.float32)
    lon = np.full((cfg.ny, cfg.nx), -97.0, np.float32)
    driver = initialize_physics(state, cfg, landmask=1.0, tsk=303.0,
                                radiation_start_time=start,
                                radiation_latitude=lat,
                                radiation_longitude=lon)
    return state, driver
