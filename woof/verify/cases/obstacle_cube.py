"""Surface-mounted cube benchmark for the opt-in resolved obstacles.

One cube of side H = 20 m sits in a periodic 12H x 6H x 5H box, resolved with
8 cells per H (dx = dz = 2.5 m). The flow is neutral and dry, under the 3-D
Smagorinsky LES closure. The same constant-flow-rate drive as the canyon case
(:mod:`woof.verify.cases.obstacle_canyon`) holds 5 m/s above 2H. The
time-mean flow along the cube's centreline, one cell above the floor, gives:

* ``reattach``: the distance from the leeward face to where the floor flow
  turns forward again, over H. Martinuzzi and Tropea (1993, channel,
  Re_H = 4e4) measured about 1.6; Shah and Ferziger (1997, LES) found
  about 1.7. In a periodic array the wake of the cube upstream shortens it;
  Coceal et al. (2006) report about 1 at lambda_p = 0.25, and this
  box is far sparser (lambda_p = 1/72);
* ``separate``: the distance upstream of the windward face where the floor
  flow first reverses (the horseshoe vortex), over H; Martinuzzi and Tropea
  give about 1.0;
* ``leak``: the rms wind inside the cube over the 2H reference wind;
* ``nan``.
"""
from __future__ import annotations

import numpy as np

from woof.config import RunConfig
from woof.verify.cases.obstacle_canyon import floor_drag

H = 20.0
DX = 2.5
U_REF = 5.0
X0 = 3 * H          # windward face
GATES = {
    "reattach": (0.8, 2.6),
    "separate": (0.2, 1.6),
    "leak": (None, 0.05),
}


def default_config(*, minutes: float = 15.0) -> RunConfig:
    return RunConfig(
        nx=int(12 * H / DX), ny=int(6 * H / DX), nz=40, dx=DX, dy=DX,
        ztop=5 * H, dt=0.1, run_seconds=minutes * 60.0, km_opt=3,
        mix_isotropic=1, c_s=0.18, bl_pbl_physics=0, sf_sfclay_physics=0,
        moist=False, mp_physics=0, isfflx=0, tke_heat_flux=0.0,
        tke_drag_coefficient=float(floor_drag(DX)), time_step_sound=28,
        obstacle_opt=1,
        case="obstacle_cube")


def morphology(cfg: RunConfig):
    from woof.static.buildings import CartesianGrid, aggregate, block_raster
    y0 = 0.5 * cfg.ny * cfg.dy - 0.5 * H
    raster = block_raster(cfg.nx, cfg.ny, cfg.dx, cfg.dy,
                          [(X0, X0 + H, y0, y0 + H, H)])
    return aggregate([raster], CartesianGrid(cfg.nx, cfg.ny, cfg.dx, cfg.dy))


def _crossings(x, u):
    """Reattachment and separation points from a floor-level u(x)."""
    behind = x > X0 + H
    reattach = np.nan
    xs, us = x[behind], u[behind]
    for n in range(1, xs.size):
        if us[n - 1] < 0.0 <= us[n]:
            frac = -us[n - 1] / (us[n] - us[n - 1])
            reattach = xs[n - 1] + frac * (xs[n] - xs[n - 1]) - (X0 + H)
            break
    ahead = x < X0
    separate = np.nan
    xa, ua = x[ahead][::-1], u[ahead][::-1]
    reversed_run = ua < 0.0
    if reversed_run.any():
        n = int(np.argmin(reversed_run)) if not reversed_run.all() else \
            reversed_run.size
        if n > 0:
            separate = X0 - xa[n - 1]
    return reattach / H, separate / H


def run(outdir=None, *, minutes: float = 15.0, average_minutes: float = 6.0
        ) -> dict:
    import cupy as cp

    from woof.core import obstacles
    from woof.core.dycore import run_steps, stability_report
    from woof.verify.cases.obstacle_canyon import (_drive, build,
                                                    sample_steps)

    cfg = default_config(minutes=minutes)
    state = build(cfg, morphology(cfg))
    z_w = obstacles.layer_heights(state)[:, 0, 0]
    z_mass = 0.5 * (z_w[1:] + z_w[:-1])
    upper = cp.asarray(np.flatnonzero(z_mass > 2 * H))
    n_total = int(round(cfg.run_seconds / cfg.dt))
    n_avg = int(round(average_minutes * 60.0 / cfg.dt))
    jc = cfg.ny // 2
    u_sum = np.zeros((cfg.nz, cfg.nx + 1))
    inside = np.zeros(2)
    sampled = set(sample_steps(n_total, n_avg))
    i0, i1 = int(X0 / DX), int((X0 + H) / DX)
    j0, j1 = jc - int(H / DX / 2), jc + int(H / DX / 2)
    k1 = int(np.searchsorted(z_mass, H))
    for step in range(n_total):
        run_steps(state, cfg, n=1)
        _drive(state, upper, cfg)
        if step in sampled:
            u = state.u
            u_sum += cp.asnumpy(0.5 * (u[:, jc - 1] + u[:, jc])).astype(
                np.float64)
            core = u[:k1, j0 + 1:j1 - 1, i0 + 2:i1 - 1]
            inside += (float(cp.mean(core * core)), 1.0)
    u_mean = u_sum / len(sampled)
    ref = float(u_mean[(z_mass > 1.9 * H) & (z_mass < 2.1 * H)].mean())
    x_u = np.arange(cfg.nx + 1) * DX
    reattach, separate = _crossings(x_u, u_mean[0])
    return dict(reattach=float(reattach), separate=float(separate),
                leak=float(np.sqrt(inside[0] / inside[1])) / ref,
                u_ref=ref, floor_u=u_mean[0].tolist(),
                nan=bool(stability_report(state)["nan"]),
                obstacles=getattr(state, "_obstacles_receipt", None))
