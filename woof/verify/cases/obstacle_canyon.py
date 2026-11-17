"""Street-canyon benchmark for the opt-in resolved obstacles (obstacle_opt=1).

A periodic row of 2-D street canyons, street width W = 20 m and building
width 20 m, resolved with 10 cells per W (dx = dz = 2 m).  The aspect ratio
H/W is the case's parameter (``aspect``, default 1: H = 20 m).  The flow
is neutral and dry, under the 3-D Smagorinsky LES closure (km_opt = 3). A
uniform body force, adjusted every step, holds the mean wind at 5 m/s above
2H (a constant-flow-rate drive, as in channel LES). The domain is quasi-2-D: four rows in y, periodic.

At H/W = 1 the canyon is in the skimming-flow regime. One primary vortex
fills it, with its centre near mid-width and slightly above mid-height, and
with reverse flow along the floor. Wind-tunnel and LES references for that
are Brown et al. (2000), Liu and Barth (2002), and Li, Liu and Leung (2008).
At H/W = 2 the primary vortex sits in the upper canyon and drives a weaker,
counter-rotating one below it, so the floor flow turns forward again (Li,
Liu and Leung 2008, LES at H/W = 2, 3 and 5; Sini, Anquetin and Mestayer
1996).
``run`` returns:

* ``vortex_x``, ``vortex_z``: the vortex centre from the in-canyon
  streamfunction, as fractions of W and H;
* ``floor_u``: the mean along-street wind one cell above the floor at the
  canyon centre, over the 2H reference wind (negative is reverse flow);
* ``roof_u``: the same at roof height;
* ``leak``: the rms wind inside the buildings, over the reference wind;
* ``vortices``: the number of stacked recirculation cells, from the sign
  changes of the mean along-street wind up the canyon's centre column
  (below the roof);
* ``nan``: whether anything went non-finite.

The gates are deliberately wide bands around those published values.
"""
from __future__ import annotations

import numpy as np

from woof.config import RunConfig
from woof.core.grid import make_base_state, make_vertical_coord

H = 20.0
W = 20.0
B = 20.0
DX = 2.0
U_REF = 5.0
#: Constant-flow-rate drive: relaxation time (s) of the mean wind above 2H.
DRIVE_TAU_S = 10.0

GATES = {
    "vortex_x": (0.25, 0.75),
    "vortex_z": (0.35, 0.85),
    "floor_u": (None, 0.0),
    "roof_u": (0.0, None),
    "leak": (None, 0.05),
}
#: H/W = 2: two stacked vortices, the floor wind forward again.
GATES_ASPECT_2 = {
    "vortices": (1.5, 2.5),
    "floor_u": (0.0, None),
    "roof_u": (0.0, None),
    "leak": (None, 0.05),
}


def floor_drag(dz: float, z0: float = 0.01) -> float:
    """The ground's drag coefficient, the same log law the obstacles' walls
    use (woof.core.obstacles), for the first level dz/2 above it.  With no
    surface layer (sf_sfclay_physics = 0, isfflx = 0) the LES closure takes
    the ground's momentum flux from tke_drag_coefficient."""
    return (0.4 / np.log(0.5 * dz / z0)) ** 2


def default_config(*, minutes: float = 10.0, aspect: float = 1.0
                   ) -> RunConfig:
    nx = int(round((W + B) / DX))
    height = aspect * W
    return RunConfig(
        nx=nx, ny=4, nz=int(round(5 * height / DX)), dx=DX, dy=DX,
        ztop=5 * height, dt=0.1,
        run_seconds=minutes * 60.0, km_opt=3, mix_isotropic=1, c_s=0.18,
        bl_pbl_physics=0, sf_sfclay_physics=0, moist=False, mp_physics=0,
        isfflx=0, tke_heat_flux=0.0,
        tke_drag_coefficient=float(floor_drag(DX)),
        time_step_sound=40, obstacle_opt=1, case="obstacle_canyon")


def _height(cfg: RunConfig) -> float:
    return cfg.ztop / 5.0


def morphology(cfg: RunConfig):
    """Buildings fill x in [W, W + B) of each period, at ztop / 5."""
    from woof.static.buildings import CartesianGrid, aggregate, block_raster
    raster = block_raster(cfg.nx, cfg.ny, cfg.dx, cfg.dy,
                          [(W, W + B, 0.0, cfg.ny * cfg.dy, _height(cfg))])
    return aggregate([raster], CartesianGrid(cfg.nx, cfg.ny, cfg.dx, cfg.dy))


def build(cfg: RunConfig, morph=None):
    from woof.core import obstacles
    from woof.core.state import init_at_rest

    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(
        np.asarray(z, float), 300.0), p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    state.u[...] = U_REF
    if cfg.obstacle_opt:
        obstacles.attach(state, cfg, morphology(cfg) if morph is None
                         else morph)
    return state


def _drive(state, upper, cfg) -> None:
    """Uniform body-force increment holding the mean wind above 2H."""
    import cupy as cp
    mean = cp.mean(state.u[upper])
    gain = 1.0 - np.exp(-cfg.dt / DRIVE_TAU_S)
    state.u += (U_REF - mean) * cp.float32(gain)


def _metrics(u, w, z_mass, z_w, cfg, nan) -> dict:
    nx = cfg.nx
    H = _height(cfg)
    x = (np.arange(nx) + 0.5) * cfg.dx
    street = x < W
    zc = z_mass
    in_canyon = zc < H
    ref = float(u[(zc > 1.9 * H) & (zc < 2.1 * H)][:, :nx].mean())
    # Streamfunction psi(x, z) = int_0^z u dz on the mass columns.
    um = 0.5 * (u[:, :nx] + u[:, 1:nx + 1])
    dz = np.diff(z_w)
    psi = np.cumsum(um * dz[:, None], axis=0)
    sub = np.where(in_canyon[:, None] & street[None], psi, np.inf)
    kz, ix = np.unravel_index(np.argmin(sub), sub.shape)
    centre_col = int(np.argmin(np.abs(x - 0.5 * W)))
    floor_u = float(um[0, centre_col]) / ref
    roof = int(np.argmin(np.abs(zc - H)))
    roof_u = float(um[roof, centre_col]) / ref
    solid = (~street)[None] & in_canyon[:, None]
    leak = float(np.sqrt(np.mean(um[solid] ** 2))) / ref
    column = um[in_canyon, centre_col]
    vortices = int(np.count_nonzero(np.diff(np.sign(column)) != 0))
    return dict(vortex_x=float(x[ix] / W), vortex_z=float(zc[kz] / H),
                floor_u=floor_u, roof_u=roof_u, leak=leak, u_ref=ref,
                vortices=vortices, centre_u=(column / ref).tolist(),
                nan=bool(nan))


def run(outdir=None, *, minutes: float = 10.0, average_minutes: float = 4.0,
        aspect: float = 1.0) -> dict:
    import cupy as cp

    from woof.core.dycore import run_steps, stability_report
    from woof.core.obstacles import layer_heights

    cfg = default_config(minutes=minutes, aspect=aspect)
    state = build(cfg)
    z_w = layer_heights(state)[:, 0, 0]
    z_mass = 0.5 * (z_w[1:] + z_w[:-1])
    upper = cp.asarray(np.flatnonzero(z_mass > 2 * _height(cfg)))
    n_total = int(round(cfg.run_seconds / cfg.dt))
    n_avg = int(round(average_minutes * 60.0 / cfg.dt))
    u_sum = np.zeros((cfg.nz, cfg.nx + 1))
    w_sum = np.zeros((cfg.nz + 1, cfg.nx))
    samples = 0
    for step in range(n_total):
        run_steps(state, cfg, n=1)
        _drive(state, upper, cfg)
        if step >= n_total - n_avg and step % 10 == 0:
            u_sum += cp.asnumpy(state.u).astype(np.float64).mean(axis=1)
            w_sum += cp.asnumpy(state.w).astype(np.float64).mean(axis=1)
            samples += 1
    metrics = _metrics(u_sum / samples, w_sum / samples, z_mass, z_w, cfg,
                       stability_report(state)["nan"])
    metrics["obstacles"] = getattr(state, "_obstacles_receipt", None)
    return metrics
