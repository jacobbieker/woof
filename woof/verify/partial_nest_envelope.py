"""Partial-depth nest (Phase D3) envelope cases and metrics.

Three idealized cases on WRF's hybrid coordinate, each run three ways from
the same analytic initial condition: the coarse parent alone, the parent
with a partial-depth two-way child (:mod:`woof.core.partial_nest_run`), and
an ALL-FINE run of the whole parent domain at the child's resolution and
step, full depth -- the reference, what the partial child approximates
below its interface.

* ``rest``: a resting, stably stratified atmosphere must stay at rest --
  no spurious w at the interface or in the top zone.
* ``bubble``: a 3 K warm bubble in a neutral layer rises through the
  interface.
* ``gravity_wave``: Skamarock & Klemp's (1994) nonhydrostatic gravity wave
  (N = 0.01 s-1, uniform 20 m/s flow) through the whole depth, the
  interface included; amplitude 0.5 K so its w stands above the coupled
  system's rest-state noise floor (the rest case measures that floor).

All three run WRF's sixth-order horizontal filter (diff_6th_opt = 2, the
default factor) on every grid, as a nested WRF configuration would.

Metrics (:func:`run_case`): the child against all-fine on the child's own
points (the grids coincide cell for cell), and -- the comparison that says
what the child buys -- the child's parent-cell means and the parent-alone
run against all-fine's parent-cell means, both on the parent cells inside
the child's interior; top-zone w variance and the interface's max |w|; the
dry-mass drift of child and parent; and the steady-state wall time of
parent+child against all-fine.  ``tools/partial_nest_envelope.py`` writes
the receipt; ``tests/test_partial_nest_envelope.py`` enforces it.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from woof.config import RunConfig
from woof.core.partial_nest import PartialNestBand

G = 9.81

#: The common geometry: parent 60 x 16 cells at 3 km, child 3:1 over parent
#: cells 16..45 x 5..12 (90 x 24 at 1 km), 30 uniform eta levels to 10 km,
#: etac 0.45; the interface is full level 21 (eta 0.3, about 5.9 km).
GEOMETRY = dict(px=60, py=16, nz=30, dx=3000.0, ztop=10000.0, dt=12.0,
                etac=0.45, ratio=3, ipos=16, jpos=5, cx=30, cy=8, diff_6th_opt=2)


@dataclasses.dataclass
class Case:
    name: str
    sounding: object          # theta(z) callable
    thp: object               # theta'(x, y, z) on absolute coordinates
    u0: float
    seconds: float


def _cases():
    stable = lambda z: 300.0 * np.exp(1.0e-4 * np.asarray(z) / G)   # N = 0.01 s-1
    neutral = lambda z: 300.0 + 0.0 * np.asarray(z)
    xc = (GEOMETRY["ipos"] - 1 + GEOMETRY["cx"] / 2.0) * GEOMETRY["dx"]   # footprint centre

    def bubble(x, y, z):
        r = np.sqrt(((x - xc) / 6000.0) ** 2 + ((z - 2000.0) / 1500.0) ** 2)
        return np.where(r < 1.0, 3.0 * np.cos(0.5 * np.pi * r) ** 2, 0.0)

    def wave(x, y, z):
        return 0.5 * np.sin(np.pi * z / GEOMETRY["ztop"]) / (1.0 + ((x - xc) / 5000.0) ** 2)

    return {
        "rest": Case("rest", stable, lambda x, y, z: 0.0 * z, 0.0, 1800.0),
        "bubble": Case("bubble", neutral, bubble, 0.0, 720.0),
        "gravity_wave": Case("gravity_wave", stable, wave, 20.0, 1200.0),
    }


CASES = _cases()


def interface_level(coord) -> int:
    """The lowest full level that is pure pressure with c1 = 0 (strictly
    above etac), at or above eta 0.3."""
    ok = (np.asarray(coord.c3f) == 0.0) & (np.asarray(coord.c1f) == 0.0)
    ok &= np.asarray(coord.znw) <= 0.3 + 1e-9
    return int(np.flatnonzero(ok)[0])


def _parent_cfg(**kw) -> RunConfig:
    g = GEOMETRY
    values = dict(nx=g["px"], ny=g["py"], nz=g["nz"], dx=g["dx"], dy=g["dx"],
                  ztop=g["ztop"], dt=g["dt"], run_seconds=0.0, hybrid_opt=2,
                  etac=g["etac"], top_lid=True, diff_6th_opt=g["diff_6th_opt"])
    values.update(kw)
    return RunConfig(**values)


def _init(cfg, coord, base, case: Case, x_origin: float, y_origin: float,
          *, cells: int = 1):
    """``woof.core.state.init_theta_perturbation`` with the case's theta'
    on ABSOLUTE coordinates (the domain's own origin added back).

    ``cells`` > 1 (the parent) takes each cell's theta' as the mean of the
    fine grid's ``cells x cells`` sub-cell values -- the parent starts as
    the restriction of the fine field (WRF's initial feedback), so the
    hydrostatically rebalanced interface heights of parent and child agree
    to the SINT/cell-mean difference, not to a coarse-sampling error."""
    from woof.core.state import init_theta_perturbation
    sub = (np.arange(cells) + 0.5) / cells - 0.5

    def thp(xc, z):
        x = xc + 0.5 * cfg.nx * cfg.dx + x_origin
        yy = y_origin + (np.arange(cfg.ny) + 0.5) * cfg.dy
        zz = z if np.ndim(z) == 3 else z[:, None, None]
        acc = np.zeros((cfg.nz, cfg.ny, cfg.nx))
        for sy in sub:
            for sx in sub:
                acc += np.broadcast_to(
                    case.thp((x + sx * cfg.dx)[None, None, :],
                             (yy + sy * cfg.dy)[None, :, None], zz),
                    acc.shape)
        return acc / (cells * cells)

    s = init_theta_perturbation(cfg, coord, base, thp)
    s.u[...] = case.u0
    return s


def build(case_name: str, *, band_overrides=None):
    """``(nest, parent_only, allfine)`` for one case: the coupled pair, and
    factories for the two references."""
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.partial_nest_dyn import truncate_state
    from woof.core.partial_nest_run import PartialNest, child_config
    case = CASES[case_name]
    g = GEOMETRY
    pcfg = _parent_cfg()
    coord = make_vertical_coord(g["nz"], hybrid_opt=2, etac=g["etac"])
    base = make_base_state(coord, case.sounding, p_surf=pcfg.p_surf, ztop=pcfg.ztop)
    k = interface_level(coord)
    band = PartialNestBand(k_top=k, parent_grid_ratio=g["ratio"],
                           parent_time_step_ratio=g["ratio"],
                           i_parent_start=g["ipos"], j_parent_start=g["jpos"],
                           nx=g["cx"] * g["ratio"], ny=g["cy"] * g["ratio"])
    if band_overrides:
        band = dataclasses.replace(band, **band_overrides)
    parent = _init(pcfg, coord, base, case, 0.0, 0.0, cells=g["ratio"])
    cfull_cfg = child_config(pcfg, band)
    x0 = (band.i_parent_start - 1) * pcfg.dx
    y0 = (band.j_parent_start - 1) * pcfg.dy
    cfull = _init(cfull_cfg, coord, base, case, x0, y0)
    child, ccfg = truncate_state(cfull, cfull_cfg, k, coord=coord)
    nest = PartialNest(parent, pcfg, coord, child, ccfg, band)

    def parent_only():
        return _init(pcfg, coord, base, case, 0.0, 0.0, cells=g["ratio"]), pcfg

    def allfine():
        fcfg = dataclasses.replace(pcfg, nx=pcfg.nx * g["ratio"], ny=pcfg.ny * g["ratio"],
                                   dx=pcfg.dx / g["ratio"], dy=pcfg.dy / g["ratio"],
                                   dt=float(np.float32(pcfg.dt) / np.float32(g["ratio"])))
        return _init(fcfg, coord, base, case, 0.0, 0.0), fcfg

    return nest, parent_only, allfine


def _host(a):
    import cupy as cp
    return cp.asnumpy(a).astype(np.float64)


def dry_mass(state) -> float:
    """Total dry column mass (Pa summed over columns), float64 host."""
    return float((_host(state.mub2d) + _host(state.mup)).sum())


def _sync():
    import cupy as cp
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:          # the Metal facade synchronizes on read
        pass


def _timed(fn) -> float:
    _sync()
    t0 = time.perf_counter()
    fn()
    _sync()
    return time.perf_counter() - t0


def run_case(case_name: str, *, seconds: float | None = None,
             band_overrides=None, references: bool = True,
             disabled=(), configure=None, timing: bool = True) -> dict:
    """Run one case three ways and measure it (module docstring)."""
    import cupy as cp
    from woof.core import dycore
    from woof.core.partial_nest_run import run
    case = CASES[case_name]
    seconds = float(case.seconds if seconds is None else seconds)
    g = GEOMETRY
    r = g["ratio"]

    def fresh():
        nest, parent_only, allfine = build(case_name, band_overrides=band_overrides)
        nest.disabled = set(disabled)
        if configure is not None:
            configure(nest)
        return nest, parent_only, allfine

    nest, parent_only, allfine = fresh()
    k = nest.k
    band = nest.band
    m0_child, m0_parent = dry_mass(nest.child), dry_mass(nest.parent)
    top_rows = slice(k - band.relax_zone, k + 1)
    w_top_var, w_iface_max = [], []

    def on_child(n, clock):
        w = n.child.w
        fr = n.frame_width
        inner = w[top_rows, fr:-fr, fr:-fr]
        w_top_var.append(float(cp.mean(inner * inner)))
        w_iface_max.append(float(cp.max(cp.abs(w[k, fr:-fr, fr:-fr]))))

    report = run(nest, run_seconds=seconds, time_step=int(g["dt"]), on_child_step=on_child)
    c = nest.child
    out = {
        "case": case_name, "seconds": seconds, "k_top": k,
        "interface_pressure_pa": nest.p_i,
        "relax_zone": band.relax_zone, "feedback": band.feedback,
        "feedback_kinds": list(nest.feedback_kinds),
        "mass_correction": band.mass_correction, "mass_route": nest.mass_route,
        "diff_6th_opt": int(nest.pcfg.diff_6th_opt),
        "parent_steps": report.steps - nest.stats["child_steps"],
        "child_steps": nest.stats["child_steps"], "forces": report.forces,
        "feedbacks": report.feedback_calls,
        "top_zone_w_variance_max": max(w_top_var),
        "top_zone_w_variance_final": w_top_var[-1],
        "interface_w_max_abs": max(w_iface_max),
        "child_w_max_abs": float(cp.max(cp.abs(c.w))),
        "child_dry_mass_drift_rel": (dry_mass(c) - m0_child) / m0_child,
        "parent_dry_mass_drift_rel": (dry_mass(nest.parent) - m0_parent) / m0_parent,
        "max_abs_mass_residual": nest.stats["max_abs_adj"],
        "finite": bool(all(np.isfinite(_host(getattr(c, f))).all()
                           for f in ("u", "v", "w", "thp", "php", "mup"))),
    }
    if not references:
        return out
    fstate, fcfg = allfine()
    nsteps = int(round(seconds / fcfg.dt))
    for _ in range(nsteps):
        dycore.step(fstate, fcfg)
    pstate, pcfg = parent_only()
    for _ in range(int(round(seconds / pcfg.dt))):
        dycore.step(pstate, pcfg)
    j0, i0 = (band.j_parent_start - 1) * r, (band.i_parent_start - 1) * r
    fr = nest.frame_width
    sl = (slice(j0 + fr, j0 + band.ny - fr), slice(i0 + fr, i0 + band.nx - fr))
    csl = (slice(fr, band.ny - fr), slice(fr, band.nx - fr))
    # The parent cells inside the child's interior (two parent cells, i.e.
    # more than the lateral frame, in from each child edge).
    pj = slice(band.j_parent_start - 1 + 2, band.j_parent_start - 1 + band.ny // r - 2)
    pi = slice(band.i_parent_start - 1 + 2, band.i_parent_start - 1 + band.nx // r - 2)

    def stats(d, f):
        return {"rmse": float(np.sqrt(np.mean(d * d))), "bias": float(d.mean()),
                "ref_rms": float(np.sqrt(np.mean(f * f)))}

    def block(a):
        n = a.shape[0]
        return a.reshape(n, a.shape[1] // r, r, a.shape[2] // r, r).mean(axis=(2, 4))

    fine_vs, coarse_vs, parent_vs = {}, {}, {}
    for name, ff, cc, pp in (("w", fstate.w, c.w, pstate.w),
                             ("theta_p", fstate.thp, c.thp, pstate.thp),
                             ("u", fstate.u[:, :, :-1], c.u[:, :, :-1], pstate.u[:, :, :-1])):
        f = _host(ff)[:k]
        ch = _host(cc)[:k]
        fine_vs[name] = stats(ch[(slice(None),) + csl] - f[(slice(None),) + sl],
                              f[(slice(None),) + sl])
        fb = block(f)
        cb = block(ch)[:, 2:band.ny // r - 2, 2:band.nx // r - 2]
        coarse_vs[name] = stats(cb - fb[:, pj, pi], fb[:, pj, pi])
        parent_vs[name] = stats(_host(pp)[:k, pj, pi] - fb[:, pj, pi], fb[:, pj, pi])
    out["child_vs_allfine"] = fine_vs
    out["child_block_mean_vs_allfine_block_mean"] = coarse_vs
    out["parent_only_vs_allfine_block_mean"] = parent_vs
    out["cell_steps_ratio"] = (
        (fcfg.nx * fcfg.ny * fcfg.nz * nsteps)
        / (pcfg.nx * pcfg.ny * pcfg.nz * out["parent_steps"]
           + nest.ccfg.nx * nest.ccfg.ny * nest.ccfg.nz * out["child_steps"]))
    if timing:
        # Steady-state wall time: every kernel already compiled by the runs
        # above, no per-step host reads; a fresh nest and a fresh all-fine
        # state over the same simulated time.
        nest2, _, allfine2 = fresh()
        t_nest = _timed(lambda: run(nest2, run_seconds=seconds, time_step=int(g["dt"])))
        f2, fcfg2 = allfine2()

        def fine_run():
            for _ in range(nsteps):
                dycore.step(f2, fcfg2)
        t_fine = _timed(fine_run)
        out["wall_parent_plus_child_s"] = t_nest
        out["wall_allfine_s"] = t_fine
        out["speedup"] = t_fine / t_nest
        out["wall_split_s"] = {key: nest2.stats[key] for key in
                               ("parent_seconds", "child_seconds", "coupling_seconds")}
    return out
