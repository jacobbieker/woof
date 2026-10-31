"""Partial-depth nest envelope cases and metrics (Phases D3-D4).

Idealized cases on WRF's hybrid coordinate, each run three ways from the
same analytic initial condition: the coarse root alone, the root with its
partial-depth two-way band(s) (:mod:`woof.core.partial_nest_run`), and an
ALL-FINE run of the whole root domain at the finest band's resolution and
step, full depth -- the reference, what the partial child approximates
below its interface.

One band (``GEOMETRY``: a 3 km root, a 1 km band to eta 0.3, ~5.9 km):

* ``rest``: a resting, stably stratified atmosphere must stay at rest --
  no spurious w at the interface or in the top zone.
* ``bubble``: a 3 K warm bubble in a neutral layer rises through the
  interface.
* ``gravity_wave``: Skamarock & Klemp's (1994) nonhydrostatic gravity wave
  (N = 0.01 s-1, uniform 20 m/s flow) through the whole depth, the
  interface included; amplitude 0.5 K.
* ``moist_bubble`` (D4): a 3 K bubble in the Weisman-Klemp (1982) moist
  sounding with Kessler microphysics and RRTMG radiation (every minute,
  afternoon sun) on every grid -- the band's radiation on its extended
  column (:mod:`woof.core.partial_nest_physics`); its cloud and vapour
  cross the interface.

Two bands (``CHAIN_GEOMETRY``, D4: 9 km aloft, 3 km to eta 0.3, 1 km to eta
0.6, ~3 km): ``chain_rest`` (two hours: the coupled chain's feedback
stability) and ``chain_gravity_wave`` (a 0.5 K wave, 15 km half-width).

All cases run WRF's sixth-order horizontal filter (diff_6th_opt = 2).

Metrics (:func:`run_case`): the finest band against all-fine on its own
points (the grids coincide cell for cell), and -- the comparison that says
what the band buys -- the finest band's parent-cell (and, for a chain, root-cell) means and the
root-alone run against all-fine's cell means, on the parent cells inside
the band's interior; top-zone w variance and the interface's max |w|; the
dry-mass drift of child and parent; and the steady-state wall time against
all-fine.  ``tools/partial_nest_envelope.py`` writes the receipt;
``tests/test_partial_nest_envelope.py`` enforces it.
"""

from __future__ import annotations

import dataclasses
import time
from datetime import datetime

import numpy as np

from woof.config import RunConfig
from woof.core.partial_nest import PartialNestBand

G = 9.81

#: One band: parent 60 x 16 cells at 3 km, child 3:1 over parent cells
#: 16..45 x 5..12 (90 x 24 at 1 km), 30 uniform eta levels to 10 km, etac
#: 0.45; the interface is full level 21 (eta 0.3, about 5.9 km).
GEOMETRY = dict(px=60, py=16, nz=30, dx=3000.0, ztop=10000.0, dt=12.0,
                etac=0.45, ratio=3, ipos=16, jpos=5, cx=30, cy=8, diff_6th_opt=2)

#: Two bands: root 40 x 12 at 9 km (dt 36 s); band 1 3:1 over root cells
#: 11..30 x 3..10 (60 x 24 at 3 km) to eta 0.3; band 2 3:1 over band-1
#: cells 16..45 x 8..16 (90 x 27 at 1 km) to eta 0.6.  etac 0.7 makes both
#: interfaces pure pressure.
CHAIN_GEOMETRY = dict(px=40, py=12, nz=30, dx=9000.0, ztop=10000.0, dt=36.0,
                      etac=0.7, diff_6th_opt=2,
                      bands=(dict(ratio=3, ipos=11, jpos=3, cx=20, cy=8, eta=0.3),
                             dict(ratio=3, ipos=16, jpos=8, cx=30, cy=9, eta=0.6)))

#: The idealized radiation clock and place (afternoon sun at 40 N).
START_TIME = datetime(2000, 6, 21, 18, 0, 0)
LATITUDE, LONGITUDE = 40.0, -100.0


@dataclasses.dataclass
class Case:
    name: str
    sounding: object          # theta(z) callable
    thp: object               # theta'(x, y, z) on absolute coordinates
    u0: float
    seconds: float
    qv: object = None         # qv(z) callable: a moist case
    physics: dict = dataclasses.field(default_factory=dict)
    chain: bool = False


def _wk82(index):
    def f(z):
        from woof.verify.cases.wk82 import wk82_sounding
        return wk82_sounding(np.asarray(z, dtype=np.float64))[index]
    return f


def _cases():
    stable = lambda z: 300.0 * np.exp(1.0e-4 * np.asarray(z) / G)   # N = 0.01 s-1
    neutral = lambda z: 300.0 + 0.0 * np.asarray(z)
    xc = (GEOMETRY["ipos"] - 1 + GEOMETRY["cx"] / 2.0) * GEOMETRY["dx"]   # footprint centre
    b1, b2 = CHAIN_GEOMETRY["bands"]
    xc2 = ((b1["ipos"] - 1) * CHAIN_GEOMETRY["dx"]
           + ((b2["ipos"] - 1) + b2["cx"] / 2.0) * CHAIN_GEOMETRY["dx"] / b1["ratio"])

    def bubble(x, y, z):
        r = np.sqrt(((x - xc) / 6000.0) ** 2 + ((z - 2000.0) / 1500.0) ** 2)
        return np.where(r < 1.0, 3.0 * np.cos(0.5 * np.pi * r) ** 2, 0.0)

    def wave(x, y, z):
        return 0.5 * np.sin(np.pi * z / GEOMETRY["ztop"]) / (1.0 + ((x - xc) / 5000.0) ** 2)

    def chain_wave(x, y, z):
        return (0.5 * np.sin(np.pi * z / CHAIN_GEOMETRY["ztop"])
                / (1.0 + ((x - xc2) / 15000.0) ** 2))

    moist_physics = dict(moist=True, mp_physics=1, moist_adv_opt=1,
                         ra_lw_physics=4, ra_sw_physics=4, radt=1.0)
    return {
        "rest": Case("rest", stable, lambda x, y, z: 0.0 * z, 0.0, 1800.0),
        "bubble": Case("bubble", neutral, bubble, 0.0, 720.0),
        "gravity_wave": Case("gravity_wave", stable, wave, 20.0, 1200.0),
        "moist_bubble": Case("moist_bubble", _wk82(0), bubble, 0.0, 900.0,
                             qv=_wk82(1), physics=moist_physics),
        "chain_rest": Case("chain_rest", stable, lambda x, y, z: 0.0 * z, 0.0,
                           7200.0, chain=True),
        "chain_gravity_wave": Case("chain_gravity_wave", stable, chain_wave, 20.0,
                                   2160.0, chain=True),
    }


CASES = _cases()


def interface_level(coord, eta: float = 0.3) -> int:
    """The lowest full level that is pure pressure with c1 = 0 (strictly
    above etac), at or above ``eta``."""
    ok = (np.asarray(coord.c3f) == 0.0) & (np.asarray(coord.c1f) == 0.0)
    ok &= np.asarray(coord.znw) <= eta + 1e-9
    return int(np.flatnonzero(ok)[0])


def _root_cfg(g, case: Case, **kw) -> RunConfig:
    values = dict(nx=g["px"], ny=g["py"], nz=g["nz"], dx=g["dx"], dy=g["dx"],
                  ztop=g["ztop"], dt=g["dt"], run_seconds=0.0, hybrid_opt=2,
                  etac=g["etac"], top_lid=True, diff_6th_opt=g["diff_6th_opt"])
    values.update(case.physics)
    values.update(kw)
    return RunConfig(**values)


def _parent_cfg(**kw) -> RunConfig:
    """D3's root config for the one-band geometry (dry)."""
    return _root_cfg(GEOMETRY, Case("dry", None, None, 0.0, 0.0), **kw)


def _init(cfg, coord, base, case: Case, x_origin: float, y_origin: float,
          *, cells: int = 1):
    """The case's state on ``cfg``'s grid, the theta' on ABSOLUTE
    coordinates (the domain's own origin added back):
    ``woof.core.state.init_theta_perturbation``, or for a moist case
    ``woof.core.moist.init_moist_balanced`` with the case's qv(z).

    ``cells`` > 1 (a coarser grid) takes each cell's theta' as the mean of
    the finest grid's ``cells x cells`` sub-cell values -- the coarse grid
    starts as the restriction of the fine field (WRF's initial feedback)."""
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

    if case.qv is not None:
        from woof.core.moist import init_moist_balanced
        s = init_moist_balanced(cfg, coord, base, case.qv, thp)
    else:
        from woof.core.state import init_theta_perturbation
        s = init_theta_perturbation(cfg, coord, base, thp)
    s.u[...] = case.u0
    return s


def _latlon(cfg):
    shape = (cfg.ny, cfg.nx)
    return np.full(shape, LATITUDE), np.full(shape, LONGITUDE)


def _attach_physics(state, cfg):
    """The idealized neutral-surface physics driver (``initialize_physics``
    defaults, ``nest_ideal_common.prepare_idealized_domain``'s)."""
    from woof.core.diagnostics import update_diagnostics
    from woof.core.physics import initialize_physics, physics_driver_required
    update_diagnostics(state, cfg.hypsometric_opt)
    if not physics_driver_required(cfg):
        return None
    lat, lon = _latlon(cfg)
    return initialize_physics(state, cfg, radiation_start_time=START_TIME,
                              radiation_latitude=lat, radiation_longitude=lon)


def _attach_child_physics(nest):
    from woof.core.partial_nest_physics import attach_child_physics
    from woof.core.physics import physics_driver_required
    if not physics_driver_required(nest.ccfg):
        return None
    lat, lon = _latlon(nest.ccfg)
    return attach_child_physics(nest, start_time=START_TIME, latitude=lat,
                                longitude=lon)


def _band_specs(case: Case, coord):
    """``[(PartialNestBand, origin_x, origin_y, cells)]`` coarse to fine."""
    if not case.chain:
        g = GEOMETRY
        return [(PartialNestBand(k_top=interface_level(coord), parent_grid_ratio=g["ratio"],
                                 parent_time_step_ratio=g["ratio"],
                                 i_parent_start=g["ipos"], j_parent_start=g["jpos"],
                                 nx=g["cx"] * g["ratio"], ny=g["cy"] * g["ratio"]))]
    out = []
    for b in CHAIN_GEOMETRY["bands"]:
        out.append(PartialNestBand(
            k_top=interface_level(coord, b["eta"]), parent_grid_ratio=b["ratio"],
            parent_time_step_ratio=b["ratio"], i_parent_start=b["ipos"],
            j_parent_start=b["jpos"], nx=b["cx"] * b["ratio"], ny=b["cy"] * b["ratio"]))
    return out


def build_chain(case_name: str, *, band_overrides=None):
    """``(nests, root_only, allfine)``: the coupled chain (coarse to fine),
    and factories for the two references.  ``band_overrides`` (a dict, or
    one dict per band) replaces band fields."""
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.partial_nest import truncated_vertical_coord, validate_band_chain
    from woof.core.partial_nest_dyn import truncate_state
    from woof.core.partial_nest_run import PartialNest, child_config
    case = CASES[case_name]
    g = CHAIN_GEOMETRY if case.chain else GEOMETRY
    rcfg = _root_cfg(g, case)
    coord = make_vertical_coord(g["nz"], hybrid_opt=2, etac=g["etac"])
    base = make_base_state(coord, case.sounding, p_surf=rcfg.p_surf, ztop=rcfg.ztop)
    bands = _band_specs(case, coord)
    if band_overrides:
        per = (band_overrides if isinstance(band_overrides, (list, tuple))
               else [band_overrides] * len(bands))
        bands = [dataclasses.replace(b, **o) for b, o in zip(bands, per)]
    validate_band_chain(bands, rcfg, coord)
    ratios = [b.parent_grid_ratio for b in bands]
    total = int(np.prod(ratios))
    root = _init(rcfg, coord, base, case, 0.0, 0.0, cells=total)
    _attach_physics(root, rcfg)
    nests = []
    parent, pcfg, pcoord = root, rcfg, coord
    x0 = y0 = 0.0
    cells = total
    for band in bands:
        cells //= band.parent_grid_ratio
        cfull_cfg = child_config(pcfg, band, nz=rcfg.nz)
        x0 += (band.i_parent_start - 1) * pcfg.dx
        y0 += (band.j_parent_start - 1) * pcfg.dy
        cfull = _init(cfull_cfg, coord, base, case, x0, y0, cells=cells)
        child, ccfg = truncate_state(cfull, cfull_cfg, band.k_top, coord=coord)
        del cfull
        nest = PartialNest(parent, pcfg, pcoord, child, ccfg, band,
                           parent_nest=nests[-1] if nests else None)
        _attach_child_physics(nest)
        nests.append(nest)
        parent, pcfg = child, ccfg
        pcoord = truncated_vertical_coord(coord, band.k_top)

    def root_only():
        s = _init(rcfg, coord, base, case, 0.0, 0.0, cells=total)
        _attach_physics(s, rcfg)
        return s, rcfg

    def allfine():
        fcfg = dataclasses.replace(rcfg, nx=rcfg.nx * total, ny=rcfg.ny * total,
                                   dx=rcfg.dx / total, dy=rcfg.dy / total,
                                   dt=float(nests[-1].ccfg.dt))
        s = _init(fcfg, coord, base, case, 0.0, 0.0)
        _attach_physics(s, fcfg)
        return s, fcfg

    return nests, root_only, allfine


def build(case_name: str, *, band_overrides=None):
    """``(nest, parent_only, allfine)`` for a one-band case (D3's entry)."""
    nests, root_only, allfine = build_chain(case_name, band_overrides=band_overrides)
    if len(nests) != 1:
        raise ValueError(f"{case_name} is a chain case: use build_chain")
    return nests[0], root_only, allfine


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


def _footprint(nests):
    """The finest band's placement on the root grid in finest-grid cells:
    ``(j0, i0)`` and the total ratio."""
    j0 = i0 = 0
    for n in nests:
        r = n.band.parent_grid_ratio
        j0 = (j0 + n.band.j_parent_start - 1) * r
        i0 = (i0 + n.band.i_parent_start - 1) * r
    return j0, i0


def run_case(case_name: str, *, seconds: float | None = None,
             band_overrides=None, references: bool = True,
             disabled=(), configure=None, timing: bool = True) -> dict:
    """Run one case three ways and measure it (module docstring)."""
    import cupy as cp
    from woof.core import dycore
    from woof.core.partial_nest_run import run
    case = CASES[case_name]
    seconds = float(case.seconds if seconds is None else seconds)
    g = CHAIN_GEOMETRY if case.chain else GEOMETRY

    def fresh():
        nests, root_only, allfine = build_chain(case_name, band_overrides=band_overrides)
        for n in nests:
            n.disabled = set(disabled)
            if configure is not None:
                configure(n)
        return nests, root_only, allfine

    nests, root_only, allfine = fresh()
    fine = nests[-1]
    k = fine.k
    band = fine.band
    m0_child = dry_mass(fine.child)
    m0_parent = dry_mass(nests[0].parent)
    top_rows = slice(k - band.relax_zone, k + 1)
    w_top_var, w_iface_max = [], []

    def on_child(n, clock):
        if n is not fine:
            return
        w = n.child.w
        fr = n.frame_width
        inner = w[top_rows, fr:-fr, fr:-fr]
        w_top_var.append(cp.mean(inner * inner))
        w_iface_max.append(cp.max(cp.abs(w[k, fr:-fr, fr:-fr])))

    report = run(nests, run_seconds=seconds, time_step=int(g["dt"]),
                 on_child_step=on_child)
    w_top_var = [float(v) for v in w_top_var]
    w_iface_max = [float(v) for v in w_iface_max]
    c = fine.child
    species = tuple(fine.species)
    child_steps = sum(n.stats["child_steps"] for n in nests)
    out = {
        "case": case_name, "seconds": seconds, "bands": len(nests),
        "k_top": [n.k for n in nests] if case.chain else k,
        "interface_pressure_pa": ([n.p_i for n in nests] if case.chain else fine.p_i),
        "relax_zone": band.relax_zone, "feedback": band.feedback,
        "feedback_kinds": list(fine.feedback_kinds), "species": list(species),
        "face_scheme": fine.face_scheme,
        "mass_correction": band.mass_correction, "mass_route": fine.mass_route,
        "diff_6th_opt": int(fine.pcfg.diff_6th_opt),
        "physics": {key: value for key, value in case.physics.items()},
        "parent_steps": report.steps - child_steps,
        "child_steps": fine.stats["child_steps"], "forces": report.forces,
        "feedbacks": report.feedback_calls,
        "top_zone_w_variance_max": max(w_top_var),
        "top_zone_w_variance_final": w_top_var[-1],
        "interface_w_max_abs": max(w_iface_max),
        "child_w_max_abs": float(cp.max(cp.abs(c.w))),
        "child_dry_mass_drift_rel": (dry_mass(c) - m0_child) / m0_child,
        "parent_dry_mass_drift_rel": (dry_mass(nests[0].parent) - m0_parent) / m0_parent,
        "max_abs_mass_residual": max(n.max_abs_mass_residual for n in nests),
        "finite": bool(all(np.isfinite(_host(getattr(n.child, f))).all()
                           for n in nests
                           for f in ("u", "v", "w", "thp", "php", "mup") + species)),
    }
    if not references:
        return out
    fstate, fcfg = allfine()
    nsteps = int(round(seconds / fcfg.dt))
    for _ in range(nsteps):
        dycore.step(fstate, fcfg)
    pstate, pcfg = root_only()
    for _ in range(int(round(seconds / pcfg.dt))):
        dycore.step(pstate, pcfg)
    r = fine.band.parent_grid_ratio            # the finest band's own ratio
    total = int(round(pcfg.dx / fcfg.dx))      # root cell = total x total fine
    j0, i0 = _footprint(nests)
    fr = fine.frame_width
    sl = (slice(j0 + fr, j0 + band.ny - fr), slice(i0 + fr, i0 + band.nx - fr))
    csl = (slice(fr, band.ny - fr), slice(fr, band.nx - fr))
    # The finest band's parent cells inside its interior (two parent cells,
    # more than the lateral frame, in from each edge), in that parent's cells.
    pj = slice(2, band.ny // r - 2)
    pi = slice(2, band.nx // r - 2)
    # The root cells wholly inside the finest band's interior.
    rj = slice(-(-(j0 + r * 2) // total), (j0 + band.ny - r * 2) // total)
    ri = slice(-(-(i0 + r * 2) // total), (i0 + band.nx - r * 2) // total)

    def stats(d, f):
        return {"rmse": float(np.sqrt(np.mean(d * d))), "bias": float(d.mean()),
                "ref_rms": float(np.sqrt(np.mean(f * f)))}

    def block(a, m):
        n = a.shape[0]
        return a.reshape(n, a.shape[1] // m, m, a.shape[2] // m, m).mean(axis=(2, 4))

    fine_vs, coarse_vs, parent_vs = {}, {}, {}
    fields = [("w", "w"), ("theta_p", "thp"), ("u", "u")] + [(q, q) for q in species]
    for label, name in fields:
        def get(s):
            a = getattr(s, name)
            return a[:, :, :-1] if name == "u" else a
        f = _host(get(fstate))[:k]
        ch = _host(get(c))[:k]
        fine_vs[label] = stats(ch[(slice(None),) + csl] - f[(slice(None),) + sl],
                               f[(slice(None),) + sl])
        # The finest band's block means against all-fine's, on that band's
        # parent cells.
        fwin = f[:, j0:j0 + band.ny, i0:i0 + band.nx]
        fb = block(fwin, r)[:, pj, pi]
        cb = block(ch, r)[:, pj, pi]
        coarse_vs[label] = stats(cb - fb, fb)
        # The root alone against all-fine's root-cell means, on the root
        # cells inside the finest band's interior.
        fr_b = block(f, total)[:, rj, ri]
        parent_vs[label] = stats(_host(get(pstate))[:k, rj, ri] - fr_b, fr_b)
        if case.chain:
            cr = block(ch[:, (rj.start * total - j0):(rj.stop * total - j0),
                          (ri.start * total - i0):(ri.stop * total - i0)], total)
            coarse_vs[label + "_root_cells"] = stats(cr - fr_b, fr_b)
    out["child_vs_allfine"] = fine_vs
    out["child_block_mean_vs_allfine_block_mean"] = coarse_vs
    out["parent_only_vs_allfine_block_mean"] = parent_vs
    cells_nest = (pcfg.nx * pcfg.ny * pcfg.nz * int(round(seconds / pcfg.dt))
                  + sum(n.ccfg.nx * n.ccfg.ny * n.ccfg.nz * n.stats["child_steps"]
                        for n in nests))
    out["cell_steps_ratio"] = fcfg.nx * fcfg.ny * fcfg.nz * nsteps / cells_nest
    if timing:
        # Steady-state wall time: every kernel already compiled by the runs
        # above, no per-step host reads; a fresh chain and a fresh all-fine
        # state over the same simulated time.
        nests2, _, allfine2 = fresh()
        t_nest = _timed(lambda: run(nests2, run_seconds=seconds, time_step=int(g["dt"])))
        f2, fcfg2 = allfine2()

        def fine_run():
            for _ in range(nsteps):
                dycore.step(f2, fcfg2)
        t_fine = _timed(fine_run)
        out["wall_parent_plus_child_s"] = t_nest
        out["wall_allfine_s"] = t_fine
        out["speedup"] = t_fine / t_nest
        out["wall_split_s"] = {key: sum(n.stats[key] for n in nests2) for key in
                               ("parent_seconds", "child_seconds", "coupling_seconds")}
    return out
