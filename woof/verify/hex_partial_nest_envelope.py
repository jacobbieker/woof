"""Partial-depth band envelope on MPAS-A (Phase D5): cases and metrics.

The hex counterpart of :mod:`woof.verify.partial_nest_envelope` (WRF D3/D4).
A small planet (radius 30 km, no rotation) carries three runs of one analytic
initial condition:

* the PARENT alone: a quasi-uniform 2562-cell global mesh (~2.1 km),
  ``config_dt`` 12 s;
* the BAND: that parent plus a regional child culled from a 10242-cell
  global mesh (~1.05 km) over a 20 km cap, cut at the flat ``zgrid`` face
  ``interface_level`` (6 km of a 10 km lid), ``config_dt`` 6 s
  (``parent_time_step_ratio`` 2) -- :mod:`woof.hex.partial_nest_run`;
* ALL-FINE: the 10242-cell mesh, full depth, ``config_dt`` 6 s -- the
  reference, what the child approximates below its interface.

The child mesh is the all-fine mesh culled (``rw_mpas_mesh --cull-parent``),
so its cells ARE all-fine cells (``indexToCellID``).

Cases (no mean flow: on a small non-rotating sphere a uniform wind is not
balanced, so the gravity wave is Skamarock & Klemp's perturbation at rest):

* ``rest``: a resting stratified atmosphere (N = 0.01 s-1) must stay at rest.
* ``gravity_wave``: a 0.5 K, 5 km half-width sin(pi z/ztop) perturbation
  through the whole depth, interface included.
* ``bubble``: a 3 K warm bubble (6 km x 1.5 km at 2 km) in a neutral layer,
  rising through the interface.
* ``moist_bubble``: refused on hex (see :data:`MOIST_REFUSAL`).

Metrics, as D3/D4: the child against all-fine on its interior cells; the
child's parent-cell means (the conservative remap's own weights) and the
parent alone against all-fine's parent-cell means (the same weights applied
to all-fine), on the parent cells the child's interior covers wholly; top-zone
w variance, interface max |w|, dry-mass drift, the interface budget residual
and wall time against all-fine.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

G, RD, CP, P0 = 9.80616, 287.0, 1004.5, 1.0e5
CV = CP - RD
RADIUS = 30_000.0

#: The geometry: meshes, levels, steps.
GEOMETRY = dict(parent_cells=2562, fine_cells=10242, cap_radius_km=4250.0,
                nz=30, ztop=10_000.0, interface_level=18, parent_dt=12.0, ratio=2,
                relax_levels=4)

MOIST_REFUSAL = (
    "moist_bubble (Kessler + RRTMG on the extended column) is refused on hex: the "
    "hex column physics is constructed only from init_atmosphere files "
    "(cuda_suite_physics_v841.seal_constructor via the forecast drivers) and its "
    "radiation column is the dycore column with a scalar p_top "
    "(cuda_arwen_physics_v841 run_phase1); an extended column through "
    "prepared_physics_atmosphere is not a seam the hex suite exposes")


@dataclasses.dataclass
class Case:
    name: str
    sounding: object          # theta(z)
    thp: object               # theta'(x, y, z) on the planet (x, y, z) unit sphere * R
    seconds: float


def _cases():
    stable = lambda z: 300.0 * np.exp(1.0e-4 * np.asarray(z) / G)
    neutral = lambda z: 300.0 + 0.0 * np.asarray(z)

    def arc(xyz):
        # Great-circle distance (m) from the cap centre (lat 0, lon 0) = +x.
        u = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
        return RADIUS * np.arccos(np.clip(u[..., 0], -1.0, 1.0))

    def bubble(xyz, z):
        r = np.sqrt((arc(xyz) / 6000.0) ** 2 + ((z - 2000.0) / 1500.0) ** 2)
        return np.where(r < 1.0, 3.0 * np.cos(0.5 * np.pi * r) ** 2, 0.0)

    def wave(xyz, z):
        return 0.5 * np.sin(np.pi * z / GEOMETRY["ztop"]) / (1.0 + (arc(xyz) / 5000.0) ** 2)

    return {
        "rest": Case("rest", stable, lambda xyz, z: 0.0 * z, 1800.0),
        "gravity_wave": Case("gravity_wave", stable, wave, 1200.0),
        "bubble": Case("bubble", neutral, bubble, 720.0),
    }


CASES = _cases()


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------

def mesh_dir() -> Path:
    base = os.environ.get("WOOF_HEX_PN_MESH_DIR")
    if base:
        return Path(base)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "woof" / "hex-partial-nest"


def _rw_mpas_mesh() -> Path:
    root = Path(__file__).resolve().parents[2]
    exe = root / "tools/rustwx/target/release/rw_mpas_mesh"
    if not exe.is_file():
        raise FileNotFoundError(
            f"{exe}: build it (cd tools/rustwx && cargo build --release --locked --offline "
            "--bin rw_mpas_mesh)")
    return exe


def ensure_meshes() -> dict[str, Path]:
    """The three grid files, generated once (rw_mpas_mesh is deterministic)."""
    d = mesh_dir()
    d.mkdir(parents=True, exist_ok=True)
    g = GEOMETRY
    paths = {"parent": d / f"p{g['parent_cells']}.nc", "fine": d / f"f{g['fine_cells']}.nc",
             "child": d / "child.nc"}
    exe = str(_rw_mpas_mesh())
    km = lambda n: str(int(round(480.0 * np.sqrt(2562.0 / n))))
    for key, n in (("parent", g["parent_cells"]), ("fine", g["fine_cells"])):
        if not paths[key].is_file():
            subprocess.run([exe, "--out", str(paths[key]), "--background-km", km(n),
                            "--cells", str(n), "--clobber"], check=True, capture_output=True)
    if not paths["child"].is_file():
        region = d / "cap.json"
        region.write_text('{"kind": "cap", "center_deg": [0, 0], "radius_km": %s}'
                          % g["cap_radius_km"])
        subprocess.run([exe, "--cull-parent", str(paths["fine"]), "--region", str(region),
                        "--out", str(paths["child"]), "--clobber"], check=True,
                       capture_output=True)
    return paths


_LENGTHS = ("xCell", "yCell", "zCell", "xEdge", "yEdge", "zEdge", "xVertex", "yVertex",
            "zVertex", "dcEdge", "dvEdge", "nominalMinDc")
_AREAS = ("areaCell", "areaTriangle", "kiteAreasOnVertex")


def load_mesh(path: Path, *, culled_from: Path | None = None):
    """A unit-sphere grid file on the small planet: ``(mesh, positions64)``,
    the mesh's float arrays float32 (the dycore's) and the binary64 cell /
    vertex positions and areas the host coupling tables are built from.

    ``culled_from``: a regional mesh's global source.  Its ``deriv_two`` is
    the source's rows (``indexToEdgeID``), as init_atmosphere computes it on
    the global mesh before a regional cull (an outer-ring cell has no full
    stencil of its own)."""
    from woof.hex.mesh import Mesh
    from woof.hex.transport import initialize_deriv_two
    m = Mesh.from_netcdf(path, validate=False)
    a = m.arrays
    for n in _LENGTHS:
        a[n] = np.asarray(a[n], np.float64) * RADIUS
    for n in _AREAS:
        a[n] = np.asarray(a[n], np.float64) * RADIUS * RADIUS
    m.attrs["sphere_radius"] = RADIUS
    nc, ne, nv, me = (m.dimensions[k] for k in ("nCells", "nEdges", "nVertices", "maxEdges"))
    a["fEdge"] = np.zeros(ne)
    a["fVertex"] = np.zeros(nv)
    a["defc_a"] = np.zeros((nc, me))
    a["defc_b"] = np.zeros((nc, me))
    if culled_from is None:
        a["deriv_two"] = np.ascontiguousarray(initialize_deriv_two(m))
    else:
        source, _ = load_mesh(culled_from)
        rows = np.asarray(a["indexToEdgeID"], np.int64) - 1
        a["deriv_two"] = np.ascontiguousarray(np.asarray(source.arrays["deriv_two"])[rows])
    positions64 = {k: np.array(a[k], np.float64) for k in
                   ("xCell", "yCell", "zCell", "xVertex", "yVertex", "zVertex",
                    "xEdge", "yEdge", "zEdge", "areaCell")}
    for k, v in list(a.items()):
        v = np.asarray(v)
        if v.dtype == np.float64:
            a[k] = v.astype(np.float32) if v.ndim == 0 else np.ascontiguousarray(
                v.astype(np.float32))
    return m, positions64


# ---------------------------------------------------------------------------
# host state
# ---------------------------------------------------------------------------

def _cast(o):
    return replace(o, **{f.name: np.ascontiguousarray(np.asarray(getattr(o, f.name), np.float32))
                         for f in fields(o) if isinstance(getattr(o, f.name), np.ndarray)
                         and getattr(o, f.name).dtype.kind == "f"})


@dataclasses.dataclass
class HostCase:
    mesh: object
    positions64: dict
    vertical: object      # float32 runtime grid
    reference: object
    state: object
    saved: object
    terrain: object
    wind: object


def host_case(mesh, positions64, case: Case, *, regional: bool = False) -> HostCase:
    """The case on ``mesh``, full depth, flat terrain, hydrostatic in the
    discrete sense the dycore holds at rest (theta(z) at layer midpoints,
    Exner integrated with MPAS's own layer-to-layer hydrostatic relation)."""
    from woof.hex.driver import DryReferenceState, DrySavedDiagnostics, TerrainMetrics
    from woof.hex.dynamics_v841 import V841ReferenceWindProfiles
    from woof.hex.state import PrognosticState
    from woof.hex.vertical import build_vertical_grid, runtime_vertical_vectors
    g = GEOMETRY
    nz, ztop = g["nz"], g["ztop"]
    nc, ne, me = (mesh.dimensions[k] for k in ("nCells", "nEdges", "maxEdges"))
    vg = build_vertical_grid(mesh, np.zeros(nc), n_vert_levels=nz, ztop=ztop,
                             scheme="specified", specified_zw=np.linspace(0.0, ztop, nz + 1),
                             terrain_smoothing_passes=0, smooth_surfaces=False, xnutr=0.0,
                             regional_halo=regional)
    vg = replace(vg, **runtime_vertical_vectors(vg))
    zmid = 0.5 * (np.asarray(vg.zgrid[:-1], np.float64) + np.asarray(vg.zgrid[1:], np.float64))
    zz = np.asarray(vg.zz, np.float64)
    # Base: theta(z) hydrostatic Exner from the surface (exact for theta(z),
    # integrated on a fine column).
    zf = np.linspace(0.0, ztop, 20001)
    th = case.sounding(zf)
    pif = 1.0 - np.concatenate([[0.0], np.cumsum(0.5 * (G / (CP * th[1:]) + G / (CP * th[:-1]))
                                                  * np.diff(zf))])
    theta_b = case.sounding(zmid)
    pi_b = np.interp(zmid, zf, pif)
    p_b = P0 * pi_b ** (CP / RD)
    rho_phys_b = p_b / (RD * theta_b * pi_b)
    rho_base = rho_phys_b / zz
    rtheta_base = rho_base * theta_b
    xyz = np.stack([positions64[k] for k in ("xCell", "yCell", "zCell")], axis=1)
    thp = case.thp(xyz[None, :, :], zmid)
    theta = theta_b + thp
    # Hold pressure at the base value (the perturbation is in density).
    rho_phys = p_b / (RD * theta * pi_b)
    rho = rho_phys / zz
    rtheta = rho * theta
    exner = (zz * RD * rtheta / P0) ** (RD / CV)
    p = P0 * exner ** (CP / RD)
    ref = DryReferenceState(rho_base=rho_base, rho_theta_base=rtheta_base,
                            pressure_base=p_b, exner_base=pi_b)
    state = PrognosticState(rho=rho, rho_theta=rtheta, rho_u=np.zeros((nz, ne)),
                            rho_w=np.zeros((nz + 1, nc)), scalars=np.zeros((1, nz, nc)),
                            time_seconds=0.0)
    saved = DrySavedDiagnostics(theta_m=theta, exner=exner, density_perturbation=rho - rho_base,
                                rho_theta_perturbation=rtheta - rtheta_base,
                                pressure_perturbation=p - p_b,
                                normal_velocity=np.zeros((nz, ne)),
                                vertical_velocity=np.zeros((nz + 1, nc)))
    terrain = TerrainMetrics(np.zeros((nz + 1, nc, me), np.float32),
                             np.zeros((nz + 1, nc, me), np.float32))
    wind = V841ReferenceWindProfiles(np.zeros(nz, np.float32), np.zeros(nz, np.float32))
    return HostCase(mesh, positions64, _cast(vg), _cast(ref), _cast(state), _cast(saved),
                    terrain, wind)


def dycore_config(dt: float, **kw):
    from woof.hex.config_v841 import V841DryDycoreConfig
    values = dict(config_dt=float(dt), config_dynamics_split_steps=3,
                  config_number_of_sub_steps=2, config_split_dynamics_transport=True,
                  config_monotonic=False, config_positive_definite=False, config_xnutr=0.0)
    values.update(kw)
    return V841DryDycoreConfig(**values)


def global_driver(h: HostCase, dt: float, *, kernel_cache=None):
    from woof.hex.cuda_backend import KernelCache
    from woof.hex.cuda_driver import CudaDryDycoreDriver
    return CudaDryDycoreDriver.from_host(
        h.mesh, h.state, h.vertical, h.reference, dycore_config(dt),
        saved_diagnostics=h.saved, terrain_metrics=h.terrain,
        reference_wind_profiles=h.wind,
        kernel_cache=KernelCache() if kernel_cache is None else kernel_cache)


# ---------------------------------------------------------------------------
# the band
# ---------------------------------------------------------------------------

def _geometry(mesh, positions64):
    from woof.hex.partial_nest_coupling import CellGeometry
    return CellGeometry.from_mesh(mesh, positions64)


def build(case_name: str, *, band_overrides: dict | None = None):
    """``(nest, parent_only, allfine, aux)`` for a case: the coupled pair and
    factories for the two references (fresh drivers)."""
    from datetime import datetime
    from woof.hex.partial_nest import HexPartialNestBand, truncate
    from woof.hex.partial_nest_run import ChildBuild, PartialNestHex
    case = CASES[case_name]
    g = GEOMETRY
    paths = ensure_meshes()
    pmesh, ppos = load_mesh(paths["parent"])
    cmesh, cpos = load_mesh(paths["child"], culled_from=paths["fine"])
    hp = host_case(pmesh, ppos, case)
    hc = host_case(cmesh, cpos, case, regional=True)
    k = int(g["interface_level"])
    nlev = g["nz"]
    band_values = dict(interface_level=k, parent_time_step_ratio=g["ratio"],
                       relax_levels=g["relax_levels"])
    band_values.update(band_overrides or {})
    band = HexPartialNestBand(**band_values)
    parent = global_driver(hp, g["parent_dt"])
    child_dt = g["parent_dt"] / g["ratio"]
    edge_xyz = np.stack([cpos[n] for n in ("xEdge", "yEdge", "zEdge")], axis=1)
    cbuild = ChildBuild(
        mesh=cmesh, geometry=_geometry(cmesh, cpos), edge_xyz=edge_xyz,
        state=truncate(hc.state, nlev, k), vertical=truncate(hc.vertical, nlev, k),
        reference=truncate(hc.reference, nlev, k), saved=truncate(hc.saved, nlev, k),
        terrain=truncate(hc.terrain, nlev, k),
        wind=truncate(hc.wind, nlev, k), config=dycore_config(child_dt, config_apply_lbcs=True),
        start_time=datetime(2000, 6, 21, 18, 0, 0))
    nest = PartialNestHex(parent, pmesh, _geometry(pmesh, ppos), cbuild, band)

    def parent_only():
        return global_driver(host_case(pmesh, ppos, case), g["parent_dt"])

    def allfine():
        fmesh, fpos = load_mesh(paths["fine"])
        return global_driver(host_case(fmesh, fpos, case), child_dt), fmesh, fpos

    aux = dict(child_mesh=cmesh, child_pos=cpos, parent_mesh=pmesh, parent_pos=ppos,
               fine_path=paths["fine"])
    return nest, parent_only, allfine, aux


def coincident(points: np.ndarray, source: np.ndarray, what: str) -> np.ndarray:
    """The source index of each point, which must coincide exactly (the
    culled mesh renumbers its cells and edges; ``indexToCellID`` is its own
    numbering, not the source's)."""
    from scipy.spatial import cKDTree
    distance, index = cKDTree(source).query(points)
    if float(np.max(distance)) != 0.0:
        raise ValueError(f"{what}: the child's points are not the all-fine mesh's")
    return np.asarray(index, np.int64)


def _fine_maps(aux: dict, fpos: dict):
    xyz = lambda pos, kind: np.stack([pos[n + kind] for n in "xyz"], axis=1)
    return (coincident(xyz(aux["child_pos"], "Cell"), xyz(fpos, "Cell"), "cells"),
            coincident(xyz(aux["child_pos"], "Edge"), xyz(fpos, "Edge"), "edges"))




# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _host(a) -> np.ndarray:
    import cupy as cp
    return cp.asnumpy(a).astype(np.float64)


def _sync() -> None:
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


def dry_mass(atmosphere, area: np.ndarray, cells: np.ndarray) -> float:
    """sum_k sum_cells rho_zz * zz * dzw * area over ``cells``, binary64."""
    rho = _host(atmosphere.state.rho)[:, cells]
    zz = _host(atmosphere.vertical.zz)[:, cells]
    zg = _host(atmosphere.vertical.zgrid)[:, cells]
    return float(np.sum(rho * zz * np.diff(zg, axis=0) * area[cells][None, :]))


def theta_perturbation(atmosphere) -> np.ndarray:
    ref = atmosphere.reference
    return _host(atmosphere.saved.theta_m) - _host(ref.rho_theta_base) / _host(ref.rho_base)


def _stats(d: np.ndarray, f: np.ndarray) -> dict:
    return {"rmse": float(np.sqrt(np.mean(d * d))), "bias": float(d.mean()),
            "ref_rms": float(np.sqrt(np.mean(f * f)))}


def _block_means(remap, values: np.ndarray) -> np.ndarray:
    """(nlev, nrows) overlap-area means of (nlev, ncells-of-the-child)
    ``values``: the remap's own overlaps, without the density weight."""
    out = np.empty((values.shape[0], len(remap.targets)))
    for r in range(len(remap.targets)):
        sl = slice(remap.row_ptr[r], remap.row_ptr[r + 1])
        w = remap.overlap[sl]
        out[:, r] = values[:, remap.cols[sl]] @ w / w.sum()
    return out


def run_case(case_name: str, *, seconds: float | None = None, band_overrides=None,
             references: bool = True, disabled=(), timing: bool = True) -> dict:
    """Run one case three ways and measure it (module docstring)."""
    import cupy as cp
    from woof.hex.partial_nest_run import advance, run
    case = CASES[case_name]
    g = GEOMETRY
    seconds = float(case.seconds if seconds is None else seconds)
    parent_steps = int(round(seconds / g["parent_dt"]))

    def fresh():
        nest, parent_only, allfine, aux = build(case_name, band_overrides=band_overrides)
        nest.top.disabled = set(disabled)
        return nest, parent_only, allfine, aux

    nest, parent_only, allfine, aux = fresh()
    k = nest.k
    band = nest.band
    cmesh = aux["child_mesh"]
    interior = np.flatnonzero(np.asarray(cmesh.arrays["bdyMaskCell"]) == 0)
    edges_in = np.flatnonzero(np.asarray(cmesh.arrays["bdyMaskEdge"]) == 0)
    carea = np.asarray(aux["child_pos"]["areaCell"], np.float64)
    parea = np.asarray(aux["parent_pos"]["areaCell"], np.float64)
    pall = np.arange(len(parea))
    m0_child = dry_mass(nest.child_driver.atmosphere, carea, interior)
    m0_parent = dry_mass(nest.parent_driver.atmosphere, parea, pall)
    top_rows = slice(k - band.relax_levels, k + 1)
    w_top_var, w_iface_max = [], []
    d_interior = cp.asarray(interior)

    def on_child(n, j):
        w = n.child_driver.atmosphere.saved.vertical_velocity[:, d_interior]
        inner = w[top_rows]
        w_top_var.append(cp.mean(inner * inner))
        w_iface_max.append(cp.max(cp.abs(w[k])))

    run(nest, parent_steps, on_child_step=on_child)
    c = nest.child_driver.atmosphere
    p = nest.parent_driver.atmosphere
    out = {
        "case": case_name, "seconds": seconds, "interface_level": k,
        "interface_height_m": nest.z_i, "relax_levels": band.relax_levels,
        "feedback": band.feedback, "feedback_fields": list(band.feedback_fields),
        "face_scheme": band.face_scheme, "mass_correction": band.mass_correction,
        "parent_steps": parent_steps, "child_steps": nest.stats["child_steps"],
        "forces": nest.stats["forces"], "feedbacks": nest.stats["feedbacks"],
        "remap_rows": int(len(nest.remap.targets)),
        "top_zone_w_variance_max": float(max(float(v) for v in w_top_var)),
        "top_zone_w_variance_final": float(w_top_var[-1]),
        "interface_w_max_abs": float(max(float(v) for v in w_iface_max)),
        "child_w_max_abs": float(cp.max(cp.abs(c.saved.vertical_velocity[:, d_interior]))),
        "child_dry_mass_drift_rel": (dry_mass(c, carea, interior) - m0_child) / m0_child,
        "parent_dry_mass_drift_rel": (dry_mass(p, parea, pall) - m0_parent) / m0_parent,
        "max_abs_mass_residual": nest.max_abs_mass_residual,
        "finite": bool(all(np.isfinite(_host(x)).all() for x in (
            c.saved.vertical_velocity, c.saved.theta_m, c.saved.normal_velocity, c.state.rho,
            p.saved.theta_m, p.state.rho))),
    }
    if not references:
        return out
    child_dt = g["parent_dt"] / g["ratio"]
    fdriver, fmesh, fpos = allfine()
    nsteps = int(round(seconds / child_dt))
    for _ in range(nsteps):
        advance(fdriver)
    pdriver = parent_only()
    for _ in range(parent_steps):
        advance(pdriver)
    f = fdriver.atmosphere
    cell_map, edge_map = _fine_maps(aux, fpos)
    fields_child = {"w": _host(c.saved.vertical_velocity)[:k],
                    "theta_p": theta_perturbation(c)[:k],
                    "u": _host(c.saved.normal_velocity)[:k]}
    fields_fine = {"w": _host(f.saved.vertical_velocity)[:k],
                   "theta_p": theta_perturbation(f)[:k],
                   "u": _host(f.saved.normal_velocity)[:k]}
    fields_parent = {"w": _host(pdriver.atmosphere.saved.vertical_velocity)[:k],
                     "theta_p": theta_perturbation(pdriver.atmosphere)[:k]}
    fine_vs, block_vs, parent_vs = {}, {}, {}
    for name in ("w", "theta_p", "u"):
        if name == "u":
            ch, fi = fields_child[name][:, edges_in], fields_fine[name][:, edge_map[edges_in]]
        else:
            ch, fi = fields_child[name][:, interior], fields_fine[name][:, cell_map[interior]]
        fine_vs[name] = _stats(ch - fi, fi)
        if name in ("w", "theta_p"):
            out[f"child_vs_allfine_{name}_rmse_by_level"] = [
                float(v) for v in np.sqrt(np.mean((ch - fi) ** 2, axis=1))]
        if name == "u":
            continue
        # Parent-cell means over the cells the child's interior covers
        # wholly: the child's, all-fine's (the same overlaps on the same
        # cells) and the parent alone at those cells.
        fine_on_child = np.zeros_like(fields_child[name])
        fine_on_child[:, :len(cell_map)] = fields_fine[name][:, cell_map]
        cb = _block_means(nest.remap, fields_child[name])
        fb = _block_means(nest.remap, fine_on_child)
        block_vs[name] = _stats(cb - fb, fb)
        pb = fields_parent[name][:, nest.remap.targets]
        parent_vs[name] = _stats(pb - fb, fb)
        # The coupled parent itself (after feedback) at those cells.
        cpb = (_host(p.saved.vertical_velocity)[:k] if name == "w"
               else theta_perturbation(p)[:k])[:, nest.remap.targets]
        block_vs[name + "_coupled_parent"] = _stats(cpb - fb, fb)
    out["child_vs_allfine"] = fine_vs
    out["child_block_mean_vs_allfine_block_mean"] = block_vs
    out["parent_only_vs_allfine_block_mean"] = parent_vs
    nlev = g["nz"]
    cells_nest = (len(parea) * nlev * parent_steps
                  + int(nest.nc_solve) * k * nest.stats["child_steps"])
    out["cell_steps_ratio"] = len(fpos["areaCell"]) * nlev * nsteps / cells_nest
    if timing:
        nest2, _, allfine2, _ = fresh()
        t_nest = _timed(lambda: run(nest2, parent_steps))
        f2, _, _ = allfine2()

        def fine_run():
            for _ in range(nsteps):
                advance(f2)
        t_fine = _timed(fine_run)
        out["wall_parent_plus_child_s"] = t_nest
        out["wall_allfine_s"] = t_fine
        out["speedup"] = t_fine / t_nest
        out["wall_split_s"] = {key: nest2.stats[key] for key in
                               ("parent_seconds", "child_seconds", "coupling_seconds")}
    return out
