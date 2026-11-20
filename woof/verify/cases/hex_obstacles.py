"""Hex benchmarks for the opt-in obstacles (``woof.hex.obstacles``).

The phase-1 cube and street canyon (:mod:`woof.verify.cases.obstacle_cube`,
:mod:`woof.verify.cases.obstacle_canyon`) on the hex dycore.  The hex route
has no planar periodic mesh (its mesh checks and LES weights are spherical
only), so the closed domain is a small sphere: an icosahedral Voronoi mesh of
radius ``R = 12 H / (2 pi)``, so the equator is the phase-1 box length 12H
and pole to pole is 6H.  By default (``balanced=True``) the frame rotates at
``-U_REF / (2 R)``, so Coriolis terms are nonzero and balance the solid-body
wind; ``balanced=False`` is the zero-Coriolis case.  The flow is a solid-body zonal
wind, held at ``U_REF`` above 2H on the equator by a body-force drive (the
phase-1 constant-flow-rate drive).  The dycore is the v8.4.1 dry resident
driver with 2-D Smagorinsky mixing; the ground takes the obstacles' own wall
law (``ground_wall=True``), as phase 1 took ``tke_drag_coefficient``.

``zoom > 1`` is the variable-resolution variant: a conformal (Moebius) zoom
puts spacing ``1/zoom`` of the uniform one at the obstacle and ``zoom`` times
it on the far side.

The mesh comes from the hex test suite's own generator
(``components/hex/tests/test_mesh_validation.graded_sphere_mesh``), so these
cases run from a source checkout.
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

H = 20.0
U_REF = 5.0
#: Constant-flow-rate drive relaxation time (s), as phase 1.
DRIVE_TAU_S = 10.0
RADIUS = 12.0 * H / (2.0 * math.pi)
#: Phase-1 gates (obstacle_cube.GATES); wide bands.
CUBE_GATES = {"reattach": (0.8, 2.6), "separate": (0.2, 1.6), "leak": (None, 0.05)}


def _tests_dir() -> Path:
    here = Path(__file__).resolve().parents[3] / "components" / "hex" / "tests"
    if not (here / "test_mesh_validation.py").is_file():
        raise RuntimeError("the hex obstacle cases build their sphere with the hex test suite's "
                           "graded_sphere_mesh; run them from a woof source checkout")
    return here


def _rotation_to(target: np.ndarray) -> np.ndarray:
    """A rotation taking unit vector ``target`` to (1, 0, 0)."""
    t = target / np.linalg.norm(target)
    x = np.array([1.0, 0.0, 0.0])
    v = np.cross(t, x)
    s = np.linalg.norm(v)
    c = float(np.dot(t, x))
    if s < 1e-15:
        return np.eye(3) if c > 0 else np.diag([-1.0, -1.0, 1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def sphere_mesh(level: int = 4, zoom: float = 1.0, radius: float = RADIUS):
    """The case mesh: refined point (if any) at lat 0, lon 0; angleEdge,
    fEdge = 0, deriv_two and the deformation weights filled in."""
    tests = str(_tests_dir())
    if tests not in sys.path:
        sys.path.insert(0, tests)
    from test_mesh_validation import graded_sphere_mesh
    from woof.hex.transport import initialize_deriv_two

    mesh = graded_sphere_mesh(level=level, zoom=float(zoom), radius=float(radius))
    a = mesh.arrays
    fine = int(np.argmin(np.asarray(a["areaCell"], np.float64)))
    xyz = lambda p: np.stack([np.asarray(a["x" + p], np.float64), np.asarray(a["y" + p], np.float64),
                              np.asarray(a["z" + p], np.float64)], axis=1)
    rot = _rotation_to(xyz("Cell")[fine]) if zoom > 1.0 else np.eye(3)
    dtype = np.asarray(a["xCell"]).dtype
    for p in ("Cell", "Edge", "Vertex"):
        q = xyz(p) @ rot.T
        a["x" + p], a["y" + p], a["z" + p] = (q[:, i].astype(dtype) for i in range(3))
        u = q / np.linalg.norm(q, axis=1)[:, None]
        a["lat" + p] = np.arcsin(np.clip(u[:, 2], -1, 1)).astype(dtype)
        a["lon" + p] = np.arctan2(u[:, 1], u[:, 0]).astype(dtype)
    nc, ne, nv = (mesh.dimensions[k] for k in ("nCells", "nEdges", "nVertices"))
    mx = mesh.dimensions["maxEdges"]
    # The test sphere's verticesOnEdge order is arbitrary (238 of 480 edges
    # at level 2 run against k x n).  The vorticity takes the circulation
    # sign from that order, so orient every edge as MPAS does: the tangent
    # k x n runs from verticesOnEdge[0] to verticesOnEdge[1].
    voe = np.asarray(a["verticesOnEdge"])
    coe = np.asarray(a["cellsOnEdge"], np.int64)
    tangent = np.cross(xyz("Edge"), xyz("Cell")[coe[:, 1]] - xyz("Cell")[coe[:, 0]])
    vert = xyz("Vertex")
    flip = np.einsum("ij,ij->i", vert[voe[:, 1]] - vert[voe[:, 0]], tangent) < 0
    a["verticesOnEdge"] = np.where(flip[:, None], voe[:, ::-1], voe)
    normal, east, north = edge_frames(mesh)
    a["angleEdge"] = np.arctan2(np.sum(normal * north, 1), np.sum(normal * east, 1)).astype(dtype)
    a["fEdge"] = np.zeros(ne, dtype)
    a["fVertex"] = np.zeros(nv, dtype)
    a["meshDensity"] = (np.asarray(a["areaCell"], np.float64).min()
                        / np.asarray(a["areaCell"], np.float64)).astype(dtype)
    a["nominalMinDc"] = np.asarray(np.asarray(a["dcEdge"]).min(), dtype)
    a["defc_a"] = np.zeros((nc, mx), dtype)
    a["defc_b"] = np.zeros((nc, mx), dtype)
    weights, edges, count = trisk_weights(mesh)
    a["weightsOnEdge"] = weights.astype(dtype)
    a["edgesOnEdge"] = edges.astype(np.asarray(a["edgesOnEdge"]).dtype)
    a["nEdgesOnEdge"] = count.astype(np.asarray(a["nEdgesOnEdge"]).dtype)
    a["deriv_two"] = np.asarray(initialize_deriv_two(mesh), dtype)
    return mesh


def trisk_weights(mesh):
    """``(weightsOnEdge, edgesOnEdge, nEdgesOnEdge)`` by Thuburn et al.
    (2009) as MPAS-Tools forms them; the test sphere ships them as zeros,
    which removes the vorticity flux, the Coriolis term and the tangential
    wind from the dynamics.  Checked: ``v = sum w u`` gives a smooth wind's
    tangential component to 5 % at 642 cells, and
    ``w(e,e') dc_e/dv_e' = -w(e',e) dc_e'/dv_e`` to rounding."""
    a = mesh.arrays
    coe = np.asarray(a["cellsOnEdge"], np.int64)
    eoc = np.asarray(a["edgesOnCell"], np.int64)
    voc = np.asarray(a["verticesOnCell"], np.int64)
    neoc = np.asarray(a["nEdgesOnCell"], np.int64)
    cov = np.asarray(a["cellsOnVertex"], np.int64)
    kite = np.asarray(a["kiteAreasOnVertex"], np.float64)
    area = np.asarray(a["areaCell"], np.float64)
    dv = np.asarray(a["dvEdge"], np.float64)
    dc = np.asarray(a["dcEdge"], np.float64)
    ne, width = coe.shape[0], 2 * eoc.shape[1]
    weights = np.zeros((ne, width))
    edges = np.zeros((ne, width), np.int64)
    count = np.zeros(ne, np.int64)
    for e in range(ne):
        n = 0
        for side, sign in ((0, 1.0), (1, -1.0)):
            c = coe[e, side]
            k = neoc[c]
            j = int(np.flatnonzero(eoc[c, :k] == e)[0])
            r = 0.0
            for i in range(1, k):
                jj = (j + i) % k
                other = eoc[c, jj]
                vertex = voc[c, jj]
                r += kite[vertex, int(np.flatnonzero(cov[vertex] == c)[0])] / area[c]
                out = 1.0 if coe[other, 0] == c else -1.0
                weights[e, n] = sign * out * (0.5 - r) * dv[other] / dc[e]
                edges[e, n] = other
                n += 1
        count[e] = n
    return weights, edges, count


def edge_frames(mesh):
    """Unit edge normal (cellsOnEdge[0] -> [1]), local east and north."""
    a = mesh.arrays
    coe = np.asarray(a["cellsOnEdge"], np.int64)
    c = np.stack([np.asarray(a[k], np.float64) for k in ("xCell", "yCell", "zCell")], 1)
    m = np.stack([np.asarray(a[k], np.float64) for k in ("xEdge", "yEdge", "zEdge")], 1)
    m /= np.linalg.norm(m, axis=1)[:, None]
    n = c[coe[:, 1]] - c[coe[:, 0]]
    n -= np.sum(n * m, 1)[:, None] * m
    n /= np.linalg.norm(n, axis=1)[:, None]
    lat = np.arcsin(np.clip(m[:, 2], -1, 1))
    lon = np.arctan2(m[:, 1], m[:, 0])
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], 1)
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], 1)
    return n, east, north


def _case_state(mesh, nz: int, ztop: float):
    """Flat, hydrostatic, isothermal and at rest (make_synthetic_x1_case on
    uniform levels)."""
    import woof.hex.driver as hex_driver
    from woof.hex.vertical import build_vertical_grid

    def uniform(m, terrain, **kw):
        # MPAS init's w Rayleigh layer (dss) over the top 40 %: the lid
        # otherwise reflects the obstacle's waves back down.
        kw.update(scheme="specified", specified_zw=np.linspace(0.0, ztop, nz + 1),
                  xnutr=0.2, damping_start=0.6 * ztop)
        return build_vertical_grid(m, terrain, **kw)

    original = hex_driver.build_vertical_grid
    hex_driver.build_vertical_grid = uniform
    try:
        return hex_driver.make_synthetic_x1_case(
            mesh, n_vert_levels=nz, ztop=ztop, perturbation_amplitude=0.0,
            wind_speed=0.0, n_scalars=1)
    finally:
        hex_driver.build_vertical_grid = original


def _f32(record):
    from dataclasses import fields
    out = {}
    for f in fields(record):
        value = getattr(record, f.name)
        if isinstance(value, np.ndarray) and value.dtype.kind == "f":
            out[f.name] = np.ascontiguousarray(value, np.float32)
    return replace(record, **out)


def mesh_radius(mesh) -> float:
    return float(mesh.attrs["sphere_radius"])


def build(level=4, zoom=1.0, *, nz=40, ztop=5 * H, dt=0.1, sub_steps=8,
          smagorinsky=True, radius=RADIUS, balanced=True, kernel_cache=None):
    """``(mesh, driver, frames)``: the resident driver with the solid-body
    wind ``U_REF cos(lat)`` everywhere.

    ``balanced`` sets the frame rotation to ``-U_REF / (2 R)``: the
    solid-body wind is then in gradient balance with a uniform pressure
    (``f u + u^2 tan(lat)/R = 0``), so air the ground or a building slows
    feels ``u (u - U_REF) sin(lat)/R`` rather than the whole cyclostrophic
    pressure gradient ``U_REF^2 sin(lat) cos(lat)/R`` (which otherwise drives
    a full-depth meridional circulation off the ground).  Both shrink
    as 1/R."""
    from woof.hex.config_v841 import V841DryDycoreConfig
    from woof.hex.cuda_driver import CudaDryDycoreDriver
    from woof.hex.driver import TerrainMetrics
    from woof.hex.dynamics_v841 import V841ReferenceWindProfiles
    from woof.hex.vertical import runtime_vertical_vectors

    mesh = sphere_mesh(level, zoom, radius=radius)
    if balanced:
        omega = -U_REF / (2.0 * radius)
        for name, lat in (("fEdge", "latEdge"), ("fVertex", "latVertex")):
            mesh.arrays[name] = (2.0 * omega * np.sin(np.asarray(mesh.arrays[lat], np.float64))
                                 ).astype(np.asarray(mesh.arrays[name]).dtype)
    nc, mx = mesh.dimensions["nCells"], mesh.dimensions["maxEdges"]
    case = _case_state(mesh, nz, ztop)
    normal, east, _ = edge_frames(mesh)
    lat_e = np.asarray(mesh.arrays["latEdge"], np.float64)
    pattern = np.sum(normal * east, 1) * np.cos(lat_e)
    coe = np.asarray(mesh.arrays["cellsOnEdge"], np.int64)
    rho = np.asarray(case.state.rho, np.float64)
    rho_edge = 0.5 * (rho[:, coe[:, 0]] + rho[:, coe[:, 1]])
    # Dry: the synthetic case's passive tracer (about 0.2) would be read as
    # water vapour by the moist dynamics coefficients.
    state = replace(case.state, rho_u=rho_edge * U_REF * pattern[None],
                    scalars=np.zeros_like(case.state.scalars))
    vertical = replace(case.vertical_grid, **runtime_vertical_vectors(case.vertical_grid))
    options = dict(config_dt=dt, config_number_of_sub_steps=sub_steps,
                   config_dynamics_split_steps=3, config_split_dynamics_transport=True,
                   config_monotonic=False, config_positive_definite=False,
                   config_apvm_upwinding=0.5, config_xnutr=0.2, config_zd=0.6 * ztop)
    if smagorinsky:
        options.update(config_horiz_mixing="2d_smagorinsky",
                       config_len_disp=float(np.sqrt(np.asarray(mesh.arrays["areaCell"]).min())))
    config = V841DryDycoreConfig(**options)
    config.validate()
    terrain = TerrainMetrics(np.zeros((nz + 1, nc, mx), np.float32),
                             np.zeros((nz + 1, nc, mx), np.float32))
    wind = V841ReferenceWindProfiles(np.zeros(nz, np.float32), np.zeros(nz, np.float32))
    kwargs = {} if kernel_cache is None else {"kernel_cache": kernel_cache}
    driver = CudaDryDycoreDriver.from_host(
        mesh, _f32(state), _f32(vertical), _f32(case.reference), config,
        terrain_metrics=terrain, reference_wind_profiles=wind, **kwargs)
    z_w = np.linspace(0.0, ztop, nz + 1)
    return mesh, driver, dict(pattern=pattern, coe=coe, z_w=z_w, normal=normal, east=east)


class Drive:
    """Holds the solid-body amplitude above 2H at ``U_REF`` (phase 1's
    constant-flow-rate drive): ``u += (U_REF - A) gain cos(lat) n.east``.

    With f = 0 the solid-body wind keeps its curvature acceleration; the
    pressure takes it up within the first acoustic adjustment (cyclostrophic
    balance, about rho U^2/2 = 15 Pa between equator and pole) and the flow
    stays steady to 0.01 m/s without obstacles.  Slowed air near the
    obstacle feels that pressure gradient unbalanced, about
    U^2 sin(lat) cos(lat)/R, or ``u (u - U) sin(lat)/R`` with ``build``'s
    balanced rotation.
    """

    def __init__(self, driver, mesh, frames, *, dt, height=H):
        cp = driver.cp
        self.cp = cp
        z_mass = 0.5 * (frames["z_w"][1:] + frames["z_w"][:-1])
        upper = np.flatnonzero(z_mass > 2 * height)
        if upper.size == 0:
            raise ValueError(f"no level above twice the building height {height:g} m to hold the drive")
        self.upper = cp.asarray(upper)
        a = mesh.arrays
        w = (np.asarray(a["dcEdge"], np.float64) * np.asarray(a["dvEdge"], np.float64)
             * (np.abs(np.asarray(a["latEdge"], np.float64)) < 0.5))
        p = frames["pattern"]
        self.p = cp.asarray(p.astype(np.float32))
        self.weight = cp.asarray((w * p / max(np.sum(w * p * p), 1e-30)).astype(np.float32))
        self.c0 = cp.asarray(frames["coe"][:, 0].astype(np.int32))
        self.c1 = cp.asarray(frames["coe"][:, 1].astype(np.int32))
        self.gain = np.float32(1.0 - math.exp(-dt / DRIVE_TAU_S))

    def enable_vertical_mixing(self, mesh, frames, *, dt, cs=0.18):
        """Phase 1's closure is 3-D Smagorinsky (km_opt = 3); the dry hex
        lane mixes only horizontally, and the walls' shear at the first
        levels then goes unstable at the grid scale.  This adds the vertical
        part, operator-split once per step on the edge normal winds:
        ``K = (cs D)^2 |du/dz|``, ``D = (area dz)^(1/3)``, zero flux at the
        ground (the wall law carries the ground stress) and at the lid."""
        cp = self.cp
        z_w = np.asarray(frames["z_w"], np.float64)
        dz = np.diff(z_w)
        dzu = 0.5 * (dz[1:] + dz[:-1])
        a = mesh.arrays
        area = 0.5 * (np.asarray(a["areaCell"], np.float64)[frames["coe"][:, 0]]
                      + np.asarray(a["areaCell"], np.float64)[frames["coe"][:, 1]])
        length = cs * np.cbrt(area[None, :] * dzu[:, None])            # (nlev-1, ne)
        self.mix_l2 = cp.asarray((length * length).astype(np.float32))
        self.mix_rdzu = cp.asarray((1.0 / dzu).astype(np.float32))[:, None]
        self.mix_dt_rdz = cp.asarray((dt / dz).astype(np.float32))[:, None]
        # Explicit stability: K dt / dz^2 <= 0.4, the cap a few cells ever reach.
        self.mix_kmax = np.float32(0.4 * float(dz.min()) ** 2 / dt)

    def mix(self, driver):
        cp = self.cp
        u = driver.atmosphere.saved.normal_velocity
        shear = (u[1:] - u[:-1]) * self.mix_rdzu
        k = cp.minimum(self.mix_l2 * cp.abs(shear), self.mix_kmax)
        flux = k * shear                                             # at interior interfaces
        change = cp.zeros_like(u)
        change[:-1] += flux
        change[1:] -= flux
        change *= self.mix_dt_rdz
        rho = driver.atmosphere.state.rho
        rho_edge = cp.float32(0.5) * (rho[:, self.c0] + rho[:, self.c1])
        u += change
        driver.atmosphere.state.rho_u += rho_edge * change

    def amplitude(self, u):
        return float(self.cp.mean(u[self.upper] @ self.weight))

    def __call__(self, driver):
        cp = self.cp
        if getattr(self, "mix_l2", None) is not None:
            self.mix(driver)
        u = driver.atmosphere.saved.normal_velocity
        delta = np.float32((U_REF - self.amplitude(u)) * self.gain)
        if delta == 0:
            return
        rho = driver.atmosphere.state.rho
        rho_edge = cp.float32(0.5) * (rho[:, self.c0] + rho[:, self.c1])
        u += delta * self.p[None]
        driver.atmosphere.state.rho_u += rho_edge * (delta * self.p[None])


def cell_wind_operator(mesh):
    """Per-cell least-squares (east, north) from the cell's edge normals."""
    a = mesh.arrays
    normal, east, north = edge_frames(mesh)
    ne_, nn_ = np.sum(normal * east, 1), np.sum(normal * north, 1)
    eoc = np.asarray(a["edgesOnCell"], np.int64)
    count = np.asarray(a["nEdgesOnCell"], np.int64)
    used = np.arange(eoc.shape[1])[None] < count[:, None]
    slot = np.where(used, eoc, 0)
    A = np.stack([ne_[slot], nn_[slot]], -1) * used[..., None]      # (nc, m, 2)
    pinv = np.linalg.pinv(A)                                         # (nc, 2, m)
    return slot, pinv


def cell_winds(u, slot, pinv):
    """(nlev, nc, 2) east/north cell winds from edge normal winds (nlev, ne)."""
    return np.einsum("cjm,lcm->lcj", pinv, u[:, slot])


def cube_morphology(mesh, *, origin=(0.0, 0.0)):
    """One H-cube centred at lat 0, lon 0 (the plane origin) as a hex
    product, aggregated through VoronoiGrid."""
    from woof.static.buildings import VoronoiGrid, aggregate, block_raster
    grid = VoronoiGrid.from_mesh(mesh, plane_origin=origin)
    span = 3.0 * H
    raster = block_raster(1, 1, span, span, [(span / 2 - H / 2, span / 2 + H / 2,
                                              span / 2 - H / 2, span / 2 + H / 2, H)],
                          resolution=H / 40)
    shifted = type(raster)(raster.height, raster.x0 - span / 2, raster.y0 - span / 2,
                           raster.px, raster.py, None)
    return aggregate([shifted], grid)


def canyon_morphology(mesh, *, width=H, building=H, height=H, lat_extent=60.0):
    """Street canyons: bars of ``building`` metres (along the equator) every
    ``width + building``, spanning |lat| < lat_extent, as solid cells by
    centre (a hex product with no porous drag)."""
    from woof.static.buildings import BuildingMorphology
    a = mesh.arrays
    nc = int(np.asarray(a["areaCell"]).size)
    lon = np.asarray(a["lonCell"], np.float64)
    lat = np.asarray(a["latCell"], np.float64)
    period = width + building
    if lat_extent > 0:
        bars = 2.0 * math.pi * mesh_radius(mesh) / period
        if abs(bars - round(bars)) > 1e-6:
            raise ValueError(f"the equator ({2.0 * math.pi * mesh_radius(mesh):g} m) is not a whole "
                             f"number of {period:g} m periods; the longitude seam would split a bar")
    x = np.mod(lon * mesh_radius(mesh), period)
    built = (x >= width) & (np.abs(np.degrees(lat)) < lat_extent)
    one = lambda v: np.asarray(v, np.float32)[None]
    hist = np.zeros((1, 1, nc), np.float32)
    hist[0, 0] = built
    hbin = np.full((1, 1, nc), height, np.float32)
    zero = one(np.zeros(nc))
    return BuildingMorphology(
        sampled_area=one(np.asarray(a["areaCell"])), lambda_p=one(built),
        mean_height=one(built * height), std_height=zero, lambda_b=one(built),
        frontal_x=zero, frontal_y=zero, frontal_d45=zero, frontal_d135=zero,
        hi=np.zeros((15, 1, nc), np.float32), bldg_hist=hist, bldg_hbin=hbin,
        bin_edges=np.array([0.0, 400.0]))


def _crossings(x, u, x_lee, x_wind):
    order = np.argsort(x)
    x, u = x[order], u[order]
    behind = x > x_lee
    reattach = np.nan
    xs, us = x[behind], u[behind]
    for n in range(1, xs.size):
        if us[n - 1] < 0.0 <= us[n]:
            frac = -us[n - 1] / (us[n] - us[n - 1])
            reattach = xs[n - 1] + frac * (xs[n] - xs[n - 1]) - x_lee
            break
    ahead = (x < x_wind) & (x > x_wind - 2.0 * H) & (u < 0.0)
    separate = x_wind - float(x[ahead].min()) if ahead.any() else np.nan
    return reattach / H, separate / H


def _sample_steps(n_total, n_avg, every=10):
    if not 1 <= n_avg <= n_total:
        raise ValueError("the averaging window must hold at least one step of the run")
    return set(range(n_total - n_avg, n_total, every))


def run_cube(*, level=4, zoom=1.0, minutes=15.0, average_minutes=6.0, dt=0.1,
             sub_steps=8, nz=40, obstacles=True, radius=RADIUS, balanced=True,
             vertical_mixing=True, report=None) -> dict:
    """The cube benchmark; returns phase 1's metrics."""
    import time

    from woof.hex import obstacles as hex_obstacles

    mesh, driver, frames = build(level, zoom, nz=nz, dt=dt, sub_steps=sub_steps,
                                 radius=radius, balanced=balanced)
    settings = hex_obstacles.ObstacleSettings()
    receipt = None
    if obstacles:
        receipt = hex_obstacles.attach(driver, settings, cube_morphology(mesh), ground_wall=True)
    else:
        # The ground alone, so the with/without pair differs only by the cube.
        empty = canyon_morphology(mesh, lat_extent=-1.0)
        receipt = hex_obstacles.attach(driver, settings, empty, ground_wall=True)
    drive = Drive(driver, mesh, frames, dt=dt)
    if vertical_mixing:
        drive.enable_vertical_mixing(mesh, frames, dt=dt)
    slot, pinv = cell_wind_operator(mesh)
    a = mesh.arrays
    lat = np.asarray(a["latCell"], np.float64)
    lon = np.asarray(a["lonCell"], np.float64)
    x = lon * mesh_radius(mesh)
    y = lat * mesh_radius(mesh)
    spacing = float(np.sqrt(np.asarray(a["areaCell"], np.float64)[
        (np.abs(x) < 3 * H) & (np.abs(y) < 3 * H)].mean()))
    centre = np.flatnonzero((np.abs(y) < 0.6 * spacing) & (np.abs(x) < 6 * H))
    z_mass = 0.5 * (frames["z_w"][1:] + frames["z_w"][:-1])
    k_cube = int(np.searchsorted(z_mass, H))
    inside = np.flatnonzero((np.abs(x) < 0.5 * H - spacing) & (np.abs(y) < 0.5 * H - spacing))
    ref_levels = (z_mass > 1.9 * H) & (z_mass < 2.1 * H)
    if not ref_levels.any():
        raise ValueError("no mass level between 1.9 H and 2.1 H to take the reference wind from")
    n_total = int(round(minutes * 60.0 / dt))
    n_avg = int(round(average_minutes * 60.0 / dt))
    sampled = _sample_steps(n_total, n_avg)
    cp = driver.cp
    floor = np.zeros(centre.size)
    ref = 0.0
    leak = np.zeros(2)
    started = time.perf_counter()
    for step in range(n_total):
        driver.atmosphere = driver.step_device().atmosphere
        drive(driver)
        if step in sampled:
            u = cp.asnumpy(driver.atmosphere.saved.normal_velocity).astype(np.float64)
            winds = cell_winds(u, slot, pinv)
            floor += winds[0, centre, 0]
            ref += float(np.mean(winds[ref_levels][:, centre, 0]))
            if inside.size and obstacles:
                core = winds[:k_cube][:, inside]
                leak += (float(np.mean(np.sum(core * core, -1))), 1.0)
        if report is not None and step % 1000 == 0:
            report(step, time.perf_counter() - started)
    seconds = time.perf_counter() - started
    floor /= len(sampled)
    ref /= len(sampled)
    nan = not bool(cp.all(cp.isfinite(driver.atmosphere.saved.normal_velocity)))
    reattach, separate = _crossings(x[centre], floor / ref, 0.5 * H, -0.5 * H)
    return dict(reattach=float(reattach), separate=float(separate),
                leak=float(np.sqrt(leak[0] / leak[1])) / ref if leak[1] else float("nan"),
                u_ref=ref, nan=nan, cells=int(lat.size), spacing_m=spacing,
                floor_x=(x[centre] / H).tolist(), floor_u=(floor / ref).tolist(),
                seconds=seconds, steps=n_total, obstacles=receipt,
                state_sha256=_state_sha256(driver))


def run_canyon(*, level=4, zoom=1.0, minutes=10.0, average_minutes=4.0, dt=0.1,
               sub_steps=8, nz=40, aspect=1.0, radius=RADIUS, balanced=True,
               vertical_mixing=True, lat_extent=88.0, report=None) -> dict:
    """The street canyon; metrics as obstacle_canyon (vortices, floor/roof
    wind, vortex centre) from the equatorial band.  The bars run to |lat| <
    ``lat_extent``; bars that stop at 60 degrees leave the street open at both
    ends and the air takes the way round (a uniform -0.2 U(2H) back-flow, no
    vortex, at 2.7 m and at 1.3 m), so the default runs them almost pole to
    pole, the closed analogue of phase 1's periodic 2-D canyon."""
    import time

    from woof.hex import obstacles as hex_obstacles

    height = aspect * H
    mesh, driver, frames = build(level, zoom, nz=nz, ztop=5 * height, dt=dt, sub_steps=sub_steps,
                                 radius=radius, balanced=balanced)
    receipt = hex_obstacles.attach(driver, hex_obstacles.ObstacleSettings(),
                                   canyon_morphology(mesh, height=height, lat_extent=lat_extent),
                                   ground_wall=True)
    drive = Drive(driver, mesh, frames, dt=dt, height=height)
    if vertical_mixing:
        drive.enable_vertical_mixing(mesh, frames, dt=dt)
    slot, pinv = cell_wind_operator(mesh)
    a = mesh.arrays
    lat = np.asarray(a["latCell"], np.float64)
    lon = np.asarray(a["lonCell"], np.float64)
    y = lat * mesh_radius(mesh)
    period = 2 * H
    xs = np.mod(lon * mesh_radius(mesh), period)
    spacing = float(np.sqrt(np.asarray(a["areaCell"], np.float64).mean()))
    band = np.flatnonzero(np.abs(y) < 0.75 * spacing)
    street = band[xs[band] < H]
    z_w = frames["z_w"]
    z_mass = 0.5 * (z_w[1:] + z_w[:-1])
    n_total = int(round(minutes * 60.0 / dt))
    sampled = _sample_steps(n_total, int(round(average_minutes * 60.0 / dt)))
    cp = driver.cp
    nbins = 10
    bins = np.minimum((xs[street] / H * nbins).astype(int), nbins - 1)
    u_bins = np.zeros((nz, nbins))
    counts = np.bincount(bins, minlength=nbins).astype(float)
    started = time.perf_counter()
    for step in range(n_total):
        driver.atmosphere = driver.step_device().atmosphere
        drive(driver)
        if step in sampled:
            u = cp.asnumpy(driver.atmosphere.saved.normal_velocity).astype(np.float64)
            east = cell_winds(u, slot, pinv)[:, street, 0]
            for b in range(nbins):
                u_bins[:, b] += east[:, bins == b].mean(axis=1) if counts[b] else 0.0
        if report is not None and step % 1000 == 0:
            report(step, time.perf_counter() - started)
    seconds = time.perf_counter() - started
    u_bins /= len(sampled)
    ref = float(u_bins[(z_mass > 1.9 * height) & (z_mass < 2.1 * height)].mean())
    in_canyon = z_mass < height
    dz = np.diff(z_w)
    psi = np.cumsum(u_bins * dz[:, None], axis=0)
    sub = np.where(in_canyon[:, None], psi, np.inf)
    kz, ix = np.unravel_index(np.argmin(sub), sub.shape)
    centre_col = nbins // 2
    column = u_bins[in_canyon, centre_col]
    vortices = int(np.count_nonzero(np.diff(np.sign(column)) != 0))
    roof = int(np.argmin(np.abs(z_mass - height)))
    return dict(vortex_x=float((ix + 0.5) / nbins) if vortices else None,
                vortex_z=float(z_mass[kz] / height) if vortices else None,
                floor_u=float(u_bins[0, centre_col] / ref), roof_u=float(u_bins[roof, centre_col] / ref),
                vortices=vortices,
                centre_u=(column / ref).tolist(), u_ref=ref, cells=int(lat.size),
                spacing_m=spacing, seconds=seconds, steps=n_total, obstacles=receipt,
                nan=not bool(cp.all(cp.isfinite(driver.atmosphere.saved.normal_velocity))),
                u_section=(u_bins / ref).tolist(), z=(z_mass / height).tolist(),
                state_sha256=_state_sha256(driver))


def _state_sha256(driver) -> str:
    import hashlib
    cp = driver.cp
    h = hashlib.sha256()
    s = driver.atmosphere.state
    for name in ("rho", "rho_theta", "rho_u", "rho_w"):
        h.update(cp.asnumpy(getattr(s, name)).tobytes())
    return h.hexdigest()


__all__ = ["CUBE_GATES", "Drive", "build", "canyon_morphology", "cube_morphology",
           "run_canyon", "run_cube", "sphere_mesh"]
