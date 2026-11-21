"""MPAS-Seaice v8.4.2: the weak-form stress divergence.

``config_stress_divergence_scheme = 'weak'`` (``mpas_seaice_velocity_solver_weak.F``):

- init, on the host (:func:`initialize`): ``seaice_init_velocity_solver_weak``,
  i.e. ``seaice_normal_vectors`` (``mpas_seaice_mesh.F``). It computes the
  edge normals of each cell (``normalVectorPolygon``) and of each vertex's
  dual triangle (``normalVectorTriangle``), plus the rotated latitudes
  (``latCellRotated``, ``latVertexRotated``). On a plane these are
  ``(xEdge - xCell)/|...|``. On a sphere each point is rotated onto the
  rotated equator (``removeMetricTerms = .true.``, the rotation matrices
  applied with ``matmul``) and projected on the great circle's normal.
- each elastic subcycle, on the GPU (``weak_v842.cu``):
  ``seaice_internal_stress_weak``, which is the strain tensor, the stress
  tensor (``seaice_evp_constitutive_relation`` and ``..._revised``, as for
  the variational scheme) and the stress divergence;
- around the subcycles: ``init_subcycle_variables``,
  ``seaice_final_divergence_shear_weak`` and ``principal_stresses_driver``
  (weak) from ``mpas_seaice_velocity_solver.F``.

The GPU kernels compute in the precision of the arrays (r4: RKIND 4, r8:
RKIND 8), one IEEE operation at a time in gfortran -O0 -ffp-contract=off's
order. The init runs in numpy and Python at RKIND, in the same order.

The C library's words. On a sphere the init calls ``atan2``, ``asin``,
``sin`` and ``cos``, and every subcycle's metric terms call ``tan`` of the
rotated latitudes. glibc 2.44 rounds ``asinf``, ``atan2f`` and ``tanf``
correctly, but not ``sinf``, ``cosf`` or the binary64 functions. So no
host-independent computation gives v8.4.2's words. Unless every argument
is one that every libm agrees on (a plane, where the latitudes are zero),
the weak form is refused by name unless ``metric_libm`` chooses one of:

- ``"host"``: the host C library through ctypes, at RKIND (``sinf`` etc.
  at RKIND 4). These are the oracle's words on a glibc 2.44 host.
- ``"correctly-rounded"``: 70-digit decimal, rounded once.

The tangents are computed on the host once per run
(:func:`metric_tangents`), because the latitudes do not change after the
init, and passed to the kernels.

The ``matmul`` of the rotation matrices is libgfortran's, so it is not
gfortran -O0 code. On an x86-64 host with AVX2 and FMA (and on arm64) its
``dest += a*b`` loop is contracted to fused multiply-adds, summing from 0
in index order. The init does the same with exact FMAs (``math.fma``).
This matches the oracle's words, which come from the pb host, AMD64 with
AVX2 and FMA.

Opt-in correction (``corrections``; v8.4.2's words stay the default):

- ``"weak_ridge_delta"``: ``seaice_final_divergence_shear_weak`` assigns
  the whole ``Delta`` array (``Delta = sqrt(...)``) in each pass of its cell
  loop. After the loop, every cell's ``Delta`` is therefore the last owned
  cell's, and that value goes into the ridging's ``ridgeShear`` under the
  column package. With the correction, each cell uses its own ``Delta``.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import decimal
import math
from fractions import Fraction
from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex.errors import ConfigurationRefusal

from .variational_basis_v842 import _DEC, _EPS, _dec_atan, _dec_asin, _dec_tan, _round_to
from .velocity_v842 import launch_options

SOURCE_FILES = (
    "src/core_seaice/shared/mpas_seaice_velocity_solver_weak.F:49-777",
    "src/core_seaice/shared/mpas_seaice_mesh.F:711-1063,1290-1611,1740-1778,2332-2363",
    "src/core_seaice/shared/mpas_seaice_velocity_solver.F:140-160,2149-2308,3026-3046,3074-3108",
)
#: strict options, as evp_v842's (the unit carries that file's constitutive relation)
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                   "--prec-div=true", "--prec-sqrt=true")
MODULE_OPTIONS = COMPILE_OPTIONS
PRECISIONS = {"r4": np.float32, "r8": np.float64}
METRIC_LIBMS = ("host", "correctly-rounded")
CORRECTIONS = ("weak_ridge_delta",)
#: the init's outputs, in the velocity_weak pool
OUTPUTS = ("normalVectorPolygon", "normalVectorTriangle", "latCellRotated", "latVertexRotated")
#: the velocity_weak pool's per-cell fields
CELL_FIELDS = ("strain11", "strain22", "strain12", "stress11", "stress22", "stress12", "replacementPressure",
               "principalStress1", "principalStress2")
_BLOCK = 128


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


def _check_corrections(corrections) -> None:
    unknown = set(corrections) - set(CORRECTIONS)
    if unknown:
        raise ValueError(f"unknown corrections {sorted(unknown)}; known: {CORRECTIONS}")


def _check_libm(metric_libm) -> None:
    if metric_libm not in (None, *METRIC_LIBMS):
        raise ValueError(f"metric_libm is one of {METRIC_LIBMS}, not {metric_libm!r}")


def _refuse_libm(what: str):
    _refuse("config_stress_divergence_scheme", "weak",
            f"on a sphere the weak form's {what} call the C library (glibc 2.44 does not round sinf, cosf "
            "or its binary64 functions correctly), so no host-independent computation gives v8.4.2's "
            "words; choose metric_libm 'host' (the host C library: the oracle's words on a glibc 2.44 "
            "host) or 'correctly-rounded'",
            "metric_libm = 'host' | 'correctly-rounded', or config_stress_divergence_scheme = 'variational'")


# --- the C library -------------------------------------------------------------

@lru_cache(maxsize=None)
def _host_function(name: str, real):
    libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
    c = ctypes.c_float if real is np.float32 else ctypes.c_double
    fn = getattr(libm, name + ("f" if real is np.float32 else ""))
    fn.restype = c
    fn.argtypes = [c, c] if name == "atan2" else [c]
    return fn


_PI = None


def _dec_pi() -> decimal.Decimal:
    global _PI
    if _PI is None:
        _PI = _DEC.multiply(_dec_atan(decimal.Decimal(1)), 4)
    return _PI


def _dec_sin_cos(x: float) -> tuple[decimal.Decimal, decimal.Decimal]:
    """Taylor series at 70 digits (|x| <= ~pi here)."""
    d = decimal.Decimal(x)
    x2, s, c = _DEC.multiply(d, d), decimal.Decimal(0), decimal.Decimal(0)
    term, n = d, 1
    while abs(term) > _EPS:
        s = _DEC.add(s, term)
        term, n = _DEC.minus(_DEC.divide(_DEC.multiply(term, x2), (n + 1) * (n + 2))), n + 2
    term, n = decimal.Decimal(1), 0
    while abs(term) > _EPS:
        c = _DEC.add(c, term)
        term, n = _DEC.minus(_DEC.divide(_DEC.multiply(term, x2), (n + 1) * (n + 2))), n + 2
    return s, c


def _cr(name: str, real, x: float, y: float | None = None):
    """The correctly rounded ``name`` at ``real`` (``atan2(x, y)`` is atan2's (y, x) order: x first)."""
    if name == "atan2":
        a, b = x, y   # atan2(a, b)
        if math.isnan(a) or math.isnan(b):
            return real(np.nan)
        if a == 0.0:
            if b > 0.0 or (b == 0.0 and math.copysign(1.0, b) > 0):
                return real(a)
            return real(math.copysign(1.0, a)) * _round_to(_dec_pi(), real)
        if b == 0.0:
            return _round_to(_DEC.multiply(_dec_pi(), decimal.Decimal("0.5") if a > 0 else
                                           decimal.Decimal("-0.5")), real)
        t = _dec_atan(_DEC.divide(decimal.Decimal(a), decimal.Decimal(b)))
        if b < 0.0:
            t = _DEC.add(t, _dec_pi()) if a > 0.0 else _DEC.subtract(t, _dec_pi())
        return _round_to(t, real)
    if math.isnan(x) or math.isinf(x):
        return real(np.nan)
    if name == "asin":
        if abs(x) > 1.0:
            return real(np.nan)
        if abs(x) == 1.0:
            return _round_to(_DEC.multiply(_dec_atan(decimal.Decimal(1)), 2 if x > 0 else -2), real)
        return real(x) if x == 0.0 else _round_to(_dec_asin(x), real)
    if name == "tan":
        return real(x) if x == 0.0 else _round_to(_dec_tan(x), real)
    if x == 0.0:
        return real(x) if name == "sin" else real(1.0)
    s, c = _dec_sin_cos(x)
    return _round_to(s if name == "sin" else c, real)


def libm(name: str, real, metric_libm: str, x, y=None):
    """``name`` (asin, atan2, sin, cos, tan) of RKIND words at ``real``."""
    if metric_libm == "host":
        fn = _host_function(name, real)
        return real(fn(float(x)) if y is None else fn(float(x), float(y)))
    return _cr(name, real, float(x), None if y is None else float(y))


# --- the init ----------------------------------------------------------------

def _fma(a, b, c, real):
    """``round(a*b + c)`` at ``real``, exactly."""
    if real is np.float64:
        return np.float64(math.fma(float(a), float(b), float(c)))
    s = math.fma(float(a), float(b), float(c))
    r = np.float32(s)
    if float(r) != s and math.isfinite(s):
        # a binary64 sum on a binary32 midpoint may have been rounded onto it
        lo, hi = sorted((float(r), float(np.nextafter(r, np.float32(np.inf if float(r) < s else -np.inf)))))
        if s == (lo + hi) / 2:
            exact = Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c))
            r = np.float32(hi if exact > Fraction(s) else lo) if exact != Fraction(s) else r
    return r


def _matvec(m, v, real):
    """libgfortran's ``matmul(m, v)`` for a 3x3 ``m`` (rows) and a 3-vector:
    ``dest = 0; dest += m(i,k)*v(k)`` for k = 1..3, contracted."""
    out = []
    for i in range(3):
        acc = real(0.0)
        for k in range(3):
            acc = _fma(m[i][k], v[k], acc, real)
        out.append(acc)
    return out


def _rotate(x, y, z, rotate):
    """``seaice_grid_rotation_forward``."""
    return (-z, y, x) if rotate else (x, y, z)


def _cross(a, b):
    """``cross_product_3space``."""
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _matrices(lat, lon, real, metric_libm):
    """The y and z rotation matrices for ``removeMetricTerms = .true.``."""
    zero, one = real(0.0), real(1.0)
    cl, sl = libm("cos", real, metric_libm, lat), libm("sin", real, metric_libm, lat)
    co, so = libm("cos", real, metric_libm, -lon), libm("sin", real, metric_libm, -lon)
    ym = ((cl, zero, sl), (zero, one, zero), (-sl, zero, cl))
    zm = ((co, -so, zero), (so, co, zero), (zero, zero, one))
    return ym, zm


def _spherical_normal(ym, zm, point, a, b, negate, real):
    """The normal's two components from the edge ``point`` and the chord ``a -> b``."""
    one = real(1.0)
    with np.errstate(all="ignore"):
        edge = _matvec(ym, _matvec(zm, point, real), real)
        ea = _matvec(ym, _matvec(zm, a, real), real)
        eb = _matvec(ym, _matvec(zm, b, real), real)
        chord = (eb[0] - ea[0], eb[1] - ea[1], eb[2] - ea[2])
        n = _cross(chord, edge)
        if negate:
            n = tuple(real(-1.0) * c for c in n)
        norm = np.sqrt((n[0] * n[0] + n[1] * n[1]) + n[2] * n[2])
        n = (n[0] / norm, n[1] / norm, n[2] / norm)
        east = (-edge[1], edge[0], real(0.0))
        scale = np.sqrt(east[0] * east[0] + east[1] * east[1])
        east = (east[0] / scale, east[1] / scale, east[2] / scale)
        first = (n[0] * east[0] + n[1] * east[1]) + n[2] * east[2]
        clipped = max(min(first, one), -one)
        second = real(math.copysign(1.0, float(n[2]))) * np.sqrt(one - clipped * clipped)
    return first, second


def normal_vectors(mesh: dict, configs: dict, *, real, out: dict, metric_libm: str | None = None) -> None:
    """``seaice_normal_vectors(..., removeMetricTerms = .true., ...)`` into
    ``out`` (numpy arrays with MPAS's extra element, C-order of the Fortran
    layout: ``normalVectorPolygon`` (nCells+1, maxEdges, 2),
    ``normalVectorTriangle`` (nVertices+1, vertexDegree, 2)). Only the words
    v8.4.2 writes are written."""
    _check_libm(metric_libm)
    nc, nv = int(mesh["nCells"]), int(mesh["nVertices"])
    vd = int(mesh["vertexDegree"])
    sphere = bool(configs.get("mesh:on_a_sphere", configs.get("on_a_sphere", False)))
    rotate = bool(configs.get("config_rotate_cartesian_grid", True))
    ne = np.asarray(mesh["nEdgesOnCell"]).astype(np.int64)
    eoc = np.asarray(mesh["edgesOnCell"]).astype(np.int64) - 1
    eov = np.asarray(mesh["edgesOnVertex"]).astype(np.int64) - 1
    interior = np.asarray(mesh["interiorVertex"])
    poly, tri = out["normalVectorPolygon"], out["normalVectorTriangle"]
    xe, ye = mesh["xEdge"].astype(real), mesh["yEdge"].astype(real)
    with np.errstate(all="ignore"):
        if not sphere:
            xc, yc = mesh["xCell"].astype(real), mesh["yCell"].astype(real)
            xv, yv = mesh["xVertex"].astype(real), mesh["yVertex"].astype(real)
            for n in sorted(set(ne[:nc].tolist())):
                cells = np.flatnonzero(ne[:nc] == n)
                if n < 1:
                    continue
                e = eoc[cells, :n]
                dx, dy = xe[e] - xc[cells, None], ye[e] - yc[cells, None]
                length = np.sqrt(dx * dx + dy * dy)
                poly[cells, :n, 0] = dx / length
                poly[cells, :n, 1] = dy / length
            tri[:nv] = 0
            inner = np.flatnonzero(interior[:nv] == 1)
            e = eov[inner, :vd]
            dx, dy = xe[e] - xv[inner, None], ye[e] - yv[inner, None]
            length = np.sqrt(dx * dx + dy * dy)
            tri[inner, :, 0] = dx / length
            tri[inner, :, 1] = dy / length
            out["latCellRotated"][...] = 0
            out["latVertexRotated"][...] = 0
            return
    if metric_libm is None:
        _refuse_libm("normal vectors and metric terms")
    radius = real(configs.get("mesh:sphere_radius", configs.get("sphere_radius", 0.0)))
    ze = mesh["zEdge"].astype(real)
    cell = [mesh[k].astype(real) for k in ("xCell", "yCell", "zCell")]
    vert = [mesh[k].astype(real) for k in ("xVertex", "yVertex", "zVertex")]
    voe = np.asarray(mesh["verticesOnEdge"]).astype(np.int64)
    coe = np.asarray(mesh["cellsOnEdge"]).astype(np.int64)

    def point(arrays, i):
        return _rotate(arrays[0][i], arrays[1][i], arrays[2][i], rotate)

    def edge_point(i):
        return _rotate(xe[i], ye[i], ze[i], rotate)

    def lat_lon(p):
        with np.errstate(all="ignore"):
            arg = p[2] / radius
        return libm("asin", real, metric_libm, arg), libm("atan2", real, metric_libm, p[1], p[0])

    for c in range(nc):
        centre = point(cell, c)
        lat, lon = lat_lon(centre)
        ym, zm = _matrices(lat, lon, real, metric_libm)
        for j in range(int(ne[c])):
            e = int(eoc[c, j])
            a, b = point(vert, voe[e, 0] - 1), point(vert, voe[e, 1] - 1)
            poly[c, j] = _spherical_normal(ym, zm, edge_point(e), a, b, c + 1 == coe[e, 1], real)
        out["latCellRotated"][c] = lat
    nvs = int(mesh["nVerticesSolve"])
    for v in range(nvs):
        tri[v] = 0
        out["latVertexRotated"][v] = 0
        if interior[v] != 1:
            continue
        lat, lon = lat_lon(point(vert, v))
        ym, zm = _matrices(lat, lon, real, metric_libm)
        for j in range(vd):
            e = int(eov[v, j])
            a, b = point(cell, coe[e, 0] - 1), point(cell, coe[e, 1] - 1)
            tri[v, j] = _spherical_normal(ym, zm, edge_point(e), a, b, v + 1 == voe[e, 0], real)
        out["latVertexRotated"][v] = lat


def initialize(*, mesh: dict, configs: dict, real, out: dict | None = None,
               metric_libm: str | None = None) -> dict:
    """``seaice_init_velocity_solver_weak`` on host arrays (numpy, MPAS's
    names, C-order with MPAS's extra element). ``mesh``: ``nCells``,
    ``nVertices``, ``nVerticesSolve``, ``vertexDegree``, ``nEdgesOnCell``,
    ``edgesOnCell``, ``edgesOnVertex``, ``verticesOnEdge``, ``cellsOnEdge``,
    ``interiorVertex`` (the boundary pool's), ``x/y/zEdge``, ``x/y/zCell``,
    ``x/y/zVertex``. ``configs``: ``config_rotate_cartesian_grid``,
    ``mesh:on_a_sphere``, ``mesh:sphere_radius``. ``out`` holds
    :data:`OUTPUTS` to write in place; missing ones are allocated as zeros."""
    nc, nv, me, vd = (int(mesh[k]) for k in ("nCells", "nVertices", "maxEdges", "vertexDegree"))
    out = {} if out is None else out
    shapes = {"normalVectorPolygon": (nc + 1, me, 2), "normalVectorTriangle": (nv + 1, vd, 2),
              "latCellRotated": (nc + 1,), "latVertexRotated": (nv + 1,)}
    for name, shape in shapes.items():
        if out.get(name) is None:
            out[name] = np.zeros(shape, real)
        elif tuple(out[name].shape) != shape:
            raise ValueError(f"{name} has shape {out[name].shape}, not {shape}")
    normal_vectors(mesh, configs, real=real, out=out, metric_libm=metric_libm)
    return out


def _host(a):
    if a is None or np.isscalar(a) or isinstance(a, (int, float, np.ndarray)):
        return a
    return a.get()


def initialize_pools(pools: dict, configs: dict, *, metric_libm: str | None = None) -> None:
    """:func:`initialize` on device pools: the mesh and ``interiorVertex``
    copied to the host, the velocity_weak pool's outputs computed in place
    (the words v8.4.2 does not write keep theirs) and copied back."""
    from .remap_init_v842 import _like
    mesh = {k: _host(v) for k, v in pools["mesh"].items()}
    mesh["interiorVertex"] = _host(pools["boundary"]["interiorVertex"])
    vw = pools["velocity_weak"]
    out = {name: (None if vw.get(name) is None else np.array(_host(vw[name]), copy=True)) for name in OUTPUTS}
    real = pools["velocity_solver"]["uVelocity"].dtype.type
    out = initialize(mesh=mesh, configs=configs, real=real, out=out, metric_libm=metric_libm)
    for name in OUTPUTS:
        target = vw.get(name)
        if isinstance(target, np.ndarray):
            target[...] = out[name]
        elif target is None:
            vw[name] = _like(pools["mesh"]["xCell"], out[name])
        else:
            target.set(np.ascontiguousarray(out[name]))


def metric_tangents(lat_cell, lat_vertex, *, metric_libm: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``tan(latCellRotated)`` and ``tan(latVertexRotated)``, every element,
    at the latitudes' precision: the words the metric terms' ``tan`` gives.
    Zero latitudes (a plane) need no library; any other refuses without
    ``metric_libm``."""
    _check_libm(metric_libm)
    out = []
    for lat in (np.asarray(_host(lat_cell)), np.asarray(_host(lat_vertex))):
        real = lat.dtype.type
        tan = np.zeros_like(lat)
        nontrivial = np.flatnonzero(lat != 0)
        if nontrivial.size and metric_libm is None:
            _refuse_libm("metric terms")
        tan[lat == 0] = lat[lat == 0]
        for i in nontrivial:
            tan[i] = libm("tan", real, metric_libm, lat[i])
        out.append(tan)
    return out[0], out[1]


# --- the GPU kernels ---------------------------------------------------------

def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    root = files("woof.hex.mpas_seaice")
    body = root.joinpath("evp_variational_v842.cu").read_text(encoding="utf-8")
    weak = root.joinpath("weak_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body + "\n" + weak


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"seaice_weak_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(precision: str, name: str):
    import cupy as cp
    return _module_on(cp.cuda.Device().id, precision, launch_options(COMPILE_OPTIONS)).get_function(name)


def _precision_of(array) -> str:
    for name, dtype in PRECISIONS.items():
        if array.dtype == dtype:
            return name
    raise ValueError(f"weak-form arrays must be float32 (r4) or float64 (r8), not {array.dtype}")


def _check(name: str, array, dtype, shape) -> None:
    import cupy as cp
    if (not isinstance(array, cp.ndarray) or array.dtype != dtype or tuple(array.shape) != tuple(shape)
            or not array.flags.c_contiguous):
        raise ValueError(f"{name} must be a contiguous device {np.dtype(dtype).name} array of shape {tuple(shape)}")


def _grid(n: int) -> tuple[int]:
    return ((max(n, 1) + _BLOCK - 1) // _BLOCK,)


def _radius(sphere_radius) -> float:
    """The routines' sphereRadius: sphere_radius, or 1 where it is 0 (a plane)."""
    return 1.0 if float(sphere_radius) == 0.0 else float(sphere_radius)


def internal_stress_weak(*, mesh: dict, solver: dict, weak: dict, consts, tangents, elastic_time_step: float,
                         revised_evp: bool, sphere_radius: float, check: bool = True) -> None:
    """``seaice_internal_stress_weak`` for one block, in place.

    ``mesh``: ``nEdgesOnCell``, ``verticesOnCell``, ``edgesOnCell``,
    ``verticesOnEdge``, ``cellsOnVertex``, ``edgesOnVertex``,
    ``cellsOnEdge``, ``dvEdge``, ``dcEdge``, ``areaCell``, ``areaTriangle``,
    ``vertexDegree``. ``solver``: ``solveStress``, ``solveVelocity``,
    ``uVelocity``, ``vVelocity``, ``icePressure``, ``stressDivergenceU/V``.
    ``weak``: ``normalVectorPolygon``, ``normalVectorTriangle``,
    ``strain11/22/12``, ``stress11/22/12``, ``replacementPressure``.
    ``tangents``: device ``(tan(latCellRotated), tan(latVertexRotated))``
    (:func:`metric_tangents`). ``consts``: ``evp_v842.init_evp``'s. The
    loops run to ``evp_v842.loop_count`` (nCells, nVertices)."""
    from .evp_v842 import loop_count
    precision = _precision_of(solver["uVelocity"])
    dtype = PRECISIONS[precision]
    n_cells = weak["strain11"].shape[0]
    n_vertices = solver["uVelocity"].shape[0]
    max_edges = mesh["verticesOnCell"].shape[1]
    vd = int(mesh["vertexDegree"])
    if check:
        _check_internal_stress(mesh, solver, weak, consts, tangents, dtype, n_cells, n_vertices, max_edges, vd)
    radius = dtype(_radius(sphere_radius))
    nl, nvl = loop_count("cells", n_cells), loop_count("vertices", n_vertices)
    w = weak
    _function(precision, "seaice_strain_tensor_weak_v842")(_grid(nl), (_BLOCK,), (
        w["strain11"], w["strain22"], w["strain12"], solver["uVelocity"], solver["vVelocity"],
        w["normalVectorPolygon"], tangents[0], solver["solveStress"], mesh["nEdgesOnCell"],
        mesh["verticesOnCell"], mesh["edgesOnCell"], mesh["verticesOnEdge"], mesh["dvEdge"], mesh["areaCell"],
        radius, np.int32(nl), np.int32(max_edges)))
    _function(precision, "seaice_stress_tensor_weak_v842")(_grid(nl), (_BLOCK,), (
        w["stress11"], w["stress22"], w["stress12"], w["strain11"], w["strain22"], w["strain12"],
        solver["icePressure"], w["replacementPressure"], solver["solveStress"], consts,
        dtype(elastic_time_step), np.int32(bool(revised_evp)), np.int32(nl)))
    _function(precision, "seaice_stress_divergence_weak_v842")(_grid(nvl), (_BLOCK,), (
        solver["stressDivergenceU"], solver["stressDivergenceV"], w["stress11"], w["stress22"], w["stress12"],
        w["normalVectorTriangle"], tangents[1], solver["solveVelocity"], mesh["cellsOnVertex"],
        mesh["edgesOnVertex"], mesh["cellsOnEdge"], mesh["areaTriangle"], mesh["dcEdge"], radius,
        np.int32(nvl), np.int32(vd)))


def _check_internal_stress(mesh, solver, weak, consts, tangents, dtype, n_cells, n_vertices, max_edges, vd):
    import cupy as cp
    n_edges = mesh["verticesOnEdge"].shape[0]
    _check("nEdgesOnCell", mesh["nEdgesOnCell"], cp.int32, (n_cells,))
    for name in ("verticesOnCell", "edgesOnCell"):
        _check(name, mesh[name], cp.int32, (n_cells, max_edges))
    for name in ("cellsOnVertex", "edgesOnVertex"):
        _check(name, mesh[name], cp.int32, (n_vertices, vd))
    for name in ("verticesOnEdge", "cellsOnEdge"):
        _check(name, mesh[name], cp.int32, (n_edges, 2))
    for name in ("dvEdge", "dcEdge"):
        _check(name, mesh[name], dtype, (n_edges,))
    _check("areaCell", mesh["areaCell"], dtype, (n_cells,))
    _check("areaTriangle", mesh["areaTriangle"], dtype, (n_vertices,))
    _check("solveStress", solver["solveStress"], cp.int32, (n_cells,))
    _check("solveVelocity", solver["solveVelocity"], cp.int32, (n_vertices,))
    for name in ("uVelocity", "vVelocity", "stressDivergenceU", "stressDivergenceV"):
        _check(name, solver[name], dtype, (n_vertices,))
    _check("icePressure", solver["icePressure"], dtype, (n_cells,))
    _check("normalVectorPolygon", weak["normalVectorPolygon"], dtype, (n_cells, max_edges, 2))
    _check("normalVectorTriangle", weak["normalVectorTriangle"], dtype, (n_vertices, vd, 2))
    for name in ("strain11", "strain22", "strain12", "stress11", "stress22", "stress12", "replacementPressure"):
        _check(name, weak[name], dtype, (n_cells,))
    _check("tan(latCellRotated)", tangents[0], dtype, (n_cells,))
    _check("tan(latVertexRotated)", tangents[1], dtype, (n_vertices,))
    _check("consts", consts, dtype, (3,))


def init_subcycle_variables(*, mesh: dict, solver: dict, weak: dict) -> None:
    """``init_subcycle_variables`` (weak), in place: the vertex part is
    :func:`velocity_v842.init_subcycle_variables`'s kernel; the strains are
    zeroed whole and the stresses of the cells not solved for zeroed."""
    import cupy as cp
    from . import velocity_v842 as vel
    from .evp_v842 import loop_count
    precision = _precision_of(solver["uVelocity"])
    dtype = PRECISIONS[precision]
    n_vertices = solver["uVelocity"].shape[0]
    n_cells = weak["strain11"].shape[0]
    nvs = int(mesh["nVerticesSolve"])
    vel._solve_bounds(nvs, n_vertices)
    vel._vertex_fields(solver, ("uVelocity", "vVelocity", "oceanStressCoeff", "stressDivergenceU",
                                "stressDivergenceV"), n_vertices, dtype)
    _check("solveVelocity", solver["solveVelocity"], cp.int32, (n_vertices,))
    _check("solveStress", solver["solveStress"], cp.int32, (n_cells,))
    for name in ("strain11", "strain22", "strain12", "stress11", "stress22", "stress12"):
        _check(name, weak[name], dtype, (n_cells,))
    vel._function(precision, "seaice_init_subcycle_vertices_v842")(vel._grid(n_vertices), (vel._BLOCK,), (
        solver["stressDivergenceU"], solver["stressDivergenceV"], solver["uVelocity"], solver["vVelocity"],
        solver["oceanStressCoeff"], solver["solveVelocity"], np.int32(nvs), np.int32(n_vertices)))
    w = weak
    _function(precision, "seaice_init_subcycle_cells_weak_v842")(_grid(n_cells), (_BLOCK,), (
        w["strain11"], w["strain22"], w["strain12"], w["stress11"], w["stress22"], w["stress12"],
        solver["solveStress"], np.int32(n_cells), np.int32(loop_count("cells", n_cells))))


def final_divergence_shear_weak(*, mesh: dict, solver: dict, weak: dict, ridging: dict | None,
                                use_column_package: bool, corrections=()) -> None:
    """``seaice_final_divergence_shear_weak``, in place over the first
    ``nCellsSolve`` cells: ``solver["divergence"]``, ``solver["shear"]`` (in
    s^-1) and, with ``use_column_package``, ``ridging["ridgeConvergence"]``
    and ``ridging["ridgeShear"]`` (v8.4.2's whole-array ``Delta``: see
    ``weak_ridge_delta``)."""
    _check_corrections(corrections)
    precision = _precision_of(solver["divergence"])
    dtype = PRECISIONS[precision]
    n_cells = weak["strain11"].shape[0]
    ncs = int(mesh["nCellsSolve"])
    if not 0 <= ncs <= n_cells:
        raise ValueError("nCellsSolve must lie within nCells")
    for name in ("divergence", "shear"):
        _check(name, solver[name], dtype, (n_cells,))
    for name in ("strain11", "strain22", "strain12"):
        _check(name, weak[name], dtype, (n_cells,))
    if use_column_package:
        if ridging is None:
            raise ValueError("use_column_package needs the ridging arrays")
        for name in ("ridgeConvergence", "ridgeShear"):
            _check(name, ridging[name], dtype, (n_cells,))
        ridge = (ridging["ridgeConvergence"], ridging["ridgeShear"])
    else:
        ridge = (solver["divergence"], solver["shear"])   # not touched
    _function(precision, "seaice_final_divergence_shear_weak_v842")(_grid(ncs), (_BLOCK,), (
        solver["divergence"], solver["shear"], *ridge, weak["strain11"], weak["strain22"], weak["strain12"],
        np.int32(bool(use_column_package)), np.int32("weak_ridge_delta" in corrections), np.int32(ncs)))


def principal_stresses(*, mesh: dict, weak: dict) -> None:
    """``principal_stresses_driver`` (weak), in place over the first
    ``nCellsSolve`` cells: velocity_v842's per-point kernel, one point per cell."""
    import cupy as cp
    from . import velocity_v842 as vel
    precision = _precision_of(weak["stress11"])
    dtype = PRECISIONS[precision]
    n_cells = weak["stress11"].shape[0]
    ncs = int(mesh["nCellsSolve"])
    if not 0 <= ncs <= n_cells:
        raise ValueError("nCellsSolve must lie within nCells")
    for name in ("stress11", "stress22", "stress12", "replacementPressure", "principalStress1",
                 "principalStress2"):
        _check(name, weak[name], dtype, (n_cells,))
    ones = cp.ones(n_cells, cp.int32)
    w = weak
    vel._function(precision, "seaice_principal_stresses_variational_v842")(vel._grid(ncs), (vel._BLOCK,), (
        w["principalStress1"], w["principalStress2"], w["stress11"], w["stress22"], w["stress12"],
        w["replacementPressure"], ones, np.int32(ncs), np.int32(1)))
