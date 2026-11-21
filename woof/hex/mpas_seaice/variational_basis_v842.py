"""MPAS-Seaice v8.4.2: the variational solver's basis functions, computed on the host.

``seaice_init_velocity_solver_variational``
(``mpas_seaice_velocity_solver_variational.F``) sets up, once, what the
variational internal stress (:mod:`~woof.hex.mpas_seaice.evp_v842`) reads:

- ``seaice_calc_local_coords`` (``..._variational_shared.F``): each cell's
  vertices in the cell's frame; on a plane ``xVertex - xCell``; on a sphere
  the (rotated, under ``config_rotate_cartesian_grid``) vertex position
  projected onto the local east and north unit vectors at the (rotated)
  cell centre (``seaice_project_3D_vector_onto_local_2D``,
  ``mpas_seaice_mesh.F``);
- ``seaice_calc_variational_metric_terms``: ``tanLatVertexRotatedOverRadius``,
  ``tan(asin(z'/R))/R`` at each vertex under ``config_include_metric_terms``
  (``z'`` the rotated ``z``), else zero;
- ``seaice_cell_vertices_at_vertex`` (``mpas_seaice_mesh.F``):
  ``cellVerticesAtVertex``;
- the basis, by ``config_variational_basis``:

  - ``'wachspress'`` (the default; ``..._wachspress.F``): the Wachspress
    coordinates' coefficients (``calc_wachspress_coefficients``), their
    gradients at the cell's vertices (``basisGradientU/V``) and the integrals
    of a basis function times another's gradient and times the other
    (``basisIntegralsU/V/Metric``) over the cell's triangles about its
    centre, by ``config_wachspress_integration_type`` (``'trapezoidal'``,
    ``'dunavant'`` (the default), ``'fekete'``) at
    ``config_wachspress_integration_order`` (default 8);
  - ``'pwl'`` (``..._pwl.F``): the piecewise-linear basis on the same
    triangles, with ``seaice_solve_linear_basis_system``'s LU solve with
    partial pivoting (``mpas_seaice_numerics.F``).

It runs once, on the host, in numpy at the run's precision (float32 for
RKIND 4, float64 for RKIND 8), one IEEE operation at a time in the Fortran's
order (gfortran -O0 -ffp-contract=off: left to right, ``x**2`` = ``x*x``,
``DOT_PRODUCT`` and sums from zero), cells with the same number of edges
together, so its words are v8.4.2's on any host
(``tests/test_mpas_seaice_variational_basis_parity.py``, against
``tools/mpas_seaice_oracle/run_variational_basis.F90``). Only + - * / sqrt
and comparisons are used, but for the metric terms' ``asin`` and ``tan``:

- RKIND 4: glibc 2.41 and later round ``asinf`` and ``tanf`` correctly
  (CORE-MATH); so does this port (binary64 first, the few arguments near a
  rounding boundary in 70-digit decimal), so the words are the oracle's;
- RKIND 8: glibc's binary64 ``asin`` and ``tan`` are not correctly rounded
  (glibc 2.44: 24 and 28 of 20,000 random arguments differ from the correctly
  rounded value by an ulp), so no host-independent computation gives the
  oracle's words. Unless every argument is one whose value every libm agrees
  on (zero, or NaN), the metric terms at RKIND 8 are refused by name unless
  ``metric_libm`` chooses: ``"host"``, the host C library's ``asin``/``tan``
  (the oracle's words on a glibc 2.44 host), or ``"correctly-rounded"``.

Outputs follow MPAS's layout (C-order of the Fortran arrays):
``basisGradientU(iBasisVertex, iGradientVertex, iCell)`` is
``[iCell, iGradientVertex, iBasisVertex]``. Only the words v8.4.2 writes are
written: the Wachspress gradients' rows ``1..nEdgesOnCell`` over all
``maxEdges`` columns, the PWL gradients and every integral over
``nEdgesOnCell`` x ``nEdgesOnCell``, cells ``1..nCells``.

Refused by name: an unknown ``config_variational_basis`` (v8.4.2 then
silently leaves the basis unset; QUIRKS.md), an unknown integration type or
an order v8.4.2 has no rule for (it stops), a trapezoidal order below 1
(v8.4.2 divides by zero), and a singular PWL system (v8.4.2 stops).

Opt-in corrections (``corrections``; v8.4.2's words stay the default):

- ``"planar_metric_terms"``: v8.4.2 computes the metric terms on a planar
  mesh too, from the rotated ``z' = xVertex`` under the default rotation
  (non-zero, non-physical) and as ``0/0`` (NaN) where the planar mesh's
  ``sphere_radius`` is 0; corrected, they are zero on a plane;
- ``"periodic_local_coords"``: on a periodic planar mesh v8.4.2 takes
  ``xVertex - xCell`` without unwrapping the period, so the cells across the
  period's seam get a wrong basis; corrected, each offset is wrapped into
  ``(-period/2, period/2]`` as ``mpas_fix_periodicity`` does.
"""
from __future__ import annotations

import decimal
import math
from fractions import Fraction

import numpy as np

from woof.hex.errors import ConfigurationRefusal, MpasPortError

from .remap_init_v842 import literal
from .wachspress_quadrature_v842 import DUNAVANT, FEKETE

SOURCE_FILES = (
    "src/core_seaice/shared/mpas_seaice_velocity_solver_variational.F:46-111",
    "src/core_seaice/shared/mpas_seaice_velocity_solver_variational_shared.F:41-361",
    "src/core_seaice/shared/mpas_seaice_velocity_solver_wachspress.F:46-2066",
    "src/core_seaice/shared/mpas_seaice_velocity_solver_pwl.F:44-485",
    "src/core_seaice/shared/mpas_seaice_numerics.F:44-294",
    "src/core_seaice/shared/mpas_seaice_mesh.F:633-701,1740-1763,2003-2043,2254-2365",
    "src/core_seaice/shared/mpas_seaice_velocity_solver.F:140-175",
)

BASES = ("wachspress", "pwl")
INTEGRATION_TYPES = ("trapezoidal", "dunavant", "fekete")
METRIC_LIBMS = ("host", "correctly-rounded")
CORRECTIONS = ("planar_metric_terms", "periodic_local_coords")
#: the outputs, in the velocity_variational pool
OUTPUTS = ("basisGradientU", "basisGradientV", "basisIntegralsU", "basisIntegralsV", "basisIntegralsMetric",
           "tanLatVertexRotatedOverRadius", "cellVerticesAtVertex")


class BasisError(MpasPortError):
    """Where v8.4.2 stops (MPAS_LOG_CRIT) while setting the basis up."""


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


# --- the metric terms' asin and tan ------------------------------------------

_DEC = decimal.Context(prec=70, rounding=decimal.ROUND_HALF_EVEN, Emax=999999, Emin=-999999)
_EPS = decimal.Decimal(10) ** -75


def _dec_atan(y: decimal.Decimal) -> decimal.Decimal:
    k = 0
    while abs(y) > decimal.Decimal("0.1"):
        # atan(y) = 2 atan(y / (1 + sqrt(1 + y^2)))
        y = _DEC.divide(y, _DEC.add(1, _DEC.sqrt(_DEC.add(1, _DEC.multiply(y, y)))))
        k += 1
    total, term, n, y2 = decimal.Decimal(0), y, 1, _DEC.multiply(y, y)
    while abs(term) > _EPS:
        total = _DEC.add(total, _DEC.divide(term, n))
        term, n = _DEC.minus(_DEC.multiply(term, y2)), n + 2
    return _DEC.multiply(total, 2 ** k)


def _dec_asin(x: float) -> decimal.Decimal:
    d = decimal.Decimal(x)
    return _dec_atan(_DEC.divide(d, _DEC.sqrt(_DEC.subtract(1, _DEC.multiply(d, d)))))


def _dec_tan(x: float) -> decimal.Decimal:
    d = decimal.Decimal(x)
    # |x| < pi/2 here: Taylor series of sin and cos
    x2, s, c = _DEC.multiply(d, d), decimal.Decimal(0), decimal.Decimal(0)
    term, n = d, 1
    while abs(term) > _EPS:
        s = _DEC.add(s, term)
        term, n = _DEC.minus(_DEC.divide(_DEC.multiply(term, x2), (n + 1) * (n + 2))), n + 2
    term, n = decimal.Decimal(1), 0
    while abs(term) > _EPS:
        c = _DEC.add(c, term)
        term, n = _DEC.minus(_DEC.divide(_DEC.multiply(term, x2), (n + 1) * (n + 2))), n + 2
    return _DEC.divide(s, c)


def _round_to(value: decimal.Decimal, real):
    exact = Fraction(value)
    if real is np.float64:
        return np.float64(float(exact))
    guess = np.float32(float(exact))
    candidates = (np.nextafter(guess, np.float32(-np.inf)), guess, np.nextafter(guess, np.float32(np.inf)))
    return np.float32(min(candidates, key=lambda c: (abs(Fraction(float(c)) - exact),
                                                     int(np.asarray(c).view(np.uint32)) & 1)))


def _trivial(x: float) -> bool:
    """An argument whose asin and tan every libm agrees on."""
    return x == 0.0 or math.isnan(x) or math.isinf(x) or abs(x) > 1.0


def _cr(function: str, x, real):
    """The correctly rounded ``asin``/``tan`` of the ``real`` word ``x``."""
    v = float(x)
    if function == "asin":
        if abs(v) > 1.0 or math.isnan(v):
            return real(np.nan)
        if abs(v) == 1.0:
            return _round_to(_DEC.multiply(_dec_atan(decimal.Decimal(1)), 2 if v > 0 else -2), real)
        if v == 0.0:
            return real(v)
        return _round_to(_dec_asin(v), real)
    if math.isnan(v) or math.isinf(v):
        return real(np.nan)
    if v == 0.0:
        return real(v)
    return _round_to(_dec_tan(v), real)


def _cr_binary32(function: str, x: np.ndarray) -> np.ndarray:
    """Correctly rounded binary32 ``asinf``/``tanf`` (glibc >= 2.41's words):
    the binary64 value rounded once, except within a few binary64 ulps of a
    binary32 rounding boundary, where 70-digit decimal decides."""
    x64 = x.astype(np.float64)
    with np.errstate(invalid="ignore"):
        y = np.arcsin(x64) if function == "asin" else np.tan(x64)
    out = y.astype(np.float32)
    up = np.nextafter(out, np.float32(np.inf)).astype(np.float64)
    down = np.nextafter(out, np.float32(-np.inf)).astype(np.float64)
    o64 = out.astype(np.float64)
    margin = 8.0 * np.spacing(np.abs(y))
    with np.errstate(invalid="ignore"):
        near = (np.abs(y - (o64 + up) * 0.5) <= margin) | (np.abs(y - (o64 + down) * 0.5) <= margin)
    for i in np.flatnonzero(near & np.isfinite(y)):
        out[i] = _cr(function, x[i], np.float32)
    return out


def metric_terms(mesh: dict, *, rotate: bool, include: bool, real, metric_libm: str | None,
                 planar_zero: bool = False) -> np.ndarray:
    """``seaice_calc_variational_metric_terms``: ``tanLatVertexRotatedOverRadius``
    over vertices ``1..nVertices``."""
    nv = int(mesh["nVertices"])
    if not include or planar_zero:
        return np.zeros(nv, real)
    radius = real(mesh["sphere_radius"])
    # seaice_grid_rotation_forward: zp = x under the rotation
    z = (mesh["xVertex"] if rotate else mesh["zVertex"])[:nv].astype(real)
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = z / radius
    if real is np.float32:
        lat = _cr_binary32("asin", arg)
        tan = _cr_binary32("tan", lat)
    else:
        nontrivial = [i for i in range(nv) if not _trivial(float(arg[i]))]
        if nontrivial and metric_libm is None:
            _refuse("config_include_metric_terms", True,
                    "at RKIND 8 the metric terms call glibc's binary64 asin and tan, which are not correctly "
                    "rounded, so no host-independent computation gives v8.4.2's words; choose metric_libm "
                    "'host' (the host C library: the oracle's words on a glibc 2.44 host) or "
                    "'correctly-rounded'",
                    "config_include_metric_terms = .false., or metric_libm = 'host' | 'correctly-rounded'")
        if metric_libm not in (None, *METRIC_LIBMS):
            raise ValueError(f"metric_libm is one of {METRIC_LIBMS}, not {metric_libm!r}")
        lat = np.empty(nv, real)
        tan = np.empty(nv, real)
        for i in range(nv):
            v = float(arg[i])
            if metric_libm == "host" and not _trivial(v):
                lat[i] = math.asin(v)
                tan[i] = math.tan(float(lat[i]))
            elif _trivial(v):
                lat[i] = real(np.nan) if (math.isnan(v) or abs(v) > 1.0) else real(v)
                tan[i] = lat[i]
            else:
                lat[i] = _cr("asin", v, real)
                tan[i] = _cr("tan", lat[i], real)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (tan / radius).astype(real)


# --- local coordinates --------------------------------------------------------

def _unit_vectors(x, y, z, real):
    """``local_eastern_and_northern_unit_vectors`` (mpas_seaice_mesh.F:2254-2318)."""
    zero, one = real(0.0), real(1.0)
    e1, e2, e3 = -y, x, np.zeros_like(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.sqrt((e1 * e1 + e2 * e2) + e3 * e3)
        east = (e1 / mag, e2 / mag, e3 / mag)
        nonzero = z != zero
        zs = np.where(nonzero, z, one)
        n1, n2, n3 = -x, -y, (x * x + y * y) / zs
        mag = np.sqrt((n1 * n1 + n2 * n2) + n3 * n3)
        n1, n2, n3 = n1 / mag, n2 / mag, n3 / mag
    south = z < zero
    n1, n2, n3 = np.where(south, -n1, n1), np.where(south, -n2, n2), np.where(south, -n3, n3)
    north = (np.where(nonzero, n1, zero), np.where(nonzero, n2, zero), np.where(nonzero, n3, one))
    return east, north


def local_coords(mesh: dict, *, sphere: bool, rotate: bool, real, periodic: dict | None = None):
    """``seaice_calc_local_coords``: ``xLocal``/``yLocal`` as (nCells, maxEdges)
    (entries past ``nEdgesOnCell`` are zero and never read)."""
    nc, me = int(mesh["nCells"]), int(mesh["maxEdges"])
    ne = mesh["nEdgesOnCell"][:nc].astype(np.int64)
    voc = mesh["verticesOnCell"][:nc, :me].astype(np.int64) - 1
    valid = np.arange(me)[None, :] < ne[:, None]
    iv = np.where(valid, voc, 0)
    xl, yl = np.zeros((nc, me), real), np.zeros((nc, me), real)
    xv, yv = mesh["xVertex"].astype(real), mesh["yVertex"].astype(real)
    xc, yc = mesh["xCell"][:nc].astype(real), mesh["yCell"][:nc].astype(real)
    if not sphere:
        dx, dy = xv[iv] - xc[:, None], yv[iv] - yc[:, None]
        if periodic is not None:
            for d, period in ((dx, periodic.get("x")), (dy, periodic.get("y"))):
                if period:
                    p = real(period)
                    half = real(p * real(0.5))
                    d[...] = np.where(d > half, d - p, np.where(d <= -half, d + p, d))
        xl[valid], yl[valid] = dx[valid], dy[valid]
        return xl, yl
    zv, zc = mesh["zVertex"].astype(real), mesh["zCell"][:nc].astype(real)
    if rotate:
        vx, vy, vz = -zv[iv], yv[iv], xv[iv]
        cx, cy, cz = -zc, yc, xc
    else:
        vx, vy, vz = xv[iv], yv[iv], zv[iv]
        cx, cy, cz = xc, yc, zc
    east, north = _unit_vectors(cx, cy, cz, real)
    e = [a[:, None] for a in east]
    n = [a[:, None] for a in north]
    with np.errstate(invalid="ignore"):
        dx = (vx * e[0] + vy * e[1]) + vz * e[2]
        dy = (vx * n[0] + vy * n[1]) + vz * n[2]
    xl[valid], yl[valid] = dx[valid], dy[valid]
    return xl, yl


def cell_vertices_at_vertex(mesh: dict) -> np.ndarray:
    """``seaice_cell_vertices_at_vertex``: (nVertices, vertexDegree), the last
    matching position (1-based) of the vertex in each surrounding cell, else 0."""
    nv, vd = int(mesh["nVertices"]), int(mesh["vertexDegree"])
    out = np.zeros((nv, vd), np.int32)
    cov = mesh["cellsOnVertex"]
    noc, voc = mesh["nEdgesOnCell"], mesh["verticesOnCell"]
    for v in range(nv):
        for d in range(vd):
            c = int(cov[v, d]) - 1
            for k in range(int(noc[c])):
                if int(voc[c, k]) == v + 1:
                    out[v, d] = k + 1
    return out


# --- Wachspress ---------------------------------------------------------------

def integration_factors(kind: str, order: int, real) -> tuple[np.ndarray, np.ndarray, np.ndarray, object, list]:
    """``get_integration_factors``: (u, v, weights, normalizationFactor,
    warnings v8.4.2 logs)."""
    warnings = []
    if kind == "trapezoidal":
        if order < 1:
            _refuse("config_wachspress_integration_order", order,
                    "the trapezoidal rule needs a positive order (v8.4.2 divides by zero at order 0 and has "
                    "no integration points, so zero integrals, below it)",
                    "config_wachspress_integration_order >= 1")
        u, v, w = [], [], []
        n = real(order)
        for i in range(order + 1):
            for j in range(order - i + 1):
                u.append(real(real(i) / n))
                v.append(real(real(j) / n))
                if i == order or j == order or (i == 0 and j == 0):
                    w.append(real(1.0))
                elif ((j == 0 and i != 0 and i != order) or (i == 0 and j != 0 and j != order)
                      or (i == order - j and i != 0 and j != 0)):
                    w.append(real(3.0))
                else:
                    w.append(real(6.0))
        norm = real(real(6.0) * real(n * n))
        return np.array(u, real), np.array(v, real), np.array(w, real), norm, warnings
    if kind == "dunavant":
        table = DUNAVANT
        if order % 2 != 0:
            warnings.append({"message": "get_integration_factors_dunavant: odd orders of integration not recommended",
                             "ints": [], "reals": []})
    elif kind == "fekete":
        table = FEKETE
    else:
        _refuse("config_wachspress_integration_type", kind,
                "unknown Wachspress integration type (v8.4.2 stops)",
                "config_wachspress_integration_type = 'trapezoidal', 'dunavant' or 'fekete'")
    if order not in table:
        _refuse("config_wachspress_integration_order", order,
                f"no {kind} rule of that order in v8.4.2 (it stops); orders: {sorted(table)}",
                f"config_wachspress_integration_order in {sorted(table)}")
    u, v, w = (np.array([literal(s, real) for s in values], real) for values in table[order])
    return u, v, w, real(2.0), warnings


def _edge(A, B, k, x, y, one):
    """``wachspress_edge_equation`` for edge k at the points."""
    a, b = A[:, k], B[:, k]
    shape = (-1,) + (1,) * (x.ndim - 1)
    return (one - a.reshape(shape) * x) - b.reshape(shape) * y


def _subsets(n: int) -> list[list[int]]:
    """``wachspress_indexes`` (0-based): each vertex's other vertices but the next."""
    return [[k for k in range(n) if k != j and k != (j + 1) % n] for j in range(n)]


def _evaluate(A, B, kappa, x, y, real):
    """The numerators, their sum, the numerators' derivatives and their sum at
    the points (``wachspress_numerator``/``_numerator_derivative``, as
    ``wachspress_basis_function``/``_derivative`` accumulate them)."""
    n = A.shape[1]
    zero, one = real(0.0), real(1.0)
    shape = (-1,) + (1,) * (x.ndim - 1)
    edges = [_edge(A, B, k, x, y, one) for k in range(n)]
    num, der = [], []
    den = np.zeros_like(x)
    sumd = [np.zeros_like(x), np.zeros_like(x)]
    for j, sub in enumerate(_subsets(n)):
        k_j = kappa[:, j].reshape(shape)
        value = np.full_like(x, one)
        for k in sub:
            value = value * edges[k]
        value = value * k_j
        num.append(value)
        den = den + value
        d = []
        for comp, coef in ((0, A), (1, B)):
            total = np.zeros_like(x)
            for kk, k in enumerate(sub):
                product = np.full_like(x, one)
                for l in sub[:kk]:
                    product = product * edges[l]
                product = product * (-coef[:, k]).reshape(shape)
                for l in sub[kk + 1:]:
                    product = product * edges[l]
                total = total + product
            d.append(total * k_j)
            sumd[comp] = sumd[comp] + d[comp]
        der.append(d)
    return num, den, der, sumd


def _coefficients(xl, yl, real):
    """``calc_wachspress_coefficients`` for cells of n edges: A, B and kappa (the
    same for every basis vertex)."""
    n = xl.shape[1]
    A, B = np.empty_like(xl), np.empty_like(xl)
    for i in range(n):
        i1, i2 = (i - 1) % n, i
        A[:, i] = (yl[:, i2] - yl[:, i1]) / (xl[:, i1] * yl[:, i2] - xl[:, i2] * yl[:, i1])
        B[:, i] = (xl[:, i1] - xl[:, i2]) / (xl[:, i1] * yl[:, i2] - xl[:, i2] * yl[:, i1])
    kappa = np.empty_like(xl)
    kappa[:, 0] = real(1.0)
    for j in range(1, n):
        i0, i1, i2 = j - 1, j, (j + 1) % n
        kappa[:, j] = (kappa[:, j - 1] * (A[:, i2] * (xl[:, i0] - xl[:, i1]) + B[:, i2] * (yl[:, i0] - yl[:, i1]))
                       / (A[:, i0] * (xl[:, i1] - xl[:, i0]) + B[:, i0] * (yl[:, i1] - yl[:, i0])))
    return A, B, kappa


def _wachspress_group(xl, yl, factors, real, max_edges: int):
    """The Wachspress gradients (G, maxEdges[iG], n[iB]) and integrals (G, n[iV], n[iS])
    for G cells of n edges."""
    G, n = xl.shape
    A, B, kappa = _coefficients(xl, yl, real)
    # calculate_wachspress_derivatives: at the vertices
    num, den, der, sumd = _evaluate(A, B, kappa, xl, yl, real)
    grad_u = np.zeros((G, max_edges, n), real)
    grad_v = np.zeros((G, max_edges, n), real)
    for ib in range(n):
        q = num[ib] / (den * den)
        du = der[ib][0] / den - q * sumd[0]
        dv = der[ib][1] / den - q * sumd[1]
        for ig in (ib, (ib - 1) % n, (ib + 1) % n):
            grad_u[:, ig, ib] = du[:, ig]
            grad_v[:, ig, ib] = dv[:, ig]
    # integrate_wachspress: over each sub-triangle (centre, vertex s, vertex s+1)
    u, v, w, norm, _ = factors
    one, zero = real(1.0), real(0.0)
    ints = {name: np.zeros((G, n, n), real) for name in ("U", "V", "Metric")}
    for s in range(n):
        i1, i2 = s, (s + 1) % n
        u1, v1, u2, v2 = xl[:, i1], yl[:, i1], xl[:, i2], yl[:, i2]
        # get_triangle_mapping(x1=1, y1=0, x2=0, y2=1, u1, v1, u2, v2)
        # (u2*y1 - u1*y2) / (x2*y1 - x1*y2), (u1*x2 - u2*x1) / (y1*x2 - y2*x1), ...
        m11 = (u2 * zero - u1 * one) / (zero * zero - one * one)
        m12 = (u1 * zero - u2 * one) / (zero * zero - one * one)
        m21 = (v2 * zero - v1 * one) / (zero * zero - one * one)
        m22 = (v1 * zero - v2 * one) / (zero * zero - one * one)
        jac = m11 * m22 - m12 * m21
        px = m11[:, None] * u[None, :] + m12[:, None] * v[None, :]
        py = m21[:, None] * u[None, :] + m22[:, None] * v[None, :]
        num, den, der, sumd = _evaluate(A, B, kappa, px, py, real)
        basis = [num[i] / den for i in range(n)]
        deriv = []
        for i in range(n):
            q = num[i] / (den * den)
            deriv.append((der[i][0] / den - q * sumd[0], der[i][1] / den - q * sumd[1]))
        # (G, n[iS], P) and (G, n[iV], P)
        sb = np.stack(basis, axis=1)
        vb = sb
        du = np.stack([d[0] for d in deriv], axis=1)
        dv = np.stack([d[1] for d in deriv], axis=1)
        sums = {name: np.zeros((G, n, n), real) for name in ints}
        for p in range(u.shape[0]):
            # ((jacobian * weight) * stressBasis) * velocity term; [iV, iS]
            jw = (jac * w[p])[:, None, None]
            js = jw * sb[:, None, :, p]
            sums["U"] = sums["U"] + js * du[:, :, None, p]
            sums["V"] = sums["V"] + js * dv[:, :, None, p]
            sums["Metric"] = sums["Metric"] + js * vb[:, :, None, p]
        for name in ints:
            ints[name] = ints[name] + sums[name] / norm
    return grad_u, grad_v, ints


# --- PWL ----------------------------------------------------------------------

def _maxloc(values: list[np.ndarray]) -> np.ndarray:
    """gfortran MAXLOC (0-based): the first maximum; NaNs never win (all NaN: the first)."""
    best = np.zeros(values[0].shape, np.int64)
    best_value = np.where(np.isnan(values[0]), -np.inf, values[0])
    seen = ~np.isnan(values[0])
    for i, v in enumerate(values[1:], start=1):
        take = ~np.isnan(v) & ((v > best_value) | ~seen)
        best = np.where(take, i, best)
        best_value = np.where(take, v, best_value)
        seen = seen | ~np.isnan(v)
    return best


def solve_linear_basis_system(matrix: np.ndarray, rhs: np.ndarray, real) -> np.ndarray:
    """``seaice_solve_linear_basis_system`` over a batch: ``matrix`` (G, n, n)
    (``[g, row, column]``), ``rhs`` (G, n); ``lu_decomposition`` with implicit
    scaling and partial pivoting, then ``lu_back_subsitution``."""
    a = matrix.astype(real).copy()
    b = rhs.astype(real).copy()
    G, n, _ = a.shape
    zero, one = real(0.0), real(1.0)
    tiny = literal("1.0e-20", real)
    maxa = np.fmax.reduce(np.abs(a), axis=2)  # MAXVAL skips NaNs
    if (maxa == zero).any():
        raise BasisError("lu_decomposition: singular matrix (v8.4.2 stops, MPAS_LOG_CRIT)")
    maxa = one / maxa
    rows = np.arange(G)
    indices = np.zeros((G, n), np.int64)
    for j in range(n):
        jmax = j + _maxloc([maxa[:, i] * np.abs(a[:, i, j]) for i in range(j, n)])
        swap = jmax != j
        row_j, row_m = a[rows, j, :].copy(), a[rows, jmax, :].copy()
        a[rows, j, :] = np.where(swap[:, None], row_m, row_j)
        a[rows, jmax, :] = np.where(swap[:, None], row_j, row_m)
        maxa[rows, jmax] = np.where(swap, maxa[:, j], maxa[rows, jmax])
        indices[:, j] = jmax
        a[:, j, j] = np.where(a[:, j, j] == zero, tiny, a[:, j, j])
        for i in range(j + 1, n):
            a[:, i, j] = a[:, i, j] / a[:, j, j]
        for i in range(j + 1, n):
            for k in range(j + 1, n):
                a[:, i, k] = a[:, i, k] - a[:, i, j] * a[:, j, k]
    first = np.full(G, -1, np.int64)  # lu_back_subsitution's j (0: not yet)
    for i in range(n):
        k = indices[:, i]
        sums = b[rows, k].copy()
        b[rows, k] = b[:, i]
        started = first >= 0
        dot = np.zeros(G, real)
        for m in range(i):
            dot = np.where(started & (m >= first), dot + a[:, i, m] * b[:, m], dot)
        sums = np.where(started, sums - dot, sums)
        first = np.where(~started & (sums != zero), i, first)
        b[:, i] = sums
    for i in range(n - 1, -1, -1):
        dot = np.zeros(G, real)
        for m in range(i + 1, n):
            dot = dot + a[:, i, m] * b[:, m]
        b[:, i] = (b[:, i] - dot) / a[:, i, i]
    return b


def _pwl_group(xl, yl, dv, area, real, max_edges: int):
    """``init_velocity_solver_pwl_basis`` for G cells of n edges: gradients
    (G, n[iG], n[iB]) and integrals (G, n[iV], n[iS])."""
    G, n = xl.shape
    zero, one, half = real(0.0), real(1.0), real(0.5)
    two, three, twelve = real(2.0), real(3.0), real(12.0)
    alpha = real(one / real(n))
    xc, yc = np.zeros(G, real), np.zeros(G, real)
    for i in range(n):
        xc = xc + alpha * xl[:, i]
        yc = yc + alpha * yl[:, i]
    sub = np.zeros((G, n), real)
    total = np.zeros(G, real)
    with np.errstate(invalid="ignore"):
        for s in range(n):
            i1, i2 = s, (s + 1) % n
            c = dv[:, s]
            dx1, dy1 = xl[:, i1] - xc, yl[:, i1] - yc
            dx2, dy2 = xl[:, i2] - xc, yl[:, i2] - yc
            a = np.sqrt(dx1 * dx1 + dy1 * dy1)
            b = np.sqrt(dx2 * dx2 + dy2 * dy2)
            sp = ((a + b) + c) * half
            sub[:, s] = np.sqrt(((sp * (sp - a)) * (sp - b)) * (sp - c))
            total = total + sub[:, s]
    with np.errstate(divide="ignore", invalid="ignore"):
        sub = sub * (area / total)[:, None]
    # the linear basis on each sub-triangle
    g = np.zeros((G, n, 3, 2), real)  # [cell, subcell, basis, (u, v)]
    matrix = np.zeros((G * n, 3, 3), real)
    for s in range(n):
        i1, i2 = s, (s + 1) % n
        block = matrix[s::n]
        block[:, 0, 0], block[:, 0, 1], block[:, 0, 2] = xl[:, i1] - xc, yl[:, i1] - yc, one
        block[:, 1, 0], block[:, 1, 1], block[:, 1, 2] = xl[:, i2] - xc, yl[:, i2] - yc, one
        block[:, 2, 0], block[:, 2, 1], block[:, 2, 2] = zero, zero, one
    for basis in range(2):
        rhs = np.zeros((G * n, 3), real)
        rhs[:, basis] = one
        solution = solve_linear_basis_system(matrix, rhs, real).reshape(G, n, 3)
        g[:, :, basis, 0], g[:, :, basis, 1] = solution[:, :, 0], solution[:, :, 1]
    g[:, :, 2, 0] = -g[:, :, 0, 0] - g[:, :, 1, 0]
    g[:, :, 2, 1] = -g[:, :, 0, 1] - g[:, :, 1, 1]
    # subCellGradient(iBasisVertex, iSubCell)
    scg = np.zeros((G, n, n, 2), real)
    for ib in range(n):
        for s in range(n):
            value = g[:, s, 2, :] * alpha
            if s == ib:
                value = value + g[:, s, 0, :]
            elif s == (ib - 1) % n:
                value = value + g[:, s, 1, :]
            scg[:, ib, s, :] = value
    grad = np.zeros((G, n, n, 2), real)  # [cell, iG, iB, comp]
    for ib in range(n):
        for ig in range(n):
            grad[:, ig, ib, :] = half * (scg[:, ib, ig, :] + scg[:, ib, (ig - 1) % n, :])
    ints = {name: np.zeros((G, n, n), real) for name in ("U", "V", "Metric")}
    a1 = alpha + one
    for i_s in range(n):
        for i_v in range(n):
            for s in range(n):
                if s == i_s or s == (i_s - 1) % n:
                    bi = (a1 * sub[:, s]) / three
                else:
                    bi = (alpha * sub[:, s]) / three
                ints["U"][:, i_v, i_s] = ints["U"][:, i_v, i_s] + scg[:, i_v, s, 0] * bi
                ints["V"][:, i_v, i_s] = ints["V"][:, i_v, i_s] + scg[:, i_v, s, 1] * bi
    aa = alpha * alpha
    values = {(1, 1): two * aa + two * alpha + two, (1, 2): two * aa + two * alpha + one,
              (1, 3): two * aa + alpha, (3, 3): two * aa}

    def kind(s, vertex):
        return 1 if s == vertex else (2 if s == (vertex - 1) % n else 3)

    for i_s in range(n):
        for i_v in range(n):
            for s in range(n):
                ts, tv = kind(s, i_s), kind(s, i_v)
                if (ts, tv) in ((1, 1), (2, 2)):
                    value = values[(1, 1)]
                elif (ts, tv) in ((1, 2), (2, 1)):
                    value = values[(1, 2)]
                elif ts == 3 and tv == 3:
                    value = values[(3, 3)]
                else:
                    value = values[(1, 3)]
                ints["Metric"][:, i_v, i_s] = ints["Metric"][:, i_v, i_s] + (value * sub[:, s]) / twelve
    grad_u = np.zeros((G, max_edges, n), real)
    grad_v = np.zeros((G, max_edges, n), real)
    grad_u[:, :n, :], grad_v[:, :n, :] = grad[..., 0], grad[..., 1]
    return grad_u, grad_v, ints


# --- the init -----------------------------------------------------------------

def check_config(configs: dict) -> None:
    """Refuse, by name, what this port does not set up."""
    basis = str(configs.get("config_variational_basis", "wachspress")).rstrip(" ")
    if basis not in BASES:
        _refuse("config_variational_basis", basis,
                "unknown basis: v8.4.2 sets none up and runs on whatever the basis arrays hold",
                "config_variational_basis = 'wachspress' or 'pwl'")
    if basis == "wachspress":
        kind = str(configs.get("config_wachspress_integration_type", "dunavant")).rstrip(" ")
        integration_factors(kind, int(configs.get("config_wachspress_integration_order", 8)), np.float64)


def initialize(*, mesh: dict, configs: dict, real, out: dict | None = None, metric_libm: str | None = None,
               corrections=()) -> tuple[dict, list[dict]]:
    """``seaice_init_velocity_solver_variational`` on host arrays (numpy,
    C-order with MPAS's extra element, MPAS's names).

    ``mesh``: ``nCells``, ``nVertices``, ``maxEdges``, ``vertexDegree``,
    ``nEdgesOnCell``, ``verticesOnCell``, ``edgesOnCell``, ``cellsOnVertex``,
    ``x/y/zVertex``, ``x/y/zCell``, ``dvEdge``, ``areaCell``. ``configs``:
    ``config_variational_basis``, ``config_wachspress_integration_type``/
    ``_order``, ``config_rotate_cartesian_grid``, ``config_include_metric_terms``,
    ``mesh:on_a_sphere``, ``mesh:sphere_radius`` (and, for the periodic
    correction, ``mesh:is_periodic``, ``mesh:x_period``, ``mesh:y_period``).
    ``out`` holds the outputs (:data:`OUTPUTS`, the velocity_variational
    pool's shapes) to write in place; missing ones are allocated as zeros.
    Returns the outputs and the warnings v8.4.2 logs."""
    check_config(configs)
    unknown = set(corrections) - set(CORRECTIONS)
    if unknown:
        raise ValueError(f"unknown corrections {sorted(unknown)}; known: {CORRECTIONS}")
    nc, nv, me = int(mesh["nCells"]), int(mesh["nVertices"]), int(mesh["maxEdges"])
    vd = int(mesh["vertexDegree"])
    sphere = bool(configs.get("mesh:on_a_sphere", configs.get("on_a_sphere", False)))
    rotate = bool(configs.get("config_rotate_cartesian_grid", True))
    include = bool(configs.get("config_include_metric_terms", True))
    out = {} if out is None else out
    shapes = {name: (nc + 1, me, me) for name in OUTPUTS[:5]}
    shapes.update(tanLatVertexRotatedOverRadius=(nv + 1,), cellVerticesAtVertex=(nv + 1, vd))
    for name, shape in shapes.items():
        if out.get(name) is None:
            out[name] = np.zeros(shape, np.int32 if name == "cellVerticesAtVertex" else real)
        elif tuple(out[name].shape) != shape:
            raise ValueError(f"{name} has shape {out[name].shape}, not {shape}")
    m = dict(mesh)
    m["sphere_radius"] = configs.get("mesh:sphere_radius", configs.get("sphere_radius", 0.0))
    periodic = None
    if "periodic_local_coords" in corrections and not sphere and configs.get("mesh:is_periodic", False):
        periodic = {"x": configs.get("mesh:x_period"), "y": configs.get("mesh:y_period")}
    xl, yl = local_coords(m, sphere=sphere, rotate=rotate, real=real, periodic=periodic)
    out["tanLatVertexRotatedOverRadius"][:nv] = metric_terms(
        m, rotate=rotate, include=include, real=real, metric_libm=metric_libm,
        planar_zero="planar_metric_terms" in corrections and not sphere)
    out["cellVerticesAtVertex"][:nv] = cell_vertices_at_vertex(m)
    basis = str(configs.get("config_variational_basis", "wachspress")).rstrip(" ")
    warnings: list[dict] = []
    factors = None
    if basis == "wachspress":
        factors = integration_factors(str(configs.get("config_wachspress_integration_type", "dunavant")).rstrip(" "),
                                      int(configs.get("config_wachspress_integration_order", 8)), real)
        warnings = factors[4]
    ne = np.asarray(mesh["nEdgesOnCell"][:nc]).astype(np.int64)
    dv = mesh["dvEdge"].astype(real)
    area = mesh["areaCell"][:nc].astype(real)
    eoc = mesh["edgesOnCell"][:nc, :me].astype(np.int64) - 1
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for n in sorted(set(ne.tolist())):
            if n < 1:
                continue
            cells = np.flatnonzero(ne == n)
            if basis == "wachspress":
                gu, gv, ints = _wachspress_group(xl[cells, :n], yl[cells, :n], factors, real, me)
                # every column of the basis vertices' rows
                out["basisGradientU"][cells, :, :n] = gu
                out["basisGradientV"][cells, :, :n] = gv
            else:
                gu, gv, ints = _pwl_group(xl[cells, :n], yl[cells, :n], dv[eoc[cells, :n]], area[cells], real, me)
                out["basisGradientU"][cells, :n, :n] = gu[:, :n, :]
                out["basisGradientV"][cells, :n, :n] = gv[:, :n, :]
            for name in ("U", "V", "Metric"):
                out[f"basisIntegrals{name}"][cells, :n, :n] = ints[name]
    return out, warnings


def initialize_pools(pools: dict, configs: dict, *, metric_libm: str | None = None, corrections=()) -> list[dict]:
    """:func:`initialize` on device pools (cupy arrays, or numpy): the mesh is
    copied to the host, the velocity_variational pool's outputs computed in
    place (the words v8.4.2 does not write keep theirs) and copied back."""
    def host(a):
        if a is None or np.isscalar(a) or isinstance(a, (int, float, np.ndarray)):
            return a
        return a.get()
    mesh = {k: host(v) for k, v in pools["mesh"].items()}
    vv = pools["velocity_variational"]
    out = {name: host(vv.get(name)) for name in OUTPUTS}
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in out.items()}
    real = pools["velocity_solver"]["uVelocity"].dtype.type
    out, warnings = initialize(mesh=mesh, configs=configs, real=real, out=out, metric_libm=metric_libm,
                               corrections=corrections)
    for name in OUTPUTS:
        target = vv.get(name)
        if isinstance(target, np.ndarray):
            target[...] = out[name]
        elif target is None:
            if isinstance(pools["mesh"]["xCell"], np.ndarray):
                vv[name] = out[name]
            else:
                import cupy as cp
                vv[name] = cp.asarray(out[name])
        else:
            target.set(np.ascontiguousarray(out[name]))
    return warnings
