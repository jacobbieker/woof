"""WRF v4.7.1's two-way fire coupling (``ifire = 2``) on a running domain.

The level-set fire (:mod:`woof.fire.levelset`), the fuel physics
(:mod:`woof.fire.wrf_sfire`) and the fuel statics (:mod:`woof.static.fuel`)
are WRF-SFIRE's pieces.  This module is the driver glue WRF v4.7.1 wraps
them in, statement for statement:

* ``fire_driver_em_init`` / ``fire_driver_em_step``
  (``phys/module_fr_fire_driver_wrf.F``) and ``fire_driver_em`` /
  ``fire_driver_phys`` (``phys/module_fr_fire_driver.F``): one fire step
  per atmospheric step (WRF takes no sub-steps), at ``time_start =
  itimestep * dt`` in REAL(4) with ``itimestep`` the domain's 1-based step
  number (``dyn_em/solve_em.F:376`` advances it before the physics);
  ``set_nfuel_cat`` and ``set_fire_params`` at initialization;
  ``init_no_fire`` at ``time_start = 0``; ``fire_ignition_convert``'s
  ignition lines, counted up to the last with a positive radius, in metres
  from the lower-left corner or in longitude/latitude with
  ``unit_fxlong``/``unit_fxlat`` from ``cen_lat``;
* ``interpolate_atm2fire`` (``module_fr_fire_driver.F``): the staggered
  wind at ``fire_wind_height`` above the ground (log-linear from zero at
  the roughness length in the first layer, log-linear between layers
  above), continued one cell past the domain and refined bilinearly onto
  the fire mesh;
* ``fire_tendency`` (``phys/module_fr_fire_atm.F``): the fire's heat and
  moisture fluxes as coupled theta and vapour tendencies, decaying
  exponentially over ``fire_ext_grnd`` (surface fire) and above
  ``fire_crwn_hgt`` over ``fire_ext_crwn`` (crown fire; WRF v4.7.1 has no
  canopy flux yet, so that part is zero), with ``phy_prep``'s ``z_at_w``,
  ``dz8w`` and ``rho``.  WRF adds them to the RK slow tendencies every RK
  stage (``phy_fr_ten``), whatever the PBL scheme, and so does woof
  (:func:`woof.core.dycore` through :meth:`FireRuntime.slow_tendencies`).

and, for the smoke tracer, ``add_fire_tracer_emissions``' source
(``tracer_opt = 3``) as the surface flux the tracer's emission seam takes
(``state.smoke_emis``, kg m-2 s-1; :func:`woof.core.smoke_tracer
.add_surface_source` puts it in the lowest layer).

Every routine is held to WRF's own Fortran -- the pinned files compiled at
-O0 by ``tools/fire_coupling_wrf471_oracle`` and the whole driver run on a
stub ``TYPE(domain)`` -- at max ULP 0 on the host references and on the
device (``tests/test_fire_coupling_wrf471_parity.py``).  The device unit is
``kernels/fire_coupling.cu``, NVRTC with ``-fmad=false --ftz=false`` and
glibc's ``logf``/``expf``.

Deliberate differences, each written down in the oracle's PROVENANCE.md:

* ``fire_boundary_guard``: WRF stops the run when the fire comes within
  that many fire-mesh nodes of the edge; woof lets it leave the mesh
  (the level set's own difference);
* the smoke source is a flux into woof's tracer, which counts mass per kg
  of dry air, not WRF's per-step increment of g/kg of total air, and it
  reaches every interior column -- WRF v4.7.1's loop reaches only the first
  of each tile (``module_fr_fire_atm.F:81``, ``do i=i_st,i_st``);
* ``fire_smoke_factor = "pm25"`` (woof's default) takes the PM2.5 factor
  of each fire cell's fuel (:mod:`woof.smoke.emission`) in place of WRF's
  one ``fire_tracer_smoke``; ``"fire_tracer_smoke"`` is WRF's form.

The state the fire carries lives on the :class:`DomainState` (serialized
for restarts); the runtime -- mesh, fuel parameters, ignitions and the
wind tables, all rebuilt identically from the configuration and statics --
is :class:`FireRuntime` on ``state._fire``.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from woof.fire import levelset as ls

F = np.float32
#: module_model_constants g.
G = F(9.81)
#: cp = 7.*r_d/2. (r_d = 287.) and xlv, as module_model_constants folds them.
CP = F(1004.5)
XLV = F(2.5e6)
#: The modes of the fire's smoke factor (RunConfig.fire_smoke_factor).
SMOKE_FACTORS = ("pm25", "fire_tracer_smoke")
#: The fuel tables set_fire_params may load (woof.fire.wrf_sfire.TABLES).
FUEL_TABLES = ("namelist", "defaults")
#: fire_fuel_read values WRF v4.7.1's set_nfuel_cat takes: -1 and 2 keep the
#: input map, 0 is uniform fire_fuel_cat, 1 is by altitude.
FUEL_READ = (-1, 0, 1, 2)
#: The ignition record's fields, WRF's fire_ignition_*N names in order.
IGNITION_FIELDS = ("start_x", "start_y", "end_x", "end_y", "start_lon",
                   "start_lat", "end_lon", "end_lat", "radius", "ros",
                   "start_time", "end_time")
#: WRF's defaults for one ignition (Registry/registry.fire:152-211).
IGNITION_DEFAULTS = dict.fromkeys(IGNITION_FIELDS, 0.0) | {"ros": 0.01}
#: fire_driver_em's fire_max_ignitions.
MAX_IGNITIONS = 5

MODULE_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
MODULE_KEY = "woof.fire.coupling:fire_coupling"
_SOURCES = ("glibc_flt32.cuh", "fire_coupling.cu")
_BLOCK2 = (32, 8)
_BLOCK1 = 256


def r4(value) -> np.float32:
    """A namelist REAL(4): the decimal rounded once, as gfortran reads it."""
    from woof.fire.wrf_sfire import r4_value

    return r4_value(value)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def enabled(cfg) -> bool:
    """Whether this configuration runs the fire model."""
    return int(getattr(cfg, "ifire", 0) or 0) > 0


@dataclass(frozen=True)
class FireSettings:
    """The &fire values one domain runs with, as WRF holds them (REAL(4))."""

    sr_x: int
    sr_y: int
    fuel_read: int
    fuel_cat: int
    fuel_table: str
    wind_height: np.float32
    ext_grnd: np.float32
    ext_crwn: np.float32
    crwn_hgt: np.float32
    atm_feedback: np.float32
    ignitions: tuple
    smoke_factor: str
    tracer_smoke: np.float32
    sfc_lu_index: int
    options: ls.FireOptions

    @classmethod
    def from_config(cls, cfg) -> "FireSettings":
        g = getattr
        options = ls.FireOptions(
            fire_advection=int(g(cfg, "fire_advection", 1)),
            fire_lsm_reinit=bool(g(cfg, "fire_lsm_reinit", True)),
            fire_lsm_reinit_iter=int(g(cfg, "fire_lsm_reinit_iter", 1)),
            fire_lsm_band_ngp=int(g(cfg, "fire_lsm_band_ngp", 4)),
            fire_viscosity=float(r4(g(cfg, "fire_viscosity", 0.4))),
            fire_viscosity_bg=float(r4(g(cfg, "fire_viscosity_bg", 0.4))),
            fire_viscosity_band=float(r4(g(cfg, "fire_viscosity_band",
                                           0.5))),
            fire_viscosity_ngp=int(g(cfg, "fire_viscosity_ngp", 2)),
            fire_lfn_ext_up=float(r4(g(cfg, "fire_lfn_ext_up", 1.0))),
            fire_slope_factor=float(r4(g(cfg, "fire_slope_factor", 1.0))),
            fire_atm_feedback=float(r4(g(cfg, "fire_atm_feedback", 1.0))))
        return cls(
            sr_x=int(g(cfg, "sr_x", 0)), sr_y=int(g(cfg, "sr_y", 0)),
            fuel_read=int(g(cfg, "fire_fuel_read", -1)),
            fuel_cat=int(g(cfg, "fire_fuel_cat", 1)),
            fuel_table=str(g(cfg, "fire_fuel_table", "namelist")),
            wind_height=r4(g(cfg, "fire_wind_height", 6.096)),
            ext_grnd=r4(g(cfg, "fire_ext_grnd", 50.0)),
            ext_crwn=r4(g(cfg, "fire_ext_crwn", 50.0)),
            crwn_hgt=r4(g(cfg, "fire_crwn_hgt", 15.0)),
            atm_feedback=r4(g(cfg, "fire_atm_feedback", 1.0)),
            ignitions=tuple(tuple(r) for r in g(cfg, "fire_ignitions", ())),
            smoke_factor=str(g(cfg, "fire_smoke_factor", "pm25")),
            tracer_smoke=r4(g(cfg, "fire_tracer_smoke", 0.02)),
            sfc_lu_index=int(g(cfg, "sfc_lu_index", 28)),
            options=options)


def ignition_record(table) -> tuple:
    """One ignition as the config stores it: :data:`IGNITION_FIELDS` order.

    ``table`` is a mapping with any of WRF's per-ignition names (without
    the trailing number) or an already ordered sequence.  Missing keys take
    the Registry defaults.
    """
    if isinstance(table, (list, tuple)):
        values = tuple(float(v) for v in table)
        if len(values) != len(IGNITION_FIELDS):
            raise ValueError(
                f"an ignition record has {len(IGNITION_FIELDS)} values "
                f"{IGNITION_FIELDS}, got {len(values)}")
        return values
    unknown = sorted(set(table) - set(IGNITION_FIELDS))
    if unknown:
        raise ValueError(
            f"unknown ignition keys {unknown}; known: {list(IGNITION_FIELDS)}")
    merged = IGNITION_DEFAULTS | {k: float(v) for k, v in table.items()}
    return tuple(float(merged[k]) for k in IGNITION_FIELDS)


@dataclass(frozen=True)
class IgnitionPlan:
    """``fire_ignition_convert``'s output: the lines and their units."""

    lines: tuple            # levelset.Ignition, WRF's order
    longlat: bool           # fire_ignition_longlat
    unit_x: np.float32      # unit_fxlong (m per coordinate unit)
    unit_y: np.float32      # unit_fxlat


def _libm_cosf():
    lib = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
    fn = lib.cosf
    fn.restype = ctypes.c_float
    fn.argtypes = [ctypes.c_float]
    return fn


def convert_ignitions(records, *, cen_lat: float = 0.0) -> IgnitionPlan:
    """``fire_ignition_convert`` (module_fr_fire_driver.F:1041-1163).

    ``records`` are :func:`ignition_record` tuples, WRF's 1..5 in order.
    Ideal coordinates when the first line's ``start_x``/``start_y`` is not
    zero, longitude/latitude when its ``start_lon``/``start_lat`` is not;
    WRF crashes on both and so does this.  The count is the last line with
    a positive radius; a zero end takes the start, a zero end time the
    start time.  A counted line with no positive radius would ignite
    nothing and is refused; so is a sixth line (WRF reads five).
    """
    records = [ignition_record(r) for r in records]
    if len(records) > MAX_IGNITIONS:
        raise ValueError(f"WRF v4.7.1 reads at most {MAX_IGNITIONS} "
                         f"ignition lines; got {len(records)}")
    if not records:
        return IgnitionPlan((), False, F(1.0), F(1.0))
    rows = [dict(zip(IGNITION_FIELDS, (r4(v) for v in rec)))
            for rec in records]
    first = rows[0]
    ideal = first["start_x"] != 0 or first["start_y"] != 0
    real = first["start_lon"] != 0 or first["start_lat"] != 0
    if ideal and real:
        raise ValueError("Only one of the ideal or real coordinates may be "
                         "given (WRF fire_ignition_convert)")
    count = 0
    for i, row in enumerate(rows, start=1):
        if row["radius"] > 0:
            count = i
    lines = []
    if real:
        lat_ctr = r4(cen_lat)
        pi2 = F(2.0) * F(3.1415926)
        reradius = F(1.0) / F(6370.0e03)
        unit_y = pi2 / (F(360.0) * reradius)
        arg = (lat_ctr * pi2) / F(360.0)
        unit_x = F(_libm_cosf()(float(arg))) * unit_y
    else:
        unit_x = unit_y = F(1.0)
    for row in rows[:count]:
        if real:
            sx, sy = row["start_lon"], row["start_lat"]
            ex, ey = row["end_lon"], row["end_lat"]
        else:
            sx, sy = row["start_x"], row["start_y"]
            ex, ey = row["end_x"], row["end_y"]
        ex = sx if ex == 0 else ex
        ey = sy if ey == 0 else ey
        st, et = row["start_time"], row["end_time"]
        et = st if et == 0 else et
        if not row["radius"] > 0:
            raise ValueError(
                "an ignition line before the last one with a radius has no "
                "positive radius; WRF would ignite only nodes exactly on it")
        lines.append(ls.Ignition(start=(sx, sy), end=(ex, ey), start_time=st,
                                 end_time=et, radius=row["radius"],
                                 ros=row["ros"], unit_x=unit_x,
                                 unit_y=unit_y))
    return IgnitionPlan(tuple(lines), bool(real), F(unit_x), F(unit_y))


# ---------------------------------------------------------------------------
# fuel categories (set_nfuel_cat)
# ---------------------------------------------------------------------------

def nfuel_by_altitude(zsf) -> np.ndarray:
    """set_nfuel_cat with ifuelread = 1 (module_fr_fire_phys.F:1499-1526)."""
    t1 = np.asarray(zsf, dtype=F)
    out = np.zeros(t1.shape, dtype=F)
    rules = ((t1 <= F(1524.0), 3.0),
             ((t1 >= F(1524.0)) & (t1 <= F(2073.0)), 2.0),
             ((t1 >= F(2073.0)) & (t1 <= F(2438.0)), 8.0),
             ((t1 > F(2438.0)) & (t1 <= F(3354.0)), 10.0),
             ((t1 > F(3354.0)) & (t1 <= F(3658.0)), 1.0),
             (t1 > F(3658.0), 14.0))
    done = np.zeros(t1.shape, dtype=bool)
    for mask, cat in rules:            # the if/else-if chain, first wins
        pick = mask & ~done
        out[pick] = F(cat)
        done |= mask
    return out


# ---------------------------------------------------------------------------
# interpolate_atm2fire: the tables and the host reference
# ---------------------------------------------------------------------------

def refinement_axis(n_fine: int, r: int, rip1, lo: int, hi: int):
    """interpolate_2d along one axis: ``(cell, weight)`` per fine node.

    Coarse cells ``lo .. hi - 1`` (WRF's its2 .. ite2-1); cell ``i2``
    starts at fine coordinate ``rio = rip1 + r*(i2 - 1)`` and covers nodes
    ``ceiling(rio) .. floor(rio) + r`` of ``1 .. n_fine``.  A node keeps the
    LAST cell WRF's loop writes it from, with ``tx = (i1 - rio) * (1./r)``.
    """
    rip1 = F(rip1)
    rip2 = F(1.0)
    rx = F(1.0) / F(r)
    cell = np.full(n_fine, -10 ** 6, dtype=np.int32)
    weight = np.zeros(n_fine, dtype=F)
    for i2 in range(lo, hi):
        rio = rip1 + F(r) * (F(i2) - rip2)
        start = max(1, math.ceil(float(rio)))
        end = min(n_fine, math.floor(float(rio)) + r)
        for i1 in range(start, end + 1):
            cell[i1 - 1] = i2
            weight[i1 - 1] = (F(i1) - rio) * rx
    if (cell == -10 ** 6).any():  # pragma: no cover - always covered
        raise AssertionError("a fire node outside every coarse cell")
    return cell, weight


@dataclass(frozen=True)
class WindTables:
    """Per-axis refinement tables, cells as indices into the plane arrays.

    ``ua`` is (ny + 2, nx + 1) and ``va`` (ny + 1, nx + 2) (the layout of
    ``kernels/fire_coupling.cu``).
    """

    ux: np.ndarray
    tux: np.ndarray
    uy: np.ndarray
    tuy: np.ndarray
    vx: np.ndarray
    tvx: np.ndarray
    vy: np.ndarray
    tvy: np.ndarray

    @classmethod
    def build(cls, nx: int, ny: int, ir: int, jr: int) -> "WindTables":
        half = F(0.5)
        # u: rip1 = ifds - 0.5, rjp1 = jfds + (jr-1)*0.5; cells i2 = 1..nx
        # (plane column i2 - 1), j2 = 0..ny (plane row j2).
        ux, tux = refinement_axis(nx * ir, ir, F(1.0) - half, 1, nx + 1)
        uy, tuy = refinement_axis(ny * jr, jr, F(1.0) + F(jr - 1) * half,
                                  0, ny + 1)
        # v: rip1 = ifds + (ir-1)*0.5, rjp1 = jfds - 0.5; cells i2 = 0..nx
        # (plane column i2), j2 = 1..ny (plane row j2 - 1).
        vx, tvx = refinement_axis(nx * ir, ir, F(1.0) + F(ir - 1) * half,
                                  0, nx + 1)
        vy, tvy = refinement_axis(ny * jr, jr, F(1.0) - half, 1, ny + 1)
        return cls(ux - 1, tux, uy, tuy, vx, tvx, vy - 1, tvy)


def _logf(x) -> np.float32:
    from woof.core.noahmp_libm import logf

    return F(logf(F(x)))


def _ex(a, b):
    """continue_at_boundary's EX, bias 0."""
    bias = F(0.0)
    lin = F(2.0) * a - b
    return (F(1.0) - bias) * lin + bias * np.fmax(np.fmax(lin, a), b)


def _column_np(phi_a, phi_b, wind, zr, fwh):
    """One staggered column (``fc_column``), scalar float32."""
    if not fwh > zr:
        return F(0.0)
    logfwh = _logf(fwh)
    alt0 = F(0.5) * (phi_a[0] / G + phi_b[0] / G)
    altk, hlast = alt0, F(0.0)
    nz = wind.shape[0]
    for k in range(nz):
        altk1 = F(0.5) * (phi_a[k + 1] / G + phi_b[k + 1] / G)
        ht = F(0.5) * (altk + altk1) - alt0
        if not ht < fwh:
            loght = _logf(ht)
            if k == 0:
                logz0 = _logf(zr)
                return (wind[0] * (logfwh - logz0)) / (loght - logz0)
            loglast = _logf(hlast)
            return wind[k - 1] + ((wind[k] - wind[k - 1])
                                  * (logfwh - loglast)) / (loght - loglast)
        hlast = ht
        altk = altk1
    return F(wind[nz - 1])


def wind_planes_np(u, v, phi, z0, fwh):
    """``(ua, va)``: the vertical step and the continuation, host float32."""
    u, v, phi, z0 = (np.asarray(a, dtype=F) for a in (u, v, phi, z0))
    fwh = F(fwh)
    nz, ny, nx = u.shape[0], u.shape[1], v.shape[2]
    ua = np.full((ny + 2, nx + 1), np.nan, dtype=F)
    va = np.full((ny + 1, nx + 2), np.nan, dtype=F)
    with np.errstate(all="ignore"):
        for j in range(ny):
            for iu in range(1, nx):
                zr = F(0.5) * (z0[j, iu] + z0[j, iu - 1])
                ua[j + 1, iu] = _column_np(phi[:, j, iu - 1], phi[:, j, iu],
                                           u[:, j, iu], zr, fwh)
        for jv in range(1, ny):
            for i in range(nx):
                zr = F(0.5) * (z0[jv - 1, i] + z0[jv, i])
                va[jv, i + 1] = _column_np(phi[:, jv - 1, i], phi[:, jv, i],
                                           v[:, jv, i], zr, fwh)
        # continue_at_boundary(1, 1, 0.): rows, columns, then corners; each
        # reads interior values only.
        ua[1:ny + 1, 0] = _ex(ua[1:ny + 1, 1], ua[1:ny + 1, 2])
        ua[1:ny + 1, nx] = _ex(ua[1:ny + 1, nx - 1], ua[1:ny + 1, nx - 2])
        ua[0, 1:nx] = _ex(ua[1, 1:nx], ua[2, 1:nx])
        ua[ny + 1, 1:nx] = _ex(ua[ny, 1:nx], ua[ny - 1, 1:nx])
        ua[0, 0] = _ex(ua[1, 1], ua[2, 2])
        ua[ny + 1, 0] = _ex(ua[ny, 1], ua[ny - 1, 2])
        ua[0, nx] = _ex(ua[1, nx - 1], ua[2, nx - 2])
        ua[ny + 1, nx] = _ex(ua[ny, nx - 1], ua[ny - 1, nx - 2])
        va[1:ny, 0] = _ex(va[1:ny, 1], va[1:ny, 2])
        va[1:ny, nx + 1] = _ex(va[1:ny, nx], va[1:ny, nx - 1])
        va[0, 1:nx + 1] = _ex(va[1, 1:nx + 1], va[2, 1:nx + 1])
        va[ny, 1:nx + 1] = _ex(va[ny - 1, 1:nx + 1], va[ny - 2, 1:nx + 1])
        va[0, 0] = _ex(va[1, 1], va[2, 2])
        va[ny, 0] = _ex(va[ny - 1, 1], va[ny - 2, 2])
        va[0, nx + 1] = _ex(va[1, nx], va[2, nx - 1])
        va[ny, nx + 1] = _ex(va[ny - 1, nx], va[ny - 2, nx - 1])
    return ua, va


def refine_np(plane, cx, tx, cy, ty):
    """interpolate_2d at every fine node (``fc_wind_refine``)."""
    plane = np.asarray(plane, dtype=F)
    x = tx[None, :]
    y = ty[:, None]
    r, c = cy[:, None], cx[None, :]
    one = F(1.0)
    with np.errstate(all="ignore"):
        out = (one - x) * (one - y) * plane[r, c]
        out = out + (one - x) * y * plane[r + 1, c]
        out = out + x * (one - y) * plane[r, c + 1]
        out = out + x * y * plane[r + 1, c + 1]
    return out


def interpolate_atm2fire_np(u, v, phi, z0, fwh, tables: WindTables):
    """``(uf, vf, uah, vah)``: the fire-mesh wind and WRF's UAH/VAH."""
    ua, va = wind_planes_np(u, v, phi, z0, fwh)
    uf = refine_np(ua, tables.ux, tables.tux, tables.uy, tables.tuy)
    vf = refine_np(va, tables.vx, tables.tvx, tables.vy, tables.tvy)
    return uf, vf, ua[1:, :], va[:, 1:]


# ---------------------------------------------------------------------------
# fire_tendency, the smoke flux, the active area: host references
# ---------------------------------------------------------------------------

def rho_np(alt, qv):
    """phy_prep's ``rho = 1./alt*(1.+qv)``."""
    alt, qv = np.asarray(alt, dtype=F), np.asarray(qv, dtype=F)
    return (F(1.0) / alt) * (F(1.0) + qv)


def fire_tendency_np(grnhfx, grnqfx, canhfx, canqfx, zs, phi, mu, c1h, c2h,
                     rho, *, alfg, alfc, z1can, msft=None):
    """``(rthfrten, rqvfrten, rth_slow)``, fire_sfc_flx = 0 (``fc_tendency``).

    ``phi`` is ``phb + ph`` on the (nz + 1) w levels; ``rho`` phy_prep's
    density on the nz mass levels.  ``rth_slow`` is ``rthfrten / msft``.
    """
    from woof.fire.levelset import _expf

    grnhfx, grnqfx, canhfx, canqfx, zs, mu = (
        np.asarray(a, dtype=F) for a in (grnhfx, grnqfx, canhfx, canqfx, zs,
                                         mu))
    phi, rho = np.asarray(phi, dtype=F), np.asarray(rho, dtype=F)
    c1h, c2h = np.asarray(c1h, dtype=F), np.asarray(c2h, dtype=F)
    nz = rho.shape[0]
    ny, nx = zs.shape
    alfg, alfc, z1can = F(alfg), F(alfc), F(z1can)
    cp_i = F(1.0) / CP
    xlv_i = F(1.0) / XLV
    alfg_i = F(1.0) / alfg
    alfc_i = F(1.0) / alfc
    rth = np.zeros((nz, ny, nx), dtype=F)
    rqv = np.zeros((nz, ny, nx), dtype=F)
    inner = (slice(1, ny - 1), slice(1, nx - 1))
    with np.errstate(all="ignore"):
        z = phi[:, 1:ny - 1, 1:nx - 1] / G
        z_w = z[:nz] - zs[inner]
        below = z_w < z1can
        eg = _expf(-(alfg_i * z_w))
        ec = _expf(-(alfc_i * (z_w - z1can)))
        hfx = (cp_i * eg) * grnhfx[inner] + np.where(
            below, cp_i, cp_i * ec) * canhfx[inner]
        qfx = (xlv_i * eg) * grnqfx[inner] + np.where(
            below, xlv_i, xlv_i * ec) * canqfx[inner]
        dz8w = z[1:nz] - z[:nz - 1]
        rho_i = F(1.0) / rho[:nz - 1, 1:ny - 1, 1:nx - 1]
        chm = (c1h[:nz - 1, None, None] * mu[inner][None]
               + c2h[:nz - 1, None, None])
        rth[:nz - 1, 1:ny - 1, 1:nx - 1] = -((chm * rho_i
                                               * (hfx[1:] - hfx[:-1])) / dz8w)
        rqv[:nz - 1, 1:ny - 1, 1:nx - 1] = -((chm * rho_i
                                               * (qfx[1:] - qfx[:-1])) / dz8w)
        slow = rth if msft is None else rth / np.asarray(msft, dtype=F)[None]
    slow = np.where(np.broadcast_to(_interior(ny, nx), slow.shape), slow,
                    F(0.0)).astype(F)
    return rth, rqv, slow


def _interior(ny, nx):
    m = np.zeros((ny, nx), dtype=bool)
    m[1:ny - 1, 1:nx - 1] = True
    return m


def smoke_flux_np(burnt, fgip, factor, dt, *, ir, jr, ny, nx):
    """The fire's smoke flux (kg m-2 s-1, ``fc_smoke_flux``)."""
    burnt, fgip = np.asarray(burnt, dtype=F), np.asarray(fgip, dtype=F)
    field = np.ndim(factor) > 0
    fts = np.asarray(factor, dtype=F) if field else F(factor)
    avgw = F(1.0) / F(ir * jr)
    acc = np.zeros((ny - 2, nx - 2), dtype=F)
    for joff in range(jr):
        for ioff in range(ir):
            cells = (slice(jr + joff, (ny - 1) * jr, jr),
                     slice(ir + ioff, (nx - 1) * ir, ir))
            e = avgw * (fts[cells] if field else fts)
            e = e * burnt[cells]
            e = e * fgip[cells]
            acc = acc + e
    out = np.zeros((ny, nx), dtype=F)
    out[1:ny - 1, 1:nx - 1] = acc / F(dt)
    return out


def active_area_np(fire_area, tign_m, fuel_time, time_now, cell, *, ir, jr,
                   ny, nx):
    """The actively burning area of each atmospheric cell, m2."""
    fire_area = np.asarray(fire_area, dtype=F)
    tign = np.asarray(tign_m, dtype=F)[1:-1, 1:-1]
    active = (fire_area > 0) & (F(time_now) - tign < np.asarray(fuel_time,
                                                                dtype=F))
    a = np.where(active, fire_area, F(0.0)).reshape(ny, jr, nx, ir)
    acc = np.zeros((ny, nx), dtype=F)
    for joff in range(jr):
        for ioff in range(ir):
            acc = acc + a[:, joff, :, ioff]
    return acc * F(cell)


# ---------------------------------------------------------------------------
# the device unit
# ---------------------------------------------------------------------------

def module_source() -> str:
    """The exact source string NVRTC compiles."""
    from woof.core.kernels import _preamble

    root = Path(__file__).resolve().parents[1] / "core" / "kernels"
    return _preamble() + "".join(
        (root / name).read_text(encoding="utf-8") for name in _SOURCES)


@lru_cache(maxsize=None)
def _module(device: int):
    """NVRTC directly, -fmad=false and --ftz=false (woof.fire.levelset)."""
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler

    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile

    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(
            source, MODULE_OPTIONS, None, "fire_coupling.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS,
                  module=None)
    return module


@lru_cache(maxsize=None)
def _function(device: int, name: str):
    return _module(device).get_function(name)


def _kernel(name: str):
    import cupy as cp

    return _function(cp.cuda.Device().id, name)


def _grid2(nx: int, ny: int):
    bx, by = _BLOCK2
    return ((max(nx, 1) + bx - 1) // bx, (max(ny, 1) + by - 1) // by), _BLOCK2


def _grid1(n: int):
    return ((max(n, 1) + _BLOCK1 - 1) // _BLOCK1,), (_BLOCK1,)


@dataclass(frozen=True)
class DeviceTables:
    """:class:`WindTables` on the device."""

    ux: object
    tux: object
    uy: object
    tuy: object
    vx: object
    tvx: object
    vy: object
    tvy: object

    @classmethod
    def upload(cls, tables: WindTables) -> "DeviceTables":
        import cupy as cp

        return cls(**{name: cp.asarray(getattr(tables, name))
                      for name in ("ux", "tux", "uy", "tuy", "vx", "tvx",
                                   "vy", "tvy")})


def wind_planes_gpu(u, v, phi, z0, fwh, ua, va):
    """``fc_wind_u``/``fc_wind_v`` then ``fc_wind_extend``, into ua/va."""
    i32 = np.int32
    nz, ny = u.shape[0], u.shape[1]
    nx = v.shape[2]
    grid, block = _grid2(nx - 1, ny)
    _kernel("fc_wind_u")(grid, block, (u, phi, z0, ua, F(fwh), i32(nz),
                                       i32(ny), i32(nx)))
    grid, block = _grid2(nx, ny - 1)
    _kernel("fc_wind_v")(grid, block, (v, phi, z0, va, F(fwh), i32(nz),
                                       i32(ny), i32(nx)))
    n = ny + (nx - 1) + 1 + (ny - 1) + nx + 1
    grid, block = _grid1(n)
    _kernel("fc_wind_extend")(grid, block, (ua, va, i32(ny), i32(nx)))


def refine_gpu(plane, cx, tx, cy, ty, out):
    """``fc_wind_refine`` into ``out`` (nyf, nxf)."""
    nyf, nxf = out.shape
    grid, block = _grid2(nxf, nyf)
    _kernel("fc_wind_refine")(grid, block, (
        plane, np.int32(plane.shape[1]), cx, tx, cy, ty, out,
        np.int32(nxf), np.int32(nyf)))


def rho_gpu(alt, qv, out):
    n = out.size
    grid, block = _grid1(n)
    _kernel("fc_rho")(grid, block, (alt, qv, out, np.int64(n)))


def fire_tendency_gpu(grnhfx, grnqfx, canhfx, canqfx, zs, phi, mu, c1h, c2h,
                      rho, *, alfg, alfc, z1can, msft, rth, rqv, rth_slow):
    nz, ny, nx = rth.shape
    grid, block = _grid2(nx, ny)
    _kernel("fc_tendency")(grid, block, (
        grnhfx, grnqfx, canhfx, canqfx, zs, phi, mu, c1h, c2h, rho,
        np.uintp(0) if msft is None else msft, F(alfg), F(alfc), F(z1can),
        rth, rqv, rth_slow, np.int32(nz), np.int32(ny), np.int32(nx)))


def smoke_flux_gpu(burnt, fgip, factor, scalar, dt, out, *, ir, jr):
    ny, nx = out.shape
    grid, block = _grid2(nx, ny)
    _kernel("fc_smoke_flux")(grid, block, (
        burnt, fgip, np.uintp(0) if factor is None else factor, F(scalar),
        F(1.0) / F(ir * jr), F(dt), out, np.int32(ir), np.int32(jr),
        np.int32(ny), np.int32(nx)))


def active_area_gpu(fire_area, tign_m, fuel_time, time_now, cell, out, *, ir,
                    jr):
    ny, nx = out.shape
    grid, block = _grid2(nx, ny)
    _kernel("fc_active_area")(grid, block, (
        fire_area, tign_m, fuel_time, F(time_now), F(cell), out,
        np.int32(ir), np.int32(jr), np.int32(ny), np.int32(nx)))


# ---------------------------------------------------------------------------
# the runtime
# ---------------------------------------------------------------------------

#: DomainState attributes the fire carries, all serialized: the prognostic
#: state (lfn and tign with the level set's one-node halo, fuel_frac) and
#: the last step's outputs, so a restored state publishes the frame the
#: live one would.  Fire mesh unless named in :data:`ATMOSPHERE_ATTRS`.
STATE_ATTRS = ("fire_lfn", "fire_tign", "fire_fuel_frac", "fire_ros",
               "fire_area_frac", "fire_burnt", "fire_fgrnhfx",
               "fire_fgrnqfx", "fire_uf", "fire_vf", "fire_grnhfx",
               "fire_grnqfx", "fire_canhfx", "fire_canqfx",
               "fire_avg_fuel_frac", "fire_heat_flux", "fire_area")
#: The (ny, nx) members of :data:`STATE_ATTRS`.
ATMOSPHERE_ATTRS = ("fire_grnhfx", "fire_grnqfx", "fire_canhfx",
                    "fire_canqfx", "fire_avg_fuel_frac", "fire_heat_flux",
                    "fire_area")
#: The level-set arrays, (nyf + 2, nxf + 2).
PADDED_ATTRS = ("fire_lfn", "fire_tign")
#: Rebuilt at every fire step before any reader: WRF's RTHFRTEN/RQVFRTEN
#: (coupled) and the theta tendency over the map factor the RK stages add.
REBUILT_ATTRS = ("fire_rthfrten", "fire_rqvfrten", "fire_rth_slow")


def state_shapes(cfg) -> dict[str, tuple[int, ...]]:
    """The fire's DomainState arrays for ``cfg`` (empty when off)."""
    if not enabled(cfg):
        return {}
    nx, ny, nz = int(cfg.nx), int(cfg.ny), int(cfg.nz)
    nxf, nyf = nx * int(cfg.sr_x), ny * int(cfg.sr_y)
    shapes = {}
    for name in STATE_ATTRS:
        if name in ATMOSPHERE_ATTRS:
            shapes[name] = (ny, nx)
        elif name in PADDED_ATTRS:
            shapes[name] = (nyf + 2, nxf + 2)
        else:
            shapes[name] = (nyf, nxf)
    for name in REBUILT_ATTRS:
        shapes[name] = (nz, ny, nx)
    return shapes


def allocate_state(state, cfg, zeros) -> None:
    """Allocate the fire's arrays on ``state`` (DomainState.__init__).

    The level set starts as ``init_no_fire`` at time 0 leaves it
    (:func:`woof.fire.levelset.initial_state`): the runtime's attach
    writes it, so a cold start and a restore both find every array here.
    """
    for name, shape in state_shapes(cfg).items():
        setattr(state, name, zeros(*shape))


class FireRuntime:
    """One domain's fire: fixed inputs on the device and the step.

    Built by :func:`attach`.  Holds nothing a restart needs: the mesh, the
    fuel parameters (set_fire_params of the categories), the ignitions, the
    wind tables, the PM2.5 factor field and the node coordinates are all
    functions of the configuration and the statics.
    """

    def __init__(self, cfg, *, nx, ny, nz, dx, dy, nfuel_cat, dzdxf, dzdyf,
                 coords=None, cen_lat=0.0, z0_fallback=None, xp=None):
        from woof.fire import wrf_sfire

        if xp is None:
            import cupy as xp
        cp = xp
        self.xp = xp
        s = FireSettings.from_config(cfg)
        self.settings = s
        self.mesh = ls.FireMesh(nx, ny, dx, dy, s.sr_x, s.sr_y)
        self.nz = int(nz)
        nfuel = np.asarray(nfuel_cat, dtype=F)
        wp = wrf_sfire.set_fire_params(nfuel, None, table=s.fuel_table)
        fuelmc = wrf_sfire._scalars()["fuelmc_g"]
        no_fuel = np.asarray(wp.fgip) == 0
        fmc_g = np.where(no_fuel, F(0.0), fuelmc).astype(F)
        self.nfuel_cat = nfuel
        self.params = ls.FireParams.build(
            self.mesh, xp=cp, vx=0.0, vy=0.0, dzdxf=np.asarray(dzdxf, F),
            dzdyf=np.asarray(dzdyf, F), r_0=wp.r_0, bbb=wp.bbb,
            phiwc=wp.phiwc, betafl=wp.betafl, fgip=wp.fgip,
            fuel_time=wp.fuel_time, fmc_g=fmc_g, ischap=wp.ischap)
        plan = convert_ignitions(s.ignitions, cen_lat=cen_lat)
        self.plan = plan
        if plan.longlat and coords is None:
            raise ValueError(
                "fire ignitions in longitude/latitude need the domain's "
                "XLAT_M/XLONG_M statics (WRF interpolates them to the fire "
                "mesh); this route hands none.  Give the ignitions as "
                "start_x/start_y in m from the lower-left corner.")
        self.coords = (None if not plan.longlat else
                       tuple(cp.asarray(np.ascontiguousarray(c, dtype=F))
                             for c in coords))
        tables = WindTables.build(nx, ny, s.sr_x, s.sr_y)
        self.tables = tables if xp is np else DeviceTables.upload(tables)
        self.factor = None
        if s.smoke_factor == "pm25":
            from woof.smoke.emission import emission_factor_field_np
            self.factor = cp.asarray(np.ascontiguousarray(
                emission_factor_field_np(nfuel.astype(np.int64)), dtype=F))
        self.z0_fallback = (None if z0_fallback is None else
                            cp.asarray(np.broadcast_to(
                                np.asarray(z0_fallback, F), (ny, nx)).copy()))
        self.ua = cp.zeros((ny + 2, nx + 1), dtype=F)
        self.va = cp.zeros((ny + 1, nx + 2), dtype=F)
        self.phi = cp.empty((nz + 1, ny, nx), dtype=F)
        self.rho = cp.empty((nz, ny, nx), dtype=F)

    # -- the step ---------------------------------------------------------
    def time_start(self, itimestep: int, dt) -> np.float32:
        """``time_start = itimestep * dt`` in REAL(4)."""
        return F(int(itimestep)) * F(dt)

    def step(self, state, itimestep: int, dt, *, z0, smoke_emis=None,
             rho=None):
        """fire_driver_em_step for one atmospheric step.

        Passes 3-6 (winds, fire step, fluxes), fire_tendency into
        ``state.fire_rthfrten``/``fire_rqvfrten``/``fire_rth_slow``, the
        smoke flux into ``smoke_emis`` when given, and the seams
        ``state.fire_heat_flux`` and ``state.fire_area``.  On the device it
        queues kernels only: nothing is read back.  ``rho`` overrides
        phy_prep's density from ``state.alt``/``state.qv`` (the oracle
        replay hands WRF's own).
        """
        cp = xp = self.xp
        host = xp is np
        s, mesh = self.settings, self.mesh
        if z0 is None:
            z0 = self.z0_fallback
        phb = state.phb
        if phb.ndim == 1:
            xp.add(phb[:, None, None], state.php, out=self.phi)
        else:
            xp.add(phb, state.php, out=self.phi)
        # pass 3: interpolate_atm2fire
        t = self.tables
        if host:
            ua, va = wind_planes_np(state.u, state.v, self.phi, z0,
                                    s.wind_height)
            self.ua[...] = ua
            self.va[...] = va
            state.fire_uf[...] = refine_np(ua, t.ux, t.tux, t.uy, t.tuy)
            state.fire_vf[...] = refine_np(va, t.vx, t.tvx, t.vy, t.tvy)
        else:
            wind_planes_gpu(state.u, state.v, self.phi, z0, s.wind_height,
                            self.ua, self.va)
            refine_gpu(self.ua, t.ux, t.tux, t.uy, t.tuy, state.fire_uf)
            refine_gpu(self.va, t.vx, t.tvx, t.vy, t.tvy, state.fire_vf)
        # passes 4-6: the level set, ignitions, fuel and heat
        params = replace(self.params, vx=state.fire_uf, vy=state.fire_vf)
        ts = self.time_start(itimestep, dt)
        fstate = ls.FireState(time=ts, lfn_m=state.fire_lfn,
                              tign_m=state.fire_tign,
                              fuel_frac=state.fire_fuel_frac)
        new, out = ls.step(fstate, mesh, params, F(dt),
                           ignitions=self.plan.lines, options=s.options,
                           coords=self.coords)
        cp.copyto(state.fire_lfn, new.lfn_m)
        cp.copyto(state.fire_tign, new.tign_m)
        cp.copyto(state.fire_fuel_frac, new.fuel_frac)
        for name, value in (("fire_ros", out.ros),
                            ("fire_area_frac", out.fire_area),
                            ("fire_burnt", out.fuel_frac_burnt),
                            ("fire_fgrnhfx", out.fgrnhfx),
                            ("fire_fgrnqfx", out.fgrnqfx),
                            ("fire_grnhfx", out.GRNHFX),
                            ("fire_grnqfx", out.GRNQFX),
                            ("fire_canhfx", out.CANHFX),
                            ("fire_canqfx", out.CANQFX),
                            ("fire_avg_fuel_frac", out.AVG_FUEL_FRAC)):
            cp.copyto(getattr(state, name), value)
        # fire_tendency
        if rho is None:
            if host:
                self.rho[...] = rho_np(state.alt, state.qv)
            else:
                rho_gpu(state.alt, state.qv, self.rho)
            rho = self.rho
        msft = state.msft if getattr(state, "has_msf", False) else None
        tend_args = (state.fire_grnhfx, state.fire_grnqfx, state.fire_canhfx,
                     state.fire_canqfx, state.ht, self.phi, state.total_mu(),
                     state.c1h, state.c2h, rho)
        if host:
            rth, rqv, slow = fire_tendency_np(
                *tend_args, alfg=s.ext_grnd, alfc=s.ext_crwn,
                z1can=s.crwn_hgt, msft=msft)
            state.fire_rthfrten[...] = rth
            state.fire_rqvfrten[...] = rqv
            state.fire_rth_slow[...] = slow
        else:
            fire_tendency_gpu(
                *tend_args, alfg=s.ext_grnd, alfc=s.ext_crwn,
                z1can=s.crwn_hgt, msft=msft, rth=state.fire_rthfrten,
                rqv=state.fire_rqvfrten, rth_slow=state.fire_rth_slow)
        # the seams: the heat into the atmosphere and the burning area
        xp.add(state.fire_grnhfx, state.fire_canhfx, out=state.fire_heat_flux)
        end = ts + F(dt)
        cell = F(mesh.dx) * F(mesh.dy)
        ny, nx = mesh.atm_shape
        if host:
            state.fire_area[...] = active_area_np(
                state.fire_area_frac, state.fire_tign, params.fuel_time, end,
                cell, ir=mesh.sr_x, jr=mesh.sr_y, ny=ny, nx=nx)
        else:
            active_area_gpu(state.fire_area_frac, state.fire_tign,
                            params.fuel_time, end, cell, state.fire_area,
                            ir=mesh.sr_x, jr=mesh.sr_y)
        if smoke_emis is not None:
            factor = s.tracer_smoke if self.factor is None else self.factor
            if host:
                smoke_emis[...] = smoke_flux_np(
                    state.fire_burnt, params.fgip, factor, F(dt),
                    ir=mesh.sr_x, jr=mesh.sr_y, ny=ny, nx=nx)
            else:
                smoke_flux_gpu(state.fire_burnt, params.fgip, self.factor,
                               s.tracer_smoke, F(dt), smoke_emis,
                               ir=mesh.sr_x, jr=mesh.sr_y)


def itimestep_of(state, cfg) -> int:
    """The domain's 1-based WRF step number at the start of this step.

    The physics driver's own formula (woof.core.physics), from the model
    clock: steps since the domain started, plus one.
    """
    now = float(getattr(state, "elapsed_seconds", 0.0))
    offset = float(getattr(state, "domain_start_offset", 0.0) or 0.0)
    return int(math.floor((now - offset) / float(cfg.dt) + 0.5)) + 1


def z0_from_landuse(sfc_lu_index: int) -> np.float32:
    """WRF's em_fire roughness without a physics driver: USGS SFZ0/100.

    ``landuse_init`` (module_physics_init.F:1968) sets ``z0 =
    SFZ0(lu_index, season)/100`` from LANDUSE.TBL whatever the physics; an
    em_fire domain's ``lu_index`` is ``sfc_lu_index`` (28 by default) and
    its season is summer at ``cen_lat = 0``.
    """
    from woof.core.landuse import LANDUSE_COLUMNS, load_landuse_table

    table = load_landuse_table("USGS")
    index = int(sfc_lu_index)
    if not 1 <= index <= table.lucats:
        raise ValueError(f"sfc_lu_index = {index} is not a USGS land-use "
                         f"category (1-{table.lucats})")
    value = table.values[0, index - 1, LANDUSE_COLUMNS.index("z0_cm")]
    return F(r4(value) / F(100.0))


def attach(state, cfg, *, static=None, grid=None):
    """Build ``state._fire`` and start the level set (``fire_driver_em_init``).

    ``static`` is the domain's prepared static field set (the fuel statics
    of :mod:`woof.static.fuel`, and XLAT_M/XLONG_M for longitude/latitude
    ignitions); ``grid`` gives ``cen_lat``.  Without statics the fire-mesh
    terrain and slope are refined from the state's own terrain with the
    statics' own functions, and the fuel must come from ``fire_fuel_read``
    0 or 1.  On a resumed state the restore overwrites the level set
    afterwards, as WRF's restart skips init_no_fire.
    """
    import cupy as cp

    from woof.static import fuel as fuel_statics

    s = FireSettings.from_config(cfg)
    nx, ny = int(cfg.nx), int(cfg.ny)
    nxf, nyf = nx * s.sr_x, ny * s.sr_y

    def static_field(name):
        if static is None:
            return None
        value = static.get(name) if hasattr(static, "get") else None
        if value is None:
            return None
        value = value.get() if hasattr(value, "get") and not isinstance(
            value, np.ndarray) else value
        return np.asarray(value)

    zsf = static_field("ZSF")
    if zsf is not None and zsf.shape == (nyf, nxf):
        dzdxf = static_field("DZDXF")
        dzdyf = static_field("DZDYF")
    else:
        ht = np.asarray(state.ht.get() if hasattr(state.ht, "get")
                        else state.ht, dtype=np.float64)
        zsf = fuel_statics.refine_zsf(ht, s.sr_x, s.sr_y)
        dzdxf, dzdyf = fuel_statics.fire_mesh_slope(
            zsf, fuel_statics.fire_mesh_spacing(cfg.dx, s.sr_x),
            fuel_statics.fire_mesh_spacing(cfg.dy, s.sr_y))
    if s.fuel_read in (-1, 2):
        cats = static_field("NFUEL_CAT_FIRE")
        if cats is None or cats.shape != (nyf, nxf):
            raise ValueError(
                f"fire_fuel_read = {s.fuel_read} on grid_id = "
                f"{getattr(cfg, 'grid_id', 1)} reads the fuel map from the "
                "domain's statics (NFUEL_CAT_FIRE, woof.static.fuel), and "
                "this route hands none at sr_x = "
                f"{s.sr_x}, sr_y = {s.sr_y}.  Prepare the domain with "
                "ifire set (it requests the fuel statics), or use "
                "fire_fuel_read = 0 with fire_fuel_cat.")
        nfuel = cats.astype(F)
    elif s.fuel_read == 0:
        nfuel = np.full((nyf, nxf), F(s.fuel_cat), dtype=F)
    else:
        nfuel = nfuel_by_altitude(zsf)
    coords = None
    lat = static_field("XLAT_M")
    lon = static_field("XLONG_M")
    if (lat is None or lon is None) and grid is not None \
            and hasattr(grid, "latlon_mass"):
        # The mass-point coordinates geogrid writes as XLAT_M/XLONG_M
        # (woof's module_llxy transcription), when the statics omit them.
        lat, lon = (np.asarray(a) for a in grid.latlon_mass())
    if lat is not None and lon is not None and lat.shape == (ny, nx):
        coords = (fuel_statics.refine_zsf(lon, s.sr_x, s.sr_y),
                  fuel_statics.refine_zsf(lat, s.sr_x, s.sr_y))
    cen_lat = float(getattr(grid, "cen_lat", 0.0) or 0.0)
    physics = getattr(state, "physics", None)
    z0_fallback = None
    if physics is None or "z0" not in getattr(physics, "fields", {}):
        z0_fallback = z0_from_landuse(s.sfc_lu_index)
    runtime = FireRuntime(cfg, nx=nx, ny=ny, nz=int(cfg.nz), dx=cfg.dx,
                          dy=cfg.dy, nfuel_cat=nfuel, dzdxf=dzdxf,
                          dzdyf=dzdyf, coords=coords, cen_lat=cen_lat,
                          z0_fallback=z0_fallback)
    # init_no_fire only on a level set nothing has written: a freshly
    # allocated state holds zeros (init_no_fire's level set is positive
    # everywhere), and a restored one -- resumed on a route that attaches
    # after the restore, or lazily at its first step -- keeps its fire, as
    # WRF's restart skips init_no_fire.  One read, at attach.
    if not bool(cp.any(state.fire_lfn != 0)):
        init = ls.initial_state(runtime.mesh, time=0.0, xp=cp)
        cp.copyto(state.fire_lfn, init.lfn_m)
        cp.copyto(state.fire_tign, init.tign_m)
        cp.copyto(state.fire_fuel_frac, init.fuel_frac)
    state._fire = runtime
    return runtime


def step_domain(state, cfg) -> None:
    """The fire's part of one atmospheric step (woof.core.dycore).

    Attaches the runtime from the state's own terrain on first use when no
    route attached one with statics, then runs :meth:`FireRuntime.step`.
    """
    runtime = getattr(state, "_fire", None)
    if runtime is None:
        runtime = attach(state, cfg)
    physics = getattr(state, "physics", None)
    fields = getattr(physics, "fields", None) or {}
    z0 = fields.get("z0")
    smoke = getattr(state, "smoke_emis", None)
    runtime.step(state, itimestep_of(state, cfg), F(cfg.dt), z0=z0,
                 smoke_emis=smoke)


def merge_tendencies(state, physics_tendencies):
    """This step's physics tendencies with the fire's added (phy_fr_ten).

    WRF's update_phy_ten adds rthfrten to rt_tendf and rqvfrten to the
    vapour tendency after every other physics term (add_a2a), and the RK
    stages consume the sum.  Here the fire's coupled theta tendency over
    the map factor and its coupled vapour tendency are added to the
    driver's held ones into NEW arrays -- the driver's own, which a PBL
    slot may hold across steps, are never written -- and with no physics
    at all they are the whole tendency.
    """
    import dataclasses

    from woof.core.physics import PhysicsTendencies

    runtime = state._fire
    if physics_tendencies is None:
        # The fire alone: zero momentum and condensate terms, allocated
        # once per runtime, and the fire's own two arrays.
        zeros = getattr(runtime, "_alone", None)
        if zeros is None:
            zeros = runtime._alone = PhysicsTendencies.zeros(state)
        zeros.rtheta = state.fire_rth_slow
        zeros.rqv = state.fire_rqvfrten
        return zeros
    xp = runtime.xp
    sums = getattr(runtime, "_sums", None)
    if sums is None or sums[0].shape != physics_tendencies.rtheta.shape:
        sums = runtime._sums = (xp.empty_like(physics_tendencies.rtheta),
                                xp.empty_like(physics_tendencies.rqv))
    xp.add(physics_tendencies.rtheta, state.fire_rth_slow, out=sums[0])
    xp.add(physics_tendencies.rqv, state.fire_rqvfrten, out=sums[1])
    merged = dataclasses.replace(physics_tendencies, rtheta=sums[0],
                                 rqv=sums[1])
    extra = getattr(physics_tendencies, "extra_scalars", None)
    if extra is not None:
        merged.extra_scalars = extra
    return merged


__all__ = ["FireRuntime", "FireSettings", "IGNITION_FIELDS", "IgnitionPlan",
           "STATE_ATTRS", "REBUILT_ATTRS", "WindTables", "attach",
           "allocate_state", "convert_ignitions", "merge_tendencies",
           "enabled", "fire_tendency_np", "ignition_record",
           "interpolate_atm2fire_np", "itimestep_of", "nfuel_by_altitude",
           "refinement_axis", "smoke_flux_np", "state_shapes", "step_domain"]
