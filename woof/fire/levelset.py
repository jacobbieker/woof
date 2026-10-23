"""Level-set fire spread on a refined fire mesh: WRF v4.7.1's fire model.

This is WRF v4.7.1's own fire spread (the WRF-SFIRE lineage, ``ifire = 2``),
REAL(4) word for word, on the device and in a NumPy float32 reference:

* **the level set** -- ``lfn`` (m) is negative inside the fire; it obeys
  ``lfn_t + ROS(x, n) |grad lfn| = 0``, advanced by ``prop_ls_rk3``
  (``phys/module_fr_fire_core.F``): three Runge-Kutta stages
  ``lfn_0 + dt/3 tend``, ``lfn_0 + dt/2 tend``, ``lfn_0 + dt tend`` of
  ``tend_ls`` with WRF's default ``fire_upwinding = 9`` -- ENO1 upwind
  differences within ten nodes of the domain edge and outside a band of
  ``fire_lsm_band_ngp * dx`` around the front, WENO5 inside it -- the
  front normal taken from the same upwinded differences, and WRF's
  spatially variable artificial viscosity (``fire_viscosity``, and
  ``fire_viscosity_bg`` within ``fire_viscosity_ngp * dx`` of the front);
* **reinitialization** -- ``reinit_ls_rk3`` (``fire_lsm_reinit``, on by
  default): RK3 pseudo-time steps of the Sussman-Smereka-Osher equation
  with ``fire_upwinding_reinit = 4``, then ``lfn = min(lfn, lfn_in)`` so the
  fire only grows;
* **the rate of spread** -- ``fire_ros`` (``module_fr_fire_phys.F``) from
  the given ``fire_params`` arrays (:class:`FireParams`): Rothermel's
  ``ROS = r_0 0.00508 (1 + phiwc umid^bbb + 5.275 betafl^-0.3 tan^2)``,
  with the wind and slope projected on the front normal (``fire_advection``)
  and the 6 m/s cap taken out of the wind and slope parts.  No fuel model is
  evaluated here: the coefficients come in as arrays;
* **arrival time** -- ``tign_update``: interpolated in time where ``lfn``
  crosses zero, the step end (nudged up) where it has not;
* **ignitions** -- ``ignite_fire`` with ``nearest`` (:class:`Ignition`):
  WRF's ignition lines, a point when start and end coincide, walked from
  start to end between their start and end times and spreading out to
  ``radius`` at the ignition ``ros``;
* **fuel and heat** -- :mod:`woof.fire.heat`: ``fuel_left`` (the 2 x 2
  subcell quadrature), ``heat_fluxes`` and ``sum_2d_cells`` onto the
  atmospheric grid as ``GRNHFX``/``GRNQFX``/``CANHFX``/``CANQFX``.

One :func:`step` is one call of WRF's ``fire_model`` passes 4, 5 and 6, in
WRF's order, plus the driver's aggregation.  :func:`advance` takes a
CFL-limited number of equal steps inside a longer ``dt`` (WRF itself only
warns when ``dt`` exceeds the bound).

WRF is the oracle: ``tools/fire_levelset_wrf471_oracle`` compiles the pinned,
byte-unmodified ``module_fr_fire_{core,phys,util}.F`` at -O0 and
``tests/test_fire_levelset_wrf471_parity.py`` holds both paths here to it at
max ULP 0.  The device unit is ``kernels/fire_levelset.cu`` compiled by NVRTC
with ``-fmad=false --ftz=false`` and glibc's ``expf``/``powf``
(``glibc_flt32.cuh``); the host reference evaluates each statement in
float32 with glibc's ``expf``/``powf`` (:mod:`woof.core.noahmp_libm`).

What is not ported, by design:

* the chaparral arm of ``fire_ros`` (``ischap``): WRF leaves ``ros_base``
  undefined there, so there is no defined answer to reproduce;
* ``fire_upwinding`` other than 9, ``fire_upwind_split``, the other
  ``fuel_left_method`` and the ``fire_boundary_guard`` stop (a run control,
  not a field: the fire simply leaves the mesh here).

The mesh (:class:`FireMesh`) refines each atmospheric cell ``sr_x x sr_y``
times, as WRF-SFIRE's ``sr_x``/``sr_y`` do; fire node ``(i, j)`` (1-based)
lies in atmospheric cell ``((i - 1) // sr_x, (j - 1) // sr_y)`` and sits at
``((i - 1/2) dx, (j - 1/2) dy)`` (``set_ideal_coord``).  Fire-mesh arrays are
``(ny, nx)``; the level set and arrival time are carried with WRF's one-node
halo, ``(ny + 2, nx + 2)``.  Nothing here is wired into the model.

On the device a step queues every kernel without reading anything back;
:func:`advance` reads one number per call -- the rate-of-spread bound that
sizes its sub-steps -- unless the caller passes ``ros_max`` or
``substeps_count``.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

F = np.float32

#: No FMA contraction (gfortran -O0 contracts nothing) and no flush to zero.
MODULE_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
MODULE_KEY = "woof.fire.levelset:fire_levelset"
_SOURCES = ("glibc_flt32.cuh", "fire_levelset.cu")
#: tend_ls / advance_ls_reinit's ``bdy_eno1``: ENO1 this close to the edge.
BDY_ENO1 = 10
_BLOCK2 = (32, 8)
_BLOCK1 = 256


def _lit(text: str) -> np.float32:
    """A Fortran REAL(4) literal, rounded once from its decimal (as gfortran).

    ``np.float32(float(text))`` rounds twice (decimal -> binary64 ->
    binary32); :func:`woof.core.urban_tables.decimal_to_f32` does not.
    """
    from woof.core.urban_tables import decimal_to_f32

    return decimal_to_f32(text)


#: WRF's REAL(4) literals and folded constants, as the kernel spells them.
EPS = F(2.0 ** -23)                       # epsilon(0.0)
TINY = F(2.0 ** -126)                     # tiny(0.0)
TOL = F(100) * EPS                        # tend_ls's tol = 100*eps
ROS_MAX = F(6.0)                          # fire_ros's cap, m/s
FTMIN_TO_MS = _lit(".00508")              # ft/min -> m/s
MS_TO_FTMIN = _lit("196.850")
SLOPE_COEF = _lit("5.275")
BETA_EXP = _lit("-0.3")
WIND_CAP = F(30.0)
REINIT_DT = _lit("0.01")                  # dt_s = 0.01*dx
RK_REINIT = (F(1.0) / F(3.0), F(1.0) / F(2.0), F(1.0))


def array_module(*arrays):
    """NumPy, or CuPy when any of ``arrays`` lives on a device.

    CuPy is consulted only when something has already imported it, so the
    NumPy path never opens a device.
    """
    cupy = sys.modules.get("cupy")
    if cupy is None:
        return np
    return cupy.get_array_module(*[a for a in arrays if a is not None])


def on_host(*values) -> bool:
    """Whether every value is a host scalar or NumPy array (no device)."""
    return all(array_module(v) is np for v in values)


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
    """NVRTC directly, -fmad=false and --ftz=false, one module per device.

    Not ``cupy.RawModule``, which appends ``-ftz=true`` after the caller's
    options and NVRTC honours the last one (``woof/core/terrain_drag.py``'s
    loader, which this follows, records the measurement).
    """
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler

    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile

    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(
            source, MODULE_OPTIONS, None, "fire_levelset.cu")
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
    return ((nx + bx - 1) // bx, (ny + by - 1) // by), _BLOCK2


def _grid1(n: int):
    return ((n + _BLOCK1 - 1) // _BLOCK1,), (_BLOCK1,)


# ---------------------------------------------------------------------------
# host float32 libm and the shared selectors
# ---------------------------------------------------------------------------

def _expf(x) -> np.ndarray:
    """glibc's ``expf`` over a float32 array, word for word.

    ``sysdeps/ieee754/flt-32/e_expf.c`` evaluated in binary64 and integer
    array arithmetic with :mod:`woof.core.noahmp_libm`'s tables, operation
    for operation as its scalar ``expf`` (and ``gfk_exp`` on the device).
    """
    from woof.core import noahmp_libm as lm

    x = np.asarray(x, dtype=F)
    # inf and NaN take the special path below; the table path computes
    # garbage for them first.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        xd = x.astype(np.float64)
        z = lm._EXP2F_INVLN2_SCALED * xd
        kd = z + lm._EXP2F_SHIFT
        ki = kd.view(np.uint64)
        kd = kd - lm._EXP2F_SHIFT
        r = z - kd
        t = lm._EXP2F_TAB_U64[(ki & np.uint64(31)).astype(np.intp)]
        t = t + (ki << np.uint64(47))
        s = t.view(np.float64)
        p0, p1, p2 = lm._EXP2F_POLY_SCALED
        zz = p0 * r + p1
        r2 = r * r
        y = p2 * r + 1.0
        y = zz * r2 + y
        out = (y * s).astype(F)
        top = (x.view(np.uint32) >> np.uint32(20)) & np.uint32(0x7FF)
        special = top >= np.uint32(0x42B)          # |x| >= 88, inf or NaN
        if np.any(special):
            hi = np.uint32(0x42B17218).view(F)
            lo = np.uint32(0x42CFF1B4).view(F)
            out = np.where(special & (x > hi), F(np.inf), out)
            out = np.where(special & (x < -lo), F(0.0), out)
            out = np.where(special & np.isnan(x), x + x, out)
    return np.asarray(out, dtype=F)


def _powf(x, y) -> np.ndarray:
    """glibc's ``powf`` over float32 arrays, word for word.

    A zero base (a flat level set's gradient, a calm wind) is answered here
    as glibc answers it, so the element-wise slow path is not taken for it.
    """
    from woof.core.noahmp_libm import powf_array

    x = np.asarray(x, dtype=F)
    y = np.asarray(y, dtype=F)
    shape = np.broadcast_shapes(x.shape, y.shape)
    x = np.broadcast_to(x, shape)
    y = np.broadcast_to(y, shape)
    zero = x == 0
    if not np.any(zero):
        return np.asarray(powf_array(x, y), dtype=F)
    out = np.empty(shape, dtype=F)
    rest = ~zero
    if np.any(rest):
        out[rest] = powf_array(x[rest], y[rest])
    # powf(+-0, y): y > 0 -> +0 (-0 for a negative zero and odd integer y),
    # y = 0 -> 1, y < 0 -> inf (-inf likewise).
    yz, xz = y[zero], x[zero]
    odd = (np.floor(yz) == yz) & (np.mod(np.abs(yz), F(2)) == 1)
    with np.errstate(divide="ignore"):
        out[zero] = np.where(
            yz > 0, np.where(odd, xz, F(0.0)),
            np.where(yz == 0, F(1.0),
                     np.where(odd, F(1.0) / xz, F(np.inf))))
    return out


# Fortran MAX and MIN are np.fmax/np.fmin on the host and fmaxf/fminf on the
# device: both return the number when one operand is NaN (np.maximum would
# return the NaN).  No NaN arises in the domain FireParams.build admits, and
# outside it the host and device still agree.


def _sel_eno(dl, dr):
    """select_eno, branch for branch."""
    a = ~(dl > 0) & ~(dr > 0)
    b = ~(dl < 0) & ~(dr < 0)
    c = ~(dl < 0) & ~(dr > 0)
    pick = np.where(~(np.abs(dr) < np.abs(dl)), dr, dl)
    return np.where(a, dr, np.where(b, dl, np.where(c, pick, F(0.0))))


def _sel_4th(dx, li, lim1, lim2, lip1, lip2):
    d12 = F(12.0) * dx
    p = (F(7.0) * lip1 + F(7.0) * li - lip2 - lim1) / d12
    m = (F(7.0) * li + F(7.0) * lim1 - lip1 - lim2) / d12
    return p - m


_G1 = F(1.0) / F(10.0)
_G2 = F(3.0) / F(5.0)
_G3 = F(3.0) / F(10.0)
_C1312 = F(13.0) / F(12.0)
_WTOL = _lit("1e-6")


def _sel_weno5(dx, it, im1t, im2t, im3t, ip1t, ip2t, ip3t, uf):
    """select_weno5, both face fluxes, in WRF's order of evaluation."""
    up = uf >= 0
    li = it
    lim1 = np.where(up, im1t, ip1t)
    lim2 = np.where(up, im2t, ip2t)
    lim3 = np.where(up, im3t, ip3t)
    lip1 = np.where(up, ip1t, im1t)
    lip2 = np.where(up, ip2t, im2t)
    six, q = F(6.0), F(0.25)

    def sq(v):
        return v * v

    def weights(b1, b2, b3):
        w1t = _G1 / sq(b1 + _WTOL)
        w2t = _G2 / sq(b2 + _WTOL)
        w3t = _G3 / sq(b3 + _WTOL)
        total = w1t + w2t + w3t
        return w1t / total, w2t / total, w3t / total

    fh1 = (F(2.0) * lim2 - F(7.0) * lim1 + F(11.0) * li) / six
    fh2 = (F(-1.0) * lim1 + F(5.0) * li + F(2.0) * lip1) / six
    fh3 = (F(2.0) * li + F(5.0) * lip1 - F(1.0) * lip2) / six
    w1, w2, w3 = weights(
        _C1312 * sq(lim2 - F(2.0) * lim1 + li)
        + q * sq(lim2 - F(4.0) * lim1 + F(3.0) * li),
        _C1312 * sq(lim1 - F(2.0) * li + lip1) + q * sq(lim1 - lip1),
        _C1312 * sq(li - F(2.0) * lip1 + lip2)
        + q * sq(F(3.0) * li - F(4.0) * lip1 + lip2))
    flux_p = w1 * fh1 + w2 * fh2 + w3 * fh3
    fh1 = (F(2.0) * lim3 - F(7.0) * lim2 + F(11.0) * lim1) / six
    fh2 = (F(-1.0) * lim2 + F(5.0) * lim1 + F(2.0) * li) / six
    fh3 = (F(2.0) * lim1 + F(5.0) * li - F(1.0) * lip1) / six
    w1, w2, w3 = weights(
        _C1312 * sq(lim3 - F(2.0) * lim2 + lim1)
        + q * sq(lim3 - F(4.0) * lim2 + F(3.0) * lim1),
        _C1312 * sq(lim2 - F(2.0) * lim1 + li) + q * sq(lim2 - li),
        _C1312 * sq(lim1 - F(2.0) * li + lip1)
        + q * sq(F(3.0) * lim1 - F(4.0) * li + lip1))
    flux_m = w1 * fh1 + w2 * fh2 + w3 * fh3
    return np.where(up, (flux_p - flux_m) / dx, (flux_m - flux_p) / dx)


# ---------------------------------------------------------------------------
# mesh, options, parameters, ignitions, state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FireMesh:
    """A fire mesh refining an ``ny_atm x nx_atm`` atmospheric grid.

    ``dx_atm``/``dy_atm`` are the atmospheric cell sizes (m); the fire nodes
    are ``dx_atm / sr_x`` by ``dy_atm / sr_y`` apart (WRF's ``fdx``/``fdy``,
    REAL(4)).
    """

    nx_atm: int
    ny_atm: int
    dx_atm: float
    dy_atm: float
    sr_x: int = 10
    sr_y: int = 10

    def __post_init__(self):
        for name in ("nx_atm", "ny_atm", "sr_x", "sr_y"):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer, got "
                                 f"{value}")
            object.__setattr__(self, name, int(value))
        for name in ("dx_atm", "dy_atm"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.nx < 4 or self.ny < 4:
            raise ValueError("the fire mesh needs at least 4 x 4 nodes")

    @property
    def nx(self) -> int:
        return self.nx_atm * self.sr_x

    @property
    def ny(self) -> int:
        return self.ny_atm * self.sr_y

    @property
    def dx(self) -> np.float32:
        return F(F(self.dx_atm) / F(self.sr_x))

    @property
    def dy(self) -> np.float32:
        return F(F(self.dy_atm) / F(self.sr_y))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def padded_shape(self) -> tuple[int, int]:
        return (self.ny + 2, self.nx + 2)

    @property
    def atm_shape(self) -> tuple[int, int]:
        return (self.ny_atm, self.nx_atm)

    @property
    def cell_area(self) -> float:
        return float(self.dx) * float(self.dy)

    def coordinates(self, xp=np):
        """``(x, y)`` node coordinates, ``(ny, nx)`` float32 (set_ideal_coord)."""
        x = (xp.arange(self.nx, dtype=F) + F(0.5)) * self.dx
        y = (xp.arange(self.ny, dtype=F) + F(0.5)) * self.dy
        return (xp.broadcast_to(x[None, :], self.shape),
                xp.broadcast_to(y[:, None], self.shape))


@dataclass(frozen=True)
class FireOptions:
    """WRF's ``&fire`` switches this port honours, at their defaults.

    The defaults are the Registry's (``Registry/registry.fire``), which is
    what a WRF run gets through ``module_fr_fire_driver``'s ``set_flags``.
    """

    fire_advection: int = 1
    fire_lsm_reinit: bool = True
    fire_lsm_reinit_iter: int = 1
    fire_lsm_band_ngp: int = 4
    fire_viscosity: float = 0.4
    fire_viscosity_bg: float = 0.4
    fire_viscosity_band: float = 0.5
    fire_viscosity_ngp: int = 2
    fire_lfn_ext_up: float = 1.0
    fire_slope_factor: float = 1.0
    fire_atm_feedback: float = 1.0

    def __post_init__(self):
        if self.fire_advection not in (0, 1):
            raise ValueError("fire_advection must be 0 or 1")
        if int(self.fire_lsm_reinit_iter) < 1:
            raise ValueError("fire_lsm_reinit_iter must be >= 1")


#: The arrays ``fire_ros`` reads, in the kernel's argument order.
PARAM_NAMES = ("vx", "vy", "dzdxf", "dzdyf", "r_0", "bbb", "phiwc", "betafl")
FUEL_NAMES = ("fgip", "fuel_time", "fmc_g")


@dataclass(frozen=True)
class FireParams:
    """WRF's ``fire_params`` on the fire mesh: given arrays, no fuel model.

    ``vx``/``vy`` the wind (m/s), ``dzdxf``/``dzdyf`` the terrain gradient,
    ``r_0`` the no-wind no-slope spread rate (ft/min; ROS = r_0 * 0.00508
    m/s), ``bbb``/``phiwc`` the wind coefficients and ``betafl`` the packing
    ratio of Rothermel's wind and slope factors; ``fgip`` the fuel load
    (kg m-2), ``fuel_time`` the burn time to 1/e (s) and ``fmc_g`` the fuel
    moisture (kg/kg).  Build with :meth:`build`, which holds every field as
    a float32 ``(ny, nx)`` array on one device.
    """

    vx: object
    vy: object
    dzdxf: object
    dzdyf: object
    r_0: object
    bbb: object
    phiwc: object
    betafl: object
    fgip: object
    fuel_time: object
    fmc_g: object

    @classmethod
    def build(cls, mesh: FireMesh, *, xp=np, vx=0.0, vy=0.0, dzdxf=0.0,
              dzdyf=0.0, r_0=0.0, bbb=1.0, phiwc=0.0, betafl=1.0, fgip=0.0,
              fuel_time=None, fmc_g=0.08, validate: bool = True
              ) -> "FireParams":
        """Every field as a C-contiguous float32 ``(ny, nx)`` array on ``xp``.

        The defaults are WRF's no-fuel cell (``set_fire_params``): betafl 1
        and bbb 1 keep ``betafl**(-0.3)`` finite, fuel_time 7/0.85, and
        fmc_g is ``fuelmc_g``'s 0.08.

        ``validate`` refuses what WRF has no defined answer for: a
        non-finite field, ``fuel_time <= 0`` (fuel_left divides by it),
        ``betafl <= 0`` (``betafl**(-0.3)``), ``r_0 < 0``, ``fmc_g <= -1``,
        and a base rate ``r_0 * 0.00508`` above fire_ros's 6 m/s cap, where
        the cap divides 0 by 0.  On a device that is one read, at build
        time; a caller rebuilding parameters inside its time loop passes
        ``validate=False`` and owns the domain.
        """
        if fuel_time is None:
            fuel_time = F(7.0) / _lit("0.85")
        values = dict(vx=vx, vy=vy, dzdxf=dzdxf, dzdyf=dzdyf, r_0=r_0,
                      bbb=bbb, phiwc=phiwc, betafl=betafl, fgip=fgip,
                      fuel_time=fuel_time, fmc_g=fmc_g)
        out = {}
        for name, value in values.items():
            array = xp.asarray(value).astype(F)
            if array.shape not in ((), mesh.shape):
                raise ValueError(f"{name} has shape {array.shape}, expected "
                                 f"{mesh.shape}")
            out[name] = xp.ascontiguousarray(
                xp.broadcast_to(array, mesh.shape)).astype(F, copy=True)
        params = cls(**out)
        if validate:
            params._validate(xp)
        return params

    def _validate(self, xp) -> None:
        bad = []
        for name in PARAM_NAMES + FUEL_NAMES:
            if not bool(xp.all(xp.isfinite(getattr(self, name)))):
                bad.append(f"{name} is not finite everywhere")
        if not bool(xp.all(self.fuel_time > 0)):
            bad.append("fuel_time must be positive")
        if not bool(xp.all(self.betafl > 0)):
            bad.append("betafl must be positive")
        if not bool(xp.all(self.r_0 >= 0)):
            bad.append("r_0 must be non-negative")
        if not bool(xp.all(self.fmc_g > -1)):
            bad.append("fmc_g must exceed -1")
        if not bool(xp.all(self.r_0 * FTMIN_TO_MS <= ROS_MAX)):
            bad.append("r_0 * 0.00508 exceeds fire_ros's 6 m/s cap, where "
                       "WRF divides 0 by 0")
        if bad:
            raise ValueError("; ".join(bad))

    @classmethod
    def isotropic(cls, mesh: FireMesh, ros, *, xp=np, **fuel) -> "FireParams":
        """Spread at ``ros`` (m/s) whatever the normal: r_0 = ros / 0.00508."""
        r0 = xp.asarray(ros, dtype=F) / FTMIN_TO_MS
        return cls.build(mesh, xp=xp, r_0=r0, **fuel)

    @classmethod
    def wind_driven(cls, mesh: FireMesh, *, back, head, wind_u, wind_v,
                    bbb=1.0, xp=np, **fuel) -> "FireParams":
        """Backing rate ``back`` and head rate ``head`` (m/s) in this wind.

        ``phiwc`` makes ``phiwc * umid**bbb = head / back - 1`` at the wind
        speed, so the front facing the wind runs at ``head`` and the one
        backing into it at ``back``; with ``fire_advection = 1`` the wind
        part goes with the cosine between the wind and the front normal.
        """
        u = xp.asarray(wind_u, dtype=np.float64)
        v = xp.asarray(wind_v, dtype=np.float64)
        umid = xp.minimum(xp.hypot(u, v), 30.0) * 196.85
        back64 = xp.asarray(back, dtype=np.float64)
        ratio = xp.asarray(head, dtype=np.float64) / back64 - 1.0
        calm = umid <= 0
        phiwc = xp.where(calm, 0.0,
                         ratio / xp.where(calm, 1.0, umid) ** float(bbb))
        return cls.build(mesh, xp=xp, vx=wind_u, vy=wind_v,
                         r_0=back64 / 0.00508, bbb=bbb, phiwc=phiwc, **fuel)

    def ros_bound(self, options: "FireOptions | None" = None) -> float:
        """An upper bound on the rate of spread (m/s), for the step count.

        tend_ls applies ``fire_slope_factor`` after fire_ros's 6 m/s cap, so
        a factor above 1 can carry the rate past the cap.  For device
        arrays this is the one device-to-host read of a call.
        """
        factor = max(float((options or FireOptions()).fire_slope_factor),
                     1.0)
        xp = array_module(self.r_0)
        f8 = np.float64
        speed = xp.minimum(xp.hypot(self.vx.astype(f8), self.vy.astype(f8)),
                           30.0) * 196.85
        tan2 = self.dzdxf.astype(f8) ** 2 + self.dzdyf.astype(f8) ** 2
        rate = self.r_0.astype(f8) * 0.00508 * (
            1.0 + self.phiwc.astype(f8) * speed ** self.bbb.astype(f8)
            + factor * 5.275 * self.betafl.astype(f8) ** -0.3 * tan2)
        return min(float(xp.max(rate)) * (1.0 + 1e-6),
                   factor * float(ROS_MAX))


def fuel_time_from_weight(weight):
    """WRF v4.7.1's burn time to 1/e from the fuel weight: ``weight/0.85`` s."""
    if np.isscalar(weight):
        return F(F(weight) / _lit("0.85"))
    return np.asarray(weight, dtype=F) / _lit("0.85")


@dataclass(frozen=True)
class Ignition:
    """One WRF ignition line (``ignition_line_type``), coordinates in m.

    A point when ``end`` is omitted.  The line is walked from ``start`` at
    ``start_time`` to ``end`` at ``end_time`` (s); every node within
    ``radius`` of the walked part ignites, the ignition spreading out to the
    radius at ``ros`` (m/s; the Registry's ``fire_ignition_ros`` is 0.01).
    """

    start: tuple[float, float]
    end: tuple[float, float] | None = None
    start_time: float = 0.0
    end_time: float | None = None
    radius: float = 10.0
    ros: float = 0.01

    def __post_init__(self):
        if not self.radius > 0:
            raise ValueError("an ignition needs a positive radius")
        if not self.ros > 0:
            raise ValueError("an ignition needs a positive ros")
        if self.end_time is not None and self.end_time < self.start_time:
            raise ValueError("end_time must not precede start_time")

    def scalars(self, start_ts, end_ts):
        """ignite_fire's prologue in REAL(4): the kernel's scalars, or None.

        None where WRF returns early: the step is wholly before the
        ignition, or after its end plus the time to spread to its radius.
        """
        start_x, start_y = (F(c) for c in self.start)
        end_x, end_y = (F(c) for c in (self.end or self.start))
        start_time = F(self.start_time)
        end_time = F(self.start_time if self.end_time is None
                     else self.end_time)
        radius, ros = F(self.radius), F(self.ros)
        tos = radius / ros
        st = start_time
        et = min(F(end_ts), end_time)
        if F(start_ts) > et + tos or F(end_ts) < st:
            return None
        if start_time < end_time:
            sx, sy = start_x, start_y
            rele = (et - start_time) / (end_time - start_time)
            ex = start_x + rele * (end_x - start_x)
            ey = start_y + rele * (end_y - start_y)
        else:
            sx, sy, ex, ey = start_x, start_y, end_x, end_y
        unit = F(1.0) * F(1.0)                 # unit_xf * unit_xf, ideal (m)
        return (sx, sy, ex, ey, st, et, unit, unit, radius, ros)


@dataclass(frozen=True)
class FireState:
    """WRF's fire state at ``time`` (s, REAL(4)).

    ``lfn_m`` and ``tign_m`` are the level set (m) and ignition time (s)
    with WRF's one-node halo, ``(ny + 2, nx + 2)`` (the halo holds WRF's
    zero fill); ``fuel_frac`` is the fuel fraction left, ``(ny, nx)``.
    """

    time: np.float32
    lfn_m: object
    tign_m: object
    fuel_frac: object

    @property
    def lfn(self):
        return self.lfn_m[1:-1, 1:-1]

    @property
    def tign(self):
        return self.tign_m[1:-1, 1:-1]

    @property
    def burning(self):
        return self.lfn <= 0


@dataclass(frozen=True)
class FireOutputs:
    """What one step produces besides the state.

    Fire mesh ``(ny, nx)``: ``ros`` (m/s, the last stage's), ``fire_area``,
    ``fuel_frac_burnt``, ``fgrnhfx``/``fgrnqfx`` (W m-2).  Atmospheric grid
    ``(ny_atm, nx_atm)``: ``GRNHFX``, ``GRNQFX``, ``CANHFX``, ``CANQFX``
    (W m-2; WRF has no canopy fluxes yet, so those are zero) and
    ``AVG_FUEL_FRAC``.  ``tbound`` is prop_ls_rk3's CFL bound (s, 0-d):
    the least of its three stages' ``1 / (max + tol)``.
    """

    ros: object
    fire_area: object
    fuel_frac_burnt: object
    fgrnhfx: object
    fgrnqfx: object
    GRNHFX: object
    GRNQFX: object
    CANHFX: object
    CANQFX: object
    AVG_FUEL_FRAC: object
    tbound: object


def initial_state(mesh: FireMesh, *, time: float = 0.0, xp=np) -> FireState:
    """``init_no_fire``: no fire, ignition times a little in the future."""
    t = F(time)
    lfn_init = F(2) * max(F(mesh.nx) * mesh.dx, F(mesh.ny) * mesh.dy)
    time_init = t + max(t, F(1.0)) * EPS
    lfn = xp.zeros(mesh.padded_shape, dtype=F)
    tign = xp.zeros(mesh.padded_shape, dtype=F)
    lfn[1:-1, 1:-1] = lfn_init
    tign[1:-1, 1:-1] = time_init
    return FireState(time=t, lfn_m=lfn, tign_m=tign,
                     fuel_frac=xp.ones(mesh.shape, dtype=F))


def burned_area(state: FireState, mesh: FireMesh) -> float:
    """The area of the burning nodes' cells, m^2."""
    xp = array_module(state.lfn_m)
    return float(xp.count_nonzero(state.lfn <= 0)) * mesh.cell_area


# ---------------------------------------------------------------------------
# the host reference, kernel for kernel
# ---------------------------------------------------------------------------

def _extend_cpu(a: np.ndarray, bias) -> None:
    """continue_at_boundary(1, 1, bias) on a padded array, in place."""
    bias = F(bias)
    one = F(1.0)

    def ex(x, y):
        lin = F(2.0) * x - y
        return (one - bias) * lin + bias * np.fmax(np.fmax(lin, x), y)

    a[1:-1, 0] = ex(a[1:-1, 1], a[1:-1, 2])
    a[1:-1, -1] = ex(a[1:-1, -2], a[1:-1, -3])
    a[0, 1:-1] = ex(a[1, 1:-1], a[2, 1:-1])
    a[-1, 1:-1] = ex(a[-2, 1:-1], a[-3, 1:-1])
    a[0, 0] = ex(a[1, 1], a[2, 2])
    a[-1, 0] = ex(a[-2, 1], a[-3, 2])
    a[0, -1] = ex(a[1, -2], a[2, -3])
    a[-1, -1] = ex(a[-2, -2], a[-3, -3])


def _band(mesh: FireMesh):
    """``(edge, inner)``: within the ENO1 edge band; strictly inside it."""
    i = np.arange(1, mesh.nx + 1)[None, :]
    j = np.arange(1, mesh.ny + 1)[:, None]
    b = BDY_ENO1
    edge = (i < 1 + b) | (i > mesh.nx - b) | (j < 1 + b) | (j > mesh.ny - b)
    inner = (i > 1 + b) & (i < mesh.nx - b) & (j > 1 + b) & (j < mesh.ny - b)
    return edge, inner


def _shifted(a: np.ndarray):
    """``s(di, dj)``: the padded array's value at node (i + di, j + dj).

    Offsets up to 3 reach past the one-node halo; those entries are read
    only where the ten-node ENO1 band discards them, so they are zeros.
    """
    wide = np.pad(a, 2)
    ny, nx = a.shape[0] - 2, a.shape[1] - 2

    def s(di, dj):
        return wide[3 + dj:3 + dj + ny, 3 + di:3 + di + nx]

    return s


def _fire_ros_cpu(nvx, nvy, p: FireParams, advection: int):
    """fire_ros (Rothermel, not chaparral), element for element."""
    vx, vy, dzx, dzy = p.vx, p.vy, p.dzdxf, p.dzdyf
    if advection:
        speed = np.sqrt(vx * vx + vy * vy) + TINY
        tanphi = np.sqrt(dzx * dzx + dzy * dzy) + TINY
        cor_wind = np.fmax(F(0.0), (vx * nvx + vy * nvy) / speed)
        cor_slope = np.fmax(F(0.0), (dzx * nvx + dzy * nvy) / tanphi)
    else:
        speed = vx * nvx + vy * nvy
        tanphi = dzx * nvx + dzy * nvy
        cor_wind = cor_slope = F(1.0)
    spdms = np.fmax(speed, F(0.0))
    umidm = np.fmin(spdms, WIND_CAP)
    umid = umidm * MS_TO_FTMIN
    phiw = _powf(umid, p.bbb) * p.phiwc
    phis = np.where(tanphi > 0, SLOPE_COEF * _powf(p.betafl, BETA_EXP)
                    * (tanphi * tanphi), F(0.0))
    rb = p.r_0 * FTMIN_TO_MS
    rw = rb * phiw
    rs = rb * phis
    rw = rw * cor_wind
    rs = rs * cor_slope
    excess = rb + rw + rs - ROS_MAX
    cut = excess > 0
    if np.any(cut):
        rw2 = rw - (excess * rw) / (rw + rs)
        rs2 = rs - (excess * rs) / (rw2 + rs)
        rw = np.where(cut, rw2, rw)
        rs = np.where(cut, rs2, rs)
    return rb, rw, rs


def _tend_stage_cpu(lfn, lfn0, mesh, p, opts, coef, edge, inner):
    """One prop_ls_rk3 stage: ``(lfn_0 + coef*tend, ros, CFL contribution)``."""
    dx, dy = mesh.dx, mesh.dy
    s = _shifted(lfn)
    c = s(0, 0)
    drx = (s(1, 0) - c) / dx
    dlx = (c - s(-1, 0)) / dx
    dry = (s(0, 1) - c) / dy
    dly = (c - s(0, -1)) / dy
    thr_hl = F(opts.fire_lsm_band_ngp) * dx
    thr_av = F(opts.fire_viscosity_ngp) * dx
    eno = edge | ~(np.abs(c) < thr_hl)
    d2x = _sel_eno(dlx, drx)
    d2y = _sel_eno(dly, dry)
    if not np.all(eno):
        s4x = _sel_4th(dx, c, s(-1, 0), s(-2, 0), s(1, 0), s(2, 0))
        s4y = _sel_4th(dy, c, s(0, -1), s(0, -2), s(0, 1), s(0, 2))
        a = s4x * p.vx + s4y * p.vy
        w5x = _sel_weno5(dx, c, s(-1, 0), s(-2, 0), s(-3, 0), s(1, 0),
                         s(2, 0), s(3, 0), a * s4x)
        w5y = _sel_weno5(dy, c, s(0, -1), s(0, -2), s(0, -3), s(0, 1),
                         s(0, 2), s(0, 3), a * s4y)
        d2x = np.where(eno, d2x, w5x)
        d2y = np.where(eno, d2y, w5y)
    grad = np.sqrt(d2x * d2x + d2y * d2y)
    scale = np.sqrt(_powf(grad, F(2.0)) + EPS)
    nvx = d2x / scale
    nvy = d2y / scale
    rb, rw, rs = _fire_ros_cpu(nvx, nvy, p, opts.fire_advection)
    rr = rb + rw + F(opts.fire_slope_factor) * rs
    rr = np.fmax(rr, F(0.0))
    te = -(rr * grad)
    positive = grad > 0
    cb = np.where(positive, (rr * (np.abs(d2x) / dx + np.abs(d2y) / dy))
                  / np.where(positive, grad, F(1.0)), F(0.0))
    al = np.abs(c)
    visc, bg = F(opts.fire_viscosity), F(opts.fire_viscosity_bg)
    band = F(opts.fire_viscosity_band)
    ramp = np.fmin(bg + ((visc - bg) * (al - thr_av)) / (band * thr_av),
                      visc)
    var = np.where((al < thr_av) & inner, bg,
                   np.where((al >= thr_av) & (al < thr_av * (F(1.0) + band))
                            & inner, ramp, visc))
    te = te + var * np.abs(rr) * ((drx - dlx) + (dry - dly))
    out = lfn0.copy()
    out[1:-1, 1:-1] = lfn0[1:-1, 1:-1] + coef * te
    return out, np.asarray(rr, dtype=F), np.asarray(cb, dtype=F)


def _reinit_stage_cpu(s0, ini, curr, mesh, coef, thr_hl, edge):
    """One advance_ls_reinit stage (fire_upwinding_reinit = 4)."""
    dx, dy = mesh.dx, mesh.dy
    s = _shifted(curr)
    c = s(0, 0)
    sg = s0[1:-1, 1:-1]
    eno = edge | ~(c < thr_hl)
    d2x = _sel_eno((c - s(-1, 0)) / dx, (s(1, 0) - c) / dx)
    d2y = _sel_eno((c - s(0, -1)) / dy, (s(0, 1) - c) / dy)
    if not np.all(eno):
        s4x = _sel_4th(dx, c, s(-1, 0), s(-2, 0), s(1, 0), s(2, 0))
        s4y = _sel_4th(dy, c, s(0, -1), s(0, -2), s(0, 1), s(0, 2))
        w5x = _sel_weno5(dx, c, s(-1, 0), s(-2, 0), s(-3, 0), s(1, 0),
                         s(2, 0), s(3, 0), sg * s4x)
        w5y = _sel_weno5(dy, c, s(0, -1), s(0, -2), s(0, -3), s(0, 1),
                         s(0, 2), s(0, 3), sg * s4y)
        d2x = np.where(eno, d2x, w5x)
        d2y = np.where(eno, d2y, w5y)
    grad = np.sqrt(d2x * d2x + d2y * d2y)
    tend = sg * (F(1.0) - grad)
    out = ini.copy()
    out[1:-1, 1:-1] = ini[1:-1, 1:-1] + coef * tend
    return out


def _ignite_cpu(lfn_m, tign_m, mesh, sc, start_ts, end_ts):
    """ignite_fire (with nearest) for one ignition line, in place."""
    sx, sy, ex, ey, st, et, cx2, cy2, radius, ros = sc
    half = F(0.5)
    ax, ay = mesh.coordinates(np)
    mx = (sx + ex) * half
    my = (sy + ey) * half
    dam2 = (ax - mx) * (ax - mx) * cx2 + (ay - my) * (ay - my) * cy2
    des2 = (ex - sx) * (ex - sx) * cx2 + (ey - sy) * (ey - sy) * cy2
    dames = dam2 * des2
    am_es = (ax - mx) * (ex - sx) * cx2 + (ay - my) * (ey - sy) * cy2
    pos = dames > 0
    cos2 = np.where(pos, (am_es * am_es) / np.where(pos, dames, F(1.0)),
                    F(0.0))
    dmc2 = dam2 * cos2
    inside = F(4.0) * dmc2 < des2
    safe = des2 if des2 > 0 else F(1.0)
    mcrel = np.where(inside,
                     np.copysign(np.sqrt((F(4.0) * dmc2) / safe), am_es),
                     np.where(am_es > 0, F(1.0), F(-1.0))).astype(F)
    cx = (ex + sx) * half + mcrel * (ex - sx) * half
    cy = (ey + sy) * half + mcrel * (ey - sy) * half
    d = np.sqrt((ax - cx) * (ax - cx) * cx2 + (ay - cy) * (ay - cy) * cy2)
    t = (et + st) * half + mcrel * (et - st) * half
    lfn_new = d - np.fmin(radius, ros * (F(end_ts) - t))
    lfn = lfn_m[1:-1, 1:-1]
    tign = tign_m[1:-1, 1:-1]
    lit = (lfn > 0) & ~(lfn_new > 0)
    ti = t + d / ros
    ti = np.where((ti < start_ts) | (ti > end_ts),
                  np.fmin(np.fmax(ti, F(start_ts)), F(end_ts)), ti)
    tign[...] = np.where(lit, ti, tign)
    lfn[...] = np.fmin(lfn, lfn_new)


# ---------------------------------------------------------------------------
# one WRF fire step
# ---------------------------------------------------------------------------

def _check(state: FireState, mesh: FireMesh, params: FireParams):
    xp = array_module(state.lfn_m)
    for name, array, shape in (("lfn_m", state.lfn_m, mesh.padded_shape),
                               ("tign_m", state.tign_m, mesh.padded_shape),
                               ("fuel_frac", state.fuel_frac, mesh.shape)):
        if (array_module(array) is not xp or array.shape != shape
                or array.dtype != F or not array.flags.c_contiguous):
            raise ValueError(f"state.{name} must be a C-contiguous float32 "
                             f"{shape} array beside the level set")
    for name in PARAM_NAMES + FUEL_NAMES:
        value = getattr(params, name)
        if (array_module(value) is not xp or value.shape != mesh.shape
                or value.dtype != F or not value.flags.c_contiguous):
            raise ValueError(f"params.{name} must be a C-contiguous float32 "
                             f"{mesh.shape} array beside the state "
                             "(FireParams.build)")
    return xp


def step(state: FireState, mesh: FireMesh, params: FireParams, dt: float, *,
         ignitions=(), options: FireOptions = FireOptions()
         ) -> tuple[FireState, FireOutputs]:
    """One WRF fire step of ``dt`` seconds: passes 4, 5 and 6, aggregated.

    Runs where the state lives (NumPy or CuPy); the state is not modified.
    """
    xp = _check(state, mesh, params)
    if not float(dt) > 0:
        raise ValueError(f"dt must be positive, got {dt}")
    run = _step_cpu if xp is np else _step_gpu
    return run(state, mesh, params, F(dt), tuple(ignitions), options)


def _step_cpu(state, mesh, params, dt, ignitions, opts):
    from woof.fire import heat

    ts = F(state.time)
    edge, inner = _band(mesh)
    bias = F(opts.fire_lfn_ext_up)
    lfn_in = state.lfn_m
    tign = state.tign_m.copy()
    # pass 4: prop_ls_rk3
    lfn0 = np.zeros_like(lfn_in)
    lfn0[1:-1, 1:-1] = lfn_in[1:-1, 1:-1]
    tb = []
    cur = lfn0
    with np.errstate(all="ignore"):
        for coef in (dt / F(3.0), dt / F(2.0), dt):
            _extend_cpu(cur, bias)
            cur, ros, cb = _tend_stage_cpu(cur, lfn0, mesh, params, opts,
                                           coef, edge, inner)
            tb.append(F(1.0) / (F(np.max(cb, initial=F(0.0))) + TOL))
        lfn_out = cur
        tbound = np.asarray(min(min(tb[0], tb[1]), tb[2]), dtype=F)
        # tign_update
        tn = ts + dt
        tn = tn + np.abs(tn) * EPS * F(2.0)
        li, lo = lfn_in[1:-1, 1:-1], lfn_out[1:-1, 1:-1]
        t_int = tign[1:-1, 1:-1]
        t_int[...] = np.where(~(lo > 0) & (li > 0),
                              ts + (dt * li) / (li - lo), t_int)
        t_int[...] = np.where(lo > 0, tn, t_int)
        # reinit_ls_rk3, then pass 5's lfn = lfn_out
        lfn = lfn_in.copy()
        if opts.fire_lsm_reinit:
            thr_hl = F(opts.fire_lsm_band_ngp) * mesh.dx
            s0 = np.zeros_like(lfn_in)
            s3 = np.zeros_like(lfn_in)
            s0[1:-1, 1:-1] = lo / np.sqrt(_powf(lo, F(2.0))
                                          + _powf(mesh.dx, F(2.0)))
            s3[1:-1, 1:-1] = lo
            _extend_cpu(s3, bias)
            dts = REINIT_DT * mesh.dx
            for _ in range(int(opts.fire_lsm_reinit_iter)):
                s1 = _reinit_stage_cpu(s0, s3, s3, mesh, dts * RK_REINIT[0],
                                       thr_hl, edge)
                _extend_cpu(s1, bias)
                s2 = _reinit_stage_cpu(s0, s3, s1, mesh, dts * RK_REINIT[1],
                                       thr_hl, edge)
                _extend_cpu(s2, bias)
                s3 = _reinit_stage_cpu(s0, s3, s2, mesh, dts * RK_REINIT[2],
                                       thr_hl, edge)
                _extend_cpu(s3, bias)
            lfn[1:-1, 1:-1] = np.fmin(s3[1:-1, 1:-1], lfn_in[1:-1, 1:-1])
        else:
            lfn[1:-1, 1:-1] = lo
        # pass 5: the ignitions, in order
        end_ts = ts + dt
        for ignition in ignitions:
            sc = ignition.scalars(ts, end_ts)
            if sc is not None:
                _ignite_cpu(lfn, tign, mesh, sc, ts, end_ts)
        # pass 6: fuel_left, heat_fluxes; then the driver's aggregation
        fuel_frac = state.fuel_frac.copy()
        fine = heat.fuel_heat_cpu(lfn, tign, params, fuel_frac, end_ts, dt)
    coarse = heat.aggregate_cpu(fine, fuel_frac, mesh, opts)
    new = FireState(time=end_ts, lfn_m=lfn, tign_m=tign, fuel_frac=fuel_frac)
    return new, FireOutputs(ros=ros, tbound=tbound, **fine, **coarse)


def _step_gpu(state, mesh, params, dt, ignitions, opts):
    import cupy as cp

    from woof.fire import heat

    nx, ny = mesh.nx, mesh.ny
    i32 = np.int32
    ts = F(state.time)
    bias = F(opts.fire_lfn_ext_up)
    grid2, block2 = _grid2(nx, ny)
    ring, ring_block = _grid1(nx + ny + 1)
    extend = _kernel("fl_extend")
    lfn_in = state.lfn_m
    tign = state.tign_m.copy()
    # pass 4: prop_ls_rk3
    # Work arrays need no fill: fl_extend writes each halo before it is read.
    lfn0 = cp.empty_like(lfn_in)
    lfn0[1:-1, 1:-1] = lfn_in[1:-1, 1:-1]
    stage_out = [cp.empty_like(lfn_in) for _ in range(3)]
    ros = cp.empty(mesh.shape, dtype=F)
    blocks = grid2[0] * grid2[1]
    partial = cp.empty(blocks, dtype=F)
    tbound = cp.empty(4, dtype=F)
    p = [getattr(params, name) for name in PARAM_NAMES]
    thr_hl = F(opts.fire_lsm_band_ngp) * mesh.dx
    thr_av = F(opts.fire_viscosity_ngp) * mesh.dx
    tend = _kernel("fl_tend_stage")
    reduce = _kernel("fl_bound_reduce")
    cur = lfn0
    for k, coef in enumerate((dt / F(3.0), dt / F(2.0), dt)):
        extend(ring, ring_block, (cur, bias, i32(nx), i32(ny)))
        tend(grid2, block2,
             (cur, lfn0, stage_out[k], ros, partial, *p, mesh.dx, mesh.dy,
              coef, thr_hl, thr_av, F(opts.fire_viscosity),
              F(opts.fire_viscosity_bg), F(opts.fire_viscosity_band),
              F(opts.fire_slope_factor), i32(opts.fire_advection), i32(nx),
              i32(ny)))
        reduce((1,), (_BLOCK1,), (partial, i32(blocks), TOL, tbound,
                                  i32(k)))
        cur = stage_out[k]
    lfn_out = cur
    tbound = tbound[3]
    tn = ts + dt
    tn = tn + np.abs(tn) * EPS * F(2.0)
    _kernel("fl_tign_update")(grid2, block2, (lfn_in, lfn_out, tign, ts, dt,
                                              tn, i32(nx), i32(ny)))
    lfn = lfn_in.copy()
    final = _kernel("fl_reinit_final")
    if opts.fire_lsm_reinit:
        s0, s1, s2, s3 = (cp.empty_like(lfn_in) for _ in range(4))
        dx2 = F(_powf(mesh.dx, F(2.0)))             # dx**2.0, once
        _kernel("fl_reinit_init")(grid2, block2, (lfn_out, s0, s3, dx2,
                                                  i32(nx), i32(ny)))
        extend(ring, ring_block, (s3, bias, i32(nx), i32(ny)))
        stage = _kernel("fl_reinit_stage")
        dts = REINIT_DT * mesh.dx
        for _ in range(int(opts.fire_lsm_reinit_iter)):
            for src, dst, rk in ((s3, s1, RK_REINIT[0]),
                                 (s1, s2, RK_REINIT[1]),
                                 (s2, s3, RK_REINIT[2])):
                stage(grid2, block2, (s0, s3, src, dst, mesh.dx, mesh.dy,
                                      dts * rk, thr_hl, i32(nx), i32(ny)))
                extend(ring, ring_block, (dst, bias, i32(nx), i32(ny)))
        final(grid2, block2, (s3, lfn, i32(1), i32(nx), i32(ny)))
    else:
        final(grid2, block2, (lfn_out, lfn, i32(0), i32(nx), i32(ny)))
    end_ts = ts + dt
    ignite = _kernel("fl_ignite")
    for ignition in ignitions:
        sc = ignition.scalars(ts, end_ts)
        if sc is not None:
            ignite(grid2, block2, (lfn, tign, *sc, ts, end_ts, mesh.dx,
                                   mesh.dy, i32(nx), i32(ny)))
    fuel_frac = state.fuel_frac.copy()
    fine = heat.fuel_heat_gpu(lfn, tign, params, fuel_frac, end_ts, dt, mesh)
    coarse = heat.aggregate_gpu(fine, fuel_frac, mesh, opts)
    new = FireState(time=end_ts, lfn_m=lfn, tign_m=tign, fuel_frac=fuel_frac)
    return new, FireOutputs(ros=ros, tbound=tbound, **fine, **coarse)


# ---------------------------------------------------------------------------
# a longer interval in CFL-limited steps
# ---------------------------------------------------------------------------

#: Courant number for :func:`advance`'s steps.  tend_ls's own bound,
#: ``1 / max(ROS (|D2x|/dx + |D2y|/dy) / |grad|)``, is at least
#: ``1 / (ROS_max (1/dx + 1/dy))``.
DEFAULT_CFL = 0.5


def substeps(mesh: FireMesh, ros_max: float, dt: float, *,
             cfl: float = DEFAULT_CFL) -> int:
    """Equal steps for ``dt``: ``ceil(dt ROS_max (1/dx + 1/dy) / cfl)``."""
    if not dt > 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if not 0 < cfl <= 1:
        raise ValueError(f"cfl must be in (0, 1], got {cfl}")
    if not ros_max >= 0:
        raise ValueError(f"ros_max must be >= 0, got {ros_max}")
    rate = float(ros_max) * (1.0 / float(mesh.dx) + 1.0 / float(mesh.dy))
    return max(1, math.ceil(float(dt) * rate / cfl * (1.0 - 1e-12)))


ATMOSPHERE_FLUX_NAMES = ("GRNHFX", "GRNQFX", "CANHFX", "CANQFX")


def advance(state: FireState, mesh: FireMesh, params: FireParams, dt: float,
            *, ignitions=(), options: FireOptions = FireOptions(),
            substeps_count: int | None = None, ros_max: float | None = None,
            cfl: float = DEFAULT_CFL):
    """``dt`` seconds in ``n`` equal :func:`step` calls of ``dt/n``.

    Returns ``(state, last, mean)``: the final state, the last step's
    :class:`FireOutputs`, and the atmospheric fluxes over the whole ``dt``
    (``GRNHFX``, ``GRNQFX``, ``CANHFX``, ``CANQFX``: summed in step order,
    then divided by ``n``).  ``n`` is ``substeps_count`` when given, else
    :func:`substeps` from ``ros_max`` or the parameters' own bound (one
    device read).  The clock advances by ``dt/n`` in REAL(4) each step.
    """
    if substeps_count is None:
        bound = (params.ros_bound(options) if ros_max is None
                 else float(ros_max))
        substeps_count = substeps(mesh, bound, dt, cfl=cfl)
    n = int(substeps_count)
    if n < 1:
        raise ValueError("substeps_count must be >= 1")
    h = F(F(dt) / F(n))
    ignitions = tuple(ignitions)
    xp = array_module(state.lfn_m)
    mean = {name: xp.zeros(mesh.atm_shape, dtype=F)
            for name in ATMOSPHERE_FLUX_NAMES}
    last = None
    for _ in range(n):
        state, last = step(state, mesh, params, h, ignitions=ignitions,
                           options=options)
        for name in ATMOSPHERE_FLUX_NAMES:
            _accumulate(mean[name], getattr(last, name))
    for name in ATMOSPHERE_FLUX_NAMES:
        _scale(mean[name], F(n))
    return state, last, mean


def _accumulate(acc, x) -> None:
    if array_module(acc) is np:
        acc += x
        return
    grid, block = _grid1(acc.size)
    _kernel("fl_accumulate")(grid, block, (acc, x, np.int32(acc.size)))


def _scale(acc, divisor) -> None:
    if array_module(acc) is np:
        acc /= divisor
        return
    grid, block = _grid1(acc.size)
    _kernel("fl_scale")(grid, block, (acc, F(divisor), np.int32(acc.size)))
