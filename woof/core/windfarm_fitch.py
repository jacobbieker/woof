"""The Fitch et al. (2012) wind-farm parameterization from WRF v4.7.1.

WRF's ``windfarm_opt = 1`` (``phys/module_wind_fitch.F``) represents each
turbine as an elevated momentum sink and TKE source in the model levels its
rotor sweeps.  The PBL driver calls it right after the PBL scheme
(``module_pbl_driver.F:2044-2063``), adding into RUBLTEN/RVBLTEN and QKE on
the mass-point winds:

* the rotor's levels are those its blade tips cut; each level's share of the
  swept disk is the circular segment between the level's interfaces
  (``turbine_area``), :func:`rotor_overlap` here;
* the thrust, power and TKE coefficients are read once per turbine from its
  curve at the hub-height speed (``dragcof``): Ct linear in the thrust
  table, Cp linear in the power coefficients P/(0.5*1.23*V^3*A) the power
  table implies, and C_TKE = windfarm_tke_factor*(Ct - Cp), floored at 0.
  The thrust a rotor exerts is split into the power it extracts (Cp) and the
  turbulence it makes (Ct - Cp);
* the power is POWER1 = 0.5*1.23*Vhub^3*A*Cp, WRF's 1.23 kg m-3 being the
  density the power table's coefficients are formed at;
* per rotor level, with ec = POWER1/POWER2 (POWER2 the same power summed at
  the levels' own speeds, so the levels together extract the hub's power)::

      du -= 0.5*u*Ct*|V|*A_k/(dx*dy*dz)*ec
      dv -= 0.5*v*Ct*|V|*A_k/(dx*dy*dz)*ec
      QKE += |V|^3*A_k*C_TKE*dt/(dx*dy*dz)*ec      (QKE = 2*TKE)

  That is Fitch's thrust 0.5*rho*Ct*A*|V|*V divided by the layer's air
  mass rho*dx*dy*dz, which is why the tendencies need no density.

The turbines come from a :class:`FitchLayout` of turbine GROUPS: each group
is ``count`` identical turbines in one cell, of one curve, hub height and
rotor diameter, so a cell may hold several machines; a uniform cell is one
group.  WRF loops over single turbines, each over its own rotor levels, and
a cell's tendencies, TKE and power are the sum over its turbines in the
order of its turbine list; a cell's groups are added in the order given, so
a group of n is n consecutive identical turbines of that list, and a list
reading A, B, A in one cell is three groups.  The curves are WRF's
``wind-turbine-N.tbl`` (speed, Ct, power in kW, and the standing thrust
coefficient below cut-in).  WRF's three deviations (where it reads an
undefined value or would form a NaN) are listed in
``kernels/windfarm_fitch.cu``.  WRF takes ``dx*dx`` for the cell area;
``dy`` defaults to ``dx`` so the default is WRF's.

:func:`launch_fitch` adds into caller-owned ``du``/``dv`` (and ``qke``) in
place, as :func:`woof.core.terrain_drag.launch_gwdo` does, so a PBL slot can
call it beside the orographic drag.  The layout's arrays and curves are
copied to each device once and stay there; a call allocates its outputs and
launches one kernel, with no host synchronisation.

The kernel is held word for word (max ULP 0) to WRF's own Fortran by
``tools/windfarm_fitch_wrf471_oracle`` and to the binary32 CPU mirror
:func:`woof.verify.windfarm_fitch_ref.f32_fitch_column`
(``tests/test_windfarm_fitch_wrf471_parity.py``); the float64 mirror
:func:`woof.verify.windfarm_fitch_ref.np_fitch_column` is a secondary
reference for the closed forms (``tests/test_windfarm_fitch.py``).  The
model calls it through :mod:`woof.core.windfarm_coupling` (``windfarm_opt =
1``) from the PBL slot, held word for word to this launcher by
``tests/test_windfarm_coupling_gpu.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

#: WRF v4.7.1 Registry.EM_COMMON:3005, ``windfarm_tke_factor`` (Archer et
#: al. 2020's correction of the TKE source).
WINDFARM_TKE_FACTOR = 0.25
#: module_model_constants' ``piconst`` as a REAL.
PICONST = np.float32(3.1415926535897932384626433)

#: No FMA contraction (gfortran contracts nothing) and no flush to zero.
MODULE_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
MODULE_KEY = "woof.core.windfarm_fitch:windfarm_fitch"
_SOURCES = ("glibc_flt32.cuh", "glibc_trig_flt32.cuh", "windfarm_fitch.cu")
_BLOCK = 128
#: A null device pointer for the kernel's optional arrays.
_NULL = np.uintp(0)

_GROUP_FIELDS = ("cell", "count", "curve", "hub_height_m", "rotor_diameter_m")
_CURVE_FIELDS = ("speeds_ms", "thrust_coefficient", "power_kw", "nval",
                 "standing_thrust")
_DEVICE_FIELDS = ("cell_offsets", "count", "curve", "hub_height_m",
                  "rotor_diameter_m") + _CURVE_FIELDS


def _frozen(value, dtype):
    """A private, read-only C-ordered copy: the device copy is made once."""
    out = np.array(value, dtype=dtype, order="C", copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, eq=False)
class FitchLayout:
    """The turbines of one (ny, nx) mass grid, as groups, plus their curves.

    Per group, (ngroups,): ``cell``, the flat mass cell ``j*nx + i``;
    ``count`` identical turbines of curve ``curve`` with hub height
    ``hub_height_m`` (above ground) and rotor ``rotor_diameter_m``.  Per
    curve, (ncurves, maxvals) tables ascending in wind speed and zero-padded
    past ``nval`` entries: ``speeds_ms``, ``thrust_coefficient`` (Ct) and
    ``power_kw``, WRF's ``wind-turbine-N.tbl`` rows; and
    ``standing_thrust`` (ncurves,), the Ct below cut-in.  Cut-in is a
    curve's first speed, cut-out its last, and the Ct above cut-out its last
    Ct, as ``init_module_wind_fitch`` reads them.

    ORDER.  A cell's groups are added in the order given (a group's n
    turbines one after another), which is WRF's order when each group is a
    run of consecutive identical turbines in the cell's turbine list.

    STORAGE (CSR).  The groups are held sorted by cell, stably, so a cell's
    groups keep the order given; ``cell_offsets`` (ny*nx + 1) indexes them:
    cell c owns groups ``cell_offsets[c] .. cell_offsets[c+1]-1``.  One GPU
    thread owns one column and walks its own short run of groups, so a
    cell without turbines costs one comparison and no column pays for
    another's mix.
    """

    ny: int
    nx: int
    cell: np.ndarray
    count: np.ndarray
    curve: np.ndarray
    hub_height_m: np.ndarray
    rotor_diameter_m: np.ndarray
    speeds_ms: np.ndarray
    thrust_coefficient: np.ndarray
    power_kw: np.ndarray
    nval: np.ndarray
    standing_thrust: np.ndarray
    cell_offsets: np.ndarray = field(init=False, repr=False)
    _device: dict = field(default_factory=dict, init=False, repr=False,
                          compare=False)

    def __post_init__(self):
        put = lambda name, value: object.__setattr__(self, name, value)  # noqa: E731
        put("ny", int(self.ny))
        put("nx", int(self.nx))
        groups = {name: np.ravel(np.asarray(getattr(self, name)))
                  for name in _GROUP_FIELDS}
        ngroups = groups["cell"].size
        for name, value in groups.items():
            if value.size != ngroups:
                raise ValueError(f"{name} has {value.size} groups, cell has "
                                 f"{ngroups}")
        for name in ("cell", "count", "curve"):
            value = groups[name]
            if value.size and not np.array_equal(
                    value, np.round(value.astype(np.float64))):
                raise ValueError(f"{name} must be integers")
        for name in ("hub_height_m", "rotor_diameter_m"):
            if not np.isfinite(groups[name].astype(np.float64)).all():
                raise ValueError(f"{name} must be finite")
        cell = groups["cell"].astype(np.int64)
        if ((cell < 0) | (cell >= self.ny * self.nx)).any():
            raise ValueError(f"group cells must lie in [0, {self.ny * self.nx})")
        order = np.argsort(cell, kind="stable")
        for name in _GROUP_FIELDS:
            dtype = np.float32 if name.endswith("_m") else np.int32
            put(name, _frozen(groups[name][order], dtype))
        for name in _CURVE_FIELDS:
            dtype = np.int32 if name == "nval" else np.float32
            put(name, _frozen(getattr(self, name), dtype))
        offsets = np.zeros(self.ny * self.nx + 1, np.int64)
        np.cumsum(np.bincount(self.cell, minlength=self.ny * self.nx),
                  out=offsets[1:])
        put("cell_offsets", _frozen(offsets, np.int32))
        self._validate()

    def _validate(self):
        if self.speeds_ms.ndim != 2:
            raise ValueError("speeds_ms must be (ncurves, maxvals)")
        ncurves, maxvals = self.speeds_ms.shape
        for name in ("thrust_coefficient", "power_kw"):
            if getattr(self, name).shape != (ncurves, maxvals):
                raise ValueError(
                    f"{name} must match speeds_ms' {self.speeds_ms.shape}")
        for name in ("nval", "standing_thrust"):
            if getattr(self, name).shape != (ncurves,):
                raise ValueError(f"{name} must be ({ncurves},)")
        if (self.count < 0).any():
            raise ValueError("turbine counts must be non-negative")
        for t in range(ncurves):
            n = int(self.nval[t])
            if not 2 <= n <= maxvals:
                raise ValueError(f"curve {t}: nval {n} outside [2, {maxvals}]")
            ws = self.speeds_ms[t, :n]
            if not (np.diff(ws) > 0).all():
                raise ValueError(f"curve {t}: table speeds must ascend")
            if ws[0] <= 0:
                # dragcof divides by the bracketing speeds cubed.
                raise ValueError(f"curve {t}: cut-in speed must be positive")
            if (self.speeds_ms[t, n:] != 0).any():
                # dragcof's search runs over the whole row; a nonzero pad
                # would be found as a table node past cut-out.
                raise ValueError(f"curve {t}: speeds past nval must be 0")
        on = self.count > 0
        curves = self.curve[on]
        if ((curves < 0) | (curves >= ncurves)).any():
            raise ValueError(f"turbine curves must lie in [0, {ncurves})")
        dia = self.rotor_diameter_m[on]
        if (dia <= 0).any():
            raise ValueError("rotor diameters must be positive")
        if (self.hub_height_m[on] - dia / np.float32(2) < 0).any():
            # WRF's level search would leave k_turbine_bot = 0, off the array.
            raise ValueError("a rotor reaches below the ground "
                             "(hub_height_m < rotor_diameter_m/2)")

    @classmethod
    def single_type(cls, count, *, hub_height_m: float,
                    rotor_diameter_m: float, speeds_ms, thrust_coefficient,
                    power_kw, standing_thrust: float):
        """One group per cell with turbines, all of one curve, from a
        (ny, nx) grid of counts."""
        count = np.asarray(count, dtype=np.int32)
        ny, nx = count.shape
        cell = np.flatnonzero(count)
        n = cell.size
        return cls(ny=ny, nx=nx, cell=cell, count=count.ravel()[cell],
                   curve=np.zeros(n, np.int32),
                   hub_height_m=np.full(n, hub_height_m),
                   rotor_diameter_m=np.full(n, rotor_diameter_m),
                   speeds_ms=np.asarray(speeds_ms)[None, :],
                   thrust_coefficient=np.asarray(thrust_coefficient)[None, :],
                   power_kw=np.asarray(power_kw)[None, :],
                   nval=[len(speeds_ms)], standing_thrust=[standing_thrust])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    def groups_in(self, j: int, i: int) -> range:
        """The group indices of cell (j, i), in their given order."""
        c = j * self.nx + i
        return range(int(self.cell_offsets[c]), int(self.cell_offsets[c + 1]))

    def device_arrays(self) -> dict:
        """The layout on the current device, copied once per device."""
        import cupy as cp

        device = cp.cuda.Device().id
        arrays = self._device.get(device)
        if arrays is None:
            arrays = {name: cp.asarray(getattr(self, name))
                      for name in _DEVICE_FIELDS}
            self._device[device] = arrays
        return arrays


def module_source() -> str:
    """The exact source string NVRTC compiles."""
    from woof.core.kernels import _preamble

    root = Path(__file__).parent / "kernels"
    return _preamble() + "".join(
        (root / name).read_text(encoding="utf-8") for name in _SOURCES)


@lru_cache(maxsize=None)
def _module(device: int):
    """NVRTC directly, -fmad=false and --ftz=false, one module per device.

    Not ``cupy.RawModule``, which appends ``-ftz=true`` after the caller's
    options (the reasoning and measurements are in
    ``woof.core.terrain_drag._module``).
    """
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler

    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile

    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(
            source, MODULE_OPTIONS, None, "windfarm_fitch.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS,
                  module=None)
    return module


def _kernel(name: str):
    import cupy as cp

    return _module(cp.cuda.Device().id).get_function(name)


def _f32(xp, array, shape, name):
    out = xp.ascontiguousarray(xp.asarray(array, dtype=xp.float32))
    if out.shape != tuple(shape):
        raise ValueError(f"{name} has shape {out.shape}, expected {shape}")
    return out


def _target(array, shape, name):
    """A caller's in-place float32 array, refused rather than copied."""
    if array.shape != tuple(shape):
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if array.dtype != np.float32 or not array.flags.c_contiguous:
        raise TypeError(f"{name} is added into in place and must be a "
                        "C-contiguous float32 array")
    return array


def launch_fitch(u, v, z_interface, layout: FitchLayout, du, dv, *,
                 dx: float, dy: float | None = None, qke=None,
                 dt: float | None = None,
                 tke_factor: float = WINDFARM_TKE_FACTOR, rho=None,
                 power=None, diagnostics=None):
    """Add the wind-farm drag to ``du``/``dv`` in place; return the power.

    ``u``, ``v`` are mass-point winds (nz, ny, nx), m s-1, as the PBL
    driver's ``u_phy``/``v_phy``; ``z_interface`` the full-level heights
    (nz+1, ny, nx), m above sea level (WRF's ``z_at_w``; the layer depths
    are its differences, WRF's ``dz8w``).  ``du``/``dv`` (m s-2) and
    ``qke`` (QKE = 2*TKE, m2 s-2, with ``dt`` the PBL step in s) are
    float32 arrays added into in place, as WRF adds into RUBLTEN, RVBLTEN
    and QKE.  Returns ``power`` (ny, nx), W, WRF's POWER summed over the
    cell's groups, written into the array given or a new one.

    ``diagnostics``, when a dict, receives ``overlap`` (nz, ny, nx), each
    level's fraction of the cell's swept area (count-weighted over its
    groups; one rotor's fraction for a uniform cell), and with ``rho``
    (nz, ny, nx, kg m-3) ``thrust`` (ny, nx), N, the momentum the cell's
    rotors remove per second (rho*|dV/dt|*dx*dy*dz over the rotor levels).
    """
    import cupy as cp

    for name, array in (("du", du), ("dv", dv), ("qke", qke),
                        ("power", power)):
        if array is not None and not isinstance(array, cp.ndarray):
            raise TypeError(f"{name} is added into on the device and must be "
                            "a cupy array")
    if du.ndim != 3:
        raise ValueError(f"du must be (nz, ny, nx), got {du.shape}")
    nz, ny, nx = du.shape
    if layout.shape != (ny, nx):
        raise ValueError(f"layout is {layout.shape}, tendencies {(ny, nx)}")
    _target(du, (nz, ny, nx), "du")
    _target(dv, (nz, ny, nx), "dv")
    if qke is not None:
        _target(qke, (nz, ny, nx), "qke")
        if dt is None:
            raise ValueError("qke is incremented over a step: dt is required")
    if rho is not None and diagnostics is None:
        raise ValueError("rho feeds only the thrust diagnostic; pass a "
                         "diagnostics dict to receive it")
    u = _f32(cp, u, (nz, ny, nx), "u")
    v = _f32(cp, v, (nz, ny, nx), "v")
    z_w = _f32(cp, z_interface, (nz + 1, ny, nx), "z_interface")
    if power is None:
        power = cp.empty((ny, nx), dtype=cp.float32)
    else:
        _target(power, (ny, nx), "power")
    thrust = overlap = None
    if diagnostics is not None:
        overlap = cp.empty((nz, ny, nx), dtype=cp.float32)
        if rho is not None:
            rho = _f32(cp, rho, (nz, ny, nx), "rho")
            thrust = cp.empty((ny, nx), dtype=cp.float32)
    dy = dx if dy is None else dy
    # wfdensity = 1.0/(dx*dx)  (module_wind_fitch.F:109)
    wfdensity = np.float32(np.float32(1.0)
                           / (np.float32(dx) * np.float32(dy)))
    lay = layout.device_arrays()
    maxvals = layout.speeds_ms.shape[1]
    n = ny * nx
    _kernel("fitch_column")(
        ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
        (u, v, z_w, rho if thrust is not None else _NULL,
         lay["cell_offsets"], lay["count"], lay["curve"],
         lay["hub_height_m"], lay["rotor_diameter_m"], lay["speeds_ms"],
         lay["thrust_coefficient"], lay["power_kw"], lay["nval"],
         lay["standing_thrust"], du, dv,
         qke if qke is not None else _NULL, power,
         thrust if thrust is not None else _NULL,
         overlap if overlap is not None else _NULL,
         wfdensity, np.float32(0.0 if dt is None else dt),
         np.float32(tke_factor), PICONST,
         np.int32(nz), np.int32(ny), np.int32(nx), np.int32(maxvals)))
    if diagnostics is not None:
        diagnostics["overlap"] = overlap
        if thrust is not None:
            diagnostics["thrust"] = thrust
    return power


def rotor_overlap(z_interface, layout: FitchLayout):
    """(nz, ny, nx): each level's fraction of the cell's swept area.

    Zero outside the rotor levels and in cells without turbines; a group
    whose rotor reaches above the column top is left out, as WRF skips it.
    """
    import cupy as cp

    z_w = cp.asarray(z_interface, dtype=cp.float32)
    nz = z_w.shape[0] - 1
    ny, nx = layout.shape
    zeros = cp.zeros((nz, ny, nx), dtype=cp.float32)
    diagnostics: dict = {}
    launch_fitch(zeros, zeros, z_w, layout, zeros.copy(), zeros.copy(),
                 dx=1.0, diagnostics=diagnostics)
    return diagnostics["overlap"]
