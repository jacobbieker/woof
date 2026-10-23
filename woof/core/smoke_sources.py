"""The smoke tracer's injection profile and its removal (sources and sinks).

:mod:`woof.core.smoke_tracer` carries ``state.smoke`` and adds the emission
(``state.smoke_emis``, kg m-2 s-1: the prescribed ``smoke_emission_file``
today, a coupled fire model tomorrow) to the lowest layer.  This module
routes that same emission through a vertical profile and removes smoke at
the ground, using the merged :mod:`woof.smoke` library's WRF v4.7.1 ports
unchanged; every kernel here only forms their inputs or applies their
outputs.

INJECTION (``smoke_injection``).  The emission enters the held forward
tendency ``smag_rsmoke`` that every RK stage reads, as a coupled source
``F * frac(k) * 1e9 * g / |dnw(k)|`` (:func:`prepare_injection`): the column
receives exactly ``F`` whatever ``frac`` is, because ``frac`` sums to one.

* ``surface`` (the default): smoke_tracer's own lowest-layer source, the
  identical launch and words, so a default run is byte-for-byte the run
  before this module existed.
* ``pbl``: the library's ``pbl_uniform_profile`` (uniform in height from
  the ground to the PBL height of the scheme's last call), every step.
* ``plumerise``: the WRF-Chem Freitas plume model
  (:mod:`woof.smoke.plume_rise`), run per burning column on the fire's own
  heat flux and burning area and cached as ``state.smoke_inj_frac`` between
  calls; WRF's ``plumerisefire_frq`` schedule, :func:`plume_due`.  The
  profile is WRF's ``plumerise_driver`` conversion under
  ``scale_fire_emiss = .true.``: the flaming share ``mean_fct`` injected
  per metre between the plume tops, the smouldering remainder at the
  surface, the whole renormalized so the column takes the emission's total.

  The fire seam: when the fire model is coupled it writes, every step, the
  2-D device planes ``state.fire_heat_flux`` (total sensible heat flux,
  W m-2) and ``state.fire_area`` (burning area per cell, m2).  They are read
  by name (``getattr``) and never allocated here.  Without them the
  emission file's ``fire_heat_flux``/``fire_area`` variables are used, or
  failing those the scalars ``smoke_plume_heat_flux`` and
  ``smoke_plume_fire_area`` on every column whose emission is positive.
  The column's fire runs as WRF's single-heat-flux biome (aggregated
  biome 4): one plume, the flaming share from level 3 to the level above
  its top.

REMOVAL, after the step's transport and microphysics, where WRF-Chem's
``chem_driver`` runs ``dry_dep_driver`` (:func:`apply_sinks`):

* ``smoke_drydep``: Zhang et al. (2001), ``zhang2001_aer_drydepvel_1`` as
  ``aer_drydep_driver`` calls it for ``aer_drydep_opt`` 301/302/311/312
  (USGS classes, MODIS mapped through WRF's ``luse2usgs``), its mass
  velocity capped at 0.5 m s-1 as ``dry_dep_driver`` caps it, on the
  aerodynamic resistance from u*, z0 and the Obukhov length; applied to the
  lowest layer through the library's surface tendency, one forward step,
  held non-negative.
* ``smoke_settling``: GOCART ``settling`` through the whole column, the
  tracer as one bin of radius ``smoke_particle_radius`` and density
  ``smoke_particle_density`` (GOCART's finest dust bin by default).

Both book the column mass they remove, ``old - new`` in the tracer's own
mass metric ``C(mu) |dnw| / g`` as ``dry_dep_driver`` books its ddmass,
into ``state.smoke_deposited`` (ug m-2; settling also into
``state.smoke_graset``), so emitted = airborne + deposited + outflow
closes to rounding.

ORDER IN THE STEP, against WRF-Chem (``dyn_em/solve_em.F`` calls
``chem_driver`` after the RK loop and microphysics; ``chem_driver`` runs
``emissions_driver`` -- plumerise at its frequency, then the emission --
then ``dry_dep_driver``: deposition velocities, ``vertmx`` with them as the
lower boundary, then GOCART settling):

1. physics (woof's PBL mixing of the tracer, ``smoke_tracer.mix_pbl``);
2. the profile and the emission tendency, from the time-t state
   (WRF: from the post-transport state of the step before -- one step's
   lag);
3. RK3 transport with the held tendency;
4. microphysics;
5. dry deposition, then settling -- WRF's order inside ``dry_dep_driver``.
   WRF applies the deposition velocity as ``vertmx``'s lower boundary;
   woof's tracer mixes in the PBL call instead, so the velocity is applied
   as the library's lowest-layer tendency.  WRF skips ``dry_dep_driver``
   on its first two steps (``ktau > 2``), which only waits for the surface
   layer's first u*; woof has u* from the first step and removes from it.

COST.  ``pbl`` is two column launches per step.  ``plumerise`` is one
launch over every column (non-burning columns return at once) at about
0.09 ms per burning column on an RTX A6000, every
``smoke_plumerise_interval_s`` (WRF's default 180 min), plus one column
launch per step.  Deposition is six launches over the surface plane and
settling one column launch, every step.  No step reads anything back to
the host.
"""
from __future__ import annotations

import math
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

DTYPE = np.float32
F = np.float32

#: The injection profiles ``smoke_injection`` selects.
INJECTION_MODES = ("surface", "pbl", "plumerise")
#: The ``aer_drydep_opt`` values whose Zhang (2001) path is ported
#: (module_aer_drydep.F:319-354): 30x season by date, 3x2 summer always;
#: 3x1 the published coefficients, 31x the modified ones.
AER_DRYDEP_OPTS = (301, 302, 311, 312)
#: WRF-Chem's plumerisefire_frq default (Registry/registry.chem), 180 min.
PLUMERISE_INTERVAL_S = 10800.0
#: GOCART's finest dust bin (phys/module_data_gocart_dust.F:18-19),
#: effective radius (m) and density (kg m-3): the PM2.5 bin.
GOCART_PM25_RADIUS = 0.73e-6
GOCART_PM25_DENSITY = 2500.0
#: GOCART's dynamic viscosity and WRF's g, as dry_dep_driver hands them
#: to settling.
GOCART_DYN_VISC = 1.5e-5
WRF_G = 9.81
#: Every RunConfig field this module reads, at its default: the run before
#: the sources and sinks existed.  The identity and checkpoint writers drop
#: each key at this value (woof.io.restart, woof.core.model), and a
#: checkpoint without it compares as this value.
SMOKE_SOURCE_DEFAULTS = {
    "smoke_injection": "surface",
    "smoke_plumerise_interval_s": PLUMERISE_INTERVAL_S,
    "smoke_plume_heat_flux": 0.0,
    "smoke_plume_fire_area": 0.0,
    "smoke_plume_flaming_fraction": 1.0,
    "smoke_drydep": False,
    "smoke_aer_drydep_opt": 301,
    "smoke_settling": False,
    "smoke_particle_radius": GOCART_PM25_RADIUS,
    "smoke_particle_density": GOCART_PM25_DENSITY,
}
#: The plume model's column limits (woof.smoke.plume_rise: 3 to nkp - 2).
PLUME_MIN_LEVELS = 3
PLUME_MAX_LEVELS = 198

#: share/module_model_constants.F, folded in real(4).
_R_D = F(287.0)
_CP = F(F(F(7.0) * _R_D) / F(2.0))
_RCP = F(_R_D / _CP)
_P1000 = F(100000.0)
_G = F(WRF_G)

MODULE_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
MODULE_KEY = "woof.core.smoke_sources:smoke_sources"
#: The composed unit: glibc's powf, the WRF-Chem plume model's device code
#: (for the fire-driven plume entry) and this module's glue.
SOURCES = ("glibc_flt32.cuh", "smoke_plumerise.cu", "smoke_sources.cu")
_BLOCK = 128

#: The infra attribute recording which emission file was read for plume
#: fields, as ``(path, carries_them)``; when it carries them they are in
#: the ``smoke_src_heat``/``smoke_src_area`` slots.  A provenance marker,
#: re-read after a resume as the emission is.
_PLUME_BOUND_ATTRIBUTE = "_smoke_plume_bound"
#: The emission file's optional plume variables (NetCDF only).
HEAT_FLUX_VARIABLE = "fire_heat_flux"
FIRE_AREA_VARIABLE = "fire_area"


def injection(cfg) -> str:
    """The configured injection profile (``surface`` when the tracer is off)."""
    if not getattr(cfg, "smoke_tracer", False):
        return "surface"
    return str(getattr(cfg, "smoke_injection", "surface"))


def drydep_on(cfg) -> bool:
    return bool(getattr(cfg, "smoke_tracer", False)
                and getattr(cfg, "smoke_drydep", False))


def settling_on(cfg) -> bool:
    return bool(getattr(cfg, "smoke_tracer", False)
                and getattr(cfg, "smoke_settling", False))


def sinks_on(cfg) -> bool:
    return drydep_on(cfg) or settling_on(cfg)


def state_fields(cfg) -> dict[str, str]:
    """The state arrays the sources and sinks allocate for ``cfg``.

    ``{name: "mass" | "surface"}``, the one gate state.py and
    device_inventory both read; empty without the tracer or at the
    defaults, so no other inventory moves.
    """
    if not getattr(cfg, "smoke_tracer", False):
        return {}
    mode = injection(cfg)
    out: dict[str, str] = {}
    if mode == "plumerise":
        out["smoke_inj_frac"] = "mass"
    if mode in ("pbl", "plumerise"):
        out["smoke_inj_top"] = "surface"
    if drydep_on(cfg):
        out["smoke_drydep_vel"] = "surface"
    if sinks_on(cfg):
        out["smoke_deposited"] = "surface"
    if settling_on(cfg):
        out["smoke_graset"] = "surface"
    return out


def zhang_optaa(opt: int) -> int:
    """aer_drydep_driver's ioptaa: 2 for 311/312, else 1."""
    return 2 if int(opt) in (311, 312) else 1


def zhang_season(opt: int, start_time, elapsed_seconds: float) -> int:
    """aer_drydep_driver's iseason (module_aer_drydep.F:334-339).

    1 (summer), or 2 when the valid date's julian day is before 90 or after
    270 -- for 302/312 always 1.
    """
    if int(opt) in (302, 312):
        return 1
    if start_time is None:
        raise RuntimeError(
            f"smoke_aer_drydep_opt = {int(opt)} sets Zhang's season from the "
            "valid date, and this domain's physics has no start time; use "
            "302 or 312 (summer always), or give the domain its start time")
    valid = start_time + timedelta(seconds=float(elapsed_seconds))
    julday = int(valid.timetuple().tm_yday)
    return 2 if (julday < 90 or julday > 270) else 1


class SmokeSurface(NamedTuple):
    """What the deposition reads of a domain's surface beyond its fields.

    Attached to the physics driver as ``smoke_surface`` by
    initialize_physics (rebuilt from the same arguments on a resume).
    """

    #: The land-use dataset ``ivgtyp`` indexes ("USGS" or
    #: "MODIFIED_IGBP_MODIS_NOAH").
    dataset: str
    #: The domain's UTC start time, or None (Zhang's season by date then
    #: refuses).
    start_time: object


LANDUSE_DATASETS = ("USGS", "MODIFIED_IGBP_MODIS_NOAH")


def smoke_surface(cfg, landuse_dataset, start_time) -> SmokeSurface | None:
    """The driver attachment for ``cfg``, or None when no deposition runs.

    Refuses, at setup rather than at the first deposition step, a land use
    the Zhang tables cannot read and a dated season with no start date.
    """
    if not drydep_on(cfg):
        return None
    opt = int(getattr(cfg, "smoke_aer_drydep_opt", 301))
    if start_time is None and opt not in (302, 312):
        from woof.explain import layered
        raise ValueError(layered(
            f"smoke_aer_drydep_opt = {opt} dates Zhang's season from the "
            "valid time, and this domain has no start time; use 302 or 312 "
            "(summer always), or give the domain its start time.",
            "aer_drydep_driver sets iseason from the julian day."))
    if landuse_dataset not in LANDUSE_DATASETS:
        from woof.explain import layered
        raise ValueError(layered(
            f"smoke_drydep = true reads the USGS land-use classes, and this "
            f"domain's land use is {landuse_dataset!r}.",
            "WRF-Chem's dry deposition admits USGS and MODIS_NOAH maps only "
            "(chemics_init.F:484-487); MODIS classes are mapped to USGS by "
            "dep_init's luse2usgs table."))
    return SmokeSurface(str(landuse_dataset), start_time)


# --------------------------------------------------------------------------
# the NVRTC unit


def module_source() -> str:
    """The exact source string NVRTC compiles."""
    from woof.core.kernels import _preamble

    root = Path(__file__).parent / "kernels"
    return _preamble() + "".join(
        (root / name).read_text(encoding="utf-8") for name in SOURCES)


@lru_cache(maxsize=None)
def _module(device: int):
    """NVRTC directly, -fmad=false and --ftz=false, one module per device.

    The woof.core.terrain_drag loader, for its reason: ``cupy.RawModule``
    appends ``-ftz=true`` after the caller's options.
    """
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler

    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile

    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(
            source, MODULE_OPTIONS, None, "smoke_sources.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS,
                  module=None)
    return module


def _kernel(name: str):
    import cupy as cp

    return _module(cp.cuda.Device().id).get_function(name)


def _library(name: str):
    from woof.smoke import _kernels

    return _kernels.kernel(name)


def _blocks(n: int) -> tuple[int]:
    return ((int(n) + _BLOCK - 1) // _BLOCK,)


def _null():
    return np.uintp(0)


# --------------------------------------------------------------------------
# the ambient column


def _env(state, *, plume=False, zw=False, sinks=False) -> dict:
    """One ``smoke_source_env`` launch filling the requested fields.

    ``plume``: u, v, theta, pi, qv, zt, zw, dz (the plume model's inputs);
    ``zw``: the interface heights above ground; ``sinks``: t, rho, dz, zref.
    Each field lives in its registered ``smoke_src_*`` slot.
    """
    nz, ny, nx = state.p.shape
    m = (nz, ny, nx)
    out = {}
    if plume:
        out["u"] = state.scratch(m, "smoke_src_u")
        out["v"] = state.scratch(m, "smoke_src_v")
        out["theta"] = state.scratch(m, "smoke_src_theta")
        out["pi"] = state.scratch(m, "smoke_src_pi")
        out["qv"] = state.scratch(m, "smoke_src_qv")
        out["zt"] = state.scratch(m, "smoke_src_zt")
    if plume or zw:
        out["zw"] = state.scratch((nz + 1, ny, nx), "smoke_src_zw")
    if plume or sinks:
        out["dz"] = state.scratch(m, "smoke_src_dz")
    if sinks:
        out["t"] = state.scratch(m, "smoke_src_t")
        out["rho"] = state.scratch(m, "smoke_src_rho")
        out["zref"] = state.scratch((ny, nx), "smoke_src_zref")

    def ptr(name):
        return out.get(name, _null())

    qv = getattr(state, "qv", None)
    _kernel("smoke_source_env")(
        _blocks(ny * nx), (_BLOCK,),
        (state.u, state.v, state.thb, np.int32(state.thb.ndim == 3),
         state.thp, state.p, _null() if qv is None else qv, state.phb,
         np.int32(state.phb.ndim == 3), state.php, state.alt,
         _G, _RCP, _CP, _P1000,
         ptr("u"), ptr("v"), ptr("theta"), ptr("pi"), ptr("qv"), ptr("zt"),
         ptr("zw"), ptr("dz"), ptr("t"), ptr("rho"), ptr("zref"),
         np.int32(nz), np.int32(ny), np.int32(nx)))
    return out


def env_reference(u, v, thb, thp, p, qv, phb, php, alt) -> dict:
    """NumPy twin of ``smoke_source_env``: every field, in its statements."""
    from woof.core import noahmp_libm

    thp = np.asarray(thp, F)
    nz, ny, nx = thp.shape
    phb = np.asarray(phb, F)
    phb3 = phb if phb.ndim == 3 else np.broadcast_to(
        phb[:, None, None], (nz + 1, ny, nx))
    thb = np.asarray(thb, F)
    thb3 = thb if thb.ndim == 3 else np.broadcast_to(
        thb[:, None, None], (nz, ny, nx))
    z = (phb3 + np.asarray(php, F)) / _G
    zs = z[0]
    zw = (z - zs[None]).astype(F)
    dz = (z[1:] - z[:-1]).astype(F)
    zt = (F(0.5) * (z[:-1] + z[1:]) - zs[None]).astype(F)
    q = np.zeros((nz, ny, nx), F) if qv is None else np.asarray(qv, F)
    theta = thb3 + thp
    ratio = noahmp_libm.powf_array(np.asarray(p, F) / _P1000,
                                   np.full((nz, ny, nx), _RCP, F))
    t_phy = (theta * ratio).astype(F)
    pi_in = (_CP * ratio).astype(F)
    u = np.asarray(u, F)
    v = np.asarray(v, F)
    return {
        "u": (F(0.5) * (u[:, :, :-1] + u[:, :, 1:])).astype(F),
        "v": (F(0.5) * (v[:, :-1, :] + v[:, 1:, :])).astype(F),
        "theta": ((t_phy / pi_in) * _CP).astype(F), "pi": pi_in,
        "qv": q.copy(), "zt": zt, "zw": zw, "dz": dz, "t": t_phy,
        "rho": ((F(1.0) / np.asarray(alt, F)) * (F(1.0) + q)).astype(F),
        "zref": zt[0].copy()}


# --------------------------------------------------------------------------
# injection


def plume_due(state, cfg) -> bool:
    """Whether this step runs the plume model (emissions_driver.F:681-698).

    WRF calls ``plumerise_driver`` on the domain's first step (``ktau ==
    1``) and then whenever the step ends at or past the next multiple of
    ``plumerisefire_frq``: ``curr_secs + dt + 0.01 >= INT(curr_secs/frq +
    1) * frq``, ``curr_secs`` the domain's own time.  A pure function of the
    clock, so a resumed run calls it on the same steps as the run it
    continues; between calls the serialized ``smoke_inj_frac`` is used.
    """
    tau = (float(state.elapsed_seconds)
           - float(getattr(state, "domain_start_offset", 0.0) or 0.0))
    dt = float(cfg.dt)
    if tau < 0.5 * dt:
        return True
    frq = float(getattr(cfg, "smoke_plumerise_interval_s",
                        PLUMERISE_INTERVAL_S))
    if frq <= 0.0:
        return True
    return tau + dt + 0.01 >= (math.floor(tau / frq) + 1.0) * frq


#: The plume variables' accepted units (the first the default when the
#: variable has none) and their factor to W m-2 and m2.
_PLUME_UNITS = {
    HEAT_FLUX_VARIABLE: ("W m-2", {"W m-2": 1.0, "W/m2": 1.0,
                                   "W m**-2": 1.0, "kW m-2": 1.0e3,
                                   "kW/m2": 1.0e3}),
    FIRE_AREA_VARIABLE: ("m2", {"m2": 1.0, "m^2": 1.0, "m**2": 1.0}),
}


def load_plume_fields(path, shape) -> tuple[np.ndarray, np.ndarray] | None:
    """The emission file's ``fire_heat_flux`` (W m-2) and ``fire_area``
    (m2), as host FP32 planes, or None when the file carries neither.

    NetCDF only (a ``.npy`` emission carries no plume fields).  Both or
    neither: one without the other is refused by name, as is a wrong shape,
    a masked, negative or non-finite value.
    """
    from woof.explain import layered

    source = Path(path).expanduser()
    if source.suffix.lower() != ".nc":
        return None
    from netCDF4 import Dataset

    with Dataset(source, "r") as dataset:
        present = [name for name in (HEAT_FLUX_VARIABLE, FIRE_AREA_VARIABLE)
                   if name in dataset.variables]
        if not present:
            return None
        if len(present) == 1:
            raise ValueError(layered(
                f"smoke_emission_file {str(path)!r} has {present[0]!r} but "
                "not its partner; the plume model needs both "
                f"{HEAT_FLUX_VARIABLE} (W m-2) and {FIRE_AREA_VARIABLE} (m2).",
                "A heat flux without a burning area (or the reverse) has no "
                "plume to rise."))
        planes = []
        for name in (HEAT_FLUX_VARIABLE, FIRE_AREA_VARIABLE):
            variable = dataset.variables[name]
            units = str(getattr(variable, "units",
                                _PLUME_UNITS[name][0])).strip()
            scale = _PLUME_UNITS[name][1].get(units)
            if scale is None:
                raise ValueError(layered(
                    f"smoke_emission_file {str(path)!r} gives {name} in "
                    f"{units!r}; use one of "
                    f"{', '.join(repr(u) for u in _PLUME_UNITS[name][1])}.",
                    "A burnt fraction or another unit read as these would "
                    "lift the plume by an unknown factor."))
            raw = variable[...]
            if np.ma.is_masked(raw):
                raise ValueError(
                    f"smoke_emission_file {str(path)!r} has missing "
                    f"(_FillValue) cells in {name}; write 0 where nothing "
                    "burns")
            values = np.asarray(np.ma.getdata(raw), dtype=np.float64)
            if values.ndim == 3 and values.shape[0] == 1:
                values = values[0]
            if values.shape != tuple(shape):
                raise ValueError(
                    f"smoke_emission_file {str(path)!r} holds a "
                    f"{values.shape} {name}; this domain's mass grid is "
                    f"(ny, nx) = {tuple(shape)}")
            if not np.all(np.isfinite(values)) or bool((values < 0).any()):
                raise ValueError(
                    f"smoke_emission_file {str(path)!r}: {name} must be "
                    "finite and non-negative everywhere")
            planes.append((values * scale).astype(F))
    return planes[0], planes[1]


def _fire_planes(state, cfg):
    """The (heat flux, area) planes the plume reads, or (None, None).

    The fire model's seam first; then the emission file's own variables,
    read once into their slots; None means the prescribed scalars, which
    must then both be positive (refused by name otherwise).
    """
    heat = getattr(state, "fire_heat_flux", None)
    area = getattr(state, "fire_area", None)
    if heat is not None or area is not None:
        if heat is None or area is None:
            raise RuntimeError(
                "the fire coupling wrote only one of state.fire_heat_flux and "
                "state.fire_area; the plume model needs both")
        return heat, area
    path = getattr(cfg, "smoke_emission_file", "") or ""
    if path:
        ny, nx = state.smoke_emis.shape
        slot_heat = state.scratch((ny, nx), "smoke_src_heat")
        slot_area = state.scratch((ny, nx), "smoke_src_area")
        bound = getattr(state, _PLUME_BOUND_ATTRIBUTE, None)
        if not (isinstance(bound, tuple) and bound[0] == path):
            planes = load_plume_fields(path, (ny, nx))
            if planes is not None:
                import cupy as cp
                slot_heat[...] = cp.asarray(planes[0])
                slot_area[...] = cp.asarray(planes[1])
            bound = (path, planes is not None)
            setattr(state, _PLUME_BOUND_ATTRIBUTE, bound)
        if bound[1]:
            return slot_heat, slot_area
    if not (getattr(cfg, "smoke_plume_heat_flux", 0.0) > 0.0
            and getattr(cfg, "smoke_plume_fire_area", 0.0) > 0.0):
        from woof.explain import layered
        raise RuntimeError(layered(
            "smoke_injection = \"plumerise\" has no fire to raise a plume "
            "from: the fire model wrote no state.fire_heat_flux/fire_area, "
            "the emission file carries no fire_heat_flux/fire_area and "
            "smoke_plume_heat_flux/smoke_plume_fire_area are not both "
            "positive.",
            "Without a heat flux every column would be injected at the "
            "surface under the name of a plume-rise run."))
    return None, None


def run_plume(state, cfg) -> None:
    """The plume model over the domain, into ``smoke_inj_frac``/``_top``.

    The ambient column from the time-t state, the fire's planes
    (:func:`_fire_planes`), one ``smoke_plume_fire_columns`` launch (one
    block per column; non-burning columns return at once) and one
    ``smoke_plume_fraction`` launch.
    """
    from woof.smoke.plume_rise import SHARED_BYTES, THREADS

    nz, ny, nx = state.p.shape
    ncol = ny * nx
    env = _env(state, plume=True)
    heat, area = _fire_planes(state, cfg)
    eburn = state.scratch((nz, ny, nx), "smoke_src_eburn")
    ztop = state.scratch((ny, nx), "smoke_src_ztop")
    k1 = state.scratch((ny, nx), "smoke_src_k1", dtype=np.int32)
    k2 = state.scratch((ny, nx), "smoke_src_k2", dtype=np.int32)
    status = state.scratch((ny, nx), "smoke_src_status", dtype=np.int32)
    _kernel("smoke_plume_fire_columns")(
        (ncol,), (THREADS,),
        (np.int32(nz), np.int32(ncol), env["u"], env["v"], env["theta"],
         env["pi"], env["qv"], env["zt"], env["zw"],
         _null() if heat is None else heat,
         _null() if area is None else area, state.smoke_emis,
         F(getattr(cfg, "smoke_plume_heat_flux", 0.0)),
         F(getattr(cfg, "smoke_plume_fire_area", 0.0)),
         F(getattr(cfg, "smoke_plume_flaming_fraction", 1.0)),
         eburn, ztop, k1, k2, status),
        shared_mem=SHARED_BYTES)
    _kernel("smoke_plume_fraction")(
        _blocks(ncol), (_BLOCK,),
        (eburn, env["dz"], env["zw"], status, state.smoke_inj_frac,
         state.smoke_inj_top, np.int32(nz), np.int32(ncol)))


def plume_fraction_reference(eburn_out, dz, zw, status):
    """NumPy twin of ``smoke_plume_fraction``: (frac, inj_top)."""
    eburn_out = np.asarray(eburn_out, F)
    dz = np.asarray(dz, F)
    nz = eburn_out.shape[0]
    total = eburn_out[0].copy()
    for k in range(1, nz):
        total = (total + eburn_out[k] * dz[k]).astype(F)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total > 0, eburn_out[0] / total, F(0.0)).astype(F)
    frac = np.empty_like(eburn_out)
    frac[0] = eburn_out[0] * ratio
    for k in range(1, nz):
        frac[k] = (eburn_out[k] * dz[k]) * ratio
    failed = np.asarray(status) != 0
    frac[:, failed] = 0.0
    frac[0, failed] = 1.0
    top = _injection_top(frac, zw)
    top[failed] = np.nan
    return frac, top


def _injection_top(frac, zw):
    zw = np.asarray(zw, F)
    nz = frac.shape[0]
    top = np.zeros(frac.shape[1:], dtype=np.int64)
    for k in range(nz):
        top = np.where(frac[k] != 0, k, top)
    return np.take_along_axis(zw, (top + 1)[None], axis=0)[0].astype(F)


def run_pbl_profile(state, cfg):
    """The library's PBL-uniform profile as fractions, every step.

    ``smoke_inject_uniform`` (woof.smoke.injection.pbl_uniform_profile's
    kernel) on a unit flux and unit density from the ground to the PBL
    height, then ``smoke_pbl_fraction``.  Returns the fraction slot.
    """
    nz, ny, nx = state.p.shape
    ncol = ny * nx
    pblh = _physics_field(state, "pblh", "smoke_injection = \"pbl\"")
    env = _env(state, zw=True)
    unit = state.scratch((nz, ny, nx), "smoke_src_unit")
    unit.fill(1.0)
    flux = state.scratch((ny, nx), "smoke_src_flux1")
    flux.fill(1.0)
    source = state.scratch((nz, ny, nx), "smoke_src_pblsrc")
    frac = state.scratch((nz, ny, nx), "smoke_src_frac")
    zw = env["zw"]
    _library("smoke_inject_uniform")(
        _blocks(ncol), (_BLOCK,),
        (zw, unit, flux, zw[0], pblh, source, F(1.0), np.int32(nz),
         np.int32(ncol)))
    _kernel("smoke_pbl_fraction")(
        _blocks(ncol), (_BLOCK,),
        (source, zw, frac, state.smoke_inj_top, np.int32(nz),
         np.int32(ncol)))
    return frac


def pbl_fraction_reference(source, zw):
    """NumPy twin of ``smoke_pbl_fraction``: (frac, inj_top)."""
    source = np.asarray(source, F)
    zw = np.asarray(zw, F)
    frac = (source * (zw[1:] - zw[:-1])).astype(F)
    return frac, _injection_top(frac, zw)


def prepare_injection(state, cfg, held) -> None:
    """Add the emission through the configured profile to ``held``.

    ``surface`` is smoke_tracer.add_surface_source itself.  ``pbl`` forms
    its profile every step; ``plumerise`` refreshes the cached profile on
    :func:`plume_due` steps.  Then one ``smoke_profile_source`` launch.
    """
    from woof.core import smoke_tracer

    mode = injection(cfg)
    if mode == "surface":
        smoke_tracer.add_surface_source(state, held)
        return
    if mode == "pbl":
        frac = run_pbl_profile(state, cfg)
    elif mode == "plumerise":
        if plume_due(state, cfg):
            run_plume(state, cfg)
        frac = state.smoke_inj_frac
    else:
        raise ValueError(f"smoke_injection={mode!r} is not one of "
                         f"{INJECTION_MODES}")
    nz, ny, nx = state.p.shape
    _kernel("smoke_profile_source")(
        _blocks(ny * nx), (_BLOCK,),
        (held, state.smoke_emis, frac, state.dnw, smoke_tracer.UG_G,
         np.int32(nz), np.int32(ny * nx)))


def profile_source_reference(tend, emis, frac, dnw) -> np.ndarray:
    """NumPy twin of ``smoke_profile_source``: the new tendency words."""
    from woof.core.smoke_tracer import UG_G

    tend = np.array(tend, dtype=F)
    emis = np.asarray(emis, F)
    frac = np.asarray(frac, F)
    dnw = np.asarray(dnw, F)
    nz = frac.shape[0]
    empty = ~np.any(frac != 0, axis=0)
    surface = (emis * UG_G) / (-dnw[0])
    for k in range(nz):
        source = ((emis * frac[k]) * UG_G) / (-dnw[k])
        use = ~empty if k == 0 else (~empty & (frac[k] != 0))
        tend[k] = np.where(use, tend[k] + source, tend[k])
    tend[0] = np.where(empty, tend[0] + surface, tend[0])
    return tend


# --------------------------------------------------------------------------
# removal


def _physics_field(state, name, who):
    physics = getattr(state, "physics", None)
    fields = getattr(physics, "fields", None) or {}
    value = fields.get(name) if hasattr(fields, "get") else None
    if value is None:
        raise RuntimeError(
            f"{who} reads the physics field {name!r}, and this domain's "
            "physics publishes none (it needs a surface-layer and PBL "
            "scheme)")
    return value


def _layer_mass_args(state):
    return (state.c1h, state.c2h, state.dnw, state.mub2d, state.mup)


def apply_sinks(state, cfg) -> None:
    """Dry deposition, then settling, on the completed step's tracer.

    WRF's ``dry_dep_driver`` order.  The ambient fields are the completed
    step's (chem_prep after the RK loop and microphysics).
    """
    if not sinks_on(cfg):
        return
    env = _env(state, sinks=True)
    if drydep_on(cfg):
        _apply_drydep(state, cfg, env)
    if settling_on(cfg):
        _apply_settling(state, cfg, env)


def _apply_drydep(state, cfg, env) -> None:
    import cupy as cp

    nz, ny, nx = state.p.shape
    n = ny * nx
    s2 = (ny, nx)
    who = "smoke_drydep"
    ust = _physics_field(state, "ust", who)
    znt = _physics_field(state, "znt", who)
    ivgtyp = _physics_field(state, "ivgtyp", who)
    physics = state.physics
    rmol = physics.fields.get("rmol")
    surface = getattr(physics, "smoke_surface", None)
    if surface is None:
        raise RuntimeError(
            "smoke_drydep needs the physics driver's smoke_surface "
            "attachment (initialize_physics builds it); this driver has none")
    opt = int(getattr(cfg, "smoke_aer_drydep_opt", 301))
    season = zhang_season(opt, surface.start_time, state.elapsed_seconds)
    radius = float(getattr(cfg, "smoke_particle_radius", GOCART_PM25_RADIUS))
    density = float(getattr(cfg, "smoke_particle_density",
                            GOCART_PM25_DENSITY))
    dg = state.scratch(s2, "smoke_src_dg")
    lnsig = state.scratch(s2, "smoke_src_lnsig")
    dens = state.scratch(s2, "smoke_src_dens")
    airtemp = state.scratch(s2, "smoke_src_airtemp")
    airdens = state.scratch(s2, "smoke_src_airdens")
    obukhov = state.scratch(s2, "smoke_src_obukhov")
    iseason = state.scratch(s2, "smoke_src_season", dtype=np.int32)
    isurf = state.scratch(s2, "smoke_src_surftype", dtype=np.int32)
    iwet = state.scratch(s2, "smoke_src_wetsurf", dtype=np.int32)
    kinvisc = state.scratch(s2, "smoke_src_kinvisc")
    freepath = state.scratch(s2, "smoke_src_freepath")
    _kernel("smoke_drydep_prepare")(
        _blocks(n), (_BLOCK,),
        (env["t"], env["rho"], _null() if rmol is None else rmol,
         cp.ascontiguousarray(ivgtyp, dtype=cp.int32)
         if ivgtyp.dtype != cp.int32 else ivgtyp,
         np.int32(surface.dataset == "MODIFIED_IGBP_MODIS_NOAH"),
         np.int32(season), F(2.0 * radius), F(density),
         dg, lnsig, dens, airtemp, airdens, obukhov, iseason, isurf, iwet,
         kinvisc, freepath, np.int32(n)))
    resist_a = state.scratch(s2, "smoke_src_resist_a")
    _library("smoke_deposition_resist_a")(
        _blocks(n), (_BLOCK,),
        (ust, znt, env["zref"], obukhov, resist_a, np.int32(n)))
    d0 = state.scratch(s2, "smoke_src_depvel0")
    d3 = state.scratch(s2, "smoke_src_depvel3")
    r0 = state.scratch(s2, "smoke_src_resist_s0")
    r3 = state.scratch(s2, "smoke_src_resist_s3")
    v0 = state.scratch(s2, "smoke_src_vsettl0")
    v3 = state.scratch(s2, "smoke_src_vsettl3")
    _library("smoke_deposition_zhang2001")(
        _blocks(n), (_BLOCK,),
        (dg, lnsig, dens, airtemp, airdens, ust, resist_a, iseason, isurf,
         iwet, kinvisc, freepath, d0, d3, r0, r3, v0, v3,
         np.int32(zhang_optaa(opt)), np.int32(2), np.int32(1), np.int32(1),
         np.int32(n)))
    _kernel("smoke_drydep_cap")(
        _blocks(n), (_BLOCK,), (d3, state.smoke_drydep_vel, np.int32(n)))
    tendency = state.scratch(s2, "smoke_src_drydep_tend")
    flux = state.scratch(s2, "smoke_src_drydep_flux")
    _library("smoke_deposition_surface_tendency")(
        _blocks(n), (_BLOCK,),
        (state.smoke[0], env["dz"][0], state.smoke_drydep_vel,
         env["rho"][0], tendency, flux, np.int32(n)))
    _kernel("smoke_drydep_apply")(
        _blocks(n), (_BLOCK,),
        (state.smoke, tendency, *_layer_mass_args(state),
         state.smoke_deposited, F(cfg.dt), _G, np.int32(n)))


@lru_cache(maxsize=None)
def _bin_constants(device: int, radius: float, density: float):
    """The one bin's REAL*8 radius and density, on the device once."""
    import cupy as cp

    with cp.cuda.Device(device):
        return (cp.asarray(np.array([radius], dtype=np.float64)),
                cp.asarray(np.array([density], dtype=np.float64)))


def _apply_settling(state, cfg, env) -> None:
    import cupy as cp

    nz, ny, nx = state.p.shape
    n = ny * nx
    radius = float(getattr(cfg, "smoke_particle_radius", GOCART_PM25_RADIUS))
    density = float(getattr(cfg, "smoke_particle_density",
                            GOCART_PM25_DENSITY))
    reff, den = _bin_constants(cp.cuda.Device().id, radius, density)
    # The routine's REAL*8 results: a per-call transient (priced by
    # woof.core.preflight.smoke_sources_transient_shapes), since the
    # shared scratch arena is float32-only.
    tc = cp.empty((nz, ny, nx), dtype=cp.float64)
    graset = cp.empty((ny, nx), dtype=cp.float64)
    grasetvel = cp.empty((ny, nx), dtype=cp.float64)
    _library("smoke_deposition_gocart_settling")(
        _blocks(n), (_BLOCK,),
        (state.smoke, env["t"], state.p, env["dz"], env["rho"], reff, den,
         tc, graset, grasetvel, F(cfg.dt), F(WRF_G), F(GOCART_DYN_VISC),
         np.int32(0), np.int32(1), np.int32(nz), np.int32(n)))
    _kernel("smoke_settling_apply")(
        _blocks(n), (_BLOCK,),
        (state.smoke, tc, *_layer_mass_args(state), state.smoke_deposited,
         state.smoke_graset, _G, np.int32(nz), np.int32(n)))
    del tc, graset, grasetvel


def _layer_mass_reference(c1h, c2h, dnw, mub, mup, k):
    mu = np.asarray(mub, F) + np.asarray(mup, F)
    chm = np.asarray(c1h, F)[k] * mu + np.asarray(c2h, F)[k]
    return (chm * ((-np.asarray(dnw, F)[k]) / _G)).astype(F)


def drydep_prepare_reference(t_phy0, rho0, rmol, ivgtyp, *, modis, iseason,
                             diameter, density) -> dict:
    """NumPy twin of ``smoke_drydep_prepare``."""
    shape = np.shape(t_phy0)
    lu = np.asarray(ivgtyp, np.int32)
    if modis:
        table = np.array([14, 13, 12, 11, 15, 8, 9, 10, 10, 7, 17, 4, 1, 5,
                          24, 19, 16, 21, 22, 23, 16], np.int32)
        inside = (lu >= 1) & (lu <= 21)
        lu = np.where(inside, table[np.clip(lu, 1, 21) - 1], 0)
    with np.errstate(divide="ignore"):
        obukhov = (np.full(shape, np.inf, F) if rmol is None
                   else (F(1.0) / np.asarray(rmol, F)).astype(F))
    return {"dg_num": np.full(shape, F(diameter), F),
            "lnsig": np.zeros(shape, F),
            "dens": np.full(shape, F(density), F),
            "airtemp": np.asarray(t_phy0, F).copy(),
            "airdens": np.asarray(rho0, F).copy(),
            "obukhov": obukhov, "season": np.full(shape, iseason, np.int32),
            "surftype": lu.astype(np.int32),
            "wetsurf": np.zeros(shape, np.int32),
            "kinvisc": np.zeros(shape, F), "freepath": np.zeros(shape, F)}


def drydep_cap_reference(depvel) -> np.ndarray:
    depvel = np.asarray(depvel, F)
    return np.where(F(0.5) < depvel, F(0.5), depvel).astype(F)


def drydep_apply_reference(smoke0, tendency, c1h, c2h, dnw, mub, mup,
                           deposited, dt):
    """NumPy twin of ``smoke_drydep_apply``: (new lowest layer, deposited)."""
    q0 = np.asarray(smoke0, F)
    qn = (q0 + F(dt) * np.asarray(tendency, F)).astype(F)
    qn = np.where(qn < 0, F(0.0), qn).astype(F)
    layer = _layer_mass_reference(c1h, c2h, dnw, mub, mup, 0)
    dep = (np.asarray(deposited, F) + layer * (q0 - qn)).astype(F)
    return qn, dep


def settling_apply_reference(smoke, tc, c1h, c2h, dnw, mub, mup, deposited,
                             graset):
    """NumPy twin of ``smoke_settling_apply``: (smoke, deposited, graset)."""
    smoke = np.asarray(smoke, F)
    tc = np.asarray(tc, np.float64)
    nz = smoke.shape[0]
    new = tc.astype(F)
    loss = np.zeros(smoke.shape[1:], np.float64)
    for k in range(nz):
        layer = _layer_mass_reference(c1h, c2h, dnw, mub, mup, k)
        loss = loss + layer.astype(np.float64) * (
            smoke[k].astype(np.float64) - new[k].astype(np.float64))
    booked = np.where(loss > 0.0, loss, 0.0).astype(F)
    return (new, (np.asarray(deposited, F) + booked).astype(F),
            (np.asarray(graset, F) + booked).astype(F))


__all__ = ["SMOKE_SOURCE_DEFAULTS", "AER_DRYDEP_OPTS", "GOCART_PM25_DENSITY", "GOCART_PM25_RADIUS",
           "INJECTION_MODES", "LANDUSE_DATASETS", "MODULE_KEY",
           "MODULE_OPTIONS", "PLUMERISE_INTERVAL_S", "SmokeSurface",
           "apply_sinks", "drydep_on", "env_reference", "injection",
           "load_plume_fields", "module_source", "plume_due",
           "prepare_injection", "run_plume", "settling_on", "sinks_on",
           "smoke_surface", "zhang_optaa", "zhang_season"]
