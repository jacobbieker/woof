"""MPAS-Ocean v8.4.2: CVMix vertical mixing coefficients, on the GPU (RKIND 8).

:func:`coefs_build` is ``ocn_vmix_coefs_cvmix_build`` (``mpas_ocn_vmix_cvmix.F``)
for one block, with the CVMix v0.84-beta routines it reaches, in the
Fortran's order and halo extents:

1. ``vertViscTopOfCell``, ``vertDiffTopOfCell`` and ``vertNonLocalFlux``
   zeroed over ``nCellsArray(4)``, CVMix's background added over
   ``nCellsArray(3)`` (``config_use_cvmix_background``);
2. KPP's first stage over ``nCellsArray(3)`` (``config_use_cvmix_kpp``): the
   bulk Richardson numbers of a boundary layer at each cell bottom (surface
   layer means of the velocity and potential density, the unresolved shear),
   the boundary-layer depth where they cross
   ``config_cvmix_kpp_criticalBulkRichardsonNumber`` (or
   ``config_cvmix_kpp_boundary_layer_depth`` with
   ``config_use_cvmix_fixed_boundary_layer``) and its limits;
3. the boundary-layer depth smoothed over ``nCellsArray(2)``
   (``config_cvmix_use_BLD_smoothing``);
4. the second stage over ``nCellsArray(3)``, or ``nCellsArray(2)`` after the
   smoothing (a quirk, below): the Richardson number smoothed
   (``config_cvmix_num_ri_smooth_loops``), shear mixing (``PP`` less the
   background, or ``KPP``), KPP's boundary-layer diffusivities
   (``SimpleShapes`` or ``MatchBoth``, the enhanced diffusivity at the layer's
   base) and non-local flux, convection (``config_use_cvmix_convection``,
   Brunt-Vaisala based), the no-flux top and bottom.

``woof.hex.mpas_ocean.vmix_v842`` runs it inside ``ocn_vmix_coefs``, averages
the viscosity to the edges and solves the tracers with KPP's non-local flux
(``config_cvmix_kpp_nonlocal_with_implicit_mix``).

The kernels (``cvmix_v842.cu``) are held word for word to v8.4.2 with CVMix
v0.84-beta, compiled verbatim with gfortran ``-O0 -ffp-contract=off`` at
RKIND 8 (``run_vmix_cvmix_r8``, ``tests/test_mpas_ocean_cvmix_parity.py``).

v8.4.2's own behaviours kept as written: with ``maxLevelCell ==
nVertLevels`` the squared buoyancy frequency KPP's unresolved shear reads at
the column's last interface is the one above it, copied over the bottom
level's (``Nsqr_iface(k:maxLevelCell+1) = Nsqr_iface(k-1)`` with ``k =
min(maxLevelCell+1, nVertLevels)``); convection without KPP is applied from
the top (CVMix's ``kOBL_depth`` is never set, so its index is 1); a negative
``config_cvmix_convective_triggerBVF`` gives no convection at all (MPAS clips
the frequency at zero before CVMix reads it); the latitude CVMix is handed
uses ``180/3.14`` (unused on this path); the smoothing sets the cell loop's
shared bound ``nCells = nCellsArray(2)``, so with
``config_cvmix_use_BLD_smoothing`` the second stage (KPP, convection, the
no-flux zeroing) stops one halo layer short of ``nCellsArray(3)``.

Refused by name: CVMix at RKIND 4 (``mpas_ocn_vmix_cvmix.F`` does not compile
there: CVMix is real(8)); ``config_cvmix_kpp_EkmanOBL = .true.`` and
``config_cvmix_kpp_use_enhanced_diff = .false.``, whose v0.84-beta logic the
Apache-2.0-licensed CVMix this port is taken under changed (NOTICE,
PROVENANCE); the options MPAS-Ocean itself stops on (a matching other than
``SimpleShapes``/``MatchBoth``, PP shear without the background,
``config_cvmix_convective_basedOnBVF = .false.``, a shear scheme other than
``PP``/``KPP``, an interpolation CVMix does not know) and a surface layer
extent outside (0, 1].
"""
from __future__ import annotations

import re
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

from woof.hex.errors import ConfigurationRefusal

from .tend_v842 import COMPILE_OPTIONS, _precision_of, launch_options

SOURCE_FILES = (
    "src/core_ocean/shared/mpas_ocn_vmix_cvmix.F",
    "CVMix v0.84-beta src/shared/cvmix_kpp.F90",
    "CVMix v0.84-beta src/shared/cvmix_math.F90",
    "CVMix v0.84-beta src/shared/cvmix_shear.F90",
    "CVMix v0.84-beta src/shared/cvmix_convection.F90",
)
MODULE_OPTIONS = COMPILE_OPTIONS
#: RKIND 8 only
PRECISIONS = {"r8": np.float64}
MATH_HEADERS = ("glibc_flt64.cuh",)
#: cvmix_init_kpp's names -> CVMIX_MATH_INTERP_{LINEAR,QUAD,CUBE_SPLINE}
INTERPOLATIONS = {"line": 1, "linear": 1, "quad": 2, "quadratic": 2, "cube": 3, "cubic": 3, "cubic_spline": 3,
                  "cubic spline": 3}
SHEAR_SCHEMES = {"PP": 1, "KPP": 2}
_BLOCK = 128


def module_source(precision: str = "r8") -> str:
    if precision not in PRECISIONS:
        raise ValueError("CVMix runs at RKIND 8 only")
    import woof.core.kernels as kernels
    root = Path(kernels.__file__).parent
    math = "\n".join((root / name).read_text(encoding="utf-8") for name in MATH_HEADERS)
    body = files("woof.hex.mpas_ocean").joinpath("cvmix_v842.cu").read_text(encoding="utf-8")
    return "#define MPAS_RKIND8 1\n" + math + "\n" + body


@lru_cache(maxsize=None)
def _work_arrays() -> int:
    """cvmix_v842.cu's per-column work arrays, ``W_N``, read from its enum."""
    body = files("woof.hex.mpas_ocean").joinpath("cvmix_v842.cu").read_text(encoding="utf-8")
    names = re.search(r"enum \{\s*(W_ZW\b[^}]*)\}", body).group(1).split(",")
    assert names[-1].strip() == "W_N"
    return len(names) - 1


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"ocn_cvmix_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _launch(name: str, n: int, args: tuple) -> None:
    import cupy as cp
    if n > 0:
        _module_on(cp.cuda.Device().id, "r8", launch_options()).get_function(name)(
            ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), args)


def _refuse(knob: str, value, reason: str, declaration: str) -> None:
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


def check_config(configs: dict, precision: str | None = None) -> None:
    """Refuse, by name, the CVMix options this port does not carry (``config_use_cvmix`` on)."""
    if not configs.get("config_use_cvmix", False):
        return
    if precision is not None and precision not in PRECISIONS:
        _refuse("config_use_cvmix", True,
                "CVMix at RKIND 4: mpas_ocn_vmix_cvmix.F does not compile in MPAS's single-precision build "
                "(CVMix is real(8) throughout), so there is no v8.4.2 behaviour to hold it to",
                "config_use_cvmix = .false. at RKIND 4, or run RKIND 8 (float64 arrays)")
    if configs.get("config_use_cvmix_shear", False):
        scheme = configs.get("config_cvmix_shear_mixing_scheme", "PP")
        if scheme not in SHEAR_SCHEMES:
            _refuse("config_cvmix_shear_mixing_scheme", scheme, "CVMix stops on a shear scheme other than PP or KPP",
                    "config_cvmix_shear_mixing_scheme = 'PP' | 'KPP'")
        if scheme == "PP" and not configs.get("config_use_cvmix_background", False):
            _refuse("config_cvmix_shear_mixing_scheme", scheme,
                    "MPAS-Ocean stops on PP shear mixing without config_use_cvmix_background",
                    "config_use_cvmix_background = .true. with PP shear mixing")
    if configs.get("config_use_cvmix_convection", False) and not configs.get(
            "config_cvmix_convective_basedOnBVF", True):
        _refuse("config_cvmix_convective_basedOnBVF", False,
                "MPAS-Ocean stops on convection not based on the Brunt-Vaisala frequency",
                "config_cvmix_convective_basedOnBVF = .true.")
    if not configs.get("config_use_cvmix_kpp", False):
        return
    if configs.get("config_cvmix_kpp_EkmanOBL", False):
        _refuse("config_cvmix_kpp_EkmanOBL", True,
                "CVMix v0.84-beta (the tag v8.4.2 builds) caps the boundary layer at the Ekman depth whatever the "
                "surface buoyancy flux; the Apache-2.0-licensed CVMix this port is taken under caps it only when "
                "the forcing is stable and f /= 0, so the v0.84-beta words are not carried",
                "config_cvmix_kpp_EkmanOBL = .false.")
    if not configs.get("config_cvmix_kpp_use_enhanced_diff", True):
        _refuse("config_cvmix_kpp_use_enhanced_diff", False,
                "CVMix v0.84-beta leaves the interface below the boundary layer at the boundary layer's zero when "
                "the enhancement is off; the Apache-2.0-licensed CVMix this port is taken under keeps the interior "
                "value there, so the v0.84-beta words are not carried",
                "config_cvmix_kpp_use_enhanced_diff = .true.")
    matching = configs.get("config_cvmix_kpp_matching", "SimpleShapes")
    if matching not in ("SimpleShapes", "MatchBoth"):
        _refuse("config_cvmix_kpp_matching", matching, "MPAS-Ocean stops on a matching other than "
                "SimpleShapes or MatchBoth", "config_cvmix_kpp_matching = 'SimpleShapes' | 'MatchBoth'")
    interp = configs.get("config_cvmix_kpp_interpolationOMLType", "quadratic")
    if interp not in INTERPOLATIONS:
        _refuse("config_cvmix_kpp_interpolationOMLType", interp,
                "CVMix stops on an interpolation it does not know (MPAS passes it as both interp_type and "
                "interp_type2, so LMD94 is not reachable)",
                "config_cvmix_kpp_interpolationOMLType = 'linear' | 'quadratic' | 'cubic'")
    sle = float(configs.get("config_cvmix_kpp_surface_layer_extent", 0.1))
    if not 0.0 < sle <= 1.0:
        _refuse("config_cvmix_kpp_surface_layer_extent", sle,
                "CVMix stops outside [0, 1] and divides by it in the unresolved shear at 0",
                "0 < config_cvmix_kpp_surface_layer_extent <= 1")


def _parameters(configs: dict):
    import cupy as cp
    c = configs.get
    rp = np.array([
        c("config_cvmix_kpp_surface_layer_extent", 0.1), c("config_cvmix_kpp_criticalBulkRichardsonNumber", 0.25),
        c("config_cvmix_kpp_boundary_layer_depth", 30.0), c("configure_cvmix_kpp_minimum_OBL_under_sea_ice", 10.0),
        c("config_density0", 1026.0), c("config_cvmix_background_viscosity", 1.0e-4),
        c("config_cvmix_background_diffusion", 1.0e-5), c("config_cvmix_shear_PP_nu_zero", 0.005),
        c("config_cvmix_shear_PP_alpha", 5.0), c("config_cvmix_shear_PP_exp", 2.0),
        c("config_cvmix_shear_KPP_nu_zero", 0.005), c("config_cvmix_shear_KPP_Ri_zero", 0.7),
        c("config_cvmix_shear_KPP_exp", 3.0), c("config_cvmix_convective_viscosity", 1.0),
        c("config_cvmix_convective_diffusion", 1.0), c("config_cvmix_convective_triggerBVF", 0.0)], np.float64)
    shear = SHEAR_SCHEMES[c("config_cvmix_shear_mixing_scheme", "PP")] if c("config_use_cvmix_shear", False) else 0
    ip = np.array([
        int(bool(c("config_use_cvmix_kpp", False))), int(bool(c("config_use_cvmix_fixed_boundary_layer", False))),
        INTERPOLATIONS.get(c("config_cvmix_kpp_interpolationOMLType", "quadratic"), 2),
        int(bool(c("config_cvmix_kpp_MonObOBL", False))),
        int(c("config_cvmix_kpp_matching", "SimpleShapes") == "MatchBoth"), shear,
        int(bool(c("config_use_cvmix_convection", False))), int(c("config_cvmix_num_ri_smooth_loops", 2))], np.int32)
    return cp.asarray(rp), cp.asarray(ip)


def coefs_build(*, mesh: dict, state: dict, diagnostics: dict, forcing: dict, configs: dict) -> None:
    """``ocn_vmix_coefs_cvmix_build`` for one block, in place on the diagnostics'
    ``vertViscTopOfCell``, ``vertDiffTopOfCell``, ``vertNonLocalFlux``, the
    boundary-layer depth (smoothed), its index and the bulk Richardson numbers.
    The caller has checked ``configs`` (:func:`check_config`, as
    ``vmix_v842.vmix_implicit`` does)."""
    import cupy as cp
    from . import vmix_v842

    h = state["layerThickness"]
    precision = _precision_of(h)
    i32 = np.int32
    m, d = mesh, diagnostics
    nl = int(m["nVertLevels"])
    cells = [int(x) for x in m["nCellsArray"]]
    me = int(m["edgesOnCell"].shape[1])
    visc, diff, nonlocal_flux = d["vertViscTopOfCell"], d["vertDiffTopOfCell"], d["vertNonLocalFlux"]
    visc[: cells[-1]] = 0
    diff[: cells[-1]] = 0
    nonlocal_flux[: cells[-1]] = 0
    if configs.get("config_use_cvmix_background", False):
        n = cells[2] * (nl + 1)
        vmix_v842._launch(precision, "ocn_vmix_add_const_v842", n, (
            visc, i32(cells[2]), i32(nl + 1), np.float64(configs.get("config_cvmix_background_viscosity", 1.0e-4))))
        vmix_v842._launch(precision, "ocn_vmix_add_const_v842", n, (
            diff, i32(cells[2]), i32(nl + 1), np.float64(configs.get("config_cvmix_background_diffusion", 1.0e-5))))
    rp, ip = _parameters(configs)
    work = cp.zeros(max(cells[2], 1) * _work_arrays() * (nl + 2), dtype=np.float64)
    bld = d["boundaryLayerDepth"]
    if configs.get("config_use_cvmix_kpp", False):
        iwork = cp.zeros(max(cells[2], 1) * (nl + 2), dtype=np.int32)
        _launch("ocn_cvmix_kpp_stage1_v842", cells[2], (
            d["bulkRichardsonNumber"], d["bulkRichardsonNumberBuoy"], d["bulkRichardsonNumberShear"], bld, work,
            iwork, h, state["normalVelocity"], d["potentialDensity"], d["BruntVaisalaFreqTop"],
            d["surfaceFrictionVelocity"], d["surfaceBuoyancyForcing"], forcing["iceFraction"], m["areaCell"],
            m["dcEdge"], m["dvEdge"], m["nEdgesOnCell"], m["edgesOnCell"], m["maxLevelCell"], i32(cells[2]),
            i32(nl), i32(me), rp, ip))
    if configs.get("config_cvmix_use_BLD_smoothing", True):
        smooth = d["boundaryLayerDepthSmooth"]
        _launch("ocn_cvmix_bld_smooth_v842", cells[1], (
            smooth, bld, m["nEdgesOnCell"], m["cellsOnCell"], m["cellMask"], m["areaCell"], i32(cells[1]), i32(nl),
            i32(me)))
        bld[: cells[1]] = smooth[: cells[1]]
        # quirk: the smoothing reassigns the shared loop bound ``nCells = nCellsArray(2)``, so stage 2
        # (KPP coefficients, convection, the no-flux zeroing) then runs one halo layer short
        stage2 = cells[1]
    else:
        stage2 = cells[2]
    _launch("ocn_cvmix_stage2_v842", stage2, (
        visc, diff, nonlocal_flux, bld, d["indexBoundaryLayerDepth"], work, h, d["RiTopOfCell"],
        d["BruntVaisalaFreqTop"], d["surfaceFrictionVelocity"], d["surfaceBuoyancyForcing"], m["maxLevelCell"],
        i32(stage2), i32(nl), i32(nonlocal_flux.shape[-1]), i32(int(d.get("index_vertNonLocalFluxTemp", 1)) - 1),
        rp, ip))


def visc_to_edges(*, mesh: dict, diagnostics: dict) -> None:
    """``ocn_vmix_implicit``'s CVMix average: ``vertViscTopOfEdge`` over edges
    1..``nEdgesArray(1)``, the mean of its cells' ``vertViscTopOfCell`` above
    ``maxLevelEdgeTop``, zero below."""
    nl = int(mesh["nVertLevels"])
    n_edges = int(mesh["nEdgesArray"][0])
    _launch("ocn_cvmix_visc_edge_v842", n_edges * (nl + 1), (
        diagnostics["vertViscTopOfEdge"], diagnostics["vertViscTopOfCell"], mesh["cellsOnEdge"],
        mesh["maxLevelEdgeTop"], np.int32(n_edges), np.int32(nl)))
