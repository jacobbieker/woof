"""MPAS-Seaice v8.4.2: incremental remap advection on the GPU (planar meshes).

``mpas_seaice_advection_incremental_remap.F`` (with its tracers module) moves
the ice area, the ice and snow volume and every tracer by incremental
remapping (Dukowicz and Baumgardner 2000; Lipscomb and Hunke 2004):

- :func:`tracer_tree` is ``seaice_add_tracers_to_linked_list``: the tracers the
  packages switch on, each with its parent, in the order the remap needs them
  (mass first, then the tracers with one parent, then two, then three);
- :func:`volume_to_thickness` and :func:`thickness_to_volume` are the
  conversions ``seaice_run_advection_incremental_remap`` wraps around the
  remap (it treats every field after the ice area as a tracer, so the volumes
  travel as thicknesses).

The kernels (``remap_v842.cu``) are held word for word to v8.4.2 compiled
verbatim with gfortran ``-O0 -ffp-contract=off`` at RKIND 4 and 8
(``tools/mpas_seaice_oracle/run_incremental_remap.F90``,
``tests/test_mpas_seaice_remap_parity.py``).

The remap's geometry (``seaice_init_advection_incremental_remap``: the vertex
coordinates in each cell's and edge's plane, the remap neighbours of each
edge, the cells' geometric moments, ``coeffs_reconstruct``) is the input's,
as the variational basis is the velocity solver's.

Refused by name: upwind advection (v8.4.2 cannot run it,
:data:`UPWIND_REASON`), spherical meshes, halo exchanges, the conservation
and monotonicity checks.

Pools are dicts of device arrays named as in MPAS, in the Fortran layout
reversed (a tracer ``t(nLayers, nCategories, nCells+1)`` is C-order
``(nCells+1, nCategories, nLayers)``), float32 (``r4``) or float64 (``r8``),
with MPAS's missing-cell element.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

SOURCE_FILES = (
    "src/core_seaice/shared/mpas_seaice_advection_incremental_remap_tracers.F:165-911",
    "src/core_seaice/shared/mpas_seaice_advection_incremental_remap.F:66-77,2510-2552,2619-2661,9309-9387",
)
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
MODULE_OPTIONS = COMPILE_OPTIONS
PRECISIONS = {"r4": np.float32, "r8": np.float64}
_BLOCK = 128

#: seaice_add_tracers_to_linked_list (tracers.F:223-464): (tracer, parent,
#: the package that adds it or None). pondArea's parent is levelIceArea under
#: config_use_level_meltponds, else iceAreaCategory.
TRACER_TABLE = (
    ("iceAreaCategory", "None", None),
    ("iceVolumeCategory", "iceAreaCategory", None),
    ("snowVolumeCategory", "iceAreaCategory", None),
    ("surfaceTemperature", "iceAreaCategory", None),
    ("iceEnthalpy", "iceVolumeCategory", None),
    ("iceSalinity", "iceVolumeCategory", None),
    ("snowEnthalpy", "snowVolumeCategory", None),
    ("iceAge", "iceVolumeCategory", "pkgColumnTracerIceAgeActive"),
    ("firstYearIceArea", "iceAreaCategory", "pkgColumnTracerFirstYearIceActive"),
    ("levelIceArea", "iceAreaCategory", "pkgColumnTracerLevelIceActive"),
    ("levelIceVolume", "iceVolumeCategory", "pkgColumnTracerLevelIceActive"),
    ("pondArea", "<level ponds>", "pkgColumnTracerPondsActive"),
    ("pondDepth", "pondArea", "pkgColumnTracerPondsActive"),
    ("pondLidThickness", "pondArea", "pkgColumnTracerLidThicknessActive"),
    ("snowScatteringAerosol", "snowVolumeCategory", "pkgColumnTracerAerosolsActive"),
    ("snowBodyAerosol", "snowVolumeCategory", "pkgColumnTracerAerosolsActive"),
    ("iceScatteringAerosol", "iceVolumeCategory", "pkgColumnTracerAerosolsActive"),
    ("iceBodyAerosol", "iceVolumeCategory", "pkgColumnTracerAerosolsActive"),
    ("brineFraction", "iceVolumeCategory", "pkgTracerBrineActive"),
    ("skeletalAlgaeConc", "iceAreaCategory", "pkgTracerSkeletalAlgaeActive"),
    ("skeletalNitrateConc", "iceAreaCategory", "pkgTracerSkeletalNitrateActive"),
    ("skeletalDOCConc", "iceAreaCategory", "pkgTracerSkeletalCarbonActive"),
    ("skeletalDICConc", "iceAreaCategory", "pkgTracerSkeletalCarbonActive"),
    ("skeletalDONConc", "iceAreaCategory", "pkgTracerSkeletalDONActive"),
    ("skeletalAmmoniumConc", "iceAreaCategory", "pkgTracerSkeletalAmmoniumActive"),
    ("skeletalSilicateConc", "iceAreaCategory", "pkgTracerSkeletalSilicateActive"),
    ("skeletalDMSConc", "iceAreaCategory", "pkgTracerSkeletalDMSActive"),
    ("skeletalDMSPpConc", "iceAreaCategory", "pkgTracerSkeletalDMSActive"),
    ("skeletalDMSPdConc", "iceAreaCategory", "pkgTracerSkeletalDMSActive"),
    ("skeletalNonreactiveConc", "iceAreaCategory", "pkgTracerSkeletalNonreactiveActive"),
    ("skeletalHumicsConc", "iceAreaCategory", "pkgTracerSkeletalHumicsActive"),
    ("skeletalParticulateIronConc", "iceAreaCategory", "pkgTracerSkeletalIronActive"),
    ("skeletalDissolvedIronConc", "iceAreaCategory", "pkgTracerSkeletalIronActive"),
    ("mobileFraction", "brineFraction", "pkgTracerMobileFractionActive"),
    ("verticalAlgaeSnow", "snowVolumeCategory", "pkgTracerVerticalAlgaeActive"),
    ("verticalAlgaeIce", "brineFraction", "pkgTracerVerticalAlgaeActive"),
    ("verticalNitrateSnow", "snowVolumeCategory", "pkgTracerVerticalNitrateActive"),
    ("verticalNitrateIce", "brineFraction", "pkgTracerVerticalNitrateActive"),
    ("verticalDOCSnow", "snowVolumeCategory", "pkgTracerVerticalCarbonActive"),
    ("verticalDICSnow", "snowVolumeCategory", "pkgTracerVerticalCarbonActive"),
    ("verticalDOCIce", "brineFraction", "pkgTracerVerticalCarbonActive"),
    ("verticalDICIce", "brineFraction", "pkgTracerVerticalCarbonActive"),
    ("verticalDONSnow", "snowVolumeCategory", "pkgTracerVerticalDONActive"),
    ("verticalDONIce", "brineFraction", "pkgTracerVerticalDONActive"),
    ("verticalAmmoniumSnow", "snowVolumeCategory", "pkgTracerVerticalAmmoniumActive"),
    ("verticalAmmoniumIce", "brineFraction", "pkgTracerVerticalAmmoniumActive"),
    ("verticalSilicateSnow", "snowVolumeCategory", "pkgTracerVerticalSilicateActive"),
    ("verticalSilicateIce", "brineFraction", "pkgTracerVerticalSilicateActive"),
    ("verticalDMSSnow", "snowVolumeCategory", "pkgTracerVerticalDMSActive"),
    ("verticalDMSPpSnow", "snowVolumeCategory", "pkgTracerVerticalDMSActive"),
    ("verticalDMSPdSnow", "snowVolumeCategory", "pkgTracerVerticalDMSActive"),
    ("verticalDMSIce", "brineFraction", "pkgTracerVerticalDMSActive"),
    ("verticalDMSPpIce", "brineFraction", "pkgTracerVerticalDMSActive"),
    ("verticalDMSPdIce", "brineFraction", "pkgTracerVerticalDMSActive"),
    ("verticalNonreactiveSnow", "snowVolumeCategory", "pkgTracerVerticalNonreactiveActive"),
    ("verticalNonreactiveIce", "brineFraction", "pkgTracerVerticalNonreactiveActive"),
    ("verticalHumicsSnow", "snowVolumeCategory", "pkgTracerVerticalHumicsActive"),
    ("verticalHumicsIce", "brineFraction", "pkgTracerVerticalHumicsActive"),
    ("verticalParticulateIronSnow", "snowVolumeCategory", "pkgTracerVerticalIronActive"),
    ("verticalDissolvedIronSnow", "snowVolumeCategory", "pkgTracerVerticalIronActive"),
    ("verticalParticulateIronIce", "brineFraction", "pkgTracerVerticalIronActive"),
    ("verticalDissolvedIronIce", "brineFraction", "pkgTracerVerticalIronActive"),
    ("verticalAerosolsIce", "brineFraction", "pkgTracerZAerosolsActive"),
    ("verticalAerosolsSnow", "snowVolumeCategory", "pkgTracerZAerosolsActive"),
    ("verticalSalinity", "brineFraction", "pkgTracerZSalinityActive"),
)
#: the packages seaice_add_tracers_to_linked_list reads (tracers.F:233-294)
PACKAGES = tuple(dict.fromkeys(p for _, _, p in TRACER_TABLE if p is not None)) + (
    "pkgColumnBiogeochemistryActive",)


@dataclass(eq=False)
class Tracer:
    """One element of the remap's tracer list (tracers.F:37-144)."""
    name: str
    parent_name: str
    parent: "Tracer | None" = None
    has_child: bool = False
    n_parents: int = 0
    active: bool = True
    children: list = field(default_factory=list, repr=False)

    def ancestors(self) -> list["Tracer"]:
        """Parent, grandparent, ... up to the mass field."""
        chain, t = [], self.parent
        while t is not None:
            chain.append(t)
            t = t.parent
        return chain


def tracer_tree(packages: dict, configs: dict, is_active) -> list[Tracer]:
    """``seaice_add_tracers_to_linked_list``: the ordered tracer list.

    ``packages`` maps the package switches (absent = off); ``is_active(name)``
    is ``mpas_pool_get_field_info``'s ``isActive`` for a tracer field. The
    steps are v8.4.2's: add in table order (tracers.F:223-464), connect each
    tracer to its parent by name (create_tracers_connectivity: every tracer of
    that name, and the parent's hasChild is set), count parents
    (set_parent_numbers), drop inactive tracers (remove_inactive_tracers:
    hasChild is not revisited), then order by the number of parents, keeping
    the list order within a level (set_tracer_order)."""
    level_ponds = bool(configs.get("config_use_level_meltponds", True))
    initial: list[Tracer] = []
    for name, parent, package in TRACER_TABLE:
        if package is not None and not packages.get(package, False):
            continue
        if parent == "<level ponds>":
            parent = "levelIceArea" if level_ponds else "iceAreaCategory"
        initial.append(Tracer(name, parent))
    # create_tracers_connectivity: the last match wins, every match marks hasChild
    for this in initial:
        for candidate in initial:
            if candidate.name == this.parent_name:
                this.parent = candidate
                candidate.has_child = True
    for this in initial:
        this.n_parents = len(this.ancestors())
    for this in initial:
        this.active = bool(is_active(this.name))
    for this in initial:
        if this.active or not this.has_child:
            continue
        for child in initial:
            if child.n_parents > 1 and child.parent.name == this.name and child.active:
                raise ConfigurationRefusal(
                    knob=f"tracer {this.name}", value="inactive",
                    reason=f"IR tracers: cannot have an inactive parent ({this.name}) with an active child "
                           f"({child.name}); v8.4.2 stops (MPAS_LOG_CRIT)",
                    declaration="every active tracer's parent active")
    kept = [t for t in initial if t.active]
    if not kept:
        raise ConfigurationRefusal(knob="tracers", value="none active", reason="IR tracers: No active tracers for IR",
                                   declaration="iceAreaCategory active")
    levels = max(t.n_parents for t in kept)
    ordered = [t for level in range(levels + 1) for t in kept if t.n_parents == level]
    for t in ordered:
        if t.parent is not None:
            t.parent.children.append(t)
    return ordered


#: why upwind stays refused (v8.4.2's mpas_seaice_advection_upwind.F)
UPWIND_REASON = (
    "upwind advection cannot run in MPAS-Seaice v8.4.2: mpas_seaice_advection_upwind.F reads configs the "
    "sea-ice Registry does not declare (config_convert_volume_to_thickness, config_limit_ice_concentration, "
    "config_clean_tracers, config_horiz_tracer_adv_order, config_vert_tracer_adv_order, config_coef_3rd_order) "
    "and mesh fields it does not carry (derivTwo, advCoefs, advCoefs3rd, nAdvCellsForEdge, advCellsForEdge, "
    "highOrderAdvectionMask, boundaryCell); MPAS_pool_get_config hands back a null pointer for an undeclared "
    "config and prepare_tracers dereferences it, so the model crashes")


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


def check_config(configs: dict) -> None:
    """Refuse, by name, what this port does not carry."""
    scheme = configs.get("config_advection_type", "incremental_remap")
    if scheme == "upwind":
        _refuse("config_advection_type", scheme, UPWIND_REASON, "config_advection_type = 'incremental_remap'")
    if scheme != "incremental_remap":
        _refuse("config_advection_type", scheme, "v8.4.2 knows 'incremental_remap' and 'upwind' only",
                "config_advection_type = 'incremental_remap'")
    if configs.get("mesh:on_a_sphere", configs.get("on_a_sphere", False)):
        _refuse("on_a_sphere", True, "the spherical remap (the triangles' edge-vector basis, compute_gradient on "
                                     "the sphere, the rotated grid) is not ported; planar meshes are",
                "a planar mesh (on_a_sphere = .false.)")
    for knob, reason in (
            ("config_use_halo_exch", "halo exchanges: this port runs one block"),
            ("config_conservation_check", "the conservation check sums over ranks (mpas_dmpar_sum_real) and "
                                          "only logs; not ported"),
            ("config_monotonicity_check", "the monotonicity check (local min/max with a halo update) only "
                                          "logs; not ported")):
        if configs.get(knob, False):
            _refuse(knob, True, reason, f"{knob} = .false.")


# ---------------------------------------------------------------------------
# kernels

def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    body = files("woof.hex.mpas_seaice").joinpath("remap_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"seaice_remap_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(precision: str, name: str):
    import cupy as cp
    options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
    return _module_on(cp.cuda.Device().id, precision, options).get_function(name)


def precision_of(array) -> str:
    for name, dtype in PRECISIONS.items():
        if array.dtype == dtype:
            return name
    raise ValueError(f"remap arrays must be float32 (r4) or float64 (r8), not {array.dtype}")


def _grid(n: int) -> tuple[int]:
    return ((max(n, 1) + _BLOCK - 1) // _BLOCK,)


def _launch(precision: str, name: str, n: int, args) -> None:
    _function(precision, name)(_grid(n), (_BLOCK,), args)


def _category_view(array):
    """A tracer that is 2-D physically (first extent 1): its (nCells+1, nCategories) words."""
    if array.ndim != 3 or array.shape[2] != 1:
        raise ValueError(f"the ice and snow volume are (nCells+1, nCategories, 1), not {tuple(array.shape)}")
    return array.reshape(array.shape[0], array.shape[1])


def _conversion(tree, tracers: dict, remap: dict, kernel: str) -> None:
    # seaice_run_advection_incremental_remap:2525-2548 / 2634-2657: the ice area is
    # copied into workCategoryCell, which then converts the volumes after it
    work = remap["workCategoryCell"]
    for t in tree:
        if t.name == "iceAreaCategory":
            area = _category_view(tracers[t.name])
            if work.shape != area.shape:
                raise ValueError(f"workCategoryCell is {tuple(work.shape)}, the ice area {tuple(area.shape)}")
            work[...] = area
        elif t.name in ("iceVolumeCategory", "snowVolumeCategory"):
            volume = _category_view(tracers[t.name])
            _launch(precision_of(volume), kernel, volume.size, (volume, work, np.int32(volume.size)))


def volume_to_thickness(tree, *, tracers: dict, remap: dict) -> None:
    """The ice and snow volume to thickness (``volume_to_thickness``), in place;
    ``workCategoryCell`` (the ``remap`` pool's) takes the ice area."""
    _conversion(tree, tracers, remap, "seaice_ir_volume_to_thickness_v842")


def thickness_to_volume(tree, *, tracers: dict, remap: dict) -> None:
    """The ice and snow thickness back to volume (``thickness_to_volume``), in place."""
    _conversion(tree, tracers, remap, "seaice_ir_thickness_to_volume_v842")
