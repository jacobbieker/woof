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
  travel as thicknesses);
- :func:`departure_points`, :func:`departure_triangles` (with
  ``shift_vertices_of_departure_triangle``) and :func:`quadrature_points`: the
  region each edge's flux comes from, as triangles in their source cells;
- :func:`make_masks` and :func:`construct_linear_fields`: each tracer as a
  linear function in each cell, its gradient limited (no new extrema at the
  cell's vertices) about its parent's barycenter;
- :func:`integrate_fluxes`, :func:`mass_tracer_products`,
  :func:`update_mass_and_tracers`, :func:`recover_tracer_means` (the
  ``config_recover_tracer_means_check`` warnings) and :func:`zap_small_mass`;
- :func:`remap_block` is ``incremental_remap_block`` and :func:`run`
  ``seaice_run_advection_incremental_remap``;
  :mod:`~woof.hex.mpas_seaice.timestep_v842` calls it each dynamics subcycle.

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
and monotonicity checks, and tracer lists v8.4.2's update cannot run.

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
    "src/core_seaice/shared/mpas_seaice_advection_incremental_remap.F:66-77,2345-2678,2751-3403,3418-5375,"
    "5390-6674,6689-8043,8825-8947,8995-9295,9309-9387",
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


class RemapError(RuntimeError):
    """A condition under which v8.4.2's remap stops the model (or would run on garbage)."""


def _int(value) -> np.int32:
    return np.int32(int(value))


def _dims(mesh: dict) -> tuple[int, int, int, int, int]:
    return (int(mesh["nCells"]), int(mesh["nEdges"]), int(mesh["nVertices"]), int(mesh["maxEdges"]),
            int(mesh["vertexDegree"]))


def departure_points(*, mesh: dict, velocity: dict, remap: dict, dt) -> int:
    """``find_departure_points``: ``departurePoint`` (zeroed, then the drift
    back over ``dt`` at vertices 1..nVertices). Returns how many vertices
    trip v8.4.2's CFL check (it logs them and carries on)."""
    import cupy as cp
    dp = remap["departurePoint"]
    precision = precision_of(dp)
    real = PRECISIONS[precision]
    nv = int(mesh["nVertices"])
    dp[...] = 0
    cfl = cp.zeros(max(nv, 1), cp.int32)
    _launch(precision, "seaice_ir_departure_points_v842", nv, (
        dp, cfl, velocity["uVelocity"], velocity["vVelocity"], remap["minLengthEdgesOnVertex"], real(dt),
        _int(nv)))
    return int(cfl[:nv].sum())


def departure_triangles(*, mesh: dict, remap: dict) -> None:
    """``find_departure_triangles`` with ``shift_vertices_of_departure_triangle``
    (planar): ``maskEdge``, ``xTriangle``/``yTriangle`` (vertices 1-3 of each
    triangle in its source cell's frame), ``iCellTriangle`` and the signed
    ``triangleArea``."""
    import cupy as cp
    nc, ne, nv, me, vd = _dims(mesh)
    precision = precision_of(remap["xTriangle"])
    for name in ("xTriangle", "yTriangle", "triangleArea", "iCellTriangle", "maskEdge"):
        remap[name][...] = 0
    error = cp.zeros(1, cp.int32)
    _launch(precision, "seaice_ir_departure_triangles_v842", ne, (
        remap["xTriangle"], remap["yTriangle"], remap["iCellTriangle"], remap["triangleArea"], remap["maskEdge"],
        error, remap["departurePoint"], remap["xVertexOnEdge"], remap["yVertexOnEdge"], remap["xVertexOnCell"],
        remap["yVertexOnCell"], mesh["nEdgesOnCell"], mesh["verticesOnCell"], mesh["verticesOnEdge"],
        remap["remapEdge"], remap["cellsOnEdgeRemap"], remap["edgesOnEdgeRemap"], _int(ne), _int(nc), _int(me),
        _int(vd)))
    if int(error[0]):
        raise RemapError("find_departure_triangles: a departure triangle without its source cell's vertex, or "
                         "more than nTriPerEdgeRemap triangles on an edge (the mesh and its remap geometry "
                         "disagree; v8.4.2 would read or write outside its arrays)")


def quadrature_points(*, mesh: dict, remap: dict) -> None:
    """``get_triangle_quadrature_points``: each masked edge's triangles' six
    quadrature points, over their vertices."""
    ne = int(mesh["nEdges"])
    _launch(precision_of(remap["xTriangle"]), "seaice_ir_quadrature_points_v842", ne * 6, (
        remap["xTriangle"], remap["yTriangle"], remap["maskEdge"], _int(ne)))


#: incremental_remap_block's mask threshold (eps11, remap.F:67)
EPS11 = 1.0e-11
#: the cell moments compute_barycenter_coordinates reads, in its order
MOMENTS = ("x", "y", "xx", "xy", "yy", "xxx", "xxy", "xyy", "yyy", "xxxx", "xxxy", "xxyy", "xyyy", "yyyy")
#: per-tracer pool fields: (pool, suffix)
_AUX = {"mask": ("tracer_masks", "Mask"), "edgeFlux": ("tracer_edge_fluxes", "EdgeFlux"),
        "product": ("tracer_products", "Product"), "triangle": ("tracer_triangles", "Triangle"),
        "xBarycenter": ("tracer_barycenter", "Barycenterx"), "yBarycenter": ("tracer_barycenter", "Barycentery"),
        "center": ("tracer_reconstruction", "Center"), "xGrad": ("tracer_reconstruction", "Gradx"),
        "yGrad": ("tracer_reconstruction", "Grady")}


def aux(pools: dict, tracer: Tracer, what: str):
    """One of a tracer's remap fields (``iceAreaCategoryMask`` ...) from its pool."""
    pool, suffix = _AUX[what]
    return pools[pool][tracer.name + suffix]


def _layers(array) -> int:
    return int(array.shape[2])


def make_masks(tree, *, pools: dict, mesh: dict, threshold: float = EPS11) -> None:
    """``make_masks``: each tracer's mask (and the remap pool's ``maskCell``)."""
    nc = int(mesh["nCells"])
    remap, tracers = pools["incremental_remap"], pools["tracers"]
    for t in tree:
        field = tracers[t.name]
        precision = precision_of(field)
        mask = aux(pools, t, "mask")
        mask[...] = 0
        ncat, nl = int(field.shape[1]), _layers(field)
        if t.n_parents == 0:
            _launch(precision, "seaice_ir_mask_mass_v842", nc, (mask, remap["maskCell"], field, _int(nc),
                                                                _int(ncat * nl)))
        else:
            parent = tracers[t.parent.name]
            if nl == 1 and _layers(parent) != 1:
                raise RemapError(f"{t.name}: a 2-D tracer under a 3-D parent ({t.parent.name})")
            _launch(precision, "seaice_ir_mask_child_v842", nc * ncat * nl, (
                mask, parent, PRECISIONS[precision](threshold), _int(nc), _int(ncat), _int(nl),
                _int(_layers(parent))))


def construct_linear_fields(tree, *, pools: dict, mesh: dict) -> None:
    """``construct_linear_tracer_fields`` (planar): each tracer's limited
    gradient, centre value and, for parents, barycenter, in tree order."""
    nc, ne, nv, me, vd = _dims(mesh)
    remap, tracers = pools["incremental_remap"], pools["tracers"]
    moments = [remap[f"{m}AvgCell"] for m in MOMENTS]
    for t in tree:
        field = tracers[t.name]
        precision = precision_of(field)
        ncat, nl = int(field.shape[1]), _layers(field)
        out = [aux(pools, t, k) for k in ("center", "xGrad", "yGrad", "xBarycenter", "yBarycenter")]
        for array in out:
            array[...] = 0
        if t.has_child and t.n_parents >= 3:
            raise RemapError(f"{t.name}: a tracer with three parents and a child (v8.4.2 stops, MPAS_LOG_CRIT)")
        parent = t.parent if t.n_parents >= 1 else None
        grand = parent.parent if t.n_parents >= 2 else None

        def fields_of(tr):
            if tr is None:
                return [field] * 4, 1
            return [tracers[tr.name], aux(pools, tr, "center"), aux(pools, tr, "xGrad"),
                    aux(pools, tr, "yGrad")], _layers(tracers[tr.name])
        pf, pl = fields_of(parent)
        gf, gl = fields_of(grand)
        pxb = aux(pools, parent, "xBarycenter") if parent is not None else field
        pyb = aux(pools, parent, "yBarycenter") if parent is not None else field
        _launch(precision, "seaice_ir_construct_v842", nc * ncat * nl, (
            *out, field, aux(pools, t, "mask"), remap["maskCell"], mesh["nEdgesOnCell"], mesh["cellsOnCell"],
            mesh["edgesOnCell"], mesh["cellsOnEdge"], mesh["dcEdge"], mesh["coeffs_reconstruct"],
            remap["xVertexOnCell"], remap["yVertexOnCell"], *moments, pxb, pyb, *pf, *gf, _int(pl), _int(gl),
            _int(t.n_parents), _int(t.has_child), _int(nc), _int(ncat), _int(nl), _int(me)))


def integrate_fluxes(tree, *, pools: dict, mesh: dict) -> None:
    """``integrate_fluxes_over_triangles``: each tracer's edge fluxes (and its
    triangle values), in tree order. Raises :class:`RemapError` where v8.4.2
    stops on a negative ice area at a quadrature point."""
    import cupy as cp
    ne = int(mesh["nEdges"])
    remap, tracers = pools["incremental_remap"], pools["tracers"]
    negative = cp.zeros(1, cp.int32)
    for t in tree:
        field = tracers[t.name]
        precision = precision_of(field)
        ncat, nl = int(field.shape[1]), _layers(field)
        flux = aux(pools, t, "edgeFlux")
        flux[...] = 0
        if t.n_parents == 0:
            parent, pl = None, 1
        else:
            parent, pl = aux(pools, t.parent, "triangle"), _layers(tracers[t.parent.name])
        _launch(precision, "seaice_ir_integrate_fluxes_v842", ne * ncat * nl, (
            flux, aux(pools, t, "triangle"), negative, parent, aux(pools, t, "center"), aux(pools, t, "xGrad"),
            aux(pools, t, "yGrad"), remap["xTriangle"], remap["yTriangle"], remap["triangleArea"],
            remap["iCellTriangle"], remap["maskEdge"], _int(t.name == "iceAreaCategory"), _int(ne), _int(ncat),
            _int(nl), _int(pl)))
        if int(negative[0]):
            raise RemapError("integrate_fluxes_over_triangles: negative ice area at a departure triangle's "
                             "quadrature point (v8.4.2: SEAICE_ERROR_IR_NEG_AREA)")


def _parent_product(pools, t):
    if t.n_parents == 0:
        return None, 1
    return aux(pools, t.parent, "product"), _layers(pools["tracers"][t.parent.name])


def mass_tracer_products(tree, *, pools: dict, mesh: dict) -> None:
    """``compute_mass_tracer_products`` over cells 1..nCells."""
    nc = int(mesh["nCells"])
    for t in tree:
        field = pools["tracers"][t.name]
        ncat, nl = int(field.shape[1]), _layers(field)
        parent, pl = _parent_product(pools, t)
        _launch(precision_of(field), "seaice_ir_products_v842", nc * ncat * nl, (
            aux(pools, t, "product"), field, parent, _int(nc), _int(ncat), _int(nl), _int(pl)))


def update_mass_and_tracers(tree, *, pools: dict, mesh: dict) -> None:
    """``update_mass_and_tracers`` over cells 1..nCellsSolve, in tree order (a
    child divides by its parent's new product)."""
    import cupy as cp
    ncs, me = int(mesh["nCellsSolve"]), int(mesh["maxEdges"])
    negative = cp.zeros(1, cp.int32)
    for t in tree:
        field = pools["tracers"][t.name]
        ncat, nl = int(field.shape[1]), _layers(field)
        parent, pl = _parent_product(pools, t)
        _launch(precision_of(field), "seaice_ir_update_v842", ncs * ncat * nl, (
            field, aux(pools, t, "product"), negative, parent, aux(pools, t, "edgeFlux"), mesh["nEdgesOnCell"],
            mesh["edgesOnCell"], mesh["cellsOnEdge"], mesh["areaCell"], _int(ncs), _int(ncat), _int(nl), _int(pl),
            _int(me)))
        if int(negative[0]):
            raise RemapError(f"update_mass_and_tracers: negative {t.name} (v8.4.2: SEAICE_ERROR_IR_NEG_MASS)")


def zap_small_mass(tree, *, pools: dict, mesh: dict) -> None:
    """``zap_small_mass``: the mass-like field (the list's head) below 1e-22 to
    zero, with every other tracer in those categories."""
    ncs = int(mesh["nCellsSolve"])
    remap = pools["incremental_remap"]
    head = tree[0]
    field = pools["tracers"][head.name]
    ncat = int(field.shape[1])
    if _layers(field) != 1:
        raise RemapError(f"zap_small_mass: a 3-D mass-like field ({head.name}) stops v8.4.2")
    remap["maskCategoryCell"][...] = 0
    precision = precision_of(field)
    _launch(precision, "seaice_ir_zap_mass_v842", ncs * ncat, (
        field, aux(pools, head, "mask"), remap["maskCategoryCell"], _int(ncs), _int(ncat)))
    for t in tree[1:]:
        f = pools["tracers"][t.name]
        _launch(precision, "seaice_ir_zap_tracer_v842", ncs * ncat * _layers(f), (
            f, aux(pools, t, "mask"), remap["maskCategoryCell"], _int(ncs), _int(ncat), _int(_layers(f))))


def recover_tracer_means(tree, *, pools: dict, mesh: dict) -> list[dict]:
    """``recover_tracer_means`` (``config_recover_tracer_means_check``): the
    warnings v8.4.2 logs where a cell's reconstruction does not recover its
    product of means, in its order (cell, tracer, category, layer), each as
    the four log entries (message, integer and real arguments)."""
    import cupy as cp
    nc, me = int(mesh["nCells"]), int(mesh["maxEdges"])
    remap, tracers = pools["incremental_remap"], pools["tracers"]
    # the check's layer count is the last 3-D tracer's (recover_tracer_means:7672, read at 7973)
    last_layers = 0
    for t in tree:
        if _layers(tracers[t.name]) > 1:
            last_layers = _layers(tracers[t.name])
    per_tracer = []
    for t in tree:
        field = tracers[t.name]
        precision = precision_of(field)
        ncat, nl = int(field.shape[1]), _layers(field)
        if nl > 1 and last_layers > nl:
            raise RemapError(f"recover_tracer_means: {t.name} has {nl} layers but the check loops over "
                             f"{last_layers} (v8.4.2 reads past its arrays)")
        chain = [t] + t.ancestors()
        if len(chain) > 4:
            raise RemapError(f"recover_tracer_means: {t.name} has more than three parents")
        sets, layers = [], []
        for k in range(4):
            tr = chain[k] if k < len(chain) else chain[-1]
            sets += [aux(pools, tr, "center"), aux(pools, tr, "xGrad"), aux(pools, tr, "yGrad")]
            layers.append(_layers(tracers[tr.name]))
        n = nc * ncat * nl
        warn = cp.zeros(max(n, 1), cp.int32)
        rec = cp.zeros(max(n, 1), field.dtype)
        diff = cp.zeros(max(n, 1), field.dtype)
        _launch(precision, "seaice_ir_recover_means_v842", n, (
            warn, rec, diff, aux(pools, t, "product"), *sets, _int(layers[1]), _int(layers[2]), _int(layers[3]),
            _int(len(chain)), mesh["nEdgesOnCell"], mesh["edgesOnCell"], mesh["dcEdge"], mesh["dvEdge"],
            remap["xVertexOnCell"], remap["yVertexOnCell"], _int(nc), _int(ncat), _int(nl), _int(me)))
        product = aux(pools, t, "product")
        per_tracer.append((t, ncat, nl, cp.asnumpy(warn[:n]).reshape(nc, ncat, nl),
                           cp.asnumpy(rec[:n]).reshape(nc, ncat, nl), cp.asnumpy(diff[:n]).reshape(nc, ncat, nl),
                           cp.asnumpy(product[:nc]).reshape(nc, ncat, nl)))
    warnings = []
    for c in range(nc):
        for t, ncat, nl, warn, rec, diff, prod in per_tracer:
            layers = range(1) if nl == 1 else range(min(nl, last_layers))
            for cat in range(ncat):
                for lay in layers:
                    if not warn[c, cat, lay]:
                        continue
                    head = ("Reconstructed tracer does not recover the mean, iCat, iCell, tracer = $i $i $r "
                            if nl == 1 else "Reconstructed tracer does not recover the mean, iCat, iCell, "
                                            "tracer = $i $i")
                    warnings += [{"message": head + t.name, "ints": [cat + 1, c + 1], "reals": []},
                                 {"message": "Product of means:      $r", "ints": [],
                                  "reals": [float(prod[c, cat, lay])]},
                                 {"message": "Reconstructed product: $r", "ints": [],
                                  "reals": [float(rec[c, cat, lay])]},
                                 {"message": "Difference:            $r", "ints": [],
                                  "reals": [float(diff[c, cat, lay])]}]
    return warnings


def remap_block(tree, *, pools: dict, mesh: dict, configs: dict, dt) -> tuple[int, list[dict]]:
    """``incremental_remap_block`` (planar, without the monotonicity and
    conservation checks): returns the CFL-flagged vertex count and the
    mean-recovery warnings."""
    make_masks(tree, pools=pools, mesh=mesh)
    construct_linear_fields(tree, pools=pools, mesh=mesh)
    remap = pools["incremental_remap"]
    cfl = departure_points(mesh=mesh, velocity=pools["velocity_solver"], remap=remap, dt=dt)
    departure_triangles(mesh=mesh, remap=remap)
    quadrature_points(mesh=mesh, remap=remap)
    integrate_fluxes(tree, pools=pools, mesh=mesh)
    mass_tracer_products(tree, pools=pools, mesh=mesh)
    warnings = []
    if configs.get("config_recover_tracer_means_check", False):
        warnings = recover_tracer_means(tree, pools=pools, mesh=mesh)
    update_mass_and_tracers(tree, pools=pools, mesh=mesh)
    zap_small_mass(tree, pools=pools, mesh=mesh)
    return cfl, warnings


def check_tree(tree) -> None:
    """Refuse a tracer list v8.4.2's remap cannot run: a second mass-like
    field (a tracer whose parent the packages leave out) is updated against a
    freed dummy tracer, and a tracer whose parent was dropped as inactive (an
    inactive iceAreaCategory: remove_inactive_tracers only checks children
    with nParents > 1) is reconstructed against a parent outside the list."""
    kept = set(map(id, tree))
    orphan = [t for t in tree if t.parent is not None and id(t.parent) not in kept]
    if orphan:
        raise ConfigurationRefusal(
            knob=f"tracer {orphan[0].parent.name}", value="inactive",
            reason=f"{orphan[0].parent.name} is inactive but its child {orphan[0].name} is active; v8.4.2's "
                   "remove_inactive_tracers drops the parent without stopping (it checks only children with "
                   "nParents > 1), so the remap would read a parent outside the tracer list",
            declaration="every active tracer's parent active")
    extra = [t.name for t in tree[1:] if t.n_parents == 0]
    if extra:
        raise ConfigurationRefusal(
            knob=f"tracer {extra[0]}", value="no parent",
            reason=f"{extra[0]}'s parent is not in the tracer list (e.g. pondArea under config_use_level_meltponds "
                   "without the level-ice tracers), so it is a second mass-like field and v8.4.2's "
                   "update_mass_and_tracers divides it by a freed dummy tracer",
            declaration="every tracer's parent in the list")


def run(tree, *, pools: dict, mesh: dict, configs: dict) -> tuple[int, list[dict]]:
    """``seaice_run_advection_incremental_remap`` (time level 1, one block, no
    halo exchanges): the volumes to thickness, the remap at
    ``dynamicsTimeStep``, the thicknesses back to volume. Returns the
    CFL-flagged vertex count and the mean-recovery warnings."""
    check_config(configs)
    check_tree(tree)
    if int(mesh.get("nQuadPoints", 6)) != 6:
        raise ConfigurationRefusal(knob="nQuadPoints", value=mesh["nQuadPoints"],
                                   reason="the Registry fixes nQuadPoints = 6", declaration="nQuadPoints = 6")
    tracers, remap = pools["tracers"], pools["incremental_remap"]
    volume_to_thickness(tree, tracers=tracers, remap=remap)
    dt = pools["velocity_solver"]["dynamicsTimeStep"].item()
    result = remap_block(tree, pools=pools, mesh=mesh, configs=configs, dt=dt)
    thickness_to_volume(tree, tracers=tracers, remap=remap)
    return result
