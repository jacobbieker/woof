"""MPAS-A v8.4.1 LES closure (3-D Smagorinsky and prognostic 1.5-order TKE).

CPU authority for the ``config_les_model`` / ``config_les_surface`` branches
of MPAS-Atmosphere v8.4.1.  Every formula below is transcribed from the
published v8.4.1 release source (``MPAS-Dev/MPAS-Model`` tag ``v8.4.1``);
paths are relative to ``src/core_atmosphere``:

* option strings: ``dynamics/mpas_atm_dissipation_models.F:30-116``
  (``les_model_from_string``: ``none`` | ``3d_smagorinsky`` |
  ``prognostic_1.5_order``; ``les_surface_from_string``: ``none`` |
  ``specified`` | ``varying``), Registry defaults ``Registry.xml:149-172``.
* the gradient weights for ``dw/dx`` and ``dw/dy``:
  ``mpas_atm_core.F:1826-1844`` (``atm_initialize_deformation_weights``,
  ``deformation_coef_c`` / ``deformation_coef_s`` -- unlike the c2/s2/cs
  triple they are NOT sign-flipped by the ``cellsOnEdge`` orientation).
* Brunt-Vaisala frequency: ``mpas_atm_dissipation_models.F:462-573``
  (``calculate_n2``), dry branch and Durran-Klemp (1982) moist branch.
* eddy viscosities and the TKE source: ``mpas_atm_dissipation_models.F:
  208-458`` (``les_models``), called from ``mpas_atm_time_integration.F:
  6371-6386`` on RK step 1 of every dynamics substep with
  ``ur_cell = uReconstructZonal`` and ``vr_cell = uReconstructMeridional``.
* u: ``u_dissipation_3d`` (``mpas_atm_dissipation_models.F:577-945``) --
  the del2 term with ``tau_12_factor = 1`` (the LES "2 x grad div" term),
  the del4 background filter, and the LES vertical flux with the surface
  drag of ``config_les_surface``.
* w: ``w_dissipation_3d`` (``mpas_atm_dissipation_models.F:949-1151``).
* theta_m: ``scalar_dissipation_3d_les`` (``mpas_atm_dissipation_models.F:
  1155-1604``), including the specified/varying surface heat and moisture
  flux.

What the port does NOT carry, by name (each is refused or labelled where it
would otherwise be silent):

* ``config_mix_scalars`` (``scalar_dissipation_3d_les`` lines 1332-1407 and
  1557-1591): the hex scalar transport has no source slot for a dynamics
  tendency, so scalar LES mixing is refused at the configuration, exactly as
  before this module existed.
* TKE TRANSPORT.  Native carries ``tke`` in the scalar array (Registry
  ``var_array scalars``, group ``turbulence``, package ``les``) and advects
  it with the 3rd-order scalar transport using ``tend_scalars`` as its
  source.  The hex scalar block has no ``tke`` row, so this port owns the
  TKE field itself and advances it once per model step with the native
  source held from dynamics substep 1 plus a PORT-LOCAL first-order upwind
  flux-form transport on the step-start mass fluxes
  (:func:`advance_tke_v841`).  This is a declared divergence.
* TKE COLD START.  Native reads ``tke`` from the init file; hex inits carry
  none, and with ``tke = 0`` the closure has ``K = 0`` and no production
  forever.  The port therefore seeds a declared uniform cold-start value.
* ``les_surface = 'varying'`` needs ``hfx``/``qfx``/``ustm`` from a surface
  layer.  The authority below accepts them as arrays; the forecast seam
  refuses the option until a provider exists.

Notes on fidelity (verified against the source text, mirrored, NOT
corrected):

* the interior ``du/dz`` (lines 335-339) divides by
  ``zgrid(k+2)+zgrid(k+1)-zgrid(k)-zgrid(k-1)``, i.e. twice the
  mid-level spacing, so on uniform levels it is half the centred gradient;
  the end levels (lines 341-349) use a one-sided difference over one layer.
* ``d_13`` uses ``dw/dx`` at the bottom face of the layer (``dwdx(k)``).
* the 3-D Smagorinsky vertical viscosity is not capped; the horizontal one
  and both TKE viscosities are (lines 367, 409, 411).
* with ``les_surface = 'none'`` the u and theta vertical fluxes are
  zero-gradient at both ends (``flux(1) = flux(2)``,
  ``flux(nlev+1) = flux(nlev)``), lines 922-926 and 1544-1548.

Precision: the native reference builds run single precision, so float32
is this authority's execution dtype; every literal is typed in the working
precision, the convention of :mod:`woof.hex.mixing_v841`.  In a
double-precision native build, Fortran default-real literals such as
``0.01``, ``1./3.`` and ``0.76`` are single-precision constants promoted
to double; the float64 path here types them in double instead and is the
pinning scaffold, not a claim about such a build.

Array layout is level-first, ``(nVertLevels, nCells)`` / ``(nVertLevels,
nEdges)``; ``w`` and ``zgrid`` carry ``nVertLevels + 1`` interfaces.
Interface ``k`` (0-based, ``1..nlev-1``) uses ``rdzu[k]`` and
``fzm[k]*X[k] + fzp[k]*X[k-1]``, the convention of the hex dycore.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .errors import ConfigurationRefusal

FloatArray = NDArray[np.floating[Any]]

# ---------------------------------------------------------------------------
# option strings (mpas_atm_dissipation_models.F:30-116)
# ---------------------------------------------------------------------------
LES_MODEL_NONE = "none"
LES_MODEL_3D_SMAGORINSKY = "3d_smagorinsky"
LES_MODEL_PROGNOSTIC_15_ORDER = "prognostic_1.5_order"
LES_MODELS = (LES_MODEL_NONE, LES_MODEL_3D_SMAGORINSKY, LES_MODEL_PROGNOSTIC_15_ORDER)

LES_SURFACE_NONE = "none"
LES_SURFACE_SPECIFIED = "specified"
LES_SURFACE_VARYING = "varying"
LES_SURFACES = (LES_SURFACE_NONE, LES_SURFACE_SPECIFIED, LES_SURFACE_VARYING)

#: ``woof hex forecast --les-model`` spelling -> native Registry value.
LES_MODEL_CLI = {
    "off": LES_MODEL_NONE,
    "3d_smagorinsky": LES_MODEL_3D_SMAGORINSKY,
    "prognostic_tke": LES_MODEL_PROGNOSTIC_15_ORDER,
}
LES_MODEL_CLI_CHOICES = tuple(LES_MODEL_CLI)

#: ``woof hex forecast --les-surface`` choices the forecast seam can feed.
LES_SURFACE_CLI_CHOICES = (LES_SURFACE_NONE, LES_SURFACE_SPECIFIED)

#: PBL selections that mean "no PBL scheme runs".
PBL_OFF_VALUES = ("off", "none")

#: Port-local cold-start TKE for the prognostic closure (m^2 s^-2).  Native
#: reads ``tke`` from its init; see the module docstring.
DEFAULT_COLD_START_TKE = 0.1

# ---------------------------------------------------------------------------
# constants (mpas_atm_dissipation_models.F:26-28, mpas_constants.F, and the
# calculate_n2 locals at lines 467-483, identical to mpas_atmphys_constants)
# ---------------------------------------------------------------------------
C_K = 0.25
EPSILON_BV = 1.0e-06
PRANDTL = 1.0
GRAVITY = 9.80616
RGAS = 287.0
RV = 461.6
CP = 7.0 * RGAS / 2.0
XLV = 2.50e6
SVP1 = 0.6112
SVP2 = 17.67
SVP3 = 29.65
SVPT0 = 273.15
QC_CR = 0.00001

TKE_TRANSPORT_LABEL = (
    "port-local first-order upwind flux-form transport on the step-start "
    "mass fluxes; native advects tke through the 3rd-order scalar transport"
)
TKE_COLD_START_LABEL = (
    "port-local uniform cold start; native reads tke from its init file and "
    "hex inits carry none"
)


def _refuse(knob: str, value: object, reason: str, declaration: str) -> None:
    raise ConfigurationRefusal(knob, value, reason, declaration)


def les_model_from_cli(name: str) -> str:
    """Map the ``--les-model`` spelling to the native Registry value."""

    try:
        return LES_MODEL_CLI[str(name)]
    except KeyError:
        _refuse(
            "--les-model",
            name,
            "not an LES model this port carries",
            f"--les-model in {list(LES_MODEL_CLI_CHOICES)}",
        )
    raise AssertionError("unreachable")


def les_label(les_model: str) -> str:
    """The receipt/history label ``les_model=...`` for a native value."""

    return f"les_model={les_model}"


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def validate_les_selection(config: object) -> None:
    """Works-or-refuses check of the LES knobs on a dycore configuration.

    Reads ``config_les_model`` / ``config_les_surface`` and the knobs they
    depend on.  Called from :class:`woof.hex.config_v841.V841DryDycoreConfig`.
    The PBL rule: native runs the LES closure with the PBL scheme off
    (``config_pbl_scheme='off'``); a configuration that carries a PBL field
    must name it off, and one that carries none must be a dry dycore with no
    physics suite.  Anything else is refused naming ``--pbl off``.
    """

    model = getattr(config, "config_les_model", LES_MODEL_NONE)
    surface = getattr(config, "config_les_surface", LES_SURFACE_NONE)
    if model not in LES_MODELS:
        _refuse(
            "config_les_model",
            model,
            "not a v8.4.1 LES model (les_model_from_string returns "
            "LES_INVALID_OPT)",
            f"config_les_model in {list(LES_MODELS)}",
        )
    if surface not in LES_SURFACES:
        _refuse(
            "config_les_surface",
            surface,
            "not a v8.4.1 LES surface option (les_surface_from_string "
            "returns LES_INVALID_OPT)",
            f"config_les_surface in {list(LES_SURFACES)}",
        )
    if model == LES_MODEL_NONE:
        if surface != LES_SURFACE_NONE:
            _refuse(
                "config_les_surface",
                surface,
                "the LES surface fluxes are applied only inside the LES "
                "branches, so with config_les_model='none' they would be "
                "accepted and never read",
                "config_les_surface='none' or an LES model",
            )
        _refuse_unread_surface_knobs(config)
        return
    # The PBL rule.
    pbl = getattr(config, "config_pbl_scheme", None)
    if pbl is not None:
        if str(pbl) not in PBL_OFF_VALUES:
            _refuse(
                "config_pbl_scheme",
                pbl,
                f"config_les_model={model!r} replaces the boundary-layer "
                "closure: native v8.4.1 applies the LES vertical fluxes on "
                "top of whatever the PBL scheme does, so running both mixes "
                "the boundary layer twice",
                "--pbl off (config_pbl_scheme='off') with --les-model",
            )
    else:
        suite = str(getattr(config, "config_physics_suite", "none"))
        if suite != "none":
            _refuse(
                "config_pbl_scheme",
                None,
                f"config_les_model={model!r} requires the PBL scheme off and "
                f"this configuration (physics suite {suite!r}) does not say "
                "which PBL it runs",
                "--pbl off (config_pbl_scheme='off') with --les-model",
            )
    if getattr(config, "config_horiz_mixing", None) != "2d_smagorinsky":
        _refuse(
            "config_horiz_mixing",
            getattr(config, "config_horiz_mixing", None),
            "the port applies the LES closure at the explicit-mixing seam of "
            "the 2-D Smagorinsky lane (it supplies c_s, config_len_disp and "
            "the del4 background filter the LES branch reuses, "
            "mpas_atm_dissipation_models.F:274-275)",
            "config_horiz_mixing='2d_smagorinsky' (--horiz-mixing "
            "2d_smagorinsky) with --les-model",
        )
    for knob in ("config_v_mom_eddy_visc2", "config_v_theta_eddy_visc2"):
        value = float(getattr(config, knob, 0.0))
        if value != 0.0:
            _refuse(
                knob,
                value,
                "the constant vertical eddy viscosity branch is not ported "
                "beside the LES vertical fluxes",
                f"{knob}=0.0",
            )
    if bool(getattr(config, "config_mix_scalars", False)):
        _refuse(
            "config_mix_scalars",
            True,
            "LES scalar mixing needs a dynamics source slot in the hex "
            "scalar transport, which does not exist",
            "config_mix_scalars=false",
        )
    if surface != LES_SURFACE_SPECIFIED:
        _refuse_unread_surface_knobs(config)
    else:
        for knob in (
            "config_surface_heat_flux",
            "config_surface_moisture_flux",
            "config_surface_drag_coefficient",
        ):
            value = float(getattr(config, knob, 0.0))
            if not np.isfinite(value):
                _refuse(knob, value, "must be finite", f"a finite {knob}")
        drag = float(getattr(config, "config_surface_drag_coefficient", 0.0))
        if drag < 0.0:
            _refuse(
                "config_surface_drag_coefficient",
                drag,
                "a negative drag coefficient accelerates the surface wind",
                "config_surface_drag_coefficient>=0",
            )


def _refuse_unread_surface_knobs(config: object) -> None:
    for knob in (
        "config_surface_heat_flux",
        "config_surface_moisture_flux",
        "config_surface_drag_coefficient",
    ):
        value = float(getattr(config, knob, 0.0))
        if not np.isfinite(value) or value != 0.0:
            _refuse(
                knob,
                value,
                "this knob is read only by config_les_surface='specified' "
                "and must not be silently ignored",
                f"{knob}=0.0 or config_les_surface='specified'",
            )


@dataclass(frozen=True, slots=True)
class LesV841Config:
    """The numerical knobs the LES branches read, already resolved."""

    les_model: str
    les_surface: str = LES_SURFACE_NONE
    smagorinsky_coef: float = 0.125
    len_disp: float = 0.0
    visc4_2dsmag: float = 0.05
    del4u_div_factor: float = 10.0
    surface_heat_flux: float = 0.0
    surface_moisture_flux: float = 0.0
    surface_drag_coefficient: float = 0.0

    def validate(self) -> None:
        if self.les_model not in (LES_MODEL_3D_SMAGORINSKY, LES_MODEL_PROGNOSTIC_15_ORDER):
            _refuse(
                "config_les_model",
                self.les_model,
                "this authority runs an LES branch",
                "config_les_model in ['3d_smagorinsky', 'prognostic_1.5_order']",
            )
        if self.les_surface not in LES_SURFACES:
            _refuse(
                "config_les_surface",
                self.les_surface,
                "not a v8.4.1 LES surface option",
                f"config_les_surface in {list(LES_SURFACES)}",
            )
        for name in (
            "smagorinsky_coef",
            "len_disp",
            "visc4_2dsmag",
            "del4u_div_factor",
            "surface_heat_flux",
            "surface_moisture_flux",
            "surface_drag_coefficient",
        ):
            if not np.isfinite(float(getattr(self, name))):
                _refuse(name, getattr(self, name), "must be finite", f"a finite {name}")
        if float(self.len_disp) <= 0.0:
            _refuse(
                "config_len_disp",
                self.len_disp,
                "the LES length scale must be resolved and positive",
                "config_len_disp>0 (resolve 0 from nominalMinDc first)",
            )
        if float(self.smagorinsky_coef) < 0.0 or float(self.visc4_2dsmag) < 0.0:
            _refuse(
                "config_smagorinsky_coef",
                self.smagorinsky_coef,
                "coefficients must be non-negative",
                "config_smagorinsky_coef>=0 and config_visc4_2dsmag>=0",
            )

    @property
    def prognostic(self) -> bool:
        return self.les_model == LES_MODEL_PROGNOSTIC_15_ORDER


def les_config_from_dycore(config: object, *, len_disp: float) -> LesV841Config:
    """Build the resolved LES knobs from a validated dycore configuration."""

    validate_les_selection(config)
    result = LesV841Config(
        les_model=str(getattr(config, "config_les_model")),
        les_surface=str(getattr(config, "config_les_surface")),
        smagorinsky_coef=float(getattr(config, "config_smagorinsky_coef")),
        len_disp=float(len_disp),
        visc4_2dsmag=float(getattr(config, "config_visc4_2dsmag")),
        del4u_div_factor=float(getattr(config, "config_del4u_div_factor")),
        surface_heat_flux=float(getattr(config, "config_surface_heat_flux", 0.0)),
        surface_moisture_flux=float(getattr(config, "config_surface_moisture_flux", 0.0)),
        surface_drag_coefficient=float(
            getattr(config, "config_surface_drag_coefficient", 0.0)
        ),
    )
    result.validate()
    return result


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def _mesh_array(mesh: object, name: str) -> NDArray[Any]:
    arrays = getattr(mesh, "arrays", None)
    if isinstance(arrays, Mapping) and name in arrays:
        return np.asarray(arrays[name])
    try:
        return np.asarray(getattr(mesh, name))
    except AttributeError:
        raise AttributeError(f"mesh has no MPAS field {name!r}") from None


@dataclass(frozen=True, slots=True)
class LesGradientWeightsV841:
    """``deformation_coef_c`` / ``deformation_coef_s``, shape (nCells, maxEdges)."""

    coef_c: FloatArray
    coef_s: FloatArray


def initialize_les_gradient_weights_v841(
    mesh: object, *, dtype: Any = np.float32
) -> LesGradientWeightsV841:
    """Mirror ``deformation_coef_c/s`` of ``atm_initialize_deformation_weights``.

    ``mpas_atm_core.F:1826-1844``: on the same tangent-plane polygon as the
    c2/s2/cs triple (built exactly as :func:`woof.hex.mixing_v841.
    initialize_deformation_weights_v841` builds it), ``coef_c = dl *
    cos(theta_edge) / area`` and ``coef_s = dl * sin(theta_edge) / area``,
    with the orientation flip commented out in native (lines 1842-1843).
    Spherical meshes only; a cell with a neighbour outside the mesh keeps
    zero weights (the native halo guard, core lines 1702-1716).
    """

    from .mixing_v841 import _sphere_angle, _sphere_arc_length

    out_dtype = np.dtype(dtype)
    if out_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("gradient weights dtype must be float32 or float64")
    attrs = getattr(mesh, "attrs", {}) or {}
    if str(attrs.get("on_a_sphere", "NO")).strip().upper() != "YES":
        _refuse(
            "on_a_sphere",
            attrs.get("on_a_sphere"),
            "only the spherical deformation-weight branch is ported",
            "a spherical MPAS mesh",
        )
    radius = out_dtype.type(float(attrs["sphere_radius"]))
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("sphere_radius must be finite and positive")
    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    cells_on_cell = np.asarray(_mesh_array(mesh, "cellsOnCell"), dtype=np.int64)
    vertices_on_cell = np.asarray(_mesh_array(mesh, "verticesOnCell"), dtype=np.int64)
    x_cell = np.asarray(_mesh_array(mesh, "xCell"), dtype=out_dtype)
    y_cell = np.asarray(_mesh_array(mesh, "yCell"), dtype=out_dtype)
    z_cell = np.asarray(_mesh_array(mesh, "zCell"), dtype=out_dtype)
    x_vertex = np.asarray(_mesh_array(mesh, "xVertex"), dtype=out_dtype)
    y_vertex = np.asarray(_mesh_array(mesh, "yVertex"), dtype=out_dtype)
    z_vertex = np.asarray(_mesh_array(mesh, "zVertex"), dtype=out_dtype)
    n_cells, max_edges = vertices_on_cell.shape
    one = out_dtype.type(1.0)
    zero = out_dtype.type(0.0)
    quarter = out_dtype.type(0.25)
    two = one + one
    pii = two * np.arcsin(one)
    coef_c = np.zeros((n_cells, max_edges), dtype=out_dtype)
    coef_s = np.zeros((n_cells, max_edges), dtype=out_dtype)
    for cell in range(n_cells):
        count = int(counts[cell])
        if count < 3 or count > max_edges:
            raise ValueError(f"cell {cell} has invalid nEdgesOnCell {count}")
        neighbors = cells_on_cell[cell, :count]
        if np.any((neighbors < 0) | (neighbors >= n_cells)):
            continue
        verts = vertices_on_cell[cell, :count]
        cx = x_cell[cell] / radius
        cy = y_cell[cell] / radius
        cz = z_cell[cell] / radius
        vx = x_vertex[verts] / radius
        vy = y_vertex[verts] / radius
        vz = z_vertex[verts] / radius
        if cz == one:
            theta_abs = pii / two
        else:
            theta_abs = pii / two - _sphere_angle(
                cx, cy, cz, vx[0], vy[0], vz[0], zero, zero, one, one
            )
        thetav = np.zeros(count, dtype=out_dtype)
        dl_sphere = np.zeros(count, dtype=out_dtype)
        for j in range(count):
            jp1 = (j + 1) % count
            thetav[j] = _sphere_angle(
                cx, cy, cz, vx[j], vy[j], vz[j], vx[jp1], vy[jp1], vz[jp1], one
            )
            dl_sphere[j] = radius * _sphere_arc_length(
                cx, cy, cz, vx[j], vy[j], vz[j], one
            )
        thetat = np.zeros(count, dtype=out_dtype)
        thetat[0] = theta_abs
        for j in range(1, count):
            thetat[j] = thetat[j - 1] + thetav[j - 1]
        xp = np.cos(thetat) * dl_sphere
        yp = np.sin(thetat) * dl_sphere
        area_cell = zero
        theta_edge = np.zeros(count, dtype=out_dtype)
        for j in range(count):
            jp1 = (j + 1) % count
            dx = xp[jp1] - xp[j]
            dy = yp[jp1] - yp[j]
            area_cell = (
                area_cell
                + quarter * (xp[j] + xp[jp1]) * (yp[jp1] - yp[j])
                - quarter * (yp[j] + yp[jp1]) * (xp[jp1] - xp[j])
            )
            theta_edge[j] = np.arctan2(dy, dx) - pii / two
        for j in range(count):
            jp1 = (j + 1) % count
            dx = xp[jp1] - xp[j]
            dy = yp[jp1] - yp[j]
            dl = np.sqrt(dx * dx + dy * dy)
            coef_c[cell, j] = dl * np.cos(theta_edge[j]) / area_cell
            coef_s[cell, j] = dl * np.sin(theta_edge[j]) / area_cell
    if not (np.all(np.isfinite(coef_c)) and np.all(np.isfinite(coef_s))):
        raise ValueError("LES gradient weights contain non-finite values")
    return LesGradientWeightsV841(coef_c=coef_c, coef_s=coef_s)


@dataclass(frozen=True, slots=True)
class LesGeometryV841:
    """Host geometry every LES stencil reads (0-based, non-negative tables).

    Built from a closed (global) mesh, or from a padded regional view whose
    sentinels already point at a zero garbage element.
    """

    cells_on_edge: NDArray[np.int64]
    vertices_on_edge: NDArray[np.int64]
    edges_on_cell: NDArray[np.int64]
    n_edges_on_cell: NDArray[np.int64]
    edges_on_vertex: NDArray[np.int64]
    dc_edge: FloatArray
    dv_edge: FloatArray
    area_cell: FloatArray
    area_triangle: FloatArray
    scale_del2: FloatArray
    scale_del4: FloatArray
    coef_c2: FloatArray
    coef_s2: FloatArray
    coef_cs: FloatArray
    coef_c: FloatArray
    coef_s: FloatArray
    nominal_min_dc: float

    @property
    def n_cells(self) -> int:
        return int(self.n_edges_on_cell.size)

    @property
    def n_edges(self) -> int:
        return int(self.dc_edge.size)

    @property
    def n_vertices(self) -> int:
        return int(self.area_triangle.size)

    @property
    def max_edges(self) -> int:
        return int(self.edges_on_cell.shape[1])


def les_geometry_from_mesh(
    mesh: object,
    *,
    config_h_ScaleWithMesh: bool = True,
    dtype: Any = np.float32,
) -> LesGeometryV841:
    """Collect the host geometry and every weight table for a closed mesh."""

    from .mixing import compute_mesh_mixing_scaling
    from .mixing_v841 import initialize_deformation_weights_v841

    out = np.dtype(dtype)
    cells_on_edge = np.asarray(_mesh_array(mesh, "cellsOnEdge"), dtype=np.int64)
    if np.any(cells_on_edge < 0):
        _refuse(
            "cellsOnEdge",
            "negative sentinel",
            "the CPU LES authority gathers through every edge table and a "
            "regional sentinel would index outside the mesh",
            "a closed mesh, or a padded regional view",
        )
    deformation = initialize_deformation_weights_v841(mesh, dtype=out)
    gradient = initialize_les_gradient_weights_v841(mesh, dtype=out)
    scaling = compute_mesh_mixing_scaling(
        mesh, config_h_ScaleWithMesh=bool(config_h_ScaleWithMesh), dtype=out
    )
    nominal = np.asarray(_mesh_array(mesh, "nominalMinDc"), dtype=np.float64)
    return LesGeometryV841(
        cells_on_edge=cells_on_edge,
        vertices_on_edge=np.asarray(_mesh_array(mesh, "verticesOnEdge"), dtype=np.int64),
        edges_on_cell=np.asarray(_mesh_array(mesh, "edgesOnCell"), dtype=np.int64),
        n_edges_on_cell=np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64),
        edges_on_vertex=np.asarray(_mesh_array(mesh, "edgesOnVertex"), dtype=np.int64),
        dc_edge=np.asarray(_mesh_array(mesh, "dcEdge"), dtype=out),
        dv_edge=np.asarray(_mesh_array(mesh, "dvEdge"), dtype=out),
        area_cell=np.asarray(_mesh_array(mesh, "areaCell"), dtype=out),
        area_triangle=np.asarray(_mesh_array(mesh, "areaTriangle"), dtype=out),
        scale_del2=np.asarray(scaling.del2, dtype=out),
        scale_del4=np.asarray(scaling.del4, dtype=out),
        coef_c2=deformation.coef_c2,
        coef_s2=deformation.coef_s2,
        coef_cs=deformation.coef_cs,
        coef_c=gradient.coef_c,
        coef_s=gradient.coef_s,
        nominal_min_dc=float(nominal.reshape(-1)[0]),
    )


def _cell_edge_sign(geom: LesGeometryV841) -> FloatArray:
    """``edgesOnCell_sign`` (core lines 1236-1247): +1 for cellsOnEdge(1)."""

    eoc = geom.edges_on_cell
    active = np.arange(geom.max_edges)[None, :] < geom.n_edges_on_cell[:, None]
    first = geom.cells_on_edge[np.where(active, eoc, 0), 0]
    cells = np.arange(geom.n_cells)[:, None]
    return np.where(active, np.where(first == cells, 1.0, -1.0), 0.0)


def _vertex_edge_sign(geom: LesGeometryV841) -> FloatArray:
    """``edgesOnVertex_sign`` (core lines 1220-1229): +1 for verticesOnEdge(2)."""

    eov = geom.edges_on_vertex
    second = geom.vertices_on_edge[np.clip(eov, 0, None), 1]
    vertices = np.arange(eov.shape[0])[:, None]
    return np.where(second == vertices, 1.0, -1.0)


# ---------------------------------------------------------------------------
# calculate_n2 (mpas_atm_dissipation_models.F:462-573)
# ---------------------------------------------------------------------------
def calculate_n2_v841(
    *,
    theta_m: object,
    exner: object,
    pressure_base: object,
    pressure_p: object,
    zgrid: object,
    qv: object | None = None,
    qc: object | None = None,
    qtot: object | None = None,
) -> FloatArray:
    """Squared Brunt-Vaisala frequency ``bn2`` at cell centres.

    ``qv=None`` is a dry dycore (``theta = theta_m``, no vapour gradient);
    ``qc=None`` takes the dry branch everywhere (native ``index_qc <= 0``);
    ``qtot=None`` is a zero total condensate+vapour loading.
    """

    th_m = np.asarray(theta_m)
    dtype = th_m.dtype
    t = dtype.type
    nlev, ncells = th_m.shape
    if nlev < 3:
        raise ValueError("calculate_n2 needs at least three levels")
    ex = np.asarray(exner, dtype=dtype)
    pb = np.asarray(pressure_base, dtype=dtype)
    pp = np.asarray(pressure_p, dtype=dtype)
    zg = np.asarray(zgrid, dtype=dtype)
    vapour = np.zeros_like(th_m) if qv is None else np.asarray(qv, dtype=dtype)
    loading = np.zeros_like(th_m) if qtot is None else np.asarray(qtot, dtype=dtype)
    rvord = t(RV) / t(RGAS)
    ep_2 = t(RGAS) / t(RV)
    one = t(1.0)
    theta = th_m / (one + rvord * vapour)
    temp = ex * theta
    p = pb + pp
    esw = t(1000.0) * t(SVP1) * np.exp(t(SVP2) * (temp - t(SVPT0)) / (temp - t(SVP3)))
    esw = np.where(p < esw, p * t(0.99), esw)
    qvsw = ep_2 * esw / (p - esw)
    coefa = (one + t(XLV) * qvsw / t(RGAS) / temp) / (
        one + t(XLV) * t(XLV) * qvsw / t(CP) / t(RV) / temp / temp
    )
    bn2 = np.zeros_like(th_m)
    half = t(0.5)
    g = t(GRAVITY)
    for k in range(1, nlev - 1):
        dz = half * (zg[k + 2] + zg[k + 1]) - half * (zg[k] + zg[k - 1])
        rdz = one / dz
        dry = (
            g
            * (
                (theta[k + 1] - theta[k - 1]) / theta[k] * rdz
                + rvord * (vapour[k + 1] - vapour[k - 1]) * rdz
                - (loading[k + 1] - loading[k - 1]) * rdz
            )
        )
        moist = g * (
            coefa[k]
            * (
                (theta[k + 1] - theta[k - 1]) / theta[k] * rdz
                + t(XLV) / t(CP) / temp[k] * (qvsw[k + 1] - qvsw[k - 1]) * rdz
            )
            - (loading[k + 1] - loading[k - 1]) * rdz
        )
        if qc is None:
            bn2[k] = dry
        else:
            cloud = np.asarray(qc, dtype=dtype)[k]
            bn2[k] = np.where(cloud >= t(QC_CR), moist, dry)
    bn2[0] = bn2[1]
    bn2[nlev - 1] = bn2[nlev - 2]
    return bn2


# ---------------------------------------------------------------------------
# les_models (mpas_atm_dissipation_models.F:208-458)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LesEddyViscosityV841:
    eddy_visc_horz: FloatArray
    eddy_visc_vert: FloatArray
    #: ``prandtl_3d_inv``; native leaves it unset in the 3-D Smagorinsky
    #: branch (where nothing reads it), this port writes ``1/prandtl``.
    prandtl_3d_inv: FloatArray
    #: ``rho_zz * (shear + buoyancy + dissipation)``; ``None`` unless the
    #: prognostic branch ran on dynamics substep 1.
    tend_tke: FloatArray | None
    #: the TKE after the native ``max(0, tke)`` bound, or ``None``.
    tke: FloatArray | None
    h_mom_eddy_visc4: np.floating[Any]
    h_theta_eddy_visc4: np.floating[Any]


def _strain_tensor(
    geom: LesGeometryV841,
    u: FloatArray,
    v: FloatArray,
    ur_cell: FloatArray,
    vr_cell: FloatArray,
    w: FloatArray,
    zgrid: FloatArray,
) -> tuple[FloatArray, ...]:
    """``d_11 .. d_23`` at cell centres (les_models lines 280-359)."""

    dtype = u.dtype
    t = dtype.type
    nlev = u.shape[0]
    ncells = geom.n_cells
    dudx = np.zeros((nlev, ncells), dtype=dtype)
    dudy = np.zeros_like(dudx)
    dvdx = np.zeros_like(dudx)
    dvdy = np.zeros_like(dudx)
    dwdx = np.zeros((nlev + 1, ncells), dtype=dtype)
    dwdy = np.zeros_like(dwdx)
    c2 = geom.coef_c2.astype(dtype, copy=False)
    s2 = geom.coef_s2.astype(dtype, copy=False)
    cs = geom.coef_cs.astype(dtype, copy=False)
    cc = geom.coef_c.astype(dtype, copy=False)
    ss = geom.coef_s.astype(dtype, copy=False)
    half = t(0.5)
    for slot in range(geom.max_edges):
        active = slot < geom.n_edges_on_cell
        edge = np.where(active, geom.edges_on_cell[:, slot], 0)
        ue = u[:, edge]
        ve = v[:, edge]
        a_c2 = c2[:, slot]
        a_s2 = s2[:, slot]
        a_cs = cs[:, slot]
        dudx = np.where(active, dudx + (a_c2 * ue - a_cs * ve), dudx)
        dudy = np.where(active, dudy + (a_cs * ue - a_s2 * ve), dudy)
        dvdx = np.where(active, dvdx + (a_cs * ue + a_c2 * ve), dvdx)
        dvdy = np.where(active, dvdy + (a_s2 * ue + a_cs * ve), dvdy)
        cell1 = geom.cells_on_edge[edge, 0]
        cell2 = geom.cells_on_edge[edge, 1]
        wk = half * (w[:, cell1] + w[:, cell2])
        dwdx = np.where(active, dwdx + cc[:, slot] * wk, dwdx)
        dwdy = np.where(active, dwdy + ss[:, slot] * wk, dwdy)
    one = t(1.0)
    dwdz = (w[1:] - w[:-1]) * (one / (zgrid[1:] - zgrid[:-1]))
    dudz = np.zeros((nlev, ncells), dtype=dtype)
    dvdz = np.zeros_like(dudz)
    for k in range(1, nlev - 1):
        rdz = one / (zgrid[k + 2] + zgrid[k + 1] - zgrid[k] - zgrid[k - 1])
        dudz[k] = (ur_cell[k + 1] - ur_cell[k - 1]) * rdz
        dvdz[k] = (vr_cell[k + 1] - vr_cell[k - 1]) * rdz
    rdz = one / (zgrid[1] - zgrid[0])
    dudz[0] = (ur_cell[1] - ur_cell[0]) * rdz
    dvdz[0] = (vr_cell[1] - vr_cell[0]) * rdz
    k = nlev - 2
    rdz = one / (zgrid[k + 1] - zgrid[k])
    dudz[k + 1] = (ur_cell[k + 1] - ur_cell[k]) * rdz
    dvdz[k + 1] = (vr_cell[k + 1] - vr_cell[k]) * rdz
    two = t(2.0)
    d_11 = two * dudx
    d_22 = two * dvdy
    d_33 = two * dwdz
    d_12 = dudy + dvdx
    d_13 = dwdx[:nlev] + dudz
    d_23 = dwdy[:nlev] + dvdz
    return d_11, d_22, d_33, d_12, d_13, d_23


def les_models_v841(
    geom: LesGeometryV841,
    config: LesV841Config,
    *,
    u: object,
    v: object,
    ur_cell: object,
    vr_cell: object,
    w: object,
    bn2: object,
    zgrid: object,
    rho_zz: object,
    dt: float,
    tke: object | None = None,
    dynamics_substep: int = 1,
) -> LesEddyViscosityV841:
    """Mirror ``les_models``: eddy viscosities and, for TKE, its source."""

    config.validate()
    u = np.asarray(u)
    dtype = u.dtype
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("LES authority dtype must be float32 or float64")
    t = dtype.type
    v = np.asarray(v, dtype=dtype)
    ur = np.asarray(ur_cell, dtype=dtype)
    vr = np.asarray(vr_cell, dtype=dtype)
    ww = np.asarray(w, dtype=dtype)
    n2 = np.asarray(bn2, dtype=dtype)
    zg = np.asarray(zgrid, dtype=dtype)
    rho = np.asarray(rho_zz, dtype=dtype)
    timestep = float(dt)
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("dt must be finite and positive")
    d_11, d_22, d_33, d_12, d_13, d_23 = _strain_tensor(geom, u, v, ur, vr, ww, zg)
    length = t(config.len_disp)
    c_s = t(config.smagorinsky_coef)
    inv_dt = t(1.0) / t(timestep)
    pr_inv = t(1.0) / t(PRANDTL)
    h4 = t(config.visc4_2dsmag) * (length * length * length)
    delta_z = zg[1:] - zg[:-1]
    ceiling_h = (t(0.01) * (length * length)) * inv_dt
    zero = t(0.0)
    half = t(0.5)
    if config.les_model == LES_MODEL_3D_SMAGORINSKY:
        def2 = (
            half * (d_11 * d_11 + d_22 * d_22 + d_33 * d_33)
            + d_12 * d_12
            + d_13 * d_13
            + d_23 * d_23
        )
        root = np.sqrt(np.maximum(zero, def2 - pr_inv * n2))
        kh = np.minimum(((c_s * length) * (c_s * length)) * root, ceiling_h)
        kv = ((c_s * delta_z) * (c_s * delta_z)) * root
        return LesEddyViscosityV841(
            eddy_visc_horz=kh,
            eddy_visc_vert=kv,
            prandtl_3d_inv=np.full_like(kh, pr_inv),
            tend_tke=None,
            tke=None,
            h_mom_eddy_visc4=h4,
            h_theta_eddy_visc4=h4,
        )
    if tke is None:
        raise ValueError("the prognostic 1.5-order closure needs the tke field")
    e = np.maximum(zero, np.asarray(tke, dtype=dtype))
    third = t(1.0) / t(3.0)
    delta_s = ((length * length) * delta_z) ** third
    bv = np.maximum(np.sqrt(np.abs(n2)), t(EPSILON_BV))
    sqrt_e = np.sqrt(e)
    tke_length = np.where(n2 > t(1.0e-06), t(0.76) * sqrt_e / bv, delta_s)
    tke_length = np.minimum(tke_length, delta_z)
    diss_length = np.minimum(delta_s, np.maximum(tke_length, t(0.01) * delta_s))
    diss_length = np.where(n2 <= zero, delta_s, diss_length)
    l_horizontal = length
    l_vertical = np.minimum(delta_z, tke_length)
    diss_length = np.where(n2 <= zero, delta_z, diss_length)
    c_k = t(C_K)
    kh = np.minimum(c_k * l_horizontal * sqrt_e, ceiling_h)
    kv = np.minimum(c_k * l_vertical * sqrt_e, (t(0.01) * (delta_z * delta_z)) * inv_dt)
    shear = kh * (d_11 * d_11 + d_22 * d_22 + d_12 * d_12) + kv * (
        d_33 * d_33 + d_13 * d_13 + d_23 * d_23
    )
    buoyancy = -kv * n2
    c_diss = t(1.9) * c_k + np.maximum(zero, t(0.93) - t(1.9) * c_k) * diss_length / delta_s
    dissipation = -c_diss * (e ** t(1.5)) / diss_length
    prandtl_3d_inv = t(1.0) + (t(2.0) * l_vertical / delta_z)
    tend = None
    if int(dynamics_substep) == 1:
        tend = rho * (shear + buoyancy + dissipation)
    return LesEddyViscosityV841(
        eddy_visc_horz=kh,
        eddy_visc_vert=kv,
        prandtl_3d_inv=prandtl_3d_inv,
        tend_tke=tend,
        tke=e,
        h_mom_eddy_visc4=h4,
        h_theta_eddy_visc4=h4,
    )


# ---------------------------------------------------------------------------
# u_dissipation_3d (mpas_atm_dissipation_models.F:577-945), LES branches
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LesSurfaceFieldsV841:
    """``les_surface='varying'`` inputs: ``ustm`` (m/s), ``hfx`` (W/m2), ``qfx``."""

    ustm: FloatArray
    hfx: FloatArray
    qfx: FloatArray


def u_dissipation_les_v841(
    geom: LesGeometryV841,
    config: LesV841Config,
    *,
    u: object,
    v: object,
    divergence: object,
    vorticity: object,
    rho_edge: object,
    rho_zz: object,
    zz: object,
    rdzu: object,
    rdzw: object,
    fzm: object,
    fzp: object,
    eddy_visc_horz: object,
    eddy_visc_vert: object,
    h_mom_eddy_visc4: float,
    surface: LesSurfaceFieldsV841 | None = None,
) -> FloatArray:
    """``tend_u_euler`` of the LES branch: del2 (with ``tau_12``), del4, vertical."""

    uu = np.asarray(u)
    dtype = uu.dtype
    t = dtype.type
    vv = np.asarray(v, dtype=dtype)
    div = np.asarray(divergence, dtype=dtype)
    vort = np.asarray(vorticity, dtype=dtype)
    rho_e = np.asarray(rho_edge, dtype=dtype)
    rho = np.asarray(rho_zz, dtype=dtype)
    zzc = np.asarray(zz, dtype=dtype)
    rdzu_ = np.asarray(rdzu, dtype=dtype)
    rdzw_ = np.asarray(rdzw, dtype=dtype)
    fzm_ = np.asarray(fzm, dtype=dtype)
    fzp_ = np.asarray(fzp, dtype=dtype)
    kh = np.asarray(eddy_visc_horz, dtype=dtype)
    kv = np.asarray(eddy_visc_vert, dtype=dtype)
    nlev = uu.shape[0]
    c1 = geom.cells_on_edge[:, 0]
    c2 = geom.cells_on_edge[:, 1]
    v1 = geom.vertices_on_edge[:, 0]
    v2 = geom.vertices_on_edge[:, 1]
    one = t(1.0)
    inv_dc = one / geom.dc_edge.astype(dtype)
    inv_dv = one / geom.dv_edge.astype(dtype)
    r_dc = inv_dc
    r_dv = np.minimum(inv_dv, t(4.0) * inv_dc)
    tau_12 = one  # les_model_opt /= LES_MODEL_NONE (line 685)
    grad_div = (div[:, c2] - div[:, c1]) * r_dc
    u_diffusion = grad_div - (vort[:, v2] - vort[:, v1]) * r_dv
    u_diffusion_les = u_diffusion + tau_12 * grad_div
    delsq_u = u_diffusion
    half = t(0.5)
    kdiffu = half * (kh[:, c1] + kh[:, c2])
    scale2 = geom.scale_del2.astype(dtype)
    tend = rho_e * kdiffu * u_diffusion_les * scale2
    h4 = t(h_mom_eddy_visc4)
    if h4 > t(0.0):
        nv = geom.n_vertices
        delsq_vort = np.zeros((nlev, nv), dtype=dtype)
        vsign = _vertex_edge_sign(geom).astype(dtype)
        inv_at = one / geom.area_triangle.astype(dtype)
        dc = geom.dc_edge.astype(dtype)
        for slot in range(geom.edges_on_vertex.shape[1]):
            edge = geom.edges_on_vertex[:, slot]
            edge_sign = inv_at * dc[edge] * vsign[:, slot]
            delsq_vort = delsq_vort + edge_sign * delsq_u[:, edge]
        delsq_div = np.zeros((nlev, geom.n_cells), dtype=dtype)
        csign = _cell_edge_sign(geom).astype(dtype)
        inv_ac = one / geom.area_cell.astype(dtype)
        dv = geom.dv_edge.astype(dtype)
        for slot in range(geom.max_edges):
            active = slot < geom.n_edges_on_cell
            edge = np.where(active, geom.edges_on_cell[:, slot], 0)
            edge_sign = inv_ac * dv[edge] * csign[:, slot]
            delsq_div = np.where(active, delsq_div + edge_sign * delsq_u[:, edge], delsq_div)
        u_mix_scale = geom.scale_del4.astype(dtype) * h4
        r_dc4 = u_mix_scale * t(config.del4u_div_factor) * inv_dc
        r_dv4 = u_mix_scale * np.minimum(inv_dv, t(4.0) * inv_dc)
        filt = rho_e * (
            (delsq_div[:, c2] - delsq_div[:, c1]) * r_dc4
            - (delsq_vort[:, v2] - delsq_vort[:, v1]) * r_dv4
        )
        tend = tend - filt
    # LES vertical flux (lines 886-938).
    nedges = uu.shape[1]
    flux = np.zeros((nlev + 1, nedges), dtype=dtype)
    for k in range(1, nlev):
        rho_k_1 = fzm_[k] * rho[k, c1] * zzc[k, c1] * kv[k, c1] + fzp_[k] * rho[k - 1, c1] * zzc[k - 1, c1] * kv[k - 1, c1]
        rho_k_2 = fzm_[k] * rho[k, c2] * zzc[k, c2] * kv[k, c2] + fzp_[k] * rho[k - 1, c2] * zzc[k - 1, c2] * kv[k - 1, c2]
        rho_k_at_w = half * (rho_k_1 + rho_k_2)
        zz_1 = fzm_[k] * zzc[k, c1] + fzp_[k] * zzc[k - 1, c1]
        zz_2 = fzm_[k] * zzc[k, c2] + fzp_[k] * zzc[k - 1, c2]
        zz_at_w = half * (zz_1 + zz_2)
        flux[k] = -rho_k_at_w * zz_at_w * rdzu_[k] * (uu[k] - uu[k - 1])
    if config.les_surface == LES_SURFACE_SPECIFIED:
        speed = np.sqrt(uu[0] * uu[0] + vv[0] * vv[0])
        flux[0] = -rho_e[0] * t(config.surface_drag_coefficient) * uu[0] * speed
        flux[nlev] = flux[nlev - 1]
    elif config.les_surface == LES_SURFACE_VARYING:
        if surface is None:
            raise ValueError("les_surface='varying' needs ustm/hfx/qfx")
        ustm = np.asarray(surface.ustm, dtype=dtype)
        ust_edge = half * (ustm[c1] + ustm[c2])
        speed = np.maximum(np.sqrt(uu[0] * uu[0] + vv[0] * vv[0]), t(0.1))
        flux[0] = -rho_e[0] * ust_edge * ust_edge * (uu[0] / speed)
        flux[nlev] = flux[nlev - 1]
    else:
        flux[0] = flux[1]
        flux[nlev] = flux[nlev - 1]
    tend = tend - rdzw_[:, None] * (flux[1:] - flux[:-1])
    return tend


def w_dissipation_les_v841(
    geom: LesGeometryV841,
    *,
    w: object,
    rho_edge: object,
    rho_zz: object,
    divergence: object,
    zz: object,
    rdzu: object,
    rdzw: object,
    eddy_visc_horz: object,
    eddy_visc_vert: object,
    h_mom_eddy_visc4: float,
) -> FloatArray:
    """``tend_w_euler`` of ``w_dissipation_3d`` with an LES model active."""

    ww = np.asarray(w)
    dtype = ww.dtype
    t = dtype.type
    rho_e = np.asarray(rho_edge, dtype=dtype)
    rho = np.asarray(rho_zz, dtype=dtype)
    div = np.asarray(divergence, dtype=dtype)
    zzc = np.asarray(zz, dtype=dtype)
    rdzu_ = np.asarray(rdzu, dtype=dtype)
    rdzw_ = np.asarray(rdzw, dtype=dtype)
    kh = np.asarray(eddy_visc_horz, dtype=dtype)
    kv = np.asarray(eddy_visc_vert, dtype=dtype)
    nlev = rho.shape[0]
    ncells = geom.n_cells
    tend = np.zeros((nlev + 1, ncells), dtype=dtype)
    delsq_w = np.zeros((nlev, ncells), dtype=dtype)
    csign = _cell_edge_sign(geom).astype(dtype)
    one = t(1.0)
    inv_ac = one / geom.area_cell.astype(dtype)
    inv_dc = one / geom.dc_edge.astype(dtype)
    dv = geom.dv_edge.astype(dtype)
    scale2 = geom.scale_del2.astype(dtype)
    scale4 = geom.scale_del4.astype(dtype)
    half = t(0.5)
    quarter = t(0.25)
    for slot in range(geom.max_edges):
        active = slot < geom.n_edges_on_cell
        edge = np.where(active, geom.edges_on_cell[:, slot], 0)
        edge_sign = half * inv_ac * csign[:, slot] * dv[edge] * inv_dc[edge]
        c1 = geom.cells_on_edge[edge, 0]
        c2 = geom.cells_on_edge[edge, 1]
        for k in range(1, nlev):
            flux = edge_sign * (rho_e[k, edge] + rho_e[k - 1, edge]) * (ww[k, c2] - ww[k, c1])
            delsq_w[k] = np.where(active, delsq_w[k] + flux, delsq_w[k])
            flux = flux * scale2[edge] * quarter * (
                kh[k, c1] + kh[k, c2] + kh[k - 1, c1] + kh[k - 1, c2]
            )
            tend[k] = np.where(active, tend[k] + flux, tend[k])
    h4 = t(h_mom_eddy_visc4)
    if h4 > t(0.0):
        r_area = h4 * inv_ac
        for slot in range(geom.max_edges):
            active = slot < geom.n_edges_on_cell
            edge = np.where(active, geom.edges_on_cell[:, slot], 0)
            c1 = geom.cells_on_edge[edge, 0]
            c2 = geom.cells_on_edge[edge, 1]
            edge_sign = scale4[edge] * r_area * dv[edge] * csign[:, slot] * inv_dc[edge]
            for k in range(1, nlev):
                tend[k] = np.where(
                    active, tend[k] - edge_sign * (delsq_w[k, c2] - delsq_w[k, c1]), tend[k]
                )
    two = t(2.0)
    flux = -rho * kv * zzc * (two * zzc * rdzw_[:, None] * (ww[1:] - ww[:-1]) + div)
    for k in range(1, nlev):
        tend[k] = tend[k] - rdzu_[k] * (flux[k] - flux[k - 1])
    return tend


def theta_dissipation_les_v841(
    geom: LesGeometryV841,
    config: LesV841Config,
    *,
    theta_m: object,
    rho_edge: object,
    rho_zz: object,
    zz: object,
    rdzu: object,
    rdzw: object,
    fzm: object,
    fzp: object,
    eddy_visc_horz: object,
    eddy_visc_vert: object,
    prandtl_3d_inv: object,
    h_theta_eddy_visc4: float,
    qv_lowest: object | None = None,
    surface: LesSurfaceFieldsV841 | None = None,
) -> FloatArray:
    """``tend_theta_euler`` of ``scalar_dissipation_3d_les`` (theta branch)."""

    th = np.asarray(theta_m)
    dtype = th.dtype
    t = dtype.type
    rho_e = np.asarray(rho_edge, dtype=dtype)
    rho = np.asarray(rho_zz, dtype=dtype)
    zzc = np.asarray(zz, dtype=dtype)
    rdzu_ = np.asarray(rdzu, dtype=dtype)
    rdzw_ = np.asarray(rdzw, dtype=dtype)
    fzm_ = np.asarray(fzm, dtype=dtype)
    fzp_ = np.asarray(fzp, dtype=dtype)
    kh = np.asarray(eddy_visc_horz, dtype=dtype)
    kv = np.asarray(eddy_visc_vert, dtype=dtype)
    pr3d = np.asarray(prandtl_3d_inv, dtype=dtype)
    nlev, ncells = th.shape
    one = t(1.0)
    half = t(0.5)
    prandtl_inv = one / t(PRANDTL)
    csign = _cell_edge_sign(geom).astype(dtype)
    inv_ac = one / geom.area_cell.astype(dtype)
    inv_dc = one / geom.dc_edge.astype(dtype)
    dv = geom.dv_edge.astype(dtype)
    scale2 = geom.scale_del2.astype(dtype)
    scale4 = geom.scale_del4.astype(dtype)
    tend = np.zeros((nlev, ncells), dtype=dtype)
    delsq = np.zeros((nlev, ncells), dtype=dtype)
    for slot in range(geom.max_edges):
        active = slot < geom.n_edges_on_cell
        edge = np.where(active, geom.edges_on_cell[:, slot], 0)
        edge_sign = inv_ac * csign[:, slot] * dv[edge] * inv_dc[edge]
        pr_scale = prandtl_inv * scale2[edge]
        c1 = geom.cells_on_edge[edge, 0]
        c2 = geom.cells_on_edge[edge, 1]
        flux = edge_sign * (th[:, c2] - th[:, c1]) * rho_e[:, edge]
        delsq = np.where(active, delsq + flux, delsq)
        flux = flux * half * (kh[:, c1] + kh[:, c2]) * pr_scale
        tend = np.where(active, tend + flux, tend)
    h4 = t(h_theta_eddy_visc4)
    if h4 > t(0.0):
        r_area = h4 * prandtl_inv * inv_ac
        for slot in range(geom.max_edges):
            active = slot < geom.n_edges_on_cell
            edge = np.where(active, geom.edges_on_cell[:, slot], 0)
            edge_sign = scale4[edge] * r_area * dv[edge] * csign[:, slot] * inv_dc[edge]
            c1 = geom.cells_on_edge[edge, 0]
            c2 = geom.cells_on_edge[edge, 1]
            tend = np.where(active, tend - edge_sign * (delsq[:, c2] - delsq[:, c1]), tend)
    # vertical (lines 1470-1555)
    flux = np.zeros((nlev + 1, ncells), dtype=dtype)
    for k in range(1, nlev):
        if config.les_model == LES_MODEL_3D_SMAGORINSKY:
            pr1d = prandtl_inv
        else:
            pr1d = fzm_[k] * pr3d[k] + fzp_[k] * pr3d[k - 1]
        rho_k_at_w = fzm_[k] * rho[k] * zzc[k] * zzc[k] * kv[k] + fzp_[k] * rho[k - 1] * zzc[k - 1] * zzc[k - 1] * kv[k - 1]
        zz_at_w = fzm_[k] * zzc[k] + fzp_[k] * zzc[k - 1]
        flux[k] = -pr1d * rho_k_at_w * zz_at_w * rdzu_[k] * (th[k] - th[k - 1])
    if config.les_surface in (LES_SURFACE_SPECIFIED, LES_SURFACE_VARYING):
        if config.les_surface == LES_SURFACE_SPECIFIED:
            moisture_flux = np.full(ncells, t(config.surface_moisture_flux), dtype=dtype)
            heat_flux = np.full(ncells, t(config.surface_heat_flux), dtype=dtype)
        else:
            if surface is None:
                raise ValueError("les_surface='varying' needs ustm/hfx/qfx")
            heat_flux = np.asarray(surface.hfx, dtype=dtype) / rho[0] / t(CP)
            moisture_flux = np.asarray(surface.qfx, dtype=dtype) / rho[0]
        qv_cell = (
            np.zeros(ncells, dtype=dtype)
            if qv_lowest is None
            else np.asarray(qv_lowest, dtype=dtype)
        )
        rvord = t(RV) / t(RGAS)
        theta_cell = th[0] / (one + rvord * qv_cell)
        theta_m_flux = heat_flux * (one + rvord * qv_cell) + rvord * theta_cell * moisture_flux
        flux[0] = theta_m_flux * rho[0]
        flux[nlev] = flux[nlev - 1]
    else:
        flux[0] = flux[1]
        flux[nlev] = flux[nlev - 1]
    tend = tend - rdzw_[:, None] * (flux[1:] - flux[:-1])
    return tend


# ---------------------------------------------------------------------------
# the composed RK-step-1 increments
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LesTendenciesV841:
    bn2: FloatArray
    eddy_visc_horz: FloatArray
    eddy_visc_vert: FloatArray
    prandtl_3d_inv: FloatArray
    tend_u_euler: FloatArray
    tend_w_euler: FloatArray
    tend_theta_euler: FloatArray
    tend_tke: FloatArray | None
    tke: FloatArray | None
    h_mom_eddy_visc4: np.floating[Any]


def compute_les_tendencies_v841(
    geom: LesGeometryV841,
    config: LesV841Config,
    *,
    u: object,
    v: object,
    ur_cell: object,
    vr_cell: object,
    w: object,
    theta_m: object,
    rho_edge: object,
    rho_zz: object,
    divergence: object,
    vorticity: object,
    exner: object,
    pressure_base: object,
    pressure_p: object,
    zgrid: object,
    zz: object,
    rdzu: object,
    rdzw: object,
    fzm: object,
    fzp: object,
    dt: float,
    qv: object | None = None,
    qc: object | None = None,
    qtot: object | None = None,
    tke: object | None = None,
    dynamics_substep: int = 1,
    surface: LesSurfaceFieldsV841 | None = None,
) -> LesTendenciesV841:
    """``calculate_n2`` -> ``les_models`` -> u/w/theta LES dissipation."""

    bn2 = calculate_n2_v841(
        theta_m=theta_m,
        exner=exner,
        pressure_base=pressure_base,
        pressure_p=pressure_p,
        zgrid=zgrid,
        qv=qv,
        qc=qc,
        qtot=qtot,
    )
    visc = les_models_v841(
        geom,
        config,
        u=u,
        v=v,
        ur_cell=ur_cell,
        vr_cell=vr_cell,
        w=w,
        bn2=bn2,
        zgrid=zgrid,
        rho_zz=rho_zz,
        dt=dt,
        tke=tke,
        dynamics_substep=dynamics_substep,
    )
    tend_u = u_dissipation_les_v841(
        geom,
        config,
        u=u,
        v=v,
        divergence=divergence,
        vorticity=vorticity,
        rho_edge=rho_edge,
        rho_zz=rho_zz,
        zz=zz,
        rdzu=rdzu,
        rdzw=rdzw,
        fzm=fzm,
        fzp=fzp,
        eddy_visc_horz=visc.eddy_visc_horz,
        eddy_visc_vert=visc.eddy_visc_vert,
        h_mom_eddy_visc4=visc.h_mom_eddy_visc4,
        surface=surface,
    )
    tend_w = w_dissipation_les_v841(
        geom,
        w=w,
        rho_edge=rho_edge,
        rho_zz=rho_zz,
        divergence=divergence,
        zz=zz,
        rdzu=rdzu,
        rdzw=rdzw,
        eddy_visc_horz=visc.eddy_visc_horz,
        eddy_visc_vert=visc.eddy_visc_vert,
        h_mom_eddy_visc4=visc.h_mom_eddy_visc4,
    )
    qv_lowest = None if qv is None else np.asarray(qv)[0]
    tend_theta = theta_dissipation_les_v841(
        geom,
        config,
        theta_m=theta_m,
        rho_edge=rho_edge,
        rho_zz=rho_zz,
        zz=zz,
        rdzu=rdzu,
        rdzw=rdzw,
        fzm=fzm,
        fzp=fzp,
        eddy_visc_horz=visc.eddy_visc_horz,
        eddy_visc_vert=visc.eddy_visc_vert,
        prandtl_3d_inv=visc.prandtl_3d_inv,
        h_theta_eddy_visc4=visc.h_theta_eddy_visc4,
        qv_lowest=qv_lowest,
        surface=surface,
    )
    return LesTendenciesV841(
        bn2=bn2,
        eddy_visc_horz=visc.eddy_visc_horz,
        eddy_visc_vert=visc.eddy_visc_vert,
        prandtl_3d_inv=visc.prandtl_3d_inv,
        tend_u_euler=tend_u,
        tend_w_euler=tend_w,
        tend_theta_euler=tend_theta,
        tend_tke=visc.tend_tke,
        tke=visc.tke,
        h_mom_eddy_visc4=visc.h_mom_eddy_visc4,
    )


# ---------------------------------------------------------------------------
# TKE step (port-local transport; see the module docstring)
# ---------------------------------------------------------------------------
def advance_tke_v841(
    geom: LesGeometryV841,
    *,
    tke: object,
    tend_tke: object,
    rho_zz: object,
    rho_u: object,
    rho_w: object,
    rdzw: object,
    dt: float,
) -> FloatArray:
    """One model step of the TKE field.

    ``e_new = max(0, e + dt * (tend_tke - A) / rho_zz)`` where ``tend_tke``
    is the native ``rho_zz * (shear + buoyancy + dissipation)`` held from
    dynamics substep 1 (the scalar-transport update ``scalars_old +
    dt*scalar_tend/rho_zz_old``, ``mpas_atm_time_integration.F:5208``) and
    ``A`` is first-order upwind advection in the conservative-minus-continuity
    form: for every face, ``F*(e_up - e_cell)`` summed over inflow faces, so
    a uniform field is preserved exactly.  Horizontal faces use the edge
    mass flux ``rho_u`` and ``dvEdge/areaCell``; vertical faces use ``rho_w``
    and ``rdzw``, with no flux through the bottom and top.
    """

    e = np.asarray(tke)
    dtype = e.dtype
    t = dtype.type
    src = np.asarray(tend_tke, dtype=dtype)
    rho = np.asarray(rho_zz, dtype=dtype)
    ru = np.asarray(rho_u, dtype=dtype)
    rw = np.asarray(rho_w, dtype=dtype)
    rdzw_ = np.asarray(rdzw, dtype=dtype)
    nlev, ncells = e.shape
    one = t(1.0)
    zero = t(0.0)
    csign = _cell_edge_sign(geom).astype(dtype)
    inv_ac = one / geom.area_cell.astype(dtype)
    dv = geom.dv_edge.astype(dtype)
    adv = np.zeros_like(e)
    for slot in range(geom.max_edges):
        active = slot < geom.n_edges_on_cell
        edge = np.where(active, geom.edges_on_cell[:, slot], 0)
        c1 = geom.cells_on_edge[edge, 0]
        c2 = geom.cells_on_edge[edge, 1]
        neighbour = np.where(c1 == np.arange(ncells), c2, c1)
        # outward mass flux through this face (positive leaves the cell)
        outward = csign[:, slot] * ru[:, edge] * dv[edge] * inv_ac
        inflow = np.maximum(-outward, zero)
        term = inflow * (e[:, neighbour] - e)
        adv = np.where(active, adv + term, adv)
    # vertical faces: rw[k] at interface k, positive upward
    for k in range(nlev):
        if k > 0:
            up_from_below = np.maximum(rw[k], zero)
            adv[k] = adv[k] + rdzw_[k] * up_from_below * (e[k - 1] - e[k])
        if k < nlev - 1:
            down_from_above = np.maximum(-rw[k + 1], zero)
            adv[k] = adv[k] + rdzw_[k] * down_from_above * (e[k + 1] - e[k])
    # adv is the inflow-weighted tendency of rho_zz*e (gain), so it adds.
    updated = e + t(dt) * (src + adv) / rho
    return np.maximum(zero, updated)


__all__ = [
    "C_K",
    "DEFAULT_COLD_START_TKE",
    "LES_MODELS",
    "LES_MODEL_3D_SMAGORINSKY",
    "LES_MODEL_CLI",
    "LES_MODEL_CLI_CHOICES",
    "LES_MODEL_NONE",
    "LES_MODEL_PROGNOSTIC_15_ORDER",
    "LES_SURFACES",
    "LES_SURFACE_CLI_CHOICES",
    "LES_SURFACE_NONE",
    "LES_SURFACE_SPECIFIED",
    "LES_SURFACE_VARYING",
    "LesEddyViscosityV841",
    "LesGeometryV841",
    "LesGradientWeightsV841",
    "LesSurfaceFieldsV841",
    "LesTendenciesV841",
    "LesV841Config",
    "TKE_COLD_START_LABEL",
    "TKE_TRANSPORT_LABEL",
    "advance_tke_v841",
    "calculate_n2_v841",
    "compute_les_tendencies_v841",
    "initialize_les_gradient_weights_v841",
    "les_config_from_dycore",
    "les_geometry_from_mesh",
    "les_label",
    "les_model_from_cli",
    "les_models_v841",
    "theta_dissipation_les_v841",
    "u_dissipation_les_v841",
    "validate_les_selection",
    "w_dissipation_les_v841",
]
