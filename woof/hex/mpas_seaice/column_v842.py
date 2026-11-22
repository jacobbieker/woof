"""MPAS-Seaice v8.4.2: the column package (colpkg) on the GPU, init and aggregate.

This is the first stage of the column package's port. It covers what
``seaice_init`` (``mpas_seaice_initialize.F``) does to the state when
``config_use_column_package`` is on, and the column package's aggregation:

- :func:`init_parameters`, ``seaice_init_column_physics_package_parameters``:
  the packages (``setup_packages_column_physics``), the configs checked
  (``check_column_package_configs``), the tracer object
  (``init_column_tracer_object``: the tracer count, indices, parents,
  first-ancestor mask and ancestors, which the column package's tracer
  arithmetic walks) and the column package's parameters
  (``init_column_package_configs``, the ones this stage reads). Host side.
- :func:`initialize_coupler_fields`, ``initialize_coupler_fields``: the wind
  speed, the sea freezing temperature from the sea-surface salinity, the
  sea-surface temperature, the upward longwave and the initial air drag
  coefficient, the aerosol flux.
- :func:`init_variables`, ``seaice_init_column_physics_package_variables``:
  the level-ice tracers, the thickness-category limits (``colpkg_init_itd``),
  the initial salinity and melting-temperature profiles
  (``colpkg_init_thermo``, ``colpkg_liquidus_temperature``) and the ridging
  history variable.
- :func:`init_ice_state`, ``init_ice_state``: the ``cice_default`` (with
  ``initial_category_areas_and_volumes`` and ``colpkg_init_trcr``),
  ``uniform``, ``no_ice`` and ``none`` initial conditions, the ``uniform`` and
  ``none`` initial velocities, and ``seaice_column_aggregate``.
- :func:`reset_coupler_fluxes`, ``seaice_reset_coupler_fluxes``.
- :func:`aggregate`, ``seaice_column_aggregate`` (``colpkg_aggregate`` with
  ``colpkg_compute_tracers``) and :func:`update_state`,
  ``seaice_column_update_state``.

:func:`initialize` runs them in ``seaice_init``'s order. The column package
is double precision (``dbl_kind``) and ``mpas_seaice_column.F`` passes it
RKIND arrays, so the wrappers only compile at RKIND 8: this port is float64
only.

Every word is held to the wrappers and the column package compiled verbatim
with gfortran ``-O0 -ffp-contract=off`` (``tools/mpas_seaice_oracle/run_column.F90``,
``tests/test_mpas_seaice_column_parity.py``). The library maths are the
oracle host's: ``tanh`` (``colpkg_init_itd``) is CORE-MATH's (glibc 2.44's
own, ``cr_tanh_flt64.cuh``), ``pow`` the shared glibc transcription
(``glibc_pow``) and ``cos`` CORE-MATH's ``cr_cos`` (``uw_cos``), which equals
glibc's ``cos`` on every argument ``colpkg_init_thermo`` forms for 1 to
:data:`MAX_ICE_LAYERS` ice layers except :data:`COS_REFUSED_LAYERS`, where
glibc's cos (LGPL IBM code, not transcribed) is 1 ULP off the correctly rounded
word on one profile level (graded on the oracle host by
``tests/test_mpas_seaice_column_parity.py`` against ``init-thermo-r8.npy``);
those counts and more layers are refused by name. ``log`` and the powers in the initial air
drag coefficient and upward longwave are constants gfortran folds at compile
time (correctly rounded, :func:`_folded_constants`).

The column biogeochemistry's framework and brine height
(``config_use_column_biogeochemistry`` with ``config_use_brine``): the tracer
object's ``index_brineFraction`` (``colpkg_init_zbgc``), the ocean
concentrations and deposition ``initialize_coupler_fields`` sets,
``init_column_biogeochemistry_profiles`` (:func:`hbrine_grids`, the brine
fraction) and the ocean fluxes the reset zeroes. The brine height's step is
:mod:`woof.hex.mpas_seaice.column_bgc_v842`; the other routines refuse the
biogeochemistry, and every other biogeochemistry flag is refused by name.

Refused by name: the other biogeochemistry flags; more than one snow layer
(``init_column_tracer_object_parent_indices`` sets only the first snow
layer's parent in v8.4.2, so the others' are undefined); a restart
(``config_do_restart``: the shortwave re-initialisation is a later stage);
the initial conditions and velocities this stage does not port. The column
package's own configuration errors (``check_column_package_configs``) are
raised as :class:`ValueError` with v8.4.2's message.

Arrays are device float64 (int32 masks) in the Fortran layout, MPAS's
missing-cell element included: ``a(nIceLayers, nCategories, nCells)`` is
C-order ``(nCells + 1, nCategories, nIceLayers)``. The pools are dicts named
as in MPAS (``tracers``, ``tracers_aggregate``, ``icestate``, ``initial``,
``ridging``, ``atmos_coupling``, ``atmos_forcing``, ``atmos_fluxes``,
``ocean_coupling``, ``ocean_fluxes``, ``shortwave``, ``drag``, ``aerosols``,
``velocity_solver``, ``boundary``, ``diagnostics``, ``mesh``); the fields of
packages that are off are ``None``, and dimensions are host integers in
``mesh``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from .velocity_v842 import COMPILE_OPTIONS, _grid

SOURCE_FILES = (
    "src/core_seaice/shared/mpas_seaice_column.F:200-300,359-420,758-800,3413-3600,4561-5435,5595-6079,"
    "7805-8222,10138-10232,10232-10420,10868-10990,11714-11990",
    "src/core_seaice/shared/mpas_seaice_initialize.F:231-850,1309-1377,1707-1944",
    "src/core_seaice/shared/mpas_seaice_forcing.F:1600-1840",
    "src/core_seaice/model_forward/mpas_seaice_core_interface.F:222-531",
    "src/core_seaice/column/ice_colpkg.F90:82-210,326-375,416-490,999-1011,1099-1121,1599-1718,3170-3290,"
    "5268-5442",
    "src/core_seaice/column/ice_colpkg_tracers.F90:180-245",
    "src/core_seaice/column/ice_mushy_physics.F90:1-120,199-330",
    "src/core_seaice/column/constants/cice/ice_constants_colpkg.F90",
)
MODULE_OPTIONS = COMPILE_OPTIONS
#: The shared libm transcriptions the unit prepends
MATH_HEADERS = ("glibc_flt64.cuh", "cr_tanh_flt64.cuh")
#: colpkg_init_thermo's cos arguments are graded against glibc's cos for this many ice layers
MAX_ICE_LAYERS = 64
#: the ice-layer counts whose profile meets one of glibc's not correctly rounded cos words
COS_REFUSED_LAYERS = frozenset({11, 31, 33, 55})
#: the biology layers the brine kernel's local arrays hold (the Registry fixes nBioLayers at 7)
MAX_BIO_LAYERS = 32
_BLOCK = 128
_DATA = files("woof.hex.mpas_seaice").joinpath("data")

#: the initial conditions and velocities init_ice_state runs here
INITIAL_CONDITIONS = ("cice_default", "uniform", "no_ice", "none")
INITIAL_VELOCITIES = ("uniform", "none")

# config_cice_int (mpas_seaice_column.F:10138-10218)
_CICE_INT = {
    "config_thermodynamics_type": {"zero layer": 0, "BL99": 1, "mushy": 2},
    "config_itd_conversion_type": {"delta function": 0, "linear remap": 1},
    "config_category_bounds_type": {"single category": -1, "original": 0, "new": 1, "WMO": 2, "asymptotic": 3},
    "config_ice_strength_formulation": {"Hibler79": 0, "Rothrock75": 1},
    "config_ridging_participation_function": {"Thorndike75": 0, "exponential": 1},
    "config_ridging_redistribution_function": {"Hibler80": 0, "exponential": 1},
}

# the physics tracers in the tracer array's order (set_cice_physics_tracer_array_category):
# (field, layers, the package switch that carries it)
_BGC_FLAGS = ("config_use_brine", "config_use_vertical_zsalinity", "config_use_vertical_biochemistry",
              "config_use_shortwave_bioabsorption", "config_use_vertical_tracers",
              "config_use_skeletal_biochemistry", "config_use_nitrate", "config_use_carbon",
              "config_use_chlorophyll", "config_use_ammonium", "config_use_silicate", "config_use_DMS",
              "config_use_nonreactive", "config_use_humics", "config_use_DON", "config_use_iron",
              "config_use_modal_aerosols", "config_use_zaerosols")


# ---------------------------------------------------------------------------
# the Registry: configs, dimensions, packages, fields
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def registry_options() -> tuple[dict, ...]:
    return tuple(json.loads(_DATA.joinpath("registry-options.json").read_text(encoding="utf-8"))["options"])


@lru_cache(maxsize=None)
def registry_fields() -> dict:
    return json.loads(_DATA.joinpath("registry-fields.json").read_text(encoding="utf-8"))


def _typed(option: dict):
    value, kind = option["default"], option["type"]
    if kind == "logical":
        return value.strip().lower() in ("true", ".true.", "t")
    if kind == "integer":
        return int(value)
    if kind == "real":
        return float(value.lower().replace("d", "e"))
    return value


def default_configs() -> dict:
    """Every v8.4.2 sea-ice namelist option at its Registry default."""
    return {o["name"]: _typed(o) for o in registry_options()}


def dimensions(configs: dict, mesh_dims: dict) -> dict[str, int]:
    """The Registry's dimensions: the mesh's (``nCells``, ``nEdges``,
    ``maxEdges``, ``maxEdges2``, ``nVertices``, ``vertexDegree``), the
    namelist's and the fixed and derived ones."""
    import ast
    out = dict(mesh_dims)
    pending = {d["name"]: d["definition"] for d in registry_fields()["dims"] if d["name"] not in out}
    while pending:
        progress = False
        for name, definition in list(pending.items()):
            if definition.startswith("namelist:"):
                out[name] = int(configs[definition.split(":", 1)[1]])
            elif definition == "":
                raise ValueError(f"the mesh dimension {name} is missing")
            else:
                tree = ast.parse(definition, mode="eval")
                names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
                if not names <= out.keys():
                    continue
                for node in ast.walk(tree):
                    if not isinstance(node, (ast.Expression, ast.BinOp, ast.Name, ast.Constant, ast.Load,
                                             ast.Add, ast.Sub, ast.Mult)):
                        raise ValueError(f"dimension {name}: unexpected definition {definition!r}")
                out[name] = int(eval(compile(tree, name, "eval"), {"__builtins__": {}}, dict(out)))
            del pending[name]
            progress = True
        if not progress:
            raise ValueError(f"unresolvable dimensions: {sorted(pending)}")
    return out


def packages(configs: dict) -> dict[str, bool]:
    """The packages' switches (``<name>Active``): ``setup_packages_dynamics``,
    ``setup_packages_column_physics`` and ``setup_packages_other``
    (``mpas_seaice_core_interface.F:132-587``)."""
    c = configs
    p = {f"{name}Active": False for name in registry_fields()["packages"]}
    if c.get("config_use_velocity_solver", True):
        scheme = c.get("config_stress_divergence_scheme", "variational")
        if scheme == "weak":
            p["pkgWeakActive"] = True
        elif scheme == "variational":
            p["pkgVariationalActive"] = True
            basis = c.get("config_variational_basis", "wachspress")
            p["pkgWachspressActive"] = basis == "wachspress"
            p["pkgPieceWiseLinearActive"] = basis == "pwl"
    column, bgc = bool(c["config_use_column_package"]), bool(c["config_use_column_biogeochemistry"])
    ponds = c["config_use_cesm_meltponds"] or c["config_use_level_meltponds"] or c["config_use_topo_meltponds"]
    p["pkgColumnPackageActive"] = column
    p["pkgColumnBiogeochemistryActive"] = bgc and column
    physics = {"pkgColumnTracerIceAgeActive": c["config_use_ice_age"],
               "pkgColumnTracerFirstYearIceActive": c["config_use_first_year_ice"],
               "pkgColumnTracerLevelIceActive": c["config_use_level_ice"],
               "pkgColumnTracerPondsActive": ponds,
               "pkgColumnTracerLidThicknessActive": c["config_use_level_meltponds"] or c["config_use_topo_meltponds"],
               "pkgColumnTracerAerosolsActive": c["config_use_aerosols"]}
    skeletal, vertical = c["config_use_skeletal_biochemistry"], c["config_use_vertical_tracers"]
    tracers = {"pkgTracerBrineActive": c["config_use_brine"], "pkgTracerMobileFractionActive": vertical,
               "pkgTracerSkeletalAlgaeActive": skeletal,
               "pkgTracerVerticalAlgaeActive": vertical and c["config_use_vertical_biochemistry"],
               "pkgTracerZAerosolsActive": c["config_use_zaerosols"],
               "pkgTracerZSalinityActive": c["config_use_vertical_zsalinity"]}
    for what, flag in (("Nitrate", "config_use_nitrate"), ("Carbon", "config_use_carbon"),
                       ("Ammonium", "config_use_ammonium"), ("Silicate", "config_use_silicate"),
                       ("DMS", "config_use_DMS"), ("Nonreactive", "config_use_nonreactive"),
                       ("Humics", "config_use_humics"), ("DON", "config_use_DON"), ("Iron", "config_use_iron")):
        tracers[f"pkgTracerSkeletal{what}Active"] = skeletal and c[flag]
        tracers[f"pkgTracerVertical{what}Active"] = vertical and c[flag]
    if not column:
        physics = dict.fromkeys(physics, False)
    if not column or not bgc:
        tracers = dict.fromkeys(tracers, False)
    p.update({k: bool(v) for k, v in physics.items()})
    p.update({k: bool(v) for k, v in tracers.items()})
    p["pkgColumnFormDragActive"] = bool(column and c["config_use_form_drag"])
    p["pkgForcingActive"] = bool(c.get("config_use_forcing", False))
    p["pkgTestingSystemTestActive"] = bool(c.get("config_testing_system_test", False))
    return p


def field_table(pool_names, dims: dict[str, int], active: dict[str, bool]) -> list[tuple]:
    """The Registry's real and integer fields of ``pool_names``: (pool, name,
    'real' | 'integer', C-order shape, active). A field is active when its
    var_struct's packages and any one of its own are; the time dimension is
    dropped and an nCells, nEdges or nVertices extent carries MPAS's
    missing element."""
    rows = []
    extra = {"nCells", "nEdges", "nVertices"}
    for struct in registry_fields()["var_structs"]:
        if struct["name"] not in pool_names:
            continue
        struct_on = all(active[f"{p}Active"] for p in struct["packages"])
        for name, kind, dimension_text, package_text in struct["vars"]:
            if kind not in ("real", "integer"):
                continue
            names = [d for d in dimension_text.split() if d != "Time"]
            shape = tuple(dims[d] + (1 if d in extra else 0) for d in reversed(names))
            own = [p for p in package_text.split(";") if p]
            on = struct_on and (not own or any(active[f"{p}Active"] for p in own))
            rows.append((struct["name"], name, kind, shape, on))
    return rows


# ---------------------------------------------------------------------------
# configs: check_column_package_configs and this stage's refusals
# ---------------------------------------------------------------------------

def _crit(*lines: str):
    raise ValueError("\n".join(lines))


def check_column_package_configs(configs: dict, n_categories: int) -> None:
    """``check_column_package_configs`` (``mpas_seaice_column.F:7841-8100``):
    its MPAS_LOG_CRIT aborts as :class:`ValueError` with v8.4.2's messages."""
    c = configs

    def choice(name, valid, text):
        if c[name] not in valid:
            _crit(f"config_error: {name} has invalid value", f"{name}: {c[name]}", f"valid options: {text}")

    choice("config_thermodynamics_type", ("zero layer", "BL99", "mushy"), "'zero layer', 'BL99' or 'mushy'")
    choice("config_heat_conductivity_type", ("MU71", "bubbly"), "'MU71' or 'bubbly'")
    choice("config_shortwave_type", ("ccsm3", "dEdd"), "'ccsm3' or 'dEdd'")
    choice("config_albedo_type", ("ccsm3", "constant"), "'ccsm3' or 'constant'")
    choice("config_ice_strength_formulation", ("Hibler79", "Rothrock75"), "'Hibler79' or 'Rothrock75'")
    choice("config_ridging_participation_function", ("Thorndike75", "exponential"),
           "'Thorndike75' or 'exponential'")
    choice("config_ridging_redistribution_function", ("Hibler80", "exponential"), "'Hibler80' or 'exponential'")
    choice("config_atmos_boundary_method", ("ccsm3", "constant"), "'ccsm3' or 'constant'")
    choice("config_itd_conversion_type", ("delta function", "linear remap"), "'delta function' or 'linear remap'")
    choice("config_category_bounds_type", ("single category", "original", "new", "WMO", "asymptotic"),
           "'single category', 'original', 'new', 'WMO' or 'asymptotic'")
    choice("config_pond_refreezing_type", ("cesm", "hlid"), "'cesm' or 'hlid'")
    cesm, level, topo = c["config_use_cesm_meltponds"], c["config_use_level_meltponds"], c["config_use_topo_meltponds"]
    p = "check_column_package_configs: "
    if int(cesm) + int(level) + int(topo) > 1:
        _crit(p + "More than one melt pond scheme active")
    if n_categories == 1 and c["config_itd_conversion_type"] == "linear remap":
        _crit(p + "Remapping the ITD is not allowed for nCategories=1",
              "Use config_itd_conversion_type = 'delta function' with config_category_bounds_type = 'original'",
              "or for column configurations use config_category_bounds_type = 'single category'")
    if n_categories != 1 and c["config_category_bounds_type"] == "single category":
        _crit(p + "nCategories /= 1 .and. config_category_bounds_type = 'single category'")
    if level and abs(c["config_snow_to_ice_transition_depth"]) > 1.0e-11:
        _crit(p + "config_use_level_meltponds = .true. and config_snow_to_ice_transition_depth /= 0")
    if cesm and c["config_pond_refreezing_type"] != "cesm":
        _crit(p + "config_use_cesm_meltponds = .true. and config_pond_refreezing_type /= 'cesm'")
    ponds = cesm or level or topo
    if c["config_shortwave_type"] != "dEdd" and ponds and c["config_calc_surface_temperature"]:
        _crit(p + "config_shortwave_type) /= 'dEdd' .and. use_meltponds = .true.",
              ".and. config_calc_surface_temperature ==.true.")
    for name in ("config_min_meltwater_retained_fraction", "config_max_meltwater_retained_fraction"):
        if c[name] < 0.0 or c[name] > 1.0:
            _crit(p + f"{name} out of bounds")
    thermo = c["config_thermodynamics_type"]
    if thermo == "mushy" and not c["config_calc_surface_temperature"]:
        _crit(p + "config_thermodynamics_type = 'mushy' and config_calc_surface_temperature = .false.")
    drag = c["config_use_form_drag"]
    if drag and c["config_atmos_boundary_method"] == "constant":
        _crit(p + "config_use_form_drag = .true. and config_atmos_boundary_method = 'constant'")
    if drag and not c["config_calc_surface_stresses"]:
        _crit(p + "config_use_form_drag = .true. and config_calc_surface_stresses = .false.")
    if drag and cesm:
        _crit(p + "config_use_form_drag = .true. and config_use_cesm_meltponds = .true.")
    if drag and not c["config_use_level_ice"]:
        _crit(p + "config_use_form_drag = .true. and config_use_level_ice = .false.")
    if not drag and c["config_ocean_heat_transfer_type"] == "Cdn_ocn":
        _crit(p + "config_use_form_drag = .false. and config_ocean_heat_transfer_type == 'Cdn_ocn'")
    if thermo == "BL99" and c["config_sea_freezing_temperature_type"] != "linear_salt":
        _crit(p + "config_thermodynamics_type == 'BL99' and config_sea_freezing_temperature_type /= 'linear_salt'")
    if thermo == "mushy" and c["config_sea_freezing_temperature_type"] != "mushy":
        _crit(p + "config_thermodynamics_type == 'mushy' and config_sea_freezing_temperature_type /= 'mushy'")
    if not c["config_use_column_biogeochemistry"] and any(c[f] for f in _BGC_FLAGS):
        _crit(p + "config_use_column_biogeochemistry = false. All biogeochemistry namelist flags must also be false")
    if c["config_use_vertical_zsalinity"] and (not c["config_use_brine"] or thermo != "BL99"):
        _crit(p + "vertical zSalinity requires config_use_brine = true and 'BL99' ")
    if (c["config_use_vertical_biochemistry"] or c["config_use_zaerosols"]) and \
            (not c["config_use_brine"] or not c["config_use_vertical_tracers"]):
        _crit(p + "vertical biochemistry and zaerosols require config_use_brine and config_use_vertical_tracer = true")
    if c["config_use_vertical_tracers"] and c["config_use_skeletal_biochemistry"]:
        _crit(p + "vertical bio tracers and skeletal bio tracers cannot both be true")
    if c["config_use_shortwave_bioabsorption"] and c["config_shortwave_type"] != "dEdd":
        _crit(p + "shortwave bioabsorption requires config_shortwave_type ==  'dEdd'")
    if (c["config_use_vertical_biochemistry"] or c["config_use_skeletal_biochemistry"]) and \
            not c["config_use_nitrate"]:
        _crit(p + "biochemistry needs at the very least config_use_nitrate = true")


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


#: Opt-in corrections of v8.4.2 bugs the column port otherwise keeps word for word.
#: ``configs["woof_v842_corrections"]`` names the ones to apply (default none: the
#: oracle's words). Each is documented where v8.4.2's behaviour is, and tested on its
#: own for physical sanity; none is held to the oracle.
CORRECTIONS = {
    "set_sfcflux_order": (
        "colpkg_step_therm1 passes set_sfcflux its coupled fluxes in the wrong order "
        "(ice_colpkg.F90:2095-2101 against ice_flux_colpkg.F90:169-177), so with "
        "config_calc_surface_temperature off the coupled conductive flux becomes the surface flux and the "
        "other way round; the correction passes them in set_sfcflux's order"),
    "ice_velocity_at_cell": (
        "column_vertical_thermodynamics hands colpkg the vertex field uVelocity at the cell's index "
        "(mpas_seaice_column.F:1511-1512), and colpkg_atm_boundary drops the vvel passed in (it sets "
        "worku = uvel twice, ice_colpkg.F90:3474-3479, and workv stays 0), so the ice velocity in the "
        "atmosphere's relative wind is another vertex's u and no v; the correction uses the cell's ice "
        "velocity, both components (seaice_interpolate_vertex_to_cell)"),
    "ccsm3_absorbed_solar_layers": (
        "shortwave_ccsm3 passes ncat as absorbed_solar's nilyr (ice_shortwave.F90:236), so the penetrating "
        "light is spread over ncat layers of thickness hi/ncat (and past the category's layers when "
        "ncat > nilyr); the correction passes nilyr"),
    "dedd_zero_volume_ice": (
        "compute_dEdd's chlorophyll absorption kalg*(0.5/dzk(klev)) (ice_shortwave.F90:2212) divides by a zero "
        "layer thickness when a category has area but no volume: the bottom layer's extinction is infinite, so "
        "with algae the visible light reaching it is absorbed in ice that is not there, and with kalg = 0 "
        "(0*Inf) the category's shortwave is NaN; the correction gives layers without thickness no "
        "chlorophyll absorption, so they are transparent"),
    "zbgc_c2n_proteins": (
        "colpkg_init_zbgc sets R_C2N(1) = ratio_C2N_proteins after R_C2N(1) = ratio_C2N_diatoms "
        "(ice_colpkg.F90:1108,1121) where it means R_C2N_DON(1), so the diatoms' carbon to nitrogen ratio is "
        "the proteins' and R_C2N_DON keeps its static zero; initialize_coupler_fields hands both to "
        "carbonToNitrogenRatioAlgae and carbonToNitrogenRatioDON (colpkg_init_ocean_conc); the correction "
        "gives the diatoms config_ratio_C_to_N_diatoms and the proteins config_ratio_C_to_N_proteins"),
    "bgc_inactive_diagnostics": (
        "seaice_column_reinitialize_diagnostics_bgc zeroes every biogeochemical diagnostic "
        "(mpas_seaice_column.F:12450-12471) without asking whether its package is on, and the Registry "
        "allocates no array for a field whose packages are off: with the brine height alone, "
        "totalChlorophyll (pkgTracerVerticalAlgae;pkgTracerSkeletalAlgae) is a null pointer whose "
        "descriptor gfortran gives one element, so the zeroing writes to address 0 and every time step "
        "with the column biogeochemistry dies there; the correction zeroes only the fields that have an "
        "array, which nothing else reads"),
    "zsal_solve": (
        "colpkg_init_parameters takes solve_zsal_in but never assigns the column package's solve_zsal "
        "(ice_colpkg.F90:4325-4345; mpas_seaice_column.F:9608 has it commented out), so it stays false, and "
        "set/get_cice_biogeochemistry_tracer_array_category copy verticalSalinity between the pool and the "
        "tracer array only with config_use_vertical_tracers (mpas_seaice_column.F:6292,6425-6431,6664,6802-6808 and the cell arrays' at 7075 and 7562): "
        "config_use_vertical_zsalinity never solves the vertical salinity (the brine height runs "
        "compute_microS_mushy, BL99's dTemp stays 0.01, colpkg_init_zsalinity's sss*salt_loss never reaches the "
        "pool); the correction sets solve_zsal and copies the vertical salinity, so the init writes it, "
        "column_biogeochemistry runs compute_microS and zsalinity, BL99 uses dTemp = 0.1, and the ITD and the "
        "ridging run their solve_zsal branches (add_new_ice's salt loss, add_new_ice_bgc's adjust_tracer_profile, "
        "lateral_melt_bgc, zap_small_areas' dfzsal). Held to the "
        "oracle built with tools/mpas_seaice_oracle/zsal_solve.patch"),
}


def corrections(configs: dict) -> frozenset[str]:
    """The opt-in corrections ``configs`` names (``woof_v842_corrections``),
    refusing names that are not in :data:`CORRECTIONS`."""
    names = configs.get("woof_v842_corrections", ())
    if isinstance(names, str):
        names = (names,)
    declaration = "names from woof.hex.mpas_seaice.column_v842.CORRECTIONS: " + ", ".join(sorted(CORRECTIONS))
    try:
        names = tuple(names)
        if not all(isinstance(name, str) for name in names):
            raise TypeError
    except TypeError:
        _refuse("woof_v842_corrections", names, "the corrections are a name or a sequence of names", declaration)
    unknown = sorted(set(names) - CORRECTIONS.keys())
    if unknown:
        _refuse("woof_v842_corrections", unknown, "these are not corrections of the column port", declaration)
    return frozenset(names)


#: the biogeochemistry flags the port carries (the brine height); the others are refused by name
BGC_PORTED = ("config_use_brine",)
#: the vertical salinity: every biogeochemistry stage carries v8.4.2's inert one and the solved
#: one (ZSAL_SOLVE)
ZSAL_FLAG = "config_use_vertical_zsalinity"
#: the opt-in correction that solves the vertical salinity (CORRECTIONS)
ZSAL_SOLVE = "zsal_solve"


#: the vertical biogeochemical tracers (z_tracers): the init and column_biogeochemistry
#: carry them with the nonreactive tracer (VT_PORTED) alone; every other stage refuses them
VT_FLAG = "config_use_vertical_tracers"
#: the vertical tracers' flags the port carries
VT_PORTED = ("config_use_nonreactive",)


def vertical_tracers(configs: dict) -> bool:
    """Whether the column package's z_tracers are on (the column
    biogeochemistry with ``config_use_vertical_tracers``)."""
    return bool(configs["config_use_column_biogeochemistry"] and configs[VT_FLAG])


def solves_zsal(configs: dict) -> bool:
    """Whether the column package solves the vertical salinity: never in
    v8.4.2 (its solve_zsal is never set), only under the opt-in correction
    ``zsal_solve`` (:data:`CORRECTIONS`) with the biogeochemistry, the brine
    height and ``config_use_vertical_zsalinity`` on."""
    return bool(configs["config_use_column_biogeochemistry"] and configs["config_use_brine"]
                and configs[ZSAL_FLAG] and "zsal_solve" in corrections(configs))


def refuse_unported(configs: dict, dims: dict[str, int], biogeochemistry: bool = False,
                    z_tracers: bool = False) -> None:
    """Refuse, by name, what this stage of the column port does not carry.

    ``biogeochemistry`` is the caller's: the init (:func:`initialize`),
    :func:`woof.hex.mpas_seaice.column_bgc_v842.biogeochemistry`, the
    vertical and ITD thermodynamics, the ridging, the shortwave and the
    column step carry the column biogeochemistry's framework and its brine
    height (``config_use_brine``); every other routine refuses
    ``config_use_column_biogeochemistry``. Those stages all carry the
    vertical salinity (``config_use_vertical_zsalinity``): v8.4.2's inert
    one (its tracers counted, never solved or copied) and, under the
    ``zsal_solve`` correction, the solved one.

    ``z_tracers`` is the caller's too: the init, ``column_biogeochemistry``,
    the ITD thermodynamics and the ridging carry the vertical tracers
    (``config_use_vertical_tracers``) with the nonreactive tracer alone
    (:data:`VT_PORTED`); every other stage refuses them."""
    c = configs
    if not c.get("config_use_column_package", False):
        _refuse("config_use_column_package", False, "the column port runs only with the column package on",
                "config_use_column_package = .true.")
    if c["config_use_column_biogeochemistry"]:
        if not biogeochemistry:
            _refuse("config_use_column_biogeochemistry", True,
                    "the column biogeochemistry is ported for the init, the column stages and the column "
                    "step only",
                    "config_use_column_biogeochemistry = .false.")
        if c[VT_FLAG] and not z_tracers:
            _refuse(VT_FLAG, True, "the vertical tracers are ported for the init, column_biogeochemistry, "
                    "the ITD thermodynamics and the ridging only", f"{VT_FLAG} = .false.")
        carried = BGC_PORTED + (ZSAL_FLAG,) + ((VT_FLAG,) + VT_PORTED if c[VT_FLAG] else ())
        for flag in _BGC_FLAGS:
            if flag not in carried and c[flag]:
                _refuse(flag, True, "of the column biogeochemistry only the brine height, the vertical "
                        "salinity and the vertical tracers' nonreactive tracer are ported", f"{flag} = .false.")
        if c[VT_FLAG]:
            _refuse_z_tracers(c)
        _refuse_bio_layers(c, dims)
    if dims["nSnowLayers"] != 1:
        _refuse("config_nSnowLayers", dims["nSnowLayers"],
                "v8.4.2's init_column_tracer_object_parent_indices sets only the first snow layer's parent "
                "(mpas_seaice_column.F:5100-5102), so the other snow layers' parents are undefined",
                "config_nSnowLayers = 1")
    if not 1 <= dims["nIceLayers"] <= MAX_ICE_LAYERS:
        _refuse("config_nIceLayers", dims["nIceLayers"],
                f"colpkg_init_thermo's cos is graded against glibc's for 1 to {MAX_ICE_LAYERS} ice layers",
                f"1 <= config_nIceLayers <= {MAX_ICE_LAYERS}")
    if dims["nIceLayers"] in COS_REFUSED_LAYERS:
        _refuse("config_nIceLayers", dims["nIceLayers"],
                "colpkg_init_thermo's salinity profile at this many ice layers meets a glibc cos "
                "word that is not correctly rounded, and glibc's cos is not transcribed",
                f"config_nIceLayers not in {sorted(COS_REFUSED_LAYERS)}")
    if c["config_use_level_meltponds"] and not c["config_use_level_ice"]:
        _refuse("config_use_level_ice", False,
                "level melt ponds without the level-ice tracer: v8.4.2 leaves index_levelIceArea at 0 and makes "
                "it the ponds' ancestor (init_column_tracer_object, mpas_seaice_column.F:5106-5110), so the "
                "aggregate reads the tracer array out of bounds",
                "config_use_level_ice = .true. with config_use_level_meltponds")
    if c["config_do_restart"]:
        _refuse("config_do_restart", True, "a restart re-initialises the column shortwave, which is not ported",
                "config_do_restart = .false.")
    if c["config_initial_condition_type"] not in INITIAL_CONDITIONS:
        _refuse("config_initial_condition_type", c["config_initial_condition_type"],
                "this column stage ports the initial conditions " + ", ".join(INITIAL_CONDITIONS),
                "config_initial_condition_type in " + ", ".join(INITIAL_CONDITIONS))
    if c["config_initial_velocity_type"] not in INITIAL_VELOCITIES:
        _refuse("config_initial_velocity_type", c["config_initial_velocity_type"],
                "this column stage ports the initial velocities " + ", ".join(INITIAL_VELOCITIES),
                "config_initial_velocity_type in " + ", ".join(INITIAL_VELOCITIES))


def _refuse_z_tracers(configs: dict) -> None:
    """What the vertical tracers' port does not carry, by name."""
    c = configs
    if not c["config_use_brine"]:
        _refuse("config_use_brine", False, "the vertical tracers are ported with the brine height only "
                "(colpkg_init_zbgc makes them depend on the brine fraction)",
                f"config_use_brine = .true. with {VT_FLAG}")
    if not all(c[flag] for flag in VT_PORTED):
        _refuse("config_use_nonreactive", False, "the vertical tracers are ported with the nonreactive tracer",
                f"config_use_nonreactive = .true. with {VT_FLAG}")
    if c[ZSAL_FLAG]:
        # set/get_cice_biogeochemistry_tracer_array_category copy verticalSalinity at
        # index_verticalSalinity with the vertical tracers (mpas_seaice_column.F:6292,6664),
        # which MPAS counts after the brine fraction while colpkg_init_zbgc never places
        # nt_bgc_S there unless solve_zsal (never set in v8.4.2): the copies land on the
        # vertical tracers' indices
        _refuse(ZSAL_FLAG, True, "with the vertical tracers, MPAS copies verticalSalinity to and from tracer "
                "indices the column package gives the vertical tracers", f"{ZSAL_FLAG} = .false. with {VT_FLAG}")
    if c["config_scale_initial_vertical_bgc"] and c["config_thermodynamics_type"] == "mushy":
        _refuse("config_scale_initial_vertical_bgc", True, "colpkg_init_bgc's salinity-scaled profiles (the "
                "mushy remap) are not ported", "config_scale_initial_vertical_bgc = .false.")


def _refuse_bio_layers(configs: dict, dims: dict[str, int]) -> None:
    """compute_microS_mushy remaps the ice salinity and enthalpy onto the
    nBioLayers biology layers into ``trtmp(ntrcr+2)`` at the tracer's own
    index (``ice_brine.F90:279-297``): past ``ntrcr + 2`` v8.4.2 writes out of
    bounds. ``ntrcr`` is the column package's tracer count for this
    configuration (``ice_brine.F90:234``), the brine fraction included: the
    bound is v8.4.2's own, not a conservative one."""
    n_bio = dims.get("nBioLayers", 7)
    to = tracer_object(configs, dims)
    last = to.index["iceSalinity"] + n_bio - 1
    if not 1 <= n_bio <= MAX_BIO_LAYERS or last > to.n_tracers + 2:
        _refuse("nBioLayers", n_bio,
                "compute_microS_mushy's remap writes the biology layers past its trtmp(ntrcr+2) "
                f"(the ice salinity's index {to.index['iceSalinity']} plus nBioLayers - 1 > {to.n_tracers} + 2), "
                f"or more than {MAX_BIO_LAYERS} layers", "more ice layers or physics tracers")


# ---------------------------------------------------------------------------
# the tracer object (init_column_tracer_object, mpas_seaice_column.F:4561-5287)
# ---------------------------------------------------------------------------

@dataclass
class TracerObject:
    """``ciceTracerObject``'s physics part: 1-based indices as in MPAS (0 is
    missing). ``first_ancestor_mask`` is ``(nBaseTracers, nTracers)`` and
    ``ancestor_indices`` ``(nMaxAncestorTracers, nTracers)``, the Fortran
    arrays' C-order. ``layout`` lists the physics tracer fields in the tracer
    array's order: (tracers-pool field, first index, layers)."""
    n_tracers: int
    index: dict[str, int]
    parent_index: np.ndarray
    first_ancestor_mask: np.ndarray
    ancestor_number: np.ndarray
    ancestor_indices: np.ndarray
    nt_tsfc: int
    nt_fbri: int
    layout: list[tuple[str, int, int]] = field(default_factory=list)
    #: nTracersNotBio: the physics tracers' count (the biogeochemistry's tracers follow)
    n_tracers_not_bio: int = 0
    #: nt_bgc_S: index_verticalSalinity, the vertical salinity's first tracer (0 without it)
    nt_bgc_s: int = 0
    #: the vertical tracers (z_tracers): nbtrcr, nt_zbgc_frac (index_mobileFraction), and per
    #: bio tracer its first tracer (bio_index, nblyr + 3 layers: the ice's nblyr + 1, then the
    #: snow's two) and its place in the ocean concentrations (bio_index_o, 1-based)
    n_bio_tracers: int = 0
    nt_zbgc_frac: int = 0
    bio_index: tuple = ()
    bio_index_o: tuple = ()

    def log_rows(self) -> dict[str, list[int]]:
        """The oracle log's ``TOBJ`` rows."""
        i = self.index
        return {"nTracers": [self.n_tracers],
                "index": [i[k] for k in ("surfaceTemperature", "iceEnthalpy", "snowEnthalpy", "iceSalinity",
                                         "iceAge", "firstYearIceArea", "levelIceArea", "levelIceVolume",
                                         "pondArea", "pondDepth", "pondLidThickness", "aerosols")],
                "parentIndex": self.parent_index.tolist(),
                "firstAncestorMask": [int(x) for row in self.first_ancestor_mask for x in row],
                "ancestorNumber": self.ancestor_number.tolist(),
                "ancestorIndices": [int(x) for row in self.ancestor_indices for x in row],
                "colpkg": [self.nt_tsfc, self.nt_fbri],
                **({"colpkg_zsal": [self.nt_bgc_s]} if self.nt_bgc_s else {}),
                **({"colpkg_zbgc": [self.n_bio_tracers, self.nt_zbgc_frac, *self.bio_index, *self.bio_index_o]}
                   if self.n_bio_tracers else {})}


def tracer_object(configs: dict, dims: dict[str, int]) -> TracerObject:
    """``init_column_tracer_object``: the tracer number, child (index),
    parent, first-ancestor and ancestor routines, with the biogeochemistry's
    brine fraction (``nt_fbri``) when the column biogeochemistry and
    ``config_use_brine`` are on (the only biogeochemistry tracer ported)."""
    c = configs
    n_ice, n_snow, n_aero = dims["nIceLayers"], dims["nSnowLayers"], dims["nAerosols"]
    cesm, level, topo = c["config_use_cesm_meltponds"], c["config_use_level_meltponds"], c["config_use_topo_meltponds"]
    idx = {k: 0 for k in ("iceAge", "firstYearIceArea", "levelIceArea", "levelIceVolume", "pondArea", "pondDepth",
                          "pondLidThickness", "aerosols")}
    layout = []
    n = 1
    idx["surfaceTemperature"] = 1
    layout.append(("surfaceTemperature", 1, 1))
    idx["iceEnthalpy"] = n + 1
    layout.append(("iceEnthalpy", n + 1, n_ice))
    n += n_ice
    idx["snowEnthalpy"] = n + 1
    layout.append(("snowEnthalpy", n + 1, n_snow))
    n += n_snow
    idx["iceSalinity"] = n + 1
    layout.append(("iceSalinity", n + 1, n_ice))
    n += n_ice
    for flag, names in (("config_use_ice_age", ("iceAge",)), ("config_use_first_year_ice", ("firstYearIceArea",)),
                        ("config_use_level_ice", ("levelIceArea", "levelIceVolume"))):
        if c[flag]:
            for name in names:
                n += 1
                idx[name] = n
                layout.append((name, n, 1))
    if cesm or level or topo:
        for name in ("pondArea", "pondDepth"):
            n += 1
            idx[name] = n
            layout.append((name, n, 1))
    if level or topo:
        n += 1
        idx["pondLidThickness"] = n
        layout.append(("pondLidThickness", n, 1))
    if c["config_use_aerosols"]:
        idx["aerosols"] = n + 1
        for a in range(n_aero):
            for k, name in enumerate(("snowScatteringAerosol", "snowBodyAerosol", "iceScatteringAerosol",
                                      "iceBodyAerosol")):
                layout.append((f"{name}:{a}", n + 1 + 4 * a + k, 1))
        n += 4 * n_aero
    n_tracers = n
    parent = np.zeros(n_tracers, np.int32)
    parent[idx["surfaceTemperature"] - 1] = 0
    for k in range(n_ice):
        parent[idx["iceEnthalpy"] - 1 + k] = 1
        parent[idx["iceSalinity"] - 1 + k] = 1
    # only the first snow layer's parent is set (nSnowLayers > 1 is refused)
    parent[idx["snowEnthalpy"] - 1] = 2
    if c["config_use_ice_age"]:
        parent[idx["iceAge"] - 1] = 1
    if c["config_use_first_year_ice"]:
        parent[idx["firstYearIceArea"] - 1] = 0
    if c["config_use_level_ice"]:
        parent[idx["levelIceArea"] - 1] = 0
        parent[idx["levelIceVolume"] - 1] = 1
    if cesm:
        parent[idx["pondArea"] - 1] = 0
        parent[idx["pondDepth"] - 1] = 2 + idx["pondArea"]
    if level:
        parent[idx["pondArea"] - 1] = 2 + idx["levelIceArea"]
        parent[idx["pondDepth"] - 1] = 2 + idx["pondArea"]
        parent[idx["pondLidThickness"] - 1] = 2 + idx["pondArea"]
    if topo:
        parent[idx["pondArea"] - 1] = 0
        parent[idx["pondDepth"] - 1] = 2 + idx["pondArea"]
        parent[idx["pondLidThickness"] - 1] = 2 + idx["pondArea"]
    if c["config_use_aerosols"]:
        for a in range(n_aero):
            base = idx["aerosols"] - 1 + 4 * a
            parent[base:base + 4] = (2, 2, 1, 1)
    mask = np.zeros((3, n_tracers), np.float64)
    for it in range(n_tracers):
        mask[{0: 0, 1: 1, 2: 2}.get(int(parent[it]), 0), it] = 1.0
    number = np.zeros(n_tracers, np.int32)
    ancestors = np.zeros((2, n_tracers), np.int32)
    if cesm:
        number[idx["pondDepth"] - 1] = 1
        ancestors[0, idx["pondDepth"] - 1] = idx["pondArea"]
    if level:
        number[idx["pondArea"] - 1] = 1
        ancestors[0, idx["pondArea"] - 1] = idx["levelIceArea"]
        for name in ("pondDepth", "pondLidThickness"):
            number[idx[name] - 1] = 2
            ancestors[1, idx[name] - 1] = idx["pondArea"]
            ancestors[0, idx[name] - 1] = idx["levelIceArea"]
    if topo:
        for name in ("pondDepth", "pondLidThickness"):
            number[idx[name] - 1] = 1
            ancestors[0, idx[name] - 1] = idx["pondArea"]
    # colpkg_init_tracer_indices: nt_Tsfc is the surface temperature's index;
    # nt_fbri is index_brineFraction, which only the biogeochemistry sets: the
    # module variable's zero (ciceTracerObject is static), which no tracer matches
    nt_fbri = 0
    if c["config_use_column_biogeochemistry"] and c["config_use_brine"]:
        # init_column_tracer_object_tracer_number counts the brine height after the
        # physics tracers (nTracersNotBio); colpkg_init_zbgc (ice_colpkg.F90:999-1011)
        # makes it index_brineFraction = ntrcr + 1, volume-weighted with no strata:
        # trcr_depend 1, trcr_base (0, 1, 0), n_trcr_strata 0, nt_strata 0. The
        # first-ancestor mask MPAS filled for it before from a parentIndex not yet
        # set is overwritten whole by trcr_base.
        nt_fbri = n_tracers + 1
        parent = np.append(parent, np.int32(1))
        column = np.zeros((3, 1), np.float64)
        column[1, 0] = 1.0
        mask = np.concatenate([mask, column], axis=1)
        number = np.append(number, np.int32(0))
        ancestors = np.concatenate([ancestors, np.zeros((2, 1), np.int32)], axis=1)
        layout.append(("brineFraction", nt_fbri, 1))
    nt_bgc_s = 0
    if c["config_use_column_biogeochemistry"] and c["config_use_brine"] and c[ZSAL_FLAG]:
        # init_column_tracer_object_tracer_number counts nBioLayers vertical-salinity
        # tracers after the brine fraction; colpkg_init_zbgc (ice_colpkg.F90:1016-1028)
        # makes nt_bgc_S = ntrcr + 1 and every layer's trcr_depend 2 + nt_fbri (the
        # brine volume), but sets trcr_base, n_trcr_strata and nt_strata for nt_bgc_S
        # alone (its k loop writes index nt_bgc_S each time): the first layer is
        # brine-volume weighted, (0, 1, 0), with the brine fraction its one ancestor.
        # The other layers keep what MPAS filled before colpkg_init_zbgc: no ancestors
        # (ancestorNumber and ancestorIndices are zeroed) and the first-ancestor mask
        # init_column_tracer_object_first_ancestor_mask made from a parentIndex not yet
        # set (allocated, never initialised): the ice area's column unless that word
        # happens to be 1 or 2.
        n_bio = dims["nBioLayers"]
        nt_bgc_s = n_tracers + (nt_fbri != 0) + 1
        parent = np.append(parent, np.full(n_bio, 2 + nt_fbri, np.int32))
        column = np.zeros((3, n_bio), np.float64)
        column[0, 1:] = 1.0
        column[1, 0] = 1.0
        mask = np.concatenate([mask, column], axis=1)
        number = np.append(number, np.zeros(n_bio, np.int32))
        number[nt_bgc_s - 1] = 1
        more = np.zeros((2, n_bio), np.int32)
        more[0, 0] = nt_fbri
        ancestors = np.concatenate([ancestors, more], axis=1)
        if solves_zsal(c):
            # v8.4.2 copies verticalSalinity(Cell) to and from the tracer arrays only with the
            # vertical tracers (mpas_seaice_column.F:6292,6664,7075,7562): its tracers are
            # counted and never read or written back, so only zsal_solve maps the pool field
            layout.append(("verticalSalinity", nt_bgc_s, n_bio))
    bio_index, bio_index_o, nt_zbgc_frac = [], [], 0
    if vertical_tracers(c) and c["config_use_brine"] and c["config_use_nonreactive"]:
        # colpkg_init_zbgc's z_tracers (ice_colpkg.F90:1362-1371 and 1426-1490), the
        # nonreactive tracer (tr_bgc_PON) alone: colpkg_init_bgc_trcr gives its nblyr + 1
        # ice layers trcr_depend nt_depend = 2 + nt_fbri (ntd = 0 with the brine
        # fraction), trcr_base (0, 1, 0) and the brine fraction as their one stratum,
        # then its two snow layers trcr_depend 2, trcr_base (0, 0, 1) and no strata;
        # bio_index_o(nlt_bgc_PON) = 2*max_algae + max_doc + max_dic + 7. The mobile
        # fractions follow at nt_zbgc_frac: trcr_depend 2 + nt_fbri, (0, 1, 0), the
        # brine fraction their stratum. Every word MPAS filled before is overwritten.
        nk = dims["nBioLayers"] + 1
        first = len(parent) + 1
        bio_index.append(first)
        bio_index_o.append(2 * _MAX_ALGAE + _MAX_DOC + _MAX_DIC + 7)
        n_bio = len(bio_index)
        nt_zbgc_frac = first + nk + 2
        ice, snow = np.full(nk, 2 + nt_fbri, np.int32), np.full(2, 2, np.int32)
        frac = np.full(n_bio, 2 + nt_fbri, np.int32)
        parent = np.concatenate([parent, ice, snow, frac])
        column = np.zeros((3, nk + 2 + n_bio), np.float64)
        column[1, :nk] = 1.0
        column[2, nk:nk + 2] = 1.0
        column[1, nk + 2:] = 1.0
        mask = np.concatenate([mask, column], axis=1)
        more_number = np.zeros(nk + 2 + n_bio, np.int32)
        more_number[:nk] = 1
        more_number[nk + 2:] = 1
        number = np.append(number, more_number)
        more = np.zeros((2, nk + 2 + n_bio), np.int32)
        more[0, :nk] = nt_fbri
        more[0, nk + 2:] = nt_fbri
        ancestors = np.concatenate([ancestors, more], axis=1)
        layout.append(("verticalNonreactiveConc", first, nk + 2))
        layout.append(("mobileFraction", nt_zbgc_frac, n_bio))
    return TracerObject(n_tracers=len(parent), index=idx, parent_index=parent,
                        first_ancestor_mask=mask, ancestor_number=number, ancestor_indices=ancestors,
                        nt_tsfc=idx["surfaceTemperature"], nt_fbri=nt_fbri, layout=layout,
                        n_tracers_not_bio=n_tracers, nt_bgc_s=nt_bgc_s, n_bio_tracers=len(bio_index),
                        nt_zbgc_frac=nt_zbgc_frac, bio_index=tuple(bio_index), bio_index_o=tuple(bio_index_o))


# ---------------------------------------------------------------------------
# the column package's parameters this stage reads
# ---------------------------------------------------------------------------

@dataclass
class ColumnState:
    """What ``seaice_init_column_physics_package_parameters`` leaves on the
    host: the dimensions, packages, tracer object and the column package's
    parameters (``ktherm``, ``kitd``, ``kcatbound``, ``calc_Tsfc``, and
    ``heat_capacity`` and ``hi_min`` once :func:`init_variables` has run
    ``colpkg_init_thermo`` and ``colpkg_init_itd``)."""
    dims: dict[str, int]
    packages: dict[str, bool]
    tracers: TracerObject
    ktherm: int
    kitd: int
    kcatbound: int
    calc_tsfc: bool
    heat_capacity: bool | None = None
    hi_min: float | None = None
    #: column_therm_v842.vertical_thermodynamics' (ierr, cell, message)
    abort: tuple | None = None
    #: the warnings the last column routine logged (column_itd_v842: zap_snow_temperature's
    #: count; column_ridge_v842: the graded lines, in log order)
    warnings: int | list = 0


def init_parameters(pools: dict, configs: dict, biogeochemistry: bool = False,
                    z_tracers: bool = False) -> ColumnState:
    """``seaice_init_column_physics_package_parameters``: host side. The
    stand-in pointers of the packages that are off change no state the port
    reads (the absent fields stay ``None``). ``biogeochemistry`` and
    ``z_tracers``: the caller carries the column biogeochemistry and its
    vertical tracers (:func:`refuse_unported`)."""
    mesh = pools["mesh"]
    dims = {k: int(mesh[k]) for k in ("nCells", "nCellsSolve", "nVertices", "nEdges", "maxEdges",
                                       "vertexDegree") if k in mesh}
    for k, v in mesh.items():
        if isinstance(v, (int, np.integer)):
            dims.setdefault(k, int(v))
    refuse_unported(configs, dims, biogeochemistry, z_tracers)
    check_column_package_configs(configs, dims["nCategories"])
    return ColumnState(
        dims=dims, packages=packages(configs), tracers=tracer_object(configs, dims),
        ktherm=_CICE_INT["config_thermodynamics_type"][configs["config_thermodynamics_type"]],
        kitd=_CICE_INT["config_itd_conversion_type"][configs["config_itd_conversion_type"]],
        kcatbound=_CICE_INT["config_category_bounds_type"][configs["config_category_bounds_type"]],
        calc_tsfc=bool(configs["config_calc_surface_temperature"]))


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

def module_source() -> str:
    import woof.core.kernels as kernels
    root = Path(kernels.__file__).parent
    math = "\n".join((root / name).read_text(encoding="utf-8") for name in MATH_HEADERS)
    body = files("woof.hex.mpas_seaice").joinpath("column_v842.cu").read_text(encoding="utf-8")
    return math + "\n" + body


@lru_cache(maxsize=None)
def _module_on(device: int, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(), options, None, "seaice_column_v842.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(name: str):
    import cupy as cp
    options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
    return _module_on(cp.cuda.Device().id, options).get_function(name)


def _launch(name: str, n: int, *args) -> None:
    _function(name)(_grid(n), (_BLOCK,), args)


def _f64(x) -> np.float64:
    return np.float64(x)


def _need(pool: dict, name: str, shape: tuple) -> object:
    import cupy as cp
    a = pool.get(name)
    if (not isinstance(a, cp.ndarray) or a.dtype != np.float64 or tuple(a.shape) != tuple(shape)
            or not a.flags.c_contiguous):
        raise ValueError(f"{name} must be a contiguous device float64 array of shape {tuple(shape)}")
    return a


def _need_int(pool: dict, name: str, shape: tuple) -> object:
    import cupy as cp
    a = pool.get(name)
    if (not isinstance(a, cp.ndarray) or a.dtype != np.int32 or tuple(a.shape) != tuple(shape)
            or not a.flags.c_contiguous):
        raise ValueError(f"{name} must be a contiguous device int32 array of shape {tuple(shape)}")
    return a


def _fill(array, value: float) -> None:
    """A whole-array assignment of a constant (``a = value``)."""
    _launch("seaice_column_fill_v842", array.size, array, _f64(value), np.int32(array.size))


@lru_cache(maxsize=None)
def _folded_constants() -> dict[str, float]:
    """The constants gfortran folds at compile time (MPFR, each operation
    rounded to nearest at 53 bits): ``seaice_column_initial_air_drag_coefficient``
    (``(k/log(z/r)) * (k/log(z/r))``) and initialize_coupler_fields' upward
    longwave (``-sigma * Tf**4``)."""
    from decimal import Context, Decimal
    from fractions import Fraction as F

    ratio = float(F(10.0) / F(0.0005))                      # zref / iceruf
    log_ratio = float(Context(prec=60).ln(Decimal(ratio)))           # log, correctly rounded
    k_over = float(F(0.4) / F(log_ratio))                   # vonkar / log(...)
    drag = float(F(k_over) * F(k_over))
    longwave = -float(F(567.0e-10) * F(float(F(273.15) ** 4)))   # -stefan_boltzmann * Tffresh**4
    return {"airDragCoefficient": drag, "longwaveUp": longwave}


# ---------------------------------------------------------------------------
# the init (seaice_init's column steps) and the aggregate
# ---------------------------------------------------------------------------

def initialize_coupler_fields(pools: dict, configs: dict, state: ColumnState) -> None:
    """``initialize_coupler_fields`` (``mpas_seaice_initialize.F:1707-1944``),
    the biogeochemistry's part too (:func:`_init_coupler_bgc`): the
    whole-array assignments cover MPAS's missing-cell element, the freezing
    temperature loop nCells."""
    n = state.dims["nCells"]
    atmos, ocean = pools["atmos_coupling"], pools["ocean_coupling"]
    c1 = (n + 1,)
    _launch("seaice_wind_speed_v842", n + 1, _need(pools["atmos_forcing"], "windSpeed", c1),
            _need(atmos, "uAirVelocity", c1), _need(atmos, "vAirVelocity", c1), np.int32(n + 1))
    _launch("seaice_sea_freezing_temperature_v842", n, _need(ocean, "seaFreezingTemperature", c1),
            _need(ocean, "seaSurfaceSalinity", c1), np.int32(n), np.int32(state.ktherm))
    if not configs["config_do_restart"]:
        _launch("seaice_column_copy_v842", n + 1, _need(ocean, "seaSurfaceTemperature", c1),
                ocean["seaFreezingTemperature"], np.int32(n + 1))
    folded = _folded_constants()
    _fill(_need(pools["atmos_fluxes"], "longwaveUp", c1), folded["longwaveUp"])
    _fill(_need(pools["drag"], "airDragCoefficient", c1), folded["airDragCoefficient"])
    if configs["config_use_aerosols"]:
        flux = pools["aerosols"]["atmosAerosolFlux"]
        if flux is not None and flux.size:
            _fill(flux, 1.0e-12)
    if configs["config_use_column_biogeochemistry"]:
        _init_coupler_bgc(pools, configs, n)


def _init_coupler_bgc(pools: dict, configs: dict, n: int) -> None:
    """``initialize_coupler_fields``' biogeochemistry (``mpas_seaice_initialize.F:1846-1940``):
    ``atmosBioFluxes`` zeroed whole, the black carbon and dust deposition of
    the uncoupled build (``1e-12``, ``1e-13``) and ``colpkg_init_ocean_conc``'s
    ocean concentrations on cells 1 to nCells, and its carbon to nitrogen
    ratios (``R_C2N``, ``R_C2N_DON``: see :data:`CORRECTIONS` ``zbgc_c2n_proteins``)."""
    bgc = pools["biogeochemistry"]
    _fill(bgc["atmosBioFluxes"], 0.0)
    for name, value in (("atmosBlackCarbonFlux", 1.0e-12), ("atmosDustFlux", 1.0e-13),
                        ("oceanAmmoniumConc", 1.0), ("oceanDMSPConc", 0.1), ("oceanDMSConc", 0.1),
                        ("oceanDICConc", 1.0), ("oceanDONConc", 12.9), ("oceanDissolvedIronConc", 0.4),
                        ("oceanParticulateIronConc", 2.0), ("oceanHumicsConc", 1.0), ("oceanNitrateConc", 12.0),
                        ("oceanSilicateConc", 25.0), ("oceanZAerosolConc", 0.0)):
        bgc[name][:n] = value
    for name, values in (("oceanAlgaeConc", (1.0, 0.0057, 0.0027)), ("oceanDOCConc", (16.2, 9.0, 1.0))):
        for k, value in enumerate(values):
            bgc[name][:n, k] = value
    c = configs
    if "zbgc_c2n_proteins" in corrections(c):
        algae = (c["config_ratio_C_to_N_diatoms"], c["config_ratio_C_to_N_small_plankton"],
                 c["config_ratio_C_to_N_phaeocystis"])
        don = c["config_ratio_C_to_N_proteins"]
    else:
        algae = (c["config_ratio_C_to_N_proteins"], c["config_ratio_C_to_N_small_plankton"],
                 c["config_ratio_C_to_N_phaeocystis"])
        don = 0.0
    if n:
        for k, value in enumerate(algae):
            bgc["carbonToNitrogenRatioAlgae"][k] = float(value)
        bgc["carbonToNitrogenRatioDON"][0] = float(don)


def _init_thermo(n_ice_layers: int, ktherm: int):
    """``colpkg_init_thermo``'s profile on the device: ``l_brine`` is
    ``saltmax > min_salin .and. heat_capacity``, so BL99 or mushy."""
    import cupy as cp
    profile = cp.zeros(n_ice_layers + 1, np.float64)
    _function("colpkg_init_thermo_v842")((1,), (1,), (profile, np.int32(n_ice_layers), np.int32(ktherm != 0)))
    return profile


def init_thermo_profile(n_ice_layers: int, ktherm: int = 1) -> np.ndarray:
    """``colpkg_init_thermo``'s salinity profile, ``sprofile(1:nilyr+1)``, on the host."""
    import cupy as cp
    return cp.asnumpy(_init_thermo(n_ice_layers, ktherm))


def init_column_itd(pools: dict, state: ColumnState) -> None:
    """``init_column_itd``: ``colpkg_init_itd`` writes ``categoryThicknessLimits``
    and sets the column package's ``hi_min``."""
    import cupy as cp
    ncat = state.dims["nCategories"]
    limits = _need(pools["initial"], "categoryThicknessLimits", (ncat + 1,))
    hi_min = cp.zeros(1, np.float64)
    status = cp.zeros(1, np.int32)
    _function("colpkg_init_itd_v842")((1,), (1,), (limits, hi_min, status, np.int32(ncat),
                                                    np.int32(state.kcatbound), np.int32(state.kitd)))
    if int(status.get()[0]):
        raise ValueError("init_column_itd: kcatbound=2 (WMO) must have ncat=5, 6 or 7")
    state.hi_min = float(hi_min.get()[0])


def init_variables(pools: dict, configs: dict, state: ColumnState) -> None:
    """``seaice_init_column_physics_package_variables`` without the restart
    shortwave (refused); with the biogeochemistry, its brine height's profiles
    (:func:`init_biogeochemistry_profiles`)."""
    d = state.dims
    n_cells, n_solve, ncat, nil = d["nCells"], d["nCellsSolve"], d["nCategories"], d["nIceLayers"]
    tracers = pools["tracers"]
    # init_column_level_ice_tracers: the whole arrays
    if configs["config_use_level_ice"] and not configs["config_do_restart"]:
        _fill(tracers["levelIceArea"], 1.0)
        _fill(tracers["levelIceVolume"], 1.0)
    init_column_itd(pools, state)
    # init_column_thermodynamic_profiles: colpkg_init_thermo, then every owned cell's profiles
    state.heat_capacity = state.ktherm != 0
    profile = _init_thermo(nil, state.ktherm)
    shape = (n_cells + 1, nil + 1)
    _launch("seaice_init_thermodynamic_profiles_v842", n_solve,
            _need(pools["initial"], "initialSalinityProfile", shape),
            _need(pools["initial"], "initialMeltingTemperatureProfile", shape), profile, np.int32(n_solve),
            np.int32(nil), np.int32(state.ktherm))
    if configs["config_use_column_biogeochemistry"]:
        init_biogeochemistry_profiles(pools, configs, state)
    # init_column_history_variables
    _fill(_need(pools["ridging"], "ratioRidgeThicknessToIce", (n_cells + 1, ncat)), 1.0)


def hbrine_grids(n_bio_layers: int, n_ice_layers: int) -> dict[str, np.ndarray]:
    """``colpkg_init_hbrine``'s grids (``ice_colpkg.F90:5268-5355``), each
    operation a binary64 one as gfortran's: ``biologyGrid`` (bgrid),
    ``interfaceBiologyGrid`` (igrid), ``verticalGrid`` (cgrid),
    ``interfaceGrid`` (icgrid) and ``verticalShortwaveGrid`` (swgrid)."""
    nb, ni = int(n_bio_layers), int(n_ice_layers)
    bgrid = [0.0] * (nb + 2)
    bgrid[nb + 1] = 1.0
    igrid = [0.0] * (nb + 1)
    igrid[nb] = 1.0
    zspace = 1.0 / max(1.0, float(nb))
    for k in range(2, nb + 2):
        bgrid[k - 1] = zspace * (float(k) - 1.5)
    for k in range(2, nb + 1):
        igrid[k - 1] = 0.5 * (bgrid[k] + bgrid[k - 1])
    zspace = 1.0 / float(ni)
    cgrid = [0.0] * (ni + 1)
    icgrid = [0.0] * (ni + 1)
    for k in range(2, ni + 2):
        cgrid[k - 1] = zspace * (float(k) - 1.5)
        icgrid[k - 1] = zspace * (float(k) - 1.0)
    swgrid = [0.0] * (ni + 1)
    half = zspace / 2.0
    swgrid[0] = 1.0 / 60.0 if 1.0 / 60.0 < half else half
    swgrid[1] = half
    for k in range(3, ni + 2):
        swgrid[k - 1] = zspace * (float(k) - 1.5)
    return {name: np.array(values, np.float64) for name, values in (
        ("biologyGrid", bgrid), ("interfaceBiologyGrid", igrid), ("verticalGrid", cgrid),
        ("interfaceGrid", icgrid), ("verticalShortwaveGrid", swgrid))}


def init_biogeochemistry_profiles(pools: dict, configs: dict, state: ColumnState) -> None:
    """``init_column_biogeochemistry_profiles`` (``mpas_seaice_column.F:11714-11990``)
    with the brine height: ``colpkg_init_hbrine``'s grids, and on every owned
    cell, without ``config_do_restart_hbrine``, ``newlyFormedIce`` 1 and
    ``brineFraction`` 1 (the tracer array's round trip changes no word).
    ``colpkg_init_hbrine`` also rewrites ``config_snow_porosity_at_ice_surface``
    to ``1 - rhos/rhoi`` when it is not positive, after the column package took
    its copy (``colpkg_init_parameters``); nothing reads the config again."""
    import cupy as cp
    d = state.dims
    bgc = pools["biogeochemistry"]
    for name, values in hbrine_grids(d["nBioLayers"], d["nIceLayers"]).items():
        target = bgc[name]
        target[...] = cp.asarray(values)
    if not configs["config_do_restart_hbrine"]:
        n_solve = d["nCellsSolve"]
        bgc["newlyFormedIce"][:n_solve] = 1
        # v8.4.2 sets it unconditionally (mpas_seaice_column.F:11904); without
        # config_use_brine the array is the inactive package's placeholder,
        # which nothing reads. With it, a missing array is an error, not a skip.
        if configs["config_use_brine"]:
            _need(pools["tracers"], "brineFraction", (d["nCells"] + 1, d["nCategories"], 1))[:n_solve] = 1.0
    if vertical_tracers(configs):
        init_bgc(pools, configs, state)
    if configs[ZSAL_FLAG] and not configs["config_do_restart_zsalinity"]:
        # colpkg_init_zsalinity per owned cell (ice_colpkg.F90:5448-5494): no Rayleigh
        # criterion yet and every category's vertical salinity sss*salt_loss. Its
        # dts_b (300 s up to 7 biology layers, else 50 s) is zsalinity_dts. v8.4.2
        # copies the tracer array back to verticalSalinity only with the vertical
        # tracers (mpas_seaice_column.F:6664,6802-6808): only zsal_solve writes it.
        n_solve = d["nCellsSolve"]
        n1 = d["nCells"] + 1
        bgc["rayleighCriteriaReal"][:n_solve] = 0.0
        if not solves_zsal(configs):
            return
        sss = _need(pools["ocean_coupling"], "seaSurfaceSalinity", (n1,))[:n_solve]
        zsal = _need(pools["tracers"], "verticalSalinity", (n1, d["nCategories"], d["nBioLayers"]))
        zsal[:n_solve] = (sss * 0.4)[:, None, None]


#: colpkg_init_OceanConcArray's (ice_colpkg.F90:5540-5603) layout of the ocean
#: concentrations, nZBGCTracers long: max_algae 3, max_doc 3, max_dic 1, max_don 1,
#: max_fe 2, max_aero 6
_MAX_ALGAE, _MAX_DOC, _MAX_DIC, _MAX_DON, _MAX_FE, _MAX_AERO = 3, 3, 1, 1, 2, 6


def zbgc_parameters(configs: dict, tracers: TracerObject) -> dict:
    """``colpkg_init_zbgc``'s per-bio-tracer parameters (``ice_colpkg.F90:1446-1490``),
    the nonreactive tracer's alone: its transport type (``nitratetype``,
    ``config_mobility_type_nitrate``), the retention and release times by
    type band (1 s each below 0, where none applies), ``zbgc_frac_init`` (1)
    and ``zbgc_init_frac`` (``frazil_scav``, or ``initbio_frac`` below 0)."""
    c = configs
    tau_min = float(c["config_rapid_mobile_to_stationary_time"])
    tau_max = float(c["config_long_mobile_to_stationary_time"])
    out = {"type": [], "tau_ret": [], "tau_rel": [], "frac_init": [], "init_frac": []}
    for _ in range(tracers.n_bio_tracers):
        kind = float(c["config_mobility_type_nitrate"])
        ret = rel = 1.0
        if 0.0 <= kind < 0.5:
            ret, rel = tau_min, tau_max
        elif 0.5 <= kind < 1.0:
            ret, rel = tau_min, tau_min
        elif 1.0 <= kind < 2.0:
            ret, rel = tau_max, tau_min
        elif kind >= 2.0:
            ret, rel = tau_max, tau_max
        out["type"].append(kind)
        out["tau_ret"].append(ret)
        out["tau_rel"].append(rel)
        out["frac_init"].append(1.0)
        out["init_frac"].append(float(c["config_new_ice_fraction_biotracer"]) if kind < 0.0
                                else float(c["config_fraction_biotracer_in_frazil"]))
    return out


def ocean_conc_array(pools: dict, configs: dict, dims: dict[str, int]) -> None:
    """``colpkg_init_OceanConcArray`` into ``oceanBioConcentrations`` on the
    owned cells, from the ocean concentrations ``initialize_coupler_fields``
    set, ``R_chl2N`` and ``R_S2N`` the chlorophyll and sulfur to nitrogen
    ratios (``ice_colpkg.F90:1096-1114``)."""
    import cupy as cp
    c, bgc = configs, pools["biogeochemistry"]
    n = dims["nCellsSolve"]
    out = _need(bgc, "oceanBioConcentrations", (dims["nCells"] + 1, dims["nZBGCTracers"]))
    for name in ("oceanAlgaeConc", "oceanDOCConc", "oceanDICConc", "oceanDONConc", "oceanDissolvedIronConc",
                 "oceanParticulateIronConc", "oceanZAerosolConc", "oceanNitrateConc", "oceanAmmoniumConc",
                 "oceanSilicateConc", "oceanDMSPConc", "oceanDMSConc", "oceanHumicsConc"):
        if not isinstance(bgc.get(name), cp.ndarray) or len(bgc[name]) < n:
            raise ValueError(f"{name} must be a device array over the owned cells")
    algae, doc, dic, don = (bgc[k][:n] for k in ("oceanAlgaeConc", "oceanDOCConc", "oceanDICConc", "oceanDONConc"))
    fed, fep, zaero = (bgc[k][:n] for k in ("oceanDissolvedIronConc", "oceanParticulateIronConc",
                                            "oceanZAerosolConc"))
    chl = [np.float64(c[f"config_ratio_chla_to_N_{a}"]) for a in ("diatoms", "small_plankton", "phaeocystis")]
    s2n = [np.float64(c[f"config_ratio_S_to_N_{a}"]) for a in ("diatoms", "small_plankton", "phaeocystis")]
    o = cp.zeros_like(out[:n])
    for k in range(_MAX_ALGAE):
        o[:, k] = algae[:, k]
        o[:, _MAX_ALGAE + _MAX_DOC + _MAX_DIC + 1 + k] = cp.multiply(chl[k], algae[:, k])
    ks = _MAX_ALGAE + 1
    for k in range(_MAX_DOC):
        o[:, ks + k] = doc[:, k]
    ks += _MAX_DOC
    for k in range(_MAX_DIC):
        o[:, ks + k] = dic[:, k]
    ks = 2 * _MAX_ALGAE + _MAX_DOC + _MAX_DIC + 7
    for k in range(_MAX_DON):
        o[:, ks + k] = don[:, k]
    # ocean_bio_all(ks) for the 1-based ks below is column ks - 1
    o[:, _MAX_ALGAE] = bgc["oceanNitrateConc"][:n]
    ks = 2 * _MAX_ALGAE + _MAX_DOC + 2 + _MAX_DIC
    o[:, ks - 1] = bgc["oceanAmmoniumConc"][:n]
    o[:, ks] = bgc["oceanSilicateConc"][:n]
    o[:, ks + 1] = cp.add(cp.add(cp.multiply(s2n[0], algae[:, 0]), cp.multiply(s2n[1], algae[:, 1])),
                          cp.multiply(s2n[2], algae[:, 2]))
    o[:, ks + 2] = bgc["oceanDMSPConc"][:n]
    o[:, ks + 3] = bgc["oceanDMSConc"][:n]
    o[:, ks + 4] = bgc["oceanNitrateConc"][:n]                       # PON
    ks = 2 * _MAX_ALGAE + _MAX_DOC + 7 + _MAX_DIC + _MAX_DON
    for k in range(_MAX_FE):
        o[:, ks + k] = fed[:, k]
    ks += _MAX_FE
    for k in range(_MAX_FE):
        o[:, ks + k] = fep[:, k]
    ks += _MAX_FE
    for k in range(_MAX_AERO):
        o[:, ks + k] = zaero[:, k]
    ks += _MAX_AERO + 1
    o[:, ks - 1] = bgc["oceanHumicsConc"][:n]
    out[:n] = o


def init_bgc(pools: dict, configs: dict, state: ColumnState) -> None:
    """``colpkg_init_bgc`` per owned cell (``ice_colpkg.F90:575-695``) with the
    vertical tracers: the ocean concentrations (:func:`ocean_conc_array`)
    and, without ``config_do_restart_bgc``, every category's tracers: the ice
    layers ``ocean_bio_all(bio_index_o)*zbgc_init_frac``, the snow's two 0,
    the mobile fraction ``zbgc_frac_init``. With the brine height and the
    BL99 thermodynamics (the mushy remap is refused) and the vertical
    salinity off (refused with the vertical tracers),
    ``config_scale_initial_vertical_bgc`` takes this branch too."""
    d, to = state.dims, state.tracers
    n_solve, nbl = d["nCellsSolve"], d["nBioLayers"]
    ocean_conc_array(pools, configs, d)
    if configs["config_do_restart_bgc"] or not n_solve:
        return
    par = zbgc_parameters(configs, to)
    bgc, t = pools["biogeochemistry"], pools["tracers"]
    conc = bgc["oceanBioConcentrations"]
    pon = _need(t, "verticalNonreactiveConc", (d["nCells"] + 1, d["nCategories"], nbl + 3))
    frac = _need(t, "mobileFraction", (d["nCells"] + 1, d["nCategories"], d["nZBGCTracers"]))
    for m, o in enumerate(to.bio_index_o):
        ice = conc[:n_solve, o - 1] * np.float64(par["init_frac"][m])
        pon[:n_solve, :, :nbl + 1] = ice[:, None, None]
        pon[:n_solve, :, nbl + 1:] = 0.0
        frac[:n_solve, :, m] = par["frac_init"][m]


def zsalinity_dts(n_bio_layers: int) -> float:
    """``colpkg_init_zsalinity``'s ``dts_b``, the vertical salinity's time step."""
    return 300.0 if n_bio_layers <= 7 else 50.0


def init_ice_state(pools: dict, configs: dict, state: ColumnState) -> None:
    """``init_ice_state`` (cold start): the initial condition, the initial
    cell area, the initial velocity, then ``seaice_column_aggregate``."""
    import cupy as cp
    d = state.dims
    n_cells, n_solve, ncat = d["nCells"], d["nCellsSolve"], d["nCategories"]
    nil, nsl = d["nIceLayers"], d["nSnowLayers"]
    t, agg = pools["tracers"], pools["tracers_aggregate"]
    cat = (n_cells + 1, ncat, 1)
    area, ice, snow = (_need(t, k, cat) for k in ("iceAreaCategory", "iceVolumeCategory", "snowVolumeCategory"))
    tsfc = _need(t, "surfaceTemperature", cat)
    kind = configs["config_initial_condition_type"]
    lat = _need(pools["mesh"], "latCell", (n_cells + 1,))
    north, south = _f64(configs["config_initial_latitude_north"]), _f64(configs["config_initial_latitude_south"])
    if kind == "uniform":
        # the routine binds its nCellsSolve pointer to the nCells dimension
        _launch("seaice_init_uniform_ice_v842", n_cells, area, ice, snow, tsfc,
                _need(agg, "iceAreaCell", (n_cells + 1,)), _need(agg, "surfaceTemperatureCell", (n_cells + 1,)),
                lat, north, south, _f64(configs["config_initial_ice_area"]),
                _f64(configs["config_initial_ice_volume"]), _f64(configs["config_initial_snow_volume"]),
                np.int32(n_cells), np.int32(ncat))
    elif kind == "cice_default":
        limits = pools["initial"]["categoryThicknessLimits"]
        cat_area = cp.zeros(ncat, np.float64)
        cat_thickness = cp.zeros(ncat, np.float64)
        _function("seaice_initial_category_areas_v842")((1,), (1,), (cat_area, cat_thickness, limits,
                                                                    np.int32(ncat)))
        ocean = pools["ocean_coupling"]
        initial = pools["initial"]
        _launch("seaice_init_cice_default_v842", n_solve, area, ice, snow, tsfc,
                _need(t, "iceEnthalpy", (n_cells + 1, ncat, nil)), _need(t, "snowEnthalpy", (n_cells + 1, ncat, nsl)),
                _need(t, "iceSalinity", (n_cells + 1, ncat, nil)),
                _need(pools["atmos_coupling"], "airTemperature", (n_cells + 1,)),
                _need(ocean, "seaSurfaceTemperature", (n_cells + 1,)),
                _need(ocean, "seaFreezingTemperature", (n_cells + 1,)),
                _need_int(ocean, "landIceMask", (n_cells + 1,)), lat,
                initial["initialSalinityProfile"], initial["initialMeltingTemperatureProfile"],
                cat_area, cat_thickness, north, south, np.int32(n_solve), np.int32(ncat), np.int32(nil),
                np.int32(nsl), np.int32(state.ktherm), np.int32(state.calc_tsfc),
                np.int32(bool(state.heat_capacity)))
    # no_ice, none: nothing
    _launch("seaice_column_copy_v842", n_cells + 1, _need(pools["icestate"], "iceAreaCellInitial", (n_cells + 1,)),
            _need(agg, "iceAreaCell", (n_cells + 1,)), np.int32(n_cells + 1))
    if configs["config_initial_velocity_type"] == "uniform":
        n_vertices = d["nVertices"]
        v1 = (n_vertices + 1,)
        solver = pools["velocity_solver"]
        _launch("seaice_init_velocity_uniform_v842", n_vertices, _need(solver, "uVelocity", v1),
                _need(solver, "vVelocity", v1), _need_int(pools["boundary"], "interiorVertex", v1),
                _need_int(pools["ocean_coupling"], "landIceMaskVertex", v1),
                _f64(configs["config_initial_uvelocity"]), _f64(configs["config_initial_vvelocity"]),
                np.int32(n_vertices))
    aggregate(pools, configs, state)


def reset_coupler_fluxes(pools: dict, configs: dict, state: ColumnState) -> None:
    """``seaice_reset_coupler_fluxes``, the biogeochemistry's ocean fluxes too."""
    for pool, name in (("atmos_fluxes", "sensibleHeatFlux"), ("atmos_fluxes", "latentHeatFlux"),
                       ("atmos_fluxes", "longwaveUp"), ("atmos_fluxes", "evaporativeWaterFlux"),
                       ("velocity_solver", "airStressCellU"), ("velocity_solver", "airStressCellV"),
                       ("shortwave", "absorbedShortwaveFlux"), ("atmos_coupling", "atmosReferenceSpeed10m"),
                       ("atmos_coupling", "atmosReferenceTemperature2m"),
                       ("atmos_coupling", "atmosReferenceHumidity2m"), ("ocean_fluxes", "oceanFreshWaterFlux"),
                       ("ocean_fluxes", "oceanSaltFlux"), ("ocean_fluxes", "oceanHeatFlux"),
                       ("ocean_fluxes", "oceanShortwaveFlux")):
        _fill(pools[pool][name], 0.0)
    if configs["config_use_column_biogeochemistry"]:
        # reset_atmospheric_coupler_fluxes' and reset_ocean_coupler_fluxes' biogeochemistry
        # (mpas_seaice_forcing.F): the deposition initialize_coupler_fields set is zeroed;
        # oceanAlgaeFlux is fetched, not reset
        for name in _BGC_ATMOS_FLUXES + _BGC_OCEAN_FLUXES:
            _fill(pools["biogeochemistry"][name], 0.0)


_BGC_ATMOS_FLUXES = ("atmosBioFluxes", "atmosBlackCarbonFlux", "atmosDustFlux")
_BGC_OCEAN_FLUXES = ("oceanBioFluxes", "oceanDOCFlux", "oceanDICFlux", "oceanDONFlux", "oceanNitrateFlux",
                     "oceanSilicateFlux", "oceanAmmoniumFlux", "oceanDMSFlux", "oceanDMSPpFlux", "oceanDMSPdFlux",
                     "oceanHumicsFlux", "oceanParticulateIronFlux", "oceanDissolvedIronFlux", "oceanDustIronFlux")


def initialize(pools: dict, configs: dict) -> ColumnState:
    """``seaice_init``'s column steps in its order: the parameters, the
    coupler fields, the column variables, the ice state, the coupler-flux
    reset. The mesh, forcing, velocity-solver and advection inits are the
    other modules' (:mod:`~woof.hex.mpas_seaice.timestep_v842`)."""
    state = init_parameters(pools, configs, biogeochemistry=True)
    initialize_coupler_fields(pools, configs, state)
    init_variables(pools, configs, state)
    init_ice_state(pools, configs, state)
    reset_coupler_fluxes(pools, configs, state)
    return state


def _tracer_field(pools: dict, name: str):
    """A tracer-array entry's field and layer: ``name`` or ``name:aerosol``."""
    base, _, sub = name.partition(":")
    return pools["tracers"][base], (int(sub) if sub else 0)


def _cell_field(pools: dict, name: str):
    base, _, sub = name.partition(":")
    return pools["tracers_aggregate"][f"{base}Cell"], (int(sub) if sub else 0)


def aggregate(pools: dict, configs: dict, state: ColumnState) -> None:
    """``seaice_column_aggregate``: per owned cell, the category tracer array
    gathered (``set_cice_tracer_array_category``), ``colpkg_aggregate``, and the
    cell tracer array scattered (``get_cice_tracer_array_cell``)."""
    import cupy as cp
    d = state.dims
    n_cells, n_solve, ncat = d["nCells"], d["nCellsSolve"], d["nCategories"]
    to = state.tracers
    nt = to.n_tracers
    t, agg = pools["tracers"], pools["tracers_aggregate"]
    trcrn = cp.zeros((max(n_solve, 1), ncat, nt), np.float64)
    trcr = cp.zeros((max(n_solve, 1), nt), np.float64)
    atrcr = cp.zeros((max(n_solve, 1), nt), np.float64)
    for name, first, layers in to.layout:
        src, sub = _tracer_field(pools, name)
        field_layers = src.shape[2]
        _launch("seaice_gather_tracer_v842", n_solve * ncat * layers, trcrn, src, np.int32(first - 1),
                np.int32(layers), np.int32(sub), np.int32(field_layers), np.int32(nt), np.int32(ncat),
                np.int32(n_solve))
    cat = (n_cells + 1, ncat, 1)
    c1 = (n_cells + 1,)
    _launch("colpkg_aggregate_v842", n_solve, _need(agg, "iceAreaCell", c1), _need(agg, "iceVolumeCell", c1),
            _need(agg, "snowVolumeCell", c1), _need(pools["icestate"], "openWaterArea", c1), trcr, atrcr,
            trcrn, _need(t, "iceAreaCategory", cat), _need(t, "iceVolumeCategory", cat),
            _need(t, "snowVolumeCategory", cat), cp.asarray(to.parent_index),
            cp.asarray(np.ascontiguousarray(to.first_ancestor_mask)), cp.asarray(to.ancestor_number),
            cp.asarray(np.ascontiguousarray(to.ancestor_indices)), np.int32(nt), np.int32(to.nt_tsfc),
            np.int32(to.nt_fbri), np.int32(n_solve), np.int32(ncat))
    for name, first, layers in to.layout + _cell_only_layout(to, d):
        if name in _NO_CELL_FIELD:
            continue
        dst, sub = _cell_field(pools, name)
        field_layers = dst.shape[1] if dst.ndim == 2 else 1
        _launch("seaice_scatter_cell_tracer_v842", n_solve * layers, dst, trcr, np.int32(first - 1),
                np.int32(layers), np.int32(sub), np.int32(field_layers), np.int32(nt), np.int32(n_solve))


#: tracer-array entries get_cice_tracer_array_cell does not copy (the mobile fraction has no cell field)
_NO_CELL_FIELD = ("mobileFraction",)


def _cell_only_layout(to: TracerObject, dims: dict[str, int]) -> list[tuple[str, int, int]]:
    """The cell fields get_cice_biogeochemistry_tracer_array_cell also writes:
    the nonreactive tracer's ice layers again, as ``verticalNonreactiveIceCell``
    (``mpas_seaice_column.F:7701-7709``)."""
    return [("verticalNonreactiveIce", to.bio_index[0], dims["nBioLayers"] + 1)] if to.n_bio_tracers else []


def update_state(pools: dict, configs: dict, state: ColumnState, update_type: str, dt: float,
                 ice_age_time_offset: float) -> None:
    """``seaice_column_update_state``: the aggregate, then the thermodynamic
    or transport area, volume and age tendencies over the owned cells."""
    if update_type not in ("transport", "thermodynamics"):
        raise ValueError(f"seaice_column_update_state: Unknown update type: {update_type}")
    aggregate(pools, configs, state)
    d = state.dims
    n_cells, n_solve = d["nCells"], d["nCellsSolve"]
    c1 = (n_cells + 1,)
    suffix = "Transport" if update_type == "transport" else "Thermodynamics"
    diag, agg = pools["diagnostics"], pools["tracers_aggregate"]
    use_age = bool(configs["config_use_ice_age"])
    age_cell = agg["iceAgeCell"] if use_age else _need(agg, "iceAreaCell", c1)
    age_tend = diag[f"iceAgeTendency{suffix}"] if use_age else _need(diag, f"iceAreaTendency{suffix}", c1)
    _launch("seaice_column_update_state_v842", n_solve, _need(diag, f"iceAreaTendency{suffix}", c1),
            _need(diag, f"iceVolumeTendency{suffix}", c1), age_tend, _need(agg, "iceAreaCell", c1),
            _need(agg, "iceVolumeCell", c1), age_cell, _f64(dt), _f64(ice_age_time_offset), np.int32(use_age),
            np.int32(n_solve))
