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

Refused by name: the column biogeochemistry; more than one snow layer
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
    "7805-8222,10138-10232,10232-10420,10868-10990",
    "src/core_seaice/shared/mpas_seaice_initialize.F:231-850,1309-1377,1707-1944",
    "src/core_seaice/shared/mpas_seaice_forcing.F:1600-1840",
    "src/core_seaice/model_forward/mpas_seaice_core_interface.F:222-531",
    "src/core_seaice/column/ice_colpkg.F90:82-210,326-375,416-490,1599-1718,3170-3290",
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


def refuse_unported(configs: dict, dims: dict[str, int]) -> None:
    """Refuse, by name, what this stage of the column port does not carry."""
    c = configs
    if not c.get("config_use_column_package", False):
        _refuse("config_use_column_package", False, "the column port runs only with the column package on",
                "config_use_column_package = .true.")
    if c["config_use_column_biogeochemistry"]:
        _refuse("config_use_column_biogeochemistry", True, "the column biogeochemistry is not ported",
                "config_use_column_biogeochemistry = .false.")
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
                "colpkg": [self.nt_tsfc, self.nt_fbri]}


def tracer_object(configs: dict, dims: dict[str, int]) -> TracerObject:
    """``init_column_tracer_object`` without the biogeochemistry: the tracer
    number, child (index), parent, first-ancestor and ancestor routines."""
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
    return TracerObject(n_tracers=n_tracers, index=idx, parent_index=parent, first_ancestor_mask=mask,
                        ancestor_number=number, ancestor_indices=ancestors, nt_tsfc=idx["surfaceTemperature"],
                        nt_fbri=0, layout=layout)


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


def init_parameters(pools: dict, configs: dict) -> ColumnState:
    """``seaice_init_column_physics_package_parameters``: host side. The
    stand-in pointers of the packages that are off change no state the port
    reads (the absent fields stay ``None``)."""
    mesh = pools["mesh"]
    dims = {k: int(mesh[k]) for k in ("nCells", "nCellsSolve", "nVertices", "nEdges", "maxEdges",
                                       "vertexDegree") if k in mesh}
    for k, v in mesh.items():
        if isinstance(v, (int, np.integer)):
            dims.setdefault(k, int(v))
    refuse_unported(configs, dims)
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
    """``initialize_coupler_fields`` (``mpas_seaice_initialize.F:1707-1944``)
    without the biogeochemistry (refused): the whole-array assignments cover
    MPAS's missing-cell element, the freezing temperature loop nCells."""
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
    """``seaice_init_column_physics_package_variables`` without the
    biogeochemistry and the restart shortwave (refused)."""
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
    # init_column_history_variables
    _fill(_need(pools["ridging"], "ratioRidgeThicknessToIce", (n_cells + 1, ncat)), 1.0)


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
    """``seaice_reset_coupler_fluxes`` without the biogeochemistry (refused)."""
    for pool, name in (("atmos_fluxes", "sensibleHeatFlux"), ("atmos_fluxes", "latentHeatFlux"),
                       ("atmos_fluxes", "longwaveUp"), ("atmos_fluxes", "evaporativeWaterFlux"),
                       ("velocity_solver", "airStressCellU"), ("velocity_solver", "airStressCellV"),
                       ("shortwave", "absorbedShortwaveFlux"), ("atmos_coupling", "atmosReferenceSpeed10m"),
                       ("atmos_coupling", "atmosReferenceTemperature2m"),
                       ("atmos_coupling", "atmosReferenceHumidity2m"), ("ocean_fluxes", "oceanFreshWaterFlux"),
                       ("ocean_fluxes", "oceanSaltFlux"), ("ocean_fluxes", "oceanHeatFlux"),
                       ("ocean_fluxes", "oceanShortwaveFlux")):
        _fill(pools[pool][name], 0.0)


def initialize(pools: dict, configs: dict) -> ColumnState:
    """``seaice_init``'s column steps in its order: the parameters, the
    coupler fields, the column variables, the ice state, the coupler-flux
    reset. The mesh, forcing, velocity-solver and advection inits are the
    other modules' (:mod:`~woof.hex.mpas_seaice.timestep_v842`)."""
    state = init_parameters(pools, configs)
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
    for name, first, layers in to.layout:
        dst, sub = _cell_field(pools, name)
        field_layers = dst.shape[1] if dst.ndim == 2 else 1
        _launch("seaice_scatter_cell_tracer_v842", n_solve * layers, dst, trcr, np.int32(first - 1),
                np.int32(layers), np.int32(sub), np.int32(field_layers), np.int32(nt), np.int32(n_solve))


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
