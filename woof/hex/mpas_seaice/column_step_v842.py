"""MPAS-Seaice v8.4.2: the column package's part of the time step, on the GPU.

``seaice_timestep`` (``mpas_seaice_time_integration.F:42-174``) runs the
column package around the dynamics. This module is that column part, as
MPAS runs it:

- :func:`reinitialize_diagnostics_thermodynamics` and
  :func:`reinitialize_diagnostics_bgc`, before the pre-dynamics;
- :func:`predynamics`, ``seaice_column_predynamics_time_integration``:
  ``column_prep_radiation`` (under the column shortwave with a computed
  surface temperature), ``column_vertical_thermodynamics``,
  ``column_biogeochemistry`` (under ``config_use_column_biogeochemistry``),
  ``column_itd_thermodynamics`` and ``seaice_column_update_state``
  ("thermodynamics", ``config_dt``, ``config_dt``);
- per dynamics subcycle, :func:`reinitialize_diagnostics_dynamics` and
  then, after the velocity solver and the advection, :func:`dynamics`,
  ``seaice_column_dynamics_time_integration``: ``column_ridging`` and
  ``seaice_column_update_state`` ("transport", ``dynamicsTimeStep``, 0);
- :func:`postdynamics`, ``seaice_column_postdynamics_time_integration``:
  ``column_radiation`` (not at initialisation) under the column shortwave,
  then ``seaice_column_coupling_prep`` and ``seaice_column_scale_fluxes``;
- :func:`reinitialize_fluxes`, ``seaice_timestep_finalize``'s
  ``seaice_column_reinitialize_fluxes`` (the atmospheric and oceanic
  fluxes), after the step's output.

Each stage is the port in :mod:`~woof.hex.mpas_seaice.column_therm_v842`,
:mod:`~woof.hex.mpas_seaice.column_bgc_v842`,
:mod:`~woof.hex.mpas_seaice.column_itd_v842`,
:mod:`~woof.hex.mpas_seaice.column_ridge_v842`,
:mod:`~woof.hex.mpas_seaice.column_radiation_v842`,
:mod:`~woof.hex.mpas_seaice.coupling_prep_v842` and
:func:`~woof.hex.mpas_seaice.column_v842.update_state`; the day of the year
and the seconds into the day are the clock's current time's, on
``config_calendar_type``'s calendar, as ``get_day_of_year`` and
``get_seconds_into_day`` read them. The re-initialisations are whole-array
assignments, MPAS's missing-cell element included.

:func:`column_step` is one step's column part without the dynamics (the
velocity solver and advection between the subcycles' re-initialisation and
column dynamics), as ``tools/mpas_seaice_oracle/run_column.F90``'s
``column_step`` stage runs it; held word for word over several steps to
v8.4.2 (``tests/test_mpas_seaice_column_step_parity.py``). An abort returns
``(ierr, cell, message)`` from the stage that aborts, and the step stops
there, as ``seaice_timestep`` returns on ``ierr > 0``.

The column biogeochemistry is carried with its brine height alone
(``config_use_brine``; each stage refuses the biogeochemical tracers): the
brine fraction travels through every stage, and a ``column_biogeochemistry``
that v8.4.2 dies on raises
:class:`~woof.hex.mpas_seaice.column_bgc_v842.CriticalAbort`. v8.4.2 itself
cannot take a step with the brine height alone: its
``seaice_column_reinitialize_diagnostics_bgc`` zeroes ``totalChlorophyll``,
which the algae's packages allocate, through a null pointer and dies. The
step refuses that by name; the opt-in ``bgc_inactive_diagnostics`` correction
(:data:`~woof.hex.mpas_seaice.column_v842.CORRECTIONS`) zeroes only the
fields that exist, and the rest of the step is v8.4.2's word for word.
v8.4.2's inert vertical salinity (``config_use_vertical_zsalinity``: never
solved, never copied to the tracer arrays) is carried too, and so is the
solved one (the ``zsal_solve`` correction).

Refused by name, besides each stage's own refusals: the vertical snow and ice tracers
(``column_separate_snow_ice_tracers`` and ``_combine_``), the ocean mixed
layer (``coupling_prep_v842``) and, under the velocity solver, the weak-form
stress divergence (its principal stresses are re-initialised in the
``velocity_weak`` pool, which this port does not carry).
"""
from __future__ import annotations

import numpy as np

from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_ocean.forward_v842 import format_time

from . import column_bgc_v842 as cbgc
from . import column_itd_v842 as citd
from . import column_radiation_v842 as crad
from . import column_ridge_v842 as crdg
from . import column_therm_v842 as ct
from . import column_v842 as cv
from . import coupling_prep_v842 as cprep

SOURCE_FILES = (
    "src/core_seaice/shared/mpas_seaice_time_integration.F:42-174,240-256",
    "src/core_seaice/shared/mpas_seaice_column.F:837-1047,10977-10990,11004-11157,11991-12491",
)

#: ``seaiceIceOceanDragCoefficient``: the stand-alone column constants' ``dragio``
#: (``column/constants/cice/ice_constants_colpkg.F90``)
ICE_OCEAN_DRAG_COEFFICIENT = 0.00536

_FORM_DRAG_FIELDS = (
    "airOceanDragCoefficientRatio", "oceanDragCoefficientSkin", "oceanDragCoefficientFloe",
    "oceanDragCoefficientKeel", "airDragCoefficientSkin", "airDragCoefficientFloe", "airDragCoefficientPond",
    "airDragCoefficientRidge", "dragFreeboard", "dragIceSnowDraft", "dragRidgeHeight", "dragRidgeSeparation",
    "dragKeelDepth", "dragKeelSeparation", "dragFloeLength", "dragFloeSeparation")
_BGC_FIELDS = (
    "primaryProduction", "totalChlorophyll", "netSpecificAlgalGrowthRate", "netBrineHeight", "zSalinityFlux",
    "zSalinityGDFlux", "oceanBioFluxes", "atmosIceBioFluxes", "snowIceBioFluxes", "totalVerticalBiologyIce",
    "totalVerticalBiologySnow")


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


def check_config(configs: dict) -> None:
    """Refuse, by name, what the column step does not carry (each stage
    refuses its own on top)."""
    c = configs
    if not c.get("config_use_column_package", False):
        _refuse("config_use_column_package", False, "the column step runs with the column package on",
                "config_use_column_package = .true.")
    if c.get("config_use_column_biogeochemistry", False):
        _refuse_inactive_bgc_diagnostics(c)
    if c.get("config_use_vertical_tracers", False):
        _refuse("config_use_vertical_tracers", True,
                "column_separate_snow_ice_tracers and column_combine_snow_ice_tracers are not ported",
                "config_use_vertical_tracers = .false.")
    if c.get("config_use_velocity_solver", True) and \
            str(c.get("config_stress_divergence_scheme", "variational")).rstrip(" ") == "weak":
        _refuse("config_stress_divergence_scheme", "weak",
                "the weak-form stress divergence with the column package is not held against an oracle "
                "(the column oracle carries no velocity_weak pool); it runs without the column package",
                "config_stress_divergence_scheme = 'variational'")
    crdg.check_config(c)
    cprep.check_config(c)


def _zero(pool: dict, *names: str) -> None:
    for name in names:
        pool[name][...] = 0


def _refuse_inactive_bgc_diagnostics(configs: dict) -> None:
    """Refuse v8.4.2's write to the diagnostics it has no array for
    (:func:`inactive_bgc_diagnostics`) unless the opt-in correction
    ``bgc_inactive_diagnostics`` is named."""
    if "bgc_inactive_diagnostics" in cv.corrections(configs):
        return
    missing = inactive_bgc_diagnostics(configs)
    if missing:
        _refuse("config_use_column_biogeochemistry", True,
                "v8.4.2's seaice_column_reinitialize_diagnostics_bgc zeroes " + ", ".join(missing)
                + ", whose packages are off and which therefore has no array: it writes to address 0 and "
                "the time step dies", "woof_v842_corrections = ('bgc_inactive_diagnostics',) (zero only "
                "the fields that exist), or config_use_column_biogeochemistry = .false.")


def inactive_bgc_diagnostics(configs: dict) -> tuple[str, ...]:
    """The fields ``seaice_column_reinitialize_diagnostics_bgc`` zeroes whose
    own Registry packages are all off, so v8.4.2 has no array for them (the
    Registry allocates a var by its own packages, not its struct's)."""
    active = cv.packages(configs)
    own = {name: [p for p in package_text.split(";") if p]
           for struct in cv.registry_fields()["var_structs"] if struct["name"] == "biogeochemistry"
           for name, _, _, package_text in struct["vars"]}
    return tuple(name for name in _BGC_FIELDS
                 if own[name] and not any(active[f"{p}Active"] for p in own[name]))


def reinitialize_atmospheric_fluxes(pools: dict, configs: dict) -> None:
    """``seaice_column_reinitialize_atmospheric_fluxes``."""
    _zero(pools["velocity_solver"], "airStressCellU", "airStressCellV")
    _zero(pools["atmos_coupling"], "atmosReferenceTemperature2m", "atmosReferenceHumidity2m",
          "atmosReferenceSpeed10m")
    if configs["config_use_column_package"]:
        _zero(pools["shortwave"], "absorbedShortwaveFlux")
        _zero(pools["atmos_fluxes"], "sensibleHeatFlux", "latentHeatFlux", "evaporativeWaterFlux", "longwaveUp")


def reinitialize_oceanic_fluxes(pools: dict, configs: dict) -> None:
    """``seaice_column_reinitialize_oceanic_fluxes``."""
    if configs["config_use_column_package"]:
        _zero(pools["ocean_fluxes"], "oceanFreshWaterFlux", "oceanSaltFlux", "oceanHeatFlux", "oceanShortwaveFlux")


def reinitialize_fluxes(pools: dict, configs: dict) -> None:
    """``seaice_column_reinitialize_fluxes`` (``seaice_timestep_finalize``)."""
    reinitialize_atmospheric_fluxes(pools, configs)
    reinitialize_oceanic_fluxes(pools, configs)


def _tendency(diagnostics: dict, aggregate: dict, kind: str, use_age: bool) -> None:
    """The tendencies' start: the cell area, volume and (with the ice age) age."""
    diagnostics[f"iceAreaTendency{kind}"][...] = aggregate["iceAreaCell"]
    diagnostics[f"iceVolumeTendency{kind}"][...] = aggregate["iceVolumeCell"]
    if use_age:
        diagnostics[f"iceAgeTendency{kind}"][...] = aggregate["iceAgeCell"]
    else:
        diagnostics[f"iceAgeTendency{kind}"][...] = 0


def reinitialize_diagnostics_thermodynamics(pools: dict, configs: dict) -> None:
    """``seaice_column_reinitialize_diagnostics_thermodynamics``."""
    if not configs["config_use_column_package"]:
        return
    _zero(pools["atmos_fluxes"], "surfaceHeatFlux", "surfaceConductiveFlux", "surfaceHeatFluxCategory",
          "surfaceConductiveFluxCategory", "latentHeatFluxCategory", "sensibleHeatFluxCategory")
    _zero(pools["melt_growth_rates"], "congelation", "frazilFormation", "snowiceFormation", "snowThicknessChange",
          "surfaceIceMelt", "snowMelt", "basalIceMelt", "lateralIceMelt")
    _tendency(pools["diagnostics"], pools["tracers_aggregate"], "Thermodynamics", bool(configs["config_use_ice_age"]))
    _zero(pools["ponds"], "pondFreshWaterFlux")
    _zero(pools["shortwave"], "bareIceAlbedoCell", "snowAlbedoCell", "pondAlbedoCell")
    drag = pools["drag"]
    drag["oceanDragCoefficient"][...] = ICE_OCEAN_DRAG_COEFFICIENT
    drag["airDragCoefficient"][...] = cv._folded_constants()["airDragCoefficient"]
    if configs["config_use_form_drag"]:
        _zero(drag, *_FORM_DRAG_FIELDS)


def reinitialize_diagnostics_dynamics(pools: dict, configs: dict) -> None:
    """``seaice_column_reinitialize_diagnostics_dynamics`` (the variational
    principal stresses; the weak form is refused)."""
    if not configs["config_use_column_package"]:
        return
    _zero(pools["velocity_solver"], "oceanStressU", "oceanStressV", "airStressVertexU", "airStressVertexV",
          "stressDivergenceU", "stressDivergenceV", "surfaceTiltForceU", "surfaceTiltForceV")
    if configs.get("config_use_velocity_solver", True) and \
            configs.get("config_stress_divergence_scheme", "variational") == "variational":
        _zero(pools["velocity_variational"], "principalStress1", "principalStress2", "replacementPressure")
    _zero(pools["ridging"], "areaLossRidge", "areaGainRidge", "iceVolumeRidged", "openingRateRidge")
    _tendency(pools["diagnostics"], pools["tracers_aggregate"], "Transport", bool(configs["config_use_ice_age"]))


def reinitialize_diagnostics_bgc(pools: dict, configs: dict) -> None:
    """``seaice_column_reinitialize_diagnostics_bgc``
    (``mpas_seaice_column.F:12403-12491``): the biogeochemistry's diagnostics
    and fluxes, and the shortwave's ``bioTracerShortwave``, zeroed; under the
    ``bgc_inactive_diagnostics`` correction, only those whose packages are on
    (:func:`inactive_bgc_diagnostics`)."""
    if not configs["config_use_column_package"]:
        return
    if configs["config_use_column_biogeochemistry"]:
        # the fields v8.4.2 has no array for: refused unless corrected, and then skipped
        _refuse_inactive_bgc_diagnostics(configs)
        skip = inactive_bgc_diagnostics(configs)
        _zero(pools["biogeochemistry"], *(name for name in _BGC_FIELDS if name not in skip))
    if configs["config_use_column_biogeochemistry"] or configs["config_use_column_shortwave"]:
        _zero(pools["biogeochemistry"], "bioTracerShortwave")


def clock_day(configs: dict, now: str) -> tuple[float, int]:
    """``get_day_of_year`` and ``get_seconds_into_day`` at the clock's time ``now``."""
    return ct.day_of_year(now, configs["config_calendar_type"]), crad.seconds_into_day(now)


def predynamics(pools: dict, configs: dict, state: "cv.ColumnState", now: str) -> tuple:
    """``seaice_column_predynamics_time_integration`` at the clock's time
    ``now``: ``(ierr, cell, message)``; ``state.warnings`` the ITD's
    zap_snow_temperature lines."""
    check_config(configs)
    c = configs
    day, _ = clock_day(c, now)
    state.warnings = []
    if c["config_use_column_shortwave"] and c["config_calc_surface_temperature"]:
        crad.prep_radiation(pools, c, state)
    if c["config_use_column_vertical_thermodynamics"]:
        abort = ct.vertical_thermodynamics(pools, c, state, day=day)
        if abort[0] > 0:
            return abort
    if c["config_use_column_biogeochemistry"]:
        abort = cbgc.biogeochemistry(pools, c, state)
        if abort[0] > 0:
            return abort
    if c["config_use_column_itd_thermodynamics"]:
        abort = citd.itd_thermodynamics(pools, c, state, day=day)
        warnings = [citd.ZAP_WARNING] * int(state.warnings)
        state.warnings = warnings
        if abort[0] > 0:
            return abort
    dt = float(c["config_dt"])
    cv.update_state(pools, c, state, "thermodynamics", dt, dt)
    return 0, None, None


def dynamics(pools: dict, configs: dict, state: "cv.ColumnState") -> tuple:
    """``seaice_column_dynamics_time_integration``: ``(ierr, cell, message)``;
    ``state.warnings`` the ridging's graded lines."""
    import cupy as cp

    check_config(configs)
    state.warnings = []
    if configs["config_use_column_ridging"]:
        abort = crdg.ridging(pools, configs, state)
        if abort[0] > 0:
            return abort
    step = float(cp.asnumpy(pools["velocity_solver"]["dynamicsTimeStep"]).reshape(-1)[0])
    cv.update_state(pools, configs, state, "transport", step, 0.0)
    return 0, None, None


def postdynamics(pools: dict, configs: dict, state: "cv.ColumnState", now: str) -> None:
    """``seaice_column_postdynamics_time_integration`` at the clock's time ``now``."""
    check_config(configs)
    if configs["config_use_column_shortwave"]:
        day, sec = clock_day(configs, now)
        crad.radiation(pools, configs, state, initialization=False, day=day, sec=sec)
    n_solve = state.dims["nCellsSolve"]
    cprep.coupling_prep(pools=pools, n_cells_solve=n_solve, dt=float(configs["config_dt"]), configs=configs)


def column_step(pools: dict, configs: dict, state: "cv.ColumnState", now: str) -> tuple:
    """One step's column part at the clock's (advanced) time ``now``, without
    the dynamics between the subcycles' re-initialisation and column
    dynamics: ``(ierr, cell, message)`` and, in ``state.warnings``, the
    graded warning lines in log order (the ITD's, then each subcycle's
    ridging's)."""
    check_config(configs)
    reinitialize_diagnostics_thermodynamics(pools, configs)
    reinitialize_diagnostics_bgc(pools, configs)
    abort = predynamics(pools, configs, state, now)
    warnings = list(state.warnings)
    if abort[0] > 0:
        state.warnings = warnings
        return abort
    for _ in range(int(configs["config_dynamics_subcycle_number"])):
        reinitialize_diagnostics_dynamics(pools, configs)
        abort = dynamics(pools, configs, state)
        warnings += state.warnings
        if abort[0] > 0:
            state.warnings = warnings
            return abort
    postdynamics(pools, configs, state, now)
    state.warnings = warnings
    return 0, None, None


def time_stamps(configs: dict, n_steps: int) -> list[str]:
    """The clock's time after each of ``n_steps`` steps of ``config_dt`` from
    ``config_start_time``, on ``config_calendar_type``'s calendar."""
    from woof.hex.mpas_seaice.timestep_v842 import clock_from_configs
    clock = clock_from_configs(configs)
    out = []
    for _ in range(n_steps):
        clock.now += clock.step
        out.append(format_time(clock.now, clock.calendar))
    return out


def run(pools: dict, configs: dict, state: "cv.ColumnState", n_steps: int) -> tuple:
    """``n_steps`` column steps from ``config_start_time`` (each after the
    first preceded by :func:`reinitialize_fluxes`), stopping at an abort, as
    ``run_column.F90``'s ``column_step`` stage; the last step's fluxes are
    left for the coupler. Needs ``init_column_itd`` (``state.hi_min``)."""
    warnings: list[str] = []
    abort = (0, None, None)
    for i, now in enumerate(time_stamps(configs, n_steps)):
        if i:
            reinitialize_fluxes(pools, configs)
        abort = column_step(pools, configs, state, now)
        warnings += state.warnings
        if abort[0] > 0:
            break
    state.warnings = warnings
    return abort


__all__ = ("check_config", "reinitialize_fluxes", "reinitialize_atmospheric_fluxes", "reinitialize_oceanic_fluxes",
           "reinitialize_diagnostics_thermodynamics", "reinitialize_diagnostics_dynamics",
           "reinitialize_diagnostics_bgc", "predynamics", "dynamics", "postdynamics", "column_step", "run",
           "time_stamps", "clock_day")
