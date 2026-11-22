"""MPAS-Seaice v8.4.2: the forward model's time step on the GPU.

:func:`initialize` and :func:`advance` are what ``seaice_core_init`` and
``seaice_core_run`` (``mpas_seaice_core.F``) do to the state with forcing off,
and with the column package off or on (see "The column package" below):

- init: the velocity solver's time steps (``dynamicsTimeStep = config_dt /
  config_dynamics_subcycle_number``, ``elasticTimeStep = dynamicsTimeStep /
  config_elastic_subcycle_number``) and ``seaice_initialize_time_diagnostics``
  (``xtime`` and, on a cold start, ``simulationStartTime``). The variational
  basis, masks and the rest of the mesh setup are the input's; under
  ``config_use_advection`` the remap's init geometry is computed (see
  :func:`initialize`);
- each step: the clock advanced, then ``seaice_timestep``:
  ``seaice_set_time_diagnostics`` (``xtime``, ``daysSinceStartOfSim``),
  ``store_timestep_counter`` (``itimestep``, ``timeAverageTestVariable``), and
  ``config_dynamics_subcycle_number`` times ``seaice_run_velocity_solver``
  (:mod:`~woof.hex.mpas_seaice.presubcycle_v842`'s pre-subcycle,
  ``config_elastic_subcycle_number`` elastic subcycles of
  :mod:`~woof.hex.mpas_seaice.evp_v842` and
  :mod:`~woof.hex.mpas_seaice.velocity_v842`, and the post-subcycle) and,
  under ``config_use_advection``, ``seaice_run_advection``'s incremental remap
  (:mod:`~woof.hex.mpas_seaice.remap_v842`, on the geometry
  :mod:`~woof.hex.mpas_seaice.remap_init_v842` derives at init); then
  ``seaice_timestep_finalize`` (the atmospheric fluxes re-initialised).

The pools are dicts of device arrays named as in MPAS (``mesh``, ``boundary``,
``velocity_solver``, ``velocity_variational``, ``icestate``, ``tracers``,
``tracers_aggregate``, ``atmos_coupling``, ``ocean_coupling``, ``ridging``,
``diagnostics``), with MPAS's missing-cell element; dimensions are host
integers. :func:`woof.hex.mpas_ocean.forward_v842.write_restart` and
``read_restart`` round-trip a run (with the clock and step counter).

Held word for word over N steps to v8.4.2 (``seaice_timestep``,
``store_timestep_counter``, ``seaice_timestep_finalize``, the column wrappers
and the time diagnostics cut verbatim, ``mpas_seaice_velocity_solver.F``
compiled verbatim; ``tools/mpas_seaice_oracle/run_seaice_step.F90``,
``tests/test_mpas_seaice_timestep_parity.py``; with advection
``tests/test_mpas_seaice_timestep_remap_parity.py``).

The column package (``config_use_column_package``, RKIND 8 only, as
``mpas_seaice_column.F`` does not compile at RKIND 4): each step runs
``seaice_timestep``'s column calls around the dynamics, as
:mod:`~woof.hex.mpas_seaice.column_step_v842` ports them: the thermodynamic
and biogeochemical diagnostics' re-initialisation and the pre-dynamics before
the subcycles; in each subcycle the dynamic diagnostics' re-initialisation
before the velocity solver and the column dynamics (ridging) after the
advection; the post-dynamics after them; and ``seaice_timestep_finalize``'s
``seaice_column_reinitialize_fluxes``. The velocity solver runs with the
column flags: under the column vertical thermodynamics its ice strength is
``colpkg_ice_strength``'s
(:func:`~woof.hex.mpas_seaice.column_ridge_v842.ice_strength`), its air
stress the thermodynamics', its initial cell area the thermodynamics', and
its final divergence and shear give the ridging's convergence and shear. The
day of the year and the seconds into the day are the advanced clock's. A
column abort raises :class:`ColumnAbort` (``seaice_check_critical_error``
stops the model there) with the pools as MPAS leaves them. :func:`initialize`
runs ``init_column_itd`` (the thickness limits and ``hi_min``); the rest of
``seaice_init_column_physics_package_variables`` and the initial state are
the input's (:mod:`~woof.hex.mpas_seaice.column_v842` ports them). Held word
for word over N steps to v8.4.2's ``seaice_timestep`` with the column
package and the velocity solver compiled verbatim
(``tools/mpas_seaice_oracle/run_column.F90``'s ``seaice_step`` stage,
``tests/test_mpas_seaice_coupled_step_parity.py``).

The weak-form stress divergence (``config_stress_divergence_scheme =
'weak'``) runs :mod:`~woof.hex.mpas_seaice.weak_v842` in the variational
solver's place: its init (``seaice_init_velocity_solver_weak``, under
``basis="compute"``) and each subcycle's internal stress, with the metric
terms' tangents worked out once per :func:`advance` (``metric_libm``, which
a sphere needs; ``weak_corrections``). Held word for word over N steps
(``tests/test_mpas_seaice_weak_parity.py``).

Refused by name: upwind advection (v8.4.2 cannot run it), a periodic
planar mesh under advection, forcing, the weak form with the column
package, the unit tests, a restart start, halo exchanges, and what the column step
refuses (the column biogeochemistry but its brine height, the vertical snow
and ice tracers, the ocean mixed layer's constant boundary layer). ``seaice_check_state`` changes
no state and is not run.
"""
from __future__ import annotations

import logging

import numpy as np

from woof.hex.errors import ConfigurationRefusal, MpasPortError
from woof.hex.mpas_ocean.forward_v842 import (Clock, calendar_of, format_time, parse_time, read_restart,
                                              write_restart)

from . import evp_v842, presubcycle_v842, remap_v842, velocity_v842

_log = logging.getLogger(__name__)

SOURCE_FILES = (
    "src/core_seaice/model_forward/mpas_seaice_core.F:260-410",
    "src/core_seaice/shared/mpas_seaice_time_integration.F:42-259",
    "src/core_seaice/shared/mpas_seaice_diagnostics.F:47-156",
    "src/core_seaice/shared/mpas_seaice_column.F:11028-11157",
    "src/core_seaice/shared/mpas_seaice_velocity_solver.F:80-135",
    "src/core_seaice/shared/mpas_seaice_advection.F:85-143",
    "src/core_seaice/shared/mpas_seaice_velocity_solver.F:1263-1430",
)

__all__ = ("Clock", "ColumnAbort", "initialize", "advance", "timestep_finalize", "write_restart", "read_restart",
           "clock_from_configs")


class ColumnAbort(MpasPortError):
    """A column routine aborted in step ``step`` (``ierr``, the first aborting
    owned cell's ``indexToCellID`` and the logged message): v8.4.2's
    ``seaice_timestep`` returns there and ``seaice_check_critical_error``
    stops the model. The pools are as MPAS leaves them."""

    def __init__(self, step: int, ierr: int, cell: int | None, message: str | None) -> None:
        self.step, self.ierr, self.cell, self.message = step, ierr, cell, message
        super().__init__(f"step {step}: ierr {ierr} at indexToCellID {cell}: {message}")


PRECISIONS = {"r4": np.float32, "r8": np.float64}


def _precision_of(array) -> str:
    for precision, dtype in PRECISIONS.items():
        if array.dtype == dtype:
            return precision
    raise ValueError(f"MPAS-Seaice arrays are float32 or float64, not {array.dtype}")


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


def check_config(configs: dict) -> None:
    """Refuse, by name, what the step does not carry."""
    if configs.get("config_use_column_package", False):
        from . import column_step_v842 as cstep
        cstep.check_config(configs)
        cf = configs.get("config_ratio_ridging_work_to_PE", 17.0)
        if cf != 17.0:
            _refuse("config_ratio_ridging_work_to_PE", cf,
                    "v8.4.2 does not read it: colpkg_ice_strength uses the column constants' Cf = 17",
                    "config_ratio_ridging_work_to_PE = 17.0")
    for knob, reason in (
            ("config_use_forcing", "the sea-ice forcing streams are not ported; the forcing fields are held"),
            ("config_perform_unit_test", "the unit tests are not ported"),
            ("config_do_restart", "this port restarts from its own restart (read_restart), not MPAS's stream"),
            ("config_use_halo_exch", "halo exchanges: this port runs one block")):
        if configs.get(knob, False):
            _refuse(knob, True, reason, f"{knob} = .false.")
    if configs.get("config_use_advection", False):
        remap_v842.check_config(configs)
    if not configs.get("config_use_velocity_solver", True):
        return
    scheme = _scheme(configs)
    if scheme not in ("variational", "weak"):
        _refuse("config_stress_divergence_scheme", scheme,
                "only the variational and weak stress divergences are ported",
                "config_stress_divergence_scheme = 'variational' | 'weak'")
    if scheme == "variational":
        from . import variational_basis_v842
        variational_basis_v842.check_config(configs)


def _scheme(configs: dict) -> str:
    return str(configs.get("config_stress_divergence_scheme", "variational")).rstrip(" ")


def advection_tree(pools: dict, configs: dict):
    """The remap's tracer list: the packages are the configs ``package:NAME``,
    a tracer field is active where the tracers pool carries it."""
    packages = {k.split(":", 1)[1]: bool(v) for k, v in configs.items() if k.startswith("package:")}
    tracers = pools["tracers"]
    return remap_v842.tracer_tree(packages, configs, lambda name: tracers.get(name) is not None)


def clock_from_configs(configs: dict) -> Clock:
    """The clock: ``config_start_time``, the step ``config_dt`` (whole seconds),
    on ``config_calendar_type``'s calendar (the Registry default
    ``gregorian_noleap``, or ``gregorian``)."""
    dt = float(configs["config_dt"])
    if dt != int(dt):
        _refuse("config_dt", dt, "the clock runs on whole seconds", "config_dt a whole number of seconds")
    calendar = calendar_of(configs)
    start = parse_time(configs["config_start_time"], calendar)
    return Clock(start, start, int(dt), calendar)


def _tilt(configs: dict) -> str:
    if not configs.get("config_use_surface_tilt", False):
        return "none"
    return "geostrophic" if configs.get("config_geostrophic_surface_tilt", False) else "ssh_gradient"


def initialize(*, pools: dict, configs: dict, remap_geometry: str | None = None, basis: str | None = None,
               metric_libm: str | None = None, basis_corrections=()) -> Clock:
    """The init's time steps and time diagnostics; the clock at its start.

    Under ``config_use_advection`` with ``remap_geometry="compute"`` (the
    default then), ``seaice_init_advection_incremental_remap``'s geometry
    (:mod:`~woof.hex.mpas_seaice.remap_init_v842`: the frames, vertex
    coordinates, moments, rotated coordinates, the RBF vectors and
    ``coeffs_reconstruct``) is derived from the mesh into the pools;
    ``"input"`` keeps the pools' own (a harness replaying a fixture whose
    geometry came from elsewhere). The halo-layer warnings v8.4.2 logs go to
    this module's logger. With the column package, ``init_column_itd``
    writes the thickness limits (``categoryThicknessLimits``).

    With the velocity solver on, ``basis="compute"`` (the default then; v8.4.2
    always computes it) runs ``seaice_init_velocity_solver_variational``
    (:mod:`~woof.hex.mpas_seaice.variational_basis_v842`) after
    ``seaice_init_evp``'s place: the basis gradients and integrals,
    ``tanLatVertexRotatedOverRadius`` and ``cellVerticesAtVertex`` from the
    mesh, into the velocity_variational pool. ``metric_libm`` and
    ``basis_corrections`` go to it (at RKIND 8 the metric terms need
    ``metric_libm`` unless they are off or trivially zero or NaN). ``"input"``
    keeps the pool's own (a harness replaying a fixture whose basis came
    from elsewhere). Under the weak form ``"compute"`` runs
    ``seaice_init_velocity_solver_weak`` instead
    (:func:`~woof.hex.mpas_seaice.weak_v842.initialize_pools`: the normal
    vectors and rotated latitudes into the velocity_weak pool; on a sphere
    it needs ``metric_libm``)."""
    check_config(configs)
    solver_on = configs.get("config_use_velocity_solver", True)
    if basis is None:
        basis = "compute" if solver_on else "input"
    if basis not in ("compute", "input"):
        raise ValueError(f"basis is 'compute' or 'input', not {basis!r}")
    if configs.get("config_use_column_package", False):
        from . import column_v842 as cv
        _column_precision(pools)
        cv.init_column_itd(pools, cv.init_parameters(pools, configs, biogeochemistry=True))
    advection = configs.get("config_use_advection", False)
    if remap_geometry is None:
        remap_geometry = "compute" if advection else "input"
    if remap_geometry not in ("compute", "input"):
        raise ValueError(f"remap_geometry is 'compute' or 'input', not {remap_geometry!r}")
    if advection and remap_geometry == "compute":
        from . import remap_init_v842
        for w in remap_init_v842.initialize_pools(pools, configs):
            _log.info("%s %s", w["message"], w["ints"])
    if solver_on and basis == "compute" and _scheme(configs) == "weak":
        from . import weak_v842
        weak_v842.initialize_pools(pools, configs, metric_libm=metric_libm)
    elif solver_on and basis == "compute":
        from . import variational_basis_v842
        for w in variational_basis_v842.initialize_pools(pools, configs, metric_libm=metric_libm,
                                                         corrections=basis_corrections):
            _log.warning("%s", w["message"])
    solver = pools["velocity_solver"]
    real = PRECISIONS[_precision_of(solver["uVelocity"])]
    dynamics = real(real(configs["config_dt"]) / real(configs["config_dynamics_subcycle_number"]))
    solver["dynamicsTimeStep"][...] = dynamics
    solver["elasticTimeStep"][...] = real(dynamics / real(configs["config_elastic_subcycle_number"]))
    clock = clock_from_configs(configs)
    d = pools["diagnostics"]
    d["xtime"] = clock.time_stamp
    d["simulationStartTime"] = d["xtime"]
    return clock


def _column_precision(pools: dict) -> None:
    if _precision_of(pools["velocity_solver"]["uVelocity"]) != "r8":
        _refuse("config_use_column_package", True,
                "the column package is RKIND 8 only (mpas_seaice_column.F does not compile at RKIND 4)",
                "float64 pools, or config_use_column_package = .false.")


def _column_state(pools: dict, configs: dict):
    """The column package's host parameters, and ``hi_min`` from
    ``colpkg_init_itd`` run on a copy of the thickness limits (the pools'
    keep what the run has made of them, as a running model's do)."""
    from . import column_v842 as cv
    state = cv.init_parameters(pools, configs, biogeochemistry=True)
    limits = pools["initial"]["categoryThicknessLimits"]
    cv.init_column_itd({"initial": {"categoryThicknessLimits": limits.copy()}}, state)
    return state


def _velocity_solver(pools: dict, configs: dict, consts, column_state=None, weak_tangents=None,
                     weak_corrections=()) -> None:
    """``seaice_run_velocity_solver``: pre-subcycle, elastic subcycles,
    post-subcycle. ``weak_tangents`` (the weak form's): the device
    ``tan(latCellRotated)`` and ``tan(latVertexRotated)``."""
    p = pools
    mesh = {**p["mesh"]}
    solver, variational, icestate = p["velocity_solver"], p.get("velocity_variational"), p["icestate"]
    weak = weak_tangents is not None
    if weak:
        from . import weak_v842
        vw = p["velocity_weak"]
    column = column_state is not None
    flags = {"use_column_package": column,
             "use_column_vertical_thermodynamics": column and bool(
                 configs.get("config_use_column_vertical_thermodynamics", True))}
    strength = None
    if flags["use_column_vertical_thermodynamics"]:
        from . import column_ridge_v842 as crdg

        def strength():
            crdg.ice_strength(pools, configs, column_state.dims["nIceLayers"])
    use_ocean_stress = bool(configs.get("config_use_ocean_stress", False))
    revised = bool(configs.get("config_revised_evp", False))
    dynamics_dt = solver["dynamicsTimeStep"].item()
    elastic_dt = solver["elasticTimeStep"].item()
    real = PRECISIONS[_precision_of(solver["uVelocity"])]
    dynamics_dt, elastic_dt = real(dynamics_dt), real(elastic_dt)
    # the category tracers' (1, nCategories, nCells) as the pre-subcycle reads them, (nCategories, nCells)
    tracers = {name: p["tracers"][name].reshape(p["tracers"][name].shape[0], p["tracers"][name].shape[1])
               for name in ("iceAreaCategory", "iceVolumeCategory", "snowVolumeCategory")}
    presubcycle_v842.pre_subcycle(
        mesh=mesh, boundary=p["boundary"], tracers=tracers, aggregate=p["tracers_aggregate"],
        icestate=icestate, ocean=p["ocean_coupling"], atmos=p["atmos_coupling"], solver=solver,
        variational=variational, use_air_stress=bool(configs.get("config_use_air_stress", False)),
        use_ocean_stress=use_ocean_stress, tilt=_tilt(configs), column_ice_strength=strength,
        init_subcycle=(lambda: weak_v842.init_subcycle_variables(mesh=mesh, solver=solver, weak=vw)) if weak
        else None, **flags)
    inertia = consts[1:2]
    radius = configs.get("mesh:sphere_radius", configs.get("sphere_radius", 0.0))
    # the subcycles run on the same arrays, so the first one checks them for all
    for subcycle in range(int(configs["config_elastic_subcycle_number"])):
        if weak:
            weak_v842.internal_stress_weak(mesh=mesh, solver=solver, weak=vw, consts=consts,
                                           tangents=weak_tangents, elastic_time_step=elastic_dt,
                                           revised_evp=revised, sphere_radius=radius, check=subcycle == 0)
        else:
            evp_v842.internal_stress_variational(mesh=mesh, solver=solver, variational=variational,
                                                 consts=consts, elastic_time_step=elastic_dt,
                                                 revised_evp=revised, check=subcycle == 0)
        velocity_v842.velocity_update(mesh=mesh, icestate=icestate, solver=solver, inertia=inertia,
                                      use_ocean_stress=use_ocean_stress, revised_evp=revised,
                                      elastic_time_step=elastic_dt, dynamics_time_step=dynamics_dt,
                                      check=subcycle == 0)
    if weak:
        weak_v842.final_divergence_shear_weak(mesh=mesh, solver=solver, weak=vw, ridging=p["ridging"],
                                              use_column_package=column, corrections=weak_corrections)
        weak_v842.principal_stresses(mesh=mesh, weak=vw)
    else:
        evp_v842.final_divergence_shear_variational(mesh=mesh, solver=solver, variational=variational,
                                                    ridging=p["ridging"], use_column_package=column)
        velocity_v842.principal_stresses(mesh=mesh, variational=variational)
    velocity_v842.ocean_stress_final(mesh=mesh, boundary=p["boundary"], icestate=icestate, solver=solver,
                                     use_ocean_stress=use_ocean_stress)


def timestep_finalize(pools: dict, configs: dict) -> None:
    """``seaice_timestep_finalize``: ``seaice_column_reinitialize_fluxes``
    (the atmospheric fluxes and, with the column package, the column's
    atmospheric and oceanic fluxes). :func:`advance` runs it after every
    step unless told to leave the last step's fluxes for a coupler, which
    then runs it after reading them."""
    if configs.get("config_use_column_package", False):
        from . import column_step_v842 as cstep
        cstep.reinitialize_fluxes(pools, configs)
        return
    solver = pools["velocity_solver"]
    solver["airStressCellU"][...] = 0
    solver["airStressCellV"][...] = 0
    a = pools["atmos_coupling"]
    for name in ("atmosReferenceTemperature2m", "atmosReferenceHumidity2m", "atmosReferenceSpeed10m"):
        a[name][...] = 0


def advance(*, pools: dict, configs: dict, clock: Clock, n_steps: int, first_step: int = 1,
            finalize_last: bool = True, after_clock=None, metric_libm: str | None = None,
            weak_corrections=()) -> int:
    """``seaice_core_run``'s loop, ``n_steps`` times from step ``first_step``, in
    place on the pools and clock; returns the next step's number. With
    ``finalize_last`` false the last step is not finalized (a coupler reads
    its fluxes, then calls :func:`timestep_finalize`). ``after_clock(step)``,
    if given, runs after each clock advance and before the step (a driver's
    work there, e.g. ``seaice_init_post_clock_advance``). A column abort
    raises :class:`ColumnAbort`."""
    import cupy as cp
    check_config(configs)
    clock.check_calendar(configs)
    m, d = pools["mesh"], pools["diagnostics"]
    solver = pools["velocity_solver"]
    precision = _precision_of(solver["uVelocity"])
    real = PRECISIONS[precision]
    nc, ne, nv = int(m["nCells"]), int(m["nEdges"]), int(m["nVertices"])
    days_per_second = real(real(1.0) / real(real(24.0) * real(3600.0)))
    seconds_per_day = real(real(24.0) * real(3600.0))
    config_dt = real(configs["config_dt"])
    column = None
    if configs.get("config_use_column_package", False):
        from . import column_step_v842 as cstep
        _column_precision(pools)
        column = _column_state(pools, configs)
    consts = None
    if configs.get("config_use_velocity_solver", True):
        # seaice_init_evp, at the init's time steps
        consts = evp_v842.init_evp(m["dvEdge"][:ne], int(m["nEdgesSolve"]), solver["dynamicsTimeStep"].item(),
                                   solver["elasticTimeStep"].item())
    tangents = None
    weak = configs.get("config_use_velocity_solver", True) and _scheme(configs) == "weak"
    if weak_corrections and not weak:
        raise ValueError(f"weak_corrections {tuple(weak_corrections)} need config_stress_divergence_scheme = "
                         "'weak' with the velocity solver on")
    if weak:
        # the weak form's metric terms: tan of the init's rotated latitudes, once
        from . import weak_v842
        weak_v842._check_corrections(weak_corrections)
        sphere = bool(configs.get("mesh:on_a_sphere", configs.get("on_a_sphere", False)))
        radius = float(configs.get("mesh:sphere_radius", configs.get("sphere_radius", 0.0)))
        if sphere and not radius > 0.0:
            _refuse("mesh:sphere_radius", radius,
                    "the weak form's metric terms divide by the sphere's radius, and on a sphere it must be "
                    "given (0 would stand in 1 and scale every metric term by the radius)",
                    "mesh:sphere_radius = the sphere's radius in m")
        vw = pools["velocity_weak"]
        tangents = tuple(cp.asarray(t) for t in weak_v842.metric_tangents(
            vw["latCellRotated"], vw["latVertexRotated"], metric_libm=metric_libm))
    tree = advection_tree(pools, configs) if configs.get("config_use_advection", False) else None
    itimestep = first_step
    for i in range(n_steps):
        clock.now += clock.step
        if after_clock is not None:
            after_clock(itimestep)
        # seaice_set_time_diagnostics
        d["xtime"] = clock.time_stamp
        days = real(real(clock.now - clock.seconds(d["simulationStartTime"], "diagnostics:simulationStartTime"))
                    * days_per_second)
        d["daysSinceStartOfSim"][...] = cp.asarray(days)
        # store_timestep_counter
        m["itimestep"][0] = itimestep
        d["timeAverageTestVariable"][...] = d["timeAverageTestVariable"] + real(config_dt / seconds_per_day)
        now = format_time(clock.now, clock.calendar)
        if column is not None:
            cstep.reinitialize_diagnostics_thermodynamics(pools, configs)
            cstep.reinitialize_diagnostics_bgc(pools, configs)
            _column_abort(itimestep, cstep.predynamics(pools, configs, column, now))
        for _ in range(int(configs["config_dynamics_subcycle_number"])):
            if column is not None:
                cstep.reinitialize_diagnostics_dynamics(pools, configs)
            if configs.get("config_use_velocity_solver", True):
                with evp_v842.block_counts(cells=nc, vertices=nv):
                    _velocity_solver(pools, configs, consts, column, tangents, weak_corrections)
            if tree is not None:
                # seaice_run_advection: incremental remap at dynamicsTimeStep
                remap_v842.run(tree, pools=pools, mesh=m, configs=configs)
            if column is not None:
                _column_abort(itimestep, cstep.dynamics(pools, configs, column))
        if column is not None:
            cstep.postdynamics(pools, configs, column, now)
        if finalize_last or i < n_steps - 1:
            timestep_finalize(pools, configs)
        itimestep += 1
    return itimestep


def _column_abort(step: int, abort: tuple) -> None:
    if abort[0] > 0:
        raise ColumnAbort(step, *abort)
