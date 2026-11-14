"""MPAS-Seaice v8.4.2: the forward model's time step, dynamics only, on the GPU.

:func:`initialize` and :func:`advance` are what ``seaice_core_init`` and
``seaice_core_run`` (``mpas_seaice_core.F``) do to the state when the column
package, advection and forcing are off:

- init: the velocity solver's time steps (``dynamicsTimeStep = config_dt /
  config_dynamics_subcycle_number``, ``elasticTimeStep = dynamicsTimeStep /
  config_elastic_subcycle_number``) and ``seaice_initialize_time_diagnostics``
  (``xtime`` and, on a cold start, ``simulationStartTime``). The variational
  basis, masks and the rest of the mesh setup are the input's;
- each step: the clock advanced, then ``seaice_timestep``:
  ``seaice_set_time_diagnostics`` (``xtime``, ``daysSinceStartOfSim``),
  ``store_timestep_counter`` (``itimestep``, ``timeAverageTestVariable``), and
  ``config_dynamics_subcycle_number`` times ``seaice_run_velocity_solver``
  (:mod:`~woof.hex.mpas_seaice.presubcycle_v842`'s pre-subcycle,
  ``config_elastic_subcycle_number`` elastic subcycles of
  :mod:`~woof.hex.mpas_seaice.evp_v842` and
  :mod:`~woof.hex.mpas_seaice.velocity_v842`, and the post-subcycle); then
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
``tests/test_mpas_seaice_timestep_parity.py``).

Refused by name: the column package, advection, forcing, the weak-form
stress divergence, the unit tests, a restart start, and halo exchanges.
``seaice_check_state`` changes no state and is not run.
"""
from __future__ import annotations

import numpy as np

from woof.hex.errors import ConfigurationRefusal
from woof.hex.mpas_ocean.forward_v842 import Clock, parse_time, read_restart, write_restart

from . import evp_v842, presubcycle_v842, velocity_v842

__all__ = ("Clock", "initialize", "advance", "write_restart", "read_restart", "clock_from_configs")
PRECISIONS = {"r4": np.float32, "r8": np.float64}


def _precision_of(array) -> str:
    for precision, dtype in PRECISIONS.items():
        if array.dtype == dtype:
            return precision
    raise ValueError(f"MPAS-Seaice arrays are float32 or float64, not {array.dtype}")


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


def check_config(configs: dict) -> None:
    """Refuse, by name, what the dynamics-only step does not carry."""
    for knob, reason in (
            ("config_use_column_package", "the column package (colpkg: thermodynamics, ITD, ridging, shortwave) "
                                          "is not ported"),
            ("config_use_advection", "sea-ice advection (incremental remap, upwind) is not ported"),
            ("config_use_forcing", "the sea-ice forcing streams are not ported; the forcing fields are held"),
            ("config_perform_unit_test", "the unit tests are not ported"),
            ("config_do_restart", "this port restarts from its own restart (read_restart), not MPAS's stream"),
            ("config_use_halo_exch", "halo exchanges: this port runs one block")):
        if configs.get(knob, False):
            _refuse(knob, True, reason, f"{knob} = .false.")
    if not configs.get("config_use_velocity_solver", True):
        return
    scheme = configs.get("config_stress_divergence_scheme", "variational")
    if scheme != "variational":
        _refuse("config_stress_divergence_scheme", scheme, "only the variational stress divergence is ported",
                "config_stress_divergence_scheme = 'variational'")


def clock_from_configs(configs: dict) -> Clock:
    """The clock: ``config_start_time``, the step ``config_dt`` (whole seconds)."""
    dt = float(configs["config_dt"])
    if dt != int(dt):
        _refuse("config_dt", dt, "the clock runs on whole seconds", "config_dt a whole number of seconds")
    start = parse_time(configs["config_start_time"])
    return Clock(start, start, int(dt))


def _tilt(configs: dict) -> str:
    if not configs.get("config_use_surface_tilt", False):
        return "none"
    return "geostrophic" if configs.get("config_geostrophic_surface_tilt", False) else "ssh_gradient"


def initialize(*, pools: dict, configs: dict) -> Clock:
    """The init's time steps and time diagnostics; the clock at its start."""
    check_config(configs)
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


def _cut(pool: dict, sizes: dict) -> dict:
    """The pool with every cell, edge and vertex array cut to the block's count."""
    out = {}
    for name, value in pool.items():
        on_device = value is not None and hasattr(value, "shape") and not isinstance(value, np.ndarray)
        n = sizes.get(int(value.shape[0])) if on_device and value.ndim else None
        out[name] = value[:n] if n is not None else value
    return out


def _velocity_solver(pools: dict, configs: dict, consts, sizes: dict) -> None:
    """``seaice_run_velocity_solver``: pre-subcycle, elastic subcycles, post-subcycle."""
    p = {name: _cut(pool, sizes) for name, pool in pools.items()}
    mesh = {**p["mesh"]}
    solver, variational, icestate = p["velocity_solver"], p["velocity_variational"], p["icestate"]
    flags = {"use_column_package": False, "use_column_vertical_thermodynamics": False}
    use_ocean_stress = bool(configs.get("config_use_ocean_stress", False))
    revised = bool(configs.get("config_revised_evp", False))
    dynamics_dt = solver["dynamicsTimeStep"].item()
    elastic_dt = solver["elasticTimeStep"].item()
    real = PRECISIONS[_precision_of(solver["uVelocity"])]
    dynamics_dt, elastic_dt = real(dynamics_dt), real(elastic_dt)
    presubcycle_v842.pre_subcycle(
        mesh=mesh, boundary=p["boundary"], tracers=p["tracers"], aggregate=p["tracers_aggregate"],
        icestate=icestate, ocean=p["ocean_coupling"], atmos=p["atmos_coupling"], solver=solver,
        variational=variational, use_air_stress=bool(configs.get("config_use_air_stress", False)),
        use_ocean_stress=use_ocean_stress, tilt=_tilt(configs), **flags)
    inertia = consts[1:2]
    for _ in range(int(configs["config_elastic_subcycle_number"])):
        evp_v842.internal_stress_variational(mesh=mesh, solver=solver, variational=variational, consts=consts,
                                             elastic_time_step=elastic_dt, revised_evp=revised)
        velocity_v842.velocity_update(mesh=mesh, icestate=icestate, solver=solver, inertia=inertia,
                                      use_ocean_stress=use_ocean_stress, revised_evp=revised,
                                      elastic_time_step=elastic_dt, dynamics_time_step=dynamics_dt)
    evp_v842.final_divergence_shear_variational(mesh=mesh, solver=solver, variational=variational,
                                                ridging=p["ridging"], use_column_package=False)
    velocity_v842.principal_stresses(mesh=mesh, variational=variational)
    velocity_v842.ocean_stress_final(mesh=mesh, boundary=p["boundary"], icestate=icestate, solver=solver,
                                     use_ocean_stress=use_ocean_stress)


def advance(*, pools: dict, configs: dict, clock: Clock, n_steps: int, first_step: int = 1) -> int:
    """``seaice_core_run``'s loop, ``n_steps`` times from step ``first_step``, in
    place on the pools and clock; returns the next step's number."""
    import cupy as cp
    check_config(configs)
    m, d = pools["mesh"], pools["diagnostics"]
    solver = pools["velocity_solver"]
    precision = _precision_of(solver["uVelocity"])
    real = PRECISIONS[precision]
    nc, ne, nv = int(m["nCells"]), int(m["nEdges"]), int(m["nVertices"])
    sizes = {nc + 1: nc, ne + 1: ne, nv + 1: nv}
    days_per_second = real(real(1.0) / real(real(24.0) * real(3600.0)))
    seconds_per_day = real(real(24.0) * real(3600.0))
    config_dt = real(configs["config_dt"])
    consts = None
    if configs.get("config_use_velocity_solver", True):
        # seaice_init_evp, at the init's time steps
        consts = evp_v842.init_evp(m["dvEdge"][:ne], int(m["nEdgesSolve"]), solver["dynamicsTimeStep"].item(),
                                   solver["elasticTimeStep"].item())
    itimestep = first_step
    for _ in range(n_steps):
        clock.now += clock.step
        # seaice_set_time_diagnostics
        d["xtime"] = clock.time_stamp
        days = real(real(clock.now - parse_time(d["simulationStartTime"])) * days_per_second)
        d["daysSinceStartOfSim"][...] = cp.asarray(days)
        # store_timestep_counter
        m["itimestep"][0] = itimestep
        d["timeAverageTestVariable"][...] = d["timeAverageTestVariable"] + real(config_dt / seconds_per_day)
        for _ in range(int(configs["config_dynamics_subcycle_number"])):
            if configs.get("config_use_velocity_solver", True):
                _velocity_solver(pools, configs, consts, sizes)
        # seaice_timestep_finalize: seaice_column_reinitialize_atmospheric_fluxes
        solver["airStressCellU"][...] = 0
        solver["airStressCellV"][...] = 0
        a = pools["atmos_coupling"]
        for name in ("atmosReferenceTemperature2m", "atmosReferenceHumidity2m", "atmosReferenceSpeed10m"):
            a[name][...] = 0
        itimestep += 1
    return itimestep

