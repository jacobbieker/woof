"""The two-time-level semi-Lagrangian semi-implicit step.

For each advected variable ``X``, with ``A`` an arrival point (which is a
grid point, always) and ``D`` its departure point::

    X^{n+1}_A - alpha dt (L X)^{n+1}_A = Q_X(D) + (dt/2) [2 N^n - N^{n-1}]_A
    Q_X = X^n + dt [ (1 - alpha) (L X)^n + (1/2) N^n ]

``L`` is the semi-implicit linear operator the shipped core already
carries and ``N = A - L`` the nonlinear residual of the advective-form
tendency.  This is the trapezoidal two-time-level scheme with the
arrival-side nonlinear term extrapolated to ``t^{n+1}`` (SETTLS-type) and
off-centred by ``alpha``.

For theta the material tendency ``A`` is zero (adiabatic flow), so
``N = -L theta`` and the WHOLE variable is gathered: a parcel's reference
change is then whatever the vertical stencil reads between its departure
and arrival levels, which is exactly consistent with what the same stencil
reads of the deviation from the reference.  Until 2026-09-06 the core
gathered ``theta - theta_ref(k)`` and carried the reference profile's
material tendency as a grid tendency in ``N``; :mod:`.rhs` records what
that did at the model lid and the measurement that retired it.

Pre-combining the whole departure-side bundle into ``Q_X`` before the
interpolation is what keeps this to ONE interpolated field per advected
variable: three Cartesian wind components, ``theta``, vapour, ten grid
tracers and one two-dimensional field for ``ln ps``.  Interpolating the
state and its tendencies separately would double that, and the gather is
the step's largest new cost.

The left-hand side is the shipped Helmholtz solve, unchanged.  Written
out, ``(I - tau L) y = R`` with ``tau = alpha dt`` is exactly
``VerticalModeSemiImplicit.solve_shifted``, the same routine the IMEX
integrator calls for its stage solves and with the same per-tau cached
inverse; the semi-Lagrangian core calls it ONCE per step where the IMEX
pair calls it twice.

What is NOT clipped, and why.  The quasi-monotone option bounds the
interpolated value by the values of the cell that surrounds the
departure point.  That is right for a positive-definite species: cloud
water has no business exceeding the cloud water around it.  It is wrong
for the dynamical bundle.  ``Q_u`` is a wind plus a tendency, so its
physical bounds are not its neighbours' values at all, and clipping a
wind to its neighbours removes the extremes of a field whose extremes
ARE the flow, every step, with no conservation and no measurement of
what was taken.  So the wind, ``theta'`` and ``ln ps`` are gathered
unlimited by default and vapour and the ten tracers are gathered
limited; ``[semilag] quasi_monotone_dynamics`` reaches the other arm so
the choice can be measured rather than argued.
"""
from __future__ import annotations

from .interpolate import Stencil, gather_batch
from .options import HORIZONTAL_ORDER, PHYSICS_ARRIVAL_WEIGHT
from .rhs import ReferenceProfile, advective_tendencies
from .state import TrajectoryState
from .tables import SphericalGridTables
from .tracers import DEFICIT_FIXERS, area_weights, fix_mass
from .trajectory import (
    CartesianWind,
    cartesian_wind,
    convergence,
    departure_points,
    level_rate_from_mass_flux,
    lipschitz,
    refuse_beyond_lipschitz,
)
from .vectors import transport_to_arrival

from ..constants import CONDENSATE_SPECIES, GRAVITY_M_S2, GRID_TRACERS
from ..profile import profiler_of
from ..spill import resident, spilled
from ..state import MoistHybridState

#: The integrator names this package answers to.
SEMILAG_INTEGRATORS = ("sl_si",)


def grid_tables(model) -> SphericalGridTables:
    """The grid geometry of this model's transform, built once."""
    cached = getattr(model, "_semilag_tables", None)
    backend = model.transform.backend
    if cached is None or cached.dtype != backend.float_dtype:
        cached = SphericalGridTables.create(
            model.transform.grid, xp=backend.xp, dtype=backend.float_dtype
        )
        model._semilag_tables = cached
    return cached


def reference_profile(model) -> ReferenceProfile:
    """The semi-implicit reference column, built once."""
    cached = getattr(model, "_semilag_reference", None)
    if cached is None:
        cached = ReferenceProfile(model)
        model._semilag_reference = cached
    return cached


def _surface_stencil(stencil: Stencil, tables: SphericalGridTables):
    """The horizontal departure points of the LOWEST model level, as a
    four-level stencil with no vertical displacement.

    ``ln ps`` is two-dimensional and the gather is a tricubic on a volume,
    so the surface field is read through four identical levels with the
    departure level index pinned to an exact node.  Three of the four
    vertical Lagrange weights are then exactly zero and the fourth is
    exactly one, so the result is the plain bicubic, computed by the same
    compiled kernel every other field is read by rather than by a second
    interpolation with its own bracket search and its own polar map.  It
    costs a tenth of one volume field on a forty-level stack.
    """
    xp = tables.xp
    dtype = tables.dtype
    nlev, nlat, nlon = stencil.shape
    xi = xp.ascontiguousarray(
        xp.broadcast_to(stencil.xi[nlev - 1][None], (4, nlat, nlon))
    )
    phi = xp.ascontiguousarray(
        xp.broadcast_to(stencil.phi[nlev - 1][None], (4, nlat, nlon))
    )
    level = xp.ascontiguousarray(
        xp.broadcast_to(
            xp.arange(4, dtype=dtype)[:, None, None], (4, nlat, nlon)
        )
    )
    return Stencil(xi=xi, phi=phi, level=level, tables=tables)


def _gather_surface(field, surface: Stencil, *, monotone: bool, order: int = 4):
    xp = surface.tables.xp
    nlat, nlon = surface.tables.shape
    tiled = xp.ascontiguousarray(
        xp.broadcast_to(field[None], (4, nlat, nlon))
    )
    return gather_batch([tiled], surface, monotone=monotone, batch=1,
                        order=order)[0][1]


def _physics_increment(model, atmosphere, pre, tables, *, riding: bool):
    """The FIRST physics half's increment, in grid space.

    Taken as a difference in SPECTRAL space and synthesized ONCE, rather
    than as the difference of two grid syntheses: the increment is five
    spectral fields and one synthesis, where the two-synthesis form would
    pay the 4.15 ms T255 stacked synthesis twice for the same numbers.
    The ten grid tracers are already grid fields and their increment is a
    plain subtraction with no transform at all.
    """
    transform = model.transform
    xp = transform.backend.xp
    nlev = int(model.nlev)
    d_u, d_v = model.vector.wind_from_vordiv(
        atmosphere.vorticity - pre.vorticity,
        atmosphere.divergence - pre.divergence,
    )
    stacked = xp.concatenate([
        atmosphere.theta - pre.theta, atmosphere.qv - pre.qv,
    ], axis=0)
    grid = model._chunked(transform.inverse, stacked)
    del stacked
    d_x, d_y, d_z = cartesian_wind(d_u, d_v, tables)
    return {
        "u": d_u, "v": d_v, "cartesian": (d_x, d_y, d_z),
        "theta": grid[:nlev], "qv": grid[nlev:],
        "lnps": transform.inverse(
            atmosphere.log_surface_pressure - pre.log_surface_pressure
        ),
        "tracers": _tracer_increment(atmosphere, pre) if riding else {},
    }


def _tracer_increment(atmosphere, pre) -> dict:
    """The physics half's increment of the ten grid tracers.

    REFUSED while the pinned host tier holds them: the physics half-step
    writes a parked tracer's slot in place (dynamics.apply_physics,
    band by band), so the state the half started from no longer exists
    as an array to subtract, and a difference taken against the slot
    would read zero on every tracer and couple no physics at all.  The
    shipped coupling ("advected") never takes this difference; the two
    that do run with the tier off.
    """
    parked = [name for name in GRID_TRACERS
              if spilled(getattr(atmosphere, name)) or spilled(getattr(pre, name))]
    if parked:
        raise ValueError(
            "semilag physics_coupling needs the grid tracers' physics "
            f"increment and the pinned host tier holds {', '.join(parked)}: "
            "the physics half-step writes a parked tracer's slot in place, "
            "so the pre-physics tracers no longer exist to subtract and the "
            "increment would read zero.  Run with [memory] host_spill = "
            "\"off\" or with [semilag] physics_coupling = \"advected\"."
        )
    return {
        name: getattr(atmosphere, name) - getattr(pre, name)
        for name in GRID_TRACERS
    }


def semilag_step(model, atmosphere: MoistHybridState, dt_s: float, mark=None):
    """One two-time-level semi-Lagrangian semi-implicit step of the
    adiabatic core, tracers included.

    Returns ``(advanced, metrics)`` in the shape
    :meth:`woof.globe.dynamics.MoistHybridModel.integrate_dynamics`
    contracts for, with the trajectory's own diagnostics beside the
    semi-implicit increment.
    """
    options = model.semilag
    transform = model.transform
    backend = transform.backend
    xp = backend.xp
    scalar = backend.float_dtype
    dt = float(dt_s)
    alpha = float(model.semi_implicit.off_centring_weight)
    tables = grid_tables(model)
    reference = reference_profile(model)
    previous = model.trajectory_state()
    startup = previous is None
    prof = profiler_of(model)
    # How much of the first physics half's increment is moved from the
    # departure point to the arrival point (semilag.options): zero under
    # the plain Strang split, which is the default and pays nothing at
    # all, because the increment is then never separated from the state.
    arrival_weight = float(PHYSICS_ARRIVAL_WEIGHT[options.physics_coupling])
    # The dynamical bundle's horizontal stencil width (semilag.options
    # HORIZONTAL_INTERPOLATIONS); the limited species stay on the cubic.
    order = int(HORIZONTAL_ORDER[options.horizontal_interpolation])
    pre_physics = model.take_pre_physics() if arrival_weight else None
    if arrival_weight and pre_physics is None:
        raise ValueError(
            f"semilag.physics_coupling = {options.physics_coupling!r} needs "
            "the state as it stood before the first physics half, and the "
            "step did not hand one over; a coupling that silently fell back "
            "to 'advected' would produce a different run under the same "
            "config and say nothing"
        )

    with prof.section("grid_state"):
        # One set of sources feeds both the grid view and the mass-flux
        # pass, because the band loop keys its memo and its per-band
        # syntheses off the sources object rather than off a grid dict.
        sources = model.grid_sources(atmosphere)
        g, _stack, _names = model.grid_band(sources, model.whole_rows)
        _spec, _div_mass, ps_t, omega_half = model._mass_flux_and_omega(sources)

    with prof.section("trajectory"):
        level_rate = level_rate_from_mass_flux(omega_half, g["dp"], xp=xp)
        vx, vy, vz = cartesian_wind(g["u"], g["v"], tables)
        wind = CartesianWind(vx=vx, vy=vy, vz=vz, level_rate=level_rate)
        if startup or options.extrapolation == "none":
            extrapolated = wind
        else:
            ex_u = 2.0 * g["u"] - previous.u_prev
            ex_v = 2.0 * g["v"] - previous.v_prev
            ex_x, ex_y, ex_z = cartesian_wind(ex_u, ex_v, tables)
            del ex_u, ex_v
            extrapolated = CartesianWind(
                vx=ex_x, vy=ex_y, vz=ex_z,
                level_rate=2.0 * level_rate - previous.s_prev,
            )
        deformation = lipschitz(wind, tables, dt)
        refuse_beyond_lipschitz(deformation, model.maximum_lipschitz)
        stencil, trajectory = departure_points(
            wind, tables, dt,
            iterations=int(options.trajectory_iterations),
            extrapolated=extrapolated,
        )
        del extrapolated
        convergence(trajectory, options.trajectory_convergence_cells)
        surface = _surface_stencil(stencil, tables)

    with prof.section("tendencies"):
        tendencies = advective_tendencies(
            model, atmosphere, g, omega_half, ps_t, reference
        )
        n_u = tendencies["a_u"] - tendencies["l_u"]
        n_v = tendencies["a_v"] - tendencies["l_v"]
        # Adiabatic flow: the material tendency of theta is ZERO, so its
        # nonlinear residual is minus the operator's own row and the
        # reference profile is read along the trajectory by the gather of
        # the whole variable (rhs.advective_tendencies says why the
        # reference-subtracted form was retired, with the measurement).
        n_theta = -tendencies["l_theta"]
        n_lnps = tendencies["a_lnps"] - tendencies["l_lnps"]
        half = scalar(0.5 * dt)
        lead = scalar((1.0 - alpha) * dt)
        bundle_u = g["u"] + lead * tendencies["l_u"] + half * n_u
        bundle_v = g["v"] + lead * tendencies["l_v"] + half * n_v
        bundle_theta = g["theta"] + lead * tendencies["l_theta"] + half * n_theta
        bundle_lnps = (
            g["logps"] + lead * tendencies["l_lnps"] + half * n_lnps
        )
        del tendencies
        bundle_x, bundle_y, bundle_z = cartesian_wind(
            bundle_u, bundle_v, tables
        )
        del bundle_u, bundle_v

    with prof.section("gather"):
        batch = int(options.gather_batch)
        gathered = gather_batch(
            [bundle_x, bundle_y, bundle_z, bundle_theta], stencil,
            monotone=bool(options.quasi_monotone_dynamics), batch=batch,
            order=order,
        )
        del bundle_x, bundle_y, bundle_z, bundle_theta
        departure_x, departure_y, departure_z, departure_theta = gathered
        del gathered
        # Under tracer_scheme = "flux_form" the ten tracers are left for
        # dynamics.step()'s Eulerian sweep and only vapour is gathered
        # here; the trial state carries the originals through the solve
        # by reference, exactly as it does on the IMEX path.
        riding = options.tracer_scheme == "semi_lagrangian"
        # The additive fixer puts a species' mass back where the limiter
        # took it, so it needs the limiter's own record of that: the
        # gather reports it out of arithmetic it already does, and the
        # values it returns are bit for bit the plain limited gather's.
        limiter = "quasi_monotone" if options.quasi_monotone else "none"
        want_deficit = bool(
            riding and limiter != "none"
            and options.tracer_fixer in DEFICIT_FIXERS
        )
        # The grid tracers may live in the pinned host tier (spill): the
        # gather stages a parked field one batch at a time and the copy
        # dies with the batch (interpolate.gather_batch).  MEASURED
        # 2026-09-07, T533 L40 sl_si with all three slices parked on an
        # RTX 5070 Ti: handed the slot itself, the kernel refused it by
        # type at step 1; handed all ten staged at once, the card ran out
        # at 15.96 GB inside the gather.
        # A tracer the pinned host tier holds is staged onto the card
        # here, where the gather reads it; under ``flux_form`` the ten are
        # NOT gathered and must stay parked, because they then ride through
        # the solve by reference and dynamics.step's Eulerian sweep is what
        # stages and re-parks them.  Staging them here as well would take
        # the tier off the tracers for the rest of the run.
        species = gather_batch(
            [g["qv"], *(
                (getattr(atmosphere, name) for name in GRID_TRACERS)
                if riding else ()
            )],
            stencil, monotone=limiter, batch=batch,
            deficit=want_deficit,
        )
        if want_deficit:
            departure_qv = species[0][0]
            advected = dict(zip(GRID_TRACERS,
                                (row[0] for row in species[1:])))
            deficits = dict(zip(GRID_TRACERS,
                                (row[1] for row in species[1:])))
        else:
            departure_qv = species[0]
            advected = (
                dict(zip(GRID_TRACERS, species[1:])) if riding
                else atmosphere.grid_tracers()
            )
            deficits = None
        del species
        departure_lnps = _gather_surface(
            bundle_lnps, surface, monotone=bool(options.quasi_monotone_dynamics),
            order=order,
        )
        del bundle_lnps
        arrival_u, arrival_v = transport_to_arrival(
            departure_x, departure_y, departure_z, stencil, tables
        )
        del departure_x, departure_y, departure_z

    with prof.section("physics_coupling"):
        # The first physics half ran at the grid points, which are the
        # arrival points, and its increment went into the state BEFORE the
        # gather, so under the default coupling a parcel reads it at its
        # departure point along with everything else.  The other two
        # couplings move a fraction of it to the arrival point, by taking
        # it back out where the gather put it and adding it where the
        # parcel lands:
        #
        #     X = gather(X_pre + I + ...)  +  w * (I_A - gather(I))
        #
        # with w = 1 for "arrival" and 1/2 for "trajectory_average".  What
        # it buys is that a field the physics has just created is not
        # interpolated on the step that created it; what it costs is one
        # synthesis of the increment and one more gathered field per
        # advected variable, and, for a physics SINK, that the removal
        # lands where it was not computed, which the positivity repair
        # then has to hold.  That is why the default is the plain split.
        if arrival_weight:
            inc = _physics_increment(
                model, atmosphere, pre_physics, tables, riding=riding
            )
            del pre_physics
            names = list(inc["tracers"])
            # The dynamical rows read the bundle's own stencil width and
            # the species rows the cubic, so what is taken back out is
            # what the gather put in, field by field.
            rows = gather_batch(
                [*inc["cartesian"], inc["theta"]], stencil,
                monotone=False, batch=batch, order=order,
            ) + gather_batch(
                [inc["qv"], *(inc["tracers"][name] for name in names)],
                stencil, monotone=False, batch=batch,
            )
            gathered_u, gathered_v = transport_to_arrival(
                rows[0], rows[1], rows[2], stencil, tables
            )
            weight = scalar(arrival_weight)
            arrival_u = arrival_u + weight * (inc["u"] - gathered_u)
            arrival_v = arrival_v + weight * (inc["v"] - gathered_v)
            del gathered_u, gathered_v
            departure_theta = departure_theta + weight * (
                inc["theta"] - rows[3]
            )
            departure_qv = departure_qv + weight * (inc["qv"] - rows[4])
            for index, name in enumerate(names):
                advected[name] = advected[name] + weight * (
                    inc["tracers"][name] - rows[5 + index]
                )
            departure_lnps = departure_lnps + weight * (
                inc["lnps"] - _gather_surface(
                    inc["lnps"], surface, monotone=False, order=order
                )
            )
            del inc, rows, names
    del surface, stencil

    with prof.section("assemble"):
        if startup or options.extrapolation == "none":
            arrival_u = arrival_u + half * n_u
            arrival_v = arrival_v + half * n_v
            departure_theta = departure_theta + half * n_theta
            departure_lnps = departure_lnps + half * n_lnps
        else:
            arrival_u = arrival_u + half * (2.0 * n_u - previous.n_u)
            arrival_v = arrival_v + half * (2.0 * n_v - previous.n_v)
            departure_theta = departure_theta + half * (
                2.0 * n_theta - previous.n_theta
            )
            departure_lnps = departure_lnps + half * (
                2.0 * n_lnps - previous.n_lnps
            )
        trial_zeta, trial_divergence = model.vector.vordiv_from_wind(
            arrival_u, arrival_v
        )
        del arrival_u, arrival_v
        stacked = xp.concatenate([departure_theta, departure_qv], axis=0)
        del departure_theta, departure_qv
        analysed = model._chunked(
            lambda block: transform.project(transform.forward(block)), stacked
        )
        del stacked
        nlev = int(model.nlev)
        trial = MoistHybridState(
            vorticity=trial_zeta,
            divergence=trial_divergence,
            theta=analysed[:nlev],
            log_surface_pressure=transform.project(
                transform.forward(departure_lnps)
            ),
            qv=analysed[nlev:],
            **advected,
            time_s=atmosphere.time_s,
            step=atmosphere.step,
        )
        del analysed, departure_lnps, trial_zeta, previous
        if mark is not None:
            mark("semilag_transport", trial)

    with prof.section("solve"):
        advanced = model.semi_implicit.solve_shifted(
            trial, transform, model.vertical, alpha * dt
        )
        increment = float(
            xp.max(xp.abs(advanced.divergence - trial_divergence))
        )
        del trial_divergence

    with prof.section("tracers"):
        del trial
        fixer: dict[str, float] = {}
        water_relative = 0.0
        if riding:
            after = model.grid_state(advanced, only=("dp",))
            # The mass fixer measures the advected species against the
            # species as they stood, and the water reading below reads the
            # same ones, so the pinned host tier is read ONCE here and the
            # staged copy serves both.  It is read here rather than held
            # from the gather because ten grid volumes are 1.9 GiB at T533
            # and holding them across the trajectory and the solve would
            # put them beside the step's own peak; two lines of the same
            # section is a lifetime the peak does not see.  A resident run
            # gets its own arrays back and copies nothing.
            before = {name: resident(xp, value) for name, value
                      in atmosphere.grid_tracers().items()}
            fixed, fixer = fix_mass(
                advected, before, g["dp"], after["dp"],
                transform, scheme=options.tracer_fixer, deficits=deficits,
            )
            # The advanced ten stay on the card from here to the second
            # physics half, which returns them to the tier's own slots
            # (dynamics._physics_result_to_state, and the sponge-only pass
            # on the dry route).  Parking them here as well would write
            # 1.9 GiB back at T533 and read it again a few operators
            # later, for a peak that is unchanged: the step's peak is in
            # the physics half, where one device copy of the ten exists
            # either way.
            advanced = advanced.with_grid_tracers(fixed)
            # The fixer's NET correction as a fraction of the atmosphere's
            # own water, which is the quantity the breakage is about: a
            # correction of four percent of a species whose global mass is
            # a millionth of the column water is four hundredths of a
            # millionth of the water, and the per-species relative number
            # on its own cannot say that.  The sum is SIGNED, because a
            # species the fixer fed and a species it starved are the same
            # water moving between them and not two losses; a per-species
            # magnitude belongs to the per-species row.  Number moments are
            # counts per kilogram, not water, and are not in this sum.  The
            # 1/g that turns a Pa-weighted mass into kg/m2 divides both
            # sides and is written on both so the two are the same quantity.
            cell = area_weights(transform)
            column = float(xp.sum(
                (g["qv"] + sum(before[name]
                               for name in CONDENSATE_SPECIES))
                * g["dp"] * cell
            )) / GRAVITY_M_S2
            moved = sum(
                value for name, value in fixer.items()
                if name.startswith("semilag_tracer_mass_fixer_kg_m2__")
                and name.rsplit("__", 1)[1] in CONDENSATE_SPECIES
            ) / GRAVITY_M_S2
            water_relative = abs(moved) / max(column, 1.0e-30)
            del fixed, after, cell, before
        del advected, deficits

    model.set_trajectory_state(TrajectoryState(
        u_prev=g["u"], v_prev=g["v"], s_prev=level_rate,
        n_u=n_u, n_v=n_v, n_theta=n_theta, n_lnps=n_lnps,
    ))
    del g, n_u, n_v, n_theta, n_lnps, level_rate, wind, vx, vy, vz

    if mark is not None:
        mark("semilag_implicit", advanced)
    # The largest RELATIVE correction, over species.  Named explicitly
    # rather than taken as the maximum of the fixer's whole record: that
    # record also carries the correction in kg/m2 and the clip share, and a
    # number moment's correction in counts per kilogram is a number like
    # 2.6e9, which a bare max over the values reports as a relative
    # magnitude of 2.6e9 and fails every gate that reads it.
    fixer_max = max(
        (value for name, value in fixer.items()
         if name.startswith("semilag_tracer_mass_fixer_relative__")),
        default=0.0,
    )
    # The row dynamics.step() reads where the flux-form sweep's own
    # metrics would be.  The names are kept so a receipt of either arm
    # has the same shape; ``scheme`` says which produced them and the
    # values mean what they say.  A semi-Lagrangian gather crosses the
    # cells it crosses in ONE pass -- that is the whole point -- so the
    # sub-cycle counts are one and the displacement is reported under the
    # Courant names it is the semi-Lagrangian analogue of.  There is no
    # pseudo-density here and therefore no second continuity
    # discretization to disagree with the first, so that gap is zero by
    # construction rather than by measurement.
    transport = {
        "scheme": "semi_lagrangian",
        "max_courant_x": float(trajectory.displacement_max_cells),
        "max_courant_y": float(trajectory.displacement_max_cells),
        "max_courant_z": float(trajectory.displacement_max_levels),
        "substeps_x": 1, "substeps_y": 1, "substeps_z": 1,
        "floor_clip_kg_m2": 0.0,
        "pseudo_density_mismatch_relative": 0.0,
        "pseudo_density_mismatch_mean_relative": 0.0,
        "mass_fixer_max_relative": float(fixer_max),
        **fixer,
    }
    metrics = {
        **({"tracer_transport": transport} if riding else {}),
        "semi_implicit_max_divergence_increment_s1": increment,
        "semilag_startup_step": bool(startup),
        "semilag_off_centring_weight": alpha,
        "semilag_tracer_mass_fixer_relative": float(fixer_max),
        "semilag_tracer_mass_fixer_water_relative": float(water_relative),
        "semilag_tracer_limiter": limiter,
        **fixer,
        **deformation.as_dict(),
        **trajectory.as_dict(),
    }
    return advanced, metrics


__all__ = ["SEMILAG_INTEGRATORS", "grid_tables", "reference_profile",
           "semilag_step"]
