# Small atmosphere–ocean coupling experiment with NumericalEarth + Breeze + Oceananigans.
#
# Asks one question: what regional (limited-area) atmosphere–ocean coupling works today
# with the released packages, at a size that runs on a CPU in minutes?
#
# Cases (pick any subset on the command line; default = all, in this order):
#
#   periodic_slab    2-D periodic Breeze LES over a 10 m SlabOcean
#                    (breeze_over_four_oceans.jl, reduced).
#   periodic_catke   2-D periodic Breeze LES over a 50 m hydrostatic Oceananigans ocean with CATKE.
#   lam_slab         Limited-area Breeze atmosphere (NestedModel driven through open lateral
#                    boundaries by a synthetic PrescribedAtmosphere parent) over a SlabOcean.
#   lam_catke        The same limited-area atmosphere over a closed-basin (walls) hydrostatic
#                    Oceananigans ocean with CATKE vertical mixing.
#   ocean_open       Ocean only: the same hydrostatic ocean with open lateral boundaries driven by
#                    an external FieldTimeSeries (a synthetic stand-in for GLORYS/ECCO): T/S
#                    Dirichlet, baroclinic u/v PerturbationAdvection, barotropic Flather, η Chapman.
#   lam_catke_open   Limited-area atmosphere over the open-boundary ocean (the "regional
#                    atmosphere + regional ocean" configuration).
#
#   <case>_unpatched The same case built without the halo workaround below. The default list
#                    includes periodic_slab_unpatched and lam_slab_unpatched (eight cases in all);
#                    any coupled case name accepts the suffix.
#
# Every case is wrapped in try/catch: a failure is recorded (with the error and a trimmed
# stack trace) rather than aborting the remaining cases. Results go to `$OC_OUT` (default
# ./results): summary.csv plus one <case>.log per case.
#
# No downloads: every input (parent atmosphere, external ocean state) is synthetic.

using NumericalEarth
using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: ValueBoundaryCondition, NormalFlowBoundaryCondition,
                                       PerturbationAdvection,
                                       GravityWaveRadiationBoundaryCondition,
                                       SurfaceWaveRadiationBoundaryCondition
using Oceananigans.Operators: Azᶜᶜᶜ
using NumericalEarth.NestedModels: parent_boundary_conditions
using Printf

const OUT = get(ENV, "OC_OUT", joinpath(@__DIR__, "results"))
mkpath(OUT)

const ALL_CASES = ["periodic_slab_unpatched", "periodic_slab", "periodic_catke",
                   "lam_slab_unpatched", "lam_slab", "lam_catke", "ocean_open", "lam_catke_open"]
const CASES = isempty(ARGS) ? ALL_CASES : ARGS

const BreezeExt = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)

#####
##### Workaround for a halo bug found by this experiment (NumericalEarth 0.8.2 + Breeze 0.11.3)
#####
#
# `interface_kernel_parameters` computes the interface fluxes over 0:N+1 in every non-Flat
# horizontal direction (InterfaceComputations.jl:139-142), but some exchange-state fields have
# halos that nothing fills:
#   - the `SlabOcean` temperature (its halo stays at the allocation value, 0 K);
#   - the Breeze atmosphere state written by `interpolate_state!` (interior only).
# The flux solve in a halo column therefore sees a 0 K ocean and returns NaN, and
# `_assemble_net_atmosphere_fluxes!` interpolates that NaN into the first ρu face (ℑxᶠᵃᵃ at i = 1).
#   - On a periodic grid that face is prognostic, so the atmosphere is NaN after one coupled step.
#     This is measured in periodic_slab_unpatched and in four_oceans_slab_check.jl, the published
#     `breeze_over_four_oceans.jl` slab configuration.
#   - On the bounded limited-area grid that face is a boundary face, and the unpatched case runs.
#   - Hydrostatic Oceananigans oceans fill their own halos, and their unpatched cases run too.
# Upstream reports the same mechanism for `PrescribedOcean` (NumericalEarth.jl issue #754).
#
# The patch adds a more specific method for the "no altitude correction" hook that runs right
# after `interpolate_state!` (time_step_earth_system_model.jl:57-63) and fills the halos of every
# exchange-state field. It does not change any interior value.

const ESMIC = NumericalEarth.EarthSystemModels.InterfaceComputations
const HALO_FIX = Ref(true)

function ESMIC.correct_state!(::Nothing, exchanger::ESMIC.ComponentExchanger, grid)
    HALO_FIX[] || return nothing
    for field in values(exchanger.state)
        field isa Oceananigans.Fields.Field && Oceananigans.BoundaryConditions.fill_halo_regions!(field)
    end
    return nothing
end

#####
##### Shared configuration
#####

const T_ocean_K = 293.15          # initial SST (K): 5 K warmer than the near-surface air
const S_ocean   = 35.0
const ocean_depth = 100.0         # regional ocean depth (m)

# Limited-area box: 2° × 2° at 36.6°N (≈ 180 km × 220 km), 16 × 16 columns (≈ 12 km spacing).
const λ_child = (-1.0, 1.0)
const φ_child = (35.6, 37.6)
const λ_parent = (-2.0, 2.0)
const φ_parent = (34.6, 38.6)
const Nh = 16
const Nz_atmos = 16
const Nz_ocean = 10
const halo = 6   # hydrostatic ocean defaults (WENO) need at least (6, 6, 3)

# Parent (driving) atmosphere: hourly-like nodes every 10 minutes over 2 hours.
const parent_times = collect(0.0:600.0:7200.0)

ocean_T_profile(z) = (T_ocean_K - 273.15) + (z + 10) / 50 * (z < -10)   # °C, mixed 10 m, then stratified

#####
##### Bookkeeping
#####

struct Outcome
    case       :: String
    status     :: String
    iterations :: Int
    sim_time   :: Float64
    first_wall :: Float64   # wall seconds for the first coupled step (includes JIT compilation)
    rest_wall  :: Float64   # wall seconds for the remaining steps
    sst0       :: Float64   # area-mean SST at start (°C)
    sst1       :: Float64   # area-mean SST at end (°C)
    mean_Q     :: Float64   # time- and area-mean turbulent heat flux into the atmosphere (W m⁻²), sensible + latent
    budget_rel :: Float64   # relative residual of the ocean heat budget (NaN when not checked)
    note       :: String
end

failed(case, note) = Outcome(case, "FAILED", 0, NaN, NaN, NaN, NaN, NaN, NaN, NaN, note)

relabel(o::Outcome, case) = Outcome(case, o.status, o.iterations, o.sim_time, o.first_wall, o.rest_wall,
                                    o.sst0, o.sst1, o.mean_Q, o.budget_rel, o.note)

function write_summary(outcomes)
    open(joinpath(OUT, "summary.csv"), "w") do io
        println(io, "case,status,iterations,sim_time_s,first_step_wall_s,rest_wall_s,",
                    "sst_start_C,sst_end_C,mean_turbulent_heat_flux_Wm2,ocean_heat_budget_rel_residual,note")
        for o in outcomes
            @printf(io, "%s,%s,%d,%.1f,%.2f,%.2f,%.4f,%.4f,%.2f,%.2e,\"%s\"\n",
                    o.case, o.status, o.iterations, o.sim_time, o.first_wall, o.rest_wall,
                    o.sst0, o.sst1, o.mean_Q, o.budget_rel, replace(o.note, "\"" => "'"))
        end
    end
end

#####
##### Component builders
#####

function parent_atmosphere()
    grid = LatitudeLongitudeGrid(CPU(); size = (12, 12, 8),
                                 longitude = λ_parent, latitude = φ_parent,
                                 z = (0, 16000), halo = (5, 5, 5),
                                 topology = (Bounded, Bounded, Bounded))
    parent = PrescribedAtmosphere(grid, parent_times)
    set!(parent.temperature,       (λ, φ, z, t) -> 288 - 6.5e-3 * z)
    set!(parent.specific_humidity, (λ, φ, z, t) -> 0.006 * exp(-z / 2500))
    set!(parent.velocities.u,      (λ, φ, z, t) -> 8)
    set!(parent.velocities.v,      (λ, φ, z, t) -> 2)
    set!(parent.pressure,          (λ, φ, z, t) -> 1e5 * exp(-z / 8000))
    return parent
end

function limited_area_atmosphere(Δt)
    parent = parent_atmosphere()
    child_grid = LatitudeLongitudeGrid(CPU(); size = (Nh, Nh, Nz_atmos),
                                       longitude = λ_child, latitude = φ_child,
                                       z = (0, 16000), halo = (halo, halo, halo),
                                       topology = (Bounded, Bounded, Bounded))
    nest = nested_atmosphere_model(parent, child_grid;
                                   relaxation_rate = 1/300, relaxation_width = 3,
                                   base_pressure = 1e5, coriolis = nothing,
                                   terrain = nothing, parent_condensates = nothing)
    BreezeExt.initialize_nested_child!(nest, nothing, first(parent_times), ""; balancer = false)
    return Simulation(nest; Δt), child_grid
end

function slab_ocean_on(grid_horizontal)
    slab = SlabOcean(grid_horizontal, depth = 10)
    set!(slab, T = T_ocean_K)
    return slab
end

lam_surface_grid() = LatitudeLongitudeGrid(CPU(); size = (Nh, Nh), halo = (halo, halo),
                                           longitude = λ_child, latitude = φ_child,
                                           topology = (Bounded, Bounded, Flat))

lam_ocean_grid() = LatitudeLongitudeGrid(CPU(); size = (Nh, Nh, Nz_ocean), halo = (halo, halo, halo),
                                         longitude = λ_child, latitude = φ_child,
                                         z = (-ocean_depth, 0),
                                         topology = (Bounded, Bounded, Bounded))

# External ("GLORYS-like") ocean state on a larger, coarser grid surrounding the regional ocean.
function external_ocean_state()
    grid = LatitudeLongitudeGrid(CPU(); size = (8, 8, Nz_ocean), halo = (halo, halo, halo),
                                 longitude = λ_parent, latitude = φ_parent,
                                 z = (-ocean_depth, 0),
                                 topology = (Bounded, Bounded, Bounded))
    times = [0.0, 7200.0]
    T = FieldTimeSeries{Center, Center, Center}(grid, times)
    S = FieldTimeSeries{Center, Center, Center}(grid, times)
    u = FieldTimeSeries{Face,   Center, Center}(grid, times)
    v = FieldTimeSeries{Center, Face,   Center}(grid, times)
    # A 1 K north–south SST front and a 5 cm/s eastward current, slowly warming in time.
    set!(T, (λ, φ, z, t) -> ocean_T_profile(z) + 0.5 * (36.6 - φ) + 1e-4 * t)
    set!(S, (λ, φ, z, t) -> S_ocean)
    set!(u, (λ, φ, z, t) -> 0.05)
    set!(v, (λ, φ, z, t) -> 0.0)
    return (; T, S, u, v)
end

function open_boundary_conditions(grid)
    ext = external_ocean_state()
    scheme = PerturbationAdvection(inflow_timescale = 1hour, outflow_timescale = 1day)
    normal_x = (west = NormalFlowBoundaryCondition, east = NormalFlowBoundaryCondition,
                south = ValueBoundaryCondition, north = ValueBoundaryCondition)
    normal_y = (west = ValueBoundaryCondition, east = ValueBoundaryCondition,
                south = NormalFlowBoundaryCondition, north = NormalFlowBoundaryCondition)
    baroclinic = parent_boundary_conditions(grid;
                                            variables = (T = ext.T, S = ext.S, u = ext.u, v = ext.v),
                                            bc_types  = (T = ValueBoundaryCondition, S = ValueBoundaryCondition,
                                                         u = normal_x, v = normal_y),
                                            schemes   = (u = scheme, v = scheme))
    # Barotropic mode: Flather on the transports (external transport U = 0.05 m/s × depth on the
    # zonal sides, 0 on the meridional ones), Chapman on the free surface.
    Uext = 0.05 * ocean_depth
    barotropic = (U = FieldBoundaryConditions(west = GravityWaveRadiationBoundaryCondition((Uext, 0)),
                                              east = GravityWaveRadiationBoundaryCondition((Uext, 0))),
                  V = FieldBoundaryConditions(south = GravityWaveRadiationBoundaryCondition((0, 0)),
                                              north = GravityWaveRadiationBoundaryCondition((0, 0))),
                  η = FieldBoundaryConditions(west  = SurfaceWaveRadiationBoundaryCondition(),
                                              east  = SurfaceWaveRadiationBoundaryCondition(),
                                              south = SurfaceWaveRadiationBoundaryCondition(),
                                              north = SurfaceWaveRadiationBoundaryCondition()))
    return merge(baroclinic, barotropic)
end

function hydrostatic_ocean(grid, Δt; boundary_conditions = NamedTuple(), kw...)
    ocean = ocean_simulation(grid; Δt, closure = CATKEVerticalDiffusivity(),
                             boundary_conditions, warn = false, kw...)
    set!(ocean.model, T = (λ, φ, z) -> ocean_T_profile(z), S = S_ocean)
    return ocean
end

#####
##### Diagnostics
#####

# Horizontal cell areas of `grid`, cached per grid (they never change).
const AREAS = IdDict{Any, Matrix{Float64}}()
cell_areas(grid) = get!(AREAS, grid) do
    [Azᶜᶜᶜ(i, j, 1, grid) for i in 1:size(grid, 1), j in 1:size(grid, 2)]
end

# Area-weighted mean of a 2-D field (or of level k of a 3-D field) on `grid`.
function area_mean(field, grid; k = 1)
    a = Array(interior(field))[:, :, k]
    A = cell_areas(grid)
    return sum(a .* A) / sum(A)
end

sst_celsius(ocean::SlabOcean) = area_mean(ocean.temperature, ocean.grid) - 273.15
sst_celsius(ocean::Simulation) = area_mean(ocean.model.tracers.T, ocean.model.grid; k = size(ocean.model.grid, 3))

# Ocean heat content per unit area (J m⁻²) relative to 0 °C, using the coupler's ρ cₚ.
function heat_content(ocean::SlabOcean, ρc)
    return ρc * ocean.depth * area_mean_slab(ocean)
end
area_mean_slab(ocean) = area_mean(ocean.temperature, ocean.grid) - 273.15

function heat_content(ocean::Simulation, ρc)
    grid = ocean.model.grid
    T = Array(interior(ocean.model.tracers.T))
    Nx, Ny, Nz = size(T)
    Δz = [Oceananigans.zspacing(1, 1, k, grid, Center(), Center(), Center()) for k in 1:Nz]
    A  = cell_areas(grid)
    column = dropdims(sum(T .* reshape(Δz, 1, 1, Nz), dims = 3), dims = 3)
    return ρc * sum(column .* A) / sum(A)
end

# The net temperature flux (K m s⁻¹, positive upward) the coupler hands the ocean: the slab's
# flux field, or the field inside the hydrostatic ocean's top T boundary condition.
surface_temperature_flux_field(ocean::SlabOcean) = ocean.temperature_flux
surface_temperature_flux_field(ocean::Simulation) =
    NumericalEarth.Diagnostics.flux_field(ocean.model.tracers.T.boundary_conditions.top.condition)

#####
##### Driver
#####

# The Breeze model inside the atmosphere `Simulation` (a `NestedModel` forwards to its child).
atmosphere_model(atmosphere::Simulation) = atmosphere_model(atmosphere.model)
atmosphere_model(model::NestedModel) = model.child
atmosphere_model(model) = model

function run_coupled(case, atmosphere, ocean; Δt, stop_time, check_budget = true)
    model = AtmosphereOceanModel(atmosphere, ocean)
    simulation = Simulation(model; Δt, stop_time, verbose = false)

    ρc = model.interfaces.ocean_properties.reference_density * model.interfaces.ocean_properties.heat_capacity
    fluxes = model.interfaces.atmosphere_ocean_interface.fluxes
    exchange_grid = model.interfaces.exchanger.grid
    Jgrid = ocean isa SlabOcean ? ocean.grid : ocean.model.grid

    sst0 = sst_celsius(ocean)
    H0 = heat_content(ocean, ρc)

    # The flux computed at the end of step n is applied during step n + 1, so integrate the
    # flux that was current *before* each step.
    J_prev = Ref(area_mean(surface_temperature_flux_field(ocean), Jgrid))
    Q_prev = Ref(area_mean(fluxes.sensible_heat, exchange_grid) + area_mean(fluxes.latent_heat, exchange_grid))
    ∫J = Ref(0.0)
    ∫Q = Ref(0.0)
    first_nan = Ref(0)   # first iteration whose interface heat flux is not finite (0 = never)
    function accumulate!(sim)
        # Oceananigans also calls callbacks once at iteration 0 (in `initialize!`), before any
        # step; counting that call would add the first step's flux twice.
        iteration(sim) == 0 && return nothing
        ∫J[] += J_prev[] * sim.Δt
        ∫Q[] += Q_prev[] * sim.Δt
        J_prev[] = area_mean(surface_temperature_flux_field(ocean), Jgrid)
        Q_prev[] = area_mean(fluxes.sensible_heat, exchange_grid) + area_mean(fluxes.latent_heat, exchange_grid)
        first_nan[] == 0 && !isfinite(Q_prev[]) && (first_nan[] = iteration(sim))
        return nothing
    end
    add_callback!(simulation, accumulate!, IterationInterval(1))

    function progress(sim)
        @info @sprintf("[%s] iter %d, t = %s, SST = %.4f °C, Q = %.1f W m⁻²", case, iteration(sim),
                       prettytime(sim), sst_celsius(ocean), Q_prev[])
    end
    add_callback!(simulation, progress, IterationInterval(60))

    # First step separately: it carries the JIT compilation cost.
    simulation.stop_iteration = 1
    first_wall = @elapsed run!(simulation)
    simulation.stop_iteration = Inf
    rest_wall = @elapsed run!(simulation)

    sst1 = sst_celsius(ocean)
    H1 = heat_content(ocean, ρc)
    t = model.clock.time
    # ΔH = -ρc ∫ Jᵀ dt (Jᵀ positive upward, K m s⁻¹)
    expected = -ρc * ∫J[]
    budget_rel = check_budget ? abs((H1 - H0) - expected) / max(abs(expected), eps()) : NaN

    atmosphere_fields = Oceananigans.prognostic_fields(atmosphere_model(atmosphere))
    finite = all(isfinite, Array(interior(surface_temperature_flux_field(ocean)))) && isfinite(sst1) &&
             isfinite(Q_prev[]) && all(f -> all(isfinite, Array(interior(f))), values(atmosphere_fields))
    status = finite ? "OK" : "NONFINITE"
    note = @sprintf("ΔH=%.4e J/m2 vs -ρc∫Jᵀdt=%.4e J/m2", H1 - H0, expected)
    first_nan[] > 0 && (note *= @sprintf("; interface flux first non-finite after step %d", first_nan[]))
    return Outcome(case, status, iteration(simulation), t, first_wall, rest_wall,
                   sst0, sst1, ∫Q[] / t, budget_rel, note)
end

function run_ocean_only(case, ocean; stop_time)
    simulation = ocean
    simulation.stop_time = stop_time
    simulation.verbose = false
    sst0 = sst_celsius(ocean)
    simulation.stop_iteration = 1
    first_wall = @elapsed run!(simulation)
    simulation.stop_iteration = Inf
    rest_wall = @elapsed run!(simulation)
    sst1 = sst_celsius(ocean)
    u = Array(interior(ocean.model.velocities.u))
    finite = all(isfinite, u) && isfinite(sst1)
    note = @sprintf("max|u| = %.3f m/s after %s", maximum(abs, u), prettytime(ocean.model.clock.time))
    return Outcome(case, finite ? "OK" : "NONFINITE", iteration(simulation), ocean.model.clock.time,
                   first_wall, rest_wall, sst0, sst1, NaN, NaN, note)
end

#####
##### Cases
#####

function periodic_atmosphere()
    grid = RectilinearGrid(CPU(); size = (32, 32), halo = (halo, halo),
                           x = (-10kilometers, 10kilometers), z = (0, 10kilometers),
                           topology = (Periodic, Flat, Bounded))
    reference_state = ReferenceState(grid, ThermodynamicConstants(); potential_temperature = 285)
    dynamics = AnelasticDynamics(reference_state)
    atmos = atmosphere_simulation(grid; dynamics, coriolis = FPlane(latitude = 36.6))
    # Deterministic 0.1 K pseudo-random perturbation below 500 m (no RNG, so reruns repeat).
    θᵢ(x, z) = 285 + 0.1 * (2 * mod(sin(12.9898x + 78.233z) * 43758.5453, 1) - 1) * (z < 500)
    set!(atmos.model, θ = θᵢ, u = 8)
    return atmos, grid
end

function build_case(case)
    if case == "periodic_slab"
        Δt = 5.0
        atmos, grid = periodic_atmosphere()
        sst_grid = RectilinearGrid(CPU(); size = grid.Nx, halo = grid.Hx, x = (-10kilometers, 10kilometers),
                                   topology = (Periodic, Flat, Flat))
        return () -> run_coupled(case, atmos, slab_ocean_on(sst_grid); Δt, stop_time = 30minutes)

    elseif case == "periodic_catke"
        Δt = 5.0
        atmos, grid = periodic_atmosphere()
        ocean_grid = RectilinearGrid(CPU(); size = (grid.Nx, 20), halo = (grid.Hx, halo),
                                     x = (-10kilometers, 10kilometers), z = (-50, 0),
                                     topology = (Periodic, Flat, Bounded))
        ocean = ocean_simulation(ocean_grid; Δt, coriolis = FPlane(latitude = 36.6),
                                 closure = CATKEVerticalDiffusivity(),
                                 momentum_advection = nothing, tracer_advection = nothing, warn = false)
        set!(ocean.model, T = (x, z) -> ocean_T_profile(z), S = S_ocean)
        return () -> run_coupled(case, atmos, ocean; Δt, stop_time = 30minutes)

    elseif case == "lam_slab"
        Δt = 5.0
        atmos, _ = limited_area_atmosphere(Δt)
        return () -> run_coupled(case, atmos, slab_ocean_on(lam_surface_grid()); Δt, stop_time = 1hour)

    elseif case == "lam_catke"
        Δt = 5.0
        atmos, _ = limited_area_atmosphere(Δt)
        ocean = hydrostatic_ocean(lam_ocean_grid(), Δt)
        return () -> run_coupled(case, atmos, ocean; Δt, stop_time = 1hour)

    elseif case == "ocean_open"
        Δt = 5.0
        grid = lam_ocean_grid()
        ocean = hydrostatic_ocean(grid, Δt; boundary_conditions = open_boundary_conditions(grid))
        return () -> run_ocean_only(case, ocean; stop_time = 1hour)

    elseif case == "lam_catke_open"
        Δt = 5.0
        atmos, _ = limited_area_atmosphere(Δt)
        grid = lam_ocean_grid()
        ocean = hydrostatic_ocean(grid, Δt; boundary_conditions = open_boundary_conditions(grid))
        # Lateral exchange through open boundaries changes heat content, so the closed-budget
        # check does not apply.
        return () -> run_coupled(case, atmos, ocean; Δt, stop_time = 1hour, check_budget = false)
    else
        error("unknown case $case; choose from $(ALL_CASES)")
    end
end

outcomes = Outcome[]
for case in CASES
    log = joinpath(OUT, case * ".log")
    @info "=== case $case ==="
    outcome = try
        # A "<case>_unpatched" run builds <case> without the halo workaround above.
        HALO_FIX[] = !endswith(case, "_unpatched")
        build_wall = @elapsed runner = build_case(replace(case, "_unpatched" => ""))
        @info @sprintf("[%s] built in %.1f s", case, build_wall)
        relabel(runner(), case)
    catch err
        open(log, "w") do io
            showerror(io, err)
            println(io)
            bt = catch_backtrace()
            frames = stacktrace(bt)
            for f in frames[1:min(end, 40)]
                println(io, "  ", f)
            end
        end
        msg = first(split(sprint(showerror, err), '\n'))
        @error "[$case] FAILED: $msg (full trace in $log)"
        failed(case, first(msg, 300))
    end
    if outcome.status != "FAILED"
        open(log, "w") do io
            show(io, MIME"text/plain"(), outcome)
            println(io)
        end
    end
    push!(outcomes, outcome)
    write_summary(outcomes)
end

println()
println(read(joinpath(OUT, "summary.csv"), String))
