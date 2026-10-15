# Breeze MC3E timing driver.
#
# The model is NumericalEarth.jl's examples/breeze_downscaling_era5.jl at
# commit 66112eac, line for line where it builds the nest, the land and the
# radiation (lines 62-219).  What is different, and only that:
#
#   * no plotting packages, no domain map, no ERA5 omega series (they are
#     for the figures, lines 154-169) and no output writers (lines 221-259),
#     so the timed loop is the model alone;
#   * the run length, architecture and an optional horizontal coarsening
#     are environment knobs (below);
#   * a callback stamps wall time (after a device synchronise) at the end
#     of a warm-up and at the end of the run, and a JSON summary is written.
#
# Environment:
#   BREEZE_HOURS              simulated hours (default 1)
#   BREEZE_ARCH               GPU (default) or CPU
#   BREEZE_COARSEN            integer factor dividing Nx and Ny (default 1);
#                             with BREEZE_ARCH=CPU and BREEZE_SETUP_ONLY=1
#                             this is the download-only prefetch: same lat/lon
#                             box, so the same ERA5/ETOPO/CGLS requests
#   BREEZE_SETUP_ONLY         1: build everything, take no step
#   BREEZE_WARMUP_ITERATIONS  iterations excluded from the window (default 30)
#   BREEZE_ERA5_DIR           ERA5 cache directory (default ./era5)
#   BREEZE_SUMMARY            summary JSON path (default ./breeze_mc3e_timing.json)
#   BREEZE_ALBEDO             "cgls" (default, the example's CGLS 1 km albedo
#                             from the CDS; needs the satellite-albedo licence
#                             accepted on the CDS account) or a number, used as
#                             a constant surface albedo.  Radiation's cost is
#                             the same either way; the surface energy budget
#                             is not.
#   JULIA_CUDA_HARD_MEMORY_LIMIT  the device-memory cap (CUDA.jl), e.g. 6GiB

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Breeze
using CopernicusClimateDataStore
using CloudMicrophysics
using RRTMGP
using CUDA
using Printf
using Dates: DateTime, Second

env(name, default) = get(ENV, name, default)
hours      = parse(Float64, env("BREEZE_HOURS", "1"))
coarsen    = parse(Int, env("BREEZE_COARSEN", "1"))
warmup     = parse(Int, env("BREEZE_WARMUP_ITERATIONS", "30"))
setup_only = env("BREEZE_SETUP_ONLY", "0") == "1"
arch       = env("BREEZE_ARCH", "GPU") == "CPU" ? CPU() : GPU()
era5_datadir = env("BREEZE_ERA5_DIR", "era5")
summary_path = env("BREEZE_SUMMARY", "breeze_mc3e_timing.json")
albedo_choice = env("BREEZE_ALBEDO", "cgls")
surface_albedo = albedo_choice == "cgls" ? CopernicusAlbedo() : parse(Float64, albedo_choice)

wall_start = time_ns()
Oceananigans.defaults.FloatType = Float32

# --- example lines 62-93 ---------------------------------------------------
duration = hours * 3600
start_date = DateTime(2011, 05, 20, 0)
stop_date = start_date + Second(round(Int, duration))
φ₀, λ₀ = 36.605, -97.485
Lλ, Lφ = 16.7, 15.1
Δλ = Δφ = 1/9
Nx = round(Int, Lλ / Δλ) ÷ coarsen
Ny = round(Int, Lφ / Δφ) ÷ coarsen
dates = (start_date, stop_date)
λ_west, λ_east   = λ₀ .+ [-1, 1] .* Lλ / 2
φ_south, φ_north = φ₀ .+ [-1, 1] .* Lφ / 2

z = ReferenceToStretchedDiscretization(extent = 19525.0,
                                       bias = :left,
                                       bias_edge = 0,
                                       constant_spacing = 60,
                                       constant_spacing_extent = 60,
                                       maximum_spacing = 490,
                                       stretching = LinearStretching(0.15))
Nz = length(z)

# --- example lines 103-138 -------------------------------------------------
grid = LatitudeLongitudeGrid(arch;
                             longitude = (λ_west,  λ_east),
                             latitude  = (φ_south, φ_north),
                             z = TerrainFollowingVerticalDiscretization(z),
                             size = (Nx, Ny, Nz),
                             halo = (5, 5, 5),
                             topology = (Bounded, Bounded, Bounded))

dataset = ERA5HourlyPressureLevels()
relax_width = 5
nest = nested_atmosphere_model(grid, dataset;
                               dates,
                               dir = era5_datadir,
                               terrain = ETOPO2022(),
                               terrain_blend_width = relax_width,
                               relaxation_rate = 1/300,
                               relaxation_width = relax_width,
                               momentum_advection = WENO(order = 5))
parent = nest.parent
era5_region = BoundingBox(parent.grid)

# --- example lines 178-219 -------------------------------------------------
land_grid = LatitudeLongitudeGrid(arch;
                                  longitude = (λ_west,  λ_east),
                                  latitude  = (φ_south, φ_north),
                                  size = (Nx, Ny),
                                  halo = (5, 5),
                                  topology = (Bounded, Bounded, Flat))
land = SlabLand(land_grid)
skin_temperature = Metadatum(:skin_temperature; dataset = ERA5HourlySingleLevel(),
                             date = start_date, region = era5_region, dir = era5_datadir)
set!(land.temperature, skin_temperature)
set!(land; M = 20)

radiation = RadiativeTransferModel(grid, AllSkyOptics(), nest.child.thermodynamic_constants;
                                   solar_position = ApparentSolarPosition(epoch = start_date),
                                   surface_albedo,
                                   schedule = TimeInterval(1hour))

Δt = 10
atmosphere = Simulation(nest; Δt)
model = AtmosphereLandModel(atmosphere, land; radiation)
simulation = Simulation(model; Δt, stop_time=duration)
conjure_time_step_wizard!(simulation, IterationInterval(3); cfl=0.5, max_Δt=Δt)

synchronize_device() = arch isa GPU ? CUDA.synchronize() : nothing
setup_seconds = (time_ns() - wall_start) / 1e9

# --- timing ------------------------------------------------------------------
stamps = Dict{Symbol, Tuple{Int, Float64, UInt64}}()
function stamp_warmup(sim)
    if !haskey(stamps, :warm) && sim.model.clock.iteration >= warmup
        synchronize_device()
        stamps[:warm] = (sim.model.clock.iteration, Float64(sim.model.clock.time), time_ns())
    end
    return nothing
end
add_callback!(simulation, stamp_warmup, IterationInterval(1))

function progress(sim)
    child = sim.model.atmosphere.model.child
    u, v, w = child.velocities
    @info @sprintf("iter=%5d t=%s Δt=%s max|u,v,w|=(%.1f, %.1f, %.2f) wall=%.1f s",
                   sim.model.clock.iteration, prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   (time_ns() - wall_start) / 1e9)
    return nothing
end
add_callback!(simulation, progress, IterationInterval(200))

escape_json(text) = replace(text, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n")

function json_value(value)
    value isa AbstractString && return "\"" * escape_json(value) * "\""
    value === nothing && return "null"
    value isa AbstractFloat && !isfinite(value) && return "null"
    return string(value)
end

function write_summary(fields)
    open(summary_path, "w") do io
        println(io, "{")
        entries = collect(fields)
        for (i, (key, value)) in enumerate(entries)
            println(io, "  \"", key, "\": ", json_value(value), i < length(entries) ? "," : "")
        end
        println(io, "}")
    end
end

base = Pair{String, Any}[
    "model" => "breeze",
    "case" => "mc3e",
    "arch" => string(typeof(arch)),
    "device" => arch isa GPU ? CUDA.name(CUDA.device()) : "cpu",
    "nx" => Nx, "ny" => Ny, "nz" => Nz,
    "cells" => Nx * Ny * Nz,
    "coarsen" => coarsen,
    "hours" => hours,
    "setup_seconds" => setup_seconds,
    "breeze_version" => string(pkgversion(Breeze)),
    "numericalearth_version" => string(pkgversion(NumericalEarth)),
    "oceananigans_version" => string(pkgversion(Oceananigans)),
    "memory_limit" => get(ENV, "JULIA_CUDA_HARD_MEMORY_LIMIT", "none"),
    "surface_albedo" => albedo_choice,
]

if setup_only
    write_summary(vcat(base, ["status" => "setup_only"]))
    @info "setup only: built the nest, took no step" setup_seconds
    exit(0)
end

run_start = time_ns()
run!(simulation)
synchronize_device()
run_end = time_ns()

iterations = simulation.model.clock.iteration
model_seconds = Float64(simulation.model.clock.time)
# No warm stamp means the run ended inside its warm-up: the window then
# starts at run! and includes JIT compilation, and the status says so.
steady = haskey(stamps, :warm)
warm_iteration, warm_time, warm_ns = get(stamps, :warm, (0, 0.0, run_start))
window_iterations = iterations - warm_iteration
window_wall = (run_end - warm_ns) / 1e9
window_model = model_seconds - warm_time

child = nest.child
u, v, w = child.velocities
finite = all(isfinite, Array(interior(w)))
fields = vcat(base, [
    "status" => !finite ? "nonfinite" : steady ? "ok" : "no_steady_window",
    "iterations" => iterations,
    "model_seconds" => model_seconds,
    "run_wall_seconds" => (run_end - run_start) / 1e9,
    "warmup_iterations" => warm_iteration,
    "window_iterations" => window_iterations,
    "window_model_seconds" => window_model,
    "window_wall_seconds" => window_wall,
    "max_abs_w" => Float64(maximum(abs, w)),
    "max_abs_u" => Float64(maximum(abs, u)),
])
write_summary(fields)
@info "done" iterations model_seconds window_iterations window_wall window_model
