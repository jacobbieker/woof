# One-off check: the slab-ocean configuration of NumericalEarth's published
# `examples/breeze_over_four_oceans.jl` (v0.8.2), unpatched, stepped 12 times with a progress
# line after every step. Same grid (64 × 64, 20 km × 10 km, periodic in x), θ = 250 K, SST = 290 K,
# u = 10 m/s, f-plane at 33°N, gustiness off, Δt = 5 s. It does NOT load the halo workaround in
# coupled_regional_box.jl.
#
#   OC=... JULIA_DEPOT_PATH=$OC/depot $OC/julia-1.12.7/bin/julia --project=. four_oceans_slab_check.jl

using NumericalEarth, Breeze, Oceananigans, Oceananigans.Units, Printf

grid = RectilinearGrid(CPU(); size = (64, 64), halo = (5, 5),
                       x = (-10kilometers, 10kilometers), z = (0, 10kilometers),
                       topology = (Periodic, Flat, Bounded))
reference_state = ReferenceState(grid, ThermodynamicConstants(); potential_temperature = 250)
atmos = atmosphere_simulation(grid; dynamics = AnelasticDynamics(reference_state),
                              coriolis = FPlane(latitude = 33))
θᵢ(x, z) = 250 + 0.1 * randn() * (z < 500)
set!(atmos.model, θ = θᵢ, u = 10)

sst_grid = RectilinearGrid(CPU(); size = 64, halo = 5, x = (-10kilometers, 10kilometers),
                           topology = (Periodic, Flat, Flat))
slab = SlabOcean(sst_grid, depth = 10)
set!(slab, T = 290)

interfaces = ComponentInterfaces(atmos, slab; atmosphere_ocean_fluxes = SimilarityTheoryFluxes(subgrid_velocities = nothing))
model = AtmosphereOceanModel(atmos, slab; interfaces)
simulation = Simulation(model; Δt = 5, stop_iteration = 12, verbose = false)

fluxes = model.interfaces.atmosphere_ocean_interface.fluxes
function report(sim)
    u, v, w = atmos.model.velocities
    @info @sprintf("iteration %d: max|u| = %.2f, max|w| = %.2f, SST = %.3f K, mean sensible = %.1f W m⁻²",
                   iteration(sim), maximum(abs, u), maximum(abs, w), maximum(slab.temperature),
                   sum(interior(fluxes.sensible_heat)) / 64)
end
add_callback!(simulation, report, IterationInterval(1))
run!(simulation)
