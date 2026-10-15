#####
##### Breeze side of the woof/Breeze dry-CBL comparison (FastEddy SWiFT case).
#####
##### Pinned: Breeze v0.12.0, Oceananigans 0.113.x, CUDA.jl 6.x, Julia 1.12.7,
##### resolved exactly by the Manifest.toml beside this script's Project.toml
##### (tools/compare/breeze_cbl/julia/).  The case is Breeze's own
##### benchmarking/src/convective_boundary_layer.jl at tag v0.12.0
##### (commit 740c6938), re-expressed here so the closure, advection, drag form
##### and acoustic substepping can be matched to woof; with
#####   --dynamics=anelastic|compressible_splitexplicit --advection=WENO5
#####   --closure=none --drag=ustar --substeps=12 --distribution=proportional
#####   --forward-weight=0.65
##### it is the same model the BreezeBenchmarks CI times.
#####
##### Case constants mirror tools/compare/breeze_cbl/case.py (marked
##### "case.py:NAME"); analyze.py cross-checks the recorded values.
#####
##### Modes (same contract as run_woof.py):
#####   timing    warm-up steps excluded (JIT compilation happens there), then
#####             --reps blocks of --steps time_step! calls, each bracketed by
#####             CUDA.synchronize().  No output, no Simulation object.
#####   simulate  --hours of model time with slab-mean profiles every
#####             --sample-every seconds -> STEM.json + STEM_profiles.json.
#####
##### Usage: julia --project=tools/compare/breeze_cbl/julia run_breeze.jl \
#####            --mode timing --size 256x256x128 --dynamics anelastic --out STEM
#####

using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: SmagorinskyLilly
using Oceananigans.Advection: FluxFormAdvection
using Breeze
using Breeze: CompressibleDynamics, SplitExplicitTimeDiscretization,
              MonolithicFirstStage, ProportionalSubsteps
using CUDA
using JSON
using Statistics: mean, median
using Random

# ---------------------------------------------------------------- arguments

function parse_args(argv)
    a = Dict{String,String}(
        "mode" => "timing", "size" => "128x128x64", "dynamics" => "anelastic",
        "advection" => "WENO5", "closure" => "none", "drag" => "bulk",
        "substeps" => "12", "distribution" => "wrf", "forward-weight" => "0.55",
        "dt" => "0.5", "warmup" => "20", "steps" => "100", "reps" => "3",
        "hours" => "2.0", "sample-every" => "300", "seed" => "0",
        "label" => "", "out" => "", "arch" => "gpu")
    i = 1
    while i <= length(argv)
        arg = argv[i]
        startswith(arg, "--") || error("unexpected argument $arg")
        key = arg[3:end]
        if occursin('=', key)
            key, val = split(key, '='; limit=2)
        else
            i += 1
            i <= length(argv) || error("missing value for --$key")
            val = argv[i]
        end
        haskey(a, key) || error("unknown option --$key")
        a[String(key)] = String(val)
        i += 1
    end
    isempty(a["out"]) && error("--out STEM is required")
    return a
end

# ------------------------------------------------------------------- case

const LX = 12_000.0           # case.py:LX
const LZ = 3_000.0            # case.py:LZ
const UG = 9.0                # case.py:UG
const VG = 0.0                # case.py:VG
const LATITUDE = 33.5         # case.py:LATITUDE
const OMEGA = 7.2921e-5       # case.py:OMEGA
const THETA_SURFACE = 309.0   # case.py:THETA_SURFACE
const Z_INVERSION = 600.0     # case.py:Z_INVERSION
const LAPSE = 0.004           # case.py:LAPSE
const P_SURFACE = 101_325.0   # case.py:P_SURFACE
const HEAT_FLUX = 0.35        # case.py:HEAT_FLUX
const PERT_AMPLITUDE = 0.25   # case.py:PERT_AMPLITUDE
const PERT_DEPTH = 400.0      # case.py:PERT_DEPTH
const Z0 = 0.05               # case.py:Z0
const KARMAN = 0.4            # case.py:KARMAN
const USTAR_BREEZE = 0.4      # case.py:USTAR_BREEZE
const C_S = 0.18              # case.py:C_S
const PRANDTL = 1 / 3         # case.py:PRANDTL

coriolis_f() = 2 * OMEGA * sind(LATITUDE)
drag_coefficient(Nz) = (KARMAN / log(0.5 * LZ / Nz / Z0))^2    # case.py:drag_coefficient
θ̄(z) = THETA_SURFACE + max(0, z - Z_INVERSION) * LAPSE

function make_advection(name, FT)
    name == "WENO5" && return WENO(FT; order=5)
    name == "WENO9" && return WENO(FT; order=9)
    # WRF/woof's operator: 5th-order upwind horizontal, 3rd-order upwind vertical.
    name == "Upwind53" && return FluxFormAdvection(UpwindBiased(FT; order=5),
                                                   UpwindBiased(FT; order=5),
                                                   UpwindBiased(FT; order=3))
    name == "Upwind5" && return UpwindBiased(FT; order=5)
    name == "Centered4" && return Centered(FT; order=4)
    name == "Centered2" && return Centered(FT; order=2)
    error("unknown advection $name (WENO5, WENO9, Upwind53, Upwind5, Centered4, Centered2)")
end

function make_closure(name, FT)
    name == "none" && return nothing
    # WRF km_opt=3 (woof smag2d.cu): K_m = (c_s l)^2 sqrt(max(0, def2 - N^2/Pr)),
    # K_h = K_m/Pr, l = (dx dy dz)^(1/3), Pr = 1/3, def2 = 2 S_ij S_ij.
    # Oceananigans Lilly: nu = (C l)^2 sqrt(2 Sigma^2) sqrt(1 - Cb N^2/Sigma^2)
    # = (C l)^2 sqrt(2 Sigma^2 - 2 Cb N^2) with Sigma^2 = S_ij S_ij
    # (Smagorinskys/smagorinsky.jl:103, lilly_coefficient.jl:131), so the
    # WRF form is C = c_s, Cb = 1/(2 Pr) = 1.5.
    name == "Smagorinsky" && return SmagorinskyLilly(FT; C=C_S, Cb=1 / (2 * PRANDTL), Pr=PRANDTL)
    name == "SmagorinskyDefault" && return SmagorinskyLilly(FT)
    error("unknown closure $name (none, Smagorinsky, SmagorinskyDefault)")
end

function make_dynamics(a, FT)
    name = a["dynamics"]
    name == "anelastic" && return nothing
    if name == "compressible_splitexplicit"
        distribution = a["distribution"] == "wrf" ? MonolithicFirstStage() :
                       a["distribution"] == "proportional" ? ProportionalSubsteps() :
                       error("--distribution must be wrf or proportional")
        td = SplitExplicitTimeDiscretization(FT; substeps=parse(Int, a["substeps"]),
                                             substep_distribution=distribution,
                                             forward_weight=FT(parse(Float64, a["forward-weight"])))
        return CompressibleDynamics(td)
    end
    error("unknown dynamics $name (anelastic, compressible_splitexplicit)")
end

@inline ρu_drag_ustar(x, y, t, ρu, ρv, p) = -p.ρ₀ * p.u★^2 * ρu / sqrt(ρu^2 + ρv^2 + p.ϵ)
@inline ρv_drag_ustar(x, y, t, ρu, ρv, p) = -p.ρ₀ * p.u★^2 * ρv / sqrt(ρu^2 + ρv^2 + p.ϵ)
@inline ρu_drag_bulk(x, y, t, ρu, ρv, p) = -p.Cd * sqrt(ρu^2 + ρv^2 + p.ϵ) * ρu / p.ρ₀
@inline ρv_drag_bulk(x, y, t, ρu, ρv, p) = -p.Cd * sqrt(ρu^2 + ρv^2 + p.ϵ) * ρv / p.ρ₀

on_gpu(a) = a["arch"] == "gpu"
sync(a) = on_gpu(a) ? CUDA.synchronize() : nothing

function build_model(a; FT=Float32)
    arch = on_gpu(a) ? GPU() : a["arch"] == "cpu" ? CPU() : error("--arch must be gpu or cpu")
    Oceananigans.defaults.FloatType = FT
    Nx, Ny, Nz = parse.(Int, split(lowercase(a["size"]), 'x'))
    grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), x=(0, LX), y=(0, LX), z=(0, LZ),
                           halo=(5, 5, 5), topology=(Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants()

    dynamics = make_dynamics(a, FT)
    if isnothing(dynamics)
        reference_state = ReferenceState(grid, constants; base_pressure=P_SURFACE,
                                         potential_temperature=THETA_SURFACE)
        dynamics = AnelasticDynamics(reference_state)
    else
        dynamics = CompressibleDynamics(dynamics.time_discretization;
                                        base_pressure=P_SURFACE,
                                        reference_potential_temperature=θ̄)
    end

    coriolis = FPlane(; f=coriolis_f())
    q₀ = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
    ρ₀ = Breeze.Thermodynamics.density(FT(THETA_SURFACE), FT(P_SURFACE), q₀, constants)
    ρθ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρ₀ * FT(HEAT_FLUX)))

    ϵ = FT(1e-10)
    if a["drag"] == "ustar"           # Breeze benchmark: constant u* along the wind
        params = (; ρ₀, u★=FT(USTAR_BREEZE), ϵ)
        fu, fv = ρu_drag_ustar, ρv_drag_ustar
    elseif a["drag"] == "bulk"        # woof/WRF isfflx=0: tau = -rho Cd |U| U
        params = (; ρ₀, Cd=FT(drag_coefficient(Nz)), ϵ)
        fu, fv = ρu_drag_bulk, ρv_drag_bulk
    else
        error("--drag must be bulk or ustar")
    end
    ρu_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(fu; field_dependencies=(:ρu, :ρv), parameters=params))
    ρv_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(fv; field_dependencies=(:ρu, :ρv), parameters=params))

    geostrophic = geostrophic_forcings(z -> UG, z -> VG)
    model = AtmosphereModel(grid; dynamics, coriolis,
                            advection=make_advection(a["advection"], FT),
                            closure=make_closure(a["closure"], FT),
                            forcing=(; u=geostrophic.u, v=geostrophic.v),
                            boundary_conditions=(ρθ=ρθ_bcs, ρu=ρu_bcs, ρv=ρv_bcs))

    Random.seed!(parse(Int, a["seed"]))
    θᵢ(x, y, z) = θ̄(z) + PERT_AMPLITUDE * (2 * rand() - 1) * (z < PERT_DEPTH)
    if model.dynamics isa CompressibleDynamics
        # as Breeze's case: rescale the hydrostatic reference density by θ̄/θ so
        # ρθ (hence pressure) matches the reference and no acoustic pulse is seeded
        θ = CenterField(grid); set!(θ, θᵢ)
        θb = CenterField(grid); set!(θb, (x, y, z) -> θ̄(z))
        ρ = CenterField(grid)
        set!(ρ, model.dynamics.reference_state.density * θb / θ)
        set!(model, θ=θ, u=UG, v=VG, ρ=ρ)
    else
        set!(model, θ=θᵢ, u=UG, v=VG)
    end
    return model, (Nx, Ny, Nz)
end

# --------------------------------------------------------------- profiles

slab(a) = vec(mean(a; dims=(1, 2)))

function profiles(model, Nz)
    u, v, w = model.velocities
    θ = liquid_ice_potential_temperature(model)
    U = Array{Float64}(interior(u)); V = Array{Float64}(interior(v))
    W = Array{Float64}(interior(w))           # Nz+1 faces (Bounded)
    Θ = Array{Float64}(interior(θ))
    ub, vb, wb, θb = slab(U), slab(V), slab(W), slab(Θ)
    Wp = W .- reshape(wb, 1, 1, :)
    ww = slab(Wp .^ 2)
    Wc = 0.5 .* (W[:, :, 1:end-1] .+ W[:, :, 2:end])
    Wcp = Wc .- reshape(slab(Wc), 1, 1, :)
    Θp = Θ .- reshape(θb, 1, 1, :)
    wth_res = slab(Wcp .* Θp)
    Up = U .- reshape(ub, 1, 1, :); Vp = V .- reshape(vb, 1, 1, :)
    tke_res = 0.5 .* (slab(Up .^ 2) .+ slab(Vp .^ 2) .+ slab(Wcp .^ 2))
    wth_sgs = zeros(Nz + 1)                   # interior faces only, as woof
    if !isnothing(model.closure)
        νₑ = Array{Float64}(interior(model.closure_fields.νₑ))
        Pr = PRANDTL
        if model.closure.Pr isa Number
            Pr = Float64(model.closure.Pr)
        elseif model.closure.Pr isa NamedTuple && !isempty(model.closure.Pr)
            Pr = Float64(first(values(model.closure.Pr)))
        end
        Δz = LZ / Nz
        for k in 2:Nz
            κ = 0.5 .* (νₑ[:, :, k-1] .+ νₑ[:, :, k]) ./ Pr
            wth_sgs[k] = mean(-κ .* (Θ[:, :, k] .- Θ[:, :, k-1]) ./ Δz)
        end
    end
    return Dict("theta" => θb, "u" => ub, "v" => vb, "ww" => ww,
                "wth_res" => wth_res, "wth_sgs" => wth_sgs, "tke_res" => tke_res,
                "e_sgs" => zeros(Nz), "w_max" => maximum(abs, W),
                "finite" => all(isfinite, Θ) && all(isfinite, W))
end

# ------------------------------------------------------------------- meta

function meta(a, dims, model)
    Nx, Ny, Nz = dims
    dyn = a["dynamics"]
    ns = dyn == "anelastic" ? nothing : parse(Int, a["substeps"])
    acoustic = isnothing(ns) ? nothing :
               a["distribution"] == "wrf" ? 1 + ns ÷ 2 + ns :
               cld(ns, 3) + cld(ns, 2) + ns
    return Dict(
        "model" => "breeze", "label" => isempty(a["label"]) ? "breeze_$(dyn)_$(a["advection"])_$(a["closure"])" : a["label"],
        "dynamics" => dyn, "advection" => a["advection"], "closure" => a["closure"],
        "nx" => Nx, "ny" => Ny, "nz" => Nz, "dx" => LX / Nx, "dz" => LZ / Nz,
        "dt" => parse(Float64, a["dt"]),
        "substeps" => ns, "substep_distribution" => isnothing(ns) ? nothing : a["distribution"],
        "acoustic_steps_per_dt" => acoustic,
        "forward_weight" => isnothing(ns) ? nothing : parse(Float64, a["forward-weight"]),
        "drag" => a["drag"],
        "drag_coefficient" => a["drag"] == "bulk" ? drag_coefficient(Nz) : nothing,
        "heat_flux" => HEAT_FLUX, "f" => coriolis_f(), "ug" => UG, "vg" => VG,
        "p_surf" => P_SURFACE, "float_type" => "Float32", "seed" => parse(Int, a["seed"]),
        "gpu" => on_gpu(a) ? CUDA.name(CUDA.device()) : "cpu", "julia" => string(VERSION),
        "breeze" => string(pkgversion(Breeze)), "oceananigans" => string(pkgversion(Oceananigans)),
        "cuda_jl" => string(pkgversion(CUDA)), "host" => gethostname(),
        "julia_cuda_hard_memory_limit" => get(ENV, "JULIA_CUDA_HARD_MEMORY_LIMIT", nothing))
end

# ------------------------------------------------------------------- modes

function run_timing(a)
    t0 = time_ns()
    model, dims = build_model(a)
    sync(a)
    build_s = (time_ns() - t0) / 1e9
    Δt = Float32(parse(Float64, a["dt"]))
    nwarm, nsteps, nreps = parse(Int, a["warmup"]), parse(Int, a["steps"]), parse(Int, a["reps"])
    t0 = time_ns()
    for _ in 1:nwarm
        time_step!(model, Δt)
    end
    sync(a)
    warm_s = (time_ns() - t0) / 1e9
    per_step = Float64[]
    for _ in 1:nreps
        t0 = time_ns()
        for _ in 1:nsteps
            time_step!(model, Δt)
        end
        sync(a)
        push!(per_step, (time_ns() - t0) / 1e9 / nsteps)
    end
    w = Array(interior(model.velocities.w))
    med = median(per_step)
    npts = prod(dims)
    out = merge(meta(a, dims, model), Dict(
        "mode" => "timing", "warmup_steps" => nwarm, "steps_per_rep" => nsteps,
        "reps" => nreps, "build_seconds" => build_s, "warmup_seconds" => warm_s,
        "seconds_per_step" => per_step, "median_seconds_per_step" => med,
        "mpts_steps_per_second" => npts / med / 1e6,
        "simulated_seconds_per_wall_second" => Δt / med,
        "finite" => all(isfinite, w), "w_max" => Float64(maximum(abs, w)),
        "memory_gib" => on_gpu(a) ? CUDA.used_memory() / 2^30 : nothing,
        "memory_counter" => "CUDA.used_memory (live bytes in the CUDA.jl pool)"))
    open(a["out"] * ".json", "w") do io
        JSON.print(io, out, 1)
    end
    println(JSON.json(Dict(k => out[k] for k in ("label", "nx", "ny", "nz",
        "median_seconds_per_step", "mpts_steps_per_second", "finite"))))
    return out["finite"]
end

function run_simulate(a)
    model, dims = build_model(a)
    Nz = dims[3]
    Δt = Float32(parse(Float64, a["dt"]))
    ntotal = round(Int, parse(Float64, a["hours"]) * 3600 / Δt)
    nchunk = max(round(Int, parse(Float64, a["sample-every"]) / Δt), 1)
    z_f = collect(Float64, znodes(model.grid, Face()))
    samples = [merge(profiles(model, Nz), Dict("t" => 0.0))]
    done = 0
    step_wall = 0.0
    finite = true
    w_max = 0.0
    while done < ntotal
        n = min(nchunk, ntotal - done)
        sync(a)
        t0 = time_ns()
        for _ in 1:n
            time_step!(model, Δt)
        end
        sync(a)
        step_wall += (time_ns() - t0) / 1e9
        done += n
        p = profiles(model, Nz)
        w_max = max(w_max, p["w_max"])
        if !p["finite"]
            finite = false
            println("non-finite state at t=$(done * Δt)s")
            break
        end
        push!(samples, merge(p, Dict("t" => Float64(done * Δt))))
        println("  breeze t=$(round(done * Δt; digits=0))s w_max=$(round(p["w_max"]; digits=2)) wall=$(round(step_wall; digits=1))s")
        flush(stdout)
    end
    keys_ = ("theta", "u", "v", "ww", "wth_res", "wth_sgs", "tke_res", "e_sgs")
    arrays = Dict{String,Any}(k => [s[k] for s in samples] for k in keys_)
    arrays["t_seconds"] = [s["t"] for s in samples]
    arrays["z_f"] = [z_f for _ in samples]
    open(a["out"] * "_profiles.json", "w") do io
        JSON.print(io, arrays)
    end
    out = merge(meta(a, dims, model), Dict(
        "mode" => "simulate", "hours" => parse(Float64, a["hours"]),
        "sample_every_s" => parse(Float64, a["sample-every"]), "steps" => done,
        "stepping_wall_seconds" => step_wall,
        "note_wall" => "stepping wall time includes JIT compilation of the first steps; use timing mode for throughput",
        "finite" => finite, "w_max" => w_max, "memory_gib" => on_gpu(a) ? CUDA.used_memory() / 2^30 : nothing,
        "memory_counter" => "CUDA.used_memory (live bytes in the CUDA.jl pool)"))
    open(a["out"] * ".json", "w") do io
        JSON.print(io, out, 1)
    end
    return finite
end

function main(argv)
    a = parse_args(argv)
    mkpath(dirname(abspath(a["out"])))
    ok = a["mode"] == "timing" ? run_timing(a) :
         a["mode"] == "simulate" ? run_simulate(a) :
         error("--mode must be timing or simulate")
    exit(ok ? 0 : 1)
end

main(ARGS)
