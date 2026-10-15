# ClimaAtmos.jl driver for the woof idealized-benchmark comparison.
#
# Runs one of ClimaAtmos's own idealized YAML configs (optionally with
# key=value overrides), times a block of steps after compilation/warm-up,
# and dumps the final-state fields on the native spectral-element nodes so
# the Python analysis can score them against the same analytic/reference
# solutions it uses for woof.
#
# Usage (see README.md; env.sh sets JULIA, CA_ENV, CA_REPO):
#   $JULIA --project=$CA_ENV ca_driver.jl --config <yml> --out <dir>
#       [--set key=value ...] [--warm-steps 20] [--timed-steps 200]
#       [--mode accuracy|timing] [--straka-ic] [--job-id NAME]
#   $JULIA --project=$CA_ENV ca_driver.jl --batch <file>
#       (one argument list per line, one Julia session: compile once)
#
# Modes
#   timing   : diagnostics and state output disabled; warm-up steps, then a
#              timed block of steps bracketed by device synchronisation, then
#              stop.  Reports seconds/step excluding JIT, compilation and
#              output.
#   accuracy : the same warm-up and timed block (the config's diagnostics and
#              state-save callbacks stay on; their periods in the shipped
#              configs do not fall inside the block), then integrate to t_end and
#              dump the final state (x, z, u, w, theta, and for steady-state
#              cases ClimaAtmos's own predicted u/w) as raw float64 files.
#
# --straka-ic changes DryDensityCurrentProfile's bubble centre and amplitude
# to the Straka et al. (1993) definition (T' = -15 K at z_c = 3 km, so
# theta' = T'/pi_b(z)); the rest of ClimaAtmos's profile code (theta_b = 300 K,
# x_c = 25.6 km, radii 4 km x 2 km, its pressure/Exner construction) is
# unchanged.  ClimaAtmos's stock config uses theta' = -15 K at z_c = 2 km.

redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))

function parse_args(argv)
    opts = Dict{String, Any}(
        "config" => nothing,
        "out" => nothing,
        "set" => String[],
        "warm-steps" => 20,
        "timed-steps" => 200,
        "mode" => "accuracy",
        "straka-ic" => false,
        "job-id" => nothing,
    )
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--straka-ic"
            opts["straka-ic"] = true
            i += 1
            continue
        end
        startswith(a, "--") || error("unexpected argument $a")
        key = a[3:end]
        haskey(opts, key) || error("unknown option $a")
        i + 1 <= length(argv) || error("option $a needs a value")
        val = argv[i + 1]
        if key == "set"
            push!(opts["set"], val)
        elseif key in ("warm-steps", "timed-steps")
            opts[key] = parse(Int, val)
        else
            opts[key] = val
        end
        i += 2
    end
    opts["config"] === nothing && error("--config is required")
    opts["out"] === nothing && error("--out is required")
    opts["mode"] in ("accuracy", "timing") || error("--mode accuracy|timing")
    return opts
end

import YAML
import ClimaComms
ClimaComms.@import_required_backends
const ON_CUDA = get(ENV, "CLIMACOMMS_DEVICE", "CPU") == "CUDA"
if ON_CUDA
    import CUDA
end
import ClimaAtmos as CA
import ClimaCore: Fields, Spaces, Geometry
import Random

devsync() = ON_CUDA ? CUDA.synchronize() : nothing

const STRAKA_INSTALLED = Ref(false)

# Straka et al. (1993): T' = -15 K (cos(pi L) + 1)/2 inside the ellipse of
# radii 4 km x 2 km centred 3 km above the surface, applied as
# theta' = T'/pi_b(z) on the 300 K isentropic base state.  Same structure
# as ClimaAtmos's DryDensityCurrentProfile otherwise.  Installed by method
# redefinition (the IC is broadcast on the device, so it cannot branch on a
# host flag); once installed it stays for the rest of the Julia session.
function install_straka_ic!()
    STRAKA_INSTALLED[] && return
    @eval CA.Setups function center_initial_condition(
        ::DryDensityCurrentProfile,
        local_geometry,
        params,
    )
        FT = eltype(params)
        grav = CAP.grav(params)
        cp_d = CAP.cp_d(params)
        p_0 = CAP.p_ref_theta(params)
        R_d = CAP.R_d(params)
        (; x, z) = local_geometry.coordinates
        x_c = FT(25600)
        x_r = FT(4000)
        z_c = FT(3000)
        z_r = FT(2000)
        θ_b = FT(300)
        T_c = FT(-15)
        r = sqrt(((x - x_c) / x_r)^2 + ((z - z_c) / z_r)^2)
        π_b = FT(1) - grav * z / cp_d / θ_b
        θ_p = r < FT(1) ? FT(1 / 2) * T_c * (FT(1) + cospi(r)) / π_b : FT(0)
        θ = θ_b + θ_p
        π_exn = FT(1) - grav * z / cp_d / θ
        T = π_exn * θ
        p = p_0 * π_exn^(cp_d / R_d)
        return physical_state(; T, p)
    end
    STRAKA_INSTALLED[] = true
    return
end

function parse_override(s)
    k, v = split(s, "="; limit = 2)
    parsed = YAML.load(String(v))
    return String(k) => parsed
end

function to_host_vec(f)
    return vec(Array(parent(f)))
end

function materialize_diag(name, Y, p, t)
    d = CA.Diagnostics.ALL_DIAGNOSTICS[name]
    return Base.Broadcast.materialize(d.compute(Y, p, t))
end

function dump_fields(dir, integrator, want_predicted)
    Y = integrator.u
    p = integrator.p
    t = integrator.t
    ccoord = Fields.coordinate_field(Y.c)
    fcoord = Fields.coordinate_field(Y.f)
    arrays = Dict{String, Vector{Float64}}()
    arrays["c_x"] = Float64.(to_host_vec(ccoord.x))
    arrays["c_z"] = Float64.(to_host_vec(ccoord.z))
    arrays["f_x"] = Float64.(to_host_vec(fcoord.x))
    arrays["f_z"] = Float64.(to_host_vec(fcoord.z))
    arrays["c_ua"] = Float64.(to_host_vec(materialize_diag("ua", Y, p, t)))
    arrays["c_thetaa"] = Float64.(to_host_vec(materialize_diag("thetaa", Y, p, t)))
    arrays["f_wa"] = Float64.(to_host_vec(materialize_diag("wa", Y, p, t)))
    if want_predicted
        arrays["c_uapredicted"] =
            Float64.(to_host_vec(materialize_diag("uapredicted", Y, p, t)))
        arrays["f_wapredicted"] =
            Float64.(to_host_vec(materialize_diag("wapredicted", Y, p, t)))
    end
    # Surface elevation on the native horizontal nodes.
    sfc = Fields.level(fcoord.z, Fields.half)
    arrays["sfc_x"] = Float64.(to_host_vec(Fields.level(fcoord.x, Fields.half)))
    arrays["sfc_z"] = Float64.(to_host_vec(sfc))
    meta = Dict{String, Any}()
    for (k, v) in arrays
        write(joinpath(dir, k * ".f64"), v)
        meta[k] = length(v)
    end
    return meta
end

function json_value(io, v)
    if v isa AbstractString
        print(io, '"', escape_string(v), '"')
    elseif v isa Bool
        print(io, v ? "true" : "false")
    elseif v isa Integer
        print(io, v)
    elseif v isa Real
        isfinite(v) ? print(io, Float64(v)) : print(io, "null")
    elseif v isa AbstractVector
        print(io, '[')
        for (i, x) in enumerate(v)
            i > 1 && print(io, ", ")
            json_value(io, x)
        end
        print(io, ']')
    elseif v isa AbstractDict
        print(io, '{')
        for (i, k) in enumerate(sort(collect(keys(v)); by = string))
            i > 1 && print(io, ", ")
            print(io, '"', escape_string(string(k)), "\": ")
            json_value(io, v[k])
        end
        print(io, '}')
    elseif v === nothing
        print(io, "null")
    else
        json_value(io, string(v))
    end
end

function write_json(path, d)
    open(path, "w") do io
        json_value(io, d)
        println(io)
    end
end

function run_one(OPTS)
    if OPTS["straka-ic"]
        install_straka_ic!()
    elseif STRAKA_INSTALLED[] && occursin("density_current", OPTS["config"])
        error("stock density-current run after a --straka-ic run in the same " *
              "session: put --straka-ic runs last in the batch")
    end
    # invokelatest: the IC method may have just been redefined (world age).
    return Base.invokelatest(run_one_inner, OPTS)
end

function run_one_inner(OPTS)
    T_START = time()
    Random.seed!(1234)
    config_file = abspath(OPTS["config"])
    cfgdict = YAML.load_file(config_file)
    for s in OPTS["set"]
        k, v = parse_override(s)
        cfgdict[k] = v
    end
    if OPTS["mode"] == "timing"
        cfgdict["output_default_diagnostics"] = false
        cfgdict["diagnostics"] = Any[]
        cfgdict["dt_save_state_to_disk"] = "Inf"
        cfgdict["check_steady_state"] = false
        # Timing blocks may run past the case's t_end.
        cfgdict["t_end"] = "1000days"
    end
    out_dir = abspath(OPTS["out"])
    mkpath(out_dir)
    cfgdict["output_dir"] = joinpath(out_dir, "ca_output")
    cfgdict["output_dir_style"] = "RemovePreexisting"
    job_id = something(OPTS["job-id"], splitext(basename(config_file))[1])
    
    config = CA.AtmosConfig(cfgdict; job_id, config_files = [config_file])
    pa = config.parsed_args
    t_setup0 = time()
    simulation = CA.get_simulation(config)
    (; integrator) = simulation
    devsync()
    t_setup = time() - t_setup0
    
    FT = eltype(integrator.u)
    dt = Float64(pa["dt"] isa AbstractString ? CA.time_to_seconds(pa["dt"]) : pa["dt"])
    
    # Node counts.  A periodic SE line of nelem elements with nh_poly-degree
    # GLL quadrature has nelem*nh_poly unique nodes and nelem*(nh_poly+1)
    # stored nodes (element-boundary nodes are duplicated in storage).
    cspace = axes(integrator.u.c)
    nlev = Spaces.nlevels(cspace)
    npoly = pa["nh_poly"]
    nx_unique = pa["x_elem"] * npoly
    nx_stored = pa["x_elem"] * (npoly + 1)
    ny_unique = pa["config"] == "box" ? pa["y_elem"] * npoly : 1
    ny_stored = pa["config"] == "box" ? pa["y_elem"] * (npoly + 1) : 1
    unique_points = nx_unique * ny_unique * nlev
    stored_points = nx_stored * ny_stored * nlev
    
    # Warm-up: the first step compiles the whole tendency/solver stack.
    warm = OPTS["warm-steps"]
    timed = OPTS["timed-steps"]
    t_warm0 = time()
    for _ in 1:warm
        CA.CTS.step!(integrator)
    end
    devsync()
    t_warm = time() - t_warm0
    t0 = time()
    for _ in 1:timed
        CA.CTS.step!(integrator)
    end
    devsync()
    t_timed = time() - t0
    sec_per_step = t_timed / timed
    nan_after_timed = any(isnan, parent(integrator.u))
    
    result = Dict{String, Any}(
        "model" => "ClimaAtmos.jl",
        "config_file" => config_file,
        "overrides" => OPTS["set"],
        "straka_ic" => OPTS["straka-ic"],
        "mode" => OPTS["mode"],
        "float_type" => string(FT),
        "device" => ON_CUDA ? "CUDA" : "CPU",
        "gpu_name" => ON_CUDA ? CUDA.name(CUDA.device()) : "",
        "dt" => dt,
        "x_elem" => pa["x_elem"],
        "z_elem" => pa["z_elem"],
        "nh_poly" => npoly,
        "nlev" => nlev,
        "nx_unique" => nx_unique,
        "unique_points" => unique_points,
        "stored_points" => stored_points,
        "warm_steps" => warm,
        "warm_seconds" => t_warm,
        "setup_seconds" => t_setup,
        "timed_steps" => timed,
        "timed_seconds" => t_timed,
        "sec_per_step" => sec_per_step,
        "point_steps_per_sec" => unique_points / sec_per_step,
        "stored_point_steps_per_sec" => stored_points / sec_per_step,
        "sim_seconds_per_wall_second" => dt / sec_per_step,
        "nan" => nan_after_timed,
        "julia_version" => string(VERSION),
        "climaatmos_version" => string(pkgversion(CA)),
    )
    if OPTS["mode"] == "accuracy"
        # Integrate to t_end with ClimaAtmos's own loop (callbacks, diagnostics).
        t_rest0 = time()
        sol_res = CA.solve_atmos!(simulation)
        devsync()
        result["rest_wall_seconds"] = time() - t_rest0
        result["solve_ret_code"] = string(sol_res.ret_code)
        result["t_final"] = Float64(integrator.t)
        result["nan"] = any(isnan, parent(integrator.u))
        want_predicted = get(pa, "check_steady_state", false) == true
        fields_dir = joinpath(out_dir, "fields")
        mkpath(fields_dir)
        result["fields"] = dump_fields(fields_dir, integrator, want_predicted)
    end
    result["total_wall_seconds"] = time() - T_START
    write_json(joinpath(out_dir, "result.json"), result)
    @info "ca_driver done" sec_per_step result["point_steps_per_sec"] result["nan"]
    return result
end

function batch_lines(path)
    runs = Vector{Vector{String}}()
    for line in eachline(path)
        line = strip(line)
        (isempty(line) || startswith(line, "#")) && continue
        push!(runs, String.(split(line)))
    end
    return runs
end

if length(ARGS) == 2 && ARGS[1] == "--batch"
    # One Julia session for several runs: compilation is paid once per
    # distinct model type instead of once per run.
    n_failed = 0
    for argv in batch_lines(ARGS[2])
        @info "ca_driver batch run" argv
        try
            run_one(parse_args(argv))
        catch err
            global n_failed += 1
            @error "run failed" argv exception = (err, catch_backtrace())
        end
    end
    n_failed > 0 && (@error "$n_failed batch run(s) failed"; exit(1))
else
    run_one(parse_args(ARGS))
end
