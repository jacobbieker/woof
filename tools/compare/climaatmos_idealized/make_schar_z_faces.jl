# Write the face heights of ClimaAtmos's Schaer vertical mesh (z_elem = 100,
# z_max = 21 km, dz_bottom = 10 m, HyperbolicTangentStretching: see
# ClimaAtmos src/simulation/grids.jl get_stretching) as schar_z_faces.json,
# the woof_driver default.  CPU only; seconds.
#   $JULIA --project=$CA_ENV make_schar_z_faces.jl [out.json] [z_elem]
import ClimaCore: Geometry, Domains, Meshes

out = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "schar_z_faces.json")
nelem = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100
FT = Float32
domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(0),
    Geometry.ZPoint{FT}(21000);
    boundary_names = (:bottom, :top),
)
mesh = Meshes.IntervalMesh(
    domain,
    Meshes.HyperbolicTangentStretching{FT}(FT(10));
    nelems = nelem,
)
z = [Float64(p.z) for p in mesh.faces]
open(out, "w") do io
    println(io, "{")
    println(io, " \"source\": \"ClimaCore IntervalMesh(0..21000 m, ",
        "HyperbolicTangentStretching(dz_bottom = 10 m), nelems = $nelem), ",
        "Float32: ClimaAtmos plane_schar_mountain_float32_test.yml\",")
    println(io, " \"z_faces\": [", join(string.(round.(z; digits = 4)), ", "), "]")
    println(io, "}")
end
println("$(length(z)) faces, dz_bottom = $(z[2] - z[1]) m, top = $(z[end]) m")
