#!/usr/bin/env bash
# Build the hex helpers a source checkout needs where no hex helper archive is
# published (macOS): the regional contract's host references and the MPAS
# suite route's table decoders, from the gpuwm-hex 0.4.14 crates vendored in
# rust/ (rust/PROVENANCE.md), and the Registry pool reader from this tree's
# tools/rustwx.
#
#   tools/build_hex_helpers.sh [NETCDF_LIB_DIR]
#
# NETCDF_LIB_DIR holds libnetcdf (the pool reader links netCDF-C); default: the
# active conda/pixi environment's lib ($CONDA_PREFIX/lib).  Every build is
# --release --locked.  Nothing here enters a wheel: rust/ is not a
# package and the outputs stay in each crate's target/ (git-ignored).
#
# The doors find the outputs on their own:
#   rust/host-prep/target/release/hex_contract_hostprep  (woof.hex.drivers.contract_hostprep)
#   rust/physics-tables/                                 (built again into the physics-resources cache)
#   tools/rustwx/target/release/rw_mpas_physics_pool     (woof.hex.mpasphys.resources_build)
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
netcdf=${1:-${CONDA_PREFIX:-}/lib}
for crate in host-prep physics-tables; do
  cargo build --release --locked --manifest-path "$root/rust/$crate/Cargo.toml"
done
if ! ls "$netcdf"/libnetcdf.* >/dev/null 2>&1; then
  echo "no libnetcdf in '$netcdf'; pass NETCDF_LIB_DIR (the pool reader links netCDF-C)" >&2
  exit 2
fi
RUSTFLAGS="${RUSTFLAGS:-} -L $netcdf -C link-arg=-Wl,-rpath,$netcdf" \
  cargo build --release --locked --manifest-path "$root/tools/rustwx/Cargo.toml" \
  -p rw-mpas --features system-netcdf --bin rw_mpas_physics_pool
"$root/tools/rustwx/target/release/rw_mpas_physics_pool" --abi
echo "built: $root/rust/host-prep/target/release/hex_contract_hostprep"
echo "built: $root/rust/physics-tables/target/release/{table_decode,rw_mpas_rrtmg_lw_tables,rw_mpas_rrtmg_sw_tables,ruc_table_decode,tempo_table_compute,tempo_table_init,liblookup_decode}"
echo "built: $root/tools/rustwx/target/release/rw_mpas_physics_pool"
