# Vendored gpuwm-hex helper crates

`host-prep/` and `physics-tables/` are copied byte for byte from the gpuwm-hex
0.4.14 source distribution on PyPI, the hex release `components/hex` tracks:

- archive: `gpuwm_hex-0.4.14.tar.gz`
- sha256: `c47368e24d63c4b8fc69845b530938164957c4ec12853ac83bbc3c9a8c4f2db5`
- paths in the archive: `gpuwm_hex-0.4.14/rust/host-prep/`, `gpuwm_hex-0.4.14/rust/physics-tables/`

They are not edited here. `diff -r` against the archive's two directories is
empty; a change belongs upstream, then a new copy with this note updated.

Licence: both crates are gpuwm-hex project code under Apache-2.0
(`LICENSE-gpuwm-hex`, the archive's `LICENSE`). `NOTICE-gpuwm-hex` is the
archive's `NOTICE`; its section 12 names these crates ("project code,
Apache-2.0", Rust standard library only). Neither crate has a dependency
outside the Rust standard library (each `Cargo.lock` lists only the crate
itself), so no third-party licence text travels with them.

What they build:

- `host-prep/`: `hex_contract_hostprep`, the regional contract deck's float32
  host references (`woof.hex.drivers.contract_hostprep`).
- `physics-tables/`: the MPAS suite route's table decoders (`liblookup_decode`,
  `table_decode`, `rw_mpas_rrtmg_lw_tables`, `rw_mpas_rrtmg_sw_tables`,
  `ruc_table_decode`, `tempo_table_compute`, `tempo_table_init`;
  `woof.hex.mpasphys.resources_build`).

Why: the hex release publishes these helpers only in its Linux helper archive,
so a macOS checkout could not run the contract stage on a new regional geometry
or the MPAS v8.4.2 suite route. Build them with `tools/build_hex_helpers.sh`.
They are built from source only: `rust/` is not a Python package, so no wheel
(metal-woof or recast-woof) carries them.
