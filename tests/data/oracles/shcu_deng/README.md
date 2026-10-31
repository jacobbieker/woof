# Deng shallow cumulus WRF v4.7.1 column-tile oracle fixture

Graded by `tests/test_shcu_deng_wrf471_oracle.py`.  Produced by
`tools/shcu_deng_wrf471_oracle` (see its README for the recipe) on 2026-10-29:
macOS 26 arm64, conda-forge gfortran 16.2.0 (`pixi exec -s gfortran`),
`-O0 -ffp-contract=off`, float32 libm pinned to WOOF's glibc 2.39
transcription (`libm_shim.cpp`).  `compiler.txt` and `oracle-sha256sums.txt`
are the build's own receipts (every source hashed).

| file | what it is | SHA-256 |
|---|---|---|
| `columns-structured.bin.gz` | 8 tiles, 400 columns, 38 steps (`deng_columns.py write`, seed 20261029), the exact float32 words both sides read | `9a441d158b6502e6d0c6d9eda41643588981f40a8187dbcfe2d947383d925388` |
| (uncompressed) | | `f32249cff2df92309e002977bbabdc7d1b964a979f315f83250f1cc79fd89e21` |
| `columns-structured.json` | per-column regime and whether it starts from an injected NBC/updraft history | `1cccd8d1a7f62b36fd002f878fb9f697ff525f83a98c3ab11f14c877f1dc0d9f` |
| `oracle-structured-O0.out.gz` | every output of the Deng arm of `shallowcu_driver`, every step | `0538f6541a9a4e69f021889987566621ee3b432b05f30f54c7a234fb434b4156` |
| (uncompressed) | | `9d40450ce909c5eac45321df733e691a92f04ac7b10022852d644e4322f5b0fc` |

WRF sources: `phys/module_shcu_deng.F`
`d03fd43559e1562bde415eb7a50906b993c6da3f5dc3e501f957b48031629645`,
`phys/module_shallowcu_driver.F`
`068cf22b8c93cfac74a6223a72d61f4a2f890bc44e212da5d2c2a131c5ae271a` (v4.7.1,
LF).

Receipts from the same run:

- The SNaN-initialised build (`-finit-real=snan -finit-integer=-2147483647`)
  wrote a byte-identical oracle file (same SHA-256 as above), so no output
  depends on memory the Fortran never wrote.
- No step reached `wrf_error_fatal` or `wrf_message`; no column reached a
  `STOP`.
- 1,737,808 graded words per run: 18 level fields, 12 surface fields, 3
  integer fields and the 100-deep AINCKFSA history, for every column of
  every step.  160 column-steps rain, 106 carry a >= 4 km cloud (the
  downdraft/drizzle regime), 988 carry NBC.
- `coverage-never-executed.txt`: 1,868 lines of the module executed, and the
  ones never executed, with the reasons in the tool README.
