# WRF-Chem v4.7.1 Wesely gas column oracle

Build only in the authorized CPU scratch directory:

```sh
cd ~/work/aq-cams-wesely
nice -n 15 bash harness/build.sh source build
```

The committed fixtures were built with GNU Fortran (Ubuntu 15.2.0-16ubuntu1)
and Ubuntu GLIBC 2.43-2ubuntu2.4.  On a host without that compiler, an
`ubuntu:26.04` image with `gfortran curl ca-certificates binutils` installed
(`apt-get install --no-install-recommends`) is that toolchain; mount every
path explicitly:

```sh
docker run --rm -u $(id -u):$(id -g) -v /path/to/wrf-src:/wrf:ro \
    -v "$PWD":/repo:ro -v /path/to/build:/build -w /build wesely-oracle:u2604 \
    bash /repo/tools/chem_wrf471_oracle/wesely/build.sh /wrf /build/out
```

`SOURCES.sha256` pins the CRLF form of `chem/module_dep_simple.F` (the
pinned reference tree is a CRLF checkout; the v4.7.1 tag's LF file,
sha256 `063b72f5...`, is the same text).  For an LF tree, write
`sed 's/$/\r/'` of the file into `/wrf/chem/` first.  Then copy
`/path/to/build/out/fixtures/case_*` to `tests/data/oracles/chem/cams/wesely`
and `fixtures/radm2/*` to `tests/data/oracles/chem/wesely_radm2`; the
2026-11-11 rebuild reproduced every existing `case_*` file byte for byte.

The public entry point is `build.sh WRF_SOURCE_ROOT BUILD_DIR`. The source root
needs `chem/module_dep_simple.F`. The script fetches the two public v4.7.1
support modules into the build directory and checks all three compiled WRF
files against `SOURCES.sha256` before compiling. The deposition module stays
byte-unmodified. All object files linked into the reference use `-O0
-fno-fast-math` and the seven WRF defines recorded in fixture PROVENANCE.md.
Every linked object is inspected for forbidden `_ZGV*` references. The copied
Noah positive control must emit a `_ZGV*` reference at `-Ofast` before running
any fixture. The compiler is `/usr/bin/gfortran`.

No `-DNETCDF` is used. Only the MOZART seasonal-PFT path requires it, at
chem/module_dep_simple.F:1668-1684; this oracle takes the non-MOZART path.
The real model constants and error module are compiled. Stubs supply state
indices, package constants copied from Registry/registry.chem with individual
line citations, configuration queries and services. No stub computes physics.
The unused MOZART HL lookup has an empty service table and an aborting string
service. `wrf_message` is supplied by the compiled error module.

`run_wesely.F90:8` has nine gas slots: absent index 1, six potential real
rows at 2:7, and two trailing dummy gases at 8:9. Thus the non-MOZART zeroing
at chem/module_dep_simple.F:243-244 never lands on a real row. NH3 is absent
(index 1) except cases 07 and 08. SO2 is present in all cases. High/low NH3
input columns alternate in the extra cases. The driver directly calls
`dep_init` and `wesely_driver` with GOCART_SIMPLE=300. The normal WRF
`dry_dep_driver.F:259-282` call site excludes 300, so this oracle is a direct
column exercise of the requested parameter branch, not that normal call site.

Each fixture is 24 by 18 columns. Fortran raw arrays retain their logical
shape in `manifest.txt` and column-major bytes; Python reshapes with order F
and transposes into C-order `(nrow,ny,nx)`. `oracle_io.F90` supports REAL(4)
and INTEGER(4), ranks 0 through 3, with little-endian stream writes.

| fixture | map | julday | rows | purpose |
| --- | --- | ---: | --- | --- |
| case_01 | USGS | 90 | four v1 gases | summer boundary |
| case_02 | USGS | 89 | four v1 gases | winter boundary |
| case_03 | USGS | 270 | four v1 gases | summer end |
| case_04 | USGS | 271 | four v1 gases | winter start |
| case_05 | MODIFIED_IGBP_MODIS_NOAH | 180 | four v1 gases | all 20 mapped classes |
| case_06 | MODIFIED_IGBP_MODIS_NOAH | 89 | four v1 gases | winter map |
| case_07 | USGS | 180 | six including extras | NH3/HNO3 and both highnh3 states |
| case_08 | USGS | 89 | six including extras | winter NH3 arms |
| case_09 | USGS | 180 | four v1 gases | state A |
| case_10 | USGS | 180 | four v1 gases | changed shortwave, state B |
| case_11 | USGS | 180 | four v1 gases | A/B/A calls; saved B and final A |

All USGS cases exercise classes 1:24, including water 16 and ice 24.
Inputs span gsw 0/50/800, dry/RH 80/RH 85/wet, qr and raincv rain flags,
stable/neutral/unstable rmol and the near-zero snap, low/high ust including
ust below the aerodynamic-resistance floor, low/high znt, and tc
-10/-5/-3/-1/0/1/2/3/45 degrees C. Case 10 shifts gsw by 100. The A/B/A
reference proves no memory and is compared to independent A and B fixtures.
There are 4752 fixture columns, with two additional calls in case 11.

## RADM2/RADM2SORG layout (`run_wesely_radm2.F90`)

A second executable, linked against the same objects, runs `dep_init` and
`wesely_driver` with the 42 gases of Registry/registry.chem:3929-3930 at
chem indices 2:43 (`numgas = p_ho2 = 43`, share/module_chem_share.F
`get_last_gas`) and every other `p_*` index absent (1).  `num_chem` is a
stub variable, not a parameter, so this driver can size the chem array 43.
It dumps `hstar`, `dhr`, `f0`, `dratio` and `scpr23` for all 42, and `dvj`:
dep_init keeps `dvj` local, so the driver carries the source values
(:1956, :2880-2920) and stops unless `0.242/(dvj*(293.15/298.15)**1.75)`
equals dep_init's `dratio` word for word.  `ddvel` covers all 42 gases,
including WRF's own `srfres(numgas-1:numgas) = 0` for ho and ho2 (:244),
`vgs(p_sulf) = vgp` (:1437-1447) and the RADM2 copy
`ddvel(p_hcl) = ddvel(p_hno3)` (:259-264).  NH3 alternates around twice SO2
column by column, so both `highnh3` states occur.  Output:
`tests/data/oracles/chem/wesely_radm2`.

| fixture | map | julday | chem_opt |
| --- | --- | ---: | ---: |
| case_01 | USGS | 180 | 1 |
| case_02 | USGS | 89 | 2 |
| case_03 | MODIFIED_IGBP_MODIS_NOAH | 180 | 2 |
| case_04 | MODIFIED_IGBP_MODIS_NOAH | 271 | 1 |

## Port contract and transcription differences

- `woof/core/chem_drydep_gas.py:84` packs four float32 row numbers and an
  integer arm. No process branch tests a species name or species index.
  Names are used only for error labels, generic state lookup and diagnostics.
- Arm 0 general, 1 ozone, 2 sulfur_dioxide, 3 ammonia, 4 zero_surface,
  5 particle. Source arms are
  chem/module_dep_simple.F:965-1038,1044-1057,1081-1150,1154-1287,244,
  1437-1447. zero_surface sets rc = 0 and keeps the literal
  `1/(1/v + rc)`; particle replaces the velocity by
  `vgp = 1/((1/vgpart) + (1/dvpart))` before cellvg.  hcl's row carries
  hno3's numbers on the general arm, which is WRF's copy bit for bit.
  The tables and both land maps come from JSON, with source citations.
- The CUDA kernel fuses the gas subtree into one column thread. Packed table
  offsets and Fortran-to-C array transposes are storage changes only.
  `deppart` is transcribed; its particle velocity feeds the particle arm
  and its fog velocity is unused.  MOZART, chemical-package aliases and seasons 3/4/5 are excluded because
  they are outside the requested non-MOZART gas path.
- WRF mutates very small rmol to zero. The port uses the snapped value locally
  for the same calculations and leaves the caller's met array unchanged.
- Every transcendental uses the shared glibc float32 header. Integer squares
  use multiplication, as gfortran's depvel object does at -O0.
- `WFloat` stores exactly one float32 word and every operation is a scalar
  round-to-nearest PTX instruction without an FTZ modifier, because CuPy
  appends `-ftz=true` after user options; an inline-asm operation cannot be
  contracted into an FMA, so no loader option is needed. KARMAN = 0.4 comes
  from share/module_model_constants.F:82.
- Each row carries the six dep_init numbers, including `dratio` and `scpr23`
  as the float32 values the Fortran derived at chem/module_dep_simple.F:
  3478-3489 (a rounded dratio is not an invertible encoding of dvj, so the
  row carries scpr23 itself). The parity test asserts every row number equals
  the Fortran dump bit for bit.
- rc's highnh3 test reads the rows on the ammonia and sulfur_dioxide arms.
- The process (`woof/core/chem_drydep_gas.py`) follows
  `woof/core/chem_context.py`: it writes each Wesely row's plane of
  `ctx.ddvel` from the third step on (chem/chem_driver.F:1047) and publishes
  `wesely_aer_res_def`/`wesely_aer_res_zcen` for the aerosol deposition.

## Validation

* CPU: `python -m pytest tests/test_chem_wesely_wrf471_parity.py -m "not gpu"`
  (tables, land maps and every row number equal the Fortran dump; packing and
  refusals).
* GPU: `python -m pytest tests/test_chem_wesely_wrf471_parity.py -m gpu`:
  0 ULP for every row's ddvel and both aerodynamic resistances on all 11 cases,
  measured on an RTX 5090 (sm_120) and an RTX 4090 (sm_89); and on the 4
  RADM2 cases, all 42 gases, on an RTX A6000 (CUDA) and an Apple M3 Max
  (Metal).
* Mutation: changing one constant of rc in the kernel (rluo1's 3000 to 3001)
  fails all 11 GPU cases; changing one JSON table entry (ri(2,1) 60 to 120)
  fails all 11 table cases. Both were reverted.

## Refusals (woof/core/chem_drydep_gas.py)

- Unsupported MMINLU: "WRF-Chem's dep_init leaves luse2usgs unset for <name>,
  so every land class would index garbage".
- sf_surface_physics = 0: "WRF-Chem requires a soil model: iland would be zero
  in deppart and index outside every table".
- A land-use dataset the surface scheme's tables do not name: the Wesely
  classes cannot be mapped.
- A row naming drydep.wesely without the wesely block, an unknown arm, a
  missing or nonfinite number, a nonpositive dratio or scpr23, or two rows on
  the ammonia or sulfur_dioxide arm.
- Wrong dtype, shape or contiguity of a launch field, and ivgtyp outside the
  land map.
