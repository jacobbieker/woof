# MPAS-Seaice v8.4.2 column-package oracle: the brine fraction through the ITD, the ridging and the time step

Written by `tools/mpas_seaice_oracle/build.sh` and
`make_column_fixture.py --cases carry` on pb (Linux 6.12 x86_64, glibc 2.44),
2026-11-21.

- MPAS source: MPAS-Model v8.4.2 (`f34984b2`), every file checked against
  `tools/mpas_seaice_oracle/SOURCES.sha256`. The build is the same as
  `../mpas_seaice_bgc_brine/PROVENANCE.md` (the column package and
  `mpas_seaice_column.F` compiled verbatim, gfortran 16.2.0, `-O0
  -ffp-contract=off`, RKIND 8 only); the Fortran is unchanged from it.
- Configuration: `config_use_column_biogeochemistry` and `config_use_brine`,
  every other biogeochemistry flag off, so the brine fraction (`nt_fbri`,
  parent the ice volume) is the only biogeochemical tracer.
- Cases: `woof.verify.mpas_seaice_column_oracle.BGC_CARRY_CASES` and
  `BGC_CARRY_COUPLED_CASES`, each an existing stage's case with the brine
  height on and a brine fraction drawn from u(0.6, 1.1):
  - the ITD thermodynamics (`itd_thermodynamics`): mushy with level ponds;
    BL99 updating the ocean fluxes; BL99 with the delta-function conversion
    and four categories; BL99 with a single category; three ice layers with
    topographic ponds. With two or more categories, cell 2's second
    category is emptied into its first, so `add_new_ice_bgc`'s empty
    category (brine fraction 1) is reached; cell 3 is zapped
    (`zap_small_areas` sets the brine fraction to 1) and cell 1 forms new
    ice (`vi0new` added to the first category's brine volume);
  - the ridging (`ridging`): mushy; BL99 with Thorndike and Hibler; three
    subcycles with topographic ponds and an e-folding scale of 4.5. The
    brine fraction is a volume tracer of the ridged strata;
  - the column step (`column_step`, three steps): BL99 with the
    delta-Eddington shortwave; that with two subcycles, four ice layers,
    three categories and the Gregorian calendar; mushy with level ponds and
    no vertical thermodynamics. `column_biogeochemistry` runs after the
    vertical thermodynamics; `coupling_prep` zeroes the owned cells'
    fourteen ocean biogeochemistry fluxes and `scale_fluxes` divides them by
    the ice area (zero where there is no ice);
  - the coupled step (`seaice_step`, with the velocity solver, no
    advection): BL99 with the delta-Eddington shortwave and Rothrock's
    strength on a 12x6 torus, 3 steps; mushy with level ponds and no
    vertical thermodynamics on a 10x5 torus, 2 steps.
- v8.4.2's own fault, not reproduced: `seaice_column_reinitialize_diagnostics_bgc`
  (`mpas_seaice_column.F:12450-12471`) zeroes `totalChlorophyll`, whose
  Registry packages are `pkgTracerVerticalAlgae;pkgTracerSkeletalAlgae`.
  With the brine height alone both are off, MPAS allocates no array, the
  field's pointer is null (gfortran gives it size 1), and the write goes to
  address 0: v8.4.2 dies on the first step of every case with the
  biogeochemistry on and no algae (reproduced on this host under ASLR and
  in a standalone gfortran test, -O0 and -O2). The port refuses such a
  configuration by default; the step cases declare the opt-in correction
  `woof_v842_corrections = ('bgc_inactive_diagnostics',)` (zero only the
  fields that exist). For those cases the oracle's input stream gives
  `totalChlorophyll` an array (`field_rows`; `stand_ins`), so the verbatim
  Fortran runs; that field is a stand-in, not v8.4.2's words, and is not
  compared (`expected_outputs`). Every other output is v8.4.2's.
- v8.4.2's own behaviour, kept word for word: `add_new_ice_bgc` adds the
  thickness surplus's brine volume to every category and sets an empty
  category's brine fraction to 1 (`ice_zbgc.F90:29-251`); radiation reads
  the brine fraction only under `dEdd_algae` (refused).
- `oracle-r8.npz`: per case, every output field whose words differ from the
  input's. `column-logs.json`: the tracer object and the log lines.
  `inputs.sha256`: the input-stream digests.
- libm: the step cases' delta-Eddington shortwave and orbit use glibc 2.44's
  `sin`, `cos`, `atan` and `asin`; on the GPU these need the LGPL opt-in
  `WOOF_GLIBC239_LIBM64=1` (as `../mpas_seaice_column_dedd`). The ITD and
  ridging cases need no opt-in.
