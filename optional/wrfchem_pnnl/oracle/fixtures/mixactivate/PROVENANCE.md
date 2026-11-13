# mixactivate oracle provenance

WRF v4.7.1, commit `f52c197ed39d12e087d02c50f412d90d418f6186`. The lane and
the build are `oracle/activate`'s; see `../activate/PROVENANCE.md` for the
compiler, libc, flags and rebuild command. The build also copies
`cal_cldfra2` byte for byte from `phys/module_radiation_driver.F` into a
module of that name (sha256-checked against `SOURCES.sha256`). `stub_state.F90`
holds the two Registry names that `cal_cldfra2`'s USE line imports.

The driver is `run_mixactivate.F90` (sha256 `3fc57cdde57c9d9c5c48cf296bdadae9a981c46ae2c622479aff6ef789e37d26`). It runs `mixactivate`
over one 4 x 2 tile of 20-level columns, with a halo in the memory dimensions,
for two consecutive calls, in two descriptions:
- `modal`: SORGAM-like, two types (Aitken + accumulation, and coarse), 4 and
  2 components, interstitial and cloud-borne phases, two water pointers;
- `sectional`: MOSAIC-like, one type, four sections, msectional = 20.

The columns cover several cloud states:
- cloud forming in place;
- cloud forming from below under an updraft;
- dissipating, persisting and clear columns;
- ice only;
- a thin cloud just over the threshold;
- cloud at the surface.

Between the calls the old cloud fraction takes the new one's value and the
cloud water changes, as in the harness.

Coverage of the extracted lines over both drivers, without the
print and write statements (gfortran --coverage, measured with gfortran 16 on
the mac):

| Routine | Coverage |
| --- | --- |
| mixactivate | 93 % |
| explmix | 100 % |
| loadaer | 89 % |
| activate | 83 % |
| maxsat | 96 % |

The unexecuted lines in mixactivate are the column-mass diagnostics
(`icheck_colmass = 0`) and fatal paths.
