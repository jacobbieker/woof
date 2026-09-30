# Scientific Manual (engine 2.5)

WOOF is an independent, GPU-native implementation of a WRF-ARW-class regional
atmospheric model. It is not affiliated with or endorsed by NCAR or UCAR. The name is
a wordmark for "the ARW solver, GPU-native"; the Python package is named `woof`
[README.md:1-7].

WOOF is a research and educational tool. It is never a substitute for official
forecasts and warnings from your national meteorological service. Do not use it to
make safety decisions [README.md:9-12].

This manual is written for atmospheric researchers who know WRF and are evaluating
WOOF for real work: severe-storm simulation, downscaling, ensembles, and LES. It
states measured limits with the same prominence as measured capabilities, because a
number without its conditions is a different claim.

## Chapters

1. [What WOOF is and how it relates to WRF-ARW](01-arwen-and-wrf.md)
2. [Dynamics, grids, nesting, and LES](02-dynamics-grids-nesting.md)
3. [The physics suite](03-physics.md)
4. [GPU numerics a researcher must know](04-gpu-numerics.md)
5. [Initialization: the arbitrary-source engine](05-initialization.md)
6. [The pipeline: fetch, prep, sim, render](06-pipeline.md)
7. [Verification instruments](07-verification.md)
8. [Operational envelope](08-operational-envelope.md)
9. [Limits and roadmap](09-limits-and-roadmap.md)

## Conventions

**Citations.** Every quantitative claim in this manual carries a bracketed source.
Bare paths (`docs/public/PHYSICS.md`, `tests/test_run_stamp.py`) are repo-relative
files in the `woof` repository; tests are behavioral receipts held green by the
suite. Paths prefixed `receipt:` are the project's measurement reports and paths
prefixed `gallery:` its evidence figures; both are cited for provenance and are not
part of this repository. A claim with no bracket is a definition
or a description of code behavior, not a measurement.

**Figures.** Weather-field figures referenced here are existing artifacts of the real
Rust renderer (`rw_wrfbatch` / `rw_ensbatch`); analysis charts (spectra, timing bars)
are matplotlib. Figure references give the `gallery:` path; no figure was produced
for this manual.

**Version.** This manual was written against engine 2.5.0, the line the WOOF engine
descends from. Where a measurement was taken on an earlier engine version (notably
the GPU-vs-CPU reference tables, measured on the engine's 1.5.0 wheel), the version
is stated beside the number.

**Vocabulary.** WOOF is the product; `woof` is the Python package and CLI;
`rw_wrfbatch` and `rw_ensbatch` are the Rust renderers driven through `woof.rustwx`;
`wrf-rust` is the science core used for diagnostics. The physics maturity vocabulary
is defined in chapter 1. A translation table from the product's vocabulary to the
WPS nouns a WRF user knows (prep is WPS plus `real.exe`; sim is `wrf.exe`) is
`docs/public/GLOSSARY.md`; this manual uses the product vocabulary and does not
duplicate that table.
