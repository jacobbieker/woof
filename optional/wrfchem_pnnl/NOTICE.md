# WRF-Chem PNNL opt-in: terms and provenance

Everything under `optional/wrfchem_pnnl/` is an **opt-in**. It is off by
default. It is in no wheel (`metal-woof`, `recast-woof`) and no default
build, and the release snapshot excludes it (`RELEASE-EXCLUDE.txt`). It loads
only through `woof.optin.load("wrfchem_pnnl")`, with
`WOOF_ENABLE_WRFCHEM_PNNL=1` set in a source checkout. The project owner chose
this arrangement for WRF-Chem's Battelle/PNNL files
(`docs/chem-coupled-aerosol-roadmap.md`, "Licence findings").

## Terms

The code here is a CUDA transcription of WRF v4.7.1 files that carry the
Battelle Memorial Institute notice (DOE contract DE-AC05-76RL01830). Those
files point to the MOSAIC terms of use in `chem/module_mosaic_driver.F`.
`LICENSE` in this directory holds that text, copied verbatim. In short:

1. Users are requested to consult the primary author (Rahul A. Zaveri, PNNL)
   before modifying the code or incorporating it in another code.
2. The code is intended for research and educational purposes. Users are
   requested to contact the primary author about any commercial application.
3. Publications that use it are requested to cite the references `LICENSE`
   lists.

These are requests, not a licence grant. That is why the code is never in a
wheel and never in the default build. **Commercial users must ask PNNL
before enabling this opt-in.** The government and the contractor give no
warranty and accept no liability. No endorsement by PNNL, Battelle or the
DOE is implied.

The default build gets the same capabilities from clean-room implementations
written from the published papers: activation from Abdul-Razzak and Ghan
(2000), and optics from volume-mixing and Maxwell-Garnett averaging with
Bohren and Huffman (1983) Mie. It takes nothing from these files. They are
graded against this opt-in within a stated tolerance.

## Numerical Recipes

`phys/module_mixactivate.F` computes its error function with
`erfc_num_recipes`, which is headed "from press et al, numerical recipes,
1990". Numerical Recipes' licence does not permit redistribution, so that
function is **not** here, opt-in or not. The exact path builds it on your
machine from your own WRF tree (`WOOF_WRF_SOURCE=/path/to/WRF`, the rule
`woof/core/chem_byo.py` follows), and without that tree the activation
kernel refuses by name.

## Source

| WOOF file | WRF v4.7.1 source |
| --- | --- |
| `woof_wrfchem_pnnl/activate.cu`, `activate.py`, `tools/transcribe_activate.py` | `phys/module_mixactivate.F`: `activate`, `maxsat`, `maxsat_init`, `erf_alt`, translated by `tools/chem_f2cu` with the control-flow restatements the transcriber lists; `erfc_num_recipes` is translated only into a local cache from your own tree |
| `oracle/activate/` (harness, extraction list, build script; the WRF files are read from your tree, not copied), `oracle/fixtures/activate/` (numeric output only) | the same routines, compiled unchanged |
