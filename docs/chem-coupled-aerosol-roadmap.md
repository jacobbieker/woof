# Coupled gas-aerosol chemistry: roadmap

WOOF's WRF-Chem subset (`chem_opt` 300/401 and the HRRR smoke package) carries
bulk aerosols with prescribed oxidants.  This roadmap adds a fully coupled
gas-phase and aerosol package with CUDA and Metal support, ported routine by
routine from WRF v4.7.1's `chem/` sources and graded word for word against
gfortran oracles built from those sources, as every other chem port is.

## Choice of package

The request was WRF-CMAQ, or a better coupled aerosol model if there is one.
The choice is **WRF-Chem's RADM2 gas phase with the MADE/SORGAM modal aerosol
(`chem_opt = 2`, RADM2SORG), then its CMAQ aqueous-chemistry variant
(`chem_opt = 41`, RADM2SORG_AQCHEM)**.

* **Online and coupled.** WRF-Chem runs chemistry on the model grid at every
  chemistry step, so the plumbing that already exists here (emissions,
  transport, deposition, optics and radiative feedback) serves it too.
  WRF-CMAQ (WRF 4.3+ with CMAQ 5.3+) calls CMAQ every N steps and feeds back
  the direct aerosol effect only.  Its code (CB6r3, AERO7, EBI/Rosenbrock
  solvers and its own I/O layer) shares nothing with what is here.
* **Licences decide between the WRF-Chem packages.** The first candidate was
  CBM-Z gas phase with MOSAIC sectional aerosol (`chem_opt` 7-10).  Their
  sources are not public domain or permissively licensed (see below), so
  WOOF carries none of them.  RADM2 and MADE/SORGAM carry no notice beyond
  WRF's public-domain statement.
* **No KPP.** RADM2's solver, like CBM-Z's, is hand-coded QSSA
  (`module_radm.F`), so there is no generated KPP code to reproduce.
* **CMAQ is not better.** CMAQ itself is now MIT-licensed (US EPA), but its
  AERO6/7 thermodynamics is ISORROPIA II, which carries
  "COPYRIGHT 1996-2012, UNIVERSITY OF MIAMI, CARNEGIE MELLON UNIVERSITY,
  GEORGIA INSTITUTE OF TECHNOLOGY" and no grant.  WRF-CMAQ also has no
  aerosol-cloud (indirect) effect.  The CMAQ piece worth having, AQCHEM, is
  already in WRF as `share/module_ctrans_aqchem.F`, under the EPA Models-3
  grant, and is what `chem_opt = 41` uses.

## Licence findings (WRF v4.7.1 `chem/`, identical in v4.6.1)

| Source | Terms | Status |
|---|---|---|
| `module_radm.F`, `module_data_radm2.F`, `module_aerosols_sorgam.F`, `module_data_sorgam.F`, `module_sorgam_aqchem.F`, `module_prep_wetscav_sorgam.F`, `module_ftuv_*.F` | No notice; WRF `LICENSE.txt`: public domain, "for use by any person or entity for any purpose" | Portable: on the roadmap (`module_radm.F` done) |
| `share/module_ctrans_aqchem.F` (CMAQ AQCHEM, HLCONST) | EPA Models-3: "permission to use, prepare derivative works, and distribute copies"; copies of derivatives requested | Portable: on the roadmap |
| `share/module_ctrans_aqchem.F`, function `INDEX1` | I/O API, LGPL-2.1 | Not ported (a name search; the port indexes at compile time) |
| `module_cbmz*.F`, `module_mosaic*.F` (all), `module_cmu_bulkaqchem.F`, `module_cmu_dvode_solver.F`, `cloudchem_driver.F`, `module_wetscav_driver.F`, `optical_driver.F`, `module_aer_drydep.F`, `module_mixactivate_wrappers.F`, `module_sorgam_cloudchem.F`, `module_peg_util.F`, `module_fastj_*.F` | Battelle/PNNL, DOE contract DE-AC05-76RL01830, with the terms below, given directly or by "see module_mosaic_driver.F for references and terms of use" | Opt-in only |
| `phys/module_mixactivate.F` (Abdul-Razzak and Ghan activation behind every WRF-Chem aerosol-cloud coupling) | Same pointer to the MOSAIC terms | Opt-in only |
| `module_optical_averaging.F`, `module_fastj_mie.F` | Same terms, given directly | Opt-in only |
| `module_phot_mad.F` | A GNU GPL-2 TUV section | Opt-in only, if ever |
| `isocom.F`, `isofwd.F`, `isorev.F` (ISORROPIA, called by SORGAM's `eql3` when `isorop = 1`, which `aeroproc` hard-codes) | "COPYRIGHT 1996-2006, UNIVERSITY OF MIAMI, CARNEGIE MELLON UNIVERSITY, GEORGIA INSTITUTE OF TECHNOLOGY", no grant | Never committed, opt-in or not: an exact build only from the user's own WRF tree (bring your own source, below); the default build runs SORGAM's own RPMARES branch (`isorop = 0`) |
| `module_aerosols_sorgam.F` routines `newt`, `lnsrch`, `fmin`, `ludcmp`, `lubksb` (SORGAM's SOA Newton solver) | Each headed "Adopted from Numerical Recipes in FORTRAN, Chapter 9.7 [2.3], 2nd ed."; Numerical Recipes' licence does not permit redistributing its source or derivatives | Never committed, opt-in or not: an exact build only from the user's own WRF tree; the default build runs WOOF's clean-room solver for the same equations |

The PNNL terms, as `module_mosaic_driver.F` gives them (`module_cbmz.F` has
the same text naming CBMZ):

> Terms of Use:
> 1) Users are requested to consult the primary author prior to modifying
>    the MOSAIC code or incorporating it or its submodules in another code.
>    This is meant to ensure that the any linkages and/or assumptions will
>    not adversely affect the operation of MOSAIC.
> 2) The MOSAIC source code is intended for research and educational
>    purposes. Users are requested to contact the primary author regarding
>    the use of MOSAIC code for any commercial application.
> 3) Users preparing publications resulting from the usage of MOSAIC are
>    requested to cite one or more of the references below (depending on
>    the application) for proper acknowledgement.

These are requests, not conditions, but they are not a licence grant either.
So these files enter WOOF only as opt-in packages under `optional/`, like
WRF-Hydro and NEMO: off by default, enabled through `woof/optin.py`, the
terms text beside the code, and excluded from the published wheels.  The
default build gets the same capabilities from clean-room implementations
written from the published papers (below), never from these files.

## Phases

`tools/chem_f2cu` translates whole WRF-Chem subroutines to CUDA under the
same exactness rules as the RADM2 transcriber (rounded unfused operations in
gfortran's association, exact constant folding, gfortran's integer-power
chains, glibc's words); every kernel it builds is regenerated against the
pinned source by a `--check` test and graded against a Fortran oracle.

**Bring your own source.**  Code whose terms forbid redistribution outright
(Numerical Recipes derivatives, ISORROPIA) is never committed, not even under
`optional/`.  Its exact path is built on the user's machine:
`WOOF_WRF_SOURCE=/path/to/WRF` (a WRF v4.7.1 or v4.6.1 checkout, digests pinned, a
mismatched tree refused by name) in a WOOF source checkout makes
`woof/core/chem_byo.py` run the translator on the user's own files into a
local cache (`~/.cache/woof/byo`).  The oracle fixtures committed for these
phases are numeric outputs only.  Tests of the exact path skip, naming the
rule, without `WOOF_WRF_SOURCE`.

**Oracle version.**  The lanes are pinned to WRF v4.7.1 (`f52c197e`), the
version upstream WOOF tracks.  P1, P2 and P4 were first built against v4.6.1
(`d66e442f`); every file they compile is byte-identical in v4.7.1, so the
lanes moved by re-pinning: the oracles rebuilt from the v4.7.1 tree reproduce
the committed fixtures byte for byte, and `WOOF_WRF_SOURCE` accepts either
tree.  Fast-TUV (P5) is v4.7.1 only: its driver changed between the two
(`ftuv_timestep_init` recomputes the Earth-Sun distance every call).

Each phase is one PR, tested on CUDA (pb) and Metal (mac) with
`GPUWM_WRF_EXACT=0` and `=1`.  Every port of a WRF source is graded bitwise
against its oracle; clean-room code (P8, P9, SORGAM's partitioning solver in
P2) is graded within stated tolerances instead, against the exact port.

| Phase | Content | WRF source | Status |
|---|---|---|---|
| P1 | RADM2 gas phase: rate constants, QSSA solver, adaptive column step | `module_radm.F` (1.7k lines) | **Done**: `woof/core/chem_radm2.py`, `chem_radm2.cu`, oracle `tools/chem_wrf471_oracle/radm2` |
| P2 | MADE/SORGAM aerosol step, `sorgam_driver`'s whole call tree: nucleation, condensation, coagulation, mode parameters, RPMARES inorganic equilibrium, SORGAM secondary organics | `module_aerosols_sorgam.F` with `aeroproc`'s `isorop = 0` (RPMARES); WOOF's clean-room partitioning solver (Schell et al. 2001 equations, damped Newton per Dennis and Schnabel 1996) | **Done**: `woof/core/chem_sorgam.py`, `chem_sorgam.cu` (generated by `tools/chem_f2cu`), oracle `tools/chem_wrf471_oracle/sorgam`; default build within a stated tolerance of WRF, exact path bitwise |
| P3 | Exact SORGAM equilibrium, ISORROPIA (`isorop = 1`), bring your own source | `isocom.F`, `isofwd.F`, `isorev.F` | |
| P4 | SORGAM dry deposition (`sorgam_depdriver` with `VDVG_2`, `aer_drydep_opt = 1`, and `VDVG`, `= 11`), PM sums and initial conditions (`aer_ic_opt` 1 and 101) | `sorgam_depdriver`, `VDVG`, `VDVG_2`, `sum_pm_sorgam`, `aerosols_sorgam_init`, `sorgam_init_aer_ic_pnnl` | **Done**: `launch_sorgam_drydep`, `launch_sorgam_sum_pm`, `launch_sorgam_ic`, bitwise against the oracle |
| P5 | Photolysis for RADM2 (Fast-TUV, `phot_opt = 3`) | `module_ftuv_*.F` | **Done**: `woof/core/chem_ftuv.py`, `chem_ftuv.cu` (photo's call tree generated by `tools/chem_ftuv/transcribe_ftuv.py`, ftuv_driver's column glue by hand), oracle `tools/chem_wrf471_oracle/ftuv`; bitwise on CUDA and Metal |
| P6 | Emissions for the RADM2/SORGAM species; namelist acceptance of `chem_opt = 1, 2` | `emissions_driver.F` mapping | **In progress**: emissions kernel (#292), background gas profile (#300), the `radm2`/`radm2sorg` sets with `emission.radm2`, `photolysis.ftuv`, `drydep.sorgam`, `chem.radm2`, `aerosol.sorgam` and the cold start (#301); `chem_opt = 1, 2` with `emiss_opt = 3`, `phot_opt = 3`, `gas_drydep_opt = aer_drydep_opt = 1` import (this PR) |
| P7 | Opt-in `optional/wrfchem_pnnl/`: `phys/module_mixactivate.F` and the modal optics (`optical_driver.F`, `module_optical_averaging.F`, `module_fastj_mie.F`), ported exactly and graded bitwise | PNNL-terms files | **In progress**: the opt-in door (`woof.optin.load("wrfchem_pnnl")`, `WOOF_ENABLE_WRFCHEM_PNNL=1`), `activate` (with `maxsat`) and the column driver `mixactivate` (with `explmix`, `loadaer`), bitwise against their oracles in `optional/wrfchem_pnnl/oracle/activate` on CUDA and Metal; `erfc_num_recipes` (Numerical Recipes) only from your own tree |
| P8 | Clean-room aerosol activation in the default build (Abdul-Razzak and Ghan 2000; Abdul-Razzak et al. 1998), validated against P7 within stated tolerances | the papers | **Done** (modal; sectional is the opt-in's): `woof/core/chem_activation.py`, `chem_activation.cu`; ARG00 maximum supersaturation and number/surface/mass fractions over Ghan et al. (1997)'s Gaussian updraft integral, behind the opt-in's `launch_activate(cases)` contract, `implementation()` choosing `cleanroom` (default) or `pnnl` (`WOOF_CHEM_ACTIVATION`). Against the opt-in run as a black box (`tests/test_chem_activation_cleanroom.py`): one updraft within 0.005 in every fraction (measured 0.0023); an updraft distribution within 0.10 (mean 0.01). No forecast process calls activation yet: P10's cloud-borne aerosol is its first user |
| P9 | Clean-room modal optics in the default build (volume-mixing and Maxwell-Garnett refractive-index averaging, Mie from Bohren and Huffman 1983), validated against P7 within stated tolerances | the papers | **Done** for the SORGAM rows: `woof/core/chem_modal_optics.py`, `chem_modal_optics.cu`, tables from `tools/chem_modal_optics/make_tables.py`; `aer_ra_feedback = 1` with RADM2SORG (legacy RRTMG, `aer_op_opt` 1 or 2) feeds RRTMG's shortwave and longwave and Fast-TUV. P7's optics are not ported yet, so it is validated against direct Mie integration and physical limits (`tests/test_chem_modal_optics.py`); the grading against P7 follows when the opt-in exists |
| P10 | CMAQ aqueous chemistry (`chem_opt = 41`), cloud-borne aerosol through P8 (default) or P7 (opt-in) | `module_sorgam_aqchem.F`, `share/module_ctrans_aqchem.F` | |
| P11 | Opt-in CBM-Z gas phase (`module_cbmz*.F`, about 4k lines plus solvers) and MOSAIC (about 70k lines across 28 files), `chem_opt` 7-10, exact | PNNL-terms files | |

Activation and optics come straight after the SORGAM aerosol, so aerosol-cloud
and aerosol-radiation coupling are available as soon as there is a modal
aerosol to couple.  The exact opt-in ports come first because they are the
reference the clean-room versions are validated against.  The radiative
feedback plumbing (`docs/chem-radiation-feedback.md`) serves either.

Clean-room rules: the default-build activation and optics are written from
the cited papers only, with no text, structure or constants taken from the
PNNL files.  They are graded against the opt-in ports within a stated
tolerance, never bitwise.

## Opt-in: the so2so4 cloud threshold (`so2so4_qc_threshold`)

RADM2SORG converts SO2 to sulfate in cloud through GOCART's `so2so4`
(`chem_driver.F:1350-1360`), which calls a cell cloudy when `qc > 0`.  In a
real-data run the specified boundary ring carries round-off cloud water
(1e-38 to 1e-10 kg/kg) from the lateral forcing.  In such a cell `so2so4`
turns all the SO2 that H2O2 allows into sulfate in one chem step, and SORGAM
nucleates it into an Aitken burst.  An outflow cell carries the interior's SO2
there.  On a 9 km, 6 h GFS case (80 x 100 cells), every ring-0 cell with
`nu0 > 1e13 /kg` had `qc > 0`, and none with `qc = 0` did.  The 5-cell edge
band's surface `nu0` averaged 1.6e12 /kg, against 2.3e11 in the interior.
The thin inflow aerosol was not the cause: the inflow cells hold
`bdy_chem_value_sorgam`'s ~1e9 /kg.

`so2so4_qc_threshold` (kg/kg, default 0) is not a WRF key.  At 0 the run is
WRF's: the kernel runs on `qc` itself, and the history is byte-identical to
a run without the key.  A positive value makes a cell cloudy only when
`qc > threshold`.  `1e-6` is WRF's own cloud test in `cal_cldfra`
(`phys/module_radiation_driver.F:3714`).  The threshold applies to every
cell, not only the boundary.  The door refuses it for a selection without
`aerosol.sorgam`, since nothing else runs `so2so4`.  It needs no boundary
source.

On the same case with `so2so4_qc_threshold = 1e-6`, CUDA and Metal agree to
the digits shown:

- Ring-0 cells with `nu0 > 1e13 /kg` in the lowest 20 levels fall from 360-530
  per hourly frame to 0-1.
- At 06Z, the edge band's surface `nu0` falls from 1.56e12 /kg to 8.9e10 /kg,
  below the interior's 1.8e11-2.2e11.
- The threshold also skips round-off cloud in the interior. Surface PM2.5 at
  06Z falls from 1.9 to 1.7 ug/m3 there.

Applying the threshold domain-wide is the user's decision (2026-11-13).
Round-off cloud water is not cloud anywhere, so the interior effect is
expected and physically motivated.  The default stays 0 (WRF's behaviour),
and the opt-in value is 1e-6.

## Refusals until then

`woof/namelist_chem.py` maps `chem_opt = 1, 2` onto the `radm2` and
`radm2sorg` sets and refuses `chem_opt = 41` as "being ported",
pointing here, and refuses every CBM-Z/MOSAIC package by name (including
`chem_opt = 120`, CBMZ_BB_KPP, which shares CBM-Z's emission and
initial-condition modules), citing the
terms and the planned opt-in.  Every other uncarried package keeps the
generic refusal.
