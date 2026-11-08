# Coupled gas-aerosol chemistry: roadmap

WOOF's WRF-Chem subset (`chem_opt` 300/401 and the HRRR smoke package) carries
bulk aerosols with prescribed oxidants.  This roadmap adds a fully coupled
gas-phase and aerosol package with CUDA and Metal support, ported routine by
routine from WRF v4.6.1's `chem/` sources and graded word for word against
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

## Licence findings (WRF v4.6.1 `chem/`, identical in v4.7.1)

| Source | Terms | Status |
|---|---|---|
| `module_radm.F`, `module_data_radm2.F`, `module_aerosols_sorgam.F`, `module_data_sorgam.F`, `module_sorgam_aqchem.F`, `module_prep_wetscav_sorgam.F`, `module_ftuv_*.F` | No notice; WRF `LICENSE.txt`: public domain, "for use by any person or entity for any purpose" | Portable: on the roadmap (`module_radm.F` done) |
| `share/module_ctrans_aqchem.F` (CMAQ AQCHEM, HLCONST) | EPA Models-3: "permission to use, prepare derivative works, and distribute copies"; copies of derivatives requested | Portable: on the roadmap |
| `share/module_ctrans_aqchem.F`, function `INDEX1` | I/O API, LGPL-2.1 | Not ported (a name search; the port indexes at compile time) |
| `module_cbmz*.F`, `module_mosaic*.F` (all), `module_cmu_bulkaqchem.F`, `module_cmu_dvode_solver.F`, `cloudchem_driver.F`, `module_wetscav_driver.F`, `optical_driver.F`, `module_aer_drydep.F`, `module_mixactivate_wrappers.F`, `module_sorgam_cloudchem.F`, `module_peg_util.F`, `module_fastj_*.F` | Battelle/PNNL, DOE contract DE-AC05-76RL01830, with the terms below, given directly or by "see module_mosaic_driver.F for references and terms of use" | Opt-in only |
| `phys/module_mixactivate.F` (Abdul-Razzak and Ghan activation behind every WRF-Chem aerosol-cloud coupling) | Same pointer to the MOSAIC terms | Opt-in only |
| `module_optical_averaging.F`, `module_fastj_mie.F` | Same terms, given directly | Opt-in only |
| `module_phot_mad.F` | A GNU GPL-2 TUV section | Opt-in only, if ever |
| `module_aerosols_sorgam.F` routines `newt`, `lnsrch`, `fmin`, `ludcmp`, `lubksb` (SORGAM's SOA Newton solver) | Each headed "Adopted from Numerical Recipes in FORTRAN, Chapter 9.7 [2.3], 2nd ed."; Numerical Recipes' licence does not permit redistributing its source or derivatives without a licence from Numerical Recipes Software | Opt-in only; the default build needs a clean-room solver for the same partitioning equations |

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

Each phase is one PR, tested on CUDA (pb) and Metal (mac) with
`GPUWM_WRF_EXACT=0` and `=1`.  Every port of a WRF source (all phases but
P8 and P9) is graded bitwise against its oracle; the clean-room P8 and P9
are graded against the P7 port within stated tolerances instead.

| Phase | Content | WRF source | Status |
|---|---|---|---|
| P1 | RADM2 gas phase: rate constants, QSSA solver, adaptive column step | `module_radm.F` (1.7k lines) | **Done**: `woof/core/chem_radm2.py`, `chem_radm2.cu`, oracle `tools/chem_wrf461_oracle/radm2` |
| P2 | MADE/SORGAM aerosol dynamics: nucleation, condensation, coagulation, mode parameters | `module_aerosols_sorgam.F` `rpmmod3`, `aeroproc`, `aerostep`, `nuclcond`, `coagrate`, `modpar` | Next |
| P3 | Inorganic equilibrium (RPMARES) and SORGAM secondary organics | `rpmares_old`, `eql3`, `awater`, `sorgam`, `soa_part`, Newton solver | |
| P4 | SORGAM dry deposition and settling, PM sums, initial conditions | `sorgam_depdriver`, `VDVG`, `sum_pm_sorgam`, `aerosols_sorgam_init` | |
| P5 | Photolysis for RADM2 (Fast-TUV, `phot_opt = 3`) | `module_ftuv_*.F` | |
| P6 | Emissions for the RADM2/SORGAM species; namelist acceptance of `chem_opt = 1, 2` | `emissions_driver.F` mapping | |
| P7 | Opt-in `optional/wrfchem_pnnl/`: `phys/module_mixactivate.F` and the modal optics (`optical_driver.F`, `module_optical_averaging.F`, `module_fastj_mie.F`), ported exactly and graded bitwise | PNNL-terms files | |
| P8 | Clean-room aerosol activation in the default build (Abdul-Razzak and Ghan 2000; Abdul-Razzak et al. 1998), validated against P7 within stated tolerances | the papers | |
| P9 | Clean-room modal optics in the default build (volume-mixing and Maxwell-Garnett refractive-index averaging, Mie from Bohren and Huffman 1983), validated against P7 within stated tolerances | the papers | |
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

## Refusals until then

`woof/namelist_chem.py` refuses `chem_opt = 1, 2, 41` as "being ported",
pointing here, and refuses every CBM-Z/MOSAIC package by name (including
`chem_opt = 120`, CBMZ_BB_KPP, which shares CBM-Z's emission and
initial-condition modules), citing the
terms and the planned opt-in.  Every other uncarried package keeps the
generic refusal.
