"""Pure launcher for WRF-Chem's MADE/SORGAM modal aerosol (chem_opt = 2).

``launch_sorgam`` is one sorgam_driver call (chem/module_aerosols_sorgam.F,
WRF v4.6.1) on every cell: new-particle formation, condensation,
coagulation, inorganic equilibrium and SORGAM secondary organics, with
WRF's own unit conversions on the way in and out.  It updates the aerosol,
gas and non-advected rows in place.

The default build runs SORGAM's RPMARES equilibrium (aeroproc's isorop = 0
branch; WRF v4.6.1 hard-codes isorop = 1, which calls ISORROPIA, copyright
with no grant) and WOOF's clean-room solver for SORGAM's SOA partitioning
(WRF's is derived from Numerical Recipes, which forbids redistribution).
``exact=True`` runs WRF's own solver instead, built locally from the user's
WRF tree (woof.core.chem_byo); nothing derived from those routines ships
with WOOF.  docs/chem-coupled-aerosol-roadmap.md.
"""
from __future__ import annotations

import numpy as np

from woof.core.chem_column import columns, field, timestep

#: The radm2sorg aerosol rows in package order (Registry/registry.chem:3930):
#: ug/kg dry air, numbers (nu0, ac0, corn) in #/kg.
AEROSOLS = ("so4aj", "so4ai", "nh4aj", "nh4ai", "no3aj", "no3ai", "naaj", "naai",
            "claj", "clai", "orgaro1j", "orgaro1i", "orgaro2j", "orgaro2i",
            "orgalk1j", "orgalk1i", "orgole1j", "orgole1i", "orgba1j", "orgba1i",
            "orgba2j", "orgba2i", "orgba3j", "orgba3i", "orgba4j", "orgba4i",
            "orgpaj", "orgpai", "ecj", "eci", "p25j", "p25i", "antha", "seas",
            "soila", "nu0", "ac0", "corn")
#: The gas rows SORGAM exchanges with (ppmv).
GASES = ("sulf", "hno3", "nh3", "hcl")
#: The non-advected aerosol state WRF keeps between calls (Registry state).
AUX = ("h2oaj", "h2oai", "nu3", "ac3", "cor3", "cvaro1", "cvaro2", "cvalk1",
       "cvole1", "cvapi1", "cvapi2", "cvlim1", "cvlim2")
#: module_data_sorgam ldrog: radm's organic aerosol-precursor increments.
LDROG = 17
#: cw_phase for a package without cloud-borne aerosol
#: (module_prep_wetscav_sorgam.F:37).
NO_CLOUD_PHASE = -999888777


class SorgamFatal(RuntimeError):
    """A cell reached one of WRF's wrf_error_fatal calls (or the excluded
    ISORROPIA branch); the value is the module_aerosols_sorgam.F line."""


def _table(arrays):
    import cupy as cp
    return cp.asarray(np.array([a.data.ptr for a in arrays], dtype=np.uint64))


def launch_sorgam(aerosol, gas, aux, vdrog, t_phy, p_phy, qv, alt, vcsulf_old,
                  dtstep, *, cw_phase=NO_CLOUD_PHASE, exact=False):
    """One sorgam_driver call on levels 0..nz-1 of every column, in place.

    aerosol: 38 rows in :data:`AEROSOLS` order; gas: 4 rows in :data:`GASES`
    order; aux: 13 rows in :data:`AUX` order; vdrog: 17 rows (radm2's
    precursor increments, launch_radm2's ``vdrog``); vcsulf_old: sulfate
    before this step's gas chemistry (ppmv).  t_phy K, p_phy Pa, qv kg/kg,
    alt m3/kg.  exact: WRF's own SOA solver from ``WOOF_WRF_SOURCE``
    (woof.core.chem_byo.unavailable_reason says why it cannot run).
    """
    import cupy as cp
    from woof.core.kernels import get_kernel
    for label, rows, n in (("aerosol", aerosol, len(AEROSOLS)), ("gas", gas, len(GASES)),
                           ("aux", aux, len(AUX)), ("vdrog", vdrog, LDROG)):
        if len(rows) != n:
            raise ValueError(f"{label} needs {n} rows, got {len(rows)}")
    shape, _ = columns(list(aerosol) + list(gas) + list(aux) + list(vdrog))
    for label, a in (("t_phy", t_phy), ("p_phy", p_phy), ("qv", qv), ("alt", alt),
                     ("vcsulf_old", vcsulf_old)):
        field(a, shape, label)
    dtstep = timestep(dtstep)
    nz = shape[0]
    ncell = int(np.prod(shape))
    fatal = cp.zeros(1, dtype=cp.int32)
    if exact:
        from woof.core.chem_byo import exact_sorgam_kernel
        kernel = exact_sorgam_kernel()
    else:
        kernel = get_kernel("chem_sorgam", "chem_sorgam")
    kernel(
        ((ncell + 63) // 64,), (64,),
        (_table(aerosol), _table(gas), _table(aux), _table(vdrog), t_phy, p_phy, qv,
         alt, vcsulf_old, np.int32(nz), np.int32(ncell), dtstep, np.int32(cw_phase),
         fatal))
    line = int(fatal.get()[0])
    if line:
        raise SorgamFatal(f"module_aerosols_sorgam.F:{line} reached on the device")
