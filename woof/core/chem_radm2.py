"""Pure launcher for WRF-Chem's RADM2 gas-phase chemistry (chem_opt = 1, 2).

``launch_radm2`` is radm_driver (chem/module_radm.F, WRF v4.6.1) for one
chemistry call: WRF's hand-coded QSSA solver, adaptive substeps chosen per
column, species in ppmv, photolysis rates in 1/min as WRF's ph_* fields
carry them.  It updates the species rows in place and, when ``vdrog`` is
given (the RADM2SORG coupling, ``p_nu0 > 1``), writes the 11 RADM2 organic
aerosol-precursor increments SORGAM reads.

The step limits are the driver's binary64 clock arithmetic
(module_radm.F:82, 174-179), evaluated here on the host in the same kinds and
rounded to float32 where WRF assigns to a REAL.
"""
from __future__ import annotations

import numpy as np

from woof.core.chem_column import columns, field

#: module_data_radm2.F species order (lso2 = 1 ... lho2 = 41).
SPECIES = ("so2", "sulf", "no2", "no", "o3", "hno3", "h2o2", "ald", "hcho",
           "op1", "op2", "paa", "ora1", "ora2", "nh3", "n2o5", "no3", "pan",
           "hc3", "hc5", "hc8", "eth", "co", "ol2", "olt", "oli", "tol", "xyl",
           "aco3", "tpan", "hono", "hno4", "ket", "gly", "mgly", "dcb", "onit",
           "csl", "iso", "ho", "ho2")
#: radm_driver's rj order (module_radm.F:190-212), as WRF's ph_<name> fields.
PHOTOLYSIS = ("no2", "o31d", "o33p", "hno2", "hno3", "hno4", "no3o2", "no3o",
              "h2o2", "ch2om", "ch2or", "ch3cho", "ch3o2h", "ch3coch3",
              "ch3coo2h", "ch3coc2h5", "hcocho", "hcochob", "ch3cocho",
              "hcochest", "ch3ono2")
#: module_data_sorgam.F ldrog; radm writes the first 11 (pxyl .. polt3).
LDROG = 17
MAX_LEVELS = 256


def step_limits(dtstep, curr_secs):
    """(dt60, dtcmin, dtcmax) in minutes, module_radm.F:82 and 174-179."""
    dtstep = np.float32(dtstep)
    curr_secs = np.float64(curr_secs)
    if not np.isfinite(dtstep) or not 0 < dtstep <= 86400:
        raise ValueError("dtstep must be finite and in (0, 86400] seconds")
    if not np.isfinite(curr_secs) or curr_secs < 0:
        raise ValueError("curr_secs must be finite and nonnegative")
    xtime = curr_secs / np.float64(60.0)
    dt60 = np.float32(dtstep / np.float32(60.0))
    xtimin = max(np.float64(0.0), xtime - np.float64(dt60))
    dtcmin = np.float32(min(np.float64(0.05), xtime - xtimin))
    dtcmin = max(dtcmin, np.float32(np.float32(0.5) / np.float32(60.0)))
    dtcmax = min(np.float32(5.0), dt60)
    dtcmax = np.float32(min(np.float64(dtcmax), xtime - xtimin))
    return dt60, np.float32(dtcmin), dtcmax


def _pointer_table(arrays):
    import cupy as cp
    return cp.asarray(np.array([a.data.ptr for a in arrays], dtype=np.uint64))


def launch_radm2(species, t_phy, p_phy, qv, photolysis, dtstep, curr_secs, *,
                 vdrog=None, clamp=()):
    """One radm_driver call on every column, levels 0..nz-1 (kts..kte).

    species: 41 (nz, ny, nx) float32 rows in :data:`SPECIES` order, ppmv.
    photolysis: mapping of :data:`PHOTOLYSIS` names to (nz, ny, nx) 1/min.
    vdrog: None (chem_opt = 1) or 17 (nz, ny, nx) rows, of which the first
    11 are overwritten (chem_opt = 2).  clamp: the package's other chem rows,
    which radm_driver floors at 1e-16 (module_radm.F:98) and nothing else.
    t_phy in K, p_phy in Pa, qv in kg/kg.
    """
    from woof.core.kernels import get_kernel
    if len(species) != len(SPECIES):
        raise ValueError(f"RADM2 needs {len(SPECIES)} species rows, got {len(species)}")
    shape, _ = columns(list(species) + list(clamp))
    nz, ny, nx = shape
    if nz > MAX_LEVELS:
        raise ValueError(f"RADM2 runs one thread per level; at most {MAX_LEVELS} levels")
    for label, a in (("t_phy", t_phy), ("p_phy", p_phy), ("qv", qv)):
        field(a, shape, label)
    missing = [n for n in PHOTOLYSIS if n not in photolysis]
    if missing:
        raise ValueError(f"photolysis rates missing: {', '.join(missing)}")
    ph = [field(photolysis[n], shape, f"ph_{n}") for n in PHOTOLYSIS]
    if vdrog is not None:
        if len(vdrog) != LDROG:
            raise ValueError(f"vdrog needs {LDROG} rows")
        for i, a in enumerate(vdrog):
            field(a, shape, f"vdrog[{i}]")
    dt60, dtcmin, dtcmax = step_limits(dtstep, curr_secs)
    rows = _pointer_table(list(species) + list(clamp))
    ph_table = _pointer_table(ph)
    vd_table = _pointer_table(vdrog) if vdrog is not None else np.uint64(0)
    nc = ny * nx
    threads = max(32, -(-nz // 32) * 32)
    get_kernel("chem_radm2", "chem_radm2")(
        (nc,), (threads,),
        (rows, np.int32(len(clamp)), t_phy, p_phy, qv, ph_table, vd_table,
         np.int32(nz), np.int32(nc), dt60, dtcmin, dtcmax))
