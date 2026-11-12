"""Pure launcher for WRF-Chem's RADM2 gas-phase chemistry (chem_opt = 1, 2).

``launch_radm2`` is radm_driver (chem/module_radm.F, WRF v4.7.1) for one
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


# ---------------------------------------------------------------------------
# The process (``chem.radm2``): the chem driver's view.
# ---------------------------------------------------------------------------
#
# Column-based and mesh-agnostic: every launch here and in the kernels it
# calls treats the (nz, ny, nx) fields as ny*nx independent columns, so the
# hex route can hand the same module (nz, 1, ncells) views.

from woof.core.chem_context import ChemAllocation  # noqa: E402

KEY = "chem.radm2"
LEDGER = "chemistry"
#: The kernel translation units step() launches (priced by the preflight);
#: so2so4 is an inline elementwise kernel.
KERNEL_MODULES = ("chem_radm2",)
REQUIRES = ("t_phy", "p_phy", "qv", "z_at_w")
#: module_data_radm2.F epsilc: radm_driver's floor (module_radm.F:98) and
#: mechanism_driver's vcsulf_old floor (mechanism_driver.F:130-131).
EPSILC = np.float32(1.e-16)
#: The aerosol process whose rows make the package RADM2SORG (chem_opt = 2).
SORGAM_KEY = "aerosol.sorgam"
PHOTOLYSIS_KEY = "photolysis.ftuv"
#: The sets of the two RADM2 packages (woof/data/chem/sets).
RADM2_SETS = ("radm2", "radm2sorg")
ALLOCATES = (
    ChemAllocation("radm2_last_chem_secs", "2d", dtype="float64",
                   description="WRF's last_chem_time as seconds since the "
                               "start (chem_driver.F:366-417; chemics_init.F"
                               ":2070-2073 starts it at -dt), every element "
                               "equal; carried by a restart"),
    ChemAllocation("radm2_vcsulf_old", "3d", restart="rebuild", units="ppmv",
                   description="sulfate before this chem step's RADM2 call "
                               "(mechanism_driver.F:128-131), read by SORGAM"),
    *(ChemAllocation(f"radm2_vdrog{n + 1:02d}", "3d", restart="rebuild",
                     units="ppmv",
                     description="RADM2's organic aerosol-precursor increment "
                                 f"vdrog3(:,:,:,{n + 1}) (module_radm.F), "
                                 "zeroed every chem step (chem_driver.F:731)")
      for n in range(LDROG)),
)


def rows(table):
    return table.rows_for(KEY)


def _package_rows(table):
    """The rows radm_driver floors with its own species (module_radm.F:98):
    every chem.radm2 and aerosol.sorgam row, in table order."""
    seen, out = set(), []
    for row in (*table.rows_for(KEY), *table.rows_for(SORGAM_KEY)):
        if row.name not in seen:
            seen.add(row.name)
            out.append(row)
    return out


def _table_refusal(table, cfg):
    names = {row.name for row in rows(table)}
    missing = [name for name in SPECIES if name not in names]
    if missing:
        return (f"the RADM2 mechanism needs all {len(SPECIES)} of its gas rows "
                f"to name chem.radm2, and {missing} do not, so their reactions "
                "would read absent species")
    if not table.rows_for(PHOTOLYSIS_KEY):
        return ("RADM2 reads its photolysis rates from photolysis.ftuv, which "
                "no active row names, so every photolytic reaction would run "
                "at zero")
    others = [name for name in table.sets if name not in RADM2_SETS]
    if others:
        return (f"chem_sets {list(table.sets)} combine a RADM2 package with "
                f"{others}; WRF runs one chem_opt package, and a second set's "
                "processes and table diagnostics would read RADM2's species "
                "with another package's meaning (sulf is RADM2's gas-phase "
                "H2SO4, GOCART's particulate sulfate)")
    if table.rows_for("chem.sulfur"):
        return ("RADM2 and GOCART's sulfur mechanism both act on so2 and "
                "sulf in this selection; one WRF chem_opt carries one "
                "mechanism, so combine the RADM2 sets with no GOCART set")
    if int(getattr(cfg, "cu_diag", 0) or 0) == 1 and table.rows_for(SORGAM_KEY):
        return ("cu_diag = 1 makes WRF's so2so4 read the cumulus cloud "
                "fraction gd_cldfr, which no cumulus scheme here publishes")
    return None


def refusal(cfg):
    from woof.chem_table import load

    table = load(cfg)
    return None if table is None else _table_refusal(table, cfg)


def _wrf_due(ktau, curr_secs, dt, every_min):
    """chem_driver.F:349-362 and 393-412 (the adaptive branch WRF always
    takes on the EM core): ktau 1, or ``every_min <= 0``, or the next
    multiple of the period is reached within this step.  ``every_min`` is a
    REAL namelist value; ``every_min*60.`` is formed in REAL before
    ``real(...,8)``."""
    if ktau == 1:
        return True
    every = np.float32(every_min)
    if every <= 0:
        return True
    period = np.float64(np.float32(every * np.float32(60.0)))
    curr = np.float64(curr_secs)
    reach = np.float64(int(curr / period + np.float64(1.0))) * period
    return bool(curr + np.float64(np.float32(dt)) + np.float64(0.01) >= reach)


def chem_step_clock(ctx, dt, ktau):
    """``(run, dtstepc)`` for this step's mechanism and aerosol calls.

    do_chemstep and dtstepc as chem_driver computes them
    (chem_driver.F:366-417): dtstepc is grid%dt at ktau 1, else the time
    since last_chem_time, which only the adaptive branch (ktau > 1) moves
    -- so the second chem step spans two steps when ``chemdt <= 0``, as in
    WRF.  Decided once per step and cached for aerosol.sorgam.
    """
    chem = ctx.state.chem
    cached = getattr(chem, "radm2_chem_clock", None)
    if cached is not None and cached[0] == ktau:
        return cached[1], cached[2]
    curr = np.float64(ctx.clock.curr_secs)
    last_arr = ctx.diag["radm2_last_chem_secs"]
    last = getattr(chem, "radm2_last_chem_host", None)
    if last is None:
        import cupy as cp
        last = float(cp.asnumpy(last_arr.reshape(-1)[:1])[0])
    if ktau == 1:
        dtstepc = np.float32(dt)
    else:
        dtstepc = np.float32(curr - np.float64(last))
    run = _wrf_due(ktau, curr, dt, getattr(ctx.cfg, "chemdt", 0.0))
    if run and ktau != 1:
        last = float(curr)
        last_arr[...] = last
    chem.radm2_last_chem_host = last
    chem.radm2_chem_clock = (ktau, run, float(dtstepc))
    return run, float(dtstepc)


def init(ctx):
    reason = _table_refusal(ctx.table, ctx.cfg)
    if reason:
        raise ValueError(reason)


def cold_start(ctx):
    """real.exe's chem_in_opt = 0 start for the RADM2 gases
    (module_input_chem_data.F input_chem_profile -> make_chem_profile), on
    the rows no enabled data-store source filled at initialisation (a CAMS
    row keeps CAMS's analysis, as WRF-Chem's mozbc overwrites the profile),
    and WRF's initial last_chem_time = start - dt (chemics_init.F:2070)."""
    import cupy as cp
    from woof.core.chem_gas_profile import radm2_background_profile

    ctx.diag["radm2_last_chem_secs"][...] = -float(ctx.cfg.dt)
    ctx.state.chem.radm2_last_chem_host = -float(ctx.cfg.dt)
    enabled = set(getattr(ctx.table, "enabled_sources", ()))
    targets = [row for row in rows(ctx.table)
               if not any(ref["source"] in enabled for ref in row.boundary)]
    if not targets:
        return
    chem_opt = 2 if ctx.table.rows_for(SORGAM_KEY) else 1
    profile = radm2_background_profile(cp.asnumpy(ctx.met("z_at_w")),
                                       chem_opt=chem_opt)
    for row in targets:
        if row.name not in profile:
            raise ValueError(f"chem.radm2 row {row.name!r} has no WRF "
                             "background profile to start from")
        ctx.field(row)[...] = cp.asarray(profile[row.name], dtype=cp.float32)


def boundary_inflow(table, z, alt, convfac):
    """WRF's lateral inflow for the RADM2 gases without an enabled boundary
    source (have_bcs_chem false: flow_dep_bdy_chem's i_bdy_method 1,
    bdy_chem_value at the inflow cell's mass-level height z), ppmv.
    Any array shape; alt and convfac are unused (the routine reads none)."""
    from woof.core.chem_gas_profile import radm2_boundary_value

    del alt, convfac
    enabled = set(getattr(table, "enabled_sources", ()))
    names = [row.name for row in rows(table)
             if not any(ref["source"] in enabled for ref in row.boundary)]
    if not names:
        return {}
    chem_opt = 2 if table.rows_for(SORGAM_KEY) else 1
    values = radm2_boundary_value(z, chem_opt=chem_opt)
    return {name: values[name] for name in names}


_SO2SO4 = None


def _so2so4_kernel():
    """chem/module_gocart_so2so4.F so2so4 for ils = 0 without the cumulus
    cloud fraction (gd_on = 0): the cell is cloudy when qc > 0, in the
    Fortran's order of operations, every rounding explicit."""
    global _SO2SO4
    if _SO2SO4 is None:
        import cupy as cp
        _SO2SO4 = cp.ElementwiseKernel(
            "float32 qc, float32 t, int32 has_qc",
            "float32 so2, float32 sulf, float32 h2o2",
            """
            float cldf = 0.f;
            if (has_qc && qc > 0.f) cldf = 1.f;
            float tc2 = so2;
            if (cldf > 0.f && tc2 > 0.f && t > 258.f) {
                float tc3 = sulf;
                float hp = h2o2;
                if (tc2 > hp) {
                    cldf = __fmul_rn(cldf, __fdiv_rn(hp, tc2));
                    hp = __fmul_rn(hp, __fsub_rn(1.f, cldf));
                } else {
                    hp = __fmul_rn(hp, __fsub_rn(1.f, __fdiv_rn(__fmul_rn(cldf, tc2), hp)));
                }
                so2 = fmaxf(1.e-16f, __fmul_rn(tc2, __fsub_rn(1.f, cldf)));
                sulf = fmaxf(1.e-16f, __fadd_rn(tc3, __fmul_rn(tc2, cldf)));
                h2o2 = fmaxf(1.e-16f, hp);
            }
            """, "woof_radm2sorg_so2so4")
    return _SO2SO4


def launch_so2so4(so2, sulf, h2o2, t_phy, qc=None):
    """RADM2SORG's in-cloud SO2 -> sulfate by H2O2 (chem_driver.F:1350-1360
    -> module_gocart_so2so4.F), in place on every cell."""
    import cupy as cp
    shape, _ = columns([so2, sulf, h2o2])
    field(t_phy, shape, "t_phy")
    if qc is not None:
        field(qc, shape, "qc")
    has_qc = np.int32(qc is not None)
    _so2so4_kernel()(qc if qc is not None else t_phy, t_phy, has_qc,
                     so2, sulf, h2o2)


def step(ctx, dt, ktau):
    run, dtstepc = chem_step_clock(ctx, dt, ktau)
    if not run:
        return
    table = ctx.table
    sorgam = bool(table.rows_for(SORGAM_KEY))
    species = [ctx.field(table.row(name)) for name in SPECIES]
    clamp = [ctx.field(row) for row in _package_rows(table)
             if row.name not in SPECIES]
    photolysis = {name: ctx.diag[f"ph_{name}"] for name in PHOTOLYSIS}
    vdrog = None
    if sorgam:
        vdrog = [ctx.diag[f"radm2_vdrog{n + 1:02d}"] for n in range(LDROG)]
        for a in vdrog:
            a[...] = 0
        import cupy as cp
        cp.maximum(ctx.field(table.row("sulf")), EPSILC,
                   out=ctx.diag["radm2_vcsulf_old"])
    launch_radm2(species, ctx.met("t_phy"), ctx.met("p_phy"), ctx.met("qv"),
                 photolysis, dtstepc, float(ctx.clock.curr_secs),
                 vdrog=vdrog, clamp=clamp)
    if sorgam:
        qc = ctx.met("qc") if ctx.has("qc") else None
        launch_so2so4(ctx.field(table.row("so2")), ctx.field(table.row("sulf")),
                      ctx.field(table.row("h2o2")), ctx.met("t_phy"), qc)
