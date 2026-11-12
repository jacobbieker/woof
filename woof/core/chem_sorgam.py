"""Pure launcher for WRF-Chem's MADE/SORGAM modal aerosol (chem_opt = 2).

``launch_sorgam`` is one sorgam_driver call (chem/module_aerosols_sorgam.F,
WRF v4.7.1) on every cell: new-particle formation, condensation,
coagulation, inorganic equilibrium and SORGAM secondary organics, with
WRF's own unit conversions on the way in and out.  It updates the aerosol,
gas and non-advected rows in place.

The default build runs SORGAM's RPMARES equilibrium (aeroproc's isorop = 0
branch; WRF v4.7.1 hard-codes isorop = 1, which calls ISORROPIA, copyright
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
#: Each aerosol row's cblk / vgsa slot (module_data_sorgam vso4aj = 1 ...
#: vcorn = 38): the package order and SORGAM's own coincide.
_AER_V = tuple(range(1, len(AEROSOLS) + 1))
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


def launch_sorgam_drydep(aerosol, gas, aux, t_phy, p_phy, alt, ust, rmol, znt, pbl,
                         aer_res, *, vdvg_2=True, cw_phase=NO_CLOUD_PHASE, vgsa=None):
    """sorgam_depdriver: the 38 aerosol rows' dry deposition velocities (m/s).

    Reads the lowest level of the 3-D (nz, ny, nx) fields; ust, rmol, znt,
    pbl and aer_res are (ny, nx).  vdvg_2=True is aer_drydep_opt = 1
    (VDVG_2), False is 11 (VDVG).  Returns 38 (ny, nx) arrays in
    :data:`AEROSOLS` order (``vgsa`` to write into existing ones).
    """
    import cupy as cp
    from woof.core.kernels import get_kernel
    shape, _ = columns(list(aerosol) + list(gas) + list(aux))
    for label, a in (("t_phy", t_phy), ("p_phy", p_phy), ("alt", alt)):
        field(a, shape, label)
    plane = shape[1:]
    for label, a in (("ust", ust), ("rmol", rmol), ("znt", znt), ("pbl", pbl),
                     ("aer_res", aer_res)):
        field(a, plane, label)
    if vgsa is None:
        vgsa = [cp.empty(plane, dtype=cp.float32) for _ in AEROSOLS]
    for i, a in enumerate(vgsa):
        field(a, plane, f"vgsa[{i}]")
    ncol = int(np.prod(plane))
    get_kernel("chem_sorgam", "chem_sorgam_drydep")(
        ((ncol + 63) // 64,), (64,),
        (_table(aerosol), _table(gas), _table(aux), t_phy, p_phy, alt, ust, rmol, znt,
         pbl, aer_res, _table(vgsa), np.int32(ncol), np.int32(bool(vdvg_2)),
         np.int32(cw_phase)))
    return vgsa


def launch_sorgam_sum_pm(aerosol, h2oaj, h2oai, alt, *, out=None):
    """sum_pm_sorgam: (pm2_5_dry, pm2_5_water, pm2_5_dry_ec, pm10), ug/m3."""
    import cupy as cp
    from woof.core.kernels import get_kernel
    shape, _ = columns(list(aerosol))
    for label, a in (("h2oaj", h2oaj), ("h2oai", h2oai), ("alt", alt)):
        field(a, shape, label)
    if out is None:
        out = tuple(cp.empty(shape, dtype=cp.float32) for _ in range(4))
    for i, a in enumerate(out):
        field(a, shape, f"out[{i}]")
    ncell = int(np.prod(shape))
    pm25, water, ec, pm10 = out
    get_kernel("chem_sorgam", "chem_sorgam_sum_pm")(
        ((ncell + 127) // 128,), (128,),
        (_table(aerosol), h2oaj, h2oai, alt, pm25, water, ec, pm10, np.int32(ncell)))
    return out


AER_IC_DEFAULT, AER_IC_PNNL = 1, 101


def launch_sorgam_ic(aerosol, sulf, convfac, z_at_w, *, aer_ic_opt=AER_IC_DEFAULT):
    """aerosols_sorgam_init's cold start: the 38 aerosol rows (ug/kg, #/kg)
    and, for AER_IC_DEFAULT, the gas sulfate that moves into them (sulf,
    ppmv, updated in place).  convfac is the p/(R T) WRF hands the routine;
    z_at_w has nz + 1 w levels (m).  Every mass level is set."""
    import cupy as cp
    from woof.core.kernels import get_kernel
    if aer_ic_opt not in (AER_IC_DEFAULT, AER_IC_PNNL):
        raise ValueError("aer_ic_opt must be 1 (AER_IC_DEFAULT) or 101 (AER_IC_PNNL)")
    shape, _ = columns(list(aerosol) + [sulf])
    field(convfac, shape, "convfac")
    field(z_at_w, (shape[0] + 1,) + shape[1:], "z_at_w")
    nz, nc = shape[0], shape[1] * shape[2]
    fatal = cp.zeros(1, dtype=cp.int32)
    get_kernel("chem_sorgam", "chem_sorgam_ic")(
        ((nz * nc + 127) // 128,), (128,),
        (_table(aerosol), sulf, convfac, z_at_w, np.int32(nz), np.int32(nc),
         np.int32(aer_ic_opt), fatal))
    if int(fatal.get()[0]):
        raise SorgamFatal("aerosols_sorgam_init: unable to parse aer_ic_opt")


# ---------------------------------------------------------------------------
# The process (``aerosol.sorgam``): the chem driver's view.
# ---------------------------------------------------------------------------
#
# Column-based: every launch above treats the (nz, ny, nx) fields as
# independent cells or columns, so the hex route can hand (nz, 1, ncells)
# views of the same rows and allocations.

from woof.core.chem_context import ChemAllocation  # noqa: E402

KEY = "aerosol.sorgam"
LEDGER = "chemistry"
KERNEL_MODULES = ("chem_sorgam",)
REQUIRES = ("t_phy", "p_phy", "qv", "alt", "z_at_w")
#: aerosol_driver's floor on the Aitken number before sorgam_driver
#: (aerosol_driver.F:229-237; also chem_driver.F:705-715).
NU0_FLOOR = np.float32(1.e7)
#: module_data_sorgam epsilc and rgasuniv (J mol-1 K-1, the convfac start_em
#: hands aerosols_sorgam_init: dyn_em/start_em.F:1812-1813).
EPSILC = np.float32(1.e-16)
RGASUNIV = np.float32(8.314)
RADM2_KEY = "chem.radm2"
#: WRF's PM output names (Registry state pm2_5_dry ... pm10, sum_pm_driver).
PM_OUTPUTS = (("pm2_5_dry", "PM2_5_DRY", "dry PM2.5"),
              ("pm2_5_water", "PM2_5_WATER", "PM2.5 aerosol water"),
              ("pm2_5_dry_ec", "PM2_5_DRY_EC", "dry PM2.5 elemental carbon"),
              ("pm10", "PM10", "PM10"))
ALLOCATES = (
    *(ChemAllocation(f"sorgam_{name}", "3d",
                     description=f"SORGAM's non-advected {name} (Registry "
                                 "state, chem/aerosol_driver.F), carried "
                                 "between chem steps and by a restart")
      for name in AUX),
    *(ChemAllocation(f"sorgam_{name}", "3d", restart="rebuild", units="ug m-3",
                     output_name=output, description=f"{text} (sum_pm_sorgam, "
                     "chem/module_aerosols_sorgam.F), every history frame")
      for name, output, text in PM_OUTPUTS),
)


def rows(table):
    return table.rows_for(KEY)


def _table_refusal(table):
    names = {row.name for row in rows(table)}
    missing = [name for name in AEROSOLS if name not in names]
    if missing:
        return (f"sorgam_driver updates all {len(AEROSOLS)} SORGAM aerosol "
                f"rows, and {missing} do not name aerosol.sorgam")
    gas = {row.name for row in table.rows_for(RADM2_KEY)}
    if not set(GASES) <= gas:
        return (f"SORGAM exchanges {list(GASES)} with the RADM2 gas phase "
                "and reads its precursor increments, so it runs only beside "
                "chem.radm2 with those rows")
    return None


def refusal(cfg):
    from woof.chem_table import load

    table = load(cfg)
    return None if table is None else _table_refusal(table)


def init(ctx):
    reason = _table_refusal(ctx.table)
    if reason:
        raise ValueError(reason)


def cold_start(ctx):
    """chemics_init's RADM2SORG start for chem_in_opt = 0
    (chem/chemics_init.F:1796-1828): aerosols_sorgam_init's cold start
    (AER_IC_DEFAULT: the gas sulfate moves into the two modes), the
    conversion of its ug m-3 to mixing ratios (``chem*alt``), then every
    package row floored at epsilc.  Runs after chem.radm2's gas profile, as
    chemics_init runs after real.exe."""
    import cupy as cp

    table = ctx.table
    aerosol = [ctx.field(table.row(name)) for name in AEROSOLS]
    t_phy = ctx.met("t_phy")
    p_phy = ctx.met("p_phy")
    convfac = cp.ascontiguousarray(
        (p_phy / RGASUNIV / t_phy).astype(cp.float32))
    launch_sorgam_ic(aerosol, ctx.field(table.row("sulf")), convfac,
                     ctx.met("z_at_w"))
    alt = ctx.met("alt")
    for a in aerosol:
        cp.multiply(a, alt, out=a)
    from woof.core.chem_radm2 import _package_rows

    for row in _package_rows(table):
        a = ctx.field(row)
        cp.maximum(a, EPSILC, out=a)


def step(ctx, dt, ktau):
    from woof.core.chem_radm2 import chem_step_clock
    import cupy as cp

    run, dtstepc = chem_step_clock(ctx, dt, ktau)
    if not run:
        return
    table = ctx.table
    aerosol = [ctx.field(table.row(name)) for name in AEROSOLS]
    nu0 = aerosol[AEROSOLS.index("nu0")]
    cp.maximum(nu0, NU0_FLOOR, out=nu0)
    gas = [ctx.field(table.row(name)) for name in GASES]
    aux = [ctx.diag[f"sorgam_{name}"] for name in AUX]
    vdrog = [ctx.diag[f"radm2_vdrog{n + 1:02d}"] for n in range(LDROG)]
    launch_sorgam(aerosol, gas, aux, vdrog, ctx.met("t_phy"), ctx.met("p_phy"),
                  ctx.met("qv"), ctx.met("alt"), ctx.diag["radm2_vcsulf_old"],
                  dtstepc)


def refresh_output(state):
    """sum_pm_driver every step (chem_driver.F:1696-1700), into the
    PM2_5_DRY ... PM10 outputs."""
    from woof.core.chem_state import process_attr

    chem = state.chem
    table = chem.table
    prep = getattr(chem, "prep", None)
    if prep is None or prep.get("alt") is None:
        return
    by_name = {a.name: a for a in ALLOCATES}
    aerosol = [getattr(state, table.row(name).state_attr) for name in AEROSOLS]
    out = tuple(getattr(state, process_attr(by_name[f"sorgam_{name}"]))
                for name, _o, _t in PM_OUTPUTS)
    launch_sorgam_sum_pm(aerosol,
                         getattr(state, process_attr(by_name["sorgam_h2oaj"])),
                         getattr(state, process_attr(by_name["sorgam_h2oai"])),
                         prep.get("alt"), out=out)
