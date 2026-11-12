"""Pure launcher for WRF-Chem's RADM2/SORGAM anthropogenic emissions
(chem_opt = 1/2, emiss_opt = 3).

``launch_radm2_emiss`` is what emissions_driver (chem/emissions_driver.F,
WRF v4.7.1) does with ``emis_ant`` for RADM2 and RADM2SORG in one chemistry
call: add_anthropogenics (chem/module_emissions_anthropogenics.F) adds the
gases, in mol km-2 hr-1, through ``conv_rho = 4.828e-4/rho_phy*dtstep/
(dz8w*60.)``; for RADM2SORG, sorgam_addemiss (chem/module_aerosols_sorgam.F,
the ``emissopt /= 5`` branch with no biomass burning, sea salt or dust) adds
the aerosol mass rows and the three number rows from the ug m-2 s-1 rates
through ``factor = alt*dtstep/dz8w``.  Kernel:
``woof/core/kernels/chem_radm2_emiss.cu``.

The species pairing is WRF's, not WOOF's: :data:`GAS_TERMS` lists, in
add_anthropogenics' order, every ``p_x = p_x + p_e_y`` line of its RADM2
branch whose chem row (Registry package radm2 / radm2sorg) and emission row
(package eradmsorg, Registry/registry.chem:4055) both exist.  The lines it
leaves out add to WRF's index-1 dummy (``p_ch3cl``) or are skipped by WRF's
own ``>= param_first_scalar`` guards (ch4, ete, api/lim, co2).
"""
from __future__ import annotations

import numpy as np

from woof.core.chem_column import columns, field, timestep

#: (chem row, emis_ant row) in add_anthropogenics' order
#: (module_emissions_anthropogenics.F:75-81, 199-228).
GAS_TERMS = (("so2", "e_so2"), ("co", "e_co"), ("no", "e_no"), ("no2", "e_no2"),
             ("nh3", "e_nh3"), ("hcl", "e_hcl"),
             ("csl", "e_csl"), ("iso", "e_iso"), ("ald", "e_ald"), ("hcho", "e_hcho"),
             ("ora2", "e_ora2"), ("hc3", "e_hc3"), ("hc5", "e_hc5"), ("hc8", "e_hc8"),
             ("eth", "e_eth"), ("olt", "e_olt"), ("oli", "e_oli"), ("tol", "e_tol"),
             ("xyl", "e_xyl"), ("ket", "e_ket"), ("ol2", "e_ol2"), ("hono", "e_hono"))
#: The emis_ant rows sorgam_addemiss reads, in the kernel's frame order.
AEROSOL_EMISSIONS = ("e_pm25i", "e_pm25j", "e_pm_10", "e_eci", "e_ecj", "e_orgi",
                     "e_orgj", "e_so4i", "e_so4j", "e_no3i", "e_no3j", "e_naaj",
                     "e_naai")
#: The chem rows sorgam_addemiss adds to, in the kernel's order.
AEROSOL_ROWS = ("nu0", "ac0", "corn", "antha", "p25j", "p25i", "ecj", "eci", "naaj",
                "naai", "orgpaj", "orgpai", "so4aj", "so4ai", "no3aj", "no3ai")
#: Every eradmsorg row WOOF reads (the others -- e_ch4, e_terp, e_co2, e_ch3cl,
#: e_cli, e_clj and the e_org*_a/_bb splits -- reach no RADM2/SORGAM row).
EMISSIONS = tuple(e for _, e in GAS_TERMS) + AEROSOL_EMISSIONS

#: module_data_sorgam's PARAMETERs anthfac, orgfac, so4fac, no3fac
#: (module_data_sorgam.F:690-711, folded by gfortran in REAL(4)) and the
#: numbers aerosols_sorgam_init computes (module_aerosols_sorgam.F:7484-7486),
#: as float32 words; tests/test_chem_radm2_emiss_wrf471_parity.py checks each
#: against the oracle's.
SORGAM_FACTORS = {
    "factnumn": 0x65DEC2C4, "factnuma": 0x618B8FEE, "factnumc": 0x5B86E669,
    "anthfac": 0x2B745A7B, "orgfac": 0x2C0664F7, "so4fac": 0x2B9553BD,
    "no3fac": 0x2B9553BD,
}


def factor_values() -> dict[str, np.float32]:
    return {k: np.uint32(v).view(np.float32) for k, v in SORGAM_FACTORS.items()}


def _table(arrays):
    import cupy as cp
    return cp.asarray(np.array([a.data.ptr for a in arrays], dtype=np.uint64))


def launch_radm2_emiss(gas, gas_emis, rho_phy, alt, dz8w, dtstep, *, kemit,
                       aerosol=None, aerosol_emis=None):
    """Add one step's anthropogenic emissions in place.

    gas: the chem rows of :data:`GAS_TERMS` (ppmv), gas_emis their emission
    frames (mol km-2 hr-1).  aerosol / aerosol_emis: None for chem_opt = 1,
    else the rows of :data:`AEROSOL_ROWS` (ug kg-1, numbers kg-1) and the
    frames of :data:`AEROSOL_EMISSIONS` (ug m-2 s-1).  Every row is float32
    (nz, ny, nx); every frame float32 (nlev, ny, nx) or (ny, nx), reading 0
    above its top level.  kemit: WRF's emission depth in levels; the gases
    stop at min(kemit, nz), and sorgam_addemiss, which WRF runs on
    kts .. kemit unclipped, needs kemit <= nz.
    """
    import cupy as cp
    from woof.core.kernels import get_kernel
    if len(gas) != len(GAS_TERMS) or len(gas_emis) != len(GAS_TERMS):
        raise ValueError(f"RADM2 emissions need {len(GAS_TERMS)} gas rows and frames")
    if (aerosol is None) != (aerosol_emis is None):
        raise ValueError("aerosol rows and aerosol frames come together")
    rows = list(gas) + list(aerosol or ())
    shape, _ = columns(rows)
    nz, ny, nx = shape
    for label, a in (("rho_phy", rho_phy), ("alt", alt), ("dz8w", dz8w)):
        field(a, shape, label)
    kemit = int(kemit)
    if kemit < 1:
        raise ValueError("kemit must be at least 1")
    if aerosol is not None:
        if len(aerosol) != len(AEROSOL_ROWS) or len(aerosol_emis) != len(AEROSOL_EMISSIONS):
            raise ValueError(f"SORGAM emissions need {len(AEROSOL_ROWS)} rows and "
                             f"{len(AEROSOL_EMISSIONS)} frames")
        if kemit > nz:
            raise ValueError(f"kemit = {kemit} exceeds the {nz} model levels; WRF's "
                             "sorgam_addemiss would index past the column")
    frames = []
    for i, e in enumerate(list(gas_emis) + list(aerosol_emis or ())):
        if e.ndim == 2:
            if e.shape != (ny, nx):
                raise ValueError(f"emission frame {i} must be ({ny}, {nx}), got {e.shape}")
            e = e.reshape(1, ny, nx)
        if e.dtype != cp.float32 or e.ndim != 3 or e.shape[1:] != (ny, nx) \
                or not e.flags.c_contiguous:
            raise ValueError(f"emission frame {i} must be C-contiguous float32 (nlev, {ny}, {nx})")
        frames.append(e)
    dtstep = timestep(dtstep)
    f = factor_values()
    ncol = ny * nx
    threads = 128
    get_kernel("chem_radm2_emiss", "chem_radm2_emiss")(
        ((ncol + threads - 1) // threads,), (threads,),
        (rho_phy, alt, dz8w, _table(gas), _table(frames),
         cp.asarray(np.array([e.shape[0] for e in frames], np.int32)),
         np.int32(len(gas)),
         _table(aerosol) if aerosol is not None else _table(gas[:1]),
         np.int32(aerosol is not None),
         f["factnumn"], f["factnuma"], f["factnumc"], f["anthfac"], f["orgfac"],
         f["so4fac"], f["no3fac"], np.float32(dtstep), np.int32(kemit), np.int32(nz),
         np.int64(ncol)))


# ---------------------------------------------------------------------------
# The process (``emission.radm2``): the chem driver's view.
# ---------------------------------------------------------------------------
#
# Column-based: the kernel runs one thread per column over ny*nx, so the hex
# route can hand (nz, 1, ncells) views and frames on the cells.

from woof.core.chem_context import ChemAllocation  # noqa: E402

KEY = "emission.radm2"
LEDGER = "emitted"
KERNEL_MODULES = ("chem_radm2_emiss",)
REQUIRES = ("rho", "alt", "dz8w")
#: module_data_radm2.F epsilc: chem_driver's chem_minval
#: (chem_driver.F:454, 693-701).
EPSILC = np.float32(1.e-16)
#: chem_driver.F:705-715: RADM2SORG's floor on the Aitken number.
NU0_FLOOR = np.float32(1.e7)
SORGAM_KEY = "aerosol.sorgam"
#: The level-1 emis_ant planes vertical mixing reads for WRF's urban mixing
#: enhancement (dry_dep_driver.F:672-690): e_co, e_pm25i, e_pm25j.
PUBLISHED = (("anth_co_kts", "e_co"), ("anth_pm25i_kts", "e_pm25i"),
             ("anth_pm25j_kts", "e_pm25j"))
ALLOCATES = tuple(
    ChemAllocation(name, "2d", restart="rebuild", units="emis_ant units",
                   description=f"this step's {e} at the lowest level, for "
                               "vertical mixing (dry_dep_driver.F:672-690)")
    for name, e in PUBLISHED)


def rows(table):
    return table.rows_for(KEY)


def _valid_time(ctx):
    from datetime import timedelta

    start = (getattr(ctx.cfg, "start_time", None)
             or getattr(ctx.state.chem, "start_time", None))
    if start is None:
        raise ValueError("emission.radm2 needs the run's start time to pick "
                         "an emission frame")
    return start + timedelta(seconds=float(ctx.clock.curr_secs))


def emission_frames(ctx):
    """``{e_name: frame}`` from the run's enabled inventory, or None when the
    run enables none (WRF's emiss_opt = 0: nothing is emitted).

    The inventory module maps a source's fields onto WRF's emis_ant names
    in WRF's units (woof.chem_inventory_frames)."""
    from woof.chem_table import catalog

    sources = catalog().sources
    if not any(sources[name].kind == "emission"
               for name in getattr(ctx.table, "enabled_sources", ())
               if name in sources):
        return None
    from woof.chem_inventory_frames import radm2_frames_for_context

    return radm2_frames_for_context(ctx, _valid_time(ctx))


def init(ctx):
    names = {row.name for row in rows(ctx.table)}
    missing = [chem for chem, _e in GAS_TERMS if chem not in names]
    if ctx.table.rows_for(SORGAM_KEY):
        missing += [chem for chem in AEROSOL_ROWS if chem not in names]
    if missing:
        raise ValueError(f"emission.radm2: rows {missing} of the package do "
                         "not name the process, so their emissions would "
                         "be dropped")


def step(ctx, dt, ktau):
    import cupy as cp

    table = ctx.table
    active = rows(table)
    for row in active:
        a = ctx.field(row)
        cp.maximum(a, EPSILC, out=a)
    sorgam = bool(table.rows_for(SORGAM_KEY))
    if sorgam:
        nu0 = ctx.field(table.row("nu0"))
        cp.maximum(nu0, NU0_FLOOR, out=nu0)
    frames = emission_frames(ctx)
    if frames is None:
        for name, _e in PUBLISHED:
            ctx.diag[name][...] = 0
        return
    gas = [ctx.field(table.row(chem)) for chem, _e in GAS_TERMS]
    gas_emis = [frames[e] for _c, e in GAS_TERMS]
    aerosol = aerosol_emis = None
    if sorgam:
        aerosol = [ctx.field(table.row(chem)) for chem in AEROSOL_ROWS]
        aerosol_emis = [frames[e] for e in AEROSOL_EMISSIONS]
    launch_radm2_emiss(gas, gas_emis, ctx.met("rho"), ctx.met("alt"),
                       ctx.met("dz8w"), float(dt),
                       kemit=int(getattr(ctx.cfg, "kemit", 9)),
                       aerosol=aerosol, aerosol_emis=aerosol_emis)
    for name, e in PUBLISHED:
        frame = frames[e]
        ctx.diag[name][...] = frame if frame.ndim == 2 else frame[0]
