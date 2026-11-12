"""MADE/SORGAM aerosol dry deposition velocities (process ``drydep.sorgam``).

WRF-Chem's dry_dep_driver for the RADM2SORG package with aer_drydep_opt = 1
(chem/dry_dep_driver.F:441-534): sorgam_depdriver writes the deposition
velocities of the package's aerosol rows from the lowest layer's modal
state (VDVG_2 of chem/module_aerosols_sorgam.F), with the aerodynamic
resistance the Wesely gas pass computed for the same column (aer_res_def,
:457-464, aer_aerodynres_opt's default), then caps every aerosol velocity
at 0.5 m/s (:552-559).  The velocities go into ``ctx.ddvel``, which vertical
mixing reads as its lower boundary; this process moves no mass itself.

Cadence: WRF runs the whole dry_dep_driver only when ``vertmix_onoff > 0``
and ``ktau > 2`` (chem/chem_driver.F:1046-1047); so does this process.

Column-based: one thread per column over ny*nx, so the hex route can hand
(nz, 1, ncells) views.  Imports no CuPy until a launch.
"""
from __future__ import annotations

import numpy as np

from woof.core.chem_sorgam import AEROSOLS, AUX, GASES, launch_sorgam_drydep

KEY = "drydep.sorgam"
#: Velocities only; vertical mixing books the deposited mass.
LEDGER = "deposited"
KERNEL_MODULES = ("chem_sorgam",)
REQUIRES = ("t_phy", "p_phy", "alt", "ust", "rmol", "znt", "pblh")
ALLOCATES: tuple = ()
#: dry_dep_driver.F:552-559: aerosol velocities never exceed 0.5 m/s.
MAX_VELOCITY = np.float32(0.50)
#: The Wesely pass that computes the aerodynamic resistance first.
WESELY_KEY = "drydep.wesely"
SORGAM_KEY = "aerosol.sorgam"


def rows(table):
    return table.rows_for(KEY)


def _table_refusal(table):
    names = [row.name for row in rows(table)]
    if names != [name for name in AEROSOLS if name in names] or len(names) != len(AEROSOLS):
        return (f"sorgam_depdriver writes all {len(AEROSOLS)} SORGAM aerosol "
                f"rows, and the rows naming drydep.sorgam are {names}")
    if not table.rows_for(SORGAM_KEY):
        return ("sorgam_depdriver reads the aerosol state aerosol.sorgam "
                "keeps (h2oaj ... cvlim2), and no row names that process")
    if not table.rows_for(WESELY_KEY):
        return ("sorgam_depdriver takes its aerodynamic resistance from the "
                "Wesely gas pass (drydep.wesely), which no active row names")
    return None


def refusal(cfg):
    from woof.chem_table import load

    table = load(cfg)
    return None if table is None else _table_refusal(table)


def init(ctx):
    reason = _table_refusal(ctx.table)
    if reason:
        raise ValueError(reason)


def step(ctx, dt, ktau):
    if ktau <= 2 or getattr(ctx.cfg, "vertmix_onoff", 1) <= 0:
        return
    import cupy as cp

    table = ctx.table
    aerosol = [ctx.field(table.row(name)) for name in AEROSOLS]
    gas = [ctx.field(table.row(name)) for name in GASES]
    aux = [ctx.diag[f"sorgam_{name}"] for name in AUX]
    planes = [ctx.ddvel[ctx.row_index(table.row(name))] for name in AEROSOLS]
    vgsa = [cp.empty(planes[0].shape, dtype=cp.float32) for _ in AEROSOLS]
    launch_sorgam_drydep(
        aerosol, gas, aux, ctx.met("t_phy"), ctx.met("p_phy"), ctx.met("alt"),
        _plane(ctx.met("ust")), _plane(ctx.met("rmol")), _plane(ctx.met("znt")),
        _plane(ctx.met("pblh")), ctx.diag["wesely_aer_res_def"], vdvg_2=True,
        vgsa=vgsa)
    for plane, v in zip(planes, vgsa):
        cp.minimum(MAX_VELOCITY, v, out=plane)


def _plane(a):
    import cupy as cp

    return cp.ascontiguousarray(a, dtype=cp.float32)
