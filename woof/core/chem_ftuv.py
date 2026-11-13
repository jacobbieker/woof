"""Pure launcher for WRF-Chem's Fast-TUV photolysis (phot_opt = 3) under the
RADM2 packages (chem_opt = 1 RADM2, 2 RADM2SORG).

``launch_ftuv`` is ftuv_driver (chem/module_ftuv_driver.F, WRF v4.7.1) for
one photolysis call on every column: the photolysis rates WRF's ph_* fields
carry, in 1/min, named as photolysis_driver names them (``ph_no2`` ...).  The
one-time photo_inti state for a model depth is built by a single-thread
initialisation kernel and cached.

ftuv_driver hands photo the column's SORGAM masses (acb1 .. asal), but photoin
never reads them (it calls setaer nowhere), so they are not inputs here.  The
aerosol optical depths enter only with aer_ra_feedback = 1.
"""
from __future__ import annotations

import re

import numpy as np

from woof.core.chem_column import field

#: Output rows, in photolysis_driver's ph_<name> names (radm_driver reads
#: no2 .. ch3ono2; the rest are the other fields ftuv_driver writes for
#: these packages).
PHOTOLYSIS = ("no2", "o31d", "o33p", "hno2", "hno3", "hno4", "no3o2", "no3o", "h2o2",
              "ch2om", "ch2or", "ch3cho", "ch3o2h", "o2", "ch3coo2h", "ch3coch3",
              "hcocho", "hcochob", "ch3cocho", "hcochest", "ch3ono2", "macr",
              "ch3coc2h5", "cl2", "hocl", "fmcl", "pan", "n2o5")
#: photo_inti's exo levels above the model top for its fixed 20 km exo top
#: (module_ftuv_driver.F:916-933: kcon = 23, nabv = 15).
NABV = 15
#: Columns per launch: each needs its own workspace (about 0.6 MB at 45
#: module levels).
BATCH = 2048
_STATE = {}


def _defines():
    from woof.core.kernels import module_source
    src = module_source("chem_ftuv")
    out = {}
    for name in ("TV_SHARED_D", "TV_SHARED_I", "TV_WS_D", "TV_WS_I", "TV_SQ_SIZE"):
        m = re.search(r"^#define %s\(nz\) (.*)$" % name, src, re.M)
        out[name] = m.group(1)
    return out


def _eval(expr, nz):
    if not re.fullmatch(r"[0-9nz+\-*() ]+", expr):
        raise ValueError(f"unexpected size expression {expr!r}")
    return int(eval(expr, {"__builtins__": {}}, {"nz": nz}))  # noqa: S307 (our own #define)


def _state(kte, chem_opt):
    """photo_inti's module state for kte model levels (shared, read-only)."""
    import cupy as cp
    from woof.core.kernels import get_kernel
    key = (kte, chem_opt)
    if key in _STATE:
        return _STATE[key]
    nzm = kte + NABV
    d = _defines()
    shared_d = cp.zeros(_eval(d["TV_SHARED_D"], nzm), dtype=cp.float64)
    shared_i = cp.zeros(max(1, _eval(d["TV_SHARED_I"], nzm)), dtype=cp.int32)
    scal_d = cp.zeros(1, dtype=cp.float64)
    scal_i = cp.zeros(4, dtype=cp.int32)
    ws_d_len = _eval(d["TV_WS_D"], nzm) + 1
    ws_i_len = _eval(d["TV_WS_I"], nzm) + 1
    ws_d = cp.zeros(ws_d_len, dtype=cp.float64)
    ws_i = cp.zeros(ws_i_len, dtype=cp.int32)
    get_kernel("chem_ftuv", "chem_ftuv_init")(
        (1,), (1,), (shared_d, shared_i, scal_d, scal_i, ws_d, ws_i,
                     np.int64(ws_d_len), np.int64(ws_i_len),
                     np.int32(kte), np.int32(nzm), np.int32(chem_opt)))
    got = scal_i.get()
    if got[3] != 0:
        raise RuntimeError(f"Fast-TUV initialisation stopped at module_ftuv_*.F line {got[3]}")
    if got[2] != nzm:
        raise RuntimeError(f"photo_inti set nz = {got[2]}, expected {nzm}")
    _STATE[key] = (shared_d, shared_i, scal_d, scal_i, nzm)
    return _STATE[key]


def launch_ftuv(fields, *, chem_opt, julday, gmt, curr_secs, aer_ra_feedback=0,
                aerosol_optics=None):
    """One ftuv_driver call on every column.

    fields: mapping of (nz, ny, nx) float32 CuPy arrays t_phy (K), p_phy
    (Pa), rho_phy (kg/m3), qv, qc (kg/kg), o3 (ppmv), plus (nz + 1, ny, nx)
    p8w, t8w (only level 0 is read) and z_at_w (m), and (ny, nx) xlat, xlong
    (degrees).  aerosol_optics: with aer_ra_feedback = 1, 12 (nz, ny, nx)
    rows tauaer1..4, waer1..4, gaer1..4.  Returns {name: (nz, ny, nx)} for
    :data:`PHOTOLYSIS`, 1/min.
    """
    import cupy as cp
    from woof.core.kernels import get_kernel
    if chem_opt not in (1, 2):
        raise ValueError("this Fast-TUV port serves chem_opt = 1 and 2")
    if aer_ra_feedback not in (0, 1):
        raise ValueError("aer_ra_feedback must be 0 or 1")
    t = fields["t_phy"]
    nz, ny, nx = t.shape
    shape, shape_w, shape_2 = (nz, ny, nx), (nz + 1, ny, nx), (ny, nx)
    nin = 23
    inp = cp.zeros((nin, nz + 1, ny, nx), dtype=cp.float32)
    for row, name in enumerate(("t_phy", "p_phy", "rho_phy")):
        inp[row, :nz] = field(fields[name], shape, name)
    for row, name in ((3, "p8w"), (4, "t8w"), (5, "z_at_w")):
        inp[row] = field(fields[name], shape_w, name)
    for row, name in ((6, "qv"), (7, "qc"), (8, "o3")):
        inp[row, :nz] = field(fields[name], shape, name)
    if aer_ra_feedback == 1:
        if aerosol_optics is None or len(aerosol_optics) != 12:
            raise ValueError("aer_ra_feedback = 1 needs 12 aerosol optics rows")
        for n, a in enumerate(aerosol_optics):
            inp[9 + n, :nz] = field(a, shape, f"aerosol_optics[{n}]")
    inp[21, 0] = field(fields["xlat"], shape_2, "xlat")
    inp[22, 0] = field(fields["xlong"], shape_2, "xlong")
    shared_d, shared_i, scal_d, scal_i, nzm = _state(nz, chem_opt)
    d = _defines()
    ws_d_per = _eval(d["TV_WS_D"], nzm) + _eval(d["TV_SQ_SIZE"], nzm) + (nz + 1) * (19 + 58) \
        + 2 * nzm * 58 + nzm * 17
    ws_i_per = max(1, _eval(d["TV_WS_I"], nzm))
    ncol = ny * nx
    batch = min(BATCH, ncol)
    ws_d = cp.empty(batch * ws_d_per, dtype=cp.float64)
    ws_i = cp.empty(batch * ws_i_per, dtype=cp.int32)
    out = cp.zeros((len(PHOTOLYSIS), nz, ny, nx), dtype=cp.float32)
    fatal = cp.zeros(batch, dtype=cp.int32)
    kernel = get_kernel("chem_ftuv", "chem_ftuv")
    for col0 in range(0, ncol, batch):
        n = min(batch, ncol - col0)
        kernel((-(-n // 64),), (64,),
               (shared_d, shared_i, scal_d, scal_i, ws_d, ws_i,
                np.int64(ws_d_per), np.int64(ws_i_per), inp, out, fatal,
                np.int32(col0), np.int32(n), np.int32(nx), np.int32(ny), np.int32(nz),
                np.int32(nzm), np.int32(chem_opt), np.int32(aer_ra_feedback),
                np.int32(julday), np.float32(gmt), np.float64(curr_secs)))
        bad = fatal[:n].get()
        if bad.any():
            line = int(bad[bad != 0][0])
            what = "workspace overrun in the routine at" if line < 0 else "fatal at"
            raise RuntimeError(f"Fast-TUV {what} module_ftuv_*.F line {abs(line)}")
    return {name: out[k] for k, name in enumerate(PHOTOLYSIS)}


# ---------------------------------------------------------------------------
# The process (``photolysis.ftuv``): the chem driver's view.
# ---------------------------------------------------------------------------
#
# Column-based: launch_ftuv batches columns over ny*nx and reads latitude and
# longitude per column, so the hex route can hand (nz, 1, ncells) views.

from woof.core.chem_context import ChemAllocation  # noqa: E402

KEY = "photolysis.ftuv"
#: Photolysis moves no mass; the bucket is never touched.
LEDGER = "chemistry"
KERNEL_MODULES = ("chem_ftuv",)
REQUIRES = ("t_phy", "p_phy", "rho", "qv", "p8w", "t8w", "z_at_w", "xlat",
            "xlong")
#: The aerosol process whose rows make the package RADM2SORG (chem_opt = 2).
SORGAM_KEY = "aerosol.sorgam"
#: WRF's ph_<name> state (Registry state, restart 'r'): rewritten on
#: photolysis steps and read by the mechanism on every chem step between,
#: so a restart carries it.
ALLOCATES = tuple(
    ChemAllocation(f"ph_{name}", "3d", units="min-1",
                   description=f"photolysis rate ph_{name} "
                               "(photolysis_driver.F, Fast-TUV)")
    for name in PHOTOLYSIS)


def rows(table):
    return table.rows_for(KEY)


def chem_opt_for(table) -> int:
    """WRF's chem_opt for the active RADM2 package: 2 (RADM2SORG) when the
    SORGAM rows are active, else 1 (RADM2)."""
    return 2 if table.rows_for(SORGAM_KEY) else 1


def refusal(cfg):
    from woof.chem_table import load

    table = load(cfg)
    if (int(getattr(cfg, "aer_ra_feedback", 0)) != 0 and table is not None
            and not table.rows_for(SORGAM_KEY)):
        return ("aer_ra_feedback = 1 hands Fast-TUV the SORGAM aerosol optical "
                "depths (ftuv_driver), and this selection has no SORGAM aerosol "
                "(RADM2 alone, chem_opt = 1)")
    if table is not None and len(rows(table)) != 1:
        return ("photolysis.ftuv reads one ozone row (the column the TUV "
                f"radiative transfer sees); {[r.name for r in rows(table)]} "
                "name it")
    return None


def init(ctx):
    reason = refusal(ctx.cfg)
    if reason:
        raise ValueError(reason)


def step(ctx, dt, ktau):
    from woof.core.chem_radm2 import _wrf_due
    import cupy as cp

    if not _wrf_due(ktau, ctx.clock.curr_secs, dt,
                    getattr(ctx.cfg, "photdt", 0.0)):
        return
    (o3,) = rows(ctx.table)
    t = ctx.met("t_phy")
    qc = ctx.met("qc") if ctx.has("qc") else cp.zeros_like(t)
    fields = {"t_phy": t, "p_phy": ctx.met("p_phy"), "rho_phy": ctx.met("rho"),
              "qv": ctx.met("qv"), "qc": qc, "o3": ctx.field(o3),
              "p8w": ctx.met("p8w"), "t8w": ctx.met("t8w"),
              "z_at_w": ctx.met("z_at_w"),
              "xlat": cp.ascontiguousarray(ctx.met("xlat"), dtype=cp.float32),
              "xlong": cp.ascontiguousarray(ctx.met("xlong"), dtype=cp.float32)}
    clock = ctx.clock
    feedback = int(getattr(ctx.cfg, "aer_ra_feedback", 0) or 0)
    aerosol_optics = None
    if feedback:
        # The SORGAM optics at this model time (woof.core.chem_modal_optics,
        # the default build's clean-room optics): tauaer1..4, waer1..4,
        # gaer1..4, as ftuv_driver reads them.
        from woof.core.chem_modal_optics import sorgam_optics
        optics = sorgam_optics(ctx)
        aerosol_optics = [cp.ascontiguousarray(optics[name][i])
                          for name in ("tauaer", "waer", "gaer") for i in range(4)]
    out = launch_ftuv(fields, chem_opt=chem_opt_for(ctx.table),
                      julday=int(clock.julday), gmt=float(clock.gmt),
                      curr_secs=float(clock.curr_secs), aer_ra_feedback=feedback,
                      aerosol_optics=aerosol_optics)
    for name in PHOTOLYSIS:
        ctx.diag[f"ph_{name}"][...] = out[name]
