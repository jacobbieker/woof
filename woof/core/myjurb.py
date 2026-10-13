"""WRF v4.7.1 MYJ under BEP/BEP+BEM, with QV and CWM diffusion.

``launch_myjurb`` has the allocating call shape of ``myj_pbl_step``.
Columns and tendencies use bottom-up contiguous float32 (nz, ny, nx).
``columns['qc']`` is WRF's CWM, which the PBL driver supplies as QC_CURR.
Ice and other hydrometeors are not diffused by this routine.
"""

from __future__ import annotations

from collections.abc import Mapping

import cupy as cp
import numpy as np

from woof.core.kernels import get_kernel
from woof.core.myjpbl import myj_tile_columns
from woof.core.physics_inventory import MYJ_BLOCK, myjurb_workspace_floats
from woof.core.state import DTYPE
from woof.physics_vertical_contract import (
    outside_vertical_bounds, refuse_vertical_levels)

#: The launch block IS the workspace's lane width (``MYJURBWS_LANES``).
_TPB = MYJ_BLOCK
#: Refusal ceiling, equal to kernels/myjurb.cu's MYJ_KMAX guard.  The 24
#: column arrays live in a workspace sized to the runtime nz, so no array
#: is sized by it.
MYJURB_MAX_COLUMN_LEVELS = 256
MYJURB_MIN_COLUMN_LEVELS = 4
_COLUMN_INPUTS = ("dz", "u", "v", "t", "th", "exner", "qv", "qc", "p")
_SURFACE_INPUTS = ("psfc", "ust", "tsk", "chklowq", "xland", "sice", "snow",
                   "akhs", "akms", "elflx", "uz0", "vz0")
MYJURB_INOUT = ("thz0", "qz0", "qsfc", "ct")
MYJURB_COLUMN_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten",
                          "rqcblten", "el_myj", "exch_h", "exch_m")
MYJURB_SURFACE_OUTPUTS = ("pblh", "kpbl", "mixht")
BEP_INPUTS = ("a_u_bep", "a_v_bep", "a_t_bep", "a_q_bep", "a_e_bep",
              "b_u_bep", "b_v_bep", "b_t_bep", "b_q_bep", "b_e_bep",
              "dlg_bep", "dl_u_bep", "vl_bep", "sf_bep")


def launch_myjurb(columns, surface, state, tke, *, dtturbl: float,
                  bep: Mapping[str, cp.ndarray], frc_urb2d, ht=None,
                  flag_bep: bool = True, idiff: int = 0,
                  outputs=None) -> dict:
    """Run one MYJURB step and return WRF outputs and coupling aliases.

    TKE and ``MYJURB_INOUT`` arrays change in place. ``ht`` is WRF's HT,
    which seeds the interface heights (ZINT(KTE+1) = HT, :283); the default
    (None) is zero, the engine's MYJ convention of heights above ground
    (kernels/myjpbl.cu header), which differs from WRF only in float32
    rounding -- the oracle test passes the fixture's HT to grade WRF's
    arithmetic exactly. ``flag_bep=False`` uses the Fortran's rural arm.
    Four to 256 full levels are supported, with WRF LOWLYR=1.

    Optional ``outputs`` supplies carried EXCH_H/M: Fortran writes only their
    levels 1..nz-1 and leaves level 0 unchanged (:453-463). Fresh allocations
    start at zero. ``mixht`` exports MIXLEN's local BLMX for engine call-shape
    compatibility; MYJURB has no WRF MIXHT output. IDIFF=1 skips the primary
    tendency solves (:491-767); supplied tendencies then remain untouched.
    """
    shape = surface["psfc"].shape
    nz = int(columns["dz"].shape[0])
    bounds = (MYJURB_MIN_COLUMN_LEVELS, MYJURB_MAX_COLUMN_LEVELS)
    if outside_vertical_bounds(nz, bounds):
        raise refuse_vertical_levels(
            "MYJURB PBL", bounds, nz,
            breakage=(
                f"the kernel refuses more than {MYJURB_MAX_COLUMN_LEVELS} "
                "levels per column (the MYJ_KMAX guard in "
                "kernels/myjurb.cu); below four levels the TKE "
                "tridiagonal solve has no interior rows."))
    if len(shape) != 2:
        raise ValueError("launch_myjurb requires (ny, nx) surface arrays")
    if set(bep) != set(BEP_INPUTS):
        raise ValueError(f"MYJURB BEP keys must be exactly {BEP_INPUTS}")
    if isinstance(idiff, bool) or not isinstance(idiff, (int, np.integer)):
        raise TypeError("MYJURB idiff must be an integer")

    def check(array, expected_shape, dtype=DTYPE):
        if (array.shape != expected_shape or array.dtype != dtype
                or not array.flags.c_contiguous):
            raise ValueError(
                f"launch_myjurb requires contiguous {dtype} {expected_shape} arrays")

    if ht is None:
        ht = cp.zeros(shape, dtype=DTYPE)
    column_arrays = [columns[name] for name in _COLUMN_INPUTS]
    for array in (*column_arrays, tke):
        check(array, (nz, *shape))
    for name in BEP_INPUTS:
        check(bep[name], (nz + int(name == "sf_bep"), *shape))
    for array in (frc_urb2d, ht,
                  *(surface[name] for name in _SURFACE_INPUTS),
                  *(state[name] for name in MYJURB_INOUT)):
        check(array, shape)
    if outputs is None:
        outputs = {name: cp.zeros((nz, *shape), dtype=DTYPE)
                   for name in MYJURB_COLUMN_OUTPUTS}
        outputs.update(pblh=cp.empty(shape, dtype=DTYPE),
                       mixht=cp.empty(shape, dtype=DTYPE),
                       kpbl=cp.empty(shape, dtype=cp.int32))
    for name in MYJURB_COLUMN_OUTPUTS:
        check(outputs[name], (nz, *shape))
    for name in MYJURB_SURFACE_OUTPUTS:
        check(outputs[name], shape, cp.int32 if name == "kpbl" else DTYPE)
    n = int(np.prod(shape))
    kernel = get_kernel("myjurb", "myjurb_column")
    # Column arrays in a global workspace sized to the tile in flight and
    # the runtime nz (kernels/myjurb.cu header); tiles offset by col0.
    tile = myj_tile_columns(kernel, n)
    wskp = nz + 1
    ws = cp.empty(myjurb_workspace_floats(nz, tile), dtype=DTYPE)
    args = (tuple(column_arrays) + (tke,)
            + tuple(surface[name] for name in _SURFACE_INPUTS)
            + tuple(state[name] for name in MYJURB_INOUT)
            + tuple(outputs[name] for name in MYJURB_COLUMN_OUTPUTS)
            + tuple(outputs[name] for name in MYJURB_SURFACE_OUTPUTS)
            + tuple(bep[name] for name in BEP_INPUTS) + (frc_urb2d, ht)
            + (DTYPE(dtturbl), np.int32(bool(flag_bep)), np.int32(idiff),
               np.int32(nz), np.int32(n), ws, np.int32(wskp)))
    for col0 in range(0, n, tile):
        blocks = (min(tile, n - col0) + _TPB - 1) // _TPB
        kernel((blocks,), (_TPB,), args + (np.int32(col0),))
    # MYJURB has no ice arm (module_bl_myjurb.F diffuses QV and CWM only),
    # so RQIBLTEN is a published zero -- present because the MYJ slot's
    # validation reads it -- and no ``dqi`` alias is returned, so nothing
    # couples an ice tendency.
    if "rqiblten" not in outputs:
        outputs["rqiblten"] = cp.zeros((nz, *shape), dtype=DTYPE)
    outputs.update(du=outputs["rublten"], dv=outputs["rvblten"],
                   dtheta=outputs["rthblten"], dqv=outputs["rqvblten"],
                   dqc=outputs["rqcblten"])
    return outputs
