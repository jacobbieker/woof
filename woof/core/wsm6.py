"""WSM6 single-moment six-class microphysics (WRF ``mp_physics=6``).

The CUDA kernel owns one complete column and updates qv/qc/qr/qi/qs/qg,
potential temperature, surface rain/snow/graupel, frozen fraction, and the
scheme-native radiation effective radii.  WRF v4.6.1
``phys/physics_mmm/mp_wsm6.F90`` and ``mp_wsm6_effectRad.F90`` are the
primary numerical authorities.

The local process equations and WRF's forward semi-Lagrangian monotone-PLM
sedimentation are ported and checked against direct calls into the v4.6.1
Fortran.  This remains an executable integration port, not a whole-model or
bitwise WRF-parity claim; device and coupled-trajectory evidence are separate
release gates.
"""

from __future__ import annotations

import numpy as np
import cupy as cp

from woof.config import RunConfig
from woof.core import constants as c
from woof.core.kernels import get_kernel
from woof.core.state import DTYPE, DomainState

from woof.core.wsm6_constants import (WSM6_BLOCK, WSM6_MAX_LEVELS,
                                       WSM6_TILE_BLOCKS_PER_SM,
                                       WSM6_VERTICAL_LEVEL_BOUNDS,
                                       wsm6_workspace_extent,
                                       wsm6_workspace_floats)

_COLUMN_TPB = WSM6_BLOCK
_KMAX = WSM6_MAX_LEVELS
VERTICAL_LEVEL_BOUNDS = WSM6_VERTICAL_LEVEL_BOUNDS


def wsm6_tile_columns(fn, ncol: int) -> int:
    """Columns to keep in flight: enough to fill the card, and no more.

    The workspace is charged per column IN FLIGHT where the local-memory
    backing store it replaced was charged per thread the card could ever
    hold; over-sizing the tile hands that ratio back.  The device's
    occupancy query only ever lowers the measured blocks-per-SM constant.
    """
    try:
        sms = int(cp.cuda.Device().attributes["MultiProcessorCount"])
    except Exception:                              # noqa: BLE001
        # No device to size against (a host-side test drives the launcher
        # with a stand-in kernel): one tile, the whole domain.
        return int(max(WSM6_BLOCK, int(ncol)))
    per_sm = WSM6_TILE_BLOCKS_PER_SM
    try:
        from cupy_backends.cuda.api import driver

        resident = driver.occupancyMaxActiveBlocksPerMultiprocessor(
            fn.kernel.ptr, WSM6_BLOCK, 0)
        per_sm = min(per_sm, int(resident))
    except Exception:                              # noqa: BLE001
        pass                                       # keep the measured value
    per_sm = max(1, int(per_sm))
    # Floored at one block: the tile is the STEP of the launcher's loop.
    return int(max(WSM6_BLOCK, min(int(ncol), sms * per_sm * WSM6_BLOCK)))


def launch_wsm6(theta, qv, qc, qr, qi, qs, qg, rho, pii, pressure, dz,
                rainnc, rainncv, snownc, snowncv,
                graupelnc, graupelncv, sr, dt: float, *,
                effc, effi, effs, hail_opt: int = 0) -> None:
    """Launch one WSM6 call over contiguous FP32 ``(nz,ny,nx)`` fields."""
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"WSM6 fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    wskp = wsm6_workspace_extent(nz)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if int(hail_opt) not in (0, 1):
        raise ValueError(f"WSM6 hail_opt must be 0 or 1, got {hail_opt}")
    volume = {
        "theta": theta, "qv": qv, "qc": qc, "qr": qr,
        "qi": qi, "qs": qs, "qg": qg, "rho": rho, "pii": pii,
        "pressure": pressure, "dz": dz,
        "effc": effc, "effi": effi, "effs": effs,
    }
    for name, value in volume.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    surface_shape = (ny, nx)
    for name, value in {
            "rainnc": rainnc, "rainncv": rainncv,
            "snownc": snownc, "snowncv": snowncv,
            "graupelnc": graupelnc, "graupelncv": graupelncv,
            "sr": sr}.items():
        if value.shape != surface_shape or value.dtype != DTYPE:
            raise ValueError(f"{name} must be float32 {surface_shape}, got "
                             f"{value.dtype} {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    ncol = ny * nx
    kernel = get_kernel("wsm6", "wsm6_column")
    # The column arrays live in a global workspace sized to the columns in
    # flight and to this call's nz (kernels/wsm6.cu says why).  Columns go
    # in tiles of `tile`, each launched with the SAME one-thread-one-column
    # mapping, shifted by `col0` because the (nz, ny, nx) fields make a
    # column tile a strided scatter rather than a slice.
    tile = wsm6_tile_columns(kernel, ncol)
    ws = cp.empty(wsm6_workspace_floats(nz, tile), dtype=DTYPE)
    try:
        for col0 in range(0, ncol, tile):
            count = min(tile, ncol - col0)
            blocks = (count + _COLUMN_TPB - 1) // _COLUMN_TPB
            kernel(
                (blocks,), (_COLUMN_TPB,),
                (theta, qv, qc, qi, qr, qs, qg, rho, pressure, pii, dz,
                 rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr,
                 effc, effi, effs, DTYPE(dt), np.int32(hail_opt),
                 np.int32(nz), np.int32(ny), np.int32(nx),
                 ws, np.int32(wskp), np.int32(col0)))
    finally:
        del ws


def apply(state: DomainState, cfg: RunConfig, dt: float, *,
          refl_10cm_due: bool = False):
    """Prepare WRF fields, run WSM6, and finish retained MP heating."""
    nz, ny, nx = state.p.shape
    required = ("qi", "qs", "qg", "effc", "effi", "effs")
    missing = [name for name in required if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=6 state lacks WSM6 fields: "
                         + ", ".join(missing))
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    theta = state.scratch((nz, ny, nx), "wsm6_theta")
    rho = state.scratch((nz, ny, nx), "wsm6_rho")
    pii = state.scratch((nz, ny, nx), "wsm6_pii")
    dz = state.scratch((nz, ny, nx), "wsm6_dz")
    z8w = state.scratch((nz + 1, ny, nx), "wsm6_z8w")
    theta[...] = thb + state.thp
    rho[...] = 1.0 / state.alt
    pii[...] = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    dz[...] = z8w[1:] - z8w[:-1]

    surface = (ny, nx)
    rainnc = state.scratch(surface, "mp_rainnc")
    rainncv = state.scratch(surface, "mp_rainncv")
    snownc = state.scratch(surface, "mp_snownc")
    snowncv = state.scratch(surface, "mp_snowncv")
    graupelnc = state.scratch(surface, "mp_graupelnc")
    graupelncv = state.scratch(surface, "mp_graupelncv")
    sr = state.scratch(surface, "mp_sr")

    from woof.core.microphysics import (MicrophysicsDiagnostics,
                                         moist_physics_finish,
                                         save_pre_mp_theta)
    save_pre_mp_theta(state)
    launch_wsm6(
        theta, state.qv, state.qc, state.qr, state.qi, state.qs, state.qg,
        rho, pii, state.p, dz, rainnc, rainncv, snownc, snowncv,
        graupelnc, graupelncv, sr, dt, effc=state.effc,
        effi=state.effi, effs=state.effs, hail_opt=cfg.wsm6_hail_opt)
    if refl_10cm_due:
        from woof.core.refl import compute_and_stash_refl_10cm
        refl_t = state.scratch((nz, ny, nx), "refl_t")
        refl_t[...] = theta * pii
        compute_and_stash_refl_10cm(state, cfg, refl_t, state.p)
    moist_physics_finish(state, cfg, theta, dt)
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv)


__all__ = ["apply", "launch_wsm6", "wsm6_tile_columns"]
