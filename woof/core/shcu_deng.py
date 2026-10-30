"""Deng shallow cumulus (WRF ``shcu_physics = 5``, WRF-Solar): device launcher.

The numerical authority is WRF v4.7.1 itself: ``kernels/shcu_deng.cu`` is a
line-for-line port of ``deng_shcu_driver``/``deng_shcu``
(phys/module_shcu_deng.F) and of the Deng arm of ``shallowcu_driver``
(phys/module_shallowcu_driver.F), graded at 0 ULP on every output against
the unmodified Fortran by the column-tile oracle in
``tools/shcu_deng_wrf471_oracle`` (gate:
``tests/test_shcu_deng_wrf471_oracle.py``).

Array layout is ``(nz, ny, nx)`` with x fastest; ``z_at_w``/``w`` are
``(nz+1, ny, nx)`` and ``ainckfsa`` is ``(100, ny, nx)``.

The scheme carries the state WRF's Registry gives package ``dengshcuscheme``
(Registry.EM_COMMON:3232) from step to step -- :data:`STATE_3D`,
:data:`STATE_2D`, :data:`STATE_INT` and ``ainckfsa`` -- and two fields the
next radiation call reads, ``cldfra_sh`` and ``cw_rad``
(module_radiation_driver.F:1566-1575).  :class:`DengState` owns them; every
one is a restart field (WRF marks them all ``r``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields
from functools import lru_cache

import numpy as np

DTYPE = np.float32
_TPB = 256
_WS_LANES = 32
_KMAX = 256
VERTICAL_LEVEL_BOUNDS = (3, _KMAX)
#: AINCKFSA's level extent (Registry ``i{nsh}j``; deng_shcu_init's 1:100).
NSH = 100
#: Must match shcu_deng.cu's DENGWS_SLOTS; the oracle test re-derives it.
DENGWS_SLOTS = 88
DENG_TILE_BLOCKS_PER_SM = 1

#: The PBL schemes whose TKE/K/L Deng reads (module_shallowcu_driver.F:
#: 605-630): MYJ hands TKE_PBL, MYNN 0.5*QKE.  Anything else is
#: wrf_error_fatal in WRF.
SUPPORTED_PBL = {2: "myj", 5: "mynn"}

#: Status bits the kernel sets where WRF stops or reads memory it never
#: wrote (shcu_deng.cu DENG_STATUS_*).
STATUS_BITS = {1: "STOP 907 (|UPOLD| < 0.001 at the updraft top)",
               2: "STOP 901-904 (cloud depth equals 4 km - LFC + LCL)",
               4: "ice settling NSTEP > 1000",
               8: "mass imbalance exit before TG is formed (ISTOP)",
               16: "post-convective LCL at the lowest level (K = 0)",
               32: "GAMMA at a non-positive argument",
               64: "no model level at or below 500 hPa (L5 unset)"}

STATE_3D = ("w0avg", "tkeavg", "cldareaa", "cldareab", "cldliqa",
            "cldliqb", "wub", "cldfra_sh", "ca_rad", "cw_rad")
STATE_2D = ("rainsh", "rainshv", "rainshvb", "pblmax", "capesave", "xtime1",
            "radsave", "clddpthb", "cldtopb", "pblhavg")
STATE_INT = ("ltopb", "kdcldtop", "kdcldbas")
OUTPUT_3D = ("rthshten", "rqvshten", "rqcshten", "rqrshten", "rushten",
             "rvshten", "rdcashten", "rqcdcshten")


def deng_workspace_floats(nz: int, columns: int) -> int:
    blocks = (int(columns) + _TPB - 1) // _TPB
    return blocks * DENGWS_SLOTS * (int(nz) + 1) * _TPB


def deng_tile_columns(fn, ncol: int) -> int:
    import cupy as cp

    dev = cp.cuda.Device()
    try:
        sms = int(dev.attributes["MultiProcessorCount"])
    except Exception:                               # noqa: BLE001
        sms = 64
    per_sm = DENG_TILE_BLOCKS_PER_SM
    try:
        from cupy_backends.cuda.api import driver

        handle = getattr(fn, "kernel", fn)
        resident = driver.occupancyMaxActiveBlocksPerMultiprocessor(
            handle.ptr, _TPB, 0)
        per_sm = min(per_sm, int(resident))
    except Exception:                               # noqa: BLE001
        pass
    per_sm = max(1, int(per_sm))
    return int(min(int(ncol), sms * per_sm * _TPB))


#: Its own direct-NVRTC route, as KF's: ``--fmad=false`` (gfortran contracts
#: nothing at -O0 -ffp-contract=off) and ``--ftz=false`` (the updraft's
#: condensate and the NBC water decay through the subnormal range, and
#: CuPy's RawModule would append ``-ftz=true``).
DENG_MODULE_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")
DENG_MODULE_KEY = "woof.core.kernels:shcu_deng"


@lru_cache(maxsize=None)
def _deng_module(device):
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler
    from woof.certify.kernel_manifest import record_module
    from woof.core.kernels import module_source
    from woof.kernel_compile_notice import observe_module_compile

    source = module_source("shcu_deng")
    with cp.cuda.Device(device), observe_module_compile(DENG_MODULE_KEY):
        binary, _ = compiler.compile_using_nvrtc(
            source, DENG_MODULE_OPTIONS, None, "shcu_deng.cu")
        module = cp.cuda.function.Module()
        module.load(binary.encode() if isinstance(binary, str) else binary)
    from woof.physics_params import note_compiled
    note_compiled("shcu_deng")
    record_module(DENG_MODULE_KEY, source=source,
                  options=DENG_MODULE_OPTIONS, module=None)
    return module


def _deng_function(name: str):
    import cupy as cp

    return _deng_module(cp.cuda.runtime.getDevice()).get_function(name)


@dataclass
class DengState:
    """The Deng fields WRF keeps in the grid between steps.

    Cold start is ``deng_shcu_init`` (module_shcu_deng.F:3629-3674): every
    real field 0, LTOPB/KDCLDTOP/KDCLDBAS 1.  CA_RAD/CW_RAD/CLDFRA_SH start
    at 0 (deng_shcu_init and the Registry default) and are rewritten by every
    call; the radiation that runs before the next call reads them.
    """
    w0avg: object
    tkeavg: object
    cldareaa: object
    cldareab: object
    cldliqa: object
    cldliqb: object
    wub: object
    cldfra_sh: object
    ca_rad: object
    cw_rad: object
    rainsh: object
    rainshv: object
    rainshvb: object
    pblmax: object
    capesave: object
    xtime1: object
    radsave: object
    clddpthb: object
    cldtopb: object
    pblhavg: object
    ltopb: object
    kdcldtop: object
    kdcldbas: object
    ainckfsa: object

    @classmethod
    def cold(cls, nz: int, ny: int, nx: int) -> "DengState":
        import cupy as cp

        z3 = {k: cp.zeros((nz, ny, nx), dtype=DTYPE) for k in STATE_3D}
        z2 = {k: cp.zeros((ny, nx), dtype=DTYPE) for k in STATE_2D}
        zi = {k: cp.ones((ny, nx), dtype=cp.int32) for k in STATE_INT}
        return cls(**z3, **z2, **zi,
                   ainckfsa=cp.zeros((NSH, ny, nx), dtype=DTYPE))

    def arrays(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}


def kernel_capacity(nz: int) -> int:
    nz = int(nz)
    lo, hi = VERTICAL_LEVEL_BOUNDS
    if nz < lo or nz > hi:
        raise ValueError(f"Deng shallow cumulus requires {lo} <= nz <= {hi}, "
                         f"got {nz}")
    return nz


def launch_deng(*, u, v, t, qv, qc, qr, p, pii, rho, dz8w, tke, exch_h,
                el_pbl, rthratenlw, rthratensw, z_at_w, w, pblh, ht, xlong,
                dnw, mavail, state: DengState, dt: float, dx: float,
                xtime: float, gmt: float, adaptive: bool, bl_pbl_physics: int,
                pratesh=None, raise_on_status: bool = True) -> dict:
    """One ``shallowcu_driver`` call of the Deng scheme over every column.

    Inputs are WRF's: ``u``/``v``/``t``/``th``-free (the scheme reads T),
    ``qv``/``qc``/``qr``, ``p`` (p_hyd), ``pii`` (PI), ``rho`` ((1+qv)/alt),
    ``dz8w``, ``z_at_w`` and ``w`` (full levels), the PBL's ``tke``
    (MYJ's TKE_PBL, or MYNN's QKE -- halved here as the driver does),
    ``exch_h`` and ``el_pbl``, the RRTMG ``rthratenlw``/``rthratensw``,
    ``pblh``, ``ht``, ``xlong`` (degrees east), ``dnw`` (nz,) and the
    surface ``mavail`` (updated in place, module_shcu_deng.F:365-370).
    ``state`` is advanced in place.  ``xtime`` is WRF's minutes since start
    at this step, ``gmt`` the start hour.

    Returns the eight tendencies (``rthshten`` ... ``rqcdcshten``) and
    ``status`` (int32 per column, :data:`STATUS_BITS`).
    """
    import cupy as cp

    if int(bl_pbl_physics) not in SUPPORTED_PBL:
        raise ValueError(
            "Deng shallow cumulus reads TKE/EXCH_H/EL_PBL from MYJ "
            "(bl_pbl_physics=2) or MYNN (5); WRF stops on "
            f"bl_pbl_physics={bl_pbl_physics}")
    shape = tuple(int(n) for n in t.shape)
    nz, ny, nx = shape
    kernel_capacity(nz)
    for name, arr, want in (
            ("u", u, shape), ("v", v, shape), ("t", t, shape),
            ("qv", qv, shape), ("qc", qc, shape), ("qr", qr, shape),
            ("p", p, shape), ("pii", pii, shape), ("rho", rho, shape),
            ("dz8w", dz8w, shape), ("tke", tke, shape),
            ("exch_h", exch_h, shape), ("el_pbl", el_pbl, shape),
            ("rthratenlw", rthratenlw, shape),
            ("rthratensw", rthratensw, shape),
            ("z_at_w", z_at_w, (nz + 1, ny, nx)), ("w", w, (nz + 1, ny, nx)),
            ("pblh", pblh, (ny, nx)), ("ht", ht, (ny, nx)),
            ("xlong", xlong, (ny, nx)), ("mavail", mavail, (ny, nx)),
            ("dnw", dnw, (nz,))):
        if (not isinstance(arr, cp.ndarray) or tuple(arr.shape) != want
                or arr.dtype != DTYPE or not arr.flags.c_contiguous):
            raise ValueError(f"Deng {name} must be a C-contiguous float32 "
                             f"device array of shape {want}")
    if pratesh is None:
        pratesh = cp.zeros((ny, nx), dtype=DTYPE)
    out = {k: cp.empty(shape, dtype=DTYPE) for k in OUTPUT_3D}
    status = cp.zeros((ny, nx), dtype=cp.int32)
    s = state
    kernel = _deng_function("deng_shcu_column")
    ncol = ny * nx
    tile = deng_tile_columns(kernel, ncol)
    ws = cp.empty(deng_workspace_floats(nz, tile), dtype=DTYPE)
    for col0 in range(0, ncol, tile):
        span = min(tile, ncol - col0)
        blocks = (span + _TPB - 1) // _TPB
        kernel((blocks,), (_TPB,), (
            u, v, t, qv, qc, qr, p, pii, rho, dz8w, tke, exch_h, el_pbl,
            rthratenlw, rthratensw, z_at_w, w, pblh, ht, xlong, dnw, pratesh,
            s.w0avg, s.tkeavg, s.cldareaa, s.cldareab, s.cldliqa, s.cldliqb,
            s.wub, s.rainsh, s.rainshv, s.rainshvb, s.pblmax, s.capesave,
            s.xtime1, s.radsave, s.clddpthb, s.cldtopb, mavail, s.pblhavg,
            s.ltopb, s.kdcldtop, s.kdcldbas, s.ainckfsa,
            out["rthshten"], out["rqvshten"], out["rqcshten"],
            out["rqrshten"], out["rushten"], out["rvshten"],
            out["rdcashten"], out["rqcdcshten"], s.cldfra_sh, s.ca_rad,
            s.cw_rad, status, ws, DTYPE(dt), DTYPE(dx), DTYPE(xtime),
            DTYPE(gmt), np.int32(1 if adaptive else 0),
            np.int32(1 if int(bl_pbl_physics) == 5 else 0),
            np.int32(nz), np.int32(ny), np.int32(nx), np.int32(col0)))
    del ws
    out["status"] = status
    if raise_on_status:
        bad = int(cp.count_nonzero(status))
        if bad:
            bits = int(cp.bitwise_or.reduce(status.ravel()))
            names = [v for b, v in STATUS_BITS.items() if bits & b]
            raise RuntimeError(
                f"Deng shallow cumulus: {bad} column(s) reached a point where "
                f"WRF stops or reads unwritten memory: {'; '.join(names)}")
    return out


__all__ = ["DENGWS_SLOTS", "DengState", "NSH", "OUTPUT_3D", "STATE_2D",
           "STATE_3D", "STATE_INT", "STATUS_BITS", "SUPPORTED_PBL",
           "deng_tile_columns", "deng_workspace_floats", "kernel_capacity",
           "launch_deng"]
