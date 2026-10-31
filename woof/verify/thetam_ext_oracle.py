"""The rest of use_theta_m = 1 against compiled WRF v4.7.1.

Two records grade the paths this module ports:

* tests/data/wrf471_thetam_ext (tools/bigstep_wrf471_oracle/
  thetam_ext_build.sh): phy_prep_part2's RTHFTEN block
  (kernels/theta_m_cumulus.cu, with and without its decoupling) and
  vertical_diffusion_2's hflux block (kernels/smag2d_theta_m.cu, isfflx 0, 1
  and 2), on the real WRF patch.
* tests/data/wrf471_thetam (the first record): moist_physics_prep_em and
  moist_physics_finish_em, through woof.core.theta_m.microphysics_window
  around stand-in schemes that finish as each family of woof's adapters
  does -- the shared woof.core.microphysics.moist_physics_finish, and
  classic Thompson's two fused dry finishes (thompson_adapter_finish,
  tc_finish) followed by woof.core.theta_m.finish_microphysics.

:func:`port_outputs` and :func:`window_outputs` run woof's own kernels and
launchers; :func:`compare` counts identical words.
"""

from __future__ import annotations

import hashlib
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ORACLE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "wrf471_thetam_ext"

#: thetam_ext_run.F90's output records, in order (all mass-shaped; the
#: hflux records' ``hfx`` is the 2-D field on the k = 0 plane).
RECORDS = (
    ("prep_th_phy", "m"), ("prep_rho", "m"),
    ("part2_coupled_rthften", "m"), ("part2_coupled_rqvften", "m"),
    ("part2_decoupled_rthften", "m"), ("part2_decoupled_rqvften", "m"),
    *((f"hflux_isfflx{c}_{f}", "m") for c in (0, 1, 2)
      for f in ("rt_tendf", "hfx")),
)
HEAT_FLUX = 0.24


def load_inputs(directory: Path = ORACLE_DIR) -> dict[str, np.ndarray]:
    with np.load(directory / "thetam_ext_inputs.npz") as z:
        return {name: z[name] for name in z.files}


def load_outputs(directory: Path = ORACLE_DIR,
                 inputs: dict | None = None) -> dict[str, np.ndarray]:
    inputs = load_inputs(directory) if inputs is None else inputs
    nz, ny, nx = inputs["t_m"].shape
    shape = (nx + 1, nz + 1, ny + 1)
    raw = np.fromfile(directory / "thetam_ext_outputs.bin", "<f4")
    size = int(np.prod(shape))
    if raw.size != size * len(RECORDS):
        raise ValueError(f"record holds {raw.size} words, expected "
                         f"{size * len(RECORDS)}")
    out = {}
    for n, (name, _kind) in enumerate(RECORDS):
        mem = raw[n * size:(n + 1) * size].reshape(shape, order="F")
        cut = mem[:nx, :nz, :ny]
        if name.endswith("_hfx"):
            cut = cut[:, :1, :]
        out[name] = np.ascontiguousarray(np.transpose(cut, (1, 2, 0)))
    return out


def port_outputs(inputs: dict) -> dict[str, np.ndarray]:
    """woof's words for every record of the second record."""
    import cupy as cp
    from woof.core.kernels import get_kernel
    f32 = np.float32
    nz, ny, nx = inputs["t_m"].shape
    n = nz * ny * nx
    tpb = 256
    dev = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=f32))
    thb = dev(np.full(nz, 300.0, f32))
    out: dict[str, np.ndarray] = {}

    # phy_prep's th_phy (kernels/theta_m.cu theta_dry_from_moist + thb, the
    # physics window's words) and rho are graded by the first record; here
    # they are the inputs WRF's hflux read, kept for the comparison table.
    dry = cp.empty((nz, ny, nx), f32)
    get_kernel("theta_m", "theta_dry_from_moist")(
        ((n + tpb - 1) // tpb,), (tpb,),
        (dev(inputs["t_m"]), dev(inputs["qv"]), thb, np.int32(0),
         np.int32(nz), np.int32(ny * nx), dry))
    out["prep_th_phy"] = (f32(300.0) + cp.asnumpy(dry)).astype(f32)

    # phy_prep_part2: the coupled pair (decouple = 1) and the decoupled one.
    mut = dev((inputs["mub"] + inputs["mu"]).astype(f32))
    kernel = get_kernel("theta_m_cumulus", "phy_prep_part2_rthften_theta_m")
    for name, decouple in (("coupled", 1), ("decoupled", 0)):
        rth = dev(inputs["rthften"])
        rqv = dev(inputs["rqvften"])
        kernel(((n + tpb - 1) // tpb,), (tpb,),
               (rth, rqv, dev(inputs["t_m"]), dev(inputs["qv"]), thb,
                np.int32(0), dev(inputs["c1h"]), dev(inputs["c2h"]), mut,
                np.int32(decouple), np.int32(nz), np.int32(ny * nx)))
        out[f"part2_{name}_rthften"] = cp.asnumpy(rth)
        out[f"part2_{name}_rqvften"] = cp.asnumpy(rqv)

    # vertical_diffusion_2's hflux, theta_m branch.
    ones2 = np.ones((ny, nx), f32)
    common = [
        dev(inputs["u"]), dev(inputs["v"]), dev(np.zeros((nz + 1, ny, nx), f32)),
        dev(inputs["ph"]), dev(inputs["phb"]), dev(inputs["alt"]),
        dev(inputs["qv"]), dev(ones2), dev(np.ones((ny, nx + 1), f32)),
        dev(np.ones((ny + 1, nx), f32)), dev(inputs["fnm"]),
        dev(inputs["fnp"]), dev(inputs["dnw"]), dev(inputs["dnw"]),
        f32(1.0 / 3000.0), f32(1.0 / 3000.0), f32(3000.0), f32(3000.0),
        f32(0.0), f32(0.0), f32(0.0), np.int32(1)]
    tail = [np.int32(nz), np.int32(ny), np.int32(nx), np.int32(1),
            np.int32(1), np.int32(1)]
    grid = ((nx + tpb - 1) // tpb, ny, 1)
    for isfflx in (0, 1, 2):
        rt = dev(inputs["t_tendf"])
        hfx = dev(inputs["hfx"])
        qfx = dev(inputs["qfx"])
        if isfflx in (0, 2):
            get_kernel("smag2d_theta_m", "wrf_smag_surface_heat_const_theta_m")(
                grid, (tpb, 1, 1),
                tuple(common + [f32(HEAT_FLUX), hfx, np.int32(1), qfx,
                                np.int32(1), dev(inputs["t_m"]), thb,
                                np.int32(0), rt] + tail))
        get_kernel("smag2d_theta_m", "wrf_smag_surface_scalars_theta_m")(
            grid, (tpb, 1, 1),
            tuple(common + [hfx, qfx, np.int32(isfflx == 1), np.int32(0),
                            dev(inputs["t_m"]), thb, np.int32(0), rt, rt]
                  + tail))
        out[f"hflux_isfflx{isfflx}_rt_tendf"] = cp.asnumpy(rt)
        out[f"hflux_isfflx{isfflx}_hfx"] = cp.asnumpy(hfx)[None]
    return out


def compare(port: dict, record: dict) -> dict[str, dict[str, int]]:
    """Per record: words compared and words not bit-identical."""
    result = {}
    for name, a in port.items():
        if name not in record:
            continue
        a = np.ascontiguousarray(a, np.float32).view(np.uint32)
        b = np.ascontiguousarray(record[name], np.float32).view(np.uint32)
        if a.shape != b.shape:
            raise ValueError(f"{name}: {a.shape} vs {b.shape}")
        result[name] = {"words": int(a.size), "differ": int((a != b).sum())}
    return result


def record_sha256(directory: Path = ORACLE_DIR) -> dict[str, str]:
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("thetam_ext_inputs.npz", "thetam_ext_outputs.bin")}


# ---------------------------------------------------------------------------
# moist_physics_prep_em / moist_physics_finish_em through the window
# ---------------------------------------------------------------------------

class _WindowState:
    """The few DomainState members microphysics_window reads and writes."""

    def __init__(self, cp, thp, qv, thb):
        self.thp = thp
        self.qv = qv
        self.thb = thb
        self.h_diabatic = cp.zeros_like(thp)
        self.theta_m = True
        self._scratch = {}
        self._cp = cp
        self.p = thp

    def scratch(self, shape, name, dtype=None):
        array = self._scratch.get(name)
        if array is None or array.shape != tuple(shape):
            array = self._cp.empty(tuple(shape), dtype=self._cp.float32)
            self._scratch[name] = array
        return array

    def existing_scratch(self, name):
        return self._scratch.get(name)


#: The stand-in schemes, each finishing as one family of adapters does.
WINDOW_FINISHES = ("moist_physics_finish", "thompson_adapter_finish",
                   "tc_finish")


def window_outputs(inputs: dict, record: dict, finish: str,
                   case: str) -> dict[str, np.ndarray]:
    """theta_m', h_diabatic and th_phy_m_t0 after microphysics_window around
    a stand-in scheme whose call adds the record's theta increment
    (``th_post``) to the prep's th_phy and leaves the record's post-call
    vapour, finishing through ``finish``; plus the window's own
    moist_physics_prep_em h_diabatic.  ``case`` is "a" (mp_tend_lim 10) or
    "b" (0.07, the clamped call)."""
    import cupy as cp
    from woof.core import theta_m
    from woof.verify.thetam_oracle import DT, LIM
    f32 = np.float32
    nz, ny, nx = inputs["t_dry"].shape
    dev = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=f32))
    state = _WindowState(cp, dev(record["init_t_m"]), dev(inputs["qv"]),
                         dev(np.full(nz, 300.0, f32)))
    cfg = SimpleNamespace(mp_physics=8, no_mp_heating=0,
                          mp_tend_lim=f32(LIM[case]), specified=False,
                          nested=False, spec_zone=0)
    seen = {}
    with theta_m.microphysics_window(state, cfg):
        seen["h_pre"] = cp.asnumpy(state.scratch((nz, ny, nx), theta_m.H_PRE_SLOT))
        # The adapters' save_pre_mp_theta and their th = thb + thp.
        thb3 = state.thb[:, None, None]
        cp.add(thb3, state.thp, out=state.h_diabatic)
        th = cp.empty((nz, ny, nx), f32)
        cp.add(thb3, state.thp, out=th)
        th = th + dev(inputs["th_post"])
        state.qv[...] = dev(inputs["qv_post"])
        if finish == "moist_physics_finish":
            from woof.core.microphysics import moist_physics_finish
            moist_physics_finish(state, cfg, th, DT)
        elif finish == "thompson_adapter_finish":
            from woof.core.kernels import get_kernel
            from woof.core.state import DTYPE
            size = nz * ny * nx
            zeros2 = cp.zeros((ny, nx), f32)
            pii = cp.ones((nz, ny, nx), f32)
            out_th = cp.empty((nz, ny, nx), f32)
            get_kernel("thompson", "thompson_adapter_finish")(
                ((size + 255) // 256,), (256,),
                (th, pii, out_th, state.thp, state.h_diabatic, zeros2,
                 zeros2, zeros2, cp.zeros((ny, nx), f32),
                 DTYPE(DTYPE(cfg.mp_tend_lim) * DTYPE(DT)), DTYPE(DT),
                 np.int32(0), np.int32(size), np.int32(ny * nx)))
            theta_m.finish_microphysics(state, cfg, out_th, DT)
        elif finish == "tc_finish":
            from woof.core.kernels import get_kernel
            from woof.core.state import DTYPE
            size = nz * ny * nx
            saved = state.h_diabatic.copy()
            get_kernel("thompson_classic", "tc_finish")(
                ((size + 255) // 256,), (256,),
                (th, state.thp, state.h_diabatic, saved,
                 DTYPE(DTYPE(cfg.mp_tend_lim) * DTYPE(DT)), DTYPE(DT),
                 np.int32(0), np.int32(size)))
            theta_m.finish_microphysics(state, cfg, th, DT)
        else:
            raise ValueError(finish)
    return {
        "mp_prep_h_diabatic": seen["h_pre"],
        f"finish_{case}_t_new": cp.asnumpy(state.thp),
        f"finish_{case}_h_diabatic": cp.asnumpy(state.h_diabatic),
        f"finish_{case}_th_phy_m_t0": cp.asnumpy(
            state.scratch((nz, ny, nx), theta_m.DRY_SLOT)),
    }
