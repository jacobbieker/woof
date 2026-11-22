"""Acoustic-substep TL/AD launchers (WRFPLUS module_small_step_em_{tl,ad}).

Each routine takes its trajectory (the base state the primal read) as a
dict of device arrays in woof's (k, j, i) layout, the same names as the
primal launcher in woof/core/acoustic.py, and the geometry as a
:class:`UVGeometry`.  The real type is the arrays' dtype: float64 (the
test and oracle build, CUDA and the host backend) or float32 (Metal,
which has no binary64).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ._mode import tl_options

KERNELS = Path(__file__).resolve().parent / "kernels"
THREADS = 256
#: Levels per thread in the adjoint's face pass: dpn is shared within a
#: chunk; more chunks give more threads.
AD_KCHUNK = 8

#: Trajectory arrays of advance_uv, in kernel argument order.
UV_TRAJECTORY = ("ru_t", "rv_t", "p_pp", "p_pp_old", "ph_pp", "php", "phb",
                 "alt", "al_pp", "pb", "mup", "mu_pp", "mub2d", "c1h", "c2h",
                 "fnm", "fnp", "rdnw", "cqu", "cqv", "mudf", "msfu", "msfv")

#: The perturbed inputs of advance_uv besides u_pp/v_pp, in kernel order.
UV_ACTIVE = ("ru_t", "rv_t", "p_pp", "p_pp_old", "ph_pp", "php", "alt",
             "al_pp", "mup", "mu_pp", "cqu", "cqv", "mudf")


@dataclass(frozen=True)
class UVGeometry:
    """advance_uv's scalars, as woof/core/acoustic.py passes them.

    ``xscale``/``yscale`` are the strict launcher's ``-emdiv*dx`` and
    ``-emdiv*dy``; ``msfv`` is then MSFV_INV, as in the primal.
    """
    nz: int
    ny: int
    nx: int
    rdx: float
    rdy: float
    dtau: float
    smdiv: float = 0.0
    cf1: float = 0.0
    cf2: float = 0.0
    cf3: float = 0.0
    spec_zone: int = 0
    base3d: int = 1
    top_lid: int = 0
    moist_cq: int = 0
    use_emdiv: int = 0
    has_msf: int = 0
    xscale: float = 0.0
    yscale: float = 0.0

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (i(self.nz), i(self.ny), i(self.nx), i(self.spec_zone),
                i(self.base3d), i(self.top_lid), i(self.moist_cq),
                i(self.use_emdiv), i(self.has_msf), r(self.rdx), r(self.rdy),
                r(self.dtau), r(self.smdiv), r(self.cf1), r(self.cf2),
                r(self.cf3), r(self.xscale), r(self.yscale))


def _real(dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype == np.float64:
        return "double"
    if dtype == np.float32:
        return "float"
    raise TypeError(f"TL/AD kernels run in float64 or float32, not {dtype}")


def _module(unit: str, real: str, defines: tuple = ()):
    import cupy as cp
    return _device_module(unit, real, cp.cuda.Device().id, defines, tl_options())


@lru_cache(maxsize=None)
def _device_module(unit: str, real: str, device: int, defines: tuple = (),
                   build: tuple = ()):
    import cupy as cp
    source = (KERNELS / f"{unit}.cu").read_text(encoding="utf-8")
    return cp.RawModule(code=source,
                        options=("--fmad=false", f"-DWRFPLUS_REAL={real}")
                        + tuple(f"-D{k}={v}" for k, v in defines) + build)


def _kernel(unit: str, name: str, dtype, defines: tuple = ()):
    return _module(unit, _real(dtype), defines).get_function(name)


def _blocks(n: int) -> tuple[int]:
    return ((n + THREADS - 1) // THREADS,)


def _dtype(arrays) -> np.dtype:
    kinds = {np.dtype(a.dtype) for a in arrays}
    if len(kinds) != 1:
        raise TypeError(f"mixed real types {sorted(map(str, kinds))}")
    return kinds.pop()


def _check(geom, named, shapes=None, min_nz: int = 3) -> None:
    """Shapes, contiguity and distinct buffers: the kernels index raw
    C-order memory and declare every pointer ``__restrict__``."""
    if geom.nz < 1:
        raise ValueError("nz must be >= 1")
    if geom.nz < min_nz:
        raise ValueError(f"nz must be >= {min_nz} here (advance_uv, "
                         "advance_w and set_w_surface read u/v levels 0-2)")
    shapes = uv_shapes(geom) if shapes is None else shapes
    seen = {}
    for label, name, a in named:
        if tuple(a.shape) != shapes[name]:
            raise ValueError(f"{label} {name}: shape {tuple(a.shape)}, "
                             f"expected {shapes[name]}")
        flags = getattr(a, "flags", None)
        if flags is not None and not flags.c_contiguous:
            raise ValueError(f"{label} {name} is not C-contiguous")
        ptr = getattr(getattr(a, "data", None), "ptr", None)
        if ptr is not None:
            # Byte ranges, not start addresses: two views that overlap
            # without sharing a start would still alias under __restrict__.
            end = ptr + max(int(getattr(a, "nbytes", 0)), 1)
            for (lo, hi), other in seen.items():
                if ptr < hi and lo < end:
                    raise ValueError(f"{label} {name} aliases {other}")
            seen[(ptr, end)] = f"{label} {name}"


def advance_uv_tl(u_pp, v_pp, u_d, v_d, traj: dict, pert: dict,
                  geom: UVGeometry, *, nl: bool = True) -> None:
    """WRFPLUS g_advance_uv: update u_d/v_d (and u_pp/v_pp when ``nl``).

    ``pert`` holds the perturbations of :data:`UV_ACTIVE`, by name.
    """
    traj_args = tuple(traj[n] for n in UV_TRAJECTORY)
    pert_args = tuple(pert[n] for n in UV_ACTIVE)
    _check(geom, [("state", "u_pp", u_pp), ("state", "v_pp", v_pp),
                  ("pert", "u_pp", u_d), ("pert", "v_pp", v_d)]
           + [("traj", n, traj[n]) for n in UV_TRAJECTORY]
           + [("pert", n, pert[n]) for n in UV_ACTIVE])
    dtype = _dtype((u_pp, v_pp, u_d, v_d) + traj_args + pert_args)
    g = geom
    _kernel("advance_uv", "advance_uv_tl", dtype)(
        _blocks(g.nz * (g.ny + 1) * (g.nx + 1)), (THREADS,),
        (u_pp, v_pp, u_d, v_d) + traj_args + pert_args
        + (np.int32(bool(nl)),) + g.args(dtype))


def advance_uv_ad(u_a, v_a, traj: dict, adj: dict, geom: UVGeometry) -> None:
    """WRFPLUS a_advance_uv: add the adjoints of :data:`UV_ACTIVE` into
    ``adj`` (by name).  ``u_a``/``v_a`` pass through unchanged, because
    the update is ``u_out = u_in + ...``.
    """
    import cupy as cp
    traj_args = tuple(traj[n] for n in UV_TRAJECTORY)
    dtype = _dtype((u_a, v_a) + traj_args + tuple(adj[n] for n in UV_ACTIVE))
    _check(geom, [("adj", "u_pp", u_a), ("adj", "v_pp", v_a)]
           + [("traj", n, traj[n]) for n in UV_TRAJECTORY]
           + [("adj", n, adj[n]) for n in UV_ACTIVE])
    g = geom
    lu, lv = g.ny * (g.nx + 1), (g.ny + 1) * g.nx
    kc = AD_KCHUNK
    if kc < 1:
        raise ValueError(f"AD_KCHUNK must be >= 1, not {kc}")
    nch = (g.nz + kc - 1) // kc
    su = cp.empty(3 * (g.nz + nch) * lu, dtype=dtype)
    sv = cp.empty(3 * (g.nz + nch) * lv, dtype=dtype)
    geo = g.args(dtype)
    _kernel("advance_uv", "advance_uv_ad_face", dtype)(
        _blocks(nch * (g.ny + 1) * (g.nx + 1)), (THREADS,),
        (u_a, v_a) + traj_args
        + (adj["ru_t"], adj["rv_t"], adj["cqu"], adj["cqv"], su, sv,
           np.int32(kc)) + geo)
    _kernel("advance_uv", "advance_uv_ad_gather", dtype)(
        _blocks((g.nz + 1) * g.ny * g.nx), (THREADS,),
        traj_args + (su, sv, adj["ph_pp"], adj["php"], adj["alt"],
                     adj["al_pp"], adj["p_pp"], adj["p_pp_old"], adj["mup"],
                     adj["mu_pp"], adj["mudf"], np.int32(kc)) + geo)


def uv_shapes(geom: UVGeometry) -> dict[str, tuple[int, ...]]:
    """Array shapes of advance_uv's trajectory and outputs, by name."""
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    mass, wlev, col = (nz, ny, nx), (nz + 1, ny, nx), (ny, nx)
    return {
        "u_pp": (nz, ny, nx + 1), "v_pp": (nz, ny + 1, nx),
        "ru_t": (nz, ny, nx + 1), "rv_t": (nz, ny + 1, nx),
        "p_pp": mass, "p_pp_old": mass, "ph_pp": wlev, "php": wlev,
        "phb": wlev if geom.base3d else (nz + 1,),
        "alt": mass, "al_pp": mass, "pb": mass,
        "mup": col, "mu_pp": col, "mub2d": col,
        "c1h": (nz,), "c2h": (nz,), "fnm": (nz + 1,), "fnp": (nz + 1,),
        "rdnw": (nz,), "cqu": (nz, ny, nx + 1), "cqv": (nz, ny + 1, nx),
        "mudf": col, "msfu": (ny, nx + 1), "msfv": (ny + 1, nx),
    }


#: Trajectory arrays of advance_w (with calc_p_rho), in kernel argument
#: order, as woof/core/acoustic.py passes them to advance_w_phi.
W_TRAJECTORY = ("rw_t", "rph_t", "ww_pp", "mu_pp", "mu_pp_old", "th_pp",
                "th_pp_old", "thp", "php", "phb", "alt", "c2a", "a", "alpha",
                "gam", "mup", "u_pp", "v_pp", "ht", "rdn", "rdnw", "fnm",
                "fnp", "c1h", "c2h", "c1f", "c2f", "mub2d", "cqw", "msft")

#: The perturbed inputs of advance_w besides the state (w_pp, ph_pp,
#: p_pp, al_pp), in kernel order.
W_ACTIVE = ("rw_t", "rph_t", "ww_pp", "mu_pp", "mu_pp_old", "th_pp",
            "th_pp_old", "thp", "php", "alt", "c2a", "a", "alpha", "gam",
            "mup", "u_pp", "v_pp", "cqw")

#: advance_w's in/out state.  p_pp and al_pp are outputs on the solved
#: columns and pass through on the frame, as in the primal.
W_STATE = ("w_pp", "ph_pp", "p_pp", "al_pp")


@dataclass(frozen=True)
class WGeometry:
    """advance_w_phi's scalars (no damper: WRFPLUS has damp_opt 0 only;
    ``dampmag`` other than 0 is refused).  ``has_msf`` selects
    advance_w_phi_msf's map factors (WRF advance_w's msfty, with msftx =
    msfty = msft); without it ``msft`` is not read, but the trajectory
    must still carry an (ny, nx) array."""
    nz: int
    ny: int
    nx: int
    rdx: float
    rdy: float
    dtau: float
    epssm: float = 0.1
    cf1: float = 0.0
    cf2: float = 0.0
    cf3: float = 0.0
    spec_zone: int = 0
    base3d: int = 1
    top_lid: int = 0
    moist_cq: int = 0
    boundary_x: int = 0
    boundary_y: int = 0
    dampmag: float = 0.0
    has_msf: int = 0

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (i(self.nz), i(self.ny), i(self.nx), i(self.spec_zone),
                i(self.base3d), i(self.top_lid), i(self.moist_cq),
                i(self.boundary_x), i(self.boundary_y), r(self.cf1),
                r(self.cf2), r(self.cf3), r(self.rdx), r(self.rdy),
                r(self.dtau), r(self.epssm), i(self.has_msf))


def _w_defines(geom: WGeometry) -> tuple:
    if geom.dampmag != 0.0:
        raise NotImplementedError("advance_w TL/AD: the damp_opt=3 w damper "
                                  "is not linearised (WRFPLUS has none)")
    return (("WV_LEV", geom.nz + 1),)


def advance_w_tl(state: dict, pert_state: dict, traj: dict, pert: dict,
                 geom: WGeometry, *, nl: bool = True) -> None:
    """WRFPLUS g_advance_w then g_calc_p_rho (step 0): update
    ``pert_state`` (and ``state`` when ``nl``) in place, both dicts of
    :data:`W_STATE`.  ``pert`` holds the perturbations of
    :data:`W_ACTIVE`, by name."""
    defines = _w_defines(geom)
    sh = w_shapes(geom)
    st = tuple(state[n] for n in W_STATE)
    sd = tuple(pert_state[n] for n in W_STATE)
    traj_args = tuple(traj[n] for n in W_TRAJECTORY)
    pert_args = tuple(pert[n] for n in W_ACTIVE)
    _check(geom, [("state", n, state[n]) for n in W_STATE]
           + [("pert", n, pert_state[n]) for n in W_STATE]
           + [("traj", n, traj[n]) for n in W_TRAJECTORY]
           + [("pert", n, pert[n]) for n in W_ACTIVE], sh)
    dtype = _dtype(st + sd + traj_args + pert_args)
    g = geom
    _kernel("advance_w", "advance_w_tl", dtype, defines)(
        _blocks(g.ny * g.nx), (THREADS,),
        st + sd + traj_args + pert_args + (np.int32(bool(nl)),)
        + g.args(dtype))


def advance_w_ad(adj_state: dict, state: dict, traj: dict, adj: dict,
                 geom: WGeometry) -> None:
    """WRFPLUS a_calc_p_rho then a_advance_w.  ``adj_state`` (the
    :data:`W_STATE` adjoints) maps output adjoints to input adjoints in
    place; ``state`` holds the input w_pp/ph_pp; the adjoints of
    :data:`W_ACTIVE` are added into ``adj``."""
    import cupy as cp
    defines = _w_defines(geom)
    sh = w_shapes(geom)
    sa = tuple(adj_state[n] for n in W_STATE)
    s0 = (state["w_pp"], state["ph_pp"])
    traj_args = tuple(traj[n] for n in W_TRAJECTORY)
    _check(geom, [("adj", n, adj_state[n]) for n in W_STATE]
           + [("state", n, state[n]) for n in ("w_pp", "ph_pp")]
           + [("traj", n, traj[n]) for n in W_TRAJECTORY]
           + [("adj", n, adj[n]) for n in W_ACTIVE], sh)
    dtype = _dtype(sa + s0 + traj_args + tuple(adj[n] for n in W_ACTIVE))
    g = geom
    geo = g.args(dtype)
    aw0 = cp.empty((g.ny, g.nx), dtype=dtype)
    col = tuple(adj[n] for n in W_ACTIVE if n not in ("u_pp", "v_pp"))
    _kernel("advance_w", "advance_w_ad", dtype, defines)(
        _blocks(g.ny * g.nx), (THREADS,),
        sa + s0 + traj_args + col + (aw0,) + geo)
    _kernel("advance_w", "advance_w_ad_bc", dtype, defines)(
        _blocks(3 * (g.ny + 1) * (g.nx + 1)), (THREADS,),
        (aw0, traj["ht"], adj["u_pp"], adj["v_pp"]) + geo)


def w_shapes(geom: WGeometry) -> dict[str, tuple[int, ...]]:
    """Array shapes of advance_w's state and trajectory, by name."""
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    mass, wlev, col = (nz, ny, nx), (nz + 1, ny, nx), (ny, nx)
    return {
        "w_pp": wlev, "ph_pp": wlev, "p_pp": mass, "al_pp": mass,
        "rw_t": wlev, "rph_t": wlev, "ww_pp": wlev,
        "mu_pp": col, "mu_pp_old": col, "th_pp": mass, "th_pp_old": mass,
        "thp": mass, "php": wlev, "phb": wlev if geom.base3d else (nz + 1,),
        "alt": mass, "c2a": mass, "a": wlev, "alpha": wlev, "gam": wlev,
        "mup": col, "u_pp": (nz, ny, nx + 1), "v_pp": (nz, ny + 1, nx),
        "ht": col, "rdn": (nz,), "rdnw": (nz,), "fnm": (nz + 1,),
        "fnp": (nz + 1,), "c1h": (nz,), "c2h": (nz,), "c1f": (nz + 1,),
        "c2f": (nz + 1,), "mub2d": col, "cqw": wlev, "msft": col,
    }


#: calc_coef_w's outputs (WRF small_step_prep's c2a and calc_coef_w's
#: factors, as woof's calc_coefs fuses them), active inputs and passive
#: column constants, in kernel order.
CW_STATE = ("c2a", "a", "alpha", "gam")
CW_ACTIVE = ("p", "alt", "mup", "cqw")
CW_PASSIVE = ("rdn", "rdnw", "c1h", "c2h", "c1f", "c2f", "mub2d")


@dataclass(frozen=True)
class CWGeometry:
    """calc_coefs's scalars, as woof/core/acoustic.py passes them."""
    nz: int
    ny: int
    nx: int
    dtau: float
    epssm: float = 0.1
    moist_cq: int = 0
    top_lid: int = 0

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (i(self.nz), i(self.ny), i(self.nx), i(self.moist_cq),
                i(self.top_lid), r(self.dtau), r(self.epssm))


def cw_shapes(geom: CWGeometry) -> dict[str, tuple[int, ...]]:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    mass, wlev, col = (nz, ny, nx), (nz + 1, ny, nx), (ny, nx)
    return {"c2a": mass, "a": wlev, "alpha": wlev, "gam": wlev,
            "p": mass, "alt": mass, "mup": col, "cqw": wlev,
            "rdn": (nz,), "rdnw": (nz,), "c1h": (nz,), "c2h": (nz,),
            "c1f": (nz + 1,), "c2f": (nz + 1,), "mub2d": col}


def calc_coef_w_tl(state: dict, pert_state: dict, traj: dict, pert: dict,
                   geom: CWGeometry, *, nl: bool = True) -> None:
    """woof calc_coefs' TL (c2a, then WRFPLUS g_calc_coef_w): write
    ``pert_state`` (and ``state`` when ``nl``), dicts of :data:`CW_STATE`;
    ``pert`` holds the perturbations of :data:`CW_ACTIVE`."""
    sh = cw_shapes(geom)
    st = tuple(state[n] for n in CW_STATE)
    sd = tuple(pert_state[n] for n in CW_STATE)
    tr = tuple(traj[n] for n in CW_ACTIVE)
    pe = tuple(pert[n] for n in CW_ACTIVE)
    ps = tuple(traj[n] for n in CW_PASSIVE)
    _check(geom, [("state", n, state[n]) for n in CW_STATE]
           + [("pert", n, pert_state[n]) for n in CW_STATE]
           + [("traj", n, traj[n]) for n in CW_ACTIVE + CW_PASSIVE]
           + [("pert", n, pert[n]) for n in CW_ACTIVE], sh)
    dtype = _dtype(st + sd + tr + pe + ps)
    _kernel("calc_coef_w", "calc_coef_w_tl", dtype, (("WV_LEV", geom.nz + 1),))(
        _blocks(geom.ny * geom.nx), (THREADS,),
        st + sd + tr + pe + ps + (np.int32(bool(nl)),) + geom.args(dtype))


def calc_coef_w_ad(adj_state: dict, traj: dict, adj: dict,
                   geom: CWGeometry) -> None:
    """Adjoint of :func:`calc_coef_w_tl`: consume the :data:`CW_STATE`
    adjoints of the written entries and add those of :data:`CW_ACTIVE`
    into ``adj``."""
    sh = cw_shapes(geom)
    sa = tuple(adj_state[n] for n in CW_STATE)
    tr = tuple(traj[n] for n in CW_ACTIVE)
    aa = tuple(adj[n] for n in CW_ACTIVE)
    ps = tuple(traj[n] for n in CW_PASSIVE)
    _check(geom, [("adj", n, adj_state[n]) for n in CW_STATE]
           + [("traj", n, traj[n]) for n in CW_ACTIVE + CW_PASSIVE]
           + [("adj", n, adj[n]) for n in CW_ACTIVE], sh)
    dtype = _dtype(sa + tr + aa + ps)
    _kernel("calc_coef_w", "calc_coef_w_ad", dtype, (("WV_LEV", geom.nz + 1),))(
        _blocks(geom.ny * geom.nx), (THREADS,),
        sa + tr + aa + ps + geom.args(dtype))


#: advance_mu_t's state (woof advance_mu_th's in/out and output arrays,
#: plus muts for the specified-frame kernel), active inputs and passive
#: constants, in kernel order.  thp is the theta the strict primal
#: advects (WRF t_1) and ww_ref the stage-start Omega (WRF ww_1).
MT_STATE = ("mu_pp", "th_pp", "mu_old", "th_old", "p_old", "mudf", "ww_pp",
            "muts")
MT_ACTIVE = ("u_pp", "v_pp", "u", "v", "mup", "rmu_t", "thp", "rth_t",
             "ww_ref", "p_pp")
MT_PASSIVE = ("dnw", "rdnw", "fnm", "fnp", "c1h", "c2h", "mub2d", "msft",
              "msfu", "msfv")


@dataclass(frozen=True)
class MTGeometry:
    """advance_mu_th's scalars, as woof/core/acoustic.py passes them.
    ``mapped`` selects advance_mu_th_msf's map factors (msfv is the
    strict path's msfv_inv, a multiplier); ``exact_frame`` adds
    advance_exact_frame_mu_t when ``spec_zone > 0``, as the strict
    launcher does."""
    nz: int
    ny: int
    nx: int
    rdx: float
    rdy: float
    dtau: float
    spec_zone: int = 0
    open_x: int = 0
    open_y: int = 0
    mapped: int = 0
    write_mudf: int = 1
    exact_frame: int = 1

    @property
    def frame(self) -> bool:
        return bool(self.exact_frame) and self.spec_zone > 0

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (i(self.nz), i(self.ny), i(self.nx), i(self.open_x),
                i(self.open_y), i(self.spec_zone), i(self.mapped),
                i(self.write_mudf), r(self.rdx), r(self.rdy), r(self.dtau))


def mt_shapes(geom: MTGeometry) -> dict[str, tuple[int, ...]]:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    mass, wlev, col = (nz, ny, nx), (nz + 1, ny, nx), (ny, nx)
    us, vs = (nz, ny, nx + 1), (nz, ny + 1, nx)
    return {"mu_pp": col, "th_pp": mass, "mu_old": col, "th_old": mass,
            "p_old": mass, "mudf": col, "ww_pp": wlev, "muts": col,
            "u_pp": us, "v_pp": vs, "u": us, "v": vs, "mup": col,
            "rmu_t": col, "thp": mass, "rth_t": mass, "ww_ref": wlev,
            "p_pp": mass, "dnw": (nz,), "rdnw": (nz,), "fnm": (nz + 1,),
            "fnp": (nz + 1,), "c1h": (nz,), "c2h": (nz,), "mub2d": col,
            "msft": col, "msfu": (ny, nx + 1), "msfv": (ny + 1, nx)}


def advance_mu_t_tl(state: dict, pert_state: dict, traj: dict, pert: dict,
                    geom: MTGeometry, *, nl: bool = True) -> None:
    """WRFPLUS g_advance_mu_t, then (``geom.frame``) the specified-frame
    g_spec_bdyupdate of t_2, mu_2 and muts: update ``pert_state`` (and
    ``state`` when ``nl``) in place, dicts of :data:`MT_STATE`; ``pert``
    holds the perturbations of :data:`MT_ACTIVE`."""
    sh = mt_shapes(geom)
    st = tuple(state[n] for n in MT_STATE)
    sd = tuple(pert_state[n] for n in MT_STATE)
    tr = tuple(traj[n] for n in MT_ACTIVE)
    pe = tuple(pert[n] for n in MT_ACTIVE)
    ps = tuple(traj[n] for n in MT_PASSIVE)
    _check(geom, [("state", n, state[n]) for n in MT_STATE]
           + [("pert", n, pert_state[n]) for n in MT_STATE]
           + [("traj", n, traj[n]) for n in MT_ACTIVE + MT_PASSIVE]
           + [("pert", n, pert[n]) for n in MT_ACTIVE], sh)
    dtype = _dtype(st + sd + tr + pe + ps)
    defines = (("WV_LEV", geom.nz + 1),)
    g = geom
    blocks = _blocks(g.ny * g.nx)
    _kernel("advance_mu_t", "advance_mu_t_tl", dtype, defines)(
        blocks, (THREADS,),
        st[:7] + sd[:7] + tr + pe + ps + (np.int32(bool(nl)),) + g.args(dtype))
    if g.frame:
        _kernel("advance_mu_t", "frame_mu_t_tl", dtype, defines)(
            blocks, (THREADS,),
            (state["mu_pp"], state["th_pp"], state["muts"], pert_state["mu_pp"],
             pert_state["th_pp"], pert_state["muts"], traj["rmu_t"],
             traj["rth_t"], traj["mup"], pert["rmu_t"], pert["rth_t"],
             pert["mup"], traj["mub2d"], np.int32(bool(nl))) + g.args(dtype))


def advance_mu_t_ad(adj_state: dict, traj: dict, adj: dict,
                    geom: MTGeometry) -> None:
    """Adjoint of :func:`advance_mu_t_tl`: map the :data:`MT_STATE`
    adjoints in place (written entries consumed) and add those of
    :data:`MT_ACTIVE` into ``adj``."""
    import cupy as cp
    sh = mt_shapes(geom)
    sa = tuple(adj_state[n] for n in MT_STATE)
    tr = tuple(traj[n] for n in MT_ACTIVE)
    aa = tuple(adj[n] for n in MT_ACTIVE)
    ps = tuple(traj[n] for n in MT_PASSIVE)
    _check(geom, [("adj", n, adj_state[n]) for n in MT_STATE]
           + [("traj", n, traj[n]) for n in MT_ACTIVE + MT_PASSIVE]
           + [("adj", n, adj[n]) for n in MT_ACTIVE], sh)
    dtype = _dtype(sa + tr + aa + ps)
    defines = (("WV_LEV", geom.nz + 1),)
    g = geom
    blocks = _blocks(g.ny * g.nx)
    if g.frame:
        _kernel("advance_mu_t", "frame_mu_t_ad", dtype, defines)(
            blocks, (THREADS,),
            (adj_state["mu_pp"], adj_state["th_pp"], adj_state["muts"],
             adj["rmu_t"], adj["rth_t"], adj["mup"]) + g.args(dtype))
    scratch_d = cp.empty((g.nz, g.ny, g.nx), dtype=dtype)
    scratch_h = cp.empty((g.nz, g.ny, g.nx), dtype=dtype)
    _kernel("advance_mu_t", "advance_mu_t_ad", dtype, defines)(
        blocks, (THREADS,),
        sa[:7] + tr + aa + ps + (scratch_d, scratch_h) + g.args(dtype))
    _kernel("advance_mu_t", "advance_mu_t_ad_gather", dtype, defines)(
        blocks, (THREADS,),
        tr + aa + ps + (scratch_d, scratch_h) + g.args(dtype))


#: small_step_init's outputs (woof small_step_init_uv + _column: WRF
#: small_step_prep, then calc_p_rho at step 0), active inputs (u0/v0/w0/
#: thp0/php0/mup0 the stage-start state, u/v/w/thp/php/mup the current
#: one) and passive constants, in kernel order.
SI_STATE = ("u_pp", "v_pp", "w_pp", "th_pp", "ph_pp", "mu_pp", "al_pp",
            "p_pp", "p_pp_old")
SI_ACTIVE = ("u0", "v0", "u", "v", "w0", "w", "thp0", "thp", "php0", "php",
             "p", "alt", "mup0", "mup")
SI_PASSIVE = ("mub2d", "c1h", "c2h", "c1f", "c2f", "rdnw", "msfu", "msfv",
              "msft")

#: small_step_finish's in/out state, active inputs (muts is WRF's
#: grid%muts, the strict launcher's muts_wrf) and passive constants.
SF_STATE = ("u", "v", "w", "thp", "php", "mup")
SF_ACTIVE = ("u_pp", "v_pp", "w_pp", "th_pp", "ph_pp", "mu_pp", "muts",
             "h_diabatic")
SF_PASSIVE = ("mub2d", "c1h", "c2h", "c1f", "c2f", "msfu", "msfv", "msft")


@dataclass(frozen=True)
class SSGeometry:
    """The strict small_step_init_*/finish_* scalars, as
    woof/core/dycore.py passes them.  ``boundary_x``/``boundary_y`` clamp
    the face masses at the edges (non-periodic).  The traj's ``msfv`` is
    what the strict launcher passes: MSFV_INV to init and msfv to finish,
    a multiplier in both."""
    nz: int
    ny: int
    nx: int
    has_msf: int = 0
    boundary_x: int = 1
    boundary_y: int = 1
    rk_step: int = 1
    remove_hdiab: int = 0
    hdiab_dt: float = 0.0

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (i(self.nz), i(self.ny), i(self.nx), i(self.has_msf),
                i(self.boundary_x), i(self.boundary_y), i(self.rk_step),
                i(self.remove_hdiab), r(self.hdiab_dt))


def ss_shapes(geom: SSGeometry) -> dict[str, tuple[int, ...]]:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    mass, wlev, col = (nz, ny, nx), (nz + 1, ny, nx), (ny, nx)
    us, vs = (nz, ny, nx + 1), (nz, ny + 1, nx)
    return {"u_pp": us, "v_pp": vs, "w_pp": wlev, "th_pp": mass,
            "ph_pp": wlev, "mu_pp": col, "al_pp": mass, "p_pp": mass,
            "p_pp_old": mass, "u0": us, "v0": vs, "u": us, "v": vs,
            "w0": wlev, "w": wlev, "thp0": mass, "thp": mass, "php0": wlev,
            "php": wlev, "p": mass, "alt": mass, "mup0": col, "mup": col,
            "muts": col, "h_diabatic": mass, "mub2d": col, "c1h": (nz,),
            "c2h": (nz,), "c1f": (nz + 1,), "c2f": (nz + 1,),
            "rdnw": (nz,), "msfu": (ny, nx + 1), "msfv": (ny + 1, nx),
            "msft": col}


def _ss_launch(geom, names, sh):
    _check(geom, names, sh, min_nz=1)


def _ss_nck(geom) -> int:
    return -(-geom.nz // AD_KCHUNK)


def _ss_scratch(geom, dtype):
    """The chunked adjoints' scratch: per-chunk face sums (u faces x2, v
    faces x2), column partials (x2) and the chunk-boundary ph'' adjoints.
    Every slot read by the gather is written first."""
    import cupy as cp
    g, n = geom, _ss_nck(geom)
    e = lambda *s: cp.empty(s, dtype=dtype)  # noqa: E731
    return (e(n, g.ny, g.nx + 1), e(n, g.ny, g.nx + 1), e(n, g.ny + 1, g.nx),
            e(n, g.ny + 1, g.nx), e(n, g.ny, g.nx), e(n, g.ny, g.nx),
            e(n, g.ny, g.nx))


_SS_DEFINES = (("SS_KCHUNK", AD_KCHUNK),)


def small_step_init_tl(state: dict, pert_state: dict, traj: dict, pert: dict,
                       geom: SSGeometry, *, nl: bool = True) -> None:
    """woof's strict small_step_init_uv + small_step_init_column (WRFPLUS
    g_small_step_prep, then g_calc_p_rho at step 0): write ``pert_state``
    (and ``state`` when ``nl``), dicts of :data:`SI_STATE`; ``pert`` holds
    the perturbations of :data:`SI_ACTIVE`."""
    sh = ss_shapes(geom)
    st = tuple(state[n] for n in SI_STATE)
    sd = tuple(pert_state[n] for n in SI_STATE)
    tr = tuple(traj[n] for n in SI_ACTIVE)
    pe = tuple(pert[n] for n in SI_ACTIVE)
    ps = tuple(traj[n] for n in SI_PASSIVE)
    _ss_launch(geom, [("state", n, state[n]) for n in SI_STATE]
               + [("pert", n, pert_state[n]) for n in SI_STATE]
               + [("traj", n, traj[n]) for n in SI_ACTIVE + SI_PASSIVE]
               + [("pert", n, pert[n]) for n in SI_ACTIVE], sh)
    dtype = _dtype(st + sd + tr + pe + ps)
    g = geom
    args = st + sd + tr + pe + ps + (np.int32(bool(nl)),) + g.args(dtype)
    _kernel("small_step", "small_step_init_uv_tl", dtype)(
        _blocks(g.nz * (g.ny + 1) * (g.nx + 1)), (THREADS,), args)
    _kernel("small_step", "small_step_init_column_tl", dtype)(
        _blocks(g.ny * g.nx), (THREADS,), args)


def small_step_init_ad(adj_state: dict, traj: dict, adj: dict,
                       geom: SSGeometry) -> None:
    """Adjoint of :func:`small_step_init_tl`: consume the :data:`SI_STATE`
    adjoints and add those of :data:`SI_ACTIVE` into ``adj``."""
    sh = ss_shapes(geom)
    sa = tuple(adj_state[n] for n in SI_STATE)
    tr = tuple(traj[n] for n in SI_ACTIVE)
    aa = tuple(adj[n] for n in SI_ACTIVE)
    ps = tuple(traj[n] for n in SI_PASSIVE)
    _ss_launch(geom, [("adj", n, adj_state[n]) for n in SI_STATE]
               + [("traj", n, traj[n]) for n in SI_ACTIVE + SI_PASSIVE]
               + [("adj", n, adj[n]) for n in SI_ACTIVE], sh)
    dtype = _dtype(sa + tr + aa + ps)
    g = geom
    scr = _ss_scratch(g, dtype)
    faces, chunk = scr[:4], scr[4:]
    n = _ss_nck(g)
    k = lambda name: _kernel("small_step", name, dtype, _SS_DEFINES)  # noqa: E731
    k("small_step_init_column_ad")(
        _blocks(n * g.ny * g.nx), (THREADS,),
        sa + tr + aa + chunk + ps + g.args(dtype))
    k("small_step_init_uv_ad")(
        _blocks(n * (g.ny + 1) * (g.nx + 1)), (THREADS,),
        sa + tr + aa + faces + ps + g.args(dtype))
    # ``pre`` (adj_state mu_pp, already consumed) is unused when init.
    k("small_step_mass_gather")(
        _blocks(g.ny * g.nx), (THREADS,),
        (adj["mup"], adj["mup0"], adj_state["mu_pp"], adj["php0"], adj["php"]) + faces
        + chunk + (np.int32(0),) + g.args(dtype))


def small_step_finish_tl(state: dict, pert_state: dict, traj: dict,
                         pert: dict, geom: SSGeometry, *,
                         nl: bool = True) -> None:
    """woof's strict small_step_finish_uv + small_step_finish_column
    (WRFPLUS G_SMALL_STEP_FINISH, with muus/muvs from calc_mu_uv_1 of
    muts): update ``pert_state`` (and ``state`` when ``nl``) in place,
    dicts of :data:`SF_STATE`; ``pert`` holds those of :data:`SF_ACTIVE`."""
    sh = ss_shapes(geom)
    st = tuple(state[n] for n in SF_STATE)
    sd = tuple(pert_state[n] for n in SF_STATE)
    tr = tuple(traj[n] for n in SF_ACTIVE)
    pe = tuple(pert[n] for n in SF_ACTIVE)
    ps = tuple(traj[n] for n in SF_PASSIVE)
    _ss_launch(geom, [("state", n, state[n]) for n in SF_STATE]
               + [("pert", n, pert_state[n]) for n in SF_STATE]
               + [("traj", n, traj[n]) for n in SF_ACTIVE + SF_PASSIVE]
               + [("pert", n, pert[n]) for n in SF_ACTIVE], sh)
    dtype = _dtype(st + sd + tr + pe + ps)
    g = geom
    args = st + sd + tr + pe + ps + (np.int32(bool(nl)),) + g.args(dtype)
    # The face pass reads mup before the column pass updates it.
    _kernel("small_step", "small_step_finish_uv_tl", dtype)(
        _blocks(g.nz * (g.ny + 1) * (g.nx + 1)), (THREADS,), args)
    _kernel("small_step", "small_step_finish_column_tl", dtype)(
        _blocks(g.ny * g.nx), (THREADS,), args)


def small_step_finish_ad(adj_state: dict, state: dict, traj: dict,
                         adj: dict, geom: SSGeometry) -> None:
    """Adjoint of :func:`small_step_finish_tl` about the pre-update
    ``state``: map the :data:`SF_STATE` adjoints in place and add those of
    :data:`SF_ACTIVE` into ``adj``."""
    sh = ss_shapes(geom)
    sa = tuple(adj_state[n] for n in SF_STATE)
    sb = tuple(state[n] for n in SF_STATE)
    tr = tuple(traj[n] for n in SF_ACTIVE)
    aa = tuple(adj[n] for n in SF_ACTIVE)
    ps = tuple(traj[n] for n in SF_PASSIVE)
    _ss_launch(geom, [("adj", n, adj_state[n]) for n in SF_STATE]
               + [("state", n, state[n]) for n in SF_STATE]
               + [("traj", n, traj[n]) for n in SF_ACTIVE + SF_PASSIVE]
               + [("adj", n, adj[n]) for n in SF_ACTIVE], sh)
    dtype = _dtype(sa + sb + tr + aa + ps)
    g = geom
    scr = _ss_scratch(g, dtype)
    faces, chunk = scr[:4], scr[4:]
    n = _ss_nck(g)
    k = lambda name: _kernel("small_step", name, dtype, _SS_DEFINES)  # noqa: E731
    k("small_step_finish_column_ad")(
        _blocks(n * g.ny * g.nx), (THREADS,),
        sa + sb + tr + aa + chunk + ps + g.args(dtype))
    k("small_step_finish_uv_ad")(
        _blocks(n * (g.ny + 1) * (g.nx + 1)), (THREADS,),
        sa + sb + tr + aa + faces + ps + g.args(dtype))
    # mu_pp's adjoint takes mup's output adjoint before the gather adds
    # the face-mass and column terms of the input mup; the ph slots
    # (th_pp, ph_pp) are unused when finishing.
    k("small_step_mass_gather")(
        _blocks(g.ny * g.nx), (THREADS,),
        (adj_state["mup"], adj["muts"], adj["mu_pp"], adj["th_pp"],
         adj["ph_pp"]) + faces + chunk + (np.int32(1),) + g.args(dtype))


#: advance_mu_t's names for the arrays advance_uv/advance_w call by the
#: strict primal's state names (woof's acoustic_mu_pp_old/_th_pp_old
#: scratch and state.p_pp_old).
MT_ALIAS = {"mu_old": "mu_pp_old", "th_old": "th_pp_old",
            "p_old": "p_pp_old"}


def _mt_view(d: dict) -> dict:
    return {n: d[MT_ALIAS.get(n, n)]
            for n in MT_STATE + MT_ACTIVE + MT_PASSIVE
            if MT_ALIAS.get(n, n) in d}


def _substep_written(geoms) -> tuple:
    return (("u_pp", "v_pp"),
            tuple(MT_ALIAS.get(n, n) for n in MT_STATE),
            W_STATE)


def acoustic_substep_tl(state: dict, pert: dict, geoms: tuple,
                        tape: list | None = None) -> None:
    """One acoustic substep in woof's strict order (advance_uv, then
    advance_mu_t, then advance_w with calc_p_rho), on dicts keyed by the
    strict primal's names (advance_mu_t's mu_old/th_old/p_old are
    mu_pp_old/th_pp_old/p_pp_old; ww_ref is the stage-start Omega).

    ``state`` advances along the trajectory (each operator is linearised
    about its own entry state) and ``pert`` maps to the substep's output
    perturbation, both in place.  When ``tape`` is a list, each
    operator's written state is appended before it runs, for
    :func:`acoustic_substep_ad`.  ``geoms`` is (UVGeometry, MTGeometry,
    WGeometry).
    """
    uvg, mtg, wg = geoms
    written = _substep_written(geoms)
    snap = (lambda k: tape.append({n: state[n].copy() for n in written[k]})) \
        if tape is not None else (lambda k: None)
    snap(0)
    advance_uv_tl(state["u_pp"], state["v_pp"], pert["u_pp"], pert["v_pp"],
                  state, pert, uvg, nl=True)
    snap(1)
    advance_mu_t_tl(_mt_view(state), _mt_view(pert), _mt_view(state),
                    _mt_view(pert), mtg, nl=True)
    snap(2)
    advance_w_tl(state, pert, state, pert, wg, nl=True)


def acoustic_substep_ad(adj: dict, state: dict, tape: list,
                        geoms: tuple) -> dict:
    """Adjoint of :func:`acoustic_substep_tl`: ``adj`` (every state and
    active name, :func:`woof.grad.wrfplus.cases.substep_active`, muts
    included even without a frame) maps from the output adjoint to the input adjoint in
    place.  ``state`` is the base after the substep and ``tape`` the
    three snapshots its TL appended (consumed from the end).  Returns
    the base before the substep (views of ``state`` and the tape), the
    ``state`` of the previous substep's adjoint."""
    uvg, mtg, wg = geoms
    entries, cur = [], dict(state)
    for _ in range(3):
        cur = {**cur, **tape.pop()}
        entries.append(cur)
    w_in, mt_in, uv_in = entries
    advance_w_ad(adj, w_in, w_in, adj, wg)
    advance_mu_t_ad(_mt_view(adj), _mt_view(mt_in), _mt_view(adj), mtg)
    advance_uv_ad(adj["u_pp"], adj["v_pp"], uv_in, adj, uvg)
    return uv_in
