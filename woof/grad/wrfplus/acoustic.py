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
    return _device_module(unit, real, cp.cuda.Device().id, defines)


@lru_cache(maxsize=None)
def _device_module(unit: str, real: str, device: int, defines: tuple = ()):
    import cupy as cp
    source = (KERNELS / f"{unit}.cu").read_text(encoding="utf-8")
    return cp.RawModule(code=source,
                        options=("--fmad=false", f"-DWRFPLUS_REAL={real}")
                        + tuple(f"-D{k}={v}" for k, v in defines))


def _kernel(unit: str, name: str, dtype, defines: tuple = ()):
    return _module(unit, _real(dtype), defines).get_function(name)


def _blocks(n: int) -> tuple[int]:
    return ((n + THREADS - 1) // THREADS,)


def _dtype(arrays) -> np.dtype:
    kinds = {np.dtype(a.dtype) for a in arrays}
    if len(kinds) != 1:
        raise TypeError(f"mixed real types {sorted(map(str, kinds))}")
    return kinds.pop()


def _check(geom, named, shapes=None) -> None:
    """Shapes, contiguity and distinct buffers: the kernels index raw
    C-order memory and declare every pointer ``__restrict__``."""
    if geom.nz < 3:
        raise ValueError("advance_uv/advance_w need nz >= 3 (they read "
                         "u/v levels 0-2)")
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
            if ptr in seen:
                raise ValueError(f"{label} {name} aliases {seen[ptr]}")
            seen[ptr] = f"{label} {name}"


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
                "fnp", "c1h", "c2h", "c1f", "c2f", "mub2d", "cqw")

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
    """advance_w_phi's scalars (msf == 1, no damper: WRFPLUS has damp_opt
    0 only; ``dampmag`` other than 0 is refused)."""
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

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (i(self.nz), i(self.ny), i(self.nx), i(self.spec_zone),
                i(self.base3d), i(self.top_lid), i(self.moist_cq),
                i(self.boundary_x), i(self.boundary_y), r(self.cf1),
                r(self.cf2), r(self.cf3), r(self.rdx), r(self.rdy),
                r(self.dtau), r(self.epssm))


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
        "c2f": (nz + 1,), "mub2d": col, "cqw": wlev,
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
