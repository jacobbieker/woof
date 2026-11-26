"""diff_opt 1 / km_opt 4 TL/AD (P3a item 5): WRF compute_diff_metrics +
cal_deform_and_div (the deformation) and smag2d_km at diff_opt 1 on
woof's strict wrf_smag_deform and wrf_diff_opt1_km4 + wrf_smag_km_bc, and
the horizontal_diffusion rows on wrf_diff_opt1_horizontal
(docs/dev/P3A-WRF-DRYCORE-ADJOINT.md section 5, item 5).

Same conventions as :mod:`woof.grad.wrfplus.acoustic`: dicts of arrays by
name, ``*_tl(state, pert_state, traj, pert, geom, nl=True)`` updates the
perturbation (and the base state when ``nl``), ``*_ad`` adds into the
inputs' adjoints.  The deformation and the coefficients overwrite their
outputs, so their AD consumes (zeroes) the outputs' adjoints; the rows
accumulate into a tendency, whose adjoint their AD leaves as it is.

Deformation: ``u`` and ``v`` are the (coupled-to-map-factor) winds the
primal reads, ``php`` the w-level perturbation geopotential; ``phb`` is
the 3-D base geopotential.  Coefficients: the horizontal eddy viscosity
``xkmh`` and diffusivity ``xkhh = xkmh / prandtl`` from the deformation,
with the 10 * mlen cap and the cold-zero outer row of a bounded axis.
Rows: the u, v and w (xkmh) and theta and scalar (xkhh) tendencies of
the field, the coefficient and the column mass ``mut`` (hybrid mass
c1 mut + c2; theta mixes its departure from the initial field).
Not covered: diff_opt 2's slope limiter, the TC horizontal length, a 1-D
base geopotential and m_opt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acoustic import THREADS, _blocks, _check, _dtype
from .acoustic import _kernel as _unit_kernel

#: woof's dycore._PRANDTL, as the primal passes it (the strict-primal test
#: checks the two agree).
PRANDTL = 1.0 / 3.0


def _kernel(name: str, dtype):
    return _unit_kernel("diffusion", name, dtype)


def _named(label, names, d):
    return [(label, n, d[n]) for n in names]


def _axes(geom) -> None:
    for name in ("nz", "ny", "nx"):
        if getattr(geom, name) < 1:
            raise ValueError(f"{name} must be >= 1, got {getattr(geom, name)}")
    for name in ("boundary_x", "boundary_y"):
        if getattr(geom, name) not in (0, 1):
            raise ValueError(f"{name} is 0 or 1, got {getattr(geom, name)}")
    # the zeroing launches take an int32 count (as the primal's grids)
    if (geom.nz + 1) * (geom.ny + 1) * (geom.nx + 1) >= 2 ** 31:
        raise ValueError("the grid is too large for the int32 launches")


def _flags(geom) -> tuple:
    """The primal's boundary flags (dycore.smag_boundary_flags): 1 bounded,
    0 periodic, 2 periodic with WRF's cold seam slope at the top w level."""
    return tuple(1 if b else 2 * int(geom.seam_cold)
                 for b in (geom.boundary_x, geom.boundary_y))


# ------------------------------------------------------------- deformation

DF_STATE = ("defor11", "defor22", "defor12")
DF_ACTIVE = ("u", "v", "php")
DF_PASSIVE = ("phb", "msft", "msfu", "msfv", "fnm", "fnp", "dn", "dnw")


@dataclass(frozen=True)
class DFGeometry:
    """wrf_smag_deform's scalars: ``rdx = 1/dx`` and ``rdy = 1/dy`` as the
    primal rounds them, the bottom extrapolation weights cf1-cf3 and the
    bounded (open or specified) axes.  ``seam_cold`` = 1 is WRF's
    compute_diff_metrics at a periodic axis's seam: the slope at the top w
    level keeps its cold zero there (RunConfig.diff_metrics_seam_slope off,
    the primal's boundary flag 2, woof.core.dycore.smag_boundary_flags);
    0 takes the true slope."""
    nz: int
    ny: int
    nx: int
    rdx: float
    rdy: float
    cf1: float
    cf2: float
    cf3: float
    boundary_x: int = 0
    boundary_y: int = 0
    seam_cold: int = 0

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        # the primal's flags: 1 bounded, 0 periodic, 2 periodic with WRF's
        # cold seam slope at the top w level
        bx, by = ((1 if b else 2 * int(self.seam_cold))
                  for b in (self.boundary_x, self.boundary_y))
        return (r(self.rdx), r(self.rdy), r(self.cf1), r(self.cf2), r(self.cf3),
                i(bx), i(by), i(self.nz), i(self.ny), i(self.nx))

    def validate(self) -> None:
        _axes(self)
        if self.seam_cold not in (0, 1):
            raise ValueError(f"seam_cold is 0 or 1, got {self.seam_cold}")
        if self.nz < 3:
            raise ValueError(f"the deformation needs nz >= 3 (the bottom "
                             f"extrapolation reads levels 0-2), got {self.nz}")
        if not (np.isfinite(self.rdx) and np.isfinite(self.rdy)
                and self.rdx > 0 and self.rdy > 0):
            raise ValueError("rdx and rdy must be positive and finite")


def df_shapes(geom) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    mass, wlev = (nz, ny, nx), (nz + 1, ny, nx)
    return {"defor11": mass, "defor22": mass, "defor12": mass, "u": (nz, ny, nx + 1),
            "v": (nz, ny + 1, nx), "php": wlev, "phb": wlev, "msft": (ny, nx),
            "msfu": (ny, nx + 1), "msfv": (ny + 1, nx), "fnm": (nz,), "fnp": (nz,),
            "dn": (nz,), "dnw": (nz,)}


def df_tl(state, pert_state, traj, pert, geom: DFGeometry, *, nl=True):
    """WRFPLUS g_compute_diff_metrics + g_cal_deform_and_div on woof's
    strict wrf_smag_deform: defor11, defor22 and defor12 (overwritten) of
    u, v and php."""
    geom.validate()
    _check(geom, _named("state", DF_STATE, state) + _named("pert_state", DF_STATE, pert_state)
           + _named("traj", DF_ACTIVE + DF_PASSIVE, traj)
           + _named("pert", DF_ACTIVE, pert), df_shapes(geom), min_nz=3)
    arrs = (tuple(state[n] for n in DF_STATE) + tuple(pert_state[n] for n in DF_STATE)
            + tuple(a for n in DF_ACTIVE for a in (traj[n], pert[n]))
            + tuple(traj[n] for n in DF_PASSIVE))
    dtype = _dtype(arrs)
    _kernel("df_tl", dtype)(_blocks(geom.nz * geom.ny * geom.nx), (THREADS,),
                            arrs + (np.int32(bool(nl)),) + geom.args(dtype))


def df_ad(adj_state, traj, adj, geom: DFGeometry):
    """Adjoint of :func:`df_tl`: add into ``adj`` (u, v, php) and consume
    the defor11, defor22 and defor12 adjoints."""
    geom.validate()
    _check(geom, _named("adj", DF_STATE, adj_state)
           + _named("traj", DF_ACTIVE + DF_PASSIVE, traj)
           + _named("adj", DF_ACTIVE, adj), df_shapes(geom), min_nz=3)
    arrs = (tuple(adj_state[n] for n in DF_STATE) + tuple(traj[n] for n in DF_ACTIVE)
            + tuple(adj[n] for n in DF_ACTIVE) + tuple(traj[n] for n in DF_PASSIVE))
    dtype = _dtype(arrs)
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    n = nz * ny * (nx + 1) + nz * (ny + 1) * nx + (nz + 1) * ny * nx
    _kernel("df_ad", dtype)(_blocks(n), (THREADS,), arrs + geom.args(dtype))
    m = nz * ny * nx
    _kernel("df_zero3", dtype)(_blocks(m), (THREADS,),
                               tuple(adj_state[n] for n in DF_STATE) + (np.int32(m),))


# -------------------------------------------------- smag2d_km (diff_opt 1)

KM_STATE = ("xkmh", "xkhh")
KM_ACTIVE = DF_STATE
KM_PASSIVE = ("msft",)


@dataclass(frozen=True)
class KMGeometry:
    """wrf_diff_opt1_km4's scalars: the grid spacings, c_s, the Prandtl
    number and the bounded axes (whose outer row keeps zero)."""
    nz: int
    ny: int
    nx: int
    dx: float
    dy: float
    c_s: float = 0.25
    prandtl: float = PRANDTL
    boundary_x: int = 0
    boundary_y: int = 0

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (r(self.dx), r(self.dy), r(self.c_s), r(self.prandtl), i(self.boundary_x),
                i(self.boundary_y), i(self.nz), i(self.ny), i(self.nx))

    def validate(self) -> None:
        _axes(self)
        if not all(np.isfinite(x) and x > 0 for x in (self.dx, self.dy, self.prandtl)):
            raise ValueError("dx, dy and prandtl must be positive and finite")


def km_shapes(geom) -> dict:
    mass = (geom.nz, geom.ny, geom.nx)
    return {"xkmh": mass, "xkhh": mass, "defor11": mass, "defor22": mass, "defor12": mass,
            "msft": (geom.ny, geom.nx)}


def km_tl(state, pert_state, traj, pert, geom: KMGeometry, *, nl=True):
    """WRFPLUS G_SMAG2D_KM (diff_opt 1) on woof's strict wrf_diff_opt1_km4
    + wrf_smag_km_bc: xkmh and xkhh (overwritten) of the deformation."""
    geom.validate()
    _check(geom, _named("state", KM_STATE, state) + _named("pert_state", KM_STATE, pert_state)
           + _named("traj", KM_ACTIVE + KM_PASSIVE, traj)
           + _named("pert", KM_ACTIVE, pert), km_shapes(geom), min_nz=1)
    arrs = (tuple(state[n] for n in KM_STATE) + tuple(pert_state[n] for n in KM_STATE)
            + tuple(a for n in KM_ACTIVE for a in (traj[n], pert[n])) + (traj["msft"],))
    dtype = _dtype(arrs)
    _kernel("km_tl", dtype)(_blocks(geom.nz * geom.ny * geom.nx), (THREADS,),
                            arrs + (np.int32(bool(nl)),) + geom.args(dtype))


def km_ad(adj_state, traj, adj, geom: KMGeometry):
    """Adjoint of :func:`km_tl`: add into ``adj`` (defor11, defor22,
    defor12) and consume the xkmh and xkhh adjoints."""
    geom.validate()
    _check(geom, _named("adj", KM_STATE, adj_state)
           + _named("traj", KM_ACTIVE + KM_PASSIVE, traj)
           + _named("adj", KM_ACTIVE, adj), km_shapes(geom), min_nz=1)
    arrs = (tuple(adj_state[n] for n in KM_STATE) + tuple(traj[n] for n in KM_ACTIVE)
            + tuple(adj[n] for n in KM_ACTIVE) + (traj["msft"],))
    dtype = _dtype(arrs)
    m = geom.nz * geom.ny * geom.nx
    _kernel("km_ad", dtype)(_blocks(m), (THREADS,), arrs + geom.args(dtype))
    _kernel("km_zero2", dtype)(_blocks(m), (THREADS,),
                               tuple(adj_state[n] for n in KM_STATE) + (np.int32(m),))


# ------------------------------------- horizontal_diffusion (diff_opt 1)

#: The rows of wrf_diff_opt1_horizontal: stagger and whether the field is
#: mixed as a perturbation from a base (horizontal_diffusion_3dmp).
HD_ROWS = {"u": (1, 0), "v": (2, 0), "w": (3, 0), "t": (0, 1), "s": (0, 0)}
HD_STATE = ("tend",)
HD_ACTIVE = ("field", "xk", "mut")
HD_PASSIVE = ("c1", "c2", "msft", "msfu", "msfv")


@dataclass(frozen=True)
class HDGeometry:
    """wrf_diff_opt1_horizontal's scalars: ``rdx = 1/dx`` and ``rdy = 1/dy``
    as the primal rounds them, the row (``u``, ``v``, ``w``; ``t`` theta
    minus its initial field, horizontal_diffusion_3dmp; ``s`` a scalar)
    and the bounded (open or specified) axes."""
    nz: int
    ny: int
    nx: int
    rdx: float
    rdy: float
    row: str = "t"
    boundary_x: int = 0
    boundary_y: int = 0

    def __post_init__(self):
        if self.row not in HD_ROWS:   # before stag/perturb index HD_ROWS
            raise ValueError(f"row is one of {sorted(HD_ROWS)}, got {self.row!r}")

    @property
    def stag(self) -> int:
        return HD_ROWS[self.row][0]

    @property
    def perturb(self) -> int:
        return HD_ROWS[self.row][1]

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return (r(self.rdx), r(self.rdy), i(self.stag), i(self.perturb), i(self.boundary_x),
                i(self.boundary_y), i(self.nz), i(self.ny), i(self.nx))

    def validate(self) -> None:
        _axes(self)   # the row is checked in __post_init__
        if not (np.isfinite(self.rdx) and np.isfinite(self.rdy)
                and self.rdx > 0 and self.rdy > 0):
            raise ValueError("rdx and rdy must be positive and finite")


def hd_passive(geom) -> tuple:
    return HD_PASSIVE + (("base",) if geom.perturb else ())


def hd_shapes(geom) -> dict:
    nz, ny, nx, st = geom.nz, geom.ny, geom.nx, geom.stag
    row = (nz + (st == 3), ny + (st == 2), nx + (st == 1))
    nc = nz + 1 if st == 3 else nz
    return {"tend": row, "field": row, "xk": (nz, ny, nx), "mut": (ny, nx), "c1": (nc,),
            "c2": (nc,), "base": (nz, ny, nx), "msft": (ny, nx), "msfu": (ny, nx + 1),
            "msfv": (ny + 1, nx)}


def _hd_passive_args(traj, geom):
    base = traj["base"] if geom.perturb else traj["field"]
    return (traj["c1"], traj["c2"], base, traj["msft"], traj["msfu"], traj["msfv"])


def hd_tl(state, pert_state, traj, pert, geom: HDGeometry, *, nl=True):
    """WRFPLUS G_HORIZONTAL_DIFFUSION / g_horizontal_diffusion_3dmp on
    woof's strict wrf_diff_opt1_horizontal: ``tend`` (accumulated) of the
    field, the coefficient ``xk`` and ``mut``."""
    geom.validate()
    _check(geom, _named("state", HD_STATE, state) + _named("pert_state", HD_STATE, pert_state)
           + _named("traj", HD_ACTIVE + hd_passive(geom), traj)
           + _named("pert", HD_ACTIVE, pert), hd_shapes(geom), min_nz=1)
    arrs = ((state["tend"], pert_state["tend"])
            + tuple(a for n in HD_ACTIVE for a in (traj[n], pert[n]))
            + _hd_passive_args(traj, geom))
    dtype = _dtype(arrs)
    n = int(np.prod(hd_shapes(geom)["tend"]))
    _kernel("hd_tl", dtype)(_blocks(n), (THREADS,), arrs + (np.int32(bool(nl)),)
                            + geom.args(dtype))


def hd_ad(adj_state, traj, adj, geom: HDGeometry):
    """Adjoint of :func:`hd_tl`: add into ``adj`` (field, xk, mut).  The
    tendency accumulates, so its adjoint is left as it is."""
    geom.validate()
    _check(geom, _named("adj", HD_STATE, adj_state)
           + _named("traj", HD_ACTIVE + hd_passive(geom), traj)
           + _named("adj", HD_ACTIVE, adj), hd_shapes(geom), min_nz=1)
    arrs = ((adj_state["tend"],) + tuple(traj[n] for n in HD_ACTIVE)
            + tuple(adj[n] for n in HD_ACTIVE) + _hd_passive_args(traj, geom))
    dtype = _dtype(arrs)
    sh = hd_shapes(geom)
    n = sum(int(np.prod(sh[k])) for k in HD_ACTIVE)
    _kernel("hd_ad", dtype)(_blocks(n), (THREADS,), arrs + geom.args(dtype))


# --------------------------------- horizontal_diffusion_2 (diff_opt 2)

#: The rows of WRF horizontal_diffusion_2 (woof wrf_smag_hd_u / _v / _w
#: and wrf_smag_flux_s + wrf_smag_hd_s) and of vertical_diffusion_2 (woof
#: wrf_smag_vd_u / _v / _w / _s with the surface kernels of
#: dycore.launch_wrf_smag2d_vertical): the kernel's row code.  Theta is the
#: scalar row on woof's strict thp (= theta - 300, WRF's t_2); ``vt`` and
#: ``vs`` are the vertical scalar row with theta's or qv's surface flux.
H2_ROWS = {"u": 1, "v": 2, "w": 3, "t": 0, "s": 0, "vu": 4, "vv": 5, "vw": 6, "vt": 7, "vs": 7}
H2_STATE = ("tend",)
H2_PASSIVE = ("phb", "msft", "msfu", "msfv", "fnm", "fnp", "dn", "dnw")
#: Each row's active inputs, before ``qv`` (moist density).
_H2_ACTIVE = {1: ("defor11", "defor12", "xk", "alt", "php"),
              2: ("defor22", "defor12", "xk", "alt", "php"),
              3: ("u", "v", "w", "xk", "alt", "php"),
              0: ("field", "xk", "alt", "php"),
              4: ("u", "v", "w", "xk", "alt", "php"),
              5: ("u", "v", "w", "xk", "alt", "php"),
              6: ("w", "xk", "alt", "php"),
              7: ("field", "xk", "alt", "php")}
#: vertical_diffusion_2's surface momentum flux: none, USTM (isfflx 1/2) or
#: the constant drag coefficient (isfflx 0); its heat flux: none, the
#: prescribed tke_heat_flux (isfflx 0/2) or HFX (isfflx 1).
H2_SURFACE = {"none": 0, "ustm": 1, "cd0": 2}
H2_HEAT = {"none": 0, "const": 1, "hfx": 2}


@dataclass(frozen=True)
class H2Geometry:
    """horizontal_diffusion_2's scalars: ``rdx = 1/dx`` and ``rdy = 1/dy``
    as the primal rounds them, the bottom extrapolation weights cf1-cf3,
    the row (``u``, ``v``, ``w``; ``t`` theta and ``s`` a scalar, both the
    scalar operator), the bounded (open or specified) axes and ``moist``
    (the density 1/alt times 1 + qv, wrf_rho's moist branch)."""
    nz: int
    ny: int
    nx: int
    rdx: float
    rdy: float
    cf1: float
    cf2: float
    cf3: float
    row: str = "s"
    boundary_x: int = 0
    boundary_y: int = 0
    moist: int = 0
    #: vertical rows only: the vu / vv surface stress (H2_SURFACE) and cd0;
    #: vt's surface heat flux (H2_HEAT), the prescribed kinematic flux and
    #: full_theta (thp mixed with thb, wrf_smag_vd_s's full_theta); vs's
    #: qflux (the QFX surface moisture flux of the qv row, moist only).
    #: The geometry carries no isfflx or surface-layer flag: the caller
    #: picks ``surface`` / ``heat`` / ``qflux`` as
    #: dycore.launch_wrf_smag2d_vertical's SELECT CASE(isfflx) would (none
    #: where the surface layer supplies no fields).  The surface stress
    #: takes the earth-relative u and v; a moving frame's ground-relative
    #: winds are not modelled (the step binding refuses a moving frame).
    #: The rows are the GPUWM_WRF_EXACT_C_DIFFUSION statements (the
    #: default build; only GPUWM_WRF_EXACT=1 with
    #: GPUWM_WRF_EXACT_DIFFUSION=0 compiles the other forms).
    surface: str = "none"
    cd0: float = 0.0
    heat: str = "none"
    heat_flux: float = 0.0
    full_theta: int = 0
    qflux: int = 0
    #: vt only: 0 is km_opt 4's row, the surface heat flux alone
    #: (smag2d_km leaves xkhv = 0, so launch_wrf_smag2d_vertical runs no
    #: interior scalar row): its active inputs are alt (and qv) alone.
    interior: int = 1
    #: WRF's cold seam slope at a periodic axis's top w level (DFGeometry).
    seam_cold: int = 0

    def __post_init__(self):
        if self.row not in H2_ROWS:
            raise ValueError(f"row is one of {sorted(H2_ROWS)}, got {self.row!r}")
        if self.surface not in H2_SURFACE:
            raise ValueError(f"surface is one of {sorted(H2_SURFACE)}, got {self.surface!r}")
        if self.heat not in H2_HEAT:
            raise ValueError(f"heat is one of {sorted(H2_HEAT)}, got {self.heat!r}")
        if self.surface != "none" and self.row not in ("vu", "vv"):
            raise ValueError("a surface stress belongs to the vu and vv rows")
        if (self.heat != "none" or self.full_theta) and self.row != "vt":
            raise ValueError("a surface heat flux and full_theta belong to the vt row")
        if self.qflux and (self.row != "vs" or not self.moist):
            raise ValueError("qflux belongs to the vs row of a moist run (the qv row)")
        if self.full_theta not in (0, 1) or self.qflux not in (0, 1):
            raise ValueError("full_theta and qflux are 0 or 1")
        if self.interior not in (0, 1) or self.seam_cold not in (0, 1):
            raise ValueError("interior and seam_cold are 0 or 1")
        if not self.interior and (self.row != "vt" or self.full_theta):
            raise ValueError("a surface-only row (interior 0) is vt's, without full_theta")

    @property
    def vertical(self) -> bool:
        return self.code >= 4

    @property
    def code(self) -> int:
        return H2_ROWS[self.row]

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        bx, by = _flags(self)
        return (r(self.rdx), r(self.rdy), r(self.cf1), r(self.cf2), r(self.cf3),
                i(bx), i(by), i(self.nz), i(self.ny), i(self.nx), i(self.code), i(self.moist))

    def validate(self) -> None:
        _axes(self)
        if self.nz < 3:
            raise ValueError(f"diff_opt 2 rows need nz >= 3 (the scalar faces' and "
                             f"defor13/23's bottom extrapolation read levels 0-2; "
                             f"vw, vt and vs alone would take 2, held at 3 with "
                             f"the rest), got {self.nz}")
        if self.moist not in (0, 1):
            raise ValueError(f"moist is 0 or 1, got {self.moist}")
        if not (np.isfinite(self.rdx) and np.isfinite(self.rdy)
                and self.rdx > 0 and self.rdy > 0):
            raise ValueError("rdx and rdy must be positive and finite")


def h2_active(geom) -> tuple:
    act = _H2_ACTIVE[geom.code] if geom.interior else ("alt", "php")
    return act + (("qv",) if geom.moist else ())


def h2_passive(geom) -> tuple:
    """H2_PASSIVE and the vertical rows' surface fields: ``ustm`` (vu / vv
    with the USTM stress), ``hfx`` (vt with HFX), ``thb`` (vt with
    full_theta, the 3-D base theta) and ``qfx`` (vs with qflux)."""
    extra = ((("ustm",) if geom.surface == "ustm" else ())
             + (("hfx",) if geom.heat == "hfx" else ())
             + (("thb",) if geom.full_theta else ())
             + (("qfx",) if geom.qflux else ()))
    return H2_PASSIVE + extra


def h2_shapes(geom) -> dict:
    nz, ny, nx, c = geom.nz, geom.ny, geom.nx, geom.code
    mass, wlev = (nz, ny, nx), (nz + 1, ny, nx)
    tend = {1: (nz, ny, nx + 1), 2: (nz, ny + 1, nx), 3: wlev, 0: mass,
            4: (nz, ny, nx + 1), 5: (nz, ny + 1, nx), 6: wlev, 7: mass}[c]
    return {"tend": tend, "defor11": mass, "defor22": mass, "defor12": mass, "xk": mass,
            "alt": mass, "qv": mass, "field": mass, "u": (nz, ny, nx + 1),
            "v": (nz, ny + 1, nx), "w": wlev, "php": wlev, "phb": wlev, "msft": (ny, nx),
            "msfu": (ny, nx + 1), "msfv": (ny + 1, nx), "fnm": (nz,), "fnp": (nz,),
            "dn": (nz,), "dnw": (nz,), "ustm": (ny, nx), "hfx": (ny, nx), "qfx": (ny, nx),
            "thb": mass}


def _h2_temp_shapes(geom) -> dict:
    """The stored intermediates (kernels/diffusion.cu, diff_opt 2)."""
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    sh = {"zx": (nz + 1, ny, nx + 1), "zy": (nz + 1, ny + 1, nx), "rdzw": (nz, ny, nx),
          "rdz": (nz + 1, ny, nx), "rho": (nz, ny, nx)}
    sh.update({1: {"s1": (nz, ny, nx), "s2": (nz, ny + 1, nx + 1)},
               2: {"s1": (nz, ny, nx), "s2": (nz, ny + 1, nx + 1)},
               3: {"s1": (nz + 1, ny, nx + 1), "s2": (nz + 1, ny + 1, nx)},
               4: {"s1": (nz + 1, ny, nx + 1), "s2": (nz + 1, ny + 1, nx)},
               5: {"s1": (nz + 1, ny, nx + 1), "s2": (nz + 1, ny + 1, nx)},
               0: {"s1": (nz, ny, nx + 1), "s2": (nz, ny + 1, nx)},
               6: {}, 7: {}}[geom.code])
    return sh


def _h2_temps(geom, dtype, fill=None) -> dict:
    import cupy as cp
    make = cp.empty if fill is None else cp.zeros
    return {n: make(s, dtype=dtype) for n, s in _h2_temp_shapes(geom).items()}


def _h2_check(geom, named) -> None:
    geom.validate()
    _check(geom, named, h2_shapes(geom), min_nz=3)


def _h2_launch(name, dtype, n, args, geom):
    _kernel(name, dtype)(_blocks(n), (THREADS,), args + geom.args(dtype))


def _h2_sizes(geom) -> dict:
    return {n: int(np.prod(s)) for n, s in _h2_temp_shapes(geom).items()}


def _h2_forward(traj, pert, geom, dtype, tl):
    """The staged forward: the intermediates (and with ``tl`` their
    tangents).  Returns (values, tangents); with ``tl`` 0 the tangents are
    one-word placeholders the kernels never touch."""
    import cupy as cp
    T = _h2_temps(geom, dtype)
    D = _h2_temps(geom, dtype) if tl else {n: cp.zeros(1, dtype=dtype) for n in T}
    P = pert if tl else traj
    sz, c, flag = _h2_sizes(geom), geom.code, np.int32(tl)
    _h2_launch("h2_met_tl", dtype, sz["zx"] + sz["zy"] + sz["rdzw"] + sz["rdz"],
               (traj["php"], P["php"], traj["phb"], T["zx"], D["zx"], T["zy"], D["zy"],
                T["rdzw"], D["rdzw"], T["rdz"], D["rdz"], flag), geom)
    qv = traj["qv"] if geom.moist else traj["alt"]
    qvd = P["qv"] if geom.moist else P["alt"]
    _h2_launch("h2_rho_tl", dtype, sz["rho"],
               (traj["alt"], P["alt"], qv, qvd, T["rho"], D["rho"], flag), geom)
    if c in (1, 2):
        da = "defor11" if c == 1 else "defor22"
        _h2_launch("h2_tuv_tl", dtype, sz["s1"] + sz["s2"],
                   (T["rho"], D["rho"], traj["xk"], P["xk"], traj[da], P[da],
                    traj["defor12"], P["defor12"], T["s1"], D["s1"], T["s2"], D["s2"], flag),
                   geom)
    elif c in (3, 4, 5):
        _h2_launch("h2_tw_tl", dtype, sz["s1"] + sz["s2"],
                   (traj["u"], P["u"], traj["v"], P["v"], traj["w"], P["w"], T["zx"], D["zx"],
                    T["zy"], D["zy"], T["rdz"], D["rdz"], T["rho"], D["rho"], traj["xk"],
                    P["xk"], traj["msft"], traj["msfu"], traj["msfv"], traj["fnm"],
                    traj["fnp"], T["s1"], D["s1"], T["s2"], D["s2"], flag), geom)
    elif c == 0:
        _h2_launch("h2_hs_tl", dtype, sz["s1"] + sz["s2"],
                   (traj["field"], P["field"], traj["xk"], P["xk"], T["rho"], D["rho"],
                    T["zx"], D["zx"], T["zy"], D["zy"], T["rdzw"], D["rdzw"], traj["msfu"],
                    traj["msfv"], traj["fnm"], traj["fnp"], traj["dn"], traj["dnw"],
                    T["s1"], D["s1"], T["s2"], D["s2"], flag), geom)
    return T, D


def _h2_rz(geom) -> str:
    return "rdz" if geom.code == 3 else "rdzw"


def _h2_row_passive(traj) -> tuple:
    return tuple(traj[n] for n in ("msft", "msfu", "msfv", "fnm", "fnp", "dn", "dnw"))


def h2_tl(state, pert_state, traj, pert, geom: H2Geometry, *, nl=True):
    """WRF horizontal_diffusion_2's row (horizontal_diffusion_u_2, _v_2,
    _w_2 or _s) on woof's strict wrf_smag_hd_u / _v / _w or wrf_smag_flux_s
    + wrf_smag_hd_s: ``tend`` (accumulated) of the row's active inputs
    (:func:`h2_active`)."""
    act, pas = h2_active(geom), h2_passive(geom)
    traj = _h2_traj3(traj, geom)
    _h2_check(geom, _named("state", H2_STATE, state) + _named("pert_state", H2_STATE, pert_state)
              + _named("traj", act + pas, traj) + _named("pert", act, pert))
    dtype = _dtype((state["tend"], pert_state["tend"]) + tuple(traj[n] for n in act + pas)
                   + tuple(pert[n] for n in act))
    T, D = _h2_forward(traj, pert, geom, dtype, 1)
    if geom.vertical:
        _h2_vrow_tl(state, pert_state, traj, pert, T, D, geom, dtype, nl)
        return
    rz = _h2_rz(geom)
    _h2_launch("h2_row_tl", dtype, int(np.prod(h2_shapes(geom)["tend"])),
               (state["tend"], pert_state["tend"], T["s1"], D["s1"], T["s2"], D["s2"],
                T["zx"], D["zx"], T["zy"], D["zy"], T[rz], D[rz]) + _h2_row_passive(traj)
               + (np.int32(bool(nl)),), geom)


def h2_ad(adj_state, traj, adj, geom: H2Geometry):
    """Adjoint of :func:`h2_tl`: add into ``adj`` (the row's active
    inputs).  The tendency accumulates, so its adjoint is left as it is.
    Runs the base forward into the intermediates, then each stage's
    adjoint in reverse order."""
    act, pas = h2_active(geom), h2_passive(geom)
    traj = _h2_traj3(traj, geom)
    _h2_check(geom, _named("adj", H2_STATE, adj_state) + _named("traj", act + pas, traj)
              + _named("adj", act, adj))
    dtype = _dtype((adj_state["tend"],) + tuple(traj[n] for n in act + pas)
                   + tuple(adj[n] for n in act))
    T, _ = _h2_forward(traj, None, geom, dtype, 0)
    A = _h2_temps(geom, dtype, fill=0)
    sz, c, rz = _h2_sizes(geom), geom.code, _h2_rz(geom)
    if geom.vertical:
        _h2_vrow_ad(adj_state, traj, adj, T, A, geom, dtype)
    else:
        _h2_launch("h2_row_ad", dtype, sz["s1"] + sz["s2"] + sz["zx"] + sz["zy"] + sz[rz],
                   (adj_state["tend"], T["s1"], T["s2"], T["zx"], T["zy"], T[rz])
                   + _h2_row_passive(traj) + (A["s1"], A["s2"], A["zx"], A["zy"], A[rz]),
                   geom)
    m = sz["rho"]
    if c in (1, 2):
        da = "defor11" if c == 1 else "defor22"
        _h2_launch("h2_tuv_ad", dtype, 4 * m,
                   (A["s1"], A["s2"], T["rho"], traj["xk"], traj[da], traj["defor12"],
                    A["rho"], adj["xk"], adj[da], adj["defor12"]), geom)
    elif c in (3, 4, 5):
        sh = h2_shapes(geom)
        n = (sum(int(np.prod(sh[k])) for k in ("u", "v", "w")) + sz["zx"] + sz["zy"]
             + sz["rdz"] + 2 * m)
        _h2_launch("h2_tw_ad", dtype, n,
                   (A["s1"], A["s2"], traj["u"], traj["v"], traj["w"], T["zx"], T["zy"],
                    T["rdz"], T["rho"], traj["xk"], traj["msft"], traj["msfu"], traj["msfv"],
                    traj["fnm"], traj["fnp"], adj["u"], adj["v"], adj["w"], A["zx"], A["zy"],
                    A["rdz"], A["rho"], adj["xk"]), geom)
    elif c == 0:
        _h2_launch("h2_hs_ad", dtype, 3 * m + sz["zx"] + sz["zy"] + sz["rdzw"],
                   (A["s1"], A["s2"], traj["field"], traj["xk"], T["rho"], T["zx"], T["zy"],
                    T["rdzw"], traj["msfu"], traj["msfv"], traj["fnm"], traj["fnp"],
                    traj["dn"], traj["dnw"], adj["field"], adj["xk"], A["rho"], A["zx"],
                    A["zy"], A["rdzw"]), geom)
    import cupy as cp
    qv = traj["qv"] if geom.moist else traj["alt"]
    aqv = adj["qv"] if geom.moist else cp.zeros(1, dtype=dtype)
    _h2_launch("h2_rho_ad", dtype, m * (2 if geom.moist else 1),
               (A["rho"], traj["alt"], qv, adj["alt"], aqv), geom)
    _h2_launch("h2_met_ad", dtype, int(np.prod(h2_shapes(geom)["php"])),
               (A["zx"], A["zy"], A["rdzw"], A["rdz"], traj["php"], traj["phb"], adj["php"]),
               geom)


def _h2_thb3(thb, geom):
    """The base theta as the 3-D mass field the kernel reads; a 1-D
    column (wrf_smag_vd_s's thb3d = 0 branch, thb[k]) is broadcast."""
    import cupy as cp
    if thb.ndim == 1:
        if thb.shape != (geom.nz,):
            raise ValueError(f"a 1-D thb has nz = {geom.nz} levels, got {thb.shape}")
        thb = cp.ascontiguousarray(cp.broadcast_to(thb[:, None, None],
                                                   (geom.nz, geom.ny, geom.nx)))
    return thb


def _h2_traj3(traj, geom) -> dict:
    """``traj`` with a 1-D base theta broadcast to 3-D (:func:`_h2_thb3`)."""
    thb = traj.get("thb") if geom.full_theta else None
    if thb is None or getattr(thb, "ndim", 3) != 1:
        return traj
    return dict(traj, thb=_h2_thb3(thb, geom))


def _h2_vargs(traj, geom, dtype) -> tuple:
    """The vertical rows' passive arguments (H2_V_PASSIVE)."""
    import cupy as cp
    r, i = np.dtype(dtype).type, np.int32
    dummy = cp.zeros(1, dtype=dtype)
    return (traj["fnm"], traj["fnp"], traj["dn"], traj["dnw"],
            traj["ustm"] if geom.surface == "ustm" else dummy,
            traj["hfx"] if geom.heat == "hfx" else dummy,
            traj["qfx"] if geom.qflux else dummy,
            traj["thb"] if geom.full_theta else dummy,
            r(geom.cd0), r(geom.heat_flux), i(H2_SURFACE[geom.surface]), i(H2_HEAT[geom.heat]),
            i(geom.full_theta), i(geom.qflux), i(geom.interior))


def _h2_vsel(geom, X, fallback) -> tuple:
    """(stress, u, v, w, field, rho, K, rdzw-or-rdz, qv) of the vertical
    row from the dict ``X`` (values, tangents or adjoints); ``fallback``
    stands in for the inputs the row does not read."""
    c = geom.code
    st = X["s1"] if c == 4 else X["s2"] if c == 5 else fallback("st")
    rz = X["rdzw"] if c == 6 else X["rdz"] if c == 7 else fallback("rz")
    return (st, X["u"] if c in (4, 5) else fallback("u"),
            X["v"] if c in (4, 5) else fallback("v"),
            X["w"] if c in (4, 5, 6) else fallback("w"),
            X["field"] if (c == 7 and geom.interior) else fallback("field"), X["rho"],
            X["xk"] if geom.interior else fallback("xk"), rz,
            X["qv"] if geom.moist else fallback("qv"))


def _h2_vrow_tl(state, pert_state, traj, pert, T, D, geom, dtype, nl):
    import cupy as cp
    dummy = cp.zeros(1, dtype=dtype)
    val = _h2_vsel(geom, {**traj, **T}, lambda n: dummy)
    tan = _h2_vsel(geom, {**pert, **D}, lambda n: dummy)
    args = (state["tend"], pert_state["tend"])
    for a, b in zip(val, tan):
        args += (a, b)
    _h2_launch("h2_vrow_tl", dtype, int(np.prod(h2_shapes(geom)["tend"])),
               args + _h2_vargs(traj, geom, dtype) + (np.int32(bool(nl)),), geom)


def _h2_vrow_count(geom) -> int:
    """Words of the vrow stage's inputs (kernels h2_inputs, stage vrow)."""
    sh, c = h2_shapes(geom), geom.code
    n = lambda k: int(np.prod(sh[k]))  # noqa: E731
    m = n("xk")
    if c in (4, 5):
        st = int(np.prod(_h2_temp_shapes(geom)["s1" if c == 4 else "s2"]))
        return st + ((n("u") + n("v") + m) if geom.surface != "none" else 0)
    if c == 6:
        return n("w") + 3 * m
    q = m if geom.moist and geom.heat == "hfx" else 0
    if not geom.interior:
        return m + q
    return 3 * m + n("php") + q


def _h2_vrow_ad(adj_state, traj, adj, T, A, geom, dtype):
    import cupy as cp
    dummy = cp.zeros(1, dtype=dtype)
    val = _h2_vsel(geom, {**traj, **T}, lambda n: dummy)
    # the adjoint targets: the row's temps, else the active inputs; one
    # distinct scratch word per input the row does not read
    out = _h2_vsel(geom, {**adj, **A}, lambda n: cp.zeros(1, dtype=dtype))
    _h2_launch("h2_vrow_ad", dtype, _h2_vrow_count(geom),
               (adj_state["tend"],) + val + _h2_vargs(traj, geom, dtype) + out, geom)


# ------------------------------------- the diff_opt 2 coefficients

#: smag2d_km at diff_opt 2 (km_opt 4, the slope limiter): woof
#: wrf_smag2d_km then wrf_smag_km_bc.
K2_STATE = ("xkmh", "xkhh")
K2_ACTIVE = ("defor11", "defor22", "defor12", "php")
K2_PASSIVE = ("phb", "msft")
#: calculate_N2, dry: woof wrf_calc_n2 on the time-t thp (theta - thb).
N2_STATE = ("bn2",)
N2_ACTIVE = ("thp", "php")
N2_PASSIVE = ("thb", "phb")
#: smag_km (km_opt 3): woof wrf_smag3d_km then wrf_smag_km_bc on both pairs.
K3_STATE = ("xkmh", "xkhh", "xkmv", "xkhv")
K3_ACTIVE = ("defor11", "defor22", "defor12", "u", "v", "w", "php", "bn2")
K3_PASSIVE = ("phb", "msft", "msfu", "msfv")


@dataclass(frozen=True)
class KCGeometry:
    """The diff_opt 2 coefficients' scalars as the primal rounds them:
    ``rdx = 1/dx``, ``rdy = 1/dy``, the spacings, c_s and the Prandtl
    number; km_opt 3's ``dt``, ``mix_upper_bound`` and ``isotropic``
    (mix_isotropic); the bounded axes (whose outer row keeps zero) and
    WRF's cold seam slope (DFGeometry.seam_cold).  The statements are the
    GPUWM_WRF_EXACT_C_DIFFUSION ones (the default build)."""
    nz: int
    ny: int
    nx: int
    rdx: float
    rdy: float
    dx: float
    dy: float
    c_s: float = 0.25
    prandtl: float = PRANDTL
    dt: float = 1.0
    mix_upper_bound: float = 0.1
    isotropic: int = 0
    boundary_x: int = 0
    boundary_y: int = 0
    seam_cold: int = 0

    def h2(self) -> H2Geometry:
        """The geometry of the metric stage (zx, zy, rdzw and rdz of ph')."""
        return H2Geometry(self.nz, self.ny, self.nx, self.rdx, self.rdy, 0.0, 0.0, 0.0,
                          boundary_x=self.boundary_x, boundary_y=self.boundary_y,
                          seam_cold=self.seam_cold)

    def args(self, dtype) -> tuple:
        i, r = np.int32, np.dtype(dtype).type
        return self.h2().args(dtype) + (r(self.dx), r(self.dy), r(self.c_s), r(self.prandtl),
                                        r(self.dt), r(self.mix_upper_bound),
                                        i(self.isotropic))

    def validate(self) -> None:
        _axes(self)
        if self.nz < 3:
            raise ValueError(f"the diff_opt 2 coefficients need nz >= 3 (calculate_N2's "
                             f"centred levels and defor13/23), got {self.nz}")
        if self.seam_cold not in (0, 1) or self.isotropic not in (0, 1):
            raise ValueError("seam_cold and isotropic are 0 or 1")
        if not all(np.isfinite(x) and x > 0 for x in (self.rdx, self.rdy, self.dx, self.dy,
                                                     self.prandtl, self.dt)):
            raise ValueError("rdx, rdy, dx, dy, prandtl and dt must be positive and finite")


def kc_shapes(geom) -> dict:
    nz, ny, nx = geom.nz, geom.ny, geom.nx
    mass, wlev = (nz, ny, nx), (nz + 1, ny, nx)
    return {**{n: mass for n in K3_STATE + N2_STATE + DF_STATE + ("thp", "thb")},
            "u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx), "w": wlev, "php": wlev,
            "phb": wlev, "msft": (ny, nx), "msfu": (ny, nx + 1), "msfv": (ny + 1, nx)}


def _kc_check(geom, named) -> None:
    geom.validate()
    _check(geom, named, kc_shapes(geom), min_nz=3)


def _kc_met(traj, pert, geom, dtype, tl):
    """Stage met of ph' (:func:`_h2_forward`'s first launch): the stored
    zx, zy, rdzw and rdz (and with ``tl`` their tangents)."""
    import cupy as cp
    g = geom.h2()
    sh = _h2_temp_shapes(g)
    names = ("zx", "zy", "rdzw", "rdz")
    T = {n: cp.empty(sh[n], dtype=dtype) for n in names}
    D = ({n: cp.empty(sh[n], dtype=dtype) for n in names} if tl
         else {n: cp.zeros(1, dtype=dtype) for n in names})
    P = pert if tl else traj
    n = sum(int(np.prod(sh[k])) for k in names)
    _h2_launch("h2_met_tl", dtype, n,
               (traj["php"], P["php"], traj["phb"], T["zx"], D["zx"], T["zy"], D["zy"],
                T["rdzw"], D["rdzw"], T["rdz"], D["rdz"], np.int32(tl)), g)
    return T, D


def _kc_met_ad(traj, adj, A, geom):
    _h2_launch("h2_met_ad", _dtype((A["zx"],)), int(np.prod(kc_shapes(geom)["php"])),
               (A["zx"], A["zy"], A["rdzw"], A["rdz"], traj["php"], traj["phb"], adj["php"]),
               geom.h2())


def _kc_zeros(geom, dtype) -> dict:
    import cupy as cp
    sh = _h2_temp_shapes(geom.h2())
    return {n: cp.zeros(sh[n], dtype=dtype) for n in ("zx", "zy", "rdzw", "rdz")}


def _kc_consume(adj_state, names, geom):
    m = geom.nz * geom.ny * geom.nx
    a = [adj_state[n] for n in names]
    a += [a[0]] * (4 - len(a))
    _kernel("kc_zero4", _dtype(a[:1]))(_blocks(m), (THREADS,),
                                       tuple(a) + (np.int32(m), np.int32(len(names))))


def k2_tl(state, pert_state, traj, pert, geom: KCGeometry, *, nl=True):
    """WRFPLUS G_SMAG2D_KM at diff_opt 2 on woof's strict wrf_smag2d_km +
    wrf_smag_km_bc: xkmh and xkhh (overwritten) of the deformation and,
    through the slope limiter's zx, zy and rdzw, php."""
    _kc_check(geom, _named("state", K2_STATE, state)
              + _named("pert_state", K2_STATE, pert_state)
              + _named("traj", K2_ACTIVE + K2_PASSIVE, traj) + _named("pert", K2_ACTIVE, pert))
    dtype = _dtype(tuple(state[n] for n in K2_STATE) + tuple(traj[n] for n in K2_ACTIVE)
                   + tuple(pert[n] for n in K2_ACTIVE))
    T, D = _kc_met(traj, pert, geom, dtype, 1)
    _kernel("h2_km2_tl", dtype)(
        _blocks(geom.nz * geom.ny * geom.nx), (THREADS,),
        tuple(a for n in DF_STATE for a in (traj[n], pert[n]))
        + (T["zx"], D["zx"], T["zy"], D["zy"], T["rdzw"], D["rdzw"], traj["msft"],
           state["xkmh"], pert_state["xkmh"], state["xkhh"], pert_state["xkhh"],
           np.int32(bool(nl))) + geom.args(dtype))


def k2_ad(adj_state, traj, adj, geom: KCGeometry):
    """Adjoint of :func:`k2_tl`: add into ``adj`` (defor11, defor22,
    defor12, php) and consume the xkmh and xkhh adjoints."""
    _kc_check(geom, _named("adj", K2_STATE, adj_state)
              + _named("traj", K2_ACTIVE + K2_PASSIVE, traj) + _named("adj", K2_ACTIVE, adj))
    dtype = _dtype(tuple(adj_state[n] for n in K2_STATE) + tuple(traj[n] for n in K2_ACTIVE)
                   + tuple(adj[n] for n in K2_ACTIVE))
    T, _ = _kc_met(traj, None, geom, dtype, 0)
    A = _kc_zeros(geom, dtype)
    m = geom.nz * geom.ny * geom.nx
    n = 4 * m + int(np.prod(T["zx"].shape)) + int(np.prod(T["zy"].shape))
    _kernel("h2_km2_ad", dtype)(
        _blocks(n), (THREADS,),
        (adj_state["xkmh"], adj_state["xkhh"]) + tuple(traj[k] for k in DF_STATE)
        + (T["zx"], T["zy"], T["rdzw"], traj["msft"]) + tuple(adj[k] for k in DF_STATE)
        + (A["zx"], A["zy"], A["rdzw"]) + geom.args(dtype))
    _kc_met_ad(traj, adj, A, geom)
    _kc_consume(adj_state, K2_STATE, geom)


def _n2_traj(traj, geom) -> dict:
    """``traj`` with a 1-D base theta broadcast to 3-D."""
    thb = traj.get("thb")
    if thb is None or getattr(thb, "ndim", 3) != 1:
        return traj
    return dict(traj, thb=_h2_thb3(thb, geom))


def n2_tl(state, pert_state, traj, pert, geom: KCGeometry, *, nl=True):
    """WRFPLUS g_calculate_N2, dry, on woof's strict wrf_calc_n2: bn2
    (overwritten) of thp and, through rdz and rdzw, php."""
    traj = _n2_traj(traj, geom)
    _kc_check(geom, _named("state", N2_STATE, state)
              + _named("pert_state", N2_STATE, pert_state)
              + _named("traj", N2_ACTIVE + N2_PASSIVE, traj) + _named("pert", N2_ACTIVE, pert))
    dtype = _dtype((state["bn2"],) + tuple(traj[n] for n in N2_ACTIVE)
                   + tuple(pert[n] for n in N2_ACTIVE))
    T, D = _kc_met(traj, pert, geom, dtype, 1)
    _kernel("h2_n2_tl", dtype)(
        _blocks(geom.nz * geom.ny * geom.nx), (THREADS,),
        (traj["thp"], pert["thp"], traj["thb"], T["rdz"], D["rdz"], T["rdzw"], D["rdzw"],
         state["bn2"], pert_state["bn2"], np.int32(bool(nl))) + geom.h2().args(dtype))


def n2_ad(adj_state, traj, adj, geom: KCGeometry):
    """Adjoint of :func:`n2_tl`: add into ``adj`` (thp, php) and consume
    the bn2 adjoint."""
    traj = _n2_traj(traj, geom)
    _kc_check(geom, _named("adj", N2_STATE, adj_state)
              + _named("traj", N2_ACTIVE + N2_PASSIVE, traj) + _named("adj", N2_ACTIVE, adj))
    dtype = _dtype((adj_state["bn2"],) + tuple(traj[n] for n in N2_ACTIVE)
                   + tuple(adj[n] for n in N2_ACTIVE))
    T, _ = _kc_met(traj, None, geom, dtype, 0)
    A = _kc_zeros(geom, dtype)
    m = geom.nz * geom.ny * geom.nx
    n = 2 * m + int(np.prod(T["rdz"].shape))
    _kernel("h2_n2_ad", dtype)(
        _blocks(n), (THREADS,),
        (adj_state["bn2"], traj["thp"], traj["thb"], T["rdz"], T["rdzw"], adj["thp"],
         A["rdz"], A["rdzw"]) + geom.h2().args(dtype))
    _kc_met_ad(traj, adj, A, geom)
    _kc_consume(adj_state, N2_STATE, geom)


def k3_tl(state, pert_state, traj, pert, geom: KCGeometry, *, nl=True):
    """WRFPLUS G_SMAG_KM (km_opt 3) on woof's strict wrf_smag3d_km +
    wrf_smag_km_bc: xkmh, xkhh, xkmv and xkhv (overwritten) of the
    deformation, u, v, w (defor13, defor23, defor33), php and bn2."""
    _kc_check(geom, _named("state", K3_STATE, state)
              + _named("pert_state", K3_STATE, pert_state)
              + _named("traj", K3_ACTIVE + K3_PASSIVE, traj) + _named("pert", K3_ACTIVE, pert))
    dtype = _dtype(tuple(state[n] for n in K3_STATE) + tuple(traj[n] for n in K3_ACTIVE)
                   + tuple(pert[n] for n in K3_ACTIVE))
    T, D = _kc_met(traj, pert, geom, dtype, 1)
    act = tuple(a for n in DF_STATE + ("u", "v", "w") for a in (traj[n], pert[n]))
    met = tuple(a for n in ("zx", "zy", "rdz", "rdzw") for a in (T[n], D[n]))
    out = tuple(a for n in K3_STATE for a in (state[n], pert_state[n]))
    _kernel("h2_km3_tl", dtype)(
        _blocks(geom.nz * geom.ny * geom.nx), (THREADS,),
        act + met + (traj["bn2"], pert["bn2"], traj["msft"], traj["msfu"], traj["msfv"])
        + out + (np.int32(bool(nl)),) + geom.args(dtype))


def k3_ad(adj_state, traj, adj, geom: KCGeometry):
    """Adjoint of :func:`k3_tl`: add into ``adj`` (:data:`K3_ACTIVE`) and
    consume the four coefficients' adjoints."""
    _kc_check(geom, _named("adj", K3_STATE, adj_state)
              + _named("traj", K3_ACTIVE + K3_PASSIVE, traj) + _named("adj", K3_ACTIVE, adj))
    dtype = _dtype(tuple(adj_state[n] for n in K3_STATE) + tuple(traj[n] for n in K3_ACTIVE)
                   + tuple(adj[n] for n in K3_ACTIVE))
    T, _ = _kc_met(traj, None, geom, dtype, 0)
    A = _kc_zeros(geom, dtype)
    sh = kc_shapes(geom)
    size = {k: int(np.prod(v.shape)) for k, v in T.items()}
    n = (5 * int(np.prod(sh["bn2"])) + sum(int(np.prod(sh[k])) for k in ("u", "v", "w"))
         + size["zx"] + size["zy"] + size["rdz"])
    _kernel("h2_km3_ad", dtype)(
        _blocks(n), (THREADS,),
        tuple(adj_state[k] for k in K3_STATE)
        + tuple(traj[k] for k in DF_STATE + ("u", "v", "w"))
        + (T["zx"], T["zy"], T["rdz"], T["rdzw"], traj["bn2"], traj["msft"], traj["msfu"],
           traj["msfv"])
        + tuple(adj[k] for k in DF_STATE + ("u", "v", "w"))
        + (A["zx"], A["zy"], A["rdz"], A["rdzw"], adj["bn2"]) + geom.args(dtype))
    _kc_met_ad(traj, adj, A, geom)
    _kc_consume(adj_state, K3_STATE, geom)
