"""WRF-Solar EPS: ``multi_perturb = 1`` and its per-scheme perturbations.

WRF v4.7.1 (f52c197ed) perturbs inputs of four WRF-Solar schemes with
spatially and temporally correlated random fields (Registry/registry.stoch
:92-130, :174-188; Kim et al. 2022, the WRF-Solar EPS):

* the fields: one per row of run/STOCHPERT.TBL, read by INITIALIZE_STOCH
  (dyn_em/module_stoch.F:273-446) -- ALBEDO, AOD, ANGSTROM, ASSYMFAC,
  QVAPOR, QCLOUD, QICE, QSNOW, NI, TH, TKE, SMOIS, TSOIL, W -- each a
  WRF random-perturbation pattern (SETUP_RAND_PERTURB 'R', RAND_PERT_UPDATE
  'T') with its own standard deviation, length and time scale, cutoff,
  seed and vertical structure (0 = the same at every level, 1 = random
  phase).  They are WRF's ``pert3d`` 4-D array, updated on domain 1 only,
  on the first step and every ``stepsp = max(nint(spdt*60/dt), 1)`` steps
  (first_rk_step_part2.F:262-317) -- after the first-RK physics, before
  the microphysics.  Nested domains keep the zero WRF allocated.
* the hooks, each scheme's Add_/Remove_ pair around its call (or, for
  icloud = 3, the inputs of cal_cldfra3), active with ``multi_perturb = 1``
  and the scheme's ``pert_<scheme>`` switch; a ``pert_<scheme>_<var>``
  factor scales its field (0 leaves the input as it is, up to the clamps):

  ===========  ==============================================  ===========================
  switch       perturbed (factor x field)                       WRF
  ===========  ==============================================  ===========================
  pert_farms   albedo, AOD, Angstrom exponent, asymmetry        module_radiation_driver.F
               (lowest pattern level); qv, qc, qs               :3052-3082, :5623-5708
  pert_deng    qv, qc, theta; w above the surface (with         module_shallowcu_driver.F
               ``pert_deng_t`` on the way in and                :632-672, :695-765
               ``pert_deng_w`` on the way out -- WRF's own)
  pert_thom    qv, qc, qi, qs, ni (mp_physics 8 and 28)        module_microphysics_driver.F
                                                                :1120-1213, :1314-1381, :2937-3008
  pert_cld3    cal_cldfra3's t and qv (icloud = 3)              module_radiation_driver.F :1472-1474
  ===========  ==============================================  ===========================

  The "Remove" is not an inverse: FARMS and Thompson subtract
  ``pert*factor*saved`` from the post-scheme value, FARMS' 2-D fields and
  Deng divide, and each result is clamped (qv >= 0, Thompson's 1e-12 and
  1e-6 floors, albedo and asymmetry inside [0, 1] and [-1, 1]).  What the
  round trip leaves behind stays in the model state, as it does in WRF.

Words: every hook statement is the kernel ``eps_apply`` in
woof/core/kernels/wrfsolar_eps.cu, held word for word to the drivers'
routines compiled from WRF v4.7.1 (tools/wrfsolar_eps_wrf471_oracle,
tests/test_wrfsolar_eps_wrf471_oracle.py).  The pattern arithmetic --
SP_AMP and ALPH, the AR(1) update, the vertical structure and the clip --
is WRF's word for word given the same Gaussian draws; the draws are
counter-based Philox, not the compiler's RANDOM_NUMBER (woof's stochastic
policy, woof/ensemble/stochastic.py), and the backward FFT is cuFFT or
pocketfft rather than FFTPACK5 (graded to a tolerance).

WRF-Solar EPS also perturbs MYNN (``pert_mynn``: qv, qc, theta, qke) and
Noah (``pert_noah``: qv, theta, soil moisture and temperature).  Those two
are not ported and are refused by name (woof/namelist_stochastic.py).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

F = np.float32

#: STOCHPERT.TBL's field names in WRF's pert3d order (P_PALBEDO = 2 ..
#: P_PW = 15, the order of registry.stoch's ``multi_perturb`` package).
FIELDS: tuple[str, ...] = ("ALBEDO", "AOD", "ANGSTROM", "ASSYMFAC", "QVAPOR",
                           "QCLOUD", "QICE", "QSNOW", "NI", "TH", "TKE",
                           "SMOIS", "TSOIL", "W")

#: WRF v4.7.1 run/STOCHPERT.TBL, verbatim (sha256 below).
WRF_STOCHPERT_TBL = """\
  Change only the magnitude of the values
   Name     Std   Lengthscale  Decorrelation Std cutoff  Seed  2d pert (0)
                  [m]          time [s]                        or 3d pert (1)
1  ALBEDO   0.10  100000.0     86400.0       3.0         17    0
2  AOD      0.25  100000.0     3600.0        3.0         18    0
3  ANGSTROM 0.10  100000.0     3600.0        3.0         19    0
4  ASSYMFAC 0.05  100000.0     3600.0        3.0         20    0
5  QVAPOR   0.05  100000.0     3600.0        3.0         21    1
6  QCLOUD   0.10  100000.0     3600.0        3.0         22    1
7  QICE     0.10  100000.0     3600.0        3.0         23    1
8  QSNOW    0.10  100000.0     3600.0        3.0         24    1
9  NI       0.05  100000.0     3600.0        3.0         25    1
10 TH       0.001 100000.0     3600.0        3.0         26    1
11 TKE      0.05   80000.0      600.0        3.0         27    1
12 SMOIS    0.10   80000.0    21600.0        3.0         28    1
13 TSOIL    0.001  80000.0    21600.0        3.0         29    1
14 W        0.10   80000.0    21600.0        3.0         30    1
"""
WRF_STOCHPERT_TBL_SHA256 = (
    "e0e6f147628f91ab3e7f3f4be6fc27296a43e1da8ded404b7fe06501fb42559e")

#: The fields each scheme reads (the drivers' perts_* arguments,
#: first_rk_step_part1.F:331-351, :1653-1662, solve_em.F:3975-3986).
CONSUMED: dict[str, tuple[str, ...]] = {
    "pert_farms": ("ALBEDO", "AOD", "ANGSTROM", "ASSYMFAC", "QVAPOR",
                   "QCLOUD", "QSNOW"),
    "pert_deng": ("QVAPOR", "QCLOUD", "TH", "W"),
    "pert_thom": ("QVAPOR", "QCLOUD", "QICE", "QSNOW", "NI"),
    "pert_cld3": ("TH", "QVAPOR"),
}

#: The factors of each switch, in WRF's argument order.
FACTORS: dict[str, tuple[str, ...]] = {
    "pert_farms": ("pert_farms_albedo", "pert_farms_aod", "pert_farms_angexp",
                   "pert_farms_aerasy", "pert_farms_qv", "pert_farms_qc",
                   "pert_farms_qs"),
    "pert_deng": ("pert_deng_qv", "pert_deng_qc", "pert_deng_t", "pert_deng_w"),
    "pert_thom": ("pert_thom_qv", "pert_thom_qc", "pert_thom_qi",
                  "pert_thom_qs", "pert_thom_ni"),
    "pert_cld3": ("pert_cld3_qv", "pert_cld3_t"),
}
SWITCHES: tuple[str, ...] = tuple(FACTORS)

# The drivers' clamp parameters.
_RAD = dict(ALBEDO_MIN=F(0.0), ALBEDO_MAX=F(1.0), ANGEXP_MIN=F(0.0),
            AOD_MIN=F(0.0), AERASY_MIN=F(-1.0), AERASY_MAX=F(1.0),
            QVAPOR_MIN=F(0.0), QCLOUD_MIN=F(0.0), QSNOW_MIN=F(0.0))
_SHCU = dict(QCLOUD_MIN=F(0.0), QVAPOR_MIN=F(0.0))
_MP = dict(QX_MIN=F(1.e-12), NI_MIN=F(1.e-6))

_OP_MUL, _OP_DIV, _OP_SUB = 0, 1, 2
# Clamp forms (woof/core/kernels/wrfsolar_eps.cu): which operand a
# signed-zero tie keeps is GCC's choice per statement; each statement below
# names the form the x86-64 record shows.
_CLAMP_NONE, _KEEP_Y, _KEEP_LO, _RANGE_KEEP_Y, _RANGE_KEEP_LO = 0, 1, 2, 3, 4


class StochpertTableError(ValueError):
    """A STOCHPERT.TBL WRF would stop on, or one woof cannot run."""


@dataclass(frozen=True)
class StochpertRow:
    name: str
    std: float
    lengthscale: float
    timescale: float
    cutoff: float
    seed: int
    vertstruc: int


def _fortran_real(token: str) -> float:
    value = float(token.lower().replace("d", "e"))
    return float(F(value))


def parse_stochpert_table(text: str, *, num_pert_3d: int = 15) -> dict[str, StochpertRow]:
    """INITIALIZE_STOCH's read (module_stoch.F:293-364): three header lines
    skipped, then ``num_pert_3d - 1`` list-directed rows ``nfield varname
    std lengthscale timescale std_cutoff iseed vertstruc``, each landing on
    its field by name.  Returns {name: row} for the rows read."""
    if type(num_pert_3d) is not int or not 2 <= num_pert_3d <= 15:
        raise StochpertTableError(
            f"num_pert_3d={num_pert_3d!r}: WRF reads num_pert_3d - 1 table rows "
            "into the 14 pert3d fields; woof runs 2 through 15")
    lines = text.splitlines()
    rows: dict[str, StochpertRow] = {}
    body = [line for line in lines[3:]]
    tokens: list[str] = []
    for line in body:
        tokens.extend(line.replace(",", " ").split())
    need = num_pert_3d - 1
    for r in range(need):
        chunk = tokens[8 * r: 8 * r + 8]
        if len(chunk) < 8:
            raise StochpertTableError(
                f"STOCHPERT.TBL holds {r} complete rows; num_pert_3d = "
                f"{num_pert_3d} reads {need} (WRF stops at the end of file)")
        _, name, std, length, tau, cutoff, seed, vert = chunk
        name = name.strip("'\"").upper()
        if name not in FIELDS:
            raise StochpertTableError(
                f"STOCHPERT.TBL row {r + 1}: {name!r} -- WRF stops with "
                "'module_stoch.F: Invalid entry in STOCHPERT.TBL'")
        if name in rows:
            raise StochpertTableError(
                f"STOCHPERT.TBL names {name} twice; WRF keeps the later row's "
                "parameters but sets up the field by its index, which woof "
                "does not reproduce")
        try:
            row = StochpertRow(name, _fortran_real(std), _fortran_real(length),
                               _fortran_real(tau), _fortran_real(cutoff),
                               int(seed), int(vert))
        except ValueError as problem:
            raise StochpertTableError(f"STOCHPERT.TBL row {r + 1}: {problem}") from problem
        if row.vertstruc not in (0, 1):
            raise StochpertTableError(
                f"STOCHPERT.TBL {name}: vertical structure {row.vertstruc}; WRF "
                "sets up 0 (constant) and 1 (random phase) only")
        for label, value in (("std", row.std), ("lengthscale", row.lengthscale),
                             ("decorrelation time", row.timescale),
                             ("std cutoff", row.cutoff)):
            if not math.isfinite(value) or value < 0:
                raise StochpertTableError(f"STOCHPERT.TBL {name}: {label} {value}")
        if row.timescale <= 0:
            raise StochpertTableError(f"STOCHPERT.TBL {name}: decorrelation time must be positive")
        rows[name] = row
    return rows


def load_stochpert_table(path: str, *, num_pert_3d: int = 15) -> dict[str, StochpertRow]:
    """``""`` is WRF v4.7.1's shipped run/STOCHPERT.TBL."""
    text = WRF_STOCHPERT_TBL if not path else Path(path).read_text()
    return parse_stochpert_table(text, num_pert_3d=num_pert_3d)


def field_index(name: str) -> int:
    """WRF's P_P<name> (2..15)."""
    return FIELDS.index(name) + 2


def wrf_nint(x: np.float32) -> int:
    """Fortran NINT: round half away from zero."""
    x = float(x)
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def stepsp(spdt: float, dt: float) -> int:
    """``grid%stepsp = max(nint(spdt*60.0/grid%dt), 1)`` (module_stoch.F:276-277),
    float32 left to right."""
    q = F(F(F(spdt) * F(60.0)) / F(dt))
    return max(wrf_nint(q), 1)


def wrf_time_step(dt: float) -> int:
    """WRF's integer namelist ``time_step`` behind ``grid%dt``: the whole
    seconds (``time_step_fract_num/den`` is the rest)."""
    return int(math.floor(float(dt)))


def active_switches(cfg) -> tuple[str, ...]:
    if int(getattr(cfg, "multi_perturb", 0) or 0) != 1:
        return ()
    return tuple(s for s in SWITCHES if bool(getattr(cfg, s, False)))


def consumed_fields(cfg) -> tuple[str, ...]:
    """The pert3d fields this domain's hooks read, in pert3d order."""
    want = set()
    for switch in active_switches(cfg):
        want.update(CONSUMED[switch])
    return tuple(name for name in FIELDS if name in want)


def factors(cfg, switch: str) -> tuple[np.float32, ...]:
    return tuple(F(getattr(cfg, name)) for name in FACTORS[switch])


def field_seed(nens: int, iseed: int, name: str) -> int:
    """The 64-bit Philox key of one field: WRF's labels (nens, the table's
    seed) bound to the field, as woof binds every stochastic process
    (woof/ensemble/stochastic_seeds.py)."""
    payload = f"woof-wrfsolar-mult3d.v1:{int(nens)}:{int(iseed)}:{name}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


# ---------------------------------------------------------------------------
# The hook kernel.

def _kernel(name):
    from woof.core.kernels import get_kernel
    return get_kernel("wrfsolar_eps", name)


_BLOCK = 256


def _dummy():
    import cupy as cp
    return cp.zeros(1, dtype=cp.float32)


def apply(x, pert, fac, *, op, clamp=_CLAMP_NONE, lo=0.0, hi=0.0, tmp=None,
          save=False, n=None, x_off=0, pert_off=0, planar=False, plane=1):
    """One hook statement over ``n`` elements of ``x`` (in place).

    ``pert`` is read at ``pert_off + i`` (or ``pert_off + i % plane`` when
    ``planar``).  Returns the saved pre-update values when ``save``."""
    import cupy as cp
    if x.dtype != cp.float32 or pert.dtype != cp.float32:
        raise TypeError("multi_perturb hooks run on float32 arrays")
    if not x.flags.c_contiguous or not pert.flags.c_contiguous:
        raise ValueError("multi_perturb hooks need C-contiguous arrays")
    n = x.size - x_off if n is None else int(n)
    if n <= 0:
        return cp.empty(0, cp.float32) if save else None
    saved = cp.empty(n, dtype=cp.float32) if save else None
    dummy = _dummy()
    if op == _OP_SUB:
        if tmp is None or tmp.size != n or tmp.dtype != cp.float32:
            raise ValueError("the subtracting Remove_ statement needs Add_'s saved values")
    _kernel("eps_apply")(
        ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
        (np.int32(n), x, np.int32(x_off), pert, np.int32(pert_off),
         np.int32(1 if planar else 0), np.int32(plane), F(fac), np.int32(op),
         np.int32(clamp), F(lo), F(hi),
         tmp if tmp is not None else dummy,
         saved if saved is not None else dummy, np.int32(1 if save else 0)))
    return saved


class Pattern:
    """One pert3d field on a domain: ``values`` is (nlev, ny, nx) float32,
    nlev = 1 for a field constant in height, nz otherwise."""

    def __init__(self, values):
        self.values = values

    @property
    def planar(self) -> bool:
        return self.values.shape[0] == 1

    def level_args(self, x_shape, *, k0=0):
        """kernel arguments to read this pattern against a (nk, ny, nx)
        operand that starts at pattern level ``k0``."""
        plane = int(np.prod(self.values.shape[1:]))
        if self.planar:
            return dict(pert_off=0, planar=True, plane=plane)
        return dict(pert_off=k0 * plane, planar=False, plane=plane)

    def grid(self, nz):
        """(nz, ny, nx) contiguous values (a broadcast copy when planar)."""
        import cupy as cp
        if self.planar:
            return cp.ascontiguousarray(cp.broadcast_to(self.values, (nz, *self.values.shape[1:])))
        return self.values[:nz]


def _plane_of(shape):
    return int(np.prod(shape[-2:]))


# -- FARMS ------------------------------------------------------------------

def farms_add(perts, fac, *, albedo, aod, angexp, aerasy, qv, qc, qs):
    """Add_multi_perturb_swrad_perturbations.  ``perts`` maps field names
    to :class:`Pattern`; ``fac`` the seven factors in WRF's order.  Returns
    Add_'s saved (qv_tmp, qc_tmp, qs_tmp)."""
    f_alb, f_aod, f_ang, f_asy, f_qv, f_qc, f_qs = (F(v) for v in fac)
    plane = _plane_of(albedo.shape)
    # Add_'s 2-D statements bound the albedo above by AERASY_MAX (Remove_
    # uses ALBEDO_MAX); both are 1.0.
    for x, name, f, clamp, lo, hi in (
            (albedo, "ALBEDO", f_alb, _RANGE_KEEP_LO, _RAD["ALBEDO_MIN"], _RAD["AERASY_MAX"]),
            (angexp, "ANGSTROM", f_ang, _KEEP_Y, _RAD["ANGEXP_MIN"], 0.0),
            (aod, "AOD", f_aod, _KEEP_Y, _RAD["AOD_MIN"], 0.0),
            (aerasy, "ASSYMFAC", f_asy, _RANGE_KEEP_LO, _RAD["AERASY_MIN"], _RAD["AERASY_MAX"])):
        # k = 1: the pattern's lowest level, whatever its depth
        apply(x, perts[name].values, f, op=_OP_MUL, clamp=clamp, lo=lo, hi=hi,
              pert_off=0, planar=True, plane=plane)
    saved = []
    # WRF's 3-D loop interleaves qv, qc, qs per element; each statement
    # reads only its own array, so field-by-field passes give the same words.
    for x, name, f, lo in ((qv, "QVAPOR", f_qv, _RAD["QVAPOR_MIN"]),
                           (qc, "QCLOUD", f_qc, _RAD["QCLOUD_MIN"]),
                           (qs, "QSNOW", f_qs, _RAD["QSNOW_MIN"])):
        saved.append(apply(x, perts[name].values, f, op=_OP_MUL,
                           clamp=_KEEP_Y, lo=lo, save=True,
                           **perts[name].level_args(x.shape)))
    return tuple(saved)


def farms_remove(perts, fac, saved, *, albedo, aod, angexp, aerasy, qv, qc, qs):
    """Remove_multi_perturb_swrad_perturbations."""
    f_alb, f_aod, f_ang, f_asy, f_qv, f_qc, f_qs = (F(v) for v in fac)
    plane = _plane_of(albedo.shape)
    for x, name, f, clamp, lo, hi in (
            (albedo, "ALBEDO", f_alb, _RANGE_KEEP_Y, _RAD["ALBEDO_MIN"], _RAD["ALBEDO_MAX"]),
            (angexp, "ANGSTROM", f_ang, _KEEP_LO, _RAD["ANGEXP_MIN"], 0.0),
            (aod, "AOD", f_aod, _KEEP_LO, _RAD["AOD_MIN"], 0.0),
            (aerasy, "ASSYMFAC", f_asy, _RANGE_KEEP_Y, _RAD["AERASY_MIN"], _RAD["AERASY_MAX"])):
        apply(x, perts[name].values, f, op=_OP_DIV, clamp=clamp, lo=lo, hi=hi,
              pert_off=0, planar=True, plane=plane)
    qv_tmp, qc_tmp, qs_tmp = saved
    for x, name, f, lo, tmp in ((qv, "QVAPOR", f_qv, _RAD["QVAPOR_MIN"], qv_tmp),
                                (qc, "QCLOUD", f_qc, _RAD["QCLOUD_MIN"], qc_tmp),
                                (qs, "QSNOW", f_qs, _RAD["QSNOW_MIN"], qs_tmp)):
        apply(x, perts[name].values, f, op=_OP_SUB, clamp=_KEEP_LO,
              lo=lo, tmp=tmp, **perts[name].level_args(x.shape))


# -- Deng ---------------------------------------------------------------------

def deng_add(perts, fac, *, th, qv, qc, w):
    """Add_multi_perturb_shcu_perturbations.  ``w`` is (nz+1, ny, nx); its
    levels 2..nz (kts+1..kte) take ``pert_deng_t`` -- WRF's own argument."""
    f_qv, f_qc, f_t, f_w = (F(v) for v in fac)
    apply(qc, perts["QCLOUD"].values, f_qc, op=_OP_MUL, clamp=_KEEP_Y,
          lo=_SHCU["QCLOUD_MIN"], **perts["QCLOUD"].level_args(qc.shape))
    apply(qv, perts["QVAPOR"].values, f_qv, op=_OP_MUL, clamp=_KEEP_Y,
          lo=_SHCU["QVAPOR_MIN"], **perts["QVAPOR"].level_args(qv.shape))
    apply(th, perts["TH"].values, f_t, op=_OP_MUL, **perts["TH"].level_args(th.shape))
    _deng_w(perts, f_t, w, qv.shape, op=_OP_MUL)


def deng_remove(perts, fac, *, th, qv, qc, w):
    """Remove_multi_perturb_shcu_perturbations (w divided by ``pert_deng_w``)."""
    f_qv, f_qc, f_t, f_w = (F(v) for v in fac)
    apply(qc, perts["QCLOUD"].values, f_qc, op=_OP_DIV, clamp=_KEEP_Y,
          lo=_SHCU["QCLOUD_MIN"], **perts["QCLOUD"].level_args(qc.shape))
    apply(qv, perts["QVAPOR"].values, f_qv, op=_OP_DIV, clamp=_KEEP_Y,
          lo=_SHCU["QVAPOR_MIN"], **perts["QVAPOR"].level_args(qv.shape))
    apply(th, perts["TH"].values, f_t, op=_OP_DIV, **perts["TH"].level_args(th.shape))
    _deng_w(perts, f_w, w, qv.shape, op=_OP_DIV)


def _deng_w(perts, f, w, mass_shape, *, op):
    nz = mass_shape[0]
    plane = _plane_of(mass_shape)
    if nz < 2:
        return
    args = perts["W"].level_args(mass_shape, k0=1)
    apply(w, perts["W"].values, f, op=op, n=(nz - 1) * plane, x_off=plane, **args)


# -- Thompson -------------------------------------------------------------------

_THOM = (("QVAPOR", "QX_MIN"), ("QCLOUD", "QX_MIN"), ("QICE", "QX_MIN"),
         ("QSNOW", "QX_MIN"), ("NI", "NI_MIN"))


def thompson_add(perts, fac, *, qv, qc, qi, qs, ni):
    """Add_multi_perturb_mp_perturbations; returns the five saved arrays."""
    saved = []
    for x, (name, floor), f in zip((qv, qc, qi, qs, ni), _THOM, fac):
        saved.append(apply(x, perts[name].values, F(f), op=_OP_MUL,
                           clamp=_KEEP_Y, lo=_MP[floor], save=True,
                           **perts[name].level_args(x.shape)))
    return tuple(saved)


def thompson_remove(perts, fac, saved, *, qv, qc, qi, qs, ni):
    """Remove_multi_perturb_mp_perturbations on the post-microphysics values."""
    for x, (name, floor), f, tmp in zip((qv, qc, qi, qs, ni), _THOM, fac, saved):
        apply(x, perts[name].values, F(f), op=_OP_SUB, clamp=_KEEP_Y,
              lo=_MP[floor], tmp=tmp, **perts[name].level_args(x.shape))


# -- icloud = 3 -------------------------------------------------------------------

def cld3_inputs(t, qv, pert_th, pert_qv, f_qv, f_t):
    """The t_1d/qv_1d cal_cldfra3 reads under pert_cld3 (:1472-1474):
    ``(1 + perts_th*pert_cld3_t)*t`` and ``max(0, (1 +
    perts_qvapor*pert_cld3_qv)*qv)`` -- new arrays, inputs untouched.  The
    four arrays share one layout (any; the statements are elementwise)."""
    import cupy as cp
    t_out = cp.array(t, dtype=cp.float32, copy=True, order="C")
    qv_out = cp.array(qv, dtype=cp.float32, copy=True, order="C")
    pert_th = cp.ascontiguousarray(pert_th, dtype=cp.float32)
    pert_qv = cp.ascontiguousarray(pert_qv, dtype=cp.float32)
    if pert_th.shape != t_out.shape or pert_qv.shape != qv_out.shape:
        raise ValueError("pert_cld3: the patterns must be laid out as t and qv are")
    apply(t_out, pert_th, F(f_t), op=_OP_MUL)
    apply(qv_out, pert_qv, F(f_qv), op=_OP_MUL, clamp=_KEEP_Y, lo=F(0.0))
    return t_out, qv_out


# ---------------------------------------------------------------------------
# The pattern generator (module_stoch.F, variable 'R'/'T').

def pattern_field_name(name: str) -> str:
    """WRF's Registry name of a pert3d component (lower case)."""
    return "p" + name.lower()


def spectrum_field_names(name: str) -> tuple[str, str]:
    """The SPFORCC3d/SPFORCS3d slices of one field, as driver fields."""
    p = pattern_field_name(name)
    return f"spforcc3d_{p}", f"spforcs3d_{p}"


def multi_perturb_field_shapes(cfg, nz: int, ny: int, nx: int) -> dict:
    """Name -> (shape, dtype) of every multi_perturb field a domain's driver
    holds (and the checkpoint carries): each consumed pert3d component,
    (nz, ny, nx) under a random-phase row and (1, ny, nx) under a constant
    one, and its spectral coefficients on the staggered (ny+1, nx+1) grid."""
    fields = consumed_fields(cfg)
    if not fields:
        return {}
    table = load_stochpert_table(getattr(cfg, "stochpert_tbl", ""),
                                 num_pert_3d=int(cfg.num_pert_3d))
    shapes = {}
    for name in fields:
        row = table.get(name)
        nlev = nz if (row is not None and row.vertstruc == 1) else 1
        shapes[pattern_field_name(name)] = ((nlev, ny, nx), np.float32)
        for spec in spectrum_field_names(name):
            shapes[spec] = ((ny + 1, nx + 1), np.float32)
    return shapes


def allocate_multi_perturb_fields(fields: dict, cfg, nz: int, ny: int, nx: int) -> None:
    """INITIALIZE_STOCH's start: pert3d and the spectra are zero."""
    import cupy as cp
    for name, (shape, _dtype) in multi_perturb_field_shapes(cfg, nz, ny, nx).items():
        fields[name] = cp.zeros(shape, dtype=cp.float32)


def backward_fft2(real, imag):
    """do_fftback_along_x then do_fftback_along_y (module_stoch.F:1354-1452):
    FFTPACK5's unnormalized complex backward transform along each row, then
    each column, of (..., KMAX, LMAX) RAND_REAL + i RAND_IMAG.  Returns the
    real part, float32.  cuFFT on CUDA; NumPy's pocketfft on the host
    elsewhere (the facade's FFTs are host FFTs)."""
    import cupy as cp
    from woof import backend
    if backend.active() == "cuda":
        z = real.astype(cp.complex64)
        z.imag = imag
        out = cp.fft.ifft2(z, axes=(-2, -1), norm="forward")
        return cp.ascontiguousarray(out.real, dtype=cp.float32)
    r = np.asarray(cp.asnumpy(real), dtype=np.float32)
    i = np.asarray(cp.asnumpy(imag), dtype=np.float32)
    z = r.astype(np.complex64)
    z.imag = i
    out = np.fft.ifft2(z, axes=(-2, -1), norm="forward")
    return cp.asarray(np.ascontiguousarray(out.real, dtype=np.float32))


class FieldGenerator:
    """One STOCHPERT.TBL field's process on one domain (SP_AMP3d, ALPH_RAND3d,
    SPFORCC3d/SPFORCS3d and its pert3d component)."""

    def __init__(self, name, row, *, lmax, kmax, dx, dy, itime_step, nens,
                 nz, kminforct=1, lminforct=1, kmaxforct=1000000,
                 lmaxforct=1000000):
        import cupy as cp
        self.name, self.row = name, row
        self.lmax, self.kmax = int(lmax), int(kmax)
        self.dx, self.dy = F(dx), F(dy)
        self.nz = int(nz)
        self.nlev = self.nz if row.vertstruc == 1 else 1
        self.seed = field_seed(nens, row.seed, name)
        index = field_index(name)
        self.streams = (np.uint32(64 + 2 * index), np.uint32(65 + 2 * index))
        zchi = cp.empty(self.lmax * self.kmax, dtype=cp.float32)
        self.sp_amp = cp.empty((self.kmax, self.lmax), dtype=cp.float32)
        out = cp.zeros(3, dtype=cp.float32)
        _kernel("mp_setup_amplitude")((1,), (1,), (
            np.int32(self.lmax), np.int32(self.kmax), self.dx, self.dy,
            np.int32(itime_step), F(row.std), F(row.lengthscale),
            F(row.timescale), np.int32(kminforct), np.int32(lminforct),
            np.int32(kmaxforct), np.int32(lmaxforct), zchi, self.sp_amp, out))
        host = cp.asnumpy(out)
        self.alph, self.zconstf0, self.zgamman = (F(v) for v in host)

    def draws(self, step: int):
        """ZRANDNOSS, ZRANDNOSC for this update (Philox, see the kernel)."""
        import cupy as cp
        n = self.lmax * self.kmax
        noss = cp.empty((self.kmax, self.lmax), dtype=cp.float32)
        nosc = cp.empty_like(noss)
        _kernel("mp_noise")(((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), (
            np.int32(n), np.uint32(step), self.streams[0], self.streams[1],
            np.uint64(self.seed), noss, nosc))
        return noss, nosc

    def ar1(self, spforcc, spforcs, noss, nosc):
        n = self.lmax * self.kmax
        _kernel("mp_ar1")(((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), (
            np.int32(self.lmax), np.int32(self.kmax), self.alph, self.sp_amp,
            noss, nosc, spforcc, spforcs))

    def fill(self, spforcc, spforcs):
        import cupy as cp
        shape = (self.nlev, self.kmax, self.lmax)
        real = cp.empty(shape, dtype=cp.float32)
        imag = cp.empty(shape, dtype=cp.float32)
        n = int(np.prod(shape))
        # SETUP_RAND_PERTURB filled VERTSTRUCC/S over kts..kte = kps..kpe,
        # the staggered column (start_em.F:147): every mass level.
        _kernel("mp_fill_levels")(((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), (
            np.int32(self.lmax), np.int32(self.kmax), np.int32(self.nlev),
            np.int32(self.row.vertstruc), np.int32(self.nz + 1), self.dx, self.dy,
            spforcc, spforcs, real, imag))
        return real, imag

    def clip(self, rand_real, pert):
        n = pert.size
        _kernel("mp_clip")(((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), (
            np.int32(self.lmax), np.int32(self.kmax), np.int32(self.nlev),
            F(self.row.cutoff), F(self.row.std), rand_real, pert))

    def update(self, step: int, spforcc, spforcs, pert):
        """RAND_PERT_UPDATE(grid, 'T', ...) for this field, in place."""
        noss, nosc = self.draws(step)
        self.ar1(spforcc, spforcs, noss, nosc)
        real, imag = self.fill(spforcc, spforcs)
        self.clip(backward_fft2(real, imag), pert)


class MultiPerturb:
    """A domain's WRF-Solar EPS patterns, held in the driver's fields.

    Built on every domain with ``multi_perturb = 1``; only domain 1 updates
    (first_rk_step_part2.F:267, ``grid%id == 1``), so a nest's patterns stay
    the zero WRF allocated and its hooks run on zero perturbations (the
    clamps still apply).  ``update(itimestep)`` is the first_rk_step_part2
    call: the first step, then every ``stepsp`` steps."""

    def __init__(self, cfg, fields: dict, *, nz: int, ny: int, nx: int):
        self.cfg = cfg
        self.fields = fields
        self.switches = active_switches(cfg)
        self.names = consumed_fields(cfg)
        table = load_stochpert_table(getattr(cfg, "stochpert_tbl", ""),
                                     num_pert_3d=int(cfg.num_pert_3d))
        self.stepsp = stepsp(float(cfg.spdt), float(cfg.dt))
        self.itime_step = self.stepsp * wrf_time_step(float(cfg.dt))
        self.updates = int(getattr(cfg, "grid_id", 1)) == 1
        self.generators = {}
        for name in self.names:
            row = table.get(name)
            if row is None or field_index(name) > int(cfg.num_pert_3d):
                # never set up or updated: WRF's zero pert3d component
                continue
            self.generators[name] = FieldGenerator(
                name, row, lmax=nx + 1, kmax=ny + 1, dx=cfg.dx, dy=cfg.dy,
                itime_step=self.itime_step, nens=int(cfg.nens), nz=nz)
        self.nz = nz

    def due(self, itimestep: int) -> bool:
        if not self.updates:
            return False
        test = (itimestep % self.stepsp == 0) if self.stepsp > 0 else False
        return bool(test or itimestep == 1)

    def update(self, itimestep: int) -> bool:
        if not self.due(int(itimestep)):
            return False
        for name, gen in self.generators.items():
            c, s = spectrum_field_names(name)
            gen.update(int(itimestep), self.fields[c], self.fields[s],
                       self.fields[pattern_field_name(name)])
        return True

    def patterns(self) -> dict:
        """{field name: Pattern} over the consumed fields."""
        return {name: Pattern(self.fields[pattern_field_name(name)])
                for name in self.names}


    def factors(self, switch: str):
        """The switch's factors when it is on here, else None."""
        return factors(self.cfg, switch) if switch in self.switches else None

    def cld3_columns(self, transform):
        """pert_cld3's (pert_th, pert_qv) in a cal_cldfra3 caller's column
        layout: ``transform`` maps a (nz, ny, nx) field to the layout its
        t and qv take.  None when pert_cld3 is off."""
        if "pert_cld3" not in self.switches:
            return None
        pats = self.patterns()
        return dict(pert_th=transform(pats["TH"].grid(self.nz)),
                    pert_qv=transform(pats["QVAPOR"].grid(self.nz)),
                    factors=factors(self.cfg, "pert_cld3"))
