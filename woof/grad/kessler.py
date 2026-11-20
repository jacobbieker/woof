"""Kessler warm rain (``mp_physics = 1``) as differentiable PyTorch (plan P1.2).

The forward pass is woof's ``kessler_column`` kernel
(``woof/core/kernels/kessler.cu``; its NumPy mirror is
:func:`woof.verify.npref.np_kessler_column`), written as torch operations
on ``[level, column]`` tensors, so PyTorch's autograd gives the
tangent-linear (forward mode) and the adjoint (``backward``) of the scheme.
It is opt-in and separate: woof's forecasts keep running the kernel, and
nothing on a forecast path imports this module.

What it reproduces, and how closely
-----------------------------------
Every expression is the kernel's, in the kernel's order: fall speed and
the split count ``nfall = max(1, nint(0.5 + crmax / 0.75))``, the
time-split flux-upstream sedimentation with its per-split surface rain,
the re-derived split count, then autoconversion/accretion on the
pre-sedimentation fields, the Teten saturation adjustment with woof's
``es <= 0.99 p`` guard, rain evaporation and the clamped update.  The
loop over split steps runs per column with a mask (each column keeps its
own ``nfall``), until no column has steps left.

It is not bit-identical to the kernel: the kernel's powers and exponential
are woof's glibc-exact ``gfk_pow``/``gfk_exp`` compiled with
``--fmad=false``, torch uses its backend's ``pow``/``exp``.  In float32
the words differ by a few ULP, and a column whose ``crmax / 0.75`` sits on
a rounding boundary can take another split count.
``tests/test_grad_kessler.py`` and ``tools/grad_kessler_g1.py`` measure
both against the kernel on the WRF oracle fixture.

Derivatives
-----------
The scheme is piecewise smooth.  Autograd differentiates the branch the
forward took (plan section 3(f), option 1):

* ``max``/``min`` clamps pass the derivative of the side they took;
* the split count ``nfall`` and the split length are integers chosen from
  the state: they are held fixed (zero derivative), as a branch;
* a fractional power ``x ** e`` (``e`` < 1) of a quantity that is zero
  has an infinite one-sided derivative; :func:`_spow` gives it a zero
  derivative there (the rain-free branch), and the true, large derivative
  wherever the quantity is positive.

So the adjoint is exact for the realised branch and wrong across a switch,
like any branch-taken adjoint: finite differences agree with it only for
perturbations that keep every column on its branch.
"""

from __future__ import annotations

from typing import Tuple

import torch

from woof.core import constants as _c

RD = float(_c.RD)
RV = float(_c.RV)
CP = float(_c.CP)
XLV = float(_c.XLV)
SVP1 = float(_c.SVP1)
SVP2 = float(_c.SVP2)
SVP3 = float(_c.SVP3)
SVPT0 = float(_c.SVPT0)
RHOWATER = float(_c.RHOWATER)
EP2 = float(_c.EP2)

C1K, C2K, C3K, C4K = 0.001, 0.001, 2.2, 0.875
MAX_CR_SEDIMENTATION = 0.75
F5 = SVP2 * (SVPT0 - SVP3) * XLV / CP
#: Rd/cp of WRF's Exner function (woof.core.microphysics.prepare_microphysics_exner).
RCP = RD / CP

# TorchScript (the seat's artefact format) cannot read module-level floats,
# so the functions below spell these constants as literals;
# tests/test_grad_kessler.py holds each literal to the value above.


def _min(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """min(a, b) that gives a tie to ``a``: ``torch.minimum`` splits the
    gradient half and half on an exact tie, which is the wrong one-sided
    derivative where the branches meet (e.g. qr == 0 in subsaturated air)."""
    return torch.where(a <= b, a, b)


def _spow(x: torch.Tensor, e: float) -> torch.Tensor:
    """``x ** e`` for ``x >= 0``, with a zero derivative where ``x == 0``
    (where ``e < 1`` makes the true one-sided derivative infinite)."""
    positive = x > 0
    safe = torch.where(positive, x, torch.ones_like(x))
    return torch.where(positive, safe ** e, torch.zeros_like(x))


def _nint_split(crmax: torch.Tensor) -> torch.Tensor:
    """``max(1, nint(0.5 + crmax / 0.75))``, Fortran NINT, as an integer
    tensor (no derivative: a branch)."""
    n = torch.floor(0.5 + crmax.detach() / 0.75 + 0.5)
    return torch.clamp(n, min=1.0).to(torch.int64)


def kessler_columns(t: torch.Tensor, qv: torch.Tensor, qc: torch.Tensor, qr: torch.Tensor,
                    rho: torch.Tensor, pii: torch.Tensor, z: torch.Tensor, dz8w: torch.Tensor,
                    dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One Kessler call on ``[nz, ncol]`` columns (level 0 at the bottom).

    ``t`` full (dry) potential temperature (K), ``qv/qc/qr`` (kg/kg),
    ``rho`` dry density, ``pii`` Exner, ``z`` half-level heights and
    ``dz8w`` layer depths (m), as the kernel takes them.  Returns new
    tensors ``(t, qv, qc, qr, rainncv)``; ``rainncv`` ``[ncol]`` is the
    last split step's surface rain (mm), the kernel's RAINNCV (RAINNC
    gains the sum over split steps; :func:`kessler_rain` returns it)."""
    out = _kessler(t, qv, qc, qr, rho, pii, z, dz8w, dt)
    return out[0], out[1], out[2], out[3], out[4]


def kessler_rain(t: torch.Tensor, qv: torch.Tensor, qc: torch.Tensor, qr: torch.Tensor,
                 rho: torch.Tensor, pii: torch.Tensor, z: torch.Tensor, dz8w: torch.Tensor,
                 dt: float) -> torch.Tensor:
    """The call's RAINNC increment (mm), ``[ncol]``."""
    return _kessler(t, qv, qc, qr, rho, pii, z, dz8w, dt)[5]


_State = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]


def _kessler(t: torch.Tensor, qv: torch.Tensor, qc: torch.Tensor, qr: torch.Tensor,
             rho: torch.Tensor, pii: torch.Tensor, z: torch.Tensor, dz8w: torch.Tensor,
             dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    vtden, rdzw, rdzk, s = _setup(qr, rho, z, dz8w, dt)
    while bool((s[0] > 0).any()):
        s = _split_step(s, rho, rdzk, rdzw, vtden)
    return _production(t, qv, qc, qr, s[4], rho, pii, s[5], s[6], dt)


def _setup(qr: torch.Tensor, rho: torch.Tensor, z: torch.Tensor, dz8w: torch.Tensor,
           dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, _State]:
    """Fall speed, the split count and the sedimentation loop's state
    ``(nfall, dtfall, time_sediment, vt, prodk, rain, rainncv)``.  ``prodk``
    is a copy of ``qr``: no state slot aliases an input (``torch.compile``'s
    backward mis-routes the gradient of an input returned as an output)."""
    nz = qr.shape[0]

    # --- terminal velocity and the stable time-split count
    rhok = rho
    vtden = torch.sqrt(rhok[0:1] / rhok)
    rdzw = 1.0 / dz8w
    vt = 36.34 * _spow(torch.clamp(qr * 0.001 * rhok, min=0.0), 0.1364) * vtden
    crmax = torch.amax(vt * dt * rdzw, dim=0)
    rdzk = torch.cat([1.0 / (z[1:] - z[:nz - 1]), 1.0 / (z[nz - 1:] - z[nz - 2:nz - 1])], dim=0)

    nfall = _nint_split(crmax)
    dtfall = dt / nfall.to(qr.dtype)
    time_sediment = torch.full_like(crmax, dt)
    rain = torch.zeros_like(crmax)
    rainncv = torch.zeros_like(crmax)
    return vtden, rdzw, rdzk, (nfall, dtfall, time_sediment, vt, qr.clone(), rain, rainncv)


def _split_step(s: _State, rhok: torch.Tensor, rdzk: torch.Tensor, rdzw: torch.Tensor,
                vtden: torch.Tensor) -> _State:
    """One time-split sedimentation step of every column with steps left
    (flux-upstream fallout, the surface rain, the re-derived split)."""
    nfall, dtfall, time_sediment, vt, prodk, rain, rainncv = s
    nz = prodk.shape[0]
    active = nfall > 0
    a = active.to(prodk.dtype)
    time_sediment = time_sediment - a * dtfall
    ppt = rhok[0] * prodk[0] * vt[0] * dtfall / 1000.0
    rainncv = torch.where(active, ppt * 1000.0, rainncv)
    rain = rain + a * ppt * 1000.0

    flux = rhok * prodk * vt
    lower = prodk[:nz - 1] - (dtfall * rdzk[:nz - 1] / rhok[:nz - 1]) * (flux[:nz - 1] - flux[1:])
    top = prodk[nz - 1:] - (dtfall * rdzk[nz - 1:]) * prodk[nz - 1:] * vt[nz - 1:]
    prodk = torch.where(active, torch.cat([lower, top], dim=0), prodk)

    more = active & (nfall > 1)
    nfall = torch.where(more, nfall - 1, torch.zeros_like(nfall))
    vt_new = 36.34 * _spow(torch.clamp(prodk * 0.001 * rhok, min=0.0), 0.1364) * vtden
    vt = torch.where(more, vt_new, vt)
    crmax = torch.amax(vt_new * time_sediment * rdzw, dim=0)
    nfall_new = _nint_split(crmax)
    change = more & (nfall_new != nfall)
    nfall = torch.where(change, nfall_new, nfall)
    safe_n = torch.clamp(nfall, min=1).to(prodk.dtype)
    dtfall = torch.where(change, time_sediment / safe_n, dtfall)
    return nfall, dtfall, time_sediment, vt, prodk, rain, rainncv


def _production(t: torch.Tensor, qv: torch.Tensor, qc: torch.Tensor, qr: torch.Tensor, prodk: torch.Tensor,
                rho: torch.Tensor, pii: torch.Tensor, rain: torch.Tensor, rainncv: torch.Tensor,
                dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autoconversion and accretion, the saturation adjustment and rain
    evaporation, on the pre-sedimentation ``qc``/``qr`` and the fallen
    ``prodk``."""
    factorn = 1.0 / (1.0 + 2.2 * dt * _spow(torch.clamp(qr, min=0.0), 0.875))
    qrprod = qc * (1.0 - factorn) + factorn * 0.001 * dt * torch.clamp(qc - 0.001, min=0.0)
    rcgs = 0.001 * rho

    qck = torch.clamp(qc - qrprod, min=0.0)
    qrk = (qr + prodk) - qr                  # the file's own expression
    qrk = torch.clamp(qrk + qrprod, min=0.0)

    temp = pii * t
    pressure = 1.0e5 * pii ** (1004.0 / 287.0)
    gam = 2.5e6 / (1004.0 * pii)
    es = 1000.0 * 0.6112 * torch.exp(17.67 * (temp - 273.15) / (temp - 29.65))
    es = _min(es, 0.99 * pressure)  # woof's es >= p guard
    qvs = 0.6217504332755632 * es / (pressure - es)
    prod = (qv - qvs) / (1.0 + pressure / (pressure - es) * qvs * 10708424.589347932 / ((temp - 29.65) * (temp - 29.65)))
    rq = rcgs * qrk
    # a tie goes to qrk (all the rain evaporates), the branch a small
    # positive qr takes
    ern = _min(qrk, _min(
        dt * (((1.6 + 124.9 * _spow(rq, 0.2046)) * _spow(rq, 0.525))
              / (2.55e8 / (pressure * qvs) + 5.4e5))
        * (torch.clamp(qvs - qv, min=0.0) / (rcgs * qvs)),
        torch.clamp(-prod - qck, min=0.0)))

    product = torch.where(prod >= -qck, prod, -qck)
    t_out = t + gam * (product - ern)
    qv_out = torch.clamp(qv - product + ern, min=0.0)
    qc_out = qck + product
    qr_out = qrk - ern
    return t_out, qv_out, qc_out, qr_out, rainncv, rain


_COMPILED: dict = {}


def kessler_compiled(t: torch.Tensor, qv: torch.Tensor, qc: torch.Tensor, qr: torch.Tensor,
                     rho: torch.Tensor, pii: torch.Tensor, z: torch.Tensor, dz8w: torch.Tensor, dt: float):
    """:func:`_kessler` with its three stages fused by ``torch.compile``
    (the same expressions; the loop over split steps stays in Python, so
    each stage compiles once).  Returns the six outputs of
    :func:`_kessler`.  Needs a working ``torch.compile`` backend (Triton on
    CUDA).  Triton flushes float32 denormals to zero, so a denormal mixing
    ratio is zero here (and off the ``x > 0`` branch of the power laws)
    where eager torch keeps it; otherwise the outputs agree with eager to
    rounding and the adjoint passes the dot-product test against eager's
    tangent-linear.  Forward-mode AD does not pass through it."""
    if not _COMPILED:
        _COMPILED.update(setup=torch.compile(_setup, dynamic=False),
                         step=torch.compile(_split_step, dynamic=False),
                         production=torch.compile(_production, dynamic=False))
    vtden, rdzw, rdzk, s = _COMPILED["setup"](qr, rho, z, dz8w, dt)
    while bool((s[0] > 0).any()):
        s = _COMPILED["step"](s, rho, rdzk, rdzw, vtden)
    return _COMPILED["production"](t, qv, qc, qr, s[4], rho, pii, s[5], s[6], dt)


class KesslerColumns(torch.nn.Module):
    """:func:`kessler_columns` as a module with a fixed time step."""

    def __init__(self, dt: float):
        super().__init__()
        self.dt = float(dt)

    def forward(self, t, qv, qc, qr, rho, pii, z, dz8w):
        return kessler_columns(t, qv, qc, qr, rho, pii, z, dz8w, self.dt)


class KesslerSeatModel(torch.nn.Module):
    """Kessler as a column model for woof's NN seat (``woof.nn``).

    Takes the seat's ``[column, feature]`` float32 batch, features
    ``theta, qv, qc, qr, p, z, dz`` over ``levels`` levels each (the
    seat's vocabulary), and returns ``[column, 4 * levels]``: the rates
    ``dtheta, dqv, dqc, dqr`` of one Kessler call of length ``dt`` from
    that state, ``(K(x) - x) / dt``.  The Exner function is WRF's
    ``(p / 1e5) ** (Rd / cp)`` and the dry density follows from WRF's
    moist equation of state, ``p = rho_d Rd T (1 + qv Rv / Rd)`` (the
    kernel is handed ``1 / alt``; the two agree to rounding)."""

    def __init__(self, dt: float, levels: int):
        super().__init__()
        self.dt = float(dt)
        self.levels = int(levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = self.levels
        f = x.t()
        theta, qv, qc, qr = f[0:L], f[L:2 * L], f[2 * L:3 * L], f[3 * L:4 * L]
        p, z, dz = f[4 * L:5 * L], f[5 * L:6 * L], f[6 * L:7 * L]
        pii = (p / 1.0e5) ** 0.2857142857142857
        rho = p / (287.0 * theta * pii * (1.0 + qv * (461.6 / 287.0)))
        t1, qv1, qc1, qr1, _ = kessler_columns(theta, qv, qc, qr, rho, pii, z, dz, self.dt)
        rates = torch.cat([t1 - theta, qv1 - qv, qc1 - qc, qr1 - qr], dim=0) / self.dt
        return rates.t().contiguous()
