"""Clean-room multimode aerosol activation, the default build's (WOOF P8).

Droplet activation of lognormal aerosol modes by an updraft, after
Abdul-Razzak, Ghan and Rivera-Carpio 1998 (JGR 103, 6123; "ARG98") and
Abdul-Razzak and Ghan 2000 (JGR 105, 6837; "ARG00"), integrated over a
Gaussian sub-grid updraft distribution as in Ghan et al. 1997 (JGR 102,
21777).  Written from those papers and textbook thermodynamics only: no
WRF-Chem source (``phys/module_mixactivate.F`` carries Battelle/PNNL
terms) was read to write it.  The exact port of WRF's routine is the
``wrfchem_pnnl`` opt-in (:mod:`woof.optin`); :func:`implementation` picks
between the two behind one contract, and the clean-room code is graded
against the opt-in within the tolerance stated in
``tests/test_chem_activation_cleanroom.py``, never bitwise.

The equations (each also cited at its line in
``woof/core/kernels/chem_activation.cu``)::

    A      = 2 sigma_w M_w / (rho_w R T)                           ARG98 eq. 5
    alpha  = g M_w L / (c_p R T^2) - g M_a / (R T)                 ARG98 eq. 11
    gamma  = R T / (e_s M_w) + M_w L^2 / (c_p p M_a T)             ARG98 eq. 12
    1/G    = rho_w R T / (e_s D_v M_w)
             + L rho_w / (k_a T) (L M_w / (R T) - 1)               ARG98 eq. 16
    S_m,i  = 2 / sqrt(B_i) (A / (3 r_g,i))^(3/2)                   ARG98 eq. 9
    zeta   = 2A/3 (alpha w / G)^(1/2)                              ARG00 eq. 10
    eta_i  = (alpha w / G)^(3/2) / (2 pi rho_w gamma N_i)          ARG00 eq. 9
    f_i    = 0.5 exp(2.5 ln^2 sigma_i),  g_i = 1 + 0.25 ln sigma_i  ARG00 eqs. 7-8
    S_max  = [sum_i S_m,i^-2 (f_i (zeta/eta_i)^(3/2)
              + g_i (S_m,i^2 / (eta_i + 3 zeta))^(3/4))]^(-1/2)     ARG00 eq. 6
    u_i    = 2 ln(S_m,i / S_max) / (3 sqrt(2) ln sigma_i)
    number fraction  0.5 erfc(u_i)                                  ARG00 eq. 13
    surface fraction 0.5 erfc(u_i - sqrt(2) ln sigma_i)
    mass fraction    0.5 erfc(u_i - 3/sqrt(2) ln sigma_i)           ARG00 eq. 14

with e_s from Bolton 1980, D_v and k_a from Pruppacher and Klett 1997
(eqs. 13.3, 13.18a), p = rho R_d T, and r_g,i the number median radius from
the mode's number and volume.  Over the updraft distribution p(w) (mean
wbar, standard deviation sigw, bounds wmin..wmax) the outputs are Ghan et
al. 1997's: ``fn = int_{w>0} f(w) p(w) dw``, ``fluxn = int_{w>0} w f(w)
p(w) dw``, and ``flux_fullact = int_{w>0} w p(w) dw`` (likewise for the
surface and mass fractions).  ``wdiab``, a diabatic cooling expressed as an
updraft, adds to w in the supersaturation only.  A prescribed S_max
replaces ARG00 eq. 6.

Units: na m-3, volc m3 m-3, hygro is ARG's B (dimensionless), w m s-1,
tair K, rhoair kg m-3; supersaturations are fractions (0.01 = 1 %).
Sectional aerosol (``msectional != 0``) is not represented: it is refused.
"""

from __future__ import annotations

import math
import os

import numpy as np

#: The opt-in's case contract (woof_wrfchem_pnnl.activate): (size, type)
#: arrays of these extents; modal aerosol uses size index 0 of each type.
MAXD_ASIZE, MAXD_ATYPE = 8, 2
INPUTS = ("na", "volc", "dlo_sect", "dhi_sect", "sigman", "hygro")
OUTPUTS = ("fn", "fs", "fm", "fluxn", "fluxs", "fluxm")
#: Modes one kernel thread carries (ACT_MAXMODE in the kernel).
MAXMODE = 8
#: Simpson intervals over the updraft range (ACT_NQ) and the standard
#: deviation at or below which the distribution is one updraft (SIGW_DELTA).
NQ = 64
SIGW_DELTA = 1.0e-4

# Physical constants, SI (the kernel's #defines).
MW_WATER, MW_AIR, R_UNIV, R_DRY = 0.018015, 0.028966, 8.314, 287.04
RHO_WATER, L_VAP, CP_AIR, GRAV, SURF_TENSION = 1000.0, 2.501e6, 1004.6, 9.81, 0.076

#: Environment variable selecting the implementation: ``cleanroom`` (the
#: default) or ``pnnl`` (the opt-in's exact port; needs
#: WOOF_ENABLE_WRFCHEM_PNNL=1 and, for its erfc, WOOF_WRF_SOURCE).
IMPLEMENTATION_ENV = "WOOF_CHEM_ACTIVATION"
IMPLEMENTATIONS = ("cleanroom", "pnnl")


def implementation(name: str | None = None):
    """The module whose ``launch_activate(cases)`` serves activation.

    ``cleanroom`` is this module.  ``pnnl`` is the opt-in's exact port,
    loaded only through :func:`woof.optin.load` (which refuses, naming the
    licence, unless WOOF_ENABLE_WRFCHEM_PNNL=1)."""
    name = name or os.environ.get(IMPLEMENTATION_ENV, "cleanroom")
    if name not in IMPLEMENTATIONS:
        raise ValueError(f"{IMPLEMENTATION_ENV}={name!r}: one of {IMPLEMENTATIONS}")
    if name == "cleanroom":
        import sys
        return sys.modules[__name__]
    from woof import optin
    optin.load("wrfchem_pnnl")
    from woof_wrfchem_pnnl import activate  # noqa: PLC0415 (opt-in door)
    return activate


# ---------------------------------------------------------------------------
# Float64 host reference: the same equations and quadrature, for tests.
# ---------------------------------------------------------------------------

def _thermo(T, rho):
    p = rho * R_DRY * T
    es = 611.2 * math.exp(17.67 * (T - 273.15) / (T - 29.65))          # Bolton 1980
    dv = 0.211e-4 * (T / 273.15) ** 1.94 * (101325.0 / p)                # P&K 13.3
    ka = 1.0e-3 * (4.39 + 0.071 * T)                                     # P&K 13.18a
    A = 2.0 * SURF_TENSION * MW_WATER / (RHO_WATER * R_UNIV * T)
    alpha = GRAV * MW_WATER * L_VAP / (CP_AIR * R_UNIV * T * T) - GRAV * MW_AIR / (R_UNIV * T)
    gamma = R_UNIV * T / (es * MW_WATER) + MW_WATER * L_VAP ** 2 / (CP_AIR * p * MW_AIR * T)
    ginv = (RHO_WATER * R_UNIV * T / (es * dv * MW_WATER)
            + L_VAP * RHO_WATER / (ka * T) * (L_VAP * MW_WATER / (R_UNIV * T) - 1.0))
    return A, alpha * ginv, gamma


def _smax(th, w, modes):
    A, aw_g, gamma = th
    awg = aw_g * w
    zeta = 2.0 * A / 3.0 * math.sqrt(awg)
    total = 0.0
    for n, sm, lnsg in modes:
        if n <= 0 or sm <= 0:
            continue
        eta = awg ** 1.5 / (2.0 * math.pi * RHO_WATER * gamma * n)
        f = 0.5 * math.exp(2.5 * lnsg * lnsg)
        g = 1.0 + 0.25 * lnsg
        total += (f * (zeta / eta) ** 1.5 + g * (sm * sm / (eta + 3.0 * zeta)) ** 0.75) / (sm * sm)
    return 1.0 / math.sqrt(total) if total > 0 else 0.0


def _fractions(th, weff, sgiven, modes):
    smx = sgiven if sgiven >= 0 else (_smax(th, weff, modes) if weff > 0 else 0.0)
    out = []
    for n, sm, lnsg in modes:
        if sm <= 0 or smx <= 0:
            out.append((0.0, 0.0, 0.0))
            continue
        u = 2.0 * math.log(sm / smx) / (3.0 * math.sqrt(2.0) * lnsg)
        out.append((0.5 * math.erfc(u), 0.5 * math.erfc(u - math.sqrt(2.0) * lnsg),
                    0.5 * math.erfc(u - 1.5 * math.sqrt(2.0) * lnsg)))
    return out


def reference_case(na, volc, sigman, hygro, *, wbar, sigw, tair, rhoair, wdiab=0.0,
                   wmin=-1e30, wmax=1e30, smax=None, nq=NQ):
    """One case in float64 on the host: ``(fractions (nmode, 6), flux_fullact,
    smax)`` with the columns fn, fs, fm, fluxn, fluxs, fluxm."""
    th = _thermo(tair, rhoair)
    A = th[0]
    modes = []
    for n, v, s, b in zip(na, volc, sigman, hygro):
        lnsg = math.log(max(s, 1.0001))
        if n > 0 and v > 0 and b > 0:
            rg = 0.5 * (6.0 * v / (math.pi * n)) ** (1.0 / 3.0) * math.exp(-1.5 * lnsg * lnsg)
            modes.append((n, 2.0 / math.sqrt(b) * (A / (3.0 * rg)) ** 1.5, lnsg))
        else:
            modes.append((0.0, 0.0, lnsg))
    sgiven = -1.0 if smax is None else float(smax)
    nm = len(modes)
    acc = np.zeros((nm, 3))
    fl = np.zeros((nm, 3))
    full = 0.0
    if sigw <= SIGW_DELTA:
        if wbar > 0:
            f = np.array(_fractions(th, wbar + wdiab, sgiven, modes))
            acc, fl, full = f, wbar * f, wbar
    else:
        lo = max(wmin, 0.0, wbar - 8.0 * sigw)
        hi = min(wmax, wbar + 8.0 * sigw)
        if hi > lo:
            h = (hi - lo) / nq
            norm = 1.0 / (math.sqrt(2.0 * math.pi) * sigw)
            for j in range(nq + 1):
                w = lo + h * j
                wt = 1.0 if j in (0, nq) else (4.0 if j % 2 else 2.0)
                q = wt * h / 3.0 * norm * math.exp(-0.5 * ((w - wbar) / sigw) ** 2)
                full += q * w
                if w > 0:
                    f = np.array(_fractions(th, w + wdiab, sgiven, modes))
                    acc += q * f
                    fl += q * w * f
    if sgiven >= 0:
        s_out = sgiven
    else:
        s_out = _smax(th, wbar + wdiab, modes) if wbar + wdiab > 0 else 0.0
    return np.concatenate([acc, fl], axis=1), full, s_out


# ---------------------------------------------------------------------------
# Device launchers.
# ---------------------------------------------------------------------------

def launch_activation(na, volc, sigman, hygro, *, wbar, sigw, tair, rhoair,
                      wdiab=None, wmin=None, wmax=None, smax=None):
    """Activation of ``nmode`` modes in ``ncase`` independent cases.

    na, volc, sigman, hygro: float32 device arrays (nmode, ncase); wbar,
    sigw, tair, rhoair and the optional wdiab (default 0), wmin (default
    -inf), wmax (default +inf) and smax (prescribed maximum supersaturation;
    negative or None: ARG00) are float32 (ncase,).  Returns a dict of device
    arrays: fn, fs, fm, fluxn, fluxs, fluxm (nmode, ncase), flux_fullact and
    smax (ncase,)."""
    import cupy as cp
    from woof.core.kernels import get_kernel

    planes = [cp.ascontiguousarray(a, dtype=cp.float32) for a in (na, volc, sigman, hygro)]
    if planes[0].ndim != 2 or any(p.shape != planes[0].shape for p in planes):
        raise ValueError("na, volc, sigman and hygro must share one (nmode, ncase) shape")
    nmode, ncase = planes[0].shape
    if not 1 <= nmode <= MAXMODE:
        raise ValueError(f"1 to {MAXMODE} modes, got {nmode}")

    def per_case(a, default, label):
        if a is None:
            return cp.full(ncase, default, dtype=cp.float32)
        a = cp.ascontiguousarray(cp.broadcast_to(cp.asarray(a, dtype=cp.float32), (ncase,)))
        return a

    cases = [per_case(a, d, l) for a, d, l in (
        (wbar, None, "wbar"), (sigw, None, "sigw"), (wdiab, 0.0, "wdiab"),
        (wmin, -3.0e38, "wmin"), (wmax, 3.0e38, "wmax"), (tair, None, "tair"),
        (rhoair, None, "rhoair"), (smax, -1.0, "smax"))]
    out = {k: cp.zeros((nmode, ncase), dtype=cp.float32) for k in OUTPUTS}
    full = cp.zeros(ncase, dtype=cp.float32)
    smax_out = cp.zeros(ncase, dtype=cp.float32)
    threads = 64
    get_kernel("chem_activation", "chem_activation_modal")(
        ((ncase + threads - 1) // threads,), (threads,),
        (*planes, *cases, *(out[k] for k in OUTPUTS), full, smax_out,
         np.int32(nmode), np.int64(ncase)))
    out["flux_fullact"] = full
    out["smax"] = smax_out
    return out


def _check_case(c):
    if int(c.get("msectional", 0)) != 0:
        raise ValueError("the clean-room activation represents modal aerosol only "
                         "(msectional = 0); sectional activation is the opt-in's")
    ntype = int(c["ntype_aer"])
    if not 1 <= ntype <= MAXD_ATYPE:
        raise ValueError(f"ntype_aer must be 1..{MAXD_ATYPE}, got {ntype}")
    nsize = np.asarray(c["nsize_aer"]).reshape(-1)[:ntype]
    if any(int(s) != 1 for s in nsize):
        raise ValueError("modal aerosol has one size per type (nsize_aer = 1)")
    return ntype


def launch_activate(cases):
    """The opt-in's case contract (``woof_wrfchem_pnnl.activate.launch_activate``)
    on the clean-room kernel: each case a dict with wbar, sigw, wdiab, wminf,
    wmaxf, tair, rhoair, msectional (0), ntype_aer, nsize_aer, smax (None =
    not prescribed) and (8, 2) arrays :data:`INPUTS` (index [0, type]).
    :data:`OUTPUTS` and flux_fullact start from the case's values where
    given, and every type's size-0 entry is written.  Returns
    ``{name: (ncase, 8, 2) float32}`` plus ``flux_fullact`` (ncase,)."""
    import cupy as cp

    ntypes = [_check_case(c) for c in cases]
    nc = len(cases)
    nmode = max(ntypes) if ntypes else 1
    planes = {k: np.zeros((nmode, nc), np.float32) for k in ("na", "volc", "sigman", "hygro")}
    scal = {k: np.zeros(nc, np.float32) for k in
            ("wbar", "sigw", "wdiab", "wminf", "wmaxf", "tair", "rhoair", "smax")}
    for i, (c, nt) in enumerate(zip(cases, ntypes)):
        for k in planes:
            planes[k][:nt, i] = np.asarray(c[k], np.float32)[0, :nt]
        for k in scal:
            if k == "smax":
                scal[k][i] = -1.0 if c.get("smax") is None else c["smax"]
            else:
                scal[k][i] = c[k]
    dev = launch_activation(
        *(cp.asarray(planes[k]) for k in ("na", "volc", "sigman", "hygro")),
        wbar=cp.asarray(scal["wbar"]), sigw=cp.asarray(scal["sigw"]),
        tair=cp.asarray(scal["tair"]), rhoair=cp.asarray(scal["rhoair"]),
        wdiab=cp.asarray(scal["wdiab"]), wmin=cp.asarray(scal["wminf"]),
        wmax=cp.asarray(scal["wmaxf"]), smax=cp.asarray(scal["smax"]))
    result = {}
    for k in OUTPUTS:
        host = cp.asnumpy(dev[k])
        arr = np.zeros((nc, MAXD_ASIZE, MAXD_ATYPE), np.float32)
        for i, (c, nt) in enumerate(zip(cases, ntypes)):
            if k in c and c[k] is not None:
                arr[i] = np.asarray(c[k], np.float32)
            arr[i, 0, :nt] = host[:nt, i]
        result[k] = arr
    result["flux_fullact"] = cp.asnumpy(dev["flux_fullact"]).astype(np.float32)
    return result
