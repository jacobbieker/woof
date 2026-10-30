"""Nakajima & Tanaka (1988) IMS secondary-scattering correction of radiances.

Clean-room from the paper (JQSRT 40, 51-69, section 2.4, eqs. 13, 20, 23,
24).  After the TMS correction (exact single scattering, done in the
solver kernel), the remaining error near the solar direction comes from
light scattered twice through the forward peak that the delta-M method
replaced by a delta function.  For an optically averaged homogeneous
atmosphere above the output level the paper's eq. (23) gives the deviation

    du = -(F0 / 4 pi) (1 - f w) w_hat^2 [2 P_hat(cos T) - P_hat2(cos T)] h,
    w_hat = f w / (1 - f w),

with the forward-peak phase function P_hat = sum (2n+1) g_hat_n P_n,
g_hat_n = 1 for n < 2N (the delta-M truncation order) and g_n / f beyond,
its secondary-scattering counterpart P_hat2 = sum (2n+1) g_hat_n^2 P_n
(eq. 13), and the geometrical factor

    h = 1/(mu mu0*) int_0^tau t exp(-t/mu0*) exp((t - tau)/mu) dt,
    mu0* = mu0 / (1 - f w),

for downward radiance (cosine mu > 0 of the polar angle from the nadir)
at optical depth tau below the top.

The averages and the unscaled optical depth in h were fixed by holding
the correction to uvspec 2.0.6's ``disort_intcor moments`` radiances
(probed as a black box, docs/dev/radtran.md): tau = sum of the layer
optical depths above the level, w = sum(w dtau)/tau, f = sum(f w dtau) /
sum(w dtau), and g_hat_n averaged with weights f w dtau.  uvspec applies
the correction only to downward directions whose polar angle is within
:data:`WINDOW_DEG` of the solar zenith angle (probed: corrected at 9.5 and
9.6 deg, not at 10.5 deg, for solar zenith angles 30 and 60 deg).
"""

from __future__ import annotations

import math

import numpy as np

#: uvspec corrects downward radiances within this polar angle of the sun (probed).
WINDOW_DEG = 10.0


def _in_window(mu, umu0):
    return np.abs(np.degrees(np.arccos(np.clip(mu, -1, 1))) - np.degrees(np.arccos(np.clip(umu0, -1, 1)))) < WINDOW_DEG


def h_factor(tau, mu, mu0s):
    """h(tau, mu, mu0*, mu0*) of NT88 eq. (24), closed form (vectorised)."""
    tau = np.asarray(tau, dtype=np.float64)
    a = 1.0 / mu - 1.0 / mu0s
    x = a * tau
    small = np.abs(x) < 1e-3
    xs = np.where(small, 1.0, x)
    # int_0^tau t e^{a t} dt = tau^2 [e^x (x - 1) + 1] / x^2 ; series for small x
    big = (np.exp(xs) * (xs - 1.0) + 1.0) / (xs * xs)
    ser = 0.5 + x / 3.0 + x * x / 8.0 + x ** 3 / 30.0
    integral = tau * tau * np.where(small, ser, big)
    return integral * np.exp(-tau / mu) / (mu * mu0s)


def series_length(components, nstr, tol=1e-10, cap=3000) -> int:
    """Highest Legendre order kept: where every Henyey-Greenstein g^n < tol."""
    nmax = nstr
    for kind, ts, g in components:
        if kind == "hg":
            gm = float(np.max(np.where(ts > 0, np.abs(g), 0.0), initial=0.0))
            if gm > 0.0:
                nmax = max(nmax, int(math.ceil(math.log(tol) / math.log(min(gm, 1 - 1e-6)))))
        else:
            nmax = max(nmax, g.shape[-1] - 1)
    return min(nmax, cap)


def _peak_moments(components, lev, nmax):
    """sum over layers above ``lev`` and components of tau_sca * g_n, n = 0..nmax."""
    S = 0.0
    n = np.arange(nmax + 1)
    for kind, ts, g in components:
        t = ts[:, :lev]
        if not np.any(t > 0):
            continue
        if kind == "hg":
            gl = g[:, :lev]
            S = S + np.einsum("pl,pln->pn", t, np.power(gl[..., None], n))
        else:                                   # tabulated moments (zero beyond the table)
            m = min(nmax, g.shape[-1] - 1)
            add = np.zeros((t.shape[0], nmax + 1))
            add[:, :m + 1] = np.einsum("pl,pln->pn", t, g[:, :lev, :m + 1])
            S = S + add
    return S


def correction(*, dtau, ssa, f, components, nstr, umu0, umu, cosang, outlev, fbeam=1.0):
    """IMS radiance deviation (nprob, nout, numu, nphi) to add to TMS radiances.

    dtau, ssa, f: (nprob, L) unscaled layer optical depth, single-scattering
    albedo and delta-M fraction of the mixture; ``components``: the
    scatterers with forward peaks, ("hg", tau_sca, g) or ("mom", tau_sca,
    chi[..., n]) per layer; cosang: (nprob, numu, nphi) scattering-angle
    cosines; umu: user polar cosines (< 0 downward, uvspec's sign).
    """
    from woof.radtran.mie import legendre_table
    dtau = np.asarray(dtau, dtype=np.float64)
    nprob, L = dtau.shape
    umu = np.asarray(umu, dtype=np.float64)
    umu0 = np.broadcast_to(np.asarray(umu0, dtype=np.float64), (nprob,))
    numu, nphi = cosang.shape[1], cosang.shape[2]
    out = np.zeros((nprob, len(outlev), numu, nphi))
    down = [iu for iu, u in enumerate(umu) if u < 0.0 and np.any(_in_window(-u, umu0))]
    if not down or not components:
        return out
    nmax = series_length(components, nstr)
    n = np.arange(nmax + 1)
    sw = ssa * dtau
    wf = f * sw
    P = legendre_table(nmax, cosang[:, down, :].reshape(-1)).reshape(nmax + 1, nprob, len(down), nphi)
    for o, lev in enumerate(outlev):
        if lev == 0:
            continue
        tau = dtau[:, :lev].sum(axis=1)
        swt = sw[:, :lev].sum(axis=1)
        wft = wf[:, :lev].sum(axis=1)
        live = wft > 0
        if not np.any(live):
            continue
        safe = np.where(live, wft, 1.0)
        S = _peak_moments(components, lev, nmax)
        gh = np.where(n[None, :] < nstr, 1.0, S / safe[:, None])        # forward-peak moments
        wbar = np.where(live, swt / tau, 0.0)
        fw = np.where(live, wft / np.where(swt > 0, swt, 1.0), 0.0) * wbar
        what = fw / (1.0 - fw)
        mu0s = umu0 / (1.0 - fw)
        c1 = ((2 * n + 1)[None, :] * gh).T                              # (nmax+1, nprob)
        c2 = ((2 * n + 1)[None, :] * gh * gh).T
        Ph = np.einsum("np,npuv->puv", c1, P)
        Ph2 = np.einsum("np,npuv->puv", c2, P)
        for k, iu in enumerate(down):
            mu = -umu[iu]
            h = h_factor(tau, mu, mu0s)
            d = -(fbeam / (4.0 * math.pi)) * (1.0 - fw) * what * what * h
            d = np.where(live & _in_window(mu, umu0), d, 0.0)
            out[:, o, iu, :] = d[:, None] * (2.0 * Ph[:, k, :] - Ph2[:, k, :])
    return out
