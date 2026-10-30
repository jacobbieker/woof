"""Mie scattering by homogeneous spheres, and lognormal size-distribution integrals.

Clean-room implementation from the published theory (Bohren & Huffman 1983,
"Absorption and Scattering of Light by Small Particles", ch. 4, eqs. 4.53,
4.56-4.57, 4.61-4.62, 4.74; Wiscombe 1980, Appl. Opt. 19, 1505, for the
series truncation N = x + 4 x^(1/3) + 2 and the downward recurrence of the
logarithmic derivative D_n(mx)).  Vectorised over size parameter.

Conventions: m = n_re + i n_im with n_im >= 0 for absorption (the OPAC
refractive-index files list the imaginary part with a negative sign; the
caller passes its magnitude).
"""

from __future__ import annotations

import numpy as np


def coefficients(x: np.ndarray, m: complex, nmax: int | None = None):
    """Mie coefficients a_n, b_n (n = 1..nmax) for size parameters x (1-D).

    Terms beyond a size parameter's own truncation N(x) are zero; the
    recurrences may overflow there (chi_n grows without bound for n >> x),
    which the mask discards.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _coefficients(x, m, nmax)


def _coefficients(x, m, nmax):
    x = np.asarray(x, dtype=np.float64)
    nstop = np.floor(x + 4.0 * np.cbrt(x) + 2.0).astype(int)
    N = int(nstop.max()) if nmax is None else int(nmax)
    mx = m * x
    nmx = int(max(N, np.abs(mx).max())) + 16
    # D_n(mx), downward recurrence from D_nmx = 0
    D = np.zeros((nmx + 1, x.size), dtype=np.complex128)
    for n in range(nmx, 0, -1):
        D[n - 1] = n / mx - 1.0 / (D[n] + n / mx)
    a = np.zeros((N, x.size), dtype=np.complex128)
    b = np.zeros((N, x.size), dtype=np.complex128)
    psi0, psi1 = np.cos(x), np.sin(x)          # psi_{-1}, psi_0
    chi0, chi1 = -np.sin(x), np.cos(x)         # chi_{-1}, chi_0
    xi1 = psi1 - 1j * chi1
    for n in range(1, N + 1):
        psi = (2.0 * n - 1.0) / x * psi1 - psi0
        chi = (2.0 * n - 1.0) / x * chi1 - chi0
        xi = psi - 1j * chi
        da = D[n] / m + n / x
        db = m * D[n] + n / x
        an = (da * psi - psi1) / (da * xi - xi1)
        bn = (db * psi - psi1) / (db * xi - xi1)
        live = n <= nstop
        a[n - 1] = np.where(live, an, 0.0)
        b[n - 1] = np.where(live, bn, 0.0)
        psi0, psi1 = psi1, psi
        chi0, chi1 = chi1, chi
        xi1 = psi1 - 1j * chi1
    return a, b


def efficiencies(x, m):
    """(Q_ext, Q_sca, a, b) for size parameters x."""
    a, b = coefficients(x, m)
    n = np.arange(1, a.shape[0] + 1)[:, None]
    x = np.asarray(x, dtype=np.float64)
    qext = 2.0 / x ** 2 * np.sum((2 * n + 1) * (a + b).real, axis=0)
    qsca = 2.0 / x ** 2 * np.sum((2 * n + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2), axis=0)
    return qext, qsca, a, b


def amplitudes(a, b, mu):
    """S1, S2 at scattering-angle cosines mu (Bohren & Huffman eq. 4.74)."""
    mu = np.asarray(mu, dtype=np.float64)
    N = a.shape[0]
    pi0 = np.zeros_like(mu)
    pi1 = np.ones_like(mu)
    S1 = np.zeros((mu.size, a.shape[1]), dtype=np.complex128)
    S2 = np.zeros_like(S1)
    for n in range(1, N + 1):
        if n == 1:
            pin = pi1
        else:
            pin = ((2.0 * n - 1.0) / (n - 1.0)) * mu * pi1 - (n / (n - 1.0)) * pi0
            pi0, pi1 = pi1, pin
        tau = n * mu * pin - (n + 1.0) * pi0 if n > 1 else mu * pin
        f = (2.0 * n + 1.0) / (n * (n + 1.0))
        S1 += f * (np.outer(pin, a[n - 1]) + np.outer(tau, b[n - 1]))
        S2 += f * (np.outer(tau, a[n - 1]) + np.outer(pin, b[n - 1]))
    return S1, S2


def legendre_table(lmax, mu):
    P = np.zeros((lmax + 1, mu.size))
    P[0] = 1.0
    if lmax >= 1:
        P[1] = mu
    for l in range(2, lmax + 1):
        P[l] = ((2 * l - 1) * mu * P[l - 1] - (l - 1) * P[l - 2]) / l
    return P


def lognormal_optics(lam_um, m, r_mod, sigma, r_min, r_max, density, *, nmom=32, nr=2000,
                     nmu=None, angles_cos=None):
    """Bulk optics of a lognormal number distribution between r_min and r_max.

    Returns dict with ext [km^-1 per g m^-3], ssa, g, pmom (l = 0..nmom),
    and the normalised phase function at ``angles_cos`` if given.  The
    integral runs over ln r with ``nr`` trapezoid nodes.
    """
    lnr = np.linspace(np.log(r_min), np.log(r_max), nr)
    r = np.exp(lnr)                                       # um
    ls = np.log(sigma)
    # number per ln r (unnormalised): exp(-(ln r - ln r_mod)^2 / (2 ln^2 sigma))
    tw = np.full(nr, lnr[1] - lnr[0])
    tw[0] *= 0.5
    tw[-1] *= 0.5
    w = np.exp(-0.5 * ((lnr - np.log(r_mod)) / ls) ** 2) * tw
    x = 2.0 * np.pi * r / lam_um
    qext, qsca, a, b = efficiencies(x, m)
    area = np.pi * r ** 2                                  # um^2
    cext = np.sum(w * qext * area)
    csca = np.sum(w * qsca * area)
    mass = np.sum(w * density * 4.0 / 3.0 * np.pi * r ** 3)   # g/cm3 * um^3
    # ext per mass: (um^2) / (g cm^-3 um^3) = 1/(um * g cm^-3) -> km^-1 per g m^-3
    # 1 / (1e-6 m * 1e6 g m^-3) = 1 m^-1 per g m^-3 = 1e3 km^-1 per g m^-3
    ext = cext / mass * 1e3
    out = {"ext": ext, "ssa": csca / cext}
    nmu = nmu or max(2 * nmom + 2, 400)
    xg, wg = np.polynomial.legendre.leggauss(nmu)
    S1, S2 = amplitudes(a, b, xg)
    s11 = 0.5 * (np.abs(S1) ** 2 + np.abs(S2) ** 2)          # (nmu, nr)
    p = (s11 * (w / x ** 2 * area)) .sum(axis=1)
    # normalise: (1/2) int p dmu = 1
    p = p / (0.5 * np.sum(wg * p))
    P = legendre_table(nmom, xg)
    out["pmom"] = 0.5 * (P * (wg * p)[None, :]).sum(axis=1)
    out["g"] = out["pmom"][1]
    if angles_cos is not None:
        A1, A2 = amplitudes(a, b, np.asarray(angles_cos, dtype=np.float64))
        s = 0.5 * (np.abs(A1) ** 2 + np.abs(A2) ** 2)
        pa = (s * (w / x ** 2 * area)).sum(axis=1)
        out["phase"] = pa / (0.5 * np.sum(wg * (s11 * (w / x ** 2 * area)).sum(axis=1)))
    return out
