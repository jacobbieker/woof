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
    PI = np.zeros((mu.size, N))
    TAU = np.zeros((mu.size, N))
    pi0 = np.zeros_like(mu)
    pi1 = np.ones_like(mu)
    for n in range(1, N + 1):
        if n == 1:
            pin = pi1
        else:
            pin = ((2.0 * n - 1.0) / (n - 1.0)) * mu * pi1 - (n / (n - 1.0)) * pi0
            pi0, pi1 = pi1, pin
        tau = n * mu * pin - (n + 1.0) * pi0 if n > 1 else mu * pin
        f = (2.0 * n + 1.0) / (n * (n + 1.0))
        PI[:, n - 1] = f * pin
        TAU[:, n - 1] = f * tau
    # sum over n as matrix products: S1 = sum_n f_n (a_n pi_n + b_n tau_n), S2 likewise swapped
    S1 = PI @ a + TAU @ b
    S2 = TAU @ a + PI @ b
    return S1, S2


def legendre_table(lmax, mu):
    P = np.zeros((lmax + 1, mu.size))
    P[0] = 1.0
    if lmax >= 1:
        P[1] = mu
    for l in range(2, lmax + 1):
        P[l] = ((2 * l - 1) * mu * P[l - 1] - (l - 1) * P[l - 2]) / l
    return P


# ---------------------------------------------------------------------------
# Exact Legendre moments of the size-integrated phase function
#
# With A_n = (2n+1)(a_n + b_n) and B_n = (2n+1)(a_n - b_n) the amplitude
# functions are S2 + S1 = sum_n A_n d^n_{11}(theta) and S2 - S1 =
# sum_n B_n d^n_{1,-1}(theta) (Wigner d-functions; pi_n + tau_n =
# n(n+1) d^n_{11}, pi_n - tau_n = n(n+1) d^n_{1,-1}).  The Clebsch-Gordan
# series of a product of d-functions (Edmonds 1957, eq. 4.3.2) gives
#
#   |S1|^2 + |S2|^2 = sum_l P_l(cos theta) c_l,
#   c_l = 1/2 sum_{n,m} G(n,m,l) [Re A_n A_m* + (-1)^(n+m+l) Re B_n B_m*],
#   G(n,m,l) = <n 1, m -1 | l 0>^2 = (2l+1) (n m l; 1 -1 0)^2,
#
# so the phase-function moments chi_l = c_l / ((2l+1) c_0) follow exactly
# from size-integrated products A_n A_m* with |n - m| <= l: no angular
# quadrature, however peaked the forward lobe.  The 3j symbols come from
# the three-term recursion in j3 (Schulten & Gordon 1975, J. Math. Phys.
# 16, 1961), started at j3 = |n - m| from the closed form for the stretched
# case j2 = j1 + j3 (Edmonds eq. 3.7.? / Messiah C.21) and, for n = m, from
# (n n 0; 1 -1 0)^2 = 1/(2n+1) and (n n 1; 1 -1 0)^2 = 1/(n(n+1)(2n+1)).
# ---------------------------------------------------------------------------

def _lgamma(v):
    import math
    return np.vectorize(math.lgamma, otypes=[np.float64])(v)


def cg_squares(nmax: int, lmax: int):
    """G[d] (shape (lmax+1, nmax-d)): <n 1, n+d -1 | l 0>^2 for n = 1..nmax-d.

    Entries with l < d are zero.  d runs 0..lmax.
    """
    out = []
    for d in range(lmax + 1):
        n = np.arange(1, nmax - d + 1, dtype=np.float64)
        G = np.zeros((lmax + 1, n.size))
        if n.size == 0:
            out.append(G)
            continue
        f = np.zeros((lmax + 2, n.size))
        if d == 0:
            f[0] = 1.0 / np.sqrt(2.0 * n + 1.0)
            f[1] = 1.0 / np.sqrt(n * (n + 1.0) * (2.0 * n + 1.0))
            j0 = 1
        else:
            lf = (_lgamma(2 * n + 1) + _lgamma(2 * d + 1) + _lgamma(n + d + 2) + _lgamma(n + d)
                  - _lgamma(2 * n + 2 * d + 2) - _lgamma(n + 2) - _lgamma(n) - 2.0 * _lgamma(d + 1))
            f[d] = np.exp(0.5 * lf)
            j0 = d
        s = 2.0 * n + d + 1.0                       # j1 + j2 + 1

        def A(j):
            return j * np.sqrt(np.maximum(j * j - d * d, 0.0) * np.maximum(s * s - j * j, 0.0))

        for j in range(j0, lmax):
            B = -2.0 * (2 * j + 1) * j * (j + 1)
            prev = f[j - 1] if j >= 1 else 0.0
            live = j + 1 <= 2.0 * n + d                 # the 3j vanishes for j3 > j1 + j2
            den = np.where(live, j * A(float(j + 1)), 1.0)
            f[j + 1] = np.where(live, -(B * f[j] + (j + 1) * A(float(j)) * prev) / den, 0.0)
        for l in range(d, lmax + 1):
            G[l] = (2 * l + 1) * f[l] ** 2
        out.append(G)
    return out




class _MomentSums:
    """Accumulates sum_r w_r A_n(r) A_{n+d}(r)* (and B) for d = 0..lmax."""

    def __init__(self, lmax):
        self.lmax = lmax
        self.pa = [np.zeros(0) for _ in range(lmax + 1)]
        self.pb = [np.zeros(0) for _ in range(lmax + 1)]

    def add(self, a, b, w):
        N = a.shape[0]
        n = np.arange(1, N + 1)[:, None]
        A = (2 * n + 1) * (a + b)
        B = (2 * n + 1) * (a - b)
        w = np.asarray(w, dtype=np.float64)[None, :]
        for d in range(self.lmax + 1):
            if N - d <= 0:
                break
            pa = np.sum(w * (A[:N - d] * np.conj(A[d:])).real, axis=1)
            pb = np.sum(w * (B[:N - d] * np.conj(B[d:])).real, axis=1)
            for lst, v in ((self.pa, pa), (self.pb, pb)):
                if lst[d].size < v.size:
                    lst[d] = np.concatenate((lst[d], np.zeros(v.size - lst[d].size)))
                lst[d][:v.size] += v

    def moments(self):
        lmax = self.lmax
        N = self.pa[0].size
        G = cg_squares(N, lmax)
        c = np.zeros(lmax + 1)
        for d in range(lmax + 1):
            pa, pb = self.pa[d], self.pb[d]
            if pa.size == 0:
                break
            k = 1.0 if d == 0 else 2.0                  # (n, m) and (m, n)
            for l in range(d, lmax + 1):
                sgn = -1.0 if (d + l) % 2 else 1.0
                c[l] += 0.5 * k * np.sum(G[d][l][:pa.size] * (pa + sgn * pb))
        return c / (c[0] * (2 * np.arange(lmax + 1) + 1))


def moments_from_coefficients(a, b, weights, lmax: int):
    """Normalised Legendre moments chi_0..chi_lmax of sum_r w_r (|S1|^2+|S2|^2).

    ``a``, ``b``: (N, nr) Mie coefficients (zero beyond each radius's
    truncation); ``weights``: (nr,) number weights.  Exact (see above).
    """
    acc = _MomentSums(lmax)
    acc.add(a, b, weights)
    return acc.moments()


def size_grid(lam_um, r_min, r_max, *, dlnr=0.005, dx=0.05):
    """Radius nodes [um] between r_min and r_max: steps of ``dlnr`` in ln r
    while the size-parameter step stays below ``dx``, then steps of ``dx`` in
    size parameter x = 2 pi r / lambda (resolving the interference structure
    of large particles)."""
    k = 2.0 * np.pi / lam_um
    x0, x1 = k * r_min, k * r_max
    xc = dx / dlnr                                   # x where both steps are equal
    parts = []
    if x0 < xc:
        hi = min(xc, x1)
        n = max(2, int(np.ceil(np.log(hi / x0) / dlnr)) + 1)
        parts.append(np.exp(np.linspace(np.log(x0), np.log(hi), n)))
    if x1 > xc:
        lo = max(xc, x0)
        n = max(2, int(np.ceil((x1 - lo) / dx)) + 1)
        seg = np.linspace(lo, x1, n)
        parts.append(seg[1:] if parts else seg)
    return np.concatenate(parts) / k


def lognormal_bulk(lam_um, m, r_mod, sigma, r_min, r_max, density, *, lmax=32, dlnr=0.005, dx=0.05,
                   chunk=1024, angles_cos=None):
    """Bulk optics of a lognormal number distribution between r_min and r_max [um].

    n(ln r) ~ exp(-(ln r - ln r_mod)^2 / (2 ln^2 sigma)), integrated by the
    trapezoid rule in ln r on :func:`size_grid`.  Returns ext [km^-1 per
    g m^-3 of particle mass at ``density`` g cm^-3], ssa, chi (Legendre
    moments 0..lmax, exact for the discretised distribution) and, given
    ``angles_cos``, the phase function p (normalised to (1/2) int p dmu = 1)
    at those cosines.
    """
    r = size_grid(lam_um, r_min, r_max, dlnr=dlnr, dx=dx)
    lnr = np.log(r)
    h = np.diff(lnr)
    tw = np.zeros(r.size)
    tw[:-1] += 0.5 * h
    tw[1:] += 0.5 * h
    w = np.exp(-0.5 * ((lnr - np.log(r_mod)) / np.log(sigma)) ** 2) * tw
    x = 2.0 * np.pi * r / lam_um
    area = np.pi * r ** 2
    cext = csca = 0.0
    acc = _MomentSums(lmax)
    mu = None if angles_cos is None else np.asarray(angles_cos, dtype=np.float64)
    s_ang = 0.0
    norm = 0.0
    for i0 in range(0, r.size, chunk):
        sl = slice(i0, min(r.size, i0 + chunk))
        qext, qsca, a, b = efficiencies(x[sl], m)
        cext += np.sum(w[sl] * qext * area[sl])
        csca += np.sum(w[sl] * qsca * area[sl])
        acc.add(a, b, w[sl])
        if mu is not None:
            S1, S2 = amplitudes(a, b, mu)
            s_ang = s_ang + (0.5 * (np.abs(S1) ** 2 + np.abs(S2) ** 2) * w[None, sl]).sum(axis=1)
            n = np.arange(1, a.shape[0] + 1)[:, None]
            norm += np.sum(w[None, sl] * (2 * n + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2))
    mass = np.sum(w * density * 4.0 / 3.0 * np.pi * r ** 3)
    out = {"ext": cext / mass * 1e3, "ssa": csca / cext, "chi": acc.moments(), "nr": r.size}
    out["g"] = out["chi"][1]
    if mu is not None:
        # (1/2) int p dmu = 1: p = 2 s / sum_n (2n+1)(|a_n|^2 + |b_n|^2), size-weighted
        out["phase"] = 2.0 * s_ang / norm
    return out
