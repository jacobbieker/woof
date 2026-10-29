"""NumPy reference of woof's discrete-ordinate solver (CPU, float64).

This is the readable statement of the algorithm that the GPU kernel
(woof/core/kernels/radtran_disort.cu) implements; tests hold the kernel to it
and both to libRadtran's uvspec, which only ever runs as a black-box oracle.

Clean-room derivation, following the published method only:

* Stamnes, Tsay, Wiscombe & Jayaweera (1988), Appl. Opt. 27, 2502: the
  azimuthal Fourier decomposition, double-Gauss quadrature, the homogeneous
  solution by the eigenvalue reduction of Stamnes & Swanson (1981), the beam
  particular solution, scaled exponentials in the boundary-value problem, and
  intensities at user angles by integrating the source function.
* Wiscombe (1977), J. Atmos. Sci. 34, 1408: delta-M scaling.
* Dahlback & Stamnes (1991), Planet. Space Sci. 39, 671: the
  pseudo-spherical beam (slant optical depth through spherical shells, a
  layer-average secant in the particular solution).
* Nakajima & Tanaka (1988), JQSRT 40, 51: the TMS single-scattering
  correction of radiances after delta-M.

Conventions: optical depth tau grows downward from the top; mu > 0 is
upward-travelling radiation; the beam travels downward with cosine umu0;
the Fourier series is I = sum_m I^m cos(m (phi - phi0)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DisortResult:
    edir: np.ndarray      # (nlev,) direct (delta-M scaled) horizontal irradiance
    edn: np.ndarray       # (nlev,) diffuse downward irradiance
    eup: np.ndarray       # (nlev,) diffuse upward irradiance
    uavg: np.ndarray      # (nlev,) diffuse mean intensity
    uu: np.ndarray        # (nlev, numu, nphi) radiance


def double_gauss(n: int):
    """Gauss-Legendre nodes and weights on (0, 1), ascending."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def norm_legendre(m: int, lmax: int, mu) -> np.ndarray:
    """Lambda_l^m(mu) = sqrt((l-m)!/(l+m)!) P_l^m(mu), l = 0..lmax (zero for l < m)."""
    mu = np.atleast_1d(np.asarray(mu, dtype=np.float64))
    out = np.zeros((lmax + 1, mu.size))
    s = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    pmm = np.ones_like(mu)
    for k in range(1, m + 1):
        pmm = pmm * math.sqrt((2.0 * k - 1.0) / (2.0 * k)) * s
    if m > lmax:
        return out
    out[m] = pmm
    if m + 1 <= lmax:
        out[m + 1] = math.sqrt(2.0 * m + 1.0) * mu * pmm
    for l in range(m + 2, lmax + 1):
        out[l] = ((2.0 * l - 1.0) * mu * out[l - 1]
                  - math.sqrt((l - 1.0) ** 2 - m * m) * out[l - 2]) / math.sqrt(l * l - m * m)
    return out


def chapman_slant(z_km, dtau, umu0, radius_km):
    """Slant optical depth from TOA to each level along the sun's path.

    Spherical shells between the levels ``z_km`` (top first, nlev = nlyr+1),
    each layer homogeneous with optical depth ``dtau``; the local solar zenith
    angle at every level is acos(umu0) (Dahlback & Stamnes 1991).
    """
    z = np.asarray(z_km, dtype=np.float64)
    nlyr = dtau.size
    r = radius_km + z
    sin2 = 1.0 - umu0 * umu0
    ch = np.zeros(nlyr + 1)
    for k in range(1, nlyr + 1):
        rk = r[k]
        p2 = rk * rk * sin2
        tot = 0.0
        if umu0 >= 0.0:
            for j in range(k):          # layers above level k: j = 0..k-1
                ds = math.sqrt(r[j] ** 2 - p2) - math.sqrt(r[j + 1] ** 2 - p2)
                tot += dtau[j] * ds / (r[j] - r[j + 1])
        else:
            raise NotImplementedError("sun below the horizon")
        ch[k] = tot
    return ch


def solve(dtau, ssa, pmom, *, umu0, fbeam=1.0, albedo=0.0, nstr=8,
          umu=(), phi=(), deltam=True, z_km=None, radius_km=6370.0,
          intensity_correction=True) -> DisortResult:
    """Fluxes at every level and radiances at (umu, phi = phi - phi0).

    ``pmom[n, l]`` are the Legendre moments of layer n's phase function
    (pmom[:, 0] == 1).  ``z_km`` (top first) switches on the pseudo-spherical
    beam.
    """
    dtau = np.asarray(dtau, dtype=np.float64)
    ssa = np.asarray(ssa, dtype=np.float64).copy()
    pmom = np.asarray(pmom, dtype=np.float64)
    nlyr = dtau.size
    N = nstr // 2
    nmom = pmom.shape[1] - 1
    umu = np.asarray(umu, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)

    # delta-M (Wiscombe 1977): truncate at moment 2N.
    fz = np.zeros(nlyr)
    if deltam and nmom >= nstr:
        fz = pmom[:, nstr].copy()
    g = np.zeros((nlyr, nstr))
    lm = min(nmom, nstr - 1)
    g[:, :lm + 1] = (pmom[:, :lm + 1] - fz[:, None]) / (1.0 - fz[:, None])
    dtaus = (1.0 - ssa * fz) * dtau
    ssas = ssa * (1.0 - fz) / (1.0 - ssa * fz)
    taus = np.concatenate(([0.0], np.cumsum(dtaus)))

    # Beam attenuation to each level and the secant used inside each layer.
    if z_km is None:
        ch = taus / umu0
    else:
        ch = chapman_slant(z_km, dtaus, umu0, radius_km)
    lam = np.where(dtaus > 0.0, (ch[1:] - ch[:-1]) / np.where(dtaus > 0, dtaus, 1.0), 1.0 / umu0)

    mu, w = double_gauss(N)
    nlev = nlyr + 1
    edir = umu0 * fbeam * np.exp(-ch)
    edn = np.zeros(nlev)
    eup = np.zeros(nlev)
    uavg = np.zeros(nlev)
    numu, nphi = umu.size, phi.size
    uu = np.zeros((nlev, numu, nphi))

    mmax = 0 if numu == 0 else nstr - 1
    for m in range(mmax + 1):
        lq = norm_legendre(m, nstr - 1, mu)          # (nstr, N)
        l0 = norm_legendre(m, nstr - 1, umu0)[:, 0]   # (nstr,)
        sgn = np.array([(-1.0) ** (l + m) for l in range(nstr)])
        coef = np.arange(nstr) * 2.0 + 1.0
        G = np.zeros((nlyr, 2 * N, N))   # rows: [+mu_i ; -mu_i]
        K = np.zeros((nlyr, N))
        Z = np.zeros((nlyr, 2 * N))
        for n in range(nlyr):
            gl = coef * g[n] * 0.5 * ssas[n]
            dp = (lq.T * gl) @ lq              # D(mu_i, mu_j)
            dm = (lq.T * (gl * sgn)) @ lq      # D(mu_i, -mu_j)
            a = (dp * w[None, :] - np.eye(N)) / mu[:, None]
            b = (dm * w[None, :]) / mu[:, None]
            apb, amb = a + b, a - b
            k2, v = np.linalg.eig(amb @ apb)
            k2 = k2.real
            v = v.real
            k = np.sqrt(k2)
            order = np.argsort(k)
            k, v = k[order], v[:, order]
            w2 = (apb @ v) / k[None, :]
            G[n, :N] = 0.5 * (v + w2)
            G[n, N:] = 0.5 * (v - w2)
            K[n] = k
            # beam particular solution over the 2N directions
            mus = np.concatenate((mu, -mu))
            lall = norm_legendre(m, nstr - 1, mus)            # (nstr, 2N)
            dfull = (lall.T * gl) @ lall                       # D(mu_i, mu_j) all 2N
            wfull = np.concatenate((w, w))
            amat = np.diag(1.0 + lam[n] * mus) - dfull * wfull[None, :]
            x0 = (ssas[n] * fbeam / (4.0 * math.pi) * (2.0 - (m == 0))
                  * (lall.T @ (coef * g[n] * sgn * l0)))
            Z[n] = np.linalg.solve(amat, x0)
        # Boundary-value problem; unknowns per layer: A_j (decaying from the
        # top), B_j (decaying from the bottom).
        nun = 2 * N * nlyr
        mat = np.zeros((nun, nun))
        rhs = np.zeros(nun)
        row = 0
        att = np.exp(-ch[:-1])           # beam factor at each layer top
        attb = np.exp(-ch[1:])
        for n in range(nlyr):
            e = np.exp(-K[n] * dtaus[n])
            Gp, Gm = G[n, :N], G[n, N:]
            # I+(t) = Gp A e^{-kt} + Gm B e^{-k(D-t)} ; I-(t) = Gm A e^{-kt} + Gp B e^{-k(D-t)}
            if n == 0:
                mat[row:row + N, 0:N] = Gm
                mat[row:row + N, N:2 * N] = Gp * e[None, :]
                rhs[row:row + N] = -Z[0, N:] * att[0]
                row += N
            if n < nlyr - 1:
                e1 = np.exp(-K[n + 1] * dtaus[n + 1])
                Gp1, Gm1 = G[n + 1, :N], G[n + 1, N:]
                c0 = 2 * N * n
                c1 = 2 * N * (n + 1)
                # upward rows then downward rows: layer n bottom == layer n+1 top
                mat[row:row + N, c0:c0 + N] = Gp * e[None, :]
                mat[row:row + N, c0 + N:c0 + 2 * N] = Gm
                mat[row:row + N, c1:c1 + N] = -Gp1
                mat[row:row + N, c1 + N:c1 + 2 * N] = -Gm1 * e1[None, :]
                rhs[row:row + N] = (Z[n + 1, :N] * att[n + 1] - Z[n, :N] * attb[n])
                row += N
                mat[row:row + N, c0:c0 + N] = Gm * e[None, :]
                mat[row:row + N, c0 + N:c0 + 2 * N] = Gp
                mat[row:row + N, c1:c1 + N] = -Gm1
                mat[row:row + N, c1 + N:c1 + 2 * N] = -Gp1 * e1[None, :]
                rhs[row:row + N] = (Z[n + 1, N:] * att[n + 1] - Z[n, N:] * attb[n])
                row += N
            else:
                c0 = 2 * N * n
                # I+(D) - 2A sum w mu I-(D) = A/pi umu0 F0 e^{-ch}  (m = 0 only)
                refl = (2.0 * albedo * (w * mu))[None, :] if m == 0 else np.zeros((1, N))
                mat[row:row + N, c0:c0 + N] = Gp * e[None, :] - (np.ones((N, 1)) * refl) @ (Gm * e[None, :])
                mat[row:row + N, c0 + N:c0 + 2 * N] = Gm - (np.ones((N, 1)) * refl) @ Gp
                zb_up = Z[n, :N] * attb[n]
                zb_dn = Z[n, N:] * attb[n]
                src = (albedo / math.pi * umu0 * fbeam * attb[n]) if m == 0 else 0.0
                rhs[row:row + N] = src - zb_up + (np.ones((N, 1)) * refl) @ zb_dn
                row += N
        sol = np.linalg.solve(mat, rhs)
        A = sol.reshape(nlyr, 2, N)[:, 0, :]
        B = sol.reshape(nlyr, 2, N)[:, 1, :]

        # Intensities at the quadrature angles at every level (layer tops,
        # plus the bottom of the last layer).
        Iq = np.zeros((nlev, 2 * N))
        for n in range(nlyr):
            e = np.exp(-K[n] * dtaus[n])
            Iq[n, :N] = G[n, :N] @ A[n] + G[n, N:] @ (B[n] * e) + Z[n, :N] * att[n]
            Iq[n, N:] = G[n, N:] @ A[n] + G[n, :N] @ (B[n] * e) + Z[n, N:] * att[n]
        n = nlyr - 1
        e = np.exp(-K[n] * dtaus[n])
        Iq[nlev - 1, :N] = G[n, :N] @ (A[n] * e) + G[n, N:] @ B[n] + Z[n, :N] * attb[n]
        Iq[nlev - 1, N:] = G[n, N:] @ (A[n] * e) + G[n, :N] @ B[n] + Z[n, N:] * attb[n]
        if m == 0:
            eup[:] = 2.0 * math.pi * (Iq[:, :N] @ (w * mu))
            edn[:] = 2.0 * math.pi * (Iq[:, N:] @ (w * mu))
            uavg[:] = 0.5 * (Iq[:, :N] @ w + Iq[:, N:] @ w)

        if numu:
            um = _user_intensity(m, umu, mu, w, nstr, g, ssas, dtaus, K, G, Z, A, B,
                                 att, attb, lam, umu0, fbeam, albedo, Iq, edir, sgn, coef, l0)
            cosm = np.cos(m * np.deg2rad(phi))
            uu += um[:, :, None] * cosm[None, None, :]

    if numu and intensity_correction:
        uu += _tms(umu, phi, umu0, fbeam, pmom, fz, ssas, dtaus, ch, lam, nstr, nmom)
    return DisortResult(edir=edir, edn=edn, eup=eup, uavg=uavg, uu=uu)


def _user_intensity(m, umu, mu, w, nstr, g, ssas, dtaus, K, G, Z, A, B,
                    att, attb, lam, umu0, fbeam, albedo, Iq, edir, sgn, coef, l0):
    """Source-function integration of the m-th Fourier intensity at umu."""
    nlyr = dtaus.size
    N = mu.size
    numu = umu.size
    out = np.zeros((nlyr + 1, numu))
    lu = norm_legendre(m, nstr - 1, umu)                  # (nstr, numu)
    lq = norm_legendre(m, nstr - 1, np.concatenate((mu, -mu)))
    wfull = np.concatenate((w, w))
    for iu in range(numu):
        u = umu[iu]
        # per-layer source coefficients at angle u
        srcA = np.zeros((nlyr, N))
        srcB = np.zeros((nlyr, N))
        srcZ = np.zeros(nlyr)
        for n in range(nlyr):
            gl = coef * g[n] * 0.5 * ssas[n]
            du = (lu[:, iu] * gl) @ lq                     # D(u, mu_j) over 2N
            Gfull_A = G[n]                                 # (2N, N) for e^{-kt}
            Gfull_B = np.concatenate((G[n, N:], G[n, :N]))  # for e^{-k(D-t)}
            srcA[n] = (du * wfull) @ Gfull_A * A[n]
            srcB[n] = (du * wfull) @ Gfull_B * B[n]
            x0 = (ssas[n] * fbeam / (4.0 * math.pi) * (2.0 - (m == 0))
                  * np.sum(coef * g[n] * sgn * l0 * lu[:, iu]))
            srcZ[n] = (du * wfull) @ Z[n] + x0
        if u > 0.0:
            # start at the surface
            if m == 0:
                fdn = 2.0 * math.pi * float(Iq[nlyr, N:] @ (w * mu))
                val = albedo / math.pi * (fdn + edir[nlyr])
            else:
                val = 0.0
            out[nlyr, iu] = val
            for n in range(nlyr - 1, -1, -1):
                d = dtaus[n]
                tr = math.exp(-d / u)
                s = 0.0
                for j in range(N):
                    k = K[n, j]
                    # A term: e^{-k t}; integral over t in [0,D] of e^{-kt} e^{-t/u} dt/u
                    s += srcA[n, j] * (1.0 - math.exp(-(k + 1.0 / u) * d)) / (1.0 + k * u)
                    # B term: e^{-k(D-t)}
                    s += srcB[n, j] * _grow(k, u, d)
                s += srcZ[n] * att[n] * (1.0 - math.exp(-(lam[n] + 1.0 / u) * d)) / (1.0 + lam[n] * u)
                val = val * tr + s
                out[n, iu] = val
        else:
            a = -u
            val = 0.0
            out[0, iu] = 0.0
            for n in range(nlyr):
                d = dtaus[n]
                tr = math.exp(-d / a)
                s = 0.0
                # radiance at bottom: int_0^D S(t) e^{-(D-t)/a} dt/a
                for j in range(N):
                    k = K[n, j]
                    s += srcA[n, j] * _grow(k, a, d)
                    s += srcB[n, j] * (1.0 - math.exp(-(k + 1.0 / a) * d)) / (1.0 + k * a)
                s += srcZ[n] * att[n] * _grow(lam[n], a, d)
                val = val * tr + s
                out[n + 1, iu] = val
    return out


def _grow(k, u, d):
    """int_0^d e^{-k (d - t)} e^{-t/u} dt / u  ==  int_0^d e^{-k t} e^{-(d-t)/u} dt/u."""
    x = k * u - 1.0
    if abs(x) < 1e-6 * max(1.0, k * u):
        return d / u * math.exp(-d / u)
    return (math.exp(-d / u) - math.exp(-k * d)) / x


def _tms(umu, phi, umu0, fbeam, pmom, fz, ssas, dtaus, ch, lam, nstr, nmom):
    """Nakajima-Tanaka TMS: swap the truncated single scattering for the exact one."""
    nlyr = dtaus.size
    nlev = nlyr + 1
    out = np.zeros((nlev, umu.size, phi.size))
    if not np.any(fz > 0.0):
        return out
    att = np.exp(-ch[:-1])
    for iu, u in enumerate(umu):
        for ip, ph in enumerate(phi):
            cosang = (-umu0 * u + math.sqrt(max(0.0, 1 - u * u)) * math.sqrt(max(0.0, 1 - umu0 * umu0))
                      * math.cos(math.radians(ph)))
            pl = norm_legendre(0, nmom, cosang)[:, 0]
            coef = np.arange(nmom + 1) * 2.0 + 1.0
            # exact and truncated phase functions per layer
            ptrue = (pmom * coef[None, :]) @ pl
            ptr = np.zeros(nlyr)
            for n in range(nlyr):
                gl = np.zeros(nmom + 1)
                lm = min(nmom, nstr - 1)
                gl[:lm + 1] = (pmom[n, :lm + 1] - fz[n]) / (1.0 - fz[n])
                ptr[n] = np.sum(coef[:nstr] * gl[:nstr] * pl[:nstr])
            dp = ptrue / (1.0 - fz) - ptr
            # single scattering with scaled optics and the corrected phase function
            src = ssas * fbeam / (4.0 * math.pi) * dp
            if u > 0.0:
                val = 0.0
                for n in range(nlyr - 1, -1, -1):
                    d = dtaus[n]
                    val = val * math.exp(-d / u) + src[n] * att[n] * (
                        1.0 - math.exp(-(lam[n] + 1.0 / u) * d)) / (1.0 + lam[n] * u)
                    out[n, iu, ip] = val
            else:
                a = -u
                val = 0.0
                for n in range(nlyr):
                    d = dtaus[n]
                    val = val * math.exp(-d / a) + src[n] * att[n] * _grow(lam[n], a, d)
                    out[n + 1, iu, ip] = val
    return out
