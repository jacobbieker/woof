"""The radtran kernel against woof's NumPy reference and closed forms (no oracle).

* host kernel == woof/radtran/reference.py to ~1e-11 (same algorithm,
  different linear algebra: NumPy eig/solve vs Cholesky+Jacobi/banded LU);
* the two-stream kernel against the closed-form quadrature two-stream
  (Meador & Weaver 1980), evaluated in 50-digit decimal arithmetic;
* energy conservation for a conservative atmosphere over a black surface.
"""

from __future__ import annotations

import math
import shutil
from decimal import Decimal, getcontext

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not any(shutil.which(c) for c in ("clang++", "g++", "c++")),
                                reason="needs a host C++ compiler")


def _atmos(rng, L=20, cloud_tau=4.0, g=0.85, nmom=40):
    z = np.linspace(40.0, 0.0, L + 1)
    tr = 0.3 * np.exp(-z[1:] / 8.0) / L * 4
    ta = 0.01 * rng.random(L)
    tc = np.zeros(L)
    tc[-6:-3] = cloud_tau / 3
    tsca = tr + tc * 0.9999
    dtau = tr + ta + tc
    pm = np.zeros((L, nmom + 1))
    for l in range(nmom + 1):
        pm[:, l] = tc * 0.9999 * g ** l
    pm[:, 0] += tr
    pm[:, 2] += tr * 0.0956
    pm /= tsca[:, None]
    return z, dtau, tsca / dtau, pm


@pytest.mark.parametrize("nstr,sph", [(4, False), (8, False), (16, False), (8, True), (16, True)])
def test_kernel_matches_numpy_reference(nstr, sph):
    from woof.radtran import reference, solver
    rng = np.random.default_rng(nstr + 100 * sph)
    z, dtau, ssa, pm = _atmos(rng)
    umu = (-0.8, -0.3, 0.4, 0.9)
    phi = (0.0, 90.0, 170.0)
    mu0 = 0.6
    ref = reference.solve(dtau, ssa, pm[:, :nstr + 1], umu0=mu0, albedo=0.3, nstr=nstr, umu=umu, phi=phi,
                          z_km=z if sph else None, intensity_correction=False)
    flags = solver.DELTAM | (solver.SPHERICAL if sph else 0)
    L = dtau.size
    f, r = solver.solve(nstr=nstr, flags=flags, dtau=dtau[None], ssa=ssa[None], pmom=pm[None, :, :nstr + 1],
                        umu0=mu0, fbeam=1.0, albedo=0.3, zlev=z[None], outlev=list(range(L + 1)),
                        umu=umu, phi=phi, backend="host")
    # reference reports the scaled direct beam; compare totals and diffuse up
    glob_k = f[0, :, 0] + f[0, :, 1]
    glob_r = ref.edir + ref.edn
    assert np.allclose(glob_k, glob_r, rtol=1e-10, atol=1e-14)
    assert np.allclose(f[0, :, 2], ref.eup, rtol=1e-10, atol=1e-14)
    assert np.allclose(r[0], ref.uu, rtol=1e-9, atol=1e-13)


def _mw(tau, w, g, mu0):
    """Closed-form quadrature two-stream reflectance (Meador & Weaver 1980)."""
    getcontext().prec = 50
    D = lambda v: Decimal(repr(float(v)))
    tau, w, g, mu0 = map(D, (tau, w, g, mu0))
    s3 = Decimal(3).sqrt()
    g1 = s3 * (2 - w * (1 + g)) / 2
    g2 = s3 * w * (1 - g) / 2
    g3 = (1 - s3 * g * mu0) / 2
    lam = (g1 ** 2 - g2 ** 2).sqrt()
    a2 = g1 * g3 + g2 * (1 - g3)
    den = (1 - lam ** 2 * mu0 ** 2) * ((lam + g1) * (lam * tau).exp() + (lam - g1) * (-lam * tau).exp())
    R = w / den * ((1 - lam * mu0) * (a2 + lam * g3) * (lam * tau).exp()
                   - (1 + lam * mu0) * (a2 - lam * g3) * (-lam * tau).exp()
                   - 2 * lam * (g3 - a2 * mu0) * (-tau / mu0).exp())
    return float(R)


@pytest.mark.parametrize("tau,w,g", [(0.5, 0.9, 0.85), (5.0, 0.9999, 0.85), (50.0, 0.999997, 0.86),
                                     (50.0, 0.99999, 0.5)])
def test_twostream_matches_closed_form(tau, w, g):
    from woof.radtran import solver
    mu0 = math.cos(math.radians(30.0))
    f, _ = solver.solve(nstr=2, flags=solver.TWOSTREAM, dtau=np.array([[tau]]), ssa=np.array([[w]]),
                        pmom=np.array([[[1.0, g, g * g]]]), umu0=mu0, fbeam=1.0, albedo=0.0,
                        outlev=[0, 1], backend="host")
    R = f[0, 0, 2] / mu0
    assert abs(R / _mw(tau, w, g, mu0) - 1) < 1e-10


def test_conservative_atmosphere_conserves_energy():
    from woof.radtran import solver
    rng = np.random.default_rng(3)
    z, dtau, ssa, pm = _atmos(rng, cloud_tau=10.0)
    ssa = np.ones_like(ssa)
    mu0 = 0.7
    for nstr, flags in ((8, solver.DELTAM), (16, solver.DELTAM), (2, solver.TWOSTREAM | solver.DELTAM)):
        f, _ = solver.solve(nstr=nstr, flags=flags, dtau=dtau[None], ssa=ssa[None], pmom=pm[None, :, :nstr + 1],
                            umu0=mu0, fbeam=1.0, albedo=0.0, outlev=[0, dtau.size], backend="host")
        absorbed = mu0 - f[0, 0, 2] - (f[0, 1, 0] + f[0, 1, 1])
        assert abs(absorbed) < 1e-7 * mu0, (nstr, absorbed)
