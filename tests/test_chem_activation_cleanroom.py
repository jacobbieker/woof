"""The clean-room multimode activation (WOOF P8, woof/core/chem_activation.py).

Three layers:

1. The float64 host reference against physics it must obey: ARG00's
   single-mode limits and monotonicity, fn <= fs <= fm, a prescribed
   supersaturation, and Ghan et al. 1997's updraft integral against the
   closed-form Gaussian moment.
2. The device kernel against that reference (CUDA and Metal), to float32
   rounding.
3. The default build against WRF's exact routine, the ``wrfchem_pnnl``
   opt-in, run as a black box (WOOF_ENABLE_WRFCHEM_PNNL=1 and, for the
   opt-in's erfc, WOOF_WRF_SOURCE).  Two independent codes for the same
   published parameterisation agree within TOLERANCE, never bitwise.

TOLERANCE, and why.  ARG00 itself reproduces a detailed parcel model only to
about 10 % in the activated number over its test range (ARG00 figs. 1-6);
two implementations of it are interchangeable when they differ by much less
than that.  The sweep covers 1-2 lognormal modes, N 1e6-2e10 m-3, number
median diameter 0.01-2 um, sigma_g 1.3-2.2 (SORGAM's modes are 1.7/2.0/2.2),
B 0.03-1.26, T 240-305 K, rho 0.4-1.3 kg m-3, w 0.01-10 m/s:

* one updraft (sigw = 0): every fraction and flux within 0.005 absolute
  (fluxes relative to |wbar|).  Measured over 12,000 cases: 0.0023 at most,
  mean 2e-4; the residual is the thermodynamic constants' choice.
* a Gaussian updraft distribution whose bounds do not cut it (wmin <= wbar -
  6 sigw - 1, wmax >= wbar + 6 sigw + 1): the fractions within 0.10 (mean
  <= 0.01), the fluxes within 0.10 of max(|wbar|, sigw) (mean <= 0.002).
  Measured over 15,000 cases: fractions 0.080 at most (p99 0.043, mean
  0.005), fluxes 0.055 (p99 0.006, mean 9e-4).  This residual is the updraft quadrature: the
  clean room's flux_fullact equals the closed-form Gaussian moment to 2e-6
  (test_updraft_integral_matches_the_gaussian_moment); the opt-in's differs
  from it by about 0.3 %.

Not graded, reported in the PR: (a) bounds that cut into the distribution,
where the opt-in's flux_fullact departs from the Gaussian moment by up to
15 % of the updraft scale; (b) cases where the opt-in activates nothing
(fn = 0) while ARG00 gives a small number fraction (<~ 2e-3) and, through
the lognormal mass tail of eq. 14, a mass fraction up to ~0.25; the opt-in
applies a cutoff the papers do not have, so fs/fm are graded only where it
activates (fn > 1e-3).
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core import chem_activation as CR

SINGLE_TOL = 0.005
DIST_FRACTION_TOL, DIST_FRACTION_MEAN = 0.10, 0.01
DIST_FLUX_TOL, DIST_FLUX_MEAN = 0.10, 0.002


def _mode_volume(n, dg, sg):
    return n * math.pi / 6.0 * dg ** 3 * math.exp(4.5 * math.log(sg) ** 2)


def _one(w, n=1e9, dg=0.1e-6, sg=2.0, b=0.5, T=283.0, rho=1.2, **kw):
    f, full, smax = CR.reference_case([n], [_mode_volume(n, dg, sg)], [sg], [b], wbar=w,
                                      sigw=kw.pop("sigw", 0.0), tair=T, rhoair=rho, **kw)
    return f[0], full, smax


def test_fractions_are_ordered_and_bounded():
    for w in (0.01, 0.1, 1.0, 10.0):
        for sg in (1.3, 1.7, 2.2):
            f, _full, smax = _one(w, sg=sg)
            fn, fs, fm = f[:3]
            assert 0.0 <= fn <= fs <= fm <= 1.0
            assert 0.0 < smax < 0.05


def test_activation_rises_with_updraft_and_hygroscopicity_and_falls_with_number():
    fn_w = [_one(w)[0][0] for w in (0.05, 0.2, 1.0, 5.0)]
    assert all(a < b for a, b in zip(fn_w, fn_w[1:]))
    fn_b = [_one(0.5, b=b)[0][0] for b in (0.05, 0.2, 0.6, 1.2)]
    assert all(a < b for a, b in zip(fn_b, fn_b[1:]))
    # More particles compete for the vapour: a lower fraction (ARG00 eq. 6).
    fn_n = [_one(0.5, n=n)[0][0] for n in (1e7, 1e8, 1e9, 1e10)]
    assert all(a > b for a, b in zip(fn_n, fn_n[1:]))


def test_large_updraft_activates_a_coarse_mode_completely():
    f, _full, _s = _one(10.0, n=1e6, dg=1.0e-6, sg=1.5, b=0.6)
    assert f[0] > 0.999 and f[2] > 0.9999


def test_prescribed_supersaturation_is_the_erfc_of_eq_13():
    sg, dg, b, T = 2.0, 0.1e-6, 0.5, 283.0
    f, _full, smax = _one(0.5, sg=sg, dg=dg, b=b, T=T, smax=0.003)
    assert smax == 0.003
    A = 2 * CR.SURF_TENSION * CR.MW_WATER / (CR.RHO_WATER * CR.R_UNIV * T)
    sm = 2 / math.sqrt(b) * (A / (3 * dg / 2)) ** 1.5
    u = 2 * math.log(sm / 0.003) / (3 * math.sqrt(2) * math.log(sg))
    assert f[0] == pytest.approx(0.5 * math.erfc(u), rel=1e-12)


def test_updraft_integral_matches_the_gaussian_moment():
    """flux_fullact = int_{lo}^{hi} w p(w) dw in closed form."""
    for wbar, sigw, wmin, wmax in ((0.5, 0.5, -10, 10), (-0.2, 0.5, -10, 10),
                                   (2.0, 1.0, 0.0, 2.5), (0.05, 0.02, -1, 1)):
        _f, full, _s = _one(wbar, sigw=sigw, wmin=wmin, wmax=wmax)
        lo, hi = max(wmin, 0.0, wbar - 8 * sigw), min(wmax, wbar + 8 * sigw)

        def F(w):
            z = (w - wbar) / sigw
            return (wbar * 0.5 * (1 + math.erf(z / math.sqrt(2)))
                    - sigw / math.sqrt(2 * math.pi) * math.exp(-0.5 * z * z))
        assert full == pytest.approx(F(hi) - F(lo), rel=2e-5, abs=1e-9)


def test_single_updraft_flux_is_fraction_times_updraft():
    f, full, _s = _one(1.3)
    assert full == 1.3
    np.testing.assert_allclose(f[3:], 1.3 * f[:3], rtol=1e-15)
    f, full, _s = _one(-0.4)
    assert full == 0.0 and not f.any()


def _sweep(rng, regime, ncase):
    cases = []
    for _ in range(ncase):
        nt = int(rng.integers(1, 3))
        sig = 0.0 if regime == "single" else float(10 ** rng.uniform(-1.5, 0.3))
        wb = float(10 ** rng.uniform(-2, 1))
        c = dict(wbar=wb, sigw=sig, wdiab=0.0, wminf=wb - 6 * sig - 1, wmaxf=wb + 6 * sig + 1,
                 tair=float(rng.uniform(240, 305)), rhoair=float(rng.uniform(0.4, 1.3)),
                 msectional=0, ntype_aer=nt, nsize_aer=np.array([1, 1], np.int32), smax=None,
                 flux_fullact=0.0)
        for k in CR.INPUTS + CR.OUTPUTS:
            c[k] = np.zeros((CR.MAXD_ASIZE, CR.MAXD_ATYPE), np.float32)
        for t in range(nt):
            n, s = 10 ** rng.uniform(6, 10.3), rng.uniform(1.3, 2.2)
            dg = 10 ** rng.uniform(-8, -5.7)
            c["na"][0, t], c["sigman"][0, t] = n, s
            c["hygro"][0, t] = 10 ** rng.uniform(-1.5, 0.1)
            c["volc"][0, t] = _mode_volume(n, dg, s)
            c["dlo_sect"][0, t], c["dhi_sect"][0, t] = 1e-9, 1e-4
        cases.append(c)
    return cases


@requires_gpu
def test_kernel_matches_the_float64_reference():
    rng = np.random.default_rng(7)
    cases = _sweep(rng, "single", 300) + _sweep(rng, "distribution", 300)
    cases[0]["smax"] = 0.002
    cases[1]["wdiab"] = 0.3
    out = CR.launch_activate(cases)
    worst = 0.0
    for i, c in enumerate(cases):
        nt = c["ntype_aer"]
        f, full, _s = CR.reference_case(
            c["na"][0, :nt], c["volc"][0, :nt], c["sigman"][0, :nt], c["hygro"][0, :nt],
            wbar=c["wbar"], sigw=c["sigw"], tair=c["tair"], rhoair=c["rhoair"],
            wdiab=c["wdiab"], wmin=c["wminf"], wmax=c["wmaxf"], smax=c["smax"])
        scale = max(abs(c["wbar"]), c["sigw"])
        for j, k in enumerate(CR.OUTPUTS):
            d = np.abs(out[k][i, 0, :nt] - f[:, j]) / (1.0 if j < 3 else scale)
            worst = max(worst, float(d.max()))
        assert abs(out["flux_fullact"][i] - full) <= 2e-5 * scale + 1e-7
    assert worst < 2e-5, worst


def test_sectional_aerosol_is_refused():
    c = _sweep(np.random.default_rng(1), "single", 1)[0]
    c["msectional"] = 1
    with pytest.raises(ValueError, match="modal aerosol only"):
        CR.launch_activate([c])


def test_implementation_door():
    assert CR.implementation() is CR
    assert CR.implementation("cleanroom") is CR
    with pytest.raises(ValueError):
        CR.implementation("other")


def _opt_in():
    from woof import optin
    reason = optin.unavailable_reason("wrfchem_pnnl")
    if reason:
        pytest.skip(reason)
    pnnl = CR.implementation("pnnl")
    reason = pnnl.byo_reason()
    if reason:
        pytest.skip(reason)
    return pnnl


def _grade(regime, seeds, ncase):
    pnnl = _opt_in()
    stats = {k: [] for k in CR.OUTPUTS}
    for seed in seeds:
        cases = _sweep(np.random.default_rng(seed), regime, ncase)
        a, b = CR.launch_activate(cases), pnnl.launch_activate(cases)
        for i, c in enumerate(cases):
            nt = c["ntype_aer"]
            scale = max(abs(c["wbar"]), c["sigw"])
            active = b["fn"][i, 0, :nt] > 1e-3
            for j, k in enumerate(CR.OUTPUTS):
                d = np.abs(a[k][i, 0, :nt] - b[k][i, 0, :nt]) / (1.0 if j < 3 else scale)
                stats[k].extend(d if j % 3 == 0 else d[active])
    return {k: np.asarray(v) for k, v in stats.items()}


@requires_gpu
def test_single_updraft_against_the_opt_in():
    stats = _grade("single", (500, 501, 502), 1000)
    for k, d in stats.items():
        assert d.max() <= SINGLE_TOL, (k, d.max())


@requires_gpu
def test_updraft_distribution_against_the_opt_in():
    stats = _grade("distribution", (510, 511, 512), 1000)
    for j, (k, d) in enumerate(stats.items()):
        tol, mean = ((DIST_FRACTION_TOL, DIST_FRACTION_MEAN) if j < 3
                     else (DIST_FLUX_TOL, DIST_FLUX_MEAN))
        assert d.max() <= tol and d.mean() <= mean, (k, d.max(), d.mean())
