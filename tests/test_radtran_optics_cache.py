"""The radtran host optics' wavelength-free cache keeps the words.

driver._layer_optics_columns runs every representative wavelength of a band
group per column block with one cache of the wavelength-free parts
(optics.molecular, clouds.hu_stamnes), REPTRAN brackets are shared across
species and wavelengths (reptran.Lookup.bracket / sigma_at), and
atmosphere.model_columns splices in place.  Each must give the words of the
uncached, per-wavelength evaluation; the legacy forms are spelled out here.
"""

from __future__ import annotations

import numpy as np
import pytest


def _need_data():
    from woof.radtran import data
    try:
        data.data_dir()
    except data.LibradtranDataMissing as exc:
        pytest.skip(str(exc))


def _bits(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64)).view(np.uint64)


def _legacy_sigma(lk, iw, p_hpa, t, vmr=None):
    """reptran.Lookup.sigma before the bracket split."""
    from woof.radtran.reptran import XSEC_TO_CM2
    x = lk.xsec[:, :, lk._row[iw], :]
    p = np.asarray(p_hpa, dtype=np.float64) * 100.0
    t = np.asarray(t, dtype=np.float64)
    P = lk.pressure
    Pa = P[::-1]
    ip = np.clip(np.searchsorted(Pa, p) - 1, 0, Pa.size - 2)
    p0, p1 = Pa[ip], Pa[ip + 1]
    wp1 = (p - p0) / (p1 - p0)
    out = np.zeros(np.broadcast(p, t).shape)
    for node, wpn in ((P.size - 1 - ip, 1.0 - wp1), (P.size - 2 - ip, wp1)):
        dt = t - lk.t_ref[node]
        it = np.clip(np.searchsorted(lk.t_pert, dt) - 1, 0, lk.t_pert.size - 2)
        t0, t1 = lk.t_pert[it], lk.t_pert[it + 1]
        wt1 = (dt - t0) / (t1 - t0)
        if lk.vmrs.size > 1 and vmr is not None:
            v = np.asarray(vmr, dtype=np.float64)
            iv = np.clip(np.searchsorted(lk.vmrs, v) - 1, 0, lk.vmrs.size - 2)
            v0, v1 = lk.vmrs[iv], lk.vmrs[iv + 1]
            wv1 = (v - v0) / (v1 - v0)
            val = 0.0
            for ti, wt in ((it, 1.0 - wt1), (it + 1, wt1)):
                for vi, wv in ((iv, 1.0 - wv1), (iv + 1, wv1)):
                    val = val + wt * wv * x[ti, vi, node]
        else:
            val = (1.0 - wt1) * x[it, 0, node] + wt1 * x[it + 1, 0, node]
        out = out + wpn * val
    return out * XSEC_TO_CM2


def _levels(rng, ncol=257, nlev=64):
    # top first; pressures and temperatures past both ends of the tables
    p = np.sort(rng.uniform(1e-4, 1100.0, (ncol, nlev)), axis=1)
    t = rng.uniform(150.0, 340.0, p.shape)
    v = rng.uniform(0.0, 0.06, p.shape)
    return p, t, v


def test_shared_brackets_give_the_legacy_cross_sections():
    _need_data()
    from woof.radtran import data
    from woof.radtran.reptran import SPECIES, open_reptran
    rep = open_reptran(str(data.reptran_file("coarse")))
    p, t, v = _levels(np.random.default_rng(11))
    shared = {}
    for sp in SPECIES:
        lk = rep.lookup(sp)
        for vmr in (None, v):
            key = lk.bracket_key(vmr is not None)
            br = shared.setdefault(key, lk.bracket(p, t, vmr))
            for iw in sorted(lk._row)[::max(1, len(lk._row) // 7)]:
                want = _legacy_sigma(lk, iw, p, t, vmr)
                assert np.array_equal(_bits(lk.sigma_at(iw, br)), _bits(want)), (sp, iw)
                assert np.array_equal(_bits(lk.sigma(iw, p, t, vmr)), _bits(want)), (sp, iw)
    # one grid: H2O with its vmr, and every other lookup (and H2O without it)
    assert len(shared) == 3


def test_cached_molecular_optics_keep_the_words():
    _need_data()
    from woof.radtran import atmosphere, data, optics
    from woof.radtran.reptran import open_reptran
    rep = open_reptran(str(data.reptran_file("coarse")))
    rng = np.random.default_rng(5)
    ncol, nz = 300, 30
    z = np.sort(rng.uniform(0.0, 18.0, (ncol, nz)), axis=1)[:, ::-1] + np.linspace(0.6, 0.0, nz)
    p = 1013.25 * np.exp(-z / 7.6)
    t = 288.0 - 6.5 * np.minimum(z, 11.0) + rng.normal(0, 2, z.shape)
    col = atmosphere.model_columns(z, p, t, h2o_vmr=rng.uniform(0, 0.03, z.shape), co2_vmr=4e-4)
    iws = sorted({int(i) for lam in (310.0, 550.0, 760.5, 940.0, 1400.0, 2100.0)
                  for i in rep.band(rep.band_containing(lam)).iwvl})
    cache = {}
    for iw in iws:
        want = optics.molecular(col, rep, iw)
        got = optics.molecular(col, rep, iw, cache=cache)
        for name in ("dtau_ray", "dtau_abs", "chi2"):
            assert np.array_equal(_bits(getattr(got, name)), _bits(getattr(want, name))), (iw, name)
        assert sorted(got.by_species) == sorted(want.by_species)
        for sp in want.by_species:
            assert np.array_equal(_bits(got.by_species[sp]), _bits(want.by_species[sp])), (iw, sp)


def test_cached_hu_stamnes_keeps_the_words():
    _need_data()
    from woof.radtran import clouds
    rng = np.random.default_rng(9)
    reff = np.where(rng.random((200, 40)) < 0.6, 10.0, rng.uniform(2.5, 60.0, (200, 40)))
    cache = {}
    for lam in (*rng.uniform(300.0, 3900.0, 12), 550.0, 550.0, 760.5):
        want = clouds.hu_stamnes(lam, reff)
        got = clouds.hu_stamnes(lam, reff, cache=cache)
        for a, b in zip(got, want):
            assert np.array_equal(_bits(a), _bits(b)), lam


def test_model_columns_splice_keeps_the_words():
    _need_data()
    from woof.radtran import atmosphere
    rng = np.random.default_rng(2)
    ncol, nlev = 120, 41
    z = rng.uniform(0.0, 3.0, (ncol, 1)) + np.linspace(22.0, 0.0, nlev)[None, :]
    p = 1013.25 * np.exp(-z / 7.6)
    t = 288.0 - 6.5 * np.minimum(z, 11.0) + rng.normal(0, 1, z.shape)
    h2o, o3 = rng.uniform(0, 0.02, z.shape), rng.uniform(0, 1e-5, z.shape)
    got = atmosphere.model_columns(z, p, t, h2o_vmr=h2o, o3_vmr=o3, co2_vmr=4.1e-4)
    # the legacy concatenation
    na = p * 100.0 / (atmosphere.KB * t) * 1e-6
    ref = atmosphere.afgl("us")
    top = ref.z > z[:, 0].max() + 1.0
    rep = lambda a: np.broadcast_to(a[top], (ncol, int(top.sum())))
    given = {"H2O": h2o, "O3": o3, "CO2": 4.1e-4}
    for gas in atmosphere.GASES:
        v = (np.broadcast_to(np.asarray(given[gas], dtype=np.float64), z.shape) if gas in given
             else np.interp(-z, -ref.z, ref.vmr(gas)))
        want = np.concatenate((rep(ref.dens[gas]), v * na), axis=1)
        assert np.array_equal(_bits(got.dens[gas]), _bits(want)), gas
    for name, a in (("z", z), ("p", p), ("t", t), ("n_air", na)):
        want = np.concatenate((rep(getattr(ref, name)), a), axis=1)
        assert np.array_equal(_bits(getattr(got, name)), _bits(want)), name


@pytest.mark.parametrize("chunk", [10**9, 37])
def test_stacked_block_optics_keep_the_per_wavelength_words(monkeypatch, chunk):
    """Every wavelength per column block with one cache, written into the
    stacked arrays, against the uncached single call per wavelength (the
    radiance phase functions and the IMS peaks included)."""
    _need_data()
    from woof.radtran import atmosphere, data, driver
    from woof.radtran.options import Options
    from woof.radtran.reptran import open_reptran
    rep = open_reptran(str(data.reptran_file("coarse")))
    rng = np.random.default_rng(4)
    ncol, nz = 150, 25
    z = rng.uniform(0.0, 2.0, (ncol, 1)) + np.linspace(20.0, 0.0, nz + 1)[None, :]
    p = 1013.25 * np.exp(-z / 7.6)
    t = 288.0 - 6.5 * np.minimum(z, 11.0) + rng.normal(0, 1, z.shape)
    col = atmosphere.model_columns(z, p, t, h2o_vmr=rng.uniform(0, 0.02, z.shape), co2_vmr=4e-4)
    L = col.z.shape[1] - 1
    lwc = np.where(rng.random((ncol, L)) < 0.2, rng.uniform(0, 0.5, (ncol, L)), 0.0)
    iwc = np.where(rng.random((ncol, L)) < 0.2, rng.uniform(0, 0.05, (ncol, L)), 0.0)
    cl = driver.CloudLayers(lwc=lwc, reff_wc=rng.uniform(3, 20, (ncol, L)),
                            iwc=iwc, reff_ic=rng.uniform(10, 40, (ncol, L)))
    opts = Options(wavelength=(440.0, 940.0), umu=(0.6, -0.8), phi=(0.0, 90.0)).validate()
    nstr = opts.streams
    m = rng.uniform(0.1, 1.0, ncol)[:, None, None]
    u = np.array([0.6, -0.8])[None, :, None]
    ph = np.radians([0.0, 90.0])[None, None, :]
    ang = -m * u + np.sqrt(1 - u * u) * np.sqrt(1 - m * m) * np.cos(ph)
    iws = sorted({int(i) for lam in (440.0, 550.0, 760.5, 940.0)
                  for i in rep.band(rep.band_containing(lam)).iwvl})
    monkeypatch.setattr(driver, "OPTICS_CHUNK", chunk)
    monkeypatch.setattr(driver.os, "cpu_count", lambda: 3)      # blocks on any host
    text, ssa, pmom, ptrue, peaks = driver._layer_optics_columns(opts, col, rep, iws, cl, nstr, ang)
    assert text.shape == (len(iws), ncol, L)
    for k, iw in enumerate(iws):
        want = driver._layer_optics(opts, col, rep, iw, cl, nstr, ang)
        for got, ref in zip((text[k], ssa[k], pmom[k], ptrue[k]), want[:4]):
            assert np.array_equal(_bits(got), _bits(ref)), iw
        assert [kind for kind, _, _ in peaks[k]] == [kind for kind, _, _ in want[4]]
        for (_, ts, g), (_, ts_w, g_w) in zip(peaks[k], want[4]):
            assert np.array_equal(_bits(ts), _bits(ts_w)), iw
            assert np.array_equal(_bits(g), _bits(np.broadcast_to(g_w, ts_w.shape))), iw
