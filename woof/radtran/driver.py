"""woof.radtran's spectral calculation: optics -> GPU solve -> REPTRAN bands.

:func:`compute` takes one or many atmospheric columns (levels top first, the
same number of levels in every column), optional per-layer clouds, and
:class:`~woof.radtran.options.Options`; it returns band-averaged spectral
irradiances (and radiances) at the requested levels for every REPTRAN band
overlapping the wavelength range.

Per band b with representative wavelengths i and weights w_i (Gasteiger et
al. 2014), with E0_i the REPTRAN extraterrestrial irradiance at lambda_i and
d the Sun-Earth distance factor:

    E_b = d * sum_i w_i E0_i E_i,     E_i = solver result for a unit beam.

``output_quantity transmittance`` divides by d * sum_i w_i E0_i.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from woof.radtran import aerosol as opac
from woof.radtran import atmosphere, clouds, data, ims, optics, solver
from woof.radtran.options import Options
from woof.radtran.reptran import open_reptran

FLUX_NAMES = ("edir", "edn", "eup", "uavgdir", "uavgdn", "uavgup")


@dataclass
class CloudLayers:
    """Per-layer cloud water [g m-3] and effective radius [um], (ncol, nlyr)."""
    lwc: np.ndarray | None = None
    reff_wc: np.ndarray | None = None
    iwc: np.ndarray | None = None
    reff_ic: np.ndarray | None = None


@dataclass
class Result:
    wavelength: np.ndarray        # (nband,) band centre [nm]
    wvlmin: np.ndarray
    wvlmax: np.ndarray
    zout: np.ndarray              # (ncol, nout) altitude above surface [km]
    edir: np.ndarray              # (ncol, nout, nband)
    edn: np.ndarray
    eup: np.ndarray
    uavgdir: np.ndarray
    uavgdn: np.ndarray
    uavgup: np.ndarray
    uu: np.ndarray                # (ncol, nout, numu, nphi, nband)
    umu: np.ndarray
    phi: np.ndarray


def sun_distance_factor(day_of_year: int | None) -> float:
    """(r0/r)^2 after Spencer (1971), Search 2, 172; 1 when no day given."""
    if day_of_year is None:
        return 1.0
    g = 2.0 * math.pi * (day_of_year - 1) / 365.0
    return (1.000110 + 0.034221 * math.cos(g) + 0.001280 * math.sin(g)
            + 0.000719 * math.cos(2 * g) + 0.000077 * math.sin(2 * g))


def read_cloud_file(path: str, column: atmosphere.Column, kind: str):
    """uvspec ``wc_file 1D`` / ``ic_file 1D`` semantics on the column's layers.

    Each row (z [km], content [g m-3], r_eff [um]) defines the layer from its
    altitude to the next row above; the top row defines nothing.  Rows must
    sit on the column's levels.
    """
    rows = []
    for line in open(path):
        s = line.split("#", 1)[0].split()
        if s:
            rows.append([float(x) for x in s[:3]])
    rows = np.array(sorted(rows, key=lambda r: r[0]))
    z = column.z if column.z.ndim == 1 else column.z[0]
    zb = z[1:]
    content = np.zeros(zb.shape)
    reff = np.zeros(zb.shape)
    for k in range(rows.shape[0] - 1):
        lo, hi = rows[k, 0], rows[k + 1, 0]
        sel = (zb >= lo - 1e-9) & (zb < hi - 1e-9)
        content[sel] = rows[k, 1]
        reff[sel] = rows[k, 2]
    return content, reff


def _hg_phase(g, mu):
    g = np.asarray(g)
    return (1.0 - g * g) / (1.0 + g * g - 2.0 * g * mu) ** 1.5


def _layer_optics(opts, col, rep, iw, cl, nstr, numu_angles, aer=None, aer_nodes=None, cache=None):
    """(dtau, ssa, pmom, ptrue) for every column at representative wavelength iw.

    ``cache``: a dict kept across wavelengths for one ``col`` and ``cl``
    (:func:`woof.radtran.optics.molecular`, :func:`woof.radtran.clouds.hu_stamnes`)."""
    memo = optics._memo
    mo = optics.molecular(col, rep, iw, cache=cache)
    lam = float(rep.wvl[iw])
    tr = mo.dtau_ray
    ta = mo.dtau_abs
    if opts.no_absorption:
        ta = np.zeros_like(ta)
    if opts.no_scattering:
        tr = np.zeros_like(tr)
    shape = tr.shape
    nmom = nstr
    coef = np.zeros(shape + (nmom + 1,))
    coef[..., 0] = tr
    if nmom >= 2:
        coef[..., 2] = tr * mo.chi2[..., None] if np.ndim(mo.chi2) else tr * float(mo.chi2)
    tsca = tr.copy()
    text = tr + ta
    hg = []                       # (tau_sca, g) per cloud kind for the exact phase function
    dz_km = memo(cache, "dz_km", lambda: col.z[..., :-1] - col.z[..., 1:])
    if cl is not None and cl.lwc is not None:
        lwc = np.broadcast_to(cl.lwc, shape)
        reff = memo(cache, "reff_wc", lambda: np.where(lwc > 0, np.broadcast_to(cl.reff_wc, shape), 10.0))
        ext, w, g = clouds.hu_stamnes(lam, reff, cache=None if cache is None else cache.setdefault("hu", {}))
        tau = lwc * ext * dz_km
        hg.append(_add_cloud(opts, tau, w, g, coef, nmom))
        tsca = tsca + hg[-1][0]
        text = text + (hg[-1][0] if opts.no_absorption else tau)
    if cl is not None and cl.iwc is not None:
        iwc = np.broadcast_to(cl.iwc, shape)
        reff = memo(cache, "reff_ic", lambda: np.where(iwc > 0, np.broadcast_to(cl.reff_ic, shape), 30.0))
        ext, w, g = clouds.fu96(lam, reff, reff_def_fu=opts.ic_fu_reff_def)
        tau = iwc * ext * dz_km * 1000.0
        hg.append(_add_cloud(opts, tau, w, g, coef, nmom))
        tsca = tsca + hg[-1][0]
        text = text + (hg[-1][0] if opts.no_absorption else tau)
    la = None
    if aer is not None:
        la = opac.layer_optics(aer, aer_nodes, lam, dz_km)
        ts = la.tau_ext if opts.no_absorption else la.tau_sca
        if opts.no_scattering:
            ts = np.zeros_like(ts)
        la_scale = np.divide(ts, la.tau_sca, out=np.zeros_like(ts), where=la.tau_sca > 0)
        coef += (la.sca_chi[..., :nmom + 1] * la_scale[..., None])
        tsca = tsca + ts
        text = text + (ts if opts.no_absorption else la.tau_ext)
    with np.errstate(invalid="ignore", divide="ignore"):
        ssa = np.where(text > 0, tsca / text, 0.0)
        pmom = np.where(tsca[..., None] > 0, coef / tsca[..., None], 0.0)
    pmom[..., 0] = 1.0
    # forward-peaked components for the IMS correction (scattering optical depth, g or moments)
    peaks = [("hg", ts, g) for ts, g in hg]
    if la is not None:
        for s_, ts_, ih_ in la.by_species:
            sa = opac.species_at(s_, lam)
            chi = np.divide(sa.sca_chi[ih_], sa.sca[ih_][..., None], out=np.zeros(ih_.shape + (opac.LMAX + 1,)),
                            where=sa.sca[ih_][..., None] > 0)
            peaks.append(("mom", ts_ * la_scale, chi))
    ptrue = None
    if numu_angles is not None:
        cosang = numu_angles[:, None, :, :]                   # (ncol, 1, numu, nphi)
        p2 = 0.5 * (3.0 * cosang * cosang - 1.0)
        chi2 = np.broadcast_to(np.asarray(mo.chi2, dtype=np.float64), (cosang.shape[0],))
        pr = 1.0 + 5.0 * chi2[:, None, None, None] * p2
        num = tr[..., None, None] * pr
        for ts, g in hg:
            num = num + ts[..., None, None] * _hg_phase(g[..., None, None], cosang)
        if la is not None:
            num = num + opac.layer_phase_sum(la, lam, cosang, la_scale)
        with np.errstate(invalid="ignore", divide="ignore"):
            ptrue = np.where(tsca[..., None, None] > 0, num / tsca[..., None, None], 1.0)
    return text, ssa, pmom, ptrue, peaks


#: Columns per host thread in :func:`_layer_optics_columns`.
OPTICS_CHUNK = 4096

#: Default host-optics bound of one solve (compute's ``group_bytes``).
GROUP_BYTES = 2 << 30


def optics_bytes_per_wavelength(ncol, nlyr, nstr, nang=0):
    """Host bytes one representative wavelength's optics take in a solve:
    dtau, ssa, the nstr+1 moments, the user-angle phase functions and zlev
    (float64), counted three times: the stacked arrays, the solver's
    contiguous copies, and an allowance for each column block's
    wavelength-free cache (air and gas columns, REPTRAN brackets, cloud
    selections), which lives through the group's wavelengths."""
    return 3 * 8 * int(ncol) * (int(nlyr) * (2 + int(nstr) + 1 + int(nang)) + int(nlyr) + 1)


def band_groups(rep, bands, per_wavelength, group_bytes=None):
    """Indices into ``bands`` in consecutive groups, each group's distinct
    representative wavelengths within ``group_bytes`` (at least one band per
    group).  Grouping is storage only: each problem solves alone and each band
    sums its own wavelengths in its own order."""
    if group_bytes is None:
        group_bytes = int(os.environ.get("WOOF_RADTRAN_GROUP_BYTES", GROUP_BYTES))
    groups, cur, seen = [], [], set()
    for j, b in enumerate(bands):
        iws = {int(i) for i in rep.band(b).iwvl}
        if cur and len(seen | iws) * per_wavelength > group_bytes:
            groups.append(cur)
            cur, seen = [], set()
        cur.append(j)
        seen |= iws
    if cur:
        groups.append(cur)
    return groups


def _layer_optics_columns(opts, col, rep, iws, cl, nstr, numu_angles, aer=None, aer_nodes=None):
    """:func:`_layer_optics` at every representative wavelength of ``iws``:
    (text, ssa, pmom, ptrue) stacked (nw, ncol, ...) and one peaks list per
    wavelength.

    Every operation in ``_layer_optics`` is per column (elementwise, or along
    a column's own layers), so a block of columns gives the block's words of
    the whole-domain call; numpy releases the GIL inside its loops, so the
    blocks run in parallel on host threads and their temporaries stay small.
    Each block runs every wavelength with one cache of its wavelength-free
    parts and writes its rows of the stacked arrays in place.  Aerosol
    layers (per-species node selections) keep the single block.
    """
    nw = len(iws)
    if col.z.ndim != 2:
        # one column (levels only): its optics have no column axis to block or
        # to write rows of, so stack the per-wavelength results
        cache = {}
        res = [_layer_optics(opts, col, rep, iw, cl, nstr, numu_angles, aer, aer_nodes, cache)
               for iw in iws]
        text, ssa, pmom, ptrue = (None if r[0][i] is None else np.stack([x[i] for x in r])
                                  for i, r in ((i, res) for i in range(4)))
        return text, ssa, pmom, ptrue, [x[4] for x in res]
    nd = col.z.shape[0]
    nthreads = min(8, os.cpu_count() or 1)
    out = {}

    def put(k, c0, res):
        text, ssa, pmom, ptrue, peaks = res
        if k not in out:
            # allocated by the first block, which runs alone on this thread
            if c0 != 0:
                raise RuntimeError("_layer_optics_columns: block 0 must run first and alone")
            if not out:
                out["arrays"] = [None if a is None else np.empty((nw, nd) + a.shape[1:], dtype=a.dtype)
                                 for a in (text, ssa, pmom, ptrue)]
            out[k] = (peaks if text.shape[0] == nd else
                      [(kind, np.empty((nd,) + ts.shape[1:], ts.dtype), np.empty((nd,) + g.shape[1:], g.dtype))
                       for kind, ts, g in peaks])
        rows = slice(c0, c0 + text.shape[0])
        for dst, a in zip(out["arrays"], (text, ssa, pmom, ptrue)):
            if dst is not None:
                dst[k, rows] = a
        if out[k] is not peaks:
            for (_, ts_d, g_d), (_, ts, g) in zip(out[k], peaks):
                ts_d[rows] = ts
                g_d[rows] = g

    if aer is not None or nd <= OPTICS_CHUNK or nthreads < 2:
        cache = {}
        for k, iw in enumerate(iws):
            put(k, 0, _layer_optics(opts, col, rep, iw, cl, nstr, numu_angles, aer, aer_nodes, cache))
    else:
        from concurrent.futures import ThreadPoolExecutor
        L = col.z.shape[1] - 1

        def block(c0):
            rows = slice(c0, min(c0 + OPTICS_CHUNK, nd))
            cb = atmosphere.Column(z=col.z[rows], p=col.p[rows], t=col.t[rows], n_air=col.n_air[rows],
                                   dens={k: v[rows] for k, v in col.dens.items()})
            clb = None if cl is None else CloudLayers(**{
                k: (None if v is None else np.broadcast_to(v, (nd, L))[rows]) for k, v in cl.__dict__.items()})
            ang = None if numu_angles is None else numu_angles[rows]
            cache = {}
            for k, iw in enumerate(iws):
                put(k, c0, _layer_optics(opts, cb, rep, iw, clb, nstr, ang, cache=cache))

        # the first block alone: it loads the lazily read tables (netCDF) once,
        # off the threads, and allocates the stacked arrays
        block(0)
        with ThreadPoolExecutor(nthreads) as pool:
            list(pool.map(block, range(OPTICS_CHUNK, nd, OPTICS_CHUNK)))
    text, ssa, pmom, ptrue = out["arrays"]
    return text, ssa, pmom, ptrue, [out[k] for k in range(nw)]


def _add_cloud(opts, tau, w, g, coef, nmom):
    ts = tau * (1.0 if opts.no_absorption else w)
    if opts.no_scattering:
        ts = np.zeros_like(ts)
    gl = np.ones_like(g)
    for l in range(nmom + 1):
        coef[..., l] += ts * gl
        gl = gl * g
    return ts, g


def select_bands(rep, opts: Options, wavelengths=None) -> list[int]:
    """REPTRAN bands: every band overlapping ``opts.wavelength`` (uvspec's
    rule), or, given ``wavelengths``, the band containing each one."""
    if wavelengths is not None:
        return sorted({rep.band_containing(float(w)) for w in wavelengths})
    lo, hi = opts.wavelength
    bands = rep.bands_overlapping(lo, hi)
    if not bands:
        raise ValueError(f"no REPTRAN band overlaps {lo}-{hi} nm")
    return bands


def compute(opts: Options, col: atmosphere.Column, clouds_in: CloudLayers | None = None,
            *, aerosol: "opac.AerosolLayers | None" = None, backend: str = "gpu", umu0=None, albedo=None,
            wavelengths=None, budget_bytes: int | None = None, group_bytes: int | None = None) -> Result:
    """Band-averaged spectral irradiances/radiances for every column.

    ``umu0`` and ``albedo`` (per column) override ``opts.sza`` and
    ``opts.albedo``; columns with the sun at or below the horizon (umu0 <= 0)
    get zeros.  ``wavelengths`` selects the bands containing those
    wavelengths instead of ``opts.wavelength``'s range.  ``group_bytes``
    bounds the host optics one solve holds (band_groups; default GROUP_BYTES
    or WOOF_RADTRAN_GROUP_BYTES); it moves no word.
    """
    opts.validate()
    if col.z.ndim == 1:
        col = atmosphere.Column(z=col.z[None], p=col.p[None], t=col.t[None],
                                n_air=col.n_air[None], dens={k: v[None] for k, v in col.dens.items()})
    ncol, nlev = col.z.shape
    L = nlev - 1
    rep = open_reptran(str(data.reptran_file(opts.reptran_resolution)))
    bands = select_bands(rep, opts, wavelengths)
    nstr = opts.streams
    mu0_all = (np.full(ncol, math.cos(math.radians(opts.sza))) if umu0 is None
               else np.broadcast_to(np.asarray(umu0, dtype=np.float64), (ncol,)).copy())
    alb_all = np.broadcast_to(np.asarray(opts.albedo if albedo is None else albedo, dtype=np.float64),
                              (ncol,)).copy()
    # output levels
    zsur = col.z[:, -1]
    outlev = []
    for zo in opts.zout:
        if zo == "sur":
            outlev.append(L)
        elif zo == "toa":
            outlev.append(0)
        else:
            k = np.nonzero(np.abs(col.z[0] - zsur[0] - float(zo)) < 1e-6)[0]
            if k.size == 0:
                raise ValueError(f"zout {zo} km is not a level of the column")
            outlev.append(int(k[0]))
    nout = len(outlev)
    phis = [p - opts.phi0 for p in (opts.phi if opts.phi else ((0.0,) if opts.umu else ()))]
    numu, nphi = len(opts.umu), len(phis)
    nb = len(bands)
    fo = np.zeros((6, ncol, nout, nb))
    uu = np.zeros((ncol, nout, numu, nphi, nb))
    centers = [0.5 * (rep.wvlmin[b] + rep.wvlmax[b]) for b in bands]
    zout = np.array([[col.z[c, k] - zsur[c] for k in outlev] for c in range(ncol)])
    result = Result(wavelength=np.array(centers), wvlmin=rep.wvlmin[bands].copy(),
                    wvlmax=rep.wvlmax[bands].copy(), zout=zout,
                    **{n: fo[i] for i, n in enumerate(FLUX_NAMES)}, uu=uu,
                    umu=np.asarray(opts.umu, dtype=np.float64), phi=np.asarray(opts.phi, dtype=np.float64))
    day = np.nonzero(mu0_all > 0.0)[0]
    if day.size == 0:
        return result
    if day.size < ncol:
        col = atmosphere.Column(z=col.z[day], p=col.p[day], t=col.t[day], n_air=col.n_air[day],
                                dens={k: v[day] for k, v in col.dens.items()})
        if clouds_in is not None:
            clouds_in = CloudLayers(**{k: (None if v is None else np.broadcast_to(v, (ncol, L))[day])
                                       for k, v in clouds_in.__dict__.items()})
        if aerosol is not None:
            aerosol = opac.AerosolLayers(aerosol.species, np.broadcast_to(
                aerosol.mass, (ncol, L, len(aerosol.species)))[day])
    nd = day.size
    mu0 = mu0_all[day]
    # user angles: cosine of the scattering angle per (column, umu, phi)
    ang = None
    if opts.umu:
        u = np.asarray(opts.umu, dtype=np.float64)[None, :, None]
        ph = np.radians(np.asarray(phis, dtype=np.float64))[None, None, :]
        m = mu0[:, None, None]
        ang = -m * u + np.sqrt(np.maximum(0.0, 1 - u * u)) * np.sqrt(np.maximum(0.0, 1 - m * m)) * np.cos(ph)
    flags = (solver.DELTAM if opts.use_deltam else 0)
    if opts.rte_solver == "twostr":
        flags |= solver.TWOSTREAM
    if opts.pseudospherical:
        flags |= solver.SPHERICAL
    if opts.umu and opts.disort_intcor == "moments":
        flags |= solver.TMS
    aer_nodes = None if aerosol is None else opac.humidity_nodes(aerosol, col)
    dist = sun_distance_factor(opts.day_of_year)
    # Bands in groups whose representative wavelengths' host optics fit the
    # group budget (band_groups): every (wavelength, column) problem is solved
    # on its own and a band sums only its own wavelengths, so a group gives
    # the words of the one-solve call while the host holds one group's optics.
    per_iw = optics_bytes_per_wavelength(nd, L, nstr, numu * nphi if flags & solver.TMS else 0)
    nw_total = len({int(i) for b in bands for i in rep.band(b).iwvl})
    for group in band_groups(rep, bands, per_iw, group_bytes):
        giws = sorted({int(i) for j in group for i in rep.band(bands[j]).iwvl})
        D, S, P, PT, PK = _layer_optics_columns(opts, col, rep, giws, clouds_in, nstr,
                                                ang if flags & solver.TMS else None, aerosol, aer_nodes)
        nw = len(giws)
        dtau = D.reshape(nw * nd, L)
        ssa = S.reshape(nw * nd, L)
        pmom = P.reshape(nw * nd, L, nstr + 1)
        ptrue = None if PT is None else PT.reshape(nw * nd, L, numu, nphi)
        del PT
        zlev = np.tile(col.z, (nw, 1))
        flux, rad = solver.solve(nstr=nstr, flags=flags, dtau=dtau, ssa=ssa, pmom=pmom, umu0=np.tile(mu0, nw),
                                 fbeam=1.0, albedo=np.tile(alb_all[day], nw), zlev=zlev,
                                 radius=opts.earth_radius, outlev=outlev, umu=opts.umu, phi=phis, ptrue=ptrue,
                                 backend=backend, budget_bytes=budget_bytes, nprob_total=nw_total * nd)
        del dtau, ssa, pmom, ptrue, zlev
        flux = flux.reshape(nw, nd, nout, 6)
        rad = rad.reshape(nw, nd, nout, numu, nphi)
        if flags & solver.TMS and flags & solver.DELTAM:
            # Nakajima & Tanaka (1988) IMS: secondary scattering through the forward peak
            for k in range(nw):
                f = P[k][..., nstr]
                rad[k] += ims.correction(dtau=D[k], ssa=S[k], f=f, components=PK[k], nstr=nstr, umu0=mu0,
                                         umu=opts.umu, cosang=ang, outlev=outlev, fbeam=1.0)
        del D, S, P, PK
        pos = {iw: k for k, iw in enumerate(giws)}
        for j in group:
            B = rep.band(bands[j])
            e0 = 0.0
            accf = np.zeros((6, nd, nout))
            accr = np.zeros((nd, nout, numu, nphi))
            for iw, w in zip(B.iwvl, B.weight):
                k = pos[int(iw)]
                f = w * rep.extra[iw] * dist
                e0 += f
                accf += f * np.moveaxis(flux[k], -1, 0)
                accr += f * rad[k]
            if opts.output_quantity == "transmittance":
                accf /= e0
                accr /= e0
            fo[..., j][:, day, :] = accf
            uu[..., j][day] = accr
        del flux, rad
    return result
