"""Clean-room modal aerosol optics for the SORGAM rows (WOOF P9).

The default build's optical properties of the MADE/SORGAM aerosol
(``chem_opt = 2``) for ``aer_ra_feedback = 1``: per layer, the optical depth,
single-scattering albedo and asymmetry parameter at WRF-Chem's four
shortwave wavelengths (300, 400, 600, 999 nm; tauaer1..4, waer1..4,
gaer1..4, which RRTMG's shortwave spreads over its 14 bands and Fast-TUV
reads), and the absorption optical depth in RRTMG's 16 longwave bands
(tauaerlw1..16).  Written from the textbooks only: no WRF-Chem optics source
(``chem/module_optical_averaging.F`` and ``optical_driver.F`` carry
Battelle/PNNL terms) was read to write it.

Each of SORGAM's three lognormal modes (Aitken, accumulation, coarse, with
its fixed sigma_g 1.70, 2.00, 2.50) is an internal mixture of six component
classes, its wet volume the SORGAM masses over their SORGAM densities plus
the aerosol water SORGAM carries (h2oai, h2oaj):

=============  ==========================================  ===============
class          SORGAM rows                                 density kg m-3
=============  ==========================================  ===============
water          h2oai, h2oaj (SORGAM's aerosol water)       1000
inorganic      so4, nh4, no3 (i, j)                        1800
sea_salt       na, cl (i, j); seas (coarse)                2200
organic        orgaro1/2, orgalk1, orgole1, orgba1-4,       1000
               orgpa (i, j)
elemental      ec (i, j)                                   2200
mineral        p25 (i, j), antha (coarse) 2200; soila 2600
=============  ==========================================  ===============

(the densities are module_aerosols_sorgam's third-moment factors).  The
mixture's refractive index is the volume average (``aer_op_opt = 1``) or
Maxwell Garnett with the elemental carbon as inclusions (``aer_op_opt = 2``,
Bohren and Huffman 1983 eq. 8.50), from the component indices in
``woof/data/chem/modal_optics/refractive_index.v1.json`` (sources in that
file and in tools/chem_modal_optics/make_tables.py).  The mode's optics come
from the lognormal Mie table beside it (WOOF's clean-room Mie code,
woof.radtran.mie), trilinear in (ln x_g, n, ln k).  The kernel is
``woof/core/kernels/chem_modal_optics.cu``.

Validated in ``tests/test_chem_modal_optics.py`` against direct Mie
integration (not against the opt-in: its optics are not ported yet).
Limitations, stated: the inorganic and organic classes carry their visible
refractive index at every wavelength, so their infrared absorption bands are
not represented; the mineral class is fused silica (a silicate proxy, no
iron-oxide absorption in the visible); the coarse mode is dry (SORGAM carries
no coarse water).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_RELATIVE = "chem/modal_optics"
MODES = ("aitken", "accumulation", "coarse")
SIGMA_G = (1.70, 2.00, 2.50)
CLASSES = ("water", "inorganic", "sea_salt", "organic", "elemental_carbon", "mineral")
NSW, NLW = 4, 16
#: aer_op_opt -> the kernel's mixing rule.
MIXING = {1: 1, 2: 2}
_ORG = ("orgaro1", "orgaro2", "orgalk1", "orgole1", "orgba1", "orgba2", "orgba3",
        "orgba4", "orgpa")
#: (row, mode, class, density kg m-3).
ROWS = tuple(
    [(f"{s}a{m}", mi, c, d) for mi, m in ((0, "i"), (1, "j"))
     for s, c, d in (("so4", 1, 1800.0), ("nh4", 1, 1800.0), ("no3", 1, 1800.0),
                     ("na", 2, 2200.0), ("cl", 2, 2200.0))]
    + [(f"{o}{m}", mi, 3, 1000.0) for mi, m in ((0, "i"), (1, "j")) for o in _ORG]
    + [(f"ec{m}", mi, 4, 2200.0) for mi, m in ((0, "i"), (1, "j"))]
    + [(f"p25{m}", mi, 5, 2200.0) for mi, m in ((0, "i"), (1, "j"))]
    + [("antha", 2, 5, 2200.0), ("seas", 2, 2, 2200.0), ("soila", 2, 5, 2600.0)])
#: Aerosol water (SORGAM aux state, ug/kg like the rows) and the numbers (#/kg).
WATER = (("h2oai", 0), ("h2oaj", 1))
NUMBERS = ("nu0", "ac0", "corn")
WATER_DENSITY = 1000.0


def data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / DATA_RELATIVE


@lru_cache(maxsize=1)
def host_data():
    """(manifest, refractive indices (n, k) (NCLASS, 20), wavelengths m,
    table float32 (3, 3, nx, nn, nk)), checked against the manifest."""
    import hashlib
    d = data_dir()
    manifest = json.loads((d / "MANIFEST.json").read_text())
    ri_raw = (d / "refractive_index.v1.json").read_bytes()
    tb_raw = (d / "mie_lognormal.v1.npy").read_bytes()
    for name, raw in (("refractive_index.v1.json", ri_raw), ("mie_lognormal.v1.npy", tb_raw)):
        if hashlib.sha256(raw).hexdigest() != manifest["sha256"][name]:
            raise RuntimeError(f"{d / name} does not match its MANIFEST digest")
    ri = json.loads(ri_raw)
    n = np.array([ri["classes"][c]["n"] for c in CLASSES], np.float64)
    k = np.array([ri["classes"][c]["k"] for c in CLASSES], np.float64)
    wl = np.array(ri["wavelength_um"], np.float64) * 1e-6
    table = np.load(d / "mie_lognormal.v1.npy")
    return manifest, n, k, wl, table


@lru_cache(maxsize=1)
def _device_data():
    import cupy as cp
    manifest, n, k, wl, table = host_data()
    return (cp.asarray(n.astype(np.float32)), cp.asarray(k.astype(np.float32)),
            cp.asarray(wl.astype(np.float32)),
            cp.asarray(np.log(np.array(SIGMA_G)).astype(np.float32)),
            cp.asarray(np.ascontiguousarray(table)))


def _grid(manifest):
    (x0, x1, nx), (n0, n1, nn), (k0, k1, nk) = manifest["ln_xg"], manifest["n"], manifest["ln_k"]
    return (x0, (x1 - x0) / (nx - 1), int(nx), n0, (n1 - n0) / (nn - 1), int(nn),
            k0, (k1 - k0) / (nk - 1), int(nk))


def launch_modal_optics(vol, num, dz, *, mixing=1):
    """The kernel on ``ncell`` cells: vol (3*6, ncell) component volumes
    (m3 m-3), num (3, ncell) numbers (m-3), dz (ncell,) layer depths (m), all
    float32 device arrays.  Returns device arrays tau, ssa, g (4, ncell) and
    tau_lw (16, ncell)."""
    import cupy as cp
    from woof.core.kernels import get_kernel
    if mixing not in (1, 2):
        raise ValueError("mixing is 1 (volume average) or 2 (Maxwell Garnett)")
    vol = cp.ascontiguousarray(vol, dtype=cp.float32)
    num = cp.ascontiguousarray(num, dtype=cp.float32)
    dz = cp.ascontiguousarray(dz, dtype=cp.float32).reshape(-1)
    ncell = dz.size
    if vol.shape != (len(MODES) * len(CLASSES), ncell) or num.shape != (len(MODES), ncell):
        raise ValueError(f"vol must be (18, {ncell}) and num (3, {ncell})")
    manifest = host_data()[0]
    ri_n, ri_k, wl, lnsig, table = _device_data()
    x0, dx, nx, n0, dn, nn, k0, dk, nk = _grid(manifest)
    tau = cp.zeros((NSW, ncell), cp.float32)
    ssa = cp.zeros((NSW, ncell), cp.float32)
    g = cp.zeros((NSW, ncell), cp.float32)
    tau_lw = cp.zeros((NLW, ncell), cp.float32)
    threads = 64
    get_kernel("chem_modal_optics", "chem_modal_optics")(
        ((ncell + threads - 1) // threads,), (threads,),
        (vol, num, dz, ri_n, ri_k, wl, lnsig, table,
         np.float32(x0), np.float32(dx), np.int32(nx), np.float32(n0), np.float32(dn),
         np.int32(nn), np.float32(k0), np.float32(dk), np.int32(nk), np.int32(mixing),
         tau, ssa, g, tau_lw, np.int64(ncell)))
    return {"tau": tau, "ssa": ssa, "g": g, "tau_lw": tau_lw}


def mixture_index(v, n, k, mixing):
    """Host float64 twin of the kernel's mixing: v (6,) volumes, n/k (6,)
    component indices at one wavelength; returns the complex index."""
    m = n + 1j * k
    vt = v.sum()
    ec = CLASSES.index("elemental_carbon")
    if mixing == 2 and v[ec] > 0 and vt > v[ec]:
        rest = [c for c in range(len(CLASSES)) if c != ec]
        host = (v[rest] * m[rest]).sum() / v[rest].sum()
        eh, ei = host * host, m[ec] * m[ec]
        beta = (ei - eh) / (ei + 2 * eh)
        f = v[ec] / vt
        return np.sqrt(eh * (1 + 2 * f * beta) / (1 - f * beta))
    return (v * m).sum() / vt


def table_lookup(sigma_index, lnx, m):
    """Host float64 twin of the kernel's table interpolation and
    extrapolation: (E_abs, E_sca, g) per unit volume times lambda."""
    manifest, _n, _k, _wl, table = host_data()
    x0, dx, nx, n0, dn, nn, k0, dk, nk = _grid(manifest)
    x1, n1, k1 = x0 + dx * (nx - 1), n0 + dn * (nn - 1), k0 + dk * (nk - 1)
    lnxc = min(max(lnx, x0), x1)
    nre = min(max(m.real, n0), n1)
    lnk = min(max(np.log(max(m.imag, 1e-30)), k0), k1)
    f = [(lnxc - x0) / dx, (nre - n0) / dn, (lnk - k0) / dk]
    idx = [min(max(int(np.floor(v)), 0), s - 2) for v, s in zip(f, (nx, nn, nk))]
    w = [min(max(v - i, 0.0), 1.0) for v, i in zip(f, idx)]
    q = np.zeros(3)
    t = table[sigma_index].astype(np.float64)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                wt = (w[0] if a else 1 - w[0]) * (w[1] if b else 1 - w[1]) * (w[2] if c else 1 - w[2])
                q += wt * t[:, idx[0] + a, idx[1] + b, idx[2] + c]
    lea, les = q[0], q[1]
    if lnx < x0:
        les += 3.0 * (lnx - x0)
    elif lnx > x1:
        lea -= lnx - x1
        les -= lnx - x1
    return np.exp(lea), np.exp(les), min(max(q[2], -1.0), 1.0)


def reference_cell(v, num, dz, *, mixing=1):
    """Float64 host twin of one kernel cell: v (3, 6), num (3,), dz."""
    _manifest, n, k, wl, _t = host_data()
    out = {"tau": np.zeros(NSW), "ssa": np.ones(NSW), "g": np.zeros(NSW), "tau_lw": np.zeros(NLW)}
    for l, lam in enumerate(wl):
        ba = bs = bg = 0.0
        for mo in range(3):
            vt = v[mo].sum()
            if vt <= 0 or num[mo] <= 0:
                continue
            dg = (6 * vt / (np.pi * num[mo])) ** (1 / 3) * np.exp(-1.5 * np.log(SIGMA_G[mo]) ** 2)
            m = mixture_index(v[mo], n[:, l], k[:, l], mixing)
            ea, es, g = table_lookup(mo, np.log(np.pi * dg / lam), m)
            ba += vt / lam * ea
            bs += vt / lam * es
            bg += vt / lam * es * g
        if l < NSW:
            out["tau"][l] = (ba + bs) * dz
            out["ssa"][l] = bs / (ba + bs) if ba + bs > 0 else 1.0
            out["g"][l] = bg / bs if bs > 0 else 0.0
        else:
            out["tau_lw"][l - NSW] = ba * dz
    return out


def sorgam_volumes(aerosol, water, alt):
    """Component-class volumes (18, ncell) and mode numbers (3, ncell), m3
    m-3 and m-3, from the SORGAM rows (ug/kg and #/kg of dry air, mapping
    name -> (nz, ny, nx) float32), the aerosol water (h2oai, h2oaj) and alt
    (m3/kg): conc = q 1e-9 / alt."""
    import cupy as cp
    ncell = alt.size
    inv_alt = (1.0 / alt.reshape(-1).astype(cp.float64))
    vol = cp.zeros((len(MODES) * len(CLASSES), ncell), cp.float64)
    for name, mode, cls, rho in ROWS:
        vol[mode * len(CLASSES) + cls] += cp.maximum(aerosol[name].reshape(-1), 0) * (1e-9 / rho) * inv_alt
    for name, mode in WATER:
        vol[mode * len(CLASSES)] += cp.maximum(water[name].reshape(-1), 0) * (1e-9 / WATER_DENSITY) * inv_alt
    num = cp.stack([cp.maximum(aerosol[n].reshape(-1), 0) * inv_alt for n in NUMBERS])
    return vol.astype(cp.float32), num.astype(cp.float32)


def sorgam_optics(ctx):
    """The radiation clock's SORGAM optics on every cell: ``tauaer``,
    ``waer``, ``gaer`` (4, nz, ny, nx) and ``tauaerlw`` (16, nz, ny, nx),
    float32 device arrays, cached on the chem state for the current model
    time so radiation and photolysis share one evaluation."""
    chem = ctx.state.chem
    clock = getattr(ctx, "clock", None)
    now = float(clock.curr_secs if clock is not None else ctx.state.elapsed_seconds)
    cached = getattr(chem, "modal_optics", None)
    if cached is not None and getattr(chem, "modal_optics_time", None) == now:
        return cached
    from woof.core.chem_sorgam import AEROSOLS
    table = ctx.table
    aerosol = {name: ctx.field(table.row(name)) for name in AEROSOLS}
    water = {name: ctx.diag[f"sorgam_{name}"] for name, _m in WATER}
    alt = ctx.met("alt")
    nz, ny, nx = alt.shape
    vol, num = sorgam_volumes(aerosol, water, alt)
    mixing = MIXING[int(getattr(ctx.cfg, "aer_op_opt", 1) or 1)]
    out = launch_modal_optics(vol, num, ctx.met("dz8w").reshape(-1), mixing=mixing)
    shape4, shape16 = (NSW, nz, ny, nx), (NLW, nz, ny, nx)
    result = {"tauaer": out["tau"].reshape(shape4), "waer": out["ssa"].reshape(shape4),
              "gaer": out["g"].reshape(shape4), "tauaerlw": out["tau_lw"].reshape(shape16)}
    chem.modal_optics = result
    chem.modal_optics_time = now
    return result


def refusal(cfg):
    """Why the SORGAM optics cannot serve this configuration, or None."""
    if not int(getattr(cfg, "aer_ra_feedback", 0) or 0):
        return None
    op = int(getattr(cfg, "aer_op_opt", 0) or 0)
    if op not in MIXING:
        return (f"aer_op_opt={op}: the default build's SORGAM optics mix by volume "
                "(aer_op_opt = 1) or Maxwell Garnett (2); WRF-Chem's other rules are "
                "not carried, so the optics would silently use another")
    from woof.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
    if rrtmg_variant(cfg) != RRTMG_VARIANT_LEGACY:
        return ("aer_ra_feedback with the SORGAM aerosol feeds the legacy RRTMG "
                "(ra_rrtmg_variant = \"rrtmg_legacy\"); the RRTMGP path's MERRA "
                "tables describe GOCART rows only")
    return None
