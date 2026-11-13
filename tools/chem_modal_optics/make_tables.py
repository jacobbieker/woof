"""Build the clean-room modal aerosol optics data (WOOF P9).

    pixi run -e mlx python -m tools.chem_modal_optics.make_tables          # write
    pixi run -e mlx python -m tools.chem_modal_optics.make_tables --check  # compare

Writes ``woof/data/chem/modal_optics/``:

* ``refractive_index.v1.json``: the complex refractive index of each SORGAM
  component class at WRF-Chem's four shortwave wavelengths (300, 400, 600,
  999 nm) and at the centre wavenumber of each of RRTMG's 16 longwave bands.
  Tabulated classes are interpolated (linear in ln lambda for n and ln k)
  from refractiveindex.info's database, CC0-1.0, at the pinned commit
  :data:`RII_COMMIT` (each file's sha256 is checked and recorded):
  water, Hale and Querry 1973 (Appl. Opt. 12, 555); sea salt, NaCl, Querry
  1987 (CRDEC-CR-88009); mineral and unspeciated particulate, fused silica,
  Franta et al. 2016 (Proc. SPIE 9890, 989014), a silicate proxy.  The
  others are wavelength-independent literature values: elemental carbon
  1.95 + 0.79i (Bond and Bergstrom 2006, Aerosol Sci. Technol. 40, 27,
  their recommended value); inorganic salts (sulfate, nitrate, ammonium)
  1.52 + 0i (Toon, Pollack and Khare 1976, JGR 81, 5733, ammonium sulfate
  in the visible); organics 1.45 + 0i (secondary organic aerosol, Kim and
  Paulson 2013, ACP 13, 7711, measured 1.4-1.5).
* ``mie_lognormal.v1.npy``: float32 (3, 3, NX, NN, NK): for SORGAM's three
  fixed geometric standard deviations (Aitken 1.70, accumulation 2.00,
  coarse 2.50; module_aerosols_sorgam's xxlsgn/xxlsga/xxlsgc), the
  lognormal-number-distribution optics per unit particle volume as a
  function of the median size parameter x_g = pi D_g / lambda and the
  refractive index n + ik: ln E_abs, ln E_sca and g, where
  E = beta lambda / V = (3 pi / 2) int Q x^2 n(ln x) / int x^3 n(ln x)
  (beta the extinction coefficient, V the particle volume per unit air
  volume).  Q and g come from WOOF's clean-room Mie code
  (woof.radtran.mie: Bohren and Huffman 1983 ch. 4, Wiscombe 1980), with
  g Q_sca from Bohren and Huffman eq. 4.62.
* ``MANIFEST.json``: the grids, the sources and the sha256 of both files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "woof" / "data" / "chem" / "modal_optics"
RII_COMMIT = "e6d15e364f70de1cad8a96a20f75689f2fba8b21"
RII_URL = ("https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/"
           f"{RII_COMMIT}/database/data/main/")
RII_FILES = {
    "water": ("H2O/nk/Hale.yml", "Hale and Querry 1973, Appl. Opt. 12, 555-563"),
    "sea_salt": ("NaCl/nk/Querry.yml", "Querry 1987, CRDEC-CR-88009 (NaCl)"),
    "mineral": ("SiO2/nk/Franta.yml", "Franta et al. 2016, Proc. SPIE 9890, 989014 (fused silica)"),
}
CONSTANT = {
    "inorganic": (1.52, 0.0, "Toon, Pollack and Khare 1976, JGR 81, 5733 (ammonium sulfate, visible)"),
    "organic": (1.45, 0.0, "Kim and Paulson 2013, ACP 13, 7711 (SOA, 1.4-1.5)"),
    "elemental_carbon": (1.95, 0.79, "Bond and Bergstrom 2006, Aerosol Sci. Technol. 40, 27"),
}
#: Component classes in the kernel's order.
CLASSES = ("water", "inorganic", "sea_salt", "organic", "elemental_carbon", "mineral")
SW_UM = (0.300, 0.400, 0.600, 0.999)
#: RRTMG longwave band limits, cm-1 (woof/core/rrtmg_lw.py band docstrings).
LW_BANDS = ((10, 350), (350, 500), (500, 630), (630, 700), (700, 820), (820, 980),
            (980, 1080), (1080, 1180), (1180, 1390), (1390, 1480), (1480, 1800),
            (1800, 2080), (2080, 2250), (2250, 2380), (2380, 2600), (2600, 3250))
LW_UM = tuple(1.0e4 / (0.5 * (lo + hi)) for lo, hi in LW_BANDS)

SIGMA = (1.70, 2.00, 2.50)
LNX = (math.log(1.0e-3), math.log(100.0), 47)
#: Wide enough for the infrared resonances of the tabulated classes (NaCl
#: near 60 um, silica near 9 and 20 um).
NRE = (0.30, 3.20, 59)
LNK = (math.log(1.0e-5), math.log(4.0), 18)
#: Fine size-parameter grid of the size integral, and the largest x the Mie
#: series is summed at (beyond it the efficiencies are held at their value
#: there: within about 1 % of their geometric-optics limits).
FINE = (math.log(1.0e-5), math.log(1.0e5), 0.01)
X_MIE_MAX = 3000.0
CACHE = Path.home() / ".cache" / "woof" / "modal-optics-src"


def _fetch(rel: str) -> bytes:
    path = CACHE / RII_COMMIT / rel
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(RII_URL + rel, timeout=60) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def _tabulated(raw: bytes):
    text = raw.decode()
    block = text.split("data: |", 1)[1]
    rows = []
    for line in block.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0][0].isdigit():
            rows.append([float(p) for p in parts])
        elif rows and line.strip() and not line.strip()[0].isdigit():
            break
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2]


def _interp(lam, n, k, at):
    ll = np.log(lam)
    nn = np.interp(math.log(at), ll, n)
    kk = math.exp(np.interp(math.log(at), ll, np.log(np.maximum(k, 1e-12))))
    return float(nn), float(kk if kk > 1e-11 else 0.0)


def refractive_indices():
    wl = SW_UM + LW_UM
    table = {"wavelength_um": [round(w, 6) for w in wl], "classes": {}, "sources": {}}
    for name in CLASSES:
        if name in RII_FILES:
            rel, cite = RII_FILES[name]
            raw = _fetch(rel)
            lam, n, k = _tabulated(raw)
            if not (lam[0] <= min(wl) and lam[-1] >= max(wl)):
                raise SystemExit(f"{rel} covers {lam[0]}-{lam[-1]} um, not {min(wl)}-{max(wl)}")
            vals = [_interp(lam, n, k, w) for w in wl]
            table["sources"][name] = {"citation": cite, "database": "refractiveindex.info (CC0-1.0)",
                                      "file": rel, "commit": RII_COMMIT,
                                      "sha256": hashlib.sha256(raw).hexdigest()}
        else:
            n0, k0, cite = CONSTANT[name]
            vals = [(n0, k0)] * len(wl)
            table["sources"][name] = {"citation": cite, "wavelength_independent": True}
        table["classes"][name] = {"n": [round(v[0], 6) for v in vals],
                                  "k": [float(f"{v[1]:.6g}") for v in vals]}
    return table


def mie_on_grid(x, m):
    """Q_ext, Q_sca and g Q_sca on a 1-D grid of size parameters."""
    from woof.radtran import mie

    qe = np.empty(x.size)
    qs = np.empty(x.size)
    gq = np.empty(x.size)
    edges = np.unique(np.concatenate([[0], np.searchsorted(x, [0.1, 1, 10, 30, 100, 300, 1000]), [x.size]]))
    for lo, hi in zip(edges[:-1], edges[1:]):
        for c0 in range(lo, hi, 128):
            sl = slice(c0, min(hi, c0 + 128))
            xs = x[sl]
            e, s, a, b = mie.efficiencies(xs, m)
            n = np.arange(1, a.shape[0] + 1)[:, None]
            # Bohren and Huffman eq. 4.62 (terms past each x's truncation are 0).
            t1 = (n[:-1] * (n[:-1] + 2) / (n[:-1] + 1)
                  * (a[:-1] * np.conj(a[1:]) + b[:-1] * np.conj(b[1:])).real).sum(axis=0)
            t2 = ((2 * n + 1) / (n * (n + 1)) * (a * np.conj(b)).real).sum(axis=0)
            qe[sl], qs[sl], gq[sl] = e, s, 4.0 / xs ** 2 * (t1 + t2)
    return qe, qs, gq


def lognormal_table():
    lnxg = np.linspace(*LNX)
    nre = np.linspace(*NRE)
    lnk = np.linspace(*LNK)
    lnx = np.arange(FINE[0], FINE[1] + FINE[2] / 2, FINE[2])
    x = np.exp(lnx)
    mie_n = int(np.searchsorted(x, X_MIE_MAX, side="right"))
    out = np.zeros((len(SIGMA), 3, lnxg.size, nre.size, lnk.size))
    # trapezoid weights in ln x
    tw = np.full(x.size, FINE[2])
    tw[0] = tw[-1] = 0.5 * FINE[2]
    for i, n_re in enumerate(nre):
        for j, lk in enumerate(lnk):
            qe, qs, gq = mie_on_grid(x[:mie_n], complex(n_re, math.exp(lk)))
            pad = x.size - mie_n
            qe = np.concatenate([qe, np.full(pad, qe[-1])])
            qs = np.concatenate([qs, np.full(pad, qs[-1])])
            gq = np.concatenate([gq, np.full(pad, gq[-1])])
            for s, sg in enumerate(SIGMA):
                ls = math.log(sg)
                w = np.exp(-0.5 * ((lnx[None, :] - lnxg[:, None]) / ls) ** 2) * tw
                vol = w @ x ** 3
                ext = 1.5 * math.pi * (w @ (qe * x ** 2)) / vol
                sca = 1.5 * math.pi * (w @ (qs * x ** 2)) / vol
                g = (w @ (gq * x ** 2)) / (w @ (qs * x ** 2))
                out[s, 0, :, i, j] = np.log(np.maximum(ext - sca, 1e-30))
                out[s, 1, :, i, j] = np.log(np.maximum(sca, 1e-30))
                out[s, 2, :, i, j] = g
        print(f"n = {n_re:.3f} done", file=sys.stderr, flush=True)
    return out.astype(np.float32)


def manifest(ri_bytes, table_bytes):
    return {"schema": "woof.chem.modal_optics.v1",
            "sigma_g": list(SIGMA),
            "classes": list(CLASSES),
            "shortwave_um": list(SW_UM),
            "longwave_band_cm-1": [list(b) for b in LW_BANDS],
            "table_axes": ["sigma (Aitken, accumulation, coarse)", "quantity (ln E_abs, ln E_sca, g)",
                           "ln x_g", "n", "ln k"],
            "ln_xg": list(LNX), "n": list(NRE), "ln_k": list(LNK),
            "fine_ln_x": list(FINE), "x_mie_max": X_MIE_MAX,
            "generator": "tools/chem_modal_optics/make_tables.py",
            "sha256": {"refractive_index.v1.json": hashlib.sha256(ri_bytes).hexdigest(),
                       "mie_lognormal.v1.npy": hashlib.sha256(table_bytes).hexdigest()}}


def _npy_bytes(a):
    import io
    buf = io.BytesIO()
    np.save(buf, a, allow_pickle=False)
    return buf.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="rebuild and compare with the committed files (tolerance 1e-5)")
    args = ap.parse_args(argv)
    ri = refractive_indices()
    ri_bytes = (json.dumps(ri, indent=1, sort_keys=True) + "\n").encode()
    table = lognormal_table()
    if args.check:
        old_ri = json.loads((OUT / "refractive_index.v1.json").read_text())
        assert old_ri == ri, "refractive_index.v1.json differs from its sources"
        old = np.load(OUT / "mie_lognormal.v1.npy")
        diff = float(np.abs(old.astype(np.float64) - table).max())
        print(f"table max abs difference {diff:.3g}")
        return 0 if diff < 1e-5 else 1
    tb = _npy_bytes(table)
    (OUT / "refractive_index.v1.json").write_bytes(ri_bytes)
    (OUT / "mie_lognormal.v1.npy").write_bytes(tb)
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest(ri_bytes, tb), indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
