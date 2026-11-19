"""Refit REF_BAND_FRACTION (woof/spacewx/solar_spectrum.py) from NOAA CDR SSI files.

usage: python -m tools.spacewx_ssi_reference FILE.nc [FILE.nc ...]
       python -m tools.spacewx_ssi_reference --spectrum DIR OUT.json

Prints the mean over the files' days of S_b / TSI for RRTMG SW's 14 bands
(16-29), with the sum and each band's range.  The shipped values are the mean
over 2009-01-01 to 2019-12-31 (solar cycle 24) of ssi_v03r00_daily_s{Y}*.nc.

``--spectrum`` rebuilds woof/data/spacewx/ssi_reference_v03_2009_2019.json,
the same mean per CDR bin below 760 nm (photolysis's reference), from LISIRD's
LaTiS CSV copies of the same product, DIR/ssi_{2009..2019}.csv and DIR/tsi.csv::

    https://lasp.colorado.edu/lisird/latis/dap/nnl_ssi_P1D.csv?time>=Y-01-01&time<Y+1-01-01&wavelength<760&project(time,wavelength,irradiance)
    https://lasp.colorado.edu/lisird/latis/dap/nnl_tsi_P1D.csv?time>=2009-01-01&time<2020-01-01

It checks the result against REF_BAND_FRACTION in bands 25-28 (inside 760 nm)
and writes nothing unless every band agrees to its 8 printed decimals.
"""
from __future__ import annotations

import sys

import numpy as np

from woof.spacewx.solar_spectrum import BANDS_CM1, BANDS_NM, REF_BAND_FRACTION, band_irradiance, read_series

DESCRIPTION = ("S(lambda)/TSI, nm-1: the NOAA CDR SSI v03 (NNLSSI, standard resolution) daily spectrum over "
               "its TSI, averaged over 2009-01-01..2019-12-31 ({days} days), on the CDR's own bins below 760 nm. "
               "The same days and product as REF_BAND_FRACTION (woof/spacewx/solar_spectrum.py); built by "
               "tools/spacewx_ssi_reference.py --spectrum. NOAA CDR: open data, no restrictions.")


def spectrum(folder: str, out: str) -> int:
    import json

    tsi = np.loadtxt(f"{folder}/tsi.csv", delimiter=",", skiprows=1, usecols=(0, 1))
    tmap = dict(zip(tsi[:, 0], tsi[:, 1]))
    acc, wl, days = None, None, 0
    for year in range(2009, 2020):
        a = np.loadtxt(f"{folder}/ssi_{year}.csv", delimiter=",", skiprows=1)
        t, w = np.unique(a[:, 0]), np.unique(a[:, 1])
        wl = w if wl is None else wl
        if not np.array_equal(w, wl) or a.shape[0] != t.size * w.size:
            raise SystemExit(f"ssi_{year}.csv: not a full day x wavelength grid like 2009's")
        s = a[:, 2].reshape(t.size, w.size)
        if not np.array_equal(a[:, 1].reshape(t.size, w.size)[0], w):
            raise SystemExit(f"ssi_{year}.csv: not time-major")
        tot = np.array([tmap.get(x, np.nan) for x in t])
        ok = np.isfinite(tot) & np.all(np.isfinite(s), axis=1)
        f = (s[ok] / tot[ok, None]).sum(0)
        acc = f if acc is None else acc + f
        days += int(ok.sum())
    if days == 0:
        raise SystemExit("no day has both an SSI spectrum and a TSI (do the CSVs' times match?)")
    ref = acc / days
    width = np.where(wl < 750.0, 1.0, 5.0)
    bands = band_irradiance(wl, width, ref, BANDS_NM[9:13])
    for b, v in enumerate(bands, start=9):
        print(f"band {16 + b}: spectrum {v:.8f} REF_BAND_FRACTION {REF_BAND_FRACTION[b]:.8f} ratio "
              f"{v / REF_BAND_FRACTION[b]:.6f}")
    if not (np.all(np.isfinite(ref)) and np.all(ref > 0)
            and np.allclose(bands, REF_BAND_FRACTION[9:13], rtol=0, atol=5e-9)):
        raise SystemExit("the spectrum does not reproduce REF_BAND_FRACTION in bands 25-28; nothing written")
    doc = {"description": DESCRIPTION.format(days=days), "days": days, "wavelength_nm": wl.tolist(),
           "width_nm": width.tolist(), "fraction_per_nm": [float(f"{x:.8e}") for x in ref]}
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"{days} days, {wl.size} bins -> {out}")
    return 0


def main(argv=None) -> int:
    paths = list(sys.argv[1:] if argv is None else argv)
    if paths[:1] == ["--spectrum"] and len(paths) == 3:
        return spectrum(paths[1], paths[2])
    if not paths or paths[0].startswith("--"):
        print(__doc__)
        return 2
    series = read_series(paths)
    frac = series.band / series.tsi[:, None]
    mean = frac.mean(axis=0)
    print(f"{series.days.size} days; in-band share of TSI {mean.sum():.6f}")
    for b, (lo, hi) in enumerate(BANDS_CM1):
        rel = (frac[:, b].max() - frac[:, b].min()) / mean[b]
        print(f"band {16 + b:2d} {lo:7.0f}-{hi:7.0f} cm-1  {mean[b]:.8f}  range/mean {rel:.2e}")
    print("REF_BAND_FRACTION = (" + ", ".join(f"{v:.8f}" for v in mean) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
