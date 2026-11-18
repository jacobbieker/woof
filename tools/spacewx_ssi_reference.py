"""Refit REF_BAND_FRACTION (woof/spacewx/solar_spectrum.py) from NOAA CDR SSI files.

usage: python -m tools.spacewx_ssi_reference FILE.nc [FILE.nc ...]

Prints the mean over the files' days of S_b / TSI for RRTMG SW's 14 bands
(16-29), with the sum and each band's range.  The shipped values are the mean
over 2009-01-01 to 2019-12-31 (solar cycle 24) of ssi_v03r00_daily_s{Y}*.nc.
"""
from __future__ import annotations

import sys

import numpy as np

from woof.spacewx.solar_spectrum import BANDS_CM1, read_series


def main(argv=None) -> int:
    paths = list(sys.argv[1:] if argv is None else argv)
    if not paths:
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
