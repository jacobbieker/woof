"""Check the OVATION Prime port (woof.spacewx.ovation) two ways.

``testdata``: against the IDL distribution's own outputs for the storm of
24-25 Oct 2011 (``testdata/ov_*_2011_*.txt`` in the pinned zip, run with the
distribution's OMNI2 file).  It reports the largest cell difference and the
hemispheric power of each output.

``poes``: against observations, the NOAA POES/MetOp SEM-2 Total Energy
Detector.  The 16-second averaged level-2 text files are at NOAA NCEI,
``https://www.ncei.noaa.gov/data/poes-metop-space-environment-monitor/access/l2/v01r00/txt/YYYY/SAT/poes_XNN_YYYYMMDD.txt``
(1998-2014; public).  Their ``ted`` column is the 0.05-20 keV energy flux
(erg cm-2 s-1) and ``econtr`` its electron fraction.  The electron energy flux
``ted x econtr`` at each record's 120-km field-line foot (``folat``,
``folon``) is compared with the model's diffuse + monoenergetic + broadband
electron energy flux at the same point and time, on woof's IGRF dipole frame.

Two statistics are reported per hemisphere:
- the records poleward of 50 deg geomagnetic: means, and the correlation of
  log10(0.01 + flux);
- polar passes, each the run of records poleward of 50 deg in one hemisphere:
  the flux summed along the pass.  This is the quantity a hemispheric power
  index is built from, and it is less sensitive than single records to where
  an arc happens to sit.  It reports the correlation and the median of the
  model's sum over the observed one.

OVATION is a statistical model of mean precipitation, so single records
scatter widely about it.

    python -m tools.spacewx_ovation_validate testdata
    python -m tools.spacewx_ovation_validate poes --poes poes_n19_20111024.txt poes_n18_20111024.txt \\
        --solar-wind omni2_2011.dat --json out.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np


def testdata(args) -> dict:
    """Per output file: its build (2.3.0 or NOAA's later trunk, from the path in
    its first line), the largest cell difference and, for energy flux, the
    hemispheric power."""
    from woof.spacewx import assets, ovation, solarwind

    path = Path(args.zip) if getattr(args, "zip", None) else assets.fetch_pinned(ovation.OVATION_ZIP)
    coef = ovation.load(path)
    archive = zipfile.ZipFile(path)
    names = {Path(n).name: n for n in archive.namelist()}
    text = archive.read(names["omni2_2011.dat"]).decode("ascii")
    # The distribution's OMNI2 file predates word 55; pad it for the reader.
    wind = solarwind.parse_omni2("\n".join(f"{ln} 0" for ln in text.splitlines() if ln.strip()), "omni2_2011")
    rows = []
    for name in sorted(names):
        m = re.fullmatch(r"ov_(diff|mono|wave|ions)_(Eflux|flux)_2011_(\d\d)(\d\d)_(\d\d)00\.txt", name)
        if not m:
            continue
        kind, what, mo, dy, hh = m.groups()
        when = datetime(2011, int(mo), int(dy), int(hh))
        lines = archive.read(names[name]).decode("ascii").splitlines()
        build = "trunk" if "trunk" in lines[0] else "2.3.0"
        lines = lines[1:]
        cells = np.array([[float(x) for x in ln.split()] for ln in lines
                          if len(ln.split()) == 3 and re.fullmatch(r"[\d.\seE+-]+", ln)])
        number = what == "flux"
        grid = ovation.combine_hemispheres(ovation.fit_grid(
            coef, ovation.coupling(wind, when), when.timetuple().tm_yday, ovation.TYPES.index(kind), number))
        i = (cells[:, 0] * 4 + 1e-3).astype(int)
        j = ((cells[:, 1] - 50.0) * 2 + 1e-3).astype(int)
        err = np.abs(grid[i, j] - cells[:, 2])
        row = {"file": name, "build": build, "number": number, "max_abs_diff": float(err.max()),
               "max_ref": float(np.abs(cells[:, 2]).max())}
        if not number:
            row["hp_gw"] = ovation.hemispheric_power(grid)
            row["hp_ref_gw"] = float(next(ln for ln in lines if ln.startswith("Hemispherical power")).split(":")[1])
        rows.append(row)
        print(f"{name:34s} {build:5s} max |diff| {row['max_abs_diff']:9.3g} (of {row['max_ref']:9.3g})"
              + (f"  HP {row['hp_gw']:8.3f} ref {row['hp_ref_gw']:8.3f} GW" if not number else ""))
    return {"zip": str(path), "outputs": rows}


def read_poes(path) -> dict:
    """Times, foot latitude/longitude and electron energy flux of a level-2 text file."""
    lines = Path(path).read_text(encoding="ascii", errors="replace").splitlines()
    head = lines[0].split()
    need = ("year", "mo", "dy", "hr", "mi", "second", "folat", "folon", "ted", "econtr")
    if not all(k in head for k in need):
        raise ValueError(f"{path}: not a POES level-2 text file (header {head[:8]}...)")
    col = {k: head.index(k) for k in need}
    data = np.array([[float(w[col[k]]) for k in need] for w in (ln.split() for ln in lines[1:])
                     if len(w) == len(head)]).reshape(-1, len(need))
    t = [datetime(int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4])) + timedelta(seconds=float(r[5]))
         for r in data]
    ok = (data[:, 8] >= 0.0) & (data[:, 9] >= 0.0) & (data[:, 9] <= 1.0) & (np.abs(data[:, 6]) <= 90.0)
    return {"time": np.array(t)[ok], "lat": data[ok, 6], "lon": data[ok, 7],
            "eflux": (data[:, 8] * data[:, 9])[ok], "total": data[ok, 8], "source": str(path)}


def _corr(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if a.size > 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan")


def poes(args) -> dict:
    from woof.spacewx import aurora, igrf, ovation, solarwind

    sats = [s for s in (read_poes(p) for p in args.poes) if s["time"].size]
    if not sats:
        raise SystemExit("no POES records with a valid electron flux")
    first = min(s["time"].min() for s in sats)
    last = max(s["time"].max() for s in sats)
    wind = solarwind.load(args.solar_wind, first - timedelta(hours=4), last + timedelta(hours=1))
    coef = ovation.load(args.ovation)
    frame = aurora.dipole_frame(igrf.load(), first)
    out = {"poes": [s["source"] for s in sats], "solar_wind": f"{wind.source} sha256:{wind.sha256}",
           "frame": "IGRF-14 centred dipole", "hemispheres": {}}
    passes = {"north": ([], [], [], []), "south": ([], [], [], [])}
    points = {"north": ([], []), "south": ([], [])}
    for sat in sats:
        p = aurora._unit(sat["lat"], sat["lon"])
        sat["mlat"] = np.rad2deg(np.arcsin(np.clip(p @ frame[0], -1.0, 1.0)))
        sat["mlon"] = np.rad2deg(np.arctan2(p @ frame[2], p @ frame[1]))
        sat["model"] = np.full(sat["mlat"].size, np.nan)
        sat["model_ion"] = np.zeros(sat["mlat"].size)
    # All satellites' records in time order, so that only one minute's grids
    # are held at a time.
    order = sorted((when, n, k) for n, sat in enumerate(sats) for k, when in enumerate(sat["time"]))
    key, grids = None, None
    for when, n, k in order:
        sat = sats[n]
        if when.replace(second=0, microsecond=0) != key:
            key = when.replace(second=0, microsecond=0)
            try:
                grids = ovation.hemisphere_grids(coef, ovation.coupling(wind, key), key)
            except ovation.OvationError:
                grids = None
            mlon_sun = aurora.sun_magnetic_longitude(frame, key)
        if grids is None:
            continue
        mlt = np.mod(12.0 + (sat["mlon"][k:k + 1] - mlon_sun) / 15.0, 24.0)
        cols = ovation.energy_columns(grids, sat["mlat"][k:k + 1], mlt)
        sat["model"][k], sat["model_ion"][k] = cols["eflux"][0], cols["eflux_ion"][0]
    for sat in sats:
        mlat, model, model_ion = sat["mlat"], sat["model"], sat["model_ion"]
        polar = (np.abs(mlat) >= 50.0) & np.isfinite(model)
        for h, sign in (("north", 1.0), ("south", -1.0)):
            sel = polar & (np.sign(mlat) == sign)
            points[h][0].append(sat["eflux"][sel])
            points[h][1].append(model[sel])
            idx = np.flatnonzero(sel)
            if idx.size == 0:
                continue
            gaps = np.flatnonzero(np.diff([t.timestamp() for t in sat["time"][idx]]) > 300.0)
            for run in np.split(idx, gaps + 1):
                if run.size >= 20:          # at least about 5 minutes poleward of 50 deg
                    passes[h][0].append(float(sat["eflux"][run].sum()))
                    passes[h][1].append(float(model[run].sum()))
                    passes[h][2].append(float(sat["total"][run].sum()))
                    passes[h][3].append(float((model + model_ion)[run].sum()))
    for h in ("north", "south"):
        obs, mod = (np.concatenate(v) if v else np.zeros(0) for v in points[h])
        po, pm, pt, pmt = (np.asarray(v) for v in passes[h])
        good = po > 0
        stats = {
            "records": int(obs.size), "obs_mean": float(obs.mean()) if obs.size else float("nan"),
            "model_mean": float(mod.mean()) if mod.size else float("nan"),
            "log_corr": _corr(np.log10(0.01 + obs), np.log10(0.01 + mod)),
            "passes": int(po.size), "pass_corr": _corr(po, pm),
            "pass_median_ratio": float(np.median(pm[good] / po[good])) if good.any() else float("nan"),
            # Electrons and ions together against the whole TED flux.
            "pass_total_corr": _corr(pt, pmt),
            "pass_total_median_ratio": float(np.median(pmt[pt > 0] / pt[pt > 0])) if (pt > 0).any() else float("nan"),
        }
        out["hemispheres"][h] = stats
        print(f"{h}: {stats['records']} records, mean obs {stats['obs_mean']:.3f} model {stats['model_mean']:.3f} "
              f"erg/cm2/s, log r {stats['log_corr']:.2f}; {stats['passes']} passes, r {stats['pass_corr']:.2f}, "
              f"median model/obs {stats['pass_median_ratio']:.2f}; with ions r {stats['pass_total_corr']:.2f}, "
              f"median {stats['pass_total_median_ratio']:.2f}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.spacewx_ovation_validate", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("testdata", help="against the IDL distribution's own test outputs")
    t.add_argument("--zip", help="the distribution's zip (default: the pinned copy, fetched)")
    p = sub.add_parser("poes", help="against POES/MetOp TED electron energy flux")
    p.add_argument("--poes", nargs="+", required=True, help="NCEI level-2 text files (poes_XNN_YYYYMMDD.txt)")
    p.add_argument("--solar-wind", default="omni", help="'omni' (fetched) or OMNI2 file(s) joined by ':'")
    p.add_argument("--ovation", help="the coefficients' zip or premodel directory (default: the pinned zip)")
    for q in (t, p):
        q.add_argument("--json", type=Path, help="write the statistics here")
    args = parser.parse_args(argv)
    doc = testdata(args) if args.mode == "testdata" else poes(args)
    if args.json is not None:
        args.json.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
