"""Grade the L3a ionosphere's vertical TEC against GNSS global ionosphere maps.

The reference is an IONEX file (Schaer, Gurtner & Feltens 1998, IONEX 1.0):
IGS, CODE, JPL, ESA or UPC global ionosphere maps from NASA CDDIS
(``https://cddis.nasa.gov/archive/gnss/products/ionex/YYYY/DDD/``, Earthdata
login).  Each TEC map is compared node by node with the model evaluated at
the same nodes and epoch; statistics are weighted by cos(latitude), overall
and in three bands of geographic latitude (|lat| < 30, 30-60, > 60).

GIM TEC counts the plasmasphere up to GPS orbit (about 20 200 km); the model
integrates to ``--alt-max`` (2000 km by default), so a positive GIM-minus-
model residual of a few TECU is expected even for a perfect ionosphere.

    python -m tools.spacewx_gim_validate --ionex IGS0OPSRAP_20241310000_01D_02H_GIM.INX.gz \
        --indices Kp_ap_Ap_SN_F107_since_1932.txt --json out.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import gzip
import json
from pathlib import Path
import sys

import numpy as np

BANDS = (("all", 0.0, 90.0), ("low |lat|<30", 0.0, 30.0), ("mid 30-60", 30.0, 60.0), ("high >60", 60.0, 90.0))


class IonexError(ValueError):
    pass


def _label(line: str) -> str:
    return line[60:].strip()


def read_ionex(path) -> dict:
    """Epochs, node latitudes/longitudes and TEC maps (TECU, NaN where missing)."""
    path = Path(path)
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    lines = raw.decode("ascii", "replace").splitlines()
    exponent = 0
    lat_spec = lon_spec = None
    i = 0
    while i < len(lines) and _label(lines[i]) != "END OF HEADER":
        label = _label(lines[i])
        if label == "EXPONENT":
            exponent = int(lines[i][:6])
        elif label == "LAT1 / LAT2 / DLAT":
            lat_spec = [float(w) for w in lines[i][:60].split()]
        elif label == "LON1 / LON2 / DLON":
            lon_spec = [float(w) for w in lines[i][:60].split()]
        i += 1
    if lat_spec is None or lon_spec is None:
        raise IonexError(f"{path}: no LAT1/LON1 grid in the header")
    nlat = int(round((lat_spec[1] - lat_spec[0]) / lat_spec[2])) + 1
    nlon = int(round((lon_spec[1] - lon_spec[0]) / lon_spec[2])) + 1
    lat = lat_spec[0] + lat_spec[2] * np.arange(nlat)
    lon = lon_spec[0] + lon_spec[2] * np.arange(nlon)
    epochs, maps = [], []
    while i < len(lines):
        if _label(lines[i]) != "START OF TEC MAP":
            i += 1
            continue
        i += 1
        if _label(lines[i]) != "EPOCH OF CURRENT MAP":
            raise IonexError(f"{path}: TEC map without an epoch at line {i + 1}")
        y, mo, d, h, mi, s = (int(w) for w in lines[i][:36].split())
        epoch = datetime(y, mo, d, h, mi, s)
        i += 1
        scale = 10.0 ** exponent
        grid = np.full((nlat, nlon), np.nan)
        row = -1
        values: list[int] = []
        while _label(lines[i]) != "END OF TEC MAP":
            label = _label(lines[i])
            if label == "EXPONENT":
                scale = 10.0 ** int(lines[i][:6])
            elif label == "LAT/LON1/LON2/DLON/H":
                if row >= 0:
                    grid[row] = _row(values, nlon, scale, path)
                row += 1
                values = []
            else:
                values.extend(int(lines[i][k:k + 5]) for k in range(0, len(lines[i].rstrip()), 5))
            i += 1
        grid[row] = _row(values, nlon, scale, path)
        if row != nlat - 1:
            raise IonexError(f"{path}: map {epoch} has {row + 1} latitude rows, expected {nlat}")
        epochs.append(epoch)
        maps.append(grid)
        i += 1
    if not maps:
        raise IonexError(f"{path}: no TEC maps")
    return {"epochs": epochs, "lat": lat, "lon": lon, "tec": np.stack(maps)}


def _row(values, nlon, scale, path):
    if len(values) != nlon:
        raise IonexError(f"{path}: a latitude row has {len(values)} values, expected {nlon}")
    out = np.asarray(values, dtype=np.float64)
    return np.where(out == 9999, np.nan, out * scale)


def statistics(model: np.ndarray, reference: np.ndarray, lat: np.ndarray) -> dict:
    """cos(lat)-weighted bias (model minus reference), RMSE and correlation per band."""
    lat2 = np.broadcast_to(lat[None, :, None], model.shape)
    out = {}
    for name, lo, hi in BANDS:
        band = (np.abs(lat2) >= lo) & ((np.abs(lat2) < hi) | (hi >= 90.0))
        mask = np.isfinite(model) & np.isfinite(reference) & band
        if not mask.any():
            out[name] = {"nodes": 0}
            continue
        w = np.cos(np.deg2rad(lat2[mask]))
        m, r = model[mask], reference[mask]
        diff = m - r
        mm, rm = np.average(m, weights=w), np.average(r, weights=w)
        cov = np.average((m - mm) * (r - rm), weights=w)
        corr = cov / np.sqrt(np.average((m - mm) ** 2, weights=w) * np.average((r - rm) ** 2, weights=w))
        out[name] = {"bias": float(np.average(diff, weights=w)), "rmse": float(np.sqrt(np.average(diff**2, weights=w))),
                     "corr": float(corr), "model_mean": float(mm), "gim_mean": float(rm), "nodes": int(mask.sum())}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ionex", type=Path, nargs="+", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--indices", help="'gfz' or a file in GFZ's Kp/ap/F10.7 format")
    source.add_argument("--f107", type=float)
    parser.add_argument("--f107-mode", choices=("daily", "81day"), default="81day")
    parser.add_argument("--maps", choices=("ccir", "ursi"), default="ccir")
    parser.add_argument("--alt-max", type=float, default=2000.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    from woof.spacewx import indices
    from woof.spacewx.ionosphere import Ionosphere, receipt
    import cupy as cp  # after woof: the Metal backend installs its shim on import

    gims = [(path, read_ionex(path)) for path in args.ionex]
    # The F10.7 driver at the last map reads the UT day before it.
    last = max(max(gim["epochs"]) for _, gim in gims)
    record = None if args.f107 is not None else indices.load(args.indices, last - timedelta(days=1))
    report = {"files": [], "per_file": {}}
    all_model, all_ref, lat_ref, lon_ref = [], [], None, None
    previous_last = None    # a daily file's 24:00 map is the next file's 00:00 map: count it once
    for path, gim in gims:
        lat2, lon2 = np.meshgrid(gim["lat"], gim["lon"], indexing="ij")
        epochs = sorted(set(gim["epochs"]))
        iono = Ionosphere(lat2, lon2, when=epochs[0], family=args.maps, alt_hi=args.alt_max,
                          nalt=int(round(args.alt_max - 60.0)))
        model, reference, kept = [], [], []
        for idx, epoch in enumerate(gim["epochs"]):
            if gim["epochs"].index(epoch) != idx:
                continue
            kept.append(epoch)
            f107 = args.f107 if record is None else record.f107_driver(epoch, args.f107_mode)
            model.append(cp.asnumpy(iono.evaluate(epoch, f107)["tec"]).reshape(lat2.shape).astype(np.float64))
            reference.append(gim["tec"][idx])
        model, reference = np.stack(model), np.stack(reference)
        stats = statistics(model, reference, gim["lat"])
        report["files"].append(str(path.name))
        report["per_file"][path.name] = {"first": epochs[0].isoformat(), "maps": len(model), "stats": stats}
        fresh = [i for i, epoch in enumerate(kept) if not (i == 0 and epoch == previous_last)]
        previous_last = kept[-1]
        if lat_ref is not None and not (np.array_equal(lat_ref, gim["lat"])
                                        and np.array_equal(lon_ref, gim["lon"])):
            raise IonexError(f"{path}: its latitude/longitude grid differs from the earlier files'")
        all_model.append(model[fresh])
        all_ref.append(reference[fresh])
        lat_ref, lon_ref = gim["lat"], gim["lon"]
        s = stats["all"]
        print(f"{path.name}: {len(model)} maps  bias {s['bias']:+6.2f}  rmse {s['rmse']:6.2f}  corr {s['corr']:5.3f}  "
              f"model {s['model_mean']:6.2f}  GIM {s['gim_mean']:6.2f} TECU")
    total = statistics(np.concatenate(all_model), np.concatenate(all_ref), lat_ref)
    report["all_files"] = total
    report["receipt"] = receipt(iono, "fixed" if record is None else f"{record.source} sha256:{record.sha256}",
                                "fixed" if record is None else args.f107_mode)
    for name, s in total.items():
        if not s["nodes"]:
            continue
        print(f"  {name:14s} bias {s['bias']:+6.2f}  rmse {s['rmse']:6.2f}  corr {s['corr']:5.3f}  "
              f"model {s['model_mean']:6.2f}  GIM {s['gim_mean']:6.2f}")
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
