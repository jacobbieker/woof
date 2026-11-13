"""Regenerate the pinned GEM-15 cell index and the fallback-region outlines.

GEM-15 (Google Research; https://doi.org/10.31223/X55N75) is published as
one Cloud-Optimized BigTIFF per 1x1-degree cell in the public bucket
``gs://climate-and-sustainability/gem-15``, indexed by a static STAC
catalog.  woof never constructs a cell URL from a formula: it reads the
pinned snapshot this tool writes (``woof/data/gem15_index.v1.json.gz``).

What is pinned, per cell: the 10-degree group and cell id the bucket uses,
the cell's south-west corner and UTM EPSG code read from its STAC Item, and
for each of the ``dtm`` and ``dsm`` objects the GCS object metadata Google
publishes (size, MD5, CRC32C and generation; MD5 is null for the few
objects GCS stores as composites, which publish CRC32C only) plus that asset's pixel grid
from the Item (shape and transform, checked against every COG header woof
reads).  GCS
publishes no SHA-256; woof records one on first download instead
(:mod:`woof.static.gem15`).

Usage (network; anonymous HTTPS only, no Earth Engine, no account)::

    pixi run -e mlx python tools/gem15_index.py --work ~/.cache/gem15-index \
        --out woof/data/gem15_index.v1.json.gz

    pixi run -e mlx python tools/gem15_index.py --exclusions \
        ne_110m_admin_0_countries.geojson \
        --exclusions-out woof/data/gem15_fallback_regions.v1.json

``--work`` caches the listing pages and STAC Items so an interrupted run
resumes.  The second form rebuilds the fallback-region outlines from the
Natural Earth 1:110m admin-0 countries GeoJSON (public domain,
https://www.naturalearthdata.com/), which the caller downloads.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

BUCKET = "climate-and-sustainability"
PREFIX = "gem-15/"
PUBLIC = f"https://storage.googleapis.com/{BUCKET}/gem-15"
LISTING = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
ASSETS = ("dtm", "dsm")
INDEX_SCHEMA = "woof.gem15-index.v1"
REGIONS_SCHEMA = "woof.gem15-fallback-regions.v1"

#: The regions the producer excludes or offsets (GEM-15 catalog page and
#: preprint), with the margin, in degrees, a footprint must keep from the
#: 1:110m outline before GEM-15 is used.  0.25 degrees absorbs the
#: generalisation of the 1:110m coastline/border; India's 1.0 degree is the
#: producer's offset band, which tapers "over tens of km" outside the
#: border.  Antarctica is listed although it lies south of the 61 S
#: envelope, so the outline file names every exclusion the producer states.
REGIONS = {
    "RUS": ("Russia", 0.25, "excluded from GEM-15 by the producer"),
    "CHN": ("China", 0.25, "excluded from GEM-15 by the producer"),
    "MNG": ("Mongolia", 0.25, "excluded from GEM-15 by the producer"),
    "GRL": ("Greenland", 0.25, "excluded from GEM-15 by the producer"),
    "ATA": ("Antarctica", 0.25, "excluded from GEM-15 by the producer"),
    "IND": ("India", 1.0, "GEM-15 is intentionally offset inside India and "
                          "in a band tapering over tens of km outside its "
                          "border"),
}


def _get_json(url: str, cache: Path | None):
    if cache is not None and cache.is_file() and cache.stat().st_size:
        return json.loads(cache.read_text(encoding="utf-8"))
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(payload)
    return json.loads(payload)


def list_objects(asset: str, work: Path) -> dict[str, dict]:
    """``{"<group>/<cell>": object metadata}`` for every ``asset`` mosaic."""
    out, token, page = {}, "", 0
    while True:
        query = {"prefix": PREFIX,
                 "matchGlob": f"{PREFIX}*/*/{asset}/zall/mosaic.tif",
                 "fields": "items(name,size,md5Hash,crc32c,generation),"
                           "nextPageToken",
                 "maxResults": "1000"}
        if token:
            query["pageToken"] = token
        listing = _get_json(LISTING + "?" + urllib.parse.urlencode(query),
                            work / "list" / f"{asset}.{page}.json")
        for item in listing.get("items", []):
            _, group, cell = item["name"].split("/")[:3]
            out[f"{group}/{cell}"] = item
        token = listing.get("nextPageToken", "")
        page += 1
        if not token:
            return out


def build_index(work: Path) -> dict:
    objects = {asset: list_objects(asset, work) for asset in ASSETS}
    if set(objects["dtm"]) != set(objects["dsm"]):
        raise SystemExit("dtm and dsm listings name different cells")
    keys = sorted(objects["dtm"])

    def item(key):
        group, cell = key.split("/")
        return key, _get_json(f"{PUBLIC}/stac/gem-15/{group}/{cell}/{cell}.json",
                              work / "items" / f"{cell}.json")

    with ThreadPoolExecutor(32) as pool:
        items = dict(pool.map(item, keys))
    cells = {}
    for key in keys:
        group, cell = key.split("/")
        stac = items[key]
        west, south, east, north = stac["bbox"]
        if (round(east - west, 9), round(north - south, 9)) != (1.0, 1.0):
            raise SystemExit(f"{cell}: STAC bbox {stac['bbox']} is not 1x1")
        # The two assets share a UTM zone but not always an extent (the
        # DSM of a cell can stop rows short of its DTM), so each asset's
        # own pixel grid is pinned.
        epsgs = {int(stac["assets"][asset]["proj:epsg"]) for asset in ASSETS}
        if len(epsgs) != 1:
            raise SystemExit(f"{cell}: dtm and dsm name different CRSs")
        row = {"group": group, "sw": [int(round(south)), int(round(west))],
               "epsg": epsgs.pop()}
        for asset in ASSETS:
            meta, grid = objects[asset][key], stac["assets"][asset]
            row[asset] = [int(meta["size"]), meta.get("md5Hash"), meta["crc32c"],
                          meta["generation"],
                          [int(v) for v in grid["proj:shape"]],
                          [float(v) for v in grid["proj:transform"][:6]]]
        cells[cell] = row
    return {"schema": INDEX_SCHEMA,
            "snapshot_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            "bucket_url": PUBLIC,
            "stac_url": f"{PUBLIC}/stac/catalog.json",
            "object_fields": ["bytes", "md5_base64", "crc32c_base64",
                              "generation", "shape", "transform"],
            "cells": cells}


def build_regions(geojson: Path) -> dict:
    data = json.loads(geojson.read_text(encoding="utf-8"))
    regions = {}
    for feature in data["features"]:
        code = feature["properties"]["ADM0_A3"]
        if code not in REGIONS:
            continue
        name, margin, why = REGIONS[code]
        geometry = feature["geometry"]
        polygons = (geometry["coordinates"]
                    if geometry["type"] == "MultiPolygon"
                    else [geometry["coordinates"]])
        # Outer rings only: a hole in a 1:110m country outline is a lake
        # or enclave, and a footprint over it is still at the border.
        rings = [[[round(x, 3), round(y, 3)] for x, y in polygon[0]]
                 for polygon in polygons]
        regions[code] = {"name": name, "margin_deg": margin, "why": why,
                         "rings": rings}
    missing = sorted(set(REGIONS) - set(regions))
    if missing:
        raise SystemExit(f"{geojson} has no feature for {missing}")
    return {"schema": REGIONS_SCHEMA,
            "source": "Natural Earth 1:110m Cultural Vectors, Admin 0 - "
                      "Countries (public domain), outer rings rounded to "
                      "0.001 degree",
            "regions": regions}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", type=Path, default=None,
                        help="cache directory for listing pages and Items")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the gzip JSON cell index here")
    parser.add_argument("--exclusions", type=Path, default=None,
                        help="Natural Earth 1:110m admin-0 countries GeoJSON")
    parser.add_argument("--exclusions-out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.out is not None:
        if args.work is None:
            parser.error("--out needs --work")
        index = build_index(args.work)
        text = json.dumps(index, sort_keys=True, separators=(",", ":"))
        # mtime=0: the same index always compresses to the same bytes.
        args.out.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
        print(f"{args.out}: {len(index['cells'])} cells", file=sys.stderr)
    if args.exclusions is not None:
        if args.exclusions_out is None:
            parser.error("--exclusions needs --exclusions-out")
        regions = build_regions(args.exclusions)
        args.exclusions_out.write_text(
            json.dumps(regions, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8")
        print(f"{args.exclusions_out}: {sorted(regions['regions'])}",
              file=sys.stderr)
    if args.out is None and args.exclusions is None:
        parser.error("nothing to do: pass --out and/or --exclusions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
