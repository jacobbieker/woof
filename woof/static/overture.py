"""Fetch Overture Maps building footprints for a box (``woof buildings --overture``).

Reads the public Overture release (``theme=buildings/type=building``) from
the anonymous S3 bucket with pyarrow's own S3 filesystem, keeps only the
rows whose per-row ``bbox`` struct meets the requested box (pyarrow dataset
filter push-down, so only the matching row groups are read) and only the
columns woof needs, and writes a local GeoParquet file (WKB geometry,
``geo`` metadata, OGC:CRS84) that :func:`woof.static.buildings.read_footprints`
reads unchanged with ``source="overture"``.  Beside it goes a JSON sidecar
with the release, box, location, fetch time, footprint count, per-dataset
counts and licences.  woof ships and redistributes no data.

Licence and attribution (https://docs.overturemaps.org/attribution/, checked
2026-11-19): the buildings theme is licensed ODbL 1.0 with the attribution
"(c) OpenStreetMap contributors"; individual records can come from other
datasets under their own licences (Esri Community Maps contributors and
Google Open Buildings CC BY 4.0, Microsoft Global ML Building Footprints
ODbL, USGS 3DEP, others).  Each record's ``sources[]`` entry carries its
dataset and licence, and both are counted in the sidecar.  woof's
attribution for the source is the ``BUILDING_SOURCES["overture"]`` row,
"(c) OpenStreetMap contributors, Overture Maps Foundation".
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import sys
from pathlib import Path

#: Anonymous public bucket and release prefix (bucket/key form, no scheme).
OVERTURE_RELEASE_BASE = "overturemaps-us-west-2/release"
OVERTURE_REGION = "us-west-2"
THEME_PATH = "theme=buildings/type=building"
#: Columns read from the release (``bbox`` is only used for the filter).
#: ``sources`` is read for the provenance counts only; the cached file
#: drops it (the sidecar holds the provenance).
COLUMNS = ("id", "geometry", "height", "num_floors", "sources")
FILE_COLUMNS = ("id", "geometry", "height", "num_floors")
THEME_LICENSE = "ODbL-1.0"
ATTRIBUTION = "© OpenStreetMap contributors, Overture Maps Foundation"
SIDECAR_SCHEMA = "woof-overture-fetch.v1"

_RELEASE = re.compile(r"^\d{4}-\d{2}-\d{2}\.\d+$")
_SAFE_RELEASE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]*$")


class OvertureRefusal(ValueError):
    """The footprints cannot be fetched or found in the cache."""


_NO_PYARROW = ("fetching Overture footprints needs pyarrow built with S3 "
               "support; install pyarrow (the woof extra that provides it, "
               "or `pip install pyarrow`), or download the footprints and "
               "pass them with --footprints PATH:overture")


def _s3():
    try:
        import pyarrow.fs as pafs
        fs = pafs.S3FileSystem
    except (ImportError, AttributeError) as exc:
        raise OvertureRefusal(_NO_PYARROW) from exc
    # Retries and longer timeouts: the default 3 s connect timeout fails
    # on a slow link or a busy bucket.
    try:
        return fs(
            anonymous=True, region=OVERTURE_REGION, connect_timeout=15.0,
            request_timeout=120.0,
            retry_strategy=pafs.AwsStandardS3RetryStrategy(max_attempts=6))
    except (ImportError, NotImplementedError) as exc:
        raise OvertureRefusal(_NO_PYARROW) from exc


def latest_release(filesystem=None, base: str = OVERTURE_RELEASE_BASE) -> str:
    """The newest ``YYYY-MM-DD.N`` release under ``base``."""
    try:
        import pyarrow.fs as pafs
    except ImportError as exc:
        raise OvertureRefusal(_NO_PYARROW) from exc

    fs = filesystem if filesystem is not None else _s3()
    try:
        infos = fs.get_file_info(pafs.FileSelector(base))
    except OSError as exc:
        raise OvertureRefusal(f"cannot list Overture releases under "
                              f"{base}: {exc}") from exc
    names = [Path(i.path).name for i in infos
             if i.type == pafs.FileType.Directory]
    found = sorted((n for n in names if _RELEASE.match(n)),
                   key=lambda n: (n.split(".")[0], int(n.split(".")[1])))
    if not found:
        raise OvertureRefusal(f"no Overture release found under {base}")
    return found[-1]


def _box(bbox) -> tuple[float, float, float, float]:
    """(lon_min, lat_min, lon_max, lat_max) rounded to 6 decimals."""
    return tuple(round(float(v), 6) for v in (
        bbox.lon_min, bbox.lat_min, bbox.lon_max, bbox.lat_max))


def _box_key(box, base) -> str:
    """Hash of the box and the source location, so a fixture or mirror
    never aliases the public bucket's cache entry."""
    return hashlib.sha256(json.dumps([box, str(base)]).encode()
                          ).hexdigest()[:12]


def _cache_names(cache_dir: Path, release: str, key: str):
    stem = f"overture-buildings-{release}-{key}"
    return cache_dir / f"{stem}.parquet", cache_dir / f"{stem}.json"


def _cached(cache_dir: Path, release: str, key: str):
    """(parquet, sidecar) when both are cached, else None."""
    data, side = _cache_names(cache_dir, release, key)
    return (data, side) if data.is_file() and side.is_file() else None


def _newest_cached(cache_dir: Path, key: str):
    best = None
    for data in cache_dir.glob(f"overture-buildings-*-{key}.parquet"):
        release = data.name[len("overture-buildings-"):-len(f"-{key}.parquet")]
        if not _RELEASE.match(release) or not data.with_suffix(".json").is_file():
            continue
        rank = (release.split(".")[0], int(release.split(".")[1]))
        if best is None or rank > best[0]:
            best = (rank, release)
    return best[1] if best else None


def _geo_metadata() -> bytes:
    # No ``crs`` key: the GeoParquet specification defines that as
    # OGC:CRS84, and read_footprints reads it that way.
    meta = {"version": "1.1.0", "primary_column": "geometry",
            "columns": {"geometry": {
                "encoding": "WKB",
                "geometry_types": ["Polygon", "MultiPolygon"]}}}
    return json.dumps(meta).encode()


def _distinct_counts(rows, values) -> dict[str, int]:
    """Footprints (distinct ``rows``) per value; null values are named
    ``unspecified``."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if len(values) == 0:
        return {}
    pairs = pa.table({"row": rows, "v": pc.fill_null(values, "unspecified")})
    pairs = pairs.group_by(["row", "v"]).aggregate([])
    counted = pairs.group_by("v").aggregate([("row", "count")])
    return dict(sorted(zip(counted.column("v").to_pylist(),
                           counted.column("row_count").to_pylist())))


def _provenance(table) -> tuple[dict, dict]:
    """Footprints per ``sources[].dataset`` and per ``sources[].license``
    (a footprint counts once per distinct value)."""
    import pyarrow.compute as pc

    if "sources" not in table.column_names or table.num_rows == 0:
        return {}, {}
    col = table.column("sources").combine_chunks()
    flat = pc.list_flatten(col)
    rows = pc.list_parent_indices(col)
    out = []
    for field in ("dataset", "license"):
        if flat.type.get_field_index(field) < 0:
            out.append({})
            continue
        out.append(_distinct_counts(rows, pc.struct_field(flat, field)))
    return out[0], out[1]


def _replace_atomic(path: Path, write) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def fetch_overture_buildings(bbox, cache_dir, *, release: str = "latest",
                             offline: bool = False, filesystem=None,
                             base: str = OVERTURE_RELEASE_BASE
                             ) -> tuple[Path, dict]:
    """GeoParquet of the Overture buildings meeting ``bbox``, cached.

    ``bbox`` has ``lon_min/lat_min/lon_max/lat_max``.  ``release`` is an
    explicit release id or ``"latest"`` (discovered from the bucket; if the
    listing fails, or offline, the newest cached release for this box).  ``filesystem`` and
    ``base`` replace the anonymous S3 bucket (tests).  Returns
    ``(parquet path, sidecar dict)``; a second call for the same release,
    box and source reads the cache (an explicit release touches no network).
    """
    cache_dir = Path(cache_dir)
    box = _box(bbox)
    key = _box_key(box, base)

    def cached(rel):
        hit = _cached(cache_dir, rel, key)
        return (hit[0], json.loads(hit[1].read_text(encoding="utf-8"))
                ) if hit else None

    if release != "latest" and not _SAFE_RELEASE.match(str(release)):
        raise OvertureRefusal(f"bad Overture release id {release!r}")
    if offline:
        if release == "latest":
            release = _newest_cached(cache_dir, key)
        got = release and cached(release)
        if not got:
            raise OvertureRefusal(
                f"--offline: no cached Overture buildings for this box "
                f"under {cache_dir}; run once without --offline")
        return got
    if release == "latest":
        try:
            release = latest_release(filesystem, base)
        except OvertureRefusal as exc:
            newest = _newest_cached(cache_dir, key)
            if newest is None:
                raise
            print(f"woof buildings: cannot list Overture releases ({exc}); "
                  f"using the cached release {newest}", file=sys.stderr)
            release = newest
    got = cached(release)
    if got:
        return got

    try:
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise OvertureRefusal(_NO_PYARROW) from exc

    fs = filesystem if filesystem is not None else _s3()
    location = f"{base}/{release}/{THEME_PATH}/"
    xmin, ymin, xmax, ymax = box
    flt = ((ds.field("bbox", "xmin") <= xmax)
           & (ds.field("bbox", "xmax") >= xmin)
           & (ds.field("bbox", "ymin") <= ymax)
           & (ds.field("bbox", "ymax") >= ymin))
    try:
        dataset = ds.dataset(location, filesystem=fs, format="parquet")
        table = dataset.to_table(filter=flt, columns=list(COLUMNS))
    except OSError as exc:
        raise OvertureRefusal(
            f"cannot read Overture release {release!r} at {location}: "
            f"{exc}") from exc
    datasets, licenses = _provenance(table)
    sidecar = {
        "schema": SIDECAR_SCHEMA,
        "release": release,
        "bbox": dict(lon_min=xmin, lat_min=ymin, lon_max=xmax, lat_max=ymax),
        "location": ("s3://" + location if filesystem is None
                     else location),
        "fetched_utc": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_footprints": int(table.num_rows),
        "columns": list(COLUMNS),
        "sources_by_dataset": datasets,
        "sources_by_license": licenses,
        "theme_license": THEME_LICENSE,
        "attribution": ATTRIBUTION,
        "attribution_url": "https://docs.overturemaps.org/attribution/",
    }
    table = table.select(list(FILE_COLUMNS)).replace_schema_metadata(
        {b"geo": _geo_metadata()})
    cache_dir.mkdir(parents=True, exist_ok=True)
    data, side = _cache_names(cache_dir, release, key)
    # Sidecar first: a parquet file is only a cache hit once its sidecar
    # exists, so an interrupted fetch never leaves a half entry.
    _replace_atomic(side, lambda p: p.write_text(
        json.dumps(sidecar, indent=1, sort_keys=True), encoding="utf-8"))
    _replace_atomic(data, lambda p: pq.write_table(table, p))
    return data, sidecar


__all__ = ["ATTRIBUTION", "OvertureRefusal", "fetch_overture_buildings",
           "latest_release"]
