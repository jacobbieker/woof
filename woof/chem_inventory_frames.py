"""Annual anthropogenic inventory fields on a domain's mass points, and their
RADM2/SORGAM speciation (WRF-Chem emiss_opt = 3).

Two kinds of source row are served (:func:`serves_source`):

* an inventory of global regular latitude/longitude flux densities, pinned
  archive by archive (``edgar-v81``): each variable's selector names its
  HTTPS ``url``, zip ``archive``, ``archive_sha256``, ``archive_bytes``,
  NetCDF ``member`` and ``variable``.  The archive is fetched once into the
  emission cache (``WOOF_CHEM_SOURCE_CACHE`` or ``~/.woof/chem-sources``,
  one sub-directory per source, :func:`woof.chem_emission_frames.source_cache_dir`)
  and every read checks its size and sha256 against the pin; a cached file
  whose bytes differ is refused, never re-fetched over.  The NetCDF is read
  from the zip in memory (netCDF4), its coordinates, units and shape are
  checked against the row, and a non-finite value is refused by name;
* an analytic stand-in (``format: synthetic``, ``synthetic-anthro``) that
  needs no download: Gaussian urban plumes placed relative to the domain's
  own extent plus a weak background, so a run without network still emits.

Remap.  The inventory is interpolated BILINEARLY in latitude and longitude
(periodic in longitude for a global grid) on the flux density, which is
what the edgar-v81 row declares.  Bilinear sampling of a density is exact
for fields linear across a source cell and unbiased for a smooth field, but
not mass-conservative cell by cell: a 9 km cell sees the density at its
centre, not the mean over its footprint.  Making it conservative needs each
point's footprint, which a mesh-agnostic list of points does not carry; the
row names cell_sum on flux x cell area as the conservative v2 route.

Time.  Annual 2022 means, constant in time: no monthly, weekly or diurnal
profile (EDGAR v8.1's monthly sector files under .../monthly/ are the
future route).  ``valid_time`` is accepted for the SourceFrames call shape
(:mod:`woof.chem_emission_frames`) and recorded, never used to select.

Geometry.  Points are the domain's mass-point latitude/longitude in degrees,
of ANY shape: WRF's (ny, nx), a hex mesh's (1, ncells) or (ncells,).  Every
array returned has that shape.  No i/j structure is assumed.

Nothing here imports CuPy; the frames are host numpy arrays the emission
process copies to the device.
"""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

__all__ = ["InventoryFrames", "InventoryFrameError", "INVENTORY_PROCESSES",
           "SPECIATION_ROOT", "load_speciation", "serves_source",
           "serves_emission", "radm2_emission_frames",
           "radm2_emission_receipt"]

#: Process keys whose rows read their emissions through this provider (the
#: chem door's test, :func:`serves_emission`).
INVENTORY_PROCESSES = frozenset({"emission.radm2"})

SPECIATION_ROOT = Path(__file__).resolve().parent / "data" / "chem" / "speciation"

#: The speciation each process applies.
_PROCESS_SPECIATION = {"emission.radm2": "radm2"}

#: Selector members an inventory archive must pin for this provider.
_ARCHIVE_KEYS = ("url", "archive", "archive_sha256", "archive_bytes", "member",
                 "variable")


class InventoryFrameError(ValueError):
    """An inventory field cannot be supplied; the message names why."""


# --------------------------------------------------------------- speciation

_SPECIATION_CACHE: dict[str, tuple[dict, str]] = {}


def load_speciation(name: str = "radm2") -> tuple[dict, str]:
    """``(document, sha256)`` of ``speciation/<name>.v1.json``, validated."""
    cached = _SPECIATION_CACHE.get(name)
    if cached is not None:
        return cached
    from woof.chem_table import _validate

    path = SPECIATION_ROOT / f"{name}.v1.json"
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    _validate(document, "speciation_file", f"speciation/{path.name}")
    if document["name"] != name:
        raise InventoryFrameError(
            f"speciation/{path.name} defines {document['name']!r}, not {name!r}")
    stray = sorted({term["field"] for row in document["outputs"].values()
                    for term in row["terms"]} - set(document["inputs"]))
    if stray:
        raise InventoryFrameError(
            f"speciation/{path.name}: terms read {stray}, which are not its inputs")
    cached = _SPECIATION_CACHE[name] = (document, hashlib.sha256(raw).hexdigest())
    return cached


# --------------------------------------------------------------------- door

def serves_source(source) -> bool:
    """Can this provider produce fields of the source row ``source``?"""
    if source.kind != "emission":
        return False
    if source.format == "synthetic":
        return source.remap == "analytic" and (source.grid or {}).get(
            "kind") == "analytic_plumes"
    if source.format != "netcdf" or source.remap != "bilinear":
        return False
    grid = source.grid or {}
    if grid.get("kind") != "regular_latlon":
        return False
    return all(all(key in (spec.get("selector") or {}) for key in _ARCHIVE_KEYS)
               and spec.get("units") == "kg m-2 s-1"
               for spec in source.variables.values())


def serves_emission(row, ref, source) -> bool:
    """Is the species row ``row``'s emission reference ``ref`` to the source
    row ``source`` one this provider serves?

    It is when the row runs an inventory process (:data:`INVENTORY_PROCESSES`,
    i.e. ``emission.radm2``), the reference's ``field`` is an output of that
    process's speciation (an ``e_*`` name such as ``e_co``), and the source
    is servable and carries every input field the speciation reads.
    """
    if not serves_source(source):
        return False
    for process in INVENTORY_PROCESSES & set(row.processes):
        document, _digest = load_speciation(_PROCESS_SPECIATION[process])
        if ref["field"] in document["outputs"] and set(document["inputs"]) <= set(
                source.variables):
            return True
    return False


# ------------------------------------------------------------------ sources

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path, size: int) -> None:
    from woof.fetch_routes import _download_object

    _download_object(url, dest, magic="", declared_size=size)


def fetch_archive(source, field: str, cache: Path, *, download=None) -> Path:
    """The pinned archive of ``source``'s ``field`` in ``cache``, fetched once.

    A cached archive is used only when its size and sha256 equal the pin;
    otherwise it is refused by name (a changed file would be different
    emissions under the same identity).  A missing one is downloaded over
    HTTPS into a scratch file, checked against the pin, then moved in.
    """
    selector = source.variables[field]["selector"]
    name, url = selector["archive"], selector["url"]
    want_sha, want_size = selector["archive_sha256"], int(selector["archive_bytes"])
    if Path(name).name != name or not url.startswith("https://"):
        raise InventoryFrameError(
            f"{source.name}/{field}: archive {name!r} at {url!r} is not a plain "
            "file name served over HTTPS")
    cache = Path(cache)
    path = cache / name
    if not path.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as scratch:
            staged = Path(scratch) / name
            try:
                (download or _download)(url, staged, want_size)
            except Exception as error:
                raise InventoryFrameError(
                    f"{source.name}/{field}: cannot fetch {name} from {url} "
                    f"({error}); place the pinned file at {path} (sha256 "
                    f"{want_sha}) or use a source that needs no download, such "
                    "as synthetic-anthro") from error
            _check_archive(source, field, staged, want_size, want_sha)
            staged.replace(path)
    _check_archive(source, field, path, want_size, want_sha)
    return path


def _check_archive(source, field, path: Path, size: int, sha: str) -> None:
    got_size = path.stat().st_size
    if got_size != size:
        raise InventoryFrameError(
            f"{source.name}/{field}: {path} is {got_size} bytes, the pin says "
            f"{size}; refusing a different file under the pinned identity")
    got = _sha256(path)
    if got != sha:
        raise InventoryFrameError(
            f"{source.name}/{field}: {path} has sha256 {got}, the pin says "
            f"{sha}; refusing a different file under the pinned identity")


def decode_archive(source, field: str, path: Path) -> np.ndarray:
    """``(nlat, nlon)`` float64 flux density of ``field``, south to north."""
    import netCDF4

    selector = source.variables[field]["selector"]
    grid = source.grid
    with zipfile.ZipFile(path) as archive:
        try:
            raw = archive.read(selector["member"])
        except KeyError as error:
            raise InventoryFrameError(
                f"{source.name}/{field}: {path.name} has no member "
                f"{selector['member']!r}") from error
    with netCDF4.Dataset("inventory", mode="r", memory=raw) as dataset:
        if selector["variable"] not in dataset.variables:
            raise InventoryFrameError(
                f"{source.name}/{field}: {selector['member']} has no variable "
                f"{selector['variable']!r}")
        variable = dataset.variables[selector["variable"]]
        units = getattr(variable, "units", None)
        if units != source.variables[field]["units"]:
            raise InventoryFrameError(
                f"{source.name}/{field}: file units {units!r} differ from the "
                f"row's {source.variables[field]['units']!r}")
        data = variable[:]
        masked = int(np.ma.count_masked(data))
        if masked:
            raise InventoryFrameError(
                f"{source.name}/{field}: {masked} values in {selector['member']} "
                "are the file's fill or missing value; a missing flux is not "
                "a flux")
        values = np.asarray(np.ma.getdata(data), dtype=np.float64)
        lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
    shape = (int(grid["nlat"]), int(grid["nlon"]))
    if values.shape != shape:
        raise InventoryFrameError(
            f"{source.name}/{field}: {selector['variable']} is {values.shape}, "
            f"the row's grid is {shape}")
    step = float(grid["spacing_deg"])
    want_lat = float(grid["lat_first"]) + step * np.arange(shape[0])
    want_lon = float(grid["lon_first"]) + step * np.arange(shape[1])
    if (lat.shape != want_lat.shape or lon.shape != want_lon.shape
            or np.max(np.abs(lat - want_lat)) > 1e-6
            or np.max(np.abs(lon - want_lon)) > 1e-6):
        raise InventoryFrameError(
            f"{source.name}/{field}: the file's lat/lon are not the row's "
            f"{step} degree grid from ({grid['lat_first']}, {grid['lon_first']})")
    bad = int(np.count_nonzero(~np.isfinite(values)))
    if bad:
        raise InventoryFrameError(
            f"{source.name}/{field}: {bad} non-finite values in "
            f"{selector['member']}; a missing flux is not a zero flux")
    if np.any(values < 0.0):
        raise InventoryFrameError(
            f"{source.name}/{field}: negative flux densities in {selector['member']}")
    return values


# ------------------------------------------------------------------ remaps

def bilinear(values: np.ndarray, grid, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a regular south-to-north grid at points.

    A grid spanning 360 degrees of longitude wraps; a ``global`` grid clamps
    latitude to its first/last row beyond the outermost centres (the polar
    half-cells).  A regional grid refuses any point outside its centres:
    extrapolating would invent emissions where the inventory says nothing.
    """
    nlat, nlon = values.shape
    step = float(grid["spacing_deg"])
    lat0, lon0 = float(grid["lat_first"]), float(grid["lon_first"])
    periodic = abs(nlon * step - 360.0) < 1e-6
    fi = (lat - lat0) / step
    if periodic:
        fj = np.mod(lon - lon0, 360.0) / step
    else:
        centre = lon0 + 0.5 * (nlon - 1) * step
        fj = (centre + np.mod(lon - centre + 180.0, 360.0) - 180.0 - lon0) / step
    tolerance = 1e-9
    outside = (fi < -tolerance) | (fi > nlat - 1 + tolerance)
    if not periodic:
        outside |= (fj < -tolerance) | (fj > nlon - 1 + tolerance)
    if np.any(outside) and not (grid.get("global") and periodic):
        raise InventoryFrameError(
            f"{int(np.count_nonzero(outside))} points lie outside the source "
            f"grid's cell centres ({lat0}..{lat0 + (nlat - 1) * step} N, "
            f"{lon0}..{lon0 + (nlon - 1) * step} E); the inventory has no "
            "value there")
    fi = np.clip(fi, 0.0, nlat - 1.0)
    i0 = np.minimum(np.floor(fi).astype(np.int64), nlat - 2)
    wi = fi - i0
    if periodic:
        j0 = np.floor(fj).astype(np.int64) % nlon
        wj = fj - np.floor(fj)
        j1 = (j0 + 1) % nlon
    else:
        fj = np.clip(fj, 0.0, nlon - 1.0)
        j0 = np.minimum(np.floor(fj).astype(np.int64), nlon - 2)
        wj = fj - j0
        j1 = j0 + 1
    i1 = i0 + 1
    return ((1.0 - wi) * ((1.0 - wj) * values[i0, j0] + wj * values[i0, j1])
            + wi * ((1.0 - wj) * values[i1, j0] + wj * values[i1, j1]))


def _unwrapped(lon: np.ndarray) -> np.ndarray:
    """Longitudes continuous about their circular mean (no dateline cut)."""
    rad = np.radians(lon)
    centre = math.degrees(math.atan2(float(np.mean(np.sin(rad))),
                                     float(np.mean(np.cos(rad)))))
    return centre + np.mod(lon - centre + 180.0, 360.0) - 180.0


def synthetic_shape(grid, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """The dimensionless pattern every synthetic field scales (see the row)."""
    radius = float(grid["earth_radius_m"])
    ulon = _unwrapped(lon)
    lat_lo, lat_hi = float(lat.min()), float(lat.max())
    lon_lo, lon_hi = float(ulon.min()), float(ulon.max())
    phi = np.radians(lat)
    shape = np.full(lat.shape, float(grid["background_fraction"]))
    for plume in grid["plumes"]:
        fy, fx = plume["at"]
        clat = math.radians(lat_lo + fy * (lat_hi - lat_lo))
        clon = math.radians(lon_lo + fx * (lon_hi - lon_lo))
        dphi = phi - clat
        dlam = np.radians(ulon) - clon
        h = (np.sin(0.5 * dphi) ** 2
             + np.cos(phi) * math.cos(clat) * np.sin(0.5 * dlam) ** 2)
        distance = 2.0 * radius * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
        sigma = float(plume["sigma_km"]) * 1000.0
        shape += float(plume["scale"]) * np.exp(-0.5 * (distance / sigma) ** 2)
    return shape


# ---------------------------------------------------------------- provider

class InventoryFrames:
    """Inventory fields on one domain's mass points, in kg m-2 s-1.

    ``latitude``/``longitude``: degrees, any equal shapes.  ``cell_area_m2``
    (optional, same shape) is each point's cell area; with it, receipts carry
    domain totals (WRF: dx*dy/(msftx*msfty)).  ``catalog``: the chem catalog
    (default :func:`woof.chem_table.catalog`).  ``cache_dir``: overrides the
    per-source cache directory.  ``download(url, dest, size)``: the transport
    (default the engine's HTTPS download).
    """

    def __init__(self, latitude, longitude, *, cell_area_m2=None, catalog=None,
                 cache_dir=None, download=None):
        lat = np.asarray(latitude, dtype=np.float64)
        lon = np.asarray(longitude, dtype=np.float64)
        if lat.shape != lon.shape or lat.size == 0:
            raise InventoryFrameError(
                f"latitude {lat.shape} and longitude {lon.shape} must be equal, "
                "non-empty shapes")
        if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))
                and np.all(np.abs(lat) <= 90.0)):
            raise InventoryFrameError("latitude/longitude must be finite degrees")
        self.shape = lat.shape
        self._lat = np.ascontiguousarray(lat.reshape(-1))
        self._lon = np.ascontiguousarray(lon.reshape(-1))
        self.cell_area_m2 = None
        if cell_area_m2 is not None:
            area = np.asarray(cell_area_m2, dtype=np.float64)
            if area.shape != self.shape or not np.all(area > 0.0):
                raise InventoryFrameError(
                    f"cell_area_m2 must be positive with shape {self.shape}")
            self.cell_area_m2 = area
        self._catalog = catalog
        self._cache_dir = None if cache_dir is None else Path(cache_dir)
        self._download = download
        self._fields: dict[tuple[str, str], np.ndarray] = {}
        #: (source, field) -> what produced it (archive pins or the pattern).
        self.receipts: dict[tuple[str, str], dict] = {}

    def source(self, name: str):
        catalog = self._catalog
        if catalog is None:
            from woof.chem_table import catalog as load

            catalog = self._catalog = load()
        row = catalog.sources.get(name)
        if row is None:
            raise InventoryFrameError(
                f"no chem source row {name!r} (woof/data/chem/sources/)")
        if not serves_source(row):
            raise InventoryFrameError(
                f"source {name!r} is not an inventory this provider serves "
                "(a pinned regular lat/lon NetCDF on a bilinear remap, or a "
                "synthetic analytic row)")
        return row

    def _cache(self, name: str) -> Path:
        if self._cache_dir is not None:
            return self._cache_dir / name
        from woof.chem_emission_frames import source_cache_dir

        return source_cache_dir(name)

    def field(self, source: str, field: str, valid_time: datetime | None = None
              ) -> np.ndarray:
        """``field`` of ``source`` on the points (shape of ``latitude``), float64
        kg m-2 s-1.  Annual mean: ``valid_time`` does not select."""
        del valid_time
        key = (source, field)
        cached = self._fields.get(key)
        if cached is None:
            row = self.source(source)
            if field not in row.variables:
                raise InventoryFrameError(
                    f"source {source!r} has no field {field!r} (it carries "
                    f"{sorted(row.variables)})")
            if row.format == "synthetic":
                peak = float(row.variables[field]["peak"])
                values = peak * synthetic_shape(row.grid, self._lat, self._lon)
                self.receipts[key] = {"synthetic": True, "peak": peak,
                                      "pattern": "analytic_plumes"}
            else:
                path = fetch_archive(row, field, self._cache(source),
                                     download=self._download)
                grid_values = decode_archive(row, field, path)
                values = bilinear(grid_values, row.grid, self._lat, self._lon)
                selector = row.variables[field]["selector"]
                self.receipts[key] = {
                    "archive": selector["archive"],
                    "archive_sha256": selector["archive_sha256"],
                    "archive_bytes": int(selector["archive_bytes"]),
                    "member": selector["member"], "remap": row.remap}
            cached = values.reshape(self.shape)
            cached.setflags(write=False)
            self._fields[key] = cached
        return cached

    def at(self, source: str, field: str, valid_time: datetime | None = None):
        """The SourceFrames call shape (:mod:`woof.chem_emission_frames`)."""
        return self.field(source, field, valid_time)


# ------------------------------------------------------------------- RADM2

def radm2_emission_frames(provider: InventoryFrames, source: str,
                          valid_time: datetime | None = None
                          ) -> dict[str, np.ndarray]:
    """``{e_name: float32 array}`` for every name in
    :data:`woof.core.chem_radm2_emiss.EMISSIONS`, in WRF emis_ant units
    (gases mol km-2 hr-1, aerosols ug m-2 s-1), each C-contiguous with the
    provider's point shape (a 2-D (ny, nx) frame is level 1 for
    launch_radm2_emiss).  Rows the speciation leaves without terms are
    zeros; every other row is the float64 sum of its terms, clamped at zero
    where the speciation says so, rounded once to float32.
    """
    from woof.core.chem_radm2_emiss import EMISSIONS

    document, _digest = load_speciation("radm2")
    outputs = document["outputs"]
    missing = [name for name in EMISSIONS if name not in outputs]
    if missing:
        raise InventoryFrameError(
            f"speciation radm2 has no rows for {missing}, which "
            "launch_radm2_emiss reads")
    inputs = {}

    def read(field):
        values = inputs.get(field)
        if values is None:
            values = inputs[field] = provider.field(source, field, valid_time)
        return values

    totals = {}
    for name in EMISSIONS:
        spec = outputs[name]
        total = np.zeros(provider.shape, dtype=np.float64)
        for term in spec["terms"]:
            total += float(term["weight"]) * read(term["field"])
        if spec["clamp_at_zero"]:
            np.maximum(total, 0.0, out=total)
        totals[name] = total
    for cap in document.get("caps", ()):
        members = cap["members"]
        limit = float(cap["limit"]["weight"]) * read(cap["limit"]["field"])
        summed = sum(totals[name] for name in members)
        over = summed > limit
        if np.any(over):
            scale = np.where(over, limit / np.where(over, summed, 1.0), 1.0)
            for name in members:
                totals[name] = totals[name] * scale
    return {name: np.ascontiguousarray(totals[name], dtype=np.float32)
            for name in EMISSIONS}


def _source_sha256(catalog, row) -> str:
    """The row file's sha256 as its catalog recorded it at load; a catalog
    that records none (a hand-built test catalog) hashes the packaged file."""
    hashes = getattr(catalog, "file_hashes", None)
    if hashes is not None and row.source_file in hashes:
        return hashes[row.source_file]
    from woof.chem_table import CHEM_DATA_ROOT
    return hashlib.sha256((CHEM_DATA_ROOT / row.source_file).read_bytes()).hexdigest()


def radm2_emission_receipt(provider: InventoryFrames, source: str,
                           frames: dict[str, np.ndarray],
                           valid_time: datetime | None = None) -> dict:
    """What the forecast receipt records about ``frames`` from ``source``.

    Keys: ``source``, ``source_file``, ``source_sha256`` (the row file's
    bytes), ``synthetic``, ``archives`` ({field: {archive, archive_sha256,
    archive_bytes, member, remap}}, empty for synthetic), ``speciation``,
    ``speciation_sha256``, ``time_profile``, ``valid_time``, ``units``
    ({e_name: units}), and ``totals``: per e_name the domain sum, gases in
    mol hr-1, aerosols in ug s-1 (None without the provider's
    ``cell_area_m2``).
    """
    row = provider.source(source)
    document, digest = load_speciation("radm2")
    fields = sorted({term["field"] for spec in document["outputs"].values()
                     for term in spec["terms"]})
    archives = {}
    for field in fields:
        receipt = provider.receipts.get((source, field))
        if receipt is None:
            raise InventoryFrameError(
                f"{source}/{field} was never read by this provider; build the "
                "frames before their receipt")
        if not receipt.get("synthetic"):
            archives[field] = dict(receipt)
    area = provider.cell_area_m2
    totals = {}
    for name, frame in frames.items():
        units = document["outputs"][name]["units"]
        if area is None:
            totals[name] = None
        elif units == "mol km-2 hr-1":
            totals[name] = float(np.sum(frame.astype(np.float64) * area * 1.0e-6))
        else:
            totals[name] = float(np.sum(frame.astype(np.float64) * area))
    catalog = provider._catalog
    if catalog is None:
        from woof.chem_table import catalog as load
        catalog = load()
    return {
        "source": source,
        "source_file": f"woof/data/chem/{row.source_file}",
        # The bytes the provider's catalog loaded the row from.
        "source_sha256": _source_sha256(catalog, row),
        "synthetic": row.format == "synthetic",
        "archives": archives,
        "speciation": "woof/data/chem/speciation/radm2.v1.json",
        "speciation_sha256": digest,
        "time_profile": "annual_mean_constant",
        "valid_time": None if valid_time is None else valid_time.isoformat(),
        "units": {name: document["outputs"][name]["units"] for name in frames},
        "totals": totals,
    }
