"""Versioned file contracts shared by every ``woof energy`` stage.

Four documents chain the stages together.  Each carries a ``schema`` string
and is refused on read when the string, a required field or a value range is
wrong; nothing is silently coerced.

``woof-energy.assets.v1`` (``assets.geojson``)
    A GeoJSON FeatureCollection in EPSG:4326.  One Feature per grid asset
    (line, cable, substation, plant, generator, tower).  Feature properties
    are the fields of :class:`Asset` other than ``geometry``.  The top-level
    ``woof`` member holds ``schema`` and a list of provenance records
    (``sources``), each naming where the assets came from, when, and under
    what licence.

``woof-energy.sites.v1`` (``sites.json``)
    The points where a forecast is wanted, stored columnar (one list per
    field, all the same length) because a national grid sampled every 100 m
    is a few hundred thousand points.  ``heights_m`` is set-wide: the
    above-ground heights every site is sampled at.

``woof-energy.plan.v1`` (``plan/plan.json``)
    The domains a topology planner emitted, with paths *relative to the
    plan file's directory*: the WRF configuration or MPAS mesh documents,
    the run directory and the output glob the extractor reads, the domain
    footprint, and the sites each domain is responsible for.

``woof-energy.forecast.v1`` (``forecast.nc``)
    A netCDF/Zarr dataset with dimensions ``(time, site, height)``; the
    variable table is :data:`FORECAST_VARIABLES` and the coordinate table is
    :data:`FORECAST_COORDINATES`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ASSETS_SCHEMA = "woof-energy.assets.v1"
SITES_SCHEMA = "woof-energy.sites.v1"
PLAN_SCHEMA = "woof-energy.plan.v1"
FORECAST_SCHEMA = "woof-energy.forecast.v1"

#: Asset kinds, named after the OSM ``power=*`` values they come from.
ASSET_KINDS = (
    "line", "minor_line", "cable", "substation", "plant", "generator",
    "tower",
)
LINEAR_KINDS = ("line", "minor_line", "cable")

#: ``generator:source`` / ``plant:source`` vocabulary kept on an asset.
GENERATOR_SOURCES = (
    "wind", "solar", "hydro", "tidal", "wave", "gas", "oil", "coal",
    "nuclear", "biomass", "biogas", "waste", "geothermal", "battery",
    "other",
)

#: What a forecast site stands for.
SITE_KINDS = ("line_sample", "tower", "substation", "turbine", "pv", "plant")

#: Domain topologies a planner may emit.
TOPOLOGIES = ("wrf-nests", "wrf-tiles", "hex-swath")

#: Role of a domain inside a plan.  ``parent`` domains exist only to force
#: children and own no sites; ``child`` domains own sites; ``mesh`` is the
#: single MPAS limited-area mesh of a ``hex-swath`` plan.
DOMAIN_ROLES = ("parent", "child", "mesh")

_GEOMETRY_BY_KIND = {
    "line": ("LineString", "MultiLineString"),
    "minor_line": ("LineString", "MultiLineString"),
    "cable": ("LineString", "MultiLineString"),
    "substation": ("Point", "Polygon", "MultiPolygon"),
    "plant": ("Point", "Polygon", "MultiPolygon"),
    "generator": ("Point", "Polygon", "MultiPolygon"),
    "tower": ("Point",),
}


class ContractError(ValueError):
    """A ``woof energy`` document does not satisfy its contract."""


class EnergyNotImplemented(NotImplementedError):
    """A ``woof energy`` stage that this build does not implement yet."""

    def __init__(self, what: str):
        super().__init__(
            f"{what} is not implemented in this build of woof energy")


# --------------------------------------------------------------------------
# helpers


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(path: str | Path, *, relative_to: str | Path | None = None
             ) -> dict[str, str]:
    """``{"path", "sha256"}`` binding of an input document."""

    path = Path(path)
    shown = path
    if relative_to is not None:
        try:
            shown = path.resolve().relative_to(Path(relative_to).resolve())
        except ValueError:
            shown = path
    return {"path": str(shown), "sha256": sha256_file(path)}


def _finite(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{what} must be a number, got {value!r}") from error
    if not math.isfinite(number):
        raise ContractError(f"{what} must be finite, got {value!r}")
    return number


def _optional_positive(value: Any, what: str) -> float | None:
    if value is None:
        return None
    number = _finite(value, what)
    if number <= 0.0:
        raise ContractError(f"{what} must be positive, got {value!r}")
    return number


def _check_lonlat(lon: Any, lat: Any, what: str) -> tuple[float, float]:
    lon = _finite(lon, f"{what} longitude")
    lat = _finite(lat, f"{what} latitude")
    if not -180.0 <= lon <= 180.0:
        raise ContractError(f"{what} longitude {lon} is outside [-180, 180]")
    if not -90.0 <= lat <= 90.0:
        raise ContractError(f"{what} latitude {lat} is outside [-90, 90]")
    return lon, lat


def _positions(geometry: Mapping[str, Any]) -> Iterable[Sequence[float]]:
    kind = geometry.get("type")
    coords = geometry.get("coordinates")
    if kind == "Point":
        yield coords
    elif kind in ("LineString", "MultiPoint"):
        yield from coords
    elif kind in ("MultiLineString", "Polygon"):
        for part in coords:
            yield from part
    elif kind == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                yield from ring
    else:
        raise ContractError(f"unsupported geometry type {kind!r}")


def validate_geometry(geometry: Mapping[str, Any], *, what: str = "geometry"
                      ) -> None:
    if not isinstance(geometry, Mapping) or "type" not in geometry:
        raise ContractError(f"{what} is not a GeoJSON geometry")
    if geometry.get("coordinates") is None:
        raise ContractError(f"{what} has no coordinates")
    count = 0
    for position in _positions(geometry):
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            raise ContractError(f"{what} has a malformed position {position!r}")
        _check_lonlat(position[0], position[1], what)
        count += 1
    if count == 0:
        raise ContractError(f"{what} is empty")
    kind = geometry["type"]
    if kind == "LineString" and count < 2:
        raise ContractError(f"{what} LineString needs at least two positions")
    if kind == "Polygon":
        for ring in geometry["coordinates"]:
            if len(ring) < 4 or list(ring[0][:2]) != list(ring[-1][:2]):
                raise ContractError(f"{what} Polygon ring must be closed "
                                    "with at least four positions")


def _schema_of(document: Mapping[str, Any], expected: str, what: str) -> None:
    found = document.get("schema")
    if found != expected:
        raise ContractError(f"{what} declares schema {found!r}; "
                            f"this woof reads {expected!r}")


def _write_json(document: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".partial")
    staging.write_text(json.dumps(document, indent=1, sort_keys=False) + "\n")
    staging.replace(path)
    return path


def _read_json(path: str | Path, what: str) -> dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text())
    except json.JSONDecodeError as error:
        raise ContractError(f"{what} {path} is not JSON: {error}") from error
    if not isinstance(document, dict):
        raise ContractError(f"{what} {path} is not a JSON object")
    return document


# --------------------------------------------------------------------------
# assets


@dataclass(frozen=True)
class Asset:
    """One piece of grid infrastructure.

    ``asset_id`` is stable and source-qualified (``osm:way/123``,
    ``pypsa-eur:line/8``, ``repd:4567``).  ``voltage_kv`` lists every
    voltage a line or substation carries, highest first.  ``tags`` holds
    any further source attributes as strings.
    """

    asset_id: str
    kind: str
    geometry: dict
    source: str
    source_ref: str | None = None
    name: str | None = None
    operator: str | None = None
    voltage_kv: tuple[float, ...] = ()
    circuits: int | None = None
    cables: int | None = None
    frequency_hz: float | None = None
    generator_source: str | None = None
    capacity_mw: float | None = None
    hub_height_m: float | None = None
    rotor_diameter_m: float | None = None
    license: str | None = None
    tags: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.asset_id, str) or not self.asset_id:
            raise ContractError("asset_id must be a non-empty string")
        if self.kind not in ASSET_KINDS:
            raise ContractError(f"{self.asset_id}: kind {self.kind!r} is not "
                                f"one of {ASSET_KINDS}")
        if not isinstance(self.source, str) or not self.source:
            raise ContractError(f"{self.asset_id}: source must be named")
        validate_geometry(self.geometry, what=self.asset_id)
        allowed = _GEOMETRY_BY_KIND[self.kind]
        if self.geometry["type"] not in allowed:
            raise ContractError(f"{self.asset_id}: a {self.kind} must be "
                                f"{'/'.join(allowed)}, got "
                                f"{self.geometry['type']}")
        voltages = tuple(sorted((_finite(v, f"{self.asset_id} voltage_kv")
                                 for v in self.voltage_kv), reverse=True))
        if any(v <= 0.0 for v in voltages):
            raise ContractError(f"{self.asset_id}: voltage_kv must be positive")
        object.__setattr__(self, "voltage_kv", voltages)
        if (self.generator_source is not None
                and self.generator_source not in GENERATOR_SOURCES):
            raise ContractError(f"{self.asset_id}: generator_source "
                                f"{self.generator_source!r} is not one of "
                                f"{GENERATOR_SOURCES}")
        for name in ("capacity_mw", "hub_height_m", "rotor_diameter_m",
                     "frequency_hz"):
            object.__setattr__(self, name, _optional_positive(
                getattr(self, name), f"{self.asset_id} {name}"))
        for name in ("circuits", "cables"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ContractError(f"{self.asset_id}: {name} must be a "
                                    f"positive integer, got {value!r}")
        if not all(isinstance(k, str) and isinstance(v, str)
                   for k, v in self.tags.items()):
            raise ContractError(f"{self.asset_id}: tags must map str to str")

    @property
    def max_voltage_kv(self) -> float | None:
        return self.voltage_kv[0] if self.voltage_kv else None

    def to_feature(self) -> dict[str, Any]:
        properties = {f.name: getattr(self, f.name) for f in fields(self)
                      if f.name != "geometry"}
        properties["voltage_kv"] = list(self.voltage_kv)
        return {"type": "Feature", "id": self.asset_id,
                "geometry": self.geometry, "properties": properties}

    @classmethod
    def from_feature(cls, feature: Mapping[str, Any]) -> "Asset":
        if feature.get("type") != "Feature":
            raise ContractError("asset is not a GeoJSON Feature")
        properties = dict(feature.get("properties") or {})
        known = {f.name for f in fields(cls)} - {"geometry"}
        unknown = set(properties) - known
        if unknown:
            raise ContractError(f"asset {properties.get('asset_id')!r} has "
                                f"unknown properties {sorted(unknown)}")
        properties["voltage_kv"] = tuple(properties.get("voltage_kv") or ())
        properties["tags"] = dict(properties.get("tags") or {})
        return cls(geometry=feature.get("geometry"), **properties)


@dataclass
class AssetCollection:
    """An ``assets.v1`` document: assets plus where they came from.

    Each ``sources`` record has at least ``source`` and ``license``; fetchers
    add ``retrieved_utc``, ``endpoint``/``url``, ``query`` and
    ``attribution``.
    """

    assets: list[Asset]
    sources: list[dict] = field(default_factory=list)

    def __post_init__(self):
        seen: set[str] = set()
        for asset in self.assets:
            if asset.asset_id in seen:
                raise ContractError(f"duplicate asset_id {asset.asset_id!r}")
            seen.add(asset.asset_id)
        for record in self.sources:
            if not isinstance(record, dict) or "source" not in record:
                raise ContractError("each sources record must name a source")

    def __len__(self) -> int:
        return len(self.assets)

    def by_kind(self, *kinds: str) -> list[Asset]:
        return [a for a in self.assets if a.kind in kinds]

    def to_geojson(self) -> dict[str, Any]:
        return {"type": "FeatureCollection",
                "woof": {"schema": ASSETS_SCHEMA, "sources": self.sources},
                "features": [a.to_feature() for a in self.assets]}

    @classmethod
    def from_geojson(cls, document: Mapping[str, Any]) -> "AssetCollection":
        if document.get("type") != "FeatureCollection":
            raise ContractError("assets document is not a FeatureCollection")
        meta = document.get("woof") or {}
        _schema_of(meta, ASSETS_SCHEMA, "assets document")
        assets = [Asset.from_feature(f) for f in document.get("features", [])]
        return cls(assets=assets, sources=list(meta.get("sources") or []))


def load_assets(path: str | Path) -> AssetCollection:
    return AssetCollection.from_geojson(_read_json(path, "assets document"))


def dump_assets(collection: AssetCollection, path: str | Path) -> Path:
    return _write_json(collection.to_geojson(), path)


# --------------------------------------------------------------------------
# sites

#: Columns of a ``sites.v1`` document, in order, with whether they may be
#: null.  ``bearing_deg`` is the line azimuth at the sample, clockwise from
#: true north in [0, 360); ``chainage_m`` is the distance along the asset.
SITE_COLUMNS: tuple[tuple[str, bool], ...] = (
    ("site_id", False),
    ("asset_id", False),
    ("kind", False),
    ("lat", False),
    ("lon", False),
    ("bearing_deg", True),
    ("chainage_m", True),
    ("voltage_kv", True),
    ("hub_height_m", True),
    ("capacity_mw", True),
)


@dataclass(frozen=True)
class Site:
    site_id: str
    asset_id: str
    kind: str
    lat: float
    lon: float
    bearing_deg: float | None = None
    chainage_m: float | None = None
    voltage_kv: float | None = None
    hub_height_m: float | None = None
    capacity_mw: float | None = None

    def __post_init__(self):
        if not self.site_id or not self.asset_id:
            raise ContractError("site_id and asset_id must be non-empty")
        if self.kind not in SITE_KINDS:
            raise ContractError(f"{self.site_id}: kind {self.kind!r} is not "
                                f"one of {SITE_KINDS}")
        lon, lat = _check_lonlat(self.lon, self.lat, self.site_id)
        object.__setattr__(self, "lon", lon)
        object.__setattr__(self, "lat", lat)
        if self.bearing_deg is not None:
            bearing = _finite(self.bearing_deg, f"{self.site_id} bearing_deg")
            if not 0.0 <= bearing < 360.0:
                raise ContractError(f"{self.site_id}: bearing_deg {bearing} "
                                    "is outside [0, 360)")
            object.__setattr__(self, "bearing_deg", bearing)
        if self.chainage_m is not None:
            chainage = _finite(self.chainage_m, f"{self.site_id} chainage_m")
            if chainage < 0.0:
                raise ContractError(f"{self.site_id}: chainage_m is negative")
            object.__setattr__(self, "chainage_m", chainage)
        for name in ("voltage_kv", "hub_height_m", "capacity_mw"):
            object.__setattr__(self, name, _optional_positive(
                getattr(self, name), f"{self.site_id} {name}"))


@dataclass
class SiteSet:
    """A ``sites.v1`` document."""

    sites: list[Site]
    heights_m: tuple[float, ...]
    spacing_m: float | None = None
    assets_ref: dict | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        heights = tuple(_finite(h, "heights_m") for h in self.heights_m)
        if not heights:
            raise ContractError("heights_m must name at least one height")
        if any(h <= 0.0 for h in heights):
            raise ContractError("heights_m must be above ground (positive)")
        if list(heights) != sorted(set(heights)):
            raise ContractError("heights_m must be strictly increasing")
        self.heights_m = heights
        self.spacing_m = _optional_positive(self.spacing_m, "spacing_m")
        seen: set[str] = set()
        for site in self.sites:
            if site.site_id in seen:
                raise ContractError(f"duplicate site_id {site.site_id!r}")
            seen.add(site.site_id)

    def __len__(self) -> int:
        return len(self.sites)

    def column(self, name: str) -> list[Any]:
        return [getattr(site, name) for site in self.sites]

    def as_arrays(self) -> dict[str, Any]:
        """Columns as numpy arrays (floats with NaN for nulls)."""

        import numpy as np

        out: dict[str, Any] = {}
        for name, nullable in SITE_COLUMNS:
            values = self.column(name)
            if name in ("site_id", "asset_id", "kind"):
                out[name] = np.asarray(values, dtype=object)
            else:
                out[name] = np.asarray(
                    [np.nan if v is None else v for v in values],
                    dtype=np.float64)
        return out

    def to_json(self) -> dict[str, Any]:
        return {"schema": SITES_SCHEMA,
                "heights_m": list(self.heights_m),
                "spacing_m": self.spacing_m,
                "assets": self.assets_ref,
                "provenance": self.provenance,
                "count": len(self.sites),
                "columns": {name: self.column(name)
                            for name, _ in SITE_COLUMNS}}

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> "SiteSet":
        _schema_of(document, SITES_SCHEMA, "sites document")
        columns = document.get("columns") or {}
        missing = [name for name, _ in SITE_COLUMNS if name not in columns]
        if missing:
            raise ContractError(f"sites document lacks columns {missing}")
        lengths = {len(columns[name]) for name, _ in SITE_COLUMNS}
        if len(lengths) != 1:
            raise ContractError("sites document columns differ in length")
        count = lengths.pop()
        if document.get("count", count) != count:
            raise ContractError("sites document count disagrees with columns")
        for name, nullable in SITE_COLUMNS:
            if not nullable and any(v is None for v in columns[name]):
                raise ContractError(f"sites column {name} has nulls")
        sites = [Site(**{name: columns[name][i] for name, _ in SITE_COLUMNS})
                 for i in range(count)]
        return cls(sites=sites, heights_m=tuple(document.get("heights_m") or ()),
                   spacing_m=document.get("spacing_m"),
                   assets_ref=document.get("assets"),
                   provenance=dict(document.get("provenance") or {}))


def load_sites(path: str | Path) -> SiteSet:
    return SiteSet.from_json(_read_json(path, "sites document"))


def dump_sites(site_set: SiteSet, path: str | Path) -> Path:
    return _write_json(site_set.to_json(), path)


# --------------------------------------------------------------------------
# plan


@dataclass(frozen=True)
class PlanDomain:
    """One domain of a plan.

    Paths are relative to the plan file's directory.

    ``config``/``wps_namelist``
        The emitted WOOF experiment TOML and its namelist.wps (WRF
        topologies).  Several ``wrf-nests`` domains share one ``config`` and
        differ by ``grid_id``.
    ``grid_id``
        WRF domain number inside ``config`` (1 for a root).
    ``parent``
        ``domain_id`` of the domain whose *output* forces this one: the
        parent run of a ``wrf-tiles`` child.  ``None`` for roots and for
        ``wrf-nests`` children (they are forced inside the same run).
    ``run_dir``/``output_glob``
        Where ``woof energy run`` puts the domain's output and the glob,
        relative to ``run_dir``, matching its history files
        (``wrfout_d02_*``; ``history.*.nc`` for MPAS).
    ``footprint``
        Closed ``[lon, lat]`` ring of the domain's mass-point extent.
    ``site_ids``
        Sites this domain is the finest owner of.  Each site has at most
        one owner across a plan.
    ``mesh``
        ``hex-swath`` only: ``{"mesh_spec": path, "cull_region": path, ...}``.
    """

    domain_id: str
    topology: str
    role: str
    dx_m: float
    run_dir: str
    output_glob: str
    footprint: tuple[tuple[float, float], ...]
    config: str | None = None
    wps_namelist: str | None = None
    grid_id: int | None = None
    parent: str | None = None
    site_ids: tuple[str, ...] = ()
    mesh: dict | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.domain_id:
            raise ContractError("domain_id must be non-empty")
        if self.topology not in TOPOLOGIES:
            raise ContractError(f"{self.domain_id}: topology "
                                f"{self.topology!r} is not one of {TOPOLOGIES}")
        if self.role not in DOMAIN_ROLES:
            raise ContractError(f"{self.domain_id}: role {self.role!r} is "
                                f"not one of {DOMAIN_ROLES}")
        dx = _finite(self.dx_m, f"{self.domain_id} dx_m")
        if dx <= 0.0:
            raise ContractError(f"{self.domain_id}: dx_m must be positive")
        object.__setattr__(self, "dx_m", dx)
        ring = tuple(tuple(_check_lonlat(p[0], p[1],
                                         f"{self.domain_id} footprint"))
                     for p in self.footprint)
        if len(ring) < 4 or ring[0] != ring[-1]:
            raise ContractError(f"{self.domain_id}: footprint must be a "
                                "closed ring of at least four positions")
        object.__setattr__(self, "footprint", ring)
        object.__setattr__(self, "site_ids", tuple(self.site_ids))
        for name in ("run_dir", "output_glob", "config", "wps_namelist"):
            value = getattr(self, name)
            if value is not None and Path(value).is_absolute():
                raise ContractError(f"{self.domain_id}: {name} must be "
                                    "relative to the plan directory")
        if self.topology.startswith("wrf-") and self.role != "mesh":
            if self.config is None or self.grid_id is None:
                raise ContractError(f"{self.domain_id}: a WRF domain needs "
                                    "config and grid_id")
        if self.topology == "hex-swath" and not self.mesh:
            raise ContractError(f"{self.domain_id}: a hex-swath domain needs "
                                "mesh documents")
        if self.role == "parent" and self.site_ids:
            raise ContractError(f"{self.domain_id}: parent domains own no "
                                "sites")


@dataclass
class Plan:
    """A ``plan.v1`` document."""

    topology: str
    dx_m: float
    start: str
    hours: float
    domains: list[PlanDomain]
    source: str | None = None
    sites_ref: dict | None = None
    notes: list[str] = field(default_factory=list)
    path: Path | None = field(default=None, compare=False)

    def __post_init__(self):
        if self.topology not in TOPOLOGIES:
            raise ContractError(f"plan topology {self.topology!r} is not one "
                                f"of {TOPOLOGIES}")
        self.dx_m = _finite(self.dx_m, "plan dx_m")
        self.hours = _finite(self.hours, "plan hours")
        if self.hours <= 0.0:
            raise ContractError("plan hours must be positive")
        ids = [d.domain_id for d in self.domains]
        if len(ids) != len(set(ids)):
            raise ContractError("plan domain_ids are not unique")
        known = set(ids)
        owners: dict[str, str] = {}
        for domain in self.domains:
            if domain.topology != self.topology:
                raise ContractError(f"{domain.domain_id}: topology differs "
                                    "from the plan's")
            if domain.parent is not None and domain.parent not in known:
                raise ContractError(f"{domain.domain_id}: parent "
                                    f"{domain.parent!r} is not in the plan")
            for site_id in domain.site_ids:
                if site_id in owners:
                    raise ContractError(f"site {site_id!r} is owned by both "
                                        f"{owners[site_id]} and "
                                        f"{domain.domain_id}")
                owners[site_id] = domain.domain_id

    @property
    def directory(self) -> Path:
        if self.path is None:
            raise ContractError("plan has no file location to resolve against")
        return self.path.parent

    def resolve(self, relative: str) -> Path:
        return self.directory / relative

    def domain(self, domain_id: str) -> PlanDomain:
        for candidate in self.domains:
            if candidate.domain_id == domain_id:
                return candidate
        raise ContractError(f"plan has no domain {domain_id!r}")

    def run_order(self) -> list[PlanDomain]:
        """Domains with every ``parent`` before its children."""

        placed: list[PlanDomain] = []
        done: set[str] = set()
        pending = list(self.domains)
        while pending:
            ready = [d for d in pending if d.parent is None or d.parent in done]
            if not ready:
                raise ContractError("plan parent links form a cycle")
            for domain in ready:
                placed.append(domain)
                done.add(domain.domain_id)
                pending.remove(domain)
        return placed

    def to_json(self) -> dict[str, Any]:
        domains = []
        for domain in self.domains:
            record = asdict(domain)
            record["footprint"] = [list(p) for p in domain.footprint]
            record["site_ids"] = list(domain.site_ids)
            domains.append(record)
        return {"schema": PLAN_SCHEMA, "topology": self.topology,
                "dx_m": self.dx_m, "start": self.start, "hours": self.hours,
                "source": self.source, "sites": self.sites_ref,
                "notes": list(self.notes), "domains": domains}

    @classmethod
    def from_json(cls, document: Mapping[str, Any], *,
                  path: str | Path | None = None) -> "Plan":
        _schema_of(document, PLAN_SCHEMA, "plan document")
        known = {f.name for f in fields(PlanDomain)}
        domains = []
        for record in document.get("domains") or []:
            unknown = set(record) - known
            if unknown:
                raise ContractError(f"plan domain has unknown fields "
                                    f"{sorted(unknown)}")
            record = dict(record)
            record["footprint"] = tuple(tuple(p) for p in record["footprint"])
            record["site_ids"] = tuple(record.get("site_ids") or ())
            record["extra"] = dict(record.get("extra") or {})
            domains.append(PlanDomain(**record))
        try:
            return cls(topology=document["topology"], dx_m=document["dx_m"],
                       start=document["start"], hours=document["hours"],
                       domains=domains, source=document.get("source"),
                       sites_ref=document.get("sites"),
                       notes=list(document.get("notes") or []),
                       path=Path(path) if path is not None else None)
        except KeyError as error:
            raise ContractError(f"plan document lacks {error}") from error


def load_plan(path: str | Path) -> Plan:
    path = Path(path)
    return Plan.from_json(_read_json(path, "plan document"), path=path)


def dump_plan(plan: Plan, path: str | Path) -> Path:
    plan.path = Path(path)
    return _write_json(plan.to_json(), path)


# --------------------------------------------------------------------------
# forecast

#: Dimension tuples used in the forecast tables.
PROFILE_DIMS = ("time", "site", "height")
SURFACE_DIMS = ("time", "site")

#: ``forecast.v1`` data variables: name -> (dims, units, long_name).
#: Winds are earth-relative.  ``wind_from_direction`` is meteorological
#: (direction the wind blows FROM, clockwise from true north).
#: ``line_normal_wind`` is the wind component perpendicular to the site's
#: ``bearing_deg`` (absolute value; NaN where the site has no bearing) and
#: ``wind_attack_angle`` the acute angle between the wind and the conductor
#: axis, 0-90 degrees -- the two inputs conductor-cooling models need.
FORECAST_VARIABLES: dict[str, tuple[tuple[str, ...], str, str]] = {
    "u": (PROFILE_DIMS, "m s-1", "eastward wind"),
    "v": (PROFILE_DIMS, "m s-1", "northward wind"),
    "w": (PROFILE_DIMS, "m s-1", "upward air velocity"),
    "wind_speed": (PROFILE_DIMS, "m s-1", "wind speed"),
    "wind_from_direction": (PROFILE_DIMS, "degree", "wind from direction"),
    "line_normal_wind": (PROFILE_DIMS, "m s-1",
                         "wind component normal to the conductor"),
    "wind_attack_angle": (PROFILE_DIMS, "degree",
                          "angle between wind and conductor axis"),
    "air_temperature": (PROFILE_DIMS, "K", "air temperature"),
    "air_pressure": (PROFILE_DIMS, "Pa", "air pressure"),
    "air_density": (PROFILE_DIMS, "kg m-3", "air density"),
    "specific_humidity": (PROFILE_DIMS, "kg kg-1", "specific humidity"),
    "relative_humidity": (PROFILE_DIMS, "%", "relative humidity"),
    "cloud_liquid_mixing_ratio": (PROFILE_DIMS, "kg kg-1",
                                  "cloud liquid water mixing ratio"),
    "rain_mixing_ratio": (PROFILE_DIMS, "kg kg-1", "rain mixing ratio"),
    "ice_mixing_ratio": (PROFILE_DIMS, "kg kg-1", "cloud ice mixing ratio"),
    "snow_mixing_ratio": (PROFILE_DIMS, "kg kg-1", "snow mixing ratio"),
    "t2": (SURFACE_DIMS, "K", "2 m air temperature"),
    "q2": (SURFACE_DIMS, "kg kg-1", "2 m water vapour mixing ratio"),
    "rh2": (SURFACE_DIMS, "%", "2 m relative humidity"),
    "u10": (SURFACE_DIMS, "m s-1", "10 m eastward wind"),
    "v10": (SURFACE_DIMS, "m s-1", "10 m northward wind"),
    "wind_speed_10m": (SURFACE_DIMS, "m s-1", "10 m wind speed"),
    "psfc": (SURFACE_DIMS, "Pa", "surface pressure"),
    "ghi": (SURFACE_DIMS, "W m-2", "global horizontal irradiance"),
    "dni": (SURFACE_DIMS, "W m-2", "direct normal irradiance"),
    "dhi": (SURFACE_DIMS, "W m-2", "diffuse horizontal irradiance"),
    "cos_solar_zenith": (SURFACE_DIMS, "1", "cosine of solar zenith angle"),
    "precipitation_rate": (SURFACE_DIMS, "kg m-2 s-1",
                           "total precipitation rate"),
}

#: ``forecast.v1`` coordinates: name -> (dims, units-or-None, long_name).
#: ``inside`` is 0 for a site its owning domain could not sample (all of
#: its data NaN); ``domain_id`` names that owner.
FORECAST_COORDINATES: dict[str, tuple[tuple[str, ...], str | None, str]] = {
    "time": (("time",), None, "valid time (UTC)"),
    "height": (("height",), "m", "height above ground level"),
    "site_id": (("site",), None, "site identifier"),
    "asset_id": (("site",), None, "asset identifier"),
    "kind": (("site",), None, "site kind"),
    "lat": (("site",), "degree_north", "latitude"),
    "lon": (("site",), "degree_east", "longitude"),
    "bearing_deg": (("site",), "degree", "conductor azimuth"),
    "terrain_height": (("site",), "m", "model terrain height"),
    "domain_id": (("site",), None, "owning plan domain"),
    "dx_m": (("site",), "m", "grid spacing of the owning domain"),
    "inside": (("site",), "1", "1 where the owning domain sampled the site"),
}


__all__ = [
    "ASSETS_SCHEMA", "SITES_SCHEMA", "PLAN_SCHEMA", "FORECAST_SCHEMA",
    "ASSET_KINDS", "LINEAR_KINDS", "GENERATOR_SOURCES", "SITE_KINDS",
    "TOPOLOGIES", "DOMAIN_ROLES", "SITE_COLUMNS",
    "PROFILE_DIMS", "SURFACE_DIMS", "FORECAST_VARIABLES",
    "FORECAST_COORDINATES",
    "ContractError", "EnergyNotImplemented",
    "Asset", "AssetCollection", "Site", "SiteSet", "PlanDomain", "Plan",
    "load_assets", "dump_assets", "load_sites", "dump_sites", "load_plan",
    "dump_plan", "validate_geometry", "sha256_file", "file_ref",
]
