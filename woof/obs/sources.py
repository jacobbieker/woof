"""Observation sources: what the scorer asks of an archive, answered.

These are the ingest lane's side of the seam declared in
``woof.verify.obs.contracts``. Each class satisfies one of that module's
protocols over packs and records the Rust front doors wrote:

* :class:`MrmsCompositeSource`, ``GriddedObsSource`` for composite
  reflectivity in dBZ;
* :class:`OperaCompositeSource`, the same quantity from the European
  network, for domains MRMS does not cover;
* :class:`Stage4PrecipSource`, ``GriddedObsSource`` for one accumulation
  window of precipitation in mm;
* :class:`AsosSurfaceSource`: ``StationObsSource`` for surface reports in
  SI.
* :class:`DynamicalAsosSurfaceSource`: the same protocol over the
  Dynamical.org ASOS Parquet re-packaging of the IEM archive, fetched per
  valid time through the optional :mod:`woof.obs.dynamical_asos` reader, so
  stations can be scored anywhere that archive has them.

The contracts module is imported lazily, and deliberately. The scorer lane
and this lane land in separate integration waves; a module-level import
would make every ingest test depend on the other lane's tree being present,
which is exactly the coupling the seam exists to avoid. Constructing a
source without the contracts module raises and says so.

**Frame selection.** A source answers a valid time with the nearest frame
inside its registered matching window. Under a registered coverage floor
(``minimum_observed_fraction``, registration v2.1) it answers with the
nearest frame that *meets* the floor, walking outward inside the same
window; the window never widens. When the window holds frames but none of
them clears the floor, the source raises the seam's
``ObservedFractionBelowFloor`` so the scorer can record that lead as
missing-obs by name -- an upstream ingest outage should not be scored as if
it were weather, and it should not kill a case either.

**Re-hashing.** Every source exposes ``verify(provenance)``, which re-reads
the archive object the provenance names and compares its digest to the one
taken at fetch. The promotion rule's integrity clause requires that round
trip on every scored input, and an unperformed check counts as a failed one:
a missing artifact raises (that is a wiring fault) and a changed one returns
False (that is the corruption this exists to catch).
"""

from __future__ import annotations

import bisect
import hashlib
import importlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath

import numpy as np

from woof.obs.obspack import read_geo_pack, read_grid_pack, sha256_of

#: Seam quantity ids, repeated here so a source can name its own quantity
#: without importing the contracts module at import time.
QUANTITY_COMPOSITE_REFLECTIVITY = "composite_reflectivity"
QUANTITY_PRECIPITATION_ACCUMULATION = "precipitation_accumulation"

_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

#: Default matching window for a gridded frame, in seconds. The battery
#: registers +/- 4 minutes around each hourly valid time for the two-minute
#: reflectivity cadence.
DEFAULT_MATCH_SECONDS = 240


def _contracts():
    """The scorer's contract module, imported at use rather than at import."""

    try:
        from woof.verify.obs import contracts
    except ImportError as error:  # pragma: no cover - exercised at integration
        raise RuntimeError(
            "woof.verify.obs.contracts is not importable; the observation "
            "sources build the scorer's own dataclasses and cannot do so "
            "without it. This module is the scoring lane's; a tree that has "
            "the ingest lane but not the scorer lane can still fetch, decode "
            "and verify, but cannot construct seam objects."
        ) from error
    return contracts


def _parse_time(value: str) -> datetime:
    return datetime.strptime(str(value), _TIME_FORMAT)


@dataclass(frozen=True)
class _Frame:
    """One decoded frame on disk, indexed by its own valid time.

    ``observed_fraction`` is the share of the packed subdomain the analysis
    actually observed, as the front door recorded it. ``None`` means the
    pack does not carry the number, and an unknown coverage is never screened
    on -- that is the behaviour every frame had before the coverage floor
    existed, and inventing a value for it would be worse than not screening.
    """

    valid_time: datetime
    pack_path: Path
    observed_fraction: float | None = None


def _index_packs(paths, quantity: str) -> list[_Frame]:
    """Read each pack's metadata and index it by valid time.

    Only the metadata is read here; the payload is left on disk until a
    frame is actually asked for, because a 24-hour case is seventeen scored
    frames out of seven hundred available ones.
    """

    frames: list[_Frame] = []
    for path in paths:
        pack = read_grid_pack(path)
        found = str(pack.meta.get("quantity", ""))
        if found != quantity:
            raise ValueError(
                f"{path} carries quantity {found!r}, this source serves "
                f"{quantity!r}")
        sentinels = dict(pack.meta.get("sentinels", {}) or {})
        observed = sentinels.get("observed_fraction")
        frames.append(_Frame(valid_time=_parse_time(pack.meta["valid_time"]),
                             pack_path=Path(path),
                             observed_fraction=(None if observed is None
                                                else float(observed))))
    frames.sort(key=lambda frame: frame.valid_time)
    times = [frame.valid_time for frame in frames]
    duplicates = {t for t in times if times.count(t) > 1}
    if duplicates:
        raise ValueError(
            f"two packs claim the same valid time(s): "
            f"{sorted(t.strftime(_TIME_FORMAT) for t in duplicates)}; which "
            f"one is the observation would then be an ordering accident")
    return frames


class _GriddedSource:
    """Shared machinery for a gridded archive read out of packs."""

    def __init__(self, pack_paths, geo_pack, *, quantity: str, units: str,
                 match_seconds: int = DEFAULT_MATCH_SECONDS,
                 source: str = "", product: str = "",
                 minimum_observed_fraction: float | None = None):
        paths = [Path(p) for p in pack_paths]
        if not paths:
            raise ValueError(
                f"a {quantity} source needs at least one pack; an empty "
                f"source scores every arm on nothing and reports a clean zero")
        if match_seconds <= 0:
            raise ValueError("match_seconds must be positive")
        if (minimum_observed_fraction is not None
                and not 0.0 <= float(minimum_observed_fraction) <= 1.0):
            raise ValueError("a coverage floor is a fraction in [0, 1]")
        self.minimum_observed_fraction = (
            None if minimum_observed_fraction is None
            else float(minimum_observed_fraction))
        self._frames = _index_packs(paths, quantity)
        self._times = [frame.valid_time for frame in self._frames]
        self._geo_pack = Path(geo_pack)
        self._quantity = quantity
        self._units = units
        self._match = timedelta(seconds=int(match_seconds))
        self._source = source
        self._product = product
        self._geo: tuple[np.ndarray, np.ndarray] | None = None

    # -- the GriddedObsSource protocol ---------------------------------

    def quantity(self) -> str:
        """The seam quantity id this source serves."""

        return self._quantity

    def field(self, valid_time: str):
        """The observed field nearest ``valid_time`` within the window."""

        contracts = _contracts()
        target = contracts.parse_valid_time(valid_time)
        frame = self._nearest(target)
        pack = read_grid_pack(frame.pack_path)
        values = np.asarray(pack.array("values"), dtype=np.float64)
        valid = np.asarray(pack.array("valid")).astype(bool)
        latitude, longitude = self._geometry()
        if latitude.shape != values.shape:
            raise ValueError(
                f"{frame.pack_path}: values {values.shape} do not match the "
                f"geometry pack {self._geo_pack} {latitude.shape}; a field "
                f"and the coordinates it is scored on must be one grid")
        meta_provenance = dict(pack.meta.get("provenance", {}))
        provenance = contracts.ObsProvenance(
            source=meta_provenance.get("source", self._source),
            product=meta_provenance.get("product", self._product),
            uri=meta_provenance.get("uri", str(frame.pack_path)),
            sha256=meta_provenance.get("sha256", ""),
            fetched_at=meta_provenance.get("fetched_at", ""),
        )
        return contracts.ObsGridField(
            quantity=self._quantity,
            valid_time=contracts.format_valid_time(frame.valid_time),
            values=values,
            valid=valid,
            latitude=latitude,
            longitude=longitude,
            units=self._units,
            provenance=provenance,
        )

    # -- integrity ------------------------------------------------------

    def verify(self, provenance, *, root=None) -> bool:
        """Re-hash the archive object ``provenance`` names.

        Returns True when the bytes still hash to the digest taken at fetch.
        A missing artifact raises rather than returning False: that is a
        wiring fault, and reporting it as a corruption would send the reader
        looking for a disk problem that is not there.

        The front doors record an absolute path, so the common case needs
        nothing else. ``root`` covers the case where the cache was moved
        between fetching and scoring: the object is then looked for by its
        own basename under ``root``. It is an explicit argument rather than a
        fallback the reader applies on its own, because a search that happens
        silently is a search that will one day find a different file with the
        same name and report it verified.

        The basename is taken with the rule that reads *both* separators,
        because the recorded path carries the separators of whichever box
        fetched the bytes. A POSIX ``Path`` does not split a Windows path at
        all -- ``name`` is then the whole string -- so relocating a
        Windows-pulled archive onto a Linux node, which is the direction this
        argument exists for, would otherwise look for one absurdly named file
        and raise. Nothing about the comparison changes: the digest still
        decides, and an object that is not the registered one fails.
        """

        uri = getattr(provenance, "uri", None) or dict(provenance)["uri"]
        expected = (getattr(provenance, "sha256", None)
                    or dict(provenance)["sha256"])
        path = Path(uri)
        if root is not None and not path.is_file():
            path = Path(root) / PureWindowsPath(str(uri)).name
        if not path.is_file():
            raise FileNotFoundError(
                f"cannot re-hash {path}: the archive object this field's "
                f"provenance names is not on disk")
        return sha256_of(path) == str(expected).lower()

    # -- internals ------------------------------------------------------

    def valid_times(self) -> tuple[str, ...]:
        """Every valid time this source holds a frame for, ascending."""

        return tuple(t.strftime(_TIME_FORMAT) for t in self._times)

    def _geometry(self) -> tuple[np.ndarray, np.ndarray]:
        if self._geo is None:
            self._geo = read_geo_pack(self._geo_pack)
        return self._geo

    def _inside_window(self, target: datetime) -> list[_Frame]:
        """Every frame inside the matching window, nearest first.

        The order is the selection rule, written once: ascending ``|dt|``,
        and on an exact tie the earlier frame. Deterministic and independent
        of pack order on disk, because a selection that depends on a
        directory listing is a selection nobody can reproduce.
        """

        return sorted(
            (frame for frame in self._frames
             if abs(frame.valid_time - target) <= self._match),
            key=lambda frame: (abs(frame.valid_time - target),
                               frame.valid_time))

    def _nearest(self, target: datetime) -> _Frame:
        """The frame this source scores at ``target``.

        Nearest inside the matching window, and -- when a coverage floor is
        registered -- the nearest one that *meets* it. The window itself
        never widens: a better-covered frame outside it is not a candidate,
        because a distant frame scored as coincident is a wrong number that
        looks like a right one.

        When the window holds frames but none of them clears the floor, this
        raises the seam's own
        :class:`~woof.verify.obs.contracts.ObservedFractionBelowFloor`
        rather than the plain frame-search failure, so the scorer can record
        the lead as missing-obs by name instead of scoring a mask.
        """

        inside = self._inside_window(target)
        if not inside:
            position = bisect.bisect_left(self._times, target)
            best: _Frame | None = None
            for index in (position - 1, position):
                if 0 <= index < len(self._frames):
                    candidate = self._frames[index]
                    if best is None or abs(candidate.valid_time - target) < abs(
                            best.valid_time - target):
                        best = candidate
            offset = ("none" if best is None
                      else f"{abs(best.valid_time - target).total_seconds():.0f} s")
            raise LookupError(
                f"no {self._quantity} frame within "
                f"{self._match.total_seconds():.0f} s of "
                f"{target.strftime(_TIME_FORMAT)} (nearest: {offset}); "
                f"refusing rather than reaching further, because a distant "
                f"frame scored as coincident is a wrong number that looks "
                f"like a right one")

        floor = self.minimum_observed_fraction
        if floor is None:
            return inside[0]
        for frame in inside:
            # An unrecorded coverage is not screened on; see :class:`_Frame`.
            if frame.observed_fraction is None or frame.observed_fraction >= floor:
                return frame

        contracts = _contracts()
        candidates = [
            {"valid_time": frame.valid_time.strftime(_TIME_FORMAT),
             "offset_seconds": abs(
                 (frame.valid_time - target).total_seconds()),
             "observed_fraction": frame.observed_fraction}
            for frame in inside]
        raise contracts.ObservedFractionBelowFloor(
            f"every {self._quantity} frame within "
            f"{self._match.total_seconds():.0f} s of "
            f"{target.strftime(_TIME_FORMAT)} observes less than "
            f"{floor:.4g} of the packed subdomain "
            f"({', '.join(str(row['observed_fraction']) for row in candidates)}"
            f"); the window is not widened for a better-covered frame, and "
            f"this lead is missing observations rather than mostly mask",
            valid_time=target.strftime(_TIME_FORMAT),
            minimum_observed_fraction=float(floor),
            candidates=candidates)


class MrmsCompositeSource(_GriddedSource):
    """MRMS composite reflectivity, in dBZ, from decoded packs.

    ``minimum_observed_fraction`` is the registered coverage floor for frame
    selection (registration v2.1). Left unset, selection is nearest-frame
    exactly as before.
    """

    def __init__(self, pack_paths, geo_pack, *,
                 match_seconds: int = DEFAULT_MATCH_SECONDS,
                 minimum_observed_fraction: float | None = None):
        super().__init__(
            pack_paths, geo_pack,
            quantity=QUANTITY_COMPOSITE_REFLECTIVITY, units="dBZ",
            match_seconds=match_seconds, source="mrms",
            product="MergedReflectivityQCComposite_00.50",
            minimum_observed_fraction=minimum_observed_fraction)


class OperaCompositeSource(_GriddedSource):
    """EUMETNET OPERA composite reflectivity, in dBZ, from decoded packs.

    The European counterpart of :class:`MrmsCompositeSource`, and deliberately
    the same class of thing: same seam quantity, same units, same pack schema,
    same coverage-floor behaviour. A scorer picks between them by which
    network covers the domain, not by which code path it has to take.

    Two differences are real and both are defaults rather than logic. The
    cadence is five minutes rather than two, so the matching window is 300 s;
    and the product is the OPERA column-maximum composite rather than the
    MRMS one, which is what the provenance says when a pack does not carry
    its own.

    One caveat travels with every frame and is not this class's to resolve:
    the composite's own ``/how/comment`` can declare that part of the field
    was produced by advection extrapolation rather than observed. The front
    door records that note in the pack's ``production`` block. It is a
    statement about what the numbers are, and a verification campaign should
    read it before treating the field as ground truth.
    """

    def __init__(self, pack_paths, geo_pack, *,
                 match_seconds: int = 300,
                 minimum_observed_fraction: float | None = None):
        super().__init__(
            pack_paths, geo_pack,
            quantity=QUANTITY_COMPOSITE_REFLECTIVITY, units="dBZ",
            match_seconds=match_seconds, source="opera",
            product="opera-comp-dbzh-max",
            minimum_observed_fraction=minimum_observed_fraction)


class Stage4PrecipSource(_GriddedSource):
    """Stage-IV precipitation for ONE accumulation window, in mm.

    One instance serves one window; the scorer holds a mapping
    ``{1: hourly, 6: six_hourly}`` because that is how the archive stores
    them and how the spec scores them.
    """

    def __init__(self, pack_paths, geo_pack, *, accumulation_hours: int,
                 match_seconds: int = 1800):
        super().__init__(
            pack_paths, geo_pack,
            quantity=QUANTITY_PRECIPITATION_ACCUMULATION, units="mm",
            match_seconds=match_seconds, source="stage4",
            product=f"ST4-{int(accumulation_hours):02d}h")
        self.accumulation_hours = int(accumulation_hours)
        for path in (frame.pack_path for frame in self._frames):
            pack = read_grid_pack(path)
            found = int(pack.meta.get("accumulation_hours", -1))
            if found != self.accumulation_hours:
                raise ValueError(
                    f"{path} is a {found}-hour accumulation; this source "
                    f"serves {self.accumulation_hours}-hour. Mixing windows "
                    f"would compare one model total against another's obs")


def _asos_record_rows(record, valid_times, contracts):
    """The seam stations and reports of one ``asos-surface`` record.

    Shared by every reader of the record format, whichever archive the record
    was built from, so the IEM front door's records and the Dynamical.org
    re-packaging are parsed by one function and cannot drift.  Reports are
    kept only at the requested ``valid_times``; every station the record
    froze is returned, reporting or not, because the scorer's completeness
    screen needs to see the silent ones.
    """

    wanted = {str(t) for t in valid_times}
    for text in wanted:
        contracts.parse_valid_time(text)
    stations = tuple(
        contracts.Station(
            station_id=str(row["station_id"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            elevation_m=float(row["elevation_m"]),
        )
        for row in record.get("stations", ())
    )
    reports = tuple(
        contracts.StationReport(
            station_id=str(row["station_id"]),
            valid_time=str(row["valid_time"]),
            values={str(k): float(v)
                    for k, v in dict(row.get("values", {})).items()},
            flags=tuple(str(f) for f in row.get("flags", ())),
        )
        for row in record.get("reports", ())
        if str(row["valid_time"]) in wanted
    )
    return stations, reports


def _read_asos_record(path: Path, schemas) -> dict:
    record = json.loads(Path(path).read_text())
    schema = str(record.get("schema", ""))
    if schema not in schemas:
        raise ValueError(
            f"{path} declares schema {schema!r}, expected one of "
            f"{list(schemas)}")
    return record


class AsosSurfaceSource:
    """Surface reports from a decoded ``gpuwm-obs.asos-surface`` record.

    Reads ``v2`` (every report carries its ``observation_time``) and the
    ``v1`` records written before it, whose reports are dated to the hour
    they were matched to.
    """

    SCHEMAS = ("gpuwm-obs.asos-surface.v2", "gpuwm-obs.asos-surface.v1")

    def __init__(self, record_path):
        self.path = Path(record_path)
        self.record = _read_asos_record(self.path, self.SCHEMAS)

    # -- the StationObsSource protocol ----------------------------------

    def observations(self, valid_times):
        """Every report covering ``valid_times``, with its stations."""

        contracts = _contracts()
        stations, reports = _asos_record_rows(self.record, valid_times,
                                              contracts)
        meta = dict(self.record.get("provenance", {}))
        if meta.get("source") == DYNAMICAL_SOURCE:
            # One hour the Dynamical.org reader wrote, scored on its own: its
            # upstream object is a remote Parquet year that cannot be
            # re-hashed here, so the record itself is the archive object, and
            # the reader's UTC stamp is put in seam spelling.
            meta["uri"] = str(self.path.resolve())
            meta["sha256"] = sha256_of(self.path)
            meta["fetched_at"] = (_seam_time_or_none(
                meta.get("fetched_at"), contracts) or meta.get("fetched_at"))
            meta["attribution"] = str(meta.get("attribution")
                                      or DYNAMICAL_ATTRIBUTION)
        provenance = contracts.ObsProvenance(
            source=meta.get("source", "asos"),
            product=meta.get("product", "iem-asos-metar"),
            uri=meta.get("uri", str(self.path)),
            sha256=meta.get("sha256", ""),
            fetched_at=meta.get("fetched_at", ""),
            attribution=str(meta.get("attribution", "") or ""),
        )
        return contracts.StationObsSet(stations=stations, reports=reports,
                                       provenance=provenance)

    # -- integrity ------------------------------------------------------

    verify = _GriddedSource.verify

    @property
    def station_table_sha256(self) -> str:
        """The digest of the frozen station table this record was built on."""

        return str(self.record.get("station_table_sha256", ""))


#: The archive id the Dynamical.org ASOS Parquet reader stamps into every
#: record it writes (``woof.obs.dynamical_asos.SOURCE``), repeated here so the
#: source can be named, selected and routed for re-hashing without importing
#: the reader -- which needs ``pyarrow`` and is optional.
DYNAMICAL_SOURCE = "dynamical-asos-parquet"

#: The product name, the catalog's own id for the dataset.
DYNAMICAL_PRODUCT = "asos-parquet"

#: The credit line the dataset asks its users to carry.  It travels in every
#: provenance this source builds and so in every score file scored against it.
DYNAMICAL_ATTRIBUTION = (
    "ASOS/AWOS observations from the Iowa Environmental Mesonet (Iowa State "
    "University); original reports NOAA/NWS/FAA (public domain); Parquet "
    "re-packaging by dynamical.org (https://dynamical.org/catalog/"
    "asos-parquet/, marked experimental); hosted by Source Cooperative")

#: Schema of the manifest one ``observations`` call writes over its per-hour
#: records.  The set's provenance names this file and its digest, and the
#: manifest names every record with its own, so one re-hash covers them all.
DYNAMICAL_MANIFEST_SCHEMA = "gpuwm-obs.dynamical-asos-manifest.v1"

_DYNAMICAL_MODULE = "woof.obs.dynamical_asos"


class ObsSourceUnavailable(RuntimeError):
    """An observation source that cannot be read here, said by name.

    Distinct from a ``LookupError`` (the archive was read and holds nothing
    for the request) and from a ``ValueError`` (the bytes were read and are
    wrong): this is the source itself being out of reach -- its optional
    reader is not installed, its dependency is missing, its host did not
    answer.  It never manufactures a stand-in (that is
    :mod:`woof.verify.obs.stubs`' job, behind its acknowledgement); a caller
    states it and stops, or scores without the source.
    """

    def __init__(self, source: str, reason: str):
        super().__init__(f"observation source {source!r} is unavailable: "
                         f"{reason}")
        self.source = str(source)
        self.reason = str(reason)


def _dynamical_module():
    """The Dynamical.org ASOS reader, imported at use, or a typed refusal.

    ``importlib`` rather than ``from woof.obs import ...`` so a module entry
    in ``sys.modules`` (including a ``None`` that blocks it) is what decides,
    not an attribute left on the package by an earlier import.
    """

    try:
        return importlib.import_module(_DYNAMICAL_MODULE)
    except ImportError as error:
        raise ObsSourceUnavailable(
            DYNAMICAL_SOURCE,
            f"{_DYNAMICAL_MODULE} cannot be imported ({error}); the "
            f"Dynamical.org ASOS Parquet reader and its pyarrow dependency "
            f"are needed to fetch this archive") from error


def _check_bbox(bbox) -> tuple[float, float, float, float]:
    try:
        west, south, east, north = (float(v) for v in bbox)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"a bounding box is four numbers west, south, east, north; got "
            f"{bbox!r}") from error
    if not all(math.isfinite(v) for v in (west, south, east, north)):
        raise ValueError(f"bounding box {bbox!r} carries a non-finite edge")
    if not -90.0 <= south < north <= 90.0:
        raise ValueError(
            f"bounding box {bbox!r}: need -90 <= south < north <= 90")
    if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0
            and west != east):
        raise ValueError(
            f"bounding box {bbox!r}: need west and east in [-180, 180] and "
            f"apart (west > east is a box across the antimeridian)")
    return west, south, east, north


def parse_bbox(text: str) -> tuple[float, float, float, float]:
    """``W,S,E,N`` from a command line, checked."""

    parts = [p for p in str(text).replace(" ", "").split(",") if p]
    if len(parts) != 4:
        raise ValueError(f"a bounding box is W,S,E,N; got {text!r}")
    return _check_bbox(parts)


def bbox_of(latitude, longitude, *, margin_deg: float = 0.0
            ) -> tuple[float, float, float, float]:
    """The west, south, east, north box around a grid's points.

    Longitudes are wrapped into ``[-180, 180)`` first.  A regional grid
    straddling 180 degrees comes back as a box across the antimeridian
    (``west > east``, the reader's convention); a global grid, whose
    longitudes leave no gap wider than their spacing, comes back as the full
    range.  The scorer's own admission keeps only the stations inside the
    grid either way.
    """

    lat = np.asarray(latitude, dtype=np.float64).ravel()
    lon = _contracts().normalize_longitude(
        np.asarray(longitude, dtype=np.float64).ravel())
    if lat.size == 0 or lon.size == 0:
        raise ValueError("a bounding box needs at least one grid point")
    margin = max(0.0, float(margin_deg))
    south = max(-90.0, float(lat.min()) - margin)
    north = min(90.0, float(lat.max()) + margin)
    ordered = np.unique(lon)
    wrap_gap = float(ordered[0] + 360.0 - ordered[-1])
    gaps = np.diff(ordered)
    inner_gap = float(gaps.max()) if ordered.size > 1 else 0.0
    if inner_gap < wrap_gap:
        west = max(-180.0, float(ordered[0]) - margin)
        east = min(180.0, float(ordered[-1]) + margin)
    elif inner_gap > wrap_gap and inner_gap - 2.0 * margin > 0.0:
        cut = int(np.argmax(gaps))
        west = float(ordered[cut + 1]) - margin
        east = float(ordered[cut]) + margin
    else:
        west, east = -180.0, 180.0
    if north <= south:
        north = min(90.0, south + 1e-6)
    if east == west:
        east = min(180.0, west + 1e-6)
    return _check_bbox((west, south, east, north))


def _seam_time_or_none(text, contracts) -> str | None:
    """A record's timestamp in seam spelling, whichever ISO form it used."""

    value = str(text or "").strip()
    if not value:
        return None
    try:
        contracts.parse_valid_time(value)
        return value
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed.strftime(_TIME_FORMAT)


class DynamicalAsosSurfaceSource:
    """``StationObsSource`` over the Dynamical.org ASOS Parquet archive.

    The archive (https://dynamical.org/catalog/asos-parquet/) re-packages the
    Iowa Environmental Mesonet's ASOS/AWOS METAR archive as yearly Parquet
    files, which makes surface verification possible wherever that archive
    has stations without running the IEM front door.  The reader,
    :mod:`woof.obs.dynamical_asos`, is optional (it needs ``pyarrow``) and is
    imported only when observations are asked for; without it, or when it
    reports the archive unreachable, this source raises
    :class:`ObsSourceUnavailable` and never substitutes anything.

    One ``observations`` call fetches each requested valid time into its own
    folder under ``folder`` -- the reader writes a ``gpuwm-obs.asos-surface``
    v2 record there -- parses every record with the same function that reads
    the IEM front door's records, and merges them into one ``StationObsSet``.
    A valid time with no surviving report is recorded in
    :attr:`empty_valid_times` and is scored as missing; when no valid time
    has a report the call raises ``LookupError``.

    The set's provenance names a manifest written beside the records: its
    digest, and through it every record's digest, is what :meth:`verify`
    re-hashes.  The attribution the dataset asks for travels on the
    provenance and so into every score file.
    """

    SCHEMAS = AsosSurfaceSource.SCHEMAS
    SOURCE = DYNAMICAL_SOURCE

    def __init__(self, folder, *, bbox=None, station_ids=None,
                 timeout: float = 120.0, refresh: bool = False):
        if bbox is None and not station_ids:
            raise ValueError(
                "the Dynamical.org ASOS source needs a bounding box or a "
                "list of station ids; it does not fetch the world by default")
        self.folder = Path(folder)
        # With station ids alone the reader selects by id and takes no box.
        self.bbox = _check_bbox(bbox) if bbox is not None else None
        self.station_ids = (tuple(str(s).strip()
                                  for s in station_ids if str(s).strip())
                            if station_ids else None)
        if station_ids and not self.station_ids:
            raise ValueError("the station id list is empty")
        self.timeout = float(timeout)
        if not np.isfinite(self.timeout) or self.timeout <= 0.0:
            raise ValueError("the fetch timeout must be positive seconds")
        self.refresh = bool(refresh)
        self.records: dict[str, Path] = {}
        self.empty_valid_times: tuple[str, ...] = ()
        self.manifest_path: Path | None = None

    def _selection(self) -> str:
        if self.station_ids:
            return f"stations {','.join(self.station_ids)}"
        return "the box " + ", ".join(f"{v:g}" for v in self.bbox)

    # -- the StationObsSource protocol ----------------------------------

    def observations(self, valid_times):
        """Every report covering ``valid_times``, fetched and merged."""

        contracts = _contracts()
        times = sorted({str(t) for t in valid_times})
        if not times:
            raise ValueError("observations need at least one valid time")
        for text in times:
            contracts.parse_valid_time(text)
        module = _dynamical_module()
        unavailable = getattr(module, "DynamicalUnavailable", None)
        if not (isinstance(unavailable, type)
                and issubclass(unavailable, Exception)):
            unavailable = ()

        stations: dict[str, object] = {}
        reports: dict[tuple[str, str], object] = {}
        members: list[dict[str, object]] = []
        empty: list[str] = []
        fetched: list[str] = []
        records: dict[str, Path] = {}
        self.folder.mkdir(parents=True, exist_ok=True)
        for valid_time in times:
            hour_folder = self.folder / valid_time.replace(":", "")
            try:
                # The reader takes an aware UTC instant, not a seam string.
                instant = contracts.parse_valid_time(valid_time).replace(
                    tzinfo=timezone.utc)
                path = Path(module.fetch_surface(
                    self.bbox, instant, hour_folder,
                    timeout=self.timeout, refresh=self.refresh,
                    station_ids=(list(self.station_ids)
                                 if self.station_ids else None)))
            except unavailable as error:
                raise ObsSourceUnavailable(DYNAMICAL_SOURCE,
                                           str(error)) from error
            except (KeyError, IndexError):
                # A LookupError, but a reader's own bug rather than an hour
                # the archive does not hold: never scored as missing.
                raise
            except LookupError:
                empty.append(valid_time)
                continue
            except (ImportError, OSError) as error:
                # A dependency imported at call time, a socket timeout, a
                # refused connection: the archive is out of reach here.
                raise ObsSourceUnavailable(
                    DYNAMICAL_SOURCE,
                    f"{type(error).__name__}: {error}") from error
            record = _read_asos_record(path, self.SCHEMAS)
            meta = dict(record.get("provenance", {}))
            if str(meta.get("source", "")) != DYNAMICAL_SOURCE:
                raise ValueError(
                    f"{path} names source {meta.get('source')!r}, not "
                    f"{DYNAMICAL_SOURCE!r}; a record from another archive "
                    f"would be scored under this one's name")
            if bool(meta.get("is_stub", False)):
                raise ValueError(
                    f"{path} is a stand-in record; this source serves "
                    f"observations only")
            hour_stations, hour_reports = _asos_record_rows(
                record, [valid_time], contracts)
            for station in hour_stations:
                stations.setdefault(station.station_id, station)
            for report in hour_reports:
                reports.setdefault((report.station_id, report.valid_time),
                                   report)
            stamp = _seam_time_or_none(meta.get("fetched_at"), contracts)
            if stamp is not None:
                fetched.append(stamp)
            # The reader rewrites its hourly record on a refresh, so the
            # record a manifest names is a content-addressed copy that no
            # later fetch touches: an earlier score file keeps verifying.
            digest = sha256_of(path)
            frozen = (self.folder / "records"
                      / f"{valid_time.replace(':', '')}.{digest[:16]}.json")
            if not frozen.is_file():
                frozen.parent.mkdir(parents=True, exist_ok=True)
                staging = frozen.with_name(frozen.name + ".partial")
                shutil.copyfile(path, staging)
                staging.replace(frozen)
            members.append({
                "valid_time": valid_time,
                "path": frozen.relative_to(self.folder).as_posix(),
                "fetched_record": str(path),
                "sha256": digest,
                "stations": len(hour_stations),
                "reports": len(hour_reports),
                "provenance": meta,
            })
            records[valid_time] = frozen
        self.records = records
        self.empty_valid_times = tuple(empty)
        if not members:
            raise LookupError(
                f"the Dynamical.org ASOS archive holds no report that "
                f"survived screening at any of {times} for "
                f"{self._selection()}")

        first = members[0]["provenance"]
        product = str(first.get("product") or DYNAMICAL_PRODUCT)
        attribution = str(first.get("attribution")
                          or getattr(module, "ATTRIBUTION", "")
                          or DYNAMICAL_ATTRIBUTION)
        fetched_at = max(fetched) if fetched else datetime.now(
            timezone.utc).strftime(_TIME_FORMAT)
        manifest = {
            "schema": DYNAMICAL_MANIFEST_SCHEMA,
            "source": DYNAMICAL_SOURCE,
            "product": product,
            "attribution": attribution,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "station_ids": (list(self.station_ids) if self.station_ids
                            else None),
            "valid_times": times,
            "empty_valid_times": list(empty),
            "fetched_at": fetched_at,
            "members": members,
        }
        text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        # Named by its own content: a later call never overwrites the
        # manifest an earlier score file's provenance names.
        manifest_path = self.folder / f"dynamical-asos.{key}.manifest.json"
        if not manifest_path.is_file():
            staging = manifest_path.with_name(manifest_path.name + ".partial")
            staging.write_text(text, encoding="utf-8")
            staging.replace(manifest_path)
        self.manifest_path = manifest_path
        provenance = contracts.ObsProvenance(
            source=DYNAMICAL_SOURCE, product=product,
            uri=str(manifest_path.resolve()),
            sha256=sha256_of(manifest_path), fetched_at=fetched_at,
            attribution=attribution)
        return contracts.StationObsSet(
            stations=tuple(stations.values()),
            reports=tuple(reports.values()), provenance=provenance)

    # -- integrity ------------------------------------------------------

    def verify(self, provenance, *, root=None) -> bool:
        """Re-hash the manifest the provenance names, then every record in it.

        Member records are found relative to the manifest's own folder, so a
        working folder moved whole verifies where it lands (``root`` names
        the folder the manifest now sits in, exactly as for every other
        source).  A missing manifest or member raises; a changed one returns
        False.
        """

        if not _GriddedSource.verify(self, provenance, root=root):
            return False
        uri = getattr(provenance, "uri", None) or dict(provenance)["uri"]
        manifest_path = Path(uri)
        if root is not None and not manifest_path.is_file():
            manifest_path = Path(root) / PureWindowsPath(str(uri)).name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != DYNAMICAL_MANIFEST_SCHEMA:
            return False
        for member in manifest.get("members", ()):
            path = Path(str(member["path"]))
            if not path.is_absolute():
                path = manifest_path.parent / path
            if not path.is_file():
                raise FileNotFoundError(
                    f"cannot re-hash {path}: a record the Dynamical.org ASOS "
                    f"manifest {manifest_path} names is not on disk")
            if sha256_of(path) != str(member["sha256"]).lower():
                return False
        return True


#: The station observation sources a command line can select by name.
#: ``asos`` reads one record the IEM front door wrote (the default
#: everywhere); ``dynamical-asos`` fetches the Dynamical.org ASOS Parquet
#: archive for the requested hours.
STATION_SOURCES: tuple[str, ...] = ("asos", "dynamical-asos")

#: The exit status a command uses when the station source it was asked for
#: raised :class:`ObsSourceUnavailable`.  Not 1 (a refusal of the request),
#: not 2 (argparse's) and not 3 (a Rust door missing or off its pin).
SOURCE_UNAVAILABLE_EXIT = 4


def split_station_ids(text) -> list[str] | None:
    """A comma-separated station id list from a command line, or None."""

    if not text:
        return None
    ids = [part.strip() for part in str(text).split(",") if part.strip()]
    return ids or None


def station_obs_source(name: str, *, record=None, folder=None, bbox=None,
                       station_ids=None, timeout: float = 120.0,
                       refresh: bool = False):
    """A ``StationObsSource`` by name, for the command lines that select one.

    ``asos`` needs ``record`` (a ``gpuwm-obs.asos-surface`` file);
    ``dynamical-asos`` needs ``folder`` and a ``bbox`` or ``station_ids``.
    """

    if name == "asos":
        if record is None:
            raise ValueError("the asos station source needs a record path")
        return AsosSurfaceSource(record)
    if name == "dynamical-asos":
        if folder is None:
            raise ValueError(
                "the dynamical-asos station source needs a working folder")
        return DynamicalAsosSurfaceSource(
            folder, bbox=bbox, station_ids=station_ids, timeout=timeout,
            refresh=refresh)
    raise ValueError(
        f"unknown station source {name!r}; the sources are "
        f"{list(STATION_SOURCES)}")


__all__ = ["AsosSurfaceSource", "DEFAULT_MATCH_SECONDS",
           "DYNAMICAL_ATTRIBUTION", "DYNAMICAL_MANIFEST_SCHEMA",
           "DYNAMICAL_PRODUCT", "DYNAMICAL_SOURCE",
           "DynamicalAsosSurfaceSource", "MrmsCompositeSource",
           "ObsSourceUnavailable", "OperaCompositeSource",
           "QUANTITY_COMPOSITE_REFLECTIVITY",
           "QUANTITY_PRECIPITATION_ACCUMULATION", "SOURCE_UNAVAILABLE_EXIT",
           "STATION_SOURCES", "Stage4PrecipSource", "bbox_of", "parse_bbox",
           "split_station_ids", "station_obs_source"]
