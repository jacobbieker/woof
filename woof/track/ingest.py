"""Reading a 3-D path -- time, lat, lon, height -- from CSV, JSON or Parquet.

Nothing here knows a particular dataset.  :class:`ColumnMapping` says
which columns hold time, latitude, longitude and height, which columns
stand in where the height is missing, how times are written, what datum
the heights are on, and which rows to drop by value; it comes from
arguments or a small JSON/TOML mapping file.

Quality control, in order, each drop counted in ``Track3D.qc``:

* rows with no time, no position or no height, rows a ``drop_when``
  rule names (e.g. a receiver's own no-fix flag), ``0, 0`` fixes,
  coordinates off the globe and heights no balloon or aircraft reaches;
* duplicate times (the first row wins) after a stable sort by time,
  which also repairs rows that arrived out of order;
* gross position outliers far from the median position of their
  nearest-in-time neighbours (a longitude sign or zero glitch lands
  thousands of km away);
* isolated position and height spikes: a fix that only an impossible
  speed connects to its neighbours on BOTH sides.

The API is :func:`load_track`; :mod:`woof.track.cli` is the
``woof track`` command on top of it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from woof.explain import layered
from woof.track.model import (DEFAULT_MIN_CLIMB_M, HEIGHT_DATUMS, Track3D,
                              haversine_km, track_from_fixes)


#: Readers by file suffix.
SUFFIXES = {".csv": "csv", ".tsv": "csv", ".txt": "csv", ".json": "json",
            ".parquet": "parquet", ".pq": "parquet"}
#: Keys a JSON document may wrap its rows in.
JSON_ROW_KEYS = ("records", "rows", "fixes", "points", "data")

#: Heights outside this range (m) are not a balloon's or an aircraft's.
MIN_HEIGHT_M = -500.0
MAX_HEIGHT_M = 45_000.0
#: A fix further than this from the median position of its
#: GROSS_OUTLIER_WINDOW nearest-in-time fixes on each side is a glitch.  A
#: local median follows a track however far it goes, and a sign or zero
#: glitch lands thousands of km away; a run of fewer glitches than the
#: window still loses the vote.
GROSS_OUTLIER_KM = 500.0
GROSS_OUTLIER_WINDOW = 5
#: Spike thresholds.  A fix is a spike only if reaching it from BOTH
#: neighbours needs more than the speed AND more than the floor distance
#: (the floor stops a few metres of jitter between fixes one second apart
#: reading as an impossible speed).  Stratospheric jets stay well under
#: the horizontal limit, and so does a jet airliner; a free fall after a
#: balloon bursts stays under the vertical one.
MAX_HORIZONTAL_SPEED_MS = 400.0
HORIZONTAL_SPIKE_FLOOR_KM = 2.0
MAX_VERTICAL_SPEED_MS = 200.0
VERTICAL_SPIKE_FLOOR_M = 500.0


@dataclass(frozen=True)
class ColumnMapping:
    """Where a file keeps each quantity, and how to read it.

    ``time_format`` is a :func:`~datetime.datetime.strptime` format,
    ``"epoch"`` for Unix seconds, or ``None`` for ISO 8601 (a space or a
    ``T`` between date and time; a trailing ``Z`` or offset honoured).
    Times without a zone are UTC.  ``height_fallback`` columns are tried
    in order where ``height`` is blank.  ``drop_when`` maps a column to
    the values that disqualify a row.
    """

    time: str = "time"
    lat: str = "lat"
    lon: str = "lon"
    height: str = "height"
    height_fallback: tuple[str, ...] = ()
    time_format: str | None = None
    height_datum: str = "msl"
    drop_when: tuple[tuple[str, str], ...] = field(default=())

    def __post_init__(self) -> None:
        if self.height_datum not in HEIGHT_DATUMS:
            raise ValueError(layered(
                f"height datum {self.height_datum!r} is not one of "
                f"{', '.join(HEIGHT_DATUMS)}.\n"
                "  remedy: say msl for heights above sea level (GPS "
                "altitude) or agl for heights above the ground.",
                "The datum is carried into the track document so a "
                "planner never mixes the two; it is not inferred."))

    @property
    def columns(self) -> tuple[str, ...]:
        """The columns a file must have."""
        return (self.time, self.lat, self.lon, self.height)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ColumnMapping":
        """A mapping from a dict with this class's field names; a string
        stands for a one-item ``height_fallback``, and ``drop_when`` may
        be ``{column: value-or-values}`` or ``["column=value", ...]``."""

        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(layered(
                f"the column mapping names {', '.join(unknown)}, which it "
                "does not define.\n"
                f"  remedy: use only {', '.join(sorted(known))}.",
                "An unknown key is most often a misspelt one, and a "
                "misspelt column key would silently fall back to the "
                "default column name."))
        values = dict(data)
        fallback = values.get("height_fallback", ())
        values["height_fallback"] = ((fallback,) if isinstance(fallback, str)
                                     else tuple(fallback))
        values["drop_when"] = parse_drop_when(values.get("drop_when", ()))
        return cls(**values)

    @classmethod
    def from_file(cls, path: str | Path) -> "ColumnMapping":
        """A mapping from a JSON or TOML file holding :meth:`from_dict`'s
        keys."""

        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".toml":
            import tomllib
            return cls.from_dict(tomllib.loads(text))
        return cls.from_dict(json.loads(text))

    def with_overrides(self, **overrides: Any) -> "ColumnMapping":
        """This mapping with every override that is not ``None`` applied."""
        return replace(self, **{key: value for key, value in overrides.items()
                                if value is not None})


def parse_drop_when(spec: Any) -> tuple[tuple[str, str], ...]:
    """``(column, value)`` pairs from ``{column: value(s)}`` or
    ``["column=value", ...]``."""

    pairs: list[tuple[str, str]] = []
    if isinstance(spec, Mapping):
        for column, value in spec.items():
            for item in (value if isinstance(value, (list, tuple)) else [value]):
                pairs.append((str(column), str(item).strip()))
        return tuple(pairs)
    for item in spec:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append((str(item[0]), str(item[1]).strip()))
            continue
        column, sep, value = str(item).partition("=")
        if not sep or not column.strip():
            raise ValueError(f"drop rule {item!r} is not COLUMN=VALUE")
        pairs.append((column.strip(), value.strip()))
    return tuple(pairs)


def load_track(path: str | Path, mapping: ColumnMapping | None = None, *,
               name: str | None = None,
               min_climb_m: float = DEFAULT_MIN_CLIMB_M) -> Track3D:
    """Read, clean and model one track file.

    ``mapping`` defaults to columns named ``time``, ``lat``, ``lon`` and
    ``height`` (MSL, ISO 8601 times).  ``name`` defaults to the file's
    stem.  ``min_climb_m`` is the climb above the starting ground that
    makes a launch.
    """

    path = Path(path)
    mapping = mapping or ColumnMapping()
    name = check_name(path.stem if name is None else name)
    times, lat, lon, height, qc = read_fixes(path, mapping)
    return track_from_fixes(name, times, lat, lon, height,
                            height_datum=mapping.height_datum,
                            min_climb_m=min_climb_m, qc=qc,
                            source_path=str(path))


def check_name(name: str) -> str:
    """``name`` if it can name a file of its own, else a refusal."""

    if name.strip() in ("", ".", "..") or "/" in name or "\\" in name:
        raise ValueError(layered(
            f"track name {name!r} cannot name a file.\n"
            "  remedy: pass --name with no path separators.",
            "A track's JSON document is written as <name>.json in the "
            "--json directory, so a name with a separator would write "
            "somewhere else."))
    return name


def read_fixes(path: Path, mapping: ColumnMapping):
    """``(times, lat, lon, height, qc)``: one file's quality-controlled,
    time-ordered fixes."""

    rows, header = _read_rows(path, mapping)
    wanted = [*mapping.columns, *mapping.height_fallback,
              *(column for column, _ in mapping.drop_when)]
    missing = [column for column in dict.fromkeys(wanted) if column not in header]
    if missing:
        raise ValueError(layered(
            f"{path} has no {', '.join(missing)} column (it has: "
            f"{', '.join(header[:40]) or 'none'}).\n"
            "  remedy: name its columns with --time-col, --lat-col, "
            "--lon-col, --height-col (and --height-fallback-col, "
            "--drop-when), or a --mapping file.",
            "Columns are never guessed: a guessed height column could be "
            "a pressure or a temperature, and the track would still look "
            "plausible."))

    qc = {"rows": len(rows)}
    drop = {}
    for column, value in mapping.drop_when:
        drop.setdefault(column, set()).add(value)
    fixes: list[tuple[datetime, float, float, float, bool]] = []
    for row in rows:
        lat, lon = _number(row.get(mapping.lat)), _number(row.get(mapping.lon))
        if lat is None or lon is None:
            _count(qc, "no_position")
            continue
        moment = _time(row.get(mapping.time), mapping.time_format)
        if moment is None:
            _count(qc, "bad_time")
            continue
        if any(_matches(row.get(column), values) for column, values in drop.items()):
            _count(qc, "dropped_by_rule")
            continue
        if lat == 0.0 and lon == 0.0:
            _count(qc, "null_island")
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            _count(qc, "position_range")
            continue
        height = _number(row.get(mapping.height))
        fallback = False
        for column in mapping.height_fallback:
            if height is not None:
                break
            height, fallback = _number(row.get(column)), True
        if height is None:
            _count(qc, "no_height")
            continue
        if not MIN_HEIGHT_M <= height <= MAX_HEIGHT_M:
            _count(qc, "height_range")
            continue
        fixes.append((moment, lat, lon, height, fallback))

    # Late arrivals: rows timed before one already read.
    latest = None
    for fix in fixes:
        if latest is not None and fix[0] < latest:
            _count(qc, "reordered")
        latest = fix[0] if latest is None else max(latest, fix[0])
    # Stable sort, then the first row at each time wins.
    kept: list[tuple[datetime, float, float, float, bool]] = []
    for fix in sorted(fixes, key=lambda fix: fix[0]):
        if kept and fix[0] == kept[-1][0]:
            _count(qc, "duplicate_time")
            continue
        kept.append(fix)
    if not kept:
        raise ValueError(layered(
            f"{path} has no usable fixes ({len(rows)} rows).\n"
            "  remedy: check the column options name its time, lat, lon "
            "and height columns, and --time-format matches its times.",
            "Each fix needs a parseable time, a latitude and longitude "
            "away from 0, 0, and a height; rows without them are dropped "
            "and counted."))

    times = [fix[0] for fix in kept]
    lat = np.array([fix[1] for fix in kept])
    lon = np.array([fix[2] for fix in kept])
    height = np.array([fix[3] for fix in kept])
    fallback = np.array([fix[4] for fix in kept])
    keep = _gross_outliers(lat, lon)
    _count(qc, "position_outlier", int((~keep).sum()))
    keep = _despike(times, lat, lon, height, keep, qc)
    index = np.nonzero(keep)[0]
    _count(qc, "height_from_fallback", int(fallback[index].sum()))
    qc["fixes"] = int(index.size)
    return ([times[k] for k in index], lat[index], lon[index], height[index],
            {key: value for key, value in qc.items() if value})


def _read_rows(path: Path, mapping: ColumnMapping) -> tuple[list[Mapping[str, Any]], list[str]]:
    """``(rows, header)`` from one file, by its suffix."""

    kind = SUFFIXES.get(path.suffix.lower())
    if kind is None:
        raise ValueError(layered(
            f"{path} is not a file type a track is read from.\n"
            f"  remedy: pass a {', '.join(sorted(SUFFIXES))} file.",
            "CSV (and tab-separated .tsv) is read as text with a header "
            "row; JSON as a list of row objects, an object of equal-length "
            "columns, or rows under one of "
            f"{', '.join(JSON_ROW_KEYS)}; Parquet through pyarrow."))
    if kind == "csv":
        # utf-8-sig strips a byte order mark, which would otherwise glue
        # itself to the first column name.
        with open(path, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(
                handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
            header = [name.strip() for name in (reader.fieldnames or ())]
            reader.fieldnames = header
            return list(reader), header
    if kind == "json":
        return _json_rows(path, json.loads(path.read_text(encoding="utf-8")))
    try:
        import pyarrow.parquet
    except ImportError as error:
        raise ValueError(
            f"reading {path} needs pyarrow, which is not installed: pip "
            "install 'recast-woof[obs]'") from error
    table = pyarrow.parquet.read_table(path)
    return table.to_pylist(), list(table.column_names)


def _json_rows(path: Path, document: Any) -> tuple[list[Mapping[str, Any]], list[str]]:
    if isinstance(document, Mapping):
        for key in JSON_ROW_KEYS:
            if isinstance(document.get(key), list):
                return _json_rows(path, document[key])
        lists = {key: value for key, value in document.items()
                 if isinstance(value, list)}
        lengths = {len(value) for value in lists.values()}
        if lists and len(lengths) == 1:
            count = lengths.pop()
            rows = [{key: value[k] for key, value in lists.items()}
                    for k in range(count)]
            return rows, list(lists)
    elif isinstance(document, list) and all(isinstance(row, Mapping) for row in document):
        header: dict[str, None] = {}
        for row in document:
            header.update(dict.fromkeys(row))
        return document, list(header)
    raise ValueError(layered(
        f"{path} is JSON but holds neither a list of row objects nor an "
        "object of equal-length column lists.\n"
        "  remedy: write the track as [{\"time\": ..., \"lat\": ..., "
        "\"lon\": ..., \"height\": ...}, ...].",
        f"Rows may also sit under one of {', '.join(JSON_ROW_KEYS)}."))


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _matches(value: Any, wanted: set[str]) -> bool:
    """Whether a cell equals one of a drop rule's values: as text, or as
    numbers when both are (a flag stored as 0.0 matches the rule 0)."""

    text = _text(value)
    if text in wanted:
        return True
    number = _number(value)
    return number is not None and any(_number(item) == number for item in wanted)


def _number(value: Any) -> float | None:
    """A finite float from a cell, else ``None`` (blank, ``nan``, text)."""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value if isinstance(value, (int, float)) else str(value).strip())
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _time(value: Any, time_format: str | None) -> datetime | None:
    """A UTC datetime from a cell, else ``None``."""

    if isinstance(value, datetime):
        moment = value
    elif time_format == "epoch":
        seconds = _number(value)
        if seconds is None:
            return None
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    else:
        text = _text(value)
        if not text:
            return None
        try:
            if time_format is None:
                moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
            else:
                moment = datetime.strptime(text, time_format)
        except ValueError:
            return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _count(qc: dict[str, int], reason: str, count: int = 1) -> None:
    if count:
        qc[reason] = qc.get(reason, 0) + count


def _gross_outliers(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Mask of fixes NOT grossly far from their neighbours' median."""

    k = GROSS_OUTLIER_WINDOW
    if lat.size <= 2:
        return np.ones(lat.size, dtype=bool)

    def local_median(values: np.ndarray) -> np.ndarray:
        padded = np.concatenate((np.full(k, np.nan), values, np.full(k, np.nan)))
        window = np.lib.stride_tricks.sliding_window_view(padded, 2 * k + 1).copy()
        window[:, k] = np.nan        # a fix does not vote for itself
        return np.nanmedian(window, axis=1)

    distance = haversine_km(local_median(lat), local_median(lon), lat, lon)
    return distance <= GROSS_OUTLIER_KM


def _despike(times, lat, lon, height, keep: np.ndarray,
             qc: dict[str, int]) -> np.ndarray:
    """Drop isolated position and height spikes, repeatedly, until none
    are left.  Each fix is judged against its nearest kept neighbours on
    both sides; an end fix, which has only one side, against its two
    nearest there, so one bad fix next to a good end cannot take the
    good end with it."""

    seconds = np.array([(t - times[0]).total_seconds() for t in times])
    keep = keep.copy()
    while True:
        index = np.nonzero(keep)[0]
        n = index.size
        if n < 3:
            return keep
        t, h = seconds[index], height[index]
        la, lo = lat[index], lon[index]
        # Two partners per fix: its neighbours on both sides, or, for an
        # end fix, its two nearest on its one side.
        k = np.arange(n)
        partners = (np.where(k == 0, 1, k - 1),
                    np.where(k == 0, 2, np.where(k == n - 1, n - 3, k + 1)))
        h_spike = np.ones(n, dtype=bool)
        v_spike = np.ones(n, dtype=bool)
        for j in partners:
            dt = np.abs(t[j] - t)
            horizontal = haversine_km(la, lo, la[j], lo[j])
            vertical = np.abs(h[j] - h)
            h_spike &= ((horizontal > HORIZONTAL_SPIKE_FLOOR_KM)
                        & (horizontal * 1000.0 > MAX_HORIZONTAL_SPEED_MS * dt))
            v_spike &= ((vertical > VERTICAL_SPIKE_FLOOR_M)
                        & (vertical > MAX_VERTICAL_SPEED_MS * dt))
        if not (h_spike.any() or v_spike.any()):
            return keep
        _count(qc, "position_spike", int(h_spike.sum()))
        _count(qc, "height_spike", int((v_spike & ~h_spike).sum()))
        keep[index[h_spike | v_spike]] = False


def supported(paths: Iterable[Path]) -> list[Path]:
    """Files to read: each file given, and every readable file directly
    inside each directory given, sorted."""

    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(child for child in path.iterdir()
                                if child.is_file()
                                and child.suffix.lower() in SUFFIXES))
        else:
            found.append(path)
    return found
