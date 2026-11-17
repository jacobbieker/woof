"""The observation cost ``J_o`` of a forecast over a time window (EXPERIMENTAL).

P0.1 of ``docs/dev/DIFFERENTIABLE-HYBRID-PLAN.md``.  For a forecast's model
frames ``x_t`` (history frames, WRF route or hex) and an observation set
``y``::

    J_o = 1/2 * sum_i ((H_i(x_{t(i)}) - y_i) / sigma_i) ** 2

where ``t(i)`` is the frame nearest the observation's time (First Guess at
the Appropriate Time, FGAT: H is applied to the model state at the output
time nearest each observation, within half the output interval) and
``sigma_i`` is the observation error, diagonal ``R``, taken from the
observation when the reader supplies one and from the type table
(``woof/data/grad/obs-cost-types.v1.json``) otherwise.

Operators are the ones the DA code and the referees already use:

* surface temperature, dewpoint and 10 m wind: the frame's ``t2``,
  ``q2``/``psfc`` (dewpoint by :func:`woof.obs.column_operators.dewpoint_from_specific_humidity`)
  and earth-relative ``u10``/``v10`` (WRF winds rotated by
  :func:`woof.da.obsop.earth_relative_winds`) in the station's column;
* upper-air temperature and wind (sondes, aircraft): the column's own
  temperature and earth-relative wind, linear in ln p, no extrapolation
  (as :mod:`woof.da.obs_point` and :mod:`woof.hex.sonde_profile_referee`);
* GNSS radio-occultation refractivity at a tangent-point height: pressure
  log-linear and temperature and humidity linear in height, then
  :func:`woof.obs.column_operators.refractivity_n` (the
  :mod:`woof.da.obs_point` operator without its eta interfaces);
* GOES cloud water path: :func:`woof.da.obsop_cwp.cloud_water_path` on the
  WRF route (``qc + qi + qs`` over the model's own column mass), the same
  sum over ``rho * dz`` on a hex mesh.

Each observation goes to the column that owns it, decided by
:func:`woof.hex.sonde_profile_referee.nearest_cells` (the full-mesh nearest
centre, then the boundary mask, then the footprint bound), on WRF mass
points and hex cells alike.  Columns in a regional boundary zone are never
used.

**Frozen quality control.**  A pair failing the background check
(``|H(x) - y| > gross_sigma * sigma``) on the reference trajectory is
dropped; every other evaluation of the same window can be handed the
reference's accepted set (``accept=``), which skips the check, so a
perturbed run is scored on exactly the reference's pairs and ``J`` stays a
smooth function of the control (a background check re-run on each member
would switch observations in and out and show up as noise in every
finite difference and regression).

**Device residency.**  The frame fields may be CuPy arrays; only the
paired columns (one per observation) are gathered to the host for the
vertical interpolation.

Nothing on a forecast path imports this module.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

SCHEMA = "woof.grad.obs-cost/v1"
OBS_SCHEMA = "woof.grad.obs-set/v1"
TYPES_SCHEMA = "woof.grad.obs-cost-types/v1"
DEFAULT_TYPES = Path(__file__).resolve().parents[1] / "data" / "grad" / "obs-cost-types.v1.json"

SURFACE_KINDS = ("surface_temperature", "surface_dewpoint", "surface_wind_u",
                 "surface_wind_v", "surface_wind_speed")
PRESSURE_KINDS = ("upper_temperature", "upper_wind_u", "upper_wind_v")
HEIGHT_KINDS = ("refractivity",)
COLUMN_KINDS = ("cloud_water_path",)
KINDS = SURFACE_KINDS + PRESSURE_KINDS + HEIGHT_KINDS + COLUMN_KINDS

_TEXT = ("platform", "kind", "station")
_FLOAT = ("lat", "lon", "pressure_pa", "height_m", "elevation_m", "value", "error")
OBS_FIELDS = _TEXT + ("time",) + _FLOAT


class ObsCostError(ValueError):
    """Observations, frames or a table that cannot be scored as given."""


def _utc(value) -> datetime:
    if isinstance(value, np.datetime64):
        value = value.astype("datetime64[us]").astype(datetime)
    if not isinstance(value, datetime):
        raise ObsCostError(f"{value!r} is not a time")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _dt64(value) -> np.datetime64:
    return np.datetime64(_utc(value).replace(tzinfo=None), "s")


# --------------------------------------------------------------------------- observations
@dataclass(frozen=True)
class ObsSet:
    """A flat observation table: one row per scalar observation.

    ``pressure_pa`` is set for upper-air kinds, ``height_m`` (above mean sea
    level) for refractivity, ``elevation_m`` for station reports; the rest
    are NaN.  ``error`` is NaN when the reader supplies none (the type table
    then decides).
    """

    platform: np.ndarray
    kind: np.ndarray
    station: np.ndarray
    time: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    pressure_pa: np.ndarray
    height_m: np.ndarray
    elevation_m: np.ndarray
    value: np.ndarray
    error: np.ndarray

    def __post_init__(self):
        n = len(self.value)
        for name in OBS_FIELDS:
            if len(getattr(self, name)) != n:
                raise ObsCostError(f"observation column {name} has {len(getattr(self, name))} rows, value has {n}")
        unknown = sorted(set(self.kind.tolist()) - set(KINDS))
        if unknown:
            raise ObsCostError(f"unknown observation kind(s) {unknown}; known: {list(KINDS)}")

    def __len__(self) -> int:
        return len(self.value)

    @classmethod
    def from_columns(cls, **columns) -> "ObsSet":
        """Build from keyword columns; absent float columns are NaN, absent text ''."""
        unknown = sorted(set(columns) - set(OBS_FIELDS))
        if unknown:
            raise ObsCostError(f"unknown observation column(s) {unknown}")
        n = len(columns["value"])
        out = {}
        for name in _TEXT:
            raw = columns.get(name)
            out[name] = (np.full(n, "", dtype="U1") if raw is None
                         else np.asarray([str(x) for x in np.asarray(raw).tolist()], dtype=str))
        time = columns["time"]
        out["time"] = np.asarray([_dt64(t) for t in time], dtype="datetime64[s]") if len(time) else np.zeros(0, "datetime64[s]")
        for name in _FLOAT:
            raw = columns.get(name)
            out[name] = np.full(n, np.nan) if raw is None else np.asarray(raw, dtype=np.float64).reshape(-1)
        return cls(**out)

    @classmethod
    def empty(cls) -> "ObsSet":
        return cls.from_columns(value=[], time=[])

    def take(self, index) -> "ObsSet":
        index = np.asarray(index)
        return ObsSet(**{name: getattr(self, name)[index] for name in OBS_FIELDS})

    @staticmethod
    def concat(sets: Iterable["ObsSet"]) -> "ObsSet":
        sets = [s for s in sets if len(s)]
        if not sets:
            return ObsSet.empty()
        return ObsSet(**{name: np.concatenate([getattr(s, name) for s in sets]) for name in OBS_FIELDS})

    def digest(self) -> str:
        """SHA-256 over every column's bytes, in field order."""
        h = hashlib.sha256(OBS_SCHEMA.encode())
        for name in OBS_FIELDS:
            column = getattr(self, name)
            if name in _TEXT:
                h.update("\x1f".join(column.tolist()).encode("utf-8"))
            else:
                h.update(np.ascontiguousarray(column).tobytes())
        return h.hexdigest()

    def save(self, path) -> None:
        """An ``.npz`` of plain arrays (no pickled objects)."""
        np.savez(path, schema=np.asarray(OBS_SCHEMA),
                 **{name: getattr(self, name) for name in OBS_FIELDS})

    @classmethod
    def load(cls, path) -> "ObsSet":
        with np.load(path, allow_pickle=False) as data:
            if str(data["schema"]) != OBS_SCHEMA:
                raise ObsCostError(f"{path} is a {data['schema']} file, not {OBS_SCHEMA}")
            return cls(**{name: data[name] for name in OBS_FIELDS})


# --------------------------------------------------------------------------- error table
@dataclass(frozen=True)
class ErrorType:
    platform: str
    kind: str
    error: float | None
    relative: tuple[tuple[float, ...], tuple[float, ...]] | None
    gross_sigma: float
    max_elevation_diff_m: float | None


@dataclass(frozen=True)
class ErrorTable:
    rows: Mapping[tuple[str, str], ErrorType]
    path: str
    sha256: str

    def row(self, platform: str, kind: str) -> ErrorType | None:
        return self.rows.get((platform, kind))

    def sigma(self, obs: ObsSet) -> np.ndarray:
        """Each observation's error: its own when finite and positive, else the table's (NaN: no row)."""
        out = np.where(np.isfinite(obs.error) & (obs.error > 0), obs.error, np.nan)
        for (platform, kind), row in self.rows.items():
            sel = (obs.platform == platform) & (obs.kind == kind) & ~np.isfinite(out)
            if not np.any(sel):
                continue
            if row.relative is not None:
                heights, fractions = row.relative
                out[sel] = np.abs(obs.value[sel]) * np.interp(obs.height_m[sel], heights, fractions)
            else:
                out[sel] = row.error
        return out


def read_error_table(path=None) -> ErrorTable:
    path = Path(path) if path is not None else DEFAULT_TYPES
    raw = path.read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    if doc.get("schema") != TYPES_SCHEMA:
        raise ObsCostError(f"{path} is not a {TYPES_SCHEMA} table")
    rows = {}
    for entry in doc.get("types", []):
        extra = set(entry) - {"platform", "kind", "error", "relative_error", "gross_sigma", "max_elevation_diff_m"}
        if extra:
            raise ObsCostError(f"{path}: row {entry} has unknown field(s) {sorted(extra)}")
        kind = entry["kind"]
        if kind not in KINDS:
            raise ObsCostError(f"{path}: unknown kind {kind!r}")
        relative = entry.get("relative_error")
        error = entry.get("error")
        if (relative is None) == (error is None):
            raise ObsCostError(f"{path}: row {entry['platform']}/{kind} needs exactly one of error and relative_error")
        if error is not None and not (np.isfinite(error) and error > 0):
            raise ObsCostError(f"{path}: row {entry['platform']}/{kind} has error {error!r}; it must be positive")
        if relative is not None:
            heights = tuple(float(x) for x in relative["height_m"])
            fractions = tuple(float(x) for x in relative["fraction"])
            if len(heights) != len(fractions) or not heights or np.any(np.diff(heights) <= 0) or min(fractions) <= 0:
                raise ObsCostError(f"{path}: row {entry['platform']}/{kind} relative_error needs increasing heights "
                                   "and positive fractions of one length")
            relative = (heights, fractions)
        gross = float(entry.get("gross_sigma", np.inf))
        if not gross > 0:
            raise ObsCostError(f"{path}: row {entry['platform']}/{kind} has gross_sigma {gross!r}")
        key = (entry["platform"], kind)
        if key in rows:
            raise ObsCostError(f"{path}: two rows for {key}")
        rows[key] = ErrorType(platform=entry["platform"], kind=kind,
                              error=None if error is None else float(error), relative=relative,
                              gross_sigma=gross, max_elevation_diff_m=entry.get("max_elevation_diff_m"))
    return ErrorTable(rows=rows, path=str(path), sha256=hashlib.sha256(raw).hexdigest())


# --------------------------------------------------------------------------- model frames
@dataclass
class ModelFrame:
    """One model state at one valid time, as columns.

    3-D fields are ``(nz, ncol)``, bottom-up: ``p`` (Pa), ``t`` (K), ``q``
    (specific humidity), ``u``/``v`` (earth-relative, m/s), ``z`` (mass-level
    height above mean sea level, m).  2-D fields are ``(ncol,)``: ``t2``,
    ``q2`` (specific), ``u10``/``v10`` (earth-relative), ``psfc``,
    ``terrain``, and ``cwp`` (g m-2) when the frame carries condensate.
    """

    valid: datetime
    lat: np.ndarray
    lon: np.ndarray
    usable: np.ndarray
    terrain: np.ndarray
    p: object
    t: object
    q: object
    u: object
    v: object
    z: object
    t2: object
    q2: object
    u10: object
    v10: object
    psfc: object
    cwp: object = None
    source: str = ""

    @property
    def ncol(self) -> int:
        return int(np.size(self.lat))


def _host(array):
    get = getattr(array, "get", None)
    return np.asarray(get() if callable(get) else array)


def _record(variable, index):
    data = variable[index] if variable.dimensions and variable.dimensions[0] == "Time" else variable[:]
    return np.ma.filled(np.ma.asarray(data, dtype=np.float64), np.nan)


def _wrf_times(dataset) -> list[datetime]:
    raw = dataset.variables["Times"][:]
    out = []
    for row in np.atleast_2d(raw):
        text = "".join(c.decode() if isinstance(c, bytes) else str(c) for c in np.ma.filled(row, b" ")).strip()
        out.append(datetime.strptime(text, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc))
    return out


#: WRF's ``T`` is potential temperature minus this constant (module_model_constants t0).
WRF_T0 = 300.0


def wrf_frames(path, *, rim: int = 5) -> list[ModelFrame]:
    """Every record of one WRF-style history file (woof's or WRF's ``wrfout``).

    ``rim`` is the boundary zone left unused (``spec_bdy_width``, 5 by default).
    """
    import netCDF4

    from woof.core import constants as c
    from woof.da.obsop import destagger_u, destagger_v, earth_relative_winds

    frames = []
    with netCDF4.Dataset(path) as d:
        v = d.variables
        need = ("XLAT", "XLONG", "HGT", "P", "PB", "T", "QVAPOR", "U", "V", "PH", "PHB",
                "T2", "Q2", "U10", "V10", "PSFC", "SINALPHA", "COSALPHA")
        missing = [name for name in need if name not in v]
        if missing:
            raise ObsCostError(f"{path} lacks {missing}; the cost needs a full-state history frame")
        for index, valid in enumerate(_wrf_times(d)):
            lat, lon = _record(v["XLAT"], index), _record(v["XLONG"], index)
            ny, nx = lat.shape
            sina, cosa = _record(v["SINALPHA"], index), _record(v["COSALPHA"], index)
            p = _record(v["P"], index) + _record(v["PB"], index)
            theta = _record(v["T"], index) + WRF_T0  # WRF's T is theta - 300 K (T0, not the base state's T00)
            qv = _record(v["QVAPOR"], index)
            t = theta * (p / c.P0) ** c.RCP
            u, vv = earth_relative_winds(destagger_u(_record(v["U"], index)),
                                         destagger_v(_record(v["V"], index)), sina, cosa)
            zw = (_record(v["PH"], index) + _record(v["PHB"], index)) / c.G
            z = 0.5 * (zw[:-1] + zw[1:])
            u10, v10 = earth_relative_winds(_record(v["U10"], index), _record(v["V10"], index), sina, cosa)
            q2 = _record(v["Q2"], index)
            usable = np.zeros((ny, nx), bool)
            if ny > 2 * rim and nx > 2 * rim:
                usable[rim:ny - rim, rim:nx - rim] = True
            cwp = None
            if all(name in v for name in ("QCLOUD", "MU", "MUB", "C1H", "C2H", "DNW")):
                from woof.da.obsop_cwp import cloud_water_path
                species = [_record(v[name], index) for name in ("QCLOUD", "QICE", "QSNOW") if name in v]
                cwp = cloud_water_path(species, c1h=np.ravel(_record(v["C1H"], index)),
                                       c2h=np.ravel(_record(v["C2H"], index)),
                                       dnw=np.ravel(_record(v["DNW"], index)),
                                       mu=_record(v["MU"], index) + _record(v["MUB"], index)).reshape(-1)
            nz = p.shape[0]
            frames.append(ModelFrame(
                valid=valid, lat=lat.reshape(-1), lon=lon.reshape(-1), usable=usable.reshape(-1),
                terrain=_record(v["HGT"], index).reshape(-1),
                p=p.reshape(nz, -1), t=t.reshape(nz, -1), q=(qv / (1.0 + qv)).reshape(nz, -1),
                u=u.reshape(nz, -1), v=vv.reshape(nz, -1), z=z.reshape(nz, -1),
                t2=_record(v["T2"], index).reshape(-1), q2=(q2 / (1.0 + q2)).reshape(-1),
                u10=u10.reshape(-1), v10=v10.reshape(-1), psfc=_record(v["PSFC"], index).reshape(-1),
                cwp=cwp, source=f"{path}#{index}"))
    return frames


def _mpas_name(variables, *names):
    for name in names:
        if name in variables:
            return name
    return None


def mpas_frames(path, *, mesh=None) -> list[ModelFrame]:
    """Every record of one MPAS-layout history file (woof hex or native MPAS-A).

    Fields absent from the history (``zgrid``, ``ter``, ``bdyMaskCell``,
    ``latCell``/``lonCell``) are read from ``mesh`` (a grid, static or init
    file of the same mesh).  A regional mesh with no ``bdyMaskCell`` is
    refused, as the referees refuse it.
    """
    import netCDF4

    from woof.core import constants as c
    from woof.hex.sonde_profile_referee import usable_cells

    def geometry(name, source, extra):
        if name in source.variables:
            return np.asarray(source.variables[name][:], dtype=np.float64)
        if extra is not None and name in extra.variables:
            return np.asarray(extra.variables[name][:], dtype=np.float64)
        return None

    frames = []
    extra = netCDF4.Dataset(mesh) if mesh is not None else None
    try:
        with netCDF4.Dataset(path) as d:
            v = d.variables
            lat_r, lon_r = geometry("latCell", d, extra), geometry("lonCell", d, extra)
            zgrid, ter = geometry("zgrid", d, extra), geometry("ter", d, extra)
            bdy = geometry("bdyMaskCell", d, extra)
            if lat_r is None or zgrid is None:
                raise ObsCostError(f"{path} carries no latCell/zgrid; pass mesh= with a grid, static or init file")
            lat, lon = np.degrees(lat_r).reshape(-1), np.degrees(lon_r).reshape(-1)
            if zgrid.ndim == 3:
                zgrid = zgrid[0]
            zw = zgrid.T  # (nz+1, nCells)
            z = 0.5 * (zw[:-1] + zw[1:])
            terrain = zw[0] if ter is None else ter.reshape(-1)
            usable = usable_cells(lat, lon, None if bdy is None else bdy.reshape(-1).astype(np.int64))
            usable = np.ones(lat.size, bool) if usable is None else np.asarray(usable)
            uname = _mpas_name(v, "u_zonal", "uReconstructZonal")
            vname = _mpas_name(v, "v_meridional", "uReconstructMeridional")
            missing = [n for n, k in (("theta", "theta"), ("pressure", "pressure"), ("qv", "qv"),
                                      ("zonal wind", uname), ("meridional wind", vname),
                                      ("t2m", "t2m"), ("q2", "q2"), ("u10", "u10"), ("v10", "v10"))
                       if k is None or k not in v]
            if missing:
                raise ObsCostError(f"{path} lacks {missing}; the cost needs a full-state history frame")
            stamps = v["xtime"][:]
            for index in range(stamps.shape[0]):
                text = "".join(ch.decode() if isinstance(ch, bytes) else str(ch)
                               for ch in np.ma.filled(stamps[index], b" ")).strip()
                valid = datetime.strptime(text, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)

                def field3(name):
                    return np.ma.filled(np.ma.asarray(v[name][index], dtype=np.float64), np.nan).T

                def field2(name):
                    return np.ma.filled(np.ma.asarray(v[name][index], dtype=np.float64), np.nan).reshape(-1)

                p = field3("pressure")
                t = field3("theta") * (p / c.P0) ** c.RCP
                qv = field3("qv")
                psfc = field2("surface_pressure") if "surface_pressure" in v else p[0]
                cwp = None
                if "rho" in v and "qc" in v:
                    cond = sum(field3(name) for name in ("qc", "qi", "qs") if name in v)
                    cwp = 1000.0 * np.sum(cond * field3("rho") * np.diff(zw, axis=0), axis=0)
                q2 = field2("q2")
                frames.append(ModelFrame(
                    valid=valid, lat=lat, lon=lon, usable=usable, terrain=terrain,
                    p=p, t=t, q=qv / (1.0 + qv), u=field3(uname), v=field3(vname), z=z,
                    t2=field2("t2m"), q2=q2 / (1.0 + q2), u10=field2("u10"), v10=field2("v10"),
                    psfc=psfc, cwp=cwp, source=f"{path}#{index}"))
    finally:
        if extra is not None:
            extra.close()
    return frames


def read_frames(paths: Sequence, *, route: str = "auto", mesh=None, rim: int = 5) -> list[ModelFrame]:
    """Frames from history files, sorted by valid time; ``route`` is wrf, mpas or auto."""
    import netCDF4

    frames = []
    for path in paths:
        kind = route
        if kind == "auto":
            with netCDF4.Dataset(path) as d:
                kind = "wrf" if "Times" in d.variables else "mpas"
        frames.extend(wrf_frames(path, rim=rim) if kind == "wrf" else mpas_frames(path, mesh=mesh))
    frames.sort(key=lambda f: f.valid)
    for a, b in zip(frames, frames[1:]):
        if a.valid == b.valid:
            raise ObsCostError(f"two frames at {a.valid.isoformat()} ({a.source}, {b.source})")
    return frames


# --------------------------------------------------------------------------- pairing
@dataclass
class Pairing:
    obs_index: np.ndarray
    frame_index: np.ndarray
    column: np.ndarray
    counts: dict = field(default_factory=dict)


def default_max_offset_s(frames: Sequence[ModelFrame]) -> float:
    if len(frames) < 2:
        return 1800.0
    gaps = np.diff([f.valid.timestamp() for f in frames])
    return 0.5 * float(np.min(gaps))


def pair(obs: ObsSet, frames: Sequence[ModelFrame], table: ErrorTable, *, max_offset_s: float | None = None,
         start=None, end=None) -> Pairing:
    """Assign each observation a frame (FGAT, nearest valid time) and an owning column."""
    from collections import Counter

    from woof.hex.sonde_profile_referee import nearest_cells, unit_vectors

    if not frames:
        raise ObsCostError("no model frames to score")
    counts = Counter(input=len(obs))
    geo = frames[0]
    for frame in frames[1:]:
        if frame.ncol != geo.ncol:
            raise ObsCostError(f"{frame.source} has {frame.ncol} columns and {geo.source} {geo.ncol}")
    if max_offset_s is None:
        max_offset_s = default_max_offset_s(frames)
    keep = np.ones(len(obs), bool)
    obs_t = obs.time.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    if start is not None:
        keep &= obs_t >= _utc(start).timestamp()
    if end is not None:
        keep &= obs_t <= _utc(end).timestamp()
    counts["outside_window"] = int(np.sum(~keep))
    frame_t = np.array([f.valid.timestamp() for f in frames])
    offset = np.abs(obs_t[:, None] - frame_t[None, :]) if len(obs) else np.zeros((0, len(frames)))
    frame_index = np.argmin(offset, axis=1) if len(obs) else np.zeros(0, np.int64)
    near = offset[np.arange(len(obs)), frame_index] <= max_offset_s if len(obs) else np.zeros(0, bool)
    counts["no_frame_within_offset"] = int(np.sum(keep & ~near))
    keep &= near
    typed = np.array([table.row(p, k) is not None for p, k in zip(obs.platform, obs.kind)], bool)
    counts["no_error_type"] = int(np.sum(keep & ~typed))
    keep &= typed
    column = np.full(len(obs), -1, np.int64)
    if np.any(keep):
        sel = np.nonzero(keep)[0]
        column[sel] = nearest_cells(unit_vectors(geo.lat, geo.lon), obs.lat[sel], obs.lon[sel], geo.usable)
    counts["outside_usable_columns"] = int(np.sum(keep & (column < 0)))
    keep &= column >= 0
    surface = np.isin(obs.kind, SURFACE_KINDS)
    limit = np.array([(table.row(p, k).max_elevation_diff_m if table.row(p, k) is not None else None) or np.inf
                      for p, k in zip(obs.platform, obs.kind)], float) if len(obs) else np.zeros(0)
    checked = keep & surface & np.isfinite(limit)
    diff = np.full(len(obs), np.inf)
    if np.any(checked):
        diff[checked] = np.abs(obs.elevation_m[checked] - geo.terrain[column[checked]])
    bad = checked & ~(diff <= limit)  # a NaN station elevation fails the check
    counts["surface_elevation_mismatch"] = int(np.sum(bad))
    keep &= ~bad
    index = np.nonzero(keep)[0]
    return Pairing(obs_index=index, frame_index=frame_index[index], column=column[index], counts=dict(counts))


# --------------------------------------------------------------------------- operators
def interp_increasing(coord, values, target):
    """Row-wise linear interpolation: ``coord`` (m, nz) increasing, NaN outside [coord[0], coord[-1]]."""
    coord = np.asarray(coord, np.float64)
    values = np.asarray(values, np.float64)
    target = np.asarray(target, np.float64)
    m, n = coord.shape
    if m == 0:
        return np.zeros(0)
    below = np.sum(coord <= target[:, None], axis=1)
    k = np.clip(below - 1, 0, n - 2)
    rows = np.arange(m)
    x0, x1 = coord[rows, k], coord[rows, k + 1]
    v0, v1 = values[rows, k], values[rows, k + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (target - x0) / (x1 - x0)
        out = v0 + w * (v1 - v0)
    ok = (target >= coord[:, 0]) & (target <= coord[:, -1]) & (x1 > x0) & np.isfinite(out)
    return np.where(ok, out, np.nan)


def _gather(field, column):
    """Columns of a (nz, ncol) or (ncol,) field, as host float64; the gather runs where the field lives."""
    if field is None:
        return None
    try:
        import cupy as cp
        xp = cp.get_array_module(field)
    except ImportError:  # no CUDA stack: the field is a NumPy array
        xp = np
    index = xp.asarray(column)
    picked = field[:, index].T if field.ndim == 2 else field[index]
    return _host(picked).astype(np.float64)


def simulate(obs: ObsSet, frames: Sequence[ModelFrame], pairing: Pairing) -> np.ndarray:
    """``H(x)`` for every paired observation (NaN where the operator has no value)."""
    from woof.obs.column_operators import dewpoint_from_specific_humidity, refractivity_n

    h = np.full(len(pairing.obs_index), np.nan)
    for fi in np.unique(pairing.frame_index):
        frame = frames[int(fi)]
        rows = np.nonzero(pairing.frame_index == fi)[0]
        oi, col = pairing.obs_index[rows], pairing.column[rows]
        kind = obs.kind[oi]
        for name in SURFACE_KINDS:
            sel = kind == name
            if not np.any(sel):
                continue
            c = col[sel]
            if name == "surface_temperature":
                h[rows[sel]] = _gather(frame.t2, c)
            elif name == "surface_dewpoint":
                h[rows[sel]] = dewpoint_from_specific_humidity(_gather(frame.q2, c), _gather(frame.psfc, c))
            elif name == "surface_wind_u":
                h[rows[sel]] = _gather(frame.u10, c)
            elif name == "surface_wind_v":
                h[rows[sel]] = _gather(frame.v10, c)
            else:
                h[rows[sel]] = np.hypot(_gather(frame.u10, c), _gather(frame.v10, c))
        upper = np.isin(kind, PRESSURE_KINDS)
        if np.any(upper):
            c = col[upper]
            lnp = -np.log(_gather(frame.p, c))
            target = -np.log(obs.pressure_pa[oi[upper]])
            for name, source in (("upper_temperature", frame.t), ("upper_wind_u", frame.u),
                                 ("upper_wind_v", frame.v)):
                sel = kind[upper] == name
                if np.any(sel):
                    h[rows[upper][sel]] = interp_increasing(lnp[sel], _gather(source, c[sel]), target[sel])
        ro = kind == "refractivity"
        if np.any(ro):
            c = col[ro]
            z = _gather(frame.z, c)
            height = obs.height_m[oi[ro]]
            p = np.exp(interp_increasing(z, np.log(_gather(frame.p, c)), height))
            t = interp_increasing(z, _gather(frame.t, c), height)
            q = interp_increasing(z, _gather(frame.q, c), height)
            with np.errstate(invalid="ignore"):
                h[rows[ro]] = refractivity_n(p, t, q)
        cw = kind == "cloud_water_path"
        if np.any(cw) and frame.cwp is not None:
            h[rows[cw]] = _gather(frame.cwp, col[cw])
    return h


# --------------------------------------------------------------------------- cost
@dataclass
class CostResult:
    J: float
    obs_index: np.ndarray
    frame_index: np.ndarray
    column: np.ndarray
    h: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    platform: np.ndarray
    kind: np.ndarray
    counts: dict
    frames: list
    obs_digest: str
    table_sha256: str

    @property
    def departure(self) -> np.ndarray:
        return self.h - self.y

    @property
    def n(self) -> int:
        return int(len(self.obs_index))

    def by_type(self) -> dict:
        out = {}
        for platform, kind in sorted(set(zip(self.platform.tolist(), self.kind.tolist()))):
            sel = (self.platform == platform) & (self.kind == kind)
            d = self.departure[sel]
            out[f"{platform}/{kind}"] = {
                "n": int(sel.sum()), "J": float(0.5 * np.sum((d / self.sigma[sel]) ** 2)),
                "bias": float(np.mean(d)), "rmse": float(np.sqrt(np.mean(d * d))),
                "mean_sigma": float(np.mean(self.sigma[sel]))}
        return out

    def summary(self) -> dict:
        return {"schema": SCHEMA, "J": float(self.J), "n": self.n,
                "J_per_obs": float(self.J / self.n) if self.n else None,
                "by_type": self.by_type(), "counts": dict(self.counts), "frames": list(self.frames),
                "obs_sha256": self.obs_digest, "error_table_sha256": self.table_sha256}

    def save_pairs(self, path) -> None:
        np.savez(path, schema=np.asarray(SCHEMA), obs_sha256=np.asarray(self.obs_digest),
                 obs_index=self.obs_index, frame_index=self.frame_index, column=self.column,
                 h=self.h, y=self.y, sigma=self.sigma, platform=self.platform, kind=self.kind)


def accepted_from(path) -> tuple[np.ndarray, str]:
    """The accepted observation indices and the observation-set digest of a saved reference."""
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"]) != SCHEMA:
            raise ObsCostError(f"{path} is not a {SCHEMA} pairs file")
        return np.asarray(data["obs_index"]), str(data["obs_sha256"])


def evaluate(obs: ObsSet, frames: Sequence[ModelFrame], *, table: ErrorTable | None = None,
             max_offset_s: float | None = None, start=None, end=None,
             accept: tuple[np.ndarray, str] | None = None) -> CostResult:
    """``J_o`` of ``frames`` against ``obs``.

    ``accept`` (``(indices, obs_sha256)`` from a reference's pairs file)
    freezes the observation set: the background check is skipped and only
    those observations are scored.  An accepted observation the operator
    cannot simulate on these frames is counted as ``lost_from_reference``
    and left out of ``J`` (it would otherwise be a silent change of the
    cost's definition between two runs).
    """
    table = table or read_error_table()
    pairing = pair(obs, frames, table, max_offset_s=max_offset_s, start=start, end=end)
    counts = dict(pairing.counts)
    h = simulate(obs, frames, pairing)
    oi = pairing.obs_index
    sigma = table.sigma(obs)[oi]
    y = obs.value[oi]
    valid = np.isfinite(h) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    if accept is not None:
        indices, digest = accept
        if digest != obs.digest():
            raise ObsCostError("the reference pairs were made against another observation set "
                               f"({digest[:12]} != {obs.digest()[:12]}); score both against the same file")
        wanted = np.zeros(len(obs), bool)
        wanted[np.asarray(indices, np.int64)] = True
        mask = valid & wanted[oi]
        counts["lost_from_reference"] = int(np.sum(wanted) - np.sum(mask))
        counts["not_in_reference"] = int(np.sum(valid & ~wanted[oi]))
    else:
        counts["operator_invalid"] = int(np.sum(~valid))
        gross = np.array([table.row(p, k).gross_sigma for p, k in zip(obs.platform[oi], obs.kind[oi])], float) \
            if len(oi) else np.zeros(0)
        with np.errstate(invalid="ignore"):
            failed = valid & (np.abs(h - y) > gross * sigma)
        counts["gross_check"] = int(np.sum(failed))
        mask = valid & ~failed
    counts["accepted"] = int(np.sum(mask))
    d = (h - y)[mask]
    J = 0.5 * float(np.sum((d / sigma[mask]) ** 2))
    return CostResult(J=J, obs_index=oi[mask], frame_index=pairing.frame_index[mask], column=pairing.column[mask],
                      h=h[mask], y=y[mask], sigma=sigma[mask], platform=obs.platform[oi][mask],
                      kind=obs.kind[oi][mask], counts=counts,
                      frames=[f"{f.valid.strftime('%Y-%m-%dT%H:%M:%SZ')} {f.source}" for f in frames],
                      obs_digest=obs.digest(), table_sha256=table.sha256)
