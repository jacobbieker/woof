"""Unstructured-mesh field-file sources for ``woof hex intermediate``: DWD ICON global.

THE GAP THIS CLOSES.  ICON publishes on an icosahedral mesh: WMO
grid-definition template 101, which carries a cell count and a mesh
identity and no coordinates (those travel in the separate CLAT and CLON
records).  ``met_intermediate`` reads uniform lat-lon grids only, so ICON
needs one stage before it: ``gdt101_remap`` (``tools/grib1_bridge``,
contract ``arwen.gdt101-regional-remap.v1``) builds a spherical
four-neighbour plan from the same-cycle CLAT, CLON and FR_LAND records and
writes each native field onto a regular 0.125-degree window.  That window
covers the target hex mesh plus a one-degree halo, and it must be regional:
under 180 degrees of longitude, inside +/-88 degrees of latitude and no more
than two million points.

WHAT THIS MODULE OWNS, and nothing else:

* the product table (:data:`ICON_GLOBAL_PL`, :data:`ICON_GLOBAL_ML`): every
  native record a lead needs, its seven GRIB selector values, the remap
  method its level admits, and its path in DWD's v1 open-data tree
  ``<PARAM>[/lvt1/<type>/lv1/<value>]/r/<YYYY-MM-DDTHH:MM>/s/PT<HHH>H00M.grib2``
  (the tree ``woof fetch --source icon-global`` mirrors under its ``--out``);
* locating those files, all of them, refusing with the missing ones named;
* the window: :func:`choose_remap` reads the target mesh and decides between
  the regional window and a global target.  A global target is refused here
  and names the ``gdt101_remap --global`` road; that one function is the
  seam where the global branch attaches;
* running the remapper: one plan per run, one ``apply`` per lead, and the
  remapped records of each lead concatenated into one regular GRIB2 file
  that ``met_intermediate`` reads like any lat-lon source.

The remapper verifies the grid identity, cell count, centre, product
template, reference time, lead and every selector value of every object
against what this table asks for, so a wrong file under a right name is
refused by the binary, not read.  Interpolation, missing-value handling and
GRIB encoding are the remapper's; this module adds no numbers.
"""

from __future__ import annotations

from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable, Mapping, NoReturn, Sequence

#: The remapper's contract marker and its bridge name.
REMAP_CONTRACT = "arwen.gdt101-regional-remap.v1"
REMAPPER_NAME = "gdt101_remap"
#: The regional window's spacing and halo, and the remapper's envelope.
STEP_DEGREES = 0.125
HALO_DEGREES = 1.0
MAX_ABS_LATITUDE = 88.0
MAX_LONGITUDE_SPAN = 180.0
MAX_POINTS = 2_000_000
#: The global road (another stage owns it); named by the refusal.
GLOBAL_REMAP_COMMAND = "gdt101_remap --global --dx-deg 0.125"
#: Where the remapped GRIB2 and the plan are kept, under --out-dir.
REMAP_DIR = "remap"
REMAP_TIMEOUT_SECONDS = 1800
#: How many missing files a refusal lists by name before it counts the rest.
MISSING_NAMED = 24

Selector = tuple[int, int, int, int, float, int, float]


def _refuse(message: str) -> NoReturn:
    # Deferred: met_intermediate_door imports this module for its rows.
    from .met_intermediate_door import GlobalIntermediateRefusal

    raise GlobalIntermediateRefusal(message)


# ---------------------------------------------------------------------------
# the product table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NativeRecord:
    """One native field object: where it sits in the tree, how it is read."""

    param: str
    #: discipline, category, parameter, first level type, first level
    #: value, second level type, second level value -- what the remapper
    #: checks every object against.
    selector: Selector
    #: The remapper's method; the level type constrains it (100 takes idw4
    #: or nearest, 106 takes soil-nearest).
    method: str
    #: The ``lvt1`` directory, or None for a single-level field.
    level_type: int | None = None
    #: The ``lv1`` directory, spelled as the publisher spells it.
    level: str | None = None

    @property
    def label(self) -> str:
        return self.param if self.level is None else f"{self.param}@{self.level}"

    def relative(self, cycle: datetime, lead: int) -> Path:
        parts = [self.param]
        if self.level_type is not None:
            parts += ["lvt1", str(self.level_type), "lv1", str(self.level)]
        parts += ["r", f"{cycle:%Y-%m-%dT%H:%M}", "s", f"PT{int(lead):03d}H00M.grib2"]
        return Path(*parts)

    def job_tokens(self) -> list[str]:
        d, c, p, t1, v1, t2, v2 = self.selector
        return [str(d), str(c), str(p), str(t1), repr(float(v1)), str(t2), repr(float(v2))]


@dataclass(frozen=True)
class FieldFileProduct:
    """An unstructured-mesh product published one field object per file."""

    name: str
    title: str
    cells: int
    centre: int
    #: Latitude then longitude, in the order the remapper's plan reads them.
    geometry: tuple[NativeRecord, NativeRecord]
    #: The land fraction the plan classifies cells by.
    land_fraction: NativeRecord
    #: Hour-zero records remapped once and handed to every lead as the
    #: engine's ``--invariant`` file.
    invariants: tuple[NativeRecord, ...]
    #: Records every lead carries.
    per_lead: tuple[NativeRecord, ...]
    cycle_hours: tuple[int, ...]
    #: (cycle hours, last lead) pairs.
    horizons: tuple[tuple[tuple[int, ...], int], ...]
    #: Hourly output to this lead, then every ``late_step_hours``.
    hourly_until: int
    late_step_hours: int
    publisher: str
    archive: str
    licence: str
    attribution: str
    vertical: str = "isobaric"

    def horizon(self, cycle: datetime) -> int:
        for hours, last in self.horizons:
            if cycle.hour in hours:
                return last
        _refuse(
            f"{self.name} runs at {', '.join(f'{h:02d}' for h in self.cycle_hours)} UTC; "
            f"--cycle {cycle:%Y-%m-%dT%H} is not one of them"
        )

    def refuse_unpublished(self, cycle: datetime, leads: Sequence[int]) -> None:
        if cycle.minute or cycle.second:
            _refuse(f"--cycle {cycle:%Y-%m-%dT%H:%M} is not on the hour")
        last = self.horizon(cycle)
        beyond = [lead for lead in leads if lead > last]
        if beyond:
            _refuse(
                f"{self.name} publishes the {cycle:%H}Z run to f{last:03d}; --hours asks "
                f"for f{beyond[-1]:03d}"
            )
        off = [lead for lead in leads if lead > self.hourly_until and lead % self.late_step_hours]
        if off:
            _refuse(
                f"{self.name} is hourly to f{self.hourly_until:03d} and every "
                f"{self.late_step_hours} h after it; f{off[0]:03d} is not published"
            )


_PRESSURE_HPA = (30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850,
                 900, 925, 950, 1000)
#: TERRA soil temperature nodes (m), as the v1 tree spells its directories.
_T_SO_NODES = ("0.0", "0.005", "0.02", "0.06", "0.18", "0.54", "1.62", "4.86", "14.58")
#: TERRA soil water layers: top (the directory) and bottom (m).
_W_SO_LAYERS = (("0.0", 0.01), ("0.01", 0.03), ("0.03", 0.09), ("0.09", 0.27),
                ("0.27", 0.81), ("0.81", 2.43), ("2.43", 7.29), ("7.29", 21.87))
ICON_MODEL_LEVELS = 120


def _single(param: str, d: int, c: int, p: int, t1: int, v1: float, method: str,
            t2: int = 255, v2: float = 0.0) -> NativeRecord:
    return NativeRecord(param, (d, c, p, t1, float(v1), t2, float(v2)), method)


_CLAT = _single("CLAT", 0, 191, 1, 1, 0, "nearest")
_CLON = _single("CLON", 0, 191, 2, 1, 0, "nearest")
_FR_LAND = _single("FR_LAND", 2, 0, 0, 1, 0, "nearest")
_HSURF = _single("HSURF", 0, 3, 6, 1, 0, "surface-idw4", t2=101)

_SURFACE = (
    _single("PS", 0, 3, 0, 1, 0, "surface-idw4"),
    _single("PMSL", 0, 3, 1, 101, 0, "idw4"),
    _single("T_2M", 0, 0, 0, 103, 2, "surface-idw4"),
    _single("RELHUM_2M", 0, 1, 1, 103, 2, "surface-idw4"),
    _single("U_10M", 0, 2, 2, 103, 10, "surface-idw4"),
    _single("V_10M", 0, 2, 3, 103, 10, "surface-idw4"),
    _single("T_G", 0, 0, 0, 1, 0, "surface-idw4"),
    _single("W_SNOW", 0, 1, 60, 1, 0, "surface-idw4"),
    _single("FR_ICE", 10, 2, 0, 1, 0, "seaice-nearest"),
)
_SOIL = (
    *(NativeRecord("T_SO", (2, 3, 18, 106, float(node), 255, 0.0), "soil-nearest", 106, node)
      for node in _T_SO_NODES),
    *(NativeRecord("W_SO", (2, 3, 20, 106, float(top), 106, bottom), "soil-nearest", 106, top)
      for top, bottom in _W_SO_LAYERS),
)
_PRESSURE = tuple(
    NativeRecord(param, (0, c, p, 100, float(hpa * 100), 255, 0.0), "idw4", 100, str(hpa * 100))
    for param, c, p in (("T", 0, 0), ("U", 2, 2), ("V", 2, 3), ("FI", 3, 4), ("RELHUM", 1, 1))
    for hpa in _PRESSURE_HPA
)
_MODEL = tuple(
    NativeRecord(param, (0, c, p, 150, float(n), 150, float(n + 1)), "idw4", 150, str(n))
    for param, c, p in (("T", 0, 0), ("U", 2, 2), ("V", 2, 3), ("QV", 1, 0), ("P", 3, 0))
    for n in range(1, ICON_MODEL_LEVELS + 1)
)
_HHL = tuple(
    NativeRecord("HHL", (0, 3, 6, 150, float(n), 101, 0.0), "idw4", 150, str(n))
    for n in range(1, ICON_MODEL_LEVELS + 2)
)

_ICON_COMMON: dict[str, Any] = dict(
    cells=2_949_120,
    centre=78,
    geometry=(_CLAT, _CLON),
    land_fraction=_FR_LAND,
    cycle_hours=(0, 6, 12, 18),
    horizons=(((0, 12), 180), ((6, 18), 120)),
    hourly_until=78,
    late_step_hours=3,
    publisher="https://opendata.dwd.de/weather/nwp/v1/m/icon/p/",
    archive="https://data.source.coop/bkr/icon/icon/p/",
    licence="CC BY 4.0",
    attribution="Source: Deutscher Wetterdienst (DWD), CC BY 4.0",
)

ICON_GLOBAL_PL = FieldFileProduct(
    name="icon-global",
    title="DWD ICON global R03B07 (~13 km), 18 pressure levels 1000-30 hPa",
    invariants=(_HSURF, _FR_LAND),
    per_lead=(*_PRESSURE, *_SURFACE, *_SOIL),
    **_ICON_COMMON,
)
ICON_GLOBAL_ML = FieldFileProduct(
    name="icon-global-ml",
    title="DWD ICON global R03B07 (~13 km), 120 model levels and the 121 HHL half levels",
    invariants=(_HSURF, _FR_LAND, *_HHL),
    per_lead=(*_MODEL, *_SURFACE, *_SOIL),
    vertical="model-level",
    **_ICON_COMMON,
)


# ---------------------------------------------------------------------------
# locating the files
# ---------------------------------------------------------------------------
@dataclass
class LocatedFieldFiles:
    root: Path
    geometry: list[Path]
    land_fraction: Path
    invariants: list[tuple[NativeRecord, Path]]
    leads: dict[int, list[tuple[NativeRecord, Path]]]


def _tree_root(product: FieldFileProduct, grib_dir: Path, cycle: datetime) -> Path:
    """``--grib-dir`` itself, or its ``p/`` child, whichever holds this cycle."""

    probe = product.geometry[0].relative(cycle, 0)
    candidates = [root for root in (grib_dir, grib_dir / "p") if (root / probe).is_file()]
    if len(candidates) > 1:
        _refuse(
            f"{grib_dir} and {grib_dir / 'p'} both hold {probe}; one --grib-dir is one "
            f"tree, so move one aside"
        )
    return candidates[0] if candidates else grib_dir


def locate_field_files(product: FieldFileProduct, grib_dir: Path, cycle: datetime,
                       leads: Sequence[int]) -> LocatedFieldFiles:
    """Every file the run reads, or one refusal naming the missing ones."""

    root = Path(grib_dir).expanduser()
    if not root.is_dir():
        _refuse(f"--grib-dir {root} is not a directory")
    product.refuse_unpublished(cycle, leads)
    root = _tree_root(product, root, cycle)
    # (lead, param) -> [(record, relative path)], in table order.
    missing: dict[tuple[int, str], list[tuple[NativeRecord, Path]]] = {}

    def find(record: NativeRecord, lead: int) -> Path:
        relative = record.relative(cycle, lead)
        path = root / relative
        if not path.is_file():
            group = missing.setdefault((int(lead), record.param), [])
            if all(seen != relative for _, seen in group):
                group.append((record, relative))
        return path

    geometry = [find(record, 0) for record in product.geometry]
    land = find(product.land_fraction, 0)
    invariants = [(record, find(record, 0)) for record in product.invariants]
    by_lead = {int(lead): [(record, find(record, lead)) for record in product.per_lead]
               for lead in leads}
    if missing:
        lines = []
        for (lead, param), group in missing.items():
            record, relative = group[0]
            if len(group) == 1:
                lines.append(f"f{lead:03d} {record.label} ({relative})")
            else:
                levels = [r.level for r, _ in group]
                shown = ", ".join(levels if len(levels) <= 6 else [*levels[:3], "...", *levels[-2:]])
                lines.append(f"f{lead:03d} {param} at {len(group)} levels ({shown}; the first "
                             f"is {relative})")
        count = sum(len(group) for group in missing.values())
        named = "; ".join(lines[:MISSING_NAMED])
        more = f"; and {len(lines) - MISSING_NAMED} more" if len(lines) > MISSING_NAMED else ""
        _refuse(
            f"{root} is missing {count} {product.name} {cycle:%Y-%m-%dT%H}Z "
            f"file(s) in DWD's v1 layout: {named}{more}.  `woof fetch --source "
            f"icon-global --cycle {cycle:%Y-%m-%dT%H} ... --out {Path(grib_dir).expanduser()}` "
            f"writes them (from {product.publisher}, or {product.archive} for an aged-out "
            f"cycle)"
        )
    return LocatedFieldFiles(root=root, geometry=geometry, land_fraction=land,
                             invariants=invariants, leads=by_lead)


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegionalWindow:
    west: float
    south: float
    nx: int
    ny: int
    step: float = STEP_DEGREES

    @property
    def north(self) -> float:
        return self.south + (self.ny - 1) * self.step

    @property
    def east(self) -> float:
        return self.west + (self.nx - 1) * self.step

    def text(self) -> str:
        return (f"{self.west:.9f} {self.south:.9f} {self.step:.9f} {self.step:.9f} "
                f"{self.nx} {self.ny}\n")

    def as_dict(self) -> dict[str, Any]:
        return {"west": self.west, "south": self.south, "east": self.east,
                "north": self.north, "dx": self.step, "dy": self.step,
                "nx": self.nx, "ny": self.ny}


@dataclass(frozen=True)
class RemapChoice:
    """Which remap road a target takes, and the facts that chose it."""

    mode: str
    window: RegionalWindow | None
    basis: Mapping[str, Any]


def read_target_coordinates(path: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Degrees of every cell centre and vertex of an MPAS mesh file."""

    import numpy as np

    path = Path(path).expanduser()
    if not path.is_file():
        _refuse(f"--grid {path} is not a file")
    try:
        from netCDF4 import Dataset
    except ImportError as error:  # pragma: no cover - partial install
        _refuse(f"netCDF4 is not importable ({error}); the --grid mesh cannot be read")
    lats, lons = [], []
    with Dataset(str(path)) as ds:
        ds.set_auto_mask(False)
        if "latCell" not in ds.variables or "lonCell" not in ds.variables:
            _refuse(
                f"--grid {path} carries no latCell/lonCell; pass the hex mesh (the grid, "
                f"static or init file) the intermediate must cover"
            )
        pairs = [("latCell", "lonCell")]
        if "latVertex" in ds.variables and "lonVertex" in ds.variables:
            pairs.append(("latVertex", "lonVertex"))
        for lat_name, lon_name in pairs:
            lats.append(np.asarray(ds.variables[lat_name][:], dtype=np.float64).ravel())
            lons.append(np.asarray(ds.variables[lon_name][:], dtype=np.float64).ravel())
    lat, lon = np.concatenate(lats), np.concatenate(lons)
    if lat.size == 0 or not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
        _refuse(f"--grid {path}: the mesh coordinates are empty or not finite")
    if float(np.max(np.abs(lat))) > math.pi / 2 + 1e-6:
        _refuse(
            f"--grid {path}: latitudes exceed pi/2, so the coordinates are not radians, "
            f"the only unit MPAS writes"
        )
    basis = {"grid": str(path.absolute()), "points": int(lat.size),
             "coordinates": "+".join(name for pair in pairs for name in pair)}
    return np.degrees(lat), np.degrees(lon), basis


def _longitude_cover(lon_deg: Any) -> tuple[float, float, float]:
    """The shortest arc holding every longitude: (west, east, largest gap)."""

    import numpy as np

    ordered = np.unique(np.mod(np.asarray(lon_deg, dtype=np.float64), 360.0))
    if ordered.size == 1:
        return float(ordered[0]), float(ordered[0]), 360.0
    gaps = np.diff(np.append(ordered, ordered[0] + 360.0))
    cut = int(np.argmax(gaps))
    west = float(ordered[(cut + 1) % ordered.size])
    east = west + float(np.max(np.mod(ordered - west, 360.0)))
    return west, east, float(gaps[cut])


def mesh_is_global(lat_deg: Any, lon_deg: Any) -> bool:
    """A mesh that wraps the sphere: no longitude gap wider than 10 degrees and
    latitudes spanning more than 170 degrees."""

    import numpy as np

    lat = np.asarray(lat_deg, dtype=np.float64)
    _, _, gap = _longitude_cover(lon_deg)
    return gap < 10.0 and float(lat.max() - lat.min()) > 170.0


def regional_window(lat_deg: Any, lon_deg: Any, *, step: float = STEP_DEGREES,
                    halo: float = HALO_DEGREES) -> RegionalWindow:
    """The regular window covering every point plus ``halo``, or a refusal
    naming the remapper's regional limit it would break."""

    import numpy as np

    lat = np.asarray(lat_deg, dtype=np.float64)
    west0, east0, _ = _longitude_cover(lon_deg)
    west = math.floor((west0 - halo) / step) * step
    east = math.ceil((east0 + halo) / step) * step
    south = math.floor((float(lat.min()) - halo) / step) * step
    north = math.ceil((float(lat.max()) + halo) / step) * step
    nx = int(round((east - west) / step)) + 1
    # Keep the western anchor in [-180, 180); the east may pass +180.
    west -= math.floor((west + 180.0) / 360.0) * 360.0
    ny = int(round((north - south) / step)) + 1
    span = (nx - 1) * step
    if span >= MAX_LONGITUDE_SPAN:
        _refuse(
            f"the target plus its {halo:g} degree halo spans {span:g} degrees of longitude; "
            f"the regional remap needs under {MAX_LONGITUDE_SPAN:g}.  Split the domain, or "
            f"take the global road ({GLOBAL_REMAP_COMMAND})"
        )
    if south < -MAX_ABS_LATITUDE or north > MAX_ABS_LATITUDE:
        _refuse(
            f"the target plus its {halo:g} degree halo reaches {south:g}..{north:g} degrees "
            f"of latitude; the regional remap stays within +/-{MAX_ABS_LATITUDE:g} (a pole "
            f"needs the global road, {GLOBAL_REMAP_COMMAND})"
        )
    if nx * ny > MAX_POINTS:
        _refuse(
            f"the window is {nx} x {ny} = {nx * ny:,} points at {step:g} degrees; the "
            f"regional remap holds at most {MAX_POINTS:,}.  Split the domain"
        )
    return RegionalWindow(west=float(west), south=float(south), nx=nx, ny=ny, step=step)


def choose_remap(lat_deg: Any, lon_deg: Any, basis: Mapping[str, Any] | None = None) -> RemapChoice:
    """The one place the road is chosen: regional window, or the global road.

    The global road is ``gdt101_remap --global``, whole-sphere output at
    0.125 degrees; this door does not drive it, and says so rather than
    cropping a global mesh into a window it does not fit.
    """

    import numpy as np

    facts = dict(basis or {})
    lat = np.asarray(lat_deg, dtype=np.float64)
    west0, east0, gap = _longitude_cover(lon_deg)
    facts.update(lat_range=[float(lat.min()), float(lat.max())],
                 lon_arc=[west0, east0], largest_longitude_gap=gap)
    if mesh_is_global(lat_deg, lon_deg):
        _refuse(
            f"the target mesh is global (latitudes {lat.min():.1f}..{lat.max():.1f}, no "
            f"longitude gap wider than {gap:.2f} degrees); the regional remap covers under "
            f"180 degrees of longitude inside +/-88.  A global hex init needs the global "
            f"road, `{GLOBAL_REMAP_COMMAND}`, which this door does not drive yet"
        )
    window = regional_window(lat_deg, lon_deg)
    facts["halo_degrees"] = HALO_DEGREES
    return RemapChoice(mode="regional", window=window, basis=facts)


# ---------------------------------------------------------------------------
# the remapper
# ---------------------------------------------------------------------------
def resolve_remapper(explicit: Path | None = None) -> Path:
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if not candidate.is_file():
            _refuse(f"the remapper {candidate} is not a file")
        return candidate.absolute()
    try:
        from woof import bridges
    except ImportError as error:  # pragma: no cover - partial install
        _refuse(f"woof.bridges is not importable ({error}); gdt101_remap cannot be resolved")
    try:
        found = bridges.find_bridge(REMAPPER_NAME)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        _refuse(f"gdt101_remap could not be resolved: {error}")
    if found is None:
        _refuse("gdt101_remap is not built or staged.\n" + bridges.bridge_remedy(REMAPPER_NAME))
    ok, evidence = bridges.bridge_abi_matches(REMAPPER_NAME, Path(found))
    if not ok:
        _refuse(f"{found} does not speak {REMAP_CONTRACT} ({evidence}); rebuild tools/grib1_bridge")
    return Path(found)


def _run(argv: list[str], log_path: Path) -> None:
    started = time.perf_counter()
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=REMAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        log_path.write_text(f"$ {' '.join(argv)}\n[timeout {REMAP_TIMEOUT_SECONDS} s]\n", encoding="utf-8")
        _refuse(f"gdt101_remap exceeded {REMAP_TIMEOUT_SECONDS} s; its log is {log_path}")
    elapsed = time.perf_counter() - started
    log_path.write_text(
        f"$ {' '.join(argv)}\n[{elapsed:.1f} s, exit {done.returncode}]\n"
        f"--- stdout ---\n{done.stdout}\n--- stderr ---\n{done.stderr}\n",
        encoding="utf-8",
    )
    if done.returncode != 0:
        tail = (done.stderr or done.stdout or "").strip().splitlines()
        _refuse(
            f"gdt101_remap exited {done.returncode}; its log is {log_path}"
            + (f".  It said: {tail[-1]}" if tail else "")
        )


def _concatenate(pieces: Sequence[tuple[NativeRecord, Path]], out: Path, log_path: Path) -> None:
    absent = [record.label for record, piece in pieces
              if not piece.is_file() or piece.stat().st_size == 0]
    if absent:
        _refuse(
            f"gdt101_remap exited 0 but wrote no output for {len(absent)} record(s) "
            f"({', '.join(absent[:8])}{', ...' if len(absent) > 8 else ''}); its log is {log_path}"
        )
    with out.open("xb") as sink:
        for _, piece in pieces:
            with piece.open("rb") as source:
                shutil.copyfileobj(source, sink, 4 * 1024 * 1024)


def _apply(remapper: Path, plan: Path, records: Sequence[tuple[NativeRecord, Path]],
           cycle: datetime, lead: int, work: Path, out: Path, tag: str, log_dir: Path) -> Path:
    """Remap one set of records valid at ``lead`` and concatenate them into ``out``."""

    pieces_dir = work / f"pieces.{tag}"
    pieces_dir.mkdir()
    rows, pieces = [], []
    for index, (record, source) in enumerate(records):
        piece = pieces_dir / f"{index:04d}.{record.param}.grib2"
        fields = [str(source.absolute()), str(piece), *record.job_tokens(), record.method,
                  f"{cycle:%Y%m%d%H}", str(int(lead) * 3600)]
        for path in (fields[0], fields[1]):
            if any(c in path for c in "\t\n\r"):
                _refuse(f"{path!r} contains a tab or newline, which a remap job row cannot carry")
        rows.append("\t".join(fields))
        pieces.append((record, piece))
    jobs = work / f"jobs.{tag}.tsv"
    jobs.write_text("\n".join(rows) + "\n", encoding="utf-8")
    log_path = log_dir / f"gdt101_remap.apply.{tag}.log"
    _run([str(remapper), "apply", str(plan), str(jobs)], log_path)
    _concatenate(pieces, out, log_path)
    shutil.rmtree(pieces_dir)
    return out


@dataclass
class RemapResult:
    directory: Path
    choice: RemapChoice
    remapper: Path
    plan: Path
    invariant_grib: Path
    lead_gribs: dict[int, Path]
    located: LocatedFieldFiles

    def receipt(self, describe: Callable[[Path], dict[str, Any]], product: FieldFileProduct) -> dict[str, Any]:
        return {
            "contract": REMAP_CONTRACT,
            "mode": self.choice.mode,
            "window": self.choice.window.as_dict() if self.choice.window else None,
            "window_basis": dict(self.choice.basis),
            "remapper": describe(self.remapper),
            "plan": describe(self.plan),
            "native_grid": {"template": 101, "cells": product.cells,
                            "originating_centre": product.centre},
            "geometry": [describe(p) for p in self.located.geometry],
            "land_fraction": describe(self.located.land_fraction),
            "invariant_grib": describe(self.invariant_grib),
            "invariants": [describe(p) for _, p in self.located.invariants],
            "methods": sorted({f"{r.param}:{r.method}" for r in
                               (*product.invariants, *product.per_lead)}),
            "tree_root": str(self.located.root.absolute()),
        }


def remap_series(product: FieldFileProduct, located: LocatedFieldFiles, choice: RemapChoice,
                 cycle: datetime, directory: Path, *, remapper: Path, workers: int = 4,
                 log: Callable[[str], None] = print, log_dir: Path | None = None) -> RemapResult:
    """Plan once, then remap the invariants and every lead into ``directory``.

    The remapper's logs go to ``log_dir`` (default: beside ``directory``), so
    they outlive a failed stage whose directory the caller removes.
    """

    if choice.mode != "regional" or choice.window is None:
        _refuse(f"remap mode {choice.mode!r} is not one this door drives")
    log_dir = directory.parent if log_dir is None else log_dir
    directory.mkdir(parents=True)
    target = directory / "target.txt"
    target.write_text(choice.window.text(), encoding="ascii")
    plan = directory / "plan.bin"
    lat, lon = located.geometry
    argv = [str(remapper), "plan",
            str(lat.absolute()), ",".join(product.geometry[0].job_tokens()),
            str(lon.absolute()), ",".join(product.geometry[1].job_tokens()),
            str(located.land_fraction.absolute()), ",".join(product.land_fraction.job_tokens()),
            str(target), str(plan), f"{cycle:%Y%m%d%H}", str(product.cells), str(product.centre)]
    window = choice.window
    log(f"{product.name}: plan {window.nx}x{window.ny} at {window.step:g} deg, "
        f"{window.west:g}..{window.east:g} E, {window.south:g}..{window.north:g} N")
    _run(argv, log_dir / "gdt101_remap.plan.log")
    invariant = _apply(remapper, plan, located.invariants, cycle, 0, directory,
                       directory / f"{product.name}.invariant.grib2", "invariant", log_dir)
    leads = sorted(located.leads)
    pool_size = max(1, min(int(workers), len(leads)))
    pool = ThreadPoolExecutor(max_workers=pool_size)
    try:
        futures = [pool.submit(_apply, remapper, plan, located.leads[lead], cycle, lead,
                               directory, directory / f"{product.name}.f{lead:03d}.grib2",
                               f"f{lead:03d}", log_dir) for lead in leads]
        wait(futures, return_when=FIRST_EXCEPTION)
        outs = [future.result() for future in futures]
    except BaseException:
        # The first failure is the answer: leads not yet started are dropped.
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    log(f"{product.name}: remapped {len(located.invariants)} invariant and "
        f"{sum(len(v) for v in located.leads.values())} lead records")
    return RemapResult(directory=directory, choice=choice, remapper=remapper, plan=plan,
                       invariant_grib=invariant, lead_gribs=dict(zip(leads, outs)),
                       located=located)


__all__ = [
    "FieldFileProduct",
    "GLOBAL_REMAP_COMMAND",
    "ICON_GLOBAL_ML",
    "ICON_GLOBAL_PL",
    "LocatedFieldFiles",
    "NativeRecord",
    "REMAP_CONTRACT",
    "RegionalWindow",
    "RemapChoice",
    "RemapResult",
    "choose_remap",
    "locate_field_files",
    "mesh_is_global",
    "read_target_coordinates",
    "regional_window",
    "remap_series",
    "resolve_remapper",
]
