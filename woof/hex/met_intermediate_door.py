"""Global sources onto WPS intermediates: ``woof hex intermediate --source gfs|gdas|ecmwf-open-data|era5|aifs|icon-global|icon-global-ml``.

THE GAP THIS CLOSES.  ``components/hex/docs/source-matrix.md`` drove GFS,
GDAS, ECMWF IFS open data, ERA5 and AIFS through the whole hex chain to a
green forecast smoke, every one of them through the Rust ``met_intermediate``
writer in ``tools/grib1_bridge`` -- and there was no door: each intermediate
was minted by hand.  These products are already on a uniform cylindrical
grid, so unlike HRRR (:mod:`woof.hex.hrrr_intermediate`) they need no
regridding; what they need is the right Vtable, the right producing-centre
label, the right files per valid time and a check that what came out is a
file the init and boundary engines will accept.

WHAT THIS DOOR OWNS, and nothing else:

* the source table (:data:`GLOBAL_SOURCES`): where ``woof fetch`` writes
  each source's GRIB per lead, which packaged Vtable and which
  ``--map-source`` label decode it, and which fields ride an hour-zero
  file only (AIFS's land mask and surface geopotential);
* the time series: ``--cycle`` plus ``--hours START-END`` and
  ``--interval-hours`` give one ``PREFIX:YYYY-MM-DD_HH`` file per valid
  time, the series ``woof hex lbc`` consumes;
* validation: each file is read back through this tree's reader
  (:mod:`woof.hex.wps_intermediate`) and the init door's own met check
  (:func:`woof.hex.init_door.scan_met_file`), and the engine's receipt must
  say every matched message was valid at the time the file is named for;
* provenance: one receipt naming the source, the Vtable and engine digests,
  and every input file with its sha256.

Decoding, unit conversion and the ``rrpr`` repairs are the engine's.  The
door adds no numbers.

DWD ICON global (``icon-global`` on pressure levels, ``icon-global-ml`` on
model levels) is the one row that is not already lat-lon: it publishes one
file per field and level on an unstructured (GDT-101) mesh.  Its rows use the
``field-files`` layout: :mod:`woof.hex.gdt101_door` collects every file a
lead needs, remaps them through ``gdt101_remap`` onto a regular window over
the target mesh (``--grid``), and hands the engine one regular GRIB2 file per
lead plus the hour-zero invariants.  From there the series is checked
exactly like every other row.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

from .errors import MpasPortError
from .gdt101_door import ICON_GLOBAL_ML, ICON_GLOBAL_PL, FieldFileProduct

GLOBAL_SCHEMA = "gpuwm-hex.global-intermediate/v1"
#: The receipt the engine prints on stdout; the door requires this one,
#: because older builds neither drop time-processed messages nor report
#: the valid times they matched.
ENGINE_RECEIPT_SCHEMA = "gpuwm.rw-wps.met-intermediate/v3"
ENGINE_NAME = "met_intermediate"
ENGINE_ENV = "WOOF_MET_INTERMEDIATE"
#: Column tops above this pressure (Pa) sit below the 30 km hex model top
#: (about 1200 Pa), where ``--extrap-airtemp lapse-rate`` is the
#: engine's own fatal (source-matrix.md, IFS row).
LAPSE_RATE_TOP_PA = 1000.0
#: Fields whose level participates in the first-guess level table (the
#: init door's set, plus the HGT spelling ungrib writes and the engine
#: renames to GHT on read).
PROFILE_FIELDS = frozenset({"TT", "RH", "SPECHUMD", "GHT", "HGT", "PRES", "PRESSURE", "UU", "VV"})
SURFACE_LEVEL = 200100.0
MSL_LEVEL = 201300.0


class GlobalIntermediateRefusal(MpasPortError):
    """The global-source intermediate cannot be written, and the message says why."""


# ---------------------------------------------------------------------------
# the source table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GlobalSource:
    """One uniform lat-lon source and how ``met_intermediate`` reads it."""

    name: str
    description: str
    #: The engine's closed ``--map-source`` vocabulary (KNOWN_MAP_SOURCES).
    map_source: str
    #: A file under ``woof/data/vtables``.
    vtable: str
    #: File names per lead, alternatives tried in order, formatted with
    #: ``cycle`` (datetime) and ``hour`` (int).  A ``combined`` source has
    #: one file holding every valid time.
    file_patterns: tuple[str, ...]
    #: The interval between published leads the fetch writes by default.
    default_interval_hours: int
    #: "per-lead": one file per forecast lead; "combined": one file holds
    #: the series and each valid time is selected out of it;
    #: "field-files": one file per field and level on an unstructured mesh,
    #: remapped onto a window over the target mesh first (``product``).
    layout: str = "per-lead"
    #: Leads the source publishes, inclusive; None when unbounded here.
    max_hour: int | None = None
    #: Fields that may come from the hour-zero file when a lead lacks them.
    invariant_fields: tuple[str, ...] = ()
    #: The init switch the published moisture implies.
    use_spechumd: str = "no"
    #: A short note on what the product does not carry.
    notes: str = ""
    #: The field table and native mesh of a "field-files" row.
    product: FieldFileProduct | None = None
    #: The attribution the publisher's licence requires, copied into the receipt.
    attribution: str = ""

    def lead_files(self, root: Path, cycle: datetime, hour: int) -> list[Path]:
        return [root / pattern.format(cycle=cycle, hour=hour) for pattern in self.file_patterns]


_IFS_PATTERN = "{cycle:%Y%m%d%H}0000-{hour}h-oper-fc.grib2"

GLOBAL_SOURCES: Mapping[str, GlobalSource] = {
    "gfs": GlobalSource(
        name="gfs",
        description="NCEP GFS pgrb2.0p25 (0.25-degree global GRIB2, or the "
                    "NOMADS area subset `woof fetch --area` writes)",
        map_source="ncep-gfs",
        vtable="Vtable.GFS.rw",
        file_patterns=(
            "gfs.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}",
            "gfs.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}.subset.grib2",
        ),
        default_interval_hours=3,
        max_hour=384,
    ),
    "gdas": GlobalSource(
        name="gdas",
        description="NCEP GDAS pgrb2.0p25 (record-for-record the GFS catalogue, "
                    "hourly f000..f009)",
        map_source="ncep-gfs",
        vtable="Vtable.GFS.rw",
        file_patterns=(
            "gdas.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}",
            "gdas.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}.subset.grib2",
        ),
        default_interval_hours=1,
        max_hour=9,
    ),
    "ecmwf-open-data": GlobalSource(
        name="ecmwf-open-data",
        description="ECMWF IFS oper open data (0.25-degree global GRIB2, CCSDS "
                    "packing, 14 pressure levels to 50 hPa, ordinal soil)",
        map_source="ecmwf",
        vtable="Vtable.ECMWF-OD.rw",
        file_patterns=(_IFS_PATTERN,),
        default_interval_hours=3,
        max_hour=360,
        invariant_fields=("LANDSEA", "SOILGEO"),
        notes="no sea-ice cover is published (ice thickness only), so SEAICE "
              "is not carried",
    ),
    "aifs": GlobalSource(
        name="aifs",
        description="ECMWF AIFS single open data (0.25-degree global GRIB2, an "
                    "AI emulator: specific humidity only aloft, two soil layers)",
        map_source="ecmwf",
        vtable="Vtable.ECMWF-OD.rw",
        file_patterns=(_IFS_PATTERN,),
        default_interval_hours=6,
        max_hour=360,
        invariant_fields=("LANDSEA", "SOILGEO"),
        use_spechumd="yes",
        notes="the land mask and surface geopotential ride the 0-hour file "
              "alone; a bare-ground/no-snow start is a property of the product",
    ),
    "era5": GlobalSource(
        name="era5",
        description="ERA5 reanalysis, the CDS combined GRIB1 retrieval "
                    "`woof fetch --source era5` writes (every analysis time in "
                    "one file)",
        map_source="ecmwf",
        vtable="Vtable.ERA5.rw",
        file_patterns=("era5-combined.grib",),
        default_interval_hours=6,
        layout="combined",
    ),
    "icon-global": GlobalSource(
        name="icon-global",
        description="DWD ICON global R03B07 (~13 km icosahedral, GDT-101), 18 "
                    "pressure levels 1000-30 hPa, from DWD's v1 open-data tree "
                    "(one file per field and level), remapped to a 0.125-degree "
                    "window over the target mesh",
        map_source="dwd",
        vtable="Vtable.ICON-GLOBAL.rw",
        file_patterns=(),
        default_interval_hours=3,
        layout="field-files",
        max_hour=180,
        invariant_fields=("LANDSEA", "SOILHGT"),
        product=ICON_GLOBAL_PL,
        attribution=ICON_GLOBAL_PL.attribution,
        notes="limited-area meshes only (a global mesh needs gdt101_remap "
              "--global); the land mask and terrain ride the 0-hour HSURF and "
              "FR_LAND objects",
    ),
    "icon-global-ml": GlobalSource(
        name="icon-global-ml",
        description="DWD ICON global R03B07 (~13 km icosahedral, GDT-101), 120 "
                    "model levels with the 121 hour-zero HHL half levels, from "
                    "DWD's v1 open-data tree, remapped to a 0.125-degree window "
                    "over the target mesh",
        map_source="dwd",
        vtable="Vtable.ICON-GLOBAL-ML.rw",
        file_patterns=(),
        default_interval_hours=3,
        layout="field-files",
        max_hour=180,
        invariant_fields=("LANDSEA", "SOILHGT", "GHT"),
        use_spechumd="yes",
        product=ICON_GLOBAL_ML,
        attribution=ICON_GLOBAL_ML.attribution,
        notes="limited-area meshes only (a global mesh needs gdt101_remap "
              "--global); specific humidity aloft (QV, no model-level RH); "
              "heights come from the 0-hour HHL",
    ),
}

#: Sources with a registry row that this door refuses, by name, with the
#: reason the source matrix measured.  Anything else is refused with the
#: list of what is admitted.
REFUSED_SOURCES: Mapping[str, str] = {
    "gefs": "GEFS needs the pgrb2a+pgrb2b pair of one member and the "
            "ncep-gefs label; it is not one of the sources this door admits",
    "aigfs": "AIGFS publishes no land mask, no soil and no skin temperature; "
             "the init door refuses its intermediate (no LANDSEA)",
    "aigefs": "AIGEFS publishes no land mask, no soil and no skin temperature; "
              "the init door refuses its intermediate (no LANDSEA)",
    "rap": "a projected (Lambert) product: met_intermediate refuses a "
           "non-uniform grid; only --source hrrr has a regridding door",
    "rrfs": "a projected (Lambert) product: met_intermediate refuses a "
            "non-uniform grid; only --source hrrr has a regridding door",
    "nam": "a projected product with no field mapping yet",
    "hrrr-prs": "the pressure-level HRRR product; --source hrrr regrids the "
                "native wrfnat pair instead",
    "icon-eu": "met_intermediate has no 'dwd' map source and no slot to convert "
               "ICON's layer-mass soil moisture (kg m-2) to a fraction",
    "gem-gdps": "met_intermediate has no 'eccc-gdps' map source",
    "20crv3": "met_intermediate has no 'ncep-20crv3' map source, and the "
              "native grid is Gaussian",
    "era5-l137": "hybrid model levels are not a Vtable operation; use the "
                 "mapped route",
}


def global_source(name: str) -> GlobalSource:
    key = str(name).strip().lower()
    if key in GLOBAL_SOURCES:
        return GLOBAL_SOURCES[key]
    refuse_unknown_source(key)
    raise AssertionError("unreachable")  # pragma: no cover


def refuse_unknown_source(name: str) -> None:
    key = str(name).strip().lower()
    admitted = ", ".join(["hrrr", *GLOBAL_SOURCES, *ICON_SOURCES])
    if key in REFUSED_SOURCES:
        raise GlobalIntermediateRefusal(
            f"--source {name!r} is refused by this door: {REFUSED_SOURCES[key]}.  "
            f"Admitted sources: {admitted}"
        )
    raise GlobalIntermediateRefusal(
        f"--source {name!r} is not a source `woof hex intermediate` knows.  "
        f"Admitted sources: {admitted}.  A uniform lat-lon GRIB source is a "
        f"row in woof.hex.met_intermediate_door.GLOBAL_SOURCES (a Vtable and a "
        f"met_intermediate --map-source label); a projected one needs a "
        f"regridding door like hrrr's"
    )


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
def packaged_vtable(name: str) -> Path:
    path = Path(__file__).resolve().parents[1] / "data" / "vtables" / name
    if not path.is_file():
        raise GlobalIntermediateRefusal(
            f"the packaged Vtable {name} is missing from {path.parent}; this "
            f"install is incomplete (reinstall woof) or predates the source's "
            f"Vtable (pass --vtable FILE)"
        )
    return path


def resolve_engine(explicit: Path | None = None) -> Path:
    """The ``met_intermediate`` executable, or a refusal naming what supplies it."""

    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if not candidate.is_file():
            raise GlobalIntermediateRefusal(f"--decoder {candidate} is not a file")
        return candidate.absolute()
    try:
        from woof import bridges
    except ImportError as error:  # pragma: no cover - partial install
        raise GlobalIntermediateRefusal(
            f"woof.bridges is not importable ({error}); pass --decoder "
            f"pointing at a met_intermediate build"
        ) from error
    try:
        found = bridges.find_bridge(ENGINE_NAME)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise GlobalIntermediateRefusal(
            f"met_intermediate could not be resolved: {error}"
        ) from error
    if found is None:
        raise GlobalIntermediateRefusal(
            "met_intermediate is not built or staged.\n"
            + bridges.bridge_remedy(ENGINE_NAME)
            + "\n  or pass --decoder FILE"
        )
    ok, evidence = bridges.bridge_abi_matches(ENGINE_NAME, Path(found))
    if not ok:
        raise GlobalIntermediateRefusal(
            f"{found} does not speak this door's met_intermediate contract "
            f"({evidence}); rebuild tools/grib1_bridge"
        )
    return Path(found)


def refuse_engine_without_map_source(engine: Path, label: str) -> None:
    """Refuse an engine whose bytes do not carry ``label`` at all.

    Static and one-sided, like the contract-marker check: the engine's map
    sources are string literals in its table, so a build without the label
    cannot carry its bytes.  A build that does carry them may still refuse,
    and then says so itself.
    """

    try:
        carried = label.encode() in engine.read_bytes()
    except OSError as error:
        raise GlobalIntermediateRefusal(f"{engine} cannot be read ({error})") from error
    if not carried:
        raise GlobalIntermediateRefusal(
            f"{engine} carries no {label!r} map source (met_intermediate's "
            f"KNOWN_MAP_SOURCES); this build predates the {label} rules, so it would "
            f"refuse every lead after the remap.  Rebuild tools/grib1_bridge with them, "
            f"or pass --decoder pointing at one that has them"
        )


def _described(path: Path, cache: dict[Path, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Path, size and sha256; ``cache`` hashes a file shared by many leads once."""

    from .wps_intermediate import sha256_file

    if cache is not None and path in cache:
        return dict(cache[path])
    described = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if cache is not None:
        cache[path] = described
    return dict(described)


def refuse_occupied_out_dir(out_dir: Path) -> None:
    """An output directory already holding WPS intermediates is refused.

    ``woof hex lbc --met-dir`` admits every file with a WPS header in the
    directory, whatever its name, so a file left by an earlier run (another
    cycle, another cadence) would join this series without anyone asking.
    """

    from .init_door import _probe_wps_header

    if not out_dir.is_dir():
        return
    held = sorted(child.name for child in out_dir.iterdir()
                  if child.is_file() and _probe_wps_header(child))
    if held:
        raise GlobalIntermediateRefusal(
            f"--out-dir {out_dir} already holds WPS intermediates ({', '.join(held[:4])}"
            f"{', ...' if len(held) > 4 else ''}); `woof hex lbc --met-dir` reads every one "
            f"of them, so an earlier run's files would join this series.  Pass a fresh "
            f"--out-dir"
        )


# ---------------------------------------------------------------------------
# the time series
# ---------------------------------------------------------------------------
def parse_lead_range(text: str | None, interval_hours: int) -> tuple[int, ...]:
    """``START-END`` (or one lead) at ``interval_hours``: every lead, ascending."""

    if interval_hours is None or int(interval_hours) <= 0:
        raise GlobalIntermediateRefusal(
            f"--interval-hours {interval_hours!r} is not a positive number of hours"
        )
    step = int(interval_hours)
    spec = "0" if text is None else str(text).strip()
    try:
        if "-" in spec:
            first_text, last_text = spec.split("-", 1)
            first, last = int(first_text), int(last_text)
        else:
            first = last = int(spec)
    except ValueError as error:
        raise GlobalIntermediateRefusal(
            f"--hours {text!r} is not START-END or one lead in whole hours"
        ) from error
    if first < 0 or last < first:
        raise GlobalIntermediateRefusal(
            f"--hours {text!r} runs backwards or starts before the cycle"
        )
    if (last - first) % step:
        raise GlobalIntermediateRefusal(
            f"--hours {text!r} is not a whole number of {step} h intervals; a "
            f"boundary series must end on a published time, or the last "
            f"interval would freeze the boundary"
        )
    return tuple(range(first, last + 1, step))


# ---------------------------------------------------------------------------
# the request and its plan
# ---------------------------------------------------------------------------
@dataclass
class GlobalRequest:
    source: GlobalSource
    grib_dir: Path
    cycle: datetime
    leads: tuple[int, ...]
    interval_hours: int
    out_dir: Path
    prefix: str = "MET"
    engine: Path | None = None
    vtable: Path | None = None
    workers: int = 4
    #: "field-files" rows: the hex mesh the remap window covers.
    window_grid: Path | None = None
    #: "field-files" rows: a gdt101_remap overriding the bridge ladder.
    remapper: Path | None = None


@dataclass
class LeadPlan:
    lead: int
    valid_time: datetime
    inputs: list[Path]
    invariants: list[Path]
    out: Path
    argv: list[str] = field(default_factory=list)
    #: "field-files" rows: the native objects the remapped input came from.
    native: list[Path] = field(default_factory=list)

    @property
    def hdate(self) -> str:
        return self.valid_time.strftime("%Y-%m-%d_%H:%M:%S")


def _existing(candidates: Sequence[Path]) -> list[Path]:
    return [path for path in candidates if path.is_file()]


def locate_inputs(request: GlobalRequest) -> dict[int, tuple[list[Path], list[Path]]]:
    """Every lead's GRIB inputs and invariant files, or one refusal naming all gaps."""

    source = request.source
    root = Path(request.grib_dir).expanduser()
    if not root.is_dir():
        raise GlobalIntermediateRefusal(f"--grib-dir {root} is not a directory")
    if source.max_hour is not None and request.leads[-1] > source.max_hour:
        raise GlobalIntermediateRefusal(
            f"{source.name} publishes leads to f{source.max_hour:03d}; --hours "
            f"asks for f{request.leads[-1]:03d}"
        )
    missing: list[str] = []
    ambiguous: list[str] = []
    found: dict[int, tuple[list[Path], list[Path]]] = {}
    if source.layout == "field-files":
        located = _locate_field_files(request)
        shared = list(dict.fromkeys([*located.geometry, located.land_fraction,
                                     *(path for _, path in located.invariants)]))
        return {lead: ([path for _, path in located.leads[lead]], shared)
                for lead in request.leads}
    if source.layout == "combined":
        hits = _existing(source.lead_files(root, request.cycle, 0))
        if not hits:
            netcdf = root / "era5-combined.nc"
            extra = (
                f"; {netcdf.name} is the ARCO NetCDF container, and the "
                f"WPS-intermediate chain is GRIB-only (fetch with the CDS "
                f"provider, which writes era5-combined.grib)"
                if netcdf.is_file() else ""
            )
            raise GlobalIntermediateRefusal(
                f"{root} holds no {' or '.join(source.file_patterns)} for "
                f"{source.name}{extra}.  `woof fetch --source {source.name} ... "
                f"--out {root}` writes it"
            )
        for lead in request.leads:
            found[lead] = ([hits[0]], [])
        return found
    for lead in request.leads:
        hits = _existing(source.lead_files(root, request.cycle, lead))
        if not hits:
            missing.append(f"f{lead:03d} ({' or '.join(p.name for p in source.lead_files(root, request.cycle, lead))})")
            continue
        if len(hits) > 1:
            ambiguous.append(f"f{lead:03d} ({', '.join(p.name for p in hits)})")
            continue
        invariants: list[Path] = []
        if source.invariant_fields and lead != 0:
            zero = _existing(source.lead_files(root, request.cycle, 0))
            if len(zero) != 1:
                missing.append(
                    f"f000 ({source.file_patterns[0].format(cycle=request.cycle, hour=0)}), "
                    f"which supplies {', '.join(source.invariant_fields)} when a lead lacks them"
                )
                continue
            invariants = zero
        found[lead] = (hits, invariants)
    if ambiguous:
        raise GlobalIntermediateRefusal(
            f"{root} holds more than one candidate file for "
            f"{'; '.join(ambiguous)}; a whole object and an area subset of the "
            f"same lead cannot both be the input, so move one aside"
        )
    if missing:
        raise GlobalIntermediateRefusal(
            f"{root} is missing {source.name} {request.cycle:%Y-%m-%dT%H}Z "
            f"{'; '.join(dict.fromkeys(missing))}.  `woof fetch --source "
            f"{source.name} --cycle {request.cycle:%Y-%m-%dT%H} ... --out {root}` "
            f"writes them"
        )
    return found


def _locate_field_files(request: GlobalRequest):
    from .gdt101_door import locate_field_files

    if request.source.product is None:
        raise GlobalIntermediateRefusal(
            f"{request.source.name} declares the field-files layout without a product table"
        )
    return locate_field_files(request.source.product, Path(request.grib_dir),
                              request.cycle, request.leads)


def stage_field_files(request: GlobalRequest, out_dir: Path, *,
                      log: Callable[[str], None] = print):
    """Locate, choose the window, and remap a field-files row into ``out_dir/remap``.

    Returns the :class:`woof.hex.gdt101_door.RemapResult`; the caller
    removes the remap directory if the series later fails.
    """

    from .gdt101_door import (
        REMAP_DIR,
        choose_remap,
        read_target_coordinates,
        remap_series,
        resolve_remapper,
    )

    source = request.source
    if request.window_grid is None:
        raise GlobalIntermediateRefusal(
            f"--source {source.name} is remapped from its native mesh onto a window over "
            f"the target hex mesh; pass --grid GRID.nc (the grid, static or init file "
            f"of the mesh the intermediates must cover)"
        )
    located = _locate_field_files(request)
    lat, lon, basis = read_target_coordinates(Path(request.window_grid))
    choice = choose_remap(lat, lon, basis)
    remapper = resolve_remapper(request.remapper)
    refuse_occupied_out_dir(out_dir)
    remap_dir = out_dir / REMAP_DIR
    if remap_dir.exists():
        raise GlobalIntermediateRefusal(
            f"--out-dir {out_dir} already holds {REMAP_DIR}/ from an earlier run; pass a "
            f"fresh --out-dir"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        return remap_series(source.product, located, choice, request.cycle, remap_dir,
                            remapper=remapper, workers=request.workers, log=log)
    except BaseException:
        shutil.rmtree(remap_dir, ignore_errors=True)
        raise


def engine_argv(
    engine: Path, vtable: Path, source: GlobalSource, plan: LeadPlan, partial: Path
) -> list[str]:
    argv = [
        str(engine), "--vtable", str(vtable), "--date", plan.hdate,
        "--map-source", source.map_source, "--out", str(partial),
    ]
    if source.layout == "combined":
        argv.append("--select-valid-time")
    if plan.invariants:
        for path in plan.invariants:
            argv += ["--invariant", str(path)]
        for name in source.invariant_fields:
            argv += ["--invariant-field", name]
    argv += [str(path) for path in plan.inputs]
    return argv


def plan_leads(request: GlobalRequest, engine: Path, vtable: Path,
               staged: Any = None) -> list[LeadPlan]:
    """One engine run per lead.  A field-files row plans over its remapped
    GRIB (``staged``, from :func:`stage_field_files`), not the native files."""

    out_dir = Path(request.out_dir).expanduser().absolute()
    native: dict[int, list[Path]] = {}
    if request.source.layout == "field-files":
        if staged is None:
            raise GlobalIntermediateRefusal(
                f"--source {request.source.name} is remapped before the engine runs; "
                f"build_global_intermediates stages it"
            )
        found = {lead: ([staged.lead_gribs[lead]], [staged.invariant_grib])
                 for lead in request.leads}
        native = {lead: [path for _, path in staged.located.leads[lead]]
                  for lead in request.leads}
    else:
        found = locate_inputs(request)
    plans: list[LeadPlan] = []
    for lead in request.leads:
        inputs, invariants = found[lead]
        valid = request.cycle + timedelta(hours=int(lead))
        out = out_dir / f"{request.prefix}:{valid:%Y-%m-%d_%H}"
        plan = LeadPlan(
            lead=lead, valid_time=valid,
            inputs=[p.absolute() for p in inputs],
            invariants=[p.absolute() for p in invariants], out=out,
            native=[p.absolute() for p in native.get(lead, [])],
        )
        plan.argv = engine_argv(engine, vtable, request.source, plan, _partial(out))
        plans.append(plan)
    return plans


def _partial(out: Path) -> Path:
    return out.with_name(out.name + ".partial")


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def parse_engine_receipt(stdout: str) -> dict[str, Any]:
    try:
        receipt = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise GlobalIntermediateRefusal(
            f"met_intermediate exited 0 but its stdout is not its JSON receipt ({error})"
        ) from error
    if receipt.get("schema") != ENGINE_RECEIPT_SCHEMA:
        raise GlobalIntermediateRefusal(
            f"met_intermediate printed receipt schema {receipt.get('schema')!r}, "
            f"not {ENGINE_RECEIPT_SCHEMA!r}: the build predates the time-identity "
            f"contract (it cannot drop a 2 m maximum temperature that shares "
            f"the 2 m temperature's key, nor report what time it matched); "
            f"rebuild tools/grib1_bridge"
        )
    return receipt


def validate_intermediate(
    path: Path, *, hdate: str, engine_receipt: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Read a written intermediate back and apply the init door's met check.

    Returns the facts the receipt and the init/boundary switches need.
    Refuses when the file is not one ``rw_mpas_init`` would accept for
    ``hdate``.
    """

    from .init_door import InitDoorRefusal, scan_met_file
    from .wps_intermediate import inventory

    if engine_receipt is not None:
        matched = list(engine_receipt.get("matched_valid_times") or [])
        if matched != [hdate]:
            raise GlobalIntermediateRefusal(
                f"{path.name}: the engine matched messages valid at "
                f"{matched or 'no time'}, not exactly {hdate}; an intermediate "
                f"named for one time must carry that time only (a file holding "
                f"several leads needs --select-valid-time)"
            )
    seen = inventory(path)
    records = seen["records"]
    if engine_receipt is not None and int(engine_receipt.get("records_written", -1)) != int(seen["field_records"]):
        raise GlobalIntermediateRefusal(
            f"{path.name}: the engine reports {engine_receipt.get('records_written')} "
            f"records written and the reader finds {seen['field_records']}"
        )
    stamps = sorted({str(r["valid_time"])[:19] for r in records})
    if stamps != [hdate]:
        raise GlobalIntermediateRefusal(
            f"{path.name}: record headers carry valid times {stamps}, not {hdate}"
        )
    grids = sorted({
        (int(r["nx"]), int(r["ny"]), str(r["projection"]["name"]),
         round(float(r["projection"]["start_latitude"]), 6),
         round(float(r["projection"]["start_longitude"]), 6),
         round(float(r["projection"]["delta_latitude"]), 6),
         round(float(r["projection"]["delta_longitude"]), 6))
        for r in records
    })
    if len(grids) != 1 or grids[0][2] != "latlon":
        raise GlobalIntermediateRefusal(
            f"{path.name}: records are not on one regular lat-lon grid ({grids})"
        )
    profile_levels = sorted({float(r["level"]) for r in records if r["field"] in PROFILE_FIELDS})
    # A 3-D pressure field selects the engines' model-level branch: its
    # levels are model-level ordinals, not pressures, and the column top is
    # the lowest pressure the file carries.
    model_level = any(r["field"] in ("PRES", "PRESSURE")
                      and float(r["level"]) not in (SURFACE_LEVEL, MSL_LEVEL) for r in records)
    isobaric = [] if model_level else [
        lvl for lvl in profile_levels if lvl not in (SURFACE_LEVEL, MSL_LEVEL)]
    soil_t = sorted({str(r["field"]) for r in records if str(r["field"]).startswith(("ST", "SOILT"))})
    soil_m = sorted({str(r["field"]) for r in records if str(r["field"]).startswith(("SM", "SOILM"))})
    try:
        scan_met_file(path, nfglevels=len(profile_levels), start_time=hdate)
    except InitDoorRefusal as error:
        raise GlobalIntermediateRefusal(f"the init door would refuse {path.name}: {error}") from error
    if len(soil_t) != len(soil_m):
        raise GlobalIntermediateRefusal(
            f"{path.name}: {len(soil_t)} soil temperature layers ({soil_t}) and "
            f"{len(soil_m)} soil moisture layers ({soil_m}); the init's soil "
            f"column is one count"
        )
    nx, ny, _, start_lat, start_lon, dlat, dlon = grids[0]
    return {
        "path": str(path), "bytes": int(seen["bytes"]), "sha256": str(seen["sha256"]),
        "valid_time": hdate, "field_records": int(seen["field_records"]),
        "fields": dict(seen["fields"]),
        "grid": {"nx": nx, "ny": ny, "projection": "latlon",
                 "start_lat": start_lat, "start_lon": start_lon,
                 "dlat": dlat, "dlon": dlon},
        "profile_levels": len(profile_levels),
        "vertical": "model-level" if model_level else "isobaric",
        "isobaric_levels": len(isobaric),
        "top_pressure_pa": (_lowest_model_pressure(path) if model_level
                            else (min(isobaric) if isobaric else None)),
        "soil_layers": len(soil_t),
        "soil_fields": soil_t + soil_m,
        "has_rh": any(r["field"] == "RH" and float(r["level"]) not in (SURFACE_LEVEL, MSL_LEVEL) for r in records),
        "has_spechumd": any(r["field"] == "SPECHUMD" and float(r["level"]) not in (SURFACE_LEVEL, MSL_LEVEL) for r in records),
    }


def _lowest_model_pressure(path: Path) -> float:
    import numpy as np

    from .wps_intermediate import WpsIntermediateReader

    lowest: float | None = None
    with WpsIntermediateReader(path) as reader:
        for item in reader.iter_fields(include={"PRES", "PRESSURE"}):
            if float(item.level) in (SURFACE_LEVEL, MSL_LEVEL) or item.values is None:
                continue
            values = np.asarray(item.values, dtype=np.float64)
            if not np.all(np.isfinite(values)) or float(values.min()) <= 0.0:
                raise GlobalIntermediateRefusal(
                    f"{path.name}: model level {item.level:g} of {item.field} holds a "
                    f"missing or non-positive pressure"
                )
            low = float(values.min())
            lowest = low if lowest is None else min(lowest, low)
    if lowest is None:
        raise GlobalIntermediateRefusal(f"{path.name}: no model-level pressure could be read")
    return lowest


def init_switches(source: GlobalSource, files: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The first-guess switches the written series implies, with the reason for each."""

    levels = max(int(f["profile_levels"]) for f in files)
    soil = {int(f["soil_layers"]) for f in files}
    if len(soil) != 1:
        raise GlobalIntermediateRefusal(
            f"the series' files carry different soil column depths {sorted(soil)}; "
            f"--nfgsoillevels is one number"
        )
    tops = [f["top_pressure_pa"] for f in files if f["top_pressure_pa"] is not None]
    top = max(tops) if tops else None
    spechumd = source.use_spechumd
    if spechumd == "no" and not all(f["has_rh"] for f in files):
        spechumd = "yes"
    if spechumd == "yes" and not all(f["has_spechumd"] for f in files):
        raise GlobalIntermediateRefusal(
            "the series carries neither pressure-level RH in every file nor "
            "SPECHUMD in every file; the init has no moisture source"
        )
    if top is None:
        raise GlobalIntermediateRefusal(
            "no file in the series carries a pressure-level profile; the init has "
            "no first-guess column"
        )
    extrap = "constant" if top > LAPSE_RATE_TOP_PA else "lapse-rate"
    return {
        "--nfglevels": levels,
        "--nfgsoillevels": soil.pop(),
        "--use-spechumd": spechumd,
        "--extrap-airtemp": extrap,
        "column_top_pa": top,
        "why_extrap": (
            f"the source column tops at {top:g} Pa, below the ~1200 Pa (30 km) "
            f"model top; lapse-rate extrapolation above the first-guess top is "
            f"the engine's own fatal (source-matrix.md, IFS row)"
            if extrap == "constant" else
            f"the source column tops at {top:g} Pa, above the model top"
        ),
    }


# ---------------------------------------------------------------------------
# the door
# ---------------------------------------------------------------------------
def _run_one(plan: LeadPlan, log_dir: Path) -> tuple[LeadPlan, subprocess.CompletedProcess, float]:
    started = time.perf_counter()
    completed = subprocess.run(plan.argv, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    (log_dir / f"met_intermediate.f{plan.lead:03d}.log").write_text(
        f"$ {' '.join(plan.argv)}\n[{elapsed:.1f} s, exit {completed.returncode}]\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}\n",
        encoding="utf-8",
    )
    return plan, completed, elapsed


def build_global_intermediates(
    request: GlobalRequest, *, log: Callable[[str], None] = print
) -> dict[str, Any]:
    source = request.source
    engine = resolve_engine(request.engine)
    if request.vtable is not None:
        vtable = Path(request.vtable).expanduser().absolute()
        if not vtable.is_file():
            raise GlobalIntermediateRefusal(f"--vtable {vtable} is not a file")
    else:
        vtable = packaged_vtable(source.vtable)
    out_dir = Path(request.out_dir).expanduser().absolute()
    if source.layout != "field-files":
        plans = plan_leads(request, engine, vtable)
        refuse_occupied_out_dir(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return _convert_series(request, engine, vtable, out_dir, plans, None, log=log)
    # Refuse before the remap, which reads every native object: an engine
    # that cannot take this row's label would only refuse after it.
    refuse_engine_without_map_source(engine, source.map_source)
    staged = stage_field_files(request, out_dir, log=log)
    try:
        plans = plan_leads(request, engine, vtable, staged)
        return _convert_series(request, engine, vtable, out_dir, plans, staged, log=log)
    except BaseException:
        # A series is published whole or not at all, its remap included.
        shutil.rmtree(staged.directory, ignore_errors=True)
        raise


def _remap_receipt(source: GlobalSource, staged: Any,
                   digests: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    product = source.product
    window = staged.choice.window
    return {
        "regridding": (
            f"gdt101_remap {staged.choice.mode}: the native GDT-101 mesh "
            f"({product.cells:,} cells) remapped onto a regular {window.step:g}-degree "
            f"window over the target mesh plus its halo, from a four-neighbour plan "
            f"built on the same-cycle CLAT, CLON and FR_LAND; atmosphere by inverse "
            f"squared chord distance, surface fields from same-class donors, soil "
            f"from the nearest land column, sea ice nearest on water"
        ),
        "remap": staged.receipt(lambda p: _described(p, digests), product),
        "publisher": product.publisher,
        "archive": product.archive,
        "licence": product.licence,
        "attribution": source.attribution,
    }


def _convert_series(
    request: GlobalRequest, engine: Path, vtable: Path, out_dir: Path,
    plans: list[LeadPlan], staged: Any, *, log: Callable[[str], None] = print,
) -> dict[str, Any]:
    source = request.source
    workers = max(1, min(int(request.workers), len(plans)))
    log(f"{source.name}: {len(plans)} valid time(s) "
        f"{plans[0].hdate}..{plans[-1].hdate} through {engine.name}, {workers} worker(s)")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda p: _run_one(p, out_dir), plans))
    failures = [(plan, done) for plan, done, _ in results if done.returncode != 0]

    def discard_partials() -> None:
        for each in plans:
            _partial(each.out).unlink(missing_ok=True)

    if failures:
        discard_partials()
        plan, done = failures[0]
        tail = (done.stderr or done.stdout or "").strip().splitlines()
        # The engine's own "met_intermediate: ..." line names the fault; a
        # multi-line refusal explains it below that line.
        said = next((line for line in tail if line.startswith(f"{ENGINE_NAME}:")),
                    tail[-1] if tail else "")
        raise GlobalIntermediateRefusal(
            f"met_intermediate exited {done.returncode} for f{plan.lead:03d} "
            f"({len(failures)} of {len(plans)} valid times failed); its log is "
            f"{out_dir / f'met_intermediate.f{plan.lead:03d}.log'}"
            + (f".  It said: {said}" if said else "")
        )
    # Validate every file before publishing any: a series is published
    # whole or not at all, so no half-series is left for `woof hex lbc`.
    files: list[dict[str, Any]] = []
    digests: dict[Path, dict[str, Any]] = {}
    try:
        checked_all = []
        for plan, done, elapsed in results:
            receipt = parse_engine_receipt(done.stdout)
            checked_all.append((plan, receipt, elapsed, validate_intermediate(
                _partial(plan.out), hdate=plan.hdate, engine_receipt=receipt)))
        grids = {json.dumps(item[3]["grid"], sort_keys=True) for item in checked_all}
        if len(grids) != 1:
            raise GlobalIntermediateRefusal(
                f"the series' files are on {len(grids)} different grids; one boundary "
                f"series is one source grid"
            )
        init_switches(source, [item[3] for item in checked_all])
    except Exception:
        discard_partials()
        raise
    # A field-files row reads hundreds of native objects per lead: digest
    # them on the worker pool rather than one after another.
    natives = list(dict.fromkeys(path for plan in plans for path in plan.native))
    if staged is not None:
        natives += [path for path in (*staged.located.geometry,
                                      *(p for _, p in staged.located.invariants))
                    if path not in digests]
    if natives:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for path, row in zip(natives, pool.map(_described, natives)):
                digests[path] = row
    for plan, receipt, elapsed, checked in checked_all:
        os.replace(_partial(plan.out), plan.out)
        checked["path"] = str(plan.out)
        checked["lead"] = plan.lead
        checked["engine_seconds"] = round(elapsed, 2)
        checked["inputs"] = [_described(p, digests) for p in plan.inputs]
        checked["invariants"] = [_described(p, digests) for p in plan.invariants]
        if plan.native:
            checked["native_inputs"] = [_described(p, digests) for p in plan.native]
        checked["engine_receipt"] = {
            key: receipt.get(key) for key in (
                "schema", "map_source", "grib_messages_read", "grib_messages_matched",
                "records_written", "valid_time_selection", "messages_other_valid_time",
                "messages_time_processed_skipped", "matched_valid_times",
                "isobaric_levels_dropped_above_pmin", "bitmap_masked_points",
                "invariants", "rules", "fields_held",
            )
        }
        files.append(checked)
        log(f"WROTE {plan.out.name}: {checked['field_records']} records, "
            f"{checked['profile_levels']} profile levels, {checked['soil_layers']} soil "
            f"layers, {checked['bytes'] / 1e6:.1f} MB, {checked['engine_seconds']} s")
    switches = init_switches(source, files)
    receipt = {
        "schema": GLOBAL_SCHEMA,
        "source": source.name,
        "source_description": source.description,
        "source_notes": source.notes,
        "regridding": "none: the source grid is uniform lat-lon and is written as is",
        "cycle": f"{request.cycle:%Y-%m-%d_%H:%M:%S}",
        "leads": list(request.leads),
        "interval_hours": int(request.interval_hours),
        "valid_times": [plan.hdate for plan in plans],
        "grib_dir": str(Path(request.grib_dir).expanduser().absolute()),
        "map_source": source.map_source,
        "vtable": _described(vtable, digests),
        "engine": _described(engine, digests),
        "engine_receipt_schema": ENGINE_RECEIPT_SCHEMA,
        "files": files,
        "init_switches": switches,
    }
    if staged is not None:
        receipt.update(_remap_receipt(source, staged, digests))
    elif source.attribution:
        receipt["attribution"] = source.attribution
    receipt_path = out_dir / "intermediate-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    log(f"RECEIPT {receipt_path}")
    return receipt


# ---------------------------------------------------------------------------
# ICON (native GDT-101): regional or global target
# ---------------------------------------------------------------------------
# MERGE SEAM.  ICON is not written as is: its native unstructured mesh is
# regridded first, and WHERE TO depends on the MPAS mesh the intermediates
# are for.  `icon_target_kind` is the one function that decides:
#
# * a global mesh (no regional boundary zone, cells tiling the sphere)
#   -> the full-sphere 0.125-degree regrid, `woof.hex.icon_global_door`;
# * a regional mesh, or a --from-plan/--point box -> the regional ICON door,
#   `run_icon_regional_intermediate` below, which runs the ``field-files``
#   rows of GLOBAL_SOURCES through `woof.hex.gdt101_door`.

#: The ICON source ids this door admits, and what each carries.
ICON_SOURCES: Mapping[str, str] = {
    "icon-global": "DWD ICON global, pressure levels (native GDT 3.101, regridded)",
    "icon-global-ml": "DWD ICON global, model levels (native GDT 3.101, regridded)",
}


def icon_target_kind(arguments: argparse.Namespace) -> tuple[str, dict[str, Any] | None]:
    """``("global", mesh_facts)`` or ``("regional", mesh_facts_or_None)``, or a refusal.

    ``--grid MESH.nc`` names the mesh; it is classified by content (a
    nonempty bdyMask zone is regional, a sphere-tiling cell set is global).
    Without it a --from-plan/--point box is a regional target, and nothing
    at all is refused: the door does not guess which mesh it is writing for.
    """

    from .icon_global_door import classify_mesh

    source = str(arguments.source).strip().lower()
    boxed = [flag for flag, attribute in (("--from-plan", "from_plan"), ("--point", "point"))
             if getattr(arguments, attribute, None) is not None]
    grid = getattr(arguments, "grid", None)
    if grid is not None:
        mesh = classify_mesh(Path(grid))
        if mesh["kind"] == "global" and boxed:
            raise GlobalIntermediateRefusal(
                f"--grid {grid} is a global mesh, which takes the full-sphere regrid; "
                f"{', '.join(boxed)} cut a regional box.  Drop {', '.join(boxed)}, or "
                f"pass the regional mesh"
            )
        return mesh["kind"], mesh
    if boxed:
        return "regional", None
    raise GlobalIntermediateRefusal(
        f"--source {source} needs the mesh it is for: --grid MESH.nc (an MPAS grid, "
        f"static or init file; a global mesh gets the full-sphere {0.125:g}-degree "
        f"regrid, a regional cull the regional door), or --from-plan/--point for a "
        f"regional box"
    )


def run_icon_regional_intermediate(arguments: argparse.Namespace,
                                   mesh: dict[str, Any] | None) -> int:
    """The regional ICON door: the ``field-files`` rows of :data:`GLOBAL_SOURCES`.

    The remap window comes from ``--grid`` (the regional mesh's extent plus
    a halo, :func:`woof.hex.gdt101_door.choose_remap`); a --from-plan/--point
    box is refused there by name.
    """

    return run_global_intermediate(arguments)


def run_icon_intermediate(arguments: argparse.Namespace) -> int:
    kind, mesh = icon_target_kind(arguments)
    if kind == "global":
        from .icon_global_door import run_icon_global_intermediate

        return run_icon_global_intermediate(arguments, mesh or {})
    return run_icon_regional_intermediate(arguments, mesh)


# ---------------------------------------------------------------------------
# argparse glue (the parser itself lives in hrrr_intermediate)
# ---------------------------------------------------------------------------
def request_from_arguments(arguments: argparse.Namespace) -> GlobalRequest:
    from .hrrr_intermediate import parse_cycle

    source = global_source(arguments.source)
    boxed = [flag for flag, value in (
        ("--from-plan", getattr(arguments, "from_plan", None)),
        ("--point", getattr(arguments, "point", None)),
        ("--radius-km", getattr(arguments, "radius_km", None)),
    ) if value is not None]
    remapped = source.layout == "field-files"
    if boxed and remapped:
        raise GlobalIntermediateRefusal(
            f"{', '.join(boxed)} cut a regridding box, and --source {source.name} takes "
            f"its remap window from the target mesh itself: pass --grid GRID.nc"
        )
    if boxed:
        raise GlobalIntermediateRefusal(
            f"{', '.join(boxed)} cut a regridding box, and --source {source.name} "
            f"is not regridded: it is written on its own uniform lat-lon grid.  "
            f"Fetch an area subset instead (`woof fetch --area`) if the file "
            f"size matters"
        )
    grid = getattr(arguments, "grid", None)
    if grid is not None and not remapped:
        raise GlobalIntermediateRefusal(
            f"--grid places the remap window of an unstructured-mesh source "
            f"({', '.join(n for n, s in GLOBAL_SOURCES.items() if s.layout == 'field-files')}); "
            f"--source {source.name} is written on its own uniform lat-lon grid"
        )
    if remapped and grid is None:
        raise GlobalIntermediateRefusal(
            f"--source {source.name} is remapped from its native mesh onto a window over "
            f"the target hex mesh; pass --grid GRID.nc (the grid, static or init file "
            f"of the mesh the intermediates must cover)"
        )
    # Flags that belong to other rows of the shared parser (some exist only
    # once the wrfout door is in the tree; getattr keeps this row agnostic).
    foreign = [flag for flag, attribute in (
        ("--wrfout-glob", "wrfout_glob"), ("--cull-region", "cull_region"),
        ("--halo-km", "halo_km"), ("--wrf-edge-cells", "wrf_edge_cells"),
    ) if getattr(arguments, attribute, None) is not None]
    if foreign:
        raise GlobalIntermediateRefusal(
            f"{', '.join(foreign)} belong to another --source; --source {source.name} "
            f"is a GRIB row read from --grib-dir"
        )
    # Per-row required arguments and per-row defaults: the shared parser
    # leaves them unset so each row can say what it needs.
    for flag, value in (("--grib-dir", getattr(arguments, "grib_dir", None)),
                        ("--cycle", getattr(arguments, "cycle", None))):
        if value is None:
            raise GlobalIntermediateRefusal(f"--source {source.name} needs {flag}; it has no default")
    interval = getattr(arguments, "interval_hours", None)
    interval = source.default_interval_hours if interval is None else int(interval)
    leads = parse_lead_range(getattr(arguments, "hours", None), interval)
    workers = getattr(arguments, "workers", None)
    workers = 4 if workers is None else int(workers)
    if workers < 1:
        raise GlobalIntermediateRefusal(f"--workers {workers} is not a worker count")
    return GlobalRequest(
        source=source, grib_dir=Path(arguments.grib_dir), cycle=parse_cycle(arguments.cycle),
        leads=leads, interval_hours=interval, out_dir=Path(arguments.out_dir),
        prefix=str(getattr(arguments, "prefix", "MET") or "MET"),
        engine=getattr(arguments, "decoder", None),
        vtable=getattr(arguments, "vtable", None), workers=workers,
        window_grid=None if grid is None else Path(grid),
    )


def run_global_intermediate(arguments: argparse.Namespace) -> int:
    request = request_from_arguments(arguments)
    receipt = build_global_intermediates(request)
    switches = receipt["init_switches"]
    first = receipt["files"][0]["path"]
    flags = (
        f"--nfglevels {switches['--nfglevels']} --nfgsoillevels "
        f"{switches['--nfgsoillevels']} --use-spechumd {switches['--use-spechumd']} "
        f"--extrap-airtemp {switches['--extrap-airtemp']}"
    )
    summary: dict[str, Any] = {
        "files": [item["path"] for item in receipt["files"]],
        "receipt": str(Path(request.out_dir).expanduser().absolute() / "intermediate-receipt.json"),
        "init_switches": switches,
        "next": f"woof hex init --met {first} ... {flags}",
    }
    if receipt.get("attribution"):
        summary["attribution"] = receipt["attribution"]
    if len(receipt["files"]) > 1:
        summary["next_lbc"] = (
            f"woof hex lbc --grid INIT.nc --met-dir {Path(request.out_dir).expanduser().absolute()} "
            f"--out-dir LBC --start-time {receipt['valid_times'][0]} --stop-time "
            f"{receipt['valid_times'][-1]} --nfglevels {switches['--nfglevels']} "
            f"--use-spechumd {switches['--use-spechumd']} --extrap-airtemp "
            f"{switches['--extrap-airtemp']}"
        )
    print(json.dumps(summary, indent=2))
    return 0


__all__ = [
    "ENGINE_RECEIPT_SCHEMA",
    "GLOBAL_SCHEMA",
    "GLOBAL_SOURCES",
    "GlobalIntermediateRefusal",
    "GlobalRequest",
    "GlobalSource",
    "ICON_SOURCES",
    "LeadPlan",
    "REFUSED_SOURCES",
    "build_global_intermediates",
    "engine_argv",
    "global_source",
    "icon_target_kind",
    "init_switches",
    "locate_inputs",
    "packaged_vtable",
    "parse_engine_receipt",
    "parse_lead_range",
    "plan_leads",
    "refuse_occupied_out_dir",
    "refuse_unknown_source",
    "request_from_arguments",
    "resolve_engine",
    "run_global_intermediate",
    "run_icon_intermediate",
    "run_icon_regional_intermediate",
    "stage_field_files",
    "validate_intermediate",
]
