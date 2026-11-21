"""The space-weather history rows (spacewx_ionosphere_opt, spacewx_aurora_opt) on the MPAS-A v8.4.2 suite route.

A WOOF-only output; MPAS-A v8.4.2 has none.  The run names WRF's keys with ``--physics``, as the WRF route names
them in its namelist; this route takes only these keys out of the overrides (woof.hex.mpasphys.forecast_v842),
records them in the driver receipt and, with the drivers each history object resolved, in the run log (the
``[spacewx suite]`` line), and keeps refusing every other WRF key by name (the MPAS Registry has none of them)::

    spacewx_ionosphere_opt   0 / 1          the climatological ionosphere (woof.spacewx.history)
    spacewx_f107_source      'gfz', a path, or a flux in sfu
    spacewx_f107_mode        '81day' / 'daily'
    spacewx_ionosphere_maps  'ccir' / 'ursi'
    spacewx_aurora_opt       0 / 1 / 2 / 3  the auroral oval (2: with OVATION Prime's energy flux; 3: and the Joule power)
    spacewx_kp_source        'gfz', 'swpc', 'omni', a path, or a Kp value

The keys are judged by the WRF route's own checks (woof.config.validate_spacewx_ionosphere_keys /
validate_spacewx_aurora_keys); the WRF route's radiation-on condition does not apply, as the rows sit on the cell
centres here.

The fields are ``diag`` fields [nCells] in the ``diagnostics`` stream, the WRF route's rows under lower-case names
(``wrf_name`` the wrfout name) with its units and text, packaged ``spacewx_ionosphere``, ``spacewx_aurora`` and
``spacewx_aurora_energy`` (options 2 and 3) and ``spacewx_joule`` (option 3).  With the keys absent nothing here is allocated or constructed, and the
Registry, the default streams and the driver receipt are the ones a run without this module has.

Nothing is re-derived: the rows are woof.spacewx.history.IonosphereHistory and woof.spacewx.aurora_history.
AuroraHistory, built at setup on the solve cells' centres (latCell/lonCell in degrees) and the case start, and
evaluated at each due frame's valid time.  Diagnostic only: nothing the step reads, so the keys stay out of the
checkpoint's physics identity (a resume may switch the rows on or off, as on the WRF route).  A checkpoint carries
the rows' start, and a continuation rebuilds them on it, so it writes the rows the straight run would.  Each rank
of a multi-rank mesh builds its own cells' rows; a fetched Kp is held per source and start in the space-weather
cache, so the ranks read one value.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from .utils import Direction

ION_PACKAGE, AURORA_PACKAGE = "spacewx_ionosphere", "spacewx_aurora"
ENERGY_PACKAGE, JOULE_PACKAGE = "spacewx_aurora_energy", "spacewx_joule"
PACKAGES = (ION_PACKAGE, AURORA_PACKAGE, ENERGY_PACKAGE, JOULE_PACKAGE)
#: The --physics keys this route takes, with the WRF RunConfig defaults.
DEFAULTS = {"spacewx_ionosphere_opt": 0, "spacewx_f107_source": "", "spacewx_f107_mode": "81day",
            "spacewx_ionosphere_maps": "ccir", "spacewx_aurora_opt": 0, "spacewx_kp_source": ""}
KEYS = tuple(DEFAULTS)
_INTEGERS = ("spacewx_ionosphere_opt", "spacewx_aurora_opt")


def _rows():
    from woof.spacewx import aurora_history as ah
    from woof.spacewx import history as hist
    return {ION_PACKAGE: hist.HISTORY_FIELDS, AURORA_PACKAGE: ah.HISTORY_FIELDS,
            ENERGY_PACKAGE: ah.ENERGY_HISTORY_FIELDS, JOULE_PACKAGE: ah.JOULE_HISTORY_FIELDS}


def _field(package: str, wrf_name: str, description: str, units: str) -> dict[str, Any]:
    return {"type": "real", "dimensions": ["nCells", "Time"],
            "attrs": {"units": units, "long_name": description, "wrf_name": wrf_name}, "gates": [package]}


FIELDS = {wrf.lower(): _field(package, wrf, *row[-2:])            # (..., description, units)
          for package, rows in _rows().items() for wrf, row in rows.items()}
NAMES = tuple(FIELDS)


def field_definition(name: str) -> dict[str, Any]:
    row = FIELDS[name]
    return {"type": row["type"], "dimensions": list(row["dimensions"]),
            "attrs": dict(row["attrs"]), "gates": list(row["gates"])}


@dataclass(frozen=True)
class SpacewxConfig:
    settings: Mapping[str, Any]

    @property
    def ionosphere(self) -> bool:
        return int(self.settings["spacewx_ionosphere_opt"]) == 1

    @property
    def aurora(self) -> int:
        return int(self.settings["spacewx_aurora_opt"])

    @property
    def packages(self) -> set[str]:
        return ({ION_PACKAGE} if self.ionosphere else set()) | ({AURORA_PACKAGE} if self.aurora else set()) | (
            {ENERGY_PACKAGE} if self.aurora >= 2 else set()) | ({JOULE_PACKAGE} if self.aurora == 3 else set())

    def receipt(self) -> dict:
        return dict(self.settings)


def is_key(key: str) -> bool:
    return key.strip().casefold() in KEYS


def resolve(given: Mapping[str, str]) -> SpacewxConfig | None:
    """The --physics spacewx keys (KEY -> the text after ``=``), typed and judged as the WRF route judges them;
    None when none is given.  Off with every key at its default (``spacewx_ionosphere_opt=0``) is None too."""
    if not given:
        return None
    values = dict(DEFAULTS)
    for key, text in given.items():
        key, text = key.strip().casefold(), str(text).strip().strip("\"'")
        if key in _INTEGERS:
            try:
                values[key] = int(text)
            except ValueError:
                raise ValueError(f"--physics {key}={text}: an integer") from None
        else:
            values[key] = text
    from woof.config import ConfigurationRefused, validate_spacewx_aurora_keys, validate_spacewx_ionosphere_keys
    try:
        ion = validate_spacewx_ionosphere_keys(values["spacewx_ionosphere_opt"], values["spacewx_f107_source"],
                                               values["spacewx_f107_mode"], values["spacewx_ionosphere_maps"])
        aurora = validate_spacewx_aurora_keys(values["spacewx_aurora_opt"], values["spacewx_kp_source"])
    except ConfigurationRefused as exc:
        raise ValueError(f"--physics spacewx keys on the MPAS v8.4.2 suite route: {exc}") from None
    if not (ion or aurora):
        return None
    return SpacewxConfig(values)


def configure(config, given: Mapping[str, str]):
    """``config`` (a DriverConfig) with the rows the keys ask for, recorded in the run log."""
    spacewx = resolve(given)
    if spacewx is None:
        return config
    from dataclasses import replace
    return replace(config, spacewx_diagnostics=spacewx)


def packages(config: Any) -> set[str]:
    spacewx = getattr(config, "spacewx_diagnostics", None)
    return spacewx.packages if spacewx is not None else set()


def extend_registry(root, active: set[str]) -> None:
    """Add the active packages' fields to the ``diag`` struct and the ``diagnostics`` stream of a parsed Registry."""
    import xml.etree.ElementTree as ET
    struct = root.find("./var_struct[@name='diag']")
    stream = root.find("./streams/stream[@name='diagnostics']")
    for name, row in FIELDS.items():
        package = row["gates"][0]
        if package not in active:
            continue
        if struct.find(f"./var[@name='{name}']") is None:
            ET.SubElement(struct, "var", {
                "name": name, "type": "real", "dimensions": " ".join(row["dimensions"]),
                "units": row["attrs"]["units"], "description": row["attrs"]["long_name"],
                "default_value": "0.0", "packages": package})
        if stream is not None and stream.find(f"./var[@name='{name}']") is None:
            ET.SubElement(stream, "var", {"name": name, "packages": package})


def holders(spacewx: SpacewxConfig, *, latitude_deg, longitude_deg, start_time, run_seconds) -> tuple:
    """The WRF route's history objects on the cell centres (degrees), ionosphere then aurora."""
    from woof.spacewx.aurora_history import AuroraHistory
    from woof.spacewx.history import IonosphereHistory
    lat = np.asarray(latitude_deg, dtype=np.float64).reshape(1, -1)
    lon = np.asarray(longitude_deg, dtype=np.float64).reshape(1, -1)
    cfg = SimpleNamespace(**spacewx.settings, run_seconds=float(run_seconds))
    geometry = SimpleNamespace(start_time=start_time, latitude_deg=lat, longitude_deg=lon)
    built = []
    if spacewx.ionosphere:
        built.append(IonosphereHistory.for_run(cfg, geometry))
    if spacewx.aurora:
        built.append(AuroraHistory.for_run(cfg, geometry))
    return tuple(built)


START_KEY = "spacewx:start"


class SpacewxModule:
    """Build the rows at setup (so a feed that cannot be read refuses before step 1); publish them for every due
    frame.  ``owner`` is the route's ResidentOutputDiagnostics."""

    def __init__(self, owner, config: SpacewxConfig):
        self.owner, self.config = owner, config
        self.holders = None
        self._when = None

    def setup(self, utils):
        dims = self.owner.session.layout.dimensions
        self.nc = int(dims["nCellsSolve"])
        self.utils = utils
        self.dt = float(self.owner.dycore.config.config_dt)
        active = self.config.packages
        self.names = tuple(n for n in NAMES if FIELDS[n]["gates"][0] in active)
        self.carried = {n for n in self.names if utils.stream_inclusion_count(n, Direction.OUTPUT) > 0}
        # The straight run's start, as reduction_diag_v842's origin: a continuation's writer starts at the
        # checkpoint and the steps before it are the owner's step_offset.  A checkpoint that carries the rows' own
        # start (state()) replaces it on resume(), so an exact-mode continuation (step_offset 0) keeps it too.
        from woof.spacewx.history import _naive_utc
        writer = self.owner.writer
        start = _naive_utc(writer.start) - timedelta(seconds=int(getattr(self.owner, "step_offset", 0)) * self.dt)
        self._build(start, _naive_utc(writer.final))

    def _build(self, start, final=None):
        if final is not None:
            self._final = final
        cp = self.owner.cp
        lat = np.degrees(cp.asnumpy(self.owner.authority("mesh", "latCell")[:self.nc]).astype(np.float64))
        lon = np.degrees(cp.asnumpy(self.owner.authority("mesh", "lonCell")[:self.nc]).astype(np.float64))
        self.start = start
        self.holders = holders(self.config, latitude_deg=lat, longitude_deg=lon, start_time=start,
                               run_seconds=max(0.0, (self._final - start).total_seconds()))
        print("[spacewx suite] " + json.dumps({"settings": self.config.receipt(), "start": start.isoformat(),
                                               "rows": {type(h).__name__: h.receipt() for h in self.holders}},
                                              sort_keys=True, default=str), flush=True)

    def begin(self, step, when):
        """The frame's valid time (a datetime, or MPAS's ``YYYY-MM-DD_hh:mm:ss``)."""
        from woof.spacewx.history import _naive_utc
        if not isinstance(when, datetime):
            when = datetime.strptime(str(when), "%Y-%m-%d_%H:%M:%S")
        self._when = _naive_utc(when)

    def rows(self, names) -> dict:
        """lower-case name -> the (nCellsSolve,) float32 device row at the frame begin() named, for the history
        objects that carry one of ``names``."""
        out = {}
        for holder in self.holders:
            from woof.spacewx.history import HISTORY_FIELDS
            carries = getattr(holder, "names", None) or tuple(HISTORY_FIELDS)     # IonosphereHistory: the five
            if not {wrf.lower() for wrf in carries} & set(names):
                continue
            elapsed = (self._when - holder.start_time).total_seconds()
            for wrf, row in holder.history_fields(elapsed).items():
                out[wrf.lower()] = row.reshape(-1)[:self.nc]
        return out

    def compute(self):
        """Publish what a due stream will write."""
        due = [n for n in self.carried if self.utils.field_will_be_written(n)]
        if not due:
            return
        rows = self.rows(due)
        for name in due:
            self.owner.buffer("diag", name)[:self.nc] = rows[name]

    # -- checkpoint continuation: the rows' start, so the continuation reads the straight run's drivers ---------
    def state(self):
        return {START_KEY: np.array([(self.start - datetime(1970, 1, 1)).total_seconds()], dtype=np.float64)}

    def resume(self, state):
        if START_KEY not in state:
            return
        start = datetime(1970, 1, 1) + timedelta(seconds=float(np.asarray(state[START_KEY]).reshape(-1)[0]))
        if start != self.start:
            self._build(start)
