"""WRF's CLWRF statistics, RASM mean_diag/diurnal_diag and surface_energy_diag on the MPAS-A v8.4.2 suite route.

A WOOF-only output; MPAS-A v8.4.2 has none.  The WOOF namelist group ``&diagnostics`` takes WRF's controls under MPAS
style names::

    config_output_diagnostics    logical  WRF output_diagnostics (default false): CLWRF climate statistics
    config_auxhist3_interval     integer  WRF auxhist3_interval, minutes: the CLWRF window (needed with it, and only then)
    config_mean_diag             logical  WRF mean_diag (default false): RASM interval means
    config_mean_freq             integer  WRF mean_freq 1 s, 2 min, 3 h, 4 day, 5 calendar month
    config_mean_interval         integer  WRF mean_interval, in mean_freq units (not read with mean_freq 5)
    config_diurnal_diag          logical  WRF diurnal_diag (default false): RASM monthly three-hour diurnal means
    config_surface_energy_diag   logical  WRF surface_energy_diag (WOOF's output control): GSW and the accumulated
                                          surface fluxes

and writes ``diag`` fields (members of the ``diagnostics`` stream, with the WRF names and text of woof.core.clwrf,
woof.core.rasm and the WRF Registry), under the packages named below.  With every option off nothing here is allocated,
constructed or compiled, and the Registry, the default streams and the driver receipt are the ones a run without this
module has.

Fields::

    package clwrf (CLWRF, module_diag_cl.F)  38 x [nCells]  clwrf_t2min ... clwrf_rainncstd: min/max/mean/std (and the time of
                                    min/max, minutes) of T2 (Celsius accumulation, +273 on output), Q2, TSK, the 10 m wind
                                    components and speed, and the convective and grid-scale precipitation rates.  The
                                    ``clwrf_`` prefix keeps them apart from nwp_diag_v842's T2MAX/T2MIN, which WRF's
                                    Registry shares with CLWRF under one name.
    package rasm_mean               18 x [nCells]   rasm_<f>_mean: interval means (WRF PSFC_M ... LWDNT_M)
    package rasm_diurnal            17 x [nDiurnalBins = 8, nCells]  rasm_<f>_diurn: the monthly mean of each three-hour bin
    package surface_energy_diag     achfx aclhf acgrdflx [nCells], J m-2: hfx, lh and grdflx times dt, summed over the run
                                    (WRF's ACHFX, ACLHF, ACGRDFLX); and ``gsw`` (the Registry's net shortwave at the ground)
                                    is added to the ``diagnostics`` stream.

Nothing is re-derived.  CLWRF is woof/core/kernels/clwrf.cu (``clwrf_output_calc``), RASM is
woof/core/kernels/native_surface_diagnostics.cu (``rasm_mean``, ``rasm_diurnal``) through woof.core.native_surface_diagnostics
and woof.core.rasm's own calendar alarms, the flux sums are woof.core.bandwidth_glue.accumulate_scaled; each runs with one mesh
cell as one WRF column (``nx = nCells``, ``ny = 1``; every kernel is column local).  The inputs are MPAS's own words:
``t2m th2m q2 u10 v10 skintemp hfx lh grdflx swdnb glw lwupb swupb swupt swdnt lwupt lwdnt raincv rainncv``
(``diag_physics``), ``surface_pressure`` (``diag``), and, for RASM's PMSL, the lowest level's ``theta - 300``,
``pressure_p``, ``pressure_base``, ``qv`` and the terrain height ``zgrid(1)``.

Clock.  Every statistic folds after each step with the clock of the step's START (``when - writer.start - dt``), as WRF's
``curr_secs2`` is: reset on the step that starts a window, finish on the step that ends it, so the frame at a window's
end holds the finished statistic, and the t = 0 frame (no step has run) holds zero.  The monthly and three-hour
calendars use the run's start time plus that elapsed time.  A window must be a whole number of steps and the diurnal
three-hour bins must fall on step boundaries (``10800 % dt == 0``), or WRF's float32 MOD test never fires: refused at setup.

Differences from the WRF route:

* RASM's PMSL reads the lowest level's potential temperature of the state the step ended on; WRF's reads the preceding
  solve's (its ``th_phy_m_t0``), a lag with no MPAS counterpart;
* the radiation fluxes are the held values MPAS keeps between radiation calls, as WRF's are.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Mapping

import numpy as np

from woof.core import clwrf as _clwrf
from woof.core import rasm as _rasm
from woof.core.native_surface_diagnostics import MEANS, SURFACE

from .utils import Direction

GROUP = "diagnostics"
OPTIONS = ("config_output_diagnostics", "config_auxhist3_interval", "config_mean_diag", "config_mean_freq",
           "config_mean_interval", "config_diurnal_diag", "config_surface_energy_diag")
CLWRF_PACKAGE, MEAN_PACKAGE, DIURNAL_PACKAGE, ENERGY_PACKAGE = "clwrf", "rasm_mean", "rasm_diurnal", "surface_energy_diag"
PACKAGES = (CLWRF_PACKAGE, MEAN_PACKAGE, DIURNAL_PACKAGE, ENERGY_PACKAGE)
DIMENSIONS = {"nDiurnalBins": DIURNAL_PACKAGE}
DIURNAL_BINS = 8

#: the MPAS field that is each WRF surface argument of RASM: (pool, name)
SURFACE_SOURCE = {
    "psfc": ("diag", "surface_pressure"), "tsk": ("diag_physics", "skintemp"), "t2": ("diag_physics", "t2m"),
    "th2": ("diag_physics", "th2m"), "q2": ("diag_physics", "q2"), "u10": ("diag_physics", "u10"),
    "v10": ("diag_physics", "v10"), "hfx": ("diag_physics", "hfx"), "lh": ("diag_physics", "lh"),
    "swdnb": ("diag_physics", "swdnb"), "glw": ("diag_physics", "glw"), "lwupb": ("diag_physics", "lwupb"),
    "swupb": ("diag_physics", "swupb"), "swupt": ("diag_physics", "swupt"), "swdnt": ("diag_physics", "swdnt"),
    "lwupt": ("diag_physics", "lwupt"), "lwdnt": ("diag_physics", "lwdnt")}
#: CLWRF's kernel inputs and where MPAS keeps them
CLWRF_SOURCE = {"t2": ("diag_physics", "t2m"), "q2": ("diag_physics", "q2"), "tsk": ("diag_physics", "skintemp"),
                "u10": ("diag_physics", "u10"), "v10": ("diag_physics", "v10"),
                "raincv": ("diag_physics", "raincv"), "rainncv": ("diag_physics", "rainncv")}
#: WRF's surface-energy accumulators (woof.core.physics_inventory.SURFACE_ENERGY_ACCUMULATORS) and their sources
ENERGY = {"achfx": ("hfx", "ACCUMULATED UPWARD HEAT FLUX AT THE SURFACE", "ACHFX"),
          "aclhf": ("lh", "ACCUMULATED UPWARD LATENT HEAT FLUX AT THE SURFACE", "ACLHF"),
          "acgrdflx": ("grdflx", "ACCUMULATED GROUND HEAT FLUX", "ACGRDFLX")}

CLWRF_NAMES = tuple("clwrf_" + symbol.lower() for symbol, _, _ in _clwrf.HISTORY.values())
CLWRF_SLOT = dict(zip(CLWRF_NAMES, _clwrf.HISTORY))          # MPAS name -> woof.core.clwrf slot
MEAN_NAMES = tuple(f"rasm_{k}_mean" for k in MEANS)
DIURNAL_NAMES = tuple(f"rasm_{k}_diurn" for k in SURFACE)
ENERGY_NAMES = tuple(ENERGY)

FIELDS: dict[str, dict[str, Any]] = {}


def _add(name, dims, units, text, wrf, gate):
    FIELDS[name] = {"type": "real", "dimensions": [*dims, "Time"],
                    "attrs": {"units": units, "long_name": text, "wrf_name": wrf}, "gates": [gate]}


for _name, _slot in CLWRF_SLOT.items():
    _symbol, _units, _text = _clwrf.HISTORY[_slot]
    _add(_name, ["nCells"], _units.strip(), _text, _symbol, CLWRF_PACKAGE)
for _name in MEAN_NAMES:
    _symbol, _units, _text = _rasm.HISTORY[_name]
    _add(_name, ["nCells"], _units, _text, _symbol, MEAN_PACKAGE)
for _name in DIURNAL_NAMES:
    _symbol, _units, _text = _rasm.HISTORY[_name]
    _add(_name, ["nDiurnalBins", "nCells"], _units, _text, _symbol, DIURNAL_PACKAGE)
for _name, (_source, _text, _wrf) in ENERGY.items():
    _add(_name, ["nCells"], "J m-2", _text, _wrf, ENERGY_PACKAGE)
NAMES = tuple(FIELDS)

#: Prefix of this module's keys in the output diagnostics' checkpoint accumulator dict.
STATE_PREFIX = "red:"


@dataclass(frozen=True)
class ReductionConfig:
    clwrf_interval: int = 0                  # config_auxhist3_interval with config_output_diagnostics, minutes
    mean_diag: bool = False
    mean_freq: int = 0
    mean_interval: int = 0
    diurnal_diag: bool = False
    surface_energy_diag: bool = False


def field_definition(name: str) -> dict[str, Any]:
    row = FIELDS[name]
    return {"type": row["type"], "dimensions": list(row["dimensions"]),
            "attrs": dict(row["attrs"]), "gates": list(row["gates"])}


def _logical(key: str, value: Any) -> bool:
    if isinstance(value, str) and value.strip().strip(".").casefold() in ("true", "t", "false", "f"):
        value = value.strip().strip(".").casefold() in ("true", "t")
    if not isinstance(value, bool):
        raise ValueError(f"{key} is logical (.true./.false.), not {value!r}")
    return value


def _integer(key: str, value: Any, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{key} is an integer >= {minimum}, not {value!r}")
    return value


def resolve(groups: Mapping[str, Mapping[str, Any]]) -> ReductionConfig | None:
    """The options from the namelist's &diagnostics group (None when none is on)."""
    s = {str(k).casefold(): v for k, v in groups.get(GROUP, {}).items()}
    clwrf = _logical("config_output_diagnostics", s.get("config_output_diagnostics", False))
    interval = _integer("config_auxhist3_interval", s.get("config_auxhist3_interval", 0), 0)
    mean = _logical("config_mean_diag", s.get("config_mean_diag", False))
    diurnal = _logical("config_diurnal_diag", s.get("config_diurnal_diag", False))
    energy = _logical("config_surface_energy_diag", s.get("config_surface_energy_diag", False))
    if clwrf and interval <= 0:
        raise ValueError("config_output_diagnostics needs config_auxhist3_interval > 0 (minutes): WRF's CLWRF takes MOD "
                         "by the window")
    if interval and not clwrf:
        raise ValueError("config_auxhist3_interval is set while config_output_diagnostics is off; it would be read by nothing")
    freq = _integer("config_mean_freq", s.get("config_mean_freq", 0), 0)
    length = _integer("config_mean_interval", s.get("config_mean_interval", 0), 0)
    if mean:
        if freq not in (1, 2, 3, 4, 5) or (freq != 5 and length <= 0):
            raise ValueError("config_mean_diag needs config_mean_freq 1..5 and, unless it is 5 (calendar month), a positive "
                             "config_mean_interval, as WRF's RASM")
    elif freq or length:
        raise ValueError("config_mean_freq/config_mean_interval are set while config_mean_diag is off; they would be read "
                         "by nothing")
    if not (clwrf or mean or diurnal or energy):
        return None
    return ReductionConfig(interval if clwrf else 0, mean, freq, length, diurnal, energy)


def packages(config: Any) -> set[str]:
    r = getattr(config, "reduction_diagnostics", None)
    if r is None:
        return set()
    return ({CLWRF_PACKAGE} if r.clwrf_interval else set()) | ({MEAN_PACKAGE} if r.mean_diag else set()) \
        | ({DIURNAL_PACKAGE} if r.diurnal_diag else set()) | ({ENERGY_PACKAGE} if r.surface_energy_diag else set())


def extend_registry(root, active: set[str]) -> None:
    """Add the dimension, the ``diag`` fields and their ``diagnostics`` stream members (and ``gsw``) of each active
    package to a parsed Registry tree; otherwise leave it as parsed."""
    import xml.etree.ElementTree as ET
    if not active:
        return
    dims = root.find("./dims")
    struct = root.find("./var_struct[@name='diag']")
    stream = root.find("./streams/stream[@name='diagnostics']")
    for dim, package in DIMENSIONS.items():
        if package in active and dims.find(f"./dim[@name='{dim}']") is None:
            ET.SubElement(dims, "dim", {"name": dim, "definition": str(DIURNAL_BINS)})
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
    if ENERGY_PACKAGE in active and stream is not None and stream.find("./var[@name='gsw']") is None:
        ET.SubElement(stream, "var", {"name": "gsw", "packages": ENERGY_PACKAGE})


class ReductionModule:
    """Fold the statistics after every step, publish them before a write.  ``owner`` is the route's
    ResidentOutputDiagnostics (its device pools, buffers and stream views)."""

    def __init__(self, owner, config: ReductionConfig):
        self.owner, self.config = owner, config
        self._seconds = None
        self._now = None

    # -- setup -----------------------------------------------------------------------------------------
    def setup(self, utils, dt):
        cp = self.cp = self.owner.cp
        dims = self.owner.session.layout.dimensions
        self.nc, self.nl = int(dims["nCellsSolve"]), int(dims["nVertLevels"])
        self.utils, self.dt = utils, float(dt)
        c, nc, f32 = self.config, self.nc, cp.float32
        carried = lambda names: [n for n in names if utils.stream_inclusion_count(n, Direction.OUTPUT) > 0]
        self.clwrf = carried(CLWRF_NAMES) if c.clwrf_interval else []
        self.mean = carried(MEAN_NAMES) if c.mean_diag else []
        self.diurnal = carried(DIURNAL_NAMES) if c.diurnal_diag else []
        self.energy = carried(ENERGY_NAMES) if c.surface_energy_diag else []
        self.accumulates = bool(self.clwrf or self.mean or self.diurnal or self.energy)
        windows = []
        if self.clwrf:
            windows.append(("config_auxhist3_interval", 60.0 * c.clwrf_interval))
        if self.mean and c.mean_freq != 5:
            windows.append(("config_mean_interval", c.mean_interval * {1: 1, 2: 60, 3: 3600, 4: 86400}[c.mean_freq]))
        for key, seconds in windows:
            if abs(seconds / self.dt - round(seconds / self.dt)) > 1e-9 or seconds / self.dt < 0.5:
                raise ValueError(f"{key} ({seconds:g} s) is not a whole number of {self.dt:g} s steps: WRF's window clock "
                                 "(NINT and MOD of the step count) would never close a window")
        if self.diurnal and not (self.dt == int(self.dt) and 10800 % int(self.dt) == 0):
            raise ValueError("config_diurnal_diag needs a whole-second step that divides three hours: WRF's three-hour "
                             "bins must close on a step boundary")
        self.words = {}
        z2 = lambda: cp.zeros(nc, dtype=f32)
        if self.clwrf:
            self.words.update({_clwrf_slot: z2() for _clwrf_slot in _clwrf.HISTORY})
            self.words["clwrf_nsteps"] = cp.zeros(nc, dtype=cp.int32)
        if self.mean:
            self.words.update({n: z2() for n in MEAN_NAMES})
            self.words["rasm_nsteps_mean"] = cp.zeros(nc, dtype=cp.int32)
        if self.diurnal:
            self.words.update({n: cp.zeros((DIURNAL_BINS, nc), dtype=f32) for n in DIURNAL_NAMES})
            self.words.update({f"rasm_{k}_dtmp": z2() for k in SURFACE})
            self.words["rasm_nsteps_dtmp"] = cp.zeros(nc, dtype=cp.int32)
            self.words["rasm_nsteps_diurn"] = cp.zeros(nc, dtype=cp.int32)
        if self.energy:
            self.words.update({n: z2() for n in ENERGY_NAMES})
        self.zero = z2()

    # -- inputs ----------------------------------------------------------------------------------------
    def _has(self, pool, name):
        return any(row["pool"] == pool and row.get("pool_name", row.get("name")) == name
                   for row in self.owner.session.layout.fields)

    def _plane(self, pool, name):
        if not self._has(pool, name):
            return self.zero
        return self.cp.ascontiguousarray(self.owner.authority(pool, name)[:self.nc], dtype=self.cp.float32)

    def _level0(self, pool, name):
        return self.cp.ascontiguousarray(self.owner.authority(pool, name)[0, :self.nc], dtype=self.cp.float32)

    def _qv0(self):
        indices = self.owner.session.layout.indices
        key = ("state", "index_qv")
        if key not in indices or int(indices[key]) < 1:
            return self.zero
        row = int(indices[key]) - 1
        return self.cp.ascontiguousarray(self.owner.authority("state", "scalars")[row, 0, :self.nc], dtype=self.cp.float32)

    # -- the per-step fold -----------------------------------------------------------------------------
    def begin(self, step, when):
        """The step the next :meth:`update` folds (steps <= 0 fold nothing); its START time."""
        if int(step) <= 0:
            self._seconds = self._now = None
            return
        start = self.owner.writer.start
        self._seconds = (when - start).total_seconds() - self.dt
        self._now = start + timedelta(seconds=self._seconds)

    def update(self):
        if not self.accumulates or self._seconds is None or not self.nc:
            return
        c, s, nc, np_ = self.config, self.words, self.nc, np
        if self.clwrf:
            inputs = {k: self._plane(*src) for k, src in CLWRF_SOURCE.items()}
            _clwrf.launch(inputs, {slot: s[slot] for slot in _clwrf.HISTORY}, s["clwrf_nsteps"], dt=self.dt,
                          xtime=self._seconds / 60.0, seconds=self._seconds, interval=c.clwrf_interval)
        if self.mean or self.diurnal:
            surface = {k: self._plane(*SURFACE_SOURCE[k]).reshape(1, nc) for k in SURFACE}
        if self.mean:
            from woof.core.native_surface_diagnostics import rasm_mean
            interval = c.mean_interval * {1: 1, 2: 60, 3: 3600, 4: 86400, 5: 1}[c.mean_freq]
            reset, finish = _rasm.alarm(self._seconds, self.dt, interval, monthly=c.mean_freq == 5, now=self._now)
            atmosphere = dict(t=(self._level0("diag", "theta") - np_.float32(300)).reshape(1, nc),
                              p=self._level0("diag", "pressure_p").reshape(1, nc),
                              pb=self._level0("diag", "pressure_base").reshape(1, nc),
                              moist=self._qv0().reshape(1, nc),
                              ht=self.cp.ascontiguousarray(self.owner.authority("mesh", "zgrid")[0, :nc],
                                                           dtype=self.cp.float32).reshape(1, nc))
            rasm_mean(surface, atmosphere, {k: s[f"rasm_{k}_mean"].reshape(1, nc) for k in MEANS},
                      s["rasm_nsteps_mean"].reshape(1, nc), reset=reset, finish=finish, _validated=True)
        if self.diurnal:
            from woof.core.native_surface_diagnostics import rasm_diurnal
            reset, finish = _rasm.alarm(self._seconds, self.dt, 10800)
            rm, fm = _rasm.alarm(self._seconds, self.dt, 1, monthly=True, now=self._now)
            hour = (self._now + timedelta(seconds=int(self.dt))).hour
            if finish and hour % 3:
                raise RuntimeError("native RASM cannot select this diurnal cycle")
            rasm_diurnal(surface, {k: s[f"rasm_{k}_dtmp"].reshape(1, nc) for k in SURFACE},
                         {k: s[f"rasm_{k}_diurn"] for k in SURFACE}, s["rasm_nsteps_dtmp"].reshape(1, nc),
                         s["rasm_nsteps_diurn"].reshape(1, nc), reset=reset, reset_month=rm, finish=finish,
                         finish_month=fm, cycle=hour // 3 or 8, _validated=True)
        if self.energy:
            from woof.core.bandwidth_glue import accumulate_scaled
            scale = np_.float32(self.dt)
            for name, (source, _, _) in ENERGY.items():
                src = self._plane("diag_physics", source)
                if not accumulate_scaled(s[name], src, scale):
                    s[name] += src * scale

    # -- publication -----------------------------------------------------------------------------------
    def compute(self):
        """Publish the statistics into the diag buffers a due stream will write."""
        nc, s, write = self.nc, self.words, self.utils.field_will_be_written
        for name in self.clwrf:
            if write(name):
                self.owner.buffer("diag", name)[:nc] = s[CLWRF_SLOT[name]]
        for name in (*self.mean, *self.energy):
            if write(name):
                self.owner.buffer("diag", name)[:nc] = s[name]
        for name in self.diurnal:
            if write(name):
                self.owner.buffer("diag", name)[:DIURNAL_BINS, :nc] = s[name]

    # -- checkpoint continuation -----------------------------------------------------------------------
    def state(self):
        return {STATE_PREFIX + k: self.cp.asnumpy(v) for k, v in self.words.items()}

    def resume(self, state):
        for key, value in state.items():
            if key.startswith(STATE_PREFIX) and key[len(STATE_PREFIX):] in self.words:
                target = self.words[key[len(STATE_PREFIX):]]
                target[...] = self.cp.asarray(value).reshape(target.shape)
