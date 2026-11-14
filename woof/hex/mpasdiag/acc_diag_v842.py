"""WRF's prec_acc_dt precipitation windows and acc_phy_tend physics-tendency accumulators on the MPAS-A v8.4.2 suite route.

A WOOF-only output; MPAS-A v8.4.2 has none (its ``rainc``/``rainnc`` accumulate from the start of the run, and the
``tend_physics`` rates are instantaneous).  The WOOF namelist group ``&diagnostics`` takes::

    config_prec_acc_dt            real     WRF prec_acc_dt, minutes (default 0 = off): the window of PREC_ACC_C,
                                           PREC_ACC_NC and SNOW_ACC_NC
    config_acc_phy_tend           logical  WRF acc_phy_tend (default false): the sixteen A*TEN accumulators
    config_acc_phy_tend_interval  real     WRF's history_interval, minutes: the window the tendency accumulators
                                           restart at (required with config_acc_phy_tend, and only then)

and writes the fields below as ``diag`` fields (packages ``prec_acc_dt`` and ``acc_phy_tend``, members of the
``diagnostics`` stream); with neither option on nothing here is allocated, constructed or compiled, and the Registry,
the default streams and the driver receipt are the ones a run without this module has.

Fields (MPAS style lower case of WRF's names)::

    prec_acc_c   PREC_ACC_C    mm   [nCells]  accumulated convective precipitation over the window
    prec_acc_nc  PREC_ACC_NC   mm   [nCells]  accumulated grid-scale precipitation over the window
    snow_acc_nc  SNOW_ACC_NC   mm   [nCells]  accumulated snow (frozen convective rain at 2 m T < 273.15 K added)
    athmpten ... athratensw    the sixteen woof.core.physics_tendency_output.NAMES   [nVertLevels, nCells]

Nothing is re-derived.  The accumulation is WRF 4.7.1's phys/module_diag_misc.F (lines 229-291 and 415-448), run
through the WRF route's own kernels in woof/core/kernels/output_diagnostics.cu (``native_precipitation_buckets`` and
``native_physics_tendency_accumulation``: the same float32 arithmetic, the same ``fmodf`` window clock), with one mesh
cell as one WRF column (``nx = nCells``, ``ny = 1``; both kernels are column local).  The inputs are MPAS's own
step values:

* ``raincv``, ``rainncv``, ``snowncv`` (``diag_physics``, mm per step) and ``t2m`` are WRF's RAINCV, RAINNCV, SNOWNCV
  and T2; a field the run's packages do not allocate is zero, as WRF's ``amount()`` of a scheme that is not on;
* ``rthmpten``, ``rqvmpten``, ``rthcuten``, ``rqvcuten``, ``rucuten``, ``rvcuten``, ``rthblten``, ``rqvblten``,
  ``rublten``, ``rvblten``, ``rthratenlw`` and ``rthratensw`` (``tend_physics``, uncoupled rates per second) are
  WRF's H_DIABATIC, QV_DIABATIC, RTHCUTEN ... RTHRATENSW.  MPAS has no shallow-cumulus tendency pool, so the four
  A*SHTEN fields stay zero, as WRF's do with ``shcu_physics = 0``.  The cumulus group follows the cumulus scheme's
  package (``rthcuten`` allocated), as WRF's ``cu_used``.

Clock.  WRF resets on the step whose START time ``curr_secs2`` is a multiple of the window and folds that step's
amounts in.  A hex step ends at the time ``when`` the output hooks carry, so the clock fed to the kernel is
``when - writer.start - dt``: the frame at the window's end then holds the steps that started inside it, and the t = 0
frame (no step has run) holds zero, as WRF's.  Like WRF's ``curr_secs2`` the origin is the start of this run (a
restart's first leg for WRF; the leg a checkpoint continuation begins for hex), and the window must be a whole
number of steps, or the ``fmodf`` test never fires after the first window (refused at setup).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from woof.core.physics_tendency_output import HISTORY as _TEND_HISTORY, NAMES as TEND_NAMES

from .utils import Direction

GROUP = "diagnostics"
PREC_OPTION = "config_prec_acc_dt"
TEND_OPTION = "config_acc_phy_tend"
TEND_INTERVAL_OPTION = "config_acc_phy_tend_interval"
OPTIONS = (PREC_OPTION, TEND_OPTION, TEND_INTERVAL_OPTION)
PREC_PACKAGE = "prec_acc_dt"
TEND_PACKAGE = "acc_phy_tend"

#: woof.core.output_diagnostics.PRECIP_HISTORY's stock rows (graup_acc_nc and the "1" slots are the HRRR fork's).
PREC_NAMES = ("prec_acc_c", "prec_acc_nc", "snow_acc_nc")
_PREC_ROWS = {
    "prec_acc_c": "Interval convective precipitation",
    "prec_acc_nc": "Interval grid-scale precipitation",
    "snow_acc_nc": "Interval snow including frozen convective rain",
}

FIELDS: dict[str, dict[str, Any]] = {}
for _name, _text in _PREC_ROWS.items():
    FIELDS[_name] = {"type": "real", "dimensions": ["nCells", "Time"],
                     "attrs": {"units": "mm", "long_name": _text, "wrf_name": _name.upper()},
                     "gates": [PREC_PACKAGE]}
for _name in TEND_NAMES:
    _wrf, _units, _text = _TEND_HISTORY[_name]
    FIELDS[_name] = {"type": "real", "dimensions": ["nVertLevels", "nCells", "Time"],
                     "attrs": {"units": _units.strip(), "long_name": _text, "wrf_name": _wrf},
                     "gates": [TEND_PACKAGE]}
NAMES = tuple(FIELDS)

#: WRF's rate argument -> the MPAS tend_physics field that carries it, in physics_tendency_output.NAMES order
#: (None: MPAS has no such pool member; the accumulator stays zero).
RATES = (
    "rthmpten", "rqvmpten", "rthcuten", "rqvcuten", "rucuten", "rvcuten", None, None, None, None,
    "rthblten", "rqvblten", "rublten", "rvblten", "rthratenlw", "rthratensw")

#: Prefix of this module's keys in the output diagnostics' checkpoint accumulator dict.
STATE_PREFIX = "acc:"


@dataclass(frozen=True)
class AccConfig:
    prec_acc_dt: float = 0.0
    acc_phy_tend: bool = False
    acc_phy_tend_interval: float = 0.0


def field_definition(name: str) -> dict[str, Any]:
    row = FIELDS[name]
    return {"type": row["type"], "dimensions": list(row["dimensions"]),
            "attrs": dict(row["attrs"]), "gates": list(row["gates"])}


def _real(key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"&{GROUP} {key} is real (minutes), not {value!r}")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"&{GROUP} {key} must be finite and nonnegative minutes, not {value!r}")
    return value


def resolve(groups: Mapping[str, Mapping[str, Any]]) -> AccConfig | None:
    """The options from the namelist's &diagnostics group (None when neither accumulator is on)."""
    supplied = {str(k).casefold(): v for k, v in groups.get(GROUP, {}).items()}
    prec = _real(PREC_OPTION, supplied.get(PREC_OPTION, 0.0))
    tend = supplied.get(TEND_OPTION, False)
    if isinstance(tend, str) and tend.strip().strip(".").casefold() in ("true", "t", "false", "f"):
        tend = tend.strip().strip(".").casefold() in ("true", "t")
    if not isinstance(tend, bool):
        raise ValueError(f"{TEND_OPTION} is logical (.true./.false.), not {tend!r}")
    interval = _real(TEND_INTERVAL_OPTION, supplied.get(TEND_INTERVAL_OPTION, 0.0))
    if tend:
        if interval <= 0 or interval != int(interval):
            raise ValueError(f"{TEND_OPTION} needs {TEND_INTERVAL_OPTION}: WRF's history_interval, positive whole minutes "
                             "(its float32 MOD window reset)")
    elif TEND_INTERVAL_OPTION in supplied:
        raise ValueError(f"{TEND_INTERVAL_OPTION} is set while {TEND_OPTION} is off; it would be read by nothing")
    if not (prec > 0 or tend):
        return None
    return AccConfig(prec, tend, interval)


def packages(config: Any) -> set[str]:
    acc = getattr(config, "acc_diagnostics", None)
    if acc is None:
        return set()
    return ({PREC_PACKAGE} if acc.prec_acc_dt > 0 else set()) | ({TEND_PACKAGE} if acc.acc_phy_tend else set())


def extend_registry(root, active: set[str]) -> None:
    """Add the fields to the ``diag`` struct and the ``diagnostics`` stream of a parsed Registry tree when their
    package is active; otherwise leave it as parsed."""
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


class AccumulatorModule:
    """Fold the windows after every step, publish them before a write.  ``owner`` is the route's
    ResidentOutputDiagnostics (its device pools, buffers and stream views)."""

    def __init__(self, owner, config: AccConfig):
        self.owner, self.config = owner, config
        self._seconds = None

    # -- setup -----------------------------------------------------------------------------------------
    def setup(self, utils, dt):
        cp = self.cp = self.owner.cp
        dims = self.owner.session.layout.dimensions
        self.nc, self.nl = int(dims["nCellsSolve"]), int(dims["nVertLevels"])
        self.utils, self.dt = utils, float(dt)
        for key, minutes in (("prec_acc_dt", self.config.prec_acc_dt),
                             ("acc_phy_tend_interval", self.config.acc_phy_tend_interval)):
            steps = 60.0 * minutes / self.dt
            if minutes and abs(steps - round(steps)) > 1e-9:
                raise ValueError(f"{key} = {minutes:g} min is not a whole number of {self.dt:g} s steps: WRF's window "
                                 "reset (MOD of the start-of-step clock) would never fire after the first window")
        carried = lambda names: [n for n in names if utils.stream_inclusion_count(n, Direction.OUTPUT) > 0]
        self.prec_names = carried(PREC_NAMES) if self.config.prec_acc_dt > 0 else []
        self.tend_names = carried(TEND_NAMES) if self.config.acc_phy_tend else []
        self.accumulates = bool(self.prec_names or self.tend_names)
        nc, f32 = self.nc, cp.float32
        self.epoch = cp.zeros(2, dtype=f32)
        if self.prec_names:
            self.acc = {n: cp.zeros(nc, dtype=f32) for n in PREC_NAMES}
            self.acc["graup_acc_nc"] = cp.zeros(nc, dtype=f32)     # the kernel's fork graupel bucket; not published
            self.zero = cp.zeros(nc, dtype=f32)
        if self.tend_names:
            self.tend = {n: cp.zeros((self.nl, nc), dtype=f32) for n in TEND_NAMES}
            self.qv_before = cp.zeros(1, dtype=f32)                # read only on the (unused) qv-difference path

    # -- inputs ----------------------------------------------------------------------------------------
    def _has(self, pool, name):
        return any(row["pool"] == pool and row.get("pool_name", row.get("name")) == name
                   for row in self.owner.session.layout.fields)

    def _plane(self, pool, name):
        if not self._has(pool, name):
            return self.zero
        return self.cp.ascontiguousarray(self.owner.authority(pool, name)[:self.nc], dtype=self.cp.float32)

    def _levels(self, pool, name):
        if not self._has(pool, name):
            return None
        return self.cp.ascontiguousarray(self.owner.authority(pool, name)[:self.nl, :self.nc], dtype=self.cp.float32)

    # -- the per-step fold -----------------------------------------------------------------------------
    def begin(self, step, when):
        """The step the next :meth:`update` folds and the time its state has (the END of the step)."""
        if int(step) <= 0:
            self._seconds = None
            return
        elapsed = (when - self.owner.writer.start).total_seconds()
        self._seconds = elapsed - self.dt

    def update(self):
        if not self.accumulates or self._seconds is None or not self.nc:
            return
        from woof.core.kernels import get_kernel
        nc, nl = self.nc, self.nl
        if self.prec_names:
            acc = self.acc
            get_kernel("output_diagnostics", "native_precipitation_buckets")(
                ((nc + 255) // 256,), (256,),
                (self._plane("diag_physics", "raincv"), self._plane("diag_physics", "rainncv"),
                 self._plane("diag_physics", "snowncv"), self.zero, self._plane("diag_physics", "t2m"),
                 acc["prec_acc_c"], acc["prec_acc_nc"], acc["snow_acc_nc"], acc["graup_acc_nc"],
                 self.epoch, np.int32(nc), np.float64(self._seconds),
                 np.float32(self.config.prec_acc_dt)))
        if self.tend_names:
            null = np.uint64(0)
            rates = [self._levels("tend_physics", r) if r else None for r in RATES]
            cu = int(rates[2] is not None)
            interval = 60.0 * self.config.acc_phy_tend_interval
            count = nl * nc
            get_kernel("output_diagnostics", "native_physics_tendency_accumulation")(
                ((count + 255) // 256,), (256,),
                (*(null if a is None else a for a in rates), *(self.tend[n] for n in TEND_NAMES),
                 self.qv_before, self.qv_before, self.epoch,
                 np.int32(count), np.int32(nc), np.int32(1), np.int32(0), np.int32(0),
                 np.int32(cu), np.int32(0), np.float64(self._seconds), np.float32(interval),
                 np.float32(self.dt)))

    # -- publication -----------------------------------------------------------------------------------
    def compute(self):
        """Publish the accumulators into the diag buffers a due stream will write."""
        nc, nl = self.nc, self.nl
        for name in self.prec_names:
            if self.utils.field_will_be_written(name):
                self.owner.buffer("diag", name)[:nc] = self.acc[name]
        for name in self.tend_names:
            if self.utils.field_will_be_written(name):
                self.owner.buffer("diag", name)[:nl, :nc] = self.tend[name]

    # -- checkpoint continuation -----------------------------------------------------------------------
    def _targets(self):
        out = {}
        if self.prec_names:
            out.update(self.acc)
        if self.tend_names:
            out.update(self.tend)
        return out

    def state(self):
        cp = self.cp
        return {STATE_PREFIX + name: cp.asnumpy(value) for name, value in self._targets().items()}

    def resume(self, state):
        targets = self._targets()
        for key, value in state.items():
            if key.startswith(STATE_PREFIX) and key[len(STATE_PREFIX):] in targets:
                target = targets[key[len(STATE_PREFIX):]]
                target[...] = self.cp.asarray(value).reshape(target.shape)
