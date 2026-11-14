"""WRF-HAILCAST (hailcast_opt = 1, haildt) on the MPAS-A v8.4.2 suite route.

A WOOF-only output; MPAS-A v8.4.2 has none.  The WOOF namelist group ``&diagnostics`` takes::

    config_hailcast_opt      logical  WRF hailcast_opt (default false)
    config_haildt            real     WRF haildt, s between HAILCAST calls (WRF's default 0: every step; 0..60)
    config_hailcast_version  string   the stone model: 'wrf_461' (WRF's public module_diag_hailcast.F, the default
                                      here) or 'wrf_39_noaa' (operational HRRR v4.1.21), woof.core.hail.HAILCAST_KERNELS

The fields are ``diag`` fields [nCells] in the ``diagnostics`` stream (package ``hailcast``) with the WRF route's
names, units (mm) and text (woof.core.hail.HISTORY)::

    hailcast_diam_max hailcast_diam_mean hailcast_diam_std   the interval's surviving-stone statistics
    hailcast_dhail1 .. hailcast_dhail5                         each embryo's interval maximum diameter
    hailcast_runmax                                            the run maximum (a WOOF field; never reset)

With the option off nothing here is allocated, constructed or compiled, and the Registry, the default streams and the
driver receipt are the ones a run without this module has.

Nothing is re-derived.  The columns run through the WRF route's own kernels (woof/core/kernels/hail.cu
``hail_prepare`` and ``hailcast_columns`` / ``hailcast_columns_wrf39``, translated from WRF's
module_diag_hailcast.F by tools/hail_wrf461), with one mesh cell as one WRF column (``ny = 1``, ``nx = nCells``),
handed over as the nwp/AFWA modules hand one over: ``theta`` is the diag ``theta``, ``p`` is
``pressure_base + pressure_p``, ``alt = 1 / (rho_zz * zz)``, ``PHB = 9.81 * zgrid``, ``PH = 0``, ``w`` the state's
``w`` at the layer faces, ``ht = zgrid(1)``, the hydrometeors the run's scalars (zeros where the scheme carries
none).

The one mesh-specific step is the updraft duration.  WRF accumulates, per column, the time an updraft of at least
10 m/s has persisted (``HAILCAST_WDUR``) from the previous step's mask and duration over the column's 3x3
neighbourhood; HAILCAST runs only in columns whose duration exceeds 900 s.  On the mesh the neighbourhood is the cell
and its ``cellsOnCell`` neighbours (as afwa_diag_v842's tornado duration), formed with exact device maxima and the
one float32 add WRF does (``hail_duration``'s arithmetic, word for word); WRF's domain-edge rows have no mesh
counterpart.  A multi-rank mesh is refused (the neighbours' previous words would need an exchange).  The WRF route's
kernel cannot be used as is: with ``ny = 1`` and a non-periodic y it zeroes every duration, which is why HAILCAST was
inert on the WOOF column route (R1).

Timing.  As on the WRF route (woof.core.hail.update): the mask and duration advance after every step, HAILCAST runs on
the steps its cadence selects (every step for ``haildt = 0``, the first step, and every step that crosses a multiple
of ``haildt``), on the step's end state.  The interval fields restart after each frame that writes one of them (the
WRF route's reset after a durable history frame); the run maximum, the mask and the duration ride the checkpoint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from woof.core.hail import EMBRYOS, HAILCAST_KERNELS, HISTORY

from .utils import Direction

GROUP = "diagnostics"
PACKAGE = "hailcast"
OPTION, DT_OPTION, VERSION_OPTION = "config_hailcast_opt", "config_haildt", "config_hailcast_version"
OPTIONS = (OPTION, DT_OPTION, VERSION_OPTION)
NAMES = ("hailcast_diam_max", "hailcast_diam_mean", "hailcast_diam_std", *EMBRYOS, "hailcast_runmax")
#: reset after a frame that writes them (woof.core.hail.WINDOW's HAILCAST rows)
WINDOW = tuple(n for n in NAMES if n != "hailcast_runmax")
STATE = ("hailcast_wup_mask", "hailcast_wdur")
SPECIES = ("qv", "qi", "qc", "qr", "qs", "qg")
STATE_PREFIX = "hailcast:"
DEFAULT_VERSION = "wrf_461"
_G = np.float32(9.81)
#: the updraft that marks a column (hail.cu hail_prepare) and the duration HAILCAST needs (hailcast_column)
UPDRAFT_MS, DURATION_S = 10.0, 900.0


@dataclass(frozen=True)
class HailcastConfig:
    haildt: float = 0.0
    version: str = DEFAULT_VERSION

    def receipt(self) -> dict:
        return {"haildt": self.haildt, "version": self.version}


FIELDS = {name: {"type": "real", "dimensions": ["nCells", "Time"],
                 "attrs": {"units": HISTORY[name][1], "long_name": HISTORY[name][2], "wrf_name": HISTORY[name][0]},
                 "gates": [PACKAGE]} for name in NAMES}


def field_definition(name: str) -> dict[str, Any]:
    row = FIELDS[name]
    return {"type": row["type"], "dimensions": list(row["dimensions"]),
            "attrs": dict(row["attrs"]), "gates": list(row["gates"])}


def _flag(value: Any) -> bool:
    if isinstance(value, str) and value.strip().strip(".").casefold() in ("true", "t", "false", "f"):
        return value.strip().strip(".").casefold() in ("true", "t")
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):      # WRF's integer hailcast_opt
        return bool(value)
    raise ValueError(f"{OPTION} is .true./.false. (or WRF's 0/1), not {value!r}")


def resolve(groups: Mapping[str, Mapping[str, Any]]) -> HailcastConfig | None:
    """The options from the namelist's &diagnostics group (None when HAILCAST is off)."""
    supplied = {str(k).casefold(): v for k, v in groups.get(GROUP, {}).items()}
    on = _flag(supplied.get(OPTION, False))
    if not on:
        extra = [k for k in (DT_OPTION, VERSION_OPTION) if k in supplied]
        if extra:
            raise ValueError(f"&{GROUP} {extra[0]} is set while {OPTION} is off; it would be read by nothing")
        return None
    haildt = supplied.get(DT_OPTION, 0.0)
    if isinstance(haildt, bool) or not isinstance(haildt, (int, float)) or not math.isfinite(haildt) \
            or not 0 <= haildt <= 60:
        raise ValueError(f"&{GROUP} {DT_OPTION} must be between 0 (every step) and 60 seconds, not {haildt!r}")
    version = str(supplied.get(VERSION_OPTION, DEFAULT_VERSION)).strip().strip("'\"")
    if version not in HAILCAST_KERNELS:
        raise ValueError(f"&{GROUP} {VERSION_OPTION} is one of {', '.join(HAILCAST_KERNELS)}, not {version!r}")
    return HailcastConfig(float(haildt), version)


def packages(config: Any) -> set[str]:
    return {PACKAGE} if getattr(config, "hailcast_diagnostics", None) is not None else set()


def extend_registry(root, active: set[str]) -> None:
    """Add the fields to the ``diag`` struct and the ``diagnostics`` stream of a parsed Registry tree."""
    import xml.etree.ElementTree as ET
    if PACKAGE not in active:
        return
    struct = root.find("./var_struct[@name='diag']")
    stream = root.find("./streams/stream[@name='diagnostics']")
    for name, row in FIELDS.items():
        if struct.find(f"./var[@name='{name}']") is None:
            ET.SubElement(struct, "var", {
                "name": name, "type": "real", "dimensions": " ".join(row["dimensions"]),
                "units": row["attrs"]["units"], "description": row["attrs"]["long_name"],
                "default_value": "0.0", "packages": PACKAGE})
        if stream is not None and stream.find(f"./var[@name='{name}']") is None:
            ET.SubElement(stream, "var", {"name": name, "packages": PACKAGE})


class HailcastModule:
    """Advance the updraft duration after every step, run HAILCAST on the due steps, publish for a due frame."""

    def __init__(self, owner, config: HailcastConfig):
        self.owner, self.config = owner, config
        self._step = None

    def setup(self, utils):
        cp = self.cp = self.owner.cp
        dims = self.owner.session.layout.dimensions
        self.nc, self.nl = int(dims["nCellsSolve"]), int(dims["nVertLevels"])
        self.utils = utils
        if getattr(self.owner, "halo", None) is not None:
            raise ValueError(f"&{GROUP} {OPTION} is not available on a multi-rank mesh: the updraft duration reads "
                             "the neighbours' previous words, which are not exchanged")
        if self.nl > 256:
            raise ValueError("HAILCAST column storage supports at most 256 mass levels")
        self.dt = float(np.float32(self.owner.dycore.config.config_dt))
        self.carried = {n for n in NAMES if utils.stream_inclusion_count(n, Direction.OUTPUT) > 0}
        self.w = {name: cp.zeros(self.nc, dtype=cp.float32) for name in (*NAMES, *STATE)}
        self.zero = cp.zeros((self.nl, self.nc), dtype=cp.float32)
        self._reset = False
        self.calls = 0

    # -- inputs ----------------------------------------------------------------------------------------
    def _get(self, pool, name):
        return self.owner.authority(pool, name)

    def _levels(self, pool, name, levels):
        return self.cp.ascontiguousarray(self._get(pool, name)[:levels, :self.nc], dtype=self.cp.float32)

    def _scalar(self, name):
        indices = self.owner.session.layout.indices
        key = ("state", "index_" + name)
        if key not in indices or int(indices[key]) < 1:
            return self.zero
        row = int(indices[key]) - 1
        return self.cp.ascontiguousarray(self._get("state", "scalars")[row, :self.nl, :self.nc],
                                         dtype=self.cp.float32)

    def neighbours(self):
        """Zero-based neighbour indices (maxEdges, nCells); a missing neighbour (MPAS's nCells + 1, or beyond the
        cell's nEdgesOnCell) points at the zero word appended at index nCells."""
        if getattr(self, "_neighbours", None) is None:
            cp, nc = self.cp, self.nc
            cells = cp.asnumpy(self._get("mesh", "cellsOnCell")[:, :nc]).astype(np.int64)
            edges = cp.asnumpy(self._get("mesh", "nEdgesOnCell")[:nc]).astype(np.int64)
            slot = np.arange(cells.shape[0])[:, None]
            valid = (slot < edges[None, :]) & (cells >= 1) & (cells <= nc)
            self._neighbours = cp.asarray(np.where(valid, cells - 1, nc).astype(np.int32))
        return self._neighbours

    def _prepare(self):
        """hail_prepare on the mesh columns: temperature, mid-layer height, density and the >= 10 m/s mask."""
        from woof.core.kernels import get_kernel
        cp, nc, nl = self.cp, self.nc, self.nl
        f32, i32 = np.float32, np.int32
        zgrid = self._levels("mesh", "zgrid", nl + 1)
        phb = cp.ascontiguousarray(cp.multiply(zgrid, _G))
        php = cp.zeros_like(phb)
        alt = cp.divide(f32(1), cp.multiply(self._levels("state", "rho_zz", nl), self._levels("mesh", "zz", nl)))
        p = cp.add(self._levels("diag", "pressure_base", nl), self._levels("diag", "pressure_p", nl))
        w = self._levels("state", "w", nl + 1)
        qv = self._scalar("qv")
        t, z, rho = (cp.zeros((nl, nc), dtype=cp.float32) for _ in range(3))
        mask = cp.zeros(nc, dtype=cp.float32)
        get_kernel("hail", "hail_prepare")(((nc + 127) // 128,), (128,),
            (self._levels("diag", "theta", nl), p, alt, qv, phb, php, w, t, z, rho, mask,
             i32(nc), i32(1), i32(nl), i32(0)))
        return dict(t=t, z=z, rho=rho, p=p, w=w, qv=qv, ht=cp.ascontiguousarray(zgrid[0]), mask=mask)

    def duration(self, mask):
        """hail_duration over the cell and its cellsOnCell neighbours: the updraft time, else 0."""
        cp, nc = self.cp, self.nc
        index = self.neighbours()
        pad = lambda a: cp.concatenate((a, cp.zeros(1, dtype=cp.float32)))
        old_mask, old_dur = pad(self.w["hailcast_wup_mask"]), pad(self.w["hailcast_wdur"])
        seen = cp.maximum(mask, cp.maximum(self.w["hailcast_wup_mask"], cp.max(old_mask[index], axis=0)))
        longest = cp.maximum(self.w["hailcast_wdur"], cp.max(old_dur[index], axis=0))
        longest = cp.maximum(longest, np.float32(0))
        return cp.where(seen == np.float32(1), cp.add(longest, np.float32(self.dt)), np.float32(0)).astype(cp.float32)

    # -- the per-step fold ------------------------------------------------------------------------------
    def begin(self, step, when):
        """The step the next :meth:`update` folds (steps <= 0 fold nothing), on the straight run's clock.

        ``step`` is the loop's local step; a checkpoint continuation's owner carries the straight run's steps before
        the segment (``step_offset``), as the physics clock does, so the haildt cadence and its first-step call
        follow the straight run's schedule, not a restart at the checkpoint.
        """
        self._step = int(step) + int(getattr(self.owner, "step_offset", 0)) if int(step) > 0 else None

    def due(self) -> bool:
        """woof.core.hail.update's cadence on the step that :meth:`begin` named (the straight run's elapsed time)."""
        interval = self.config.haildt
        start = (self._step - 1) * self.dt
        end = start + self.dt
        return interval <= 0 or start == 0 or int(end / interval) > int(start / interval)

    def update(self):
        if self._step is None or not self.nc:
            return
        from woof.core.kernels import get_kernel
        if self._reset:
            for name in WINDOW:
                self.w[name].fill(0)
            self._reset = False
        col = self._prepare()
        dur = self.duration(col["mask"])
        self.w["hailcast_wup_mask"][...] = col["mask"]
        self.w["hailcast_wdur"][...] = dur
        if not self.due():
            return
        i32, nc, nl, w = np.int32, self.nc, self.nl, self.w
        get_kernel("hail", HAILCAST_KERNELS[self.config.version])(((nc + 31) // 32,), (32,),
            (col["t"], col["z"], col["p"], col["rho"], col["qv"], *(self._scalar(n) for n in SPECIES[1:]),
             col["w"], col["ht"], dur, *(w[n] for n in EMBRYOS), w["hailcast_diam_max"], w["hailcast_diam_mean"],
             w["hailcast_diam_std"], w["hailcast_runmax"], i32(nc), i32(1), i32(nl)))
        self.calls += 1

    # -- publication and reset -------------------------------------------------------------------------
    def compute(self):
        for name in self.carried:
            if self.utils.field_will_be_written(name):
                self.owner.buffer("diag", name)[:self.nc] = self.w[name]

    def reset(self):
        """After a frame is written: the interval fields restart on the next step if a due stream carried one."""
        if any(self.utils.field_will_be_written(n) for n in WINDOW if n in self.carried):
            self._reset = True

    # -- checkpoint continuation -----------------------------------------------------------------------
    def state(self):
        state = {STATE_PREFIX + n: self.cp.asnumpy(v) for n, v in self.w.items()}
        state[STATE_PREFIX + "reset"] = np.array([int(self._reset)], dtype=np.int32)
        return state

    def resume(self, state):
        for key, value in state.items():
            if key.startswith(STATE_PREFIX) and key[len(STATE_PREFIX):] in self.w:
                target = self.w[key[len(STATE_PREFIX):]]
                target[...] = self.cp.asarray(value).reshape(target.shape)
        if STATE_PREFIX + "reset" in state:
            self._reset = bool(np.asarray(state[STATE_PREFIX + "reset"]).reshape(-1)[0])
