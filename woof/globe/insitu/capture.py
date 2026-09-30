"""Per-component physics tendency capture: device reductions only.

At the seam where a physics suite runs its components in sequence (the
reference suite's process functions in ``ReferencePhysics.step``, the
native runtime's scheme calls in ``NativePhysicsRuntime.run``) each
component marks the live prognostic fields; the capture records, per
component per call, the global-mean and max-abs change of theta, qv, u
and v since the previous mark.  The results stay on the device as one
small vector per mark and cross to the host with the ledger's batched
flush.

Marks copy the four fields: the reference suite's convective adjustment
and every native kernel update their arrays IN PLACE, so a capture that
held references to "before" would read exactly zero change and report a
component as inert - the silent-instrument failure the 2026-08-31 night
taught (five instruments gave confident wrong answers).  The copy is the
price of a measurement that cannot lie that way.

"Global mean" here is the area-weighted mean over the grid averaged over
levels with equal weight (no dp weighting: the marks carry no dp on the
reference path and the native batch is bottom-to-top float32; level
ordering does not matter to either reduction).

THE PHYSICS RUNS A LATITUDE BAND AT A TIME (dynamics.apply_physics), so a
mark arrives with ``band=(first, last)`` and carries that band's rows of
the four fields.  The record of a call is still ONE vector per component:
the per-row stage of each mean (the zonal mean, then the level mean, both
row-local) is written into a resident ``(nlat,)`` buffer as the bands
arrive, and the area-weighted sum over latitude runs once when the suite
closes the call (:meth:`ComponentCapture.close_call`), in grid order --
the same operand the whole-grid mark reduces, so the number does not
depend on the band count.  The max-abs stage is exactly associative and
folds as it goes.  A whole-grid mark (``band=None``, a hand-driven suite)
records as it always did, immediately.
"""
from __future__ import annotations

import time

import numpy as np

CAPTURE_FIELDS = ("theta", "qv", "u", "v")
CAPTURE_STATS = ("mean_change", "max_abs_change")
CAPTURE_NAMES = tuple(
    f"{field}_{stat}" for field in CAPTURE_FIELDS for stat in CAPTURE_STATS
)
START_MARK = "start"


class ComponentCapture:
    def __init__(self, quadrature_weights):
        self._weights_host = np.asarray(quadrature_weights, dtype=np.float64)
        self._weights_by_module: dict[int, object] = {}
        self._nlat = int(self._weights_host.shape[0])
        self.active = True
        self.records: list[tuple[int, str, object]] = []
        self._previous: dict[str, object] | None = None
        self._previous_band: tuple[int, int] | None = None
        self._call = -1
        # The open banded call's per-component partials, in the order the
        # components first marked: name -> [rows covered, {field: (nlat,)
        # meridional buffer}, {field: max-abs so far}].
        self._pending: dict[str, list] = {}
        self._pending_order: list[str] = []
        self._open = False
        # Host wall time spent inside marks (on cupy: launch time only).
        self.wall_s = 0.0

    def begin_step(self) -> None:
        """Reset the call counter so marks inside the next step number
        the physics calls from zero (0 = first Strang half, 1 = second)."""
        self._call = -1
        self._previous = None
        self._previous_band = None
        self._pending = {}
        self._pending_order = []
        self._open = False

    def _weights(self, xp):
        key = id(xp)
        weights = self._weights_by_module.get(key)
        if weights is None:
            weights = xp.asarray(self._weights_host, dtype=xp.float64)
            self._weights_by_module[key] = weights
        return weights

    def _rows(self, band, fields) -> tuple[int, int]:
        rows = int(fields[CAPTURE_FIELDS[0]].shape[-2])
        if band is None:
            if rows != self._nlat:
                raise ValueError(
                    f"a whole-grid component mark carries {rows} rows against a "
                    f"grid of {self._nlat}; a band mark names its rows with band="
                )
            return (0, self._nlat)
        first, last = int(band[0]), int(band[1])
        if first < 0 or last > self._nlat or last - first != rows:
            raise ValueError(
                f"component mark band {band!r} does not cover the {rows} rows "
                f"of the fields it arrived with (grid of {self._nlat} rows)"
            )
        return first, last

    def _vector(self, xp, buffers, maxima):
        weights = self._weights(xp)
        stats = []
        for key in CAPTURE_FIELDS:
            stats.append(0.5 * xp.sum(buffers[key] * weights))
            stats.append(maxima[key])
        return xp.stack(stats).astype(xp.float64)

    def mark(self, name: str, fields: dict[str, object], xp, *, band=None) -> None:
        """Record the change since the previous mark; ``start`` opens a
        call (with ``band``, one band of the open call)."""
        if not self.active:
            return
        started = time.perf_counter()
        first, last = self._rows(band, fields)
        if name == START_MARK:
            if band is None or not self._open:
                # A new call: the whole grid's, or a banded call's first
                # band.  Later bands of the same call start without
                # advancing the counter.
                self._call += 1
                self._open = band is not None
            self._previous = {key: fields[key].copy() for key in CAPTURE_FIELDS}
            self._previous_band = (first, last)
            self.wall_s += time.perf_counter() - started
            return
        if self._previous is None or self._previous_band != (first, last):
            raise ValueError(
                f"component mark {name!r} for rows {(first, last)} arrived "
                f"before a {START_MARK!r} mark for those rows; the capture "
                "cannot difference against nothing"
            )
        if band is None:
            buffers = {key: xp.empty(self._nlat, dtype=xp.float64) for key in CAPTURE_FIELDS}
            maxima: dict[str, object] = {}
        else:
            entry = self._pending.get(name)
            if entry is None:
                entry = [
                    0,
                    {key: xp.empty(self._nlat, dtype=xp.float64) for key in CAPTURE_FIELDS},
                    {key: None for key in CAPTURE_FIELDS},
                ]
                self._pending[name] = entry
                self._pending_order.append(name)
            buffers, maxima = entry[1], entry[2]
        for key in CAPTURE_FIELDS:
            current = fields[key]
            delta = current - self._previous[key]
            zonal = xp.mean(delta, axis=-1, dtype=xp.float64)
            # (nlev, rows) -> level-equal mean -> these rows of the (nlat,)
            # buffer the area weighting reduces once, in grid order.
            buffers[key][first:last] = xp.mean(zonal, axis=0)
            band_max = xp.max(xp.abs(delta))
            held = maxima.get(key)
            maxima[key] = band_max if held is None else xp.maximum(held, band_max)
            self._previous[key] = current.copy()
        if band is None:
            self.records.append((self._call, name, self._vector(xp, buffers, maxima)))
        else:
            entry[0] += last - first
        self.wall_s += time.perf_counter() - started

    def close_call(self, xp) -> None:
        """The banded call's marks have all arrived: reduce every
        component's buffers once and record the vectors, in the order the
        components first marked.  A component whose marks did not cover
        the globe is refused by name rather than recorded on the rows it
        happened to see."""
        if not self.active or not self._open:
            return
        started = time.perf_counter()
        for name in self._pending_order:
            covered, buffers, maxima = self._pending[name]
            if covered != self._nlat:
                raise ValueError(
                    f"component {name!r} marked {covered} of {self._nlat} rows "
                    "in this physics call; a component that marks on some bands "
                    "and not others cannot be reported as the call's"
                )
            self.records.append((self._call, name, self._vector(xp, buffers, maxima)))
        self._pending = {}
        self._pending_order = []
        self._open = False
        self._previous = None
        self._previous_band = None
        self.wall_s += time.perf_counter() - started

    def drain(self) -> list[tuple[int, str, object]]:
        records = self.records
        self.records = []
        return records


def attach_capture(physics, capture: ComponentCapture) -> list[str]:
    """Install the capture on every physics object that declares an
    ``observer`` slot (the reference suite; the native column suite behind
    the bridge).  Returns the class names it attached to."""
    attached: list[str] = []
    targets = [physics]
    adapter = getattr(physics, "adapter", None)
    if adapter is not None:
        targets.append(adapter)
    for target in targets:
        if target is not None and hasattr(type(target), "observer"):
            target.observer = capture
            attached.append(type(target).__name__)
    return attached


__all__ = [
    "CAPTURE_FIELDS",
    "CAPTURE_NAMES",
    "CAPTURE_STATS",
    "ComponentCapture",
    "START_MARK",
    "attach_capture",
]
