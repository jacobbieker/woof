"""Opt-in frozen McICA seeds for gradient and sensitivity runs.

RRTMG's McICA cloud generator seeds each column's KISS stream from the
fractional digits of its bottom four layer pressures,
``int((p - int(p)) * 1e9)`` (MPAS module_ra_rrtmg_lw.F:2432-2440 and the SW
equivalent; WRF's generate_stochastic_clouds).  A one-ulp change in pressure
therefore reshuffles every sub-grid cloud of the column, which puts about
+-5 J of noise on every finite-difference or ensemble-sensitivity leg of
``woof grad``.

``mcica_seed_mode``:

``pressure`` (default)
    Upstream seeding.  Nothing in this module runs; every default path is
    byte-identical to a build without it.
``cell``
    The four seeds are a fixed hash of (column id, radiation call time,
    stream), independent of the model state.  LW and SW use different fixed
    streams.  Each seed is in [1, 999999998], inside the range the
    pressure rule produces, so the xorshift word is nonzero (on the WRF
    route the seeds pass through a stand-in pressure, which can truncate a
    word of 1 to 0: about 1 in 1e9).  The column id is the column's index in
    the radiation call's arrays, so a decomposed run needs the same ordering.

Select it with ``--mcica-seed-mode cell`` on ``woof hex`` forecasts and
``woof go``, or ``WOOF_MCICA_SEED_MODE=cell`` in the environment.  It is
for sensitivity legs only: it is not WRF or MPAS physics.
"""
from __future__ import annotations

import os

import numpy as np

MODES = ("pressure", "cell")
ENV = "WOOF_MCICA_SEED_MODE"
LW_STREAM = 0x4C57   # "LW"
SW_STREAM = 0x5357   # "SW"
RRTMGK_STREAM = 0x524B   # "RK": one McICA call serves both RRTMG-K spectra
_SEED_MOD = 999999998

_state = {"mode": None, "time": 0, "base": 0, "ids": None}


def mode() -> str:
    """The active mode: set_mode(), else the environment, else "pressure"."""
    value = _state["mode"]
    if value is None:
        value = os.environ.get(ENV, "pressure").strip().lower() or "pressure"
    if value not in MODES:
        raise ValueError(f"{ENV}/mcica_seed_mode must be one of {MODES}, got {value!r}")
    return value


def set_mode(value) -> None:
    """Select the mode for this process (None returns to the environment)."""
    if value is not None and value not in MODES:
        raise ValueError(f"mcica_seed_mode must be one of {MODES}, got {value!r}")
    _state["mode"] = value


def active() -> bool:
    return mode() == "cell"


def set_context(*, time_seconds=None, base=None, ids=None) -> None:
    """Record the radiation call time and the column ids of the next call.

    ``ids`` (array) wins over ``base`` (column i has id base + i).  Both
    reset to base 0 when a new time is set without them.  A no-op in
    ``pressure`` mode.
    """
    if not active():
        return
    if time_seconds is not None:
        _state["time"] = int(round(float(time_seconds)))
        _state["base"], _state["ids"] = 0, None
    if base is not None:
        _state["base"], _state["ids"] = int(base), None
    if ids is not None:
        _state["ids"] = ids


def _mix(x):
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def seed_words(ids, stream: int, time_seconds=None) -> np.ndarray:
    """(n, 4) uint32 seeds, a pure function of (id, time, stream, word)."""
    ids = np.asarray(ids).astype(np.uint64).reshape(-1)
    t = _state["time"] if time_seconds is None else int(round(float(time_seconds)))
    with np.errstate(over="ignore"):
        tarr = np.array([t], dtype=np.int64).view(np.uint64)
        key = _mix(_mix(tarr ^ np.uint64(stream)) + np.uint64(stream))[0]
        word = np.arange(4, dtype=np.uint64)
        h = _mix(key + (ids[:, None] * np.uint64(4) + word[None, :])
                 * np.uint64(0x9E3779B97F4A7C15))
    return ((h >> np.uint64(24)) % np.uint64(_SEED_MOD) + np.uint64(1)).astype(np.uint32)


def context_ids(n: int) -> np.ndarray:
    ids = _state["ids"]
    if ids is not None:
        if hasattr(ids, "get"):
            ids = ids.get()
        ids = np.asarray(ids).reshape(-1)
        if ids.size != n:
            raise ValueError(f"mcica seed ids cover {ids.size} columns, call has {n}")
        return ids
    return _state["base"] + np.arange(n)


def seed_play(play, stream: int):
    """Layer pressures for the McICA generator alone, giving cell-mode seeds.

    The generators read ``play`` (hPa) only to form their seeds, so cell
    mode hands them a stand-in whose bottom four layers hold the seed
    fractions (p*100 in [0, 1)): layer 0 in [0.5, 1) and layer 1 in
    [0, 0.5) keep the generator's pressure-order guard satisfied.  Other
    layers are returned unchanged.  In ``pressure`` mode ``play`` itself is
    returned.  Works on numpy and cupy arrays of shape (ncol, nlay).
    """
    if not active():
        return play
    n, nl = play.shape
    u = seed_words(context_ids(n), stream).astype(np.float64) * 0.9999e-9
    u[:, 0] = 0.5 + 0.5 * u[:, 0]
    u[:, 1] = 0.5 * u[:, 1]
    syn = (u.astype(np.float32) / np.float32(100.0)).astype(np.float32)
    out = play.copy()
    if hasattr(out, "get"):
        import cupy as cp
        out[:, :4] = cp.asarray(syn)
    else:
        out[:, :4] = syn
    return out
