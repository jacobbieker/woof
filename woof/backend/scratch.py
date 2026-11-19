"""Scratch allocation: ``xp.empty`` for a buffer whose every read word is written first.

On CUDA (and NumPy) this is ``xp.empty``.  On the Metal facade ``empty`` is
``zeros`` (MLX has no uninitialized allocation), and the fill is a GPU pass
over the whole buffer; scratch skips it (woof/backend/facade/_scratch.py).
Use it only where CUDA's garbage would give the same words: a kernel's private
arena, never a buffer read before it is written.

Both switches are read once, at the first allocation that asks (a scratch
allocation, or with ``all`` a facade ``cupy.empty``); set them before.
``WOOF_METAL_SCRATCH_UNINIT=0`` (or off/false/no) restores the filled
allocation (A/B).  ``WOOF_METAL_SCRATCH_UNINIT=all`` (opt-in) also drops the
fill from the facade's own C-ordered ``cupy.empty``/``empty_like`` (``zeros``
and F order keep it), and poisons every facade ``empty`` when
``WOOF_SCRATCH_POISON`` is set.  ``WOOF_SCRATCH_POISON`` fills scratch with one
byte on every backend: ``1``/``on`` is 0xFF (a NaN in every float word), two
hex digits name another byte (``a5``).  Runs poisoned with two bytes whose
words match the ordinary run show every read word is written first (bool
buffers may hold the byte normalized to 0/1, so they are not audited).
"""

from __future__ import annotations

import functools
import os
import sys

import numpy as np

_OFF = frozenset({"", "0", "off", "false", "no"})


@functools.lru_cache(maxsize=1)
def _uninit_mode() -> str:
    """WOOF_METAL_SCRATCH_UNINIT: ``"off"``, ``"scratch"`` (the default) or ``"all"``."""
    value = os.environ.get("WOOF_METAL_SCRATCH_UNINIT", "1").strip().lower()
    if value == "all":
        return "all"
    return "off" if value in _OFF - {""} else "scratch"


def _uninit_enabled() -> bool:
    return _uninit_mode() != "off"


@functools.lru_cache(maxsize=1)
def facade_empty_fill():
    """How the Metal facade's own ``empty``/``empty_like`` fill a new buffer:
    None (MLX's zero fill, the default), ``"none"`` (uninitialized,
    ``WOOF_METAL_SCRATCH_UNINIT=all``) or a poison byte (``all`` with
    ``WOOF_SCRATCH_POISON``).  Cached: it is asked on every allocation."""
    if _uninit_mode() != "all":
        return None
    poison = _poison()
    return "none" if poison is None else poison


def poison_fill(out, byte: int):
    """Write ``byte`` over every byte of ``out`` (C- or F-contiguous)."""
    base = out if out.flags.c_contiguous else out.T
    if out.size and base.flags.c_contiguous:
        base.reshape(-1).view(np.uint8)[...] = byte
    return out


def _reset_caches() -> None:
    """For tests that change the switches."""
    _uninit_mode.cache_clear()
    facade_empty_fill.cache_clear()
    _poison.cache_clear()


@functools.lru_cache(maxsize=1)
def _poison() -> int | None:
    value = os.environ.get("WOOF_SCRATCH_POISON", "0").strip().lower()
    if value in _OFF:
        return None
    if value in ("1", "on", "true", "yes"):
        return 0xFF
    try:
        byte = int(value, 16)
    except ValueError:
        byte = -1
    if not 0 <= byte <= 0xFF:
        raise ValueError(f"WOOF_SCRATCH_POISON must be 0/1 or one byte in hex (a5), got {value!r}")
    return byte


def _facade(xp) -> bool:
    if xp is not sys.modules.get("cupy"):
        return False
    from woof import backend
    return backend.active() == "mlx"


def scratch_empty(xp, shape, dtype):
    """``xp.empty(shape, dtype)`` (C order) with no fill on the Metal facade."""
    poison = _poison()
    if _facade(xp) and (poison is not None or _uninit_enabled()):
        from woof.backend.facade._scratch import uninitialized
        out = uninitialized(shape, dtype)
    else:
        out = xp.empty(shape, dtype=dtype)
    return out if poison is None else poison_fill(out, poison)


def scratch_like(xp, a):
    """``scratch_empty`` of ``a``'s shape and dtype.

    C order, which is ``empty_like``'s layout for a C-contiguous ``a``; any
    other ``a`` takes ``xp.empty_like`` (its layout), poisoned the same way.
    """
    if not a.flags.c_contiguous:
        out = xp.empty_like(a)
        poison = _poison()
        return out if poison is None else poison_fill(out, poison)
    return scratch_empty(xp, a.shape, a.dtype)
