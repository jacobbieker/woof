"""Scratch allocation: ``xp.empty`` for a buffer whose every read word is written first.

On CUDA (and NumPy) this is ``xp.empty``.  On the Metal facade ``empty`` is
``zeros`` (MLX has no uninitialized allocation), and the fill is a GPU pass
over the whole buffer; scratch skips it (woof/backend/facade/_scratch.py).
Use it only where CUDA's garbage would give the same words: a kernel's private
arena, never a buffer read before it is written.

Both switches are read once, at the first scratch allocation; set them before.
``WOOF_METAL_SCRATCH_UNINIT=0`` (or off/false/no) restores the filled
allocation (A/B).  ``WOOF_SCRATCH_POISON`` fills scratch with one byte on every
backend: ``1``/``on`` is 0xFF (a NaN in every float word), two hex digits name
another byte (``a5``).  Runs poisoned with two bytes whose words match the
ordinary run show every read word is written first.
"""

from __future__ import annotations

import functools
import os
import sys

_OFF = frozenset({"", "0", "off", "false", "no"})


@functools.lru_cache(maxsize=1)
def _uninit_enabled() -> bool:
    return os.environ.get("WOOF_METAL_SCRATCH_UNINIT", "1").strip().lower() not in _OFF - {""}


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


def scratch_empty(xp, shape, dtype):
    """``xp.empty(shape, dtype)`` (C order) with no fill on the Metal facade."""
    poison = _poison()
    if poison is not None:
        out = xp.empty(shape, dtype=dtype)
        out.reshape(-1).view(xp.uint8)[...] = poison
        return out
    if _uninit_enabled() and xp is sys.modules.get("cupy"):
        from woof import backend
        if backend.active() == "mlx":
            from woof.backend.facade._scratch import uninitialized
            return uninitialized(shape, dtype)
    return xp.empty(shape, dtype=dtype)


def scratch_like(xp, a):
    """``scratch_empty`` of ``a``'s shape and dtype.

    C order, which is ``empty_like``'s layout for a C-contiguous ``a``; any
    other ``a`` takes ``xp.empty_like`` (its layout), poisoned the same way.
    """
    if not a.flags.c_contiguous:
        out = xp.empty_like(a)
        poison = _poison()
        base = out if out.flags.c_contiguous else out.T
        if poison is not None and base.flags.c_contiguous:
            base.reshape(-1).view(xp.uint8)[...] = poison
        return out
    return scratch_empty(xp, a.shape, a.dtype)
