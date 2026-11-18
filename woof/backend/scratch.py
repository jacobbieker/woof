"""Scratch allocation: ``xp.empty`` for a buffer whose every read word is written first.

On CUDA (and NumPy) this is ``xp.empty``.  On the Metal facade ``empty`` is
``zeros`` (MLX has no uninitialized allocation), and the fill is a GPU pass
over the whole buffer; scratch skips it (woof/backend/facade/_scratch.py).
Use it only where CUDA's garbage would give the same words: a kernel's private
arena, never a buffer read before it is written.
``WOOF_METAL_SCRATCH_UNINIT=0`` restores the filled allocation (A/B), and
``WOOF_SCRATCH_POISON=1`` fills scratch with 0xA5 bytes on every backend: a
run whose words match the ordinary run proves every read word is written first.
"""

from __future__ import annotations

import functools
import os
import sys


@functools.lru_cache(maxsize=1)
def _uninit_enabled() -> bool:
    return os.environ.get("WOOF_METAL_SCRATCH_UNINIT", "1") != "0"


@functools.lru_cache(maxsize=1)
def _poison() -> bool:
    return os.environ.get("WOOF_SCRATCH_POISON", "0") not in ("", "0")


def scratch_empty(xp, shape, dtype):
    """``xp.empty(shape, dtype)`` (C order) with no fill on the Metal facade."""
    if _poison():
        out = xp.empty(shape, dtype=dtype)
        out.reshape(-1).view(xp.uint8)[...] = 0xA5
        return out
    if _uninit_enabled() and xp is sys.modules.get("cupy"):
        from woof import backend
        if backend.active() == "mlx":
            from woof.backend.facade._scratch import uninitialized
            return uninitialized(shape, dtype)
    return xp.empty(shape, dtype=dtype)
