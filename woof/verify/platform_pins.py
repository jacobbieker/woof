"""Measured-word pins per kernel backend.

A pin file records the words one compile platform produced, and their
distance from compiled WRF.  CUDA's pins carry NVRTC's multiply-add
contraction (``--fmad=true`` is the default, and fast numerics drops
``--fmad=false``).  The Metal translation never contracts, so on the MLX
backend the same measurement gives CUDA's ``--fmad=false`` words; those are
recorded beside the CUDA pin as ``<stem>-metal<suffix>``.

There is no fallback that loosens a gate: a backend without its own file is
held to the CUDA pin, and fails on any drift.
"""
from __future__ import annotations

from pathlib import Path

METAL_SUFFIX = "-metal"


def backend_variant() -> str | None:
    """The pin variant of the active kernel backend (None for CUDA)."""
    from woof import backend
    return "metal" if backend.active() == "mlx" else None


def variant_path(path: Path, variant: str) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}-{variant}{path.suffix}")


def pin_path(path: Path) -> Path:
    """``path``, or its sibling for the active backend when one is recorded."""
    variant = backend_variant()
    if variant is not None:
        own = variant_path(path, variant)
        if own.exists():
            return own
    return Path(path)
