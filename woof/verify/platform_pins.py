"""Measured-word pins per kernel backend.

A pin file records the words one compile platform produced, and their
distance from compiled WRF.  CUDA's pins carry NVRTC's multiply-add
contraction (``--fmad=true`` is the default, and fast numerics drops
``--fmad=false``).  The Metal translation never contracts, so on the MLX
backend the same measurement gives CUDA's ``--fmad=false`` words; those are
recorded beside the CUDA pin as ``<stem>-metal<suffix>``.  A CUDA process
under ``GPUWM_WRF_EXACT=1`` compiles the WRF-exact sources without
contraction; its words are recorded as ``<stem>-wrf-exact<suffix>``.

There is no fallback that loosens a gate: a backend without its own file is
held to the CUDA pin, and fails on any drift.
"""
from __future__ import annotations

from pathlib import Path

METAL_SUFFIX = "-metal"
WRF_EXACT_SUFFIX = "-wrf-exact"


def backend_variant() -> str | None:
    """The pin variant of the active kernel backend (None for default CUDA)."""
    from woof import backend
    if backend.active() == "mlx":
        return "metal"
    from woof.wrf_exact import ENABLED
    return "wrf-exact" if ENABLED else None


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
