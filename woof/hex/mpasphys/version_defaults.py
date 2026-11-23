"""Scheme generations taken from a default, and the pinned-lane guard.

Upstream WOOF 1.1.7 made the newest released scheme generations the hex defaults in correct mode:
MMM-physics 550b5b4 for YSU and the revised MM5 layer (mmm_version.py), MYNN v6.1 and TEMPO v3.2.5 on the
RRFS routes (rrfs/driver_rrfs.py, rrfs/tempo_v303.py).  Exact mode and the scheme modules' own arguments
still fall back to the pins: MMM a4baf7f (MPAS-A v8.4.2), the v8.4.2 YSU drag, MYNN "pinned" (44c0849)
and TEMPO v3.0.3.

A pinned oracle lane compares hex with pinned Fortran (MPAS-A v8.4.2 or RRFS 44c0849).  Such a lane names
its scheme versions rather than taking a default.  Inside :func:`pinned_lane`, or with
``WOOF_HEX_PINNED_LANE=1`` in the environment (subprocess forecasts), a version taken from a default
raises :class:`DefaultedVersion`.  So a lane that leaned on a default fails by name, instead of quietly
comparing a newer scheme with the pinned Fortran once that default moves.  Outside a lane,
:func:`defaulted` returns the default unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Iterator, TypeVar

ENVIRONMENT = "WOOF_HEX_PINNED_LANE"
#: Every selector a pinned lane must name (the hex routes' scheme-generation keys).
SELECTORS = ("mmm_version", "gwdo_version", "config_mynn_version", "config_tempo_version")

_depth = 0
T = TypeVar("T")


class DefaultedVersion(RuntimeError):
    """A pinned lane took a scheme version from a default instead of naming it."""


def strict() -> bool:
    """Whether a defaulted version is refused here (inside :func:`pinned_lane` or under the environment flag)."""
    return _depth > 0 or os.environ.get(ENVIRONMENT, "").strip() == "1"


def defaulted(selector: str, value: T, *, where: str) -> T:
    """``value``, the default ``selector`` takes at ``where``; refused inside a pinned lane."""
    if strict():
        raise DefaultedVersion(
            f"{where}: {selector} was not named, so it took the default {value!r}.  A pinned oracle lane names "
            f"its scheme versions ({selector}=<pin>), so a moved default cannot change what it compares")
    return value


@contextmanager
def pinned_lane() -> Iterator[None]:
    """Refuse every defaulted scheme version for the duration (nests; process-wide, so threads see it)."""
    global _depth
    _depth += 1
    try:
        yield
    finally:
        _depth -= 1
