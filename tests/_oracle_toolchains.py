"""The Fortran oracle toolchain probes the parity tests' rebuild tests share.

Each MPAS-Model v8.4.2 parity test rebuilds its committed fixture only when a
gfortran and an MPAS source tree are at hand; the probe is the same for every
core, so it lives here once.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def mpas_v842(core: str) -> tuple[str, Path] | None:
    """``(gfortran, source)`` when ``GFORTRAN`` (or gfortran on PATH) and
    ``WOOF_MPAS_V842_SOURCE``, naming an MPAS-Model v8.4.2 tree that carries
    ``src/<core>``, are both set; else None."""
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", core).is_dir():
        return None
    return gfortran, Path(source)
