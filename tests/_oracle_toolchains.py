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


def e3sm_v320(*marker: str, directory: bool = False, icepack: bool = False) -> tuple | None:
    """``(gfortran, e3sm, mct)``, or ``(gfortran, e3sm, mct, icepack)`` with
    ``icepack``, when ``GFORTRAN`` (or gfortran on PATH), ``WOOF_E3SM_V320_SOURCE``
    and ``WOOF_MCT_SOURCE`` (and ``WOOF_ICEPACK_SOURCE``, carrying
    ``columnphysics``) are set and the E3SM tree carries ``marker`` (a file, or a
    directory with ``directory``); else None."""
    e3sm, mct = os.environ.get("WOOF_E3SM_V320_SOURCE"), os.environ.get("WOOF_MCT_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not e3sm or not mct or not gfortran:
        return None
    marked = Path(e3sm, *marker)
    if not (marked.is_dir() if directory else marked.is_file()):
        return None
    if not icepack:
        return gfortran, Path(e3sm), Path(mct)
    ice = os.environ.get("WOOF_ICEPACK_SOURCE")
    if not ice or not Path(ice, "columnphysics").is_dir():
        return None
    return gfortran, Path(e3sm), Path(mct), Path(ice)
