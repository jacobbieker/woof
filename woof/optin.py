"""Explicit opt-ins for code under licences the default build cannot carry.

Some ports are allowed in this repository only as opt-ins, the way the glibc
2.39 libm transcription is (#145): their source lives under ``optional/<name>``
at the repository root, outside every Python package, with its licence text
beside it.  No wheel and no default build contains it.  This module is the
only door to it, and it opens only when the opt-in's environment variable is
set and the source tree is present:

==============  ==========================  ==================================
opt-in          environment variable        licence
==============  ==========================  ==================================
``wrf_hydro``   ``WOOF_ENABLE_WRF_HYDRO=1`` UCAR WRF-Hydro licence (no sale
                                            or transfer for a fee)
==============  ==========================  ==================================

``WOOF_OPTIONAL_ROOT`` points at another ``optional/`` directory (an install
whose checkout lives elsewhere); by default it is the one beside the package.
Nothing here is derived from the opted-in code.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

#: opt-in -> (environment variable, importable package inside optional/<name>,
#: one line naming the licence for the refusal text)
OPT_INS = {
    "wrf_hydro": ("WOOF_ENABLE_WRF_HYDRO", "woof_wrf_hydro",
                  "UCAR's WRF-Hydro licence, which forbids selling or "
                  "transferring for a fee any work that contains it"),
}


class OptInUnavailable(NotImplementedError):
    """The opt-in is not enabled or its source is not present."""


def optional_root() -> Path:
    root = os.environ.get("WOOF_OPTIONAL_ROOT")
    if root:
        return Path(root)
    return Path(__file__).resolve().parents[1] / "optional"


def unavailable_reason(name: str) -> str | None:
    """Why ``load(name)`` would refuse, or None when it would load."""
    if name not in OPT_INS:
        return f"{name!r} is not a WOOF opt-in; known: {sorted(OPT_INS)}"
    variable, package, licence = OPT_INS[name]
    if os.environ.get(variable) != "1":
        return (f"the {name} opt-in is off: it is under {licence}, so it is "
                f"in no wheel and no default build; set {variable}=1 in a "
                "source checkout to enable it")
    if not (optional_root() / name / package / "__init__.py").is_file():
        return (f"{variable}=1 but {optional_root() / name} has no {package} "
                "package; the opt-in source ships only in a source checkout "
                "(set WOOF_OPTIONAL_ROOT to its optional/ directory)")
    return None


def load(name: str) -> ModuleType:
    """Import and return the opt-in's package, or raise OptInUnavailable."""
    reason = unavailable_reason(name)
    if reason is not None:
        raise OptInUnavailable(reason)
    _variable, package, _licence = OPT_INS[name]
    path = str(optional_root() / name)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module(package)
