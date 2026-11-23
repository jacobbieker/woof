"""GFDL Eta radiation (ra_lw/sw_physics = 99) storage tiers, importable without CuPy.

Stock WRF has no 81-layer ceiling for this scheme: ``module_ra_gfdleta.F``
sizes every model-level array from ``kte``/``kme`` (CO2O3 ``ALLOCATE``s its
``LP1`` tables, CONRAD and RADTN use automatic arrays).  Its ``NL=81`` is the
GFDL ozone climatology's fixed pressure grid (``XDUO3N(37,NL)``,
``PRGFDL(NL)``), which OZON2D interpolates to any number of model layers.

WOOF's transcription (tools/gfdleta_wrf461_oracle) gives every per-column
array fixed storage ``GDMAX``.  The shipped units are generated at 81, so a
column of at most 81 layers runs the unchanged kernels word for word.  The
deep tier regenerates the same lowering at ``GFDLETA_MAXL=256``
(``tools/gfdleta_wrf461_oracle/build_tier.py``) into the ``_l256`` modules
and layouts; the climatology stays on its 81 levels at both tiers.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Storage bound of the unchanged shipped units.
SHIPPED_MAXL = 81
#: The deep tier, the dynamical core's level ceiling.
DEEP_MAXL = 256
TIERS = (SHIPPED_MAXL, DEEP_MAXL)
VERTICAL_LEVEL_BOUNDS = (1, DEEP_MAXL)

_ROOT = Path(__file__).parent


def maxl_tier(nz: int) -> int:
    """Storage tier an ``nz``-layer column runs at."""
    nz = int(nz)
    if nz < 1:
        raise ValueError(f"GFDL Eta radiation needs at least one layer; this domain has {nz}")
    if nz <= SHIPPED_MAXL:
        return SHIPPED_MAXL
    if nz <= DEEP_MAXL:
        return DEEP_MAXL
    raise ValueError(
        f"GFDL Eta radiation holds at most {DEEP_MAXL} layers per column; "
        f"this domain has {nz}")


def _check(maxl: int) -> int:
    if maxl not in TIERS:
        raise ValueError(f"GFDL Eta storage tier must be one of {TIERS}, not {maxl!r}")
    return int(maxl)


def suffix(maxl: int) -> str:
    return "" if _check(maxl) == SHIPPED_MAXL else f"_l{maxl}"


def column_module(maxl: int) -> str:
    return "gfdleta_column" + suffix(maxl)


def co2_init_module(maxl: int) -> str:
    return "gfdleta_co2_init" + suffix(maxl)


def column_layout(maxl: int) -> dict:
    return json.loads((_ROOT / f"gfdleta_column_layout{suffix(maxl)}.json").read_text())


def init_layout(maxl: int) -> dict:
    return json.loads((_ROOT / f"gfdleta_init_layout{suffix(maxl)}.json").read_text())


def column_slots(ncol: int, column_chunk: int, maxl: int) -> int:
    """Concurrent column slots of one pooled launch.

    ``column_chunk`` is the run's workspace lever at both tiers: a deep slot
    holds 1.30 MB against 234 KB, and preflight prices ``slots x slot``, so
    an over-budget deep run lowers ``column_chunk`` as any other does.
    Pooled slots are reused in order, so the slot count changes no word.
    """
    _check(maxl)
    return max(1, min(int(ncol), int(column_chunk)))
