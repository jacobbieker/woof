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

import functools
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
    from woof.config import ConfigurationRefused
    raise ConfigurationRefused(
        f"GFDL Eta radiation (ra_lw_physics/ra_sw_physics = 99) holds at most "
        f"{DEEP_MAXL} layers per column, its deepest storage tier; this domain "
        f"has {nz}. Use at most {DEEP_MAXL} levels or another radiation scheme")


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


# Parsed once per tier; callers treat the layouts as read-only.
@functools.lru_cache(maxsize=None)
def column_layout(maxl: int) -> dict:
    return json.loads((_ROOT / f"gfdleta_column_layout{suffix(maxl)}.json").read_text())


@functools.lru_cache(maxsize=None)
def init_layout(maxl: int) -> dict:
    return json.loads((_ROOT / f"gfdleta_init_layout{suffix(maxl)}.json").read_text())


def column_slots(ncol: int, column_chunk: int, maxl: int) -> int:
    """Concurrent column slots of one pooled launch.

    ``column_chunk`` counts shipped-tier slots, so it is a scratch budget:
    a deep slot holds 1.30 MB against 234 KB, and the deep tier runs as many
    slots as fit the same bytes (the default 3125 shipped slots, 732 MB, are
    563 deep slots, not 4.1 GB).  Preflight prices ``slots x slot``; an
    over-budget run lowers ``column_chunk`` at either tier.  Pooled slots are
    reused in order, so the slot count changes no word.
    """
    maxl = _check(maxl)
    budget = int(column_chunk)
    if maxl != SHIPPED_MAXL:
        budget = (budget * column_layout(SHIPPED_MAXL)["scratch_bytes"]
                  // column_layout(maxl)["scratch_bytes"])
    return max(1, min(int(ncol), budget))
