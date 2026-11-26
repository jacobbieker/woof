"""WRF Fu-Liou-Gu (FLG) level tiers, importable without CuPy.

WRF's ``module_ra_flg.F`` sizes its level scratch by ``PARA_FILE``'s
``nvx = 100`` (``nv1x = nvx + 1``).  The generated CUDA keeps that bound as
the compile-time macro ``FLG_NVX``, whose ``#ifndef`` default is the stock
100: a column of at most 100 layers compiles stock WRF's bound, word for word
the pre-tier kernel.  A 101 to 256 layer column adds ``-D FLG_NVX=256``, the
bound of the patched-``nvx`` oracle it is verified against
(``tools/flg_wrf461_oracle/nvx256.patch``).

In the forecast column (``flg_column``) every per-level array already lives
in the global workspace sized by ``FLG_NV1`` (the column's interface count),
so ``FLG_NVX`` there sizes only the aerosol arrays that stock RAD_FLG
(``naero = 0``) never touches; the local frame does not grow with the tier.
In the standalone component oracles (``flg``, ``flg_gases``,
``flg_streams``) it sizes thread-local level arrays.
"""

from __future__ import annotations

#: The ``#ifndef`` default of ``FLG_NVX``: stock WRF's ``nvx``.
SHIPPED_NVX = 100
#: The deep tier, the dynamical core's level ceiling.
DEEP_NVX = 256
#: Model layers FLG admits: its three maximum-random cloud groups need a
#: layer each above the surface layer, and the deep tier is the ceiling.
VERTICAL_LEVEL_BOUNDS = (4, DEEP_NVX)


def nvx_tier(layers: int) -> int:
    """``FLG_NVX`` a column of ``layers`` layers compiles at."""
    layers = int(layers)
    if layers <= SHIPPED_NVX:
        return SHIPPED_NVX
    if layers <= DEEP_NVX:
        return DEEP_NVX
    raise ValueError(
        f"WRF FLG holds at most {DEEP_NVX} layers per column (FLG_NVX tier); "
        f"this column has {layers}")


def tier_defines(nvx: int) -> tuple[tuple[str, int], ...]:
    """Loader integer defines for an ``FLG_NVX = nvx`` build: none at the
    shipped tier, so it compiles the unchanged source."""
    nvx = int(nvx)
    if nvx not in (SHIPPED_NVX, DEEP_NVX):
        raise ValueError(f"FLG_NVX tiers are {SHIPPED_NVX} and {DEEP_NVX}, not {nvx}")
    return () if nvx == SHIPPED_NVX else (("FLG_NVX", nvx),)


def nvx_defines(layers: int) -> tuple[tuple[str, int], ...]:
    """Loader integer defines for a column of ``layers`` layers."""
    return tier_defines(nvx_tier(layers))
