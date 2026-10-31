"""Where woof finds libRadtran's data files at run time.

woof never ships or copies libRadtran data (GPL-2 distribution); it reads the
user's own installation:

1. ``$WOOF_LIBRADTRAN_DATA`` (the ``data`` directory of a libRadtran
   install), else
2. the project's pixi ``libradtran`` environment
   (``.pixi/envs/libradtran/share/libRadtran/data``), else
3. ``$CONDA_PREFIX/share/libRadtran/data``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ENV = "WOOF_LIBRADTRAN_DATA"


class LibradtranDataMissing(FileNotFoundError):
    pass


def _candidates():
    v = os.environ.get(ENV)
    if v:
        yield Path(v)
    here = Path(__file__).resolve()
    for root in here.parents:
        yield root / ".pixi" / "envs" / "libradtran" / "share" / "libRadtran" / "data"
        if (root / "pyproject.toml").exists():
            break
    cp = os.environ.get("CONDA_PREFIX")
    if cp:
        yield Path(cp) / "share" / "libRadtran" / "data"


@lru_cache(maxsize=1)
def data_dir() -> Path:
    for c in _candidates():
        if (c / "atmmod").is_dir():
            return c
    raise LibradtranDataMissing(
        "woof.radtran reads libRadtran's data files at run time; set "
        f"${ENV} to a libRadtran 'data' directory or run `pixi install -e libradtran`")


def path(*parts: str) -> Path:
    p = data_dir().joinpath(*parts)
    if not p.exists():
        raise LibradtranDataMissing(f"libRadtran data file not found: {p}")
    return p


def reptran_file(resolution: str = "coarse", kind: str = "solar") -> Path:
    return path("correlated_k", "reptran", f"reptran_{kind}_{resolution}.cdf")
