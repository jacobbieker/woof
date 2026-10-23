"""Reader and word-level comparison for the WRF v4.7.1 smoke oracle fixtures.

The fixtures are written by ``tools/smoke_wrf471_oracle/oracle_io.F90``:
one directory per case, one little-endian ``<name>.bin`` per array (raw
float32 ``f4``, float64 ``f8`` or int32 ``i4``, Fortran column-major) and a
``MANIFEST.txt`` line per array ``<name> <kind> <rank> <extents...>``.
:func:`load_case` reads them back with ``order='F'``, so a Fortran
``(k, n)`` array arrives indexed ``[k, n]``.

They are test data in a source checkout (``tests/data/oracles/smoke``), not
package data (:mod:`woof.verify.wrf471_fixtures`).

Equality is word equality: :func:`max_ulp` is the largest distance in units
in the last place between two same-dtype arrays, on the monotone integer
key of each float format (so ``-0`` and ``+0`` are one ULP apart, and two
NaNs compare by their bits), and :func:`assert_words_equal` refuses any
non-zero distance by name, with the first offending index.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from woof.verify.wrf471_fixtures import require_fixture_dir

#: The published fixtures, in a source checkout.
ORACLE = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles"
          / "smoke")
#: The harness that wrote them.
TOOL = (Path(__file__).resolve().parents[2] / "tools"
        / "smoke_wrf471_oracle")

_KINDS = {"f4": np.dtype("<f4"), "f8": np.dtype("<f8"),
          "i4": np.dtype("<i4")}


def load_case(case: str | Path) -> dict[str, np.ndarray]:
    """Every array of one fixture case, by name.

    ``case`` is a directory, or a path relative to :data:`ORACLE`
    (``"emission/case01"``).
    """
    directory = Path(case)
    if not directory.is_absolute():
        directory = require_fixture_dir(ORACLE, "smoke") / directory
    out: dict[str, np.ndarray] = {}
    for line in (directory / "MANIFEST.txt").read_text().splitlines():
        if not line.strip():
            continue
        name, kind, rank, *extents = line.split()
        shape = tuple(int(e) for e in extents)
        if len(shape) != int(rank):
            raise ValueError(f"{directory}: bad manifest line {line!r}")
        data = np.fromfile(directory / f"{name}.bin", dtype=_KINDS[kind])
        if data.size != int(np.prod(shape, dtype=np.int64)):
            raise ValueError(f"{directory}/{name}.bin holds {data.size} "
                             f"words, the manifest says {shape}")
        out[name] = data.reshape(shape, order="F") if shape else data[0]
    return out


def case_names(component: str) -> list[str]:
    """The case directories of one component, sorted."""
    root = require_fixture_dir(ORACLE, "smoke") / component
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _key(values: np.ndarray) -> np.ndarray:
    """Monotone signed-integer key of each float word."""
    # A negative word's key is -(magnitude bits) - 1, so -0.0 maps to -1
    # and +0.0 to 0: adjacent, one ULP apart.
    if values.dtype == np.float32:
        bits = values.view(np.int32).astype(np.int64)
        return np.where(bits < 0, np.int64(-2**31) - bits - 1, bits)
    if values.dtype == np.float64:
        bits = values.view(np.int64)
        return np.where(bits < 0, (np.int64(-2**63) - bits) - 1, bits)
    raise TypeError(f"no ULP key for {values.dtype}")


def ulp_distance(got, want) -> np.ndarray:
    """Per-word ULP distance; the arrays must share shape and dtype."""
    got, want = np.asarray(got), np.asarray(want)
    if got.shape != want.shape or got.dtype != want.dtype:
        raise ValueError(f"cannot compare {got.dtype}{got.shape} with "
                         f"{want.dtype}{want.shape}")
    if got.dtype.kind in "iub":
        return (got != want).astype(np.int64)
    a, b = _key(got), _key(want)
    # Python ints: the float64 key difference can exceed int64.
    diff = np.abs(a.astype(object) - b.astype(object))
    return np.asarray(diff, dtype=object)


def max_ulp(got, want) -> int:
    """Largest word distance, 0 for raw-word equality."""
    distance = ulp_distance(got, want)
    return int(distance.max()) if distance.size else 0


def assert_words_equal(got, want, label: str = "") -> None:
    """Refuse any word that differs, naming the first one."""
    got, want = np.asarray(got), np.asarray(want)
    distance = ulp_distance(got, want)
    if distance.size and int(distance.max()) != 0:
        first = np.unravel_index(int(np.argmax(distance != 0)), got.shape)
        raise AssertionError(
            f"{label}: {int((distance != 0).sum())} of {got.size} words "
            f"differ, max {int(distance.max())} ULP; first at {first}: "
            f"got {got[first]!r}, want {want[first]!r}")
