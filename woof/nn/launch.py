"""``--nn FILE``: the launch's NN column-physics settings.

The file is a TOML document with one ``[nn]`` table (keys below).  The
launch exports its resolved path as ``WOOF_NN_CONFIG`` for its duration,
the way ``--wrf-hydro`` does (:mod:`woof.wrf_hydro_route`), so every
process the launch starts (``woof go``'s forecast worker, ``woof run``)
reads the same settings; the WRF route's single-domain integrator attaches
the seat when the variable is set (:func:`woof.nn.wrf_seat.attach`).
Without ``--nn`` nothing here runs and nothing under :mod:`woof.nn` is
imported by a forecast.

``[nn]`` keys
-------------
``mode``         ``"tap"``: record the scheme's inputs and tendencies, change
                 nothing.  (Applying a model, ``"residual"`` and
                 ``"replace"``, is plan item P2.3.)
``scheme``       whose tendencies: ``"pbl"``, ``"radiation"``, ``"cumulus"``
                 or ``"total"`` (every non-microphysics tendency the step
                 holds, after SPPT/SKEBS).
``inputs``       vocabulary names (:data:`woof.nn.wrf_seat.VOCABULARY`);
                 default per scheme.
``outputs``      ``dtheta``, ``dqv``, ``dqc``, ``dqr``, ``dqi``, ``dqs``,
                 ``du``, ``dv``; default per scheme.
``levels``       keep the lowest ``levels`` model levels of every 3-D field
                 (default: all).
``tap``          the store to write (a directory; ``*.icechunk`` makes an
                 Icechunk repository, anything else plain Zarr).
``tap_every``    record every n-th dycore step (default 1).
``tap_columns``  columns drawn per recorded step (default 256), uniformly
                 from the interior: a specified or nested domain's
                 ``spec_bdy_width`` ring is excluded.
``tap_seed``     the draw's seed (default 0).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ENV_CONFIG = "WOOF_NN_CONFIG"

MODES = ("tap",)
LATER_MODES = ("residual", "replace")
SCHEMES = ("pbl", "radiation", "cumulus", "total")
_KEYS = {"mode", "scheme", "inputs", "outputs", "levels", "tap", "tap_every", "tap_columns", "tap_seed"}


class NNConfigError(ValueError):
    """An ``[nn]`` table that cannot be run."""


@dataclass(frozen=True)
class NNSettings:
    source: Path
    mode: str
    scheme: str
    inputs: tuple[str, ...] | None
    outputs: tuple[str, ...] | None
    levels: int | None
    tap: Path | None
    tap_every: int
    tap_columns: int
    tap_seed: int


def add_arguments(parser) -> None:
    parser.add_argument(
        "--nn", default=None, metavar="NN.toml",
        help="opt-in neural-network column physics from the [nn] table of this TOML file "
             "(docs/nn-physics.md); mode = \"tap\" records a scheme's inputs and tendencies "
             "as training data and leaves the forecast unchanged.  Not WRF-identical when a "
             "model is applied")


def read_settings(path) -> NNSettings:
    """Parse and check the ``[nn]`` table of ``path``; paths in it are
    anchored at the file."""
    import tomllib

    path = Path(path).resolve()
    try:
        raw = tomllib.loads(path.read_text())
    except OSError as error:
        raise NNConfigError(f"--nn {path}: cannot read it ({error})") from error
    except tomllib.TOMLDecodeError as error:
        raise NNConfigError(f"--nn {path}: not TOML ({error})") from error
    table = raw.get("nn")
    if not isinstance(table, dict):
        raise NNConfigError(f"--nn {path}: no [nn] table")
    unknown = sorted(set(table) - _KEYS)
    if unknown:
        raise NNConfigError(f"--nn {path}: unknown [nn] keys {unknown}; known: {sorted(_KEYS)}")
    mode = table.get("mode", "tap")
    if mode in LATER_MODES:
        raise NNConfigError(
            f"--nn {path}: mode = {mode!r} applies a model, which this build does not do yet "
            "(plan item P2.3); mode = \"tap\" records training data")
    if mode not in MODES:
        raise NNConfigError(f"--nn {path}: mode must be one of {MODES}; got {mode!r}")
    scheme = table.get("scheme")
    if scheme not in SCHEMES:
        raise NNConfigError(f"--nn {path}: scheme must be one of {SCHEMES}; got {scheme!r}")

    def names(key):
        value = table.get(key)
        if value is None:
            return None
        if not (isinstance(value, list) and value and all(isinstance(v, str) for v in value)):
            raise NNConfigError(f"--nn {path}: {key} must be a non-empty list of names")
        if len(set(value)) != len(value):
            raise NNConfigError(f"--nn {path}: {key} names a field twice")
        return tuple(value)

    def positive(key, default):
        value = table.get(key, default)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise NNConfigError(f"--nn {path}: {key} must be a positive integer; got {value!r}")
        return value

    seed = table.get("tap_seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise NNConfigError(f"--nn {path}: tap_seed must be a non-negative integer; got {seed!r}")
    tap = table.get("tap")
    if mode == "tap" and not tap:
        raise NNConfigError(f"--nn {path}: mode = \"tap\" needs tap = \"<store directory>\"")
    if tap is not None and not isinstance(tap, str):
        raise NNConfigError(f"--nn {path}: tap must be a path string")
    return NNSettings(
        source=path, mode=mode, scheme=scheme, inputs=names("inputs"), outputs=names("outputs"),
        levels=positive("levels", None), tap=None if tap is None else (path.parent / tap).resolve(),
        tap_every=positive("tap_every", 1), tap_columns=positive("tap_columns", 256), tap_seed=seed)


@contextmanager
def launch_scope(path=None, members=None):
    """Export ``--nn``'s file for the launch's duration.  Checked here, at
    the door, so a bad table is refused before any preparation.  An
    ensemble that taps (``members`` > 1) is refused: every member would
    write the one store."""
    if path is None:
        yield
        return
    settings = read_settings(path)
    if settings.mode == "tap" and members is not None and int(members) > 1:
        raise NotImplementedError(
            f"--nn with mode = \"tap\" and {int(members)} ensemble members: the members would "
            "write one store; record from a single member")
    saved = os.environ.get(ENV_CONFIG)
    os.environ[ENV_CONFIG] = str(settings.source)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(ENV_CONFIG, None)
        else:
            os.environ[ENV_CONFIG] = saved


def configured() -> str | None:
    """The ``--nn`` file this launch carries, or None (the default)."""
    return os.environ.get(ENV_CONFIG) or None


def refuse_multiple_domains(count: int) -> None:
    if configured() and int(count) > 1:
        raise NotImplementedError(
            f"--nn with {count} domains: the NN seat attaches to a single-domain WRF-route "
            "forecast; run it on one domain")


def refuse_streamed() -> None:
    raise NotImplementedError(
        "--nn on a [tiles]-streamed domain: the seat reads and writes whole-domain arrays, "
        "which a streamed step never holds; run it resident")
