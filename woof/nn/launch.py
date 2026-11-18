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
                 nothing.  ``"residual"``: add the model's output to the
                 scheme's tendencies.  ``"replace"``: use the model's output
                 in place of the scheme's (the scheme still runs: its
                 tendency is the fallback and the fixer's budget).  Either
                 applying mode makes the run NOT WRF-identical and is
                 refused under ``GPUWM_WRF_EXACT=1``.
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

Applying modes only (the inputs, outputs and levels are the artefact's):

``model``          the artefact directory (:mod:`woof.nn.manifest`).
``backend``        ``"auto"`` (MLX on Metal, PyTorch on CUDA), ``"torch"``,
                   ``"mlx"`` or ``"numpy"``.
``device``         a torch device string (default: the array's).
``out_of_range``   ``"fallback"`` (default), ``"refuse"`` or ``"clip"``
                   (:class:`woof.nn.safety.SafetyPolicy`).
``margin``         widen the trained range by this fraction (default 0).
``fixer``          ``"none"`` (default), ``"moisture"`` or ``"moisture+heat"``.
``limit_outputs``  clamp to the manifest's ``output_limit`` (default true).
``positivity``     keep ``q + dt * dq >= min(q, 0)`` (default true).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ENV_CONFIG = "WOOF_NN_CONFIG"

MODES = ("tap", "residual", "replace")
APPLYING = ("residual", "replace")
SCHEMES = ("pbl", "radiation", "cumulus", "total")
_TAP_KEYS = {"inputs", "outputs", "levels", "tap", "tap_every", "tap_columns", "tap_seed"}
_APPLY_KEYS = {"model", "backend", "device", "out_of_range", "margin", "fixer", "limit_outputs",
               "positivity"}
_KEYS = {"mode", "scheme"} | _TAP_KEYS | _APPLY_KEYS


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
    model: Path | None = None
    backend: str = "auto"
    device: str | None = None
    out_of_range: str = "fallback"
    margin: float = 0.0
    fixer: str = "none"
    limit_outputs: bool = True
    positivity: bool = True

    @property
    def applies(self) -> bool:
        return self.mode in APPLYING

    def identity(self) -> dict:
        """What a checkpoint of an applying run binds: the settings that
        change the trajectory, with ``auto`` resolved to the runner this
        process uses (the artefact's own identity is added by the seat)."""
        backend = self.backend
        if backend == "auto":
            from woof.nn.models import default_backend
            backend = default_backend()
        return {"mode": self.mode, "scheme": self.scheme, "backend": backend, "device": self.device,
                "out_of_range": self.out_of_range, "margin": float(self.margin), "fixer": self.fixer,
                "limit_outputs": self.limit_outputs, "positivity": self.positivity}


def add_arguments(parser) -> None:
    parser.add_argument(
        "--nn", default=None, metavar="NN.toml",
        help="opt-in neural-network column physics from the [nn] table of this TOML file "
             "(docs/nn-physics.md); mode = \"tap\" records a scheme's inputs and tendencies "
             "as training data and leaves the forecast unchanged; \"residual\" and \"replace\" "
             "apply a column model, which makes the run not WRF-identical")


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
    if mode not in MODES:
        raise NNConfigError(f"--nn {path}: mode must be one of {MODES}; got {mode!r}")
    stray = sorted(set(table) & (_TAP_KEYS if mode in APPLYING else _APPLY_KEYS))
    if stray:
        raise NNConfigError(
            f"--nn {path}: mode = {mode!r} does not take {stray}"
            + ("; an applying mode reads its inputs, outputs and levels from the artefact"
               if mode in APPLYING else "; those keys apply a model"))
    if mode in APPLYING:
        refuse_identity_door(f"--nn {path}: mode = {mode!r}")
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
    apply = {}
    if mode in APPLYING:
        from woof.nn.models import BACKENDS
        from woof.nn.safety import FIXERS, OUT_OF_RANGE

        model = table.get("model")
        if not isinstance(model, str) or not model:
            raise NNConfigError(f"--nn {path}: mode = {mode!r} needs model = \"<artefact directory>\"")
        choices = {"backend": BACKENDS, "out_of_range": OUT_OF_RANGE, "fixer": FIXERS}
        defaults = {"backend": "auto", "out_of_range": "fallback", "fixer": "none"}
        for key, allowed in choices.items():
            value = table.get(key, defaults[key])
            if value not in allowed:
                raise NNConfigError(f"--nn {path}: {key} must be one of {allowed}; got {value!r}")
            apply[key] = value
        margin = table.get("margin", 0.0)
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not 0.0 <= margin <= 1.0:
            raise NNConfigError(f"--nn {path}: margin must be a fraction in [0, 1]; got {margin!r}")
        for key in ("limit_outputs", "positivity"):
            value = table.get(key, True)
            if not isinstance(value, bool):
                raise NNConfigError(f"--nn {path}: {key} must be true or false; got {value!r}")
            apply[key] = value
        device = table.get("device")
        if device is not None and not isinstance(device, str):
            raise NNConfigError(f"--nn {path}: device must be a string such as \"cuda:0\"")
        apply.update(model=(path.parent / model).resolve(), margin=float(margin), device=device)
    return NNSettings(
        source=path, mode=mode, scheme=scheme, inputs=names("inputs"), outputs=names("outputs"),
        levels=positive("levels", None), tap=None if tap is None else (path.parent / tap).resolve(),
        tap_every=positive("tap_every", 1), tap_columns=positive("tap_columns", 256), tap_seed=seed,
        **apply)


def refuse_identity_door(what: str) -> None:
    """An applying mode is not WRF-identical; the identity doors refuse it."""
    if os.environ.get("GPUWM_WRF_EXACT", "0") == "1":
        raise NNConfigError(
            f"{what} applies a neural-network column model, which is not WRF-identical; "
            "GPUWM_WRF_EXACT=1 runs only WRF-identical physics. Unset it, or use mode = \"tap\"")


@contextmanager
def launch_scope(path=None, members=None, resume=False):
    """Export ``--nn``'s file for the launch's duration.  Checked here, at
    the door, so a bad table is refused before any preparation.  An
    ensemble (``members`` > 1) is refused: every member would write the one
    store (a member leg refuses ``--nn`` itself, however its count was
    set).  A tap store that already holds a record is refused unless the
    launch resumes (``resume``), which appends to it."""
    if path is None:
        yield
        return
    settings = read_settings(path)
    if members is not None and int(members) > 1:
        raise NotImplementedError(
            f"--nn with {int(members)} ensemble members: the NN seat attaches to a single "
            "deterministic forecast")
    if settings.tap is not None and not resume and settings.tap.exists() \
            and (not settings.tap.is_dir() or any(settings.tap.iterdir())):
        raise NNConfigError(f"--nn {settings.source}: tap store {settings.tap} already exists; "
                            "remove it or name another (a resumed run appends to it)")
    saved = os.environ.get(ENV_CONFIG)
    os.environ[ENV_CONFIG] = str(settings.source)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(ENV_CONFIG, None)
        else:
            os.environ[ENV_CONFIG] = saved


def registry_rows() -> dict:
    """The physics registry's ``nn_column_physics`` block: what ``--nn``
    offers.  Not a namelist selection; an applying mode is never
    WRF-identical."""
    from woof.nn import NOT_WRF_IDENTICAL
    from woof.nn.safety import FIXERS, OUT_OF_RANGE

    common = {"implemented": True, "maturity": "experimental-runtime", "scientific_evidence": "none",
              "launch": "--nn NN.toml ([nn] table; docs/nn-physics.md)", "routes": ["wrf"],
              "schemes": list(SCHEMES), "consumer": "woof.nn.wrf_seat.WrfColumnSeat"}
    return {
        "tap": {**common, "wrf_identical": True,
                "behavior": "records a scheme's column inputs and decoupled tendencies to Zarr or "
                            "Icechunk; the forecast is byte-identical to the same run without --nn",
                "restart": "binds nothing; a resumed run trims and appends to its own record"},
        **{mode: {**common, "wrf_identical": False, "note": NOT_WRF_IDENTICAL,
                  "behavior": ("adds the model's rate to the scheme's" if mode == "residual"
                               else "uses the model's rate in place of the scheme's (the scheme still "
                                    "runs: fallback columns and the fixer's budget)"),
                  "backends": ["torch (CUDA)", "mlx (Metal)", "numpy"],
                  "safety": {"out_of_range": list(OUT_OF_RANGE), "fixer": list(FIXERS),
                             "limit_outputs": True, "positivity": True},
                  "refused_under": "GPUWM_WRF_EXACT=1",
                  "restart": "checkpoint header binds the artefact hashes and the settings"}
           for mode in APPLYING},
    }


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
