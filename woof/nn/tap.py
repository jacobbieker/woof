"""The tendency tap: a scheme's column inputs and tendencies as training data.

One store per run, written as a Zarr v3 group (or an Icechunk repository
when the path ends in ``.icechunk``, committed at every flush):

``inputs``        float32 ``[sample, n_inputs, column]``
``targets``       float32 ``[sample, n_outputs, column]``
``time_seconds``  float64 ``[sample]``, forecast seconds at the step
``step``          int64 ``[sample]``, the dycore step (counted by the seat)
``column_index``  int64 ``[sample, column]``, ``j * nx + i`` on the mass grid

The group attributes carry ``schema`` (:data:`SCHEMA`), the route, the
scheme, the input and output field specs (``name``, ``units``, ``levels``,
in row order, the layout :mod:`woof.nn.manifest` takes), the grid shape and
the note that a model trained on it is not WRF-identical.

Samples are buffered on the host and written ``chunk`` at a time, so a
recorded step costs one device-to-host copy of the drawn columns and no
store write.  :meth:`TendencyTap.close` writes the rest; an interpreter
that exits without it (an exception in the forecast) still flushes through
``atexit``, which holds the tap until then.

zarr (and icechunk) are imported when a tap is opened, never by a forecast
without ``--nn``.
"""

from __future__ import annotations

import atexit
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from woof.nn import NOT_WRF_IDENTICAL
from woof.nn.manifest import FieldSpec

SCHEMA = "woof.nn.tendency-tap/v1"
ARRAYS = ("inputs", "targets", "time_seconds", "step", "column_index")


class TapError(RuntimeError):
    """A tap store that cannot be opened or appended to."""


def _zarr():
    try:
        import zarr
    except ImportError as error:
        raise TapError("the tendency tap writes Zarr: install zarr (pip install 'zarr>=3')") from error
    return zarr


def _icechunk():
    try:
        import icechunk
    except ImportError as error:
        raise TapError("a tap path ending in .icechunk writes an Icechunk repository: install "
                       "icechunk, or name a plain Zarr directory") from error
    return icechunk


def _is_icechunk(path: Path) -> bool:
    return path.suffix == ".icechunk"


def _specs(fields: Sequence[FieldSpec]) -> list[dict]:
    return [asdict(f) for f in fields]


class TendencyTap:
    """Append-only writer of one tap store.

    ``resume`` reopens an existing store written with the same attributes
    (a restarted forecast continues its record); otherwise an existing
    path is refused rather than overwritten."""

    def __init__(self, path, *, inputs: Sequence[FieldSpec], outputs: Sequence[FieldSpec],
                 columns: int, attrs: Mapping[str, Any] | None = None, chunk: int = 64,
                 resume: bool = False):
        self.path = Path(path)
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)
        self.n_in = sum(f.levels for f in self.inputs)
        self.n_out = sum(f.levels for f in self.outputs)
        self.columns = int(columns)
        self.chunk = int(chunk)
        self.attrs = {"schema": SCHEMA, "inputs": _specs(self.inputs), "outputs": _specs(self.outputs),
                      "columns": self.columns, "note": NOT_WRF_IDENTICAL, **dict(attrs or {})}
        self._buffer: list[tuple] = []
        self.samples = 0
        self._closed = False
        zarr = _zarr()
        self._zarr = zarr
        if self.path.exists() and not self.path.is_dir():
            raise TapError(f"tap store {self.path} is a file; name a directory")
        exists = self.path.is_dir() and any(self.path.iterdir())
        if exists and not resume:
            raise TapError(f"tap store {self.path} already exists; remove it or name another")
        self._repo = None
        try:
            if _is_icechunk(self.path):
                ic = _icechunk()
                storage = ic.local_filesystem_storage(str(self.path))
                self._repo = ic.Repository.open(storage) if exists else ic.Repository.create(storage)
            if exists:
                group = self._open_group("r")
                stored = dict(group.attrs)
                lengths = {name: int(group[name].shape[0]) for name in ARRAYS}
            else:
                self._create()
        except TapError:
            raise
        except Exception as error:
            raise TapError(f"tap store {self.path}: cannot open it as a tendency tap "
                           f"({type(error).__name__}: {error})") from error
        if exists:
            for key in ("schema", "inputs", "outputs", "columns", "route", "scheme", "grid",
                        "levels", "tap_every", "tap_seed", "target"):
                if key == "target":     # absent means the scheme's tendency, on either side
                    if (stored.get(key) or "physics") != (self.attrs.get(key) or "physics"):
                        raise TapError(f"tap store {self.path} records the {stored.get(key) or 'physics'} "
                                       f"target; this run taps the {self.attrs.get(key) or 'physics'} one")
                elif key in self.attrs and stored.get(key) != self.attrs[key]:
                    raise TapError(f"tap store {self.path} was written with a different {key} "
                                   f"({stored.get(key)!r}); a resumed run appends only to its own record")
            # A flush interrupted part-way leaves the arrays of unequal
            # length (plain Zarr has no transaction); the record is the
            # samples every array holds.
            self.samples = min(lengths.values())
            if len(set(lengths.values())) > 1:
                self._resize(self.samples)
        atexit.register(self._exit_flush)

    # ------------------------------------------------------------- store
    def _open_group(self, mode):
        if self._repo is None:
            return self._zarr.open_group(str(self.path), mode=mode)
        session = (self._repo.readonly_session(branch="main") if mode == "r"
                   else self._repo.writable_session("main"))
        self._session = session
        return self._zarr.open_group(session.store, mode=mode)

    def _commit(self, message):
        if self._repo is not None:
            self._session.commit(message)

    def _create(self):
        group = self._open_group("w")
        group.attrs.update(self.attrs)
        c, n = self.chunk, self.columns
        group.create_array("inputs", shape=(0, self.n_in, n), chunks=(c, self.n_in, n), dtype="float32")
        group.create_array("targets", shape=(0, self.n_out, n), chunks=(c, self.n_out, n), dtype="float32")
        group.create_array("time_seconds", shape=(0,), chunks=(max(c, 1024),), dtype="float64")
        group.create_array("step", shape=(0,), chunks=(max(c, 1024),), dtype="int64")
        group.create_array("column_index", shape=(0, n), chunks=(c, n), dtype="int64")
        self._commit("create tendency tap")

    # ------------------------------------------------------------- write
    def append(self, inputs, targets, *, time_seconds: float, step: int, column_index) -> None:
        """Buffer one recorded step: ``inputs`` ``[n_inputs, column]`` and
        ``targets`` ``[n_outputs, column]`` (host arrays)."""
        if self._closed:
            raise TapError(f"tap store {self.path} is closed")
        x = np.asarray(inputs, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32)
        idx = np.asarray(column_index, dtype=np.int64)
        if x.shape != (self.n_in, self.columns) or y.shape != (self.n_out, self.columns) \
                or idx.shape != (self.columns,):
            raise TapError(f"tap sample shapes {x.shape}, {y.shape}, {idx.shape}; the store takes "
                           f"({self.n_in}, {self.columns}), ({self.n_out}, {self.columns}), ({self.columns},)")
        self._buffer.append((x, y, float(time_seconds), int(step), idx))
        if len(self._buffer) >= self.chunk:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        x, y, t, s, idx = zip(*self._buffer)
        group = self._open_group("a")
        group["inputs"].append(np.stack(x), axis=0)
        group["targets"].append(np.stack(y), axis=0)
        group["time_seconds"].append(np.asarray(t, dtype=np.float64), axis=0)
        group["step"].append(np.asarray(s, dtype=np.int64), axis=0)
        group["column_index"].append(np.stack(idx), axis=0)
        self.samples += len(self._buffer)
        self._buffer.clear()
        self._commit(f"tendency tap: {self.samples} samples")

    def _resize(self, samples: int) -> None:
        group = self._open_group("a")
        for name in ARRAYS:
            array = group[name]
            array.resize((samples,) + tuple(array.shape[1:]))
        self._commit(f"tendency tap: kept {samples} samples")

    def truncate(self, time_seconds: float) -> None:
        """Drop the samples recorded at model time ``time_seconds`` or
        later: a resumed run writes them again."""
        self.flush()
        if not self.samples:
            return
        times = np.asarray(self._open_group("r")["time_seconds"][...])[:self.samples]
        keep = int(np.searchsorted(times, float(time_seconds), side="left"))
        if keep < self.samples:
            self._resize(keep)
            self.samples = keep

    def close(self) -> None:
        if not self._closed:
            self.flush()
            self._closed = True
            atexit.unregister(self._exit_flush)

    def _exit_flush(self) -> None:
        try:
            self.close()
        except Exception:  # the interpreter is exiting; the record keeps what it had
            pass


def read_tap(path) -> dict:
    """The arrays (NumPy) and attributes of a tap store."""
    path = Path(path)
    zarr = _zarr()
    if _is_icechunk(path):
        ic = _icechunk()
        repo = ic.Repository.open(ic.local_filesystem_storage(str(path)))
        group = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")
    else:
        group = zarr.open_group(str(path), mode="r")
    attrs = dict(group.attrs)
    if attrs.get("schema") != SCHEMA:
        raise TapError(f"{path} is not a tendency tap ({attrs.get('schema')!r}, expected {SCHEMA!r})")
    out = {name: np.asarray(group[name][...]) for name in ARRAYS}
    out["attrs"] = attrs
    out["inputs_spec"] = tuple(FieldSpec(**f) for f in attrs["inputs"])
    out["outputs_spec"] = tuple(FieldSpec(**f) for f in attrs["outputs"])
    return out
