"""The passive smoke tracer (``RunConfig.smoke_tracer``).

``state.smoke`` is a positive-definite mass mixing ratio in micrograms of
smoke per kilogram of DRY air, the coupling every moist scalar of the ARW
core uses.  It is the analogue of WRF's ``tracer`` array, which
``tracer_opt`` fills for fire smoke: it is advected, mixed and bounded like
a scalar, and nothing reads it back -- no radiation, microphysics or
dynamics term sees it, so turning it on cannot move any other field.

Where each process lives:

* RK3 transport, positive-definite on the final stage, plus the held
  per-step forward tendency (km_opt mixing, sixth-order filter and the
  surface emission below) -- :func:`woof.core.moist.advance_smoke_stage`,
  called from :func:`woof.core.dycore.step` beside the TKE carrier.
* Lateral boundaries: zero inflow and zero-gradient outflow on a specified
  domain (WRF ``flow_dep_bdy``, the treatment WRF gives a tracer with no
  boundary file), the parent's forcing tables on a nest, zero-gradient
  outflow on a radiative-open edge.
* PBL mixing: :func:`mix_pbl`, operator-split at every PBL call with the
  scheme's own heat diffusivity ``exch_h`` through WRF's ``diff4d``
  implicit solve (the ``scalar_pblmix`` kernel, which conserves column
  mass and cannot make a value negative).
* Surface emission: :func:`bind_emission` reads the prescribed flux once
  into ``state.smoke_emis`` and :func:`add_surface_source` turns it into a
  coupled tendency on the lowest layer.  ``smoke_emis`` is serialized, so
  a coupled producer may later write it every step instead of the file.
* ``state.smoke_col``: the column load in ug m-2, refreshed at the end of
  every step by :func:`refresh_column_load` for the history writer.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from woof.core import constants as c

DTYPE = np.float32

#: Units of ``state.smoke`` (history attribute and docs).
SMOKE_UNITS = "ug kg-1"
#: Units of ``state.smoke_col``.
SMOKE_COLUMN_UNITS = "ug m-2"
#: Units of the prescribed emission file and of ``state.smoke_emis``.
EMISSION_UNITS = "kg m-2 s-1"
#: The variable a NetCDF emission file carries the flux under.
EMISSION_VARIABLE = "smoke_emission"
#: kg -> ug.
UG_PER_KG = 1.0e9
#: The largest column batch one PBL mixing launch takes.  The diff4d solve
#: allocates its output pair and a (5*nz + 1)-float scratch row per column,
#: so the batch bounds that transient instead of letting it scale with the
#: domain (woof.core.preflight prices exactly this batch).
PBL_COLUMN_PIECE = 16384

#: ``float32(1e9 * g)``: kg -> ug times WOOF's gravity, formed once in
#: binary64 and rounded, the same word on the device and in the references.
UG_G = DTYPE(UG_PER_KG * c.G)
#: Gravity as the column-load kernel divides by it.
G = DTYPE(c.G)

#: No FMA contraction and no flush to zero (woof.core.terrain_drag).
MODULE_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
MODULE_KEY = "woof.core.smoke_tracer:smoke_tracer"
_SOURCE = "smoke_tracer.cu"
_BLOCK = 128

#: The state attribute that records which emission file is in
#: ``smoke_emis``.  An infra attribute (woof/io/restart.py): it is a
#: provenance marker, not model state.
_BOUND_ATTRIBUTE = "_smoke_emission_bound"


def enabled(cfg) -> bool:
    """Whether this configuration carries the smoke tracer."""
    return bool(getattr(cfg, "smoke_tracer", False))


def load_emission(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """The prescribed surface emission, kg m-2 s-1, as a host FP32 array.

    ``.npy``: a 2-D array in kg m-2 s-1.  ``.nc``: the ``smoke_emission``
    variable, 2-D (or 3-D with a leading length-1 time axis); when it has
    a ``units`` attribute it must be ``kg m-2 s-1`` or ``ug m-2 s-1``, and
    the second is converted.  Refuses by name a file that does not open,
    a field of the wrong shape, and a value that is negative or not finite.
    """
    from woof.explain import layered

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(layered(
            f"smoke_emission_file {str(path)!r} does not exist.",
            "The prescribed smoke emission is read once when the tracer "
            "first steps; a missing file is refused rather than run as no "
            "emission."))
    suffix = source.suffix.lower()
    scale = 1.0
    if suffix == ".npy":
        values = np.load(source, allow_pickle=False)
    elif suffix == ".nc":
        from netCDF4 import Dataset

        with Dataset(source, "r") as dataset:
            if EMISSION_VARIABLE not in dataset.variables:
                raise ValueError(layered(
                    f"smoke_emission_file {str(path)!r} has no "
                    f"{EMISSION_VARIABLE!r} variable.",
                    f"Variables present: {sorted(dataset.variables)}."))
            variable = dataset.variables[EMISSION_VARIABLE]
            # netCDF4's default unpacking: scale_factor/add_offset applied,
            # _FillValue/missing_value masked.  A masked cell has no
            # emission value at all, so it is refused rather than filled.
            raw = variable[...]
            if np.ma.is_masked(raw):
                raise ValueError(layered(
                    f"smoke_emission_file {str(path)!r} has "
                    f"{int(np.ma.count_masked(raw))} missing (_FillValue) "
                    f"cells in {EMISSION_VARIABLE}.",
                    "A missing cell is not a zero emission; write 0 where "
                    "nothing burns."))
            values = np.asarray(np.ma.getdata(raw))
            units = str(getattr(variable, "units", EMISSION_UNITS)).strip()
        if units in ("ug m-2 s-1", "ug/m2/s", "µg m-2 s-1"):
            scale = 1.0 / UG_PER_KG
        elif units not in ("kg m-2 s-1", "kg/m2/s"):
            raise ValueError(layered(
                f"smoke_emission_file {str(path)!r} gives "
                f"{EMISSION_VARIABLE} in {units!r}; use 'kg m-2 s-1' or "
                "'ug m-2 s-1'.",
                "Any other unit would be read as one of these two and "
                "scale the source by an unknown factor."))
    else:
        raise ValueError(
            f"smoke_emission_file {str(path)!r} must be a .npy or .nc file")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.shape != tuple(shape):
        raise ValueError(layered(
            f"smoke_emission_file {str(path)!r} holds a {values.shape} "
            f"field; this domain's mass grid is (ny, nx) = {tuple(shape)}.",
            "The emission is a field on ONE domain's grid.  On a tree, give "
            "each domain its own file with [[domain]] smoke_emission_file "
            "(an empty string emits nothing on that domain)."))
    values = values * scale
    if not np.all(np.isfinite(values)) or bool((values < 0.0).any()):
        raise ValueError(
            f"smoke_emission_file {str(path)!r} must be finite and "
            "non-negative everywhere: a negative source would remove smoke "
            "the positive-definite transport cannot account for")
    return values.astype(DTYPE)


def bind_emission(state, cfg) -> None:
    """Load ``cfg.smoke_emission_file`` into ``state.smoke_emis`` once.

    Idempotent: the file named in the configuration is read and uploaded
    to the device the first time a state steps with it and never again in
    that process, so no step after the first copies anything for it.  A
    resumed state already holds the serialized field and re-reads the same
    file to the same words -- the forecast and restart identities bind the
    file's SHA-256 (:func:`emission_file_digest`), so a file changed under
    a checkpoint is refused rather than re-read.  An empty path binds
    nothing: that domain emits nothing.
    """
    path = getattr(cfg, "smoke_emission_file", "") or ""
    target = getattr(state, "smoke_emis", None)
    if not path or target is None:
        return
    if getattr(state, _BOUND_ATTRIBUTE, None) == path:
        return
    import cupy as cp

    target[...] = cp.asarray(load_emission(path, target.shape))
    setattr(state, _BOUND_ATTRIBUTE, path)


def emission_file_digest(path: str | Path) -> str:
    """The SHA-256 of an emission file's bytes, or ``"missing"``.

    Bound into the forecast and restart identities beside the path, so the
    identity names the source the run integrated and not only where it was.
    A file that is not there is recorded as such; the tracer's first step
    refuses it by name (:func:`load_emission`).
    """
    import hashlib

    source = Path(path).expanduser()
    if not source.is_file():
        return "missing"
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


#: The key the emission file's SHA-256 is bound under, beside its path, in
#: the forecast identity, the checkpoint echo and the checkpoint digest.
EMISSION_IDENTITY_KEY = "smoke_emission_sha256"
_DIGESTS: dict[tuple, str] = {}


def bind_emission_identity(values) -> dict:
    """Add the emission file's SHA-256 to a config document, in place.

    The rrtmg_smoke_identity rule: empty is absent, and a saved digest stays
    saved -- digesting a stored checkpoint must not replace the bytes it ran
    with by the file now at that path, which is what lets a resume refuse a
    file changed under it.  The digest is cached by path, size and mtime, so
    a checkpoint write does not re-read an unchanged file.
    """
    path = values.get("smoke_emission_file", "") or ""
    if not path:
        values.pop("smoke_emission_file", None)
        values.pop(EMISSION_IDENTITY_KEY, None)
        return values
    if EMISSION_IDENTITY_KEY not in values:
        # The loaders anchor a relative path beside its config; whatever
        # reaches here is keyed by the file it resolves to now, so a
        # changed working directory cannot reuse another file's digest.
        source = Path(path).expanduser().resolve()
        try:
            stat = source.stat()
            key = (str(source), stat.st_size, stat.st_mtime_ns)
        except OSError:
            key = None
        digest = _DIGESTS.get(key) if key is not None else None
        if digest is None:
            digest = emission_file_digest(source)
            if key is not None:
                _DIGESTS[key] = digest
        values[EMISSION_IDENTITY_KEY] = digest
    return values


def module_source() -> str:
    """The exact source string NVRTC compiles (``kernels/smoke_tracer.cu``)."""
    from woof.core.kernels import _preamble

    root = Path(__file__).parent / "kernels"
    return _preamble() + (root / _SOURCE).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _module(device: int):
    """NVRTC directly, -fmad=false and --ftz=false, one module per device.

    The woof.core.terrain_drag loader, for its reason: ``cupy.RawModule``
    appends ``-ftz=true`` after the caller's options, and CuPy array maths
    may contract or flush, so neither holds the words the NumPy references
    below produce.
    """
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler

    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile

    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(
            source, MODULE_OPTIONS, None, _SOURCE)
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS,
                  module=None)
    return module


def _kernel(name: str):
    import cupy as cp

    return _module(cp.cuda.Device().id).get_function(name)


def _blocks(n: int) -> tuple[int]:
    return ((n + _BLOCK - 1) // _BLOCK,)


def add_surface_source(state, tend) -> None:
    """Add the emission's coupled tendency to the lowest layer of ``tend``.

    A flux ``F`` (kg m-2 s-1) into a layer holding ``C(mu)*|dnw|/g`` kg of
    dry air per square metre raises its mixing ratio at
    ``F*1e9*g / (C(mu)*|dnw|)`` ug kg-1 s-1, so the COUPLED tendency the RK
    update divides by ``C(mu)`` is ``F*1e9*g/|dnw|``: independent of the
    column mass, and exactly mass-conserving.  The map factor cancels: the
    flux and the layer mass are both per unit physical area.  One launch
    of ``smoke_surface_source``; :func:`surface_source_reference` is its
    word-exact NumPy twin.
    """
    emis = getattr(state, "smoke_emis", None)
    if emis is None:
        return
    ncol = emis.size
    _kernel("smoke_surface_source")(
        _blocks(ncol), (_BLOCK,),
        (tend[0], emis, state.dnw, UG_G, np.int32(ncol)))


def surface_source_reference(tend0, emis, dnw) -> np.ndarray:
    """NumPy twin of ``smoke_surface_source``: the new lowest-layer words."""
    tend0 = np.asarray(tend0, dtype=DTYPE)
    emis = np.asarray(emis, dtype=DTYPE)
    depth = -np.asarray(dnw, dtype=DTYPE)[0]
    return tend0 + (emis * UG_G) / depth


def mix_pbl(state, atmosphere, exch_h, delt: float) -> None:
    """Operator-split PBL diffusion of the tracer over one PBL interval.

    WRF's ``diff4d`` local scalar diffusion (module_pbl_driver.F; the
    ``scalar_pblmix`` kernel, woof.core.mynn_scalar_mix_gpu): an implicit
    tridiagonal solve in flux form with a zero surface flux, so a column's
    density-weighted load is conserved and no value turns negative.
    ``exch_h`` is the PBL scheme's heat diffusivity (m2 s-1) at each
    layer's lower interface; ``delt`` is the PBL interval.  Physics runs on
    the time-t state after ``dycore`` saved its RK copy, so the time-t copy
    takes the mixed value too -- every RK stage starts from it.
    """
    import cupy as cp

    from woof.core.mynn_scalar_mix_gpu import scalar_pblmix_columns_cuda

    smoke = state.smoke
    nz, ny, nx = smoke.shape
    ncol = ny * nx
    columns = smoke.reshape(nz, ncol)
    dz = atmosphere["dz"].reshape(nz, ncol)
    rho = atmosphere["rho"].reshape(nz, ncol)
    kh = exch_h.reshape(nz, ncol)
    piece = min(PBL_COLUMN_PIECE, ncol)
    if hasattr(state, "scratch"):
        # The registered work slot (woof.core.preflight prices it), reused
        # by every PBL call rather than allocated per call.
        scratch = state.scratch((piece * (5 * nz + 1),), "smoke_pbl_work")
    else:
        scratch = cp.empty(piece * (5 * nz + 1), dtype=DTYPE)
    for lo in range(0, ncol, piece):
        hi = min(lo + piece, ncol)
        n = hi - lo
        solved, rate = scalar_pblmix_columns_cuda(
            cp.asfortranarray(columns[:, lo:hi].T),
            cp.asfortranarray(dz[:, lo:hi].T),
            cp.asfortranarray(rho[:, lo:hi].T),
            cp.asfortranarray(kh[:, lo:hi].T),
            DTYPE(delt),
            scratch=scratch[:n * (5 * nz + 1)].reshape(
                (n, 5 * nz + 1), order="F"))
        columns[:, lo:hi] = solved.T
        del solved, rate
    smoke0 = getattr(state, "smoke0", None)
    if smoke0 is not None:
        smoke0[...] = smoke


def refresh_column_load(state) -> None:
    """``smoke_col = sum_k smoke*C(mu)*|dnw|/g``, ug m-2, in place.

    One ``smoke_column_load`` launch, one thread per column summing upward
    from the lowest layer: a fixed order shared with
    :func:`column_load_reference`, no atomics, no tree.
    """
    col = getattr(state, "smoke_col", None)
    if col is None:
        return
    nz = state.smoke.shape[0]
    ncol = col.size
    _kernel("smoke_column_load")(
        _blocks(ncol), (_BLOCK,),
        (col, state.smoke, state.c1h, state.c2h, state.dnw, state.mub2d,
         state.mup, G, np.int32(nz), np.int32(ncol)))


def column_load_reference(smoke, c1h, c2h, dnw, mub, mup) -> np.ndarray:
    """NumPy twin of ``smoke_column_load``, in its summation order."""
    smoke = np.asarray(smoke, dtype=DTYPE)
    nz = smoke.shape[0]
    c1h, c2h, dnw = (np.asarray(a, dtype=DTYPE) for a in (c1h, c2h, dnw))
    mu = np.asarray(mub, dtype=DTYPE) + np.asarray(mup, dtype=DTYPE)
    acc = np.zeros(mu.shape, dtype=DTYPE)
    for k in range(nz):
        chm = c1h[k] * mu + c2h[k]
        layer = chm * ((-dnw[k]) / G)
        acc = acc + smoke[k] * layer
    return acc


__all__ = ["DTYPE", "EMISSION_UNITS", "EMISSION_VARIABLE", "MODULE_KEY",
           "MODULE_OPTIONS", "PBL_COLUMN_PIECE", "SMOKE_COLUMN_UNITS",
           "SMOKE_UNITS", "UG_G", "UG_PER_KG", "add_surface_source",
           "EMISSION_IDENTITY_KEY", "bind_emission", "bind_emission_identity",
           "column_load_reference", "emission_file_digest",
           "enabled",
           "load_emission", "mix_pbl", "module_source",
           "refresh_column_load", "surface_source_reference"]
