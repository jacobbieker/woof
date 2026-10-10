"""ctypes seam onto the Rust wrfout site sampler (``rw-sitesample``).

The library is ``tools/rustwx/crates/rw-sitesample``.  It holds every
per-cell operation :func:`woof.energy.sample.sample_wrfout` performs on a
history window: destaggering, bilinear interpolation at fractional
mass-point indices, height above model terrain from ``PH + PHB``, linear
interpolation in height and the earth-relative wind rotation.  WOOF's Python
boundary (``docs/dev/static-rust-port.md``) puts that arithmetic in Rust, so
there is no Python implementation behind this seam and no fallback to one.

Same loading discipline as :mod:`woof.isobaric_bridge` and
:mod:`woof.obs_score_bridge`: ctypes over a cdylib, an environment override
first (:data:`SITESAMPLE_BRIDGE_ENV`), then the checkout's
``tools/rustwx/target`` and the staged bridge directories, an ABI version
probe at load, and the library's own error message on refusal.

Fields cross as C-contiguous float32 (the history's own precision, so a
window read crosses without a copy); indices, heights and outputs cross as
float64.  Every function takes the window's MASS dimensions from the arrays
it is given and refuses inconsistent shapes before calling the library.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Final

import numpy as np

#: The C ABI version this module speaks; a library reporting anything else
#: is refused rather than called.
SITESAMPLE_ABI: Final[int] = 1

#: Environment override for the library path, first rung of the ladder.
SITESAMPLE_BRIDGE_ENV: Final[str] = "WOOF_SITESAMPLE_BRIDGE"

#: The exported symbol that identifies THIS contract, for
#: :data:`woof.bridges.BRIDGE_ABI_MARKERS`.
ABI_MARKER: Final[bytes] = b"gpuwm_sitesample_wind_profile"

#: The crate, for build hints.
CRATE: Final[str] = "rw-sitesample"

#: Stagger codes of ``gpuwm_sitesample_profile``.
STAGGER_MASS, STAGGER_X, STAGGER_Y, STAGGER_Z = 0, 1, 2, 3


class SiteSampleBridgeMissing(FileNotFoundError):
    """The ``librw_sitesample`` library is not built or staged here."""


class SiteSampleBridgeError(RuntimeError):
    """The site sampler library is incompatible or refused a call."""


def library_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("rw_sitesample.dll",)
    if os.uname().sysname == "Darwin":  # pragma: no cover - platform route
        return ("librw_sitesample.dylib",)
    return ("librw_sitesample.so",)


def library_candidates() -> tuple[Path, ...]:
    """Deterministic candidate paths, best first (the obs-score ladder)."""
    from woof.bridges import (default_bridge_dir, legacy_bridge_candidates,
                               packaged_bridge_dir)
    from woof.rustwx import crate_dir

    filename = library_names()[0]
    candidates: list[Path] = []
    override = os.environ.get(SITESAMPLE_BRIDGE_ENV)
    if override:
        candidates.append(Path(override))
    # The libexec rung of the sibling seams (woof/isobaric_bridge.py,
    # woof/obs_score_bridge.py climb four parents from one level shallower).
    root = Path(__file__).resolve().parents[4]
    candidates.extend((
        crate_dir() / "target" / "release" / filename,
        crate_dir() / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
        *legacy_bridge_candidates(filename),
    ))
    return tuple(candidates)


def build_hint() -> str:
    from woof import bridges

    separator = ";" if bridges.WINDOWS_SHELL else " &&"
    return ("  # build it from a checkout (all native libraries):\n"
            "  pixi run build-native\n"
            "  # or just this one:\n"
            f"  cd tools/rustwx{separator} cargo build --release --locked "
            f"--offline -p {CRATE}{separator} cd ../..\n"
            "  # or stage the published bundle:\n"
            "  woof fetch-bridges")


def resolve_sitesample_bridge() -> Path:
    """First existing candidate, or :class:`SiteSampleBridgeMissing`."""
    override = os.environ.get(SITESAMPLE_BRIDGE_ENV)
    for candidate in library_candidates():
        if candidate.is_file():
            from woof.bridges import accept_resolved

            return accept_resolved(candidate.resolve(), executable=False)
        if override and candidate == Path(override):
            raise SiteSampleBridgeMissing(
                f"{SITESAMPLE_BRIDGE_ENV} names a missing file: {candidate}\n"
                + build_hint())
    rendered = "\n  ".join(str(c) for c in library_candidates())
    raise SiteSampleBridgeMissing(
        "the Rust site sampler library (librw_sitesample) was not found; "
        "woof energy samples wrfout history at sites only through it. "
        "Searched:\n  " + rendered + "\n" + build_hint())


_LIBRARY: ctypes.CDLL | None = None


def unavailable_reason() -> str | None:
    """Why the Rust sampler is not loadable here, or None when it is."""
    try:
        load()
    except (FileNotFoundError, OSError, RuntimeError) as error:
        # RuntimeError covers this module's ABI refusal and woof.bridges'
        # stale-checkout-build and release-pin refusals.
        return f"{type(error).__name__}: {error}"
    return None


def _export(library: ctypes.CDLL, name: str, path: Path):
    try:
        return getattr(library, name)
    except AttributeError as error:
        raise SiteSampleBridgeError(
            f"{path} is missing {name}; rebuild tools/rustwx "
            f"(cargo build --release -p {CRATE})") from error


def load() -> ctypes.CDLL:
    """Load the library once, check its ABI and bind every signature."""
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    path = resolve_sitesample_bridge()
    library = ctypes.CDLL(str(path))
    probe = _export(library, "gpuwm_sitesample_abi_version", path)
    probe.argtypes = []
    probe.restype = ctypes.c_uint32
    observed = int(probe())
    if observed != SITESAMPLE_ABI:
        raise SiteSampleBridgeError(
            f"{path} speaks site-sample ABI {observed}, this woof needs "
            f"{SITESAMPLE_ABI}; rebuild tools/rustwx")

    f32p = ctypes.POINTER(ctypes.c_float)
    f64p = ctypes.POINTER(ctypes.c_double)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    size = ctypes.c_size_t
    f64 = ctypes.c_double
    u32 = ctypes.c_uint32
    signatures = {
        "inside": [f64p, f64p, size, size, size, u8p],
        "heights_agl": [f32p, f32p, f32p, size, size, size, f64, f64p, f64p,
                        size, f64p],
        "profile": [f32p, f32p, f64, f64, u32, size, size, size, f64p, f64p,
                    size, f64p, f64p, size, f64p],
        "wind_profile": [f32p, f32p, f32p, f32p, size, size, size, f64p, f64p,
                         size, f64p, f64p, size, f64p, f64p],
        "surface": [f32p, size, size, f64p, f64p, size, f64p],
        "wind_surface": [f32p, f32p, f32p, f32p, size, size, f64p, f64p, size,
                         f64p, f64p],
    }
    for name, argtypes in signatures.items():
        function = _export(library, "gpuwm_sitesample_" + name, path)
        function.argtypes = argtypes
        function.restype = ctypes.c_int32
    last_error = _export(library, "gpuwm_sitesample_last_error", path)
    last_error.argtypes = [u8p, size]
    last_error.restype = size
    _LIBRARY = library
    return library


def _check(library: ctypes.CDLL, status: int) -> None:
    if status == 0:
        return
    length = int(library.gpuwm_sitesample_last_error(None, 0))
    if length == 0:
        raise SiteSampleBridgeError(
            "the site sampler reported failure without a message")
    buffer = (ctypes.c_uint8 * length)()
    library.gpuwm_sitesample_last_error(
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)), length)
    raise SiteSampleBridgeError(bytes(buffer).decode("utf-8", "replace"))


def _f32(array) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array, dtype=np.float32))


def _f64(array) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array, dtype=np.float64))


def _ptr(array: np.ndarray | None):
    if array is None:
        return None
    ctype = {np.dtype("float32"): ctypes.c_float,
             np.dtype("float64"): ctypes.c_double,
             np.dtype("uint8"): ctypes.c_uint8}[array.dtype]
    return array.ctypes.data_as(ctypes.POINTER(ctype))


def _sites(fi, fj) -> tuple[np.ndarray, np.ndarray, int]:
    fi, fj = _f64(fi), _f64(fj)
    if fi.ndim != 1 or fi.shape != fj.shape:
        raise SiteSampleBridgeError(
            f"site indices must be two 1-D arrays of one length, got "
            f"{fi.shape} and {fj.shape}")
    return fi, fj, int(fi.size)


def _plane(name: str, array, shape: tuple[int, int]) -> np.ndarray:
    plane = _f32(array)
    if plane.shape != shape:
        raise SiteSampleBridgeError(
            f"{name} plane has shape {plane.shape}, expected {shape}")
    return plane


def _mass_dims(stagger: int, shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise SiteSampleBridgeError(
            f"a profile field must be 3-D (level, row, column), got {shape}")
    nz, ny, nx = shape
    if stagger == STAGGER_X:
        nx -= 1
    elif stagger == STAGGER_Y:
        ny -= 1
    elif stagger == STAGGER_Z:
        nz -= 1
    elif stagger != STAGGER_MASS:
        raise SiteSampleBridgeError(f"unknown stagger code {stagger}")
    return nz, ny, nx


def inside(fi, fj, ny: int, nx: int) -> np.ndarray:
    """Which sites have a bilinear stencil on an ``ny`` by ``nx`` mass grid."""
    fi, fj, ns = _sites(fi, fj)
    out = np.zeros(ns, dtype=np.uint8)
    library = load()
    _check(library, library.gpuwm_sitesample_inside(
        _ptr(fi), _ptr(fj), ns, int(ny), int(nx), _ptr(out)))
    return out.astype(bool)


def heights_agl(ph, phb, hgt, fi, fj, gravity: float) -> np.ndarray:
    """Mass-level heights above terrain at each site, shape ``(sites, nz)``."""
    ph, phb = _f32(ph), _f32(phb)
    if ph.ndim != 3 or ph.shape != phb.shape:
        raise SiteSampleBridgeError(
            f"PH {ph.shape} and PHB {phb.shape} must be one 3-D "
            "bottom_top_stag window")
    nz, ny, nx = _mass_dims(STAGGER_Z, ph.shape)
    hgt = _plane("HGT", hgt, (ny, nx))
    fi, fj, ns = _sites(fi, fj)
    out = np.empty((ns, nz), dtype=np.float64)
    library = load()
    _check(library, library.gpuwm_sitesample_heights_agl(
        _ptr(ph), _ptr(phb), _ptr(hgt), nz, ny, nx, float(gravity),
        _ptr(fi), _ptr(fj), ns, _ptr(out)))
    return out


def _heights(zagl, ns: int, nz: int, heights) -> tuple[np.ndarray, np.ndarray]:
    zagl = _f64(zagl)
    if zagl.shape != (ns, nz):
        raise SiteSampleBridgeError(
            f"site heights have shape {zagl.shape}, expected {(ns, nz)}")
    heights = _f64(heights)
    if heights.ndim != 1:
        raise SiteSampleBridgeError("requested heights must be 1-D")
    return zagl, heights


def profile(field, fi, fj, zagl, heights, *, stagger: int = STAGGER_MASS,
            plus=None, scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """``scale * (field + plus) + offset`` at sites and heights, ``(S, H)``."""
    field = _f32(field)
    nz, ny, nx = _mass_dims(stagger, field.shape)
    if plus is not None:
        plus = _f32(plus)
        if plus.shape != field.shape:
            raise SiteSampleBridgeError(
                f"added field {plus.shape} does not match {field.shape}")
    fi, fj, ns = _sites(fi, fj)
    zagl, heights = _heights(zagl, ns, nz, heights)
    out = np.empty((ns, heights.size), dtype=np.float64)
    library = load()
    _check(library, library.gpuwm_sitesample_profile(
        _ptr(field), _ptr(plus), float(scale), float(offset), int(stagger),
        nz, ny, nx, _ptr(fi), _ptr(fj), ns, _ptr(zagl), _ptr(heights),
        heights.size, _ptr(out)))
    return out


def wind_profile(u, v, sinalpha, cosalpha, fi, fj, zagl, heights
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Earth-relative ``(U, V)`` at sites and heights, each ``(S, H)``."""
    u, v = _f32(u), _f32(v)
    nz, ny, nx = _mass_dims(STAGGER_X, u.shape)
    if _mass_dims(STAGGER_Y, v.shape) != (nz, ny, nx):
        raise SiteSampleBridgeError(
            f"U window {u.shape} and V window {v.shape} are not one grid")
    sinalpha = _plane("SINALPHA", sinalpha, (ny, nx))
    cosalpha = _plane("COSALPHA", cosalpha, (ny, nx))
    fi, fj, ns = _sites(fi, fj)
    zagl, heights = _heights(zagl, ns, nz, heights)
    out_u = np.empty((ns, heights.size), dtype=np.float64)
    out_v = np.empty((ns, heights.size), dtype=np.float64)
    library = load()
    _check(library, library.gpuwm_sitesample_wind_profile(
        _ptr(u), _ptr(v), _ptr(sinalpha), _ptr(cosalpha), nz, ny, nx,
        _ptr(fi), _ptr(fj), ns, _ptr(zagl), _ptr(heights), heights.size,
        _ptr(out_u), _ptr(out_v)))
    return out_u, out_v


def surface(field, fi, fj) -> np.ndarray:
    """A mass-plane field at each site, ``(S,)``."""
    field = _f32(field)
    if field.ndim != 2:
        raise SiteSampleBridgeError(
            f"a surface field must be 2-D, got {field.shape}")
    ny, nx = field.shape
    fi, fj, ns = _sites(fi, fj)
    out = np.empty(ns, dtype=np.float64)
    library = load()
    _check(library, library.gpuwm_sitesample_surface(
        _ptr(field), ny, nx, _ptr(fi), _ptr(fj), ns, _ptr(out)))
    return out


def wind_surface(u10, v10, sinalpha, cosalpha, fi, fj
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Earth-relative ``(U10, V10)`` at each site, each ``(S,)``."""
    u10 = _f32(u10)
    if u10.ndim != 2:
        raise SiteSampleBridgeError(f"U10 must be 2-D, got {u10.shape}")
    shape = u10.shape
    v10 = _plane("V10", v10, shape)
    sinalpha = _plane("SINALPHA", sinalpha, shape)
    cosalpha = _plane("COSALPHA", cosalpha, shape)
    fi, fj, ns = _sites(fi, fj)
    out_u = np.empty(ns, dtype=np.float64)
    out_v = np.empty(ns, dtype=np.float64)
    library = load()
    _check(library, library.gpuwm_sitesample_wind_surface(
        _ptr(u10), _ptr(v10), _ptr(sinalpha), _ptr(cosalpha), *shape,
        _ptr(fi), _ptr(fj), ns, _ptr(out_u), _ptr(out_v)))
    return out_u, out_v
