"""ctypes seam to the Rust TEMPO init (tools/rustwx/crates/rw-tempo).

The Rust library is TEMPO's ``tempo_init`` (NCAR/TEMPO 9adf9ef, Apache-2.0):
every parameter, derived constant, bin set and computed lookup table plus
the four table-file readers, producing the float64 blob the GPU TEMPO port
reads.  It replaces the Fortran dump (tools/tempo_ref/tempo_ref_driver.F90),
which stays as the parity reference (tests/test_tempo_rust_init.py).

Library resolution, best first (the static-fields ladder):
``WOOF_TEMPO_BRIDGE`` naming the built file; the checkout's
``tools/rustwx/target/{release,debug}``; ``<root>/libexec/bridges``; the
package's bundled bridges; ``~/.woof/bridges``.  Build from a checkout:
``cd tools/rustwx; cargo build --release -p rw-tempo --offline``.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

TEMPO_BRIDGE_ENV = "WOOF_TEMPO_BRIDGE"
ABI_VERSION = 1

_LIBRARY: ctypes.CDLL | None = None


def library_name() -> str:
    if os.name == "nt":
        return "rw_tempo.dll"
    if os.uname().sysname == "Darwin":  # pragma: no cover - platform route
        return "librw_tempo.dylib"
    return "librw_tempo.so"


def library_candidates() -> tuple[Path, ...]:
    from woof.bridges import default_bridge_dir, packaged_bridge_dir
    from woof.rustwx import crate_dir

    name = library_name()
    out: list[Path] = []
    override = os.environ.get(TEMPO_BRIDGE_ENV)
    if override:
        out.append(Path(override))
    root = Path(__file__).resolve().parent.parent.parent
    out.extend((
        crate_dir() / "target" / "release" / name,
        crate_dir() / "target" / "debug" / name,
        root / "libexec" / "bridges" / name,
        packaged_bridge_dir() / name,
        default_bridge_dir() / name,
    ))
    return tuple(out)


def resolve() -> Path:
    override = os.environ.get(TEMPO_BRIDGE_ENV)
    for candidate in library_candidates():
        if candidate.is_file():
            return candidate.resolve()
        if override and candidate == Path(override):
            raise FileNotFoundError(f"{TEMPO_BRIDGE_ENV} names a missing file: {candidate}")
    rendered = "\n  ".join(str(c) for c in library_candidates())
    raise FileNotFoundError(
        "the Rust TEMPO init library was not found; searched:\n  " + rendered
        + "\n  # build it from a checkout:\n  cd tools/rustwx; cargo build --release -p rw-tempo --offline")


def load() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    lib = ctypes.CDLL(str(resolve()))
    lib.gpuwm_tempo_abi_version.restype = ctypes.c_uint32
    abi = lib.gpuwm_tempo_abi_version()
    if abi != ABI_VERSION:
        raise RuntimeError(f"rw-tempo ABI {abi}, this woof speaks {ABI_VERSION}; rebuild rw-tempo")
    lib.gpuwm_tempo_field_count.restype = ctypes.c_size_t
    lib.gpuwm_tempo_field_len.restype = ctypes.c_uint64
    lib.gpuwm_tempo_field_len.argtypes = (ctypes.c_size_t,)
    lib.gpuwm_tempo_field_name.restype = ctypes.c_size_t
    lib.gpuwm_tempo_field_name.argtypes = (ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t)
    lib.gpuwm_tempo_blob_len.restype = ctypes.c_uint64
    lib.gpuwm_tempo_build_blob.restype = ctypes.c_int32
    lib.gpuwm_tempo_build_blob.argtypes = (
        ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double), ctypes.c_uint64)
    lib.gpuwm_tempo_write_records.restype = ctypes.c_int32
    lib.gpuwm_tempo_write_records.argtypes = (
        ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_char_p, ctypes.c_size_t)
    lib.gpuwm_tempo_last_error.restype = ctypes.c_size_t
    lib.gpuwm_tempo_last_error.argtypes = (ctypes.c_char_p, ctypes.c_size_t)
    _LIBRARY = lib
    return lib


def _last_error(lib) -> str:
    n = lib.gpuwm_tempo_last_error(None, 0)
    buf = ctypes.create_string_buffer(n)
    lib.gpuwm_tempo_last_error(buf, n)
    return buf.raw[:n].decode("utf-8", "replace")


def layout() -> list[tuple[str, int]]:
    """(name, count) of the Rust blob, in blob order."""
    lib = load()
    out = []
    for i in range(lib.gpuwm_tempo_field_count()):
        n = lib.gpuwm_tempo_field_name(i, None, 0)
        buf = ctypes.create_string_buffer(n)
        lib.gpuwm_tempo_field_name(i, buf, n)
        out.append((buf.raw[:n].decode(), int(lib.gpuwm_tempo_field_len(i))))
    return out


def build_blob(table_dir: str | Path, *, hailaware: bool = True,
               refl10cm_from_melting: bool = False,
               max_hail_diameter: bool = True) -> np.ndarray:
    """TEMPO's tempo_init state as the flat float64 blob (TEMPO_FIELDS order).

    ``table_dir`` holds ccn_activate.bin, freeze_water_data_tempo_v3,
    qr_acr_qs_data_tempo_v3 and qr_acr_qg_data_tempo_v3 (for example the
    directory woof.tempo_assets.ensure_tempo_tables() returns).
    """
    lib = load()
    n = int(lib.gpuwm_tempo_blob_len())
    out = np.empty(n, dtype=np.float64)
    path = str(table_dir).encode("utf-8")
    rc = lib.gpuwm_tempo_build_blob(
        path, len(path), int(hailaware), int(refl10cm_from_melting), int(max_hail_diameter),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n)
    if rc != 0:
        raise RuntimeError(f"TEMPO init failed: {_last_error(lib)}")
    return _glibc_libm_fields(out)


#: tempo_init's tables computed with the C library's exp/log/pow/atan
#: (collision efficiencies, drop evaporation, ice autoconversion), as the
#: reference build computes them on glibc 2.44, x86-64: the same words for
#: every init option.  Elsewhere the platform libm moves their last bits
#: (macOS: 2772 words, in these six fields only; tgamma is answered from
#: glibc's words inside rw-tempo), so they are overlaid from the shipped
#: glibc words and the blob is the same on every platform.  The digests are
#: of the glibc fields; a field that already matches is left alone.
GLIBC_LIBM_FIELDS = Path(__file__).resolve().parents[1] / "data" / "tempo" / "glibc-libm-fields.npz"
GLIBC_LIBM_FIELDS_SHA256 = "17460c09cf26dd48ffef096f7934e81b186443bbf410fa4bf85565cff62645fb"
GLIBC_FIELD_SHA256 = {
    "t_efrw": "59fab3b02e25c2dd4f127fa49ee3e71d5792530d747d72cf43f5843f734f25e7",
    "t_efsw": "3e22181d31012b2bb111df37e4ef25ada1bf353aa04e0b98e476d0d2b8ad874e",
    "tpc_wev": "0d706c8d3eba6e205fbd99d4d075a140b0b6237f629ed472040fd51f0bab33ca",
    "tnc_wev": "3798cd80a2f5004c05f1281f6a1a125d8ce5fe85a33f563bb3ecdd020ccd2c12",
    "tps_iaus": "b1ee2adf948d319a044c60b897b3e27b7f2dc050424d2e049f821893880c14b1",
    "tni_iaus": "0f4d1b05538aff8540cb1915a7cfbc1d894ce2d8a315b9e00f684b063f628746",
}


def _glibc_libm_fields(blob: np.ndarray) -> np.ndarray:
    import hashlib
    at, spans = 0, {}
    for name, n in layout():
        if name in GLIBC_FIELD_SHA256:
            spans[name] = (at, n)
        at += n
    stale = [k for k, (a, n) in spans.items()
             if hashlib.sha256(blob[a:a + n].tobytes()).hexdigest() != GLIBC_FIELD_SHA256[k]]
    if not stale:
        return blob
    payload = GLIBC_LIBM_FIELDS.read_bytes()
    if hashlib.sha256(payload).hexdigest() != GLIBC_LIBM_FIELDS_SHA256:
        raise RuntimeError(f"{GLIBC_LIBM_FIELDS} is not the recorded glibc TEMPO fields")
    import io
    with np.load(io.BytesIO(payload)) as words:
        for k in stale:
            a, n = spans[k]
            field = words[k]
            if field.shape != (n,):
                raise RuntimeError(f"glibc TEMPO field {k} has {field.size} words; the blob has {n}")
            blob[a:a + n] = field
    return blob


def write_records(table_dir: str | Path, out_path: str | Path, *, hailaware: bool = True,
                  refl10cm_from_melting: bool = False, max_hail_diameter: bool = True) -> None:
    """Write the blob as the named-record file the Fortran reference writes."""
    lib = load()
    d = str(table_dir).encode("utf-8")
    p = str(out_path).encode("utf-8")
    rc = lib.gpuwm_tempo_write_records(
        d, len(d), int(hailaware), int(refl10cm_from_melting), int(max_hail_diameter), p, len(p))
    if rc != 0:
        raise RuntimeError(f"TEMPO init failed: {_last_error(lib)}")
