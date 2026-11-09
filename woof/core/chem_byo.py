"""Bring-your-own-source builds of WRF-Chem code WOOF may not redistribute.

Some WRF-Chem routines carry terms that forbid redistribution outright, so
neither WOOF's repository nor its wheels contain them or anything derived
from them, opt-in or not:

* SORGAM's SOA Newton solver (chem/module_aerosols_sorgam.F ``newt``,
  ``lnsrch``, ``fmin``, ``ludcmp``, ``lubksb``), each headed "Adopted from
  Numerical Recipes in FORTRAN, 2nd ed.";
* ISORROPIA (chem/isocom.F, isofwd.F, isorev.F), "COPYRIGHT 1996-2006,
  UNIVERSITY OF MIAMI, CARNEGIE MELLON UNIVERSITY, GEORGIA INSTITUTE OF
  TECHNOLOGY", with no grant.

The exact path is built on the user's machine from the user's own WRF v4.7.1
checkout: set ``WOOF_WRF_SOURCE=/path/to/WRF`` in a WOOF source checkout and
the translator (tools/chem_f2cu, tools/chem_sorgam) turns the pinned files
into a kernel in a local cache (``$XDG_CACHE_HOME/woof/byo`` or
``~/.cache/woof/byo``).  A tree whose files do not match the pinned digests
is refused, naming the file.  The default build never needs any of this: it
runs WOOF's clean-room solver and SORGAM's RPMARES equilibrium.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

ENV = "WOOF_WRF_SOURCE"
ROOT = Path(__file__).resolve().parents[2]
_TOOLS = ROOT / "tools"
#: WRF v4.7.1 (f52c197e) files the exact SORGAM build reads (each byte-identical
#: in v4.6.1, so either tree builds).
SORGAM_PINS = {
    "chem/module_aerosols_sorgam.F": "20c5ac49779c7df414d46e8d8c44982f4bcac1341f24d11cf1c3f30b874bf13a",
    "chem/module_data_sorgam.F": "51ab49d24e570f5565fbc0cb137df566e6403205d885fc7c1fae8999ca010b9b",
    "chem/module_data_radm2.F": "101beef4c417aa35e4526081aa016be60f2ef777cf5b0d574006ce2d041c723f",
}


class ByoUnavailable(RuntimeError):
    """The bring-your-own-source build cannot run here (the reason names why)."""


def wrf_source() -> Path | None:
    value = os.environ.get(ENV)
    return Path(value) if value else None


def unavailable_reason(pins: dict[str, str] = SORGAM_PINS) -> str | None:
    """Why the exact build would refuse, or None when it would build."""
    src = wrf_source()
    if src is None:
        return (f"the exact path builds from your own WRF v4.7.1 checkout: set "
                f"{ENV}=/path/to/WRF (code under terms that forbid "
                "redistribution is never shipped with WOOF)")
    if not (_TOOLS / "chem_f2cu" / "f2cu.py").is_file():
        return ("the exact path needs a WOOF source checkout (tools/chem_f2cu); "
                "an installed wheel does not carry the translator")
    for rel, digest in pins.items():
        path = src / rel
        if not path.is_file():
            return f"{ENV}={src} has no {rel}"
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != digest:
            return (f"{ENV}={src}: {rel} is not WRF v4.7.1's (sha256 {got}, "
                    f"expected {digest})")
    return None


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "woof" / "byo"


def _transcriber():
    path = _TOOLS / "chem_sorgam" / "transcribe_sorgam.py"
    spec = importlib.util.spec_from_file_location("woof_byo_transcribe_sorgam", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def exact_sorgam_source() -> str:
    """chem_sorgam's assembled source with WRF's own solver instead of
    WOOF's clean-room one, generated from the user's tree and cached."""
    reason = unavailable_reason()
    if reason is not None:
        raise ByoUnavailable(reason)
    from woof.core.kernels import module_source
    base = module_source("chem_sorgam")
    key = hashlib.sha256((base + "".join(SORGAM_PINS.values())).encode()).hexdigest()[:16]
    cached = _cache_dir() / f"chem_sorgam_exact-{key}.cu"
    if cached.is_file():
        return cached.read_text()
    block = _transcriber().generate(wrf_source(), exact=True)
    start = base.index("// BEGIN GENERATED")
    end = base.index("// END GENERATED") + len("// END GENERATED\n")
    text = base[:start] + block + base[end:]
    s = text.index("// BEGIN CLEAN-ROOM SOLVER")
    e = text.index("// END CLEAN-ROOM SOLVER") + len("// END CLEAN-ROOM SOLVER\n")
    text = text[:s] + text[e:]
    cached.parent.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(cached)
    return text


_MODULE = None


def exact_sorgam_kernel():
    """The exact chem_sorgam kernel function (same arguments as the default)."""
    global _MODULE
    if _MODULE is None:
        from woof import backend
        from woof.core import kernels
        src = exact_sorgam_source()
        options = kernels.module_options("chem_sorgam")
        if backend.active() == "cuda":
            _MODULE = kernels._load_module_preserving_subnormals(
                "chem_sorgam_byo_exact", src, options)
        else:
            _MODULE = kernels._translated_module(
                "chem_sorgam", src, key="chem_sorgam_byo_exact", options=options)
    return _MODULE.get_function("chem_sorgam")
