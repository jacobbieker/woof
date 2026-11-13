# WOOF-OPTIN: wrfchem_pnnl
"""Abdul-Razzak and Ghan activation, WRF v4.7.1 ``activate`` (phys/module_mixactivate.F), on device.

``launch_activate`` runs ``activate`` once per case, one thread each, on (MAXD_ASIZE, MAXD_ATYPE) = (8, 2)
arrays (MOSAIC's sections; SORGAM's modes fit), graded word for word against the gfortran -O0 oracle
(oracle/activate, tests/test_wrfchem_pnnl_activate_parity.py).

``erfc_num_recipes`` is Numerical Recipes code and is never shipped, even here: the kernel is built only with
``WOOF_WRF_SOURCE`` naming the user's own WRF v4.7.1 tree, from which ``erfc_num_recipes`` is translated into
a local cache (``$XDG_CACHE_HOME/woof/byo``), as woof/core/chem_byo.py does for SORGAM's solver.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

from . import ROOT, WRF_PINS
from ._compile import raw_module

MAXD_ASIZE, MAXD_ATYPE = 8, 2
#: ain / aout rows, in the kernel's order
INPUTS = ("na", "volc", "dlo_sect", "dhi_sect", "sigman", "hygro")
OUTPUTS = ("fn", "fs", "fm", "fluxn", "fluxs", "fluxm")
_MARK = "// BYO: erfc_num_recipes"
_DEPS = ("glibc_flt32.cuh", "glibc_flt64.cuh", "f64cr.cuh")
_MODULES: dict = {}


class ByoUnavailable(RuntimeError):
    """The user's WRF tree is not available for the Numerical Recipes routine."""


def byo_reason() -> str | None:
    """Why the kernel cannot be built here, or None."""
    value = os.environ.get("WOOF_WRF_SOURCE")
    if not value:
        return ("activate needs erfc_num_recipes (Numerical Recipes), which WOOF never ships: set "
                "WOOF_WRF_SOURCE=/path/to/WRF-4.7.1 to build it from your own tree")
    for rel, digest in WRF_PINS.items():
        path = Path(value) / rel
        if not path.is_file():
            return f"WOOF_WRF_SOURCE={value} has no {rel}"
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != digest:
            return f"WOOF_WRF_SOURCE={value}: {rel} is not WRF v4.7.1's (sha256 {got}, expected {digest})"
    if not (ROOT.parents[1] / "tools" / "chem_f2cu" / "f2cu.py").is_file():
        return "the exact path needs a WOOF source checkout (tools/chem_f2cu)"
    return None


def _transcriber():
    path = ROOT / "tools" / "transcribe_activate.py"
    spec = importlib.util.spec_from_file_location("woof_wrfchem_pnnl_transcribe_activate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _erfc_body() -> str:
    reason = byo_reason()
    if reason is not None:
        raise ByoUnavailable(reason)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    key = hashlib.sha256("".join(WRF_PINS.values()).encode()
                         + (ROOT.parents[1] / "tools" / "chem_f2cu" / "f2cu.py").read_bytes()).hexdigest()[:16]
    cached = Path(base) / "woof" / "byo" / f"mixactivate_erfc-{key}.cu"
    if cached.is_file():
        return cached.read_text()
    body = _transcriber().erfc_source(Path(os.environ["WOOF_WRF_SOURCE"]))
    cached.parent.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(".tmp")
    tmp.write_text(body)
    tmp.replace(cached)
    return body


def _module():
    if "m" not in _MODULES:
        kernels = Path(__import__("woof.core.kernels", fromlist=["x"]).__file__).parent
        text = (Path(__file__).resolve().parent / "activate.cu").read_text(encoding="utf-8")
        lines = text.splitlines()
        if lines.count(_MARK) != 1:
            raise RuntimeError(f"activate.cu must hold the line {_MARK!r} exactly once")
        at = lines.index(_MARK)
        text = "\n".join(lines[:at + 1] + [_erfc_body()] + lines[at + 1:]) + "\n"
        source = "\n".join((kernels / dep).read_text(encoding="utf-8") for dep in _DEPS) + "\n" + text
        _MODULES["m"] = raw_module(source, ("--fmad=false",))
    return _MODULES["m"]


def launch_activate(cases):
    """Run ``activate`` on each case: a dict with wbar, sigw, wdiab, wminf, wmaxf, tair, rhoair, msectional,
    ntype_aer, nsize_aer (2), smax (None = not present) and the (8, 2) arrays INPUTS and OUTPUTS (OUTPUTS and
    flux_fullact are intent(inout): their incoming values stand where activate does not write).  Returns
    ``{name: (ncase, 8, 2) float32}`` for OUTPUTS plus ``flux_fullact`` (ncase,)."""
    import cupy as cp
    n = len(cases)
    fpar = np.zeros((n, 8), np.float32)
    ipar = np.zeros((n, 6), np.int32)
    ain = np.zeros((n, 6, MAXD_ASIZE * MAXD_ATYPE), np.float32)
    aout = np.zeros_like(ain)
    full = np.zeros(n, np.float32)
    for c, case in enumerate(cases):
        smax = case.get("smax")
        fpar[c] = [case[k] for k in ("wbar", "sigw", "wdiab", "wminf", "wmaxf", "tair", "rhoair")] + \
                  [0.0 if smax is None else smax]
        ipar[c] = [case["msectional"], case["ntype_aer"], smax is not None, 0, *case["nsize_aer"]]
        for k, name in enumerate(INPUTS):
            ain[c, k] = np.asarray(case[name], np.float32).reshape(MAXD_ASIZE, MAXD_ATYPE).ravel(order="F")
        for k, name in enumerate(OUTPUTS):
            aout[c, k] = np.asarray(case[name], np.float32).reshape(MAXD_ASIZE, MAXD_ATYPE).ravel(order="F")
        full[c] = case["flux_fullact"]
    d = [cp.asarray(a) for a in (fpar, ipar, ain, aout, full)]
    fatal = cp.zeros(1, cp.int32)
    _module().get_function("wrfchem_pnnl_activate")(
        ((n + 63) // 64,), (64,), (*d, fatal, np.int32(n)))
    if int(fatal.get()[0]):
        raise RuntimeError(f"activate: wrf_error_fatal at phys/module_mixactivate.F:{int(fatal.get()[0])}")
    out = cp.asnumpy(d[3])
    result = {name: out[:, k].reshape(n, MAXD_ATYPE, MAXD_ASIZE).transpose(0, 2, 1)
              for k, name in enumerate(OUTPUTS)}
    result["flux_fullact"] = cp.asnumpy(d[4])
    return result
