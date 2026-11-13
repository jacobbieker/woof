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


#: mixactivate's compile-time aerosol extents (the transcriber's MAXD_* : the caller's descriptor arrays
#: must have these shapes)
MAXD_ACOMP, MAXD_APHASE = 20, 2
#: the per-column fields, in the kernel's order, each (i, k, j) over memory dims
MIX_FIELDS = ("qv", "qc", "qi", "qndrop3d", "temp", "w", "rho", "zm", "dz8w", "p_at_w", "t_at_w", "kvh",
              "cldfra", "cldfra_old")


def launch_mixactivate(dims, desc, fields, *, chem, hygro, ddvel, dtstep, grid_id=1, ktau=1,
                       idrydep_onoff=1, f_qc=True, f_qi=True):
    """WRF's ``mixactivate`` over one tile, one thread per column.

    ``dims``: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte.
    ``desc``: msectional, ntype_aer, nphase_aer, ai_phase, cw_phase, ncomp_aer (2), nsize_aer (2),
    numptr_aer (8, 2, 2), massptr_aer (20, 8, 2, 2), waterptr_aer (8, 2), dlo_sect, dhi_sect, sigmag_aer,
    dpvolmean_aer (8, 2), dens_aer, mw_aer (20, 2).  ``fields``: MIX_FIELDS, NumPy arrays indexed [i, k, j]
    over the memory dims (Fortran's order, as the oracle writes them); ``chem`` [i, k, j, species];
    ``hygro`` [i, k, j, m, n] over the tile; ``ddvel`` [i, j, species] over the tile.
    Returns chem, qndrop3d, cldfra, cldfra_old, qsrflx [i, j, species], ccn [i, k, j, 6], nsource."""
    import cupy as cp
    (ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte) = dims
    ni, nj, nk = ite - its + 1, jte - jts + 1, kte - kts + 1
    num_chem = chem.shape[-1]
    nkm = kme - kms + 1
    ws_per_column = 2 * nkm * num_chem + 64 * nkm + num_chem + 8192
    wsi_per_column = 4096
    ip = np.array([desc["msectional"], num_chem, idrydep_onoff, desc["ntype_aer"], desc["nphase_aer"],
                   desc["ai_phase"], desc["cw_phase"], ids, ide, jds, jde, kds, kde, ims, ime, jms, jme,
                   kms, kme, kts, kte, grid_id, ktau, int(f_qc), int(f_qi), its, jts, ni, ws_per_column,
                   wsi_per_column],
                  np.int32)
    f = lambda a, dt=np.float32: np.asarray(a, dt).ravel(order="F")
    idesc = np.concatenate([f(desc[k], np.int32) for k in
                            ("ncomp_aer", "nsize_aer", "numptr_aer", "massptr_aer", "waterptr_aer")])
    fdesc = np.concatenate([f(desc[k]) for k in
                            ("dlo_sect", "dhi_sect", "sigmag_aer", "dpvolmean_aer", "dens_aer", "mw_aer")])
    assert idesc.size == 4 + 32 + 640 + 16 and fdesc.size == 64 + 80
    mem = (ime - ims + 1, nkm, jme - jms + 1)
    shape = lambda a: tuple(np.shape(a))
    for name in MIX_FIELDS:
        if shape(fields[name]) != mem:
            raise ValueError(f"{name} is {shape(fields[name])}, the memory dims are {mem}")
    columns = [(i, j) for j in range(nj) for i in range(ni)]
    hyg = np.concatenate([np.asarray(hygro, np.float32)[i, :, j, :, :].ravel(order="F") for i, j in columns])
    ddv = np.concatenate([np.asarray(ddvel, np.float32)[i, j, :] for i, j in columns])
    d = {name: cp.asarray(f(fields[name])) for name in MIX_FIELDS}
    out_like = lambda: cp.full(int(np.prod(mem)), -9.0, cp.float32)
    chem_d = cp.asarray(f(chem))
    qsrflx = cp.full(mem[0] * mem[2] * num_chem, -9.0, cp.float32)
    ccn = [out_like() for _ in range(6)]
    nsource = out_like()
    ws = cp.zeros(len(columns) * ws_per_column, cp.float32)
    wsi = cp.zeros(len(columns) * wsi_per_column, cp.int32)
    fatal = cp.zeros(1, cp.int32)
    _module().get_function("wrfchem_pnnl_mixactivate")(
        ((len(columns) + 31) // 32,), (32,),
        (cp.asarray(ip), cp.asarray(idesc), cp.asarray(fdesc), chem_d, d["qv"], d["qc"], d["qi"],
         d["qndrop3d"], d["temp"], d["w"], d["rho"], d["zm"], d["dz8w"], d["p_at_w"], d["t_at_w"], d["kvh"],
         d["cldfra"], d["cldfra_old"], qsrflx, *ccn, nsource, cp.asarray(hyg), cp.asarray(ddv), ws, wsi,
         np.float32(dtstep), fatal, np.int32(len(columns))))
    code = int(fatal.get()[0])
    if code:
        raise RuntimeError("mixactivate: workspace overrun" if code < 0 else
                           f"mixactivate: wrf_error_fatal at line {code}")
    back = lambda a, s: cp.asnumpy(a).reshape(s, order="F")
    return {"chem": back(chem_d, mem + (num_chem,)), "qndrop3d": back(d["qndrop3d"], mem),
            "cldfra": back(d["cldfra"], mem), "cldfra_old": back(d["cldfra_old"], mem),
            "qsrflx": back(qsrflx, (mem[0], mem[2], num_chem)),
            "ccn": np.stack([back(a, mem) for a in ccn], axis=-1), "nsource": back(nsource, mem)}
