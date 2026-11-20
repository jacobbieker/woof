"""E3SM v3.2.0's coupler surface fractions update on the GPU, opt-in.

E3SM's coupler (``driver-mct/main/seq_frac_mct.F90``) keeps each grid's
surface fractions in an attribute vector (``fractions_a``, ``_i``, ``_o``)
and updates them every coupling step from the sea ice's fraction with
``seq_frac_set``. :func:`seq_frac_set` is that routine, word for word:

- with the sea ice present, ``fractions_i``'s ``ifrac`` is i2x's
  ``Si_ifrac`` times the ice domain's ``frac`` and its ``ofrac`` is ``frac``
  less that; ``seq_frac_check`` tests it ("ice set");
- with the ocean present, ``ofrac`` and ``ifrac`` are mapped to the ocean
  grid through the ice-to-ocean mapper and tested ("ocn set");
- with the atmosphere present, they are mapped to the atmosphere grid
  through the ice-to-atmosphere mapper and tested ("atm set").

A mapper is a copy (``copy_only``), a rearrangement (``rearrange_only``:
the same grid, so on one task a copy; MPAS-Ocean and MPAS-Seaice share a
mesh) or a sparse matrix (``seq_map_avNormArr`` with ``norm=.false.`` over
``mct_sMat_avMult``): see :class:`Mapper`. ``seq_frac_check`` stops the
coupler (``shr_sys_abort``) when a checked fraction is below ``-eps`` or
above ``1 + eps`` (``eps_fracval`` 1e-2) or, on a grid with the land, ice
and ocean fractions, when their sum is more than ``eps_fracsum`` (1e-2) from
1; unless the models are dead (``seq_frac_dead``), when it only reports.
Here that is :class:`FracAbort`, or the stages returned.

Kept as E3SM has it: an i2x without ``Si_ifrac`` (MCT's copy copies
nothing) leaves ``ifrac`` as the vector had it, times ``frac``. Refused: an
ice domain without ``frac`` (E3SM reads ``rAttr(0,:)``, outside the
vector); a copy or rearrangement between grids of different sizes (MCT
stops); a missing ``ifrac`` or ``ofrac`` (MCT stops).

``atm_frac_correct`` is a ``.false.`` parameter in v3.2.0, so its branch is
dead and not ported. The nonlinear maps (``seq_nlmap_mod``) are not reached
(``omit_nonlinear``). NaN inputs are outside the word-for-word claim
(``seq_frac_check``'s ``minval``/``maxval`` and the GPU's comparisons treat
a NaN differently).

This is not called by any default path. ``real(r8)`` only, one task (one
block). The kernels (``seq_frac_e3sm.cu``) are held word for word to E3SM's
``seq_frac_set``, ``seq_frac_check``, ``seq_map_map`` and
``seq_map_avNormArr`` cut verbatim, over the pinned MCT, built with gfortran
``-O0 -ffp-contract=off`` (``tools/e3sm_coupler_oracle``,
``tests/test_e3sm_seq_frac_parity.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files

import numpy as np

E3SM_TAG = "v3.2.0"
E3SM_COMMIT = "648b167b7f95267a159a9fdee25faf604c2bfc4f"
MCT_COMMIT = "82b0071e69d14330b75d23b0bc68543ebea9aadc"
SOURCE_FILES = ("driver-mct/main/seq_frac_mct.F90:seq_frac_set,seq_frac_check",
                "driver-mct/main/seq_map_mod.F90:seq_map_map,seq_map_avNormAvF,seq_map_avNormArr",
                "share/util/mct_mod.F90:mct_rearr_rearrange_fldlist")
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
MODULE_OPTIONS = COMPILE_OPTIONS
_BLOCK = 128

EPS_FRACSUM = 1.0e-02
EPS_FRACVAL = 1.0e-02
#: the fractions seq_frac_check tests the bounds of, in its order
CHECKED = ("afrac", "lfrac", "ofrac", "ifrac", "gfrac", "rfrac", "wfrac", "zfrac", "lfrin")
STAGES = ("ice set", "ocn set", "atm set")
KINDS = ("copy", "rearrange", "matrix")


class FracAbort(RuntimeError):
    """``seq_frac_check`` stopping the coupler (``shr_sys_abort``)."""

    def __init__(self, stage: str):
        super().__init__(f"(seq_frac_check) [{stage}] ERROR aborting")
        self.stage = stage


@dataclass(frozen=True)
class Mapper:
    """A ``seq_map`` as one task holds it.

    ``kind`` is ``"copy"`` (``copy_only``), ``"rearrange"``
    (``rearrange_only``; on one task the identity) or ``"matrix"``: the
    mapping file's elements, ``rows`` and ``cols`` 1-based global indices
    and ``weights``, in the file's order, onto ``n_dst`` points. MCT keeps
    them sorted by row and then column (stable: repeated elements keep the
    file's order), and that is each row's summation order."""

    kind: str
    rows: tuple = ()
    cols: tuple = ()
    weights: tuple = ()
    n_dst: int = 0
    n_src: int = 0
    _csr: tuple = field(default=(), compare=False, repr=False)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown mapper kind {self.kind!r}: one of {KINDS}")
        if self.kind != "matrix":
            return
        rows, cols = np.asarray(self.rows, dtype=np.int64), np.asarray(self.cols, dtype=np.int64)
        w = np.asarray(self.weights, dtype=np.float64)
        if not (rows.ndim == cols.ndim == w.ndim == 1 and rows.size == cols.size == w.size):
            raise ValueError("rows, cols and weights must be 1-D and the same length")
        if self.n_dst < 0 or self.n_src < 0:
            raise ValueError("n_dst and n_src must not be negative")
        if rows.size and (rows.min() < 1 or rows.max() > self.n_dst or cols.min() < 1 or cols.max() > self.n_src):
            raise ValueError("a matrix element's row or column is outside the grids (1-based)")
        order = np.lexsort((cols, rows))  # stable: by row, then column, then the file's order
        start = np.zeros(self.n_dst + 1, dtype=np.int32)
        np.cumsum(np.bincount(rows - 1, minlength=self.n_dst), out=start[1:])
        object.__setattr__(self, "_csr", (start, (cols[order] - 1).astype(np.int32),
                                          np.ascontiguousarray(w[order])))


def module_source() -> str:
    return files("woof.hex.coupled").joinpath("seq_frac_e3sm.cu").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _module_on(device: int, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(), options, None, "seq_frac_e3sm.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _options() -> tuple[str, ...]:
    from woof.hex import numerics as _hex_numerics
    return _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True, numerics=_hex_numerics._bind())


def _launch(kernel: str, n: int, args) -> None:
    import cupy as cp
    if n > 0:
        _module_on(cp.cuda.Device().id, _options()).get_function(kernel)(
            ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), tuple(args))


def _av(name, array, fields, n=None):
    """An attribute vector: a C-contiguous float64 cupy array ``(lsize, nflds)``."""
    import cupy as cp
    if not isinstance(array, cp.ndarray):
        raise TypeError(f"{name} must be a cupy (device) array; it is {type(array).__name__}")
    if array.dtype != np.float64 or not array.flags.c_contiguous or array.ndim != 2:
        raise ValueError(f"{name} must be a C-contiguous float64 (lsize, nflds) array")
    fields = tuple(fields)
    if len(set(fields)) != len(fields) or int(array.shape[1]) != len(fields):
        raise ValueError(f"{name} has {array.shape[1]} fields; its list {fields} must name each once")
    if n is not None and int(array.shape[0]) != n:
        raise ValueError(f"{name} must have {n} points; it has {array.shape[0]}")
    return {f: k for k, f in enumerate(fields)}


def _need(pos, name, av):
    if name not in pos:
        raise ValueError(f'm_AttrVect::indexRA_:: FATAL--attribute not found: "{name}" ({av})')
    return pos[name]


def _check(av, pos, stage, dead, errors):
    import cupy as cp
    n = int(av.shape[0])
    if n == 0:
        return
    k = np.array([pos.get(f, -1) for f in CHECKED], dtype=np.int32)
    bad = cp.zeros(n, dtype=np.int32)
    _launch("seq_frac_check", n, (av, np.int32(av.shape[1]), cp.asarray(k), np.int32(k.size),
                                  np.int32(pos.get("lfrac", -1)), np.int32(pos.get("ofrac", -1)),
                                  np.int32(pos.get("ifrac", -1)), np.float64(EPS_FRACVAL),
                                  np.float64(EPS_FRACSUM), bad, np.int32(n)))
    if bool(bad.any()):
        if not dead:
            raise FracAbort(stage)
        errors.append(stage)


def _map(mapper: Mapper, src, spos, dst, dpos, label):
    import cupy as cp
    ns, nd = int(src.shape[0]), int(dst.shape[0])
    args = (dst, np.int32(dst.shape[1]), np.int32(_need(dpos, "ofrac", label)), np.int32(_need(dpos, "ifrac", label)),
            src, np.int32(src.shape[1]), np.int32(spos["ofrac"]), np.int32(spos["ifrac"]))
    if mapper.kind != "matrix":
        if ns != nd:
            raise ValueError(f"a {mapper.kind} mapper joins grids of one size; {label} has {nd} points, the ice {ns}")
        _launch("seq_frac_copy", nd, args + (np.int32(nd),))
        return
    if mapper.n_dst != nd or mapper.n_src != ns:
        raise ValueError(f"the {label} matrix maps {mapper.n_src} points to {mapper.n_dst}; the grids have "
                         f"{ns} and {nd}")
    start, cols, w = mapper._csr
    _launch("seq_frac_matrix", nd, args + (cp.asarray(start), cp.asarray(cols if cols.size else np.zeros(1, np.int32)),
                                           cp.asarray(w if w.size else np.zeros(1)), np.int32(nd)))


def seq_frac_set(*, i2x, i2x_fields, dom, dom_fields, fractions_i, fi_fields, fractions_o, fo_fields,
                 fractions_a, fa_fields, mapper_i2o: Mapper, mapper_i2a: Mapper, atm_present: bool = True,
                 ice_present: bool = True, ocn_present: bool = True, dead: bool = False) -> list[str]:
    """``seq_frac_set``: updates ``fractions_i``, ``_o`` and ``_a`` in place.

    Each vector is a cupy ``(lsize, nflds)`` float64 array (MCT's
    ``rAttr(nflds, lsize)`` in memory order) with its field list. ``dom`` is
    the ice domain's ``data`` (it must hold ``frac``). Raises
    :class:`FracAbort` where E3SM stops; with ``dead`` returns the stages
    ``seq_frac_check`` reported instead (otherwise ``[]``)."""
    n_i = int(i2x.shape[0]) if hasattr(i2x, "shape") else -1
    ipos = _av("i2x", i2x, i2x_fields)
    dpos = _av("dom", dom, dom_fields, n_i)
    fipos = _av("fractions_i", fractions_i, fi_fields, n_i)
    fopos = _av("fractions_o", fractions_o, fo_fields)
    fapos = _av("fractions_a", fractions_a, fa_fields)
    for m in (mapper_i2o, mapper_i2a):
        if not isinstance(m, Mapper):
            raise TypeError("the mappers must be seq_frac_e3sm.Mapper")
    errors: list[str] = []
    if not ice_present:
        return errors
    # mct_aVect_copy(i2x_i, fractions_i, "Si_ifrac", "ifrac"): nothing without Si_ifrac (kept)
    kif = ipos.get("Si_ifrac", -1)
    ki, ko = _need(fipos, "ifrac", "fractions_i"), _need(fipos, "ofrac", "fractions_i")
    if "frac" not in dpos:
        raise ValueError('the ice domain has no "frac": E3SM\'s indexRA (perrWith) reports it and returns 0, '
                         "and seq_frac_set reads rAttr(0,:), outside the vector")
    kf = dpos["frac"]
    _launch("seq_frac_ice", n_i, (fractions_i, np.int32(len(fipos)), np.int32(ki), np.int32(ko), i2x,
                                  np.int32(len(ipos)), np.int32(kif), dom, np.int32(len(dpos)), np.int32(kf),
                                  np.int32(n_i)))
    _check(fractions_i, fipos, "ice set", dead, errors)
    if ocn_present:
        _map(mapper_i2o, fractions_i, fipos, fractions_o, fopos, "fractions_o")
        _check(fractions_o, fopos, "ocn set", dead, errors)
    if atm_present:
        _map(mapper_i2a, fractions_i, fipos, fractions_a, fapos, "fractions_a")
        _check(fractions_a, fapos, "atm set", dead, errors)
    return errors
