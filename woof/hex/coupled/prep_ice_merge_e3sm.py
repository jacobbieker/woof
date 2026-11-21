"""E3SM v3.2.0's sea-ice import merge (``prep_ice_merge``) on the GPU, opt-in.

E3SM's coupler (``driver-mct``, cpl7) builds the sea ice's import attribute
vector ``x2i`` from what the ocean (``o2x``), atmosphere (``a2x``), land ice
(``g2x``), waves (``w2x``) and runoff (``r2x``) send on the ice's grid.
:func:`plan` is the first call's bookkeeping (MCT's shared-name indices and
the lookups, with MCT's stops); :func:`merge` is every call's arithmetic,
word for word:

- the fields of the same name copied from ``o2x``, ``a2x``, ``g2x`` and,
  when ``wav_ice_coup`` is ``'twoway'``, ``w2x``, in that order (``x2i`` is
  not zeroed first: a field no sender carries keeps its value);
- ``Faxa_rain`` = ``Faxa_rainc + Faxa_rainl``, ``Faxa_snow`` =
  ``Faxa_snowc + Faxa_snowl`` and ``Fixx_rofi`` = ``Figg_rofi + Firr_rofi``,
  each then multiplied by ``flux_epbalfact`` (two roundings).

This is not called by any default path. ``real(r8)`` only, the coupler's
kind. The water isotopes are refused by name: E3SM's branch for an isotope
runs when ``x2i`` holds its ``Faxa_rain_<iso>`` (``16O``, ``18O``, ``HDO``)
and writes only ``x2i``'s ``Faxa_rain_<iso>`` and ``Faxa_snow_<iso>`` from
``a2x``'s ``Faxa_rainc/rainl/snowc/snowl_<iso>``; it is not ported. An
``x2i`` with ``Faxa_snow_<iso>`` and no ``Faxa_rain_<iso>`` takes only the
copies, as in E3SM.

The kernel (``prep_ice_merge_e3sm.cu``) is held word for word to E3SM's
routine cut verbatim out of ``driver-mct/main/prep_ice_mod.F90`` and run on
the pinned MCT, built with gfortran ``-O0 -ffp-contract=off``
(``tools/e3sm_coupler_oracle``, ``tests/test_e3sm_ice_merge_parity.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import numpy as np

E3SM_TAG = "v3.2.0"
E3SM_COMMIT = "648b167b7f95267a159a9fdee25faf604c2bfc4f"
MCT_COMMIT = "82b0071e69d14330b75d23b0bc68543ebea9aadc"
SOURCE_FILES = ("driver-mct/main/prep_ice_mod.F90:prep_ice_merge",)
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
_BLOCK = 128

#: the attribute vectors prep_ice_merge reads, in its argument order
SOURCES = ("a2x", "o2x", "r2x", "g2x", "w2x")
#: the shared-name copies' order, and the source codes the kernel takes
_COPY_ORDER = (("o2x", 0), ("a2x", 1), ("g2x", 2), ("w2x", 3))
#: the lookups, in the Fortran's order and the kernel's X_* order
_LOOKUPS = (("a2x", "Faxa_snowc"), ("a2x", "Faxa_snowl"), ("a2x", "Faxa_rainc"), ("a2x", "Faxa_rainl"),
            ("g2x", "Figg_rofi"), ("r2x", "Firr_rofi"),
            ("x2i", "Faxa_rain"), ("x2i", "Faxa_snow"), ("x2i", "Fixx_rofi"))
_KERNEL_ORDER = (("a2x", "Faxa_rainc"), ("a2x", "Faxa_rainl"), ("a2x", "Faxa_snowc"), ("a2x", "Faxa_snowl"),
                 ("g2x", "Figg_rofi"), ("r2x", "Firr_rofi"),
                 ("x2i", "Faxa_rain"), ("x2i", "Faxa_snow"), ("x2i", "Fixx_rofi"))
ISOTOPES = ("16O", "18O", "HDO")
WAV_ICE_COUPLINGS = ("none", "oneway", "twoway")


class MergeAbort(RuntimeError):
    """MCT's ``die`` on a field ``mct_aVect_indexRA`` must find; the message
    is MCT's."""


@dataclass(frozen=True)
class Plan:
    """The first call's bookkeeping, as index arrays (0-based)."""

    lists: dict
    copies: tuple  # (source code, source field, x2i field), in the Fortran's order
    explicit: tuple  # the kernel's X_* indices


def plan(lists: dict, *, wav_ice_coup: str = "none") -> Plan:
    """``prep_ice_merge``'s first-call bookkeeping for these field lists.

    ``lists`` maps ``a2x``, ``o2x``, ``r2x``, ``g2x``, ``w2x`` and ``x2i`` to
    their field names in order (MCT's ``rList``); ``wav_ice_coup`` is
    ``seq_flds_mod``'s.
    """
    lists = {k: tuple(lists.get(k, ())) for k in SOURCES + ("x2i",)}
    for vector, names in lists.items():
        if len(set(names)) != len(names):
            raise ValueError(f"{vector} repeats a field name; MCT's lists hold each once")
    if wav_ice_coup not in WAV_ICE_COUPLINGS:
        raise ValueError(f"wav_ice_coup {wav_ice_coup!r}: one of {WAV_ICE_COUPLINGS}")
    found = {}
    for vector, name in _LOOKUPS:
        try:
            found[vector, name] = lists[vector].index(name)
        except ValueError:
            raise MergeAbort(f'm_AttrVect::indexRA_:: FATAL--attribute not found: "{name}"') from None
    # E3SM's gate for each isotope's branch: x2i's Faxa_rain_<iso> (prep_ice_mod.F90's index_x2i_Faxa_rain_<iso> /= 0)
    isotopes = [f"Faxa_rain_{iso}" for iso in ISOTOPES if f"Faxa_rain_{iso}" in lists["x2i"]]
    if isotopes:
        raise NotImplementedError(f"E3SM's water isotopes in x2i ({', '.join(isotopes)}) are not ported")
    position = {name: k for k, name in enumerate(lists["x2i"])}
    copies = []
    for vector, code in _COPY_ORDER:
        if vector == "w2x" and wav_ice_coup != "twoway":
            continue
        copies += [(code, k, position[name]) for k, name in enumerate(lists[vector]) if name in position]
    return Plan(lists=lists, copies=tuple(copies), explicit=tuple(found[key] for key in _KERNEL_ORDER))


def module_source() -> str:
    return files("woof.hex.coupled").joinpath("prep_ice_merge_e3sm.cu").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _module_on(device: int, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(), options, None, "prep_ice_merge_e3sm.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _options() -> tuple[str, ...]:
    from woof.hex import numerics as _hex_numerics
    return _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True, numerics=_hex_numerics._bind())


def _span(a):
    start = int(a.data.ptr)
    return start, start + int(a.nbytes)


def merge(the_plan: Plan, vectors: dict, x2i, *, flux_epbalfact: float) -> None:
    """One ``prep_ice_merge`` call into ``x2i``, in place.

    ``vectors`` maps each of :data:`SOURCES` to its device ``rAttr`` as
    ``(lsize, nRattr)`` float64 (point-major, so ``rAttr(k, n)`` is
    ``[n, k]``); ``x2i`` is ``(lsize, len(the_plan.lists['x2i']))``, a
    separate array.
    """
    import cupy as cp
    if not isinstance(the_plan, Plan):
        raise TypeError("the_plan must be prep_ice_merge_e3sm.plan(...)'s")
    if not isinstance(x2i, cp.ndarray):
        raise TypeError(f"x2i must be a cupy (device) array; it is {type(x2i).__name__}")
    lsize = int(x2i.shape[0]) if x2i.ndim == 2 else -1
    nx = len(the_plan.lists["x2i"])
    if x2i.dtype != np.float64 or x2i.shape != (lsize, nx) or not x2i.flags.c_contiguous:
        raise ValueError(f"x2i must be C-contiguous float64 of shape (lsize, {nx})")
    device = cp.cuda.Device().id
    args = {}
    for vector in SOURCES:
        width = len(the_plan.lists[vector])
        array = vectors.get(vector)
        if width == 0:
            args[vector] = (cp.zeros(1, dtype=np.float64), np.int32(0))
            continue
        if not isinstance(array, cp.ndarray) or array.dtype != np.float64 or array.shape != (lsize, width) \
                or not array.flags.c_contiguous:
            raise ValueError(f"{vector} must be a C-contiguous float64 cupy array of shape ({lsize}, {width})")
        if int(array.device.id) != device or int(x2i.device.id) != device:
            raise ValueError(f"{vector} or x2i is not on the current device {device}")
        if array.size and x2i.size:
            (a0, a1), (b0, b1) = _span(array), _span(x2i)
            if a0 < b1 and b0 < a1:
                raise ValueError(f"{vector} and x2i share memory; prep_ice_merge needs separate arrays")
        args[vector] = (array, np.int32(width))

    def ints(values):
        return cp.asarray(np.array(list(values) or [0], dtype=np.int32))

    copies = the_plan.copies
    if lsize > 0:
        _module_on(device, _options()).get_function("prep_ice_merge_e3sm")(
            ((lsize + _BLOCK - 1) // _BLOCK,), (_BLOCK,), (
                x2i, np.int32(nx), *args["o2x"], *args["a2x"], *args["g2x"], *args["w2x"], *args["r2x"],
                ints(c[0] for c in copies), ints(c[1] for c in copies), ints(c[2] for c in copies),
                np.int32(len(copies)), ints(the_plan.explicit), np.float64(flux_epbalfact), np.int32(lsize)))
