"""E3SM v3.2.0's ocean import merge (``prep_ocn_merge``) on the GPU, opt-in.

E3SM's coupler (``driver-mct``, cpl7) builds the ocean's import attribute
vector ``x2o`` from what the atmosphere (``a2x``), sea ice (``i2x``), runoff
(``r2x``), waves (``w2x``), land ice (``g2x``) and the coupler's own
atmosphere/ocean fluxes (``xao``) send on the ocean's grid, weighted by the
ice and open-ocean fractions (``fractions_o``: ``ifrac``/``ofrac``, and the
radiative ``ifrad``/``ofrad``). :func:`plan` is the first call's bookkeeping
(the field matching by the name after the first ``_``, the merge flags and
the shared-name copies, with E3SM's aborts); :func:`merge` is every call's
arithmetic, word for word:

- ``x2o`` zeroed, then the fields of the same name copied from ``a2x``,
  ``i2x``, ``r2x``, ``w2x`` (only when ``wav_ocn_coup`` is ``'twoway'``),
  ``xao`` and ``g2x``, in that order;
- per point, the fractions normalised by their sum where it is nonzero;
- ``Foxx_swnet`` = the open ocean's absorbed shortwave (``xao``'s albedos)
  times ``afracr`` plus ``Fioi_swpen`` times ``ifrac``; under
  ``seq_flds_i2o_per_cat`` also ``Sf_afrac``, ``Sf_afracr`` and
  ``Foxx_swnet_afracr``;
- ``Faxa_snow``/``Faxa_rain`` the convective plus large-scale fall times
  ``afrac`` and ``flux_epbalfact``, ``Faxa_prec`` their sum; ``Foxx_rofl`` and
  ``Foxx_rofi`` the runoff, flood and land-ice runoff times
  ``flux_epbalfact``; the river nutrients copied under ``rof2ocn_nutrients``;
- every other matched flux: ``ifrac`` times the ice's, then ``afrac`` times
  the atmosphere's and the atmosphere/ocean flux's, each added, or replacing
  the field when it is the sender's own (same full name, a flux).

This is not called by any default path: the hex coupled step keeps its
matched exchange (:mod:`woof.hex.coupled.exchange`) unless a caller asks for
this merge. ``real(r8)`` only, the coupler's kind.

The water isotopes (``*_16O``, ``*_18O``, ``*_HDO`` in ``x2o``) are refused by
name: E3SM's isotope branch is not ported. A field list without a ``_`` in a
name, a repeated name, or fractions without ``ifrac``/``ofrac``/``ifrad``/
``ofrad`` is refused too (E3SM reads out of bounds there).

The kernel (``prep_ocn_merge_e3sm.cu``) is held word for word to E3SM's
routine cut verbatim out of ``driver-mct/main/prep_ocn_mod.F90`` and run on
the pinned MCT, built with gfortran ``-O0 -ffp-contract=off``
(``tools/e3sm_coupler_oracle``, ``tests/test_e3sm_ocean_merge_parity.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import numpy as np

E3SM_TAG = "v3.2.0"
E3SM_COMMIT = "648b167b7f95267a159a9fdee25faf604c2bfc4f"
MCT_COMMIT = "82b0071e69d14330b75d23b0bc68543ebea9aadc"
SOURCE_FILES = ("driver-mct/main/prep_ocn_mod.F90:prep_ocn_merge",)
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
MODULE_OPTIONS = COMPILE_OPTIONS
_BLOCK = 128

#: the attribute vectors prep_ocn_merge reads, in its argument order
SOURCES = ("a2x", "i2x", "r2x", "w2x", "g2x", "xao", "fractions")
#: the shared-name copies' order, and the source codes the kernel takes
_COPY_ORDER = (("a2x", 0), ("i2x", 1), ("r2x", 2), ("w2x", 3), ("xao", 4), ("g2x", 5))
ISOTOPES = ("16O", "18O", "HDO")
NUTRIENTS = ("DIN", "DIP", "DON", "DOP", "DOC", "PP", "DSi", "POC", "PN", "DIC", "Fe")
#: the explicit fields, in the kernel's X_* order: (vector, name)
_EXPLICIT = (
    ("a2x", "Faxa_swvdr"), ("a2x", "Faxa_swvdf"), ("a2x", "Faxa_swndr"), ("a2x", "Faxa_swndf"),
    ("i2x", "Fioi_swpen"),
    ("xao", "So_avsdr"), ("xao", "So_anidr"), ("xao", "So_avsdf"), ("xao", "So_anidf"),
    ("a2x", "Faxa_snowc"), ("a2x", "Faxa_snowl"), ("a2x", "Faxa_rainc"), ("a2x", "Faxa_rainl"),
    ("r2x", "Forr_rofl"), ("r2x", "Forr_rofi"), ("r2x", "Flrr_flood"), ("g2x", "Fogg_rofl"), ("g2x", "Fogg_rofi"),
    ("x2o", "Foxx_swnet"), ("x2o", "Faxa_snow"), ("x2o", "Faxa_rain"), ("x2o", "Faxa_prec"),
    ("x2o", "Foxx_rofl"), ("x2o", "Foxx_rofi"),
)
_PER_CAT = (("x2o", "Sf_afrac"), ("x2o", "Sf_afracr"), ("x2o", "Foxx_swnet_afracr"))
_FRACTIONS = ("ifrac", "ofrac", "ifrad", "ofrad")


class MergeAbort(RuntimeError):
    """What E3SM stops on (``shr_sys_abort``, or MCT's ``die`` on a field
    ``mct_aVect_indexRA`` must find); the message is the Fortran's."""


@dataclass(frozen=True)
class Plan:
    """The first call's bookkeeping, as index arrays (0-based, -1 for none)."""

    lists: dict
    copies: tuple  # (source code, source field, x2o field), in the Fortran's order
    explicit: tuple  # the kernel's X_* indices
    nutrient_from: tuple
    nutrient_to: tuple
    aindx: tuple
    iindx: tuple
    xindx: tuple
    amerge: tuple
    imerge: tuple
    xmerge: tuple
    nutrients: bool
    per_cat: bool


def _itemc(name: str) -> str:
    return name[name.index("_"):]


def _require(lists: dict, vector: str, name: str) -> int:
    try:
        return lists[vector].index(name)
    except ValueError:
        raise MergeAbort(f'm_AttrVect::indexRA_:: FATAL--attribute not found: "{name}"') from None


def plan(lists: dict, *, nutrients: bool = False, per_cat: bool = False, wav_ocn_coup: str = "none") -> Plan:
    """``prep_ocn_merge``'s first-call bookkeeping for these field lists.

    ``lists`` maps ``a2x``, ``i2x``, ``r2x``, ``w2x``, ``g2x``, ``xao``,
    ``fractions`` and ``x2o`` to their field names in order (MCT's ``rList``).
    ``nutrients`` is ``rof2ocn_nutrients``, ``per_cat``
    ``seq_flds_i2o_per_cat``; ``wav_ocn_coup`` is ``seq_flds_mod``'s.
    """
    lists = {k: tuple(lists.get(k, ())) for k in SOURCES + ("x2o",)}
    for vector, names in lists.items():
        if len(set(names)) != len(names):
            raise ValueError(f"{vector} repeats a field name; MCT's lists hold each once")
        if vector != "fractions" and any("_" not in n for n in names):
            raise ValueError(f"{vector} has a field name without '_', which prep_ocn_merge cannot split")
    x2o = lists["x2o"]
    isotopes = [n for n in x2o for iso in ISOTOPES if n.endswith("_" + iso)
                and n.rsplit("_", 1)[0] in ("Faxa_rain", "Faxa_snow", "Faxa_prec", "Foxx_rofl", "Foxx_rofi")]
    if isotopes:
        raise NotImplementedError(f"E3SM's water isotopes in x2o ({', '.join(isotopes)}) are not ported")
    # the lookups in the Fortran's order, so a missing field stops where E3SM's does
    found = {}
    lookups = list(_EXPLICIT[:9]) + [("x2o", "Foxx_swnet")] + list(_EXPLICIT[9:15])
    lookups += [("r2x", f"Forr_rof{t}") for t in NUTRIENTS] if nutrients else []
    lookups += list(_EXPLICIT[15:18]) + [("x2o", n) for n in ("Faxa_snow", "Faxa_rain", "Faxa_prec",
                                                               "Foxx_rofl", "Foxx_rofi")]
    lookups += [("x2o", f"Foxx_rof{t}") for t in NUTRIENTS] if nutrients else []
    lookups += list(_PER_CAT) if per_cat else []
    for vector, name in lookups:
        found[vector, name] = _require(lists, vector, name)
    explicit = [found[key] for key in _EXPLICIT] + [found.get(key, -1) for key in _PER_CAT]
    nutrient_from = tuple(found["r2x", f"Forr_rof{t}"] for t in NUTRIENTS) if nutrients else ()
    nutrient_to = tuple(found["x2o", f"Foxx_rof{t}"] for t in NUTRIENTS) if nutrients else ()
    missing = [f for f in _FRACTIONS if f not in lists["fractions"]]
    if missing:
        raise ValueError(f"fractions_o lacks {', '.join(missing)}; E3SM reads out of bounds without them")
    explicit += [lists["fractions"].index(f) for f in _FRACTIONS]

    a2x, i2x, xao = lists["a2x"], lists["i2x"], lists["xao"]
    no = len(x2o)
    aindx, iindx, xindx = [-1] * no, [-1] * no, [-1] * no
    amerge, imerge, xmerge = [True] * no, [True] * no, [True] * no
    for ko, field in enumerate(x2o):
        if field[:2] == "PF":
            continue
        if field[:1] == "S" and field[1:2] != "x":
            continue
        if field in ("Foxx_swnet", "Faxa_snow", "Faxa_rain", "Faxa_prec"):
            continue
        if field.startswith(("Faxa_snow_", "Faxa_rain_", "Faxa_prec_")):
            continue
        item = _itemc(field)
        for vector, names, indx, merge, skip in (("ka", a2x, aindx, amerge, None),
                                                  ("ki", i2x, iindx, imerge, "aii"),
                                                  ("kx", xao, xindx, xmerge, None)):
            for k, name in enumerate(names):
                if skip and name[:1] == "F" and name[1:4] == skip:
                    continue
                if item == _itemc(name):
                    if field == name and name[:1] == "F":
                        merge[ko] = False
                    if indx[ko] != -1:
                        raise MergeAbort(f"(prep_ocn_merge)  ERROR multiple {vector} field matches")
                    indx[ko] = k
        if aindx[ko] >= 0 and xindx[ko] >= 0:
            raise MergeAbort("(prep_ocn_merge)  ERROR aindx and xindx both non-zero")
        if aindx[ko] >= 0 and iindx[ko] >= 0 and amerge[ko] != imerge[ko]:
            raise MergeAbort("(prep_ocn_merge)  ERROR aindx and iindx merge logic error")
        if aindx[ko] >= 0 and xindx[ko] >= 0 and amerge[ko] != xmerge[ko]:
            raise MergeAbort("(prep_ocn_merge)  ERROR aindx and xindx merge logic error")
        if xindx[ko] >= 0 and iindx[ko] >= 0 and xmerge[ko] != imerge[ko]:
            raise MergeAbort("(prep_ocn_merge)  ERROR xindx and iindx merge logic error")

    position = {name: ko for ko, name in enumerate(x2o)}
    copies = []
    for vector, code in _COPY_ORDER:
        if vector == "w2x" and wav_ocn_coup != "twoway":
            continue
        copies += [(code, k, position[name]) for k, name in enumerate(lists[vector]) if name in position]
    return Plan(lists=lists, copies=tuple(copies), explicit=tuple(explicit), nutrient_from=nutrient_from,
                nutrient_to=nutrient_to, aindx=tuple(aindx), iindx=tuple(iindx), xindx=tuple(xindx),
                amerge=tuple(amerge), imerge=tuple(imerge), xmerge=tuple(xmerge), nutrients=bool(nutrients),
                per_cat=bool(per_cat))


def module_source() -> str:
    return files("woof.hex.coupled").joinpath("prep_ocn_merge_e3sm.cu").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _module_on(device: int, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(), options, None, "prep_ocn_merge_e3sm.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _options() -> tuple[str, ...]:
    from woof.hex import _hex_numerics
    return _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True, numerics=_hex_numerics._bind())


def merge(the_plan: Plan, vectors: dict, x2o, *, flux_epbalfact: float) -> None:
    """One ``prep_ocn_merge`` call into ``x2o``, in place.

    ``vectors`` maps each of :data:`SOURCES` to its device ``rAttr`` as
    ``(lsize, nRattr)`` float64 (point-major, so ``rAttr(k, n)`` is
    ``[n, k]``); ``x2o`` is ``(lsize, len(the_plan.lists['x2o']))``.
    """
    import cupy as cp
    lsize = int(x2o.shape[0])
    no = len(the_plan.lists["x2o"])
    if x2o.dtype != np.float64 or x2o.shape != (lsize, no):
        raise ValueError(f"x2o must be float64 of shape ({lsize}, {no})")
    args = []
    for vector in ("a2x", "i2x", "r2x", "w2x", "g2x", "xao", "fractions"):
        width = len(the_plan.lists[vector])
        array = vectors.get(vector)
        if width == 0:
            array = cp.zeros(1, dtype=np.float64)
        elif array is None or array.dtype != np.float64 or array.shape != (lsize, width):
            raise ValueError(f"{vector} must be float64 of shape ({lsize}, {width})")
        args += [cp.ascontiguousarray(array), np.int32(width)]

    def ints(values):
        return cp.asarray(np.array(list(values) or [0], dtype=np.int32))

    copies = the_plan.copies
    if lsize > 0:
        _module_on(cp.cuda.Device().id, _options()).get_function("prep_ocn_merge_e3sm")(
            ((lsize + _BLOCK - 1) // _BLOCK,), (_BLOCK,), (
                x2o, np.int32(no), *args,
                ints(c[0] for c in copies), ints(c[1] for c in copies), ints(c[2] for c in copies),
                np.int32(len(copies)), ints(the_plan.explicit),
                ints(the_plan.nutrient_from), ints(the_plan.nutrient_to), np.int32(the_plan.nutrients),
                np.int32(the_plan.per_cat),
                ints(the_plan.aindx), ints(the_plan.iindx), ints(the_plan.xindx),
                ints(the_plan.amerge), ints(the_plan.imerge), ints(the_plan.xmerge),
                np.float64(flux_epbalfact), np.int32(lsize)))
