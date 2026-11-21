"""E3SM v3.2.0's ocean albedos on the coupler's flux grid, on the GPU, opt-in.

``seq_flux_ocnalb_mct`` (``driver-mct/main/seq_flux_mct.F90``), word for
word, without its orbital branch:

- with ``flux_albav``: ``So_avsdr``/``So_anidr`` set to the namelist's
  ``seq_flux_mct_albdir`` and ``So_avsdf``/``So_anidf`` to its
  ``seq_flux_mct_albdif``, then the radiative fractions ``ifrad``/``ofrad``
  copied from ``ifrac``/``ofrac``;
- otherwise ``Faox_swdn`` and ``Faox_swup`` from the albedos ``xao`` holds
  and the atmosphere's four shortwave bands (0 where ``So_anidr`` is 1, the
  dark side), and, when ``nextsw_cday`` is below -0.5, nothing else.

Refused (:class:`AlbedoRefusal`): the orbital branch (no ``flux_albav`` and
``nextsw_cday >= -0.5``), which needs ``shr_orb_decl`` and ``shr_orb_cosz``
(glibc's sin, cos and asin) and is not ported. Where E3SM stops (the
namelist's albedos negative: ``shr_sys_abort``) :class:`AlbedoAbort`; where
MCT stops (a field missing) :class:`CouplingAbort`.

This is not called by any default path. ``real(r8)`` only, one task. The
kernel (``ocnalb_e3sm.cu``) is held word for word to the routine cut
verbatim and run on the pinned MCT, built with gfortran ``-O0
-ffp-contract=off`` (``tools/e3sm_coupler_oracle``,
``tests/test_e3sm_ocnalb_parity.py``).
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

E3SM_TAG = "v3.2.0"
E3SM_COMMIT = "648b167b7f95267a159a9fdee25faf604c2bfc4f"
MCT_COMMIT = "82b0071e69d14330b75d23b0bc68543ebea9aadc"
SOURCE_FILES = ("driver-mct/main/seq_flux_mct.F90:seq_flux_ocnalb_mct",)
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
_BLOCK = 128

#: the lookups, in the Fortran's order
XAO_FIELDS = ("So_anidr", "So_anidf", "So_avsdr", "So_avsdf", "Faox_swdn", "Faox_swup")
A2X_FIELDS = ("Faxa_swndr", "Faxa_swndf", "Faxa_swvdr", "Faxa_swvdf")
FRACTIONS = ("ifrac", "ifrad", "ofrac", "ofrad")
#: the kernel's X_* order
_KERNEL_XAO = ("So_avsdr", "So_anidr", "So_avsdf", "So_anidf", "Faox_swdn", "Faox_swup")


class CouplingAbort(RuntimeError):
    """MCT stopping the coupler (``die``), with its message."""


class AlbedoAbort(RuntimeError):
    """E3SM stopping the coupler (``shr_sys_abort``), with its message."""


class AlbedoRefusal(ValueError):
    """A configuration this port does not reproduce."""


def module_source() -> str:
    return files("woof.hex.coupled").joinpath("ocnalb_e3sm.cu").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _module_on(device: int, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(), options, None, "ocnalb_e3sm.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _options() -> tuple[str, ...]:
    from woof.hex import numerics as _hex_numerics
    return _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True, numerics=_hex_numerics._bind())


def _device_of(a) -> int:
    return int(a.device.id)


def _av(name, array, fields, n=None):
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
        raise CouplingAbort(f'm_AttrVect::indexRA_:: FATAL--attribute not found: "{name}" ({av})')
    return pos[name]


def _overlap(a, b) -> bool:
    if not a.size or not b.size:
        return False
    a0, b0 = int(a.data.ptr), int(b.data.ptr)
    return a0 < b0 + int(b.nbytes) and b0 < a0 + int(a.nbytes)


def seq_flux_ocnalb(*, a2x, a2x_fields, fractions, fractions_fields, xao, xao_fields, flux_albav: bool,
                    albdif: float, albdir: float, nextsw_cday: float) -> None:
    """``seq_flux_ocnalb_mct`` on the flux grid: updates ``xao`` and (with
    ``flux_albav``) ``fractions`` in place. Each is a cupy ``(lsize, nflds)``
    float64 array, MCT's rAttr in memory order. ``albdif``/``albdir`` are
    the namelist's ``seq_flux_mct_albdif``/``albdir``."""
    import cupy as cp
    for name, value in (("flux_albav", flux_albav),):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a bool")
    xpos = _av("xao", xao, xao_fields)
    n = int(xao.shape[0])
    apos = _av("a2x", a2x, a2x_fields, n)
    fpos = _av("fractions", fractions, fractions_fields, n)
    for x, y in ((xao, a2x), (xao, fractions), (fractions, a2x)):
        if _overlap(x, y):
            raise ValueError("xao, a2x and fractions must be separate arrays; two share memory")
    device = cp.cuda.Device().id
    for a in (xao, a2x, fractions):
        if _device_of(a) != device:
            raise ValueError(f"an array is on device {_device_of(a)}; the current device is {device}")
    # the first call's checks and lookups, in the Fortran's order
    if albdif < 0 or albdir < 0:
        raise AlbedoAbort("(seq_flux_ocnalb_mct) ERROR seq_flux_mct_inparm namelist not initialized")
    for f in XAO_FIELDS:
        _need(xpos, f, "xao")
    for f in A2X_FIELDS:
        _need(apos, f, "a2x")
    if not flux_albav and nextsw_cday >= -0.5:
        raise AlbedoRefusal("seq_flux_ocnalb_mct's orbital albedo (flux_albav off, nextsw_cday >= -0.5) is not "
                            "ported: shr_orb_decl and shr_orb_cosz call glibc's sin, cos and asin")
    kf = [-1] * 4
    if flux_albav:
        kf = [_need(fpos, f, "fractions") for f in FRACTIONS]
    if n == 0:
        return
    kx = np.array([xpos[f] for f in _KERNEL_XAO], dtype=np.int32)
    ka = np.array([apos[f] for f in A2X_FIELDS], dtype=np.int32)
    _module_on(device, _options()).get_function("ocnalb")(
        ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
        (xao, np.int32(len(xpos)), cp.asarray(kx), a2x, np.int32(len(apos)), cp.asarray(ka), fractions,
         np.int32(len(fpos)), cp.asarray(np.array(kf, dtype=np.int32)), np.int32(bool(flux_albav)),
         np.float64(albdif), np.float64(albdir), np.int32(n)))
