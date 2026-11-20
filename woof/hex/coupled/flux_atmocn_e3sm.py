"""E3SM v3.2.0's atmosphere-ocean flux on the GPU, opt-in.

E3SM's coupler computes the turbulent fluxes between the atmosphere and the
ocean (the ``xao`` attribute vector) on its flux grid with
``seq_flux_atmocn_mct`` (``driver-mct/main/seq_flux_mct.F90``) over
``shr_flux_atmOcn`` (``share/util/shr_flux_mod.F90``). :func:`flux_mask` is
``seq_flux_init_mct``'s mask and :func:`seq_flux_atmocn` the flux, word for
word:

- the points the mask keeps read a2x and o2x (the dead models: E3SM's
  constants everywhere, and the mask set to 1 for good); ``emask`` drops a
  density below the single-precision literal ``1.0e-12`` (float32(1e-12)
  widened) and an ocean temperature below 1 K;
- ``shr_flux_atmOcn``'s E3SMv1 scheme (``ocn_surface_flux_scheme = 0``):
  the neutral drag, Monin-Obukhov iteration (``flux_max_iteration``,
  ``flux_convergence``), with the atmosphere's gustiness
  (``atm_gustiness``), the implicit stress (``atm_flux_method =
  'implicit_stress'``, ``shr_flux_update_stress``) and the cold-air
  outbreak scaling (``coldair_outbreak_mod``); points ``emask`` drops get
  ``shr_const_spval``;
- the write into ``xao`` where the mask is not 0: the fluxes, the
  reference-height values, ``So_u10withgusts = sqrt(So_duu10n)``, the ocean's
  ``So_fswpen``, and the diurnal fields as read (0 with dead models).

Where E3SM stops (``shr_sys_abort``: no iteration done, the vectors' sizes
differ, a scheme other than 0 or 1, the diurnal cycle with the UA scheme)
this raises :class:`FluxAbort` with E3SM's message, and ``xao`` is
unchanged. Where MCT stops (a required field missing) :class:`CouplingAbort`.

Refused (:class:`FluxRefusal`), not ported: the COARE (1) and UA (2)
schemes, the diurnal cycle (``flux_diurnal``), two-way wave coupling
(``wav_atm_coup = 'twoway'``), and an xao with the water isotopes' fluxes.

``psimhu`` calls glibc's binary64 ``atan`` on every iteration; it is LGPL
and not correctly rounded, so the unit compiles only with the opt-in
transcription (``WOOF_GLIBC239_LIBM64=1``, :mod:`woof.core.glibc239_gpu`)
and is refused by name without it. ``exp``, ``log`` and ``pow`` (``x**0.5``)
are the shared MIT transcriptions of glibc's (``glibc_flt64.cuh``); ``x**4``
is libgcc's ``__powidf2`` as gfortran -O0 calls it; ``log(zref/ztref)``,
which gfortran folds, is folded here on the host.

This is not called by any default path. ``real(r8)`` only, one task. The
kernels (``flux_atmocn_e3sm.cu``) are held word for word to E3SM's
``seq_flux_init_mct`` and ``seq_flux_atmocn_mct`` cut verbatim, over
``shr_flux_mod`` compiled verbatim and the pinned MCT, built with gfortran
``-O0 -ffp-contract=off`` (``tools/e3sm_coupler_oracle``,
``tests/test_e3sm_flux_atmocn_parity.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

E3SM_TAG = "v3.2.0"
E3SM_COMMIT = "648b167b7f95267a159a9fdee25faf604c2bfc4f"
MCT_COMMIT = "82b0071e69d14330b75d23b0bc68543ebea9aadc"
SOURCE_FILES = ("driver-mct/main/seq_flux_mct.F90:seq_flux_init_mct,seq_flux_atmocn_mct",
                "share/util/shr_flux_mod.F90:shr_flux_atmOcn,shr_flux_update_stress")
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
MATH_HEADERS = ("glibc_flt64.cuh",)
_BLOCK = 128

# shr_const_mod (v3.2.0), each operation of its parameter expressions rounded
# as gfortran folds them (MPFR at 53 bits: IEEE double, round to nearest)
SHR_CONST_BOLTZ = 1.38065e-23
SHR_CONST_AVOGAD = 6.02214e26
SHR_CONST_RGAS = SHR_CONST_AVOGAD * SHR_CONST_BOLTZ
SHR_CONST_RDAIR = SHR_CONST_RGAS / 28.966
SHR_CONST_RWV = SHR_CONST_RGAS / 18.016
SHR_CONST_ZVIR = (SHR_CONST_RWV / SHR_CONST_RDAIR) - 1.0
SHR_CONST_CPDAIR = 1.00464e3
SHR_CONST_CPVIR = (1.810e3 / SHR_CONST_CPDAIR) - 1.0
SHR_CONST_KARMAN = 0.4
SHR_CONST_G = 9.80616
SHR_CONST_LATVAP = 2.501e6
SHR_CONST_STEBOL = 5.670374419e-8
SHR_CONST_SPVAL = 1.0e30
ZREF, ZTREF = 10.0, 2.0
#: shr_flux_atmOcn's line that stops when a point did no iteration
NO_ITERATION_LINE = 476

#: the a2x and o2x fields read, in the kernel's order (None: optional)
A2X_FIELDS = ("Sa_z", "Sa_u", "Sa_v", "Sa_wsresp", "Sa_tau_est", "Sa_ugust", "Sa_ptem", "Sa_shum", "Sa_dens",
              "Sa_tbot")
#: a2x fields seq_flux_atmocn_mct looks up (MCT stops without them) and does not use here
A2X_LOOKED_UP = ("Sa_pslv", "Faxa_lwdn", "Faxa_rainc", "Faxa_rainl", "Faxa_snowc", "Faxa_snowl")
O2X_FIELDS = ("So_t", "So_u", "So_v", "So_fswpen")
O2X_LOOKED_UP = ("So_s",)
#: the xao fields written, in the kernel's order
XAO_OUT = ("Faox_sen", "Faox_lat", "Faox_taux", "Faox_tauy", "Faox_evap", "So_tref", "So_qref", "So_ustar", "So_re",
           "So_ssq", "Faox_lwup", "So_duu10n", "So_u10", "So_u10withgusts", "So_fswpen")
XAO_DIURNAL = ("So_warm_diurn", "So_salt_diurn", "So_speed_diurn", "So_regime_diurn", "So_tskin_diurn",
               "So_tskin_day_diurn", "So_tskin_night_diurn", "So_cskin_diurn", "So_cskin_night_diurn",
               "So_tbulk_diurn", "So_warmmax_diurn", "So_windmax_diurn", "So_qsolavg_diurn", "So_windavg_diurn",
               "So_warmmaxinc_diurn", "So_windmaxinc_diurn", "So_qsolinc_diurn", "So_windinc_diurn",
               "So_ninc_diurn")
XAO_LOOKED_UP = ("Faox_swdn", "Faox_swup")
XAO_ISOTOPES = ("Faox_evap_16O", "Faox_evap_HDO", "Faox_evap_18O")
FLUX_METHODS = ("explicit", "implicit_stress")
WAV_ATM_COUPLINGS = ("none", "twoway")
_F_DEAD, _F_IMPLICIT, _F_GUST, _F_COLDAIR = 1, 2, 4, 8


class CouplingAbort(RuntimeError):
    """MCT stopping the coupler (``die``), with its message."""


class FluxAbort(RuntimeError):
    """E3SM stopping the coupler (``shr_sys_abort``), with its message."""


class FluxRefusal(ValueError):
    """A configuration this port does not reproduce."""


@dataclass(frozen=True)
class FluxOptions:
    """The switches ``seq_flux_atmocn_mct`` reads: infodata's
    (``dead_comps``, ``ocn_surface_flux_scheme``, ``flux_diurnal``,
    ``coldair_outbreak_mod``, ``flux_convergence``, ``flux_max_iteration``,
    -1 meaning 30 with the implicit stress and 2 otherwise), seq_flds_mod's
    (``atm_flux_method``, ``atm_gustiness``, ``wav_atm_coup``) and the
    namelist's ``seq_flux_atmocn_minwind``."""

    dead_comps: bool = False
    ocn_surface_flux_scheme: int = 0
    flux_diurnal: bool = False
    coldair_outbreak_mod: bool = False
    flux_convergence: float = 0.0
    flux_max_iteration: int = -1
    atm_flux_method: str = "explicit"
    atm_gustiness: bool = False
    wav_atm_coup: str = "none"
    seq_flux_atmocn_minwind: float = 0.5

    def check(self) -> None:
        if self.atm_flux_method not in FLUX_METHODS:
            raise ValueError(f"atm_flux_method {self.atm_flux_method!r}: one of {FLUX_METHODS}")
        if self.wav_atm_coup not in WAV_ATM_COUPLINGS:
            raise ValueError(f"wav_atm_coup {self.wav_atm_coup!r}: one of {WAV_ATM_COUPLINGS}")
        for name in ("ocn_surface_flux_scheme", "flux_max_iteration"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
        for name in ("flux_convergence", "seq_flux_atmocn_minwind"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), (int, float, np.floating)):
                raise TypeError(f"{name} must be a real")

    def max_iteration(self) -> int:
        if self.flux_max_iteration == -1:
            return 30 if self.atm_flux_method == "implicit_stress" else 2
        return int(self.flux_max_iteration)


def _refuse(option: str, value, why: str) -> None:
    raise FluxRefusal(f"{option} = {value!r} is not ported: {why}")


@lru_cache(maxsize=None)
def folded_al2() -> float:
    """``log(zref/ztref)``, which gfortran folds (correctly rounded)."""
    from decimal import Context, Decimal
    return float(Context(prec=60).ln(Decimal(ZREF / ZTREF)))


def constants(options: FluxOptions) -> np.ndarray:
    """The kernel's constants array (``flux_atmocn_e3sm.cu``'s C_* order)."""
    return np.array([SHR_CONST_ZVIR, SHR_CONST_CPDAIR, SHR_CONST_CPVIR, SHR_CONST_KARMAN, SHR_CONST_G,
                     SHR_CONST_LATVAP, SHR_CONST_STEBOL, folded_al2(), float(options.seq_flux_atmocn_minwind),
                     float(options.flux_convergence)], dtype=np.float64)


def module_source() -> str:
    import woof.core.kernels as kernels
    from woof.hex.mpas_ice import rkind
    root = Path(kernels.__file__).parent
    math = "\n".join((root / name).read_text(encoding="utf-8") for name in MATH_HEADERS)
    body = files("woof.hex.coupled").joinpath("flux_atmocn_e3sm.cu").read_text(encoding="utf-8")
    return rkind.r8_trig_prelude() + math + "\n" + body


@lru_cache(maxsize=None)
def _module_on(device: int, options: tuple[str, ...], trig: bool):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(), options, None, "flux_atmocn_e3sm.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _options() -> tuple[str, ...]:
    from woof.hex import numerics as _hex_numerics
    return _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True, numerics=_hex_numerics._bind())


def _device_of(a) -> int:
    return int(a.device.id)


def _launch(kernel: str, n: int, args) -> None:
    import cupy as cp
    from woof.hex.mpas_ice import rkind
    device = cp.cuda.Device().id
    for a in args:
        if isinstance(a, cp.ndarray) and _device_of(a) != device:
            raise ValueError(f"an array of {kernel} is on device {_device_of(a)}; the current device is {device}")
    if n > 0:
        _module_on(device, _options(), rkind.trig64()).get_function(kernel)(
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
        raise CouplingAbort(f'm_AttrVect::indexRA_:: FATAL--attribute not found: "{name}" ({av})')
    return pos[name]


def _span(a):
    start = int(a.data.ptr)
    return start, start + int(a.nbytes)


def _overlap(a, b) -> bool:
    if not a.size or not b.size:
        return False
    (a0, a1), (b0, b1) = _span(a), _span(b)
    return a0 < b1 and b0 < a1


def flux_mask(*, dom, dom_fields, fractions, fractions_fields):
    """``seq_flux_init_mct``'s mask on the flux grid: an int32 cupy array, 0
    where the domain's ``mask`` is below 0.5 or ``ofrac + ifrac <= 0``."""
    import cupy as cp
    dpos = _av("dom", dom, dom_fields)
    n = int(dom.shape[0])
    fpos = _av("fractions", fractions, fractions_fields, n)
    if "mask" not in dpos:
        raise CouplingAbort('m_AttrVect::indexRA_:: FATAL--attribute not found: "mask" (dom)')
    ko, ki = _need(fpos, "ofrac", "fractions"), _need(fpos, "ifrac", "fractions")
    mask = cp.zeros(n, dtype=np.int32)
    _check_trig()
    _launch("flux_init_mask", n, (mask, dom, np.int32(len(dpos)), np.int32(dpos["mask"]), fractions,
                                  np.int32(len(fpos)), np.int32(ko), np.int32(ki), np.int32(n)))
    return mask


def _check_trig() -> None:
    from woof.hex.mpas_ice import rkind
    if not rkind.trig64():
        raise FluxRefusal("seq_flux_atmocn = the E3SMv1 scheme is refused by default: "
                          + rkind.trig_refusal("shr_flux_atmOcn's psimhu calls glibc's double atan"))


def seq_flux_atmocn(*, a2x, a2x_fields, o2x, o2x_fields, xao, xao_fields, mask, options: FluxOptions) -> None:
    """``seq_flux_atmocn_mct``: updates ``xao`` (a cupy ``(lsize, nflds)``
    float64 array, MCT's rAttr in memory order) in place from ``a2x`` and
    ``o2x`` on the same grid, with :func:`flux_mask`'s ``mask`` (int32; set
    to 1 everywhere with dead models, as E3SM's module mask is). Raises
    :class:`FluxAbort` where E3SM stops, leaving ``xao`` unchanged."""
    import cupy as cp
    if not isinstance(options, FluxOptions):
        raise TypeError("options must be a flux_atmocn_e3sm.FluxOptions")
    options.check()
    xpos = _av("xao", xao, xao_fields)
    n = int(xao.shape[0])
    apos = _av("a2x", a2x, a2x_fields)
    opos = _av("o2x", o2x, o2x_fields)
    if not isinstance(mask, cp.ndarray) or mask.dtype != np.int32 or mask.shape != (n,) \
            or not mask.flags.c_contiguous:
        raise ValueError(f"mask must be flux_mask's C-contiguous int32 cupy array of {n} points")
    for x, y in (("a2x", "xao"), ("o2x", "xao"), ("mask", "xao"), ("mask", "a2x"), ("mask", "o2x")):
        if _overlap({"a2x": a2x, "o2x": o2x, "xao": xao, "mask": mask}[x],
                    {"a2x": a2x, "o2x": o2x, "xao": xao, "mask": mask}[y]):
            raise ValueError(f"{x} and {y} share memory; seq_flux_atmocn needs separate arrays")
    # the indices seq_flux_atmocn_mct looks up on its first call (MCT stops without one)
    kx = [_need(xpos, f, "xao") for f in ("So_tref", "So_qref", "So_ustar", "So_re", "So_ssq", "So_u10",
                                           "So_u10withgusts", "So_duu10n", "Faox_taux", "Faox_tauy", "Faox_lat",
                                           "Faox_sen", "Faox_evap")]
    present = [f for f in XAO_ISOTOPES if f in xpos]
    kx += [_need(xpos, f, "xao") for f in ("Faox_lwup",) + XAO_LOOKED_UP + ("So_fswpen",) + XAO_DIURNAL]
    implicit = options.atm_flux_method == "implicit_stress"
    a_needed = ("Sa_z", "Sa_u", "Sa_v") + (("Sa_wsresp", "Sa_tau_est") if implicit else ()) \
        + (("Sa_ugust",) if options.atm_gustiness else ()) + ("Sa_tbot", "Sa_pslv", "Sa_ptem", "Sa_shum", "Sa_dens",
                                                               "Faxa_lwdn", "Faxa_rainc", "Faxa_rainl", "Faxa_snowc",
                                                               "Faxa_snowl")
    for f in a_needed:
        _need(apos, f, "a2x")
    for f in ("So_t", "So_u", "So_v", "So_fswpen", "So_s"):
        _need(opos, f, "o2x")
    if options.wav_atm_coup == "twoway":
        _refuse("wav_atm_coup", options.wav_atm_coup, "the two-way wave coupling's drag (cdn_wave) is not ported")
    if present:
        _refuse("xao fields", tuple(present), "the water isotopes' fluxes (wiso_flxoce) are not ported")
    if int(a2x.shape[0]) != n or int(o2x.shape[0]) != n:
        raise FluxAbort("(seq_flux_atmocn_mct) ERROR nloc sizes do not match")
    if options.flux_diurnal:
        if options.ocn_surface_flux_scheme == 2:
            raise FluxAbort("(seq_flux_atmocn_mct) ERROR cannot use flux_diurnal with UA flux scheme")
        _refuse("flux_diurnal", True, "shr_flux_atmOcn_diurnal is not ported")
    if options.ocn_surface_flux_scheme in (1, 2):
        _refuse("ocn_surface_flux_scheme", options.ocn_surface_flux_scheme,
                "only the E3SMv1 scheme (0) is ported; COARE (1) and UA (2) are not")
    if options.ocn_surface_flux_scheme != 0:
        raise FluxAbort("(shr_flux_atmOcn)  subroutine shr_flux_atmOcn requires ocn_surface_flux_scheme = 0 or 1")
    _check_trig()
    ka = np.array([apos.get(f, -1) if f in a_needed else -1 for f in A2X_FIELDS], dtype=np.int32)
    ko = np.array([opos[f] for f in O2X_FIELDS], dtype=np.int32)
    flags = (_F_DEAD * options.dead_comps) | (_F_IMPLICIT * implicit) | (_F_GUST * options.atm_gustiness) \
        | (_F_COLDAIR * options.coldair_outbreak_mod)
    if n == 0:
        return
    work = cp.empty((n, 14), dtype=np.float64)
    noiter = cp.zeros(n, dtype=np.int32)
    keep = mask.copy()
    _launch("flux_atmocn", n, (work, noiter, mask, a2x, np.int32(len(apos)), cp.asarray(ka), o2x,
                               np.int32(len(opos)), cp.asarray(ko), cp.asarray(constants(options)),
                               np.int32(options.max_iteration()), np.int32(flags), np.int32(n)))
    if bool(noiter.any()):
        mask[...] = keep
        raise FluxAbort(f"No iterations performed ERROR in shr_flux_mod.F90 at line {NO_ITERATION_LINE}")
    kout = np.array([xpos[f] for f in XAO_OUT], dtype=np.int32)
    kdiurnal = np.array([xpos[f] for f in XAO_DIURNAL], dtype=np.int32)
    _launch("flux_atmocn_write", n, (xao, np.int32(len(xpos)), cp.asarray(kout), cp.asarray(kdiurnal),
                                     np.int32(kdiurnal.size), work, mask, np.int32(bool(options.dead_comps)),
                                     np.int32(n)))
