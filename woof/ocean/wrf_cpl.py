"""WRF v4.6.1's external-ocean exchange (``frame/module_cpl.F``) on device.

WRF couples to NEMO through OASIS3-MCT.  Its own side of that coupling is two
routines of ``module_cpl.F``, which this module runs as CUDA kernels
(``wrf_cpl.cu``) graded bitwise against the pinned source:

* ``cpl_snd2`` -- what the atmosphere hands the ocean every step
  (:data:`SEND_FIELDS`): freshwater ``EVAP-PRECIP = QFX - (RAINCV+RAINNCV)/DT``
  (kg m-2 s-1), ``SURF_NET_SOLAR = GSW`` and ``SURF_NET_NON-SOLAR = GLW -
  STBOLT*EMISS*SST**4 - LH - HFX`` (W m-2), and the stress against the surface
  current, ``TAUX``, ``TAUY`` and ``TAUMOD = RHO*UST**2`` (N m-2);
* ``cpl_rcv`` -- how WRF merges what the ocean hands back
  (:data:`RECEIVE_FIELDS`): ``SST`` onto the input SST, ``UOCE`` and ``VOCE``
  onto zero, under ``CPLMASK``, one plane per external ocean domain.

Between those, OASIS averages each sent field over the coupling period and
remaps it to the ocean grid.  Neither is OASIS code here: :class:`FluxAverager`
is WOOF's own in-process time average (an integral of field*dt, so the flux
the ocean receives over a run equals the flux the atmosphere emitted), and
only a shared atmosphere/ocean grid is accepted -- OASIS's SCRIP remapping is
not ported and a different grid is refused.

Nothing of NEMO is in the default build.  NEMO is CeCILL-licensed; its
port is an explicit opt-in (optional/nemo, docs/coupling/WRF-HYDRO-NEMO.md).
Any solver that implements :class:`woof.ocean.interface.ExternalOcean` can
sit on the other side of this exchange; :mod:`woof.ocean.external` wires
the three WRF calls into the driver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping

from .interface import DeviceArray, validate_plane

#: ``cpl_snd2``'s fields, in the order it sends them (module_cpl.F:322-379).
SEND_FIELDS = ("EVAP-PRECIP", "SURF_NET_SOLAR", "SURF_NET_NON-SOLAR",
               "TAUX", "TAUY", "TAUMOD")

#: The fields WRF 4.6.1 receives (module_surface_driver.F:1510-1527).
RECEIVE_FIELDS = ("SST", "UOCE", "VOCE")

#: Exchange fields an ocean coupling might want that WRF 4.6.1's
#: ``module_cpl.F`` does not carry.  Each is refused by name rather than
#: invented: there is no WRF line to port for it.
NOT_IN_WRF_461 = {
    "SEAICE": "WRF 4.6.1 receives no sea-ice fraction from the coupler "
              "(module_surface_driver.F:1510-1527 calls cpl_rcv for SST, "
              "UOCE and VOCE only)",
    "SSH": "WRF 4.6.1 receives no sea-surface height",
    "RUNOFF": "WRF 4.6.1 sends no river runoff (cpl_snd2 sends EVAP-PRECIP "
              "only)",
}

#: ``share/module_model_constants.F:84``.
STBOLT = 5.67051e-8

_TPB = 128


def _module():
    import cupy as cp

    from woof.certify.kernel_manifest import record_module

    source = Path(__file__).with_name("wrf_cpl.cu").read_text(encoding="utf-8")
    options = ("--fmad=false",)
    module = cp.RawModule(code=source, options=options)
    record_module("woof.ocean.wrf_cpl", source=source, options=options,
                  module=module)
    return module


_MODULE = None


def _kernel(name):
    global _MODULE
    if _MODULE is None:
        _MODULE = _module()
    return _MODULE.get_function(name)


def _blocks(n):
    return ((n + _TPB - 1) // _TPB,)


@dataclass(frozen=True)
class AtmosphereSurface:
    """The grid fields ``cpl_snd2`` reads, as ``(ny, nx)`` float32 device
    planes; ``u_phy``/``v_phy``/``rho`` are the lowest model level."""

    qfx: DeviceArray      # kg m-2 s-1
    raincv: DeviceArray   # mm this step, cumulus
    rainncv: DeviceArray  # mm this step, grid scale
    gsw: DeviceArray      # net shortwave at the surface, W m-2
    glw: DeviceArray      # downward longwave, W m-2
    emiss: DeviceArray
    sst: DeviceArray      # K
    lh: DeviceArray       # W m-2
    hfx: DeviceArray      # W m-2
    u_phy: DeviceArray    # m s-1, grid-relative
    v_phy: DeviceArray
    uoce: DeviceArray     # surface current WRF holds, m s-1
    voce: DeviceArray
    rho: DeviceArray      # kg m-3
    ust: DeviceArray      # m s-1


@dataclass(frozen=True)
class AtmosphereToOcean:
    """``cpl_snd2``'s six fields, ``(ny, nx)`` float32 device planes."""

    evap_precip: DeviceArray         # kg m-2 s-1, positive upward
    surf_net_solar: DeviceArray      # W m-2, positive downward
    surf_net_non_solar: DeviceArray  # W m-2, positive downward
    taux: DeviceArray                # N m-2, grid x
    tauy: DeviceArray                # N m-2, grid y
    taumod: DeviceArray              # N m-2


def _check_grid(value, shape):
    for f in fields(value):
        validate_plane(getattr(value, f.name), shape, f.name)


def send(surface: AtmosphereSurface, out: AtmosphereToOcean, *,
         dt: float) -> None:
    """Fill ``out`` from ``surface`` as ``cpl_snd2`` does, on the current
    stream."""
    import numpy as np

    shape = tuple(surface.sst.shape)
    _check_grid(surface, shape)
    _check_grid(out, shape)
    if not (math.isfinite(dt) and dt > 0):
        raise ValueError("cpl_snd2 needs a finite positive time step")
    n = shape[0] * shape[1]
    args = [getattr(surface, f.name) for f in fields(surface)]
    args += [getattr(out, f.name) for f in fields(out)]
    args += [np.float32(dt), np.float32(STBOLT), np.int32(n)]
    _kernel("wrf_cpl_snd")(_blocks(n), (_TPB,), tuple(args))


def receive(field: str, cplmask, received, out, *, obs=None) -> None:
    """Merge one received field as ``cpl_rcv`` does.

    ``cplmask`` and ``received`` are ``(nedom, ny, nx)`` float32: one plane
    per external ocean domain, in WRF's domain order.  ``obs`` is required
    for ``SST`` (WRF's ``SST_INPUT``) and refused for ``UOCE``/``VOCE``, which
    WRF merges onto zero.
    """
    import numpy as np

    if field in NOT_IN_WRF_461:
        raise NotImplementedError(f"{field}: {NOT_IN_WRF_461[field]}")
    if field not in RECEIVE_FIELDS:
        raise ValueError(f"{field!r} is not one of WRF 4.6.1's received "
                         f"fields {RECEIVE_FIELDS}")
    if (field == "SST") != (obs is not None):
        raise ValueError("cpl_rcv merges SST onto SST_INPUT and UOCE/VOCE "
                         "onto zero: pass obs for SST only")
    shape = tuple(out.shape)
    validate_plane(out, shape, field)
    if obs is not None:
        validate_plane(obs, shape, "SST_INPUT")
    nedom = int(cplmask.shape[0])
    for name, a in (("cplmask", cplmask), ("received", received)):
        if tuple(a.shape) != (nedom, *shape) or a.dtype != np.float32:
            raise ValueError(f"{name} must be float32 ({nedom}, {shape[0]}, "
                             f"{shape[1]})")
        if not a.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    if nedom < 1:
        raise ValueError("cpl_rcv needs at least one external domain")
    n = shape[0] * shape[1]
    _kernel("wrf_cpl_rcv")(_blocks(n), (_TPB,), (
        cplmask, received, obs if obs is not None else out, out,
        np.int32(obs is not None), np.int32(nedom), np.int32(n)))


def validate_cplmask(cplmask) -> None:
    """``cpl_defdomain``'s checks (module_cpl.F:163-183): every mask value in
    [0, 1].  Host-side, once, when the coupling is defined."""
    import cupy as cp

    if float(cp.min(cplmask)) < 0.0:
        raise ValueError("cplmask below 0 (cpl_defdomain aborts)")
    if float(cp.max(cplmask)) > 1.0:
        raise ValueError("cplmask above 1 (cpl_defdomain aborts)")


class FluxAverager:
    """Coupling-interval averages of the sent fields, on device.

    WRF calls ``oasis_put`` every step and the coupler hands the ocean each
    field's average over the coupling period.  This class is WOOF's own
    accumulator for the in-process case: it integrates ``field*dt`` (so an
    adaptive step weights correctly) with a compensated float32 sum, and at
    :meth:`deliver` writes ``integral/elapsed`` and carries the remainder the
    float32 mean could not represent into the next interval.  The integral
    the ocean receives over a whole run therefore equals the one the
    atmosphere emitted, to the rounding of one carried remainder.
    """

    def __init__(self, shape: tuple[int, int], *, interval: float):
        import cupy as cp

        if not (math.isfinite(interval) and interval > 0):
            raise ValueError("coupling interval must be finite and positive")
        self.shape = tuple(shape)
        self.interval = float(interval)
        self.elapsed = 0.0
        self._total = {f.name: cp.zeros(shape, cp.float32)
                       for f in fields(AtmosphereToOcean)}
        self._carry = {f.name: cp.zeros(shape, cp.float32)
                       for f in fields(AtmosphereToOcean)}

    def accumulate(self, sent: AtmosphereToOcean, dt: float) -> None:
        import numpy as np

        if not (math.isfinite(dt) and dt > 0):
            raise ValueError("accumulation needs a finite positive dt")
        if self.elapsed + dt > self.interval * (1 + 1e-9):
            raise ValueError("step crosses the coupling time; partition dt at "
                             "the coupling interval")
        _check_grid(sent, self.shape)
        n = self.shape[0] * self.shape[1]
        kernel = _kernel("wrf_cpl_accumulate")
        for f in fields(sent):
            plane = getattr(sent, f.name)
            kernel(_blocks(n), (_TPB,), (plane, self._total[f.name],
                                         self._carry[f.name], np.float32(dt),
                                         np.int32(n)))
        self.elapsed += dt

    @property
    def due(self) -> bool:
        return self.elapsed >= self.interval * (1 - 1e-9)

    def deliver(self, out: AtmosphereToOcean) -> None:
        import numpy as np

        if self.elapsed <= 0:
            raise ValueError("nothing accumulated")
        _check_grid(out, self.shape)
        n = self.shape[0] * self.shape[1]
        kernel = _kernel("wrf_cpl_deliver")
        for f in fields(out):
            plane = getattr(out, f.name)
            kernel(_blocks(n), (_TPB,), (self._total[f.name],
                                         self._carry[f.name], plane,
                                         np.float32(self.elapsed), np.int32(n)))
        self.elapsed = 0.0


def require_shared_grid(atmosphere_shape, ocean_shape,
                        remap: Mapping | None = None) -> None:
    """Refuse a different ocean grid by name: OASIS's SCRIP remapping
    (conservative or bilinear) between the WRF and NEMO grids is not ported."""
    if remap:
        raise NotImplementedError("OASIS/SCRIP remapping between the WRF and "
                                  "ocean grids is not ported; couple on a "
                                  "shared grid")
    if tuple(atmosphere_shape) != tuple(ocean_shape):
        raise NotImplementedError(
            f"ocean grid {tuple(ocean_shape)} differs from the atmosphere "
            f"grid {tuple(atmosphere_shape)}; OASIS/SCRIP remapping is not "
            "ported")
