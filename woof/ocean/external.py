"""WRF v4.6.1's three external-ocean calls, wired around an ``ExternalOcean``.

WRF couples to an ocean model (NEMO, through OASIS3-MCT) with three calls,
each at its own place in the step:

* ``cpl_store_input`` (dyn_em/solve_em.F:374, module_cpl.F:458-499): at
  ``itimestep == 1`` (and after an auxinput4 read, which WOOF does not do:
  ``sst_update`` cycling is refused at import) the analysed SST is kept as
  SST_INPUT, the field the coupled SST is merged onto.
  :meth:`ExternalOceanCoupling.store_input`.
* ``cpl_rcv`` (phys/module_surface_driver.F:1506-1527), at the top of every
  surface-driver call: SST merged under CPLMASK onto SST_INPUT, then
  ``sst_update = 1``'s water lines (:1636-1790): TSK and the top soil layer
  take SST over water while 250 K < SST < 350 K.  WRF-NEMO runs set
  ``sst_update = 1``; without it the merged SST never reaches TSK.  UOCE and
  VOCE are merged onto zero by WRF; WOOF's surface layers do not read them,
  so an ocean that hands back currents is refused.
  :meth:`ExternalOceanCoupling.receive`.
* ``cpl_snd`` (frame/module_integrate.F:460) once per completed step:
  :func:`woof.ocean.wrf_cpl.send`'s ``cpl_snd2`` fields, from the step's
  surface fields and the lowest level as ``phy_prep`` left it (WRF's U_PHY,
  V_PHY and RHO are the start-of-step values when cpl_snd runs).  OASIS's
  coupling-interval average is WOOF's own :class:`FluxAverager`; at each
  coupling time the average is handed to the ocean, which steps the interval
  and offers a new SST (sequential coupling on a shared grid).
  :meth:`ExternalOceanCoupling.send`.

Nothing here is ocean-model code.  The ocean is any
:class:`woof.ocean.interface.ExternalOcean`; the one implementation is the
NEMO opt-in's column (``optional/nemo``, CeCILL, loaded through
:func:`woof.optin.load` only).
"""

from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path

from .interface import ExternalOcean
from . import wrf_cpl

#: External-ocean models: name -> (opt-in, class).  Each lives behind its
#: opt-in; none is in the default build.
MODELS = {"nemo_column": ("nemo", "NemoColumnOcean")}
#: module_surface_driver.F :1724 (sst_update = 1)
SST_LOW, SST_HIGH = 250.0, 350.0
#: the arrays an ``external_ocean.state`` npz holds
STATE_ARRAYS = ("mbkt", "e3t", "e3w", "gdept", "gdepw", "temperature", "salinity", "cplmask")


class ExternalOceanCoupling:
    """WRF's store/receive/send around one ``ExternalOcean`` on the
    atmosphere's own mass grid (one external domain)."""

    def __init__(self, ocean, *, cplmask, interval, dt, shape):
        import cupy as cp

        if not isinstance(ocean, ExternalOcean):
            raise TypeError("the ocean must implement woof.ocean.interface.ExternalOcean")
        if getattr(ocean, "couples_currents", True):
            raise NotImplementedError(
                "an ocean that returns surface currents: WRF merges UOCE/VOCE "
                "(module_surface_driver.F:1515-1525) but WOOF's surface layers "
                "do not read them")
        interval, dt = float(interval), float(dt)
        n = interval / dt
        if not (math.isfinite(n) and n >= 1 and abs(n - round(n)) < 1e-9):
            raise ValueError("the coupling interval must be a whole number of model steps")
        self.shape = tuple(shape)
        wrf_cpl.require_shared_grid(self.shape, tuple(ocean.surface.sst.shape))
        mask = cp.ascontiguousarray(cp.asarray(cplmask, dtype=cp.float32).reshape(1, *self.shape))
        wrf_cpl.validate_cplmask(mask)
        self.ocean = ocean
        self.cplmask = mask
        self.interval = interval
        self.averager = wrf_cpl.FluxAverager(self.shape, interval=interval)
        zeros = lambda: cp.zeros(self.shape, cp.float32)  # noqa: E731
        self.sent = wrf_cpl.AtmosphereToOcean(**{f.name: zeros() for f in fields(
            wrf_cpl.AtmosphereToOcean)})
        self.delivered = wrf_cpl.AtmosphereToOcean(**{f.name: zeros() for f in fields(
            wrf_cpl.AtmosphereToOcean)})
        self.sst_input = None
        self.u_phy, self.v_phy, self.rho = zeros(), zeros(), zeros()
        self.uoce, self.voce = zeros(), zeros()   # WRF's UOCE/VOCE: merged onto zero

    def store_input(self, sst) -> None:
        """cpl_store_input at itimestep 1: SST_INPUT = SST."""
        import cupy as cp

        self.sst_input = cp.array(sst, dtype=cp.float32, copy=True)

    def receive(self, fields_) -> None:
        """cpl_rcv's SST merge and sst_update's water lines, every step."""
        import cupy as cp

        if self.sst_input is None:
            raise RuntimeError("cpl_store_input has not captured SST_INPUT")
        received = cp.ascontiguousarray(self.ocean.surface.sst.reshape(1, *self.shape))
        sst = fields_["sst"]
        wrf_cpl.receive("SST", self.cplmask, received, sst, obs=self.sst_input)
        water = ((fields_["xland"] > cp.float32(1.5)) & (sst < cp.float32(SST_HIGH))
                 & (sst > cp.float32(SST_LOW)))
        fields_["tsk"][...] = cp.where(water, sst, fields_["tsk"])
        tslb = fields_.get("tslb")
        if tslb is not None:
            tslb[0][...] = cp.where(water, sst, tslb[0])

    def capture_lowest_level(self, atmosphere) -> None:
        """phy_prep's U_PHY, V_PHY and RHO at the lowest level, this step."""
        import cupy as cp

        cp.copyto(self.u_phy, atmosphere["u"][0])
        cp.copyto(self.v_phy, atmosphere["v"][0])
        cp.copyto(self.rho, atmosphere["rho"][0])

    def send(self, fields_, *, raincv, rainncv, dt) -> None:
        """cpl_snd at the end of a step; at a coupling time the ocean steps."""
        import cupy as cp

        surface = wrf_cpl.AtmosphereSurface(
            qfx=fields_["qfx"], raincv=raincv if raincv is not None else cp.zeros(
                self.shape, cp.float32), rainncv=rainncv, gsw=fields_["gsw"],
            glw=fields_["glw"], emiss=fields_["emiss"], sst=fields_["sst"], lh=fields_["lh"],
            hfx=fields_["hfx"], u_phy=self.u_phy, v_phy=self.v_phy, uoce=self.uoce,
            voce=self.voce, rho=self.rho, ust=fields_["ust"])
        wrf_cpl.send(surface, self.sent, dt=dt)
        self.averager.accumulate(self.sent, dt)
        if self.averager.due:
            self.averager.deliver(self.delivered)
            self.ocean.set_fluxes(self.delivered)
            self.ocean.step(self.interval)


def attach(driver, cfg, *, latitude=None) -> None:
    """Attach the ``[external_ocean]`` of ``cfg.ocean_config`` to ``driver``
    on the atmosphere's mass grid, and capture SST_INPUT (cpl_store_input at
    itimestep 1)."""
    import tomllib

    doc = tomllib.loads(Path(cfg.ocean_config).resolve().read_text())
    if set(doc) - {"external_ocean"}:
        raise ValueError("an external_ocean TOML carries no other ocean tables "
                         "(waves/surge couple through [ocean])")
    if latitude is None:
        raise ValueError("external ocean coupling needs the atmosphere latitude")
    if getattr(driver, "lake", None) is not None:
        raise NotImplementedError("the lake model (lakemodel = 1) under an external ocean: "
                                  "sst_update's lake lines are not wired")
    xice = driver.fields.get("xice")
    if xice is not None and float(xice.max()) > 0.0:
        raise NotImplementedError("sea ice under an SST-only external ocean: sst_update's "
                                  "ice transitions are not wired")
    for name in ("sst", "tsk", "xland", "qfx", "gsw", "glw", "emiss", "lh", "hfx", "ust"):
        if name not in driver.fields:
            raise NotImplementedError(f"external ocean coupling needs the surface field "
                                      f"{name.upper()}, which this configuration does not carry")
    config = read_config(doc, cfg.ocean_config)
    shape = tuple(driver.fields["psfc"].shape)
    driver.external_ocean = build(config, latitude=latitude, shape=shape, dt=float(cfg.dt))
    driver.external_ocean.store_input(driver.fields["sst"])


def configured(path) -> bool:
    """Whether the ocean TOML at ``path`` names an ``[external_ocean]``."""
    import tomllib

    return "external_ocean" in tomllib.loads(Path(path).resolve().read_text())


def read_config(doc, path):
    """The ``[external_ocean]`` table of an ocean TOML."""
    cfg = dict(doc["external_ocean"])
    unknown = set(cfg) - {"model", "coupling_interval_seconds", "state", "options"}
    if unknown:
        raise ValueError(f"unknown external_ocean keys: {sorted(unknown)}")
    if cfg.get("model") not in MODELS:
        raise ValueError(f"external_ocean.model must be one of {sorted(MODELS)}")
    interval = cfg.get("coupling_interval_seconds")
    if not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
        raise ValueError("external_ocean.coupling_interval_seconds must be positive and finite")
    if not isinstance(cfg.get("state"), str):
        raise ValueError("external_ocean.state must name an npz of the ocean's grid and state")
    cfg["state"] = str((Path(path).parent / cfg["state"]).resolve())
    cfg.setdefault("options", {})
    return cfg


def build(cfg, *, latitude, shape, dt):
    """Construct the configured ocean (through its opt-in) and its coupling."""
    import numpy as np

    from woof import optin

    name, cls = MODELS[cfg["model"]]
    module = optin.load(name)
    with np.load(cfg["state"]) as z:
        missing = [k for k in STATE_ARRAYS if k not in z]
        if missing:
            raise ValueError(f"external_ocean.state lacks {missing}")
        state = {k: np.array(z[k]) for k in STATE_ARRAYS}
    cplmask = state.pop("cplmask")
    lat = np.asarray(latitude.get() if hasattr(latitude, "get") else latitude, np.float64)
    ocean = getattr(module, cls)(latitude=lat.reshape(shape), coupling_interval=float(
        cfg["coupling_interval_seconds"]), **state, **cfg["options"])
    return ExternalOceanCoupling(ocean, cplmask=cplmask, interval=cfg["coupling_interval_seconds"],
                                 dt=dt, shape=shape)
