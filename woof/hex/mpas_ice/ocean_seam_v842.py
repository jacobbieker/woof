"""MALI's side of the land-ice/ocean seam: the fields it exports and imports.

MALI v8.4.2 standalone has no coupler, so the seam is built from the model's
own fields after a step of :mod:`woof.hex.mpas_ice.forward_v842`. The names,
units and signs were agreed with the MPAS-Ocean coupling (the ocean reads
``landIcePressure``, ``landIceFraction`` and ``landIceMask``, its land-ice
pieces in :mod:`woof.hex.mpas_ocean.land_ice_v842`). Remapping between meshes
and the channel wiring of :mod:`woof.hex.coupled.exchange` belong to the
ocean/coupler side: every field here is on MALI's mesh, over its owned cells
(``nCellsSolve``), at MALI's RKIND. The seam is single-rank: on a
partitioned mesh, liquid runoff whose receiving ocean cell is a halo cell
(at or beyond ``nCellsSolve``) belongs to another rank, so
:func:`export_fields` refuses it by name instead of dropping it.

Exports (:func:`export_fields`):

- ``landIceFraction`` (1, real 0 or 1) and ``landIceMask`` (int32 0 or 1):
  floating ice, ``li_mask_is_floating_ice`` of ``cellMask``. Grounded ice is
  not ocean.
- ``landIcePressure`` (Pa): the floating ice's overburden,
  ``(config_ice_density * gravity) * thickness``, 0 elsewhere.
- ``li2ocn_solid_runoff`` (kg m^-2 s^-1, positive into the ocean): the step's
  calving, ``(calvingThickness * config_ice_density) / deltat``, in the
  calved cell.
- ``li2ocn_liquid_runoff`` (kg m^-2 s^-1, positive into the ocean): the
  subglacial hydrology's net discharge into the ocean. For each owned edge
  between a grounded-ice cell and an ocean cell (floating ice, or no ice on
  a bed below ``config_sea_level``), the outward volume rate ``waterFlux *
  dvEdge + channelDischarge`` (signed by ``cellsOnEdge``'s order), times
  ``rho_water``, over the ocean cell's ``areaCell``; edges are taken in
  index order. Water the hydrology moves onto ice-free land is not the
  ocean's. Without ``config_SGH`` it is 0: v8.4.2 routes no meltwater
  without the hydrology. ``waterFlux`` and ``channelDischarge`` are positive
  from ``cellsOnEdge(1)`` to ``cellsOnEdge(2)``, as in v8.4.2's hydrology
  (mpas_li_subglacial_hydro.F: the slope ``hydropotentialBaseSlopeNormal =
  (phi(cell2) - phi(cell1)) / dcEdge`` at line 786, ``channelDischarge = -Kc
  ... * hydropotentialBaseSlopeNormal`` at line 1468, and the divergence's
  ``- waterFlux * dvEdge * edgeSignOnCell`` at line 489).
- ``landIceDraft`` (m, diagnostic, not a channel): ``lowerSurface`` under
  floating ice, 0 elsewhere.

Import (:func:`import_fields`): ``ocn2li_floating_basal_mass_bal`` (kg m^-2
s^-1, MALI's sign: negative is melt; the ocean's ``-landIceFreshwaterFlux``)
into ``floatingBasalMassBal``. It is v8.4.2's ``config_basal_mass_bal_float =
'file'`` path ("the value set by an ESM coupler"), so any other setting is
refused by name.

Refused by name: ``li2ocn_momentum_x`` and ``li2ocn_momentum_y`` (v8.4.2
computes no calving momentum). There is no oracle for the seam; the tests
grade it by conservation (``tests/test_mali_ocean_seam.py``).
"""
from __future__ import annotations

import numpy as np

from woof.hex.errors import ConfigurationRefusal

from . import rkind

#: li_constants (mpas_li_constants.F)
GRAVITY = 9.80616
RHO_WATER = 1000.0
#: li_mask_ValueIce, li_mask_ValueFloating
MASK_ICE, MASK_FLOATING = 32, 4

EXPORTS = ("landIceFraction", "landIceMask", "landIcePressure", "li2ocn_liquid_runoff", "li2ocn_solid_runoff")
DIAGNOSTICS = ("landIceDraft",)
IMPORTS = ("ocn2li_floating_basal_mass_bal",)
REFUSED = ("li2ocn_momentum_x", "li2ocn_momentum_y")


def refuse_channel(name: str) -> None:
    """Refuse a seam channel MALI does not carry, by name."""
    if name in REFUSED:
        raise ConfigurationRefusal(
            knob="coupled.channel", value=name, reason="MALI v8.4.2 computes no calving momentum",
            declaration="MALI exports " + ", ".join(EXPORTS) + " and imports " + ", ".join(IMPORTS))
    if name not in EXPORTS + DIAGNOSTICS + IMPORTS:
        raise ConfigurationRefusal(
            knob="coupled.channel", value=name, reason="not a field of MALI's side of the ocean seam",
            declaration="MALI exports " + ", ".join(EXPORTS + DIAGNOSTICS) + " and imports " + ", ".join(IMPORTS))


def _host(value) -> np.ndarray:
    import cupy as cp
    return cp.asnumpy(value) if isinstance(value, cp.ndarray) else np.asarray(value)


def _owned(pools: dict) -> tuple[int, int]:
    m = pools["mesh"]
    n_cells = int(pools["geometry"]["thickness"].shape[0])
    n_edges = int(m["dvEdge"].shape[0])
    ncs = int(_host(m["nCellsSolve"]).reshape(-1)[0]) if "nCellsSolve" in m else n_cells
    nes = int(_host(m["nEdgesSolve"]).reshape(-1)[0]) if "nEdgesSolve" in m else n_edges
    return ncs, nes


def floating(cell_mask: np.ndarray) -> np.ndarray:
    """``li_mask_is_floating_ice``."""
    return ((cell_mask & MASK_FLOATING) == MASK_FLOATING) & ((cell_mask & MASK_ICE) == MASK_ICE)


def grounded(cell_mask: np.ndarray) -> np.ndarray:
    """``li_mask_is_grounded_ice``."""
    return ((cell_mask & MASK_FLOATING) != MASK_FLOATING) & ((cell_mask & MASK_ICE) == MASK_ICE)


def ocean(cell_mask: np.ndarray, bed: np.ndarray, sea_level: float) -> np.ndarray:
    """Floating ice, or no ice on a bed below sea level."""
    no_ice = (cell_mask & MASK_ICE) != MASK_ICE
    return floating(cell_mask) | (no_ice & (bed < sea_level))


def grounding_line_discharge(pools: dict, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """The hydrology's outward volume rate (m^3/s) on each owned edge between
    a grounded-ice cell and an ocean cell, and the ocean cell (0-based; -1
    where the edge is not such an edge)."""
    m, g = pools["mesh"], pools["geometry"]
    _, nes = _owned(pools)
    real = rkind.PRECISIONS[rkind.precision_of(g["thickness"])]
    rate = np.zeros(nes, real)
    receiver = np.full(nes, -1, np.int64)
    if not config.get("config_SGH", False):
        return rate, receiver
    h = pools["hydro"]
    mask = _host(g["cellMask"])
    cells_on_edge = _host(m["cellsOnEdge"]).reshape(-1, 2)[:nes].astype(np.int64) - 1
    n_cells = mask.shape[0]
    flux = _host(h["waterFlux"]).astype(real)
    dv = _host(m["dvEdge"]).astype(real)
    channel = _host(h["channelDischarge"]).astype(real)
    is_grounded = grounded(mask)
    is_ocean = ocean(mask, _host(g["bedTopography"]).astype(real), real(config["config_sea_level"]))
    # perf: a host loop over nEdgesSolve; vectorise only if the edge-order summation in export_fields is kept.
    for e in range(nes):
        c1, c2 = cells_on_edge[e]
        if not (0 <= c1 < n_cells and 0 <= c2 < n_cells):
            continue
        if is_grounded[c1] and is_ocean[c2]:
            q = real(flux[e] * dv[e]) + channel[e]
            rate[e], receiver[e] = q, c2
        elif is_grounded[c2] and is_ocean[c1]:
            q = real(flux[e] * dv[e]) + channel[e]
            rate[e], receiver[e] = -q, c1
    return rate, receiver


def export_fields(*, pools: dict, config: dict, diagnostics: bool = False) -> dict:
    """MALI's exports after a step, over the owned cells (device arrays)."""
    import cupy as cp
    m, g = pools["mesh"], pools["geometry"]
    ncs, _ = _owned(pools)
    real = rkind.PRECISIONS[rkind.precision_of(g["thickness"])]
    mask = _host(g["cellMask"])
    thickness = _host(g["thickness"]).astype(real)
    area = _host(m["areaCell"]).astype(real)
    rhoi = real(config["config_ice_density"])
    deltat = real(_host(m["deltat"]).reshape(-1)[0])
    if not deltat > 0:
        raise ValueError("the seam's runoff needs a step's deltat > 0 (export after a step)")
    shelf = floating(mask)[:ncs]
    zero = real(0)
    out = {
        "landIceFraction": np.where(shelf, real(1), zero).astype(real),
        "landIceMask": shelf.astype(np.int32),
        "landIcePressure": np.where(shelf, (rhoi * real(GRAVITY)) * thickness[:ncs], zero).astype(real),
        "li2ocn_solid_runoff": ((_host(g["calvingThickness"]).astype(real)[:ncs] * rhoi) / deltat).astype(real),
    }
    rate, receiver = grounding_line_discharge(pools, config)
    halo = receiver >= ncs
    if halo.any():
        raise ConfigurationRefusal(
            knob="mesh.partition", value=f"nCellsSolve={ncs}",
            reason=f"{int(halo.sum())} grounding-line edge(s) discharge into halo cells owned by another rank",
            declaration="the land-ice/ocean seam is single-rank: export on an unpartitioned mesh")
    liquid = np.zeros(ncs, real)
    rho_w = real(RHO_WATER)
    for e in np.nonzero(receiver >= 0)[0]:
        c = receiver[e]
        liquid[c] = liquid[c] + (rate[e] * rho_w) / area[c]
    out["li2ocn_liquid_runoff"] = liquid
    if diagnostics:
        out["landIceDraft"] = np.where(shelf, _host(g["lowerSurface"]).astype(real)[:ncs], zero).astype(real)
    return {k: cp.asarray(np.ascontiguousarray(v)) for k, v in out.items()}


def import_fields(*, pools: dict, config: dict, fields: dict) -> None:
    """Put the ocean's fields into MALI's pools (before a step)."""
    import cupy as cp
    for name in fields:
        if name not in IMPORTS:
            refuse_channel(name)
            raise ConfigurationRefusal(knob="coupled.channel", value=name, reason="MALI exports this field",
                                       declaration="MALI imports " + ", ".join(IMPORTS))
    if "ocn2li_floating_basal_mass_bal" not in fields:
        return
    if config.get("config_basal_mass_bal_float") != "file":
        raise ConfigurationRefusal(
            knob="config_basal_mass_bal_float", value=config.get("config_basal_mass_bal_float"),
            reason="the ocean's floating basal mass balance is v8.4.2's 'file' path",
            declaration="config_basal_mass_bal_float = 'file' to take ocn2li_floating_basal_mass_bal")
    g = pools["geometry"]
    ncs, _ = _owned(pools)
    target = g["floatingBasalMassBal"]
    value = cp.asarray(fields["ocn2li_floating_basal_mass_bal"])
    if value.shape != (ncs,) or value.dtype != target.dtype:
        raise ValueError(f"ocn2li_floating_basal_mass_bal must be {np.dtype(target.dtype).name} of shape ({ncs},)")
    target[:ncs] = value
