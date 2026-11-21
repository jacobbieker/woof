"""WRF's flow-dependent specified zone for the species a regional hex run does not drive.

An MPAS limited-area run drives the scalars its boundary stream carries
(``atm_bdy_adjust_scalars`` on the relaxation rings, ``atm_bdy_set_scalars``
on the specified rings 6-7).  The other microphysics species of the hex
column route -- the ice and the number concentrations of a two-moment scheme
when the stream carries only ``qv``/``qc``/``qr`` -- take no boundary step
at all: their specified-zone values are whatever the transport left there.
That is the default (``--undriven-species-boundary free``), and it is
byte-identical to every earlier run.

WRF treats the same species differently.  Under ``specified`` boundaries a
moist species without boundary data (``.NOT. have_bcs_moist``, every moist
species but ``qv``) and a scalar without it (``.NOT. have_bcs_scalar``) take
``flow_dep_bdy`` after every RK scalar update (WRF 4.7.1
``dyn_em/solve_em.F:2341-2353`` moist, ``:2888-2946`` scalars):

* ``flow_dep_bdy`` (``share/module_bc.F:2335-2456``): at each specified-zone
  point, OUTFLOW through the point's outer face (the coupled mass flux
  ``ru_m``/``rv_m``, sign only) copies the value at the first point inside
  the specified zone (``field(i_inner, k, jbs+spec_zone)``); inflow, a zero
  flux included, sets ``0``.
* ``flow_dep_bdy_qnn`` (``module_bc.F:2460-2583``, the inflow writes at
  ``:2518, :2538, :2557, :2575``): the same with inflow ``ccn_conc`` instead
  of ``0``, for ``P_QNN`` (``solve_em.F:2898-2908``).  ``ccn_conc`` is the
  namelist value for WDM5/6/7 (``Registry.EM_COMMON:2668``, default 1.0e8,
  read by ``wdm6init`` at ``phys/module_physics_init.F:4604``) and
  ``nssl_cccn/1.225`` for the NSSL schemes (``module_physics_init.F:4670``,
  ``dyn_em/start_em.F:1756``).  WOOF spells WRF's ``P_QNN`` ``nn`` for the
  WDM schemes and ``qnn`` for NSSL (``woof.core.moist``).
* ``flow_dep_bdy_fixed_inflow`` (``module_bc.F:2586-2689``): inflow keeps
  the point's own value, for NTU's ``qdcn``/``qtcn``/``qnin``
  (``solve_em.F:2889-2897``).
* ``nwfa``/``nifa``/``nbca`` take ``flow_dep_bdy`` only under
  ``aer_init_opt = 0`` (``solve_em.F:2909-2946``, the truth table at ``:2915-2935``).  A
  hex run has no ``aer_init_opt``; WRF's default is ``0``, so an undriven
  aerosol number follows ``flow_dep_bdy`` here.  A driven one (the Thompson
  aerosol row's boundary block) is never touched.

``--undriven-species-boundary wrf-flow-dependent`` carries that onto the
mesh, applied right after each RK stage's ``bdy_adjust_scalars``, the MPAS
point that matches WRF's (the stage's scalar update):

* the specified zone is MPAS's own (``bdyMaskCell > nRelaxZone``);
* WRF's ``i_inner`` row is MPAS's own ``nearestRelaxationCell``
  (``mpas_atm_boundaries.F:702-751``): the nearest ring-5 cell, the first
  cell inside the specified zone, which the theta/rho reset already reads;
* WRF's outer face is the cell's edges to a cell of a higher ring or to no
  cell at all, and its sign test is the sign of the net outward split mass
  flux (``ruAvg``, the time-averaged coupled flux the scalar transport used,
  WRF's ``ru_m``) summed over those edges with their ``dvEdge``, level by
  level.  Outflow (``> 0``) copies the inner cell, otherwise the inflow
  value, exactly WRF's tie (a zero flux is inflow).

Only the microphysics row's own species are touched: ``qv`` (always driven),
every driven species, and any declared extra (chemistry, a lane's tracers)
keep their present treatment.  The receipt lists the species and their WRF
rule.  A partitioned run is refused: a rank's specified cells may read an
inner cell on another rank.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

#: The CLI values; ``free`` is the default and builds nothing.
MODES = ("free", "wrf-flow-dependent")
DEFAULT_MODE = "free"

#: WRF's per-species rule names.
ZERO_INFLOW = "flow_dep_bdy"
CCN_INFLOW = "flow_dep_bdy_qnn"
FIXED_INFLOW = "flow_dep_bdy_fixed_inflow"

#: NTU's three species that take ``flow_dep_bdy_fixed_inflow`` (solve_em.F:2890).
NTU_FIXED_INFLOW = frozenset({"qdcn", "qtcn", "qnin"})

#: WRF's WDM selectors plus 27, whose ``P_QNN`` WOOF spells ``nn`` (woof.core.moist).
WDM_LIKE = frozenset({14, 16, 26, 27})

#: The MPAS relaxation-zone width (mpas_atm_boundaries.F, nRelaxZone).
N_RELAX_ZONE = 5


@dataclass(frozen=True)
class SpeciesRule:
    index: int
    name: str
    rule: str
    inflow: float | None  # None: the point keeps its own value


def ccn_inflow(mp_physics: int | None, *, wdm6_ccn_conc: float, nssl_cccn: float) -> float:
    """WRF's ``grid%ccn_conc`` for the selected scheme, as WOOF's WRF route reads it."""

    if mp_physics in WDM_LIKE:
        return float(np.float32(wdm6_ccn_conc))
    from woof.core.nssl2_contract import nssl2_background_ccn_per_kg

    return float(nssl2_background_ccn_per_kg(nssl_cccn))


def species_rules(
    names: Sequence[str],
    driven: Sequence[int],
    *,
    row_names: Sequence[str],
    mp_physics: int | None,
    wdm6_ccn_conc: float = 1.0e8,
    nssl_cccn: float = 0.5e9,
) -> tuple[SpeciesRule, ...]:
    """The undriven microphysics species and WRF's rule for each, in model order."""

    driven_set = {int(i) for i in driven}
    row = set(row_names)
    qnn_name = "nn" if mp_physics in WDM_LIKE else "qnn"
    rules: list[SpeciesRule] = []
    for index, name in enumerate(names):
        if name == "qv" or index in driven_set or name not in row:
            continue
        if name in NTU_FIXED_INFLOW:
            rules.append(SpeciesRule(index, name, FIXED_INFLOW, None))
        elif name == qnn_name:
            if mp_physics is None:
                raise ValueError(
                    f"--undriven-species-boundary wrf-flow-dependent: {name} takes WRF's ccn_conc inflow, "
                    "which depends on the microphysics selector, and this run names none")
            rules.append(SpeciesRule(index, name, CCN_INFLOW,
                                     ccn_inflow(mp_physics, wdm6_ccn_conc=wdm6_ccn_conc, nssl_cccn=nssl_cccn)))
        else:
            rules.append(SpeciesRule(index, name, ZERO_INFLOW, 0.0))
    return tuple(rules)


def boundary_geometry(
    bdy_mask_cell: np.ndarray,
    nearest_relaxation_cell: np.ndarray,
    *,
    n_edges_on_cell: np.ndarray,
    edges_on_cell: np.ndarray,
    cells_on_edge: np.ndarray,
    dv_edge: np.ndarray,
    n_relax_zone: int = N_RELAX_ZONE,
) -> dict[str, np.ndarray]:
    """Specified cells, their inner cells, and their outer edges with outward weights.

    All connectivity is 0-based.  ``weights[s, j]`` is ``+dvEdge`` when the
    edge's normal points out of the cell (the cell is ``cellsOnEdge(1)``)
    and ``-dvEdge`` otherwise; padded slots carry edge 0 with weight 0.
    """

    mask = np.asarray(bdy_mask_cell, dtype=np.int64).reshape(-1)
    ncells = int(mask.size)
    nearest = np.asarray(nearest_relaxation_cell, dtype=np.int64).reshape(-1)
    counts = np.asarray(n_edges_on_cell, dtype=np.int64).reshape(-1)
    coe = np.asarray(cells_on_edge, dtype=np.int64).reshape(-1, 2)
    eoc = np.asarray(edges_on_cell, dtype=np.int64)
    eoc = eoc.reshape(eoc.shape[0], -1) if eoc.ndim == 2 else eoc.reshape(counts.size, -1)
    dv = np.asarray(dv_edge, dtype=np.float32).reshape(-1)
    spec = np.flatnonzero(mask > n_relax_zone).astype(np.int64)
    inner = nearest[spec] if spec.size else np.zeros(0, np.int64)
    bad = spec[(inner < 0) | (inner >= ncells)]
    if bad.size:
        raise ValueError(f"specified cell {int(bad[0])} has no nearest relaxation cell; the mesh's "
                         "bdyMaskCell rings are not the seven MPAS rings")
    if spec.size and not np.all(mask[inner] == n_relax_zone):
        raise ValueError("a specified cell's nearest relaxation cell is not on ring 5")
    outer: list[list[tuple[int, float]]] = []
    for cell in spec.tolist():
        found: list[tuple[int, float]] = []
        for slot in range(int(counts[cell])):
            edge = int(eoc[cell, slot])
            if edge < 0 or edge >= coe.shape[0]:
                continue
            first, second = int(coe[edge, 0]), int(coe[edge, 1])
            other = second if first == cell else first
            if other >= 0 and other < ncells and mask[other] <= mask[cell]:
                continue
            sign = np.float32(1.0) if first == cell else np.float32(-1.0)
            found.append((edge, float(sign * dv[edge])))
        outer.append(found)
    width = max((len(f) for f in outer), default=0) or 1
    edges = np.zeros((spec.size, width), np.int64)
    weights = np.zeros((spec.size, width), np.float32)
    for s, found in enumerate(outer):
        for j, (edge, weight) in enumerate(found):
            edges[s, j], weights[s, j] = edge, weight
    return {"spec": spec, "inner": inner, "edges": edges, "weights": weights,
            "outer_edge_counts": np.asarray([len(f) for f in outer], np.int64)}


def outflow_mask(xp, mass_flux, edges, weights):
    """``(nlev, nspec)`` True where the net outward mass flux is positive (WRF's outflow)."""

    net = (mass_flux[:, edges] * weights[None, :, :]).sum(axis=-1)
    return net > 0


def apply(xp, scalars, mass_flux, geometry: Mapping, rules: Sequence[SpeciesRule]) -> None:
    """WRF's flow-dependent specified zone for every rule's species, in place."""

    spec, inner = geometry["spec"], geometry["inner"]
    if not rules or int(spec.size) == 0:
        return
    out = outflow_mask(xp, mass_flux, geometry["edges"], geometry["weights"])
    for rule in rules:
        s = int(rule.index)
        inside = scalars[s:s + 1, :, inner][0]
        if rule.inflow is None:
            inflow = scalars[s:s + 1, :, spec][0]
        else:
            inflow = xp.full(inside.shape, rule.inflow, dtype=scalars.dtype)
        scalars[s:s + 1, :, spec] = xp.where(out, inside, inflow)[None]


class UndrivenSpeciesBoundary:
    """The resident form, bound to one regional runtime."""

    def __init__(self, runtime, rules: Sequence[SpeciesRule]):
        import cupy as cp

        self.cp = cp
        self.rules = tuple(rules)
        mesh = runtime.atmosphere.mesh
        masks = runtime.masks_host
        host = boundary_geometry(
            masks.bdy_mask_cell, masks.nearest_relaxation_cell,
            n_edges_on_cell=cp.asnumpy(mesh.n_edges_on_cell),
            edges_on_cell=cp.asnumpy(mesh.edges_on_cell).reshape(-1, int(mesh.max_edges)),
            cells_on_edge=cp.asnumpy(mesh.cells_on_edge),
            dv_edge=cp.asnumpy(mesh.dv_edge))
        self.counts = host["outer_edge_counts"]
        self.geometry = {
            "spec": cp.asarray(host["spec"]),
            "inner": cp.asarray(host["inner"]),
            "edges": cp.asarray(host["edges"]),
            "weights": cp.asarray(host["weights"]),
        }
        self.n_spec = int(host["spec"].size)
        self.applications = 0

    def apply(self, scalars, mass_flux) -> None:
        if mass_flux is None:
            raise RuntimeError("--undriven-species-boundary wrf-flow-dependent needs the stage's mass flux; "
                               "this boundary call carries none")
        apply(self.cp, scalars, mass_flux, self.geometry, self.rules)
        self.applications += 1

    def receipt(self) -> dict:
        return {
            "mode": "wrf-flow-dependent",
            "law": "WRF 4.7.1 share/module_bc.F flow_dep_bdy (2335-2456), flow_dep_bdy_qnn (2460-2583), "
                   "flow_dep_bdy_fixed_inflow (2586-2689); dyn_em/solve_em.F 2341-2353, 2888-2946",
            "species": [{"name": r.name, "index": r.index, "rule": r.rule,
                         "inflow": "own value" if r.inflow is None else r.inflow} for r in self.rules],
            "specified_cells": self.n_spec,
            "outer_edges_per_cell": {"min": int(self.counts.min(initial=0)),
                                     "max": int(self.counts.max(initial=0))},
            # A specified cell with no edge to a higher ring or to no cell (a cull notch) has no outer face:
            # its net outward flux is 0, which WRF's tie reads as inflow, so it always takes the inflow value.
            "cells_without_outer_edge": int((self.counts == 0).sum()),
            "inner_cell": "nearestRelaxationCell (mpas_atm_boundaries.F:702-751)",
            "applications": self.applications,
        }


def _physics_value(host: Mapping, key: str, default: float) -> float:
    physics = dict((host.get("constructor_values") or {}).get("physics") or {})
    value = physics.get(key)
    return float(default if value is None else value)


def attach(driver, host: Mapping, mode: str | None):
    """Bind the rule to a regional run; None (nothing built) for ``free``."""

    if mode in (None, DEFAULT_MODE):
        return None
    if mode not in MODES:
        raise ValueError(f"--undriven-species-boundary {mode!r} must be one of {', '.join(MODES)}")
    runtime = getattr(driver, "regional_v841", None)
    if runtime is None:
        raise ValueError("--undriven-species-boundary sets a regional mesh's specified zone; this mesh is global")
    if getattr(getattr(runtime.padded, "base", runtime.padded), "partition_layout_v841", None) is not None:
        raise ValueError("--undriven-species-boundary wrf-flow-dependent needs the whole specified zone on one "
                         "card: a rank's specified cell may read an inner cell another rank owns")
    from .species_row import species_row_for_names

    names = tuple(host["scalar_names"])
    row = species_row_for_names(names)
    extras = {d.name for d in row.declared_extras()}
    row_names = tuple(n for n in row.names() if n not in extras)
    driven = sorted({first + i for first, _pool, count in runtime.scalar_segments for i in range(count)})
    # RunConfig's defaults (woof.config: wdm6_ccn_conc 1.0e8, nssl_cccn 0.5e9) unless the selection assigns them.
    rules = species_rules(
        names, driven, row_names=row_names, mp_physics=getattr(row, "mp_physics", None),
        wdm6_ccn_conc=_physics_value(host, "wdm6_ccn_conc", 1.0e8),
        nssl_cccn=_physics_value(host, "nssl_cccn", 0.5e9))
    runtime.undriven_species_boundary = UndrivenSpeciesBoundary(runtime, rules)
    return runtime.undriven_species_boundary.receipt()


__all__ = ["CCN_INFLOW", "DEFAULT_MODE", "FIXED_INFLOW", "MODES", "SpeciesRule", "UndrivenSpeciesBoundary",
           "ZERO_INFLOW", "apply", "attach", "boundary_geometry", "ccn_inflow", "outflow_mask", "species_rules"]
