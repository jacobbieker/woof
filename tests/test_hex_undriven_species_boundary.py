"""WRF's flow-dependent specified zone for a regional hex run's undriven species.

``woof.hex.undriven_species_boundary`` carries WRF's ``flow_dep_bdy`` (WRF
4.7.1 share/module_bc.F:2335-2456) and its ``_qnn``/``_fixed_inflow`` variants
onto the mesh.  On a synthetic square patch whose rings count inward from the
edge as bdyMaskCell does, every specified cell but the four diagonal corners
matches WOOF's own port of WRF's routine (woof.ingest.lateral_bc) on the same
field and face fluxes, word for word; the corners, which WRF gives to its
y sides, take the net flux of both outer faces.  The species rules, the
default (``free``: nothing built) and the door's refusals are held too.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import undriven_species_boundary as usb
from woof.hex.regional_v841 import derive_regional_masks
from woof.ingest.lateral_bc import _apply_flow_dependent_boundary_generic

NX, NY, NLEV = 16, 14, 3
SPEC_ZONE = 2  # MPAS rings 6-7


def _patch():
    j, i = np.mgrid[0:NY, 0:NX]
    d = np.minimum.reduce([i, j, NX - 1 - i, NY - 1 - j])
    mask = np.where(d < 7, 7 - d, 0).reshape(-1)
    ncells = NX * NY
    cid = lambda jj, ii: jj * NX + ii if 0 <= jj < NY and 0 <= ii < NX else ncells  # noqa: E731
    # 8-connected cellsOnCell for MPAS's nearestRelaxationCell search (a hex cell's six neighbours
    # reach the diagonal ring-5 cell in two steps; four would not on a square patch).
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    coc = np.array([[cid(jj + a, ii + b) for a, b in offsets]
                    for jj in range(NY) for ii in range(NX)], np.int64)
    coc = np.where(coc == ncells, -1, coc)
    mask_mesh = SimpleNamespace(
        bdyMaskCell=mask, bdyMaskEdge=np.zeros(1, np.int64), bdyMaskVertex=np.zeros(1, np.int64),
        nEdgesOnCell=np.full(ncells, 8), cellsOnCell=coc,
        xCell=i.reshape(-1).astype(np.float32), yCell=j.reshape(-1).astype(np.float32),
        zCell=np.zeros(ncells, np.float32))
    masks = derive_regional_masks(mask_mesh, np.dtype(np.float32))
    # Faces: x-normal edges (west cell, east cell) then y-normal edges (south cell, north cell).
    coe, ew, ns = [], {}, {}
    for jj in range(NY):
        for ii in range(NX + 1):
            ew[jj, ii] = len(coe)
            coe.append((cid(jj, ii - 1), cid(jj, ii)))
    for jj in range(NY + 1):
        for ii in range(NX):
            ns[jj, ii] = len(coe)
            coe.append((cid(jj - 1, ii), cid(jj, ii)))
    eoc = np.array([[ew[jj, ii], ew[jj, ii + 1], ns[jj, ii], ns[jj + 1, ii]]
                    for jj in range(NY) for ii in range(NX)], np.int64)
    geometry = usb.boundary_geometry(
        masks.bdy_mask_cell, masks.nearest_relaxation_cell, n_edges_on_cell=np.full(ncells, 4),
        edges_on_cell=eoc, cells_on_edge=np.array(coe, np.int64), dv_edge=np.ones(len(coe), np.float32))
    return masks, geometry, ew, ns, len(coe)


def _fluxes(rng, ew, ns, nedges):
    ru = rng.standard_normal((NLEV, nedges)).astype(np.float32)
    ru[:, rng.integers(0, nedges, 20)] = 0.0  # WRF's tie: a zero flux is inflow
    u = np.zeros((NLEV, NY, NX + 1), np.float32)
    v = np.zeros((NLEV, NY + 1, NX), np.float32)
    for (jj, ii), e in ew.items():
        u[:, jj, ii] = ru[:, e]
    for (jj, ii), e in ns.items():
        v[:, jj, ii] = ru[:, e]
    return ru, u, v


def _diagonal_corner(jj, ii):
    dj = min(jj, NY - 1 - jj)
    di = min(ii, NX - 1 - ii)
    return dj < SPEC_ZONE and di < SPEC_ZONE and dj == di


@pytest.mark.parametrize("inflow", [0.0, 1.0e8])
def test_the_mesh_rule_is_wrf_flow_dep_bdy_on_every_specified_cell_but_the_corners(inflow):
    masks, geometry, ew, ns, nedges = _patch()
    rng = np.random.default_rng(7)
    ru, u, v = _fluxes(rng, ew, ns, nedges)
    field = rng.uniform(1.0, 2.0, (NLEV, NY, NX)).astype(np.float32)
    wrf = field.copy()
    _apply_flow_dependent_boundary_generic(wrf, u, v, SPEC_ZONE, np, inflow)
    scalars = field.reshape(1, NLEV, NY * NX).copy()
    rule = usb.SpeciesRule(0, "qi", usb.ZERO_INFLOW if inflow == 0.0 else usb.CCN_INFLOW, inflow)
    usb.apply(np, scalars, ru, geometry, (rule,))
    mesh = scalars.reshape(NLEV, NY, NX)
    corners = np.array([[_diagonal_corner(jj, ii) for ii in range(NX)] for jj in range(NY)])
    assert np.array_equal(mesh[:, ~corners].view(np.uint32), wrf[:, ~corners].view(np.uint32))
    assert not np.array_equal(field[:, masks.bdy_mask_cell.reshape(NY, NX) > 5],
                              mesh[:, masks.bdy_mask_cell.reshape(NY, NX) > 5])
    # The corners: the net of both outer faces, the same inner cell WRF reads.
    for jj, ii in zip(*np.nonzero(corners)):
        c = jj * NX + ii
        s = int(np.flatnonzero(geometry["spec"] == c)[0])
        assert geometry["outer_edge_counts"][s] == 2
        inner = int(geometry["inner"][s])
        assert (inner // NX, inner % NX) == (min(max(jj, SPEC_ZONE), NY - 1 - SPEC_ZONE),
                                             min(max(ii, SPEC_ZONE), NX - 1 - SPEC_ZONE))
        net = (ru[:, geometry["edges"][s]] * geometry["weights"][s]).sum(-1)
        expect = np.where(net > 0, field.reshape(NLEV, -1)[:, inner], np.float32(inflow))
        assert np.array_equal(mesh[:, jj, ii], expect)


def test_fixed_inflow_keeps_its_own_value_and_the_interior_is_untouched():
    masks, geometry, ew, ns, nedges = _patch()
    rng = np.random.default_rng(3)
    ru, _u, _v = _fluxes(rng, ew, ns, nedges)
    field = rng.uniform(1.0, 2.0, (2, NLEV, NY * NX)).astype(np.float32)
    scalars = field.copy()
    usb.apply(np, scalars, ru, geometry, (usb.SpeciesRule(1, "qdcn", usb.FIXED_INFLOW, None),))
    assert np.array_equal(scalars[0], field[0])
    spec = geometry["spec"]
    out = usb.outflow_mask(np, ru, geometry["edges"], geometry["weights"])
    assert np.array_equal(scalars[1][:, spec], np.where(out, field[1][:, geometry["inner"]], field[1][:, spec]))
    others = np.setdiff1d(np.arange(NY * NX), spec)
    assert np.array_equal(scalars[1][:, others], field[1][:, others])


def test_the_species_rules_follow_wrf_solve_em():
    names = ("qv", "qc", "qr", "qi", "qs", "qg", "nn", "nc", "nr", "tracer_x")
    rules = usb.species_rules(names, driven=(0, 1, 2), row_names=names[:-1], mp_physics=16,
                              wdm6_ccn_conc=2.0e8)
    assert [(r.name, r.rule) for r in rules] == [
        ("qi", usb.ZERO_INFLOW), ("qs", usb.ZERO_INFLOW), ("qg", usb.ZERO_INFLOW),
        ("nn", usb.CCN_INFLOW), ("nc", usb.ZERO_INFLOW), ("nr", usb.ZERO_INFLOW)]
    assert [r.inflow for r in rules if r.name == "nn"] == [2.0e8]
    from woof.core.nssl2_contract import nssl2_background_ccn_per_kg
    nssl = usb.species_rules(("qv", "qc", "qnn", "nn"), driven=(0,), row_names=("qv", "qc", "qnn", "nn"),
                             mp_physics=18, nssl_cccn=0.6e9)
    assert {r.name: r.rule for r in nssl} == {"qc": usb.ZERO_INFLOW, "qnn": usb.CCN_INFLOW, "nn": usb.ZERO_INFLOW}
    assert [r.inflow for r in nssl if r.name == "qnn"] == [float(nssl2_background_ccn_per_kg(0.6e9))]
    ntu = usb.species_rules(("qv", "qdcn", "qtcn", "qnin", "qccn"), driven=(0,),
                            row_names=("qv", "qdcn", "qtcn", "qnin", "qccn"), mp_physics=None)
    assert {r.name: r.rule for r in ntu} == {"qdcn": usb.FIXED_INFLOW, "qtcn": usb.FIXED_INFLOW,
                                              "qnin": usb.FIXED_INFLOW, "qccn": usb.ZERO_INFLOW}
    with pytest.raises(ValueError, match="ccn_conc"):
        usb.species_rules(("qv", "qnn"), driven=(0,), row_names=("qv", "qnn"), mp_physics=None)


def test_free_builds_nothing_and_the_door_passes_nothing():
    assert usb.attach(None, {}, "free") is None
    assert usb.attach(None, {}, None) is None
    from woof.hex.forecast_door import _resolve_undriven_species_boundary
    assert _resolve_undriven_species_boundary(SimpleNamespace(undriven_species_boundary="free")) is None
    assert _resolve_undriven_species_boundary(SimpleNamespace()) is None
    assert _resolve_undriven_species_boundary(
        SimpleNamespace(undriven_species_boundary="wrf-flow-dependent")) == "wrf-flow-dependent"
    with pytest.raises(Exception, match="local-timestep"):
        _resolve_undriven_species_boundary(
            SimpleNamespace(undriven_species_boundary="wrf-flow-dependent", local_timestep=True))
    with pytest.raises(ValueError, match="global"):
        usb.attach(SimpleNamespace(regional_v841=None), {}, "wrf-flow-dependent")


# -- the MPAS v8.4.2 suite route (R2) ------------------------------------------


def _lbc_file(path, names):
    from netCDF4 import Dataset

    with Dataset(path, "w") as dataset:
        dataset.createDimension("nCells", 4)
        dataset.createDimension("nVertLevels", 2)
        for name in names:
            dataset.createVariable(f"lbc_{name}", "f4", ("nCells", "nVertLevels"))
    return str(path)


def test_the_suite_route_drives_only_the_leading_species_the_files_hold(tmp_path):
    names = ("qv", "qc", "qr", "qi", "qs", "qg")
    first = _lbc_file(tmp_path / "a.nc", ("qv", "qc", "qr", "u"))
    second = _lbc_file(tmp_path / "b.nc", ("qv", "qc", "qr"))
    assert usb.lbc_driven_prefix([first, second], names) == ("lbc_qv", "lbc_qc", "lbc_qr")
    # A held species after a missing one would put a pool row on the wrong species.
    gap = _lbc_file(tmp_path / "gap.nc", ("qv", "qc", "qi"))
    with pytest.raises(ValueError, match="leading block"):
        usb.lbc_driven_prefix([gap], names)
    # The held block may not change between boundary times.
    more = _lbc_file(tmp_path / "more.nc", ("qv", "qc", "qr", "qi"))
    with pytest.raises(ValueError, match="change"):
        usb.lbc_driven_prefix([first, more], names)
    with pytest.raises(ValueError, match="no boundary files"):
        usb.lbc_driven_prefix([], names)


def test_the_suite_schemes_take_wrf_flow_dep_bdy_for_what_the_files_do_not_hold():
    from woof.hex.species_row import species_row_for_names

    # MPAS-A v8.4.2 WSM6 (mesoscale_reference), Thompson (convection_permitting) and aerosol-aware Thompson
    # (nwfa/nifa under WRF's default aer_init_opt = 0), with the qv/qc/qr stream init_atmosphere writes
    six = ("qv", "qc", "qr", "qi", "qs", "qg")
    for scheme, names in (("mp_wsm6", six), ("mp_thompson", six + ("ni", "nr")),
                          ("mp_thompson_aerosols", six + ("ni", "nr", "nc", "nwfa", "nifa"))):
        row = species_row_for_names(names)
        assert row.scheme_name == scheme
        rules = usb.species_rules(names, (0, 1, 2), row_names=row.names(), mp_physics=row.mp_physics)
        assert [(r.name, r.rule, r.inflow) for r in rules] == [
            (n, usb.ZERO_INFLOW, 0.0) for n in names[3:]]


def test_the_suite_route_attach_builds_nothing_for_free_and_refuses_a_global_mesh():
    assert usb.attach_v842(None, ("qv",), "free") is None
    assert usb.attach_v842(None, ("qv",), None) is None
    with pytest.raises(ValueError, match="global"):
        usb.attach_v842(SimpleNamespace(regional_v841=None), ("qv",), "wrf-flow-dependent")
    with pytest.raises(ValueError, match="must be one of"):
        usb.attach_v842(SimpleNamespace(regional_v841=None), ("qv",), "nearest")


def test_the_suite_route_door_carries_the_rule_on_its_one_card_runtime(monkeypatch):
    from woof.hex import forecast_door as door

    for key in ("WOOF_HEX_HOSTS",):
        monkeypatch.delenv(key, raising=False)
    v842 = SimpleNamespace(table_key="mpasphys_v842")
    rrfs = SimpleNamespace(table_key="mpasphys_rrfs")
    # Correct mode (the umbrella's default) turns it on for a regional suite run; exact leaves it off.
    assert door._resolve_undriven_species_boundary(SimpleNamespace(lbc_dir="x"), v842) == "wrf-flow-dependent"
    assert door._resolve_undriven_species_boundary(
        SimpleNamespace(lbc_dir="x", physics_mode="exact"), v842) is None
    assert door._resolve_undriven_species_boundary(
        SimpleNamespace(lbc_dir="x", physics_mode="exact", undriven_species_boundary="wrf-flow-dependent"),
        v842) == "wrf-flow-dependent"
    # Seats that do not carry it: off under the umbrella ...
    for extra in ({"hosts": "a,b"}, {"samehost_devices": "0,1"}):
        assert door._resolve_undriven_species_boundary(SimpleNamespace(lbc_dir="x", **extra), v842) is None
    assert door._resolve_undriven_species_boundary(SimpleNamespace(lbc_dir="x"), rrfs) is None
    # ... and an explicit flag is refused by name.
    with pytest.raises(Exception, match="fork dycore"):
        door._resolve_undriven_species_boundary(
            SimpleNamespace(lbc_dir="x", undriven_species_boundary="wrf-flow-dependent"), rrfs)
    explicit = SimpleNamespace(lbc_dir="x", hosts="a,b", undriven_species_boundary="wrf-flow-dependent")
    explicit.undriven_species_boundary_mode = door._resolve_undriven_species_boundary(explicit, v842)
    assert "--undriven-species-boundary" in door.suite_route_seat_refusal(explicit, environ={})
    one_card = SimpleNamespace(undriven_species_boundary_mode="wrf-flow-dependent")
    assert door.suite_route_seat_refusal(one_card, environ={}) is None
