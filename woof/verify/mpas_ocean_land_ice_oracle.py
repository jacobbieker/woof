"""Cases and the stream layout for the MPAS-Ocean land-ice oracle.

``tools/mpas_ocean_oracle/run_land_ice.F90`` drives MPAS-Model v8.4.2's
``ocn_surface_land_ice_fluxes_init``, ``_build_arrays``, ``_thick``, ``_vel``
and ``_tracers`` and ``ocn_effective_density_in_land_ice_update``
(``mpas_ocn_surface_land_ice_fluxes.F`` and
``mpas_ocn_effective_density_in_land_ice.F`` compiled verbatim with the real
equation of state) with gfortran -O0 -ffp-contract=off at RKIND 4 and 8. The
stream is :mod:`woof.verify.mpas_ocean_stream`'s; a field's time level 2 is
``NAME@2``.

The cases take every branch: ISOMIP and Jenkins in 'standalone' (cells under
land ice and open, the interface salinity at its floor, an ocean salinity at
or below zero, pressures below the cavity's reference pressure, halo cells
beyond ``nCellsSolve`` the builder leaves alone, the var_arrays' indices
swapped), 'coupled' (the coupler's fluxes added, the effective density with
masked and missing neighbours) and 'pressure_only' (nothing written). Every
output starts from noise, so a word a routine does not write is checked too.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_eos_oracle as _eos
from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_land_ice")
CASES_PER_PRECISION = True
REAL, INT, DIM, SCRATCH = _stream.REAL, _stream.INT, _stream.DIM, _stream.SCRATCH
FIELDS = (
    ("mesh", "nVertLevels", DIM, ()), ("mesh", "nCells", DIM, ()), ("mesh", "nEdges", DIM, ()),
    ("mesh", "nCellsSolve", DIM, ()),
    ("mesh", "cellsOnCell", INT, ("ME", "C1")), ("mesh", "nEdgesOnCell", INT, ("C1",)),
    ("mesh", "cellMask", INT, ("L", "C1")),
    ("state", "ssh", REAL, ("C1",)), ("state", "ssh@2", REAL, ("C1",)),
    ("state", "accumulatedLandIceMass", REAL, ("C1",)), ("state", "accumulatedLandIceMass@2", REAL, ("C1",)),
    ("state", "accumulatedLandIceHeat", REAL, ("C1",)), ("state", "accumulatedLandIceHeat@2", REAL, ("C1",)),
    ("state", "effectiveDensityInLandIce", REAL, ("C1",)), ("state", "effectiveDensityInLandIce@2", REAL, ("C1",)),
    ("diagnostics", "index_landIceBoundaryLayerTemperature", DIM, ()),
    ("diagnostics", "index_landIceBoundaryLayerSalinity", DIM, ()),
    ("diagnostics", "index_landIceHeatTransferVelocity", DIM, ()),
    ("diagnostics", "index_landIceSaltTransferVelocity", DIM, ()),
    ("diagnostics", "landIceBoundaryLayerTracers", REAL, (2, "C1")),
    ("diagnostics", "landIceTracerTransferVelocities", REAL, (2, "C1")),
    ("diagnostics", "landIceFrictionVelocity", REAL, ("C1",)),
    ("diagnostics", "topDrag", REAL, ("E1",)), ("diagnostics", "topDragMagnitude", REAL, ("C1",)),
    ("forcing", "landIcePressure", REAL, ("C1",)), ("forcing", "landIceFraction", REAL, ("C1",)),
    ("forcing", "landIceMask", INT, ("C1",)),
    ("forcing", "index_landIceInterfaceTemperature", DIM, ()), ("forcing", "index_landIceInterfaceSalinity", DIM, ()),
    ("forcing", "landIceInterfaceTracers", REAL, (2, "C1")),
    ("forcing", "landIceFreshwaterFlux", REAL, ("C1",)), ("forcing", "landIceHeatFlux", REAL, ("C1",)),
    ("forcing", "heatFluxToLandIce", REAL, ("C1",)),
    ("forcing", "surfaceThicknessFlux", REAL, ("C1",)), ("forcing", "surfaceStress", REAL, ("E1",)),
    ("forcing", "surfaceStressMagnitude", REAL, ("C1",)),
    ("tracersSurfaceFlux", "activeTracersSurfaceFlux", REAL, ("T", "C1")),
    ("scratch", "effectiveDensityScratch", SCRATCH, ("C1",)),
)
#: the package each field belongs to (Registry.xml): landIceFluxesPKG, landIceCouplingPKG
_FLUXES = {"accumulatedLandIceMass", "accumulatedLandIceHeat", "landIceBoundaryLayerTracers",
           "landIceTracerTransferVelocities", "landIceFrictionVelocity", "topDrag", "topDragMagnitude",
           "landIceInterfaceTracers", "landIceFreshwaterFlux", "landIceHeatFlux", "heatFluxToLandIce"}
_COUPLING = {"effectiveDensityInLandIce"}
#: Registry.xml defaults (v8.4.2) of the land-ice and equation-of-state configs
BASE_CONFIG = {
    **_eos.DEFAULTS, "config_eos_type": "linear",
    "config_land_ice_flux_mode": "standalone", "config_land_ice_flux_formulation": "Jenkins",
    "config_land_ice_flux_useHollandJenkinsAdvDiff": False,
    "config_land_ice_flux_attenuation_coefficient": 10.0, "config_land_ice_flux_boundaryLayerThickness": 10.0,
    "config_land_ice_flux_boundaryLayerNeighborWeight": 0.0, "config_land_ice_flux_cp_ice": 2.009e3,
    "config_land_ice_flux_rho_ice": 918.0, "config_land_ice_flux_topDragCoeff": 2.5e-3,
    "config_land_ice_flux_ISOMIP_gammaT": 1.0e-4, "config_land_ice_flux_rms_tidal_velocity": 5.0e-2,
    "config_land_ice_flux_jenkins_heat_transfer_coefficient": 0.011,
    "config_land_ice_flux_jenkins_salt_transfer_coefficient": 3.1e-4,
}
_CAVITY = {"config_land_ice_cavity_freezing_temperature_reference_pressure": 1.0e6}
#: (name, seed, nCells, nTracers, cells solved (nCellsSolve), swapped indices, dt, overrides)
CASES = (
    ("standalone_isomip", 20262101, 40, 2, 40, False, 600.0, {"config_land_ice_flux_formulation": "ISOMIP"}),
    ("standalone_jenkins", 20262102, 44, 3, 44, False, 300.0, {}),
    ("standalone_jenkins_halo_configured", 20262103, 48, 2, 31, True, 1800.0, {
        **_CAVITY, "config_density0": 1027.5, "config_land_ice_flux_cp_ice": 2.1e3,
        "config_land_ice_flux_jenkins_heat_transfer_coefficient": 0.02,
        "config_land_ice_flux_jenkins_salt_transfer_coefficient": 5.0e-4}),
    ("standalone_isomip_halo_configured", 20262104, 36, 2, 25, True, 120.0, {
        **_CAVITY, "config_land_ice_flux_formulation": "ISOMIP", "config_land_ice_flux_ISOMIP_gammaT": 3.0e-4,
        "config_density0": 1030.0}),
    ("coupled", 20262105, 42, 2, 42, False, 600.0, {"config_land_ice_flux_mode": "coupled"}),
    ("pressure_only", 20262106, 30, 2, 30, False, 600.0, {"config_land_ice_flux_mode": "pressure_only"}),
)


def present(case: dict, name: str) -> bool:
    """Whether a field's package is on (MPAS allocates it only then)."""
    mode = case["config"]["config_land_ice_flux_mode"]
    base = name.split("@")[0]
    if base in _FLUXES:
        return mode in ("standalone", "coupled")
    if base in _COUPLING:
        return mode == "coupled"
    return True


def make_case(name, seed, n_cells, n_tracers, solved, swapped, dt, overrides, precision="r4") -> dict:
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    nc, nt, ne, nl, me = n_cells, n_tracers, 3 * n_cells, 4, 6
    config = dict(BASE_CONFIG, **overrides)

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    mask = (rs.random_sample(nc + 1) < 0.6).astype(np.int32)
    mask[nc] = 0
    fraction = np.where(mask == 1, uniform(0.5, 1.0, nc + 1), 0.0)
    # pressures from below the reference pressure to deep cavities
    pressure = np.where(mask == 1, uniform(2.0e5, 8.0e6, nc + 1), uniform(0.0, 1.0e5, nc + 1))
    bl = np.empty((nc + 1, 2))
    bl[:, 0] = uniform(-2.6, 1.5, nc + 1)
    bl[:, 1] = uniform(30.0, 35.0, nc + 1)
    # salinities that drive the interface salinity to its 0.001 floor, zero and below zero
    fresh = rs.random_sample(nc + 1) < 0.12
    bl[fresh, 1] = uniform(-0.5, 5.0e-4, int(fresh.sum()))
    tv = np.empty((nc + 1, 2))
    tv[:, 0] = uniform(1.0e-5, 4.0e-4, nc + 1)
    tv[:, 1] = tv[:, 0] * uniform(0.01, 0.05, nc + 1)
    it, isl = (2, 1) if swapped else (1, 2)
    if swapped:
        bl, tv = bl[:, ::-1].copy(), tv[:, ::-1].copy()
    cells_on_cell = rs.randint(1, nc + 1, (nc + 1, me)).astype(np.int32)
    cells_on_cell[rs.random_sample((nc + 1, me)) < 0.1] = nc + 1
    n_edges_on_cell = rs.randint(3, me + 1, nc + 1).astype(np.int32)
    cell_mask = (rs.random_sample((nc + 1, nl)) < 0.8).astype(np.int32)
    cell_mask[nc] = 0
    ssh2 = np.where(mask == 1, uniform(-600.0, -20.0, nc + 1), uniform(-2.0, 2.0, nc + 1))
    dims = {"nCells": nc, "nEdges": ne, "nVertLevels": nl, "nTracers": nt, "maxEdges": me}
    case = {"name": name, "precision": precision, "dims": dims, "config": config, "dt": dt}
    values = {
        "nVertLevels": nl, "nCells": nc, "nEdges": ne, "nCellsSolve": solved,
        "cellsOnCell": cells_on_cell, "nEdgesOnCell": n_edges_on_cell, "cellMask": cell_mask, "ssh@2": ssh2,
        "accumulatedLandIceMass": uniform(0.0, 1.0e4, nc + 1), "accumulatedLandIceHeat": uniform(-1.0e9, 1.0e9, nc + 1),
        "effectiveDensityInLandIce": uniform(900.0, 1030.0, nc + 1),
        "index_landIceBoundaryLayerTemperature": it, "index_landIceBoundaryLayerSalinity": isl,
        "index_landIceHeatTransferVelocity": it, "index_landIceSaltTransferVelocity": isl,
        "landIceBoundaryLayerTracers": bl, "landIceTracerTransferVelocities": tv,
        "landIcePressure": pressure, "landIceFraction": fraction, "landIceMask": mask,
        "index_landIceInterfaceTemperature": it, "index_landIceInterfaceSalinity": isl,
        # the coupler's fluxes in 'coupled' (noise the builder overwrites in 'standalone')
        "landIceFreshwaterFlux": uniform(-1.0e-4, 1.0e-4, nc + 1), "landIceHeatFlux": uniform(-50.0, 50.0, nc + 1),
    }
    pools: dict[str, dict] = {}
    for pool, field, kind, fshape in FIELDS:
        if kind == DIM:
            value = values[field]
        elif kind == SCRATCH or not present(case, field):
            value = None
        elif field in values:
            value = values[field]
        else:
            value = uniform(-1.0, 1.0, _stream.shape(case, fshape))
        if kind == REAL and value is not None:
            value = np.asarray(value).astype(real)
        pools.setdefault(pool, {})[field] = value
    case["pools"] = pools
    return case


def cases(precision: str = "r4") -> list[dict]:
    return [make_case(*row, precision=precision) for row in CASES]


def input_bytes(case: dict) -> bytes:
    return _stream.input_bytes(FIELDS, case)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    return _stream.read_outputs(FIELDS, raw, case)


word_mismatches = _stream.word_mismatches


def load_fixture(precision: str) -> dict[str, dict[str, np.ndarray]]:
    return _tend.load_fixture(precision, FIXTURE_DIR)


def run_gpu(case: dict) -> dict[str, np.ndarray]:
    """Run woof.hex.mpas_ocean.land_ice_v842 on ``case`` as run_land_ice.F90 drives the Fortran."""
    import cupy as cp
    from woof.hex.mpas_ocean import land_ice_v842 as li

    dev = _stream.on_device(FIELDS, case)
    m, f, d = dev["mesh"], dev["forcing"], dev["diagnostics"]
    config = case["config"]
    nc, ne = int(m["nCells"]), int(m["nEdges"])
    li.build_arrays(dt=case["dt"], mesh=m, state=dev["state"], diagnostics=d, forcing=f, configs=config)
    li.thick_tend(forcing=f, surface_thickness_flux=f["surfaceThicknessFlux"], n_cells=nc, configs=config)
    li.vel_tend(diagnostics=d, surface_stress=f["surfaceStress"], surface_stress_magnitude=f["surfaceStressMagnitude"],
                n_edges=ne, n_cells=nc, configs=config)
    li.tracers_tend(forcing=f, tracers_surface_flux=dev["tracersSurfaceFlux"]["activeTracersSurfaceFlux"], n_cells=nc,
                    configs=config)
    li.effective_density_update(mesh=m, forcing=f, state=dev["state"], configs=config)
    return {f"{pool}/{name}": cp.asnumpy(dev[pool][name]) for pool, name, kind, _ in FIELDS
            if kind in (REAL, INT) and dev[pool][name] is not None}


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases(precision):
        got = run_gpu(case)
        want = reference[case["name"]]
        report[case["name"]] = {name: word_mismatches(got[name], want[name]) for name in want}
    return report
