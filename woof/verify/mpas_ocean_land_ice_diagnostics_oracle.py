"""Cases and the stream layout for ``ocn_diagnostic_solve`` with the land-ice fluxes on.

The reference is the diagnostics oracle (``tools/mpas_ocean_oracle/run_diagnostics.F90``,
``ocn_diagnostic_solve`` and ``ocn_compute_land_ice_flux_input_fields`` cut
verbatim out of v8.4.2's ``mpas_ocn_diagnostics.F``) in the 'standalone' and
'coupled' land-ice modes: the field table is
:mod:`woof.verify.mpas_ocean_diagnostics_oracle`'s plus the land-ice flux
package's diagnostics, the mesh's ``cellsOnCell`` and ``cellMask``, and the
boundary layer's two scratch fields.

The cases take each branch of the input fields: ISOMIP and Jenkins, the
boundary layer's neighbours weighted (with masked and missing neighbours) or
not, a boundary layer thinner and thicker than the top layers, nested halos,
and the land-ice attenuation coefficient. The kinetic energy beyond the
loops' extents is drawn non-negative (it feeds a square root).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_diagnostics_oracle as _diag
from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_land_ice_diagnostics")
CASES_PER_PRECISION = True
REAL, INT, DIM, SCRATCH = _stream.REAL, _stream.INT, _stream.DIM, _stream.SCRATCH
_EXTRA = (
    ("mesh", "cellsOnCell", INT, ("ME", "C1")), ("mesh", "cellMask", INT, ("L", "C1")),
    ("diagnostics", "index_landIceBoundaryLayerTemperature", DIM, ()),
    ("diagnostics", "index_landIceBoundaryLayerSalinity", DIM, ()),
    ("diagnostics", "index_landIceHeatTransferVelocity", DIM, ()),
    ("diagnostics", "index_landIceSaltTransferVelocity", DIM, ()),
    ("diagnostics", "landIceFrictionVelocity", REAL, ("C1",)),
    ("diagnostics", "topDrag", REAL, ("E1",)), ("diagnostics", "topDragMagnitude", REAL, ("C1",)),
    ("diagnostics", "landIceBoundaryLayerTracers", REAL, (2, "C1")),
    ("diagnostics", "landIceTracerTransferVelocities", REAL, (2, "C1")),
    ("scratch", "boundaryLayerTemperatureScratch", SCRATCH, ("C1",)),
    ("scratch", "boundaryLayerSalinityScratch", SCRATCH, ("C1",)),
)
FIELDS = _diag.FIELDS + _EXTRA
ARRAYS = tuple((pool, name) for pool, name, kind, _ in FIELDS if kind in (REAL, INT))
_LAND_ICE = {"config_land_ice_flux_formulation": "Jenkins", "config_land_ice_flux_attenuation_coefficient": 10.0,
             "config_land_ice_flux_boundaryLayerThickness": 10.0,
             "config_land_ice_flux_boundaryLayerNeighborWeight": 0.0, "config_land_ice_flux_topDragCoeff": 2.5e-3,
             "config_land_ice_flux_rms_tidal_velocity": 5.0e-2,
             "config_land_ice_flux_jenkins_heat_transfer_coefficient": 0.011,
             "config_land_ice_flux_jenkins_salt_transfer_coefficient": 3.1e-4}
#: (name, seed, nCells, nVertLevels, nested halos, config overrides, swapped var_array indices)
CASES = (
    ("standalone_jenkins_neighbours", 20262201, 48, 12, True,
     {**_LAND_ICE, "config_land_ice_flux_mode": "standalone",
      "config_land_ice_flux_boundaryLayerNeighborWeight": 0.5}, False),
    ("standalone_isomip_thick_layer", 20262202, 40, 10, False,
     {**_LAND_ICE, "config_land_ice_flux_mode": "standalone", "config_land_ice_flux_formulation": "ISOMIP",
      "config_eos_type": "jm", "config_land_ice_flux_boundaryLayerThickness": 150.0,
      "config_land_ice_flux_attenuation_coefficient": 25.0}, True),
    ("coupled_jenkins_ke_vertex_apvm", 20262203, 52, 14, True,
     {**_LAND_ICE, "config_land_ice_flux_mode": "coupled", "config_include_KE_vertex": True,
      "config_apvm_scale_factor": 0.5, "config_land_ice_flux_boundaryLayerNeighborWeight": 1.25,
      "config_land_ice_flux_topDragCoeff": 5.0e-3, "config_land_ice_flux_rms_tidal_velocity": 0.1,
      "config_land_ice_flux_jenkins_heat_transfer_coefficient": 0.02,
      "config_land_ice_flux_jenkins_salt_transfer_coefficient": 6.0e-4,
      "config_use_frazil_ice_formation": True}, False),
)


def make_case(name, seed, n_cells, n_levels, nested, overrides, swapped, precision="r4") -> dict:
    case = _diag.make_case(name, seed, n_cells, n_levels, nested, overrides, precision=precision)
    rs = np.random.RandomState(seed + 11)
    real = PRECISIONS[precision]
    nc, nl, me = case["dims"]["nCells"], case["dims"]["nVertLevels"], case["dims"]["maxEdges"]
    cells_on_cell = rs.randint(1, nc + 1, (nc + 1, me)).astype(np.int32)
    cells_on_cell[rs.random_sample((nc + 1, me)) < 0.1] = nc + 1
    cell_mask = (rs.random_sample((nc + 1, nl)) < 0.8).astype(np.int32)
    cell_mask[nc] = 0
    it, isl = (2, 1) if swapped else (1, 2)
    values = {"cellsOnCell": cells_on_cell, "cellMask": cell_mask,
              "index_landIceBoundaryLayerTemperature": it, "index_landIceBoundaryLayerSalinity": isl,
              "index_landIceHeatTransferVelocity": it, "index_landIceSaltTransferVelocity": isl}
    pools = case["pools"]
    for pool, field, kind, fshape in _EXTRA:
        if kind == DIM or field in values:
            value = values[field]
        elif kind == SCRATCH:
            value = None
        else:
            value = (-1.0e3 + 2.0e3 * rs.random_sample(_stream.shape(case, fshape))).astype(real)
        pools.setdefault(pool, {})[field] = value
    # the kinetic energy the loops do not reach feeds the top drag's square root
    ke = pools["diagnostics"]["kineticEnergyCell"]
    pools["diagnostics"]["kineticEnergyCell"] = rs.random_sample(ke.shape).astype(real)
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
    """Run woof.hex.mpas_ocean.diagnostics_v842 on ``case`` as run_diagnostics.F90
    drives the Fortran; the same ``pool/name`` outputs."""
    import cupy as cp
    from woof.hex.mpas_ocean import diagnostics_v842 as dg

    dev = _stream.on_device(FIELDS, case)
    dg.diagnostic_solve(dt=case["dt"], mesh=dev["mesh"], state=dev["state"], tracers=dev["tracers"],
                        diagnostics=dev["diagnostics"], forcing=dev["forcing"], configs=case["config"])
    return {f"{pool}/{name}": cp.asnumpy(dev[pool][name]) for pool, name in ARRAYS
            if dev[pool][name] is not None}


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases(precision):
        got = run_gpu(case)
        want = reference[case["name"]]
        report[case["name"]] = {name: word_mismatches(got[name], want[name]) for name in want}
    return report
