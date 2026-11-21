"""Cases and the stream layout for MPAS-Seaice's coupling preparation.

The reference is ``tools/mpas_seaice_oracle/run_coupling_prep.F90``.
``seaice_column_coupling_prep`` and ``seaice_column_scale_fluxes`` are cut
verbatim out of v8.4.2's ``mpas_seaice_column.F``. They use the shared stream
of :mod:`woof.verify.mpas_ocean_stream`, with ``NCAT`` as the category
dimension.

The cases cover the following branches:

- cells with ice and cells without (the open-water longwave, the reference
  values from the air);
- the sun up and down (``solarZenithAngleCosine`` above and below
  ``seaicePuny``, exactly zero, exactly ``seaicePuny`` and between them included);
- the pond freshwater feedback on and off;
- one category and five;
- halo cells beyond ``nCellsSolve``, which must be left alone;
- the column biogeochemistry with the brine height alone (no biogeochemical
  tracer): coupling_prep zeroes the owned cells' ocean biogeochemistry
  fluxes and scale_fluxes scales them over the aggregate pool's
  ``nCellsSolve``, which these cases set past and short of the tracers
  pool's, so the scaled-but-not-zeroed and the zeroed-but-not-scaled cells
  both show.

Without the biogeochemistry its fields are absent; the cases with it use
:data:`BGC_FIELDS` (the Registry's extents).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_stream as _stream
from woof.verify.mpas_ocean_tend_oracle import PRECISIONS

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_seaice_coupling_prep")
CASES_PER_PRECISION = True
REAL, DIM = _stream.REAL, _stream.DIM
_BGC_1D = ("oceanNitrateFlux", "oceanSilicateFlux", "oceanAmmoniumFlux", "oceanDMSFlux", "oceanDMSPpFlux",
           "oceanDMSPdFlux", "oceanHumicsFlux", "oceanDustIronFlux")
_BGC_2D = ("oceanBioFluxes", "oceanAlgaeFlux", "oceanDOCFlux", "oceanDICFlux", "oceanDONFlux",
           "oceanParticulateIronFlux", "oceanDissolvedIronFlux")
_BGC_DIMS = {"nZBGCTracers": 0, "maxAlgaeType": 3, "maxDOCType": 3, "maxDICType": 1, "maxDONType": 1,
             "maxAerosolType": 6, "maxIronType": 2, "maxBCType": 2, "maxDustType": 4}
_CATEGORY = ("albedoVisibleDirectCategory", "albedoVisibleDiffuseCategory", "albedoIRDirectCategory",
             "albedoIRDiffuseCategory", "bareIceAlbedoCategory", "snowAlbedoCategory", "pondAlbedoCategory",
             "effectivePondAreaCategory")
FIELDS = (
    *(("mesh", name, DIM, ()) for name in _BGC_DIMS),
    ("tracers", "nCellsSolve", DIM, ()), ("tracers", "nCategories", DIM, ()),
    ("tracers", "iceAreaCategory", REAL, (1, "NCAT", "C1")),
    ("tracers_aggregate", "nCellsSolve", DIM, ()), ("tracers_aggregate", "iceAreaCell", REAL, ("C1",)),
    ("ocean_coupling", "freezingMeltingPotential", REAL, ("C1",)),
    ("ocean_coupling", "seaFreezingTemperature", REAL, ("C1",)),
    ("diagnostics", "freezingMeltingPotentialInitial", REAL, ("C1",)),
    *(("shortwave", name, REAL, ("NCAT", "C1") if name in _CATEGORY else ("C1",)) for name in (
        "albedoVisibleDirectCell", "albedoVisibleDiffuseCell", "albedoIRDirectCell", "albedoIRDiffuseCell",
        *_CATEGORY[:4], "albedoVisibleDirectArea", "albedoVisibleDiffuseArea", "albedoIRDirectArea",
        "albedoIRDiffuseArea", "solarZenithAngleCosine", "bareIceAlbedoCell", "snowAlbedoCell", "pondAlbedoCell",
        *_CATEGORY[4:7], "effectivePondAreaCell", _CATEGORY[7], "shortwaveScalingFactor",
        "absorbedShortwaveFlux")),
    *(("atmos_coupling", name, REAL, ("C1",)) for name in (
        "shortwaveVisibleDirectDown", "shortwaveVisibleDiffuseDown", "shortwaveIRDirectDown",
        "shortwaveIRDiffuseDown", "atmosReferenceHumidity2m", "atmosReferenceTemperature2m", "airTemperature",
        "airSpecificHumidity")),
    ("ponds", "pondFreshWaterFlux", REAL, ("C1",)),
    *(("ocean_fluxes", name, REAL, ("C1",)) for name in (
        "oceanFreshWaterFlux", "oceanSaltFlux", "oceanHeatFlux", "oceanShortwaveFlux", "oceanFreshWaterFluxArea",
        "oceanSaltFluxArea", "oceanHeatFluxArea", "oceanShortwaveFluxArea")),
    *(("biogeochemistry", name, REAL, ("C1",)) for name in _BGC_1D),
    *(("biogeochemistry", name, REAL, (1, "C1")) for name in _BGC_2D),
    ("velocity_solver", "airStressCellU", REAL, ("C1",)), ("velocity_solver", "airStressCellV", REAL, ("C1",)),
    *(("atmos_fluxes", name, REAL, ("C1",)) for name in (
        "sensibleHeatFlux", "latentHeatFlux", "evaporativeWaterFlux", "longwaveUp")),
)
#: the Registry's extents of the biogeochemistry's fields (nZBGCTracers is
#: oceanBioFluxesAll's, which coupling_prep reads 29 words of)
_BGC_EXTENTS = {"nZBGCTracers": 29, "maxAlgaeType": 3, "maxDOCType": 3, "maxDICType": 1, "maxDONType": 1,
                "maxAerosolType": 6, "maxIronType": 2, "maxBCType": 2, "maxDustType": 4}
_BGC_2D_EXTENT = {"oceanBioFluxes": "nZBGCTracers", "oceanAlgaeFlux": "maxAlgaeType", "oceanDOCFlux": "maxDOCType",
                  "oceanDICFlux": "maxDICType", "oceanDONFlux": "maxDONType",
                  "oceanParticulateIronFlux": "maxIronType", "oceanDissolvedIronFlux": "maxIronType"}
#: the stream with the column biogeochemistry on: FIELDS with each two-dimensional
#: biogeochemistry field at its Registry extent
BGC_FIELDS = tuple((pool, name, kind, (_BGC_EXTENTS[_BGC_2D_EXTENT[name]], "C1"))
                   if pool == "biogeochemistry" and name in _BGC_2D else (pool, name, kind, fshape)
                   for pool, name, kind, fshape in FIELDS)
ARRAYS = tuple((pool, name) for pool, name, kind, _ in FIELDS if kind == REAL)
BASE_CONFIG = {"config_use_ocean_mixed_layer": False, "config_include_pond_freshwater_feedback": False,
               "config_use_column_biogeochemistry": False}
_BGC = {"config_use_column_biogeochemistry": True, "config_use_brine": True}
#: (name, seed, nCells, nCellsSolve, nCategories, config_dt, config overrides[, the
#: aggregate pool's nCellsSolve, by default the tracers pool's])
CASES = (
    ("five_categories", 20262401, 40, 40, 5, 3600.0, {}),
    ("pond_feedback_halo", 20262402, 48, 33, 5, 1800.0, {"config_include_pond_freshwater_feedback": True}),
    ("one_category_low_sun", 20262403, 30, 30, 1, 600.0, {"config_include_pond_freshwater_feedback": True}),
    ("bgc_brine_aggregate_past_solve", 20262404, 48, 33, 5, 1800.0, _BGC, 40),
    ("bgc_brine_aggregate_short_of_solve", 20262405, 30, 30, 3, 600.0,
     {**_BGC, "config_include_pond_freshwater_feedback": True}, 24),
)


def fields_of(case: dict) -> tuple:
    """The case's stream layout: :data:`BGC_FIELDS` with the biogeochemistry, else :data:`FIELDS`."""
    return BGC_FIELDS if case["config"]["config_use_column_biogeochemistry"] else FIELDS


def make_case(name, seed, n_cells, solved, n_categories, dt, overrides, aggregate=None, precision="r4") -> dict:
    rs = np.random.RandomState(seed)
    real = PRECISIONS[precision]
    nc = n_cells
    config = dict(BASE_CONFIG, config_dt=dt, **overrides)
    dims = {"nCells": nc, "nEdges": 0, "nVertLevels": 1, "nCategories": n_categories}
    case = {"name": name, "precision": precision, "dims": dims, "config": config, "dt": dt}

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    area = uniform(0.0, 1.0 / n_categories, (nc + 1, n_categories, 1))
    area[rs.random_sample(nc + 1) < 0.25] = 0.0  # open water
    coszen = uniform(-0.4, 1.0, nc + 1)
    if name == "one_category_low_sun":
        coszen = uniform(-1.0, 0.1, nc + 1)
    coszen[rs.random_sample(nc + 1) < 0.1] = 0.0
    # the sun test's edge: exactly seaicePuny and between zero and it
    edge = rs.random_sample(nc + 1)
    coszen[edge < 0.05] = 1.0e-11
    coszen[(edge >= 0.05) & (edge < 0.1)] = 0.5e-11
    bgc = bool(config["config_use_column_biogeochemistry"])
    values = dict(_BGC_EXTENTS if bgc else _BGC_DIMS)
    values.update({
        "nCellsSolve": solved, "nCategories": n_categories, "iceAreaCategory": area,
        "iceAreaCell": area.sum(axis=(1, 2)), "solarZenithAngleCosine": coszen,
        "freezingMeltingPotential": uniform(-200.0, 200.0, nc + 1),
        "seaFreezingTemperature": uniform(-2.0, -1.6, nc + 1),
        "pondFreshWaterFlux": uniform(0.0, 2.0e-3, nc + 1),
        "oceanFreshWaterFlux": uniform(-1.0e-4, 1.0e-4, nc + 1), "oceanSaltFlux": uniform(-1.0e-5, 1.0e-5, nc + 1),
        "oceanHeatFlux": uniform(-100.0, 20.0, nc + 1), "oceanShortwaveFlux": uniform(0.0, 300.0, nc + 1),
        "airTemperature": uniform(240.0, 275.0, nc + 1), "airSpecificHumidity": uniform(1.0e-4, 5.0e-3, nc + 1),
        "atmosReferenceTemperature2m": uniform(240.0, 275.0, nc + 1),
        "atmosReferenceHumidity2m": uniform(1.0e-4, 5.0e-3, nc + 1),
        "airStressCellU": uniform(-0.3, 0.3, nc + 1), "airStressCellV": uniform(-0.3, 0.3, nc + 1),
        "sensibleHeatFlux": uniform(-50.0, 50.0, nc + 1), "latentHeatFlux": uniform(-50.0, 10.0, nc + 1),
        "evaporativeWaterFlux": uniform(-1.0e-5, 1.0e-5, nc + 1), "longwaveUp": uniform(-320.0, -200.0, nc + 1),
        "absorbedShortwaveFlux": uniform(0.0, 300.0, nc + 1),
    })
    for name_ in ("shortwaveVisibleDirectDown", "shortwaveVisibleDiffuseDown", "shortwaveIRDirectDown",
                  "shortwaveIRDiffuseDown"):
        values[name_] = uniform(0.0, 400.0, nc + 1)
    for name_ in _CATEGORY:
        values[name_] = uniform(0.0, 1.0, (nc + 1, n_categories))
    pools: dict[str, dict] = {}
    for pool, field, kind, fshape in fields_of(case):
        if kind == DIM:
            value = values[field]
            if (pool, field) == ("tracers_aggregate", "nCellsSolve") and aggregate is not None:
                value = aggregate
        elif pool == "biogeochemistry" and not bgc:
            value = None
        elif field in values:
            value = np.asarray(values[field]).astype(real)
        else:
            # a field the routines write before they read: noise a word left alone would show
            value = uniform(-1.0, 1.0, _stream.shape(case, fshape)).astype(real)
        pools.setdefault(pool, {})[field] = value
    case["pools"] = pools
    return case


def cases(precision: str = "r4") -> list[dict]:
    return [make_case(*row, precision=precision) for row in CASES]


def input_bytes(case: dict) -> bytes:
    return _stream.input_bytes(fields_of(case), case)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    return _stream.read_outputs(fields_of(case), raw, case)


word_mismatches = _stream.word_mismatches


def load_fixture(precision: str) -> dict[str, dict[str, np.ndarray]]:
    with np.load(FIXTURE_DIR / f"oracle-{precision}.npz") as data:
        out: dict[str, dict[str, np.ndarray]] = {}
        for key in data.files:
            case, field = key.split("/", 1)
            out.setdefault(case, {})[field] = data[key]
    return out


def run_gpu(case: dict) -> dict[str, np.ndarray]:
    """Run woof.hex.mpas_seaice.coupling_prep_v842 on ``case`` as run_coupling_prep.F90 drives the Fortran."""
    import cupy as cp
    from woof.hex.mpas_seaice import coupling_prep_v842 as cpl

    dev = _stream.on_device(fields_of(case), case)
    cpl.coupling_prep(pools=dev, n_cells_solve=int(dev["tracers"]["nCellsSolve"]),
                      n_cells_solve_aggregate=int(dev["tracers_aggregate"]["nCellsSolve"]),
                      dt=case["config"]["config_dt"], configs=case["config"])
    return {f"{pool}/{name}": cp.asnumpy(dev[pool][name]) for pool, name in ARRAYS if dev[pool][name] is not None}


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases(precision):
        got = run_gpu(case)
        want = reference[case["name"]]
        report[case["name"]] = {name: word_mismatches(got[name], want[name]) for name in want}
    return report
