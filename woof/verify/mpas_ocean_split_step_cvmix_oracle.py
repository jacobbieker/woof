"""Cases for one MPAS-Ocean split-explicit step with CVMix on (RKIND 8).

The driver is the split step oracle's (``tools/mpas_ocean_oracle/run_split_step.F90``,
:mod:`woof.verify.mpas_ocean_split_step_oracle`) linked with CVMix v0.84-beta
and ``mpas_ocean_vmix_cvmix.F`` verbatim in place of ``stubs_vmix.F90``
(``run_split_step_cvmix_r8``; build.sh). With ``config_use_cvmix_kpp`` each
iteration's tracer tendency computes KPP's input fields
(``ocn_compute_KPP_input_fields`` at time level 2, the surface fluxes on time
level 1's tracers) and adds the non-local flux the input's
``vertNonLocalFlux`` (the previous step's) carries, or leaves it to the
implicit solve (``config_cvmix_kpp_nonlocal_with_implicit_mix``); the implicit
vertical mixing then builds CVMix's coefficients (and this step's
``vertNonLocalFlux``). CVMix is real(8) throughout: RKIND 8 only.

The field table is the split step's with ``vertNonLocalFlux`` at its Registry
shape (one member, ``vertNonLocalFluxTemp``) and the fields CVMix and KPP's
input fields read or write added (:data:`woof.verify.mpas_ocean_cvmix_oracle.FIELDS`,
:data:`woof.verify.mpas_ocean_tend_tracer_oracle.KPP_FIELDS`).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_cvmix_oracle as _cv
from woof.verify import mpas_ocean_split_step_oracle as _ss
from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_ocean_tend_oracle as _tend
from woof.verify import mpas_ocean_tend_tracer_oracle as _tt

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "mpas_ocean_split_step_cvmix")
#: RKIND 8 only: CVMix does not compile at RKIND 4
PRECISIONS = ("r8",)
CASES_PER_PRECISION = True
REAL, INT, DIM, SCRATCH = _stream.REAL, _stream.INT, _stream.DIM, _stream.SCRATCH
#: vertNonLocalFlux is a var_array of one member (Registry.xml)
_RESHAPED = {("diagnostics", "vertNonLocalFlux"): (1, "LP1", "C1")}


def field_table(base: tuple) -> tuple:
    """``base`` with ``vertNonLocalFlux`` at its Registry shape and CVMix's and
    KPP's input fields' rows it lacks added."""
    rows = [(p, n, k, _RESHAPED.get((p, n), f)) for p, n, k, f in base]
    seen = {(p, n) for p, n, _, _ in rows}
    for row in (*_cv.FIELDS, *_tt.KPP_FIELDS):
        if (row[0], row[1]) not in seen:
            seen.add((row[0], row[1]))
            rows.append(row)
    return tuple(rows)


FIELDS = field_table(_ss.FIELDS)
#: CVMix's Registry defaults (CVMix on), KPP's input fields' attenuation defaults
CVMIX_ON = {**_cv.CVMIX_CONFIG, "config_flux_attenuation_coefficient": 0.001,
            "config_flux_attenuation_coefficient_runoff": 0.001}
KPP = {**CVMIX_ON, **_cv.KPP}
#: (name, seed, nCells, nVertLevels, config overrides; ``_ice`` puts sea ice over half the cells)
CASES = (
    ("kpp_split_defaults", 20261801, 40, 12, {**KPP}),
    ("kpp_split_jm_bulk_implicit_budgets", 20261802, 44, 14, {
        **KPP, "config_eos_type": "jm", "config_pressure_gradient_type": "Jacobian_from_TS",
        "config_use_activeTracers_surface_bulk_forcing": True, "config_use_bulk_thickness_flux": True,
        "config_sw_absorption_type": "jerlov", "config_compute_active_tracer_budgets": True,
        "config_cvmix_kpp_nonlocal_with_implicit_mix": True, "config_n_ts_iter": 3}),
    ("kpp_split_match_both_ice_attenuated", 20261803, 40, 12, {
        **KPP, "config_cvmix_kpp_matching": "MatchBoth", "config_cvmix_kpp_interpolationOMLType": "cubic",
        "config_flux_attenuation_coefficient": 10.0, "config_flux_attenuation_coefficient_runoff": 4.0,
        "configure_cvmix_kpp_minimum_OBL_under_sea_ice": 20.0, "config_compute_active_tracer_budgets": True,
        "_ice": True}),
    ("cvmix_unsplit_no_kpp", 20261804, 32, 10, {
        **CVMIX_ON, "config_use_cvmix_background": True, "config_use_cvmix_shear": True,
        "config_use_cvmix_convection": True, "config_time_integrator": "unsplit_explicit",
        "config_vel_correction": False, "config_n_bcl_iter_mid": 1}),
    # the bulk wind stress, which alone gives the step's KPP a friction velocity (ocn_tend_vel zeroes the
    # surface stress, then adds it); at RKIND 8 its cos and sin are glibc's, behind WOOF_GLIBC239_LIBM64=1
    ("kpp_split_wind_match_both", 20261805, 40, 12, {
        **KPP, "config_use_bulk_wind_stress": True, "config_cvmix_kpp_matching": "MatchBoth",
        "config_compute_active_tracer_budgets": True}),
)


def add_cvmix_inputs(case: dict, rs: np.random.RandomState, overrides: dict) -> None:
    """CVMix's and KPP's input rows in ``case``'s pools: the Registry's shapes,
    an ocean's surface stress and thickness flux (without the bulk thickness
    flux; the step's tendencies zero both before the bulk forcing, when on,
    sets them), sea ice (``_ice``), the previous step's boundary layer and
    non-local flux; noise for what the step writes before it reads."""
    real = _tend.PRECISIONS[case["precision"]]
    pools = case["pools"]
    nc, ne, nl = case["dims"]["nCells"], case["dims"]["nEdges"], case["dims"]["nVertLevels"]
    mesh = pools["mesh"]

    def uniform(lo, hi, shp):
        return lo + (hi - lo) * rs.random_sample(shp)

    mlc = mesh["maxLevelCell"]
    lat = mesh["latCell"].astype(np.float64)
    cell_mask = (np.arange(nl)[None, :] < mlc[:, None]).astype(np.int32)
    cell_mask[nc] = 0
    ice = np.zeros(nc + 1)
    if overrides.get("_ice"):
        ice = np.where(rs.random_sample(nc + 1) < 0.5, uniform(0.0, 1.0, nc + 1), 0.0)
    values = {
        "nVertLevelsP1": nl + 1, "index_vertNonLocalFluxTemp": 1, "cellMask": cell_mask,
        # sin(lat)'s cubic, not the host libm: the same input bytes everywhere
        "fCell": 2.0 * 7.292e-5 * (lat - lat * lat * lat / 6.0),
        "vertNonLocalFlux": uniform(0.0, 1.0, (nc + 1, nl + 1, 1)) * (np.arange(nl + 1)[None, :, None] > 0),
        "boundaryLayerDepth": uniform(5.0, 60.0, nc + 1), "boundaryLayerDepthSmooth": uniform(5.0, 60.0, nc + 1),
        "indexBoundaryLayerDepth": uniform(1.0, float(nl), nc + 1), "iceFraction": ice,
        "surfaceStress": uniform(-0.3, 0.3, ne + 1),
    }
    if not case["config"].get("config_use_bulk_thickness_flux", False):
        values["surfaceThicknessFlux"] = uniform(-2.0e-6, 2.0e-6, nc + 1)
    present = {(p, n) for p, n, _, _ in _ss.FIELDS}
    for pool, field, kind, fshape in FIELDS:
        if field in values:
            value = values[field]
        elif (pool, field) in present and (pool, field) not in _RESHAPED:
            continue
        elif kind in (REAL, INT):
            value = uniform(-1.0, 1.0, _stream.shape(case, fshape))
        else:
            value = None
        if kind == REAL and value is not None:
            value = np.asarray(value).astype(real)
        pools.setdefault(pool, {})[field] = value


def make_case(name: str, seed: int, n_cells: int, n_levels: int, overrides: dict, precision: str = "r8") -> dict:
    """The split step oracle's case with CVMix's inputs added (from their own seed)."""
    case = _ss.make_case(name, seed, n_cells, n_levels, overrides, precision)
    add_cvmix_inputs(case, np.random.RandomState(seed + 7), overrides)
    return case


def cases(precision: str = "r8") -> list[dict]:
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
    """Run woof.hex.mpas_ocean.split_step_v842 on ``case`` as run_split_step.F90
    drives the Fortran: the starting diagnostics, then the step."""
    import cupy as cp
    from woof.hex.mpas_ocean import diagnostics_v842 as dg
    from woof.hex.mpas_ocean import split_step_v842 as ss

    dev = _stream.on_device(FIELDS, case)
    config = case["config"]
    st, tr = dev["state"], dev["tracers"]
    dg.diagnostic_solve(dt=case["dt"], mesh=dev["mesh"],
                        state={n: st[n] for n in ("layerThickness", "normalVelocity", "ssh")},
                        tracers={n: tr[n] for n in ("activeTracers", "index_temperature", "index_salinity")},
                        diagnostics=dev["diagnostics"], forcing=dev["forcing"], configs=config)
    ss.split_step(dt=case["dt"], n_btr_subcycles=int(config["nBtrSubcycles"]), pools=dev, configs=config)
    return {f"{pool}/{name}": cp.asnumpy(dev[pool][name]) for pool, name, kind, _ in FIELDS
            if kind in (REAL, INT) and dev[pool][name] is not None}


def runs(case: dict) -> bool:
    """Whether this checkout runs the case: the bulk wind stress (and the sphere)
    at RKIND 8 need the glibc 2.39 opt-in (``WOOF_GLIBC239_LIBM64=1``)."""
    from woof.hex.mpas_ice import rkind
    wind = case["config"].get("config_use_bulk_wind_stress", False)
    return _ss.runs(case) and not (wind and case["precision"] == "r8" and not rkind.trig64())


def measure(precision: str, reference: dict | None = None) -> dict[str, dict[str, int]]:
    """Word mismatches per case and field, over the cases this checkout runs."""
    reference = load_fixture(precision) if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases(precision):
        if not runs(case):
            continue
        got = run_gpu(case)
        want = reference[case["name"]]
        report[case["name"]] = {name: word_mismatches(got[name], want[name]) for name in want}
    return report
