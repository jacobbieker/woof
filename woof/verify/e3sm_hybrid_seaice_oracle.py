"""The MPAS v8.4.2 sea ice under E3SM v3.2.0's coupler: cases, streams, GPU run.

"E3SM v3.2.0 coupler with MPAS v8.4.2 components, not an E3SM
configuration" (:mod:`woof.hex.coupled.hybrid_seaice`).
``tools/e3sm_coupler_oracle/run_hybrid_seaice.F90`` drives v8.4.2's time
step with the column package and the velocity solver (every module verbatim,
at RKIND 8, as ``run_column.F90``'s ``seaice_step`` stage builds it) in the
order E3SM's ``ice_init_mct`` and ``ice_run_mct`` call it, with E3SM's
``ice_import_mct`` and ``ice_export_mct`` cut verbatim and compiled against
v8.4.2's modules.

A case is one of the column oracle's ``seaice_step`` starts
(:data:`~woof.verify.mpas_seaice_column_oracle.COUPLED_CASES`) with what the
cap looks up that v8.4.2 lacks added (``frazilMassAdjust`` zero,
``nFrequencies`` 0, the cap's four configs off, the mesh's
``sphere_radius`` 1: the torus's cells sit on the unit sphere), E3SM's x2i
and i2x lists, and per coupling window an x2i drawn round the start's own
forcing (so the column runs as the step oracle's cases do) and the
coupler's ``nextsw_cday``. The oracle writes ``i2x`` after the init's export
and after each window's, the rotation's words, then every field.
"""
from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from woof.verify import mpas_ocean_stream as _stream
from woof.verify import mpas_seaice_column_oracle as _col
from woof.verify.e3sm_seaice_coupling_oracle import AEROSOL_X2I, I2X_BASE, X2I_BASE

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "e3sm_hybrid_seaice"
REAL, DIM = _stream.REAL, _stream.DIM
X2I_FIELDS = tuple(X2I_BASE[:9] + AEROSOL_X2I[:7] + X2I_BASE[9:] + AEROSOL_X2I[7:])
I2X_FIELDS = tuple(I2X_BASE)
TFFRESH = 273.15

#: (name, column case, steps per window, windows, config overrides)
CASES = (
    # BL99 with the dEdd shortwave (its init on the first step), one step a window
    ("hybrid_bl99_dedd", "seaice_bl99_dedd_rothrock", 1, 3, {}),
    # two steps a window (the step counter restarts; the fluxes sum over the window), gregorian
    ("hybrid_bl99_two_steps_gregorian", "seaice_bl99_dedd_two_subcycles_exponential_gregorian", 2, 2, {}),
    # mushy thermodynamics: E3SM's frazil mass on the mushy enthalpy (Icepack's functions as colpkg's)
    ("hybrid_mushy_level_ponds", "seaice_mushy_level_ponds_no_vertical_thermodynamics", 2, 2, {}),
    # ccsm3 shortwave, Hibler strength, no surface tilt, an unrotated grid
    ("hybrid_bl99_ccsm3_unrotated", "seaice_bl99_hibler_strength_ccsm3_winter", 1, 2,
     {"config_rotate_cartesian_grid": False}),
)

#: x2i field -> (pool, array, offset): the start's own forcing the window's draws centre on
_FROM_POOLS = {
    "So_t": ("ocean_coupling", "seaSurfaceTemperature", TFFRESH), "So_s": ("ocean_coupling", "seaSurfaceSalinity", 0),
    "So_u": ("ocean_coupling", "uOceanVelocity", 0), "So_v": ("ocean_coupling", "vOceanVelocity", 0),
    "So_dhdx": ("ocean_coupling", "seaSurfaceTiltU", 0), "So_dhdy": ("ocean_coupling", "seaSurfaceTiltV", 0),
    "Sa_z": ("atmos_coupling", "airLevelHeight", 0), "Sa_ptem": ("atmos_coupling", "airPotentialTemperature", 0),
    "Sa_tbot": ("atmos_coupling", "airTemperature", 0), "Sa_shum": ("atmos_coupling", "airSpecificHumidity", 0),
    "Sa_dens": ("atmos_coupling", "airDensity", 0), "Sa_u": ("atmos_coupling", "uAirVelocity", 0),
    "Sa_v": ("atmos_coupling", "vAirVelocity", 0), "Faxa_lwdn": ("atmos_coupling", "longwaveDown", 0),
    "Faxa_rain": ("atmos_coupling", "rainfallRate", 0), "Faxa_snow": ("atmos_coupling", "snowfallRate", 0),
    "Faxa_swvdr": ("atmos_coupling", "shortwaveVisibleDirectDown", 0),
    "Faxa_swvdf": ("atmos_coupling", "shortwaveVisibleDiffuseDown", 0),
    "Faxa_swndr": ("atmos_coupling", "shortwaveIRDirectDown", 0),
    "Faxa_swndf": ("atmos_coupling", "shortwaveIRDiffuseDown", 0),
}
#: x2i fields drawn outright: (low, high)
_DRAWN = {"Fioo_q": (-40.0, 40.0), "Fioo_frazil": (0.0, 2.0e-5)}

#: the configs E3SM's cap reads that v8.4.2 lacks, as the hybrid accepts them
CAP_CONFIGS = {"config_column_physics_type": "column_package", "config_use_data_icebergs": False,
               "config_use_floe_size_distribution": False, "config_use_column_waves": False}
#: the fields the cap looks up that v8.4.2 lacks
CAP_FIELDS = (("ocean_coupling", "frazilMassAdjust", REAL, ("C1",)), ("mesh", "nFrequencies", DIM, ()))


def _x2i(rs, pools: dict, lsize: int, windows: int) -> np.ndarray:
    out = np.zeros((windows, lsize, len(X2I_FIELDS)))
    for k, name in enumerate(X2I_FIELDS):
        if name in _FROM_POOLS:
            pool, array, offset = _FROM_POOLS[name]
            base = np.asarray(pools[pool][array], np.float64)[:lsize] + offset
            scale = np.maximum(np.abs(base), 1.0e-12) * 0.05
            drawn = base + (rs.random_sample((windows, lsize)) - 0.5) * scale
            out[:, :, k] = np.where(base >= 0.0, np.abs(drawn), drawn)  # a flux or density stays >= 0
        elif name in _DRAWN:
            lo, hi = _DRAWN[name]
            out[:, :, k] = rs.random_sample((windows, lsize)) * (hi - lo) + lo
    return out


def make_case(name: str, column_case: str, steps_per_window: int, windows: int, overrides: dict) -> dict:
    row = next(r for r in _col.COUPLED_CASES if r[0] == column_case)
    case = _col.make_case(*row)
    case["name"] = name
    c = case["config"]
    c.update(CAP_CONFIGS)
    c.update(overrides)
    c["mesh:sphere_radius"] = 1.0
    c.setdefault("config_ocean_surface_type", "free")
    pools = case["pools"]
    nc = int(case["dims"]["nCells"])
    pools["ocean_coupling"]["frazilMassAdjust"] = np.zeros(nc + 1)
    pools["mesh"]["nFrequencies"] = 0
    case["fields"] = list(case["fields"]) + [f for f in CAP_FIELDS if (f[0], f[1]) not in
                                             {(p, n) for p, n, _, _ in case["fields"]}]
    rs = np.random.RandomState(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
    lsize = int(case["dims"]["nCellsSolve"])
    dt = int(float(c["config_dt"]))
    nextsw = np.concatenate([[-1.0], 150.0 + rs.random_sample(windows)])
    case["hybrid"] = {"steps_per_window": steps_per_window, "windows": windows, "ice_cpl_dt": steps_per_window * dt,
                      "x2i": _x2i(rs, pools, lsize, windows), "nextsw": nextsw, "lsize": lsize}
    return case


def cases() -> list[dict]:
    return [make_case(*row) for row in CASES]


def start_bytes(case: dict) -> bytes:
    return _stream.input_bytes(case["fields"], case)


def coupling_bytes(case: dict) -> bytes:
    h = case["hybrid"]
    x2i_list, i2x_list = ":".join(X2I_FIELDS).encode(), ":".join(I2X_FIELDS).encode()
    parts = [struct.pack("<5i", h["ice_cpl_dt"], h["windows"], h["lsize"], len(X2I_FIELDS), len(I2X_FIELDS)),
             struct.pack("<i", len(x2i_list)), x2i_list, struct.pack("<i", len(i2x_list)), i2x_list,
             np.ascontiguousarray(h["nextsw"], "<f8").tobytes(), np.ascontiguousarray(h["x2i"], "<f8").tobytes()]
    return b"".join(parts)


def input_digest(case: dict) -> str:
    return hashlib.sha256(start_bytes(case) + coupling_bytes(case)).hexdigest()


def read_outputs(raw: bytes, pools_raw: bytes, case: dict) -> dict[str, np.ndarray]:
    h = case["hybrid"]
    n_i2x = (h["windows"] + 1) * h["lsize"] * len(I2X_FIELDS)
    nc = int(case["dims"]["nCells"])
    if len(raw) != 8 * (n_i2x + 8 * nc):
        raise ValueError(f"oracle output: {len(raw)} bytes, not {8 * (n_i2x + 8 * nc)}")
    words = np.frombuffer(raw, "<f8").astype(np.float64)
    i2x = words[:n_i2x].reshape(h["windows"] + 1, h["lsize"], len(I2X_FIELDS))
    out = {f"i2x/{w}": i2x[w] for w in range(h["windows"] + 1)}
    out["oracle/trig"] = words[n_i2x:].reshape(nc, 8)
    out.update(_stream.read_outputs(case["fields"], pools_raw, case))
    return out


word_mismatches = _stream.word_mismatches


def parse_log(text: str) -> dict:
    return _col.parse_log(text)


def run_binary(build_dir: Path, case: dict) -> tuple[int, dict | None, dict | str]:
    """``run_hybrid_seaice`` on ``case``: (status, outputs, parsed log or stderr)."""
    binary = Path(build_dir) / "run_hybrid_seaice"
    with tempfile.TemporaryDirectory() as tmp:
        start, cpl, out, log = (Path(tmp) / n for n in ("start.bin", "coupling.bin", "out.bin", "log.txt"))
        start.write_bytes(start_bytes(case))
        cpl.write_bytes(coupling_bytes(case))
        done = subprocess.run([str(binary), str(start), str(cpl), str(out), str(log)], capture_output=True, text=True)
        pools = Path(str(out) + ".pools")
        if done.returncode != 0 or not out.exists() or not pools.exists():
            return done.returncode or -1, None, done.stderr + done.stdout
        parsed = parse_log(log.read_text(encoding="utf-8"))
        return 0, read_outputs(out.read_bytes(), pools.read_bytes(), case), parsed


def load_fixture() -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    with np.load(FIXTURE_DIR / "oracle-r8.npz") as z:
        for key in z.files:
            case, _, name = key.partition("/")
            out.setdefault(case, {})[name] = z[key]
    return out


def load_logs() -> dict:
    return json.loads((FIXTURE_DIR / "logs.json").read_text(encoding="utf-8"))


def expected_keys(case: dict) -> list[str]:
    """The fields compared: every present real and integer one but the
    column oracle's stand-ins (not v8.4.2's arrays)."""
    skip = _col.stand_ins(case["config"])
    return [f"{p}/{n}" for p, n, kind, _ in case["fields"]
            if kind in (REAL, _stream.INT) and case["pools"][p][n] is not None and (p, n) not in skip]


def run_gpu(case: dict, *, trig=None, mutate=None) -> dict[str, np.ndarray]:
    """:class:`~woof.hex.coupled.hybrid_seaice.HybridSeaIce` on ``case`` as the
    oracle drives the Fortran: every i2x and every array. ``trig``
    ``(nCells, 8)`` is the rotation's words (the oracle's, from the fixture,
    when given; else computed). ``mutate(ice)`` runs after construction (tests)."""
    import cupy as cp

    from woof.hex.coupled.hybrid_seaice import HybridSeaIce
    fields = case["fields"]
    dev = _stream.on_device(fields, case)
    c = case["config"]
    dev["diagnostics"]["xtime"] = c["diagnostics:xtime"]
    dev["diagnostics"]["simulationStartTime"] = c["diagnostics:simulationStartTime"]
    given = {(pool, name): dev[pool][name] for pool, name, kind, _ in fields
             if kind in (REAL, _stream.INT) and dev[pool][name] is not None}
    h = case["hybrid"]
    ice = HybridSeaIce(pools=dev, configs=c, x2i_fields=X2I_FIELDS, i2x_fields=I2X_FIELDS,
                       ice_cpl_dt=h["ice_cpl_dt"], nextsw_cday=float(h["nextsw"][0]),
                       trig=None if trig is None else cp.asarray(np.ascontiguousarray(trig, np.float64)),
                       sphere_radius=float(c["mesh:sphere_radius"]))
    if mutate is not None:
        mutate(ice)
    out = {"i2x/0": cp.asnumpy(ice.initial_i2x())}
    for w in range(h["windows"]):
        out[f"i2x/{w + 1}"] = cp.asnumpy(ice.run_window(cp.asarray(h["x2i"][w]), float(h["nextsw"][w + 1])))
    out.update({f"{pool}/{name}": cp.asnumpy(array) for (pool, name), array in given.items()})
    return out


def measure(reference: dict | None = None, *, mutate=None) -> dict[str, dict[str, int]]:
    reference = load_fixture() if reference is None else reference
    report: dict[str, dict[str, int]] = {}
    for case in cases():
        want = reference[case["name"]]
        got = run_gpu(case, trig=want["oracle/trig"], mutate=mutate)
        keys = [k for k in (*[f"i2x/{w}" for w in range(case["hybrid"]["windows"] + 1)], *expected_keys(case))
                if k in want]
        report[case["name"]] = {k: word_mismatches(got[k], want[k]) for k in keys}
    return report


__all__ = ("CASES", "FIXTURE_DIR", "X2I_FIELDS", "I2X_FIELDS", "cases", "make_case", "start_bytes",
           "coupling_bytes", "input_digest", "run_binary", "load_fixture", "load_logs", "run_gpu", "measure",
           "expected_keys")
