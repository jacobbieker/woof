"""E3SM v3.2.0's atmosphere-ocean flux: cases, stream, GPU run.

``tools/e3sm_coupler_oracle/run_flux_atmocn.F90`` drives E3SM v3.2.0's
``seq_flux_init_mct`` (the mask) and ``seq_flux_atmocn_mct`` (cut verbatim
out of ``driver-mct/main/seq_flux_mct.F90``) over ``shr_flux_mod`` (compiled
verbatim) and the pinned MCT on one task, built with gfortran -O0
-ffp-contract=off. Each case is one process and one call: the switches, the
flux grid's domain and fractions, a2x, o2x and xao. The oracle writes xao
after the call.

The cases reach: the explicit and implicit stress, gustiness and the
cold-air outbreak scaling, each alone and together; the iteration's count
(E3SM's -1, 1, a convergence tolerance that stops points early); the
minimum wind; dead models; the mask (the domain's, at 0.5; no ocean or ice,
+0 and -0), ``emask`` (a density of 0, one between the single-precision
``1.0e-12`` E3SM compares with and a double 1e-12, an ocean below 1 K and
at 1 K); calm points (the 0.01 floor), stable, unstable and neutral air,
|z/L| clamped at 10, ``zbot`` at the reference height; and E3SM and MCT
stopping (no iteration, with a tolerance of 1 or none allowed; the sizes; a
missing field; an unknown scheme; the diurnal cycle with the UA scheme).
"""
from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path

import numpy as np

from woof.verify.e3sm_ocean_merge_oracle import word_mismatches  # noqa: F401

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "e3sm_flux_atmocn"

DOM_COORDS = ("lat", "lon", "hgt")
DOM_OTHER = ("area", "aream", "mask", "frac")
FR_FIELDS = ("afrac", "ifrac", "ofrac", "ifrad", "ofrad")
A2X_FIELDS = ("Sa_topo", "Sa_z", "Sa_u", "Sa_v", "Sa_tbot", "Sa_ptem", "Sa_shum", "Sa_pbot", "Sa_dens", "Sa_pslv",
              "Sa_wsresp", "Sa_tau_est", "Sa_ugust", "Faxa_lwdn", "Faxa_rainc", "Faxa_rainl", "Faxa_snowc",
              "Faxa_snowl")
O2X_FIELDS = ("So_t", "So_s", "So_u", "So_v", "So_dhdx", "So_fswpen", "So_bldepth")
XAO_FIELDS = ("So_tref", "So_qref", "So_ssq", "So_re", "So_u10", "So_u10withgusts", "So_duu10n", "So_ustar",
              "Faox_taux", "Faox_tauy", "Faox_lat", "Faox_sen", "Faox_evap", "Faox_lwup", "Faox_swup", "Faox_swdn",
              "So_fswpen", "So_warm_diurn", "So_salt_diurn", "So_speed_diurn", "So_regime_diurn", "So_tskin_diurn",
              "So_tskin_day_diurn", "So_tskin_night_diurn", "So_cskin_diurn", "So_cskin_night_diurn",
              "So_tbulk_diurn", "So_warmmax_diurn", "So_windmax_diurn", "So_qsolavg_diurn", "So_windavg_diurn",
              "So_warmmaxinc_diurn", "So_windmaxinc_diurn", "So_qsolinc_diurn", "So_windinc_diurn", "So_ninc_diurn")
#: float32(1.0e-12) widened, the density E3SM's emask compares with, and a value just above it
DENS_LIMIT = float(np.float32(1.0e-12))
DENS_BETWEEN = 9.999999999e-13


def _spec(name, *, seed, n=96, dead=False, implicit=False, gust=False, coldair=False, maxit=-1, tol=0.0,
          minwind=0.5, scheme=0, diurnal=False, aborts=False, n_a2x=None, drop=()):
    return dict(name=name, seed=seed, n=n, dead=dead, implicit=implicit, gust=gust, coldair=coldair, maxit=maxit,
                tol=tol, minwind=minwind, scheme=scheme, diurnal=diurnal, aborts=aborts,
                n_a2x=n if n_a2x is None else n_a2x, drop=tuple(drop))


SPECS = (
    _spec("explicit_default", seed=1),
    _spec("implicit_stress", seed=2, implicit=True, tol=0.01),
    _spec("gustiness", seed=3, gust=True),
    _spec("coldair_outbreak", seed=4, coldair=True),
    _spec("implicit_gust_coldair", seed=5, implicit=True, gust=True, coldair=True, tol=0.01),
    _spec("converges_early", seed=6, tol=1.0e-3, maxit=10),
    _spec("dead_models", seed=7, dead=True),
    _spec("dead_implicit_gust", seed=8, dead=True, implicit=True, gust=True),
    _spec("high_minwind", seed=9, minwind=4.0, maxit=3),
    _spec("one_iteration", seed=10, maxit=1),
    _spec("abort_no_iteration", seed=11, maxit=0, aborts=True),
    _spec("abort_tolerance_one", seed=12, tol=1.0, aborts=True),
    _spec("abort_sizes", seed=13, n_a2x=95, aborts=True),
    _spec("abort_a2x_without_pslv", seed=14, drop=("a2x:Sa_pslv",), aborts=True),
    _spec("abort_gust_without_ugust", seed=15, gust=True, drop=("a2x:Sa_ugust",), aborts=True),
    _spec("abort_xao_without_ninc", seed=16, drop=("xao:So_ninc_diurn",), aborts=True),
    _spec("abort_scheme_3", seed=17, scheme=3, aborts=True),
    _spec("abort_diurnal_ua", seed=18, scheme=2, diurnal=True, aborts=True),
)


def _uniform(rng, low: float, high: float, n) -> np.ndarray:
    return (rng.random(n) * (high - low)) + low


def _fields(spec, key, base):
    drop = {d.split(":", 1)[1] for d in spec["drop"] if d.startswith(key + ":")}
    return tuple(f for f in base if f not in drop)


def _build(spec) -> dict:
    rng = np.random.default_rng(2000 + spec["seed"])
    n, na = spec["n"], spec["n_a2x"]
    dom_fields = DOM_COORDS + DOM_OTHER
    dom = _uniform(rng, -1.0, 1.0, (n, len(dom_fields)))
    dom[:, dom_fields.index("mask")] = 1.0
    dom[[3, 17], dom_fields.index("mask")] = 0.0
    dom[21, dom_fields.index("mask")] = 0.5        # kept: the mask is 0 below 0.5
    dom[22, dom_fields.index("mask")] = 0.4999     # dropped
    fr = np.column_stack([_uniform(rng, 0.0, 1.0, n), _uniform(rng, 0.0, 0.5, n), _uniform(rng, 0.0, 0.5, n),
                          _uniform(rng, 0.0, 1.0, n), _uniform(rng, 0.0, 1.0, n)])
    fr[[5, 30], 1] = 0.0
    fr[[5, 30], 2] = 0.0                           # land: no ocean, no ice
    fr[31, 1], fr[31, 2] = -0.0, 0.0               # +0 + -0: dropped
    fr[32, 1], fr[32, 2] = 0.0, 1.0e-300           # kept
    a = {f: np.zeros(na) for f in A2X_FIELDS}
    a["Sa_topo"] = _uniform(rng, 0.0, 10.0, na)
    a["Sa_z"] = _uniform(rng, 5.0, 60.0, na)
    a["Sa_u"] = _uniform(rng, -15.0, 15.0, na)
    a["Sa_v"] = _uniform(rng, -15.0, 15.0, na)
    a["Sa_ptem"] = _uniform(rng, 255.0, 310.0, na)
    a["Sa_shum"] = _uniform(rng, 1.0e-4, 2.0e-2, na)
    a["Sa_pbot"] = _uniform(rng, 9.5e4, 1.03e5, na)
    a["Sa_dens"] = _uniform(rng, 1.0, 1.35, na)
    a["Sa_pslv"] = _uniform(rng, 9.6e4, 1.04e5, na)
    a["Sa_wsresp"] = _uniform(rng, 0.0, 3.0, na)
    a["Sa_tau_est"] = _uniform(rng, 0.0, 0.6, na)
    a["Sa_ugust"] = _uniform(rng, 0.0, 4.0, na)
    for f in ("Faxa_lwdn", "Faxa_rainc", "Faxa_rainl", "Faxa_snowc", "Faxa_snowl"):
        a[f] = _uniform(rng, 0.0, 1.0, na)
    o = {f: np.zeros(n) for f in O2X_FIELDS}
    o["So_t"] = _uniform(rng, 271.0, 305.0, n)
    o["So_s"] = _uniform(rng, 30.0, 37.0, n)
    o["So_u"] = _uniform(rng, -1.0, 1.0, n)
    o["So_v"] = _uniform(rng, -1.0, 1.0, n)
    o["So_dhdx"] = _uniform(rng, -1.0, 1.0, n)
    o["So_fswpen"] = _uniform(rng, 0.0, 1.0, n)
    o["So_bldepth"] = _uniform(rng, 5.0, 50.0, n)
    m = min(n, na)
    # the atmosphere's temperature: a few kelvin under its potential temperature,
    # and cold air over warm water (tbot - ts below -10) at every fifth point
    a["Sa_tbot"][:m] = a["Sa_ptem"][:m] - _uniform(rng, 0.0, 3.0, m)
    cold = np.arange(0, m, 5)
    a["Sa_tbot"][cold] = o["So_t"][cold] - _uniform(rng, 10.5, 25.0, cold.size)
    # calm: the atmosphere moves with the ocean (wind0 is the 0.01 floor), with strong stability
    for k, d in ((40, 12.0), (41, -12.0), (42, 0.0)):
        a["Sa_u"][k], a["Sa_v"][k] = o["So_u"][k], o["So_v"][k]
        a["Sa_ptem"][k] = o["So_t"][k] + d          # 42: neutral, delt exactly 0
    a["Sa_u"][43], a["Sa_v"][43] = o["So_u"][43] + 1.0e-3, o["So_v"][43]
    a["Sa_ptem"][44] = o["So_t"][44]
    a["Sa_z"][45] = 10.0                            # zbot at the reference height: alz = 0
    a["Sa_dens"][50] = 0.0                          # emask drops it
    a["Sa_dens"][51] = DENS_BETWEEN                 # kept: above float32(1e-12), below 1e-12
    o["So_t"][52] = 0.5                             # emask drops it
    o["So_t"][53] = 1.0                             # kept
    a["Sa_wsresp"][[60, 61]] = (0.0, 40.0)          # no response; the 0.99 * wind0 cap
    a["Sa_tau_est"][61] = 0.0
    a["Sa_ugust"][62] = 0.0
    xao = _uniform(rng, -5.0, 5.0, (n, len(XAO_FIELDS)))
    a2x_fields = _fields(spec, "a2x", A2X_FIELDS)
    xao_fields = _fields(spec, "xao", XAO_FIELDS)
    a2x = np.column_stack([a[f] for f in a2x_fields])
    o2x = np.column_stack([o[f] for f in O2X_FIELDS])
    xao = np.ascontiguousarray(xao[:, [XAO_FIELDS.index(f) for f in xao_fields]])
    return dict(name=spec["name"], spec=spec, aborts=spec["aborts"], dom_fields=dom_fields, dom=dom, fractions=fr,
                a2x_fields=a2x_fields, a2x=a2x, o2x_fields=O2X_FIELDS, o2x=o2x, xao_fields=xao_fields, xao=xao)


def cases() -> list[dict]:
    return [_build(s) for s in SPECS]


def input_bytes(case: dict) -> bytes:
    s = case["spec"]
    out = [struct.pack("<11i", int(s["dead"]), int(s["implicit"]), int(s["gust"]), int(s["coldair"]), s["maxit"],
                       s["scheme"], int(s["diurnal"]), 0, s["n"], s["n_a2x"], s["n"]),
           struct.pack("<2d", s["tol"], s["minwind"])]
    lists = (DOM_OTHER, FR_FIELDS, case["a2x_fields"], case["o2x_fields"], case["xao_fields"])
    arrays = (case["dom"], case["fractions"], case["a2x"], case["o2x"], case["xao"])
    for names, a in zip(lists, arrays):
        text = ":".join(names).encode()
        out.append(struct.pack("<i", len(text)) + text)
        out.append(np.ascontiguousarray(a, dtype="<f8").tobytes())
    return b"".join(out)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    n, nf = case["spec"]["n"], len(case["xao_fields"])
    if len(raw) != 8 * n * nf:
        raise ValueError(f"{case['name']}: {len(raw)} output bytes, expected {8 * n * nf}")
    return {"xao": np.frombuffer(raw, dtype="<f8").reshape(n, nf).astype(np.float64)}


_PATH = re.compile(r"ERROR in \S*/(\S+) at line")


def abort_line(stderr: str) -> str:
    """E3SM's or MCT's line as it stops, without the stand-in's prefix and
    with the source file's directory (the build's) left out."""
    for line in stderr.splitlines():
        if "shr_sys_abort" in line or "FATAL" in line:
            line = line.strip()
            if line.startswith("shr_sys_abort: "):
                line = line[len("shr_sys_abort: "):]
            return _PATH.sub(r"ERROR in \1 at line", line).strip()
    raise RuntimeError(f"no abort line in {stderr!r}")


def load_fixture() -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    import json
    with np.load(FIXTURE_DIR / "oracle-r8.npz") as z:
        arrays = {k: z[k] for k in z.files}
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for key, value in arrays.items():
        case, _, name = key.partition("/")
        outputs.setdefault(case, {})[name] = value
    aborts = json.loads((FIXTURE_DIR / "aborts.json").read_text(encoding="utf-8"))
    return outputs, aborts


def options_for(case: dict):
    from woof.hex.coupled import flux_atmocn_e3sm as fx
    s = case["spec"]
    return fx.FluxOptions(dead_comps=s["dead"], ocn_surface_flux_scheme=s["scheme"], flux_diurnal=s["diurnal"],
                          coldair_outbreak_mod=s["coldair"], flux_convergence=s["tol"],
                          flux_max_iteration=s["maxit"],
                          atm_flux_method="implicit_stress" if s["implicit"] else "explicit",
                          atm_gustiness=s["gust"], seq_flux_atmocn_minwind=s["minwind"])


def run_gpu(case: dict) -> dict[str, np.ndarray]:
    """The port, xao as the oracle writes it."""
    import cupy as cp
    from woof.hex.coupled import flux_atmocn_e3sm as fx
    dev = {k: cp.asarray(np.ascontiguousarray(case[k], dtype=np.float64))
           for k in ("dom", "fractions", "a2x", "o2x", "xao")}
    mask = fx.flux_mask(dom=dev["dom"], dom_fields=case["dom_fields"], fractions=dev["fractions"],
                        fractions_fields=FR_FIELDS)
    fx.seq_flux_atmocn(a2x=dev["a2x"], a2x_fields=case["a2x_fields"], o2x=dev["o2x"],
                       o2x_fields=case["o2x_fields"], xao=dev["xao"], xao_fields=case["xao_fields"], mask=mask,
                       options=options_for(case))
    return {"xao": cp.asnumpy(dev["xao"])}


def measure() -> dict[str, dict[str, int]]:
    """case -> xao -> words that differ from the oracle."""
    outputs, _ = load_fixture()
    report = {}
    for case in cases():
        if case["aborts"]:
            continue
        got, want = run_gpu(case), outputs[case["name"]]
        report[case["name"]] = {k: word_mismatches(got[k], want[k]) for k in want}
    return report
