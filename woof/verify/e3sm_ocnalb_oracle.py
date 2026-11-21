"""E3SM v3.2.0's ocean albedos: cases, stream, GPU run.

``tools/e3sm_coupler_oracle/run_flux_ocnalb.F90`` drives E3SM v3.2.0's
``seq_flux_ocnalb_mct`` (cut verbatim out of
``driver-mct/main/seq_flux_mct.F90``) over the pinned MCT on one task, built
with gfortran -O0 -ffp-contract=off. Each case is one process and one or more
calls (the first does the checks and the lookups): the namelist's albedos
and ``flux_albav``, then per call ``nextsw_cday``, a2x, the fractions and
xao. The oracle writes xao and the fractions after each call.

The cases reach: ``flux_albav`` (the namelist's albedos, ``ifrad`` and
``ofrad``); the net shortwave from xao's albedos with the dark side
(``So_anidr`` exactly 1, and the values beside it), signed zeros and
``nextsw_cday`` just below -0.5; albedos of +0 and -0 (neither below 0); a
missing fraction the computed branch never looks up; and E3SM and MCT
stopping (each albedo below 0, ahead of a missing field; a missing xao,
a2x or fractions field).
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np

from woof.verify.e3sm_flux_atmocn_oracle import abort_line  # noqa: F401
from woof.verify.e3sm_ocean_merge_oracle import word_mismatches  # noqa: F401

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "e3sm_ocnalb"

DOM_COORDS = ("lat", "lon", "hgt")
DOM_OTHER = ("area", "aream", "mask", "frac")
FR_FIELDS = ("afrac", "ifrac", "ofrac", "ifrad", "ofrad")
#: the shortwave bands among other fields, out of the lookups' order
A2X_FIELDS = ("Sa_z", "Faxa_swvdf", "Faxa_swndr", "Faxa_lwdn", "Faxa_swvdr", "Faxa_swndf")
XAO_FIELDS = ("So_tref", "Faox_swup", "So_avsdf", "So_anidr", "Faox_swdn", "So_u10", "So_anidf", "So_avsdr")
CDAY_BELOW = float(np.nextafter(-0.5, -1.0))


def _spec(name, *, seed, n=64, albav=False, albdif=0.06, albdir=0.07, cdays=(-1.0,), aborts=False, drop=()):
    return dict(name=name, seed=seed, n=n, albav=albav, albdif=albdif, albdir=albdir, cdays=tuple(cdays),
                aborts=aborts, drop=tuple(drop))


SPECS = (
    _spec("albav", seed=1, albav=True, cdays=(-1.0, 120.25)),
    _spec("albav_zero_albedos", seed=2, albav=True, albdif=0.0, albdir=-0.0),
    _spec("computed", seed=3, cdays=(-1.0, -1.0, -3.0)),
    _spec("computed_cday_below_half", seed=4, cdays=(CDAY_BELOW,)),
    _spec("computed_negative_zero_albedos", seed=5, albdif=-0.0, albdir=-0.0),
    _spec("computed_without_ifrad", seed=6, drop=("fractions:ifrad",)),
    _spec("abort_albdif_negative", seed=7, albdif=-1.0e-300, aborts=True),
    _spec("abort_albdir_negative", seed=8, albdir=-0.07, drop=("xao:So_anidr",), aborts=True),
    _spec("abort_xao_without_anidr", seed=9, drop=("xao:So_anidr",), aborts=True),
    _spec("abort_xao_without_swup", seed=10, drop=("xao:Faox_swup",), aborts=True),
    _spec("abort_a2x_without_swvdf", seed=11, drop=("a2x:Faxa_swvdf",), aborts=True),
    _spec("abort_albav_without_ofrad", seed=12, albav=True, drop=("fractions:ofrad",), aborts=True),
)


def _uniform(rng, low: float, high: float, n) -> np.ndarray:
    return (rng.random(n) * (high - low)) + low


def _fields(spec, key, base):
    drop = {d.split(":", 1)[1] for d in spec["drop"] if d.startswith(key + ":")}
    return tuple(f for f in base if f not in drop)


def _call(rng, n, a2x_fields, xao_fields) -> dict:
    a2x = _uniform(rng, 0.0, 400.0, (n, len(A2X_FIELDS)))
    a2x[[2, 3], :] = -0.0                                   # no sunlight: -0 sums
    a2x[4, :] = 0.0
    xao = _uniform(rng, 0.0, 1.0, (n, len(XAO_FIELDS)))
    k = XAO_FIELDS.index("So_anidr")
    xao[[5, 9], k] = 1.0                                    # the dark side: both 0
    xao[6, k] = float(np.nextafter(1.0, 0.0))               # beside it: computed
    xao[7, k] = float(np.nextafter(1.0, 2.0))
    xao[8, [k, XAO_FIELDS.index("So_anidf")]] = 0.0          # -0 products
    fr = _uniform(rng, 0.0, 1.0, (n, len(FR_FIELDS)))
    fr[10, 1:3] = (-0.0, 0.0)
    a2x = np.ascontiguousarray(a2x[:, [A2X_FIELDS.index(f) for f in a2x_fields]])
    xao = np.ascontiguousarray(xao[:, [XAO_FIELDS.index(f) for f in xao_fields]])
    return dict(a2x=a2x, xao=xao, fr=fr)


def _build(spec) -> dict:
    rng = np.random.default_rng(3000 + spec["seed"])
    n = spec["n"]
    dom_fields = DOM_COORDS + DOM_OTHER
    dom = _uniform(rng, -90.0, 90.0, (n, len(dom_fields)))
    a2x_fields = _fields(spec, "a2x", A2X_FIELDS)
    xao_fields = _fields(spec, "xao", XAO_FIELDS)
    fr_fields = _fields(spec, "fractions", FR_FIELDS)
    calls = []
    for cday in spec["cdays"]:
        c = _call(rng, n, a2x_fields, xao_fields)
        c["fr"] = np.ascontiguousarray(c["fr"][:, [FR_FIELDS.index(f) for f in fr_fields]])
        c["cday"] = cday
        calls.append(c)
    return dict(name=spec["name"], spec=spec, aborts=spec["aborts"], dom_fields=dom_fields, dom=dom,
                a2x_fields=a2x_fields, xao_fields=xao_fields, fr_fields=fr_fields, calls=calls)


def cases() -> list[dict]:
    return [_build(s) for s in SPECS]


def input_bytes(case: dict) -> bytes:
    s = case["spec"]
    out = [struct.pack("<3i", int(s["albav"]), s["n"], len(case["calls"])),
           struct.pack("<2d", s["albdif"], s["albdir"])]
    for names in (DOM_OTHER, case["a2x_fields"], case["fr_fields"], case["xao_fields"]):
        text = ":".join(names).encode()
        out.append(struct.pack("<i", len(text)) + text)
    for i, c in enumerate(case["calls"]):
        out.append(struct.pack("<d", c["cday"]))
        if i == 0:
            out.append(np.ascontiguousarray(case["dom"], dtype="<f8").tobytes())
        for k in ("a2x", "fr", "xao"):
            out.append(np.ascontiguousarray(c[k], dtype="<f8").tobytes())
    return b"".join(out)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    n, nx, nf = case["spec"]["n"], len(case["xao_fields"]), len(case["fr_fields"])
    per = 8 * n * (nx + nf)
    if len(raw) != per * len(case["calls"]):
        raise ValueError(f"{case['name']}: {len(raw)} output bytes, expected {per * len(case['calls'])}")
    words = np.frombuffer(raw, dtype="<f8").astype(np.float64)
    out = {}
    for i in range(len(case["calls"])):
        w = words[i * (per // 8):(i + 1) * (per // 8)]
        out[f"call{i}/xao"] = w[:n * nx].reshape(n, nx)
        out[f"call{i}/fractions"] = w[n * nx:].reshape(n, nf)
    return out


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


def run_gpu(case: dict) -> dict[str, np.ndarray]:
    """The port, call by call, as the oracle writes it."""
    import cupy as cp
    from woof.hex.coupled import ocnalb_e3sm as oa
    s = case["spec"]
    out = {}
    for i, c in enumerate(case["calls"]):
        dev = {k: cp.asarray(np.ascontiguousarray(c[k], dtype=np.float64)) for k in ("a2x", "fr", "xao")}
        oa.seq_flux_ocnalb(a2x=dev["a2x"], a2x_fields=case["a2x_fields"], fractions=dev["fr"],
                           fractions_fields=case["fr_fields"], xao=dev["xao"], xao_fields=case["xao_fields"],
                           flux_albav=s["albav"], albdif=s["albdif"], albdir=s["albdir"], nextsw_cday=c["cday"])
        out[f"call{i}/xao"] = cp.asnumpy(dev["xao"])
        out[f"call{i}/fractions"] = cp.asnumpy(dev["fr"])
    return out


def measure() -> dict[str, dict[str, int]]:
    """case -> output -> words that differ from the oracle."""
    outputs, _ = load_fixture()
    report = {}
    for case in cases():
        if case["aborts"]:
            continue
        got, want = run_gpu(case), outputs[case["name"]]
        report[case["name"]] = {k: word_mismatches(got[k], want[k]) for k in want}
    return report
