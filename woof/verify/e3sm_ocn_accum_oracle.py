"""E3SM v3.2.0's ocean-input accumulation: cases, stream, GPU run.

``tools/e3sm_coupler_oracle/run_ocn_accum.F90`` drives E3SM v3.2.0's
``prep_ocn_accum`` and ``prep_ocn_accum_avg`` (cut verbatim out of
``driver-mct/main/prep_ocn_mod.F90``) over ``mct_avect_accum`` and
``mct_aVect_avg`` (cut verbatim out of ``share/util/mct_mod.F90``) and the
pinned MCT on one task, built with gfortran -O0 -ffp-contract=off. Each case
is one process: x2o's fields and starting words, then a sequence of
accumulations (each with the x2o a merge would have written) and averages.
The oracle writes the count, the accumulator and x2o after each.

The cases reach: windows of 1, 2, 3, 5 and 7 samples (1/3, 1/5 and 1/7 are
rounded, so ``x * (1/n)`` is not ``x / n``); an average with nothing
accumulated (the accumulator's zeros, then the last average, copied again);
a window's first sample copied over the last average, not added; signed
zeros, subnormals and sums that overflow.
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np

from woof.verify.e3sm_ocean_merge_oracle import word_mismatches  # noqa: F401

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "e3sm_ocn_accum"

X2O_FIELDS = ("So_duu10n", "Sa_pslv", "Foxx_taux", "Foxx_tauy", "Foxx_swnet", "Foxx_lat", "Foxx_sen",
              "Foxx_lwup", "Faxa_lwdn", "Faxa_rain", "Faxa_snow", "Foxx_evap", "Fioi_melth", "Si_ifrac")
ACC, AVG = "accum", "avg"


def _spec(name, *, seed, ops, n=48, special=False):
    return dict(name=name, seed=seed, ops=tuple(ops), n=n, special=special)


SPECS = (
    _spec("window_of_three", seed=1, ops=[ACC] * 3 + [AVG]),
    _spec("window_of_two", seed=2, ops=[ACC] * 2 + [AVG]),
    _spec("window_of_one", seed=3, ops=[ACC, AVG]),
    _spec("average_without_samples", seed=4, ops=[AVG, ACC, ACC, AVG, AVG, ACC, AVG]),
    _spec("windows_of_seven_and_five", seed=5, ops=[ACC] * 7 + [AVG] + [ACC] * 5 + [AVG]),
    _spec("signed_zeros_subnormals_overflow", seed=6, ops=[ACC] * 3 + [AVG] + [ACC] * 2 + [AVG], special=True),
)


def _x2o(rng, n, special) -> np.ndarray:
    x = (rng.random((n, len(X2O_FIELDS))) - 0.5) * 10.0 ** rng.integers(-3, 6, (n, len(X2O_FIELDS)))
    if special:
        x[0:4, :] = -0.0                                  # -0 + -0 stays -0
        x[4:6, :] = 0.0                                   # -0 + +0 is +0 at rows mixed below
        x[6:10, :] = rng.integers(1, 2 ** 40, (4, x.shape[1])) * 5e-324 * rng.choice([-1, 1], (4, x.shape[1]))
        x[10:12, :] = 1.6e308                             # the sum overflows
        x[12, :] = -1.6e308
        x[13, :] = np.nextafter(0.0, 1.0)                 # the smallest subnormal times 1/3
    return x


def _build(spec) -> dict:
    rng = np.random.default_rng(4000 + spec["seed"])
    n = spec["n"]
    start = _x2o(rng, n, False)
    samples = []
    for k, op in enumerate(spec["ops"]):
        if op == ACC:
            x = _x2o(rng, n, spec["special"])
            if spec["special"] and k % 2:
                x[4:6, :] = -0.0
            samples.append(x)
        else:
            samples.append(None)
    return dict(name=spec["name"], spec=spec, aborts=False, fields=X2O_FIELDS, start=start, samples=samples)


def cases() -> list[dict]:
    return [_build(s) for s in SPECS]


def input_bytes(case: dict) -> bytes:
    s = case["spec"]
    text = ":".join(case["fields"]).encode()
    out = [struct.pack("<3i", s["n"], len(s["ops"]), len(text)), text,
           np.ascontiguousarray(case["start"], dtype="<f8").tobytes()]
    for op, x in zip(s["ops"], case["samples"]):
        out.append(struct.pack("<i", 1 if op == ACC else 2))
        if op == ACC:
            out.append(np.ascontiguousarray(x, dtype="<f8").tobytes())
    return b"".join(out)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    n, nf, nops = case["spec"]["n"], len(case["fields"]), len(case["spec"]["ops"])
    per = 4 + 2 * 8 * n * nf
    if len(raw) != per * nops:
        raise ValueError(f"{case['name']}: {len(raw)} output bytes, expected {per * nops}")
    out = {}
    counts = []
    for i in range(nops):
        block = raw[i * per:(i + 1) * per]
        counts.append(struct.unpack("<i", block[:4])[0])
        words = np.frombuffer(block[4:], dtype="<f8").astype(np.float64)
        out[f"op{i}/acc"] = words[:n * nf].reshape(n, nf)
        out[f"op{i}/x2o"] = words[n * nf:].reshape(n, nf)
    out["counts"] = np.array(counts, dtype=np.int32)
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
    """The port, operation by operation, as the oracle writes it."""
    import cupy as cp
    from woof.hex.coupled import ocn_accum_e3sm as oa
    s = case["spec"]
    acc = oa.OceanAccumulator(s["n"], case["fields"])
    x2o = cp.asarray(np.ascontiguousarray(case["start"], dtype=np.float64))
    out, counts = {}, []
    for i, (op, x) in enumerate(zip(s["ops"], case["samples"])):
        if op == ACC:
            x2o[...] = cp.asarray(x)
            acc.accum(x2o)
        else:
            acc.accum_avg(x2o)
        counts.append(acc.count)
        out[f"op{i}/acc"] = cp.asnumpy(acc.acc)
        out[f"op{i}/x2o"] = cp.asnumpy(x2o)
    out["counts"] = np.array(counts, dtype=np.int32)
    return out


def measure() -> dict[str, dict[str, int]]:
    """case -> output -> words that differ from the oracle."""
    outputs, _ = load_fixture()
    report = {}
    for case in cases():
        got, want = run_gpu(case), outputs[case["name"]]
        report[case["name"]] = {k: word_mismatches(got[k], want[k]) if k != "counts"
                                else int((got[k] != want[k]).sum()) for k in want}
    return report
