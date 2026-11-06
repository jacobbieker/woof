"""The MLX-vs-CUDA parity ledger: one record per swept test, its open gaps
named.

A sweep runs the same tests on both backends with every kernel launch
traced (tools/launch_trace.py through tests/conftest.py's
``WOOF_LAUNCH_TRACE_DIR``) and a JUnit report, under the sweep's
environment (``SWEEP_ENVIRONMENT``):

    GPUWM_WRF_EXACT=1 GPUWM_NUMERICS=strict WOOF_LAUNCH_TRACE_ZERO_EMPTY=1 \\
    WOOF_LAUNCH_TRACE_DIR=/tmp/sweep-cuda/traces \\
      pixi run -e default python -m pytest --junitxml=/tmp/sweep-cuda/junit.xml \\
      $(python tools/backend_parity_ledger.py tests)

(``-e mlx`` on the Mac.)  ``GPUWM_NUMERICS=strict`` keeps the strict tests
in the traced process (tests/wrf_exact_process.py), and zeroed
``cupy.empty`` keeps the pool's old words out of the comparison.

``host-words SWEEP`` records, on each machine, the words of the host
computations that feed the kernels (the Lambert projection), and
``receipt`` condenses one backend's sweep into
``tests/data/backend_receipts/<backend>.json``: each test's outcome and,
per kernel launch, the kernel's name and a digest of the words of every
argument after the launch (the MLX facade's own element-wise kernels,
which CuPy runs untraced, are left out).  ``ledger`` compares the two
receipts into ``woof/verify/backend_parity_ledger.json``.  A test whose
outcomes differ, or whose launches part, is a gap; the ledger names its
cause through ``kernel_causes`` (by the first launch that parts) or
``test_causes`` (by test), and keeps those maps across regenerations.
tests/test_backend_parity_ledger.py holds the ledger to the receipts.

    python tools/backend_parity_ledger.py host-words /tmp/sweep-cuda   # on pb
    python tools/backend_parity_ledger.py host-words /tmp/sweep-mlx    # on the Mac
    python tools/backend_parity_ledger.py receipt cuda /tmp/sweep-cuda
    python tools/backend_parity_ledger.py receipt mlx /tmp/sweep-mlx
    python tools/backend_parity_ledger.py receipt mlx-cuda-powf /tmp/sweep-mlx-cuda-powf
    python tools/backend_parity_ledger.py ledger
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "tests" / "data" / "backend_receipts"
LEDGER = ROOT / "woof" / "verify" / "backend_parity_ledger.json"
RECEIPT_SCHEMA = "woof.backend-receipt/v1"
LEDGER_SCHEMA = "woof.backend-parity-ledger/v1"
BACKENDS = ("cuda", "mlx")
#: Sweeps with an opt-in, each run on a backend: the label, its backend and
#: the variables it sets.  WOOF_STRICT_POWF=cuda puts the opt-in
#: transcription of CUDA's powf in Metal's strict power (NOTICE).
OPT_IN_RECEIPTS = {"mlx-cuda-powf": ("mlx", {"WOOF_STRICT_POWF": "cuda"})}
#: The default receipts are made without any opt-in.
DEFAULT_OFF = {"WOOF_STRICT_POWF": None}
#: Gap causes an opt-in sweep must close: cause -> opt-in receipt.
CLOSED_BY_OPT_IN = {"independent-powf": "mlx-cuda-powf"}
SWEEP_ENVIRONMENT = {"GPUWM_WRF_EXACT": "1", "GPUWM_NUMERICS": "strict",
                     "WOOF_LAUNCH_TRACE_ZERO_EMPTY": "1"}
#: The swept test files.
SWEEP_TESTS = (
    "tests/test_dudhia_sw_wrf461_parity.py",
    "tests/test_gf_wrf461_parity.py",
    "tests/test_gwdo_kim_wrf461_parity.py",
    "tests/test_kessler_wrf461_parity.py",
    "tests/test_lin_wrf461_parity.py",
    "tests/test_morrison_wrf461_parity.py",
    "tests/test_myjpbl_wrf461_parity.py",
    "tests/test_myjsfc_wrf461_parity.py",
    "tests/test_noah_seaice_wrf461_parity.py",
    "tests/test_noah_wrf461_parity.py",
    "tests/test_rrtm_lw_wrf461_parity.py",
    "tests/test_sfclay_classic_wrf461_parity.py",
    "tests/test_sfclayrev_wrf461_parity.py",
    "tests/test_shinhong_wrf461_parity.py",
    "tests/test_topo_wind_wrf461_chain_parity.py",
    "tests/test_uh_wrf461_parity.py",
    "tests/test_urban_ucm_wrf461_parity.py",
    "tests/test_wsm3_wrf461_column_exact.py",
    "tests/test_wsm6_wrf461_column_exact.py",
    "tests/test_wsm6_wrf461_gpu_parity.py",
    "tests/test_wsm6r_wrf461_column_exact.py",
    "tests/test_wsm7_wrf461_column_exact.py",
    "tests/test_ysu_wrf461_parity.py",
    "tests/test_ysu_wrf461_wide_parity.py",
    "tests/test_nssl2_column_oracle.py",
    "tests/test_strict_float32_power.py",
    "tests/test_partial_nest_kernel_words.py",
    "tests/test_partial_nest_d4_kernel_words.py",
    "tests/test_projection_oracle.py",
    "tests/test_lambert.py",
)
#: Kernels only the MLX facade launches (CuPy's own element-wise kernels
#: are not traced on CUDA).
FACADE_KERNEL = re.compile(r"^woof_f32_")


def trace_name(nodeid: str) -> str:
    """The file tests/conftest.py traces ``nodeid`` to."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", nodeid)[:200] + ".jsonl"


def _outcomes(junit: Path) -> dict[str, str]:
    out = {}
    for case in ET.parse(junit).getroot().iter("testcase"):
        module = case.get("classname", "")
        parts = module.split(".")
        # classname is the module path, plus a class for methods
        path = "/".join(parts) + ".py"
        if not (ROOT / path).exists() and len(parts) > 1:
            path = "/".join(parts[:-1]) + ".py::" + parts[-1]
        nodeid = f"{path}::{case.get('name')}"
        outcome = "passed"
        for tag in ("failure", "error", "skipped"):
            if case.find(tag) is not None:
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
        out[nodeid] = outcome
    return out


def _launch_digest(row) -> str:
    h = hashlib.sha256(row["kernel"].encode())
    for arg in row["args"]:
        # an array: dtype, shape and its words after the launch
        h.update(json.dumps(arg[:3] if isinstance(arg, list) else arg).encode())
    return h.hexdigest()[:12]


#: Host computations whose words feed the kernels, recorded per machine:
#: the Lambert projection of tests/fixtures/portable_static_grids.json.
HOST_GRIDS = ROOT / "tests" / "fixtures" / "portable_static_grids.json"
HOST_FIELDS = ("latlon_mass", "mapfac_m", "coriolis_m", "rotation_m")


def host_words() -> dict[str, str]:
    """Digest of each host projection field, this machine's words."""
    import numpy as np
    from woof.static.lambert import LambertGrid
    out = {}
    for key, row in json.loads(HOST_GRIDS.read_text()).items():
        grid = LambertGrid(**row["spec"])
        if row.get("nest") is not None:
            # the witness grid tests/test_static_projection_portability.py uses
            grid = grid.nest(*row["nest"])
        for name in HOST_FIELDS:
            value = getattr(grid, name)
            value = value() if callable(value) else value
            parts = value if isinstance(value, tuple) else (value,)
            h = hashlib.sha256()
            for part in parts:
                h.update(np.ascontiguousarray(part).tobytes())
            out[f"lambert:{key}:{name}"] = h.hexdigest()[:16]
    return out


def build_receipt(backend: str, sweep: Path) -> dict:
    tests = {}
    for nodeid, outcome in sorted(_outcomes(sweep / "junit.xml").items()):
        record = {"outcome": outcome, "launches": []}
        trace = sweep / "traces" / trace_name(nodeid)
        if trace.exists():
            for line in trace.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if FACADE_KERNEL.match(row["kernel"]):
                    continue
                record["launches"].append([row["kernel"], _launch_digest(row)])
        tests[nodeid] = record
    host = sweep / "host_words.json"
    recorded = sweep / "traces" / "environment.json"
    if not recorded.exists():
        raise SystemExit(f"{recorded}: the sweep did not record its environment "
                         "(tools/launch_trace.py writes it)")
    environment = json.loads(recorded.read_text())
    wrong = {k: environment.get(k) for k, v in SWEEP_ENVIRONMENT.items()
             if environment.get(k) != v}
    if wrong:
        raise SystemExit(f"{sweep}: not the sweep's environment: {wrong}")
    runs_on, opt_in = OPT_IN_RECEIPTS.get(backend, (backend, DEFAULT_OFF))
    if environment.get("backend") != runs_on:
        raise SystemExit(f"{sweep}: traced on backend {environment.get('backend')!r}, "
                         f"not {runs_on!r}")
    wrong = {k: environment.get(k) for k, v in opt_in.items() if (environment.get(k) or None) != v}
    if wrong:
        raise SystemExit(f"{sweep}: not the {backend} sweep's opt-ins: {wrong}")
    return {"schema": RECEIPT_SCHEMA, "backend": backend,
            "environment": environment,
            "host_words": json.loads(host.read_text()) if host.exists() else {},
            "tests": tests}


def compare(cuda: dict, mlx: dict, traced_on_cuda=None) -> dict:
    """``{"launches": agree|reordered|diverge|none, "first_divergence": ...}``.

    Launches are matched as sets of (kernel, words) digests.  A test that
    issues its calls in another order on one backend (a thread pool, a
    cache an earlier test filled), or a launch the CUDA tracer records
    twice (a kernel reached through two wrapped loaders), computed the same
    words, and is ``reordered``.  Otherwise the first divergence is the
    first launch, in each backend's order, with no counterpart on the
    other.  ``traced_on_cuda``: the kernels the CUDA tracer records at
    all; CuPy's own ElementwiseKernels (``gpuwm_rrtm_sequential_sum``, ...)
    are not, so their MLX facade launches are left out, as
    tools/backend_receipts.py does."""
    a, b = cuda["launches"], mlx["launches"]
    if traced_on_cuda is not None:
        b = [y for y in b if y[0] in traced_on_cuda]
    if not a and not b:
        return {"launches": "none", "first_divergence": None}
    if a == b:
        return {"launches": "agree", "first_divergence": None}
    sa, sb = {tuple(x) for x in a}, {tuple(y) for y in b}
    if sa == sb:
        return {"launches": "reordered", "first_divergence": None}
    only_cuda = [i for i, x in enumerate(a) if tuple(x) not in sb]
    only_mlx = [i for i, y in enumerate(b) if tuple(y) not in sa]
    return {"launches": "diverge", "first_divergence": {
        "cuda": [only_cuda[0], a[only_cuda[0]][0]] if only_cuda else None,
        "mlx": [only_mlx[0], b[only_mlx[0]][0]] if only_mlx else None,
        "unmatched": {"cuda": len(only_cuda), "mlx": len(only_mlx)}}}


def is_gap(entry: dict) -> bool:
    return entry["cuda"] != entry["mlx"] or entry["launches"] == "diverge"


def cause_of(nodeid: str, entry: dict, ledger: dict):
    """The named cause of a gap: by test first, then by the kernel whose
    launch parts first."""
    if nodeid in ledger["test_causes"]:
        return ledger["test_causes"][nodeid]
    first = entry["first_divergence"]
    if first is not None:
        return ledger["kernel_causes"].get((first["cuda"] or first["mlx"])[1])
    return None


def build_ledger(receipts: dict, previous: dict | None) -> dict:
    previous = previous or {}
    ledger = {
        "schema": LEDGER_SCHEMA,
        "receipts": {b: str((RECEIPTS / f"{b}.json").relative_to(ROOT))
                     for b in (*BACKENDS, *OPT_IN_RECEIPTS) if b in receipts},
        "environment": SWEEP_ENVIRONMENT,
        "sweep_tests": list(SWEEP_TESTS),
        "causes": previous.get("causes", {}),
        "kernel_causes": previous.get("kernel_causes", {}),
        "test_causes": previous.get("test_causes", {}),
        "host_causes": previous.get("host_causes", {}),
        "host_entries": {},
        "entries": {},
    }
    hc, hm = receipts["cuda"].get("host_words", {}), receipts["mlx"].get("host_words", {})
    for name in sorted(set(hc) | set(hm)):
        equal = hc.get(name) == hm.get(name)
        family = name.split(":", 1)[0]
        ledger["host_entries"][name] = {
            "equal": equal, "cause": None if equal else ledger["host_causes"].get(family)}
    cuda, mlx = receipts["cuda"]["tests"], receipts["mlx"]["tests"]
    traced_on_cuda = {x[0] for t in cuda.values() for x in t["launches"]}
    for nodeid in sorted(set(cuda) | set(mlx)):
        c = cuda.get(nodeid, {"outcome": "absent", "launches": []})
        m = mlx.get(nodeid, {"outcome": "absent", "launches": []})
        entry = {"cuda": c["outcome"], "mlx": m["outcome"], **compare(c, m, traced_on_cuda)}
        entry["cause"] = cause_of(nodeid, entry, ledger) if is_gap(entry) else None
        label = CLOSED_BY_OPT_IN.get(entry["cause"])
        if label is not None and label in receipts:
            o = receipts[label]["tests"].get(nodeid, {"outcome": "absent", "launches": []})
            closed = {"cuda": c["outcome"], "mlx": o["outcome"], **compare(c, o, traced_on_cuda)}
            entry["closed_by_opt_in"] = {"receipt": label, "closed": not is_gap(closed)}
        ledger["entries"][nodeid] = entry
    gaps = [e for e in ledger["entries"].values() if is_gap(e)]
    ledger["summary"] = {
        "tests": len(ledger["entries"]),
        "agree": sum(1 for e in ledger["entries"].values() if not is_gap(e)),
        "gaps": len(gaps),
        "gaps_by_cause": dict(sorted(collections.Counter(
            str(e["cause"]) for e in gaps).items())),
        "host_words": len(ledger["host_entries"]),
        "host_words_differing": sum(1 for e in ledger["host_entries"].values() if not e["equal"]),
    }
    return ledger


def _dump(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=False) + "\n", encoding="utf-8")


def _dump_receipt(path: Path, receipt: dict) -> None:
    """One line per test, so a regenerated receipt diffs test by test."""
    head = {k: v for k, v in receipt.items() if k != "tests"}
    lines = [json.dumps(head, sort_keys=True)[:-1] + ', "tests": {']
    items = sorted(receipt["tests"].items())
    for i, (nodeid, record) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        lines.append(f" {json.dumps(nodeid)}: {json.dumps(record, separators=(',', ':'))}{comma}")
    lines.append("}}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    if argv[0] == "tests":
        print(" ".join(SWEEP_TESTS))
        return 0
    if argv[0] == "host-words":
        out = Path(argv[1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "host_words.json").write_text(json.dumps(host_words(), indent=1) + "\n")
        return 0
    if argv[0] == "receipt":
        backend, sweep = argv[1], Path(argv[2])
        if backend not in BACKENDS and backend not in OPT_IN_RECEIPTS:
            raise SystemExit(f"backend must be one of {BACKENDS + tuple(OPT_IN_RECEIPTS)}")
        _dump_receipt(RECEIPTS / f"{backend}.json", build_receipt(backend, sweep))
        return 0
    if argv[0] == "ledger":
        receipts = {b: json.loads((RECEIPTS / f"{b}.json").read_text())
                    for b in (*BACKENDS, *OPT_IN_RECEIPTS) if (RECEIPTS / f"{b}.json").exists()}
        previous = json.loads(LEDGER.read_text()) if LEDGER.exists() else None
        ledger = build_ledger(receipts, previous)
        _dump(LEDGER, ledger)
        print(json.dumps(ledger["summary"], indent=1))
        unnamed = [k for k, e in ledger["entries"].items() if is_gap(e) and not e["cause"]]
        unnamed += [k for k, e in ledger["host_entries"].items() if not e["equal"] and not e["cause"]]
        for k in unnamed:
            print("gap without a named cause:", k)
        return 1 if unnamed else 0
    raise SystemExit(f"unknown command {argv[0]!r}")


if __name__ == "__main__":
    sys.exit(main())
