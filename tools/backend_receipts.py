"""Compare two directories of per-test launch traces, CUDA against MLX.

Each directory holds one ``<test id>.jsonl`` per test, written with
WOOF_LAUNCH_TRACE_DIR (tests/conftest.py, tools/launch_trace.py).  For each
test present in both, this reports whether every launch's array words agree
and, if not, the first launch and kernel where they part -- the first
kernel to examine, all before it having agreed.  A summary counts tests and
first-divergent kernels, so a known delta (a vendor libm kernel) is seen
once rather than test by test.

    python tools/backend_receipts.py /tmp/cuda-traces /tmp/mlx-traces
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def first_divergence(a_path: Path, b_path: Path):
    a = [json.loads(l) for l in a_path.read_text().splitlines() if l.strip()]
    b = [json.loads(l) for l in b_path.read_text().splitlines() if l.strip()]
    for x, y in zip(a, b):
        if x["kernel"] != y["kernel"]:
            return ("sequence", x["seq"], f"{x['kernel']} vs {y['kernel']}")
        if x["args"] != y["args"]:
            bad = [i for i, (p, q) in enumerate(zip(x["args"], y["args"])) if p != q]
            return ("words", x["seq"], f"{x['kernel']} args {bad}")
    if len(a) != len(b):
        return ("length", min(len(a), len(b)), f"{len(a)} vs {len(b)} launches")
    return None


def main(argv=None):
    argv = argv or sys.argv[1:]
    da, db = Path(argv[0]), Path(argv[1])
    common = sorted(set(p.name for p in da.glob("*.jsonl")) & set(p.name for p in db.glob("*.jsonl")))
    agree, kernels = 0, collections.Counter()
    for name in common:
        d = first_divergence(da / name, db / name)
        if d is None:
            agree += 1
            continue
        kind, seq, what = d
        kernels[what.split(" args")[0] if kind == "words" else kind] += 1
        print(f"{name[:-6]}: {kind} at launch {seq}: {what}")
    print(f"\n{len(common)} tests traced on both: {agree} agree on every launch word")
    for k, n in kernels.most_common():
        print(f"  {n:4d}  first divergence in {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
