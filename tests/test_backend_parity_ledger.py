"""The MLX-vs-CUDA parity ledger says what its receipts say.

woof/verify/backend_parity_ledger.json records, for every test of the
sweep in tools/backend_parity_ledger.py, its outcome on each backend and
whether the words of its kernel launches agree (the receipts in
tests/data/backend_receipts, one per backend).  Every gap carries a named
cause.  These checks need no GPU:

* the ledger is what the tool computes from the committed receipts, so no
  entry can claim agreement the receipts do not show, and a gap that has
  closed (a fix, a regenerated receipt) cannot keep its old cause;
* every gap and every differing host word has a cause that the ledger
  defines, and every defined cause is used;
* both receipts come from the sweep's environment and cover every swept
  test file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import backend_parity_ledger as bpl  # noqa: E402


@pytest.fixture(scope="module")
def receipts():
    return {b: json.loads((bpl.RECEIPTS / f"{b}.json").read_text(encoding="utf-8"))
            for b in (*bpl.BACKENDS, *bpl.OPT_IN_RECEIPTS)}


@pytest.fixture(scope="module")
def ledger():
    return json.loads(bpl.LEDGER.read_text(encoding="utf-8"))


def test_ledger_is_what_the_receipts_give(receipts, ledger):
    """Regenerating from the receipts, with the ledger's own cause maps,
    gives the committed ledger."""
    assert bpl.build_ledger(receipts, ledger) == ledger


def test_every_gap_has_a_defined_cause(ledger):
    unnamed = [k for k, e in ledger["entries"].items()
               if bpl.is_gap(e) and e["cause"] not in ledger["causes"]]
    unnamed += [k for k, e in ledger["host_entries"].items()
                if not e["equal"] and e["cause"] not in ledger["causes"]]
    assert not unnamed, unnamed


def test_every_cause_is_used_and_explained(ledger):
    used = {e["cause"] for e in ledger["entries"].values() if e["cause"]}
    used |= {e["cause"] for e in ledger["host_entries"].values() if e["cause"]}
    assert used == set(ledger["causes"]), set(ledger["causes"]) ^ used
    for name, cause in ledger["causes"].items():
        assert cause.get("summary") and cause.get("evidence"), name
    stale = [k for k in ledger["test_causes"] if k not in ledger["entries"]
             or not bpl.is_gap(ledger["entries"][k])]
    assert not stale, f"cause kept for a test that is no gap: {stale}"


def test_receipts_come_from_the_sweep(receipts):
    for backend, receipt in receipts.items():
        assert receipt["schema"] == bpl.RECEIPT_SCHEMA
        assert receipt["backend"] == backend
        # as the traced process recorded it (tools/launch_trace.py)
        recorded = {k: receipt["environment"].get(k) for k in bpl.SWEEP_ENVIRONMENT}
        assert recorded == bpl.SWEEP_ENVIRONMENT, recorded
        runs_on, opt_in = bpl.OPT_IN_RECEIPTS.get(backend, (backend, bpl.DEFAULT_OFF))
        assert receipt["environment"]["backend"] == runs_on
        assert {k: receipt["environment"].get(k) or None for k in opt_in} == opt_in
        files = {nodeid.split("::")[0] for nodeid in receipt["tests"]}
        assert files == set(bpl.SWEEP_TESTS), set(bpl.SWEEP_TESTS) ^ files
        assert receipt["host_words"], f"{backend}: no host words recorded"


def test_comparison_matches_launches_as_sets():
    """A reordered or twice-recorded launch is no gap; a launch with no
    counterpart is, and the first one is named."""
    a = {"launches": [["k", "1"], ["k", "2"], ["k", "2"]]}
    assert bpl.compare(a, {"launches": [["k", "2"], ["k", "1"]]})["launches"] == "reordered"
    parted = bpl.compare(a, {"launches": [["k", "1"], ["k", "3"]]})
    assert parted["launches"] == "diverge"
    assert parted["first_divergence"]["cuda"] == [1, "k"]
    assert parted["first_divergence"]["mlx"] == [1, "k"]
    # a kernel CUDA's tracer never records (a CuPy ElementwiseKernel)
    assert bpl.compare({"launches": [["k", "1"]]},
                       {"launches": [["k", "1"], ["cupy_like", "9"]]},
                       traced_on_cuda={"k"})["launches"] == "agree"


def test_opt_in_closes_the_gaps_it_should(ledger):
    """Each gap whose cause an opt-in removes (the independent powf) agrees
    with CUDA in that opt-in's sweep."""
    for nodeid, entry in ledger["entries"].items():
        if entry["cause"] in bpl.CLOSED_BY_OPT_IN:
            assert entry.get("closed_by_opt_in", {}).get("closed") is True, nodeid
