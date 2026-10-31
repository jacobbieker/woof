"""Partial-depth nest (Phases D3-D4): the measured envelope, enforced.

Re-runs the envelope cases of woof/verify/partial_nest_envelope.py and holds
every accuracy, noise and mass metric inside the envelope the backend's own
receipt measured (tests/data/receipts/partial-nest-d4/envelope-<backend>.json,
written by tools/partial_nest_envelope.py): an error may not grow beyond 25%
(plus a small absolute floor for metrics measured near zero).  Wall time and
speedup are recorded in the receipt and not enforced (they belong to the
machine).

And the claims the receipt records, held directly: the two-way band is at
least as good as the root alone on the gravity wave (block-mean theta'), and
better on the dry and moist bubbles; D3's configuration (upwinded face,
mu/u/v/theta feedback) was not.  Minutes, not seconds: marked slow.
"""

import json
import os
from pathlib import Path

import pytest
from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

RECEIPTS = Path(__file__).resolve().parent / "data/receipts/partial-nest-d4"
GROWTH, FLOOR = 1.25, 1e-6
CASES = ["rest", "bubble", "gravity_wave", "moist_bubble", "chain_rest",
         "chain_gravity_wave"]


def _receipt():
    backend = os.environ.get("WOOF_BACKEND", "cuda")
    path = RECEIPTS / f"envelope-{backend}.json"
    if not path.is_file():
        pytest.skip(f"no envelope receipt for backend {backend!r}")
    return json.loads(path.read_text())


#: The rest cases measure the coupled noise floor (~1e-4 m/s, 1e-4 K), which
#: is not bit-reproducible run to run on Metal: MEASURED, chain_rest's
#: root-cell w RMSE 5.9e-5 in the receipt and 9.5e-5 on a rerun of the same
#: commit.  Their accuracy rows get an absolute allowance at that floor.
REST_FLOOR = 2e-4


def _bounded(got, want, what, floor=FLOOR):
    assert got <= GROWTH * abs(want) + floor, f"{what}: {got} outside {GROWTH} x {want}"


def _theta(row, group="child_block_mean_vs_allfine_block_mean"):
    return row[group]["theta_p"]["rmse"]


def test_the_receipt_records_the_d4_claims():
    rec = _receipt()
    cases, variants = rec["cases"], rec["variants"]
    gw = cases["gravity_wave"]
    assert _theta(gw) <= _theta(gw, "parent_only_vs_allfine_block_mean")
    assert _theta(variants["gravity_wave_d3"]) > _theta(gw, "parent_only_vs_allfine_block_mean")
    for name in ("bubble", "moist_bubble"):
        assert _theta(cases[name]) < _theta(cases[name], "parent_only_vs_allfine_block_mean")
    for name in CASES:
        assert cases[name]["finite"], name


@requires_gpu
@pytest.mark.parametrize("case", CASES)
def test_case_stays_inside_its_measured_envelope(case):
    from woof import wrf_exact
    if wrf_exact.ENABLED:
        pytest.skip("the partial-depth nest runs in the default build only")
    from woof.verify.partial_nest_envelope import run_case
    want = _receipt()["cases"][case]
    floor = REST_FLOOR if case.endswith("rest") else FLOOR
    got = run_case(case, timing=False)
    assert got["finite"]
    assert (got["parent_steps"], got["child_steps"], got["forces"], got["feedbacks"]) == (
        want["parent_steps"], want["child_steps"], want["forces"], want["feedbacks"])
    for group in ("child_vs_allfine", "child_block_mean_vs_allfine_block_mean"):
        for var, row in want[group].items():
            _bounded(got[group][var]["rmse"], row["rmse"], f"{group}.{var}.rmse", floor)
            _bounded(abs(got[group][var]["bias"]), abs(row["bias"]) + 0.5 * row["rmse"],
                     f"{group}.{var}.bias", floor)
    for key in ("top_zone_w_variance_max", "interface_w_max_abs", "child_w_max_abs"):
        _bounded(got[key], want[key], key)
    for key in ("child_dry_mass_drift_rel", "parent_dry_mass_drift_rel"):
        _bounded(abs(got[key]), abs(want[key]) + 1e-8, key)
