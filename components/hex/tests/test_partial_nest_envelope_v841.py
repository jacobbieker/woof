"""Hex partial-depth band (Phase D5): the measured envelope, enforced.

Re-runs the envelope cases of woof/verify/hex_partial_nest_envelope.py and
holds every accuracy, noise and mass metric inside the envelope the
backend's own receipt measured (tests/data/receipts/partial-nest-d5/
envelope-<backend>.json, written by tools/hex_partial_nest_envelope.py): an
error may not grow beyond 25% (plus a small absolute floor for metrics
measured near zero).  Wall time and speedup are recorded in the receipt and
not enforced (they belong to the machine).

And the claims the receipt records, held directly:

* the band beats the parent alone on the warm bubble's block-mean theta';
* D3's upwinded interface face is worse than the parent's second-order face
  on the gravity wave (the D4 lesson, reproduced on MPAS);
* a resting atmosphere stays at rest to the coupled noise floor.

What the receipt also records and this file does NOT claim: on the gravity
wave the band is worse than the parent alone (block-mean theta'), and
feedback makes it worse still (the one-way row).  The moist bubble with
radiation is refused (the receipt's ``refused`` entry).  Minutes, not
seconds: marked slow; the meshes need tools/rustwx's rw_mpas_mesh.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow]

RECEIPTS = Path(__file__).resolve().parents[3] / "tests/data/receipts/partial-nest-d5"
GROWTH, FLOOR = 1.25, 1e-6
CASES = ["rest", "gravity_wave", "bubble"]
#: The rest case measures the coupled noise floor (~1e-5 m/s, 1e-5 K).
REST_FLOOR = 2e-4


def _receipt():
    backend = os.environ.get("WOOF_BACKEND", "cuda")
    path = RECEIPTS / f"envelope-{backend}.json"
    if not path.is_file():
        pytest.skip(f"no envelope receipt for backend {backend!r}")
    return json.loads(path.read_text())


def _theta(row, group="child_block_mean_vs_allfine_block_mean"):
    return row[group]["theta_p"]["rmse"]


def test_the_receipt_records_the_d5_claims():
    rec = _receipt()
    cases, variants = rec["cases"], rec["variants"]
    for name in CASES:
        assert cases[name]["finite"], name
    bubble = cases["bubble"]
    assert _theta(bubble) < _theta(bubble, "parent_only_vs_allfine_block_mean")
    gw = cases["gravity_wave"]
    assert _theta(variants["gravity_wave_upwind_face"]) > _theta(gw)
    assert cases["rest"]["child_w_max_abs"] < 1e-3
    assert "moist_bubble" in rec["refused"]


def _bounded(got, want, what, floor=FLOOR):
    assert got <= GROWTH * abs(want) + floor, f"{what}: {got} outside {GROWTH} x {want}"


def _device():
    pytest.importorskip("cupy")
    from woof.hex.cuda_backend import KernelCache
    from woof.hex.cuda_backend.runtime import CudaRefusal
    try:
        KernelCache()
    except CudaRefusal as error:
        pytest.skip(f"no admitted hex device: {error}")


@pytest.mark.parametrize("case", CASES)
def test_case_stays_inside_its_measured_envelope(case):
    _device()
    from woof.verify import hex_partial_nest_envelope as envelope
    try:
        envelope.ensure_meshes()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    want = _receipt()["cases"][case]
    floor = REST_FLOOR if case == "rest" else FLOOR
    got = envelope.run_case(case, timing=False)
    assert got["finite"]
    assert (got["parent_steps"], got["child_steps"], got["forces"], got["feedbacks"]) == (
        want["parent_steps"], want["child_steps"], want["forces"], want["feedbacks"])
    for group in ("child_vs_allfine", "child_block_mean_vs_allfine_block_mean"):
        for var, row in want[group].items():
            _bounded(got[group][var]["rmse"], row["rmse"], f"{group}.{var}.rmse", floor)
            _bounded(abs(got[group][var]["bias"]), abs(row["bias"]) + 0.5 * row["rmse"],
                     f"{group}.{var}.bias", floor)
    for key in ("top_zone_w_variance_max", "interface_w_max_abs", "child_w_max_abs"):
        _bounded(got[key], want[key], key, floor)
    for key in ("child_dry_mass_drift_rel", "parent_dry_mass_drift_rel"):
        _bounded(abs(got[key]), abs(want[key]) + 1e-8, key)
