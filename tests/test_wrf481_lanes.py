"""The F1 lane ledger: which dycore lanes carry to WRF v4.8.1 unchanged.

Breakage prevented: a lane re-pinned to 4.8.1 that cites a file the 4.8.1
pin does not hold, a re-pinned lane whose cited files differ from its pin in
a way the ledger did not weigh, a held lane that loses its reason, and a
ledger that no longer lists a lane directory the repository has.
"""
from __future__ import annotations

import json

from tools.wrf481_source import lanes, manifest

LEDGER = lanes.load()["lanes"]
PINNED = json.loads(manifest.MANIFEST.read_text(encoding="utf-8"))["files"]


def test_every_f1_lane_present_in_the_repository_is_in_the_ledger():
    present = {lane for lane in lanes.F1_LANES if (lanes.ROOT / lane).exists()}
    assert present == set(LEDGER)


def test_ledger_pins_match_the_lane_table():
    assert {lane: row["pin"] for lane, row in LEDGER.items()} == {
        lane: lanes.F1_LANES[lane] for lane in LEDGER}


def test_a_held_lane_names_why():
    for lane, row in LEDGER.items():
        if row["verdict"] == "held":
            assert row["exercised_changed_routines"] or row["held_because"], lane
        else:
            assert row["verdict"] == "re-pinned 4.8.1"
            assert not row["exercised_changed_routines"], lane
            assert not row.get("held_because"), lane
            assert row["files"], f"{lane} re-pinned with nothing compared"


def test_repinned_lane_files_agree_with_the_pin_manifest():
    for lane in lanes.repinned():
        for rel, row in LEDGER[lane]["files"].items():
            if row["sha256_v4.8.1"] is None or rel not in PINNED:
                continue
            assert row["sha256_v4.8.1"] == PINNED[rel]["sha256"], (lane, rel)


def test_a_differing_file_of_a_repinned_lane_carries_its_routine_delta():
    for lane in lanes.repinned():
        for rel, row in LEDGER[lane]["files"].items():
            if row["sha256_pin"] != row["sha256_v4.8.1"]:
                assert "delta" in row, (lane, rel)


def test_the_dycore_ports_stay_held():
    # R1-R4 reach these lanes; they wait for P1b/P1c (WRF-4.8.1-PLAN.md).
    for lane in ("tools/smallstep_wrf471_oracle", "tools/sms3dtke_wrf471_oracle",
                 "tools/diffopt1_wrf461_oracle", "tools/wrf_diffopt1_oracle",
                 "tools/wrf480_gate", "tools/wrf480_dycore_confirm"):
        assert LEDGER[lane]["verdict"] == "held", lane
