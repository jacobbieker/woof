"""``woof energy hindcast optimize`` (woof.energy.cost_optimize).

Pinned here: the search re-cuts the sites once per (card memory, leaf
spacing) and reuses the cut on disk; rows meeting the deadline are ranked
by the market's central cost and the first is ``chosen`` with a recipe;
every row is marked an estimate with the band, the utilisation and the
price table's age; the spin-up sensitivity re-prices the chosen row at
0-3 h; and every input outside the requirement (spacing outside 100-300 m,
a level count below the lean wind-energy column, an off-cycle kept window,
an impossible deadline) is a named refusal.  CPU only; level checks run
WRF's compute_eta bridge and say so.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from datetime import date
from pathlib import Path

import pytest

from conftest import requires_wrf_eta_bridge
from woof.energy import cost
from woof.energy import cost_optimize as opt

FIXTURES = Path(__file__).parent / "fixtures" / "energy"
SITES_WORLD = FIXTURES / "sites_world.json"
TODAY = date(2025, 7, 2)


def _parse(argv):
    from woof.energy.cli import register_cli

    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers(dest="command"))
    return parser.parse_args(["energy", "hindcast", "optimize", *argv])


def _cli(argv):
    args = _parse(argv)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = args.func(args)
    return code, json.loads(out.getvalue()), err.getvalue()


SMALL = ["--sites", str(SITES_WORLD), "--deadline-days", "30",
         "--leaf-dx-m", "300,200", "--nz", "49",
         "--instance", "aws:g5.xlarge", "--instance", "aws:g6.xlarge",
         "--instance", "aws:p5.48xlarge"]


@requires_wrf_eta_bridge
def test_optimize_ranks_meets_the_deadline_and_chooses(tmp_path):
    parts = tmp_path / "parts"
    code, doc, err = _cli([*SMALL, "--partition-dir", str(parts),
                           "-o", str(tmp_path / "opt.json")])
    assert code == 0, doc
    assert doc["schema"] == opt.OPTIMIZE_SCHEMA and doc["kind"] == "estimate"
    assert json.loads((tmp_path / "opt.json").read_text())["top"] == doc["top"]
    # One cut per (card memory, spacing): A10G and L4 share ~22.5 GiB.
    built = doc["search"]["partitions"]
    assert sorted((p["leaf_dx_m"], p["card"]) for p in built) == [
        (200.0, "a10g-24gb"), (200.0, "h100-sxm-80gb"),
        (300.0, "a10g-24gb"), (300.0, "h100-sxm-80gb")]
    assert all(Path(p["path"]).is_file() for p in built)
    # 2 spacings x 1 level count x 1 layout x 3 offers.
    assert doc["search"]["configurations"] == 6
    top = doc["top"]
    costs = [row["cost_usd"]["spot"]["central"] for row in top]
    assert costs == sorted(costs)
    assert [row["rank"] for row in top] == list(range(1, len(top) + 1))
    for row in top:
        assert row["kind"] == "estimate" and row["deadline"]["meets_deadline"]
        assert row["gpu_hours"]["low"] < row["gpu_hours"]["central"] \
            < row["gpu_hours"]["high"]
        assert row["deadline"]["instances_needed"]["high"] >= \
            row["deadline"]["instances_needed"]["central"]
        assert row["deadline"]["wall_clock_days"] <= 30.0
        assert (row["segment_hours"], row["spinup_hours"]) == (9, 3)
        assert row["segments"] == 1460
    chosen = doc["chosen"]
    assert chosen["offer"] == top[0]["offer"] and chosen["rank"] == 1
    # The coarse spacing is the cheaper on the same offer.
    by = {(r["leaf_dx_m"], r["offer"]): r["cost_usd"]["spot"]["central"]
          for r in top}
    assert by[(300.0, "aws:g5.xlarge")] < by[(200.0, "aws:g5.xlarge")]
    assert chosen["leaf_dx_m"] == 300.0 and chosen["nz"] == 49
    assert chosen["recipe"][0].startswith("woof energy partition")
    assert "--leaf-dx-m 300" in chosen["recipe"][0]
    assert "--segment-hours 9 --spinup-hours 3" in chosen["recipe"][2]
    # The estimate step prices the leaves the plan step writes.
    assert chosen["recipe"][1].endswith("--store-preset windfarm")
    assert "--leaf-output-preset" not in chosen["recipe"][2]
    # Honesty fields: band, utilisation and the price table's age.
    assert doc["model"]["throughput_band"]["pessimistic"] == 0.5
    assert "ASSUMPTIONS" in doc["model"]["utilisation"]["status"]
    assert doc["prices"]["as_of"] == "2025-07-01"
    assert doc["prices"]["age_days"] > 0 and "ILLUSTRATIVE" in doc["prices"]["spot"]
    # Spin-up sensitivity: 0-3 h at the same kept window, 3 h the reference.
    sens = {s["spinup_hours"]: s for s in doc["spinup_sensitivity"]}
    assert sorted(sens) == [0, 1, 2, 3]
    assert sens[3]["relative_to_chosen"] == 1.0
    assert sens[0]["cost_usd"]["spot"] < sens[3]["cost_usd"]["spot"]
    assert sens[0]["segment_hours"] == 6
    assert "chosen:" in err and "ESTIMATES" in err
    # A second search reuses the cuts on disk.
    code, again, _ = _cli([*SMALL, "--partition-dir", str(parts)])
    assert code == 0
    assert {p["origin"] for p in again["search"]["partitions"]} == {"reused"}
    assert again["chosen"]["cost_usd"] == chosen["cost_usd"]


@requires_wrf_eta_bridge
def test_spinup_options_are_ranked_when_asked(tmp_path):
    code, doc, _ = _cli(["--sites", str(SITES_WORLD), "--deadline-days", "30",
                         "--leaf-dx-m", "300", "--nz", "49",
                         "--spinup-hours", "0,3", "--instance",
                         "aws:g5.xlarge"])
    assert code == 0
    spins = [row["spinup_hours"] for row in doc["top"]]
    assert spins == [0, 3]          # less spin-up is cheaper, never hidden
    assert doc["top"][0]["segment_hours"] == 6


@requires_wrf_eta_bridge
def test_max_instances_and_deadline_refusals(tmp_path):
    code, doc, _ = _cli([*SMALL, "--max-instances", "1"])
    if code == 0:
        assert all(r["deadline"]["instances"] == 1 for r in doc["top"])
    else:
        assert "--max-instances" in doc["refused"]
    code, doc, err = _cli([*SMALL[:3], "0.0001", *SMALL[4:]])
    assert code == 2 and "deadline" in doc["refused"]
    assert doc["schema"] == opt.OPTIMIZE_SCHEMA and "REFUSED" in err


@requires_wrf_eta_bridge
@pytest.mark.parametrize("extra, match", [
    (["--leaf-dx-m", "350"], "outside the reanalysis range"),
    (["--leaf-dx-m", "50"], "outside the reanalysis range"),
    (["--nz", "48"], "smallest lean column has 49"),
    (["--keep-hours", "5"], "6-hourly global cycle"),
    (["--spinup-hours", "-1"], "negative"),
])
def test_out_of_requirement_inputs_are_refused(extra, match):
    base = ["--sites", str(SITES_WORLD), "--deadline-days", "30",
            "--instance", "aws:g5.xlarge"]
    code, doc, _ = _cli([*base, *extra])
    assert code == 2 and match in doc["refused"], doc


def test_input_refusals_need_no_bridge(tmp_path):
    code, doc, _ = _cli(["--deadline-days", "30"])
    assert code == 2 and "exactly one of --sites" in doc["refused"]
    # A malformed partition is a refusal (exit 2, JSON), not a traceback.
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "something-else"}))
    code, doc, _ = _cli(["--partition", str(bad), "--deadline-days", "30"])
    assert code == 2 and doc["schema"] == opt.OPTIMIZE_SCHEMA
    code, doc, _ = _cli(["--sites", str(SITES_WORLD), "--no-repartition",
                         "--deadline-days", "30"])
    assert code == 2 and "needs --partition" in doc["refused"]


@requires_wrf_eta_bridge
def test_no_repartition_prices_the_partition_as_cut(tmp_path):
    from woof.energy.cli import register_cli  # noqa: F401 (parser import)
    from woof.cli import build_parser

    partition = tmp_path / "partition.json"
    args = build_parser().parse_args(
        ["energy", "partition", str(SITES_WORLD), "--card", "a10g",
         "--leaf-dx-m", "250", "-o", str(partition)])
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        assert args.func(args) == 0
    code, doc, _ = _cli(["--partition", str(partition), "--no-repartition",
                         "--deadline-days", "30", "--nz", "49",
                         "--instance", "aws:g5.xlarge",
                         "--instance", "aws:p5.48xlarge"])
    assert code == 0, doc
    assert doc["search"]["repartitioned"] is False
    assert {r["leaf_dx_m"] for r in doc["top"]} == {250.0}
    code, doc, _ = _cli(["--partition", str(partition), "--no-repartition",
                         "--leaf-dx-m", "300", "--deadline-days", "30",
                         "--nz", "49", "--instance", "aws:g5.xlarge"])
    assert code == 2 and "cut at 250 m" in doc["refused"]
    # Without --no-repartition the partition's sites are re-cut per spacing.
    code, doc, _ = _cli(["--partition", str(partition), "--leaf-dx-m", "300",
                         "--deadline-days", "30", "--nz", "49",
                         "--instance", "aws:g5.xlarge"])
    assert code == 0 and doc["chosen"]["leaf_dx_m"] == 300.0
    assert doc["inputs"]["partition"]["leaf_dx_m"] == 250.0


def test_optimize_core_ranks_with_an_injected_partition():
    # The search without the CLI, the bridge or a real cut: one synthetic
    # cluster priced for two offers, ranked on-demand.
    specs = [cost.ClusterSpec("c", 400.0, 300.0, None)]
    record = {"specs": specs, "clusters": 1, "path": None, "leaf_dx_m": 300.0,
              "card": "x", "vram_gib": 22.5}
    table, _ = cost.load_prices(today=TODAY)
    offers = cost.select_offers(table, ["aws:g5.xlarge", "aws:g6.xlarge"])
    result = opt.optimize(
        partitions_for=lambda gib, dx, card: record, offers=offers,
        price_table=table, leaf_dx=(300.0,), nz=(49, 70),
        layouts=(opt.Layout(6, 3),), period_hours=8760.0, deadline_days=30.0,
        rank_market="on_demand")
    rows = result["rows"]
    assert len(rows) == 4 and result["other_market"] == "spot"
    keys = [r["cost_usd"]["on_demand"]["central"] for r in rows]
    assert keys == sorted(keys)
    cost_of = {(r["offer"], r["nz"]): r["cost_usd"]["on_demand"]["central"]
               for r in rows}
    for offer in ("aws:g5.xlarge", "aws:g6.xlarge"):
        assert cost_of[(offer, 49)] < cost_of[(offer, 70)]
    with pytest.raises(opt.OptimizeRefused, match="deadline"):
        opt.optimize(partitions_for=lambda gib, dx, card: record,
                     offers=offers, price_table=table, leaf_dx=(300.0,),
                     nz=(49,), layouts=(opt.Layout(6, 3),),
                     period_hours=8760.0, deadline_days=1.0,
                     max_instances=1)
