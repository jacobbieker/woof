"""Cloud card rows and the hindcast cost estimator (woof energy estimate).

CPU only: nothing here probes a GPU.  The plan fixture is written by hand
in the shapes ``woof energy plan --topology wrf-tiles`` emits (plan.json,
a tile spec, the tile TOML and the parent TOML), so the estimator's
duck-typed reader is held to the planner's real field names without
running the planner.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import pytest

from woof import card_catalog
from woof.card_catalog import CARD_CATALOG, DECLARED, MEASURED, card_row, cloud_card_row
from woof.energy import cost
from woof.energy.cost import (
    COST_ESTIMATE_SCHEMA,
    Campaign,
    ClusterSpec,
    CostRefusal,
    DomainWork,
    calibration,
    clusters_from_partition,
    domain_seconds,
    domains_from_plan,
    load_prices,
    read_job,
    select_offers,
    synth_cluster,
)

DATACENTRE = ("h100-sxm-80gb", "h100-pcie-80gb", "h200-141gb", "a100-80gb",
              "a100-40gb", "l40s-48gb", "l4-24gb", "a10g-24gb")


# --------------------------------------------------------------------------
# card catalog
# --------------------------------------------------------------------------
def test_every_cloud_part_is_a_declared_datacentre_row():
    for key in DATACENTRE:
        row = CARD_CATALOG[key]
        assert row.provenance == DECLARED and row.datacentre, key
        assert row.as_dict()["provenance_note"] == "declared, not measured locally"
        assert row.sm_count > 0 and row.bandwidth_gbs > 0 and row.memory_mib > 0
    measured = [row for row in CARD_CATALOG.values() if row.provenance == MEASURED]
    assert {row.key for row in measured} == {"rtx-5090", "rtx-5070-ti", "rtx-3080"}
    assert not any(row.datacentre for row in measured)


@pytest.mark.parametrize("spelling, key", [
    ("h100-80gb", "h100-pcie-80gb"), ("H100", "h100-pcie-80gb"),
    ("NVIDIA H100 SXM", "h100-sxm-80gb"), ("h100-sxm-80gb", "h100-sxm-80gb"),
    ("h200", "h200-141gb"), ("a100", "a100-40gb"), ("a100-80gb", "a100-80gb"),
    ("A100 40GB", "a100-40gb"), ("l40s", "l40s-48gb"), ("L4", "l4-24gb"),
    ("a10g", "a10g-24gb"),
])
def test_cloud_spellings_resolve_to_their_row(spelling, key):
    assert cloud_card_row(spelling).key == key


def test_card_capacity_reads_cloud_rows_and_leaves_consumer_spellings_alone():
    from woof.domain_wizard import card_capacity_gib

    assert card_capacity_gib("h100-80gb") == pytest.approx(81_559 / 1024)
    assert card_capacity_gib("l40s") == pytest.approx(46_068 / 1024)
    assert card_capacity_gib("a10g") == pytest.approx(23_028 / 1024)
    assert card_capacity_gib("a100-40gb") == 40.0
    # Untouched paths: tiers, consumer models, sizes, the A10 (not A10G).
    assert card_capacity_gib("24gb") == 24.0
    assert card_capacity_gib("RTX 3080") == 10.0
    assert card_capacity_gib("5070 Ti") == 16.0
    assert card_capacity_gib("a10") == 24.0
    assert card_capacity_gib("l40") == 48.0
    assert card_capacity_gib("not-a-card") is None
    assert cloud_card_row("RTX 5090") is None


def test_declared_cloud_card_sizes_without_probing(monkeypatch):
    import woof.domain_wizard as dw

    def no_probe():
        raise AssertionError("a declared card must not probe the local GPU")

    monkeypatch.setattr(dw, "device_memory_probe_subprocess", no_probe)
    budget = dw.resolve_sizing_budget("h100-80gb", None)
    assert budget.vram_gib == pytest.approx(81_559 / 1024)
    assert budget.measured is False and budget.device_profile is None


def test_catalog_aliases_are_unambiguous():
    # The index is built at import and raises on a collision.
    assert card_catalog._alias_index()
    assert card_row("h100 foo") is None


# --------------------------------------------------------------------------
# throughput model
# --------------------------------------------------------------------------
def test_calibration_reproduces_each_receipt_within_its_spread():
    cal = calibration()
    k = cal["k_cell_steps_per_s_per_gbs"]
    assert {r["id"] for r in cal["receipts"]} == {
        "first-light-rtx4070-single-domain",
        "live-products-rtx5070ti-three-grid-stack"}
    assert cal["excluded"] and cal["excluded"][0]["reason"]
    for receipt in cost.CALIBRATION_RECEIPTS:
        card = CARD_CATALOG[receipt["card"]]
        predicted = sum(
            domain_seconds(DomainWork("r", "parent", nx, ny, nz, 1000.0, 1.0,
                                      float(steps), restart_s=None,
                                      history_s=1e12), card, k)["compute_s"]
            for nx, ny, nz, steps in receipt["domains"])
        assert predicted == pytest.approx(receipt["wall_s"], rel=0.15), receipt["id"]


def _domain(**over):
    base = dict(domain_id="d", role="leaf", nx=500, ny=500, nz=60, dx_m=200.0,
                dt_s=1.0, run_seconds=3600.0)
    base.update(over)
    return DomainWork(**base)


def test_cost_scales_with_steps_cells_substeps_and_card():
    k = calibration()["k_cell_steps_per_s_per_gbs"]
    h100 = CARD_CATALOG["h100-sxm-80gb"]
    l4 = CARD_CATALOG["l4-24gb"]
    base = domain_seconds(_domain(), h100, k)["compute_s"]
    assert domain_seconds(_domain(dt_s=0.5), h100, k)["compute_s"] == \
        pytest.approx(2 * base)
    assert domain_seconds(_domain(sound_steps=8), h100, k)["compute_s"] == \
        pytest.approx(base * 12 / 8)
    assert domain_seconds(_domain(), l4, k)["compute_s"] > 10 * base
    # Occupancy: per-cell cost of a small grid exceeds a large one's.
    small = domain_seconds(_domain(nx=50, ny=50), h100, k)["compute_s"] / (50 * 50)
    assert small > base / (500 * 500)
    # Band: optimistic is faster, pessimistic slower.
    assert domain_seconds(_domain(), h100, k, factor=1.5)["compute_s"] < base


# --------------------------------------------------------------------------
# plan mode
# --------------------------------------------------------------------------
PARENT_TOML = """
[experiment]
name = "p"
run_seconds = 108000.0
restart_interval_s = 3600.0

[shared]
nz = 49
time_step_sound = 4

[[domain]]
grid_id = 1
nx = 134
ny = 66
time_step = 5
dx = 1000.0
history_interval_s = 900.0
"""

TILE_TOML = """
[grid]
nx = 520
ny = 180
nz = 60
dx = 200.0

[run]
dt = 1.0
time_step_sound = 6
output_interval_s = 900.0
restart_interval_s = 3600.0
"""


def write_plan(root: Path, *, peak: float = 7.3) -> Path:
    (root / "parent").mkdir(parents=True)
    (root / "tiles").mkdir()
    (root / "parent" / "parent.toml").write_text(PARENT_TOML)
    (root / "tiles" / "tile-0001.toml").write_text(TILE_TOML)
    (root / "tiles" / "tile-0001.json").write_text(json.dumps({
        "schema": "woof-energy.wrf-tile.v1", "domain_id": "tile-0001",
        "role": "child", "level": 1, "dx_m": 200.0, "nx": 520, "ny": 180,
        "dt_s": 1.0, "nz": 60,
        "child_config": {"path": "tiles/tile-0001.toml", "sha256": "0" * 64},
        "vram": {"peak_envelope_gib": peak, "capacity_gib": 80.0}}))
    plan = {
        "schema": "woof-energy.plan.v1", "topology": "wrf-tiles",
        "dx_m": 200.0, "start": "2025-01-01T00", "hours": 30.0,
        "domains": [
            {"domain_id": "parent", "role": "parent", "parent": None,
             "dx_m": 1000.0, "config": "parent/parent.toml", "grid_id": 1,
             "extra": {"nx": 134, "ny": 66, "history_interval_s": 900.0,
                       "restart_interval_s": 3600.0, "run_seconds": 108000.0,
                       "peak_envelope_gib": 3.05}},
            {"domain_id": "tile-0001", "role": "child", "parent": "parent",
             "dx_m": 200.0, "config": "tiles/tile-0001.json", "grid_id": 1,
             "extra": {"nx": 520, "ny": 180}},
        ]}
    path = root / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def test_plan_reader_takes_fields_from_plan_spec_and_tomls(tmp_path):
    rows = {d.domain_id: d for d in domains_from_plan(write_plan(tmp_path))}
    parent, tile = rows["parent"], rows["tile-0001"]
    assert (parent.role, parent.nx, parent.ny, parent.nz, parent.dt_s) == \
        ("parent", 134, 66, 49, 5.0)
    assert (tile.role, tile.nx, tile.ny, tile.nz, tile.dt_s, tile.sound_steps) \
        == ("leaf", 520, 180, 60, 1.0, 6)
    assert tile.level == 1 and parent.level == 0
    assert tile.peak_gib == 7.3 and tile.run_seconds == 108000.0
    shorter = domains_from_plan(tmp_path / "plan.json", run_seconds=3600.0)
    assert all(d.run_seconds == 3600.0 for d in shorter)


def test_plan_reader_refuses_other_documents(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text(json.dumps({"schema": "woof-energy.sites.v1"}))
    with pytest.raises(CostRefusal, match="not a woof-energy.plan.v1"):
        domains_from_plan(bad)
    bad.write_text(json.dumps({"schema": "woof-energy.plan.v1",
                               "topology": "hex-swath", "domains": []}))
    with pytest.raises(CostRefusal, match="hex-swath"):
        domains_from_plan(bad)


def _parse(argv):
    from woof.energy.cli import register_cli

    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


def _run(argv, capsys):
    args = _parse(["energy", "estimate", *argv])
    code = args.func(args)
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


def test_plan_estimate_end_to_end(tmp_path, capsys):
    plan = write_plan(tmp_path)
    out = tmp_path / "estimate.json"
    code, doc, err = _run(["--plan", str(plan), "--segments", "365",
                           "--instance", "aws:p5.48xlarge",
                           "--instance", "gcp:a3-highgpu-8g",
                           "--instance", "aws:g5.xlarge",
                           "-o", str(out)], capsys)
    assert code == 0
    assert doc["schema"] == COST_ESTIMATE_SCHEMA and doc["mode"] == "plan"
    assert json.loads(out.read_text())["offers"] == doc["offers"]
    assert "aws:p5.48xlarge" in err and "ESTIMATE" in err
    p5, a3, g5 = doc["offers"]
    assert p5["card"]["key"] == a3["card"]["key"] == "h100-sxm-80gb"
    # Same card, same GPU-hours; the price differs.
    assert p5["gpu_hours"] == a3["gpu_hours"]
    low, central, high = (p5["gpu_hours"][k] for k in ("low", "central", "high"))
    assert low < central < high
    for market in ("on_demand", "spot"):
        c = p5["cost_usd"][market]
        assert c["low"] < c["central"] < c["high"]
    # Billed instance-hours carry the packing loss and spot the rework.
    assert p5["instance_hours"]["central"] == pytest.approx(
        central / (8 * 0.85), rel=1e-3)
    assert p5["cost_usd"]["spot"]["central"] == pytest.approx(
        p5["instance_hours"]["central"] * 1.10 * 27.50, rel=1e-3)
    comp = p5["gpu_hours_by_component"]
    assert sum(comp.values()) == pytest.approx(central, rel=1e-3)
    assert comp["compute_h"] > comp["startup_h"] > 0
    # Storage: leaf interiors only, 24 stored hours per segment at 900 s.
    storage = doc["storage"]
    assert storage["frames_per_segment"] == 96
    assert storage["leaf_columns_stored"] == (520 - 22) * (180 - 22)
    assert storage["stored_bytes"] == pytest.approx(
        storage["raw_bytes"] * 0.35, rel=1e-6)
    assert p5["storage_usd"]["per_month"] > 0
    # A tile sized for an 80 GB card still fits a 24 GB A10G at 7.3 GiB.
    assert g5["fits"] is True
    assert doc["work_per_segment"][1]["role"] == "leaf"


def test_plan_offer_too_small_for_its_tiles_is_refused_not_priced(tmp_path, capsys):
    plan = write_plan(tmp_path, peak=60.0)
    code, doc, err = _run(["--plan", str(plan), "--instance", "aws:g5.xlarge",
                           "--instance", "aws:p5.48xlarge"], capsys)
    assert code == 0
    g5, p5 = doc["offers"]
    assert g5["fits"] is False and "re-plan with --card a10g-24gb" in g5["refused"]
    assert "cost_usd" not in g5 and p5["fits"] is True
    assert "REFUSED" in err


def test_wall_clock_and_deadline(tmp_path, capsys):
    plan = write_plan(tmp_path)
    code, one, _ = _run(["--plan", str(plan), "--instance", "aws:p5.48xlarge",
                         "--deadline-days", "1"], capsys)
    code, many, _ = _run(["--plan", str(plan), "--instance", "aws:p5.48xlarge",
                          "--instances", "1000"], capsys)
    o1, om = one["offers"][0], many["offers"][0]
    assert o1["wall_clock_days"]["central"] == pytest.approx(
        o1["instance_hours"]["central"] / 24.0, rel=1e-3)
    # Enough instances: the wall is one segment's chain, not less.
    assert om["wall_clock_days"]["central"] == pytest.approx(
        om["critical_path_hours_per_segment"] / 24.0, abs=1e-3)
    # Spot is priced, so the deadline is sized on spot's rework-inflated hours.
    deadline = o1["instances_for_deadline"]
    assert deadline["market"] == "spot"
    assert o1["spot_instance_hours"]["central"] == pytest.approx(
        1.10 * o1["instance_hours"]["central"], rel=1e-3)
    assert deadline["central"] == math.ceil(
        o1["spot_instance_hours"]["central"] / 24.0)
    assert o1["wall_clock_days_spot"]["central"] > o1["wall_clock_days"]["central"]
    _, od, _ = _run(["--plan", str(plan), "--instance", "aws:p5.48xlarge",
                     "--deadline-days", "1", "--market", "on-demand"], capsys)
    o2 = od["offers"][0]
    assert o2["instances_for_deadline"]["market"] == "on_demand"
    assert o2["instances_for_deadline"]["central"] == math.ceil(
        o2["instance_hours"]["central"] / 24.0)
    assert "spot_instance_hours" not in o2


def test_store_preset_moves_leaf_output_time_and_bytes(tmp_path, capsys):
    plan = write_plan(tmp_path)
    _, full, _ = _run(["--plan", str(plan), "--instance", "aws:p5.48xlarge"],
                      capsys)
    _, energy, _ = _run(["--plan", str(plan), "--instance", "aws:p5.48xlarge",
                         "--store-preset", "energy"], capsys)
    f, e = full["offers"][0], energy["offers"][0]
    assert f["gpu_hours_by_component"]["output_h"] > \
        e["gpu_hours_by_component"]["output_h"]
    assert f["storage"]["stored_bytes"] > e["storage"]["stored_bytes"]
    assert full["storage"]["offer"] == "aws:p5.48xlarge"


def test_storage_is_per_offer_in_synthesised_modes(capsys):
    _, doc, _ = _run(["--clusters", "2", "--footprint-km2", "20000",
                      "--instance", "aws:g5.xlarge",
                      "--instance", "aws:p5.48xlarge"], capsys)
    g5, p5 = doc["offers"]
    # Smaller cards cut the cover into more tiles, each with its own rim.
    assert g5["work_per_segment"]["leaf"]["domains"] > \
        p5["work_per_segment"]["leaf"]["domains"]
    assert doc["storage"]["stored_bytes"] == g5["storage"]["stored_bytes"]
    assert g5["storage"]["leaf_columns_stored"] != \
        p5["storage"]["leaf_columns_stored"]


@pytest.mark.parametrize("argv, match", [
    (["--leaf-dx-m", "100"], "--leaf-dx-m shape synthesised clusters"),
    (["--cover-factor", "1.5", "--nz", "80"], "--nz, --cover-factor"),
    (["--footprint-km2", "10"], "--footprint-km2"),
])
def test_plan_refuses_flags_it_would_ignore(argv, match, tmp_path, capsys):
    code, doc, _ = _run(["--plan", str(write_plan(tmp_path)), *argv], capsys)
    assert code == 2 and match in doc["refused"]


def test_plan_domain_without_dx_is_refused(tmp_path):
    plan = write_plan(tmp_path)
    document = json.loads(plan.read_text())
    del document["domains"][1]["dx_m"]
    plan.write_text(json.dumps(document))
    with pytest.raises(CostRefusal, match="no positive dx_m"):
        domains_from_plan(plan)


def test_partition_refuses_footprint_flag_and_mismatched_chains(tmp_path, capsys):
    doc = tmp_path / "p.json"
    doc.write_text(json.dumps({"clusters": [
        {"id": "a", "area_km2": 10.0},
        {"id": "b", "area_km2": 10.0, "dx_m": 500.0}]}))
    code, out, _ = _run(["--partition", str(doc), "--footprint-km2", "9"], capsys)
    assert code == 2 and "--footprint-km2 is summary-mode" in out["refused"]
    # The flag's chain lands on the 200 m cluster only; b keeps 2500->500.
    code, out, _ = _run(["--partition", str(doc), "--chain-m", "5000,1000,200",
                         "--instance", "aws:p5.48xlarge"], capsys)
    assert code == 0
    roles = out["offers"][0]["work_per_segment"]
    assert roles["intermediate"]["dx_m"] == [1000.0]
    assert roles["leaf"]["dx_m"] == [200.0, 500.0]
    assert sorted(roles["parent"]["dx_m"]) == [2500.0, 5000.0]
    with pytest.raises(CostRefusal, match="coarsest first"):
        synth_cluster("x", 10.0, leaf_dx_m=500.0, chain_m=(1000.0, 200.0),
                      card=CARD_CATALOG["h100-sxm-80gb"], leaf_nz=60,
                      parent_nz=49, run_seconds=3600.0)
    doc.write_text(json.dumps({"clusters": [{"id": "a", "area_km2": 1.0,
                                             "chain": "1km,200"}]}))
    with pytest.raises(CostRefusal, match="not a list of metres"):
        clusters_from_partition(doc, leaf_dx_m=200.0, cover_factor=2.0)


def test_job_with_no_segments_is_refused(tmp_path):
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"segments": []}))
    with pytest.raises(CostRefusal, match="no segments"):
        read_job(job)
    job.write_text(json.dumps({"segment_count": "all"}))
    with pytest.raises(CostRefusal, match="not a number"):
        read_job(job)


def test_measured_catalog_rows_agree_with_the_device_admission_record():
    from woof.hex.device_admission import KNOWN_CARDS

    pairs = {"rtx-5090": "32gib-170sm", "rtx-5070-ti": "16gib-70sm",
             "rtx-3080": "10gib-68sm"}
    for key, profile in pairs.items():
        assert CARD_CATALOG[key].sm_count == KNOWN_CARDS[profile].multiprocessors


# --------------------------------------------------------------------------
# summary and partition modes
# --------------------------------------------------------------------------
def test_synthesised_tiles_fit_the_card_and_split_when_large():
    h100, l4 = CARD_CATALOG["h100-sxm-80gb"], CARD_CATALOG["l4-24gb"]
    for card in (h100, l4):
        rows = synth_cluster("c", 5000.0, leaf_dx_m=200.0, chain_m=None,
                             card=card, leaf_nz=60, parent_nz=49,
                             run_seconds=3600.0)
        assert [r.role for r in rows] == ["parent", "leaf"]
        assert rows[0].dx_m == 1000.0 and rows[1].dx_m == 200.0
        assert all(r.cells <= cost.max_tile_cells(card) for r in rows)
    big = synth_cluster("c", 50000.0, leaf_dx_m=200.0, chain_m=None, card=l4,
                        leaf_nz=60, parent_nz=49, run_seconds=3600.0)
    small = synth_cluster("c", 50000.0, leaf_dx_m=200.0, chain_m=None, card=h100,
                          leaf_nz=60, parent_nz=49, run_seconds=3600.0)
    assert big[1].count > small[1].count >= 1
    chained = synth_cluster("c", 100.0, leaf_dx_m=100.0,
                            chain_m=(2500.0, 500.0, 100.0), card=h100,
                            leaf_nz=60, parent_nz=49, run_seconds=3600.0)
    assert [r.role for r in chained] == ["parent", "intermediate", "leaf"]
    assert [r.dt_s for r in chained] == [12.5, 2.5, 0.5]


def test_summary_is_n_times_one_cluster_and_matches_partition(tmp_path, capsys):
    common = ["--instance", "aws:p5.48xlarge", "--segments", "10"]
    _, one, _ = _run(["--clusters", "1", "--footprint-km2", "50", *common], capsys)
    _, three, _ = _run(["--clusters", "3", "--footprint-km2", "150", *common],
                       capsys)
    assert three["campaign"]["clusters"] == 3
    for key in ("central", "low", "high"):
        assert three["offers"][0]["gpu_hours"][key] == pytest.approx(
            3 * one["offers"][0]["gpu_hours"][key], rel=1e-3, abs=0.03)
        assert three["offers"][0]["cost_usd"]["spot"][key] == pytest.approx(
            3 * one["offers"][0]["cost_usd"]["spot"][key], rel=1e-3, abs=0.5)
    assert three["storage"]["stored_bytes"] == pytest.approx(
        3 * one["storage"]["stored_bytes"], rel=1e-6)
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps({
        "schema": "woof-energy.partition.v1",
        "clusters": [{"cluster_id": f"c{i}", "footprint_km2": 50.0,
                      "leaf_dx_m": 200.0} for i in range(3)]}))
    _, part, _ = _run(["--partition", str(partition), *common], capsys)
    assert part["mode"] == "partition" and part["campaign"]["clusters"] == 3
    assert part["offers"][0]["gpu_hours"]["central"] == pytest.approx(
        three["offers"][0]["gpu_hours"]["central"], rel=1e-3)
    assert [c["cluster_id"] for c in part["offers"][0]["clusters"]] == \
        ["c0", "c1", "c2"]


def test_partition_reads_regions_chains_and_refuses_arealess_rows(tmp_path):
    region = tmp_path / "r.geojson"
    region.write_text(json.dumps({"type": "Polygon", "coordinates": [[
        [0.0, 50.0], [0.1, 50.0], [0.1, 50.1], [0.0, 50.1], [0.0, 50.0]]]}))
    doc = tmp_path / "p.json"
    doc.write_text(json.dumps({"clusters": [
        {"id": "a", "region": [0.0, 50.0, 0.1, 50.1], "chain": "1000->200",
         "est_tile_count": 3},
        {"id": "b", "region": "r.geojson", "dx_m": 100.0},
        {"id": "c", "area_km2": 10.0},
    ]}))
    a, b, c = clusters_from_partition(doc, leaf_dx_m=200.0, cover_factor=2.0)
    assert a.leaf_area_km2 == pytest.approx(7.15 * 11.12, rel=0.02)
    assert a.chain_m == (1000.0, 200.0) and a.est_tiles == 3
    assert b.leaf_area_km2 == pytest.approx(a.leaf_area_km2, rel=1e-6)
    assert b.leaf_dx_m == 100.0
    assert c.leaf_area_km2 == 20.0 and c.footprint_km2 == 10.0
    doc.write_text(json.dumps({"clusters": [{"id": "x", "sites": 3}]}))
    with pytest.raises(CostRefusal, match="carries no area"):
        clusters_from_partition(doc, leaf_dx_m=200.0, cover_factor=2.0)
    doc.write_text(json.dumps({"clusters": []}))
    with pytest.raises(CostRefusal, match="no 'clusters'"):
        clusters_from_partition(doc, leaf_dx_m=200.0, cover_factor=2.0)


def test_job_document_supplies_segments_and_flags_override(tmp_path, capsys):
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "schema": "woof-energy.hindcast-job.v1", "cluster": "c0",
        "segment_hours": 30, "spinup_hours": 6,
        "segments": [{"start": f"2025-01-{d:02d}T00"} for d in range(1, 11)],
        "planner_args": ["--dx-m", "200", "--nz", "60"]}))
    facts = read_job(job)
    assert facts["segments"] == 10 and facts["leaf_dx_m"] == "200"
    _, doc, _ = _run(["--clusters", "1", "--footprint-km2", "20", "--job",
                      str(job), "--instance", "aws:p5.48xlarge"], capsys)
    assert doc["campaign"]["segments"] == 10
    assert doc["campaign"]["stored_hours_per_segment"] == 24.0
    _, doc, _ = _run(["--clusters", "1", "--footprint-km2", "20", "--job",
                      str(job), "--segments", "5", "--instance",
                      "aws:p5.48xlarge"], capsys)
    assert doc["campaign"]["segments"] == 5


# --------------------------------------------------------------------------
# prices and refusals
# --------------------------------------------------------------------------
def test_shipped_price_table_is_dated_sourced_and_complete():
    table, _ = load_prices(today=date(2025, 7, 2))
    assert table["as_of"] and table["sources"]
    wanted = {("aws", "p5.48xlarge"), ("aws", "p4d.24xlarge"),
              ("aws", "g6e.xlarge"), ("aws", "g5.xlarge"),
              ("gcp", "a3-highgpu-8g"), ("gcp", "a2-ultragpu-1g"),
              ("gcp", "g2-standard-4")}
    have = {(r["provider"], r["instance"]) for r in table["instances"]}
    assert wanted <= have
    for row in table["instances"]:
        assert row["on_demand_usd_per_hour"] > row["spot_usd_per_hour"] > 0
        assert card_row(row["card"]).datacentre
    _, notes = load_prices(today=date(2030, 1, 1))
    assert notes and "refresh mark" in notes[0]


def test_price_table_refusals(tmp_path):
    bad = tmp_path / "prices.json"
    bad.write_text(json.dumps({"schema": "woof-energy.cloud-prices.v1",
                               "as_of": "2025-07-01", "instances": [
                                   {"provider": "x", "instance": "y",
                                    "card": "warp-drive", "gpus": 1}]}))
    with pytest.raises(CostRefusal, match="no row for"):
        load_prices(bad)
    bad.write_text(json.dumps({"schema": "woof-energy.cloud-prices.v1"}))
    with pytest.raises(CostRefusal, match="as_of"):
        load_prices(bad)
    table, _ = load_prices()
    with pytest.raises(CostRefusal, match="not in the price table"):
        select_offers(table, ["aws:p9.huge"])
    assert {r["provider"] for r in select_offers(table, ["gcp"])} == {"gcp"}


@pytest.mark.parametrize("argv, match", [
    ([], "exactly one of"),
    (["--clusters", "3"], "needs --footprint-km2"),
    (["--clusters", "3", "--footprint-km2", "10", "--chain-m", "1000,300"],
     "ends at 300"),
    (["--clusters", "3", "--footprint-km2", "10", "--chain-m", "200,1000"],
     "coarsest first"),
    (["--clusters", "3", "--footprint-km2", "10", "--spinup-hours", "40"],
     "leaves nothing stored"),
])
def test_cli_refusals_exit_2_with_json(argv, match, capsys):
    code, doc, err = _run(argv, capsys)
    assert code == 2 and doc["schema"] == COST_ESTIMATE_SCHEMA
    assert match in doc["refused"] and "REFUSED" in err


def test_register_estimate_mounts_under_another_group():
    from woof.energy.cli import register_estimate

    parser = argparse.ArgumentParser()
    group = parser.add_subparsers(dest="g").add_parser("hindcast") \
        .add_subparsers(dest="h")
    register_estimate(group)
    args = parser.parse_args(["hindcast", "estimate", "--clusters", "2",
                              "--footprint-km2", "5"])
    assert args.clusters == 2 and callable(args.func)


def test_campaign_stored_hours():
    assert Campaign(365, 30.0, 6.0).stored_hours == 24.0
    assert ClusterSpec("c", 1.0, 200.0, None).region_km2 is None


# --------------------------------------------------------------- windfarm


def test_windfarm_bytes_come_from_the_presets_own_field_list():
    from woof.io.history_selection import WINDFARM_FIELD_REASONS
    from woof.io.wrf_output_schema import HUB_HEIGHT_DIMENSIONED_FIELDS

    layout = cost.store_layout("windfarm", 60, hub_heights=6)
    fields = layout["fields"]
    assert set(fields["two_d"]) | set(fields["hub_height"]) | set(
        fields["other"]) == set(WINDFARM_FIELD_REASONS)
    assert set(fields["hub_height"]) == (set(WINDFARM_FIELD_REASONS)
                                         & HUB_HEIGHT_DIMENSIONED_FIELDS)
    assert layout["three_d"] == 0
    assert layout["planes"] == len(fields["two_d"]) + 6 * len(
        fields["hub_height"])
    # 19-byte Times padded to 20, ZNU/ZNW columns, four scalars, HUB_HEIGHT.
    assert layout["extra_bytes"] == 20 + 4 * (60 + 61) + 4 * 4 + 4 * 6
    # Linear in the hub heights, independent of the levels (no volume).
    two = cost.store_layout("windfarm", 60, hub_heights=2)
    assert layout["planes"] - two["planes"] == 4 * len(fields["hub_height"])
    assert cost.store_layout("windfarm", 120, 6)["planes"] == layout["planes"]
    energy = cost.store_layout("energy", 60)
    full = cost.store_layout("full", 60)
    assert layout["planes"] * 8 < energy["planes"] < full["planes"]
    with pytest.raises(cost.CostRefusal, match="not one of"):
        cost.store_layout("severe", 60)


def test_windfarm_two_d_members_are_planes_on_a_real_inventory():
    """Every member counted as one plane is (ny, nx) on the writer's own
    inventory: the cost model cannot price a volume as a plane."""
    from woof.config import RunConfig
    from woof.io.history_layout import produced_history_shapes

    cfg = RunConfig(nx=20, ny=10, nz=8, dx=1000.0, dy=1000.0, ztop=10000.0,
                    dt=6.0, run_seconds=60.0)
    shapes = produced_history_shapes(cfg)
    layout = cost.store_layout("windfarm", 8)
    checked = [name for name in layout["fields"]["two_d"] if name in shapes]
    assert {"XLAT", "XLONG", "HGT", "MUB", "SINALPHA"} <= set(checked)
    for name in checked:
        assert shapes[name] == (10, 20), name


def test_store_preset_windfarm_prices_far_less_than_energy(tmp_path, capsys):
    plan = write_plan(tmp_path)
    _, energy, _ = _run(["--plan", str(plan), "--instance", "aws:p5.48xlarge",
                         "--store-preset", "energy"], capsys)
    _, wind, table = _run(["--plan", str(plan), "--instance",
                           "aws:p5.48xlarge", "--store-preset", "windfarm"],
                          capsys)
    _, wind2, _ = _run(["--plan", str(plan), "--instance", "aws:p5.48xlarge",
                        "--store-preset", "windfarm", "--hub-heights", "2"],
                       capsys)
    assert wind["storage"]["stored_bytes"] * 5 < energy["storage"]["stored_bytes"]
    assert wind2["storage"]["stored_bytes"] < wind["storage"]["stored_bytes"]
    per_frame = wind["storage"]["fields_per_frame"]
    assert per_frame["three_d"] == 0 and per_frame["hub_heights"] == 6
    assert "WSPD_HUB" in per_frame["fields"]["hub_height"]
    assert "no 3-D field" in wind["storage"]["basis"]
    assert "2-D and hub planes only" in table
    assert wind["offers"][0]["gpu_hours_by_component"]["output_h"] < \
        energy["offers"][0]["gpu_hours_by_component"]["output_h"]
    code, refused, _ = _run(["--plan", str(plan), "--hub-heights", "3"], capsys)
    assert code == 2 and "--hub-heights" in refused["refused"]


def test_the_job_names_the_store_preset_and_hub_heights(tmp_path, capsys):
    plan = write_plan(tmp_path)
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "segments": 4, "history": {"leaf_output_preset": "windfarm",
                                   "leaf_hub_heights_m": [80.0, 100.0, 120.0]}}))
    _, doc, _ = _run(["--plan", str(plan), "--job", str(job),
                      "--instance", "aws:p5.48xlarge"], capsys)
    assert doc["storage"]["preset"] == "windfarm"
    assert doc["storage"]["fields_per_frame"]["hub_heights"] == 3
    assert any("leaf_output_preset" in n for n in doc["notes"])
    _, flagged, _ = _run(["--plan", str(plan), "--job", str(job),
                          "--instance", "aws:p5.48xlarge",
                          "--store-preset", "energy"], capsys)
    assert flagged["storage"]["preset"] == "energy"
