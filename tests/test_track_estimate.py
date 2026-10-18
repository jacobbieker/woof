"""woof estimate-tracks: the moving-nest LES track estimator."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from woof import track_estimate as te


def _tracks():
    return [{"name": "a", "duration_h": 2.0, "envelope_km": [30.0, 10.0]},
            {"name": "b", "duration_h": 5.5, "envelope_km": [200.0, 80.0]}]


def test_cell_steps_per_domain_follow_the_design():
    doc = te.estimate_tracks(_tracks())
    a = doc["tracks"][0]
    d01, d02, d03 = a["domains"]
    # d02 and d03 have fixed shapes; d03 spins up 1 h, d02 3 h.
    assert (d02["nx"], d02["ny"], d02["nz"]) == (241, 241, 256)
    assert d02["cells"] == 241 * 241 * 256
    assert d02["steps"] == math.ceil((3 + 2) * 3600 / 1.5)
    assert d03["cells"] == 325 * 325 * 256
    assert d03["steps"] == (1 + 2) * 3600 * 2
    assert d03["cell_steps"] == d03["cells"] * d03["steps"]
    # d01: envelope + 60 km each side at 1.5 km, floored by the envelope
    # plus d02's box plus the parent rim.
    assert d01["nx"] == max(math.ceil((30 + 120) / 1.5) + 1,
                            math.ceil((30000 + 241 * 300) / 1500) + 20)
    assert d01["ny"] == max(math.ceil((10 + 120) / 1.5) + 1,
                            math.ceil((10000 + 241 * 300) / 1500) + 20)
    assert d01["steps"] == math.ceil(5 * 3600 / 7.5)
    assert a["cell_steps"] == sum(d["cell_steps"] for d in a["domains"])
    b = doc["tracks"][1]
    assert b["domains"][0]["nx"] == math.ceil(320 / 1.5) + 1


def test_output_is_the_leaf_every_interval_with_the_compression_band():
    doc = te.estimate_tracks(_tracks())
    out = doc["tracks"][0]["output"]
    assert out["domain"] == "d03"
    assert out["frames"] == 2 * 3600 // 120 + 1
    # The balloon preset as the disk planner prices it: thirteen volumes,
    # about 52 bytes per cell per frame at 256 levels.
    assert 52.0 < out["frame_raw_bytes"] / (325 * 325 * 256) < 53.0
    lo, hi = out["stored_bytes"]
    assert lo == pytest.approx(out["raw_bytes"] * 0.26)
    assert hi == pytest.approx(out["raw_bytes"] * 0.40)
    t = doc["totals"]
    assert t["output_stored_tb"][0] == pytest.approx(
        sum(r["output"]["stored_bytes"][0] for r in doc["tracks"]) / 1e12)


def test_output_preset_is_priced_by_the_disk_planner():
    from woof.config import RunConfig
    from woof.disk_budget import history_frame_bytes
    from woof.io.history_selection import HistorySelection

    design = te.replace(te.TrackDesign(), output_preset="full")
    doc = te.estimate_tracks(_tracks(), design, cards=["a6000"])
    cfg = RunConfig(nx=325, ny=325, nz=256, dx=100.0, dy=100.0,
                    ztop=25000.0, dt=0.5, run_seconds=3600.0,
                    **te.LEAF_PHYSICS)
    assert doc["tracks"][0]["output"]["frame_raw_bytes"] == history_frame_bytes(
        cfg, HistorySelection.from_mapping({"preset": "full"}))
    with pytest.raises(te.EstimateRefusal, match="--output-preset"):
        te.estimate_tracks(_tracks(), te.replace(design,
                                                 output_preset="no-such"))


def test_only_the_a6000_is_calibrated_and_the_rest_scale_by_bandwidth():
    doc = te.estimate_tracks(_tracks())
    cards = doc["totals"]["cards"]
    assert cards["rtx-a6000"]["calibrated"]
    a6000 = cards["rtx-a6000"]["ordinary_cell_steps_per_s_central"]
    for key in ("l40s-48gb", "a100-40gb", "a100-80gb", "h100-sxm-80gb",
                "h200-141gb"):
        c = cards[key]
        assert not c["calibrated"]
        assert c["calibration"].startswith("uncalibrated")
        assert c["aws"]["label"] == "verify before launch"
    ratio = (cards["h100-sxm-80gb"]["ordinary_cell_steps_per_s_central"] / a6000
             * te.BAND_CALIBRATED[1] / te.BAND_UNCALIBRATED[1])
    assert ratio == pytest.approx(3350.0 / 768.0)
    assert cards["rtx-a6000"]["aws"] is None


def test_gpu_hours_bands_and_cost_are_consistent():
    doc = te.estimate_tracks(_tracks())
    c = doc["totals"]["cards"]["h100-sxm-80gb"]
    g = c["gpu_hours"]
    assert g["optimistic"] < g["central"] < g["pessimistic"]
    assert c["aws"]["usd_per_gpu_hour"] == pytest.approx(55.04 / 8)
    assert c["aws"]["usd"]["central"] == pytest.approx(
        g["central"] * 55.04 / 8)
    per = [r["cards"]["h100-sxm-80gb"]["wall_hours"]["central"]
           for r in doc["tracks"]]
    assert g["central"] == pytest.approx(sum(per))
    # Central wall time is the cell-steps at the card's central rate plus
    # the preparation and history-copy overheads.
    r0 = doc["tracks"][0]
    rate = c["ordinary_cell_steps_per_s_central"]
    from woof.energy import cost

    over = (cost.PREP_INTERCEPT_S + cost.PREP_S_PER_MCELL * r0["cells"] / 1e6
            + r0["output"]["raw_bytes"] / (cost.D2H_GBPS * 1e9))
    assert per[0] == pytest.approx(
        (r0["ordinary_cell_steps"] / rate + over) / 3600)


def test_each_domain_pays_its_own_radiation_cadence():
    doc = te.estimate_tracks(_tracks(), cards=["a6000"])
    d01, d02, d03 = doc["tracks"][0]["domains"]
    extra = te.CALIBRATION["radiation_step_ratio"] - 1.0
    assert d01["radiation_factor"] == pytest.approx(1 + extra / 40)
    assert d02["radiation_factor"] == pytest.approx(1 + extra / 200)
    assert d03["radiation_factor"] == pytest.approx(1 + extra / 600)
    # The leaf's factor turns the ordinary rate into the measured mean.
    assert (te.CALIBRATION["cell_steps_per_s_ordinary"]
            / d03["radiation_factor"]) == pytest.approx(
        te.CALIBRATION["cell_steps_per_s"], rel=5e-3)


def test_track3d_documents_are_held_to_the_row_checks(tmp_path):
    from woof.track import load_track

    data = load_track(FIXTURE, _mapping()).to_dict()
    data["qc"] = []
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(te.EstimateRefusal):
        te.load_tracks_json(bad)


def test_memory_fit_uses_the_calibrated_line():
    doc = te.estimate_tracks(_tracks())
    r0 = doc["tracks"][0]
    peak = r0["cards"]["rtx-a6000"]["peak_gib"]
    assert peak == pytest.approx(
        te.CALIBRATION["fixed_gib"]
        + r0["cells"] * te.CALIBRATION["bytes_per_cell"] / 2**30)
    huge = te.estimate_tracks(
        [{"name": "x", "duration_h": 1, "envelope_km": [3000, 3000]}],
        cards=["rtx-a6000"])
    assert huge["totals"]["cards"]["rtx-a6000"][
        "tracks_that_do_not_fit"] == ["x"]


def test_a_small_margin_still_holds_the_moving_child():
    design = te.TrackDesign()
    doms = (te.replace(design.domains[0], margin_km=1.0),) + design.domains[1:]
    doc = te.estimate_tracks(
        [{"name": "a", "duration_h": 1, "envelope_km": [200.0, 0.0]}],
        te.replace(design, domains=doms), cards=["a6000"])
    d01 = doc["tracks"][0]["domains"][0]
    assert d01["nx"] * 1.5 >= 200 + 72.3 + 2 * 10 * 1.5


@pytest.mark.parametrize("change, match", [
    (dict(nz=1), "--nz"),
    (dict(output_interval_s=0.0), "--output-interval-s"),
    ("d03=-5", "--d03-n must be positive"),
    ("d03=800", "does not fit inside d02"),
])
def test_impossible_designs_are_refused(change, match):
    design = te.TrackDesign()
    if isinstance(change, dict):
        design = te.replace(design, **change)
    else:
        n = int(change.split("=")[1])
        doms = design.domains[:2] + (te.replace(design.domains[2], nx=n,
                                                ny=n),)
        design = te.replace(design, domains=doms)
    with pytest.raises(te.EstimateRefusal, match=match):
        te.estimate_tracks(_tracks(), design)


def test_cards_dedupe_and_spellings():
    doc = te.estimate_tracks(_tracks(), cards=["a6000", "NVIDIA RTX A6000 48GB",
                                               "rtx_a6000", "l40s"])
    assert list(doc["totals"]["cards"]) == ["rtx-a6000", "l40s-48gb"]


def test_price_table_override_and_unknown_card():
    prices = {"label": "test", "instances": {
        "l40s-48gb": {"instance": "x", "gpus": 2, "usd_per_hour": 10.0}},
        "s3_usd_per_gb_month": 0.0}
    doc = te.estimate_tracks(_tracks(), cards=["l40s-48gb"], prices=prices)
    c = doc["totals"]["cards"]["l40s-48gb"]
    assert c["aws"]["usd_per_gpu_hour"] == 5.0
    assert doc["totals"]["s3_usd_per_month"] == [0.0, 0.0]
    with pytest.raises(te.EstimateRefusal, match="not a card"):
        te.estimate_tracks(_tracks(), cards=["no-such-card"])
    with pytest.raises(te.EstimateRefusal, match="JSON object"):
        te.estimate_tracks(_tracks(), prices=[1, 2])
    with pytest.raises(te.EstimateRefusal, match="positive gpus"):
        te.estimate_tracks(_tracks(), cards=["l40s"], prices={
            "instances": {"l40s-48gb": {"gpus": 0, "usd_per_hour": 1}}})


@pytest.mark.parametrize("rows, match", [
    ([{"name": "a", "duration_h": 1}], "envelope_km"),
    ([{"name": "a", "duration_h": -1, "envelope_km": [1, 1]}], "positive"),
    ([{"name": "a", "duration_h": 1, "envelope_km": [1, 1]}] * 2, "unique"),
    ([], "no tracks"),
])
def test_bad_tracks_are_refused_by_name(rows, match):
    with pytest.raises(te.EstimateRefusal, match=match):
        te.estimate_tracks(rows)


FIXTURE = Path(__file__).parent / "fixtures" / "track" / "track_synthetic.csv"
FLAGS = ["--time-col", "timestamp", "--lat-col", "latitude", "--lon-col",
         "longitude", "--height-col", "alt_gps_m"]


def _mapping():
    from woof.track import ColumnMapping

    return ColumnMapping(time="timestamp", lat="latitude", lon="longitude",
                         height="alt_gps_m")


def test_path_files_are_read_by_woof_track(tmp_path):
    from woof.track import load_track

    (track,) = te.read_path_files([FIXTURE], _mapping())
    t3d = load_track(FIXTURE, _mapping())
    assert track.name == FIXTURE.stem
    assert track.duration_h == pytest.approx(t3d.duration_h)
    assert track.envelope_km == (t3d.extent_ew_km, t3d.extent_ns_km)
    assert track.max_height_m == t3d.max_height
    assert track.qc == t3d.qc
    # A Track3D document reads to the same track through --tracks.
    doc = tmp_path / "t.json"
    doc.write_text(t3d.to_json(), encoding="utf-8")
    assert te.load_tracks_json(doc) == [track]


def test_path_file_with_a_missing_column_is_refused():
    from woof.track import ColumnMapping

    with pytest.raises(te.EstimateRefusal, match="track_synthetic"):
        te.read_path_files([FIXTURE], ColumnMapping(time="no-such-column"))


def test_select_by_pattern():
    tracks = [te.Track(f"run-{i:02d}", 1.0, (1.0, 1.0)) for i in range(1, 21)]
    chosen = te.select_tracks(tracks, "run-1[1-9],run-20")
    assert [t.name for t in chosen] == [f"run-{i}" for i in range(11, 21)]
    with pytest.raises(te.EstimateRefusal, match="matches none"):
        te.select_tracks(tracks, "nothing")


def test_cli_door_writes_the_json(tmp_path, capsys):
    from woof.cli import build_parser

    src = tmp_path / "tracks.json"
    src.write_text(json.dumps({"tracks": _tracks()}), encoding="utf-8")
    out = tmp_path / "est.json"
    args = build_parser().parse_args(
        ["estimate-tracks", "--tracks", str(src), "--cards", "a6000,l40s",
         "--d03-n", "300", "--compression", "0.3,0.35", "--per-track",
         "-o", str(out)])
    assert args.func(args) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema"] == te.SCHEMA
    assert doc["design"]["compression"] == [0.3, 0.35]
    assert doc["tracks"][0]["domains"][2]["nx"] == 300
    assert set(doc["totals"]["cards"]) == {"rtx-a6000", "l40s-48gb"}
    err = capsys.readouterr().err
    assert "uncalibrated" in err and "verify before launch" in err


def test_cli_door_reads_path_files_by_column_flags(capsys):
    from woof.cli import build_parser

    args = build_parser().parse_args(
        ["estimate-tracks", "--paths", str(FIXTURE), *FLAGS,
         "--cards", "a6000"])
    assert args.func(args) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["tracks"] == [{"name": FIXTURE.stem}]
    assert doc["totals"]["max_height_m"] > 1600.0


@pytest.mark.parametrize("argv", [
    [],                                              # neither input
    ["--tracks", "missing.json"],                    # no such file
    ["--tracks", "."],                               # a directory
    ["--paths", "bad.csv"],                          # not UTF-8
])
def test_cli_door_refuses_with_exit_2(tmp_path, capsys, monkeypatch, argv):
    from woof.cli import build_parser

    monkeypatch.chdir(tmp_path)
    (tmp_path / "bad.csv").write_bytes(b"time,lat,lon\n\xff\xfe,1,2\n")
    args = build_parser().parse_args(["estimate-tracks", *argv])
    assert args.func(args) == 2
    assert "REFUSED" in capsys.readouterr().err
