"""``woof estimate``: how long a run takes on a runner.

CPU-only.  No card is consulted: every runner is a table row, a
``[runner]`` mapping, or detection fed canned ``nvidia-smi`` text, and
the calibration run is a fake launcher on a fake clock.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import textwrap

import pytest

from woof import runtime_calibration as calib
from woof import runtime_estimate as rte
from woof.runner_cards import (CUDA, GIB, METAL, RUNNER_CARDS,
                               RunnerSpecError, card_for, detect_local_runner,
                               load_runner_toml, parse_nvidia_smi,
                               runner_from_card, runner_from_mapping)


ETA = ", ".join(
    str(v) for v in (
        1.0, 0.9978, 0.99519, 0.99212, 0.98849, 0.98422, 0.97918, 0.97325,
        0.96627, 0.95808, 0.94846, 0.93719, 0.92402, 0.90866, 0.89079,
        0.87006, 0.84612, 0.81857, 0.78706, 0.75124, 0.7108, 0.66556,
        0.61547, 0.56067, 0.50519, 0.45474, 0.40886, 0.36713, 0.32918,
        0.29466, 0.26328, 0.23473, 0.20877, 0.18516, 0.16369, 0.14417,
        0.12641, 0.11026, 0.09557, 0.08222, 0.07007, 0.05902, 0.04898,
        0.03984, 0.03153, 0.02398, 0.0171, 0.01085, 0.00517, 0.0))


def _config(tmp_path, *, nx=200, ny=160, dt=15, radt=12.0, hours=6,
            adaptive=False, history=3600.0, restart=0.0, name="est"):
    text = textwrap.dedent(f"""\
        [experiment]
        name = "{name}"
        start_time = 2026-01-01T00:00:00
        run_seconds = {hours * 3600.0}
        restart_interval_s = {restart}

        [projection]
        map_proj = "lambert"
        ref_lat = 40.0
        ref_lon = -100.0
        truelat1 = 30.0
        truelat2 = 50.0
        stand_lon = -100.0

        [shared]
        nz = 49
        ztop = 20000.0
        p_top = 5000.0
        eta_levels = [{ETA}]
        moist = true
        mp_physics = 10
        bl_pbl_physics = 1
        sf_sfclay_physics = 91
        sf_surface_physics = 2
        ra_lw_physics = 4
        ra_sw_physics = 4
        ra_rrtmg_variant = "rte-rrtmgp"
        wrf_rrtmg_compatibility = "wrf-rrtmg-4-4-to-rte-rrtmgp-v2"
        num_soil_layers = 4
        use_adaptive_time_step = {str(adaptive).lower()}

        [[domain]]
        grid_id = 1
        parent_id = 0
        i_parent_start = 1
        j_parent_start = 1
        parent_grid_ratio = 1
        parent_time_step_ratio = 1
        nx = {nx}
        ny = {ny}
        time_step = {dt}
        dx = 3000.0
        specified = true
        nested = false
        history_interval_s = {history}
        radt = {radt}
        cu_physics = 0
        cudt_minutes = 0.0
        """)
    path = tmp_path / f"{name}.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def config(tmp_path):
    return _config(tmp_path)


@pytest.fixture
def exp(config):
    from woof.experiment import load_experiment

    return load_experiment(config)


@pytest.fixture(autouse=True)
def _no_local_calibration(tmp_path, monkeypatch):
    monkeypatch.setenv("WOOF_RUNTIME_CALIBRATION",
                       str(tmp_path / "calibration.json"))


def _with_run(exp, **changes):
    """``exp`` with the root's RunConfig fields replaced (no validation:
    the estimator reads fields, it does not re-check the config)."""
    run = copy.copy(exp.domains[0].run)
    for key, value in changes.items():
        object.__setattr__(run, key, value)
    domain = copy.copy(exp.domains[0])
    object.__setattr__(domain, "run", run)
    out = copy.copy(exp)
    object.__setattr__(out, "domains", [domain] + list(exp.domains[1:]))
    return out


def _a6000():
    return runner_from_card(RUNNER_CARDS["rtx-a6000"])


# ---------------------------------------------------------------- card table
def test_every_card_row_cites_a_source_and_has_bandwidth():
    for card in RUNNER_CARDS.values():
        assert card.source.strip(), card.key
        assert card.bandwidth_gbs > 0, card.key
        assert card.backend in (CUDA, METAL)


def test_the_asked_cards_are_in_the_table():
    for name in ("RTX A6000", "A100 80GB", "H100 SXM", "L4", "L40S",
                 "RTX 3090", "RTX 4090", "RTX 5090", "M3 Max"):
        assert card_for(name) is not None, name


@pytest.mark.parametrize("spelling,key", [
    ("NVIDIA RTX A6000", "rtx-a6000"),
    ("a6000", "rtx-a6000"),
    ("NVIDIA GeForce RTX 4090", "rtx-4090"),
    ("NVIDIA A100-SXM4-80GB", "a100-80gb"),
    ("Apple M3 Max", "m3-max"),
])
def test_card_spellings_resolve(spelling, key):
    assert card_for(spelling).key == key


def test_an_unknown_card_is_none():
    assert card_for("Voodoo 5 6000") is None
    assert card_for("") is None


# ---------------------------------------------------------------- detection
SMI = "NVIDIA RTX A6000, 49140, 37000, 4, 16\n"


def test_nvidia_smi_rows_parse():
    rows = parse_nvidia_smi(SMI + "NVIDIA RTX A6000, 49140, 49000, 4, 16\n")
    assert len(rows) == 2
    assert rows[0] == {"name": "NVIDIA RTX A6000", "memory_mib": 49140,
                       "free_mib": 37000, "pcie_gen": 4, "pcie_width": 16}


def test_detection_from_nvidia_smi_layers_the_table_row():
    runner = detect_local_runner(smi_text=SMI, platform="linux")
    assert runner.card_key == "rtx-a6000"
    assert runner.bandwidth_gbs == 768
    assert runner.vram_bytes == 49140 << 20
    assert runner.free_vram_bytes == 37000 << 20
    assert runner.pcie == "PCIe 4.0 x16"
    assert runner.provenance["vram_bytes"].startswith("detected")


def test_detection_of_an_unlisted_card_states_no_bandwidth():
    runner = detect_local_runner(smi_text="Mystery GPU, 8192, 8000, 3, 8\n",
                                 platform="linux")
    assert runner.card_key is None and runner.bandwidth_gbs is None
    assert "unknown" in runner.provenance["bandwidth_gbs"]


def test_detection_of_an_apple_chip_without_system_profiler():
    runner = detect_local_runner(smi_text="", apple_chip="Apple M3 Max",
                                 platform="darwin")
    assert runner.backend == METAL and runner.card_key == "m3-max"
    assert runner.vram_bytes == runner.host_ram_bytes


def test_no_gpu_is_none():
    assert detect_local_runner(smi_text="", platform="linux") is None


# ---------------------------------------------------------------- TOML
def test_runner_toml_overrides_the_card(tmp_path):
    path = tmp_path / "runner.toml"
    path.write_text('[runner]\ncard = "a6000"\nbandwidth_gbs = 700\n'
                    'cpu_cores = 12\nslots_per_gpu = "auto"\nmps = true\n')
    runner = load_runner_toml(path)
    assert runner.card_key == "rtx-a6000"
    assert runner.bandwidth_gbs == 700.0
    assert runner.cpu_cores == 12 and runner.mps is True
    assert runner.slots_per_gpu == "auto"
    assert runner.provenance["bandwidth_gbs"].startswith("toml")


def test_runner_toml_refuses_unknown_keys_and_bad_values():
    with pytest.raises(RunnerSpecError, match="does not read"):
        runner_from_mapping({"bandwith": 1}, base=None)
    with pytest.raises(RunnerSpecError, match="positive"):
        runner_from_mapping({"bandwidth_gbs": 0}, base=None)
    with pytest.raises(RunnerSpecError, match="names no row"):
        runner_from_mapping({"card": "voodoo"}, base=None)
    with pytest.raises(RunnerSpecError, match="backend"):
        runner_from_mapping({"backend": "rocm"}, base=None)


# ---------------------------------------------------------------- scaling
def test_bandwidth_factor_brackets_the_exponents():
    fast, slow = rte.bandwidth_factor(1792, 768)
    ratio = 1792 / 768
    assert fast == pytest.approx(ratio ** rte.BANDWIDTH_EXPONENT[0])
    assert slow == pytest.approx(ratio ** rte.BANDWIDTH_EXPONENT[1])
    assert rte.bandwidth_factor(768, 768) == (1.0, 1.0)


def test_same_card_anchor_is_measured_and_other_cards_are_scaled():
    a6000 = _a6000()
    bracket = rte.rate_bracket(a6000, rte.WRF, "full", columns=18_000)
    assert bracket.provenance == rte.MEASURED
    assert {c.anchor.card for c in bracket.candidates} == {"rtx-a6000"}
    l4 = runner_from_card(RUNNER_CARDS["l4"])
    scaled = rte.rate_bracket(l4, rte.WRF, "full", columns=18_000)
    assert scaled.provenance == rte.SCALED
    assert any("SCALED" in b.upper() for b in scaled.basis)


def test_a_slower_card_is_slower(exp):
    fast = rte.estimate(exp, runner_from_card(RUNNER_CARDS["h100-sxm"]))
    slow = rte.estimate(exp, runner_from_card(RUNNER_CARDS["l4"]))
    assert slow.low_s > fast.low_s and slow.high_s > fast.high_s


def test_a_card_with_no_bandwidth_is_a_named_substitute(exp):
    runner = runner_from_mapping({"name": "mystery", "vram_gib": 24},
                                 base=None)
    est = rte.estimate(exp, runner)
    assert est.rate.provenance == rte.SUBSTITUTED
    assert any("SUBSTITUTE" in b for b in est.rate.basis)


def test_metal_is_priced_through_the_measured_a6000_ratio(exp):
    m3 = runner_from_card(RUNNER_CARDS["m3-max"], host_ram_bytes=64 * GIB)
    est = rte.estimate(exp, m3)
    a6 = rte.estimate(exp, _a6000())
    steps = [p for p in est.phases if p.name == "model steps"][0]
    a6_steps = [p for p in a6.phases if p.name == "model steps"][0]
    assert steps.provenance == rte.SCALED
    assert steps.high_s > a6_steps.high_s


def test_a_rung_without_anchors_substitutes_the_slowest():
    rows, premium, note = rte._rung_anchors(rte.HEX, "dry", rte.ANCHORS)
    assert premium == 1.0 and "SUBSTITUTE" in note
    rows, premium, note = rte._rung_anchors(rte.WRF, "full+mynn+noahmp",
                                            rte.ANCHORS)
    assert premium > 1.0 and all(a.rung == "full" for a in rows)


# ---------------------------------------------------------------- config
def _steps(est):
    return sum(p.high_s for p in est.phases
               if p.name in ("model steps", "radiation steps"))


def test_cost_scales_with_levels(exp):
    base = rte.estimate(exp, _a6000())
    deep = rte.estimate(_with_run(exp, nz=98), _a6000())
    assert _steps(deep) == pytest.approx(2 * _steps(base), rel=1e-9)


def test_cost_scales_with_steps(exp):
    base = rte.estimate(exp, _a6000())
    halved = rte.estimate(_with_run(exp, dt=7.5), _a6000())

    def phase(e, name):
        return [p for p in e.phases if p.name == name][0].high_s

    # Twice the steps; the same number of radiation calls (radt is a
    # wall of model time, not a step count).
    assert halved.domains[0].steps == 2 * base.domains[0].steps
    assert halved.domains[0].as_dict()["radiation_steps"] == \
        base.domains[0].as_dict()["radiation_steps"]
    assert phase(halved, "model steps") >= 1.8 * phase(base, "model steps")
    assert halved.seconds_per_step[0] < base.seconds_per_step[0]


def test_radiation_cadence_moves_the_radiation_phase(exp):
    def rad(e):
        return [p for p in e.phases if p.name == "radiation steps"][0]

    every12 = rte.estimate(exp, _a6000())
    every4 = rte.estimate(_with_run(exp, radt=4.0, radt_minutes=4.0),
                          _a6000())
    assert rad(every4).high_s > 2.5 * rad(every12).high_s
    assert _steps(every4) > _steps(every12)
    assert every4.domains[0].radiation_fraction == pytest.approx(
        3 * every12.domains[0].radiation_fraction)


def test_output_frequency_moves_the_output_phase(tmp_path):
    from woof.experiment import load_experiment

    hourly = rte.estimate(load_experiment(_config(tmp_path, name="h")),
                          _a6000())
    quarter = rte.estimate(load_experiment(
        _config(tmp_path, history=900.0, name="q")), _a6000())

    def out(e):
        return [p for p in e.phases if p.name == "output"][0].high_s

    assert out(quarter) > 3.5 * out(hourly)


def test_an_adaptive_clock_lowers_the_low_end(tmp_path):
    from woof.experiment import load_experiment

    fixed = rte.estimate(load_experiment(_config(tmp_path, name="f")),
                         _a6000())
    adaptive = rte.estimate(load_experiment(
        _config(tmp_path, adaptive=True, name="a")), _a6000())
    assert _steps(adaptive) == pytest.approx(_steps(fixed))
    lo = [p for p in adaptive.phases if p.name == "model steps"][0].low_s
    lo_fixed = [p for p in fixed.phases if p.name == "model steps"][0].low_s
    assert lo == pytest.approx(lo_fixed * rte.ADAPTIVE_WALL_RATIO)


def test_nests_add_their_steps(exp):
    root = exp.domains[0]
    nest_run = copy.copy(root.run)
    for key, value in (("nx", 100), ("ny", 100), ("dt", 5.0)):
        object.__setattr__(nest_run, key, value)
    nest = copy.copy(root)
    object.__setattr__(nest, "run", nest_run)
    object.__setattr__(nest, "grid_id", 2)
    tree = copy.copy(exp)
    object.__setattr__(tree, "domains", [root, nest])
    one = rte.estimate(exp, _a6000())
    two = rte.estimate(tree, _a6000())
    assert len(two.domains) == 2
    assert two.domains[1].steps == 3 * one.domains[0].steps
    assert _steps(two) > _steps(one)


def test_hex_route_prices_from_hex_anchors_and_says_prep_is_unpriced(exp):
    est = rte.estimate(exp, _a6000(), route=rte.HEX)
    assert {c.anchor.route for c in est.rate.candidates} == {rte.HEX}
    prep = [p for p in est.phases if p.name == "prepare"][0]
    assert prep.provenance == rte.SUBSTITUTED and prep.high_s == 0.0


def test_da_phases(exp):
    var = rte.estimate(exp, _a6000(), da="3dvar", da_cycles=2)
    phase = [p for p in var.phases if p.name == "data assimilation"][0]
    assert phase.low_s == pytest.approx(2 * 2.3)
    letkf = rte.estimate(exp, _a6000(), da="letkf", members=20)
    phase = [p for p in letkf.phases if p.name == "data assimilation"][0]
    assert phase.low_s == pytest.approx(10.1)
    with pytest.raises(ValueError):
        rte.estimate(exp, _a6000(), da="4dvar")


def test_prepared_and_warm_cache_drop_their_phases(exp):
    est = rte.estimate(exp, _a6000(), prepared=True, kernel_cache_cold=False)
    for name in ("prepare", "kernel compile"):
        assert [p for p in est.phases if p.name == name][0].high_s == 0.0


@pytest.mark.parametrize("dt,columns", [(15.0, 32_000), (7.5, 162_000)])
def test_total_is_the_sum_of_phases_and_low_never_exceeds_high(exp, dt,
                                                               columns):
    est = rte.estimate(_with_run(exp, dt=dt, nx=450, ny=columns // 450),
                       _a6000(), first_run_on_box=True,
                       kernel_cache_cold=True)
    assert est.low_s == pytest.approx(sum(p.low_s for p in est.phases))
    for phase in est.phases:
        assert 0.0 <= phase.low_s <= phase.high_s, phase.name


# ---------------------------------------------------------------- sharing
def test_sharing_measured_row_and_unmeasured_extrapolation():
    two = rte.sharing_bracket(2, columns=1_000_000, mps=False)
    assert two["measured"] is True
    assert two["per_run_slowdown"][0] == pytest.approx(22.6 / 11.66)
    three = rte.sharing_bracket(3, columns=1_000_000, mps=False)
    assert three["measured"] is False and "UNMEASURED" in three["basis"]
    assert three["per_run_slowdown"][1] == 3.0
    assert rte.sharing_bracket(1, columns=10, mps=False)[
        "per_run_slowdown"] == (1.0, 1.0)


def test_admission_caps_copies_by_priced_memory(exp, monkeypatch):
    import woof.supervisor as supervisor

    monkeypatch.setattr(supervisor, "priced_reservation_bytes",
                        lambda e: 10 * GIB)
    copies, detail = rte.admitted_copies(exp, _a6000(), "auto")
    assert copies == 4  # (48 GiB - 1 GiB margin) // 10 GiB
    copies, _ = rte.admitted_copies(exp, _a6000(), 2)
    assert copies == 2
    monkeypatch.setattr(supervisor, "priced_reservation_bytes",
                        lambda e: None)
    copies, detail = rte.admitted_copies(exp, _a6000(), "auto")
    assert copies == 1 and "runs alone" in detail["reason"]


# ---------------------------------------------------------------- CLI
def _parse(*argv):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    rte.register_cli(sub)
    return parser.parse_args(["estimate", *argv])


def test_cli_json(config, capsys):
    args = _parse(str(config), "--card", "a6000", "--json")
    assert args.func(args) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == rte.SCHEMA
    assert doc["runner"]["card_key"] == "rtx-a6000"
    assert doc["wall_seconds_low"] <= doc["wall_seconds_high"]
    names = [p["name"] for p in doc["phases"]]
    for name in ("prepare", "kernel compile", "model steps",
                 "radiation steps", "output"):
        assert name in names
    assert doc["domains"][0]["steps"] == 6 * 3600 // 15


def test_cli_text(config, capsys):
    args = _parse(str(config), "--gpu", "RTX 4090")
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "NVIDIA GeForce RTX 4090" in out or "RTX 4090" in out
    assert "total" in out and "model steps" in out and "basis:" in out


def test_cli_list_cards(capsys):
    args = _parse("--list-cards")
    assert args.func(args) == 0
    assert "rtx-a6000" in capsys.readouterr().out


def test_cli_refuses_an_unknown_card(config, capsys):
    args = _parse(str(config), "--card", "voodoo")
    assert args.func(args) == 2
    assert "names no row" in capsys.readouterr().err


def test_cli_runner_toml(config, tmp_path, capsys):
    spec = tmp_path / "r.toml"
    spec.write_text('[runner]\ncard = "l4"\n')
    args = _parse(str(config), "--card", "a6000", "--runner", str(spec),
                  "--json")
    assert args.func(args) == 0
    assert json.loads(capsys.readouterr().out)["runner"]["card_key"] == "l4"


def test_cli_copies_report_sharing(config, capsys, monkeypatch):
    import woof.supervisor as supervisor

    monkeypatch.setattr(supervisor, "priced_reservation_bytes",
                        lambda e: 4 * GIB)
    args = _parse(str(config), "--card", "a6000", "--copies", "2", "--json")
    assert args.func(args) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["sharing"]["copies"] == 2
    assert doc["admission"]["by_memory"] >= 2


# ---------------------------------------------------------------- calibration
def _records(n, *, ordinary_ms=100, radiation_every=48, radiation_ms=400,
             dt=15.0):
    out, t = [], 1_000_000
    for step in range(1, n + 1):
        t += radiation_ms if step % radiation_every == 1 else ordinary_ms
        out.append({"event": "step", "domain": 1, "step": step,
                    "model_seconds": step * dt, "emitted_unix_ms": t})
    return out


def test_step_walls_drop_warmup_and_divide_thinned_gaps():
    recs = _records(20)
    walls = calib.step_walls(recs, warmup=5)
    assert len(walls) == 15
    thinned = recs[::2]
    walls = calib.step_walls(thinned, warmup=0)
    assert all(w > 0 for w in walls)


def test_extrapolation_from_whole_radiation_periods():
    walls = [0.1] * 47 + [0.4]
    s = calib.summarize_steps(walls * 2, steps_per_radiation=48)
    central, lo, hi = calib.extrapolate(s, steps=1440, radiation_fraction=1 / 48,
                                        ratio_bracket=(3.4, 9.99))
    assert central == lo == hi == pytest.approx(1440 * (47 * 0.1 + 0.4) / 48)


def test_extrapolation_with_a_seen_ratio_and_without():
    s = calib.summarize_steps([0.1] * 10 + [0.4], steps_per_radiation=48)
    assert s.radiation_ratio == pytest.approx(4.0)
    central, lo, hi = calib.extrapolate(s, steps=480, radiation_fraction=0.1,
                                        ratio_bracket=(3.4, 9.99))
    assert lo == hi == pytest.approx(0.1 * 480 * (1 + 3.0 * 0.1))
    s = calib.summarize_steps([0.1] * 10, steps_per_radiation=48)
    central, lo, hi = calib.extrapolate(s, steps=480, radiation_fraction=0.1,
                                        ratio_bracket=(3.4, 9.99))
    assert lo < central < hi


def test_calibrate_with_a_fake_run_reports_and_records(config, exp, tmp_path):
    est = rte.estimate(exp, _a6000(), config=str(config))

    def fake_measure(cfg, **kw):
        return {"command": ["woof", "go", str(cfg)], "records": _records(60),
                "reason": "steps measured", "launch_to_first_step_s": 42.0,
                "log": "go.log"}

    store = tmp_path / "cal.json"
    report = calib.calibrate(config, steps=50, warmup=5, estimate=est,
                             runner=_a6000(), workdir=tmp_path / "w",
                             path=store, measure_fn=fake_measure)
    assert report["measured_steps"] == 55
    assert report["launch_to_first_step_s"] == 42.0
    assert report["measured_seconds_per_step"] == pytest.approx(0.1)
    expected_ordinary = 0.1 / (200 * 160)
    assert report["ordinary_s_per_column_step_nz49"] == pytest.approx(
        expected_ordinary)
    row = calib.matching_calibration(_a6000(), route=rte.WRF, config=config,
                                     path=store)
    assert row is not None and row["runner_key"] == "rtx-a6000"
    # A recorded calibration prices the next estimate, labelled so.
    priced = rte.estimate(exp, _a6000(), calibration=row)
    assert priced.rate.provenance == rte.CALIBRATED
    steps = [p for p in priced.phases if p.name == "model steps"][0]
    assert steps.low_s == pytest.approx(0.1 * 1440)


def test_calibrate_with_no_settled_steps_says_why(config, exp, tmp_path):
    est = rte.estimate(exp, _a6000(), config=str(config))
    with pytest.raises(ValueError, match="exited"):
        calib.calibrate(config, steps=10, warmup=5, estimate=est,
                        runner=_a6000(), workdir=tmp_path / "w",
                        path=tmp_path / "c.json", measure_fn=lambda c, **k: {
                            "command": ["x"], "records": _records(3),
                            "reason": "the run exited (1) first",
                            "log": "go.log"})


def test_measure_stops_after_enough_steps(tmp_path):
    """The real ``measure`` loop on a fake process and clock."""
    run = tmp_path / "w" / "run" / "x"
    stopped = []

    class FakeProcess:
        pid = 999_999_999
        returncode = None

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    def launch(command, **kw):
        run.mkdir(parents=True)
        lines = "\n".join(json.dumps(r) for r in _records(12))
        (run / "progress.jsonl").write_text(lines + "\n")
        return FakeProcess()

    ticks = iter(range(1000))
    import woof.runtime_calibration as module

    original = module.stop_process_tree
    module.stop_process_tree = lambda p, **k: stopped.append(p)
    try:
        out = calib.measure(tmp_path / "c.toml", steps=5, warmup=5,
                            workdir=tmp_path / "w", launch=launch,
                            clock=lambda: float(next(ticks)),
                            sleep=lambda s: None)
    finally:
        module.stop_process_tree = original
    assert out["reason"] == "steps measured"
    assert len(out["records"]) == 12 and len(stopped) == 1
    assert "--products" in out["command"]


# ---------------------------------------------------------------- live ETA
def test_eta_seconds():
    assert calib.eta_seconds(model_seconds=3600, run_seconds=7200,
                             anchor_wall=0, anchor_model=0,
                             now_wall=600) == pytest.approx(600)
    assert calib.eta_seconds(model_seconds=0, run_seconds=7200,
                             anchor_wall=0, anchor_model=0,
                             now_wall=10) is None


def test_live_eta_anchors_then_extrapolates():
    eta = calib.LiveEta(21600)
    assert eta.note({"model_elapsed_seconds": 0}, 0.0) == ""
    assert eta.note({"model_elapsed_seconds": 600}, 10.0) == ""
    note = eta.note({"model_elapsed_seconds": 1200}, 20.0)
    # 600 model s per 10 wall s; 20400 model s left -> 340 s.
    assert "ETA ~5.7 min" in note and "6% of the forecast" in note
    assert calib.LiveEta(None).note({"model_elapsed_seconds": 5}, 1) == ""
    assert calib.LiveEta(100).note(None, 1) == ""


def test_go_heartbeat_carries_the_eta(tmp_path, capsys):
    from woof import go_cli

    progress = tmp_path / "progress.json"
    script = tmp_path / "stage.py"
    script.write_text(textwrap.dedent(f"""\
        import json, time
        p = {str(progress)!r}
        for model in (600, 1200, 1800, 2400):
            json.dump({{"status": "INTEGRATING",
                       "model_elapsed_seconds": model}}, open(p, "w"))
            time.sleep(0.25)
        """))
    import sys

    go_cli._run_stage("forecast", [sys.executable, str(script)],
                      explain=False, progress=progress,
                      heartbeat_seconds=0.2, run_seconds=3600.0)
    out = capsys.readouterr().out
    assert "ETA ~" in out


def test_fmt_duration():
    assert rte.fmt_duration(30) == "30 s"
    assert rte.fmt_duration(90) == "1.5 min"
    assert rte.fmt_duration(7200) == "2.0 h"
    assert not math.isnan(rte.ANCHORS[0].rate)


def test_anchor_rows_name_card_contention_and_source():
    for anchor in rte.ANCHORS:
        assert anchor.card in RUNNER_CARDS
        assert anchor.contention and anchor.source
        assert anchor.s_per_step > 0


def test_estimate_json_is_serialisable(exp):
    doc = rte.estimate(exp, _a6000()).to_json()
    json.dumps(doc)
    assert doc["provenance"] in (rte.MEASURED, rte.SCALED, rte.SUBSTITUTED,
                                 rte.CALIBRATED)

