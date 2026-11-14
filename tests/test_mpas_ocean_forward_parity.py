"""MPAS-Ocean's forward mode over N steps, word for word against v8.4.2 at RKIND 4 and 8, with a bitwise restart.

The reference is MPAS-Model v8.4.2's forward mode as
``tools/mpas_ocean_oracle/run_forward.F90`` drives it, compiled with gfortran
-O0 -ffp-contract=off at RKIND 4 and 8:
- ``ocn_time_integration_split_init`` and ``ocn_timestep``, cut verbatim;
- the state part of ``ocn_init_routines_block``;
- per step: the absorbed fractions, the step and the time-level shift.
Every module the step reaches is compiled verbatim. The meshes are
:mod:`woof.verify.mpas_ocean_hexmesh`'s culled hexagonal tori, whose
consistent topology keeps MPAS's sentinels where a real mesh keeps them. The
outputs are committed (tests/data/oracles/mpas_ocean_forward, PROVENANCE.md).
When ``GFORTRAN`` (or gfortran on PATH) and ``WOOF_MPAS_V842_SOURCE`` are set,
the fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A forward driver that drifts from MPAS over steps: the time levels, the
   accumulated coupler means, the subcycle count, the cold start.
2. A restart that does not continue the run bitwise.
3. A clock that disagrees with MPAS's calendar.
4. A gate that cannot fail: a mutated cold start, absorbed fraction, subcycle
   count or level shift must produce mismatches.
"""
from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.hex.errors import ConfigurationRefusal
from woof.verify import mpas_ocean_forward_oracle as oo

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": oo.input_digest(case) for p in BOTH for case in oo.cases(p)}


def test_the_clock_is_mpas_calendar():
    from woof.hex.mpas_ocean import forward_v842 as fw
    epoch = datetime.datetime(1, 1, 1)
    for text in ("0001-01-01_00:00:00", "2000-02-28_23:58:00", "2000-02-29_12:00:01", "1900-03-01_00:00:00",
                 "2024-12-31_23:59:59", "9999-12-31_23:59:59"):
        seconds = fw.parse_time(text)
        assert seconds == int((datetime.datetime.strptime(text, "%Y-%m-%d_%H:%M:%S") - epoch).total_seconds())
        assert fw.format_time(seconds) == text
    assert fw.interval_seconds("0000_00:00:50") == 50 and fw.interval_seconds("01:02:03") == 3723
    assert fw.btr_subcycle_count("00:02:00", "0000_00:00:50") == 3
    assert fw.btr_subcycle_count("00:03:00", "0000_00:01:00") == 3


@pytest.mark.parametrize("precision", BOTH)
def test_fixture_covers_the_branches(precision):
    fixture = oo.load_fixture(precision)
    cases = {case["name"]: case for case in oo.cases(precision)}
    assert set(fixture) == set(cases)
    for name, out in fixture.items():
        case = cases[name]
        nc = case["dims"]["nCells"]
        assert case["config"]["oracle_n_steps"] >= 2
        days = float(out["diagnostics/daysSinceStartOfSim"])
        from woof.hex.mpas_ocean import forward_v842 as fw
        elapsed = (fw.parse_time(case["config"]["config_start_time"]) - fw.parse_time(
            case["config"]["diagnostics:simulationStartTime"]) + case["config"]["oracle_n_steps"] * case["dt"])
        assert abs(days - elapsed / 86400.0) < 1e-5 * max(1.0, days)
        assert np.isfinite(out["state/ssh"][:nc]).all() and np.isfinite(out["state/ssh@2"][:nc]).all()
    assert any(case["config"]["oracle_n_steps"] % 2 for case in cases.values())
    assert any(case["config"]["oracle_n_steps"] % 2 == 0 for case in cases.values())


def test_rk4_and_mpas_restarts_are_refused_by_name():
    from woof.hex.mpas_ocean import forward_v842 as fw
    config = oo.cases()[0]["config"]
    fw.check_config(config)
    for knob, value in (("config_time_integrator", "RK4"), ("config_do_restart", True)):
        with pytest.raises(ConfigurationRefusal, match=knob):
            fw.check_config(dict(config, **{knob: value}))


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word_over_the_steps(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_a_restart_continues_the_run_bitwise(tmp_path, precision):
    """k steps, a restart file, a fresh read, the rest: the oracle's words."""
    import cupy as cp
    from woof.hex.mpas_ocean import forward_v842 as fw
    from woof.verify import mpas_ocean_stream as stream
    case = next(c for c in oo.cases(precision) if c["config"]["oracle_n_steps"] >= 3)
    n = int(case["config"]["oracle_n_steps"])
    dev = stream.on_device(oo.FIELDS, case)
    clock = fw.cold_start(pools=dev, configs=case["config"])
    fw.advance(pools=dev, configs=case["config"], clock=clock, n_steps=1)
    fw.write_restart(tmp_path / "restart.npz", pools=dev, clock=clock)
    pools, clock = fw.read_restart(tmp_path / "restart.npz", like=dev)
    del dev
    assert clock.now - clock.start == clock.step and pools["diagnostics"]["xtime"] == clock.time_stamp
    fw.advance(pools=pools, configs=case["config"], clock=clock, n_steps=n - 1)
    want = oo.load_fixture(precision)[case["name"]]
    two_level = {(p, name) for p, names in fw.TWO_LEVEL.items() for name in names}
    bad = {}
    for key, words in want.items():
        pool, name = key.split("/", 1)
        base = name.split("@")[0]
        current = name
        if (pool, base) in two_level and n % 2:
            # the oracle writes the arrays it was given: after an odd number of
            # shifts NAME holds what is now NAME@2
            current = f"{base}@2" if name == base else base
        mismatches = oo.word_mismatches(cp.asnumpy(pools[pool][current]), words)
        if mismatches:
            bad[key] = mismatches
    assert not bad, bad


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
@pytest.mark.parametrize("original, mutated, field", [
    # the cold start's barotropic mean by a reciprocal
    ("btr[e] = D(flux, thick);", "btr[e] = M(flux, D(C(1.0), thick));", "state/normalBarotropicVelocity"),
])
def test_a_mutated_cold_start_is_caught(monkeypatch, precision, original, mutated, field):
    from woof.hex.mpas_ocean import forward_v842
    real = forward_v842.module_source
    assert real(precision).count(original) == 1
    monkeypatch.setattr(forward_v842, "module_source", lambda p="r4": real(p).replace(original, mutated))
    forward_v842._module_on.cache_clear()
    try:
        case = oo.cases(precision)[0]
        got = oo.run_gpu(case)
        assert oo.word_mismatches(got[field], oo.load_fixture(precision)[case["name"]][field]) > 0
    finally:
        forward_v842._module_on.cache_clear()


@requires_gpu
def test_a_mutated_absorbed_fraction_is_caught(monkeypatch):
    from woof.hex.mpas_ocean import forcing_v842
    real = forcing_v842.module_source
    original = "fraction[iCell * nl + k] = S(tTop, tBot);"
    assert real("r4").count(original) == 1
    monkeypatch.setattr(forcing_v842, "module_source",
                        lambda p="r4": real(p).replace(original, "fraction[iCell * nl + k] = S(tBot, tTop);"))
    forcing_v842._module_on.cache_clear()
    try:
        case = oo.cases("r4")[0]
        got = oo.run_gpu(case)
        assert oo.word_mismatches(got["forcing/fractionAbsorbed"],
                                  oo.load_fixture("r4")[case["name"]]["forcing/fractionAbsorbed"]) > 0
    finally:
        forcing_v842._module_on.cache_clear()


@requires_gpu
@pytest.mark.parametrize("mutation", ["subcycles_without_remainder", "no_level_shift", "no_coupler_count"])
def test_a_mutated_driver_is_caught(monkeypatch, mutation):
    from woof.hex.mpas_ocean import forward_v842 as fw
    from woof.hex.mpas_ocean import split_step_v842 as ss
    if mutation == "subcycles_without_remainder":
        monkeypatch.setattr(fw, "btr_subcycle_count",
                            lambda dt, btr: fw.interval_seconds(dt) // fw.interval_seconds(btr))
        field = "state/ssh"
    elif mutation == "no_level_shift":
        monkeypatch.setattr(fw, "_shift", lambda pools: None)
        field = "state/ssh"
    else:
        real_step = ss.split_step

        def step_without_count(**kw):
            count = kw["pools"]["forcing"]["nAccumulatedCoupled"]
            real_step(**kw)
            kw["pools"]["forcing"]["nAccumulatedCoupled"] = count
        monkeypatch.setattr(ss, "split_step", step_without_count)
        field = "forcing/avgSSHGradient"
    case = oo.cases("r8")[0]
    got = oo.run_gpu(case)
    assert oo.word_mismatches(got[field], oo.load_fixture("r8")[case["name"]][field]) > 0


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_ocean").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MPAS-Ocean oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_ocean_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_ocean_oracle" / "make_fixture.py"),
                    str(tmp_path / "build"), "--piece", "forward", "--out", str(tmp_path / "fixture")],
                   check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
