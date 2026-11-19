"""Stock WRF go routing preserves clocks, checkpoints and ensemble admission.

Through go_cli._go_wrfinput (the compatibility entry for the shared
stock-directory door, woof.wrfinput_go.launch), with go's own parser, so a
flag the door does not consume is refused rather than dropped.  The input
resolution (real.exe files on disk) is stood in; everything after it is the
door's own code up to run_wrf_forecast.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.go_cli import GoRefusal, _add_go_arguments, _go_wrfinput


def args(*extra):
    parser = argparse.ArgumentParser()
    _add_go_arguments(parser)
    return parser.parse_args(["--wrfinput", "native", "--outdir", "result",
                              "--run-seconds", "10800", "--restart", "resume.nc",
                              "--benchmark", "--allow-shared-gpu",
                              "--products", "none",
                              "--checkpoint-at", "2024-05-25_19:00:00",
                              "--run-stamp", "off", *extra])


@pytest.fixture
def runner(monkeypatch):
    import woof.wrfinput_forecast as worker
    from woof import wrfinput_go
    received = {}

    def run(source, out, **kw):
        received.update(source=source, out=out, **kw)
        return 0
    monkeypatch.setattr(worker, "run_wrf_forecast", run)
    monkeypatch.setattr(wrfinput_go, "_resolve_inputs", lambda directory, a: SimpleNamespace(
        experiment=SimpleNamespace(start_time=None)))
    return received


def test_stock_go_preserves_run_arguments(runner):
    assert _go_wrfinput(args()) == 0
    assert runner["source"] == Path("native") and runner["out"] == Path("result")
    assert runner["run_seconds"] == 10800
    assert runner["restart"] == Path("resume.nc")
    assert runner["benchmark"] is True
    assert runner["allow_shared_gpu"] is True
    assert runner["render_products"] == "none"
    assert runner["checkpoint_at"] == "2024-05-25_19:00:00"
    assert "soil_source" not in runner or runner["soil_source"] is None


def test_native_one_trajectory_cannot_duplicate_ensemble_members(runner):
    with pytest.raises(ValueError, match="members|member"):
        _go_wrfinput(args("--members", "2"))
    assert not runner


def test_native_restart_is_forwarded(runner):
    _go_wrfinput(args("--restart-from", "wrfrst_d01"))
    assert runner["restart_from"] == Path("wrfrst_d01")


def test_native_and_config_domain_owners_cannot_mix(runner):
    a = args()
    a.config = Path("other.toml")
    with pytest.raises(GoRefusal, match="CONFIG"):
        _go_wrfinput(a)
    assert not runner
