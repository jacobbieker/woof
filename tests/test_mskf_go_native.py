"""The stock-WRF go route binds checkpoint selection and retention."""
import argparse
from types import SimpleNamespace

from woof import go_cli


def test_native_go_forwards_checkpoint_controls(tmp_path, monkeypatch):
    """--native-shared-gpu (= --allow-shared-gpu --no-supervise), the
    checkpoint time, the retention and the products reach the runner."""
    import os
    import woof.wrfinput_forecast as worker
    from woof import wrfinput_go
    from woof.resume import KEEP_CHECKPOINTS_ENV
    parser = argparse.ArgumentParser()
    go_cli._add_go_arguments(parser)
    args = parser.parse_args(['--wrfinput', str(tmp_path), '--native-shared-gpu',
                              '--checkpoint-at', 'end', '--keep-checkpoints', '2',
                              '--products', 'none', '--run-stamp', 'off'])
    seen = {}

    def run(source, out, **kwargs):
        seen.update(kwargs, source=source,
                    keep=os.environ.get(KEEP_CHECKPOINTS_ENV))
        return 0
    monkeypatch.setattr(worker, 'run_wrf_forecast', run)
    monkeypatch.setattr(wrfinput_go, '_resolve_inputs', lambda directory, a: SimpleNamespace(
        experiment=SimpleNamespace(start_time=None)))
    assert go_cli._go_wrfinput(args) == 0
    assert seen['source'] == tmp_path
    assert seen['checkpoint_at'] == 'end'
    assert seen['allow_shared_gpu'] is True
    assert seen['exclusive_gpu'] is False
    assert seen['keep'] == '2'
    assert seen['render_products'] == 'none'


def test_native_shared_gpu_is_refused_on_a_configuration_launch(tmp_path):
    import pytest
    parser = argparse.ArgumentParser()
    go_cli._add_go_arguments(parser)
    config = tmp_path / "exp.toml"
    config.write_text("")
    args = parser.parse_args([str(config), '--native-shared-gpu'])
    with pytest.raises(go_cli.GoRefusal, match='native-shared-gpu'):
        go_cli.go_main(args)
