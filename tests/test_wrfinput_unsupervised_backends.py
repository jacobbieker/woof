"""A WRF-directory run is supervised only where the supervisor can reserve a card.

The supervisor selects its card through nvidia-smi, so on Metal (and the host
backends) a supervised `woof run --wrfinput` failed closed before its first
step.  These backends now integrate in-process, as --no-supervise does.
"""

import pytest

from woof import wrfinput_forecast


@pytest.mark.parametrize("name", ["mlx", "cpu", "host"])
def test_backends_without_an_nvidia_card_run_unsupervised(monkeypatch, name):
    monkeypatch.setenv("WOOF_BACKEND", name)
    assert wrfinput_forecast.supervised_launch(True) is False


def test_cuda_stays_supervised(monkeypatch):
    monkeypatch.setenv("WOOF_BACKEND", "cuda")
    assert wrfinput_forecast.supervised_launch(True) is True


@pytest.mark.parametrize("name", ["cuda", "mlx"])
def test_no_supervise_is_kept(monkeypatch, name):
    monkeypatch.setenv("WOOF_BACKEND", name)
    assert wrfinput_forecast.supervised_launch(False) is False
