"""The door's --w-damping resolution (host)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from woof.hex import forecast_door as door
from woof.hex.cuda_w_damping import W_ALPHA, W_BETA, WDamping


def _args(**kw):
    return SimpleNamespace(**{"w_damping": False, "w_crit_cfl": None, **kw})


def test_w_damping_is_off_by_default_and_a_crit_cfl_alone_is_refused():
    assert door._resolve_w_damping(_args()) is None
    with pytest.raises(Exception, match="accepted and never read"):
        door._resolve_w_damping(_args(w_crit_cfl=1.2))


def test_w_damping_takes_woofs_default_or_the_given_w_crit_cfl():
    assert door._resolve_w_damping(_args(w_damping=True)) == 1.0
    assert door._resolve_w_damping(_args(w_damping=True, w_crit_cfl=1.2)) == 1.2
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(Exception, match="finite and positive"):
            door._resolve_w_damping(_args(w_damping=True, w_crit_cfl=bad))


@pytest.mark.parametrize("flag", ["local_timestep", "fullstep_local_timestep", "partial_nest"])
def test_w_damping_is_refused_where_its_driver_is_not_used(flag):
    with pytest.raises(Exception, match="never applied"):
        door._resolve_w_damping(_args(w_damping=True, **{flag: True}))


def test_the_constants_are_wrfs():
    assert (W_ALPHA, W_BETA) == (0.3, 1.0)
    with pytest.raises(ValueError):
        WDamping(w_crit_cfl=0.0)


def test_the_forecast_parser_carries_the_flags():
    import argparse

    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    assert vars(parser.parse_args([])).get("w_damping") is False
    got = parser.parse_args(["--w-damping", "--w-crit-cfl", "1.2"])
    assert got.w_damping is True and got.w_crit_cfl == 1.2
