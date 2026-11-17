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


def _request(**kw):
    values = dict(nudging=None, obs_nudging=None, tc_climate=None, w_damping=None, inflow_perturbation=None)
    values.update(kw)
    return SimpleNamespace(**values)


def test_the_partitioned_samehost_executor_refuses_what_its_ranks_do_not_install():
    """The same-host rank stacks are built by their own construction, which never installs the limiter or the
    inflow perturbation (execute_forecast does, on the one-card driver).  THE BREAKAGE: --w-damping on a
    partitioned same-host run was recorded in the receipt and never applied."""
    assert door._partitioned_samehost_refusal(_request()) is None
    assert "--w-damping" in door._partitioned_samehost_refusal(_request(w_damping=1.0))
    assert "--inflow-perturbation" in door._partitioned_samehost_refusal(_request(inflow_perturbation={"seed": 0}))
    assert "HEX_NUDGING_PARTITIONED" in door._partitioned_samehost_refusal(_request(nudging={}))


def test_the_partitioned_refusals_name_the_one_card_route():
    """--samehost-devices selects the partitioned executor even when it names one card, so "pin one device" was
    no way out: each refusal names the route that reaches the one-card driver."""
    for changes in (dict(nudging={}), dict(tc_climate={}), dict(w_damping=1.0),
                    dict(inflow_perturbation={"seed": 0})):
        message = door._partitioned_samehost_refusal(_request(**changes))
        assert "omit --samehost-devices" in message and "CUDA_VISIBLE_DEVICES" in message, changes


def test_the_driver_refuses_nudging_before_the_class_executor(monkeypatch):
    """execute_forecast hands an active full-step LTS run to the class executor, which is given no nudging.  The
    door refuses it (HEX_NUDGING_LTS); the driver's own entry point now does too, before the delegation."""
    from woof.hex.drivers import fullstep_lts_forecast_v841 as fullstep
    from woof.hex.drivers import run_cuda_v841_forecast as drv
    from woof.hex import wrf_hydro_hex
    from woof.hex.nudging import HexNudgingRefusal

    delegated = []
    monkeypatch.setattr(drv, "join_prepared_fingerprint", lambda host: None)
    monkeypatch.setattr(wrf_hydro_hex, "refuse_unwired", lambda *a, **k: None)
    monkeypatch.setattr(fullstep, "execute_fullstep_lts", lambda *a, **k: delegated.append(k) or {})
    kwargs = dict(host={}, schedule={}, cache_root=None, output_root=None, arwen_checkout=None, source_receipt={},
                  authority_receipt={}, fingerprint_every=0,
                  fullstep_lts_options={"fullstep_lts_mode": "active", "fullstep_lts_max_rate": 8})
    drv.execute_forecast(**kwargs)
    assert len(delegated) == 1
    with pytest.raises(HexNudgingRefusal, match="HEX_NUDGING_LTS"):
        drv.execute_forecast(**kwargs, nudging={"mode": "grid"})
    assert len(delegated) == 1
