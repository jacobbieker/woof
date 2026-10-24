"""The resident step's deferred health reads, on a device.

The CPU half (the switch, its named fallbacks, the host carrier check) is
tests/test_resident_health_ledger.py; this half needs CuPy and a card.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core import health_ledger
from test_resident_health_ledger import _carrier_fields, _check


# ---------------------------------------------------------------------------
# The ledger on a device
# ---------------------------------------------------------------------------

def _status(value):
    import cupy as cp

    return cp.asarray([value], dtype=cp.uint32)


def _refuse(message, kind=FloatingPointError):
    def describe(_flags):
        raise kind(message + health_ledger.deferred_note())
    return describe


@pytest.mark.gpu
@requires_gpu
def test_a_deferred_failure_is_raised_by_the_step_that_produced_it(
        monkeypatch):
    monkeypatch.delenv(health_ledger.RESIDENT_ENV, raising=False)
    with pytest.raises(FloatingPointError,
                       match="end of this model step") as caught:
        with health_ledger.resident_step() as ledger:
            assert ledger is not None and health_ledger.active() is ledger
            # Deferred: reported as "no fault" now ...
            assert health_ledger.read_status(
                _status(4), site="ysu",
                describe=_refuse("ysu went bad")) == 0
    assert "WOOF_HEALTH_READBACK=immediate" in str(caught.value)
    # ... raised at the exit, and cleared, so the next step is clean.
    with health_ledger.resident_step():
        health_ledger.read_status(_status(0), site="ysu",
                                  describe=_refuse("never"))
    assert health_ledger.active() is None


@pytest.mark.gpu
@requires_gpu
def test_a_site_the_resident_ledger_does_not_admit_reads_immediately(
        monkeypatch):
    """RRTMGP's input guard (and every unlisted site) keeps its read."""
    monkeypatch.delenv(health_ledger.RESIDENT_ENV, raising=False)
    with health_ledger.resident_step() as ledger:
        assert not ledger.admits("rrtmgp")
        assert health_ledger.read_status(
            _status(6), site="rrtmgp", describe=_refuse("never")) == 6
        assert "rrtmgp" not in ledger._index


@pytest.mark.gpu
@requires_gpu
def test_the_first_failure_in_program_order_is_the_one_raised(monkeypatch):
    """Not the site that happened to register first."""
    monkeypatch.delenv(health_ledger.RESIDENT_ENV, raising=False)
    # Register both sites, healthy, in the order kf then ysu.
    with health_ledger.resident_step():
        health_ledger.read_status(_status(0), site="kain-fritsch",
                                  describe=_refuse("kf"))
        health_ledger.read_status(_status(0), site="ysu",
                                  describe=_refuse("ysu"))
    # Now ysu fails FIRST and kf after it: an immediate run raises ysu's.
    with pytest.raises(FloatingPointError, match="^ysu"):
        with health_ledger.resident_step():
            health_ledger.read_status(_status(1), site="ysu",
                                      describe=_refuse("ysu"))
            health_ledger.read_status(_status(1), site="kain-fritsch",
                                      describe=_refuse("kf"))


@pytest.mark.gpu
@requires_gpu
def test_an_earlier_deferred_failure_wins_over_a_later_exception(
        monkeypatch):
    monkeypatch.delenv(health_ledger.RESIDENT_ENV, raising=False)
    with pytest.raises(FloatingPointError, match="earlier") as caught:
        with health_ledger.resident_step():
            health_ledger.read_status(_status(1), site="microphysics",
                                      describe=_refuse("earlier"))
            raise RuntimeError("later")
    assert isinstance(caught.value.__cause__, RuntimeError)
    # A healthy step that raises keeps its own exception.
    with pytest.raises(RuntimeError, match="only"):
        with health_ledger.resident_step():
            health_ledger.read_status(_status(0), site="microphysics",
                                      describe=_refuse("never"))
            raise RuntimeError("only")


@pytest.mark.gpu
@requires_gpu
def test_a_drain_that_cannot_read_never_hides_the_step_s_exception(
        monkeypatch):
    """A broken drain (a sticky CUDA fault, say) is not a health refusal."""
    monkeypatch.delenv(health_ledger.RESIDENT_ENV, raising=False)

    def broken(self, **_kwargs):
        raise RuntimeError("cannot read the device")

    with pytest.raises(KeyError, match="original"):
        with health_ledger.resident_step() as ledger:
            health_ledger.read_status(_status(0), site="microphysics",
                                      describe=_refuse("never"))
            monkeypatch.setattr(type(ledger), "drain", broken)
            raise KeyError("original")


@pytest.mark.gpu
@requires_gpu
def test_the_resident_ledger_is_per_thread(monkeypatch):
    monkeypatch.delenv(health_ledger.RESIDENT_ENV, raising=False)
    import cupy as cp

    barrier = threading.Barrier(2)
    seen: dict[int, object] = {}
    errors: list[BaseException] = []

    def worker(index):
        try:
            cp.cuda.Device(0).use()
            with health_ledger.resident_step() as ledger:
                barrier.wait(timeout=30)
                seen[index] = ledger
                assert health_ledger.active() is ledger
                barrier.wait(timeout=30)
        except BaseException as exc:          # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    assert seen[0] is not None and seen[1] is not None
    assert seen[0] is not seen[1]
    assert health_ledger.active() is None


# ---------------------------------------------------------------------------
# MYNN's guards
# ---------------------------------------------------------------------------

def _mynn_checks():
    import cupy as cp

    from woof.core.mynn_pbl_gpu import _nonfinite, _nonpositive

    good = cp.ones(8, dtype=cp.float32)
    nan = cp.ones(8, dtype=cp.float32)
    nan[3] = np.float32(np.nan)
    negative = cp.ones(8, dtype=cp.float32)
    negative[5] = np.float32(-1.0)
    return _nonfinite, _nonpositive, good, nan, negative


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("bad", ["first", "second", "both", "none"])
def test_a_deferred_refusal_names_the_check_the_eager_one_names(bad):
    import cupy as cp

    from woof.core.mynn_pbl_gpu import _first_refusal

    nonfinite, nonpositive, good, nan, negative = _mynn_checks()
    checks = (
        (nonfinite(), (good, nan if bad in ("first", "both") else good),
         "check one"),
        (nonpositive(), (good,), "check two (always passes)"),
        (nonpositive(),
         (negative if bad in ("second", "both") else good, good),
         "check three"),
    )
    flags = cp.zeros(16, dtype=cp.int32)
    eager = _first_refusal(checks, flags)
    ledger = health_ledger.HealthLedger(label="test", ordered=True,
                                        admit=health_ledger.RESIDENT_SITES)
    # Upstream 1.1.0's _first_refusal defers only at a named call site (every
    # production caller passes one, e.g. "mynn-tendency"); "mynn-" sites are
    # resident sites.
    with health_ledger.deferring(ledger):
        assert _first_refusal(checks, flags, site="mynn-test") is None
    if eager is None:
        ledger.drain()
        return
    with pytest.raises(ValueError) as caught:
        ledger.drain()
    assert str(caught.value).startswith(eager)
    assert "deferred health" in str(caught.value)


@pytest.mark.gpu
@requires_gpu
def test_a_deferred_flag_mask_reports_every_flagged_array():
    """The zero-forcing refusal lists every nonzero name, deferred too."""
    import cupy as cp

    from woof.core.mynn_pbl_gpu import (_flag_mask, _flag_mask_deferred,
                                        _nonzero)

    flags = cp.zeros(8, dtype=cp.int32)
    arrays = [cp.zeros(5, dtype=cp.float32) for _ in range(6)]
    for index in (1, 4):
        arrays[index][2] = np.float32(1.0)
    expected = _flag_mask(_nonzero(), arrays, flags)
    seen = []

    def refuse(mask):
        seen.append(mask)
        raise ValueError("refused")

    # No ledger: nothing launched, the caller keeps the immediate read.
    assert not _flag_mask_deferred(_nonzero(), arrays, flags, site="t",
                                   refuse=refuse)
    ledger = health_ledger.HealthLedger(label="test")
    with health_ledger.deferring(ledger):
        assert _flag_mask_deferred(_nonzero(), arrays, flags,
                                   site="zero-forcing-test", refuse=refuse)
        # A group longer than the flag block keeps the immediate read.
        assert not _flag_mask_deferred(_nonzero(), arrays * 2, flags,
                                       site="t2", refuse=refuse)
    with pytest.raises(ValueError, match="refused"):
        ledger.drain()
    assert seen == [expected]


# ---------------------------------------------------------------------------
# The surface-radiation carrier check on device carriers
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("land", [False, True])
def test_a_device_nonfinite_carrier_is_masked_by_the_footprint(land):
    import cupy as cp

    from woof.core.radiation_carriers import (
        CARRIER_SOURCE_DECLARED_CONSTANT, CarrierContract,
        CarrierContractError)

    contract = CarrierContract()
    contract.declare("glw", source=CARRIER_SOURCE_DECLARED_CONSTANT)
    contract.declare("swdown", source=CARRIER_SOURCE_DECLARED_CONSTANT)
    fields = _carrier_fields(cp, land=land, glw=float("nan"))
    ledger = health_ledger.HealthLedger(
        label="test", footprint_reads=True, ordered=True,
        admit=health_ledger.RESIDENT_SITES)
    with health_ledger.deferring(ledger):
        _check(contract, fields)                          # never raises here
    if land:
        with pytest.raises(CarrierContractError, match="not finite"):
            ledger.drain()
    else:
        ledger.drain()            # all water: the eager check never ran


# ---------------------------------------------------------------------------
# A forecast: bit-identical, and the sites really defer
# ---------------------------------------------------------------------------

_WINDFARM = dict(moist=True, ztop=20000.0, mp_physics=8, km_opt=4,
                 sf_sfclay_physics=5, bl_pbl_physics=5, bldt=0.0,
                 sf_surface_physics=2, ra_sw_physics=4, ra_lw_physics=4,
                 radt_minutes=5.0, cu_physics=0, dx=300.0, dy=300.0, dt=1.5,
                 time_step_sound=4)


def _windfarm_run(steps=4):
    import cupy as cp

    from tilestream import harness
    from tilestream import physics_inventory as physinv
    from tilestream.test_gate import hash_arrays
    from woof.core import dycore

    cfg = harness.make_config(24, 20, 16, **_WINDFARM)
    state, driver = physinv.default_builder(cfg, 20_260_731)
    for index in range(steps):
        # Step 1 is the mandatory radiation call; force one more mid-run.
        driver.radiation_due_override = True if index == 2 else None
        dycore.step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()
    return hash_arrays(physinv.carrier_inventory(state)), state


@pytest.mark.gpu
@requires_gpu
def test_deferred_and_immediate_forecasts_are_bit_identical(monkeypatch):
    monkeypatch.setenv(health_ledger.RESIDENT_ENV, "immediate")
    immediate, _ = _windfarm_run()
    monkeypatch.setenv(health_ledger.RESIDENT_ENV, "deferred")
    before = health_ledger.resident_receipt()
    deferred, _ = _windfarm_run()
    after = health_ledger.resident_receipt()
    assert deferred == immediate
    assert after["deferred_steps"] - before["deferred_steps"] == 4
    assert after["drains"] > before["drains"]
    # The sites a wind-farm step used to read one at a time now report
    # into the resident ledger.
    import cupy as cp

    ledger = health_ledger._LOCAL.ledgers[int(cp.cuda.runtime.getDevice())]
    sites = set(ledger._index)
    assert "microphysics" in sites
    assert any(site.startswith("carrier-contract:") for site in sites)
    # The MYNN refusal sites are upstream 1.1.0's named ones
    # ("mynn-tendency", ...), not the fork's "mynn-refusal:<message>".
    assert any(site.startswith("mynn-") and site != "mynn-tendencies-finite"
               for site in sites)
    assert "mynn-tendencies-finite" in sites


@pytest.mark.gpu
@requires_gpu
def test_a_sick_forecast_still_stops_in_the_step_that_went_bad(monkeypatch):
    import cupy as cp

    from woof.core import dycore

    monkeypatch.setenv(health_ledger.RESIDENT_ENV, "deferred")
    _digest, state = _windfarm_run(steps=1)
    cfg = state.physics.cfg if hasattr(state.physics, "cfg") else None
    from tilestream import harness

    cfg = cfg or harness.make_config(24, 20, 16, **_WINDFARM)
    state.qv[3, 4, 5] = cp.float32(np.nan)
    with pytest.raises((ValueError, FloatingPointError)) as caught:
        dycore.step(state, cfg)
    assert "deferred health" in str(caught.value)


@pytest.mark.gpu
@requires_gpu
def test_a_host_footprint_beside_a_device_carrier_is_exact():
    """Host xland, device carrier: all water must not refuse."""
    import cupy as cp

    from woof.core.radiation_carriers import (
        CARRIER_SOURCE_DECLARED_CONSTANT, CarrierContract)

    contract = CarrierContract()
    contract.declare("glw", source=CARRIER_SOURCE_DECLARED_CONSTANT)
    contract.declare("swdown", source=CARRIER_SOURCE_DECLARED_CONSTANT)
    fields = _carrier_fields(np, land=False)
    fields["glw"] = cp.full((2, 2), np.nan, dtype=cp.float32)
    fields["swdown"] = cp.zeros((2, 2), dtype=cp.float32)
    ledger = health_ledger.HealthLedger(
        label="test", footprint_reads=True, ordered=True,
        admit=health_ledger.RESIDENT_SITES)
    with health_ledger.deferring(ledger):
        _check(contract, fields)
    ledger.drain()
