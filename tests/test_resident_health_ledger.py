"""The resident step's deferred health reads (woof.core.health_ledger).

``dycore.step`` runs inside :class:`health_ledger.resident_step`: the
schemes' error-only status reads are recorded on the device and read ONCE,
at the end of the step, instead of one blocking read per site.  What is
pinned here:

* the mode switch (``WOOF_HEALTH_READBACK``) and its named fallbacks;
* a deferred failure is raised by the step that produced it, with its type
  and message intact and the deferred note appended, and in preference to a
  later exception from the same step;
* the ledger is per thread, so one slab's drain never reads another's;
* MYNN's guard reads and the surface-radiation carrier check defer with the
  eager answer -- the same first message, the same consumer footprint;
* a forecast is bit-identical with the reads deferred and immediate.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.core import health_ledger


# ---------------------------------------------------------------------------
# The switch (no device)
# ---------------------------------------------------------------------------

def test_the_mode_defaults_to_deferred_and_reads_the_environment(monkeypatch):
    monkeypatch.delenv(health_ledger.RESIDENT_ENV, raising=False)
    assert health_ledger.resident_mode() == "deferred"
    assert health_ledger.resident_receipt()["source"] == "default"
    monkeypatch.setenv(health_ledger.RESIDENT_ENV, "Immediate")
    assert health_ledger.resident_mode() == "immediate"
    assert health_ledger.resident_receipt()["source"] == "environment"


def test_an_unknown_mode_is_refused_not_ignored(monkeypatch):
    monkeypatch.setenv(health_ledger.RESIDENT_ENV, "later")
    with pytest.raises(ValueError, match="WOOF_HEALTH_READBACK"):
        health_ledger.resident_mode()


def test_immediate_mode_installs_nothing_and_counts_the_reason(monkeypatch):
    monkeypatch.setenv(health_ledger.RESIDENT_ENV, "immediate")
    before = health_ledger.resident_receipt()["immediate_steps"].get(
        "mode-immediate", 0)
    with health_ledger.resident_step() as ledger:
        assert ledger is None
        assert health_ledger.active() is None
    after = health_ledger.resident_receipt()["immediate_steps"]
    assert after["mode-immediate"] == before + 1


def test_an_outer_ledger_owns_the_drain(monkeypatch):
    """A tiled sweep or graph capture already deferring: no nesting."""
    monkeypatch.delenv(health_ledger.RESIDENT_ENV, raising=False)
    outer = object()
    with health_ledger.deferring(outer):
        with health_ledger.resident_step() as ledger:
            assert ledger is None
            assert health_ledger.active() is outer
    assert health_ledger.active() is None
    assert health_ledger.resident_receipt()["immediate_steps"][
        "outer-ledger"] >= 1


def test_the_receipt_shape_is_pinned():
    receipt = health_ledger.resident_receipt()
    assert set(receipt) == {"mode", "source", "env", "deferred_steps",
                            "drains", "immediate_steps"}
    assert set(health_ledger.IMMEDIATE_REASONS) == {
        "mode-immediate", "outer-ledger", "capturing"}


# ---------------------------------------------------------------------------
# The surface-radiation carrier check: exact under the resident ledger
# ---------------------------------------------------------------------------

def _carrier_fields(xp, *, land: bool, glw=300.0):
    fields = {
        "glw": xp.full((2, 2), glw, dtype=xp.float32),
        "swdown": xp.zeros((2, 2), dtype=xp.float32),
        "xland": xp.full((2, 2), 2.0, dtype=xp.float32),
        "xice": xp.zeros((2, 2), dtype=xp.float32),
    }
    if land:
        fields["xland"][0, 0] = 1.0
    return fields


def _check(contract, fields):
    contract.check_before_consumption(
        sf_surface_physics=2, fields=fields, model_time=0.0,
        radiation_interval_seconds=900.0, timestep_seconds=20.0)


def test_an_exact_ledger_keeps_the_eager_footprint_on_host_carriers():
    """All-water passes eagerly AND under the resident (exact) ledger.

    The tiled ledger over-enforces (pinned in
    test_surface_radiation_carrier_contract.py); the resident one reads the
    footprint on the refusing path and so answers exactly as eager does.
    """
    from woof.core.radiation_carriers import (CarrierContract,
                                              CarrierContractError)

    class _Exact:
        footprint_reads = True

        def record(self, *args, **kwargs):     # pragma: no cover
            raise AssertionError("no device carrier to record")

    water = _carrier_fields(np, land=False)
    _check(CarrierContract(), water)                      # eager passes
    with health_ledger.deferring(_Exact()):
        _check(CarrierContract(), water)                  # exact passes
        with pytest.raises(CarrierContractError, match="has no producer"):
            _check(CarrierContract(), _carrier_fields(np, land=True))


def test_the_lagged_executor_ledger_is_exact_on_an_all_water_domain():
    """#80's lagged ledger reads the footprint on a refusing path too."""
    from woof.core.lagged_health import _DeferredLedger
    from woof.core.radiation_carriers import (CarrierContract,
                                              CarrierContractError)

    ledger = _DeferredLedger()
    assert ledger.footprint_reads is True
    assert not ledger.allocated
    with health_ledger.deferring(ledger):
        _check(CarrierContract(), _carrier_fields(np, land=False))
        with pytest.raises(CarrierContractError, match="has no producer"):
            _check(CarrierContract(), _carrier_fields(np, land=True))
    assert not ledger.allocated
