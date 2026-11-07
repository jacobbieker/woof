"""The coupling spine: the shared clock, the matched exchange, the component registry.

These tests pin the spine that the ocean, sea-ice and land-ice cores couple to:
the clock that steps every component on one wall clock (refusing a cadence that
is not a multiple of a component's timestep, or a run length that is not a
multiple of the cadence), the matched exchange (refusing a channel that is not
conserved to the bit), and the registry (refusing a core that is not yet
ported, by name).
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.hex.coupled import clock, exchange, registry
from woof.hex.coupled.clock import CoupledClock
from woof.hex.coupled.exchange import FluxChannel, channels_for, check_matched
from woof.hex.coupled.registry import all_components, component, require_ported
from woof.hex.errors import ConfigurationRefusal


def test_registry_names_all_four_cores() -> None:
    names = [c.name for c in all_components()]
    assert names == ["atmosphere", "ocean", "seaice", "landice"]
    assert component("atmosphere").status == "ported"
    assert component("seaice").status == "stub"
    assert component("landice").core_dir == "core_landice"


def test_registry_refuses_an_unknown_component_by_name() -> None:
    with pytest.raises(ConfigurationRefusal):
        component("bogus")


def test_require_ported_passes_the_atmosphere_and_refuses_a_stub() -> None:
    assert require_ported("atmosphere").status == "ported"
    with pytest.raises(ConfigurationRefusal) as exc:
        require_ported("seaice")
    assert "docs/mpas-coupled-roadmap.md" in str(exc.value)


def test_clock_yields_intervals_and_substeps() -> None:
    clk = CoupledClock(start=0.0, end=3600.0, cadence=900.0, dts={"atmosphere": 300.0})
    intervals = list(clk.intervals())
    assert [i.index for i in intervals] == [0, 1, 2, 3]
    assert intervals[0].start == 0.0 and intervals[0].end == 900.0
    assert intervals[-1].end == 3600.0
    assert clk.substeps("atmosphere") == 3
    receipt = clk.receipt()
    assert receipt["intervals"] == 4
    assert receipt["components"]["atmosphere"]["substeps_per_cadence"] == 3


def test_clock_refuses_a_cadence_not_a_multiple_of_a_timestep() -> None:
    with pytest.raises(ValueError, match="not a multiple of the atmosphere timestep"):
        CoupledClock(start=0.0, end=3600.0, cadence=900.0, dts={"atmosphere": 130.0})


def test_clock_refuses_a_run_length_not_a_multiple_of_the_cadence() -> None:
    with pytest.raises(ValueError, match="not a multiple of the cadence"):
        CoupledClock(start=0.0, end=3000.0, cadence=900.0, dts={"atmosphere": 300.0})


def test_clock_refuses_empty_and_nonpositive_arguments() -> None:
    with pytest.raises(ValueError):
        CoupledClock(start=0.0, end=1.0, cadence=1.0, dts={})
    with pytest.raises(ValueError):
        CoupledClock(start=1.0, end=0.0, cadence=1.0, dts={"atmosphere": 1.0})
    with pytest.raises(ValueError):
        CoupledClock(start=0.0, end=1.0, cadence=0.0, dts={"atmosphere": 1.0})


def test_channels_for_returns_only_channels_whose_ends_are_present() -> None:
    assert channels_for({"atmosphere"}) == ()
    si_atm = channels_for({"atmosphere", "seaice"})
    assert {c.name for c in si_atm} == {
        "atm2si_momentum_x",
        "atm2si_momentum_y",
        "atm2si_heat",
        "atm2si_freshwater",
        "si2atm_sst",
        "si2atm_cover",
    }
    # a channel needs BOTH ends present
    assert all(c.source in {"atmosphere", "seaice"} and c.target in {"atmosphere", "seaice"}
               for c in si_atm)


def _channel() -> FluxChannel:
    return FluxChannel("atm2si_heat", "atmosphere", "seaice", "surface heat flux", "W/m^2")


def test_check_matched_passes_identical_fields() -> None:
    field = np.array([1.0, -2.0, 3.5, 0.0], dtype=np.float32)
    check_matched(_channel(), field, field.copy())


def test_check_matched_refuses_a_word_mismatch() -> None:
    sent = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    received = np.array([1.0, 2.0, 3.0009766], dtype=np.float32)
    with pytest.raises(ConfigurationRefusal) as exc:
        check_matched(_channel(), sent, received)
    assert "1 of 3 words differ" in str(exc.value)
    # one word differing in several bytes is still one word (1.0 against 2.0)
    with pytest.raises(ConfigurationRefusal) as exc:
        check_matched(_channel(), np.array([1.0, 5.0], np.float32), np.array([2.0, 5.0], np.float32))
    assert "1 of 2 words differ" in str(exc.value)


def test_check_matched_refuses_shape_and_dtype_mismatch() -> None:
    with pytest.raises(ConfigurationRefusal, match="shape"):
        check_matched(_channel(), np.ones(3, dtype=np.float32), np.ones(4, dtype=np.float32))
    with pytest.raises(ConfigurationRefusal, match="dtype"):
        check_matched(_channel(), np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float64))


def test_check_matched_refuses_a_nan_payload_delta() -> None:
    a = np.array([float("nan")], dtype=np.float64)
    b = np.array([float("nan")], dtype=np.float64)
    b.view(np.uint64)[0] = 0x7FF8000000000001  # a different NaN payload
    with pytest.raises(ConfigurationRefusal):
        check_matched(_channel(), a, b)


def test_the_coupled_door_refuses_an_unported_core_named_by_dt() -> None:
    import argparse

    from woof.hex.coupled import door

    args = argparse.Namespace(component=[], plan=True, start=0.0, end=3600.0, cadence=900.0,
                              dt=["seaice=300"])
    with pytest.raises(ConfigurationRefusal) as exc:
        door._coupled(args)
    assert "seaice" in str(exc.value)
