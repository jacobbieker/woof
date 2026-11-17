"""Each numerics child compiles hex units in the mode it names.

``woof.hex.numerics`` reads ``WOOF_HEX_NUMERICS`` (default fast), not ``GPUWM_NUMERICS``.  Until
2026-11-16 ``tests/_numerics_child.py`` set only the engine's knob, so every hex identity test's
"strict" and "wrf_exact" children compiled their hex units fast and the strict claim was never checked.
"""

from __future__ import annotations

import pytest

import _numerics_child


def _probe_hex_mode():
    """Run in the child: the hex mode and the options a hex unit compiles with follow the engine's mode."""
    from woof import numerics
    from woof.hex import numerics as hex_numerics

    assert hex_numerics.mode() == numerics.mode(), (hex_numerics.mode(), numerics.mode())
    strict = ("--std=c++17", "--fmad=false")
    chosen = hex_numerics.options(strict)
    if numerics.mode() == "strict":
        assert chosen == strict, chosen
    else:
        assert "--fmad=true" in chosen and "--fmad=false" not in chosen, chosen


@pytest.mark.parametrize(*_numerics_child.MODES)
def test_hex_units_compile_in_the_child_mode(numerics, exact):
    _numerics_child.run(__file__, ["_probe_hex_mode"], numerics, exact, timeout=300)


def test_battery_compiles_hex_units_strict():
    """tests/conftest.py names strict for the hex units as it does for the engine: until 2026-11-17 it
    pinned only GPUWM_NUMERICS, so ~120 hex-importing battery files compiled their hex units fast."""
    import os

    from woof import numerics
    from woof.hex import numerics as hex_numerics

    assert os.environ["WOOF_HEX_NUMERICS"] == "strict"
    assert hex_numerics.mode() == numerics.mode() == "strict", (hex_numerics.mode(), numerics.mode())
