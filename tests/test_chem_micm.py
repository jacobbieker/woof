"""WOOF's MICM port (woof/chem_micm.py): the solver structure MICM builds, and refusals by name.

The word-exact grading against the pinned MICM oracle is tests/test_chem_micm_gpu.py.
"""
import json
from pathlib import Path

import pytest

from woof.chem_micm import MechanismError, kernel_source, load_mechanism, markowitz

DATA = Path(__file__).resolve().parent / "data" / "micm_oracle"
FIXTURES = json.loads((DATA / "fixtures.json").read_text())["fixtures"]


@pytest.mark.parametrize("name", sorted({f["mechanism"] for f in FIXTURES}))
def test_species_and_parameter_order_are_micms(name):
    mech = load_mechanism(DATA / name)
    fixture = next(f for f in FIXTURES if f["mechanism"] == name)
    assert list(mech.species) == fixture["species_order"]
    assert list(mech.parameters) == fixture["parameter_order"]
    assert "micm_ros3" in kernel_source(mech)


def test_nitric_oxide_stays_a_name():
    mech = load_mechanism(DATA / "chempas_a_chapman_nox.yaml")
    assert "NO" in mech.species and "NO2" in mech.species


def test_third_bodies_and_constant_species_are_not_state():
    mech = load_mechanism(DATA / "woof_types.json")
    assert set(mech.parameterized) == {"M", "N2", "H2O"}
    assert not set(mech.parameterized) & set(mech.species)


def test_markowitz_unsigned_cost():
    # Node 0 has no row entries and two column entries: MICM's size_t cost (0-1)*(2-1) wraps to
    # 2^64-1, so the product-only nodes (cost 0) go first; signed arithmetic would pick node 0 (-1).
    assert markowitz({(1, 0), (2, 0)}, 3) == [1, 0, 2]


def _doc(**reaction):
    return {"version": "1.0.0", "species": [{"name": "A"}, {"name": "B"}],
            "phases": [{"name": "gas", "species": ["A", "B"]}],
            "reactions": [{"gas phase": "gas", "reactants": [{"species name": "A"}],
                           "products": [{"species name": "B"}], **reaction}]}


@pytest.mark.parametrize("reaction,match", [
    ({"type": "TUNNELING"}, "TUNNELING is not carried"),
    ({"type": "SURFACE"}, "SURFACE is not carried"),
    ({"type": "ARRHENIUS", "gas phase": "aqueous"}, "not in the gas phase"),
    ({"type": "ARRHENIUS", "Ea": 1e-21, "C": 3.0}, "mutually exclusive"),
    ({"type": "ARRHENIUS", "reactants": [{"species name": "Q"}]}, "not a gas-phase species"),
])
def test_refusals_by_name(reaction, match):
    with pytest.raises(MechanismError, match=match):
        load_mechanism(_doc(**reaction))


def test_v0_refused():
    with pytest.raises(MechanismError, match="v1"):
        load_mechanism({"version": "0.1", "species": [], "phases": [], "reactions": []})
