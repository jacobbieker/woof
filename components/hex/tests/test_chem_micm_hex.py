"""The hex chemistry config's micm block: CheMPAS-A's coupling of a MICM mechanism, refusals by name."""
import json
from pathlib import Path

import pytest

from woof.hex import chem_micm_hex
from woof.hex.chem_transport import configure, history_metadata, history_units, load_config

MECHANISM = Path(__file__).resolve().parents[3] / "tests" / "data" / "micm_oracle" / "chempas_a_chapman_nox.yaml"
PHOTO = {"jO2": {"j_max": 1e-11}, "jO3_O": {"j_max": 4e-4}, "jO3_O1D": {"j_max": 3e-5},
         "jNO2": {"j_max": 8e-3}}


def config(tmp_path, **block):
    body = {"micm_file": str(MECHANISM), "photolysis": PHOTO, **block}
    path = tmp_path / "chemistry.json"
    path.write_text(json.dumps({"schema": "gpuwm-hex.chemistry/v1", "micm": body}))
    return path


def test_rows_are_the_mechanism_species_in_lower_case(tmp_path):
    row = configure(config(tmp_path))
    assert row.appended_scalar_names == ("o1d", "o2", "o", "o3", "no", "no2")
    assert history_units("no2") == "kg kg-1"
    assert history_units("o3_column_mass") == "kg m-2"


def test_metadata_names_the_mechanism(tmp_path):
    import hashlib
    path = config(tmp_path)
    meta = json.loads(history_metadata(path, hashlib.sha256(path.read_bytes()).hexdigest()))
    assert meta["mode"] == "micm" and meta["mechanism"] == str(MECHANISM)


@pytest.mark.parametrize("block,match", [
    ({"photolysis": {}}, "jO2"),
    ({"photolysis": {**PHOTO, "jXX": {"j_max": 1.0}}}, "does not have"),
    ({"rate_parameters": {"USER.x": 1.0}}, "does not have"),
    ({"chemistry_interval": -1}, "nonnegative"),
    ({"relative_tolerance": 0}, "positive"),
    ({"initial_mass_mixing_ratio": {"NOPE": 1.0}}, "does not transport"),
    ({"tuvx": True}, "not carried"),
])
def test_refusals_by_name(tmp_path, block, match):
    with pytest.raises(ValueError, match=match):
        load_config(config(tmp_path, **block))


def test_one_mode_per_config(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"schema": "gpuwm-hex.chemistry/v1", "micm": {}, "wrf_chem": {}}))
    with pytest.raises(ValueError, match="not \\['wrf_chem', 'micm'\\]"):
        load_config(path)


def test_species_without_molar_mass_refused(tmp_path):
    mech = tmp_path / "m.json"
    mech.write_text(json.dumps({"version": "1.0.0", "species": [{"name": "A"}, {"name": "B"}],
                                "phases": [{"name": "gas", "species": ["A", "B"]}],
                                "reactions": [{"type": "ARRHENIUS", "gas phase": "gas",
                                               "reactants": [{"species name": "A"}],
                                               "products": [{"species name": "B"}]}]}))
    path = tmp_path / "chemistry.json"
    path.write_text(json.dumps({"schema": "gpuwm-hex.chemistry/v1", "micm": {"micm_file": "m.json"}}))
    with pytest.raises(ValueError, match="molecular weight"):
        load_config(path)


def test_cos_sza_is_chempas_formula():
    cp = pytest.importorskip("cupy")
    import math
    # Equinox-ish noon on the equator at 0E: the sun is near the zenith.
    value = float(chem_micm_hex.cos_sza(cp.asarray([0.0]), cp.asarray([0.0]), 80, 12.0)[0])
    assert 0.99 < value <= 1.0
    night = float(chem_micm_hex.cos_sza(cp.asarray([0.0]), cp.asarray([180.0]), 80, 12.0)[0])
    assert night < -0.99
    assert not math.isnan(value)
