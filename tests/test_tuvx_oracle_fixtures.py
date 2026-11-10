"""The TUV-x photolysis oracle fixtures (tools/tuvx_oracle): provenance, shape and licence hygiene.

The fixtures are the words WOOF's TUV-x port (MICM photolysis on hex) is graded against; this file
checks what holds before any port exists: every case carries its inputs and one finite non-negative
rate per reaction per height edge, the oracle's configuration names exactly those reactions, and the
GPL-carrying TUV-x assert module never enters the build.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "tuvx_oracle"
DATA = json.loads((ROOT / "tests" / "data" / "tuvx_oracle" / "fixtures.json").read_text())
REACTIONS = ["O2+hv->O+O", "O3+hv->O2+O(1D)", "O3+hv->O2+O(3P)", "NO2+hv->NO+O(3P)"]


def _f(words):
    return np.array([float.fromhex(w) for w in words])


def test_the_configuration_names_the_fixture_reactions():
    config = json.loads((TOOL / "photolysis.json").read_text())
    assert [r["name"] for r in config["photolysis"]["reactions"]] == REACTIONS
    assert {f["layout"] for f in DATA["fixtures"]} == {"tuv54_1km", "model_stretched"}
    assert all(f["reactions"] == REACTIONS for f in DATA["fixtures"])


def test_every_case_carries_its_column_and_one_rate_per_edge():
    for fixture in DATA["fixtures"]:
        for case in fixture["cases"]:
            edges = _f(case["height_edges_km"])
            assert np.all(np.diff(edges) > 0)
            assert _f(case["temperature"]).size == edges.size
            for key in ("air", "O2", "O3"):
                assert _f(case[key]).size == edges.size
                assert _f(case[key + "_layer"]).size == edges.size - 1
            rates = case["photolysis_rate_constants_s-1"]
            assert list(rates) == REACTIONS
            for values in rates.values():
                j = _f(values)
                assert j.size == edges.size and np.all(np.isfinite(j)) and np.all(j >= 0)
            # More light aloft: the top-of-column NO2 rate exceeds the surface rate.
            no2 = _f(rates["NO2+hv->NO+O(3P)"])
            assert no2[-1] > no2[0]


def test_the_gpl_assert_never_enters_the_build():
    build = (TOOL / "build.sh").read_text()
    assert "ebec1cb47084fd102e338fb896c6d2b80fbcd292" in build
    assert 'cp "$here/musica_assert.F90" tuv-x/src/util/assert.F90' in build
    assert "General Public License" not in (TOOL / "musica_assert.F90").read_text()
