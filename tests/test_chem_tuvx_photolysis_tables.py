"""TUV-x port, piece 2 (woof.chem_tuvx): cross sections and quantum yields against the oracle's intermediates.

For the first two cases of each column layout the oracle records every reaction's cross section and quantum yield
on five height edges (tests/data/tuvx_oracle/fixtures.json, case "intermediates").  The port computes them from the
case's edge temperatures and the pinned data files (woof.chem_tuvx_data).  Cross sections take no libm call and
must be the oracle's words everywhere.  The O3 quantum yields call exp and pow: on glibc (the oracle's libm) they
must be its words, elsewhere within one ulp.  The test runs where the data are cached or WOOF_TUVX_DATA names them.
"""
import json
import os
import platform
from pathlib import Path

import numpy as np
import pytest

from woof import chem_tuvx as tuvx
from woof import chem_tuvx_data as data

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = json.loads((ROOT / "tests" / "data" / "tuvx_oracle" / "fixtures.json").read_text())
CONFIG = json.loads((ROOT / "tools" / "tuvx_oracle" / "photolysis.json").read_text())
GLIBC = platform.system() == "Linux" and platform.libc_ver()[0] == "glibc"
LIBM = {"O3+hv->O2+O(1D)", "O3+hv->O2+O(3P)"}


def _root():
    root = os.environ.get(data.ENV_ROOT) or data.cache_root()
    if not all((Path(root) / f).is_file() for f in data.FILES):
        return None
    return data.ensure(root=root)


def _cases():
    for fixture in FIXTURES["fixtures"]:
        for n, case in enumerate(fixture["cases"]):
            if "intermediates" in case:
                yield f"{fixture['layout']}-{n}", case


def _ulps(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ia, ib = a.view(np.int64), b.view(np.int64)
    return np.abs(ia - ib)


@pytest.fixture(scope="module")
def photolysis():
    root = _root()
    if root is None:
        pytest.skip("the pinned TUV-x data are not cached here (woof.chem_tuvx_data.ensure() fetches them)")
    grid = tuvx.read_grid_csv(Path(root) / "data/grids/wavelength/combined.grid")
    return tuvx.Photolysis(CONFIG, grid, root=root)


@pytest.mark.parametrize("label,case", list(_cases()), ids=[k for k, _ in _cases()])
def test_cross_sections_and_quantum_yields_are_the_oracles(photolysis, label, case):
    inter = case["intermediates"]
    temps = np.array([float.fromhex(w) for w in case["temperature"]])[inter["height_edges"]]
    tables = photolysis.tables(temps)
    for name, (xs, qy) in tables.items():
        want_xs = np.array([[float.fromhex(w) for w in row] for row in inter["cross_section_cm2"][name]])
        want_qy = np.array([[float.fromhex(w) for w in row] for row in inter["quantum_yield"][name]])
        assert xs.shape == want_xs.shape, name
        assert xs.tobytes() == want_xs.tobytes(), (name, np.argwhere(xs != want_xs)[:5])
        if name in LIBM and not GLIBC:
            assert int(_ulps(qy, want_qy).max()) <= 1, name
        else:
            assert qy.tobytes() == want_qy.tobytes(), (name, np.argwhere(qy != want_qy)[:5])
