"""TUV-x port, configuration-time pieces (woof.chem_tuvx): word for word against the TUV-x oracle.

The oracle (tools/tuvx_oracle, NCAR TUV-x v0.17.0) reports, once its core is built, the wavelength grid's
edges and the extraterrestrial flux on its bins (tests/data/tuvx_oracle/fixtures.json "configuration").
The port reads the same pinned data files (woof.chem_tuvx_data) and must give the same binary64 words.  The
interpolators are also checked on small cases against their defining properties.  The data are fetched on
demand and never committed: the grading test runs where they are cached or WOOF_TUVX_DATA names a copy
(`python -c "from woof import chem_tuvx_data; chem_tuvx_data.ensure()"` fetches them once).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from woof import chem_tuvx as tuvx
from woof import chem_tuvx_data as data

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = json.loads((ROOT / "tests" / "data" / "tuvx_oracle" / "fixtures.json").read_text())
CONFIG = json.loads((ROOT / "tools" / "tuvx_oracle" / "photolysis.json").read_text())


def _data_root():
    """The verified data root when the data are here; None rather than a download from a test."""
    import os
    root = os.environ.get(data.ENV_ROOT) or data.cache_root()
    if not all((Path(root) / f).is_file() for f in data.FILES):
        return None
    return data.ensure(root=root)       # an explicit root is only verified, never fetched into


def test_the_grid_and_flux_are_the_oracles_words():
    root = _data_root()
    if root is None:
        pytest.skip("the pinned TUV-x data are not cached here (woof.chem_tuvx_data.ensure() fetches them)")
    grid, flux = tuvx.configuration(CONFIG, root=root)
    want = FIXTURES["configuration"]
    assert [v.hex() for v in grid.edge] == want["wavelength_edges_nm"]
    got = [v.hex() for v in flux]
    mismatch = [i for i, (a, b) in enumerate(zip(got, want["extraterrestrial_flux_photon_cm-2_s-1"])) if a != b]
    assert len(got) == len(want["extraterrestrial_flux_photon_cm-2_s-1"]) and not mismatch, mismatch[:10]


def test_conserving_keeps_the_area_of_a_linear_source():
    xs = np.array([0.0, 1.0, 2.5, 4.0, 7.0])
    ys = 3.0 + 2.0 * xs
    xt = np.array([0.5, 1.5, 3.0, 6.0])
    got = tuvx.interpolate_conserving(xt, xs, ys)
    np.testing.assert_allclose(got, 3.0 + (xt[1:] + xt[:-1]), rtol=1e-15)


def test_fractional_target_averages_bin_values():
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ys = np.array([1.0, 3.0, 5.0, 7.0, 0.0])
    got = tuvx.interpolate_fractional_target(np.array([0.5, 1.5, 3.0]), xs, ys)
    np.testing.assert_allclose(got, [(0.5 * 1.0 + 0.5 * 3.0), (0.5 * 3.0 + 1.0 * 5.0) / 1.5], rtol=1e-15)


def test_add_point_keeps_order_and_refuses_duplicates():
    x, y = tuvx.add_point([1.0, 3.0], [10.0, 30.0], 2.0, 20.0)
    assert (x, y) == ([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    with pytest.raises(tuvx.TuvxPortError):
        tuvx.add_point(x, y, 2.0, 0.0)


def test_a_short_interpolator_list_is_refused():
    grid = tuvx.WavelengthGrid([100.0, 200.0])
    with pytest.raises(tuvx.TuvxPortError, match="interpolators"):
        tuvx.extraterrestrial_flux(["a", "b"], [""], grid)


def test_the_unit_conversion_takes_the_single_precision_literal():
    assert tuvx._E13_SINGLE != 1.0e-13 and tuvx._E13_SINGLE == float(np.float32(1.0e-13))
