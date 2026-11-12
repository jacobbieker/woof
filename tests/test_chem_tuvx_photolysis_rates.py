"""TUV-x port, piece 7 (woof.chem_tuvx.Tuvx): photolysis rate constants against the oracle, every case.

The oracle's rate constants (tests/data/tuvx_oracle/fixtures.json, every case of every layout: TUV 5.4's 1 km
grid, a model-like stretched grid and a sun at and below the horizon) are core%run's: the radiation field
scaled by the Earth-Sun distance factor, times the extraterrestrial flux, floored at zero, against each
reaction's cross section x quantum yield (the O2 bands applied to O2), summed over the wavelength bins in order.
The port runs the whole column from the case's inputs, the layouts' columns batched.

On glibc (the oracle's libm) every rate must be the oracle's word.  Elsewhere the libm's last bits pass through
the two-stream solve (tests/test_chem_tuvx_delta_eddington.py) and the O3 quantum yield, so each rate is held to
``MAC_RTOL`` of the reaction's largest rate on the column.  Measured on Apple silicon: 5418 of 7984 words differ,
the worst by 3.4e-14 of that scale.  The test
runs where the TUV-x data are cached or WOOF_TUVX_DATA names them.
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
#: off glibc, relative to the reaction's largest rate on the column (measured on Apple silicon: 3.4e-14)
MAC_RTOL = 1.0e-12


def _root():
    root = os.environ.get(data.ENV_ROOT) or data.cache_root()
    if not all((Path(root) / f).is_file() for f in data.FILES):
        return None
    return data.ensure(root=root)


def _words(values):
    return np.array([float.fromhex(w) for w in values], dtype=np.float64)


@pytest.fixture(scope="module")
def core():
    root = _root()
    if root is None:
        pytest.skip("the pinned TUV-x data are not cached here (woof.chem_tuvx_data.ensure() fetches them)")
    return tuvx.Tuvx(CONFIG, root=root)


def run_layout(core, fixture):
    cases = fixture["cases"]
    stack = lambda key: np.array([_words(c[key]) for c in cases])  # noqa: E731
    sza = np.array([float.fromhex(c["solar_zenith_angle_deg"]) for c in cases])
    esd = np.array([float.fromhex(c["earth_sun_distance_au"]) for c in cases])
    return core.run(stack("height_edges_km"), stack("temperature"), stack("air_layer"), stack("O2_layer"),
                    stack("O3_layer"), sza, esd)


@pytest.mark.parametrize("fixture", FIXTURES["fixtures"], ids=[f["layout"] for f in FIXTURES["fixtures"]])
def test_rate_constants_are_the_oracles(core, fixture):
    assert core.reactions == fixture["reactions"]
    rates = run_layout(core, fixture)
    for n, case in enumerate(fixture["cases"]):
        for name in fixture["reactions"]:
            got = rates[name][n]
            want = _words(case["photolysis_rate_constants_s-1"][name])
            assert got.shape == want.shape, name
            if GLIBC:
                assert got.tobytes() == want.tobytes(), (n, name, np.argwhere(got != want)[:5].ravel())
            else:
                scale = max(float(np.abs(want).max()), np.finfo(np.float64).tiny)
                assert float(np.abs(got - want).max()) / scale <= MAC_RTOL, (n, name)


def test_the_earth_sun_factor_scales_every_rate(core):
    fixture = FIXTURES["fixtures"][0]
    case = fixture["cases"][0]
    args = [_words(case[k]) for k in ("height_edges_km", "temperature", "air_layer", "O2_layer", "O3_layer")]
    sza = float.fromhex(case["solar_zenith_angle_deg"])
    one = core.run(*args, sza, 1.0)
    two = core.run(*args, sza, 2.0)
    for name in one:
        assert two[name].tobytes() == (2.0 * one[name]).tobytes()


def test_the_earth_sun_distance_is_the_oracles():
    """The oracle's "Earth-Sun distance" profile (photolysis.json: 2002-03-21, the time grid's 12 and 14 UT)."""
    want = FIXTURES["configuration"]["earth_sun_distance_au"]
    got = [tuvx.earth_sun_distance(2002, 3, 21, hour).hex() for hour in (12.0, 14.0)]
    if GLIBC:
        assert got == want
    else:
        assert max(abs(float.fromhex(a) - float.fromhex(b)) for a, b in zip(got, want)) <= 2.0 ** -52
