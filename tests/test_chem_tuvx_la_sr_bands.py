"""TUV-x port, piece 4 (woof.chem_tuvx.LaSrBands): O2 Lyman-alpha and Schumann-Runge bands against the oracle.

For the first two cases of each layout the oracle records (tests/data/tuvx_oracle/fixtures.json, case
"intermediates" / "radiative_transfer") the O2 photolysis cross section with the bands applied
(la_sr_bands.F90 la_srb_xs, five height edges) and the O2 radiator's layer optical depth after the run, whose
band bins la_srb_OD sets (five layers).  The port takes the case's columns through its own spherical geometry
(piece 3) and the pinned Chebyshev coefficients (data/cross_sections/O2_parameters.txt).

The bands call exp and log.  On glibc (the oracle's libm) the band words must be the oracle's.  Elsewhere the
cross sections, which are libm outputs (exp of the Chebyshev fit), must stay within ``MAC_ULPS`` = 1 of them; the
Schumann-Runge optical depths difference adjacent levels' cross section x slant column (schum_OD), which turns one
ulp in a cross section into a few ulps of the difference, so they are held to ``MAC_OD_ULPS``.  Measured on Apple
silicon: one cross-section word in 540 differs (by 1 ulp) and one optical depth (by 6 ulps).  The test runs where the TUV-x data are cached or WOOF_TUVX_DATA names them.
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
#: largest ulp distance from the oracle allowed off glibc (Apple libm's exp and log)
MAC_ULPS = 1
#: ... for the differenced optical depths (see above)
MAC_OD_ULPS = 8


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


CASES = list(_cases())


def _words(values):
    return np.array([float.fromhex(w) for w in values], dtype=np.float64)


def _ulps(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.abs(a.view(np.int64) - b.view(np.int64))


def _same(got, want, what, mac_ulps=MAC_ULPS):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    assert got.shape == want.shape, what
    if GLIBC:
        assert got.tobytes() == want.tobytes(), (what, np.argwhere(got != want)[:5])
    else:
        assert int(_ulps(got, want).max(initial=0)) <= mac_ulps, (what, int(_ulps(got, want).max()))


@pytest.fixture(scope="module")
def bands():
    root = _root()
    if root is None:
        pytest.skip("the pinned TUV-x data are not cached here (woof.chem_tuvx_data.ensure() fetches them)")
    grid = tuvx.read_grid_csv(Path(root) / "data/grids/wavelength/combined.grid")
    return tuvx.LaSrBands(CONFIG["O2 absorption"], grid, root=root)


def _columns(case):
    geometry = tuvx.spherical_geometry(_words(case["height_edges_km"]), float.fromhex(case["solar_zenith_angle_deg"]))
    avc, asc = tuvx.air_mass(geometry, tuvx.exo_layer_columns(_words(case["air_layer"])))
    _ovc, osc = tuvx.air_mass(geometry, tuvx.exo_layer_columns(_words(case["O2_layer"])))
    return avc, asc, osc, _words(case["temperature"])[None, :]


@pytest.mark.parametrize("label,case", CASES, ids=[k for k, _ in CASES])
def test_band_cross_sections_and_optical_depths_are_the_oracles(bands, label, case):
    inter = case["intermediates"]
    rt = inter["radiative_transfer"]
    avc, asc, osc, temps = _columns(case)
    xs = bands.cross_section(avc, asc, osc, temps)
    od = bands.optical_depth(avc, asc, osc, temps)
    assert sorted(xs) == sorted(od) == bands.bins and len(bands.bins) == 1 + tuvx.NSRB
    want_xs = np.array([_words(row) for row in rt["o2_band_cross_section_cm2"]])
    want_od = np.array([_words(row) for row in rt["radiator_optical_depth"]["O2"]])
    for b in bands.bins:
        _same(xs[b][0][inter["height_edges"]], want_xs[:, b], f"cross section, bin {b}")
        _same(od[b][0][rt["layers"]], want_od[:, b], f"optical depth, bin {b}", MAC_OD_ULPS)


def test_the_bands_sit_on_the_combined_grid(bands):
    assert bands.has_la and bands.has_srb
    assert bands.ac.shape == bands.bc.shape == (tuvx.NPOLY, tuvx.NSRB)


def test_the_folded_constant_is_correctly_rounded():
    """exp(38._dk) is folded by gfortran (MPFR, correctly rounded): the nearest binary64 to e**38."""
    from decimal import Decimal, getcontext
    getcontext().prec = 80
    exact = Decimal(38).exp()
    below, above = np.nextafter(tuvx.SRB_COLMIN, 0.0), np.nextafter(tuvx.SRB_COLMIN, np.inf)
    error = abs(Decimal(tuvx.SRB_COLMIN) - exact)
    assert error <= abs(Decimal(float(below)) - exact) and error <= abs(Decimal(float(above)) - exact)
