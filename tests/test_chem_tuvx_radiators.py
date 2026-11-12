"""TUV-x port, piece 5 (woof.chem_tuvx.Radiators): air (Rayleigh), O2 and O3 optical depths against the oracle.

For the first two cases of each layout the oracle records (tests/data/tuvx_oracle/fixtures.json, case
"intermediates" / "radiative_transfer") each radiator's layer optical depth after the run (the O2 band bins set by
la_srb_OD) and the radiators' accumulated optical depth, single-scattering albedo and asymmetry factor, on five
layers.  The port builds them from the case's layer columns and edge temperatures, its own geometry (piece 3) and
bands (piece 4), and the pinned data.

The Rayleigh cross section calls pow and the bands exp and log.  On glibc (the oracle's libm) every word must be
the oracle's; elsewhere within ``MAC_ULPS`` (1, the libm results) or, in the O2 band bins, ``MAC_OD_ULPS`` for the
band optical depths (see tests/test_chem_tuvx_la_sr_bands.py) and ``MAC_DERIVED_ULPS`` for the accumulated optics
that sum and divide by them (measured on Apple silicon: 10 ulps in one single-scattering albedo word).  The test runs where the TUV-x data are cached or
WOOF_TUVX_DATA names them.
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
MAC_ULPS = 1
MAC_OD_ULPS = 8
MAC_DERIVED_ULPS = 16


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


def _table(rows):
    return np.array([_words(row) for row in rows])


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
def parts():
    root = _root()
    if root is None:
        pytest.skip("the pinned TUV-x data are not cached here (woof.chem_tuvx_data.ensure() fetches them)")
    grid = tuvx.read_grid_csv(Path(root) / "data/grids/wavelength/combined.grid")
    return tuvx.Radiators(CONFIG, grid, root=root), tuvx.LaSrBands(CONFIG["O2 absorption"], grid, root=root)


def radiator_states(radiators, bands, case):
    geometry = tuvx.spherical_geometry(_words(case["height_edges_km"]), float.fromhex(case["solar_zenith_angle_deg"]))
    avc, asc = tuvx.air_mass(geometry, tuvx.exo_layer_columns(_words(case["air_layer"])))
    _ovc, osc = tuvx.air_mass(geometry, tuvx.exo_layer_columns(_words(case["O2_layer"])))
    temps = _words(case["temperature"])[None, :]
    band_od = bands.optical_depth(avc, asc, osc, temps)
    layers = {name: _words(case[key])[None, :] for name, key in (("air", "air_layer"), ("O2", "O2_layer"),
                                                                 ("O3", "O3_layer"))}
    return radiators.states(layers, temps, band_od)


@pytest.mark.parametrize("label,case", CASES, ids=[k for k, _ in CASES])
def test_radiator_and_accumulated_optics_are_the_oracles(parts, label, case):
    radiators, bands = parts
    rt = case["intermediates"]["radiative_transfer"]
    keep = rt["layers"]
    states = radiator_states(radiators, bands, case)
    band = np.zeros(radiators.nbins, dtype=bool)
    band[bands.bins] = True
    for name in ("air", "O2", "O3"):
        want = _table(rt["radiator_optical_depth"][name])
        got = states[name].od[0][keep]
        _same(got[:, ~band], want[:, ~band], f"{name} optical depth")
        _same(got[:, band], want[:, band], f"{name} band optical depth", MAC_OD_ULPS)
    total = radiators.accumulate(states)
    for attr, key in (("od", "accumulated_optical_depth"), ("ssa", "accumulated_single_scattering_albedo"),
                      ("g", "accumulated_asymmetry_factor")):
        want = _table(rt[key])
        got = getattr(total, attr)[0][keep]
        _same(got[:, ~band], want[:, ~band], key)
        _same(got[:, band], want[:, band], f"band {key}", MAC_DERIVED_ULPS)


def test_rayleigh_follows_the_power_law_break():
    grid = tuvx.WavelengthGrid([500.0, 540.0, 560.0, 600.0])
    xs = tuvx.RayleighCrossSection({}, grid, ".").values
    wrk = 1.0e-3 * grid.mid
    assert xs[2] == 4.02e-28 / wrk[2] ** 4.04
    assert xs[0] == 4.02e-28 / wrk[0] ** (3.6772 + 0.389 * wrk[0] + 0.09426 / wrk[0])
