"""TUV-x port, piece 3 (woof.chem_tuvx): spherical geometry and slant columns against the oracle's intermediates.

For the first two cases of each column layout (tests/data/tuvx_oracle/fixtures.json, case "intermediates" /
"radiative_transfer") the oracle records the spherical geometry TUV-x builds for the case's solar zenith angle
(src/spherical_geometry.F90 set_parameters: ``nid`` on every edge, ``dsdh`` rows of the kept edges, both counted
from the top) and the air and O2 vertical and slant columns (air_mass).  The "twilight" layout puts the sun at
and below the horizon, where the beam passes a screening height or never reaches an edge (nid = -1).

The geometry takes one libm call per column, sin of the zenith angle; everything else is sqrt and the four
operations.  On glibc (the oracle's libm) every word must be the oracle's; elsewhere the integers must agree and
each float stay within ``MAC_ULPS`` of it (the measured spread of Apple libm's sin through the slant paths).
The inputs are in the fixtures, so the test needs no TUV-x data.
"""
import json
import platform
from pathlib import Path

import numpy as np
import pytest

from woof import chem_tuvx as tuvx

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = json.loads((ROOT / "tests" / "data" / "tuvx_oracle" / "fixtures.json").read_text())
GLIBC = platform.system() == "Linux" and platform.libc_ver()[0] == "glibc"
#: largest ulp distance from the oracle allowed off glibc
MAC_ULPS = 1


def _cases():
    for fixture in FIXTURES["fixtures"]:
        for n, case in enumerate(fixture["cases"]):
            if "intermediates" in case:
                yield f"{fixture['layout']}-{n}", case


def _words(values):
    return np.array([float.fromhex(w) for w in values], dtype=np.float64)


def _ulps(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.abs(a.view(np.int64) - b.view(np.int64))


def _same(got, want, what):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    assert got.shape == want.shape, what
    if GLIBC:
        assert got.tobytes() == want.tobytes(), (what, np.argwhere(got != want)[:5])
    else:
        assert int(_ulps(got, want).max(initial=0)) <= MAC_ULPS, (what, int(_ulps(got, want).max()))


CASES = list(_cases())


@pytest.mark.parametrize("label,case", CASES, ids=[k for k, _ in CASES])
def test_geometry_and_columns_are_the_oracles(label, case):
    inter = case["intermediates"]
    rt = inter["radiative_transfer"]
    edges = _words(case["height_edges_km"])
    sza = float.fromhex(case["solar_zenith_angle_deg"])
    geometry = tuvx.spherical_geometry(edges, sza)
    assert geometry.nid[0].tolist() == rt["nid"]
    # dsdh rows are kept at the kept edge indices, which count from the top in the geometry
    want = np.array([_words(row) for row in rt["dsdh"]])
    _same(geometry.dsdh[0][inter["height_edges"]], want, "dsdh")
    for species, layer in (("air", "air_layer"), ("o2", "O2_layer")):
        vcol, scol = tuvx.air_mass(geometry, tuvx.exo_layer_columns(_words(case[layer])))
        _same(vcol[0], _words(rt[f"{species}_vertical_column"]), f"{species} vertical column")
        _same(scol[0], _words(rt[f"{species}_slant_column"]), f"{species} slant column")


def test_a_batch_gives_each_columns_words():
    """Columns on the leading axis are independent: a batch of every case equals the cases one by one."""
    cases = [c for _, c in CASES if len(c["height_edges_km"]) == len(CASES[0][1]["height_edges_km"])]
    edges = np.array([_words(c["height_edges_km"]) for c in cases])
    sza = np.array([float.fromhex(c["solar_zenith_angle_deg"]) for c in cases])
    batch = tuvx.spherical_geometry(edges, sza)
    air = np.array([tuvx.exo_layer_columns(_words(c["air_layer"]))[0] for c in cases])
    vb, sb = tuvx.air_mass(batch, air)
    for n, c in enumerate(cases):
        one = tuvx.spherical_geometry(edges[n], sza[n])
        assert one.nid[0].tolist() == batch.nid[n].tolist()
        assert one.dsdh[0].tobytes() == batch.dsdh[n].tobytes()
        v, s = tuvx.air_mass(one, air[n])
        assert v[0].tobytes() == vb[n].tobytes() and s[0].tobytes() == sb[n].tobytes()


def test_a_sun_below_the_screening_height_never_reaches_the_surface():
    edges = np.linspace(0.0, 120.0, 121)
    geometry = tuvx.spherical_geometry(edges, 100.0)
    assert geometry.nid[0, -1] == -1
    _vcol, scol = tuvx.air_mass(geometry, tuvx.exo_layer_columns(np.ones(120)))
    assert scol[0, 0] == tuvx.LARGEST


def test_mismatched_columns_are_refused():
    with pytest.raises(tuvx.TuvxPortError):
        tuvx.spherical_geometry(np.zeros((2, 5)) + np.arange(5), [10.0])
    geometry = tuvx.spherical_geometry(np.arange(5.0), 10.0)
    with pytest.raises(tuvx.TuvxPortError):
        tuvx.air_mass(geometry, np.ones(4))
