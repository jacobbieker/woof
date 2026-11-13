"""TUV-x port, piece 6 (woof.chem_tuvx.delta_eddington): the delta-Eddington radiation field against the oracle.

For the first two cases of each layout the oracle records (tests/data/tuvx_oracle/fixtures.json, case
"intermediates") the direct, diffuse-up and diffuse-down actinic flux on five height edges for every wavelength
bin, as core%get_radiation_field returns them: for unit extraterrestrial flux, times the Earth-Sun distance factor
the harness passes to core%run (radiation_field_t apply_scale_factor).  The port solves from its own geometry
(piece 3), bands (piece 4) and radiators (piece 5); the "twilight" cases cover a sun below the horizon (mu < 0,
beams that never arrive).

The solver calls cos and exp, on top of the libm calls upstream.  On glibc (the oracle's libm) every word must be
the oracle's.  Elsewhere the libm's last-bit differences (1 ulp where they enter) pass through the two-stream
tridiagonal solve, which amplifies them where a bin is absorbed to nothing (a Schumann-Runge bin at the surface
carries ~1e-116 of its top-of-atmosphere flux).  Off glibc each part is therefore held to ``MAC_RTOL`` of the bin's
largest total actinic flux on the kept edges.  Measured on Apple silicon: 1664 of 14040 words differ, the worst by
1.8e-13 of that scale.  (A correctly rounded libm does not close the gap: glibc's exp and log are not correctly
rounded either, and still 2 to 19 words per case differ.)  The test
runs where the TUV-x data are cached or WOOF_TUVX_DATA names them.
"""
import platform
from pathlib import Path

import numpy as np
import pytest

from woof import chem_tuvx as tuvx

from test_chem_tuvx_radiators import CASES, CONFIG, _root, _table, _words, radiator_states

GLIBC = platform.system() == "Linux" and platform.libc_ver()[0] == "glibc"
MAC_RTOL = 1.0e-12


@pytest.fixture(scope="module")
def parts():
    root = _root()
    if root is None:
        pytest.skip("the pinned TUV-x data are not cached here (woof.chem_tuvx_data.ensure() fetches them)")
    grid = tuvx.read_grid_csv(Path(root) / "data/grids/wavelength/combined.grid")
    return (tuvx.Radiators(CONFIG, grid, root=root), tuvx.LaSrBands(CONFIG["O2 absorption"], grid, root=root),
            tuvx.surface_albedo(CONFIG, grid))


def radiation_field(parts, case):
    radiators, bands, albedo = parts
    states = radiator_states(radiators, bands, case)
    geometry = tuvx.spherical_geometry(_words(case["height_edges_km"]), float.fromhex(case["solar_zenith_angle_deg"]))
    field = tuvx.delta_eddington(radiators.accumulate(states), geometry, albedo)
    return field.scaled(float.fromhex(case["earth_sun_distance_au"]))


@pytest.mark.parametrize("label,case", CASES, ids=[k for k, _ in CASES])
def test_the_radiation_field_is_the_oracles(parts, label, case):
    inter = case["intermediates"]
    field = radiation_field(parts, case)
    total = sum(_table(inter[k]) for k in ("actinic_flux_direct", "actinic_flux_diffuse_up",
                                           "actinic_flux_diffuse_down"))
    scale = np.maximum(np.abs(total).max(axis=0), np.finfo(np.float64).tiny)
    for attr, key in (("fdr", "actinic_flux_direct"), ("fup", "actinic_flux_diffuse_up"),
                      ("fdn", "actinic_flux_diffuse_down")):
        got = getattr(field, attr)[0][inter["height_edges"]]
        want = _table(inter[key])
        assert got.shape == want.shape, key
        if GLIBC:
            assert got.tobytes() == want.tobytes(), (key, np.argwhere(got != want)[:5])
        else:
            assert float((np.abs(got - want) / scale).max()) <= MAC_RTOL, key


def test_tridiag_solves_a_small_system():
    rng = np.random.default_rng(3)
    n = 6
    a, c = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    b = 4.0 + rng.uniform(0, 1, n)
    r = rng.uniform(-1, 1, n)
    m = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    u = tuvx.tridiag(*(v[None, :, None] for v in (a, b, c, r)))[0, :, 0]
    np.testing.assert_allclose(m @ u, r, rtol=0, atol=1e-14)


def test_a_beam_that_never_arrives_takes_the_single_precision_literal():
    assert tuvx._NO_BEAM == float(np.float32(1.0e36)) != 1.0e36
