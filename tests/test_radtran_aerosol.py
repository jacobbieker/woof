"""woof.radtran OPAC aerosol: Mie moments, uvspec's humidity rules, tables, mapping.

* The exact Legendre moments (Clebsch-Gordan series, woof.radtran.mie) agree
  with Gauss quadrature of the phase function where quadrature is exact.
* uvspec's relative-humidity law, layer mean and humidity-node search are
  held to uvspec's own printed values (tests/data/radtran/
  uvspec_aerosol_probes.json, written by tools/uvspec_oracle/probe_aerosol.py).
* woof's OPAC tables are held to libRadtran's ``mie`` tool run as a black box
  (tests/data/radtran/opac_mie_tool.json, tools/uvspec_oracle/opac_library.py
  ``fixture``); tolerances are the measured differences, which come from the
  mie tool's coarser size integration (woof's is converged to ~2e-6).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data" / "radtran"


def _need_data():
    from woof.radtran import data
    try:
        data.data_dir()
    except data.LibradtranDataMissing as exc:
        pytest.skip(str(exc))


def test_exact_moments_match_gauss_quadrature():
    from woof.radtran import mie
    x = np.exp(np.linspace(np.log(0.5), np.log(60.0), 50))
    w = np.exp(-0.5 * ((np.log(x) - np.log(5.0)) / 0.6) ** 2)
    m = complex(1.45, 0.01)
    _, _, a, b = mie.efficiencies(x, m)
    L = 40
    chi = mie.moments_from_coefficients(a, b, w, L)
    mu, wg = np.polynomial.legendre.leggauss(300)           # exact: |S|^2 has degree <= 2 N(x_max) < 600
    S1, S2 = mie.amplitudes(a, b, mu)
    p = (0.5 * (np.abs(S1) ** 2 + np.abs(S2) ** 2) * w[None]).sum(1)
    ref = 0.5 * (mie.legendre_table(L, mu) * (wg * p)[None]).sum(1)
    assert np.abs(chi - ref / ref[0]).max() < 1e-12


def test_clebsch_gordan_squares_sum_to_one():
    from woof.radtran import mie
    L = 24
    G = mie.cg_squares(12, L)
    for n in range(1, 13):
        for d in range(0, 13 - n):
            # sum over l of <n 1, n+d -1 | l 0>^2 = 1 (l runs |d|..2n+d <= L)
            assert abs(G[d][:, n - 1].sum() - 1.0) < 1e-13, (n, d)


def test_a_dry_layer_keeps_the_node_above():
    from woof.radtran.aerosol import select_nodes
    hum = [0, 50, 70, 80, 90, 95, 98, 99]
    pick = lambda rh: [hum[i] for i in select_nodes(np.array(rh, dtype=float), hum)]
    assert pick([5, 24.9, 25.1, 5]) == [0, 0, 50, 50]          # the dry node is never re-entered
    assert pick([59, 5]) == [50, 50]
    assert pick([99, 72, 20, 0.1, 30]) == [99, 70, 70, 70, 50]  # nearest dry: the node above
    assert pick([74.9, 75.1, 92.4, 92.6, 96.6, 98.6, 120]) == [70, 80, 90, 95, 98, 99, 99]
    # one node only (dry species): always it
    assert select_nodes(np.array([10.0, 90.0]), [0.0]).tolist() == [0, 0]


def test_relative_humidity_law_matches_uvspec():
    from woof.radtran.aerosol import relative_humidity
    r = json.loads((DATA / "uvspec_aerosol_probes.json").read_text())["rh_law"]
    rh = relative_humidity(np.array(r["t"]), np.array(r["n_h2o"]))
    uv = np.array(r["uvspec_rh"])
    # uvspec prints 7 significant digits; measured 4.3e-7
    assert np.abs(rh / uv - 1).max() < 1e-6


@pytest.mark.parametrize("name", ["ascending", "descending", "dry_wet_dry", "dry_25_dry", "jumps", "random",
                                  "random_dry"])
def test_layer_humidity_and_node_match_uvspec(name):
    from woof.radtran import aerosol
    from woof.radtran.atmosphere import Column
    r = json.loads((DATA / "uvspec_aerosol_probes.json").read_text())["profiles"][name]
    t = np.array(r["level_t"])
    col = Column(z=np.zeros_like(t), p=np.zeros_like(t), t=t, n_air=np.ones_like(t), dens={"H2O": np.array(r["level_n_h2o"])})
    rh = aerosol.layer_rh(col)
    uv = np.array(r["uvspec_layer_rh"])
    assert rh.size == uv.size
    assert np.abs(rh - uv).max() < 1e-4 * np.maximum(uv, 1.0).max()
    nodes = np.array([0, 50, 70, 80, 90, 95, 98, 99.0])
    got = nodes[aerosol.select_nodes(rh, nodes)]
    # exact ties (a layer mean on a midpoint between nodes) go either way
    # with uvspec's single-precision humidity; they are left out
    mids = 0.5 * (nodes[:-1] + nodes[1:])
    tie = np.min(np.abs(rh[:, None] - mids[None, :]), axis=1) < 1e-3
    assert (~tie).sum() >= 0.8 * tie.size
    assert got[~tie].tolist() == np.asarray(r["uvspec_node"])[~tie].tolist()


def test_mixture_rows_define_the_layer_above():
    _need_data()
    from woof.radtran import aerosol
    z, species, mass = aerosol.mixture("continental_average")
    # layer bottoms at 0.5, 1.0, 1.5 and 11.5 km above ground: rows 0, 1, 1, 11
    al = aerosol.mixture_layers("continental_average", np.array([0.5, 1.0, 1.5, 11.5, 40.0]))
    assert al.species == species
    np.testing.assert_array_equal(al.mass[0], mass[0])
    np.testing.assert_array_equal(al.mass[1], mass[1])
    np.testing.assert_array_equal(al.mass[2], mass[1])
    np.testing.assert_array_equal(al.mass[3], mass[np.searchsorted(z, 11.5) - 1])
    assert not np.any(al.mass[4])                                # above the top row


def test_thompson_numbers_become_opac_masses():
    _need_data()
    from woof.radtran import aerosol, atmosphere, model
    diag = model.Diagnostic(wavelengths=(550.0,), aerosol="thompson")
    z = np.array([[3000.0, 2000.0, 1000.0, 0.0]])[:, ::-1] * 1e-3      # surface first
    zt = z[:, ::-1]
    p = 1000.0 * np.exp(-zt / 8.0)
    col = atmosphere.model_columns(zt, p, np.full_like(zt, 280.0))
    nwfa = np.array([[1e9, 2e9, 3e9]])                                   # kg-1, surface first
    nifa = np.array([[1e6, 2e6, 3e6]])
    rho = np.array([[1.2, 1.1, 1.0]])
    aer = model.aerosol_layers(diag, col, 3, rho_dry=rho, nwfa=nwfa, nifa=nifa)
    assert aer.species == ("waso", "miam")
    L = col.z.shape[1] - 1
    m = aer.mass[0]
    assert not np.any(m[:L - 3])                                         # nothing above the model top
    np.testing.assert_allclose(m[L - 3:, 0], (nwfa * rho)[0, ::-1] * aerosol.particle_mass("waso", 50.0))
    np.testing.assert_allclose(m[L - 3:, 1], (nifa * rho)[0, ::-1] * aerosol.particle_mass("miam", 0.0))
    # OPAC water-soluble at 50 %: rho 1.42 g cm-3, lognormal 0.0262 um, sigma 2.24 (cut 0.006-25 um)
    v = 4.0 / 3.0 * np.pi * 0.0262 ** 3 * np.exp(4.5 * np.log(2.24) ** 2)    # untruncated mean volume, um3
    # the 0.006 um cut removes 3.4 % of the particles (1.83 sigma below the mode): +3.5 % per particle
    assert abs(aerosol.particle_mass("waso", 50.0) / (1.42 * v * 1e-12) - 1.035) < 0.005


#: species -> (ext, ssa, chi) relative / absolute tolerance, measured against
#: the mie tool (tools/uvspec_oracle/opac_library.py fixture) x ~2.
MIE_TOL = {}


def _mie_cases():
    f = DATA / "opac_mie_tool.json"
    if not f.exists():
        return []
    rec = json.loads(f.read_text())
    return [(s, i) for s in rec["species"] for i in range(len(rec["species"][s]["rows"]))]


@pytest.mark.parametrize("species,row", _mie_cases())
def test_tables_match_the_mie_tool(species, row):
    _need_data()
    from woof.radtran import aerosol
    rec = json.loads((DATA / "opac_mie_tool.json").read_text())
    r = rec["species"][species]["rows"][row]
    tol = rec["species"][species]["tolerance"]
    il = int(np.argmin(np.abs(aerosol.wavelengths_um(species) - r["lambda_um"])))
    ih = [n.rh for n in aerosol.species_nodes(species)].index(r["rh"])
    ext, ssa, chi = aerosol.node_optics(species, il, ih)
    n = len(r["chi"])
    assert abs(ext / r["ext"] - 1) <= tol["ext"], (ext, r["ext"])
    assert abs(ssa / r["ssa"] - 1) <= tol["ssa"], (ssa, r["ssa"])
    assert np.abs(chi[:n] - np.array(r["chi"])).max() <= tol["chi"]
