"""WRF-Chem's RADM2 background gas profile (chem_in_opt = 0) and boundary
value against WRF v4.7.1, word for word.

tools/chem_wrf471_oracle/gasprofile/run_gasprofile.F90 calls the byte-copied
input_chem_profile / make_chem_profile / vinterp_chem / bdy_chem_value
(chem/module_input_chem_data.F, its module DATA included) for chem_opt 1 and
2: real.exe's whole entry on 6 x 2 grids of 1 to 60 levels (terrain below sea
level to 4.5 km, tops 300 m to 32 km), make_chem_profile on hand-picked mass
heights (every profile level, its float32 neighbours, midpoints, far below
and above), bdy_chem_value at the same heights, the gas_ic_opt = 101 path
the port refuses, and bdy_chem_value_sorgam (RADM2SORG aerosol boundary,
aer_bc_opt = 1) for its 38 rows at those heights plus every zfa_bdy level and
its float32 neighbours, in four atmospheres whose convfac is made as
flow_dep_bdy_chem makes it.  Pure NumPy: runs on every host.
"""
from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from woof.verify.chem_oracle import load

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "tests/data/oracles/chem/gasprofile"
PINS = REPO / "tools/chem_wrf471_oracle/gasprofile/SOURCES.sha256"
NGAS = 42


@pytest.fixture(scope="module")
def cases():
    return load(ROOT)


def _bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _assert_words(got: dict, want: np.ndarray, what: str):
    """``want[..., n]`` is gas n (Registry order); compare every word."""
    from woof.core.chem_gas_profile import GASES
    bad = []
    for n, name in enumerate(GASES):
        g, w = _bits(got[name]), _bits(want[..., n])
        assert g.shape == w.shape, (name, g.shape, w.shape)
        if not np.array_equal(g, w):
            bad.append((name, int(np.sum(g != w))))
    assert not bad, f"{what}: words differ in {bad}"


def test_fixture_checksums_and_coverage(cases):
    for line in (ROOT / "oracle-sha256sums.txt").read_text().splitlines():
        digest, rel = line.split()
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    nzs = sorted(c["chem"].shape[1] - 1 for k, c in cases.items() if k.startswith("ic_c1"))
    assert nzs[0] == 1 and nzs[-1] == 60
    z = cases["bdy"]["z"]
    zprof = 0.5 * (cases["tables"]["zfa"][:-1] + cases["tables"]["zfa"][1:])
    assert z.min() < 0 < zprof[0] and z.max() > zprof[-1]
    assert set(zprof) <= set(z)


def test_tables_are_the_oracles_words(cases):
    from woof.core import chem_gas_profile as gp
    t = cases["tables"]
    for name in ("xl", "zfa", "zfa_bdy", "dens"):
        want = t[name]
        got = getattr(gp, name.upper())
        assert np.array_equal(_bits(got), _bits(want)), name
    assert _bits(gp.GRAV) == _bits(t["grav"])
    assert _bits(gp.SO4VAPTOAER) == _bits(t["so4vaptoaer"])
    assert np.array_equal(gp.IREF, t["iref"][:NGAS])
    assert np.array_equal(_bits(gp.FRACREF), _bits(t["fracref"][:NGAS]))
    assert int(t["numgas_radm2"]) == int(t["numgas_radm2sorg"]) == NGAS + 1
    assert int(t["p_nu0_radm2sorg"]) > 1


def _ic_cases(cases, prefix="ic_"):
    return sorted(k for k in cases if k.startswith(prefix))


def test_initial_profile_matches_input_chem_profile(cases):
    from woof.core.chem_gas_profile import radm2_background_profile, zsigf_from_geopotential
    names = _ic_cases(cases)
    assert len(names) == 7
    for case in names:
        c = cases[case]
        # Fortran (i, k, j) -> (k, j, i)
        ph = np.transpose(c["ph_1"], (1, 2, 0))
        phb = np.transpose(c["phb"], (1, 2, 0))
        chem = np.transpose(c["chem"], (1, 2, 0, 3))
        zf = zsigf_from_geopotential(ph, phb)
        got = radm2_background_profile(zf, chem_opt=int(c["chem_opt"]), include_top=True)
        _assert_words(got, chem[..., 1:NGAS + 1], case)
        mass = radm2_background_profile(zf, chem_opt=int(c["chem_opt"]))
        for name, v in mass.items():
            assert np.array_equal(_bits(v), _bits(got[name][:-1])), (case, name)
        # rows after ho2 (radm2sorg aerosols) are untouched by the profile
        assert np.all(chem[..., NGAS + 1:] == np.float32(-7.0))


def test_make_chem_profile_edge_heights(cases):
    from woof.core.chem_gas_profile import radm2_profile_on_mass_levels
    for case in ("mk_c1", "mk_c2"):
        c = cases[case]
        got = radm2_profile_on_mass_levels(c["z"][:, 0, 0])
        _assert_words(got, c["chem"][:, 0, 0, 1:NGAS + 1], case)


@pytest.mark.parametrize("chem_opt", [1, 2])
def test_boundary_value_matches_bdy_chem_value(cases, chem_opt):
    from woof.core.chem_gas_profile import radm2_boundary_value
    c = cases["bdy"]
    got = radm2_boundary_value(c["z"], chem_opt=chem_opt)
    _assert_words(got, c["chem"][:, :, chem_opt - 1], f"bdy chem_opt={chem_opt}")


def test_sulf_boundary_differs_only_by_so4vaptoaer(cases):
    c = cases["bdy"]["chem"]
    assert np.array_equal(c[:, [0] + list(range(2, NGAS)), 0], c[:, [0] + list(range(2, NGAS)), 1])
    assert not np.array_equal(c[:, 1, 0], c[:, 1, 1])


def test_shape_agnostic(cases):
    from woof.core.chem_gas_profile import radm2_background_profile, radm2_boundary_value
    c = cases["ic_c1_nz44"]
    zf = np.transpose((c["ph_1"] + c["phb"]) / np.float32(9.80622), (1, 2, 0))  # (k, j, i)
    nzp1 = zf.shape[0]
    grid = radm2_background_profile(zf)
    flat = radm2_background_profile(zf.reshape(nzp1, -1))
    hex3 = radm2_background_profile(zf.reshape(nzp1, 1, -1))
    col = radm2_background_profile(zf[:, 1, 3])
    for name, v in grid.items():
        assert v.shape == (nzp1 - 1,) + zf.shape[1:] and v.dtype == np.float32
        assert np.array_equal(_bits(flat[name]), _bits(v.reshape(nzp1 - 1, -1))), name
        assert np.array_equal(_bits(hex3[name]), _bits(v.reshape(nzp1 - 1, 1, -1))), name
        assert np.array_equal(_bits(col[name]), _bits(v[:, 1, 3])), name
    zm = cases["bdy"]["z"]
    b1 = radm2_boundary_value(zm)
    b2 = radm2_boundary_value(np.tile(zm, (3, 1)).T)
    for name, v in b1.items():
        assert np.array_equal(_bits(b2[name][:, 2]), _bits(v)), name


def test_pnnl_options_are_refused_and_do_differ(cases):
    from woof.core.chem_gas_profile import (GASES, radm2_background_profile,
                                            radm2_boundary_value)
    zf = np.linspace(0.0, 20000.0, 11, dtype=np.float32)
    with pytest.raises(ValueError, match="GAS_IC_PNNL"):
        radm2_background_profile(zf, gas_ic_opt=101)
    with pytest.raises(ValueError, match="GAS_BC_PNNL"):
        radm2_boundary_value(zf, gas_bc_opt=101)
    with pytest.raises(ValueError, match="GAS_IC_PNNL"):
        radm2_boundary_value(zf, gas_ic_opt=101)
    with pytest.raises(ValueError, match="chem_opt"):
        radm2_background_profile(zf, chem_opt=5)
    # In WRF the 101 path rewrites the sulf profile only.
    base = cases["ic_c1_nz20"]["chem"]
    pnnl = cases["pnnl_ic_c1_nz20"]["chem"]
    for n, name in enumerate(GASES):
        same = np.array_equal(base[..., n + 1], pnnl[..., n + 1])
        assert same != (name == "sulf"), name


def test_sorgam_constants_are_the_oracles_words(cases):
    from woof.core.chem_gas_profile import _SORGAM as c
    t = cases["sorgam_tables"]
    assert float(t["pirs_lo"]) == 0.0  # REAL*8 pirs holds the REAL literal
    pairs = {"pirs": c["pirs"], "f6dpi": c["f6dpi"], "f6dpim9": c["f6dpim9"],
             "conmin": c["conmin"], "epsilc": c["conmin"]}
    for r in ("so4", "nh4", "no3", "org", "soil", "seas", "anth"):
        pairs[f"{r}fac"] = c[f"{r}fac"]
    for name, got in pairs.items():
        assert _bits(got) == _bits(t[name]), name
    modes = ("n", "a", "c")
    for name, key in (("dgini3", "dgini{}3"), ("xxlsg", "xxlsg{}"), ("l2sgini", "l2sgini{}"),
                      ("e1", "e1{}"), ("es36", "es36{}")):
        got = np.array([c[key.format(m)] for m in modes], dtype=np.float32)
        assert np.array_equal(_bits(got), _bits(t[name])), name
    got = np.array([c["m3fac_na"], c["m3fac_c"]], dtype=np.float32)
    assert np.array_equal(_bits(got), _bits(t["m3fac"]))


def _assert_aer_words(got: dict, want: np.ndarray, what: str):
    """``want[..., n]`` is aerosol n (AEROSOLS order); compare every word."""
    from woof.core.chem_gas_profile import AEROSOLS
    bad = []
    for n, name in enumerate(AEROSOLS):
        g, w = _bits(got[name]), _bits(want[..., n])
        assert g.shape == w.shape, (name, g.shape, w.shape)
        if not np.array_equal(g, w):
            bad.append((name, int(np.sum(g != w))))
    assert not bad, f"{what}: words differ in {bad}"


def test_sorgam_fixture_coverage(cases):
    from woof.core.chem_gas_profile import AEROSOLS
    from woof.core.chem_sorgam import AEROSOLS as SORGAM_ROWS
    c = cases["sorgam_bdy"]
    assert AEROSOLS == SORGAM_ROWS and len(AEROSOLS) == 38
    assert int(c["p_corn"]) - int(c["p_so4aj"]) + 1 == 38 == c["chem"].shape[1]
    assert int(c["aer_bc_default"]) == 1 and int(c["aer_bc_pnnl"]) == 101
    zb = cases["tables"]["zfa_bdy"]
    zprof = (0.5 * (zb[:-1] + zb[1:])).astype(np.float32)
    z = c["z"]
    assert set(zprof) <= set(z)
    assert set(np.nextafter(zprof, np.float32(-np.inf))) <= set(z)
    assert set(np.nextafter(zprof, np.float32(np.inf))) <= set(z)
    assert z.min() < zprof[0] and z.max() > zprof[-1]
    # alt 0.8 .. ~20 m3/kg, convfac ~50 .. ~1 mol/m3 (surface to 30+ km)
    assert c["alt"].min() < 1.0 and c["alt"].max() > 10.0
    assert c["convfac"].min() < 2.0 and c["convfac"].max() > 40.0


def test_sorgam_boundary_value_matches_bdy_chem_value_sorgam(cases):
    from woof.core.chem_gas_profile import sorgam_boundary_value
    c = cases["sorgam_bdy"]
    got = sorgam_boundary_value(c["z"], c["alt"], c["convfac"])
    _assert_aer_words(got, c["chem"], "sorgam_bdy")
    # the oracle's own per-row structure: untouched rows are conmin
    conmin = np.float32(1.e-16)
    for name in ("naaj", "orgpaj", "ecj", "p25i", "antha", "seas", "soila"):
        assert np.all(got[name] == conmin), name


def test_sorgam_shape_agnostic(cases):
    from woof.core.chem_gas_profile import sorgam_boundary_value
    c = cases["sorgam_bdy"]
    n = c["z"].size // 8 * 8
    z, alt, cf = (c[k][:n] for k in ("z", "alt", "convfac"))
    flat = sorgam_boundary_value(z, alt, cf)
    grid = sorgam_boundary_value(z.reshape(4, 2, -1), alt.reshape(4, 2, -1), cf.reshape(4, 2, -1))
    hex3 = sorgam_boundary_value(z.reshape(4, 1, -1), alt.reshape(4, 1, -1), cf.reshape(4, 1, -1))
    one = sorgam_boundary_value(z[7], alt[7], cf[7])
    scal = sorgam_boundary_value(z.reshape(4, -1), alt[5], cf[5])
    ref = sorgam_boundary_value(z.reshape(4, -1), np.full(n, alt[5]).reshape(4, -1),
                                np.full(n, cf[5]).reshape(4, -1))
    for name, v in flat.items():
        assert v.shape == (n,) and v.dtype == np.float32
        assert np.array_equal(_bits(grid[name]), _bits(v.reshape(4, 2, -1))), name
        assert np.array_equal(_bits(hex3[name]), _bits(v.reshape(4, 1, -1))), name
        assert one[name].shape == () and _bits(one[name]) == _bits(v[7]), name
        assert np.array_equal(_bits(scal[name]), _bits(ref[name])), name


def test_sorgam_aer_bc_pnnl_is_refused():
    from woof.core.chem_gas_profile import sorgam_boundary_value
    z = np.linspace(0.0, 20000.0, 5, dtype=np.float32)
    with pytest.raises(ValueError, match="AER_BC_PNNL"):
        sorgam_boundary_value(z, 1.0, 40.0, aer_bc_opt=101)
    with pytest.raises(ValueError, match="aer_bc_opt = 2"):
        sorgam_boundary_value(z, 1.0, 40.0, aer_bc_opt=2)
    with pytest.raises(ValueError, match="NaN"):
        sorgam_boundary_value(np.float32(np.nan), 1.0, 40.0)


def test_import_is_cupy_free():
    for rel in ("woof/core/chem_gas_profile.py", "woof/core/chem_gas_profile_data.py"):
        tree = ast.parse((REPO / rel).read_text())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods.add((node.module or "").split(".")[0])
        assert mods <= {"__future__", "fractions", "numpy", "woof"}, (rel, mods)
    code = ("import sys, woof; before = set(sys.modules); "
            "import woof.core.chem_gas_profile; "
            "new = set(sys.modules) - before; "
            "assert not any(m.split('.')[0] in ('cupy', 'mlx') for m in new), sorted(new)")
    env = dict(os.environ, PYTHONPATH=str(REPO))
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=REPO)


def test_data_module_is_fresh_against_the_stock_tree():
    """The pinned WRF files match WOOF_WRF471_STOCK_TREE and the transcribed
    data module is what tools/chem_gas_profile/transcribe.py writes from it."""
    tree = Path(os.environ.get("WOOF_WRF471_STOCK_TREE",
                               str(Path.home() / "Development/wrf-src/WRF-4.7.1")))
    if not (tree / "chem" / "module_input_chem_data.F").is_file():  # pragma: no cover
        pytest.skip(f"stock WRF v4.7.1 tree not present at {tree}")
    for line in PINS.read_text().splitlines():
        if line and not line.startswith("#"):
            digest, rel = line.split()
            assert hashlib.sha256((tree / rel).read_bytes()).hexdigest() == digest, rel
    subprocess.run([sys.executable, str(REPO / "tools/chem_gas_profile/transcribe.py"),
                    str(tree), "--check"], check=True, cwd=REPO)
