"""Fast-TUV photolysis (phot_opt = 3) against WRF v4.7.1's ftuv_driver, word for word.

The oracle (tools/chem_wrf471_oracle/ftuv) compiles chem/module_wave_data.F,
module_ftuv_subs.F and module_ftuv_driver.F whole and unchanged at -O0, runs
ftuv_init and then ftuv_driver on a 6 x 2 column grid of 30 levels for five
cases (chem_opt 2 and 1, aer_ra_feedback 0 and 1, five dates and times, day,
twilight and night columns, cloudy and saturated columns), and writes every
ph_* field the RADM2 packages read.  woof/core/chem_ftuv.py (the generated
woof/core/kernels/chem_ftuv.cu) must reproduce every word.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/data/oracles/chem/ftuv"
#: ftuv_driver's ph_* dummy positions (1-based, the oracle's ph rows) of
#: woof.core.chem_ftuv.PHOTOLYSIS, in that order.
ORACLE_ROW = {"no2": 1, "o31d": 2, "o33p": 3, "hno2": 4, "hno3": 5, "hno4": 6, "no3o2": 7,
              "no3o": 8, "h2o2": 9, "ch2om": 10, "ch2or": 11, "ch3cho": 12, "ch3o2h": 13,
              "o2": 14, "ch3coo2h": 15, "ch3coch3": 16, "hcocho": 17, "hcochob": 18,
              "ch3cocho": 19, "hcochest": 20, "ch3ono2": 21, "macr": 22, "ch3coc2h5": 23,
              "cl2": 45, "hocl": 46, "fmcl": 47, "pan": 25, "n2o5": 41}
CASES = ("case1", "case2", "case3", "case4", "case5")
P_O3, P_QV, P_QC = 6, 2, 3  # stub_ftuv.F90 (radm2sorg order)


def _load(case):
    d = FIXTURES / case
    out = {}
    for line in (d / "MANIFEST.txt").read_text().splitlines():
        parts = line.split()
        name, kind, nd = parts[0], parts[1], int(parts[2])
        shape = [int(x) for x in parts[3:3 + nd]]
        a = np.fromfile(d / f"{name}.bin", dtype=np.float32 if kind == "f4" else np.int32)
        out[name] = a.reshape(shape[::-1]) if shape else a[0]
    return out


def test_fixture_checksums():
    lines = (FIXTURES / "oracle-sha256sums.txt").read_text().split("\n")
    seen = 0
    for line in lines:
        if not line.strip():
            continue
        digest, rel = line.split()
        assert hashlib.sha256((FIXTURES / rel).read_bytes()).hexdigest() == digest, rel
        seen += 1
    assert seen == 5 * 21  # 20 arrays + MANIFEST.txt per case


def _cols(a):
    """Fortran (nx, k, ny) as read, (ny, k, nx) -> (k, ny, nx)."""
    return np.ascontiguousarray(np.transpose(a, (1, 0, 2)))


@pytest.mark.parametrize("case", CASES)
def test_ftuv_driver_word_for_word(case):
    cp = pytest.importorskip("cupy")
    from woof.core.chem_ftuv import PHOTOLYSIS, launch_ftuv
    f = _load(case)
    nz = 30
    g = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
    fields = {
        "t_phy": g(_cols(f["t_phy"])[:nz]), "p_phy": g(_cols(f["p_phy"])[:nz]),
        "rho_phy": g(_cols(f["rho_phy"])[:nz]),
        "p8w": g(_cols(f["p8w"])), "t8w": g(_cols(f["t8w"])), "z_at_w": g(_cols(f["z_at_w"])),
        "qv": g(_cols(f["moist"][P_QV - 1])[:nz]), "qc": g(_cols(f["moist"][P_QC - 1])[:nz]),
        "o3": g(_cols(f["chem"][P_O3 - 1])[:nz]),
        "xlat": g(f["xlat"]), "xlong": g(f["xlong"]),
    }
    optics = None
    if int(f["aer_ra_feedback"]) == 1:
        optics = [g(_cols(f[name][n])[:nz]) for name in ("tauaer", "waer", "gaer") for n in range(4)]
    got = launch_ftuv(fields, chem_opt=int(f["chem_opt"]), julday=int(f["julday"]),
                      gmt=float(f["gmt"]), curr_secs=float(f["curr_secs"]),
                      aer_ra_feedback=int(f["aer_ra_feedback"]), aerosol_optics=optics)
    bad = []
    for name in PHOTOLYSIS:
        exp = _cols(f["ph"][ORACLE_ROW[name] - 1])[:nz]
        g_ = got[name].get()
        if not np.array_equal(g_.view(np.uint32), exp.view(np.uint32)):
            n = int(np.sum(g_.view(np.uint32) != exp.view(np.uint32)))
            bad.append((name, n))
    assert not bad, f"{case}: words differ in {bad}"


def test_generated_block_is_fresh_against_the_pinned_source():
    """chem_ftuv.cu's generated block is regenerated from the pinned sources."""
    import subprocess
    import sys
    tree = Path(os.environ.get("WOOF_WRF471_STOCK_TREE",
                               str(Path.home() / "Development/wrf-src/WRF-4.7.1")))
    if not (tree / "chem" / "module_ftuv_driver.F").is_file():  # pragma: no cover
        pytest.skip(f"stock WRF v4.7.1 tree not present at {tree}")
    subprocess.run([sys.executable, str(ROOT / "tools/chem_ftuv/transcribe_ftuv.py"),
                    str(tree), "--check"], check=True)
