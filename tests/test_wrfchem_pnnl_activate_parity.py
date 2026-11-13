"""Abdul-Razzak and Ghan activation (the PNNL opt-in), bitwise against WRF v4.7.1.

``activate`` from phys/module_mixactivate.F, run by ``woof_wrfchem_pnnl.activate`` on the oracle's 112 cases:
modal and quasi-sectional, single updraft and Gaussian spectrum, smax prescribed or from maxsat, empty and
non-hygroscopic modes, the early returns.  The fixture is optional/wrfchem_pnnl/oracle/activate/build.sh
(gfortran -O0, glibc 2.39).  Every test skips, naming the rule, unless ``WOOF_ENABLE_WRFCHEM_PNNL=1`` (the
Battelle/PNNL terms) and, for the kernel, ``WOOF_WRF_SOURCE`` names your WRF v4.7.1 tree (erfc_num_recipes is
Numerical Recipes code, which WOOF never ships).
"""

from __future__ import annotations

import hashlib
import subprocess
import sys

import numpy as np
import pytest

from conftest import requires_gpu
from woof import optin

REASON = optin.unavailable_reason("wrfchem_pnnl")
pytestmark = pytest.mark.skipif(REASON is not None, reason=f"opt-in rule: {REASON}")

FIX = optin.optional_root() / "wrfchem_pnnl" / "oracle" / "fixtures" / "activate"
ARRAYS = ("na", "volc", "dlo_sect", "dhi_sect", "sigman", "hygro", "fn", "fs", "fm", "fluxn", "fluxs", "fluxm")
SCALARS = ("wbar", "sigw", "wdiab", "wminf", "wmaxf", "tair", "rhoair", "smax", "flux_fullact")


def _load(name):
    for line in (FIX / "cases" / "MANIFEST.txt").read_text().splitlines():
        parts = line.split()
        if parts[0] == name:
            dims = tuple(int(x) for x in parts[3:])
            dtype = np.float32 if parts[1] == "f4" else np.int32
            raw = np.fromfile(FIX / "cases" / f"{name}.bin", dtype=np.dtype(dtype).newbyteorder("<"))
            return raw.reshape(dims, order="F")
    raise KeyError(name)


def test_fixtures_are_the_ones_the_build_hashed():
    for line in (FIX / "oracle-sha256sums.txt").read_text().splitlines():
        digest, rel = line.split()
        assert hashlib.sha256((FIX / rel.lstrip("*")).read_bytes()).hexdigest() == digest, rel


def test_the_cases_reach_the_branches():
    ints = _load("ints")
    setup, ismax, msectional = ints[0], ints[1], ints[2]
    scalars = _load("scalars")
    assert ints.shape[1] == 112
    assert set(setup) == {1, 2, 3, 4, 5, 6, 7} and set(ismax) == {0, 1, 2}
    assert (msectional > 0).any() and (msectional == 0).any()
    sigw, wbar = scalars[1], scalars[0]
    assert ((sigw <= 1e-5) & (wbar > 0)).any() and ((sigw <= 1e-5) & (wbar <= 0)).any()
    assert ((sigw > 1e-5) & (wbar > 0)).any() and ((sigw > 1e-5) & (wbar < 0)).any()


def test_the_generated_block_is_the_pinned_files_translation():
    """The committed kernel is the transcriber's output (needs your WRF tree; nothing of it is committed)."""
    import os
    src = os.environ.get("WOOF_WRF_SOURCE")
    if not src:
        pytest.skip("WOOF_WRF_SOURCE is not set (the transcriber reads your WRF v4.7.1 tree)")
    tool = optin.optional_root() / "wrfchem_pnnl" / "tools" / "transcribe_activate.py"
    subprocess.run([sys.executable, str(tool), src, "--check"], check=True)


@requires_gpu
def test_activate_matches_the_oracle_word_for_word():
    pnnl = optin.load("wrfchem_pnnl")
    from woof_wrfchem_pnnl import activate as act
    reason = act.byo_reason()
    if reason is not None:
        pytest.skip(reason)
    del pnnl
    arrays, scalars, ints = _load("arrays"), _load("scalars"), _load("ints")
    cases = []
    for c in range(ints.shape[1]):
        case = {name: scalars[k, c] for k, name in enumerate(SCALARS) if name not in ("smax", "flux_fullact")}
        case.update({name: arrays[:, :, k, c] for k, name in enumerate(ARRAYS[:6])})
        # activate's intent(inout) outputs start where the oracle started them
        case.update({name: np.full((8, 2), -7.0, np.float32) for name in ARRAYS[6:]})
        case["flux_fullact"] = np.float32(-7.0)
        case["smax"] = None if ints[1, c] == 0 else scalars[7, c]
        case["msectional"], case["ntype_aer"] = int(ints[2, c]), int(ints[3, c])
        case["nsize_aer"] = (int(ints[4, c]), int(ints[5, c]))
        cases.append(case)
    got = act.launch_activate(cases)
    bad = []
    for k, name in enumerate(ARRAYS[6:], start=6):
        want = np.ascontiguousarray(arrays[:, :, k, :].transpose(2, 0, 1))
        diff = np.flatnonzero((got[name].view(np.uint32) != want.view(np.uint32)).reshape(len(cases), -1).any(1))
        if diff.size:
            bad.append((name, [int(ints[6, c]) for c in diff[:8]]))
    full = scalars[8]
    diff = np.flatnonzero(got["flux_fullact"].view(np.uint32) != full.view(np.uint32))
    if diff.size:
        bad.append(("flux_fullact", [int(ints[6, c]) for c in diff[:8]]))
    assert not bad, bad
