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
from pathlib import Path

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


# -- mixactivate: the column driver ---------------------------------------------------------------------

MIX = optin.optional_root() / "wrfchem_pnnl" / "oracle" / "fixtures" / "mixactivate"
MIX_DIMS = (1, 5, 1, 3, 1, 21, 0, 5, 0, 3, 1, 21, 1, 4, 1, 2, 1, 20)


def _case(directory):
    out = {}
    for line in (directory / "MANIFEST.txt").read_text().splitlines():
        parts = line.split()
        dims = tuple(int(x) for x in parts[3:])
        dtype = np.dtype("<f4" if parts[1] == "f4" else "<i4")
        out[parts[0]] = np.fromfile(directory / f"{parts[0]}.bin", dtype=dtype).reshape(dims, order="F")
    return out


def test_mixactivate_fixtures_are_the_ones_the_build_hashed():
    root = Path(str(MIX))
    for line in (root / "oracle-sha256sums.txt").read_text().splitlines():
        digest, rel = line.split()
        assert hashlib.sha256((root / rel.lstrip("*")).read_bytes()).hexdigest() == digest, rel


@requires_gpu
@pytest.mark.parametrize("config", ["modal", "sectional"])
def test_mixactivate_matches_the_oracle_word_for_word(config):
    optin.load("wrfchem_pnnl")
    from woof_wrfchem_pnnl import activate as act
    reason = act.byo_reason()
    if reason is not None:
        pytest.skip(reason)
    f = _case(MIX / config)
    msectional, ntype_aer, nphase_aer, _num_chem = (int(x) for x in f["ints"])
    desc = {"msectional": msectional, "ntype_aer": ntype_aer, "nphase_aer": nphase_aer, "ai_phase": 1,
            "cw_phase": 2, "ncomp_aer": f["ncomp_aer"], "nsize_aer": f["nsize_aer"],
            "numptr_aer": f["numptr_aer"], "massptr_aer": f["massptr_aer"], "waterptr_aer": f["waterptr_aer"],
            **{k: f[k] for k in ("dlo_sect", "dhi_sect", "sigmag_aer", "dpvolmean_aer", "dens_aer", "mw_aer")}}
    fields = {k: f[k] for k in ("qv", "qc", "qi", "temp", "w", "rho", "zm", "dz8w", "p_at_w", "t_at_w", "kvh")}
    fields["qndrop3d"] = f["qndrop3d_0"]
    fields["cldfra_old"] = f["cldfra_old_0"]
    chem = f["chem_0"]
    hygro = f["hygro"].reshape(4, 20, 2, 8, 2, order="F")
    bad = []
    for call in (1, 2):
        fields["cldfra"] = np.full_like(fields["qv"], -9.0)
        got = act.launch_mixactivate(MIX_DIMS, desc, fields, chem=chem, hygro=hygro, ddvel=f["ddvel"],
                                     dtstep=float(f["dtstep"]), ktau=call)
        for name in ("chem", "qndrop3d", "cldfra", "qsrflx", "ccn", "nsource"):
            want = f[f"{name}_{call}"]
            diff = got[name].view(np.uint32) != np.ascontiguousarray(want, np.float32).view(np.uint32)
            if diff.any():
                bad.append((call, name, int(diff.sum()), int(diff.size)))
        # the next call: as the oracle does it (its own qc edits come from the fixture's qc being re-read)
        chem, fields["qndrop3d"] = got["chem"], got["qndrop3d"]
        fields["cldfra_old"] = got["cldfra"]
        if call == 1:
            fields["qc"] = _second_qc(fields["qc"])
    assert not bad, bad


def _second_qc(qc):
    """run_mixactivate.F90's cloud edit between the calls (exact: adds a literal, or zeroes)."""
    qc = qc.copy()
    for j in range(1, 3):
        for i in range(1, 5):
            for k in range(1, 21):
                if (i + j + k) % 3 == 0:
                    qc[i, k - 1, j] = np.float32(qc[i, k - 1, j] + np.float32(2.0e-5))
                if (i + 2 * j + k) % 5 == 0:
                    qc[i, k - 1, j] = 0.0
    return qc
