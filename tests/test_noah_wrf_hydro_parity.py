"""Noah's side of the WRF-Hydro exchange, against WRF v4.7.1 built -DWRF_HYDRO.

WRF configured with ``WRF_HYDRO=1`` compiles ``phys/`` with ``-DWRF_HYDRO``.
For ``sf_surface_physics=2`` that adds three planes to ``lsm``
(``module_sf_noahdrv.F:2607-2610``) and switches on six NDHMS edits in
``module_sf_noahlsm.F``:

* SMFLX adds the ponded head the routing handed back, ``SFHEAD1RT`` in mm, to
  ``PCPDRP`` before the infiltration test (``:2775``);
* SRT returns the infiltration excess ``INFXS1RT = RUNOFF1*DT*1000``
  (``:3719-3832``);
* SSTEP pushes the water the bottom layer could not hold (``WPLUS``) back up
  the profile and adds what the top layer cannot hold to ``INFXS1RT``
  (``:4029-4062``);

and the driver hands ``soldrain = RUNOFF2*DT*1000`` and ``infxsrt`` back to
WRF-Hydro on every ordinary land column (``:1134-1139``).  woof runs that build
as ``launch_noah(wrf_hydro=...)`` -> ``noah_column_hydro``.

``noah-lsm-wrf-hydro.csv`` is ``tools/noah_wrf461_oracle/build.sh``'s hydro
pass: the same pinned, byte-unmodified sources compiled with ``-DWRF_HYDRO``,
driven over the same 80 columns as ``noah-lsm.csv`` with one of five ponded
heads per column, and ``infxsrt``/``soldrain`` entering as negative sentinels
so a column lsm did not write is visible.  The build also runs a scratch copy
of ``module_sf_noahlsm.F`` with a print at each of the three branches that
write ``INFXS1RT`` and refuses to write a fixture unless every one is reached;
``wrf-hydro-coverage.txt`` is that receipt.

The WRF-Hydro routing (subsurface, overland, channel) that consumes these
planes is not ported; see docs/coupling/WRF-HYDRO-NEMO.md.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core.fp32_ulp import fp32_ulp_distance
from woof.verify.noah_oracle import (
    NOAH_ORACLE_DIR,
    OUTPUT_COLUMNS,
    SOIL_OUTPUT_COLUMNS,
    WRF_HYDRO_COLUMNS,
    load_noah_oracle,
    noah_port_outputs,
)

FIXTURE = "noah-lsm-wrf-hydro.csv"

#: Case 25 is unrunnable in WRF and in the port alike (subnormal exchange
#: coefficients drive PENMAN to NaN); tests/test_noah_wrf461_parity.py holds
#: it out of every table for the reason given there, and it is held out here.
HELD_OUT_CASES = (25,)

#: The five ponded heads ``run_lsm.F90`` cycles over the columns, in mm.
PONDED_MM = (0.0, 0.25, 3.0, 40.0, 400.0)

#: ``run_lsm.F90``'s sentinels for the two planes lsm writes.
INFXSRT_SENTINEL = -7.0
SOLDRAIN_SENTINEL = -3.0

#: How many times the instrumented build reached each WRF_HYDRO branch over
#: the fixture (SRT runs twice per two-pass SMFLX, so it counts calls, not
#: columns).  MEASURED by build.sh in Ubuntu 24.04 (gfortran 13.3.0, glibc
#: 2.39), 2026-11-07.
COVERAGE = {"SRT_INFXS": 83, "SSTEP_WPLUS": 4, "SSTEP_SPILL": 4}


def _fixture():
    return load_noah_oracle(FIXTURE)


def _ordinary_land(fixture) -> np.ndarray:
    """The columns lsm runs SFLX on: land (XLAND < 1.5), not sea ice, not
    land ice -- the ones ``module_sf_noahdrv.F:1134`` writes the planes on."""
    xland = fixture.inputs["xland"].reshape(-1)
    xice = fixture.inputs["xice"].reshape(-1)
    return (xland < 1.5) & (xice < 0.5) & ~fixture.glacial


def _graded(fixture) -> np.ndarray:
    return np.asarray([c not in HELD_OUT_CASES for c in fixture.cases])


# --------------------------------------------------------------------------
# CPU: the fixture is what it says it is.
# --------------------------------------------------------------------------

def test_fixture_is_the_one_build_sh_hashed():
    rows = (NOAH_ORACLE_DIR / "oracle-sha256sums.txt").read_text().splitlines()
    pinned = {line.split()[1]: line.split()[0] for line in rows}
    digest = hashlib.sha256((NOAH_ORACLE_DIR / FIXTURE).read_bytes()).hexdigest()
    assert pinned[FIXTURE] == digest


def test_every_wrf_hydro_branch_is_reached_by_the_fixture():
    rows = (NOAH_ORACLE_DIR / "wrf-hydro-coverage.txt").read_text().split("\n")
    seen = {line.split()[0]: int(line.split()[1]) for line in rows if line}
    assert seen == COVERAGE


def test_fixture_cycles_the_five_ponded_heads():
    fixture = _fixture()
    head = fixture.inputs["sfcheadrt"].reshape(-1)
    want = np.asarray([PONDED_MM[(c - 1) % 5] for c in fixture.cases],
                      np.float32)
    assert np.array_equal(head, want)


def test_wrf_writes_the_planes_only_on_ordinary_land():
    """lsm leaves both planes as they came on open water, sea ice and land
    ice, writes both on every ordinary land column, and hands sfcheadrt back
    unchanged everywhere (nothing reachable in SFLX writes SFHEAD1RT)."""
    fixture = _fixture()
    land = _ordinary_land(fixture)
    infxsrt = fixture.reference["infxsrt"].reshape(-1)
    soldrain = fixture.reference["soldrain"].reshape(-1)
    assert np.all(infxsrt[~land] == np.float32(INFXSRT_SENTINEL))
    assert np.all(soldrain[~land] == np.float32(SOLDRAIN_SENTINEL))
    assert np.all(infxsrt[land] >= 0.0) and np.all(soldrain[land] >= 0.0)
    assert np.array_equal(fixture.reference["sfcheadrt"],
                          fixture.inputs["sfcheadrt"])
    assert (~land).sum() > 0 and land.sum() > 0


def test_zero_head_columns_are_the_plain_build_word_for_word():
    """With no ponded head the hydro build reaches none of its edits on this
    fixture, so every shared column of those rows equals noah-lsm.csv's.  The
    rows with a head are where the two builds part, and they do part."""
    hydro = _fixture()
    plain = load_noah_oracle("noah-lsm.csv")
    zero = hydro.inputs["sfcheadrt"].reshape(-1) == 0.0
    land = _ordinary_land(hydro)
    parted = np.zeros_like(zero)
    for name, want in plain.reference.items():
        got = hydro.reference[name].reshape(-1)
        want = want.reshape(-1)
        assert np.array_equal(got[zero].view(np.uint32),
                              want[zero].view(np.uint32)), name
        parted |= got.view(np.uint32) != want.view(np.uint32)
    assert parted[land & ~zero].sum() > 0


def test_urban_with_wrf_hydro_is_refused_by_name():
    from woof.core.noah import launch_noah

    with pytest.raises(NotImplementedError, match="WRF_HYDRO and sf_urban"):
        launch_noah({}, None, 60.0, (0.1, 0.3, 0.6, 1.0), urban={},
                    wrf_hydro={})


# --------------------------------------------------------------------------
# GPU: the measurement.
# --------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_noah_column_hydro_is_bitwise_against_wrf_hydro_build():
    """Every output of lsm, every soil layer, and the three exchange planes:
    zero ULP on every graded column (ordinary land, open water, sea ice and
    land ice).  MEASURED 2026-11-07: RTX A6000 (sm_86, NVRTC 13.x) under
    GPUWM_WRF_EXACT=0 and =1, and the MLX backend on an M3 Max."""
    fixture = _fixture()
    port = noah_port_outputs(fixture)
    keep = _graded(fixture)
    worst: dict[str, int] = {}
    for name in (*OUTPUT_COLUMNS, *WRF_HYDRO_COLUMNS):
        got = np.ascontiguousarray(port[name][:, keep], np.float32)
        want = np.ascontiguousarray(fixture.reference[name][:, keep])
        worst[name] = int(fp32_ulp_distance(got, want).max())
    for name in SOIL_OUTPUT_COLUMNS:
        got = np.ascontiguousarray(port[name][:, :, keep], np.float32)
        want = np.ascontiguousarray(fixture.soil_reference[name][:, :, keep])
        worst[name] = int(fp32_ulp_distance(got, want).max())
    assert worst == {name: 0 for name in worst}


@pytest.mark.gpu
@requires_gpu
def test_spill_columns_hand_more_than_runoff1_to_the_routing():
    """Cases 39 and 74 are where SSTEP's WPLUS reaches the top layer and the
    remainder joins INFXS1RT, so infxsrt exceeds that step's SFCRUNOFF
    increment; the port reproduces both words."""
    fixture = _fixture()
    port = noah_port_outputs(fixture)
    for case in (39, 74):
        i = fixture.cases.index(case)
        runoff_mm = (fixture.reference["sfcrunoff"].reshape(-1)[i]
                     - fixture.inputs["sfcrunoff"].reshape(-1)[i])
        want = fixture.reference["infxsrt"].reshape(-1)[i]
        assert want > runoff_mm
        assert port["infxsrt"].reshape(-1)[i] == want
