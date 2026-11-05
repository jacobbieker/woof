"""Morrison with CESM aerosol (mp_physics=40) against WRF v4.7.1.

``tests/data/oracles/morrison_aero/aero-columns.npz`` holds three sets
written by ``tools/morrison_aero_wrf471_oracle/aero_columns.py`` and
returned by WRF's own ``MP_MORR_TWO_MOMENT_AERO``
(``run_morrison_aero.F90``: the byte-unmodified v4.7.1 module at gfortran
13.3 -O0, glibc 2.39; ``receipt.txt``).  119 columns x 50 levels each: the
mp_physics=10 regime oracle's twelve weather regimes x 8 variants and ten
edge columns, plus thirteen aerosol edge columns, every one given a droplet
number, W and the PBL's KZH on w levels, and ten CESM aerosol profiles.

* ``a0_m1_dt20_nz50_x10``: aercu_opt=0 (what woof runs), hail, dt 20 s,
  ten consecutive calls;
* ``a2_m0_dt72_nz50``: aercu_opt=2 (prescribed-aerosol activation and Liu-
  Penner nucleation), graupel, dt 72 s, one call;
* ``a2_m1_dt20_nz50_x20``: aercu_opt=2, hail, aercu_fct=1.5, twenty calls.

The standard, under GPUWM_WRF_EXACT=1: the unit built with
``MORR_ROW37_OFF`` (WRF's arithmetic alone) returns every word WRF
returned -- 25 level fields and 7 surface fields, whole-domain, the
forecast's column chunking and one column a call.  Production keeps
mp_physics=10's owner-ruled corrections; it differs from WRF only in the
columns pinned below (measured on pb's RTX A6000).  Default ("fast")
numerics contract multiply-adds and flush subnormals, as every woof unit
does; graded here for finite, nonnegative water only.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from conftest import requires_gpu

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "data" / "oracles" / "morrison_aero"
TOOLS = REPO / "tools" / "morrison_aero_wrf471_oracle"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

WRF_AERO_SHA256 = (
    "d512aa16ad862b04c7278540a207a56aa10955e052bd9b9bc870e9a4f0812180")
EXACT = __import__("woof.wrf_exact", fromlist=["ENABLED"]).ENABLED
STRICT_ONLY = pytest.mark.skipif(
    not EXACT, reason="WRF word-for-word oracle of the strict "
    "(GPUWM_WRF_EXACT=1) path; a default process contracts multiply-adds")

#: Production words that differ from WRF (row 37's nucleation cap, wide
#: freezing rates and nonnegative sedimentation mass), as differing columns
#: per set and the labels they carry: measured on pb, RTX A6000,
#: GPUWM_WRF_EXACT=1, all three chunkings alike.
PRODUCTION_DIFFERING_COLUMNS = {
    "a0_m1_dt20_nz50_x10": 10, "a2_m0_dt72_nz50": 10,
    "a2_m1_dt20_nz50_x20": 14}
PRODUCTION_LABELS = {
    "a0_m1_dt20_nz50_x10": {
        "cirrus_anvil", "desert_virga", "edge_stratosphere", "high_terrain",
        "kf_seeded", "supercell"},
    "a2_m0_dt72_nz50": {
        "cirrus_anvil", "cirrus_fast", "cirrus_slow", "edge_qsmall",
        "edge_stratosphere", "edge_supersat_homog", "high_terrain",
        "supercell"},
    "a2_m1_dt20_nz50_x20": {
        "cirrus_anvil", "cirrus_fast", "edge_hail_core",
        "edge_mass_no_number", "edge_negative", "edge_stratosphere",
        "edge_subnormal", "edge_supersat_homog", "kf_seeded",
        "mcs_stratiform", "night_fog", "supercell"},
}


def _load():
    import aero_columns as ac
    sets, labels = ac.load_npz(FIXTURE / "aero-columns.npz")
    return ac, sets, labels


# ---------------------------------------------------------------- CPU only

def test_fixture_is_from_the_pinned_unmodified_wrf_module():
    receipt = (FIXTURE / "receipt.txt").read_text(encoding="ascii")
    assert f"{WRF_AERO_SHA256}  /wrf/phys/module_mp_morr_two_moment_aero.F" \
        in receipt
    assert "GLIBC 2.39" in receipt and "GNU Fortran" in receipt
    assert "13.3.0" in receipt
    # The runner that wrote the fixture is the committed one.
    runner = hashlib.sha256((TOOLS / "run_morrison_aero.F90").read_bytes())
    assert f"{runner.hexdigest()}  tools/morrison_aero_wrf471_oracle/" \
        "run_morrison_aero.F90" in receipt


def test_fixture_spans_both_modes_both_aerosol_paths_and_edges():
    ac, sets, labels = _load()
    assert [s[0] for s in sets] == list(ac.SETS)
    assert set(labels) >= set(ac.AERO_EDGES)
    assert len(labels) == 12 * 8 + 10 + len(ac.AERO_EDGES)
    assert {s[1]["mode"] for s in sets} == {0, 1}
    assert {s[1]["aercu_opt"] for s in sets} == {0, 2}
    for name, meta, fields, stag, aero, surf, want in sets:
        assert meta["ncol"] == len(labels) and meta["nz"] == 50
        words = np.concatenate([fields[k].ravel() for k in ac.LEVEL_IN])
        bits = words.view(np.uint32)
        assert (bits == 0x80000000).any(), name          # -0
        subnormal = (bits & 0x7F800000 == 0) & (bits & 0x007FFFFF != 0)
        assert subnormal.any(), name
        assert (stag["w"] < 0).any() and (stag["w"] > 5.0).any()
        assert (stag["kzh"] > 1.0e3).any()           # WVAR's 50 m/s cap
        for k in ac.LEVEL_OUT:
            assert np.isfinite(want[k]).all(), (name, k)
        if meta["aercu_opt"] == 2:
            # Activation moved the droplet number and the CCN diagnostics
            # are WRF's nonzero words.
            assert (want["ccn7"] > 0).any() and (want["ccn1"] > 0).any()
            assert not np.array_equal(want["nc"], fields["nc"])
            assert (want["wact"] != 0).any()
        else:
            for k in ("ccn1", "ccn7", "efcg", "efig", "efsg", "wact"):
                assert not want[k].any(), (name, k)
        # Without MSKF hydrometeors the second reflectivity is the first.
        np.testing.assert_array_equal(want["mskf_refl_10cm"],
                                      want["refl_10cm"])


def test_constants_header_regenerates_from_the_dumped_init_words(tmp_path):
    out = tmp_path / "morrison_aero_constants.cuh"
    configured = os.environ.get("WRF_V471_SOURCE")
    if not configured:
        pytest.skip("rule: WRF_V471_SOURCE names no WRF v4.7.1 tree "
                    "(DERF1's DATA statements are read from it)")
    source = Path(configured) / "phys" / "module_mp_morr_two_moment_aero.F"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == WRF_AERO_SHA256
    subprocess.run(
        [sys.executable, str(TOOLS / "make_header.py"),
         str(TOOLS / "constants-wrf471-aero.txt"),
         str(REPO / "tools" / "morrison_constants" / "constants-wrf461.txt"),
         str(source), str(out)], check=True)
    assert out.read_bytes() == (
        REPO / "woof" / "core" / "kernels" /
        "morrison_aero_constants.cuh").read_bytes()


def test_the_mp10_unit_compiles_no_aero_line():
    from woof.core import kernels
    mp10 = kernels.module_source("morrison")
    aero = kernels.module_source("morrison_aero")
    assert "#define MORR_AERO 1" not in mp10
    assert "#define MORR_AERO 1" in aero
    assert aero.endswith(kernels._unit_text("morrison_aero"))


# ---------------------------------------------------------------- GPU

CHUNKINGS = ("whole", "forecast", "single")


def _columns_per_call(chunking, ncol):
    from woof.core.morrison_constants import work_chunk_columns
    return {"whole": None, "forecast": work_chunk_columns(ncol),
            "single": 1}[chunking]


def _run(row37_off, chunking):
    from woof.core import kernels
    ac, sets, labels = _load()
    original = ac.row37_off() if row37_off else None
    try:
        out = {}
        for name, meta, fields, stag, aero, surf, want in sets:
            got = ac.run_gpu(meta, fields, stag, aero, surf,
                             _columns_per_call(chunking, meta["ncol"]))
            out[name] = (got, want, ac.differing(got, want))
        return out, labels
    finally:
        if original is not None:
            kernels.load_module = original
            kernels.get_kernel.cache_clear()


@pytest.mark.gpu
@requires_gpu
@STRICT_ONLY
@pytest.mark.parametrize("chunking", CHUNKINGS)
def test_wrf_arithmetic_is_bitwise_on_every_column(chunking):
    out, _ = _run(True, chunking)
    for name, (_, _, bad) in out.items():
        counts = {k: int(v.sum()) for k, v in bad.items() if v.any()}
        assert counts == {}, (name, counts)


@pytest.mark.gpu
@requires_gpu
@STRICT_ONLY
@pytest.mark.parametrize("chunking", CHUNKINGS)
def test_production_differs_only_in_the_measured_columns(chunking):
    out, labels = _run(False, chunking)
    labels = np.asarray(labels)
    for name, (_, _, bad) in out.items():
        columns = np.zeros(len(labels), bool)
        for v in bad.values():
            columns |= v.any(axis=0) if v.ndim == 2 else v
        assert int(columns.sum()) == PRODUCTION_DIFFERING_COLUMNS[name], name
        assert set(labels[columns].tolist()) == PRODUCTION_LABELS[name], name


@pytest.mark.gpu
@requires_gpu
def test_production_water_is_finite_and_nonnegative():
    out, _ = _run(False, "forecast")
    for name, (got, _, _) in out.items():
        for k in ("th", "qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr",
                  "ni", "ns", "ng", "rainnc", "snownc", "graupelnc"):
            assert np.isfinite(got[k]).all(), (name, k)
        for k in ("qc", "qr", "qi", "qs", "qg", "rainnc", "snownc",
                  "graupelnc"):
            assert (got[k] >= 0).all(), (name, k)


@pytest.mark.gpu
@requires_gpu
def test_the_wrf_nucleation_stop_is_raised():
    """WRF STOPs on more than 9999 cm-3 nucleated ice (:5946-5951); the
    launcher raises instead of returning the zeroed number."""
    import cupy as cp
    from woof.core.morrison_aero import launch_morrison_aero
    nz, n = 4, 1
    f = lambda v: cp.full((nz, 1, n), v, cp.float32)    # noqa: E731
    p = cp.asarray(np.linspace(30000.0, 20000.0, nz, dtype=np.float32)
                   .reshape(nz, 1, n))
    pii = (p / 100000.0) ** np.float32(287.0 / 1004.5)
    t = 215.0                                   # tc = -58 C, homogeneous
    theta = (t / pii).astype(cp.float32)
    rho = (p / (287.0 * t)).astype(cp.float32)
    qv = f(1.0e-4)                              # far above ice saturation
    aerocu = cp.zeros((10, nz, 1, n), cp.float32)
    aerocu[5] = 2.0e4                           # sulfate, ~10^5 cm-3
    aerocu[0] = 50.0
    s = lambda: cp.zeros((1, n), cp.float32)    # noqa: E731
    w = cp.full((nz + 1, 1, n), 5.0, cp.float32)
    with pytest.raises(RuntimeError, match="incorrect ice nucleation"):
        launch_morrison_aero(
            theta, qv, f(0.0), f(0.0), f(0.0), f(0.0), f(0.0), f(0.0),
            f(0.0), f(0.0), f(0.0), f(0.0), rho, pii, p, f(200.0), w,
            cp.zeros((nz + 1, 1, n), cp.float32), s(), s(), s(), s(), s(),
            s(), s(), 20.0, aercu_opt=2, aerocu=aerocu,
            efcg=f(0.0), efig=f(0.0), efsg=f(0.0), wact=f(0.0))
