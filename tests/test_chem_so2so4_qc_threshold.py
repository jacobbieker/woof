"""The opt-in so2so4 cloud threshold (RunConfig.so2so4_qc_threshold).

WRF's so2so4 (chem/module_gocart_so2so4.F, ils = 0, gd_on = 0) calls a cell
cloudy when qc > 0, so round-off cloud water (1e-38 .. 1e-10 kg/kg, common in
the specified boundary ring of a real-data run) turns all the SO2 its H2O2
allows into sulfate in one chem step, and SORGAM nucleates that into an
Aitken burst.  The threshold is off by default (0 = WRF's own test, the
kernel launched on qc itself); this file grades the door, the identity
bookkeeping and the kernel against a float32 transcription of the Fortran.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from conftest import requires_gpu
from woof.config import ConfigurationRefused, RunConfig, validate_chem_config


def _cfg(**overrides) -> RunConfig:
    values = dict(nx=12, ny=10, nz=6, dx=3000.0, dy=3000.0, ztop=12000.0,
                  dt=18.0, run_seconds=60.0, bl_pbl_physics=1)
    values.update(overrides)
    return RunConfig(**values)


def test_default_is_wrf_and_the_key_is_keyword_only():
    field = {f.name: f for f in dataclasses.fields(RunConfig)}["so2so4_qc_threshold"]
    assert field.default == 0.0 and field.kw_only


@pytest.mark.parametrize("value", [-1e-6, float("nan"), float("inf"), True, "1e-6"])
def test_a_bad_threshold_is_refused(value):
    with pytest.raises(ConfigurationRefused, match="so2so4_qc_threshold"):
        validate_chem_config(_cfg(chem_sets="radm2sorg", so2so4_qc_threshold=value))


def test_a_threshold_without_sorgam_is_refused():
    with pytest.raises(ConfigurationRefused, match="no aerosol.sorgam rows"):
        validate_chem_config(_cfg(chem_sets="radm2", so2so4_qc_threshold=1e-6))
    validate_chem_config(_cfg(chem_sets="radm2"))


def test_radm2sorg_admits_a_threshold():
    validate_chem_config(_cfg(chem_sets="radm2sorg", so2so4_qc_threshold=1e-6))


def test_identity_is_unchanged_at_the_default():
    from woof.ingest.prepared_cache import PREPARATION_INERT_RUN_FIELDS
    from woof.io import restart

    assert restart.CHEM_FORK_RUN_DEFAULTS["so2so4_qc_threshold"] == 0.0
    assert restart.LATE_DEFAULT_RUN_FIELDS["so2so4_qc_threshold"] == 0.0
    assert "run.so2so4_qc_threshold" in PREPARATION_INERT_RUN_FIELDS
    cfg = _cfg(chem_sets="radm2sorg")
    echo = restart.configuration_echo(cfg)
    assert "so2so4_qc_threshold" not in echo
    on = restart.configuration_echo(dataclasses.replace(cfg, so2so4_qc_threshold=1e-6))
    assert on["so2so4_qc_threshold"] == 1e-6
    with pytest.raises(restart.RestartMismatchError, match="so2so4_qc_threshold"):
        restart._require_config_match(echo, dataclasses.replace(
            cfg, so2so4_qc_threshold=1e-6), "threshold-off-to-on")


def _reference(qc, t, so2, sulf, h2o2, threshold):
    """so2so4 for ils = 0, gd_on = 0 in float32 (numpy's +-*/ round to
    nearest like the kernel's __f*_rn), with the opt-in threshold."""
    f = np.float32
    so2, sulf, h2o2 = so2.copy(), sulf.copy(), h2o2.copy()
    for n in range(qc.size):
        cldf = f(1) if qc[n] > f(threshold) else f(0)
        tc2 = so2[n]
        if cldf > 0 and tc2 > 0 and t[n] > f(258):
            tc3, hp = sulf[n], h2o2[n]
            if tc2 > hp:
                cldf = f(cldf * f(hp / tc2))
                hp = f(hp * f(f(1) - cldf))
            else:
                hp = f(hp * f(f(1) - f(f(cldf * tc2) / hp)))
            so2[n] = max(f(1e-16), f(tc2 * f(f(1) - cldf)))
            sulf[n] = max(f(1e-16), f(tc3 + f(tc2 * cldf)))
            h2o2[n] = max(f(1e-16), hp)
    return so2, sulf, h2o2


def _columns():
    rng = np.random.default_rng(7)
    shape = (8, 3, 5)
    n = int(np.prod(shape))
    # Cloud water from none through round-off to real cloud, both sides
    # of 1e-6 and exactly on it.
    qc = np.array([0.0, 1.2e-38, 6e-19, 2e-14, 5e-10, 3e-7, 1e-6, 1.01e-6,
                   4e-5, 2.2e-4] * (n // 10 + 1), np.float32)[:n]
    t = rng.uniform(250.0, 290.0, n).astype(np.float32)
    so2 = (10.0 ** rng.uniform(-5, -2, n)).astype(np.float32)
    sulf = (10.0 ** rng.uniform(-8, -6, n)).astype(np.float32)
    h2o2 = (10.0 ** rng.uniform(-4, -2.5, n)).astype(np.float32)
    return shape, qc, t, so2, sulf, h2o2


@requires_gpu
@pytest.mark.parametrize("threshold", [0.0, 1e-6])
def test_kernel_matches_the_fortran_with_and_without_the_threshold(threshold):
    import cupy as cp
    from woof.core.chem_radm2 import launch_so2so4

    shape, qc, t, so2, sulf, h2o2 = _columns()
    want = _reference(qc, t, so2, sulf, h2o2, threshold)
    dev = [cp.asarray(a.reshape(shape)) for a in (so2, sulf, h2o2)]
    qc_dev = cp.asarray(qc.reshape(shape))
    launch_so2so4(*dev, cp.asarray(t.reshape(shape)), qc_dev,
                  qc_threshold=threshold)
    for got, ref in zip(dev, want):
        assert cp.asnumpy(got).reshape(-1).tobytes() == ref.tobytes()
    # qc itself is never written.
    assert cp.asnumpy(qc_dev).reshape(-1).tobytes() == qc.tobytes()


@requires_gpu
def test_round_off_cloud_converts_only_under_wrfs_test():
    import cupy as cp
    from woof.core.chem_radm2 import launch_so2so4

    shape = (1, 1, 2)
    qc = cp.asarray(np.array([2e-14, 3e-4], np.float32).reshape(shape))
    t = cp.full(shape, 280.0, cp.float32)
    runs = {}
    for threshold in (0.0, 1e-6):
        so2 = cp.full(shape, 1e-2, cp.float32)
        sulf = cp.full(shape, 1e-7, cp.float32)
        h2o2 = cp.full(shape, 1e-3, cp.float32)
        launch_so2so4(so2, sulf, h2o2, t, qc, qc_threshold=threshold)
        runs[threshold] = cp.asnumpy(sulf).reshape(-1)
    # WRF's qc > 0: the round-off cell gets all 1e-3 ppmv H2O2 allows.
    assert runs[0.0][0] > np.float32(9e-4)
    # The threshold leaves it alone and still converts the real cloud.
    assert runs[1e-6][0] == np.float32(1e-7)
    assert runs[1e-6][1] == runs[0.0][1] > np.float32(9e-4)
