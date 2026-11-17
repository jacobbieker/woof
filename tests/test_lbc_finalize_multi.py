"""finalize_state_multi: every forced field of one spec_bdy_final in one
launch, word for word the per-field finalize_state_field launches
(kernels/lbc_state.cu, lateral_bc._launch_finalize_multi vs
_launch_finalize_field).

Both kernels evaluate the same device function per cell
(finalize_state_cell); these tests check that the table carries each
field's arrays, kind, shape and width as the per-field launch passes them:
every kind (u, v, theta, phi, qv, w, a generic scalar), with and without map
factors, a 3-D base theta, a strict-mode carried grid%muts, and more fields
than one table holds.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

import _numerics_child
from conftest import requires_gpu
from test_lbc_relax_multi import _setup

#: State attribute per forced name (the per-field launcher's mapping).
_ATTR = {"theta": "thp", "phi": "php"}


def _forced(state, cfg, *, muts):
    """The domain's forced fields after the MU install, plus a qv and a
    generic scalar row (their targets borrow theta's shape and table), with
    every target seeded; returns (interval, names, old frame, dtbc)."""
    import cupy as cp

    import woof.ingest.lateral_bc as lbc
    from woof.core.dycore import WRF_MUTS_SLOT

    rng = np.random.default_rng(41)
    if muts:
        carried = state.scratch(state.mup.shape, WRF_MUTS_SLOT)
        carried[...] = cp.asarray(
            (state.mub2d + cp.asarray(rng.normal(0.0, 40.0, state.mup.shape)
                                      .astype(np.float32))))
    device, _common, _weights = lbc._relaxation_context(state, cfg)
    fields = dict(device.fields)
    state.qv = cp.asarray(
        rng.uniform(0.0, 1e-2, state.thp.shape).astype(np.float32))
    state.qc_extra = cp.asarray(
        rng.uniform(-1e-5, 1e-3, state.thp.shape).astype(np.float32))
    fields["qv"] = fields["theta"]
    fields["qc_extra"] = fields["theta"]
    interval = types.SimpleNamespace(fields=fields)
    dtbc = 17.5
    old_frame = lbc._launch_mu_boundary_values(
        state, fields["mu"], dtbc, cfg.spec_zone)
    names = ("u", "v", "w", "theta", "phi", "qv", "qc_extra")
    return interval, names, old_frame, dtbc


def _seed(state, names, seed):
    import cupy as cp

    rng = np.random.default_rng(seed)
    for name in names:
        a = getattr(state, _ATTR.get(name, name))
        scale = 1e-3 if name.startswith("q") else 5.0
        a[...] = cp.asarray(rng.normal(0.0, scale, a.shape).astype(np.float32))


def _words(state, names):
    import cupy as cp

    return {name: cp.asnumpy(getattr(state, _ATTR.get(name, name))).view(np.uint32)
            for name in names}


def _counter(monkeypatch):
    """Count finalize_state_multi launches, keeping the real kernel, so an
    enabled run is known to have taken the table path."""
    import woof.ingest.lateral_bc as lbc

    counts = {"launches": 0}
    real_get = lbc.get_kernel

    def get_kernel(name, func, shape=()):
        kernel = real_get(name, func, shape)
        if func != "finalize_state_multi":
            return kernel

        def launch(*args, **kwargs):
            counts["launches"] += 1
            return kernel(*args, **kwargs)
        return launch

    monkeypatch.setattr(lbc, "get_kernel", get_kernel)
    return counts


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("map_factors", [False, True])
@pytest.mark.parametrize("thb_3d", [False, True])
@pytest.mark.parametrize("muts", [False, True])
@pytest.mark.parametrize("ny,nx", [(36, 40), (11, 13)])
def test_table_matches_the_per_field_launches(monkeypatch, map_factors,
                                               thb_3d, muts, ny, nx):
    import cupy as cp

    import woof.ingest.lateral_bc as lbc

    cfg, state, _ = _setup(ny=ny, nx=nx, map_factors=map_factors)
    if thb_3d:
        rng = np.random.default_rng(7)
        state.thb = cp.asarray(
            (300.0 + rng.normal(0.0, 3.0, state.thp.shape)).astype(np.float32))
    interval, names, old_frame, dtbc = _forced(state, cfg, muts=muts)
    counts = _counter(monkeypatch)
    _seed(state, names, 3)
    before = _words(state, names)
    for name in names:
        lbc._launch_finalize_field(state, name, interval.fields[name],
                                   old_frame, dtbc, cfg.spec_zone)
    per_field = _words(state, names)
    _seed(state, names, 3)
    assert lbc._launch_finalize_multi(state, names, interval, old_frame,
                                      dtbc, cfg.spec_zone)
    table = _words(state, names)
    assert counts["launches"] == 1
    for name in names:
        assert np.array_equal(per_field[name], table[name]), name
        # not vacuous: the finalizer moved the field
        assert not np.array_equal(before[name], table[name]), name


@requires_gpu
@pytest.mark.gpu
def test_more_fields_than_one_table_take_further_launches(monkeypatch):
    import cupy as cp

    import woof.ingest.lateral_bc as lbc

    cfg, state, _ = _setup(map_factors=True)
    interval, _names, old_frame, dtbc = _forced(state, cfg, muts=False)
    count = lbc.LBC_FINAL_MULTI_MAX_ROWS + 3
    names = tuple(f"qx{n}" for n in range(count))
    for name in names:
        setattr(state, name, cp.zeros_like(state.thp))
        interval.fields[name] = interval.fields["theta"]
    counts = _counter(monkeypatch)
    _seed(state, names, 11)
    for name in names:
        lbc._launch_finalize_field(state, name, interval.fields[name],
                                   old_frame, dtbc, cfg.spec_zone)
    per_field = _words(state, names)
    _seed(state, names, 11)
    assert lbc._launch_finalize_multi(state, names, interval, old_frame,
                                      dtbc, cfg.spec_zone)
    assert counts["launches"] == 2
    table = _words(state, names)
    for name in names:
        assert np.array_equal(per_field[name], table[name]), name


@requires_gpu
@pytest.mark.gpu
def test_overlapping_targets_fall_back(monkeypatch):
    """Two fields whose targets share bytes must not run concurrently."""
    import cupy as cp

    import woof.ingest.lateral_bc as lbc

    cfg, state, _ = _setup()
    interval, _names, old_frame, dtbc = _forced(state, cfg, muts=False)
    n = state.thp.size
    backing = cp.zeros((n + 1,), dtype=cp.float32)
    state.qa = backing[:n].reshape(state.thp.shape)
    state.qb = backing[1:n + 1].reshape(state.thp.shape)
    for name in ("qa", "qb"):
        interval.fields[name] = interval.fields["theta"]
    counts = _counter(monkeypatch)
    assert not lbc._launch_finalize_multi(state, ("qa", "qb"), interval,
                                          old_frame, dtbc, cfg.spec_zone)
    assert counts["launches"] == 0


@requires_gpu
@pytest.mark.gpu
def test_apply_state_boundary_values_takes_one_launch(monkeypatch):
    """The production entry, table on vs off, from the same state."""
    import cupy as cp

    import woof.ingest.lateral_bc as lbc

    cfg, state, _ = _setup(map_factors=True)
    names = ("u", "v", "w", "theta", "phi")
    mup = state.mup.copy()
    counts = _counter(monkeypatch)
    results = {}
    for enabled in (False, True):
        monkeypatch.setattr(lbc, "LBC_FINAL_MULTI_ENABLED", enabled)
        state.mup[...] = mup
        _seed(state, names, 5)
        lbc.apply_state_boundary_values(state, cfg, elapsed_seconds=20.0)
        results[enabled] = _words(state, names)
        results[enabled]["mu"] = cp.asnumpy(state.mup).view(np.uint32)
        assert counts["launches"] == (1 if enabled else 0)
    for key in results[False]:
        assert np.array_equal(results[False][key], results[True][key]), key


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize(*_numerics_child.MODES)
def test_identity_holds_under_each_numerics(numerics, exact):
    """The table against the per-field launches in a child per mode:
    production fast, the battery's strict, and WRF-exact."""
    _numerics_child.run(__file__, [
        "test_table_matches_the_per_field_launches",
        "test_more_fields_than_one_table_take_further_launches",
        "test_apply_state_boundary_values_takes_one_launch"], numerics, exact)
