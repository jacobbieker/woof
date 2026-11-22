"""woof.nn P2.4: the example hex column backend (a scheme plus a column model).

CUDA-free, in the style of ``DeterministicNumpyBackend``: the inner backend
is that double, the batch is built on the host, and the backend runs
through :func:`woof.hex.physics_seam.execute_column_backend`.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.hex.physics_seam import (ColumnBatch, DeterministicNumpyBackend, GPUWMSchemeSelection,
                                   execute_column_backend)
from woof.nn.hex_backend import NNColumnBackend
from woof.nn.launch import NNConfigError
from woof.nn.manifest import FieldSpec, write_mlp_artefact

NZ, NC, L = 5, 7, 3
SELECTION = GPUWMSchemeSelection(mp_physics=6, cu_physics=0, sf_surface_physics=0, bl_pbl_physics=0,
                                 sf_sfclay_physics=0, ra_lw_physics=0, ra_sw_physics=0, ra_rrtmg_variant="",
                                 wrf_rrtmg_compatibility="", cloud_fraction_scheme="", mpas_names={})


def _batch(seed=0):
    rng = np.random.default_rng(seed)
    f = lambda lo, hi, *s: rng.uniform(lo, hi, s)
    zi = np.cumsum(np.full((NZ + 1, NC), 300.0), axis=0) + f(0, 400, NC)[None, :]
    theta = f(285, 310, NZ, NC)
    qv = f(1e-4, 1e-2, NZ, NC)
    rho = f(0.5, 1.2, NZ, NC)
    return ColumnBatch(
        metric_density=rho, dry_density=rho, moist_density=rho * (1 + qv),
        modified_theta=theta * (1 + 1.608 * qv), theta=theta, temperature=theta * 0.95,
        eos_pressure=np.ascontiguousarray(np.linspace(1e5, 6e4, NZ)[:, None] * f(0.98, 1.0, NC)[None, :]), eos_pressure_interface=None, hydrostatic_moist_pressure=None,
        hydrostatic_moist_pressure_interface=None, hydrostatic_dry_pressure=None,
        hydrostatic_dry_pressure_interface=None, height=0.5 * (zi[:-1] + zi[1:]), height_interface=zi,
        layer_thickness=zi[1:] - zi[:-1], u=f(-10, 10, NZ, NC), v=f(-10, 10, NZ, NC), w=f(-1, 1, NZ + 1, NC),
        scalars={"qv": qv, "qc": f(0, 1e-4, NZ, NC)}, surface={"hfx": f(-20, 200, NC)}, static={})


def _artefact(path, *, outputs=("dtheta", "dqv"), out_mean=0.0, inputs=("theta", "qv", "w", "z", "hfx")):
    ins = tuple(FieldSpec(n, "1", 1 if n == "hfx" else L) for n in inputs)
    outs = tuple(FieldSpec(n, "1", L) for n in outputs)
    n_in, n_out = sum(f.levels for f in ins), sum(f.levels for f in outs)
    mean = np.broadcast_to(np.asarray(out_mean, dtype=np.float64), (n_out,)).copy()
    write_mlp_artefact(path, name="example", inputs=ins, outputs=outs,
                       layers=[(np.zeros((3, n_in)), np.zeros(3)), (np.zeros((n_out, 3)), np.zeros(n_out))],
                       activation="tanh", input_mean=np.zeros(n_in), input_std=np.ones(n_in), output_mean=mean,
                       output_std=np.ones(n_out), input_min=np.full(n_in, -1e30), input_max=np.full(n_in, 1e30))
    return path


def _run(backend, batch):
    return execute_column_backend(batch, SELECTION, backend, dt=20.0, mutation="forbid")


def test_zero_residual_returns_the_inner_backends_rates(tmp_path):
    inner = DeterministicNumpyBackend()
    nn = NNColumnBackend.from_artefact(DeterministicNumpyBackend(), _artefact(tmp_path / "m"), positivity=False)
    a, b = _run(inner, _batch()), _run(nn, _batch())
    for name in ("du", "dv", "dtheta"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    np.testing.assert_array_equal(a.dscalars["qv"], b.dscalars["qv"])


def test_constant_residual_adds_on_the_models_levels_only(tmp_path):
    model = _artefact(tmp_path / "m", out_mean=np.r_[np.full(L, 1e-3), np.zeros(L)])
    nn = NNColumnBackend.from_artefact(DeterministicNumpyBackend(theta_rate=0.01), model, positivity=False)
    out = _run(nn, _batch())
    np.testing.assert_allclose(out.dtheta[:L], 0.011)
    np.testing.assert_array_equal(out.dtheta[L:], 0.01)
    np.testing.assert_array_equal(out.du, 0.25)


def test_replace_substitutes_the_model_and_keeps_the_rest(tmp_path):
    model = _artefact(tmp_path / "m", outputs=("du", "dqc"), out_mean=np.r_[np.full(L, -1e-3), np.full(L, 2e-9)])
    nn = NNColumnBackend.from_artefact(DeterministicNumpyBackend(), model, mode="replace", positivity=False)
    out = _run(nn, _batch())
    np.testing.assert_allclose(out.du[:L], -1e-3)
    np.testing.assert_array_equal(out.du[L:], 0.25)
    np.testing.assert_array_equal(out.dv, -0.5)
    np.testing.assert_allclose(out.dscalars["qc"][:L], 2e-9)
    np.testing.assert_array_equal(out.dscalars["qc"][L:], 0)
    assert "qc" in nn.requirements(SELECTION).scalars


def test_residual_positivity_floor_is_the_schemes_remaining_moisture(tmp_path):
    batch = _batch()
    batch.scalars["qv"][:] = 1e-7
    batch.moist_density[:] = batch.dry_density * (1 + batch.scalars["qv"])
    model = _artefact(tmp_path / "m", out_mean=np.r_[np.zeros(L), np.full(L, -1.0)])
    nn = NNColumnBackend.from_artefact(DeterministicNumpyBackend(scalar_rates={"qv": 1e-8}), model)
    out = _run(nn, batch)
    q_next = batch.scalars["qv"][:L] + 20.0 * out.dscalars["qv"][:L]
    assert float(q_next.min()) >= -1e-15


def test_out_of_range_columns_keep_the_schemes_rates(tmp_path):
    model = _artefact(tmp_path / "m", out_mean=np.r_[np.full(L, 1.0), np.zeros(L)])
    nn = NNColumnBackend.from_artefact(DeterministicNumpyBackend(), model, positivity=False)
    nn.model.manifest.input_max[0] = 0.0   # theta at level 0: every column out of range
    out = _run(nn, _batch())
    np.testing.assert_array_equal(out.dtheta, 0.01)
    assert int(nn.policy.ledger.summary()["columns_out_of_range"]) == NC


def test_refusals(tmp_path):
    with pytest.raises(NNConfigError, match="residual or replace"):
        NNColumnBackend.from_artefact(DeterministicNumpyBackend(), _artefact(tmp_path / "a"), mode="tap")
    with pytest.raises(NNConfigError, match="not rates"):
        NNColumnBackend.from_artefact(DeterministicNumpyBackend(), _artefact(tmp_path / "b", outputs=("dtheta_m",)))
    nn = NNColumnBackend.from_artefact(DeterministicNumpyBackend(), _artefact(tmp_path / "c", inputs=("theta", "lai")))
    with pytest.raises(NNConfigError, match="'lai'"):
        _run(nn, _batch())
