"""woof.nn P2.4: the hex (MPAS-route) NN column seat, host arithmetic.

The seat reads the held physics sources ``ResidentNativeCoupling.stage``
assembles (``rho_theta = m (c dtheta + Rv/Rd theta dqv)``, ``scalars[q] = m
dq``, ``m = rho_zz``), turns them into physical rates for the tap, and adds a
column model's increment back with the same coupling.  These tests run on
NumPy arrays with the couplers' formula written out (the CUDA kernels
``mpas_rk_cells_v842_f32`` and ``couple_cells_v841_f32``); the GPU runs are in
``tests/test_nn_hex_gpu.py``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from woof.nn import hex_seat, launch
from woof.nn.hex_seat import OUTPUTS, RV_OVER_RD, VOCABULARY, HexColumnSeat
from woof.nn.manifest import FieldSpec, write_mlp_artefact

NAMES = ("qv", "qc", "qr", "qi", "qs", "qg")
NZ, NC = 4, 12
#: interior 0-7, relaxation 8-9, specified 10-11; the padded garbage cell is 12.
BDY = np.array([0] * 8 + [2, 5, 6, 7], dtype=np.int32)
INPUTS = ("theta", "qv", "p", "z", "hfx")


def _couple(mass, theta_m, qv, dtheta, rates):
    """MPAS's coupling, as mpas_rk_cells_v842_f32 spells it (float32)."""
    f32 = np.float32
    coeff = f32(1) + RV_OVER_RD * qv
    scalars = {q: mass * r for q, r in rates.items()}
    vapour = scalars.get("qv", np.zeros_like(mass))
    rho_theta = coeff * (mass * dtheta) + (RV_OVER_RD * theta_m) * vapour / coeff
    return rho_theta.astype(f32), scalars


def _world(seed=3, padded=True):
    rng = np.random.default_rng(seed)
    f32 = np.float32
    width = NC + 1 if padded else NC
    f = lambda lo, hi, *s: rng.uniform(lo, hi, s).astype(f32)
    rho = f(0.4, 1.2, NZ, width)
    qv = f(1e-4, 1.2e-2, NZ, width)
    scalars = np.zeros((len(NAMES), NZ, width), dtype=f32)
    scalars[0] = qv
    scalars[1] = f(0.0, 1e-4, NZ, width)
    theta = f(280.0, 320.0, NZ, width)
    theta_m = (theta * (f32(1) + RV_OVER_RD * qv)).astype(f32)
    state = SimpleNamespace(rho=rho, scalars=scalars, time_seconds=0.0)
    saved = SimpleNamespace(theta_m=theta_m, exner=f(0.8, 1.0, NZ, width),
                            vertical_velocity=f(-0.5, 0.5, NZ + 1, width))
    rates = {"qv": f(-2e-7, 2e-7, NZ, width), "qc": f(-1e-8, 1e-8, NZ, width)}
    dtheta = f(-2e-4, 2e-4, NZ, width)
    rho_theta, held = _couple(rho, theta_m, qv, dtheta, rates)
    held_scalars = np.zeros_like(scalars)
    for q, value in held.items():
        held_scalars[NAMES.index(q)] = value
    if padded:
        rho_theta[:, -1] = 0
        held_scalars[..., -1] = 0
    sources = SimpleNamespace(rho=np.zeros_like(rho), rho_u=np.zeros((NZ, 30), dtype=f32),
                              rho_theta=rho_theta, scalars=held_scalars, diabatic_theta=None)
    pools = {("diag", "pressure"): f(5e4, 1e5, NZ, NC), ("diag_physics", "hfx"): f(-50, 300, NC),
             ("diag", "uReconstructZonal"): f(-10, 10, NZ, NC),
             ("diag", "uReconstructMeridional"): f(-10, 10, NZ, NC)}
    zgrid = np.cumsum(np.full((NZ + 1, width), 250.0), axis=0) + rng.uniform(0, 500, width)[None, :]
    return SimpleNamespace(state=state, saved=saved, sources=sources, rates=rates, dtheta=dtheta,
                           pools=pools, zgrid=zgrid, zz=np.ones((NZ, width), dtype=f32))


def _settings(**over):
    base = dict(source=None, mode="tap", scheme="total", inputs=INPUTS, outputs=("dtheta", "dqv", "dqc"),
                levels=None, tap=None, tap_every=1, tap_columns=5, tap_seed=0, backend="numpy",
                limit_outputs=False, positivity=False)
    base.update(over)
    return launch.NNSettings(**base)


def _seat(world, settings, *, bdy=BDY, tap=None, mode="correct"):
    return HexColumnSeat(settings, cp=np, names=NAMES, n_cells=NC, n_levels=NZ, zgrid=world.zgrid,
                         zz=world.zz, bdy_mask=bdy, dt=30.0, mode=mode,
                         fields=lambda pool, name: world.pools[(pool, name)], tap=tap)


def _specs(inputs=INPUTS, outputs=("dtheta", "dqv"), levels=NZ):
    ins = tuple(FieldSpec(n, VOCABULARY[n][0], levels if VOCABULARY[n][1] == "3d" else 1) for n in inputs)
    outs = tuple(FieldSpec(n, OUTPUTS[n], levels) for n in outputs)
    return ins, outs


def _artefact(path, *, out_mean=0.0, outputs=("dtheta", "dqv"), inputs=INPUTS, levels=NZ, name="const",
              in_min=-1e30, in_max=1e30):
    """An MLP with zero weights: it returns ``out_mean`` everywhere."""
    ins, outs = _specs(inputs, outputs, levels)
    n_in, n_out = sum(f.levels for f in ins), sum(f.levels for f in outs)
    layers = [(np.zeros((4, n_in)), np.zeros(4)), (np.zeros((n_out, 4)), np.zeros(n_out))]
    mean = np.broadcast_to(np.asarray(out_mean, dtype=np.float64), (n_out,)).copy()
    write_mlp_artefact(path, name=name, inputs=ins, outputs=outs, layers=layers, activation="tanh",
                       input_mean=np.zeros(n_in), input_std=np.ones(n_in), output_mean=mean,
                       output_std=np.ones(n_out), input_min=np.full(n_in, in_min), input_max=np.full(n_in, in_max))
    return path


class _Tap:
    def __init__(self):
        self.rows, self.samples, self.path = [], 0, "memory"

    def append(self, x, y, *, time_seconds, step, column_index):
        self.rows.append((x, y, time_seconds, step, np.asarray(column_index)))
        self.samples += 1

    def truncate(self, time_seconds):
        self.rows = [r for r in self.rows if r[2] < time_seconds]

    def close(self):
        pass


# ------------------------------------------------------------ arithmetic --

def test_decoupling_inverts_the_mpas_coupling():
    w = _world()
    s = w.sources
    rates = hex_seat.decouple(np, rho_theta=s.rho_theta[:, :NC],
                              scalars={q: s.scalars[NAMES.index(q), :, :NC] for q in ("qv", "qc")},
                              mass=w.state.rho[:, :NC], theta_m=w.saved.theta_m[:, :NC],
                              qv=w.state.scalars[0, :, :NC])
    np.testing.assert_allclose(rates["dqv"], w.rates["qv"][:, :NC], rtol=2e-6, atol=1e-13)
    np.testing.assert_allclose(rates["dqc"], w.rates["qc"][:, :NC], rtol=2e-6, atol=1e-14)
    np.testing.assert_allclose(rates["dtheta"], w.dtheta[:, :NC], rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(rates["dtheta_m"] * w.state.rho[:, :NC], s.rho_theta[:, :NC], rtol=1e-6)


def test_coupled_increment_is_the_mpas_coupling_of_the_increment():
    w = _world()
    mass, theta_m, qv = w.state.rho[:, :NC], w.saved.theta_m[:, :NC], w.state.scalars[0, :, :NC]
    inc = {"dtheta": w.dtheta[:, :NC], "dqv": w.rates["qv"][:, :NC], "dqc": w.rates["qc"][:, :NC]}
    d_theta, d_q = hex_seat.couple_increment(np, inc, mass=mass, theta_m=theta_m, qv=qv)
    expected, held = _couple(mass, theta_m, qv, inc["dtheta"], {"qv": inc["dqv"], "qc": inc["dqc"]})
    np.testing.assert_allclose(d_theta, expected, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(d_q["qv"], held["qv"], rtol=1e-6)
    # dtheta_m is the modified-theta rate: no vapour term is added.
    d_theta_m, _ = hex_seat.couple_increment(np, {"dtheta_m": inc["dtheta"], "dqv": inc["dqv"]},
                                             mass=mass, theta_m=theta_m, qv=qv)
    np.testing.assert_array_equal(d_theta_m, mass * inc["dtheta"])
    # A vapour-only increment keeps dry theta: rho_theta takes the vapour term alone.
    d_vap, _ = hex_seat.couple_increment(np, {"dqv": inc["dqv"]}, mass=mass, theta_m=theta_m, qv=qv)
    theta = theta_m / (np.float32(1) + RV_OVER_RD * qv)
    np.testing.assert_allclose(d_vap, mass * RV_OVER_RD * theta * inc["dqv"], rtol=1e-6)
    with pytest.raises(launch.NNConfigError, match="not both"):
        hex_seat.couple_increment(np, {"dtheta": inc["dtheta"], "dtheta_m": inc["dtheta"]},
                                  mass=mass, theta_m=theta_m, qv=qv)


def test_pools_and_masks_of_a_limited_area_mesh():
    assert hex_seat.interior_pool(NC, BDY).tolist() == list(range(8))
    assert hex_seat.interior_pool(NC, None).tolist() == list(range(NC))
    mask = hex_seat.apply_mask(NC, BDY)
    assert mask.tolist() == [1.0] * 10 + [0.0, 0.0]
    assert hex_seat.apply_mask(NC, np.zeros(NC)) is None
    with pytest.raises(launch.NNConfigError, match="no interior"):
        hex_seat.interior_pool(3, np.array([1, 2, 6]))


# ------------------------------------------------------------------ tap --

def test_tap_records_interior_rates_and_leaves_the_sources_untouched():
    w = _world()
    before = {k: getattr(w.sources, k).copy() for k in ("rho_theta", "scalars")}
    tap = _Tap()
    seat = _seat(w, _settings(tap_every=2), tap=tap)
    for step in range(4):
        seat.stage(w.sources, state=w.state, saved=w.saved, time_seconds=step * 30.0)
    for k, v in before.items():
        np.testing.assert_array_equal(getattr(w.sources, k).view(np.uint32), v.view(np.uint32), err_msg=k)
    assert [r[3] for r in tap.rows] == [0, 2]
    x, y, t, step, cells = tap.rows[1]
    assert x.shape == (4 * NZ + 1, 5) and y.shape == (3 * NZ, 5)
    assert set(cells.tolist()) <= set(range(8)), "a tap column left the interior"
    np.testing.assert_array_equal(cells, seat.draw(60.0))
    np.testing.assert_allclose(y[:NZ], w.dtheta[:, cells], rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(y[NZ:2 * NZ], w.rates["qv"][:, cells], rtol=2e-6, atol=1e-13)
    theta = w.saved.theta_m / (1 + RV_OVER_RD * w.state.scalars[0])
    np.testing.assert_allclose(x[:NZ], theta[:, cells], rtol=1e-6)
    np.testing.assert_array_equal(x[2 * NZ:3 * NZ], w.pools[("diag", "pressure")][:, cells])
    np.testing.assert_array_equal(x[-1], w.pools[("diag_physics", "hfx")][cells])
    z = 0.5 * (w.zgrid[:-1] + w.zgrid[1:]) - w.zgrid[0]
    np.testing.assert_allclose(x[3 * NZ:4 * NZ], z[:, cells], rtol=1e-6)
    assert seat.identity() is None


def test_tap_writes_a_store(tmp_path):
    pytest.importorskip("zarr")
    from woof.nn.tap import read_tap
    w = _world()
    seat = _seat(w, _settings(tap=tmp_path / "tap.zarr", tap_columns=4))
    for step in range(3):
        seat.stage(w.sources, state=w.state, saved=w.saved, time_seconds=step * 30.0)
    summary = seat.close()
    assert summary["samples"] == 3 and summary["route"] == "hex" and summary["mpas_identical"]
    record = read_tap(tmp_path / "tap.zarr")
    assert record["attrs"]["route"] == "hex" and record["attrs"]["scheme"] == "total"
    assert record["inputs"].shape == (3, 4 * NZ + 1, 4)
    assert record["time_seconds"].tolist() == [0.0, 30.0, 60.0]


# ---------------------------------------------------------------- apply --

def _applying(world, tmp_path, *, out_mean, outputs=("dtheta", "dqv"), mode="residual", **over):
    model = _artefact(tmp_path / "m", out_mean=out_mean, outputs=outputs)
    return _seat(world, _settings(mode=mode, model=model, inputs=None, outputs=None, tap_columns=256, **over))


def test_zero_residual_leaves_the_sources_unchanged(tmp_path):
    w = _world()
    before = {k: getattr(w.sources, k).copy() for k in ("rho_theta", "scalars")}
    seat = _applying(w, tmp_path, out_mean=0.0)
    seat.stage(w.sources, state=w.state, saved=w.saved, time_seconds=0.0)
    for k, v in before.items():
        np.testing.assert_array_equal(getattr(w.sources, k), v, err_msg=k)


def test_constant_residual_heats_every_cell_outside_the_specified_zone(tmp_path):
    w = _world()
    before = w.sources.rho_theta.copy()
    q_before = w.sources.scalars.copy()
    rate = 1.0e-4  # K s-1, dry theta
    seat = _applying(w, tmp_path, out_mean=np.r_[np.full(NZ, rate), np.zeros(NZ)])
    seat.stage(w.sources, state=w.state, saved=w.saved, time_seconds=0.0)
    coeff = 1 + RV_OVER_RD * w.state.scalars[0]
    added = w.sources.rho_theta - before
    np.testing.assert_allclose(added[:, :10], (w.state.rho * coeff * np.float32(rate))[:, :10], rtol=1e-5)
    np.testing.assert_array_equal(added[:, 10:], 0)  # the specified zone and the padded cell
    np.testing.assert_array_equal(w.sources.scalars, q_before)
    assert seat.identity()["model"]["name"] == "const"
    assert seat.receipt()["applied_cells"] == 10


def test_replace_puts_the_model_rate_in_place_of_the_schemes(tmp_path):
    w = _world()
    seat = _applying(w, tmp_path, out_mean=np.r_[np.full(NZ, 2e-5), np.full(NZ, 1e-8)], mode="replace")
    seat.stage(w.sources, state=w.state, saved=w.saved, time_seconds=0.0)
    m = w.state.rho[:, :10]
    np.testing.assert_allclose(w.sources.scalars[0, :, :10] / m, 1e-8, rtol=1e-4)
    rates = hex_seat.decouple(np, rho_theta=w.sources.rho_theta[:, :10],
                              scalars={"qv": w.sources.scalars[0, :, :10]}, mass=m,
                              theta_m=w.saved.theta_m[:, :10], qv=w.state.scalars[0, :, :10])
    np.testing.assert_allclose(rates["dtheta"], 2e-5, rtol=2e-3)


def test_residual_positivity_holds_for_the_sum(tmp_path):
    w = _world()
    w.state.scalars[0, :, :3] = 1e-9
    m, q = w.state.rho[:, :10], w.state.scalars[0, :, :10]
    q_scheme = q + 30.0 * w.sources.scalars[0, :, :10] / m
    seat = _applying(w, tmp_path, out_mean=np.r_[np.zeros(NZ), np.full(NZ, -1e-3)], positivity=True)
    seat.stage(w.sources, state=w.state, saved=w.saved, time_seconds=0.0)
    q_next = q + 30.0 * w.sources.scalars[0, :, :10] / m
    # The model takes the vapour the scheme leaves, and no more; where the
    # scheme alone overdraws, the model adds nothing negative.
    np.testing.assert_array_less(np.minimum(q_scheme, 0) - 1e-9, q_next + 1e-12)
    assert float(np.abs(q_next).max()) < 1e-4 and float(q_next.max()) < float(q_scheme.max())


def test_out_of_range_columns_fall_back_to_the_scheme(tmp_path):
    w = _world()
    before = w.sources.rho_theta.copy()
    model = _artefact(tmp_path / "m", out_mean=1e-4, in_min=-1e30, in_max=1e30)
    manifest = json.loads((model / "manifest.json").read_text())
    n = len(manifest["bounds"]["input_max"])
    manifest["bounds"]["input_max"] = [1e30] * n
    manifest["bounds"]["input_max"][0] = 0.0  # theta at level 0 is always out of range
    (model / "manifest.json").write_text(json.dumps(manifest))
    seat = _seat(w, _settings(mode="residual", model=model, inputs=None, outputs=None))
    seat.stage(w.sources, state=w.state, saved=w.saved, time_seconds=0.0)
    np.testing.assert_array_equal(w.sources.rho_theta, before)
    assert seat.receipt()["columns_out_of_range"] == NC


# ------------------------------------------------------------- refusals --

@pytest.mark.parametrize("over, match", [
    (dict(scheme="pbl"), "only scheme = \"total\""),
    (dict(outputs=("dtheta", "du")), "edge-normal"),
    (dict(outputs=("dtheta", "dtheta_m")), "one of the two"),
    (dict(inputs=("theta", "rthraten")), "not in the hex vocabulary"),
    (dict(levels=NZ + 1), "model levels"),
])
def test_refused_settings(over, match):
    with pytest.raises(launch.NNConfigError, match=match):
        _seat(_world(), _settings(**over), tap=_Tap())


def test_refused_species_the_run_does_not_carry():
    w = _world()
    with pytest.raises(launch.NNConfigError, match="carries no such species"):
        HexColumnSeat(_settings(outputs=("dtheta", "dqg")), cp=np, names=("qv", "qc"), n_cells=NC, n_levels=NZ,
                      zgrid=w.zgrid, dt=30.0, tap=_Tap())


def test_applying_is_refused_in_exact_mode_and_under_wrf_exact(tmp_path, monkeypatch):
    w = _world()
    model = _artefact(tmp_path / "m")
    settings = _settings(mode="residual", model=model, inputs=None, outputs=None)
    with pytest.raises(launch.NNConfigError, match="physics-mode exact"):
        _seat(w, settings, mode="exact")
    monkeypatch.setenv("GPUWM_WRF_EXACT", "1")
    with pytest.raises(launch.NNConfigError, match="GPUWM_WRF_EXACT=1"):
        _seat(w, settings)
    _seat(w, _settings(), tap=_Tap(), mode="exact")  # a tap changes nothing: exact mode keeps it


def test_partitioned_lts_and_moving_runs_are_refused():
    with pytest.raises(NotImplementedError, match="same-host partition or an LTS"):
        hex_seat.refuse_unsupported(execution=object())
    with pytest.raises(NotImplementedError, match="moving region"):
        hex_seat.refuse_unsupported(moving=SimpleNamespace(control=object()))
    with pytest.raises(NotImplementedError, match="moving region"):
        hex_seat.refuse_unsupported(physics_map=object())
    hex_seat.refuse_unsupported(moving=SimpleNamespace(control=None))


def test_carried_identity_is_checked_on_resume(tmp_path):
    w = _world()
    seat = _applying(w, tmp_path, out_mean=0.0)
    identity = seat.identity()
    hex_seat.check_carried(identity, json.loads(json.dumps(identity)))
    hex_seat.check_carried(None, None)
    with pytest.raises(launch.NNConfigError, match="written without"):
        hex_seat.check_carried(identity, None)
    with pytest.raises(launch.NNConfigError, match="same --nn file"):
        hex_seat.check_carried(None, identity)
    with pytest.raises(launch.NNConfigError, match=r"differs .* \['fixer'\]"):
        hex_seat.check_carried(identity, {**identity, "fixer": "moisture"})


def test_attach_binds_the_seat_and_checks_the_carrier(tmp_path, monkeypatch):
    w = _world()
    toml = tmp_path / "nn.toml"
    toml.write_text('[nn]\nmode = "tap"\nscheme = "total"\ntap = "t.zarr"\ninputs = ["theta", "qv"]\n')
    monkeypatch.setenv(launch.ENV_CONFIG, str(toml))
    dycore = SimpleNamespace(atmosphere=SimpleNamespace(state=w.state, vertical=SimpleNamespace(zz=w.zz)))
    coupling = SimpleNamespace(session=SimpleNamespace(field=lambda pool, name: w.pools[(pool, name)]),
                               dycore=dycore, cp=np, names=NAMES)
    mesh = SimpleNamespace(arrays={"bdyMaskCell": BDY})
    pytest.importorskip("zarr")
    seat = hex_seat.attach(coupling, config=SimpleNamespace(mode="correct"), mesh=mesh, n_cells=NC,
                           vertical=SimpleNamespace(zgrid=w.zgrid), dt=30.0)
    assert coupling.nn_seat is seat and dycore.nn_seat is seat and seat.pool.tolist() == list(range(8))
    seat.close()
    with pytest.raises(NotImplementedError):
        hex_seat.attach(coupling, config=None, mesh=mesh, n_cells=NC, vertical=None, dt=30.0, execution=object())
    monkeypatch.delenv(launch.ENV_CONFIG)
    assert hex_seat.attach(coupling, config=None, mesh=mesh, n_cells=NC, vertical=None, dt=30.0) is None


def test_registry_names_the_hex_route():
    rows = launch.registry_rows()
    assert all("hex" in row["routes"] for row in rows.values())
    assert rows["residual"]["hex"]["schemes"] == ["total"]


# ----------------------------------------------------------------- door --

def _door_args(**over):
    import argparse
    from woof.hex.forecast_door import add_forecast_arguments
    parser = argparse.ArgumentParser(add_help=False)
    add_forecast_arguments(parser)
    arguments = parser.parse_args(["--nn", "nn.toml"])
    for key, value in over.items():
        setattr(arguments, key, value)
    return arguments


def test_the_forecast_door_takes_nn():
    from woof.hex.forecast_door import _refuse_nn_launch
    arguments = _door_args()
    assert str(arguments.nn) == "nn.toml"
    _refuse_nn_launch(arguments)


@pytest.mark.parametrize("over, match", [
    (dict(members="roster.json"), "--members"),
    (dict(samehost_devices="0,1"), "--samehost-devices"),
    (dict(hosts="a:2"), "--hosts"),
    (dict(fullstep_local_timestep=True), "--fullstep-local-timestep"),
])
def test_the_forecast_door_refuses_what_the_seat_cannot_attach_to(over, match):
    from woof.hex.forecast_door import ForecastDoorRefusal, _refuse_nn_launch
    with pytest.raises(ForecastDoorRefusal, match=match):
        _refuse_nn_launch(_door_args(**over))


# --------------------------------------------------------------- device --

@pytest.mark.gpu
def test_the_seat_on_the_device_matches_the_host(tmp_path):
    """The seat's gathers, slice writes and masks on CuPy (CUDA) or the
    Metal facade give the host's numbers."""
    from conftest import HAS_GPU
    from woof import backend

    if not HAS_GPU and backend.requested() != "mlx":
        pytest.skip("needs a CUDA card or the Metal backend")
    cp = pytest.importorskip("cupy")

    def device(world):
        on = lambda a: cp.asarray(a)
        world.state = SimpleNamespace(rho=on(world.state.rho), scalars=on(world.state.scalars))
        world.saved = SimpleNamespace(theta_m=on(world.saved.theta_m), exner=on(world.saved.exner),
                                      vertical_velocity=on(world.saved.vertical_velocity))
        world.sources = SimpleNamespace(rho_theta=on(world.sources.rho_theta), scalars=on(world.sources.scalars))
        world.pools = {k: on(v) for k, v in world.pools.items()}
        return world

    host = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)
    model = _artefact(tmp_path / "m", out_mean=np.r_[np.full(NZ, 1e-4), np.full(NZ, -1e-8)])
    settings = _settings(mode="residual", model=model, inputs=None, outputs=None)
    results = []
    for xp, prepare in ((np, lambda w: w), (cp, device)):
        w = prepare(_world())
        seat = HexColumnSeat(settings, cp=xp, names=NAMES, n_cells=NC, n_levels=NZ, zgrid=w.zgrid, zz=w.zz,
                             bdy_mask=BDY, dt=30.0, fields=lambda pool, name, w=w: w.pools[(pool, name)])
        seat.stage(w.sources, state=w.state, saved=w.saved, time_seconds=0.0)
        tap = _Tap()
        tapper = HexColumnSeat(_settings(), cp=xp, names=NAMES, n_cells=NC, n_levels=NZ, zgrid=w.zgrid, zz=w.zz,
                               bdy_mask=BDY, dt=30.0, fields=lambda pool, name, w=w: w.pools[(pool, name)], tap=tap)
        tapper.stage(w.sources, state=w.state, saved=w.saved, time_seconds=30.0)
        results.append((host(w.sources.rho_theta), host(w.sources.scalars), tap.rows[0][0], tap.rows[0][1]))
    for a, b in zip(*results):
        np.testing.assert_allclose(b, a, rtol=2e-6, atol=1e-12)
