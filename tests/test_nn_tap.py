"""woof.nn P2.2: the --nn settings, the tendency tap store and the WRF seat.

The seat's contract has two halves.  It records: the targets it writes
are the held tendencies decoupled by the coupler's own unit response, so
coupling them again returns the held words on the interior.  And it
records only: a tapped forecast is byte-identical to the same forecast
without --nn (every prognostic array compared after real steps).
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from woof.nn import launch
from woof.nn.manifest import FieldSpec
from woof.nn.tap import SCHEMA, TapError, TendencyTap, read_tap

HAS_ZARR = importlib.util.find_spec("zarr") is not None
HAS_ICECHUNK = importlib.util.find_spec("icechunk") is not None


def _toml(tmp_path, body, name="nn.toml"):
    path = tmp_path / name
    path.write_text(body)
    return path


# ---------------------------------------------------------------- settings
def test_settings_read_and_anchor_the_store_at_the_file(tmp_path):
    s = launch.read_settings(_toml(tmp_path, '[nn]\nscheme = "pbl"\ntap = "rec.zarr"\ntap_every = 3\n'))
    assert (s.mode, s.scheme, s.tap_every, s.tap_columns, s.tap_seed) == ("tap", "pbl", 3, 256, 0)
    assert s.tap == (tmp_path / "rec.zarr").resolve()
    assert s.inputs is None and s.outputs is None and s.levels is None


@pytest.mark.parametrize("body, words", [
    ('scheme = "pbl"\n', "needs tap"),
    ('scheme = "mp"\ntap = "a"\n', "scheme must be one of"),
    ('scheme = "pbl"\ntap = "a"\nmodel = "x"\n', "unknown [nn] keys"),
    ('scheme = "pbl"\nmode = "replace"\n', "P2.3"),
    ('scheme = "pbl"\ntap = "a"\ntap_every = 0\n', "positive integer"),
    ('scheme = "pbl"\ntap = "a"\ninputs = ["qv", "qv"]\n', "twice"),
])
def test_settings_refusals(tmp_path, body, words):
    with pytest.raises(launch.NNConfigError, match=words.replace("[", r"\[").replace("]", r"\]")):
        launch.read_settings(_toml(tmp_path, "[nn]\n" + body))
    with pytest.raises(launch.NNConfigError, match="no \\[nn\\] table"):
        launch.read_settings(_toml(tmp_path, "[other]\n", "other.toml"))


def test_launch_scope_exports_and_restores(tmp_path, monkeypatch):
    monkeypatch.delenv(launch.ENV_CONFIG, raising=False)
    path = _toml(tmp_path, '[nn]\nscheme = "pbl"\ntap = "rec.zarr"\n')
    with launch.launch_scope(path):
        assert launch.configured() == str(path.resolve())
        with pytest.raises(NotImplementedError, match="2 domains"):
            launch.refuse_multiple_domains(2)
        launch.refuse_multiple_domains(1)
    assert launch.configured() is None
    with pytest.raises(NotImplementedError, match="ensemble members"):
        with launch.launch_scope(path, members=4):
            pass
    with launch.launch_scope(None):
        assert launch.configured() is None


def test_cli_parsers_take_nn():
    import argparse

    parser = argparse.ArgumentParser()
    launch.add_arguments(parser)
    assert parser.parse_args(["--nn", "x.toml"]).nn == "x.toml"
    assert parser.parse_args([]).nn is None


# ---------------------------------------------------------------- store
SPECS_IN = (FieldSpec("theta", "K", 3), FieldSpec("hfx", "W m-2", 1))
SPECS_OUT = (FieldSpec("dtheta", "K s-1", 3),)


def _sample(rng, ncol=5):
    return (rng.normal(size=(4, ncol)).astype(np.float32), rng.normal(size=(3, ncol)).astype(np.float32),
            np.arange(ncol) * 7)


@pytest.mark.skipif(not HAS_ZARR, reason="zarr not installed")
@pytest.mark.parametrize("suffix", [".zarr", ".icechunk"])
def test_tap_round_trip_resume_and_refusals(tmp_path, suffix):
    if suffix == ".icechunk" and not HAS_ICECHUNK:
        pytest.skip("icechunk not installed")
    rng = np.random.default_rng(0)
    path = tmp_path / ("rec" + suffix)
    attrs = {"route": "wrf", "scheme": "pbl", "grid": [3, 4, 5]}
    tap = TendencyTap(path, inputs=SPECS_IN, outputs=SPECS_OUT, columns=5, attrs=attrs, chunk=2)
    written = [_sample(rng) for _ in range(5)]
    for k, (x, y, idx) in enumerate(written):
        tap.append(x, y, time_seconds=60.0 * k, step=k, column_index=idx)
    assert tap.samples == 4  # two chunks flushed, one sample buffered
    with pytest.raises(TapError, match="sample shapes"):
        tap.append(np.zeros((4, 6)), np.zeros((3, 6)), time_seconds=0, step=0, column_index=np.zeros(6))
    tap.close()
    data = read_tap(path)
    assert data["attrs"]["schema"] == SCHEMA and data["attrs"]["scheme"] == "pbl"
    assert data["inputs_spec"] == SPECS_IN and data["outputs_spec"] == SPECS_OUT
    np.testing.assert_array_equal(data["inputs"], np.stack([w[0] for w in written]))
    np.testing.assert_array_equal(data["targets"], np.stack([w[1] for w in written]))
    np.testing.assert_array_equal(data["step"], np.arange(5))
    np.testing.assert_array_equal(data["column_index"][2], written[2][2])

    with pytest.raises(TapError, match="already exists"):
        TendencyTap(path, inputs=SPECS_IN, outputs=SPECS_OUT, columns=5, attrs=attrs)
    with pytest.raises(TapError, match="different scheme"):
        TendencyTap(path, inputs=SPECS_IN, outputs=SPECS_OUT, columns=5,
                    attrs={**attrs, "scheme": "radiation"}, resume=True)
    again = TendencyTap(path, inputs=SPECS_IN, outputs=SPECS_OUT, columns=5, attrs=attrs, resume=True)
    x, y, idx = _sample(rng)
    again.append(x, y, time_seconds=300.0, step=5, column_index=idx)
    again.close()
    assert read_tap(path)["inputs"].shape == (6, 4, 5)


# ---------------------------------------------------------------- seat, host
def _host_state(nz=4, ny=6, nx=7, seed=3):
    """Arrays on the host and a stand-in coupler: enough to check the
    vocabulary, the C-to-A averages and the decoupling arithmetic."""
    rng = np.random.default_rng(seed)
    f = lambda *s: rng.uniform(0.5, 1.5, s).astype(np.float32)
    chm = f(nz, ny, nx) * 900.0
    msft = f(ny, nx)
    phys = {"rtheta": f(nz, ny, nx) * 1e-4, "rqv": f(nz, ny, nx) * 1e-7, "ru": f(nz, ny, nx + 1),
            "rv": f(nz, ny + 1, nx)}
    held = SimpleNamespace(rtheta=phys["rtheta"] * chm / msft, rqv=phys["rqv"] * chm, rqc=np.zeros_like(chm),
                           rqr=None, rqi=None, rqs=None, ru=phys["ru"], rv=phys["rv"])
    fields = {k: f(ny, nx) for k in ("hfx", "qfx", "ust", "tsk", "znt", "xland")}
    state = SimpleNamespace(
        p=f(nz, ny, nx) * 9e4, u=f(nz, ny, nx + 1), v=f(nz, ny + 1, nx), w=f(nz + 1, ny, nx),
        phb=np.linspace(0, 9.81 * 4000, nz + 1).astype(np.float32), php=f(nz + 1, ny, nx),
        ht=f(ny, nx) * 100, qv=f(nz, ny, nx) * 1e-2, qc=f(nz, ny, nx) * 1e-4, elapsed_seconds=120.0,
        physics=SimpleNamespace(fields=fields, pbl_tendencies=held, radiation_tendencies=held),
        total_theta=lambda: theta, total_mu=lambda: f(ny, nx))
    theta = f(nz, ny, nx) * 300

    def couple(state, cfg, *, rtheta, rqv):
        return SimpleNamespace(rtheta=rtheta * chm / msft, rqv=rqv * chm)

    return state, couple, phys, chm


def _cfg(**over):
    base = dict(dt=60.0, specified=False, nested=False, spec_bdy_width=5, bl_pbl_physics=1, cu_physics=0)
    base.update(over)
    return SimpleNamespace(**base)


def _settings(**over):
    base = dict(source=None, mode="tap", scheme="pbl", inputs=None, outputs=None, levels=None, tap=None,
                tap_every=1, tap_columns=10, tap_seed=0)
    base.update(over)
    return launch.NNSettings(**base)


class _ListTap:
    def __init__(self):
        self.rows, self.samples, self.path = [], 0, "memory"

    def append(self, x, y, **kw):
        self.rows.append((x, y, kw))
        self.samples += 1

    def close(self):
        pass


def test_seat_vocabulary_and_decoupling_on_host(monkeypatch):
    from woof.nn import wrf_seat

    monkeypatch.setattr(wrf_seat.WrfColumnSeat, "_check_available", lambda *a: None)
    state, couple, phys, chm = _host_state()
    nz, ny, nx = state.p.shape
    seat = wrf_seat.WrfColumnSeat(state, _cfg(), _settings(levels=3), couple=couple, tap=_ListTap())
    flat = np.array([0, 8, 41])
    x, y = seat.columns(state, _cfg(), None, flat)
    j, i = np.divmod(flat, nx)
    names = [s.name for s in seat.inputs]
    assert names == list(wrf_seat.DEFAULT_INPUTS["pbl"]) and x.shape == (7 * 3 + 6, 3)
    row = {n: k * 3 for k, n in enumerate(names[:7])}
    np.testing.assert_allclose(x[row["u"]:row["u"] + 3], 0.5 * (state.u[:3, j, i] + state.u[:3, j, i + 1]))
    np.testing.assert_allclose(x[row["v"]:row["v"] + 3], 0.5 * (state.v[:3, j, i] + state.v[:3, j + 1, i]))
    ph = state.phb[:4, None] + state.php[:4, j, i]
    np.testing.assert_allclose(x[row["z"]:row["z"] + 3], 0.5 * (ph[:-1] + ph[1:]) / 9.81 - state.ht[j, i],
                               rtol=1e-5)
    np.testing.assert_allclose(x[-1], state.physics.fields["xland"][j, i])
    # dtheta, dqv: the exact inverse of the coupling.
    np.testing.assert_allclose(y[0:3], phys["rtheta"][:3, j, i], rtol=1e-5)
    np.testing.assert_allclose(y[3:6], phys["rqv"][:3, j, i], rtol=1e-5)
    np.testing.assert_array_equal(y[6:9], 0.0)
    np.testing.assert_allclose(y[9:12], 0.5 * (phys["ru"][:3, j, i] + phys["ru"][:3, j, i + 1]) / chm[:3, j, i],
                               rtol=1e-5)

    held = object()
    seat2 = wrf_seat.WrfColumnSeat(state, _cfg(), _settings(tap_every=2), couple=couple, tap=_ListTap())
    for _ in range(5):
        assert seat2.after_nonmicrophysics(state, _cfg(), held) is held
    assert len(seat2.tap.rows) == 3 and seat2.tap.rows[0][2]["step"] == 2


def test_interior_pool_excludes_the_ring():
    from woof.nn.wrf_seat import interior_pool

    pool = interior_pool(_cfg(specified=True, spec_bdy_width=2), 8, 9)
    j, i = np.divmod(pool, 9)
    assert j.min() == 2 and j.max() == 5 and i.min() == 2 and i.max() == 6
    assert interior_pool(_cfg(), 8, 9).size == 72
    with pytest.raises(launch.NNConfigError, match="no interior"):
        interior_pool(_cfg(specified=True, spec_bdy_width=4), 8, 9)


def test_seat_refuses_what_the_run_cannot_supply(monkeypatch):
    from woof.nn import wrf_seat

    state, couple, _, _ = _host_state()
    monkeypatch.setattr(wrf_seat.WrfColumnSeat, "_check_available", lambda *a: None)
    with pytest.raises(launch.NNConfigError, match="not in the vocabulary"):
        wrf_seat.WrfColumnSeat(state, _cfg(), _settings(inputs=("theta", "cape")), couple=couple, tap=_ListTap())
    with pytest.raises(launch.NNConfigError, match="model levels"):
        wrf_seat.WrfColumnSeat(state, _cfg(), _settings(levels=9), couple=couple, tap=_ListTap())
    with pytest.raises(NotImplementedError, match="streamed"):
        wrf_seat.attach(state, _cfg(), streamed=True, settings=_settings())


# ---------------------------------------------------------------- seat, device
_BASE = dict(nx=8, ny=8, nz=30, dx=12000.0, dy=12000.0, ztop=16000.0, dt=60.0, run_seconds=0.0,
             moist=True, mp_physics=6, bldt=0.0, bl_pbl_physics=1, sf_sfclay_physics=91)


def _device_case():
    cp = pytest.importorskip("cupy")
    from woof.config import RunConfig, validate_run_config
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import DECLARED_CONSTANT_GLW_WM2, initialize_physics

    cfg = validate_run_config(RunConfig(**_BASE))
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 296.0 + 0.004 * np.asarray(z), p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, lambda z: 0.018 * np.exp(-np.asarray(z) / 3500.0))
    initialize_physics(state, cfg, landmask=1.0, tsk=305.0, glw=DECLARED_CONSTANT_GLW_WM2, swdown=0.0)
    nz, ny, nx = state.p.shape
    y = np.arange(ny)[None, :, None]
    x = np.arange(nx)[None, None, :]
    wave = np.sin(2 * np.pi * x / nx) * np.cos(2 * np.pi * y / ny)
    state.thp += cp.asarray(np.broadcast_to(0.5 * wave, (nz, ny, nx)).copy(), dtype=cp.float32)
    state.u[...] = cp.float32(5.0)
    return cp, cfg, state


_PROGNOSTIC = ("u", "v", "w", "thp", "php", "mup", "qv", "qc", "qr", "qi", "qs", "qg")


def _snapshot(cp, state):
    out = {name: cp.asnumpy(getattr(state, name)) for name in _PROGNOSTIC if getattr(state, name, None) is not None}
    out.update({f"fields/{k}": cp.asnumpy(v) for k, v in state.physics.fields.items()
                if hasattr(v, "shape") and getattr(v, "dtype", None) is not None})
    return out


@pytest.mark.gpu
def test_a_tapped_run_is_byte_identical_and_its_targets_recouple(tmp_path):
    from conftest import HAS_GPU
    from woof import backend

    if not HAS_GPU and backend.requested() != "mlx":
        pytest.skip("needs a CUDA card or the Metal backend")
    if not HAS_ZARR:
        pytest.skip("zarr not installed")
    from woof.core import dycore
    from woof.core.physics import couple_column_tendencies
    from woof.nn import wrf_seat

    steps = 4
    cp, cfg, plain = _device_case()
    for _ in range(steps):
        dycore.step(plain, cfg)
    reference = _snapshot(cp, plain)

    cp, cfg, tapped = _device_case()
    settings = _settings(tap=tmp_path / "rec.zarr", tap_columns=16, levels=20)
    seat = wrf_seat.attach(tapped, cfg, settings=settings)
    assert tapped._nn_column_seat is seat
    for _ in range(steps):
        dycore.step(tapped, cfg)
    summary = seat.close()
    assert summary["calls"] == steps and summary["samples"] == steps
    after = _snapshot(cp, tapped)
    assert sorted(after) == sorted(reference)
    for name, value in reference.items():
        assert after[name].tobytes() == value.tobytes(), f"{name} differs with the tap on"

    data = read_tap(tmp_path / "rec.zarr")
    assert data["inputs"].shape == (steps, sum(s.levels for s in seat.inputs), 16)
    assert np.isfinite(data["inputs"]).all() and np.isfinite(data["targets"]).all()
    assert np.abs(data["targets"][:, :20]).max() > 0.0  # YSU heats or cools somewhere

    # Recoupling every interior column's dtheta/dqv gives the held words back.
    nz, ny, nx = tapped.p.shape
    flat = np.arange(ny * nx)
    seat.levels = nz
    seat.output_names = ("dtheta", "dqv")
    _, y = seat.columns(tapped, cfg, None, flat)
    y = cp.asnumpy(y)
    rtheta = cp.asarray(y[:nz].reshape(nz, ny, nx))
    rqv = cp.asarray(y[nz:].reshape(nz, ny, nx))
    again = couple_column_tendencies(tapped, cfg, rtheta=rtheta, rqv=rqv)
    held = tapped.physics.pbl_tendencies
    np.testing.assert_allclose(cp.asnumpy(again.rtheta), cp.asnumpy(held.rtheta), rtol=2e-6, atol=1e-12)
    np.testing.assert_allclose(cp.asnumpy(again.rqv), cp.asnumpy(held.rqv), rtol=2e-6, atol=1e-15)
