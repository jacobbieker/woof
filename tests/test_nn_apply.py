"""woof.nn P2.3: applying a column model on the WRF-route seat.

``mode = "residual"`` adds the model's rate to the scheme's, ``"replace"``
puts it in the scheme's place; both go through the couplers' own path and
leave the step's tendencies object (and the scheme's slot) untouched.  The
safety policy decides per column: a column outside the trained range keeps
the scheme (``fallback``) or stops the run (``refuse``).  An applying run is
not WRF-identical: ``GPUWM_WRF_EXACT=1`` refuses it, and a checkpoint binds
the artefact and the settings so a resume cannot splice two trajectories.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from woof.nn import launch
from woof.nn.manifest import FieldSpec, write_mlp_artefact

INPUTS = ("theta", "qv", "u", "v", "p", "hfx")
OUTPUTS = ("dtheta", "dqv")
LEVELS = 3


def _specs(levels=LEVELS):
    from woof.nn.wrf_seat import OUTPUTS as UNITS, VOCABULARY

    ins = tuple(FieldSpec(n, VOCABULARY[n][0], levels if VOCABULARY[n][1] == "3d" else 1) for n in INPUTS)
    outs = tuple(FieldSpec(n, UNITS[n][0], levels) for n in OUTPUTS)
    return ins, outs


def _artefact(path, *, out_mean=0.0, in_min=-1e30, in_max=1e30, levels=LEVELS, name="const"):
    """An MLP whose weights are zero: it returns ``out_mean`` everywhere."""
    ins, outs = _specs(levels)
    n_in, n_out = sum(f.levels for f in ins), sum(f.levels for f in outs)
    layers = [(np.zeros((4, n_in)), np.zeros(4)), (np.zeros((n_out, 4)), np.zeros(n_out))]
    mean = np.broadcast_to(np.asarray(out_mean, dtype=np.float64), (n_out,)).copy()
    write_mlp_artefact(path, name=name, inputs=ins, outputs=outs, layers=layers, activation="tanh",
                       input_mean=np.zeros(n_in), input_std=np.ones(n_in), output_mean=mean,
                       output_std=np.ones(n_out), input_min=np.full(n_in, in_min),
                       input_max=np.full(n_in, in_max))
    return path


def _settings(**over):
    base = dict(source=None, mode="residual", scheme="pbl", inputs=None, outputs=None, levels=None, tap=None,
                tap_every=1, tap_columns=10, tap_seed=0, backend="numpy", limit_outputs=False,
                positivity=False)
    base.update(over)
    return launch.NNSettings(**base)


def _host_state(nz=4, ny=6, nx=7, seed=5):
    """Host arrays and a stand-in coupler (scalars times chm, theta also
    over msft), enough to check the apply arithmetic."""
    rng = np.random.default_rng(seed)
    f = lambda *s: rng.uniform(0.5, 1.5, s).astype(np.float32)
    chm, msft = f(nz, ny, nx) * 900.0, f(ny, nx)
    held = SimpleNamespace(rtheta=f(nz, ny, nx) * 1e-4 * chm / msft, rqv=(f(nz, ny, nx) - 1.0) * 1e-6 * chm,
                           rqc=np.zeros_like(chm), rqr=None, rqi=None, rqs=None,
                           ru=f(nz, ny, nx + 1), rv=f(nz, ny + 1, nx))
    fields = {"hfx": f(ny, nx) * 100.0}
    state = SimpleNamespace(
        p=f(nz, ny, nx) * 9e4, u=f(nz, ny, nx + 1), v=f(nz, ny + 1, nx), w=f(nz + 1, ny, nx),
        ht=f(ny, nx), qv=f(nz, ny, nx) * 1e-5, qc=f(nz, ny, nx) * 1e-5, elapsed_seconds=0.0,
        dnw=-np.full(nz, 1.0 / nz, dtype=np.float32),
        physics=SimpleNamespace(fields=fields, pbl_tendencies=held, call_counts={"ysu": 0}),
        total_theta=lambda: theta)
    theta = f(nz, ny, nx) * 300.0

    def couple(state, cfg, *, rtheta=None, rqv=None, ru=None, rv=None, **other):
        assert not any(v is not None for v in other.values()), sorted(other)
        return SimpleNamespace(rtheta=None if rtheta is None else rtheta * chm / msft,
                               rqv=None if rqv is None else rqv * chm, ru=None, rv=None)

    return state, couple, held, chm, msft


def _cfg():
    return SimpleNamespace(dt=60.0, specified=False, nested=False, spec_bdy_width=5, bl_pbl_physics=1,
                           cu_physics=0)


def _seat(monkeypatch, tmp_path, settings_over=None, **artefact):
    from woof.nn import wrf_seat

    monkeypatch.setattr(wrf_seat.WrfColumnSeat, "_check_available", lambda *a: None)
    state, couple, held, chm, msft = _host_state()
    model = _artefact(tmp_path / "model", **artefact)
    seat = wrf_seat.WrfColumnSeat(state, _cfg(), _settings(model=model, **(settings_over or {})), couple=couple)
    return seat, state, held, chm, msft


# ---------------------------------------------------------------- settings
def test_applying_settings_parse_and_refuse(tmp_path, monkeypatch):
    monkeypatch.delenv("GPUWM_WRF_EXACT", raising=False)
    path = tmp_path / "nn.toml"
    path.write_text('[nn]\nmode = "replace"\nscheme = "pbl"\nmodel = "m"\nfixer = "moisture"\nmargin = 0.1\n')
    s = launch.read_settings(path)
    assert s.applies and s.model == (tmp_path / "m").resolve() and s.fixer == "moisture" and s.margin == 0.1
    assert s.identity()["mode"] == "replace" and s.out_of_range == "fallback"
    for body, words in [('mode = "residual"\nscheme = "pbl"\n', "needs model"),
                        ('mode = "residual"\nscheme = "pbl"\nmodel = "m"\ntap = "a"\n', "does not take"),
                        ('mode = "tap"\nscheme = "pbl"\ntap = "a"\nmodel = "m"\n', "does not take"),
                        ('mode = "residual"\nscheme = "pbl"\nmodel = "m"\nfixer = "x"\n', "fixer must be"),
                        ('mode = "residual"\nscheme = "pbl"\nmodel = "m"\nmargin = 2\n', "margin must be"),
                        ('mode = "mean"\nscheme = "pbl"\n', "mode must be one of")]:
        path.write_text("[nn]\n" + body)
        with pytest.raises(launch.NNConfigError, match=words):
            launch.read_settings(path)
    path.write_text('[nn]\nmode = "residual"\nscheme = "pbl"\nmodel = "m"\n')
    monkeypatch.setenv("GPUWM_WRF_EXACT", "1")
    with pytest.raises(launch.NNConfigError, match="not WRF-identical"):
        launch.read_settings(path)
    path.write_text('[nn]\nscheme = "pbl"\ntap = "a"\n')
    assert launch.read_settings(path).mode == "tap"  # a tap changes nothing: allowed


# ---------------------------------------------------------------- apply, host
def test_residual_adds_the_coupled_rate_and_leaves_the_step_alone(monkeypatch, tmp_path):
    seat, state, held, chm, msft = _seat(monkeypatch, tmp_path, out_mean=[1e-4] * LEVELS + [0.0] * LEVELS)
    before = copy.deepcopy(held.__dict__)
    out = seat.after_nonmicrophysics(state, _cfg(), held)
    assert out is not held
    for name, value in before.items():  # the scheme's slot is never written
        if value is not None:
            np.testing.assert_array_equal(getattr(held, name), value)
    np.testing.assert_allclose(out.rtheta[:LEVELS] - held.rtheta[:LEVELS], (1e-4 * chm / msft)[:LEVELS],
                               rtol=1e-4)
    np.testing.assert_array_equal(out.rtheta[LEVELS:], held.rtheta[LEVELS:])  # above the model's levels
    np.testing.assert_array_equal(out.rqv, held.rqv)  # a zero dqv adds nothing
    assert out.ru is held.ru and out.rv is held.rv  # slots the model does not write are shared
    # The seat carries nothing from step to step: a fresh seat computes the
    # same increment from the same state (a resume is the unbroken run).
    again = seat.after_nonmicrophysics(state, _cfg(), held)
    np.testing.assert_array_equal(again.rtheta, out.rtheta)
    assert seat.identity()["model"]["weights_sha256"] and seat.identity()["mode"] == "residual"


def test_replace_swaps_the_scheme_for_the_model(monkeypatch, tmp_path):
    seat, state, held, chm, msft = _seat(monkeypatch, tmp_path, settings_over={"mode": "replace"},
                                         out_mean=[2e-4] * LEVELS + [0.0] * LEVELS)
    out = seat.after_nonmicrophysics(state, _cfg(), held)
    # The model's rate in place of the scheme's on its levels, the scheme above.
    np.testing.assert_allclose(out.rtheta[:LEVELS], (2e-4 * chm / msft)[:LEVELS], rtol=1e-4)
    np.testing.assert_allclose(out.rqv[:LEVELS], 0.0, atol=1e-6 * float(np.abs(held.rqv).max()))
    np.testing.assert_array_equal(out.rtheta[LEVELS:], held.rtheta[LEVELS:])


def test_out_of_range_columns_keep_the_scheme_or_stop(monkeypatch, tmp_path):
    from woof.nn.safety import InputOutOfRange

    seat, state, held, *_ = _seat(monkeypatch, tmp_path, settings_over={"mode": "replace"},
                                  out_mean=1.0, in_min=0.0, in_max=1e-3)
    out = seat.after_nonmicrophysics(state, _cfg(), held)
    for name in ("rtheta", "rqv"):  # every column falls back: the scheme's rate exactly
        np.testing.assert_array_equal(getattr(out, name), getattr(held, name))
    summary = seat.close()
    assert summary["columns_out_of_range"] == summary["columns_seen"] == 42
    seat, state, held, *_ = _seat(monkeypatch, tmp_path, settings_over={"out_of_range": "refuse"},
                                  out_mean=1.0, in_min=0.0, in_max=1e-3)
    with pytest.raises(InputOutOfRange, match="trained input range"):
        seat.after_nonmicrophysics(state, _cfg(), held)


def test_residual_positivity_holds_for_the_sum(monkeypatch, tmp_path):
    seat, state, held, chm, _ = _seat(monkeypatch, tmp_path, settings_over={"positivity": True},
                                      out_mean=[0.0] * LEVELS + [-1.0] * LEVELS)
    out = seat.after_nonmicrophysics(state, _cfg(), held)
    after = state.qv[:LEVELS] + _cfg().dt * out.rqv[:LEVELS] / chm[:LEVELS]
    floor = np.minimum(state.qv[:LEVELS] + _cfg().dt * held.rqv[:LEVELS] / chm[:LEVELS], 0.0)
    assert (after >= floor - 1e-6 * state.qv[:LEVELS]).all()
    assert (after < state.qv[:LEVELS]).any()  # the model did dry the column


def test_moisture_fixer_keeps_the_schemes_column_integral(monkeypatch, tmp_path):
    seat, state, held, chm, _ = _seat(monkeypatch, tmp_path,
                                      settings_over={"mode": "replace", "fixer": "moisture"},
                                      out_mean=[0.0] * LEVELS + [3e-6, -1e-6, 5e-7])
    out = seat.after_nonmicrophysics(state, _cfg(), held)
    dnw = np.abs(state.dnw[:LEVELS])[:, None, None]
    integral = lambda r: (r[:LEVELS] * dnw).sum(axis=0)  # chm * dq * |dnw|, the coupled column mass
    np.testing.assert_allclose(integral(out.rqv), integral(held.rqv), rtol=1e-3,
                               atol=1e-4 * float(np.abs(integral(held.rqv)).max()))


def test_a_model_the_run_cannot_feed_is_refused(monkeypatch, tmp_path):
    from woof.nn import wrf_seat

    monkeypatch.setattr(wrf_seat.WrfColumnSeat, "_check_available", lambda *a: None)
    state, couple, *_ = _host_state()
    deep = _artefact(tmp_path / "deep", levels=9)
    with pytest.raises(launch.NNConfigError, match="model levels"):
        wrf_seat.WrfColumnSeat(state, _cfg(), _settings(model=deep), couple=couple)
    monkeypatch.setenv("GPUWM_WRF_EXACT", "1")
    with pytest.raises(launch.NNConfigError, match="not WRF-identical"):
        wrf_seat.WrfColumnSeat(state, _cfg(), _settings(model=_artefact(tmp_path / "m")), couple=couple)


# ---------------------------------------------------------------- restart
def test_a_checkpoint_binds_the_artefact_and_settings(tmp_path, monkeypatch):
    from woof.io.restart import RestartMismatchError, _validate_nn_column_checkpoint
    from woof.nn.manifest import load_manifest

    monkeypatch.delenv("GPUWM_WRF_EXACT", raising=False)
    monkeypatch.delenv(launch.ENV_CONFIG, raising=False)
    model = _artefact(tmp_path / "model")
    path = tmp_path / "nn.toml"
    path.write_text('[nn]\nmode = "residual"\nscheme = "pbl"\nmodel = "model"\n')
    stored = {**launch.read_settings(path).identity(), "model": load_manifest(model).identity()}
    _validate_nn_column_checkpoint({})  # neither side applies a model
    with pytest.raises(RestartMismatchError, match="resume it with the same --nn"):
        _validate_nn_column_checkpoint({"nn_column_seat": stored})
    monkeypatch.setenv(launch.ENV_CONFIG, str(path))
    _validate_nn_column_checkpoint({"nn_column_seat": stored})
    with pytest.raises(RestartMismatchError, match="written without"):
        _validate_nn_column_checkpoint({})
    with pytest.raises(RestartMismatchError, match=r"\['fixer'\]"):
        _validate_nn_column_checkpoint({"nn_column_seat": {**stored, "fixer": "moisture"}})
    _artefact(tmp_path / "model", out_mean=1.0)  # retrained in place: new weights
    with pytest.raises(RestartMismatchError, match=r"\['model'\]"):
        _validate_nn_column_checkpoint({"nn_column_seat": stored})
    tap = tmp_path / "tap.toml"
    tap.write_text('[nn]\nscheme = "pbl"\ntap = "rec.zarr"\n')
    monkeypatch.setenv(launch.ENV_CONFIG, str(tap))
    _validate_nn_column_checkpoint({})  # a tap binds nothing


# ---------------------------------------------------------------- device
_BASE = dict(nx=8, ny=8, nz=30, dx=12000.0, dy=12000.0, ztop=16000.0, dt=60.0, run_seconds=0.0,
             moist=True, mp_physics=6, bldt=0.0, bl_pbl_physics=1, sf_sfclay_physics=91)


def _device_case(**over):
    cp = pytest.importorskip("cupy")
    from woof.config import RunConfig, validate_run_config
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import DECLARED_CONSTANT_GLW_WM2, initialize_physics

    cfg = validate_run_config(RunConfig(**{**_BASE, **over}))
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
    return {name: cp.asnumpy(getattr(state, name)) for name in _PROGNOSTIC
            if getattr(state, name, None) is not None}


def _device_backends():
    from woof import backend
    from woof.nn.models import default_backend

    names = ["numpy"]
    if backend.requested() == "mlx":
        names.append("mlx")
    else:
        import importlib.util
        if importlib.util.find_spec("torch") is not None:
            names.append("torch")
    assert default_backend() in names or default_backend() == "torch"
    return names


@pytest.mark.gpu
def test_applying_on_the_device(tmp_path):
    from conftest import HAS_GPU
    from woof import backend

    if not HAS_GPU and backend.requested() != "mlx":
        pytest.skip("needs a CUDA card or the Metal backend")
    from woof.core import dycore
    from woof.nn import wrf_seat

    steps = 4
    over = {}
    cp, cfg, plain = _device_case()
    for _ in range(steps):
        dycore.step(plain, cfg)
    reference = _snapshot(cp, plain)

    for runner in _device_backends():
        # A zero residual changes no prognostic value.
        cp, cfg, state = _device_case(**over)
        zero = _artefact(tmp_path / f"zero-{runner}", levels=20)
        seat = wrf_seat.attach(state, cfg, settings=_settings(model=zero, backend=runner))
        for _ in range(steps):
            dycore.step(state, cfg)
        assert seat.close()["calls"] == steps
        after = _snapshot(cp, state)
        for name, value in reference.items():
            np.testing.assert_array_equal(after[name], value, err_msg=f"{runner}: {name}")

        # A warming residual warms; replace with a zero rate stays finite.
        for mode, mean in (("residual", [1e-4] * 20 + [0.0] * 20), ("replace", 0.0)):
            cp, cfg, state = _device_case(**over)
            model = _artefact(tmp_path / f"{mode}-{runner}", levels=20, out_mean=mean)
            seat = wrf_seat.attach(state, cfg, settings=_settings(model=model, backend=runner, mode=mode,
                                                                  positivity=True))
            for _ in range(steps):
                dycore.step(state, cfg)
            seat.close()
            after = _snapshot(cp, state)
            assert all(np.isfinite(v).all() for v in after.values()), (runner, mode)
            if mode == "residual":
                warmed = after["thp"][:20] - reference["thp"][:20]
                expect = steps * cfg.dt * 1e-4
                print(f"{runner}: residual warming {np.median(warmed):.3e} K (expect {expect:.3e})")
                assert np.median(warmed) == pytest.approx(expect, rel=0.2), (runner, np.median(warmed))


def test_review_refusals_identity_and_fallback_faces(monkeypatch, tmp_path):
    from woof.nn import wrf_seat
    from woof.nn.models import default_backend

    seat, state, held, *_ = _seat(monkeypatch, tmp_path, settings_over={"backend": "auto"})
    assert seat.identity()["backend"] == default_backend() != "auto"
    # A face beside a fallback column takes the scheme's word alone.
    nz, ny, nx = seat.shape
    bad = np.zeros(ny * nx, dtype=bool)
    bad[2 * nx + 3] = True
    u, v = seat._fallback_faces(np, bad)
    assert u.shape == (ny, nx + 1) and v.shape == (ny + 1, nx)
    assert sorted(zip(*np.nonzero(u))) == [(2, 3), (2, 4)]
    assert sorted(zip(*np.nonzero(v))) == [(2, 3), (3, 3)]
    # SPPT/SKEBS act before the seat: an applying seat is refused there.
    state._ensemble_stochastic = object()
    model = tmp_path / "model"
    with pytest.raises(NotImplementedError, match="SPPT/SKEBS"):
        wrf_seat.WrfColumnSeat(state, _cfg(), _settings(model=model), couple=lambda *a, **k: None)
    del state._ensemble_stochastic
    # An output naming moisture the run does not carry is refused at attach.
    state.qv = None
    with pytest.raises(launch.NNConfigError, match="does not carry"):
        wrf_seat.WrfColumnSeat(state, _cfg(), _settings(model=model), couple=lambda *a, **k: None)


def test_the_all_column_slices_match_the_indexed_gather():
    from woof.nn.wrf_seat import _Gather

    rng = np.random.default_rng(1)
    ny, nx, nz = 5, 6, 3
    flat = np.arange(ny * nx)
    a, b = _Gather.build(np, flat, nx), _Gather.full(np, ny, nx)
    fields = {"mass": rng.random((nz, ny, nx)), "u": rng.random((nz, ny, nx + 1)),
              "v": rng.random((nz, ny + 1, nx)), "w": rng.random((nz + 1, ny, nx)), "ht": rng.random((ny, nx))}
    np.testing.assert_array_equal(a.take(fields["mass"], levels=2), b.take(fields["mass"], levels=2))
    np.testing.assert_array_equal(a.take(fields["w"], levels=4), b.take(fields["w"], levels=4))
    np.testing.assert_array_equal(a.take(fields["ht"]), b.take(fields["ht"]))
    for name, field in (("u_west", "u"), ("u_east", "u"), ("v_south", "v"), ("v_north", "v")):
        np.testing.assert_array_equal(a.take(fields[field], getattr(a, name), 2),
                                      b.take(fields[field], getattr(b, name), 2))
