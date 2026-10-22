# tests/test_smoke_tracer.py
"""The passive smoke tracer (RunConfig.smoke_tracer, woof/core/smoke_tracer.py).

CPU half: the configuration surface (field order, refusals, identity and
checkpoint inertness when off), the registration tables every enforced list
reads, and the emission reader.  GPU half: an idealised puff in uniform flow
is carried downwind, stays non-negative and keeps its mass; a surface source
injects exactly the mass its flux says; the PBL solve conserves a column;
and an off run allocates nothing.
"""
from dataclasses import fields, replace

import numpy as np
import pytest
from conftest import requires_gpu

from woof.config import KM_OPT_ZERO_ACK, RunConfig, validate_run_config


def _cfg(**kw):
    kw.setdefault("nx", 40)
    kw.setdefault("ny", 8)
    kw.setdefault("nz", 10)
    kw.setdefault("dx", 1000.0)
    kw.setdefault("dy", 1000.0)
    kw.setdefault("ztop", 10000.0)
    kw.setdefault("dt", 10.0)
    kw.setdefault("run_seconds", 0.0)
    if kw.get("km_opt") == 0:
        # The idealised transport tests isolate advection: no mixing
        # operator, acknowledged as the research control it is.
        kw.setdefault("km_opt_zero_acknowledgement", KM_OPT_ZERO_ACK)
    return RunConfig(**kw)


# --------------------------------------------------------------- config ---

def test_fields_appended_last_and_default_off():
    names = [f.name for f in fields(RunConfig)]
    assert names[-2:] == ["smoke_tracer", "smoke_emission_file"]
    cfg = _cfg()
    assert cfg.smoke_tracer is False
    assert cfg.smoke_emission_file == ""


@pytest.mark.parametrize("kw, match", [
    ({"smoke_tracer": 1}, "smoke_tracer=1 must be true or false"),
    ({"smoke_emission_file": "e.npy"}, "needs smoke_tracer = true"),
    ({"smoke_tracer": True, "smoke_emission_file": "e.txt"},
     "must name a .npy or .nc file"),
    ({"smoke_tracer": True, "smoke_emission_file": 3},
     "must be a file path string"),
])
def test_config_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        validate_run_config(_cfg(**kw))


def test_config_admits_tracer_with_and_without_file():
    validate_run_config(_cfg(smoke_tracer=True))
    validate_run_config(_cfg(smoke_tracer=True,
                             smoke_emission_file="source.nc"))


def test_load_config_anchors_relative_emission_path(tmp_path):
    from woof.config import load_config

    np.save(tmp_path / "emis.npy", np.zeros((8, 40), dtype=np.float32))
    path = tmp_path / "case.toml"
    path.write_text(
        "[grid]\nnx = 40\nny = 8\nnz = 10\ndx = 1000.0\ndy = 1000.0\n"
        "ztop = 10000.0\n[run]\ndt = 10.0\nrun_seconds = 0.0\n"
        "smoke_tracer = true\nsmoke_emission_file = \"emis.npy\"\n",
        encoding="utf-8")
    cfg = load_config(path)
    assert cfg.smoke_emission_file == str((tmp_path / "emis.npy").resolve())


def test_off_keys_drop_from_checkpoint_echo():
    from woof.io import restart

    on = restart.configuration_echo(_cfg(smoke_tracer=True))
    off = restart.configuration_echo(_cfg())
    assert "smoke_tracer" not in off
    assert "smoke_emission_file" not in off
    assert on["smoke_tracer"] is True
    assert "smoke_emission_file" not in on


def test_preparation_inert_and_per_domain_override():
    from woof.experiment import _DOMAIN_RUN_OVERRIDES
    from woof.ingest.prepared_cache import PREPARATION_INERT_RUN_FIELDS

    assert {"run.smoke_tracer", "run.smoke_emission_file"} <= \
        PREPARATION_INERT_RUN_FIELDS
    assert "smoke_emission_file" in _DOMAIN_RUN_OVERRIDES
    assert "smoke_tracer" not in _DOMAIN_RUN_OVERRIDES


# ------------------------------------------------------- registrations ---

def test_inventory_and_restart_classification():
    from woof.core.device_inventory import state_array_shapes
    from woof.io.restart import classify_state_attr

    off = state_array_shapes(_cfg())
    assert not {"smoke", "smoke0", "smoke_emis", "smoke_col"} & set(off)
    on = state_array_shapes(_cfg(smoke_tracer=True))
    assert on["smoke"] == on["smoke0"] == (10, 8, 40)
    assert on["smoke_emis"] == on["smoke_col"] == (8, 40)
    assert classify_state_attr("smoke") == "serialize"
    assert classify_state_attr("smoke_emis") == "serialize"
    assert classify_state_attr("smoke_col") == "serialize"
    assert classify_state_attr("smoke0") == "rebuild"
    assert classify_state_attr("_smoke_emission_bound") == "infra"


def test_nest_and_boundary_tables():
    from woof.core.nest_fields import nest_field_kinds
    from woof.ingest.lateral_bc import COUPLED_SCALAR_STATE_FIELDS
    from woof.ingest.nest_init import (POSITIVE_DEFINITE_MOMENTS,
                                       RK_TIME_T_SEED_PAIRS)

    assert "smoke" not in nest_field_kinds(_cfg())
    kinds = nest_field_kinds(_cfg(smoke_tracer=True))
    assert kinds[-1] == "smoke"
    assert kinds[:-1] == nest_field_kinds(_cfg())
    assert "smoke" in COUPLED_SCALAR_STATE_FIELDS
    assert ("smoke", "smoke0") in RK_TIME_T_SEED_PAIRS
    assert "smoke" in POSITIVE_DEFINITE_MOMENTS


def test_health_rules_are_non_negative():
    from woof.core.health import rule_for_field

    for name in ("smoke", "smoke_col", "smoke_emis"):
        rule = rule_for_field(name)
        assert rule.status_class == "tracer"
        assert rule.lower == 0.0


def test_history_and_schema_rows():
    from types import SimpleNamespace

    from woof.io.history_layout import live_state_history_fields
    from woof.io.wrf_output_schema import REGISTRY_VAR_META

    assert REGISTRY_VAR_META["SMOKE"][1] == "ug kg-1"
    assert REGISTRY_VAR_META["SMOKE_COL"][1] == "ug m-2"
    state = SimpleNamespace(smoke=np.zeros((2, 3, 4)),
                            smoke_col=np.zeros((3, 4)))
    out = live_state_history_fields(state)
    assert out["SMOKE"] is state.smoke
    assert out["SMOKE_COL"] is state.smoke_col
    assert "SMOKE" not in live_state_history_fields(SimpleNamespace())


def test_history_planner_prices_the_two_outputs():
    from woof.io.history_layout import produced_history_shapes

    off = produced_history_shapes(_cfg())
    on = produced_history_shapes(_cfg(smoke_tracer=True))
    assert set(on) - set(off) == {"SMOKE", "SMOKE_COL"}
    assert on["SMOKE"] == (10, 8, 40)
    assert on["SMOKE_COL"] == (8, 40)


def test_preflight_scratch_slots_and_transients():
    from woof.core.preflight import (scratch_slot_lifetime,
                                     scratch_slot_registry,
                                     smoke_tracer_transient_shapes)

    off = scratch_slot_registry(_cfg())
    on = scratch_slot_registry(_cfg(smoke_tracer=True))
    assert "smag_rsmoke" not in off
    for slot in ("smag_rsmoke", "rk_ru_m", "rk_rv_m", "rk_ww_m",
                 "moist_rq_t", "pd_fxl", "moist_pd_q0"):
        assert slot in on, slot
        assert scratch_slot_lifetime(slot) is not None, slot
    assert smoke_tracer_transient_shapes(_cfg()) == {}
    assert smoke_tracer_transient_shapes(_cfg(smoke_tracer=True))


# ------------------------------------------------------------- emission ---

def test_load_emission_npy_and_nc(tmp_path):
    from netCDF4 import Dataset

    from woof.core.smoke_tracer import load_emission

    field = np.zeros((8, 40))
    field[4, 5] = 2.0e-9
    np.save(tmp_path / "e.npy", field)
    got = load_emission(tmp_path / "e.npy", (8, 40))
    assert got.dtype == np.float32 and got[4, 5] == np.float32(2.0e-9)

    with Dataset(tmp_path / "e.nc", "w") as ds:
        ds.createDimension("south_north", 8)
        ds.createDimension("west_east", 40)
        var = ds.createVariable("smoke_emission", "f4",
                                ("south_north", "west_east"))
        var.units = "ug m-2 s-1"
        var[:] = field * 1.0e9
    got = load_emission(tmp_path / "e.nc", (8, 40))
    np.testing.assert_allclose(got[4, 5], 2.0e-9, rtol=1e-6)


def test_load_emission_refusals(tmp_path):
    from woof.core.smoke_tracer import load_emission

    with pytest.raises(FileNotFoundError):
        load_emission(tmp_path / "missing.npy", (8, 40))
    np.save(tmp_path / "shape.npy", np.zeros((4, 4)))
    with pytest.raises(ValueError, match="mass grid"):
        load_emission(tmp_path / "shape.npy", (8, 40))
    bad = np.zeros((8, 40))
    bad[0, 0] = -1.0
    np.save(tmp_path / "neg.npy", bad)
    with pytest.raises(ValueError, match="non-negative"):
        load_emission(tmp_path / "neg.npy", (8, 40))


# ------------------------------------------------------------------ GPU ---

def _state(cfg, u0=0.0):
    import cupy as cp

    from woof.core.diagnostics import update_diagnostics
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.state import init_at_rest

    vc = make_vertical_coord(cfg.nz)
    base = make_base_state(vc, lambda z: 300.0 + 0.004 * z,
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_at_rest(cfg, vc, base)
    if u0:
        state.u[...] = cp.float32(u0)
    update_diagnostics(state, cfg.hypsometric_opt)
    return state


def _mass(state):
    import cupy as cp

    layer = ((state.c1h[:, None, None] * (state.mub2d + state.mup)[None]
              + state.c2h[:, None, None]) * (-state.dnw[:, None, None]))
    return float(cp.asnumpy((state.smoke.astype(cp.float64)
                             * layer.astype(cp.float64)).sum())) / 9.81


@pytest.mark.gpu
@requires_gpu
def test_off_run_allocates_nothing():
    state = _state(_cfg(km_opt=0))
    for name in ("smoke", "smoke0", "smoke_emis", "smoke_col"):
        assert not hasattr(state, name)


@pytest.mark.gpu
@requires_gpu
def test_puff_advects_downwind_positive_and_conserved():
    import cupy as cp

    from woof.core.dycore import step

    u0 = 10.0
    cfg = _cfg(km_opt=0, smoke_tracer=True, time_step_sound=4)
    state = _state(cfg, u0=u0)
    nz, ny, nx = state.smoke.shape
    x = np.arange(nx)
    puff = np.exp(-0.5 * ((x - 10.0) / 2.0) ** 2)
    blob = np.zeros((nz, ny, nx))
    blob[1:4] = puff[None, None, :] * 100.0
    state.smoke[...] = cp.asarray(blob, dtype=cp.float32)
    mass0 = _mass(state)

    def centroid():
        col = cp.asnumpy(state.smoke[1:4].sum(axis=(0, 1))).astype(np.float64)
        return float((col * x).sum() / col.sum())

    x0 = centroid()
    nsteps = 60
    for _ in range(nsteps):
        step(state, cfg)
    smoke = cp.asnumpy(state.smoke)
    assert np.isfinite(smoke).all()
    assert smoke.min() >= 0.0
    moved = (centroid() - x0) * cfg.dx
    expected = u0 * nsteps * cfg.dt
    assert abs(moved - expected) < 0.1 * expected, (moved, expected)
    assert abs(_mass(state) - mass0) <= 1.0e-4 * mass0
    col = cp.asnumpy(state.smoke_col)
    np.testing.assert_allclose(col.sum(), _mass(state), rtol=1e-4)


@pytest.mark.gpu
@requires_gpu
def test_surface_emission_injects_its_flux(tmp_path):
    import cupy as cp

    from woof.core.dycore import step

    flux = 1.0e-8                                   # kg m-2 s-1
    field = np.zeros((8, 40))
    field[3:5, 18:22] = flux
    np.save(tmp_path / "emis.npy", field)
    cfg = _cfg(km_opt=0, smoke_tracer=True,
               smoke_emission_file=str(tmp_path / "emis.npy"))
    state = _state(cfg)
    nsteps = 12
    for _ in range(nsteps):
        step(state, cfg)
    expected = flux * 1.0e9 * 8 * nsteps * cfg.dt   # ug per m2, 8 cells
    got = _mass(state)
    assert abs(got - expected) <= 1.0e-4 * expected, (got, expected)
    assert float(cp.asnumpy(state.smoke).min()) >= 0.0
    lowest = cp.asnumpy(state.smoke[0])
    assert lowest[3, 18] > 0.0 and lowest[0, 0] == 0.0


@pytest.mark.gpu
@requires_gpu
def test_pbl_mixing_conserves_a_column():
    import cupy as cp

    from woof.core import smoke_tracer

    cfg = _cfg(smoke_tracer=True, nx=6, ny=4)
    state = _state(cfg)
    state.smoke[0] = cp.float32(50.0)
    nz = cfg.nz
    dz = cp.full((nz, 4, 6), 200.0, dtype=cp.float32)
    rho = cp.full((nz, 4, 6), 1.0, dtype=cp.float32)
    exch = cp.full((nz, 4, 6), 50.0, dtype=cp.float32)
    before = float(cp.asnumpy(state.smoke[:, 0, 0]).sum())
    smoke_tracer.mix_pbl(state, {"dz": dz, "rho": rho}, exch, 600.0)
    after = cp.asnumpy(state.smoke[:, 0, 0])
    assert after.min() >= 0.0
    assert after[1] > 0.0 and after[0] < 50.0
    # The diff4d solve conserves the density-weighted load exactly in real
    # arithmetic; FP32 elimination leaves a few ulps per level.
    np.testing.assert_allclose(after.sum(), before, rtol=2e-4)
    np.testing.assert_array_equal(cp.asnumpy(state.smoke0),
                                  cp.asnumpy(state.smoke))


@pytest.mark.gpu
@requires_gpu
def test_specified_inflow_is_zero_and_outflow_zero_gradient():
    import cupy as cp
    from woof.ingest.lateral_bc import apply_flow_dependent_boundaries

    cfg = _cfg(smoke_tracer=True)
    state = _state(cfg, u0=10.0)
    state.smoke[...] = cp.float32(5.0)
    ru = cp.full(state.u.shape, 10.0, dtype=cp.float32)
    rv = cp.zeros(state.v.shape, dtype=cp.float32)
    apply_flow_dependent_boundaries((state.smoke,), ru, rv, 1)
    smoke = cp.asnumpy(state.smoke)
    assert (smoke[:, 2:-2, 0] == 0.0).all()          # west inflow
    assert (smoke[:, 2:-2, -1] == 5.0).all()         # east outflow



# ------------------------------------------------- word-exact parity ---
#
# The standard is raw-word equality (max ULP 0).  The tracer's transport,
# km_opt mixing, sixth-order filter, flow-dependent boundaries and nest
# coupling are the moist-scalar kernels called with the same arguments, so
# they are graded by carrying one field as smoke and as a moist species in
# the same run.  The PBL solve is graded against WRF's own diff4d fixture.
# The two kernels the tracer adds are graded against their NumPy twins.

def _words(array):
    import cupy as cp

    host = cp.asnumpy(array) if hasattr(array, "device") else np.asarray(array)
    return np.ascontiguousarray(host, dtype=np.float32).view(np.uint32)


def _max_ulp(a, b):
    a = _words(a).astype(np.int64)
    b = _words(b).astype(np.int64)
    return int(np.abs(a - b).max()) if a.size else 0


@pytest.mark.gpu
@requires_gpu
def test_surface_source_kernel_matches_numpy_twin_word_for_word():
    import cupy as cp

    from woof.core import smoke_tracer
    from types import SimpleNamespace

    rng = np.random.default_rng(7)
    ny, nx, nz = 24, 33, 6
    emis = (10.0 ** rng.uniform(-12, -4, (ny, nx))).astype(np.float32)
    emis[rng.random((ny, nx)) < 0.3] = 0.0
    emis[0, :4] = np.float32(1.0e-40)               # subnormal flux words
    tend = rng.standard_normal((nz, ny, nx)).astype(np.float32)
    dnw = -np.sort(rng.uniform(0.005, 0.05, nz)).astype(np.float32)
    state = SimpleNamespace(smoke_emis=cp.asarray(emis), dnw=cp.asarray(dnw))
    device = cp.asarray(tend)
    smoke_tracer.add_surface_source(state, device)
    expected = smoke_tracer.surface_source_reference(tend[0], emis, dnw)
    assert _max_ulp(device[0], expected) == 0
    assert _max_ulp(device[1:], tend[1:]) == 0


@pytest.mark.gpu
@requires_gpu
def test_column_load_kernel_matches_numpy_twin_word_for_word():
    import cupy as cp

    from woof.core import smoke_tracer
    from types import SimpleNamespace

    rng = np.random.default_rng(11)
    nz, ny, nx = 49, 17, 23
    smoke = (rng.random((nz, ny, nx)) * 10.0 ** rng.uniform(
        -3, 4, (nz, ny, nx))).astype(np.float32)
    smoke[:, :3, :3] = 0.0
    c1h = rng.uniform(0.0, 1.0, nz).astype(np.float32)
    c2h = rng.uniform(0.0, 500.0, nz).astype(np.float32)
    dnw = -rng.uniform(0.005, 0.05, nz).astype(np.float32)
    mub = rng.uniform(8.0e4, 9.5e4, (ny, nx)).astype(np.float32)
    mup = rng.uniform(-500.0, 500.0, (ny, nx)).astype(np.float32)
    state = SimpleNamespace(
        smoke=cp.asarray(smoke), smoke_col=cp.zeros((ny, nx), cp.float32),
        c1h=cp.asarray(c1h), c2h=cp.asarray(c2h), dnw=cp.asarray(dnw),
        mub2d=cp.asarray(mub), mup=cp.asarray(mup))
    smoke_tracer.refresh_column_load(state)
    expected = smoke_tracer.column_load_reference(smoke, c1h, c2h, dnw,
                                                  mub, mup)
    assert _max_ulp(state.smoke_col, expected) == 0


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("case", range(8))
def test_pbl_mixing_matches_wrf_diff4d_fixture_word_for_word(case):
    """mix_pbl reproduces WRF v4.6.1 diff4d (the scalar_pblmix fixture).

    tests/data/scalar_pblmix_wrf461.npz is WRF's unmodified diff4d/diff/
    invert (receipt beside it); its four active columns per case run as a
    (50, 1, 4) tracer, and the rate the solve implies -- the kernel's own
    ``(solved - qn)/dt`` in FP32 -- must equal WRF's tendency words.
    """
    import cupy as cp
    from types import SimpleNamespace

    from woof.core import smoke_tracer

    path = __import__("pathlib").Path(__file__).parent / "data" / \
        "scalar_pblmix_wrf461.npz"
    with np.load(path) as data:
        cols = {key: data[key][case, :4] for key in
                ("qn", "dz", "rho", "exch_h", "tendency")}
        dt = np.float32(data["dt"][case, 0, 0])

    def volume(a):
        return cp.asarray(np.ascontiguousarray(a.T).reshape(50, 1, 4))

    state = SimpleNamespace(smoke=volume(cols["qn"]),
                            smoke0=cp.zeros((50, 1, 4), cp.float32))
    smoke_tracer.mix_pbl(state, {"dz": volume(cols["dz"]),
                                 "rho": volume(cols["rho"])},
                         volume(cols["exch_h"]), float(dt))
    solved = cp.asnumpy(state.smoke).reshape(50, 4).T
    rate = (solved - cols["qn"]) / dt
    assert _max_ulp(rate, cols["tendency"]) == 0
    assert _max_ulp(state.smoke0, state.smoke) == 0


def _twin_run(**kw):
    """Carry one field as ``qc`` and as ``smoke`` through the same steps."""
    import cupy as cp

    from woof.core.dycore import step

    cfg = _cfg(moist=True, mp_physics=0, moist_adv_opt=1, smoke_tracer=True,
               **kw)
    state = _state(cfg, u0=8.0)
    nz, ny, nx = state.smoke.shape
    rng = np.random.default_rng(3)
    blob = np.zeros((nz, ny, nx), dtype=np.float32)
    blob[1:5, 2:6, 8:16] = 1.0e-4                    # a sharp-edged puff
    blob[2, 3, 9] = 1.0e-12                          # a near-zero cell
    blob += (rng.random(blob.shape) < 0.05) * np.float32(3.0e-7)
    state.qc[...] = cp.asarray(blob)
    state.smoke[...] = cp.asarray(blob)
    # A shear and a vertical motion so every flux direction is live.
    state.v[...] = cp.asarray(
        rng.standard_normal(state.v.shape).astype(np.float32))
    for _ in range(8):
        step(state, cfg)
    return state, blob


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("kw", [
    {"km_opt": 4, "diff_6th_opt": 2, "diff_6th_factor": 0.12},
    {"km_opt": 0, "diff_6th_opt": 2, "diff_6th_factor": 0.12},
], ids=["km4-diff6", "diff6-only"])
def test_smoke_transport_is_the_moist_scalar_transport_word_for_word(kw):
    """Same field, same operators: smoke and qc carry identical words.

    This is the reuse proof for transport (the PD final stage, the
    unlimited stages, IEVA off), km_opt mixing and the sixth-order
    filter: the moist path is graded against WRF v4.7.1 by
    tests/test_advect_wrf471_parity.py and its siblings, and the tracer
    calls the same kernels with the same arguments.
    """
    state, blob = _twin_run(**kw)
    smoke = _words(state.smoke)
    assert (smoke != blob.view(np.uint32)).sum() > smoke.size // 2
    assert _max_ulp(state.smoke, state.qc) == 0, int(
        (smoke != _words(state.qc)).sum())
    assert smoke.size == 10 * 8 * 40


@pytest.mark.gpu
@requires_gpu
def test_flow_dependent_boundary_is_the_moist_kernel_word_for_word():
    """The specified-boundary flow_dep_bdy smoke takes is qc's, word for word."""
    import cupy as cp
    from woof.ingest.lateral_bc import apply_flow_dependent_boundaries

    rng = np.random.default_rng(5)
    shape = (10, 8, 40)
    field = rng.random(shape).astype(np.float32)
    ru = cp.asarray(rng.standard_normal((10, 8, 41)).astype(np.float32))
    rv = cp.asarray(rng.standard_normal((10, 9, 40)).astype(np.float32))
    smoke = cp.asarray(field)
    qc, qr = cp.asarray(field), cp.asarray(field)
    apply_flow_dependent_boundaries((smoke,), ru, rv, 1)
    apply_flow_dependent_boundaries((qc, qr), ru, rv, 1)
    assert _max_ulp(smoke, qc) == 0
    assert _max_ulp(smoke, qr) == 0
    assert (cp.asnumpy(smoke) != field).any()


@pytest.mark.gpu
@requires_gpu
def test_nest_coupling_is_the_moist_scalar_coupling_word_for_word():
    """A nest edge couples smoke with qc's half-level mu weighting."""
    import cupy as cp
    from woof.ingest.lateral_bc import couple_nest_field

    cfg = _cfg(moist=True, smoke_tracer=True, km_opt=4)
    state = _state(cfg)
    rng = np.random.default_rng(9)
    field = rng.random(state.smoke.shape).astype(np.float32)
    state.smoke[...] = cp.asarray(field)
    state.qc[...] = cp.asarray(field)
    state.mup[...] = cp.asarray(
        rng.uniform(-300, 300, state.mup.shape).astype(np.float32))
    a = cp.empty_like(state.smoke)
    b = cp.empty_like(state.qc)
    couple_nest_field(state, "smoke", out=a)
    couple_nest_field(state, "qc", out=b)
    assert _max_ulp(a, b) == 0


@pytest.mark.gpu
@requires_gpu
def test_tracer_adds_no_device_to_host_copy_per_step(tmp_path, monkeypatch):
    """Every per-step tracer operation stays on the device.

    Counts the host reads a step makes (CuPy's ``get``/``item``/scalar
    conversions and ``cupy.asnumpy``) with the tracer, its emission and a
    mixing package on, against the same step with it off: the two counts
    must match.  The emission file is read and uploaded once, before the
    counted steps.
    """
    import cupy as cp

    from woof.core.dycore import step

    np.save(tmp_path / "emis.npy", np.full((8, 40), 1.0e-9))
    counts = {"n": 0}

    def counting(original):
        def wrapper(*args, **kwargs):
            counts["n"] += 1
            return original(*args, **kwargs)
        return wrapper

    for name in ("get", "item", "__float__", "__int__", "__bool__",
                 "tolist"):
        monkeypatch.setattr(cp.ndarray, name,
                            counting(getattr(cp.ndarray, name)))
    monkeypatch.setattr(cp, "asnumpy", counting(cp.asnumpy))

    def per_step(**kw):
        cfg = _cfg(km_opt=4, diff_6th_opt=2, diff_6th_factor=0.12, **kw)
        state = _state(cfg, u0=5.0)
        step(state, cfg)                       # binds the emission once
        counts["n"] = 0
        for _ in range(3):
            step(state, cfg)
        return counts["n"]

    off = per_step()
    on = per_step(smoke_tracer=True,
                  smoke_emission_file=str(tmp_path / "emis.npy"))
    assert on == off, (on, off)


def test_sase_is_refused_by_name():
    with pytest.raises(ValueError, match="SASE"):
        validate_run_config(_cfg(smoke_tracer=True, bl_pbl_physics=900))


def test_masked_netcdf_cells_are_refused(tmp_path):
    from netCDF4 import Dataset

    from woof.core.smoke_tracer import load_emission

    with Dataset(tmp_path / "m.nc", "w") as ds:
        ds.createDimension("south_north", 8)
        ds.createDimension("west_east", 40)
        var = ds.createVariable("smoke_emission", "f4",
                                ("south_north", "west_east"),
                                fill_value=np.float32(9.96921e36))
        data = np.zeros((8, 40), dtype=np.float32)
        var[:] = np.ma.masked_array(data, mask=(np.arange(320) == 5)
                                    .reshape(8, 40))
    with pytest.raises(ValueError, match="_FillValue"):
        load_emission(tmp_path / "m.nc", (8, 40))


def test_packed_netcdf_is_unpacked(tmp_path):
    from netCDF4 import Dataset

    from woof.core.smoke_tracer import load_emission

    with Dataset(tmp_path / "p.nc", "w") as ds:
        ds.createDimension("south_north", 8)
        ds.createDimension("west_east", 40)
        var = ds.createVariable("smoke_emission", "i2",
                                ("south_north", "west_east"))
        var.scale_factor = 1.0e-10
        var[:] = np.full((8, 40), 3.0e-9)
    got = load_emission(tmp_path / "p.nc", (8, 40))
    np.testing.assert_allclose(got, 3.0e-9, rtol=1e-6)


def test_emission_digest_names_the_bytes(tmp_path):
    from woof.core.smoke_tracer import emission_file_digest

    path = tmp_path / "e.npy"
    np.save(path, np.zeros((8, 40)))
    first = emission_file_digest(path)
    np.save(path, np.ones((8, 40)))
    assert emission_file_digest(path) != first
    assert emission_file_digest(tmp_path / "absent.npy") == "missing"
