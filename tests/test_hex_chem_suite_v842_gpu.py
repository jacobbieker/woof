"""wrf_chem and micm on the MPAS v8.4.2 suite route, word for word the hex WOOF route's column runtime.

woof.hex.mpasphys.chemistry_v842.SuiteChemistry runs the WOOF route's own column runtimes
(woof.hex.chem_wrf.HexWrfChemistry, woof.hex.chem_micm_hex.HexMicmChemistry) on the rows appended after the
MPAS Registry scalars.  Its new part is the prep adapter: the WRF met names the processes read are the MPAS
physics pools' words (diag_physics.kzh as exch_h, hpbl as pblh, sfc_input.skintemp as tsk, ...), the column
config is built from the suite's scheme selections, and the clock, restart and receipt follow the suite run.

These tests drive the WRF PhysicsDriver column seam (woof.core.mpas_column_batch, as
tests/test_hex_chem_wrf_gpu.py does), publish its fields into a stand-in MPAS session under the MPAS pool
names and layout, and step the suite route and the WOOF-route runtime on the same columns: every row, every
process array and the ledger must match byte for byte at every step, the Registry rows must stay untouched,
and a restart through the suite's checkpoint payload must continue bit-identically.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import test_hex_chem_wrf_gpu as base  # noqa: E402
from woof.hex import chem_micm_hex, chem_wrf  # noqa: E402
from woof.hex.mpasphys import chemistry_v842 as suite  # noqa: E402

NZ, NCOL, DT, STEPS = base.NZ, base.NCOL, base.DT, 5
START = base.START
REGISTRY = ("qv", "qc", "qr")
LAT = np.radians(np.linspace(20.0, 34.0, NCOL))
LON = np.radians(np.linspace(-10.0, 5.0, NCOL))
#: the suite's selections matching the column seam's (WSM6, YSU, Noah-MP, no cumulus)
CONFIGS = {"config_physics_suite": "mesoscale_reference", "config_microp_scheme": "mp_wsm6",
           "config_pbl_scheme": "bl_ysu", "config_lsm_scheme": "sf_noahmp", "config_convection_scheme": "off",
           "config_len_disp": 15000.0, "config_start_time": START.strftime("%Y-%m-%d_%H:%M:%S")}
MECHANISM = Path(__file__).resolve().parent / "data" / "micm_oracle" / "chempas_a_chapman_nox.yaml"


class _Session:
    """The suite session's ``field(pool, name)``: MPAS layout, ``(nCells,)`` and ``(n, nCells)``."""

    def __init__(self):
        self.pools = {}

    def field(self, pool, name):
        return self.pools[pool, name]

    def publish(self, fields, *, albedo):
        """The column driver's WRF fields under their MPAS pool names (SUITE_MET_FIELDS)."""
        self.pools.clear()
        for wrf_name, pool, name in suite.SUITE_MET_FIELDS:
            value = fields.get(wrf_name)
            if value is not None:
                self.pools[pool, name] = cp.asarray(value).reshape(-1, NCOL).squeeze(0) \
                    if value.ndim == 2 else cp.asarray(value).reshape(value.shape[0], NCOL)
        self.pools["diag_physics", "sfc_albedo"] = albedo


def _prepared(columns):
    """The MPAS-to-physics preparation's carriers the column runtimes read."""
    t = columns["theta"] * (columns["pressure"] / np.float32(1.0e5)) ** np.float32(287.0 / 1004.0)
    return SimpleNamespace(pres_p=columns["pressure"], pres_hyd_p=columns["pressure"], th_p=columns["theta"],
                           t_p=cp.ascontiguousarray(t, dtype=cp.float32), rho_dry=columns["rho_dry"],
                           z_p=columns["z_interface"], qv_p=columns["qv"], qc_p=columns["qc"],
                           qr_p=columns["qr"], u_p=columns["u"], v_p=columns["v"], w_p=columns["w"])


def _suite(tmp_path, raw, area, *, restart=None, water=False):
    """A SuiteChemistry on a stand-in mesh, dycore and coupling; the preparation is the step's columns."""
    path = tmp_path / "chemistry.json"
    path.write_text(json.dumps(raw))
    mesh = SimpleNamespace(n_cells=NCOL, attrs={},
                           arrays={"areaCell": np.asarray(area, np.float64), "latCell": LAT, "lonCell": LON})
    zw = np.linspace(0.0, 16000.0, NZ + 1)
    vertical = SimpleNamespace(zw=zw, zgrid=np.repeat(zw[None, :], NCOL, axis=0))
    chem = suite.SuiteChemistry(path, registry_names=REGISTRY, mesh=mesh, nlev=NZ, grid_path=path,
                                start_text=CONFIGS["config_start_time"], vertical=vertical,
                                configs=dict(CONFIGS), dt=DT)
    rng = np.random.default_rng(7)
    host = SimpleNamespace(scalars=rng.uniform(1e-6, 1e-3, (len(REGISTRY), NZ, NCOL)).astype(np.float32))
    if restart is not None:
        chem.restart = restart["payload"]
        host.scalars = restart["scalars"].copy()
    chem.extend_state(host)
    dycore = SimpleNamespace(atmosphere=SimpleNamespace(state=SimpleNamespace(scalars=cp.asarray(host.scalars))))
    session = _Session()
    session.pools["sfc_input", "xland"] = cp.asarray(np.where(base.WATER & water, 2.0, 1.0), dtype=cp.float32)
    holder = SimpleNamespace(columns=None)
    chem._columns = lambda atmosphere: _prepared(holder.columns)
    chem.bind(dycore=dycore, cache=None, coupling=SimpleNamespace(session=session, names=REGISTRY))
    return chem, dycore, session, holder


@pytest.fixture(params=[case for case in sorted(base.CASES) if case != "smoke"])
def wrf_case(request, tmp_path, monkeypatch):
    """STEPS seam steps; after each, the suite route's step and the WOOF route's on the same columns."""
    case = request.param
    original = chem_wrf.attach_statics
    monkeypatch.setattr(chem_wrf, "attach_statics",
                        lambda runtime, **kw: original(runtime, lat_deg=None, lon_deg=None, fields=base._statics()))
    seam = base._seam(base.CASES[case][1])
    inputs = base._profile()
    state2 = {n: inputs[n].copy() for n in
              ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "pressure", "rho_dry", "z_interface")}
    area = np.full(NCOL, 2.25e8)
    raw = base._raw(case)
    chem, dycore, session, holder = _suite(tmp_path, raw, area, water=base.CASES[case][1])
    # The WOOF route's runtime (chem_wrf.build_column_backend's), on the seam's own config and geometry.
    ref = chem_wrf.HexWrfChemistry("chemistry.json", raw, chem.digest, cfg=seam._cfg, area=area, nlev=NZ,
                                   ncells=NCOL, offset=0, start_time=START, fnm=seam._state.fnm,
                                   fnp=seam._state.fnp, terrain=seam._state.ht)
    original(ref, lat_deg=None, lon_deg=None, fields=base._statics())
    ref.set_geography(np.degrees(LAT), np.degrees(LON))
    ref_scalars = cp.zeros((len(ref.rows), NZ, NCOL), dtype=cp.float32)
    registry = dycore.atmosphere.state.scalars[:len(REGISTRY)].copy()
    albedo = cp.asarray(np.linspace(0.08, 0.3, NCOL), dtype=cp.float32)
    rng = np.random.default_rng(13)
    history = []
    for step in range(1, STEPS + 1):
        seam.run_phase1(dt=DT, **inputs)
        seam.run_phase2(**state2)
        driver = seam._driver
        columns = base._columns(inputs, state2)
        holder.columns = columns
        # The column seam runs no shortwave it publishes as gsw: the step's net surface shortwave is synthetic,
        # and WRF's SWDOWN is gsw / (1 - albedo) on both routes.
        fields = dict(driver.fields)
        fields["gsw"] = cp.asarray(rng.uniform(0.0, 800.0, (1, NCOL)), dtype=cp.float32)
        fields["swdown"] = fields["gsw"] / (np.float32(1.0) - albedo.reshape(1, NCOL))
        session.publish(fields, albedo=albedo)
        if driver.microphysics is not None:
            session.pools["diag_physics", "rainncv"] = driver.microphysics.rainncv.reshape(NCOL)
        ref.step(ref_scalars, ref.columns(_prepared(columns), "post_rk"), dt=DT, elapsed_seconds=step * DT,
                 physics_fields=fields, microphysics=driver.microphysics)
        ref.commit()
        chem.step(session=session, dt=DT)
        history.append((cp.asnumpy(dycore.atmosphere.state.scalars).copy(), cp.asnumpy(ref_scalars).copy(),
                        {a: (cp.asnumpy(getattr(chem.runtime.state, a)), cp.asnumpy(getattr(ref.state, a)))
                         for a in ref.serialized}))
    return SimpleNamespace(case=case, chem=chem, ref=ref, history=history, registry=registry, dycore=dycore,
                           session=session, seam=seam, fields=fields, tmp_path=tmp_path)


def test_the_suite_fields_are_the_column_drivers_words(wrf_case):
    """Every WRF name the suite maps comes back from the MPAS pools as the driver's bytes and layout."""
    got = suite.suite_physics_fields(wrf_case.session, ncells=NCOL)
    want = wrf_case.fields
    mapped = [w for w, _pool, _name in suite.SUITE_MET_FIELDS if w in want]
    assert {"exch_h", "ust", "rmol", "znt", "pblh", "u10", "v10", "gsw", "tsk", "xland", "smois", "isltyp",
            "ivgtyp", "vegfra", "snowh", "snow"} <= set(mapped)
    for name in (*mapped, "swdown"):
        expected = cp.asarray(want[name])
        expected = expected.astype(cp.int32 if expected.dtype.kind in "iu" else cp.float32)
        assert got[name].shape == expected.shape and got[name].dtype == expected.dtype, name
        assert cp.asnumpy(got[name]).tobytes() == cp.asnumpy(expected).tobytes(), name


def test_the_column_config_takes_the_suites_selections(wrf_case):
    cfg, ref = wrf_case.chem.runtime.cfg, wrf_case.ref.cfg
    for key in ("nx", "ny", "nz", "dx", "dy", "dt", "mp_physics", "bl_pbl_physics", "sf_surface_physics",
                "chem_sets", "dust_opt", "seas_opt", "wetscav_onoff", "vertmix_onoff", "kemit"):
        assert getattr(cfg, key) == getattr(ref, key), key
    for name in ("fnm", "fnp", "ht"):
        got, want = (cp.asnumpy(getattr(r.state, name)) for r in (wrf_case.chem.runtime, wrf_case.ref))
        assert got.tobytes() == want.tobytes(), name


def test_rows_and_process_arrays_are_the_woof_routes_every_step(wrf_case):
    nreg = len(REGISTRY)
    for step, (scalars, ref_rows, arrays) in enumerate(wrf_case.history, 1):
        assert scalars[nreg:].tobytes() == ref_rows.tobytes(), step
        for attr, (got, want) in arrays.items():
            assert got.tobytes() == want.tobytes(), (step, attr)
    last = wrf_case.history[-1][0][nreg:]
    assert float(last.sum()) > 0.0 and (last[:, 1:] > 0).any(), "nothing was emitted and mixed"


def test_the_registry_rows_are_untouched(wrf_case):
    scalars = wrf_case.dycore.atmosphere.state.scalars[:len(REGISTRY)]
    assert cp.asnumpy(scalars).tobytes() == cp.asnumpy(wrf_case.registry).tobytes()


def test_history_receipt_and_ledger(wrf_case):
    chem, ref = wrf_case.chem, wrf_case.ref
    out = chem.history_fields()
    for row in ref.rows:
        assert out[row.name][1] == ("nVertLevels", "nCells") and out[row.name][2] == row.units
        assert out[row.name + "_column_mass"][2] == "kg m-2"
    assert json.loads(chem.history_attribute())["mode"] == "wrf_chem"
    receipt = chem.write_ledger(wrf_case.tmp_path)
    assert receipt["committed_steps"] == STEPS and receipt["route"].startswith("mpas-suite")
    assert receipt["ledger"] == ref.ledger()
    assert receipt["ledger"]["max_relative_closure"] < 1e-12
    assert len(receipt["mass_closure_kg"]) == len(ref.rows)
    assert json.loads((wrf_case.tmp_path / "chemistry-receipt.json").read_text())["mode"] == "wrf_chem"


def test_a_restart_continues_bit_identically(tmp_path, monkeypatch):
    """Two steps, the suite checkpoint payload into a fresh SuiteChemistry whose config_start_time is the
    restart time, two more: the same bytes as four straight."""
    original = chem_wrf.attach_statics
    monkeypatch.setattr(chem_wrf, "attach_statics",
                        lambda runtime, **kw: original(runtime, lat_deg=None, lon_deg=None, fields=base._statics()))
    raw, area = base._raw("dust"), np.full(NCOL, 2.25e8)
    albedo = cp.full(NCOL, 0.2, dtype=cp.float32)

    def drive(chem, session, holder, seam, inputs, state2, steps):
        for _ in range(steps):
            seam.run_phase1(dt=DT, **inputs)
            seam.run_phase2(**state2)
            holder.columns = base._columns(inputs, state2)
            session.publish(seam._driver.fields, albedo=albedo)
            if seam._driver.microphysics is not None:
                session.pools["diag_physics", "rainncv"] = seam._driver.microphysics.rainncv.reshape(NCOL)
            chem.step(session=session, dt=DT)

    def fresh():
        inputs = base._profile()
        return base._seam(), inputs, {n: inputs[n].copy() for n in
                                      ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "pressure", "rho_dry",
                                       "z_interface")}

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    straight, dycore, session, holder = _suite(tmp_path / "a", raw, area)
    drive(straight, session, holder, *fresh(), 4)

    first, dycore2, session2, holder2 = _suite(tmp_path / "b", raw, area)
    seam, inputs, state2 = fresh()
    drive(first, session2, holder2, seam, inputs, state2, 2)
    payload = json.loads(json.dumps(first.restart_state(), default=lambda a: a.tolist()))
    payload["arrays"] = {k: np.asarray(v, dtype=getattr(first.runtime.state, k).dtype)
                         for k, v in payload["arrays"].items()}
    restart = {"payload": payload, "scalars": cp.asnumpy(dycore2.atmosphere.state.scalars)}
    saved = dict(CONFIGS)
    CONFIGS["config_start_time"] = (START + datetime.timedelta(seconds=2 * DT)).strftime("%Y-%m-%d_%H:%M:%S")
    try:
        resumed, dycore3, session3, holder3 = _suite(tmp_path / "b", raw, area, restart=restart)
    finally:
        CONFIGS.update(saved)
    assert resumed.runtime.state.start_time == START
    drive(resumed, session3, holder3, seam, inputs, state2, 2)
    assert (cp.asnumpy(dycore3.atmosphere.state.scalars).tobytes()
            == cp.asnumpy(dycore.atmosphere.state.scalars).tobytes())
    for attr in straight.runtime.serialized:
        assert (cp.asnumpy(getattr(resumed.runtime.state, attr)).tobytes()
                == cp.asnumpy(getattr(straight.runtime.state, attr)).tobytes()), attr


def test_micm_is_the_woof_routes_per_point_solve(tmp_path):
    photolysis = {"jO2": {"j_max": 1e-11}, "jO3_O": {"j_max": 4e-4}, "jO3_O1D": {"j_max": 3e-5},
                  "jNO2": {"j_max": 8e-3}}
    raw = {"schema": "gpuwm-hex.chemistry/v1",
           "micm": {"micm_file": str(MECHANISM), "photolysis": photolysis,
                    "initial_mass_mixing_ratio": {"O2": 0.231, "O3": 1e-6, "NO": 2e-9, "NO2": 5e-9}}}
    area = np.full(NCOL, 2.25e8)
    chem, dycore, session, holder = _suite(tmp_path, raw, area)
    ref = chem_micm_hex.HexMicmChemistry(tmp_path / "chemistry.json", raw, chem.digest, nlev=NZ, ncells=NCOL,
                                         offset=0, start_time=START, dt=DT, lat_deg=np.degrees(LAT),
                                         lon_deg=np.degrees(LON))
    ref_scalars = cp.zeros((len(ref.rows), NZ, NCOL), cp.float32)
    chem_micm_hex.seed(ref, ref_scalars)
    inputs = base._profile()
    columns = base._columns(inputs, inputs)
    holder.columns = columns
    nreg = len(REGISTRY)
    for step in range(1, 4):
        ref.step(ref_scalars, ref.columns(_prepared(columns), "post_rk"), dt=DT, elapsed_seconds=step * DT)
        ref.commit()
        chem.step(session=session, dt=DT)
        got = cp.asnumpy(dycore.atmosphere.state.scalars[nreg:])
        assert got.tobytes() == cp.asnumpy(ref_scalars).tobytes(), step
    assert chem.restart_state()["case_start"] == CONFIGS["config_start_time"]
    payload = chem.restart_state()
    assert payload["mechanism_sha256"] == ref.settings["mechanism_sha256"]
    restart = {"payload": dict(payload, mechanism_sha256="0" * 64),
               "scalars": cp.asnumpy(dycore.atmosphere.state.scalars)}
    with pytest.raises(ValueError, match="mechanism"):
        _suite(tmp_path, raw, area, restart=restart)
    receipt = chem.write_ledger(tmp_path)
    assert receipt["mode"] == "micm" and receipt["committed_steps"] == 3 and "mass_closure_kg" not in receipt


@pytest.mark.parametrize("block, message", [
    ({"chem_opt": 401, "aer_ra_feedback": 1}, "aer_ra_feedback"),
    ({"chem_sets": "gocart_primary", "aerosol_mp_coupling": "diagnose"}, "aerosol_mp_coupling"),
])
def test_couplings_the_suite_does_not_share_are_refused(tmp_path, block, message):
    path = tmp_path / "chemistry.json"
    path.write_text(json.dumps({"schema": "gpuwm-hex.chemistry/v1", "wrf_chem": block}))
    mesh = SimpleNamespace(n_cells=NCOL, attrs={}, arrays={"areaCell": np.full(NCOL, 2.25e8)})
    with pytest.raises(ValueError, match=message):
        suite.SuiteChemistry(path, registry_names=REGISTRY, mesh=mesh, nlev=NZ, grid_path=path,
                             start_text=CONFIGS["config_start_time"], vertical=SimpleNamespace(),
                             configs=CONFIGS, dt=DT)


def test_suite_selections_resolve_as_mpas_does():
    assert suite.suite_selections({"config_physics_suite": "convection_permitting"}) == {
        "config_microp_scheme": "mp_thompson", "config_convection_scheme": "cu_grell_freitas",
        "config_pbl_scheme": "bl_mynn", "config_lsm_scheme": "sf_noah"}
    cfg = suite.suite_column_config(ncells=4, nlev=NZ, dt=DT, dx_m=3000.0, ztop=16000.0,
                                    configs={"config_physics_suite": "mesoscale_reference"})
    assert (cfg.mp_physics, cfg.cu_physics, cfg.bl_pbl_physics, cfg.sf_surface_physics) == (6, 16, 1, 2)
    with pytest.raises(ValueError, match="config_pbl_scheme"):
        suite.suite_column_config(ncells=4, nlev=NZ, dt=DT, dx_m=3000.0, ztop=16000.0,
                                  configs={"config_pbl_scheme": "bl_unknown"})
