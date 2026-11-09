"""The coupled WRF/WRF-Hydro forecast route (woof.wrf_hydro_route and the opt-in's
woof_wrf_hydro.coupled).

The door (``--wrf-hydro``, ``[wrf_hydro]``) and its refusals run in the
default build; with no hydro configuration nothing is attached and the LSM
calls are the default ones.  The coupled pieces need the opt-in
(``WOOF_ENABLE_WRF_HYDRO=1``) and a GPU: HYDRO_ini's network on a synthetic
Fulldom, the water budget across each coupling cycle, the output files, and
a short forecast through the real driver with Noah.  The coupling cycles are
graded against WRF-Hydro's own HYDRO_ini and HYDRO_exe compiled whole in
tests/test_wrf_hydro_coupled_parity.py.
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from woof import optin

import wrf_hydro_synthetic as synthetic

REASON = optin.unavailable_reason("wrf_hydro")
needs_optin = pytest.mark.skipif(REASON is not None, reason=f"opt-in rule: {REASON}")


@pytest.fixture
def no_hydro_env(monkeypatch):
    from woof import wrf_hydro_route as route

    monkeypatch.delenv(route.ENV_NAMELIST, raising=False)
    monkeypatch.delenv(route.ENV_OUTPUT, raising=False)


# -- the door, default build ------------------------------------------------
def test_without_a_hydro_configuration_nothing_is_attached(no_hydro_env):
    from woof.wrf_hydro_route import attach

    driver = SimpleNamespace(fields={})
    assert attach(driver, None, SimpleNamespace(sf_surface_physics=2)) is None
    assert vars(driver) == {"fields": {}}


def test_the_default_lsm_call_takes_no_hydro_keyword(no_hydro_env):
    from woof.core.physics import _wrf_hydro_lsm_kwargs

    assert _wrf_hydro_lsm_kwargs(SimpleNamespace()) == {}


def test_launch_scope_sets_and_restores_the_environment(no_hydro_env, tmp_path):
    import os

    from woof import wrf_hydro_route as route

    toml = tmp_path / "case.toml"
    toml.write_text('[wrf_hydro]\nnamelist = "nl/hydro.namelist"\noutput_dir = "out"\n')
    with route.launch_scope(config=toml):
        assert route.configured() == str((tmp_path / "nl/hydro.namelist").resolve())
        assert os.environ[route.ENV_OUTPUT] == str((tmp_path / "out").resolve())
    assert route.configured() is None
    with route.launch_scope(namelist=tmp_path / "cli.namelist", config=toml):
        assert route.configured() == str((tmp_path / "cli.namelist").resolve())
    assert route.configured() is None


@pytest.mark.parametrize("body,match", [
    ('[wrf_hydro]\nnamelist = "h"\nspeed = 2\n', "unknown"),
    ('[wrf_hydro]\noutput_dir = "o"\n', "namelist"),
])
def test_the_table_is_checked(tmp_path, body, match):
    from woof.wrf_hydro_route import read_table

    p = tmp_path / "c.toml"
    p.write_text(body)
    with pytest.raises(ValueError, match=match):
        read_table(p)


def test_an_ensemble_launch_is_refused(no_hydro_env, tmp_path):
    from woof import wrf_hydro_route as route

    with pytest.raises(NotImplementedError, match="ensemble members"):
        with route.launch_scope(namelist=tmp_path / "hydro.namelist", members=3):
            pass
    assert route.configured() is None
    with route.launch_scope(namelist=tmp_path / "hydro.namelist", members=1):
        assert route.configured()


def test_a_loader_without_the_launch_refuses_the_table(no_hydro_env):
    from woof.wrf_hydro_route import consume_table

    raw = {"grid": {}, "wrf_hydro": {"namelist": "hydro.namelist"}}
    with pytest.raises(ValueError, match="uncoupled"):
        consume_table(raw, "case.toml")
    assert consume_table({"grid": {}}, "case.toml") == {"grid": {}}


def test_a_launch_drops_the_table_for_the_run_config(monkeypatch):
    from woof import wrf_hydro_route as route

    monkeypatch.setenv(route.ENV_NAMELIST, "/x/hydro.namelist")
    raw = {"grid": {}, "wrf_hydro": {"namelist": "hydro.namelist"}}
    assert route.consume_table(raw, "case.toml") == {"grid": {}}


def test_more_than_one_domain_is_refused(monkeypatch):
    from woof import wrf_hydro_route as route

    route.refuse_multiple_domains(3)               # off: nothing to refuse
    monkeypatch.setenv(route.ENV_NAMELIST, "/x/hydro.namelist")
    route.refuse_multiple_domains(1)
    with pytest.raises(NotImplementedError, match="num_nests < 1"):
        route.refuse_multiple_domains(2)


@pytest.mark.parametrize("cfg,match", [
    (dict(sf_surface_physics=3), "Noah"),
    (dict(sf_surface_physics=2, grid_id=2), "nest"),
    (dict(sf_surface_physics=2, restart_interval_s=3600.0), "restart_interval_s = 0"),
    (dict(sf_surface_physics=2, sf_urban_physics=1), "urban"),
    (dict(sf_surface_physics=2, sf_surface_mosaic=1), "mosaic"),
    (dict(sf_surface_physics=4, use_adaptive_time_step=True), "adaptive"),
])
def test_attach_refuses_by_name(monkeypatch, cfg, match):
    from woof import wrf_hydro_route as route

    monkeypatch.setenv(route.ENV_NAMELIST, "/x/hydro.namelist")
    with pytest.raises(NotImplementedError, match=match):
        route.attach(SimpleNamespace(fields={}), None, SimpleNamespace(**cfg),
                     start_time=datetime.datetime(2020, 1, 1), dzs=[0.1, 0.3, 0.6, 1.0])


def test_nextgen_and_wrf_hydro_are_not_combined(monkeypatch):
    from woof import wrf_hydro_route as route

    monkeypatch.setenv(route.ENV_NAMELIST, "/x/hydro.namelist")
    with pytest.raises(NotImplementedError, match="NextGen"):
        route.attach(SimpleNamespace(fields={}, hydro=object()), None,
                     SimpleNamespace(sf_surface_physics=4), start_time=None, dzs=None)


def test_the_routing_is_reached_only_through_its_opt_in(monkeypatch):
    from woof import wrf_hydro_route as route

    monkeypatch.setenv(route.ENV_NAMELIST, "/x/hydro.namelist")
    monkeypatch.delenv("WOOF_ENABLE_WRF_HYDRO", raising=False)
    with pytest.raises(optin.OptInUnavailable, match="WRF-Hydro"):
        route.attach(SimpleNamespace(fields={}), None, SimpleNamespace(sf_surface_physics=2),
                     start_time=datetime.datetime(2020, 1, 1), dzs=[0.1, 0.3, 0.6, 1.0])


def test_a_checkpoint_of_a_coupled_run_is_refused():
    from woof.io.restart import RestartManifestError, _driver_manifest

    with pytest.raises(RestartManifestError, match="WRF-Hydro"):
        _driver_manifest(SimpleNamespace(wrf_hydro=object()))


def test_wrf_zs_is_real_exe_s():
    from woof.wrf_hydro_route import soil_centres

    zs = soil_centres([0.1, 0.3, 0.6, 1.0])
    f = np.float32
    assert zs.dtype == f
    assert zs[0] == f(f(0.5) * f(0.1))
    assert zs[3] == f(f(zs[2] + f(f(0.5) * f(0.6))) + f(f(0.5) * f(1.0)))
    assert np.allclose(zs, [0.05, 0.25, 0.7, 1.5])


# -- the opt-in's HYDRO_ini and HYDRO_exe -----------------------------------
def _coupled(tmp_path, nx=8, ny=6, **nl):
    from importlib import import_module

    from woof.wrf_hydro_route import soil_centres

    mod = optin.load("wrf_hydro")
    coupled = import_module(mod.__name__ + ".coupled")
    path = synthetic.write(tmp_path / "dom", nx=nx, ny=ny, **nl)
    veg = np.full((ny, nx), 10, np.int32)
    veg[:, -1] = 17
    sol = np.full((ny, nx), 6, np.int32)
    sol[:, -1] = 14
    return coupled, coupled.WrfHydroCoupled(
        path, ix=nx, jx=ny, zs=soil_centres([0.1, 0.3, 0.6, 1.0]), vegtyp=veg, soltyp=sol,
        hydro_dt=60.0, start=datetime.datetime(2020, 1, 1), output_dir=tmp_path / "out")


@needs_optin
@pytest.mark.parametrize("edit,match", [
    (dict(channel_option=2), "channel_option"),
    (dict(sub=0, ov=0, gw=0), "rtFlag"),
    (dict(extra="imperv_adj = 1\n"), "IMPERV_ADJ"),
    (dict(extra="GWBASESWCRT = 3\n"), "GWBASESWCRT"),
])
def test_coupled_refusals_name_the_option(tmp_path, edit, match):
    from importlib import import_module

    from woof.wrf_hydro_route import soil_centres

    coupled = import_module(optin.load("wrf_hydro").__name__ + ".coupled")
    d = tmp_path / "dom"
    synthetic.write(d, nx=8, ny=6)
    nl = synthetic.namelist(**edit)
    if "GWBASESWCRT = 3" in nl:
        nl = nl.replace("GWBASESWCRT = 1\n", "")
    (d / "hydro.namelist").write_text(nl)
    with pytest.raises(NotImplementedError, match=match):
        coupled.WrfHydroCoupled(d / "hydro.namelist", ix=8, jx=6,
                                zs=soil_centres([0.1, 0.3, 0.6, 1.0]),
                                vegtyp=np.full((6, 8), 10), soltyp=np.full((6, 8), 6),
                                hydro_dt=60.0, start=datetime.datetime(2020, 1, 1))


@needs_optin
def test_a_missing_input_is_refused_by_name(tmp_path):
    from importlib import import_module

    from woof.wrf_hydro_route import soil_centres

    rd = import_module(optin.load("wrf_hydro").__name__ + ".routing_domain")
    coupled = import_module(optin.load("wrf_hydro").__name__ + ".coupled")
    d = tmp_path / "dom"
    synthetic.write(d, nx=8, ny=6)
    for name, word in (("GWBUCKPARM.nc", "GWBUCKPARM_file"), ("LAKEPARM.nc", "route_lake_f"),
                       ("CHANPARM.TBL", "CHANPARM.TBL"), ("Fulldom_hires.nc",
                                                          "GEO_FINEGRID_FLNM")):
        synthetic.write(d, nx=8, ny=6)
        (d / name).unlink()
        with pytest.raises(rd.MissingInput, match=word):
            coupled.WrfHydroCoupled(d / "hydro.namelist", ix=8, jx=6,
                                    zs=soil_centres([0.1, 0.3, 0.6, 1.0]),
                                    vegtyp=np.full((6, 8), 10), soltyp=np.full((6, 8), 6),
                                    hydro_dt=60.0, start=datetime.datetime(2020, 1, 1))


@needs_optin
@pytest.mark.gpu
@requires_gpu
def test_hydro_ini_builds_wrf_hydros_gridded_network(tmp_path):
    _, c = _coupled(tmp_path)
    net = c.dom.network
    agg, ixrt, jxrt = 4, 32, 24
    valley = jxrt // 2
    # one channel row from i = 2 to the east edge, broken by a 3-pixel lake
    assert len(net["chanxi"]) == (ixrt - 1) - 3
    # nodes are numbered in raster order (MPP_READ_ROUTEDIM on one process)
    assert (np.diff(net["chanxi"]) > 0).all() and (net["chanyj"] == valley + 1).all()
    typel = net["typel"].tolist()
    into_lake = typel.index(2)
    assert net["lakenode"][into_lake] == 1                      # drains into lake 1
    assert typel[-1] == 1 and typel.count(1) == 1               # the east-edge pour point
    assert net["to_node"][0] == 2 and net["to_node"][into_lake] == -9999
    outlet = net["chanxi"][into_lake]               # 0-based column of the next node
    assert c.dom.ch_netrt[valley, outlet + 3] == 1  # the lake's outlet node
    # channel pixels keep 5 mm retention; CHANPARM by order; CVOL from HLINK_INIT
    assert (c.dom.retention[c.dom.ch_netrt >= 0] == np.float32(5.0)).all()
    assert set(net["bw"].tolist()) == {np.float32(1.6), np.float32(2.4)}
    f = np.float32
    k = 0
    want = f(f(f(net["bw"][k] + f(f(1.0) / net["chsslp"][k]) * net["hlink"][k]) * net["hlink"][k])
             * net["chanlen"][k])
    assert net["cvol"][k] == want
    # north of the valley the steepest descent is south-east: 3.5 m over a
    # diagonal beats 2 m south and 1.5 m east
    assert tuple(c.dom.sdir[:, valley + 3, 5]) == (7, valley + 3, 4)
    assert c.lake_mask.get()[:, -agg:].max() < 0                # cleared under LSM water


@needs_optin
@pytest.mark.gpu
@requires_gpu
def test_water_is_conserved_across_each_coupling_cycle(tmp_path):
    import cupy as cp

    _, c = _coupled(tmp_path)
    c.budget_log = []
    ny, nx = c.shape
    smois = cp.full((4, ny, nx), 0.35, cp.float32)
    smois[3] = cp.float32(0.435)                  # saturated at depth: lateral flow
    smois[:, :, -1] = 1.0
    sh2o = smois.copy()
    infx = cp.full((ny, nx), 15.0, cp.float32)
    infx[:, -1] = 0
    sold = cp.full((ny, nx), 0.2, cp.float32)
    sold[:, -1] = 0
    head = cp.zeros((ny, nx), cp.float32)
    for _ in range(6):
        c.exchange(smois=smois, sh2o=sh2o, infxsrt=infx, soldrain=sold, sfcheadrt=head)
        t = c.budget_log[-1]
        # the LSM's soil water and ponded water against what came back plus
        # what left to the channel, the lakes and over the domain edges
        out = (t["lsm_out"] + t["to_channel"] + t["to_lake"] + t["boundary_ov"]
               + t["boundary_sub"])
        assert abs(t["lsm_in"] - out) <= 1e-6 * t["lsm_in"], t
        # the bucket: SOLDRAIN in, its storage change and its baseflow out
        assert abs(t["bucket_in"] - t["soldrain"]) <= 1e-5 * t["soldrain"]
        assert abs(t["bucket_in"] - t["bucket_out"] - t["bucket_change"]) <= 1e-4 * t["bucket_in"]
        # baseflow reaches the channel as QINFLOWBASE, volume for volume
        assert abs(t["qinflowbase"] - t["bucket_out"]) <= 1e-5 * max(t["bucket_out"], 1.0)
    assert float(head.max()) > 0                   # SFCHEADRT handed back to the LSM
    assert any(t["to_lake"] > 0 for t in c.budget_log)
    assert all(q > 0 for _, _, _, q, _ in c.streamflow())


@needs_optin
@pytest.mark.gpu
@requires_gpu
def test_streamflow_and_routing_output_files(tmp_path):
    import cupy as cp
    import netCDF4

    _, c = _coupled(tmp_path)
    ny, nx = c.shape
    smois = cp.full((4, ny, nx), 0.3, cp.float32)
    sh2o = smois.copy()
    z = cp.zeros((ny, nx), cp.float32)
    for _ in range(2):
        c.exchange(smois=smois, sh2o=sh2o, infxsrt=z + 1, soldrain=z + 0.1, sfcheadrt=z.copy())
    out = tmp_path / "out"
    lines = (out / "frxst_pts_out.txt").read_text().splitlines()
    assert len(lines) == 3 * 2                     # t0 and two 1-minute frames, 2 points
    first = lines[0].split(",")
    assert first[0] == "       0" and first[1] == "2020-01-01 00:00:00"
    assert first[2].strip() == "1" and len(first) == 7 + 1
    assert lines[-1].startswith("     120,2020-01-01 00:02:00")
    with netCDF4.Dataset(out / "202001010002.CHRTOUT_DOMAIN1") as ds:
        assert np.array_equal(ds["streamflow"][:], c.qlink1.get())
    with netCDF4.Dataset(out / "202001010002.RTOUT_DOMAIN1") as ds:
        assert ds["SOIL_M"].shape == (1, 4, 24, 32)
        assert np.array_equal(ds["sfcheadsubrt"][0][::-1], c.ostate.head.get())


@needs_optin
@pytest.mark.gpu
@requires_gpu
def test_a_short_coupled_noahmp_forecast(tmp_path, monkeypatch):
    """Noah-MP's WRF_HYDRO planes (QTILEDRAIN as well) through the driver."""
    import cupy as cp

    from test_physics_driver import _full_state
    from woof import wrf_hydro_route as route
    from woof.core.dycore import step

    path = synthetic.write(tmp_path / "dom", nx=8, ny=6, dxrt=500.0)
    monkeypatch.setenv(route.ENV_NAMELIST, str(path))
    monkeypatch.setenv(route.ENV_OUTPUT, str(tmp_path / "out"))
    state, cfg, driver = _full_state(
        nx=8, ny=6, sf_surface_physics=4,
        radiation_start_time=datetime.datetime(2020, 6, 1, 12),
        radiation_latitude=np.full((6, 8), 40.0), radiation_longitude=np.full((6, 8), -100.0))
    seam = driver.wrf_hydro
    assert set(seam.planes) == {"sfcheadrt", "infxsrt", "soldrain", "qtiledrain"}
    land = driver.fields["xland"].get() < 1.5
    seam.planes["sfcheadrt"][...] = cp.asarray(np.where(land, 20.0, 0.0).astype(np.float32))
    seam.routing.budget_log = []
    for _ in range(3):
        step(state, cfg)
    assert len(seam.routing.budget_log) == 3
    assert float(seam.planes["infxsrt"].max()) > 0
    for t in seam.routing.budget_log:
        out = (t["lsm_out"] + t["to_channel"] + t["to_lake"] + t["boundary_ov"]
               + t["boundary_sub"])
        assert abs(t["lsm_in"] - out) <= 1e-6 * t["lsm_in"]


@needs_optin
@pytest.mark.gpu
@requires_gpu
def test_a_short_coupled_forecast_through_the_driver(tmp_path, monkeypatch):
    import cupy as cp

    from test_physics_driver import _full_state
    from woof import wrf_hydro_route as route
    from woof.core import noah
    from woof.core.dycore import step
    from woof.io.restart import RestartManifestError, _driver_manifest

    path = synthetic.write(tmp_path / "dom", nx=8, ny=6, dxrt=500.0)
    monkeypatch.setenv(route.ENV_NAMELIST, str(path))
    monkeypatch.setenv(route.ENV_OUTPUT, str(tmp_path / "out"))
    seen = []
    real = noah.launch_noah

    def spy(*a, **k):
        seen.append(k.get("wrf_hydro"))
        return real(*a, **k)

    monkeypatch.setattr("woof.core.physics.launch_noah", spy)
    state, cfg, driver = _full_state(nx=8, ny=6,
                                     radiation_start_time=datetime.datetime(2020, 6, 1, 12))
    seam = driver.wrf_hydro
    land = driver.fields["xland"].get() < 1.5
    seam.planes["sfcheadrt"][...] = cp.asarray(np.where(land, 20.0, 0.0).astype(np.float32))
    seam.routing.budget_log = []
    for _ in range(4):
        step(state, cfg)
    assert len(seen) == 4 and all(p is seam.planes for p in seen)
    assert len(seam.routing.budget_log) == 4          # one exchange per LSM call
    assert float(seam.planes["infxsrt"].max()) > 0    # the LSM wrote INFXSRT from the head
    for t in seam.routing.budget_log:
        out = (t["lsm_out"] + t["to_channel"] + t["to_lake"] + t["boundary_ov"]
               + t["boundary_sub"])
        assert abs(t["lsm_in"] - out) <= 1e-6 * t["lsm_in"]
    history = driver.output_fields()
    assert {"SFCHEADRT", "INFXSRT", "SOLDRAIN"} <= set(history)
    assert (tmp_path / "out" / "frxst_pts_out.txt").is_file()
    manifest = _driver_manifest(driver, digest=True)
    assert "wrf_hydro" not in str(sorted(manifest))
    with pytest.raises(RestartManifestError, match="WRF-Hydro"):
        _driver_manifest(driver)
    from woof.state_digest import canonical_state_digest
    assert "wrf_hydro/channel" not in seam.digest_arrays()
    clock = SimpleNamespace(dtbc_fp32=float(cfg.dt))
    before = canonical_state_digest(state, clock)
    seam.routing.cvol[0] += cp.float32(1.0)          # routing state enters the digest
    assert canonical_state_digest(state, clock) != before


@pytest.mark.gpu
@requires_gpu
def test_an_uncoupled_forecast_calls_the_lsm_as_before(no_hydro_env, monkeypatch):
    """No [wrf_hydro]: no plane, no attribute, the default Noah call."""
    from test_physics_driver import _full_state
    from woof.core import noah
    from woof.core.dycore import step

    calls = []
    real = noah.launch_noah

    def spy(*a, **k):
        calls.append(set(k))
        return real(*a, **k)

    monkeypatch.setattr("woof.core.physics.launch_noah", spy)
    state, cfg, driver = _full_state()
    assert not hasattr(driver, "wrf_hydro")
    for _ in range(2):
        step(state, cfg)
    assert calls and all("wrf_hydro" not in k for k in calls)
    assert not {"SFCHEADRT", "INFXSRT", "SOLDRAIN"} & set(driver.output_fields())
