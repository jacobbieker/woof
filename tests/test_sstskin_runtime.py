import numpy as np
import pytest
from conftest import requires_gpu
from test_physics_driver import _full_state
from woof.io.restart import write_restart, restore_restart

@pytest.mark.gpu
@requires_gpu
def test_skin_advances_on_held_surface_steps_and_restores(tmp_path):
    import cupy as cp
    state,cfg,driver = _full_state(sst_skin=1, bldt=1., sf_surface_physics=0, bl_pbl_physics=0)
    driver.fields["gsw"][...] = cp.float32(1000.)
    driver.fields["glw"][...] = cp.float32(400.)
    driver.compute(state,cfg)
    water = driver.fields["xland"] > 1.5
    first = driver.fields["dtw"].copy()
    calls = driver.call_counts["sfclay"]
    state.elapsed_seconds = cfg.dt
    driver.compute(state,cfg)
    assert driver.call_counts["sfclay"] == calls
    assert bool(cp.any(driver.fields["dtw"][water] != first[water]))
    path=write_restart(tmp_path/"skin.npz",state,cfg)
    resumed,rcfg,rdriver = _full_state(sst_skin=1,bldt=1.,sf_surface_physics=0,bl_pbl_physics=0)
    restore_restart(path,resumed,rcfg)
    for key in ("sst","sstsk","dtw","tsk","tslb"):
        np.testing.assert_array_equal(cp.asnumpy(rdriver.fields[key]).view(np.uint32),cp.asnumpy(driver.fields[key]).view(np.uint32))
    state.elapsed_seconds = resumed.elapsed_seconds = 2*cfg.dt
    driver.compute(state,cfg)
    rdriver.compute(resumed,rcfg)
    for key in ("sstsk","dtw","tsk"):
        np.testing.assert_array_equal(cp.asnumpy(rdriver.fields[key]).view(np.uint32),cp.asnumpy(driver.fields[key]).view(np.uint32))

@pytest.mark.gpu
@requires_gpu
def test_bulk_sst_snapshot_does_not_alias_device_skin_temperature():
    import cupy as cp
    from woof.core.physics import initialize_physics
    state,cfg,old = _full_state(sst_skin=1,sf_surface_physics=0,bl_pbl_physics=0)
    temperature=old.fields["tsk"].copy()
    driver=initialize_physics(state,cfg,landmask=0.,tsk=temperature,sst=temperature,glw=400.,swdown=1000.)
    assert driver.fields["sst"].data.ptr != driver.fields["tsk"].data.ptr
    bulk=driver.fields["sst"].copy()
    driver.fields["gsw"][...] = cp.float32(1000.)
    driver.compute(state,cfg)
    np.testing.assert_array_equal(cp.asnumpy(driver.fields["sst"]).view(np.uint32),cp.asnumpy(bulk).view(np.uint32))
    assert bool(cp.any(driver.fields["tsk"] != bulk))

@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("given", ["separate", "none"])
def test_skin_keeps_the_bound_lake_water_temperature(monkeypatch, given):
    """sst_skin=1 used to rebuild SST from the caller's arrays after the
    lake-state binding, discarding the observed lake water temperature; one
    step later TSK (SST plus the skin correction) was the forcing's again."""
    import cupy as cp
    import woof.ingest.lake_state as obs
    from woof.core.physics import initialize_physics
    state,cfg,old = _full_state(sst_skin=1,sf_surface_physics=0,bl_pbl_physics=0)
    temperature=old.fields["tsk"].copy()
    lake=cp.float32(float(temperature[0, 0]) - 8.)
    def bind(fields, cfg, **kw):
        fields["sst"][0, 0] = fields["tsk"][0, 0] = lake
        return None, dict(schema='gpuwm.lake-init.v1', status='bound')
    monkeypatch.setattr(obs, "initialize_surface", bind)
    bulk=temperature.copy() if given == "separate" else None
    driver=initialize_physics(state,cfg,landmask=0.,tsk=temperature.copy(),sst=bulk,glw=400.,swdown=1000.)
    if bulk is not None:
        # The caller's SST is not written: the binding lives in the fields.
        cp.testing.assert_array_equal(bulk, temperature)
    assert float(driver.fields["xland"][0, 0]) > 1.5
    assert float(driver.fields["sst"][0, 0]) == float(lake)
    assert driver.fields["sst"].data.ptr != driver.fields["tsk"].data.ptr
    driver.fields["gsw"][...] = cp.float32(1000.)
    driver.fields["glw"][...] = cp.float32(400.)
    driver.compute(state,cfg)
    # SST stays the bound value; TSK is that plus a skin correction of at
    # most a few kelvin, nowhere near the forcing's 8 K warmer value.
    assert float(driver.fields["sst"][0, 0]) == float(lake)
    assert abs(float(driver.fields["tsk"][0, 0]) - float(lake)) < 3.


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("given", ["sst", "none"])
def test_open_water_component_starts_from_the_bound_lake(monkeypatch, given):
    """TSK_SEA (fractional sea ice) was rebuilt from the caller's arrays
    after the lake-state binding: a host caller's lakes kept the forcing."""
    import cupy as cp
    import woof.ingest.lake_state as obs
    from woof.core.physics import initialize_physics
    state, cfg, old = _full_state(sf_surface_physics=2, fractional_seaice=1)
    water = (0, cfg.nx - 1)
    def bind(fields, cfg, **kw):
        fields["sst"][water] = fields["tsk"][water] = cp.float32(280.)
        return None, dict(schema='gpuwm.lake-init.v1', status='bound')
    monkeypatch.setattr(obs, "initialize_surface", bind)
    landmask = np.ones((cfg.ny, cfg.nx)); landmask[:, -1] = 0.
    tsk = np.where(landmask > 0, 299., 296.)
    driver = initialize_physics(
        state, cfg, landmask=landmask, tsk=tsk, sst=tsk.copy() if given == "sst" else None,
        soil_temperature=np.stack([tsk - 0.5, tsk - 1.0, tsk - 1.5, tsk - 2.0]),
        soil_moisture=np.where(landmask > 0, .31, 1.)[None].repeat(4, 0),
        liquid_moisture=np.where(landmask > 0, .31, 1.)[None].repeat(4, 0),
        ivgtyp=np.where(landmask, 10, 17), isltyp=np.where(landmask, 6, 14),
        vegfra=55.0, tmn=286.0, swdown=450.0, glw=310.0, pblh=700.0)
    assert float(driver.fields["tsk_sea"][water]) == 280.
    assert float(driver.fields["tsk_sea"][0, 0]) == float(driver.fields["sst"][0, 0])
