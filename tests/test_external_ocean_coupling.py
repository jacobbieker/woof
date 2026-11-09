"""WRF's external-ocean calls wired into the driver (woof.ocean.external).

cpl_store_input at itimestep 1, cpl_rcv's SST merge plus sst_update's water
lines at the top of every surface-driver call, and cpl_snd at the end of
every step with the coupling-interval average handed to the ocean.  The
exchange kernels themselves are graded against WRF 4.7.1 in
tests/test_wrf_cpl_parity.py; here the wiring is checked against a stub
ocean through the real driver, and the NEMO column (an opt-in) behind it.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu


class StubOcean:
    """An ExternalOcean that records what it is handed."""

    couples_currents = False

    def __init__(self, shape, sst):
        import cupy as cp

        from woof.ocean.interface import OceanSurface

        self._surface = OceanSurface(sst=cp.full(shape, sst, cp.float32),
                                     uoce=cp.zeros(shape, cp.float32),
                                     voce=cp.zeros(shape, cp.float32))
        self.fluxes, self.steps = [], []

    def set_fluxes(self, fluxes):
        self.fluxes.append({k: getattr(fluxes, k).get() for k in (
            "evap_precip", "surf_net_solar", "surf_net_non_solar", "taux", "tauy", "taumod")})

    def step(self, dt):
        self.steps.append(dt)

    @property
    def surface(self):
        return self._surface


def _toml(tmp_path, body):
    p = tmp_path / "ocean.toml"
    p.write_text(body)
    return p


@pytest.mark.parametrize("body,match", [
    ('[external_ocean]\nmodel="roms"\ncoupling_interval_seconds=60.0\nstate="s.npz"\n', "model"),
    ('[external_ocean]\nmodel="nemo_column"\ncoupling_interval_seconds=0.0\nstate="s.npz"\n',
     "coupling_interval"),
    ('[external_ocean]\nmodel="nemo_column"\ncoupling_interval_seconds=60.0\n', "state"),
    ('[external_ocean]\nmodel="nemo_column"\ncoupling_interval_seconds=60.0\nstate="s.npz"\n'
     'remap="conserv"\n', "unknown"),
])
def test_external_ocean_config_is_checked(tmp_path, body, match):
    import tomllib

    from woof.ocean import external

    path = _toml(tmp_path, body)
    with pytest.raises(ValueError, match=match):
        external.read_config(tomllib.loads(path.read_text()), path)


def test_the_nemo_column_is_reached_only_through_its_opt_in(monkeypatch, tmp_path):
    from woof import optin
    from woof.ocean import external

    monkeypatch.delenv("WOOF_ENABLE_NEMO", raising=False)
    cfg = dict(model="nemo_column", coupling_interval_seconds=60.00,
               state=str(tmp_path / "s.npz"), options={})
    with pytest.raises(optin.OptInUnavailable, match="CeCILL"):
        external.build(cfg, latitude=np.zeros((2, 3)), shape=(2, 3), dt=10.0)


def test_restart_of_an_externally_coupled_run_is_refused():
    from types import SimpleNamespace

    from woof.io.restart import RestartManifestError, _driver_manifest

    with pytest.raises(RestartManifestError, match="external ocean"):
        _driver_manifest(SimpleNamespace(external_ocean=object()))


def test_checkpoint_intervals_are_refused_before_the_first_step(tmp_path):
    """A coupled run with restart_interval_s > 0 is refused when the ocean is
    attached, not an interval in at its first checkpoint write."""
    from types import SimpleNamespace

    from woof.ocean import external

    toml = _toml(tmp_path, '[external_ocean]\nmodel="nemo_column"\n'
                           'coupling_interval_seconds=60.0\nstate="s.npz"\n')
    cfg = SimpleNamespace(ocean_config=str(toml), restart_interval_s=3600.0, dt=60.0)
    with pytest.raises(NotImplementedError, match="restart_interval_s = 0"):
        external.attach(SimpleNamespace(fields={}), cfg, latitude=np.zeros((2, 3)))


@pytest.mark.gpu
@requires_gpu
def test_coupling_refuses_an_ocean_with_currents_and_a_ragged_interval():
    from woof.ocean.external import ExternalOceanCoupling

    ocean = StubOcean((2, 3), 290.0)
    with pytest.raises(ValueError, match="whole number"):
        ExternalOceanCoupling(ocean, cplmask=np.ones((2, 3)), interval=25.0, dt=10.0, shape=(2, 3))
    ocean.couples_currents = True
    with pytest.raises(NotImplementedError, match="UOCE"):
        ExternalOceanCoupling(ocean, cplmask=np.ones((2, 3)), interval=20.0, dt=10.0, shape=(2, 3))


@pytest.mark.gpu
@requires_gpu
def test_receive_merges_onto_sst_input_and_moves_water_tsk():
    import cupy as cp

    from woof.ocean.external import ExternalOceanCoupling

    shape = (2, 3)
    ocean = StubOcean(shape, 291.5)
    mask = np.array([[1.0, 0.25, 0.0], [1.0, 1.0, 1.0]], np.float32)
    cpl = ExternalOceanCoupling(ocean, cplmask=mask, interval=20.0, dt=10.0, shape=shape)
    sst0 = np.full(shape, 288.0, np.float32)
    fields = dict(sst=cp.asarray(sst0), tsk=cp.full(shape, 300.0, cp.float32),
                  xland=cp.asarray(np.array([[2, 2, 2], [1, 2, 2]], np.float32)),
                  tslb=cp.full((4, *shape), 280.0, cp.float32))
    cpl.store_input(fields["sst"])
    fields["sst"][...] = cp.float32(260.0)            # SST_INPUT, not SST, is the base
    ocean.surface.sst[1, 2] = cp.float32(355.0)        # outside sst_update's window
    cpl.receive(fields)
    sst = fields["sst"].get()
    want = np.array([[291.5, np.float32(288.0) * np.float32(0.75) + np.float32(0.25)
                      * np.float32(291.5), 288.0], [291.5, 291.5, 355.0]], np.float32)
    assert np.allclose(sst, want, rtol=0, atol=1e-4)
    tsk = fields["tsk"].get()
    assert tsk[1, 0] == 300.0                          # land keeps its TSK
    assert tsk[1, 2] == 300.0                          # SST out of range
    assert tsk[0, 0] == sst[0, 0] and tsk[0, 2] == 288.0
    assert (fields["tslb"][0].get() == np.where([[1, 1, 1], [0, 1, 0]], sst, 280.0)).all()
    assert (fields["tslb"][1].get() == 280.0).all()


@pytest.mark.gpu
@requires_gpu
def test_the_driver_runs_the_three_calls_at_wrf_places(tmp_path, monkeypatch):
    import cupy as cp

    from test_physics_driver import _full_state
    from woof.core.dycore import step
    from woof.ocean import external

    made = {}

    def build(cfg, *, latitude, shape, dt):
        ocean = StubOcean(shape, 291.25)
        made["ocean"] = ocean
        return external.ExternalOceanCoupling(ocean, cplmask=np.ones(shape),
                                              interval=cfg["coupling_interval_seconds"],
                                              dt=dt, shape=shape)

    monkeypatch.setattr(external, "build", build)
    toml = _toml(tmp_path, '[external_ocean]\nmodel="nemo_column"\n'
                           'coupling_interval_seconds=20.0\nstate="unused.npz"\n')
    state, cfg, driver = _full_state(ocean_config=str(toml), ra_physics=0, ra_lw_physics=0,
                                     ra_sw_physics=1,
                                     radiation_start_time=__import__("datetime").datetime(
                                         2011, 6, 26, 12),
                                     radiation_latitude=np.full((2, 6), 45.0),
                                     radiation_longitude=np.full((2, 6), -5.0))
    coupling = driver.external_ocean
    sst_input = coupling.sst_input.get()
    assert (sst_input == driver.fields["sst"].get()).all()
    ocean = made["ocean"]
    for n in range(1, 5):
        step(state, cfg)
        assert len(ocean.steps) == n // 2              # a coupling time every 2 steps
    water = driver.fields["xland"].get() > 1.5
    assert water.any()
    assert (driver.fields["sst"].get() == np.float32(291.25)).all()
    assert (driver.fields["tsk"].get()[water] == np.float32(291.25)).all()
    assert ocean.steps == [20.0, 20.0]
    for f in ocean.fluxes:
        assert all(np.isfinite(v).all() for v in f.values())
        assert (f["taumod"] >= 0).all()
        assert (f["surf_net_solar"] > 0).all()        # GSW from the declared shortwave
    assert float(cp.abs(coupling.u_phy).max()) > 0     # phy_prep's lowest level, captured
    # The run's final state digest reads the driver without the ocean; a
    # checkpoint of it is still refused.
    from woof.io.restart import RestartManifestError, _driver_manifest
    assert _driver_manifest(driver, digest=True)
    with pytest.raises(RestartManifestError, match="external ocean"):
        _driver_manifest(driver)
    # ... and takes the coupled state from the coupling's own inventory, so
    # two runs that differ only in the ocean have different digests.
    from types import SimpleNamespace

    from woof.state_digest import canonical_state_digest
    inventory = coupling.digest_arrays()
    assert "external_ocean/coupling/sst_input" in inventory
    assert "external_ocean/averager/elapsed" in inventory
    assert "external_ocean/ocean/surface/sst" in inventory
    clock = SimpleNamespace(dtbc_fp32=float(cfg.dt))
    before = canonical_state_digest(state, clock)
    ocean.surface.sst[...] += cp.float32(0.5)
    assert canonical_state_digest(state, clock) != before


@pytest.mark.gpu
@requires_gpu
def test_the_nemo_column_couples_through_the_driver_wiring(tmp_path):
    """With the NEMO opt-in on: the column's SST reaches the atmosphere."""
    from woof import optin

    if optin.unavailable_reason("nemo") is not None:
        pytest.skip(f"opt-in rule: {optin.unavailable_reason('nemo')}")
    import cupy as cp

    from woof.ocean import external
    from woof.ocean.wrf_cpl import AtmosphereSurface

    shape, jpk = (2, 3), 11
    e3t = np.linspace(2.0, 12.0, jpk)
    gdepw = np.concatenate([[0.0], np.cumsum(e3t[:-1])])
    gdept = gdepw + 0.5 * e3t
    e3w = np.concatenate([[0.5 * e3t[0]], np.diff(gdept)])
    temp = np.broadcast_to((18.0 - 0.3 * np.arange(jpk))[:, None, None], (jpk, *shape)).copy()
    np.savez(tmp_path / "s.npz", mbkt=np.full(shape, jpk - 1), e3t=e3t, e3w=e3w, gdept=gdept,
             gdepw=gdepw, temperature=temp, salinity=np.full((jpk, *shape), 35.0),
             cplmask=np.ones(shape))
    cfg = dict(model="nemo_column", coupling_interval_seconds=1800.0,
               state=str(tmp_path / "s.npz"), options=dict(rn_Dt=900.0))
    cpl = external.build(cfg, latitude=np.full(shape, 50.0), shape=shape, dt=60.0)
    fields = dict(sst=cp.full(shape, 280.0, cp.float32), tsk=cp.full(shape, 280.0, cp.float32),
                  xland=cp.full(shape, 2.0, cp.float32))
    cpl.store_input(fields["sst"])
    cpl.receive(fields)
    first = fields["sst"].get().copy()
    assert np.allclose(first, 18.0 + 273.15, atol=0.1)    # the column's initial SST
    z = lambda v: cp.full(shape, v, cp.float32)  # noqa: E731
    surf = dict(qfx=z(0.0), gsw=z(0.0), glw=z(300.0), emiss=z(0.98), sst=fields["sst"],
                lh=z(0.0), hfx=z(150.0), ust=z(0.3))
    atm = dict(u=cp.full((2, *shape), 5.0, cp.float32), v=cp.zeros((2, *shape), cp.float32),
               rho=cp.full((2, *shape), 1.2, cp.float32))
    for _ in range(30):                                   # one coupling interval of 60 s steps
        cpl.capture_lowest_level(atm)
        cpl.send(surf, raincv=None, rainncv=z(0.0), dt=60.0)
    cpl.receive(fields)
    assert (fields["sst"].get() < first).all()            # a cooling flux cooled the column
    assert AtmosphereSurface is not None
