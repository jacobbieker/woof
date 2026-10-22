"""Wind-farm power, GPU tier: the kernel against the host road and the
oracles, word for word, and real dycore steps.

The host road and the oracle chain it equals are graded in
tests/test_windfarm_power_diag.py.  Here every device output -- the frame
evaluation, the hub interpolants, the window sums and maxima, the farm
pair and the frame fields -- must be the host's raw words (max ULP 0),
and the power, Ct, density and corrected speed must be the oracles'
(TurbineCurve, woof.energy.extract, woof.energy.ramps and
woof.energy.power_farm's density correction) directly.
Tiny domains only, synthetic turbines.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core import interval_diagnostics
from woof.core import windfarm_power_diag as wp
from woof.core.hub_wind_diag import locate_sites
from woof.io.history_selection import HistorySelection

from test_windfarm_power_diag import (_latlon, _rough_state, _turbine_file,
                                      max_ulp)

pytestmark = pytest.mark.gpu

_INPUTS = ("u", "v", "php", "phb", "ht", "thp", "thb", "p", "qv")

#: Many turbines over the 6 x 7 test grid: every interior cell, a range of
#: hubs and rotors (strips below the first level and above several), and
#: both a library curve and parametric ones.
def _many_rows():
    rows = []
    k = 0
    for j in range(1, 5):
        for i in range(1, 6):
            hub = 40.0 + 23.0 * k
            rotor = min(1.9 * hub, 60.0 + 7.0 * k)
            kind = ("E-115/3000," if k % 4 == 0 and abs(rotor - 115) < 1
                    else f"generic,{1500 + 350 * k}")
            if k % 5 == 0:
                hub, rotor, kind = 100.0, 115.0, "E-115/3000,"
            rows.append(f"wtg-{k:02d},{55.0 + 0.01 * j},{2.0 + 0.01 * i},"
                        f"{hub:g},{rotor:g},{kind}\n")
            k += 1
    return "".join(rows)


def _device_state(host):
    import cupy as cp

    return SimpleNamespace(**{name: cp.asarray(getattr(host, name))
                              for name in _INPUTS})


def _pair(tmp_path, host, rows=None):
    import cupy as cp

    ny, nx = host.ht.shape
    selection = HistorySelection.from_mapping(
        {"windfarm_turbines": str(_turbine_file(tmp_path, rows))})
    lat, lon = _latlon(ny, nx)
    located = locate_sites([(t.name, t.lat, t.lon)
                            for t in selection.windfarm_power.turbines],
                           lat, lon, dx=700.0, dy=1100.0)["inside"]
    host_diag = wp.WindfarmPowerDiagnostic(
        selection.windfarm_power, located, ny=ny, nx=nx)
    dev_diag = wp.WindfarmPowerDiagnostic(
        selection.windfarm_power, located, ny=ny, nx=nx, xp=cp,
        device_id=int(cp.cuda.Device().id))
    return host_diag, dev_diag


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("phb_1d", [True, False])
@pytest.mark.parametrize("dry", [False, True])
def test_gpu_equals_the_host_road_and_the_oracles_to_zero_ulp(
        tmp_path, phb_1d, dry):
    import cupy as cp
    from woof.energy.extract import air_density, air_temperature
    from woof.energy.power_farm import (
        density_corrected_speed as power_farm_corrected)
    from woof.energy.ramps import density_corrected_speed

    rng = np.random.default_rng(11)
    host = _rough_state(rng, phb_1d=phb_1d)
    if dry:
        del host.qv
    device = SimpleNamespace(**{name: cp.asarray(getattr(host, name))
                                for name in _INPUTS if hasattr(host, name)})
    host_diag, dev_diag = _pair(tmp_path, host, _many_rows())
    assert host_diag.n >= 15
    words = {}
    for step in range(5):
        u = rng.normal(9.0 + 3.0 * step, 5.0, host.u.shape).astype(np.float32)
        host.u[...] = u
        device.u[...] = cp.asarray(u)
        want = host_diag.evaluate(host)
        got = dev_diag.evaluate(device)
        assert set(got) == set(want)
        for name, value in want.items():
            assert max_ulp(got[name], value) == 0, (step, name)
            words[name] = words.get(name, 0) + value.size
        # GPU against the oracles directly.
        p = got["p_hub"].astype(np.float64)
        q = (np.zeros_like(p) if dry else got["qv_hub"].astype(np.float64))
        rho = air_density(p, air_temperature(
            got["theta_hub"].astype(np.float64), p), q)
        assert max_ulp(got["rho"], rho) == 0
        v_eq = density_corrected_speed(got["rews"], got["rho"])
        assert max_ulp(got["v_eq"], v_eq) == 0
        assert max_ulp(got["v_eq"], power_farm_corrected(got["rews"],
                                                         got["rho"])) == 0
        running = v_eq <= host_diag.cut_out_ms
        power = np.array([float(c.fraction(v_eq[n])) * (c.rated_kw * 1000.0)
                          for n, c in enumerate(host_diag.curves)])
        ct = np.array([float(c.thrust_coefficient(v_eq[n]))
                       for n, c in enumerate(host_diag.curves)])
        assert max_ulp(got["power"], np.where(running, power, 0.0)) == 0
        assert max_ulp(got["ct"], np.where(running, ct, 0.0)) == 0
        host_diag.on_step(host, 10.0 + step)
        dev_diag.on_step(device, 10.0 + step)
    sums_h, max_h, farm_h = host_diag._window_host()
    sums_d, max_d, farm_d = dev_diag._window_host()
    for q in wp.WINDOW_QUANTITIES:
        assert max_ulp(sums_d[q], sums_h[q]) == 0, q
        assert max_ulp(max_d[q], max_h[q]) == 0, q
    assert max_ulp(np.array(farm_d), np.array(farm_h)) == 0
    want_fields = host_diag.history_fields(host)
    got_fields = dev_diag.history_fields(device)
    for name, value in want_fields.items():
        assert isinstance(got_fields[name], np.ndarray), name
        assert max_ulp(got_fields[name], value) == 0, name
    path = dev_diag.close_interval(device, valid_time=datetime(2025, 1, 1),
                                   output_dir=tmp_path / "gpu", grid_id=1)
    assert path.is_file()
    print("words compared:", words)


@pytest.mark.gpu
@requires_gpu
def test_gpu_equals_the_host_road_over_a_sweep_of_turbines_and_winds(
        tmp_path):
    """A turbine in every interior cell of a 30 x 36 grid (1,088), random
    hubs, rotors and ratings, winds from calm through cut-out: max ULP 0
    on every evaluated word, every window word and the farm pair."""
    import cupy as cp
    from test_windfarm_power_diag import _state

    rng = np.random.default_rng(23)
    ny, nx, nz = 30, 36, 40
    host = _state(nz=nz, ny=ny, nx=nx, dz=22.0, hgt=0.0)
    host.ht[...] = rng.uniform(0.0, 900.0, (ny, nx))
    host.php[...] = (host.ht[None] * np.float32(9.81) + rng.uniform(
        -6.0, 6.0, host.php.shape) * np.float32(9.81)).astype(np.float32)
    host.thp[...] = rng.normal(0.0, 2.0, host.thp.shape)
    host.p[...] = rng.uniform(85000.0, 102000.0, host.p.shape)
    host.qv[...] = rng.uniform(0.0, 0.02, host.qv.shape)
    lines = []
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            rotor = float(rng.uniform(40.0, 220.0))
            hub = float(rng.uniform(0.55 * rotor, 0.55 * rotor + 120.0))
            rated = float(rng.uniform(800.0, 15000.0))
            lines.append(f"w{j:02d}{i:02d},{55.0 + 0.01 * j:.4f},"
                         f"{2.0 + 0.01 * i:.4f},{hub:.3f},{rotor:.3f},"
                         f"generic,{rated:.1f}\n")
    lines[0] = "wlib,55.0100,2.0100,100,115,E-115/3000,\n"
    path = _turbine_file(tmp_path, "".join(lines))
    selection = HistorySelection.from_mapping({"windfarm_turbines": str(path)})
    lat, lon = _latlon(ny, nx)
    located = locate_sites([(t.name, t.lat, t.lon)
                            for t in selection.windfarm_power.turbines],
                           lat, lon, dx=700.0, dy=1100.0)["inside"]
    assert len(located) == (ny - 2) * (nx - 2)
    host_diag = wp.WindfarmPowerDiagnostic(
        selection.windfarm_power, located, ny=ny, nx=nx)
    dev_diag = wp.WindfarmPowerDiagnostic(
        selection.windfarm_power, located, ny=ny, nx=nx, xp=cp,
        device_id=int(cp.cuda.Device().id))
    device = _device_state(host)
    compared = 0
    for step, mean in enumerate((0.0, 3.0, 8.0, 12.0, 20.0, 26.0, 35.0)):
        u = rng.normal(mean, 2.0 + 0.2 * mean, host.u.shape).astype(np.float32)
        v = rng.normal(0.0, 3.0, host.v.shape).astype(np.float32)
        host.u[...] = u
        host.v[...] = v
        device.u[...] = cp.asarray(u)
        device.v[...] = cp.asarray(v)
        want = host_diag.evaluate(host)
        got = dev_diag.evaluate(device)
        for name, value in want.items():
            assert max_ulp(got[name], value) == 0, (step, name)
            compared += value.size
        host_diag.on_step(host, 6.0 + step)
        dev_diag.on_step(device, 6.0 + step)
    sums_h, max_h, farm_h = host_diag._window_host()
    sums_d, max_d, farm_d = dev_diag._window_host()
    for q in wp.WINDOW_QUANTITIES:
        assert max_ulp(sums_d[q], sums_h[q]) == 0, q
        assert max_ulp(max_d[q], max_h[q]) == 0, q
    assert max_ulp(np.array(farm_d), np.array(farm_h)) == 0
    # The sweep reached every regime: parked, below rated and at rated.
    assert (want["power"] == 0.0).any()
    assert ((want["power"] > 0.0) & (want["power"] < host_diag.rated_w)).any()
    print("sweep words compared:", compared)


@pytest.mark.gpu
@requires_gpu
def test_gpu_step_does_not_synchronise_or_copy(tmp_path):
    """The fold is two kernel launches: no device-to-host copy per step."""
    import cupy as cp

    host = _rough_state(np.random.default_rng(2))
    device = _device_state(host)
    _, dev_diag = _pair(tmp_path, host)
    dev_diag.on_step(device, 10.0)                 # compile outside the count
    cp.cuda.Device().synchronize()
    calls = []
    original = cp.ndarray.get

    def counted(self, *args, **kwargs):
        calls.append(self.shape)
        return original(self, *args, **kwargs)

    cp.ndarray.get = counted
    try:
        for _ in range(5):
            dev_diag.on_step(device, 10.0)
    finally:
        cp.ndarray.get = original
    assert calls == []


@pytest.mark.gpu
@requires_gpu
def test_gpu_dycore_steps_are_inert_and_power_is_physical(tmp_path):
    """Real dycore steps on a tiny moist domain (60 model seconds): the
    attached window moves no model byte, its frame equals the host road on
    the stepped state word for word, power stays within [0, rated], Ct
    within [0, 8/9]."""
    import cupy as cp
    from woof.core.dycore import step
    from test_mp_spec_zone_ring import _square_moist_state

    def make():
        state, cfg = _square_moist_state(ny=12, nx=14, nz=30, dt=10.0,
                                         open_x=True, open_y=True)
        z = state.height_half()
        shear = (6.0 + 0.004 * z).astype(np.float32)
        state.u[...] = cp.asarray(np.broadcast_to(shear[:, None, None],
                                                  state.u.shape))
        state.v[...] = cp.float32(-1.5)
        return state, cfg

    s_off, cfg = make()
    s_on, _ = make()
    rows = ("wtg-a,55.04,2.04,100,120,generic,3500\n"
            "wtg-b,55.06,2.08,140,170,generic,6000\n"
            "wtg-c,55.08,2.10,90,80,generic,2000\n")
    selection = HistorySelection.from_mapping(
        {"windfarm_turbines": str(_turbine_file(tmp_path, rows))})
    lat, lon = _latlon(cfg.ny, cfg.nx)
    diag = wp.attach_windfarm_power(
        s_on, selection, nx=cfg.nx, ny=cfg.ny, dx=700.0, dy=1100.0,
        grid_id=1, latlon=lambda: (lat, lon))
    assert diag is not None and diag.n == 3
    for _ in range(3):
        step(s_off, cfg)
        step(s_on, cfg)
    fields = interval_diagnostics.history_fields(s_on)
    assert float(fields["WINDFARM_WINDOW_S"]) == 30.0
    host = SimpleNamespace(**{name: cp.asnumpy(getattr(s_on, name))
                              for name in _INPUTS})
    reference = wp.WindfarmPowerDiagnostic(
        selection.windfarm_power, diag.rows, ny=cfg.ny, nx=cfg.nx)
    want = reference.evaluate(host)
    got = diag.evaluate(s_on)
    for name, value in want.items():
        assert max_ulp(got[name], value) == 0, name
    assert np.isfinite(fields["POWER_TURBINE"]).all()
    assert (fields["POWER_TURBINE"] >= 0).all()
    assert (fields["POWER_TURBINE"] <= diag.rated_w).all()
    assert ((fields["CT_TURBINE"] >= 0) & (fields["CT_TURBINE"] <= 8 / 9)).all()
    assert (fields["REWS_TURBINE"] > 1.0).all()
    interval_diagnostics.close_interval(
        s_on, valid_time=datetime(2025, 1, 1, 0, 0, 30), output_dir=tmp_path,
        grid_id=1)
    for _ in range(3):
        step(s_off, cfg)
        step(s_on, cfg)
    for name in ("u", "v", "w", "thp", "php", "mup", "p", "qv", "qc", "qr"):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(s_on, name)), cp.asnumpy(getattr(s_off, name)),
            err_msg=f"the wind-farm power window moved model field {name}")
