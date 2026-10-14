"""Hub-height wind, GPU tier: the inline kernel and real dycore steps.

The CPU reference these grade against is tests/test_hub_wind_diag.py.
Tiny domains only (<= 40 x 40 x 30, <= 60 model seconds).
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core import interval_diagnostics
from woof.core import hub_wind_diag as hw
from woof.io.history_selection import HistorySelection, HubWindRequest

from test_hub_wind_diag import G, _column_state, _diag

pytestmark = pytest.mark.gpu


def _device_state(host):
    import cupy as cp

    return SimpleNamespace(**{name: cp.asarray(getattr(host, name))
                              for name in ("u", "v", "php", "phb", "ht",
                                           "sina", "cosa")})


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("phb_1d", [True, False])
def test_gpu_kernel_matches_the_reference(phb_1d):
    """Speed bit-identical (no FMA, IEEE sqrt), direction within atan2f ulps,
    and the window folded on device equals the host fold step for step."""
    import cupy as cp

    rng = np.random.default_rng(7)
    nz, ny, nx = 30, 40, 40
    host = _column_state(nz=nz, ny=ny, nx=nx, dz=35.0, hgt=0.0,
                         phb_1d=phb_1d)
    # Terrain and a rough, column-varying vertical coordinate.
    host.ht[...] = rng.uniform(0.0, 800.0, (ny, nx)).astype(np.float32)
    host.php[...] = (host.ht[None] * G + rng.uniform(
        -30.0, 30.0, (nz + 1, ny, nx)) * G).astype(np.float32)
    alpha = rng.uniform(-0.6, 0.6, (ny, nx))
    host.sina[...] = np.sin(alpha)
    host.cosa[...] = np.cos(alpha)
    heights = (20.0, 40.0, 80.0, 100.0, 120.0, 150.0, 200.0, 2000.0)
    host_diag = _diag(host, heights=heights)
    device = _device_state(host)
    request = HubWindRequest(heights_m=heights)
    dev_diag = hw.HubWindDiagnostic(request, ny=ny, nx=nx, xp=cp,
                                    device_id=int(device.u.device.id))
    for step in range(4):
        u = rng.normal(6.0, 4.0, host.u.shape).astype(np.float32)
        v = rng.normal(-2.0, 4.0, host.v.shape).astype(np.float32)
        host.u[...] = u
        host.v[...] = v
        device.u[...] = cp.asarray(u)
        device.v[...] = cp.asarray(v)
        dt = (10.0, 7.5, 12.5, 10.0)[step]
        host_diag.on_step(host, dt)
        dev_diag.on_step(device, dt)
    want = host_diag.history_fields(host)
    got = dev_diag.history_fields(device)
    for name in ("WSPD_HUB", "WSPD_HUB_MAX", "WSPD_HUB_MIN"):
        np.testing.assert_array_equal(cp.asnumpy(got[name]), want[name],
                                      err_msg=name)
    np.testing.assert_allclose(cp.asnumpy(got["WSPD_HUB_MEAN"]),
                               want["WSPD_HUB_MEAN"], rtol=1e-6)
    d_got = cp.asnumpy(got["WDIR_HUB"]).astype(np.float64)
    d_want = want["WDIR_HUB"].astype(np.float64)
    diff = np.abs(d_got - d_want)
    assert np.minimum(diff, 360.0 - diff).max() < 1e-3
    assert float(got["HUB_WINDOW_S"]) == 40.0


@pytest.mark.gpu
@requires_gpu
def test_gpu_dycore_steps_are_inert_and_the_window_restarts(tmp_path):
    """Real dycore steps on a tiny moist domain (60 model seconds): the
    attached window moves no model byte, its fields equal the reference
    evaluated on the stepped state, and it restarts at each frame."""
    import cupy as cp
    from woof.core.dycore import step
    from woof.io.wrfout import WrfoutWriter, state_frame
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
    selection = HistorySelection.from_mapping(
        {"hub_heights_m": [40, 100, 150, 200]})
    diag = hw.attach_hub_wind(s_on, selection, nx=cfg.nx, ny=cfg.ny,
                              dx=cfg.dx, dy=cfg.dy, grid_id=1)
    for _ in range(3):
        step(s_off, cfg)
        step(s_on, cfg)
    frame_fields = interval_diagnostics.history_fields(s_on)
    assert float(frame_fields["HUB_WINDOW_S"]) == 30.0
    host = SimpleNamespace(**{name: cp.asnumpy(getattr(s_on, name))
                              for name in ("u", "v", "php", "phb", "ht",
                                           "sina", "cosa")})
    want_speed, want_dir = hw.hub_wind_np(
        host.u, host.v, host.php, host.phb, host.ht, host.sina, host.cosa,
        diag.heights_m)
    np.testing.assert_array_equal(cp.asnumpy(frame_fields["WSPD_HUB"]),
                                  want_speed)
    assert float(want_speed.min()) > 1.0
    smax = cp.asnumpy(frame_fields["WSPD_HUB_MAX"])
    smin = cp.asnumpy(frame_fields["WSPD_HUB_MIN"])
    smean = cp.asnumpy(frame_fields["WSPD_HUB_MEAN"])
    assert (smax >= smean - 1e-4).all() and (smean >= smin - 1e-4).all()
    assert (smax > smin).any(), "the window saw no evolution at all"

    # The frame through the real writer carries the hub axis.
    frame = state_frame(s_on, include_diagnostic_pressure=True)
    path = tmp_path / "wrfout_d01_hub"
    with WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx,
                      dy=cfg.dy, field_schema=frame, engine="python") as w:
        w.write_frame("2025-01-01_00:00:30", frame)
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        np.testing.assert_array_equal(ds.variables["WSPD_HUB"][0], want_speed)

    interval_diagnostics.close_interval(
        s_on, valid_time=datetime(2025, 1, 1, 0, 0, 30),
        output_dir=tmp_path, grid_id=1)
    for _ in range(3):
        step(s_off, cfg)
        step(s_on, cfg)
    second = interval_diagnostics.history_fields(s_on)
    assert float(second["HUB_WINDOW_S"]) == 30.0
    # Inertness, the essential gate: six steps with and without the window.
    for name in ("u", "v", "w", "thp", "php", "mup", "p", "qv", "qc", "qr"):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(s_on, name)), cp.asnumpy(getattr(s_off, name)),
            err_msg=f"the hub-wind window moved model field {name}")
