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


# ---------------------------------------------------------------------------
# the hub-height scalars
# ---------------------------------------------------------------------------

_SCALAR_INPUTS = ("php", "phb", "ht", "thp", "thb", "p", "qv", "qc", "qr",
                  "nc")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("base_3d", [False, True])
@pytest.mark.parametrize("droplets", [False, True])
def test_gpu_scalar_kernel_matches_the_reference(base_3d, droplets):
    """P/QV/QC/QR/QNC bit-identical to the mirror (no FMA, the same
    bracketing); T and RHO pass through powf and agree to rtol 2e-6."""
    import cupy as cp

    from test_hub_wind_diag import _thermo

    rng = np.random.default_rng(11)
    nz, ny, nx = 30, 40, 40
    host = _column_state(nz=nz, ny=ny, nx=nx, dz=35.0, hgt=0.0,
                         phb_1d=not base_3d)
    host.ht[...] = rng.uniform(0.0, 800.0, (ny, nx)).astype(np.float32)
    host.php[...] = (host.ht[None] * G + rng.uniform(
        -30.0, 30.0, (nz + 1, ny, nx)) * G).astype(np.float32)
    host = _thermo(host, thb_1d=not base_3d,
                   nc=(lambda z: 1e8 + 0.0 * z) if droplets else None)
    host.thp += rng.normal(0.0, 1.5, host.thp.shape).astype(np.float32)
    host.p *= rng.uniform(0.99, 1.01, host.p.shape).astype(np.float32)
    host.qv *= rng.uniform(0.5, 1.5, host.qv.shape).astype(np.float32)
    host.qc[...] = rng.uniform(0.0, 5e-4, host.qc.shape).astype(np.float32)
    host.qr[...] = rng.uniform(0.0, 2e-4, host.qr.shape).astype(np.float32)
    if droplets:
        host.nc[...] = rng.uniform(1e7, 3e8, host.nc.shape).astype(np.float32)
    heights = (5.0, 20.0, 40.0, 80.0, 100.0, 120.0, 150.0, 200.0, 2000.0)
    device = SimpleNamespace(
        **{name: (None if getattr(host, name) is None
                  else cp.asarray(getattr(host, name)))
           for name in _SCALAR_INPUTS})
    request = HubWindRequest(heights_m=heights)
    dev_diag = hw.HubWindDiagnostic(request, ny=ny, nx=nx, xp=cp,
                                    device_id=int(device.php.device.id),
                                    scalars=True)
    got = dev_diag._scalar_fields(device)
    want = hw.hub_scalars_np(host.php, host.phb, host.ht, host.thp, host.thb,
                             host.p, heights, qv=host.qv, qc=host.qc,
                             qr=host.qr, nc=host.nc)
    assert tuple(got) == tuple(want)
    for name in want:
        if name in ("T_HUB", "RHO_HUB"):
            np.testing.assert_allclose(cp.asnumpy(got[name]), want[name],
                                       rtol=2e-6, atol=0, err_msg=name)
        else:
            np.testing.assert_array_equal(cp.asnumpy(got[name]), want[name],
                                          err_msg=name)


@pytest.mark.gpu
@requires_gpu
def test_gpu_dycore_frame_scalars_are_the_post_microphysics_state(tmp_path):
    """Real moist dycore steps (30 model seconds): the frame's scalars are
    the mirror evaluated on the state the frame is written from, the
    window moves no model byte, and the windfarm frame carries no volume."""
    import cupy as cp
    from woof.core.dycore import step
    from woof.io.wrfout import state_frame
    from test_mp_spec_zone_ring import _square_moist_state

    def make():
        state, cfg = _square_moist_state(ny=12, nx=14, nz=30, dt=10.0,
                                         open_x=True, open_y=True)
        z = state.height_half()
        shear = (6.0 + 0.004 * z).astype(np.float32)
        state.u[...] = cp.asarray(np.broadcast_to(shear[:, None, None],
                                                  state.u.shape))
        return state, cfg

    s_off, cfg = make()
    s_on, _ = make()
    selection = HistorySelection.from_mapping(
        {"preset": "windfarm", "hub_heights_m": [40, 100, 150, 200]})
    diag = hw.attach_hub_wind(s_on, selection, nx=cfg.nx, ny=cfg.ny,
                              dx=cfg.dx, dy=cfg.dy, grid_id=1)
    assert diag.scalars
    for _ in range(3):
        step(s_off, cfg)
        step(s_on, cfg)
    fields = interval_diagnostics.history_fields(s_on)
    host = {name: (None if getattr(s_on, name, None) is None
                   else cp.asnumpy(getattr(s_on, name)))
            for name in _SCALAR_INPUTS}
    want = hw.hub_scalars_np(host["php"], host["phb"], host["ht"],
                             host["thp"], host["thb"], host["p"],
                             diag.heights_m, qv=host["qv"], qc=host["qc"],
                             qr=host["qr"], nc=host["nc"])
    for name, value in want.items():
        got = cp.asnumpy(fields[name])
        if name in ("T_HUB", "RHO_HUB"):
            np.testing.assert_allclose(got, value, rtol=2e-6, err_msg=name)
        else:
            np.testing.assert_array_equal(got, value, err_msg=name)
    rho = cp.asnumpy(fields["RHO_HUB"])
    assert 0.5 < float(rho.min()) and float(rho.max()) < 1.5
    frame = selection.apply(state_frame(s_on,
                                        include_diagnostic_pressure=True))[0]
    assert {"RHO_HUB", "T_HUB", "WSPD_HUB", "HGT"} <= set(frame)
    assert not {"T", "PHB", "U", "V", "W", "PH", "P", "PB",
                "QVAPOR"} & set(frame)
    for name in ("u", "v", "w", "thp", "php", "mup", "p", "qv", "qc", "qr"):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(s_on, name)), cp.asnumpy(getattr(s_off, name)),
            err_msg=f"the hub scalars moved model field {name}")
