"""The Fitch wind-farm drag: the device kernel against its float64 mirror,
and both against the closed forms Fitch et al. (2012) state.

The CPU tests hold the reference to the analytic budget of one column (the
thrust a uniform 8 m/s wind meets, the power and the TKE its coefficients
imply), to the rotor geometry, to the group sum of a mixed cell, and the
launcher's argument packing to the kernel's own signature through a fake
kernel.  The GPU tests hold ``fitch_column`` to the reference cell for cell.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import requires_gpu
from woof.core import windfarm_fitch as wf
from woof.verify.windfarm_fitch_ref import (WRF_POWER_DENSITY,
                                            np_fitch_column, np_turbine_area)

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "woof" / "core" / "kernels" / "windfarm_fitch.cu"

#: A generic 3 MW, 120 m rotor on a 100 m tower: cut-in 3, rated 12,
#: cut-out 25 m/s; Ct 0.8 to rated, then the constant-power taper.
SPEEDS = np.arange(3.0, 26.0, 1.0)
POWER_KW = np.minimum(3000.0 * ((SPEEDS - 3.0) / 9.0) ** 3, 3000.0)
THRUST = np.where(SPEEDS <= 12.0, 0.8, 0.8 * (12.0 / SPEEDS) ** 2)
HUB, DIAMETER, STANDING = 100.0, 120.0, 0.15
#: A generic 8 MW, 160 m rotor: cut-in 4, rated 11, cut-out 28 m/s.
SPEEDS_B = np.arange(4.0, 29.0, 1.0)
POWER_KW_B = np.minimum(8000.0 * ((SPEEDS_B - 4.0) / 7.0) ** 3, 8000.0)
THRUST_B = np.where(SPEEDS_B <= 11.0, 0.78, 0.78 * (11.0 / SPEEDS_B) ** 2.2)
#: The layout holds its tables as REAL, so closed forms read them back.
CT8 = float(np.float32(0.8))
P8_W = float(np.float32(POWER_KW[5])) * 1000.0
DX = 1000.0


def _layout(count, **kwargs):
    args = dict(hub_height_m=HUB, rotor_diameter_m=DIAMETER, speeds_ms=SPEEDS,
                thrust_coefficient=THRUST, power_kw=POWER_KW,
                standing_thrust=STANDING)
    args.update(kwargs)
    return wf.FitchLayout.single_type(count, **args)


def _curves():
    speeds = np.zeros((2, 30))
    thrust = np.zeros((2, 30))
    power = np.zeros((2, 30))
    speeds[0, :23], thrust[0, :23], power[0, :23] = SPEEDS, THRUST, POWER_KW
    speeds[1, :25], thrust[1, :25], power[1, :25] = (SPEEDS_B, THRUST_B,
                                                     POWER_KW_B)
    return dict(speeds_ms=speeds, thrust_coefficient=thrust, power_kw=power,
                nval=[23, 25], standing_thrust=[STANDING, 0.1])


def _groups(ny, nx, groups):
    """A layout from (cell, count, curve, hub, diameter) rows."""
    rows = np.array(groups, dtype=np.float64).reshape(-1, 5)
    return wf.FitchLayout(ny=ny, nx=nx, cell=rows[:, 0].astype(int),
                          count=rows[:, 1].astype(int),
                          curve=rows[:, 2].astype(int),
                          hub_height_m=rows[:, 3], rotor_diameter_m=rows[:, 4],
                          **_curves())


def _column_heights(nz=50, ny=1, nx=1, surface=250.0, first=20.0,
                    stretch=1.08):
    dz = first * stretch ** np.arange(nz)
    zw = surface + np.concatenate([[0.0], np.cumsum(dz)])
    return np.ascontiguousarray(
        np.broadcast_to(zw[:, None, None], (nz + 1, ny, nx)))


def _uniform(nz, ny, nx, speed=8.0, direction=0.0):
    u = np.full((nz, ny, nx), speed * math.cos(direction))
    v = np.full((nz, ny, nx), speed * math.sin(direction))
    return u, v


def _sheared(nz, zw, speed=9.0, direction=0.7):
    above = 0.5 * (zw[1:] + zw[:-1]) - zw[0]
    s = speed * (above / 100.0) ** 0.14
    turn = direction + 0.001 * above
    return s * np.cos(turn), s * np.sin(turn)


def _rotor_levels(zw, hub=HUB, dia=DIAMETER):
    above = zw[:, 0, 0] - zw[0, 0, 0]
    lo, hi = hub - dia / 2, hub + dia / 2
    return [k for k in range(len(above) - 1)
            if above[k + 1] > lo and above[k] < hi]


# A mixed cell: four 3 MW turbines low and two 8 MW turbines high, beside
# a uniform cell and an empty one.
MIXED = [(1, 4, 0, 100.0, 120.0), (1, 2, 1, 150.0, 160.0),
         (2, 3, 0, 100.0, 120.0)]


# ---------------------------------------------------------------------------
# The reference against closed forms (no device)
# ---------------------------------------------------------------------------

def test_uniform_8ms_column_removes_fitchs_thrust_over_the_rotor_layers():
    """At 8 m/s, Ct = 0.8 (a table node): sum_k du_k*rho*dx*dy*dz_k is
    -0.5*rho*Ct*A*|V|*u, the thrust over the cell's air mass integrated over
    the rotor layers; power and TKE follow from Cp = P/(0.5*1.23*V^3*A)."""
    nz = 50
    zw = _column_heights(nz)
    direction = 0.4
    u, v = _uniform(nz, 1, 1, 8.0, direction)
    rho = np.full((nz, 1, 1), 1.17)
    out = np_fitch_column(u, v, zw, _layout([[1]]), dx=DX, dt=1.0, rho=rho)
    group = out["groups"][0]
    area = math.pi / 4 * DIAMETER ** 2
    dz = np.diff(zw[:, 0, 0])
    assert group["speedhub"] == pytest.approx(8.0, rel=1e-14)
    assert group["ct"] == pytest.approx(CT8, rel=1e-14)
    # A uniform wind extracts the hub's power layer by layer already.
    assert group["ec"] == pytest.approx(1.0, rel=1e-12)
    removed_u = np.sum(out["du"][:, 0, 0] * rho[:, 0, 0] * DX * DX * dz)
    removed_v = np.sum(out["dv"][:, 0, 0] * rho[:, 0, 0] * DX * DX * dz)
    thrust = 0.5 * 1.17 * CT8 * area * 8.0
    assert removed_u == pytest.approx(-thrust * 8.0 * math.cos(direction),
                                      rel=1e-12)
    assert removed_v == pytest.approx(-thrust * 8.0 * math.sin(direction),
                                      rel=1e-12)
    assert out["thrust"][0, 0] == pytest.approx(thrust * 8.0, rel=1e-12)
    # Power is the table's at a node: P(8) = 3000*(5/9)^3 kW.
    assert out["power"][0, 0] == pytest.approx(P8_W, rel=1e-12)
    cp8 = P8_W / (0.5 * WRF_POWER_DENSITY * 8.0 ** 3 * area)
    assert group["cp"] == pytest.approx(cp8, rel=1e-12)
    ctke = 0.25 * (CT8 - cp8)
    assert group["ctke"] == pytest.approx(ctke, rel=1e-12)
    # QKE = 2*TKE gains |V|^3*A*C_TKE per unit air volume and time.
    qke_rate = np.sum(out["dqke"][:, 0, 0] * DX * DX * dz)
    assert qke_rate == pytest.approx(8.0 ** 3 * area * ctke, rel=1e-12)
    # The disk is split exactly among the rotor layers.
    assert np.sum(out["overlap"]) == pytest.approx(1.0, rel=1e-12)


def test_no_tendency_without_turbines_or_outside_the_rotor():
    nz, ny, nx = 50, 2, 3
    zw = _column_heights(nz, ny, nx)
    u, v = _uniform(nz, ny, nx, 9.5, 1.1)
    count = np.array([[0, 2, 0], [1, 0, 0]])
    out = np_fitch_column(u, v, zw, _layout(count), dx=DX, dt=5.0)
    rotor = _rotor_levels(zw)
    outside = [k for k in range(nz) if k not in rotor]
    for name in ("du", "dv", "dqke", "overlap"):
        field = out[name]
        assert np.all(field[:, count == 0] == 0.0), name
        assert np.all(field[outside] == 0.0), name
        assert np.all(field[rotor][:, count > 0] != 0.0), name
    assert np.all(out["power"][count == 0] == 0.0)
    # Two identical turbines are twice one.
    np.testing.assert_allclose(out["du"][:, 0, 1], 2 * out["du"][:, 1, 0],
                               rtol=1e-14)


def test_a_mixed_cell_is_the_sum_of_its_groups_each_over_its_own_rotor():
    nz, ny, nx = 50, 1, 3
    zw = _column_heights(nz, ny, nx)
    u, v = _sheared(nz, zw)
    mixed = np_fitch_column(u, v, zw, _groups(ny, nx, MIXED), dx=DX, dt=4.0)
    low = np_fitch_column(u, v, zw, _groups(ny, nx, MIXED[:1]), dx=DX, dt=4.0)
    high = np_fitch_column(u, v, zw, _groups(ny, nx, MIXED[1:2]), dx=DX,
                           dt=4.0)
    for name in ("du", "dv", "dqke"):
        np.testing.assert_allclose(mixed[name][:, 0, 1],
                                   low[name][:, 0, 1] + high[name][:, 0, 1],
                                   rtol=1e-13, atol=0)
    assert mixed["power"][0, 1] == pytest.approx(
        low["power"][0, 1] + high["power"][0, 1], rel=1e-13)
    # Each group reaches only its own levels: the high rotor reaches above
    # the low one's top.
    low_k, high_k = _rotor_levels(zw), _rotor_levels(zw, 150.0, 160.0)
    only_high = sorted(set(high_k) - set(low_k))
    assert only_high
    assert np.all(low["du"][only_high, 0, 1] == 0.0)
    assert np.all(mixed["du"][only_high, 0, 1] != 0.0)
    # The swept-area fractions weight each group by its count and disk.
    a_low, a_high = (math.pi / 4 * d ** 2 for d in (120.0, 160.0))
    swept = 4 * a_low + 2 * a_high
    np.testing.assert_allclose(
        mixed["overlap"][:, 0, 1],
        (4 * a_low * low["overlap"][:, 0, 1]
         + 2 * a_high * high["overlap"][:, 0, 1]) / swept, rtol=1e-12)
    assert np.sum(mixed["overlap"][:, 0, 1]) == pytest.approx(1.0)


def test_the_turbine_area_is_the_circular_segment():
    r = DIAMETER / 2
    assert np_turbine_area(0.0, DIAMETER, DIAMETER) == pytest.approx(math.pi * r * r)
    assert np_turbine_area(0.0, r, DIAMETER) == pytest.approx(math.pi * r * r / 2)
    assert np_turbine_area(r, DIAMETER, DIAMETER) == pytest.approx(math.pi * r * r / 2)
    # The segment below a chord at height h above the lower tip.
    h = 20.0
    d = r - h
    segment = r * r * math.acos(d / r) - d * math.sqrt(r * r - d * d)
    assert np_turbine_area(0.0, h, DIAMETER) == pytest.approx(segment, rel=1e-12)


@pytest.mark.parametrize("speed,ct,cp_zero", [
    (2.0, STANDING, True),      # below cut-in: the standing Ct, no power
    (25.5, THRUST[-1], True),   # above cut-out: the last Ct, no power
    (12.5, None, False),
])
def test_dragcof_arms(speed, ct, cp_zero):
    nz = 50
    zw = _column_heights(nz)
    u, v = _uniform(nz, 1, 1, speed)
    out = np_fitch_column(u, v, zw, _layout([[1]]), dx=DX, dt=1.0)
    group = out["groups"][0]
    if ct is not None:
        assert group["ct"] == pytest.approx(float(np.float32(ct)), rel=1e-14)
    else:
        assert group["ct"] == pytest.approx(
            0.5 * (float(np.float32(THRUST[9])) + float(np.float32(THRUST[10]))),
            rel=1e-12)
    assert (out["power"][0, 0] == 0.0) == cp_zero
    assert group["ctke"] >= 0.0


def test_the_layout_refuses_what_wrf_would_index_off_its_arrays():
    with pytest.raises(ValueError, match="below the ground"):
        _layout([[1]], hub_height_m=50.0)
    with pytest.raises(ValueError, match="ascend"):
        _layout([[1]], speeds_ms=SPEEDS[::-1])
    with pytest.raises(ValueError, match="cut-in"):
        _layout([[1]], speeds_ms=SPEEDS - 3.0)
    with pytest.raises(ValueError, match="non-negative"):
        _groups(1, 1, [(0, -1, 0, 100.0, 120.0)])
    with pytest.raises(ValueError, match="group cells"):
        _groups(1, 2, [(2, 1, 0, 100.0, 120.0)])
    with pytest.raises(ValueError, match="curves must lie"):
        _groups(1, 1, [(0, 1, 2, 100.0, 120.0)])
    with pytest.raises(ValueError, match="cell must be integers"):
        wf.FitchLayout(ny=1, nx=3, cell=[1.5], count=[1], curve=[0],
                       hub_height_m=[100.0], rotor_diameter_m=[120.0],
                       **_curves())
    with pytest.raises(ValueError, match="hub_height_m must be finite"):
        _groups(1, 1, [(0, 1, 0, np.nan, 120.0)])
    # A group without turbines may carry any placeholder geometry.
    _groups(1, 1, [(0, 0, 0, 10.0, 120.0)])


def test_the_layout_is_csr_by_cell_and_read_only():
    rows = [(2, 1, 0, 100.0, 120.0), (0, 3, 1, 150.0, 160.0),
            (2, 2, 1, 140.0, 160.0)]
    layout = _groups(1, 3, rows)
    np.testing.assert_array_equal(layout.cell_offsets, [0, 1, 1, 3])
    np.testing.assert_array_equal(layout.cell, [0, 2, 2])
    # A cell's groups keep the order they were given in.
    np.testing.assert_array_equal(layout.count, [3, 1, 2])
    assert list(layout.groups_in(0, 2)) == [1, 2]
    assert list(layout.groups_in(0, 1)) == []
    with pytest.raises(ValueError, match="read-only"):
        layout.count[0] = 0
    caller = np.array([1, 2], np.int32)
    held = wf.FitchLayout.single_type(
        np.array([[1, 2]]), hub_height_m=HUB, rotor_diameter_m=DIAMETER,
        speeds_ms=SPEEDS, thrust_coefficient=THRUST, power_kw=POWER_KW,
        standing_thrust=STANDING)
    caller[0] = 7   # the layout keeps its own copy
    np.testing.assert_array_equal(held.count, [1, 2])


# ---------------------------------------------------------------------------
# The launcher through a fake kernel (no device)
# ---------------------------------------------------------------------------

def _kernel_parameters():
    source = KERNEL.read_text(encoding="utf-8")
    match = re.search(r"void fitch_column\((.*?)\)\s*\{", source, re.S)
    return [re.split(r"[\s*]+", p.strip())[-1]
            for p in match.group(1).split(",")]


def _fake_cupy():
    shim = SimpleNamespace(**{name: getattr(np, name) for name in (
        "float32", "int32", "ascontiguousarray", "asarray", "empty", "zeros")})
    shim.cuda = SimpleNamespace(Device=lambda *a: SimpleNamespace(id=0))
    shim.ndarray = np.ndarray
    return shim


def test_the_launcher_packs_every_argument_the_kernel_declares(monkeypatch):
    """A fake kernel reads the launch by the .cu's own parameter names and
    runs the float64 mirror: the launcher must add into the caller's arrays
    and pass the groups, cell area, step and factor where the kernel reads
    them."""
    names = _kernel_parameters()
    calls = []

    def fake(name):
        assert name == "fitch_column"

        def launch(grid, block, args):
            assert len(args) == len(names), (len(args), names)
            a = dict(zip(names, args))
            ny, nx = int(a["ny"]), int(a["nx"])
            assert grid[0] * block[0] >= ny * nx
            assert a["offsets"].shape == (ny * nx + 1,)
            assert a["turbws"].shape[1] == int(a["maxvals"])
            lay = SimpleNamespace(
                cell_offsets=a["offsets"], count=a["g_count"],
                curve=a["g_curve"], hub_height_m=a["g_hub"],
                rotor_diameter_m=a["g_dia"], nval=a["nval"],
                speeds_ms=a["turbws"], thrust_coefficient=a["turbtc"],
                power_kw=a["turbpw"], standing_thrust=a["stc"])
            area = 1.0 / float(a["wfdensity"])
            ref = np_fitch_column(
                a["u"], a["v"], a["z_w"], lay, dx=area, dy=1.0,
                dt=float(a["dt"]), tke_factor=float(a["tke_factor"]),
                rho=a["rho"] if isinstance(a["rho"], np.ndarray) else None)
            a["du"] += ref["du"].astype(np.float32)
            a["dv"] += ref["dv"].astype(np.float32)
            if isinstance(a["qke"], np.ndarray):
                a["qke"] += ref["dqke"].astype(np.float32)
            a["power"][...] = ref["power"]
            if isinstance(a["thrust"], np.ndarray):
                a["thrust"][...] = ref["thrust"]
            if isinstance(a["overlap"], np.ndarray):
                a["overlap"][...] = ref["overlap"]
            calls.append(a)
        return launch

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy())
    monkeypatch.setattr(wf, "_kernel", fake)
    monkeypatch.setattr(wf.FitchLayout, "device_arrays", lambda self: {
        name: getattr(self, name) for name in wf._DEVICE_FIELDS})
    nz, ny, nx = 50, 1, 3
    zw = _column_heights(nz, ny, nx).astype(np.float32)
    u, v = _sheared(nz, zw)
    layout = _groups(ny, nx, MIXED)
    du = np.full((nz, ny, nx), 1e-4, np.float32)
    dv = np.zeros((nz, ny, nx), np.float32)
    qke = np.full((nz, ny, nx), 0.5, np.float32)
    rho = np.full((nz, ny, nx), 1.2, np.float32)
    diagnostics = {}
    power = wf.launch_fitch(u, v, zw, layout, du, dv, dx=DX, qke=qke, dt=6.0,
                            tke_factor=0.5, rho=rho, diagnostics=diagnostics)
    assert len(calls) == 1
    a = calls[0]
    assert float(a["wfdensity"]) == pytest.approx(1.0 / DX ** 2, rel=1e-7)
    assert float(a["dt"]) == 6.0 and float(a["tke_factor"]) == 0.5
    assert a["piconst"] == np.float32(math.pi)
    ref = np_fitch_column(u, v, zw, layout, dx=DX, dt=6.0, tke_factor=0.5,
                          rho=rho)
    np.testing.assert_allclose(du - np.float32(1e-4), ref["du"], atol=1e-8)
    np.testing.assert_allclose(qke - 0.5, ref["dqke"], atol=1e-6)
    np.testing.assert_allclose(power, ref["power"], rtol=1e-6)
    assert set(diagnostics) == {"overlap", "thrust"}
    with pytest.raises(ValueError, match="dt is required"):
        wf.launch_fitch(u, v, zw, layout, du, dv, dx=DX, qke=qke)
    with pytest.raises(ValueError, match="diagnostics dict"):
        wf.launch_fitch(u, v, zw, layout, du, dv, dx=DX, rho=rho)
    with pytest.raises(TypeError, match="C-contiguous float32"):
        wf.launch_fitch(u, v, zw, layout, du.astype(np.float64), dv, dx=DX)
    with pytest.raises(ValueError, match=r"du must be \(nz, ny, nx\)"):
        wf.launch_fitch(u, v, zw, layout, du[0], dv, dx=DX)


def test_the_module_is_the_direct_non_ftz_route():
    assert wf.MODULE_OPTIONS == ("-std=c++17", "-fmad=false", "--ftz=false")
    source = wf.module_source()
    assert "fitch_column" in source and "glibc_asinf" in source
    assert "gfk_pow" in source


# ---------------------------------------------------------------------------
# The device kernel against the reference
# ---------------------------------------------------------------------------

def _grid_case(seed=7):
    """Sheared, veering columns over uneven terrain, mixed cells."""
    rng = np.random.default_rng(seed)
    nz, ny, nx = 50, 5, 7
    zw = np.empty((nz + 1, ny, nx))
    for j in range(ny):
        for i in range(nx):
            zw[:, j, i] = _column_heights(
                nz, surface=rng.uniform(0.0, 900.0),
                first=rng.uniform(12.0, 40.0),
                stretch=rng.uniform(1.04, 1.1))[:, 0, 0]
    above = 0.5 * (zw[1:] + zw[:-1]) - zw[0]
    speed = rng.uniform(1.0, 14.0, (1, ny, nx)) * (above / 100.0) ** 0.2
    turn = rng.uniform(0, 2 * np.pi, (1, ny, nx)) + 0.002 * above
    u, v = speed * np.cos(turn), speed * np.sin(turn)
    rows = []
    for c in range(1, ny * nx):
        for _ in range(rng.integers(0, 4)):
            curve = int(rng.integers(0, 2))
            hub = (100.0 if curve == 0 else 140.0) + rng.uniform(-5, 5)
            rows.append((c, int(rng.integers(0, 4)), curve, hub,
                         120.0 if curve == 0 else 160.0))
    layout = _groups(ny, nx, rows)
    assert (np.diff(layout.cell_offsets) > 1).any(), "no mixed cell drawn"
    rho = 1.225 - 1e-4 * above
    return u, v, zw, rho, layout


def _launch(cp, u, v, zw, rho, layout, *, dt=12.0, dx=DX, preload=0.0):
    shape = u.shape
    du = cp.full(shape, preload, dtype=cp.float32)
    dv = cp.full(shape, -preload, dtype=cp.float32)
    qke = cp.full(shape, preload, dtype=cp.float32)
    diagnostics = {}
    power = wf.launch_fitch(u, v, zw, layout, du, dv, dx=dx, qke=qke, dt=dt,
                            rho=rho, diagnostics=diagnostics)
    out = {"du": du - preload, "dv": dv + preload, "dqke": qke - preload,
           "power": power, **diagnostics}
    return {name: cp.asnumpy(value) for name, value in out.items()}


def _assert_matches(got, ref, rtol=2e-5):
    for name in ("du", "dv", "dqke", "overlap", "power", "thrust"):
        scale = np.abs(ref[name]).max()
        assert scale > 0, name
        err = np.abs(got[name] - ref[name]).max()
        assert err <= rtol * scale, (name, err, scale)
        # Zeros are exact zeros on the device, not rounding residue.
        assert np.all(got[name][ref[name] == 0.0] == 0.0), name


@requires_gpu
@pytest.mark.gpu
def test_device_kernel_matches_the_float64_reference_cell_for_cell():
    import cupy as cp

    u, v, zw, rho, layout = _grid_case()
    got = _launch(cp, u, v, zw, rho, layout)
    ref = np_fitch_column(u, v, zw, layout, dx=DX, dt=12.0, rho=rho)
    _assert_matches(got, ref)


@requires_gpu
@pytest.mark.gpu
def test_device_kernel_and_binary32_cpu_mirror_agree_on_every_word():
    """Max ULP 0 on every output, the diagnostics included, over sheared
    mixed cells on uneven terrain with the PBL's tendencies preloaded."""
    import cupy as cp

    from woof.verify.urban_oracle import ulp_table
    from woof.verify.windfarm_fitch_ref import f32_fitch_column

    u, v, zw, rho, layout = _grid_case(seed=13)
    u, v, zw, rho = (np.asarray(a, np.float32) for a in (u, v, zw, rho))
    rng = np.random.default_rng(1)
    du0, dv0 = (rng.normal(0, 1e-3, u.shape).astype(np.float32)
                for _ in range(2))
    qke0 = rng.uniform(0.1, 2.0, u.shape).astype(np.float32)
    du, dv, qke = (cp.asarray(a) for a in (du0, dv0, qke0))
    diagnostics = {}
    power = wf.launch_fitch(u, v, zw, layout, du, dv, dx=DX, qke=qke, dt=9.0,
                            rho=rho, diagnostics=diagnostics)
    cpu = dict(du=du0.copy(), dv=dv0.copy(), qke=qke0.copy())
    cpu["power"], cpu["thrust"], cpu["overlap"] = f32_fitch_column(
        u, v, zw, layout, cpu["du"], cpu["dv"], cpu["qke"], dx=DX, dt=9.0,
        rho=rho)
    gpu = dict(du=du, dv=dv, qke=qke, power=power, **diagnostics)
    for name, value in cpu.items():
        table = ulp_table(cp.asnumpy(gpu[name]), value)
        assert table["max_ulp"] == 0, (name, table)


@requires_gpu
@pytest.mark.gpu
def test_device_mixed_cell_is_the_sum_of_its_groups():
    import cupy as cp

    nz, ny, nx = 50, 1, 3
    zw = _column_heights(nz, ny, nx)
    u, v = _sheared(nz, zw)
    rho = np.full((nz, ny, nx), 1.2)
    mixed = _launch(cp, u, v, zw, rho, _groups(ny, nx, MIXED))
    parts = [_launch(cp, u, v, zw, rho, _groups(ny, nx, [row]))
             for row in MIXED[:2]]
    for name in ("du", "dv", "dqke"):
        total = parts[0][name][:, 0, 1] + parts[1][name][:, 0, 1]
        np.testing.assert_allclose(mixed[name][:, 0, 1], total, rtol=2e-6,
                                   atol=1e-6 * np.abs(total).max())
    for name in ("power", "thrust"):
        assert mixed[name][0, 1] == pytest.approx(
            parts[0][name][0, 1] + parts[1][name][0, 1], rel=2e-6)
    ref = np_fitch_column(u, v, zw, _groups(ny, nx, MIXED), dx=DX, dt=12.0,
                          rho=rho)
    _assert_matches(mixed, ref)


@requires_gpu
@pytest.mark.gpu
def test_device_kernel_adds_into_the_callers_tendencies():
    import cupy as cp

    u, v, zw, rho, layout = _grid_case(seed=3)
    base = _launch(cp, u, v, zw, rho, layout)
    added = _launch(cp, u, v, zw, rho, layout, preload=0.125)
    for name in ("du", "dv", "dqke"):
        np.testing.assert_allclose(added[name], base[name], atol=1e-7)
    np.testing.assert_array_equal(added["power"], base["power"])


@requires_gpu
@pytest.mark.gpu
def test_device_single_column_at_8ms_removes_fitchs_thrust():
    import cupy as cp

    nz = 50
    zw = _column_heights(nz)
    u, v = _uniform(nz, 1, 1, 8.0, 0.4)
    rho = np.full((nz, 1, 1), 1.17)
    got = _launch(cp, u, v, zw, rho, _layout([[1]]), dt=1.0)
    dz = np.diff(zw[:, 0, 0])
    area = math.pi / 4 * DIAMETER ** 2
    removed = np.hypot(np.sum(got["du"][:, 0, 0] * dz),
                       np.sum(got["dv"][:, 0, 0] * dz)) * 1.17 * DX * DX
    assert removed == pytest.approx(0.5 * 1.17 * CT8 * area * 64.0, rel=1e-5)
    assert got["thrust"][0, 0] == pytest.approx(removed, rel=1e-5)
    assert got["power"][0, 0] == pytest.approx(P8_W, rel=1e-5)
    rotor = _rotor_levels(zw)
    outside = [k for k in range(nz) if k not in rotor]
    for name in ("du", "dv", "dqke", "overlap"):
        assert np.all(got[name][outside] == 0.0), name
        assert np.all(got[name][rotor] != 0.0), name
    assert np.sum(got["overlap"]) == pytest.approx(1.0, abs=1e-6)


@requires_gpu
@pytest.mark.gpu
def test_a_rotor_above_the_column_top_is_skipped_as_wrf_skips_it():
    import cupy as cp

    nz = 6
    zw = _column_heights(nz, first=15.0, stretch=1.0)   # 90 m deep
    u, v = _uniform(nz, 1, 1, 8.0)
    got = _launch(cp, u, v, zw, np.ones((nz, 1, 1)), _layout([[2]]))
    ref = np_fitch_column(u, v, zw, _layout([[2]]), dx=DX, dt=12.0)
    for name in ("du", "dv", "dqke", "overlap", "power"):
        assert np.all(got[name] == 0.0) and np.all(ref[name] == 0.0), name


@requires_gpu
@pytest.mark.gpu
def test_the_rotor_overlap_helper_is_the_kernels_geometry():
    import cupy as cp

    u, v, zw, rho, layout = _grid_case(seed=11)
    got = cp.asnumpy(wf.rotor_overlap(zw, layout))
    ref = np_fitch_column(u, v, zw, layout, dx=DX)["overlap"]
    np.testing.assert_allclose(got, ref, atol=2e-6)
    on = np.zeros(layout.shape, bool).ravel()
    on[layout.cell[layout.count > 0]] = True
    np.testing.assert_allclose(got.sum(axis=0).ravel()[on], 1.0, atol=2e-6)


@requires_gpu
@pytest.mark.gpu
def test_the_layout_is_copied_to_the_device_once():
    import cupy as cp

    u, v, zw, rho, layout = _grid_case(seed=5)
    _launch(cp, u, v, zw, rho, layout)
    first = layout.device_arrays()
    _launch(cp, u, v, zw, rho, layout)
    second = layout.device_arrays()
    assert all(first[name] is second[name] for name in first)
    assert all(isinstance(value, cp.ndarray) for value in first.values())


@requires_gpu
@pytest.mark.gpu
def test_the_composed_unit_compiles_within_the_priced_frame():
    import cupy as cp

    from woof.core.preflight import CHAINED_TRANSLATION_UNIT_FRAMES

    module = wf._module(cp.cuda.Device().id)
    frame = int(cp.cuda.driver.funcGetAttribute(
        cp.cuda.driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        module.get_function("fitch_column").ptr))
    priced = CHAINED_TRANSLATION_UNIT_FRAMES["windfarm_fitch_composed"]
    assert frame <= priced.max_local_size_bytes


def test_the_cpu_libm_twins_are_the_host_glibc_words():
    """The binary32 mirror's powf (woof.core.noahmp_libm) and asinf (the
    glibc_asinf transcription) are host-independent; on a glibc whose asinf
    is CORE-MATH's (2.41 and later) they are also the C library's words."""
    import ctypes
    import ctypes.util
    import platform

    from woof.verify.windfarm_fitch_ref import _asinf, _powf

    name, version = platform.libc_ver()
    if name != "glibc" or tuple(int(p) for p in version.split(".")[:2]) < (2, 41):
        pytest.skip(f"host libm is {name} {version}, not a CORE-MATH glibc")
    libm = ctypes.CDLL(ctypes.util.find_library("m"))
    libm.asinf.restype = ctypes.c_float
    libm.asinf.argtypes = [ctypes.c_float]
    libm.powf.restype = ctypes.c_float
    libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
    rng = np.random.default_rng(2)
    xs = rng.integers(0, 0x3F800001, 4000, dtype=np.uint32).view(np.float32)
    xs = np.concatenate([xs, -xs, np.float32([0.0, 1.0, 0.5])])
    for x in xs:
        assert _asinf(x).view(np.uint32) == np.float32(
            libm.asinf(float(x))).view(np.uint32), x
    for x in rng.uniform(0.0, 40.0, 2000).astype(np.float32):
        for e in (2.0, 3.0):
            assert _powf(x, e).view(np.uint32) == np.float32(
                libm.powf(float(x), e)).view(np.uint32), (x, e)
