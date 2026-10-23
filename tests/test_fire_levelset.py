"""WRF v4.7.1's fire model (woof.fire.levelset, woof.fire.heat): physics.

``tests/test_fire_levelset_wrf471_parity.py`` holds the port to WRF word for
word.  This file holds WRF's scheme to the physics it is for, on fronts with
closed forms:

* a circle under an isotropic rate of spread grows at that rate (with WRF's
  artificial viscosity switched off; with it on, curvature slows a small
  fire by a bounded amount);
* under WRF's wind form (``fire_advection = 1``: the wind part scales with
  ``max(cos, 0)`` between wind and normal) a point grows into the hull of
  two discs, reaching ``head * t`` downwind and ``back * t`` up- and
  cross-wind;
* energy: the integrated heat release equals the fuel burnt times its heat
  content, and the aggregation onto the atmospheric grid keeps the area
  integral;
* the fuel fraction never rises, no fire crosses a rate-of-spread-zero
  break, and ignitions light where and when they say.

The device runs the same steps, word for word with the host; its time loop
never reads the device back.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import requires_gpu
from woof.fire import heat
from woof.fire.levelset import (FireMesh, FireOptions, FireParams, FireState,
                                Ignition, _expf, _powf, advance, burned_area,
                                fuel_time_from_weight, initial_state, step,
                                substeps)

#: WRF's artificial viscosity off, for fronts with a closed form.
INVISCID = FireOptions(fire_viscosity=0.0, fire_viscosity_bg=0.0)


def _mesh(n_atm=8, dx_atm=50.0, sr=10):
    return FireMesh(n_atm, n_atm, dx_atm, dx_atm, sr, sr)


def _fast(x, y, radius=20.0, **kw):
    """A point ignition that reaches its radius within the first step."""
    return Ignition(start=(x, y), radius=radius, ros=1000.0, **kw)


def _crossing(values, coords):
    """The last zero crossing (burning -> not) of a 1-D level-set profile."""
    values = np.asarray(values, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    j = np.flatnonzero(values <= 0)[-1]
    return coords[j] + (coords[j + 1] - coords[j]) * values[j] / (
        values[j] - values[j + 1])


def _run(mesh, params, dt, steps, ignitions, options=FireOptions()):
    state = initial_state(mesh)
    outputs = []
    for _ in range(steps):
        state, out = step(state, mesh, params, dt, ignitions=ignitions,
                          options=options)
        outputs.append(out)
    return state, outputs


# ---------------------------------------------------------------------------
# the front
# ---------------------------------------------------------------------------

def test_circle_grows_at_the_isotropic_rate():
    mesh = _mesh()                                   # 400 m at 5 m
    rate, r0, dt = 0.3, 20.0, 2.0
    params = FireParams.isotropic(mesh, rate, fgip=0.5)
    state = initial_state(mesh)
    x, y = mesh.coordinates()
    radii = {}
    for k in range(1, 101):
        state, _ = step(state, mesh, params, dt, ignitions=[_fast(200, 200)],
                        options=INVISCID)
        if k in (20, 100):
            mid = mesh.ny // 2
            radii[k] = (_crossing(state.lfn[mid], x[0]) - 200.0,
                        200.0 - _crossing(state.lfn[mid][::-1], x[0][::-1]),
                        _crossing(state.lfn[:, mid], y[:, 0]) - 200.0,
                        math.sqrt(burned_area(state, mesh) / math.pi))
    # WRF ignites in pass 5, after pass 4 has moved the front: the fire
    # spreads from the end of the first step.
    for k in (20, 100):
        expected = r0 + rate * (k - 1) * dt
        for radius in radii[k]:
            assert radius == pytest.approx(expected, abs=float(mesh.dx))
    growth = [(b - a) / (80 * dt) for a, b in zip(radii[20], radii[100])]
    for g in growth:
        assert g == pytest.approx(rate, rel=0.03)


def test_wrf_viscosity_slows_a_curved_front_by_a_bounded_amount():
    mesh = _mesh()
    params = FireParams.isotropic(mesh, 0.3, fgip=0.5)
    ign = [_fast(200, 200)]
    plain, _ = _run(mesh, params, 2.0, 100, ign, INVISCID)
    viscous, _ = _run(mesh, params, 2.0, 100, ign)        # WRF's 0.4
    a, b = burned_area(plain, mesh), burned_area(viscous, mesh)
    assert 0.85 * a < b < a


def test_wind_grows_the_hull_of_two_discs():
    mesh = _mesh()
    back, head, r0, dt, steps = 0.05, 0.5, 15.0, 1.0, 150
    params = FireParams.wind_driven(mesh, back=back, head=head, wind_u=6.0,
                                    wind_v=0.0, fgip=0.5)
    state, _ = _run(mesh, params, dt, steps, [_fast(100, 200, r0)],
                    INVISCID)
    x, y = mesh.coordinates()
    t = (steps - 1) * dt
    burning = state.burning
    cell = float(mesh.dx)
    assert x[burning].max() - 100 == pytest.approx(head * t + r0, abs=2 * cell)
    assert 100 - x[burning].min() == pytest.approx(back * t + r0, abs=cell)
    assert y[burning].max() - 200 == pytest.approx(back * t + r0, abs=cell)
    disc = back * t + r0
    hull = math.pi * disc ** 2 + 2 * disc * (head - back) * t
    assert burned_area(state, mesh) == pytest.approx(hull, rel=0.08)
    length = x[burning].max() - x[burning].min()
    breadth = y[burning].max() - y[burning].min()
    assert length > 2 * breadth


def test_no_fire_crosses_a_zero_rate_of_spread_break():
    mesh = _mesh()
    x, _ = mesh.coordinates()
    r0 = np.where(x < 250.0, np.float32(0.3) / np.float32(0.00508), 0.0)
    params = FireParams.build(mesh, r_0=r0, fgip=0.5)
    state, outputs = _run(mesh, params, 2.0, 300, [_fast(200, 200)])
    blocked = x >= 250.0
    assert not np.any(state.burning & blocked)
    assert np.all(state.burning[:, (x[0] > 150) & (x[0] < 245)].any(axis=0))
    # fuel_left's cells reach half a node into the break, no further.
    beyond = x >= 250.0 + float(mesh.dx)
    assert np.all(state.fuel_frac[beyond] == 1)
    assert np.all(outputs[-1].ros[blocked] == 0)


def test_ignitions_light_where_and_when_they_say():
    mesh = _mesh()
    params = FireParams.isotropic(mesh, 0.0, fgip=0.5)     # no spread
    late = Ignition(start=(100.0, 100.0), radius=20.0, ros=1000.0,
                    start_time=30.0)
    state, _ = _run(mesh, params, 2.0, 14, [late])          # to t = 28 s
    assert not np.any(state.burning)
    state, _ = _run(mesh, params, 2.0, 16, [late])          # to t = 32 s
    lit = state.burning
    assert np.any(lit)
    assert np.all((state.tign[lit] >= 30.0) & (state.tign[lit] <= 32.0))
    x, y = mesh.coordinates()
    assert np.all(np.hypot(x[lit] - 100, y[lit] - 100) <= 20.0)
    # A line walked from (60, 300) to (340, 300) over 20 s to 60 s.
    line = Ignition(start=(60.0, 300.0), end=(340.0, 300.0), start_time=20.0,
                    end_time=60.0, radius=6.0, ros=1000.0)
    state, _ = _run(mesh, params, 2.0, 20, [line])          # to t = 40 s
    on = np.abs(y - 300.0) < 3.0
    walked = x[on & state.burning]
    assert walked.min() == pytest.approx(60.0, abs=6.0)
    assert walked.max() == pytest.approx(200.0, abs=8.0)    # half way
    state, _ = _run(mesh, params, 2.0, 35, [line])          # to t = 70 s
    lit = on & state.burning & (x > 66.0) & (x < 334.0)
    assert np.all(lit[on & (x > 66.0) & (x < 334.0)])
    order = np.argsort(x[lit])
    arrival = state.tign[lit][order]
    assert np.all(np.diff(arrival) >= 0)                     # walked in order
    speed = (340.0 - 60.0) / 40.0
    np.testing.assert_allclose(arrival, 20.0 + (x[lit][order] - 60.0) / speed,
                               atol=2.5)


# ---------------------------------------------------------------------------
# fuel and heat
# ---------------------------------------------------------------------------

def test_fuel_fraction_never_rises_and_decays_with_the_fuel_weight():
    mesh = _mesh()
    finals = []
    for weight in (7.0, 180.0, 900.0):
        params = FireParams.isotropic(mesh, 0.3, fgip=0.5,
                                      fuel_time=fuel_time_from_weight(weight))
        state = initial_state(mesh)
        previous = state.fuel_frac
        for _ in range(40):
            state, _ = step(state, mesh, params, 2.0,
                            ignitions=[_fast(200, 200)])
            assert np.all(state.fuel_frac <= previous)
            assert np.all((state.fuel_frac > 0) & (state.fuel_frac <= 1))
            previous = state.fuel_frac
        finals.append(float(state.fuel_frac.min()))
    assert finals[0] < finals[1] < finals[2] < 1.0
    assert fuel_time_from_weight(7.0) == np.float32(7.0) / np.float32(0.85)


def test_heat_release_is_the_fuel_burnt_times_its_heat_content():
    mesh = _mesh()
    x, y = mesh.coordinates()
    load = np.where(x < 200.0, 0.6, 1.1)
    moisture = 0.06 + 0.04 * (y > 200.0)
    params = FireParams.isotropic(mesh, 0.3, fgip=load, fmc_g=moisture,
                                  fuel_time=12.0)
    state, outputs = _run(mesh, params, 2.0, 60, [_fast(200, 200)])
    area, dt = mesh.cell_area, 2.0
    sensible = sum(float(o.fgrnhfx.astype(np.float64).sum()) for o in
                   outputs) * dt * area
    latent = sum(float(o.fgrnqfx.astype(np.float64).sum()) for o in
                 outputs) * dt * area
    burnt = 1.0 - state.fuel_frac.astype(np.float64)
    m = params.fmc_g.astype(np.float64)
    dry = burnt * params.fgip.astype(np.float64) / (1.0 + m)
    assert float(dry.sum()) > 0
    assert sensible == pytest.approx(float(dry.sum()) * area
                                     * float(heat.CMBCNST), rel=1e-4)
    water = burnt * params.fgip.astype(np.float64) * (m + 0.56) / (1.0 + m)
    assert latent == pytest.approx(float(water.sum()) * area
                                   * float(heat.XLV), rel=1e-4)
    # sum_2d_cells keeps each step's area integral, and canopy is zero.
    atm_area = mesh.dx_atm * mesh.dy_atm
    for out in outputs[::10]:
        for coarse, fine in (("GRNHFX", "fgrnhfx"), ("GRNQFX", "fgrnqfx")):
            total = float(getattr(out, coarse).astype(np.float64).sum())
            want = float(getattr(out, fine).astype(np.float64).sum())
            assert total * atm_area == pytest.approx(want * area, rel=1e-5,
                                                     abs=1e-3)
        assert not np.any(out.CANHFX) and not np.any(out.CANQFX)
    np.testing.assert_array_equal(outputs[-1].AVG_FUEL_FRAC,
                                  heat.aggregate(state.fuel_frac, mesh))


def test_heat_fluxes_follow_wrfs_formula():
    burnt, fgip, m, dt = (np.array([0.25], np.float32), np.array([0.8], np.float32),
                          np.array([0.1], np.float32), np.float32(4.0))
    sensible, latent = heat.heat_fluxes_cpu(fgip, burnt, m, dt)
    bmst = 0.1 / 1.1
    assert sensible[0] == pytest.approx(0.8 * 0.25 / 4 * (1 - bmst) * 17.433e6,
                                        rel=1e-6)
    assert latent[0] == pytest.approx((bmst + (1 - bmst) * 0.56) * 0.8 * 0.25
                                      / 4 * 2.5e6, rel=1e-6)


def test_aggregate_is_the_cell_mean_and_refuses_other_shapes():
    mesh = FireMesh(3, 2, 90.0, 60.0, 3, 2)
    field = np.arange(mesh.nx * mesh.ny, dtype=np.float32).reshape(mesh.shape)
    coarse = heat.aggregate(field, mesh)
    np.testing.assert_allclose(
        coarse, field.reshape(2, 2, 3, 3).mean(axis=(1, 3)), rtol=1e-6)
    with pytest.raises(ValueError, match="fire mesh"):
        heat.aggregate(np.zeros((3, 3), np.float32), mesh)


# ---------------------------------------------------------------------------
# the interval, the libm, the contracts
# ---------------------------------------------------------------------------

def test_advance_splits_an_interval_in_cfl_limited_steps():
    mesh = _mesh()
    params = FireParams.isotropic(mesh, 0.3, fgip=0.5)
    assert substeps(mesh, 0.3, 60.0) == math.ceil(60 * 0.3 * 0.4 / 0.5)
    assert substeps(mesh, 0.0, 60.0) == 1
    assert params.ros_bound() == pytest.approx(0.3, rel=1e-5)
    ign = [_fast(200, 200)]
    whole, last, mean = advance(initial_state(mesh), mesh, params, 20.0,
                                ignitions=ign, substeps_count=10)
    stepped, outs = _run(mesh, params, 2.0, 10, ign)
    np.testing.assert_array_equal(whole.lfn_m, stepped.lfn_m)
    np.testing.assert_array_equal(whole.tign_m, stepped.tign_m)
    assert whole.time == np.float32(20.0)
    acc = np.zeros(mesh.atm_shape, np.float32)
    for out in outs:
        acc = acc + out.GRNHFX
    np.testing.assert_array_equal(mean["GRNHFX"], acc / np.float32(10))
    np.testing.assert_array_equal(last.GRNHFX, outs[-1].GRNHFX)
    auto, _, _ = advance(initial_state(mesh), mesh, params, 20.0,
                         ignitions=ign)
    assert auto.time == np.float32(20.0)


def test_host_expf_and_powf_are_glibcs_word_for_word():
    from woof.core import noahmp_libm as lm

    rng = np.random.default_rng(7)
    x = np.concatenate([rng.uniform(-110, 90, 20000),
                        [-np.inf, np.inf, np.nan, 0.0, -0.0, 88.72, -103.9,
                         -104.0, 1e-30, -87.5]]).astype(np.float32)
    want = np.array([lm.expf(v) for v in x], dtype=np.float32)
    got = _expf(x)
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
    base = np.concatenate([rng.uniform(0, 5000, 4000), [0.0] * 4]).astype(
        np.float32)
    expo = np.concatenate([rng.uniform(-1, 2.5, 4000),
                           [2.0, 0.0, -0.3, 1.5]]).astype(np.float32)
    with np.errstate(divide="ignore"):                 # powf(0, -0.3) = inf
        want = np.array([lm.powf(a, b) for a, b in zip(base, expo)],
                        dtype=np.float32)
    assert np.array_equal(_powf(base, expo).view(np.uint32),
                          want.view(np.uint32))


def test_contracts_refuse_what_wrf_would_not_run():
    with pytest.raises(ValueError):
        FireMesh(0, 1, 1.0, 1.0)
    with pytest.raises(ValueError):
        Ignition(start=(0.0, 0.0), radius=0.0)
    with pytest.raises(ValueError):
        Ignition(start=(0.0, 0.0), start_time=5.0, end_time=1.0)
    with pytest.raises(ValueError):
        FireOptions(fire_advection=2)
    mesh = _mesh(n_atm=2)
    with pytest.raises(ValueError, match="shape"):
        FireParams.build(mesh, r_0=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="fuel_time"):
        FireParams.build(mesh, fuel_time=0.0)
    with pytest.raises(ValueError, match="betafl"):
        FireParams.build(mesh, betafl=0.0)
    with pytest.raises(ValueError, match="6 m/s"):
        FireParams.isotropic(mesh, 7.0)            # fire_ros's cap: 0/0
    with pytest.raises(ValueError, match="finite"):
        FireParams.build(mesh, vx=np.nan)
    FireParams.build(mesh, betafl=0.0, validate=False)
    # tend_ls applies fire_slope_factor after the cap; the bound follows it.
    sloped = FireParams.build(mesh, r_0=20.0, dzdxf=0.6, betafl=0.003)
    assert sloped.ros_bound(FireOptions(fire_slope_factor=3.0)) > \
        2.5 * sloped.ros_bound()
    params = FireParams.build(mesh)
    state = initial_state(mesh)
    with pytest.raises(ValueError):
        step(state, mesh, params, 0.0)
    bad = FireState(time=state.time, lfn_m=state.lfn, tign_m=state.tign_m,
                    fuel_frac=state.fuel_frac)
    with pytest.raises(ValueError, match="lfn_m"):
        step(bad, mesh, params, 1.0)
    # The kernels index raw row-major memory: any other layout is refused.
    fortran = FireState(time=state.time, lfn_m=np.asfortranarray(state.lfn_m),
                        tign_m=state.tign_m, fuel_frac=state.fuel_frac)
    with pytest.raises(ValueError, match="C-contiguous"):
        step(fortran, mesh, params, 1.0)
    with pytest.raises(Exception):
        state.time = 1.0


# ---------------------------------------------------------------------------
# the device
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_device_steps_equal_the_host_word_for_word():
    import cupy as cp

    mesh = _mesh()
    x, y = mesh.coordinates()
    wind = dict(back=0.05, head=0.6, wind_u=5.0 + np.sin(y / 70.0),
                wind_v=np.cos(x / 90.0), fgip=0.5)
    ign = [_fast(120, 200), Ignition(start=(250.0, 80.0), end=(250.0, 320.0),
                                     start_time=10.0, end_time=40.0,
                                     radius=8.0, ros=2.0)]
    host = initial_state(mesh)
    dev = initial_state(mesh, xp=cp)
    hp = FireParams.wind_driven(mesh, **wind)
    dp = FireParams.wind_driven(mesh, xp=cp, **{
        k: cp.asarray(v) if isinstance(v, np.ndarray) else v
        for k, v in wind.items()})
    for _ in range(40):
        host, ho = step(host, mesh, hp, 1.5, ignitions=ign)
        dev, do = step(dev, mesh, dp, 1.5, ignitions=ign)
    assert np.any(host.burning)
    for name in ("lfn_m", "tign_m", "fuel_frac"):
        a, b = getattr(host, name), cp.asnumpy(getattr(dev, name))
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), name
    for name in ("ros", "fire_area", "fuel_frac_burnt", "fgrnhfx", "fgrnqfx",
                 "GRNHFX", "GRNQFX", "AVG_FUEL_FRAC", "tbound"):
        a = np.asarray(getattr(ho, name), np.float32)
        b = cp.asnumpy(getattr(do, name))
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), name


@pytest.mark.gpu
@requires_gpu
@pytest.mark.filterwarnings("ignore:cupyx.allow_synchronize:UserWarning")
def test_device_time_loop_never_reads_the_device_back():
    """Under ``allow_synchronize(False)`` a device-to-host read raises."""
    import cupy as cp
    import cupyx

    mesh = _mesh()
    params = FireParams.wind_driven(mesh, back=0.05, head=0.5,
                                    wind_u=cp.full(mesh.shape, 6.0),
                                    wind_v=cp.zeros(mesh.shape), xp=cp,
                                    fgip=0.5)
    ign = [_fast(120, 200), Ignition(start=(250.0, 80.0), end=(250.0, 320.0),
                                     start_time=4.0, end_time=30.0,
                                     radius=8.0, ros=2.0)]
    state = initial_state(mesh, xp=cp)
    cp.cuda.Device().synchronize()
    with cupyx.allow_synchronize(False):
        for _ in range(5):
            state, last, mean = advance(state, mesh, params, 6.0,
                                        ignitions=ign, ros_max=0.5)
    assert float(mean["GRNHFX"].max()) > 0
    with cupyx.allow_synchronize(False), pytest.raises(
            cupyx.DeviceSynchronized):
        advance(state, mesh, params, 6.0)           # the bound is read here
