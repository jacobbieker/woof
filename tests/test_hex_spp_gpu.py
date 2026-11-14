"""WRF's SPP on hex (matrix item 5.4): pattern law and the column physics.

``woof.hex.spp`` draws one hex spherical-harmonic pattern per selected SPP
scheme at the physics columns and binds it to the generic column seam's WRF
``PhysicsDriver``, the driver the WRF path binds its own patterns to.

* E1: given the same pattern, the seam's MYNN (``spp_pbl = 1``) columns are
  bit for bit the columns WRF's binding produces (the pattern laid out as
  ``StochasticTimestepHook.parameter_patterns`` lays out WRF's grid pattern),
  and the pattern is load-bearing (a zero pattern changes them).  The MYNN
  SPP kernels themselves are held to native WRF in test_spp_consumers_gpu.py.
* Pattern statistics against WRF's SPP rows (Registry/registry.stoch):
  the point variance of the clipped pattern, its spatial correlation at the
  row's length scale and its lag correlation at the row's time scale.
* Limited area: a subset of points draws the same words as the full set,
  the boundary zone and padded column are zero, and restart continues the
  stream bit for bit.  The default run attaches nothing.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import mpas_column_physics_cases as cases  # noqa: E402
from woof.hex import spp  # noqa: E402

NZ, NCOL, DT = cases.NZ, cases.NCOL, cases.DT
RATES = ("du", "dv", "dtheta", "dqv", "dqc", "dqr", "dqi", "dqs", "dqg")


def _fibonacci(n):
    lat = np.arcsin(1 - 2 * (np.arange(n) + 0.5) / n)
    lon = np.mod(np.arange(n) * (math.pi * (3 - math.sqrt(5))), 2 * math.pi) - math.pi
    return lat, lon


def _cap(n, radius_m, lat0=0.6, lon0=-1.7):
    """``n`` Fibonacci points spread uniformly over a spherical cap."""
    from woof.hex.stochastic import EARTH_RADIUS_M
    z = 1 - (1 - math.cos(radius_m / EARTH_RADIUS_M)) * (np.arange(n) + 0.5) / n
    phi = np.arange(n) * (math.pi * (3 - math.sqrt(5)))
    r = np.sqrt(1 - z * z)
    local = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)
    # Rotate the pole to (lat0, lon0).
    t, p = math.pi / 2 - lat0, lon0
    ry = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
    rz = np.array([[math.cos(p), -math.sin(p), 0], [math.sin(p), math.cos(p), 0], [0, 0, 1]])
    xyz = local @ (rz @ ry).T
    return np.arcsin(np.clip(xyz[:, 2], -1, 1)), np.arctan2(xyz[:, 1], xyz[:, 0])


def _source(names, lat, lon, levels=None, **kwargs):
    levels = levels or {"conv": 4, "pbl": NZ, "lsm": 4}
    return spp.HexSppPatterns(
        names, latitude_rad=lat, longitude_rad=lon,
        shapes={name: (levels[name], 1, lat.size) for name in names}, **kwargs)


# ---------------------------------------------------------------- E1 -------

def _mynn_seam():
    physics = dict(cases.selections()["surface_layer.mynn"], spp_pbl=1)
    return cases.build(physics)


def _steps(seam, bind, steps=3):
    fields, species = cases.profile(cp, seam.plan)
    theta = fields["theta"].copy()
    frames = []
    for step in range(steps):
        bind(seam._driver, step * DT)
        out = seam.run_phase1(
            dt=DT, u=fields["u"], v=fields["v"], theta=theta,
            pressure=fields["pressure"], pressure_interface=fields["pressure_interface"],
            z_interface=fields["z_interface"], w=fields["w"],
            rho_dry=fields["rho_dry"], species=species)
        frames.append({name: cp.asnumpy(getattr(out, name)).copy() for name in RATES})
        seam.run_phase2(theta=theta, pressure=fields["pressure"],
                        z_interface=fields["z_interface"], w=fields["w"],
                        rho_dry=fields["rho_dry"], species=species)
        frames[-1]["theta"] = cp.asnumpy(theta).copy()
    return frames


def _wrf_layout(values, levels):
    """WRF's hook: a (ny+1, nx+1) gridpoint pattern, mass points [:-1, :-1],
    the same plane at every level (stochastic.py before_timestep)."""
    grid = cp.zeros((2, values.size + 1), dtype=cp.float32)
    grid[0, :-1] = values
    return cp.broadcast_to(cp.ascontiguousarray(grid[:-1, :-1]), (levels, 1, values.size))


def test_e1_seam_columns_equal_wrf_binding_of_the_same_pattern():
    lat = np.radians(np.linspace(30.0, 45.0, NCOL))
    lon = np.radians(np.linspace(-110.0, -95.0, NCOL))
    hexed, drawn = _mynn_seam(), []
    source = _source(("pbl",), lat, lon, member=3)

    def hex_bind(driver, time):
        source.bind(driver, time)
        drawn.append(source.values["pbl"].copy())
        assert driver.spp_patterns["pbl"].shape == (NZ, 1, NCOL)

    a = _steps(hexed, hex_bind)
    assert all(float(cp.max(cp.abs(p))) > 0 for p in drawn)

    wrf, calls = _mynn_seam(), iter(drawn)
    b = _steps(wrf, lambda driver, time: driver.bind_spp_patterns(
        {"pbl": _wrf_layout(next(calls), NZ)}))
    for frame_a, frame_b in zip(a, b):
        for name in frame_a:
            np.testing.assert_array_equal(frame_a[name].view(np.uint32),
                                          frame_b[name].view(np.uint32), err_msg=name)

    zero = _mynn_seam()
    c = _steps(zero, lambda driver, time: driver.bind_spp_patterns(
        {"pbl": cp.zeros((NZ, 1, NCOL), dtype=cp.float32)}))
    assert any(not np.array_equal(frame_a[name], frame_c[name])
               for frame_a, frame_c in zip(a, c) for name in frame_a)


def test_seam_refuses_a_step_without_its_pattern():
    seam = _mynn_seam()
    with pytest.raises(ValueError, match="spp_pbl=1 requires a member pattern"):
        _steps(seam, lambda driver, time: None, steps=1)


# ------------------------------------------------------- statistics -------

def _clipped_variance(stddev, cutoff_sigma):
    c = cutoff_sigma
    tail = 0.5 * math.erfc(c / math.sqrt(2))
    density = math.exp(-c * c / 2) / math.sqrt(2 * math.pi)
    return stddev ** 2 * ((1 - 2 * tail) - 2 * c * density + 2 * c * c * tail)


def _truncated_correlation(length_m, degree, distance_m, radius):
    from woof.hex.stochastic import spectral_exponent
    a = spectral_exponent(length_m, radius)
    x = math.cos(distance_m / radius)
    total = norm = 0.0
    p_prev, p_cur = 1.0, x
    for ell in range(1, degree + 1):
        weight = (2 * ell + 1) * math.exp(-(ell * (ell + 1) - 2) * a)
        total += weight * p_cur
        norm += weight
        p_prev, p_cur = p_cur, ((2 * ell + 1) * x * p_cur - ell * p_prev) / (ell + 1)
    return total / norm


@pytest.mark.parametrize("name", ["conv", "pbl"])
def test_pattern_matches_wrf_spp_rows(name):
    from woof.ensemble.stochastic import StochasticConfig as WrfConfig
    from woof.hex.stochastic import EARTH_RADIUS_M
    wrf = WrfConfig.wrf_reference("spp_" + name)
    # A cap four length scales across: plenty of pairs at the length scale.
    lat, lon = _cap(1200, 4 * wrf.lengthscale_m)
    members = 160
    sources = [_source((name,), lat, lon, member=m) for m in range(members)]
    first = sources[0]
    assert first.configs[name].length_m == wrf.lengthscale_m
    assert first.configs[name].time_s == wrf.timescale_s
    degree = first.receipt["degree"][name]

    values = cp.stack([s.advance(0.0)[name] for s in sources])
    limit = np.float32(wrf.stddev * wrf.cutoff_sigma)
    assert float(cp.max(cp.abs(values))) <= float(limit)
    variance = float(cp.mean(values.astype(cp.float64) ** 2))
    assert abs(variance / _clipped_variance(wrf.stddev, wrf.cutoff_sigma) - 1) < 0.08

    # Spatial correlation of the unit process at WRF's length scale.
    unit = cp.stack([s.processes[name].pattern[:, 0] for s in sources]).astype(cp.float64)
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1)
    distance = np.arccos(np.clip(xyz @ xyz.T, -1, 1)) * EARTH_RADIUS_M
    i, j = np.nonzero(np.abs(distance - wrf.lengthscale_m) < 0.08 * wrf.lengthscale_m)
    keep = i < j
    i, j = cp.asarray(i[keep]), cp.asarray(j[keep])
    assert int(i.size) > 200
    measured = float(cp.mean(unit[:, i] * unit[:, j]))
    expected = float(np.mean([_truncated_correlation(wrf.lengthscale_m, degree, d, EARTH_RADIUS_M)
                              for d in distance[cp.asnumpy(i), cp.asnumpy(j)]]))
    assert abs(expected - math.exp(-0.5)) < 0.08
    assert abs(measured - expected) < 0.06

    # Lag correlation at WRF's time scale (the AR(1) coefficients).
    lag = 0.25 * wrf.timescale_s
    old = cp.stack([s.processes[name].coefficients[:, 0] / s.processes[name].weights
                    for s in sources]).ravel()
    new = cp.stack([s.processes[name].coefficients[:, 0] / s.processes[name].weights
                    for s in sources if s.advance(lag) is not None]).ravel()
    lagged = np.corrcoef(cp.asnumpy(old), cp.asnumpy(new))[0, 1]
    assert abs(float(lagged) - math.exp(-0.25)) < 0.02


def test_schemes_draw_independent_streams_from_wrf_seed_labels():
    lat, lon = _fibonacci(400)
    source = _source(("conv", "pbl", "lsm"), lat, lon)
    seeds = {name: c.member_seed for name, c in source.configs.items()}
    assert len(set(seeds.values())) == 3
    other = _source(("conv", "pbl", "lsm"), lat, lon, member=1)
    assert all(other.configs[n].member_seed != seeds[n] for n in seeds)


# ------------------------------------------------------- limited area -----

def test_limited_area_subset_draws_the_same_words_and_masks_the_zone():
    lat, lon = _fibonacci(600)
    whole = _source(("pbl",), lat, lon, member=5).advance(600.0)["pbl"]
    pick = np.arange(100, 400, 3)
    zone = np.zeros(pick.size, bool)
    zone[:7] = True
    part_lat = np.concatenate([lat[pick], [np.nan]])
    part_lon = np.concatenate([lon[pick], [np.nan]])
    active = np.concatenate([~zone, [True]])
    part = _source(("pbl",), part_lat, part_lon, member=5, active=active).advance(600.0)["pbl"]
    got = cp.asnumpy(part)
    np.testing.assert_array_equal(got[:-1][~zone].view(np.uint32),
                                  cp.asnumpy(whole)[pick][~zone].view(np.uint32))
    assert not np.any(got[:-1][zone]) and got[-1] == 0


def test_restart_continues_the_stream_bit_for_bit():
    lat, lon = _fibonacci(300)
    a = _source(("pbl", "conv"), lat, lon)
    for t in (0.0, 60.0, 120.0):
        a.advance(t)
    saved = a.snapshot()
    expected = {k: cp.asnumpy(v).copy() for k, v in a.advance(180.0).items()}
    b = _source(("pbl", "conv"), lat, lon)
    b.restore(saved)
    got = b.advance(180.0)
    for name in expected:
        np.testing.assert_array_equal(cp.asnumpy(got[name]).view(np.uint32),
                                      expected[name].view(np.uint32))
    with pytest.raises(ValueError, match="other schemes"):
        _source(("pbl",), lat, lon).restore(saved)


def test_attach_is_nothing_without_a_switch_and_masks_regional_rings():
    off = SimpleNamespace(_seam=SimpleNamespace(_cfg=SimpleNamespace(spp_conv=0, spp_pbl=0, spp_lsm=0)))
    assert spp.attach(off, None) is None and not hasattr(off, "spp_patterns")

    seam = _mynn_seam()
    lat, lon = _fibonacci(NCOL)
    mesh = SimpleNamespace(arrays={"latCell": np.append(lat[:-1], 0.0),
                                   "lonCell": np.append(lon[:-1], 0.0),
                                   "bdyMaskCell": np.array([7, 1, 0, 0, 0, 0, 0, 0])},
                           garbage_cell=NCOL - 1)
    backend = SimpleNamespace(_seam=seam)
    receipt = spp.attach(backend, mesh)
    assert receipt["schemes"] == ["pbl"] and receipt["masked_columns"] == 3
    values = cp.asnumpy(backend.spp_patterns.advance(0.0)["pbl"])
    assert not np.any(values[[0, 1, NCOL - 1]]) and np.all(values[2:-1] != 0)
