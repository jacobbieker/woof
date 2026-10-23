"""Wind-farm power at listed turbines: REWS, power and thrust per step.

CPU tier: the turbine-file reader and its refusals, the column arithmetic
against the hub-wind reference it reuses, the rotor-disk integral, the
density correction and the curve lookup, the interval window and its
sidecar, the ``[output] windfarm_turbines`` key and its default-off
byte-identity, the writer's ``windfarm_turbine`` axis, disk pricing, the
registry and the writer set's ownership of the window.

The GPU tier (the same code on CuPy against this reference, and real
dycore steps) is tests/test_windfarm_power_diag_gpu.py.  Every turbine
here is synthetic.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.core import interval_diagnostics
from woof.core import hub_wind_diag as hw
from woof.core import windfarm_power_diag as wp
from woof.io.history_selection import (HISTORY_PRESETS, HISTORY_VOCABULARY,
                                       WINDFARM_FIELD_REASONS,
                                       HistorySelection, WindfarmConfigError,
                                       WindfarmPowerRequest)
from woof.io.wrf_output_schema import (REGISTRY_VAR_META,
                                       WINDFARM_POWER_OUTPUT_FIELDS,
                                       WINDFARM_TURBINE_DIMENSIONED_FIELDS)

REPO = Path(__file__).resolve().parents[1]
G = np.float32(9.81)

#: A library curve the synthetic turbines use, and its rotor (m).
LIBRARY_TYPE = "E-115/3000"
LIBRARY_ROTOR = 115.0

HEADER = "name,lat,lon,hub_height_m,rotor_diameter_m,turbine_type,rated_kw\n"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write(path, body, header=HEADER):
    path.write_text(header + body)
    return path


def _turbine_file(tmp_path, rows=None):
    rows = rows if rows is not None else (
        f"wtg-a,55.02,2.02,100,{LIBRARY_ROTOR:g},{LIBRARY_TYPE},\n"
        "wtg-b,55.03,2.04,120,150,generic,5000\n"
        "wtg-c,55.04,2.03,90,80,generic,2000\n")
    return _write(tmp_path / "turbines.csv", rows)


def _state(nz=24, ny=7, nx=8, *, dz=25.0, hgt=10.0, u_of_z=None,
           v_of_z=None, phb_1d=True, theta=290.0, pressure=100000.0,
           qv=0.006):
    """A host state whose mass levels sit at ``(k + 0.5) dz`` AGL."""
    zw = np.arange(nz + 1, dtype=np.float64) * dz
    phb = (zw * float(G)).astype(np.float32)
    ph = np.full((nz + 1, ny, nx), hgt * float(G), dtype=np.float32)
    if not phb_1d:
        phb = np.ascontiguousarray(np.broadcast_to(phb[:, None, None],
                                                   (nz + 1, ny, nx)))
    zm = (np.arange(nz) + 0.5) * dz
    u_of_z = u_of_z or (lambda z: 8.0 + 0.0 * z)
    v_of_z = v_of_z or (lambda z: 0.0 * z)
    u = np.broadcast_to(u_of_z(zm)[:, None, None], (nz, ny, nx + 1))
    v = np.broadcast_to(v_of_z(zm)[:, None, None], (nz, ny + 1, nx))
    state = SimpleNamespace(
        u=np.ascontiguousarray(u, dtype=np.float32),
        v=np.ascontiguousarray(v, dtype=np.float32),
        php=ph, phb=phb, ht=np.full((ny, nx), hgt, dtype=np.float32),
        thp=np.zeros((nz, ny, nx), np.float32),
        thb=np.full(nz, theta, np.float32),
        p=np.full((nz, ny, nx), pressure, np.float32))
    if qv is not None:
        state.qv = np.full((nz, ny, nx), qv, np.float32)
    return state


def _latlon(ny=7, nx=8, lat0=55.0, lon0=2.0, step=0.01):
    lat = lat0 + step * np.arange(ny)[:, None] + 0.0 * np.arange(nx)[None]
    lon = lon0 + step * np.arange(nx)[None] + 0.0 * np.arange(ny)[:, None]
    return lat, lon


def _attached(tmp_path, state=None, rows=None):
    state = _state() if state is None else state
    ny, nx = state.ht.shape
    selection = HistorySelection.from_mapping(
        {"windfarm_turbines": str(_turbine_file(tmp_path, rows))})
    lat, lon = _latlon(ny, nx)
    diag = wp.attach_windfarm_power(
        state, selection, nx=nx, ny=ny, dx=700.0, dy=1100.0, grid_id=1,
        latlon=lambda: (lat, lon))
    return state, diag


# ---------------------------------------------------------------------------
# the turbine file
# ---------------------------------------------------------------------------

def test_the_reader_validates_every_row_in_file_order(tmp_path):
    path = _write(tmp_path / "t.csv",
                  "# a comment, then a blank line\n\n"
                  f"wtg-1,55.0,359.5,100,{LIBRARY_ROTOR:g},{LIBRARY_TYPE},\n"
                  "wtg-2, 55.1 ,2.0,120,150,generic,5000\n")
    turbines = wp.read_turbine_file(path)
    assert [t.name for t in turbines] == ["wtg-1", "wtg-2"]
    assert turbines[0].lon == pytest.approx(-0.5)        # [180, 360) folds
    assert turbines[0].turbine_type == LIBRARY_TYPE
    assert turbines[0].rated_kw is None
    assert turbines[1] == wp.Turbine("wtg-2", 55.1, 2.0, 120.0, 150.0,
                                     "generic", 5000.0)
    # Column order is free; rated_kw is optional when nothing is generic.
    reordered = _write(tmp_path / "r.csv",
                       f"x,{LIBRARY_TYPE},1.0,2.0,100,{LIBRARY_ROTOR:g}\n",
                       header="name,turbine_type,lat,lon,hub_height_m,"
                              "rotor_diameter_m\n")
    assert wp.read_turbine_file(reordered)[0].lat == 1.0


def test_a_library_name_resolves_to_its_curve_id(tmp_path):
    from woof.energy.turbine_library import load_turbine_library

    from woof.energy.turbine_library import compact

    library = load_turbine_library()
    # The first curve with an alias that names it and no other.
    curve_id, alias = next(
        (curve_id, name) for curve_id, curve in sorted(library.curves.items())
        for name in (curve.model,) + curve.names
        if name != curve_id and library._exact.get(compact(name)) == [curve_id])
    rotor = library.curves[curve_id].rotor_diameter_m
    path = _write(tmp_path / "t.csv",
                  f"wtg,55,2,{rotor:g},{rotor:g},{alias},\n")
    assert wp.read_turbine_file(path)[0].turbine_type == curve_id


@pytest.mark.parametrize("body, header, match", [
    ("", HEADER, "names no turbines"),
    ("a,1,2,100,80\n", "name,lat,lon,hub_height_m,rotor_diameter_m\n",
     "missing \\['turbine_type'\\]"),
    ("a,1,2,100,80,generic,2000,x\n", HEADER.strip() + ",colour\n",
     "unknown \\['colour'\\]"),
    ("a,1,2,100,80,generic\n", HEADER, "has 6 fields"),
    ("a,1,2,100,80,generic,2000\na,1,2,100,80,generic,2000\n", HEADER,
     "repeats turbine name 'a'"),
    (",1,2,100,80,generic,2000\n", HEADER, "name is empty"),
    ("a,north,2,100,80,generic,2000\n", HEADER, "lat must be a number"),
    ("a,91,2,100,80,generic,2000\n", HEADER, "outside \\[-90, 90\\]"),
    ("a,1,2,nan,80,generic,2000\n", HEADER, "must be finite"),
    ("a,1,2,40,80,generic,2000\n", HEADER, "exceed half the rotor"),
    ("a,1,2,100,0,generic,2000\n", HEADER, "rotor_diameter_m 0"),
    ("a,1,2,100,80,generic,\n", HEADER, "needs rated_kw"),
    ("a,1,2,100,80,generic,-5\n", HEADER, "rated_kw must be positive"),
    (f"a,1,2,100,{LIBRARY_ROTOR:g},{LIBRARY_TYPE},3000\n", HEADER,
     "carries its own rating"),
    (f"a,1,2,100,90,{LIBRARY_TYPE},\n", HEADER, "disagrees with library"),
    ("a,1,2,100,80,E-115/300O,\n", HEADER, "Did you mean .*E-115/3000"),
])
def test_a_bad_turbine_file_is_refused_by_row(tmp_path, body, header, match):
    path = _write(tmp_path / "t.csv", body, header=header)
    with pytest.raises(wp.TurbineFileError, match=match):
        wp.read_turbine_file(path)


# ---------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------

def _rough_state(rng, *, phb_1d=True):
    state = _state(nz=18, ny=6, nx=7, dz=30.0, hgt=0.0, phb_1d=phb_1d)
    state.ht[...] = rng.uniform(0.0, 400.0, state.ht.shape)
    state.php[...] = (state.ht[None] * G + rng.uniform(
        -8.0, 8.0, state.php.shape) * G).astype(np.float32)
    state.u[...] = rng.normal(7.0, 3.0, state.u.shape)
    state.v[...] = rng.normal(-1.0, 3.0, state.v.shape)
    state.thp[...] = rng.normal(0.0, 1.0, state.thp.shape)
    state.p[...] = rng.uniform(95000.0, 101000.0, state.p.shape)
    state.qv[...] = rng.uniform(0.0, 0.01, state.qv.shape)
    return state


@pytest.mark.parametrize("phb_1d", [True, False])
def test_columns_and_interpolation_are_the_hub_wind_arithmetic(phb_1d):
    """Bit-identical, turbine by turbine, to the hub-wind reference."""
    rng = np.random.default_rng(3)
    state = _rough_state(rng, phb_1d=phb_1d)
    sj = np.array([1, 4, 2, 3])
    si = np.array([1, 5, 3, 2])
    columns = wp.turbine_columns(np, state, sj, si)
    z = hw.mass_heights_agl_np(state.php, state.phb, state.ht)
    um, vm = hw.destagger_np(state.u, state.v)
    np.testing.assert_array_equal(columns["z"], z[:, sj, si])
    np.testing.assert_array_equal(columns["u"], um[:, sj, si])
    np.testing.assert_array_equal(columns["v"], vm[:, sj, si])
    # Per-turbine heights, some below the first mass level and one above
    # the top, so the hold rule is exercised.
    targets = np.array([[5.0, 40.0, 75.0, 120.0],
                        [60.0, 150.0, 200.0, 900.0]], dtype=np.float32)
    got = wp.interp_columns(np, [columns["u"], columns["v"]], columns["z"],
                            targets)
    for n, (j, i) in enumerate(zip(sj, si)):
        want = hw.interp_to_heights_np((um, vm), z, targets[:, n])
        np.testing.assert_array_equal(got[0][:, n], want[0][:, j, i])
        np.testing.assert_array_equal(got[1][:, n], want[1][:, j, i])


def _evaluate(diag, state):
    return {k: np.asarray(v) for k, v in diag.evaluate(state).items()}


def test_a_uniform_wind_has_the_hub_speed_as_its_rews(tmp_path):
    state, diag = _attached(tmp_path, _state(u_of_z=lambda z: 9.0 + 0.0 * z,
                                             v_of_z=lambda z: 3.0 + 0.0 * z))
    now = _evaluate(diag, state)
    np.testing.assert_allclose(now["rews"], np.hypot(9.0, 3.0), rtol=1e-6)


def test_rews_is_the_area_weighted_cube_mean_over_the_strips(tmp_path):
    from woof.energy.ramps import rotor_segments

    # Linear in z: interpolation to every strip centre is exact.
    state, diag = _attached(tmp_path, _state(u_of_z=lambda z: 4.0 + 0.04 * z))
    now = _evaluate(diag, state)
    hubs = [t.hub_height_m for t in diag.turbines]
    rotors = [t.rotor_diameter_m for t in diag.turbines]
    centres, weights = rotor_segments(hubs, rotors, wp.N_ROTOR_SEGMENTS)
    want = np.cbrt(np.sum(weights * (4.0 + 0.04 * centres) ** 3, axis=1))
    np.testing.assert_allclose(now["rews"], want, rtol=2e-6)
    # Shear across the disk: the cube mean exceeds the hub speed.
    assert (now["rews"] > 4.0 + 0.04 * np.asarray(hubs)).all()


def test_veer_away_from_the_hub_wind_removes_a_strips_share(tmp_path):
    # The wind turns from westerly to southerly across each rotor: strips
    # more than 90 degrees from the hub direction contribute nothing.
    state, diag = _attached(tmp_path, _state(
        nz=40, dz=10.0,
        u_of_z=lambda z: np.where(z < 100.0, 8.0, 0.0),
        v_of_z=lambda z: np.where(z < 100.0, 0.0, 8.0)))
    now = _evaluate(diag, state)
    assert (now["rews"] < 8.0 - 1e-3).all()


def max_ulp(got, want) -> int:
    """Largest distance in units in the last place between two arrays of
    one float dtype (0 is raw-word equality, but for the two zeros)."""
    got = np.ascontiguousarray(got)
    want = np.ascontiguousarray(want)
    assert got.dtype == want.dtype and got.shape == want.shape
    kind = {4: np.int32, 8: np.int64}[got.dtype.itemsize]
    a = got.view(kind).astype(np.int64)
    b = want.view(kind).astype(np.int64)
    top = np.int64(1) << (8 * got.dtype.itemsize - 1)
    # Sign-magnitude words onto one monotonic integer line.
    a = np.where(a < 0, -(a + top), a)
    b = np.where(b < 0, -(b + top), b)
    same = (got.view(kind) == want.view(kind))
    return int(np.max(np.where(same, 0, np.abs(a - b)))) if a.size else 0


def _oracle_state(rng):
    """A rough column state with the rotors' strips straddling level 0,
    the interior and (for the tallest) more than one level apart."""
    state = _rough_state(rng)
    state.u[...] = rng.normal(9.0, 4.0, state.u.shape)
    return state


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hub_interpolants_are_the_hub_wind_diagnostics_words(tmp_path, seed):
    """Oracle 1: the rotor interpolation at hub height IS hub-wind's:
    WSPD_HUB from the hub u, v and P_HUB/QV_HUB, word for word."""
    state, diag = _attached(tmp_path, _oracle_state(
        np.random.default_rng(seed)))
    now = _evaluate(diag, state)
    hubs = [t.hub_height_m for t in diag.turbines]
    for n, (row, hub) in enumerate(zip(diag.rows, hubs)):
        j, i = row[3], row[4]
        speed, _ = hw.hub_wind_np(state.u, state.v, state.php, state.phb,
                                  state.ht, np.zeros_like(state.ht),
                                  np.ones_like(state.ht), [hub])
        scalars = hw.hub_scalars_np(state.php, state.phb, state.ht,
                                    state.thp, state.thb, state.p, [hub],
                                    qv=state.qv, qc=state.qv, qr=state.qv)
        u_h = now["u_hub"][n:n + 1]
        v_h = now["v_hub"][n:n + 1]
        assert max_ulp(np.sqrt(u_h * u_h + v_h * v_h),
                       speed[0, j, i:i + 1]) == 0
        assert max_ulp(now["p_hub"][n:n + 1], scalars["P_HUB"][0, j, i:i + 1]) == 0
        assert max_ulp(now["qv_hub"][n:n + 1],
                       scalars["QV_HUB"][0, j, i:i + 1]) == 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rews_density_power_and_ct_equal_their_oracles_word_for_word(
        tmp_path, seed):
    """Oracle chain, each link independent of this module's arithmetic:

    * REWS: a scalar Python evaluation (math.sqrt, and the exact integer
      cube root woof.core.correctly_rounded_libm.cbrt_cr) of the strip
      winds woof.core.hub_wind_diag.interp_to_heights_np gives;
    * rho: woof.energy.extract.air_density of air_temperature;
    * v_eq: woof.energy.ramps.density_corrected_speed AND
      woof.energy.power_farm.density_corrected_speed (``woof energy
      power``'s), and the exact-integer cube root by hand;
    * power: TurbineCurve.fraction(v_eq) * (rated_kw * 1000);
    * Ct: TurbineCurve.thrust_coefficient(v_eq) (below cut-out).
    """
    import math

    from woof.core.correctly_rounded_libm import cbrt_cr
    from woof.energy.extract import air_density, air_temperature
    from woof.energy.power_farm import (
        density_corrected_speed as power_farm_corrected)
    from woof.energy.ramps import density_corrected_speed

    state, diag = _attached(tmp_path, _oracle_state(
        np.random.default_rng(seed)))
    now = _evaluate(diag, state)
    z = hw.mass_heights_agl_np(state.php, state.phb, state.ht)
    um, vm = hw.destagger_np(state.u, state.v)
    rews = np.empty(diag.n)
    for n, row in enumerate(diag.rows):
        j, i = row[3], row[4]
        heights = diag.targets_m[:, n]
        us, vs = hw.interp_to_heights_np((um, vm), z, heights)
        uh, vh = float(us[-1, j, i]), float(vs[-1, j, i])
        hub_norm = math.sqrt(uh * uh + vh * vh)
        cubed = 0.0
        for s in range(wp.N_ROTOR_SEGMENTS):
            a, b = float(us[s, j, i]), float(vs[s, j, i])
            speed = math.sqrt(a * a + b * b)
            norm = speed * hub_norm
            cosine = (min(max((a * uh + b * vh) / norm, 0.0), 1.0)
                      if norm > 0.0 else 1.0)
            along = max(speed * cosine, 0.0)
            cubed = cubed + float(diag.weights[s]) * (along * along * along)
        rews[n] = cbrt_cr(cubed)
    assert max_ulp(now["rews"], rews) == 0
    theta = now["theta_hub"].astype(np.float64)
    p = now["p_hub"].astype(np.float64)
    q = now["qv_hub"].astype(np.float64)
    rho = air_density(p, air_temperature(theta, p), q)
    assert max_ulp(now["rho"], rho) == 0
    v_eq = density_corrected_speed(now["rews"], now["rho"])
    assert max_ulp(now["v_eq"], v_eq) == 0
    assert max_ulp(now["v_eq"], power_farm_corrected(now["rews"],
                                                     now["rho"])) == 0
    by_hand = np.array([r * cbrt_cr(float(d) / 1.225)
                        for r, d in zip(now["rews"], now["rho"])])
    assert max_ulp(now["v_eq"], by_hand) == 0
    power = np.array([float(c.fraction(v_eq[n])) * (c.rated_kw * 1000.0)
                      for n, c in enumerate(diag.curves)])
    ct = np.array([float(c.thrust_coefficient(v_eq[n]))
                   for n, c in enumerate(diag.curves)])
    running = v_eq <= diag.cut_out_ms
    assert running.all()
    assert max_ulp(now["power"], power) == 0
    assert max_ulp(now["ct"], ct) == 0
    assert (now["power"] > 0.0).all()


def test_the_energy_products_density_correction_is_the_diagnostics_word():
    """``woof energy power`` (power_farm) and the ramps product take the
    same correctly rounded cube root as the diagnostic: max ULP 0 between
    all three and the exact integer reference, over the density and speed
    ranges a forecast reaches."""
    from woof.core.correctly_rounded_libm import cbrt_cr
    from woof.energy.power_farm import density_corrected_speed as by_power
    from woof.energy.ramps import density_corrected_speed as by_ramps

    rng = np.random.default_rng(4)
    speed = rng.uniform(0.0, 40.0, 50_000)
    rho = rng.uniform(0.6, 1.5, 50_000)
    want = np.array([v * cbrt_cr(float(d) / 1.225)
                     for v, d in zip(speed, rho)])
    assert max_ulp(by_power(speed, rho), want) == 0
    assert max_ulp(by_ramps(speed, rho), want) == 0
    assert max_ulp(speed * wp._cbrt(rho / wp.REFERENCE_DENSITY), want) == 0


def test_thrust_and_fraction_are_the_library_curve_everywhere():
    """Across cut-in, the rated knee and the Betz cap, for a measured and a
    parametric curve, the host road's fraction and Ct are the library's
    words (its pow through the C library, as NumPy's is on this host)."""
    from woof.energy.turbine_library import generic_curve, load_turbine_library

    curves = [load_turbine_library().curves[LIBRARY_TYPE],
              generic_curve(5000.0, rotor_m=150.0, label="g")]
    probe = np.concatenate([np.linspace(0.0, 30.0, 6001) + 0.0037,
                            [0.0, 3.0, 12.5, 25.0, 25.5, np.nan]])
    for n, curve in enumerate(curves):
        index = np.full(probe.shape, n)
        fraction = wp.curve_fraction(probe, curves, index)
        assert max_ulp(fraction, curve.fraction(probe)) == 0
        area = 0.5 * 1.225 * (np.pi * curve.rotor_diameter_m ** 2 / 4.0)
        ct = wp.thrust_coefficient(probe, curves, index,
                                   np.full(probe.shape, area))
        assert max_ulp(ct, curve.thrust_coefficient(probe)) == 0
        assert ct[np.isfinite(ct)].max() > 0.85     # the Betz-capped plateau


def test_power_is_bounded_by_rating_and_zero_above_cut_out(tmp_path):
    state, diag = _attached(tmp_path)
    for speed in np.linspace(0.0, 35.0, 71):
        state.u[...] = np.float32(speed)
        now = _evaluate(diag, state)
        assert (now["power"] >= 0.0).all() and (
            now["power"] <= diag.rated_w).all(), speed
        assert ((now["ct"] >= 0.0) & (now["ct"] <= 8.0 / 9.0)).all(), speed
        parked = now["v_eq"] > diag.cut_out_ms
        assert (now["power"][parked] == 0.0).all()
        assert (now["ct"][parked] == 0.0).all()
    state.u[...] = np.float32(30.0)
    assert (_evaluate(diag, state)["power"] == 0.0).all()
    state.u[...] = np.float32(17.0)
    np.testing.assert_array_equal(_evaluate(diag, state)["power"],
                                  diag.rated_w)


def test_a_dry_state_uses_dry_density(tmp_path):
    from woof.energy.extract import air_density, air_temperature

    state, diag = _attached(tmp_path, _state(qv=None))
    now = _evaluate(diag, state)
    p = now["p_hub"].astype(np.float64)
    want = air_density(p, air_temperature(
        now["theta_hub"].astype(np.float64), p), np.zeros_like(p))
    assert max_ulp(now["rho"], want) == 0


# ---------------------------------------------------------------------------
# the window, the frame and the sidecar
# ---------------------------------------------------------------------------

def test_window_is_time_weighted_and_restarts_after_each_frame(tmp_path):
    import netCDF4

    state, diag = _attached(tmp_path)
    samples = []
    for speed, dt in ((6.0, 10.0), (9.0, 30.0), (7.0, 20.0)):
        state.u[...] = np.float32(speed)
        samples.append((_evaluate(diag, state), dt))
        diag.on_step(state, dt)
    fields = interval_diagnostics.history_fields(state)
    assert tuple(fields) == WINDFARM_POWER_OUTPUT_FIELDS
    assert float(fields["WINDFARM_WINDOW_S"]) == 60.0
    # The fixed order, written out: steps in time order, turbines in file
    # order, ((0 + x0) + x1) + ... for every sum.
    total = sum(dt for _, dt in samples)
    mean_power = sum(s["power"] * dt for s, dt in samples) / total
    farm = [wp.sequential_sum(s["power"]) for s, _ in samples]
    farm_mean = sum(f * dt for f, (_, dt) in zip(farm, samples)) / total
    assert fields["POWER_FARM_MEAN"] == np.float32(farm_mean)
    # The frame's own values are the state it is written from (speed 7).
    np.testing.assert_array_equal(fields["POWER_TURBINE"],
                                  samples[-1][0]["power"].astype(np.float32))
    assert fields["POWER_FARM"] == np.float32(farm[-1])
    path = diag.close_interval(state, valid_time=datetime(2025, 1, 1, 0, 1),
                               output_dir=tmp_path / "out", grid_id=1)
    assert path.name == "windfarm_d01_2025-01-01_00:01:00.nc"
    with netCDF4.Dataset(path) as ds:
        assert list(ds["turbine_name"][:]) == ["wtg-a", "wtg-b", "wtg-c"]
        assert max_ulp(ds["POWER_MEAN"][:].data, mean_power) == 0
        assert max_ulp(ds["POWER_MAX"][:].data,
                       np.max([s["power"] for s, _ in samples], 0)) == 0
        assert max_ulp(ds["REWS_MAX"][:].data,
                       np.max([s["rews"] for s, _ in samples], 0)) == 0
        assert ds.window_seconds == 60.0
        assert float(ds["POWER_FARM_MEAN"][...]) == farm_mean
        assert float(ds["POWER_FARM_MAX"][...]) == max(farm)
        assert ds["POWER"].units == "W" and ds["CT"].units == "1"
        assert list(ds["curve_id"][:])[0] == LIBRARY_TYPE
        assert (ds["POWER_MAX"][:] <= ds["rated_power"][:]).all()
    assert diag.window_s == 0.0 and diag.window_steps == 0
    state.u[...] = np.float32(8.0)
    diag.on_step(state, 12.0)
    assert float(interval_diagnostics.history_fields(state)[
        "WINDFARM_WINDOW_S"]) == 12.0


def test_a_frame_with_no_step_reports_the_instant_as_its_window(tmp_path):
    import netCDF4

    state, diag = _attached(tmp_path)
    fields = diag.history_fields(state)
    assert float(fields["WINDFARM_WINDOW_S"]) == 0.0
    assert float(fields["POWER_FARM_MEAN"]) == pytest.approx(
        float(fields["POWER_FARM"]))
    path = diag.close_interval(state, valid_time=datetime(2025, 1, 1),
                               output_dir=tmp_path, grid_id=1)
    with netCDF4.Dataset(path) as ds:
        np.testing.assert_array_equal(ds["POWER_MEAN"][:], ds["POWER"][:])
        np.testing.assert_array_equal(ds["CT_MAX"][:], ds["CT"][:])
        assert ds.window_seconds == 0.0


def test_the_window_reads_state_and_writes_nothing_of_it(tmp_path):
    state, diag = _attached(tmp_path, _rough_state(np.random.default_rng(1)))
    before = {k: v.copy() for k, v in vars(state).items()
              if isinstance(v, np.ndarray)}
    diag.on_step(state, 10.0)
    diag.history_fields(state)
    diag.close_interval(state, valid_time=datetime(2025, 1, 1),
                        output_dir=tmp_path, grid_id=1)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(state, name), value, name)


# ---------------------------------------------------------------------------
# attachment, the registry and the roads
# ---------------------------------------------------------------------------

def test_attach_is_off_unless_asked_and_refuses_a_streamed_domain(tmp_path):
    state = _state()
    ny, nx = state.ht.shape
    assert wp.attach_windfarm_power(state, HistorySelection(), nx=nx, ny=ny,
                                    dx=700.0, dy=1100.0, grid_id=2) is None
    assert interval_diagnostics.attached(state) == ()
    selection = HistorySelection.from_mapping(
        {"windfarm_turbines": str(_turbine_file(tmp_path))})
    with pytest.raises(wp.WindfarmPowerUnsupportedRoad, match="streamed"):
        wp.attach_windfarm_power(state, selection, nx=nx, ny=ny, dx=700.0,
                                 dy=1100.0, grid_id=2, streamed=True)
    with pytest.raises(wp.WindfarmPowerError, match="latitude/longitude"):
        wp.attach_windfarm_power(state, selection, nx=nx, ny=ny, dx=700.0,
                                 dy=1100.0, grid_id=2)
    lat, lon = _latlon(ny, nx)
    diag = wp.attach_windfarm_power(state, selection, nx=nx, ny=ny, dx=700.0,
                                    dy=1100.0, grid_id=2,
                                    latlon=lambda: (lat, lon))
    assert diag.n == 3 and interval_diagnostics.attached(state) == (diag,)
    # Re-attaching the same request (a writer rebuilt on resume) is idempotent.
    assert wp.attach_windfarm_power(state, selection, nx=nx, ny=ny, dx=700.0,
                                    dy=1100.0, grid_id=2,
                                    latlon=lambda: (lat, lon)) is diag


def test_turbines_outside_the_domain_are_listed_not_sampled(tmp_path,
                                                            capsys):
    import netCDF4

    rows = ("wtg-in,55.03,2.03,100,80,generic,2000\n"
            "wtg-edge,55.0,2.0,100,80,generic,2000\n"
            "wtg-far,40.0,-70.0,100,80,generic,2000\n")
    state, diag = _attached(tmp_path, rows=rows)
    assert [t.name for t in diag.turbines] == ["wtg-in"]
    assert diag.outside == ("wtg-edge", "wtg-far")
    path = diag.close_interval(state, valid_time=datetime(2025, 1, 1),
                               output_dir=tmp_path, grid_id=1)
    with netCDF4.Dataset(path) as ds:
        assert ds.turbines_outside_domain == "wtg-edge,wtg-far"
    # None inside: nothing attaches, and the run is told why.
    elsewhere = "wtg-far,40.0,-70.0,100,80,generic,2000\n"
    capsys.readouterr()
    bare, nothing = _attached(tmp_path, rows=elsewhere)
    assert "none of the 1 [output] windfarm_turbines" in capsys.readouterr().err
    assert nothing is None and interval_diagnostics.attached(bare) == ()


def test_registry_dispatches_and_refuses_unsupported_publishers(tmp_path):
    from woof.io.history_layout import live_state_history_fields

    state = _state()
    assert not set(live_state_history_fields(state)) & set(
        WINDFARM_POWER_OUTPUT_FIELDS)
    state, diag = _attached(tmp_path, state)
    interval_diagnostics.step(state, SimpleNamespace(dt=12.0))
    assert diag.window_s == 12.0
    assert set(WINDFARM_POWER_OUTPUT_FIELDS) <= set(
        live_state_history_fields(state))
    with pytest.raises(wp.WindfarmPowerUnsupportedRoad, match="captures"):
        interval_diagnostics.refuse_unsupported_publisher(
            state, "this run captures ensemble members")
    # It coexists with the hub-wind window under its own name.
    interval_diagnostics.attach(state, hw.HubWindDiagnostic(
        HistorySelection.from_mapping({"hub_wind": True}).hub_wind,
        ny=state.ht.shape[0], nx=state.ht.shape[1]))
    assert [d.name for d in interval_diagnostics.attached(state)] == [
        wp.DIAGNOSTIC_NAME, hw.DIAGNOSTIC_NAME]


#: The inline kernel's bytes.  Changing them is a decision, not an edit:
#: re-run the GPU tier (tests/test_windfarm_power_diag_gpu.py) and re-pin.
KERNEL_SOURCE_SHA256 = (
    "fe8bbe5de70420863bd8d15fb320254b46c4a5d877505360f4b5f38b7da5d906")


def test_the_inline_kernel_source_and_its_route_are_pinned():
    import hashlib

    digest = hashlib.sha256(wp.KERNEL_SOURCE.encode("utf-8")).hexdigest()
    assert digest == wp.KERNEL_SOURCE_SHA256 == KERNEL_SOURCE_SHA256
    # No contraction, no flush, and the word-exact device libm first.
    assert wp.MODULE_OPTIONS == ("-std=c++17", "-fmad=false", "--ftz=false")
    assert wp.module_source().endswith(wp.KERNEL_SOURCE)
    assert wp.module_source().startswith(
        (REPO / "woof/core/kernels/glibc_flt64.cuh").read_text(
            encoding="utf-8")
        + (REPO / "woof/core/kernels/cr_cbrt_flt64.cuh").read_text(
            encoding="utf-8"))
    # Its one transcendental is the glibc-word pow, never CUDA's.
    import re
    code = re.sub(r"//.*", "", wp.KERNEL_SOURCE)
    calls = set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", code))
    assert calls & {"pow", "powf", "cbrt", "exp", "log", "cos", "acos",
                    "__powf", "__expf"} == set()
    assert {"glibc_pow", "uw_cbrt"} <= calls


def test_every_hook_is_wired_where_the_window_needs_it():
    wrfout = (REPO / "woof/io/wrfout.py").read_text(encoding="utf-8")
    assert "_attach_windfarm_power(node, case" in wrfout
    assert "self._ensure_windfarm_power(node)" in wrfout
    runtime = (REPO / "woof/runtime.py").read_text(encoding="utf-8")
    assert "attach_windfarm_power(" in runtime
    child = (REPO / "woof/offline_child_run.py").read_text(encoding="utf-8")
    assert "attach_windfarm_power(" in child
    batch = (REPO / "woof/ensemble/prepared_batch.py").read_text(
        encoding="utf-8")
    assert "wind-farm power folds in the original dycore epilogue" in batch


def _writer_set(selection, state):
    from woof.io.wrfout import PerDomainWrfoutWriters

    ny, nx = state.ht.shape
    lat, lon = _latlon(ny, nx)
    writers = PerDomainWrfoutWriters.__new__(PerDomainWrfoutWriters)
    writers.history_selection = selection
    node = SimpleNamespace(
        state=state, grid=SimpleNamespace(latlon_mass=lambda: (lat, lon)),
        cfg=SimpleNamespace(grid_id=2, output=None,
                            run=SimpleNamespace(nx=nx, ny=ny, dx=700.0,
                                                dy=1100.0)))
    writers.model = SimpleNamespace(_prepared_by_grid_id={2: SimpleNamespace()},
                                    node=lambda gid: node)
    return writers, node


def test_a_replaced_state_is_reattached_only_when_nothing_was_folded(
        tmp_path):
    selection = HistorySelection.from_mapping(
        {"windfarm_turbines": str(_turbine_file(tmp_path))})
    writers, node = _writer_set(selection, _state())
    writers._ensure_windfarm_power(node)
    first = interval_diagnostics.get(node.state, wp.DIAGNOSTIC_NAME)
    assert first is not None
    node.state = _state()
    writers._ensure_windfarm_power(node)
    second = interval_diagnostics.get(node.state, wp.DIAGNOSTIC_NAME)
    assert second is not None and second is not first
    second.on_step(node.state, 10.0)
    node.state = _state()
    with pytest.raises(wp.WindfarmPowerError, match="replaced"):
        writers._ensure_windfarm_power(node)
    off, bare = _writer_set(HistorySelection(), _state())
    off._ensure_windfarm_power(bare)
    assert interval_diagnostics.attached(bare.state) == ()


def test_relocation_is_refused_when_a_window_was_attached(tmp_path):
    selection = HistorySelection.from_mapping(
        {"windfarm_turbines": str(_turbine_file(tmp_path))})
    writers, node = _writer_set(selection, _state())
    writers._ensure_windfarm_power(node)
    # The mover has already swapped in a fresh state carrying no window.
    node.state = _state()
    with pytest.raises(wp.WindfarmPowerUnsupportedRoad, match="relocated"):
        writers.refresh_domain(2, grid=None, static_fields=None)


def test_a_domain_with_no_turbine_inside_is_placed_once_per_state(
        tmp_path, capsys):
    rows = "wtg-far,40.0,-70.0,100,80,generic,2000\n"
    selection = HistorySelection.from_mapping(
        {"windfarm_turbines": str(_turbine_file(tmp_path, rows))})
    writers, node = _writer_set(selection, _state())
    capsys.readouterr()
    for _ in range(3):                         # three frames
        writers._ensure_windfarm_power(node)
    assert capsys.readouterr().err.count("none of the 1") == 1
    assert interval_diagnostics.attached(node.state) == ()
    assert writers._windfarm_power_by_grid_id[2] is None


# ---------------------------------------------------------------------------
# the writer and disk pricing
# ---------------------------------------------------------------------------

def test_the_writer_puts_the_turbine_fields_on_their_own_axis(tmp_path):
    import netCDF4
    from woof.io.wrfout import WrfoutWriter

    state, diag = _attached(tmp_path, _state(nz=3, ny=7, nx=8, dz=80.0))
    assert diag.n == 3                     # n_turbine == nz: not bottom_top
    diag.on_step(state, 10.0)
    frame = {"T": np.zeros((3, 7, 8), np.float32),
             **diag.history_fields(state)}
    path = tmp_path / "wrfout_d01_wf"
    with WrfoutWriter(path, nx=8, ny=7, nz=3, dx=700.0, dy=1100.0,
                      field_schema=frame, engine="python") as writer:
        writer.write_frame("2025-01-01_00:00:00", frame)
    with netCDF4.Dataset(path) as ds:
        assert len(ds.dimensions["windfarm_turbine"]) == 3
        for name in WINDFARM_TURBINE_DIMENSIONED_FIELDS:
            assert ds[name].dimensions == ("Time", "windfarm_turbine"), name
        for name in ("WINDFARM_WINDOW_S", "POWER_FARM", "POWER_FARM_MEAN"):
            assert ds[name].dimensions == ("Time",), name
        assert ds["POWER_TURBINE"].units == "W"
        np.testing.assert_array_equal(ds["POWER_TURBINE"][0],
                                      frame["POWER_TURBINE"])


def test_a_run_without_the_diagnostic_declares_no_turbine_axis(tmp_path):
    import netCDF4
    from woof.io.wrfout import WrfoutWriter

    frame = {"T": np.zeros((2, 3, 4), np.float32)}
    path = tmp_path / "wrfout_plain"
    with WrfoutWriter(path, nx=4, ny=3, nz=2, dx=1.0, dy=1.0,
                      field_schema=frame, engine="python") as writer:
        writer.write_frame("2025-01-01_00:00:00", frame)
    with netCDF4.Dataset(path) as ds:
        assert "windfarm_turbine" not in ds.dimensions


def test_disk_pricing_counts_every_listed_turbine(tmp_path):
    from woof.config import RunConfig
    from woof.io.history_layout import history_frame_bytes

    cfg = RunConfig(nx=20, ny=10, nz=8, dx=1000.0, dy=1000.0, ztop=10000.0,
                    dt=6.0, run_seconds=60.0)
    plain = history_frame_bytes(cfg, HistorySelection())
    priced = history_frame_bytes(cfg, HistorySelection.from_mapping(
        {"windfarm_turbines": str(_turbine_file(tmp_path))}))
    values = 3 * 3 + 3                     # three rows of 3, three scalars
    assert priced - plain == 4 * values + 256 * len(
        WINDFARM_POWER_OUTPUT_FIELDS)


# ---------------------------------------------------------------------------
# the [output] key, the presets and default-off byte identity
# ---------------------------------------------------------------------------

def test_windfarm_power_is_off_by_default():
    assert HistorySelection().windfarm_power is None
    assert HistorySelection.from_mapping({}).windfarm_power is None
    assert "windfarm" not in HistorySelection().spelling()


def test_the_key_reads_a_file_relative_to_the_config(tmp_path):
    _turbine_file(tmp_path)
    selection = HistorySelection.from_mapping(
        {"windfarm_turbines": "turbines.csv"}, base_dir=tmp_path)
    request = selection.windfarm_power
    assert isinstance(request, WindfarmPowerRequest)
    assert request.source == str((tmp_path / "turbines.csv").resolve())
    assert [t.name for t in request.turbines] == ["wtg-a", "wtg-b", "wtg-c"]
    assert "windfarm_turbines=3" in selection.spelling()


@pytest.mark.parametrize("value, match", [
    (True, "must be the path of a turbine CSV"),
    ("", "must be the path of a turbine CSV"),
    ("missing.csv", "is not a file"),
])
def test_a_bad_key_value_is_refused(tmp_path, value, match):
    with pytest.raises(WindfarmConfigError, match=match):
        HistorySelection.from_mapping({"windfarm_turbines": value},
                                      base_dir=tmp_path)


def test_a_bad_file_is_refused_at_config_time(tmp_path):
    path = _write(tmp_path / "t.csv", "a,1,2,100,80,nonesuch,\n")
    with pytest.raises(WindfarmConfigError,
                       match="windfarm_turbines of <config>: .*nonesuch"):
        HistorySelection.from_mapping({"windfarm_turbines": str(path)})


def test_a_selection_that_keeps_no_turbine_field_is_refused(tmp_path):
    path = str(_turbine_file(tmp_path))
    with pytest.raises(WindfarmConfigError, match="keeps none"):
        HistorySelection.from_mapping(
            {"history_vars": ["T2"], "windfarm_turbines": path})
    kept = HistorySelection.from_mapping(
        {"history_vars": ["POWER_TURBINE"], "windfarm_turbines": path})
    assert kept.keeps("POWER_TURBINE") and not kept.keeps("CT_TURBINE")


def test_the_names_join_the_vocabulary_and_the_presets():
    assert set(WINDFARM_POWER_OUTPUT_FIELDS) <= set(REGISTRY_VAR_META)
    assert set(WINDFARM_POWER_OUTPUT_FIELDS) <= HISTORY_VOCABULARY
    assert set(WINDFARM_POWER_OUTPUT_FIELDS) <= set(WINDFARM_FIELD_REASONS)
    for preset in ("windfarm", "minimal", "severe"):
        assert set(WINDFARM_POWER_OUTPUT_FIELDS) <= HISTORY_PRESETS[preset]
    assert REGISTRY_VAR_META["POWER_TURBINE"][1] == "W"
    assert REGISTRY_VAR_META["CT_TURBINE"][1] == "1"


def test_the_output_key_table_declares_the_key():
    from woof.config import declared_key_rows
    from woof.io.history_selection import DOMAIN_OUTPUT_KEYS, OUTPUT_KEYS

    assert "windfarm_turbines" in OUTPUT_KEYS
    assert "windfarm_turbines" in DOMAIN_OUTPUT_KEYS
    rows = declared_key_rows()
    assert rows["output"]["windfarm_turbines"]["default"] is None


def test_an_output_block_that_does_not_ask_keeps_its_public_bytes(tmp_path):
    from woof.experiment import _public_config_value

    assert "windfarm_power" not in _public_config_value(HistorySelection())
    on = _public_config_value(HistorySelection.from_mapping(
        {"windfarm_turbines": str(_turbine_file(tmp_path))}))
    assert on["windfarm_power"]["turbines"][0]["name"] == "wtg-a"


def _experiment(tmp_path, output):
    from woof.experiment import load_experiment

    _turbine_file(tmp_path)
    path = tmp_path / "case.toml"
    # The experiment-route fixture of tests/test_output_history_config.py.
    path.write_text(
        '[experiment]\nname = "wf"\nstart_time = 2020-05-01T00:00:00\n'
        'run_seconds = 120.0\nrestart_interval_s = 120.0\n\n'
        '[projection]\nmap_proj = "lambert"\nref_lat = 35.0\n'
        'ref_lon = -97.0\ntruelat1 = 30.0\ntruelat2 = 40.0\n'
        'stand_lon = -97.0\n\n'
        '[shared]\nnz = 20\nztop = 20000.0\nmoist = true\nmp_physics = 0\n'
        'ra_physics = 0\nbl_pbl_physics = 0\nsf_surface_physics = 0\n'
        'cu_physics = 0\n\n'
        '[[domain]]\ngrid_id = 1\nparent_id = 0\ni_parent_start = 1\n'
        'j_parent_start = 1\nparent_grid_ratio = 1\n'
        'parent_time_step_ratio = 1\nhistory_interval_s = 60.0\ne_we = 61\n'
        'e_sn = 61\ndx = 3000.0\ndy = 3000.0\ntime_step = 15\n\n' + output)
    return load_experiment(path)


def test_identity_and_fingerprint_paths_are_untouched_when_unset(tmp_path):
    """The restart identity tolerates [output] whole; the regional
    preparation identity and public document pop the field when unset."""
    from woof.core.model import restart_identity_payload
    from woof.experiment import experiment_config_document
    from woof.regional_preparation import _experiment_identity

    plain = _experiment(tmp_path, "")
    trimmed = _experiment(tmp_path, '[output]\npreset = "minimal"\n')
    asked = _experiment(tmp_path, '[output]\npreset = "minimal"\n'
                                  'windfarm_turbines = "turbines.csv"\n')
    assert asked.output.windfarm_power is not None
    # Unset: absent from every document, so pre-feature bytes are kept.
    for exp in (plain, trimmed):
        assert "windfarm_power" not in str(experiment_config_document(exp))
        assert "windfarm_power" not in str(_experiment_identity(exp))
    # Output never binds a restart.
    assert restart_identity_payload(asked) == restart_identity_payload(trimmed)
    assert restart_identity_payload(plain) == restart_identity_payload(trimmed)
    # Set: it is part of the reviewed preparation identity, as hub_wind is.
    assert _experiment_identity(asked) != _experiment_identity(trimmed)
