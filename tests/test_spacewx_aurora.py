"""woof.spacewx L3b: the auroral oval (Starkov 1994) and the solar-wind readers.

Everything is synthetic except the cache-only check of the real IGRF-14 dipole
pole.  The kernel test runs on a card (CUDA or Metal) against the float64
reference.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from woof.spacewx import assets, aurora as au, igrf, solarwind as sw

from conftest import requires_gpu  # noqa: E402
from test_spacewx_ionosphere import _dipole  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WHEN = datetime(2024, 5, 10, 21, 0)


# ---------------------------------------------------------------- the Starkov table

def test_starkov_table_is_complete_and_frozen():
    rows = [au.STARKOV[b][c] for b in au.BOUNDARIES for c in au.COEFFICIENTS]
    assert len(rows) == 21 and all(len(r) == 4 for r in rows)
    text = ";".join(",".join(f"{v:.2f}" for v in r) for r in rows)
    # The table of Sigernes et al. 2011 (Table 2); eab A0 b0 is 1.61, not ocbpy's 1.16.
    assert au.STARKOV["eab"]["A0"][0] == 1.61
    assert hashlib.sha256(text.encode()).hexdigest() == (
        "0afa09eb5707f97989f8c884870ea870d2b2766724defccfa17e80bf786fc77b")


def test_kp_to_al_is_starkovs_cubic():
    assert au.kp_to_al(0) == pytest.approx(18.0)
    assert au.kp_to_al(3) == pytest.approx(18 - 36.9 + 244.8 - 54.0)
    assert au.kp_to_al(9) == pytest.approx(652.5)
    with pytest.raises(au.AuroraError):
        au.kp_to_al(9.5)


@pytest.mark.parametrize("kp", [0.0, 1.0, 3.0, 5.0, 7.0, 9.0])
def test_boundaries_are_ordered_and_expand_with_activity(kp):
    mlt = np.linspace(0, 24, 97)
    coef = au.coefficients(au.kp_to_al(kp))
    ocb, eab, dif = (au.boundary_colatitude(coef[i], mlt) for i in range(3))
    assert np.all(ocb > 0) and np.all(ocb < eab) and np.all(eab <= dif + 1e-9)
    assert np.all(dif < 45.0)
    if kp > 0:
        quieter = au.coefficients(au.kp_to_al(kp - 1.0))
        assert au.boundary_colatitude(coef[2], 0.0) > au.boundary_colatitude(quieter[2], 0.0)


def test_al_is_clamped_to_the_fit_range():
    np.testing.assert_array_equal(au.coefficients(1.0), au.coefficients(au.AL_RANGE[0]))
    np.testing.assert_array_equal(au.coefficients(-5000.0), au.coefficients(au.AL_RANGE[1]))
    with pytest.raises(au.AuroraError):
        au.coefficients(float("nan"))


# ---------------------------------------------------------------- coordinates

def test_axial_dipole_frame_and_mlt():
    model = _dipole()           # g10 only: the boreal pole is the geographic north pole
    frame = au.dipole_frame(model, WHEN)
    np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.cross(frame[1], frame[2]), frame[0], atol=1e-12)
    assert au.pole(frame)[0] == pytest.approx(90.0)
    oval = au.Aurora([60.0, 60.0, -60.0], [0.0, 0.0, 0.0], when=WHEN, igrf_model=model)
    sun_lat, sun_lon = au.subsolar_point(WHEN)
    oval.lon = np.array([sun_lon, sun_lon + 180.0, sun_lon + 90.0])
    ref = oval.reference(WHEN, 100.0)
    np.testing.assert_allclose(ref["mlt"], [12.0, 0.0, 18.0], atol=1e-9)
    # geodetic 60 deg is geocentric 59.83 deg on the axial dipole
    np.testing.assert_allclose(ref["mlat"], [59.8336, 59.8336, -59.8336], atol=1e-3)


def test_real_igrf14_dipole_pole():
    path = assets.cache_root() / igrf.IGRF14.subdir / igrf.IGRF14.name
    if not path.is_file():
        pytest.skip(f"IGRF-14 not in the space-weather cache ({assets.cache_root()})")
    lat, lon = au.pole(au.dipole_frame(igrf.load(), datetime(2025, 1, 1)))
    assert lat == pytest.approx(80.8, abs=0.2) and lon == pytest.approx(-72.7, abs=0.5)


def test_refuses_bad_columns():
    with pytest.raises(au.AuroraError):
        au.Aurora([91.0], [0.0], when=WHEN, igrf_model=_dipole())
    with pytest.raises(au.AuroraError):
        au.Aurora([1.0, 2.0], [0.0], when=WHEN, igrf_model=_dipole())


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("kp", [0.33, 4.0, 8.67])
def test_device_kernel_matches_the_float64_reference(kp):
    import cupy as cp
    model = igrf.parse(_dipole_tilted(), "synthetic tilted")
    lat, lon = np.meshgrid(np.arange(-90.0, 90.1, 1.0), np.arange(-180.0, 180.0, 2.0), indexing="ij")
    oval = au.Aurora(lat, lon, when=WHEN, igrf_model=model)
    got = {k: cp.asnumpy(v).astype(np.float64) for k, v in oval.evaluate(WHEN, au.kp_to_al(kp)).items()}
    ref = oval.reference(WHEN, au.kp_to_al(kp))
    np.testing.assert_allclose(got["mlat"], ref["mlat"], atol=5e-3)
    dmlt = np.abs(got["mlt"] - ref["mlt"])
    assert np.minimum(dmlt, 24.0 - dmlt).max() < 2e-3
    np.testing.assert_allclose(got["eq_dist"], ref["eq_dist"], atol=1e-2)
    clear = np.abs(ref["eq_dist"]) > 0.05
    assert (got["zone"] != ref["zone"])[clear].sum() <= 0.002 * clear.sum()
    assert set(np.unique(got["zone"])) <= {0.0, 1.0, 2.0, 3.0}
    assert {2.0, 3.0} <= set(np.unique(got["zone"]))


def _dipole_tilted():
    return ("# synthetic: a tilted dipole\nc/s  main main SV\ng/h n m 2020.0 2025.0 2025-30\n"
            "g  1  0  -29400.0 -29350.0 10.0\ng  1  1  -1450.0 -1400.0 5.0\nh  1  1  4650.0 4600.0 -10.0\n")


# ---------------------------------------------------------------- the solar wind

def test_newell_and_boyle_coupling_functions():
    v = 400.0
    assert sw.newell_coupling(v, 0.0, -5.0) == pytest.approx(v ** (4 / 3) * 5.0 ** (2 / 3))
    assert sw.newell_coupling(v, 5.0, 0.0) == pytest.approx(v ** (4 / 3) * 5.0 ** (2 / 3) * np.sin(np.pi / 4) ** (8 / 3))
    assert sw.newell_coupling(v, 0.0, 5.0) == pytest.approx(0.0, abs=1e-9)
    assert sw.boyle_potential(v, 5.0, -5.0) == pytest.approx(16.0 + 11.7 * 5.0)
    assert sw.boyle_potential(v, 5.0, 5.0) == pytest.approx(16.0)
    assert sw.kp_from_coupling(0.0, 0.0) == 0.0
    assert sw.kp_from_coupling(1e9, 1e9) == 9.0
    f = sw.KP_FIT
    x, w = 8000.0, 0.5e5
    assert sw.kp_from_coupling(x, w) == pytest.approx(f["a"] + f["b"] * 0.8 ** f["p"] + f["c"] * 0.005)


def _omni_row(t: datetime, by=1.0, bz=-3.0, n=5.0, v=400.0, kp=33, al=-120):
    words = ["0"] * 55
    doy = (t - datetime(t.year, 1, 1)).days + 1
    words[0], words[1], words[2] = str(t.year), str(doy), str(t.hour)
    words[15], words[16], words[23], words[24] = f"{by:.1f}", f"{bz:.1f}", f"{n:.1f}", f"{v:.0f}"
    words[38], words[52] = str(kp), str(al)
    return " ".join(words)


def test_omni2_reader_fills_codes_and_gaps(tmp_path):
    t0 = datetime(2024, 5, 10, 0)
    rows = [_omni_row(t0 + timedelta(hours=h)) for h in range(6)]
    rows[2] = _omni_row(t0 + timedelta(hours=2), bz=999.9, v=9999, kp=37, al=99999)
    path = tmp_path / "omni2_2024.dat"
    path.write_text("\n".join(rows) + "\n")
    wind = sw.read_omni2(path)
    assert wind.hours[0] == np.datetime64("2024-05-10T00", "h") and wind.hours.size == 6
    assert np.isnan(wind.bz[2]) and np.isnan(wind.speed[2]) and np.isnan(wind.al[2])
    assert wind.kp[0] == pytest.approx(10 / 3) and wind.kp[2] == pytest.approx(11 / 3)
    assert wind.al[0] == -120.0
    # the Kp interval 03-06 UT reads hours 02, 03, 04: hour 02 is missing, the mean skips it
    x = sw.newell_coupling(400.0, 1.0, -3.0)
    want = sw.kp_from_coupling(x, np.sqrt(5.0) * 400.0 ** 2)
    assert wind.kp_estimate(t0 + timedelta(hours=4)) == pytest.approx(want)
    gap = tmp_path / "gap.dat"
    gap.write_text("\n".join([rows[0], rows[3]]) + "\n")
    with pytest.raises(sw.SolarWindError, match="consecutive"):
        sw.read_omni2(gap)
    with pytest.raises(sw.SolarWindError, match="outside"):
        wind.hourly(t0 + timedelta(days=2))


def test_swpc_reader_keeps_the_active_spacecraft_and_needs_ten_minutes():
    t0 = datetime(2026, 11, 17, 10, 0)
    mag, wind = [], []
    for k in range(120):
        t = (t0 + timedelta(minutes=k)).isoformat()
        mag.append({"time_tag": t, "active": True, "source": "SOLAR1", "by_gsm": 2.0, "bz_gsm": -4.0})
        mag.append({"time_tag": t, "active": False, "source": "ACE", "by_gsm": 50.0, "bz_gsm": 50.0})
        if k < 60 or k % 10 == 0:      # the second hour has only 6 plasma minutes
            wind.append({"time_tag": t, "active": True, "source": "SOLAR1",
                         "proton_density": 4.0, "proton_speed": 500.0})
    record = sw.parse_swpc(json.dumps(mag[::-1]), json.dumps(wind[::-1]))
    assert record.hours.tolist() == [np.datetime64("2026-11-17T10", "h"), np.datetime64("2026-11-17T11", "h")]
    np.testing.assert_allclose(record.by, [2.0, 2.0])
    np.testing.assert_allclose(record.bz, [-4.0, -4.0])
    assert record.speed[0] == 500.0 and np.isnan(record.speed[1])
    with pytest.raises(sw.SolarWindError, match="active"):
        sw.parse_swpc(json.dumps(mag[1::2]), json.dumps(wind))


def test_load_needs_a_source():
    with pytest.raises(sw.SolarWindError):
        sw.load("", WHEN, WHEN)


# ---------------------------------------------------------------- the door

def test_aurora_door_is_registered():
    out = subprocess.run([sys.executable, "-m", "woof.cli", "spacewx", "aurora", "--help"],
                         capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "--solar-wind" in out.stdout and "--observed-al" in out.stdout
