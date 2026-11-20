"""WIS2 MW sounder reader (woof.obs.wis2_mw) and its LETKF adapter (woof.da.obs_mw).

Nothing is fetched and CRTM is not needed: the reader runs on an in-memory
mapping laid out like ``bkr/obs/wis2.icechunk`` with made-up footprints, and a
stand-in operator replaces CRTM (a channel's "brightness temperature" is the
column temperature at a fixed peak pressure, and its Jacobian peaks there).
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pytest

from woof.da.obs_mw import MwAdapterError, MwSettings, grody_clw_kgm2, mw_to_gridded_obs
from woof.da.obs_radiance import wrf_columns
from woof.obs import wis2_mw
from woof.obs.wis2_mw import MwFootprints
from woof.rtm.crtm import Simulation

MS = {"units": "milliseconds since 1970-01-01", "_FillValue": np.iinfo(np.int64).min}
SEC = {"units": "seconds since 1970-01-01 00:00:00", "_FillValue": np.iinfo(np.int64).min}
BOX = (30.0, 50.0, -110.0, -80.0)
START, END = "2026-10-28T11:00:00", "2026-10-28T13:00:00"
WHEN = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)


class Arr:
    def __init__(self, values, **attrs):
        self.values = np.asarray(values)
        self.attrs = attrs
        self.shape = self.values.shape

    def __getitem__(self, key):
        return self.values[key]


def _ms(texts):
    return Arr([np.datetime64(x, "ms").astype(np.int64) for x in texts], **MS)


def _sec(texts):
    return Arr([np.datetime64(x, "s").astype(np.int64) for x in texts], **SEC)


def _amsua_store():
    """Made-up AMSU-A rows: 0 Metop-B, 1 Metop-C, 2 a later copy of 0, 3 an unknown satellite, 4 a channel
    flagged and one out of bounds, 5 a zenith of 70 degrees, 6 a field of view outside the scan, 7 outside
    the window, 8 store-flagged."""
    n = 9
    bt = np.full((n, 15), 220.0, np.float32)
    bt[4, 2] = 50.0
    q = np.zeros((n, 15), np.uint8)
    q[4, 4] = 1
    return {"satellite/amsua": {
        "time": _ms([START.replace("11:", "12:")] * 7 + ["2026-10-28T15:00:00", "2026-10-28T12:00:00"]),
        "first_arrival_time": _sec(["2026-10-28T12:30:00", "2026-10-28T12:30:00", "2026-10-28T12:45:00"]
                                   + ["2026-10-28T12:30:00"] * 6),
        "latitude": Arr(np.array([40.0, 41.0, 40.0, 42.0, 43.0, 44.0, 45.0, 40.0, 40.0], np.float32)),
        "longitude": Arr([-95.0, -96.0, -95.0, -97.0, -98.0, -99.0, -100.0, -95.0, -95.0]),
        "quality_flag": Arr(np.array([0] * 8 + [16], np.uint16)),
        "satellite_identifier": Arr(np.array([3, 5, 3, 99, 3, 3, 3, 3, 3], np.int8), _FillValue=-128),
        "field_of_view_number": Arr(np.array([15, 1, 15, 15, 16, 30, 31, 15, 15], np.int8), _FillValue=-128),
        "satellite_zenith_angle": Arr(np.array([1.5, 58.0, 1.5, 1.5, 3.0, 70.0, 50.0, 1.5, 1.5], np.float32)),
        "satellite_azimuth_angle": Arr(np.zeros(n, np.float32)),
        "solar_zenith_angle": Arr(np.full(n, 120.0, np.float32)),
        "solar_azimuth_angle": Arr(np.zeros(n, np.float32)),
        "channel_number": Arr(np.arange(1, 16, dtype=np.int32)),
        "brightness_temperature": Arr(bt),
        "brightness_temperature_quality": Arr(q),
    }}


def test_scan_angle_from_field_of_view():
    assert wis2_mw.scan_angle_deg("amsua", [1, 15, 16, 30]).tolist() == pytest.approx(
        [-48.333, -1.667, 1.667, 48.333], abs=1e-3)
    assert wis2_mw.scan_angle_deg("mhs", [1, 90]).tolist() == pytest.approx([-49.444, 49.444], abs=1e-3)
    assert wis2_mw.scan_angle_deg("atms", [1, 96]).tolist() == pytest.approx([-52.725, 52.725], abs=1e-3)


def test_reader_checks_satellites_geometry_quality_and_copies(tmp_path):
    counts = Counter()
    fp = wis2_mw.read_mw(_amsua_store(), "amsua", START, END, BOX, counts)
    assert fp.sensor.tolist() == ["amsua_metop-b", "amsua_metop-c", "amsua_metop-b"]
    assert fp.lat.tolist() == [40.0, 41.0, 43.0]
    assert fp.received[0] == np.datetime64("2026-10-28T12:30:00")  # the first copy
    assert counts["amsua_duplicate_copy"] == 1 and counts["amsua_satellite_99_no_crtm_sensor"] == 1
    assert counts["amsua_sensor_zenith_missing_or_above_65"] == 1 and counts["amsua_field_of_view_outside_scan"] == 1
    assert counts["amsua_position_invalid"] == 1  # quality_flag bit 4
    assert counts["amsua_brightness_temperature_quality"] == 1
    assert counts["amsua_brightness_temperature_out_of_bounds"] == 1
    assert np.isnan(fp.bt[2, [2, 4]]).all() and np.isfinite(fp.bt[2, [0, 1, 3]]).all()
    assert fp.scan_angle[1] == pytest.approx(-48.333, abs=1e-3)
    path = tmp_path / "amsua.npz"
    fp.save(path)
    back = MwFootprints.load(path)
    assert back.sensor.tolist() == fp.sensor.tolist() and np.array_equal(back.bt, fp.bt, equal_nan=True)
    assert back.time.tolist() == fp.time.tolist() and back.counts["amsua_duplicate_copy"] == 1


def test_unknown_instrument_is_refused_by_name():
    from woof.obs.wis2 import Wis2Error
    with pytest.raises(Wis2Error, match="unknown instrument"):
        wis2_mw.read_mw({}, "ssmis", START, END, BOX)


# --------------------------------------------------------------------------- adapter
PEAK = {c: 900.0 for c in range(1, 23)}
PEAK.update({5: 700.0, 6: 500.0, 7: 350.0, 8: 250.0, 9: 150.0, 3: 700.0, 4: 300.0, 18: 350.0, 19: 450.0,
             20: 550.0, 21: 650.0, 22: 750.0, 10: 60.0, 11: 1.0})


class FakeOperator:
    library_path = "fake"
    receipt = None

    def __init__(self, sensors=("amsua_metop-b", "mhs_metop-b", "atms_n20")):
        self.sensors = tuple(sensors)
        self.calls = []

    def simulate(self, sensor, profiles, geometry, when, *, channels=None, jacobian=False):
        self.calls.append((sensor, profiles.count, jacobian))
        p = profiles.pressure_hpa
        bt = np.stack([np.array([np.interp(PEAK[c], p[n], profiles.temperature_k[n]) for n in range(profiles.count)])
                       for c in channels], axis=1)
        out = Simulation(sensor=sensor, channels=np.array(channels), brightness_temperature=bt)
        if jacobian:
            k = np.zeros((profiles.count, len(channels), profiles.layers))
            for b, c in enumerate(channels):
                k[:, b] = np.exp(-((np.log(p) - np.log(PEAK[c])) / 0.2) ** 2)
            out.k_temperature = k
        return out


def _grid(ny=8, nx=10):
    lon2, lat2 = np.meshgrid(np.linspace(-160.0, -151.0, nx), np.linspace(20.0, 27.0, ny))
    return lat2, lon2


def _members(lat, R=4, nz=20, seed=0, noise=0.3, land=None):
    rng = np.random.default_rng(seed)
    ny, nx = lat.shape
    eta = np.linspace(1.0, 0.0, nz + 1)
    full = 5000.0 + eta[:, None, None] * (100000.0 - 5000.0)
    p = np.broadcast_to(0.5 * (full[1:] + full[:-1]), (nz, ny, nx)).copy()
    out = []
    for _ in range(R):
        t = np.maximum(290.0 - 6.5e-3 * 7400.0 * np.log(1.0e5 / p), 215.0) + rng.normal(0, noise, (1, ny, nx))
        theta = t * (1.0e5 / p) ** (287.04 / 1004.5)
        out.append(dict(T=theta - 300.0, P=np.zeros_like(p), PB=p, QVAPOR=0.01 * (p / 1e5) ** 3,
                        PSFC=np.full((ny, nx), 100000.0), P_TOP=5000.0, TSK=np.full((ny, nx), 291.0),
                        U10=np.ones((ny, nx)), V10=np.ones((ny, nx)),
                        LANDMASK=np.zeros((ny, nx)) if land is None else land,
                        XICE=np.zeros((ny, nx)), SNOWC=np.zeros((ny, nx)), HGT=np.zeros((ny, nx))))
    return out, p


def _footprints(lat, lon, members, instrument="amsua", sensor="amsua_metop-b", offset=0.1, fov=15, edit=None):
    """A footprint at every model column whose BTs are the ensemble mean's H plus ``offset``."""
    nchan = wis2_mw.INSTRUMENTS[instrument]["channels"]
    op = FakeOperator()
    cols = wrf_columns(members)
    jj, ii = np.nonzero(np.ones(lat.shape, bool))
    sims = np.mean([op.simulate(sensor, cols(r, jj, ii), None, WHEN, channels=list(range(1, nchan + 1)))
                    .brightness_temperature for r in range(len(members))], axis=0)
    bt = sims + offset
    if edit:
        edit(bt)
    n = jj.size
    return MwFootprints(instrument=instrument, channels=np.arange(1, nchan + 1), sensor=np.full(n, sensor),
                        lat=lat.ravel(), lon=lon.ravel(), time=np.full(n, np.datetime64("2026-10-28T12:00:00", "s")),
                        received=np.full(n, np.datetime64("2026-10-28T12:30:00", "s")), fov=np.full(n, fov),
                        scan_angle=np.zeros(n), sensor_zenith=np.full(n, 10.0), sensor_azimuth=np.zeros(n),
                        solar_zenith=np.full(n, 120.0), solar_azimuth=np.zeros(n), bt=bt)


def _run(fp, lat, lon, members, p, op=None, **kw):
    return mw_to_gridded_obs(fp, lat=lat, lon=lon, pressure_pa=p, columns=wrf_columns(members),
                             members=len(members), operator=op or FakeOperator(), when=WHEN,
                             settings=MwSettings(**kw.pop("settings", {})), **kw)


def test_amsua_batches_at_the_jacobian_peak_with_wrfda_channels_and_errors():
    lat, lon = _grid()
    members, p = _members(lat)
    op = FakeOperator()
    batches, prov = _run(_footprints(lat, lon, members), lat, lon, members, p, op=op)
    assert [b.name for b in batches] == [f"mw_amsua_metop-b_ch{c}" for c in (5, 6, 7, 8, 9)]
    for b, c in zip(batches, (5, 6, 7, 8, 9)):
        k = np.argmin(np.abs(np.log(p[:, 0, 0] / 100.0) - np.log(PEAK[c])))
        assert b.mask[k].sum() == b.mask.sum() == 20  # every other column, one level each
        np.testing.assert_allclose(b.values[b.mask] - b.simulated.mean(axis=0)[b.mask], 0.1, atol=1e-6)
    assert batches[1].errors[batches[1].mask][0] == 0.25  # WRFDA's metop amsua channel 6
    assert prov["batches"]["mw_amsua_metop-b_ch5"]["omb_mean_k"] == pytest.approx(0.1)
    assert prov["counts"]["not_open_sea"] == 0 and prov["counts"]["limb"] == 0
    assert sum(1 for c in op.calls if c[2]) == 1 and len(op.calls) == 5  # four members and one Jacobian


def test_land_limb_scattering_and_departure_screens():
    lat, lon = _grid()
    land = np.zeros(lat.shape)
    land[:, :4] = 1.0
    members, p = _members(lat, land=land)
    _, prov = _run(_footprints(lat, lon, members), lat, lon, members, p)
    assert prov["counts"]["not_open_sea"] == 8 and prov["batches"]["mw_amsua_metop-b_ch5"]["assimilated"] == 12
    members, p = _members(lat)
    batches, prov = _run(_footprints(lat, lon, members, fov=2), lat, lon, members, p)
    assert prov["counts"]["limb_footprints"] == 80 and prov["counts"].get("with_footprint", 0) == 0
    assert all(b.mask.sum() == 0 for b in batches)
    # a limb footprint nearer a column gives way to an inner one further off
    fov = np.full(lat.size, 15)
    fov[1::2] = 2
    two = _footprints(lat, lon, members, fov=fov)
    _, prov = _run(two, lat, lon, members, p, settings=dict(max_distance_km=150.0))
    assert prov["counts"]["limb_footprints"] == 40 and prov["counts"]["limb"] == 0
    assert prov["batches"]["mw_amsua_metop-b_ch5"]["assimilated"] > 0

    def scatter(bt):
        bt[:, 0] = bt[:, 14] + 4.0  # 23.8 GHz 4 K above 89 GHz: precipitation scattering
    _, prov = _run(_footprints(lat, lon, members, edit=scatter), lat, lon, members, p)
    assert prov["counts"]["scattering_index"] == 20 and prov["batches"]["mw_amsua_metop-b_ch7"]["assimilated"] == 0
    # 1 K departures pass channel 5 (0.3 K error: 3 sigma is 0.9 K) only after the bias offset.
    fp = _footprints(lat, lon, members, offset=1.0)
    _, prov = _run(fp, lat, lon, members, p)
    assert prov["batches"]["mw_amsua_metop-b_ch5"]["omb_sigma_rejected"] == 20
    _, prov = _run(fp, lat, lon, members, p, settings=dict(bias_k={"amsua_metop-b:5": 1.0}))
    assert prov["batches"]["mw_amsua_metop-b_ch5"]["assimilated"] == 20
    assert prov["batches"]["mw_amsua_metop-b_ch5"]["omb_mean_k"] == pytest.approx(0.0, abs=1e-9)
    _, prov = _run(_footprints(lat, lon, members, offset=20.0), lat, lon, members, p,
                   settings=dict(error_k={5: 10.0}))
    assert prov["batches"]["mw_amsua_metop-b_ch5"]["omb_abs_rejected"] == 20


def test_amsua_without_channel_15_falls_back_to_the_channel_1_2_cloud_liquid():
    assert grody_clw_kgm2(165.0, 150.0, 0.0) == 0.0  # clear ocean
    assert grody_clw_kgm2(220.0, 200.0, 0.0) == pytest.approx(0.551, abs=0.01)
    assert np.isnan(grody_clw_kgm2(285.0, 200.0, 0.0))
    lat, lon = _grid()
    members, p = _members(lat)

    def dead_15(bt):
        bt[:, 14] = np.nan
        bt[:, 0], bt[:, 1] = 165.0, 150.0
        bt[:40, 0], bt[:40, 1] = 220.0, 200.0  # the first four rows: cloud liquid near 0.55 kg/m2
    _, prov = _run(_footprints(lat, lon, members, edit=dead_15), lat, lon, members, p)
    assert prov["counts"]["scattering_index"] == 0
    assert prov["counts"]["scattering_index_missing_clw"] == 10
    assert prov["batches"]["mw_amsua_metop-b_ch6"]["assimilated"] == 10


def test_background_cloud_liquid_screen():
    lat, lon = _grid()
    members, p = _members(lat)
    cloud = np.zeros(p.shape)
    cloud[3, :4] = 0.25  # a low cloud layer with 0.25 kg/m2 over the first four rows
    _, prov = _run(_footprints(lat, lon, members), lat, lon, members, p, cloud_liquid_kgm2=cloud)
    assert prov["counts"]["background_cloud_liquid"] == 10
    assert prov["batches"]["mw_amsua_metop-b_ch6"]["assimilated"] == 10
    _, prov = _run(_footprints(lat, lon, members), lat, lon, members, p)
    assert prov["background_cloud"].startswith("skipped")


def test_atms_screens_and_a_channel_above_the_model_top():
    lat, lon = _grid()
    members, p = _members(lat)

    def wet(bt):
        bt[10:20, 15] = bt[10:20, 16] + 4.0  # 88 GHz above 165 GHz on the second row (five thinned columns)
    fp = _footprints(lat, lon, members, instrument="atms", sensor="atms_n20", edit=wet)
    batches, prov = _run(fp, lat, lon, members, p, settings=dict(channels=(6, 11, 18)))
    by = prov["batches"]
    assert prov["counts"]["scattering_index_16_17"] == 5
    assert by["mw_atms_n20_ch6"]["assimilated"] == 20  # channels 1-8 do not read the 16-17 index
    assert by["mw_atms_n20_ch18"]["assimilated"] == 15
    assert by["mw_atms_n20_ch11"]["peak_above_model_top"] == 20 and by["mw_atms_n20_ch11"]["assimilated"] == 0

    def warm3(bt):
        bt[:, 2] += 6.0  # O-B of 6 K at 50.3 GHz
    _, prov = _run(_footprints(lat, lon, members, instrument="atms", sensor="atms_n20", edit=warm3), lat, lon,
                   members, p)
    assert prov["counts"]["scattering_omb_ch3"] == 20 and prov["batches"]["mw_atms_n20_ch6"]["assimilated"] == 0


def test_mhs_and_a_sensor_crtm_has_not_opened():
    lat, lon = _grid()
    members, p = _members(lat)
    batches, prov = _run(_footprints(lat, lon, members, instrument="mhs", sensor="mhs_metop-b", offset=0.5),
                         lat, lon, members, p)
    assert [b.name for b in batches] == [f"mw_mhs_metop-b_ch{c}" for c in (3, 4, 5)]
    assert prov["batches"]["mw_mhs_metop-b_ch4"]["error_k"] == 2.0
    _, prov = _run(_footprints(lat, lon, members, instrument="mhs", sensor="mhs_metop-b"), lat, lon, members, p,
                   op=FakeOperator(sensors=("amsua_metop-b",)))
    assert prov["counts"]["sensor_mhs_metop-b_not_open_in_crtm"] == 20
    assert prov["batches"]["mw_mhs_metop-b_ch3"]["assimilated"] == 0


def test_settings_are_validated():
    with pytest.raises(MwAdapterError):
        MwSettings(thin=0)
    with pytest.raises(MwAdapterError):
        MwSettings(error_k={5: 0.0})
    lat, lon = _grid()
    members, p = _members(lat)
    with pytest.raises(MwAdapterError, match="not \\[16\\]"):
        _run(_footprints(lat, lon, members), lat, lon, members, p, settings=dict(channels=(5, 16)))
