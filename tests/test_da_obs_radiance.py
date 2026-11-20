"""woof.da.obs_radiance: imager brightness temperatures as LETKF batches.

A stand-in operator replaces CRTM, so these run everywhere: each channel's
"brightness temperature" is the column temperature at a fixed peak pressure,
and its Jacobian peaks there.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from woof.da.obs_radiance import RadianceAdapterError, RadianceSettings, radiance_to_gridded_obs, select_columns, wrf_columns
from woof.obs.imager_scene import FLAG_LIMB, ImagerScene, Platform
from woof.rtm.crtm import Simulation

WHEN = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
PEAK = {8: 350.0, 9: 450.0, 10: 550.0, 14: 1000.0}


class FakeOperator:
    library_path = "fake"
    receipt = None

    def __init__(self):
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
    lat = np.linspace(30.0, 37.0, ny)
    lon = np.linspace(-100.0, -91.0, nx)
    lon2, lat2 = np.meshgrid(lon, lat)
    return lat2, lon2


def _members(lat, R=4, nz=20, seed=0, noise=0.5):
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
                        U10=np.ones((ny, nx)), V10=np.ones((ny, nx)), LANDMASK=np.ones((ny, nx)),
                        XICE=np.zeros((ny, nx)), SNOWC=np.zeros((ny, nx)), HGT=np.zeros((ny, nx))))
    return out, p


def _scene(lat, lon, members, *, offset=0.0, limb_row=None):
    """A scene on the model grid whose BTs are the ensemble mean's peak temperatures plus ``offset``."""

    op = FakeOperator()
    cols = wrf_columns(members)
    jj, ii = np.nonzero(np.ones(lat.shape, bool))
    profs = [cols(r, jj, ii) for r in range(len(members))]
    values = {}
    for nm, ch in ((6200, 8), (6900, 9), (7330, 10), (11200, 14)):
        sims = np.mean([op.simulate("abi_g16", p, None, WHEN, channels=[ch]).brightness_temperature[:, 0]
                        for p in profs], axis=0)
        values[nm] = (sims + offset).reshape(lat.shape).astype(np.float32)
    flags = np.zeros(lat.shape, np.int16)
    if limb_row is not None:
        flags[limb_row] = FLAG_LIMB
    plat = Platform("GOES-19", "goes_east", "abi", "abi_g16", -75.2)
    shape = lat.shape
    return ImagerScene(time=WHEN, lat=lat, lon=lon, values=values, platforms=(plat,),
                       platform_index=np.zeros(shape, np.int16), view_zenith=np.full(shape, 40.0),
                       view_azimuth=np.zeros(shape), scan_angle=np.full(shape, 6.0), solar_zenith=np.full(shape, 120.0),
                       solar_azimuth=np.zeros(shape), flags=flags, provenance=dict(source="test", licence="CC-BY-4.0"))


def test_batches_place_one_observation_per_column_at_the_jacobian_peak():
    lat, lon = _grid()
    members, p = _members(lat)
    scene = _scene(lat, lon, members, offset=0.7)
    op = FakeOperator()
    settings = RadianceSettings(thin=2)
    batches, prov = radiance_to_gridded_obs(scene, lat=lat, lon=lon, pressure_pa=p, columns=wrf_columns(members),
                                            members=len(members), operator=op, settings=settings)
    assert [b.name for b in batches] == ["bt_6200", "bt_6900", "bt_7330"]
    for b, peak in zip(batches, (350.0, 450.0, 550.0)):
        assert b.values.shape == p.shape and b.simulated.shape == (4,) + p.shape
        # One level per observed column, at the model level nearest the peak.
        assert np.all(b.mask.sum(axis=0) <= 1)
        k = np.argmin(np.abs(np.log(p[:, 0, 0] / 100.0) - np.log(peak)))
        assert b.mask[k].sum() == b.mask.sum() == 20
        # Only the thinned lattice, offset by half a block.
        jj, ii = np.nonzero(b.mask[k])
        assert set(jj.tolist()) == {1, 3, 5, 7} and set(ii.tolist()) == {1, 3, 5, 7, 9}
        np.testing.assert_allclose(b.values[b.mask] - b.simulated.mean(axis=0)[b.mask], 0.7, atol=1e-4)
        assert np.all(b.errors[b.mask] == 2.0)
    assert prov["bands"]["6200"]["assimilated"] == 20
    assert prov["bands"]["6200"]["omb_mean_k"] == pytest.approx(0.7, abs=1e-4)
    assert prov["counts"]["window_clear"] == 20
    # Members are simulated once each per sensor, plus one Jacobian on the mean.
    assert sum(1 for c in op.calls if c[2]) == 1 and len(op.calls) == 5


def test_bias_window_and_gross_screens():
    lat, lon = _grid()
    members, p = _members(lat, noise=0.05)

    def run(scene, **kw):
        return radiance_to_gridded_obs(scene, lat=lat, lon=lon, pressure_pa=p, columns=wrf_columns(members),
                                       members=len(members), operator=FakeOperator(),
                                       settings=RadianceSettings(thin=2, **kw))

    # A 5 K departure fails the 3 K window check everywhere.
    batches, prov = run(_scene(lat, lon, members, offset=5.0))
    assert prov["counts"]["window_clear"] == 0 and not any(b.mask.any() for b in batches)
    # A 2.5 K departure passes the window test; a 2.5 K bias correction removes it.
    batches, prov = run(_scene(lat, lon, members, offset=2.5), bias_k={6200: 2.5, 6900: 2.5, 7330: 2.5})
    np.testing.assert_allclose(prov["bands"]["6900"]["omb_mean_k"], 0.0, atol=1e-4)
    # Without the bias correction and with a tight error, the gross check rejects a 2.9 K departure.
    batches, prov = run(_scene(lat, lon, members, offset=2.9), error_k={6200: 0.2, 6900: 0.2, 7330: 0.2})
    assert prov["bands"]["7330"]["assimilated"] == 0
    assert prov["bands"]["7330"]["gross_rejected"] == 20


def test_background_cloud_and_limb_screens():
    lat, lon = _grid()
    members, p = _members(lat)
    scene = _scene(lat, lon, members, limb_row=1)
    cond = np.zeros(p.shape)
    cond[:, 3, 3] = 0.01          # 0.01 kg m-2 per layer: well above the threshold through the column
    batches, prov = radiance_to_gridded_obs(scene, lat=lat, lon=lon, pressure_pa=p, columns=wrf_columns(members),
                                            members=len(members), operator=FakeOperator(),
                                            settings=RadianceSettings(thin=2, max_distance_km=2000.0), condensate_kgm2=cond)
    observed = batches[0].mask.any(axis=0)
    # With a huge search radius the limb screen alone keeps row 1 pixels out: those columns use a neighbour.
    picked = select_columns(scene, lat, lon, RadianceSettings(thin=2, max_distance_km=2000.0))
    assert 1 not in (picked["pixel"] // lat.shape[1]).tolist()
    assert not observed[3, 3]             # cloudy background column
    assert observed.sum() == 20 - 1


def test_nan_simulated_value_is_rejected():
    lat, lon = _grid()
    members, p = _members(lat)
    scene = _scene(lat, lon, members)

    class NanOperator(FakeOperator):
        def simulate(self, sensor, profiles, geometry, when, *, channels=None, jacobian=False):
            out = super().simulate(sensor, profiles, geometry, when, channels=channels, jacobian=jacobian)
            if not jacobian and not hasattr(self, "done"):
                self.done = True
                out.brightness_temperature[0, 1] = np.nan      # member 0, first column, 6900 nm
            return out

    batches, prov = radiance_to_gridded_obs(scene, lat=lat, lon=lon, pressure_pa=p, columns=wrf_columns(members),
                                            members=len(members), operator=NanOperator(),
                                            settings=RadianceSettings(thin=2))
    assert prov["bands"]["6900"]["nonfinite_rejected"] == 1
    assert prov["bands"]["6900"]["assimilated"] == 19
    assert prov["bands"]["6200"]["assimilated"] == 20
    assert np.isfinite(batches[1].simulated[:, batches[1].mask]).all()


def test_condensate_below_the_cutoff_is_not_a_cloud():
    lat, lon = _grid()
    members, p = _members(lat)
    scene = _scene(lat, lon, members)
    cond = np.zeros(p.shape)
    cond[0, 3, 3] = 1.0                   # surface layer only, far below peak + 100 hPa for every band
    batches, prov = radiance_to_gridded_obs(scene, lat=lat, lon=lon, pressure_pa=p, columns=wrf_columns(members),
                                            members=len(members), operator=FakeOperator(),
                                            settings=RadianceSettings(thin=2), condensate_kgm2=cond)
    assert batches[0].mask.any(axis=0)[3, 3]
    assert prov["bands"]["6200"]["assimilated"] == 20


def test_pixel_deduplication_keeps_the_nearest_column():
    lat, lon = _grid()
    members, p = _members(lat)
    # A scene of two pixels, so many model columns share a pixel.
    scene = _scene(lat, lon, members)
    keep = np.zeros(lat.shape, bool)
    keep[1, 1] = keep[5, 5] = True
    scene.flags[~keep] = FLAG_LIMB       # only two usable pixels
    picked = select_columns(scene, lat, lon, RadianceSettings(thin=1, max_distance_km=5000.0))
    assert picked["counts"]["with_pixel"] > 2 and picked["counts"]["pixel_duplicates_dropped"] > 0
    got = set(zip(picked["jj"].tolist(), picked["ii"].tolist()))
    assert len(picked["pixel"]) == len(set(picked["pixel"].tolist()))
    assert got == {(1, 1), (5, 5)}


def test_refusals():
    lat, lon = _grid()
    members, p = _members(lat)
    scene = _scene(lat, lon, members)
    with pytest.raises(RadianceAdapterError, match="no observation error"):
        RadianceSettings(bands=(6200, 12400))
    with pytest.raises(RadianceAdapterError, match="not an infrared"):
        RadianceSettings(bands=(640,), error_k={640: 1.0})
    with pytest.raises(RadianceAdapterError, match="carries no band"):
        select_columns(scene, lat, lon, RadianceSettings(bands=(13300,), error_k={13300: 1.0}))
    with pytest.raises(RadianceAdapterError, match="two members"):
        radiance_to_gridded_obs(scene, lat=lat, lon=lon, pressure_pa=p, columns=wrf_columns(members), members=1,
                                operator=FakeOperator())
