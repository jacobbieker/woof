"""A split run over a solar eclipse publishes the resident frame.

Two breakages, one per kind of eclipse row.  ``ELAT_TRACK``/``ELON_TRACK``
are one centre-line point per domain that history publishes as a fresh 0-d
view of a carried plane, so the streamed frame plan matched them to nothing
and refused the run (ShortFrameRefused, origin "unknown").  ``ECOBSC``/
``ECMASK`` were carried, but each buffer's shadow was evaluated on a copy of
the NEUTRAL geography it was built with whenever the radiation adapter holds
its grid on the host or in FP64: the gather rewrote the adapter's grid and
the eclipse never saw it, so the dimming (and SWDOWN) was the neutral point's.
"""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

#: Inside the pinned 2026-08-12 window; the synthetic projection below puts
#: the whole domain under a partial shadow that varies across it.
ECLIPSE_START = datetime(2026, 8, 12, 18, 15)
PROJECTION = dict(ref_lat=45.0, ref_lon=-10.0, truelat1=30.0, truelat2=60.0,
                  stand_lon=-10.0)
ECLIPSE_ROWS = ("ECOBSC", "ECMASK", "ELAT_TRACK", "ELON_TRACK")


class _Setup:
    thb = np.zeros((2, 1, 1), dtype="f4")
    pb = np.zeros((2, 1, 1), dtype="f4")
    phb = np.zeros((3, 1, 1), dtype="f4")

    def output_statics(self, nz, ny, nx):
        return {}


def test_store_frame_publishes_the_track_scalars_from_their_carried_planes():
    from tilestream import output as tsout

    cfg = SimpleNamespace(nz=2, ny=3, nx=4)
    plan = tsout.FramePlan(
        order=("ELAT_TRACK", "ELON_TRACK"),
        source={name: tsout.SOURCE_DERIVED for name in ("ELAT_TRACK", "ELON_TRACK")},
        origin={name: tsout._DERIVED_RULES[name] for name in ("ELAT_TRACK", "ELON_TRACK")},
        shape={"ELAT_TRACK": (), "ELON_TRACK": ()},
        dtype={"ELAT_TRACK": np.dtype("f4"), "ELON_TRACK": np.dtype("f4")})
    store = {"fields/elat_track": np.full((3, 4), 67.25, "f4"),
             "fields/elon_track": np.full((3, 4), -26.5, "f4")}
    frame = tsout.StoreFrame(plan, store, _Setup(), cfg)
    assert frame.store_keys() == ("fields/elat_track", "fields/elon_track")
    fields = frame.fields()
    for name, key in (("ELAT_TRACK", "fields/elat_track"),
                      ("ELON_TRACK", "fields/elon_track")):
        assert fields[name].shape == () and fields[name].dtype == np.float32
        assert fields[name].tobytes() == store[key][0, 0].tobytes()


def _split_and_resident(lw, sw):
    from tilestream import harness
    from woof.config import RunConfig
    from woof.core import streaming
    from woof.core.physics import initialize_physics
    from woof.core.radiation_composition import make_radiation

    cfg = RunConfig(nx=48, ny=40, nz=12, dx=20000., dy=20000., ztop=12000.,
                    dt=10., run_seconds=120., ra_lw_physics=lw,
                    ra_sw_physics=sw, radt=.05, terrain_opt=1,
                    moist=True, mp_physics=10)
    geo = harness.make_geography(cfg, projection=PROJECTION)

    def build():
        state, _ = harness.make_physics_state(cfg, 123, geography=geo)
        radiation = make_radiation(cfg, ECLIPSE_START, geo.lat, geo.lon,
                                   column_chunk=31)
        initialize_physics(state, cfg, radiation=radiation,
                           radiation_start_time=ECLIPSE_START,
                           radiation_latitude=geo.lat,
                           radiation_longitude=geo.lon)
        return state

    resident, split = build(), build()
    assert resident.physics.eclipse is not None
    run = streaming.make_stepper(split, cfg,
        streaming.StreamingOptions(mode="on", tile_nx=24, tile_ny=20,
                                   nbuffers=2, store="host"),
        build=streaming.prepared_domain_builder(
            SimpleNamespace(state=split, cfg=SimpleNamespace(run=cfg),
                            parent=None),
            check_geography=True))
    assert len(run.tiled_run.specs) == 4
    return cfg, resident, split, run


@pytest.mark.gpu
@pytest.mark.parametrize("lw,sw", [(1, 1)])
def test_split_run_over_an_eclipse_publishes_the_resident_frame(lw, sw):
    """(1, 1) holds its grid as host FP64, the case the shadow got wrong."""
    import cupy as cp
    from tilestream import output as tsout
    from tilestream.physics_inventory import carrier_manifest
    from woof.core.dycore import step
    from woof.io import wrfout

    cfg, resident, split, run = _split_and_resident(lw, sw)
    for _ in range(3):
        step(resident, cfg)
        run(split, cfg)
        for key, value in carrier_manifest(resident).items():
            host = cp.asnumpy(value) if isinstance(value, cp.ndarray) else np.asarray(value)
            np.testing.assert_array_equal(run.store[key], host, err_msg=key)

    shadow = cp.asnumpy(resident.physics.fields["ecobsc"])
    # A shadow that does not vary could not tell the neutral point from
    # the domain's geography.
    assert 0 < shadow.min() < shadow.max() < 1

    want = wrfout._device_state_frame(resident)
    got = run.history_fields()
    got = got.materialize() if getattr(got, "deferred", False) else got
    assert set(got) == set(want)
    for name in ECLIPSE_ROWS:
        expected = tsout._host(want[name])
        actual = np.asarray(got[name])
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, name
        assert actual.tobytes() == expected.tobytes(), name
    assert float(np.asarray(got["ELAT_TRACK"])) < 1.e29
