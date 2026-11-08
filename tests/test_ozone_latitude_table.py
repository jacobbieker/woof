"""The kept ozone latitude table returns the words the per-call chain returned.

``woof.ingest.wrf_ozone.latitude_table`` keeps oznini's latitude
interpolation between radiation calls on the Python route (no native
bridge), for the regional legacy-RRTMG adapter and, through
``woof.hex.radiation_ozone_memo_v841``, the hex lanes.  These cases hold
every kept-table answer to a fresh ``interp_ozone_to_latitudes`` word for
word, across months, December-January wrap and a changed latitude set.
"""
from __future__ import annotations

import numpy as np
import pytest

from woof.ingest import wrf_ozone


@pytest.fixture(autouse=True)
def python_route(monkeypatch):
    monkeypatch.setattr(wrf_ozone, "_native_ozone_entries", lambda: None)
    monkeypatch.setattr(wrf_ozone, "_LATITUDE_TABLES", [])
    monkeypatch.delenv("WOOF_OZONE_TABLE", raising=False)
    monkeypatch.delenv("WOOF_OZONE_TABLE_MAX_GB", raising=False)


def _latitudes(seed, n=257):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-89.9, 89.9, size=n).astype(np.float32)
    lat[:4] = (-0.0, 0.0, 90.0, -90.0)
    return lat


def _words(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


JULIANS = (0.0, 14.75, 45.0, 180.5, 349.25, 364.9)


def test_kept_chain_matches_fresh_chain_word_for_word():
    climo = wrf_ozone.load_ozone_climatology()
    lat = _latitudes(1)
    for julian in JULIANS:
        kept = wrf_ozone.ozn_latitude_time_int(None, julian, lat, climo)
        fresh = wrf_ozone.ozn_time_int(None, julian, wrf_ozone.interp_ozone_to_latitudes(lat, climo))
        assert np.array_equal(_words(kept), _words(fresh)), julian
    assert len(wrf_ozone._LATITUDE_TABLES) == 1


def test_outcomes_and_rebuild_on_new_latitudes():
    climo = wrf_ozone.load_ozone_climatology()
    a, b = _latitudes(2), _latitudes(3)
    first, outcome = wrf_ozone.latitude_table(a, climo)
    assert outcome == "build"
    assert np.array_equal(_words(first), _words(wrf_ozone.interp_ozone_to_latitudes(a, climo)))
    again, outcome = wrf_ozone.latitude_table(a.copy(), climo)
    assert outcome == "hit" and again is first
    other, outcome = wrf_ozone.latitude_table(b, climo)
    assert outcome == "build"
    assert np.array_equal(_words(other), _words(wrf_ozone.interp_ozone_to_latitudes(b, climo)))
    # A sign-only change of one latitude word is a different key.
    c = a.copy()
    c[1] = -0.0
    assert wrf_ozone.latitude_table(c, climo)[1] == "build"
    assert wrf_ozone.latitude_table(a, climo)[1] == "hit"


def test_cap_and_off_compute_without_keeping(monkeypatch):
    climo = wrf_ozone.load_ozone_climatology()
    lat = _latitudes(4)
    table, outcome = wrf_ozone.latitude_table(lat, climo, max_bytes=16)
    assert outcome == "over_cap" and not wrf_ozone._LATITUDE_TABLES
    assert np.array_equal(_words(table), _words(wrf_ozone.interp_ozone_to_latitudes(lat, climo)))
    monkeypatch.setenv("WOOF_OZONE_TABLE", "off")
    assert wrf_ozone.latitude_table(lat, climo)[1] == "over_cap"
    monkeypatch.setenv("WOOF_OZONE_TABLE", "maybe")
    with pytest.raises(ValueError, match="WOOF_OZONE_TABLE"):
        wrf_ozone.latitude_table(lat, climo)


def test_hex_wrapper_counts_the_shared_table():
    from woof.hex.radiation_ozone_memo_v841 import OzoneLatitudeTableMemo

    climo = wrf_ozone.load_ozone_climatology()
    lat = _latitudes(5)
    memo = OzoneLatitudeTableMemo(wrf_ozone, 4 * 2**30)
    for julian in JULIANS:
        got = memo.ozn_latitude_time_int(None, julian, lat, climo)
        fresh = wrf_ozone.ozn_time_int(None, julian, wrf_ozone.interp_ozone_to_latitudes(lat, climo))
        assert np.array_equal(_words(got), _words(fresh))
    receipt = memo.receipt()
    assert receipt["table_builds"] == 1 and receipt["table_hits"] == len(JULIANS) - 1
    assert receipt["table_bytes"] == lat.size * wrf_ozone.LEVSIZ * 12 * 4
