"""The Dynamical.org ASOS Parquet reader, against a local archive.

A small year file is built with pyarrow (several stations, several row
groups, sorted by station like the real one) and served from a local
``ThreadingHTTPServer`` that honours ``Range`` and ``If-Match``, states an
ETag, and refuses any request that does not carry woof's user agent --
the real host answers Python's default user agent with HTTP 403.  The live
archive is exercised once, behind ``WOOF_NETWORK_TESTS=1``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from woof.obs import dynamical_asos as dyn

UTC = timezone.utc
TARGET = datetime(2025, 6, 1, 12, tzinfo=UTC)
FILLER = 6000


# ---------------------------------------------------------------------------
# The frozen table (no network, no pyarrow)
# ---------------------------------------------------------------------------

def test_the_packaged_station_table_covers_great_britain():
    stations = dyn.stations_in_bbox(-8.25, 49.85, 1.8, 60.9)
    assert len(stations) == dyn.station_count_in_bbox(-8.25, 49.85, 1.8, 60.9)
    assert len(stations) >= 20
    assert {"EGLL", "EGPF"} <= {row["station_id"] for row in stations}
    assert set(stations[0]) >= {"station_id", "name", "latitude", "longitude",
                                "elevation_m", "country"}
    record = json.loads(dyn._TABLE_PATH.read_text(encoding="utf-8"))
    assert record["schema"] == dyn.STATION_TABLE_SCHEMA
    ids = [row["station_id"] for row in record["stations"]]
    assert ids == sorted(ids) and len(ids) == len(set(ids))
    assert record["source_url"].endswith("/data.parquet") and record["year"] >= 2025


def test_a_bbox_across_the_antimeridian(monkeypatch):
    table = tuple({"station_id": sid, "name": sid, "latitude": 0.0, "longitude": lon,
                   "elevation_m": 1.0, "country": "XX"}
                  for sid, lon in (("W", -175.0), ("E", 175.0), ("M", 0.0), ("Z", -180.0)))
    monkeypatch.setattr(dyn, "_station_table", lambda: table)
    assert {r["station_id"] for r in dyn.stations_in_bbox(170, -10, -170, 10)} == {"W", "E", "Z"}
    assert {r["station_id"] for r in dyn.stations_in_bbox(170, -10, 190, 10)} == {"W", "E", "Z"}
    assert {r["station_id"] for r in dyn.stations_in_bbox(-10, -10, 10, 10)} == {"M"}
    assert dyn.station_count_in_bbox(-180, -90, 180, 90) == 4
    # A box across the prime meridian in the 0..360 convention.
    assert {r["station_id"] for r in dyn.stations_in_bbox(350, -10, 10, 10)} == {"M"}
    assert {r["station_id"] for r in dyn.stations_in_bbox(350, -10, 370, 10)} == {"M"}
    assert dyn.station_count_in_bbox(-10, 5, 10, 10) == 0


def test_the_module_imports_and_counts_without_pyarrow(monkeypatch):
    for name in ("pyarrow", "pyarrow.parquet", "pyarrow.compute"):
        monkeypatch.setitem(sys.modules, name, None)
    assert dyn.station_count_in_bbox(-8.25, 49.85, 1.8, 60.9) >= 20


def test_a_missing_pyarrow_is_unavailable_not_a_traceback(monkeypatch, tmp_path):
    for name in ("pyarrow", "pyarrow.parquet", "pyarrow.compute"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(dyn.DynamicalUnavailable, match=r"recast-woof\[obs\]"):
        dyn.fetch_surface(None, TARGET, tmp_path, station_ids=["EGLL"])


# ---------------------------------------------------------------------------
# A local archive
# ---------------------------------------------------------------------------

def _rows():
    """``(station, minutes from TARGET, tmpc, dwpc, sknt, mslp)``."""

    rows = []
    # AAA: -10 and +20 min; the -10 one is nearest and carries everything.
    rows += [("AAA", -60, 18.0, 9.0, 8.0, 1012.0), ("AAA", -10, 20.0, 10.0, 10.0, 1013.2),
             ("AAA", 20, 21.0, 10.0, 12.0, 1013.0), ("AAA", 60, 22.0, 11.0, 12.0, 1012.5)]
    # BBB: -20 and +10 min; the +10 one has no MSLP.
    rows += [("BBB", -20, 15.0, 5.0, 4.0, 1000.0), ("BBB", 10, 16.0, 6.0, 5.0, None)]
    # CCC: one report out of range in three -> the station fails the 5 % screen.
    rows += [("CCC", -60, 20.0, 10.0, 5.0, None), ("CCC", 0, 60.0, 10.0, 5.0, None),
             ("CCC", 60, 20.0, 10.0, 5.0, None)]
    # DDD: every five minutes for two hours; only the on-the-hour report has
    # a dewpoint above its temperature (1 in 25 = 4 %, under the screen), so
    # that report is dropped and the station keeps the next nearest (ties go
    # to the earlier report: -5 min).
    for minute in range(-60, 61, 5):
        dew = 25.0 if minute == 0 else 5.0
        rows.append(("DDD", minute, 10.0 + minute / 60.0, dew, 3.0, 1020.0))
    # FFF: only +/-20 min, outside the match window.
    rows += [("FFF", -20, 10.0, 5.0, 3.0, None), ("FFF", 20, 10.0, 5.0, 3.0, None)]
    # GGG: a report that carries no scored variable.
    rows += [("GGG", 0, None, None, None, None)]
    # A wind outside 0..75 m/s for one of three reports.
    rows += [("HHH", -30, 10.0, 5.0, 3.0, None), ("HHH", 0, 10.0, 5.0, 200.0, None),
             ("HHH", 30, 10.0, 5.0, 3.0, None)]
    # Stations nobody asks for, interleaved so AAA, BBB and DDD land in
    # different row groups, with incompressible values so the file is far
    # larger than the footer read.
    noise = random.Random(7)
    for station in ("AAB", "CCD", "XAA", "YAA", "ZZZ"):
        rows += [(station, m, noise.uniform(-10, 30), noise.uniform(-20, 0),
                  noise.uniform(0, 30), noise.uniform(950, 1050))
                 for m in range(-FILLER // 2, FILLER // 2)]
    return rows


def _parquet(rows, *, target=TARGET, row_group_size=FILLER) -> bytes:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    position = {"AAA": (52.0, -1.0, 100.0), "BBB": (53.0, -2.0, 50.0)}

    def column(index):
        return [r[index] for r in rows]

    stations = column(0)
    table = pa.table({
        "station": pa.array(stations, pa.large_string()),
        "valid": pa.array([target + timedelta(minutes=r[1]) for r in rows],
                          pa.timestamp("us", tz="UTC")),
        "longitude": [position.get(s, (50.0, 0.5, 10.0))[1] for s in stations],
        "latitude": [position.get(s, (50.0, 0.5, 10.0))[0] for s in stations],
        "tmpf": [None if r[2] is None else r[2] * 9 / 5 + 32 for r in rows],
        "tmpc": pa.array(column(2), pa.float64()),
        "dwpc": pa.array(column(3), pa.float64()),
        "sknt": pa.array(column(4), pa.float64()),
        "mslp": pa.array(column(5), pa.float64()),
        "p01i": [0.0] * len(rows),
        "p01m": [0.0] * len(rows),
        "name": pa.array([f"Station {s}" for s in stations], pa.large_string()),
        "country": pa.array(["GB"] * len(rows), pa.large_string()),
        "elevation": [position.get(s, (50.0, 0.5, 10.0))[2] for s in stations],
    })
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, row_group_size=row_group_size)
    return sink.getvalue().to_pybytes()


class _Archive:
    """A local copy of the archive: ``/year=YYYY/data.parquet`` -> bytes."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.log: list[tuple[str, str, int]] = []      # (method, path, bytes)
        self.refused_user_agent = 0
        self.lock = threading.Lock()
        archive = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self, method):
                if self.headers.get("User-Agent") != dyn.USER_AGENT:
                    with archive.lock:
                        archive.refused_user_agent += 1
                    self.send_response(403)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                with archive.lock:
                    body = archive.files.get(self.path)
                if body is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    with archive.lock:
                        archive.log.append((method, self.path, 0))
                    return
                etag = '"' + hashlib.md5(body).hexdigest() + '"'
                match = self.headers.get("If-Match")
                if match and match != etag:
                    self.send_response(412)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    with archive.lock:
                        archive.log.append((method, self.path, -412))
                    return
                status, payload = 200, body
                wanted = self.headers.get("Range")
                if wanted:
                    found = re.fullmatch(r"bytes=(\d*)-(\d*)", wanted)
                    first, last = found.group(1), found.group(2)
                    if first == "":
                        start, end = len(body) - int(last), len(body) - 1
                    else:
                        start = int(first)
                        end = min(int(last), len(body) - 1) if last else len(body) - 1
                    status, payload = 206, body[start:end + 1]
                self.send_response(status)
                self.send_header("ETag", etag)
                self.send_header("Last-Modified", "Sun, 01 Jun 2025 18:00:00 GMT")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(payload)))
                if status == 206:
                    self.send_header("Content-Range",
                                     f"bytes {start}-{end}/{len(body)}")
                self.end_headers()
                if method == "GET":
                    self.wfile.write(payload)
                with archive.lock:
                    archive.log.append((method, self.path, len(payload) if method == "GET" else 0))

            def do_GET(self):          # noqa: N802 - http.server API
                self._answer("GET")

            def do_HEAD(self):         # noqa: N802 - http.server API
                self._answer("HEAD")

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def gets(self) -> list:
        with self.lock:
            return [entry for entry in self.log if entry[0] == "GET"]

    def heads(self) -> list:
        with self.lock:
            return [entry for entry in self.log if entry[0] == "HEAD"]


@pytest.fixture
def archive(monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")
    origin = _Archive()
    origin.thread.start()
    monkeypatch.setenv(dyn.BASE_URL_ENV, origin.url)
    monkeypatch.setenv(dyn.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv("WOOF_FETCH_LOCK_ROOT", str(tmp_path / "locks"))
    origin.files["/year=2025/data.parquet"] = _parquet(_rows())
    try:
        yield origin
    finally:
        origin.server.shutdown()
        origin.server.server_close()


ASKED = ["AAA", "BBB", "CCC", "DDD", "FFF", "GGG", "HHH"]


def _fetch(tmp_path, **kwargs):
    kwargs.setdefault("station_ids", ASKED)
    return dyn.fetch_surface(None, kwargs.pop("when", TARGET), tmp_path / "out", **kwargs)


def test_the_decode_converts_screens_and_matches_like_rw_asos(archive, tmp_path):
    surface = _fetch(tmp_path)
    record = json.loads(surface.read_text())
    reports = {r["station_id"]: r for r in record["reports"]}

    assert sorted(reports) == ["AAA", "BBB", "DDD"]
    aaa = reports["AAA"]
    assert aaa["valid_time"] == "2025-06-01T12:00:00"
    assert aaa["observation_time"] == "2025-06-01T11:50:00"
    assert aaa["values"] == pytest.approx({
        "temperature_2m": 293.15, "dewpoint_2m": 283.15,
        "wind_speed_10m": 10 * 0.5144444444444445, "mslp": 101320.0})
    assert aaa["flags"] == []
    # The nearer report wins, and a null MSLP is an absent key.
    assert reports["BBB"]["observation_time"] == "2025-06-01T12:10:00"
    assert "mslp" not in reports["BBB"]["values"]
    assert reports["BBB"]["values"]["temperature_2m"] == pytest.approx(289.15)
    # The screened on-the-hour report is gone; the tie goes to the earlier one.
    assert reports["DDD"]["observation_time"] == "2025-06-01T11:55:00"
    assert reports["DDD"]["values"]["dewpoint_2m"] == pytest.approx(278.15)

    screen = record["screen"]
    assert screen["stations_dropped_by_screen"] == ["CCC", "HHH"]
    assert screen["range_drops"] == 2
    assert screen["dewpoint_above_temperature_drops"] == 1
    assert screen["stations_dropped_by_completeness"] == []
    assert (screen["temperature_min_k"], screen["temperature_max_k"]) == (233.15, 328.15)
    assert (screen["wind_min_ms"], screen["wind_max_ms"]) == (0.0, 75.0)
    for report in record["reports"]:
        assert all(isinstance(v, float) for v in report["values"].values())


def test_the_record_has_rw_asos_layout_and_provenance(archive, tmp_path):
    surface = _fetch(tmp_path)
    record = json.loads(surface.read_text())
    fixture = json.loads((Path(__file__).parent / "fixtures" / "asos_surface_real"
                          / "surface_subset.v2.json").read_text())
    assert set(fixture) <= set(record)
    assert set(fixture["screen"]) <= set(record["screen"])
    assert set(fixture["stations"][0]) <= set(record["stations"][0])
    assert set(fixture["reports"][0]) <= set(record["reports"][0])
    assert set(fixture["provenance"]) <= set(record["provenance"])
    assert record["schema"] == "gpuwm-obs.asos-surface.v2"
    assert record["valid_times"] == ["2025-06-01T12:00:00"]
    assert (record["match_seconds"], record["min_report_rate"], record["max_screen_rate"]) \
        == (600, 0.8, 0.05)
    provenance = record["provenance"]
    assert provenance["source"] == dyn.SOURCE
    assert provenance["uri"] == f"{archive.url}/year=2025/data.parquet"
    assert provenance["is_stub"] is False and provenance["stub_reason"] == ""
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["sha256"])
    assert "Iowa Environmental Mesonet" in provenance["attribution"]
    assert [s["station_id"] for s in record["stations"]] == ["AAA", "BBB", "DDD"]
    assert record["stations"][0]["latitude"] == 52.0
    assert record["stations"][0]["elevation_m"] == 100.0

    table = json.loads((surface.parent / "stations.json").read_text())
    assert table["schema"] == "gpuwm-obs.asos-stations.v1"
    assert table["content_sha256"] == record["station_table_sha256"]
    assert table["content_sha256"] == dyn.station_rows_digest(table["stations"])
    assert {s["station_id"] for s in table["stations"]} == set(ASKED)

    from woof.obs.sources import AsosSurfaceSource
    observed = AsosSurfaceSource(surface).observations(record["valid_times"])
    assert len(observed.reports) == 3 and len(observed.stations) == 3
    assert observed.provenance.source == dyn.SOURCE
    assert observed.provenance.sha256 == provenance["sha256"]


def test_only_the_row_groups_holding_the_stations_are_read(archive, tmp_path):
    size = len(archive.files["/year=2025/data.parquet"])
    _fetch(tmp_path, station_ids=["BBB"])
    stats = dyn.last_fetch_stats()
    assert stats["row_groups_read"] == 1
    assert stats["row_groups_total"] >= 5
    downloaded = sum(entry[2] for entry in archive.gets())
    assert downloaded == stats["bytes_downloaded"]
    assert downloaded < size / 2
    assert archive.refused_user_agent == 0


def test_a_repeat_request_moves_no_data_and_refresh_only_revalidates(archive, tmp_path):
    _fetch(tmp_path)
    first = json.loads((tmp_path / "out" / "surface.json").read_text())
    gets, heads = len(archive.gets()), len(archive.heads())
    assert gets > 0 and heads == 1

    _fetch(tmp_path)
    assert (len(archive.gets()), len(archive.heads())) == (gets, heads)
    assert dyn.last_fetch_stats()["bytes_downloaded"] == 0
    assert dyn.last_fetch_stats()["bytes_from_cache"] > 0
    again = json.loads((tmp_path / "out" / "surface.json").read_text())
    assert again["reports"] == first["reports"]
    assert again["provenance"]["sha256"] == first["provenance"]["sha256"]

    _fetch(tmp_path, refresh=True)
    assert (len(archive.gets()), len(archive.heads())) == (gets, heads + 1)


def test_a_rewritten_file_is_read_again(archive, tmp_path, monkeypatch):
    monkeypatch.setattr(dyn, "STALE_VERSION_AGE_S", -1.0)
    _fetch(tmp_path)
    rows = [(s, m, 30.0 if s == "AAA" else t, d, k, p) for s, m, t, d, k, p in _rows()]
    archive.files["/year=2025/data.parquet"] = _parquet(rows)
    # A cached version seen after the hour settled is trusted ...
    _fetch(tmp_path)
    record = json.loads((tmp_path / "out" / "surface.json").read_text())
    assert record["reports"][0]["values"]["temperature_2m"] == pytest.approx(293.15)
    # ... until it is revalidated, and then the new file is read.
    _fetch(tmp_path, refresh=True)
    record = json.loads((tmp_path / "out" / "surface.json").read_text())
    assert record["reports"][0]["values"]["temperature_2m"] == pytest.approx(303.15)
    # The superseded version's cached ranges are gone.
    url_directory = dyn._url_directory(f"{archive.url}/year=2025/data.parquet")
    assert len([path for path in url_directory.iterdir() if path.is_dir()]) == 1


def test_a_file_rewritten_between_reads_is_reread(archive, tmp_path):
    # The cache knows version 1 but holds only BBB's row group; version 2
    # replaces it before AAA is asked for, so the ranged read meets 412.
    _fetch(tmp_path, station_ids=["BBB"])
    rows = [(s, m, 30.0 if s == "AAA" else t, d, k, p) for s, m, t, d, k, p in _rows()]
    archive.files["/year=2025/data.parquet"] = _parquet(rows, row_group_size=FILLER - 10)
    _fetch(tmp_path, station_ids=["AAA"])
    assert any(entry[2] == -412 for entry in archive.log)
    record = json.loads((tmp_path / "out" / "surface.json").read_text())
    assert record["reports"][0]["values"]["temperature_2m"] == pytest.approx(303.15)


def test_requests_without_the_user_agent_are_what_the_host_refuses(archive):
    with pytest.raises(HTTPError) as refused:
        urlopen(Request(f"{archive.url}/year=2025/data.parquet", method="HEAD"), timeout=5)
    assert refused.value.code == 403


def test_no_surviving_report_is_a_lookup_error(archive, tmp_path):
    with pytest.raises(LookupError):
        _fetch(tmp_path, station_ids=["FFF", "GGG"])
    with pytest.raises(LookupError):
        _fetch(tmp_path, station_ids=["NOPE"])
    assert not (tmp_path / "out" / "surface.json").exists()


def test_an_hour_by_new_year_reads_both_year_files(archive, tmp_path):
    new_year = datetime(2025, 1, 1, tzinfo=UTC)
    archive.files["/year=2024/data.parquet"] = _parquet(
        [("AAA", -2, 1.0, 0.0, 3.0, 1000.0), ("AAA", -50, 1.0, 0.0, 3.0, 1000.0)],
        target=new_year)
    archive.files["/year=2025/data.parquet"] = _parquet(
        [("AAA", 8, 2.0, 0.0, 3.0, 1000.0), ("AAA", 50, 2.0, 0.0, 3.0, 1000.0)],
        target=new_year)
    surface = _fetch(tmp_path, station_ids=["AAA"], when=new_year)
    record = json.loads(surface.read_text())
    assert record["reports"][0]["observation_time"] == "2024-12-31T23:58:00"
    assert record["provenance"]["uri"].endswith("/year=2025/data.parquet")
    assert len(record["provenance"]["uris"]) == 2


def test_a_next_year_file_that_does_not_exist_yet_is_skipped(archive, tmp_path):
    late = datetime(2025, 12, 31, 23, 30, tzinfo=UTC)
    archive.files["/year=2025/data.parquet"] = _parquet(
        [("AAA", -3, 1.0, 0.0, 3.0, 1000.0)], target=late)
    record = json.loads(_fetch(tmp_path, station_ids=["AAA"], when=late).read_text())
    assert record["reports"][0]["observation_time"] == "2025-12-31T23:27:00"
    assert record["provenance"]["uris"] == [f"{archive.url}/year=2025/data.parquet"]


def test_an_unserved_year_is_unavailable(archive, tmp_path):
    with pytest.raises(dyn.DynamicalUnavailable):
        _fetch(tmp_path, when=datetime(2030, 6, 1, 12, tzinfo=UTC))


def test_the_bbox_selects_from_the_frozen_table(archive, tmp_path, monkeypatch):
    table = tuple({"station_id": sid, "name": sid, "latitude": 52.0, "longitude": lon,
                   "elevation_m": 1.0, "country": "GB"}
                  for sid, lon in (("AAA", -1.0), ("BBB", -2.0), ("DDD", 30.0)))
    monkeypatch.setattr(dyn, "_station_table", lambda: table)
    surface = dyn.fetch_surface((-3.0, 50.0, 0.0, 54.0), TARGET, tmp_path / "out")
    record = json.loads(surface.read_text())
    assert [r["station_id"] for r in record["reports"]] == ["AAA", "BBB"]


# ---------------------------------------------------------------------------
# The live archive
# ---------------------------------------------------------------------------

@pytest.mark.network
@pytest.mark.skipif(os.environ.get("WOOF_NETWORK_TESTS") != "1",
                    reason="live network smoke; set WOOF_NETWORK_TESTS=1")
def test_the_live_archive_scores_great_britain(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.setenv(dyn.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.delenv(dyn.BASE_URL_ENV, raising=False)
    surface = dyn.fetch_surface((-8.25, 49.85, 1.8, 60.9),
                                datetime(2025, 6, 1, 12, tzinfo=UTC), tmp_path / "gb",
                                timeout=600.0)
    stats = dyn.last_fetch_stats()
    assert stats["row_groups_read"] < stats["row_groups_total"] / 4
    record = json.loads(surface.read_text())
    assert len(record["reports"]) >= 20
    for report in record["reports"]:
        values = report["values"]
        assert 233.15 <= values["temperature_2m"] <= 328.15
        assert 0.0 <= values.get("wind_speed_10m", 0.0) <= 75.0
    from woof.obs.sources import AsosSurfaceSource
    assert AsosSurfaceSource(surface).observations(record["valid_times"]).reports
