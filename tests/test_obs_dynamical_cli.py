"""``woof obs dynamical-asos``: the Dynamical.org ASOS Parquet door.

The reader (:mod:`woof.obs.dynamical_asos`) is a separate unit and may be
absent from the build under test, so every case here injects a fake one
through ``sys.modules`` -- present, absent, and raising -- and drives the
command through the real product parser, which is also what proves it is
registered.  Nothing here touches the network.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import types

import pytest

from woof.cli import build_parser

MODULE = "woof.obs.dynamical_asos"


class _Unavailable(RuntimeError):
    pass


def _fake_reader(*, fetch=None, stations=None):
    module = types.ModuleType(MODULE)
    module.SOURCE = "dynamical-asos-parquet"
    module.BASE_URL_ENV = "WOOF_DYNAMICAL_ASOS_URL"
    module.DynamicalUnavailable = _Unavailable
    module.calls = []

    def stations_in_bbox(west, south, east, north):
        module.calls.append(("stations_in_bbox", (west, south, east, north)))
        if isinstance(stations, BaseException):
            raise stations
        return list(stations or [])

    def station_count_in_bbox(west, south, east, north):
        return len(stations_in_bbox(west, south, east, north))

    def fetch_surface(bbox, valid_time, folder, *, timeout=120.0,
                      refresh=False, station_ids=None):
        module.calls.append(("fetch_surface", dict(
            bbox=bbox, valid_time=valid_time, folder=folder, timeout=timeout,
            refresh=refresh, station_ids=station_ids)))
        if fetch is not None:
            return fetch(bbox, valid_time, folder)
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "stations.json").write_text("{}", encoding="utf-8")
        surface = folder / "surface.json"
        surface.write_text(json.dumps({
            "schema": "gpuwm-obs.asos-surface.v2",
            "stations": [{"id": "EGLL"}, {"id": "EGCC"}],
            "reports": [{}, {}, {}],
        }), encoding="utf-8")
        return surface

    module.stations_in_bbox = stations_in_bbox
    module.station_count_in_bbox = station_count_in_bbox
    module.fetch_surface = fetch_surface
    return module


def _install(monkeypatch, module):
    import woof.obs

    monkeypatch.setitem(sys.modules, MODULE, module)
    if module is None:
        monkeypatch.delattr(woof.obs, "dynamical_asos", raising=False)
    else:
        monkeypatch.setattr(woof.obs, "dynamical_asos", module, raising=False)


def _run(argv):
    namespace = build_parser().parse_args(["obs", "dynamical-asos", *argv])
    return namespace.func(namespace)


UK = "-8.25,49.85,1.8,60.9"


def test_the_command_is_on_the_real_product_parser():
    namespace = build_parser().parse_args(
        ["obs", "dynamical-asos", "--bbox", UK, "--valid-time",
         "2026-10-05T12:00Z", "--out", "x", "--stations", "EGLL, EGCC",
         "--timeout", "30", "--refresh", "--json"])
    assert namespace.func.__name__ == "_dynamical_asos"
    assert namespace.bbox == (-8.25, 49.85, 1.8, 60.9)
    assert namespace.valid_time == datetime(2026, 10, 5, 12,
                                            tzinfo=timezone.utc)
    assert namespace.stations == ["EGLL", "EGCC"]
    assert namespace.timeout == 30.0
    assert namespace.refresh and namespace.json


def test_it_is_not_a_rust_front_door():
    from woof.obs import cli as obs_cli
    from woof.obs import frontdoor

    assert "dynamical-asos" not in obs_cli._INSTRUMENTS  # noqa: SLF001
    assert "dynamical-asos" not in frontdoor.FRONT_DOORS


def test_help_works_without_the_reader(monkeypatch, capsys):
    _install(monkeypatch, None)
    with pytest.raises(SystemExit) as stop:
        build_parser().parse_args(["obs", "dynamical-asos", "--help"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert "--bbox" in out and "--list-stations" in out


@pytest.mark.parametrize("bbox", [
    "1,2,3", "a,b,c,d", "-8,60,1,50", "5,40,5,50", "-200,0,10,10",
    "0,-95,10,10", "nan,0,1,1"])
def test_bad_boxes_are_argument_errors(bbox, capsys):
    with pytest.raises(SystemExit) as stop:
        build_parser().parse_args(["obs", "dynamical-asos", "--bbox", bbox])
    assert stop.value.code == 2


@pytest.mark.parametrize("bbox", ["-8.25,49.85,1.8,60.9",
                                  "-.5,49.85,1.8,60.9"])
def test_a_western_box_parses_with_a_space(bbox):
    """``--bbox -8.25,...`` must not read as an unknown option."""

    namespace = build_parser().parse_args(
        ["obs", "dynamical-asos", "--bbox", bbox, "--list-stations"])
    assert namespace.bbox[0] < 0


def test_a_box_across_the_antimeridian_is_accepted():
    namespace = build_parser().parse_args(
        ["obs", "dynamical-asos", "--bbox", "170,-50,-170,-30",
         "--list-stations"])
    assert namespace.bbox == (170.0, -50.0, -170.0, -30.0)


@pytest.mark.parametrize("text", ["2026-10-05T12:00", "2026-10-05 12:00",
                                  "yesterday"])
def test_a_time_without_a_zone_is_refused(text):
    with pytest.raises(SystemExit) as stop:
        build_parser().parse_args(["obs", "dynamical-asos", "--bbox", UK,
                                   "--valid-time", text])
    assert stop.value.code == 2


def test_an_offset_time_is_normalised_to_utc():
    namespace = build_parser().parse_args(
        ["obs", "dynamical-asos", "--bbox", UK, "--valid-time",
         "2026-10-05T14:00+02:00"])
    assert namespace.valid_time == datetime(2026, 10, 5, 12,
                                            tzinfo=timezone.utc)


def test_absent_reader_refuses_in_one_sentence(monkeypatch, tmp_path, capsys):
    _install(monkeypatch, None)
    code = _run(["--bbox", UK, "--valid-time", "2026-10-05T12:00Z",
                 "--out", str(tmp_path)])
    assert code == 2
    err = capsys.readouterr().err.strip()
    assert err.startswith("woof obs dynamical-asos: ")
    assert "does not include the Dynamical.org ASOS reader" in err
    assert len(err.splitlines()) == 1
    assert not any(tmp_path.iterdir())


def test_a_reader_missing_its_dependency_names_the_extra(monkeypatch,
                                                         tmp_path, capsys):
    import importlib

    real = importlib.import_module

    def import_module(name, package=None):
        if name == MODULE:
            raise ModuleNotFoundError("No module named 'pyarrow'",
                                      name="pyarrow")
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)
    code = _run(["--bbox", UK, "--valid-time", "2026-10-05T12:00Z",
                 "--out", str(tmp_path)])
    assert code == 2
    err = capsys.readouterr().err.strip()
    assert "'pyarrow'" in err and "recast-woof[obs]" in err
    assert len(err.splitlines()) == 1


def test_a_fetch_prints_one_summary_line(monkeypatch, tmp_path, capsys):
    reader = _fake_reader()
    _install(monkeypatch, reader)
    code = _run(["--bbox", UK, "--valid-time", "2026-10-05T12:00Z",
                 "--out", str(tmp_path), "--stations", "EGLL,EGCC",
                 "--timeout", "15", "--refresh"])
    assert code == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert "2 station(s), 3 report(s)" in out[0]
    assert str(tmp_path / "surface.json") in out[0]
    assert "2026-10-05T12:00:00Z" in out[0]
    name, call = reader.calls[-1]
    assert name == "fetch_surface"
    assert call["bbox"] == (-8.25, 49.85, 1.8, 60.9)
    assert call["valid_time"] == datetime(2026, 10, 5, 12,
                                          tzinfo=timezone.utc)
    assert call["folder"] == tmp_path
    assert call["timeout"] == 15.0 and call["refresh"] is True
    assert call["station_ids"] == ["EGLL", "EGCC"]


def test_a_fetch_prints_json_on_request(monkeypatch, tmp_path, capsys):
    _install(monkeypatch, _fake_reader())
    code = _run(["--bbox", UK, "--valid-time", "2026-10-05T12:00Z",
                 "--out", str(tmp_path), "--json"])
    assert code == 0
    record = json.loads(capsys.readouterr().out)
    assert record["schema"] == "gpuwm-obs.dynamical-asos-fetch.v1"
    assert record["source"] == "dynamical-asos-parquet"
    assert record["stations"] == 2 and record["reports"] == 3
    assert record["surface"] == str(tmp_path / "surface.json")
    assert record["valid_time"] == "2026-10-05T12:00:00Z"


def test_no_reports_is_a_refusal_naming_box_and_time(monkeypatch, tmp_path,
                                                     capsys):
    def fetch(bbox, valid_time, folder):
        raise LookupError("0 reports within 30 minutes")

    _install(monkeypatch, _fake_reader(fetch=fetch))
    code = _run(["--bbox", UK, "--valid-time", "2026-10-05T12:00Z",
                 "--out", str(tmp_path)])
    assert code == 2
    err = capsys.readouterr().err.strip()
    assert "-8.25,49.85,1.8,60.9" in err
    assert "2026-10-05T12:00:00Z" in err
    assert "0 reports within 30 minutes" in err


@pytest.mark.parametrize("error, phrase", [
    (_Unavailable("pyarrow is not installed"), "unavailable"),
    (ImportError("No module named 'pyarrow'"), "recast-woof[obs]"),
    (OSError("disk full"), "failed"),
    (ValueError("Parquet magic bytes not found"), "--refresh"),
])
def test_an_unavailable_archive_is_a_refusal(monkeypatch, tmp_path, capsys,
                                             error, phrase):
    def fetch(bbox, valid_time, folder):
        raise error

    _install(monkeypatch, _fake_reader(fetch=fetch))
    code = _run(["--bbox", UK, "--valid-time", "2026-10-05T12:00Z",
                 "--out", str(tmp_path)])
    assert code == 2
    err = capsys.readouterr().err.strip()
    assert phrase in err and str(error) in err
    assert len(err.splitlines()) == 1


def test_a_fetch_needs_a_time_and_a_folder(monkeypatch, capsys):
    reader = _fake_reader()
    _install(monkeypatch, reader)
    assert _run(["--bbox", UK]) == 2
    err = capsys.readouterr().err
    assert "--valid-time and --out" in err
    assert not reader.calls


STATIONS = [
    {"station_id": "EGLL", "name": "London Heathrow", "latitude": 51.4775,
     "longitude": -0.4614, "elevation_m": 25.0, "country": "GB"},
    {"station_id": "EGPH", "name": "Edinburgh", "latitude": 55.95,
     "longitude": -3.3725, "elevation_m": 41.0, "country": "GB"},
]


def test_list_stations_reads_the_table_only(monkeypatch, capsys):
    reader = _fake_reader(stations=STATIONS)
    _install(monkeypatch, reader)
    assert _run(["--bbox", UK, "--list-stations"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "2 station(s)" in out[0]
    assert any("EGLL" in line and "London Heathrow" in line for line in out)
    assert [name for name, _ in reader.calls] == ["stations_in_bbox"]


def test_list_stations_json(monkeypatch, capsys):
    _install(monkeypatch, _fake_reader(stations=STATIONS))
    assert _run(["--bbox", "170,-50,-170,-30", "--list-stations",
                 "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["count"] == 2
    assert record["bbox"] == [170.0, -50.0, -170.0, -30.0]
    assert record["stations"][0]["station_id"] == "EGLL"


def test_an_unreadable_table_is_a_refusal(monkeypatch, capsys):
    _install(monkeypatch, _fake_reader(
        stations=FileNotFoundError("dynamical_asos_stations.json")))
    assert _run(["--bbox", UK, "--list-stations"]) == 2
    assert "station table could not be read" in capsys.readouterr().err


def test_a_reader_bug_is_not_reported_as_no_data(monkeypatch, tmp_path):
    def fetch(bbox, valid_time, folder):
        raise KeyError("tmpf")

    _install(monkeypatch, _fake_reader(fetch=fetch))
    with pytest.raises(KeyError):
        _run(["--bbox", UK, "--valid-time", "2026-10-05T12:00Z",
              "--out", str(tmp_path)])


def test_list_stations_honours_stations(monkeypatch, capsys):
    _install(monkeypatch, _fake_reader(stations=STATIONS))
    assert _run(["--bbox", UK, "--list-stations", "--stations", "EGPH",
                 "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert [row["station_id"] for row in record["stations"]] == ["EGPH"]
