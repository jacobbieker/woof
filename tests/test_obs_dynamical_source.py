"""The Dynamical.org ASOS Parquet archive as a station observation source.

The reader that talks to the archive (``woof.obs.dynamical_asos``) is
optional and lands separately, so every test here injects a stand-in module
whose ``fetch_surface`` writes the committed real ASOS subset, re-tagged with
the Dynamical provenance, exactly where the real reader would.  What is
pinned is everything on this side of that call:

* the source satisfies the scorer's ``StationObsSource`` protocol, parses the
  records with the same function the IEM records go through, and carries the
  archive's name and attribution on its provenance;
* the battery and the WOOF Global scorecard score against it, and the
  battery's re-hash routes back to it and covers every hourly record;
* a missing reader, a missing dependency or an unreachable archive is a
  typed :class:`ObsSourceUnavailable`, never a quiet substitute;
* the global stream registry lists it as a verification stream.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from woof.obs import sources
from woof.verify.obs import battery, registration as reg_mod
from woof.verify.obs.contracts import (
    ModelGrid, ObsProvenance, StationObsSource,
)
from woof.verify.obs.stations import StationPosition, freeze_station_set

FIXTURE = (Path(__file__).parent / "fixtures" / "asos_surface_real"
           / "surface_subset.v2.json")
MODULE = "woof.obs.dynamical_asos"
HOURS = ("2024-05-21T12:00:00", "2024-05-21T13:00:00", "2024-05-21T14:00:00")
INIT = "2024-05-21T10:00:00"
LEADS = (2, 3, 4)
IOWA = (-95.0, 40.5, -92.0, 43.0)
UPSTREAM_SHA = "ab" * 32


class _Unavailable(RuntimeError):
    pass


def _fake_reader(*, empty=(), unavailable=False, source=None, error=None,
                 fetched_at="2026-10-07T09:15:00Z"):
    """A stand-in ``woof.obs.dynamical_asos`` that writes the real subset."""

    module = types.ModuleType(MODULE)
    module.SOURCE = sources.DYNAMICAL_SOURCE
    module.DynamicalUnavailable = _Unavailable
    module.calls = []

    def fetch_surface(bbox, instant, folder, *, timeout=120.0,
                      refresh=False, station_ids=None):
        # The real reader takes an aware UTC datetime and refuses a string.
        assert isinstance(instant, datetime) and instant.utcoffset() == \
            timedelta(0), instant
        valid_time = instant.strftime("%Y-%m-%dT%H:%M:%S")
        if bbox is None and not station_ids:
            raise ValueError("fetch_surface needs a bbox or station_ids")
        module.calls.append({"bbox": None if bbox is None else tuple(bbox),
                             "valid_time": valid_time,
                             "folder": Path(folder), "timeout": timeout,
                             "refresh": refresh, "station_ids": station_ids})
        if unavailable:
            raise _Unavailable("data.source.coop did not answer")
        if error is not None:
            raise error
        if valid_time in empty:
            raise LookupError(f"no report survived at {valid_time}")
        record = json.loads(FIXTURE.read_text())
        record["provenance"] = {
            "source": source or sources.DYNAMICAL_SOURCE,
            "product": "asos-parquet",
            "uri": "https://data.source.coop/dynamical/asos-parquet/"
                   "year=2024/data.parquet",
            "sha256": UPSTREAM_SHA,
            "fetched_at": fetched_at,
            "is_stub": False, "stub_reason": "",
        }
        record["reports"] = [row for row in record["reports"]
                             if row["valid_time"] == valid_time]
        if station_ids:
            keep = set(station_ids)
            record["stations"] = [s for s in record["stations"]
                                  if s["station_id"] in keep]
            record["reports"] = [r for r in record["reports"]
                                 if r["station_id"] in keep]
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "stations.json").write_text(
            json.dumps({"stations": record["stations"]}))
        path = folder / "surface.json"
        path.write_text(json.dumps(record, indent=1))
        return path

    module.fetch_surface = fetch_surface
    return module


@pytest.fixture
def reader(monkeypatch):
    module = _fake_reader()
    monkeypatch.setitem(sys.modules, MODULE, module)
    return module


def _source(tmp_path, **kwargs):
    kwargs.setdefault("bbox", IOWA)
    return sources.DynamicalAsosSurfaceSource(tmp_path / "dyn", **kwargs)


# ------------------------------------------------------------ the source


def test_the_source_merges_one_fetch_per_valid_time_into_one_set(
        tmp_path, reader):
    source = _source(tmp_path, timeout=30.0)
    assert isinstance(source, StationObsSource)

    observations = source.observations(list(reversed(HOURS)))

    assert [call["valid_time"] for call in reader.calls] == list(HOURS)
    assert len({call["folder"] for call in reader.calls}) == 3
    assert all(call["bbox"] == IOWA and call["timeout"] == 30.0
               and call["station_ids"] is None for call in reader.calls)
    assert len(observations.stations) == 8
    assert len(observations.reports) == 24
    assert {r.valid_time for r in observations.reports} == set(HOURS)
    assert source.empty_valid_times == ()
    assert sorted(source.records) == list(HOURS)


def test_the_reports_are_the_ones_the_iem_record_reader_returns(
        tmp_path, reader):
    """One parser for both archives: the same record reads the same."""

    dynamical = _source(tmp_path).observations(HOURS)
    iem = sources.AsosSurfaceSource(FIXTURE).observations(HOURS)

    def table(obs):
        return sorted((r.station_id, r.valid_time, tuple(sorted(
            r.values.items())), r.flags) for r in obs.reports)

    assert table(dynamical) == table(iem)
    assert ({(s.station_id, s.latitude, s.longitude, s.elevation_m)
             for s in dynamical.stations}
            == {(s.station_id, s.latitude, s.longitude, s.elevation_m)
                for s in iem.stations})


def test_the_provenance_names_the_archive_and_carries_its_attribution(
        tmp_path, reader):
    source = _source(tmp_path)
    provenance = source.observations(HOURS).provenance

    assert provenance.source == "dynamical-asos-parquet"
    assert provenance.product == "asos-parquet"
    assert provenance.is_stub is False
    assert provenance.fetched_at == "2026-10-07T09:15:00"
    assert "Iowa Environmental Mesonet" in provenance.attribution
    assert "dynamical.org" in provenance.attribution
    assert "Source Cooperative" in provenance.attribution
    record = provenance.record()
    assert record["attribution"] == provenance.attribution
    assert Path(provenance.uri) == source.manifest_path.resolve()

    manifest = json.loads(source.manifest_path.read_text())
    assert manifest["schema"] == sources.DYNAMICAL_MANIFEST_SCHEMA
    assert [m["valid_time"] for m in manifest["members"]] == list(HOURS)
    assert all(m["provenance"]["sha256"] == UPSTREAM_SHA
               for m in manifest["members"])
    assert all(not Path(m["path"]).is_absolute()
               for m in manifest["members"])


def test_the_rehash_covers_the_manifest_and_every_hourly_record(
        tmp_path, reader):
    source = _source(tmp_path)
    provenance = source.observations(HOURS).provenance
    assert source.verify(provenance) is True

    changed = source.records[HOURS[1]]
    changed.write_text(changed.read_text().replace("AMW", "AMX", 1))
    assert source.verify(provenance) is False

    changed.unlink()
    with pytest.raises(FileNotFoundError, match="not on disk"):
        source.verify(provenance)


def test_a_moved_working_folder_verifies_under_its_root(tmp_path, reader):
    source = _source(tmp_path)
    provenance = source.observations(HOURS).provenance
    moved = tmp_path / "moved"
    (tmp_path / "dyn").rename(moved)

    with pytest.raises(FileNotFoundError):
        source.verify(provenance)
    assert source.verify(provenance, root=moved) is True


def test_a_station_list_is_passed_through_and_honoured(tmp_path, reader):
    source = _source(tmp_path, bbox=None, station_ids=["AMW", " DSM ", ""])
    observations = source.observations(HOURS[:1])

    # ids alone: the reader selects by id and is handed no box
    assert reader.calls[0]["station_ids"] == ["AMW", "DSM"]
    assert reader.calls[0]["bbox"] is None
    manifest = json.loads(source.manifest_path.read_text())
    assert manifest["bbox"] is None
    assert manifest["station_ids"] == ["AMW", "DSM"]
    assert sorted(s.station_id for s in observations.stations) == [
        "AMW", "DSM"]


def test_an_hour_with_no_surviving_report_is_missing_not_fatal(
        tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, MODULE,
                        _fake_reader(empty={HOURS[1]}))
    source = _source(tmp_path)
    observations = source.observations(HOURS)

    assert source.empty_valid_times == (HOURS[1],)
    assert {r.valid_time for r in observations.reports} == {HOURS[0],
                                                            HOURS[2]}
    manifest = json.loads(source.manifest_path.read_text())
    assert manifest["empty_valid_times"] == [HOURS[1]]


def test_no_surviving_report_at_any_hour_is_a_lookup_error(
        tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, MODULE, _fake_reader(empty=set(HOURS)))
    with pytest.raises(LookupError, match="no report"):
        _source(tmp_path).observations(HOURS)


def test_a_record_from_another_archive_is_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, MODULE, _fake_reader(source="asos"))
    with pytest.raises(ValueError, match="another archive"):
        _source(tmp_path).observations(HOURS)


# ----------------------------------------------------------- unavailable


def test_a_missing_reader_is_a_typed_unavailable_source(
        tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, MODULE, None)
    source = _source(tmp_path)  # constructing needs nothing optional

    with pytest.raises(sources.ObsSourceUnavailable) as caught:
        source.observations(HOURS)

    assert caught.value.source == "dynamical-asos-parquet"
    assert "pyarrow" in caught.value.reason
    assert isinstance(caught.value.__cause__, ImportError)
    assert isinstance(caught.value, RuntimeError)
    assert not (tmp_path / "dyn").exists() or not any(
        (tmp_path / "dyn").glob("*.manifest.json"))


def test_an_unreachable_archive_is_a_typed_unavailable_source(
        tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, MODULE, _fake_reader(unavailable=True))
    with pytest.raises(sources.ObsSourceUnavailable,
                       match="did not answer") as caught:
        _source(tmp_path).observations(HOURS)
    assert isinstance(caught.value.__cause__, _Unavailable)


# ------------------------------------------------------- construction


def test_the_source_needs_a_box_or_stations_and_checks_both(tmp_path):
    with pytest.raises(ValueError, match="bounding box or a list"):
        sources.DynamicalAsosSurfaceSource(tmp_path)
    with pytest.raises(ValueError, match="south < north"):
        sources.DynamicalAsosSurfaceSource(tmp_path, bbox=(0, 10, 1, 5))
    with pytest.raises(ValueError, match="apart"):
        sources.DynamicalAsosSurfaceSource(tmp_path, bbox=(10, 0, 10, 5))
    with pytest.raises(ValueError, match="in \\[-180, 180\\]"):
        sources.DynamicalAsosSurfaceSource(tmp_path, bbox=(0, 0, 200, 5))
    # west > east is the reader's box across the antimeridian
    across = sources.DynamicalAsosSurfaceSource(tmp_path,
                                                bbox=(170, 50, -170, 66))
    assert across.bbox == (170.0, 50.0, -170.0, 66.0)
    with pytest.raises(ValueError, match="empty"):
        sources.DynamicalAsosSurfaceSource(tmp_path, station_ids=[" "])
    with pytest.raises(ValueError, match="timeout"):
        sources.DynamicalAsosSurfaceSource(tmp_path, bbox=IOWA, timeout=0)


def test_sources_are_selected_by_name(tmp_path):
    assert sources.STATION_SOURCES == ("asos", "dynamical-asos")
    assert isinstance(sources.station_obs_source("asos", record=FIXTURE),
                      sources.AsosSurfaceSource)
    chosen = sources.station_obs_source(
        "dynamical-asos", folder=tmp_path, bbox=IOWA, timeout=5.0,
        refresh=True)
    assert isinstance(chosen, sources.DynamicalAsosSurfaceSource)
    assert chosen.refresh is True and chosen.timeout == 5.0
    with pytest.raises(ValueError, match="unknown station source"):
        sources.station_obs_source("madis", folder=tmp_path)
    with pytest.raises(ValueError, match="working folder"):
        sources.station_obs_source("dynamical-asos", bbox=IOWA)


def test_boxes_parse_and_wrap():
    assert sources.parse_bbox("-95, 40.5,-92,43") == IOWA
    with pytest.raises(ValueError, match="W,S,E,N"):
        sources.parse_bbox("1,2,3")
    lat = np.array([50.0, 52.0])
    lon = np.array([358.0, 2.0])  # a grid across Greenwich in 0..360
    assert sources.bbox_of(lat, lon, margin_deg=0.5) == (-2.5, 49.5, 2.5,
                                                         52.5)
    # a global grid asks for every longitude; a regional one across the
    # antimeridian gets the reader's crossing box, never the wrong half
    assert sources.bbox_of(np.array([-90.0, 90.0]),
                           np.arange(0.0, 360.0, 1.0)) == (
        -180.0, -90.0, 180.0, 90.0)
    assert sources.bbox_of(np.array([60.0, 66.0]),
                           np.array([175.0, 180.0, 185.0]),
                           margin_deg=1.0) == (174.0, 59.0, -174.0, 67.0)


def test_the_attribution_is_optional_and_absent_from_old_records():
    plain = ObsProvenance(source="asos", product="iem-asos-metar",
                          uri="x", sha256="0" * 64,
                          fetched_at="2024-05-21T12:00:00")
    assert "attribution" not in plain.record()
    assert sorted(plain.record()) == sorted(
        ["source", "product", "uri", "sha256", "fetched_at", "is_stub",
         "stub_reason"])


def test_one_dynamical_hour_scores_through_the_iem_record_reader(
        tmp_path, reader):
    """``--surface-source asos --asos-surface <one Dynamical hour>``: the
    record itself is the re-hashable object, the UTC stamp is seam-spelled,
    and the attribution comes along."""

    path = reader.fetch_surface(
        IOWA, datetime.fromisoformat(HOURS[0] + "+00:00"), tmp_path / "one")
    source = sources.AsosSurfaceSource(path)
    provenance = source.observations(HOURS[:1]).provenance

    assert provenance.source == sources.DYNAMICAL_SOURCE
    assert provenance.attribution == sources.DYNAMICAL_ATTRIBUTION
    assert provenance.fetched_at == "2026-10-07T09:15:00"
    assert Path(provenance.uri) == path.resolve()
    assert source.verify(provenance) is True
    path.write_text(path.read_text().replace("AMW", "AMX", 1))
    assert source.verify(provenance) is False


def test_a_refresh_never_breaks_an_earlier_score_files_rehash(
        tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, MODULE, _fake_reader())
    first = _source(tmp_path)
    earlier = first.observations(HOURS).provenance

    monkeypatch.setitem(sys.modules, MODULE, _fake_reader(
        fetched_at="2026-10-07T10:15:00Z"))
    second = _source(tmp_path, refresh=True)
    later = second.observations(HOURS).provenance

    assert later.uri != earlier.uri and later.sha256 != earlier.sha256
    assert second.verify(earlier) is True
    assert second.verify(later) is True


def test_a_reader_bug_is_not_scored_as_a_missing_hour(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, MODULE,
                        _fake_reader(error=KeyError("dwpc")))
    with pytest.raises(KeyError, match="dwpc"):
        _source(tmp_path).observations(HOURS)


@pytest.mark.parametrize("error", [
    TimeoutError("read timed out"), ConnectionRefusedError("refused"),
    ImportError("No module named 'pyarrow'")])
def test_network_and_dependency_failures_are_an_unavailable_source(
        tmp_path, monkeypatch, error):
    monkeypatch.setitem(sys.modules, MODULE, _fake_reader(error=error))
    with pytest.raises(sources.ObsSourceUnavailable,
                       match=type(error).__name__) as caught:
        _source(tmp_path).observations(HOURS)
    assert caught.value.__cause__ is error


# ------------------------------------------------------ the battery path


class IowaArm:
    """A forecast arm over central Iowa, constant surface fields."""

    def __init__(self, *, offset=0.0, spacing=0.03):
        lat = np.arange(40.0, 43.0 + 1e-9, spacing)
        lon = np.arange(-95.2, -91.6 + 1e-9, spacing)
        self.lon2, self.lat2 = np.meshgrid(lon, lat)
        self.spacing = spacing
        self.offset = float(offset)
        self._grid = ModelGrid(latitude=self.lat2, longitude=self.lon2,
                               dx_m=3000.0,
                               terrain_m=np.full(self.lat2.shape, 300.0))

    def grid(self):
        return self._grid

    def land_mask(self):
        return np.ones(self.lat2.shape, dtype=bool)

    def station_locator(self):
        def locate(station_id, latitude, longitude):
            return StationPosition(
                str(station_id),
                (float(longitude) - float(self.lon2[0, 0])) / self.spacing,
                (float(latitude) - float(self.lat2[0, 0])) / self.spacing)
        return locate

    def surface_field(self, valid_time, variable):
        base = {"temperature_2m": 290.0, "dewpoint_2m": 285.0,
                "wind_speed_10m": 4.0}[variable]
        return np.full(self.lat2.shape, base + self.offset)

    def record(self):
        return {"reader": "Iowa test arm"}


def _registration():
    return reg_mod.make_registration(
        evaluator_commit="3" * 40,
        reflectivity=reg_mod.reflectivity_parameters(),
        surface=reg_mod.surface_parameters(),
        precipitation=reg_mod.precipitation_parameters(),
        promotion=reg_mod.promotion_parameters(),
        cases=[{"case_id": "iowa-fixture", "init_time": INIT}],
        arms=[{"arm_id": "faithful"}],
        twin={"rung": 1},
        scored_lead_hours_=LEADS)


def test_the_battery_scores_the_surface_against_the_dynamical_source(
        tmp_path, reader):
    arm = IowaArm(offset=1.0)
    source = _source(tmp_path)
    observations = source.observations(battery.valid_times(INIT, LEADS))
    grid = arm.grid()
    locate = arm.station_locator()
    frozen = freeze_station_set(
        observations.stations,
        {s.station_id: locate(s.station_id, s.latitude, s.longitude)
         for s in observations.stations},
        observations=observations, valid_times=list(HOURS),
        interior_mask=np.ones(grid.shape, dtype=bool),
        land_mask=arm.land_mask(), terrain_m=grid.terrain_m,
        elevation_tolerance_m=100.0, minimum_reporting_fraction=0.8,
        match_tolerance_seconds=600, maximum_screen_fraction=0.05)
    assert len(frozen.station_ids) == 8

    collected: list[ObsProvenance] = []
    surface = battery.score_surface(
        registration=_registration(), model=arm, observations=observations,
        frozen=frozen, init_time=INIT, lead_hours=LEADS,
        collected_provenance=collected)

    assert collected == [observations.provenance]
    text = json.dumps(surface, default=str)
    assert "temperature_2m" in text
    assert collected[0].source == sources.DYNAMICAL_SOURCE


def _cli():
    return importlib.import_module("tools.obs_battery_score")


def _cli_argv(tmp_path, *extra):
    document = _registration()
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    packs = tmp_path / "packs"
    (packs / "packs").mkdir(parents=True, exist_ok=True)
    return ["obs_battery_score.py",
            "--run-directory", str(tmp_path / "run"),
            "--case-id", "iowa-fixture", "--arm-id", "faithful",
            "--init-time", INIT,
            "--reflectivity-packs", str(packs),
            "--boundary-width-cells", "5",
            "--registration", str(path), *extra]


def _stand_in_cli(monkeypatch, cli, arm, captured):
    def fake_score(**kwargs):
        captured.update(kwargs)
        captured["surface"] = battery.score_surface(
            registration=kwargs["registration"], model=kwargs["model"],
            observations=kwargs["station_obs"],
            frozen=kwargs["frozen_stations"],
            init_time=kwargs["init_time"], lead_hours=LEADS,
            collected_provenance=[])
        return {"reflectivity": {"primary_scalar": 0.5}}

    monkeypatch.setattr(cli.model_source, "WrfHistorySource",
                        lambda *a, **k: arm)
    monkeypatch.setattr(cli, "_packs",
                        lambda directory, pattern="*.obspack": (["frame"],
                                                                "geometry"))
    monkeypatch.setattr(cli, "MrmsCompositeSource",
                        lambda frames, geometry, **kwargs: "radar-source")
    monkeypatch.setattr(cli.battery, "score_case_arm", fake_score)


def test_the_scoring_command_selects_the_dynamical_source_by_name(
        monkeypatch, tmp_path, reader, capsys):
    cli = _cli()
    arm = IowaArm(offset=0.5)
    captured: dict[str, object] = {}
    _stand_in_cli(monkeypatch, cli, arm, captured)
    folder = tmp_path / "dyn-cli"
    monkeypatch.setattr("sys.argv", _cli_argv(
        tmp_path, "--surface-source", "dynamical-asos",
        "--dynamical-folder", str(folder)))

    assert cli.main() == 0

    station_obs = captured["station_obs"]
    assert station_obs.provenance.source == sources.DYNAMICAL_SOURCE
    assert len(captured["frozen_stations"].station_ids) == 8
    # the box defaults to the arm's own grid plus the margin
    west, south, east, north = reader.calls[0]["bbox"]
    assert (west, south) == pytest.approx((-95.7, 39.5))
    assert (east, north) == pytest.approx((-91.1, 43.5))
    assert [c["valid_time"] for c in reader.calls] == list(HOURS)
    # the re-hash routes the surface record back to the source that read it
    assert captured["rehash"](station_obs.provenance) is True
    out = capsys.readouterr().out
    assert "dynamical-asos-parquet" in out
    assert "Iowa Environmental Mesonet" in out
    assert captured["surface"]


def test_the_scoring_command_exits_four_when_the_reader_is_missing(
        monkeypatch, tmp_path, capsys):
    cli = _cli()
    monkeypatch.setitem(sys.modules, MODULE, None)
    captured: dict[str, object] = {}
    _stand_in_cli(monkeypatch, cli, IowaArm(), captured)
    monkeypatch.setattr("sys.argv", _cli_argv(
        tmp_path, "--surface-source", "dynamical-asos",
        "--dynamical-folder", str(tmp_path / "dyn"),
        "--dynamical-bbox=-95,40.5,-92,43"))

    assert cli.main() == cli.EXIT_SOURCE_UNAVAILABLE == 4
    assert "score_case_arm" not in captured and not captured
    assert "dynamical-asos-parquet" in capsys.readouterr().err


def test_the_scoring_command_needs_a_folder_for_the_dynamical_source(
        monkeypatch, tmp_path, capsys):
    cli = _cli()
    _stand_in_cli(monkeypatch, cli, IowaArm(), {})
    monkeypatch.setattr("sys.argv", _cli_argv(
        tmp_path, "--surface-source", "dynamical-asos"))
    with pytest.raises(SystemExit) as exit_code:
        cli.main()
    assert exit_code.value.code == 2
    assert "--dynamical-folder" in capsys.readouterr().err


def test_the_scoring_command_refuses_the_iem_station_table_for_dynamical(
        monkeypatch, tmp_path, capsys):
    cli = _cli()
    _stand_in_cli(monkeypatch, cli, IowaArm(), {})
    table = tmp_path / "stations.json"
    table.write_text(json.dumps({"content_sha256": "e" * 64}))
    monkeypatch.setattr("sys.argv", _cli_argv(
        tmp_path, "--surface-source", "dynamical-asos",
        "--dynamical-folder", str(tmp_path / "dyn"),
        "--asos-stations", str(table)))
    with pytest.raises(SystemExit) as exit_code:
        cli.main()
    assert exit_code.value.code == 2
    assert "--asos-stations" in capsys.readouterr().err


def test_the_scoring_command_refuses_an_archive_with_nothing_to_score(
        monkeypatch, tmp_path, capsys):
    cli = _cli()
    monkeypatch.setitem(sys.modules, MODULE, _fake_reader(empty=set(HOURS)))
    captured: dict[str, object] = {}
    _stand_in_cli(monkeypatch, cli, IowaArm(), captured)
    monkeypatch.setattr("sys.argv", _cli_argv(
        tmp_path, "--surface-source", "dynamical-asos",
        "--dynamical-folder", str(tmp_path / "dyn"),
        "--dynamical-station-ids", "AMW,DSM"))

    assert cli.main() == 1
    assert not captured
    assert "no station observations to score" in capsys.readouterr().err


# ------------------------------------------------- the global scorecard


def _global_surface_fields(offset=0.0):
    """A one-degree global model with constant fields: the scorecard reads
    global grids only."""
    from woof.globe.obs_scorecard import SurfaceFields

    lat = np.arange(-90.0, 90.01, 1.0)
    lon = np.arange(0.0, 360.0, 1.0)
    shape = (lat.size, lon.size)
    return SurfaceFields(
        latitude_deg=lat, longitude_deg=lon,
        t2_k=np.full(shape, 291.0 + offset),
        q2_kg_kg=np.full(shape, 0.010),
        u10_m_s=np.full(shape, 3.0), v10_m_s=np.full(shape, 4.0),
        surface_pressure_pa=np.full(shape, 98000.0),
        terrain_m=np.full(shape, 300.0),
        land=np.ones(shape, dtype=bool), source={"kind": "test"})


def test_the_global_scorecard_scores_against_the_dynamical_source(
        tmp_path, reader):
    from woof.globe import obs_scorecard as card

    models = {"woof": _global_surface_fields(), "control":
              _global_surface_fields(offset=1.0)}
    observations, record = card.surface_observations(
        "dynamical-asos", models, HOURS[0], folder=tmp_path / "card")
    result = card.score_surface_stations(models, observations, HOURS[0])

    # a global model asks the whole archive
    assert reader.calls[0]["bbox"] == (-180.0, -90.0, 180.0, 90.0)
    assert Path(record).name.endswith(".manifest.json")
    provenance = result["observations"]["provenance"]
    assert provenance["source"] == "dynamical-asos-parquet"
    assert "dynamical.org" in provenance["attribution"]
    assert result["common_station_count"] == 8
    t2 = result["scores"]["woof"]["temperature_2m"]
    assert t2["n"] == 8 and t2["mae"] is not None
    assert (result["scores"]["control"]["temperature_2m"]["bias"]
            == pytest.approx(t2["bias"] + 1.0))


def test_the_global_scorecard_keeps_the_iem_record_by_default(tmp_path):
    from woof.globe import obs_scorecard as card

    assert card.SURFACE_OBS_SOURCES == sources.STATION_SOURCES
    assert card.EXIT_SOURCE_UNAVAILABLE == sources.SOURCE_UNAVAILABLE_EXIT
    observations, record = card.surface_observations(
        "asos", {}, HOURS[0], record=FIXTURE)
    assert record == str(FIXTURE)
    assert observations.provenance.source == "asos"
    with pytest.raises(ValueError, match="unknown surface observation"):
        card.surface_observations("synop", {}, HOURS[0])


def test_the_global_scorecard_source_is_unavailable_without_the_reader(
        tmp_path, monkeypatch):
    from woof.globe import obs_scorecard as card

    monkeypatch.setitem(sys.modules, MODULE, None)
    with pytest.raises(sources.ObsSourceUnavailable):
        card.surface_observations(
            "dynamical-asos", {"woof": _global_surface_fields()}, HOURS[0],
            folder=tmp_path / "card")


@pytest.mark.parametrize("extra, message", [
    ((), "needs --obs"),
    (("--obs-source", "dynamical-asos", "--bbox=0,6,1,5"), "south < north"),
])
def test_the_scorecard_refuses_a_bad_request_before_reading_a_model(
        tmp_path, monkeypatch, capsys, extra, message):
    from woof.globe import obs_scorecard as card

    def no_model_reads(*args, **kwargs):
        raise AssertionError("a model was read before the request was checked")

    monkeypatch.setattr(card, "_phi_for", no_model_reads)
    with pytest.raises(SystemExit) as exit_code:
        card.main(["surface", "--model", f"woof={tmp_path}:ck",
                   "--valid", HOURS[0], "--out", str(tmp_path / "o.json"),
                   *extra])
    assert exit_code.value.code == 2
    assert message in capsys.readouterr().err


# ------------------------------------------------- the stream registry


def test_the_stream_registry_lists_dynamical_asos_as_verification():
    from woof.globe import obs_streams

    spec = obs_streams.STREAMS["dynamical-asos"]
    assert spec.verification is True
    assert spec.door is None and spec.decoder_built is False
    assert spec.public is True and spec.account_gated is False
    assert spec.latency_class == "fast" and spec.cadence_s == 3600
    assert "30 to 60 minutes" in spec.latency_basis
    assert "14 other countries" in spec.notes
    assert "Iowa Environmental Mesonet" in spec.notes
    assert spec.variables == obs_streams.STREAMS["iem-metar"].variables
    assert spec not in obs_streams.fetchable_streams()
    assert obs_streams.STREAMS["iem-metar"].verification is False
    rows = {row["stream"]: row for row in obs_streams.streams_table()}
    assert rows["dynamical-asos"]["verification"] is True
    with pytest.raises(ValueError, match="no decoding door"):
        obs_streams.fetch_stream("dynamical-asos", "2024-05-21T12:00:00Z",
                                 "2024-05-21T13:00:00Z", Path("unused"))


def test_fetching_the_verification_stream_names_the_scoring_commands(
        tmp_path, capsys):
    from woof.globe import obs_streams

    code = obs_streams.main(["fetch", "--stream", "dynamical-asos",
                             "--start", "2024-05-21T12:00:00Z",
                             "--end", "2024-05-21T13:00:00Z",
                             "--out", str(tmp_path)])
    assert code == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "NO_DECODER"
    assert "--surface-source dynamical-asos" in captured.err
    assert "--obs-source dynamical-asos" in captured.err
