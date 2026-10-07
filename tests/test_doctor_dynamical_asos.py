"""``woof doctor``'s row for the Dynamical.org ASOS Parquet archive.

The reader module is a separate unit and may be absent, so each case
injects a fake :mod:`woof.obs.dynamical_asos` (present, absent, raising)
and forces the PyArrow probe.  Every case also forbids sockets: the row is
local evidence only and must never contact the archive host.
"""

from __future__ import annotations

import socket
import sys
import types

import pytest

from woof import doctor

MODULE = "woof.obs.dynamical_asos"

STATIONS = [
    {"station_id": "EGLL", "name": "London Heathrow", "latitude": 51.48,
     "longitude": -0.46, "elevation_m": 25.0, "country": "GB"},
    {"station_id": "KORD", "name": "Chicago O'Hare", "latitude": 41.98,
     "longitude": -87.9, "elevation_m": 205.0, "country": "US"},
    {"station_id": "RJTT", "name": "Tokyo Haneda", "latitude": 35.55,
     "longitude": 139.78, "elevation_m": 6.0, "country": "JP"},
]


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    def refuse(*args, **kwargs):
        raise AssertionError("the doctor row tried to use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setenv("WOOF_DYNAMICAL_ASOS_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("WOOF_DYNAMICAL_ASOS_URL", raising=False)


def _reader(stations=STATIONS):
    module = types.ModuleType(MODULE)
    module.__file__ = doctor.__file__.replace("doctor.py",
                                              "obs/dynamical_asos.py")
    module.SOURCE = "dynamical-asos-parquet"
    module.BASE_URL_ENV = "WOOF_DYNAMICAL_ASOS_URL"
    module.DynamicalUnavailable = type("DynamicalUnavailable",
                                       (RuntimeError,), {})

    def stations_in_bbox(west, south, east, north):
        if isinstance(stations, BaseException):
            raise stations
        return list(stations)

    module.stations_in_bbox = stations_in_bbox
    module.station_count_in_bbox = (
        lambda *bbox: len(stations_in_bbox(*bbox)))
    return module


def _check(monkeypatch, module, *, arrow=(True, "17.0.0")):
    import woof.obs

    monkeypatch.setitem(sys.modules, MODULE, module)
    if module is None:
        monkeypatch.delattr(woof.obs, "dynamical_asos", raising=False)
    else:
        monkeypatch.setattr(woof.obs, "dynamical_asos", module,
                            raising=False)
    probed = []

    def probe(name, distribution=None):
        probed.append(name)
        return arrow

    monkeypatch.setattr(doctor, "_import_probe", probe)
    check = doctor._dynamical_asos_check()
    return check, probed


def test_the_row_is_in_the_assembled_report():
    assembler = getattr(doctor, "_collect_checks", doctor.collect_checks)
    assert "_dynamical_asos_check" in assembler.__code__.co_names


def test_everything_present_is_verified_and_offline(monkeypatch):
    check, probed = _check(monkeypatch, _reader())
    assert check.name == doctor.DYNAMICAL_ASOS_NAME
    assert check.status == "verified"
    assert check.blocking is False
    assert probed == ["pyarrow"]
    assert "pyarrow 17.0.0" in check.detail
    assert "3 station(s) in 3 country code(s) (GB, JP, US)" in check.detail
    assert "not contacted" in check.detail
    assert "https://data.source.coop/dynamical/asos-parquet" in check.detail
    assert "not created yet" in check.detail
    # The --explain text carries coverage and attribution.
    assert "14 countries" in check.detail
    assert "Iowa Environmental Mesonet" in check.detail
    assert "Source Cooperative" in check.detail
    assert check not in doctor.blocking_gaps([check])


def test_the_url_override_and_cache_size_are_reported(monkeypatch, tmp_path):
    cache = tmp_path / "cache" / "year=2026"
    cache.mkdir(parents=True)
    (cache / "data.parquet").write_bytes(b"x" * 2_000_000)
    monkeypatch.setenv("WOOF_DYNAMICAL_ASOS_URL", "https://mirror.example/a")
    check, _ = _check(monkeypatch, _reader())
    assert "https://mirror.example/a (from WOOF_DYNAMICAL_ASOS_URL)" in \
        check.detail
    assert "1 file(s), 2.0 MB" in check.detail


def test_absent_reader_is_info_and_names_the_update(monkeypatch):
    check, probed = _check(monkeypatch, None)
    assert check.status == "info"
    assert check.blocking is False
    assert probed == [], "no pyarrow probe is worth running without a reader"
    assert "does not include the Dynamical.org ASOS reader" in check.detail
    assert "update to a WOOF build that includes the Dynamical reader" in \
        check.remedy
    assert "Iowa Environmental Mesonet" in check.detail
    assert not doctor.blocking_gaps([check])


def test_absent_pyarrow_is_an_opt_in_gap(monkeypatch):
    check, _ = _check(monkeypatch, _reader(), arrow=(False, "not installed"))
    assert check.status == "missing"
    assert check.blocking is False
    assert check.severity == doctor.SEVERITY_OPT_IN
    assert check.action == "pip install 'recast-woof[obs]'"
    assert check.remedy.splitlines()[0] == "pip install 'recast-woof[obs]'"
    assert "pyarrow not installed" in check.detail
    assert not doctor.blocking_gaps([check])


def test_a_missing_station_table_is_a_degraded_gap(monkeypatch):
    check, _ = _check(monkeypatch, _reader(
        stations=FileNotFoundError("dynamical_asos_stations.json")))
    assert check.status == "missing"
    assert check.blocking is False
    assert check.severity == doctor.SEVERITY_DEGRADED
    assert "station table unreadable" in check.detail
    assert not doctor.blocking_gaps([check])


def test_a_reader_that_raises_on_import_is_reported(monkeypatch):
    class Exploding(types.ModuleType):
        def __getattr__(self, name):
            raise AssertionError("never reached")

    import importlib

    real = importlib.import_module

    def import_module(name, package=None):
        if name == MODULE:
            raise RuntimeError("table schema mismatch")
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)
    check, _ = _check(monkeypatch, Exploding(MODULE))
    assert check.status == "missing"
    assert check.blocking is False
    assert "table schema mismatch" in check.detail


def test_a_reader_that_imports_pyarrow_eagerly_names_the_extra(monkeypatch):
    import importlib

    real = importlib.import_module

    def import_module(name, package=None):
        if name == MODULE:
            raise ModuleNotFoundError("No module named 'pyarrow'",
                                      name="pyarrow")
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)
    check, _ = _check(monkeypatch, _reader())
    assert check.status == "missing"
    assert check.severity == doctor.SEVERITY_OPT_IN
    assert check.action == "pip install 'recast-woof[obs]'"
    assert not doctor.blocking_gaps([check])


def test_the_brief_report_stays_one_line(monkeypatch):
    check, _ = _check(monkeypatch, _reader())
    brief = doctor.format_brief([check])
    line = [row for row in brief.splitlines() if "Dynamical" in row]
    assert len(line) == 1
    assert "Iowa" not in line[0]
    report = doctor.format_report([check])
    assert "Iowa Environmental Mesonet" in report


def test_a_broken_pyarrow_is_degraded_not_opt_in(monkeypatch):
    check, _ = _check(monkeypatch, _reader(), arrow=(
        False, "installed but failed to import: ImportError: libarrow.so"))
    assert check.status == "missing"
    assert check.severity == doctor.SEVERITY_DEGRADED
    assert check.blocking is False
    assert check.action == "pip install --force-reinstall pyarrow"
    assert check.brief == "pyarrow does not import"


def test_an_unreadable_cache_says_so(monkeypatch, tmp_path):
    import os

    cache = tmp_path / "cache"
    (cache / "year=2026").mkdir(parents=True)
    cache.chmod(0)
    try:
        if os.access(cache, os.R_OK):
            pytest.skip("running with permissions that ignore mode 000")
        check, _ = _check(monkeypatch, _reader())
    finally:
        cache.chmod(0o755)
    assert "unreadable" in check.detail


def test_a_reader_without_a_file_does_not_crash_the_report(monkeypatch):
    reader = _reader()
    reader.__file__ = None
    check, _ = _check(monkeypatch, reader)
    assert check.status == "verified"
