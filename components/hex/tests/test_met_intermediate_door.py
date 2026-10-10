"""The global-source intermediate door: table, series, plan, validation, refusals.

Engine-free where it can be: a stand-in ``met_intermediate`` (a Python
script that writes a synthetic WPS file with this tree's own record writer)
drives the whole door, and the validation is proved against this tree's
reader and the init door's met check.  One test drives the real binary over
the tiny GDAS fixtures when the binary is built, and skips otherwise.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import textwrap

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import met_intermediate_door as door  # noqa: E402
from woof.hex.cli import build_parser, main as hex_main  # noqa: E402
from woof.hex.hrrr_intermediate import IntermediateRefusal, LatLonTarget, wps_record_bytes  # noqa: E402
from woof.hex.met_intermediate_door import GlobalIntermediateRefusal  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
GDAS_FIXTURES = REPO / "tests" / "fixtures" / "gdas-process-id"
CYCLE = datetime(2026, 9, 24, 12)


# ---------------------------------------------------------------------------
# a synthetic intermediate, written with the tree's own record writer
# ---------------------------------------------------------------------------
TARGET = LatLonTarget(south=50.0, west=-10.0, dlat=0.25, dlon=0.25, nx=4, ny=3)


def _records(*, landsea: bool = True, soil: int = 4, levels=(100000.0, 50000.0, 1000.0)) -> list[dict]:
    slab = np.full((TARGET.ny, TARGET.nx), 1.0, dtype=np.float32)
    out: list[dict] = []
    for level in (200100.0, *levels):
        for name, units, value in (("TT", "K", 280.0), ("RH", "%", 50.0), ("UU", "m s-1", 5.0),
                                   ("VV", "m s-1", -2.0), ("SPECHUMD", "kg kg-1", 0.004)):
            out.append({"field": name, "units": units, "description": name, "level": level,
                        "values": slab * value})
        if level != 200100.0:
            out.append({"field": "HGT", "units": "m", "description": "Height", "level": level,
                        "values": slab * 1000.0})
    surface = [("PSFC", "Pa", 101000.0), ("SOILHGT", "m", 10.0), ("SKINTEMP", "K", 285.0)]
    if landsea:
        surface.append(("LANDSEA", "proprtn", 1.0))
    names = ("000007", "007028", "028100", "100289")[:soil]
    surface += [(f"ST{n}", "K", 283.0) for n in names] + [(f"SM{n}", "m3 m-3", 0.3) for n in names]
    for name, units, value in surface:
        out.append({"field": name, "units": units, "description": name, "level": 200100.0,
                    "values": slab * value})
    return out


def write_synthetic(path: Path, hdate: str, **kwargs) -> int:
    records = _records(**kwargs)
    with path.open("wb") as handle:
        for item in records:
            handle.write(wps_record_bytes(
                hdate=hdate, xfcst=0.0, map_source="ECMWF", field_name=item["field"],
                units=item["units"], description=item["description"], level=item["level"],
                target=TARGET, values=item["values"],
            ))
    return len(records)


def engine_receipt(hdate: str, records: int, **extra) -> dict:
    receipt = {"schema": door.ENGINE_RECEIPT_SCHEMA, "matched_valid_times": [hdate],
               "records_written": records}
    receipt.update(extra)
    return receipt


# ---------------------------------------------------------------------------
# the source table
# ---------------------------------------------------------------------------
def test_the_contract_sources_are_exactly_the_admitted_rows() -> None:
    from woof.hex.hrrr_intermediate import INTERMEDIATE_SOURCES, SOURCE_ROWS

    contract = {"hrrr", "gfs", "gdas", "ecmwf-open-data", "era5", "aifs"}
    assert contract <= set(INTERMEDIATE_SOURCES)
    assert set(INTERMEDIATE_SOURCES) == set(SOURCE_ROWS) | set(door.GLOBAL_SOURCES)
    assert not set(SOURCE_ROWS) & set(door.GLOBAL_SOURCES)


@pytest.mark.parametrize("name", sorted(door.GLOBAL_SOURCES))
def test_every_row_names_a_packaged_vtable_and_a_known_map_source(name: str) -> None:
    source = door.GLOBAL_SOURCES[name]
    assert source.map_source in {"ncep-gfs", "ncep-gefs", "ncep-cdas-cfsv2", "ecmwf"}
    text = door.packaged_vtable(source.vtable).read_text()
    rows = [line.split("|") for line in text.splitlines()
            if not line.lstrip().startswith(("#", "-"))]
    names = {cells[4].strip() for cells in rows if len(cells) >= 7}
    assert "LANDSEA" in names
    assert any(n.startswith("ST") for n in names) and any(n.startswith("SM") for n in names)
    assert {"TT", "UU", "VV", "PSFC", "SKINTEMP"} <= names


def test_fetch_file_names_map_to_leads() -> None:
    root = Path("/x")
    assert [p.name for p in door.GLOBAL_SOURCES["gfs"].lead_files(root, CYCLE, 3)] == [
        "gfs.t12z.pgrb2.0p25.f003", "gfs.t12z.pgrb2.0p25.f003.subset.grib2"]
    assert [p.name for p in door.GLOBAL_SOURCES["gdas"].lead_files(root, CYCLE, 9)][0] == \
        "gdas.t12z.pgrb2.0p25.f009"
    assert door.GLOBAL_SOURCES["ecmwf-open-data"].lead_files(root, CYCLE, 12)[0].name == \
        "20260924120000-12h-oper-fc.grib2"
    assert door.GLOBAL_SOURCES["aifs"].lead_files(root, CYCLE, 0)[0].name == \
        "20260924120000-0h-oper-fc.grib2"
    assert door.GLOBAL_SOURCES["era5"].layout == "combined"


# ---------------------------------------------------------------------------
# the time series
# ---------------------------------------------------------------------------
def test_lead_ranges_step_by_the_interval() -> None:
    assert door.parse_lead_range(None, 3) == (0,)
    assert door.parse_lead_range("0-12", 3) == (0, 3, 6, 9, 12)
    assert door.parse_lead_range("6", 6) == (6,)
    assert door.parse_lead_range("3-9", 1) == tuple(range(3, 10))


@pytest.mark.parametrize("spec,interval,words", [
    ("0-10", 3, "whole number of 3 h intervals"),
    ("6-0", 3, "runs backwards"),
    ("a-b", 3, "START-END"),
    ("0-6", 0, "positive number"),
])
def test_bad_lead_ranges_refuse(spec: str, interval: int, words: str) -> None:
    with pytest.raises(GlobalIntermediateRefusal, match=words):
        door.parse_lead_range(spec, interval)


# ---------------------------------------------------------------------------
# input location and the engine argv
# ---------------------------------------------------------------------------
def _request(tmp_path: Path, source: str, leads, **kw) -> door.GlobalRequest:
    return door.GlobalRequest(
        source=door.GLOBAL_SOURCES[source], grib_dir=tmp_path / "grib", cycle=CYCLE,
        leads=tuple(leads), interval_hours=kw.pop("interval", 3), out_dir=tmp_path / "out", **kw)


def test_missing_leads_are_named_together(tmp_path: Path) -> None:
    (tmp_path / "grib").mkdir()
    (tmp_path / "grib" / "gfs.t12z.pgrb2.0p25.f000").write_bytes(b"GRIB")
    with pytest.raises(GlobalIntermediateRefusal) as caught:
        door.locate_inputs(_request(tmp_path, "gfs", (0, 3, 6)))
    assert "f003" in str(caught.value) and "f006" in str(caught.value)
    assert "woof fetch --source gfs" in str(caught.value)


def test_a_whole_object_beside_its_subset_is_ambiguous(tmp_path: Path) -> None:
    grib = tmp_path / "grib"
    grib.mkdir()
    (grib / "gfs.t12z.pgrb2.0p25.f000").write_bytes(b"GRIB")
    (grib / "gfs.t12z.pgrb2.0p25.f000.subset.grib2").write_bytes(b"GRIB")
    with pytest.raises(GlobalIntermediateRefusal, match="more than one candidate"):
        door.locate_inputs(_request(tmp_path, "gfs", (0,)))


def test_leads_past_the_published_range_refuse(tmp_path: Path) -> None:
    (tmp_path / "grib").mkdir()
    with pytest.raises(GlobalIntermediateRefusal, match="f009"):
        door.locate_inputs(_request(tmp_path, "gdas", (0, 12), interval=12))


def test_aifs_later_leads_take_the_hour_zero_invariants(tmp_path: Path) -> None:
    grib = tmp_path / "grib"
    grib.mkdir()
    for lead in (0, 6):
        (grib / f"20260924120000-{lead}h-oper-fc.grib2").write_bytes(b"GRIB")
    request = _request(tmp_path, "aifs", (0, 6), interval=6)
    plans = door.plan_leads(request, Path("/bin/met_intermediate"), Path("/v/Vtable"))
    assert "--invariant" not in plans[0].argv
    argv = plans[1].argv
    assert argv[argv.index("--invariant") + 1].endswith("20260924120000-0h-oper-fc.grib2")
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--invariant-field"] == ["LANDSEA", "SOILGEO"]
    assert argv[argv.index("--map-source") + 1] == "ecmwf"
    assert argv[argv.index("--date") + 1] == "2026-09-24_18:00:00"
    assert argv[argv.index("--out") + 1].endswith("MET:2026-09-24_18.partial")
    assert argv[-1].endswith("20260924120000-6h-oper-fc.grib2")
    assert "--select-valid-time" not in argv


def test_a_later_lead_without_hour_zero_refuses_for_its_invariants(tmp_path: Path) -> None:
    grib = tmp_path / "grib"
    grib.mkdir()
    (grib / "20260924120000-6h-oper-fc.grib2").write_bytes(b"GRIB")
    with pytest.raises(GlobalIntermediateRefusal, match="LANDSEA, SOILGEO"):
        door.locate_inputs(_request(tmp_path, "aifs", (6,), interval=6))


def test_era5_selects_each_valid_time_out_of_the_combined_file(tmp_path: Path) -> None:
    grib = tmp_path / "grib"
    grib.mkdir()
    (grib / "era5-combined.grib").write_bytes(b"GRIB")
    plans = door.plan_leads(_request(tmp_path, "era5", (0, 6), interval=6),
                            Path("/bin/met_intermediate"), Path("/v/Vtable"))
    for plan in plans:
        assert "--select-valid-time" in plan.argv
        assert plan.argv[-1].endswith("era5-combined.grib")
    assert [p.hdate for p in plans] == ["2026-09-24_12:00:00", "2026-09-24_18:00:00"]


def test_era5_arco_netcdf_is_refused_by_name(tmp_path: Path) -> None:
    grib = tmp_path / "grib"
    grib.mkdir()
    (grib / "era5-combined.nc").write_bytes(b"CDF")
    with pytest.raises(GlobalIntermediateRefusal, match="GRIB-only"):
        door.locate_inputs(_request(tmp_path, "era5", (0,), interval=6))


def test_gfs_argv_is_plain(tmp_path: Path) -> None:
    grib = tmp_path / "grib"
    grib.mkdir()
    (grib / "gfs.t12z.pgrb2.0p25.f000.subset.grib2").write_bytes(b"GRIB")
    plan, = door.plan_leads(_request(tmp_path, "gfs", (0,)), Path("/e"), Path("/v"))
    assert plan.argv[:9] == ["/e", "--vtable", "/v", "--date", "2026-09-24_12:00:00",
                             "--map-source", "ncep-gfs", "--out", plan.argv[8]]
    assert "--invariant" not in plan.argv and "--select-valid-time" not in plan.argv


# ---------------------------------------------------------------------------
# refusals by name
# ---------------------------------------------------------------------------
def test_unknown_and_refused_sources_say_why() -> None:
    with pytest.raises(GlobalIntermediateRefusal, match="pgrb2a\\+pgrb2b"):
        door.global_source("gefs")
    with pytest.raises(GlobalIntermediateRefusal, match="no land mask"):
        door.global_source("AIGFS")
    with pytest.raises(GlobalIntermediateRefusal, match="Admitted sources: hrrr, gfs"):
        door.global_source("cmc")


def test_the_cli_refuses_an_unknown_source_with_exit_two(tmp_path: Path, capsys) -> None:
    code = hex_main(["intermediate", "--source", "gefs", "--grib-dir", str(tmp_path),
                     "--cycle", "2026-09-24T12", "--out-dir", str(tmp_path / "o")])
    assert code == 2
    assert "refused by this door" in capsys.readouterr().err


def test_a_box_flag_for_a_global_source_refuses(tmp_path: Path) -> None:
    arguments = build_parser().parse_args([
        "intermediate", "--source", "gfs", "--grib-dir", str(tmp_path), "--cycle",
        "2026-09-24T12", "--out-dir", str(tmp_path / "o"), "--point", "40,-100",
    ])
    with pytest.raises(GlobalIntermediateRefusal, match="not regridded"):
        door.request_from_arguments(arguments)


def test_global_flags_for_hrrr_refuse(tmp_path: Path) -> None:
    from woof.hex import hrrr_intermediate as hi

    arguments = build_parser().parse_args([
        "intermediate", "--source", "hrrr", "--grib-dir", str(tmp_path), "--cycle",
        "2026-09-24T12", "--out-dir", str(tmp_path / "o"), "--interval-hours", "3",
        "--point", "40,-100", "--radius-km", "50",
    ])
    with pytest.raises(IntermediateRefusal, match="--interval-hours applies to the global"):
        hi.request_from_arguments(arguments)


def test_the_parser_defaults_global_requests_sensibly(tmp_path: Path) -> None:
    arguments = build_parser().parse_args([
        "intermediate", "--source", "aifs", "--grib-dir", str(tmp_path), "--cycle",
        "2026-09-24T12", "--out-dir", str(tmp_path / "o"), "--hours", "0-12",
    ])
    request = door.request_from_arguments(arguments)
    assert request.leads == (0, 6, 12) and request.interval_hours == 6
    assert request.source.use_spechumd == "yes"


def test_a_global_source_needs_its_grib_dir_and_cycle(tmp_path: Path) -> None:
    arguments = build_parser().parse_args([
        "intermediate", "--source", "gfs", "--cycle", "2026-09-24T12",
        "--out-dir", str(tmp_path / "o"),
    ])
    with pytest.raises(GlobalIntermediateRefusal, match="needs --grib-dir"):
        door.request_from_arguments(arguments)


def test_flags_of_another_row_refuse_for_a_global_source(tmp_path: Path) -> None:
    arguments = argparse.Namespace(source="gfs", from_plan=None, point=None, radius_km=None,
                                   wrfout_glob="wrfout_d01_*", grib_dir=tmp_path,
                                   cycle="2026-09-24T12")
    with pytest.raises(GlobalIntermediateRefusal, match="--wrfout-glob belong to another"):
        door.request_from_arguments(arguments)


def test_an_explicit_engine_that_is_missing_refuses(tmp_path: Path) -> None:
    with pytest.raises(GlobalIntermediateRefusal, match="is not a file"):
        door.resolve_engine(tmp_path / "nope")


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def test_a_synthetic_intermediate_validates(tmp_path: Path) -> None:
    path = tmp_path / "MET:2026-09-24_12"
    n = write_synthetic(path, "2026-09-24_12:00:00")
    facts = door.validate_intermediate(path, hdate="2026-09-24_12:00:00",
                                       engine_receipt=engine_receipt("2026-09-24_12:00:00", n))
    assert facts["field_records"] == n
    assert facts["profile_levels"] == 4 and facts["isobaric_levels"] == 3
    assert facts["top_pressure_pa"] == 1000.0
    assert facts["soil_layers"] == 4
    assert facts["grid"]["nx"] == 4 and facts["grid"]["projection"] == "latlon"


def test_an_intermediate_without_landsea_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "MET:2026-09-24_12"
    write_synthetic(path, "2026-09-24_12:00:00", landsea=False)
    with pytest.raises(GlobalIntermediateRefusal, match="LANDSEA"):
        door.validate_intermediate(path, hdate="2026-09-24_12:00:00")


def test_an_intermediate_stamped_for_another_time_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "MET:2026-09-24_12"
    write_synthetic(path, "2026-09-24_15:00:00")
    with pytest.raises(GlobalIntermediateRefusal, match="valid times"):
        door.validate_intermediate(path, hdate="2026-09-24_12:00:00")


def test_messages_from_another_time_in_the_engine_receipt_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "MET:2026-09-24_12"
    n = write_synthetic(path, "2026-09-24_12:00:00")
    receipt = engine_receipt("2026-09-24_12:00:00", n,
                             matched_valid_times=["2026-09-24_06:00:00", "2026-09-24_12:00:00"])
    with pytest.raises(GlobalIntermediateRefusal, match="--select-valid-time"):
        door.validate_intermediate(path, hdate="2026-09-24_12:00:00", engine_receipt=receipt)


def test_an_engine_record_count_disagreement_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "MET:2026-09-24_12"
    n = write_synthetic(path, "2026-09-24_12:00:00")
    with pytest.raises(GlobalIntermediateRefusal, match="records written"):
        door.validate_intermediate(path, hdate="2026-09-24_12:00:00",
                                   engine_receipt=engine_receipt("2026-09-24_12:00:00", n + 1))


def test_an_old_engine_receipt_schema_is_refused() -> None:
    with pytest.raises(GlobalIntermediateRefusal, match="rebuild tools/grib1_bridge"):
        door.parse_engine_receipt(json.dumps({"schema": "gpuwm.rw-wps.met-intermediate/v2"}))
    with pytest.raises(GlobalIntermediateRefusal, match="not its JSON receipt"):
        door.parse_engine_receipt("met_intermediate: done")


def test_init_switches_follow_the_series() -> None:
    files = [{"profile_levels": 15, "soil_layers": 2, "top_pressure_pa": 5000.0,
              "has_rh": False, "has_spechumd": True}]
    switches = door.init_switches(door.GLOBAL_SOURCES["ecmwf-open-data"], files)
    assert switches["--use-spechumd"] == "yes"  # no RH aloft: moisture from q
    assert switches["--extrap-airtemp"] == "constant"
    assert switches["--nfgsoillevels"] == 2
    files[0]["top_pressure_pa"] = 100.0
    assert door.init_switches(door.GLOBAL_SOURCES["gfs"], [dict(files[0], has_rh=True)])[
        "--extrap-airtemp"] == "lapse-rate"
    with pytest.raises(GlobalIntermediateRefusal, match="soil column depths"):
        door.init_switches(door.GLOBAL_SOURCES["gfs"],
                           [dict(files[0], has_rh=True), dict(files[0], has_rh=True, soil_layers=4)])


# ---------------------------------------------------------------------------
# the whole door against a stand-in engine
# ---------------------------------------------------------------------------
FAKE_ENGINE = textwrap.dedent('''\
    import json, sys
    sys.path[:0] = {paths!r}
    from test_met_intermediate_door import write_synthetic
    argv = sys.argv[1:]
    hdate = argv[argv.index("--date") + 1]
    out = argv[argv.index("--out") + 1]
    stamp = {stamp!r} or hdate
    n = write_synthetic(__import__("pathlib").Path(out), stamp)
    print(json.dumps({{"schema": {schema!r}, "matched_valid_times": [stamp],
                      "records_written": n, "argv": argv}}))
''')


def _fake_engine(tmp_path: Path, *, stamp: str = "") -> Path:
    script = tmp_path / "fake_met_intermediate.py"
    script.write_text(FAKE_ENGINE.format(
        paths=[str(Path(__file__).parent), *sys.path], stamp=stamp,
        schema=door.ENGINE_RECEIPT_SCHEMA))
    launcher = tmp_path / "met_intermediate"
    launcher.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
    launcher.chmod(0o755)
    return launcher


def test_the_door_writes_validates_and_receipts_a_series(tmp_path: Path) -> None:
    grib = tmp_path / "grib"
    grib.mkdir()
    for lead in (0, 3, 6):
        (grib / f"20260924120000-{lead}h-oper-fc.grib2").write_bytes(b"GRIB" + bytes([lead]))
    request = _request(tmp_path, "ecmwf-open-data", (0, 3, 6), engine=_fake_engine(tmp_path))
    receipt = door.build_global_intermediates(request, log=lambda _m: None)
    out = tmp_path / "out"
    assert [Path(f["path"]).name for f in receipt["files"]] == [
        "MET:2026-09-24_12", "MET:2026-09-24_15", "MET:2026-09-24_18"]
    assert not list(out.glob("*.partial"))
    assert receipt["schema"] == door.GLOBAL_SCHEMA and receipt["source"] == "ecmwf-open-data"
    assert receipt["vtable"]["path"].endswith("Vtable.ECMWF-OD.rw")
    assert len(receipt["vtable"]["sha256"]) == 64 and len(receipt["engine"]["sha256"]) == 64
    first, later = receipt["files"][0], receipt["files"][1]
    assert first["inputs"][0]["path"].endswith("0h-oper-fc.grib2") and first["invariants"] == []
    assert later["invariants"][0]["path"].endswith("0h-oper-fc.grib2")
    assert len(later["inputs"][0]["sha256"]) == 64
    assert receipt["init_switches"]["--nfglevels"] == 4
    assert json.loads((out / "intermediate-receipt.json").read_text())["valid_times"] == \
        receipt["valid_times"]
    assert (out / "met_intermediate.f003.log").is_file()


def test_a_file_the_engine_stamped_wrongly_is_not_published(tmp_path: Path) -> None:
    grib = tmp_path / "grib"
    grib.mkdir()
    (grib / "gfs.t12z.pgrb2.0p25.f000").write_bytes(b"GRIB")
    request = _request(tmp_path, "gfs", (0,), engine=_fake_engine(tmp_path, stamp="2026-09-24_06:00:00"))
    with pytest.raises(GlobalIntermediateRefusal, match="not exactly 2026-09-24_12:00:00"):
        door.build_global_intermediates(request, log=lambda _m: None)
    assert not list((tmp_path / "out").glob("MET*"))


def test_an_out_dir_holding_intermediates_is_refused(tmp_path: Path) -> None:
    grib = tmp_path / "grib"
    grib.mkdir()
    (grib / "gfs.t12z.pgrb2.0p25.f000").write_bytes(b"GRIB")
    out = tmp_path / "out"
    out.mkdir()
    write_synthetic(out / "OLD:2026-09-23_06", "2026-09-23_06:00:00")
    with pytest.raises(GlobalIntermediateRefusal, match="fresh --out-dir"):
        door.build_global_intermediates(
            _request(tmp_path, "gfs", (0,), engine=_fake_engine(tmp_path)), log=lambda _m: None)


def test_one_bad_lead_publishes_none_of_the_series(tmp_path: Path) -> None:
    grib = tmp_path / "grib"
    grib.mkdir()
    for lead in (0, 3):
        (grib / f"gfs.t12z.pgrb2.0p25.f{lead:03d}").write_bytes(b"GRIB")
    # The stand-in stamps every file 12Z, so f003 fails validation after
    # f000 passed: neither may be left behind, final or partial.
    request = _request(tmp_path, "gfs", (0, 3),
                       engine=_fake_engine(tmp_path, stamp="2026-09-24_12:00:00"))
    with pytest.raises(GlobalIntermediateRefusal):
        door.build_global_intermediates(request, log=lambda _m: None)
    assert not list((tmp_path / "out").glob("MET*"))


def test_an_engine_failure_names_its_log(tmp_path: Path) -> None:
    grib = tmp_path / "grib"
    grib.mkdir()
    (grib / "gfs.t12z.pgrb2.0p25.f000").write_bytes(b"GRIB")
    engine = tmp_path / "met_intermediate"
    engine.write_text("#!/bin/sh\necho 'met_intermediate: unknown --map-source' >&2\nexit 1\n")
    engine.chmod(0o755)
    with pytest.raises(GlobalIntermediateRefusal, match="exited 1 for f000.*unknown --map-source"):
        door.build_global_intermediates(_request(tmp_path, "gfs", (0,), engine=engine),
                                        log=lambda _m: None)


# ---------------------------------------------------------------------------
# the real engine over real (tiny) GDAS bytes, when it is built
# ---------------------------------------------------------------------------
def _built_engine() -> Path | None:
    try:
        from woof import bridges

        found = bridges.find_bridge("met_intermediate")
    except Exception:  # noqa: BLE001
        return None
    if found is None or not bridges.bridge_abi_matches("met_intermediate", Path(found))[0]:
        return None
    return Path(found)


@pytest.mark.skipif(_built_engine() is None or not GDAS_FIXTURES.is_dir(),
                    reason="met_intermediate (v3 contract) is not built here")
def test_real_met_intermediate_converts_the_gdas_fixtures(tmp_path: Path) -> None:
    grib = tmp_path / "grib"
    grib.mkdir()
    for lead in (0, 3, 6, 9):
        shutil.copy(GDAS_FIXTURES / f"nomads-gdas-20260729t12z-f{lead:03d}.grib2",
                    grib / f"gdas.t12z.pgrb2.0p25.f{lead:03d}.subset.grib2")
    request = door.GlobalRequest(
        source=door.GLOBAL_SOURCES["gdas"], grib_dir=grib, cycle=datetime(2026, 7, 29, 12),
        leads=(0, 3, 6, 9), interval_hours=3, out_dir=tmp_path / "out", engine=_built_engine())
    receipt = door.build_global_intermediates(request, log=lambda _m: None)
    assert receipt["valid_times"] == [
        "2026-07-29_12:00:00", "2026-07-29_15:00:00", "2026-07-29_18:00:00", "2026-07-29_21:00:00"]
    for item in receipt["files"]:
        assert item["engine_receipt"]["matched_valid_times"] == [item["valid_time"]]
        assert item["soil_layers"] == 4 and item["fields"]["LANDSEA"] == 1
    assert receipt["init_switches"]["--nfgsoillevels"] == 4
