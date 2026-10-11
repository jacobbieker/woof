"""ICON global through the intermediate door: table, file location, window, the whole door.

Engine-free: a stand-in ``gdt101_remap`` (a Python script that checks every
job row it is handed and writes a placeholder output per row) and a stand-in
``met_intermediate`` (writes a synthetic WPS file with this tree's own record
writer) drive the whole ``field-files`` road, and the validation is the
door's own read-back.  The DWD map source and its Vtables arrive with the
``met_intermediate`` work; nothing here needs them.
"""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import sys
import textwrap

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import gdt101_door as g101  # noqa: E402
from woof.hex import met_intermediate_door as door  # noqa: E402
from woof.hex.cli import build_parser  # noqa: E402
from woof.hex.hrrr_intermediate import IntermediateRefusal, LatLonTarget, wps_record_bytes  # noqa: E402
from woof.hex.met_intermediate_door import GlobalIntermediateRefusal  # noqa: E402

CYCLE = datetime(2026, 10, 11, 0)


# ---------------------------------------------------------------------------
# the product table
# ---------------------------------------------------------------------------
def test_the_rows_are_field_file_rows_with_the_dwd_label() -> None:
    for name, vtable in (("icon-global", "Vtable.ICON-GLOBAL.rw"),
                         ("icon-global-ml", "Vtable.ICON-GLOBAL-ML.rw")):
        row = door.GLOBAL_SOURCES[name]
        assert row.layout == "field-files" and row.product is not None
        assert row.map_source == "dwd" and row.vtable == vtable
        assert row.default_interval_hours == 3
        assert "Deutscher Wetterdienst" in row.attribution and "CC BY 4.0" in row.attribution
    assert door.GLOBAL_SOURCES["icon-global"].use_spechumd == "no"
    assert door.GLOBAL_SOURCES["icon-global-ml"].use_spechumd == "yes"
    assert door.GLOBAL_SOURCES["icon-global"].invariant_fields == ("LANDSEA", "SOILHGT")
    assert door.GLOBAL_SOURCES["icon-global-ml"].invariant_fields == ("LANDSEA", "SOILHGT", "GHT")


def test_the_pressure_row_needs_eighteen_levels_surface_and_soil() -> None:
    product = g101.ICON_GLOBAL_PL
    labels = [r.label for r in product.per_lead]
    assert len(labels) == len(set(labels)) == 5 * 18 + 9 + 9 + 8
    assert {r.param for r in product.per_lead if r.level_type == 100} == {"T", "U", "V", "FI", "RELHUM"}
    assert {r.level for r in product.per_lead if r.param == "T"} == {
        str(h * 100) for h in (30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700,
                               800, 850, 900, 925, 950, 1000)}
    assert [r.param for r in product.invariants] == ["HSURF", "FR_LAND"]
    assert [r.param for r in product.geometry] == ["CLAT", "CLON"]


def test_the_model_level_row_needs_every_level_and_the_half_levels() -> None:
    product = g101.ICON_GLOBAL_ML
    model = [r for r in product.per_lead if r.level_type == 150]
    assert len(model) == 5 * 120
    assert {r.param for r in model} == {"T", "U", "V", "QV", "P"}
    hhl = [r for r in product.invariants if r.param == "HHL"]
    assert [r.level for r in hhl] == [str(n) for n in range(1, 122)]
    t90 = next(r for r in model if r.param == "T" and r.level == "90")
    assert t90.selector == (0, 0, 0, 150, 90.0, 150, 91.0)
    assert hhl[-1].selector == (0, 3, 6, 150, 121.0, 101, 0.0)


def test_the_paths_are_the_v1_tree() -> None:
    t500 = next(r for r in g101.ICON_GLOBAL_PL.per_lead if r.label == "T@50000")
    assert str(t500.relative(CYCLE, 3)) == "T/lvt1/100/lv1/50000/r/2026-10-11T00:00/s/PT003H00M.grib2"
    clat = g101.ICON_GLOBAL_PL.geometry[0]
    assert str(clat.relative(CYCLE, 0)) == "CLAT/r/2026-10-11T00:00/s/PT000H00M.grib2"
    w_so = [r for r in g101.ICON_GLOBAL_PL.per_lead if r.param == "W_SO"]
    assert [r.level for r in w_so] == ["0.0", "0.01", "0.03", "0.09", "0.27", "0.81", "2.43", "7.29"]
    # A layer's selector carries its top and bottom; the directory is the top.
    assert w_so[1].selector == (2, 3, 20, 106, 0.01, 106, 0.03)
    assert w_so[-1].selector[6] == 21.87


def test_every_method_is_one_its_level_type_admits() -> None:
    for record in (*g101.ICON_GLOBAL_ML.per_lead, *g101.ICON_GLOBAL_ML.invariants,
                   *g101.ICON_GLOBAL_PL.per_lead):
        if record.selector[3] == 100:
            assert record.method in {"idw4", "nearest"}, record.label
        if record.selector[3] == 106:
            assert record.method == "soil-nearest", record.label
        assert record.method in {"idw4", "surface-idw4", "nearest", "soil-nearest",
                                 "seaice-nearest"}
    assert next(r for r in g101.ICON_GLOBAL_PL.per_lead if r.param == "FR_ICE").method == "seaice-nearest"


@pytest.mark.parametrize("cycle,leads,words", [
    (datetime(2026, 10, 11, 6), (123,), "to f120"),
    (datetime(2026, 10, 11, 0), (183,), "to f180"),
    (datetime(2026, 10, 11, 0), (79,), "f079 is not published"),
    (datetime(2026, 10, 11, 3), (0,), "is not one of them"),
])
def test_unpublished_leads_and_cycles_refuse(cycle, leads, words) -> None:
    with pytest.raises(GlobalIntermediateRefusal, match=words):
        g101.ICON_GLOBAL_PL.refuse_unpublished(cycle, leads)


def test_published_leads_pass() -> None:
    g101.ICON_GLOBAL_PL.refuse_unpublished(datetime(2026, 10, 11, 12), (0, 1, 78, 81, 180))
    g101.ICON_GLOBAL_PL.refuse_unpublished(datetime(2026, 10, 11, 18), (120,))


# ---------------------------------------------------------------------------
# locating the files
# ---------------------------------------------------------------------------
def _tree(root: Path, product, leads, *, skip: tuple[str, ...] = ()) -> list[Path]:
    """Every file ``product`` needs for ``leads``, as tiny stand-ins."""

    written = []
    wanted = [(r, 0) for r in (*product.geometry, product.land_fraction, *product.invariants)]
    wanted += [(r, lead) for lead in leads for r in product.per_lead]
    for record, lead in wanted:
        path = root / record.relative(CYCLE, lead)
        if f"f{lead:03d} {record.label}" in skip:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GRIB" + f"{record.label}.{lead}".encode())
        written.append(path)
    return written


def test_a_complete_tree_is_located_under_the_dir_or_its_p_child(tmp_path: Path) -> None:
    _tree(tmp_path / "grib" / "p", g101.ICON_GLOBAL_PL, (0, 3))
    found = g101.locate_field_files(g101.ICON_GLOBAL_PL, tmp_path / "grib", CYCLE, (0, 3))
    assert found.root == tmp_path / "grib" / "p"
    assert len(found.leads[3]) == len(g101.ICON_GLOBAL_PL.per_lead)
    assert found.geometry[0].name == "PT000H00M.grib2" and "CLAT" in found.geometry[0].parts
    assert all("PT003H00M" in p.name for _, p in found.leads[3])


def test_missing_files_are_named_with_their_count(tmp_path: Path) -> None:
    _tree(tmp_path / "grib", g101.ICON_GLOBAL_PL, (0, 3),
          skip=("f003 RELHUM@85000", "f000 HSURF"))
    request = door.GlobalRequest(
        source=door.GLOBAL_SOURCES["icon-global"], grib_dir=tmp_path / "grib", cycle=CYCLE,
        leads=(0, 3), interval_hours=3, out_dir=tmp_path / "out")
    with pytest.raises(GlobalIntermediateRefusal) as caught:
        door.locate_inputs(request)
    text = str(caught.value)
    assert "missing 2 icon-global" in text
    assert "f003 RELHUM@85000 (RELHUM/lvt1/100/lv1/85000/r/2026-10-11T00:00/s/PT003H00M.grib2)" in text
    assert "f000 HSURF" in text and "woof fetch --source icon-global" in text


def test_a_long_missing_list_is_grouped_by_field(tmp_path: Path) -> None:
    (tmp_path / "grib").mkdir()
    with pytest.raises(GlobalIntermediateRefusal) as caught:
        g101.locate_field_files(g101.ICON_GLOBAL_ML, tmp_path / "grib", CYCLE, (0, 3))
    text = str(caught.value)
    # Every file once (FR_LAND is both the plan's land fraction and an invariant).
    per_lead = len(g101.ICON_GLOBAL_ML.per_lead)
    assert f"missing {2 + 1 + 1 + 121 + 2 * per_lead} icon-global-ml" in text
    assert "f000 HHL at 121 levels (1, 2, 3, ..., 120, 121; the first is HHL/lvt1/150/lv1/1/" in text
    assert "f000 QV at 120 levels" in text
    # 5 hour-zero groups and 16 per lead: the first 24 by name, the rest counted.
    assert "; and 13 more." in text


def test_the_same_tree_twice_is_ambiguous(tmp_path: Path) -> None:
    for root in (tmp_path / "grib", tmp_path / "grib" / "p"):
        path = root / g101.ICON_GLOBAL_PL.geometry[0].relative(CYCLE, 0)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"GRIB")
    with pytest.raises(GlobalIntermediateRefusal, match="one --grib-dir is one tree"):
        g101.locate_field_files(g101.ICON_GLOBAL_PL, tmp_path / "grib", CYCLE, (0,))


def test_locate_inputs_returns_natives_and_shared_invariants(tmp_path: Path) -> None:
    _tree(tmp_path / "grib", g101.ICON_GLOBAL_ML, (0,))
    request = door.GlobalRequest(
        source=door.GLOBAL_SOURCES["icon-global-ml"], grib_dir=tmp_path / "grib", cycle=CYCLE,
        leads=(0,), interval_hours=3, out_dir=tmp_path / "out")
    natives, shared = door.locate_inputs(request)[0]
    assert len(natives) == len(g101.ICON_GLOBAL_ML.per_lead)
    # CLAT, CLON, FR_LAND (once), HSURF, 121 HHL
    assert len(shared) == 2 + 1 + 1 + 121


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------
def test_a_greenland_cap_gets_a_snapped_window_with_its_halo() -> None:
    lat = np.array([71.9, 72.1, 72.0])
    lon = np.array([-42.3, -41.7, -42.0])
    window = g101.regional_window(lat, lon)
    assert window.step == 0.125
    assert window.west == -43.375 and window.south == 70.875
    assert window.east >= -40.7 and window.north >= 73.1
    assert window.text().split()[4:] == [str(window.nx), str(window.ny)]


def test_a_window_across_the_dateline_keeps_its_western_anchor() -> None:
    window = g101.regional_window(np.array([10.0, 12.0]), np.array([175.0, -175.0]))
    assert -180.0 <= window.west < 180.0
    assert window.west == 174.0
    assert math.isclose(window.east, 186.0)


def test_a_wide_or_polar_window_refuses_with_the_limit() -> None:
    with pytest.raises(GlobalIntermediateRefusal, match="under 180"):
        g101.regional_window(np.array([0.0, 0.0, 0.0]), np.array([0.0, 90.0, 179.5]))
    with pytest.raises(GlobalIntermediateRefusal, match=r"\+/-88"):
        g101.regional_window(np.array([86.0, 87.5]), np.array([0.0, 1.0]))


def test_a_global_mesh_is_refused_naming_the_global_road() -> None:
    lat = np.repeat(np.linspace(-89.5, 89.5, 60), 72)
    lon = np.tile(np.arange(-180.0, 180.0, 5.0), 60)
    assert g101.mesh_is_global(lat, lon)
    with pytest.raises(GlobalIntermediateRefusal, match="gdt101_remap --global --dx-deg 0.125"):
        g101.choose_remap(lat, lon)


def test_a_regional_mesh_chooses_the_regional_road() -> None:
    choice = g101.choose_remap(np.array([71.9, 72.1]), np.array([-42.3, -41.7]), {"grid": "x"})
    assert choice.mode == "regional" and choice.window is not None
    assert choice.basis["grid"] == "x" and choice.basis["halo_degrees"] == 1.0


def _mesh_file(path: Path, lat_deg, lon_deg) -> Path:
    from netCDF4 import Dataset

    with Dataset(str(path), "w") as ds:
        ds.createDimension("nCells", len(lat_deg))
        for name, values in (("latCell", lat_deg), ("lonCell", lon_deg)):
            var = ds.createVariable(name, "f8", ("nCells",))
            var[:] = np.radians(np.asarray(values, dtype=np.float64))
    return path


def test_the_target_mesh_is_read_in_degrees(tmp_path: Path) -> None:
    grid = _mesh_file(tmp_path / "grid.nc", [71.9, 72.1, 72.0], [-42.3, -41.7, 318.0])
    lat, lon, basis = g101.read_target_coordinates(grid)
    assert np.allclose(lat, [71.9, 72.1, 72.0]) and np.allclose(lon, [-42.3, -41.7, 318.0])
    assert basis["points"] == 3 and basis["coordinates"] == "latCell+lonCell"


def test_a_mesh_file_without_cells_refuses(tmp_path: Path) -> None:
    from netCDF4 import Dataset

    with Dataset(str(tmp_path / "x.nc"), "w") as ds:
        ds.createDimension("n", 1)
    with pytest.raises(GlobalIntermediateRefusal, match="latCell/lonCell"):
        g101.read_target_coordinates(tmp_path / "x.nc")


# ---------------------------------------------------------------------------
# the CLI surface
# ---------------------------------------------------------------------------
def _parse(tmp_path: Path, *extra: str):
    return build_parser().parse_args([
        "intermediate", "--grib-dir", str(tmp_path), "--cycle", "2026-10-11T00",
        "--out-dir", str(tmp_path / "o"), *extra])


def test_icon_needs_its_grid(tmp_path: Path) -> None:
    with pytest.raises(GlobalIntermediateRefusal, match="pass --grid GRID.nc"):
        door.request_from_arguments(_parse(tmp_path, "--source", "icon-global"))


def test_icon_takes_its_window_from_the_grid_not_a_box(tmp_path: Path) -> None:
    with pytest.raises(GlobalIntermediateRefusal, match="from the target mesh itself"):
        door.request_from_arguments(_parse(tmp_path, "--source", "icon-global",
                                           "--point", "72,-42", "--radius-km", "10"))


def test_grid_is_refused_for_the_lat_lon_rows_and_hrrr(tmp_path: Path) -> None:
    from woof.hex import hrrr_intermediate as hi

    with pytest.raises(GlobalIntermediateRefusal, match="--grid places the remap window"):
        door.request_from_arguments(_parse(tmp_path, "--source", "gfs", "--grid", "g.nc"))
    with pytest.raises(IntermediateRefusal, match="--grid places the remap window"):
        hi.request_from_arguments(_parse(tmp_path, "--source", "hrrr", "--grid", "g.nc",
                                         "--point", "40,-100", "--radius-km", "50"))


def test_icon_requests_default_to_three_hour_leads(tmp_path: Path) -> None:
    request = door.request_from_arguments(_parse(
        tmp_path, "--source", "icon-global-ml", "--grid", "g.nc", "--hours", "0-6"))
    assert request.leads == (0, 3, 6) and request.window_grid == Path("g.nc")
    assert request.source.product is g101.ICON_GLOBAL_ML


# ---------------------------------------------------------------------------
# the whole door against stand-in binaries
# ---------------------------------------------------------------------------
FAKE_REMAPPER = textwrap.dedent('''\
    import json, sys
    from pathlib import Path
    args = sys.argv[1:]
    log = Path({log!r})
    if {fail!r} and args[0] == {fail!r}:
        sys.stderr.write("gdt101_remap: four native donors do not reach target i=0, j=0\\n")
        sys.exit(2)
    if args[0] == "plan":
        assert len(args) == 12, args
        for selector in (args[2], args[4], args[6]):
            assert len(selector.split(",")) == 7, selector
        Path(args[8]).write_bytes(b"GDT101P3")
        with log.open("a") as f:
            f.write(json.dumps({{"plan": args[1:]}}) + "\\n")
        sys.exit(0)
    assert args[0] == "apply" and len(args) == 3, args
    rows = Path(args[2]).read_text().splitlines()
    for row in rows:
        cells = row.split("\\t")
        assert len(cells) == 12, cells
        assert Path(cells[0]).is_file(), cells[0]
        level_type, method = int(cells[5]), cells[9]
        if level_type == 100:
            assert method in ("idw4", "nearest"), row
        if level_type == 106:
            assert method == "soil-nearest", row
        Path(cells[1]).write_bytes(b"GRIB" + cells[0].encode() + b"7777")
    with log.open("a") as f:
        f.write(json.dumps({{"apply": [c.split("\\t")[10:] for c in rows]}}) + "\\n")
''')

FAKE_ENGINE = textwrap.dedent('''\
    import json, sys
    sys.path[:0] = {paths!r}
    from test_icon_intermediate_door import write_icon_synthetic
    argv = sys.argv[1:]
    hdate = argv[argv.index("--date") + 1]
    out = argv[argv.index("--out") + 1]
    model = "ML" in argv[argv.index("--vtable") + 1]
    n = write_icon_synthetic(__import__("pathlib").Path(out), hdate, model_level=model)
    print(json.dumps({{"schema": {schema!r}, "matched_valid_times": [hdate],
                      "records_written": n, "argv": argv}}))
''')

TARGET = LatLonTarget(south=70.875, west=-43.375, dlat=0.125, dlon=0.125, nx=5, ny=4)
SOIL = ("001", "002", "006", "018", "054", "162", "486", "999")


def write_icon_synthetic(path: Path, hdate: str, *, model_level: bool) -> int:
    """A file shaped like the DWD rows' output: TERRA soil names, and on the
    model-level row SPECHUMD and a 3-D PRES whose top is 2 Pa."""

    slab = np.ones((TARGET.ny, TARGET.nx), dtype=np.float32)
    records: list[tuple[str, str, float, float]] = []
    if model_level:
        for k, pres in ((1, 2.0), (2, 50000.0), (3, 95000.0)):
            records += [("TT", "K", float(k), 250.0), ("UU", "m s-1", float(k), 5.0),
                        ("VV", "m s-1", float(k), 1.0), ("SPECHUMD", "kg kg-1", float(k), 0.003),
                        ("PRES", "Pa", float(k), pres), ("GHT", "m", float(k), 1000.0 * (4 - k))]
    else:
        for pa in (100000.0, 50000.0, 3000.0):
            records += [("TT", "K", pa, 250.0), ("RH", "%", pa, 60.0), ("UU", "m s-1", pa, 5.0),
                        ("VV", "m s-1", pa, 1.0), ("GHT", "m", pa, 5000.0)]
    records += [("TT", "K", 200100.0, 270.0), ("RH", "%", 200100.0, 80.0),
                ("UU", "m s-1", 200100.0, 3.0), ("VV", "m s-1", 200100.0, 1.0),
                ("PSFC", "Pa", 200100.0, 90000.0), ("PMSL", "Pa", 201300.0, 101000.0),
                ("SOILHGT", "m", 200100.0, 900.0), ("LANDSEA", "proprtn", 200100.0, 1.0),
                ("SKINTEMP", "K", 200100.0, 265.0), ("SEAICE", "proprtn", 200100.0, 0.0),
                ("SNOW", "kg m-2", 200100.0, 100.0)]
    records += [(f"SOILT{s}", "K", 200100.0, 268.0) for s in SOIL]
    records += [(f"SOILM{s}", "m3 m-3", 200100.0, 0.2) for s in SOIL]
    with path.open("wb") as handle:
        for name, units, level, value in records:
            handle.write(wps_record_bytes(
                hdate=hdate, xfcst=0.0, map_source="DWD", field_name=name, units=units,
                description=name, level=level, target=TARGET, values=slab * value))
    return len(records)


def _launcher(tmp_path: Path, name: str, body: str) -> Path:
    script = tmp_path / f"fake_{name}.py"
    script.write_text(body)
    launcher = tmp_path / name
    # The door looks for the row's label in the engine's bytes before it
    # remaps; the stand-in carries it the way a build with the table would.
    launcher.write_text(f"#!/bin/sh\n# map sources: dwd\nexec {sys.executable} {script} \"$@\"\n")
    launcher.chmod(0o755)
    return launcher


def _stand_ins(tmp_path: Path, *, fail: str = "") -> tuple[Path, Path, Path, Path]:
    log = tmp_path / "remapper.log"
    remapper = _launcher(tmp_path, "gdt101_remap", FAKE_REMAPPER.format(log=str(log), fail=fail))
    engine = _launcher(tmp_path, "met_intermediate", FAKE_ENGINE.format(
        paths=[str(Path(__file__).parent), *sys.path], schema=door.ENGINE_RECEIPT_SCHEMA))
    vtable = tmp_path / "Vtable.ICON-GLOBAL.rw"
    vtable.write_text("stand-in\n")
    return remapper, engine, vtable, log


def _icon_request(tmp_path: Path, source: str, leads, **kw) -> door.GlobalRequest:
    remapper, engine, vtable, _ = kw.pop("stand_ins")
    if source.endswith("-ml"):
        vtable = vtable.with_name("Vtable.ICON-GLOBAL-ML.rw")
        vtable.write_text("stand-in\n")
    return door.GlobalRequest(
        source=door.GLOBAL_SOURCES[source], grib_dir=tmp_path / "grib", cycle=CYCLE,
        leads=tuple(leads), interval_hours=3, out_dir=tmp_path / "out", engine=engine,
        vtable=vtable, remapper=remapper, workers=2,
        window_grid=kw.pop("grid", None) or _mesh_file(
            tmp_path / "grid.nc", [71.9, 72.1, 72.0], [-42.3, -41.7, -42.0]), **kw)


def test_the_pressure_row_remaps_converts_and_receipts_a_series(tmp_path: Path) -> None:
    stand_ins = _stand_ins(tmp_path)
    _tree(tmp_path / "grib", g101.ICON_GLOBAL_PL, (0, 3))
    request = _icon_request(tmp_path, "icon-global", (0, 3), stand_ins=stand_ins)
    receipt = door.build_global_intermediates(request, log=lambda _m: None)
    out = tmp_path / "out"
    assert [Path(f["path"]).name for f in receipt["files"]] == ["MET:2026-10-11_00", "MET:2026-10-11_03"]
    assert receipt["attribution"] == "Source: Deutscher Wetterdienst (DWD), CC BY 4.0"
    assert receipt["licence"] == "CC BY 4.0"
    assert receipt["remap"]["contract"] == "arwen.gdt101-regional-remap.v1"
    assert receipt["remap"]["window"]["west"] == -43.375
    assert receipt["regridding"].startswith("gdt101_remap regional")
    later = receipt["files"][1]
    assert len(later["native_inputs"]) == len(g101.ICON_GLOBAL_PL.per_lead)
    assert all("PT003H00M" in item["path"] for item in later["native_inputs"])
    assert later["inputs"][0]["path"].endswith("remap/icon-global.f003.grib2")
    assert later["invariants"][0]["path"].endswith("remap/icon-global.invariant.grib2")
    argv = json.loads((out / "met_intermediate.f003.log").read_text()
                      .split("--- stdout ---\n")[1].split("\n--- stderr ---")[0])["argv"]
    assert argv[argv.index("--map-source") + 1] == "dwd"
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--invariant-field"] == ["LANDSEA", "SOILHGT"]
    assert "--select-valid-time" not in argv
    switches = receipt["init_switches"]
    assert switches["--use-spechumd"] == "no" and switches["--nfgsoillevels"] == 8
    assert switches["--extrap-airtemp"] == "constant"  # the 30 hPa top sits below the model top
    # The remapper was planned once and applied to the invariants and each lead,
    # with each lead's own forecast seconds.
    calls = [json.loads(line) for line in (tmp_path / "remapper.log").read_text().splitlines()]
    assert sum("plan" in c for c in calls) == 1
    applied = [c["apply"] for c in calls if "apply" in c]
    assert sorted({tuple(row) for rows in applied for row in rows}) == [
        ("2026101100", "0"), ("2026101100", "10800")]
    plan_args = next(c["plan"] for c in calls if "plan" in c)
    assert plan_args[1] == "0,191,1,1,0.0,255,0.0" and plan_args[-2:] == ["2949120", "78"]
    assert not list((out / "remap").glob("pieces.*"))
    assert (out / "remap" / "target.txt").read_text().split()[:2] == ["-43.375000000", "70.875000000"]


def test_the_model_level_row_reads_its_column_top_from_pressure(tmp_path: Path) -> None:
    stand_ins = _stand_ins(tmp_path)
    _tree(tmp_path / "grib", g101.ICON_GLOBAL_ML, (0, 3))
    request = _icon_request(tmp_path, "icon-global-ml", (0, 3), stand_ins=stand_ins)
    receipt = door.build_global_intermediates(request, log=lambda _m: None)
    first = receipt["files"][0]
    assert first["vertical"] == "model-level" and first["isobaric_levels"] == 0
    assert first["top_pressure_pa"] == 2.0
    switches = receipt["init_switches"]
    assert switches["--use-spechumd"] == "yes"
    assert switches["--extrap-airtemp"] == "lapse-rate"  # the 2 Pa top is above the model top
    assert switches["--nfglevels"] == 4
    argv = json.loads((tmp_path / "out" / "met_intermediate.f000.log").read_text()
                      .split("--- stdout ---\n")[1].split("\n--- stderr ---")[0])["argv"]
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--invariant-field"] == [
        "LANDSEA", "SOILHGT", "GHT"]
    assert argv[argv.index("--vtable") + 1].endswith("Vtable.ICON-GLOBAL-ML.rw")


def test_a_remap_failure_publishes_nothing_and_says_why(tmp_path: Path) -> None:
    stand_ins = _stand_ins(tmp_path, fail="apply")
    _tree(tmp_path / "grib", g101.ICON_GLOBAL_PL, (0,))
    request = _icon_request(tmp_path, "icon-global", (0,), stand_ins=stand_ins)
    with pytest.raises(GlobalIntermediateRefusal, match="gdt101_remap exited 2.*four native donors"):
        door.build_global_intermediates(request, log=lambda _m: None)
    out = tmp_path / "out"
    assert not (out / "remap").exists() and not list(out.glob("MET*"))
    # The log the refusal points at outlives the removed stage.
    assert "four native donors" in (out / "gdt101_remap.apply.invariant.log").read_text()


def test_an_engine_without_the_dwd_label_is_refused_before_the_remap(tmp_path: Path) -> None:
    remapper, _, vtable, log = _stand_ins(tmp_path)
    engine = tmp_path / "met_intermediate_old"
    engine.write_text("#!/bin/sh\necho 'known --map-source labels: ncep-gfs, ecmwf' >&2\nexit 1\n")
    engine.chmod(0o755)
    _tree(tmp_path / "grib", g101.ICON_GLOBAL_PL, (0,))
    request = _icon_request(tmp_path, "icon-global", (0,), stand_ins=(remapper, engine, vtable, log))
    with pytest.raises(GlobalIntermediateRefusal, match="carries no 'dwd' map source"):
        door.build_global_intermediates(request, log=lambda _m: None)
    assert not log.exists() and not (tmp_path / "out").exists()


def test_a_remapper_that_skips_a_record_is_refused_by_name(tmp_path: Path) -> None:
    remapper, engine, vtable, log = _stand_ins(tmp_path)
    lazy = _launcher(tmp_path, "gdt101_remap_lazy", textwrap.dedent('''\
        import sys
        from pathlib import Path
        args = sys.argv[1:]
        if args[0] == "plan":
            Path(args[8]).write_bytes(b"GDT101P3")
        else:
            for row in Path(args[2]).read_text().splitlines()[1:]:
                Path(row.split("\\t")[1]).write_bytes(b"GRIB7777")
    '''))
    _tree(tmp_path / "grib", g101.ICON_GLOBAL_PL, (0,))
    request = _icon_request(tmp_path, "icon-global", (0,), stand_ins=(lazy, engine, vtable, log))
    with pytest.raises(GlobalIntermediateRefusal, match="wrote no output for 1 record.*HSURF"):
        door.build_global_intermediates(request, log=lambda _m: None)
    assert not (tmp_path / "out" / "remap").exists()


def test_an_engine_failure_takes_the_remap_with_it(tmp_path: Path) -> None:
    remapper, _, vtable, log = _stand_ins(tmp_path)
    engine = tmp_path / "met_intermediate_failing"
    engine.write_text("#!/bin/sh\necho 'met_intermediate: unknown --map-source \"dwd\"' >&2\nexit 1\n")
    engine.chmod(0o755)
    _tree(tmp_path / "grib", g101.ICON_GLOBAL_PL, (0,))
    request = _icon_request(tmp_path, "icon-global", (0,), stand_ins=(remapper, engine, vtable, log))
    with pytest.raises(GlobalIntermediateRefusal, match='unknown --map-source "dwd"'):
        door.build_global_intermediates(request, log=lambda _m: None)
    assert not (tmp_path / "out" / "remap").exists()


def test_a_global_target_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    stand_ins = _stand_ins(tmp_path)
    _tree(tmp_path / "grib", g101.ICON_GLOBAL_PL, (0,))
    lat = np.repeat(np.linspace(-89.5, 89.5, 30), 36)
    lon = np.tile(np.arange(-180.0, 180.0, 10.0) + 5.0, 30)
    grid = _mesh_file(tmp_path / "global.nc", lat, lon)
    request = _icon_request(tmp_path, "icon-global", (0,), stand_ins=stand_ins, grid=grid)
    with pytest.raises(GlobalIntermediateRefusal, match="global road"):
        door.build_global_intermediates(request, log=lambda _m: None)
    assert not (tmp_path / "out").exists()


def test_an_out_dir_with_an_earlier_remap_is_refused(tmp_path: Path) -> None:
    stand_ins = _stand_ins(tmp_path)
    _tree(tmp_path / "grib", g101.ICON_GLOBAL_PL, (0,))
    (tmp_path / "out" / "remap").mkdir(parents=True)
    request = _icon_request(tmp_path, "icon-global", (0,), stand_ins=stand_ins)
    with pytest.raises(GlobalIntermediateRefusal, match="fresh --out-dir"):
        door.build_global_intermediates(request, log=lambda _m: None)


def test_plan_leads_refuses_an_unstaged_field_file_row(tmp_path: Path) -> None:
    request = door.GlobalRequest(
        source=door.GLOBAL_SOURCES["icon-global"], grib_dir=tmp_path, cycle=CYCLE,
        leads=(0,), interval_hours=3, out_dir=tmp_path / "out")
    with pytest.raises(GlobalIntermediateRefusal, match="remapped before the engine runs"):
        door.plan_leads(request, Path("/e"), Path("/v"))
