"""The global-mesh half of the ICON door: ``woof hex intermediate --source icon-global|icon-global-ml --grid GLOBAL.nc``.

The real remap and writer run in the e2e; here a stand-in ``gdt101_remap``
and ``met_intermediate`` exercise everything the door owns: choosing the
global path from the mesh, finding every object of a lead in the v1 tree,
the selector octets and methods it declares, one plan reused for every
field and lead, the hour-zero invariants, validation before publication,
and the receipt.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import textwrap

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import icon_global_door as icon  # noqa: E402
from woof.hex import met_intermediate_door as door  # noqa: E402
from woof.hex.cli import build_parser, main as hex_main  # noqa: E402
from woof.hex.hrrr_intermediate import IntermediateRefusal, LatLonTarget, wps_record_bytes  # noqa: E402
from woof.hex.met_intermediate_door import GlobalIntermediateRefusal  # noqa: E402

CYCLE = datetime(2026, 9, 27, 12)
#: A coarse stand-in for the 0.125-degree sphere, same conventions.
SMALL = LatLonTarget(south=-90.0, west=0.0, dlat=45.0, dlon=45.0, nx=8, ny=5)
SMALL_GRID = {"nx": 8, "ny": 5, "start_lat": -90.0, "start_lon": 0.0, "dlat": 45.0, "dlon": 45.0}


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------
def write_mesh(path: Path, *, bdy: bool | None = False, area_fraction: float = 1.0,
               sphere: str = "YES", partial: bool = False) -> Path:
    from netCDF4 import Dataset

    n_cells, n_edges, n_vertices = 12, 30, 20
    with Dataset(path, "w") as ds:
        ds.on_a_sphere = sphere
        ds.sphere_radius = 6371229.0
        ds.createDimension("nCells", n_cells)
        ds.createDimension("nEdges", n_edges)
        ds.createDimension("nVertices", n_vertices)
        area = ds.createVariable("areaCell", "f8", ("nCells",))
        area[:] = np.full(n_cells, area_fraction * 4 * np.pi * 6371229.0 ** 2 / n_cells)
        if bdy is not None:
            for name, dim, size in (("bdyMaskCell", "nCells", n_cells), ("bdyMaskEdge", "nEdges", n_edges),
                                    ("bdyMaskVertex", "nVertices", n_vertices)):
                if partial and name == "bdyMaskVertex":
                    continue
                var = ds.createVariable(name, "i4", (dim,))
                values = np.zeros(size, dtype=np.int32)
                if bdy:
                    values[:3] = 7
                var[:] = values
    return path


def test_a_sphere_tiling_mesh_with_an_all_zero_mask_is_global(tmp_path: Path) -> None:
    facts = icon.classify_mesh(write_mesh(tmp_path / "g.nc"))
    assert facts["kind"] == "global" and abs(facts["area_fraction_of_sphere"] - 1) < 1e-9
    assert icon.classify_mesh(write_mesh(tmp_path / "nomask.nc", bdy=None))["kind"] == "global"


def test_a_boundary_zone_is_regional(tmp_path: Path) -> None:
    assert icon.classify_mesh(write_mesh(tmp_path / "r.nc", bdy=True))["kind"] == "regional"


def test_a_partial_sphere_without_a_mask_is_a_regional_cull(tmp_path: Path) -> None:
    facts = icon.classify_mesh(write_mesh(tmp_path / "cull.nc", bdy=None, area_fraction=0.02))
    assert facts["kind"] == "regional" and "2.0% of the sphere" in facts["reason"]


@pytest.mark.parametrize("kwargs,words", [
    ({"area_fraction": 0.4}, "does not tile the sphere"),
    ({"area_fraction": 1.3, "bdy": None}, "does not tile the sphere"),
    ({"sphere": "NO"}, "does not tile the sphere"),
    ({"partial": True}, "not the whole bdyMask triple"),
])
def test_an_unclassifiable_mesh_is_refused(tmp_path: Path, kwargs, words) -> None:
    with pytest.raises(GlobalIntermediateRefusal, match=words):
        icon.classify_mesh(write_mesh(tmp_path / "m.nc", **kwargs))


def test_a_missing_mesh_is_refused(tmp_path: Path) -> None:
    with pytest.raises(GlobalIntermediateRefusal, match="is not a file"):
        icon.classify_mesh(tmp_path / "absent.nc")


# ---------------------------------------------------------------------------
# the seam: regional or global
# ---------------------------------------------------------------------------
def _arguments(tmp_path: Path, *extra: str) -> argparse.Namespace:
    return build_parser().parse_args([
        "intermediate", "--source", "icon-global", "--grib-dir", str(tmp_path),
        "--cycle", "2026-09-27T12", "--out-dir", str(tmp_path / "out"), *extra,
    ])


def test_the_icon_rows_are_admitted_sources() -> None:
    from woof.hex.hrrr_intermediate import INTERMEDIATE_SOURCES

    assert {"icon-global", "icon-global-ml"} <= set(INTERMEDIATE_SOURCES)
    assert set(door.ICON_SOURCES) == set(icon.ICON_GLOBAL_ROWS)
    assert "icon-global" not in door.REFUSED_SOURCES
    # The regional half runs the field-files rows of GLOBAL_SOURCES.
    assert set(door.ICON_SOURCES) <= set(door.GLOBAL_SOURCES)
    assert {door.GLOBAL_SOURCES[name].layout for name in door.ICON_SOURCES} == {"field-files"}


def test_a_global_grid_chooses_the_global_regrid(tmp_path: Path) -> None:
    kind, mesh = door.icon_target_kind(_arguments(tmp_path, "--grid", str(write_mesh(tmp_path / "g.nc"))))
    assert kind == "global" and mesh["n_cells"] == 12


def test_a_regional_grid_or_a_box_is_the_regional_door(tmp_path: Path) -> None:
    kind, mesh = door.icon_target_kind(_arguments(tmp_path, "--grid", str(write_mesh(tmp_path / "r.nc", bdy=True))))
    assert kind == "regional" and mesh["kind"] == "regional"
    assert door.icon_target_kind(_arguments(tmp_path, "--point", "50,10")) == ("regional", None)


def test_no_mesh_and_no_box_is_refused(tmp_path: Path) -> None:
    with pytest.raises(GlobalIntermediateRefusal, match="needs the mesh it is for"):
        door.icon_target_kind(_arguments(tmp_path))


def test_a_box_on_a_global_mesh_is_refused(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path, "--grid", str(write_mesh(tmp_path / "g.nc")), "--point", "50,10")
    with pytest.raises(GlobalIntermediateRefusal, match="global mesh, which takes the full-sphere"):
        door.icon_target_kind(arguments)


def test_a_regional_mesh_reaches_the_regional_door(tmp_path: Path, monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(door, "run_global_intermediate", lambda arguments: seen.append(arguments.source) or 0)
    code = hex_main(["intermediate", "--source", "icon-global", "--grib-dir", str(tmp_path),
                     "--cycle", "2026-09-27T12", "--out-dir", str(tmp_path / "o"),
                     "--grid", str(write_mesh(tmp_path / "r.nc", bdy=True))])
    assert code == 0 and seen == ["icon-global"]


def test_grid_is_refused_by_the_rows_that_do_not_read_it(tmp_path: Path) -> None:
    from woof.hex import hrrr_intermediate as hi

    gfs = build_parser().parse_args([
        "intermediate", "--source", "gfs", "--grib-dir", str(tmp_path), "--cycle",
        "2026-09-27T12", "--out-dir", str(tmp_path / "o"), "--grid", str(tmp_path / "g.nc")])
    with pytest.raises(GlobalIntermediateRefusal, match="--grid places the remap window"):
        door.request_from_arguments(gfs)
    hrrr = build_parser().parse_args([
        "intermediate", "--source", "hrrr", "--grib-dir", str(tmp_path), "--cycle",
        "2026-09-27T12", "--out-dir", str(tmp_path / "o"), "--grid", str(tmp_path / "g.nc"),
        "--point", "40,-100", "--radius-km", "50"])
    with pytest.raises(IntermediateRefusal, match="--grid places the remap window of icon-global"):
        hi.request_from_arguments(hrrr)


def test_box_flags_are_refused_on_the_global_path(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path, "--spacing-deg", "0.1")
    with pytest.raises(GlobalIntermediateRefusal, match="--spacing-deg do not apply to a global mesh"):
        icon.request_from_arguments(arguments, {"kind": "global", "path": "g.nc"})


def test_the_global_request_defaults(tmp_path: Path) -> None:
    request = icon.request_from_arguments(_arguments(tmp_path, "--hours", "0-6"), {"kind": "global"})
    assert request.leads == (0, 3, 6) and request.workers == icon.DEFAULT_REMAP_WORKERS
    assert request.row.map_source == "dwd" and request.row.vtable == "Vtable.ICON-GLOBAL.rw"


# ---------------------------------------------------------------------------
# the field table
# ---------------------------------------------------------------------------
def _field(name: str, rows=icon.SURFACE):
    return next(f for f in rows if f.name == name)


def test_selectors_are_the_octets_measured_on_real_bytes() -> None:
    assert _field("W_SO").selector(0.03) == (2, 3, 20, 106, 0.03, 106, 0.09)
    assert _field("T_SO").selector(0.06) == (2, 3, 18, 106, 0.06, 255, 0.0)
    assert _field("T_2M").selector(None) == (0, 0, 0, 103, 2.0, 255, 0.0)
    assert _field("PMSL").selector(None) == (0, 3, 1, 101, 0.0, 255, 0.0)
    assert _field("FR_ICE").method == "seaice-nearest"
    assert _field("T", icon.PRESSURE_LEVEL).selector(85000.0) == (0, 0, 0, 100, 85000.0, 255, 0.0)
    assert _field("T", icon.MODEL_LEVEL).selector(90.0) == (0, 0, 0, 150, 90.0, 150, 91.0)
    assert _field("HHL", icon.MODEL_LEVEL).selector(90.0) == (0, 3, 6, 150, 90.0, 101, 0.0)
    assert icon.INVARIANTS[1].selector(None) == (0, 3, 6, 1, 0.0, 101, 0.0)
    assert icon.GEOMETRY["CLAT"].selector(None) == (0, 191, 1, 1, 0.0, 255, 0.0)
    with pytest.raises(GlobalIntermediateRefusal, match="not one of ICON's soil layers"):
        _field("W_SO").selector(0.05)


def test_aloft_is_never_land_classed_and_soil_is_land_only() -> None:
    for field in (*icon.PRESSURE_LEVEL, *icon.MODEL_LEVEL):
        assert field.method in ("idw4", "nearest")
    assert {f.method for f in icon.SURFACE if f.levels in ("soil", "soil-layer")} == {"soil-nearest"}


def test_a_job_row_is_the_remap_wire() -> None:
    row = icon.job_row(Path("/a/in.grib2"), Path("/b/out.grib2"), (2, 3, 20, 106, 0.03, 106, 0.09),
                       "soil-nearest", CYCLE, 3)
    assert row.endswith("\n")
    cells = row.rstrip("\n").split("\t")
    assert cells == ["/a/in.grib2", "/b/out.grib2", "2", "3", "20", "106", "0.03", "106", "0.09",
                     "soil-nearest", "2026092712", "10800"]
    with pytest.raises(GlobalIntermediateRefusal, match="tab or newline"):
        icon.job_row(Path("/a/in\t.grib2"), Path("/b"), (0,) * 7, "idw4", CYCLE, 0)


# ---------------------------------------------------------------------------
# the v1 tree
# ---------------------------------------------------------------------------
PL = ("3000", "50000", "85000")


def make_tree(root: Path, *, leads=(0,), profile=icon.PRESSURE_LEVEL, levels=PL,
              skip: tuple[str, ...] = ()) -> Path:
    tree = root / "p"

    def touch(field, text, lead):
        path = icon.object_path(tree, field, text, CYCLE, lead)
        if f"{field.name}@{text}" in skip or field.name in skip:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GRIB" + f"{field.name}{text}{lead}".encode() + b"7777")

    for lead in leads:
        for field in (*profile, *icon.SURFACE):
            if field.levels is None:
                touch(field, None, lead)
            elif field.levels == "soil":
                for depth in ("0.0", "0.005", "0.02"):
                    touch(field, depth, lead)
            elif field.levels == "soil-layer":
                for top in ("0.0", "0.01", "0.03"):
                    touch(field, top, lead)
            elif field.levels == "half":
                for k in range(1, len(levels) + 2):
                    touch(field, str(k), lead)
            else:
                for text in levels:
                    touch(field, text, lead)
    for field in (*icon.GEOMETRY.values(), *icon.INVARIANTS):
        touch(field, None, 0)
    return root


def test_every_object_of_a_lead_is_found(tmp_path: Path) -> None:
    root = icon.tree_root(make_tree(tmp_path))
    objects = icon.locate_lead(root, icon.ICON_GLOBAL_ROWS["icon-global"], CYCLE, 0)
    labels = [o.label for o in objects]
    assert len(objects) == 5 * len(PL) + 10 + 3 + 3
    assert "T@3000" in labels and "W_SO@0.01" in labels and "PS" in labels
    # Levels are numerically ordered, not lexically.
    assert [o.level_text for o in objects if o.field.name == "T"] == list(PL)
    assert set(icon.locate_invariants(root, CYCLE)) == {"CLAT", "CLON", "FR_LAND", "HSURF"}


def test_a_level_one_field_lacks_is_named(tmp_path: Path) -> None:
    root = icon.tree_root(make_tree(tmp_path, skip=("FI@50000",)))
    with pytest.raises(GlobalIntermediateRefusal, match="FI at 50000"):
        icon.locate_lead(root, icon.ICON_GLOBAL_ROWS["icon-global"], CYCLE, 0)


def test_a_missing_surface_field_and_mesh_record_are_named(tmp_path: Path) -> None:
    root = icon.tree_root(make_tree(tmp_path, skip=("PS", "CLON")))
    with pytest.raises(GlobalIntermediateRefusal, match="p/PS/r/2026-09-27T12:00/s/PT000H00M.grib2"):
        icon.locate_lead(root, icon.ICON_GLOBAL_ROWS["icon-global"], CYCLE, 0)
    with pytest.raises(GlobalIntermediateRefusal, match="CLON"):
        icon.locate_invariants(root, CYCLE)


def test_model_levels_need_one_more_half_level(tmp_path: Path) -> None:
    root = icon.tree_root(make_tree(tmp_path, profile=icon.MODEL_LEVEL, levels=("1", "2", "3")))
    objects = icon.locate_lead(root, icon.ICON_GLOBAL_ROWS["icon-global-ml"], CYCLE, 0)
    assert len([o for o in objects if o.field.name == "HHL"]) == 4
    root2 = icon.tree_root(make_tree(tmp_path / "b", profile=icon.MODEL_LEVEL, levels=("1", "2", "3"),
                                     skip=("HHL@4",)))
    with pytest.raises(GlobalIntermediateRefusal, match="HHL: 3 half levels for 3 full levels"):
        icon.locate_lead(root2, icon.ICON_GLOBAL_ROWS["icon-global-ml"], CYCLE, 0)


def test_the_tree_root_is_found_or_refused(tmp_path: Path) -> None:
    make_tree(tmp_path)
    assert icon.tree_root(tmp_path) == tmp_path / "p"
    assert icon.tree_root(tmp_path / "p") == tmp_path / "p"
    (tmp_path / "flat").mkdir()
    with pytest.raises(GlobalIntermediateRefusal, match="holds no p/ tree"):
        icon.tree_root(tmp_path / "flat")


# ---------------------------------------------------------------------------
# the remap engine and the memory gate
# ---------------------------------------------------------------------------
def test_a_remap_build_without_the_global_mode_is_refused(tmp_path: Path) -> None:
    stale = tmp_path / "gdt101_remap"
    stale.write_bytes(b"\x7fELF arwen.gdt101-regional-remap.v1 usage")
    with pytest.raises(GlobalIntermediateRefusal, match="predates gdt101_remap --global"):
        icon.resolve_remap(stale)
    current = tmp_path / "gdt101_remap_new"
    current.write_bytes(b"\x7fELF " + icon.REMAP_GLOBAL_MODE.encode())
    assert icon.resolve_remap(current) == current.absolute()
    foreign = tmp_path / "other"
    foreign.write_bytes(b"\x7fELF something else")
    with pytest.raises(GlobalIntermediateRefusal, match="does not speak"):
        icon.resolve_remap(foreign)


def test_a_lead_that_cannot_fit_in_memory_is_refused() -> None:
    icon.refuse_unaffordable_lead(119, available=8 * 10 ** 9, workers=4)
    with pytest.raises(GlobalIntermediateRefusal, match="about 23.9 GB"):
        icon.refuse_unaffordable_lead(721, available=8 * 10 ** 9)
    with pytest.raises(GlobalIntermediateRefusal, match="pass fewer --workers"):
        icon.refuse_unaffordable_lead(10, available=2 * 10 ** 9, workers=13)


def test_workers_are_bounded(tmp_path: Path) -> None:
    with pytest.raises(GlobalIntermediateRefusal, match="not 1..13"):
        icon.request_from_arguments(_arguments(tmp_path, "--workers", "32"), {"kind": "global"})


def test_a_missing_icon_vtable_names_the_map_source_change(tmp_path: Path) -> None:
    import dataclasses

    row = dataclasses.replace(icon.ICON_GLOBAL_ROWS["icon-global"], vtable="Vtable.NOT-HERE.rw")
    with pytest.raises(GlobalIntermediateRefusal, match="--map-source dwd"):
        icon._vtable_for(row, None)


# ---------------------------------------------------------------------------
# the whole door against stand-ins
# ---------------------------------------------------------------------------
def write_global_synthetic(path: Path, hdate: str) -> int:
    """A valid intermediate on the coarse stand-in sphere."""

    slab = np.ones((SMALL.ny, SMALL.nx), dtype=np.float32)
    records = []
    for level in (200100.0, 100000.0, 50000.0, 3000.0):
        for name, value in (("TT", 280.0), ("RH", 50.0), ("UU", 5.0), ("VV", -2.0)):
            records.append((name, level, value))
        if level != 200100.0:
            records.append(("GHT", level, 1000.0))
    for name, value in (("PSFC", 101000.0), ("SOILHGT", 10.0), ("SKINTEMP", 285.0), ("LANDSEA", 1.0),
                        ("SOILT001", 283.0), ("SOILT002", 283.0), ("SOILM001", 0.3), ("SOILM002", 0.3)):
        records.append((name, 200100.0, value))
    with path.open("wb") as handle:
        for name, level, value in records:
            handle.write(wps_record_bytes(
                hdate=hdate, xfcst=0.0, map_source="DWD", field_name=name, units="-",
                description=name, level=level, target=SMALL, values=slab * value))
    return len(records)


FAKE_ENGINE = textwrap.dedent('''\
    import json, sys
    sys.path[:0] = {paths!r}
    from pathlib import Path
    from test_icon_global_door import write_global_synthetic
    argv = sys.argv[1:]
    hdate = argv[argv.index("--date") + 1]
    out = argv[argv.index("--out") + 1]
    n = write_global_synthetic(Path(out), hdate)
    Path(out + ".argv.json").write_text(json.dumps(argv))
    print(json.dumps({{"schema": {schema!r}, "matched_valid_times": [hdate],
                      "records_written": n}}))
''')

FAKE_REMAP = textwrap.dedent('''\
    import json, sys, shutil
    from pathlib import Path
    argv = sys.argv[1:]
    assert "--global" in argv, argv
    argv = [a for a in argv if a != "--global"]
    target = {{"west": 0, "south": -90, "dx": 45.0, "dy": 45.0, "nx": 8, "ny": 5}}
    common = {{"contract": {contract!r}, "mode": {mode!r}, "global": True, "target": target,
               "grid_uuid": {uuid!r}}}
    log = Path({log!r})
    if argv[0] == "plan":
        i = argv.index("--dx-deg"); dx = argv[i + 1]; del argv[i:i + 2]
        plan = Path(argv[7]); plan.write_bytes(b"GDT101G1")
        with log.open("a") as f: f.write("plan " + dx + " " + " ".join(argv[1:7:2]) + "\\n")
        print(json.dumps(dict(common, verb="plan", source_cells=2949120, total_seconds=0.1)))
    else:
        plan, jobs = argv[1], argv[2]
        fields = []
        for line in Path(jobs).read_text().splitlines():
            cells = line.split("\\t")
            shutil.copyfile(cells[0], cells[1])
            with log.open("a") as f: f.write("apply " + plan + " " + " ".join(cells[2:12]) + "\\n")
            fields.append({{"output": cells[1], "method": cells[9], "total_seconds": 0.01}})
        print(json.dumps(dict(common, verb="apply", plan_read_seconds=0.01, fields=fields)))
''')


def _launcher(tmp_path: Path, name: str, source: str) -> Path:
    script = tmp_path / f"fake_{name}.py"
    script.write_text(source)
    launcher = tmp_path / name
    launcher.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
    launcher.chmod(0o755)
    return launcher


@pytest.fixture
def stand_ins(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(icon, "GLOBAL_GRID", dict(SMALL_GRID))
    monkeypatch.setattr(icon, "GLOBAL_DX_DEG", 45.0)
    log = tmp_path / "remap.log"
    remap = _launcher(tmp_path, "gdt101_remap", FAKE_REMAP.format(
        contract=icon.REMAP_CONTRACT, mode=icon.REMAP_GLOBAL_MODE, uuid=icon.ICON_GRID_UUID,
        log=str(log)))
    # The door checks the launcher it runs for the global-mode marker.
    remap.write_text(remap.read_text() + f"# {icon.REMAP_GLOBAL_MODE}\n")
    engine = _launcher(tmp_path, "met_intermediate", FAKE_ENGINE.format(
        paths=[str(Path(__file__).parent), *sys.path], schema=door.ENGINE_RECEIPT_SCHEMA))
    vtable = tmp_path / "Vtable.ICON-GLOBAL.rw"
    vtable.write_text("stand-in\n")
    return remap, engine, vtable, log


def _request(tmp_path: Path, stand_ins, leads=(0, 3), **kw) -> icon.IconGlobalRequest:
    remap, engine, vtable, _ = stand_ins
    return icon.IconGlobalRequest(
        row=icon.ICON_GLOBAL_ROWS["icon-global"], grib_dir=make_tree(tmp_path / "grib", leads=leads),
        cycle=CYCLE, leads=leads, interval_hours=3, out_dir=tmp_path / "out",
        mesh={"kind": "global", "path": "g.nc"}, engine=engine, vtable=vtable, remap=remap,
        workers=2, **kw)


def test_the_door_plans_once_remaps_every_field_and_receipts(tmp_path: Path, stand_ins) -> None:
    receipt = icon.build_icon_global_intermediates(_request(tmp_path, stand_ins), log=lambda _m: None)
    out = tmp_path / "out"
    assert [Path(f["path"]).name for f in receipt["files"]] == ["MET:2026-09-27_12", "MET:2026-09-27_15"]
    assert not list(out.glob("*.partial")) and not (out / ".icon-global-regrid").exists()
    log = (tmp_path / "remap.log").read_text().splitlines()
    plans = [line for line in log if line.startswith("plan")]
    assert len(plans) == 1 and plans[0].startswith("plan 45.0 ")
    assert "CLAT" in plans[0] and "CLON" in plans[0] and "FR_LAND" in plans[0]
    applies = [line for line in log if line.startswith("apply")]
    per_lead = 5 * len(PL) + 10 + 3 + 3
    assert len(applies) == 2 + 2 * per_lead  # invariants once, then each lead
    assert len({line.split()[1] for line in applies}) == 1  # one plan for all
    assert any(" 2 3 20 106 0.01 106 0.03 soil-nearest 2026092712 10800" in line for line in applies)
    assert receipt["attribution"] == "Source: Deutscher Wetterdienst (DWD), CC BY 4.0"
    regrid = receipt["regridding"]
    assert regrid["global"] is True and regrid["contract"] == "arwen.gdt101-regional-remap.v1"
    assert regrid["mode"] == icon.REMAP_GLOBAL_MODE and regrid["source_grid_uuid"] == icon.ICON_GRID_UUID
    first, later = receipt["files"]
    assert first["regridded_grib"]["fields"] == later["regridded_grib"]["fields"] == per_lead
    assert first["regridded_grib"]["invariants_as"] == "inputs"
    assert later["regridded_grib"]["invariants_as"] == "--invariant"
    assert first["remap"]["fields"] == per_lead
    assert first["grid"] == dict(SMALL_GRID, projection="latlon")
    argv0 = json.loads((out / "MET:2026-09-27_12.partial.argv.json").read_text())
    argv3 = json.loads((out / "MET:2026-09-27_15.partial.argv.json").read_text())
    assert argv0[argv0.index("--map-source") + 1] == "dwd" and "--invariant" not in argv0
    assert sum(a.endswith(".grib2") for a in argv0) == per_lead + 2  # each part, no copy
    assert argv3.count("--invariant") == 2 and "SOILHGT" in argv3
    assert argv3[argv3.index("--invariant-field") + 1] == "LANDSEA"
    assert sum(a.endswith(".grib2") for a in argv3) == per_lead + 2
    assert json.loads((out / "intermediate-receipt.json").read_text())["schema"] == icon.SCHEMA


def test_an_intermediate_off_the_sphere_is_not_published(tmp_path: Path, stand_ins, monkeypatch) -> None:
    monkeypatch.setattr(icon, "GLOBAL_GRID", dict(SMALL_GRID, nx=9))
    request = _request(tmp_path, stand_ins, leads=(0,))
    with pytest.raises(GlobalIntermediateRefusal):
        icon.build_icon_global_intermediates(request, log=lambda _m: None)
    out = tmp_path / "out"
    assert not list(out.glob("MET:*")) or all(p.name.endswith(".argv.json") for p in out.glob("MET:*"))
    assert not list((out / ".icon-global-regrid").glob("*.grib2"))


def test_a_regional_mesh_never_reaches_the_global_regrid(tmp_path: Path, stand_ins) -> None:
    request = _request(tmp_path, stand_ins, leads=(0,))
    request.mesh = {"kind": "regional", "path": "r.nc"}
    with pytest.raises(GlobalIntermediateRefusal, match="is for a global mesh"):
        icon.build_icon_global_intermediates(request, log=lambda _m: None)
