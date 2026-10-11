"""ICON global production contract, with native execution explicitly separated.

Everything here reads the SHIPPED normalization document rather than Python
constants: the point of the fold is that ICON's field codes, level ladders,
soil-depth semantics and cadence are table data.  Transport/cache tests replace
native execution with a receipt-producing fake.  They do NOT claim GRIB decode,
numerical parity, live DWD or stock-WRF validation; the Rust core suite is
executed here only when a Rust compiler is present, and the real-byte decode
fixture lives in tests/test_gdt101_real_bytes.py.
"""
from __future__ import annotations

import bz2
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from woof import source_normalization as norm
from woof import fetch_routes, source_authorities, source_adapters

ROOT = Path(__file__).resolve().parents[1]
CYCLE = datetime(2026, 9, 15, tzinfo=timezone.utc)
NORMALIZER = "icon-gdt101-pressure-v1"


@pytest.fixture(scope="module")
def spec():
    return norm.load_normalization(NORMALIZER)


def request(hours=0, cycle=CYCLE):
    return fetch_routes.resolve_request("icon-global", cycle=cycle, hours=hours)


def inventory(spec, root, hours=0, *, write=False):
    paths = [root / name for name in request(hours).primary_files]
    if write:
        for path in paths:
            # A v1 key's directories are its identity, so each object
            # lands under its own field/level/run tree.
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"GRIBfake-transport-only:"
                             + path.as_posix().encode())
    return norm.validate_inventory(spec, paths)


def wps(path, *, start="2026-09-15_00:00:00", end="2026-09-15_06:00:00",
        interval="10800"):
    path.write_text(f'''&share
 max_dom = 1,
 start_date = '{start}',
 end_date = '{end}',
 interval_seconds = {interval},
/
&geogrid
 parent_id=1,
 parent_grid_ratio=1,
 i_parent_start=1,
 j_parent_start=1,
 e_we=61,
 e_sn=49,
 dx=12000,
 dy=12000,
 map_proj='lambert',
 ref_lat=38.5,
 ref_lon=-97.5,
 truelat1=30.0,
 truelat2=60.0,
 stand_lon=-97.5,
/
''')
    return path


@pytest.mark.parametrize("alias", ["icon-global", "icon", "icon-13km",
                                   "dwd-icon", "dwd-icon-global"])
def test_source_alias_is_global_not_eu(alias, spec):
    row = source_adapters.get_source_adapter(alias)
    assert row.source_id == "icon-global"
    assert row.runnable
    assert row.status == source_adapters.AdapterStatus.RUNNABLE_NOT_CERTIFIED
    assert row.forcing_interval_seconds == 10800
    assert row.coverage_window is None
    assert row.packaged_profile == spec.profile
    assert row.stock_wrf_gate == "live-unchanged-stock-wrf-gate-pending"


def test_reference_three_hour_inventory_and_no_eu_levels(spec):
    plan = request(6)
    assert plan.leads == (0, 3, 6)
    assert len(plan.objects) == 352
    assert len(plan.primary_files) == 352
    objects = norm.validate_inventory(
        spec, [Path("/not-downloaded") / name for name in plan.primary_files])
    assert len(objects) == 352
    invariants = {name for name, row in spec.fields.items()
                  if row["kind"] == "time-invariant"}
    assert invariants == {"CLAT", "CLON", "HSURF", "FR_LAND"}
    assert {o.field for o in objects if o.kind == "time-invariant"} == invariants
    levels = {o.level for o in objects if o.kind == "pressure-level"}
    assert len(levels) == 18 and levels == set(spec.levels_for("T"))
    assert not levels.intersection({775, 825, 875})
    # DWD's open-data v1 keys, plain GRIB2: never the legacy bz2 tree,
    # and never the mirror's year archives.
    assert all(obj.key.startswith("p/") and obj.key.endswith(".grib2")
               for obj in plan.objects)
    assert not any("/icon/grib/" in obj.url or ".bz2" in obj.url
                   or ".zarr" in obj.url for obj in plan.objects)


def test_a_fresh_run_is_asked_of_dwd_v1_alone():
    """The measured example key.  The mirror is the table's second rung,
    but a mirror whose runs can be incomplete is never a fall-through
    behind the publisher: it serves a run only whole, as the head."""

    from woof import fetch_endpoints

    cycle = datetime(2026, 10, 11, 0)
    plan = fetch_routes.resolve_request(
        "icon-global", cycle=cycle, hours=3, now=cycle + timedelta(hours=4))
    assert [rung.name for rung in fetch_endpoints.ladder("icon-global")] == [
        "dwd", "source-coop"]
    assert [rung.name for rung in plan.ladder] == ["dwd"]
    t850 = next(obj for obj in plan.objects
                if obj.key.startswith("p/T/lvt1/100/lv1/85000/")
                and obj.lead == 0)
    assert t850.url == ("https://opendata.dwd.de/weather/nwp/v1/m/icon/p/T/"
                        "lvt1/100/lv1/85000/r/2026-10-11T00:00/s/PT000H00M.grib2")
    coop = fetch_endpoints.endpoint_named("icon-global", "source-coop")
    assert coop.url(t850.key) == (
        "https://data.source.coop/bkr/icon/icon/p/T/lvt1/100/lv1/85000/r/"
        "2026-10-11T00:00/s/PT000H00M.grib2")
    # The local layout mirrors the key below p/ with the run directory's
    # colon dropped: Windows file systems refuse a colon in a file name.
    assert t850.relpath == "T/lvt1/100/lv1/85000/r/2026-10-11T0000/s/PT000H00M.grib2"
    assert not any(":" in obj.relpath for obj in plan.objects)
    keys = {obj.key for obj in plan.objects}
    assert "p/T_SO/lvt1/106/lv1/0.005/r/2026-10-11T00:00/s/PT003H00M.grib2" in keys
    assert "p/W_SO/lvt1/106/lv1/7.29/r/2026-10-11T00:00/s/PT000H00M.grib2" in keys
    assert "p/T_2M/r/2026-10-11T00:00/s/PT003H00M.grib2" in keys
    # Invariants and coordinates are posted at step 0 only, once.
    for field in ("HSURF", "FR_LAND", "CLAT", "CLON"):
        assert [obj.key for obj in plan.objects
                if obj.key.startswith(f"p/{field}/")] == [
            f"p/{field}/r/2026-10-11T00:00/s/PT000H00M.grib2"]
    assert plan.supplement_files == (
        Path("HSURF/r/2026-10-11T0000/s/PT000H00M.grib2"),)
    assert all(row.magic == "GRIB" for row in plan.files)


def test_a_run_dwd_has_dropped_goes_straight_to_the_mirror():
    cycle = datetime(2026, 10, 8, 0)
    plan = fetch_routes.resolve_request(
        "icon-global", cycle=cycle, hours=3, now=datetime(2026, 10, 11, 9))
    assert [rung.name for rung in plan.ladder] == ["source-coop"]
    assert all(obj.url.startswith("https://data.source.coop/bkr/icon/icon/p/")
               for obj in plan.objects)


def test_the_mirror_is_never_asked_for_a_run_before_its_first():
    from woof import fetch_endpoints

    before = datetime(2026, 8, 25, 6)
    assert [rung.name for rung in fetch_endpoints.serving_ladder(
        "icon-global", cycle=before, now=datetime(2026, 10, 11))] == ["dwd"]
    assert [rung.name for rung in fetch_endpoints.serving_ladder(
        "icon-global", cycle=datetime(2026, 8, 25, 12),
        now=datetime(2026, 10, 11))] == ["source-coop"]


def test_the_v1_level_directory_is_the_selector_surface_value(spec):
    """85000 Pa is the 850 hPa key, 0.005 m soil key 5, W_SO by its top."""

    run = "2026-10-11T00:00"
    t850 = norm.parse_object(
        spec, Path(f"/f/T/lvt1/100/lv1/85000/r/{run}/s/PT003H00M.grib2"))
    assert (t850.kind, t850.field, t850.level, t850.lead) == (
        "pressure-level", "T", 850, 3)
    assert t850.selector[4] == 85000.0
    soil = norm.parse_object(
        spec, Path(f"/f/T_SO/lvt1/106/lv1/0.005/r/{run}/s/PT000H00M.grib2"))
    assert soil.level == 5 and soil.selector == (2, 3, 18, 106, .005, 255, 0.)
    water = norm.parse_object(
        spec, Path(f"/f/W_SO/lvt1/106/lv1/0.03/r/{run}/s/PT000H00M.grib2"))
    assert water.level == 3 and water.selector == (2, 3, 20, 106, .03, 106, .09)
    hsurf = norm.parse_object(spec, Path(f"/f/HSURF/r/{run}/s/PT000H00M.grib2"))
    assert (hsurf.kind, hsurf.lead, hsurf.level) == ("time-invariant", None, None)
    # The normalized names are the ones the legacy tree's objects got,
    # so nothing downstream of the remap sees the layout change.
    assert t850.output_name == (
        "icon_global_regular-lat-lon_pressure-level_2026101100_003_850_T.grib2")
    assert soil.output_name == (
        "icon_global_regular-lat-lon_soil-level_2026101100_000_5_T_SO.grib2")
    assert hsurf.output_name == (
        "icon_global_regular-lat-lon_time-invariant_2026101100_HSURF.grib2")


@pytest.mark.parametrize("hour,limit", [(0, 180), (6, 120), (12, 180), (18, 120)])
def test_cycle_specific_horizon(hour, limit, spec):
    cycle = CYCLE.replace(hour=hour)
    assert request(limit, cycle).leads[-1] == limit
    assert spec.horizon_hours(hour) == limit
    with pytest.raises(ValueError):
        request(limit + 3, cycle)


def test_unsupported_cycle_and_cadence_fail_at_plan_time():
    """A173: DWD posts hourly to f078, so --cadence 1 plans there and is
    refused only where the ladder turns three-hourly, naming the lead."""

    with pytest.raises(ValueError):
        request(6, CYCLE.replace(hour=3))
    plan = fetch_routes.resolve_request("icon-global", cycle=CYCLE, hours=6,
                                        cadence=1)
    assert plan.leads == tuple(range(7))
    with pytest.raises(ValueError, match="does not publish f079 at --cadence 1"):
        fetch_routes.resolve_request("icon-global", cycle=CYCLE, hours=81,
                                     cadence=1)


def test_pinned_profile_contains_real_sea_ice_not_missing_policy(spec):
    authorities = source_authorities.packaged_authorities(spec.profile)
    profile = source_authorities.packaged_profile(spec.profile)
    assert profile["input_normalizer"] == NORMALIZER
    assert "input_normalizer" not in source_authorities.packaged_profile(
        "icon-eu-regular-grib2-v1")
    mapping = json.loads(authorities["mapping"].read_text())
    assert (mapping["coordinates"]["vertical"]["levels"]
            == [100 * p for p in spec.levels_for("T")])
    assert "sea_ice_fraction" in mapping["fields"]
    assert "time_binding" not in mapping["fields"]["sea_ice_fraction"]


def test_the_normalization_document_is_pinned_and_self_naming(spec):
    profile = source_authorities.packaged_profile(spec.profile)
    pinned = profile["sha256"]["normalization"]
    assert hashlib.sha256(spec.path.read_bytes()).hexdigest() == pinned
    assert spec.name == NORMALIZER and spec.source_id == "icon-global"
    assert spec.bridge == "gdt101_remap"
    assert spec.native_grid["cells"] == 2_949_120
    assert spec.native_grid["grid_template"] == 101
    assert spec.native_grid["originating_centre"] == 78


@pytest.mark.parametrize("field,mode", [
    ("T", "idw4"), ("FR_LAND", "nearest"), ("T_SO", "soil-nearest"),
    ("W_SO", "soil-nearest"), ("FR_ICE", "seaice-nearest"),
    ("PS", "surface-idw4")])
def test_field_specific_remapping_mode(tmp_path, field, mode, spec):
    assert next(o for o in inventory(spec, tmp_path)
                if o.field == field).mode == mode


def test_soil_key_five_is_half_a_centimetre(tmp_path, spec):
    obj = next(o for o in inventory(spec, tmp_path)
               if o.field == "T_SO" and o.level == 5)
    assert obj.selector == (2, 3, 18, 106, .005, 255, 0.)
    water = next(o for o in inventory(spec, tmp_path)
                 if o.field == "W_SO" and o.level == 3)
    assert water.selector == (2, 3, 20, 106, .03, 106, .09)


@pytest.mark.parametrize("change", ["missing-atmosphere", "missing-ice",
                                    "missing-clat", "duplicate", "mixed-cycle",
                                    "gap"])
def test_inventory_refuses_incomplete_or_ambiguous_state(tmp_path, change, spec):
    # f000..f009: dropping f003 leaves f000, f006, f009, a gap.  (Over
    # f000..f006 it left f000 and f006, which since A173 is a uniform 6 h
    # series the publisher posts and the decode takes.)
    objects = inventory(spec, tmp_path, 9)
    paths = [o.path for o in objects]
    if change == "missing-atmosphere":
        paths.remove(next(o.path for o in objects if o.field == "T"))
    elif change == "missing-ice":
        paths.remove(next(o.path for o in objects if o.field == "FR_ICE"))
    elif change == "missing-clat":
        paths.remove(next(o.path for o in objects if o.field == "CLAT"))
    elif change == "duplicate":
        paths += [Path(str(paths[0]) + ".bz2")]
    elif change == "mixed-cycle":
        mixed = Path(str(paths[-1]).replace("2026-09-15T0000", "2026-09-15T1200"))
        assert mixed != paths[-1]
        paths[-1] = mixed
    else:
        paths = [o.path for o in objects if o.lead != 3]
    with pytest.raises(ValueError):
        norm.validate_inventory(spec, paths)


@pytest.mark.parametrize("name", [
    # The legacy tree's names keep their own contract; v1 keys carry no bz2.
    "icon_global_icosahedral_pressure-level_2026091500_000_775_T.grib2",
    "icon_global_icosahedral_single-level_2026091500_181_PS.grib2",
    "PS/r/2026-09-15T00:00/s/PT000H00M.grib2.bz2",
    # A level off the ladder, in hPa rather than Pa, or on the wrong surface.
    "T/lvt1/100/lv1/77500/r/2026-09-15T00:00/s/PT000H00M.grib2",
    "T/lvt1/100/lv1/850/r/2026-09-15T00:00/s/PT000H00M.grib2",
    "T/lvt1/106/lv1/85000/r/2026-09-15T00:00/s/PT000H00M.grib2",
    "T_SO/lvt1/106/lv1/0.04/r/2026-09-15T00:00/s/PT000H00M.grib2",
    "T_SO/lvt1/106/lv1/5/r/2026-09-15T00:00/s/PT000H00M.grib2",
    "T/r/2026-09-15T00:00/s/PT000H00M.grib2",
    "PS/lvt1/100/lv1/85000/r/2026-09-15T00:00/s/PT000H00M.grib2",
    # f001 was refused here while the contract was three-hourly; DWD posts
    # it (A173).  A lead past the 00Z cycle's f180 is not a time it runs.
    "PS/r/2026-09-15T00:00/s/PT181H00M.grib2",
    "PS/r/2026-09-15T06:00/s/PT123H00M.grib2",
    "PS/r/2026-09-15T03:00/s/PT000H00M.grib2",
    "PS/r/2026-09-15T00:30/s/PT000H00M.grib2",
    "PS/r/2026-09-15T00:00/s/PT000H30M.grib2",
    # The colon-free run directory the fetch writes keeps the same contract.
    "PS/r/2026-09-15T0300/s/PT000H00M.grib2",
    "PS/r/2026-09-15T0030/s/PT000H00M.grib2",
    "PS/r/2026-09-15T000/s/PT000H00M.grib2",
    "HSURF/r/2026-09-15T0000/s/PT003H00M.grib2",
    "UNKNOWN/r/2026-09-15T00:00/s/PT000H00M.grib2",
    # Invariants are posted at step 0 only.
    "HSURF/r/2026-09-15T00:00/s/PT003H00M.grib2",
])
def test_object_contract_rejects_wrong_product_time_or_field(name, spec):
    with pytest.raises(ValueError):
        norm.parse_object(spec, Path("/f") / name)


def test_both_run_directory_spellings_are_one_object(spec):
    """The fetch writes ``2026-09-15T0000`` (no colon, for Windows); a tree
    fetched before that holds ``2026-09-15T00:00``.  Both read, as one."""

    colon_free = norm.parse_object(spec, Path(
        "/f/T/lvt1/100/lv1/85000/r/2026-09-15T0000/s/PT003H00M.grib2"))
    colon = norm.parse_object(spec, Path(
        "/g/T/lvt1/100/lv1/85000/r/2026-09-15T00:00/s/PT003H00M.grib2"))
    assert colon_free.identity == colon.identity == (
        "2026091500", "pressure-level", "T", 3, 850)
    assert colon_free.output_name == colon.output_name
    with pytest.raises(ValueError, match="duplicate object identity"):
        norm.validate_inventory(spec, [colon_free.path, colon.path])


def test_objects_fetched_from_the_legacy_tree_still_prepare(spec):
    """Bytes already on disk from the bz2 tree normalize to the same names."""

    legacy = norm.parse_object(spec, Path(
        "/f/icon_global_icosahedral_soil-level_2026091500_003_5_T_SO.grib2.bz2"))
    v1 = norm.parse_object(spec, Path(
        "/f/T_SO/lvt1/106/lv1/0.005/r/2026-09-15T00:00/s/PT003H00M.grib2"))
    assert legacy.identity == v1.identity
    assert legacy.output_name == v1.output_name == (
        "icon_global_regular-lat-lon_soil-level_2026091500_003_5_T_SO.grib2")
    # The two layouts are one identity, so holding both is a duplicate.
    with pytest.raises(ValueError, match="duplicate object identity"):
        norm.validate_inventory(spec, [legacy.path, v1.path])


def test_targets_cross_dateline_without_becoming_global(spec):
    a = norm.target_from_points(spec, [10., 11., 10.5], [179., -179., 180.])
    b = norm.target_from_points(spec, [10., 11., 10.5], [179., 181., 180.])
    assert a == b
    assert a.west == 178. and (a.nx - 1) * a.dx == 4.
    assert a.south == 9. and a.ny == 25


def test_target_rounds_outward_and_adds_halo(spec):
    t = norm.target_from_points(spec, [30.01, 31.02], [-98.01, -97.03])
    assert t.west <= -99.01
    assert t.west + (t.nx - 1) * t.dx >= -96.03
    assert t.south <= 29.01 and t.south + (t.ny - 1) * t.dy >= 32.02


@pytest.mark.parametrize("lat,lon", [([], []), ([0.], []), ([float("nan")], [0.]),
                                     ([91.], [0.]), ([88.], [0.]),
                                     ([0., 0., 0.], [0., 120., 240.])])
def test_unsupported_geometry_fails_instead_of_silent_cropping(lat, lon, spec):
    with pytest.raises(ValueError):
        norm.target_from_points(spec, lat, lon)


def test_target_limits_match_native_contract(spec):
    step = float(spec.target["step_degrees"])
    with pytest.raises(ValueError):
        norm.TargetWindow(spec, 180., 0., 2, 2, step, step)
    with pytest.raises(ValueError):
        norm.TargetWindow(spec, 0., 0., 2_000_000, 2, step, step)
    with pytest.raises(ValueError):
        norm.TargetWindow(spec, 0., 0., 2, 2, .25, .25)


def test_real_projection_owner_supplies_target(tmp_path, spec):
    p = wps(tmp_path / "namelist.wps")
    target = norm.target_from_wps(spec, p)
    assert target.west < -97.5 < target.west + (target.nx - 1) * target.dx
    assert target.south < 38.5 < target.south + (target.ny - 1) * target.dy
    norm._check_wps_time_coverage(spec, p, inventory(spec, tmp_path, 6))


@pytest.mark.parametrize("interval", ["3600", "10800.5", "0"])
def test_time_contract_rejects_wrong_or_fractional_interval(tmp_path, interval, spec):
    p = wps(tmp_path / "namelist.wps", interval=interval)
    with pytest.raises(ValueError, match="interval_seconds"):
        norm._check_wps_time_coverage(spec, p, inventory(spec, tmp_path, 6))


@pytest.mark.parametrize("start,end", [
    ("2026-09-14_23:59:59", "2026-09-15_06:00:00"),
    ("2026-09-15_00:00:00", "2026-09-15_06:00:01")])
def test_time_contract_requires_both_bracketing_endpoints(tmp_path, start, end, spec):
    p = wps(tmp_path / "namelist.wps", start=start, end=end)
    with pytest.raises(ValueError, match="does not bracket"):
        norm._check_wps_time_coverage(spec, p, inventory(spec, tmp_path, 6))


@pytest.mark.parametrize("compressed", [False, True])
def test_bounded_transport_is_lossless(tmp_path, compressed, spec):
    data = b"GRIB" + bytes(range(256)) * 12
    source, destination = tmp_path / "source", tmp_path / "decoded"
    source.write_bytes(bz2.compress(data) if compressed else data)
    norm._expand(spec, source, destination)
    assert destination.read_bytes() == data


def test_expansion_bomb_and_bad_magic_do_not_leave_outputs(tmp_path, spec):
    tight = bounded_spec(spec, tmp_path, max_expanded_object_bytes=32)
    source, dest = tmp_path / "source", tmp_path / "output"
    for data in [bz2.compress(b"GRIB" + b"x" * 100), bz2.compress(b"NOPE"), b"NOPE"]:
        source.write_bytes(data)
        with pytest.raises(ValueError):
            norm._expand(tight, source, dest)
        assert not dest.exists()


def bounded_spec(spec, tmp_path, **limits):
    """The same shipped document with one declared limit lowered."""

    document = json.loads(spec.path.read_text())
    document["limits"].update(limits)
    path = tmp_path / "bounded.normalization.json"
    path.write_text(json.dumps(document))
    return norm.load_document(path)


def test_snapshot_refuses_changed_source_before_any_native_decode(tmp_path, spec):
    source, dest = tmp_path / "source", tmp_path / "output"
    source.write_bytes(b"GRIBoriginal")
    identity = norm._file_identity(source)
    source.write_bytes(b"GRIBmodified")
    with pytest.raises(ValueError, match="changed"):
        norm._snapshot_expand(spec, source, dest, identity)
    assert not dest.exists() and not dest.with_suffix(".transport").exists()


def test_native_timeout_is_an_actionable_runtime_error(monkeypatch, spec):
    def timedout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 600)
    monkeypatch.setattr(norm.subprocess, "run", timedout)
    with pytest.raises(RuntimeError, match="600-second"):
        norm._run(spec, ["not-executed"])


@pytest.fixture
def staged(tmp_path, monkeypatch, spec):
    """Fake only native execution. All paths, hashes, staging and checks are real."""

    objects = inventory(spec, tmp_path / "raw", write=True)
    bridge = tmp_path / "native-placeholder"
    bridge.write_bytes(spec.contract.encode())
    calls = []

    def fake(_spec, command):
        calls.append(command)
        if command[1] == "plan":
            # plan (PATH SELECTOR) x3  TARGET PLAN CYCLE CELLS CENTRE
            Path(command[-4]).write_bytes(b"FAKE-PLAN-NOT-NUMERICAL-EVIDENCE")
        else:
            values = Path(command[-1]).read_text().strip().split("\t")
            assert len(values) == 12
            Path(values[1]).write_bytes(
                b"GRIBFAKE-NOT-DECODE-EVIDENCE:" + values[2].encode())
    monkeypatch.setattr(norm, "_run", fake)
    return objects, bridge, calls, tmp_path / "cache", spec


def normalize_staged(staged):
    objects, bridge, calls, cache, spec = staged
    step = float(spec.target["step_degrees"])
    return norm.normalize(spec, objects,
                          norm.TargetWindow(spec, -100., 30., 2, 2, step, step),
                          cache, bridge=bridge)


def test_the_plan_call_carries_the_declared_mesh_identity(staged):
    """The cell count, the centre AND every plan record's selector reach the
    binary as arguments, not as constants compiled into it."""

    normalize_staged(staged)
    plan_call = next(c for c in staged[2] if c[1] == "plan")
    assert plan_call[-2:] == ["2949120", "78"]
    # Plan fields in the document's declared order: latitude, longitude,
    # land, each followed by the seven selector values the document pins.
    # CLAT and CLON sit on a producer-LOCAL category; that number lives in
    # the document and nowhere else.
    assert [Path(x).stem for x in plan_call[2:8:2]] == ["CLAT", "CLON",
                                                       "FR_LAND"]
    assert plan_call[3:9:2] == ["0,191,1,1,0.0,255,0.0",
                                "0,191,2,1,0.0,255,0.0",
                                "2,0,0,1,0.0,255,0.0"]


def test_atomic_normalization_and_warm_cache_reuse(staged):
    directory, provenance = normalize_staged(staged)
    assert len(provenance["normalization"]["outputs"]) == 118
    assert len(staged[2]) == 119  # one plan and 118 fields; coordinates are not output
    assert len((directory / "inputs.txt").read_text().splitlines()) == 118
    before = len(staged[2])
    assert normalize_staged(staged) == (directory, provenance)
    assert len(staged[2]) == before
    assert not list(staged[3].glob(".normalizing-*"))
    assert (provenance["normalization"]["request"]["converter"]["sha256"]
            == norm._sha(staged[1]))
    assert len(provenance["normalization"]["request"]["inputs"]) == 120
    # The normalization document is sealed beside the other three authorities.
    assert set(provenance["normalization"]["request"]["authorities"]) == {
        "mapping", "composition", "provenance", "normalization"}


@pytest.mark.parametrize("tamper", ["field", "plan", "list", "provenance-null",
                                    "outputs-null", "field-symlink",
                                    "directory-symlink"])
def test_cache_tampering_is_refused(staged, tamper):
    directory, provenance = normalize_staged(staged)
    row = provenance["normalization"]["outputs"][0]
    if tamper == "field":
        (directory / "fields" / row["name"]).write_bytes(b"changed")
    elif tamper == "plan":
        (directory / "weights.bin").write_bytes(b"changed")
    elif tamper == "list":
        (directory / "inputs.txt").write_text("changed")
    elif tamper == "field-symlink":
        p = directory / "fields" / row["name"]
        content = p.read_bytes()
        p.unlink()
        destination = directory / "outside"
        destination.write_bytes(content)
        p.symlink_to(destination)
    elif tamper == "directory-symlink":
        moved = directory.with_name(directory.name + "-moved")
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
    else:
        if tamper == "provenance-null":
            provenance = None
        else:
            provenance["normalization"]["outputs"] = None
        (directory / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError):
        normalize_staged(staged)


def test_failed_native_stage_never_publishes_cache(staged, monkeypatch):
    def failure(_spec, command):
        raise RuntimeError("intentional native failure")
    monkeypatch.setattr(norm, "_run", failure)
    with pytest.raises(RuntimeError):
        normalize_staged(staged)
    assert list(staged[3].iterdir()) == []


def test_raw_input_and_converter_changes_get_distinct_cache_keys(staged):
    first, _ = normalize_staged(staged)
    staged[0][0].path.write_bytes(b"GRIBchanged-raw")
    second, _ = normalize_staged(staged)
    assert first != second and first.exists()
    staged[1].write_bytes(staged[4].contract.encode() + b"new-binary")
    third, _ = normalize_staged(staged)
    assert third != second and second.exists()


def test_namespace_dry_run_neither_reads_data_nor_resolves_native_binary(
        tmp_path, monkeypatch, spec):
    objects = inventory(spec, tmp_path / "not-downloaded", 6)
    hsurf = next(o.path for o in objects if o.field == "HSURF")
    args = SimpleNamespace(
        wps_namelist=wps(tmp_path / "namelist.wps"),
        mapped_inputs=[o.path for o in objects],
        supplement=[f"{spec.roles['terrain']}={hsurf}"], dry_run=True)

    def fail(*a, **kw):
        raise AssertionError("dry run tried native conversion")
    monkeypatch.setattr(norm, "normalize", fail)
    result = norm.normalize_namespace(spec, args)
    assert result["dry_run"] and result["input_count"] == 352
    assert not (tmp_path / "not-downloaded").exists()


def test_namespace_switches_manifest_inputs_and_provenance_together(
        staged, tmp_path, monkeypatch):
    objects, bridge, _, _, spec = staged
    monkeypatch.setattr(norm, "_resolve_bridge", lambda _spec: bridge)
    original_paths = [o.path for o in objects]
    hsurf = next(o.path for o in objects if o.field == "HSURF")
    args = SimpleNamespace(
        wps_namelist=wps(tmp_path / "namelist.wps", end="2026-09-15_00:00:00"),
        mapped_inputs=original_paths,
        supplement=[f"{spec.roles['terrain']}={hsurf}"], dry_run=False,
        author_input_manifest=tmp_path / "manifest" / "input.json",
        input_list=None)
    result = norm.normalize_namespace(spec, args)
    assert result["output_count"] == 118
    assert args.mapped_inputs != original_paths
    assert all(p.exists() for p in original_paths)
    assert all("regular-lat-lon" in p.name for p in args.mapped_inputs)
    # One row, under the role the profile declares: the packaged provenance
    # document with this stage's receipt added under one key.
    assert args.provenance == [
        f"{spec.roles['provenance']}={result['provenance']}"]
    import json as _json
    from woof.source_authorities import (packaged_authorities,
                                          packaged_authority_sha256)
    bound = _json.loads(Path(result["provenance"]).read_text())
    receipt = bound.pop(norm.NORMALIZATION_RECEIPT_KEY)
    packaged = _json.loads(
        packaged_authorities(spec.profile)["provenance"].read_text())
    assert bound == packaged
    pins = packaged_authority_sha256(spec.profile)
    authorities = receipt["request"]["authorities"]
    assert receipt["request"]["normalizer"] == spec.name
    assert authorities["provenance"] == pins["provenance"]
    assert authorities["normalization"] == pins["normalization"]
    assert all("CLAT" not in p.name and "CLON" not in p.name
               for p in args.mapped_inputs)
    assert "regular-lat-lon" in args.supplement[0]
    assert (args.input_list.read_text().splitlines()
            == [str(p) for p in args.mapped_inputs])


def test_packaged_profile_selects_normalizer_and_owns_supplement_role(tmp_path, spec):
    from woof.source_cli import _parser, _apply_packaged_profile
    args = _parser().parse_args(
        ["--source", "icon-global", "--supplement", str(tmp_path / "HSURF")])
    errors = _apply_packaged_profile(
        args, source_adapters.get_source_adapter("icon-global"), "test")
    assert errors == []
    assert args._packaged_input_normalizer == NORMALIZER
    assert args.supplement == [
        f"{spec.roles['terrain']}={tmp_path / 'HSURF'}"]


def test_native_core_suite_when_compiler_is_available(tmp_path):
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("Rust compiler unavailable; the Rust core suite was NOT executed")
    import os
    executable = tmp_path / ("core-tests.exe" if os.name == "nt" else "core-tests")
    built = subprocess.run(
        [rustc, "--edition=2021", "--test",
         str(ROOT / "tools/grib1_bridge/src/gdt101_remap_core.rs"),
         "-o", str(executable)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stdout + ran.stderr


def test_wizard_namelist_uses_the_experiment_clock(tmp_path, monkeypatch, spec):
    import woof.experiment
    p = wps(tmp_path / "namelist.wps")
    p.write_text("\n".join(line for line in p.read_text().splitlines()
                           if "start_date" not in line and "end_date" not in line))
    with pytest.raises(ValueError, match="experiment-config"):
        norm._check_wps_time_coverage(spec, p, inventory(spec, tmp_path, 6))
    observed = []

    def load(path):
        observed.append(path)
        return SimpleNamespace(start_time=CYCLE.replace(tzinfo=None),
                               run_seconds=21600.)
    monkeypatch.setattr(woof.experiment, "load_experiment", load)
    config = tmp_path / "experiment.toml"
    norm._check_wps_time_coverage(spec, p, inventory(spec, tmp_path, 6), config)
    assert observed == [config]
    with pytest.raises(ValueError, match="does not bracket"):
        norm._check_wps_time_coverage(spec, p, inventory(spec, tmp_path, 3), config)


def test_wps_reversed_time_and_half_declared_pair_fail(tmp_path, spec):
    objects = inventory(spec, tmp_path, 6)
    p = wps(tmp_path / "namelist.wps", start="2026-09-15_06:00:00",
            end="2026-09-15_00:00:00")
    with pytest.raises(ValueError, match="precedes"):
        norm._check_wps_time_coverage(spec, p, objects)
    p.write_text("\n".join(line for line in p.read_text().splitlines()
                           if "end_date" not in line))
    with pytest.raises(ValueError, match="together"):
        norm._check_wps_time_coverage(spec, p, objects)


@pytest.mark.parametrize("cadence,hours", [(1, 3), (3, 6), (6, 12), (12, 24)])
def test_a_series_at_a_whole_multiple_of_the_hour_normalizes(
        tmp_path, cadence, hours, spec):
    """A173: DWD posts hourly to f078, so the normalization declares the
    publisher's hour and takes a uniform series at any whole multiple of
    it, with the namelist's interval equal to the series' spacing."""

    plan = fetch_routes.resolve_request("icon-global", cycle=CYCLE,
                                        hours=hours, cadence=cadence)
    objects = norm.validate_inventory(
        spec, [Path("/not-downloaded") / name for name in plan.primary_files])
    assert sorted({o.lead for o in objects if o.lead is not None}) == list(
        range(0, hours + 1, cadence))
    end = CYCLE + timedelta(hours=hours)
    p = wps(tmp_path / "namelist.wps", end=end.strftime("%Y-%m-%d_%H:%M:%S"),
            interval=str(cadence * 3600))
    norm._check_wps_time_coverage(spec, p, objects)
