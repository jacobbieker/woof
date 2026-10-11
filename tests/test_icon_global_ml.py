"""ICON global on its 120 model levels is rows on the DWD v1 route.

Every assertion reads a shipped document -- the registry row, the fetch route
table, the normalization authority or the packaged profile -- so a fact that
drifts between two of them fails here.  The real-byte checks read the GRIB2
inventory of 38 objects fetched from DWD's v1 tree for the 2026-10-11 00 UTC
run (tests/fixtures/icon-global-ml-2026101100-headers.json).

Source: Deutscher Wetterdienst (DWD), CC BY 4.0.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from woof import fetch_routes, source_adapters, source_authorities
from woof import source_normalization as norm

FIXTURE = (Path(__file__).with_name("fixtures")
           / "icon-global-ml-2026101100-headers.json")
NORMALIZER = "icon-global-ml-gdt101-model-level-v1"
PROFILE = "icon-global-ml-grib2-v1"
CYCLE = datetime(2026, 10, 11, 0)
#: The run as DWD's key spells it, and as the fetch writes it on disk
#: (no colon: Windows file systems refuse one in a file name).
RUN = "2026-10-11T00:00"
LOCAL_RUN = "2026-10-11T0000"
MODEL_LEVELS = tuple(range(1, 121))
HALF_LEVELS = tuple(range(1, 122))
MESH_CELLS = 2949120
#: 600 model-level, 9 single-level and 17 soil objects per lead.
PER_LEAD = 626
#: 121 HHL, HSURF, FR_LAND, CLAT and CLON once per run.
PER_RUN = 125


@pytest.fixture(scope="module")
def spec():
    return norm.load_normalization(NORMALIZER)


@pytest.fixture(scope="module")
def mapping(spec):
    path = source_authorities.packaged_authorities(spec.profile)["mapping"]
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def headers():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["rows"]


# ---------------------------------------------------------------------------
# The registry row
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alias", ["icon-global-ml", "icon-ml",
                                   "icon-global-model-level",
                                   "dwd-icon-global-ml", "ICON_GLOBAL_ML"])
def test_every_spelling_names_the_one_row(alias, spec):
    row = source_adapters.get_source_adapter(alias)
    assert row.source_id == "icon-global-ml"
    assert row.runnable and row.runner == "mapped_composition_v1"
    assert row.status == source_adapters.AdapterStatus.RUNNABLE_NOT_CERTIFIED
    assert row.packaged_profile == spec.profile == PROFILE
    assert row.max_forecast_hour == 180
    assert "Deutscher Wetterdienst" in row.notes and "CC BY 4.0" in row.notes
    # The pressure-level source keeps its own spellings.
    assert source_adapters.get_source_adapter("icon").source_id == "icon-global"


def test_the_registry_default_is_the_routes_and_a_multiple_the_mapping_takes(mapping):
    row = source_adapters.get_source_adapter("icon-global-ml")
    route = fetch_routes.route_for("icon-global-ml")
    assert row.forcing_interval_seconds == route.default_cadence * 3600 == 10800
    assert mapping["target"]["boundary_interval_seconds"] == 3600
    assert mapping["target"]["accept_boundary_interval_multiples"] is True


# ---------------------------------------------------------------------------
# One ladder, one cadence, one mesh across the documents
# ---------------------------------------------------------------------------

def test_route_document_and_mapping_declare_one_ladder_and_cadence(spec, mapping):
    route = fetch_routes.route_for("icon-global-ml")
    levels = {name: tuple(group.get("levels") or ())
              for groups in route.axes.values() for group in groups
              for name in group["names"]}
    for field in ("P", "QV", "T", "U", "V"):
        assert levels[field] == MODEL_LEVELS
        assert tuple(sorted(spec.levels_for(field))) == MODEL_LEVELS
    assert levels["HHL"] == HALF_LEVELS == spec.levels_for("HHL")
    vertical = mapping["coordinates"]["vertical"]
    assert vertical["kind"] == "model_level"
    assert vertical["levels"] == list(MODEL_LEVELS)
    assert vertical["interface_levels"] == list(HALF_LEVELS)
    assert (mapping["target"]["boundary_interval_seconds"]
            == spec.cadence["forcing_interval_seconds"] == 3600)
    assert list(route.cycle_hours) == list(spec.cadence["cycle_hours"]) == [0, 6, 12, 18]
    for hour, horizon in ((0, 180), (6, 120), (12, 180), (18, 120)):
        assert spec.horizon_hours(hour) == horizon
        assert fetch_routes.ladder_for(route, CYCLE.replace(hour=hour))[-1] == horizon
    assert spec.native_grid == {"cells": MESH_CELLS, "grid_template": 101,
                                "originating_centre": 78}
    # The same window the pressure-level source remaps onto.
    pressure = norm.load_normalization("icon-gdt101-pressure-v1")
    assert spec.target == pressure.target


def test_the_profile_pins_its_normalization_and_owns_its_roles(spec):
    profile = source_authorities.packaged_profile(PROFILE)
    assert profile["input_normalizer"] == NORMALIZER
    assert (source_authorities.packaged_normalization(NORMALIZER).name
            == "rw-wps-icon-global-ml-grib2.normalization.json")
    assert spec.roles == {"terrain": "icon_global_ml_invariant_surface",
                          "provenance": "icon_global_ml_invariant_surface_provenance"}
    assert (profile["data_role"], profile["provenance_role"]) == (
        spec.roles["terrain"], spec.roles["provenance"])
    route = fetch_routes.route_for("icon-global-ml")
    assert route.prep["supplement"]["role"] == spec.roles["terrain"]


def test_height_places_every_mass_level_and_pressure_is_published(mapping):
    fields = mapping["fields"]
    assert fields["geopotential_height"]["derivation"] == "height-from-interfaces"
    derivation = next(row for row in mapping["derivations"]
                      if row["name"] == "height-from-interfaces")
    assert derivation == {"name": "height-from-interfaces",
                          "operation": "height_from_interfaces",
                          "source": "interface_height"}
    assert fields["interface_height"]["source_axes"] == ["half_level", "y", "x"]
    assert fields["interface_height"]["time_binding"] == "cycle_invariant"
    for name in ("air_temperature", "eastward_wind", "northward_wind",
                 "specific_humidity", "air_pressure"):
        selectors = fields[name]["selectors"]
        assert [s["level_value"] for s in selectors] == list(MODEL_LEVELS)
        assert [s["second_level_value"] for s in selectors] == [n + 1 for n in MODEL_LEVELS]
        assert {s["level_type"] for s in selectors} == {150}
    # No pressure-level relic: nothing reads relative humidity aloft or a
    # pressure coordinate.
    assert "relative_humidity" not in fields and "geopotential" not in fields
    assert not any(row["operation"] == "pressure_from_vertical_coordinate"
                   for row in mapping["derivations"])
    assert mapping["target"]["pressure_requirement"] == "air_pressure"


# ---------------------------------------------------------------------------
# The fetch: v1 tokens, the tree layout, and the window ceiling
# ---------------------------------------------------------------------------

def test_objects_are_named_by_the_v1_tree_below_p_on_both_rungs():
    plan = fetch_routes.resolve_request("icon-global-ml", cycle=CYCLE, hours=1,
                                        cadence=1)
    names = {obj.relpath for obj in plan.objects}
    assert f"T/lvt1/150/lv1/120/r/{LOCAL_RUN}/s/PT001H00M.grib2" in names
    assert f"HHL/lvt1/150/lv1/121/r/{LOCAL_RUN}/s/PT000H00M.grib2" in names
    assert f"T_SO/lvt1/106/lv1/0.005/r/{LOCAL_RUN}/s/PT000H00M.grib2" in names
    assert f"W_SO/lvt1/106/lv1/7.29/r/{LOCAL_RUN}/s/PT001H00M.grib2" in names
    assert f"PS/r/{LOCAL_RUN}/s/PT001H00M.grib2" in names
    # HHL is published at step 0 only, whatever lead the window starts at.
    assert not any(name.startswith("HHL/") and "PT000H00M" not in name
                   for name in names)
    # The key keeps DWD's colon; the local path is the key below p/ with
    # the run directory's colon dropped.
    for obj in plan.objects:
        assert obj.key.startswith("p/") and f"/r/{RUN}/" in obj.key
        assert obj.relpath == obj.key[2:].replace(RUN, LOCAL_RUN)
        assert ":" not in obj.relpath
        assert obj.url == f"https://opendata.dwd.de/weather/nwp/v1/m/icon/{obj.key}"
    assert plan.supplement_files == (Path(f"HSURF/r/{LOCAL_RUN}/s/PT000H00M.grib2"),)
    route = fetch_routes.route_for("icon-global-ml")
    assert (route.layout, route.mirror_prefix) == ("mirror", "p/")
    assert route.attribution == "Source: Deutscher Wetterdienst (DWD), CC BY 4.0"
    # The same two rungs as icon-global, the mirror's runs listed first.
    pressure = fetch_routes.route_for("icon-global")
    assert ([(h.name, h.base, h.retention_hours, h.serves_from) for h in route.hosts]
            == [(h.name, h.base, h.retention_hours, h.serves_from)
                for h in pressure.hosts])
    mirror = route.host("source-coop")
    assert mirror.base == "https://data.source.coop/bkr/icon/icon"
    assert mirror.retention_hours is None and mirror.run_listing is not None


def test_a_later_window_still_fetches_the_step0_invariants():
    plan = fetch_routes.resolve_request("icon-global-ml", cycle=CYCLE, hours=3,
                                        start_hour=6)
    assert plan.leads == (6, 9)
    heights = [obj for obj in plan.objects if obj.role == "heights"]
    assert len(heights) == 121
    assert all(obj.key.endswith("/s/PT000H00M.grib2") for obj in heights)


def test_a_window_beyond_the_ceiling_is_refused_before_any_byte_moves():
    route = fetch_routes.route_for("icon-global-ml")
    limit = route.window_limit["max_leads"]
    assert limit == 13
    plan = fetch_routes.resolve_request("icon-global-ml", cycle=CYCLE,
                                        hours=3 * (limit - 1))
    assert len(plan.leads) == limit
    with pytest.raises(ValueError) as refused:
        fetch_routes.resolve_request("icon-global-ml", cycle=CYCLE, hours=3 * limit)
    text = str(refused.value)
    assert "asks for 14 leads" in text and "at most 13" in text
    assert "why:" in text and "2.0 GB" in text
    assert "remedy:" in text and "--source icon-global" in text
    # The refusal names the finest spacing that fits the same window.
    with pytest.raises(ValueError, match=r"--cadence 4 is 13 leads and fits"):
        fetch_routes.resolve_request("icon-global-ml", cycle=CYCLE, hours=48)
    # The ceiling counts leads, so a coarser cadence reaches further.
    assert len(fetch_routes.resolve_request(
        "icon-global-ml", cycle=CYCLE, hours=72, cadence=6).leads) == 13


def test_the_normalization_bound_holds_the_largest_window_the_route_takes(spec):
    limit = fetch_routes.route_for("icon-global-ml").window_limit["max_leads"]
    assert spec.limits["max_input_files"] == limit * PER_LEAD + PER_RUN


def test_the_route_table_loader_refuses_an_unknown_layout_or_bare_ceiling(monkeypatch):
    document = json.loads(fetch_routes._table_path().read_text(encoding="utf-8"))
    for change, words in (
            (lambda row: row.update(layout="tree"), "declares layout 'tree'"),
            (lambda row: row.update(window_limit={"max_leads": 13}), "window_limit")):
        broken = json.loads(json.dumps(document))
        change(broken["routes"]["icon-global-ml"])
        monkeypatch.setattr(fetch_routes, "_load_table", lambda doc=broken: doc)
        with pytest.raises(ValueError, match=words):
            fetch_routes._build_routes()


def test_tokens_render_the_v1_spellings():
    context = fetch_routes._cycle_context(datetime(2026, 8, 25, 12))
    assert context["RUN_ISO"] == "2026-08-25T12:00"
    cycle = datetime(2026, 8, 25, 12)
    assert fetch_routes._lead_context(context, cycle, 7)["STEP_ISO"] == "PT007H00M"
    assert fetch_routes._lead_context(context, cycle, 180)["STEP_ISO"] == "PT180H00M"
    assert {"RUN_ISO", "STEP_ISO"} <= fetch_routes.PATH_TOKENS
    route = SimpleNamespace(axes=MappingProxyType({"soil": (MappingProxyType({
        "names": ["T_SO"], "levels": ["0.0", "0.005", "14.58"]}),)}))
    assert [entry["LEVEL"] for entry in fetch_routes._axis_entries(route, "soil")] \
        == ["0.0", "0.005", "14.58"]


# ---------------------------------------------------------------------------
# What the fetch downloads is what the normalization accepts
# ---------------------------------------------------------------------------

def test_a_fetched_window_is_exactly_the_inventory_the_normalization_accepts(spec):
    plan = fetch_routes.resolve_request("icon-global-ml", cycle=CYCLE, hours=6)
    paths = [Path("/not-downloaded/icon") / name for name in plan.primary_files]
    objects = norm.validate_inventory(spec, paths)
    assert len(objects) == len(paths) == PER_LEAD * 3 + PER_RUN
    assert ({o.field for o in objects if o.kind == "time-invariant"}
            == {"CLAT", "CLON", "FR_LAND", "HSURF", "HHL"})
    assert {o.level for o in objects if o.field == "T"} == set(MODEL_LEVELS)
    assert {o.level for o in objects if o.field == "HHL"} == set(HALF_LEVELS)
    assert {o.lead for o in objects if o.lead is not None} == {0, 3, 6}
    assert all(o.cycle == "2026101100" for o in objects)
    outputs = [o.output_name for o in objects]
    assert len(set(outputs)) == len(outputs)
    assert all("/" not in name for name in outputs)


def test_every_invariant_height_is_required(spec):
    plan = fetch_routes.resolve_request("icon-global-ml", cycle=CYCLE, hours=3)
    paths = [Path("/x") / name for name in plan.primary_files
             if not str(name).startswith("HHL/lvt1/150/lv1/121/")]
    with pytest.raises(ValueError, match="missing coordinate/invariant"):
        norm.validate_inventory(spec, paths)


@pytest.mark.parametrize("run", [LOCAL_RUN, RUN])
def test_both_run_directory_spellings_read_as_one_object(run, spec):
    """The colon-free directory the fetch writes, and the colon one that
    fetches before it wrote, are the same object with the same output."""

    t90 = norm.parse_object(
        spec, Path(f"/data/icon/T/lvt1/150/lv1/90/r/{run}/s/PT003H00M.grib2"))
    assert t90.identity == ("2026101100", "model-level", "T", 3, 90)
    assert t90.output_name == (
        "icon_global_ml_regular-lat-lon_model-level_2026101100_003_90_T.grib2")
    soil = norm.parse_object(
        spec, Path(f"/data/icon/T_SO/lvt1/106/lv1/0.005/r/{run}/s/PT000H00M.grib2"))
    assert (soil.field, soil.level, soil.lead) == ("T_SO", "0.005", 0)
    hhl = norm.parse_object(
        spec, Path(f"/data/icon/HHL/lvt1/150/lv1/121/r/{run}/s/PT000H00M.grib2"))
    assert (hhl.kind, hhl.level, hhl.lead) == ("time-invariant", 121, None)


@pytest.mark.parametrize("key", [
    # temperature on a pressure level is not a model level
    f"T/lvt1/100/lv1/100/r/{RUN}/s/PT000H00M.grib2",
    # a full level past the 120 DWD publishes
    f"T/lvt1/150/lv1/121/r/{RUN}/s/PT000H00M.grib2",
    # half-level heights only exist at step 0
    f"HHL/lvt1/150/lv1/10/r/{RUN}/s/PT003H00M.grib2",
    # a level on a single-level field
    f"PS/lvt1/150/lv1/1/r/{RUN}/s/PT000H00M.grib2",
    # a soil depth the TERRA column does not have
    f"T_SO/lvt1/106/lv1/0.1/r/{RUN}/s/PT000H00M.grib2",
    # a field the route does not read
    f"QC/lvt1/150/lv1/1/r/{RUN}/s/PT000H00M.grib2",
    # past the 06Z horizon
    "T/lvt1/150/lv1/1/r/2026-10-11T06:00/s/PT123H00M.grib2",
    # not a cycle hour, and not on the hour
    "PS/r/2026-10-11T03:00/s/PT000H00M.grib2",
    "PS/r/2026-10-11T00:30/s/PT000H00M.grib2",
    "PS/r/2026-10-11T0300/s/PT000H00M.grib2",
    "PS/r/2026-10-11T0030/s/PT000H00M.grib2",
    # the legacy flat name is a different product's grammar
    "icon_global_icosahedral_single-level_2026101100_000_PS.grib2",
])
def test_the_path_grammar_refuses_what_this_route_does_not_read(key, spec):
    with pytest.raises(ValueError):
        norm.parse_object(spec, Path("/data/icon") / key)


# ---------------------------------------------------------------------------
# The real bytes
# ---------------------------------------------------------------------------

def test_the_fixture_is_real_v1_bytes_on_the_declared_mesh(headers):
    assert len(headers) == 38
    for row in headers:
        assert (row["gdt"], row["cells"], row["centre"]) == (101, MESH_CELLS, 78)
        assert row["drt"] == 42           # CCSDS, decoded by grib-core
        assert row["reference_time"] == "2026-10-11 00:00:00"


@pytest.mark.parametrize("index", range(38))
def test_every_real_object_carries_its_declared_selector(headers, spec, mapping, index):
    from woof.mapped_source import _selector_matches_record

    row = headers[index]
    obj = norm.parse_object(spec, Path("/data/icon") / row["key"])
    seen = (row["discipline"], row["category"], row["parameter"],
            row["level_type"], row["level_value"], row["second_level_type"],
            row["second_level_value"])
    assert seen == obj.selector, row["key"]
    if obj.field in ("CLAT", "CLON"):
        return          # geometry is consumed by the remap plan, not decoded
    # The normalized record keeps these octets, so exactly one mapping field
    # reads it (HSURF is terrain_height; the remap keeps Section 4).
    record = SimpleNamespace(
        discipline=row["discipline"], category=row["category"],
        parameter=row["parameter"], level_type=row["level_type"],
        level_value=row["level_value"], second_level_type=row["second_level_type"],
        second_level_value=row["second_level_value"], center=row["centre"],
        subcenter=255, master_table_version=19, local_table_version=1,
        member=None, time_semantics=(row["pdt"],))
    matches = [name for name, field in mapping["fields"].items()
               for selector in field.get("selectors", [])
               if _selector_matches_record(selector, record, "grib2")]
    if obj.field in ("T_G", "T_2M") or obj.field == "HSURF":
        assert matches, row["key"]
    else:
        assert len(matches) == 1, (row["key"], matches)


# ---------------------------------------------------------------------------
# The download is priced from its measured row
# ---------------------------------------------------------------------------

def test_a_download_is_priced_from_its_measured_row():
    from woof import download_budget

    estimate = download_budget.download_estimate({
        "source": "icon-global-ml", "cycle": "2026-10-11T00", "hours": 3,
        "cadence": 3})
    assert estimate.get("bytes"), estimate
    # Two leads of 626 objects and one run of invariants: about 4.3 GB.
    assert 3.9e9 < estimate["bytes"] < 4.6e9


def test_a_series_at_a_whole_multiple_of_the_hour_is_accepted(spec, tmp_path):
    for cadence in (1, 3, 6):
        plan = fetch_routes.resolve_request("icon-global-ml", cycle=CYCLE,
                                            hours=2 * cadence, cadence=cadence)
        objects = norm.validate_inventory(
            spec, [Path("/x") / name for name in plan.primary_files])
        namelist = tmp_path / f"namelist-{cadence}.wps"
        namelist.write_text(
            "&share\n max_dom = 1,\n"
            f" start_date = '{CYCLE:%Y-%m-%d_%H}:00:00',\n"
            f" end_date = '{CYCLE + timedelta(hours=2 * cadence):%Y-%m-%d_%H}:00:00',\n"
            f" interval_seconds = {cadence * 3600},\n/\n")
        norm._check_wps_time_coverage(spec, namelist, objects)
