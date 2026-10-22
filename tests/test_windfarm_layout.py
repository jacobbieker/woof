"""Turbine layouts on a domain's grid (``woof.windfarm.layout``).

Synthetic farms only: a 3 x 3 block at 500 m spacing of a 3 MW-class
library curve, on small Lambert grids built here and on the quickstart
experiment's own domain.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.cli import main as cli_main
from woof.core.hub_wind_diag import locate_sites
from woof.energy.contracts import Site, SiteSet
from woof.energy.turbine_library import load_turbine_library
from woof.static.lambert import LambertGrid
from woof.windfarm import layout as wl

REPO = Path(__file__).resolve().parents[1]
QUICKSTART = REPO / "configs" / "gfs_12km_quickstart.toml"
CURVE = "V112/3000"
LAT0, LON0 = 35.20, -97.90


def _grid(dx=1000.0, n=41, ref=(35.2, -97.9)):
    return LambertGrid(ref_lat=ref[0], ref_lon=ref[1], truelat1=30.0,
                       truelat2=60.0, stand_lon=ref[1], dx=dx, dy=dx,
                       e_we=n + 1, e_sn=n + 1)


def _farm(spacing=500.0, lat0=LAT0, lon0=LON0, **extra):
    rows = []
    for a in range(3):
        for b in range(3):
            lat = lat0 + a * spacing / 111195.0
            lon = lon0 + b * spacing / (111195.0 * math.cos(math.radians(lat0)))
            row = {"turbine_id": f"wt{a}{b}", "lat": lat, "lon": lon,
                   "hub_height_m": 94.0, "rotor_diameter_m": 112.0,
                   "curve": CURVE}
            row.update(extra)
            rows.append(row)
    return rows


def _write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _turbines(rows):
    return tuple(wl.turbine_from_row(row, n)
                 for n, row in enumerate(rows, start=1))


def test_three_by_three_farm_is_counted_per_cell():
    grid = _grid()
    layout = wl.build_layout(_turbines(_farm()), grid)
    assert (layout.ny, layout.nx) == (41, 41)
    assert layout.n_turbines == 9
    assert int(layout.turbine_count.sum()) == 9
    assert layout.cells_occupied == np.count_nonzero(layout.turbine_count)
    assert 2 <= layout.cells_occupied <= 4      # 1 km cells, 1 km farm
    assert layout.curve_ids == (CURVE,)
    assert layout.rated_capacity_mw == pytest.approx(27.0)
    occupied = layout.turbine_count > 0
    # uniform cells: one group each, carrying the one machine
    assert np.array_equal(layout.group_count, occupied.astype(np.int32))
    assert layout.n_groups == layout.cells_occupied
    assert np.all(layout.group_curve == 0)
    assert np.all(layout.group_hub_height_m == 94.0)
    assert np.all(layout.group_rotor_diameter_m == 112.0)
    assert np.array_equal(layout.group_turbines,
                          layout.turbine_count[layout.group_j, layout.group_i])
    assert np.allclose(layout.rated_capacity_kw,
                       3000.0 * layout.turbine_count)
    assert np.allclose(layout.swept_area_m2[occupied],
                       layout.turbine_count[occupied] * math.pi * 56.0 ** 2)
    assert np.all(layout.rotor_top_m[occupied] == 150.0)
    assert np.all(layout.rotor_bottom_m[occupied] == 38.0)
    assert np.all(np.isnan(layout.rotor_top_m[~occupied]))
    assert np.all(layout.swept_area_m2[~occupied] == 0.0)
    # the CSR offsets index the groups cell by cell
    assert layout.cell_group_offset.shape == (layout.ny * layout.nx + 1,)
    assert layout.cell_group_offset[-1] == layout.n_groups
    for g in range(layout.n_groups):
        assert g in layout.groups_of(layout.group_j[g], layout.group_i[g])
    assert np.array_equal(
        layout.group_j[layout.turbine_group], layout.turbine_j)
    # every turbine's recorded cell is a counted cell
    np.add.at(counted := np.zeros_like(layout.turbine_count),
              (layout.turbine_j, layout.turbine_i), 1)
    assert np.array_equal(counted, layout.turbine_count)
    summary = layout.summary()
    assert summary["turbines"] == 9
    assert sum(cell["turbines"] for cell in summary["cells"]) == 9
    assert json.loads(json.dumps(summary)) == summary


def test_arrays_are_read_only_and_the_layout_frozen():
    layout = wl.build_layout(_turbines(_farm()), _grid())
    with pytest.raises(ValueError):
        layout.turbine_count[0, 0] = 1
    with pytest.raises(AttributeError):
        layout.grid_id = 2


def test_tables_are_the_curve_as_wrf_reads_it():
    layout = wl.build_layout(_turbines(_farm()), _grid())
    curve = load_turbine_library().get(CURVE)
    t = layout.tables
    n = int(t.nval[0])
    ws = t.speeds_ms[0, :n]
    # rows: the curve's own points through cut-out, rounded once to float32
    assert ws.dtype == np.float32 and np.all(np.diff(ws) > 0)
    assert np.all(t.speeds_ms[0, n:] == 0.0)        # WRF's zero padding
    assert t.cut_in_ms[0] == ws[0] and t.cut_out_ms[0] == ws[-1]
    assert float(ws[-1]) == curve.cut_out_ms
    assert np.array_equal(t.power_kw[0, :n], (curve.fraction(
        ws.astype(np.float64)) * curve.rated_kw).astype(np.float32))
    assert np.array_equal(t.thrust_coefficient[0, :n], curve.thrust_coefficient(
        ws.astype(np.float64)).astype(np.float32))
    assert t.standing_thrust_cutout[0] == t.thrust_coefficient[0, n - 1]
    # the axis tables: WRF's dragcof and power1 on 0 .. 35 m/s
    speeds = layout.speeds_ms
    assert speeds.dtype == np.float32
    assert speeds[0] == 0.0 and speeds[-1] == wl.TABLE_MAX_SPEED_MS
    assert np.all(np.diff(speeds) == wl.TABLE_SPEED_STEP_MS)
    assert layout.group_power_w.shape == \
        layout.group_thrust_coefficient.shape == (layout.n_groups, speeds.size)
    below, above = speeds <= ws[0], speeds >= ws[-1]
    assert np.all(layout.group_thrust_coefficient[0][below] == t.standing_thrust[0])
    assert np.all(layout.group_thrust_coefficient[0][above] == t.standing_thrust_cutout[0])
    assert np.all(layout.group_power_w[0][below | above] == 0.0)
    assert 0.7 < layout.group_thrust_coefficient[0][speeds == 8.0][0] < 0.9
    # power1 = 0.5 * 1.23 * v^3 * A * powcof recovers the table's kW
    at = (speeds > ws[0]) & (speeds < ws[-1])
    kw = layout.group_power_w[0][at] / 1000.0
    assert np.allclose(kw, np.interp(speeds[at], ws, t.power_kw[0, :n]),
                       rtol=0.05, atol=5.0)


def test_cells_agree_with_the_hub_site_locator():
    grid = _grid()
    lat, lon = grid.latlon_mass()
    layout = wl.build_layout(_turbines(_farm()), grid)
    located = locate_sites(
        [(row["turbine_id"], row["lat"], row["lon"]) for row in _farm()],
        lat, lon, dx=grid.dx, dy=grid.dy)
    assert not located["outside"]
    by_id = {row[0]: (row[3], row[4]) for row in located["inside"]}
    for n, turbine_id in enumerate(layout.turbine_ids):
        assert by_id[turbine_id] == (layout.turbine_j[n], layout.turbine_i[n])


def test_csv_and_geojson_read_alike(tmp_path):
    rows = _farm()
    csv_path = _write_csv(tmp_path / "farm.csv", rows)
    features = [{"type": "Feature", "id": row["turbine_id"],
                 "geometry": {"type": "Point",
                              "coordinates": [row["lon"], row["lat"]]},
                 "properties": {"hub_height_m": row["hub_height_m"],
                                "rotor_diameter_m": row["rotor_diameter_m"],
                                "type": row["curve"]}} for row in rows]
    geo_path = tmp_path / "farm.geojson"
    geo_path.write_text(json.dumps({"type": "FeatureCollection",
                                    "features": features}))
    assert wl.read_turbines(csv_path) == wl.read_turbines(geo_path)


def test_blank_columns_take_the_default_curve_and_the_curves_rotor():
    rows = _farm(curve="", rotor_diameter_m="")
    layout = wl.build_layout(_turbines(rows), _grid(), default_curve=CURVE)
    assert np.all(layout.group_rotor_diameter_m == 112.0)


def test_a_model_name_resolves_through_the_library():
    rows = _farm(curve="V112-3.0 MW", manufacturer="Vestas")
    layout = wl.build_layout(_turbines(rows), _grid())
    assert layout.curve_ids == (CURVE,)
    library = load_turbine_library()
    assert wl.resolve_curve("V112 3MW", library,
                            manufacturer="Vestas").curve_id == CURVE
    # a bare rotor is several machines: not a curve
    assert wl.resolve_curve("V112", library) is None
    # a neighbouring rating inside the ladder's 3 % is another machine
    assert wl.resolve_curve("N131/3500", library,
                            manufacturer="Nordex") is None
    # an id whose manufacturer contradicts the row's is not that curve
    assert wl.resolve_curve(CURVE, library, manufacturer="Siemens") is None
    assert wl.resolve_curve(CURVE, library, manufacturer="Vestas") is not None


def test_model_column_wins_over_a_generic_type_column(tmp_path):
    rows = _farm(type="wind_turbine", model=CURVE)
    for row in rows:
        del row["curve"]
    layout = wl.build_layout(wl.read_turbines(
        _write_csv(tmp_path / "farm.csv", rows)), _grid())
    assert layout.curve_ids == (CURVE,)


def test_separator_only_csv_rows_are_blank(tmp_path):
    path = _write_csv(tmp_path / "farm.csv", _farm())
    path.write_text(path.read_text() + ",,,,,\n,,,,,\n")
    assert len(wl.read_turbines(path)) == 9


def test_a_stated_rating_must_be_the_curves():
    rows = _farm(rated_kw=2000.0)
    with pytest.raises(wl.WindFarmLayoutRefused,
                       match="ratings that disagree.*wt00 \\(2000 kW"):
        wl.build_layout(_turbines(rows), _grid())


def test_sites_document_needs_a_default_curve(tmp_path):
    sites = [Site(site_id=row["turbine_id"], asset_id="farm", kind="turbine",
                  lat=row["lat"], lon=row["lon"], hub_height_m=94.0,
                  capacity_mw=3.0) for row in _farm()]
    sites.append(Site(site_id="sub", asset_id="sub", kind="substation",
                      lat=LAT0, lon=LON0))
    path = tmp_path / "sites.json"
    path.write_text(json.dumps(SiteSet(sites=sites, heights_m=(94.0,))
                               .to_json()))
    turbines = wl.read_turbines(path)
    assert len(turbines) == 9
    with pytest.raises(wl.WindFarmLayoutRefused, match="no curve.*wt00"):
        wl.build_layout(turbines, _grid())
    layout = wl.build_layout(turbines, _grid(), default_curve=CURVE)
    assert layout.n_turbines == 9
    # the sites' 3 MW rating is checked against the curve
    with pytest.raises(wl.WindFarmLayoutRefused, match="ratings that disagree"):
        wl.build_layout(turbines, _grid(), default_curve="V90/2000")


def test_two_turbine_types_get_two_tables():
    rows = _farm(spacing=3000.0)
    for row in rows[:3]:
        row.update(curve="V90/3000", rotor_diameter_m=90.0, hub_height_m=80.0)
    layout = wl.build_layout(_turbines(rows), _grid())
    assert layout.curve_ids == ("V90/3000", CURVE)
    assert layout.tables.nval.size == 2
    assert layout.group_power_w.shape[0] == layout.n_groups == 9
    assert layout.cells_occupied == 9
    assert list(np.bincount(layout.turbine_curve)) == [3, 6]


# -- refusals, by name -----------------------------------------------------


def test_turbine_outside_the_domain_is_refused_by_name():
    rows = _farm()
    rows[4]["lat"] = LAT0 + 1.0                  # ~111 km north of a 41 km grid
    with pytest.raises(wl.WindFarmLayoutRefused, match="outside d01.*wt11"):
        wl.build_layout(_turbines(rows), _grid())


def test_turbine_in_the_outermost_row_is_refused_by_name():
    grid = _grid()
    lat, lon = grid.latlon_mass()
    rows = _farm()
    rows[0]["lat"], rows[0]["lon"] = float(lat[0, 20]), float(lon[0, 20])
    with pytest.raises(wl.WindFarmLayoutRefused, match="outermost.*wt00"):
        wl.build_layout(_turbines(rows), grid)


def test_missing_and_unknown_curves_are_refused_by_name():
    rows = _farm()
    rows[2]["curve"] = ""
    with pytest.raises(wl.WindFarmLayoutRefused, match="no curve: wt02"):
        wl.build_layout(_turbines(rows), _grid())
    rows[2]["curve"] = "NO-SUCH-MACHINE-77"
    with pytest.raises(wl.WindFarmLayoutRefused,
                       match="not in the turbine library: wt02"):
        wl.build_layout(_turbines(rows), _grid())


def test_rotor_that_is_another_machine_is_refused():
    rows = _farm()
    rows[1]["rotor_diameter_m"] = 150.0
    with pytest.raises(wl.WindFarmLayoutRefused, match="wt01 \\(150 m"):
        wl.build_layout(_turbines(rows), _grid())


def test_missing_or_too_low_hub_height_is_refused():
    rows = _farm()
    rows[3]["hub_height_m"] = ""
    rows[5]["hub_height_m"] = 40.0
    with pytest.raises(wl.WindFarmLayoutRefused,
                       match="hub height: wt10 \\(none\\), wt12 \\(40 m, "
                             "rotor bottom -16 m\\)"):
        wl.build_layout(_turbines(rows), _grid())
    # a rotor bottom exactly at the ground is still above 0 m AGL
    rows = _farm(hub_height_m=56.0)
    assert wl.build_layout(_turbines(rows), _grid()).n_turbines == 9


# -- mixed cells -------------------------------------------------------------


def _mixed_rows():
    """One cell, in table order: 3 x V112/3000 at 94 m, 2 x V90/3000 at
    120 m, then one more V112/3000 at 94 m (A, A, A, B, B, A)."""

    rows = _farm(spacing=50.0)[:6]
    for row in rows[3:5]:
        row.update(curve="V90/3000", rotor_diameter_m=90.0, hub_height_m=120.0)
    return rows


def test_a_mixed_cell_is_its_runs_in_table_order():
    layout = wl.build_layout(_turbines(_mixed_rows()), _grid())
    assert layout.cells_occupied == 1
    (j,), (i,) = np.nonzero(layout.turbine_count)
    assert layout.turbine_count[j, i] == 6
    assert layout.group_count[j, i] == 3
    groups = list(layout.groups_of(j, i))
    assert groups == [0, 1, 2] and layout.n_groups == 3
    # the runs, in the order WRF reads them: A x 3, B x 2, A x 1
    assert [layout.curve_ids[k] for k in layout.group_curve] == [
        CURVE, "V90/3000", CURVE]
    assert layout.group_turbines.tolist() == [3, 2, 1]
    assert layout.group_hub_height_m.tolist() == [94.0, 120.0, 94.0]
    assert layout.group_rotor_diameter_m.tolist() == [112.0, 90.0, 112.0]
    assert layout.turbine_group.tolist() == [0, 0, 0, 1, 1, 2]
    # cell totals sum the groups; the rotor span covers both machines
    assert layout.rated_capacity_kw[j, i] == pytest.approx(6 * 3000.0)
    assert layout.swept_area_m2[j, i] == pytest.approx(
        math.pi / 4.0 * (4 * 112.0 ** 2 + 2 * 90.0 ** 2), rel=1e-6)
    assert layout.rotor_top_m[j, i] == pytest.approx(165.0)
    assert layout.rotor_bottom_m[j, i] == pytest.approx(38.0)
    cell = layout.summary()["cells"][0]
    assert [g["turbines"] for g in cell["groups"]] == [3, 2, 1]
    assert cell["turbines"] == 6 and cell["capacity_mw"] == pytest.approx(18.0)
    # the kernel's names for the same arrays
    k = layout.kernel_groups()
    assert k["count"] is layout.group_turbines and k["cell"] is \
        layout.group_cell
    assert set(layout.kernel_curves()) >= {
        "speeds_ms", "thrust_coefficient", "power_kw", "nval",
        "standing_thrust"}


def test_a_b_a_in_one_cell_is_three_groups():
    rows = _farm(spacing=20.0)[:3]
    rows[1].update(curve="V90/3000", rotor_diameter_m=90.0)
    layout = wl.build_layout(_turbines(rows), _grid())
    assert layout.cells_occupied == 1
    assert layout.n_groups == 3
    assert layout.group_turbines.tolist() == [1, 1, 1]
    assert layout.turbine_group.tolist() == [0, 1, 2]


def test_half_a_metre_of_hub_height_is_two_groups_by_default():
    rows = _farm(spacing=20.0)[:2]
    rows[1]["hub_height_m"] = 94.5
    layout = wl.build_layout(_turbines(rows), _grid())
    assert layout.n_groups == 2
    assert layout.group_hub_height_m.tolist() == [94.0, 94.5]
    # the opt-in tolerance merge (not WRF-exact) makes them one
    merged = wl.build_layout(_turbines(rows), _grid(),
                             merge_tolerance_m=wl.GROUP_TOLERANCE_M)
    assert merged.n_groups == 1
    assert merged.group_hub_height_m.tolist() == [94.25]


def test_one_curve_at_three_hub_heights_is_three_runs_in_table_order():
    rows = _farm(spacing=50.0)
    rows[0]["hub_height_m"] = 120.0
    rows[1]["hub_height_m"] = 95.5
    layout = wl.build_layout(_turbines(rows), _grid())
    assert layout.cells_occupied == 1
    assert layout.n_groups == 3
    assert layout.group_turbines.tolist() == [1, 1, 7]
    assert layout.group_hub_height_m.tolist() == [120.0, 95.5, 94.0]


def test_duplicate_ids_are_refused():
    rows = _farm()
    rows[1]["turbine_id"] = "wt00"
    with pytest.raises(wl.WindFarmLayoutRefused, match="more than once: wt00"):
        wl.build_layout(_turbines(rows), _grid())


def test_rows_without_coordinates_are_refused(tmp_path):
    path = tmp_path / "farm.csv"
    path.write_text("turbine_id,hub_height_m,curve\nwt00,94,V112/3000\n")
    with pytest.raises(wl.WindFarmLayoutRefused, match="no lat column"):
        wl.read_turbines(path)
    with pytest.raises(wl.WindFarmLayoutRefused, match="lat 95.0"):
        wl.turbine_from_row({"lat": 95, "lon": 0}, 1)


# -- the quickstart domain and the CLI --------------------------------------


def test_quickstart_domain_places_the_farm_in_one_cell(tmp_path):
    path = _write_csv(tmp_path / "farm.csv", _farm())
    layout = wl.layout_for_config(QUICKSTART, path)
    assert (layout.ny, layout.nx, layout.dx_m) == (48, 60, 12000.0)
    assert layout.cells_occupied == 1
    assert int(layout.turbine_count.max()) == 9
    with pytest.raises(wl.WindFarmLayoutRefused, match="no domain d02"):
        wl.layout_for_config(QUICKSTART, path, grid_id=2)


def test_cli_prints_cells_turbines_and_capacity(tmp_path, capsys):
    path = _write_csv(tmp_path / "farm.csv", _farm())
    assert cli_main(["windfarm", "layout", str(QUICKSTART), str(path)]) == 0
    out = capsys.readouterr().out
    assert ("9 turbine(s) in 1 group(s) in 1 cell(s), at most 9 turbine(s) "
            "and 1 group(s) per cell") in out
    assert "rated capacity 27.00 MW" in out
    assert "(cell)" not in out              # a uniform cell has no total row
    assert cli_main(["windfarm", "layout", str(QUICKSTART), str(path),
                     "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema"] == wl.LAYOUT_SCHEMA
    assert summary["cells_occupied"] == 1
    assert summary["rated_capacity_mw"] == pytest.approx(27.0)


def test_cli_prints_and_writes_the_groups_of_a_mixed_cell(tmp_path, capsys):
    path = _write_csv(tmp_path / "farm.csv", _mixed_rows())
    assert cli_main(["windfarm", "layout", str(QUICKSTART), str(path)]) == 0
    out = capsys.readouterr().out
    assert "V90/3000" in out and CURVE in out
    assert "rotors 38-165 m" in out and "(cell)" in out
    assert "6 turbine(s) in 3 group(s) in 1 cell(s)" in out
    assert cli_main(["windfarm", "layout", str(QUICKSTART), str(path),
                     "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["groups"] == 3 and summary["max_groups_per_cell"] == 3
    (cell,) = summary["cells"]
    assert [(g["curve"], g["turbines"]) for g in cell["groups"]] == [
        (CURVE, 3), ("V90/3000", 2), (CURVE, 1)]
    assert cell["rotor_top_m"] == pytest.approx(165.0)


def test_cli_refusal_names_the_turbine(tmp_path, capsys):
    rows = _farm()
    rows[8]["curve"] = "NO-SUCH-MACHINE-77"
    path = _write_csv(tmp_path / "farm.csv", rows)
    assert cli_main(["windfarm", "layout", str(QUICKSTART), str(path)]) == 2
    err = capsys.readouterr().err
    assert "woof windfarm:" in err and "wt22" in err


# -- the device upload --------------------------------------------------------


def _check_upload(layout, device, asnumpy):
    assert (device.ny, device.nx, device.n_groups) == (
        layout.ny, layout.nx, layout.n_groups)
    assert device.n_curves == len(layout.curve_ids)
    assert device.speed_step_ms == wl.TABLE_SPEED_STEP_MS
    arrays = device.arrays()
    assert set(arrays) == set(wl.DEVICE_ARRAYS) | {
        f"tables.{name}" for name in layout.tables.arrays()}
    for name, array in arrays.items():
        host = (getattr(layout.tables, name.split(".", 1)[1])
                if name.startswith("tables.") else getattr(layout, name))
        assert array.flags.c_contiguous, name
        assert array.dtype == host.dtype, name
        assert array.dtype in (np.int32, np.float32), name
        back = np.ascontiguousarray(asnumpy(array))
        assert np.array_equal(back.view(np.uint32), np.ascontiguousarray(
            host).view(np.uint32)), name
    assert device.nbytes() > 0


def test_to_device_uploads_the_host_words(monkeypatch):
    monkeypatch.setattr(wl, "_cupy", lambda: np)
    layout = wl.build_layout(_turbines(_mixed_rows()), _grid())
    _check_upload(layout, layout.to_device(), np.asarray)


@pytest.mark.gpu
@requires_gpu
def test_to_device_round_trips_through_cupy():
    import cupy as cp

    rows = _mixed_rows() + _farm(spacing=3000.0, lat0=LAT0 + 0.05)
    for row in rows[6:]:
        row["turbine_id"] = "far-" + row["turbine_id"]
    layout = wl.build_layout(_turbines(rows), _grid())
    device = layout.to_device(device_id=0)
    assert isinstance(device.group_power_w, cp.ndarray)
    assert device.cell_group_offset.device.id == 0
    _check_upload(layout, device, cp.asnumpy)
    # the offsets select each cell's groups on the device as on the host
    offsets = cp.asnumpy(device.cell_group_offset)
    flat = cp.asnumpy(device.group_cell)
    for c in range(layout.ny * layout.nx):
        assert np.all(flat[offsets[c]:offsets[c + 1]] == c)


def test_the_opt_in_merge_joins_the_first_group_it_matches():
    """A (90.0 m, 112 m), B (90.2 m, 114 m: rotor 2 m off A), C (90.4 m,
    112 m: within 1 m of A) in one cell, merged: C joins A's group."""

    rows = _farm(spacing=20.0)[:3]
    rows[0].update(hub_height_m=90.0, rotor_diameter_m=112.0)
    rows[1].update(hub_height_m=90.2, rotor_diameter_m=114.0)
    rows[2].update(hub_height_m=90.4, rotor_diameter_m=112.0)
    exact = wl.build_layout(_turbines(rows), _grid())
    assert exact.n_groups == 3
    layout = wl.build_layout(_turbines(rows), _grid(), merge_tolerance_m=1.0)
    assert layout.n_groups == 2
    assert layout.turbine_group[0] == layout.turbine_group[2] != \
        layout.turbine_group[1]
    assert sorted(layout.group_turbines.tolist()) == [1, 2]


# -- other projections and nests ---------------------------------------------


def test_mercator_and_polar_domains_place_the_farm():
    from woof.static.projection import MercatorGrid, PolarStereoGrid

    merc = MercatorGrid(ref_lat=5.0, ref_lon=100.0, truelat1=5.0,
                        truelat2=5.0, stand_lon=100.0, dx=1000.0, dy=1000.0,
                        e_we=42, e_sn=42)
    layout = wl.build_layout(_turbines(_farm(lat0=5.0, lon0=100.0)), merc)
    assert layout.projection.code == 3 and layout.n_turbines == 9
    assert 2 <= layout.cells_occupied <= 4
    polar = PolarStereoGrid(ref_lat=78.0, ref_lon=15.0, truelat1=60.0,
                            truelat2=60.0, stand_lon=15.0, dx=1000.0,
                            dy=1000.0, e_we=42, e_sn=42)
    layout = wl.build_layout(_turbines(_farm(lat0=78.0, lon0=15.0)), polar)
    assert layout.projection.code == 2 and layout.n_turbines == 9
    assert int(layout.turbine_count.sum()) == 9


def test_a_nest_places_from_its_own_corner():
    root = _grid(dx=3000.0, n=41)
    nest = root.nest(14, 14, 3, 43, 43)
    layout = wl.build_layout(_turbines(_farm()), nest, grid_id=2)
    assert layout.dx_m == 1000.0 and (layout.ny, layout.nx) == (42, 42)
    lat11, lon11 = nest.ij_to_latlon(1.0, 1.0)
    assert layout.projection.lat1 == np.float32(float(lat11))
    assert layout.projection.dx == np.float32(1000.0)
    on_root = wl.build_layout(_turbines(_farm()), root)
    assert on_root.cells_occupied <= layout.cells_occupied


def test_a_projection_wrf_cannot_place_on_is_refused_by_name():
    class LatLonGrid:
        map_proj = "lat-lon"
        e_we = e_sn = 42
        dx = 0.1

    with pytest.raises(wl.WindFarmLayoutRefused,
                       match="map_proj 'lat-lon'.*Mercator, Lambert"):
        wl.build_layout(_turbines(_farm()), LatLonGrid())


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_a_bad_merge_tolerance_is_refused(bad):
    with pytest.raises(wl.WindFarmLayoutRefused, match="merge_tolerance_m"):
        wl.build_layout(_turbines(_farm()), _grid(), merge_tolerance_m=bad)
