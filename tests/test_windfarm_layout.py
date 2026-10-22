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
    assert np.all(layout.curve_index[occupied] == 0)
    assert np.all(layout.curve_index[~occupied] == -1)
    assert np.all(layout.hub_height_m[occupied] == 94.0)
    assert np.all(np.isnan(layout.rotor_diameter_m[~occupied]))
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


def test_tables_are_the_curve_on_the_shared_axis():
    layout = wl.build_layout(_turbines(_farm()), _grid())
    curve = load_turbine_library().get(CURVE)
    speeds = layout.speeds_ms
    assert speeds[0] == 0.0 and speeds[-1] == wl.TABLE_MAX_SPEED_MS
    assert np.allclose(np.diff(speeds), wl.TABLE_SPEED_STEP_MS)
    assert layout.power_kw.shape == layout.thrust_coefficient.shape == (
        1, speeds.size)
    assert np.array_equal(layout.power_kw[0],
                          curve.fraction(speeds) * curve.rated_kw)
    assert np.array_equal(layout.thrust_coefficient[0],
                          curve.thrust_coefficient(speeds))
    assert layout.power_kw[0].max() == pytest.approx(3000.0)
    assert 0.7 < layout.thrust_coefficient[0][speeds == 8.0][0] < 0.9
    assert layout.cut_out_ms[0] == curve.cut_out_ms


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
    occupied = layout.turbine_count > 0
    assert np.all(layout.rotor_diameter_m[occupied] == 112.0)


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
    assert layout.power_kw.shape[0] == 2
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
                       match="hub height: wt10 \\(none\\), wt12 \\(40 m\\)"):
        wl.build_layout(_turbines(rows), _grid())


def test_a_cell_with_two_machines_is_refused():
    rows = _farm(spacing=50.0)
    rows[0]["hub_height_m"] = 120.0
    with pytest.raises(wl.WindFarmLayoutRefused,
                       match="more than one machine.*wt00.*hub spread 26 m"):
        wl.build_layout(_turbines(rows), _grid())


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
    assert "9 turbine(s) in 1 cell(s), at most 9 per cell" in out
    assert "rated capacity 27.00 MW" in out
    assert cli_main(["windfarm", "layout", str(QUICKSTART), str(path),
                     "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema"] == wl.LAYOUT_SCHEMA
    assert summary["cells_occupied"] == 1
    assert summary["rated_capacity_mw"] == pytest.approx(27.0)


def test_cli_refusal_names_the_turbine(tmp_path, capsys):
    rows = _farm()
    rows[8]["curve"] = "NO-SUCH-MACHINE-77"
    path = _write_csv(tmp_path / "farm.csv", rows)
    assert cli_main(["windfarm", "layout", str(QUICKSTART), str(path)]) == 2
    err = capsys.readouterr().err
    assert "woof windfarm:" in err and "wt22" in err
