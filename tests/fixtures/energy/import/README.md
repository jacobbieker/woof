# `woof energy import` fixtures

Every file here is synthetic. The rows, identifiers, names, operators,
capacities and coordinates were made up for tests. They are loosely placed
around South Wales, the Bristol Channel and Northern Ireland so that they
can be merged into `../assets_wales.geojson`. None of it is a copy of
PyPSA-Eur, ENTSO-E, OpenStreetMap or REPD data, and it must not be read as
a map of real assets.

Only the column layouts follow the published formats. These are the
layouts `woof/energy/importers.py` documents and reads:

- `pypsa_osm/` follows the OSM-based PyPSA-Eur prebuilt network
  (doi:10.5281/zenodo.13358976, v0.7 columns). WKT is single-quoted. It
  holds:
  - two buses at one location
  - two buses about 120 m apart joined by a transformer
  - a DC bus
  - a line with no geometry (the bus-to-bus fallback)
  - a line with malformed WKT, which is refused
  - an underground DC link
  - one substation and one line that duplicate assets in
    `assets_wales.geojson`
- `pypsa_gridkit/` follows PyPSA-Eur's `data/entsoegridkit` base network
  (hstore `tags`, `True`/`False` booleans). It holds a `joint` bus and a
  wind-farm bus, neither of which is a substation, a cable with no
  geometry, a DC link with no voltage, and generators with and without
  geometry.
- `pypsa_network/` follows `pypsa.Network.export_to_csv_folder` output
  (pandas double quotes, `v_nom`, `num_parallel`, `carrier`). It holds a
  hydrogen bus and link, which are skipped, a fractional `num_parallel`,
  and generators placed at their bus.
- `repd_sample.csv` has the real REPD extract header row and invented
  rows, cp1252 encoded. It holds:
  - one row with no coordinates
  - one row with no Ref ID
  - one Northern Ireland row on the National Grid, which is kept
  - one Northern Ireland row on the Irish Grid, which is refused
  - one `Unknown` technology
- `geojson_sample.geojson` holds OSM-style properties: a `power=pole`
  feature (refused) and one feature with no `kind`/`power`, which needs
  `--kind`.
- `csv_turbines.csv` holds turbine points for `--format csv`. One row has
  an unreadable latitude, and T1 lies within 50 m of a generator in
  `assets_wales.geojson`.
