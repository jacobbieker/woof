# `woof energy` fixtures

`assets_wales.geojson` is a hand-built, synthetic `woof-energy.assets.v1`
document. It is loosely shaped like part of the South Wales transmission
network, but its coordinates, names and attributes are invented. It is not
OpenStreetMap data and must not be read as a map of real assets.

It contains:
- two 400 kV overhead lines and one 132 kV overhead line
- one 132 kV underground cable
- two substations: a point and a polygon
- a wind plant with four turbine generators carrying hub heights
- one solar generator polygon
- three towers along the 132 kV line

`sites_wales.json` is built by hand from `assets_wales.geojson`. It has a
site at each line vertex and segment midpoint, using the segment bearing,
plus the substations, turbines and the PV farm. It is not `woof energy
sites` output; it exists so units that consume sites can test without the
sites builder.

`overpass_wales_a.json`, `overpass_wales_b.json` and `overpass_timeout.json`
are hand-written Overpass API answers for `tests/test_energy_osm.py`. Their
ids, names, coordinates and tags are invented. They are not OpenStreetMap
data.
