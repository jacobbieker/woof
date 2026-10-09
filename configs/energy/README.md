# Energy forecast example: a 400 kV corridor in South Wales

`woof energy` writes its own configurations. This directory holds no TOML:
`woof energy plan` emits one experiment TOML and `namelist.wps` per domain
under the plan directory you name, and `woof energy run` runs them.

The full walk-through, with the choice of topology, resolution guidance, the
file contracts and the products, is in
[docs/energy-forecasts.md](../../docs/energy-forecasts.md).

The short version, at 100 m along the South Wales 400 kV lines:

```bash
# 1. OpenStreetMap power assets (ODbL 1.0; cached under ~/.woof/cache/energy-osm/)
woof energy fetch --bbox=-4.2,51.5,-3.3,51.8 --min-voltage-kv 132 -o wales/assets.geojson

# 2. A site every 100 m along the 400 kV lines, at 10, 30 and 50 m above ground
woof energy sites wales/assets.geojson --kinds line --min-voltage-kv 400 --spacing-m 100 --heights-m 10,30,50 -o wales/sites.json

# 3. One regional parent plus offline 100 m tiles along the lines
woof energy plan wales/sites.json --topology wrf-tiles --dx-m 100 --corridor-km 2 --hours 24 --vram-gib 24 -o wales/plan-tiles

# 4. Check the commands, then run
woof energy run wales/plan-tiles/plan.json --dry-run
woof energy run wales/plan-tiles/plan.json

# 5. Sample the sites, then rate the lines
woof energy extract wales/plan-tiles/plan.json -o wales/forecast.nc
woof energy rating wales/forecast.nc --products dlr,icing -o wales/products.nc
```

Write `--bbox=` with the equals sign. The west edge is negative, and without
the `=` the shell parser reads `-4.2` as a flag.

Swap `--topology wrf-tiles` for `wrf-nests` (one run, at most 20 nests) or
`hex-swath` (an MPAS corridor mesh) to compare topologies over the same
sites. `woof doctor` reports whether the site sampler that `extract` needs
is built.
