# Choosing the boundary source by place and date

A year-long hindcast over every public wind farm cannot name one forcing
source.  HRRR reaches only the CONUS; the Met Office global archive keeps a
rolling two years and its runs before 2025-12-11 12Z cannot be prepared (no
orography); ECMWF's open data can be prepared only from 2024-03-18 12Z; GFS
on the current S3 layout starts 2021-03-22 12Z; ERA5 reaches every date
since 1940, at 31 km.  `woof.energy.source_select.select_source` picks the
best source for one region and one cycle, and records why every better one
was passed over.

```python
from woof.energy.source_select import select_source

choice = select_source(region, "2026-01-15T00", 24, p_top_pa=5000.0,
                       now=datetime(2026, 10, 14, 12))
choice.source          # "met-office-global"
choice.rejections      # {"hrrr": ["coverage: ..."], ...}
choice.to_json()       # the woof-energy.source-choice.v1 receipt
```

Signature (stable; the hindcast driver calls it):

```python
select_source(region, cycle, hours, p_top_pa=None, priority=None, *,
              now=None, spinup_hours=0.0, margin_km=0.0,
              accept_unknown_archive=False, refuse=None) -> SourceChoice
```

* `region`: a GeoJSON geometry, Feature or FeatureCollection, a
  `{"bbox": [W, S, E, N]}` mapping, a `(W, S, E, N)` tuple, or a list of
  `(lon, lat)` points.  Its bounding box is what is checked; a region
  crossing the antimeridian is refused.
* `cycle`: the source cycle the run starts from (UTC, whole hour).
* `hours`: the run after spin-up; `spinup_hours` is added for every
  horizon and publication question.
* `p_top_pa`: the run's model top; `None` skips the column-top check.
* `priority`: source ids or aliases, best first.
* `now`: the clock every date question is asked against (default the wall
  clock; pass it for a reproducible answer).
* `margin_km`: widens the region's box before the coverage check (a
  planner's outer domain reaches past its sites).
* `refuse(source_id)`: the caller's own refusal for a candidate.

No candidate qualifying raises `SourceSelectRefused` (a `ValueError`); its
`choice` attribute holds the full receipt.  Nothing here reaches a server:
every check reads the source registry, the fetch route table and the
calendar's no-network verdict.

## Default priority

`hrrr-native,hrrr,rap,met-office-global,ecmwf-open-data,gfs,era5`

Regional sources come first and are ranked out wherever their grid does
not reach.  `hrrr-native` declares no archive start in the registry, so for
a cycle older than NOMADS' two days its date check answers "unknown" and
the candidate is rejected; `accept_unknown_archive=True` lets the download
decide instead.  RAP's runs reach 51 h only at 03/09/15/21Z (21 h
otherwise), so a 24 h run from 00Z passes it over.

## The checks

Each candidate is put to every check, in this order; a receipt lists every
failure, not only the first.

| Check | Rejected when |
| --- | --- |
| `registry` | the name is not registered, or the row is not runnable |
| `acquisition` | the fetch route table has no acquisition route, or the launch chain needs bytes already on disk |
| `planner` | the calling planner refuses it (`wrf-nests` cannot emit `hrrr`) |
| `coverage` | a sample of the region's box (widened by `margin_km`, 0.25 degree grid) lies outside the row's native-grid window |
| `horizon` | the cycle's run (route lead ladder, else cycle-grid horizons, else `max_forecast_hour`) ends before `hours + spinup_hours`; a reanalysis has no horizon |
| `column_top` | the model top is above the row's certified top and its fetch cannot extend to it; a row with no certified top passes and the preparation's vertical-coverage check is the gate |
| `date` | the calendar's verdict is not yes: before the archive's first cycle, past a rolling door's retention, a publication era with a `prep_refusal`, a cycle hour the source does not run for that length, or a run not yet published by `now` |
| `window` | the planners' own window refusal (ICON global, Met Office global), asked only when `date` passed |

## The receipt: `woof-energy.source-choice.v1`

```json
{
 "schema": "woof-energy.source-choice.v1",
 "source": "ecmwf-open-data",
 "cycle": "2025-11-01T00",
 "hours": 30.0, "spinup_hours": 0.0, "horizon_hours": 30.0,
 "p_top_pa": 5000.0,
 "priority": ["hrrr-native", "hrrr", "rap", "met-office-global",
              "ecmwf-open-data", "gfs", "era5"],
 "priority_origin": "default",
 "region": {"bbox": [W, S, E, N], "margin_km": 0.0,
            "sampled_bbox": [W, S, E, N], "samples": 35},
 "now": "2026-10-14T12:00:00Z",
 "accept_unknown_archive": false,
 "checks": ["registry", "acquisition", "planner", "coverage", "horizon",
            "column_top", "date", "window"],
 "candidates": [
  {"rank": 4, "requested": "met-office-global",
   "source": "met-office-global", "status": "rejected",
   "rejections": [{"check": "date", "reason": "... no orography ..."}],
   "notes": []},
  {"rank": 5, "requested": "ecmwf-open-data", "source": "ecmwf-open-data",
   "status": "chosen", "rejections": [], "notes": ["..."]}
 ],
 "route_table_sha256": "...",
 "notes": ["..."]
}
```

`status` is `chosen`, `eligible` (qualified but ranked below the choice) or
`rejected`.  `route_table_sha256` pins the fetch route table the dates and
eras were read from.

## In the planners: `--source auto`

`woof energy plan --topology wrf-tiles` and `--topology wrf-nests` take
`--source auto` and, with it, `--source-priority a,b,c`:

```bash
woof energy plan sites.json --topology wrf-tiles --source auto --start 2026-01-15T00 --hours 24 --vram-gib 80 -o plan-tiles
woof energy plan sites.json --topology wrf-nests --source auto --source-priority met-office-global,ecmwf-open-data,gfs --start 2025-11-01T00 --hours 24 -o plan-nests
```

The planner asks for the 5000 Pa (about 20 km) model top `woof domain`
emits by default and widens the sites' box by how far its outermost domain
can reach past them (wrf-tiles: the widest parent buffer over the
refinement chains it may try; wrf-nests: corridor, nest and root buffers
and clearance rows), plus 10 km for the wizard's rounding.  The choice is

* the plan's `source`, as if it had been named;
* the receipt, in `plan.json` under `source_choice` and beside it as
  `source-choice.json` (a re-plan with a named source removes it);
* a plan note naming the choice and each better candidate's first failed
  check;
* `source_choice` (the choice, priority and every rejection) in the JSON
  record the command prints.

`--source-priority` without `--source auto` is refused, as is either flag
with `--topology hex-swath`.  With `--source auto` and no `--start`, the
choice is made for the newest 00/06/12/18Z cycle by the clock, which a
slow publisher may not have posted yet; pass `--start` for a reproducible
plan.
