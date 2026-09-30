"""The probe behind ``woof/terrain_clock_map.json``, the terrain clock's map.

It measures what the map's ``what`` text says: a bell ridge across the grid
in a uniform cross-ridge wind, integrated through the production ``step()``
with the generated dynamics (the 49-level ladder the domain wizard emits,
the etac the vertical survey derives for the crest, ``epssm`` 0.5,
``smdiv`` 0.1, ``emdiv`` 0.01, ``w_damping`` 1, the slope-tapered
sixth-order filter at 0.12 and the Rayleigh lid from 5 km below the 20 km
top).  A step HELD when the run finished and the peak vertical velocity
stayed under ``4 x wind x slope + 20`` m/s, the slope being the grid-read
steepest slope the rule itself reads.  The domain is periodic, eight rows
deep and ``(wind x seconds + 10 ridge half-widths) / spacing`` columns wide
(never fewer than 96), so the flow cannot come back round to the ridge in
the time it is integrated.

Subcommands:

* ``geometry``: the grid-read slope, etac and thinnest-layer fraction of
  each row, on the CPU.  These are the row keys the map records, so this
  is how a rebuilt probe proves it builds the map's own ridges.
* ``cell``: one ridge, one wind, one step, for a stated time; prints held
  and the peak vertical velocity.  How a single map entry is re-checked.
* ``extend``: longer steps on rows whose entry is the map's longest
  measured step.  Each wind is tried from the step the weaker wind held,
  down the rung list, over the map's half hour; a wind whose entry is
  already below the map's longest step keeps it, and so does every
  stronger wind.  Every step held that way is then run three hours, and
  where one stops the entry walks down the rungs (three hours each) to
  the longest that held, or back to the map's own entry.  Writes the
  measured rows as JSON.
* ``merge``: writes an ``extend`` result into the map: the row's entries
  at the measured winds, and ``top_s_per_km``, the longest step tried at
  each wind, which is what the rule treats as "held everything tried".

Needs CuPy for ``cell`` and ``extend``; ``geometry`` and ``merge`` do not.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAP_PATH = Path(__file__).resolve().parents[1] / "woof" / "terrain_clock_map.json"

#: The map's half hour.
SECONDS = 1800.0
#: The three-hour check every extended entry passes.
CHECK_SECONDS = 10800.0
#: Rows of the periodic probe domain (the ridge is uniform along them).
ROWS = 8
#: Fewest columns of a probe domain.
MIN_COLUMNS = 96


@dataclass(frozen=True)
class Ridge:
    dx: float
    crest: float
    ridge_slope: float

    @property
    def halfwidth(self) -> float:
        """The half-width whose steepest point has ``ridge_slope``."""
        return (3.0 * math.sqrt(3.0) / 8.0) * self.crest / self.ridge_slope

    def columns(self, wind: float, seconds: float) -> int:
        """Columns the flow cannot wrap in ``seconds`` (always even, so the
        crest sits on a face exactly as on the map's domains)."""
        span = float(wind) * float(seconds) + 10.0 * self.halfwidth
        n = max(MIN_COLUMNS, int(math.ceil(span / self.dx)))
        return n + (n % 2)


def _eta():
    from woof.domain_wizard import _ETA_LEVELS

    return tuple(float(v) for v in _ETA_LEVELS)


def _config(ridge: Ridge, *, nx: int, dt: float, sound_steps: int,
            etac: float, seconds: float):
    from woof.config import RunConfig

    eta = _eta()
    return RunConfig(
        nx=int(nx), ny=ROWS, nz=len(eta) - 1, dx=ridge.dx, dy=ridge.dx,
        ztop=20000.0, dt=float(dt), run_seconds=float(seconds),
        time_step_sound=int(sound_steps), epssm=0.5, smdiv=0.1, emdiv=0.01,
        damp_opt=3, zdamp=5000.0, dampcoef=0.2, w_damping=1, terrain_opt=1,
        hill_height=ridge.crest, hill_halfwidth=ridge.halfwidth,
        hybrid_opt=2, etac=float(etac), top_lid=False, diff_6th_opt=2,
        diff_6th_factor=0.12, diff_6th_slopeopt=1, h_sca_adv_order=5,
        eta_levels=eta)


def _sounding(z):
    from woof.core import constants as c

    return 290.0 * np.exp(1.0e-4 * np.asarray(z, dtype=np.float64) / c.G)


def _base(cfg, etac, terrain):
    from woof.core.grid import make_base_state, make_vertical_coord

    coord = make_vertical_coord(cfg.nz, hybrid_opt=2, etac=float(etac),
                                eta_levels=np.asarray(cfg.eta_levels))
    base = make_base_state(coord, _sounding, p_surf=cfg.p_surf,
                           ztop=cfg.ztop, terrain_z=terrain)
    return coord, base


def geometry(ridge: Ridge, *, nx: int | None = None) -> dict:
    """The row keys of ``ridge``: grid-read slope, etac, thinnest layer."""
    from woof.acoustic_adaptation import steepest_slope
    from woof.core.terrain import bell_hill
    from woof.vertical_adaptation import (TerrainField,
                                           survey_vertical_coordinate)

    nx = ridge.columns(0.0, 0.0) if nx is None else int(nx)
    cfg = _config(ridge, nx=nx, dt=1.0, sound_steps=4, etac=0.2,
                  seconds=1.0)
    terrain = bell_hill(cfg)
    coord, _base_state = _base(cfg, 0.2, terrain)
    survey = survey_vertical_coordinate(
        np.asarray(cfg.eta_levels, dtype=np.float64), 2, 0.2,
        float(coord.p_top), [TerrainField("ridge", terrain)])
    etac = 0.2 if survey is None or survey.etac is None else survey.etac
    fraction = None if survey is None else survey.layer_fraction
    slope = steepest_slope(terrain, cfg.dx, cfg.dy, label="ridge").slope
    return {"slope": round(float(slope), 4), "etac": round(float(etac), 3),
            "etac_exact": float(etac),
            "thinnest_layer_fraction": (None if fraction is None
                                        else round(float(fraction), 4))}


def run_cell(ridge: Ridge, *, wind: float, per_km: float, sound_steps: int,
             seconds: float, etac: float | None = None) -> dict:
    """One ridge, one wind, one step through the production ``step()``."""
    import cupy as cp

    from woof.acoustic_adaptation import steepest_slope
    from woof.core.dycore import set_w_surface, step
    from woof.core.state import init_at_rest
    from woof.core.terrain import bell_hill

    if etac is None:
        etac = geometry(ridge)["etac_exact"]
    dt = float(per_km) * ridge.dx / 1000.0
    nx = ridge.columns(wind, seconds)
    cfg = _config(ridge, nx=nx, dt=dt, sound_steps=sound_steps, etac=etac,
                  seconds=seconds)
    terrain = bell_hill(cfg)
    slope = steepest_slope(terrain, cfg.dx, cfg.dy, label="ridge").slope
    bound = 4.0 * float(wind) * float(slope) + 20.0
    coord, base = _base(cfg, etac, terrain)
    state = init_at_rest(cfg, coord, base, terrain_z=base.terrain_z)
    state.u[...] = cp.float32(wind)
    set_w_surface(state, cfg)
    state.w[1:] = state.w[0][None] * (state.znw[1:, None, None] ** 2)
    steps = int(round(float(seconds) / dt))
    peak = 0.0
    held = True
    started = time.perf_counter()
    stopped_at = None
    for index in range(steps):
        step(state, cfg)
        value = float(cp.abs(state.w).max())
        if not math.isfinite(value) or value > bound:
            held = False
            peak = value if math.isfinite(value) else float("inf")
            stopped_at = (index + 1) * dt
            break
        peak = max(peak, value)
    del state
    cp.get_default_memory_pool().free_all_blocks()
    return {"held": held, "peak_w": peak, "bound": bound, "dt": dt,
            "per_km": float(per_km), "wind": float(wind), "nx": nx,
            "steps": steps, "stopped_at_s": stopped_at,
            "wall_s": round(time.perf_counter() - started, 2)}


def _map_row(document, ridge: Ridge, sound_steps: int):
    for row in document["rows"]:
        if (float(row["dx_m"]) == ridge.dx
                and float(row["crest_m"]) == ridge.crest
                and abs(float(row["ridge_slope"]) - ridge.ridge_slope) < 1e-9
                and int(row["sound_steps"]) == int(sound_steps)):
            return row
    raise SystemExit(f"no map row for {ridge} on {sound_steps} substeps")


def extend_row(ridge: Ridge, sound_steps: int, winds, rungs, document,
               log=print) -> dict:
    """Longer steps on one map row, at ``winds``, from ``rungs`` (s/km,
    longest first); the map's own entries stand where they are below its
    longest measured step."""
    row = _map_row(document, ridge, sound_steps)
    top = float(document["ladder_s_per_km"][0])
    map_winds = [float(w) for w in document["winds_m_s"]]
    shape = geometry(ridge)
    old = dict(zip(map_winds, row["stable_s_per_km"]))
    runs = []
    half_hour = {}
    start = 0
    for wind in winds:
        entry = old[float(wind)]
        if start is None or entry is None or float(entry) < top:
            half_hour[wind] = entry
            start = None
            continue
        held_index = None
        for index in range(start, len(rungs)):
            result = run_cell(ridge, wind=wind, per_km=rungs[index],
                              sound_steps=sound_steps, seconds=SECONDS,
                              etac=shape["etac_exact"])
            runs.append({"check": "half hour", **result})
            log(f"  {ridge.crest:.0f} m slope {ridge.ridge_slope:g} x{sound_steps} "
                f"{wind:g} m/s {rungs[index]:.3f} s/km: "
                f"{'held' if result['held'] else 'stopped'} "
                f"(peak w {result['peak_w']:.1f}, bound {result['bound']:.1f}, "
                f"{result['wall_s']} s)")
            if result["held"]:
                held_index = index
                break
        if held_index is None:
            half_hour[wind] = entry
            start = None
        else:
            half_hour[wind] = rungs[held_index]
            start = held_index
    final = {}
    ceiling = None
    for wind in winds:
        value = half_hour[wind]
        if value is None or float(value) <= top:
            final[wind] = value
            ceiling = top if value is not None else ceiling
            continue
        candidates = [r for r in rungs if r <= float(value) + 1e-9
                      and (ceiling is None or r <= ceiling + 1e-9)]
        chosen = None
        for rung in candidates:
            result = run_cell(ridge, wind=wind, per_km=rung,
                              sound_steps=sound_steps, seconds=CHECK_SECONDS,
                              etac=shape["etac_exact"])
            runs.append({"check": "three hours", **result})
            log(f"  3 h {ridge.crest:.0f} m slope {ridge.ridge_slope:g} "
                f"x{sound_steps} {wind:g} m/s {rung:.3f} s/km: "
                f"{'held' if result['held'] else 'stopped'} "
                f"(peak w {result['peak_w']:.1f}, {result['wall_s']} s)")
            if result["held"]:
                chosen = rung
                break
        final[wind] = chosen if chosen is not None else old[float(wind)]
        ceiling = float(final[wind])
    stable = []
    tops = []
    measured = {float(w) for w in winds}
    for wind in map_winds:
        if wind in measured:
            stable.append(final[wind])
            tops.append(float(rungs[0]))
        else:
            stable.append(old[wind])
            tops.append(top)
    return {"dx_m": ridge.dx, "crest_m": ridge.crest,
            "ridge_slope": ridge.ridge_slope, "sound_steps": int(sound_steps),
            "slope": shape["slope"], "etac": shape["etac"],
            "thinnest_layer_fraction": shape["thinnest_layer_fraction"],
            "map_slope": row["slope"], "map_etac": row["etac"],
            "map_thinnest_layer_fraction": row["thinnest_layer_fraction"],
            "stable_s_per_km": stable, "top_s_per_km": tops,
            "half_hour": [half_hour.get(float(w)) for w in map_winds
                          if float(w) in measured],
            "runs": runs}


def merge(document: dict, rows, *, what: str | None = None) -> dict:
    """Write measured rows into the map document (in place, returned)."""
    for measured in rows:
        ridge = Ridge(float(measured["dx_m"]), float(measured["crest_m"]),
                      float(measured["ridge_slope"]))
        row = _map_row(document, ridge, int(measured["sound_steps"]))
        row["stable_s_per_km"] = list(measured["stable_s_per_km"])
        row["top_s_per_km"] = list(measured["top_s_per_km"])
    if what is not None:
        document["what"] = what
    return document


def write_map(document: dict, path: Path = MAP_PATH) -> None:
    """The map's own layout: one row per line."""
    head = {key: value for key, value in document.items() if key != "rows"}
    lines = ["{"]
    for key, value in head.items():
        lines.append(f" {json.dumps(key)}: "
                     f"{json.dumps(value, indent=1).replace(chr(10), chr(10) + ' ')},")
    lines.append(' "rows": [')
    rows = [" " + " " + json.dumps(row) for row in document["rows"]]
    lines.append(",\n".join(rows))
    lines.append(" ]")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _floats(text):
    return [float(v) for v in str(text).split(",") if v.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("geometry")
    g.add_argument("--dx", type=float, required=True)
    g.add_argument("--crests", type=_floats, required=True)
    g.add_argument("--ridge-slopes", type=_floats, required=True)
    c = sub.add_parser("cell")
    c.add_argument("--dx", type=float, required=True)
    c.add_argument("--crest", type=float, required=True)
    c.add_argument("--ridge-slope", type=float, required=True)
    c.add_argument("--wind", type=float, required=True)
    c.add_argument("--per-km", type=float, required=True)
    c.add_argument("--sound-steps", type=int, required=True)
    c.add_argument("--seconds", type=float, default=SECONDS)
    e = sub.add_parser("extend")
    e.add_argument("--dx", type=float, required=True)
    e.add_argument("--crests", type=_floats, required=True)
    e.add_argument("--ridge-slopes", type=_floats, required=True)
    e.add_argument("--winds", type=_floats, required=True)
    e.add_argument("--sound-steps", type=_floats, default=[4, 6])
    e.add_argument("--rung-seconds", type=_floats, required=True,
                   help="steps to try, in seconds, longest first")
    e.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("merge")
    m.add_argument("measured", type=Path, nargs="+")
    m.add_argument("--what", type=Path, default=None,
                   help="a text file holding the map's new 'what' text")
    args = parser.parse_args(argv)

    if args.command == "geometry":
        for crest in args.crests:
            for slope in args.ridge_slopes:
                print(json.dumps({"dx_m": args.dx, "crest_m": crest,
                                  "ridge_slope": slope,
                                  **geometry(Ridge(args.dx, crest, slope))}))
        return 0
    if args.command == "cell":
        print(json.dumps(run_cell(
            Ridge(args.dx, args.crest, args.ridge_slope), wind=args.wind,
            per_km=args.per_km, sound_steps=args.sound_steps,
            seconds=args.seconds)))
        return 0
    if args.command == "extend":
        document = json.loads(MAP_PATH.read_text(encoding="utf-8"))
        rungs = [s * 1000.0 / args.dx for s in args.rung_seconds]
        rows = []
        for crest in args.crests:
            for slope in args.ridge_slopes:
                for count in args.sound_steps:
                    rows.append(extend_row(
                        Ridge(args.dx, crest, slope), int(count), args.winds,
                        rungs, document,
                        log=lambda text: print(text, flush=True)))
                    args.out.write_text(json.dumps(
                        {"dx_m": args.dx, "winds_m_s": args.winds,
                         "rung_seconds": args.rung_seconds,
                         "rungs_s_per_km": rungs, "seconds": SECONDS,
                         "check_seconds": CHECK_SECONDS, "rows": rows},
                        indent=1), encoding="utf-8")
        return 0
    if args.command == "merge":
        document = json.loads(MAP_PATH.read_text(encoding="utf-8"))
        rows = []
        for path in args.measured:
            rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
        what = (None if args.what is None
                else args.what.read_text(encoding="utf-8").strip())
        write_map(merge(document, rows, what=what))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
