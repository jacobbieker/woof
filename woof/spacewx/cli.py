"""``woof spacewx``: the standalone space-weather doors.

``woof spacewx ionosphere`` evaluates the L3a climatological ionosphere
(:mod:`woof.spacewx.ionosphere`) on a global latitude/longitude grid on the
GPU and writes foF2, hmF2, NmF2, foE, foF1, hmF1, M(3000)F2, the F2
thicknesses and vertical TEC to an ``.npz`` with a JSON receipt.

``woof spacewx aurora`` evaluates the L3b auroral oval (:mod:`woof.spacewx.aurora`)
on the same kind of grid: geomagnetic latitude, MLT, the auroral zone and the
distance from the diffuse aurora's equatorward boundary.  Its activity comes
from a fixed Kp, the observed Kp (GFZ), or the solar wind (OMNI2 or SWPC real
time) through the fitted Kp estimate; with OMNI2, ``--observed-al`` uses the
observed AL instead.  With a solar wind, ``--energy`` adds the precipitating
energy and number fluxes and the hemispheric power of OVATION Prime 2010
(:mod:`woof.spacewx.ovation`), and with OMNI2 the Knipp et al. 2004 Joule
power.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import numpy as np


def _utc(text: str) -> datetime:
    try:
        value = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(f"woof spacewx: --time {text!r} is not an ISO time (2024-05-10T18:00Z)") from exc
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def ionosphere_main(args) -> int:
    from woof.spacewx import indices
    from woof.spacewx.ionosphere import Ionosphere, IonosphereError, OUTPUTS, grid, receipt

    start = _utc(args.time)
    times = [start + timedelta(hours=i * args.step_hours) for i in range(max(1, args.count))]
    if args.storm and args.f107 is not None:
        raise SystemExit("woof spacewx ionosphere: --storm reads ap from the index record; "
                         "drop --f107 (use --indices)")
    if args.storm_fit and not args.storm:
        raise SystemExit("woof spacewx ionosphere: --storm-fit chooses the fit of --storm; add --storm")
    if args.f107 is not None:
        f107_source, record = f"fixed {args.f107}", None
    else:
        # The storm driver reads ap through the last time; F10.7 through the day before.
        record = indices.load(args.indices, needed=times[-1] - timedelta(days=0 if args.storm else 1))
        f107_source = f"{record.source} sha256:{record.sha256}"
    lat, lon = grid(args.dlat, args.dlon)
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    try:
        iono = Ionosphere(lat2, lon2, when=start, family=args.maps,
                          alt_lo=args.alt_min, alt_hi=args.alt_max, nalt=args.cells)
    except IonosphereError as exc:
        raise SystemExit(f"woof spacewx ionosphere: {exc}") from exc
    fit = mlat = None
    if args.storm:
        from woof.spacewx import storm
        try:
            fit = storm.load_fit(args.storm_fit)
        except (storm.StormError, OSError) as exc:
            raise SystemExit(f"woof spacewx ionosphere: storm-time fit: {exc}") from exc
        mlat = storm.magnetic_latitude(lat2.ravel(), lon2.ravel(), start)
    import cupy as cp
    fields = {k: [] for k in OUTPUTS + (("storm_factor",) if fit else ())}
    drivers = []
    for when in times:
        f107 = float(args.f107) if record is None else record.f107_driver(when, args.f107_mode)
        row = {"time": when.isoformat() + "Z", "f107": f107}
        scale = None
        if fit is not None:
            try:
                row["ap_tau"] = storm.filtered_ap(record, when, fit.tau, fit.nlag)
            except storm.StormError as exc:
                raise SystemExit(f"woof spacewx ionosphere: {exc}") from exc
            scale = storm.fof2_scale(fit, lat2, lon2, when, row["ap_tau"], mlat_deg=mlat)
            fields["storm_factor"].append((scale * scale).reshape(lat2.shape).astype(np.float32))
        out = iono.evaluate(when, f107, fo_scale=scale)
        for key in OUTPUTS:
            fields[key].append(cp.asnumpy(out[key]).reshape(lat2.shape))
        drivers.append(row)
        tec = fields["tec"][-1]
        weights = np.cos(np.deg2rad(lat2))
        storm_text = f"  ap_tau {row['ap_tau']:5.1f}" if fit is not None else ""
        print(f"{when:%Y-%m-%dT%H:%MZ}  F10.7 {f107:6.1f}{storm_text}  TEC mean {np.average(tec, weights=weights):6.2f} "
              f"max {tec.max():6.2f} TECU  foF2 max {fields['fof2'][-1].max():5.2f} MHz")
    doc = receipt(iono, f107_source, "fixed" if record is None else args.f107_mode)
    if fit is not None:
        doc.update(storm.receipt(fit, None, f107_source))
        doc.pop("spacewx_storm_ap_tau")                  # one per time, in "times"
    doc["times"] = drivers
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, lat=lat, lon=lon, **{k: np.stack(v) for k, v in fields.items()})
        args.out.with_suffix(".json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.out} and {args.out.with_suffix('.json')}")
    return 0


def _aurora_activity(args, times):
    """Per time: Kp, |AL|, and the solar-wind couplings when a solar wind is read."""
    from woof.spacewx import aurora, indices, solarwind

    if args.observed_al and not args.solar_wind:
        raise SystemExit("woof spacewx aurora: --observed-al needs --solar-wind (OMNI2 carries AL)")
    if args.observed_al and str(args.solar_wind).strip() == "swpc":
        raise SystemExit("woof spacewx aurora: --observed-al needs OMNI2 ('omni' or files); "
                         "the SWPC real-time feed carries no AL")
    if args.ovation and not args.energy:
        raise SystemExit("woof spacewx aurora: --ovation chooses the coefficients of --energy; add --energy")
    if args.energy and not args.solar_wind:
        raise SystemExit("woof spacewx aurora: --energy needs --solar-wind (OVATION Prime is driven "
                         "by the solar-wind coupling)")
    rows, record = [], None
    try:
        if args.solar_wind:
            # The Kp fit reads the hour before the 3-hour interval, up to 4 h before the first time.
            record = solarwind.load(args.solar_wind, times[0] - timedelta(hours=4),
                                    times[-1] + timedelta(hours=2))
            source = f"{record.source} sha256:{record.sha256}"
        elif args.kp is None:
            geo = indices.load(args.indices, needed=times[-1], kp=True)
            source = f"{geo.source} sha256:{geo.sha256}"
        else:
            source = f"fixed Kp {args.kp}"
    except (solarwind.SolarWindError, indices.IndicesError, OSError) as exc:
        raise SystemExit(f"woof spacewx aurora: {exc}") from exc
    for when in times:
        row = {"time": when.isoformat() + "Z"}
        try:
            if record is not None:
                hour = record.hourly(when)
                try:
                    row["kp"] = record.kp_estimate(when)
                except solarwind.SolarWindError:
                    if not args.observed_al:
                        raise
                    row["kp"] = float("nan")       # only the observed AL drives the oval
                row["newell"] = float(solarwind.newell_coupling(hour["speed"], hour["by"], hour["bz"]))
                b = float(np.hypot(hour["by"], hour["bz"]))
                row["polar_cap_kv"] = float(solarwind.boyle_potential(hour["speed"], b, hour["bz"]))
                if args.observed_al:
                    al = record.al_at(when)
                    if not np.isfinite(al):
                        raise solarwind.SolarWindError(f"{when}: OMNI2 has no AL")
                    row["al"] = abs(al)
            elif args.kp is None:
                row["kp"] = geo.kp_at(when)
            else:
                row["kp"] = float(args.kp)
            row.setdefault("al", aurora.kp_to_al(row["kp"]))
        except (solarwind.SolarWindError, indices.IndicesError, aurora.AuroraError) as exc:
            raise SystemExit(f"woof spacewx aurora: {exc}") from exc
        rows.append(row)
    return rows, source, record


@dataclass
class _Energy:
    """One time's OVATION grids (None without a coupling) and its scalars."""
    grids: dict | None
    scalars: dict


def _energy_coefficients(args):
    from woof.spacewx import ovation
    try:
        return ovation.load(args.ovation)
    except (ovation.OvationError, OSError, ValueError) as exc:
        raise SystemExit(f"woof spacewx aurora: OVATION Prime coefficients: {exc}") from exc


def _energy(coef, record, when) -> _Energy:
    """OVATION's grids in both hemispheres and the row of scalars (dPhi/dt,
    hemispheric power, Joule power) at ``when``.  A time without three good
    solar-wind hours gets no grids and NaN scalars, as the oval's gaps do."""
    from woof.spacewx import ovation, solarwind

    nan = float("nan")
    scalars = {"ovation_dphidt": nan, "hp_north_gw": nan, "hp_south_gw": nan,
               "hp_ion_north_gw": nan, "hp_ion_south_gw": nan}
    try:
        dfdt = ovation.coupling(record, when)
    except ovation.OvationError:
        grids = None
    else:
        grids = ovation.hemisphere_grids(coef, dfdt, when)
        scalars["ovation_dphidt"] = dfdt
        scalars.update(ovation.power_scalars(grids))
    try:
        pcn, dst = record.pcn_at(when), record.dst_at(when)
    except solarwind.SolarWindError:
        pcn = dst = nan
    scalars["pcn"], scalars["dst"] = pcn, dst
    scalars["joule_north_gw"] = (ovation.joule_power(pcn, dst, when)
                                 if np.isfinite(pcn) and np.isfinite(dst) else nan)
    return _Energy(grids, scalars)


def aurora_main(args) -> int:
    from woof.spacewx.aurora import Aurora, AuroraError, OUTPUTS, receipt
    from woof.spacewx.ionosphere import grid
    from woof.spacewx.ovation import COLUMN_FIELDS, energy_columns, is_pinned

    start = _utc(args.time)
    times = [start + timedelta(hours=i * args.step_hours) for i in range(max(1, args.count))]
    rows, source, record = _aurora_activity(args, times)
    coef = _energy_coefficients(args) if args.energy else None
    lat, lon = grid(args.dlat, args.dlon)
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    try:
        oval = Aurora(lat2, lon2, when=start)
    except AuroraError as exc:
        raise SystemExit(f"woof spacewx aurora: {exc}") from exc
    import cupy as cp
    fields = {k: [] for k in OUTPUTS + (COLUMN_FIELDS if coef is not None else ())}
    north = lat2 > 0
    for when, row in zip(times, rows):
        out = oval.evaluate(when, row["al"])
        for key in OUTPUTS:
            fields[key].append(cp.asnumpy(out[key]).reshape(lat2.shape))
        if coef is not None:
            energy = _energy(coef, record, when)
            row.update(energy.scalars)
            if energy.grids is None:           # no coupling at this time: fill
                cols = {key: np.full(lat2.shape, np.nan) for key in COLUMN_FIELDS}
            else:
                cols = energy_columns(energy.grids, fields["mlat"][-1].astype(np.float64),
                                      fields["mlt"][-1].astype(np.float64))
            for key in COLUMN_FIELDS:
                fields[key].append(cols[key].astype(np.float32))
        zone = fields["zone"][-1]
        edge = lat2[north & (zone >= 1)]
        extra = (f"  Newell {row['newell']:8.0f}  PCP {row['polar_cap_kv']:5.1f} kV"
                 if "newell" in row and np.isfinite(row["newell"]) else "")
        if coef is not None and np.isfinite(row["ovation_dphidt"]):
            extra += (f"  dPhi/dt {row['ovation_dphidt']:6.0f}  HP N {row['hp_north_gw']:5.1f} "
                      f"S {row['hp_south_gw']:5.1f} GW")
            if np.isfinite(row["joule_north_gw"]):
                extra += f"  Joule N {row['joule_north_gw']:6.1f} GW"
        print(f"{when:%Y-%m-%dT%H:%MZ}  Kp {row['kp']:4.2f}  |AL| {row['al']:6.1f} nT  "
              f"aurora reaches {edge.min() if edge.size else float('nan'):5.1f} N{extra}")
    doc = receipt(oval, {"spacewx_activity_source": source})
    if coef is not None:
        doc["spacewx_energy_model"] = "OVATION Prime 2010 (ovation-prime 2.3.0, NOAA)"
        doc["spacewx_ovation_source"] = f"{coef.source} sha256:{coef.sha256}"
        # Whether the tables are the pinned distribution's, not a local copy.
        doc["spacewx_ovation_pinned"] = is_pinned(coef)
        doc["spacewx_joule_model"] = "Knipp et al. 2004 Table I, northern hemisphere, |PC(N)| and |Dst|"
    # Strict JSON: a coupling over a fill-value hour is null, not NaN.
    doc["times"] = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                     for k, v in row.items()} for row in rows]
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, lat=lat, lon=lon, **{k: np.stack(v) for k, v in fields.items()})
        args.out.with_suffix(".json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.out} and {args.out.with_suffix('.json')}")
    return 0


def register_cli(sub):
    parser = sub.add_parser("spacewx", help="space weather: the opt-in ionosphere and solar layers")
    commands = parser.add_subparsers(dest="spacewx_command", required=True)
    iono = commands.add_parser(
        "ionosphere", help="climatological ionosphere (foF2, hmF2, TEC) on a global grid, on the GPU")
    iono.add_argument("--time", required=True, help="first valid time, ISO UTC (2024-05-10T18:00Z)")
    iono.add_argument("--count", type=int, default=1, help="number of times (default 1)")
    iono.add_argument("--step-hours", type=float, default=1.0, help="hours between times (default 1)")
    source = iono.add_mutually_exclusive_group()
    source.add_argument("--f107", type=float, help="a fixed F10.7 (sfu) instead of the index record")
    source.add_argument("--indices", default="gfz",
                        help="'gfz' (fetched, CC BY 4.0) or a file in GFZ's Kp/ap/F10.7 format")
    iono.add_argument("--f107-mode", choices=("daily", "81day"), default="81day",
                      help="previous day's F10.7obs, or its trailing 81-day mean (default)")
    iono.add_argument("--maps", choices=("ccir", "ursi"), default="ccir", help="foF2 map family (default ccir)")
    iono.add_argument("--storm", action="store_true",
                      help="apply woof's storm-time correction to foF2 (filtered ap from the index record)")
    iono.add_argument("--storm-fit", help="a storm-time fit file (default: the shipped one)")
    iono.add_argument("--dlat", type=float, default=2.5, help="grid spacing in latitude, deg (default 2.5)")
    iono.add_argument("--dlon", type=float, default=5.0, help="grid spacing in longitude, deg (default 5)")
    iono.add_argument("--alt-min", type=float, default=60.0, help="TEC integration floor, km (default 60)")
    iono.add_argument("--alt-max", type=float, default=2000.0, help="TEC integration top, km (default 2000)")
    iono.add_argument("--cells", type=int, default=1940, help="TEC integration cells (default 1940)")
    iono.add_argument("--out", type=Path, help="write the fields to this .npz (receipt beside it)")
    iono.set_defaults(func=ionosphere_main)
    aur = commands.add_parser(
        "aurora", help="auroral oval (Starkov 1994) from Kp, AL or the solar wind, on a global grid, on the GPU")
    aur.add_argument("--time", required=True, help="first valid time, ISO UTC (2024-05-10T18:00Z)")
    aur.add_argument("--count", type=int, default=1, help="number of times (default 1)")
    aur.add_argument("--step-hours", type=float, default=1.0, help="hours between times (default 1)")
    activity = aur.add_mutually_exclusive_group()
    activity.add_argument("--kp", type=float, help="a fixed Kp (0-9)")
    activity.add_argument("--indices", default="gfz",
                          help="observed Kp: 'gfz' (fetched, CC BY 4.0) or a file in GFZ's format")
    activity.add_argument("--solar-wind",
                          help="Kp estimated from the solar wind: 'omni' (NASA SPDF OMNI2, fetched), "
                               "'swpc' (NOAA real time, last 24 h) or OMNI2 file(s) joined by ':'")
    aur.add_argument("--observed-al", action="store_true",
                     help="with an OMNI2 solar wind, drive the oval with the observed AL instead of Kp")
    aur.add_argument("--energy", action="store_true",
                     help="with a solar wind, add OVATION Prime 2010 energy and number fluxes and hemispheric "
                          "power, and with OMNI2 the Knipp et al. 2004 Joule power")
    aur.add_argument("--ovation", help="OVATION Prime coefficients: the distribution's zip or its premodel "
                                       "directory (default: the pinned zip, fetched)")
    aur.add_argument("--dlat", type=float, default=1.0, help="grid spacing in latitude, deg (default 1)")
    aur.add_argument("--dlon", type=float, default=2.0, help="grid spacing in longitude, deg (default 2)")
    aur.add_argument("--out", type=Path, help="write the fields to this .npz (receipt beside it)")
    aur.set_defaults(func=aurora_main)
    return parser


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="woof")
    register_cli(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
