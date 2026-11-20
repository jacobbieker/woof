"""WIS2 microwave sounder radiances (AMSU-A, MHS, ATMS) from the archive (opt-in).

Reads ``satellite/amsua``, ``satellite/mhs`` and ``satellite/atms`` of the
``bkr/obs/wis2.icechunk`` store (see :mod:`woof.obs.wis2`) in one time window
and box, and returns each instrument's footprints with their geometry and
brightness temperatures (:class:`MwFootprints`).  The LETKF adapter
:func:`woof.da.obs_mw.mw_to_gridded_obs` turns them into observation batches
with CRTM (:mod:`woof.rtm.crtm`).  Nothing here runs unless asked for:
``woof obs wis2 radiances`` or a caller of :func:`read_mw`.

What the reader checks (each refusal is counted by name):

* the store's ``quality_flag`` (every bit but ``time_too_old``) and each
  channel's ``brightness_temperature_quality`` (a flagged channel is NaN);
* brightness temperatures outside 100-340 K are NaN (a fill value; no
  clear-sky scene on Earth is outside);
* a footprint needs a field-of-view number inside the scan, a satellite
  zenith angle up to 65 degrees (CRTM's transmittance fits stop there) and
  a satellite CRTM has coefficients for, by name (:data:`INSTRUMENTS`);
* copies relayed by several centres are one footprint (first arrival).

The scan angle is not in the store; it is the field of view's nominal angle,
``(fov - (n + 1) / 2) * step`` for the scan's ``n`` positions and step
(AMSU-A 30 x 3.33, MHS 90 x 1.11, ATMS 96 x 1.11 degrees).  CRTM uses it for
the polarisation of the cross-track channels.

``scan_quality_flags``, ``field_of_view_quality_flags`` and the ATMS
``granule_quality_flags`` are kept in the footprints but not interpreted: the
centres set them on most Metop scans (bits not documented in the store).
Land, sea ice, cloud and scattering screens need the background and are the
adapter's (WRFDA's ``da_qc_amsua``, ``da_qc_mhs``, ``da_qc_atms``).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from woof.obs import wis2

BT_BOUNDS = (100.0, 340.0)
MAX_SENSOR_ZENITH_DEG = 65.0  # woof.rtm.crtm.MAX_SENSOR_ZENITH_DEG

#: Per instrument: the store group, channel count, scan positions, scan step
#: (degrees) and the CRTM sensor id of each BUFR satellite identifier (0-01-007).
INSTRUMENTS = {
    "amsua": dict(group="satellite/amsua", channels=15, positions=30, step_deg=10.0 / 3.0,
                  sensors={3: "amsua_metop-b", 4: "amsua_metop-a", 5: "amsua_metop-c", 206: "amsua_n15",
                           209: "amsua_n18", 223: "amsua_n19", 784: "amsua_aqua"}),
    "mhs": dict(group="satellite/mhs", channels=5, positions=90, step_deg=10.0 / 9.0,
                sensors={3: "mhs_metop-b", 4: "mhs_metop-a", 5: "mhs_metop-c", 209: "mhs_n18", 223: "mhs_n19"}),
    "atms": dict(group="satellite/atms", channels=22, positions=96, step_deg=1.11,
                 sensors={224: "atms_npp", 225: "atms_n20", 226: "atms_j2"}),
}


@dataclass
class MwFootprints:
    """One instrument's footprints: ``n`` rows, ``bt`` ``(n, nchan)`` in K (NaN where refused)."""

    instrument: str
    channels: np.ndarray        # (nchan,) instrument channel numbers
    sensor: np.ndarray          # (n,) CRTM sensor id
    lat: np.ndarray
    lon: np.ndarray
    time: np.ndarray            # datetime64[s]
    received: np.ndarray        # datetime64[s]
    fov: np.ndarray             # field of view (scan position), 1-based
    scan_angle: np.ndarray      # degrees
    sensor_zenith: np.ndarray
    sensor_azimuth: np.ndarray
    solar_zenith: np.ndarray
    solar_azimuth: np.ndarray
    bt: np.ndarray
    counts: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.sensor)
        for name in ("lat", "lon", "fov", "scan_angle", "sensor_zenith", "sensor_azimuth", "solar_zenith",
                     "solar_azimuth"):
            setattr(self, name, np.asarray(getattr(self, name), float).reshape(n))
        self.bt = np.asarray(self.bt, float).reshape(n, len(self.channels))
        self.sensor = np.asarray(self.sensor, str)

    @property
    def count(self) -> int:
        return len(self.sensor)

    def subset(self, index) -> "MwFootprints":
        pick = {k: getattr(self, k)[index] for k in ("sensor", "lat", "lon", "time", "received", "fov",
                                                       "scan_angle", "sensor_zenith", "sensor_azimuth",
                                                       "solar_zenith", "solar_azimuth", "bt")}
        return MwFootprints(instrument=self.instrument, channels=self.channels, counts=dict(self.counts), **pick)

    def save(self, path) -> None:
        """An ``.npz`` of the footprints (read back with :meth:`load`)."""
        arrays = {k: getattr(self, k) for k in ("channels", "lat", "lon", "fov", "scan_angle", "sensor_zenith",
                                                 "sensor_azimuth", "solar_zenith", "solar_azimuth", "bt")}
        np.savez_compressed(path, instrument=np.array(self.instrument), sensor=self.sensor.astype("U32"),
                            time=self.time.astype("datetime64[s]").astype(np.int64),
                            received=self.received.astype("datetime64[s]").astype(np.int64),
                            counts=np.array(json.dumps(self.counts)), **arrays)

    @classmethod
    def load(cls, path) -> "MwFootprints":
        with np.load(path, allow_pickle=False) as z:
            fields = {k: z[k] for k in ("channels", "lat", "lon", "fov", "scan_angle", "sensor_zenith",
                                        "sensor_azimuth", "solar_zenith", "solar_azimuth", "bt", "sensor")}
            return cls(instrument=str(z["instrument"]), time=z["time"].astype("datetime64[s]"),
                       received=z["received"].astype("datetime64[s]"), counts=json.loads(str(z["counts"])),
                       **fields)


def scan_angle_deg(instrument: str, fov) -> np.ndarray:
    """The nominal scan angle of a 1-based field of view (degrees; negative on the scan's first half)."""
    spec = INSTRUMENTS[instrument]
    return (np.asarray(fov, float) - (spec["positions"] + 1) / 2.0) * spec["step_deg"]


def read_mw(root, instrument: str, start, end, box, counts: Counter | None = None) -> MwFootprints:
    """One instrument's footprints in the window and box (see the module notes for the checks)."""
    if instrument not in INSTRUMENTS:
        raise wis2.Wis2Error(f"unknown instrument {instrument!r}; known: {', '.join(INSTRUMENTS)}")
    spec = INSTRUMENTS[instrument]
    counts = Counter() if counts is None else counts
    label = instrument
    group = wis2._group(root, spec["group"])
    channels = np.asarray(group["channel_number"][:], int).reshape(-1) if "channel_number" in group \
        else np.arange(1, spec["channels"] + 1)
    if channels.size != spec["channels"]:
        raise wis2.Wis2Error(f"{spec['group']}: {channels.size} channels, expected {spec['channels']}")
    index, time = wis2._sat_select(group, start, end, box, counts, label)
    counts[f"{label}_in_window"] += len(index)
    empty = MwFootprints(instrument=instrument, channels=channels, sensor=np.array([], str), lat=[], lon=[],
                         time=np.array([], "datetime64[s]"), received=np.array([], "datetime64[s]"), fov=[],
                         scan_angle=[], sensor_zenith=[], sensor_azimuth=[], solar_zenith=[], solar_azimuth=[],
                         bt=np.zeros((0, channels.size)), counts=counts)
    if index.size == 0:
        return empty
    ok = wis2._quality_ok(group, "quality_flag", index, counts, label)
    index, time = index[ok], time[ok]
    if index.size == 0:
        return empty
    bt = np.atleast_2d(wis2._numbers(group["brightness_temperature"], index).reshape(len(index), -1))
    if "brightness_temperature_quality" in group:
        q = np.atleast_2d(wis2._numbers(group["brightness_temperature_quality"], index).reshape(len(index), -1))
        flagged = np.isfinite(q) & (q != 0) & np.isfinite(bt)  # a missing quality value does not refuse
        counts[f"{label}_brightness_temperature_quality"] += int(flagged.sum())
        bt = np.where(flagged, np.nan, bt)
    gross = np.isfinite(bt) & ((bt < BT_BOUNDS[0]) | (bt > BT_BOUNDS[1]))
    counts[f"{label}_brightness_temperature_out_of_bounds"] += int(gross.sum())
    bt = np.where(gross, np.nan, bt)
    sat = wis2._column(group["satellite_identifier"], index)
    fov = wis2._column(group["field_of_view_number"], index)
    zen = np.abs(wis2._value(group, "satellite_zenith_angle", index, counts, label))
    azi = wis2._column(group["satellite_azimuth_angle"], index)
    szen = wis2._value(group, "solar_zenith_angle", index, counts, label)
    sazi = wis2._column(group["solar_azimuth_angle"], index)
    lat, lon = wis2._column(group["latitude"], index), wis2._column(group["longitude"], index)
    received = wis2._received(group, index)
    keep, seen, sensors = [], set(), []
    arrival = np.where(np.isnat(received), np.iinfo(np.int64).max, received.astype(np.int64))
    for n in np.argsort(arrival, kind="stable"):  # the first arrival of a copy wins (unknown arrival last)
        sid = int(sat[n]) if np.isfinite(sat[n]) else -1
        sensor = spec["sensors"].get(sid)
        if sensor is None:
            counts[f"{label}_satellite_{sid}_no_crtm_sensor"] += 1
            continue
        if not (np.isfinite(fov[n]) and 1 <= fov[n] <= spec["positions"]):
            counts[f"{label}_field_of_view_outside_scan"] += 1
            continue
        if not (np.isfinite(zen[n]) and zen[n] <= MAX_SENSOR_ZENITH_DEG):
            counts[f"{label}_sensor_zenith_missing_or_above_65"] += 1
            continue
        if not np.isfinite(bt[n]).any():
            counts[f"{label}_no_valid_channel"] += 1
            continue
        key = (sid, time[n], int(fov[n]), round(float(lat[n]), 3), round(float(lon[n]), 3))
        if key in seen:
            counts[f"{label}_duplicate_copy"] += 1
            continue
        seen.add(key)
        keep.append(n)
        sensors.append(sensor)
    keep = np.array(keep, int)
    counts[f"{label}_footprints"] += len(keep)
    for sensor, n in Counter(sensors).items():
        counts[f"{label}_{sensor}_footprints"] += n
    return MwFootprints(
        instrument=instrument, channels=channels, sensor=np.array(sensors, str), lat=lat[keep], lon=lon[keep],
        time=time[keep], received=received[keep], fov=fov[keep], scan_angle=scan_angle_deg(instrument, fov[keep]),
        sensor_zenith=zen[keep], sensor_azimuth=np.nan_to_num(azi[keep], nan=0.0),
        solar_zenith=np.nan_to_num(szen[keep], nan=90.0), solar_azimuth=np.nan_to_num(sazi[keep], nan=0.0),
        bt=bt[keep], counts=counts)


# --------------------------------------------------------------------------- CLI
def _cli_radiances(args) -> int:
    instruments = [s.strip() for s in args.instruments.split(",") if s.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        root, snapshot = wis2.open_store(snapshot=args.snapshot)
        start, end = wis2._parse_time(args.start), wis2._parse_time(args.end)
        receipt = {"schema": "woof.obs.wis2-radiances/v1", "store": f"s3://{wis2.BUCKET}/{wis2.PREFIX}",
                   "snapshot": snapshot, "start": str(start), "end": str(end), "box": list(args.bbox),
                   "instruments": {}}
        for instrument in instruments:
            counts = Counter()
            fp = read_mw(root, instrument, start, end, tuple(args.bbox), counts)
            path = out / f"{instrument}.npz"
            fp.save(path)
            receipt["instruments"][instrument] = {"path": str(path), "footprints": fp.count, "qc": dict(counts)}
    except wis2.Wis2Error as error:
        print(f"woof obs wis2 radiances: {error}", file=sys.stderr)
        return 1
    text = json.dumps(receipt, indent=2, sort_keys=True, default=str)
    (out / "receipt.json").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def register_cli(wis2_sub) -> None:
    """``woof obs wis2 radiances``: MW sounder footprints to ``.npz`` (opt-in; never run by a forecast)."""
    p = wis2_sub.add_parser("radiances", help="read AMSU-A, MHS and ATMS footprints in a window and box to .npz")
    p.add_argument("--start", required=True, help="window start, ISO 8601 UTC")
    p.add_argument("--end", required=True, help="window end, ISO 8601 UTC")
    p.add_argument("--bbox", required=True, nargs=4, type=float, metavar=("SOUTH", "NORTH", "WEST", "EAST"))
    p.add_argument("--instruments", default=",".join(INSTRUMENTS), help=f"comma list of {', '.join(INSTRUMENTS)}")
    p.add_argument("--out", required=True, help="directory for <instrument>.npz and receipt.json")
    p.add_argument("--snapshot", help="read this Icechunk snapshot id instead of main's head")
    p.set_defaults(func=_cli_radiances)


__all__ = ["BT_BOUNDS", "INSTRUMENTS", "MwFootprints", "read_mw", "register_cli", "scan_angle_deg"]
