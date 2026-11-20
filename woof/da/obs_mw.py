"""Microwave sounder brightness temperatures as LETKF observation batches (opt-in).

The polar-sounder counterpart of :mod:`woof.da.obs_radiance`.  It takes the
footprints :func:`woof.obs.wis2_mw.read_mw` returns (AMSU-A, MHS, ATMS), a
background ensemble and an open :class:`woof.rtm.crtm.ClearSkyOperator`
holding the footprints' CRTM sensors, and returns one
:class:`woof.da.letkf.GriddedObs` per sensor and channel.

What it does, in order (each refusal counted in the provenance):

1. **Thinning.**  One footprint per ``thin`` x ``thin`` block of model
   columns: the footprint nearest the block's centre column, within
   ``max_distance_km``, one column per footprint.
2. **Clear sky over sea only.**  The ensemble-mean column must be open
   water (no land, sea ice or snow): CRTM's FASTEM sea emissivity is the
   only surface model used here; land comes later.
3. **H(x) per member**, CRTM on each member's column with the footprint's
   own geometry, for the assimilated channels and the channels the screens
   read.
4. **WRFDA's screens**, as ``da_qc_amsua``, ``da_qc_mhs`` and ``da_qc_atms``
   (v4.7.1) apply them over sea:

   * the scan edges: AMSU-A positions 1-3 and 28-30, MHS 1-8 and 83-90;
   * scattering: AMSU-A ``O(1) - O(15) >= 3 K`` and MHS ``O(1) - O(2) >= 3 K``
     refuse every channel (AMSU-A without channel 15, as Metop-B: the
     channel 1-2 cloud liquid retrieval of GSI at 0.2 kg/m2 instead; a
     footprint neither screen can clear is refused); ATMS ``O-B(1) >= 3 K`` refuses channels 1-8,
     ``|O-B(3)| >= 5 K`` channels 1-8 and 16-22, ``O(16) - O(17) >= 3 K``
     channels 16-22;
   * background cloud: the ensemble-mean cloud liquid path below 100 hPa at
     least 0.2 kg/m2 refuses every channel (only when the caller passes the
     condensate; recorded either way);
   * per channel ``|O-B| > 15 K`` or ``|O-B| > 3 x error``.

5. **Placement.**  As the imager adapter: one observation per column, at
   the model level nearest the peak of the ensemble-mean temperature
   Jacobian.  A channel whose Jacobian peaks in the model's top layer sees
   above the model top and is refused.

Channels and errors are WRFDA's ``radiance_info`` (``metop-1-amsua.info``,
``metop-1-mhs.info``, ``jpss-0-atms.info``; one table per instrument, the
same for every platform there).  Bias correction is a static per sensor and
channel offset (``bias_k``, O-B measured beforehand) subtracted from O;
variational bias correction is not implemented.  Errors are standard
deviations in K.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Mapping

import numpy as np

from woof.da.letkf import GriddedObs, Localization
from woof.da.obs_radiance import ColumnBuilder, _mean_profiles, _unit_vectors
from woof.obs.wis2_mw import MwFootprints

ADAPTER_SCHEMA = "gpuwm-da.mw-radiance-obs-adapter.v1"

#: WRFDA radiance_info (v4.7.1): the channels with ``iuse = 1`` and every channel's error (K).
WRFDA_CHANNELS = {
    "amsua": (5, 6, 7, 8, 9),
    "mhs": (3, 4, 5),
    "atms": (6, 7, 8, 9, 10, 18, 19, 20, 21, 22),
}
WRFDA_ERROR_K = {
    "amsua": dict(zip(range(1, 16), (7.0, 6.0, 5.0, 0.6, 0.3, 0.25, 0.27, 0.27, 0.3, 0.4, 0.7, 1.1, 2.1,
                                     10000.0, 9.0))),
    "mhs": dict(zip(range(1, 6), (2.5, 2.5, 2.5, 2.0, 2.0))),
    "atms": dict(zip(range(1, 23), (7.0, 6.0, 5.0, 0.6, 0.6, 0.3, 0.25, 0.27, 0.27, 0.3, 0.4, 0.7, 1.1, 2.1,
                                    10000.0, 2.5, 2.5, 2.0, 2.0, 2.0, 2.0, 2.0))),
}
#: Scan positions refused as limb (da_qc_*: AMSU-A <= 3 or >= 28, MHS <= 8 or >= 83, ATMS none in 1-96).
LIMB_POSITIONS = {"amsua": (3, 28), "mhs": (8, 83), "atms": (0, 97)}
#: The channels each instrument's screens read besides the assimilated ones (observed only, or O-B).
SCREEN_OBSERVED = {"amsua": (1, 15), "mhs": (1, 2), "atms": (16, 17)}
SCREEN_SIMULATED = {"amsua": (), "mhs": (), "atms": (1, 3)}


class MwAdapterError(ValueError):
    """The footprints, the ensemble and the settings cannot be reconciled."""


@dataclass(frozen=True)
class MwSettings:
    """The adapter's choices; the defaults are WRFDA's where it has one."""

    channels: tuple = ()                 #: assimilated channels; empty: WRFDA's for the instrument
    error_k: Mapping = field(default_factory=dict)   #: channel -> K; missing: WRFDA's
    #: Static observed-minus-background offset, K, per ``"<sensor>:<channel>"`` (subtracted from O).
    bias_k: Mapping = field(default_factory=dict)
    thin: int = 2
    max_distance_km: float = 60.0
    omb_max_k: float = 15.0
    omb_sigma: float = 3.0
    scattering_k: float = 3.0
    clwp_max_kgm2: float = 0.2
    error_inflation: float = 1.0

    def __post_init__(self) -> None:
        if int(self.thin) < 1:
            raise MwAdapterError("thin must be at least 1")
        if not np.isfinite(self.error_inflation) or self.error_inflation < 1.0:
            raise MwAdapterError("error_inflation must be finite and >= 1")
        if any(not np.isfinite(float(v)) or float(v) <= 0 for v in self.error_k.values()):
            raise MwAdapterError("observation errors must be finite and positive")

    def chosen(self, instrument: str) -> tuple:
        return tuple(int(c) for c in (self.channels or WRFDA_CHANNELS[instrument]))

    def error(self, instrument: str, channel: int) -> float:
        value = self.error_k.get(channel, WRFDA_ERROR_K[instrument].get(channel))
        if value is None:
            raise MwAdapterError(f"no observation error for {instrument} channel {channel}")
        return float(value) * self.error_inflation


def select_footprints(fp: MwFootprints, lat, lon, settings: MwSettings) -> dict:
    """The thinned model columns that have a footprint, and that footprint (one column each)."""
    from scipy.spatial import cKDTree

    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    ny, nx = lat.shape
    half = settings.thin // 2
    jj, ii = np.meshgrid(np.arange(half, ny, settings.thin), np.arange(half, nx, settings.thin), indexing="ij")
    jj, ii = jj.ravel(), ii.ravel()
    # a limb footprint, or one with no channel at all, never takes a column a neighbour could serve
    lo, hi = LIMB_POSITIONS[fp.instrument]
    limb = (fp.fov <= lo) | (fp.fov >= hi)
    empty = ~np.isfinite(fp.bt).any(axis=1)
    usable = np.flatnonzero(np.isfinite(fp.lat) & np.isfinite(fp.lon) & ~limb & ~empty)
    counts = dict(model_columns=int(jj.size), footprints=int(fp.count), limb_footprints=int(limb.sum()),
                  footprints_without_channels=int((empty & ~limb).sum()))
    if usable.size == 0:
        return dict(jj=jj[:0], ii=ii[:0], footprint=usable, counts=counts)
    tree = cKDTree(_unit_vectors(fp.lat[usable], fp.lon[usable]))
    chord = 2.0 * np.sin(settings.max_distance_km / 6371.0 / 2.0)
    distance, nearest = tree.query(_unit_vectors(lat[jj, ii], lon[jj, ii]), distance_upper_bound=chord)
    found = np.isfinite(distance)
    jj, ii, foot, distance = jj[found], ii[found], usable[nearest[found]], distance[found]
    order = np.lexsort((distance, foot))
    first = np.ones(order.size, bool)
    first[1:] = foot[order][1:] != foot[order][:-1]
    keep = np.zeros(order.size, bool)
    keep[order[first]] = True
    counts.update(with_footprint=int(found.sum()), footprint_duplicates_dropped=int((~keep).sum()))
    return dict(jj=jj[keep], ii=ii[keep], footprint=foot[keep], counts=counts)


def grody_clw_kgm2(tb1, tb2, zenith_deg):
    """Cloud liquid path over sea from AMSU-A channels 1-2 (Grody et al. 2001; GSI's ``ret_amsua``).

    ``cos(z) * (d0 + 0.754 ln(285 - Tb1) - 2.265 ln(285 - Tb2))`` with
    ``d0 = 8.24 - (2.622 - 1.846 cos z) cos z``, floored at 0.  A channel at
    or above 284 K (or missing) gives NaN: the retrieval does not hold there.
    """

    tb1, tb2 = np.asarray(tb1, float), np.asarray(tb2, float)
    mu = np.cos(np.radians(np.asarray(zenith_deg, float)))
    d0 = 8.24 - (2.622 - 1.846 * mu) * mu
    valid = (tb1 < 284.0) & (tb2 < 284.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        clw = mu * (d0 + 0.754 * np.log(285.0 - tb1) - 2.265 * np.log(285.0 - tb2))
    return np.where(valid, np.maximum(clw, 0.0), np.nan)


def _column_clwp(condensate, pressure, jj, ii):
    """Cloud liquid path (kg/m2) of the model columns, layers at or below 100 hPa (as WRFDA sums it)."""
    cond = np.asarray(condensate, float)[:, jj, ii]
    return np.where(np.asarray(pressure, float)[:, jj, ii] >= 10000.0, cond, 0.0).sum(axis=0)


def mw_to_gridded_obs(fp: MwFootprints, *, lat, lon, pressure_pa, columns: ColumnBuilder, members: int, operator,
                      settings: MwSettings | None = None, localization: Localization | None = None,
                      cloud_liquid_kgm2=None, when: datetime | None = None) -> tuple[list[GriddedObs], dict]:
    """One :class:`GriddedObs` per sensor and assimilated channel, and the provenance.

    ``lat``/``lon`` are the model mass points ``(ny, nx)``; ``pressure_pa`` the
    ensemble-mean model pressure ``(nz, ny, nx)``, level 0 at the bottom.
    ``columns`` builds member columns (:func:`woof.da.obs_radiance.wrf_columns`).
    ``cloud_liquid_kgm2`` is the ensemble-mean cloud liquid mass per model
    layer ``(nz, ny, nx)`` for the background cloud screen, or None to skip it
    (recorded).  ``when`` (default: the footprints' median time) dates CRTM.
    """
    settings = settings or MwSettings()
    instrument = fp.instrument
    if instrument not in WRFDA_CHANNELS:
        raise MwAdapterError(f"no WRFDA channel table for {instrument}")
    pressure = np.asarray(pressure_pa, dtype=float)
    if pressure.ndim != 3 or pressure.shape[1:] != np.shape(lat):
        raise MwAdapterError(f"pressure_pa must be (nz, ny, nx) on the lat/lon grid, got {pressure.shape}")
    if members < 2:
        raise MwAdapterError("an ensemble needs at least two members")
    chosen = settings.chosen(instrument)
    channel_list = [int(c) for c in fp.channels]
    absent = [c for c in chosen if c not in channel_list]
    if absent:
        raise MwAdapterError(f"{instrument} footprints carry channels {channel_list}, not {absent}")
    nz, ny, nx = pressure.shape
    if when is None and fp.count:
        when = datetime.fromisoformat(str(np.sort(fp.time)[fp.count // 2].astype("datetime64[s]")))
    picked = select_footprints(fp, lat, lon, settings)
    jj, ii, foot = picked["jj"], picked["ii"], picked["footprint"]
    counts = dict(picked["counts"])
    col = {c: channel_list.index(c) for c in channel_list}
    obs = fp.bt[foot]                                                     # (n, nchan) observed
    sim_channels = sorted(set(chosen) | set(SCREEN_SIMULATED[instrument]))
    n = foot.size
    simulated = np.full((members, n, len(sim_channels)), np.nan)
    peak_layer = np.full((n, len(sim_channels)), -1)
    peak_hpa = np.full((n, len(sim_channels)), np.nan)
    sea = np.zeros(n, bool)
    sensors = fp.sensor[foot]
    open_sensors = set(getattr(operator, "sensors", ()) or ())
    for sensor in sorted(set(sensors.tolist())):
        rows = np.flatnonzero(sensors == sensor)
        if open_sensors and sensor not in open_sensors:
            counts[f"sensor_{sensor}_not_open_in_crtm"] = int(rows.size)
            continue
        profiles = [columns(r, jj[rows], ii[rows]) for r in range(members)]
        mean = _mean_profiles(profiles)
        water = (mean.land_fraction <= 0.0) & (mean.ice_fraction <= 0.0) & (mean.snow_fraction <= 0.0)
        sea[rows] = water
        if not water.any():
            continue
        rows, profiles = rows[water], [p.subset(water) for p in profiles]
        mean = mean.subset(water)
        f = foot[rows]
        geometry = dict(sensor_zenith=fp.sensor_zenith[f], sensor_scan=fp.scan_angle[f],
                        sensor_azimuth=fp.sensor_azimuth[f], solar_zenith=fp.solar_zenith[f],
                        solar_azimuth=fp.solar_azimuth[f], latitude=fp.lat[f], longitude=fp.lon[f])
        for r, prof in enumerate(profiles):
            simulated[r, rows] = operator.simulate(sensor, prof, geometry, when,
                                                   channels=sim_channels).brightness_temperature
        k = operator.simulate(sensor, mean, geometry, when, channels=sim_channels, jacobian=True)
        layer = np.argmax(k.k_temperature, axis=2)                         # (rows, channels), 0 = top
        peak_layer[rows] = layer
        peak_hpa[rows] = np.take_along_axis(mean.pressure_hpa[:, None, :].repeat(len(sim_channels), 1),
                                            layer[..., None], axis=2)[..., 0]
    counts["not_open_sea"] = int((~sea).sum())
    hbar = simulated.mean(axis=0) if n else np.zeros((0, len(sim_channels)))
    spread2 = simulated.var(axis=0, ddof=1) if n else np.zeros((0, len(sim_channels)))
    sidx = {c: sim_channels.index(c) for c in sim_channels}

    # Footprint screens: a False refuses the channels named.
    good = np.ones((n, len(channel_list)), bool)
    good[~sea] = False
    lo, hi = LIMB_POSITIONS[instrument]
    limb = (fp.fov[foot] <= lo) | (fp.fov[foot] >= hi)
    good[limb] = False
    counts["limb"] = int((sea & limb).sum())

    def refuse(mask, channels, name):
        mask = mask & sea & ~limb
        for c in channels:
            if c in col:
                good[mask, col[c]] = False
        counts[name] = counts.get(name, 0) + int(mask.sum())

    def observed(c):
        return obs[:, col[c]]

    def departure(c):
        # bias-corrected, as WRFDA's tb_inv is when da_qc_atms reads it
        bias = np.array([float(settings.bias_k.get(f"{s}:{c}", 0.0)) for s in sensors])
        return observed(c) - bias - hbar[:, sidx[c]]

    with np.errstate(invalid="ignore"):
        if instrument in ("amsua", "mhs"):
            a, b = SCREEN_OBSERVED[instrument]
            index = observed(a) - observed(b)
            refuse(index >= settings.scattering_k, channel_list, "scattering_index")
            missing = ~np.isfinite(index)
            if instrument == "amsua":
                # Metop-B's channel 15 is dead: where the index cannot be
                # formed, the observed cloud liquid of channels 1-2 stands in
                # (GSI's retrieval); a footprint neither can clear is refused.
                clw = grody_clw_kgm2(observed(1), observed(2), fp.sensor_zenith[foot])
                refuse(missing & ~(clw < settings.clwp_max_kgm2), channel_list, "scattering_index_missing_clw")
            else:
                refuse(missing, channel_list, "scattering_index_missing")
        else:
            refuse(departure(1) >= settings.scattering_k, range(1, 9), "scattering_omb_ch1")
            refuse(np.abs(departure(3)) >= 5.0, list(range(1, 9)) + list(range(16, 23)), "scattering_omb_ch3")
            refuse(observed(16) - observed(17) >= settings.scattering_k, range(16, 23), "scattering_index_16_17")
            # a screen that cannot be formed does not clear its channels
            low = ~np.isfinite(departure(1) + departure(3))
            refuse(low, list(range(1, 9)) + list(range(16, 23)), "scattering_screen_missing")
            refuse(~low & ~np.isfinite(observed(16) - observed(17)), range(16, 23), "scattering_screen_missing")
    cloud_note = "skipped: no background cloud liquid supplied"
    if cloud_liquid_kgm2 is not None and n:
        clwp = _column_clwp(cloud_liquid_kgm2, pressure, jj, ii)
        cloudy = clwp >= settings.clwp_max_kgm2
        refuse(cloudy, channel_list, "background_cloud_liquid")
        cloud_note = f"{int((cloudy & sea & ~limb).sum())} sea footprints refused"

    model_p = pressure[:, jj, ii].T / 100.0                                # (n, nz) hPa, bottom first
    shape = (nz, ny, nx)
    batches, per_channel = [], {}
    for sensor in sorted(set(sensors.tolist())):
        on = sensors == sensor
        for c in chosen:
            s = sidx[c]
            error = settings.error(instrument, c)
            bias = float(settings.bias_k.get(f"{sensor}:{c}", 0.0))
            corrected = observed(c) - bias
            dep = corrected - hbar[:, s]
            keep = on & good[:, col[c]] & np.isfinite(dep) & np.isfinite(spread2[:, s])
            with np.errstate(invalid="ignore"):
                big = keep & (np.abs(dep) > settings.omb_max_k)
                sigma = keep & ~big & (np.abs(dep) > settings.omb_sigma * error)
            top = keep & ~big & ~sigma & (peak_layer[:, s] == 0)
            keep &= ~big & ~sigma & ~top
            level = np.zeros(n, int)
            if keep.any():
                level[keep] = np.argmin(np.abs(np.log(model_p[keep]) - np.log(peak_hpa[keep, s])[:, None]), axis=1)
            values, errors = np.zeros(shape), np.ones(shape)
            mask = np.zeros(shape, bool)
            sim = np.zeros((members,) + shape)
            k_, j_, i_ = level[keep], jj[keep], ii[keep]
            values[k_, j_, i_] = corrected[keep]
            errors[k_, j_, i_] = error
            mask[k_, j_, i_] = True
            sim[:, k_, j_, i_] = simulated[:, keep, s]
            name = f"mw_{sensor}_ch{c}"
            batches.append(GriddedObs(name=name, values=values, errors=errors, simulated=sim, mask=mask,
                                      localization=localization))
            d = dep[keep]
            per_channel[name] = dict(
                assimilated=int(keep.sum()), omb_abs_rejected=int(big.sum()), omb_sigma_rejected=int(sigma.sum()),
                peak_above_model_top=int(top.sum()), error_k=error, bias_k=bias,
                omb_mean_k=float(d.mean()) if d.size else None, omb_std_k=float(d.std()) if d.size else None,
                ensemble_spread_k=float(np.sqrt(spread2[keep, s].mean())) if d.size else None,
                peak_pressure_hpa=dict(zip(("p10", "p50", "p90"),
                                           np.percentile(peak_hpa[keep, s], [10, 50, 90]).round(1).tolist()))
                if d.size else None)
    provenance = dict(
        schema=ADAPTER_SCHEMA, stability="experimental", instrument=instrument,
        valid_time=when.isoformat() if when else None, sensors=sorted(set(sensors.tolist())), members=members,
        grid_shape=list(shape), channels=list(chosen), background_cloud=cloud_note, counts=counts,
        settings={k: (dict(v) if isinstance(v, Mapping) else v) for k, v in asdict(settings).items()},
        batches=per_channel, reader_counts=dict(fp.counts),
        crtm=dict(library=str(getattr(operator, "library_path", "")), receipt=getattr(operator, "receipt", None)),
        notes=["clear sky over open sea only (FASTEM); land, sea ice and snow footprints are refused",
               "WRFDA da_qc_amsua/mhs/atms screens and radiance_info channels and errors (v4.7.1)",
               "one observation per column at the ensemble-mean temperature Jacobian peak",
               "bias_k is a static offset; variational bias correction is not implemented"])
    return batches, provenance


__all__ = ["ADAPTER_SCHEMA", "LIMB_POSITIONS", "MwAdapterError", "MwSettings", "WRFDA_CHANNELS", "WRFDA_ERROR_K",
           "mw_to_gridded_obs", "select_footprints"]
