"""Geostationary imager brightness temperatures as LETKF observation batches (opt-in).

The radiance counterpart of :mod:`woof.da.obs_goes`.  It takes one
:class:`woof.obs.imager_scene.ImagerScene` (SatWeave, or a native reader), a
background ensemble and an open :class:`woof.rtm.crtm.ClearSkyOperator`, and
returns one :class:`woof.da.letkf.GriddedObs` per band.

What it does, in order:

1. **Thinning.**  One observation per ``thin`` x ``thin`` block of model
   columns, from the scene pixel nearest the block's centre column, within
   ``max_distance_km``.  The pixel must have every band, a sensor zenith
   below ``max_zenith_deg``, no limb flag and an attributed satellite.
2. **H(x) per member.**  CRTM on each member's column, grouped by the
   pixel's CRTM sensor, with the pixel's own geometry.  H is applied to the
   full member state before localisation, as the filter requires.
3. **Clear sky.**  Three tests, each counted in the provenance:
   the window band is uniform over the pixel's 3 x 3 neighbourhood
   (``uniformity_k``); its departure from the ensemble-mean H is within
   ``window_check_k``; and, when the caller passes the background
   condensate, the ensemble-mean condensate path above the level 100 hPa
   below the band's weighting-function peak is below
   ``background_cloud_kgm2``.
4. **Bias and gross error.**  A static per-band offset (``bias_k``, an
   observed-minus-background mean measured beforehand) is subtracted from
   the observation; variational bias correction comes later.  Departures
   beyond ``gross_sigma`` x sqrt(error^2 + ensemble variance of H) are
   dropped.
5. **Placement.**  Like the GOES cloud-water-path batch, a brightness
   temperature is one observation per column, so the mask is True at one
   level only: the model level nearest (in log pressure) to the peak of the
   ensemble-mean temperature Jacobian.  The vertical localisation radius
   then stands in for the weighting function's width; the provenance
   records the peak pressures so the choice can be checked.

Errors are standard deviations in K.  Nothing here changes a default run:
the module is only reached by a caller that builds CRTM and asks for it.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable, Mapping, Sequence

import numpy as np

from woof.da.letkf import GriddedObs, Localization
from woof.obs.imager_scene import FLAG_LIMB, ImagerScene, band
from woof.rtm.profiles import Profiles, profiles_from_wrf_state

ADAPTER_SCHEMA = "gpuwm-da.radiance-obs-adapter.v1"
_GRAVITY = 9.80665


class RadianceAdapterError(ValueError):
    """The scene, the ensemble and the settings cannot be reconciled."""


@dataclass(frozen=True)
class RadianceSettings:
    """The adapter's choices.  Defaults are a starting point, not a tuned set."""

    bands: tuple = (6200, 6900, 7330)
    #: Observation error standard deviation per band, K.
    error_k: Mapping = field(default_factory=lambda: {6200: 2.0, 6900: 2.0, 7330: 2.0})
    #: Static observed-minus-background offset per band, K (subtracted from O).
    bias_k: Mapping = field(default_factory=dict)
    thin: int = 2
    max_distance_km: float = 30.0
    max_zenith_deg: float = 65.0
    window_band: int = 11200
    uniformity_k: float = 1.0
    window_check_k: float = 3.0
    background_cloud_kgm2: float = 0.02
    gross_sigma: float = 3.0
    error_inflation: float = 1.0

    def __post_init__(self) -> None:
        for nm in (*self.bands, self.window_band):
            if band(nm).quantity != "brightness_temperature":
                raise RadianceAdapterError(f"band {nm} nm is not an infrared brightness temperature")
        missing = [nm for nm in self.bands if nm not in self.error_k]
        if missing:
            raise RadianceAdapterError(f"no observation error for bands {missing}")
        if any(not np.isfinite(float(self.error_k[nm])) or float(self.error_k[nm]) <= 0 for nm in self.bands):
            raise RadianceAdapterError("observation errors must be finite and positive")
        if int(self.thin) < 1:
            raise RadianceAdapterError("thin must be at least 1")
        if not np.isfinite(self.error_inflation) or self.error_inflation < 1.0:
            raise RadianceAdapterError("error_inflation must be finite and >= 1")


#: ``columns(member, jj, ii) -> Profiles`` for the model columns ``(jj, ii)``.
ColumnBuilder = Callable[[int, np.ndarray, np.ndarray], Profiles]


def wrf_columns(members: Sequence[Mapping], *, use_theta_m: bool = False) -> ColumnBuilder:
    """A column builder over WRF-named member states (3-D fields ``(nz, ny, nx)``)."""

    def build(member, jj, ii):
        state = members[member]
        picked = {}
        for name in ("T", "P", "PB", "QVAPOR", "PSFC", "TSK", "U10", "V10", "LANDMASK", "XICE", "SNOWC", "HGT"):
            value = np.asarray(state[name])
            picked[name] = value[:, jj, ii].T if value.ndim == 3 else value[jj, ii]
        picked["P_TOP"] = np.asarray(state["P_TOP"], dtype=float).reshape(-1)[0]
        return profiles_from_wrf_state(picked, use_theta_m=use_theta_m)

    return build


def _unit_vectors(lat, lon):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def _local_std(values):
    pad = np.pad(values, 1, mode="edge")
    ny, nx = values.shape
    stack = np.stack([pad[1 + a:1 + a + ny, 1 + b:1 + b + nx] for a in (-1, 0, 1) for b in (-1, 0, 1)])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        std = np.nanstd(stack, axis=0)
    # A window with fewer than 5 finite values is not shown to be uniform.
    return np.where(np.isfinite(stack).sum(axis=0) >= 5, std, np.inf)


def select_columns(scene: ImagerScene, lat, lon, settings: RadianceSettings) -> dict:
    """The thinned model columns that have a usable scene pixel, and that pixel."""

    from scipy.spatial import cKDTree

    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    ny, nx = lat.shape
    half = settings.thin // 2
    jj, ii = np.meshgrid(np.arange(half, ny, settings.thin), np.arange(half, nx, settings.thin), indexing="ij")
    jj, ii = jj.ravel(), ii.ravel()
    slat, slon = np.asarray(scene.lat, float), np.asarray(scene.lon, float)
    if slat.ndim == 1:
        slon, slat = np.meshgrid(slon, slat)
    wanted = sorted({*settings.bands, settings.window_band})
    absent = [nm for nm in wanted if nm not in scene.values]
    if absent:
        raise RadianceAdapterError(f"the scene carries no band {absent}")
    usable = ((scene.platform_index >= 0) & (scene.view_zenith < settings.max_zenith_deg)
              & ((scene.flags & FLAG_LIMB) == 0) & np.isfinite(slat) & np.isfinite(slon))
    for nm in wanted:
        usable &= np.isfinite(np.asarray(scene.values[nm], float))
    flat = np.flatnonzero(usable)
    counts = dict(model_columns=int(jj.size), usable_pixels=int(flat.size))
    if flat.size == 0:
        return dict(jj=jj[:0], ii=ii[:0], pixel=flat, counts=counts)
    tree = cKDTree(_unit_vectors(slat.ravel()[flat], slon.ravel()[flat]))
    chord = 2.0 * np.sin(settings.max_distance_km / 6371.0 / 2.0)
    distance, nearest = tree.query(_unit_vectors(lat[jj, ii], lon[jj, ii]), distance_upper_bound=chord)
    found = np.isfinite(distance)
    jj, ii, pixel, distance = jj[found], ii[found], flat[nearest[found]], distance[found]
    # Several model columns can share one scene pixel: keep the nearest column for each.
    order = np.lexsort((distance, pixel))
    first = np.ones(order.size, bool)
    first[1:] = pixel[order][1:] != pixel[order][:-1]
    survivor = np.zeros(order.size, bool)
    survivor[order[first]] = True
    duplicates = int((~survivor).sum())
    jj, ii, pixel = jj[survivor], ii[survivor], pixel[survivor]
    uniform = _local_std(np.asarray(scene.values[settings.window_band], float)).ravel()[pixel] < settings.uniformity_k
    counts.update(with_pixel=int(found.sum()), pixel_duplicates_dropped=duplicates, uniform=int(uniform.sum()))
    return dict(jj=jj[uniform], ii=ii[uniform], pixel=pixel[uniform], counts=counts)


def _mean_profiles(profiles: Sequence[Profiles]) -> Profiles:
    first = profiles[0]
    averaged = {k: np.mean([getattr(p, k) for p in profiles], axis=0)
                for k in ("level_pressure_hpa", "pressure_hpa", "temperature_k", "water_vapour_gkg", "ozone_ppmv",
                          "skin_temperature_k", "wind_speed_ms", "land_fraction", "water_fraction",
                          "snow_fraction", "ice_fraction")}
    # Wind direction is circular: average the unit vectors, not the degrees.
    rad = np.radians(np.stack([p.wind_direction_deg for p in profiles]))
    direction = np.degrees(np.arctan2(np.sin(rad).mean(axis=0), np.cos(rad).mean(axis=0))) % 360.0
    return Profiles(wind_direction_deg=direction, surface_altitude_m=first.surface_altitude_m, **averaged)


def radiance_to_gridded_obs(scene: ImagerScene, *, lat, lon, pressure_pa, columns: ColumnBuilder, members: int,
                            operator, settings: RadianceSettings | None = None,
                            localization: Localization | None = None, condensate_kgm2=None,
                            when: datetime | None = None) -> tuple[list[GriddedObs], dict]:
    """One :class:`GriddedObs` per band in ``settings.bands``, and the provenance.

    ``lat``/``lon`` are the model mass points ``(ny, nx)``; ``pressure_pa`` is
    the ensemble-mean model pressure ``(nz, ny, nx)``, level 0 at the bottom,
    which places each observation.  ``columns`` builds member columns (see
    :func:`wrf_columns`).  ``condensate_kgm2`` is the ensemble-mean condensate
    mass per model layer ``(nz, ny, nx)`` for the background cloud test, or
    None to skip that test (recorded).
    """

    settings = settings or RadianceSettings()
    pressure = np.asarray(pressure_pa, dtype=float)
    if pressure.ndim != 3 or pressure.shape[1:] != np.shape(lat):
        raise RadianceAdapterError(f"pressure_pa must be (nz, ny, nx) on the lat/lon grid, got {pressure.shape}")
    if members < 2:
        raise RadianceAdapterError("an ensemble needs at least two members")
    nz, ny, nx = pressure.shape
    when = when or scene.time
    picked = select_columns(scene, lat, lon, settings)
    jj, ii, pixel = picked["jj"], picked["ii"], picked["pixel"]
    counts = dict(picked["counts"])
    wanted = sorted({*settings.bands, settings.window_band})
    obs = np.stack([np.asarray(scene.values[nm], float).ravel()[pixel] for nm in wanted], axis=1) if pixel.size \
        else np.zeros((0, len(wanted)))
    sensors = scene.crtm_sensor().ravel()[pixel]
    geometry_all = dict(sensor_zenith=scene.view_zenith, sensor_scan=scene.scan_angle,
                        sensor_azimuth=scene.view_azimuth, solar_zenith=scene.solar_zenith,
                        solar_azimuth=scene.solar_azimuth)
    slat, slon = np.asarray(scene.lat, float), np.asarray(scene.lon, float)
    if slat.ndim == 1:
        slon, slat = np.meshgrid(slon, slat)

    n = pixel.size
    simulated = np.full((members, n, len(wanted)), np.nan)
    peak_hpa = np.full((n, len(wanted)), np.nan)
    for sensor in sorted(set(sensors.tolist())):
        rows = np.flatnonzero(sensors == sensor)
        family = scene.platforms[int(scene.platform_index.ravel()[pixel[rows[0]]])].family
        channels = [band(nm).channels.get(family) for nm in wanted]
        if None in channels:
            raise RadianceAdapterError(f"{sensor} has no channel for every band in {wanted}")
        geometry = {k: np.asarray(v).ravel()[pixel[rows]] for k, v in geometry_all.items()}
        geometry.update(latitude=slat.ravel()[pixel[rows]], longitude=slon.ravel()[pixel[rows]])
        profiles = [columns(r, jj[rows], ii[rows]) for r in range(members)]
        for r, prof in enumerate(profiles):
            simulated[r, rows] = operator.simulate(sensor, prof, geometry, when, channels=channels).brightness_temperature
        mean = _mean_profiles(profiles)
        k = operator.simulate(sensor, mean, geometry, when, channels=channels, jacobian=True)
        layer = np.argmax(k.k_temperature, axis=2)                      # (rows, bands)
        peak_hpa[rows] = np.take_along_axis(mean.pressure_hpa[:, None, :].repeat(len(wanted), 1),
                                            layer[..., None], axis=2)[..., 0]

    hbar = simulated.mean(axis=0) if n else np.zeros((0, len(wanted)))
    spread2 = simulated.var(axis=0, ddof=1) if n else np.zeros((0, len(wanted)))
    w = wanted.index(settings.window_band)
    clear = np.abs(obs[:, w] - hbar[:, w]) < settings.window_check_k
    counts["window_clear"] = int(clear.sum())

    model_p = pressure[:, jj, ii].T / 100.0                              # (n, nz) hPa, bottom first
    batches, per_band = [], {}
    shape = (nz, ny, nx)
    for b, nm in enumerate(wanted):
        if nm not in settings.bands:
            continue
        keep = clear.copy()
        cloud_note = "skipped: no background condensate supplied"
        if condensate_kgm2 is not None:
            cond = np.asarray(condensate_kgm2, dtype=float)[:, jj, ii].T
            limit = peak_hpa[:, b] + 100.0
            path = np.where(model_p <= limit[:, None], cond, 0.0).sum(axis=1)
            cloudy = path >= settings.background_cloud_kgm2
            keep &= ~cloudy
            cloud_note = f"{int((clear & cloudy).sum())} window-clear columns refused"
        error = float(settings.error_k[nm]) * settings.error_inflation
        corrected = obs[:, b] - float(settings.bias_k.get(nm, 0.0))
        departure = corrected - hbar[:, b]
        finite = np.isfinite(hbar[:, b]) & np.isfinite(spread2[:, b]) & np.isfinite(departure)
        with np.errstate(invalid="ignore"):
            gross = finite & ~(np.abs(departure) <= settings.gross_sigma * np.sqrt(error ** 2 + spread2[:, b]))
        keep &= finite & ~gross
        level = np.argmin(np.abs(np.log(model_p) - np.log(peak_hpa[:, b])[:, None]), axis=1)
        values = np.zeros(shape)
        errors = np.ones(shape)
        mask = np.zeros(shape, dtype=bool)
        sim = np.zeros((members,) + shape)
        k_, j_, i_ = level[keep], jj[keep], ii[keep]
        values[k_, j_, i_] = corrected[keep]
        errors[k_, j_, i_] = error
        mask[k_, j_, i_] = True
        sim[:, k_, j_, i_] = simulated[:, keep, b]
        batches.append(GriddedObs(name=f"bt_{nm}", values=values, errors=errors, simulated=sim, mask=mask,
                                  localization=localization))
        kept_d = departure[keep]
        per_band[str(nm)] = dict(
            assimilated=int(keep.sum()), gross_rejected=int((clear & gross).sum()),
            nonfinite_rejected=int((clear & ~finite).sum()), background_cloud=cloud_note,
            error_k=error, bias_k=float(settings.bias_k.get(nm, 0.0)),
            omb_mean_k=float(kept_d.mean()) if kept_d.size else None,
            omb_std_k=float(kept_d.std()) if kept_d.size else None,
            ensemble_spread_k=float(np.sqrt(spread2[keep, b].mean())) if kept_d.size else None,
            peak_pressure_hpa=dict(zip(("p10", "p50", "p90"),
                                       np.percentile(peak_hpa[keep, b], [10, 50, 90]).round(1).tolist()))
            if kept_d.size else None)

    provenance = dict(
        schema=ADAPTER_SCHEMA, stability="experimental", valid_time=when.isoformat(),
        scene_time=scene.time.isoformat(), scene_source=scene.provenance.get("source"),
        licence=scene.provenance.get("licence"), attribution=scene.provenance.get("attribution"),
        sensors=sorted(set(sensors.tolist())), members=members, grid_shape=list(shape),
        settings={k: (dict(v) if isinstance(v, Mapping) else v) for k, v in asdict(settings).items()},
        counts=counts, bands=per_band,
        crtm=dict(library=str(getattr(operator, "library_path", "")), receipt=getattr(operator, "receipt", None)),
        notes=["one observation per column, at the model level nearest the ensemble-mean temperature "
               "Jacobian peak; the vertical localisation radius stands in for the weighting function's width",
               "errors are standard deviations, passed through unsquared",
               "bias_k is a static offset; variational bias correction is not implemented"])
    return batches, provenance
