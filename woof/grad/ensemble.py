"""Ensemble sensitivity analysis and ensemble forecast sensitivity to observations (EFSO).

Ensemble sensitivity (Ancell & Hakim 2007; Torn & Hakim 2008): with K
forecasts started from perturbed initial states ``x_k`` and the cost ``J_k``
of each, the regression

    S_j = cov(J, x_j) / var(x_j)

estimates ``dJ/dx_j`` along the ensemble's own covariance.  The response to
an initial perturbation ``dx = eps * p_j``, where ``p_j = cov(X, x_j) /
var(x_j)`` is the regression of the whole state on ``x_j``, is then
``eps * S_j``; :func:`regression_direction` writes that ``p_j`` (localised)
so a finite difference along it checks the prediction (gate G0).

EFSO (Kalnay et al. 2012; Ota et al. 2013), with ``J`` as the forecast-error
measure: an LETKF analysis moves the mean by ``K d`` with ``K = X^a Y^a^T
R^-1 / (K-1)`` and innovations ``d = y - H(x^b)``.  To first order the change
of the verifying cost is

    dJ ~= sum_i d_i / sigma_i^2 * 1/(K-1) * sum_k Y^a'_{ik} J'_k

where ``J'_k`` are the costs of forecasts from the analysis members and
``Y^a'`` their analysis-time observation-space perturbations.  Each term is
one observation's impact; negative values lowered the forecast's cost.
``Y^a'`` comes from the LETKF's own weights (:func:`letkf_with_obs_space`).

Nothing here runs a model.
"""
from __future__ import annotations

import json
from typing import Mapping, Sequence

import numpy as np

ES_SCHEMA = "woof.grad.ensemble-sensitivity/v1"
EFSO_SCHEMA = "woof.grad.efso/v1"

#: kinds located by pressure in the column; the rest are surface or height kinds


class EnsembleError(ValueError):
    """Ensemble inputs that cannot give a sensitivity."""


def _anomalies(values, what: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] < 3:
        raise EnsembleError(f"{what} needs at least 3 members on axis 0, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise EnsembleError(f"{what} has non-finite values")
    return values - values.mean(axis=0)


def ensemble_sensitivity(J: Sequence[float], X) -> dict:
    """``S = cov(J, X) / var(X)`` and the correlation ``r``, per element of ``X`` ``(K, ...)``.

    Elements with no spread get NaN.  ``t`` is the correlation's t statistic
    with ``K - 2`` degrees of freedom; ``|t| > 2`` is a rough 95 % line.
    """
    j = _anomalies(J, "J")
    x = _anomalies(X, "X")
    if x.shape[0] != j.shape[0]:
        raise EnsembleError(f"J has {j.shape[0]} members, X has {x.shape[0]}")
    k = j.shape[0]
    cov = np.tensordot(j, x, axes=(0, 0)) / (k - 1)
    var = np.sum(x * x, axis=0) / (k - 1)
    var_j = float(np.sum(j * j) / (k - 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(var > 0, cov / var, np.nan)
        r = np.where(var > 0, cov / np.sqrt(var * var_j), np.nan) if var_j > 0 else np.full_like(var, np.nan)
        t = r * np.sqrt((k - 2) / np.maximum(1.0 - r * r, 1e-300))
    return {"S": s, "r": r, "t": t, "std": np.sqrt(var), "cov": cov, "members": k, "J_std": np.sqrt(var_j)}


def _stagger_offsets(shape, mass_shape):
    """Index offsets of a field's points from the mass grid's (k, j, i)."""
    nz, ny, nx = mass_shape
    if len(shape) != 3:
        raise EnsembleError(f"field shape {shape} is not 3-D")
    dk = -0.5 if shape[0] == nz + 1 else 0.0
    dj = -0.5 if shape[1] == ny + 1 else 0.0
    di = -0.5 if shape[2] == nx + 1 else 0.0
    if (shape[0] - (dk != 0), shape[1] - (dj != 0), shape[2] - (di != 0)) != tuple(mass_shape):
        raise EnsembleError(f"field shape {shape} is not the mass grid {mass_shape} or one of its staggers")
    return dk, dj, di


def regression_direction(X: Mapping[str, np.ndarray], *, field: str, index: Sequence[int], mass_shape,
                         dx_m: float, horizontal_m: float, vertical_levels: float) -> tuple[dict, dict]:
    """The localised regression of every field of ``X`` on ``X[field][:, k, j, i]``.

    ``X`` maps field names to member states ``(K, nz', ny', nx')`` on their
    own (possibly staggered) grids.  The pattern is ``cov(X_f, x_j) /
    var(x_j)`` tapered by Gaspari-Cohn in horizontal distance (cutoff
    ``horizontal_m``, grid metric ``dx_m``) and level distance (cutoff
    ``vertical_levels``), so it is exactly 1 at the key point.  Returns
    ``(direction, info)``.
    """
    from woof.da.letkf import gaspari_cohn

    k0, j0, i0 = (int(v) for v in index)
    key = _anomalies(np.asarray(X[field])[(slice(None), k0, j0, i0)], f"{field}{tuple(index)}")
    members = key.shape[0]
    var = float(np.sum(key * key) / (members - 1))
    if not var > 0:
        raise EnsembleError(f"{field}{tuple(index)} has no ensemble spread")
    kz, jy, ix = _stagger_offsets(np.shape(X[field])[1:], mass_shape)
    kc, jc, ic = k0 + kz, j0 + jy, i0 + ix
    out = {}
    for name, values in X.items():
        x = _anomalies(values, name)
        if x.shape[0] != members:
            raise EnsembleError(f"{name} has {x.shape[0]} members, {field} has {members}")
        dk, dj, di = _stagger_offsets(x.shape[1:], mass_shape)
        kk = np.arange(x.shape[1]) + dk - kc
        jj = np.arange(x.shape[2]) + dj - jc
        ii = np.arange(x.shape[3]) + di - ic
        horizontal = gaspari_cohn(dx_m * np.hypot(jj[:, None], ii[None, :]), horizontal_m)
        vertical = gaspari_cohn(np.abs(kk), vertical_levels)
        rho = np.asarray(vertical)[:, None, None] * np.asarray(horizontal)[None, :, :]
        out[name] = np.tensordot(key, x, axes=(0, 0)) / (members - 1) / var * rho
    info = {"kind": "localised_regression", "field": field, "index": [k0, j0, i0], "members": members,
            "key_std": float(np.sqrt(var)), "horizontal_m": float(horizontal_m),
            "vertical_levels": float(vertical_levels), "dx_m": float(dx_m)}
    return out, info


def grf_increments(shapes: Mapping[str, Sequence[int]], *, seed: int, dx_km: float, length_scale_km: float,
                   vertical_scale_levels: float, thp_k: float, wind_ms: float, rim: int) -> dict:
    """One member's initial perturbation: a Gaussian random ``thp`` field and a non-divergent wind.

    Draws come from :mod:`woof.da.perturb` (host Philox, so a seed is the
    same member on every machine), each tapered to zero over ``rim`` rows at
    the lateral boundary.  ``shapes`` holds the checkpoint's ``thp``, ``u``
    and ``v`` shapes.
    """
    from woof.da.perturb import boundary_taper, gaussian_random_field, rotational_wind_draw

    mass = tuple(int(n) for n in shapes["thp"])
    nz, ny, nx = mass
    out = {}
    if thp_k:
        field, _info = gaussian_random_field(mass, seed=seed, name="thp", dx_km=dx_km, dy_km=dx_km,
                                             length_scale_km=length_scale_km,
                                             vertical_scale_levels=vertical_scale_levels, xp=np)
        out["thp"] = float(thp_k) * np.asarray(field) * boundary_taper(ny, nx, rim, xp=np)[None]
    if wind_ms:
        u, v, _info = rotational_wind_draw(mass, seed=seed, dx_km=dx_km, dy_km=dx_km,
                                           length_scale_km=length_scale_km,
                                           vertical_scale_levels=vertical_scale_levels, xp=np)
        out["u"] = float(wind_ms) * np.asarray(u) * boundary_taper(ny, nx + 1, rim, xp=np)[None]
        out["v"] = float(wind_ms) * np.asarray(v) * boundary_taper(ny + 1, nx, rim, xp=np)[None]
    for name, values in out.items():
        if tuple(values.shape) != tuple(shapes[name]):
            raise EnsembleError(f"{name} draw is {values.shape}, the checkpoint has {tuple(shapes[name])}")
    return out


def grf_cells(lat, lon, nz: int, *, seed: int, dx_km: float, length_scale_km: float, vertical_scale_levels: float,
              sigma: float, interior=None, name: str = "theta") -> np.ndarray:
    """A Gaussian random field ``(nCells, nz)`` on an unstructured mesh.

    The field is drawn by :func:`woof.da.perturb.gaussian_random_field` on a
    plane grid of spacing ``dx_km`` tangent to the mesh centre (with a margin
    of three length scales, so the FFT's periodic wrap does not correlate
    opposite edges) and sampled at the nearest grid point to each cell.
    It is zero outside ``interior`` (the relaxation rings).
    """
    from woof.da.perturb import gaussian_random_field

    from woof.grad.sensitivity import EARTH_RADIUS_KM

    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    lat0 = float(np.mean(lat))
    lon0 = float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(lon))), np.mean(np.cos(np.radians(lon))))))
    dlon = (lon - lon0 + 180.0) % 360.0 - 180.0
    x = EARTH_RADIUS_KM * np.radians(dlon) * np.cos(np.radians(lat))
    y = EARTH_RADIUS_KM * np.radians(lat - lat0)
    if np.abs(dlon).max() > 60.0 or np.abs(lat - lat0).max() > 45.0:
        raise EnsembleError("grf_cells draws on one tangent plane, which suits a regional mesh only; this mesh "
                            f"spans {np.ptp(lat):.0f} deg of latitude and {2 * np.abs(dlon).max():.0f} of longitude")
    margin = 3.0 * length_scale_km
    x0, y0 = x.min() - margin, y.min() - margin
    nx = int(np.ceil((x.max() + margin - x0) / dx_km)) + 1
    ny = int(np.ceil((y.max() + margin - y0) / dx_km)) + 1
    field, _info = gaussian_random_field((int(nz), ny, nx), seed=seed, name=name, dx_km=dx_km, dy_km=dx_km,
                                         length_scale_km=length_scale_km,
                                         vertical_scale_levels=vertical_scale_levels, xp=np)
    i = np.clip(np.rint((x - x0) / dx_km).astype(np.int64), 0, nx - 1)
    j = np.clip(np.rint((y - y0) / dx_km).astype(np.int64), 0, ny - 1)
    out = float(sigma) * np.asarray(field)[:, j, i].T
    if interior is not None:
        out = out * np.asarray(interior, bool)[:, None]
    return out


def regression_direction_cells(X: Mapping[str, np.ndarray], *, field: str, cell: int, level: int, lat, lon,
                               horizontal_km: float, vertical_levels: float) -> tuple[dict, dict]:
    """:func:`regression_direction` on an unstructured mesh: fields ``(K, nCells, nz)``, great-circle distance."""
    from woof.da.letkf import gaspari_cohn

    from woof.grad.sensitivity import great_circle_km

    key = _anomalies(np.asarray(X[field])[:, int(cell), int(level)], f"{field}[{cell},{level}]")
    members = key.shape[0]
    var = float(np.sum(key * key) / (members - 1))
    if not var > 0:
        raise EnsembleError(f"{field}[{cell},{level}] has no ensemble spread")
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    horizontal = np.asarray(gaspari_cohn(great_circle_km(lat, lon, lat[int(cell)], lon[int(cell)]), horizontal_km))
    out = {}
    for name, values in X.items():
        x = _anomalies(values, name)
        if x.shape[:2] != (members, lat.size):
            raise EnsembleError(f"{name} is {x.shape}; expected ({members}, {lat.size}, nz)")
        vertical = np.asarray(gaspari_cohn(np.abs(np.arange(x.shape[2]) - int(level)), vertical_levels))
        out[name] = np.tensordot(key, x, axes=(0, 0)) / (members - 1) / var * horizontal[:, None] * vertical[None, :]
    info = {"kind": "localised_regression", "field": field, "cell": int(cell), "level": int(level),
            "lat_lon": [float(lat[int(cell)]), float(lon[int(cell)])], "members": members,
            "key_std": float(np.sqrt(var)), "horizontal_km": float(horizontal_km),
            "vertical_levels": float(vertical_levels)}
    return out, info


def centred(members: Sequence[Mapping[str, np.ndarray]]) -> list[dict]:
    """Members with their mean removed, so the ensemble is centred on the unperturbed state."""
    names = sorted(members[0])
    mean = {name: np.mean([m[name] for m in members], axis=0) for name in names}
    return [{name: np.asarray(m[name]) - mean[name] for name in names} for m in members]


def obs_gridpoints(obs, obs_index: np.ndarray, column: np.ndarray, frame) -> np.ndarray:
    """``(k, column)`` of each paired observation: the model level nearest it in ``frame``.

    Pressure kinds take the nearest ``ln p``, refractivity the nearest height,
    and surface kinds level 0.  ``column`` is the pairing's mass column.
    Column-integrated kinds have no level and are refused.
    """
    from woof.grad.obs_cost import COLUMN_KINDS, HEIGHT_KINDS, PRESSURE_KINDS

    kind = obs.kind[obs_index]
    if np.isin(kind, COLUMN_KINDS).any():
        raise EnsembleError(f"column-integrated observations ({sorted(set(kind[np.isin(kind, COLUMN_KINDS)]))}) "
                            "have no model level; leave them out of the LETKF")
    k = np.zeros(len(obs_index), np.int64)
    upper = np.isin(kind, PRESSURE_KINDS)
    if upper.any():
        lnp = np.log(np.asarray(frame.p)[:, column[upper]])
        k[upper] = np.argmin(np.abs(lnp - np.log(obs.pressure_pa[obs_index[upper]])[None, :]), axis=0)
    height = np.isin(kind, HEIGHT_KINDS)
    if height.any():
        z = np.asarray(frame.z)[:, column[height]]
        k[height] = np.argmin(np.abs(z - obs.height_m[obs_index[height]][None, :]), axis=0)
    return k


def layered_batches(flat_index: np.ndarray, values, errors, simulated, *, shape, name: str,
                    localization=None) -> tuple[list, list[np.ndarray]]:
    """LETKF point batches holding every observation, at most one per gridpoint per batch.

    Observations sharing a gridpoint go to successive layers, so none is
    merged or dropped.  Returns ``(batches, rows)`` with ``rows[b]`` the
    observation rows of batch ``b`` in its point order.
    """
    from woof.da.letkf import PointSet, point_batch

    flat_index = np.asarray(flat_index, np.int64)
    values = np.asarray(values, np.float64)
    errors = np.asarray(errors, np.float64)
    simulated = np.asarray(simulated, np.float64)
    layer = np.zeros(len(flat_index), np.int64)
    seen: dict[int, int] = {}
    for row, point in enumerate(flat_index.tolist()):
        layer[row] = seen.get(point, 0)
        seen[point] = layer[row] + 1
    batches, rows = [], []
    for level in range(int(layer.max()) + 1 if len(layer) else 0):
        sel = np.flatnonzero(layer == level)
        sel = sel[np.argsort(flat_index[sel], kind="stable")]
        points = PointSet(flat_index=flat_index[sel], values=values[sel], errors=errors[sel],
                          simulated=simulated[:, sel])
        batches.append(point_batch(f"{name}#{level}", shape, points, localization))
        rows.append(sel)
    return batches, rows


def letkf_with_obs_space(prior: Mapping[str, np.ndarray], batches: Sequence, rows: Sequence[np.ndarray],
                         grid, config, *, solve_namespace=None, progress=None,
                         diagnostics=None) -> tuple[dict, np.ndarray]:
    """One LETKF analysis that also returns ``Y^a``, the analysis ensemble in observation space.

    ``H(x_k)`` of each batch is carried as an extra analysed field that is
    zero except at the batch's observation points, so it is transformed by
    the same local weights as the state at those points: ``Y^a = Y^b W``.
    Returns ``(increments, ya)`` with ``ya`` ``(K, n_obs)``.  ``solve_namespace``
    (e.g. cupy), ``progress`` and ``diagnostics`` pass through to :func:`woof.da.letkf.analyze`:
    the state stays on the host and the local solves run in that namespace.

    Each layer adds one dense ``(K, nz, ny, nx)`` field, so memory and the
    solve grow with the number of layers.  Relaxation (RTPS/RTPP) is
    refused: EFSO needs the analysis perturbations of the Kalman gain
    itself, not relaxed ones.
    """
    import dataclasses

    from woof.da.letkf import analyze

    if float(getattr(config, "rtps_alpha", 0.0) or 0.0) != 0.0:
        raise EnsembleError("EFSO needs the unrelaxed analysis ensemble; run the LETKF with rtps_alpha 0")
    members = next(iter(prior.values())).shape[0]
    shape = tuple(next(iter(prior.values())).shape[1:])
    n = int(sum(len(r) for r in rows))
    ya = np.full((members, n), np.nan)
    work = dict(prior)
    names = []
    for b, batch in enumerate(batches):
        name = f"_grad_yb{b}"
        field = np.zeros((members,) + shape)
        flat = field.reshape(members, -1)
        flat[:, np.asarray(batch.points.flat_index, np.int64)] = np.asarray(batch.points.simulated)
        work[name] = field
        names.append(name)
    config = dataclasses.replace(config, analysis_fields=tuple(config.analysis_fields) + tuple(names))
    increments = analyze(work, list(batches), grid, config, solve_namespace=solve_namespace, progress=progress,
                         diagnostics=diagnostics)
    for b, (batch, sel) in enumerate(zip(batches, rows)):
        index = np.asarray(batch.points.flat_index, np.int64)
        ya[:, sel] = (work[names[b]] + increments[names[b]]).reshape(members, -1)[:, index]
    return {name: increments[name] for name in prior if name in increments}, ya


def efso(innovation, sigma, ya, jf, *, rho=None, chunk_rows: int = 2048) -> np.ndarray:
    """Each observation's first-order impact on the verifying cost.

    ``innovation`` ``d = y - H(mean background)`` and ``sigma`` are ``(n,)``;
    ``ya`` is ``(K, n)``, the unrelaxed analysis ensemble in observation
    space; ``jf`` is the analysis-member forecast costs, either ``(K,)`` or
    per verifying observation ``(K, m)``.  With ``rho`` (a localisation
    between assimilated and verifying observations: an ``(n, m)`` array, or
    a function of assimilated row indices returning their ``(rows, m)``
    block) the per-observation costs are weighted by it, ``chunk_rows``
    assimilated observations at a time so no ``(n, m)`` array need exist.
    """
    d = np.asarray(innovation, np.float64)
    s = np.asarray(sigma, np.float64)
    y = _anomalies(ya, "Y^a")
    j = _anomalies(jf, "J forecast")
    members = y.shape[0]
    if y.shape[1] != d.shape[0] or s.shape != d.shape or j.shape[0] != members:
        raise EnsembleError(f"shapes disagree: d {d.shape}, sigma {s.shape}, Y^a {y.shape}, J {j.shape}")
    weight = d / (s * s)
    if j.ndim == 1:
        if rho is not None:
            raise EnsembleError("a localisation needs per-observation forecast costs (K, m)")
        return weight * (y.T @ j) / (members - 1)
    if rho is None:
        return weight * (y.T @ j.sum(axis=1)) / (members - 1)
    block = rho if callable(rho) else (lambda rows: np.asarray(rho, np.float64)[rows])
    out = np.empty(d.shape[0])
    for start in range(0, d.shape[0], int(chunk_rows)):
        rows = np.arange(start, min(start + int(chunk_rows), d.shape[0]))
        cross = (y[:, rows].T @ j) / (members - 1)
        local = np.asarray(block(rows), np.float64)
        if local.shape != cross.shape:
            raise EnsembleError(f"rho block is {local.shape}, expected {cross.shape}")
        out[rows] = weight[rows] * (cross * local).sum(axis=1)
    return out


def impact_table(impact, platform, kind) -> dict:
    """Total, mean, count and beneficial fraction of the impact for each platform/kind."""
    impact = np.asarray(impact, np.float64)
    out = {}
    for p, k in sorted(set(zip(np.asarray(platform).tolist(), np.asarray(kind).tolist()))):
        sel = (np.asarray(platform) == p) & (np.asarray(kind) == k)
        values = impact[sel]
        out[f"{p}/{k}"] = {"n": int(sel.sum()), "total": float(values.sum()), "mean": float(values.mean()),
                           "beneficial_fraction": float(np.mean(values < 0))}
    out["all"] = {"n": int(impact.size), "total": float(impact.sum()),
                  "mean": float(impact.mean()) if impact.size else 0.0,
                  "beneficial_fraction": float(np.mean(impact < 0)) if impact.size else 0.0}
    return out


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=float)
