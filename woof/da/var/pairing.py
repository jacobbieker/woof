"""Hybrid 3DEnVar paired with an LETKF cycle (docs/dev/WRFDA-3DVAR-PLAN.md, plan item V3.2, second half).

The usual EnKF-Var pairing: the LETKF's forecast ensemble gives the hybrid its
flow-dependent ``x'_n`` (:func:`ensemble_from_letkf`), 3DEnVar analyses the
first guess (the ensemble mean, or a control forecast), the LETKF updates the
members' perturbations with its own observations, and the analysis members are
recentred on the variational analysis (:func:`recentre`)::

    x_n^a = x_n^letkf + gamma (x_var^a - mean_n x_n^letkf)

With ``gamma = 1`` the members are ``x_var^a`` plus the LETKF's perturbations: it replaces the LETKF's mean by the variational one, ``gamma = 0``
leaves the LETKF alone.  :func:`hybrid_step` runs one such analysis; the next
cycle's forecasts of the recentred members are the next ``ensemble_from_letkf``.

Scope, as built: the pairing works in the five fields the hybrid works in
(``u``, ``v``, ``t``, ``q``, ``ps``: :class:`~woof.da.var.hybrid.Ensemble`'s
variables, mass points, temperature and specific humidity), in the LETKF's
``(R, nz, ny, nx)`` layout.  The cycle controller that owns the member files
converts its WRF members to these fields (:func:`~woof.da.var.hybrid.member_fields`
reads WRF history files) and writes the recentred fields back; hydrostatic and
dry-column-mass consistency of a recentred member is that controller's
(:func:`~woof.da.var.analysis.xatowrf` does it for the variational mean).
Members with no ensemble-mean first guess are fine, but then the first guess's
own departure from the ensemble mean is part of the recentring shift.  If the
LETKF is given ``ps`` too, its mean update is replaced by the variational one.

Nothing here changes a default: 3-D Var, the hybrid and the LETKF are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from woof.da.var import hybrid as hy

__all__ = ["HybridStep", "ensemble_from_letkf", "from_var_layout", "hybrid_step", "recentre", "to_var_layout",
           "variational_state"]

FIELDS = hy.FIELDS
#: ``xa``'s name for each field (``ps`` is ``psfc``)
_XA = {"u": "u", "v": "v", "t": "t", "q": "q", "ps": "psfc"}


def _host(a):
    """A NumPy view of a NumPy or CuPy array."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _names(names) -> dict:
    out = {k: k for k in FIELDS}
    if names is not None:
        unknown = set(names) - set(FIELDS)
        if unknown:
            raise ValueError(f"pairing field names are {FIELDS}, got {sorted(unknown)}")
        out.update(names)
    return out


def to_var_layout(prior, names=None) -> dict:
    """LETKF stacks ``(R, nz, ny, nx)`` (``ps``: ``(R, ny, nx)`` or ``(R, 1, ny, nx)``) as ``(R, nx, ny, nz)``/``(R, nx, ny)``."""
    nm = _names(names)
    out = {}
    for k in FIELDS:
        a = np.asarray(_host(prior[nm[k]]), dtype=np.float64)
        if k == "ps":
            if a.ndim == 4:
                if a.shape[1] != 1:
                    raise ValueError(f"surface pressure {nm[k]!r} must be 2-D per member, got {a.shape}")
                a = a[:, 0]
            elif a.ndim != 3:
                raise ValueError(f"surface pressure {nm[k]!r} must be (R, ny, nx), got {a.shape}")
            out[k] = np.ascontiguousarray(np.transpose(a, (0, 2, 1)))
        else:
            if a.ndim != 4:
                raise ValueError(f"{nm[k]!r} must be (R, nz, ny, nx), got {a.shape}")
            out[k] = np.ascontiguousarray(np.transpose(a, (0, 3, 2, 1)))
    ne = {v.shape[0] for v in out.values()}
    if len(ne) != 1:
        raise ValueError(f"members differ per field: {sorted(ne)}")
    return out


def from_var_layout(members: dict, names=None, *, like=None) -> dict:
    """The inverse of :func:`to_var_layout`; ``like`` (the LETKF's prior) decides ``ps``'s rank and every field's dtype."""
    nm = _names(names)
    out = {}
    for k in FIELDS:
        a = np.asarray(members[k])
        if k == "ps":
            a = np.transpose(a, (0, 2, 1))
            if like is not None and np.ndim(like[nm[k]]) == 4:
                a = a[:, None]
        else:
            a = np.transpose(a, (0, 3, 2, 1))
        if like is not None:
            a = a.astype(np.asarray(_host(like[nm[k]])[:0]).dtype, copy=False)
        out[nm[k]] = np.ascontiguousarray(a)
    return out


def ensemble_from_letkf(prior, names=None) -> hy.Ensemble:
    """The hybrid's ``ep`` from the LETKF's member stacks: perturbations about their mean, scaled ``1/sqrt(R-1)``."""
    return hy.Ensemble.from_members(to_var_layout(prior, names))


def _mean(a: np.ndarray) -> np.ndarray:
    """Members summed in order, then ``/ R`` (as :meth:`Ensemble.from_members`)."""
    acc = a[0].copy()
    for n in range(1, a.shape[0]):
        acc = acc + a[n]
    return acc / float(a.shape[0])


def variational_state(result) -> dict:
    """The variational analysis in the pairing's fields: ``x_b + xa`` of a :class:`~woof.da.var.analysis.VarResult`."""
    return {k: np.asarray(result.xb[_XA[k]], dtype=np.float64) + np.asarray(result.xa[_XA[k]], dtype=np.float64)
            for k in FIELDS}


def recentre(members: dict, target: dict, *, names=None, gamma: float = 1.0) -> dict:
    """Shift each field's members so their mean moves ``gamma`` of the way to ``target``.

    ``members`` are the LETKF's analysis stacks (``(R, nz, ny, nx)``) and
    ``target`` the variational analysis in the var layout
    (:func:`variational_state`).  Returns the five fields in the LETKF layout;
    the perturbations about the mean are the LETKF's, unchanged, in the members'
    dtype.  Fields of ``members`` that are not among the five are not returned.
    """
    if not (0.0 <= float(gamma) <= 1.0):
        raise ValueError(f"gamma is the share of the shift to the variational mean, in [0, 1], got {gamma}")
    var = to_var_layout(members, names)
    out = {}
    for k in FIELDS:
        a = var[k]
        t = np.asarray(target[k], dtype=np.float64)
        if t.shape != a.shape[1:]:
            raise ValueError(f"target {k} is {t.shape}, the members {a.shape[1:]}")
        out[k] = a + float(gamma) * (t - _mean(a))[None]
    return from_var_layout(out, names, like=members)


@dataclass
class HybridStep:
    """One paired cycle step."""

    #: the variational result (its ``wrf`` is the analysed mean's WRF fields)
    var: object
    #: the LETKF's increments (R, nz, ny, nx) per field
    increments: dict
    #: the recentred analysis members in the five fields, the LETKF's layout
    members: dict
    #: ``spread_prior``/``spread_post``: ensemble standard deviation per field (rms over the grid);
    #: ``shift``: rms of the recentring shift per field; ``negative_q``: members' points with q < 0 after it
    stats: dict = field(default_factory=dict)


def _spread(a: np.ndarray) -> float:
    r = a.shape[0]
    d = a - _mean(a)[None]
    return float(np.sqrt(np.sum(d * d) / (max(r - 1, 1) * d[0].size)))


def hybrid_step(be, obs, first_guess, prior, letkf, *, hybrid, names=None, gamma: float = 1.0,
                analysis_members: dict | None = None, **analyse_kw) -> HybridStep:
    """One hybrid EnKF-Var step: 3DEnVar from the LETKF prior, the LETKF update, then the recentring.

    ``prior`` is the LETKF's forecast ensemble ``{name: (R, nz, ny, nx)}`` at
    the analysis time, ``letkf`` ``(gridded_obs, grid, config)`` for
    :func:`woof.da.letkf.analyze` (its ``analysis_fields`` must include ``u``,
    ``v``, ``t`` and ``q``; the LETKF takes only ``(R, nz, ny, nx)`` fields, so
    ``ps`` is not analysed by it and its members keep the prior's perturbations,
    recentred on the variational ``ps``), ``hybrid`` a
    :class:`~woof.da.var.hybrid.HybridConfig`.
    Recentring can leave negative ``q`` (counted in ``stats["negative_q"]``,
    not clipped: clipping would move the mean off the analysis).  The first
    guess goes through ``wrftoxb`` (hypsometric ``p``, clipped ``q``), the
    members through :func:`~woof.da.var.hybrid.member_fields`; their small
    definitional difference is part of the recentring shift.  Other analysed
    fields come back unchanged.  ``be``, ``obs`` and ``first_guess`` and the
    remaining keywords are :func:`woof.da.var.analysis.analyse`'s.  With ``analysis_members`` the LETKF
    is not run (its result, prior plus increments, is given).
    """
    from woof.da import letkf as lk
    from woof.da.var.analysis import analyse

    if analyse_kw.get("ensemble") is not None:
        raise ValueError("hybrid_step builds the ensemble from the LETKF prior; do not pass ensemble=")
    nm = _names(names)
    if analysis_members is None:
        gobs, grid, cfg = letkf
        missing = [nm[k] for k in FIELDS if k != "ps" and nm[k] not in cfg.analysis_fields]
        if missing:   # before the variational solve
            raise ValueError(f"the LETKF's analysis_fields lack {missing}; the pairing needs u, v, t and q")
    prior_v = to_var_layout(prior, names)
    res = analyse(be, obs, first_guess, ensemble=hy.Ensemble.from_members(prior_v), hybrid=hybrid, **analyse_kw)
    if analysis_members is None:
        inc = lk.analyze(prior, gobs, grid, cfg)
        analysis_members = {nm[k]: _host(prior[nm[k]]) for k in FIELDS}
        analysis_members.update({f: _host(prior[f]) + _host(inc[f]) for f in cfg.analysis_fields})
    else:
        inc = {f: _host(analysis_members[f]) - _host(prior[f]) for f in analysis_members if f in prior}
    mem = recentre(analysis_members, variational_state(res), names=names, gamma=gamma)
    before, after = to_var_layout(analysis_members, names), to_var_layout(mem, names)
    stats = {"spread_prior": {k: _spread(prior_v[k]) for k in FIELDS},
             "spread_post": {k: _spread(after[k]) for k in FIELDS},
             "shift": {k: float(np.sqrt(np.mean((_mean(after[k]) - _mean(before[k])) ** 2))) for k in FIELDS},
             "negative_q": int((after["q"] < 0.0).sum())}
    mem = {**{f: a for f, a in analysis_members.items() if f not in mem}, **mem}
    return HybridStep(var=res, increments=inc, members=mem, stats=stats)
