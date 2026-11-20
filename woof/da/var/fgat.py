"""WRFDA's FGAT (first guess at appropriate time) for the 3D-Var port (EXPERIMENTAL).

WRFDA 4.7.1 with ``num_fgat_time > 1`` and ``var4d = .false.``:

* the window ``[time_window_min, time_window_max)`` is cut into
  ``num_fgat_time`` slots by ``da_get_time_slots`` (:func:`time_slots`);
  first guess ``fgNN`` belongs to slot ``NN`` (``fg01`` at the window's
  start, the last one at its end, evenly spaced between);
* the conventional ASCII observations come one file per slot
  (``ob01.ascii`` ...), read in slot order into one ``iv`` whose per-type
  ``plocal(0:num_fgat_time)`` marks the slot boundaries
  (``da_setup_obs_structures_ascii``; :func:`read_obs_ascii_fgat`).  Each
  file is filtered by the whole window, not by its slot;
* ``da_get_innov_vector`` reads ``fgNN`` for each slot (last to first) and
  computes that slot's innovations, QC and vertical positions against it
  (:func:`innovations_fgat`), then restores ``fg`` (the analysis time);
* the minimisation is 3D-Var's: ``H'`` is linearised about ``fg`` and
  applied to the single 3-D increment for every slot
  (``da_transform_vtoy``'s non-``var4d`` branch), and the analysis is
  ``fg`` plus that increment.

So FGAT changes only the innovation vector: the observations' positions
(``zk``) and ``O - H(x_b)`` come from the first guess at their slot's time,
everything after is the 3D-Var of :mod:`woof.da.var.analysis`.

Not ported: ``max_ext_its > 1`` with FGAT (WRFDA 4.7.1 re-reads ``fgNN``
and ``fg`` from disk in every outer loop, which discards the earlier
loops' increments from the linearisation and the output; refused, see
:func:`check_fgat`), radiances, rain and PREPBUFR observations (their
slots come from ``time_slots`` by observation time).

Derived from WRFDA 4.7.1 (``var/da/da_tools/da_get_time_slots.inc``,
``da_setup_structures/da_setup_obs_structures_ascii.inc``,
``da_obs_io/da_read_obs_ascii.inc``,
``da_minimisation/da_get_innov_vector.inc``, ``da_transform_vtoy.inc``,
``da_read_basicstates.inc``), public domain; see
``licenses/LICENSE-WRF-public-domain.txt``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .obs_wrfda import Field, ObsType, WrfdaObs, _date_minutes, read_obs_ascii

__all__ = ["time_slots", "slot_minutes", "read_obs_ascii_fgat", "slot", "concat", "innovations_fgat",
           "check_fgat"]

#: ``extra`` arrays per report and per level (anything else is refused rather than guessed).
_EXTRA_PER_REPORT = ("ref_p",)
_EXTRA_PER_LEVEL = ("zk",)


def time_slots(nt: int, tmin: str, tana: str, tmax: str) -> tuple[np.ndarray, int | None]:
    """``da_get_time_slots``: the slot edges ``time_slots(0:nt)`` in WRFDA's julian
    minutes and ``itime_ana``, the first guess index (1-based) at the analysis time.

    The edges are half a step after ``tmin``, then a step apart, so the first
    and last slots are half as wide as the others.  ``itime_ana`` is None
    where WRFDA leaves it unset (the analysis time is not a first guess time).
    """
    if nt < 1:
        raise ValueError(f"num_fgat_time must be >= 1, not {nt}")
    ts = np.zeros(nt + 1, dtype=np.float64)
    t_ana = _date_minutes(tana)
    ts[0] = _date_minutes(tmin)
    ts[nt] = _date_minutes(tmax)
    dt = 0.0
    if nt > 1:
        dt = (ts[nt] - ts[0]) / float(nt - 1)
        ts[1] = ts[0] + dt * 0.5
        for it in range(2, nt):
            ts[it] = ts[it - 1] + dt
    itime_ana = None
    if nt == 1 or abs(t_ana - ts[0]) < 0.1:
        itime_ana = 1
    elif abs(t_ana - ts[nt]) < 0.1:
        itime_ana = nt
    else:
        for it in range(2, nt):
            if abs(t_ana - (ts[0] + (it - 1) * dt)) < 0.1:
                itime_ana = it
                break
    return ts, itime_ana


def slot_minutes(nt: int, tmin: str, tmax: str) -> np.ndarray:
    """The first guesses' times (julian minutes): ``fg01`` at ``tmin``, ``fgNN`` at
    ``tmin + (NN-1) dt``, the last at ``tmax`` (``da_get_time_slots``' comment)."""
    t0, t1 = _date_minutes(tmin), _date_minutes(tmax)
    if nt == 1:
        return np.array([t0])
    dt = (t1 - t0) / float(nt - 1)
    return np.array([t0 + n * dt for n in range(nt)])


# --------------------------------------------------------------------------- slots of a WrfdaObs
def _take(t: ObsType, r0: int, r1: int) -> ObsType:
    """Reports ``r0:r1`` of ``t`` (and their levels)."""
    off = t.offsets
    l0, l1 = int(off[r0]), int(off[r1])
    info = {k: v[r0:r1].copy() for k, v in t.info.items()}
    coords = {k: v[l0:l1].copy() for k, v in t.coords.items()}
    fields = {k: Field(f.inv[l0:l1].copy(), f.qc[l0:l1].copy(), f.error[l0:l1].copy()) for k, f in t.fields.items()}
    extra = {}
    for k, v in t.extra.items():
        if k == "geometry":
            continue
        if k in _EXTRA_PER_REPORT:
            extra[k] = v[r0:r1].copy()
        elif k in _EXTRA_PER_LEVEL:
            extra[k] = v[l0:l1].copy()
        else:
            raise ValueError(f"{t.name}: extra[{k!r}] is neither per report nor per level")
    return ObsType(t.name, info, coords, fields, extra)


def slot(obs: WrfdaObs, n: int) -> WrfdaObs:
    """Slot ``n`` (0-based) of an FGAT ``WrfdaObs``: WRFDA's ``n1 = plocal(n-1)+1 .. n2 = plocal(n)``."""
    if obs.plocal is None:
        raise ValueError("not an FGAT observation set (plocal is None)")
    types = {}
    for name, t in obs.types.items():
        p = obs.plocal[name]
        types[name] = _take(t, int(p[n]), int(p[n + 1]))
    return WrfdaObs(types=types, ntotal=dict(obs.ntotal), analysis_minutes=obs.analysis_minutes)


def _cat(arrays: list) -> np.ndarray:
    return np.concatenate(arrays) if arrays else np.zeros(0)


def concat(parts: Sequence[WrfdaObs], *, kte: int | None = None) -> WrfdaObs:
    """The slots' observation sets as one, in slot order, with ``plocal`` set.

    ``ntotal`` is the last part's (the reader carries it across slots).
    With ``kte`` each type's ``Geometry`` is rebuilt from the joined ``zk``
    (innovation output); without, ``geometry`` is dropped.
    """
    if not parts:
        raise ValueError("no time slots")
    names = list(parts[0].types)
    types, plocal = {}, {}
    for name in names:
        ts = [p.types[name] for p in parts]
        counts = [t.nlocal for t in ts]
        plocal[name] = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        info = {k: np.concatenate([t.info[k] for t in ts]) for k in ts[0].info}
        coords = {k: _cat([t.coords[k] for t in ts]) for k in ts[0].coords}
        fields = {k: Field(_cat([t.fields[k].inv for t in ts]), _cat([t.fields[k].qc for t in ts]).astype(np.int64),
                           _cat([t.fields[k].error for t in ts])) for k in ts[0].fields}
        extra = {}
        keys = sorted({k for t in ts for k in t.extra if k != "geometry"})
        for k in keys:
            if k not in _EXTRA_PER_REPORT + _EXTRA_PER_LEVEL:
                raise ValueError(f"{name}: extra[{k!r}] is neither per report nor per level")
            have = []
            for t in ts:
                if k in t.extra:
                    have.append(t.extra[k])
                elif t.nlocal:
                    raise ValueError(f"{name}: extra[{k!r}] is missing from a slot with {t.nlocal} reports")
            extra[k] = _cat(have)
        t = ObsType(name, info, coords, fields, extra)
        if kte is not None and "zk" in extra:
            from .innov_wrfda import Geometry

            t.extra["geometry"] = Geometry.of(t, kte)
        types[name] = t
    return WrfdaObs(types=types, ntotal=dict(parts[-1].ntotal), analysis_minutes=parts[0].analysis_minutes,
                    plocal=plocal)


# --------------------------------------------------------------------------- reading and innovations
def read_obs_ascii_fgat(paths: Sequence, proj, *, analysis_date: str, time_window_min: str,
                        time_window_max: str, **kw) -> WrfdaObs:
    """``ob01.ascii`` .. ``obNN.ascii`` as WRFDA's FGAT reads them: one file per
    slot, in slot order, each filtered by the whole window, ``ntotal`` (hence
    ``obs_global_index``) carried across files.  ``kw`` goes to
    :func:`~woof.da.var.obs_wrfda.read_obs_ascii`."""
    if len(paths) < 2:
        raise ValueError("FGAT needs at least two time slots (ob01.ascii, ob02.ascii, ...)")
    parts, ntotal = [], None
    for path in paths:
        o = read_obs_ascii(path, proj, analysis_date=analysis_date, time_window_min=time_window_min,
                           time_window_max=time_window_max, ntotal0=ntotal, **kw)
        ntotal = o.ntotal
        parts.append(o)
    return concat(parts)


def innovations_fgat(obs: WrfdaObs, xbs: Sequence[dict], **kw) -> WrfdaObs:
    """``da_get_innov_vector`` over the slots: slot ``n``'s innovations, QC and
    ``zk`` against ``xbs[n]`` (``fgNN``'s background), joined in slot order.
    ``kw`` goes to :func:`~woof.da.var.innov_wrfda.innovations`."""
    from . import innov_wrfda as iw

    if obs.plocal is None:
        raise ValueError("not an FGAT observation set (plocal is None); use innov_wrfda.innovations")
    nt = len(next(iter(obs.plocal.values()))) - 1
    if len(xbs) != nt:
        raise ValueError(f"{nt} observation slots but {len(xbs)} first guesses")
    kte = {np.asarray(xb["p"]).shape[2] for xb in xbs}
    if len(kte) != 1:
        raise ValueError(f"the first guesses have different numbers of levels: {sorted(kte)}")
    # WRFDA runs the slots last to first; no state crosses slots, so the order is immaterial.
    parts = [iw.innovations(slot(obs, n), xbs[n], **kw) for n in range(nt)]
    out = concat(parts, kte=kte.pop())
    out.ntotal = dict(obs.ntotal)
    return out


def check_fgat(n_obs: int, first_guess_times: Sequence[str], *, time_window_min: str, time_window_max: str,
               max_ext_its: int = 1) -> None:
    """Refuse an FGAT set-up WRFDA would run wrongly or that this port does not cover.

    One observation file and one first guess per slot, at least two slots,
    each first guess at its slot's time (WRFDA reads ``fgNN`` without
    checking its date), and one outer loop.
    """
    nt = len(first_guess_times)
    if nt < 2:
        raise ValueError("FGAT needs at least two time slots")
    if n_obs != nt:
        raise ValueError(f"FGAT: {n_obs} observation files for {nt} first guesses (one of each per slot)")
    want = slot_minutes(nt, time_window_min, time_window_max)
    for n, (t, w) in enumerate(zip(first_guess_times, want), start=1):
        if abs(_date_minutes(t) - w) >= 0.1:
            raise ValueError(f"FGAT slot {n}: the first guess is at {t[:19]}, the slot at "
                             f"{w - _date_minutes(time_window_min):+.1f} min from {time_window_min[:19]} "
                             f"(fg01 at the window's start, the last at its end, evenly spaced)")
    if max_ext_its != 1:
        raise NotImplementedError(
            "FGAT with max_ext_its > 1 is not ported: WRFDA 4.7.1 re-reads fgNN and fg in every outer loop, "
            "which drops the earlier loops' increments")
