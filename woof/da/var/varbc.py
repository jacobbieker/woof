"""Variational bias correction of radiances, as WRFDA's VarBC does it (plan V4.4, section 13.6).

Opt-in, with :mod:`woof.da.var.radiance`.  The bias of channel ``c`` at
location ``n`` is ``b = sum_i param_i pred_i(n)`` over the channel's active
predictors; ``param`` joins the control vector through the preconditioning
``param = param_b + vtox v`` and a background term ``Jp`` keeps it near
``param_b``.  The routines and their order are WRFDA 4.7.1's
(``var/da/da_varbc``, public domain; ``licenses/LICENSE-WRF-public-domain.txt``):

1. ``da_varbc_init``: :func:`read_varbc` (``VARBC.in``) or :func:`template`;
2. ``da_varbc_pred`` / ``da_predictor_crtm``: :func:`predictors`;
3. ``da_varbc_coldstart``: the O-B mode for a cold constant, the
   predictors' mean and spread for a cold instrument, then normalisation;
4. ``da_varbc_direct``: the bias, subtracted from O-B before QC
   (:meth:`VarBC.bias` is ``radiance_term``'s ``bias`` hook);
5. ``da_varbc_precond``: per channel the Hessian of Jo + Jp in the
   predictors, its inverse square root ``vtox`` (:meth:`VarBC.precondition`);
6. ``da_varbc_tl``/``da_varbc_adj``: the control block of the radiance term;
7. ``da_calculate_j``/``gradj``'s ``Jp``;
8. ``da_varbc_update``: ``param += vtox v`` and :func:`write_varbc`
   (``VARBC.out``).

Two deliberate departures, both documented in the plan:

- WRFDA's ``grad_jp`` is not the gradient of its ``Jp``.  ``Jp`` here is
  WRFDA's (``jp_form="wrfda"``: ``z_i = v_i sum_j vtox_ij / sqrt(bgerr_i)``)
  and its gradient is the exact one, so the Taylor test holds;
  ``jp_form="matrix"`` is the textbook ``0.5 |B^-1/2 vtox v|^2``.
- ``vtox`` comes from a fixed-order cyclic Jacobi in binary64 using only
  ``+ - * / sqrt``, not LAPACK's ``dsyev``, so it is the same word on every
  machine (the digest gates need that).

All arithmetic is on the host in binary64 with serial sums in WRFDA's
loop order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from woof.da.var.radiance import QC_GOOD, RadianceTerm, family

__all__ = [
    "NPREDMAX",
    "VARBC_IDS",
    "ChannelBC",
    "InstrumentBC",
    "VarBC",
    "VarBCSettings",
    "jacobi_eigh",
    "predictors",
    "read_varbc",
    "template",
    "write_varbc",
]

NPREDMAX = 8
GRAVITY = 9.81
GAS_CONSTANT = 287.0
#: WRFDA's ``qc_varbc_bad``: an observation counts for VarBC when ``qc > -1``.
QC_VARBC_BAD = -1
#: Families whose scan predictors are the satellite zenith (``varbc_scan = 0``).
GEO_FAMILIES = ("abi", "ahi", "seviri", "fci")

#: (platform_id, satellite_id, sensor_id) as WRFDA's ``VARBC.in`` and
#: ``radiance_info`` name them (RTTOV numbering).  Others go through ``ids=``.
VARBC_IDS = {
    "abi_g16": (4, 16, 44), "abi_g17": (4, 17, 44), "abi_g18": (4, 18, 44), "abi_g19": (4, 19, 44),
    "ahi_himawari8": (31, 8, 56), "ahi_himawari9": (31, 9, 56),
    "seviri_m08": (12, 1, 21), "seviri_m09": (12, 2, 21), "seviri_m10": (12, 3, 21), "seviri_m11": (12, 4, 21),
    "amsua_n15": (1, 15, 3), "amsua_n18": (1, 18, 3), "amsua_n19": (1, 19, 3),
    "amsua_metop-a": (10, 2, 3), "amsua_metop-b": (10, 1, 3), "amsua_metop-c": (10, 3, 3),
    "mhs_n19": (1, 19, 15), "mhs_metop-a": (10, 2, 15), "mhs_metop-b": (10, 1, 15), "mhs_metop-c": (10, 3, 15),
    "atms_npp": (17, 0, 19),
}


def _serial_sum(a) -> float:
    """Fortran ``SUM``: left to right from 0."""
    acc = 0.0
    for x in np.ravel(a).tolist():
        acc = acc + x
    return acc


# --------------------------------------------------------------------------- the coefficients

@dataclass
class ChannelBC:
    """One channel's line of ``VARBC.in``: its predictors in use and their coefficients.

    ``pred_use`` (``npredmax``): -1 unused, 0 used and cold (coefficient 0
    until estimated), 1 used with ``param`` from the file.  ``param`` holds
    one value per used predictor (``pred_use >= 0``), in predictor order.
    """

    chanl_id: int
    channel: int
    pred_use: np.ndarray
    param: np.ndarray
    # set by VarBC.precondition
    vtox: np.ndarray | None = None
    bgerr: np.ndarray | None = None
    nobs: int = 0

    def __post_init__(self):
        self.pred_use = np.asarray(self.pred_use, dtype=np.int64)
        self.param = np.asarray(self.param, dtype=np.float64)
        if self.param.size != self.npred:
            raise ValueError(f"channel {self.channel}: {self.param.size} params for {self.npred} predictors")

    @property
    def ipred(self) -> np.ndarray:
        """The used predictors, 0-based (WRFDA's ``ipred`` minus one)."""
        return np.flatnonzero(self.pred_use >= 0)

    @property
    def npred(self) -> int:
        return int(np.count_nonzero(self.pred_use >= 0))


@dataclass
class InstrumentBC:
    """One instrument's block of ``VARBC.in``."""

    ids: tuple
    nchanl: int
    npredmax: int
    pred_mean: np.ndarray
    pred_std: np.ndarray
    nbgerr: np.ndarray
    channels: list = field(default_factory=list)

    def channel(self, ch: int) -> ChannelBC | None:
        for c in self.channels:
            if c.channel == int(ch):
                return c
        return None


def template(sensor: str, channels: Sequence[int], *, ids: tuple | None = None, nbgerr: int = 5000,
             chanl_ids: Sequence[int] | None = None) -> InstrumentBC:
    """A cold instrument as WRFDA's geostationary ``VARBC.in`` entries set it up.

    Predictors 1-5 (constant, the two thicknesses, skin temperature, TPW)
    used and cold; 6-8 (zenith or scan position powers) unused.  For a
    microwave sounder whose scan bias matters, pass a ``pred_use`` of your
    own through :class:`ChannelBC` instead.
    """
    ids = ids or VARBC_IDS.get(sensor)
    if ids is None:
        raise KeyError(f"no VarBC ids for {sensor!r}; pass ids=(platform, satellite, sensor)")
    use = np.array([0, 0, 0, 0, 0, -1, -1, -1])
    chanl_ids = list(chanl_ids) if chanl_ids is not None else list(range(1, len(channels) + 1))
    return InstrumentBC(ids=tuple(int(i) for i in ids), nchanl=len(channels), npredmax=NPREDMAX,
                        pred_mean=np.zeros(NPREDMAX), pred_std=np.zeros(NPREDMAX),
                        nbgerr=np.full(NPREDMAX, nbgerr, dtype=np.int64),
                        channels=[ChannelBC(int(k), int(c), use.copy(), np.zeros(5))
                                  for k, c in zip(chanl_ids, channels)])


def read_varbc(path) -> dict:
    """``da_varbc_init``'s reading of ``VARBC.in``: ``{(platform, satellite, sensor): InstrumentBC}``."""
    lines = Path(path).read_text().splitlines()
    pos = 1
    nsensor = int(lines[pos].split()[0])
    pos += 1
    out = {}
    for _ in range(nsensor):
        pos += 3
        head = lines[pos].split()
        pos += 1
        ids = tuple(int(v) for v in head[:3])
        nchanl, npredmax = int(head[3]), int(head[4])
        pos += 1
        mean = np.array(lines[pos].split()[:npredmax], dtype=np.float64)
        std = np.array(lines[pos + 1].split()[:npredmax], dtype=np.float64)
        nbg = np.array(lines[pos + 2].split()[:npredmax], dtype=np.int64)
        pos += 4
        inst = InstrumentBC(ids=ids, nchanl=nchanl, npredmax=npredmax, pred_mean=mean, pred_std=std, nbgerr=nbg)
        for _ in range(nchanl):
            f = lines[pos].split()
            pos += 1
            use = np.array(f[2:2 + npredmax], dtype=np.int64)
            npred = int(np.count_nonzero(use >= 0))
            if npred <= 0:
                continue
            given = np.array(f[2 + npredmax:], dtype=np.float64)
            param = np.zeros(npred)
            # params are read for pred_use > 0 only, in predictor order
            warm = np.flatnonzero(use[use >= 0] > 0)
            param[warm] = given[:warm.size]
            inst.channels.append(ChannelBC(int(f[0]), int(f[1]), use, param))
        out[ids] = inst
    return out


def write_varbc(path, instruments) -> None:
    """``da_varbc_update``'s ``VARBC.out``: WRFDA's layout and formats."""
    insts = list(instruments.values()) if isinstance(instruments, Mapping) else list(instruments)
    out = [" VARBC version 1.0 - Number of instruments: ", f" {len(insts):11d}"]
    rule = " ------------------------------------------------"
    for inst in insts:
        out += [rule, " Platform_id  Sat_id  Sensor_id  Nchanl  Npredmax", rule,
                " " + "".join(f"{v:12d}" for v in (*inst.ids, inst.nchanl, inst.npredmax)),
                " -----> Bias predictor statistics:  Mean & Std & Nbgerr",
                "".join(f"{v:10.1f}" for v in inst.pred_mean), "".join(f"{v:10.1f}" for v in inst.pred_std),
                "".join(f"{int(v):10d}" for v in inst.nbgerr), " -----> Chanl_id Chanl_nb  Pred_use(-1/0/1)  Param"]
        for c in inst.channels:
            if c.npred <= 0:
                continue
            out.append(f"{c.chanl_id:4d}{c.channel:6d}" + "".join(f"{int(u):3d}" for u in c.pred_use)
                       + "".join(_f8_3(p) for p in c.param))
    Path(path).write_text("\n".join(out) + "\n")


def _f8_3(x: float) -> str:
    s = f"{x:8.3f}"
    return s if len(s) == 8 else "*" * 8


# --------------------------------------------------------------------------- predictors

def predictor_crtm(temp, hum, t_skin, pf) -> np.ndarray:
    """``da_predictor_crtm`` for one column: the 1000-300 and 200-50 hPa thicknesses, ``t_skin``, TPW.

    ``temp`` and ``hum`` (mixing ratio, g/kg) are CRTM's layers, top first;
    ``pf`` the ``nlev + 1`` interfaces (hPa), top first.
    """
    kth = GAS_CONSTANT / GRAVITY
    kpc = 100.0 / GRAVITY
    nl = temp.size
    qm = hum * 0.001
    dlp = np.log(pf[1:]) - np.log(pf[:-1])
    dp = pf[1:] - pf[:-1]
    i1000, i300, i200, i50 = (int(np.argmin(np.abs(pf - v))) for v in (1000.0, 300.0, 200.0, 50.0))
    tv = temp * (1.0 + 0.608 * qm)
    add_thk = kth * (tv[nl - 1] * (math.log(1000.0) - math.log(pf[nl]))) if 1000.0 > pf[nl] else 0.0
    # tv(index300+1:index1000) are layers index300..index1000-1 here (0-based)
    p1 = kth * _serial_sum(tv[i300:i1000] * dlp[i300:i1000]) + add_thk
    p2 = kth * _serial_sum(tv[i50:i200] * dlp[i50:i200])
    p4 = kpc * _serial_sum(qm * dp)
    return np.array([p1, p2, float(t_skin), p4])


def predictors(batch, npredmax: int = NPREDMAX) -> np.ndarray:
    """``da_varbc_pred``: the raw ``(n, npredmax)`` predictors of a :class:`~woof.da.var.radiance.RadianceBatch`."""
    cols, rec = batch.extra["columns"], batch.extra["record"]
    n = batch.n
    pred = np.zeros((n, npredmax))
    pred[:, 0] = 1.0
    for k in range(n):
        hk = predictor_crtm(cols.temperature_k[k], cols.mixing_ratio_gkg[k], cols.skin_temperature_k[k],
                            cols.level_pressure_hpa[k])
        m = min(npredmax - 1, 4)
        pred[k, 1:1 + m] = hk[:m]
    scan = rec.sensor_zenith if family(batch.sensor) in GEO_FAMILIES else rec.scanpos
    for p in range(5, min(npredmax, 8)):
        pred[:, p] = scan ** (p - 4)
    return pred


# --------------------------------------------------------------------------- Jacobi

def jacobi_eigh(a, max_sweeps: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and eigenvectors (columns) of a small symmetric matrix, bit-reproducibly.

    Cyclic Jacobi, rows then columns, with only IEEE ``+ - * / sqrt``; the
    result is the same word on any machine.  Stops after a sweep with no
    rotation (every off-diagonal exactly 0 or negligible against its
    diagonal pair).
    """
    a = np.array(a, dtype=np.float64)
    n = a.shape[0]
    v = np.eye(n)
    for _ in range(max_sweeps):
        rotated = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = float(a[p, q])
                if apq == 0.0:
                    continue
                app, aqq = float(a[p, p]), float(a[q, q])
                if abs(apq) <= 1e-18 * math.sqrt(abs(app * aqq)):
                    a[p, q] = a[q, p] = 0.0
                    continue
                rotated = True
                theta = (aqq - app) / (2.0 * apq)
                if abs(theta) > 1e150:
                    t = 0.5 / theta
                else:
                    t = 1.0 / (abs(theta) + math.sqrt(theta * theta + 1.0))
                    t = t if theta >= 0.0 else -t
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = float(a[k, p]), float(a[k, q])
                    a[k, p] = c * akp - s * akq
                    a[k, q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = float(a[p, k]), float(a[q, k])
                    a[p, k] = c * apk - s * aqk
                    a[q, k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = float(v[k, p]), float(v[k, q])
                    v[k, p] = c * vkp - s * vkq
                    v[k, q] = s * vkp + c * vkq
        if not rotated:
            break
    return np.diag(a).copy(), v


# --------------------------------------------------------------------------- VarBC

@dataclass(frozen=True)
class VarBCSettings:
    """WRFDA's ``&wrfvar4`` VarBC switches."""

    varbc_factor: float = 1.0
    varbc_nbgerr: int = 5000
    varbc_nobsmin: int = 10
    #: ``"wrfda"`` (WRFDA's Jp, exact gradient) or ``"matrix"`` (see the module notes).
    jp_form: str = "wrfda"


class VarBC:
    """The coefficients of every VarBC instrument and their control block.

    ``instruments`` maps CRTM sensor names to :class:`InstrumentBC`.  A
    sensor without an entry is not bias-corrected (``npred = 0`` in WRFDA).
    Use :meth:`bias` as :func:`~woof.da.var.radiance.radiance_term`'s
    ``bias`` hook, then :meth:`attach` the returned term.
    """

    def __init__(self, instruments: Mapping, settings: VarBCSettings | None = None):
        self.instruments = dict(instruments)
        self.settings = settings or VarBCSettings()
        if self.settings.jp_form not in ("wrfda", "matrix"):
            raise ValueError(f"jp_form must be 'wrfda' or 'matrix', not {self.settings.jp_form!r}")
        self._pred = {}

    @classmethod
    def from_file(cls, path, sensors: Sequence[str], ids: Mapping | None = None, **kw) -> "VarBC":
        """The instruments of ``VARBC.in`` that ``sensors`` name (by ``VARBC_IDS`` or ``ids``)."""
        table = read_varbc(path)
        ids = {**VARBC_IDS, **(ids or {})}
        return cls({s: table[ids[s]] for s in sensors if s in ids and ids[s] in table}, **kw)

    # -------------------------------------------------------------- innovations

    def bias(self, batch, columns=None) -> np.ndarray:
        """Predictors, cold start and ``da_varbc_direct``'s bias ``(n, nchan)`` for one batch (zero when not VarBC)."""
        inst = self.instruments.get(batch.sensor)
        out = np.zeros(batch.obs.shape)
        if inst is None:
            return out
        pred = predictors(batch, inst.npredmax)
        self._coldstart(inst, batch, pred)
        pred = self._normalise(inst, pred)
        self._pred[id(batch)] = pred
        for c, ch in enumerate(batch.channels.tolist()):
            bc = inst.channel(ch)
            if bc is None or bc.npred <= 0:
                continue
            ip = bc.ipred
            for k in range(batch.n):
                acc = 0.0
                for i in range(bc.npred):
                    acc = acc + bc.param[i] * pred[k, ip[i]]
                out[k, c] = acc
        return out

    def _coldstart(self, inst: InstrumentBC, batch, pred) -> None:
        nbins, maxhist = 200, 10.0
        zbin = 2.0 * maxhist / nbins
        for c, ch in enumerate(batch.channels.tolist()):
            bc = inst.channel(ch)
            if bc is None or bc.npred <= 0 or np.all(bc.pred_use != 0):
                continue
            bc.param = np.where(bc.pred_use[bc.ipred] == 0, 0.0, bc.param)
            if bc.pred_use[0] == 0:
                hist = np.zeros(nbins, dtype=np.int64)
                for x in batch.inv[:, c].tolist():
                    if not math.isfinite(x):
                        continue
                    ibin = _nint((x + maxhist) / zbin)
                    if 0 < ibin <= nbins:
                        hist[ibin - 1] += 1
                mode = (int(np.argmax(hist)) + 1) * zbin - maxhist if hist.sum() > 0 else 0.0
                if bc.ipred[0] == 0:
                    bc.param[0] = mode
        if not any(np.any(bc.pred_use > 0) for bc in inst.channels if bc.npred > 0):
            n = pred.shape[0]
            if n >= self.settings.varbc_nobsmin:
                mean = np.array([_serial_sum(pred[:, i]) for i in range(inst.npredmax)]) / n
                rms = np.array([_serial_sum(pred[:, i] ** 2) for i in range(inst.npredmax)]) / n
            else:
                mean, rms = np.zeros(inst.npredmax), np.ones(inst.npredmax)
            inst.pred_mean = mean
            # WRFDA: sqrt(rms - mean^2); a rounding below zero is a zero spread here
            inst.pred_std = np.sqrt(np.maximum(rms - mean * mean, 0.0))

    @staticmethod
    def _normalise(inst, pred):
        pred = pred.copy()
        for i in range(inst.npredmax):
            if inst.pred_std[i] <= 0.0:
                continue
            pred[:, i] = (pred[:, i] - inst.pred_mean[i]) / inst.pred_std[i]
        return pred

    # -------------------------------------------------------------- preconditioning

    def attach(self, term: RadianceTerm) -> RadianceTerm:
        """``da_varbc_precond`` over the term's QC'd batches, then give the term its control block."""
        s = self.settings
        layout = []
        off = 0
        for kb, b in enumerate(term.batches):
            inst = self.instruments.get(b.sensor)
            if inst is None:
                continue
            pred = self._pred[id(b)]
            for c, ch in enumerate(b.channels.tolist()):
                bc = inst.channel(ch)
                if bc is None or bc.npred <= 0:
                    continue
                ip, npred = bc.ipred, bc.npred
                good = np.flatnonzero(b.qc[:, c] > QC_VARBC_BAD)
                bc.nobs = int(good.size)
                bc.vtox = np.eye(npred)
                bc.bgerr = np.zeros(npred)
                if bc.nobs > 0:
                    err2 = b.error[good, c] ** 2
                    h = np.zeros((npred, npred))
                    for i in range(npred):
                        for j in range(i, npred):
                            h[i, j] = _serial_sum(pred[good, ip[i]] * pred[good, ip[j]] / err2)
                            h[j, i] = h[i, j]
                        if inst.nbgerr[ip[i]] <= 0:
                            continue
                        bc.bgerr[i] = _serial_sum(err2 / s.varbc_nbgerr) / bc.nobs
                        if bc.bgerr[i] > 0:
                            h[i, i] = h[i, i] + 1.0 / bc.bgerr[i]
                    h = h / s.varbc_factor ** 2
                    if npred == 1:
                        bc.vtox = np.array([[1.0 / math.sqrt(h[0, 0])]])
                    else:
                        lam, e = jacobi_eigh(h)
                        if np.any(lam <= 0):
                            bc.vtox = np.zeros((npred, npred))
                            for i in range(npred):
                                if h[i, i] > 0:
                                    bc.vtox[i, i] = 1.0 / math.sqrt(h[i, i])
                        else:
                            r = np.sqrt(1.0 / lam)
                            for i in range(npred):
                                for j in range(i, npred):
                                    bc.vtox[i, j] = _serial_sum(e[i, :] * r * e[j, :])
                                    bc.vtox[j, i] = bc.vtox[i, j]
                layout.append({"batch": kb, "channel": c, "bc": bc, "pred": pred[:, ip], "offset": off})
                off += npred
        term.varbc = _Control(layout, off, s.jp_form, [(b.qc > QC_VARBC_BAD) for b in term.batches],
                              term.offsets)
        return term

    def update(self, term: RadianceTerm, v) -> None:
        """``da_varbc_update``: ``param += vtox v`` for the block ``v``; cold predictors with enough data become warm."""
        v = np.asarray(v, dtype=np.float64)
        for entry in term.varbc.layout:
            bc = entry["bc"]
            if bc.nobs >= self.settings.varbc_nobsmin:
                bc.pred_use = np.where(bc.pred_use == 0, 1, bc.pred_use)
            seg = v[entry["offset"]: entry["offset"] + bc.npred]
            delta = np.array([_serial_sum(seg * bc.vtox[i, :]) for i in range(bc.npred)])
            bc.param = bc.param + delta

    def write(self, path, sensors: Sequence[str] | None = None) -> None:
        write_varbc(path, [self.instruments[s] for s in (sensors or self.instruments)])


def _nint(x: float) -> int:
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


class _Control:
    """The VarBC block of a :class:`RadianceTerm`: TL, AD, Jp and its gradient."""

    def __init__(self, layout, size, jp_form, good, offsets):
        self.layout = layout
        self.size = int(size)
        self.jp_form = jp_form
        self.good = good
        self.offsets = offsets

    def _param(self, entry, v):
        bc = entry["bc"]
        seg = v[entry["offset"]: entry["offset"] + bc.npred]
        return np.array([_serial_sum(seg * bc.vtox[i, :]) for i in range(bc.npred)])

    def tl_add(self, y_batches, v) -> None:
        """``da_varbc_tl``: add ``sum_i param_tl_i pred_i`` to each good observation's ``y``."""
        v = np.asarray(v, dtype=np.float64)
        for entry in self.layout:
            p = self._param(entry, v)
            yb = y_batches[entry["batch"]]
            c, pred = entry["channel"], entry["pred"]
            good = self.good[entry["batch"]][:, c]
            acc = np.zeros(pred.shape[0])
            for i in range(p.size):
                acc = acc + p[i] * pred[:, i]
            yb[:, c] = np.where(good, yb[:, c] + acc.astype(yb.dtype), yb[:, c])

    def ad(self, y_batches) -> np.ndarray:
        """``da_varbc_adj``: the block's adjoint, pixels summed in order."""
        out = np.zeros(self.size)
        for entry in self.layout:
            bc = entry["bc"]
            c, pred = entry["channel"], entry["pred"]
            good = self.good[entry["batch"]][:, c]
            yc = np.where(good, np.asarray(y_batches[entry["batch"]][:, c], dtype=np.float64), 0.0)
            pa = np.array([_serial_sum(yc[good] * pred[good, i]) for i in range(bc.npred)])
            for i in range(bc.npred):
                o = entry["offset"] + i
                out[o] = out[o] + _serial_sum(pa * bc.vtox[i, :])
        return out

    def _z(self, entry, v):
        bc = entry["bc"]
        seg = v[entry["offset"]: entry["offset"] + bc.npred]
        z = np.zeros(bc.npred)
        coef = np.zeros(bc.npred)
        for i in range(bc.npred):
            if bc.bgerr[i] > 0.0:
                coef[i] = 1.0 / math.sqrt(bc.bgerr[i])
        if self.jp_form == "wrfda":
            rows = np.array([_serial_sum(bc.vtox[i, :]) for i in range(bc.npred)])
            z = coef * seg * rows
            return z, coef * rows
        for i in range(bc.npred):
            z[i] = coef[i] * _serial_sum(seg * bc.vtox[i, :])
        return z, coef

    def jp(self, v) -> float:
        v = np.asarray(v, dtype=np.float64)
        acc = 0.0
        for entry in self.layout:
            z, _ = self._z(entry, v)
            acc = acc + _serial_sum(z * z)
        return 0.5 * acc

    def jp_grad(self, v) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        g = np.zeros(self.size)
        for entry in self.layout:
            bc = entry["bc"]
            z, w = self._z(entry, v)
            o = entry["offset"]
            if self.jp_form == "wrfda":
                g[o:o + bc.npred] = w * z
            else:
                wz = w * z
                for j in range(bc.npred):
                    g[o + j] = _serial_sum(bc.vtox[:, j] * wz)
        return g
