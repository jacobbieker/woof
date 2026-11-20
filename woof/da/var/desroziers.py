"""Desroziers-type consistency diagnostics of the 3-D-Var analysis (plan gate G4a).

Two adjoint-free diagnostics, both in WRFDA's own conventions:

- :func:`classic` — the O-B / O-A / A-B statistics of Desroziers et al. (2005)
  per observation type and variable: ``<(O-A)(O-B)>`` estimates ``R`` (compared with
  the specified error variance ``sigma_o^2``), ``<(A-B)(O-B)>`` estimates ``H B H^T``,
  and the ratios are the consistency factors.  ``O-B = d``, ``A-B = H'(xhat)``,
  ``O-A = d - H'(xhat)``, over the observations with ``qc >= 0``.
- :func:`randomised` — what WRFDA's ``da_tune_obs_desroziers`` does (Chapnik et al. 2004;
  Desroziers and Ivanov 2001): a second analysis with Gaussian noise ``sigma z`` added to
  the observations (``omb_add_noise``; the noise is read from WRFDA's ``rand_obs_error``
  so both systems use the same draw), the trace of the influence matrix per variable
  ``Tr(HK) = sum (y_p - y) z / sigma``, the expected ``Jo = (n - Tr(HK)) / 2``, the actual
  ``Jo`` at the unperturbed analysis and the error factor ``sqrt(Jo_actual / Jo_expected)``;
  the totals ``Jo``, ``Jb`` as the tool prints them.

:func:`format_report` prints the tool's table.  The reference tier is the oracle's:
the unperturbed and perturbed minimisations are :func:`woof.da.var.minimise.minimise`
runs of the same cost with only the innovations changed.

Derived from WRFDA 4.7.1 (``var/da/da_tools/da_add_noise.inc``,
``var/da/da_obs/da_add_noise_to_ob.inc``, ``var/build/da_tune_obs_desroziers.f90``), public
domain; see ``licenses/LICENSE-WRF-public-domain.txt``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["FILE_FIELDS", "TOOL_NAMES", "classic", "randomised", "format_report", "read_rand_obs_error",
           "read_obs_space_file", "DesroziersResult"]

#: The variables of each type in WRFDA's ``rand_obs_error`` / ``unpert_obs`` records, in file order.
FILE_FIELDS = {
    "synop": ("u", "v", "t", "p", "q"), "metar": ("u", "v", "t", "p", "q"), "ships": ("u", "v", "t", "p", "q"),
    "buoy": ("u", "v", "t", "p", "q"), "sonde_sfc": ("u", "v", "t", "p", "q"), "geoamv": ("u", "v"),
    "polaramv": ("u", "v"), "gpspw": ("tpw",), "sound": ("u", "v", "t", "q"), "airep": ("u", "v", "t", "q"),
    "pilot": ("u", "v"), "satem": ("thickness",), "qscat": ("u", "v"), "profiler": ("u", "v"),
    "gpsref": ("ref",),
}
#: The tool's print order and names (first five characters of the type, as ``da_calc_new_factors1``).
TOOL_NAMES = (("synop", "synop"), ("metar", "metar"), ("ships", "ships"), ("geoamv", "geoam"),
              ("polaramv", "polar"), ("gpspw", "gpspw"),
              ("sound", "sound"), ("airep", "airep"), ("pilot", "pilot"), ("satem", "satem"), ("buoy", "buoy "),
              ("sonde_sfc", "sonde"), ("profiler", "profi"), ("qscat", "qscat"), ("gpsref", "gpsre"))
_TOOL_FIELDS = {"synop": 5, "metar": 5, "ships": 5, "buoy": 5, "sonde_sfc": 5, "geoamv": 2, "polaramv": 2, "gpspw": 1,
                "sound": 4, "airep": 3, "pilot": 2, "satem": 1, "qscat": 2, "profiler": 2, "gpsref": 1}
_MISSING = -888888.0


def _in_table(n: str, v: str) -> bool:
    """The tool tabulates (and totals) the first ``_TOOL_FIELDS[n]`` variables of its types: airep has no ``q``."""
    return n in _TOOL_FIELDS and v in FILE_FIELDS[n][: _TOOL_FIELDS[n]]


def _blocks(path):
    """``(type, [[record per level] per report])``: each record the line's numbers after the 16-character prefix."""
    lines = Path(path).read_text().split("\n")
    i = 0
    while i < len(lines):
        head = lines[i]
        if len(head) < 28 or not head[20:28].strip().isdigit():
            i += 1
            continue
        name, n = head[:20].strip(), int(head[20:28])
        i += 1
        reports = []
        for _ in range(n):
            levels = int(lines[i].split()[0])
            i += 1
            reports.append([lines[i + k] for k in range(levels)])
            i += levels
        yield name, reports


def _numbers(line, width):
    body = line[16:]
    return [float(body[k:k + width]) for k in range(0, len(body.rstrip("\n")) - width + 1, width)]


def read_rand_obs_error(path) -> dict:
    """WRFDA's ``rand_obs_error``: ``{type: (error, z)}``, each ``(nreports_levels, nfields)`` with report-by-level rows.

    Missing entries (``qc < 0``, written as -888888) are NaN in ``z``.
    """
    out = {}
    for name, reports in _blocks(path):
        if name not in FILE_FIELDS:
            continue
        nf = len(FILE_FIELDS[name])
        rows = [_numbers(line, 15)[:2 * nf] for rep in reports for line in rep]
        a = np.asarray(rows, dtype=np.float64).reshape(-1, nf, 2)
        z = np.where(a[:, :, 1] == _MISSING, np.nan, a[:, :, 1])
        out[name] = (a[:, :, 0], z)
    return out


def read_obs_space_file(path) -> dict:
    """WRFDA's ``unpert_obs.0000`` / ``pert_obs.0000``: ``{type: (nlevels, nfields)}`` of ``y`` (NaN where missing)."""
    out = {}
    for name, reports in _blocks(path):
        if name not in FILE_FIELDS:
            continue
        nf = len(FILE_FIELDS[name])
        rows = [_numbers(line, 15)[:nf] for rep in reports for line in rep]
        a = np.asarray(rows, dtype=np.float64).reshape(-1, nf)
        out[name] = np.where(a == _MISSING, np.nan, a)
    return out


@dataclass
class DesroziersResult:
    """Per ``(type, variable)`` rows and the totals; see :func:`randomised`."""

    rows: dict = field(default_factory=dict)
    totals: dict = field(default_factory=dict)
    #: reports per type (``ob%num_<type>``), for the table's first count column
    reports: dict = field(default_factory=dict)


def _split(flat, slots):
    out = {}
    for n, v, a, b in slots:
        out.setdefault(n, {})[v] = np.asarray(flat[a:b])
    return out


def classic(term, y) -> dict:
    """Per ``(type, variable)``: ``n``, ``omb2``, ``sigma_o2`` (specified), ``so2_est`` ``= <(O-A)(O-B)>``,
    ``hbht_est`` ``= <(A-B)(O-B)>``, ``ratio_o`` ``= so2_est / sigma_o2`` and ``ratio_b``
    ``= hbht_est / (omb2 - sigma_o2)`` (NaN when that denominator is not positive).

    ``term`` is the :class:`~woof.da.var.obsop_wrfda.FlatObsTerm` of the analysis and ``y`` its
    ``tl`` at the analysis increment (flat).
    """
    y = np.asarray(y, dtype=np.float64)
    out = {}
    for n, v, a, b in term.slots:
        m = term.mask[a:b]
        if not m.any():
            continue
        d, e, ya = term.d[a:b][m], term.err[a:b][m], y[a:b][m]
        omb, oma, amb = d, d - ya, ya
        so2 = float(np.mean(e * e))
        omb2 = float(np.mean(omb * omb))
        est = float(np.mean(oma * omb))
        hbht = float(np.mean(amb * omb))
        den = omb2 - so2
        out[(n, v)] = {"n": int(m.sum()), "omb2": omb2, "sigma_o2": so2, "so2_est": est, "hbht_est": hbht,
                       "ratio_o": est / so2, "ratio_b": hbht / den if den > 0 else math.nan}
    return out


def randomised(term, y, y_pert, z, *, jb: float | None = None) -> DesroziersResult:
    """The tool's trace, expected and actual ``Jo`` and error factors.

    ``y`` and ``y_pert`` are the flat observation-space increments ``H'(xhat)`` of the unperturbed and
    perturbed analyses, ``z`` ``{type: {variable: per-level N(0,1) draws}}`` (NaN where the
    observation was not used).  ``jb`` is the analysis's ``Jb`` (``|cvt|^2 / 2``).
    """
    y = np.asarray(y, dtype=np.float64)
    yp = np.asarray(y_pert, dtype=np.float64)
    res = DesroziersResult()
    total_obs = 0
    trace_total = 0.0
    jo_total = 0.0
    for n, v, a, b in term.slots:
        m = term.mask[a:b] & np.isfinite(z[n][v])
        err, d = term.err[a:b], term.d[a:b]
        count = int(m.sum())
        trace = float(np.sum(((yp[a:b] - y[a:b]) * z[n][v] / err)[m])) if count else 0.0
        jo_a = 0.5 * float(np.sum(((d - y[a:b]) ** 2 / (err * err))[m]))
        res.rows[(n, v)] = {"n": count, "trace": trace, "jo_expected": 0.5 * (count - trace), "jo_actual": jo_a,
                            "factor": math.sqrt(jo_a / (0.5 * (count - trace))) if 0.5 * (count - trace) > 0 else 1.0}
        if _in_table(n, v):  # the totals cover what the table does
            total_obs += count
            trace_total += trace
            jo_total += jo_a
    res.totals = {"total_obs": total_obs, "trace_total": trace_total, "j_expected": 0.5 * total_obs,
                  "jo_actual": jo_total, "jo_expected": 0.5 * (total_obs - trace_total),
                  "jo_factor": (math.sqrt(jo_total / (0.5 * (total_obs - trace_total)))
                                if total_obs - trace_total > 0 else math.nan),
                  "jb_expected": 0.5 * trace_total}
    if jb is not None:
        res.totals["jb_actual"] = jb
        res.totals["j_actual"] = jo_total + jb
    return res


def format_report(res: DesroziersResult) -> str:
    """The table ``da_tune_obs_desroziers`` prints: per type, ``Jo (expected)``, ``Jo (actual)`` and ``Error Factor``."""
    out = []
    for n, tool in TOOL_NAMES:
        fields = [v for v in FILE_FIELDS[n] if (n, v) in res.rows][: _TOOL_FIELDS[n]]
        if not fields or not any(res.rows[(n, v)]["n"] for v in fields):  # the tool prints types it has reports of
            continue
        ex, ac, fa = ([0.0] * 5 for _ in range(3))
        for k, v in enumerate(fields):
            r = res.rows[(n, v)]
            ex[k], ac[k], fa[k] = r["jo_expected"], r["jo_actual"], r["factor"] if r["jo_expected"] > 0 else 1.0
        for k in range(len(fields), 5):
            fa[k] = 1.0
        nobs = res.reports.get(n, max(res.rows[(n, v)]["n"] for v in fields))
        tot = sum(res.rows[(n, v)]["n"] for v in fields)
        for label, vals in (("Jo (expected)=", ex), ("Jo (actual)  =", ac), ("Error Factor =", fa)):
            out.append(f" {tool} obs, {label}{nobs:9d}{tot:8d}" + "".join(f"{x:15.5f}" for x in vals))
        out.append("")
    t = res.totals
    out += ["", f" Total number of obs.    = {t['total_obs']:8d}"]
    if "j_actual" in t:
        out += [f" Total J (actual)        = {t['j_actual']:15.5f}"]
    out += [f" Total J (expected)      = {t['j_expected']:15.5f}", "",
            f" Total Jo(actual)        = {t['jo_actual']:15.5f}",
            f" Total Jo(expected)      = {t['jo_expected']:15.5f}",
            f" Total Jo factor         = {t['jo_factor']:15.5f}", ""]
    if "jb_actual" in t:
        out += [f" Total Jb(actual)        = {t['jb_actual']:15.5f}"]
    out += [f" Total Jb(expected)      = {t['jb_expected']:15.5f}"]
    return "\n".join(out)


def run(be, obs, first_guess, rand_obs_error, *, tier: str = "reference", ntmax=50, eps=0.01, window_minutes=60.0,
        orthonorm_gradient: bool = True, check_state: bool = False):
    """The unperturbed and the perturbed 3-D-Var analyses of the same cost, and the diagnostics.

    ``rand_obs_error`` is WRFDA's noise file of an ``omb_add_noise`` run (the draws ``z`` and the
    errors it used).  Returns ``(randomised result, classic statistics, term, y, y_pert)``; ``y`` and
    ``y_pert`` are ``H'(xhat)`` and ``H'(xhat_p)`` (flat, the order of ``term.slots``).  The cost is
    solved twice in place: the second solve has the innovations ``d + sigma z`` and nothing else different.
    """
    from . import be_dat as bd
    from . import innov_wrfda as iw
    from . import obs_wrfda as ow
    from .analysis import FirstGuess, obs_window, read_first_guess, wrftoxb, xa_from_cv
    from .cost import VarCost
    from .minimise import minimise
    from .obsop_wrfda import FlatObsTerm

    fg = first_guess if isinstance(first_guess, FirstGuess) else read_first_guess(first_guess)
    xb, _ = wrftoxb(fg)
    nx, ny, nz = xb["t"].shape
    b = bd.setup_be_regional(be if not isinstance(be, (str, Path)) else bd.read_be_dat(be),
                             ix=nx, jy=ny, kz=nz, ds=float(xb["ds"]))
    proj = ow.LambertWRFDA.from_attributes(fg.attrs)
    wobs = obs if not isinstance(obs, (str, Path)) else ow.read_obs_ascii(obs, proj, **obs_window(fg, window_minutes))
    iv = iw.innovations(wobs, xb)
    term = FlatObsTerm(iv, xb)
    xbu = {k: xb[k] for k in ("grid_box_area", "coefx", "coefy", "qs", "q", "psac", "dnw", "rho", "p", "t",
                              "c1h", "c2h")}
    cost = VarCost.reference(b, xbu, term) if tier == "reference" else VarCost.device(b, xbu, term, tier)

    def solve():
        _, cvt = minimise(cost, max_ext_its=1, ntmax=ntmax, eps=eps, orthonorm_gradient=orthonorm_gradient)
        cvt = np.asarray(cvt.get() if hasattr(cvt, "get") else cvt, dtype=np.float64)
        return cvt, np.asarray(term.tl(xa_from_cv(b, xbu, cvt)), dtype=np.float64)

    cvt, y = solve()
    jb = 0.5 * float(np.dot(cvt, cvt))
    noise = read_rand_obs_error(rand_obs_error)
    z = {}
    dp = term.d.copy()
    for n, v, a, c in term.slots:
        if c == a:
            z[(n, v)] = np.zeros(0)
            continue
        err, zz = noise[n]
        k = FILE_FIELDS[n].index(v)
        if len(zz) != c - a:
            raise ValueError(f"{n}: {len(zz)} noise records for {c - a} levels")
        z[(n, v)] = zz[:, k]
        use = term.mask[a:c] & np.isfinite(zz[:, k])
        # the draw must belong to these observations: the file's error is the run's sigma (printed to 7 digits)
        if not np.allclose(err[use, k], term.err[a:c][use], rtol=1e-5, atol=0.0):
            raise ValueError(f"{n} {v}: rand_obs_error's errors differ from this run's observation errors")
        dp[a:c][use] += term.err[a:c][use] * zz[:, k][use]
    zz = {}
    for (n, v), a in z.items():
        zz.setdefault(n, {})[v] = a
    cost.set_innovations([dp])
    _, y_pert = solve()
    cost.set_innovations([term.d])
    if check_state:
        _, y_again = solve()
        if not np.array_equal(y_again, y):
            raise RuntimeError("the cost carries state between solves: a repeat solve differs from the first")
    res = randomised(term, y, y_pert, zz, jb=jb)
    res.reports = {n: int(np.sum(iv.types[n].info["proc_domain"])) for n, _ in TOOL_NAMES if n in iv.types}
    return res, classic(term, y), term, y, y_pert
