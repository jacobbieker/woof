"""WRF Fitch turbine tables: what ``wind-turbine-N.tbl`` holds, and dragcof.

WRF v4.7.1's Fitch scheme (``phys/module_wind_fitch.F``) reads one table
per turbine type (``init_module_wind_fitch``:491-510)::

    nval
    hubheight  diameter  stc  npower
    ws(1)  ct(1)  pw(1)          <- nval rows, ws in m/s, pw in kW
    ...

into ``turbws/turbtc/turbpw(type, 1:MAXVALS2)`` (zero beyond ``nval``),
takes ``cutin = ws(1)``, ``cutout = ws(nval)`` and ``stc2 = ct(nval)``, and
every step ``dragcof``:277-334 interpolates them at the hub wind: the
thrust coefficient (``stc`` at or below cut-in, ``stc2`` at or above
cut-out, linear between), the power coefficient (zero outside, linear in
``pw / (0.5 * 1.23 * ws**3 * area)`` between), and the TKE coefficient
``correction_factor * (thrcof - powcof)`` clipped at zero; ``dragforce``
then forms ``power1 = 0.5 * 1.23 * speedhub**3 * area * powcof``
(:190-191), W.  Cut-out is a hard switch there: WRF has no hysteresis.

A library curve becomes a WRF table here (:func:`table_from_curve`): its
own points from the last zero-power speed above 0 m/s (cut-in) through
``cut_out_ms`` (a point added there if the curve has none), power from the
curve and thrust from :meth:`TurbineCurve.thrust_coefficient`, each rounded
once to float32 -- the words WRF's list-directed read would make of the
values :func:`write_table` prints (shortest round-trip decimals).  A row
at 0 m/s is never written: ``dragcof`` divides by ``ws(nb)**3``.

:func:`dragcof_cpu` is ``dragcof`` plus ``power1``, element by element, on
the CPU (NumPy float32; ``**`` is glibc 2.39's ``powf``,
:func:`woof.core.noahmp_libm.powf_array`, because gfortran calls ``powf``
for a REAL exponent even when it is 2.), held at max_ulp 0 against the
unmodified WRF statements by ``tests/test_windfarm_layout_wrf471_parity.py``.
It builds the layout's axis tables.  On the device ``dragcof`` is inlined in
the Fitch kernel itself (``woof/core/kernels/windfarm_fitch.cu``), held to
WRF by ``tools/windfarm_fitch_wrf471_oracle``; a second device copy here was
retired when the scheme was wired into the model.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

F = np.float32

#: module_wind_fitch.F:54-55: at most 100 turbine types, 100 rows each.
MAXVALS = 100
MAXVALS2 = 100

#: module_model_constants.F:72 piconst, a REAL parameter.
PICONST = F(3.1415926535897932384626433)

#: A parked rotor's thrust coefficient, the ``stc`` of each table header.
#: WRF reads it from every table and ships no default; woof pins one.
STANDING_THRUST_COEFFICIENT = F(0.13)


class FitchTableRefused(ValueError):
    """A curve cannot be written as a WRF Fitch table."""


@dataclass(frozen=True)
class FitchTables:
    """``turbws/turbtc/turbpw`` and the per-type header values, by curve.

    ``speeds_ms``/``thrust_coefficient``/``power_kw`` are ``(n, MAXVALS2)``
    float32, zero past each curve's ``nval`` rows (int32), exactly as
    WRF's zero-initialised arrays.  ``cut_in_ms`` = ``ws(1)``,
    ``cut_out_ms`` = ``ws(nval)``, ``standing_thrust`` = ``stc`` and
    ``standing_thrust_cutout`` = ``stc2`` = ``ct(nval)``.  The names match
    the Fitch kernel's curve fields.  The arrays may be NumPy or CuPy.
    """

    nval: Any
    speeds_ms: Any
    thrust_coefficient: Any
    power_kw: Any
    cut_in_ms: Any
    cut_out_ms: Any
    standing_thrust: Any
    standing_thrust_cutout: Any

    @classmethod
    def from_curves(cls, curves, *, standing_thrust=STANDING_THRUST_COEFFICIENT
                    ) -> "FitchTables":
        rows = [table_rows(curve) for curve in curves]
        n = len(rows)
        ws = np.zeros((n, MAXVALS2), dtype=F)
        tc = np.zeros((n, MAXVALS2), dtype=F)
        pw = np.zeros((n, MAXVALS2), dtype=F)
        count = np.zeros(n, dtype=np.int32)
        for k, (s, c, p) in enumerate(rows):
            count[k] = s.size
            ws[k, :s.size], tc[k, :s.size], pw[k, :s.size] = s, c, p
        last = np.maximum(count - 1, 0)
        index = np.arange(n)
        arrays = dict(nval=count, speeds_ms=ws, thrust_coefficient=tc,
                      power_kw=pw, cut_in_ms=ws[:, 0].copy(),
                      cut_out_ms=ws[index, last].copy(),
                      standing_thrust=np.full(n, F(standing_thrust), dtype=F),
                      standing_thrust_cutout=tc[index, last].copy())
        for value in arrays.values():
            value.setflags(write=False)
        return cls(**arrays)

    def arrays(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def table_rows(curve) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ws, ct, pw)`` float32 rows of one library curve's WRF table."""

    speeds = np.asarray(curve.speeds, dtype=np.float64)
    power = np.asarray(curve.power_kw, dtype=np.float64)
    on = np.flatnonzero(power > 0.0)
    if on.size == 0:
        raise FitchTableRefused(f"curve {curve.curve_id} never makes power")
    first = int(on[0])
    start = first - 1 if first > 0 and speeds[first - 1] > 0.0 else first
    if speeds[start] <= 0.0:
        raise FitchTableRefused(
            f"curve {curve.curve_id}: its first powered point is at 0 m/s, "
            "and WRF's dragcof divides by ws(1)**3")
    cut_out = float(curve.cut_out_ms)
    keep = (np.arange(speeds.size) >= start) & (speeds <= cut_out)
    ws = speeds[keep]
    if ws.size == 0 or ws[-1] < cut_out:
        ws = np.append(ws, cut_out)
    pw = curve.fraction(ws) * curve.rated_kw
    ct = curve.thrust_coefficient(ws)
    ws32, ct32, pw32 = ws.astype(F), ct.astype(F), pw.astype(F)
    if ws32.size < 2 or np.any(np.diff(ws32) <= 0):
        raise FitchTableRefused(
            f"curve {curve.curve_id}: its WRF table needs two or more rows "
            "of increasing speed")
    if ws32.size > MAXVALS2:
        raise FitchTableRefused(
            f"curve {curve.curve_id}: {ws32.size} rows, more than WRF's "
            f"MAXVALS2 = {MAXVALS2}")
    return ws32, ct32, pw32


def float32_word(value) -> str:
    """The shortest decimal that reads back as this float32 (what woof
    writes wherever WRF will read a REAL)."""

    return np.format_float_positional(F(value), unique=True, trim="0")


def write_table(path, tables: FitchTables, curve: int, *, hub_height_m,
                diameter_m, nominal_power_mw) -> Path:
    """One ``wind-turbine-N.tbl`` in WRF's list-directed layout."""

    rows = int(tables.nval[curve])
    lines = [str(rows),
             " ".join(float32_word(v) for v in (hub_height_m, diameter_m,
                                         tables.standing_thrust[curve],
                                         nominal_power_mw))]
    for k in range(rows):
        lines.append(" ".join(float32_word(a[curve, k]) for a in (
            tables.speeds_ms, tables.thrust_coefficient, tables.power_kw)))
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


# --------------------------------------------------------------------------
# dragcof


def dragcof_cpu(tables: FitchTables, curve, speed, diameter_m,
                correction_factor=1.0) -> dict[str, np.ndarray]:
    """WRF ``dragcof`` + ``power1`` for each ``(curve, speed, diameter)``.

    Returns float32 ``thrcof``, ``powcof``, ``tkecof`` and ``power1`` (W).
    """

    from woof.core.noahmp_libm import powf_array

    speed = np.ascontiguousarray(speed, dtype=F)
    curve = np.ascontiguousarray(curve, dtype=np.int64)
    diameter = np.broadcast_to(np.asarray(diameter_m, dtype=F), speed.shape)
    cf = F(correction_factor)
    ws = np.asarray(tables.speeds_ms, dtype=F)[curve]
    tc = np.asarray(tables.thrust_coefficient, dtype=F)[curve]
    pw = np.asarray(tables.power_kw, dtype=F)[curve]
    ci = np.asarray(tables.cut_in_ms, dtype=F)[curve]
    co = np.asarray(tables.cut_out_ms, dtype=F)[curve]
    stc = np.asarray(tables.standing_thrust, dtype=F)[curve]
    stc2 = np.asarray(tables.standing_thrust_cutout, dtype=F)[curve]
    # the bracket: first k with turbws(k) > speed (Fortran's vfound loop)
    above = ws > speed[:, None]
    found = above.any(axis=1)
    nu = np.where(found, above.argmax(axis=1), 1)
    nb = np.maximum(nu - 1, 0)
    rows = np.arange(speed.size)
    ws_nu, ws_nb = ws[rows, nu], ws[rows, nb]
    lo = speed <= ci
    hi = speed >= co
    off = lo | hi
    with np.errstate(all="ignore"):
        d = ws_nu - ws_nb
        s = speed - ws_nb
        thr_mid = tc[rows, nb] + (tc[rows, nu] - tc[rows, nb]) / d * s
        thr = np.where(lo, stc, np.where(hi, stc2, thr_mid)).astype(F)
        two, three = F(2.0), F(3.0)
        area = F(PICONST / F(4.0)) * powf_array(diameter, two)
        half_rho = F(F(0.5) * F(1.23))
        fac1 = F(1000.0) / (half_rho * powf_array(ws_nb, three) * area)
        fac2 = F(1000.0) / (half_rho * powf_array(ws_nu, three) * area)
        pw_nb, pw_nu = pw[rows, nb], pw[rows, nu]
        powcof = pw_nb * fac1 + (pw_nu * fac2 - pw_nb * fac1) / d * s
        powcof = np.where(off, F(0.0), powcof).astype(F)
        tke = cf * (thr - powcof)
        tke = np.where(tke < F(0.0), F(0.0), tke).astype(F)
        power1 = half_rho * powf_array(speed, three) * area * powcof
    return {"thrcof": thr, "powcof": powcof, "tkecof": tke,
            "power1": power1.astype(F)}


__all__ = ["MAXVALS", "MAXVALS2", "PICONST", "STANDING_THRUST_COEFFICIENT",
           "FitchTableRefused", "FitchTables", "table_rows", "write_table",
           "float32_word", "dragcof_cpu"]
