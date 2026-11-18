"""Solar-proton NOx and HOx in RADM2 chemistry (``spacewx_epp_opt = 1``).

The ion pairs that solar protons make in the polar caps
(:mod:`woof.spacewx.particles`) become NO, OH and HO2 in the RADM2 rows
``no``, ``ho`` and ``ho2`` (ppmv), at each chemistry step, inside the
``emission.radm2`` process, whose ledger books them as emitted:

- 1.25 NO per ion pair (the N atoms the ions make; RADM2 has no N, so all of
  it is NO, as in models without odd-nitrogen atoms);
- 2 HOx per ion pair, as one OH and one HO2 (the H atom becomes HO2 within
  seconds through H + O2 + M);
- only in the polar caps, |dipole magnetic latitude| >= 60 deg at the start.

The flux is the GOES integral proton flux of the last hour that has ended by
the domain's start (``spacewx_epp_source``: ``omni``, ``swpc`` or OMNI2
file(s) joined by ``:``), held for the run: persistence, like the Kp of the
aurora rows.  Each domain holds the hour before its own start, so a nest
started with its parent holds the same hour and a later nest its own.  A
fetched flux is kept per source and start in ``~/.woof/spacewx/epp-flux``,
so a resume holds the same one.  An hour below NOAA's S1 threshold (10 pfu
above 10 MeV) adds nothing; an hour whose >10 MeV flux is missing is
refused (OMNI2's proton words 43-48 are fill from 2021 on: use ``swpc``).

On the device the source is one interpolation per cell: the deposition rate
per unit depth, dF/dX / W, is tabulated on :data:`particles.DEPTH_GRID`
(uniform in log X) in float64 on the host, and each layer reads it, log-log,
at its mid depth sqrt(X_top X_bottom), with X = p8w / g.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from woof.spacewx import particles

#: The rows the source adds to and the molecules per ion pair it adds.
YIELDS = (("no", particles.NOX_PER_ION_PAIR),
          ("ho", particles.HOX_PER_ION_PAIR / 2.0),
          ("ho2", particles.HOX_PER_ION_PAIR / 2.0))
SOURCES = ("omni", "swpc")
#: ln of the floor on the tabulated deposition rate (ion pairs g-1 s-1);
#: below it a layer gets no source.
LOG_FLOOR = math.log(1.0e-30)


class EppChemError(ValueError):
    pass


def enabled(cfg) -> bool:
    return int(getattr(cfg, "spacewx_epp_opt", 0) or 0) == 1


def device_shapes(cfg) -> dict[str, tuple[int, ...]]:
    """The held float32 polar-cap mask (preflight's memory inventory)."""
    if not enabled(cfg):
        return {}
    return {"spacewx_epp/cap": (int(cfg.ny), int(cfg.nx))}


def transient_shapes(cfg) -> dict[str, tuple[int, ...]]:
    """The peak of :func:`ion_pairs` and the row update, a bound on what is
    live together: the float32 w-level pressures and their log, and eleven
    mass-level volumes (mid depth, its clipped position, index, weight, the
    two weighted table reads, their sum, its exp, the rate, its masked copy
    and one row's increment)."""
    if not enabled(cfg):
        return {}
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    shapes = {f"spacewx_epp/{name}": (nz + 1, ny, nx) for name in ("p8w", "logp")}
    shapes.update({f"spacewx_epp/{name}": (nz, ny, nx)
                   for name in ("lmid", "u", "index", "weight", "lo", "hi", "lq", "exp", "rate",
                                "masked", "increment")})
    return shapes


def parse_source(source: str) -> str:
    """``omni``, ``swpc`` or OMNI2 file path(s) joined by ``:`` (returned
    with each path's ``~`` expanded)."""
    text = str(source or "").strip()
    if not text:
        raise EppChemError("spacewx_epp_opt = 1 needs spacewx_epp_source: 'omni', 'swpc' or OMNI2 file(s)")
    if text.lower() in SOURCES:
        return text.lower()
    paths = [Path(part.strip()).expanduser() for part in text.split(":")]
    missing = [str(p) for p in paths if not str(p) or not p.is_file()]
    if missing:
        raise EppChemError(f"spacewx_epp_source {text!r} is neither 'omni' nor 'swpc' nor existing OMNI2 "
                           f"file(s) joined by ':' (missing: {', '.join(missing)})")
    return ":".join(str(p) for p in paths)


def flux_hour(start: datetime) -> datetime:
    """The start of the last hour that has ended by ``start``."""
    return start.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)


def held_flux(source: str, start: datetime, *, loader=None, memo=None) -> dict:
    """The run's held flux: ``{"channels", "flux", "spe", "record"}``.

    With ``memo`` (a directory), the result of a fetched source is kept per
    (source, start); a refused hour is never kept, so a later call reads the
    source again.  An hour whose >10 MeV flux is missing is refused, not taken
    as quiet."""
    kind = parse_source(source)
    hour = flux_hour(start)
    path = None
    if memo is not None and kind in SOURCES:
        key = hashlib.sha256(f"{kind}|{start.isoformat()}".encode()).hexdigest()[:24]
        path = Path(memo) / f"{kind}-{start:%Y%m%dT%H%M%S}-{key}.json"
        if path.is_file():
            try:
                held = json.loads(path.read_text(encoding="utf-8"))
                return {"channels": tuple(float(c) for c in held["channels"]),
                        "flux": np.asarray(held["flux"], dtype=np.float64), "spe": bool(held["spe"]),
                        "record": dict(held["record"], spacewx_epp_held_from=str(path))}
            except (ValueError, KeyError, TypeError):
                pass                                     # damaged: read the source again
    load = loader or particles.load
    try:
        rec = load(kind, hour - timedelta(hours=1), start)
        flux = np.asarray(rec.at(hour), dtype=np.float64)
    except (particles.ParticleError, OSError, ValueError) as exc:
        raise EppChemError(f"spacewx_epp_opt = 1: no proton flux for the hour from {hour:%Y-%m-%dT%H:%M}Z "
                           f"({kind}): {exc}") from None
    try:
        j10 = flux[list(rec.channels).index(10.0)]
    except ValueError:
        raise EppChemError(f"spacewx_epp_opt = 1: the proton record ({kind}) has no >10 MeV channel") from None
    if not np.isfinite(j10):
        hint = (" OMNI2's proton words (43-48) are fill from 2021 on; use spacewx_epp_source = 'swpc' "
                "for the last three days" if kind != "swpc" else "")
        raise EppChemError(f"spacewx_epp_opt = 1: the >10 MeV proton flux of the hour from "
                           f"{hour:%Y-%m-%dT%H:%M}Z is missing in {rec.source}, so whether a solar proton "
                           f"event was on is unknown.{hint}")
    out = {"channels": tuple(float(c) for c in rec.channels), "flux": flux, "spe": bool(rec.active(flux)),
           "record": {"spacewx_epp_source": rec.source, "spacewx_epp_sha256": rec.sha256,
                      "spacewx_epp_hour": f"{hour:%Y-%m-%dT%H:%M}Z"}}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f"{path.name}.{os.getpid()}.part")
        part.write_text(json.dumps({"channels": list(out["channels"]),
                                    "flux": [None if not np.isfinite(v) else float(v) for v in flux],
                                    "spe": out["spe"], "record": out["record"]}, sort_keys=True) + "\n",
                        encoding="utf-8")
        part.replace(path)
    return out


def log_rate_table(flux, channels) -> np.ndarray:
    """ln of the ion-pair rate per gram, (dF/dX) / W, on :data:`particles.DEPTH_GRID`."""
    flux = np.asarray([np.nan if v is None else v for v in flux], dtype=np.float64)
    f = particles.absorbed(flux, particles.DEPTH_GRID, tuple(channels), particles.range_table())
    x = particles.DEPTH_GRID
    rate = np.gradient(f, np.log(x)) / x / particles.W_ION_MEV
    return np.log(np.maximum(rate, math.exp(LOG_FLOOR)))


def ion_pairs(p8w, log_rate, xp=np):
    """Ion pairs per gram per second on each mass level from the w-level
    pressures ``p8w`` (Pa, (nz + 1, ...)); float32 arithmetic on the device."""
    grid = particles.DEPTH_GRID
    lx0 = float(np.log(grid[0]))
    dlx = float(np.log(grid[-1] / grid[0]) / (grid.size - 1))
    p = xp.maximum(p8w.astype(xp.float32), xp.float32(1.0e-6))
    # X = p / g in g cm-2 (kg m-2 / 10); the mid depth in log.
    lmid = xp.float32(0.5) * (xp.log(p[:-1]) + xp.log(p[1:])) + xp.float32(math.log(0.1 / particles._G))
    u = xp.clip((lmid - xp.float32(lx0)) / xp.float32(dlx), 0.0, float(grid.size - 1))
    i = xp.minimum(xp.floor(u).astype(xp.int32), grid.size - 2)
    f = u - i.astype(xp.float32)
    table = xp.asarray(log_rate, dtype=xp.float32)
    lq = table[i] * (xp.float32(1.0) - f) + table[i + 1] * f
    return xp.where(lq > xp.float32(LOG_FLOOR + 1.0), xp.exp(lq), xp.float32(0.0))


class EppSource:
    """The option's state on one domain, rebuilt from the RunConfig."""

    def __init__(self, start: datetime, held: dict, cap):
        self.start = start
        self.held = held
        self.cap = cap                                   # (ny, nx) float32 0/1 on the device
        self.log_rate = None
        # No cell in a polar cap (a mid-latitude domain): nothing to add.
        if held["spe"] and bool((cap > 0).any()):
            try:
                self.log_rate = log_rate_table(held["flux"], held["channels"])
            except particles.ParticleError as exc:
                raise EppChemError(f"spacewx_epp_opt = 1: {exc}") from None

    @classmethod
    def for_context(cls, ctx, *, loader=None, igrf_model=None):
        import cupy as cp
        from woof.spacewx import aurora, igrf
        from woof.spacewx.solar_irradiance import cache_root
        from woof.spacewx.indices import naive_utc
        cfg = ctx.cfg
        start = getattr(cfg, "start_time", None) or getattr(ctx.state.chem, "start_time", None)
        if start is None:
            raise EppChemError("spacewx_epp_opt = 1 needs the run's start time")
        try:
            lat, lon = ctx.met("xlat"), ctx.met("xlong")
        except KeyError:
            raise EppChemError("spacewx_epp_opt = 1 places the source in the polar caps and this "
                               "domain publishes no xlat/xlong") from None
        start = naive_utc(start)
        held = held_flux(getattr(cfg, "spacewx_epp_source", ""), start, loader=loader,
                         memo=None if loader is not None else cache_root() / "epp-flux")
        frame = aurora.dipole_frame(igrf_model or igrf.load(), start)
        mlat = aurora.dipole_latitude(frame, cp.asnumpy(lat), cp.asnumpy(lon))
        cap = cp.asarray(particles.polar_cap(mlat).astype(np.float32))
        return cls(start, held, cap)

    def receipt(self) -> dict:
        flux = self.held["flux"]
        j10 = dict(zip(self.held["channels"], flux)).get(10.0)
        return {**self.held["record"], "spacewx_epp_spe": self.held["spe"],
                "spacewx_epp_j10_pfu": None if j10 is None or not np.isfinite(j10) else float(j10)}

    def apply(self, rows: dict, p8w, dt: float) -> None:
        """Add one chemistry step's source to ``rows`` (``{name: (nz, ny, nx) ppmv}``)."""
        if self.log_rate is None:
            return
        import cupy as cp
        q = ion_pairs(p8w, self.log_rate, cp) * self.cap
        per_pair = np.float32(particles.vmr_tendency(1.0, 1.0) * 1.0e6 * float(dt))   # ppmv per (ion pair g-1 s-1)
        for name, yield_ in YIELDS:
            a = rows[name]
            a += q * np.float32(per_pair * yield_)


__all__ = ["YIELDS", "EppChemError", "EppSource", "enabled", "device_shapes", "transient_shapes", "parse_source", "flux_hour", "held_flux",
           "log_rate_table", "ion_pairs"]
