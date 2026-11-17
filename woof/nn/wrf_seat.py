"""The WRF route's NN column seat (plan section 4.1, P2.2).

``woof.core.dycore._step_body`` calls :meth:`WrfColumnSeat.after_nonmicrophysics`
once per dycore step, on the time-t state, right after the ensemble
stochastic binding (SPPT/SKEBS) has acted on the step's held
non-microphysics tendencies and before chemistry, IAU and FDDA fold theirs
in -- the same seat, so a scheme an NN replaces is perturbed and nudged as
the scheme was.  A state without ``_nn_column_seat`` (every forecast
without ``--nn``) skips the call; the default path is unchanged.

Columns.  :meth:`WrfColumnSeat.columns` draws a set of mass-grid columns
and assembles the vocabulary below for them as ``[feature, column]``
device arrays, levels bottom-up, each field's rows contiguous in the order
the configuration names them (the layout :mod:`woof.nn.manifest` takes).

Targets.  The step holds tendencies dry-mass-coupled (``c1h * mu + c2h``
times the physical rate, theta also divided by the mass-point map factor
in fast numerics, zero in a specified ring).  The seat recovers physical
rates by dividing by the coupler's own response to a unit rate
(:func:`woof.core.physics.couple_column_tendencies` on ones), so whatever
convention the coupler follows, strict or fast numerics, the division is
its exact inverse on the interior.  ``du``/``dv`` are the face pair's mean
over the same mass factor: C-grid to A-grid, approximate (it smooths at
the grid scale and ignores the u/v-point map factors).

``mode = "tap"`` (this module's one mode for now) records ``(inputs,
targets)`` to a :class:`woof.nn.tap.TendencyTap` and returns the step's
tendencies object untouched, so a tapped forecast is byte-identical to the
same forecast without ``--nn``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from woof.nn.launch import NNConfigError, NNSettings, configured, read_settings, refuse_streamed
from woof.nn.manifest import FieldSpec

G = 9.81
RD_OVER_CP = 287.0 / 1004.5
P0 = 1.0e5

# name -> (units, kind); kind "3d" spans the model levels, "2d" one value
# per column.  "surface:<key>" fields come from state.physics.fields.
VOCABULARY: dict[str, tuple[str, str]] = {
    "theta": ("K", "3d"), "qv": ("kg kg-1", "3d"), "qc": ("kg kg-1", "3d"),
    "qr": ("kg kg-1", "3d"), "qi": ("kg kg-1", "3d"), "qs": ("kg kg-1", "3d"),
    "qg": ("kg kg-1", "3d"), "u": ("m s-1", "3d"), "v": ("m s-1", "3d"),
    "w": ("m s-1", "3d"), "p": ("Pa", "3d"), "exner": ("1", "3d"),
    "z": ("m", "3d"), "dz": ("m", "3d"), "rthraten": ("K s-1", "3d"),
    "hgt": ("m", "2d"), "mu": ("Pa", "2d"),
    "hfx": ("W m-2", "2d"), "qfx": ("kg m-2 s-1", "2d"), "ust": ("m s-1", "2d"),
    "tsk": ("K", "2d"), "znt": ("m", "2d"), "xland": ("1", "2d"), "pblh": ("m", "2d"),
    "psfc": ("Pa", "2d"), "lh": ("W m-2", "2d"), "coszen": ("1", "2d"),
}
SURFACE = ("hfx", "qfx", "ust", "tsk", "znt", "xland", "pblh", "psfc", "lh", "coszen")
OUTPUTS: dict[str, tuple[str, str]] = {
    "dtheta": ("K s-1", "rtheta"), "dqv": ("kg kg-1 s-1", "rqv"), "dqc": ("kg kg-1 s-1", "rqc"),
    "dqr": ("kg kg-1 s-1", "rqr"), "dqi": ("kg kg-1 s-1", "rqi"), "dqs": ("kg kg-1 s-1", "rqs"),
    "du": ("m s-2", "ru"), "dv": ("m s-2", "rv"),
}
DEFAULT_INPUTS = {
    "pbl": ("theta", "qv", "qc", "u", "v", "p", "z", "hfx", "qfx", "ust", "tsk", "znt", "xland"),
    "radiation": ("theta", "qv", "qc", "p", "z", "tsk", "xland", "coszen"),
    "cumulus": ("theta", "qv", "qc", "w", "p", "z", "hfx", "qfx"),
    "total": ("theta", "qv", "qc", "u", "v", "w", "p", "z", "hfx", "qfx", "ust", "tsk", "znt",
              "xland", "rthraten"),
}
DEFAULT_OUTPUTS = {
    "pbl": ("dtheta", "dqv", "dqc", "du", "dv"),
    "radiation": ("dtheta",),
    "cumulus": ("dtheta", "dqv", "dqc", "dqr"),
    "total": ("dtheta", "dqv", "dqc", "du", "dv"),
}
SCHEME_SLOT = {"pbl": "pbl_tendencies", "radiation": "radiation_tendencies",
               "cumulus": "cumulus_tendencies"}


def _host(array) -> np.ndarray:
    return array.get() if hasattr(array, "get") else np.asarray(array)


def _xp(state):
    from woof.nn.bridge import array_module
    return array_module(state.p)


def interior_pool(cfg, ny: int, nx: int) -> np.ndarray:
    """Flat mass-grid indices ``j * nx + i`` a column may be drawn from: a
    specified or nested domain's ``spec_bdy_width`` ring is excluded
    (its held tendencies are masked to zero)."""
    ring = int(cfg.spec_bdy_width) if (cfg.specified or cfg.nested) else 0
    if 2 * ring >= min(ny, nx):
        raise NNConfigError(f"the {ny} x {nx} domain has no interior inside its {ring}-column boundary ring")
    j, i = np.mgrid[ring:ny - ring, ring:nx - ring]
    return (j * nx + i).ravel().astype(np.int64)


@dataclass
class _Gather:
    """Device gathers of the drawn columns from the mass, u-face and
    v-face grids."""

    xp: Any
    mass: Any
    u_west: Any
    u_east: Any
    v_south: Any
    v_north: Any

    @classmethod
    def build(cls, xp, flat: np.ndarray, nx: int):
        j, i = np.divmod(flat, nx)
        dev = lambda a: xp.asarray(a.astype(np.int32))
        return cls(xp, dev(flat), dev(j * (nx + 1) + i), dev(j * (nx + 1) + i + 1),
                   dev(j * nx + i), dev((j + 1) * nx + i))

    def take(self, array, index=None, levels=None):
        index = self.mass if index is None else index
        if array.ndim == 2:
            return self.xp.take(array.reshape(1, -1), index, axis=1)
        if levels is not None:
            array = array[:levels]
        return self.xp.take(array.reshape(array.shape[0], -1), index, axis=1)


class WrfColumnSeat:
    """One forecast's NN seat on the WRF route."""

    def __init__(self, state, cfg, settings: NNSettings, *, resume: bool = False,
                 couple: Callable | None = None, tap=None):
        self.settings = settings
        self.scheme = settings.scheme
        nz, ny, nx = state.p.shape
        self.shape = (nz, ny, nx)
        self.levels = nz if settings.levels is None else settings.levels
        if self.levels > nz:
            raise NNConfigError(f"--nn levels = {self.levels}, but the domain has {nz} model levels")
        self.input_names = settings.inputs or DEFAULT_INPUTS[self.scheme]
        self.output_names = settings.outputs or DEFAULT_OUTPUTS[self.scheme]
        unknown = [n for n in self.input_names if n not in VOCABULARY]
        if unknown:
            raise NNConfigError(f"--nn inputs {unknown} are not in the vocabulary {sorted(VOCABULARY)}")
        unknown = [n for n in self.output_names if n not in OUTPUTS]
        if unknown:
            raise NNConfigError(f"--nn outputs {unknown} are not tendencies the seat holds {sorted(OUTPUTS)}")
        self._check_available(state, cfg)
        self.inputs = tuple(FieldSpec(n, VOCABULARY[n][0], self.levels if VOCABULARY[n][1] == "3d" else 1)
                            for n in self.input_names)
        self.outputs = tuple(FieldSpec(n, OUTPUTS[n][0], self.levels) for n in self.output_names)
        self.pool = interior_pool(cfg, ny, nx)
        self.ncol = min(settings.tap_columns, self.pool.size)
        self.rng = np.random.default_rng(settings.tap_seed)
        if couple is None:
            from woof.core.physics import couple_column_tendencies as couple
        self._couple = couple
        self.calls = 0
        self.tap = tap
        if self.tap is None and settings.mode == "tap":
            from woof.nn.tap import TendencyTap
            self.tap = TendencyTap(
                settings.tap, inputs=self.inputs, outputs=self.outputs, columns=self.ncol, resume=resume,
                attrs={"route": "wrf", "scheme": self.scheme, "grid": [nz, ny, nx], "dt": float(cfg.dt),
                       "levels": self.levels, "tap_every": settings.tap_every,
                       "tap_seed": settings.tap_seed})

    # ------------------------------------------------------------ checks
    def _check_available(self, state, cfg) -> None:
        from woof.config import radiation_enabled
        from woof.core.physics_inventory import physics_enabled
        if not physics_enabled(cfg) or getattr(state, "physics", None) is None:
            raise NNConfigError("--nn needs the non-microphysics physics on: this configuration runs none")
        active = {"pbl": bool(cfg.bl_pbl_physics), "radiation": radiation_enabled(cfg),
                  "cumulus": bool(cfg.cu_physics), "total": True}
        if not active[self.scheme]:
            raise NNConfigError(f"--nn scheme = {self.scheme!r}, but this configuration runs no {self.scheme} scheme")
        for name in self.input_names:
            if name in ("qv", "qc", "qr", "qi", "qs", "qg") and getattr(state, name, None) is None:
                raise NNConfigError(f"--nn input {name!r}: this configuration carries no {name}")
            if name in SURFACE and name not in state.physics.fields:
                raise NNConfigError(f"--nn input {name!r}: the physics carries no {name} field here; "
                                    f"surface fields present: {sorted(k for k in SURFACE if k in state.physics.fields)}")
            if name == "rthraten" and not radiation_enabled(cfg):
                raise NNConfigError("--nn input 'rthraten': this configuration runs no radiation")

    # ------------------------------------------------------------ columns
    def draw(self) -> np.ndarray:
        """The next recorded step's columns (flat mass-grid indices, sorted)."""
        return np.sort(self.rng.choice(self.pool, size=self.ncol, replace=False))

    def _tendencies(self, state, tendencies):
        if self.scheme == "total":
            return tendencies
        return getattr(state.physics, SCHEME_SLOT[self.scheme])

    def columns(self, state, cfg, tendencies, flat: np.ndarray):
        """``(inputs, targets)`` for the columns ``flat``: device arrays
        ``[n_inputs, column]`` and ``[n_outputs, column]``, float32."""
        xp = _xp(state)
        nz, ny, nx = self.shape
        L = self.levels
        g = _Gather.build(xp, flat, nx)
        f32 = np.float32
        unit = None

        def unit_response():
            nonlocal unit
            if unit is None:
                ones = xp.ones(self.shape, dtype=f32)
                coupled = self._couple(state, cfg, rtheta=ones, rqv=ones)
                unit = (g.take(coupled.rtheta, levels=L), g.take(coupled.rqv, levels=L))
            return unit

        geo = None

        def geopotential():
            nonlocal geo
            if geo is None:
                phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
                ph = g.take(xp.ascontiguousarray(phb + state.php), levels=L + 1)
                geo = ph
            return geo

        rows = []
        for name in self.input_names:
            if name == "theta":
                rows.append(g.take(xp.ascontiguousarray(state.total_theta()), levels=L))
            elif name in ("qv", "qc", "qr", "qi", "qs", "qg", "p"):
                rows.append(g.take(getattr(state, name), levels=L))
            elif name == "exner":
                rows.append((g.take(state.p, levels=L) / f32(P0)) ** f32(RD_OVER_CP))
            elif name == "u":
                rows.append(f32(0.5) * (g.take(state.u, g.u_west, L) + g.take(state.u, g.u_east, L)))
            elif name == "v":
                rows.append(f32(0.5) * (g.take(state.v, g.v_south, L) + g.take(state.v, g.v_north, L)))
            elif name == "w":
                w = g.take(state.w, levels=L + 1)
                rows.append(f32(0.5) * (w[:-1] + w[1:]))
            elif name == "z":
                ph = geopotential()
                rows.append(f32(0.5) * (ph[:-1] + ph[1:]) / f32(G) - g.take(state.ht))
            elif name == "dz":
                ph = geopotential()
                rows.append((ph[1:] - ph[:-1]) / f32(G))
            elif name == "rthraten":
                rows.append(g.take(state.physics.radiation_tendencies.rtheta, levels=L) / unit_response()[0])
            elif name == "hgt":
                rows.append(g.take(state.ht))
            elif name == "mu":
                rows.append(g.take(state.total_mu()))
            else:
                rows.append(g.take(state.physics.fields[name]))
        held = self._tendencies(state, tendencies)
        targets = []
        for name in self.output_names:
            slot = OUTPUTS[name][1]
            value = getattr(held, slot)
            if value is None:
                targets.append(xp.zeros((L, flat.size), dtype=f32))
            elif slot == "ru":
                targets.append(f32(0.5) * (g.take(value, g.u_west, L) + g.take(value, g.u_east, L))
                               / unit_response()[1])
            elif slot == "rv":
                targets.append(f32(0.5) * (g.take(value, g.v_south, L) + g.take(value, g.v_north, L))
                               / unit_response()[1])
            elif slot == "rtheta":
                targets.append(g.take(value, levels=L) / unit_response()[0])
            else:
                targets.append(g.take(value, levels=L) / unit_response()[1])
        x = xp.concatenate([r.astype(f32, copy=False) for r in rows], axis=0)
        y = xp.concatenate([t.astype(f32, copy=False) for t in targets], axis=0)
        return x, y

    # ------------------------------------------------------------ the seat
    def after_nonmicrophysics(self, state, cfg, tendencies):
        if tendencies is None:
            raise RuntimeError("--nn: the step held no physics tendencies (physics did not run)")
        self.calls += 1
        if (self.calls - 1) % self.settings.tap_every:
            return tendencies
        flat = self.draw()
        x, y = self.columns(state, cfg, tendencies, flat)
        elapsed = float(getattr(state, "elapsed_seconds", 0.0))
        self.tap.append(_host(x), _host(y), time_seconds=elapsed,
                        step=int(round(elapsed / float(cfg.dt))), column_index=flat)
        return tendencies

    def close(self) -> dict:
        if self.tap is not None:
            self.tap.close()
        return {"calls": self.calls, "samples": 0 if self.tap is None else self.tap.samples,
                "store": None if self.tap is None else str(self.tap.path)}


def attach(state, cfg, *, streamed: bool = False, resume: bool = False, settings: NNSettings | None = None):
    """Attach the seat ``--nn`` asks for to ``state``; None without ``--nn``."""
    if settings is None:
        path = configured()
        if path is None:
            return None
        settings = read_settings(path)
    if streamed or getattr(state, "_streamed_domain", None) is not None:
        refuse_streamed()
    seat = WrfColumnSeat(state, cfg, settings, resume=resume)
    state._nn_column_seat = seat
    return seat
