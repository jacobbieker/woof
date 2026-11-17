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
its exact inverse on the interior.  ``du``/``dv`` divide each face by its
own unit response and average the pair: C-grid to A-grid, approximate (it
smooths at the grid scale).

When.  A scheme's slot holds the tendency of its last call (``radt``,
``cudt``, ``bldt`` > 0 hold it over steps), so a scheme is recorded only on
a step on which it ran (its ``PhysicsDriver.call_counts`` counter moved),
and then only on steps that are multiples of ``tap_every``.  ``total`` is
recorded on every such step; it sums slots held from different steps.

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
#: PhysicsDriver.call_counts key each scheme's call increments ("ysu" is the
#: historical name of the PBL-slot counter every PBL scheme increments).
SCHEME_COUNTER = {"pbl": "ysu", "radiation": "radiation", "cumulus": "cumulus"}


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
        # A scheme's slot holds the tendency of its LAST call (radt, cudt,
        # bldt > 0 hold it over steps); a sample pairs it with the inputs
        # of the step it was computed on, so only a step on which the
        # scheme ran is recorded.  Unknown until the first step after a
        # resume whose counters a restore has not yet rewritten.
        self._last_count = None if resume else self._count(state)
        self._resume = resume
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
    def draw(self, step: int) -> np.ndarray:
        """Step ``step``'s columns (flat mass-grid indices, sorted): a
        function of the seed and the step alone, so a resumed run draws
        what the unbroken run draws."""
        rng = np.random.default_rng([self.settings.tap_seed, int(step)])
        return np.sort(rng.choice(self.pool, size=self.ncol, replace=False))

    def _count(self, state):
        if self.scheme == "total":
            return None
        return state.physics.call_counts.get(SCHEME_COUNTER[self.scheme])

    def _ran(self, state) -> bool:
        """Whether the scheme ran on this step (always, for ``total``)."""
        count = self._count(state)
        previous, self._last_count = self._last_count, count
        if self.scheme == "total":
            return True
        return previous is not None and count != previous

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
                coupled = self._couple(state, cfg, rtheta=ones, rqv=ones, ru=ones, rv=ones)
                unit = {"rtheta": g.take(coupled.rtheta, levels=L), "rq": g.take(coupled.rqv, levels=L),
                        "ru": coupled.ru, "rv": coupled.rv}
            return unit

        def faces(value, slot, first, second):
            """Each face over its own unit response, then the faces'
            mean: C-grid to A-grid.  A face the coupler holds at zero (an
            open or specified edge) drops out of the mean."""
            ratios = []
            for index in (first, second):
                response = g.take(unit_response()[slot], index, L)
                safe = xp.where(response != 0, response, f32(1))
                ratios.append((xp.where(response != 0, g.take(value, index, L) / safe, f32(0)),
                               (response != 0).astype(f32)))
            total = ratios[0][0] + ratios[1][0]
            count = ratios[0][1] + ratios[1][1]
            return total / xp.maximum(count, f32(1))

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
                rows.append(g.take(state.physics.radiation_tendencies.rtheta, levels=L)
                            / unit_response()["rtheta"])
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
                targets.append(faces(value, "ru", g.u_west, g.u_east))
            elif slot == "rv":
                targets.append(faces(value, "rv", g.v_south, g.v_north))
            elif slot == "rtheta":
                targets.append(g.take(value, levels=L) / unit_response()["rtheta"])
            else:
                targets.append(g.take(value, levels=L) / unit_response()["rq"])
        x = xp.concatenate([r.astype(f32, copy=False) for r in rows], axis=0)
        y = xp.concatenate([t.astype(f32, copy=False) for t in targets], axis=0)
        return x, y

    # ------------------------------------------------------------ the seat
    def after_nonmicrophysics(self, state, cfg, tendencies):
        if tendencies is None:
            raise RuntimeError("--nn: the step held no physics tendencies (physics did not run)")
        self.calls += 1
        ran = self._ran(state)
        elapsed = float(getattr(state, "elapsed_seconds", 0.0))
        step = int(round(elapsed / float(cfg.dt)))
        if not ran or step % self.settings.tap_every:
            return tendencies
        if self._resume:
            # The record keeps what was written before this step; samples
            # flushed after the checkpoint the run resumed from are the
            # unbroken run's and are written again from here.
            self.tap.truncate(step)
            self._resume = False
        flat = self.draw(step)
        x, y = self.columns(state, cfg, tendencies, flat)
        self.tap.append(_host(x), _host(y), time_seconds=elapsed, step=step, column_index=flat)
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
