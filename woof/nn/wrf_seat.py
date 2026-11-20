"""The WRF route's NN column seat (plan section 4.1, P2.2).

``woof.core.dycore._step_body`` calls :meth:`WrfColumnSeat.after_nonmicrophysics`
once per dycore step, on the time-t state, right after the ensemble
stochastic binding (SPPT/SKEBS) has acted on the step's held
non-microphysics tendencies and before chemistry, IAU and FDDA fold theirs
in.  An applying seat is refused on a state that carries a stochastic
binding: a replacement there would keep the scheme's perturbation and drop
the model's (``total`` would drop SPPT altogether), and only ensemble
member legs, which refuse ``--nn``, carry one today.  A state without
``_nn_column_seat`` (every forecast without ``--nn``) skips the call; the
default path is unchanged.

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
and then only on every ``tap_every``-th dycore step the seat has seen
(``step`` in the record counts those; an adaptive clock's ``cfg.dt`` varies,
so model time does not name a step, and a resumed run counts afresh).  ``total`` is
recorded on every such step; it sums slots held from different steps.

``mode = "tap"`` records ``(inputs, targets)`` to a
:class:`woof.nn.tap.TendencyTap` and returns the step's tendencies object
untouched, so a tapped forecast is byte-identical to the same forecast
without ``--nn``.

``mode = "residual"`` / ``"replace"`` run a column model (the artefact names
the inputs, outputs and levels) on every column, every dycore step, from
the time-t state, so a resumed run computes what the unbroken one does and
the seat carries nothing across a checkpoint.  The safety policy screens
the inputs against the trained range (a fallback column keeps the scheme's
tendency), limits the outputs, runs the column fixer and the positivity
floor (:class:`woof.nn.safety.SafetyPolicy`).  The increment -- the model's
rate for a residual, the model's rate less the scheme's for a replacement --
is coupled by :func:`woof.core.physics.couple_column_tendencies`, the
couplers' own path, and added to a shallow copy of the step's tendencies:
the scheme's slots are never written.  Momentum goes through the A-grid:
a replaced ``du``/``dv`` is the scheme's faces plus the coupled difference
of the model's and the scheme's face means.  Such a run is NOT
WRF-identical (:data:`woof.nn.NOT_WRF_IDENTICAL`).
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from woof.nn import NOT_WRF_IDENTICAL
from woof.nn.launch import (NNConfigError, NNSettings, check_intended_mode, configured, read_settings,
                            refuse_identity_door, refuse_streamed)
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

    @classmethod
    def full(cls, xp, ny: int, nx: int):
        """Every column, in ``j * nx + i`` order: the gathers are slices
        (no index arrays, no gather launches)."""
        return _FullGrid(xp, "mass", "u_west", "u_east", "v_south", "v_north", ny, nx)

    def take(self, array, index=None, levels=None):
        index = self.mass if index is None else index
        if array.ndim == 2:
            return self.xp.take(array.reshape(1, -1), index, axis=1)
        if levels is not None:
            array = array[:levels]
        return self.xp.take(array.reshape(array.shape[0], -1), index, axis=1)


@dataclass
class _FullGrid(_Gather):
    ny: int = 0
    nx: int = 0

    def take(self, array, index=None, levels=None):
        index = "mass" if index is None else index
        if levels is not None and array.ndim == 3:
            array = array[:levels]
        if array.ndim == 2:
            array = array[None]
        ny, nx = self.ny, self.nx
        if index == "u_west":
            array = array[:, :, :nx]
        elif index == "u_east":
            array = array[:, :, 1:]
        elif index == "v_south":
            array = array[:, :ny, :]
        elif index == "v_north":
            array = array[:, 1:, :]
        return self.xp.ascontiguousarray(array).reshape(array.shape[0], ny * nx)


class WrfColumnSeat:
    """One forecast's NN seat on the WRF route."""

    def __init__(self, state, cfg, settings: NNSettings, *, resume: bool = False,
                 couple: Callable | None = None, tap=None):
        self.settings = settings
        self.scheme = settings.scheme
        self.applies = settings.applies
        if settings.tap_target != "physics":
            raise NNConfigError(
                f"--nn tap_target = {settings.tap_target!r} on the WRF route: the nudging-increment target is "
                "recorded on hex (woof hex forecast --nudging); the WRF route's tap records the scheme's tendency")
        nz, ny, nx = state.p.shape
        self.shape = (nz, ny, nx)
        self.model = self.manifest = self.policy = None
        if self.applies:
            refuse_identity_door(f"--nn {settings.source}: mode = {settings.mode!r}")
            self._load_model(settings, nz)
        else:
            self.levels = nz if settings.levels is None else settings.levels
            self.input_names = settings.inputs or DEFAULT_INPUTS[self.scheme]
            self.output_names = settings.outputs or DEFAULT_OUTPUTS[self.scheme]
        if self.levels > nz:
            raise NNConfigError(f"--nn levels = {self.levels}, but the domain has {nz} model levels")
        unknown = [n for n in self.input_names if n not in VOCABULARY]
        if unknown:
            raise NNConfigError(f"--nn inputs {unknown} are not in the vocabulary {sorted(VOCABULARY)}")
        unknown = [n for n in self.output_names if n not in OUTPUTS]
        if unknown:
            raise NNConfigError(f"--nn outputs {unknown} are not tendencies the seat holds {sorted(OUTPUTS)}")
        self._check_available(state, cfg)
        if self.applies:
            if getattr(state, "_ensemble_stochastic", None) is not None:
                raise NotImplementedError(
                    f"--nn mode = {settings.mode!r} with SPPT/SKEBS: the seat acts after the stochastic "
                    "binding, so the model's rate would not be perturbed as the scheme's is")
            missing = [n for n in self.output_names
                       if n.startswith("dq") and getattr(state, n[1:], None) is None]
            if missing:
                raise NNConfigError(f"--nn model {self.manifest.name}: outputs {missing} name moisture this "
                                    "configuration does not carry")
        self.inputs = tuple(FieldSpec(n, VOCABULARY[n][0], self.levels if VOCABULARY[n][1] == "3d" else 1)
                            for n in self.input_names)
        self.outputs = tuple(FieldSpec(n, OUTPUTS[n][0], self.levels) for n in self.output_names)
        if self.applies and (self.inputs != tuple(self.manifest.inputs)
                             or self.outputs != tuple(self.manifest.outputs)):
            raise NNConfigError(
                f"--nn model {self.manifest.name}: its fields {[(f.name, f.units, f.levels) for f in self.manifest.inputs]}"
                f" -> {[(f.name, f.units, f.levels) for f in self.manifest.outputs]} do not match the seat's "
                f"vocabulary units {[(f.name, f.units, f.levels) for f in self.inputs]} -> "
                f"{[(f.name, f.units, f.levels) for f in self.outputs]}")
        self.all_columns = np.arange(ny * nx, dtype=np.int64)
        self._all_gather = None
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

    def _load_model(self, settings: NNSettings, nz: int) -> None:
        from woof.nn.models import load_column_model
        from woof.nn.safety import SafetyPolicy

        self.model = load_column_model(settings.model, backend=settings.backend, device=settings.device)
        m = self.manifest = self.model.manifest
        self.input_names = tuple(f.name for f in m.inputs)
        self.output_names = tuple(f.name for f in m.outputs)
        unknown = [n for n in self.input_names if n not in VOCABULARY]
        unknown += [n for n in self.output_names if n not in OUTPUTS]
        if unknown:
            raise NNConfigError(f"--nn model {m.name}: fields {unknown} are not in the seat's vocabulary "
                                f"{sorted(VOCABULARY)} -> {sorted(OUTPUTS)}")
        depths = {f.levels for f in m.inputs if VOCABULARY[f.name][1] == "3d"} | {f.levels for f in m.outputs}
        if len(depths) != 1:
            raise NNConfigError(f"--nn model {m.name}: every 3-D field and output must span the same "
                                f"number of levels; the artefact has {sorted(depths)}")
        self.levels = depths.pop()
        check_intended_mode(m, settings)
        self.policy = SafetyPolicy(out_of_range=settings.out_of_range, margin=settings.margin,
                                   limit_outputs=settings.limit_outputs, positivity=settings.positivity,
                                   fixer=settings.fixer)

    def identity(self) -> dict | None:
        """What a checkpoint binds: None for a tap (it changes nothing)."""
        if not self.applies:
            return None
        return {**self.settings.identity(), "model": self.manifest.identity()}

    def _fallback_faces(self, xp, fallback):
        """Face masks (u ``[ny, nx+1]``, v ``[ny+1, nx]``) of the faces next
        to a fallback column: the A-grid to C-grid coupling averages a
        face's two cells, so a face beside a fallback column is given the
        scheme's word alone."""
        nz, ny, nx = self.shape
        bad = fallback.reshape(ny, nx)
        no = xp.zeros((ny, 1), dtype=bool)
        u = xp.concatenate([bad, no], axis=1) | xp.concatenate([no, bad], axis=1)
        no = xp.zeros((1, nx), dtype=bool)
        v = xp.concatenate([bad, no], axis=0) | xp.concatenate([no, bad], axis=0)
        return u, v

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
    def draw(self, elapsed: float) -> np.ndarray:
        """The columns drawn at model time ``elapsed`` seconds (flat
        mass-grid indices, sorted): a function of the seed and the time
        alone, so a resumed run draws what the unbroken run draws."""
        rng = np.random.default_rng([self.settings.tap_seed, int(round(float(elapsed) * 1000.0))])
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

    def columns(self, state, cfg, tendencies, flat: np.ndarray, extras: bool = False):
        """``(inputs, targets)`` for the columns ``flat``: device arrays
        ``[n_inputs, column]`` and ``[n_outputs, column]``, float32.  With
        ``extras``, also a mapping holding the layer mass per unit area
        ``weight`` (``(c1h mu + c2h) |dnw| / g``, zero in a masked ring)
        and the ``exner`` and moisture columns the safety policy reads."""
        xp = _xp(state)
        nz, ny, nx = self.shape
        L = self.levels
        if flat is self.all_columns:  # an applying seat's every step: build the indices once
            if self._all_gather is None:
                self._all_gather = _Gather.full(xp, ny, nx)
            g = self._all_gather
        else:
            g = _Gather.build(xp, flat, nx)
        f32 = np.float32
        unit = None

        def unit_response():
            nonlocal unit
            if unit is None:
                ones = xp.ones(self.shape, dtype=f32)
                coupled = self._couple(state, cfg, rtheta=ones, rqv=ones, ru=ones, rv=ones)
                theta, q = g.take(coupled.rtheta, levels=L), g.take(coupled.rqv, levels=L)
                # A masked ring column has a zero response and zero held
                # words; dividing by one there keeps it zero, not NaN.
                unit = {"rtheta": xp.where(theta != 0, theta, f32(1)), "rq": xp.where(q != 0, q, f32(1)),
                        "mass": q, "ru": coupled.ru, "rv": coupled.rv}
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
        if not extras:
            return x, y
        dnw = xp.abs(xp.asarray(state.dnw[:L], dtype=f32))[:, None]
        more = {"weight": unit_response()["mass"] * dnw / f32(G),
                "exner": (g.take(state.p, levels=L) / f32(P0)) ** f32(RD_OVER_CP)}
        for name in self.output_names:
            field = name[1:]
            if field.startswith("q") and getattr(state, field, None) is not None:
                more[field] = g.take(getattr(state, field), levels=L)
        return x, y, more

    # ------------------------------------------------------------ the seat
    def after_nonmicrophysics(self, state, cfg, tendencies):
        if tendencies is None:
            raise RuntimeError("--nn: the step held no physics tendencies (physics did not run)")
        if self.applies:
            self.calls += 1
            return self.apply(state, cfg, tendencies)
        # The seat's own step count: an adaptive clock changes cfg.dt from
        # step to step, so model time does not name a step.
        step = self.calls
        self.calls += 1
        ran = self._ran(state)
        elapsed = float(getattr(state, "elapsed_seconds", 0.0))
        if not ran or step % self.settings.tap_every:
            return tendencies
        if self._resume:
            # The record keeps what was written before this time; samples
            # flushed after the checkpoint the run resumed from are the
            # unbroken run's and are written again from here.
            self.tap.truncate(elapsed)
            self._resume = False
        flat = self.draw(elapsed)
        x, y = self.columns(state, cfg, tendencies, flat)
        self.tap.append(_host(x), _host(y), time_seconds=elapsed, step=step, column_index=flat)
        return tendencies

    def apply(self, state, cfg, tendencies):
        """The step's tendencies with the model's increment added (a new
        object; the step's own is not written)."""
        xp = _xp(state)
        f32 = np.float32
        nz, ny, nx = self.shape
        L, m = self.levels, self.manifest
        x, scheme, more = self.columns(state, cfg, tendencies, self.all_columns, extras=True)
        x, fallback = self.policy.screen(x, m, feature_names=[f"{f.name}" for f in m.inputs
                                                              for _ in range(f.levels)])
        y = self.model(x)
        replace = self.settings.mode == "replace"
        weight = more["weight"]
        targets = None
        if replace and self.policy.fixer != "none":
            targets = {}
            for spec, rows in m.output_slices().items():
                if spec.startswith("dq"):
                    targets[spec] = (weight * scheme[rows]).sum(axis=0)
                elif spec == "dtheta" and self.policy.fixer == "moisture+heat":
                    targets[spec] = (weight * more["exner"] * scheme[rows]).sum(axis=0)
        dt = float(cfg.dt)
        floor = more
        if not replace:
            # A residual adds to the scheme's rate: the positivity floor
            # holds for the sum, so it is taken from the moisture the
            # scheme's rate alone would leave after the step.
            floor = dict(more)
            for spec, rows in m.output_slices().items():
                field = spec[1:]
                if spec.startswith("dq") and field in more:
                    floor[field] = more[field] + f32(dt) * scheme[rows]
        y = self.policy.constrain(y, m, dt=dt, state=floor, weight=weight, targets=targets)
        delta = y - scheme if replace else y
        if fallback is not None:
            delta = xp.where(fallback[None, :], f32(0), delta)
        rates = {}
        pad = xp.zeros((nz - L, ny * nx), dtype=f32) if nz > L else None
        for name, rows in m.output_slices().items():
            value = delta[rows]
            if pad is not None:
                value = xp.concatenate([value, pad], axis=0)
            rates[OUTPUTS[name][1]] = xp.ascontiguousarray(value.reshape(nz, ny, nx))
        momentum = {}
        if "ru" in rates or "rv" in rates:
            zero = xp.zeros((nz, ny, nx), dtype=f32)
            momentum = {"ru": rates.pop("ru", zero), "rv": rates.pop("rv", zero)}
        coupled = self._couple(state, cfg, **rates, **momentum)
        faces = None
        if momentum and fallback is not None:
            faces = dict(zip(("ru", "rv"), self._fallback_faces(xp, fallback)))
        out = copy.copy(tendencies)
        for slot in (OUTPUTS[name][1] for name in self.output_names):
            held = getattr(tendencies, slot)
            if held is None:
                raise RuntimeError(f"--nn model {m.name} writes {slot}, which this run's tendencies do not "
                                   "carry (no scheme produces it)")
            increment = getattr(coupled, slot)
            if faces is not None and slot in faces:
                increment = xp.where(faces[slot][None], f32(0), increment)
            setattr(out, slot, held + increment)
        return out

    def close(self) -> dict:
        if self.tap is not None:
            self.tap.close()
        summary = {"mode": self.settings.mode, "scheme": self.scheme, "wrf_identical": not self.applies,
                   "calls": self.calls, "samples": 0 if self.tap is None else self.tap.samples,
                   "store": None if self.tap is None else str(self.tap.path)}
        if self.applies:
            summary.update(model=self.manifest.name, identity=self.identity(), runner=self.model.backend,
                           note=NOT_WRF_IDENTICAL, **self.policy.ledger.summary())
            print(f"nn: {self.settings.mode} of {self.scheme} by {self.manifest.name} "
                  f"({self.model.backend}) over {self.calls} steps; "
                  f"{summary['columns_out_of_range']} of {summary['columns_seen']} column calls outside the "
                  f"trained range ({self.settings.out_of_range}); {NOT_WRF_IDENTICAL}",
                  file=sys.stderr, flush=True)
        return summary


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
