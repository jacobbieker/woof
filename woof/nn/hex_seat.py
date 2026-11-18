"""The hex (MPAS-route) NN column seat (plan section 4.1, P2.4).

Where.  :meth:`woof.hex.mpasphys.runtime_v842.ResidentNativeCoupling.stage`
assembles a step's held physics sources once, at the first RK stage, from
``physics_get_tend`` (``mpas_atmphys_todynamics.F``).  The seat acts on them
right after the regional padding and BEFORE SPPT/SKEB, IAU and analysis
nudging fold theirs in:

* the tap records the schemes' own sum, unperturbed, which is the target a
  model is trained to reproduce;
* an applying seat changes the physics source, and SPPT then perturbs the
  physics source the step holds -- the scheme's plus the model's -- as it
  perturbs any scheme's, so the WRF seat's refusal of SPPT/SKEBS (it acts
  after the stochastic binding, where a replacement would drop the
  perturbation) does not apply here.  SKEB reads the momentum source,
  which the seat does not write.  IAU and nudging are not physics and are
  never seen by the model.

A forecast without ``--nn`` never constructs this class and imports
nothing from :mod:`woof.nn` (``tests/test_nn_default_path_imports.py``).

What the held sources are.  For every real cell the stage holds
(``m`` = ``rho_zz``, the metric dry density, ``state.rho``):

* ``scalars[q]   = m * dq/dt`` for each moisture species, kg kg-1 s-1;
* ``rho_theta    = m * (c * dtheta/dt + (Rv/Rd) * theta * dqv/dt)``, where
  ``c = 1 + (Rv/Rd) qv`` and ``theta = theta_m / c`` is the dry potential
  temperature: MPAS couples a dry-theta rate and the vapour rate into the
  modified potential temperature ``theta_m = theta * c``
  (``mpas_rk_cells_v842_f32``; the WOOF route's ``couple_cells_v841_f32``
  is the same formula), so ``rho_theta / m`` is ``dtheta_m/dt``;
* ``rho_u`` (edge-normal momentum) and ``rho`` (zero).

``scheme = "total"`` is that whole non-microphysics source (PBL, surface,
cumulus, radiation, gravity-wave drag, shallow cumulus); the lagged
microphysics heating is held apart (``diabatic_theta``) and is not part of
it.  The schemes' individual rates are not separable at this seat, so
``pbl``/``radiation``/``cumulus`` are refused.

Decoupling.  The seat divides by ``m`` and inverts the theta conversion
with the cell's own ``theta_m`` and ``qv`` (the words the coupler read):
``dqv = S_qv / m``, ``dtheta_m = S_theta / m``, ``dtheta = (dtheta_m -
(Rv/Rd) theta dqv) / c``.  Outputs (``OUTPUTS``) may name either
``dtheta`` (dry, K s-1) or ``dtheta_m`` (modified, K s-1), not both.

Coupling an increment.  A moisture increment ``d`` adds ``m * d`` to its
species' source.  A ``dtheta`` increment adds ``m * (c * d + (Rv/Rd) theta
* dqv_increment)``: the model's dry-theta rate is held as the model says
and its vapour rate enters ``theta_m`` as the scheme's does.  A model
without a theta output that changes ``dqv`` therefore still writes
``rho_theta`` (dry theta unchanged).  A ``dtheta_m`` increment adds ``m *
d`` alone: it already is the modified-theta rate.  Momentum (``du``,
``dv``) is refused: the held momentum source is edge-normal and this seat
has no cell-to-edge projection yet.

Columns.  The tap draws ``tap_columns`` cells per recorded step uniformly
from the interior (``bdyMaskCell == 0`` on a limited-area mesh; the
padded garbage cell is never a column), keyed by the seed and the model
time alone, so a resumed run draws what the unbroken one does.  An
applying seat runs every real cell outside the specified zone
(``bdyMaskCell < 6``, the cells analysis nudging writes); the specified
zone takes the lateral-boundary tendencies and is left alone.

Refused: an applying mode under ``GPUWM_WRF_EXACT=1`` or ``physics.mode =
exact`` (``--physics-mode exact``), a same-host partition or LTS execution
(``execution`` is set), a moving region or a resume remapped onto another
mesh, and (at the door) ensembles and the WOOF-physics route.

Restart.  An applying seat's identity -- the settings and the artefact's
hashes -- is sealed in the route's carrier (``nn``, written by
:meth:`woof.hex.mpasphys.moving_v842.MovingRoute.carrier_payload`) and
checked on resume by :func:`check_carried` (the hex counterpart of
:func:`woof.io.restart._validate_nn_column_checkpoint`).  A tap binds
nothing; a resumed tap trims its record at the first recorded time and
appends.  The seat carries no state across a step: an applying seat
recomputes its increment from the time-t state every step.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Mapping

import numpy as np

from woof.nn import NOT_WRF_IDENTICAL
from woof.nn.launch import NNConfigError, NNSettings, configured, read_settings, refuse_identity_door
from woof.nn.manifest import FieldSpec

#: MPAS's ``rvord`` in float32, as the couplers take it (461.6 / 287).
RV_OVER_RD = np.float32(np.float32(461.6) / np.float32(287.0))
#: bdyMaskCell of the first specified-zone ring (woof.hex.nudging.SPECIFIED_ZONE_FIRST_RING).
SPECIFIED_ZONE_FIRST_RING = 6
MOISTURE = ("qv", "qc", "qr", "qi", "qs", "qg")

# name -> (units, kind); kind "3d" spans the model levels, "2d" one value
# per cell.  Levels are bottom-up (level 0 is the lowest model level).
VOCABULARY: dict[str, tuple[str, str]] = {
    "theta": ("K", "3d"),          # dry potential temperature, theta_m / (1 + Rv/Rd qv)
    "theta_m": ("K", "3d"),        # MPAS modified potential temperature
    "qv": ("kg kg-1", "3d"), "qc": ("kg kg-1", "3d"), "qr": ("kg kg-1", "3d"),
    "qi": ("kg kg-1", "3d"), "qs": ("kg kg-1", "3d"), "qg": ("kg kg-1", "3d"),
    "u": ("m s-1", "3d"),          # reconstructed cell zonal wind (uReconstructZonal)
    "v": ("m s-1", "3d"),          # reconstructed cell meridional wind (uReconstructMeridional)
    "w": ("m s-1", "3d"),          # vertical velocity, mean of the layer's two faces
    "p": ("Pa", "3d"),             # full pressure (diag pressure)
    "exner": ("1", "3d"),
    "rho": ("kg m-3", "3d"),       # dry density, zz * rho_zz
    "z": ("m", "3d"),              # layer-centre height above the ground
    "dz": ("m", "3d"),             # layer thickness
    "hgt": ("m", "2d"),            # terrain height (zgrid at the ground)
    "hfx": ("W m-2", "2d"), "qfx": ("kg m-2 s-1", "2d"), "lh": ("W m-2", "2d"),
    "ust": ("m s-1", "2d"), "znt": ("m", "2d"), "pblh": ("m", "2d"),
    "tsk": ("K", "2d"), "xland": ("1", "2d"), "psfc": ("Pa", "2d"),
}
#: Vocabulary names read from the physics session's pools: (pool, name).
POOL_FIELDS: dict[str, tuple[str, str]] = {
    "p": ("diag", "pressure"), "u": ("diag", "uReconstructZonal"), "v": ("diag", "uReconstructMeridional"),
    "hfx": ("diag_physics", "hfx"), "qfx": ("diag_physics", "qfx"), "lh": ("diag_physics", "lh"),
    "ust": ("diag_physics", "ust"), "znt": ("diag_physics", "znt"), "pblh": ("diag_physics", "hpbl"),
    "tsk": ("sfc_input", "skintemp"), "xland": ("sfc_input", "xland"), "psfc": ("diag", "surface_pressure"),
}
OUTPUTS: dict[str, str] = {
    "dtheta": "K s-1", "dtheta_m": "K s-1",
    **{f"d{q}": "kg kg-1 s-1" for q in MOISTURE},
}
MOMENTUM = ("du", "dv")
DEFAULT_INPUTS = ("theta", "qv", "qc", "u", "v", "w", "p", "z", "hfx", "qfx", "ust", "tsk", "xland")
DEFAULT_OUTPUTS = ("dtheta", "dqv", "dqc")


# ---------------------------------------------------------------- arithmetic
def decouple(xp, *, rho_theta, scalars: Mapping[str, Any], mass, theta_m, qv) -> dict:
    """Physical rates from held dry-mass sources (``[level, column]`` each).

    ``scalars`` maps a species to its held source.  Returns ``dtheta_m``,
    ``dtheta`` and ``d<species>`` for each species given; ``dtheta`` needs
    ``qv`` in ``scalars`` (the vapour term), else it is taken as zero."""
    f32 = np.float32
    safe = xp.where(mass != 0, mass, f32(1))
    coeff = f32(1) + RV_OVER_RD * qv
    theta = theta_m / coeff
    rates = {f"d{name}": value / safe for name, value in scalars.items()}
    dqv = rates.get("dqv")
    rates["dtheta_m"] = rho_theta / safe
    vapour = RV_OVER_RD * theta * dqv if dqv is not None else f32(0)
    rates["dtheta"] = (rates["dtheta_m"] - vapour) / coeff
    return rates


def couple_increment(xp, increments: Mapping[str, Any], *, mass, theta_m, qv) -> tuple[Any, dict]:
    """``(rho_theta increment or None, {species: scalar-source increment})``
    for physical-rate increments ``d<name>`` (``[level, column]``)."""
    f32 = np.float32
    scalars = {name[1:]: mass * value for name, value in increments.items()
               if name.startswith("dq")}
    if "dtheta" in increments and "dtheta_m" in increments:
        raise NNConfigError("a hex column model writes dtheta or dtheta_m, not both")
    coeff = f32(1) + RV_OVER_RD * qv
    theta = theta_m / coeff
    dqv = increments.get("dqv")
    if "dtheta_m" in increments:
        return mass * increments["dtheta_m"], scalars
    if "dtheta" in increments:
        vapour = RV_OVER_RD * theta * dqv if dqv is not None else f32(0)
        return mass * (coeff * increments["dtheta"] + vapour), scalars
    if dqv is not None:
        return mass * (RV_OVER_RD * theta * dqv), scalars
    return None, scalars


def interior_pool(n_cells: int, bdy_mask) -> np.ndarray:
    """Cells a tap may draw: every real cell on a closed mesh, the
    interior (``bdyMaskCell == 0``) on a limited-area one."""
    if bdy_mask is None:
        return np.arange(int(n_cells), dtype=np.int64)
    pool = np.flatnonzero(np.asarray(bdy_mask)[:int(n_cells)] == 0).astype(np.int64)
    if pool.size == 0:
        raise NNConfigError("the limited-area mesh has no interior cell (bdyMaskCell == 0) to record")
    return pool


def apply_mask(n_cells: int, bdy_mask) -> np.ndarray | None:
    """float32 ``[cell]``: 1 where an applying seat writes (outside the
    specified zone), None when that is every real cell."""
    if bdy_mask is None:
        return None
    keep = np.asarray(bdy_mask)[:int(n_cells)] < SPECIFIED_ZONE_FIRST_RING
    return None if keep.all() else keep.astype(np.float32)


def check_carried(identity: dict | None, carried: dict | None) -> None:
    """A resume's seat against the identity its carrier holds (both None:
    neither applied a model)."""
    if carried is None and identity is None:
        return
    if carried is None:
        raise NNConfigError(
            "this checkpoint was written without a neural-network column model, and the launch applies "
            f"one ({identity['mode']} of {identity['scheme']} by {identity['model']['name']}); resuming "
            "would splice two trajectories. Resume without --nn, or with mode = \"tap\"")
    if identity is None:
        raise NNConfigError(
            "this checkpoint was written by a run that applied a neural-network column model "
            f"({carried.get('mode')} of {carried.get('scheme')} by {(carried.get('model') or {}).get('name')}); "
            "resume it with the same --nn file")
    if carried != identity:
        changed = sorted(k for k in set(carried) | set(identity) if carried.get(k) != identity.get(k))
        raise NNConfigError(f"this checkpoint's neural-network column model differs from the launch's in "
                            f"{changed}; resume it with the artefact and settings that wrote it")


# ---------------------------------------------------------------- the seat
class HexColumnSeat:
    """One hex forecast's NN seat on the held physics sources.

    ``fields(pool, name)`` returns a physics-pool field over the real cells
    (``ResidentNativeCoupling.session.field``).  ``zgrid`` is the
    interface heights ``[level + 1, cell]``, ``zz`` MPAS's metric ``[level,
    cell]``; ``bdy_mask`` is the mesh's ``bdyMaskCell`` (None on a closed
    mesh).  ``mode`` is the physics mode (``exact`` or ``correct``)."""

    def __init__(self, settings: NNSettings, *, cp, names, n_cells: int, n_levels: int, zgrid, zz=None,
                 bdy_mask=None, dt: float, mode: str = "correct", fields: Callable | None = None,
                 resume: bool = False, tap=None):
        self.settings = settings
        self.cp = cp
        self.names = tuple(names)
        self.nc = int(n_cells)
        self.nz = int(n_levels)
        self.dt = float(dt)
        self.applies = settings.applies
        self._fields = fields
        if settings.scheme != "total":
            raise NNConfigError(
                f"--nn scheme = {settings.scheme!r} on hex: the MPAS route's held sources are the sum of every "
                "non-microphysics scheme (physics_get_tend), so only scheme = \"total\" is available here")
        self.model = self.manifest = self.policy = None
        if self.applies:
            refuse_identity_door(f"--nn {settings.source}: mode = {settings.mode!r}")
            if str(mode) == "exact":
                raise NNConfigError(
                    f"--nn {settings.source}: mode = {settings.mode!r} applies a neural-network column model, "
                    "which is not MPAS-identical; --physics-mode exact runs native MPAS v8.4.2 only. Run in "
                    "correct mode, or use mode = \"tap\"")
            self._load_model(settings)
        else:
            self.levels = self.nz if settings.levels is None else settings.levels
            self.input_names = settings.inputs or DEFAULT_INPUTS
            self.output_names = settings.outputs or DEFAULT_OUTPUTS
        if self.applies and settings.levels is not None and settings.levels != self.levels:
            raise NNConfigError(f"--nn levels = {settings.levels}, but the model {self.manifest.name} spans "
                                f"{self.levels}; an applying run takes its levels from the artefact")
        if self.applies and self.policy.fixer == "moisture+heat" and "dtheta_m" in self.output_names:
            raise NNConfigError("--nn fixer = \"moisture+heat\" fixes a dry-theta rate; this model writes dtheta_m")
        if self.levels > self.nz:
            raise NNConfigError(f"--nn levels = {self.levels}, but the mesh has {self.nz} model levels")
        self._check_names()
        self.inputs = tuple(FieldSpec(n, VOCABULARY[n][0], self.levels if VOCABULARY[n][1] == "3d" else 1)
                            for n in self.input_names)
        self.outputs = tuple(FieldSpec(n, OUTPUTS[n], self.levels) for n in self.output_names)
        if self.applies and (self.inputs != tuple(self.manifest.inputs)
                             or self.outputs != tuple(self.manifest.outputs)):
            raise NNConfigError(
                f"--nn model {self.manifest.name}: its fields "
                f"{[(f.name, f.units, f.levels) for f in self.manifest.inputs]} -> "
                f"{[(f.name, f.units, f.levels) for f in self.manifest.outputs]} do not match the hex seat's "
                f"vocabulary units {[(f.name, f.units, f.levels) for f in self.inputs]} -> "
                f"{[(f.name, f.units, f.levels) for f in self.outputs]}")
        f32 = np.float32
        z = np.asarray(_host(zgrid), dtype=np.float64)[:self.levels + 1, :self.nc]
        if z.shape != (self.levels + 1, self.nc):
            raise NNConfigError(f"--nn: zgrid {np.shape(zgrid)} does not cover {self.levels + 1} interfaces "
                                f"of {self.nc} cells")
        self._z = cp.asarray(((z[:-1] + z[1:]) * 0.5 - z[0]).astype(f32))
        self._dz = cp.asarray((z[1:] - z[:-1]).astype(f32))
        self._hgt = cp.asarray(z[:1].astype(f32))
        self._zz = None if zz is None else cp.ascontiguousarray(cp.asarray(zz)[:self.levels, :self.nc])
        self.pool = interior_pool(self.nc, bdy_mask)
        self.ncol = min(settings.tap_columns, self.pool.size)
        mask = apply_mask(self.nc, bdy_mask)
        self._mask = None if mask is None else cp.asarray(mask > 0)[None, :]
        self.applied_cells = self.nc if mask is None else int(mask.sum())
        self._resume = resume
        self.calls = 0
        self.tap = tap
        if self.tap is None and settings.mode == "tap":
            from woof.nn.tap import TendencyTap
            self.tap = TendencyTap(
                settings.tap, inputs=self.inputs, outputs=self.outputs, columns=self.ncol, resume=resume,
                attrs={"route": "hex", "scheme": self.scheme, "grid": [self.nz, self.nc], "dt": self.dt,
                       "levels": self.levels, "tap_every": settings.tap_every, "tap_seed": settings.tap_seed,
                       "interior_cells": int(self.pool.size),
                       "column": "0-based cell index on the run's mesh (the real cells, no padding)"})

    scheme = "total"

    def _load_model(self, settings: NNSettings) -> None:
        from woof.nn.models import load_column_model
        from woof.nn.safety import SafetyPolicy

        self.model = load_column_model(settings.model, backend=settings.backend, device=settings.device)
        m = self.manifest = self.model.manifest
        self.input_names = tuple(f.name for f in m.inputs)
        self.output_names = tuple(f.name for f in m.outputs)
        self._check_names()
        depths = {f.levels for f in m.inputs if VOCABULARY[f.name][1] == "3d"} | {f.levels for f in m.outputs}
        if len(depths) != 1:
            raise NNConfigError(f"--nn model {m.name}: every 3-D field and output must span the same "
                                f"number of levels; the artefact has {sorted(depths)}")
        self.levels = depths.pop()
        self.policy = SafetyPolicy(out_of_range=settings.out_of_range, margin=settings.margin,
                                   limit_outputs=settings.limit_outputs, positivity=settings.positivity,
                                   fixer=settings.fixer)

    def _check_names(self) -> None:
        momentum = [n for n in self.output_names if n in MOMENTUM]
        if momentum:
            raise NNConfigError(
                f"--nn outputs {momentum} on hex: the held momentum source is edge-normal and this seat has no "
                "cell-to-edge projection yet; a hex column model writes theta and moisture rates "
                f"{sorted(OUTPUTS)}")
        unknown = [n for n in self.input_names if n not in VOCABULARY]
        if unknown:
            raise NNConfigError(f"--nn inputs {unknown} are not in the hex vocabulary {sorted(VOCABULARY)}")
        unknown = [n for n in self.output_names if n not in OUTPUTS]
        if unknown:
            raise NNConfigError(f"--nn outputs {unknown} are not rates the hex seat holds {sorted(OUTPUTS)}")
        if "dtheta" in self.output_names and "dtheta_m" in self.output_names:
            raise NNConfigError("--nn outputs name dtheta and dtheta_m: a model writes one of the two")
        missing = [n for n in (*self.input_names, *(o[1:] for o in self.output_names))
                   if n in MOISTURE and n not in self.names]
        if missing:
            raise NNConfigError(f"--nn fields {sorted(set(missing))}: this configuration carries no such "
                                f"species (it carries {list(self.names)})")

    def identity(self) -> dict | None:
        """What a checkpoint binds: None for a tap (it changes nothing)."""
        if not self.applies:
            return None
        return {**self.settings.identity(), "model": self.manifest.identity()}

    # ------------------------------------------------------------ columns
    def draw(self, time_seconds: float) -> np.ndarray:
        """The cells drawn at model time ``time_seconds`` (sorted): a
        function of the seed and the time alone."""
        rng = np.random.default_rng([self.settings.tap_seed, int(round(float(time_seconds) * 1000.0))])
        return np.sort(rng.choice(self.pool, size=self.ncol, replace=False))

    def _field(self, name):
        pool, key = POOL_FIELDS[name]
        if self._fields is None:
            raise NNConfigError(f"--nn input {name!r}: this seat reads no physics pools")
        try:
            value = self._fields(pool, key)
        except Exception as error:
            raise NNConfigError(f"--nn input {name!r}: the MPAS physics carries no {pool}.{key} here "
                                f"({type(error).__name__}: {error}); name inputs this suite has") from error
        return self.cp.asarray(value)

    def columns(self, sources, *, state, saved, cells=None):
        """``(x, y, extras)``: inputs ``[n_inputs, column]``, the held
        physics as physical rates ``[n_outputs, column]`` (both float32
        device arrays) and the column state the coupling and the safety
        policy read.  ``cells`` (host int array) selects columns; None is
        every real cell."""
        cp, L, nc = self.cp, self.levels, self.nc
        f32 = np.float32
        if cells is None:
            take = lambda a: a[..., :nc]
        else:
            index = cp.asarray(np.asarray(cells, dtype=np.int32))
            take = lambda a: cp.take(a[..., :nc], index, axis=a.ndim - 1)

        def level(a):
            return take(a[:L])

        iq = {name: self.names.index(name) for name in self.names}
        mass = level(state.rho)
        theta_m = level(saved.theta_m)
        qv = level(state.scalars[iq["qv"]])
        coeff = f32(1) + RV_OVER_RD * qv
        rows = []
        for name in self.input_names:
            if name == "theta":
                rows.append(theta_m / coeff)
            elif name == "theta_m":
                rows.append(theta_m)
            elif name in MOISTURE:
                rows.append(level(state.scalars[iq[name]]))
            elif name in ("p", "u", "v"):
                rows.append(level(self._field(name)))
            elif name == "w":
                w = take(saved.vertical_velocity[:L + 1])
                rows.append(f32(0.5) * (w[:-1] + w[1:]))
            elif name == "exner":
                rows.append(level(saved.exner))
            elif name == "rho":
                if self._zz is None:
                    raise NNConfigError("--nn input 'rho' needs the vertical metric zz, which this run does not lend")
                rows.append(take(self._zz) * mass)
            elif name == "z":
                rows.append(take(self._z))
            elif name == "dz":
                rows.append(take(self._dz))
            elif name == "hgt":
                rows.append(take(self._hgt))
            else:
                rows.append(take(self._field(name).reshape(1, -1)))
        species = {name[1:] for name in self.output_names if name.startswith("dq")} | {"qv"}
        held = {q: level(sources.scalars[iq[q]]) for q in species}
        rates = decouple(cp, rho_theta=level(sources.rho_theta), scalars=held, mass=mass, theta_m=theta_m, qv=qv)
        x = cp.concatenate([r.astype(f32, copy=False) for r in rows], axis=0)
        y = cp.concatenate([rates[name].astype(f32, copy=False) for name in self.output_names], axis=0)
        extras = {"mass": mass, "theta_m": theta_m, "qv": qv}
        if self.applies:
            if self._zz is not None:
                extras["weight"] = take(self._zz) * mass * take(self._dz)
            extras["exner"] = level(saved.exner)
            for q in species:
                extras[q] = level(state.scalars[iq[q]])
        return x, y, extras

    # ------------------------------------------------------------ the seat
    def stage(self, sources, *, state, saved, time_seconds: float) -> None:
        """Record (tap) or add the model's increment to ``sources`` in place."""
        if self.applies:
            self.calls += 1
            self.apply(sources, state=state, saved=saved)
            return
        step = int(round(float(time_seconds) / self.dt)) if self.dt > 0 else self.calls
        self.calls += 1
        if step % self.settings.tap_every:
            return
        if self._resume:
            # Samples flushed after the checkpoint the run resumed from are
            # the unbroken run's: drop them and write them again from here.
            self.tap.truncate(float(time_seconds))
            self._resume = False
        cells = self.draw(time_seconds)
        x, y, _ = self.columns(sources, state=state, saved=saved, cells=cells)
        self.tap.append(_host(x), _host(y), time_seconds=float(time_seconds), step=step, column_index=cells)

    def apply(self, sources, *, state, saved) -> None:
        cp, L, nc, m = self.cp, self.levels, self.nc, self.manifest
        f32 = np.float32
        x, scheme, more = self.columns(sources, state=state, saved=saved)
        x, fallback = self.policy.screen(x, m, feature_names=[f.name for f in m.inputs for _ in range(f.levels)])
        y = self.model(x)
        replace = self.settings.mode == "replace"
        weight = more.get("weight")
        if self.policy.fixer != "none" and weight is None:
            raise NNConfigError("--nn fixer needs the layer mass, and this run lends no vertical metric zz")
        slices = m.output_slices()
        targets = None
        if replace and self.policy.fixer != "none":
            targets = {}
            for name, rows in slices.items():
                if name.startswith("dq"):
                    targets[name] = (weight * scheme[rows]).sum(axis=0)
                elif name == "dtheta" and self.policy.fixer == "moisture+heat":
                    targets[name] = (weight * more["exner"] * scheme[rows]).sum(axis=0)
        floor = more
        if not replace:
            # A residual adds to the scheme's rate: the positivity floor holds
            # for the sum, taken from the moisture the scheme alone would leave.
            floor = dict(more)
            for name, rows in slices.items():
                if name.startswith("dq") and name[1:] in more:
                    floor[name[1:]] = more[name[1:]] + f32(self.dt) * scheme[rows]
        y = self.policy.constrain(y, m, dt=self.dt, state=floor, weight=weight, targets=targets)
        delta = y - scheme if replace else y
        if fallback is not None:
            delta = cp.where(fallback[None, :], f32(0), delta)
        increments = {name: delta[rows] for name, rows in slices.items()}
        d_theta, d_scalars = couple_increment(cp, increments, mass=more["mass"], theta_m=more["theta_m"],
                                              qv=more["qv"])
        # A select, not a product: a non-finite model value in the specified
        # zone must not reach it (NaN * 0 is NaN).
        mask = self._mask
        if d_theta is not None:
            if mask is not None:
                d_theta = cp.where(mask, d_theta, f32(0))
            sources.rho_theta[:L, :nc] = sources.rho_theta[:L, :nc] + d_theta.astype(f32, copy=False)
        for species, value in d_scalars.items():
            i = self.names.index(species)
            if mask is not None:
                value = cp.where(mask, value, f32(0))
            sources.scalars[i, :L, :nc] = sources.scalars[i, :L, :nc] + value.astype(f32, copy=False)

    def receipt(self) -> dict:
        summary = {"mode": self.settings.mode, "scheme": self.scheme, "route": "hex",
                   "mpas_identical": not self.applies, "calls": self.calls, "levels": self.levels,
                   "samples": 0 if self.tap is None else self.tap.samples,
                   "store": None if self.tap is None else str(self.tap.path),
                   "seat": "held physics sources after physics_get_tend, before SPPT/SKEB, IAU and nudging"}
        if self.applies:
            summary.update(model=self.manifest.name, identity=self.identity(), runner=self.model.backend,
                           applied_cells=self.applied_cells, note=NOT_WRF_IDENTICAL,
                           **self.policy.ledger.summary())
        return summary

    def close(self) -> dict:
        if self.tap is not None:
            self.tap.close()
        summary = self.receipt()
        if self.applies:
            print(f"nn: {self.settings.mode} of the hex physics source by {self.manifest.name} "
                  f"({self.model.backend}) over {self.calls} steps; {summary['columns_out_of_range']} of "
                  f"{summary['columns_seen']} column calls outside the trained range "
                  f"({self.settings.out_of_range}); {NOT_WRF_IDENTICAL}", file=sys.stderr, flush=True)
        return summary


def _host(array) -> np.ndarray:
    return array.get() if hasattr(array, "get") else np.asarray(array)


def refuse_unsupported(*, execution=None, moving=None, physics_map=None) -> None:
    """The runs the seat cannot attach to, refused by name."""
    if execution is not None:
        raise NotImplementedError(
            "--nn on a same-host partition or an LTS execution: the hex seat runs on one card's whole mesh "
            "(one process, one step clock); run it on one card without --fullstep-local-timestep")
    if physics_map is not None or (moving is not None and not getattr(moving, "fixed_mesh", False)):
        raise NotImplementedError(
            "--nn on a moving region: the seat's columns and tap record are bound to one mesh; run it on a "
            "fixed mesh (checkpoint and --restart-from resumes on the same mesh are supported)")


def _reader(session, n_cells: int):
    """A physics-pool field, live, without binding it.

    ``session`` is the MPAS physics session (``session_v842``); on a
    resident run its device views (``session.device_pools``) hold the
    authoritative words of every field a slot has bound, and the native
    host storage is stale until an output boundary publishes it.  A
    resident run therefore reads the views' held device array
    (``resident_field``, which never uploads), and only a field no slot has
    bound -- whose host storage is still authoritative -- is copied from
    the host.  A run without resident views reads the host storage, which
    is then the authority.  ``session.field`` is not used: on a resident
    run it would upload and cache an unbound field, making the device copy
    the one published (a tap must change nothing).

    The views go resident only after the seat attaches (the resident
    suite's admission), so residency is asked at every read, never once at
    attach -- a reader that decided at attach read the host storage for
    the whole run."""
    native = session.native

    def read(pool, name):
        views = getattr(session, "device_pools", None)
        if views is None and hasattr(session, "resident_field"):
            views = session   # the views themselves
        if views is not None and getattr(views, "resident", True) and hasattr(views, "resident_field"):
            value = views.resident_field(pool, name, level=1)
            if value is not None:
                return value
        storage = getattr(native, "field_storage", None)   # the live storage, sliced once below
        host = storage(pool, name, level=1) if storage is not None else native.get_field(pool, name, level=1)
        return np.array(np.asarray(host)[..., :n_cells])
    return read


def attach(coupling, *, config, mesh, n_cells: int, vertical, dt: float, execution=None, moving=None,
           physics_map=None, resumed: bool = False, carried: dict | None = None,
           settings: NNSettings | None = None):
    """Attach the seat ``--nn`` asks for to the MPAS-route coupling; None
    without ``--nn``.  ``carried`` is the resumed carrier's ``nn`` entry."""
    if settings is None:
        path = configured()
        if path is None:
            return None
        settings = read_settings(path)
    refuse_unsupported(execution=execution, moving=moving, physics_map=physics_map)
    session = coupling.session
    bdy = mesh.arrays.get("bdyMaskCell") if hasattr(mesh, "arrays") else None
    if bdy is not None and not np.any(np.asarray(bdy)[:int(n_cells)] > 0):
        bdy = None
    state = coupling.dycore.atmosphere.state
    seat = HexColumnSeat(settings, cp=coupling.cp, names=coupling.names, n_cells=n_cells,
                         n_levels=int(state.rho.shape[0]), zgrid=vertical.zgrid,
                         zz=getattr(getattr(coupling.dycore.atmosphere, "vertical", None), "zz", None),
                         bdy_mask=bdy, dt=dt, mode=getattr(config, "mode", "exact"),
                         fields=_reader(session, int(n_cells)), resume=resumed)
    if resumed:
        check_carried(seat.identity(), carried)
    coupling.nn_seat = seat
    coupling.dycore.nn_seat = seat
    return seat

