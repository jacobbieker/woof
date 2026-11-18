"""An example hex column backend: a scheme plus a neural-network column model.

:class:`NNColumnBackend` is a :class:`woof.hex.physics_seam.PhysicsBackend`
(the host protocol a ``woof.hex.rows`` provider implements): it runs an
inner backend on a :class:`~woof.hex.physics_seam.ColumnBatch`, then adds
(``residual``) or substitutes (``replace``) a column model's rates, under
the same :class:`woof.nn.safety.SafetyPolicy` the WRF and hex seats use.  It
returns uncoupled physical rates (:class:`ColumnTendencies`); the seam's
:func:`~woof.hex.physics_seam.unpack_mpas_tendencies` couples them to dry
mass as MPAS does, so the backend never sees the mesh.

It is an example, not a row this distribution registers: registering it in
this package's metadata would make every forecast's row lookup import
:mod:`woof.nn`.  A third party ships it (or its own) from its own
distribution::

    # their pyproject.toml
    [project.entry-points."woof.hex.rows"]
    my_nn_rows = "my_package.rows:rows"

where ``rows()`` returns :class:`woof.hex.physics_backend_admission.
PhysicsBackendRow` entries under a new name (a row name is never rebound),
and the row's adapter module publishes ``build_column_backend(row, **kw)``.
The production hex forecast does not run the host seam per step; the
MPAS-route ``--nn`` seat (:mod:`woof.nn.hex_seat`) is the opt-in NN path
of ``woof hex forecast``.

Vocabulary.  Inputs read from the batch: ``theta`` (dry), ``theta_m``,
moisture species, ``u``/``v`` (cell winds), ``w`` (mean of the layer's
faces), ``p`` (EOS pressure), ``exner`` (``T / theta``), ``rho`` (dry
density), ``z`` (layer centre above the ground), ``dz``, ``hgt``, and any
other name a 2-D field of ``batch.surface`` or ``batch.static`` carries.
Outputs: ``dtheta`` (dry, K s-1), ``du``/``dv`` (m s-2) and ``d<species>``
(kg kg-1 s-1) -- the batch's winds are cell-centred, so momentum is
available here, unlike the MPAS-route seat.  A model spans the lowest
``levels`` levels; the levels above keep the inner backend's rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from woof.hex.physics_seam import BackendRequirements, ColumnBatch, ColumnTendencies, GPUWMSchemeSelection
from woof.nn.launch import NNConfigError

#: Inputs the batch carries by these names (others are read from surface/static).
PROFILE_INPUTS = ("theta", "theta_m", "u", "v", "w", "p", "exner", "rho", "z", "dz")
SURFACE_INPUTS = ("hgt",)


@dataclass
class NNColumnBackend:
    """``inner`` plus a column model (``residual``) or the model alone where
    it has outputs (``replace``).  ``model`` is a loaded column model
    (:func:`woof.nn.models.load_column_model`); ``policy`` a
    :class:`~woof.nn.safety.SafetyPolicy`."""

    inner: Any
    model: Any
    policy: Any
    mode: str = "residual"
    name: str = "nn-column"
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.mode not in ("residual", "replace"):
            raise NNConfigError(f"NNColumnBackend mode {self.mode!r}: residual or replace")
        manifest = self.model.manifest
        outputs = [f.name for f in manifest.outputs]
        bad = [n for n in outputs if not (n in ("dtheta", "du", "dv") or n.startswith("dq"))]
        if bad:
            raise NNConfigError(f"model {manifest.name}: outputs {bad} are not rates this backend returns "
                                "(dtheta, du, dv, d<species>)")
        depths = {f.levels for f in manifest.outputs}
        if len(depths) != 1:
            raise NNConfigError(f"model {manifest.name}: outputs span {sorted(depths)} levels; one depth")
        self.levels = depths.pop()

    @classmethod
    def from_artefact(cls, inner, path, *, mode: str = "residual", backend: str = "numpy", **policy):
        from woof.nn.models import load_column_model
        from woof.nn.safety import SafetyPolicy

        return cls(inner=inner, model=load_column_model(path, backend=backend), policy=SafetyPolicy(**policy),
                   mode=mode)

    def requirements(self, selection: GPUWMSchemeSelection) -> BackendRequirements:
        base = self.inner.requirements(selection)
        species = {f.name for f in self.model.manifest.inputs if f.name.startswith("q")}
        species |= {f.name[1:] for f in self.model.manifest.outputs if f.name.startswith("dq")}
        return BackendRequirements(atmosphere=base.atmosphere, scalars=frozenset(base.scalars | species),
                                   surface=base.surface, static=base.static)

    # ---------------------------------------------------------------- inputs
    def _input(self, batch: ColumnBatch, name: str, levels: int) -> np.ndarray:
        L = levels
        if name == "theta":
            return batch.theta[:L]
        if name == "theta_m":
            return batch.modified_theta[:L]
        if name in ("u", "v"):
            return getattr(batch, name)[:L]
        if name == "w":
            if batch.w is None:
                raise NNConfigError("input 'w': this batch carries no vertical velocity")
            return 0.5 * (batch.w[:L] + batch.w[1:L + 1])
        if name == "p":
            return batch.eos_pressure[:L]
        if name == "exner":
            return batch.temperature[:L] / batch.theta[:L]
        if name == "rho":
            return batch.dry_density[:L]
        if name == "z":
            return batch.height[:L] - batch.height_interface[:1]
        if name == "dz":
            return batch.layer_thickness[:L]
        if name == "hgt":
            return batch.height_interface[:1]
        if name in batch.scalars:
            return batch.scalars[name][:L]
        for table in (batch.surface, batch.static):
            if name in table and np.ndim(table[name]) == 1:
                return np.asarray(table[name])[None, :]
        raise NNConfigError(f"input {name!r}: the column batch carries no such field")

    def features(self, batch: ColumnBatch) -> np.ndarray:
        """The model's inputs ``[n_inputs, column]`` from the batch."""
        rows = []
        for spec in self.model.manifest.inputs:
            value = np.asarray(self._input(batch, spec.name, spec.levels))
            if value.shape[0] != spec.levels:
                raise NNConfigError(f"input {spec.name!r} spans {value.shape[0]} levels; the model reads "
                                    f"{spec.levels}")
            rows.append(value)
        return np.concatenate(rows, axis=0).astype(batch.dtype, copy=False)

    # ------------------------------------------------------------------ run
    def run(self, batch: ColumnBatch, selection: GPUWMSchemeSelection, dt: float) -> ColumnTendencies:
        self.calls += 1
        base = self.inner.run(batch, selection, dt)
        m, L, dtype = self.model.manifest, self.levels, batch.dtype
        if L > batch.n_levels:
            raise NNConfigError(f"model {m.name} spans {L} levels; the batch has {batch.n_levels}")
        slices = m.output_slices()

        def scheme_rate(name):
            if name.startswith("dq"):
                value = base.dscalars.get(name[1:])
                return np.zeros((L, batch.n_columns), dtype) if value is None else np.asarray(value)[:L]
            return np.asarray(getattr(base, name))[:L]

        scheme = np.concatenate([scheme_rate(f.name) for f in m.outputs], axis=0)
        x, fallback = self.policy.screen(self.features(batch), m,
                                         feature_names=[f.name for f in m.inputs for _ in range(f.levels)])
        y = np.asarray(self.model(x))
        state = {"exner": batch.temperature[:L] / batch.theta[:L]}
        for f in m.outputs:
            if f.name.startswith("dq") and f.name[1:] in batch.scalars:
                q = batch.scalars[f.name[1:]][:L]
                # A residual's positivity floor is the moisture the scheme leaves.
                state[f.name[1:]] = q if self.mode == "replace" else q + dtype.type(dt) * scheme[slices[f.name]]
        weight = batch.dry_density[:L] * batch.layer_thickness[:L]
        targets = None
        if self.mode == "replace" and self.policy.fixer != "none":
            targets = {}
            for name, rows in slices.items():
                if name.startswith("dq"):
                    targets[name] = (weight * scheme[rows]).sum(axis=0)
                elif name == "dtheta" and self.policy.fixer == "moisture+heat":
                    targets[name] = (weight * state["exner"] * scheme[rows]).sum(axis=0)
        y = self.policy.constrain(y, m, dt=dt, state=state, weight=weight, targets=targets)
        new = y if self.mode == "replace" else scheme + y
        if fallback is not None:
            new = np.where(np.asarray(fallback)[None, :], scheme, new)
        new = new.astype(dtype, copy=False)
        out = ColumnTendencies(du=np.array(base.du, dtype=dtype), dv=np.array(base.dv, dtype=dtype),
                               dtheta=np.array(base.dtheta, dtype=dtype),
                               dscalars={k: np.array(v, dtype=dtype) for k, v in base.dscalars.items()},
                               surface_updates=dict(base.surface_updates), diagnostics=dict(base.diagnostics))
        for name, rows in slices.items():
            if name.startswith("dq"):
                target = out.dscalars.setdefault(name[1:], np.zeros_like(out.dtheta))
            else:
                target = getattr(out, name)
            target[:L] = new[rows]
        return out
