"""The hex column backend every chemistry mode runs on: one transaction, one restart and one product path.

``woof.hex.chem_wrf`` (WRF-Chem processes) and ``woof.hex.chem_micm_hex`` (MICM) differ only in what runs on
the columns.  Both run after the step's microphysics on the post-RK MPAS-to-physics columns, settle with the
physics transaction (commit or abort), publish their products in the diagnostic snapshot and carry a restart
payload.  This module is that shared backend; a runtime supplies:

``columns(prepared, phase)``
    the column dict it reads from an MPAS-to-physics preparation, ``phase`` one of ``"phase_one"``
    (before phase one), ``"post_rk"`` (after phase two) and ``"snapshot"`` (history products);
``step(scalars, columns, *, dt, elapsed_seconds, physics_fields, microphysics)``
    after phase two;
``commit()``, ``abort()``, ``products(scalars, prepared)``, ``receipt()``, ``ledger()`` (optional),
``restart_payload()``, ``restore_payload(payload)``;
``bind_seam(seam)`` and ``before_phase_one(scalars, prepared, *, elapsed_seconds)`` (optional: a runtime
whose chemistry the phase-one physics reads, WRF's aerosol-radiation feedback);
``phase2_winds`` (whether the post-RK preparation reconstructs the cell winds).
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path


def build(runtime_factory, *, constructor, prep_geometry, kernel_cache, gwdo_static=None,
          gwdo_kernel_cache=None, checkout=None, rollback_snapshot=True):
    """The backend, with ``runtime_factory(seam)`` bound as ``backend.chemistry``."""
    from .cuda_arwen_physics_v841 import PersistentTwoPhaseCudaPhysicsBackendV841
    from .cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841
    from .species_row import WSM6_SPECIES_ROW

    class ChemistryColumnBackend(PersistentTwoPhaseCudaPhysicsBackendV841):
        def _species_row(self, scalar_names):
            return WSM6_SPECIES_ROW

        def _prepare(self, atmosphere, scalar_names, *, post_rk):
            winds = bool(getattr(self.chemistry, "phase2_winds", False))
            return prepare_mpas_to_phys_cuda_v841(
                atmosphere, scalar_names=scalar_names, geometry=self._prep_geometry,
                kernel_cache=self._kernel_cache, post_rk_wsm6=post_rk, phase2_only=post_rk,
                phase2_winds=post_rk and winds)

        def _new_seam(self):
            seam = super()._new_seam()
            runtime = getattr(self, "chemistry", None)
            if runtime is not None and hasattr(runtime, "bind_seam"):
                runtime.bind_seam(seam)
            return seam

        def begin_step(self, *, atmosphere, scalar_names, dt, dynamics_tendencies=None):
            runtime = self.chemistry
            if getattr(runtime, "needs_phase_one", False):
                prepared = self._prepare(atmosphere, scalar_names, post_rk=False)
                runtime.before_phase_one(atmosphere.state.scalars, runtime.columns(prepared, "phase_one"),
                                         elapsed_seconds=float(atmosphere.state.time_seconds))
            return super().begin_step(atmosphere=atmosphere, scalar_names=scalar_names, dt=dt,
                                      dynamics_tendencies=dynamics_tendencies)

        def finish_step(self, *, atmosphere, scalar_names, dt, refl_10cm_due=False):
            update = super().finish_step(atmosphere=atmosphere, scalar_names=scalar_names,
                                        dt=dt, refl_10cm_due=refl_10cm_due)
            driver = self._seam._driver
            prepared = self._prepare(atmosphere, scalar_names, post_rk=True)
            self.chemistry.step(atmosphere.state.scalars, self.chemistry.columns(prepared, "post_rk"),
                                dt=float(dt), elapsed_seconds=float(atmosphere.state.time_seconds),
                                physics_fields=driver.fields,
                                microphysics=getattr(driver, "microphysics", None))
            return update

        def commit_step(self, **kwargs):
            result = super().commit_step(**kwargs)
            self.chemistry.commit()
            return result

        def abort_step(self, **kwargs):
            self.chemistry.abort()
            return super().abort_step(**kwargs)

        def diagnostic_snapshot(self, **kwargs):
            snapshot = super().diagnostic_snapshot(**kwargs)
            runtime = self.chemistry
            if runtime.driver is None:
                return snapshot
            atmosphere = runtime.driver.atmosphere
            prepared = self._prepare(atmosphere, runtime.species_row.names(), post_rk=False)
            extra = runtime.products(atmosphere.state.scalars, runtime.columns(prepared, "snapshot"))
            return replace(snapshot, surface={**snapshot.surface, **extra},
                           metadata={**snapshot.metadata, "chemistry": runtime.receipt()})

        def restart_state(self):
            state = dict(super().restart_state())
            state["chemistry"] = self.chemistry.restart_payload()
            return state

        def restore_restart_state(self, payload):
            self.chemistry.restore_payload(payload.get("chemistry"))
            super().restore_restart_state({k: v for k, v in payload.items() if k != "chemistry"})

        def write_ledger(self, output):
            runtime = self.chemistry
            receipt = runtime.receipt()
            if hasattr(runtime, "ledger"):
                receipt = {**receipt, "ledger": runtime.ledger()}
            Path(output, "chemistry-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    backend = ChemistryColumnBackend(
        constructor=constructor, prep_geometry=prep_geometry, kernel_cache=kernel_cache,
        gwdo_static=gwdo_static, gwdo_kernel_cache=gwdo_kernel_cache,
        arwen_checkout=checkout, rollback_snapshot=rollback_snapshot)
    runtime = runtime_factory(backend._seam)
    backend.chemistry = runtime
    if hasattr(runtime, "bind_seam"):
        runtime.bind_seam(backend._seam)
    return backend
