"""The microphysics spec-zone ring: its slots, its guard, and the restore normalization.

Device-free (no CuPy import), split out of woof.core.microphysics so that
woof.io.restart can normalize a restored checkpoint's ring on a host with no
GPU runtime; woof.core.microphysics re-exports every name here.
"""

from __future__ import annotations

from woof.config import RunConfig
from woof.core.state import DomainState
from woof.core.physics_inventory import (  # noqa: F401
    spec_zone_ring_save_slots,  # one home; re-exported here
    spec_zone_ring_slices,  # one home (cupy-free); re-exported here
)


#: Persistent (ny, nx) surface slots the adapters write: the accumulators
#: RAINNC/SNOWNC/GRAUPELNC plus the per-call *NCV/SR diagnostics (the
#: canonical driver mapping is physics.microphysics_scratch_slots; this is
#: its union across schemes).  WRF never writes any of them in the ring:
#: the accumulators/NCVs by the clipped tiles (zero init), SR by
#: ``grid%sr = 0.`` whole-field each call (solve_em.F:3691) followed by
#: clipped tile writes.
_RING_SURFACE_SLOTS = (
    "mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
    "mp_graupelnc", "mp_graupelncv", "mp_hailnc", "mp_hailncv",
    "mp_sr", "mp_kessler_sr",
)

#: Persistent 3-D scratch slots that survive the call (the REFL_10CM
#: output stash -- WRF's clipped tiles leave ring refl at its allocated 0).
_RING_VOLUME_SLOTS = ("refl_10cm",)


def _ring_guard_slices(state: DomainState, cfg: RunConfig):
    """WRF solve_em.F:3618-3622: ``sz = spec_zone`` iff ``specified .or.
    nested``, else 0 -- so periodic/open configs return None and keep the
    frozen whole-field call bitwise.  woof has no specified+periodic_x
    channel mode, so WRF's channel i-clip exemption
    (module_microphysics_driver.F:871-873) is deliberately not wired.
    """
    if not (getattr(cfg, "specified", False)
            or getattr(cfg, "nested", False)):
        return None
    sz = int(cfg.spec_zone)
    if sz <= 0:
        return None
    _, ny, nx = state.p.shape
    return spec_zone_ring_slices(ny, nx, sz)


def normalize_spec_zone_ring_after_restore(state: DomainState,
                                           cfg: RunConfig,
                                           *, relocated: bool = False
                                           ) -> None:
    """Restart migration normalization: zero the specified-zone ring of
    every microphysics-owned accumulator/diagnostic slot and of
    h_diabatic after a checkpoint restore.

    Checkpoints written before the ring exclusion landed carry the SAME
    restart format version, and whole-field microphysics could leave them
    with nonzero ring RAINNC/*NCV/SR/refl and a stale ring h_diabatic.
    No WRF-valid trajectory can contain such values: the fields are
    allocator-zero-initialized (frame/module_domain.F:770-777,
    tools/gen_allocs.c:411-415) and every MP tile write is clipped
    (solve_em.F:3631-3639/:4040-4048).  Without this step the guard's
    capture/restore would preserve the stale ring FOREVER, and ring
    h_diabatic would feed every RK stage (dycore.add_h_diabatic_tendency)
    before post-RK microphysics re-pins it.  Ring prognostics (theta,
    moisture) are deliberately NOT touched -- the boundary machinery owns
    and overwrites them every step.  Value-level and idempotent (post-fix
    checkpoints already carry zero rings), so the restart format version
    is unchanged: this function IS the documented migration step for
    pre-fix checkpoints.  Periodic/open domains are untouched.

    ``relocated`` NAMES THE ONE TRAJECTORY THAT BREAKS THE IDEMPOTENCE
    CLAIM ABOVE, and it is not a pre-fix file.  A relocation shifts the
    carried accumulators in INDEX space
    (:func:`woof.core.physics_continuation.shift_continuation`), so a
    column that spent the run accumulating as interior ``i = 1`` becomes
    ring ``i = 0`` the moment the nest steps one cell east -- still the
    same ground, still that ground's real rain total, and still able to
    return to the interior on the next westward step.  A post-fix
    checkpoint of a MOVING nest therefore carries a nonzero ring
    legitimately, this normalization is not idempotent on it, and firing
    it destroys accumulated precipitation that no later step recomputes
    (MEASURED: a 1 km nest 6 h into a 24 h run resumed with its whole
    west column of RAINNC zeroed, up to 3.77 mm, frozen for the rest of
    the forecast).  WRF's own moving nest shifts RAINNC wholesale for the
    same reason.

    The exemption is exactly the RELOCATION-CARRIED set, read from the
    registry that decides what a move shifts rather than restated here,
    so the two cannot drift.  ``refl_10cm`` and ring h_diabatic are NOT
    in it and are normalized either way: nothing shifts them, the ring
    guard re-pins h_diabatic on every microphysics call, and WRF's ring
    value for both is an unconditional zero.
    """
    if cfg.mp_physics == 0:
        return
    slices = _ring_guard_slices(state, cfg)
    if slices is None:
        return
    carried: frozenset[str] = frozenset()
    if relocated:
        from woof.core.physics_continuation import continuation_slots

        carried = frozenset(continuation_slots())
    for slot in _RING_SURFACE_SLOTS + _RING_VOLUME_SLOTS:
        if slot in carried:
            continue
        arr = state.existing_scratch(slot)
        if arr is None:
            continue
        for slc in slices:
            arr[slc] = 0
    if state.h_diabatic is not None:
        for slc in slices:
            state.h_diabatic[slc] = 0
