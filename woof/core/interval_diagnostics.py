"""Per-history-interval diagnostics riding on one domain's state.

A small registry, so a diagnostic that folds every model step and restarts
its window at every history write needs exactly three hooks and no more:

* :func:`step` -- called once per completed dycore step from the
  epilogue of :func:`woof.core.dycore.step`, beside ``UP_HELI_MAX``;
* :func:`history_fields` -- merged into the frame by
  :func:`woof.io.history_layout.live_state_history_fields`, so both the
  synchronous and the asynchronous writers carry the same inventory;
* :func:`close_interval` -- called by the history publishers
  (:meth:`woof.io.wrfout.PerDomainWrfoutWriters.submit` and
  :func:`woof.runtime.write_case_output`) immediately after the frame
  has been snapshotted, to flush any sidecar and restart the window.

A diagnostic is any object with a ``name`` and those three methods
(``on_step(state, dt)``, ``history_fields(state)``,
``close_interval(state, *, valid_time, output_dir, grid_id)``) plus
``refuse_publisher(reason)``, called when a publisher that cannot honour
the window would otherwise drop it silently.  It is attached with
:func:`attach` and lives in a plain attribute of the state, so a state
that never had one attached pays one ``getattr`` per step and nothing else.

PINNED: diagnostics registered here READ model state and write only their
own buffers.  Nothing in this module or its members is a trajectory input,
which is why none of them enters a restart identity (see each member's
own restart note: hub-height wind documents its reset-on-resume semantics
in woof/core/hub_wind_diag.py).

THE MEMBERS.  ``hub_wind`` (:mod:`woof.core.hub_wind_diag`, ``[output]
hub_wind``/``hub_heights_m``) and ``windfarm_power``
(:mod:`woof.core.windfarm_power_diag`, ``[output] windfarm_turbines``:
rotor-equivalent wind, power and thrust at listed turbines).  Each is
attached only when its key is set, so a run that sets neither keeps the
empty registry and every byte it wrote before them.
"""
from __future__ import annotations

#: The state attribute the registry lives in.  Deliberately not a scratch
#: slot: scratch slots are classified (serialized / carried / work) by the
#: preflight completeness scanner and the restart walk, and these objects
#: are neither model state nor restart payload.
STATE_ATTRIBUTE = "_interval_diagnostics"


class IntervalDiagnosticError(RuntimeError):
    """A registry misuse: a duplicate name, or a missing member."""


def attached(state) -> tuple:
    """Every diagnostic attached to ``state``, in attachment order."""
    return tuple(getattr(state, STATE_ATTRIBUTE, None) or ())


def get(state, name: str):
    """The attached diagnostic called ``name``, or ``None``."""
    for diagnostic in attached(state):
        if diagnostic.name == name:
            return diagnostic
    return None


def attach(state, diagnostic) -> None:
    """Attach ``diagnostic`` to ``state``; a second one of a name refuses."""
    if get(state, diagnostic.name) is not None:
        raise IntervalDiagnosticError(
            f"interval diagnostic {diagnostic.name!r} is already attached to "
            "this domain state; attaching a second would fold every step "
            "twice and publish two windows under one name")
    setattr(state, STATE_ATTRIBUTE, attached(state) + (diagnostic,))


def detach(state, name: str) -> None:
    """Remove one diagnostic (a no-op when it is absent)."""
    kept = tuple(d for d in attached(state) if d.name != name)
    setattr(state, STATE_ATTRIBUTE, kept)


def step(state, cfg) -> None:
    """Fold this completed step into every attached window."""
    members = getattr(state, STATE_ATTRIBUTE, None)
    if not members:
        return
    for diagnostic in members:
        diagnostic.on_step(state, float(cfg.dt))


def history_fields(state) -> dict:
    """Every attached diagnostic's history fields, merged in order."""
    fields: dict = {}
    for diagnostic in attached(state):
        fields.update(diagnostic.history_fields(state))
    return fields


def close_interval(state, *, valid_time, output_dir, grid_id: int) -> None:
    """The frame for ``valid_time`` is snapshotted: flush and restart."""
    for diagnostic in attached(state):
        diagnostic.close_interval(state, valid_time=valid_time,
                                  output_dir=output_dir, grid_id=int(grid_id))


def refuse_unsupported_publisher(state, reason: str) -> None:
    """Raise if any diagnostic is attached to a state ``reason`` cannot serve.

    A publisher that bypasses :func:`history_fields` (a streamed domain's
    host-store frame, an ensemble capture) would publish frames without
    the window and never restart it -- the "nothing raises, the field is
    just absent or wrong" failure.  It calls this instead.
    """
    for diagnostic in attached(state):
        diagnostic.refuse_publisher(reason)


__all__ = [
    "IntervalDiagnosticError",
    "STATE_ATTRIBUTE",
    "attach",
    "attached",
    "close_interval",
    "detach",
    "get",
    "history_fields",
    "refuse_unsupported_publisher",
    "step",
]
