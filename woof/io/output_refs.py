"""Release in-flight outputs' references to live state arrays.

Kept out of woof/core/model.py, whose clock audit
(tests/test_clock.py::test_no_float_elapsed_accumulation_audit) bans
getattr: the writers this probes do not all carry the release hook.
woof.core.model re-exports it under its old name.
"""

from __future__ import annotations


def release_output_device_refs(model) -> dict[str, int]:
    """Make in-flight outputs stop pinning live state arrays.

    The asynchronous checkpoint and the asynchronous history writers both
    stage straight from the live state: until a snapshot or a queued
    frame lets go of its source arrays, those arrays outlive whatever
    drops the state.  A relocation drops the moving domain's state and
    rebuilds it at once, so a reference still held there meant TWO copies
    of the domain on the card for as long as the output took -- on a
    long moving-nest run that is what ran a card out of memory at a
    move.  Called immediately before such a release.  Each output waits
    only for its own already-queued copy; nothing is drained, and no
    file or number changes.  Returns the references each dropped.

    One call covers every domain: the checkpoint snapshot and each
    domain's history writer are released whole, so the descendants a
    containment slide re-grounds after the mover are covered by the
    mover's call.  Restore-time relocations run before any output is in
    flight and need none.
    """
    released = {"checkpoint": 0, "history": 0}
    writer = getattr(model, "_checkpoint_writer", None)
    release = getattr(writer, "release_device_refs", None)
    if release is not None:
        released["checkpoint"] = int(release() or 0)
    writers = getattr(model, "_io_manager", None)
    release = getattr(writers, "release_device_refs", None)
    if release is not None:
        released["history"] = int(release() or 0)
    return released
