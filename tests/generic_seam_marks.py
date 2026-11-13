"""Whether this process can plan an engine (generic column seam) selection.

``woof hex physics resolve`` hands any option beyond a named suite's slots to
the engine's generic column seam, which plans it with
``woof.core.mpas_column_suite.column_physics_plan``.  That module imports the
device stack (cupy, or the Metal facade), so a CPU-only host -- the hosted
``cpu`` legs -- cannot plan one, and the door refuses by name ("the engine's
generic column seam could not be imported in this process").  A test whose
assertion is about a planned engine selection skips there, saying why; it runs
on every host that carries the engine (mac Metal, pb CUDA).
"""

import pytest


def generic_seam_importable() -> bool:
    try:
        import woof.core.mpas_column_suite  # noqa: F401
    except ImportError:
        return False
    return True


needs_generic_seam = pytest.mark.skipif(
    not generic_seam_importable(),
    reason="the generic column seam (woof.core.mpas_column_suite) imports the "
           "device stack; a CPU-only host cannot plan an engine selection")
