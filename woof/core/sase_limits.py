"""SASE constants shared by configuration, state and CPU memory accounting.

They live here, in a module with no imports at all, because of where
they have to be readable from.  The closure's authority module is
``woof.verify.sase_ref``, and ``woof/verify`` is a DEVELOPER
verification tree that the standalone CPU preprocessing distribution
deliberately omits -- so a config- or state-layer import of it breaks
that wheel (``tests/test_native_wrf_distribution.py`` is the gate that
says so, and it caught exactly that).

There is ONE definition of each value. ``sase_ref`` imports the column
limit and energy floor from here, preserving its constant registry and
configuration hash. The CUDA launch tier and CPU workspace estimator
share the reduction block size through this same import-free module.
"""

from __future__ import annotations

#: Deepest column the closure's implicit vertical solve accepts.
#:
#: An IMPLEMENTATION bound, not a physical one: the tridiagonal sweeps and
#: the BL89 / moist-N2 / plume column passes keep FP64 columns in
#: per-thread local memory, sized by the device ``SASE_KMAX`` define.  The
#: launcher compiles that define PER RUN at the smallest of
#: :data:`KMAX_TIERS` that holds the run's ``nz`` (:func:`kmax_tier`), so
#: this ceiling is the widest tier any run can compile, and the launcher's
#: rejection, ``woof.config.SASE_MAX_NZ`` and the vertical contract's SASE
#: ceiling are all this one number.  256 is the dycore's own column
#: ceiling.
MAX_COLUMN_LEVELS = 256

#: The ``SASE_KMAX`` values the launcher compiles, shallowest first.
#:
#: Powers of two, so a run compiles one of four module variants rather than
#: one per level count.  The per-thread frame is EXACTLY linear in the
#: tier (49 B per level, zero intercept; ``woof.core.preflight
#: .SASE_TIER_FRAME``), so the tier is also what the launch-time
#: local-memory reservation is priced at: 1,568 / 3,136 / 6,272 /
#: 12,544 B.  Compiling a fixed 256 would charge every SASE run the
#: 12,544 B frame -- about 2.8 GiB of reservation on a 170-SM card --
#: where a 50-level run needs the 64 tier's 3,136 B.  The floor of 32
#: keeps the variant count at four: below it the frame is already
#: inside the default 1 KiB stack and a smaller tier saves nothing.
KMAX_TIERS = (32, 64, 128, 256)

#: The ``#ifndef`` default in ``kernels/sase.cu``: the tier the PLAIN
#: loader compiles, which is what the local-frame census
#: (``tests/test_preflight.py::test_the_recorded_local_frames_match_the_
#: driver``) measures and every platform's recorded ``sase`` row (6,272 B)
#: was read at.  The launcher never uses it -- it always passes its tier
#: -- so it is the census anchor of ``SASE_TIER_FRAME``, not a run bound.
SOURCE_DEFAULT_KMAX = 128

def kmax_tier(nz: int) -> int:
    """The ``SASE_KMAX`` tier the launcher compiles for an ``nz`` column.

    The smallest of :data:`KMAX_TIERS` holding ``nz`` levels.  A column
    deeper than :data:`MAX_COLUMN_LEVELS` is refused here, by name, so no
    caller can reach a compile with an extent shorter than its column.
    """
    nz = int(nz)
    if nz < 1:
        raise ValueError(f"SASE column depth must be positive, got nz={nz}")
    for tier in KMAX_TIERS:
        if nz <= tier:
            return tier
    raise ValueError(
        f"nz={nz} exceeds the SASE column bound SASE_KMAX="
        f"{MAX_COLUMN_LEVELS} (woof.core.sase_limits.MAX_COLUMN_LEVELS)")


#: Threads in each SASE reduction block. The CUDA launch tier and the CPU
#: workspace estimator share this value so accounting never imports CuPy.
THREADS_PER_BLOCK = 128

#: Realizability floor on the prognostic subgrid energy [m2 s-2].
#:
#: The value the fused step clips to and the value a cold start fills, so
#: the state allocator and the closure agree on what "no turbulence yet"
#: is.  Registered in the closure's configuration hash.
E_MIN = 1.0e-6
