"""What the worker, the cost estimate and the deploy renderers share about
packing several hindcast jobs onto one GPU (``--slots-per-gpu``).

One home for the flag's grammar and its two pinned numbers, so
``woof energy worker``, ``woof energy estimate`` and ``woof energy deploy
render`` cannot drift apart:

* :func:`parse_slots_per_gpu` -- ``auto`` (any case) or a positive count;
  an int is accepted as itself; anything else is a ValueError;
* :data:`AUTO_MAX_SLOTS_PER_GPU` -- the most slots ``auto`` starts on one
  card (the worker) and the most copies it prices (the estimate);
* :data:`DEFAULT_MEM_MARGIN_GIB` -- card memory kept free beside the
  declared reservations.  An assumption, not a measurement: the CUDA
  driver, an MPS server and allocator fragmentation that a job's priced
  peak does not hold.

The numbers and the grammar themselves live in :mod:`woof.gpu_packing`,
shared with ``woof multi-run``'s ``runs_per_gpu``; this module pins the
energy flag's name onto them.
"""

from __future__ import annotations

from typing import Any

from woof.gpu_packing import (AUTO_MAX_SLOTS_PER_GPU, DEFAULT_MEM_MARGIN_GIB,
                              SLOTS_AUTO, parse_count_or_auto)


def parse_slots_per_gpu(value: Any) -> int | str:
    """``"auto"`` or a positive int; anything else is a ValueError."""

    return parse_count_or_auto(value, flag="--slots-per-gpu")


__all__ = ["AUTO_MAX_SLOTS_PER_GPU", "DEFAULT_MEM_MARGIN_GIB", "SLOTS_AUTO",
           "parse_slots_per_gpu"]
