"""Which build of the TL kernels a launch takes: the joint one (base state
and perturbation), or the base-only one that :func:`model.step_base` runs
to rebuild an adjoint's tape.  The base-only build compiles each unit with
``-DWRFPLUS_TL=0``, which skips every perturbation store, so the compiler
drops the tangent arithmetic and loads; its base words are the joint
build's.  The default build's options are unchanged."""

from __future__ import annotations

import contextlib
import threading

_local = threading.local()


def base_only_active() -> bool:
    return bool(getattr(_local, "base_only", False))


def tl_options() -> tuple:
    """Extra compile options for the current build."""
    return ("-DWRFPLUS_TL=0",) if base_only_active() else ()


@contextlib.contextmanager
def base_only():
    """Launch the base-only TL kernels in this block (this thread)."""
    prev = base_only_active()
    _local.base_only = True
    try:
        yield
    finally:
        _local.base_only = prev
