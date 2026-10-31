"""Hex partial-depth band (Phase D5): the kernel unit's bookkeeping.

* the unit's source is pinned, its entry points listed in source order, its
  loader options none beyond the cache's base (the process numerics);
* the existing kernel units are untouched: no regional unit carries a band
  entry point (the regional set digest is held by test_cuda_regional_kernels);
* A6000 frames: every entry point compiles to a 0 B local frame, MEASURED on
  the RTX A6000 (sm_86, NVRTC 12.9) in both numerics modes (fast and strict);
* the allocation inventory of the band runner, per function;
* the restart classification covers every attribute the band holds, and the
  "rebuilt" buffers are what a restart rebuilds, word for word.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import re

import numpy as np
import pytest

from woof.hex import cuda_partial_nest_v841 as unit
from woof.hex import partial_nest_run

#: sha256 of CUDA_PARTIAL_NEST_SOURCE.  A change to any band kernel moves it;
#: re-pin only with the oracle (test_partial_nest_oracle_v841.py) and the
#: envelope receipts re-measured.
SOURCE_SHA256 = "b9784313edf9970293a4c43ed9cdb16b1aec05713146cd4654faeb9acc68a4bc"

#: local_size_bytes per entry point, RTX A6000 sm_86, CuPy NVRTC 12.9,
#: WOOF_HEX_NUMERICS=fast and =strict alike (2026-10-31).
A6000_FRAMES = dict.fromkeys(unit.KERNELS, 0)

#: Device allocations (``cp.zeros`` / ``empty`` / ``*_like`` / ``array`` /
#: ``asarray`` / ``ascontiguousarray``) per function of partial_nest_run.py.
#: Construction allocates the tables and planes; each parent step allocates
#: its snapshot and frame temporaries and the feedback's recovery outputs;
#: each child step allocates the scalar flux's time-interpolated planes and
#: RK3's second-solve interface theta.
ALLOCATIONS = {
    "OnlineDrivingState._derive": 6,
    "ChildTop.__init__": 1,
    "ChildTop.coefficients_top": 2,
    "ChildTop.after": 1,
    "PartialNestHex.__init__": 23,
    "PartialNestHex._parent_winds": 3,
    "PartialNestHex._child_edge_normal": 2,
    "PartialNestHex.c_cos_pad": 2,
    "PartialNestHex.snapshot": 2,
    "PartialNestHex.frame": 9,
    "PartialNestHex.feedback": 12,
}
_ALLOCATORS = {"zeros", "empty", "empty_like", "zeros_like", "array", "asarray",
               "ascontiguousarray"}


def test_unit_source_is_pinned():
    assert hashlib.sha256(unit.CUDA_PARTIAL_NEST_SOURCE.encode()).hexdigest() == SOURCE_SHA256


def test_entry_points_are_listed_in_source_order():
    found = tuple(re.findall(r'extern "C" __global__ void (\w+)\(', unit.CUDA_PARTIAL_NEST_SOURCE))
    assert found == unit.KERNELS


def test_loader_options_are_the_cache_base_only():
    assert unit.OPTIONS == ()
    assert unit.MODULE_KEY == "hexcore.cuda_partial_nest_v841"


def test_no_existing_kernel_unit_carries_a_band_entry_point():
    from woof.hex.cuda_backend import regional_admission
    sources = regional_admission.REGIONAL_KERNEL_SOURCES
    texts = sources.values() if isinstance(sources, dict) else sources
    for text in texts:
        assert "pn_" not in str(text)


def _allocations() -> dict[str, int]:
    tree = ast.parse(inspect.getsource(partial_nest_run))
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for fn in node.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            n = 0
            for call in ast.walk(fn):
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and call.func.attr in _ALLOCATORS):
                    continue
                owner = call.func.value
                if (isinstance(owner, ast.Name) and owner.id == "cp") or (
                        isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name)
                        and owner.func.id == "_cp"):
                    n += 1
            if n:
                out[f"{node.name}.{fn.name}"] = n
    return out


def test_allocation_inventory():
    assert _allocations() == ALLOCATIONS


def _stored_attributes(cls) -> set[str]:
    tree = ast.parse(inspect.getsource(cls))
    return {n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
            and isinstance(n.value, ast.Name) and n.value.id == "self"}


@pytest.mark.parametrize("cls", [partial_nest_run.PartialNestHex, partial_nest_run.ChildTop])
def test_restart_classification_covers_every_attribute(cls):
    table = partial_nest_run.RESTART_SCRATCH[cls.__name__]
    assert _stored_attributes(cls) == set(table)
    assert set(table.values()) <= {"rebuilt", "table", "transient", "host", "diagnostic"}


def _device():
    pytest.importorskip("cupy")
    from woof.hex.cuda_backend import KernelCache
    from woof.hex.cuda_backend.runtime import CudaRefusal
    try:
        KernelCache()
    except CudaRefusal as error:
        pytest.skip(f"no admitted hex device: {error}")


def test_a6000_frames_are_zero_bytes():
    _device()
    from woof.backend import requested
    if requested() != "cuda":
        pytest.skip("local frames are an NVRTC measurement")
    from woof.hex.cuda_backend import KernelCache
    kernels = unit.PartialNestKernels(KernelCache())
    frames = {name: int(kernels.kernel(name).attributes["local_size_bytes"])
              for name in unit.KERNELS}
    assert frames == A6000_FRAMES


def test_rebuilt_buffers_are_what_a_restart_rebuilds():
    """After a coupled parent step the start planes and the lateral frame at
    the parent's time are exactly what a restart computes from the parent."""
    _device()
    import cupy as cp
    try:
        from woof.verify import hex_partial_nest_envelope as envelope
        envelope.ensure_meshes()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    nest, *_ = envelope.build("bubble")
    partial_nest_run.run(nest, 2)
    planes = partial_nest_run.Planes(cp.zeros_like(nest.a.cells), cp.zeros_like(nest.a.edges),
                                     cp.zeros_like(nest.a.scalars))
    nest.snapshot(planes)
    for name in ("cells", "edges", "scalars"):
        assert np.array_equal(cp.asnumpy(getattr(planes, name)), cp.asnumpy(getattr(nest.a, name)))
    when = max(nest.pool.frames)
    frame = nest.frame(when)
    for key, value in frame.fields.items():
        assert np.array_equal(cp.asnumpy(value), cp.asnumpy(nest.pool.frames[when].fields[key])), key
