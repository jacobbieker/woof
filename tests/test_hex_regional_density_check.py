"""The regional FCT target-density test raises the same flag without a host wait.

``woof.hex.cuda_transport_v841._regional_target_density_check`` once selected
the kept cells by boolean indexing, which sizes its result on the host and so
synchronised every regional step.  It now ORs a broadcast mask over the same
words.  These cases hold the flag to the boolean-indexing answer, computed
here on the host, over finite, zero, negative, NaN and infinite densities in
and out of the specified zone and on the garbage cell.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import requires_gpu


def _reference(target, nlev, ncells, mask, garbage):
    keep = mask[:ncells] < 6
    if mask.size < ncells:
        keep = np.concatenate([keep, np.zeros(ncells - mask.size, dtype=bool)])
    if garbage is not None and 0 <= garbage < ncells:
        keep[garbage] = False
    values = target.reshape(nlev, ncells)[:, keep]
    return int(np.any(~(np.isfinite(values) & (values > 0))))


def _cases():
    rng = np.random.default_rng(20261108)
    nlev, ncells = 5, 40
    mask = rng.integers(0, 8, size=ncells).astype(np.int32)
    garbage = int(np.flatnonzero(mask < 6)[0])
    clean = (rng.random((nlev, ncells)) + 0.5).astype(np.float32)
    yield "clean", clean, mask, garbage
    zone = np.flatnonzero(mask >= 6)
    kept = [c for c in np.flatnonzero(mask < 6) if c != garbage]
    for name, bad in (("zero", 0.0), ("negative", -1.0), ("nan", np.nan), ("inf", np.inf)):
        in_zone = clean.copy()
        in_zone[2, zone[0]] = bad
        yield f"{name}-specified-zone", in_zone, mask, garbage
        on_garbage = clean.copy()
        on_garbage[0, garbage] = bad
        yield f"{name}-garbage-cell", on_garbage, mask, garbage
        in_kept = clean.copy()
        in_kept[nlev - 1, kept[-1]] = bad
        yield f"{name}-kept-cell", in_kept, mask, garbage
    yield "short-mask", clean, mask[: ncells - 3], None


@requires_gpu
@pytest.mark.parametrize("name,target,mask,garbage", list(_cases()), ids=lambda v: v if isinstance(v, str) else "")
def test_flag_matches_boolean_indexing(name, target, mask, garbage):
    import cupy as cp
    from woof.hex.cuda_transport_v841 import _regional_target_density_check

    nlev, ncells = target.shape
    flag = cp.zeros((1,), dtype=cp.int32)
    _regional_target_density_check(cp, cp.asarray(target.reshape(-1)), nlev, ncells,
                                   cp.asarray(mask), garbage, flag)
    assert int(cp.asnumpy(flag)[0]) == _reference(target.reshape(-1), nlev, ncells, mask, garbage)


@requires_gpu
def test_flag_only_rises():
    import cupy as cp
    from woof.hex.cuda_transport_v841 import _regional_target_density_check

    target = np.ones((3, 8), dtype=np.float32)
    flag = cp.ones((1,), dtype=cp.int32)
    _regional_target_density_check(cp, cp.asarray(target.reshape(-1)), 3, 8,
                                   cp.zeros(8, dtype=cp.int32), None, flag)
    assert int(cp.asnumpy(flag)[0]) == 1
