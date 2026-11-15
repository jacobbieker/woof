"""The by-value garbage-column scrub writes exactly what the device-table scrub writes.

``SolveRegionKernels.scrub_columns`` restores ``array[..., solve] = pool`` on every padded argument of a
regional dycore launch.  Since 2026-11-15 it passes its descriptors by value
(``solve_region_scrub_words_v841``, the rk_copy_words form), so on Metal the launch is ordered by its words
instead of draining the queue before and after; ``WOOF_HEX_SCRUB_WORDS=0`` keeps the device-table launch.
Both must leave every word of every array identical, including more rows than one launch carries.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _arrays(seed):
    rng = np.random.default_rng(seed)
    # 40 rows: two by-value launches; 129 and 257 rows span more than one 128-thread block
    shapes = [(56, 41), (57, 41), (41,), (3, 5, 30), (56, 30), (129, 30), (257, 41), (3, 43, 30)] * 5
    return [rng.normal(0.0, 1.0, shape).astype(np.float32) for shape in shapes]


def _scrub(monkeypatch, words, host):
    from woof.hex.cuda_backend import KernelCache
    from woof.hex.cuda_solve_region_v841 import SolveRegionKernels

    monkeypatch.setenv("WOOF_HEX_SCRUB_WORDS", "1" if words else "0")
    kernels = SolveRegionKernels(KernelCache())
    device = [cp.asarray(a) for a in host]
    pools = [np.float32(1.0), np.float32(0.0), np.float32(-2.5)]
    entries = [(int(a.data.ptr), int(a.size // a.shape[-1]), int(a.shape[-1]), int(a.shape[-1]) - 1 - (i % 2),
                float(pools[i % 3])) for i, a in enumerate(device)]
    kernels.scrub_columns(entries)
    return [cp.asnumpy(a) for a in device], entries


def test_by_value_scrub_matches_the_table_scrub_word_for_word(monkeypatch):
    host = _arrays(7)
    by_value, entries = _scrub(monkeypatch, True, host)
    by_table, _ = _scrub(monkeypatch, False, host)
    assert len(entries) > 32
    for got, want in zip(by_value, by_table):
        assert got.tobytes() == want.tobytes()
    for got, original, (_p, _rows, stride, solve, pool) in zip(by_value, host, entries):
        flat_got = got.reshape(-1, stride)
        flat_in = original.reshape(-1, stride)
        assert np.all(flat_got[:, solve] == np.float32(pool))
        keep = np.ones(stride, dtype=bool)
        keep[solve] = False
        assert flat_got[:, keep].tobytes() == flat_in[:, keep].tobytes()
