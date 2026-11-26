"""GPU module replay of WRF v4.8.0 GFL on 72 real columns captured from native WRF v4.8.0 strict.

The breakage it prevents: a transcription, support-header or launcher change
that moves any word of the scheme's output. The oracle
(tests/data/oracles/gfl_wrf480_strict239, PROVENANCE.txt) holds a captured
cu_grell_freitas_li call (NCAR CONUS 12 km benchmark input, step 3) for 72
columns (48 raining, 24 of them whose words move with the gamma function) and
two POST files from the referee's own objects: glibc 2.39 (the referee as
built) and the same objects linked with a correctly rounded tgammaf.

* Strict proof mode against a glibc 2.39 referee (GPUWM_WRF_EXACT=1, the
  default GPUWM_WRF_GLIBC): every word equals the referee's.
* Every other mode uses the engine's correctly rounded gamma: every word equals
  the correctly rounded referee's, and the glibc 2.39 words differ only in the
  columns where the two native POST files differ.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests/data/oracles/gfl_wrf480_strict239"


def _replay(cp, gfl, tile=1, order=None):
    """Run the captured call (its 72 columns repeated ``tile`` times, or the captured columns ``order`` lists, in
    that order); return got, want, ncol, the oracle."""
    sys.path.insert(0, str(ROOT / "tools/gfl_873b414"))
    from replay_gfl import read_pre, read_post, LEVEL_IN, SURF_IN, SURF_INOUT, INT_INOUT, LEVEL_INOUT
    from woof import wrf_exact

    d, a, shape = read_pre(ORACLE / "pre.bin")
    ni, nk, nj = shape
    want_cr = read_post(ORACLE / "post_native_crgamma.bin", shape)
    want_glibc = read_post(ORACLE / "post_native_glibc239.bin", shape)
    ncol = ni

    def lev(x):
        if order is not None:
            return cp.ascontiguousarray(cp.asarray(x[0].T[order]))
        return cp.ascontiguousarray(cp.asarray(np.tile(x[0].T, (tile, 1))))

    def srf(x):
        if order is not None:
            assert x[0].ndim == 1
            return cp.ascontiguousarray(cp.asarray(x[0][order]))
        return cp.ascontiguousarray(cp.asarray(np.tile(x[0], tile)))

    f = {n: lev(a[n]) for n in LEVEL_IN + ("rqvften", "rthften", "rthblten", "rqvblten", "rthraten",
                                          "gdc", "gdc2") + LEVEL_INOUT}
    f.update({n: srf(a[n]) for n in SURF_IN + SURF_INOUT + ("kpbl",) + INT_INOUT})
    for n, typ in (("itimestep", cp.int32), ("ichoice_deep", cp.int32), ("ishallow_g3", cp.int32),
                   ("dt", cp.float32), ("dxcell", cp.float32), ("xlv", cp.float32), ("cp", cp.float32),
                   ("g", cp.float32), ("r_v", cp.float32)):
        f[n] = cp.asarray([d[n]], dtype=typ)
    out = gfl.launch_columns(f, ncol=ncol * tile if order is None else len(order), nz=d["kte"],
                             vertical=(d["kms"], d["kme"], d["kds"], d["kde"], d["kts"], d["kte"]))
    got = {n: cp.asnumpy(f[n]).T[None] for n in ("gdc", "gdc2") + LEVEL_INOUT}
    got.update({n: cp.asnumpy(f[n])[None] for n in SURF_INOUT + INT_INOUT})
    referee_gamma = wrf_exact.ENABLED and not wrf_exact.referee_correctly_rounded_trig()
    return got, (want_glibc if referee_gamma else want_cr), want_cr, want_glibc, ncol, referee_gamma, out


@pytest.mark.gpu
def test_gfl_replays_the_native_wrf480_columns_word_for_word():
    cp = pytest.importorskip("cupy")
    from woof.core import gfl_873b414 as gfl

    got, want, want_cr, want_glibc, ncol, referee_gamma, _ = _replay(cp, gfl)
    columns = json.loads((ORACLE / "columns.json").read_text())
    assert columns["columns"] == ncol
    different = {n: int(np.count_nonzero(got[n].view(np.uint32) != want[n].view(np.uint32))) for n in got}
    assert sum(different.values()) == 0, different
    if referee_gamma:
        return
    gamma_columns = np.zeros(ncol, bool)
    for n in got:
        x, y = want_cr[n].view(np.uint32), want_glibc[n].view(np.uint32)
        gamma_columns |= (x != y).reshape(-1, ncol).any(axis=0) if x.ndim == 3 else (x != y)[0]
    for n in got:
        moved = (got[n].view(np.uint32) != want_glibc[n].view(np.uint32))
        moved = moved.reshape(-1, ncol).any(axis=0) if moved.ndim == 3 else moved[0]
        assert not (moved & ~gamma_columns).any(), n
    assert gamma_columns.sum() == columns["gamma_sensitive"]


@pytest.mark.gpu
def test_gfl_columns_that_spill_keep_every_word(monkeypatch):
    """A column deeper than its scratch share continues in the host spill pool, word for word.

    The breakage it prevents: upstream 1.1.7 spilled to the CUDA device heap
    (in-kernel malloc/free), which Metal does not have, so every Metal unit with
    the GFL lane failed to translate. The pool replaces it on both backends.
    Here the share is cut to 8 KiB so every column spills (the captured columns
    need 15-43 KB), and the 72 columns run four times (288 columns, three
    slabs) so pool entries past the first slab can be taken; every word must
    still equal the oracle's.
    """
    cp = pytest.importorskip("cupy")
    from woof.core import gfl_873b414 as gfl

    share = 8192
    monkeypatch.setattr(gfl, "_share_bytes", lambda storage, compute: share)
    monkeypatch.setattr(gfl, "SPILL_SHARES", 16)
    tile = 4
    got, want, _, _, ncol, _, out = _replay(cp, gfl, tile=tile)
    highwater = cp.asnumpy(out["scratch_highwater"])
    spilled = int((highwater > share).sum())
    assert out["share_bytes"] == share and spilled > 128, highwater
    # Every column returned its pool entry when it ended.
    assert not cp.asnumpy(gfl._SPILL_MAP[cp.cuda.runtime.getDevice()]).any()
    for n in got:
        for k in range(tile):
            part = got[n][..., k * ncol:(k + 1) * ncol]
            assert np.array_equal(part.view(np.uint32), want[n].view(np.uint32)), (n, k)


@pytest.mark.gpu
def test_gfl_pool_entries_return_so_more_columns_spill_than_entries(monkeypatch):
    """Over a launch more columns spill than the pool has entries, few at once: every word exact.

    The breakage it prevents: a thread kept its pool entry for the whole launch, so
    a deep-column launch (the 256 MiB cap holds 384 entries at nz=256, 1920 at nz=49,
    against tens of thousands of threads) trapped once that many threads had ever
    spilled. Entries now return when their column ends; the limit is the columns
    spilling at the same moment.

    The captured columns need one of two scratch depths; the share sits between
    them, so only the deeper columns spill. One block runs (128 threads, 4 warps,
    each taking 32 consecutive columns at a time), and every aligned 32-column
    window holds 8 deep columns at rotating lanes: at most 32 spill at once, 32
    entries, while 144 spill over the launch, at lanes across the whole block.
    """
    cp = pytest.importorskip("cupy")
    from woof.core import gfl_873b414 as gfl

    _, _, _, _, ncol, _, base = _replay(cp, gfl)
    need = cp.asnumpy(base["scratch_highwater"]).astype(np.int64)        # each captured column's need
    deep, shallow = np.flatnonzero(need == need.max()), np.flatnonzero(need < need.max())
    share = int(need[shallow].max()) // 16 * 16 + 16
    assert deep.size and shallow.size and share < need.max()
    window, lanes, per_window, windows = 32, 128, 8, 18
    entries = per_window * (lanes // window)
    order, d, s = [], 0, 0
    for w in range(windows):
        hot = {(w * per_window + i) % window for i in range(per_window)}  # the lanes that spill rotate
        for lane in range(window):
            if lane in hot:
                order.append(deep[d % deep.size]); d += 1
            else:
                order.append(shallow[s % shallow.size]); s += 1
    order = np.asarray(order)
    monkeypatch.setattr(gfl, "_share_bytes", lambda storage, compute: share)
    monkeypatch.setattr(gfl, "SPILL_SHARES", 16)
    monkeypatch.setattr(gfl, "_SCRATCH", {})
    monkeypatch.setattr(gfl, "_SPILL", {})
    monkeypatch.setattr(gfl, "_scratch_blocks", lambda *a, **k: 1)
    pool = gfl._spill_pool
    monkeypatch.setattr(gfl, "_spill_pool", lambda device, spill_bytes, slabs: (pool(device, spill_bytes, slabs)[0],
                                                                                 entries))
    got, want, _, _, _, _, out = _replay(cp, gfl, order=order)
    spilled = cp.asnumpy(out["scratch_highwater"]) > share
    assert out["blocks"] == 1 and out["share_bytes"] == share
    assert int(spilled.sum()) == windows * per_window >= 4 * entries         # oversubscribed over the launch
    assert all(int(spilled[w:w + window].sum()) == per_window for w in range(0, order.size, window))
    assert not cp.asnumpy(gfl._SPILL_MAP[cp.cuda.runtime.getDevice()]).any()
    for n in got:
        assert np.array_equal(got[n].view(np.uint32), want[n][..., order].view(np.uint32)), n


