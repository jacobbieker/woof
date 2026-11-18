"""The YSU column tile width moves no word, and each backend's tile rule.

``woof.core.ysu.launch_ysu`` launches the columns in tiles of
``ysu_tile_columns`` columns (one thread per column, the tile passed as a
column offset).  On CUDA the tile is the occupancy-bounded 16 blocks per SM;
on Metal it is ``YSU_TILE_BLOCKS_PER_CORE_METAL`` blocks per core, which puts
a 228 x 276 domain in one launch instead of four.  The identity test runs in
child processes under fast, strict and WRF-exact numerics (tests/_numerics_child),
so nothing here imports the YSU launcher at module scope: that import fixes
the numerics mode.
"""

from __future__ import annotations

import numpy as np
import pytest

import _numerics_child


@pytest.mark.gpu
def test_tile_width_keeps_every_word():
    """One block per launch (35 launches, a partial last one), three blocks,
    the backend's own rule (Metal: one launch here; CUDA: the
    occupancy-bounded rule) and the whole domain in one launch give the same
    words, with the workspace poisoned with NaN so a column that read
    another column's or another tile's slots would show."""
    cp = pytest.importorskip("cupy")
    from woof.core import ysu as Y
    from test_ysu_workspace import _assert_bitwise, _bits, _demo_columns

    col, surf = _demo_columns(41, 30, 37, seed=11)
    assert (30 * 37) % Y.YSU_BLOCK, "the one-block arm needs a partial last tile"
    real_empty, original = cp.empty, Y.ysu_tile_columns

    def poisoned(shape, dtype=float, *a, **kw):
        arr = real_empty(shape, dtype, *a, **kw)
        if np.dtype(dtype).kind == "f":
            arr.fill(np.nan)
        return arr

    runs = {}
    for name, width in (("one block", lambda fn, n: Y.YSU_BLOCK),
                        ("three blocks", lambda fn, n: 3 * Y.YSU_BLOCK),
                        ("backend rule", original),
                        ("whole domain", lambda fn, n: n)):
        cp.empty, Y.ysu_tile_columns = poisoned, width
        try:
            runs[name] = _bits(Y.launch_ysu(**col, **surf, dt=30.0))
        finally:
            cp.empty, Y.ysu_tile_columns = real_empty, original
    assert np.isfinite(runs["whole domain"]["dtheta"]).all()
    for name in ("one block", "three blocks", "backend rule"):
        _assert_bitwise(runs["whole domain"], runs[name],
                        f"the {name} tile changed the result")


@pytest.mark.parametrize(*_numerics_child.MODES)
def test_tile_width_keeps_every_word_in_every_numerics_mode(numerics, exact):
    _numerics_child.run(__file__, ["test_tile_width_keeps_every_word"], numerics, exact)


def test_metal_tile_is_blocks_per_core_and_cuda_keeps_its_rule(monkeypatch):
    """Metal launches YSU_TILE_BLOCKS_PER_CORE_METAL blocks per core (no
    occupancy bound), WOOF_METAL_YSU_TILE_BLOCKS overrides it, and every
    other backend keeps the occupancy-bounded CUDA rule."""
    cp = pytest.importorskip("cupy")
    import woof.backend as wb
    from woof.core import ysu as Y

    class _Device:
        attributes = {"MultiProcessorCount": 40}

    monkeypatch.setattr(cp.cuda, "Device", _Device)
    monkeypatch.delenv("WOOF_METAL_YSU_TILE_BLOCKS", raising=False)
    monkeypatch.setattr(wb, "active", lambda: "mlx")
    assert Y.ysu_tile_columns(None, 10**6) == 40 * Y.YSU_TILE_BLOCKS_PER_CORE_METAL * Y.YSU_BLOCK
    assert Y.ysu_tile_columns(None, 62_928) == 62_928
    assert Y.ysu_tile_columns(None, 5) == Y.YSU_BLOCK
    monkeypatch.setenv("WOOF_METAL_YSU_TILE_BLOCKS", "16")
    assert Y.ysu_tile_columns(None, 62_928) == 40 * 16 * Y.YSU_BLOCK
    for bad in ("0", "many"):
        monkeypatch.setenv("WOOF_METAL_YSU_TILE_BLOCKS", bad)
        with pytest.raises(ValueError):
            Y.ysu_tile_columns(None, 62_928)
    monkeypatch.setattr(wb, "active", lambda: "cuda")
    monkeypatch.setattr(wb, "active_blocks_per_multiprocessor", lambda *a, **k: 13)

    class _Fn:
        kernel = type("K", (), {"ptr": 0})()

    try:
        from cupy_backends.cuda.api import driver  # noqa: F401  (the rule's own guard)
    except Exception:                              # noqa: BLE001
        expected = 40 * Y.YSU_TILE_BLOCKS_PER_SM * Y.YSU_BLOCK
    else:
        expected = 40 * 13 * Y.YSU_BLOCK
    assert Y.ysu_tile_columns(_Fn(), 10**6) == expected
