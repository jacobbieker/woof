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

import os

import numpy as np
import pytest

import _numerics_child


def _restore_env(saved):
    if saved is None:
        os.environ.pop("WOOF_METAL_YSU_TILE_BLOCKS", None)
    else:
        os.environ["WOOF_METAL_YSU_TILE_BLOCKS"] = saved


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
    from woof.core.physics_inventory import ysu_metal_tile_columns
    real_empty, original = cp.empty, Y.ysu_tile_columns
    saved = os.environ.get("WOOF_METAL_YSU_TILE_BLOCKS")

    def poisoned(shape, dtype=float, *a, **kw):
        arr = real_empty(shape, dtype, *a, **kw)
        if np.dtype(dtype).kind == "f":
            arr.fill(np.nan)
        return arr

    runs = {}
    # The Metal rule on a one-core device at 8 blocks per core: 256-column
    # tiles, five launches with a 86-column tail, so its geometry runs
    # multi-launch on every backend (on a real chip it covers this domain
    # in one launch, the "backend rule" arm).
    metal_rule = lambda fn, n, nz=None: ysu_metal_tile_columns(1, n, nz)  # noqa: E731
    for name, width in (("one block", lambda fn, n, nz=None: Y.YSU_BLOCK),
                        ("three blocks", lambda fn, n, nz=None: 3 * Y.YSU_BLOCK),
                        ("metal rule, one core", metal_rule),
                        ("backend rule", original),
                        ("whole domain", lambda fn, n, nz=None: n)):
        cp.empty, Y.ysu_tile_columns = poisoned, width
        os.environ["WOOF_METAL_YSU_TILE_BLOCKS"] = "8"
        try:
            runs[name] = _bits(Y.launch_ysu(**col, **surf, dt=30.0))
        finally:
            cp.empty, Y.ysu_tile_columns = real_empty, original
            _restore_env(saved)
    assert np.isfinite(runs["whole domain"]["dtheta"]).all()
    for name in ("one block", "three blocks", "metal rule, one core", "backend rule"):
        _assert_bitwise(runs["whole domain"], runs[name],
                        f"the {name} tile changed the result")


@pytest.mark.gpu
@pytest.mark.parametrize(*_numerics_child.MODES)
def test_tile_width_keeps_every_word_in_every_numerics_mode(numerics, exact):
    _numerics_child.run(__file__, ["test_tile_width_keeps_every_word"], numerics, exact)


def test_metal_tile_is_blocks_per_core_and_cuda_keeps_its_rule(monkeypatch):
    """Metal launches YSU_TILE_BLOCKS_PER_CORE_METAL blocks per core (no
    occupancy bound), WOOF_METAL_YSU_TILE_BLOCKS overrides it, and every
    other backend keeps the occupancy-bounded CUDA rule."""
    cp = pytest.importorskip("cupy")
    import woof.backend as wb
    from woof.core import physics_inventory as P
    from woof.core import ysu as Y

    class _Device:
        attributes = {"MultiProcessorCount": 40}

    monkeypatch.setattr(cp.cuda, "Device", _Device)
    monkeypatch.delenv("WOOF_METAL_YSU_TILE_BLOCKS", raising=False)
    monkeypatch.setattr(wb, "active", lambda: "mlx")
    assert Y.ysu_tile_columns(None, 10**6) == 40 * P.YSU_TILE_BLOCKS_PER_CORE_METAL * Y.YSU_BLOCK
    assert Y.ysu_tile_columns(None, 10**6, 49) == 81_920
    assert Y.ysu_tile_columns(None, 62_928, 49) == 62_928
    assert Y.ysu_tile_columns(None, 5, 49) == Y.YSU_BLOCK
    # The byte cap: whole blocks under YSU_TILE_BYTES_CAP_METAL at nz = 256.
    capped = Y.ysu_tile_columns(None, 10**6, 256)
    assert capped % Y.YSU_BLOCK == 0 and capped < 81_920
    assert P.ysu_workspace_floats(256, capped) * 4 <= P.YSU_TILE_BYTES_CAP_METAL
    assert P.ysu_workspace_floats(256, capped + Y.YSU_BLOCK) * 4 > P.YSU_TILE_BYTES_CAP_METAL
    monkeypatch.setenv("WOOF_METAL_YSU_TILE_BLOCKS", "16")
    assert Y.ysu_tile_columns(None, 62_928, 49) == 40 * 16 * Y.YSU_BLOCK
    for bad in ("0", "many", "16.0"):
        monkeypatch.setenv("WOOF_METAL_YSU_TILE_BLOCKS", bad)
        with pytest.raises(ValueError, match="WOOF_METAL_YSU_TILE_BLOCKS"):
            Y.ysu_tile_columns(None, 62_928, 49)
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


def test_preflight_prices_the_metal_tile(monkeypatch):
    """A Metal device profile prices the tile it launches (one 62,928-column
    launch on 40 cores at nz = 49), not CUDA's 16 blocks per core; a CUDA
    profile keeps its pricing, whichever backend prices it."""
    from types import SimpleNamespace

    from woof.core import physics_inventory as P
    from woof.core.preflight import ysu_column_workspace_bytes

    run = SimpleNamespace(bl_pbl_physics=1, nx=276, ny=228, nz=49)
    exp = SimpleNamespace(domains=[SimpleNamespace(run=run)])
    monkeypatch.delenv("WOOF_METAL_YSU_TILE_BLOCKS", raising=False)
    profile = SimpleNamespace(multiprocessor_count=40, backend="cuda")
    assert ysu_column_workspace_bytes(exp, profile=profile) == (
        P.ysu_workspace_floats(49, 40 * P.YSU_TILE_BLOCKS_PER_SM * P.YSU_BLOCK) * 4)
    profile = SimpleNamespace(multiprocessor_count=40, backend="mlx")
    assert ysu_column_workspace_bytes(exp, profile=profile) == (
        P.ysu_workspace_floats(49, 276 * 228) * 4)
