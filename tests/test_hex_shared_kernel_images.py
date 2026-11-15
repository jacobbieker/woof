"""A forecast binds the box's kernel image cache, not its own scratch (woof.hex.cuda_backend.cupy_cache)."""

from __future__ import annotations

from pathlib import Path

from woof.hex.cuda_backend import cupy_cache


def test_forecast_images_live_in_the_box_kernel_root(tmp_path, monkeypatch):
    monkeypatch.delenv(cupy_cache.SHARED_ENV, raising=False)
    monkeypatch.setenv("WOOF_HEX_KERNEL_CACHE_ROOT", str(tmp_path / "box"))
    scratch = tmp_path / "run-1" / "scratch"
    # Two runs in fresh work dirs share one image cache, so the second compiles nothing again.
    assert cupy_cache.forecast_directory(scratch) == tmp_path / "box" / "cupy"
    assert cupy_cache.forecast_directory(tmp_path / "run-2" / "scratch") == tmp_path / "box" / "cupy"


def test_the_default_box_root_is_the_kernel_stores(tmp_path, monkeypatch):
    from woof.hex.kernel_warm import cache_root

    for name in (cupy_cache.SHARED_ENV, "WOOF_HEX_KERNEL_CACHE_ROOT", "WOOF_HEX_CACHE_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cupy_cache.forecast_directory(tmp_path / "scratch") == cache_root() / "cupy"
    assert Path(tmp_path / "xdg") in cupy_cache.forecast_directory(tmp_path / "scratch").parents


def test_switch_off_keeps_the_per_run_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv(cupy_cache.SHARED_ENV, "0")
    monkeypatch.setenv("WOOF_HEX_KERNEL_CACHE_ROOT", str(tmp_path / "box"))
    assert cupy_cache.forecast_directory(tmp_path / "scratch") == tmp_path / "scratch"
