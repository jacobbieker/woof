"""Interpreter-startup hook for woof's facade backends in development
environments (pixi activation puts this directory on PYTHONPATH).

Installed wheels carry the same line as ``woof-backend.pth``; an editable
install has no place for it, so the pixi ``mlx`` and ``cpu`` environments
use this file instead.  With WOOF_BACKEND=mlx|cpu, ``import woof`` installs
the facade, so ``import cupy`` resolves to it in every process -- tools and
test subprocesses that import cupy before woof included.
"""
import os

if os.environ.get("WOOF_BACKEND", "").strip().lower() in ("mlx", "cpu"):
    __import__("woof")
