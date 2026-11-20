"""The gradient test kit (plan P1.3, section 3(g)).

For a differentiable map ``f`` taking a tuple of tensors to a tuple of
tensors:

* :func:`dot_product` -- the adjoint test ``<TL dx, dy> = <dx, AD dy>``
  with the tangent-linear from forward-mode AD
  (``torch.autograd.forward_ad``) and the adjoint from reverse mode, so
  the two sides come from different code paths;
* :func:`tl_convergence` -- ``||f(x + a dx) - f(x)|| / ||a TL dx||`` over
  decades of ``a`` (tends to 1 while the branch holds);
* :func:`taylor` -- the gradient check of a scalar ``J``:
  ``(J(x + a h) - J(x)) / (a <grad J, h>)``;
* :func:`directional` -- a directional derivative by central differences
  of any callable (woof's own kernel included), for comparison with
  ``TL dx``.

Everything is generic over torch's devices and dtypes; the binary64 runs
use float64 tensors (CPU or CUDA; torch-MPS has no float64).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

Tensors = Sequence[torch.Tensor]


def _tuple(value) -> tuple:
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _f64(x: torch.Tensor) -> torch.Tensor:
    """``x`` in float64 (on the CPU for torch-MPS, which has no float64)."""
    return x.cpu().double() if x.device.type == "mps" else x.double()


def _inner(a: Tensors, b: Tensors) -> torch.Tensor:
    return sum((_f64(x) * _f64(y)).sum() for x, y in zip(a, b))


def tangent_linear(f: Callable, x: Tensors, dx: Tensors) -> tuple[tuple, tuple]:
    """``(f(x), TL dx)`` by forward-mode AD."""
    import torch.autograd.forward_ad as fwad
    with fwad.dual_level():
        duals = [fwad.make_dual(a.detach(), d.to(a.dtype)) for a, d in zip(x, dx)]
        out = _tuple(f(*duals))
        primal, tangent = [], []
        for o in out:
            p, t = fwad.unpack_dual(o)
            primal.append(p.detach())
            tangent.append(torch.zeros_like(p) if t is None else t.detach())
    return tuple(primal), tuple(tangent)


def adjoint(f: Callable, x: Tensors, dy: Tensors) -> tuple[tuple, tuple]:
    """``(f(x), AD dy)`` by reverse-mode AD (``vjp``)."""
    leaves = [a.detach().clone().requires_grad_(True) for a in x]
    out = _tuple(f(*leaves))
    used = [(o, d) for o, d in zip(out, dy) if o.requires_grad]
    grads = torch.autograd.grad([o for o, _ in used], leaves, [d.to(o.dtype) for o, d in used],
                                allow_unused=True)
    grads = tuple(torch.zeros_like(a) if g is None else g for a, g in zip(leaves, grads))
    return tuple(o.detach() for o in out), grads


@dataclass
class DotProduct:
    tl_dy: float          # <TL dx, dy>
    dx_ad: float          # <dx, AD dy>

    @property
    def relative(self) -> float:
        scale = max(abs(self.tl_dy), abs(self.dx_ad), 1e-300)
        return abs(self.tl_dy - self.dx_ad) / scale


def dot_product(f: Callable, x: Tensors, dx: Tensors, dy: Tensors) -> DotProduct:
    """The adjoint test for ``f`` at ``x``."""
    _, tl = tangent_linear(f, x, dx)
    _, ad = adjoint(f, x, dy)
    return DotProduct(float(_inner(tl, dy)), float(_inner(dx, ad)))


def tl_convergence(f: Callable, x: Tensors, dx: Tensors, alphas: Sequence[float]) -> list[tuple[float, float]]:
    """``[(a, ||f(x + a dx) - f(x)|| / ||a TL dx||), ...]``."""
    base, tl = tangent_linear(f, x, dx)
    norm_tl = float(torch.sqrt(_inner(tl, tl)))
    rows = []
    with torch.no_grad():
        for a in alphas:
            moved = _tuple(f(*[v + a * d.to(v.dtype) for v, d in zip(x, dx)]))
            diff = [m - b for m, b in zip(moved, base)]
            rows.append((float(a), float(torch.sqrt(_inner(diff, diff))) / (abs(a) * norm_tl)))
    return rows


def taylor(J: Callable, x: Tensors, h: Tensors, alphas: Sequence[float]) -> list[tuple[float, float]]:
    """``[(a, (J(x + a h) - J(x)) / (a <grad J, h>)), ...]`` for a scalar ``J``."""
    leaves = [a.detach().clone().requires_grad_(True) for a in x]
    value = J(*leaves)
    grads = torch.autograd.grad(value, leaves, allow_unused=True)
    slope = float(sum((_f64(g) * _f64(d)).sum() for g, d in zip(grads, h) if g is not None))
    rows = []
    with torch.no_grad():
        for a in alphas:
            moved = float(J(*[v + a * d.to(v.dtype) for v, d in zip(x, h)]))
            rows.append((float(a), (moved - float(value)) / (a * slope)))
    return rows


def directional(f: Callable, x: Sequence, dx: Sequence, eps: float) -> tuple:
    """Central difference ``(f(x + eps dx) - f(x - eps dx)) / (2 eps)`` of
    any callable on array tuples (NumPy, CuPy or torch), per output."""
    plus = _tuple(f(*[a + eps * d for a, d in zip(x, dx)]))
    minus = _tuple(f(*[a - eps * d for a, d in zip(x, dx)]))
    return tuple((p - m) / (2.0 * eps) for p, m in zip(plus, minus))
