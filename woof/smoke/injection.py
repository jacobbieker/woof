"""Smoke injection profiles: a surface flux spread over a model column.

Each profile turns a surface emission flux ``F`` (kg m-2 s-1, e.g. from
:func:`woof.smoke.emission.pm25_emission_flux`) into a mixing-ratio source
``S[k]`` (kg kg-1 s-1) on the column's ``nz`` layers, normalised so the
column integral is the surface flux::

    sum_k rho[k] * S[k] * (z_w[k+1] - z_w[k]) == F

``z_w`` are interface heights above ground ``(nz+1, ny, nx)``, strictly
increasing, and ``rho`` the layer densities ``(nz, ny, nx)`` (kg m-3).

* :func:`lowest_layer_profile` -- the whole flux into layer 0, which is
  WRF-SFIRE's ``fire_smk_scheme = 0`` (``woof.smoke.emission``) and what a
  resolved fire-atmosphere model does: ``S[0] = F / (rho[0] dz[0])``.
* :func:`uniform_profile` -- uniform in height over ``[z_bot, z_top]``
  clipped to the column, layer ``k`` taking ``overlap_k / (z_top - z_bot)``
  of the flux; a degenerate interval (``z_top <= z_bot`` after clipping)
  puts the whole flux into the layer holding ``z_bot``, so no column loses
  its emission.
* :func:`pbl_uniform_profile` -- :func:`uniform_profile` from the ground to
  ``fraction * pbl_height``: the "mixed through the boundary layer"
  assumption of many regional smoke systems.

These are woof's own, with no WRF counterpart to hold them to.  The CuPy
forms (one kernel launch each, ``kernels/smoke_emission.cu``) and the NumPy
references (suffix ``_np``) evaluate the same real(4) statements in the
same order and agree at zero ULP; both are held to a float64 evaluation of
the same formulas, and to the normalisation above, within float32 rounding
(``tests/test_smoke_physics.py``).  For the Freitas plume-rise injection
see :mod:`woof.smoke.plume_rise`.
"""
from __future__ import annotations

import numpy as np

F = np.float32


def _column_inputs(xp, z_w, rho, flux):
    z_w = xp.ascontiguousarray(xp.asarray(z_w, dtype=xp.float32))
    rho = xp.ascontiguousarray(xp.asarray(rho, dtype=xp.float32))
    if z_w.ndim != 3 or rho.shape != (z_w.shape[0] - 1,) + z_w.shape[1:]:
        raise ValueError(f"z_w must be (nz+1, ny, nx) and rho (nz, ny, nx); "
                         f"got {z_w.shape} and {rho.shape}")
    if not bool(xp.all(z_w[1:] > z_w[:-1])):
        raise ValueError("z_w must increase strictly along axis 0")
    if not bool(xp.all(rho > 0)):
        raise ValueError("rho must be positive")
    flux = xp.ascontiguousarray(xp.broadcast_to(
        xp.asarray(flux, dtype=xp.float32), z_w.shape[1:]))
    return z_w, rho, flux


def _plane(xp, value, shape):
    return xp.ascontiguousarray(xp.broadcast_to(
        xp.asarray(value, dtype=xp.float32), shape))


# --------------------------------------------------------------------------
# references


def lowest_layer_profile_np(z_w, rho, flux) -> np.ndarray:
    """Reference: ``S[0] = F / (rho[0] (z_w[1] - z_w[0]))``, float32."""
    z_w, rho, flux = _column_inputs(np, z_w, rho, flux)
    out = np.zeros(rho.shape, dtype=np.float32)
    out[0] = flux / (rho[0] * (z_w[1] - z_w[0]))
    return out


def uniform_profile_np(z_w, rho, flux, z_bot, z_top) -> np.ndarray:
    """Reference uniform injection over ``[z_bot, z_top]``, float32.

    The statements of ``smoke_inject_uniform`` vectorised over columns:
    clip, depth, per-layer overlap, ``F * (overlap / depth) / (rho * dz)``.
    """
    return _uniform_np(z_w, rho, flux, z_bot, z_top, 1.0)


def _uniform_np(z_w, rho, flux, z_bot, z_top, top_scale):
    z_w, rho, flux = _column_inputs(np, z_w, rho, flux)
    nz = rho.shape[0]
    cols = z_w.shape[1:]
    surface, column_top = z_w[0], z_w[nz]
    bot = np.minimum(np.maximum(_plane(np, z_bot, cols), surface), column_top)
    top = np.minimum(np.maximum(_plane(np, z_top, cols) * F(top_scale),
                                surface), column_top)
    depth = top - bot
    thick = depth > F(0.0)
    safe_depth = np.where(thick, depth, F(1.0))
    out = np.zeros(rho.shape, dtype=np.float32)
    held = np.full(cols, nz - 1)
    found = np.zeros(cols, dtype=bool)
    for k in range(nz):
        lo, hi = z_w[k], z_w[k + 1]
        overlap = np.minimum(hi, top) - np.maximum(lo, bot)
        share = overlap / safe_depth
        value = flux * share / (rho[k] * (hi - lo))
        out[k] = np.where(thick & (overlap > F(0.0)), value, F(0.0))
        here = ~found & (lo <= bot) & (bot < hi)
        held = np.where(here, k, held)
        found |= here
    for k in range(nz):
        lo, hi = z_w[k], z_w[k + 1]
        value = flux / (rho[k] * (hi - lo))
        out[k] = np.where(~thick & (held == k), value, out[k])
    return out


def pbl_uniform_profile_np(z_w, rho, flux, pbl_height,
                           fraction: float = 1.0) -> np.ndarray:
    """Reference: uniform from the ground to ``fraction * pbl_height``."""
    _check_fraction(fraction)
    z_w = np.asarray(z_w, dtype=np.float32)
    return _uniform_np(z_w, rho, flux, z_w[0], pbl_height, fraction)


def _check_fraction(fraction):
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")


# --------------------------------------------------------------------------
# device


def lowest_layer_profile(z_w, rho, flux):
    """:func:`lowest_layer_profile_np` on the device, one kernel launch."""
    import cupy as cp

    from woof.smoke import _kernels

    z_w, rho, flux = _column_inputs(cp, z_w, rho, flux)
    nz, ncol = rho.shape[0], flux.size
    out = cp.empty(rho.shape, dtype=cp.float32)
    _kernels.kernel("smoke_inject_lowest")(
        _kernels.grid(ncol), (_kernels.BLOCK,),
        (z_w, rho, flux, out, np.int32(nz), np.int32(ncol)))
    return out


def uniform_profile(z_w, rho, flux, z_bot, z_top):
    """:func:`uniform_profile_np` on the device, one kernel launch."""
    return _uniform(z_w, rho, flux, z_bot, z_top, 1.0)


def _uniform(z_w, rho, flux, z_bot, z_top, top_scale):
    import cupy as cp

    from woof.smoke import _kernels

    z_w, rho, flux = _column_inputs(cp, z_w, rho, flux)
    nz, ncol = rho.shape[0], flux.size
    bot = _plane(cp, z_bot, flux.shape)
    top = _plane(cp, z_top, flux.shape)
    out = cp.empty(rho.shape, dtype=cp.float32)
    _kernels.kernel("smoke_inject_uniform")(
        _kernels.grid(ncol), (_kernels.BLOCK,),
        (z_w, rho, flux, bot, top, out, F(top_scale), np.int32(nz),
         np.int32(ncol)))
    return out


def pbl_uniform_profile(z_w, rho, flux, pbl_height, fraction: float = 1.0):
    """:func:`pbl_uniform_profile_np` on the device, one kernel launch.

    The kernel forms ``pbl_height * fraction`` itself, the reference's one
    real(4) product.
    """
    import cupy as cp

    _check_fraction(fraction)
    z_w = cp.asarray(z_w, dtype=cp.float32)
    return _uniform(z_w, rho, flux, z_w[0], pbl_height, fraction)
