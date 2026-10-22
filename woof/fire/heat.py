"""Fuel burn, fire heat release and its aggregation: WRF v4.7.1, REAL(4).

Pass 6 of WRF's ``fire_model`` (``phys/module_fr_fire_model.F``) and the
driver's sum onto the atmosphere, on the device and in a NumPy float32
reference that the device is held to word for word:

* ``fuel_left`` (``module_fr_fire_core.F``, ``fuel_left_method = 1``,
  ``fuel_left_irl = fuel_left_jrl = 2``) -- each fire cell is split into
  2 x 2 subcells; at each subcell's corners the level set and the ignition
  time are interpolated bilinearly from the nodes, and
  ``fuel_left_cell_1`` gives the subcell's burning area
  ``A = (1 - sum(lfn) / sum(|lfn|)) / 2`` (clipped to [0, 1]) and its fuel
  left ``A exp(T / fuel_time) + (1 - A)``, ``T`` the mean of the corners'
  ``min(tign - t, 0)`` over burning corners.  The cell's fuel fraction and
  fire area are the subcell means.  The fuel burnt in the step is the drop
  in the fuel fraction;
* ``heat_fluxes`` (``module_fr_fire_phys.F``) -- with the fuel load
  ``fgip`` (kg m-2) and moisture ``fmc_g``, ``bmst = fmc_g / (1 + fmc_g)``::

      GRNHFX_fire = (fgip burnt / dt) (1 - bmst) cmbcnst
      GRNQFX_fire = (bmst + (1 - bmst) 0.56) (fgip burnt / dt) XLV

  with ``cmbcnst = 17.433e6`` J/kg (dry fuel heat of combustion), ``XLV =
  2.5e6`` J/kg and 0.56 the water produced per kg of dry fuel burnt;
* ``sum_2d_cells`` (``module_fr_fire_util.F``) and the scaling of
  ``module_fr_fire_driver.F:941-953`` -- each atmospheric cell's flux is
  ``fire_atm_feedback * (sum of its sr_x * sr_y fire cells) * (1/(ir*jr))``,
  summed in WRF's fixed order (rows of the cell outer, columns inner) on
  both paths, with no atomics; ``AVG_FUEL_FRAC`` is the same mean of the
  fuel fraction.  WRF has no canopy fluxes yet: ``CANHFX``/``CANQFX`` are
  zero, as the driver sets them.

Called by :func:`woof.fire.levelset.step`.  The names follow WRF-SFIRE:
``fgrnhfx``/``fgrnqfx`` on the fire mesh, ``GRNHFX``/``GRNQFX``/``CANHFX``/
``CANQFX`` on the atmospheric grid, W m-2.
"""
from __future__ import annotations

import numpy as np

from woof.fire.levelset import (F, FireMesh, FireOptions, FireParams,
                                _expf, _grid2, _kernel, _lit)

#: module_fr_fire_phys.F's ``cmbcnst`` (DATA 17.433e+06): J per kg dry fuel.
CMBCNST = _lit("17.433e+06")
#: module_model_constants.F's ``XLV``: latent heat of vaporisation, J/kg.
XLV = _lit("2.5E6")
#: heat_fluxes's 56 % of the cellulose molecule's mass that is water.
COMBUSTION_WATER = _lit(".56")


def _bilin(tx, ty, a00, a01, a10, a11):
    one = F(1.0)
    return ((one - tx) * (one - ty) * a00 + (one - tx) * ty * a01
            + tx * (one - ty) * a10 + tx * ty * a11)


#: fuel_left's subcells in its loop order (isubcl outer, jsubcl inner):
#: (node offset di, dj; tx, ty, txx, tyy).
_SUBCELLS = (
    (-1, -1, F(0.5), F(0.5), F(1.0), F(1.0)),
    (-1, 0, F(0.5), F(0.0), F(1.0), F(0.5)),
    (0, -1, F(0.0), F(0.5), F(0.5), F(1.0)),
    (0, 0, F(0.0), F(0.0), F(0.5), F(0.5)),
)


def _fuel_cell_cpu(l00, l01, l10, l11, g00, g01, g10, g11, time_now,
                   fuel_time):
    """fuel_left_cell_1 over arrays: ``(fuel left, fire area)``."""
    zero = F(0.0)

    def age(lfn, tign):
        t = tign - time_now
        return np.where((lfn > 0) | (t > 0), zero, t)

    t00, t01 = age(l00, g00), age(l01, g01)
    t10, t11 = age(l10, g10), age(l11, g11)
    ps = l00 + l01 + l10 + l11
    aps = np.abs(l00) + np.abs(l01) + np.abs(l10) + np.abs(l11)
    aps = np.fmax(aps, F(2.0 ** -126))
    area = (-(ps / aps) + F(1.0)) / F(2.0)
    area = np.fmax(area, zero)
    area = np.fmin(area, F(1.0))
    ta = F(0.25) * (t00 + t01 + t10 + t11)
    out = np.where(area > 0, area * _expf(ta / fuel_time) + (F(1.0) - area),
                   F(1.0))
    return out, area


def fuel_left_cpu(lfn_m, tign_m, fuel_time, time_now):
    """``fuel_left``: ``(fuel_frac, fire_area)`` on the fire mesh, float32."""
    time_now = F(time_now)
    ny, nx = lfn_m.shape[0] - 2, lfn_m.shape[1] - 2
    sum1 = np.zeros((ny, nx), dtype=F)
    sum2 = np.zeros((ny, nx), dtype=F)

    def node(a, di, dj):
        return a[1 + dj:1 + dj + ny, 1 + di:1 + di + nx]

    for di, dj, tx, ty, txx, tyy in _SUBCELLS:
        corners = [node(lfn_m, di + a, dj + b) for a in (0, 1)
                   for b in (0, 1)]       # (i2, j2), (i2, j2+1), (i2+1, ...)
        times = [node(tign_m, di + a, dj + b) for a in (0, 1)
                 for b in (0, 1)]
        lij = _bilin(tx, ty, *corners)
        li1j = _bilin(txx, ty, *corners)
        lij1 = _bilin(tx, tyy, *corners)
        li1j1 = _bilin(txx, tyy, *corners)
        tij = _bilin(tx, ty, *times)
        ti1j = _bilin(txx, ty, *times)
        tij1 = _bilin(tx, tyy, *times)
        ti1j1 = _bilin(txx, tyy, *times)
        left, area = _fuel_cell_cpu(lij, lij1, li1j, li1j1, tij, tij1, ti1j,
                                    ti1j1, time_now, fuel_time)
        sum1 = sum1 + left
        sum2 = sum2 + area
    return sum1 / F(4.0), sum2 / F(4.0)


def heat_fluxes_cpu(fgip, fuel_frac_burnt, fmc_g, dt):
    """``heat_fluxes``: ``(grnhfx, grnqfx)`` on the fire mesh, W m-2."""
    dmass = fgip * fuel_frac_burnt
    bmst = fmc_g / (F(1.0) + fmc_g)
    rate = dmass / F(dt)
    sensible = rate * (F(1.0) - bmst) * CMBCNST
    latent = (bmst + (F(1.0) - bmst) * COMBUSTION_WATER) * rate * XLV
    return sensible, latent


def fuel_heat_cpu(lfn_m, tign_m, params: FireParams, fuel_frac, time_now,
                  dt) -> dict:
    """Pass 6 on the host; updates ``fuel_frac`` in place."""
    end, area = fuel_left_cpu(lfn_m, tign_m, params.fuel_time, time_now)
    burnt = fuel_frac - end
    fuel_frac[...] = end
    sensible, latent = heat_fluxes_cpu(params.fgip, burnt, params.fmc_g, dt)
    return {"fire_area": area, "fuel_frac_burnt": burnt, "fgrnhfx": sensible,
            "fgrnqfx": latent}


def sum_2d_cells_cpu(fine, mesh: FireMesh):
    """``sum_2d_cells``: each atmospheric cell's sum, in WRF's order."""
    ir, jr = mesh.sr_x, mesh.sr_y
    blocks = np.asarray(fine, dtype=F).reshape(mesh.ny_atm, jr,
                                               mesh.nx_atm, ir)
    t = np.zeros(mesh.atm_shape, dtype=F)
    for joff in range(jr):
        for ioff in range(ir):
            t = t + blocks[:, joff, :, ioff]
    return t


def _scale_factor(mesh: FireMesh):
    return F(1.0) / F(mesh.sr_x * mesh.sr_y)


def aggregate_cpu(fine: dict, fuel_frac, mesh: FireMesh,
                  opts: FireOptions) -> dict:
    """The driver's GRNHFX/GRNQFX/CANHFX/CANQFX/AVG_FUEL_FRAC, on the host."""
    s = _scale_factor(mesh)
    feedback = F(opts.fire_atm_feedback)
    zeros = np.zeros(mesh.atm_shape, dtype=F)
    return {
        "GRNHFX": feedback * sum_2d_cells_cpu(fine["fgrnhfx"], mesh) * s,
        "GRNQFX": feedback * sum_2d_cells_cpu(fine["fgrnqfx"], mesh) * s,
        "CANHFX": zeros,
        "CANQFX": zeros.copy(),
        "AVG_FUEL_FRAC": sum_2d_cells_cpu(fuel_frac, mesh) * s,
    }


def aggregate(field, mesh: FireMesh, *, feedback: float = 1.0):
    """A fire-mesh field's atmospheric-cell mean, WRF's way, on its device."""
    from woof.fire.levelset import array_module

    xp = array_module(field)
    field = xp.ascontiguousarray(xp.asarray(field, dtype=F))
    if field.shape != mesh.shape:
        raise ValueError(f"field has shape {field.shape}, expected the fire "
                         f"mesh's {mesh.shape}")
    if xp is np:
        return F(feedback) * sum_2d_cells_cpu(field, mesh) * _scale_factor(
            mesh)
    return _sum_cells_gpu(field, mesh, F(feedback), True)


# ---------------------------------------------------------------------------
# the device path
# ---------------------------------------------------------------------------

def fuel_heat_gpu(lfn_m, tign_m, params: FireParams, fuel_frac, time_now, dt,
                  mesh: FireMesh) -> dict:
    """Pass 6 on the device; updates ``fuel_frac`` in place."""
    import cupy as cp

    out = {name: cp.empty(mesh.shape, dtype=F)
           for name in ("fire_area", "fuel_frac_burnt", "fgrnhfx", "fgrnqfx")}
    grid, block = _grid2(mesh.nx, mesh.ny)
    _kernel("fl_fuel_heat")(grid, block, (
        lfn_m, tign_m, params.fuel_time, fuel_frac, out["fire_area"],
        out["fuel_frac_burnt"], params.fgip, params.fmc_g, out["fgrnhfx"],
        out["fgrnqfx"], F(time_now), F(dt), CMBCNST, XLV,
        np.int32(mesh.nx), np.int32(mesh.ny)))
    return out


def _sum_cells_gpu(fine, mesh: FireMesh, feedback, use_feedback: bool):
    import cupy as cp

    coarse = cp.empty(mesh.atm_shape, dtype=F)
    grid, block = _grid2(mesh.nx_atm, mesh.ny_atm)
    _kernel("fl_sum_cells")(grid, block, (
        fine, coarse, np.int32(mesh.sr_x), np.int32(mesh.sr_y), feedback,
        _scale_factor(mesh), np.int32(1 if use_feedback else 0),
        np.int32(mesh.nx_atm), np.int32(mesh.ny_atm)))
    return coarse


def aggregate_gpu(fine: dict, fuel_frac, mesh: FireMesh,
                  opts: FireOptions) -> dict:
    """The driver's GRNHFX/GRNQFX/CANHFX/CANQFX/AVG_FUEL_FRAC, on the device."""
    import cupy as cp

    feedback = F(opts.fire_atm_feedback)
    return {
        "GRNHFX": _sum_cells_gpu(fine["fgrnhfx"], mesh, feedback, True),
        "GRNQFX": _sum_cells_gpu(fine["fgrnqfx"], mesh, feedback, True),
        "CANHFX": cp.zeros(mesh.atm_shape, dtype=F),
        "CANQFX": cp.zeros(mesh.atm_shape, dtype=F),
        "AVG_FUEL_FRAC": _sum_cells_gpu(fuel_frac, mesh, feedback, False),
    }
