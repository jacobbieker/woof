"""MPAS-Seaice v8.4.2: the stand-alone ocean mixed layer on the GPU.

:func:`ocean_mixed_layer` is ``seaice_column_ocean_mixed_layer``
(``mpas_seaice_column.F``), which ``seaice_column_coupling_prep`` runs first
under ``config_use_ocean_mixed_layer`` (the Registry default, ``.true.``; a
run coupled to an ocean model turns it off, as the ocean then supplies the sea
surface). Per owned cell it:

- runs ``colpkg_atm_boundary`` for the open ocean (``atmo_boundary_layer``,
  sfctype ``'ocn'``): the ocean's neutral drag from the wind, the stability
  iteration, the air stress on the ocean (under ``config_calc_surface_stresses``),
  the 2 m reference temperature and humidity, and the turbulent transfer
  coefficients;
- sets the four ocean albedos to ``seaiceOceanAlbedo`` (0.06);
- runs ``colpkg_ocn_mixed_layer``: the open ocean's longwave, sensible and
  latent heat and evaporation, the mixed layer's heat balance with the fluxes
  through the ice, the deep heat flux convergence (zeroed in place when the
  mixed layer is at or below freezing), the new sea surface temperature and the
  freezing/melting potential (clamped to +-1000 W m-2);
- under ``config_use_test_ice_shelf``, zeroes the freezing/melting potential
  under land ice.

The kernel (``seaice_column_ocean_mixed_layer_v842`` in
``column_therm_v842.cu``) is held word for word to v8.4.2 compiled with
gfortran -O0 -ffp-contract=off at RKIND 8
(``tests/test_mpas_seaice_ocean_mixed_layer_parity.py``). The stability
functions call glibc's binary64 atan (``psimhu``), so it runs only with the
LGPL opt-in ``WOOF_GLIBC239_LIBM64=1`` and is refused by name without it.

v8.4.2's own behaviour, kept: ``airOceanDragCoefficientRatio`` is the grid
field v8.4.2 passes as ``Cdn_atm_ratio_n``, so under
``config_calc_surface_stresses`` the open ocean's ratio replaces the ice's
merged one. Nothing in v8.4.2 reads that field afterwards, so no correction is
offered. ``oceanHeatFluxConvergence`` is zeroed in place where the mixed layer
is at or below freezing and it would cool it; without a forcing stream that
re-reads it, it stays zero there for the rest of the run (CICE re-reads its
``qdp`` each step). This is v8.4.2's behaviour, and it is kept.

Refused by name: ``config_atmos_boundary_method = 'constant'``
(``atmo_boundary_const``, not ported) and RKIND 4 (the column package is
double precision).
"""
from __future__ import annotations

import numpy as np

from woof.hex.errors import ConfigurationRefusal

from . import column_therm_v842 as ct

SOURCE_FILES = (
    "src/core_seaice/shared/mpas_seaice_column.F:4320-4543",
    "src/core_seaice/column/ice_colpkg.F90:3414-3611",
    "src/core_seaice/column/ice_atmo.F90:46-366",
    "src/core_seaice/column/constants/cice/ice_constants_colpkg.F90",
)
_BLOCK = 128
#: the Registry defaults (v8.4.2) of the configs the routine reads
DEFAULT_CONFIGS = {"config_atmos_boundary_method": "ccsm3", "config_calc_surface_stresses": True,
                   "config_use_high_frequency_coupling": False, "config_boundary_layer_iteration_number": 5,
                   "config_use_test_ice_shelf": False}


def _config(configs: dict, name: str):
    return configs.get(name, DEFAULT_CONFIGS[name])


def _refuse(knob: str, value, reason: str, declaration: str):
    raise ConfigurationRefusal(knob=knob, value=value, reason=reason, declaration=declaration)


def check_config(configs: dict) -> None:
    """Refuse, by name, what the mixed layer port does not carry."""
    from woof.hex.mpas_ice import rkind

    method = _config(configs, "config_atmos_boundary_method")
    if method != "ccsm3":
        _refuse("config_atmos_boundary_method", method,
                "the open ocean's atmo_boundary_const is not ported", "config_atmos_boundary_method = 'ccsm3'")
    if not rkind.trig64():
        _refuse("config_use_ocean_mixed_layer", True,
                rkind.trig_refusal("the open ocean's atmo_boundary_layer (psimhu) calls glibc's double atan"),
                "WOOF_GLIBC239_LIBM64=1, or config_use_ocean_mixed_layer = .false.")


def ocean_mixed_layer(pools: dict, configs: dict, n_cells_solve: int) -> None:
    """``seaice_column_ocean_mixed_layer``, in place on ``pools`` (the Registry
    structs ocean_coupling, atmos_coupling, atmos_forcing, tracers_aggregate,
    drag, ocean_fluxes and ocean_atmosphere) over ``n_cells_solve`` cells."""
    import cupy as cp

    check_config(configs)
    oc, ac, af = pools["ocean_coupling"], pools["atmos_coupling"], pools["atmos_forcing"]
    dg, of, oa = pools["drag"], pools["ocean_fluxes"], pools["ocean_atmosphere"]
    sst = oc["seaSurfaceTemperature"]
    if sst.dtype != np.float64:
        _refuse("config_use_ocean_mixed_layer", True,
                "the column package is RKIND 8 only (mpas_seaice_column.F does not compile at RKIND 4)",
                "float64 pools, or config_use_ocean_mixed_layer = .false.")
    n_cells = sst.shape[0]
    if not 0 <= n_cells_solve <= n_cells:
        raise ValueError("nCellsSolve must lie within nCells")
    if not n_cells_solve:
        return
    test_ice_shelf = bool(_config(configs, "config_use_test_ice_shelf"))
    land_ice = oc["landIceMask"] if test_ice_shelf else cp.zeros(n_cells, np.int32)
    folded = ct.folded_constants()
    kernel = ct._function("seaice_column_ocean_mixed_layer_v842", int(configs.get("config_nIceLayers", 7)))
    kernel(((n_cells_solve + _BLOCK - 1) // _BLOCK,), (_BLOCK,), (
        sst, oc["seaFreezingTemperature"], oc["freezingMeltingPotential"], oc["oceanMixedLayerDepth"],
        oc["oceanHeatFluxConvergence"], ac["airPotentialTemperature"], ac["uAirVelocity"], ac["vAirVelocity"],
        af["windSpeed"], ac["airLevelHeight"], ac["airSpecificHumidity"], ac["airDensity"], ac["longwaveDown"],
        ac["shortwaveVisibleDirectDown"], ac["shortwaveVisibleDiffuseDown"], ac["shortwaveIRDirectDown"],
        ac["shortwaveIRDiffuseDown"], pools["tracers_aggregate"]["iceAreaCell"],
        dg["airOceanDragCoefficientRatio"], of["oceanHeatFlux"], of["oceanShortwaveFlux"],
        oa["airStressOceanU"], oa["airStressOceanV"], oa["atmosReferenceTemperature2mOcean"],
        oa["atmosReferenceHumidity2mOcean"], oa["albedoVisibleDirectOcean"], oa["albedoIRDirectOcean"],
        oa["albedoVisibleDiffuseOcean"], oa["albedoIRDiffuseOcean"], oa["longwaveUpOcean"],
        oa["sensibleHeatFluxOcean"], oa["latentHeatFluxOcean"], oa["evaporativeWaterFluxOcean"], land_ice,
        np.float64(configs["config_dt"]), np.float64(folded["al2"]),
        np.int32(_config(configs, "config_boundary_layer_iteration_number")),
        np.int32(bool(_config(configs, "config_calc_surface_stresses"))),
        np.int32(bool(_config(configs, "config_use_high_frequency_coupling"))), np.int32(test_ice_shelf),
        np.int32(n_cells_solve)))
