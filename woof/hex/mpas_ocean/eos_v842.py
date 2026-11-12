"""MPAS-Ocean v8.4.2: the equation of state and the freezing temperature on the GPU.

:func:`equation_of_state_density` is ``ocn_equation_of_state_density``
(``mpas_ocn_equation_of_state.F``) for one block: 'linear'
(``ocn_equation_of_state_linear_density``) or 'jm', Jackett and McDougall's
UNESCO fit (``ocn_equation_of_state_jm_density``), at a displacement type
'absolute', 'relative' or 'surfaceDisplaced', with the in-situ thermal
expansion and saline contraction coefficients when asked for.
:func:`freezing_temperature` and :func:`freezing_temperature_salinity_deriv`
are ``ocn_freezing_temperature`` and its salinity derivative, elementwise,
for the open ocean or a land-ice cavity.

MPAS-Ocean's own equation of state is ported here rather than the NEMO lane's:
v8.4.2 offers only 'linear' and 'jm', and NEMO's is TEOS-10's polynomial, a
different function.

The kernels (``eos_v842.cu``) are held word for word to the v8.4.2 routines,
compiled verbatim with gfortran ``-O0 -ffp-contract=off`` at RKIND 4 and 8
(``tools/mpas_ocean_oracle/run_eos.F90``, ``tests/test_mpas_ocean_eos_parity.py``).
JM's reference pressure calls ``exp``: the shared glibc transcriptions,
``expf`` at RKIND 4 (``glibc_flt32.cuh``) and ``exp`` at RKIND 8
(``glibc_flt64.cuh``, Arm optimized-routines under MIT).

Arrays are device float32 (RKIND 4) or float64 (RKIND 8) in the Fortran
layout: ``density(nVertLevels, nCells)`` is C-order ``(nCells, nVertLevels)``,
``activeTracers(nTracers, nVertLevels, nCells)`` is ``(nCells, nVertLevels,
nTracers)`` and ``tracersSurfaceValue(nTracers, nCells)`` is ``(nCells,
nTracers)``; ``maxLevelCell`` is int32. The outputs are written on each cell's
levels ``1..maxLevelCell`` only, as in the Fortran.

Refused by name: a ``config_eos_type`` other than 'linear' or 'jm' (v8.4.2's
``ocn_equation_of_state_init`` reports it as an error). An 'absolute'
displacement outside ``1..nVertLevels`` raises, as the Fortran aborts.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from .tend_v842 import COMPILE_OPTIONS, PRECISIONS, _check, _grid, _precision_of

SOURCE_FILES = (
    "src/core_ocean/shared/mpas_ocn_equation_of_state.F",
    "src/core_ocean/shared/mpas_ocn_equation_of_state_linear.F",
    "src/core_ocean/shared/mpas_ocn_equation_of_state_jm.F",
)
MODULE_OPTIONS = COMPILE_OPTIONS
#: The shared libm transcriptions each precision's unit prepends
MATH_HEADERS = {"r4": ("glibc_flt32.cuh",), "r8": ("glibc_flt64.cuh",)}
EOS_TYPES = ("linear", "jm")
DISPLACEMENT_TYPES = {"absolute": 0, "relative": 1, "surfaceDisplaced": 2}
#: Registry.xml defaults (v8.4.2) of the configs this unit reads
DEFAULT_CONFIGS = {
    "config_eos_type": "linear", "config_eos_linear_densityref": 1000.0, "config_eos_linear_alpha": 0.2,
    "config_eos_linear_beta": 0.8, "config_eos_linear_Tref": 5.0, "config_eos_linear_Sref": 35.0,
    "config_open_ocean_freezing_temperature_coeff_0": -1.8, "config_open_ocean_freezing_temperature_coeff_S": 0.0,
    "config_open_ocean_freezing_temperature_coeff_p": 0.0, "config_open_ocean_freezing_temperature_coeff_pS": 0.0,
    "config_open_ocean_freezing_temperature_reference_pressure": 0.0,
    "config_land_ice_cavity_freezing_temperature_coeff_0": 6.22e-2,
    "config_land_ice_cavity_freezing_temperature_coeff_S": -5.63e-2,
    "config_land_ice_cavity_freezing_temperature_coeff_p": -7.43e-8,
    "config_land_ice_cavity_freezing_temperature_coeff_pS": -1.74e-10,
    "config_land_ice_cavity_freezing_temperature_reference_pressure": 0.0,
}
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    import woof.core.kernels as kernels
    root = Path(kernels.__file__).parent
    math = "\n".join((root / name).read_text(encoding="utf-8") for name in MATH_HEADERS[precision])
    body = files("woof.hex.mpas_ocean").joinpath("eos_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + math + "\n" + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"ocn_eos_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(precision: str, name: str):
    import cupy as cp
    options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
    return _module_on(cp.cuda.Device().id, precision, options).get_function(name)


def _configs(configs: dict | None) -> dict:
    merged = dict(DEFAULT_CONFIGS)
    merged.update({k: v for k, v in (configs or {}).items() if k in DEFAULT_CONFIGS})
    return merged


def eos_type(configs: dict | None) -> str:
    """``config_eos_type``, or a refusal by name of what v8.4.2 rejects."""
    value = _configs(configs)["config_eos_type"]
    if value not in EOS_TYPES:
        raise ConfigurationRefusal(
            knob="config_eos_type", value=value,
            reason="ocn_equation_of_state_init reports any other value as an error",
            declaration="config_eos_type = 'linear' | 'jm'")
    return value


def equation_of_state_density(*, mesh: dict, active_tracers, tracers_surface_value, index_temperature: int,
                              index_salinity: int, density, k_displaced: int, displacement_type: str,
                              configs: dict | None = None, thermal_expansion_coeff=None,
                              saline_contraction_coeff=None, n_cells: int | None = None) -> None:
    """``ocn_equation_of_state_density`` on cells ``1..n_cells`` (all of them by
    default), in place on ``density`` and, when given, the two coefficients.

    ``mesh`` holds ``maxLevelCell`` and ``refBottomDepth``.
    """
    import cupy as cp
    kind = eos_type(configs)
    c = _configs(configs)
    if displacement_type not in DISPLACEMENT_TYPES:
        raise ValueError(f"displacement_type must be one of {sorted(DISPLACEMENT_TYPES)}")
    precision = _precision_of(density)
    dtype = PRECISIONS[precision]
    nc_all, nl = density.shape
    nc = nc_all if n_cells is None else int(n_cells)
    if not 0 <= nc <= nc_all:
        raise ValueError("n_cells must lie within the arrays' cells")
    nt = active_tracers.shape[2]
    _check("activeTracers", active_tracers, dtype, (nc_all, nl, nt))
    _check("tracersSurfaceValue", tracers_surface_value, dtype, (nc_all, nt))
    _check("maxLevelCell", mesh["maxLevelCell"], np.int32, (nc_all,))
    _check("refBottomDepth", mesh["refBottomDepth"], dtype, (nl,))
    for name, array in (("thermalExpansionCoeff", thermal_expansion_coeff),
                        ("salineContractionCoeff", saline_contraction_coeff)):
        if array is not None:
            _check(name, array, dtype, (nc_all, nl))
    for name, index in (("index_temperature", index_temperature), ("index_salinity", index_salinity)):
        if not 1 <= int(index) <= nt:
            raise ValueError(f"{name} must lie within 1..nTracers")
    type_code = DISPLACEMENT_TYPES[displacement_type]
    if displacement_type == "absolute" and not 1 <= int(k_displaced) <= nl:
        raise ValueError("k_displaced must be between 1 and nVertLevels for displacement_type = absolute "
                         "(the Fortran aborts)")
    with_alpha, with_beta = int(thermal_expansion_coeff is not None), int(saline_contraction_coeff is not None)
    alpha = thermal_expansion_coeff if with_alpha else density
    beta = saline_contraction_coeff if with_beta else density
    n = nc * nl
    if n == 0:
        return
    if kind == "linear":
        _function(precision, "ocn_eos_linear_density_v842")(_grid(n), (_BLOCK,), (
            density, alpha, beta, active_tracers, tracers_surface_value, mesh["maxLevelCell"], np.int32(nc),
            np.int32(nl), np.int32(nt), np.int32(index_temperature), np.int32(index_salinity), np.int32(type_code),
            np.int32(with_alpha), np.int32(with_beta), dtype(c["config_eos_linear_densityref"]),
            dtype(c["config_eos_linear_alpha"]), dtype(c["config_eos_linear_beta"]),
            dtype(c["config_eos_linear_Tref"]), dtype(c["config_eos_linear_Sref"])))
        return
    p_ref = cp.empty(nl, dtype=dtype)
    _function(precision, "ocn_eos_jm_reference_pressure_v842")(_grid(nl), (_BLOCK,), (
        p_ref, mesh["refBottomDepth"], np.int32(nl)))
    _function(precision, "ocn_eos_jm_density_v842")(_grid(n), (_BLOCK,), (
        density, alpha, beta, active_tracers, tracers_surface_value, p_ref, mesh["maxLevelCell"], np.int32(nc),
        np.int32(nl), np.int32(nt), np.int32(index_temperature), np.int32(index_salinity),
        np.int32(k_displaced), np.int32(type_code), np.int32(with_alpha), np.int32(with_beta)))


def _freezing(salinity, pressure, in_land_ice_cavity: bool, configs: dict | None, out, deriv: int) -> None:
    c = _configs(configs)
    precision = _precision_of(out)
    dtype = PRECISIONS[precision]
    _check("salinity", salinity, dtype, out.shape)
    _check("pressure", pressure, dtype, out.shape)
    where = "land_ice_cavity" if in_land_ice_cavity else "open_ocean"
    coeff = [dtype(c[f"config_{where}_freezing_temperature_{name}"])
             for name in ("coeff_0", "coeff_S", "coeff_p", "coeff_pS", "reference_pressure")]
    if out.size == 0:
        return
    _function(precision, "ocn_freezing_temperature_v842")(_grid(out.size), (_BLOCK,), (
        out, salinity, pressure, np.int32(out.size), np.int32(deriv), *coeff))


def freezing_temperature(*, salinity, pressure, in_land_ice_cavity: bool, configs: dict | None = None, out) -> None:
    """``ocn_freezing_temperature`` elementwise into ``out``."""
    _freezing(salinity, pressure, in_land_ice_cavity, configs, out, 0)


def freezing_temperature_salinity_deriv(*, salinity, pressure, in_land_ice_cavity: bool,
                                        configs: dict | None = None, out) -> None:
    """``ocn_freezing_temperature_salinity_deriv`` elementwise into ``out``."""
    _freezing(salinity, pressure, in_land_ice_cavity, configs, out, 1)
