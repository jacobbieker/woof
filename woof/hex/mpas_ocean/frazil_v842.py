"""MPAS-Ocean v8.4.2: frazil ice formation on the GPU.

:func:`build_arrays` is ``ocn_frazil_forcing_build_arrays``
(``mpas_ocn_frazil_forcing.F``) for one block, which ``ocn_forward_mode_run``
calls before each step: per column (cells 1..``nCellsArray(2)``, open ocean
and under land ice as ``config_frazil_in_open_ocean`` and
``config_frazil_under_land_ice`` allow), the levels from the deepest one
within ``config_frazil_maximum_depth`` of the surface up to the top, frazil
formed where the layer is below its freezing temperature
(``ocn_freezing_temperature``, the land-ice cavity's coefficients under land
ice) and melted back above it, each limited to
``config_frazil_fractional_thickness_limit`` of the layer; the thickness,
temperature and salinity tendencies, the accumulated frazil mass and salinity
at time level 2 from those at level 1 (and the land-ice frazil mass under
land ice) and, under ``config_frazil_use_surface_pressure``, the frazil's
surface pressure.

:func:`layer_thickness_tend` and :func:`active_tracers_tend` are
``ocn_frazil_forcing_layer_thickness`` and ``ocn_frazil_forcing_tracers``
(the active tracers), the last terms ``ocn_tend_thick`` and
``ocn_tend_tracer`` add.

The kernels (``frazil_v842.cu``) are held word for word to v8.4.2, the module
compiled verbatim with gfortran ``-O0 -ffp-contract=off`` at RKIND 4 and 8
(``tools/mpas_ocean_oracle/run_frazil.F90``,
``tests/test_mpas_ocean_frazil_parity.py``; over forward steps,
``tests/test_mpas_ocean_forward_frazil_parity.py``).
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from .tend_v842 import COMPILE_OPTIONS, PRECISIONS, _precision_of, launch_options

SOURCE_FILES = (
    "src/core_ocean/shared/mpas_ocn_frazil_forcing.F",
    "src/core_ocean/shared/mpas_ocn_equation_of_state.F:ocn_freezing_temperature",
)
MODULE_OPTIONS = COMPILE_OPTIONS
_BLOCK = 128

#: Registry.xml defaults (v8.4.2) of the configs ocn_frazil_forcing reads
DEFAULT_CONFIGS = {
    "config_use_frazil_ice_formation": False, "config_frazil_in_open_ocean": True,
    "config_frazil_under_land_ice": True, "config_frazil_heat_of_fusion": 3.34e5,
    "config_frazil_ice_density": 1000.0, "config_frazil_fractional_thickness_limit": 0.1,
    "config_specific_heat_sea_water": 3.996e3, "config_frazil_maximum_depth": 100.0,
    "config_frazil_sea_ice_reference_salinity": 4.0, "config_frazil_land_ice_reference_salinity": 0.0,
    "config_frazil_maximum_freezing_temperature": 0.0, "config_frazil_use_surface_pressure": True,
    "config_density0": 1026.0,
}
#: the state's frazil fields, two time levels each (NAME and NAME@2)
TWO_LEVEL = ("accumulatedFrazilIceMass", "accumulatedFrazilIceSalinity", "accumulatedLandIceFrazilMass")
_FREEZING = ("coeff_0", "coeff_S", "coeff_p", "coeff_pS", "reference_pressure")


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    body = files("woof.hex.mpas_ocean").joinpath("frazil_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"ocn_frazil_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _launch(precision: str, name: str, n: int, args: tuple) -> None:
    import cupy as cp
    if n > 0:
        options = launch_options()
        _module_on(cp.cuda.Device().id, precision, options).get_function(name)(
            ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), args)


def is_on(configs: dict) -> bool:
    """``frazilFormationOn`` as ``ocn_frazil_forcing_init`` sets it."""
    return bool(configs.get("config_use_frazil_ice_formation", False))


def _config(configs: dict, name: str):
    return configs.get(name, DEFAULT_CONFIGS[name])


def build_arrays(*, dt: float, mesh: dict, state: dict, tracers: dict, diagnostics: dict, forcing: dict,
                 configs: dict) -> None:
    """``ocn_frazil_forcing_build_arrays`` for one block, in place.

    ``state`` holds ``layerThickness`` and ``ssh`` (time level 1) and the
    three accumulated frazil fields at both levels (``NAME``, ``NAME@2``);
    ``tracers`` the level-1 ``activeTracers`` with its indices;
    ``diagnostics`` ``zMid``, ``density`` and ``pressure``; ``forcing`` the
    three frazil tendencies, ``frazilSurfacePressure`` and ``landIceMask``
    (``None`` when the land-ice package is off). ``dt`` is the clock's step.
    """
    import cupy as cp
    if not is_on(configs):
        return
    h = state["layerThickness"]
    precision = _precision_of(h)
    real = PRECISIONS[precision]
    i32 = np.int32
    f = forcing
    for name in ("frazilLayerThicknessTendency", "frazilTemperatureTendency", "frazilSalinityTendency"):
        f[name][...] = 0
    nl = int(mesh["nVertLevels"])
    n_cells = int(mesh["nCellsArray"][1])
    group = tracers["activeTracers"]
    rho_sw = real(_config(configs, "config_density0"))
    params = cp.asarray(np.array([
        _config(configs, "config_frazil_heat_of_fusion"), _config(configs, "config_frazil_ice_density"),
        _config(configs, "config_frazil_fractional_thickness_limit"), _config(configs, "config_frazil_maximum_depth"),
        _config(configs, "config_specific_heat_sea_water"),
        _config(configs, "config_frazil_sea_ice_reference_salinity"),
        _config(configs, "config_frazil_land_ice_reference_salinity"),
        _config(configs, "config_frazil_maximum_freezing_temperature"), rho_sw, dt], dtype=real))
    from .eos_v842 import _configs as eos_configs
    c = eos_configs({k: v for k, v in configs.items() if "freezing_temperature" in k})
    freeze = cp.asarray(np.array(
        [c[f"config_open_ocean_freezing_temperature_{n}"] for n in _FREEZING]
        + [c[f"config_land_ice_cavity_freezing_temperature_{n}"] for n in _FREEZING], dtype=real))
    st, d = state, diagnostics
    mask = f.get("landIceMask")
    _launch(precision, "ocn_frazil_build_arrays_v842", n_cells, (
        f["frazilLayerThicknessTendency"], f["frazilTemperatureTendency"], f["frazilSalinityTendency"],
        st["accumulatedFrazilIceMass@2"], st["accumulatedFrazilIceSalinity@2"], st["accumulatedLandIceFrazilMass@2"],
        st["accumulatedFrazilIceMass"], st["accumulatedFrazilIceSalinity"], st["accumulatedLandIceFrazilMass"],
        h, st["ssh"], group, d["zMid"], d["density"], d["pressure"], mesh["maxLevelCell"] if mask is None else mask,
        mesh["maxLevelCell"],
        params, freeze, i32(n_cells), i32(nl), i32(group.shape[2]), i32(tracers["index_temperature"]),
        i32(tracers["index_salinity"]), i32(bool(_config(configs, "config_frazil_in_open_ocean"))),
        i32(bool(_config(configs, "config_frazil_under_land_ice"))), i32(mask is not None)))
    if _config(configs, "config_frazil_use_surface_pressure"):
        _launch(precision, "ocn_frazil_surface_pressure_v842", n_cells, (
            f["frazilSurfacePressure"], st["accumulatedFrazilIceMass@2"], i32(n_cells)))


def layer_thickness_tend(*, mesh: dict, forcing: dict, tend, configs: dict) -> None:
    """``ocn_frazil_forcing_layer_thickness``: the frazil thickness tendency added into ``tend``."""
    if not is_on(configs):
        return
    precision = _precision_of(tend)
    n_cells = int(mesh["nCellsArray"][1])
    _launch(precision, "ocn_frazil_thickness_tend_v842", n_cells, (
        tend, forcing["frazilLayerThicknessTendency"], mesh["maxLevelCell"], np.int32(n_cells),
        np.int32(mesh["nVertLevels"])))


def active_tracers_tend(*, mesh: dict, forcing: dict, tend, index_temperature: int, index_salinity: int,
                        configs: dict) -> None:
    """``ocn_frazil_forcing_tracers`` for ``activeTracers``: the frazil temperature and
    salinity tendencies added into ``tend``."""
    if not is_on(configs):
        return
    precision = _precision_of(tend)
    n_cells = int(mesh["nCellsArray"][1])
    i32 = np.int32
    _launch(precision, "ocn_frazil_tracers_tend_v842", n_cells, (
        tend, forcing["frazilTemperatureTendency"], forcing["frazilSalinityTendency"], mesh["maxLevelCell"],
        i32(n_cells), i32(mesh["nVertLevels"]), i32(tend.shape[2]), i32(index_temperature), i32(index_salinity)))
