"""MALI v8.4.2: calving and the calving-front restoration on the GPU.

:func:`calve_ice` is ``li_calve_ice`` (``mpas_li_calving.F``) for every
``config_calving`` option of v8.4.2: 'none', 'thickness_threshold',
'floating', 'topographic_threshold' and 'eigencalving' (its parameter a
'scalar' or 'data'), with ``config_calving_timescale`` and
``config_data_calving``. :func:`restore_calving_front` is
``li_restore_calving_front``. The kernels (``calving_v842.cu``) transcribe
the calving routines; the masks are
:func:`woof.hex.mpas_ice.geometry_v842.calculate_mask`'s, and a restored
column is initialised by :mod:`woof.hex.mpas_ice.thermal_init_v842`'s
linear profile (``li_init_linear_temperature_in_column``).

``distribute_calving_flux`` runs in one thread in the Fortran's cell order:
each front cell's removal changes the volumes the next one sees. As written
in v8.4.2, its inward-neighbour test passes the neighbour's index, not its
mask, to ``li_mask_is_dynamic_margin``, and an uncalved remainder is recorded
as zero; the port keeps both.

Every word is held to the forward-Euler step compiled verbatim with gfortran
``-O0 -ffp-contract=off`` at RKIND 4 and RKIND 8
(``tests/test_mali_calving_parity.py``). Arrays follow
:mod:`woof.hex.mpas_ice.time_integration_fe_v842`.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from . import geometry_v842, rkind, thermal_init_v842

SOURCE_FILES = ("src/core_landice/mode_forward/mpas_li_calving.F:78-1395",)
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                   "--prec-div=true", "--prec-sqrt=true")
MODULE_OPTIONS = COMPILE_OPTIONS
CALVING = ("none", "thickness_threshold", "floating", "topographic_threshold", "eigencalving")
EIGENCALVING_PARAMETER_SOURCES = ("scalar", "data")
#: the Registry's defaults for the calving options a config may leave out
DEFAULTS = {"config_calving": "none", "config_calving_topography": -500.0, "config_calving_thickness": 100.0,
            "config_calving_eigencalving_parameter_source": "scalar",
            "config_calving_eigencalving_parameter_scalar_value": 3.14e16, "config_data_calving": False,
            "config_calving_timescale": 0.0, "config_restore_calving_front": False}
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    rkind.require(precision)
    body = files("woof.hex.mpas_ice").joinpath("calving_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"li_calving_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _function(precision: str, name: str):
    import cupy as cp
    return _module_on(cp.cuda.Device().id, precision,
                      _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)).get_function(name)


def _grid(n: int) -> tuple[int]:
    return (max(1, (n + _BLOCK - 1) // _BLOCK),)


def _mask_configs(config: dict) -> dict:
    return {name: config[name] for name in ("config_ice_density", "config_ocean_density", "config_sea_level",
                                            "config_dynamic_thickness", "config_velocity_solver")}


def _update_geometry(mesh: dict, geometry: dict, config: dict) -> None:
    """li_update_geometry alone: the shared kernel, its layer thickness sent to scratch."""
    import cupy as cp
    scratch = dict(geometry, layerThickness=cp.empty_like(geometry["layerThickness"]))
    geometry_v842.update_geometry_and_layer_thickness(
        mesh=mesh, geometry=scratch, config_ice_density=config["config_ice_density"],
        config_ocean_density=config["config_ocean_density"], config_sea_level=config["config_sea_level"])


def _with_defaults(config: dict) -> dict:
    return {**DEFAULTS, **config}


def check_config(config: dict) -> None:
    """Refuse what li_calve_ice reports as an error, before the step starts."""
    config = _with_defaults(config)
    option = config["config_calving"]
    if option not in CALVING:
        raise ConfigurationRefusal(knob="config_calving", value=option,
                                   reason="not an option li_calve_ice carries (the model logs it as an error)",
                                   declaration="config_calving = " + " | ".join(map(repr, CALVING)))
    if option == "thickness_threshold" and config["config_calving_thickness"] < config["config_dynamic_thickness"]:
        raise ValueError("config_calving_thickness must be at least config_dynamic_thickness")
    source = config["config_calving_eigencalving_parameter_source"]
    if option == "eigencalving" and source not in EIGENCALVING_PARAMETER_SOURCES:
        raise ConfigurationRefusal(knob="config_calving_eigencalving_parameter_source", value=source,
                                   reason="not a source eigencalving carries (the model logs it as an error)",
                                   declaration="config_calving_eigencalving_parameter_source = 'scalar' | 'data'")


def calve_ice(*, mesh: dict, geometry: dict, velocity: dict, config: dict) -> None:
    """``li_calve_ice`` for one block, in place on ``geometry``: ``thickness``,
    ``calvingThickness``, the masks and surfaces, and with 'eigencalving'
    ``calvingVelocity``, ``requiredCalvingVolumeRate``, ``uncalvedVolume`` and
    (with a 'scalar' source) ``eigencalvingParameter``. ``velocity`` holds
    ``eMax`` and ``eMin`` for 'eigencalving'."""
    import cupy as cp
    config = _with_defaults(config)
    check_config(config)
    option = config["config_calving"]
    if option == "none":
        return
    m, g, c = mesh, geometry, config
    precision = rkind.precision_of(g["thickness"])
    real = rkind.PRECISIONS[precision]
    n_cells = g["thickness"].shape[0]
    max_edges = m["cellsOnCell"].shape[1]
    nc, me = np.int32(n_cells), np.int32(max_edges)
    cells, block = _grid(n_cells), (_BLOCK,)
    deltat = real(m["deltat"].get()[0])
    timescale = real(c["config_calving_timescale"])
    if timescale > real(0.0):
        fraction = deltat / timescale
        fraction = real(1.0) if real(1.0) < fraction else fraction   # min(deltat/timescale, 1.0)
    else:
        fraction = real(1.0)
    original = g["thickness"].copy() if c["config_data_calving"] else None
    masks = _mask_configs(c)
    sea = real(c["config_sea_level"])
    if option == "thickness_threshold":
        active = cp.empty(n_cells, cp.int32)
        _function(precision, "li_calving_active_v842")(cells, block, (
            active, g["thickness"], g["bedTopography"], sea, real(c["config_dynamic_thickness"]),
            real(c["config_calving_thickness"]), real(c["config_ocean_density"]), real(c["config_ice_density"]), nc))
        _function(precision, "li_calving_threshold_v842")(cells, block, (
            g["thickness"], g["calvingThickness"], active, g["bedTopography"], m["nEdgesOnCell"], m["cellsOnCell"],
            sea, fraction, nc, me))
    elif option in ("floating", "topographic_threshold"):
        limit = real(c["config_calving_topography"]) + sea
        _function(precision, "li_calving_floating_v842")(cells, block, (
            g["thickness"], g["calvingThickness"], g["cellMask"], g["bedTopography"], limit, fraction,
            np.int32(option == "topographic_threshold"), nc))
    else:
        if c["config_calving_eigencalving_parameter_source"] == "scalar":
            g["eigencalvingParameter"][...] = real(c["config_calving_eigencalving_parameter_scalar_value"])
        front = cp.empty(n_cells, cp.int32)
        front_mask = _function(precision, "li_calving_front_mask_v842")
        front_mask(cells, block, (front, g["cellMask"], g["bedTopography"], m["nEdgesOnCell"], m["cellsOnCell"],
                                  sea, nc, me))
        _function(precision, "li_eigencalving_rate_v842")(cells, block, (
            g["calvingVelocity"], g["requiredCalvingVolumeRate"], g["eigencalvingParameter"], velocity["eMax"],
            velocity["eMin"], g["cellMask"], front, g["thickness"], g["bedTopography"], m["dvEdge"],
            m["nEdgesOnCell"], m["cellsOnCell"], m["edgesOnCell"], sea, nc, me))
        volume = cp.empty(n_cells, real)   # the model's scratch, set before it is read
        _function(precision, "li_distribute_calving_flux_v842")((1,), (1,), (
            g["calvingThickness"], g["uncalvedVolume"], volume, g["requiredCalvingVolumeRate"], front,
            g["cellMask"], g["thickness"], m["areaCell"], m["nEdgesOnCell"], m["cellsOnCell"], deltat, nc, me))
        _function(precision, "li_subtract_calving_v842")(cells, block, (g["thickness"], g["calvingThickness"], nc))
        geometry_v842.calculate_mask(mesh=m, geometry=g, **masks)
        front_mask(cells, block, (front, g["cellMask"], g["bedTopography"], m["nEdgesOnCell"], m["cellsOnCell"],
                                  sea, nc, me))
        _function(precision, "li_calve_thin_front_v842")(cells, block, (
            g["thickness"], g["calvingThickness"], front, real(c["config_calving_thickness"]), nc))
        geometry_v842.calculate_mask(mesh=m, geometry=g, **masks)
        _function(precision, "li_calve_isolated_floating_v842")(cells, block, (
            g["thickness"], g["calvingThickness"], g["cellMask"], m["nEdgesOnCell"], m["cellsOnCell"], nc, me))
    if original is not None:
        g["thickness"][...] = original
    geometry_v842.calculate_mask(mesh=m, geometry=g, **masks)
    _update_geometry(m, g, c)


def restore_calving_front(*, mesh: dict, geometry: dict, thermal: dict, config: dict) -> None:
    """``li_restore_calving_front`` for one block, in place: marine cells of
    the initial ice extent thinner than a tenth of
    ``config_dynamic_thickness`` are restored to it (``restoreThickness``)
    with a linear temperature column; marine ice outside the initial extent
    calves (``calvingThickness``)."""
    import cupy as cp
    config = _with_defaults(config)
    m, g, t, c = mesh, geometry, thermal, config
    precision = rkind.precision_of(g["thickness"])
    real = rkind.PRECISIONS[precision]
    n_cells = g["thickness"].shape[0]
    ncs = int(m["nCellsSolve"])
    masks = _mask_configs(c)
    geometry_v842.calculate_mask(mesh=m, geometry=g, **masks)
    restored = cp.empty(n_cells, cp.int32)
    _function(precision, "li_restore_calving_front_v842")(_grid(n_cells), (_BLOCK,), (
        g["thickness"], g["calvingThickness"], g["restoreThickness"], restored, g["cellMask"], g["bedTopography"],
        real(c["config_sea_level"]), real(0.1) * real(c["config_dynamic_thickness"]), np.int32(n_cells),
        np.int32(ncs)))
    cells = np.flatnonzero(cp.asnumpy(restored))
    if cells.size:
        # li_init_linear_temperature_in_column on the restored columns: the
        # thermal init's linear profile over just those cells
        names = ("temperature", "waterfrac", "surfaceAirTemperature", "surfaceTemperature", "basalTemperature",
                 "basalHeatFlux")
        host = {name: cp.asnumpy(t[name]) for name in names}
        column = {name: cp.asarray(np.ascontiguousarray(host[name][cells])) for name in names}
        thermal_init_v842.thermal_init(
            thermal=column, thickness=cp.asarray(cp.asnumpy(g["thickness"])[cells]),
            upper_surface=cp.zeros(cells.size, real), layer_center_sigma=m["layerCenterSigma"],
            n_cells_solve=int(cells.size), config_ice_density=c["config_ice_density"],
            config_temperature_init="linear", config_surface_air_temperature_source="file",
            config_surface_air_temperature_value=0.0, config_surface_air_temperature_lapse_rate=0.0,
            config_basal_heat_flux_source="file", config_basal_heat_flux_value=0.0)
        for name in names:
            host[name][cells] = cp.asnumpy(column[name])
            t[name][...] = cp.asarray(host[name])
    geometry_v842.calculate_mask(mesh=m, geometry=g, **masks)
    _update_geometry(m, g, c)
