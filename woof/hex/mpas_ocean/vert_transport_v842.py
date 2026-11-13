"""MPAS-Ocean v8.4.2: the vertical (ALE) transport velocity on the GPU.

:func:`vert_transport_velocity_top` is ``ocn_vert_transport_velocity_top``
(``mpas_ocn_diagnostics.F``) for one block: the layers' thickness-flux
divergence and the projected sea-surface height, the ALE thickness at the new
time (``ocn_ALE_thickness``: z-star from ``vertCoordMovementWeights``, with
the min/max thickness limits under ``config_use_min_max_thickness``), and the
transport through each layer's top integrated up from the bottom;
``config_vert_coord_movement = 'impermeable_interfaces'`` zeroes it.

The kernels (``vert_transport_v842.cu``) are held word for word to the v8.4.2
routines, compiled verbatim with gfortran ``-O0 -ffp-contract=off`` at RKIND
4 and 8 (``tools/mpas_ocean_oracle/run_vert_transport.F90``,
``tests/test_mpas_ocean_vert_transport_parity.py``).

Refused by name: the high-frequency thickness filter
(``thicknessFilterActive``, ``config_use_freq_filtered_thickness``), not
ported.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from .tend_v842 import COMPILE_OPTIONS, PRECISIONS, _precision_of

SOURCE_FILES = (
    "src/core_ocean/shared/mpas_ocn_diagnostics.F:884-1048",
    "src/core_ocean/shared/mpas_ocn_thick_ale.F",
)
MODULE_OPTIONS = COMPILE_OPTIONS
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    body = files("woof.hex.mpas_ocean").joinpath("vert_transport_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None,
                                   f"ocn_vert_transport_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _launch(precision: str, name: str, n: int, args: tuple) -> None:
    import cupy as cp
    if n > 0:
        options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
        _module_on(cp.cuda.Device().id, precision, options).get_function(name)(
            ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,), args)


def check_config(configs: dict) -> None:
    """Refuse, by name, what this port does not carry."""
    if configs.get("package:thicknessFilterActive", False) or configs.get("config_use_freq_filtered_thickness",
                                                                          False):
        raise ConfigurationRefusal(
            knob="config_use_freq_filtered_thickness", value=True,
            reason="the high-frequency thickness filter is not ported",
            declaration="config_use_freq_filtered_thickness = .false.")


def vert_transport_velocity_top(*, mesh: dict, resting_thickness, old_layer_thickness, layer_thickness_edge,
                                normal_velocity, old_ssh, dt: float, vert_ale_transport_top, configs: dict) -> None:
    """``ocn_vert_transport_velocity_top`` for one block, in place on ``vert_ale_transport_top``."""
    import cupy as cp
    check_config(configs)
    if configs.get("config_vert_coord_movement", "uniform_stretching") == "impermeable_interfaces":
        vert_ale_transport_top[...] = 0
        return
    precision = _precision_of(old_layer_thickness)
    real = PRECISIONS[precision]
    i32 = np.int32
    m = mesh
    nl = int(m["nVertLevels"])
    cells = [int(x) for x in m["nCellsArray"]]
    n_cells = int(m["nCells"])
    rows = old_layer_thickness.shape[0]
    div_hu = cp.zeros((rows, nl), dtype=real)
    projected = cp.zeros(rows, dtype=real)
    ale = cp.zeros((rows, nl), dtype=real)
    rdt = real(dt)
    _launch(precision, "ocn_vt_divergence_v842", cells[1], (
        div_hu, projected, layer_thickness_edge, normal_velocity, old_ssh, m["dvEdge"], m["areaCell"],
        m["nEdgesOnCell"], m["edgesOnCell"], m["edgeSignOnCell"], m["maxLevelEdgeBot"], i32(cells[1]), i32(nl),
        i32(m["edgesOnCell"].shape[1]), rdt))
    work = cp.zeros((max(n_cells, 1), 4 * nl), dtype=real)
    _launch(precision, "ocn_vt_ale_thickness_v842", n_cells, (
        ale, work, projected, resting_thickness, m["vertCoordMovementWeights"], m["maxLevelCell"], i32(n_cells),
        i32(nl), i32(bool(configs.get("config_use_min_max_thickness", False))),
        real(configs.get("config_max_thickness_factor", 6.0)), real(configs.get("config_min_thickness", 1.0))))
    _launch(precision, "ocn_vt_transport_top_v842", cells[1], (
        vert_ale_transport_top, div_hu, ale, old_layer_thickness, m["maxLevelCell"], i32(cells[1]), i32(nl), rdt))
