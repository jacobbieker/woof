"""MPAS-Ocean v8.4.2: horizontal tracer mixing on the GPU.

:func:`tracer_hmix_tend` is ``ocn_tracer_hmix_tend`` for one tracer group on
one block: the Laplacian (``ocn_tracer_hmix_del2_tend``,
``config_use_tracer_del2`` with ``config_tracer_del2`` > 0) and then the
biharmonic (``ocn_tracer_hmix_del4_tend``, ``config_use_tracer_del4`` with
``config_tracer_del4`` > 0) mixing, each added to ``tend`` in the Fortran's
order over its own halo extent (``nCellsArray(1)``; del4's first Laplacian
over ``nCellsArray(2)``).

The kernels (``tracer_hmix_v842.cu``) are held word for word to the v8.4.2
routines, compiled verbatim with gfortran ``-O0 -ffp-contract=off`` at RKIND
4 and 8 (``tools/mpas_ocean_oracle/run_tracer_hmix.F90``,
``tests/test_mpas_ocean_tracer_hmix_parity.py``).

Arrays are device float32 (RKIND 4) or float64 (RKIND 8) in the Fortran
layout with MPAS's missing-neighbour element: ``tracers(nTracers,
nVertLevels, nCells+1)`` is C-order ``(nCells+1, nVertLevels, nTracers)``.

Refused by name: Redi isoneutral mixing (``config_use_standardGM``, with GM),
which is not ported.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from .tend_v842 import COMPILE_OPTIONS, PRECISIONS, _precision_of

SOURCE_FILES = (
    "src/core_ocean/shared/mpas_ocn_tracer_hmix.F",
    "src/core_ocean/shared/mpas_ocn_tracer_hmix_del2.F",
    "src/core_ocean/shared/mpas_ocn_tracer_hmix_del4.F",
)
MODULE_OPTIONS = COMPILE_OPTIONS
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    body = files("woof.hex.mpas_ocean").joinpath("tracer_hmix_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None, f"ocn_tracer_hmix_v842_{precision}.cu")
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
    if configs.get("config_use_standardGM", False):
        raise ConfigurationRefusal(
            knob="config_use_standardGM", value=True,
            reason="Redi isoneutral tracer mixing (and the GM bolus velocity it comes with) is not ported",
            declaration="config_use_standardGM = .false.")


def switches(configs: dict) -> tuple[float | None, float | None]:
    """(eddyDiff2 or None, eddyDiff4 or None) as the _init routines set them."""
    del2 = float(configs["config_tracer_del2"]) if (configs.get("config_use_tracer_del2", False)
                                                    and configs["config_tracer_del2"] > 0.0) else None
    del4 = float(configs["config_tracer_del4"]) if (configs.get("config_use_tracer_del4", False)
                                                    and configs["config_tracer_del4"] > 0.0) else None
    return del2, del4


def tracer_hmix_tend(*, tracers, layer_thickness_edge, mesh: dict, tend, configs: dict) -> None:
    """``ocn_tracer_hmix_tend`` for one tracer group, in place on ``tend``."""
    import cupy as cp
    check_config(configs)
    if configs.get("config_disable_tr_hmix", False):
        return
    precision = _precision_of(tracers)
    real = PRECISIONS[precision]
    cells = [int(x) for x in mesh["nCellsArray"]]
    n_rows, nl, nt = tracers.shape
    me = mesh["edgesOnCell"].shape[1]
    m, i32 = mesh, np.int32
    del2, del4 = switches(configs)
    edges = (m["dvEdge"], m["dcEdge"], m["areaCell"], m["nEdgesOnCell"], m["edgesOnCell"], m["edgeSignOnCell"],
             m["cellsOnEdge"], m["maxLevelEdgeTop"])
    if del2 is not None:
        _launch(precision, "ocn_tracer_hmix_del2_v842", cells[0] * nl * nt, (
            tend, tracers, layer_thickness_edge, m["meshScalingDel2"], *edges, i32(cells[0]), i32(nl), i32(nt),
            i32(me), real(del2)))
    if del4 is not None:
        delsq = cp.zeros((n_rows, nl, nt), dtype=real)
        _launch(precision, "ocn_tracer_hmix_del4_laplacian_v842", cells[1] * nl * nt, (
            delsq, tracers, layer_thickness_edge, *edges, m["edgeMask"], i32(cells[1]), i32(nl), i32(nt), i32(me)))
        _launch(precision, "ocn_tracer_hmix_del4_v842", cells[0] * nl * nt, (
            tend, delsq, m["meshScalingDel4"], *edges, m["edgeMask"], i32(cells[0]), i32(nl), i32(nt), i32(me),
            real(del4)))
