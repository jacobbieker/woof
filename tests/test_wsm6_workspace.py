"""The gates the WSM6 column workspace and its 256-level ceiling own.

``kernels/wsm6.cu`` keeps its column arrays in a global workspace the
launcher allocates for the columns in flight (upstream's memory-census cut,
``WSM6_WS_ROWS(nz) = 17 nz + 8 (nz + 1)`` rows per column, lane-interleaved
per block), so ``WSM6_KMAX`` is a refusal ceiling only and the scheme runs
to 256 levels (2026-10-12).  The breakage each gate prevents, named:

1. ``test_the_python_side_mirrors_the_kernels_workspace_geometry`` -- the
   launcher allocates at least what the kernel indexes; a disagreement on
   rows, lane width or ceiling writes past the allocation.
2. ``test_no_column_array_is_left_on_the_stack`` -- one ``float x[...]``
   brings the per-thread reservation back.
3. ``test_the_wsm6_frame_stays_under_the_default_stack`` (GPU) -- the same
   property read off the driver, on whatever card runs it.
4. ``test_the_workspace_is_free_of_residue`` (GPU) -- the workspace is
   ``cp.empty`` and reused across tiles; sound only if no slot is read
   before it is written.
5. ``test_the_tiled_launch_matches_the_single_launch`` (GPU) -- tiles are
   offset by a column index; a boundary that dropped or doubled a column
   shows up here.
6. ``test_a_256_level_column_runs_and_conserves_water`` (GPU) -- the point
   of the ceiling: a column to about 1 hPa runs, stays finite and
   non-negative, and conserves water substance to the precipitation it
   reports.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from _microphysics_workspace_columns import (
    ROOT, SURFACE, column_water, differing_words, soundings)
from conftest import requires_gpu
from woof.core.wsm6_constants import (
    WSM6WS_SLOTS, WSM6_BLOCK, WSM6_MAX_LEVELS, wsm6_workspace_extent,
    wsm6_workspace_floats, wsm6_workspace_rows)

_CU = ROOT / "woof" / "core" / "kernels" / "wsm6.cu"
DEFAULT_STACK_BYTES = 1024


def _source() -> str:
    return _CU.read_text(encoding="utf-8")


def _define(name: str) -> int:
    values = re.findall(rf"^#define {name} (\d+)$", _source(), re.M)
    assert values, f"wsm6.cu has no #define {name}"
    return int(values[-1])


def test_the_python_side_mirrors_the_kernels_workspace_geometry():
    from woof.core import wsm6_constants

    assert _define("WSM6_KMAX") == WSM6_MAX_LEVELS == 256
    assert wsm6_constants.WSM6_VERTICAL_LEVEL_BOUNDS == (2, WSM6_MAX_LEVELS)
    assert "#define WSM6_WS_ROWS(nz) (17 * (nz) + 8 * ((nz) + 1))" in _source()
    for nz in (2, 40, 80, 81, 128, 256):
        assert wsm6_workspace_rows(nz) == 17 * nz + 8 * (nz + 1)
        assert wsm6_workspace_extent(nz) == nz + 1
        # The priced (and allocated) region holds the rows the kernel
        # indexes, block by block.
        for columns in (1, WSM6_BLOCK, WSM6_BLOCK + 1, 1000):
            blocks = -(-columns // WSM6_BLOCK)
            assert wsm6_workspace_floats(nz, columns) == (
                blocks * WSM6WS_SLOTS * (nz + 1) * WSM6_BLOCK)
            assert wsm6_workspace_floats(nz, columns) >= (
                blocks * WSM6_BLOCK * wsm6_workspace_rows(nz))
    for nz in (1, 257):
        with pytest.raises(ValueError, match="2 <= nz <= 256"):
            wsm6_workspace_extent(nz)


def test_no_column_array_is_left_on_the_stack():
    code = "\n".join(line for line in _source().splitlines()
                     if not line.lstrip().startswith("//"))
    # The one fixed-size per-thread array is the coefficient table (the
    # Wsm6Coef words, WSM6_NCOEF of them), which no level count sizes.
    arrays = re.findall(r"\bfloat\s+\w+\s*\[([^\]]*)\]", code)
    assert set(arrays) <= {"WSM6_NCOEF"}, (
        "a per-thread column array is back in wsm6.cu", arrays)
    assert "WSM6_KMAX]" not in code


# ---------------------------------------------------------------------------
# device gates
# ---------------------------------------------------------------------------

def _device(fields):
    import cupy as cp

    nz, ny, nx = fields["theta"].shape
    d = {name: cp.asarray(value) for name, value in fields.items()}
    for name in SURFACE:
        d[name] = cp.zeros((ny, nx), cp.float32)
    return d


def _volume_args(d):
    return (d["theta"], d["qv"], d["qc"], d["qi"], d["qr"], d["qs"], d["qg"],
            d["rho"], d["pressure"], d["pii"], d["dz"],
            d["rainnc"], d["rainncv"], d["snownc"], d["snowncv"],
            d["graupelnc"], d["graupelncv"], d["sr"],
            d["effc"], d["effi"], d["effs"])


def _raw(d, dt, *, tile, fill=None, hail=0):
    """The shipped kernel against a caller-owned workspace and tile."""
    import cupy as cp

    from woof.core.wsm6 import _init_words_on_device, wsm6_kernel

    nz, ny, nx = d["theta"].shape
    ncol = ny * nx
    ws = cp.empty(wsm6_workspace_floats(nz, tile), cp.float32)
    if fill is not None:
        ws.fill(fill)
    kernel = wsm6_kernel("wsm6_column", nz)
    coef = _init_words_on_device(hail)
    for col0 in range(0, ncol, tile):
        count = min(tile, ncol - col0)
        kernel(((count + WSM6_BLOCK - 1) // WSM6_BLOCK,), (WSM6_BLOCK,),
               _volume_args(d) + (coef, np.float32(dt),
                                  np.int32(nz), np.int32(ny), np.int32(nx),
                                  ws, np.int32(col0)))
    cp.cuda.runtime.deviceSynchronize()
    return {name: cp.asnumpy(value) for name, value in d.items()}


def _shipped(d, dts, hail=0):
    import cupy as cp

    from woof.core.wsm6 import launch_wsm6

    for dt in dts:
        launch_wsm6(d["theta"], d["qv"], d["qc"], d["qr"], d["qi"], d["qs"],
                    d["qg"], d["rho"], d["pii"], d["pressure"], d["dz"],
                    d["rainnc"], d["rainncv"], d["snownc"], d["snowncv"],
                    d["graupelnc"], d["graupelncv"], d["sr"], dt,
                    effc=d["effc"], effi=d["effi"], effs=d["effs"],
                    hail_opt=hail)
    cp.cuda.runtime.deviceSynchronize()
    return {name: cp.asnumpy(value) for name, value in d.items()}


@requires_gpu
def test_the_wsm6_frame_stays_under_the_default_stack():
    from woof.core import preflight as pf
    from woof.core.kernels import get_kernel

    frame = int(get_kernel("wsm6", "wsm6_column")
                .attributes["local_size_bytes"])
    assert frame <= DEFAULT_STACK_BYTES
    assert frame <= pf.KERNEL_MAX_LOCAL_SIZE_BYTES["wsm6"]


@requires_gpu
def test_the_workspace_is_free_of_residue():
    fields = soundings(55, 200, seed=11)
    nan = _raw(_device(fields), 300.0, tile=256, fill=np.float32("nan"))
    big = _raw(_device(fields), 300.0, tile=256, fill=np.float32(1.0e30))
    assert differing_words(nan, big) == {}
    assert all(np.isfinite(v).all() for v in nan.values())


@requires_gpu
def test_the_tiled_launch_matches_the_single_launch():
    fields = soundings(64, 300, seed=12)
    single = _raw(_device(fields), 60.0, tile=320)
    tiled = _raw(_device(fields), 60.0, tile=64)        # five tiles, ragged
    assert differing_words(single, tiled) == {}
    assert differing_words(single, _shipped(_device(fields), (60.0,))) == {}


@requires_gpu
def test_a_256_level_column_runs_and_conserves_water():
    nz = WSM6_MAX_LEVELS
    fields = soundings(nz, 512, seed=256)
    assert float(np.median(fields["pressure"][-1])) < 150.0   # ~1 hPa top
    before = column_water(fields)
    d = _device(fields)
    out = _shipped(d, (60.0, 60.0, 60.0))
    for name, value in out.items():
        assert np.isfinite(value).all(), name
    for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
        assert out[name].min() >= 0.0, name
    assert not np.array_equal(out["theta"], fields["theta"])
    assert (out["rainnc"] > 0).mean() > 0.9
    # Water substance changes only by what the scheme says it rained out --
    # to the scheme's OWN budget gap, which is not zero: on these columns
    # WRF's PLM fallout with its nonnegativity clamps leaves up to a few
    # 1e-2 kg m-2 at 40 and at 80 levels too, where the kernel is
    # bit-identical to the pre-cut one.  So the gate is that 256 levels is
    # no worse than 80 on the same generator, plus an absolute ceiling.
    def gap(result, initial):
        return (column_water(initial) - column_water(result)
                - result["rainnc"].reshape(-1))

    residual = gap(out, fields)
    shallow = soundings(80, 512, seed=256)
    reference = gap(_shipped(_device(shallow), (60.0, 60.0, 60.0)), shallow)
    assert np.abs(residual).max() <= 2.0 * max(np.abs(reference).max(), 1.0e-5)
    assert abs(residual.mean()) <= 2.0 * max(abs(reference.mean()), 1.0e-6)
    assert np.abs(residual).max() < 0.1
    # The radiation radii stay inside the driver's bounds.
    assert 2.49 <= out["effc"].min() and out["effc"].max() <= 50.0
    assert 4.99 <= out["effi"].min() and out["effi"].max() <= 125.0
    assert 9.99 <= out["effs"].min() and out["effs"].max() <= 999.0
    # And the launcher refuses one level more, before any kernel runs.
    from woof.core.wsm6_constants import wsm6_workspace_extent
    with pytest.raises(ValueError, match="2 <= nz <= 256"):
        wsm6_workspace_extent(nz + 1)
