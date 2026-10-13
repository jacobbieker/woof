"""The gates the WDM6 column workspace owns (2026-10-12).

``kernels/wdm6.cu`` used to keep ~37 column arrays per thread in the local
frame, sized by a compile-time ``WDM6_KMAX`` tier of 64 or 80: 9,264 B or
11,568 B per thread, which the driver reserves times the card's WHOLE
resident-thread capacity at first launch, and a hard 80-level ceiling.  The
arrays now live in a global workspace ``woof/core/wdm6.py`` sizes to the
columns in flight and to the runtime ``nz``, so the scheme runs to 256
levels.  The gates are ``tests/test_wsm6_workspace.py``'s, for the same
breakages; the differences are WDM6's: the cloud-ice fallout REUSES the
snow/graupel fallout's slots (it runs after them), the kernel ORs a count
status word, and the CPU mirror keeps its pre-workspace entry point.

1. ``test_the_workspace_slot_map_fits_the_declared_region`` -- slot ids are
   literal (NVRTC has no ``__COUNTER__``).  A duplicate id between two
   arrays that are live at once makes them alias -- a wrong forecast, not a
   crash; an id past ``WDM6WS_SLOTS`` runs into the next block's region.
2. ``test_the_python_side_mirrors_the_kernels_workspace_geometry`` -- the
   launcher allocates what the kernel indexes; a disagreement on slot
   count, lane width or extent writes past the allocation.
3. ``test_no_column_array_is_left_on_the_stack`` -- one ``float x[...]``
   brings the per-thread reservation back.
4. ``test_the_wdm6_frame_stays_under_the_default_stack`` (GPU) -- the same
   property read off the driver, on whatever card runs it.
5. ``test_the_workspace_is_free_of_residue`` (GPU) -- the workspace is
   ``cp.empty`` and reused across tiles; sound only if no slot is read
   before it is written.
6. ``test_the_tiled_launch_matches_the_single_launch`` (GPU) -- tiles are
   offset by ``col0``; a boundary that dropped or doubled a column shows up
   here.
7. ``test_the_workspace_kernel_is_bit_identical_to_the_pre_cut_kernel``
   (GPU) -- the cut is a PLACEMENT change: every output word at nz = 40, 64
   and 80 equals the pre-cut kernel's, compiled from git at the tier its
   launcher chose, with a perturbed-kernel control proving the comparison
   can fail.
8. ``test_a_256_level_column_runs_and_conserves_water`` (GPU) -- the point
   of the cut: a column to about 1 hPa runs, stays finite and non-negative,
   and conserves water substance to the precipitation it reports.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from _microphysics_workspace_columns import (
    ROOT, SURFACE, column_water, differing_words, pre_cut_kernel,
    pre_cut_source, soundings)
from conftest import requires_gpu
from woof.core.wdm6_constants import (
    WDM6WS_SLOTS, WDM6_BLOCK, WDM6_MAX_LEVELS, wdm6_workspace_floats)

_CU = ROOT / "woof" / "core" / "kernels" / "wdm6.cu"
DEFAULT_STACK_BYTES = 1024

#: The functions that own slots -- the map wdm6.cu's header declares.
_FUNCTIONS = ("wdm6_column_impl", "wdm6_sediment_snow_graupel",
              "wdm6_plm_arrival", "wdm6_plm_remap")


def _source() -> str:
    return _CU.read_text(encoding="utf-8")


def _define(name: str) -> int:
    """The DEVICE value of a define: the last spelling in the file, because
    the CPU mirror's ``#ifdef WDM6_CPU_MIRROR`` branch comes first."""
    values = re.findall(rf"^#define {name} (\d+)$", _source(), re.M)
    assert values, f"wdm6.cu has no #define {name}"
    return int(values[-1])


def _body(name: str) -> str:
    src = _source()
    m = re.search(rf"^__device__[^\n]*\b{name}\(", src, re.M)
    assert m, name
    end = re.search(r"^(__device__|#if|#endif|extern )", src[m.end():], re.M)
    return src[m.start(): m.end() + (end.start() if end else len(src))]


def _slot_ids(name: str) -> list[int]:
    return [int(v) for v in re.findall(r"ws\.at\((\d+)\)", _body(name))]


def test_the_workspace_slot_map_fits_the_declared_region():
    owned = {name: _slot_ids(name) for name in _FUNCTIONS}
    for name, ids in owned.items():
        assert ids, name
        assert len(ids) == len(set(ids)), f"{name} aliases a slot: {ids}"
        assert all(0 <= i < WDM6WS_SLOTS for i in ids), (name, ids)
    # The four functions can all be live at once (the impl calls sediment,
    # which calls arrival and remap), so their slots are pairwise disjoint,
    # and together they are exactly the declared region.
    union = [i for ids in owned.values() for i in ids]
    assert len(union) == len(set(union)), owned
    assert set(union) == set(range(WDM6WS_SLOTS))
    # The cloud-ice fallout runs AFTER the snow/graupel fallout returns, so
    # it reuses that function's slots -- and only those.
    ice = _slot_ids("wdm6_sediment_ice")
    assert len(ice) == len(set(ice))
    assert set(ice) <= set(owned["wdm6_sediment_snow_graupel"])
    assert len(re.findall(r"ws\.at\(", _source())) == WDM6WS_SLOTS + len(ice)


def test_the_python_side_mirrors_the_kernels_workspace_geometry():
    from woof.core import wdm6_constants

    assert _define("WDM6WS_SLOTS") == WDM6WS_SLOTS
    assert _define("WDM6WS_LANES") == WDM6_BLOCK == 32
    assert _define("WDM6_KMAX") == WDM6_MAX_LEVELS == 256
    assert wdm6_constants.WDM6_VERTICAL_LEVEL_BOUNDS == (2, WDM6_MAX_LEVELS)
    # One block's region: SLOTS x (nz + 1) x LANES, whole blocks only.
    assert wdm6_workspace_floats(80, 1) == WDM6WS_SLOTS * 81 * WDM6_BLOCK
    assert wdm6_workspace_floats(80, WDM6_BLOCK + 1) == (
        2 * WDM6WS_SLOTS * 81 * WDM6_BLOCK)
    # ... and the kernel refuses an extent the launcher did not allocate.
    assert "wskp < nz + 1\n        || wskp > WDM6_KMAX + 1" in _source()


def test_no_column_array_is_left_on_the_stack():
    code = "\n".join(line for line in _source().splitlines()
                     if not line.lstrip().startswith("//"))
    assert not re.search(r"\bfloat\s+\w+\s*\[", code), (
        "a per-thread float array is back in wdm6.cu")
    assert "WDM6_KMAX]" not in code


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
            d["nn"], d["nc"], d["nr"], d["rho"], d["pressure"], d["pii"],
            d["dz"], d["xland"],
            d["rainnc"], d["rainncv"], d["snownc"], d["snowncv"],
            d["graupelnc"], d["graupelncv"], d["sr"],
            d["effc"], d["effi"], d["effs"])


def _raw(d, dt, *, tile, fill=None, hail=0):
    """The shipped kernel against a caller-owned workspace and tile."""
    import cupy as cp

    from woof.core.kernels import get_kernel

    nz, ny, nx = d["theta"].shape
    ncol = ny * nx
    ws = cp.empty(wdm6_workspace_floats(nz, tile), cp.float32)
    if fill is not None:
        ws.fill(fill)
    kernel = get_kernel("wdm6", "wdm6_column")
    status = cp.zeros((1,), cp.uint32)
    for col0 in range(0, ncol, tile):
        count = min(tile, ncol - col0)
        kernel(((count + WDM6_BLOCK - 1) // WDM6_BLOCK,), (WDM6_BLOCK,),
               _volume_args(d) + (np.float32(dt), np.int32(hail),
                                  np.int32(nz), np.int32(ny), np.int32(nx),
                                  status, ws, np.int32(nz + 1),
                                  np.int32(col0)))
    cp.cuda.runtime.deviceSynchronize()
    assert int(status.get()[0]) == 0
    return {name: cp.asnumpy(value) for name, value in d.items()}


def _shipped(d, dts, hail=0):
    import cupy as cp

    from woof.core.wdm6 import launch_wdm6

    for dt in dts:
        launch_wdm6(d["theta"], d["qv"], d["qc"], d["qr"], d["qi"], d["qs"],
                    d["qg"], d["nn"], d["nc"], d["nr"], d["rho"], d["pii"],
                    d["pressure"], d["dz"], d["xland"],
                    d["rainnc"], d["rainncv"], d["snownc"], d["snowncv"],
                    d["graupelnc"], d["graupelncv"], d["sr"], dt,
                    effc=d["effc"], effi=d["effi"], effs=d["effs"],
                    hail_opt=hail)
    cp.cuda.runtime.deviceSynchronize()
    return {name: cp.asnumpy(value) for name, value in d.items()}


@requires_gpu
def test_the_wdm6_frame_stays_under_the_default_stack():
    from woof.core import preflight as pf
    from woof.core.kernels import get_kernel

    frame = int(get_kernel("wdm6", "wdm6_column")
                .attributes["local_size_bytes"])
    assert frame <= DEFAULT_STACK_BYTES
    assert frame <= pf.KERNEL_MAX_LOCAL_SIZE_BYTES["wdm6"]


@requires_gpu
def test_the_workspace_is_free_of_residue():
    fields = soundings(55, 200, seed=11, number_moments=True)
    nan = _raw(_device(fields), 300.0, tile=256, fill=np.float32("nan"))
    big = _raw(_device(fields), 300.0, tile=256, fill=np.float32(1.0e30))
    assert differing_words(nan, big) == {}
    assert all(np.isfinite(v).all() for v in nan.values())


@requires_gpu
def test_the_tiled_launch_matches_the_single_launch():
    fields = soundings(64, 300, seed=12, number_moments=True)
    single = _raw(_device(fields), 60.0, tile=320)
    tiled = _raw(_device(fields), 60.0, tile=64)        # five tiles, ragged
    assert differing_words(single, tiled) == {}
    assert differing_words(single, _shipped(_device(fields), (60.0,))) == {}


def _pre_cut(text, fields, dts, hail):
    import cupy as cp

    nz, ny, nx = fields["theta"].shape
    kernel = pre_cut_kernel("wdm6", nz, text)
    d = _device(fields)
    status = cp.zeros((1,), cp.uint32)
    for dt in dts:
        kernel((((ny * nx) + 31) // 32,), (32,),
               _volume_args(d) + (np.float32(dt), np.int32(hail),
                                  np.int32(nz), np.int32(ny), np.int32(nx),
                                  status))
    cp.cuda.runtime.deviceSynchronize()
    assert int(status.get()[0]) == 0
    return {name: cp.asnumpy(value) for name, value in d.items()}


@requires_gpu
@pytest.mark.parametrize("nz", [40, 64, 80])
def test_the_workspace_kernel_is_bit_identical_to_the_pre_cut_kernel(nz):
    text = pre_cut_source("wdm6")
    if text is None:
        pytest.skip("this checkout cannot reach the pre-cut commit")
    assert "float t[WDM6_KMAX]" in text, "not the pre-cut source"
    graded = 0
    for hail in (0, 1):
        # dt 20 s is one minor loop; 300 s is three (WRF's 120 s split).
        for dts in ((20.0, 20.0), (300.0,)):
            fields = soundings(nz, 1024, seed=10 * nz + hail, number_moments=True)
            old = _pre_cut(text, fields, dts, hail)
            new = _shipped(_device(fields), dts, hail)
            assert differing_words(old, new) == {}, (nz, hail, dts)
            # Work proof: the call moved state and rained somewhere.
            assert (old["qv"] != fields["qv"]).any()
            assert (old["rainncv"] > 0).mean() > 0.9
            graded += sum(v.size for v in new.values())
    assert graded > 0
    # Control: the comparison can fail -- a one-ULP change to one constant
    # in the pre-cut kernel moves output words.
    perturbed = text.replace("1.496e-6f", "1.4960001e-6f")
    assert perturbed != text
    fields = soundings(nz, 1024, seed=3, number_moments=True)
    assert differing_words(_pre_cut(perturbed, fields, (300.0,), 0),
                           _shipped(_device(fields), (300.0,))) != {}


@requires_gpu
def test_a_256_level_column_runs_and_conserves_water():
    nz = WDM6_MAX_LEVELS
    fields = soundings(nz, 512, seed=256, number_moments=True)
    assert float(np.median(fields["pressure"][-1])) < 150.0   # ~1 hPa top
    before = column_water(fields)
    d = _device(fields)
    out = _shipped(d, (60.0, 60.0, 60.0))
    for name, value in out.items():
        assert np.isfinite(value).all(), name
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "nn", "nc", "nr"):
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
    shallow = soundings(80, 512, seed=256, number_moments=True)
    reference = gap(_shipped(_device(shallow), (60.0, 60.0, 60.0)), shallow)
    assert np.abs(residual).max() <= 2.0 * max(np.abs(reference).max(), 1.0e-5)
    assert abs(residual.mean()) <= 2.0 * max(abs(reference.mean()), 1.0e-6)
    assert np.abs(residual).max() < 0.1
    # The radiation radii stay inside the driver's bounds.
    assert 2.49 <= out["effc"].min() and out["effc"].max() <= 50.0
    assert 4.99 <= out["effi"].min() and out["effi"].max() <= 125.0
    assert 9.99 <= out["effs"].min() and out["effs"].max() <= 999.0
    # And the launcher refuses one level more, before any kernel runs.
    from woof.core.wdm6_constants import wdm6_workspace_extent
    with pytest.raises(ValueError, match="2 <= nz <= 256"):
        wdm6_workspace_extent(nz + 1)
