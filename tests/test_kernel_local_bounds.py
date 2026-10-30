"""Compile-time local-array bounds in ``kf.cu`` and ``refl.cu``.

Both files declare their per-thread column arrays against a
``#ifndef``-guarded compile-time bound, and both launchers specialize that
bound to the field's own level count.  On this card the bound is not a style
question: the driver answers a kernel's FIRST launch by reserving a
local-memory backing store for the device's whole resident-thread capacity,
sized by the per-thread frame and held for the process lifetime, so
``KF_KMAX = 128`` on a 49-level case cost 5,738 MiB of device memory the
CuPy pool never saw (the law in ``woof/core/preflight.py``; the
measurements in ``docs/kernel_local_memory_bounds.md``).

``kf`` is OUT of this file.  On 2026-08-21 52 of ``kf_column``'s 54
column arrays moved into a global workspace (tests/test_kf_workspace.py);
on 2026-10-07 the kernel became a line-for-line port of WRF's KF_eta_PARA
with all 66 of its column arrays in that workspace, so ``KF_KMAX`` sizes
nothing at all -- it is the nz refusal ceiling and no allocation -- and the
"no bit moves when the bound changes" comparison this file used to make for
it has nothing left to compare.  Its frame gate is
tests/test_kf_workspace.py::test_the_kf_frame_stays_under_the_default_stack.

Two things have to hold, and both are checked against the device here:

* **No bit moves.**  The bound is an allocation size; no expression in
  either kernel reads it, and no loop runs past the runtime ``nz``.  Every
  comparison below launches the specialized kernel and the UNSPECIALIZED
  one -- the module compiled with no defines injected, i.e. exactly the
  binary that ran before 2026-07-26 -- on the same inputs in the same
  process, and compares the raw output bytes.
* **The frame really shrinks.**  ``CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`` is
  read back from the driver and compared against
  ``preflight.LEVEL_SPECIALIZED_KERNEL_FRAMES``, so a compiler or source
  change that breaks the linear model fails here instead of silently
  mispricing a rail gate.
"""
# Replaced in 2.9.1 (the WRF exact chain): test_thompson_wrf39_frames_bound_the_fork_generation_exports -> tests/test_thompson_fork_column_gpu.py::test_the_column_unit_compiles_to_the_frame_it_is_priced_at.

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu, requires_nvrtc_frames

from woof.core import preflight as pf

pytestmark = [pytest.mark.gpu, requires_gpu]


def _raw(array) -> bytes:
    import cupy as cp

    return np.ascontiguousarray(cp.asnumpy(array)).tobytes()


# ---------------------------------------------------------------------------
# The frames themselves
# ---------------------------------------------------------------------------

def test_thompson_selected_shallow_frames_bound_the_production_exports():
    """A compiler spill above admission's tier bound would cause device OOM."""
    import re
    from woof.core.kernels import get_kernel, module_source

    for module, bound in pf.THOMPSON_SHALLOW_KERNEL_FRAMES.items():
        symbols = re.findall(
            r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(',
            module_source(module))
        # The ``*_wrf39_*`` exports exist only in the THOMPSON_AA_WRF39
        # build.  No forecast launches them: thompson_version = "wrf_39_noaa"
        # runs its own column unit (preflight's thompson_fork_unit row),
        # gated in tests/test_thompson_fork_column_gpu.py.
        shallow = [name for name in symbols
                   if "_256" not in name and "_wrf39_" not in name]
        assert shallow
        measured = {name: int(get_kernel(module, name).attributes["local_size_bytes"])
                    for name in shallow}
        assert max(measured.values()) <= bound, measured


@requires_nvrtc_frames
def test_the_specialized_frames_are_what_preflight_prices():
    """Every row of the pricing model, read back off the driver."""
    from woof.core.kernels import get_kernel, get_kernel_int_defines

    # ``kf`` is absent, and that is the change of 2026-08-21: 52 of
    # ``kf_column``'s 54 column arrays moved into a global workspace, so
    # ``KF_KMAX`` sizes only the two that stayed, the frame stopped
    # following ``nz``, and the row left LEVEL_SPECIALIZED_KERNEL_FRAMES.
    # tests/test_kf_workspace.py carries the claim that replaced this one.
    cases = (
        ("refl", "refl10cm_morrison_column", "REFL_KMAX", (256, 49, 30)),
    )
    for module, symbol, define, bounds in cases:
        spec = pf.LEVEL_SPECIALIZED_KERNEL_FRAMES[module]
        assert spec.define == define
        for bound in bounds:
            kernel = (get_kernel(module, symbol)
                      if bound == spec.unspecialized_levels else
                      get_kernel_int_defines(module, symbol,
                                             ((define, bound),)))
            observed = int(kernel.attributes["local_size_bytes"])
            assert observed == spec.frame_bytes(bound), (module, bound)


@pytest.mark.parametrize("module", ["wdm6", "wsm6"])
def test_the_microphysics_workspace_kernels_hold_no_column_on_the_stack(module):
    """WDM6 and WSM6 are no longer TIERED.

    Until 2026-10-12 ``woof/core/{wdm6,wsm6}.py`` compiled ``*_KMAX`` at a
    64 or an 80 rung and the two rungs' frames (WDM6 9,264 / 11,568 B,
    WSM6 7,216 / 9,008 B) were priced through a tier ladder.  The column
    arrays now live in a launcher-owned workspace, so ONE binary serves
    every nz up to 256 and its frame is what the driver reports here: the
    recorded row, and under the 1,024 B default stack on any platform, so
    the driver reserves nothing for it.
    """
    from woof.core.kernels import get_kernel

    observed = int(get_kernel(module, f"{module}_column")
                   .attributes["local_size_bytes"])
    # WDM6 reads 0 B; WSM6 (upstream 1.1.0's WRF-exact kernel on its own
    # workspace) reads 24 B on sm_86 -- a scalar, not a column.
    assert observed <= pf.KERNEL_MAX_LOCAL_SIZE_BYTES[module] <= 1024
    assert observed <= 1024
    assert not hasattr(pf, f"{module.upper()}_TIER_FRAME")


@requires_nvrtc_frames
def test_the_sase_tier_frame_is_what_the_driver_reports_at_every_rung():
    """SASE compiles ``SASE_KMAX`` per run at 32 / 64 / 128 / 256.

    The widest of the module's exports at each rung is what a run on that
    rung reserves, so every rung the launcher can compile is read back off
    the driver and must equal ``SASE_TIER_FRAME`` exactly (49 B x tier,
    zero intercept: 1,568 / 3,136 / 6,272 / 12,544 B).  The 128 rung is
    also the plain loader's ``#ifndef`` default and so the recorded
    ``KERNEL_MAX_LOCAL_SIZE_BYTES`` row.
    """
    import re
    from pathlib import Path

    import woof.core.kernels as kernels
    from woof.core.kernels import load_module, load_module_int_defines
    from woof.core.sase_limits import (KMAX_TIERS, SOURCE_DEFAULT_KMAX,
                                       THREADS_PER_BLOCK)

    spec = pf.SASE_TIER_FRAME
    assert spec.define == "SASE_KMAX"
    assert spec.shipped_tier == SOURCE_DEFAULT_KMAX
    assert spec.lowest_tier == KMAX_TIERS[0]
    source = (Path(kernels.__file__).parent / "sase.cu").read_text()
    names = sorted(set(re.findall(
        r'extern\s+"C"\s+__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)',
        source)))

    def widest(module):
        return max(int(module.get_function(n).attributes["local_size_bytes"])
                   for n in names)

    plain = widest(load_module("sase"))
    assert plain == spec.frame_bytes(spec.shipped_tier) == (
        pf.KERNEL_MAX_LOCAL_SIZE_BYTES["sase"])
    for tier in KMAX_TIERS:
        module = load_module_int_defines(
            "sase", (("SASE_KMAX", tier), ("SASE_TPB", THREADS_PER_BLOCK)))
        assert widest(module) == spec.frame_bytes(tier) == 49 * tier, tier


@requires_nvrtc_frames
def test_the_unspecialized_source_still_compiles_to_the_recorded_ceiling():
    """The ``#ifndef`` guard must not move the default.  If it did, every
    historical measurement priced against ``KERNEL_MAX_LOCAL_SIZE_BYTES``
    would silently be describing a different binary."""
    from woof.core.kernels import get_kernel

    # ``kf`` is not here: it compiles on its own unflushed route
    # (woof/core/kf.py) with every column array in the workspace, and its
    # frame is gated against the default stack by tests/test_kf_workspace.py
    # and priced by KERNEL_MAX_LOCAL_SIZE_BYTES["kf"] as a ceiling.
    from woof.core import kf as kfmod
    from test_kf_workspace import _frame_bytes
    frame = _frame_bytes(kfmod._kf_function("kf_column"))
    assert frame <= pf.KERNEL_MAX_LOCAL_SIZE_BYTES["kf"] <= 1024
    assert int(get_kernel("refl", "refl10cm_morrison_column")
               .attributes["local_size_bytes"]
               ) == pf.KERNEL_MAX_LOCAL_SIZE_BYTES["refl"] == 18432
    # 0, not 9,264 / 7,216: WDM6's and WSM6's column arrays left the stack
    # for launcher-owned workspaces on 2026-10-12, and the KMAX bound is a
    # refusal ceiling that sizes nothing.
    assert int(get_kernel("wdm6", "wdm6_column")
               .attributes["local_size_bytes"]
               ) == pf.KERNEL_MAX_LOCAL_SIZE_BYTES["wdm6"] == 0
    assert int(get_kernel("wsm6", "wsm6_column")
               .attributes["local_size_bytes"]
               ) <= pf.KERNEL_MAX_LOCAL_SIZE_BYTES["wsm6"] <= 1024


def test_the_myj_frames_are_the_recorded_zero_rows():
    """MYJ and MYJURB hold their columns in a runtime workspace
    (2026-10-12), so the one compiled binary of each reserves nothing and
    its row says so.  Before, both were 9,232-12,304 B at MYJ_KMAX = 128,
    a bound the 256-level contract would have doubled."""
    from woof.core.kernels import get_kernel

    for module, func in (("myjpbl", "myjpbl_column"),
                         ("myjurb", "myjurb_column")):
        assert int(get_kernel(module, func).attributes["local_size_bytes"]
                   ) == pf.KERNEL_MAX_LOCAL_SIZE_BYTES[module] == 0, module


# ---------------------------------------------------------------------------
# ... and that the shrink moves no arithmetic
# ---------------------------------------------------------------------------

def test_refl_is_bitwise_identical_at_the_specialized_bound():
    """All four ``refl.cu`` kernels, every scheme option, both bounds."""
    import cupy as cp

    import test_refl as trefl
    from woof.core import refl as reflmod
    from woof.core.kernels import get_kernel

    batch = trefl._stress_columns()
    nz, ny, nx = batch["t"].shape
    assert reflmod.kernel_capacity(nz) == nz < reflmod._KMAX
    dev = {name: cp.asarray(value) for name, value in batch.items()}
    tables = reflmod._device_tables()
    column_blocks = ((ny * nx + reflmod._COLUMN_TPB - 1)
                     // reflmod._COLUMN_TPB,)
    column_block = (reflmod._COLUMN_TPB,)

    def compare(symbol, args_for):
        outputs = []
        for kernel in (reflmod._column_kernel(symbol, nz),
                       get_kernel("refl", symbol)):
            refl = cp.full((nz, ny, nx), cp.nan, dtype=cp.float32)
            grid, block, args = args_for(refl)
            kernel(grid, block, args)
            outputs.append(_raw(refl))
        assert outputs[0] == outputs[1], f"{symbol} moved"
        return outputs[0]

    for rimed in (0, 1):
        rc = reflmod.radar_init(rimed)
        compare("refl10cm_morrison_column", lambda refl, rc=rc: (
            column_blocks, column_block,
            (dev["qv"], dev["qr"], dev["nr"], dev["qs"], dev["ns"],
             dev["qg"], dev["ng"], dev["t"], dev["p"], tables,
             np.float64(rc.k_w), np.float64(rc.m_w_0.real),
             np.float64(rc.m_w_0.imag), np.float64(rc.m_i_0.real),
             np.float64(rc.m_i_0.imag), np.float64(rc.xam_g),
             refl, np.int32(nz), np.int32(ny), np.int32(nx))))

    for hail in (0, 1):
        rc = reflmod.radar_init_wsm6(hail)
        rimed = reflmod.wsm6_rimed(hail)
        compare("refl10cm_wsm6_column", lambda refl, rc=rc, rimed=rimed: (
            column_blocks, column_block,
            (dev["qv"], dev["qr"], dev["qs"], dev["qg"], dev["t"], dev["p"],
             tables, np.float64(rc.k_w), np.float64(rc.m_w_0.real),
             np.float64(rc.m_w_0.imag), np.float64(rc.m_i_0.real),
             np.float64(rc.m_i_0.imag), np.float64(rc.xam_g),
             np.float64(rimed.n0g), refl,
             np.int32(nz), np.int32(ny), np.int32(nx))))

    thompson_rc = reflmod.radar_init()
    compare("refl10cm_thompson_column", lambda refl, rc=thompson_rc: (
        column_blocks, column_block,
        (dev["qv"], dev["qr"], dev["nr"], dev["qs"], dev["qg"], dev["ng"],
         dev["t"], dev["p"], tables, np.float64(rc.k_w),
         np.float64(rc.m_w_0.real), np.float64(rc.m_w_0.imag),
         np.float64(rc.m_i_0.real), np.float64(rc.m_i_0.imag),
         refl, np.int32(nz), np.int32(ny), np.int32(nx))))

    ncell = nz * ny * nx
    cell_blocks = ((ncell + reflmod._CELL_TPB - 1) // reflmod._CELL_TPB,)
    kessler = compare("refl10cm_kessler_cell", lambda refl: (
        cell_blocks, (reflmod._CELL_TPB,),
        (dev["qv"], dev["qr"], dev["t"], dev["p"], refl, np.int32(ncell))))
    # The batch must have carried signal, not a field of floors.
    assert np.frombuffer(kessler, dtype=np.float32).max() > 0.0


# ---------------------------------------------------------------------------
# The launchers take the specialized path, and refuse past the ceiling
# ---------------------------------------------------------------------------

def test_the_production_launchers_compile_to_the_field_not_the_ceiling():
    """The regression this file exists to prevent: a ``refl`` call site that
    quietly goes back to ``get_kernel`` costs 3,708 MiB of device memory
    with no other symptom.

    ``kf`` is here for the MIRROR of that regression -- it must take
    neither loader path: it compiles on its own unflushed route
    (woof/core/kf.py) -- and the two halves are asserted separately below
    so neither can be satisfied by the other.
    """
    import cupy as cp

    import woof.core.kernels as kernels
    from woof.core import kf as kfmod
    from woof.core import refl as reflmod

    from test_kf_workspace import _INPUTS, _kf_batch

    host = _kf_batch(8)
    nz = host["temperature"].shape[0]
    requested: list[tuple] = []
    real_specialized = kernels.get_kernel_int_defines
    real_plain = kernels.get_kernel

    def spy_specialized(module, func, defines):
        requested.append(("specialized", module, func, defines))
        return real_specialized(module, func, defines)

    def spy_plain(module, func):
        requested.append(("plain", module, func))
        return real_plain(module, func)

    from woof.core import morrison as morrmod

    real_morrison_plain = morrmod.get_kernel
    kernels.get_kernel_int_defines = spy_specialized
    kernels.get_kernel = spy_plain
    morrmod.get_kernel = spy_plain
    try:
        kfmod.launch_kf(*(cp.asarray(host[name]) for name in _INPUTS),
                        dx=12000.0, dt=60.0, cudt_minutes=5.0)
        shape = (nz, 2, 4)
        zeros = cp.zeros(shape, dtype=cp.float32)
        reflmod.launch_refl10cm_morrison(
            zeros, zeros, zeros, zeros, zeros, zeros, zeros,
            cp.full(shape, 270.0, dtype=cp.float32),
            cp.full(shape, 50000.0, dtype=cp.float32),
            cp.zeros(shape, dtype=cp.float32))
    finally:
        kernels.get_kernel_int_defines = real_specialized
        kernels.get_kernel = real_plain
        morrmod.get_kernel = real_morrison_plain

    # Morrison's reflectivity is WRF's refl10cm_hm in the scheme's own
    # unit since lane/parity-mp-libm (morrison.cu mo_refl10cm, a 0-byte
    # frame), so the refl.cu column kernel and its REFL_KMAX-sized frame
    # are not reached at all.
    assert ("plain", "morrison", "mo_refl10cm") in requested
    assert not [row for row in requested if row[1] == "refl"]
    # KF asks the generic loaders for nothing: since the 2026-10-07 WRF
    # port its unit compiles on its own --fmad=false --ftz=false route, and
    # every column array is in the workspace, so there is no bound to
    # specialize.  A KF launcher back on get_kernel would run the flushed
    # build, which leaves the WRF oracle (tests/test_kf_wrf461_oracle.py).
    assert not [row for row in requested if row[1] == "kf"]


def test_the_ceiling_is_still_refused_above_the_compiled_bound():
    """Specializing downward must not turn the ceiling into a silent
    truncation: nz past the unspecialized bound is an error, not a clamp."""
    from woof.core import kf as kfmod
    from woof.core import refl as reflmod

    with pytest.raises(ValueError, match="KF requires 8 <= nz <= 256"):
        kfmod.kernel_capacity(257)
    with pytest.raises(ValueError, match="exceeds REFL_KMAX=256"):
        reflmod.kernel_capacity(257)


# ---------------------------------------------------------------------------
# acoustic.cu's WPHI_MAX_LEV: the same two claims, one tier ladder up
# ---------------------------------------------------------------------------
#
# ``acoustic`` specializes COARSELY -- a short ladder of tiers rather than the
# exact nz -- because its bound sizes one FP32 column (4 B per level) instead
# of kf's 47, so per-nz compiles would buy bytes and cost modules.  The two
# things that must hold are unchanged: the frame the driver reports is the
# frame ``woof.core.preflight`` prices, and moving the tier moves no bits.

_WPHI_SYMBOLS = ("advance_w_phi", "advance_w_phi_msf")


@requires_nvrtc_frames
def test_the_acoustic_tier_frames_are_what_preflight_prices():
    """De-provisionalizes ``preflight.ACOUSTIC_TIER_FRAME``.

    The deeper tiers' frames are extrapolated in the pricing model from the
    driver-measured shipped tier; only this test can say whether NVRTC agrees
    at 193 and 257, or spills more than the array.
    """
    from woof.core import acoustic as ac
    from woof.core.kernels import get_kernel, get_kernel_int_defines

    spec = pf.ACOUSTIC_TIER_FRAME
    assert spec.define == "WPHI_MAX_LEV"
    for tier in ac.WPHI_LEVEL_TIERS:
        frames = {}
        for symbol in _WPHI_SYMBOLS:
            kernel = (get_kernel("acoustic", symbol)
                      if tier == ac.UNSPECIALIZED_WPHI_LEVEL_TIER else
                      get_kernel_int_defines("acoustic", symbol,
                                             (("WPHI_MAX_LEV", tier),)))
            frames[symbol] = int(kernel.attributes["local_size_bytes"])
        assert max(frames.values()) == spec.frame_bytes(tier), (tier, frames)


@requires_nvrtc_frames
def test_the_unspecialized_acoustic_source_still_compiles_to_its_row():
    """The ``#ifndef`` guard must not move the shipped tier's frame."""
    from woof.core.kernels import get_kernel

    assert max(int(get_kernel("acoustic", symbol)
                   .attributes["local_size_bytes"])
               for symbol in _WPHI_SYMBOLS) == (
        pf.KERNEL_MAX_LOCAL_SIZE_BYTES["acoustic"]) == 544


def _w_phi_bytes(kernel, seed, nz):
    """One ``advance_w_phi`` launch against a supplied kernel object.

    Only the compiled bound differs between two calls, so a byte difference
    in the outputs is a difference the tier caused.
    """
    from woof.core import acoustic as acmod
    from woof.verify.npref import random_acoustic_state

    state, cfg = random_acoustic_state(seed=seed, nz=nz, ny=4, nx=8)
    real_pick = acmod._w_phi_kernel
    acmod._w_phi_kernel = lambda name, levels, ny=0, nx=0: kernel
    try:
        acmod.acoustic_substep(state, cfg, dtau=0.5, first=False)
    finally:
        acmod._w_phi_kernel = real_pick
    return tuple(_raw(getattr(state, name))
                 for name in ("w_pp", "ph_pp", "p_pp", "al_pp"))


@requires_nvrtc_frames
def test_a_deeper_acoustic_tier_moves_no_bits_below_the_ceiling():
    """AC-P2.2's tier-equivalence half, measured rather than asserted.

    A below-ceiling column solved by the 193-tier module must agree bit for
    bit with the same column solved by the shipped module.  The bound is a
    pure allocation size, so this is what "by construction" has to mean when
    a driver is asked.
    """
    from woof.core.kernels import get_kernel, get_kernel_int_defines

    nz = 64
    shipped = get_kernel("acoustic", "advance_w_phi")
    deep = get_kernel_int_defines("acoustic", "advance_w_phi",
                                  (("WPHI_MAX_LEV", 193),))
    assert (int(deep.attributes["local_size_bytes"])
            > int(shipped.attributes["local_size_bytes"])), (
        "the two kernels must be different binaries or this proves nothing")
    assert _w_phi_bytes(shipped, 11, nz) == _w_phi_bytes(deep, 11, nz)


def test_the_acoustic_ceiling_is_still_refused_above_the_top_tier():
    from woof.core import acoustic as acmod

    with pytest.raises(ValueError, match="exceeds the in-thread solve limit"):
        acmod.wphi_level_tier(acmod.MAX_ACOUSTIC_LEVELS + 1)


# ---------------------------------------------------------------------------
# RTE+RRTMGP: compiled to each call's radiation layer count
# ---------------------------------------------------------------------------

@requires_nvrtc_frames
def test_the_rrtmgp_rte_layer_frames_are_what_preflight_prices():
    """``preflight.rrtmgp_rte_frame_bytes`` against the driver, past 128.

    ``woof/core/rrtmgp.py`` compiles ``RRTMGP_MAX_LAYERS`` to the call's
    own layer count (``_rte_kernel``), the two-stream at ``nz + sw_upper``
    and the LW solver at ``nz + lw_upper``.  The price must never be below
    what either kernel compiles to at the depths a 256-level column reaches
    -- including the 1 hPa top (256 LW + 257 SW layers) and a 100 hPa top
    (281 LW) -- and the two-stream's tier must be EXACT wherever 28 n + 12
    is not a multiple of 16 (16 B over where it is, the stated ceiling).
    """
    from woof.core.rrtmgp import _rte_kernel
    from woof.physics_vertical_contract import rrtmgp_above_model_layer_counts

    spec = pf.RRTMGP_RTE_SW_TIER_FRAME
    assert (spec.module, spec.define, spec.shipped_tier) == (
        "rrtmgp_rte", "RRTMGP_MAX_LAYERS", 128)
    for layers in (128, 129, 160, 192, 256, 257, 260, 269, 281):
        sw = int(_rte_kernel("rrtmgp_sw_2stream", layers)
                 .attributes["local_size_bytes"])
        lw = int(_rte_kernel("rrtmgp_lw_noscat", layers)
                 .attributes["local_size_bytes"])
        priced = spec.frame_bytes(layers)
        assert sw <= priced <= sw + 16, (layers, sw, priced)
        if (28 * layers + 12) % 16:
            assert priced == sw, (layers, sw, priced)
        assert lw == pf.RRTMGP_RTE_LW_BYTES_PER_LAYER * layers, (layers, lw)
    for nz, p_top in ((128, 100.0), (200, 5000.0), (256, 100.0),
                      (256, 10000.0)):
        lw_upper, sw_upper = rrtmgp_above_model_layer_counts(p_top)
        compiled = max(
            int(_rte_kernel("rrtmgp_sw_2stream", nz + sw_upper)
                .attributes["local_size_bytes"]),
            int(_rte_kernel("rrtmgp_lw_noscat", nz + lw_upper)
                .attributes["local_size_bytes"]))
        assert compiled <= pf.rrtmgp_rte_frame_bytes(nz, p_top), (nz, p_top)
