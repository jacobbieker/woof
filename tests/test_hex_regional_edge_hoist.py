"""The hoisted regional edge-value kernel writes the words of the element kernel it replaces.

``woof.hex.cuda_regional_hoist_v841.transport_edge_values_regional_hoist_v841`` runs one thread per
(edge, level chunk) walking the tracers; ``transport_edge_values_regional_v841`` one thread per
(tracer, level, edge).  Both run on the same synthetic regional mesh -- every boundary-mask class (interior,
the two upwind relaxation rows, the specified zone), stencil widths under, at and over the hoisted ten,
WRF-like advection weights, and stage values that are subnormal, signed zeros and tiny enough for the
weight products to flush, so the subnormal-sensitivity fallback is taken -- into poisoned outputs, and every
word is compared, at several levels-per-thread settings.  ``test_identity_holds_under_each_numerics`` repeats
it in a child process per numerics mode (fast is production).
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import _numerics_child

NLEV, NCELLS, NEDGES, WIDTH = 13, 211, 617, 12


def _mesh(ntracers, seed=20261116):
    from woof.hex.regional_v841 import N_RELAX_ZONE

    rng = np.random.default_rng(seed)
    f32, i32 = np.float32, np.int32

    def poison(a, fraction=0.02):
        flat = a.reshape(-1)
        picks = rng.choice(flat.size, size=max(8, int(flat.size * fraction)), replace=False)
        specials = np.array([1e-40, -1e-41, -0.0, 0.0, 1e-38, -1.2e-38, 3e-39, 1e-45], dtype=f32)
        flat[picks] = specials[np.arange(picks.size) % specials.size]
        return a

    # mixing ratios: mostly 1e-6..1e-2, some exactly zero or tiny
    stage = (10.0 ** rng.uniform(-6.0, -2.0, size=(ntracers, NLEV, NCELLS))).astype(f32)
    stage *= rng.choice([1.0, 1.0, 1.0, -1.0], size=stage.shape).astype(f32)
    stage = poison(stage, 0.05)
    velocity = poison(rng.normal(0.0, 12.0, size=(NLEV, NEDGES)).astype(f32))
    velocity[:, :5] = np.array([0.0, -0.0, 1e-40, -1e-40, 5.0], dtype=f32)[None, :]
    # MPAS adv_coefs ~ dcEdge-scaled stencil weights; adv_coefs_3rd an order smaller
    adv = poison(rng.normal(0.0, 0.3, size=(NEDGES, WIDTH)).astype(f32), 0.01)
    adv3 = poison(rng.normal(0.0, 0.04, size=(NEDGES, WIDTH)).astype(f32), 0.01)
    # weights whose sum cancels into the subnormal range
    adv[7, 0], adv3[7, 0] = f32(1.0e-38), f32(-4.0e-38)
    n_adv = rng.choice([10, 10, 10, 9, 7, 12, 11, 4], size=NEDGES).astype(i32)
    adv_cells = rng.integers(0, NCELLS, size=(NEDGES, WIDTH)).astype(i32)
    mask = rng.choice(np.arange(0, N_RELAX_ZONE + 3), size=NEDGES,
                      p=None).astype(i32)
    mask[:NEDGES // 2] = 0
    rng.shuffle(mask)
    cells_on_edge = rng.integers(0, NCELLS, size=(NEDGES, 2)).astype(i32)
    dv_edge = rng.uniform(800.0, 3500.0, size=NEDGES).astype(f32)
    return dict(stage=cp.asarray(stage), velocity=cp.asarray(velocity), adv=cp.asarray(adv),
                adv3=cp.asarray(adv3), n_adv=cp.asarray(n_adv), adv_cells=cp.asarray(adv_cells),
                mask=cp.asarray(mask), cells_on_edge=cp.asarray(cells_on_edge), dv_edge=cp.asarray(dv_edge))


def _words(a):
    return cp.asnumpy(cp.ascontiguousarray(a).view(cp.uint32))


@pytest.mark.parametrize("ntracers", [1, 3])
@pytest.mark.parametrize("lpt", [1, 4, NLEV])
@pytest.mark.parametrize("coefficient", [0.25, 1.0])
def test_hoisted_regional_edge_values_match_the_element_kernel(ntracers, lpt, coefficient):
    from woof.hex import cuda_regional_hoist_v841 as hoist
    from woof.hex.cuda_backend import KernelCache
    from woof.hex.cuda_regional_v841 import CudaRegionalKernels

    cache = KernelCache()
    m = _mesh(ntracers)
    arguments = (np.int32(ntracers), np.int32(NLEV), np.int32(NCELLS), np.int32(NEDGES), np.int32(WIDTH),
                 np.float32(coefficient), m["mask"], m["dv_edge"], m["cells_on_edge"], m["stage"],
                 m["velocity"], m["adv"], m["adv3"], m["n_adv"], m["adv_cells"])
    element = cp.full((ntracers, NLEV, NEDGES), np.nan, dtype=cp.float32)
    hoisted = cp.full_like(element, -7.0)
    CudaRegionalKernels(cache).launch(hoist.ELEMENT_KERNEL, ntracers * NLEV * NEDGES, (*arguments, element))
    hoist.launch(cache, (*arguments, hoisted), lpt)
    got, want = _words(hoisted), _words(element)
    assert not np.isnan(cp.asnumpy(element)).any()
    assert np.array_equal(got, want), f"{int((got != want).sum())} words differ"


def test_the_cases_reach_every_branch():
    """The mesh exercises the specified-zone, upwind and full branches, and the hoisted and unhoisted
    stencil widths."""
    from woof.hex.regional_v841 import N_RELAX_ZONE

    m = _mesh(3)
    mask = cp.asnumpy(m["mask"])
    full = mask < N_RELAX_ZONE - 1
    assert (mask > N_RELAX_ZONE).any() and ((mask >= N_RELAX_ZONE - 1) & (mask <= N_RELAX_ZONE)).any()
    n_adv = cp.asnumpy(m["n_adv"])
    assert (full & (n_adv > 10)).any() and (full & (n_adv == 10)).any() and (full & (n_adv < 10)).any()


@pytest.mark.gpu
@pytest.mark.parametrize(*_numerics_child.MODES)
def test_identity_holds_under_each_numerics(numerics, exact):
    # The hex units read WOOF_HEX_NUMERICS (woof.hex.numerics), not GPUWM_NUMERICS: name it, or every child
    # compiles them fast.
    _numerics_child.run(__file__, ["test_hoisted_regional_edge_values_match_the_element_kernel"], numerics, exact,
                        env={"WOOF_HEX_NUMERICS": numerics})
