"""wrf_chem boundary rows behind a partly driven block: the regional boundary kernels' second run.

The standard rw_mpas_lbc stream carries qv, qc and qr; a WSM6 run integrates six masses, and the
chemistry rows follow all six.  The driving pool holds the stream's species and then the rows, so the
regional runtime (woof.hex.cuda_regional_forecast_v841) launches the scalar boundary kernels once for
the stream's leading species and once more for the rows, on their own sub-array.  These checks drive
the real kernels on a synthetic seven-ring strip:

* a run without supplied rows keeps its launches exactly (same arrays, same counts) and its words;
* the rows' run is word-identical to the same kernels on the rows alone, the leading species are
  untouched by it, and the undriven block species stay the scheme's;
* the support-local ring path and the full-extent path give the same words.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from woof.hex.cuda_regional_forecast_v841 import driven_scalar_segments
from woof.hex.regional_v841 import N_RELAX_ZONE, RegionalAdmissionError, RegionalMasks


def test_the_segments_are_one_leading_run_without_supplied_rows():
    assert driven_scalar_segments(3) == ((0, 0, 3),)
    assert driven_scalar_segments(3, None) == ((0, 0, 3),)
    # WSM6 + four rows on a qv/qc/qr stream: the rows start at model species 6.
    assert driven_scalar_segments(7, (0, 1, 2, 6, 7, 8, 9)) == ((0, 0, 3), (6, 3, 4))
    # A stream carrying the whole block makes the runs adjacent: one launch, as before.
    assert driven_scalar_segments(8, tuple(range(8))) == ((0, 0, 8),)
    # Thompson + coupling: qv/qc/qr, then nwfa, nifa and the rows after ni/nr/nc.
    assert driven_scalar_segments(6, (0, 1, 2, 9, 10, 11)) == ((0, 0, 3), (9, 3, 3))
    for bad in ((1, 2, 3), (0, 2, 1), (0, 0, 1)):
        with pytest.raises(RegionalAdmissionError):
            driven_scalar_segments(3, bad)
    with pytest.raises(RegionalAdmissionError):
        driven_scalar_segments(3, (0, 1))


# -- the kernels on a synthetic strip ------------------------------------------------------------

#: one strip of cells: two interior cells, then rings 1..7, and the garbage column (ring 7)
_RINGS = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 7], dtype=np.int64)


def _masks():
    bdy = _RINGS
    edges = np.zeros(bdy.size - 1, dtype=np.int64)
    spec = (bdy > N_RELAX_ZONE).astype(np.float32)
    return RegionalMasks(
        bdy_mask_cell=bdy, bdy_mask_edge=edges, bdy_mask_vertex=bdy,
        spec_zone_mask_cell=spec, spec_zone_mask_edge=np.zeros(edges.size, np.float32),
        spec_zone_mask_vertex=spec, nearest_relaxation_cell=np.zeros_like(bdy),
        spec_cells=np.flatnonzero(bdy > N_RELAX_ZONE), spec_edges=np.zeros(0, np.int64),
        relax_cells=np.flatnonzero((bdy > 1) & (bdy <= N_RELAX_ZONE)), relax_edges=np.zeros(0, np.int64),
        nudged_cells=np.flatnonzero(bdy > 1))


class _Driving:
    """The driving pool's two views: the full extent, and the compact boundary support (every cell)."""

    def __init__(self, cp, pool):
        self.pool = pool
        self.ncells = pool.shape[-1]
        self._garbage_cell = self.ncells - 1
        self._support = {"cell": cp.arange(self.ncells, dtype=cp.int64)}

    def state_at(self, name, when, dt):
        if name in ("rtheta_m", "rho_zz"):
            return self.pool[0] + (1.0 if name == "rho_zz" else 300.0)
        assert name == "scalars"
        return self.pool

    def state_at_support(self, name, when, dt):
        assert name == "scalars"
        return self.pool.copy(), self._support["cell"]


def _runtime(cp, pool, segments):
    from woof.hex.cuda_backend import KernelCache
    from woof.hex.cuda_regional_forecast_v841 import CudaRegionalRuntimeV841
    from woof.hex.cuda_regional_v841 import CudaRegionalKernels, DeviceRegionalMasks
    ncells = _RINGS.size
    nedges = ncells - 1
    # cell c owns edges c-1 (to its left) and c (to its right); edge e joins cells e and e+1.
    cells_on_edge = np.stack([np.arange(nedges), np.arange(1, ncells)], axis=1).astype(np.int32)
    edges_on_cell = np.zeros((ncells, 2), np.int32)
    signs = np.zeros((ncells, 2), np.float32)
    count = np.zeros(ncells, np.int32)
    for c in range(ncells):
        for e, sign in ((c - 1, -1.0), (c, 1.0)):
            if 0 <= e < nedges:
                edges_on_cell[c, count[c]] = e
                signs[c, count[c]] = sign
                count[c] += 1
    rt = object.__new__(CudaRegionalRuntimeV841)
    rt.cp = cp
    rt.kernels = CudaRegionalKernels(KernelCache())
    rt.masks = DeviceRegionalMasks.from_host(_masks())
    rt.atmosphere = NS(reference=NS(rho_theta_base=cp.zeros((int(pool.shape[1]), ncells), cp.float32)), mesh=NS(
        n_edges=nedges, max_edges=2, n_edges_on_cell=cp.asarray(count), edges_on_cell=cp.asarray(edges_on_cell),
        cells_on_edge=cp.asarray(cells_on_edge), edge_sign_on_cell=cp.asarray(signs),
        dv_edge=cp.asarray(np.linspace(0.9, 1.3, nedges, dtype=np.float32))))
    rt.context = NS(inv_dc_edge=cp.asarray(np.linspace(1.1, 0.7, nedges, dtype=np.float32)))
    rt.mesh_scaling_cell = cp.asarray(np.linspace(1.0, 2.0, ncells, dtype=np.float32))
    rt.nlev = int(pool.shape[1])
    rt.outer_dt = 30.0
    rt.time_integration_order = 3
    rt.clock = None
    rt.driving = _Driving(cp, pool)
    rt.driven_scalars = int(pool.shape[0])
    rt.scalar_segments = segments
    return rt


def _step(rt, scalars, *, ring, monkeypatch):
    monkeypatch.setenv("WOOF_HEX_REGIONAL_RING_SCALARS", "1" if ring else "0")
    rt._ring_plan_cache = None
    for rk in (1, 2, 3):
        rt.bdy_adjust_scalars(scalars, rk)
    # the end-of-step specified-zone set (reset_speczone_values; its theta reset rides along)
    cp = rt.cp
    level = scalars[0].shape
    rt.reset_speczone_values(theta_m=cp.zeros(level, cp.float32), rho_theta=cp.zeros(level, cp.float32),
                             rho_theta_perturbation=cp.zeros(level, cp.float32), scalars=scalars)
    return scalars


def _fields(nspecies, nlev, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0e-2, (nspecies, nlev, _RINGS.size)).astype(np.float32)


@pytest.mark.parametrize("ring", [False, True])
def test_the_rows_run_is_the_kernels_on_the_rows_alone(ring, monkeypatch):
    cp = pytest.importorskip("cupy")
    nlev = 4
    model = _fields(10, nlev, 1)            # qv qc qr qi qs qg | o3 no2 co so2
    pool = _fields(7, nlev, 2)              # lbc_qv lbc_qc lbc_qr | lbc_o3 lbc_no2 lbc_co lbc_so2
    chem = _step(_runtime(cp, cp.asarray(pool), driven_scalar_segments(7, (0, 1, 2, 6, 7, 8, 9))),
                 cp.asarray(model), ring=ring, monkeypatch=monkeypatch).get()
    # The leading run alone: the run without chemistry rows, on the same model and stream species.
    plain = _step(_runtime(cp, cp.asarray(pool[:3]), driven_scalar_segments(3)),
                  cp.asarray(model[:6]), ring=ring, monkeypatch=monkeypatch).get()
    rows = _step(_runtime(cp, cp.asarray(pool[3:]), driven_scalar_segments(4)),
                 cp.asarray(model[6:]), ring=ring, monkeypatch=monkeypatch).get()
    assert np.array_equal(chem[:6].view(np.uint32), plain.view(np.uint32))
    assert np.array_equal(chem[6:].view(np.uint32), rows.view(np.uint32))
    assert np.array_equal(chem[3:6], model[3:6]), "the block's undriven species stay the scheme's"
    assert not np.array_equal(chem[6:], model[6:]), "the rows are driven at the rings"
    spec = np.flatnonzero(_RINGS > N_RELAX_ZONE)
    assert np.array_equal(chem[6:, :, spec], pool[3:, :, spec]), "the specified zone takes the rows' boundary"
    assert np.array_equal(chem[:, :, :2], model[:, :, :2]), "the interior is untouched"


def test_the_ring_and_full_paths_agree_on_the_rows(monkeypatch):
    cp = pytest.importorskip("cupy")
    model, pool = _fields(10, 3, 3), _fields(7, 3, 4)
    segments = driven_scalar_segments(7, (0, 1, 2, 6, 7, 8, 9))
    full = _step(_runtime(cp, cp.asarray(pool), segments), cp.asarray(model), ring=False,
                 monkeypatch=monkeypatch).get()
    ring = _step(_runtime(cp, cp.asarray(pool), segments), cp.asarray(model), ring=True,
                 monkeypatch=monkeypatch).get()
    assert np.array_equal(full.view(np.uint32), ring.view(np.uint32))


def test_a_run_without_rows_keeps_its_launches(monkeypatch):
    """The leading run hands the kernels the model's whole scalar array and the whole driving pool, as
    the single-run code always did: the launches are the same calls, so the words are the same."""
    cp = pytest.importorskip("cupy")
    monkeypatch.setenv("WOOF_HEX_REGIONAL_RING_SCALARS", "0")
    model, pool = cp.asarray(_fields(6, 3, 5)), cp.asarray(_fields(3, 3, 6))
    rt = _runtime(cp, pool, driven_scalar_segments(3))
    seen = []
    launch = rt.kernels.launch

    def record(name, count, args):
        seen.append((name, count, args))
        return launch(name, count, args)

    monkeypatch.setattr(rt.kernels, "launch", record)
    rt.bdy_adjust_scalars(model, 1)
    names = [name for name, _, _ in seen]
    assert names == ["regional_bdy_adjust_scalars_compute_v841", "regional_bdy_adjust_scalars_copyback_v841"]
    compute, copyback = seen[0][2], seen[1][2]
    assert int(compute[0]) == 3 and compute[-3] is model and compute[-2] is pool
    assert compute[-1].shape == model.shape, "the update buffer spans the whole scalar array, as before"
    assert int(copyback[0]) == 3 and copyback[-1] is model
    masks = rt.masks
    assert seen[0][1] == 3 * 3 * (masks.n_relax_cells + masks.n_spec_cells)
