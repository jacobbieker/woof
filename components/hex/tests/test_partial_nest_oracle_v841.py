"""Hex partial-depth band (Phase D5): coupling operators held to compiled MPAS.

The fixture (tests/data/oracles/partial_nest_mpas/fixture.npz, built by
tools/partial_nest_mpas_oracle: MPAS-Model v8.4.1's own routines cut by
extract.py, gfortran -O0 -ffp-contract=off) carries every input and MPAS's
output words.  Gate (b):

* ``mpas_calculate_barycentric_weights_for_points`` (RKIND 8) against
  :func:`woof.hex.partial_nest_coupling.barycentric_weights` -- the cells
  and every binary64 weight word;
* ``mpas_cells_to_points_using_baryweights`` (RKIND 4) against the device
  kernel ``pn_bary_apply_v841`` and the host reference;
* ``atm_bdy_adjust_dynamics_relaxzone_tend`` (RKIND 4), driven one level at a
  time with every active cell and edge at ring strength 5 - d on the level
  d below the top, against the top-zone kernels ``pn_topzone_*_v841`` (MPAS's
  relaxation-zone arithmetic with the ring strength per level).

Gate (b'): the fine-to-coarse conservative overlap remap -- the device kernel
``pn_remap_rows_v841`` equals the binary64 reference
(:meth:`FeedbackRemap.reference`, the same sums in the same order) word for
word, and the reference conserves the mass-weighted integral to round-off.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURE = (Path(__file__).resolve().parents[3]
           / "tests/data/oracles/partial_nest_mpas/fixture.npz")


def requires_gpu(test):
    """Run on a device the hex lane admits (CUDA, or Metal through the MLX
    translation); skip otherwise.  The battery's numerics are strict
    (conftest.py), the bitwise route."""
    import functools

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        pytest.importorskip("cupy")
        from woof.hex.cuda_backend import KernelCache
        from woof.hex.cuda_backend.runtime import CudaRefusal
        try:
            KernelCache()
        except CudaRefusal as error:
            pytest.skip(f"no admitted hex device: {error}")
        return test(*args, **kwargs)
    return wrapper


@pytest.fixture(scope="module")
def fix():
    with np.load(FIXTURE) as data:
        return {k: data[k] for k in data.files}


def _same(got, want, what):
    g = np.ascontiguousarray(got)
    w = np.ascontiguousarray(want)
    assert g.shape == w.shape and g.dtype == w.dtype, (what, g.shape, w.shape, g.dtype, w.dtype)
    view = {4: np.uint32, 8: np.uint64}[g.dtype.itemsize]
    bad = np.flatnonzero(g.view(view).ravel() != w.view(view).ravel())
    assert bad.size == 0, (f"{what}: {bad.size} of {g.size} words differ, first at "
                           f"{np.unravel_index(bad[0], g.shape)}: {g.ravel()[bad[0]]!r} vs "
                           f"{w.ravel()[bad[0]]!r}")


@pytest.fixture(scope="module")
def parent(fix):
    from woof.hex.partial_nest_coupling import CellGeometry
    g = lambda name: fix["pg_" + name]
    return CellGeometry(g("x_cell"), g("x_vertex"), g("cells_on_vertex"), g("vertices_on_cell"),
                        g("n_edges_on_cell"), g("area_cell"), float(g("radius")[0]))


def test_barycentric_weights_are_mpas_words(fix, parent):
    from woof.hex.partial_nest_coupling import barycentric_weights
    cells, weights, inside = barycentric_weights(parent, fix["bary_points"], fix["bary_search"])
    assert np.array_equal(cells, fix["bary_cells"])
    _same(weights, fix["bary_weights"], "barycentric weights")
    assert inside.all()


def test_barycentric_application_host_reference_is_mpas_words(fix):
    from woof.hex.partial_nest_coupling import BarycentricTable, bary_apply_reference
    table = BarycentricTable(fix["bary_cells"], fix["bary_weights"],
                             np.ones(len(fix["bary_cells"]), bool))
    _same(table.weights32, fix["apply_weights"], "binary32 weights")
    _same(bary_apply_reference(table, fix["apply_field"]), fix["apply_out"], "bary apply (host)")


def _kernels():
    from woof.hex.cuda_backend import KernelCache
    from woof.hex.cuda_partial_nest_v841 import PartialNestKernels
    return PartialNestKernels(KernelCache())


I32, F32 = np.int32, np.float32


@requires_gpu
def test_barycentric_application_kernel_is_mpas_words(fix):
    import cupy as cp
    k = _kernels()
    field = fix["apply_field"]
    nlev, nsrc = field.shape
    npoints = len(fix["bary_cells"])
    out = cp.zeros((nlev, npoints), cp.float32)
    k.launch("pn_bary_apply_v841", nlev * npoints, (
        I32(nlev), I32(npoints), I32(nsrc), I32(0), I32(npoints),
        cp.asarray(fix["bary_cells"].astype(np.int32)), cp.asarray(fix["apply_weights"]),
        cp.asarray(field), out))
    _same(cp.asnumpy(out), fix["apply_out"], "pn_bary_apply_v841")


@requires_gpu
def test_top_zone_kernels_are_mpas_relaxation_zone_words(fix):
    import cupy as cp
    k = _kernels()
    r = lambda name: fix["relax_" + name]
    d = lambda a: cp.asarray(np.ascontiguousarray(a))
    nlev, nc = r("theta_m").shape
    ne = r("ru").shape[1]
    me = r("edgesOnCell").shape[1]
    vd = r("edgesOnVertex").shape[1]
    idx = lambda name: d(np.maximum(r(name), 0).astype(np.int32))
    cols_c = d(np.flatnonzero(r("activeCell")).astype(np.int32))
    cols_e = d(np.flatnonzero(r("activeEdge")).astype(np.int32))
    dt, divdamp = (np.float32(v) for v in r("scalars"))
    fifty_dt, ten_dt, relax_zone = F32(F32(50.0) * dt), F32(F32(10.0) * dt), F32(5.0)
    tend_rho, tend_rt, tend_ru = d(r("tend_rho")), d(r("tend_rt")), d(r("tend_ru"))
    rho_zz, theta_m, ru = d(r("rho_zz")), d(r("theta_m")), d(r("ru"))
    drho, drt, dru = d(r("rho_driving")), d(r("rt_driving")), d(r("ru_driving"))
    sc, se = d(r("meshScalingRegionalCell")), d(r("meshScalingRegionalEdge"))
    ncols_c, ncols_e = int(cols_c.size), int(cols_e.size)
    k.launch("pn_topzone_rayleigh_cell_v841", nlev * ncols_c, (
        I32(nlev), I32(nc), I32(nlev), I32(ncols_c), fifty_dt, relax_zone, cols_c, sc,
        rho_zz, theta_m, drho, drt, tend_rho, tend_rt))
    k.launch("pn_topzone_rayleigh_edge_v841", nlev * ncols_e, (
        I32(nlev), I32(ne), I32(nlev), I32(ncols_e), fifty_dt, relax_zone, cols_e, se,
        ru, dru, tend_ru))
    k.launch("pn_topzone_filter_cell_v841", nlev * ncols_c, (
        I32(nlev), I32(nc), I32(me), I32(nlev), I32(ncols_c), ten_dt, relax_zone, cols_c, sc,
        d(r("nEdgesOnCell")), idx("edgesOnCell"), idx("cellsOnEdge"), d(r("edgesOnCell_sign")),
        d(r("dvEdge")), d(r("invDcEdge")), rho_zz, theta_m, drho, drt, tend_rho, tend_rt))
    k.launch("pn_topzone_filter_edge_v841", nlev * ncols_e, (
        I32(nlev), I32(ne), I32(me), I32(vd), I32(nlev), I32(ncols_e), ten_dt, F32(divdamp),
        relax_zone, cols_e, se, d(r("nEdgesOnCell")), idx("edgesOnCell"), idx("cellsOnEdge"),
        idx("verticesOnEdge"), idx("edgesOnVertex"), d(r("edgesOnCell_sign")),
        d(r("edgesOnVertex_sign")), d(r("dcEdge")), d(r("dvEdge")), d(r("invDcEdge")),
        d(r("invDvEdge")), d(r("invAreaCell")), d(r("invAreaTriangle")), ru, dru, tend_ru))
    _same(cp.asnumpy(tend_rho), fix["relax_out_tend_rho"], "top zone tend_rho")
    _same(cp.asnumpy(tend_rt), fix["relax_out_tend_rt"], "top zone tend_rt")
    _same(cp.asnumpy(tend_ru), fix["relax_out_tend_ru"], "top zone tend_ru")


def _remap(fix):
    from woof.hex.partial_nest_coupling import FeedbackRemap
    return FeedbackRemap(targets=fix["remap_targets"], row_ptr=fix["remap_row_ptr"],
                         cols=fix["remap_cols"], overlap=fix["remap_overlap"],
                         coverage=fix["remap_coverage"])


def test_remap_rows_cover_their_parent_cells(fix):
    remap = _remap(fix)
    assert len(remap.targets) > 0
    assert np.all(remap.coverage >= 1.0 - 1.0e-9) and np.all(remap.coverage <= 1.0 + 1.0e-9)
    assert np.all(np.diff(remap.row_ptr) > 0)


def test_remap_reference_conserves_to_round_off(fix):
    from woof.hex.partial_nest_coupling import remap_conservation
    lhs, rhs = remap_conservation(_remap(fix), fix["remap_rho"], fix["remap_q"])
    assert np.all(np.abs(lhs - rhs) <= 1.0e-13 * np.abs(rhs)), (lhs - rhs) / rhs


@requires_gpu
def test_remap_kernel_is_the_binary64_reference(fix):
    import cupy as cp
    remap = _remap(fix)
    k = _kernels()
    rho, q = fix["remap_rho"], fix["remap_q"]
    nlev, nsrc = rho.shape
    ntarget = int(fix["remap_parent_cells"][0])
    nrows = len(remap.targets)
    out = cp.zeros((nlev, ntarget), cp.float32)
    k.launch("pn_remap_rows_v841", nlev * nrows, (
        I32(nlev), I32(nrows), I32(nsrc), I32(ntarget),
        cp.asarray(remap.row_ptr.astype(np.int32)), cp.asarray(remap.cols.astype(np.int32)),
        cp.asarray(remap.overlap), cp.asarray(remap.targets.astype(np.int32)),
        cp.asarray(rho), cp.asarray(q), out))
    want = remap.reference(rho, q).astype(np.float32)
    _same(cp.asnumpy(out)[:, remap.targets], want, "pn_remap_rows_v841")
