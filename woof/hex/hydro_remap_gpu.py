"""The hex cell <-> LSM grid exchange on the device: the opt-in route of :mod:`woof.hex.hydro_remap`.

:class:`woof.hex.hydro_remap.CellGridExchange` holds the overlap weights on the host and runs each product
through scipy, so the WRF-Hydro coupling on hex (woof.hex.wrf_hydro_hex) copies the soil, runoff and head
planes to the host and back at every LSM step.  :class:`DeviceExchange` keeps the same weights resident on the
device, built once per mesh and grid as CSR in both directions (grid rows; and cell rows, the transpose, each
cell's entries in ascending grid order), and runs the forward overlap-weighted mean, the transpose back-map,
the increment return and the category majority as kernels (``hydro_remap.cu``), so the state never leaves the
device across the hand-off.

Every output is one thread's sum over its row, in the stored order from 0.0, with the products and sums
rounded as scipy's CSR/CSC loops round them on this host (:func:`host_fuses`: a scipy built with multiply-add
contraction computes ``y += a * x`` as one fma) and the correctly rounded quotient; the majority sums each
class's area in the COO order numpy's ``add.at`` uses.  So the device words are the host operator's words on
the host the run starts from (tests/test_hex_hydro_remap_gpu.py holds them byte for byte), but the route
stays an opt-in of the coupling (woof.hex.wrf_hydro_hex): the host route is the default and is unchanged.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

_SOURCE = Path(__file__).with_name("hydro_remap.cu")
#: no flush: a binary32 cast of a subnormal mean keeps it, as numpy's astype does
OPTIONS = ("--std=c++17", "--ftz=false")
_THREADS = 128


@lru_cache(maxsize=None)
def host_fuses() -> tuple[bool, bool]:
    """Whether this host's scipy fuses ``y += a * x`` into one fma in the CSR product (``to_grid``) and in the
    transposed (CSC) product (``to_cells``).  Probed on one row whose two answers differ: ``1 * -1`` then
    ``(1 + 2^-27)(1 - 2^-27) = 1 - 2^-54``, which rounds to 1 unfused (sum 0) and stays exact fused
    (sum -2^-54)."""
    import scipy.sparse as sp
    w = np.array([1.0, 1.0 + 2.0 ** -27])
    x = np.array([-1.0, 1.0 - 2.0 ** -27])
    row = sp.csr_matrix((w, np.array([0, 1]), np.array([0, 2])), shape=(1, 2))
    column = sp.csr_matrix((w, np.array([0, 0]), np.array([0, 1, 2])), shape=(2, 1))
    out = []
    # the host's two products: overlap @ x (CSR) and overlap.T @ x (the CSC view of a CSR's transpose)
    for product in (lambda v: row @ v, lambda v: column.T @ v):
        got = {float(np.asarray(product(np.repeat(x[:, None], n, axis=1)))[0, 0]) for n in (1, 4)}
        if len(got) != 1 or not got <= {0.0, -2.0 ** -54}:
            raise RuntimeError(f"this host's scipy sparse product is neither the plain nor the fused loop: {got}")
        out.append(got == {-2.0 ** -54})
    return tuple(out)


def _module():
    import cupy as cp
    key = cp.cuda.Device().id
    if key not in _MODULES:
        from woof.certify.kernel_manifest import record_module
        source = _SOURCE.read_text()
        module = cp.RawModule(code=source, options=OPTIONS)
        record_module("woof.hex.hydro_remap_gpu", source=source, options=OPTIONS, module=module)
        _MODULES[key] = {name: module.get_function(name) for name in (
            "hydro_to_grid", "hydro_increment", "hydro_to_cells", "hydro_dominant", "hydro_budget_cells",
            "hydro_budget_grid")}
    return _MODULES[key]


_MODULES: dict = {}


def _blocks(n):
    return ((int(n) + _THREADS - 1) // _THREADS,)


class DeviceExchange:
    """:class:`woof.hex.hydro_remap.CellGridExchange`'s operator, resident on the device."""

    def __init__(self, exchange):
        import cupy as cp
        self.host = exchange
        self.shape = tuple(exchange.shape)
        self.cells, self.grid = len(exchange.cell_area), len(exchange.grid_area)
        m = exchange.overlap
        if m.shape != (self.grid, self.cells):
            raise ValueError(f"overlap has shape {m.shape}, the exchange ({self.grid}, {self.cells})")
        if m.nnz >= 2 ** 31:
            raise ValueError("the overlap has more entries than an int32 CSR holds")
        # Forward: the host CSR as stored (scipy's product visits each row in this order).
        indptr, index, weight = (np.asarray(m.indptr, np.int64), np.asarray(m.indices, np.int64),
                                 np.asarray(m.data, np.float64))
        # Transpose: scipy's CSC product of overlap.T visits grid rows in order, each row's entries as stored,
        # so a cell's terms arrive in ascending grid row (stable in the stored order).
        rows = np.repeat(np.arange(self.grid, dtype=np.int64), np.diff(indptr))
        order = np.lexsort((np.arange(len(index)), rows, index))
        t_indptr = np.zeros(self.cells + 1, np.int64)
        np.cumsum(np.bincount(index, minlength=self.cells), out=t_indptr[1:])
        i32 = lambda a: cp.asarray(np.ascontiguousarray(a, np.int32))  # noqa: E731
        f64 = lambda a: cp.asarray(np.ascontiguousarray(a, np.float64))  # noqa: E731
        self.indptr, self.index, self.weight = i32(indptr), i32(index), f64(weight)
        self.t_indptr, self.t_index, self.t_weight = i32(t_indptr), i32(rows[order]), f64(weight[order])
        self.cell_area, self.grid_area = f64(exchange.cell_area), f64(exchange.grid_area)
        self.covered = f64(np.asarray(m.sum(axis=0)).ravel())
        self.fuse = tuple(np.int32(f) for f in host_fuses())
        self.kernels = _module()

    def to_grid(self, source, *, fields, stride, exact, out):
        """Overlap-weighted means of ``fields`` binary32 cell rows of ``source`` (``stride`` words apart, the
        first ``cells`` of each row) into ``exact`` (binary64) and ``out`` (binary32), each ``(fields, grid)``."""
        n = int(fields) * self.grid
        self.kernels["hydro_to_grid"](_blocks(n), (_THREADS,), (
            self.indptr, self.index, self.weight, self.grid_area, source, np.int64(stride), np.int32(fields),
            np.int32(self.grid), self.fuse[0], exact, out))

    def increment(self, before, after, state, *, fields, stride, prior, exact):
        """``state``'s first ``cells`` words of each of ``fields`` rows (``stride`` apart) += the cells' share
        of ``after - before`` (binary32 grids), in binary64 then cast in place; ``prior`` keeps the old state
        and ``exact`` the binary64 result, each ``(fields, cells)``."""
        n = int(fields) * self.cells
        self.kernels["hydro_increment"](_blocks(n), (_THREADS,), (
            self.t_indptr, self.t_index, self.t_weight, self.cell_area, before, after, np.int32(self.grid),
            state, np.int64(stride), np.int32(fields), np.int32(self.cells), self.fuse[1], prior, exact))

    def to_cells(self, source, *, exact, out):
        """A binary32 grid field returned to the cells into ``exact`` (binary64) and ``out``'s first ``cells``
        words (binary32)."""
        self.kernels["hydro_to_cells"](_blocks(self.cells), (_THREADS,), (
            self.t_indptr, self.t_index, self.t_weight, self.cell_area, source, np.int32(self.cells),
            self.fuse[1], exact, out))

    def budget(self, dz, *, prior, state, stride, exact, head_cells, infx, sold, before, after, head, infx_grid,
               sold_grid, infx_exact, sold_exact):
        """The coupling's water budget terms (m3): six cell sums and five grid sums (``hydro_budget_cells``,
        ``hydro_budget_grid``), two device vectors.  ``dz`` is the layer thicknesses on the device."""
        import cupy as cp
        layers = np.int32(dz.size)
        cells = cp.empty((6, self.cells), cp.float64)
        grid = cp.empty((5, self.grid), cp.float64)
        self.kernels["hydro_budget_cells"](_blocks(self.cells), (_THREADS,), (
            dz, layers, self.cell_area, prior, state, np.int64(stride), exact, head_cells, infx, sold,
            self.covered, np.int32(self.cells), cells))
        self.kernels["hydro_budget_grid"](_blocks(self.grid), (_THREADS,), (
            dz, layers, self.grid_area, before, after, head, infx_grid, sold_grid, infx_exact, sold_exact,
            np.int32(self.grid), grid))
        return cp.sum(cells, axis=1), cp.sum(grid, axis=1)

    def dominant(self, categories):
        """Per grid cell, the class of ``categories`` (one int32 per hex cell, on the device) with the largest
        overlap area, ties to the smaller class: an int32 ``(ny, nx)`` device array."""
        import cupy as cp
        cats = cp.ascontiguousarray(cp.asarray(categories).reshape(-1)[:self.cells].astype(cp.int32))
        if cats.size != self.cells:
            raise ValueError("one category per hex cell")
        out = cp.empty(self.grid, cp.int32)
        self.kernels["hydro_dominant"](_blocks(self.grid), (_THREADS,), (
            self.indptr, self.index, self.weight, cats, np.int32(self.grid), out))
        return out.reshape(self.shape)
