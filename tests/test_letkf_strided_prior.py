"""A strided prior gives the contiguous prior's increments.

Breakage prevented: ``analyze`` built its increments with the prior's
layout and wrote them through ``reshape(members, -1)``; a prior that was
not C-contiguous (a fancy-indexed or transposed view) made that reshape a
copy, so every write was lost and the analysis returned all-zero
increments without an error.
"""
import numpy as np

from woof.da import letkf
from test_letkf_host_parallel import geodesic_problem


def _increments(prior, obs, grid, cfg):
    return letkf.analyze(prior, obs, grid, cfg)


def test_strided_prior_gives_the_contiguous_increments():
    prior, obs, grid, cfg = geodesic_problem(fields=2, batches=2, members=6,
                                             ny=8, nx=9, nz=6)
    want = _increments(prior, obs, grid, cfg)
    # Same values, not C-contiguous: a Fortran-ordered copy, and an axes
    # swap of a contiguous array (the layout fancy indexing can leave).
    strided = {}
    for n, (f, a) in enumerate(prior.items()):
        if n % 2:
            strided[f] = np.asfortranarray(a)
        else:
            strided[f] = np.swapaxes(np.ascontiguousarray(np.swapaxes(a, 1, 3)), 1, 3)
        assert not strided[f].flags.c_contiguous
        assert np.array_equal(strided[f], a)
    got = _increments(strided, obs, grid, cfg)
    for f in prior:
        w, g = np.asarray(want[f]), np.asarray(got[f])
        assert np.any(w != 0.0), f
        assert np.array_equal(w.view(np.uint64), g.view(np.uint64)), f
