"""The partial-depth nest's truncated column (woof/core/partial_nest.py).

Below a pure-pressure interface the child's coordinate must give the
parent's dry pressure on every shared level and the parent's mass per unit
eta, so the child's discrete equations there are the parent's.
"""

import numpy as np
import pytest

from woof.core.grid import finalize_vertical_coord, make_vertical_coord
from woof.core.partial_nest import (PartialNestRefused, child_column_mass,
                                    interface_pressure, truncated_vertical_coord)

P_TOP = 5000.0


def _parent(nz=40, etac=0.45):
    coord = make_vertical_coord(nz, hybrid_opt=2, etac=etac)
    finalize_vertical_coord(coord, P_TOP)
    return coord


def _pressure(c3, c4, mu, p_top):
    return c3[:, None] * mu[None, :] + c4[:, None] + p_top


def _close(got, want, *terms):
    """Equal to float64 rounding of the terms summed (they cancel, so the
    bound scales with their magnitudes, not with the result)."""
    scale = sum(np.abs(t) for t in terms)
    assert np.all(np.abs(got - want) <= 8 * np.finfo(np.float64).eps * scale)


def test_shared_levels_keep_the_parents_pressure_and_mass_per_eta():
    parent = _parent()
    k_top = int(np.flatnonzero(parent.c3f == 0.0)[0]) + 2      # two levels above etac
    child = truncated_vertical_coord(parent, k_top)
    mu = np.linspace(40000.0, 101000.0, 64) - P_TOP           # parent column masses
    mu_c = child_column_mass(mu, parent, k_top)
    for c3, c4, c3c, c4c in ((parent.c3f[:k_top + 1], parent.c4f[:k_top + 1], child.c3f, child.c4f),
                             (parent.c3h[:k_top], parent.c4h[:k_top], child.c3h, child.c4h)):
        want = _pressure(c3, c4, mu, P_TOP)
        got = _pressure(c3c, c4c, mu_c, child.p_top)
        _close(got, want, c3[:, None] * mu[None, :], c4[:, None], P_TOP, child.p_top)
    for c1, c2, c1c, c2c in ((parent.c1f[:k_top + 1], parent.c2f[:k_top + 1], child.c1f, child.c2f),
                             (parent.c1h[:k_top], parent.c2h[:k_top], child.c1h, child.c2h)):
        want = c1[:, None] * mu[None, :] + c2[:, None]
        got = c1c[:, None] * mu_c[None, :] + c2c[:, None]
        _close(got, want, c1[:, None] * mu[None, :], c2[:, None], c1[:, None] * child.p_top)


def test_the_interface_is_the_childs_top_at_one_pressure():
    parent = _parent()
    k_top = int(np.flatnonzero(parent.c3f == 0.0)[0])
    child = truncated_vertical_coord(parent, k_top)
    assert child.c4f[-1] == 0.0 and child.c3f[-1] == 0.0
    assert child.p_top == interface_pressure(parent, k_top) > P_TOP
    assert len(child.znw) == k_top + 1 and len(child.znu) == k_top
    assert np.array_equal(child.znw, parent.znw[:k_top + 1])
    assert np.array_equal(child.dnw, parent.dnw[:k_top])


def test_a_terrain_following_interface_is_refused_by_name():
    parent = _parent()
    k_low = int(np.flatnonzero(parent.c3f == 0.0)[0]) - 1
    with pytest.raises(PartialNestRefused, match="pure pressure level"):
        truncated_vertical_coord(parent, k_low)
    with pytest.raises(PartialNestRefused, match="interior full level"):
        truncated_vertical_coord(parent, len(parent.znu))
