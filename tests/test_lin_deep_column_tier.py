"""Lin's deep-column tier (LIN_KMAX > 128) sizes the header's arrays too.

woof/core/lin.py selects LIN_KMAX = 256 for 129..256 levels.  The tier define
used to follow lin_support.cuh, whose ``#ifndef LIN_KMAX`` had already fixed
the header functions' arrays (lin_wrf.cuh) at 128, so levels 129 and up of a
149-level hex column came back non-finite at the first call (hex option
matrix, microphysics.lin-mp2).
"""

import numpy as np
import pytest

from woof.core.kernels import module_source_int_defines


def test_the_tier_define_precedes_the_header_default():
    src = module_source_int_defines("lin", (("LIN_KMAX", 256),))
    assert src.index("#define LIN_KMAX 256") < src.index("#ifndef LIN_KMAX")


@pytest.mark.gpu
def test_a_149_level_column_stays_finite():
    cp = pytest.importorskip("cupy")
    from woof.core import lin
    nz, ny, nx = 149, 1, 64
    z = (np.linspace(30.0, 30000.0, nz, dtype=np.float32)[:, None, None]
         * np.ones((1, ny, nx), np.float32))
    # Tabulated, not libm: a standard-atmosphere-like column whose words do not
    # depend on the host's exp/pow.
    t = np.maximum(np.float32(288.0) - np.float32(0.0065) * z, np.float32(210.0))
    p = np.float32(100000.0) * (t / np.float32(288.0)) ** np.float32(5.26)
    pii = (p / np.float32(1.0e5)) ** np.float32(0.2857)
    f = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.float32))
    zeros = lambda: cp.zeros((nz, ny, nx), cp.float32)
    plane = lambda: cp.zeros((ny, nx), cp.float32)
    theta = f(t / pii)
    lin.launch_lin(theta, f(np.full_like(z, 1.0e-3)), zeros(), zeros(), zeros(), zeros(), zeros(),
                   f(p / (np.float32(287.0) * t)), f(pii), f(p), f(z), f(np.full_like(z, 30000.0 / nz)),
                   plane(), plane(), plane(), plane(), plane(), plane(), plane(), plane(), dt=30.0)
    assert bool(cp.isfinite(theta).all())
