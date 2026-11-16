"""zero_first: the per-species ``tend[...] = 0`` folded into the first
accumulating advection launch (kernels/advection.cu flux_div_scalar,
kernels/pd_advection.cu pd_renorm_apply).

With GPUWM_*_ZERO_FIRST the kernel starts every point from +0.0f, the word
the fill wrote, and adds its terms onto it with the same statements, so the
buffer it leaves must equal a fill followed by the accumulating launch,
whatever it held before.  Checked over the open/specified, map-factor,
vertical-order and horizontal-order variants the forecast launches.
"""
import numpy as np
import pytest

import _numerics_child
from conftest import requires_gpu
from test_pd_advection import _pd_case


def _garbage(rng, shape):
    return (rng.standard_normal(shape) * 1e3).astype(np.float32)


@requires_gpu
@pytest.mark.parametrize("open_, spec, msf, vorder, horder", [
    (False, False, False, 3, 5),
    (True, True, True, 3, 5),
    (True, True, False, 5, 5),
    (True, False, True, 3, 3),
    (False, False, True, 5, 0),
    (True, True, True, 3, 0),
])
def test_flux_div_scalar_zero_first_equals_fill_then_accumulate(
        open_, spec, msf, vorder, horder):
    import cupy as cp
    from woof.core.advection import launch_flux_div_scalar

    vc, q, q0, ru, rv, rw, mut, mu_old, dx, dy, dt = _pd_case(seed=5, ny=9, nx=13)
    nz, ny, nx = q.shape
    rng = np.random.default_rng(3)
    msft = (1.0 + 0.01 * rng.standard_normal((ny, nx))).astype(np.float32)
    kw = dict(open_x=open_, open_y=open_, msf=cp.asarray(msft) if msf else None,
              spec=spec, vorder=vorder, horder=horder,
              theta_offset=(horder == 0))
    dev = [cp.asarray(a) for a in (q, ru, rv, rw)]
    filled = cp.asarray(_garbage(rng, q.shape))
    filled[...] = 0
    launch_flux_div_scalar(*dev, filled, vc, dx, dy, **kw)
    fused = cp.asarray(_garbage(rng, q.shape))
    launch_flux_div_scalar(*dev, fused, vc, dx, dy, zero_first=True, **kw)
    a, b = cp.asnumpy(filled), cp.asnumpy(fused)
    assert np.abs(a).max() > 0.0
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


@requires_gpu
@pytest.mark.parametrize("open_, msf", [
    (False, False), (False, True), (True, True), (True, False),
])
def test_pd_renorm_apply_zero_first_equals_fill_then_accumulate(open_, msf):
    import cupy as cp
    from woof.core.moist import launch_pd_fluxes, launch_pd_renorm_apply
    from woof.wrf_exact import ADVECTION_ENABLED

    if ADVECTION_ENABLED:
        # The strict renorm kernel reads the state's mass carriers, which
        # this bare vertical coordinate does not carry
        # (tests/test_pd_advection.py has the same limit).
        pytest.skip("strict pd_renorm_apply needs a domain state")

    vc, q, q0, ru, rv, rw, mut, mu_old, dx, dy, dt = _pd_case(seed=11, ny=9, nx=13)
    nz, ny, nx = q.shape
    rng = np.random.default_rng(4)
    msft = (1.0 + 0.01 * rng.standard_normal((ny, nx))).astype(np.float32)
    msft_d = cp.asarray(msft) if msf else None
    bufs = (cp.zeros((nz, ny, nx + 1), cp.float32),
            cp.zeros((nz, ny, nx + 1), cp.float32),
            cp.zeros((nz, ny + 1, nx), cp.float32),
            cp.zeros((nz, ny + 1, nx), cp.float32),
            cp.zeros((nz + 1, ny, nx), cp.float32),
            cp.zeros((nz + 1, ny, nx), cp.float32))
    launch_pd_fluxes(cp.asarray(q), cp.asarray(q0), cp.asarray(ru),
                     cp.asarray(rv), cp.asarray(rw), cp.asarray(mut), vc,
                     dx, dy, dt, *bufs, msft=msft_d, has_msf=msf,
                     open_x=open_, open_y=open_)
    kw = dict(coord=vc, dx=dx, dy=dy, dt=dt, msft=msft_d, has_msf=msf,
              open_x=open_, open_y=open_)
    filled = cp.asarray(_garbage(rng, q.shape))
    filled[...] = 0
    launch_pd_renorm_apply(cp.asarray(q0), cp.asarray(mu_old), *bufs,
                           tend=filled, **kw)
    fused = cp.asarray(_garbage(rng, q.shape))
    launch_pd_renorm_apply(cp.asarray(q0), cp.asarray(mu_old), *bufs,
                           tend=fused, zero_first=True, **kw)
    a, b = cp.asnumpy(filled), cp.asnumpy(fused)
    assert np.abs(a).max() > 0.0
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


def test_zero_first_is_a_scalar_option():
    pytest.importorskip("cupy")
    from woof.core.advection import _launch
    with pytest.raises(ValueError, match="zero_first"):
        _launch("flux_div_u", None, None, None, None, None, None, None,
                None, 1.0, 1.0, 1, 1, 1, 1, 1, 1, zero_first=True)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize(*_numerics_child.MODES)
def test_identity_holds_under_each_numerics(numerics, exact):
    """zero_first against a fill followed by the accumulating launch.
    The battery runs strict; production runs fast, where NVRTC drops
    --fmad=false and Metal keeps fp contract(fast).  Each mode in its own
    child process (tests/_numerics_child.py)."""
    _numerics_child.run(__file__, ['test_flux_div_scalar_zero_first_equals_fill_then_accumulate', 'test_pd_renorm_apply_zero_first_equals_fill_then_accumulate'], numerics, exact)
