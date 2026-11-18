"""Pool-direct publication: producers that write every word write the resident pool field itself.

``prepare_mpas_transport_source_baseline_v842(out=...)`` gives the words of its
own allocation in a caller's (poisoned) buffer, and
``NativeDevicePoolViews.resident_target`` answers only for a whole-array copy.
The forecast-level evidence (Alps hex VR, fast and strict, CUDA and Metal) is
in the PR.
"""

import numpy as np
import pytest


@pytest.fixture
def cp():
    return pytest.importorskip("cupy")


@pytest.mark.parametrize("dt", [90.0, 0.125])
def test_baseline_into_a_poisoned_out_is_word_identical(cp, dt):
    from woof.hex.cuda_backend.runtime import KernelCache
    from woof.hex.cuda_physics_v841 import prepare_mpas_transport_source_baseline_v842
    rng = np.random.default_rng(7)
    ns, nl, nc = 3, 5, 11
    before = cp.asarray(rng.uniform(0, 1e-2, (ns, nl, nc)).astype(np.float32))
    source = cp.asarray(rng.uniform(-1e-5, 1e-5, (ns, nl, nc)).astype(np.float32))
    rho = cp.asarray(rng.uniform(0.5, 1.2, (nl, nc)).astype(np.float32))
    cache = KernelCache()
    args = dict(before=before, source=source, rho_old=rho, dt=dt, n_cells_solve=nc - 2, kernel_cache=cache)
    want = prepare_mpas_transport_source_baseline_v842(**args).get()
    for byte in (0xFF, 0xA5):
        out = cp.asarray(np.full(ns * nl * nc * 4, byte, np.uint8).view(np.float32).reshape(ns, nl, nc))
        got = prepare_mpas_transport_source_baseline_v842(**args, out=out)
        assert got is out and got.get().tobytes() == want.tobytes()
    with pytest.raises(ValueError):
        prepare_mpas_transport_source_baseline_v842(**args, out=source)
    with pytest.raises(ValueError):
        prepare_mpas_transport_source_baseline_v842(**args, out=cp.zeros((ns, nl, nc + 1), cp.float32))


def test_resident_target_answers_only_for_a_whole_array_copy(cp):
    from types import SimpleNamespace
    from woof.hex.mpasphys.device_pools_v842 import NativeDevicePoolViews
    fields = {("tend_physics", "rqvdynten", 1): cp.zeros((5, 11), cp.float32),
              ("state", "scalars", 2): cp.zeros((3, 5, 11), cp.float32),
              ("diag", "rho", 1): cp.zeros((5, 10), cp.float32)}
    views = NativeDevicePoolViews.__new__(NativeDevicePoolViews)
    views.resident, views.level = True, 1
    views.field = lambda pool, name: fields[pool, name, views.level]
    like = cp.ones((5, 11), cp.float32)
    assert views.resident_target("tend_physics", "rqvdynten", like) is fields["tend_physics", "rqvdynten", 1]
    assert views.resident_target("state", "scalars", cp.ones((3, 5, 11), cp.float32), level=2) \
        is fields["state", "scalars", 2] and views.level == 1
    assert views.resident_target("diag", "rho", like) is None                 # the garbage-column slice copy
    assert views.resident_target("tend_physics", "rqvdynten", like.astype(cp.int32)) is None
    views.resident = False
    assert views.resident_target("tend_physics", "rqvdynten", like) is None
