"""Pool-direct publication: producers that write every word write the resident pool field itself.

``prepare_mpas_transport_source_baseline_v842(out=...)`` gives the words of its
own allocation in a caller's (poisoned) buffer, and
``NativeDevicePoolViews.resident_target`` answers only for a whole-array copy,
whose ``write_field`` is then a no-op.  On by default; WOOF_HEX_POOL_DIRECT=0 restores the copy.
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
    for overlapping in (source, before):
        with pytest.raises(ValueError):
            prepare_mpas_transport_source_baseline_v842(**args, out=overlapping)
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


def test_write_field_of_the_target_itself_is_a_no_op(cp):
    from types import SimpleNamespace
    from woof.hex.mpasphys.device_pools_v842 import NativeDevicePoolViews
    target = cp.zeros((5, 11), cp.float32)
    views = NativeDevicePoolViews.__new__(NativeDevicePoolViews)
    views.resident, views.level, views.cp = True, 1, cp
    views.session = SimpleNamespace(layout=SimpleNamespace())
    views.field = lambda pool, name: target
    target[...] = 3.0
    views.write_field("tend_physics", "rqvdynten", target)          # in place: nothing to copy
    assert float(target.sum()) == 3.0 * target.size
    views.write_field("tend_physics", "rqvdynten", cp.ones((5, 11), cp.float32))
    assert float(target.sum()) == target.size
    with pytest.raises(ValueError):
        views.write_field("tend_physics", "rqvdynten", cp.ones((4, 11), cp.float32))


def test_pool_direct_switch(monkeypatch):
    from types import SimpleNamespace
    from woof.hex.mpasphys import runtime_v842
    asked = []
    views = SimpleNamespace(resident=True, resident_target=lambda *a, **k: asked.append(a) or "target")
    coupling = SimpleNamespace(session=SimpleNamespace(device_pools=views))
    pool_target = runtime_v842.ResidentNativeCoupling._pool_target
    monkeypatch.setattr(runtime_v842, "_POOL_DIRECT", False)
    assert pool_target(coupling, "tend_physics", "rqvdynten", None) is None and not asked
    monkeypatch.setattr(runtime_v842, "_POOL_DIRECT", True)
    assert pool_target(coupling, "tend_physics", "rqvdynten", None) == "target"
    views.resident = False
    assert pool_target(coupling, "tend_physics", "rqvdynten", None) is None


def test_pool_direct_is_on_by_default():
    """Unset or empty is on; 0/false/no/off turns it off (read at import, so in a child)."""
    import os
    import subprocess
    import sys
    code = "from woof.hex.mpasphys import runtime_v842 as r; print(r._POOL_DIRECT)"
    for value, want in ((None, "True"), ("", "True"), ("1", "True"), ("0", "False"), ("off", "False")):
        env = {k: v for k, v in os.environ.items() if k != "WOOF_HEX_POOL_DIRECT"}
        if value is not None:
            env["WOOF_HEX_POOL_DIRECT"] = value
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
        assert out.returncode == 0, out.stderr[-2000:]
        assert out.stdout.strip().splitlines()[-1] == want, (value, out.stdout)
