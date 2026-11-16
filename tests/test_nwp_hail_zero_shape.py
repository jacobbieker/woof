"""nwp_hail's zero-shape skip keeps the words of the pow it skips (#369).

kernels/nwp_hail.cu evaluates ``pow(d, mu)`` per diameter bin only when the
hail PSD shape ``mu`` is non-zero (``pow(d, 0)`` is exactly 1).  The
reference is the same unit with the skip turned off, built by the loader's
own route, over every PSD branch the leaf takes.  The product the skip
removes feeds a running sum, so the check runs in fast numerics as well as
strict (each in a child process): a contraction there would differ by mode.
"""
import numpy as np
import pytest

import _numerics_child
from conftest import requires_gpu

pytestmark = pytest.mark.gpu

SKIP = "const bool shaped=mu!=0.f;"


def _inputs(nz=9, ny=14, nx=17):
    rng = np.random.default_rng(7)
    z = {}
    for key in ("qg", "qh", "qr"):
        a = rng.random((nz, ny, nx), dtype=np.float32) * np.float32(4e-3)
        a[rng.random(a.shape) < 0.5] = 0
        z[key] = a
    for key in ("ng", "nh", "nr"):
        z[key] = rng.random((nz, ny, nx), dtype=np.float32) * np.float32(5e4) + np.float32(1.0)
    z["rho"] = np.float32(0.6) + rng.random((nz, ny, nx), dtype=np.float32) * np.float32(0.6)
    z["vol"] = z["qg"] / np.float32(500.0) * (np.float32(0.5) + rng.random((nz, ny, nx), dtype=np.float32))
    return z


def _words():
    import cupy as cp

    from woof.core import nwp_hail

    d = {k: cp.asarray(v) for k, v in _inputs().items()}
    nssl = nwp_hail.NsslSingleMoment
    cases = {
        "mp2": dict(mp_physics=2), "mp6": dict(mp_physics=6), "mp16": dict(mp_physics=16),
        "mp16h": dict(mp_physics=16, mpuse_hail=1), "mp97": dict(mp_physics=97),
        "mp8": dict(mp_physics=8, qr=d["qr"], nr=d["nr"]), "mp8ng": dict(mp_physics=8, ng=d["ng"]),
        "mp28": dict(mp_physics=28, qr=d["qr"], nr=d["nr"]),
        "mp38": dict(mp_physics=38, ng=d["ng"], volume=d["vol"]),
        "mp10": dict(mp_physics=10, ng=d["ng"]), "mp40": dict(mp_physics=40, ng=d["ng"], mpuse_hail=1),
        "mp9": dict(mp_physics=9, qh=d["qh"], nh=d["nh"]),
        "mp18a0": dict(mp_physics=18, qh=d["qh"],
                       nssl_single_moment=nssl(4e4, 4e4, 900., 900., 0., 0., 0)),
        "mp18a2": dict(mp_physics=18, qh=d["qh"],
                       nssl_single_moment=nssl(4e4, 4e4, 900., 900., 2., 1., 1)),
    }
    ny, nx = d["qg"].shape[1:]
    out = {}
    for name, kw in cases.items():
        # Poisoned outputs: the leaf takes a running maximum, so start every
        # word from a value the kernel would keep only if it never wrote.
        h1 = cp.full((ny, nx), -1.0, cp.float32)
        h2 = cp.full((ny, nx), -1.0, cp.float32)
        nwp_hail.update(d["qg"], d["rho"], h1, h2, **kw)
        out[name] = (cp.asnumpy(h1).view(np.uint32), cp.asnumpy(h2).view(np.uint32))
    return out


@requires_gpu
def test_zero_shape_skip_keeps_the_pow_words(monkeypatch):
    import woof.core.kernels as kernels

    real = kernels.module_source
    assert SKIP in real("nwp_hail")
    kernels.get_kernel.cache_clear()
    kernels.load_module.cache_clear()
    got = _words()

    def without_skip(name, **kw):
        source = real(name, **kw)
        return source.replace(SKIP, "const bool shaped=true;") if name == "nwp_hail" else source

    monkeypatch.setattr(kernels, "module_source", without_skip)
    kernels.get_kernel.cache_clear()
    kernels.load_module.cache_clear()
    try:
        want = _words()
    finally:
        monkeypatch.undo()
        kernels.get_kernel.cache_clear()
        kernels.load_module.cache_clear()
    hail = 0
    for name in want:
        for a, b in zip(got[name], want[name]):
            assert np.array_equal(a, b), (name, int((a != b).sum()))
            hail += int((a.view(np.float32) > 0).sum())
    assert hail > 0          # not vacuous: the bins produced hail sizes


@requires_gpu
@pytest.mark.parametrize(*_numerics_child.MODES)
def test_identity_holds_under_each_numerics(numerics, exact):
    """The battery runs strict; production runs fast.  Each mode in its own
    child process (tests/_numerics_child.py)."""
    _numerics_child.run(__file__, ["test_zero_shape_skip_keeps_the_pow_words"], numerics, exact)
