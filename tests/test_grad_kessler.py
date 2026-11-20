"""woof.grad.kessler: Kessler as differentiable PyTorch, and the gradient kit (plan P1.2, P1.3).

On the WRF Kessler oracle fixture (104 columns x 50 levels, 13 regimes):

* the float64 module reproduces woof's NumPy mirror of the kernel
  (``np_kessler_column``) to rounding, every dt;
* the adjoint test <TL dx, dy> = <dx, AD dy> holds to 1e-12 in float64 and
  1e-5 in float32 (gate G1), with the tangent-linear from forward-mode AD;
* gradcheck passes on smooth columns and fails, as it must, on the
  autoconversion-threshold regime, whose qc sits on the kink;
* on a GPU backend, the float32 module on the device agrees with woof's
  own kernel to a few ULP in theta and to rounding-level absolute
  differences in the mixing ratios;
* the seat model scripts with TorchScript and matches its eager self.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from woof.verify import kessler_oracle as ko

torch = None


def _need_torch():
    global torch
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed (recast-woof[nn-torch])")
    import torch as _t
    torch = _t


def _x(inputs, dtype=None, device="cpu", columns=None):
    dtype = dtype or torch.float64
    sl = slice(None) if columns is None else columns
    return [torch.from_numpy(np.ascontiguousarray(inputs[k][:, sl], np.float64)).to(dtype).to(device)
            for k in ko.INPUT_FIELDS]


def test_literals_are_woofs_constants():
    _need_torch()
    from woof.core import constants as c
    from woof.grad import kessler
    from pathlib import Path
    src = Path(kessler.__file__).read_text()
    assert repr(c.RD / c.RV) in src and repr(c.RD / c.CP) in src
    assert repr(c.SVP2 * (c.SVPT0 - c.SVP3) * c.XLV / c.CP) in src
    for value in (c.SVP1, c.SVP2, c.SVP3, c.SVPT0, c.RHOWATER, c.RD, c.RV):
        assert repr(float(value)) in src, value
    assert (kessler.EP2, kessler.F5, kessler.RCP) == (c.RD / c.RV, c.SVP2 * (c.SVPT0 - c.SVP3) * c.XLV / c.CP,
                                                      c.RD / c.CP)


@pytest.mark.parametrize("case", range(5))
def test_float64_module_is_the_kernels_numpy_mirror(case):
    _need_torch()
    from woof.grad.kessler import _kessler
    from woof.verify.npref import np_kessler_column
    inputs, _ = ko.load_fixture()
    dt = float(inputs["dt"][case])
    with torch.no_grad():
        out = _kessler(*_x(inputs), dt)
    ncol = inputs["t"].shape[1]
    ref = [np_kessler_column(*[inputs[k][:, c] for k in ko.INPUT_FIELDS], dt) for c in range(ncol)]
    for i, name in enumerate(("t", "qv", "qc", "qr")):
        want = np.stack([r[i] for r in ref], axis=1)
        np.testing.assert_allclose(out[i].numpy(), want, rtol=1e-12, atol=1e-15, err_msg=name)
    np.testing.assert_allclose(out[4].numpy(), [r[5] for r in ref], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(out[5].numpy(), [r[4] for r in ref], rtol=1e-12, atol=1e-15)


def test_the_gradient_at_a_min_tie_is_one_sided_not_halved():
    """qr == 0 in subsaturated air: every branch of the rain-evaporation
    min is 0, and torch.minimum would give each half the gradient."""
    _need_torch()
    from woof.grad.kessler import _kessler
    inputs, _ = ko.load_fixture()
    x = _x(inputs, columns=slice(0, 1))
    x[1] = torch.full_like(x[1], 1.0e-5)   # qv: far below saturation
    x[2] = torch.zeros_like(x[2])          # qc
    x[3] = torch.zeros_like(x[3])          # qr
    dt = float(inputs["dt"][2])
    qr = x[3].clone().requires_grad_(True)
    out = _kessler(x[0], x[1], x[2], qr, *x[4:], dt)
    g, = torch.autograd.grad(out[3].sum(), qr)
    h = 1.0e-12
    with torch.no_grad():
        one_sided = (_kessler(x[0], x[1], x[2], x[3] + h, *x[4:], dt)[3].sum()
                     - _kessler(*x, dt)[3].sum()) / h
    assert abs(float(g.sum()) - float(one_sided)) <= 1e-6 * max(1.0, abs(float(one_sided)))


@pytest.mark.parametrize("dtype,bound", [("float64", 1e-12), ("float32", 1e-5)])
def test_dot_product(dtype, bound):
    _need_torch()
    from woof.grad import checks
    from woof.grad.kessler import _kessler
    inputs, _ = ko.load_fixture()
    dt = 60.0
    dtype = getattr(torch, dtype)
    x = _x(inputs, dtype)
    g = torch.Generator().manual_seed(0)
    scale = {"t": 1.0, "qv": 1e-3, "qc": 1e-4, "qr": 1e-4}
    dx = [(torch.randn(a.shape, generator=g, dtype=torch.float64)
           * (scale[k] * (a != 0) if k in scale else 1e-3 * float(a.abs().mean()))).to(dtype)
          for k, a in zip(ko.INPUT_FIELDS, x)]
    dy = [(torch.randn(x[0].shape, generator=g, dtype=torch.float64) / scale[k]).to(dtype)
          for k in ("t", "qv", "qc", "qr")]
    dy += [torch.randn(x[0].shape[1], generator=g, dtype=torch.float64).to(dtype) for _ in range(2)]
    r = checks.dot_product(lambda *a: _kessler(*a, dt), x, dx, dy)
    assert r.relative < bound, r


def test_tl_converges_and_taylor_holds():
    _need_torch()
    from woof.grad import checks
    from woof.grad.kessler import _kessler
    inputs, _ = ko.load_fixture()
    dt = 18.0
    x = _x(inputs)
    g = torch.Generator().manual_seed(3)
    # theta in K, each q relative to itself (a zero word stays on its branch)
    dx = [torch.randn(x[0].shape, generator=g, dtype=torch.float64) * 0.1]
    dx += [torch.randn(x[0].shape, generator=g, dtype=torch.float64) * 0.01 * a.abs() for a in x[1:4]]
    dx += [torch.zeros_like(a) for a in x[4:]]
    rows = dict(checks.tl_convergence(lambda *a: _kessler(*a, dt), x, dx, [1e-4, 1e-5, 1e-6]))
    assert abs(rows[1e-6] - 1) < 1e-4, rows

    def J(*a):
        out = _kessler(*a, dt)
        return (out[0] ** 2).sum() + 1e6 * (out[3] ** 2).sum() + out[5].sum()

    taylor = dict(checks.taylor(J, x, dx, [1e-3, 1e-4, 1e-5]))
    assert abs(taylor[1e-5] - 1) < 1e-4, taylor


def test_gradcheck_passes_on_smooth_columns_and_not_on_the_kink():
    _need_torch()
    from woof.grad.kessler import _kessler
    inputs, _ = ko.load_fixture()
    regimes = np.asarray(inputs["regime"]).astype(str)
    dt = 60.0

    def check(col):
        cols = _x(inputs, columns=slice(col, col + 1))
        t0, qv0, qc0, qr0 = cols[:4]

        def f(ut, uqv, uqc, uqr):
            out = _kessler(t0 + ut, qv0 * (1 + uqv), qc0 * (1 + uqc), qr0 * (1 + uqr), *cols[4:], dt)
            return (out[0] - t0, (out[1] - qv0) * 1e3, (out[2] - qc0) * 1e3, (out[3] - qr0) * 1e3,
                    out[4], out[5])

        state = tuple(torch.zeros_like(t0).requires_grad_(True) for _ in range(4))
        return torch.autograd.gradcheck(f, state, eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=False)

    for regime in ("convective", "stratiform", "heavy_rain_thin", "desert_virga", "clear_dry"):
        col = int(np.flatnonzero(regimes == regime)[0])
        assert check(col), regime
    # qc == c2 = 0.001 exactly: max(qc - c2, 0) has its kink at the point
    assert not check(int(np.flatnonzero(regimes == "autoconv_threshold")[0]))


def test_the_seat_model_scripts_and_matches_the_columns():
    _need_torch()
    from woof.grad.kessler import KesslerSeatModel, kessler_columns
    inputs, _ = ko.load_fixture()
    L = 50
    dt = 60.0
    t, qv, qc, qr, rho, pii, z, dz = _x(inputs, torch.float32)
    p = 1.0e5 * pii.double() ** (1004.5 / 287.0)
    features = torch.cat([t, qv, qc, qr, p.float(), z, dz], dim=0).T.contiguous()
    model = KesslerSeatModel(dt, L)
    eager = model(features)
    scripted = torch.jit.script(model)(features)
    torch.testing.assert_close(scripted, eager, rtol=0, atol=0)
    # the rates are those of the columns with the seat's own Exner and density
    pii_s = (p.float() / 1.0e5) ** (287.0 / 1004.5)
    rho_s = p.float() / (287.0 * t * pii_s * (1.0 + qv * (461.6 / 287.0)))
    t1, qv1, qc1, qr1, _ = kessler_columns(t, qv, qc, qr, rho_s, pii_s, z, dz, dt)
    want = (torch.cat([t1 - t, qv1 - qv, qc1 - qc, qr1 - qr], dim=0) / dt).T
    torch.testing.assert_close(eager, want, rtol=1e-6, atol=1e-9)
    assert eager.shape == (inputs["t"].shape[1], 4 * L)


def _device_or_skip():
    _need_torch()
    from woof import backend
    name = backend.requested()
    if name == "mlx" and torch.backends.mps.is_available():
        return "mps"
    if name == "cuda" and torch.cuda.is_available():
        from conftest import HAS_GPU
        if HAS_GPU:
            return "cuda"
    pytest.skip("needs a CUDA card or the Metal backend, with torch on that device")


@pytest.mark.gpu
@pytest.mark.parametrize("case", range(5))
def test_float32_module_on_the_device_matches_woofs_kernel(case):
    device = _device_or_skip()
    import cupy
    from woof.core.fp32_ulp import fp32_ulp_distance
    from woof.core.microphysics import launch_kessler
    from woof.grad.kessler import _kessler
    inputs, _ = ko.load_fixture()
    dt = float(inputs["dt"][case])
    nz, ncol = inputs["t"].shape
    dev = {k: cupy.asarray(np.ascontiguousarray(inputs[k], np.float32).reshape(nz, 1, ncol))
           for k in ko.INPUT_FIELDS}
    rainnc, rainncv = cupy.zeros((1, ncol), cupy.float32), cupy.zeros((1, ncol), cupy.float32)
    launch_kessler(dev["t"], dev["qv"], dev["qc"], dev["qr"], dev["rho"], dev["pii"], dev["z"], dev["dz8w"],
                   rainnc, rainncv, dt)
    with torch.no_grad():
        out = [o.cpu().numpy() for o in _kessler(*_x(inputs, torch.float32, device), dt)]
    kern = {k: dev[k].get().reshape(nz, ncol) for k in ("t", "qv", "qc", "qr")}
    assert fp32_ulp_distance(out[0], kern["t"]).max() <= 2
    for i, name in enumerate(("qv", "qc", "qr"), start=1):
        np.testing.assert_allclose(out[i], kern[name], rtol=0, atol=2e-8, err_msg=name)
    for got, want in ((out[4], rainncv.get().ravel()), (out[5], rainnc.get().ravel())):
        np.testing.assert_allclose(got, want, rtol=2e-5, atol=1e-6)


@pytest.mark.gpu
def test_compiled_stages_match_the_eager_module_on_cuda():
    device = _device_or_skip()
    if device != "cuda" or importlib.util.find_spec("triton") is None:
        pytest.skip("torch.compile's Inductor needs Triton on a CUDA card here")
    from woof.grad.kessler import _kessler, kessler_compiled
    inputs, _ = ko.load_fixture()
    x = _x(inputs, torch.float32, device)
    # Triton flushes float32 denormals to zero, so a denormal q (the
    # fixture's edge_values regime carries qr = 1e-40) is "zero" there and
    # on its x > 0 branch in eager torch; flush them for the comparison
    tiny = torch.finfo(torch.float32).tiny
    x = [torch.where(a.abs() < tiny, torch.zeros_like(a), a) for a in x]
    leaves = [a.clone().requires_grad_(True) for a in x[:4]]
    eager = _kessler(*leaves, *x[4:], 60.0)
    g_eager = torch.autograd.grad(eager[:4], leaves, [torch.ones_like(a) for a in x[:4]])
    leaves_c = [a.clone().requires_grad_(True) for a in x[:4]]
    comp = kessler_compiled(*leaves_c, *x[4:], 60.0)
    g_comp = torch.autograd.grad(comp[:4], leaves_c, [torch.ones_like(a) for a in x[:4]])
    for a, b in zip(comp, eager):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-8)
    # Inductor fuses and reassociates the float32 backward: a few words move
    # by ~1e-3 relative, the gradient as a whole by far less
    for a, b in zip(g_comp, g_eager):
        assert float((a - b).norm() / b.norm()) < 1e-4
    from woof.grad import checks
    g = torch.Generator(device="cpu").manual_seed(0)
    dx = [(torch.randn(a.shape, generator=g) * s * (a != 0).cpu()).to(device)
          for a, s in zip(x[:4], (1.0, 1e-3, 1e-4, 1e-4))] + [torch.zeros_like(a) for a in x[4:]]
    dy = [torch.randn(x[0].shape, generator=g).to(device) / s for s in (1.0, 1e-3, 1e-4, 1e-4)]
    dy += [torch.randn(x[0].shape[1], generator=g).to(device) for _ in range(2)]
    # forward-mode AD does not pass through torch.compile, so the
    # tangent-linear is the eager module's and the adjoint the compiled one's
    _, tl = checks.tangent_linear(lambda *a: _kessler(*a, 60.0), x, dx)
    _, ad = checks.adjoint(lambda *a: kessler_compiled(*a, 60.0), x, dy)
    r = checks.DotProduct(float(checks._inner(tl, dy)), float(checks._inner(dx, ad)))
    assert r.relative < 1e-5, r
