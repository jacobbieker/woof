"""The front door and doctor on the Apple-silicon backend (woof/backend).

With ``WOOF_BACKEND=mlx`` the GPU runtime a forecast needs is MLX, not
CuPy: a door must refuse with the MLX remedy when mlx is absent, and must
not refuse for a missing CuPy wheel the backend never imports.
"""

import pytest

from woof import capabilities, doctor


def test_the_mlx_backend_needs_mlx_not_cupy(monkeypatch):
    monkeypatch.setenv("WOOF_BACKEND", "mlx")
    monkeypatch.setattr(capabilities, "is_installed", lambda module: module != "cupy")
    assert capabilities.missing((capabilities.GPU_RUNTIME,)) == ()
    capabilities.require("woof run", capabilities.GPU_RUNTIME)


def test_the_mlx_backend_without_mlx_refuses_with_the_mlx_remedy(monkeypatch):
    monkeypatch.setenv("WOOF_BACKEND", "mlx")
    monkeypatch.setattr(capabilities, "is_installed", lambda module: module != "mlx")
    with pytest.raises(capabilities.CapabilityMissing) as refused:
        capabilities.require("woof run", capabilities.GPU_RUNTIME)
    assert refused.value.requirement is capabilities.MLX_RUNTIME
    assert "pixi install -e mlx" in str(refused.value)
    assert "gpu-cu12" not in str(refused.value)


def test_the_cuda_backend_still_needs_cupy(monkeypatch):
    monkeypatch.setenv("WOOF_BACKEND", "cuda")
    monkeypatch.setattr(capabilities, "is_installed", lambda module: module != "cupy")
    assert capabilities.missing((capabilities.GPU_RUNTIME,)) == (capabilities.GPU_RUNTIME,)


def test_doctor_runs_the_metal_probe_instead_of_cupy_on_mlx(monkeypatch):
    monkeypatch.setenv("WOOF_BACKEND", "mlx")
    probe = doctor.Check("mlx (Apple GPU runtime)", "verified", "probe", brief="ok")
    monkeypatch.setattr(doctor, "_metal_runtime_check", lambda: probe)
    monkeypatch.setattr(doctor, "_cupy_check", lambda: pytest.fail("cupy probed on mlx"))
    monkeypatch.setattr(doctor, "_cuda_headers_check", lambda **k: pytest.fail("NVRTC probed on mlx"))
    names = [c.name for c in doctor.collect_checks()]
    assert "mlx (Apple GPU runtime)" in names


@pytest.mark.skipif(not capabilities.is_installed("mlx"), reason="needs mlx (Apple silicon)")
def test_the_metal_probe_verifies_on_this_mac():
    check = doctor._metal_runtime_check()
    assert check.status == "verified", check.detail
