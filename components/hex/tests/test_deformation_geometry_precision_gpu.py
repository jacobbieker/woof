"""The ``smagorinsky_v841_f32`` kernel fed native vs local64 deformation weights.

Coarse mesh: ``auto`` hands the kernel the native bytes, so its kdiff is
bit-identical to before; the local64 weights move it only at the native
evaluation's own error level.  Fine 50 m patch: kdiff from local64 weights
lands on the binary64 CPU authority where native weights were percent-level
off.  The kernel text is the shipped one, compiled from
``woof.hex.cuda_horizontal_v841``.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.hex.mixing_v841 import (
    DeformationWeightsV841,
    V841MixingConfig,
    compute_smagorinsky_coefficients_v841,
    initialize_deformation_weights_v841,
)
from woof.hex.precision_probe import synthetic_hex_patch

# The cupy import lives in a helper, which the AST auto-marker does not
# attribute to the tests that call it, so the tier is declared here.
pytestmark = pytest.mark.gpu

NLEV = 3
DT = 0.25


def _velocities(patch, seed=11):
    rng = np.random.default_rng(seed)
    nedges = patch.arrays["cellsOnEdge"].shape[0]
    u = rng.uniform(-1.0, 1.0, size=(NLEV, nedges))
    v = rng.uniform(-1.0, 1.0, size=(NLEV, nedges))
    return u, v


def _gpu_kdiff(patch, weights, u, v, length):
    import cupy as cp

    from woof.hex.cuda_backend import KernelCache
    from woof.hex.cuda_backend.runtime import CudaRefusal
    from woof.hex.cuda_horizontal_v841 import _CUDA_SOURCE

    free, _ = cp.cuda.Device().mem_info
    if free < 512 * 1024 * 1024:
        pytest.skip("shared GPU has under 512 MiB free")
    try:
        kernel = KernelCache().raw_kernel(
            "smagorinsky_v841_f32", _CUDA_SOURCE, module_key="hexcore.cuda_horizontal_v841"
        )
    except CudaRefusal:
        # A card or runtime below the production floor: compile the same
        # text with the cache's own base options.  This test measures the
        # kernel's arithmetic, not production admission.
        kernel = cp.RawModule(
            code=_CUDA_SOURCE, options=("--std=c++17", "--fmad=false")
        ).get_function("smagorinsky_v841_f32")
    arrays = patch.arrays
    ncells, max_edges = arrays["edgesOnCell"].shape
    nedges = arrays["cellsOnEdge"].shape[0]
    length32 = np.float32(length)
    cs = np.float32(0.125)
    strain_scale = np.float32((cs * length32) * (cs * length32))
    inv_dt = np.float32(np.float32(1.0) / np.float32(DT))
    ceiling = np.float32((np.float32(0.01) * (length32 * length32)) * inv_dt)
    kdiff = cp.zeros((NLEV, ncells), dtype=cp.float32)
    total = NLEV * ncells
    kernel(
        ((total + 127) // 128,),
        (128,),
        (
            cp.asarray(u, dtype=cp.float32),
            cp.asarray(v, dtype=cp.float32),
            cp.asarray(arrays["edgesOnCell"].astype(np.int32)),
            cp.asarray(arrays["nEdgesOnCell"].astype(np.int32)),
            cp.asarray(np.ascontiguousarray(weights.coef_c2, dtype=np.float32)),
            cp.asarray(np.ascontiguousarray(weights.coef_s2, dtype=np.float32)),
            cp.asarray(np.ascontiguousarray(weights.coef_cs, dtype=np.float32)),
            strain_scale,
            ceiling,
            np.int32(NLEV),
            np.int32(ncells),
            np.int32(nedges),
            np.int32(max_edges),
            kdiff,
        ),
    )
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(kdiff).astype(np.float64)


def _cpu_reference(patch, u, v, length):
    weights = initialize_deformation_weights_v841(patch, dtype=np.float64, geometry="native")
    config = V841MixingConfig(config_len_disp=length)
    return compute_smagorinsky_coefficients_v841(
        patch, u, v, weights, dt=DT, config=config
    ).kdiff


def _rel(actual, reference, rows):
    a = actual[:, rows]
    r = reference[:, rows]
    return float(np.max(np.abs(a - r)) / np.max(np.abs(r)))


def test_coarse_mesh_kdiff_is_unchanged_by_the_auto_gate():
    patch = synthetic_hex_patch(15000.0, rings=3, lat_deg=40.0, lon_deg=-97.0)
    u, v = _velocities(patch)
    auto = initialize_deformation_weights_v841(patch)
    native = initialize_deformation_weights_v841(patch, geometry="native")
    local = initialize_deformation_weights_v841(patch, geometry="local64")
    assert auto.geometry == "native"
    k_auto = _gpu_kdiff(patch, auto, u, v, 15000.0)
    k_native = _gpu_kdiff(patch, native, u, v, 15000.0)
    k_local = _gpu_kdiff(patch, local, u, v, 15000.0)
    assert k_auto.tobytes() == k_native.tobytes()
    rows = patch.interior_cells
    # local64 differs from native only by the native evaluation's own
    # binary32 error at 15 km (~1e-3 of the weights).
    assert _rel(k_local, k_native, rows) < 5.0e-3


@pytest.mark.parametrize("spacing", [50.0, 100.0])
def test_fine_patch_kdiff_lands_on_the_binary64_authority_with_local64(spacing):
    patch = synthetic_hex_patch(spacing, rings=3)
    u, v = _velocities(patch)
    reference = _cpu_reference(patch, u, v, spacing)
    native = initialize_deformation_weights_v841(patch, geometry="native")
    auto = initialize_deformation_weights_v841(patch)
    assert auto.geometry == "local64"
    rows = patch.interior_cells
    err_native = _rel(_gpu_kdiff(patch, native, u, v, spacing), reference, rows)
    err_local = _rel(_gpu_kdiff(patch, auto, u, v, spacing), reference, rows)
    assert err_local < 1.0e-5
    # Native weights: kdiff off by 0.5-1 % at 50-100 m (measured).
    assert err_native > 2.0e-3
    assert err_native > 100.0 * err_local
    assert isinstance(auto, DeformationWeightsV841)
