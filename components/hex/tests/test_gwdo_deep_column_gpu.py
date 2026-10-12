"""The hex YSU-GWDO kernel on columns up to the hex ceiling (256 levels).

The kernel is compiled per run at the run's column (``GWDO_NLEV``), so a
deeper column is the anchored source with one define changed.  These gates
run it on a synthetic, stably stratified column with orography at the
anchored 55 levels, at the old 160-level bound and at 256, and hold every
output to the FP32 CPU source-loop oracle, and hold the compiled frame to
the price ``woof.hex.device_admission`` charges for it.
"""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

from woof.hex import cuda_gwdo_v841 as gwdo
from woof.hex import device_admission
from woof.hex.cuda_physics_prep_v841 import CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256

FIELDS = (
    "rublten", "rvblten", "dtaux3d", "dtauy3d",
    "dusfcg", "dvsfcg", "rubldiff", "rvbldiff",
)
NCELLS = 24


def _inputs(nlev: int, seed: int = 7):
    """A hydrostatic column topped at 30 km with orography under every cell."""

    rng = np.random.default_rng(seed)
    zi = 30_000.0 * (np.arange(nlev + 1) / nlev) ** 1.5
    zm = 0.5 * (zi[1:] + zi[:-1])

    def temp(z):
        return np.where(z < 11_000.0, 288.15 - 6.5e-3 * z, 216.65)

    ps = 101_325.0 - 2_000.0 * rng.random(NCELLS)
    fine = np.linspace(0.0, 30_000.0, 30_001)
    integrand = 9.80616 / (287.0 * temp(fine))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]))])
    p_interface = ps[None, :] * np.exp(-np.interp(zi, fine, cum))[:, None]
    p_mass = np.sqrt(p_interface[1:] * p_interface[:-1])
    shear = np.minimum(zm / 12_000.0, 1.0)[:, None]

    def f32(value):
        return np.ascontiguousarray(value, dtype=np.float32)

    dynamic = {
        "u_p": f32(8.0 + 22.0 * shear + rng.normal(0.0, 1.0, (nlev, NCELLS))),
        "v_p": f32(4.0 * np.sin(zm / 3_000.0)[:, None]
                   + rng.normal(0.0, 1.0, (nlev, NCELLS))),
        "t_p": f32(temp(zm)[:, None] + rng.normal(0.0, 0.3, (nlev, NCELLS))),
        "qv_p": f32(0.012 * np.exp(-zm / 2_500.0)[:, None]
                    * (0.8 + 0.4 * rng.random((nlev, NCELLS)))),
        "pres_hydd_p": f32(p_mass),
        "pres2_hydd_p": f32(p_interface),
        "pi_p": f32((p_mass / 1.0e5) ** (287.0 / 1004.5)),
        "zmid_p": f32(np.broadcast_to(zm[:, None], (nlev, NCELLS))),
    }
    tendency = {
        "rublten": f32(rng.normal(0.0, 1e-4, (nlev, NCELLS))),
        "rvblten": f32(rng.normal(0.0, 1e-4, (nlev, NCELLS))),
    }
    static = {
        "mesh_density": f32(0.5 + 0.5 * rng.random(NCELLS)),
        "var2d": f32(100.0 + 400.0 * rng.random(NCELLS)),
        "con": f32(1.0 + 6.0 * rng.random(NCELLS)),
        **{f"oa{i}": f32(rng.uniform(-0.8, 0.8, NCELLS)) for i in range(1, 5)},
        **{f"ol{i}": f32(rng.random(NCELLS)) for i in range(1, 5)},
    }
    return dynamic, tendency, static


@pytest.mark.parametrize("nlev", [55, 160, 256])
def test_gwdo_matches_its_oracle_up_to_the_hex_ceiling(nlev, monkeypatch, tmp_path):
    from woof.hex.cuda_backend.runtime import KernelCache

    # The bind rebinds the column constant to the row's own column; do the
    # same here so the sealed view and the oracle accept this column.
    monkeypatch.setattr(gwdo, "X4_N_VERT_LEVELS", nlev)
    dynamic, tendency, static = _inputs(nlev)
    cache = KernelCache(cache_dir=tmp_path / "kernels")
    view = gwdo.CudaYsuGwdoColumnViewV841(
        **{name: cp.asarray(value) for name, value in dynamic.items()},
        time_seconds=0.0,
        prep_contract_sha256=CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256,
        _seal=gwdo._COLUMN_VIEW_SEAL,
    )
    sealed_static = gwdo.CudaYsuGwdoStaticV841.from_resident(
        **{name: cp.asarray(value) for name, value in static.items()},
        effective_length_scale_m=25_000.0,
    )
    result = gwdo.run_bl_ysu_gwdo_cuda_v841(
        view,
        rublten=cp.asarray(tendency["rublten"]),
        rvblten=cp.asarray(tendency["rvblten"]),
        static=sealed_static,
        dt_seconds=120.0,
        kernel_cache=cache,
    )
    receipt = result.receipt()
    assert receipt["shape"] == [nlev, NCELLS]
    if nlev != gwdo.GWDO_ANCHORED_LEVELS:
        assert receipt["kernel_level_variant"]["levels"] == nlev

    oracle = gwdo.bl_ysu_gwdo_cpu_oracle_v841(
        **dynamic, **tendency, **static,
        effective_length_scale_m=25_000.0, dt_seconds=120.0,
    )
    for name in FIELDS:
        got = cp.asnumpy(getattr(result, name))
        want = np.asarray(getattr(oracle, name), dtype=np.float32)
        assert np.isfinite(got).all(), name
        scale = max(float(np.abs(want).max()), 1e-30)
        # MEASURED on sm_86: at most 1.6e-7 of the field's own maximum (a
        # last-bit libm difference) at every column 55..256.
        assert float(np.abs(got - want).max()) <= 1e-6 * scale, name
    # The column carries real drag: a stack of zeros would pass the oracle.
    assert float(np.abs(cp.asnumpy(result.dtaux3d)).max()) > 0.0

    # The frame this column compiled is the one the admission prices.
    kernel = next(iter(cache._kernels.values()))
    frame = int(kernel.attributes["local_size_bytes"])
    assert frame <= device_admission.gwdo_local_frame_bytes(nlev)
