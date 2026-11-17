"""The regional scalar-transport edge values with the stencil read once per (edge, level chunk).

``transport_edge_values_regional_v841`` (:mod:`woof.hex.cuda_regional_v841`) runs one thread per
(tracer, level, edge) and every thread re-reads its edge's advection stencil -- ``nAdvCellsForEdge``, the
cell list and both weight rows -- so a group of T tracers reads each stencil row T x nlev times.  On the
Alps variable-resolution cull (14,751 cells, 44,480 edges, 55 levels, the convection-permitting suite's
tracers) it was 44 ms of a 667 ms Metal step (R2-C, isolated per-kernel times).

``transport_edge_values_regional_hoist_v841`` is the same kernel with the launch mapping of
``transport_edge_values_hoist_v841`` (:mod:`woof.hex.cuda_transport_v841`): one thread per (edge, level
chunk) reads the stencil once into registers and walks the levels of its chunk and every tracer.  Each
(tracer, level, edge) element runs the original element body unchanged:

* above ``nRelaxZone`` the lane is written 0.0, as before;
* in the two outer relaxation rows the first-order upwind value uses the same helper calls on the same
  operands -- its two tracer-independent velocity weights are formed once per level, and they are
  helper products whose operands do not depend on the tracer, so they are the same words;
* everywhere else the full high-order branch is the released element body, its subnormal-sensitivity test
  and exact re-evaluation included, with the slot loop over the hoisted stencil in the same ascending
  order (``transport_edge_values_hoist_v841``'s body, reading the native ``[edge][slot]`` tables).

No element reads another element's output, so the words cannot depend on which thread computed them; the
identity is also measured, not only argued (``tests/test_hex_regional_edge_hoist.py``: the two kernels
on the same inputs, every word, in fast and strict numerics, CUDA and Metal).

It is its own translation unit, compiled with the regional unit's options, so the regional unit's source
digest, its contract deck and every receipt that names it are unchanged.

``WOOF_HEX_REGIONAL_EDGE_HOIST=0`` restores the element kernel.  ``WOOF_HEX_EDGE_HOIST_CHECK=N`` runs the
element kernel beside the first N calls and compares the words (:mod:`woof.hex.cuda_edge_hoist`).
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import os
import time
import weakref
from typing import Any

from woof.backend import compilation_target as _compilation_target
from woof.hex import numerics as _hex_numerics

from .cuda_fp32 import CUDA_FTZ_HELPERS
from .cuda_regional_v841 import REGIONAL_CUDA_OPTIONS
from .regional_v841 import N_RELAX_ZONE, N_SPEC_ZONE

#: NVRTC module key for this translation unit.
MODULE_KEY = "hexcore.cuda_regional_hoist_v841"

KERNEL = "transport_edge_values_regional_hoist_v841"
#: The element kernel it replaces (woof.hex.cuda_regional_v841).
ELEMENT_KERNEL = "transport_edge_values_regional_v841"

_CUDA_SOURCE = CUDA_FTZ_HELPERS + r"""
#define RE2(k,e,ne) ((k)*(ne) + (e))
#define RQ3(t,k,c,nlev,nc) (((t)*(nlev) + (k))*(nc) + (c))
#define RQE3(t,k,e,nlev,ne) (((t)*(nlev) + (k))*(ne) + (e))
#define RADV(e,s,w) ((e)*(w) + (s))
#ifndef HEX_HOIST_ADV
#define HEX_HOIST_ADV 10
#endif

/* transport_edge_values_regional_v841 with one thread per (edge, level
   chunk): atm_advance_scalars_work F:4764-4842, the three-way edge split. */
extern "C" __global__ void transport_edge_values_regional_hoist_v841(
    const int ntracers, const int nlev, const int ncells,
    const int nedges, const int width, const float coefficient,
    const int *bdy_mask_edge, const float *dv_edge,
    const int *cells_on_edge,
    const float *stage, const float *velocity,
    const float *adv, const float *adv3,
    const int *n_adv, const int *adv_cells, float *edge_values,
    const int lpt)
{
    const int hex_chunks = (nlev + lpt - 1) / lpt;
    const int element = blockDim.x * blockIdx.x + threadIdx.x;
    if (element >= nedges * hex_chunks) return;
    const int edge = element % nedges;
    const int hex_k0 = (element / nedges) * lpt;
    const int hex_k1 = min(nlev, hex_k0 + lpt);
    const int mask = bdy_mask_edge[edge];
    if (mask > REGIONAL_N_RELAX_ZONE) {
        /* unread by any updated cell; written zero for reproducibility */
        for (int k = hex_k0; k < hex_k1; ++k) {
            for (int tracer = 0; tracer < ntracers; ++tracer) {
                edge_values[RQE3(tracer, k, edge, nlev, nedges)] = 0.0f;
            }
        }
        return;
    }
    if (mask >= REGIONAL_N_RELAX_ZONE - 1) {
        const int cell1 = cells_on_edge[2 * edge];
        const int cell2 = cells_on_edge[2 * edge + 1];
        for (int k = hex_k0; k < hex_k1; ++k) {
            const float vel = velocity[RE2(k, edge, nedges)];
            const float direction = mpas_copysign(0.5f, vel);
            const float u_positive = mpas_mul(
                dv_edge[edge], mpas_abs(mpas_add(direction, 0.5f)));
            const float u_negative = mpas_mul(
                dv_edge[edge], mpas_abs(mpas_sub(direction, 0.5f)));
            for (int tracer = 0; tracer < ntracers; ++tracer) {
                edge_values[RQE3(tracer, k, edge, nlev, nedges)] = mpas_add(
                    mpas_mul(u_positive,
                        stage[RQ3(tracer, k, cell1, nlev, ncells)]),
                    mpas_mul(u_negative,
                        stage[RQ3(tracer, k, cell2, nlev, ncells)]));
            }
        }
        return;
    }
    const int count = n_adv[edge];
    int cl[HEX_HOIST_ADV];
    float b[HEX_HOIST_ADV];
    float t[HEX_HOIST_ADV];
    const bool hoisted = count <= HEX_HOIST_ADV;
#pragma unroll
    for (int slot = 0; slot < HEX_HOIST_ADV; ++slot) {
        const bool live = hoisted && slot < count;
        cl[slot] = live ? adv_cells[RADV(edge, slot, width)] : 0;
        b[slot] = live ? adv[RADV(edge, slot, width)] : 0.0f;
        t[slot] = live ? adv3[RADV(edge, slot, width)] : 0.0f;
    }
    for (int k = hex_k0; k < hex_k1; ++k) {
        const float velocity_sign = mpas_copysign(
            1.0f, velocity[RE2(k, edge, nedges)]);
      for (int tracer = 0; tracer < ntracers; ++tracer) {
        float value = 0.0f;
        bool ftz_sensitive = false;
        if (hoisted) {
#pragma unroll
            for (int slot = 0; slot < HEX_HOIST_ADV; ++slot) {
                if (slot < count) {
                    const float base_weight = b[slot];
                    const float third_weight = t[slot];
                    const float stage_value = stage[RQ3(
                        tracer, k, cl[slot], nlev, ncells)];
                    const float correction = coefficient * velocity_sign
                        * third_weight;
                    const float weight = base_weight + correction;
                    const float contribution = weight * stage_value;
                    const float next = value + contribution;
#if MPAS_FTZ_FALLBACK_ENABLED
                    const unsigned int coefficient_mag =
                        mpas_f32_magnitude_bits(coefficient);
                    const unsigned int third_mag =
                        mpas_f32_magnitude_bits(third_weight);
                    const unsigned int correction_mag =
                        mpas_f32_magnitude_bits(correction);
                    const unsigned int base_mag =
                        mpas_f32_magnitude_bits(base_weight);
                    const unsigned int weight_mag =
                        mpas_f32_magnitude_bits(weight);
                    const unsigned int stage_mag =
                        mpas_f32_magnitude_bits(stage_value);
                    const unsigned int contribution_mag =
                        mpas_f32_magnitude_bits(contribution);
                    const unsigned int value_mag =
                        mpas_f32_magnitude_bits(value);
                    const unsigned int next_mag = mpas_f32_magnitude_bits(next);
                    ftz_sensitive = ftz_sensitive
                        || (base_mag != 0u && (base_mag >> 23) == 0u)
                        || (third_mag != 0u && (third_mag >> 23) == 0u)
                        || (stage_mag != 0u && (stage_mag >> 23) == 0u)
                        || (correction_mag == 0u && coefficient_mag != 0u
                            && third_mag != 0u)
                        || (weight_mag == 0u && base_mag != correction_mag
                            && base_mag != 0u && correction_mag != 0u)
                        || (contribution_mag == 0u && weight_mag != 0u
                            && stage_mag != 0u)
                        || (next_mag == 0u && value_mag != contribution_mag
                            && value_mag != 0u && contribution_mag != 0u);
#endif
                    value = next;
                }
            }
        } else {
            for (int slot = 0; slot < count; ++slot) {
                const int cell = adv_cells[RADV(edge, slot, width)];
                const float base_weight = adv[RADV(edge, slot, width)];
                const float third_weight = adv3[RADV(edge, slot, width)];
                const float stage_value = stage[RQ3(
                    tracer, k, cell, nlev, ncells)];
                const float correction = coefficient * velocity_sign
                    * third_weight;
                const float weight = base_weight + correction;
                const float contribution = weight * stage_value;
                const float next = value + contribution;
#if MPAS_FTZ_FALLBACK_ENABLED
                const unsigned int coefficient_mag =
                    mpas_f32_magnitude_bits(coefficient);
                const unsigned int third_mag =
                    mpas_f32_magnitude_bits(third_weight);
                const unsigned int correction_mag =
                    mpas_f32_magnitude_bits(correction);
                const unsigned int base_mag =
                    mpas_f32_magnitude_bits(base_weight);
                const unsigned int weight_mag =
                    mpas_f32_magnitude_bits(weight);
                const unsigned int stage_mag =
                    mpas_f32_magnitude_bits(stage_value);
                const unsigned int contribution_mag =
                    mpas_f32_magnitude_bits(contribution);
                const unsigned int value_mag =
                    mpas_f32_magnitude_bits(value);
                const unsigned int next_mag = mpas_f32_magnitude_bits(next);
                ftz_sensitive = ftz_sensitive
                    || (base_mag != 0u && (base_mag >> 23) == 0u)
                    || (third_mag != 0u && (third_mag >> 23) == 0u)
                    || (stage_mag != 0u && (stage_mag >> 23) == 0u)
                    || (correction_mag == 0u && coefficient_mag != 0u
                        && third_mag != 0u)
                    || (weight_mag == 0u && base_mag != correction_mag
                        && base_mag != 0u && correction_mag != 0u)
                    || (contribution_mag == 0u && weight_mag != 0u
                        && stage_mag != 0u)
                    || (next_mag == 0u && value_mag != contribution_mag
                        && value_mag != 0u && contribution_mag != 0u);
#endif
                value = next;
            }
        }
        if (ftz_sensitive) {
            value = 0.0f;
            for (int slot = 0; slot < count; ++slot) {
                const int cell = adv_cells[RADV(edge, slot, width)];
                const float weight = mpas_add(
                    adv[RADV(edge, slot, width)],
                    mpas_mul(mpas_mul(coefficient, velocity_sign),
                        adv3[RADV(edge, slot, width)]));
                value = mpas_add(value, mpas_mul(weight,
                    stage[RQ3(tracer, k, cell, nlev, ncells)]));
            }
        }
        edge_values[RQE3(tracer, k, edge, nlev, nedges)] = value;
      }
    }
}
"""


def _source() -> str:
    """The unit with the zone constants bound to the CPU lane, as the regional unit binds them."""

    return (
        f"#define REGIONAL_N_SPEC_ZONE {int(N_SPEC_ZONE)}\n"
        f"#define REGIONAL_N_RELAX_ZONE {int(N_RELAX_ZONE)}\n"
    ) + _CUDA_SOURCE


CUDA_REGIONAL_HOIST_SOURCE = _source()


def enabled() -> bool:
    """``WOOF_HEX_REGIONAL_EDGE_HOIST=0`` launches the element kernel (A/B only; the words are the same)."""

    return os.environ.get("WOOF_HEX_REGIONAL_EDGE_HOIST", "1").strip().lower() not in ("0", "off", "false", "no")


@lru_cache(maxsize=None)
def _direct_module(device_id: int, platform_sha256: str, options: tuple[str, ...]) -> tuple[Any, dict[str, Any]]:
    """Direct NVRTC, as woof.hex.cuda_regional_v841._direct_regional_module compiles the regional unit."""
    import cupy as cp
    from cupy.cuda import compiler
    from .cuda_backend.compile_contract import NvrtcCompileCapture

    with cp.cuda.Device(device_id):
        with NvrtcCompileCapture() as capture:
            binary, _ = compiler.compile_using_nvrtc(
                CUDA_REGIONAL_HOIST_SOURCE, options=options,
                arch=_compilation_target(device_id, xp=cp),
            )
        module = cp.cuda.function.Module()
        module.load(binary)
    evidence = capture.evidence(hashlib.sha256(CUDA_REGIONAL_HOIST_SOURCE.encode()).hexdigest())
    return module, evidence


#: KernelCache -> (kernel key, the resolved kernel before any observer).
_RESOLVED: "weakref.WeakKeyDictionary[Any, tuple[str, Any]]" = weakref.WeakKeyDictionary()


def kernel(kernel_cache: Any) -> Any:
    """The hoisted kernel, resolved through ``kernel_cache`` (and its post-launch observer, if armed)."""

    found = _RESOLVED.get(kernel_cache)
    if found is None:
        started = time.perf_counter()
        options = tuple(_hex_numerics.options(REGIONAL_CUDA_OPTIONS))
        manifest = kernel_cache._compile_manifest
        compiled_key = "direct-nvrtc:" + kernel_cache._module_key(CUDA_REGIONAL_HOIST_SOURCE, options)
        manifest.declare_module(
            MODULE_KEY, source=CUDA_REGIONAL_HOIST_SOURCE,
            requested_options=options, module_cache_key=compiled_key,
        )
        module, evidence = _direct_module(kernel_cache.capability.device_id, manifest.fingerprint_sha256, options)
        manifest.record_effective_compile(MODULE_KEY, evidence)
        manifest.record_kernel(MODULE_KEY, KERNEL)
        kernel_key = f"{compiled_key}:{KERNEL}"
        raw = module.get_function(KERNEL)
        kernel_cache._modules[compiled_key] = module
        kernel_cache._kernels[kernel_key] = raw
        kernel_cache._compile_seconds[kernel_key] = time.perf_counter() - started
        found = _RESOLVED[kernel_cache] = (kernel_key, raw)
    # The element kernel's observer: the forecast's garbage discipline sees the same arguments.
    return kernel_cache._observe(found[0], found[1], KERNEL, MODULE_KEY)


def launch(kernel_cache: Any, arguments: tuple[Any, ...], lpt: int) -> None:
    """``arguments`` are the element kernel's, in its order; ``lpt`` levels per thread."""
    import numpy as np

    nlev, nedges = int(arguments[1]), int(arguments[3])
    chunks = (nlev + int(lpt) - 1) // int(lpt)
    count = nedges * chunks
    if count < 1:
        return
    threads = 128
    kernel(kernel_cache)(((count + threads - 1) // threads,), (threads,), (*arguments, np.int32(lpt)))


__all__ = [
    "CUDA_REGIONAL_HOIST_SOURCE",
    "ELEMENT_KERNEL",
    "KERNEL",
    "MODULE_KEY",
    "enabled",
    "kernel",
    "launch",
]
