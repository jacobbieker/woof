"""CUDA kernels and forecast attachment for the MPAS-A v8.4.1 LES closure.

The binary32 device mirror of :mod:`woof.hex.les_v841` (every kernel names
the CPU function it mirrors; the CPU module carries the native source map).
Kernels walk a flat ``(level, entity)`` element index, entity fastest, in a
grid-stride loop, the launch convention of the other v8.4.1 translation
units, and are compiled with ``--fmad=false`` so ``a*b+c`` rounds twice as
the CPU authority does.

Why attachment rather than an edit of the dycore
-------------------------------------------------
``cuda_driver.py`` and ``cuda_horizontal_v841.py`` are exact-byte pinned
execution sources and both are in the regional kernel set, so editing either
lapses every minted regional class.  The LES closure therefore enters the
way :mod:`woof.hex.cuda_driver_lts` does: :func:`attach_les_v841` rebinds three
call targets on one built driver and leaves every pinned byte where it is.

* ``driver.horizontal.compute_dry_mixing_tendencies_v841`` -- the RK-step-1
  explicit-mixing call (native ``atm_compute_dyn_tend`` at ``rk_step == 1``).
  With an LES model active native does not call ``smagorinsky_2d`` at all;
  ``les_models`` supplies the horizontal and vertical viscosities and the
  u/w/theta dissipation routines take their LES branches.  The replacement
  returns the same :class:`CudaDryMixingTendencies` the driver already adds
  into its saved Euler pools, with ``kdiff`` the LES horizontal viscosity.
* ``driver._advance_dynamics_subcycle_v841`` -- wrapped (not replaced) so
  the closure sees the substep-start state and moist coefficients the
  subcycle itself uses, knows which dynamics substep it is in (native
  ``dynamics_substep``: the TKE source is formed on substep 1 only, and the
  reconstructed cell winds ``uReconstructZonal/Meridional`` are the
  step-start values).
* ``driver._step_device_v841`` -- wrapped so the substep count restarts at
  every outer step and the TKE advances once the whole step has returned.

With ``config_les_model='none'`` :func:`attach_les_v841` returns ``None``
and rebinds nothing, so the default run is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import ConfigurationRefusal
from .les_v841 import (
    CP,
    DEFAULT_COLD_START_TKE,
    EPSILON_BV,
    GRAVITY,
    LES_MODEL_3D_SMAGORINSKY,
    LES_MODEL_NONE,
    LES_MODEL_PROGNOSTIC_15_ORDER,
    LES_SURFACE_NONE,
    LES_SURFACE_SPECIFIED,
    LES_SURFACE_VARYING,
    PRANDTL,
    QC_CR,
    RGAS,
    RV,
    SVP1,
    SVP2,
    SVP3,
    SVPT0,
    TKE_COLD_START_LABEL,
    TKE_TRANSPORT_LABEL,
    XLV,
    C_K,
    LesV841Config,
    initialize_les_gradient_weights_v841,
    les_config_from_dycore,
    les_label,
)

MODULE_KEY = "hexcore.cuda_les_v841"
COMPILE_OPTIONS = ("--std=c++17", "--fmad=false")


def _f(value: float) -> str:
    """A float literal for the CUDA source, exact in binary32 round-trip."""

    return repr(float(np.float32(value))) + "f"


CUDA_LES_SOURCE = (
    r"""
#define LES_ELEMENT_LOOP(total) \
    for (int element = blockDim.x * blockIdx.x + threadIdx.x; \
         element < (total); element += gridDim.x * blockDim.x)
__device__ __forceinline__ int lidx(int level, int entity, int count) {
    return level * count + entity;
}
__device__ __forceinline__ float cell_sign(int cell, int edge, const int *coe) {
    return coe[2 * edge] == cell ? 1.0f : -1.0f;
}
__device__ __forceinline__ float vertex_sign(int vertex, int edge, const int *voe) {
    return voe[2 * edge + 1] == vertex ? 1.0f : -1.0f;
}
"""
    + f"""
#define LES_C_K {_f(C_K)}
#define LES_EPSILON_BV {_f(EPSILON_BV)}
#define LES_GRAVITY {_f(GRAVITY)}
#define LES_RGAS {_f(RGAS)}
#define LES_RV {_f(RV)}
#define LES_CP {_f(CP)}
#define LES_XLV {_f(XLV)}
#define LES_SVP1 {_f(SVP1)}
#define LES_SVP2 {_f(SVP2)}
#define LES_SVP3 {_f(SVP3)}
#define LES_SVPT0 {_f(SVPT0)}
#define LES_QC_CR {_f(QC_CR)}
#define LES_PRANDTL {_f(PRANDTL)}
"""
    + r"""
/* woof.hex.vector._reconstruct_common (mpas_vector_reconstruction.F). */
extern "C" __global__ void les_reconstruct_v841_f32(
    const float *u, const int *edges_on_cell, const int *n_edges_on_cell,
    const float *coeffs, const float *lat_cell, const float *lon_cell,
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const int coeff_slots, float *ur, float *vr)
{
    LES_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        float x = 0.0f, y = 0.0f, z = 0.0f;
        const int count = n_edges_on_cell[cell];
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            if (edge < 0 || edge >= nedges) continue;
            const float value = u[lidx(k, edge, nedges)];
            const int base = (cell * coeff_slots + slot) * 3;
            x = x + coeffs[base] * value;
            y = y + coeffs[base + 1] * value;
            z = z + coeffs[base + 2] * value;
        }
        const float lat = lat_cell[cell];
        const float lon = lon_cell[cell];
        const float sin_lat = sinf(lat), cos_lat = cosf(lat);
        const float sin_lon = sinf(lon), cos_lon = cosf(lon);
        const int index = lidx(k, cell, ncells);
        ur[index] = -x * sin_lon + y * cos_lon;
        vr[index] = -(x * cos_lon + y * sin_lon) * sin_lat + z * cos_lat;
    }
}

/* meshScalingDel2/Del4 (mixing.compute_mesh_mixing_scaling). */
extern "C" __global__ void les_mesh_scaling_v841_f32(
    const int *cells_on_edge, const float *mesh_density, const int nedges,
    const int scale_with_mesh, float *del2, float *del4)
{
    LES_ELEMENT_LOOP(nedges) {
        const int edge = element;
        if (!scale_with_mesh) { del2[edge] = 1.0f; del4[edge] = 1.0f; continue; }
        const float mean = 0.5f * (mesh_density[cells_on_edge[2 * edge]]
                                   + mesh_density[cells_on_edge[2 * edge + 1]]);
        del2[edge] = 1.0f / powf(mean, 0.25f);
        del4[edge] = 1.0f / powf(mean, 0.75f);
    }
}

struct LesN2Level { float theta; float temp; float qvsw; float coefa; };

__device__ __forceinline__ LesN2Level les_n2_level(
    int k, int cell, int ncells, const float *theta_m, const float *exner,
    const float *pressure_base, const float *pressure_p, const float *qv,
    const float rvord)
{
    const int i = lidx(k, cell, ncells);
    LesN2Level out;
    out.theta = theta_m[i] / (1.0f + rvord * qv[i]);
    out.temp = exner[i] * out.theta;
    const float p = pressure_base[i] + pressure_p[i];
    float esw = 1000.0f * LES_SVP1
        * expf(LES_SVP2 * (out.temp - LES_SVPT0) / (out.temp - LES_SVP3));
    if (p < esw) esw = p * 0.99f;
    const float ep_2 = LES_RGAS / LES_RV;
    out.qvsw = ep_2 * esw / (p - esw);
    out.coefa = (1.0f + LES_XLV * out.qvsw / LES_RGAS / out.temp)
        / (1.0f + LES_XLV * LES_XLV * out.qvsw / LES_CP / LES_RV / out.temp / out.temp);
    return out;
}

/* les_v841.calculate_n2_v841 (calculate_n2). */
extern "C" __global__ void les_n2_v841_f32(
    const float *theta_m, const float *exner, const float *pressure_base,
    const float *pressure_p, const float *zgrid, const float *qv,
    const float *qc, const float *qtot, const int has_qc,
    const float rvord, const int nlev, const int ncells, float *bn2)
{
    LES_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int level = element / ncells;
        int k = level;
        if (k < 1) k = 1;
        if (k > nlev - 2) k = nlev - 2;
        const LesN2Level lo = les_n2_level(k - 1, cell, ncells, theta_m, exner,
            pressure_base, pressure_p, qv, rvord);
        const LesN2Level mid = les_n2_level(k, cell, ncells, theta_m, exner,
            pressure_base, pressure_p, qv, rvord);
        const LesN2Level hi = les_n2_level(k + 1, cell, ncells, theta_m, exner,
            pressure_base, pressure_p, qv, rvord);
        const float dz = 0.5f * (zgrid[lidx(k + 2, cell, ncells)] + zgrid[lidx(k + 1, cell, ncells)])
                       - 0.5f * (zgrid[lidx(k, cell, ncells)] + zgrid[lidx(k - 1, cell, ncells)]);
        const float rdz = 1.0f / dz;
        const float dqt = qtot[lidx(k + 1, cell, ncells)] - qtot[lidx(k - 1, cell, ncells)];
        const float dqv = qv[lidx(k + 1, cell, ncells)] - qv[lidx(k - 1, cell, ncells)];
        float value;
        bool dry = true;
        if (has_qc && qc[lidx(k, cell, ncells)] >= LES_QC_CR) dry = false;
        if (dry) {
            value = LES_GRAVITY * ((hi.theta - lo.theta) / mid.theta * rdz
                                   + rvord * dqv * rdz - dqt * rdz);
        } else {
            value = LES_GRAVITY * (mid.coefa * ((hi.theta - lo.theta) / mid.theta * rdz
                                   + LES_XLV / LES_CP / mid.temp * (hi.qvsw - lo.qvsw) * rdz)
                                   - dqt * rdz);
        }
        bn2[lidx(level, cell, ncells)] = value;
    }
}

/* les_v841.les_models_v841 (les_models). model: 1 smagorinsky, 2 tke. */
extern "C" __global__ void les_models_v841_f32(
    const float *u, const float *v, const float *ur, const float *vr,
    const float *w, const float *bn2, const float *zgrid, const float *rho_zz,
    const float *coef_c2, const float *coef_s2, const float *coef_cs,
    const float *coef_c, const float *coef_s,
    const int *edges_on_cell, const int *n_edges_on_cell,
    const int *cells_on_edge,
    const float c_s, const float len, const float inv_dt,
    const int model, const int write_tend,
    const int nlev, const int ncells, const int nedges, const int max_edges,
    float *tke, float *kh, float *kv, float *pr3d, float *tend_tke)
{
    LES_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        float dudx = 0.0f, dudy = 0.0f, dvdx = 0.0f, dvdy = 0.0f;
        float dwdx = 0.0f, dwdy = 0.0f;
        const int count = n_edges_on_cell[cell];
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            if (edge < 0 || edge >= nedges) continue;
            const int ws = cell * max_edges + slot;
            const float ue = u[lidx(k, edge, nedges)];
            const float ve = v[lidx(k, edge, nedges)];
            const float a_c2 = coef_c2[ws], a_s2 = coef_s2[ws], a_cs = coef_cs[ws];
            dudx = dudx + (a_c2 * ue - a_cs * ve);
            dudy = dudy + (a_cs * ue - a_s2 * ve);
            dvdx = dvdx + (a_cs * ue + a_c2 * ve);
            dvdy = dvdy + (a_s2 * ue + a_cs * ve);
            const int c1 = cells_on_edge[2 * edge];
            const int c2 = cells_on_edge[2 * edge + 1];
            const float wk = 0.5f * (w[lidx(k, c1, ncells)] + w[lidx(k, c2, ncells)]);
            dwdx = dwdx + coef_c[ws] * wk;
            dwdy = dwdy + coef_s[ws] * wk;
        }
        const float zk = zgrid[lidx(k, cell, ncells)];
        const float zk1 = zgrid[lidx(k + 1, cell, ncells)];
        const float dwdz = (w[lidx(k + 1, cell, ncells)] - w[lidx(k, cell, ncells)])
                         * (1.0f / (zk1 - zk));
        float dudz, dvdz;
        if (k >= 1 && k <= nlev - 2) {
            const float rdz = 1.0f / (zgrid[lidx(k + 2, cell, ncells)] + zk1 - zk
                                      - zgrid[lidx(k - 1, cell, ncells)]);
            dudz = (ur[lidx(k + 1, cell, ncells)] - ur[lidx(k - 1, cell, ncells)]) * rdz;
            dvdz = (vr[lidx(k + 1, cell, ncells)] - vr[lidx(k - 1, cell, ncells)]) * rdz;
        } else if (k == 0) {
            const float rdz = 1.0f / (zk1 - zk);
            dudz = (ur[lidx(1, cell, ncells)] - ur[lidx(0, cell, ncells)]) * rdz;
            dvdz = (vr[lidx(1, cell, ncells)] - vr[lidx(0, cell, ncells)]) * rdz;
        } else {
            const float rdz = 1.0f / (zk - zgrid[lidx(k - 1, cell, ncells)]);
            dudz = (ur[lidx(k, cell, ncells)] - ur[lidx(k - 1, cell, ncells)]) * rdz;
            dvdz = (vr[lidx(k, cell, ncells)] - vr[lidx(k - 1, cell, ncells)]) * rdz;
        }
        const float d11 = 2.0f * dudx;
        const float d22 = 2.0f * dvdy;
        const float d33 = 2.0f * dwdz;
        const float d12 = dudy + dvdx;
        const float d13 = dwdx + dudz;
        const float d23 = dwdy + dvdz;
        const int index = lidx(k, cell, ncells);
        const float n2 = bn2[index];
        const float delta_z = zk1 - zk;
        const float pr_inv = 1.0f / LES_PRANDTL;
        const float ceiling_h = (0.01f * (len * len)) * inv_dt;
        if (model == 1) {
            const float def2 = 0.5f * (d11 * d11 + d22 * d22 + d33 * d33)
                             + d12 * d12 + d13 * d13 + d23 * d23;
            const float root = sqrtf(fmaxf(0.0f, def2 - pr_inv * n2));
            kh[index] = fminf(((c_s * len) * (c_s * len)) * root, ceiling_h);
            kv[index] = ((c_s * delta_z) * (c_s * delta_z)) * root;
            pr3d[index] = pr_inv;
        } else {
            const float e = fmaxf(0.0f, tke[index]);
            tke[index] = e;
            const float third = 1.0f / 3.0f;
            const float delta_s = powf((len * len) * delta_z, third);
            const float bv = fmaxf(sqrtf(fabsf(n2)), LES_EPSILON_BV);
            const float sqrt_e = sqrtf(e);
            float tke_length = n2 > 1.0e-06f ? 0.76f * sqrt_e / bv : delta_s;
            tke_length = fminf(tke_length, delta_z);
            float diss_length = fminf(delta_s, fmaxf(tke_length, 0.01f * delta_s));
            if (n2 <= 0.0f) diss_length = delta_s;
            const float l_vertical = fminf(delta_z, tke_length);
            if (n2 <= 0.0f) diss_length = delta_z;
            const float k_h = fminf(LES_C_K * len * sqrt_e, ceiling_h);
            const float k_v = fminf(LES_C_K * l_vertical * sqrt_e,
                                    (0.01f * (delta_z * delta_z)) * inv_dt);
            kh[index] = k_h;
            kv[index] = k_v;
            pr3d[index] = 1.0f + (2.0f * l_vertical / delta_z);
            if (write_tend) {
                const float shear = k_h * (d11 * d11 + d22 * d22 + d12 * d12)
                                  + k_v * (d33 * d33 + d13 * d13 + d23 * d23);
                const float buoyancy = -k_v * n2;
                const float c_diss = 1.9f * LES_C_K
                    + fmaxf(0.0f, 0.93f - 1.9f * LES_C_K) * diss_length / delta_s;
                const float dissipation = -c_diss * powf(e, 1.5f) / diss_length;
                tend_tke[index] = rho_zz[index] * (shear + buoyancy + dissipation);
            }
        }
    }
}

/* les_v841.u_dissipation_les_v841: del2 with tau_12 = 1. */
extern "C" __global__ void les_u_lap2_v841_f32(
    const float *divergence, const float *vorticity, const float *kh,
    const float *rho_edge, const float *scale2, const int *cells_on_edge,
    const int *vertices_on_edge, const float *dc_edge, const float *dv_edge,
    const int nlev, const int ncells, const int nedges, const int nvertices,
    float *delsq_u, float *tend_u)
{
    LES_ELEMENT_LOOP(nlev * nedges) {
        const int edge = element % nedges;
        const int k = element / nedges;
        const int index = lidx(k, edge, nedges);
        if (!(dc_edge[edge] > 0.0f) || !(dv_edge[edge] > 0.0f)) {
            delsq_u[index] = 0.0f;
            tend_u[index] = 0.0f;
            continue;
        }
        const int c1 = cells_on_edge[2 * edge], c2 = cells_on_edge[2 * edge + 1];
        const int v1 = vertices_on_edge[2 * edge], v2 = vertices_on_edge[2 * edge + 1];
        const float inv_dc = 1.0f / dc_edge[edge];
        const float r_dv = fminf(1.0f / dv_edge[edge], 4.0f * inv_dc);
        const float grad_div = (divergence[lidx(k, c2, ncells)] - divergence[lidx(k, c1, ncells)]) * inv_dc;
        const float u_diffusion = grad_div
            - (vorticity[lidx(k, v2, nvertices)] - vorticity[lidx(k, v1, nvertices)]) * r_dv;
        const float u_les = u_diffusion + 1.0f * grad_div;
        delsq_u[index] = u_diffusion;
        const float kdiffu = 0.5f * (kh[lidx(k, c1, ncells)] + kh[lidx(k, c2, ncells)]);
        tend_u[index] = rho_edge[index] * kdiffu * u_les * scale2[edge];
    }
}

extern "C" __global__ void les_delsq_vort_v841_f32(
    const float *delsq_u, const int *edges_on_vertex, const int *vertices_on_edge,
    const float *dc_edge, const float *area_triangle, const int nlev,
    const int nedges, const int nvertices, const int vertex_degree,
    float *delsq_vort)
{
    LES_ELEMENT_LOOP(nlev * nvertices) {
        const int vertex = element % nvertices;
        const int k = element / nvertices;
        const float inv_at = 1.0f / area_triangle[vertex];
        float acc = 0.0f;
        for (int slot = 0; slot < vertex_degree; ++slot) {
            const int edge = edges_on_vertex[vertex * vertex_degree + slot];
            if (edge < 0 || edge >= nedges) continue;
            const float edge_sign = inv_at * dc_edge[edge]
                * vertex_sign(vertex, edge, vertices_on_edge);
            acc = acc + edge_sign * delsq_u[lidx(k, edge, nedges)];
        }
        delsq_vort[lidx(k, vertex, nvertices)] = acc;
    }
}

extern "C" __global__ void les_delsq_div_v841_f32(
    const float *delsq_u, const int *edges_on_cell, const int *n_edges_on_cell,
    const int *cells_on_edge, const float *dv_edge, const float *area_cell,
    const int nlev, const int ncells, const int nedges, const int max_edges,
    float *delsq_div)
{
    LES_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const float inv_ac = 1.0f / area_cell[cell];
        float acc = 0.0f;
        const int count = n_edges_on_cell[cell];
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            if (edge < 0 || edge >= nedges) continue;
            const float edge_sign = inv_ac * dv_edge[edge]
                * cell_sign(cell, edge, cells_on_edge);
            acc = acc + edge_sign * delsq_u[lidx(k, edge, nedges)];
        }
        delsq_div[lidx(k, cell, ncells)] = acc;
    }
}

extern "C" __global__ void les_u_lap4_v841_f32(
    const float *rho_edge, const float *delsq_div, const float *delsq_vort,
    const float *scale4, const int *cells_on_edge, const int *vertices_on_edge,
    const float *dc_edge, const float *dv_edge, const float h4,
    const float div_factor, const int nlev, const int ncells, const int nedges,
    const int nvertices, float *tend_u)
{
    LES_ELEMENT_LOOP(nlev * nedges) {
        const int edge = element % nedges;
        const int k = element / nedges;
        if (!(dc_edge[edge] > 0.0f) || !(dv_edge[edge] > 0.0f)) continue;
        const int c1 = cells_on_edge[2 * edge], c2 = cells_on_edge[2 * edge + 1];
        const int v1 = vertices_on_edge[2 * edge], v2 = vertices_on_edge[2 * edge + 1];
        const float inv_dc = 1.0f / dc_edge[edge];
        const float inv_dv = 1.0f / dv_edge[edge];
        const float u_mix_scale = scale4[edge] * h4;
        const float r_dc4 = u_mix_scale * div_factor * inv_dc;
        const float r_dv4 = u_mix_scale * fminf(inv_dv, 4.0f * inv_dc);
        const int index = lidx(k, edge, nedges);
        const float filt = rho_edge[index] * (
            (delsq_div[lidx(k, c2, ncells)] - delsq_div[lidx(k, c1, ncells)]) * r_dc4
            - (delsq_vort[lidx(k, v2, nvertices)] - delsq_vort[lidx(k, v1, nvertices)]) * r_dv4);
        tend_u[index] = tend_u[index] - filt;
    }
}

__device__ __forceinline__ float les_u_interior_flux(
    int k, int c1, int c2, int edge, const float *u, const float *rho_zz,
    const float *zz, const float *kv, const float *rdzu, const float *fzm,
    const float *fzp, int ncells, int nedges)
{
    const float rho_k_1 = fzm[k] * rho_zz[lidx(k, c1, ncells)] * zz[lidx(k, c1, ncells)] * kv[lidx(k, c1, ncells)]
                        + fzp[k] * rho_zz[lidx(k - 1, c1, ncells)] * zz[lidx(k - 1, c1, ncells)] * kv[lidx(k - 1, c1, ncells)];
    const float rho_k_2 = fzm[k] * rho_zz[lidx(k, c2, ncells)] * zz[lidx(k, c2, ncells)] * kv[lidx(k, c2, ncells)]
                        + fzp[k] * rho_zz[lidx(k - 1, c2, ncells)] * zz[lidx(k - 1, c2, ncells)] * kv[lidx(k - 1, c2, ncells)];
    const float rho_k_at_w = 0.5f * (rho_k_1 + rho_k_2);
    const float zz_1 = fzm[k] * zz[lidx(k, c1, ncells)] + fzp[k] * zz[lidx(k - 1, c1, ncells)];
    const float zz_2 = fzm[k] * zz[lidx(k, c2, ncells)] + fzp[k] * zz[lidx(k - 1, c2, ncells)];
    const float zz_at_w = 0.5f * (zz_1 + zz_2);
    return -rho_k_at_w * zz_at_w * rdzu[k]
        * (u[lidx(k, edge, nedges)] - u[lidx(k - 1, edge, nedges)]);
}

/* u_dissipation_3d LES vertical flux.  surface: 0 none, 1 specified, 2 varying. */
extern "C" __global__ void les_u_vertical_v841_f32(
    const float *u, const float *v, const float *rho_edge, const float *rho_zz,
    const float *zz, const float *kv, const float *rdzu, const float *rdzw,
    const float *fzm, const float *fzp, const int *cells_on_edge,
    const float *dc_edge, const int surface, const float drag,
    const float *ustm, const int nlev, const int ncells, const int nedges,
    float *tend_u)
{
    LES_ELEMENT_LOOP(nlev * nedges) {
        const int edge = element % nedges;
        const int k = element / nedges;
        if (!(dc_edge[edge] > 0.0f)) continue;
        const int c1 = cells_on_edge[2 * edge], c2 = cells_on_edge[2 * edge + 1];
        float flux_lo, flux_hi;
        /* flux(k) */
        if (k == 0) {
            if (surface == 1) {
                const float u0 = u[lidx(0, edge, nedges)];
                const float v0 = v[lidx(0, edge, nedges)];
                const float speed = sqrtf(u0 * u0 + v0 * v0);
                flux_lo = -rho_edge[lidx(0, edge, nedges)] * drag * u0 * speed;
            } else if (surface == 2) {
                const float u0 = u[lidx(0, edge, nedges)];
                const float v0 = v[lidx(0, edge, nedges)];
                const float ust = 0.5f * (ustm[c1] + ustm[c2]);
                const float speed = fmaxf(sqrtf(u0 * u0 + v0 * v0), 0.1f);
                flux_lo = -rho_edge[lidx(0, edge, nedges)] * ust * ust * (u0 / speed);
            } else {
                flux_lo = nlev > 1 ? les_u_interior_flux(1, c1, c2, edge, u, rho_zz, zz, kv,
                                                       rdzu, fzm, fzp, ncells, nedges) : 0.0f;
            }
        } else {
            flux_lo = les_u_interior_flux(k, c1, c2, edge, u, rho_zz, zz, kv, rdzu,
                                          fzm, fzp, ncells, nedges);
        }
        /* flux(k+1); flux(nlev) = flux(nlev-1) */
        const int kk = (k + 1 == nlev) ? nlev - 1 : k + 1;
        if (kk == 0) {
            flux_hi = flux_lo;
        } else {
            flux_hi = les_u_interior_flux(kk, c1, c2, edge, u, rho_zz, zz, kv, rdzu,
                                          fzm, fzp, ncells, nedges);
        }
        const int index = lidx(k, edge, nedges);
        tend_u[index] = tend_u[index] - rdzw[k] * (flux_hi - flux_lo);
    }
}

/* w_dissipation_3d del2; element over interfaces 0..nlev. */
extern "C" __global__ void les_w_lap2_v841_f32(
    const float *w, const float *rho_edge, const float *kh, const float *scale2,
    const int *edges_on_cell, const int *n_edges_on_cell, const int *cells_on_edge,
    const float *dc_edge, const float *dv_edge, const float *area_cell,
    const int nlev, const int ncells, const int nedges, const int max_edges,
    float *delsq_w, float *tend_w)
{
    LES_ELEMENT_LOOP((nlev + 1) * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        if (k == 0 || k == nlev) {
            tend_w[lidx(k, cell, ncells)] = 0.0f;
            if (k == 0) delsq_w[lidx(0, cell, ncells)] = 0.0f;
            continue;
        }
        const float inv_ac = 1.0f / area_cell[cell];
        float dsq = 0.0f, tend = 0.0f;
        const int count = n_edges_on_cell[cell];
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            if (edge < 0 || edge >= nedges || !(dc_edge[edge] > 0.0f)) continue;
            const int c1 = cells_on_edge[2 * edge], c2 = cells_on_edge[2 * edge + 1];
            const float edge_sign = 0.5f * inv_ac * cell_sign(cell, edge, cells_on_edge)
                * dv_edge[edge] * (1.0f / dc_edge[edge]);
            float flux = edge_sign * (rho_edge[lidx(k, edge, nedges)] + rho_edge[lidx(k - 1, edge, nedges)])
                * (w[lidx(k, c2, ncells)] - w[lidx(k, c1, ncells)]);
            dsq = dsq + flux;
            flux = flux * scale2[edge] * 0.25f
                * (kh[lidx(k, c1, ncells)] + kh[lidx(k, c2, ncells)]
                   + kh[lidx(k - 1, c1, ncells)] + kh[lidx(k - 1, c2, ncells)]);
            tend = tend + flux;
        }
        delsq_w[lidx(k, cell, ncells)] = dsq;
        tend_w[lidx(k, cell, ncells)] = tend;
    }
}

extern "C" __global__ void les_w_lap4_v841_f32(
    const float *delsq_w, const float *scale4, const int *edges_on_cell,
    const int *n_edges_on_cell, const int *cells_on_edge, const float *dc_edge,
    const float *dv_edge, const float *area_cell, const float h4,
    const int nlev, const int ncells, const int nedges, const int max_edges,
    float *tend_w)
{
    LES_ELEMENT_LOOP((nlev - 1) * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells + 1;
        const float r_area = h4 * (1.0f / area_cell[cell]);
        float tend = tend_w[lidx(k, cell, ncells)];
        const int count = n_edges_on_cell[cell];
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            if (edge < 0 || edge >= nedges || !(dc_edge[edge] > 0.0f)) continue;
            const int c1 = cells_on_edge[2 * edge], c2 = cells_on_edge[2 * edge + 1];
            const float edge_sign = scale4[edge] * r_area * dv_edge[edge]
                * cell_sign(cell, edge, cells_on_edge) * (1.0f / dc_edge[edge]);
            tend = tend - edge_sign * (delsq_w[lidx(k, c2, ncells)] - delsq_w[lidx(k, c1, ncells)]);
        }
        tend_w[lidx(k, cell, ncells)] = tend;
    }
}

__device__ __forceinline__ float les_w_layer_flux(
    int j, int cell, const float *w, const float *rho_zz, const float *kv,
    const float *zz, const float *rdzw, const float *divergence, int ncells)
{
    const int i = lidx(j, cell, ncells);
    return -rho_zz[i] * kv[i] * zz[i]
        * (2.0f * zz[i] * rdzw[j] * (w[lidx(j + 1, cell, ncells)] - w[lidx(j, cell, ncells)])
           + divergence[i]);
}

extern "C" __global__ void les_w_vertical_v841_f32(
    const float *w, const float *rho_zz, const float *kv, const float *zz,
    const float *rdzu, const float *rdzw, const float *divergence,
    const int nlev, const int ncells, float *tend_w)
{
    LES_ELEMENT_LOOP((nlev - 1) * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells + 1;
        const float upper = les_w_layer_flux(k, cell, w, rho_zz, kv, zz, rdzw, divergence, ncells);
        const float lower = les_w_layer_flux(k - 1, cell, w, rho_zz, kv, zz, rdzw, divergence, ncells);
        const int index = lidx(k, cell, ncells);
        tend_w[index] = tend_w[index] - rdzu[k] * (upper - lower);
    }
}

extern "C" __global__ void les_theta_lap2_v841_f32(
    const float *theta_m, const float *rho_edge, const float *kh,
    const float *scale2, const int *edges_on_cell, const int *n_edges_on_cell,
    const int *cells_on_edge, const float *dc_edge, const float *dv_edge,
    const float *area_cell, const int nlev, const int ncells, const int nedges,
    const int max_edges, float *delsq_theta, float *tend_theta)
{
    LES_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const float inv_ac = 1.0f / area_cell[cell];
        const float prandtl_inv = 1.0f / LES_PRANDTL;
        float dsq = 0.0f, tend = 0.0f;
        const int count = n_edges_on_cell[cell];
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            if (edge < 0 || edge >= nedges || !(dc_edge[edge] > 0.0f)) continue;
            const int c1 = cells_on_edge[2 * edge], c2 = cells_on_edge[2 * edge + 1];
            const float edge_sign = inv_ac * cell_sign(cell, edge, cells_on_edge)
                * dv_edge[edge] * (1.0f / dc_edge[edge]);
            const float pr_scale = prandtl_inv * scale2[edge];
            float flux = edge_sign * (theta_m[lidx(k, c2, ncells)] - theta_m[lidx(k, c1, ncells)])
                * rho_edge[lidx(k, edge, nedges)];
            dsq = dsq + flux;
            flux = flux * 0.5f * (kh[lidx(k, c1, ncells)] + kh[lidx(k, c2, ncells)]) * pr_scale;
            tend = tend + flux;
        }
        delsq_theta[lidx(k, cell, ncells)] = dsq;
        tend_theta[lidx(k, cell, ncells)] = tend;
    }
}

extern "C" __global__ void les_theta_lap4_v841_f32(
    const float *delsq_theta, const float *scale4, const int *edges_on_cell,
    const int *n_edges_on_cell, const int *cells_on_edge, const float *dc_edge,
    const float *dv_edge, const float *area_cell, const float h4,
    const int nlev, const int ncells, const int nedges, const int max_edges,
    float *tend_theta)
{
    LES_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const float prandtl_inv = 1.0f / LES_PRANDTL;
        const float r_area = h4 * prandtl_inv * (1.0f / area_cell[cell]);
        float tend = tend_theta[lidx(k, cell, ncells)];
        const int count = n_edges_on_cell[cell];
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            if (edge < 0 || edge >= nedges || !(dc_edge[edge] > 0.0f)) continue;
            const int c1 = cells_on_edge[2 * edge], c2 = cells_on_edge[2 * edge + 1];
            const float edge_sign = scale4[edge] * r_area * dv_edge[edge]
                * cell_sign(cell, edge, cells_on_edge) * (1.0f / dc_edge[edge]);
            tend = tend - edge_sign * (delsq_theta[lidx(k, c2, ncells)] - delsq_theta[lidx(k, c1, ncells)]);
        }
        tend_theta[lidx(k, cell, ncells)] = tend;
    }
}

__device__ __forceinline__ float les_theta_interior_flux(
    int k, int cell, int model, const float *theta_m, const float *rho_zz,
    const float *zz, const float *kv, const float *pr3d, const float *rdzu,
    const float *fzm, const float *fzp, int ncells)
{
    const int i = lidx(k, cell, ncells);
    const int b = lidx(k - 1, cell, ncells);
    const float pr1d = model == 1 ? 1.0f / LES_PRANDTL
                                  : fzm[k] * pr3d[i] + fzp[k] * pr3d[b];
    const float rho_k_at_w = fzm[k] * rho_zz[i] * zz[i] * zz[i] * kv[i]
                           + fzp[k] * rho_zz[b] * zz[b] * zz[b] * kv[b];
    const float zz_at_w = fzm[k] * zz[i] + fzp[k] * zz[b];
    return -pr1d * rho_k_at_w * zz_at_w * rdzu[k] * (theta_m[i] - theta_m[b]);
}

/* scalar_dissipation_3d_les theta vertical flux and surface fluxes. */
extern "C" __global__ void les_theta_vertical_v841_f32(
    const float *theta_m, const float *rho_zz, const float *zz, const float *kv,
    const float *pr3d, const float *rdzu, const float *rdzw, const float *fzm,
    const float *fzp, const float *qv, const int model, const int surface,
    const float heat_flux_spec, const float moisture_flux_spec,
    const float *hfx, const float *qfx, const float rvord,
    const int nlev, const int ncells, float *tend_theta)
{
    LES_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        float flux_lo, flux_hi;
        if (k == 0) {
            if (surface == 1 || surface == 2) {
                float heat, moist;
                const float rho0 = rho_zz[lidx(0, cell, ncells)];
                if (surface == 1) { heat = heat_flux_spec; moist = moisture_flux_spec; }
                else { heat = hfx[cell] / rho0 / LES_CP; moist = qfx[cell] / rho0; }
                const float qv0 = qv[lidx(0, cell, ncells)];
                const float theta_cell = theta_m[lidx(0, cell, ncells)] / (1.0f + rvord * qv0);
                const float theta_m_flux = heat * (1.0f + rvord * qv0) + rvord * theta_cell * moist;
                flux_lo = theta_m_flux * rho0;
            } else {
                flux_lo = nlev > 1 ? les_theta_interior_flux(1, cell, model, theta_m, rho_zz, zz, kv,
                                                           pr3d, rdzu, fzm, fzp, ncells) : 0.0f;
            }
        } else {
            flux_lo = les_theta_interior_flux(k, cell, model, theta_m, rho_zz, zz, kv, pr3d,
                                              rdzu, fzm, fzp, ncells);
        }
        const int kk = (k + 1 == nlev) ? nlev - 1 : k + 1;
        if (kk == 0) flux_hi = flux_lo;
        else flux_hi = les_theta_interior_flux(kk, cell, model, theta_m, rho_zz, zz, kv, pr3d,
                                               rdzu, fzm, fzp, ncells);
        const int index = lidx(k, cell, ncells);
        tend_theta[index] = tend_theta[index] - rdzw[k] * (flux_hi - flux_lo);
    }
}

/* les_v841.advance_tke_v841 (port-local transport). */
extern "C" __global__ void les_tke_advance_v841_f32(
    const float *tke, const float *tend_tke, const float *rho_zz,
    const float *rho_u, const float *rho_w, const float *rdzw,
    const int *edges_on_cell, const int *n_edges_on_cell, const int *cells_on_edge,
    const float *dv_edge, const float *area_cell, const float dt,
    const int nlev, const int ncells, const int nedges, const int max_edges,
    float *tke_out)
{
    LES_ELEMENT_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int index = lidx(k, cell, ncells);
        const float e = tke[index];
        const float inv_ac = 1.0f / area_cell[cell];
        float adv = 0.0f;
        const int count = n_edges_on_cell[cell];
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            if (edge < 0 || edge >= nedges) continue;
            const int c1 = cells_on_edge[2 * edge], c2 = cells_on_edge[2 * edge + 1];
            const int neighbour = c1 == cell ? c2 : c1;
            const float outward = cell_sign(cell, edge, cells_on_edge)
                * rho_u[lidx(k, edge, nedges)] * dv_edge[edge] * inv_ac;
            const float inflow = fmaxf(-outward, 0.0f);
            adv = adv + inflow * (tke[lidx(k, neighbour, ncells)] - e);
        }
        if (k > 0) {
            const float up = fmaxf(rho_w[lidx(k, cell, ncells)], 0.0f);
            adv = adv + rdzw[k] * up * (tke[lidx(k - 1, cell, ncells)] - e);
        }
        if (k < nlev - 1) {
            const float down = fmaxf(-rho_w[lidx(k + 1, cell, ncells)], 0.0f);
            adv = adv + rdzw[k] * down * (tke[lidx(k + 1, cell, ncells)] - e);
        }
        const float updated = e + dt * (tend_tke[index] + adv) / rho_zz[index];
        tke_out[index] = fmaxf(0.0f, updated);
    }
}
"""
)

KERNEL_NAMES: tuple[str, ...] = (
    "les_reconstruct_v841_f32",
    "les_mesh_scaling_v841_f32",
    "les_n2_v841_f32",
    "les_models_v841_f32",
    "les_u_lap2_v841_f32",
    "les_delsq_vort_v841_f32",
    "les_delsq_div_v841_f32",
    "les_u_lap4_v841_f32",
    "les_u_vertical_v841_f32",
    "les_w_lap2_v841_f32",
    "les_w_lap4_v841_f32",
    "les_w_vertical_v841_f32",
    "les_theta_lap2_v841_f32",
    "les_theta_lap4_v841_f32",
    "les_theta_vertical_v841_f32",
    "les_tke_advance_v841_f32",
)

_SURFACE_CODE = {LES_SURFACE_NONE: 0, LES_SURFACE_SPECIFIED: 1, LES_SURFACE_VARYING: 2}
_MODEL_CODE = {LES_MODEL_3D_SMAGORINSKY: 1, LES_MODEL_PROGNOSTIC_15_ORDER: 2}


def _cupy() -> Any:
    import cupy as cp

    return cp


@dataclass
class CudaLesResultV841:
    """One RK-step-1 evaluation; ``kdiff`` is the LES horizontal viscosity."""

    kdiff: Any
    eddy_visc_vert: Any
    prandtl_3d_inv: Any
    bn2: Any
    h_mom_eddy_visc4: np.float32
    h_theta_eddy_visc4: np.float32
    tend_u_euler: Any
    tend_w_euler: Any
    tend_theta_euler: Any
    delsq_u: Any
    delsq_divergence: Any
    delsq_vorticity: Any
    delsq_w: Any
    delsq_theta: Any
    tend_tke: Any | None


@dataclass
class LesSurfaceDeviceFields:
    """Device ``ustm``/``hfx``/``qfx`` (nCells) for ``les_surface='varying'``."""

    ustm: Any
    hfx: Any
    qfx: Any


class CudaLesV841:
    """Device state and launches for one LES configuration on one mesh."""

    def __init__(
        self,
        *,
        config: LesV841Config,
        nlev: int,
        dims: Mapping[str, int],
        mesh: Mapping[str, Any],
        vertical: Mapping[str, Any],
        pressure_base: Any,
        kernel_cache: Any | None = None,
        initial_tke: float = DEFAULT_COLD_START_TKE,
        surface: LesSurfaceDeviceFields | None = None,
    ) -> None:
        config.validate()
        cp = _cupy()
        self.cp = cp
        self.config = config
        self.nlev = int(nlev)
        if self.nlev < 3:
            raise ValueError("the LES closure needs at least three levels")
        self.ncells = int(dims["ncells"])
        self.nedges = int(dims["nedges"])
        self.nvertices = int(dims["nvertices"])
        self.max_edges = int(dims["max_edges"])
        self.vertex_degree = int(dims.get("vertex_degree", 3))
        self.coeff_slots = int(dims.get("coeff_slots", self.max_edges))
        self.mesh = {name: cp.ascontiguousarray(value) for name, value in mesh.items()}
        self.vertical = {
            name: cp.ascontiguousarray(value) for name, value in vertical.items()
        }
        self.pressure_base = cp.ascontiguousarray(pressure_base)
        self.kernel_cache = kernel_cache
        self._kernels: dict[str, Any] = {}
        self._module: Any | None = None
        if config.les_surface == LES_SURFACE_VARYING and surface is None:
            raise ConfigurationRefusal(
                "config_les_surface",
                LES_SURFACE_VARYING,
                "the varying LES surface reads ustm/hfx/qfx from a surface "
                "layer and no provider was supplied",
                "config_les_surface='none' or 'specified'",
            )
        self.surface = surface
        self._dummy_cell = cp.zeros((self.ncells,), dtype=cp.float32)
        self._zeros_level = cp.zeros((self.nlev, self.ncells), dtype=cp.float32)
        if "scale_del2" not in self.mesh:
            self.mesh["scale_del2"] = cp.empty((self.nedges,), dtype=cp.float32)
            self.mesh["scale_del4"] = cp.empty((self.nedges,), dtype=cp.float32)
            self._launch(
                "les_mesh_scaling_v841_f32",
                self.nedges,
                (
                    self.mesh["cells_on_edge"],
                    self.mesh["mesh_density"],
                    np.int32(self.nedges),
                    np.int32(bool(dims.get("scale_with_mesh", 1))),
                    self.mesh["scale_del2"],
                    self.mesh["scale_del4"],
                ),
            )
        self.tke: Any | None = None
        self.initial_tke = float(initial_tke)
        if config.prognostic:
            if not np.isfinite(self.initial_tke) or self.initial_tke <= 0.0:
                raise ConfigurationRefusal(
                    "les_initial_tke",
                    initial_tke,
                    "with tke=0 the 1.5-order closure has K=0 and no shear "
                    "production, so it would run as no closure at all",
                    "a positive cold-start TKE (m2 s-2)",
                )
            self.tke = cp.full(
                (self.nlev, self.ncells), np.float32(self.initial_tke), dtype=cp.float32
            )
        # Per-step carry: reconstructed step-start winds and the TKE source.
        self._ur: Any | None = None
        self._vr: Any | None = None
        self.pending_tend_tke: Any | None = None
        self.last: CudaLesResultV841 | None = None
        self.calls = 0
        self.tke_updates = 0

    # -- compilation --------------------------------------------------------
    def _kernel(self, name: str) -> Any:
        kernel = self._kernels.get(name)
        if kernel is None:
            if self.kernel_cache is not None:
                kernel = self.kernel_cache.raw_kernel(
                    name, CUDA_LES_SOURCE, module_key=MODULE_KEY
                )
            else:
                if self._module is None:
                    self._module = self.cp.RawModule(
                        code=CUDA_LES_SOURCE, options=COMPILE_OPTIONS
                    )
                kernel = self._module.get_function(name)
            self._kernels[name] = kernel
        return kernel

    def _launch(self, name: str, total: int, args: tuple[Any, ...]) -> None:
        if total < 1:
            return
        threads = 128
        blocks = (int(total) + threads - 1) // threads
        self._kernel(name)((blocks,), (threads,), args)

    # -- pieces ---------------------------------------------------------------
    def reconstruct(self, u: Any) -> tuple[Any, Any]:
        cp = self.cp
        ur = cp.empty((self.nlev, self.ncells), dtype=cp.float32)
        vr = cp.empty_like(ur)
        m = self.mesh
        self._launch(
            "les_reconstruct_v841_f32",
            self.nlev * self.ncells,
            (
                cp.ascontiguousarray(u),
                m["edges_on_cell"],
                m["n_edges_on_cell"],
                m["coeffs_reconstruct"],
                m["lat_cell"],
                m["lon_cell"],
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(self.max_edges),
                np.int32(self.coeff_slots),
                ur,
                vr,
            ),
        )
        return ur, vr

    def compute(
        self,
        *,
        u: Any,
        v: Any,
        w: Any,
        theta_m: Any,
        rho_edge: Any,
        rho_zz: Any,
        divergence: Any,
        vorticity: Any,
        exner: Any,
        pressure_p: Any,
        dt: float,
        qv: Any | None = None,
        qc: Any | None = None,
        qtot: Any | None = None,
        dynamics_substep: int = 1,
        ur_cell: Any | None = None,
        vr_cell: Any | None = None,
    ) -> CudaLesResultV841:
        """``calculate_n2`` -> ``les_models`` -> u/w/theta, all on device."""

        cp = self.cp
        cfg = self.config
        nlev, ncells, nedges, nvertices = self.nlev, self.ncells, self.nedges, self.nvertices
        m = self.mesh
        vt = self.vertical
        timestep = float(dt)
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("dt must be finite and positive")
        u = cp.ascontiguousarray(u)
        v = cp.ascontiguousarray(v)
        w = cp.ascontiguousarray(w)
        theta_m = cp.ascontiguousarray(theta_m)
        rho_edge = cp.ascontiguousarray(rho_edge)
        rho_zz = cp.ascontiguousarray(rho_zz)
        divergence = cp.ascontiguousarray(divergence)
        vorticity = cp.ascontiguousarray(vorticity)
        if ur_cell is None or vr_cell is None:
            if int(dynamics_substep) == 1 or self._ur is None:
                self._ur, self._vr = self.reconstruct(u)
            ur_cell, vr_cell = self._ur, self._vr
        qv_arg = self._zeros_level if qv is None else cp.ascontiguousarray(qv)
        qc_arg = self._zeros_level if qc is None else cp.ascontiguousarray(qc)
        qtot_arg = self._zeros_level if qtot is None else cp.ascontiguousarray(qtot)
        rvord = np.float32(np.float32(RV) / np.float32(RGAS))

        bn2 = cp.empty((nlev, ncells), dtype=cp.float32)
        self._launch(
            "les_n2_v841_f32",
            nlev * ncells,
            (
                theta_m,
                cp.ascontiguousarray(exner),
                self.pressure_base,
                cp.ascontiguousarray(pressure_p),
                vt["zgrid"],
                qv_arg,
                qc_arg,
                qtot_arg,
                np.int32(qc is not None),
                rvord,
                np.int32(nlev),
                np.int32(ncells),
                bn2,
            ),
        )

        length = np.float32(cfg.len_disp)
        inv_dt = np.float32(np.float32(1.0) / np.float32(timestep))
        h4 = np.float32(np.float32(cfg.visc4_2dsmag) * ((length * length) * length))
        kh = cp.empty((nlev, ncells), dtype=cp.float32)
        kv = cp.empty_like(kh)
        pr3d = cp.empty_like(kh)
        write_tend = cfg.prognostic and int(dynamics_substep) == 1
        tend_tke = cp.zeros_like(kh) if write_tend else None
        tke_arg = self.tke if self.tke is not None else self._zeros_level
        self._launch(
            "les_models_v841_f32",
            nlev * ncells,
            (
                u,
                v,
                ur_cell,
                vr_cell,
                w,
                bn2,
                vt["zgrid"],
                rho_zz,
                m["coef_c2"],
                m["coef_s2"],
                m["coef_cs"],
                m["coef_c"],
                m["coef_s"],
                m["edges_on_cell"],
                m["n_edges_on_cell"],
                m["cells_on_edge"],
                np.float32(cfg.smagorinsky_coef),
                length,
                inv_dt,
                np.int32(_MODEL_CODE[cfg.les_model]),
                np.int32(bool(write_tend)),
                np.int32(nlev),
                np.int32(ncells),
                np.int32(nedges),
                np.int32(self.max_edges),
                tke_arg,
                kh,
                kv,
                pr3d,
                tend_tke if tend_tke is not None else self._zeros_level,
            ),
        )

        delsq_u = cp.empty((nlev, nedges), dtype=cp.float32)
        tend_u = cp.empty_like(delsq_u)
        self._launch(
            "les_u_lap2_v841_f32",
            nlev * nedges,
            (
                divergence,
                vorticity,
                kh,
                rho_edge,
                m["scale_del2"],
                m["cells_on_edge"],
                m["vertices_on_edge"],
                m["dc_edge"],
                m["dv_edge"],
                np.int32(nlev),
                np.int32(ncells),
                np.int32(nedges),
                np.int32(nvertices),
                delsq_u,
                tend_u,
            ),
        )
        delsq_div = cp.zeros((nlev, ncells), dtype=cp.float32)
        delsq_vort = cp.zeros((nlev, nvertices), dtype=cp.float32)
        if h4 > np.float32(0.0):
            self._launch(
                "les_delsq_vort_v841_f32",
                nlev * nvertices,
                (
                    delsq_u,
                    m["edges_on_vertex"],
                    m["vertices_on_edge"],
                    m["dc_edge"],
                    m["area_triangle"],
                    np.int32(nlev),
                    np.int32(nedges),
                    np.int32(nvertices),
                    np.int32(self.vertex_degree),
                    delsq_vort,
                ),
            )
            self._launch(
                "les_delsq_div_v841_f32",
                nlev * ncells,
                (
                    delsq_u,
                    m["edges_on_cell"],
                    m["n_edges_on_cell"],
                    m["cells_on_edge"],
                    m["dv_edge"],
                    m["area_cell"],
                    np.int32(nlev),
                    np.int32(ncells),
                    np.int32(nedges),
                    np.int32(self.max_edges),
                    delsq_div,
                ),
            )
            self._launch(
                "les_u_lap4_v841_f32",
                nlev * nedges,
                (
                    rho_edge,
                    delsq_div,
                    delsq_vort,
                    m["scale_del4"],
                    m["cells_on_edge"],
                    m["vertices_on_edge"],
                    m["dc_edge"],
                    m["dv_edge"],
                    h4,
                    np.float32(cfg.del4u_div_factor),
                    np.int32(nlev),
                    np.int32(ncells),
                    np.int32(nedges),
                    np.int32(nvertices),
                    tend_u,
                ),
            )
        surface_code = _SURFACE_CODE[cfg.les_surface]
        ustm = self.surface.ustm if self.surface is not None else self._dummy_cell
        self._launch(
            "les_u_vertical_v841_f32",
            nlev * nedges,
            (
                u,
                v,
                rho_edge,
                rho_zz,
                vt["zz"],
                kv,
                vt["rdzu"],
                vt["rdzw"],
                vt["fzm"],
                vt["fzp"],
                m["cells_on_edge"],
                m["dc_edge"],
                np.int32(surface_code),
                np.float32(cfg.surface_drag_coefficient),
                ustm,
                np.int32(nlev),
                np.int32(ncells),
                np.int32(nedges),
                tend_u,
            ),
        )

        delsq_w = cp.empty((nlev, ncells), dtype=cp.float32)
        tend_w = cp.empty((nlev + 1, ncells), dtype=cp.float32)
        self._launch(
            "les_w_lap2_v841_f32",
            (nlev + 1) * ncells,
            (
                w,
                rho_edge,
                kh,
                m["scale_del2"],
                m["edges_on_cell"],
                m["n_edges_on_cell"],
                m["cells_on_edge"],
                m["dc_edge"],
                m["dv_edge"],
                m["area_cell"],
                np.int32(nlev),
                np.int32(ncells),
                np.int32(nedges),
                np.int32(self.max_edges),
                delsq_w,
                tend_w,
            ),
        )
        if h4 > np.float32(0.0):
            self._launch(
                "les_w_lap4_v841_f32",
                (nlev - 1) * ncells,
                (
                    delsq_w,
                    m["scale_del4"],
                    m["edges_on_cell"],
                    m["n_edges_on_cell"],
                    m["cells_on_edge"],
                    m["dc_edge"],
                    m["dv_edge"],
                    m["area_cell"],
                    h4,
                    np.int32(nlev),
                    np.int32(ncells),
                    np.int32(nedges),
                    np.int32(self.max_edges),
                    tend_w,
                ),
            )
        self._launch(
            "les_w_vertical_v841_f32",
            (nlev - 1) * ncells,
            (
                w,
                rho_zz,
                kv,
                vt["zz"],
                vt["rdzu"],
                vt["rdzw"],
                divergence,
                np.int32(nlev),
                np.int32(ncells),
                tend_w,
            ),
        )

        delsq_theta = cp.empty((nlev, ncells), dtype=cp.float32)
        tend_theta = cp.empty_like(delsq_theta)
        self._launch(
            "les_theta_lap2_v841_f32",
            nlev * ncells,
            (
                theta_m,
                rho_edge,
                kh,
                m["scale_del2"],
                m["edges_on_cell"],
                m["n_edges_on_cell"],
                m["cells_on_edge"],
                m["dc_edge"],
                m["dv_edge"],
                m["area_cell"],
                np.int32(nlev),
                np.int32(ncells),
                np.int32(nedges),
                np.int32(self.max_edges),
                delsq_theta,
                tend_theta,
            ),
        )
        if h4 > np.float32(0.0):
            self._launch(
                "les_theta_lap4_v841_f32",
                nlev * ncells,
                (
                    delsq_theta,
                    m["scale_del4"],
                    m["edges_on_cell"],
                    m["n_edges_on_cell"],
                    m["cells_on_edge"],
                    m["dc_edge"],
                    m["dv_edge"],
                    m["area_cell"],
                    h4,
                    np.int32(nlev),
                    np.int32(ncells),
                    np.int32(nedges),
                    np.int32(self.max_edges),
                    tend_theta,
                ),
            )
        hfx = self.surface.hfx if self.surface is not None else self._dummy_cell
        qfx = self.surface.qfx if self.surface is not None else self._dummy_cell
        self._launch(
            "les_theta_vertical_v841_f32",
            nlev * ncells,
            (
                theta_m,
                rho_zz,
                vt["zz"],
                kv,
                pr3d,
                vt["rdzu"],
                vt["rdzw"],
                vt["fzm"],
                vt["fzp"],
                qv_arg,
                np.int32(_MODEL_CODE[cfg.les_model]),
                np.int32(surface_code),
                np.float32(cfg.surface_heat_flux),
                np.float32(cfg.surface_moisture_flux),
                hfx,
                qfx,
                rvord,
                np.int32(nlev),
                np.int32(ncells),
                tend_theta,
            ),
        )
        result = CudaLesResultV841(
            kdiff=kh,
            eddy_visc_vert=kv,
            prandtl_3d_inv=pr3d,
            bn2=bn2,
            h_mom_eddy_visc4=h4,
            h_theta_eddy_visc4=h4,
            tend_u_euler=tend_u,
            tend_w_euler=tend_w,
            tend_theta_euler=tend_theta,
            delsq_u=delsq_u,
            delsq_divergence=delsq_div,
            delsq_vorticity=delsq_vort,
            delsq_w=delsq_w,
            delsq_theta=delsq_theta,
            tend_tke=tend_tke,
        )
        if tend_tke is not None:
            self.pending_tend_tke = tend_tke
        self.last = result
        self.calls += 1
        return result

    def advance_tke(self, *, rho_zz: Any, rho_u: Any, rho_w: Any, dt: float) -> None:
        """One model step of the TKE (see :func:`les_v841.advance_tke_v841`)."""

        if self.tke is None:
            return
        if self.pending_tend_tke is None:
            raise RuntimeError(
                "advance_tke called with no TKE source held from dynamics substep 1"
            )
        cp = self.cp
        out = cp.empty_like(self.tke)
        m = self.mesh
        self._launch(
            "les_tke_advance_v841_f32",
            self.nlev * self.ncells,
            (
                self.tke,
                self.pending_tend_tke,
                cp.ascontiguousarray(rho_zz),
                cp.ascontiguousarray(rho_u),
                cp.ascontiguousarray(rho_w),
                self.vertical["rdzw"],
                m["edges_on_cell"],
                m["n_edges_on_cell"],
                m["cells_on_edge"],
                m["dv_edge"],
                m["area_cell"],
                np.float32(dt),
                np.int32(self.nlev),
                np.int32(self.ncells),
                np.int32(self.nedges),
                np.int32(self.max_edges),
                out,
            ),
        )
        self.tke = out
        self.pending_tend_tke = None
        self.tke_updates += 1

    def summary(self, n_cells: int | None = None) -> dict[str, Any]:
        """Ranges of the last evaluation (host sync; call at receipts only)."""

        count = self.ncells if n_cells is None else int(n_cells)
        out: dict[str, Any] = {
            "les_model": self.config.les_model,
            "les_surface": self.config.les_surface,
            "label": les_label(self.config.les_model),
            "rk1_les_calls": int(self.calls),
            "tke_updates": int(self.tke_updates),
        }

        def _range(array: Any) -> dict[str, float]:
            host = self.cp.asnumpy(array[:, :count])
            return {
                "min": float(np.min(host)),
                "max": float(np.max(host)),
                "mean": float(np.mean(host)),
                "finite": bool(np.all(np.isfinite(host))),
            }

        if self.last is not None:
            out["eddy_visc_horz"] = _range(self.last.kdiff)
            out["eddy_visc_vert"] = _range(self.last.eddy_visc_vert)
        if self.tke is not None:
            out["tke"] = _range(self.tke)
            out["tke_cold_start_m2s2"] = self.initial_tke
            out["tke_cold_start"] = TKE_COLD_START_LABEL
            out["tke_transport"] = TKE_TRANSPORT_LABEL
        return out


# ---------------------------------------------------------------------------
# standalone construction (tests, synthetic patches)
# ---------------------------------------------------------------------------
def les_device_from_host(
    geom: Any,
    *,
    lat_cell: Any,
    lon_cell: Any,
    coeffs_reconstruct: Any,
    vertical: Mapping[str, Any],
    pressure_base: Any,
    config: LesV841Config,
    initial_tke: float = DEFAULT_COLD_START_TKE,
    surface: LesSurfaceDeviceFields | None = None,
) -> CudaLesV841:
    """Upload a :class:`les_v841.LesGeometryV841` and build the closure."""

    cp = _cupy()
    f32 = lambda a: cp.asarray(np.ascontiguousarray(np.asarray(a, dtype=np.float32)))  # noqa: E731
    i32 = lambda a: cp.asarray(np.ascontiguousarray(np.asarray(a, dtype=np.int32)))  # noqa: E731
    coeffs = np.asarray(coeffs_reconstruct, dtype=np.float32)
    mesh = {
        "cells_on_edge": i32(geom.cells_on_edge),
        "vertices_on_edge": i32(geom.vertices_on_edge),
        "edges_on_cell": i32(geom.edges_on_cell),
        "n_edges_on_cell": i32(geom.n_edges_on_cell),
        "edges_on_vertex": i32(geom.edges_on_vertex),
        "dc_edge": f32(geom.dc_edge),
        "dv_edge": f32(geom.dv_edge),
        "area_cell": f32(geom.area_cell),
        "area_triangle": f32(geom.area_triangle),
        "scale_del2": f32(geom.scale_del2),
        "scale_del4": f32(geom.scale_del4),
        "coef_c2": f32(geom.coef_c2),
        "coef_s2": f32(geom.coef_s2),
        "coef_cs": f32(geom.coef_cs),
        "coef_c": f32(geom.coef_c),
        "coef_s": f32(geom.coef_s),
        "lat_cell": f32(lat_cell),
        "lon_cell": f32(lon_cell),
        "coeffs_reconstruct": f32(coeffs),
    }
    nlev = int(np.asarray(vertical["zz"]).shape[0])
    return CudaLesV841(
        config=config,
        nlev=nlev,
        dims={
            "ncells": geom.n_cells,
            "nedges": geom.n_edges,
            "nvertices": geom.n_vertices,
            "max_edges": geom.max_edges,
            "vertex_degree": int(np.asarray(geom.edges_on_vertex).shape[1]),
            "coeff_slots": int(coeffs.shape[1]),
        },
        mesh=mesh,
        vertical={name: f32(vertical[name]) for name in ("zgrid", "zz", "rdzu", "rdzw", "fzm", "fzp")},
        pressure_base=f32(pressure_base),
        initial_tke=initial_tke,
        surface=surface,
    )


# ---------------------------------------------------------------------------
# forecast attachment
# ---------------------------------------------------------------------------
@dataclass
class LesAttachmentV841:
    """What :func:`attach_les_v841` rebinds, and the closure it drives."""

    driver: Any
    closure: CudaLesV841
    original_mixing: Any
    original_subcycle: Any
    scalar_index_qv: int | None
    scalar_index_qc: int | None
    n_cells_real: int
    dynamics_splits: int
    mixing_was_instance: bool = False
    subcycle_was_instance: bool = False
    #: ``driver._step_device_v841`` when the driver has one (the real CUDA
    #: driver); the TKE then advances after the whole step returns.
    original_step: Any | None = None
    step_was_instance: bool = False
    _substep: int = 0
    _step_start: dict[str, Any] = field(default_factory=dict)
    _current: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        out = self.closure.summary(self.n_cells_real)
        out["attachment"] = (
            "woof.hex.cuda_les_v841.attach_les_v841 rebinds "
            "horizontal.compute_dry_mixing_tendencies_v841 and wraps "
            "_advance_dynamics_subcycle_v841 and _step_device_v841; no pinned "
            "source byte changes"
        )
        return out


def _pad_rows(array: np.ndarray, rows: int) -> np.ndarray:
    if array.shape[0] == rows:
        return array
    if array.shape[0] + 1 == rows:
        pad = np.zeros((1,) + array.shape[1:], dtype=array.dtype)
        return np.concatenate([array, pad], axis=0)
    raise ValueError(f"host table has {array.shape[0]} rows, device has {rows}")


def attach_les_v841(
    driver: Any,
    *,
    host_mesh: object,
    scalar_names: Sequence[str] = (),
    initial_tke: float = DEFAULT_COLD_START_TKE,
    surface: LesSurfaceDeviceFields | None = None,
) -> LesAttachmentV841 | None:
    """Attach the LES closure to a built v8.4.1 CUDA driver, or do nothing.

    ``host_mesh`` is the unpadded host mesh the driver was built from (the
    gradient weights and reconstruction coefficients are read from it and,
    on a limited-area driver, padded by native's zero garbage row).
    """

    config = driver.config
    model = str(getattr(config, "config_les_model", LES_MODEL_NONE))
    if model == LES_MODEL_NONE:
        return None
    if getattr(driver, "v841_context", None) is None:
        raise ConfigurationRefusal(
            "config_les_model", model, "the LES closure is a v8.4.1 lane",
            "a v8.4.1 CUDA driver",
        )
    if getattr(driver, "mixing_config_v841", None) is None:
        raise ConfigurationRefusal(
            "config_les_model",
            model,
            "the LES closure enters at the explicit-mixing call and this "
            "driver makes none (config_horiz_mixing is not 2d_smagorinsky)",
            "--horiz-mixing 2d_smagorinsky with --les-model",
        )
    if int(config.config_dynamics_split_steps) != 3:
        raise ConfigurationRefusal(
            "config_dynamics_split_steps",
            config.config_dynamics_split_steps,
            "the substep bookkeeping is proved for split-three only",
            "config_dynamics_split_steps=3",
        )
    if getattr(driver, "halo_exchanger_v841", None) is not None:
        raise ConfigurationRefusal(
            "config_les_model",
            model,
            "the partitioned two-rank driver has no halo exchange for the "
            "LES viscosities or the TKE",
            "a single-device run",
        )
    deformation = getattr(driver.horizontal, "_deformation_v841", None)
    if deformation is None:
        raise RuntimeError("v8.4.1 deformation weights are not attached")
    horizontal = driver.horizontal
    dmesh = horizontal.mesh
    configured = float(config.config_len_disp)
    length = float(dmesh.nominal_min_dc) if configured == 0.0 else configured
    les_cfg = les_config_from_dycore(config, len_disp=length)
    cp = _cupy()
    gradient = initialize_les_gradient_weights_v841(host_mesh, dtype=np.float32)
    arrays = getattr(host_mesh, "arrays", {})
    coeffs = np.asarray(arrays["coeffs_reconstruct"], dtype=np.float32)
    ncells = int(horizontal.ncells)
    mesh = {
        "cells_on_edge": dmesh.cells_on_edge,
        "vertices_on_edge": dmesh.vertices_on_edge,
        "edges_on_cell": dmesh.edges_on_cell,
        "n_edges_on_cell": dmesh.n_edges_on_cell,
        "edges_on_vertex": dmesh.edges_on_vertex,
        "dc_edge": dmesh.dc_edge,
        "dv_edge": dmesh.dv_edge,
        "area_cell": dmesh.area_cell,
        "area_triangle": dmesh.area_triangle,
        "lat_cell": dmesh.lat_cell,
        "lon_cell": dmesh.lon_cell,
        "mesh_density": dmesh.mesh_density,
        "coef_c2": deformation["coef_c2"],
        "coef_s2": deformation["coef_s2"],
        "coef_cs": deformation["coef_cs"],
        "coef_c": cp.asarray(_pad_rows(gradient.coef_c, ncells)),
        "coef_s": cp.asarray(_pad_rows(gradient.coef_s, ncells)),
        "coeffs_reconstruct": cp.asarray(np.ascontiguousarray(_pad_rows(coeffs, ncells))),
    }
    vertical = driver.atmosphere.vertical
    closure = CudaLesV841(
        config=les_cfg,
        nlev=int(horizontal.nlev),
        dims={
            "ncells": ncells,
            "nedges": int(horizontal.nedges),
            "nvertices": int(horizontal.nvertices),
            "max_edges": int(horizontal.max_edges),
            "vertex_degree": int(horizontal.vertex_degree),
            "coeff_slots": int(coeffs.shape[1]),
            "scale_with_mesh": int(bool(config.config_h_ScaleWithMesh)),
        },
        mesh=mesh,
        vertical={
            "zgrid": vertical.zgrid,
            "zz": vertical.zz,
            "rdzu": vertical.rdzu,
            "rdzw": vertical.rdzw,
            "fzm": vertical.fzm,
            "fzp": vertical.fzp,
        },
        pressure_base=driver.atmosphere.reference.pressure_base,
        kernel_cache=getattr(driver, "cache", None),
        initial_tke=initial_tke,
        surface=surface,
    )
    names = [str(name) for name in scalar_names]
    attachment = LesAttachmentV841(
        driver=driver,
        closure=closure,
        original_mixing=horizontal.compute_dry_mixing_tendencies_v841,
        original_subcycle=driver._advance_dynamics_subcycle_v841,
        scalar_index_qv=names.index("qv") if "qv" in names else None,
        scalar_index_qc=names.index("qc") if "qc" in names else None,
        n_cells_real=int(np.asarray(arrays["nEdgesOnCell"]).size),
        dynamics_splits=int(config.config_dynamics_split_steps),
        mixing_was_instance="compute_dry_mixing_tendencies_v841" in getattr(horizontal, "__dict__", {}),
        subcycle_was_instance="_advance_dynamics_subcycle_v841" in getattr(driver, "__dict__", {}),
        original_step=getattr(driver, "_step_device_v841", None),
        step_was_instance="_step_device_v841" in getattr(driver, "__dict__", {}),
    )

    def subcycle(state: Any, **kwargs: Any) -> Any:
        substep = attachment._substep % attachment.dynamics_splits + 1
        if substep == 1:
            attachment._step_start = {
                "rho": state.rho,
                "rho_u": state.rho_u,
                "rho_w": state.rho_w,
            }
        attachment._current = {
            "substep": substep,
            "state": state,
            "saved": kwargs.get("time_level_one"),
            "moist": kwargs.get("moist_coefficients"),
        }
        result = attachment.original_subcycle(state, **kwargs)
        attachment._substep += 1
        if (
            attachment.original_step is None
            and substep == attachment.dynamics_splits
        ):
            advance_tke()
        return result

    def advance_tke() -> None:
        if closure.tke is None or closure.pending_tend_tke is None:
            return
        start = attachment._step_start
        closure.advance_tke(
            rho_zz=start["rho"],
            rho_u=start["rho_u"],
            rho_w=start["rho_w"],
            dt=float(driver.config.config_dt),
        )

    def step(*args: Any, **kwargs: Any) -> Any:
        # One outer step: the substep count restarts here, so a step that
        # raised part-way cannot leave the next one out of phase, and the
        # TKE advances only once the whole step has returned (a refused step
        # leaves it where it was, like the state it belongs to).
        attachment._substep = 0
        attachment._current = {}
        closure.pending_tend_tke = None
        result = attachment.original_step(*args, **kwargs)
        advance_tke()
        return result

    def mixing(
        normal_velocity: Any,
        tangential_velocity: Any,
        vertical_velocity: Any,
        theta_m: Any,
        rho_edge: Any,
        divergence: Any,
        vorticity: Any,
        *,
        dt: float,
        config: Any | None = None,
    ) -> Any:
        from .cuda_horizontal import CudaDryMixingTendencies

        current = attachment._current
        if not current:
            raise RuntimeError(
                "the LES mixing call arrived outside a wrapped dynamics subcycle"
            )
        state = current["state"]
        saved = current["saved"]
        moist = current["moist"]
        scalars = getattr(state, "scalars", None)
        qv = qc = None
        if scalars is not None and attachment.scalar_index_qv is not None:
            qv = scalars[attachment.scalar_index_qv]
        if scalars is not None and attachment.scalar_index_qc is not None:
            qc = scalars[attachment.scalar_index_qc]
        result = closure.compute(
            u=normal_velocity,
            v=tangential_velocity,
            w=vertical_velocity,
            theta_m=theta_m,
            rho_edge=rho_edge,
            rho_zz=state.rho,
            divergence=divergence,
            vorticity=vorticity,
            exner=saved.exner,
            pressure_p=saved.pressure_perturbation,
            dt=dt,
            qv=qv,
            qc=qc,
            qtot=None if moist is None else moist.qtot,
            dynamics_substep=int(current["substep"]),
        )
        return CudaDryMixingTendencies(
            kdiff=result.kdiff,
            h_mom_eddy_visc4=result.h_mom_eddy_visc4,
            h_theta_eddy_visc4=result.h_theta_eddy_visc4,
            tend_u_euler=result.tend_u_euler,
            tend_w_euler=result.tend_w_euler,
            tend_theta_euler=result.tend_theta_euler,
            delsq_u=result.delsq_u,
            delsq_divergence=result.delsq_divergence,
            delsq_vorticity=result.delsq_vorticity,
            delsq_w=result.delsq_w,
            delsq_theta=result.delsq_theta,
        )

    horizontal.compute_dry_mixing_tendencies_v841 = mixing
    driver._advance_dynamics_subcycle_v841 = subcycle
    if attachment.original_step is not None:
        driver._step_device_v841 = step
    driver.les_v841 = attachment
    return attachment


def detach_les_v841(attachment: LesAttachmentV841) -> None:
    """Put both call targets back."""

    driver = attachment.driver
    if attachment.mixing_was_instance:
        driver.horizontal.compute_dry_mixing_tendencies_v841 = attachment.original_mixing
    else:
        del driver.horizontal.compute_dry_mixing_tendencies_v841
    if attachment.subcycle_was_instance:
        driver._advance_dynamics_subcycle_v841 = attachment.original_subcycle
    else:
        del driver._advance_dynamics_subcycle_v841
    if attachment.original_step is not None:
        if attachment.step_was_instance:
            driver._step_device_v841 = attachment.original_step
        else:
            del driver._step_device_v841
    driver.les_v841 = None


def stamp_les_history(path: Any, attachment: LesAttachmentV841 | None, *, les_model: str) -> None:
    """Label a written history file ``les_model=...`` and add the LES fields."""

    from netCDF4 import Dataset

    with Dataset(path, "a") as dataset:
        dataset.setncattr("les_model", str(les_model))
        dataset.setncattr("les_label", les_label(les_model))
        if attachment is None:
            return
        closure = attachment.closure
        dataset.setncattr("les_surface", closure.config.les_surface)
        count = attachment.n_cells_real
        dims = ("Time", "nCells", "nVertLevels")
        fields: dict[str, tuple[Any, str]] = {}
        if closure.last is not None:
            fields["les_eddy_visc_horz"] = (closure.last.kdiff, "m^2 s^{-1}")
            fields["les_eddy_visc_vert"] = (closure.last.eddy_visc_vert, "m^2 s^{-1}")
        if closure.tke is not None:
            fields["tke"] = (closure.tke, "m^2 s^{-2}")
            dataset.setncattr("tke_transport", TKE_TRANSPORT_LABEL)
            dataset.setncattr("tke_cold_start", TKE_COLD_START_LABEL)
        for name, (array, units) in fields.items():
            host = closure.cp.asnumpy(array[:, :count])
            if name in dataset.variables:
                continue
            variable = dataset.createVariable(name, "f4", dims, zlib=True, complevel=1)
            variable[:] = host.T[None]
            variable.setncattr("units", units)


__all__ = [
    "CUDA_LES_SOURCE",
    "CudaLesResultV841",
    "CudaLesV841",
    "KERNEL_NAMES",
    "LesAttachmentV841",
    "LesSurfaceDeviceFields",
    "MODULE_KEY",
    "attach_les_v841",
    "detach_les_v841",
    "les_device_from_host",
    "stamp_les_history",
]
