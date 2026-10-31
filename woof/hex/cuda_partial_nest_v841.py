"""CUDA entry points of the hex partial-depth band (Phase D5).

NEW entry points only, in a new translation unit (module key
``hexcore.cuda_partial_nest_v841``): no byte of the dycore's units -- the
regional kernel set :data:`woof.hex.cuda_backend.regional_admission.
REGIONAL_KERNEL_SOURCES` -- changes, and nothing here runs in a configuration
without a partial band.  Every floating operation goes through
``cuda_fp32.CUDA_FTZ_HELPERS`` (``mpas_add`` ...), the helpers every hex unit
uses, so in strict numerics (and on Metal, which never contracts) each word is
the arithmetic written here; binary64 accumulations convert through the
helpers' IEEE paths.

Indexing follows the dycore: ``C2(k, cell, ncells)``, ``E2(k, edge, nedges)``,
with ``nlev`` the CHILD's layer count, so face ``nlev`` is the interface.

The entry points (woof/hex/partial_nest_run.py drives them):

* the interface in the acoustic step -- ``pn_coefficients_top_v841`` (the
  implicit system's interface row: ``coftz(nlev)`` from the parent's theta
  above, and ``c_tri``/``gamma_tri`` of the last interior face, in
  ``acoustic_coefficients_v841``'s statements) and ``pn_column_top_v841``
  (before the column solve: the specified ``rw_p`` at the interface and its
  share of ``ww_avg``);
* the interface in the big step and the scalar transport --
  ``pn_theta_flux_top_v841``, ``pn_u_flux_top_v841``, ``pn_w_flux_top_v841``,
  ``pn_scalar_flux_top_v841``: the flux at the interface face (the parent's
  second-order face value by default) and at the face below it (the parent's
  own third-order interior flux, whose stencil now reaches the parent's level
  above), written into the flux arrays before their finish kernels read them;
* ``pn_recover_top_v841``: the recovered interface ``rw``, mass flux and w;
* coarse to fine -- ``pn_bary_apply_v841`` (MPAS
  ``mpas_cells_to_points_using_baryweights``), ``pn_reconstruct_v841``
  (``mpas_reconstruct_2d``'s zonal/meridional winds), ``pn_edge_normal_v841``
  (init_atmosphere's ``cos(angleEdge)*u + sin(angleEdge)*v``);
* fine to coarse -- ``pn_remap_rows_v841`` (the conservative overlap remap,
  binary64 in the reference's order), ``pn_feedback_delta_v841`` and
  ``pn_budget_residual_v841`` (the interface mass budget, one fixed-order
  binary64 sum);
* the top zone -- ``pn_topzone_{rayleigh,filter}_{cell,edge}_v841``: MPAS's
  relaxation-zone statements (``atm_bdy_adjust_dynamics_relaxzone_tend``) with
  the ring strength taken per LEVEL below the interface instead of per cell;
* ``pn_lerp_v841``: ``a + f*(b - a)``, the time interpolation across a parent
  step.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .cuda_fp32 import CUDA_FTZ_HELPERS

MODULE_KEY = "hexcore.cuda_partial_nest_v841"
THREADS = 128

CUDA_PARTIAL_NEST_SOURCE = CUDA_FTZ_HELPERS + r"""
#define C2(k,c,nc) ((size_t)(k)*(nc) + (c))
#define E2(k,e,ne) ((size_t)(k)*(ne) + (e))
#define PN_LOOP(total) \
    for (int element = blockDim.x * blockIdx.x + threadIdx.x; \
         element < (total); element += gridDim.x * blockDim.x)

__constant__ float pn_third_order_denominator = 12.0f;

static __device__ __forceinline__ float pn_lerp(float a, float b, float f)
{
    return mpas_add(a, mpas_mul(f, mpas_sub(b, a)));
}

/* The dycore's third-order vertical flux (vertical_u_flux_f32,
   theta_vertical_flux_f32, w_vertical_flux_f32, transport_vertical_flux):
   v*(7(qi+qm1) - (qp1+qm2))/12 + c|v|((qp1-qm2) - 3(qi-qm1))/12. */
static __device__ __forceinline__ float pn_flux3(
    float velocity, float coefficient, float qm2, float qm1, float qi, float qp1)
{
    return mpas_add(mpas_div(mpas_mul(velocity, mpas_sub(
            mpas_mul(7.0f, mpas_add(qi, qm1)), mpas_add(qp1, qm2))),
            pn_third_order_denominator),
        mpas_div(mpas_mul(mpas_mul(coefficient, mpas_abs(velocity)),
            mpas_sub(mpas_sub(qp1, qm2), mpas_mul(3.0f, mpas_sub(qi, qm1)))),
            pn_third_order_denominator));
}

/* The specified-top rows of the implicit acoustic system, launched right
   after acoustic_coefficients_v841 (whose statements these are, at the one
   face where the child's top differs from a lid):
     coftz(nlev) = fzm_top*theta_above + fzp_top*theta(nlev-1)
     c_tri(nlev-1) and gamma_tri(nlev-1) re-formed with that coftz.
   alpha_tri(nlev-1) does not depend on c_tri(nlev-1). */
extern "C" __global__ void pn_coefficients_top_v841(
    const int nlev, const int ncells, const float dts,
    const float fzm_top, const float fzp_top,
    const float *theta, const float *theta_above, const float *zz,
    const float *rdzw, const float *etp, const float *ewp,
    const float *cofwz, const float *cofwr, const float *cofwt,
    const float *cofrz, const float *alpha_tri,
    float *coftz, float *gamma_tri)
{
    PN_LOOP(ncells) {
        const int cell = element;
        const size_t top = C2(nlev, cell, ncells);
        const size_t i = C2(nlev - 1, cell, ncells);
        const int k = nlev - 1;
        const float face = mpas_add(mpas_mul(fzm_top, theta_above[cell]),
                                    mpas_mul(fzp_top, theta[i]));
        coftz[top] = face;
        float cvalue = mpas_sub(0.0f, mpas_mul(mpas_mul(mpas_mul(
            cofwz[i], face), rdzw[k]), zz[i]));
        cvalue = mpas_sub(cvalue, mpas_mul(cofwr[i], cofrz[k]));
        cvalue = mpas_add(cvalue, mpas_mul(mpas_mul(cofwt[i], face), rdzw[k]));
        cvalue = mpas_mul(cvalue, etp[k]);
        const float c_tri = mpas_mul(cvalue, ewp[k + 1]);
        const float dts2 = mpas_mul(dts, dts);
        gamma_tri[i] = mpas_mul(mpas_mul(dts2, c_tri), alpha_tri[i]);
    }
}

/* Before the column solve of each acoustic substep: rw_p at the interface
   becomes the specified perturbation at the substep's end,
     rw_p(nlev) = I(rw_a, rw_b, frac) - rw_save(nlev),
   and ww_avg(nlev) gains what a full column's solve adds there,
   ewm*rw_p(old) then ewp*rw_p(new) (acoustic_column_solve_v841's order).
   The rs/ts launch before it read rw_p(nlev) as the substep's OLD value. */
extern "C" __global__ void pn_column_top_v841(
    const int nlev, const int ncells, const float frac,
    const float *rw_a, const float *rw_b, const float *rw_save,
    const float *ewm, const float *ewp, float *rw_p, float *ww_avg)
{
    PN_LOOP(ncells) {
        const int cell = element;
        const size_t top = C2(nlev, cell, ncells);
        const float old_value = rw_p[top];
        const float new_value = mpas_sub(pn_lerp(rw_a[cell], rw_b[cell], frac),
                                         rw_save[top]);
        float average = mpas_add(ww_avg[top], mpas_mul(ewm[nlev], old_value));
        average = mpas_add(average, mpas_mul(ewp[nlev], new_value));
        ww_avg[top] = average;
        rw_p[top] = new_value;
    }
}

/* theta's vertical flux at the face below the interface (the parent's
   third-order interior flux plus its rw_save correction,
   theta_vertical_flux_f32) and at the interface itself:
     scheme 0: rw*(fzm_top*th_above + fzp_top*th(nlev-1))
     scheme 1: the parent's third-order flux with two levels above
     scheme 2: rw upwinded (D3's, to measure)
   each plus (rw_save - rw)*(fzm_top*ths_above + fzp_top*ths(nlev-1)). */
extern "C" __global__ void pn_theta_flux_top_v841(
    const int nlev, const int ncells, const float coefficient, const int scheme,
    const float fzm_top, const float fzp_top,
    const float *theta, const float *theta_saved,
    const float *rw, const float *rw_saved,
    const float *fzm, const float *fzp,
    const float *theta_above, const float *theta_above2,
    const float *theta_saved_above, float *flux)
{
    PN_LOOP(ncells) {
        const int cell = element;
        {
            const int k = nlev - 1;
            const size_t index = C2(k, cell, ncells);
            float value = pn_flux3(rw[index], coefficient,
                theta[C2(k - 2, cell, ncells)], theta[C2(k - 1, cell, ncells)],
                theta[index], theta_above[cell]);
            value = mpas_add(value, mpas_mul(
                mpas_sub(rw_saved[index], rw[index]), mpas_add(
                    mpas_mul(fzm[k], theta_saved[index]),
                    mpas_mul(fzp[k], theta_saved[C2(k - 1, cell, ncells)]))));
            flux[index] = value;
        }
        const size_t top = C2(nlev, cell, ncells);
        const float below = theta[C2(nlev - 1, cell, ncells)];
        const float velocity = rw[top];
        float value;
        if (scheme == 1) {
            value = pn_flux3(velocity, coefficient,
                theta[C2(nlev - 2, cell, ncells)], below, theta_above[cell],
                theta_above2[cell]);
        } else if (scheme == 2) {
            value = mpas_mul(velocity, velocity > 0.0f ? below : theta_above[cell]);
        } else {
            value = mpas_mul(velocity, mpas_add(mpas_mul(fzm_top, theta_above[cell]),
                                                mpas_mul(fzp_top, below)));
        }
        value = mpas_add(value, mpas_mul(mpas_sub(rw_saved[top], velocity), mpas_add(
            mpas_mul(fzm_top, theta_saved_above[cell]),
            mpas_mul(fzp_top, theta_saved[C2(nlev - 1, cell, ncells)]))));
        flux[top] = value;
    }
}

/* u's vertical flux (vertical_u_flux_f32) at the face below the interface
   (third order, the parent's u above) and at the interface. */
extern "C" __global__ void pn_u_flux_top_v841(
    const int nlev, const int ncells, const int nedges, const int scheme,
    const float fzm_top, const float fzp_top,
    const float *u, const float *rw, const int *cells_on_edge,
    const float *u_above, const float *u_above2, float *flux)
{
    PN_LOOP(nedges) {
        const int edge = element;
        const int c0 = cells_on_edge[2 * edge];
        const int c1 = cells_on_edge[2 * edge + 1];
        {
            const int k = nlev - 1;
            const float velocity = mpas_mul(0.5f, mpas_add(
                rw[C2(k, c0, ncells)], rw[C2(k, c1, ncells)]));
            flux[E2(k, edge, nedges)] = pn_flux3(velocity, 1.0f,
                u[E2(k - 2, edge, nedges)], u[E2(k - 1, edge, nedges)],
                u[E2(k, edge, nedges)], u_above[edge]);
        }
        const float velocity = mpas_mul(0.5f, mpas_add(
            rw[C2(nlev, c0, ncells)], rw[C2(nlev, c1, ncells)]));
        const float below = u[E2(nlev - 1, edge, nedges)];
        float value;
        if (scheme == 1) {
            value = pn_flux3(velocity, 1.0f, u[E2(nlev - 2, edge, nedges)], below,
                             u_above[edge], u_above2[edge]);
        } else if (scheme == 2) {
            value = mpas_mul(velocity, velocity > 0.0f ? below : u_above[edge]);
        } else {
            value = mpas_mul(velocity, mpas_add(mpas_mul(fzm_top, u_above[edge]),
                                                mpas_mul(fzp_top, below)));
        }
        flux[E2(nlev, edge, nedges)] = value;
    }
}

/* w's vertical flux (w_vertical_flux_f32; entry k is the half level k-1):
   entry nlev-1 third order with w(nlev) (the specified interface w), entry
   nlev (the child's top layer) second order in the dycore's own form, or
   the parent's third order with w at the face above. */
extern "C" __global__ void pn_w_flux_top_v841(
    const int nlev, const int ncells, const int scheme,
    const float *w, const float *rw, const float *w_above2, float *flux)
{
    PN_LOOP(ncells) {
        const int cell = element;
        {
            const int k = nlev - 1;
            const float velocity = mpas_mul(0.5f, mpas_add(
                rw[C2(k, cell, ncells)], rw[C2(k - 1, cell, ncells)]));
            flux[C2(k, cell, ncells)] = pn_flux3(velocity, 1.0f,
                w[C2(k - 2, cell, ncells)], w[C2(k - 1, cell, ncells)],
                w[C2(k, cell, ncells)], w[C2(k + 1, cell, ncells)]);
        }
        const int k = nlev;
        const float velocity = mpas_mul(0.5f, mpas_add(
            rw[C2(k, cell, ncells)], rw[C2(k - 1, cell, ncells)]));
        float value;
        if (scheme == 1) {
            value = pn_flux3(velocity, 1.0f, w[C2(k - 2, cell, ncells)],
                w[C2(k - 1, cell, ncells)], w[C2(k, cell, ncells)], w_above2[cell]);
        } else {
            value = mpas_mul(mpas_mul(0.5f, velocity), mpas_add(
                w[C2(k, cell, ncells)], w[C2(k - 1, cell, ncells)]));
        }
        flux[C2(k, cell, ncells)] = value;
    }
}

/* The scalars' vertical flux (transport_vertical_flux; QI3 layout) at the
   face below the interface (third order with the parent's q above) and at
   the interface: scheme 0 the second-order face value, an OUTGOING face
   (velocity > 0, upward out of the child) carrying at most the child's own
   top value (D4's cap); scheme 1 the parent's third order; scheme 2 upwind. */
extern "C" __global__ void pn_scalar_flux_top_v841(
    const int ntracers, const int nlev, const int ncells,
    const float coefficient, const int scheme,
    const float fzm_top, const float fzp_top,
    const float *stage, const float *velocity,
    const float *q_above, const float *q_above2, float *vertical_flux)
{
    PN_LOOP(ntracers * ncells) {
        const int cell = element % ncells;
        const int tracer = element / ncells;
        const size_t base = (size_t)tracer * nlev * ncells;
        const size_t fbase = (size_t)tracer * (nlev + 1) * ncells;
        const float above = q_above[(size_t)tracer * ncells + cell];
        {
            const int k = nlev - 1;
            vertical_flux[fbase + C2(k, cell, ncells)] = pn_flux3(
                velocity[C2(k, cell, ncells)], coefficient,
                stage[base + C2(k - 2, cell, ncells)], stage[base + C2(k - 1, cell, ncells)],
                stage[base + C2(k, cell, ncells)], above);
        }
        const float vel = velocity[C2(nlev, cell, ncells)];
        const float below = stage[base + C2(nlev - 1, cell, ncells)];
        float value;
        if (scheme == 1) {
            value = pn_flux3(vel, coefficient, stage[base + C2(nlev - 2, cell, ncells)],
                             below, above, q_above2[(size_t)tracer * ncells + cell]);
        } else if (scheme == 2) {
            value = mpas_mul(vel, vel > 0.0f ? below : above);
        } else {
            float face = mpas_add(mpas_mul(fzm_top, above), mpas_mul(fzp_top, below));
            if (vel > 0.0f) face = mpas_min(face, below);
            value = mpas_mul(vel, face);
        }
        vertical_flux[fbase + C2(nlev, cell, ncells)] = value;
    }
}

/* After a stage's recovery (recover_interfaces_f32 and the endpoint
   enforcement zero the interface): the interface row in the dycore's own
   statements, rw = rw_save + rw_p, flux_w = rw_save + ww_avg*(1/steps),
   and w at the interface from the parent at the stage's end. */
extern "C" __global__ void pn_recover_top_v841(
    const int nlev, const int ncells, const int acoustic_steps, const float frac,
    const float *rw_saved, const float *rw_p, const float *ww_avg,
    const float *w_a, const float *w_b, float *rw, float *flux_w, float *w)
{
    PN_LOOP(ncells) {
        const int cell = element;
        const size_t top = C2(nlev, cell, ncells);
        rw[top] = mpas_add(rw_saved[top], rw_p[top]);
        flux_w[top] = mpas_add(rw_saved[top], mpas_mul(ww_avg[top],
            mpas_div(1.0f, (float)acoustic_steps)));
        w[top] = pn_lerp(w_a[cell], w_b[cell], frac);
    }
}

/* Set the interface row of a state (rw and w) to planes at fraction frac. */
extern "C" __global__ void pn_set_top_v841(
    const int nlev, const int ncells, const float frac,
    const float *a, const float *b, float *field)
{
    PN_LOOP(ncells) {
        field[C2(nlev, element, ncells)] = pn_lerp(a[element], b[element], frac);
    }
}

extern "C" __global__ void pn_lerp_v841(
    const int n, const float frac, const float *a, const float *b, float *out)
{
    PN_LOOP(n) { out[element] = pn_lerp(a[element], b[element], frac); }
}

/* mpas_cells_to_points_using_baryweights (mpas_geometry_utils.F:775-823),
   per level: w1*f(c1) + w2*f(c2) + w3*f(c3), left to right.  Source rows
   k0 .. k0+nlev_out-1 of a (nlev_src, nsrc) field, written with row
   stride out_stride (>= npoints: a padded regional array). */
extern "C" __global__ void pn_bary_apply_v841(
    const int nlev_out, const int npoints, const int nsrc, const int k0,
    const int out_stride,
    const int *bary_cells, const float *bary_weights,
    const float *src, float *out)
{
    PN_LOOP(nlev_out * npoints) {
        const int point = element % npoints;
        const int k = element / npoints;
        const size_t row = (size_t)(k0 + k) * nsrc;
        const int c1 = bary_cells[3 * point];
        const int c2 = bary_cells[3 * point + 1];
        const int c3 = bary_cells[3 * point + 2];
        float value = mpas_add(mpas_mul(bary_weights[3 * point], src[row + c1]),
                               mpas_mul(bary_weights[3 * point + 1], src[row + c2]));
        value = mpas_add(value, mpas_mul(bary_weights[3 * point + 2], src[row + c3]));
        out[E2(k, point, out_stride)] = value;
    }
}

/* mpas_reconstruct_2d (mpas_vector_reconstruction.F:274-321): the RBF
   reconstruction at cell centres and its zonal / meridional components.
   coeffs (ncells, max_edges, 3); trig = (cos lat, sin lat, cos lon, sin lon)
   per cell (inputs: MPAS forms them with the build's own libm). */
extern "C" __global__ void pn_reconstruct_v841(
    const int nlev, const int ncells, const int nedges, const int max_edges,
    const int *n_edges_on_cell, const int *edges_on_cell, const float *coeffs,
    const float *trig, const float *u, float *u_zonal, float *u_meridional)
{
    PN_LOOP(nlev * ncells) {
        const int cell = element % ncells;
        const int k = element / ncells;
        float x = 0.0f, y = 0.0f, z = 0.0f;
        for (int i = 0; i < n_edges_on_cell[cell]; ++i) {
            const int edge = edges_on_cell[(size_t)cell * max_edges + i];
            const float value = u[E2(k, edge, nedges)];
            const size_t c = ((size_t)cell * max_edges + i) * 3;
            x = mpas_add(x, mpas_mul(coeffs[c], value));
            y = mpas_add(y, mpas_mul(coeffs[c + 1], value));
            z = mpas_add(z, mpas_mul(coeffs[c + 2], value));
        }
        const float clat = trig[4 * cell], slat = trig[4 * cell + 1];
        const float clon = trig[4 * cell + 2], slon = trig[4 * cell + 3];
        u_zonal[C2(k, cell, ncells)] = mpas_add(mpas_sub(0.0f, mpas_mul(x, slon)),
                                                mpas_mul(y, clon));
        u_meridional[C2(k, cell, ncells)] = mpas_add(
            mpas_sub(0.0f, mpas_mul(mpas_add(mpas_mul(x, clon), mpas_mul(y, slon)), slat)),
            mpas_mul(z, clat));
    }
}

/* init_atmosphere's edge-normal wind (mpas_init_atm_cases.F:5014-5015):
   u = cos(angleEdge)*u_zonal + sin(angleEdge)*v_meridional. */
extern "C" __global__ void pn_edge_normal_v841(
    const int nlev, const int nedges, const float *cos_angle, const float *sin_angle,
    const float *u_zonal, const float *v_meridional, float *u)
{
    PN_LOOP(nlev * nedges) {
        const int edge = element % nedges;
        u[element] = mpas_add(mpas_mul(cos_angle[edge], u_zonal[element]),
                              mpas_mul(sin_angle[edge], v_meridional[element]));
    }
}

/* Elementwise helpers for the LBC frame: rho = rho_zz*zz and dry theta =
   theta_m / (1 + rvord*qv) (the inverse of regional_lbc_derive_v841). */
extern "C" __global__ void pn_frame_cells_v841(
    const int n, const float rvord, const float *rho_zz, const float *zz,
    const float *theta_m, const float *qv, float *rho, float *theta)
{
    PN_LOOP(n) {
        rho[element] = mpas_mul(rho_zz[element], zz[element]);
        theta[element] = mpas_div(theta_m[element],
                                  mpas_add(1.0f, mpas_mul(rvord, qv[element])));
    }
}

/* The conservative overlap remap (fine to coarse), binary64 in the
   reference's order: for target row r and level k,
     num = sum_j  (O_rj * rho(k, s_j)) * q(k, s_j)
     den = sum_j   O_rj * rho(k, s_j)
     out(k, target(r)) = binary32(num / den)
   over the CSR row's entries in stored order (woof.hex.partial_nest_coupling
   .remap_reference states the same sums in NumPy). */
extern "C" __global__ void pn_remap_rows_v841(
    const int nlev, const int nrows, const int nsrc, const int ntarget,
    const int *row_ptr, const int *cols, const double *overlap,
    const int *targets, const float *rho, const float *q, float *out)
{
    PN_LOOP(nlev * nrows) {
        const int r = element % nrows;
        const int k = element / nrows;
        double num = 0.0, den = 0.0;
        for (int j = row_ptr[r]; j < row_ptr[r + 1]; ++j) {
            const size_t s = (size_t)k * nsrc + cols[j];
            const double mass = overlap[j] * mpas_f32_to_f64_ieee(rho[s]);
            num += mass * mpas_f32_to_f64_ieee(q[s]);
            den += mass;
        }
        out[(size_t)k * ntarget + targets[r]] = mpas_f64_to_f32_ieee(num / den);
    }
}

/* The fed-back theta as an rtheta increment, rtheta_pp = rho*theta_new -
   (rtheta_p + rtheta_base), zero outside the fed-back cells: the dycore's
   own recover_cells_f32 then forms rtheta, theta, exner and p' from it. */
extern "C" __global__ void pn_feedback_delta_v841(
    const int nlev, const int nrows, const int ncells, const int *targets,
    const float *theta_new, const float *rho, const float *rtheta_p,
    const float *rtheta_base, float *delta)
{
    PN_LOOP(nlev * nrows) {
        const int r = element % nrows;
        const int k = element / nrows;
        const size_t i = C2(k, targets[r], ncells);
        delta[i] = mpas_sub(mpas_mul(rho[i], theta_new[i]),
                            mpas_add(rtheta_p[i], rtheta_base[i]));
    }
}

/* Copy fed-back scalar rows (k < nlev) of the remapped field into the state. */
extern "C" __global__ void pn_scatter_rows_v841(
    const int nlev, const int nrows, const int ncells, const int *targets,
    const float *src, float *dst)
{
    PN_LOOP(nlev * nrows) {
        const int r = element % nrows;
        const int k = element / nrows;
        const size_t i = C2(k, targets[r], ncells);
        dst[i] = src[i];
    }
}

/* The interface mass budget over the child (D4's nest_top_budget_residual,
   on the overlap): the parent's flux through the area the child covers less
   the child's interpolated flux, per unit child area,
     ((sum_p cover_p*rw_p) - (sum_c area_c*rw_c)) / (sum_c area_c),
   one fixed-order binary64 sum; added to the child's interface plane. */
extern "C" __global__ void pn_budget_residual_v841(
    const int nparent, const int nchild, const double *cover, const float *rw_parent,
    const double *area, const float *rw_child, float *residual)
{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    double sp = 0.0, sc = 0.0, a = 0.0;
    for (int p = 0; p < nparent; ++p)
        sp += cover[p] * mpas_f32_to_f64_ieee(rw_parent[p]);
    for (int c = 0; c < nchild; ++c) {
        sc += area[c] * mpas_f32_to_f64_ieee(rw_child[c]);
        a += area[c];
    }
    residual[0] = mpas_f64_to_f32_ieee((sp - sc) / a);
}

extern "C" __global__ void pn_add_scalar_v841(
    const int n, const float *scalar, float *field)
{
    PN_LOOP(n) { field[element] = mpas_add(field[element], scalar[0]); }
}

/* The top zone: atm_bdy_adjust_dynamics_relaxzone_tend
   (mpas_atm_time_integration.F:8053-8189) on the columns the lateral zones
   do not own, with the ring strength a function of the LEVEL: the layer d
   below the interface (k = nlev-1-d) relaxes as a bdyMask = 5 - d ring,
   coefficient (mask - 1)/nRelaxZone/(50 dt scaling) (Rayleigh) and
   (mask - 1)/nRelaxZone/(10 dt scaling) (the Laplacian filters). */
#define PN_TOP_MASK(d) ((float)(5 - (d)))

extern "C" __global__ void pn_topzone_rayleigh_cell_v841(
    const int nlev, const int ncells, const int nlevels, const int ncols,
    const float fifty_dt, const float relax_zone, const int *cols,
    const float *mesh_scaling_cell, const float *rho_zz, const float *theta_m,
    const float *rho_driving, const float *rt_driving,
    float *tend_rho, float *tend_rt)
{
    PN_LOOP(nlevels * ncols) {
        const int slot = element % ncols;
        const int d = element / ncols;
        const int k = nlev - 1 - d;
        const int cell = cols[slot];
        const float mask = PN_TOP_MASK(d);
        const float coef = mpas_div(mpas_div(mpas_sub(mask, 1.0f), relax_zone),
                                    mpas_mul(fifty_dt, mesh_scaling_cell[cell]));
        const size_t i = C2(k, cell, ncells);
        tend_rho[i] = mpas_sub(tend_rho[i], mpas_mul(coef,
            mpas_sub(rho_zz[i], rho_driving[i])));
        tend_rt[i] = mpas_sub(tend_rt[i], mpas_mul(coef,
            mpas_sub(mpas_mul(rho_zz[i], theta_m[i]), rt_driving[i])));
    }
}

extern "C" __global__ void pn_topzone_rayleigh_edge_v841(
    const int nlev, const int nedges, const int nlevels, const int ncols,
    const float fifty_dt, const float relax_zone, const int *cols,
    const float *mesh_scaling_edge, const float *ru, const float *ru_driving,
    float *tend_ru)
{
    PN_LOOP(nlevels * ncols) {
        const int slot = element % ncols;
        const int d = element / ncols;
        const int k = nlev - 1 - d;
        const int edge = cols[slot];
        const float mask = PN_TOP_MASK(d);
        const float coef = mpas_div(mpas_div(mpas_sub(mask, 1.0f), relax_zone),
                                    mpas_mul(fifty_dt, mesh_scaling_edge[edge]));
        const size_t i = E2(k, edge, nedges);
        tend_ru[i] = mpas_sub(tend_ru[i], mpas_mul(coef, mpas_sub(ru[i], ru_driving[i])));
    }
}

extern "C" __global__ void pn_topzone_filter_cell_v841(
    const int nlev, const int ncells, const int max_edges, const int nlevels,
    const int ncols, const float ten_dt, const float relax_zone, const int *cols,
    const float *mesh_scaling_cell, const int *n_edges_on_cell,
    const int *edges_on_cell, const int *cells_on_edge,
    const float *edge_sign_on_cell, const float *dv_edge, const float *inv_dc_edge,
    const float *rho_zz, const float *theta_m, const float *rho_driving,
    const float *rt_driving, float *tend_rho, float *tend_rt)
{
    PN_LOOP(nlevels * ncols) {
        const int slot = element % ncols;
        const int d = element / ncols;
        const int k = nlev - 1 - d;
        const int cell = cols[slot];
        const float mask = PN_TOP_MASK(d);
        const float filter_coef = mpas_div(mpas_div(mpas_sub(mask, 1.0f), relax_zone),
                                           mpas_mul(ten_dt, mesh_scaling_cell[cell]));
        const size_t i = C2(k, cell, ncells);
        float rt = tend_rt[i];
        float rho = tend_rho[i];
        for (int s = 0; s < n_edges_on_cell[cell]; ++s) {
            const int edge = edges_on_cell[(size_t)cell * max_edges + s];
            float edge_sign = mpas_mul(edge_sign_on_cell[(size_t)cell * max_edges + s],
                                       dv_edge[edge]);
            edge_sign = mpas_mul(edge_sign, inv_dc_edge[edge]);
            edge_sign = mpas_mul(edge_sign, filter_coef);
            const size_t i1 = C2(k, cells_on_edge[2 * edge], ncells);
            const size_t i2 = C2(k, cells_on_edge[2 * edge + 1], ncells);
            rt = mpas_add(rt, mpas_mul(edge_sign, mpas_sub(
                mpas_sub(mpas_mul(rho_zz[i2], theta_m[i2]), rt_driving[i2]),
                mpas_sub(mpas_mul(rho_zz[i1], theta_m[i1]), rt_driving[i1]))));
            rho = mpas_add(rho, mpas_mul(edge_sign, mpas_sub(
                mpas_sub(rho_zz[i2], rho_driving[i2]),
                mpas_sub(rho_zz[i1], rho_driving[i1]))));
        }
        tend_rt[i] = rt;
        tend_rho[i] = rho;
    }
}

extern "C" __global__ void pn_topzone_filter_edge_v841(
    const int nlev, const int nedges, const int max_edges, const int vertex_degree,
    const int nlevels, const int ncols, const float ten_dt, const float divdamp,
    const float relax_zone, const int *cols, const float *mesh_scaling_edge,
    const int *n_edges_on_cell, const int *edges_on_cell, const int *cells_on_edge,
    const int *vertices_on_edge, const int *edges_on_vertex,
    const float *edge_sign_on_cell, const float *edge_sign_on_vertex,
    const float *dc_edge, const float *dv_edge, const float *inv_dc_edge,
    const float *inv_dv_edge, const float *inv_area_cell,
    const float *inv_area_triangle, const float *ru, const float *ru_driving,
    float *tend_ru)
{
    PN_LOOP(nlevels * ncols) {
        const int slot = element % ncols;
        const int d = element / ncols;
        const int k = nlev - 1 - d;
        const int edge = cols[slot];
        const float mask = PN_TOP_MASK(d);
        const float dc = dc_edge[edge];
        const float filter_coef = mpas_div(
            mpas_div(mpas_mul(mpas_mul(dc, dc), mpas_sub(mask, 1.0f)), relax_zone),
            mpas_mul(ten_dt, mesh_scaling_edge[edge]));
        const int cell1 = cells_on_edge[2 * edge];
        const int cell2 = cells_on_edge[2 * edge + 1];
        const int vertex1 = vertices_on_edge[2 * edge];
        const int vertex2 = vertices_on_edge[2 * edge + 1];
        const float r_dc = inv_dc_edge[edge];
        const float r_dv = mpas_min(inv_dv_edge[edge], mpas_mul(4.0f, inv_dc_edge[edge]));
        float divergence1 = 0.0f, divergence2 = 0.0f;
        float vorticity1 = 0.0f, vorticity2 = 0.0f;
        for (int s = 0; s < n_edges_on_cell[cell1]; ++s) {
            const int ed = edges_on_cell[(size_t)cell1 * max_edges + s];
            float sign = mpas_mul(inv_area_cell[cell1], dv_edge[ed]);
            sign = mpas_mul(sign, edge_sign_on_cell[(size_t)cell1 * max_edges + s]);
            const size_t j = E2(k, ed, nedges);
            divergence1 = mpas_add(divergence1, mpas_mul(sign, mpas_sub(ru[j], ru_driving[j])));
        }
        for (int s = 0; s < n_edges_on_cell[cell2]; ++s) {
            const int ed = edges_on_cell[(size_t)cell2 * max_edges + s];
            float sign = mpas_mul(inv_area_cell[cell2], dv_edge[ed]);
            sign = mpas_mul(sign, edge_sign_on_cell[(size_t)cell2 * max_edges + s]);
            const size_t j = E2(k, ed, nedges);
            divergence2 = mpas_add(divergence2, mpas_mul(sign, mpas_sub(ru[j], ru_driving[j])));
        }
        for (int s = 0; s < vertex_degree; ++s) {
            const int ed = edges_on_vertex[vertex1 * vertex_degree + s];
            float sign = mpas_mul(inv_area_triangle[vertex1], dc_edge[ed]);
            sign = mpas_mul(sign, edge_sign_on_vertex[vertex1 * vertex_degree + s]);
            const size_t j = E2(k, ed, nedges);
            vorticity1 = mpas_add(vorticity1, mpas_mul(sign, mpas_sub(ru[j], ru_driving[j])));
        }
        for (int s = 0; s < vertex_degree; ++s) {
            const int ed = edges_on_vertex[vertex2 * vertex_degree + s];
            float sign = mpas_mul(inv_area_triangle[vertex2], dc_edge[ed]);
            sign = mpas_mul(sign, edge_sign_on_vertex[vertex2 * vertex_degree + s]);
            const size_t j = E2(k, ed, nedges);
            vorticity2 = mpas_add(vorticity2, mpas_mul(sign, mpas_sub(ru[j], ru_driving[j])));
        }
        const size_t i = E2(k, edge, nedges);
        const float divergence_part = mpas_mul(
            mpas_mul(divdamp, mpas_sub(divergence2, divergence1)), r_dc);
        const float vorticity_part = mpas_mul(mpas_sub(vorticity2, vorticity1), r_dv);
        tend_ru[i] = mpas_add(tend_ru[i], mpas_mul(filter_coef,
            mpas_sub(divergence_part, vorticity_part)));
    }
}
"""

#: Every entry point of the unit, in source order (the pin test holds both).
KERNELS = (
    "pn_coefficients_top_v841", "pn_column_top_v841", "pn_theta_flux_top_v841",
    "pn_u_flux_top_v841", "pn_w_flux_top_v841", "pn_scalar_flux_top_v841",
    "pn_recover_top_v841", "pn_set_top_v841", "pn_lerp_v841", "pn_bary_apply_v841",
    "pn_reconstruct_v841", "pn_edge_normal_v841", "pn_frame_cells_v841",
    "pn_remap_rows_v841", "pn_feedback_delta_v841", "pn_scatter_rows_v841",
    "pn_budget_residual_v841", "pn_add_scalar_v841",
    "pn_topzone_rayleigh_cell_v841", "pn_topzone_rayleigh_edge_v841",
    "pn_topzone_filter_cell_v841", "pn_topzone_filter_edge_v841",
)

#: Loader options: none beyond the cache's base options (the process's hex
#: numerics: strict ``--fmad=false``, or fast with the FTZ fallback compiled
#: out), exactly the dycore units' route.
OPTIONS: tuple[str, ...] = ()


class PartialNestKernels:
    """The unit's kernels, resolved through one :class:`KernelCache`."""

    def __init__(self, kernel_cache: Any) -> None:
        self.cache = kernel_cache
        self._kernels: dict[str, Any] = {}

    def kernel(self, name: str) -> Any:
        if name not in KERNELS:
            raise KeyError(f"{name!r} is not a partial-nest entry point")
        found = self._kernels.get(name)
        if found is None:
            found = self.cache.raw_kernel(name, CUDA_PARTIAL_NEST_SOURCE,
                                          module_key=MODULE_KEY, options=OPTIONS)
            self._kernels[name] = found
        return found

    def launch(self, name: str, count: int, args: tuple) -> None:
        count = int(count)
        if count < 1:
            return
        blocks = (count + THREADS - 1) // THREADS
        self.kernel(name)((blocks,), (THREADS,), args)


def i32(value: int) -> np.int32:
    return np.int32(value)


def f32(value: float) -> np.float32:
    return np.float32(value)


__all__ = ["CUDA_PARTIAL_NEST_SOURCE", "KERNELS", "MODULE_KEY", "OPTIONS",
           "PartialNestKernels", "THREADS"]
