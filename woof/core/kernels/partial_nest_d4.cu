// woof/core/kernels/partial_nest_d4.cu
//
// Partial-depth nest, Phase D4 (plans/sorted-mapping-hummingbird.md): the
// running child's moist scalars through its interface, and the per-RK-stage
// interface forcing fused into one launch.  NEW entry points only; nothing
// here is called by a configuration without a partial nest, and no existing
// unit (partial_nest.cu, nest_top.cu included) is touched.  Driven by
// woof/core/partial_nest_run.py.
//
// Every floating operation that has to agree across backends is an explicit
// round-to-nearest intrinsic, so no compiler contraction policy moves a word
// (tests/test_partial_nest_d4_kernel_words.py holds each entry point to a
// NumPy statement of its arithmetic on host-made inputs).

static __device__ __forceinline__
real pd4_lerp(real a, real b, real frac)
{
    // partial_nest.cu pn_lerp's words: a + frac*(b - a), each rounded once.
    return __fadd_rn(a, __fmul_rn(frac, __fsub_rn(b, a)));
}

// Once per RK stage, everything partial_nest.cu's nest_top_stage_forcing
// writes, word for word, plus what D3 formed in three further launches:
//   u_above = I(u_k, f_star) on u points, v_above = I(v_k, f_star) on v
//   points (nest_top_time_interp's words), and the child's total theta at
//   its top half level, theta_top[nz-1] = thb[nz-1] + thp[nz-1] -- the only
//   level of D3's full-column theta_total the face flux reads.
// One thread per point of the largest of the three planes.
extern "C" __global__
void nest_top_stage_forcing_fused(
    const real* __restrict__ om_a, const real* __restrict__ om_b,
    const real* __restrict__ adj_a, const real* __restrict__ adj_b,
    const real* __restrict__ th_a, const real* __restrict__ th_b,
    const real* __restrict__ phu_a, const real* __restrict__ phu_b,
    const real* __restrict__ u_a, const real* __restrict__ u_b,
    const real* __restrict__ v_a, const real* __restrict__ v_b,
    const real* __restrict__ w, const real* __restrict__ php,
    const real* __restrict__ phb, const real* __restrict__ thp,
    const real* __restrict__ thb, int base3d, int nz, int ny, int nx,
    real frac_star,
    real* __restrict__ ww_rf, real* __restrict__ adj_rf,
    real* __restrict__ t_above, real* __restrict__ pht_above,
    real* __restrict__ w_star, real* __restrict__ ph_star,
    real* __restrict__ u_above, real* __restrict__ v_above,
    real* __restrict__ theta_top)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int n = ny * nx;
    int nu = ny * (nx + 1);
    int nv = (ny + 1) * nx;
    if (t < n) {
        ww_rf[t] = pd4_lerp(om_a[t], om_b[t], frac_star);
        adj_rf[t] = pd4_lerp(adj_a[t], adj_b[t], frac_star);
        t_above[t] = pd4_lerp(th_a[t], th_b[t], frac_star);
        pht_above[t] = pd4_lerp(phu_a[t], phu_b[t], frac_star);
        size_t f = (size_t)nz * n + t;
        w_star[t] = w[f];
        ph_star[t] = __fadd_rn(phb[base3d ? f : (size_t)nz], php[f]);
        size_t h = (size_t)(nz - 1) * n + t;
        theta_top[h] = __fadd_rn(thb[base3d ? h : (size_t)(nz - 1)], thp[h]);
    }
    if (t < nu) u_above[t] = pd4_lerp(u_a[t], u_b[t], frac_star);
    if (t < nv) v_above[t] = pd4_lerp(v_a[t], v_b[t], frac_star);
}

// The interface flux of one moist scalar over an RK stage.  The scalars
// advect with the stage's acoustic time-averaged mass flux (WRF sumflux,
// ww_m), whose interface face ww_m[nz] the child's own scalar kernels treat
// as a lid; this is that face's flux, per column:
//   F = ww_m[nz] * q_face,  q_face = fnm*q_above + fnp*q[nz-1]
// -- the second-order face value of the existing kernels' boundary-adjacent
// faces (WRF advect_scalar's vflux = rom*(fzm*f(k) + fzp*f(k-1))) -- with
// q_above the parent's value at half level k_top at the stage reference.
// An OUTGOING face (ww_m < 0: upward, out of the child) carries at most the
// child's own top-level value, q_face = min(q_face, q[nz-1]), so the flux
// never removes more than the donor holds where the parent is moister above.
// ww is Omega-signed (positive downward, into the child).
extern "C" __global__
void nest_top_scalar_face_flux(const real* __restrict__ q,
                               const real* __restrict__ q_above,
                               const real* __restrict__ ww_m,
                               real fnm_top, real fnp_top,
                               real* __restrict__ flux, int nz, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    real q_top = q[(size_t)(nz - 1) * n + t];
    real vel = ww_m[(size_t)nz * n + t];
    real qf = __fadd_rn(__fmul_rn(fnm_top, q_above[t]), __fmul_rn(fnp_top, q_top));
    if (vel < 0.0f && qf > q_top) qf = q_top;
    flux[t] = __fmul_rn(vel, qf);
}

// The flux above, applied after the stage's scalar update.  The existing
// update forms q = (C(mu_t)*q_t + dt*tend) / C(mu_new); the interface term
// tend(nz-1) -= rdnw(nz-1)*F joins it as
//   q[nz-1] += dt * (-rdnw(nz-1)*F) / C(mu_new)(nz-1),
//   C(mu)(k) = c1h(k)*(mub + mu') + c2h(k),
// clamped at zero (the PD stage's own rounding clamp).
extern "C" __global__
void nest_top_scalar_apply(real* __restrict__ q,
                           const real* __restrict__ flux,
                           const real* __restrict__ mub2d,
                           const real* __restrict__ mup,
                           const real* __restrict__ c1h,
                           const real* __restrict__ c2h,
                           const real* __restrict__ rdnw,
                           real dt, int nz, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int k = nz - 1;
    real mass = __fadd_rn(__fmul_rn(c1h[k], __fadd_rn(mub2d[t], mup[t])), c2h[k]);
    real tend = __fmul_rn(__fsub_rn(0.0f, rdnw[k]), flux[t]);
    size_t i = (size_t)k * n + t;
    real v = __fadd_rn(q[i], __fdiv_rn(__fmul_rn(dt, tend), mass));
    q[i] = v > 0.0f ? v : 0.0f;
}

// The radiation call's extended column (D4): the child's own levels below
// the interface and the parent's SINT-ed levels above it, in one array per
// field.  `above` is (nabove, ny, nx), the parent's levels k_top .. nz_ext-1
// (half) or k_top .. nz_ext (full) already on the child grid; `below` is the
// child's field (nbelow levels); `out` is (nbelow + nabove, ny, nx) minus the
// shared level for a full-level field (`shared` = 1: the interface itself is
// the child's own top level, the parent's copy of it is skipped).
extern "C" __global__
void nest_ext_column_fill(const real* __restrict__ below,
                          const real* __restrict__ above,
                          real* __restrict__ out,
                          int nbelow, int nout, int shared, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nout * n) return;
    int k = t / n, c = t - k * n;
    out[t] = k < nbelow ? below[t] : above[(size_t)(k - nbelow + shared) * n + c];
}

// partial_nest.cu's nest_top_apply_held for the five relaxed fields (u, v,
// theta, phi, w) in one launch: field f's tendency tend_f gains its held
// top-zone increment held_f on the columns the top zone owns -- the
// specified rows replaced, the relaxation rows added -- word for word as
// the single-field entry point.  The shapes are the child's staggerings:
// u (nz, ny, nx+1), v (nz, ny+1, nx), theta (nz, ny, nx) on half levels,
// phi and w (nz+1, ny, nx) on full levels.
static __device__ __forceinline__
void pd4_apply_held(real* __restrict__ tend, const real* __restrict__ held,
                    int t, int spec_levels, int relax_levels, int frame,
                    int half, int nzf, int nyf, int nxf)
{
    int plane = nyf * nxf;
    if (t >= relax_levels * plane) return;
    int d = t / plane, r = t - d * plane;
    int j = r / nxf, i = r - j * nxf;
    if (i < frame || i >= nxf - frame || j < frame || j >= nyf - frame) return;
    int k = nzf - 1 + half - d;
    if (k >= nzf || k < 1) return;
    size_t idx = (size_t)k * plane + r;
    tend[idx] = d < spec_levels ? held[idx] : __fadd_rn(tend[idx], held[idx]);
}

extern "C" __global__
void nest_top_apply_held_all(real* __restrict__ tend_u, const real* __restrict__ held_u,
                             real* __restrict__ tend_v, const real* __restrict__ held_v,
                             real* __restrict__ tend_t, const real* __restrict__ held_t,
                             real* __restrict__ tend_ph, const real* __restrict__ held_ph,
                             real* __restrict__ tend_w, const real* __restrict__ held_w,
                             int spec_levels, int relax_levels, int frame,
                             int nz, int ny, int nx)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    pd4_apply_held(tend_u, held_u, t, spec_levels, relax_levels, frame, 1, nz, ny, nx + 1);
    pd4_apply_held(tend_v, held_v, t, spec_levels, relax_levels, frame, 1, nz, ny + 1, nx);
    pd4_apply_held(tend_t, held_t, t, spec_levels, relax_levels, frame, 1, nz, ny, nx);
    pd4_apply_held(tend_ph, held_ph, t, spec_levels, relax_levels, frame, 0, nz + 1, ny, nx);
    pd4_apply_held(tend_w, held_w, t, spec_levels, relax_levels, frame, 0, nz + 1, ny, nx);
}

// nest_top.cu's nest_top_face_flux for theta (mass points), u and v in one
// launch, every operation an explicit round-to-nearest (D2's entry point
// leaves contraction to the compiler, so its CUDA words are these words
// only under --fmad=false):
//   vel = ww_top (mass), 0.5*(ww_top(i-1) + ww_top(i)) (u),
//         0.5*(ww_top(j-1) + ww_top(j)) (v), periodic or clamped at an open
//         or nested edge as D2's;
//   q_face = centered ? fnm*q_above + fnp*q(nz-1) : (vel > 0 ? q_above : q(nz-1));
//   F = vel*q_face; flux_out = F; tend(nz-1) = tend(nz-1) - rdnw(nz-1)*F.
static __device__ __forceinline__
void pd4_face_flux(real* __restrict__ tend, const real* __restrict__ q,
                   const real* __restrict__ q_above, real* __restrict__ flux_out,
                   real vel, real rdnw_top, real fnm_top, real fnp_top,
                   int centered, size_t top, int t)
{
    real qf;
    if (centered)
        qf = __fadd_rn(__fmul_rn(fnm_top, q_above[t]), __fmul_rn(fnp_top, q[top]));
    else
        qf = vel > 0.0f ? q_above[t] : q[top];
    real flux = __fmul_rn(vel, qf);
    flux_out[t] = flux;
    tend[top] = __fsub_rn(tend[top], __fmul_rn(rdnw_top, flux));
}

extern "C" __global__
void nest_top_face_flux_all(real* __restrict__ tend_t, const real* __restrict__ q_t,
                            const real* __restrict__ above_t, real* __restrict__ flux_t,
                            real* __restrict__ tend_u, const real* __restrict__ q_u,
                            const real* __restrict__ above_u, real* __restrict__ flux_u,
                            real* __restrict__ tend_v, const real* __restrict__ q_v,
                            const real* __restrict__ above_v, real* __restrict__ flux_v,
                            const real* __restrict__ ww_top,
                            const real* __restrict__ rdnw,
                            real fnm_top, real fnp_top, int centered,
                            int open_x, int open_y, int nz, int ny, int nx)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    real rd = rdnw[nz - 1];
    if (t < ny * nx) {
        pd4_face_flux(tend_t, q_t, above_t, flux_t, ww_top[t], rd, fnm_top, fnp_top,
                      centered, (size_t)(nz - 1) * ny * nx + t, t);
    }
    if (t < ny * (nx + 1)) {
        int nxs = nx + 1;
        int j = t / nxs, i = t - j * nxs;
        int ia = open_x ? min(i, nx - 1) : i % nx;
        int ib = open_x ? max(i - 1, 0) : (i - 1 + nx) % nx;
        real vel = __fmul_rn(0.5f, __fadd_rn(ww_top[(size_t)j * nx + ib],
                                             ww_top[(size_t)j * nx + ia]));
        pd4_face_flux(tend_u, q_u, above_u, flux_u, vel, rd, fnm_top, fnp_top,
                      centered, (size_t)(nz - 1) * ny * nxs + t, t);
    }
    if (t < (ny + 1) * nx) {
        int j = t / nx, i = t - j * nx;
        int ja = open_y ? min(j, ny - 1) : j % ny;
        int jb = open_y ? max(j - 1, 0) : (j - 1 + ny) % ny;
        real vel = __fmul_rn(0.5f, __fadd_rn(ww_top[(size_t)jb * nx + i],
                                             ww_top[(size_t)ja * nx + i]));
        pd4_face_flux(tend_v, q_v, above_v, flux_v, vel, rd, fnm_top, fnp_top,
                      centered, (size_t)(nz - 1) * (ny + 1) * nx + t, t);
    }
}
