// woof/core/kernels/partial_nest.cu
//
// Partial-depth nest, Phase D3 (plans/sorted-mapping-hummingbird.md): the
// running child's coupling with its parent above the interface.  NEW entry
// points only; nothing here is called by a configuration without a partial
// nest, and no existing unit is touched.  Driven by
// woof/core/partial_nest_run.py.
//
// Every floating operation that has to agree across backends, or with the
// gfortran oracle (tools/partial_nest_wrf471_oracle), is an explicit
// round-to-nearest intrinsic, so no compiler contraction policy moves a word.
//
// Layout conventions:
//   * child-grid planes are (ny, nx) of the field's own staggering;
//   * a TOP STRIP is (width, ny, nx) with row d = 0 at the INTERFACE, the
//     specified row: a full-level field's row d is its full level nz-d (row
//     0 is the interface itself), a half-level field's row d is the half
//     level nz-d (row 0 is the parent's half level just above the interface,
//     rows 1.. the child's own).  This is the lateral tables' "width index 0
//     at the domain edge" orientation turned on its side: WRF's south-side
//     array field_bdy_ys(i, k, b_dist+1) is strip[d][j][i], WRF's carried
//     index k standing for our j.

// One parent-step linear interpolation, nest_top_time_interp's words:
// a + frac*(b - a), each operation rounded once.
static __device__ __forceinline__
real pn_lerp(real a, real b, real frac)
{
    return __fadd_rn(a, __fmul_rn(frac, __fsub_rn(b, a)));
}

// Per acoustic substep: everything specified at the interface that moves
// with the substep, from the parent's interface planes at the start (a) and
// end (b) of its step, already on the child grid.  The perturbation
// variables are taken relative to the child's own RK-stage reference:
//   ww_pp     = I(om_k, f)  - ww_rf        (Omega'' at the interface)
//   w_top     = c_top * (I(w_k, f) - w_star)   (coupled w'', c_top = c2f[nz])
//   ph_top    = I(ph_k, f)  - ph_star      (phi'', total geopotential)
//   ww_above  = I(om_kp1, f) - I(om_kp1, f_star)
//   p_above   = I(p_k, f)   - I(p_k, f_star)
//   p_above_old = I(p_k, f_old) - I(p_k, f_star)
//   adj_pp    = I(adj, f)   - adj_rf       (mass-budget correction)
// f is the substep's fraction of the parent step, f_old the previous
// substep's, f_star the stage reference's.
extern "C" __global__
void nest_top_substep_forcing(
    const real* __restrict__ om_a, const real* __restrict__ om_b,
    const real* __restrict__ w_a, const real* __restrict__ w_b,
    const real* __restrict__ ph_a, const real* __restrict__ ph_b,
    const real* __restrict__ omu_a, const real* __restrict__ omu_b,
    const real* __restrict__ p_a, const real* __restrict__ p_b,
    const real* __restrict__ adj_a, const real* __restrict__ adj_b,
    const real* __restrict__ ww_rf, const real* __restrict__ adj_rf,
    const real* __restrict__ w_star, const real* __restrict__ ph_star,
    real c_top, real frac, real frac_old, real frac_star,
    real* __restrict__ ww_pp, real* __restrict__ w_top,
    real* __restrict__ ph_top, real* __restrict__ ww_above,
    real* __restrict__ p_above, real* __restrict__ p_above_old,
    real* __restrict__ adj_pp, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    ww_pp[t] = __fsub_rn(pn_lerp(om_a[t], om_b[t], frac), ww_rf[t]);
    w_top[t] = __fmul_rn(c_top, __fsub_rn(pn_lerp(w_a[t], w_b[t], frac), w_star[t]));
    ph_top[t] = __fsub_rn(pn_lerp(ph_a[t], ph_b[t], frac), ph_star[t]);
    ww_above[t] = __fsub_rn(pn_lerp(omu_a[t], omu_b[t], frac),
                            pn_lerp(omu_a[t], omu_b[t], frac_star));
    real p_star = pn_lerp(p_a[t], p_b[t], frac_star);
    p_above[t] = __fsub_rn(pn_lerp(p_a[t], p_b[t], frac), p_star);
    p_above_old[t] = __fsub_rn(pn_lerp(p_a[t], p_b[t], frac_old), p_star);
    adj_pp[t] = __fsub_rn(pn_lerp(adj_a[t], adj_b[t], frac), adj_rf[t]);
}

// Once per RK stage: the stage-reference values at the interface,
//   ww_rf = I(om_k, f_star), adj_rf = I(adj, f_star),
//   t_above = I(th_k, f_star), pht_above = I(ph_kp1, f_star),
// and the child's own reference words at its top full level,
//   w_star = w[nz], ph_star = phb[nz] + php[nz]
// (phb is a profile unless base3d).
extern "C" __global__
void nest_top_stage_forcing(
    const real* __restrict__ om_a, const real* __restrict__ om_b,
    const real* __restrict__ adj_a, const real* __restrict__ adj_b,
    const real* __restrict__ th_a, const real* __restrict__ th_b,
    const real* __restrict__ phu_a, const real* __restrict__ phu_b,
    const real* __restrict__ w, const real* __restrict__ php,
    const real* __restrict__ phb, int base3d, int nz,
    real frac_star,
    real* __restrict__ ww_rf, real* __restrict__ adj_rf,
    real* __restrict__ t_above, real* __restrict__ pht_above,
    real* __restrict__ w_star, real* __restrict__ ph_star, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    ww_rf[t] = pn_lerp(om_a[t], om_b[t], frac_star);
    adj_rf[t] = pn_lerp(adj_a[t], adj_b[t], frac_star);
    t_above[t] = pn_lerp(th_a[t], th_b[t], frac_star);
    pht_above[t] = pn_lerp(phu_a[t], phu_b[t], frac_star);
    size_t f = (size_t)nz * n + t;
    w_star[t] = w[f];
    ph_star[t] = __fadd_rn(phb[base3d ? f : (size_t)nz], php[f]);
}

// The parent's interface budget over the child's footprint (D3 mass
// correction).  SINT is not conservative: the child's interpolated
// interface flux, summed over its cells, differs from the parent's flux
// through the same footprint.  One thread forms both sums serially (row by
// row, west to east: a fixed order, the same words on every backend) and
// writes the uniform per-child-cell residual
//   residual = (nri*nrj * sum_parent - sum_child) / (nyc*nxc)   (binary64)
// that makes them equal; the driver adds it to the child's interface flux.
// Child mass cell (j, i) lies in parent cell (jpos-1 + j/nrj, ipos-1 + i/nri)
// (0-based), so the footprint is parent cells jpos-1 .. jpos-1 + nyc/nrj - 1
// by ipos-1 .. ipos-1 + nxc/nri - 1 (whole parent cells: the D0 rule).
extern "C" __global__
void nest_top_budget_residual(const real* __restrict__ parent,
                              const real* __restrict__ child,
                              real* __restrict__ residual,
                              int ipos, int jpos, int nri, int nrj,
                              int nyp, int nxp, int nyc, int nxc)
{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    // Sums and differences only (each parent value added once per child
    // cell it covers): nothing a contracting compiler could fuse.
    double sp = 0.0, sc = 0.0;
    for (int pj = 0; pj < nyc / nrj; ++pj)
        for (int pi = 0; pi < nxc / nri; ++pi) {
            double v = (double)parent[(size_t)(jpos - 1 + pj) * nxp + (ipos - 1 + pi)];
            for (int c = 0; c < nri * nrj; ++c) sp += v;
        }
    for (int j = 0; j < nyc; ++j)
        for (int i = 0; i < nxc; ++i)
            sc += (double)child[(size_t)j * nxc + i];
    residual[0] = (real)((sp - sc) / (double)(nyc * nxc));
}

// The top strip's VALUE and TENDENCY tables at FORCE, in bdy_interp1's own
// arithmetic (kernels/nest.cu nest_bdy_interp1, interp_fcn.F:2583-2584):
//   value    = the child's current coupled strip
//   tendency = rdt * (SINT(coupled parent @ t+dtp) - child), rdt = 1/cdt in
//              binary64, the difference formed in binary32.
// psca is the SINT result over the whole child plane (nest_interp.sint with
// the bdy-wrapper registration the lateral tables use), so wherever the
// strip overlaps the lateral frame its words are the lateral tables' words.
extern "C" __global__
void nest_top_strip_tables(const real* __restrict__ psca,
                           const real* __restrict__ child,
                           real* __restrict__ top_val,
                           real* __restrict__ top_tend,
                           real cdt, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    real nv = child[t];
    double rdt = 1.0 / (double)cdt;
    top_tend[t] = (real)(rdt * (double)(psca[t] - nv));
    top_val[t] = nv;
}

// The child's coupled field at the top strip's rows (header: row d is the
// level `top - d`, top = nzf - 1 + half).  A half-level field has no row 0
// of its own -- that row is the parent's half level above the interface --
// so its row 0 is the target itself (psca), which makes its tendency zero
// and its relaxation residual exactly zero wherever a stencil reads it.
extern "C" __global__
void nest_top_strip_gather(const real* __restrict__ field,
                           const real* __restrict__ psca,
                           real* __restrict__ strip,
                           int width, int half, int nzf, int nyf, int nxf)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int plane = nyf * nxf;
    if (t >= width * plane) return;
    int d = t / plane, r = t - d * plane;
    int k = nzf - 1 + half - d;
    strip[t] = k >= nzf ? psca[t] : field[(size_t)k * plane + r];
}

// WRF spec_bdytend + relax_bdytend_core (share/module_bc.F:1221-1549) on the
// TOP strip: WRF's south side transposed.  Its outer row (b_dist = 0) is the
// INTERFACE: for a full-level field (w, phi) the child's own top level, which
// the acoustic top entry points specify (nest_top.cu), for a half-level field
// (u, v, theta) the parent's level above it, which enters the child only as
// the target of the first relaxation row.  The inner rows are the levels
// below; row d is level top - d with top = nzf - 1 + half.  The along-boundary
// stencil runs in x with WRF's clamps (im1 = max(i-1, ibs), ip1 = min(i+1,
// ibe-1)); WRF's carried index k stands for y.
//   d <  spec_levels:  field_tend = top_tend[d]                 (spec_bdytend)
//   d <  relax_levels: field_tend = field_tend + fcx(d)*fls0
//                                   - gcx(d)*(fls1+fls2+fls3+fls4-4.*fls0)
// with flsN = (top_val + dtbc*top_tend) - field, left to right, each
// operation rounded once (the order gfortran -O0 -ffp-contract=off gives;
// lbc_state.cu's LBC_RELAX_ADD is the same macro for the lateral rows).  A
// row the child does not have (a half-level field's row 0) is read as its
// own target, top_val (its tendency is zero), as on WRF's specified rows,
// whose field IS the boundary value.
// Ownership at the corners: the lateral frame owns every column within
// `frame` points of the field's own edges (its specified and relaxation
// rows, at every level); the top zone takes the remaining columns.  mu is
// never passed here (it is not relaxed).
static __device__ __forceinline__
real pn_bdy(const real* __restrict__ val, const real* __restrict__ tend,
            size_t b, real dtbc)
{
    return __fadd_rn(val[b], __fmul_rn(dtbc, tend[b]));
}

extern "C" __global__
void nest_top_relaxation(const real* __restrict__ field,
                         real* __restrict__ field_tend,
                         const real* __restrict__ top_val,
                         const real* __restrict__ top_tend,
                         const real* __restrict__ fcx,
                         const real* __restrict__ gcx,
                         real dtbc, int width, int spec_levels,
                         int relax_levels, int frame, int half,
                         int nzf, int nyf, int nxf)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int plane = nyf * nxf;
    if (t >= relax_levels * plane) return;
    int d = t / plane, r = t - d * plane;
    int j = r / nxf, i = r - j * nxf;
    if (i < frame || i >= nxf - frame || j < frame || j >= nyf - frame) return;
    int top = nzf - 1 + half;
    int k = top - d;
    if (k >= nzf || k < 1) return;               // no such child row
    size_t idx = (size_t)k * plane + r;
    size_t b0 = (size_t)d * plane + r;
    if (d < spec_levels) {
        field_tend[idx] = top_tend[b0];
        return;
    }
    if (d + 1 >= width) return;
    int im = i - 1 > 0 ? i - 1 : 0;
    int ip = i + 1 < nxf - 1 ? i + 1 : nxf - 1;
    size_t row = (size_t)d * plane + (size_t)j * nxf;
    size_t frow = (size_t)k * plane + (size_t)j * nxf;
    size_t bo = b0 - plane, bi = b0 + plane;
    real outer = k + 1 < nzf ? field[idx + plane] : top_val[bo];
    real fls0 = __fsub_rn(pn_bdy(top_val, top_tend, b0, dtbc), field[idx]);
    real fls1 = __fsub_rn(pn_bdy(top_val, top_tend, row + im, dtbc), field[frow + im]);
    real fls2 = __fsub_rn(pn_bdy(top_val, top_tend, row + ip, dtbc), field[frow + ip]);
    real fls3 = __fsub_rn(pn_bdy(top_val, top_tend, bo, dtbc), outer);
    real fls4 = __fsub_rn(pn_bdy(top_val, top_tend, bi, dtbc), field[idx - plane]);
    real lap = __fsub_rn(__fadd_rn(__fadd_rn(__fadd_rn(fls1, fls2), fls3), fls4),
                         __fmul_rn(4.0f, fls0));
    field_tend[idx] = __fsub_rn(__fadd_rn(field_tend[idx], __fmul_rn(fcx[d], fls0)),
                                __fmul_rn(gcx[d], lap));
}

// Every RK stage: the held top-zone increments (nest_top_relaxation run on
// stage 1 from the time-t state into a zeroed `held`) enter the stage's
// tendency as WRF's relax_bdy_dry *_save rows do -- the specified rows'
// tendency replaced, the relaxation rows' increment added -- on the columns
// the top zone owns.
extern "C" __global__
void nest_top_apply_held(real* __restrict__ field_tend,
                         const real* __restrict__ held,
                         int spec_levels, int relax_levels, int frame, int half,
                         int nzf, int nyf, int nxf)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int plane = nyf * nxf;
    if (t >= relax_levels * plane) return;
    int d = t / plane, r = t - d * plane;
    int j = r / nxf, i = r - j * nxf;
    if (i < frame || i >= nxf - frame || j < frame || j >= nyf - frame) return;
    int k = nzf - 1 + half - d;
    if (k >= nzf || k < 1) return;
    size_t idx = (size_t)k * plane + r;
    field_tend[idx] = d < spec_levels ? held[idx]
                                      : __fadd_rn(field_tend[idx], held[idx]);
}

// The interface's phi'' predictor CONSISTENT with the specified interface
// values: advance_w_phi_top closes phi''(nz) = rhs(nz) + 0.5*dtau*g*(1+epssm)
// * w''(nz) / (c1f*muts + c2f) in a full column, and with both w''(nz) =
// w_top and phi''(nz) = ph_top specified, the rhs(nz) the implicit w solve
// must see is
//   rhs_top = ph_top - 0.5*dtau*g*(1+epssm) * w_top / (c1f(nz)*muts + c2f(nz)),
// muts = mub + mu' + mu'' after advance_mu_th_top.  D2's continuation
// predictor (nest_top_ph_predictor) instead rebuilds rhs(nz) from the
// child's own big-step phi tendency at its top level, which the existing
// rhs_ph kernel forms one-sided (a lid's top row, no vertical advection
// through the interface), so the w solve below the interface sees an
// interface geopotential the specified ph_top then overwrites.
extern "C" __global__
void nest_top_specified_predictor(real* __restrict__ rhs_top,
                                  const real* __restrict__ w_top,
                                  const real* __restrict__ ph_top,
                                  const real* __restrict__ mub2d,
                                  const real* __restrict__ mup,
                                  const real* __restrict__ mu_pp,
                                  const real* __restrict__ c1f,
                                  const real* __restrict__ c2f,
                                  real dtau, real epssm, int nz, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    real muts = __fadd_rn(__fadd_rn(mub2d[t], mup[t]), mu_pp[t]);
    real mass = __fadd_rn(__fmul_rn(c1f[nz], muts), c2f[nz]);
    real coef = __fmul_rn(__fmul_rn(__fmul_rn(0.5f, dtau), G), __fadd_rn(1.0f, epssm));
    rhs_top[t] = __fsub_rn(ph_top[t], __fdiv_rn(__fmul_rn(coef, w_top[t]), mass));
}
