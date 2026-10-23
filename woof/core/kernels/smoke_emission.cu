// Smoke emission and injection: WRF-SFIRE's fire smoke tracer source and
// the two simple injection profiles of woof.smoke.injection.
//
//   smoke_fire_tracer_emissions  phys/module_fr_fire_atm.F:15-106
//                         (add_fire_tracer_emissions, fire_smk_scheme = 0):
//                         every fire-grid cell under an interior atmospheric
//                         column adds
//                             avgw*fire_tracer_smoke*burnt_area_dt*fgip*1000
//                                 / (rho*dz8w)
//                         g of smoke per kg of air to the column's lowest
//                         level, one cell at a time in WRF's joff-outer,
//                         ioff-inner order.  WRF applies it to the first
//                         interior column of each tile only (its `do
//                         i=i_st,i_st`); this kernel applies the same
//                         statement to every interior column (woof.smoke.
//                         emission, PROVENANCE.md).
//   smoke_pm25_flux       burned fuel mass rate times a per-cell emission
//                         factor: one product per cell.
//   smoke_inject_uniform  a surface flux spread uniformly in height over
//                         [z_bot, z_top] of a column, as a mixing-ratio
//                         source; woof's own (no WRF counterpart).
//   smoke_inject_lowest   the whole surface flux into the lowest layer.
//
// Every statement is written in the Fortran's (or the NumPy reference's)
// order of evaluation and the unit compiles with -fmad=false, so no product
// is fused into a sum.  Fields are (nz, ny, nx) C order, a column's levels
// ncol = ny*nx apart; the fire grid is (ny*jr, nx*ir) C order.

extern "C" __global__ void smoke_fire_tracer_emissions(
    float *__restrict__ plane,            // (ny, nx): tracer at level 1, in/out
    const float *__restrict__ rho,        // (ny, nx): rho at level 1
    const float *__restrict__ dz8w,       // (ny, nx): dz8w at level 1
    const float *__restrict__ burnt_area_dt,   // fire grid
    const float *__restrict__ fgip,            // fire grid
    const float *__restrict__ factor,     // fire grid, or null for the scalar
    float factor_scalar, int nx, int ny, int ir, int jr)
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= nx * ny) return;
    const int j = cell / nx;
    const int i = cell - j * nx;
    // i_st..i_en = max(its,ids+1)..min(ite,ide-2) with ide = nx+1: the
    // interior columns 2..nx-1 (one-based), and the same in j.
    if (i < 1 || i > nx - 2 || j < 1 || j > ny - 2) return;
    // avgw = 1.0/(ir*jr): the integer product converted, then divided.
    const float avgw = 1.0f / (float)(ir * jr);
    const int nxf = nx * ir;
    float tracer = plane[cell];
    for (int joff = 0; joff < jr; ++joff) {
        const int j_f = j * jr + joff;
        for (int ioff = 0; ioff < ir; ++ioff) {
            const int f = j_f * nxf + i * ir + ioff;
            const float fts = factor ? factor[f] : factor_scalar;
            float emis = avgw * fts;
            emis = emis * burnt_area_dt[f];
            emis = emis * fgip[f];
            emis = emis * 1000.0f;
            emis = emis / (rho[cell] * dz8w[cell]);
            tracer = tracer + emis;
        }
    }
    plane[cell] = tracer;
}

extern "C" __global__ void smoke_pm25_flux(
    const float *__restrict__ burn_rate, const float *__restrict__ factor,
    float *__restrict__ flux, int n)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    flux[c] = burn_rate[c] * factor[c];
}

extern "C" __global__ void smoke_inject_uniform(
    const float *__restrict__ z_w,        // (nz+1, ny, nx) interface heights
    const float *__restrict__ rho,        // (nz, ny, nx)
    const float *__restrict__ flux,       // (ny, nx) kg m-2 s-1
    const float *__restrict__ z_bot,      // (ny, nx)
    const float *__restrict__ z_top,      // (ny, nx)
    float *__restrict__ source,           // (nz, ny, nx) kg kg-1 s-1
    float top_scale, int nz, int ncol)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const float zs = z_w[c];
    const float zt = z_w[(size_t)nz * ncol + c];
    // top_scale: the PBL fraction (1 for a given top; x*1 is x exactly).
    const float bot = fminf(fmaxf(z_bot[c], zs), zt);
    const float top = fminf(fmaxf(z_top[c] * top_scale, zs), zt);
    const float depth = top - bot;
    const float f = flux[c];
    for (int k = 0; k < nz; ++k) source[(size_t)k * ncol + c] = 0.0f;
    if (depth > 0.0f) {
        for (int k = 0; k < nz; ++k) {
            const float lo = z_w[(size_t)k * ncol + c];
            const float hi = z_w[(size_t)(k + 1) * ncol + c];
            const float overlap = fminf(hi, top) - fmaxf(lo, bot);
            if (overlap > 0.0f) {
                const float share = overlap / depth;
                source[(size_t)k * ncol + c] =
                    f * share / (rho[(size_t)k * ncol + c] * (hi - lo));
            }
        }
        return;
    }
    // A degenerate interval: the layer holding z_bot (the top layer when
    // z_bot sits on the column top).
    int kk = nz - 1;
    for (int k = 0; k < nz; ++k) {
        const float lo = z_w[(size_t)k * ncol + c];
        const float hi = z_w[(size_t)(k + 1) * ncol + c];
        if (lo <= bot && bot < hi) { kk = k; break; }
    }
    const float lo = z_w[(size_t)kk * ncol + c];
    const float hi = z_w[(size_t)(kk + 1) * ncol + c];
    source[(size_t)kk * ncol + c] = f / (rho[(size_t)kk * ncol + c] * (hi - lo));
}

extern "C" __global__ void smoke_inject_lowest(
    const float *__restrict__ z_w, const float *__restrict__ rho,
    const float *__restrict__ flux, float *__restrict__ source,
    int nz, int ncol)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    for (int k = 1; k < nz; ++k) source[(size_t)k * ncol + c] = 0.0f;
    source[c] = flux[c] / (rho[c] * (z_w[ncol + c] - z_w[c]));
}
