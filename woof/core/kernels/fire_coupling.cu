// WRF v4.7.1 fire-atmosphere coupling (ifire = 2): the driver glue around
// the level-set fire of fire_levelset.cu, statement for statement.
//
//   fc_wind_u /
//   fc_wind_v        interpolate_atm2fire's vertical step
//                    (module_fr_fire_driver.F:1290-1415): the staggered wind
//                    at fire_wind_height above the ground, log-linear from 0
//                    at the roughness length in the first layer, log-linear
//                    between layers above, the top layer's value above the
//                    model and 0 where the roughness exceeds the height.
//   fc_wind_extend   its continue_at_boundary(1, 1, 0.) on both staggered
//                    planes (module_fr_fire_util.F:293-384).
//   fc_wind_refine   interpolate_2d onto the fire mesh (util.F:519-592),
//                    lined up as interpolate_atm2fire lines it up; each fine
//                    node takes the last coarse cell WRF's loop writes it
//                    from, the host tables carrying that cell and weight.
//   fc_tendency      fire_tendency (module_fr_fire_atm.F:110-306),
//                    fire_sfc_flx = 0, with phy_prep's z_at_w, dz8w and rho
//                    (module_big_step_utilities_em.F:4856-4877) formed in
//                    place; also the theta tendency over the map factor
//                    (rk_addtend_dry's division) for woof's slow slot.
//   fc_smoke_flux    add_fire_tracer_emissions' source (atm.F:15-106,
//                    tracer_opt = 3, fire_smk_scheme = 0) as a surface flux
//                    in kg m-2 s-1 for the smoke tracer's emission seam.
//   fc_active_area   the actively burning area of each atmospheric cell, m2.
//
// Layout (woof): (nz, ny, nx) row-major; u (nz, ny, nx + 1); v
// (nz, ny + 1, nx); geopotential (nz + 1, ny, nx).  The extended wind planes
// are ua (ny + 2, nx + 1) -- row jj is WRF j = jj (0 .. ny + 1), column iu
// is WRF u index iu + 1 -- and va (ny + 1, nx + 2) -- row jv is WRF v index
// jv + 1, column ii is WRF i = ii.  Fire-mesh fields are (nyf, nxf).
//
// Compiled by woof/fire/coupling.py with -fmad=false --ftz=false: no product
// is fused into a sum (gfortran -O0 fuses none) and subnormals stay.
// LOG and EXP are glibc's logf/expf (gfk_log/gfk_exp); divisions are
// __fdiv_rn.  WRF's REAL(4) literals are spelled as the float32 words.

#define FC_G 9.81f              // module_model_constants g

__device__ __forceinline__ float fc_div(float a, float b)
{ return __fdiv_rn(a, b); }

// continue_at_boundary's EX with bias 0.
__device__ __forceinline__ float fc_ex(float a, float b)
{
    const float bias = 0.0f;
    float lin = 2.0f * a - b;
    return (1.0f - bias) * lin + bias * fmaxf(fmaxf(lin, a), b);
}

// The log interpolation of one staggered column.  The column sits between
// mass columns a and b of the (nz + 1, ny, nx) geopotential; its height
// above the ground at level k is hgt(k) = 0.5*(altb(k)+altb(k+1)) -
// altb(0), altb(k) = 0.5*(altw_a(k)+altw_b(k)), WRF's hgtu/hgtv.  The
// wind is w[k * wstride].  WRF's layer loop, its exit and its fall-through.
__device__ float fc_column(const float *phi, long a, long b, long plane,
                           const float *w, long wstride, int nz, float zr,
                           float fwh)
{
    if (!(fwh > zr)) return 0.0f;
    float logfwh = gfk_log(fwh);
    float alt0 = 0.5f * (fc_div(phi[a], FC_G) + fc_div(phi[b], FC_G));
    float altk = alt0, hlast = 0.0f;
    for (int k = 0; k < nz; ++k) {
        long up = (long)(k + 1) * plane;
        float altk1 = 0.5f * (fc_div(phi[up + a], FC_G)
                              + fc_div(phi[up + b], FC_G));
        float ht = 0.5f * (altk + altk1) - alt0;
        if (!(ht < fwh)) {
            float loght = gfk_log(ht);
            if (k == 0) {
                float logz0 = gfk_log(zr);
                return fc_div(w[0] * (logfwh - logz0), loght - logz0);
            }
            float loglast = gfk_log(hlast);
            float wk = w[k * wstride], wl = w[(k - 1) * wstride];
            return wl + fc_div((wk - wl) * (logfwh - loglast),
                               loght - loglast);
        }
        hlast = ht;
        altk = altk1;
    }
    return w[(long)(nz - 1) * wstride];
}

// One thread per interior u column: WRF u index 2 .. nx (iu = 1 .. nx-1).
extern "C" __global__ void fc_wind_u(const float *u, const float *phi,
                                     const float *z0, float *ua, float fwh,
                                     int nz, int ny, int nx)
{
    int iu = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (iu > nx - 1 || j >= ny) return;
    float zr = 0.5f * (z0[j * nx + iu] + z0[j * nx + iu - 1]);
    ua[(j + 1) * (nx + 1) + iu] = fc_column(
        phi, (long)j * nx + iu - 1, (long)j * nx + iu, (long)ny * nx,
        u + (long)j * (nx + 1) + iu, (long)ny * (nx + 1), nz, zr, fwh);
}

// One thread per interior v column: WRF v index 2 .. ny (jv = 1 .. ny-1).
extern "C" __global__ void fc_wind_v(const float *v, const float *phi,
                                     const float *z0, float *va, float fwh,
                                     int nz, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int jv = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nx || jv > ny - 1) return;
    float zr = 0.5f * (z0[(jv - 1) * nx + i] + z0[jv * nx + i]);
    va[jv * (nx + 2) + i + 1] = fc_column(
        phi, (long)(jv - 1) * nx + i, (long)jv * nx + i, (long)ny * nx,
        v + (long)jv * nx + i, (long)(ny + 1) * nx, nz, zr, fwh);
}

// continue_at_boundary on both planes.  Every value written reads only the
// interior, so the threads are independent: t < ny is a ua row's two x ends,
// then nx - 1 ua columns' two y ends, the four ua corners, then the same for
// va.
extern "C" __global__ void fc_wind_extend(float *ua, float *va, int ny,
                                          int nx)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int wu = nx + 1, wv = nx + 2;
    if (t < ny) {                                   // ua, x direction
        int jj = t + 1;
        ua[jj * wu + 0] = fc_ex(ua[jj * wu + 1], ua[jj * wu + 2]);
        ua[jj * wu + nx] = fc_ex(ua[jj * wu + nx - 1], ua[jj * wu + nx - 2]);
        return;
    }
    t -= ny;
    if (t < nx - 1) {                               // ua, y direction
        int iu = t + 1;
        ua[0 * wu + iu] = fc_ex(ua[1 * wu + iu], ua[2 * wu + iu]);
        ua[(ny + 1) * wu + iu] = fc_ex(ua[ny * wu + iu],
                                       ua[(ny - 1) * wu + iu]);
        return;
    }
    t -= nx - 1;
    if (t == 0) {                                   // ua corners
        ua[0] = fc_ex(ua[1 * wu + 1], ua[2 * wu + 2]);
        ua[(ny + 1) * wu] = fc_ex(ua[ny * wu + 1], ua[(ny - 1) * wu + 2]);
        ua[nx] = fc_ex(ua[1 * wu + nx - 1], ua[2 * wu + nx - 2]);
        ua[(ny + 1) * wu + nx] = fc_ex(ua[ny * wu + nx - 1],
                                       ua[(ny - 1) * wu + nx - 2]);
        return;
    }
    t -= 1;
    if (t < ny - 1) {                               // va, x direction
        int jv = t + 1;
        va[jv * wv + 0] = fc_ex(va[jv * wv + 1], va[jv * wv + 2]);
        va[jv * wv + nx + 1] = fc_ex(va[jv * wv + nx], va[jv * wv + nx - 1]);
        return;
    }
    t -= ny - 1;
    if (t < nx) {                                   // va, y direction
        int ii = t + 1;
        va[ii] = fc_ex(va[1 * wv + ii], va[2 * wv + ii]);
        va[ny * wv + ii] = fc_ex(va[(ny - 1) * wv + ii],
                                 va[(ny - 2) * wv + ii]);
        return;
    }
    t -= nx;
    if (t == 0) {                                   // va corners
        va[0] = fc_ex(va[1 * wv + 1], va[2 * wv + 2]);
        va[ny * wv] = fc_ex(va[(ny - 1) * wv + 1], va[(ny - 2) * wv + 2]);
        va[nx + 1] = fc_ex(va[1 * wv + nx], va[2 * wv + nx - 1]);
        va[ny * wv + nx + 1] = fc_ex(va[(ny - 1) * wv + nx],
                                     va[(ny - 2) * wv + nx - 1]);
    }
}

// interpolate_2d at one fine node: (cx, tx) and (cy, ty) are the plane's
// column and row of the coarse cell's lower-left corner and the weights.
extern "C" __global__ void fc_wind_refine(const float *plane, int width,
                                          const int *cx, const float *tx,
                                          const int *cy, const float *ty,
                                          float *out, int nxf, int nyf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nxf || j >= nyf) return;
    float x = tx[i], y = ty[j];
    int a = cy[j] * width + cx[i];
    float v = (1.0f - x) * (1.0f - y) * plane[a];
    v = v + (1.0f - x) * y * plane[a + width];
    v = v + x * (1.0f - y) * plane[a + 1];
    v = v + x * y * plane[a + width + 1];
    out[j * nxf + i] = v;
}

// fire_tendency, fire_sfc_flx = 0, one thread per column; rho_in is
// phy_prep's density (fc_rho).  Columns on the
// domain edge, and the top layer, get zero (WRF's i_st/i_en, j_st/j_en and
// k_en - 1 bounds after its zero fill).  rth_slow is rthfrten over msft
// (msft == NULL: no map factor).
extern "C" __global__ void fc_tendency(
    const float *grnhfx, const float *grnqfx, const float *canhfx,
    const float *canqfx, const float *zs, const float *phi, const float *mu,
    const float *c1h, const float *c2h, const float *rho_in,
    const float *msft, float alfg, float alfc, float z1can,
    float *rthfrten, float *rqvfrten, float *rth_slow, int nz, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    long plane = (long)ny * nx;
    long c = (long)j * nx + i;
    if (i < 1 || i > nx - 2 || j < 1 || j > ny - 2) {
        for (int k = 0; k < nz; ++k) {
            rthfrten[k * plane + c] = 0.0f;
            rqvfrten[k * plane + c] = 0.0f;
            rth_slow[k * plane + c] = 0.0f;
        }
        return;
    }
    // cp = 7.*r_d/2. folded (1004.5), xlv = 2.5E6
    const float cp_i = fc_div(1.0f, 1004.5f);
    const float xlv_i = fc_div(1.0f, 2500000.0f);
    const float alfg_i = fc_div(1.0f, alfg);
    const float alfc_i = fc_div(1.0f, alfc);
    const float mu_c = mu[c];
    const float rmsf = msft == NULL ? 1.0f : msft[c];
    // Walk the w levels upward.  At level k: z_at_w(k) (zk), hfx(k) and
    // qfx(k); layer k - 1 then takes (hfx(k) - hfx(k-1)) / dz8w(k-1).
    float zprev = 0.0f, hprev = 0.0f, qprev = 0.0f;
    for (int k = 0; k < nz; ++k) {
        float zk = fc_div(phi[k * plane + c], FC_G);       // z_at_w
        float z_w = zk - zs[c];
        // WRF evaluates each EXP twice (heat, then vapour) on the same
        // argument; once is the same word.
        float eg = gfk_exp(-(alfg_i * z_w));
        bool below = z_w < z1can;
        float ec = below ? 0.0f : gfk_exp(-(alfc_i * (z_w - z1can)));
        float fact_g = cp_i * eg;
        float fact_c = below ? cp_i : cp_i * ec;
        float h = fact_g * grnhfx[c] + fact_c * canhfx[c];
        fact_g = xlv_i * eg;
        fact_c = below ? xlv_i : xlv_i * ec;
        float q = fact_g * grnqfx[c] + fact_c * canqfx[c];
        if (k > 0) {
            int kl = k - 1;
            long o = kl * plane + c;
            float dz8w = zk - zprev;
            float rho = rho_in[o];
            float rho_i = fc_div(1.0f, rho);
            float chm = c1h[kl] * mu_c + c2h[kl];
            float th = -fc_div(chm * rho_i * (h - hprev), dz8w);
            rthfrten[o] = th;
            rqvfrten[o] = -fc_div(chm * rho_i * (q - qprev), dz8w);
            rth_slow[o] = msft == NULL ? th : fc_div(th, rmsf);
        }
        zprev = zk; hprev = h; qprev = q;
    }
    long top = (long)(nz - 1) * plane + c;
    rthfrten[top] = 0.0f;
    rqvfrten[top] = 0.0f;
    rth_slow[top] = 0.0f;
}

// The fire's smoke as a surface flux (kg m-2 s-1) on the atmospheric grid.
// WRF adds, cell by cell (joff outer, ioff inner), avgw * fire_tracer_smoke
// * burnt_area_dt * fgip * 1000 / (rho dz8w) g/kg to the lowest level of an
// interior column; the flux is the same running sum of the first four
// factors -- kg of smoke per m2 of the atmospheric cell this step -- over
// dt.  ``factor`` is a fire-mesh field (the fuel's PM2.5 factor) or NULL
// for the scalar ``fts``.  Edge columns emit nothing, as in WRF.
extern "C" __global__ void fc_smoke_flux(const float *burnt,
                                         const float *fgip,
                                         const float *factor, float fts,
                                         float avgw, float dt, float *flux,
                                         int ir, int jr, int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    if (i < 1 || i > nx - 2 || j < 1 || j > ny - 2) {
        flux[j * nx + i] = 0.0f;
        return;
    }
    int nxf = nx * ir;
    float acc = 0.0f;
    for (int joff = 0; joff < jr; ++joff) {
        for (int ioff = 0; ioff < ir; ++ioff) {
            long f = (long)(j * jr + joff) * nxf + (i * ir + ioff);
            float s = factor == NULL ? fts : factor[f];
            float e = avgw * s;
            e = e * burnt[f];
            e = e * fgip[f];
            acc = acc + e;
        }
    }
    flux[j * nx + i] = fc_div(acc, dt);
}

// The actively burning area of each atmospheric cell (m2): the fire-mesh
// cells whose burning fraction FIRE_AREA is positive and whose node
// ignited less than one burn-out time (fuel_time, the 1/e time of
// fuel_left) before the step's end, each counting FIRE_AREA * dxf * dyf,
// summed in sum_2d_cells' order.
extern "C" __global__ void fc_active_area(const float *fire_area,
                                          const float *tign,
                                          const float *fuel_time,
                                          float time_now, float cell,
                                          float *area, int ir, int jr,
                                          int ny, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int nxf = nx * ir;
    float acc = 0.0f;
    for (int joff = 0; joff < jr; ++joff) {
        for (int ioff = 0; ioff < ir; ++ioff) {
            int jf = j * jr + joff, ifn = i * ir + ioff;
            long f = (long)jf * nxf + ifn;
            // tign carries the level set's one-node halo
            float tg = tign[(long)(jf + 1) * (nxf + 2) + ifn + 1];
            float a = fire_area[f];
            if (a > 0.0f && time_now - tg < fuel_time[f]) acc = acc + a;
        }
    }
    area[j * nx + i] = acc * cell;
}

// phy_prep's density, rho = 1./alt*(1.+qv) (module_big_step_utilities_em.F
// :4856), over n points.
extern "C" __global__ void fc_rho(const float *alt, const float *qv,
                                  float *rho, long n)
{
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    rho[t] = fc_div(1.0f, alt[t]) * (1.0f + qv[t]);
}
