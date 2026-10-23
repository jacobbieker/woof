// The Fitch et al. (2012) wind-farm drag from WRF v4.7.1
// (phys/module_wind_fitch.F, windfarm_opt = 1): the momentum the rotors
// extract, the turbulence they add, and the power they make.
//
//   fitch_column   dragforce (:61-237) with turbine_area (:247-274) and
//                  dragcof (:277-337) inlined.  One thread owns one column
//                  and every turbine group in it.
//
// A cell's turbines come as groups (CSR): groups offsets[col] ..
// offsets[col+1]-1 of the flat group arrays, each group `count` identical
// turbines of one curve, hub height and rotor diameter.  WRF loops over
// single turbines; a group is WRF's sequence for its identical turbines, and
// each group works over its own rotor levels.  Per turbine, as WRF orders it:
//   1. the levels the rotor cuts, from the cumulative layer depths (:121-139);
//   2. the hub-height speed, linear between the mass levels about the hub
//      (:144-171);
//   3. the thrust, power and TKE coefficients from the hub speed (dragcof):
//      Ct linear in the thrust table, Cp linear in the table's power
//      coefficients P/(0.5*1.23*V^3*A), C_TKE = factor*(Ct - Cp) >= 0;
//   4. the power at the hub, POWER1 = 0.5*1.23*Vhub^3*A*Cp, and the same
//      power summed over the rotor layers at their own speeds, POWER2;
//   5. per rotor layer, with ec = POWER1/POWER2 so the layers extract the
//      hub power:  du -= 0.5*u*Ct*|V|*A_k/(dx*dy*dz)*ec, dv likewise, and
//      QKE += |V|^3*A_k*C_TKE*dt/(dx*dy*dz)*ec.
//
// Deviations from WRF, each where WRF reads an undefined value or would
// produce a NaN; elsewhere every statement is WRF's, in WRF's order:
//   * hub below the first mass level: WRF scales the first-level speed by
//     hub/z1 with z1 never assigned on that path; z1 here is that first
//     mass level's height, 0.5*dz(1).
//   * hub above the top mass level: WRF leaves kbot/ktop unassigned; the
//     top level's speed is used.
//   * the clipped layer offsets z1, z2 are held in [0, D]: WRF clips z1
//     below and z2 above only, and a rotor edge lying a rounding error
//     outside its layer would give sqrt of a negative number.
//
// Fortran's real powers (u**2., speed**3.) are glibc's powf (gfk_pow), asin
// is glibc 2.43's correctly rounded asinf (glibc_asinf), sqrt is IEEE.  The
// unit is compiled with -fmad=false --ftz=false (woof/core/windfarm_fitch.py)
// so no product is fused into a sum and no subnormal is flushed.
//
// Layout: 3-D fields are (nz, ny, nx) mass levels, z_w (nz+1, ny, nx) full
// levels above sea level; offsets (ny*nx + 1); the group arrays (ngroups);
// the curve tables (ncurves, maxvals), zero-padded past each curve's nval
// entries.  rho, qke, thrust and overlap may be null.

__device__ __forceinline__ real wf_area_term(real zc, real r)
{
    // zc*sqrt(r*r-zc*zc)+r*r*asin(zc/r)  (turbine_area, :263-270)
    return zc * sqrtf(r * r - zc * zc) + r * r * glibc_asinf(zc / r);
}

// turbine_area (:247-274): the rotor disk's area between two heights z1 <
// z2 measured up from the lower blade tip, times wfdensity.
__device__ real wf_turbine_area(real z1, real z2, real tdiameter,
                                real wfdensity)
{
    real r = __fdiv_rn(tdiameter, 2.0f);
    z1 = r - z1;
    z2 = r - z2;
    real zc1 = fabsf(z1);
    real zc2 = fabsf(z2);
    real tarea;
    if (z1 > 0.0f && z2 > 0.0f) {
        tarea = wf_area_term(zc1, r) - wf_area_term(zc2, r);
    } else if (z1 < 0.0f && z2 < 0.0f) {
        tarea = wf_area_term(zc2, r) - wf_area_term(zc1, r);
    } else {
        // :269-270 has no parentheses: ((a2 + b2) + a1) + b1.
        tarea = wf_area_term(zc2, r) + zc1 * sqrtf(r * r - zc1 * zc1)
                + r * r * glibc_asinf(zc1 / r);
    }
    return tarea * wfdensity;
}

// The rotor's area in level k times wfdensity (:194-199 and :214-220): the
// level's interfaces as offsets above the lower blade tip, clipped to the
// rotor.  The last two clips are the deviation in the header.
__device__ real wf_layer_area(const real *z_w, size_t i, size_t ncol,
                              real blade_l, real zw0, real dia,
                              real wfdensity)
{
    real z1 = z_w[i] - blade_l - zw0;
    real z2 = z_w[i + ncol] - blade_l - zw0;
    if (z1 < 0.0f) z1 = 0.0f;
    if (z2 > dia) z2 = dia;
    if (z1 > dia) z1 = dia;
    if (z2 < 0.0f) z2 = 0.0f;
    return wf_turbine_area(z1, z2, dia, wfdensity);
}

__device__ __forceinline__ real wf_speed(const real *u, const real *v,
                                         size_t i)
{
    return sqrtf(gfk_pow(u[i], 2.0f) + gfk_pow(v[i], 2.0f));
}

extern "C" __global__
void fitch_column(const real *u, const real *v, const real *z_w,
                  const real *rho, const int *offsets, const int *g_count,
                  const int *g_curve, const real *g_hub, const real *g_dia,
                  const real *turbws, const real *turbtc, const real *turbpw,
                  const int *nval, const real *stc,
                  real *du, real *dv, real *qke, real *power, real *thrust,
                  real *overlap, real wfdensity, real dt, real tke_factor,
                  real piconst, int nz, int ny, int nx, int maxvals)
{
    size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t ncol = (size_t)ny * nx;
    if (col >= ncol) return;
    // power = 0. (:110); the diagnostics start from zero with it.
    real pw = 0.0f, th = 0.0f, swept = 0.0f;
    if (overlap) {
        for (int k = 0; k < nz; ++k) overlap[(size_t)k * ncol + col] = 0.0f;
    }
    real zw0 = z_w[col];
    for (int g = offsets[col]; g < offsets[col + 1]; ++g) {
        int nturb = g_count[g];
        if (nturb <= 0) continue;
        real hub = g_hub[g];
        real dia = g_dia[g];
        int t = g_curve[g];
        const real *ws = turbws + (size_t)t * maxvals;
        const real *tc = turbtc + (size_t)t * maxvals;
        const real *tp = turbpw + (size_t)t * maxvals;
        int nv = nval[t];
        real cutin = ws[0];
        real cutout = ws[nv - 1];
        real stc1 = stc[t];
        real stc2 = tc[nv - 1];

        // :117-118
        real half = __fdiv_rn(dia, 2.0f);
        real blade_l = hub - half;
        real blade_u = hub + half;

        // :121-139  levels cut by the blades, from the cumulative depths.
        int k_bot = -1, k_top = -1;
        bool kfound = false;
        real zl = 0.0f;
        for (int k = 0; k < nz && !kfound; ++k) {
            real dz = z_w[(size_t)(k + 1) * ncol + col]
                      - z_w[(size_t)k * ncol + col];
            real zu = zl + dz;
            if (blade_l >= zl && blade_l <= zu) k_bot = k;
            if (blade_u >= zl && blade_u <= zu) { k_top = k; kfound = true; }
            zl = zu;
        }
        // A rotor reaching above the column top is skipped, as WRF skips it.
        if (!kfound || k_bot < 0) continue;

        // :144-171  hub-height speed.
        int kb = nz - 1, kt = nz - 1;
        bool hfound = false;
        real z1 = 0.0f, z2 = 0.0f;
        zl = 0.0f;
        for (int k = 0; k < nz && !hfound; ++k) {
            real dz = z_w[(size_t)(k + 1) * ncol + col]
                      - z_w[(size_t)k * ncol + col];
            z2 = zl + 0.5f * dz;
            if (hub >= z2) {
                kb = k;
            } else {
                kt = k;
                hfound = true;
            }
            if (!hfound) z1 = z2;
            zl = z2 + 0.5f * dz;
        }
        real speedhub;
        if (!hfound) {
            // Deviation: the hub lies above the top mass level.
            speedhub = wf_speed(u, v, (size_t)(nz - 1) * ncol + col);
        } else if (kt == 0) {
            // Deviation: z1 is the first mass level, unassigned in WRF.
            z1 = z2;
            speedhub = wf_speed(u, v, col) * hub / z1;
        } else {
            real speed1 = wf_speed(u, v, (size_t)kb * ncol + col);
            real speed2 = wf_speed(u, v, (size_t)kt * ncol + col);
            speedhub = speed1 + ((speed2 - speed1) / (z2 - z1)) * (hub - z1);
        }

        // dragcof (:277-337)
        real area = __fdiv_rn(piconst, 4.0f) * gfk_pow(dia, 2.0f);
        int nu = 0, nb = 0;
        bool vfound = false;
        for (int k = 0; k < maxvals; ++k) {
            if (!vfound && ws[k] > speedhub) {
                nu = k; nb = k - 1; vfound = true;
            }
        }
        real thrcof, powcof;
        if (speedhub <= cutin) {
            thrcof = stc1;
        } else if (speedhub >= cutout) {
            thrcof = stc2;
        } else {
            thrcof = tc[nb] + (tc[nu] - tc[nb]) / (ws[nu] - ws[nb])
                              * (speedhub - ws[nb]);
        }
        if (speedhub <= cutin || speedhub >= cutout) {
            powcof = 0.0f;
        } else {
            real fac1 = 1000.0f / (0.5f * 1.23f * gfk_pow(ws[nb], 3.0f) * area);
            real fac2 = 1000.0f / (0.5f * 1.23f * gfk_pow(ws[nu], 3.0f) * area);
            powcof = tp[nb] * fac1 + (tp[nu] * fac2 - tp[nb] * fac1)
                                     / (ws[nu] - ws[nb]) * (speedhub - ws[nb]);
        }
        real tkecof = thrcof - powcof;
        tkecof = tke_factor * tkecof;
        if (tkecof < 0.0f) tkecof = 0.0f;

        // :175-177
        real power1 = 0.5f * 1.23f * gfk_pow(speedhub, 3.0f) * area * powcof;

        // :180-189  the same power at the layer speeds.
        real power2 = 0.0f;
        for (int k = k_bot; k <= k_top; ++k) {
            size_t i = (size_t)k * ncol + col;
            real tarea = wf_layer_area(z_w, i, ncol, blade_l, zw0, dia,
                                       wfdensity);
            real speed = wf_speed(u, v, i);
            power2 = power2 + 0.5f * powcof * 1.23f * gfk_pow(speed, 3.0f)
                              * tarea / wfdensity;
        }
        // :208-212
        real ec = (power1 == 0.0f || power2 == 0.0f) ? 1.0f : power1 / power2;

        // :176 and :193-232, once per identical turbine of the group; each
        // level accumulates on its own, so the level loop may lead.
        for (int n = 0; n < nturb; ++n) pw = power1 + pw;
        for (int k = k_bot; k <= k_top; ++k) {
            size_t i = (size_t)k * ncol + col;
            real tarea = wf_layer_area(z_w, i, ncol, blade_l, zw0, dia,
                                       wfdensity);
            real speed = wf_speed(u, v, i);
            real dz = z_w[i + ncol] - z_w[i];
            real dq = gfk_pow(speed, 3.0f) * tarea * tkecof * dt / dz * ec;
            real fu = 0.5f * u[i] * thrcof * speed * tarea / dz * ec;
            real fv = 0.5f * v[i] * thrcof * speed * tarea / dz * ec;
            real du_i = du[i], dv_i = dv[i];
            real qke_i = qke ? qke[i] : 0.0f;
            for (int n = 0; n < nturb; ++n) {
                qke_i = qke_i + dq;
                du_i = du_i - fu;
                dv_i = dv_i - fv;
            }
            du[i] = du_i;
            dv[i] = dv_i;
            if (qke) qke[i] = qke_i;
            if (overlap) {
                // The cell's swept area in the level, m2; normalised below.
                overlap[i] = overlap[i] + (real)nturb * (tarea / wfdensity);
            }
            if (thrust && rho) {
                // The layer's thrust, N: rho * |dV/dt| * dz / wfdensity.
                real f = rho[i] * (0.5f * thrcof * speed * speed * tarea * ec)
                         / wfdensity;
                for (int n = 0; n < nturb; ++n) th = th + f;
            }
        }
        swept = swept + (real)nturb * area;
    }
    if (overlap && swept > 0.0f) {
        for (int k = 0; k < nz; ++k) {
            size_t i = (size_t)k * ncol + col;
            overlap[i] = overlap[i] / swept;
        }
    }
    power[col] = pw;
    if (thrust) thrust[col] = th;
}
