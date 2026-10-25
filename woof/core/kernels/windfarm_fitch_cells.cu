// The Fitch et al. (2012) wind-farm drag on columns of unequal area: WRF
// v4.7.1's dragforce (phys/module_wind_fitch.F:61-237) for an unstructured
// mesh, where every column has its own horizontal area.
//
//   fitch_column_cells  fitch_column (windfarm_fitch.cu) with WRF's scalar
//                       wfdensity = 1/(dx*dx) (:109) replaced by the
//                       column's own wfdensity[col] = 1/area[col].
//
// This unit is compiled AFTER windfarm_fitch.cu in one NVRTC module
// (woof/core/windfarm_fitch.py, cells_module_source) and reuses its device
// helpers (wf_layer_area, wf_turbine_area, wf_speed) unchanged.  The body
// below is fitch_column's, statement for statement and in the same order;
// the one difference is where wfdensity is read.  So a column whose
// wfdensity word equals WRF's 1/(dx*dx) word gets every output word of
// fitch_column (tests/test_windfarm_fitch_cells.py holds the two kernels to
// max ULP 0 on the WRF v4.7.1 oracle columns and on random columns), and
// the three deviations fitch_column documents are this kernel's too.
//
// The drag is divided by the cell VOLUME area*dz, which is why the area
// takes dx*dx's place: Fitch's thrust 0.5*rho*Ct*A*|V|*V over the layer's
// air mass rho*area*dz.
//
// Layout: 3-D fields are (nz, ncol) levels (the (nz, ny, nx) of
// fitch_column with ny*nx = ncol), z_w (nz+1, ncol) interfaces above sea
// level; offsets (ncol + 1); the group arrays (ngroups); the curve tables
// (ncurves, maxvals); wfdensity (ncol).  rho, qke, thrust and overlap may be
// null.

extern "C" __global__
void fitch_column_cells(const real *u, const real *v, const real *z_w,
                        const real *rho, const int *offsets,
                        const int *g_count, const int *g_curve,
                        const real *g_hub, const real *g_dia,
                        const real *turbws, const real *turbtc,
                        const real *turbpw, const int *nval, const real *stc,
                        real *du, real *dv, real *qke, real *power,
                        real *thrust, real *overlap,
                        const real *wfdensity_col, real dt, real tke_factor,
                        real piconst, int nz, int ncol_in, int maxvals)
{
    size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t ncol = (size_t)ncol_in;
    if (col >= ncol) return;
    // power = 0. (:110); the diagnostics start from zero with it.
    real pw = 0.0f, th = 0.0f, swept = 0.0f;
    if (overlap) {
        for (int k = 0; k < nz; ++k) overlap[(size_t)k * ncol + col] = 0.0f;
    }
    // A column without turbines reads nothing else (its wfdensity is never
    // formed into a value).
    if (offsets[col] == offsets[col + 1]) {
        power[col] = pw;
        if (thrust) thrust[col] = th;
        return;
    }
    real wfdensity = wfdensity_col[col];
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
