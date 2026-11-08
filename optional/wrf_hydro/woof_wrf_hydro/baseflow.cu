// WOOF-OPTIN: wrf-hydro
// Derived from WRF-Hydro (UCAR) as shipped in WRF 4.6.1 hydro/; see
// optional/wrf_hydro/LICENSE and NOTICE.md.  Not part of the default build or
// of any wheel.
//
// WRF-Hydro's groundwater bucket, simp_gw_buck in
// hydro/Routing/module_GW_baseflow.F90 (:282-514), without MPP_LAND.  One
// thread per basin sums its LSM cells' drainage (and, without overland
// routing, surface runoff) in REAL*8 in WRF's i-outer, j-inner order through
// a CSR list built on the host; the bucket level is REAL*8 inside the call
// and REAL between calls, as in WRF; EXP is glibc's binary64 exp.

extern "C" __global__
void gw_bucket(const int* __restrict__ start, const int* __restrict__ cells,
               const float* __restrict__ runoff1, const float* __restrict__ runoff2,
               const float* __restrict__ area, const float* __restrict__ C,
               const float* __restrict__ ex, const float* __restrict__ z_mx,
               float* __restrict__ z_tmp, float* __restrict__ qin_out,
               float* __restrict__ qout_out, float dt, int sw, int ovrt, int nb)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    double sum8 = 0.0, sum8s = 0.0, ct8 = 0.0;
    for (int p = start[b]; p < start[b + 1]; ++p) {
        int c = cells[p];
        float r2 = runoff2[c], r1 = runoff1[c];
        if ((r2 < 0.0f) || (r2 > 1000.0f)) r2 = 0.0f;
        if ((r1 < 0.0f) || (r1 > 1000.0f)) r1 = 0.0f;
        if (ovrt == 0) sum8s = __dadd_rn(sum8s, (double)r1);
        sum8 = __dadd_rn(sum8, (double)r2);
        ct8 = __dadd_rn(ct8, 1.0);
    }
    float sum = __double2float_rn(sum8), sums = __double2float_rn(sum8s);
    float ct = __double2float_rn(ct8);
    float ca = __fmul_rn(ct, area[b]);
    float net = __fdiv_rn(sum, ct);
    float qin = __fdiv_rn(__fmul_rn(__fmul_rn(__fdiv_rn(net, 1000.0f), ct), area[b]), dt);
    float nets = __fdiv_rn(sums, ct);
    float qins = __fdiv_rn(__fmul_rn(__fmul_rn(__fdiv_rn(nets, 1000.0f), ct), area[b]), dt);
    double z = (double)z_tmp[b];
    z = __dadd_rn(z, (double)__fmul_rn(__fdiv_rn(__fmul_rn(qin, dt), ca), 1000.0f));
    float qout = 0.0f, qout_spill = 0.0f;
    if (sw == 1) {
        float z_gw_spill;
        if (z > (double)z_mx[b]) {
            z_gw_spill = __double2float_rn(__dsub_rn(z, (double)z_mx[b]));
            z = (double)z_mx[b];
        } else {
            z_gw_spill = 0.0f;
        }
        qout_spill = __fdiv_rn(__fmul_rn(__fdiv_rn(z_gw_spill, 1000.0f), ca), dt);
        float qout_max = __double2float_rn(__ddiv_rn(__dmul_rn(__ddiv_rn(z, 1000.0), (double)ca),
                                                      (double)dt));
        double e = glibc_exp(__ddiv_rn(__dmul_rn((double)ex[b], z), (double)z_mx[b]));
        qout = __double2float_rn(__dmul_rn((double)C[b], __dsub_rn(e, 1.0)));
        qout = fminf(qout_max, qout);
    } else {
        // GWBASESWCRT = 2.  WRF's qout_spill is set only in the exponential
        // arm, so here it holds whatever the stack held; the port's is 0.
        qout = qin;
    }
    z = __dsub_rn(z, (double)__fmul_rn(__fdiv_rn(__fmul_rn(qout, dt), ca), 1000.0f));
    qout = __fadd_rn(__fadd_rn(qout, qout_spill), qins);
    z_tmp[b] = __double2float_rn(z);
    qin_out[b] = qin;
    qout_out[b] = qout;
}

// qinflowbase on the routing grid (:500-509): each stream pixel of basin b
// gets that basin's outflow spread over its pixels, in mm per step.
extern "C" __global__
void gw_inflowbase(const int* __restrict__ lind, const float* __restrict__ qout,
                   const float* __restrict__ pixct, const float* __restrict__ dist9,
                   float* __restrict__ qinflowbase, float dt, int nrt)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nrt) return;
    float q = 0.0f;
    int b = lind[c];
    if (b > 0)
        q = __fdiv_rn(__fdiv_rn(__fmul_rn(__fmul_rn(qout[b - 1], 1000.0f), dt), pixct[b - 1]),
                      dist9[c]);
    qinflowbase[c] = q;
}
