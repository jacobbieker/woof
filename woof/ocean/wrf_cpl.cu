// WRF v4.7.1's external-ocean exchange (frame/module_cpl.F), one thread per
// column.  This is the atmosphere's half of WRF's OASIS coupling with NEMO:
//
//   cpl_snd2 (:298-382)  what WRF hands the ocean every step: EVAP-PRECIP,
//                        SURF_NET_SOLAR, SURF_NET_NON-SOLAR, TAUX, TAUY and
//                        TAUMOD, from the grid's surface fields;
//   cpl_rcv  (:408-455)  how WRF merges what the ocean hands back -- SST onto
//                        the input SST, UOCE and VOCE onto zero -- under the
//                        per-external-domain coupling mask CPLMASK.
//
// WRF is public domain (WRF LICENSE.txt); no OASIS or NEMO code is here.
// Every operation is spelled with an __f*_rn intrinsic in WRF's left-to-right
// order, so no contraction or reassociation is left to a compiler, and graded
// bitwise against the pinned source by tests/test_wrf_cpl_wrf461_parity.py
// (tools/wrf_cpl_wrf461_oracle).

// gfortran -O0 forms an integer power x**4 with libgcc's __powisf2: y = 1,
// then x = x*x and y = y*x at each set bit, i.e. (x*x)*(x*x).
__device__ __forceinline__ float wrf_cpl_pow4(float x)
{
    float x2 = __fmul_rn(x, x);
    return __fmul_rn(x2, x2);
}

// cpl_snd2.  stbolt is module_model_constants' STBOLT; raincv and rainncv are
// the step's precipitation in mm, so (RAINCV+RAINNCV)/DT is kg m-2 s-1.
extern "C" __global__
void wrf_cpl_snd(const float* __restrict__ qfx,
                 const float* __restrict__ raincv,
                 const float* __restrict__ rainncv,
                 const float* __restrict__ gsw,
                 const float* __restrict__ glw,
                 const float* __restrict__ emiss,
                 const float* __restrict__ sst,
                 const float* __restrict__ lh,
                 const float* __restrict__ hfx,
                 const float* __restrict__ u_phy,
                 const float* __restrict__ v_phy,
                 const float* __restrict__ uoce,
                 const float* __restrict__ voce,
                 const float* __restrict__ rho,
                 const float* __restrict__ ust,
                 float* __restrict__ evap_precip,
                 float* __restrict__ surf_net_solar,
                 float* __restrict__ surf_net_non_solar,
                 float* __restrict__ taux,
                 float* __restrict__ tauy,
                 float* __restrict__ taumod,
                 float dt, float stbolt, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // :325-326
    evap_precip[i] = __fsub_rn(qfx[i],
                               __fdiv_rn(__fadd_rn(raincv[i], rainncv[i]), dt));
    // :332
    surf_net_solar[i] = gsw[i];
    // :338-340
    surf_net_non_solar[i] = __fsub_rn(__fsub_rn(__fsub_rn(glw[i],
        __fmul_rn(__fmul_rn(stbolt, emiss[i]), wrf_cpl_pow4(sst[i]))),
        lh[i]), hfx[i]);
    // :349-352, the stress against the surface current
    float u_uo = __fsub_rn(u_phy[i], uoce[i]);
    float v_vo = __fsub_rn(v_phy[i], voce[i]);
    float wspd = fmaxf(__fsqrt_rn(__fadd_rn(__fmul_rn(u_uo, u_uo),
                                            __fmul_rn(v_vo, v_vo))), 1.e-7f);
    float taut = __fmul_rn(rho[i], __fmul_rn(ust[i], ust[i]));
    // :358, :365, :371
    taux[i] = __fdiv_rn(__fmul_rn(taut, u_uo), wspd);
    tauy[i] = __fdiv_rn(__fmul_rn(taut, v_vo), wspd);
    taumod[i] = taut;
}

// cpl_rcv for one field.  cplmask and received are (nedom, n): one plane per
// external domain, in WRF's domain order.  With obs (SST's SST_INPUT) the
// field starts as obs*(1-SUM(mask)), else at zero (:438-442); then each
// domain adds received*mask in turn (:444-449).  SUM is gfortran's
// left-to-right running sum from zero.
extern "C" __global__
void wrf_cpl_rcv(const float* __restrict__ cplmask,
                 const float* __restrict__ received,
                 const float* __restrict__ obs,
                 float* __restrict__ out,
                 int has_obs, int nedom, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float field = 0.0f;
    if (has_obs) {
        float masksum = 0.0f;
        for (int e = 0; e < nedom; ++e)
            masksum = __fadd_rn(masksum, cplmask[(size_t)e * n + i]);
        field = __fmul_rn(obs[i], __fsub_rn(1.0f, masksum));
    }
    for (int e = 0; e < nedom; ++e)
        field = __fadd_rn(field, __fmul_rn(received[(size_t)e * n + i],
                                           cplmask[(size_t)e * n + i]));
    out[i] = field;
}

// The coupling-interval average of a sent field: WRF calls oasis_put every
// step and the coupler averages over the coupling period (namcouple
// AVERAGE).  This is WOOF's own in-process accumulator, not OASIS code: it
// integrates field*dt in float32 with a Neumaier-compensated sum, so the
// integral handed to the ocean over any run equals the atmosphere's to
// float32 rounding of a single carried remainder.
extern "C" __global__
void wrf_cpl_accumulate(const float* __restrict__ field,
                        float* __restrict__ total,
                        float* __restrict__ carry,
                        float dt, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float add = __fmul_rn(field[i], dt);
    float s = total[i];
    float t = __fadd_rn(s, add);
    float c;
    if (fabsf(s) >= fabsf(add)) c = __fadd_rn(__fsub_rn(s, t), add);
    else c = __fadd_rn(__fsub_rn(add, t), s);
    total[i] = t;
    carry[i] = __fadd_rn(carry[i], c);
}

// Deliver mean = (total + carry) / elapsed and keep what the float32 mean
// could not carry: the remainder (total - mean*elapsed) + carry, formed from
// the two components (one fused multiply-add, so the product is not rounded
// first) rather than from their rounded sum, starts the next interval.
extern "C" __global__
void wrf_cpl_deliver(float* __restrict__ total,
                     float* __restrict__ carry,
                     float* __restrict__ mean,
                     float elapsed, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float integral = __fadd_rn(total[i], carry[i]);
    float m = __fdiv_rn(integral, elapsed);
    mean[i] = m;
    total[i] = __fadd_rn(fmaf(-m, elapsed, total[i]), carry[i]);
    carry[i] = 0.0f;
}
