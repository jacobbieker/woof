// WOOF-OPTIN: wrf-hydro
// Derived from WRF-Hydro (UCAR) as shipped in WRF 4.7.1 hydro/; see
// optional/wrf_hydro/LICENSE and NOTICE.md.  Not part of the default build or
// of any wheel.
//
// WRF-Hydro's LSM <-> routing grid transforms: disaggregateDomain
// (hydro/Routing/Noah_distr_routing.F90:896-1241) and aggregateDomain
// (hydro/HYDRO_drv/module_HYDRO_drv.F90:1010-1286), without MPP_LAND.  One
// thread per LSM cell walks its AGGFACTRT x AGGFACTRT routing cells in WRF's
// order (AGGFACYRT, then AGGFACXRT, each from AGGFACTRT-1 down to 0), so each
// running sum, and the one place disaggregation rewrites the LSM cell's SICE
// mid-walk, happen as in WRF.  LSM planes are (jx, ix) and routing planes
// (jxrt, ixrt), C order with i fastest; soil fields carry nsoil in front.

#define GT_ERR_SMCMAX_DISAGG 1   // SMCMAX exceeded upon disaggregation3
#define GT_ERR_DEPLETED      2   // SMCRT depleted (and the layer not frozen)
#define GT_ERR_SMCMAX_AGG    4   // SMCMAX exceeded upon aggregation
#define GT_ERR_NEG_SH2OX     8   // negative SH2OX upon aggregation

extern "C" __global__
void hyd_disaggregate(int ix, int jx, int nsoil, int agg, int iswater,
                      float* __restrict__ sice, const float* __restrict__ smc,
                      const float* __restrict__ sh2ox, const float* __restrict__ infxsrt,
                      const float* __restrict__ area_lsm, const float* __restrict__ smcmax1,
                      const float* __restrict__ smcref1, const float* __restrict__ smcwlt1,
                      const int* __restrict__ vegtyp, const float* __restrict__ lksat,
                      const float* __restrict__ nexp, const float* __restrict__ dist9,
                      const float* __restrict__ infxswgt, const float* __restrict__ lksatfac,
                      const int* __restrict__ ch_netrt, const float* __restrict__ sh2owgt,
                      float* __restrict__ smcrefrt, float* __restrict__ infxsubrt,
                      float* __restrict__ smcmaxrt, float* __restrict__ smcwltrt,
                      float* __restrict__ smcrt, int* __restrict__ lake_mskrt,
                      float* __restrict__ lksatrt, float* __restrict__ nexprt,
                      const float* __restrict__ sldpth, int* __restrict__ soiltyprt,
                      const int* __restrict__ soiltyp, int* __restrict__ err)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int n = ix * jx;
    if (c >= n) return;
    int I = c % ix + 1, J = c / ix + 1;
    int ixrt = ix * agg, jxrt = jx * agg, nrt = ixrt * jxrt;
    // :1036 SICE = SMC - SH2OX (the whole array, before the loops)
    for (int k = 0; k < nsoil; ++k)
        sice[(size_t)k * n + c] = __fsub_rn(smc[(size_t)k * n + c], sh2ox[(size_t)k * n + c]);
    float lsmvol = __fmul_rn(infxsrt[c], area_lsm[c]);
    for (int ay = agg - 1; ay >= 0; --ay) {
        for (int ax = agg - 1; ax >= 0; --ax) {
            int ii = I * agg - ax, jj = J * agg - ay;
            int f = (jj - 1) * ixrt + (ii - 1);
            infxsubrt[f] = __fdiv_rn(__fmul_rn(lsmvol, infxswgt[f]), dist9[f]);
            for (int krt = 1; krt <= nsoil; ++krt) {
                size_t o = (size_t)(krt - 1) * nrt + f;
                float si = sice[(size_t)(krt - 1) * n + c];
                if (si > 0.0f) {
                    smcmaxrt[o] = __fsub_rn(smcmax1[c], si);
                    smcrefrt[o] = __fsub_rn(smcref1[c], si);
                    float watholdcap = __fsub_rn(smcmax1[c], smcwlt1[c]);
                    if (si <= watholdcap) {
                        smcwltrt[o] = smcwlt1[c];
                    } else {
                        if (si < smcmax1[c])
                            smcwltrt[o] = __fsub_rn(smcwlt1[c], __fsub_rn(si, watholdcap));
                        if (si >= smcmax1[c]) smcwltrt[o] = 0.0f;
                    }
                } else {
                    smcmaxrt[o] = smcmax1[c];
                    smcrefrt[o] = smcref1[c];
                    smcwltrt[o] = smcwlt1[c];
                }
                if (smcmaxrt[o] > 0.0f) {
                    smcrt[o] = __fmul_rn(sh2ox[(size_t)(krt - 1) * n + c], sh2owgt[o]);
                } else {
                    smcrt[o] = 0.001f;
                    smcmaxrt[o] = 0.001f;
                }
                if (smcrt[o] > smcmaxrt[o]) {
                    float excs = __fmul_rn(__fmul_rn(__fsub_rn(smcrt[o], smcmaxrt[o]),
                                                     sldpth[krt - 1]), 1000.0f);
                    smcrt[o] = smcmaxrt[o];
                    for (int kf = krt - 1; kf >= 1; --kf) {
                        size_t p = (size_t)(kf - 1) * nrt + f;
                        smcrt[p] = __fadd_rn(smcrt[p], __fdiv_rn(excs,
                                       __fmul_rn(sldpth[kf - 1], 1000.0f)));
                        if (smcrt[p] > smcmaxrt[p]) {
                            excs = __fmul_rn(__fmul_rn(__fsub_rn(smcrt[p], smcmaxrt[p]),
                                                       sldpth[kf - 1]), 1000.0f);
                            smcrt[p] = smcmaxrt[p];
                        } else {
                            excs = 0.0f;
                            break;
                        }
                    }
                    if (excs > 0.0f) {
                        infxsubrt[f] = __fadd_rn(infxsubrt[f], excs);
                        excs = 0.0f;
                    }
                }
            }
            for (int krt = 1; krt <= nsoil; ++krt) {
                size_t o = (size_t)(krt - 1) * nrt + f;
                size_t l = (size_t)(krt - 1) * n + c;
                if (smcrt[o] > smcmaxrt[o]) {
                    atomicOr(err, GT_ERR_SMCMAX_DISAGG);
                } else if (smcrt[o] <= 0.0f) {
                    if (fabsf(__fsub_rn(smc[l], sice[l])) <= 0.00001f) {
                        smcrt[o] = 0.001f;
                        sice[l] = smc[l];
                    } else {
                        atomicOr(err, GT_ERR_DEPLETED);
                    }
                }
            }
            size_t b = (size_t)(nsoil - 1) * nrt + f;
            float scale = __fdiv_rn(__fsub_rn(smcrt[b], smcrefrt[b]),
                                    __fsub_rn(smcmaxrt[b], smcrefrt[b]));
            scale = fmaxf(0.0f, scale);
            scale = fminf(1.0f, scale);
            lksatrt[f] = __fmul_rn(__fmul_rn(lksat[c], lksatfac[f]), scale);
            nexprt[f] = nexp[c];
            if (vegtyp[c] == iswater && ch_netrt[f] <= 0) lake_mskrt[f] = -9999;
            soiltyprt[f] = soiltyp[c];
        }
    }
}

// :1076-1079, after the loops: WHERE (ELRT < -9998) INFXSUBRT = 0
extern "C" __global__
void hyd_disaggregate_ocean(const float* __restrict__ elrt, float* __restrict__ infxsubrt,
                            int nrt)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nrt) return;
    if (elrt[f] < -9998.0f) infxsubrt[f] = 0.0f;
}

extern "C" __global__
void hyd_aggregate(int ix, int jx, int nsoil, int agg,
                   const float* __restrict__ head_rt, const float* __restrict__ dist9,
                   const float* __restrict__ smcrt, const float* __restrict__ smcmaxrt,
                   float* __restrict__ head_lsm, float* __restrict__ sh2ox,
                   const float* __restrict__ sice, float* __restrict__ smc,
                   float* __restrict__ infxswgt, float* __restrict__ sh2owgt,
                   int* __restrict__ err)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int n = ix * jx;
    if (c >= n) return;
    int I = c % ix + 1, J = c / ix + 1;
    int ixrt = ix * agg, nrt = ixrt * jx * agg;
    float sfchead = 0.0f, lsmvol = 0.0f;
    float sh2oagg[16];
    for (int k = 0; k < nsoil; ++k) sh2oagg[k] = 0.0f;
    for (int ay = agg - 1; ay >= 0; --ay)
        for (int ax = agg - 1; ax >= 0; --ax) {
            int f = (J * agg - ay - 1) * ixrt + (I * agg - ax - 1);
            sfchead = __fadd_rn(sfchead, head_rt[f]);
            lsmvol = __fadd_rn(lsmvol, __fmul_rn(head_rt[f], dist9[f]));
            for (int k = 0; k < nsoil; ++k)
                sh2oagg[k] = __fadd_rn(sh2oagg[k], smcrt[(size_t)k * nrt + f]);
        }
    float agg2 = (float)(agg * agg);
    head_lsm[c] = __fdiv_rn(sfchead, agg2);
    for (int k = 0; k < nsoil; ++k) sh2ox[(size_t)k * n + c] = __fdiv_rn(sh2oagg[k], agg2);
    for (int ay = agg - 1; ay >= 0; --ay)
        for (int ax = agg - 1; ax >= 0; --ax) {
            int f = (J * agg - ay - 1) * ixrt + (I * agg - ax - 1);
            if (lsmvol > 0.0f)
                infxswgt[f] = __fdiv_rn(__fmul_rn(head_rt[f], dist9[f]), lsmvol);
            else
                infxswgt[f] = __fdiv_rn(1.0f, agg2);
            for (int k = 0; k < nsoil; ++k) {
                size_t o = (size_t)k * nrt + f;
                float x = sh2ox[(size_t)k * n + c];
                if (__fsub_rn(smcrt[o], smcmaxrt[o]) > 0.000001f) atomicOr(err, GT_ERR_SMCMAX_AGG);
                if (x < 0.0f) atomicOr(err, GT_ERR_NEG_SH2OX);
                float w = (x > 0.0f) ? __fdiv_rn(smcrt[o], x) : 0.0f;
                sh2owgt[o] = fmaxf(1.0E-05f, w);
            }
        }
    // :1236 SMC = SH2OX + SICE
    for (int k = 0; k < nsoil; ++k)
        smc[(size_t)k * n + c] = __fadd_rn(sh2ox[(size_t)k * n + c], sice[(size_t)k * n + c]);
}
