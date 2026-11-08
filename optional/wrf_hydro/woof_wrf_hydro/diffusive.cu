// WOOF-OPTIN: wrf-hydro
// Derived from WRF-Hydro as distributed with WRF 4.6.1 (UCAR licence); see
// optional/wrf_hydro/LICENSE and NOTICE.md.  Not part of the default build or
// of any wheel.
//
// WRF-Hydro's diffusive-wave channel routing on nodes, the channel_option = 3
// branch of drive_CHANNEL (hydro/Routing/module_channel_routing.F90
// :873-1281) with DIFFUSION (:73-147), HEAD (:1315-1368), AREAf and SGNf, for
// a network without lakes or groundwater-channel exchange.  float32 as WRF-Hydro
// (REAL), the node mass sum QSUM8 in double as WRF-Hydro keeps it, the
// Fortran expressions in Fortran order, powf as glibc's (gfk_pow).  One
// thread block runs the whole call: each sub-step's passes over the nodes are
// separated by barriers, the scatters of drive_CHANNEL are ordered gathers
// over precomputed link lists (so every sum runs in WRF-Hydro's link order),
// and the Courant test that restarts the interval with half the step is a
// block-wide flag.

#define DIFF_ERR_AREA 1
#define DIFF_ERR_COEF 2
#define DIFF_ERR_TYPE 4

__device__ static float diff_sgnf(float val)
{
    if (val < 0.0f) return -1.0f;
    else if (val > 0.0f) return 1.0f;
    return 0.0f;
}

// DIFFUSION (:73-147)
__device__ static float diff_diffusion(float z1, float z20, float h1, float h2, float dx,
                                       float n, float Bw, float Cs, int* err)
{
    float z2;
    float dzx = (z1 - z20) / dx;
    if (dzx < 0.002f) z2 = z1 - dx * 0.002f;
    else z2 = z20;
    if (n <= 0.0f || Cs <= 0.0f || Bw <= 0.0f) atomicOr(err, DIFF_ERR_COEF);
    float Sf;
    if (fabsf(z1 - z2) > 1.0E5f) Sf = ((h1 - h2)) / dx;
    else Sf = ((z1 - z2) + (h1 - h2)) / dx;
    float sgn = diff_sgnf(Sf);
    float w = 0.5f * (sgn + 1.0f);
    float z = 1.0f / Cs;
    const float p23 = 2.0f / 3.0f;
    float R = ((Bw + z * h1) * h1) / (Bw + 2.0f * h1 * sqrtf(1.0f + z * z));
    float AREA = (Bw + z * h1) * h1;
    float Ku = (1.0f / n) * (gfk_pow(R, p23)) * AREA;
    R = ((Bw + z * h2) * h2) / (Bw + 2.0f * h2 * sqrtf(1.0f + z * z));
    AREA = (Bw + z * h2) * h2;
    float Kd = (1.0f / n) * (gfk_pow(R, p23)) * AREA;
    float Kf = (1.0f - w) * Kd + w * Ku;
    return Kf * sqrtf(fabsf(Sf)) * sgn;
}

__device__ static float diff_areaf(float AREA, float Bw, float h, float z)
{
    return (Bw + z * h) * h - AREA;
}

// HEAD (:1315-1368): the depth holding wetted area AREA, by bisection
__device__ static float diff_head(float AREA, float Bw, float z)
{
    float error = 1.0f;
    int maxiter = 0;
    float hl = 0.00001f, hu = 30.0f, hr, hrold, fl, fr, test;
    if (AREA < 0.00001f) {
        hr = 0.0f;
    } else {
        while ((diff_areaf(AREA, Bw, hl, z) * diff_areaf(AREA, Bw, hu, z)) > 0.0f && maxiter < 100) {
            if (AREA < 1.0f) hl = hl / 2.0f;
            else hu = hu * 2.0f;
            maxiter = maxiter + 1;
        }
        maxiter = 0;
        hr = 0.0f;
        fl = diff_areaf(AREA, Bw, hl, z);
        while (error > 0.0001f && maxiter < 1000) {
            hrold = hr;
            hr = (hl + hu) / 2.0f;
            fr = diff_areaf(AREA, Bw, hr, z);
            maxiter = maxiter + 1;
            if (hr != 0.0f) error = fabsf((hr - hrold) / hr);
            test = fl * fr;
            if (test < 0.0f) hu = hr;
            else if (test > 0.0f) { hl = hr; fl = fr; }
            else error = 0.0f;
        }
    }
    return hr;
}

// One drive_CHANNEL call, channel_option 3.  Node arrays are 0-based here;
// from_node/to_node keep WRF-Hydro's 1-based numbers (0: none).  tstart/tlist
// are, per node, the TYPEL 0 links that flow into it; fstart/flist the links
// that flow out of it, both in ascending link order.
extern "C" __global__ void wrfhydro_diffusive(
    const int* __restrict__ cell, const float* __restrict__ qstrmvolrt,
    const float* __restrict__ qinflowbase, const float* __restrict__ node_area,
    const int* __restrict__ from_node, const int* __restrict__ to_node,
    const int* __restrict__ typel, const float* __restrict__ zelev,
    const float* __restrict__ chanlen, const float* __restrict__ mannn,
    const float* __restrict__ so, const float* __restrict__ chsslp,
    const float* __restrict__ bw, const int* __restrict__ regular,
    const int* __restrict__ tstart, const int* __restrict__ tlist,
    const int* __restrict__ fstart, const int* __restrict__ flist,
    float* __restrict__ hlink, float* __restrict__ cvol, float* __restrict__ qlink1,
    float* __restrict__ qlateral, float* __restrict__ hlinktmp, float* __restrict__ cvoltmp,
    float* __restrict__ cd, float* __restrict__ qsum, float* __restrict__ dtct_io,
    int* __restrict__ err, float dt, float dtrt_ch, float retdep_chan, int nlinks)
{
    __shared__ float dtct;
    __shared__ int krt, dt_steps, flag;
    const float minDTCT = 0.01f;
    int tid = threadIdx.x, nt = blockDim.x;
    if (tid == 0) {
        float d = dtct_io[0] * 2.0f;                     // DTCT = min(max(DTCT*2.0, minDTCT),DTRT_CH)
        d = d > minDTCT ? d : minDTCT;
        dtct = dtrt_ch < d ? dtrt_ch : d;
        krt = 0;
    }
    for (int i = tid; i < nlinks; i += nt) {             // drive_CHANNEL :688-703, :883-884
        hlinktmp[i] = hlink[i];
        cvoltmp[i] = cvol[i];
        cd[i] = 0.0f;
        qlateral[i] = 0.0f;
    }
    __syncthreads();
    for (;;) {                                            // crnt
        if (tid == 0) {
            dt_steps = (int)(dt / dtct);
            flag = 99;
        }
        __syncthreads();
        float dtc = dtct;
        int steps = dt_steps;
        for (int i = tid; i < nlinks; i += nt) {
            qlateral[i] = 0.0f;
            if (node_area[i] == 0.0f) atomicOr(err, DIFF_ERR_AREA);
            if (regular[i]) {
                int c = cell[i];
                qlateral[i] = 0.0f + ((qstrmvolrt[c] + qinflowbase[c])
                                      / (float)steps * node_area[i] / 1000.0f / dtc);
            }
        }
        __syncthreads();
        for (int i = tid; i < nlinks; i += nt) {         // conveyances across node faces
            int fr = from_node[i], to = to_node[i];
            if (typel[i] == 0 && hlinktmp[fr - 1] > retdep_chan) {
                if (fr != to && to > 0 && fr > 0)
                    qlink1[i] = diff_diffusion(zelev[fr - 1], zelev[to - 1], hlinktmp[fr - 1],
                                               hlinktmp[to - 1], chanlen[i], mannn[i], bw[i],
                                               chsslp[i], err);
            } else {
                qlink1[i] = 0.0f;
            }
        }
        __syncthreads();
        for (int n = tid; n < nlinks; n += nt) {         // QSUM8 in, then QSUM out
            double s = 0.0;
            for (int p = tstart[n]; p < tstart[n + 1]; ++p) s = s + (double)qlink1[tlist[p]];
            float q = (float)s;
            for (int p = fstart[n]; p < fstart[n + 1]; ++p) q = q - qlink1[flist[p]];
            qsum[n] = q;
        }
        __syncthreads();
        for (int i = tid; i < nlinks; i += nt)            // the Courant test
            if (typel[i] == 0 && cvoltmp[i] >= 0.001f
                && (cvoltmp[i] - qsum[i] * dtc) / cvoltmp[i] <= -0.01f)
                atomicExch(&flag, -99);
        __syncthreads();
        bool restart = false;
        if (flag < 0 && dtc > 0.1f) {
            if (dtc > minDTCT) {
                restart = true;
                __syncthreads();
                if (tid == 0) {
                    float h = dtc / 2.0f;
                    dtct = h > minDTCT ? h : minDTCT;
                    krt = 0;
                }
                for (int i = tid; i < nlinks; i += nt) {
                    hlinktmp[i] = hlink[i];
                    cvoltmp[i] = cvol[i];
                }
            } else {
                __syncthreads();
                if (tid == 0) dtct = 0.1f;
                for (int i = tid; i < nlinks; i += nt) {
                    hlinktmp[i] = hlink[i];
                    cvoltmp[i] = cvol[i];
                }
            }
            __syncthreads();
        }
        if (restart) continue;
        dtc = dtct;
        for (int i = tid; i < nlinks; i += nt) {         // volume and depth at each node
            float qq = qsum[i] + qlateral[i];
            if (typel[i] == 0) {
                cvoltmp[i] = cvoltmp[i] + (qsum[i] + qlateral[i]) * dtc;
                if (cvoltmp[i] < 0.0f) cvoltmp[i] = 0.0f;
            } else if (typel[i] == 1) {
                if (!(qq < 0.0f)) cd[i] = hlinktmp[i];
                cvoltmp[i] = cvoltmp[i] + (qsum[i] + qlateral[i]
                    - diff_diffusion(zelev[i], zelev[i] - (so[i] * chanlen[i]), hlinktmp[i],
                                     cd[i], chanlen[i], mannn[i], bw[i], chsslp[i], err)) * dtc;
            } else {
                atomicOr(err, DIFF_ERR_TYPE);
            }
            if (typel[i] == 0)
                hlinktmp[i] = diff_head(cvoltmp[i] / chanlen[i], bw[i], 1.0f / chsslp[i]);
            else
                hlinktmp[i] = cd[i];
        }
        __syncthreads();
        for (int i = tid; i < nlinks; i += nt)            // the flows of the new depths
            if (typel[i] == 0) {
                int fr = from_node[i], to = to_node[i];
                qlink1[i] = diff_diffusion(zelev[fr - 1], zelev[to - 1], hlinktmp[fr - 1],
                                           hlinktmp[to - 1], chanlen[i], mannn[i], bw[i],
                                           chsslp[i], err);
            }
        __syncthreads();
        if (tid == 0) krt = krt + 1;
        __syncthreads();
        if (krt == dt_steps) break;
        __syncthreads();
    }
    for (int i = tid; i < nlinks; i += nt) {
        hlink[i] = hlinktmp[i];
        cvol[i] = cvoltmp[i];
    }
    if (tid == 0) dtct_io[0] = dtct;
}
