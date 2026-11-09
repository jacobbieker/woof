// WOOF-OPTIN: wrf-hydro
// Derived from WRF-Hydro as distributed with WRF 4.7.1 (UCAR licence); see
// optional/wrf_hydro/LICENSE and NOTICE.md.  Not part of the default build or
// of any wheel.
//
// lake_option 3's reservoirs (hydro/Routing/Reservoirs in WRF 4.7.1's
// hydro/), compiled after levelpool.cu:
//   run_hybrid_reservoir (Persistence_Level_Pool_Hybrid/
//     module_persistence_levelpool_hybrid.F90 :215-452), the persistence /
//     level-pool hybrid of USGS (type 2) and USACE (type 3) gages;
//   run_rfc_forecasts_reservoir (RFC_Forecasts/module_rfc_forecasts.F90
//     :204-431), RFC forecast outflow (type 4) or inflow plus forecast
//     (type 5);
//   modify_for_projected_storage (module_reservoir_utilities.F90 :118-164).
// Each runs its own level pool (lake_opt 1, as their init passes).  What the
// readers return is host data (woof_wrf_hydro.reservoir_da): the k-th
// timeslice read of a hybrid lake is obs_q[k * n + l] / obs_lb[k * n + l];
// the persistence coefficients and the RFC discharges sit in rtab.  The
// RFC's first-call checks (all synthetic, negative or >= 90000 values) are
// made on the host before the first call.  float32 in Fortran order;
// warnings are not printed.
//
// ri (RES_NI ints per lake) and rf (RES_NF floats per lake): see the RI_/RF_
// indices below.

#define RES_NI 16
#define RES_NF 16
#define RI_KIND 0          // 0 level pool, 1 hybrid, 2 RFC
#define RI_TYPE 1          // reservoir_type 1-5
#define RI_TIME 2          // current_time (s)
#define RI_UPDATE 3        // timeslice_update_time / time_series_update_time
#define RI_WUPDATE 4       // weight_update_time
#define RI_INDEX 5         // persistence_weight_index / time_series_index
#define RI_EVENT 6         // timeslice reads made so far
#define RI_COUNT 7         // number of coefficients / total_counts
#define RI_OFFSET 8        // into rtab
#define RI_INTERVAL 9      // observation_update_time_interval_seconds / time_step_seconds
#define RI_FOUND 10        // forecast_found
#define RI_PERSIST 11      // rfc_forecast_persist_seconds
#define RI_DYNAMIC 12      // dynamic_reservoir_type (out)
#define RI_WINTERVAL 13    // weight_update_time_interval (seconds_in_day)
#define RF_AREA 0          // lake_area * 1.0E6
#define RF_MINS 1          // min_storage
#define RF_MAX 2           // max_storage (hybrid) / max_water_elevation (RFC)
#define RF_STORAGE 3       // current_storage
#define RF_ELEV 4          // the reservoir's own water_elevation
#define RF_PW 5            // persistence_current_weight
#define RF_LW 6            // levelpool_current_weight
#define RF_PERSISTED 7     // persisted_outflow
#define RF_GAGE 8          // gage_discharge
#define RF_ASSIM 9         // assimilated_value (out)
#define RF_ORIFICE 10      // orifice_elevation
#define RES_ERR_EVENTS 64  // the host supplied too few timeslice reads

__device__ static void res_projected_storage(float inflow, float storage, float min_storage,
                                             float max_storage, int interval, float& outflow,
                                             bool& max_reached)
{
    if (outflow < 0.0f) outflow = 0.0f;
    float projected = storage + (inflow - outflow) * (float)interval;
    if (projected > max_storage) {
        max_reached = true;
    } else if (projected < min_storage && projected > 0.0f) {
        outflow = (projected - min_storage) / (float)interval + outflow;
    } else if (projected <= 0.0f) {
        outflow = inflow;
    }
    if (outflow < 0.0f) outflow = 0.0f;
}

__device__ static void res_levelpool(const float* p, float qi0, float qi1, float ql, float rp,
                                     float& H, float& qo)
{
    // area, we, wc, wl, dl, oe, oc, oa, maxh (levelpool.py's _PROPS order)
    levelpool_physics(1, qi0, qi1, qo, ql, rp, H, p[0], p[1], p[8], p[2], p[3], p[4], p[5],
                      p[6], p[7]);
}

__device__ static void res_hybrid(int* ri, float* rf, const float* rtab, const float* obs_q,
                                  const int* obs_lb, int nev, int n, int l, const float* p,
                                  float qi0, float qi1, float ql, float& H, float& qo, float rp,
                                  int* err)
{
    int lookback = 0;
    bool max_reached = false;
    if (rf[RF_ELEV] != H) {
        rf[RF_ELEV] = H;
        rf[RF_STORAGE] = (rf[RF_ELEV] - rf[RF_ORIFICE]) * rf[RF_AREA];
    }
    float local = rf[RF_ELEV];
    const float* coef = rtab + ri[RI_OFFSET];
    int ncoef = ri[RI_COUNT];
    if (ri[RI_TIME] >= ri[RI_UPDATE]) {
        int k = ri[RI_EVENT];
        if (k >= nev) {
            atomicOr(err, RES_ERR_EVENTS);
            k = nev - 1;
        }
        rf[RF_GAGE] = obs_q[(size_t)k * n + l];
        lookback = obs_lb[(size_t)k * n + l];
        ri[RI_EVENT] = ri[RI_EVENT] + 1;
        if (rf[RF_GAGE] < 0.0f) {
            if (ri[RI_TIME] >= ri[RI_WUPDATE]) {
                ri[RI_INDEX] = ri[RI_INDEX] + 1;
                ri[RI_WUPDATE] = ri[RI_WUPDATE] + ri[RI_WINTERVAL];
            }
            rf[RF_PW] = ri[RI_INDEX] > ncoef ? 0.0f : coef[ri[RI_INDEX] - 1];
        } else {
            rf[RF_PERSISTED] = rf[RF_GAGE];
            rf[RF_ASSIM] = rf[RF_GAGE];
            ri[RI_INDEX] = 1;
            rf[RF_PW] = coef[0];
            ri[RI_WUPDATE] = ri[RI_TIME] + ri[RI_WINTERVAL] - lookback;
        }
        rf[RF_LW] = 1.0f - rf[RF_PW];
        ri[RI_UPDATE] = ri[RI_UPDATE] + ri[RI_INTERVAL];
    } else if (ri[RI_TIME] >= ri[RI_WUPDATE]) {
        ri[RI_INDEX] = ri[RI_INDEX] + 1;
        rf[RF_PW] = ri[RI_INDEX] <= ncoef ? coef[ri[RI_INDEX] - 1] : 0.0f;
        rf[RF_LW] = 1.0f - rf[RF_PW];
        ri[RI_WUPDATE] = ri[RI_WUPDATE] + ri[RI_WINTERVAL];
    }
    float lp_out;
    res_levelpool(p, qi0, qi1, ql, rp, local, lp_out);
    if (rf[RF_LW] >= 1.0f - 1.1920929e-07f) {       // 1.0 - epsilon(1.0)
        ri[RI_DYNAMIC] = 1;
        rf[RF_ASSIM] = -9999.0f;
    } else {
        ri[RI_DYNAMIC] = ri[RI_TYPE];
    }
    float out = rf[RF_PW] * rf[RF_PERSISTED] + rf[RF_LW] * lp_out;
    res_projected_storage(qi1, rf[RF_STORAGE], rf[RF_MINS], rf[RF_MAX], (int)rp, out, max_reached);
    if (max_reached && out < lp_out) {
        out = lp_out;
        ri[RI_DYNAMIC] = 1;
        rf[RF_ASSIM] = -9999.0f;
    }
    float delta = (qi1 - out) * rp;
    rf[RF_STORAGE] = rf[RF_STORAGE] + delta;
    rf[RF_ELEV] = rf[RF_ELEV] + delta / rf[RF_AREA];
    qo = out;
    H = rf[RF_ELEV];
    ri[RI_TIME] = ri[RI_TIME] + (int)rp;
}

__device__ static void res_rfc(int* ri, float* rf, const float* rtab, const float* p, float qi0,
                               float qi1, float ql, float& H, float& qo, float rp)
{
    ri[RI_TIME] = ri[RI_TIME] + (int)rp;
    rf[RF_ELEV] = H;
    float lpwe = H, lp_out;
    res_levelpool(p, qi0, qi1, ql, rp, lpwe, lp_out);
    if (rp > 3600.0f) ri[RI_FOUND] = 0;
    const float* q = rtab + ri[RI_OFFSET];
    float out;
    bool type4 = ri[RI_TYPE] == 4;
    if (ri[RI_FOUND] && ri[RI_TIME] <= ri[RI_PERSIST]) {
        if (ri[RI_TIME] >= ri[RI_UPDATE] && ri[RI_INDEX] < ri[RI_COUNT]) {
            ri[RI_INDEX] = ri[RI_INDEX] + 1;
            ri[RI_UPDATE] = ri[RI_UPDATE] + ri[RI_INTERVAL];
        }
        out = type4 ? q[ri[RI_INDEX] - 1] : qi1 + q[ri[RI_INDEX] - 1];
        rf[RF_ELEV] = rf[RF_ELEV] + ((qi1 - out) / rf[RF_AREA]) * rp;
        if (rf[RF_ELEV] < 0.0f) rf[RF_ELEV] = 0.0f;
        else if (rf[RF_ELEV] > rf[RF_MAX]) rf[RF_ELEV] = rf[RF_MAX];
        ri[RI_DYNAMIC] = ri[RI_TYPE];
        rf[RF_ASSIM] = q[ri[RI_INDEX] - 1];
        if (out < 0.0f) {
            int m = ri[RI_INDEX];
            while (out < 0.0f && m > 1) {
                m = m - 1;
                out = q[m - 1];
            }
            if (out < 0.0f) {
                out = type4 ? lp_out : qi1;
                rf[RF_ELEV] = lpwe;
                ri[RI_DYNAMIC] = 1;
                rf[RF_ASSIM] = -9999.0f;
            }
        }
    } else {
        out = type4 ? lp_out : qi1;
        rf[RF_ELEV] = lpwe;
        ri[RI_DYNAMIC] = 1;
        rf[RF_ASSIM] = -9999.0f;
    }
    qo = out;
    H = rf[RF_ELEV];
}

// One reservoir call for lake l: the level pool (kind 0, lake_opt as given),
// the hybrid or the RFC reservoir.
__device__ static void reservoir_run(int l, int n, int lake_opt, int* ri_all, float* rf_all,
                                     const float* rtab, const float* obs_q, const int* obs_lb,
                                     int nev, const float* props, float qi0, float qi1, float ql,
                                     float& H, float& qo, float rp, int* err)
{
    const float* p = props + (size_t)l * 9;
    int* ri = ri_all + (size_t)l * RES_NI;
    float* rf = rf_all + (size_t)l * RES_NF;
    int kind = ri[RI_KIND];
    if (kind == 1)
        res_hybrid(ri, rf, rtab, obs_q, obs_lb, nev, n, l, p, qi0, qi1, ql, H, qo, rp, err);
    else if (kind == 2)
        res_rfc(ri, rf, rtab, p, qi0, qi1, ql, H, qo, rp);
    else
        levelpool_physics(lake_opt, qi0, qi1, qo, ql, rp, H, p[0], p[1], p[8], p[2], p[3], p[4],
                          p[5], p[6], p[7]);
}

// The reservoirs on their own, one thread per lake, for the parity test.
extern "C" __global__ void wrfhydro_reservoir_da(
    const float* __restrict__ qi0, const float* __restrict__ qi1, const float* __restrict__ ql,
    float* __restrict__ h, float* __restrict__ qo, const float* __restrict__ props,
    const int* __restrict__ lake_opt, int* ri, float* rf, const float* __restrict__ rtab,
    const float* __restrict__ obs_q, const int* __restrict__ obs_lb, int nev, float rp,
    int* err, int n)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= n) return;
    float H = h[l], q;
    reservoir_run(l, n, lake_opt[l], ri, rf, rtab, obs_q, obs_lb, nev, props, qi0[l], qi1[l],
                  ql[l], H, q, rp, err);
    h[l] = H;
    qo[l] = q;
}
