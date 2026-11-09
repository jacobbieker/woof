// WOOF-OPTIN: wrf-hydro
// Derived from WRF-Hydro (UCAR) as shipped in WRF 4.7.1 hydro/; see
// optional/wrf_hydro/LICENSE and NOTICE.md.  Not part of the default build or
// of any wheel.
//
// The one piece of HYDRO_exe (hydro/HYDRO_drv/module_HYDRO_drv.F90) the
// coupled driver needs beyond the routing units: RunoffAggregate (:1321-1346),
// the routing grid's infiltration excess back onto the LSM cell when overland
// routing is off and subsurface routing is on, summed in WRF's order (fine j
// then fine i, each from the cell's far corner) in REAL and divided by
// AGGFACTRT**2.  MPP_LAND on one process: left_id and down_id are negative.

extern "C" __global__
void hyd_runoff_aggregate(const float* __restrict__ runoff_in, float* __restrict__ runoff_out,
                          int ix, int jx, int agg)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ix * jx) return;
    int I = c % ix + 1, J = c / ix + 1;
    int ixrt = ix * agg;
    float acc = 0.0f;
    for (int ay = agg - 1; ay >= 0; --ay) {
        for (int ax = agg - 1; ax >= 0; --ax) {
            int ii = I * agg - ax, jj = J * agg - ay;
            acc = __fadd_rn(acc, runoff_in[(jj - 1) * ixrt + (ii - 1)]);
        }
    }
    runoff_out[c] = __fdiv_rn(acc, (float)(agg * agg));
}
