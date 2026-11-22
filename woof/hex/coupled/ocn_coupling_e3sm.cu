// E3SM v3.2.0 MPAS-Ocean coupler import and export (ocn_import_mct,
// ocn_export_mct).
//
// A modified derivative of E3SM v3.2.0 (E3SM-Project/E3SM, tag v3.2.0,
// 648b167b7f95267a159a9fdee25faf604c2bfc4f), transcribed from
//   components/mpas-ocean/driver/ocn_comp_mct.F (ocn_import_mct, ocn_export_mct)
//   components/mpas-ocean/src/shared/mpas_ocn_equation_of_state.F
//     (ocn_freezing_temperature)
// E3SM: Copyright 2024, UChicago Argonne, LLC, BSD-3-Clause, the terms in
// licenses/LICENSE-E3SM-BSD-3-Clause.txt; MPAS-Ocean: Copyright (c) 2013,
// Los Alamos National Security, LLC (LANS) and the University Corporation
// for Atmospheric Research (UCAR), BSD-3-Clause, the terms in
// licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version available
// from the E3SM or MPAS projects.
//
// real(r8) only. Every operation is an explicit round-to-nearest intrinsic in
// the Fortran's evaluation order (gfortran -O0 -ffp-contract=off). gfortran's
// MAX(a, b) and MIN(a, b) keep a unless b is strictly greater (smaller), so
// MAX(-0.0, 0.0) is -0.0 and MIN(-0.0, 0.0) is -0.0, as the oracle shows.
// One thread per cell; a cell's work is the Fortran's, in its order. No cell
// reads another, so the per-cell order gives the Fortran's words.
//
// Layout: an MPAS array (k, nCells) is a[i*k + m]; x2o/o2x's rAttr(nflds,
// lsize) is av[n*nflds + k]. The column indices are worked out on the host
// (mpaso_cpl_indices_set).

typedef double r8;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))

// chosen on the words as integer selects, so no compiler turns them into the
// IEEE min/max instructions (which order -0.0 below 0.0). NVRTC still lowers
// the MIN select to min.f64; that agrees with gfortran's MIN wherever the
// second argument is +0.0 or the positive pressure cap, as it always is here.
__device__ __forceinline__ r8 fmax_first(r8 a, r8 b) {
    return __longlong_as_double(b > a ? __double_as_longlong(b) : __double_as_longlong(a));
}
__device__ __forceinline__ r8 fmin_first(r8 a, r8 b) {
    return __longlong_as_double(b < a ? __double_as_longlong(b) : __double_as_longlong(a));
}

// the import's row codes
enum { R_NONE = -1, R_COPY = 0, R_TAU_WAVE = 1, R_ROFL = 2, R_ROFI = 3, R_SWNET = 4, R_RAIN = 5, R_BPRESS = 6,
       R_SQRT = 7 };

// ocn_import_mct, one row of buf per forcing array (a 2-D array's
// components are rows of their own). zeroFirst rows are zeroed over every
// cell (the removed runoff, the 10 m wind) before the cells' loop.
extern "C" __global__ void ocn_import_e3sm(
    r8* __restrict__ buf, int nCells, int nRows,
    const int* __restrict__ code, const int* __restrict__ c1, const int* __restrict__ c2,
    const int* __restrict__ c3, const int* __restrict__ zeroFirst,
    const r8* __restrict__ x2o, int nx2o, int nCellsSolve, const r8* __restrict__ latCell,
    r8 precFactor, r8 density0, r8 gravity)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    for (int j = 0; j < nRows; ++j)
        if (zeroFirst[j]) buf[(size_t)j * nCells + i] = 0.0;
    if (i >= nCellsSolve) return;
    const r8* x = x2o + (size_t)i * nx2o;
    for (int j = 0; j < nRows; ++j) {
        r8* dst = buf + (size_t)j * nCells + i;
        switch (code[j]) {
        case R_COPY: *dst = x[c1[j]]; break;
        case R_TAU_WAVE: *dst = S(x[c1[j]], S(x[c2[j]], x[c3[j]])); break;
        case R_ROFL:
        case R_ROFI:
            *dst = x[c1[j]];
            // c2: the removed row, or -1 when the removal is off
            if (c2[j] >= 0 && latCell[i] < (code[j] == R_ROFL ? -1.04719666667 : -0.99483767345)) {
                buf[(size_t)c2[j] * nCells + i] = *dst;
                *dst = 0.0;
            }
            break;
        case R_SWNET: *dst = fmax_first(x[c1[j]], 0.0); break;
        case R_RAIN: *dst = M(x[c1[j]], precFactor); break;
        case R_BPRESS: *dst = fmin_first(x[c1[j]], M(M(density0, gravity), 5.0)); break;
        case R_SQRT: *dst = __dsqrt_rn(x[c1[j]]); break;
        default: break;
        }
    }
}

// the export's o2x columns (-1 when not written)
enum { O_T, O_S, O_U, O_V, O_SSH, O_DHDX, O_DHDY, O_H2OTEMP, O_ISMW, O_ISMH, O_RROFL, O_RROFI, O_RROFIH,
       O_Q, O_FRAZIL, O_Q_LI, O_FRAZIL_LI, O_BLT, O_BLS, O_HTV, O_STV, O_RHOEFF, O_N };

__device__ __forceinline__ r8 freezing_open_ocean(r8 salinity, r8 pressure, const r8* fc)
{
    // ocn_freezing_temperature(salinity, pressure, inLandIceCavity=.false.)
    r8 coeff_mushy = D(1.0, fc[4]);
    return A(A(A(A(fc[0], M(fc[1], salinity)), M(fc[2], pressure)), M(M(fc[3], pressure), salinity)),
             D(M(coeff_mushy, salinity), S(1.0, D(salinity, 1e3))));
}

// ocn_export_mct. liMode: 0 none, 1 standalone, 2 data, 3 coupled.
extern "C" __global__ void ocn_export_e3sm(
    r8* __restrict__ o2x, int no2x, const int* __restrict__ col, int nCellsSolve,
    const r8* __restrict__ tsv, int nTsv, int iT, int iS,
    const r8* __restrict__ vel, int nVel, int iU, int iV,
    const r8* __restrict__ ssh, const r8* __restrict__ grad, int nGrad, int iDX, int iDY,
    const r8* __restrict__ fwtf, r8 rho_sw, r8 cp_sw, r8 T0_Kelvin,
    int liMode, const r8* __restrict__ liFW, const r8* __restrict__ liHeat,
    int removeRiver, const r8* __restrict__ remRiver, int removeIce, const r8* __restrict__ remIce,
    const r8* __restrict__ remIceHeat,
    int frazilOn, int hasMask, const int* __restrict__ landIceMask, r8* __restrict__ seaIceEnergy,
    r8* __restrict__ frazilSurfacePressure, r8* __restrict__ accumulated, const r8* __restrict__ layerThickness,
    int nVert, r8 heatOfFusion, r8 dt, const r8* __restrict__ freezing,
    const r8* __restrict__ blTracers, int nBLT, int iBLT, int iBLS,
    const r8* __restrict__ transferVel, int nTV, int iHT, int iST,
    int glc3d, int nGlcZ, const int* __restrict__ tfCol, const int* __restrict__ tfMaskCol,
    const r8* __restrict__ tf, const r8* __restrict__ tfMask, int nZDim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCellsSolve) return;
    r8* o = o2x + (size_t)i * no2x;
    const r8* t = tsv + (size_t)i * nTsv;
    o[col[O_T]] = t[iT];
    o[col[O_S]] = t[iS];
    o[col[O_U]] = vel[(size_t)i * nVel + iU];
    o[col[O_V]] = vel[(size_t)i * nVel + iV];
    o[col[O_SSH]] = ssh[i];
    o[col[O_DHDX]] = grad[(size_t)i * nGrad + iDX];
    o[col[O_DHDY]] = grad[(size_t)i * nGrad + iDY];
    o[col[O_H2OTEMP]] = M(M(fwtf[i], rho_sw), cp_sw);
    bool standaloneOrData = liMode == 1 || liMode == 2;
    if (standaloneOrData) {
        o[col[O_ISMW]] = liFW[i];
        o[col[O_ISMH]] = liHeat[i];
    }
    if (removeRiver) o[col[O_RROFL]] = remRiver[i];
    if (removeIce) {
        o[col[O_RROFI]] = remIce[i];
        o[col[O_RROFIH]] = remIceHeat[i];
    }
    if (frazilOn) {
        bool keepFrazil = true;
        if (hasMask && landIceMask[i] == 1) keepFrazil = false;
        if (keepFrazil) {
            if (accumulated[i] > 0.0) {
                seaIceEnergy[i] = M(accumulated[i], heatOfFusion);
            } else {
                r8 surfaceFreezingTemp = freezing_open_ocean(t[iS], 0.0, freezing);
                seaIceEnergy[i] = fmin_first(
                    M(M(M(rho_sw, cp_sw), layerThickness[(size_t)i * nVert]),
                      S(A(surfaceFreezingTemp, T0_Kelvin), t[iT])), 0.0);
            }
            o[col[O_Q]] = D(seaIceEnergy[i], dt);
            o[col[O_FRAZIL]] = D(accumulated[i], dt);
        } else {
            o[col[O_Q]] = 0.0;
            o[col[O_FRAZIL]] = 0.0;
            if (standaloneOrData) {
                o[col[O_Q_LI]] = D(M(accumulated[i], heatOfFusion), dt);
                o[col[O_FRAZIL_LI]] = D(accumulated[i], dt);
            }
        }
        seaIceEnergy[i] = 0.0;
        accumulated[i] = 0.0;
        frazilSurfacePressure[i] = 0.0;
    }
    if (liMode == 1 || liMode == 3) {
        o[col[O_BLT]] = blTracers[(size_t)i * nBLT + iBLT];
        o[col[O_BLS]] = blTracers[(size_t)i * nBLT + iBLS];
        o[col[O_HTV]] = transferVel[(size_t)i * nTV + iHT];
        o[col[O_STV]] = transferVel[(size_t)i * nTV + iST];
        o[col[O_RHOEFF]] = 0.0;
    }
    if (glc3d) {
        for (int l = 0; l < nGlcZ; ++l) {
            r8 m = tfMask[(size_t)i * nZDim + l];
            if (m > 0.9) {
                o[tfCol[l]] = D(tf[(size_t)i * nZDim + l], m);
                o[tfMaskCol[l]] = 1.0;
            } else {
                o[tfCol[l]] = 0.0;
                o[tfMaskCol[l]] = 0.0;
            }
        }
    }
}
