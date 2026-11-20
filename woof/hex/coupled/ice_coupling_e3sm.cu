// E3SM v3.2.0 MPAS-Seaice coupler import and export (ice_import_mct,
// ice_export_mct).
//
// A modified derivative of E3SM v3.2.0 (E3SM-Project/E3SM, tag v3.2.0,
// 648b167b7f95267a159a9fdee25faf604c2bfc4f) and the Icepack it pins
// (CICE-Consortium/Icepack by way of E3SM-Project/Icepack, commit
// cf1d460fbb6a13d28988f798b33734f46d70a9dc), transcribed from
//   components/mpas-seaice/driver/ice_comp_mct.F (ice_import_mct,
//     ice_export_mct, basal_pressure, frazil_mass)
//   components/mpas-seaice/src/shared/mpas_seaice_mesh.F
//     (seaice_latlon_vector_rotation_backward and the routines it calls)
//   columnphysics/icepack_therm_shared.F90 (icepack_sea_freezing_temperature,
//     icepack_liquidus_temperature)
//   columnphysics/icepack_mushy_physics.F90 (liquidus_brine_salinity_mush,
//     liquidus_temperature_mush, icepack_enthalpy_mush,
//     icepack_mushy_liquid_fraction)
// E3SM: Copyright 2024, UChicago Argonne, LLC, BSD-3-Clause, the terms in
// licenses/LICENSE-E3SM-BSD-3-Clause.txt; MPAS-Seaice: Copyright (c) 2013,
// Los Alamos National Security, LLC (LANS) and the University Corporation
// for Atmospheric Research (UCAR), BSD-3-Clause, the terms in
// licenses/LICENSE-MPAS-BSD-3-Clause.txt; Icepack: Copyright 1998-2026,
// Triad National Security, LLC, BSD-3-Clause, the terms in
// licenses/LICENSE-Icepack-BSD-3-Clause.txt. This is not the version
// available from the E3SM, MPAS or CICE projects.
//
// real(r8) only. Every operation is an explicit round-to-nearest intrinsic in
// the Fortran's evaluation order (gfortran -O0 -ffp-contract=off); Icepack's
// merge() masks are multiplied in as the Fortran does. gfortran's MIN(a, b)
// and MAX(a, b) keep a unless b is strictly smaller (greater). The stress
// rotation's sin, cos, asin and atan2 are glibc's: their words come in as
// trig (8 per cell), so no libm runs here. One thread per cell; no cell
// reads another.
//
// Layout: an MPAS array (k, nCells) is a[i*k + m]; x2i/i2x's rAttr(nflds,
// lsize) is av[n*nflds + k]. The column indices are worked out on the host
// (mpassi_cpl_indices_set).

typedef double r8;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))

__device__ __forceinline__ r8 fmax_first(r8 a, r8 b) {
    return __longlong_as_double(b > a ? __double_as_longlong(b) : __double_as_longlong(a));
}
__device__ __forceinline__ r8 fmin_first(r8 a, r8 b) {
    return __longlong_as_double(b < a ? __double_as_longlong(b) : __double_as_longlong(a));
}

// Icepack's constants as MPAS-Seaice sets them (from seaice_constants)
struct IceConst {
    r8 rhoi, rhow, cp_ice, cp_ocn, Lfresh, depressT, Tocnfrz, puny;
};

// icepack_mushy_physics' liquidus (Assur 1958)
#define AZ1_LIQ (-18.48)
#define BZ1_LIQ 0.0
#define AZ2_LIQ (-10.3085)
#define BZ2_LIQ 62.4
#define TB_LIQ (-7.6362968855167352)
#define SB_LIQ 123.66702800276086

__device__ r8 liquidus_temperature_mush(r8 Sbr)
{
    const r8 az1p = D(AZ1_LIQ, 1000.0), az2p = D(AZ2_LIQ, 1000.0);
    const r8 M1 = AZ1_LIQ, N1 = -az1p, O1 = -D(BZ1_LIQ, AZ1_LIQ);
    const r8 M2 = AZ2_LIQ, N2 = -az2p, O2 = -D(BZ2_LIQ, AZ2_LIQ);
    const r8 t_high = Sbr <= SB_LIQ ? 1.0 : 0.0;
    return A(M(A(D(Sbr, A(M1, M(N1, Sbr))), O1), t_high),
             M(A(D(Sbr, A(M2, M(N2, Sbr))), O2), S(1.0, t_high)));
}

__device__ r8 liquidus_brine_salinity_mush(r8 zTin)
{
    const r8 bz1p = D(BZ1_LIQ, 1000.0), bz2p = D(BZ2_LIQ, 1000.0);
    const r8 J1 = D(BZ1_LIQ, AZ1_LIQ), K1 = D(1.0, 1000.0), L1 = D(A(1.0, bz1p), AZ1_LIQ);
    const r8 J2 = D(BZ2_LIQ, AZ2_LIQ), K2 = D(1.0, 1000.0), L2 = D(A(1.0, bz2p), AZ2_LIQ);
    const r8 t_high = zTin > TB_LIQ ? 1.0 : 0.0;
    const r8 lsubzero = zTin <= 0.0 ? 1.0 : 0.0;
    r8 Sbr = A(M(D(A(zTin, J1), A(M(K1, zTin), L1)), t_high),
               M(D(A(zTin, J2), A(M(K2, zTin), L2)), S(1.0, t_high)));
    return M(Sbr, lsubzero);
}

__device__ r8 icepack_liquidus_temperature(r8 Sin, int ktherm, const IceConst& c)
{
    return ktherm == 2 ? liquidus_temperature_mush(Sin) : -M(c.depressT, Sin);
}

__device__ r8 icepack_enthalpy_mush(r8 zTin, r8 zSin, const IceConst& c)
{
    const r8 Sbr = fmax_first(liquidus_brine_salinity_mush(zTin), c.puny);
    const r8 phi = D(zSin, fmax_first(Sbr, zSin));
    return S(A(M(M(phi, S(M(c.cp_ocn, c.rhow), M(c.cp_ice, c.rhoi))), zTin), M(M(c.rhoi, c.cp_ice), zTin)),
             M(M(S(1.0, phi), c.rhoi), c.Lfresh));
}

// the import's row codes
enum { R_COPY = 0, R_SUM = 1, R_ZERO = 2, R_SST = 3, R_TF = 4, R_FRAZIL = 5 };
#define NC 8

// ice_import_mct: one row of buf per pool array (a 2-D array's components
// are rows of their own); a row's value is a function of the cell's x2i.
extern "C" __global__ void ice_import_e3sm(
    r8* __restrict__ buf, int nCells, int nRows, const int* __restrict__ code, const int* __restrict__ nc,
    const int* __restrict__ c, const r8* __restrict__ x2i, int nx2i, int nCellsSolve,
    r8 freshWaterFreezingPoint, int tfrz, int ktherm, int mushy,
    r8 densityIce, r8 latentHeatMelting, r8 frazilSalinityReduction, r8 frazilIcePorosity,
    r8 rhoi, r8 rhow, r8 cp_ice, r8 cp_ocn, r8 Lfresh, r8 depressT, r8 Tocnfrz, r8 puny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCellsSolve) return;
    const IceConst k = {rhoi, rhow, cp_ice, cp_ocn, Lfresh, depressT, Tocnfrz, puny};
    const r8* x = x2i + (size_t)i * nx2i;
    for (int j = 0; j < nRows; ++j) {
        const int* cj = c + (size_t)j * NC;
        r8* dst = buf + (size_t)j * nCells + i;
        switch (code[j]) {
        case R_COPY: *dst = x[cj[0]]; break;
        case R_SUM: {
            r8 s = x[cj[0]];
            for (int m = 1; m < nc[j]; ++m) s = A(s, x[cj[m]]);
            *dst = s;
            break;
        }
        case R_ZERO: *dst = 0.0; break;
        // the second loop's seaSurfaceTemperature(i) - seaiceFreshWaterFreezingPoint
        case R_SST: *dst = S(x[cj[0]], freshWaterFreezingPoint); break;
        // icepack_sea_freezing_temperature(seaSurfaceSalinity(i))
        case R_TF: {
            const r8 sss = x[cj[0]];
            *dst = tfrz == 0 ? icepack_liquidus_temperature(sss, ktherm, k)
                 : tfrz == 1 ? -M(depressT, sss)
                 : tfrz == 2 ? Tocnfrz : -1.8;
            break;
        }
        // frazilMassFlux - frazil_mass(freezingMeltingPotential(i), seaSurfaceSalinity(i))
        case R_FRAZIL: {
            const r8 potential = x[cj[1]], sss = x[cj[2]];
            r8 rev;
            if (potential > 0.0) {
                r8 qi0new;
                if (mushy) {
                    r8 Si0new;
                    if (sss > M(2.0, frazilSalinityReduction)) Si0new = S(sss, frazilSalinityReduction);
                    else Si0new = D(M(sss, sss), M(4.0, frazilSalinityReduction));
                    const r8 Ti = icepack_liquidus_temperature(D(Si0new, frazilIcePorosity), ktherm, k);
                    qi0new = icepack_enthalpy_mush(Ti, Si0new, k);
                } else {
                    qi0new = -M(densityIce, latentHeatMelting);
                }
                rev = -D(M(potential, densityIce), qi0new);
            } else {
                rev = 0.0;
            }
            *dst = S(x[cj[0]], rev);
            break;
        }
        default: break;
        }
    }
}

// the export's i2x columns (-1 when not written)
enum { I_IFRAC, I_BERGW, I_BERGH, I_IFLOE, I_T, I_BPRESS, I_U10, I_U10G, I_TREF, I_QREF, I_SNOWH, I_ITHICK,
       I_TAUXA, I_TAUYA, I_LAT, I_SEN, I_LWUP, I_EVAP, I_SWNET, I_AVSDR, I_ANIDR, I_AVSDF, I_ANIDF, I_MELTH,
       I_SWPEN, I_MELTW, I_SALT, I_TAUXO, I_TAUYO, I_DUST1, I_N };
// the export's 1-D arrays, in this order: arr is (A_N, nCellsSolve)
enum { A_AREA, A_IVOL, A_SVOL, A_PDEPTH, A_PLID, A_PAREA, A_TSFC, A_AIRU, A_AIRV, A_OCNU, A_OCNV,
       A_AVSDR, A_ANIDR, A_AVSDF, A_ANIDF, A_SWABS, A_U10, A_TREF, A_QREF, A_FRZADJ, A_LAT, A_SEN, A_LWUP,
       A_EVAP, A_MELTH, A_SWPEN, A_MELTW, A_SALT, A_BERGW, A_BERGH, A_FLOE, A_DUST, A_N };

// seaice_latlon_vector_rotation_backward with its trigonometric words
// t = sin/cos of lat, of lon, of the rotated grid's lat, of its lon
__device__ void rotate_backward(r8& u, r8& v, r8 up, r8 vp, const r8* t, int rotate)
{
    const r8 sLat = t[0], cLat = t[1], sLon = t[2], cLon = t[3];
    const r8 sLatp = t[4], cLatp = t[5], sLonp = t[6], cLonp = t[7];
    // latlon_vector_to_xyz_vector on the rotated grid
    const r8 vxp = S(M(-up, sLonp), M(M(vp, sLatp), cLonp));
    const r8 vyp = S(M(up, cLonp), M(M(vp, sLatp), sLonp));
    const r8 vzp = M(vp, cLatp);
    // xyz_vector_rotation_backward
    r8 vx = vxp, vy = vyp, vz = vzp;
    if (rotate) {
        vx = vzp;
        vy = vyp;
        vz = -vxp;
    }
    // xyz_vector_to_latlon_vector
    u = A(M(-sLon, vx), M(cLon, vy));
    v = A(A(M(M(-sLat, cLon), vx), M(M(-sLat, sLon), vy)), M(cLat, vz));
}

extern "C" __global__ void ice_export_e3sm(
    r8* __restrict__ i2x, int ni2x, const int* __restrict__ col, int nCellsSolve,
    const r8* __restrict__ arr, const r8* __restrict__ trig, int rotate, int ponds, int columnSW,
    int bergs, int floe, int dust,
    r8 freshWaterFreezingPoint, r8 densityIce, r8 densitySnow, r8 densitySeaWater, r8 gravity,
    r8 referenceSalinity)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCellsSolve) return;
    r8* o = i2x + (size_t)i * ni2x;
    for (int k = 0; k < ni2x; ++k) o[k] = 0.0;
#define AR(n) arr[(size_t)(n) * nCellsSolve + i]
    const r8 ailohi = fmin_first(AR(A_AREA), 1.0);
    const r8 Tsrf = A(freshWaterFreezingPoint, AR(A_TSFC));
    r8 basalPressure = 0.0;
    if (ailohi > 0.0) {
        // basal_pressure
        r8 mass = A(M(AR(A_IVOL), densityIce), M(AR(A_SVOL), densitySnow));
        if (ponds)
            mass = A(A(mass, M(M(AR(A_PDEPTH), AR(A_PAREA)), densitySeaWater)),
                     M(M(AR(A_PLID), AR(A_PAREA)), densityIce));
        basalPressure = M(mass, gravity);
    }
    r8 tauxa, tauya, tauxo, tauyo;
    const r8* t = trig + (size_t)i * 8;
    rotate_backward(tauxa, tauya, AR(A_AIRU), AR(A_AIRV), t, rotate);
    rotate_backward(tauxo, tauyo, -AR(A_OCNU), -AR(A_OCNV), t, rotate);
    o[col[I_IFRAC]] = ailohi;
    if (bergs) {
        o[col[I_BERGW]] = AR(A_BERGW);
        o[col[I_BERGH]] = AR(A_BERGH);
    }
    if (floe) o[col[I_IFLOE]] = AR(A_FLOE);
    if (ailohi > 0.0) {
        o[col[I_T]] = Tsrf;
        o[col[I_BPRESS]] = basalPressure;
        o[col[I_U10]] = AR(A_U10);
        o[col[I_U10G]] = AR(A_U10);
        o[col[I_TREF]] = AR(A_TREF);
        o[col[I_QREF]] = AR(A_QREF);
        o[col[I_SNOWH]] = D(AR(A_SVOL), ailohi);
        o[col[I_ITHICK]] = D(AR(A_IVOL), ailohi);
        o[col[I_TAUXA]] = tauxa;
        o[col[I_TAUYA]] = tauya;
        o[col[I_LAT]] = AR(A_LAT);
        o[col[I_SEN]] = AR(A_SEN);
        o[col[I_LWUP]] = AR(A_LWUP);
        o[col[I_EVAP]] = AR(A_EVAP);
        o[col[I_SWNET]] = AR(A_SWABS);
        if (columnSW) {
            o[col[I_AVSDR]] = AR(A_AVSDR);
            o[col[I_ANIDR]] = AR(A_ANIDR);
            o[col[I_AVSDF]] = AR(A_AVSDF);
            o[col[I_ANIDF]] = AR(A_ANIDF);
        }
        o[col[I_MELTH]] = AR(A_MELTH);
        o[col[I_SWPEN]] = AR(A_SWPEN);
        o[col[I_MELTW]] = A(AR(A_MELTW), D(AR(A_FRZADJ), ailohi));
        o[col[I_SALT]] = A(AR(A_SALT), D(M(M(referenceSalinity, 0.001), AR(A_FRZADJ)), ailohi));
        o[col[I_TAUXO]] = tauxo;
        o[col[I_TAUYO]] = tauyo;
        if (dust) o[col[I_DUST1]] = AR(A_DUST);
    }
#undef AR
}
