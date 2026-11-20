// MPAS-Seaice v8.4.2: the column package's coupling preparation, the last
// thing seaice_column_postdynamics_time_integration does each step. It sets the
// cell albedos, the shortwave scaling factor and the ocean fluxes the coupler
// reads (per unit area), then rescales the fluxes from per grid cell area to
// per unit of ice area.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_seaice/shared/mpas_seaice_column.F:
//     3634-4041 seaice_column_coupling_prep (without the ocean mixed layer
//               and the column biogeochemistry, which are refused by name)
//     4055-4306 seaice_column_scale_fluxes (likewise without the
//               biogeochemistry)
//   with seaice_constants' seaicePuny, seaiceDensityFreshwater,
//   seaiceStefanBoltzmann and seaiceFreshWaterFreezingPoint
//   (mpas_seaice_constants.F, column/constants/cice/ice_constants_colpkg.F90).
// (Adrian K. Turner, LANL). Copyright (c) 2013-2019, Los Alamos National
// Security, LLC (LANS) and the University Corporation for Atmospheric Research
// (UCAR); BSD-3-Clause, the terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt.
// This is not the version available from LANS and UCAR.
//
// One source for both of MPAS's precisions: RKIND = 4 by default and
// RKIND = 8 when MPAS_RKIND8 is defined. Every operation is an explicit
// round-to-nearest intrinsic in the Fortran's evaluation order (gfortran -O0
// -ffp-contract=off: left to right, parentheses kept, no contraction).
// x**4 is (x*x)*(x*x), which is how libgcc's __powi and gfortran's inline
// expansion both evaluate it. A leading unary minus negates the whole product.
// The seaice_constants values arrive as arguments, each a dbl_kind parameter
// converted to RKIND.
//
// Layout: a(nCategories, nCells) is a[iCell*nCategories + n];
// iceAreaCategory(1, nCategories, nCells) is a[iCell*nCategories + n].

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define ONE 1.0
#define ZERO 0.0
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define ONE 1.0f
#define ZERO 0.0f
#endif

__device__ __forceinline__ int cp_thread() { return blockIdx.x * blockDim.x + threadIdx.x; }

// seaice_column_coupling_prep's loop over the cells (1..nCellsSolve of the tracers pool)
extern "C" __global__ void seaice_coupling_prep_v842(
    rk *freezingMeltingPotentialInitial, const rk *freezingMeltingPotential, const rk *iceAreaCategory,
    rk *albedoVisibleDirectCell, rk *albedoVisibleDiffuseCell, rk *albedoIRDirectCell, rk *albedoIRDiffuseCell,
    const rk *albedoVisibleDirectCategory, const rk *albedoVisibleDiffuseCategory,
    const rk *albedoIRDirectCategory, const rk *albedoIRDiffuseCategory,
    rk *albedoVisibleDirectArea, rk *albedoVisibleDiffuseArea, rk *albedoIRDirectArea, rk *albedoIRDiffuseArea,
    const rk *solarZenithAngleCosine, rk *bareIceAlbedoCell, rk *snowAlbedoCell, rk *pondAlbedoCell,
    const rk *bareIceAlbedoCategory, const rk *snowAlbedoCategory, const rk *pondAlbedoCategory,
    rk *effectivePondAreaCell, const rk *effectivePondAreaCategory,
    rk *pondFreshWaterFlux, rk *oceanFreshWaterFlux, const rk *oceanSaltFlux, const rk *oceanHeatFlux,
    const rk *oceanShortwaveFlux, rk *oceanFreshWaterFluxArea, rk *oceanSaltFluxArea, rk *oceanHeatFluxArea,
    rk *oceanShortwaveFluxArea, rk *shortwaveScalingFactor, const rk *shortwaveVisibleDirectDown,
    const rk *shortwaveVisibleDiffuseDown, const rk *shortwaveIRDirectDown, const rk *shortwaveIRDiffuseDown,
    int nCells, int nCategories, int pondFreshwaterFeedback, rk puny, rk densityFreshwater, rk dt)
{
    const int iCell = cp_thread();
    if (iCell >= nCells) return;
    freezingMeltingPotentialInitial[iCell] = freezingMeltingPotential[iCell];
    rk visDir = ZERO, visDif = ZERO, irDir = ZERO, irDif = ZERO;
    rk bare = ZERO, snow = ZERO, pond = ZERO, effectivePond = ZERO;
    const bool sunlit = solarZenithAngleCosine[iCell] > puny;
    for (int n = 0; n < nCategories; ++n) {
        const size_t at = (size_t)iCell * nCategories + n;
        const rk area = iceAreaCategory[at];
        visDir = A(visDir, M(albedoVisibleDirectCategory[at], area));
        visDif = A(visDif, M(albedoVisibleDiffuseCategory[at], area));
        irDir = A(irDir, M(albedoIRDirectCategory[at], area));
        irDif = A(irDif, M(albedoIRDiffuseCategory[at], area));
        if (sunlit) {
            bare = A(bare, M(bareIceAlbedoCategory[at], area));
            snow = A(snow, M(snowAlbedoCategory[at], area));
            pond = A(pond, M(pondAlbedoCategory[at], area));
        }
        effectivePond = A(effectivePond, M(effectivePondAreaCategory[at], area));
    }
    albedoVisibleDirectCell[iCell] = visDir;
    albedoVisibleDiffuseCell[iCell] = visDif;
    albedoIRDirectCell[iCell] = irDir;
    albedoIRDiffuseCell[iCell] = irDif;
    bareIceAlbedoCell[iCell] = bare;
    snowAlbedoCell[iCell] = snow;
    pondAlbedoCell[iCell] = pond;
    effectivePondAreaCell[iCell] = effectivePond;
    if (pondFreshwaterFeedback) {
        pondFreshWaterFlux[iCell] = D(M(pondFreshWaterFlux[iCell], densityFreshwater), dt);
        oceanFreshWaterFlux[iCell] = S(oceanFreshWaterFlux[iCell], pondFreshWaterFlux[iCell]);
    }
    albedoVisibleDirectArea[iCell] = visDir;
    albedoVisibleDiffuseArea[iCell] = visDif;
    albedoIRDirectArea[iCell] = irDir;
    albedoIRDiffuseArea[iCell] = irDif;
    oceanFreshWaterFluxArea[iCell] = oceanFreshWaterFlux[iCell];
    oceanSaltFluxArea[iCell] = oceanSaltFlux[iCell];
    oceanHeatFluxArea[iCell] = oceanHeatFlux[iCell];
    oceanShortwaveFluxArea[iCell] = oceanShortwaveFlux[iCell];
    shortwaveScalingFactor[iCell] =
        A(A(A(M(shortwaveVisibleDirectDown[iCell], S(ONE, visDir)),
              M(shortwaveVisibleDiffuseDown[iCell], S(ONE, visDif))),
            M(shortwaveIRDirectDown[iCell], S(ONE, irDir))),
          M(shortwaveIRDiffuseDown[iCell], S(ONE, irDif)));
}

// seaice_column_scale_fluxes' loop over the cells (1..nCellsSolve of tracers_aggregate)
extern "C" __global__ void seaice_scale_fluxes_v842(
    const rk *iceAreaCell, rk *airStressCellU, rk *airStressCellV, rk *sensibleHeatFlux, rk *latentHeatFlux,
    rk *absorbedShortwaveFlux, rk *longwaveUp, rk *evaporativeWaterFlux, rk *atmosReferenceTemperature2m,
    rk *atmosReferenceHumidity2m, rk *oceanFreshWaterFlux, rk *oceanSaltFlux, rk *oceanHeatFlux,
    rk *oceanShortwaveFlux, rk *albedoVisibleDirectCell, rk *albedoIRDirectCell, rk *albedoVisibleDiffuseCell,
    rk *albedoIRDiffuseCell, const rk *airTemperature, const rk *airSpecificHumidity,
    const rk *seaFreezingTemperature, int nCells, rk stefanBoltzmann, rk freshWaterFreezingPoint)
{
    const int iCell = cp_thread();
    if (iCell >= nCells) return;
// every field the routine rescales, in its order
#define CP_SCALED(X) X(airStressCellU) X(airStressCellV) X(sensibleHeatFlux) X(latentHeatFlux) \
    X(absorbedShortwaveFlux) X(longwaveUp) X(evaporativeWaterFlux) X(atmosReferenceTemperature2m) \
    X(atmosReferenceHumidity2m) X(oceanFreshWaterFlux) X(oceanSaltFlux) X(oceanHeatFlux) X(oceanShortwaveFlux) \
    X(albedoVisibleDirectCell) X(albedoIRDirectCell) X(albedoVisibleDiffuseCell) X(albedoIRDiffuseCell)
    if (iceAreaCell[iCell] > ZERO) {
        const rk iceAreaInverse = D(ONE, iceAreaCell[iCell]);
#define CP_SCALE(f) f[iCell] = M(f[iCell], iceAreaInverse);
        CP_SCALED(CP_SCALE)
    } else {
#define CP_ZERO(f) f[iCell] = ZERO;
        CP_SCALED(CP_ZERO)
        const rk t = A(seaFreezingTemperature[iCell], freshWaterFreezingPoint);
        const rk t2 = M(t, t);
        longwaveUp[iCell] = -M(stefanBoltzmann, M(t2, t2));
        atmosReferenceTemperature2m[iCell] = airTemperature[iCell];
        atmosReferenceHumidity2m[iCell] = airSpecificHumidity[iCell];
    }
}
