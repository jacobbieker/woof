// MPAS land-ice (MALI) v8.4.2: calving (li_calve_ice) and the calving-front
// restoration (li_restore_calving_front).
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_landice/mode_forward/mpas_li_calving.F:78-276 (li_calve_ice), 293-544
//   (li_restore_calving_front), 582-820 (thickness_calving), 822-884 (floating_calving),
//   886-964 (topographic_calving), 966-1173 (eigencalving), 1175-1328
//   (distribute_calving_flux), 1330-1395 (calculate_calving_front_mask)
// Copyright (c) 2013-2019, Los Alamos National Security, LLC (LANS) and the
// University Corporation for Atmospheric Research (UCAR); BSD-3-Clause, the
// terms in licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version
// available from LANS and UCAR.
//
// RKIND = 4 unless MPAS_RKIND8 is defined. Every operation is an explicit
// round-to-nearest intrinsic in the Fortran's evaluation order (gfortran -O0
// -ffp-contract=off). Fortran MIN/MAX keep the first argument unless the
// second compares less/greater (x86-64 minss/maxss), chosen on the words.
//
// Layout: cellsOnCell(maxEdges, nCells) is c[iCell*maxEdges + j]; a
// (nVertLevels, nCells) field is a[iCell*nVertLevels + k]. Connectivity is
// MPAS's 1-based indices.

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define LI_C(x) x
__device__ __forceinline__ rk li_min(rk a, rk b) {
    return __longlong_as_double(b < a ? __double_as_longlong(b) : __double_as_longlong(a));
}
__device__ __forceinline__ rk li_max(rk a, rk b) {
    return __longlong_as_double(b > a ? __double_as_longlong(b) : __double_as_longlong(a));
}
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define LI_C(x) x##f
__device__ __forceinline__ rk li_min(rk a, rk b) {
    return __int_as_float(b < a ? __float_as_int(b) : __float_as_int(a));
}
__device__ __forceinline__ rk li_max(rk a, rk b) {
    return __int_as_float(b > a ? __float_as_int(b) : __float_as_int(a));
}
#endif

#define LI_ICE 32
#define LI_DYNAMIC_ICE 2
#define LI_FLOATING 4
#define LI_DYNAMIC_MARGIN 16
#define LI_INITIAL_ICE_EXTENT 1

__device__ __forceinline__ bool li_is(int mask, int value) { return (mask & value) == value; }

// thickness_calving's activeForCalvingMask, one thread per cell
extern "C" __global__ void li_calving_active_v842(
    int *active, const rk *thickness, const rk *bedTopography, rk seaLevel, rk dynamicThickness,
    rk calvingThickness, rk oceanDensity, rk iceDensity, int nCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    int a = 0;
    const rk h = thickness[i];
    if (bedTopography[i] >= seaLevel) {
        if (h > dynamicThickness) a = 1;
    } else {
        const rk flotation = M(D(oceanDensity, iceDensity), S(seaLevel, bedTopography[i]));
        if (h < flotation) {
            if (h > calvingThickness) a = 1;
        } else if (h > dynamicThickness) {
            a = 1;
        }
    }
    active[i] = a;
}

// thickness_calving: the inactive margin and the calving of open-ocean ice
// with no active neighbour
extern "C" __global__ void li_calving_threshold_v842(
    rk *thickness, rk *calvingThicknessOut, const int *active, const rk *bedTopography,
    const int *nEdgesOnCell, const int *cellsOnCell, rk seaLevel, rk fraction, int nCells, int maxEdges)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    int inactiveMargin = 0;
    if (active[i] == 0) {
        for (int j = 0; j < nEdgesOnCell[i]; ++j) {
            if (active[cellsOnCell[i * maxEdges + j] - 1] == 1) {
                inactiveMargin = 1;
                break;
            }
        }
    }
    const int ocean = (bedTopography[i] < seaLevel && active[i] == 0) ? 1 : 0;
    rk calved = LI_C(0.0);
    if (ocean == 1 && inactiveMargin == 0 && thickness[i] > LI_C(0.0)) calved = M(thickness[i], fraction);
    calvingThicknessOut[i] = calved;
    thickness[i] = S(thickness[i], calved);
}

// floating_calving (topographic == 0) and topographic_calving (1)
extern "C" __global__ void li_calving_floating_v842(
    rk *thickness, rk *calvingThicknessOut, const int *cellMask, const rk *bedTopography, rk topographyLimit,
    rk fraction, int topographic, int nCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    rk calved = LI_C(0.0);
    if (li_is(cellMask[i], LI_FLOATING) && (!topographic || bedTopography[i] < topographyLimit))
        calved = M(thickness[i], fraction);
    calvingThicknessOut[i] = calved;
    thickness[i] = S(thickness[i], calved);
}

// calculate_calving_front_mask, one thread per cell
extern "C" __global__ void li_calving_front_mask_v842(
    int *front, const int *cellMask, const rk *bedTopography, const int *nEdgesOnCell, const int *cellsOnCell,
    rk seaLevel, int nCells, int maxEdges)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    int f = 0;
    const int mask = cellMask[i];
    if (li_is(mask, LI_FLOATING) && li_is(mask, LI_DYNAMIC_MARGIN)) {
        bool oceanNeighbour = false;
        for (int j = 0; j < nEdgesOnCell[i]; ++j) {
            const int jc = cellsOnCell[i * maxEdges + j] - 1;
            const int jm = cellMask[jc];
            if (li_is(jm, LI_FLOATING) && !li_is(jm, LI_DYNAMIC_ICE)) {
                for (int k = 0; k < nEdgesOnCell[jc]; ++k) {
                    const int kc = cellsOnCell[jc * maxEdges + k] - 1;
                    if (!li_is(cellMask[kc], LI_ICE) && bedTopography[kc] < seaLevel) oceanNeighbour = true;
                }
            }
            if (!li_is(jm, LI_ICE) && bedTopography[jc] < seaLevel) oceanNeighbour = true;
        }
        if (oceanNeighbour) f = 1;
    }
    front[i] = f;
}

// eigencalving: the potential calving velocity of floating ice and, at the
// front, the volume rate it requires
extern "C" __global__ void li_eigencalving_rate_v842(
    rk *calvingVelocity, rk *requiredCalvingVolumeRate, const rk *eigencalvingParameter, const rk *eMax,
    const rk *eMin, const int *cellMask, const int *front, const rk *thickness, const rk *bedTopography,
    const rk *dvEdge, const int *nEdgesOnCell, const int *cellsOnCell, const int *edgesOnCell, rk seaLevel,
    int nCells, int maxEdges)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    const rk floating = li_is(cellMask[i], LI_FLOATING) ? LI_C(1.0) : LI_C(0.0);
    const rk v = M(M(M(eigencalvingParameter[i], li_max(LI_C(0.0), eMax[i])), li_max(LI_C(0.0), eMin[i])), floating);
    calvingVelocity[i] = v;
    rk rate = LI_C(0.0);
    if (front[i] == 1) {
        rk length = LI_C(0.0), height = LI_C(0.0);
        for (int j = 0; j < nEdgesOnCell[i]; ++j) {
            const int jc = cellsOnCell[i * maxEdges + j] - 1;
            const int jm = cellMask[jc];
            if ((li_is(jm, LI_FLOATING) && !li_is(jm, LI_DYNAMIC_ICE)) ||
                (!li_is(jm, LI_ICE) && bedTopography[jc] < seaLevel))
                length = A(length, dvEdge[edgesOnCell[i * maxEdges + j] - 1]);
            height = li_max(height, thickness[jc]);
        }
        rate = M(M(v, length), height);
    }
    requiredCalvingVolumeRate[i] = rate;
}

// distribute_calving_flux, in the Fortran's cell order (one thread: each
// front cell's removal changes the volumes its successors see). As written,
// the inward-neighbour test asks li_mask_is_dynamic_margin of the neighbour's
// index rather than its mask, and an uncalved remainder is recorded as zero.
extern "C" __global__ void li_distribute_calving_flux_v842(
    rk *calvingThickness, rk *uncalvedVolume, rk *cellVolume, const rk *requiredCalvingVolumeRate,
    const int *front, const int *cellMask, const rk *thickness, const rk *areaCell, const int *nEdgesOnCell,
    const int *cellsOnCell, rk deltat, int nCells, int maxEdges)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int i = 0; i < nCells; ++i) {
        calvingThickness[i] = LI_C(0.0);
        uncalvedVolume[i] = LI_C(0.0);
        cellVolume[i] = M(areaCell[i], thickness[i]);
    }
    for (int i = 0; i < nCells; ++i) {
        if (front[i] != 1) continue;
        rk left = M(requiredCalvingVolumeRate[i], deltat);
        const int *nb = cellsOnCell + i * maxEdges;
        for (int j = 0; j < nEdgesOnCell[i]; ++j) {
            const int jc = nb[j] - 1;
            if (li_is(cellMask[jc], LI_FLOATING) && !li_is(cellMask[jc], LI_DYNAMIC_ICE)) {
                const rk here = li_min(left, cellVolume[jc]);
                calvingThickness[jc] = A(calvingThickness[jc], D(here, areaCell[jc]));
                cellVolume[jc] = S(cellVolume[jc], here);
                left = S(left, here);
            }
        }
        if (left > LI_C(0.0)) {
            const rk here = li_min(left, cellVolume[i]);
            calvingThickness[i] = A(calvingThickness[i], D(here, areaCell[i]));
            cellVolume[i] = S(cellVolume[i], here);
            left = S(left, here);
        }
        if (left > LI_C(0.0)) {
            int inward = 0;
            for (int j = 0; j < nEdgesOnCell[i]; ++j) {
                const int jn = nb[j];
                if (li_is(cellMask[jn - 1], LI_FLOATING) && li_is(cellMask[jn - 1], LI_DYNAMIC_ICE) &&
                    !li_is(jn, LI_DYNAMIC_MARGIN))
                    ++inward;
            }
            for (int j = 0; j < nEdgesOnCell[i]; ++j) {
                const int jn = nb[j], jc = jn - 1;
                if (li_is(cellMask[jc], LI_FLOATING) && li_is(cellMask[jc], LI_DYNAMIC_ICE) &&
                    !li_is(jn, LI_DYNAMIC_MARGIN)) {
                    const rk here = li_min(D(left, (rk)inward), cellVolume[jc]);
                    calvingThickness[jc] = A(calvingThickness[jc], D(here, areaCell[jc]));
                    cellVolume[jc] = S(cellVolume[jc], here);
                    left = S(left, here);
                }
            }
        }
        if (left > LI_C(0.0)) uncalvedVolume[i] = LI_C(0.0);
    }
}

// thickness(:) = thickness(:) - calvingThickness(:)
extern "C" __global__ void li_subtract_calving_v842(rk *thickness, const rk *calvingThickness, int nCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    thickness[i] = S(thickness[i], calvingThickness[i]);
}

// eigencalving: thin ice at the front calves whole
extern "C" __global__ void li_calve_thin_front_v842(
    rk *thickness, rk *calvingThickness, const int *front, rk calvingThicknessLimit, int nCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    if (front[i] == 1 && thickness[i] < calvingThicknessLimit) {
        calvingThickness[i] = A(calvingThickness[i], thickness[i]);
        thickness[i] = LI_C(0.0);
    }
}

// eigencalving: floating ice without a dynamic neighbour calves whole
extern "C" __global__ void li_calve_isolated_floating_v842(
    rk *thickness, rk *calvingThickness, const int *cellMask, const int *nEdgesOnCell, const int *cellsOnCell,
    int nCells, int maxEdges)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    if (!li_is(cellMask[i], LI_FLOATING)) return;
    bool dynamicNeighbour = false;
    for (int j = 0; j < nEdgesOnCell[i]; ++j)
        if (li_is(cellMask[cellsOnCell[i * maxEdges + j] - 1], LI_DYNAMIC_ICE)) dynamicNeighbour = true;
    if (!dynamicNeighbour) {
        calvingThickness[i] = A(calvingThickness[i], thickness[i]);
        thickness[i] = LI_C(0.0);
    }
}

// li_restore_calving_front's cell loop over nCellsSolve: marine cells of the
// initial extent thinner than restoreMin are restored to it (restoreFlag 1,
// their columns re-initialised by the driver), ice outside it calves
extern "C" __global__ void li_restore_calving_front_v842(
    rk *thickness, rk *calvingThickness, rk *restoreThickness, int *restored, const int *cellMask,
    const rk *bedTopography, rk seaLevel, rk restoreMin, int nCells, int nCellsSolve)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    calvingThickness[i] = LI_C(0.0);
    restoreThickness[i] = LI_C(0.0);
    restored[i] = 0;
    if (i >= nCellsSolve || !(bedTopography[i] < seaLevel)) return;
    const bool initial = li_is(cellMask[i], LI_INITIAL_ICE_EXTENT);
    if (initial && thickness[i] < restoreMin) {
        restoreThickness[i] = S(restoreMin, thickness[i]);
        thickness[i] = restoreMin;
        restored[i] = 1;
    } else if (!initial && thickness[i] > LI_C(0.0)) {
        calvingThickness[i] = thickness[i];
        thickness[i] = LI_C(0.0);
    }
}
