// E3SM v3.2.0 coupler: the sea ice's import merge (prep_ice_merge).
//
// A modified derivative of E3SM v3.2.0 (E3SM-Project/E3SM, tag v3.2.0,
// 648b167b7f95267a159a9fdee25faf604c2bfc4f), transcribed from
//   driver-mct/main/prep_ice_mod.F90 (prep_ice_merge)
// with MCT (externals/mct at 82b0071e69d14330b75d23b0bc68543ebea9aadc)'s
// m_AttrVect shared-index copy, which prep_ice_merge calls.
// driver-mct (cpl7): University Corporation for Atmospheric Research and
// UChicago Argonne, LLC; E3SM: Copyright 2024, UChicago Argonne, LLC; MCT:
// Copyright 2018, UChicago Argonne, LLC. BSD-3-Clause, the terms in
// licenses/LICENSE-E3SM-BSD-3-Clause.txt. This is not the version available
// from the E3SM project.
//
// real(r8) only (the coupler's kind). Every operation is an explicit
// round-to-nearest intrinsic in the Fortran's order (gfortran -O0
// -ffp-contract=off). One thread per point of the ice's grid; a point's work
// is the Fortran's, in its order: the shared-name copies (o2x, a2x, g2x, then
// w2x when coupled two ways; x2i is not zeroed first), then rain, snow and
// the ice runoff summed and, in separate statements, scaled by
// flux_epbalfact. No stage reads another point.
//
// Layout: an attribute vector's rAttr(nRattr, lsize) is av[n*nRattr + k]
// (k, n 0-based). The copies are worked out on the host from the field
// lists, as the Fortran's first call does.

typedef double r8;
#define A(a,b) __dadd_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))

// the explicit fields' indices, in their vectors (0-based)
enum { X_A_RAINC, X_A_RAINL, X_A_SNOWC, X_A_SNOWL, X_G_ROFI, X_R_ROFI, X_I_RAIN, X_I_SNOW, X_I_ROFI, X_COUNT };

extern "C" __global__ void prep_ice_merge_e3sm(
    r8 *__restrict__ x2i, int ni,
    const r8 *__restrict__ o2x, int no, const r8 *__restrict__ a2x, int na,
    const r8 *__restrict__ g2x, int ng, const r8 *__restrict__ w2x, int nw,
    const r8 *__restrict__ r2x, int nr,
    // the shared-name copies, in the Fortran's order: source (0 o2x, 1 a2x,
    // 2 g2x, 3 w2x), its field, the x2i field
    const int *__restrict__ copySource, const int *__restrict__ copyFrom, const int *__restrict__ copyTo, int nCopies,
    const int *__restrict__ ix, r8 flux_epbalfact, int lsize)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= lsize) return;
    r8 *x = x2i + (size_t)n * ni;
    for (int c = 0; c < nCopies; ++c) {
        const r8 *src;
        int width;
        switch (copySource[c]) {
            case 0: src = o2x; width = no; break;
            case 1: src = a2x; width = na; break;
            case 2: src = g2x; width = ng; break;
            default: src = w2x; width = nw; break;
        }
        x[copyTo[c]] = src[(size_t)n * width + copyFrom[c]];
    }
    const r8 *a = a2x + (size_t)n * na;
    x[ix[X_I_RAIN]] = A(a[ix[X_A_RAINC]], a[ix[X_A_RAINL]]);
    x[ix[X_I_SNOW]] = A(a[ix[X_A_SNOWC]], a[ix[X_A_SNOWL]]);
    x[ix[X_I_ROFI]] = A(g2x[(size_t)n * ng + ix[X_G_ROFI]], r2x[(size_t)n * nr + ix[X_R_ROFI]]);
    x[ix[X_I_RAIN]] = M(x[ix[X_I_RAIN]], flux_epbalfact);
    x[ix[X_I_SNOW]] = M(x[ix[X_I_SNOW]], flux_epbalfact);
    x[ix[X_I_ROFI]] = M(x[ix[X_I_ROFI]], flux_epbalfact);
}
