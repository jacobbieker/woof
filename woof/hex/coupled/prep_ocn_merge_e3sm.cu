// E3SM v3.2.0 coupler: the ocean's import merge (prep_ocn_merge).
//
// A modified derivative of E3SM v3.2.0 (E3SM-Project/E3SM, tag v3.2.0,
// 648b167b7f95267a159a9fdee25faf604c2bfc4f), transcribed from
//   driver-mct/main/prep_ocn_mod.F90 (prep_ocn_merge)
// with MCT (externals/mct at 82b0071e69d14330b75d23b0bc68543ebea9aadc)'s
// m_AttrVect zero and shared-index copy, which prep_ocn_merge calls.
// driver-mct (cpl7): University Corporation for Atmospheric Research and
// UChicago Argonne, LLC; E3SM: Copyright 2024, UChicago Argonne, LLC; MCT:
// Copyright 2018, UChicago Argonne, LLC. BSD-3-Clause, the terms in
// licenses/LICENSE-E3SM-BSD-3-Clause.txt. This is not the version available
// from the E3SM project.
//
// real(r8) only (the coupler's kind). Every operation is an explicit
// round-to-nearest intrinsic in the Fortran's evaluation order (gfortran -O0
// -ffp-contract=off). One thread per point of the ocean's grid; a point's
// work is the Fortran's, in its order: the zeroing, the shared-name copies
// (a2x, i2x, r2x, w2x when coupled two ways, xao, g2x), the explicit
// shortwave, precipitation and runoff, then the generic merge field by
// field. The Fortran runs each stage over every point before the next; no
// stage reads another point, so the per-point order gives the same words.
//
// Layout: an attribute vector's rAttr(nRattr, lsize) is av[n*nRattr + k]
// (k, n 0-based). The merge plan (which field goes where) is worked out on
// the host from the field lists, as the Fortran's first call does.

typedef double r8;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))

// the explicit fields' indices (0-based; -1 when the plan leaves one out)
enum {
    X_A_SWVDR, X_A_SWVDF, X_A_SWNDR, X_A_SWNDF, X_I_SWPEN,
    X_X_AVSDR, X_X_ANIDR, X_X_AVSDF, X_X_ANIDF,
    X_A_SNOWC, X_A_SNOWL, X_A_RAINC, X_A_RAINL,
    X_R_ROFL, X_R_ROFI, X_R_FLOOD, X_G_ROFL, X_G_ROFI,
    X_O_SWNET, X_O_SNOW, X_O_RAIN, X_O_PREC, X_O_ROFL, X_O_ROFI,
    X_O_AFRAC, X_O_AFRACR, X_O_SWNET_AFRACR,
    X_F_IFRAC, X_F_OFRAC, X_F_IFRAD, X_F_OFRAD,
    X_COUNT
};
#define NUTRIENTS 11

// the fractions, normalised by their sum where it is nonzero
__device__ __forceinline__ void pom_fractions(const r8 *fr, int nf, int n, int ki, int ko, r8 *ifrac, r8 *afrac)
{
    r8 i = fr[(size_t)n * nf + ki];
    r8 a = fr[(size_t)n * nf + ko];
    const r8 frac_sum = A(i, a);
    if (frac_sum != 0.0) {
        i = D(i, frac_sum);
        a = D(a, frac_sum);
    }
    *ifrac = i;
    *afrac = a;
}

extern "C" __global__ void prep_ocn_merge_e3sm(
    r8 *__restrict__ x2o, int no,
    const r8 *__restrict__ a2x, int na, const r8 *__restrict__ i2x, int ni,
    const r8 *__restrict__ r2x, int nr, const r8 *__restrict__ w2x, int nw,
    const r8 *__restrict__ g2x, int ng, const r8 *__restrict__ xao, int nx,
    const r8 *__restrict__ fr, int nf,
    // the shared-name copies, in the Fortran's order: source (0 a2x, 1 i2x,
    // 2 r2x, 3 w2x, 4 xao, 5 g2x), its field, the x2o field
    const int *__restrict__ copySource, const int *__restrict__ copyFrom, const int *__restrict__ copyTo, int nCopies,
    const int *__restrict__ ix,
    // the nutrients' r2x and x2o fields, when rof2ocn_nutrients
    const int *__restrict__ nutrientFrom, const int *__restrict__ nutrientTo, int nutrients, int perCat,
    // the generic merge per x2o field: aindx, iindx, xindx (0-based, -1 none)
    // and their merge flags (1 adds, 0 replaces)
    const int *__restrict__ aindx, const int *__restrict__ iindx, const int *__restrict__ xindx,
    const int *__restrict__ amerge, const int *__restrict__ imerge, const int *__restrict__ xmerge,
    r8 flux_epbalfact, int lsize)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= lsize) return;
    r8 *o = x2o + (size_t)n * no;
    const r8 *a = a2x + (size_t)n * na;
    const r8 *ice = i2x + (size_t)n * ni;
    const r8 *r = r2x + (size_t)n * nr;
    const r8 *g = g2x + (size_t)n * ng;
    const r8 *x = xao + (size_t)n * nx;

    // mct_aVect_zero(x2o_o)
    for (int k = 0; k < no; ++k) o[k] = 0.0;
    // mct_aVect_copy(aVin=..., aVout=x2o_o, sharedIndices=...)
    for (int c = 0; c < nCopies; ++c) {
        const int s = copySource[c];
        const r8 *src = s == 0 ? a : s == 1 ? ice : s == 2 ? r : s == 3 ? w2x + (size_t)n * nw : s == 4 ? x : g;
        o[copyTo[c]] = src[copyFrom[c]];
    }

    r8 ifrac, afrac, ifracr, afracr;
    pom_fractions(fr, nf, n, ix[X_F_IFRAC], ix[X_F_OFRAC], &ifrac, &afrac);
    pom_fractions(fr, nf, n, ix[X_F_IFRAD], ix[X_F_OFRAD], &ifracr, &afracr);

    const r8 avsdr = x[ix[X_X_AVSDR]];
    const r8 anidr = x[ix[X_X_ANIDR]];
    const r8 avsdf = x[ix[X_X_AVSDF]];
    const r8 anidf = x[ix[X_X_ANIDF]];
    const r8 fswabsv = A(M(a[ix[X_A_SWVDR]], S(1.0, avsdr)), M(a[ix[X_A_SWVDF]], S(1.0, avsdf)));
    const r8 fswabsi = A(M(a[ix[X_A_SWNDR]], S(1.0, anidr)), M(a[ix[X_A_SWNDF]], S(1.0, anidf)));
    o[ix[X_O_SWNET]] = A(M(A(fswabsv, fswabsi), afracr), M(ice[ix[X_I_SWPEN]], ifrac));
    if (perCat) {
        o[ix[X_O_AFRAC]] = afrac;
        o[ix[X_O_AFRACR]] = afracr;
        o[ix[X_O_SWNET_AFRACR]] = M(A(fswabsv, fswabsi), afracr);
    }
    o[ix[X_O_SNOW]] = A(M(a[ix[X_A_SNOWC]], afrac), M(a[ix[X_A_SNOWL]], afrac));
    o[ix[X_O_RAIN]] = A(M(a[ix[X_A_RAINC]], afrac), M(a[ix[X_A_RAINL]], afrac));
    o[ix[X_O_SNOW]] = M(o[ix[X_O_SNOW]], flux_epbalfact);
    o[ix[X_O_RAIN]] = M(o[ix[X_O_RAIN]], flux_epbalfact);
    o[ix[X_O_PREC]] = A(o[ix[X_O_RAIN]], o[ix[X_O_SNOW]]);
    o[ix[X_O_ROFL]] = M(A(A(r[ix[X_R_ROFL]], r[ix[X_R_FLOOD]]), g[ix[X_G_ROFL]]), flux_epbalfact);
    o[ix[X_O_ROFI]] = M(A(r[ix[X_R_ROFI]], g[ix[X_G_ROFI]]), flux_epbalfact);
    if (nutrients)
        for (int k = 0; k < NUTRIENTS; ++k) o[nutrientTo[k]] = r[nutrientFrom[k]];

    // the generic merge: ice, then atmosphere, then the atmosphere/ocean fluxes
    for (int ko = 0; ko < no; ++ko) {
        pom_fractions(fr, nf, n, ix[X_F_IFRAC], ix[X_F_OFRAC], &ifrac, &afrac);
        if (iindx[ko] >= 0) {
            if (imerge[ko]) o[ko] = A(o[ko], M(ice[iindx[ko]], ifrac));
            else o[ko] = M(ice[iindx[ko]], ifrac);
        }
        if (aindx[ko] >= 0) {
            if (amerge[ko]) o[ko] = A(o[ko], M(a[aindx[ko]], afrac));
            else o[ko] = M(a[aindx[ko]], afrac);
        }
        if (xindx[ko] >= 0) {
            if (xmerge[ko]) o[ko] = A(o[ko], M(x[xindx[ko]], afrac));
            else o[ko] = M(x[xindx[ko]], afrac);
        }
    }
}
