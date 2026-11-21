// E3SM v3.2.0's ocean albedos and net shortwave on the coupler's flux grid:
// seq_flux_ocnalb_mct (driver-mct/main/seq_flux_mct.F90:795-973) without its
// orbital branch. real(r8); every operation an explicit round-to-nearest
// intrinsic in the Fortran's order (gfortran -O0 -ffp-contract=off).
//
// A modified derivative of E3SM v3.2.0 (E3SM-Project/E3SM, tag v3.2.0,
// 648b167b7f95267a159a9fdee25faf604c2bfc4f). driver-mct (cpl7): University
// Corporation for Atmospheric Research and UChicago Argonne, LLC; E3SM:
// Copyright 2024, UChicago Argonne, LLC. BSD-3-Clause, the terms in
// licenses/LICENSE-E3SM-BSD-3-Clause.txt. This is not the version available
// from the E3SM project.
//
// Attribute vectors are MCT's rAttr(nflds, lsize): field k of point n at
// av[n * nflds + k]. One thread per point; no point reads another.

typedef double r8;
#define A(a,b) __dadd_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))

enum { X_AVSDR, X_ANIDR, X_AVSDF, X_ANIDF, X_SWDN, X_SWUP, NX };
enum { A_SWNDR, A_SWNDF, A_SWVDR, A_SWVDF, NA };
enum { F_IFRAC, F_IFRAD, F_OFRAC, F_OFRAD, NF };

// flux_albav: the namelist's albedos everywhere, then ifrad = ifrac and
// ofrad = ofrac. Otherwise the net shortwave from the albedos xao holds
// (0 where anidr is 1, the dark side), and nothing else.
extern "C" __global__ void ocnalb(
    r8* __restrict__ xao, int nx, const int* __restrict__ kx, const r8* __restrict__ a2x, int na,
    const int* __restrict__ ka, r8* __restrict__ fr, int nf, const int* __restrict__ kf, int albav,
    r8 albdif, r8 albdir, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    r8* x = xao + (size_t)i * nx;
    if (albav) {
        const r8 anidr = albdir, avsdr = albdir, anidf = albdif, avsdf = albdif;
        x[kx[X_AVSDR]] = avsdr;
        x[kx[X_ANIDR]] = anidr;
        x[kx[X_AVSDF]] = avsdf;
        x[kx[X_ANIDF]] = anidf;
        r8* f = fr + (size_t)i * nf;
        f[kf[F_IFRAD]] = f[kf[F_IFRAC]];
        f[kf[F_OFRAD]] = f[kf[F_OFRAC]];
        return;
    }
    const r8 avsdr = x[kx[X_AVSDR]], anidr = x[kx[X_ANIDR]], avsdf = x[kx[X_AVSDF]], anidf = x[kx[X_ANIDF]];
    const r8* a = a2x + (size_t)i * na;
    r8 swupc = A(A(A(M(a[ka[A_SWNDR]], -anidr), M(a[ka[A_SWNDF]], -anidf)), M(a[ka[A_SWVDR]], -avsdr)),
                 M(a[ka[A_SWVDF]], -avsdf));
    r8 swdnc = A(A(A(a[ka[A_SWNDR]], a[ka[A_SWNDF]]), a[ka[A_SWVDR]]), a[ka[A_SWVDF]]);
    if (anidr == 1.0) {
        swupc = 0.0;
        swdnc = 0.0;
    }
    x[kx[X_SWDN]] = swdnc;
    x[kx[X_SWUP]] = swupc;
}
