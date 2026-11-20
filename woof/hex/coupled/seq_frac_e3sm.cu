// E3SM v3.2.0's coupler surface fractions update, seq_frac_set
// (driver-mct/main/seq_frac_mct.F90), with the maps it makes through
// seq_map_map (driver-mct/main/seq_map_mod.F90) and MCT's sparse matrix
// multiply, and seq_frac_check's bounds and sum tests. real(r8); every
// operation an explicit round-to-nearest intrinsic, in the Fortran's order.
//
// Attribute vectors are MCT's rAttr(nflds, lsize): field k of point n at
// av[n * nflds + k].

typedef double r8;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))

// fractions_i: ifrac = Si_ifrac * frac (mct_aVect_copy, then the product;
// without Si_ifrac in i2x the copy copies nothing, kif < 0, and ifrac is the
// vector's own); ofrac = frac - ifrac
extern "C" __global__ void seq_frac_ice(
    r8* __restrict__ fi, int nfi, int ki, int ko, const r8* __restrict__ i2x, int ni2x, int kif,
    const r8* __restrict__ dom, int ndom, int kf, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const r8 frac = dom[(size_t)i * ndom + kf];
    const r8 ifrac = M(kif >= 0 ? i2x[(size_t)i * ni2x + kif] : fi[(size_t)i * nfi + ki], frac);
    fi[(size_t)i * nfi + ki] = ifrac;
    fi[(size_t)i * nfi + ko] = S(frac, ifrac);
}

// seq_map_map with fldlist 'ofrac:ifrac' through a copy or a one-task
// rearrangement: the two fields, word for word
extern "C" __global__ void seq_frac_copy(
    r8* __restrict__ dst, int nd, int dko, int dki, const r8* __restrict__ src, int ns, int sko, int ski, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[(size_t)i * nd + dko] = src[(size_t)i * ns + sko];
    dst[(size_t)i * nd + dki] = src[(size_t)i * ns + ski];
}

// seq_map_avNormArr with norm=.false. over mct_sMat_avMult (XandY, one
// task): the matrix's elements ordered by row then column (MCT's stable
// sorts; repeated elements keep the file's order), each row's sum from zero
// as y = y + w * x, then MCT's summing rearrangement into the zeroed output,
// 0 + y; rows with no element are 0
extern "C" __global__ void seq_frac_matrix(
    r8* __restrict__ dst, int nd, int dko, int dki, const r8* __restrict__ src, int ns, int sko, int ski,
    const int* __restrict__ row_start, const int* __restrict__ col, const r8* __restrict__ w, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    r8 yo = 0.0, yi = 0.0;
    for (int e = row_start[i]; e < row_start[i + 1]; ++e) {
        const r8* x = src + (size_t)col[e] * ns;
        yo = A(yo, M(w[e], x[sko]));
        yi = A(yi, M(w[e], x[ski]));
    }
    dst[(size_t)i * nd + dko] = A(0.0, yo);
    dst[(size_t)i * nd + dki] = A(0.0, yi);
}

// seq_frac_check: a point is in error where a checked fraction (afrac,
// lfrac, ofrac, ifrac, gfrac, rfrac, wfrac, zfrac, lfrin; k[] < 0 when the
// vector lacks it) is below 0 - eps_fracval or above 1 + eps_fracval, or,
// with lfrac, ofrac and ifrac all present, |1 - (ofrac + lfrac + ifrac)|
// exceeds eps_fracsum. The Fortran tests the vector's minimum and maximum;
// with no NaN that is the same as testing every point.
extern "C" __global__ void seq_frac_check(
    const r8* __restrict__ av, int nf, const int* __restrict__ k, int nk, int kl, int ko, int ki,
    r8 eps_fracval, r8 eps_fracsum, int* __restrict__ bad, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const r8* a = av + (size_t)i * nf;
    const r8 lo = S(0.0, eps_fracval), hi = A(1.0, eps_fracval);
    int e = 0;
    for (int j = 0; j < nk; ++j)
        if (k[j] >= 0 && (a[k[j]] < lo || a[k[j]] > hi)) e = 1;
    if (kl >= 0 && ko >= 0 && ki >= 0) {
        const r8 sum = A(A(a[ko], a[kl]), a[ki]);
        if (fabs(S(1.0, sum)) > eps_fracsum) e = 1;
    }
    bad[i] = e;
}
