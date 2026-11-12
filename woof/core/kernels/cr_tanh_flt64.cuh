// ======================================================================
// THIRD-PARTY NOTICE. Original CUDA adaptation is distributed under
// Apache-2.0. The MIT grant below covers the transcribed upstream part.
// CORE-MATH binary64 tanh (src/binary64/tanh/tanh.c, revision
// dbd377a9e30beba33237c40a5c4e09c1ba18a8fe, sha256
// e89441340424ea7d137269ce4626b9c1426c37fade09c0853c0ffcca606fb111,
// kept verbatim at tools/oracle_libm/upstream/tanh.c).
// cr_tanh is CORE-MATH's correctly rounded binary64 tanh, rounding to
// nearest, ties to even. glibc 2.44 carries CORE-MATH's tanh as its own
// binary64 tanh (its x86-64 __tanh_fma and __tanh_sse2 use CORE-MATH's
// __sinhcosh tables and as_exp_accurate; neither calls expm1), so these are
// the words the oracle host's gfortran tanh returns. No glibc or fdlibm
// source is involved.
// PROOF: tools/oracle_libm/grade_tanh.py: equal to the oracle host's glibc
// tanh on 2^33 CPU and 2^28 GPU arguments, and to an exact decimal tanh on
// a subsample (tools/oracle_libm/README.md).
// ======================================================================
// Correctly rounded hyperbolic tangent function for binary64 values.
//
// Copyright (c) 2023-2026 Alexei Sibidanov, Cyprien Peignier, Paul Zimmermann
//
// Alexei Sibidanov designed the original algorithm, while Cyprien Peignier and
// Paul Zimmermann extended the fma formula for |x0| <= 0x1.d12ed0af1a27fp-27,
// and improved the minimax polynomial for x0 <= |x| < 0.25.
//
// This file is part of the CORE-MATH project
// (https://core-math.gitlabpages.inria.fr/).
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// ADAPTATION (the only changes from upstream): rounding to nearest only;
// errno is not set; every arithmetic operation is an explicit
// round-to-nearest intrinsic (__dadd_rn, __dsub_rn, __dmul_rn, __ddiv_rn,
// __fma_rn for upstream's __builtin_fma) in upstream's C evaluation order,
// so no compiler option can contract or reassociate it; fabs and copysign
// are sign-bit operations; the x86-64 _mm_and_pd branch is upstream's
// portable integer mask; the static tables are __device__ constants at
// namespace scope.
#pragma once
#if !defined(__CUDACC_RTC__)
#include <stdint.h>
#else
typedef unsigned long long uint64_t;
typedef long long int64_t;
#endif

namespace g64_tanh {

__device__ __forceinline__ double asdouble(uint64_t u) { return __longlong_as_double((long long)u); }
__device__ __forceinline__ uint64_t asuint64(double f) { return (uint64_t)__double_as_longlong(f); }
__device__ __forceinline__ double dabs(double x) { return asdouble(asuint64(x) & 0x7fffffffffffffffULL); }
__device__ __forceinline__ double dcopysign(double a, double b)
{
    return asdouble((asuint64(a) & 0x7fffffffffffffffULL) | (asuint64(b) & 0x8000000000000000ULL));
}

__device__ __forceinline__ double fasttwosum(double x, double y, double *e)
{
    double s = __dadd_rn(x, y), z = __dsub_rn(s, x);
    *e = __dsub_rn(y, z);
    return s;
}

__device__ __forceinline__ double fasttwosub(double x, double y, double *e)
{
    double s = __dsub_rn(x, y), z = __dsub_rn(x, s);
    *e = __dsub_rn(z, y);
    return s;
}

__device__ __forceinline__ double muldd_acc(double xh, double xl, double ch, double cl, double *l)
{
    double plh = __dmul_rn(xl, ch), phl = __dmul_rn(xh, cl), phh = __dmul_rn(xh, ch),
           phh_rest = __fma_rn(xh, ch, -phh);
    phh_rest = __dadd_rn(phh_rest, __dadd_rn(phl, plh));
    return fasttwosum(phh, phh_rest, l);
}

__device__ __forceinline__ double mulddd_acc(double xh, double xl, double c, double *l)
{
    double pl = __dmul_rn(xl, c), ph = __dmul_rn(xh, c), ph_rest = __fma_rn(xh, c, -ph);
    ph_rest = __dadd_rn(ph_rest, pl);
    return fasttwosum(ph, ph_rest, l);
}

// at input, l is the approximation of the upper part of the polynomial
// (evaluated with double arithmetic only)
__device__ __forceinline__ double polydd(double xh, double xl, int n, const double c[][2], double *l)
{
    int i = n - 1;
    double ch, cl, t;
    ch = fasttwosum(c[i][0], *l, &t);
    cl = __dadd_rn(t, c[i][1]);
    while (--i >= 0) {
        ch = muldd_acc(xh, xl, ch, cl, &cl);
        double th, tl;
        th = fasttwosum(c[i][0], ch, &tl);
        ch = th;
        cl = __dadd_rn(__dadd_rn(cl, c[i][1]), tl);
    }
    *l = cl;
    return ch;
}

__device__ const double exp_acc_ch[][2] = {
    {0x1p+0, 0x1.6c16bd194535dp-94}, {0x1p-1, -0x1.8259d904fd34fp-93},
    {0x1.5555555555555p-3, 0x1.53e93e9f26e62p-57}};

__device__ __noinline__ double as_exp_accurate(double x, double t, double th, double tl, double *l)
{
    const double l2h = 0x1.62e42ffp-13, l2l = 0x1.718432a1b0e26p-47, l2ll = 0x1.9ff0342542fc3p-102;
    double dx = __dsub_rn(x, __dmul_rn(l2h, t)), dxl = __dmul_rn(l2l, t),
           dxll = __dadd_rn(__dmul_rn(l2ll, t), __fma_rn(l2l, t, -dxl));
    double dxh = __dadd_rn(dx, dxl);
    dxl = __dadd_rn(__dadd_rn(__dsub_rn(dx, dxh), dxl), dxll);
    double fl = __dmul_rn(dxh, __dadd_rn(0x1.5555555555555p-5,
                __dmul_rn(dxh, __dadd_rn(0x1.11111113e93e9p-7, __dmul_rn(dxh, 0x1.6c16c169400a7p-10)))));
    double fh = polydd(dxh, dxl, 3, exp_acc_ch, &fl);
    fh = muldd_acc(dxh, dxl, fh, fl, &fl);
    fh = muldd_acc(th, tl, fh, fl, &fl);
    double zh = __dadd_rn(th, fh), zl = __dadd_rn(__dsub_rn(th, zh), fh);
    double uh = __dadd_rn(zh, tl), ul = __dadd_rn(__dadd_rn(__dsub_rn(zh, uh), tl), zl);
    double vh = __dadd_rn(uh, fl), vl = __dadd_rn(__dadd_rn(__dsub_rn(uh, vh), fl), ul);
    *l = vl;
    return vh;
}

__device__ const double tanh_db[][3] = {
    {0x1.ac343b179fec4p-3, 0x1.a612499c53078p-3,  0x1p-57},
    {0x1.00764a988bf73p-2, 0x1.f676484c0703bp-3, -0x1p-104},
    {0x1.17d1e8a63711fp-2, 0x1.110e96a6c2d96p-2, -0x1p-105},
    {0x1.291c601a05276p-2, 0x1.210b7d0c03743p-2,  0x1p-56},
    {0x1.36f33d51c264dp-2, 0x1.2dbb7b1c91363p-2, -0x1p-106},
    {0x1.43eaea23649c3p-2, 0x1.39877ed028641p-2, -0x1p-55},
    {0x1.d88d7550b2826p-2, 0x1.b9a3637366afdp-2,  0x1p-56},
    {0x1.e611aa58ab608p-2, 0x1.c493dc899e4a6p-2, -0x1p-55},
    {0x1.01efe7ac8c15dp-1, 0x1.dc3fe1b524821p-2, -0x1p-104},
    {0x1.1005ec0bccabbp-1, 0x1.f20b1c8557dedp-2, -0x1p-55},
    {0x1.33dfeb0fa4bfep-1, 0x1.1372f9ee76e99p-1,  0x1p-55},
    {0x1.49f24ac5cac35p-1, 0x1.22c495ff06104p-1, -0x1p-104},
};

__device__ __noinline__ double as_tanh_database(double x, double f)
{
    int a = 0, b = (int)(sizeof(tanh_db) / sizeof(tanh_db[0])) - 1, m = (a + b) / 2;
    double ax = dabs(x);
    while (a <= b) {
        if (tanh_db[m][0] < ax)
            a = m + 1;
        else if (tanh_db[m][0] == ax) {
            f = __dadd_rn(__dmul_rn(dcopysign(1.0, x), tanh_db[m][1]),
                          __dmul_rn(dcopysign(1.0, x), tanh_db[m][2]));
            break;
        } else
            b = m - 1;
        m = (a + b) / 2;
    }
    return f;
}

__device__ const double tanh_zero_ch[][2] = {
        {-0x1.5555555555555p-2, -0x1.5555555554cc4p-56},
        {0x1.1111111111111p-3, 0x1.111110f8c0178p-59},
        {-0x1.ba1ba1ba1ba1cp-5, 0x1.7917c1d676ff5p-59},
        {0x1.664f4882c10fap-6, -0x1.9d5cb27c0af28p-63},
        {-0x1.226e355e6c23cp-7, -0x1.c9674586913f3p-61},
        {0x1.d6d3d0e157db3p-9, -0x1.71376fa06ce94p-65},
        {-0x1.7da36452b5e46p-10, -0x1.aba8d51bd9cp-65},
        {0x1.3558247faa32dp-11, -0x1.e0cfb423aedfdp-65},
        {-0x1.f57d76ea30928p-13, -0x1.c30601213cae9p-67},
};
__device__ const double tanh_zero_cl[] = {
        0x1.967e0a63ca836p-14,  -0x1.497b99d2a77d1p-15, 0x1.0ae346258cbdep-16,
        -0x1.aade68fb2f076p-18, 0x1.22e609bf8671fp-19,
};

__device__ __noinline__ double as_tanh_zero(double x)  // |x|<0.25
{
    const double *cl = tanh_zero_cl;
    double x2 = __dmul_rn(x, x), x2l = __fma_rn(x, x, -x2);
    double y2 = __dmul_rn(x2, __dadd_rn(cl[0], __dmul_rn(x2, __dadd_rn(cl[1],
                __dmul_rn(x2, __dadd_rn(cl[2], __dmul_rn(x2, __dadd_rn(cl[3], __dmul_rn(x2, cl[4])))))))));
    double y1 = polydd(x2, x2l, 9, tanh_zero_ch, &y2);
    y1 = mulddd_acc(y1, y2, x, &y2);
    y1 = muldd_acc(y1, y2, x2, x2l, &y2);
    double y0 = fasttwosum(x, y1, &y1);
    y1 = fasttwosum(y1, y2, &y2);
    uint64_t tu = asuint64(y1);
    if (!(tu & (~0ULL >> 12))) {
        uint64_t wu = asuint64(y2);
        if ((wu ^ tu) >> 63)
            tu--;
        else
            tu++;
        y1 = asdouble(tu);
        if (dabs(x) == 0x1.ac343b179fec4p-3)
            return as_tanh_database(x, __dadd_rn(y0, y1));
    }
    return __dadd_rn(y0, y1);
}

__device__ const double tanh_t0[][2] = {
    {0x0p+0, 0x1p+0}, {-0x1.19083535b085ep-56, 0x1.02c9a3e778061p+0},
      {0x1.d73e2a475b466p-55, 0x1.059b0d3158574p+0}, {0x1.186be4bb285p-57, 0x1.0874518759bc8p+0},
      {0x1.8a62e4adc610ap-54, 0x1.0b5586cf9890fp+0}, {0x1.03a1727c57b52p-59, 0x1.0e3ec32d3d1a2p+0},
      {-0x1.6c51039449b3ap-54, 0x1.11301d0125b51p+0}, {-0x1.32fbf9af1369ep-54, 0x1.1429aaea92dep+0},
      {-0x1.19041b9d78a76p-55, 0x1.172b83c7d517bp+0}, {0x1.e5b4c7b4968e4p-55, 0x1.1a35beb6fcb75p+0},
      {0x1.e016e00a2643cp-54, 0x1.1d4873168b9aap+0}, {0x1.dc775814a8494p-55, 0x1.2063b88628cd6p+0},
      {0x1.9b07eb6c70572p-54, 0x1.2387a6e756238p+0}, {0x1.2bd339940e9dap-55, 0x1.26b4565e27cddp+0},
      {0x1.612e8afad1256p-55, 0x1.29e9df51fdee1p+0}, {0x1.0024754db41d4p-54, 0x1.2d285a6e4030bp+0},
      {0x1.6f46ad23182e4p-55, 0x1.306fe0a31b715p+0}, {0x1.32721843659a6p-54, 0x1.33c08b26416ffp+0},
      {-0x1.63aeabf42eae2p-54, 0x1.371a7373aa9cbp+0}, {-0x1.5e436d661f5e2p-56, 0x1.3a7db34e59ff7p+0},
      {0x1.ada0911f09ebcp-55, 0x1.3dea64c123422p+0}, {-0x1.ef3691c309278p-58, 0x1.4160a21f72e2ap+0},
      {0x1.89b7a04ef80dp-59, 0x1.44e086061892dp+0}, {0x1.3c1a3b69062fp-56, 0x1.486a2b5c13cdp+0},
      {0x1.d4397afec42e2p-56, 0x1.4bfdad5362a27p+0}, {-0x1.4b309d25957e4p-54, 0x1.4f9b2769d2ca7p+0},
      {-0x1.07abe1db13cacp-55, 0x1.5342b569d4f82p+0}, {0x1.9bb2c011d93acp-54, 0x1.56f4736b527dap+0},
      {0x1.6324c054647acp-54, 0x1.5ab07dd485429p+0}, {0x1.ba6f93080e65ep-54, 0x1.5e76f15ad2148p+0},
      {-0x1.383c17e40b496p-54, 0x1.6247eb03a5585p+0}, {-0x1.bb60987591c34p-54, 0x1.6623882552225p+0},
      {-0x1.bdd3413b26456p-54, 0x1.6a09e667f3bcdp+0}, {-0x1.bbe3a683c88aap-57, 0x1.6dfb23c651a2fp+0},
      {-0x1.16e4786887a9ap-55, 0x1.71f75e8ec5f74p+0}, {-0x1.0245957316dd4p-54, 0x1.75feb564267c9p+0},
      {-0x1.41577ee04993p-55, 0x1.7a11473eb0187p+0}, {0x1.05d02ba15797ep-56, 0x1.7e2f336cf4e62p+0},
      {-0x1.d4c1dd41532d8p-54, 0x1.82589994cce13p+0}, {-0x1.fc6f89bd4f6bap-54, 0x1.868d99b4492edp+0},
      {0x1.6e9f156864b26p-54, 0x1.8ace5422aa0dbp+0}, {0x1.5cc13a2e3976cp-55, 0x1.8f1ae99157736p+0},
      {-0x1.75fc781b57ebcp-57, 0x1.93737b0cdc5e5p+0}, {-0x1.d185b7c1b85dp-54, 0x1.97d829fde4e5p+0},
      {0x1.c7c46b071f2bep-56, 0x1.9c49182a3f09p+0}, {-0x1.359495d1cd532p-54, 0x1.a0c667b5de565p+0},
      {-0x1.d2f6edb8d41e2p-54, 0x1.a5503b23e255dp+0}, {0x1.0fac90ef7fd32p-54, 0x1.a9e6b5579fdbfp+0},
      {0x1.7a1cd345dcc82p-54, 0x1.ae89f995ad3adp+0}, {-0x1.2805e3084d708p-57, 0x1.b33a2b84f15fbp+0},
      {-0x1.5584f7e54ac3ap-56, 0x1.b7f76f2fb5e47p+0}, {0x1.23dd07a2d9e84p-55, 0x1.bcc1e904bc1d2p+0},
      {0x1.11065895048dep-55, 0x1.c199bdd85529cp+0}, {0x1.2884dff483cacp-54, 0x1.c67f12e57d14bp+0},
      {0x1.503cbd1e949dcp-56, 0x1.cb720dcef9069p+0}, {-0x1.cbc3743797a9cp-54, 0x1.d072d4a07897cp+0},
      {0x1.2ed02d75b3706p-55, 0x1.d5818dcfba487p+0}, {0x1.c2300696db532p-54, 0x1.da9e603db3285p+0},
      {-0x1.1a5cd4f184b5cp-54, 0x1.dfc97337b9b5fp+0}, {0x1.39e8980a9cc9p-55, 0x1.e502ee78b3ff6p+0},
      {-0x1.e9c23179c2894p-54, 0x1.ea4afa2a490dap+0}, {0x1.dc7f486a4b6bp-54, 0x1.efa1bee615a27p+0},
      {0x1.9d3e12dd8a18ap-54, 0x1.f50765b6e454p+0}, {0x1.74853f3a5931ep-55, 0x1.fa7c1819e90d8p+0}
};
__device__ const double tanh_t1[][2] = {
    {0x0p+0, 0x1p+0}, {0x1.ae8e38c59c72ap-54, 0x1.000b175effdc7p+0},
    {-0x1.7b5d0d58ea8f4p-58, 0x1.00162f3904052p+0}, {0x1.4115cb6b16a8ep-54, 0x1.0021478e11ce6p+0},
    {-0x1.d7c96f201bb2ep-55, 0x1.002c605e2e8cfp+0}, {0x1.84711d4c35eap-54, 0x1.003779a95f959p+0},
    {-0x1.0484245243778p-55, 0x1.0042936faa3d8p+0}, {-0x1.4b237da2025fap-54, 0x1.004dadb113dap+0},
    {-0x1.5e00e62d6b30ep-56, 0x1.0058c86da1c0ap+0}, {0x1.a1d6cedbb948p-54, 0x1.0063e3a559473p+0},
    {-0x1.4acf197a00142p-54, 0x1.006eff583fc3dp+0}, {-0x1.eaf2ea42391a6p-57, 0x1.007a1b865a8cap+0},
    {0x1.da93f90835f76p-56, 0x1.0085382faef83p+0}, {-0x1.6a79084ab093cp-55, 0x1.00905554425d4p+0},
    {0x1.86364f8fbe8f8p-54, 0x1.009b72f41a12bp+0}, {-0x1.82e8e14e3110ep-55, 0x1.00a6910f3b6fdp+0},
    {-0x1.4f6b2a7609f72p-55, 0x1.00b1afa5abcbfp+0}, {-0x1.e1a258ea8f71ap-56, 0x1.00bcceb7707ecp+0},
    {0x1.4362ca5bc26f2p-56, 0x1.00c7ee448ee02p+0}, {0x1.095a56c919d02p-54, 0x1.00d30e4d0c483p+0},
    {-0x1.406ac4e81a646p-57, 0x1.00de2ed0ee0f5p+0}, {0x1.b5a6902767e08p-54, 0x1.00e94fd0398ep+0},
    {-0x1.91b206085932p-54, 0x1.00f4714af41d3p+0}, {0x1.427068ab22306p-55, 0x1.00ff93412315cp+0},
    {0x1.c1d0660524e08p-54, 0x1.010ab5b2cbd11p+0}, {-0x1.e7bdfb3204be8p-54, 0x1.0115d89ff3a8bp+0},
    {0x1.843aa8b9cbbc6p-55, 0x1.0120fc089ff63p+0}, {-0x1.34104ee7edae8p-56, 0x1.012c1fecd613bp+0},
    {-0x1.2b6aeb6176892p-56, 0x1.0137444c9b5b5p+0}, {0x1.a8cd33b8a1bb2p-56, 0x1.01426927f5278p+0},
    {0x1.2edc08e5da99ap-56, 0x1.014d8e7ee8d2fp+0}, {0x1.57ba2dc7e0c72p-55, 0x1.0158b4517bb88p+0},
    {0x1.b61299ab8cdb8p-54, 0x1.0163da9fb3335p+0}, {-0x1.90565902c5f44p-54, 0x1.016f0169949edp+0},
    {0x1.70fc41c5c2d54p-55, 0x1.017a28af25567p+0}, {0x1.4b9a6e145d76cp-54, 0x1.018550706ab62p+0},
    {-0x1.008eff5142bfap-56, 0x1.019078ad6a19fp+0}, {-0x1.77669f033c7dep-54, 0x1.019ba16628de2p+0},
    {-0x1.09bb78eeead0ap-54, 0x1.01a6ca9aac5f3p+0}, {0x1.371231477ece6p-54, 0x1.01b1f44af9f9ep+0},
    {0x1.5e7626621eb5ap-56, 0x1.01bd1e77170b4p+0}, {-0x1.bc72b100828a4p-54, 0x1.01c8491f08f08p+0},
    {-0x1.ce39cbbab8bbep-57, 0x1.01d37442d507p+0}, {0x1.16996709da2e2p-55, 0x1.01de9fe280ac8p+0},
    {-0x1.c11f5239bf536p-55, 0x1.01e9cbfe113efp+0}, {0x1.e1d4eb5edc6b4p-55, 0x1.01f4f8958c1c6p+0},
    {-0x1.afb99946ee3fp-54, 0x1.020025a8f6a35p+0}, {-0x1.8f06d8a148a32p-54, 0x1.020b533856324p+0},
    {-0x1.2bf310fc54eb6p-55, 0x1.02168143b0281p+0}, {-0x1.c95a035eb4176p-54, 0x1.0221afcb09e3ep+0},
    {-0x1.491793e46834cp-54, 0x1.022cdece68c4fp+0}, {-0x1.3e8d0d9c4909p-56, 0x1.02380e4dd22adp+0},
    {-0x1.314aa16278aa4p-54, 0x1.02433e494b755p+0}, {0x1.48daf888e965p-55, 0x1.024e6ec0da046p+0},
    {0x1.56dc8046821f4p-55, 0x1.02599fb483385p+0}, {0x1.45b42356b9d46p-54, 0x1.0264d1244c719p+0},
    {-0x1.082ef51b61d7ep-56, 0x1.027003103b10ep+0}, {0x1.2106ed0920a34p-56, 0x1.027b357854772p+0},
    {-0x1.fd4cf26ea5d0ep-54, 0x1.0286685c9e059p+0}, {-0x1.09f8775e78084p-54, 0x1.02919bbd1d1d8p+0},
    {0x1.64cbba902ca28p-58, 0x1.029ccf99d720ap+0}, {0x1.4383ef231d206p-54, 0x1.02a803f2d170dp+0},
    {0x1.4a47a505b3a46p-54, 0x1.02b338c811703p+0}, {0x1.e47120223468p-54, 0x1.02be6e199c811p+0},
};

__device__ __forceinline__ double cr_tanh(double x)
{
    // tanh(x) is a minimax polynomial for |x|<0.25; elsewhere
    // tanh(|x|) = 1 - 2*exp(-2*|x|)/(1 + exp(-2*|x|)), with a less precise
    // exponential for large |x|>3.683.
    double ax = dabs(x);
    uint64_t aix = asuint64(ax);
    // for |x| >= 0x1.30fc1931f09cap+4, tanh(x) rounds to +1 or -1 to nearest
    if (aix >= 0x40330fc1931f09caULL) {
        if (aix > 0x7ff0000000000000ULL) return __dadd_rn(x, x);  // nan
        double f = dcopysign(1.0, x);
        if (aix == 0x7ff0000000000000ULL) return f;
        double df = dcopysign(0x1p-55, x);
        return __dsub_rn(f, df);
    }
    const double s = -0x1.71547652b82fep+13;
    double v0 = __fma_rn(ax, s, 0x1.8000004p+25);
    uint64_t jt = asuint64(v0);
    double t = __dsub_rn(asdouble(jt & 0xfffffffff8000000ULL), 0x1.8p25);
    int64_t i1 = (jt >> 27) & 0x3f, i0 = (jt >> 33) & 0x3f, ie = (int64_t)(jt << 13) >> 52;
    const double sp = asdouble((uint64_t)(1023 + ie) << 52);
    const double ch0 = 0x1p+1, ch1 = 0x1p+1, ch2 = 0x1.55555557e54ffp+0, ch3 = 0x1.55555553a12f4p-1;
    double t0h = tanh_t0[i0][1], t1h = tanh_t1[i1][1], th = __dmul_rn(t0h, t1h), tl;
    if (aix < 0x400d76c8b4395810ULL) {  // |x| ~< 3.683
        if (aix < 0x3fd0000000000000ULL) {  // |x| < 0x1p-2
            if (aix <= 0x3e4d12ed0af1a27fULL) {  // |x| <= 0x1.d12ed0af1a27fp-27
                if (!aix) return x;
                return __fma_rn(x, -0x1p-55, x);
            }
            const double c0 = -0x1.5555555555555p-2, c1 = 0x1.1111111110f33p-3,
                         c2 = -0x1.ba1ba1b9b8ea6p-5, c3 = 0x1.664f4838e0a43p-6,
                         c4 = -0x1.226e17d1bc09bp-7, c5 = 0x1.d6c64dfba2565p-9,
                         c6 = -0x1.7bdd094d327afp-10, c7 = 0x1.1535ad0c31d0ep-11;
            double x2 = __dmul_rn(x, x), x3 = __dmul_rn(x2, x), x4 = __dmul_rn(x2, x2),
                   x8 = __dmul_rn(x4, x4);
            double p1 = __dadd_rn(__dadd_rn(c4, __dmul_rn(x2, c5)), __dmul_rn(x4, __dadd_rn(c6, __dmul_rn(x2, c7))));
            double p0 = __dadd_rn(__dadd_rn(c0, __dmul_rn(x2, c1)), __dmul_rn(x4, __dadd_rn(c2, __dmul_rn(x2, c3))));
            p0 = __dadd_rn(p0, __dmul_rn(x8, p1));
            p0 = __dmul_rn(p0, x3);
            double rl, rh = fasttwosum(x, p0, &rl);
            double e = __dmul_rn(x3, 0x1.c0p-52), lb = __dadd_rn(rh, __dsub_rn(rl, e)),
                   ub = __dadd_rn(rh, __dadd_rn(rl, e));
            if (lb == ub) return lb;
            return as_tanh_zero(x);
        }

        double t0l = tanh_t0[i0][0], t1l = tanh_t1[i1][0];
        tl = __dadd_rn(__dadd_rn(__dmul_rn(t0h, t1l), __dmul_rn(t1h, t0l)), __fma_rn(t0h, t1h, -th));
        th = __dmul_rn(th, sp);
        tl = __dmul_rn(tl, sp);
        const double l2h = -0x1.62e42ffp-14, l2l = -0x1.718432a1b0e26p-48;
        double dx = __dsub_rn(__dsub_rn(__dmul_rn(l2h, t), ax), __dmul_rn(l2l, t)), dx2 = __dmul_rn(dx, dx);
        double p = __dmul_rn(dx, __dadd_rn(__dadd_rn(ch0, __dmul_rn(dx, ch1)),
                                           __dmul_rn(dx2, __dadd_rn(ch2, __dmul_rn(dx, ch3)))));
        double rh = th, rl = __dadd_rn(tl, __dmul_rn(rh, p));
        rh = fasttwosum(rh, rl, &rl);

        double ph = rh, pl = rl;
        double qh = rh, ql = rl, qd;
        qh = fasttwosum(1.0, qh, &qd);
        ql = __dadd_rn(ql, qd);

        double rqh = __ddiv_rn(1.0, qh),
               rql = __dmul_rn(__dadd_rn(__dmul_rn(ql, rqh), __fma_rn(rqh, qh, -1.0)), -rqh);
        ph = muldd_acc(ph, pl, rqh, rql, &pl);

        double e = __dmul_rn(rh, 0x1.0dp-62);
        rh = fasttwosub(0.5, ph, &rl);
        rl = __dsub_rn(rl, pl);
        rh = __dmul_rn(rh, dcopysign(2.0, x));
        rl = __dmul_rn(rl, dcopysign(2.0, x));
        double lb = __dadd_rn(rh, __dsub_rn(rl, e)), ub = __dadd_rn(rh, __dadd_rn(rl, e));
        if (lb == ub) return lb;
    } else {  // 3.683 ~< x < 0x1.30fc1931f09cap+4
        const double l2 = -0x1.62e42fefa39efp-14;
        double dx = __fma_rn(l2, t, -ax), dx2 = __dmul_rn(dx, dx);
        double p = __dmul_rn(dx, __dadd_rn(__dadd_rn(ch0, __dmul_rn(dx, ch1)),
                                           __dmul_rn(dx2, __dadd_rn(ch2, __dmul_rn(dx, ch3)))));
        double rh = __dmul_rn(th, sp);
        rh = __dadd_rn(rh, __dmul_rn(__dadd_rn(p, __dmul_rn(2 * 0x1.3p-55, ax)), rh));
        double e = __dmul_rn(rh, 0x1.1p-49);
        rh = __ddiv_rn(__dmul_rn(2.0, rh), __dadd_rn(1.0, rh));
        double one = dcopysign(1.0, x);
        rh = dcopysign(rh, x);
        double lb = __dsub_rn(one, __dadd_rn(rh, e)), ub = __dsub_rn(one, __dsub_rn(rh, e));
        if (lb == ub) return lb;

        double t0l = tanh_t0[i0][0], t1l = tanh_t1[i1][0];
        tl = __dadd_rn(__dadd_rn(__dmul_rn(t0h, t1l), __dmul_rn(t1h, t0l)), __fma_rn(t0h, t1h, -th));
        th = __dmul_rn(th, sp);
        tl = __dmul_rn(tl, sp);
    }
    double rl, rh = as_exp_accurate(__dmul_rn(-2.0, ax), t, th, tl, &rl);
    double qd, qh = fasttwosum(1.0, rh, &qd), ql = __dadd_rn(rl, qd);
    qh = fasttwosum(qh, ql, &ql);
    double rqh = __ddiv_rn(1.0, qh),
           rql = __dmul_rn(__dadd_rn(__dmul_rn(ql, rqh), __fma_rn(rqh, qh, -1.0)), -rqh);
    double pl, ph = muldd_acc(rh, rl, rqh, rql, &pl);
    rh = fasttwosub(0.5, ph, &rl);
    rl = __dsub_rn(rl, pl);
    rh = fasttwosum(rh, rl, &rl);
    double res = __dadd_rn(__dmul_rn(dcopysign(2.0, x), rh), __dmul_rn(dcopysign(2.0, x), rl));
    uint64_t lu = asuint64(rl);
    if (((lu + 32) & (~0ULL >> 12)) < 65) return as_tanh_database(x, res);
    return res;
}

}  // namespace g64_tanh
