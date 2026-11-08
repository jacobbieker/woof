// OPT-IN ONLY.  CUDA's powf as CuPy's float32 power runs it, for the MLX
// facade's cupy.power under GPUWM_WRF_EXACT=1 when WOOF_STRICT_POWF=cuda
// is set in a source checkout (woof/backend/facade/_ops.py).  The default
// is woof/backend/translate/strict_powf.cuh, WOOF's own correctly rounded
// powf.  This file is not package data: no wheel (recast-woof, metal-woof)
// and no default build contains it.  See NOTICE, "CUDA powf transcription".
//
// PROVENANCE.  This is a transcription of NVIDIA's code: the PTX that
// NVRTC (CUDA 13.4) emits for powf from its libdevice, under the strict
// options woof/wrf_exact.py gives CuPy's kernels (--ftz=false,
// --fmad=false).  Its algorithm structure and its constants are NVIDIA's.
// Every operation is the PTX's, rounded to nearest, with subnormals kept
// (float is f32s in this process).  Never compiled for CUDA.
//
// One step is not transcribed: the PTX's rcp.approx.ftz.f32 (the GPU's
// approximate reciprocal; 13 % of its words differ from the correctly
// rounded one) is the correctly rounded reciprocal here.  Measured against
// strict CuPy on an RTX A6000: with the A6000's own reciprocal words
// substituted, this function gave CuPy's word on every input measured
// (every binary32 x in [0.5, 1) at y = 287/1004.5, 8388608 random pairs,
// 234 special-value pairs); with the correctly rounded reciprocal, 131 of
// the 8388608 Exner-range words and 915 of the random pairs differ, each
// by 1 ULP.
#ifndef WOOF_CUDA_POWF
#define WOOF_CUDA_POWF

__device__ __forceinline__ float woof_cuda_powf(float x, float y)
{
    const float infinity = __int_as_float(0x7F800000);
    const float half_y = __fmul_rn(y, 0.5f);
    const float odd = fabsf(__fsub_rn(y, __fadd_rn(truncf(half_y), truncf(half_y))));
    const float ax = fabsf(x);

    // log2|x| = e + log2(m), m in [sqrt(1/2), sqrt(2)), as hi + lo
    const bool tiny = ax < __int_as_float(0x00800000);
    const int bits = __float_as_int(tiny ? __fmul_rn(ax, __int_as_float(0x4B800000)) : ax);
    const int top = (bits - 0x3F3504F3) & (int)0xFF800000u;
    const float m = __int_as_float(bits - top);
    const float e = __fmaf_rn(__int2float_rn(top), __int_as_float(0x34000000),
                              tiny ? -24.0f : 0.0f);
    const float mm1 = __fsub_rn(m, 1.0f);
    const float r = __frcp_rn(__fadd_rn(m, 1.0f));  // CUDA: rcp.approx.ftz.f32
    const float twice = __fadd_rn(mm1, mm1);
    const float q = __fmul_rn(twice, r);
    const float q2 = __fmul_rn(q, q);
    const float d = __fsub_rn(mm1, q);
    const float ql = __fmul_rn(r, __fmaf_rn(-q, mm1, __fadd_rn(d, d)));
    const float p = __fmul_rn(__fmaf_rn(__fmaf_rn(__fmaf_rn(
        __int_as_float(0x3A2C32E4), q2, __int_as_float(0x3B52E7DB)), q2,
        __int_as_float(0x3C93BB73)), q2, __int_as_float(0x3DF6384F)), q2);
    const float log2e = __int_as_float(0x3FB8AA3B);
    const float hi = __fmaf_rn(q, log2e, e);
    float lo = __fmaf_rn(q, log2e, __fsub_rn(e, hi));
    lo = __fmaf_rn(ql, log2e, lo);
    lo = __fmaf_rn(q, __int_as_float(0x32A55E34), lo);
    lo = __fmaf_rn(__fmul_rn(p, 3.0f), ql, lo);
    lo = __fmaf_rn(p, q, lo);
    const float lg = __fadd_rn(hi, lo);
    const float lg_lo = __fsub_rn(lo, __fsub_rn(lg, hi));

    // y * log2|x| = n + f, then 2^f * 2^n
    const float t = __fmul_rn(lg, y);
    const float t_lo = __fmaf_rn(lg_lo, y, __fmaf_rn(lg, y, -t));
    const float n = rintf(t);
    const float f = __fadd_rn(t_lo, __fsub_rn(t, n));
    float s = __fmaf_rn(__int_as_float(0x391FCB8E), f, __int_as_float(0x3AAF85ED));
    s = __fmaf_rn(s, f, __int_as_float(0x3C1D9856));
    s = __fmaf_rn(s, f, __int_as_float(0x3D6357BB));
    s = __fmaf_rn(s, f, __int_as_float(0x3E75FDEC));
    s = __fmaf_rn(s, f, __int_as_float(0x3F317218));
    s = __fmaf_rn(s, f, 1.0f);
    const int bias = n > 0.0f ? 0 : (int)0x83000000u;
    const float scaled = __fmul_rn(s, __int_as_float(bias + 0x7F000000));
    float result = __fmul_rn(
        scaled, __int_as_float((int)(((unsigned)(int)n << 23) - (unsigned)bias)));
    if (fabsf(t) > 152.0f) result = t < 0.0f ? 0.0f : infinity;

    // the PTX's special cases, in its order
    if (x == 1.0f || y == 0.0f) return 1.0f;
    // x + y: a NaN operand gives CUDA's canonical NaN word
    if (!(ax <= infinity) || !(fabsf(y) <= infinity)) return __int_as_float(0x7FFFFFFF);
    if (x == 0.0f || ax == infinity) {
        int w = __float_as_int(__fadd_rn(x, x));
        if (y < 0.0f) w ^= 0x7F800000;
        return __int_as_float(odd == 1.0f ? w : (w & 0x7FFFFFFF));
    }
    if (x == -1.0f && fabsf(y) == infinity) return 1.0f;
    if (!(x < 0.0f)) return result;
    if (floorf(y) != y) return __int_as_float(0x7FFFFFFF);
    return odd == 1.0f ? -result : result;
}

#endif
