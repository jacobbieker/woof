// Declarations that let libclang parse woof's CUDA translation units as
// C++17 for woof.backend.translate.  Never emitted into Metal: the parse
// pulls it in with -include, so its offsets are outside the main file.
//
// The CUDA keywords themselves are macros given on the command line
// (translate.PARSE_DEFINES), so the preprocessed text the rewriter edits
// carries them as annotate() attributes it can find.
#pragma once
typedef unsigned long size_t;
typedef long ptrdiff_t;
typedef signed char int8_t; typedef unsigned char uint8_t;
typedef short int16_t; typedef unsigned short uint16_t;
typedef int int32_t; typedef unsigned int uint32_t;
typedef long long int64_t; typedef unsigned long long uint64_t;
extern const float INFINITY;
extern const float NAN;
extern const float FLT_MAX; extern const float FLT_MIN; extern const float FLT_EPSILON;
extern const double DBL_MAX; extern const double DBL_MIN; extern const double DBL_EPSILON;
extern const int INT_MAX; extern const int INT_MIN; extern const unsigned int UINT_MAX;

struct __woof_uint3 { unsigned int x, y, z; };
struct __woof_dim3 { unsigned int x, y, z; };
extern const __woof_uint3 threadIdx;
extern const __woof_uint3 blockIdx;
extern const __woof_dim3 blockDim;
extern const __woof_dim3 gridDim;
extern const int warpSize;

void __syncthreads();
void __threadfence();
void __threadfence_block();

// binary32 rounded operations and conversions
float __fadd_rn(float, float); float __fsub_rn(float, float);
float __fmul_rn(float, float); float __fdiv_rn(float, float);
float __frcp_rn(float); float __fsqrt_rn(float); float __fmaf_rn(float, float, float);
float __fadd_rz(float, float); float __fmul_rz(float, float);
float __int_as_float(int); int __float_as_int(float);
float __uint_as_float(unsigned int); unsigned int __float_as_uint(float);
int __float2int_rn(float); int __float2int_rz(float); int __float2int_rd(float); int __float2int_ru(float);
unsigned int __float2uint_rn(float); unsigned int __float2uint_rz(float);
float __int2float_rn(int); float __uint2float_rn(unsigned int);
float __saturatef(float);
float __expf(float); float __logf(float); float __powf(float, float);
float __sinf(float); float __cosf(float); float __tanf(float);
float __exp10f(float); float __log2f(float); float __log10f(float);
float __fdividef(float, float);

// binary64 rounded operations and conversions
double __dadd_rn(double, double); double __dsub_rn(double, double);
double __dmul_rn(double, double); double __ddiv_rn(double, double);
double __drcp_rn(double); double __dsqrt_rn(double); double __fma_rn(double, double, double);
double __dadd_rz(double, double); double __dmul_rz(double, double);
long long __double_as_longlong(double); double __longlong_as_double(long long);
int __double2hiint(double); int __double2loint(double); double __hiloint2double(int, int);
float __double2float_rn(double); float __double2float_rz(double);
int __double2int_rn(double); int __double2int_rz(double); int __double2int_rd(double); int __double2int_ru(double);
long long __double2ll_rn(double); long long __double2ll_rz(double);
double __int2double_rn(int); double __ll2double_rn(long long);

// integer intrinsics
int __clz(int); int __clzll(long long); int __popc(unsigned int); int __popcll(unsigned long long);
int __ffs(int); int __ffsll(long long);
unsigned int __umulhi(unsigned int, unsigned int); unsigned long long __umul64hi(unsigned long long, unsigned long long);
int __mulhi(int, int); long long __mul64hi(long long, long long);
unsigned int __brev(unsigned int); unsigned long long __brevll(unsigned long long);
int min(int, int); int max(int, int);
unsigned int min(unsigned int, unsigned int); unsigned int max(unsigned int, unsigned int);
long long min(long long, long long); long long max(long long, long long);
unsigned long long min(unsigned long long, unsigned long long); unsigned long long max(unsigned long long, unsigned long long);
float min(float, float); float max(float, float);
double min(double, double); double max(double, double);

// atomics (int, unsigned, float, double, unsigned long long)
int atomicAdd(int*, int); unsigned int atomicAdd(unsigned int*, unsigned int);
float atomicAdd(float*, float); double atomicAdd(double*, double);
unsigned long long atomicAdd(unsigned long long*, unsigned long long);
int atomicSub(int*, int); unsigned int atomicSub(unsigned int*, unsigned int);
int atomicMax(int*, int); unsigned int atomicMax(unsigned int*, unsigned int);
int atomicMin(int*, int); unsigned int atomicMin(unsigned int*, unsigned int);
int atomicExch(int*, int); unsigned int atomicExch(unsigned int*, unsigned int); float atomicExch(float*, float);
int atomicCAS(int*, int, int); unsigned int atomicCAS(unsigned int*, unsigned int, unsigned int);
unsigned long long atomicCAS(unsigned long long*, unsigned long long, unsigned long long);
int atomicOr(int*, int); unsigned int atomicOr(unsigned int*, unsigned int);
int atomicAnd(int*, int); unsigned int atomicAnd(unsigned int*, unsigned int);

// warp shuffles
float __shfl_down_sync(unsigned int, float, unsigned int, int = 32);
double __shfl_down_sync(unsigned int, double, unsigned int, int = 32);
int __shfl_down_sync(unsigned int, int, unsigned int, int = 32);
float __shfl_xor_sync(unsigned int, float, int, int = 32);
double __shfl_xor_sync(unsigned int, double, int, int = 32);
float __shfl_sync(unsigned int, float, int, int = 32);

// libm: the C names CUDA provides, binary32 and binary64
#define WOOF_F1(n) float n##f(float); double n(double); float n(float);
#define WOOF_F2(n) float n##f(float, float); double n(double, double); float n(float, float);
WOOF_F1(sqrt) WOOF_F1(cbrt) WOOF_F1(exp) WOOF_F1(exp2) WOOF_F1(exp10) WOOF_F1(expm1)
WOOF_F1(log) WOOF_F1(log2) WOOF_F1(log10) WOOF_F1(log1p) WOOF_F1(sin) WOOF_F1(cos)
WOOF_F1(tan) WOOF_F1(asin) WOOF_F1(acos) WOOF_F1(atan) WOOF_F1(sinh) WOOF_F1(cosh)
WOOF_F1(tanh) WOOF_F1(asinh) WOOF_F1(acosh) WOOF_F1(atanh) WOOF_F1(fabs) WOOF_F1(floor)
WOOF_F1(ceil) WOOF_F1(trunc) WOOF_F1(round) WOOF_F1(rint) WOOF_F1(nearbyint) WOOF_F1(erf)
WOOF_F1(erfc) WOOF_F1(tgamma) WOOF_F1(lgamma) WOOF_F1(rsqrt)
WOOF_F2(pow) WOOF_F2(atan2) WOOF_F2(fmin) WOOF_F2(fmax) WOOF_F2(fmod) WOOF_F2(copysign)
WOOF_F2(hypot) WOOF_F2(fdim) WOOF_F2(remainder)
#undef WOOF_F1
#undef WOOF_F2
float fmaf(float, float, float); double fma(double, double, double); float fma(float, float, float);
float ldexpf(float, int); double ldexp(double, int); float ldexp(float, int);
float frexpf(float, int*); double frexp(double, int*);
float modff(float, float*); double modf(double, double*);
void sincosf(float, float*, float*); void sincos(double, double*, double*);
long lroundf(float); long lround(double); long long llroundf(float); long long llround(double);
long lrintf(float); long lrint(double);
int isnan(float); int isnan(double); int isinf(float); int isinf(double);
int isfinite(float); int isfinite(double); int signbit(float); int signbit(double);
int abs(int); long long llabs(long long); long labs(long);
int printf(const char*, ...);
void* memcpy(void*, const void*, size_t);
void* memset(void*, int, size_t);

// CUDA vector types (layouts as CUDA's).
#define WOOF_VEC2(T, N) struct N##2 { T x, y; }; N##2 make_##N##2(T, T);
#define WOOF_VEC3(T, N) struct N##3 { T x, y, z; }; N##3 make_##N##3(T, T, T);
#define WOOF_VEC4(T, N) struct __attribute__((aligned(16))) N##4 { T x, y, z, w; }; N##4 make_##N##4(T, T, T, T);
WOOF_VEC2(float, float) WOOF_VEC3(float, float) WOOF_VEC4(float, float)
WOOF_VEC2(int, int) WOOF_VEC3(int, int) WOOF_VEC4(int, int)
WOOF_VEC2(unsigned int, uint) WOOF_VEC3(unsigned int, uint) WOOF_VEC4(unsigned int, uint)
WOOF_VEC2(double, double) WOOF_VEC3(double, double) WOOF_VEC4(double, double)
#undef WOOF_VEC2
#undef WOOF_VEC3
#undef WOOF_VEC4
float nanf(const char*); double nan(const char*);
void __trap();
void __syncwarp(unsigned int = 0xffffffffu);
unsigned int __ballot_sync(unsigned int, int);
int __any_sync(unsigned int, int); int __all_sync(unsigned int, int);
float __shfl_up_sync(unsigned int, float, unsigned int, int = 32);
int __shfl_up_sync(unsigned int, int, unsigned int, int = 32);
unsigned int __shfl_down_sync(unsigned int, unsigned int, unsigned int, int = 32);
unsigned int __shfl_sync(unsigned int, unsigned int, int, int = 32);
int __shfl_sync(unsigned int, int, int, int = 32);
double __shfl_sync(unsigned int, double, int, int = 32);
int atomicXor(int*, int); unsigned int atomicXor(unsigned int*, unsigned int);
float __double2float_ru(double); float __double2float_rd(double);
long long __float2ll_rz(float); long long __float2ll_rn(float);
int __syncthreads_or(int); int __syncthreads_and(int); int __syncthreads_count(int);
unsigned long long atomicOr(unsigned long long*, unsigned long long);
unsigned long long atomicAnd(unsigned long long*, unsigned long long);
unsigned long long atomicXor(unsigned long long*, unsigned long long);
unsigned long long atomicMin(unsigned long long*, unsigned long long);
unsigned long long atomicMax(unsigned long long*, unsigned long long);
template <class A, class B> auto max(A a, B b) -> decltype(a + b);
template <class A, class B> auto min(A a, B b) -> decltype(a + b);
// Inline PTX without .ftz, rewritten (rewrite_ptx_asm): IEEE binary32 with subnormals.
float __woof_ieee_add(float, float); float __woof_ieee_sub(float, float);
float __woof_ieee_mul(float, float); float __woof_ieee_div(float, float);
float __woof_ieee_min(float, float); float __woof_ieee_max(float, float);
