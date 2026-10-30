// The oracle's float32 libm: WOOF's own transcription of glibc 2.39's
// expf, logf and powf (woof/core/kernels/glibc_flt32.cuh), compiled for
// the host and linked into the oracle executable ahead of the system
// library.  The Fortran objects are byte-unmodified; only the library they
// call is pinned, so the oracle's words do not depend on which libm the
// build host carries (macOS libsystem_m, glibc 2.39, glibc 2.44 ...).
// The GPU kernel calls the SAME source (gfk_exp/gfk_log/gfk_pow), so the
// gate grades the port's arithmetic, not two libraries' rounding.
//
// libm-report.txt records the undefined float32 libm symbols of every WRF
// object; build.sh refuses a build that calls any float32 transcendental
// this file does not define.
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#define __device__
#define __forceinline__ inline
static inline uint32_t __float_as_uint(float x){uint32_t u;std::memcpy(&u,&x,4);return u;}
static inline float __uint_as_float(uint32_t u){float x;std::memcpy(&x,&u,4);return x;}
static inline float __int_as_float(int32_t u){float x;std::memcpy(&x,&u,4);return x;}
static inline long long __double_as_longlong(double x){long long u;std::memcpy(&u,&x,8);return u;}
static inline double __longlong_as_double(long long u){double x;std::memcpy(&x,&u,8);return x;}
static inline float __double2float_rn(double x){return (float)x;}
static inline double __dadd_rn(double a,double b){return a+b;}
static inline double __dsub_rn(double a,double b){return a-b;}
static inline double __dmul_rn(double a,double b){return a*b;}
static inline double __ddiv_rn(double a,double b){return a/b;}
static inline double __fma_rn(double a,double b,double c){return std::fma(a,b,c);}
static inline float __fadd_rn(float a,float b){return a+b;}
static inline float __fsub_rn(float a,float b){return a-b;}
static inline float __fmul_rn(float a,float b){return a*b;}
static inline float __fdiv_rn(float a,float b){return a/b;}
static inline float __fsqrt_rn(float a){return std::sqrt(a);}
#include "glibc_flt32.cuh"
extern "C" {
float expf(float x) { return gfk_exp(x); }
float logf(float x) { return gfk_log(x); }
float powf(float x, float y) { return gfk_pow(x, y); }
// deng's own GAMMA (module_shcu_deng.F:4823) calls SIN only for a
// non-positive argument; the scheme calls GAMMA(3.+BVTS) and GAMMA(3.+BVT)
// and nothing else.  The shim pins the symbol to an abort, so a run that
// reached it could not have produced an oracle file.
float sinf(float) { std::abort(); }
}
