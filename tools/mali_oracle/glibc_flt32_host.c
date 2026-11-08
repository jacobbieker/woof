/* powf and expf for the MALI oracles on a host whose libm is not glibc.
 *
 * The committed MALI fixtures were written on x86-64 Linux, where gfortran's
 * real(4) ** real(4) and exp() call glibc 2.39's powf and expf (the FMA
 * builds: the reference host has FMA, TOOLCHAIN.txt records it).  Another
 * libm (Apple's, on macOS) returns different words for some arguments, so an
 * oracle rebuilt there would not reproduce the fixture.  build.sh links this
 * object into the oracles on such a host.  Being part of the executable, its
 * powf and expf take the place of the libm ones.
 *
 * It is the repository's glibc float32 transcription (Arm
 * optimized-routines, MIT; woof/core/kernels/glibc_flt32.cuh), the same
 * text the GPU kernels call, compiled for the host.  The NVIDIA
 * round-to-nearest intrinsics it spells are plain IEEE operations here
 * (compiled at -O0 -ffp-contract=off), and __fma_rn is C99 fma(), which is
 * exactly rounded on every conforming libm.  No glibc source is involved.
 */
#include <math.h>
#include <stdbool.h>
#include <string.h>

#define __device__ static

static float __int_as_float(int i) { float f; memcpy(&f, &i, sizeof f); return f; }
static float __uint_as_float(unsigned int i) { float f; memcpy(&f, &i, sizeof f); return f; }
static unsigned int __float_as_uint(float f) { unsigned int i; memcpy(&i, &f, sizeof i); return i; }
static long long __double_as_longlong(double d) { long long i; memcpy(&i, &d, sizeof i); return i; }
static double __longlong_as_double(long long i) { double d; memcpy(&d, &i, sizeof d); return d; }
static float __double2float_rn(double d) { return (float)d; }
static float __fadd_rn(float a, float b) { return a + b; }
static float __fsub_rn(float a, float b) { return a - b; }
static float __fmul_rn(float a, float b) { return a * b; }
static float __fdiv_rn(float a, float b) { return a / b; }
static float __fsqrt_rn(float a) { return sqrtf(a); }
static double __dadd_rn(double a, double b) { return a + b; }
static double __dsub_rn(double a, double b) { return a - b; }
static double __dmul_rn(double a, double b) { return a * b; }
static double __ddiv_rn(double a, double b) { return a / b; }
static double __fma_rn(double a, double b, double c) { return fma(a, b, c); }

#include "glibc_flt32.cuh"

float powf(float x, float y) { return gfk_pow(x, y); }
float expf(float x) { return gfk_exp(x); }
