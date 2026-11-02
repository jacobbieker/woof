// Force-included ahead of fsbm_host.cpp (host.py BUILD_FLAGS, -include).
//
// The C library calls the fast_sbm headers make (fast_sbm_common.cuh's
// host libm, fast_sbm_init.cuh's direct calls) go through pointers that
// start at the platform libm.  tools/fast_sbm_host/host.py rebinds them
// (fsbm_host_bind_libm) to the CPU bridge's glibc 2.39 transcription, the
// C library WRF's oracle linked, so the words are the oracle's on any
// host.  The macros route the headers' ::exp, ::logf ... through them;
// sqrt is correctly rounded everywhere and stays direct.  The standard
// headers fsbm_host.cpp includes come first, so the macros reach only the
// fast_sbm sources.  fsbm_host.cpp itself is pinned by the packed-table
// manifest and is not edited for this.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace fsbm_libm {
inline float (*p_expf)(float) = ::expf;
inline float (*p_logf)(float) = ::logf;
inline float (*p_powf)(float, float) = ::powf;
inline double (*p_exp)(double) = ::exp;
inline double (*p_log)(double) = ::log;
inline double (*p_pow)(double, double) = ::pow;
inline double (*p_sin)(double) = ::sin;
inline double (*p_cos)(double) = ::cos;
}  // namespace fsbm_libm

// expf, logf, powf, exp, log, pow, sin, cos, in that order.
extern "C" void fsbm_host_bind_libm(void* const* f) {
  fsbm_libm::p_expf = (float (*)(float))f[0];
  fsbm_libm::p_logf = (float (*)(float))f[1];
  fsbm_libm::p_powf = (float (*)(float, float))f[2];
  fsbm_libm::p_exp = (double (*)(double))f[3];
  fsbm_libm::p_log = (double (*)(double))f[4];
  fsbm_libm::p_pow = (double (*)(double, double))f[5];
  fsbm_libm::p_sin = (double (*)(double))f[6];
  fsbm_libm::p_cos = (double (*)(double))f[7];
}

#define expf fsbm_libm::p_expf
#define logf fsbm_libm::p_logf
#define powf fsbm_libm::p_powf
#define exp fsbm_libm::p_exp
#define log fsbm_libm::p_log
#define pow fsbm_libm::p_pow
#define sin fsbm_libm::p_sin
#define cos fsbm_libm::p_cos
