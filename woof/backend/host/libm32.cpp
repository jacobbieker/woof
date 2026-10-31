// glibc's binary32 libm, word for word on any host: woof's CUDA ports
// (woof/core/kernels/glibc_flt32.cuh, glibc_trig_flt32.cuh, log10_flt32.cuh)
// compiled as host C++ behind cuda_host_shim.h.  Host code that reproduces
// gfortran's calls into glibc (WRF's real(4) COS/SIN/TAN/ALOG/ALOG10/**)
// calls these instead of the platform's C library when that is not glibc.
#include "cuda_host_shim.h"
#include "glibc_flt32.cuh"
#include "glibc_trig_flt32.cuh"
#include "log10_flt32.cuh"

extern "C" {
float woof_expf(float x) { return gfk_exp(x); }
float woof_logf(float x) { return gfk_log(x); }
float woof_log10f(float x) { return gfk_log10(x); }
float woof_powf(float x, float y) { return gfk_pow(x, y); }
float woof_sinf(float x) { return glibc_sinf(x); }
float woof_cosf(float x) { return glibc_cosf(x); }
float woof_tanf(float x) { return glibc_tanf(x); }
float woof_asinf(float x) { return glibc_asinf(x); }
float woof_acosf(float x) { return glibc_acosf(x); }
float woof_atanf(float x) { return glibc_atanf(x); }
}
