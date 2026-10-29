// The binary32 libm the FARMS oracle links: woof's word-exact ports
// compiled for the host behind cuda_host_shim.h, under woof_* names
// (libm_shim.c gives them the C library's names).  The same bodies the
// device kernel compiles, so the oracle's libm is one fixed function set on
// any host (macOS libSystem, glibc 2.39 or 2.44):
//   expf, logf, powf   glibc 2.39 (Arm optimized-routines), glibc_flt32.cuh
//   log10f             glibc 2.39 log10f over gfk_log, log10_flt32.cuh
//   sinf, cosf, acosf  glibc 2.43 (Arm sinf/cosf FMA ifunc; CORE-MATH
//                      acosf), glibc_trig_flt32.cuh
//   sinhf              correctly rounded (glibc >= 2.41 / CORE-MATH),
//                      sinhf_flt32.cuh, proven over all 2^32 arguments
//                      (sinhf-proof.log)
#include "cuda_host_shim.h"
#include "glibc_flt32.cuh"
#include "glibc_trig_flt32.cuh"
#include "log10_flt32.cuh"
#include "sinhf_flt32.cuh"

extern "C" {
float woof_expf(float x) { return gfk_exp(x); }
float woof_logf(float x) { return gfk_log(x); }
float woof_log10f(float x) { return gfk_log10(x); }
float woof_powf(float x, float y) { return gfk_pow(x, y); }
float woof_sinf(float x) { return glibc_sinf(x); }
float woof_cosf(float x) { return glibc_cosf(x); }
float woof_acosf(float x) { return glibc_acosf(x); }
float woof_sinhf(float x) { return gfk_sinh(x); }
}
