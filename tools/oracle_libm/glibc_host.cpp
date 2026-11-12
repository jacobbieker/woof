// glibc's libm words for the MPAS oracles on a host whose libm is not glibc.
//
// The committed MPAS oracle fixtures (tools/mali_oracle, tools/mpas_ocean_oracle)
// were written on x86-64 Linux, where gfortran's exp, real ** real, cos, sin,
// atan and tanh call glibc's expf/powf/cosf/sinf/atanf/tanhf (RKIND 4) and exp/pow/tanh
// (RKIND 8), the
// FMA builds the x86-64 ifunc selects.  Another libm (Apple's, on macOS)
// returns different words for some arguments, so an oracle rebuilt there would
// not reproduce the fixture.  On such a host the oracle build.sh scripts link
// this object into the oracles.  Being part of the executable, its functions
// take the place of the libm ones.
//
// It is the repository's own device transcriptions, compiled for the host
// through woof/backend/host/cuda_host_shim.h, the same text the GPU kernels
// call: glibc_flt32.cuh (expf, powf) and glibc_trig_flt32.cuh (cosf, sinf, atanf)
// for RKIND 4, glibc_flt64.cuh (exp, pow) for RKIND 8, all Arm
// optimized-routines transcriptions under MIT, sfire_libm.cuh (tanhf):
// CORE-MATH's correctly rounded tanhf under MIT, which glibc carries since
// 2.41, and cr_tanh_flt64.cuh (tanh): CORE-MATH's correctly rounded binary64
// tanh under MIT, which glibc 2.44 carries.  No glibc source is involved.
// One object serves both precisions.
//
//   c++ -c -O0 -ffp-contract=off -std=c++17 -fno-exceptions -fno-rtti \
//       -I woof/backend/host -I woof/core/kernels tools/oracle_libm/glibc_host.cpp
#include "cuda_host_shim.h"
#include "glibc_flt32.cuh"
#include "glibc_trig_flt32.cuh"
#include "glibc_flt64.cuh"
#include "sfire_libm.cuh"
#include "cr_tanh_flt64.cuh"

extern "C" {
float expf(float x) { return gfk_exp(x); }
float powf(float x, float y) { return gfk_pow(x, y); }
float cosf(float x) { return glibc_cosf(x); }
float sinf(float x) { return glibc_sinf(x); }
float atanf(float x) { return glibc_atanf(x); }
float tanhf(float x) { return sfire_tanhf(x); }
double exp(double x) { return glibc_exp(x); }
double pow(double x, double y) { return glibc_pow(x, y); }
double tanh(double x) { return g64_tanh::cr_tanh(x); }
}
