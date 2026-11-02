// Force-included ahead of mirror_main.cpp when the CPU bridge carries
// woof.core.host_libm's glibc 2.39 transcription
// (tests/test_wdm6_wrf461_oracle.py links the mirror against it):
//   g++ ... -include glibc_libm_shim.h mirror_main.cpp
//       -L<bridge dir> -lgpuwm_preprocess_cpu -Wl,-rpath,<bridge dir>
// The mirror arm's float32 exp/log/pow (gfk_exp/gfk_log/gfk_pow in
// wdm6.cu) are then glibc 2.39's words, the C library the WRF oracle
// linked, on any host rather than the platform libm's.
#include <cmath>
#include <vector>

extern "C" float gpuwm_glibc239_expf(float);
extern "C" float gpuwm_glibc239_logf(float);
extern "C" float gpuwm_glibc239_powf(float, float);

#define expf gpuwm_glibc239_expf
#define logf gpuwm_glibc239_logf
#define powf gpuwm_glibc239_powf
