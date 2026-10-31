// woof/backend/host/libm32.cpp's functions against the host C library,
// word for word: every binary32 input for the unary functions, 2^28 sampled
// pairs for powf.  Meaningful on a glibc host (the functions are glibc's).
//   c++ -O2 -std=c++17 -ffp-contract=off -fno-unsafe-math-optimizations \
//     -I woof/backend/host -I woof/core/kernels tools/host_libm32_check.cpp -o /tmp/check
#include "cuda_host_shim.h"
#include "glibc_flt32.cuh"
#include "glibc_trig_flt32.cuh"
#include "log10_flt32.cuh"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>

static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float flt(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool same(float a, float b) { return bits(a) == bits(b) || (std::isnan(a) && std::isnan(b)); }

template <class F, class G>
static int unary(const char* name, F port, G libc) {
  uint64_t bad = 0; uint32_t first = 0;
  for (uint64_t u = 0; u <= 0xFFFFFFFFull; ++u) {
    float x = flt((uint32_t)u);
    if (!same(port(x), libc(x))) { if (!bad) first = (uint32_t)u; ++bad; }
  }
  std::printf("%-7s %llu mismatches%s", name, (unsigned long long)bad, bad ? "" : "\n");
  if (bad) std::printf(" (first input 0x%08x)\n", first);
  return bad != 0;
}

int main() {
  int status = 0;
  status |= unary("expf", [](float x) { return gfk_exp(x); }, [](float x) { return expf(x); });
  status |= unary("logf", [](float x) { return gfk_log(x); }, [](float x) { return logf(x); });
  status |= unary("log10f", [](float x) { return gfk_log10(x); }, [](float x) { return log10f(x); });
  status |= unary("sinf", [](float x) { return glibc_sinf(x); }, [](float x) { return sinf(x); });
  status |= unary("cosf", [](float x) { return glibc_cosf(x); }, [](float x) { return cosf(x); });
  status |= unary("tanf", [](float x) { return glibc_tanf(x); }, [](float x) { return tanf(x); });
  status |= unary("asinf", [](float x) { return glibc_asinf(x); }, [](float x) { return asinf(x); });
  status |= unary("acosf", [](float x) { return glibc_acosf(x); }, [](float x) { return acosf(x); });
  status |= unary("atanf", [](float x) { return glibc_atanf(x); }, [](float x) { return atanf(x); });
  uint64_t bad = 0, state = 0x9E3779B97F4A7C15ull;
  for (uint64_t k = 0; k < (1ull << 28); ++k) {
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    float x = flt((uint32_t)state), y = flt((uint32_t)(state >> 32));
    if (k & 1) y = std::nearbyint(y);  // integer exponents: the checkint paths
    if (!same(gfk_pow(x, y), powf(x, y))) ++bad;
  }
  std::printf("powf    %llu mismatches in 2^28 samples\n", (unsigned long long)bad);
  return status | (bad != 0);
}
