// corem_log10f.cuh (cm_log10f) compiled for the host against the host's
// libm.so.6 log10f over every float32 input.  Run where libm is glibc 2.43
// (README.md): the reference the WRF v4.7.1 column oracles link.
#include "../glibc_trig_flt32_proof/host_shim.hpp"
#define FADD(a, b) __fadd_rn((a), (b))
#define FSUB(a, b) __fsub_rn((a), (b))
#define FDIV(a, b) __fdiv_rn((a), (b))
#define __constant__
inline int __clz(int x) { return x == 0 ? 32 : __builtin_clz((unsigned)x); }
#include "../../woof/core/kernels/corem_log10f.cuh"
#include <omp.h>
#include <dlfcn.h>
#include <cstdio>
#include <chrono>
#include <cfenv>
int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  std::fesetround(FE_TONEAREST);
  Dl_info d{};
  dladdr((void*)::log10f, &d);
  printf("log10f resolved=%s threads=%d\n", d.dli_fname, omp_get_max_threads());
  auto t = std::chrono::steady_clock::now();
  uint64_t bad = 0, first = UINT64_MAX;
#pragma omp parallel for reduction(+ : bad) reduction(min : first) schedule(static)
  for (uint64_t k = 0; k < 0x100000000ULL; k++) {
    float x = __uint_as_float((uint32_t)k);
    float ra = ::log10f(x), rb = cm_log10f(x);
    uint32_t a = __float_as_uint(ra), b = __float_as_uint(rb);
    if (a != b && !(std::isnan(ra) && std::isnan(rb))) {
      bad++;
      if (k < first) first = k;
    }
  }
  printf("log10f inputs=4294967296 mismatches=%llu seconds=%.3f",
         (unsigned long long)bad,
         std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count());
  if (bad) printf(" first=%08x", (uint32_t)first);
  puts("");
}
