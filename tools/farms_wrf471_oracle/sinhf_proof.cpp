// gfk_sinh (woof/core/kernels/sinhf_flt32.cuh) against the host C
// library's sinhf over every binary32 argument, and (with -DLONG_DOUBLE_REF)
// each disagreement against a long-double reference, so a disagreement can
// be attributed: the host's word or ours is the misrounded one.
//
//   c++ -O2 -std=c++17 -ffp-contract=off -fno-builtin -fopenmp \
//     -I woof/backend/host -I woof/core/kernels \
//     tools/farms_wrf471_oracle/sinhf_proof.cpp -o sinhf_proof
//   ./sinhf_proof            # prints a summary and every mismatch
//
// Meaningful against glibc >= 2.41 (sinhf is CORE-MATH's correctly rounded
// function there).  The gate is zero mismatches.
#include "cuda_host_shim.h"
#include "sinhf_flt32.cuh"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float flt(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

int main() {
  std::vector<uint32_t> bad;
  uint64_t count = 0;
#pragma omp parallel
  {
    std::vector<uint32_t> local;
    uint64_t local_count = 0;
#pragma omp for schedule(static)
    for (int64_t u = 0; u <= 0xFFFFFFFFll; ++u) {
      float x = flt((uint32_t)u);
      float a = gfk_sinh(x);
      float b = sinhf(x);
      ++local_count;
      if (bits(a) != bits(b) && !(std::isnan(a) && std::isnan(b))) local.push_back((uint32_t)u);
    }
#pragma omp critical
    {
      bad.insert(bad.end(), local.begin(), local.end());
      count += local_count;
    }
  }
  std::sort(bad.begin(), bad.end());
  std::printf("sinhf: %llu arguments, %zu mismatches\n", (unsigned long long)count, bad.size());
  for (uint32_t u : bad) {
    float x = flt(u);
    long double ref = sinhl((long double)x);
    std::printf("  x=0x%08x (%a) woof=0x%08x host=0x%08x sinhl=%La\n", u, (double)x,
                bits(gfk_sinh(x)), bits(sinhf(x)), ref);
  }
  return bad.empty() ? 0 : 1;
}
