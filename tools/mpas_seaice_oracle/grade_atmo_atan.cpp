// Grade the host C library's binary64 atan against WOOF's correctly rounded
// wcr_atan (woof/core/kernels/f64cr.cuh) on the arguments ice_atmo.F90's
// psimhu forms: xqq = sqrt(max(sqrt(|1 - 16 hol|), 1)) with hol clamped to
// [-10, 10], so xqq lies in [1, 161**0.25] (< 3.5622).  Every binary64 in
// [1, 3.5625) is drawn uniformly by bit pattern (splitmix64), and the
// sweep also walks N consecutive words up from 1 and down from 3.5625.
//
//   c++ -O2 -std=c++17 -I woof/core/kernels grade_atmo_atan.cpp -o grade -lm
//   ./grade LOG2_SAMPLES
//
// Prints the sample count, the mismatches and the first few mismatching
// arguments as hex words.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "f64cr.cuh"

static uint64_t bits(double x) { uint64_t u; std::memcpy(&u, &x, 8); return u; }
static double word(uint64_t u) { double x; std::memcpy(&x, &u, 8); return x; }

int main(int argc, char **argv) {
    int lg = argc > 1 ? std::atoi(argv[1]) : 28;
    uint64_t n = 1ULL << lg, lo = bits(1.0), hi = bits(3.5625), span = hi - lo;
    uint64_t s = 0x9e3779b97f4a7c15ULL, bad = 0, total = 0;
    auto check = [&](uint64_t u) {
        double x = word(u);
        double want = std::atan(x), got = wcr_atan(x);
        ++total;
        if (bits(want) != bits(got)) {
            if (bad < 16) std::printf("mismatch x=%016llx glibc=%016llx cr=%016llx\n",
                                      (unsigned long long)u, (unsigned long long)bits(want),
                                      (unsigned long long)bits(got));
            ++bad;
        }
    };
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= z >> 31;
        check(lo + z % span);
    }
    for (uint64_t i = 0; i < (n >> 4); ++i) { check(lo + i); check(hi - 1 - i); }
    std::printf("samples %llu mismatches %llu ambiguous %llu\n", (unsigned long long)total,
                (unsigned long long)bad, (unsigned long long)wcr_ambiguous_count);
    return bad != 0;
}
