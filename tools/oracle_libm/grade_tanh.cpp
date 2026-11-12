// Grading kernels for woof/core/kernels/cr_tanh_flt64.cuh (see grade_tanh.py).
//
// Built as a shared object by grade_tanh.py:
//   c++ -std=c++17 -O2 -ffp-contract=off -fopenmp -shared -fPIC \
//       -I woof/backend/host -I woof/core/kernels -I tools/oracle_libm grade_tanh.cpp -ldl
//
// tanh_args writes the deterministic argument stream (SplitMix64 of the
// index, eight interleaved families, the same on every host); the other
// entry points evaluate the transcription or the host libm over an argument array.  The
// device leg compiles the same tanh_arg function from TANH_ARGS_SOURCE in
// grade_tanh.py, so this file must stay in step with it.
#include "cuda_host_shim.h"
#include "cr_tanh_flt64.cuh"
#include <dlfcn.h>
#include <stdint.h>

#include "grade_tanh_args.h"

extern "C" {

void tanh_args(uint64_t start, uint64_t n, double *out) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)n; ++i) out[i] = tanh_arg(start + (uint64_t)i);
}

static double (*host_tanh)(double) = nullptr;

int tanh_bind_host_libm(void) {
    void *h = dlopen("libm.so.6", RTLD_NOW);
    if (!h) return 0;
    host_tanh = (double (*)(double))dlsym(h, "tanh");
    return host_tanh != nullptr;
}

void tanh_host_libm(const double *x, uint64_t n, double *out) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)n; ++i) out[i] = host_tanh(x[i]);
}

void tanh_transcription(const double *x, uint64_t n, double *out) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)n; ++i) out[i] = g64_tanh::cr_tanh(x[i]);
}

// Streams start..start+n through the transcription and the host libm and
// counts word mismatches (every NaN equals every NaN).
void tanh_sweep(uint64_t start, uint64_t n, uint64_t *mismatch, uint64_t *first_mismatch) {
    uint64_t ml = 0, first = ~0ULL;
#pragma omp parallel for schedule(static) reduction(+:ml) reduction(min:first)
    for (int64_t i = 0; i < (int64_t)n; ++i) {
        double x = tanh_arg(start + (uint64_t)i);
        double a = g64_tanh::cr_tanh(x), b = host_tanh(x);
        if (!(a != a && b != b) && g64_tanh::asuint64(a) != g64_tanh::asuint64(b)) {
            ++ml;
            if (start + (uint64_t)i < first) first = start + (uint64_t)i;
        }
    }
    *mismatch = ml; *first_mismatch = first;
}

}
