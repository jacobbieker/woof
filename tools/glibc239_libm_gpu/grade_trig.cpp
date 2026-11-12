// Grading kernels for tools/glibc239_libm_gpu/glibc239_trig_flt64.cuh (see
// grade_trig.py).  Built as a shared object by grade_trig.py:
//   c++ -std=c++17 -O2 -ffp-contract=off -fopenmp -shared -fPIC \
//       -I woof/backend/host -I tools/glibc239_libm_gpu grade_trig.cpp -ldl
// trig_args writes the deterministic argument stream; the other entry
// points evaluate the transcription or the host libm over an argument
// array, or sweep a range of the stream through both.
#include "cuda_host_shim.h"
#include "glibc239_trig_flt64.cuh"
#include <dlfcn.h>
#include <stdint.h>

#include "grade_trig_args.h"

extern "C" {

static double (*host_fn[3])(double) = {nullptr, nullptr, nullptr};

static inline double port(int f, double x) {
    return f == 0 ? g239::g239_sin(x) : f == 1 ? g239::g239_cos(x) : g239::g239_atan(x);
}
static inline uint64_t word(double x) { return (uint64_t)__double_as_longlong(x); }
// The range where the oracle host's glibc 2.44 build of branred.c may differ.
static inline bool branred_range(int f, double x) { return f < 2 && fabs(x) >= 105414350.0; }

void trig_args(int f, uint64_t start, uint64_t n, double *out) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)n; ++i) out[i] = trig_arg(f, start + (uint64_t)i);
}

int trig_bind_host_libm(void) {
    void *h = dlopen("libm.so.6", RTLD_NOW);
    if (!h) return 0;
    host_fn[0] = (double (*)(double))dlsym(h, "sin");
    host_fn[1] = (double (*)(double))dlsym(h, "cos");
    host_fn[2] = (double (*)(double))dlsym(h, "atan");
    return host_fn[0] && host_fn[1] && host_fn[2];
}

void trig_host_libm(int f, const double *x, uint64_t n, double *out) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)n; ++i) out[i] = host_fn[f](x[i]);
}

void trig_transcription(int f, const double *x, uint64_t n, double *out) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)n; ++i) out[i] = port(f, x[i]);
}

// Streams start..start+n through the transcription and the host libm.
// counts[0]: mismatches outside the branred range, counts[1]: inside it,
// counts[2]: arguments inside it.  Every NaN equals every NaN.
void trig_sweep(int f, uint64_t start, uint64_t n, uint64_t *counts, uint64_t *first_mismatch) {
    uint64_t c0 = 0, c1 = 0, c2 = 0, first = ~0ULL;
#pragma omp parallel for schedule(static) reduction(+:c0,c1,c2) reduction(min:first)
    for (int64_t i = 0; i < (int64_t)n; ++i) {
        const double x = trig_arg(f, start + (uint64_t)i);
        const double a = port(f, x), b = host_fn[f](x);
        const bool huge = branred_range(f, x);
        c2 += huge;
        if (!(a != a && b != b) && word(a) != word(b)) {
            if (huge) ++c1;
            else {
                ++c0;
                if (start + (uint64_t)i < first) first = start + (uint64_t)i;
            }
        }
    }
    counts[0] = c0; counts[1] = c1; counts[2] = c2; *first_mismatch = first;
}

}
