// CUDA C++ on the host, for the host backend (WOOF_BACKEND=host).
//
// woof/backend/host/loader.py preprocesses a unit -- the exact text NVRTC
// is given, with its -D options -- after this header, and compiles the
// result with the host C++ compiler as one shared library.  This header is
// CUDA's language and runtime surface for that build:
//
// * the keywords: __device__/__host__/__global__ vanish, __shared__ is
//   per-worker storage (one block runs on one worker at a time),
//   __constant__ and __device__ variables are ordinary globals;
// * threadIdx/blockIdx/blockDim/gridDim are the running thread's (thread
//   local; a fiber switch sets them);
// * __syncthreads, __syncwarp, the warp shuffles/votes and the
//   __syncthreads_* reductions are barriers over the block's fibers
//   (rt_pool.cpp); atomics are the host's atomic instructions, because
//   blocks run on several workers at once;
// * binary32/binary64 arithmetic is the host's IEEE arithmetic, compiled
//   with -ffp-contract=off: CUDA's --fmad=false words.  A unit CUDA builds
//   with -ftz=true runs with the SSE flush-to-zero and denormals-are-zero
//   modes set for the launch; a --ftz=false unit keeps subnormals.
//
// The launcher at the end (woof_host::launch) is instantiated once per
// kernel by loader.py's generated tail.
#pragma once
#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfenv>
#include <limits>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

// ---------------------------------------------------------------- keywords

// loader.py reads the kernels off the preprocessed unit by this marker
// (WOOF_HOST_SCAN), and defines it away for the build.
#ifdef WOOF_HOST_SCAN
#define __global__ __woof_kernel__
#else
#define __global__
#endif
#define __device__
#define __host__
#define __shared__ static thread_local
#define __constant__
#define __managed__
#define __forceinline__ inline
#define __noinline__ __attribute__((noinline))
#define __restrict__ __restrict
#define __launch_bounds__(...)
#define __align__(n) alignas(n)
#define __grid_constant__
#define __location__(x) __woof_location_##x
#define __woof_location_global
#define __woof_location_device
#define __woof_location_host
#define __woof_location_shared static thread_local
#define __woof_location_constant
#define __CUDACC__ 1
#define __CUDACC_RTC__ 1
// NVRTC compiles device code only, so its __CUDA_ARCH__ branches are the
// ones CUDA runs (the Metal translator defines the same; translate.CUDA_ARCH).
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 860
#endif
#define __WOOF_HOST__ 1

// ---------------------------------------------------------------- vector types

#if defined(__FLT16_MAX__)
typedef _Float16 half;
typedef _Float16 __half;
#endif

#define WOOF_VEC1(T, N, A) struct alignas(A) N##1 { T x; }; \
    static inline N##1 make_##N##1(T x) { N##1 r; r.x = x; return r; }
#define WOOF_VEC2(T, N, A) struct alignas(A) N##2 { T x, y; }; \
    static inline N##2 make_##N##2(T x, T y) { N##2 r; r.x = x; r.y = y; return r; }
#define WOOF_VEC3(T, N, A) struct alignas(A) N##3 { T x, y, z; }; \
    static inline N##3 make_##N##3(T x, T y, T z) { N##3 r; r.x = x; r.y = y; r.z = z; return r; }
#define WOOF_VEC4(T, N, A) struct alignas(A) N##4 { T x, y, z, w; }; \
    static inline N##4 make_##N##4(T x, T y, T z, T w) \
    { N##4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }
// CUDA's alignments (vector_types.h)
WOOF_VEC1(char, char, 1) WOOF_VEC2(char, char, 2) WOOF_VEC3(char, char, 1) WOOF_VEC4(char, char, 4)
WOOF_VEC1(unsigned char, uchar, 1) WOOF_VEC2(unsigned char, uchar, 2)
WOOF_VEC3(unsigned char, uchar, 1) WOOF_VEC4(unsigned char, uchar, 4)
WOOF_VEC1(short, short, 2) WOOF_VEC2(short, short, 4) WOOF_VEC3(short, short, 2) WOOF_VEC4(short, short, 8)
WOOF_VEC1(unsigned short, ushort, 2) WOOF_VEC2(unsigned short, ushort, 4)
WOOF_VEC3(unsigned short, ushort, 2) WOOF_VEC4(unsigned short, ushort, 8)
WOOF_VEC1(int, int, 4) WOOF_VEC2(int, int, 8) WOOF_VEC3(int, int, 4) WOOF_VEC4(int, int, 16)
WOOF_VEC1(unsigned int, uint, 4) WOOF_VEC2(unsigned int, uint, 8)
WOOF_VEC3(unsigned int, uint, 4) WOOF_VEC4(unsigned int, uint, 16)
WOOF_VEC1(long long, longlong, 8) WOOF_VEC2(long long, longlong, 16)
WOOF_VEC3(long long, longlong, 8) WOOF_VEC4(long long, longlong, 16)
WOOF_VEC1(unsigned long long, ulonglong, 8) WOOF_VEC2(unsigned long long, ulonglong, 16)
WOOF_VEC3(unsigned long long, ulonglong, 8) WOOF_VEC4(unsigned long long, ulonglong, 16)
WOOF_VEC1(float, float, 4) WOOF_VEC2(float, float, 8) WOOF_VEC3(float, float, 4) WOOF_VEC4(float, float, 16)
WOOF_VEC1(double, double, 8) WOOF_VEC2(double, double, 16)
WOOF_VEC3(double, double, 8) WOOF_VEC4(double, double, 16)
#undef WOOF_VEC1
#undef WOOF_VEC2
#undef WOOF_VEC3
#undef WOOF_VEC4

struct dim3 {
    unsigned int x, y, z;
    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
    constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    constexpr operator uint3() const { return uint3{x, y, z}; }
};

static thread_local uint3 threadIdx;
static thread_local uint3 blockIdx;
static thread_local dim3 blockDim;
static thread_local dim3 gridDim;
static constexpr int warpSize = 32;

// ---------------------------------------------------------------- the runtime (rt_pool.cpp)

struct woof_fiber;
extern "C" {
int woof_rt_run(unsigned long long nblocks, int workers,
                void (*fn)(void *, unsigned long long), void *ctx);
void woof_rt_fail(const char *message);
int woof_rt_failed();
woof_fiber *woof_rt_fiber(int index);
void woof_rt_fiber_start(woof_fiber *f, void (*entry)(void *), void *arg);
void woof_rt_resume(woof_fiber *f);
void woof_rt_yield(woof_fiber *f);
}

namespace woof_host {

// A failed device assertion or __trap(): the thread stops (the exception
// unwinds to its launcher frame) and the launch reports the error.
struct Trap {};

enum { F_NEW, F_RUN, F_BLOCK, F_WARP, F_DONE };

struct Fiber {
    woof_fiber *fiber;
    int state;
    int stack;          // index of the worker fiber (stack) it runs on, -1 none
    uint3 tid;
    unsigned lane, warp;
    int pred;
};

// One block's fibers on the worker running it.
struct Sched {
    Fiber f[1024];
    int n = 0;
    int cur = -1;
    void (*body)(void *) = nullptr;
    void *ctx = nullptr;
    int bar_count = 0, bar_or = 0, bar_and = 0;
    unsigned warp_mask[32];                 // lanes released by the last warp barrier
    unsigned long long slots[32][32];       // shuffle/vote exchange, [warp][lane]
    int free_stacks[1024];
    int n_free = 0, n_stacks = 0;
};

static thread_local Sched *sched = nullptr;
static thread_local int ftz = 0;            // this launch flushes binary32 subnormals
static thread_local unsigned char *dynamic_shared = nullptr;

static inline void fail(const char *what) { woof_rt_fail(what); throw Trap(); }

static inline void yield_as(int state)
{
    Sched *s = sched;
    Fiber &f = s->f[s->cur];
    f.state = state;
    woof_rt_yield(f.fiber);
}

static void fiber_entry(void *arg)
{
    Sched *s = sched;
    const int i = (int)(intptr_t)arg;
    try {
        s->body(s->ctx);
    } catch (const Trap &) {
    }
    s->f[i].state = F_DONE;
    woof_rt_yield(s->f[i].fiber);
    __builtin_unreachable();
}

// Run every thread of the current block as a fiber; returns when all are done.
static void run_block_fibers(Sched &s, unsigned nx, unsigned ny, unsigned nz)
{
    const int n = (int)(nx * ny * nz);
    s.n = n;
    for (int t = 0; t < n; ++t) {
        Fiber &f = s.f[t];
        f.state = F_NEW;
        f.stack = -1;
        f.tid = uint3{(unsigned)t % nx, ((unsigned)t / nx) % ny, (unsigned)t / (nx * ny)};
        f.lane = (unsigned)t & 31u;
        f.warp = (unsigned)t >> 5;
        f.pred = 0;
    }
    int live = n;
    for (;;) {
        for (int t = 0; t < n; ++t) {
            Fiber &f = s.f[t];
            if (f.state == F_NEW) {
                int k = s.n_free ? s.free_stacks[--s.n_free] : s.n_stacks++;
                f.fiber = woof_rt_fiber(k);
                if (f.fiber == nullptr) {
                    woof_rt_fail("the host backend could not map a fiber stack");
                    return;
                }
                f.stack = k;
                woof_rt_fiber_start(f.fiber, fiber_entry, (void *)(intptr_t)t);
                f.state = F_RUN;
            }
            if (f.state != F_RUN) continue;
            s.cur = t;
            threadIdx = f.tid;
            woof_rt_resume(f.fiber);
            s.cur = -1;
            if (f.state == F_DONE) {
                s.free_stacks[s.n_free++] = f.stack;
                f.stack = -1;
                --live;
            }
        }
        if (live == 0) return;
        // Everyone left is waiting.  A block barrier releases when every
        // live thread is at it; else each warp whose live lanes are all at
        // a warp barrier is released.
        bool block = true;
        for (int t = 0; t < n && block; ++t)
            block = s.f[t].state == F_DONE || s.f[t].state == F_BLOCK;
        bool released = false;
        if (block) {
            int count = 0, any = 0, all = 1;
            for (int t = 0; t < n; ++t) {
                if (s.f[t].state != F_BLOCK) continue;
                count += s.f[t].pred != 0;
                any |= s.f[t].pred != 0;
                all &= s.f[t].pred != 0;
                s.f[t].state = F_RUN;
            }
            s.bar_count = count;
            s.bar_or = any;
            s.bar_and = all;
            released = true;
        } else {
            const int warps = (n + 31) / 32;
            for (int w = 0; w < warps; ++w) {
                bool ready = true, waiting = false;
                for (int t = w * 32; t < n && t < w * 32 + 32; ++t) {
                    if (s.f[t].state == F_WARP) waiting = true;
                    else if (s.f[t].state != F_DONE) ready = false;
                }
                if (!(ready && waiting)) continue;
                unsigned mask = 0;
                for (int t = w * 32; t < n && t < w * 32 + 32; ++t)
                    if (s.f[t].state == F_WARP) {
                        s.f[t].state = F_RUN;
                        mask |= 1u << (t & 31);
                    }
                s.warp_mask[w] = mask;
                released = true;
            }
        }
        if (!released) {
            // Some threads wait at __syncthreads while others of the same
            // warp wait at a warp collective: no schedule completes it.
            woof_rt_fail("barrier deadlock: threads of one block wait at different barriers "
                         "(divergent __syncthreads or warp collective)");
            for (int t = 0; t < n; ++t)
                if (s.f[t].stack >= 0) s.free_stacks[s.n_free++] = s.f[t].stack;
            return;
        }
    }
}

}  // namespace woof_host

// ---------------------------------------------------------------- barriers and warp collectives

static inline void __woof_serial_barrier()
{
    if (blockDim.x * blockDim.y * blockDim.z > 1)
        woof_host::fail("__syncthreads in a unit built without fibers (the host backend "
                        "did not see the barrier when it read the unit)");
}

static inline void __syncthreads()
{
    if (woof_host::sched) woof_host::yield_as(woof_host::F_BLOCK);
    else __woof_serial_barrier();
}

static inline int __woof_sync_reduce(int pred, int which)
{
    woof_host::Sched *s = woof_host::sched;
    if (!s) {
        __woof_serial_barrier();
        return which == 0 ? (pred != 0) : (pred != 0);
    }
    s->f[s->cur].pred = pred;
    woof_host::yield_as(woof_host::F_BLOCK);
    return which == 0 ? s->bar_count : which == 1 ? s->bar_or : s->bar_and;
}

static inline int __syncthreads_count(int pred) { return __woof_sync_reduce(pred, 0); }
static inline int __syncthreads_or(int pred) { return __woof_sync_reduce(pred, 1); }
static inline int __syncthreads_and(int pred) { return __woof_sync_reduce(pred, 2); }

static inline void __syncwarp(unsigned int = 0xffffffffu)
{
    if (woof_host::sched) woof_host::yield_as(woof_host::F_WARP);
}

static inline void __threadfence() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
static inline void __threadfence_block() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
static inline void __threadfence_system() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }

// Exchange ``bits`` across the warp: every live lane deposits, then reads
// lane ``src``'s word.  Two warp barriers: all deposit before any reads,
// and all read before any lane's next deposit.
static inline unsigned long long __woof_exchange(unsigned long long bits, int src, bool own)
{
    woof_host::Sched *s = woof_host::sched;
    if (!s) return bits;
    woof_host::Fiber &f = s->f[s->cur];
    const unsigned warp = f.warp;
    s->slots[warp][f.lane] = bits;
    woof_host::yield_as(woof_host::F_WARP);
    unsigned long long r = own ? bits : s->slots[warp][src & 31];
    woof_host::yield_as(woof_host::F_WARP);
    return r;
}

template <class T>
static inline unsigned long long __woof_bits(T v)
{
    static_assert(sizeof(T) <= 8, "warp shuffle of a value wider than 8 bytes");
    unsigned long long b = 0;
    std::memcpy(&b, &v, sizeof(T));
    return b;
}

template <class T>
static inline T __woof_from_bits(unsigned long long b)
{
    T v;
    std::memcpy(&v, &b, sizeof(T));
    return v;
}

static inline unsigned __woof_lane()
{
    woof_host::Sched *s = woof_host::sched;
    return s ? s->f[s->cur].lane : 0u;
}

template <class T>
static inline T __shfl_sync(unsigned, T v, int src, int width = 32)
{
    const unsigned lane = __woof_lane();
    const unsigned base = lane & ~(unsigned)(width - 1);
    const unsigned from = base + ((unsigned)src & (unsigned)(width - 1));
    return __woof_from_bits<T>(__woof_exchange(__woof_bits(v), (int)from, false));
}

template <class T>
static inline T __shfl_up_sync(unsigned, T v, unsigned delta, int width = 32)
{
    const unsigned lane = __woof_lane();
    const unsigned base = lane & ~(unsigned)(width - 1);
    const bool own = lane < base + delta;
    return __woof_from_bits<T>(__woof_exchange(__woof_bits(v), (int)(lane - delta), own));
}

template <class T>
static inline T __shfl_down_sync(unsigned, T v, unsigned delta, int width = 32)
{
    const unsigned lane = __woof_lane();
    const unsigned base = lane & ~(unsigned)(width - 1);
    const bool own = lane + delta >= base + (unsigned)width;
    return __woof_from_bits<T>(__woof_exchange(__woof_bits(v), (int)(lane + delta), own));
}

template <class T>
static inline T __shfl_xor_sync(unsigned, T v, int mask, int width = 32)
{
    const unsigned lane = __woof_lane();
    const unsigned base = lane & ~(unsigned)(width - 1);
    const unsigned to = lane ^ (unsigned)mask;
    const bool own = to < base || to >= base + (unsigned)width;
    return __woof_from_bits<T>(__woof_exchange(__woof_bits(v), (int)to, own));
}

static inline unsigned __woof_vote(int pred)
{
    woof_host::Sched *s = woof_host::sched;
    if (!s) return pred ? 1u : 0u;
    woof_host::Fiber &f = s->f[s->cur];
    const unsigned warp = f.warp;
    s->slots[warp][f.lane] = pred != 0;
    woof_host::yield_as(woof_host::F_WARP);
    unsigned ballot = 0;
    const unsigned members = s->warp_mask[warp];
    for (unsigned l = 0; l < 32; ++l)
        if ((members >> l) & 1u && s->slots[warp][l]) ballot |= 1u << l;
    woof_host::yield_as(woof_host::F_WARP);
    return ballot;
}

static inline unsigned __ballot_sync(unsigned, int pred) { return __woof_vote(pred); }
static inline int __any_sync(unsigned, int pred) { return __woof_vote(pred) != 0u; }
static inline int __all_sync(unsigned, int pred)
{
    woof_host::Sched *s = woof_host::sched;
    if (!s) return pred != 0;
    const unsigned ballot = __woof_vote(pred);
    return ballot == s->warp_mask[s->f[s->cur].warp];
}

static inline unsigned __activemask()
{
    woof_host::Sched *s = woof_host::sched;
    if (!s) return 1u;
    const unsigned warp = s->f[s->cur].warp;
    unsigned mask = 0;
    for (int t = (int)warp * 32; t < s->n && t < (int)warp * 32 + 32; ++t)
        if (s->f[t].state != woof_host::F_DONE) mask |= 1u << (t & 31);
    return mask;
}

static inline void __trap() { woof_host::fail("__trap() in a kernel"); }
static inline void __brkpt() {}
static inline void __nanosleep(unsigned) {}
#if defined(__x86_64__)
static inline long long clock64() { return (long long)__builtin_ia32_rdtsc(); }
#else
static inline long long clock64() { return 0; }
#endif
static inline void __woof_assert_fail(const char *expr, const char *file, int line)
{
    char text[512];
    std::snprintf(text, sizeof text, "device assertion failed: %s (%s:%d)", expr, file, line);
    woof_host::fail(text);
}
#undef assert
#define assert(e) ((e) ? (void)0 : __woof_assert_fail(#e, __FILE__, __LINE__))

static inline void *__woof_dynamic_shared() { return woof_host::dynamic_shared; }

// ---------------------------------------------------------------- atomics

// Blocks run on several workers at once, so these are the host's atomic
// read-modify-writes (sequentially consistent).  A floating add is a
// compare-and-swap loop, as CUDA's own fallback is.
#define WOOF_ATOMIC_INT(T) \
    static inline T atomicAdd(T *a, T v) { return __atomic_fetch_add(a, v, __ATOMIC_SEQ_CST); } \
    static inline T atomicSub(T *a, T v) { return __atomic_fetch_sub(a, v, __ATOMIC_SEQ_CST); } \
    static inline T atomicExch(T *a, T v) { return __atomic_exchange_n(a, v, __ATOMIC_SEQ_CST); } \
    static inline T atomicAnd(T *a, T v) { return __atomic_fetch_and(a, v, __ATOMIC_SEQ_CST); } \
    static inline T atomicOr(T *a, T v) { return __atomic_fetch_or(a, v, __ATOMIC_SEQ_CST); } \
    static inline T atomicXor(T *a, T v) { return __atomic_fetch_xor(a, v, __ATOMIC_SEQ_CST); } \
    static inline T atomicCAS(T *a, T cmp, T v) \
    { __atomic_compare_exchange_n(a, &cmp, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); return cmp; } \
    static inline T atomicMin(T *a, T v) \
    { T o = __atomic_load_n(a, __ATOMIC_SEQ_CST); \
      while (v < o && !__atomic_compare_exchange_n(a, &o, v, true, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {} \
      return o; } \
    static inline T atomicMax(T *a, T v) \
    { T o = __atomic_load_n(a, __ATOMIC_SEQ_CST); \
      while (v > o && !__atomic_compare_exchange_n(a, &o, v, true, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {} \
      return o; }
WOOF_ATOMIC_INT(int)
WOOF_ATOMIC_INT(unsigned int)
WOOF_ATOMIC_INT(long)
WOOF_ATOMIC_INT(unsigned long)
WOOF_ATOMIC_INT(long long)
WOOF_ATOMIC_INT(unsigned long long)
#undef WOOF_ATOMIC_INT

static inline unsigned int atomicInc(unsigned int *a, unsigned int limit)
{
    unsigned int o = __atomic_load_n(a, __ATOMIC_SEQ_CST);
    while (!__atomic_compare_exchange_n(a, &o, o >= limit ? 0u : o + 1u, true,
                                        __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
    return o;
}

static inline unsigned int atomicDec(unsigned int *a, unsigned int limit)
{
    unsigned int o = __atomic_load_n(a, __ATOMIC_SEQ_CST);
    while (!__atomic_compare_exchange_n(a, &o, (o == 0u || o > limit) ? limit : o - 1u, true,
                                        __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
    return o;
}

template <class F, class U>
static inline F __woof_atomic_float(F *a, F v, int op)
{
    static_assert(sizeof(F) == sizeof(U), "word size");
    U *w = reinterpret_cast<U *>(a);
    U o = __atomic_load_n(w, __ATOMIC_SEQ_CST);
    for (;;) {
        F old;
        std::memcpy(&old, &o, sizeof old);
        F nv = op == 0 ? old + v : op == 1 ? old - v : v;
        U n;
        std::memcpy(&n, &nv, sizeof n);
        if (__atomic_compare_exchange_n(w, &o, n, true, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
            return old;
    }
}

static inline float atomicAdd(float *a, float v) { return __woof_atomic_float<float, unsigned int>(a, v, 0); }
static inline float atomicSub(float *a, float v) { return __woof_atomic_float<float, unsigned int>(a, v, 1); }
static inline float atomicExch(float *a, float v) { return __woof_atomic_float<float, unsigned int>(a, v, 2); }
static inline double atomicAdd(double *a, double v)
{ return __woof_atomic_float<double, unsigned long long>(a, v, 0); }
static inline double atomicExch(double *a, double v)
{ return __woof_atomic_float<double, unsigned long long>(a, v, 2); }

#define WOOF_SCOPED(name) \
    template <class P, class V> static inline auto name##_block(P p, V v) -> decltype(name(p, v)) \
    { return name(p, v); } \
    template <class P, class V> static inline auto name##_system(P p, V v) -> decltype(name(p, v)) \
    { return name(p, v); }
WOOF_SCOPED(atomicAdd) WOOF_SCOPED(atomicSub) WOOF_SCOPED(atomicExch) WOOF_SCOPED(atomicMin)
WOOF_SCOPED(atomicMax) WOOF_SCOPED(atomicAnd) WOOF_SCOPED(atomicOr) WOOF_SCOPED(atomicXor)
#undef WOOF_SCOPED

// ---------------------------------------------------------------- math

// CUDA's math is overloaded in the global namespace (sqrt(float) is
// sqrtf); the C library's is not, so std's overloads are brought in.
using std::abs; using std::acos; using std::acosh; using std::asin; using std::asinh;
using std::atan; using std::atan2; using std::atanh; using std::cbrt; using std::ceil;
using std::copysign; using std::cos; using std::cosh; using std::erf; using std::erfc;
using std::exp; using std::exp2; using std::expm1; using std::fabs; using std::fdim;
using std::floor; using std::fma; using std::fmod; using std::frexp;
using std::hypot; using std::ilogb; using std::ldexp; using std::lgamma; using std::llrint;
using std::llround; using std::log; using std::log10; using std::log1p; using std::log2;
using std::logb; using std::lrint; using std::lround; using std::modf; using std::nearbyint;
using std::nextafter; using std::pow; using std::remainder; using std::rint; using std::round;
using std::scalbn; using std::sin; using std::sinh; using std::sqrt; using std::tan;
using std::tanh; using std::tgamma; using std::trunc;
using std::isfinite; using std::isinf; using std::isnan; using std::signbit;

// min.f32/max.f32 (and .f64): a NaN operand yields the other, two NaNs the
// canonical NaN, and -0 orders below +0.  The C library's fminf returns
// either zero.
static inline float __woof_fminf(float a, float b)
{
    if (a != a) return b != b ? __builtin_nanf("") : b;
    if (b != b) return a;
    if (a == b) return std::signbit(a) ? a : b;
    return a < b ? a : b;
}
static inline float __woof_fmaxf(float a, float b)
{
    if (a != a) return b != b ? __builtin_nanf("") : b;
    if (b != b) return a;
    if (a == b) return std::signbit(a) ? b : a;
    return a > b ? a : b;
}
static inline double __woof_fmin(double a, double b)
{
    if (a != a) return b != b ? __builtin_nan("") : b;
    if (b != b) return a;
    if (a == b) return std::signbit(a) ? a : b;
    return a < b ? a : b;
}
static inline double __woof_fmax(double a, double b)
{
    if (a != a) return b != b ? __builtin_nan("") : b;
    if (b != b) return a;
    if (a == b) return std::signbit(a) ? b : a;
    return a > b ? a : b;
}
static inline float __woof_fmin(float a, float b) { return __woof_fminf(a, b); }
static inline float __woof_fmax(float a, float b) { return __woof_fmaxf(a, b); }
template <class A, class B, class = typename std::enable_if<
    std::is_arithmetic<A>::value && std::is_arithmetic<B>::value>::type>
static inline double __woof_fmin(A a, B b) { return __woof_fmin((double)a, (double)b); }
template <class A, class B, class = typename std::enable_if<
    std::is_arithmetic<A>::value && std::is_arithmetic<B>::value>::type>
static inline double __woof_fmax(A a, B b) { return __woof_fmax((double)a, (double)b); }
#define fminf __woof_fminf
#define fmaxf __woof_fmaxf
#define fmin __woof_fmin
#define fmax __woof_fmax

// CUDA's min/max: integer and floating overloads, and the usual arithmetic
// conversions for mixed operands.
#define WOOF_MINMAX(T) \
    static inline T min(T a, T b) { return a < b ? a : b; } \
    static inline T max(T a, T b) { return a > b ? a : b; }
WOOF_MINMAX(int) WOOF_MINMAX(unsigned int) WOOF_MINMAX(long) WOOF_MINMAX(unsigned long)
WOOF_MINMAX(long long) WOOF_MINMAX(unsigned long long)
#undef WOOF_MINMAX
static inline float min(float a, float b) { return __woof_fminf(a, b); }
static inline float max(float a, float b) { return __woof_fmaxf(a, b); }
static inline double min(double a, double b) { return __woof_fmin(a, b); }
static inline double max(double a, double b) { return __woof_fmax(a, b); }
template <class A, class B, class = typename std::enable_if<
    std::is_arithmetic<A>::value && std::is_arithmetic<B>::value &&
    !std::is_same<A, B>::value>::type>
static inline auto min(A a, B b) -> decltype(a + b)
{ typedef decltype(a + b) T; return min((T)a, (T)b); }
template <class A, class B, class = typename std::enable_if<
    std::is_arithmetic<A>::value && std::is_arithmetic<B>::value &&
    !std::is_same<A, B>::value>::type>
static inline auto max(A a, B b) -> decltype(a + b)
{ typedef decltype(a + b) T; return max((T)a, (T)b); }

// CUDA functions the C library lacks, or declares with another meaning
// (glibc's math.h declares __expf etc. as its own internal names, and C23's
// rsqrt/sinpi): woof's own definitions, under macros.
static inline float __woof_rsqrtf(float x) { return 1.0f / sqrtf(x); }
static inline double __woof_rsqrt(double x) { return 1.0 / std::sqrt(x); }
static inline float __woof_rsqrt(float x) { return __woof_rsqrtf(x); }
static inline float __woof_rcbrtf(float x) { return 1.0f / cbrtf(x); }
static inline double __woof_rcbrt(double x) { return 1.0 / std::cbrt(x); }
static inline float __woof_rcbrt(float x) { return __woof_rcbrtf(x); }
static inline float __woof_exp10f(float x) { return exp10f(x); }
static inline double __woof_exp10(double x) { return ::exp10(x); }
static inline float __woof_exp10(float x) { return exp10f(x); }
static inline float __woof_sinpif(float x) { return sinf((float)M_PI * x); }
static inline float __woof_cospif(float x) { return cosf((float)M_PI * x); }
static inline double __woof_sinpi(double x) { return std::sin(M_PI * x); }
static inline double __woof_cospi(double x) { return std::cos(M_PI * x); }
static inline void __woof_sincospif(float x, float *s, float *c)
{ *s = __woof_sinpif(x); *c = __woof_cospif(x); }
static inline float __woof_normcdff(float x) { return 0.5f * erfcf(-x * (float)M_SQRT1_2); }
static inline double __woof_normcdf(double x) { return 0.5 * std::erfc(-x * M_SQRT1_2); }
static inline float __woof_fdividef(float a, float b) { return a / b; }
static inline float __woof_saturatef(float x)
{ return x != x ? 0.0f : x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }
#define rsqrtf __woof_rsqrtf
#define rsqrt __woof_rsqrt
#define rcbrtf __woof_rcbrtf
#define rcbrt __woof_rcbrt
#define exp10f __woof_exp10f
#define exp10 __woof_exp10
#define sinpif __woof_sinpif
#define cospif __woof_cospif
#define sinpi __woof_sinpi
#define cospi __woof_cospi
#define sincospif __woof_sincospif
#define normcdff __woof_normcdff
#define normcdf __woof_normcdf
#define fdividef __woof_fdividef
#define __saturatef __woof_saturatef

// The fast-math intrinsics are approximations in CUDA's hardware; here they
// are the library functions (a recorded difference, docs/dev/host-backend.md).
static inline float __woof_fast_expf(float x) { return expf(x); }
static inline float __woof_fast_exp10f(float x) { return __woof_exp10f(x); }
static inline float __woof_fast_logf(float x) { return logf(x); }
static inline float __woof_fast_log2f(float x) { return log2f(x); }
static inline float __woof_fast_log10f(float x) { return log10f(x); }
static inline float __woof_fast_powf(float x, float y) { return powf(x, y); }
static inline float __woof_fast_sinf(float x) { return sinf(x); }
static inline float __woof_fast_cosf(float x) { return cosf(x); }
static inline float __woof_fast_tanf(float x) { return tanf(x); }
static inline void __woof_fast_sincosf(float x, float *s, float *c) { *s = sinf(x); *c = cosf(x); }
#define __expf __woof_fast_expf
#define __exp10f __woof_fast_exp10f
#define __logf __woof_fast_logf
#define __log2f __woof_fast_log2f
#define __log10f __woof_fast_log10f
#define __powf __woof_fast_powf
#define __sinf __woof_fast_sinf
#define __cosf __woof_fast_cosf
#define __tanf __woof_fast_tanf
#define __sincosf __woof_fast_sincosf
#define __fdividef __woof_fdividef

// ---------------------------------------------------------------- rounded operations

// Round-to-nearest: one IEEE operation each (contraction is off).
static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __frcp_rn(float a) { return 1.0f / a; }
static inline float __fsqrt_rn(float a) { return sqrtf(a); }
static inline float __frsqrt_rn(float a) { return 1.0f / sqrtf(a); }
static inline float __fmaf_rn(float a, float b, float c) { return fmaf(a, b, c); }
static inline double __dadd_rn(double a, double b) { return a + b; }
static inline double __dsub_rn(double a, double b) { return a - b; }
static inline double __dmul_rn(double a, double b) { return a * b; }
static inline double __ddiv_rn(double a, double b) { return a / b; }
static inline double __drcp_rn(double a) { return 1.0 / a; }
static inline double __dsqrt_rn(double a) { return std::sqrt(a); }
static inline double __fma_rn(double a, double b, double c) { return std::fma(a, b, c); }

// Directed rounding: the operation under the rounding mode it names.  The
// operands pass through volatile so the compiler cannot fold the operation
// under its own (nearest) rounding.
template <class F>
static inline auto __woof_rounded(int mode, F op) -> decltype(op())
{
    const int saved = std::fegetround();
    std::fesetround(mode);
    auto r = op();
    std::fesetround(saved);
    return r;
}
#define WOOF_DIRECTED(name, T, mode, expr) \
    static inline T name(T a, T b) { volatile T va = a, vb = b; \
        return __woof_rounded(mode, [&]() -> T { T x = va, y = vb; T r = expr; volatile T vr = r; return vr; }); }
WOOF_DIRECTED(__fadd_rz, float, FE_TOWARDZERO, x + y) WOOF_DIRECTED(__fadd_ru, float, FE_UPWARD, x + y)
WOOF_DIRECTED(__fadd_rd, float, FE_DOWNWARD, x + y) WOOF_DIRECTED(__fmul_rz, float, FE_TOWARDZERO, x * y)
WOOF_DIRECTED(__fmul_ru, float, FE_UPWARD, x * y) WOOF_DIRECTED(__fmul_rd, float, FE_DOWNWARD, x * y)
WOOF_DIRECTED(__fdiv_rz, float, FE_TOWARDZERO, x / y) WOOF_DIRECTED(__fdiv_ru, float, FE_UPWARD, x / y)
WOOF_DIRECTED(__fdiv_rd, float, FE_DOWNWARD, x / y) WOOF_DIRECTED(__fsub_rz, float, FE_TOWARDZERO, x - y)
WOOF_DIRECTED(__dadd_rz, double, FE_TOWARDZERO, x + y) WOOF_DIRECTED(__dadd_ru, double, FE_UPWARD, x + y)
WOOF_DIRECTED(__dadd_rd, double, FE_DOWNWARD, x + y) WOOF_DIRECTED(__dmul_rz, double, FE_TOWARDZERO, x * y)
WOOF_DIRECTED(__dmul_ru, double, FE_UPWARD, x * y) WOOF_DIRECTED(__dmul_rd, double, FE_DOWNWARD, x * y)
WOOF_DIRECTED(__ddiv_rz, double, FE_TOWARDZERO, x / y) WOOF_DIRECTED(__ddiv_ru, double, FE_UPWARD, x / y)
WOOF_DIRECTED(__ddiv_rd, double, FE_DOWNWARD, x / y)
#undef WOOF_DIRECTED
static inline float __double2float_rn(double d) { return (float)d; }
static inline float __woof_d2f(int mode, double d)
{ volatile double vd = d; return __woof_rounded(mode, [&]() -> float { volatile float r = (float)vd; return r; }); }
static inline float __double2float_rz(double d) { return __woof_d2f(FE_TOWARDZERO, d); }
static inline float __double2float_ru(double d) { return __woof_d2f(FE_UPWARD, d); }
static inline float __double2float_rd(double d) { return __woof_d2f(FE_DOWNWARD, d); }

// Conversions to integers saturate and take NaN to 0, as cvt.*i does
// (a C++ cast of an out-of-range value is undefined).
template <class I, class F>
static inline I __woof_to_int(F x)
{
    if (x != x) return 0;
    if (x <= (F)std::numeric_limits<I>::min()) return std::numeric_limits<I>::min();
    if (x >= (F)std::numeric_limits<I>::max()) return std::numeric_limits<I>::max();
    return (I)x;
}
static inline int __float2int_rn(float x) { return __woof_to_int<int>(nearbyintf(x)); }
static inline int __float2int_rz(float x) { return __woof_to_int<int>(truncf(x)); }
static inline int __float2int_rd(float x) { return __woof_to_int<int>(floorf(x)); }
static inline int __float2int_ru(float x) { return __woof_to_int<int>(ceilf(x)); }
static inline unsigned int __float2uint_rn(float x) { return __woof_to_int<unsigned int>(nearbyintf(x)); }
static inline unsigned int __float2uint_rz(float x) { return __woof_to_int<unsigned int>(truncf(x)); }
static inline unsigned int __float2uint_rd(float x) { return __woof_to_int<unsigned int>(floorf(x)); }
static inline unsigned int __float2uint_ru(float x) { return __woof_to_int<unsigned int>(ceilf(x)); }
static inline long long __float2ll_rn(float x) { return __woof_to_int<long long>(nearbyintf(x)); }
static inline long long __float2ll_rz(float x) { return __woof_to_int<long long>(truncf(x)); }
static inline long long __float2ll_rd(float x) { return __woof_to_int<long long>(floorf(x)); }
static inline long long __float2ll_ru(float x) { return __woof_to_int<long long>(ceilf(x)); }
static inline int __double2int_rn(double x) { return __woof_to_int<int>(std::nearbyint(x)); }
static inline int __double2int_rz(double x) { return __woof_to_int<int>(std::trunc(x)); }
static inline int __double2int_rd(double x) { return __woof_to_int<int>(std::floor(x)); }
static inline int __double2int_ru(double x) { return __woof_to_int<int>(std::ceil(x)); }
static inline unsigned int __double2uint_rn(double x) { return __woof_to_int<unsigned int>(std::nearbyint(x)); }
static inline unsigned int __double2uint_rz(double x) { return __woof_to_int<unsigned int>(std::trunc(x)); }
static inline long long __double2ll_rn(double x) { return __woof_to_int<long long>(std::nearbyint(x)); }
static inline long long __double2ll_rz(double x) { return __woof_to_int<long long>(std::trunc(x)); }
static inline long long __double2ll_rd(double x) { return __woof_to_int<long long>(std::floor(x)); }
static inline long long __double2ll_ru(double x) { return __woof_to_int<long long>(std::ceil(x)); }
static inline float __int2float_rn(int i) { return (float)i; }
static inline float __uint2float_rn(unsigned int i) { return (float)i; }
static inline float __ll2float_rn(long long i) { return (float)i; }
static inline float __ull2float_rn(unsigned long long i) { return (float)i; }
static inline double __int2double_rn(int i) { return (double)i; }
static inline double __uint2double_rn(unsigned int i) { return (double)i; }
static inline double __ll2double_rn(long long i) { return (double)i; }
static inline double __ull2double_rn(unsigned long long i) { return (double)i; }

// ---------------------------------------------------------------- bit reinterpretation and integer intrinsics

static inline float __int_as_float(int i) { float f; std::memcpy(&f, &i, 4); return f; }
static inline int __float_as_int(float f) { int i; std::memcpy(&i, &f, 4); return i; }
static inline float __uint_as_float(unsigned int u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline unsigned int __float_as_uint(float f) { unsigned int u; std::memcpy(&u, &f, 4); return u; }
static inline long long __double_as_longlong(double d) { long long l; std::memcpy(&l, &d, 8); return l; }
static inline double __longlong_as_double(long long l) { double d; std::memcpy(&d, &l, 8); return d; }
static inline int __double2hiint(double d) { return (int)((unsigned long long)__double_as_longlong(d) >> 32); }
static inline int __double2loint(double d) { return (int)(unsigned int)__double_as_longlong(d); }
static inline double __hiloint2double(int hi, int lo)
{ return __longlong_as_double((long long)(((unsigned long long)(unsigned int)hi << 32) | (unsigned int)lo)); }

static inline int __clz(int x) { return x == 0 ? 32 : __builtin_clz((unsigned)x); }
static inline int __clzll(long long x) { return x == 0 ? 64 : __builtin_clzll((unsigned long long)x); }
static inline int __popc(unsigned int x) { return __builtin_popcount(x); }
static inline int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
static inline int __ffs(int x) { return __builtin_ffs(x); }
static inline int __ffsll(long long x) { return __builtin_ffsll(x); }
static inline unsigned int __brev(unsigned int x)
{
    unsigned int r = 0;
    for (int i = 0; i < 32; ++i) r |= ((x >> i) & 1u) << (31 - i);
    return r;
}
static inline unsigned long long __brevll(unsigned long long x)
{
    unsigned long long r = 0;
    for (int i = 0; i < 64; ++i) r |= ((x >> i) & 1ull) << (63 - i);
    return r;
}
static inline unsigned int __umulhi(unsigned int a, unsigned int b)
{ return (unsigned int)(((unsigned long long)a * b) >> 32); }
static inline int __mulhi(int a, int b) { return (int)(((long long)a * b) >> 32); }
static inline unsigned long long __umul64hi(unsigned long long a, unsigned long long b)
{ return (unsigned long long)(((unsigned __int128)a * b) >> 64); }
static inline long long __mul64hi(long long a, long long b)
{ return (long long)(((__int128)a * b) >> 64); }
static inline unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s)
{
    const unsigned long long v = ((unsigned long long)y << 32) | x;
    unsigned int r = 0;
    for (int i = 0; i < 4; ++i) r |= (unsigned int)((v >> (8 * ((s >> (4 * i)) & 7u))) & 0xffu) << (8 * i);
    return r;
}
template <class T> static inline T __ldg(const T *p) { return *p; }

// ---------------------------------------------------------------- inline PTX (translate.rewrite_ptx_asm)

static inline void __woof_compiler_fence() { __atomic_signal_fence(__ATOMIC_SEQ_CST); }

// PTX without .ftz keeps binary32 subnormals whatever -ftz says: in a
// flushing launch these clear the SSE flush modes around the one operation.
#if defined(__x86_64__)
#define WOOF_IEEE_OP(name, insn) \
    static inline float __woof_ieee_##name(float a, float b) \
    { \
        if (!woof_host::ftz) return __woof_ieee_plain_##name(a, b); \
        unsigned int saved = _mm_getcsr(), ieee = saved & ~0x8040u; \
        __asm__ volatile("ldmxcsr %2\n\t" insn " %1, %0\n\tldmxcsr %3" \
                         : "+x"(a) : "x"(b), "m"(ieee), "m"(saved)); \
        return a; \
    }
#else
#define WOOF_IEEE_OP(name, insn) \
    static inline float __woof_ieee_##name(float a, float b) { return __woof_ieee_plain_##name(a, b); }
#endif
static inline float __woof_ieee_plain_add(float a, float b) { return a + b; }
static inline float __woof_ieee_plain_sub(float a, float b) { return a - b; }
static inline float __woof_ieee_plain_mul(float a, float b) { return a * b; }
static inline float __woof_ieee_plain_div(float a, float b) { return a / b; }
WOOF_IEEE_OP(add, "addss") WOOF_IEEE_OP(sub, "subss") WOOF_IEEE_OP(mul, "mulss") WOOF_IEEE_OP(div, "divss")
#undef WOOF_IEEE_OP
static inline float __woof_ieee_min(float a, float b) { return __woof_fminf(a, b); }
static inline float __woof_ieee_max(float a, float b) { return __woof_fmaxf(a, b); }

// ---------------------------------------------------------------- the launcher

namespace woof_host {

struct Launch {
    unsigned grid[3];
    unsigned block[3];
    unsigned long long shared_bytes;
    int workers;
    int ftz;
    int fibers;
    int pad;
};

template <class T>
static inline T load_arg(void *p)
{
    typedef typename std::remove_cv<T>::type U;
    alignas(U) unsigned char buf[sizeof(U)];
    std::memcpy(buf, p, sizeof(U));
    return *reinterpret_cast<U *>(buf);
}

template <class... A, std::size_t... I>
static inline std::tuple<typename std::decay<A>::type...>
unpack(void **argv, std::index_sequence<I...>)
{
    return std::tuple<typename std::decay<A>::type...>(
        load_arg<typename std::decay<A>::type>(argv[I])...);
}

template <class... A>
struct Call {
    void (*kernel)(A...);
    std::tuple<typename std::decay<A>::type...> args;
    const Launch *launch;
};

template <class... A>
static void run_thread(void *ctx)
{
    Call<A...> *c = static_cast<Call<A...> *>(ctx);
    std::apply(c->kernel, c->args);
}

struct DynamicShared {
    unsigned char *p = nullptr;
    unsigned long long size = 0;
    ~DynamicShared() { std::free(p); }
};
static thread_local DynamicShared dyn;
static thread_local Sched *worker_sched = nullptr;

template <class... A>
static void run_block(void *ctx, unsigned long long b)
{
    Call<A...> *c = static_cast<Call<A...> *>(ctx);
    const Launch &L = *c->launch;
    if (woof_rt_failed()) return;
    gridDim = dim3(L.grid[0], L.grid[1], L.grid[2]);
    blockDim = dim3(L.block[0], L.block[1], L.block[2]);
    blockIdx = uint3{(unsigned)(b % L.grid[0]), (unsigned)((b / L.grid[0]) % L.grid[1]),
                     (unsigned)(b / ((unsigned long long)L.grid[0] * L.grid[1]))};
    if (L.shared_bytes > dyn.size) {
        std::free(dyn.p);
        dyn.p = static_cast<unsigned char *>(std::aligned_alloc(64, (L.shared_bytes + 63) & ~63ull));
        dyn.size = L.shared_bytes;
    }
    dynamic_shared = dyn.p;
    ftz = L.ftz;
#if defined(__x86_64__)
    const unsigned int saved = _mm_getcsr();
    _mm_setcsr(L.ftz ? (saved | 0x8040u) : (saved & ~0x8040u));
#endif
    if (L.fibers && L.block[0] * L.block[1] * L.block[2] > 1) {
        if (worker_sched == nullptr) worker_sched = new Sched;
        Sched &s = *worker_sched;
        s.body = &run_thread<A...>;
        s.ctx = ctx;
        sched = &s;
        run_block_fibers(s, L.block[0], L.block[1], L.block[2]);
        sched = nullptr;
    } else {
        for (unsigned z = 0; z < L.block[2]; ++z)
            for (unsigned y = 0; y < L.block[1]; ++y)
                for (unsigned x = 0; x < L.block[0]; ++x) {
                    threadIdx = uint3{x, y, z};
                    try {
                        std::apply(c->kernel, c->args);
                    } catch (const Trap &) {
                        goto out;
                    }
                }
    }
out:
#if defined(__x86_64__)
    _mm_setcsr(saved);
#endif
    return;
}

template <class... A>
static int launch(void (*kernel)(A...), const Launch *L, void **argv)
{
    Call<A...> call{kernel, unpack<A...>(argv, std::index_sequence_for<A...>{}), L};
    const unsigned long long blocks =
        (unsigned long long)L->grid[0] * L->grid[1] * L->grid[2];
    if (woof_rt_run(blocks, L->workers, &run_block<A...>, &call) != 0) return 1;
    return woof_rt_failed() ? 2 : 0;
}

// The parameter kinds Python packs arguments by: p (pointer), b (bool),
// f/i/u (floating, signed, unsigned) with the byte size, s (by-value
// struct) with its size.
template <class D, bool = std::is_enum<D>::value>
struct Scalar { typedef D type; };
template <class D>
struct Scalar<D, true> { typedef typename std::underlying_type<D>::type type; };

template <class T>
static inline void describe(char *&out)
{
    typedef typename std::decay<T>::type D;
    typedef typename Scalar<D>::type S;
    char kind = std::is_pointer<D>::value ? 'p'
              : std::is_same<S, bool>::value ? 'b'
              : std::is_floating_point<S>::value ? 'f'
              : std::is_integral<S>::value ? (std::is_signed<S>::value ? 'i' : 'u')
              : 's';
    out += std::sprintf(out, "%c%zu,", kind, sizeof(D));
}

template <class... A>
static const char *signature(void (*)(A...))
{
    static char text[16 * (sizeof...(A) + 1)];
    char *out = text;
    *out = '\0';
    int unused[] = {0, (describe<A>(out), 0)...};
    (void)unused;
    return text;
}

}  // namespace woof_host
