// The tanh grading argument stream, shared verbatim by the host (grade_tanh.cpp)
// and device (grade_tanh.py) legs: argument i is a pure function of i.
#pragma once
__device__ __forceinline__ uint64_t tanh_mix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
// The branch points of cr_tanh and the hard cases it tabulates.
__device__ const uint64_t tanh_edges[16] = {
    0x3e4d12ed0af1a27fULL, 0x3fd0000000000000ULL, 0x400d76c8b4395810ULL, 0x40330fc1931f09caULL,
    0x3fcac343b179fec4ULL, 0x3fd00764a988bf73ULL, 0x3fd17d1e8a63711fULL, 0x3fd291c601a05276ULL,
    0x3fd36f33d51c264dULL, 0x3fd43eaea23649c3ULL, 0x3fdd88d7550b2826ULL, 0x3fde611aa58ab608ULL,
    0x3fe01efe7ac8c15dULL, 0x3fe1005ec0bccabbULL, 0x3fe33dfeb0fa4bfeULL, 0x3fe49f24ac5cac35ULL,
};
__device__ __forceinline__ uint64_t tanh_span(uint64_t r, uint64_t lo, uint64_t hi) { return lo + r % (hi - lo); }
__device__ __forceinline__ double tanh_arg(uint64_t i) {
    uint64_t r = tanh_mix(i), s = tanh_mix(r) & 0x8000000000000000ULL, u;
    switch (i & 7) {
    case 0: u = r; break;                                                     // every binary64 word
    case 1: u = tanh_span(r, 0x3e10000000000000ULL, 0x4034000000000000ULL) | s; break;  // 2^-30 <= |x| < 20
    case 2: u = tanh_span(r, 0x3c30000000000000ULL, 0x3e60000000000000ULL) | s; break;  // 2^-60 .. 2^-25
    case 3: u = tanh_span(r, 0x3fd0000000000000ULL, 0x400d76c8b4395810ULL) | s; break;  // 1/4 .. ~3.683
    case 4: u = tanh_span(r, 0x400d76c8b4395810ULL, 0x40330fc1931f09caULL) | s; break;  // ~3.683 .. ~19.06
    case 5: u = tanh_span(r, 0x3e4d12ed0af1a27fULL, 0x3fd0000000000000ULL) | s; break;  // 2^-27 .. 1/4
    case 6: u = (tanh_edges[(r >> 40) & 15] + (int64_t)(r & 0xffffff) - 0x800000) | s; break;  // edges +- 2^23 ulp
    default: u = (r & 0x7fffffffffffffffULL) % 0x4034000000000000ULL | s; break;  // |x| < 20, all binades
    }
    return __longlong_as_double((long long)u);
}
