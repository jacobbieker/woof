// The passive smoke tracer's own per-step arithmetic (woof/core/smoke_tracer.py).
//
// Transport, mixing, boundaries and nest exchange reuse the moist-scalar
// kernels unchanged; these two are the only operations the tracer adds.
// Compiled by woof/core/smoke_tracer.py's direct NVRTC loader with
// -fmad=false --ftz=false, every operation an explicit round-to-nearest
// intrinsic, so the NumPy references in the same module reproduce every
// output word (tests/test_smoke_tracer.py).

// The surface emission as a coupled tendency on the lowest layer:
//   tend0 += (emis * ug_g) / (-dnw[0])
// with ug_g = float32(1e9 * g).  A flux F (kg m-2 s-1) into a layer of
// C(mu)*|dnw|/g kg m-2 of dry air is a coupled rate F*1e9*g/|dnw|.  One
// thread per column; no reduction.
extern "C" __global__ void smoke_surface_source(
    float* __restrict__ tend0, const float* __restrict__ emis,
    const float* __restrict__ dnw, const float ug_g, const int ncol)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const float depth = -dnw[0];
    const float source = __fdiv_rn(__fmul_rn(emis[c], ug_g), depth);
    tend0[c] = __fadd_rn(tend0[c], source);
}

// The column load, ug m-2:
//   col = sum_{k = 0 .. nz-1} smoke[k] * ((c1h[k]*mu + c2h[k]) * (-dnw[k] / g))
// summed upward from k = 0 by one thread per column -- a fixed order, the
// order the NumPy reference takes, with no atomics and no tree.
extern "C" __global__ void smoke_column_load(
    float* __restrict__ col, const float* __restrict__ smoke,
    const float* __restrict__ c1h, const float* __restrict__ c2h,
    const float* __restrict__ dnw, const float* __restrict__ mub,
    const float* __restrict__ mup, const float g, const int nz,
    const int ncol)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const float mu = __fadd_rn(mub[c], mup[c]);
    float acc = 0.0f;
    for (int k = 0; k < nz; ++k) {
        const float chm = __fadd_rn(__fmul_rn(c1h[k], mu), c2h[k]);
        const float depth = __fdiv_rn(-dnw[k], g);
        const float layer = __fmul_rn(chm, depth);
        acc = __fadd_rn(acc, __fmul_rn(smoke[(size_t)k * ncol + c], layer));
    }
    col[c] = acc;
}
