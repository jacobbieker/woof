// Each operator retains the rounding boundary of its eager CuPy launch.
// Four adjacent values use vector transactions when pointers are aligned.
struct GlueWordTable {
    unsigned long long rows[64][3];
};

#if __CUDA_ARCH__ >= 700
#define GLUE_TABLE_PARAM const __grid_constant__ GlueWordTable
#else
#define GLUE_TABLE_PARAM const GlueWordTable
#endif

__device__ __forceinline__ bool glue_aligned(const void* p) {
    return ((unsigned long long)p & 15ull) == 0;
}

__device__ __forceinline__ void glue_add_chunk(
    const float* src, float* dst, unsigned long long n) {
    unsigned long long at = ((unsigned long long)blockIdx.x * blockDim.x
                             + threadIdx.x) * 4;
    if (glue_aligned(src) && glue_aligned(dst)) {
        if (at >= n) return;
        if (at + 3 >= n) {
            for (int q = 0; q != 4 && at + q < n; ++q)
                dst[at + q] = __fadd_rn(dst[at + q], src[at + q]);
            return;
        }
        float4 a = ((const float4*)dst)[at / 4];
        float4 b = ((const float4*)src)[at / 4];
        float4 out;
        out.x = __fadd_rn(a.x, b.x);
        out.y = __fadd_rn(a.y, b.y);
        out.z = __fadd_rn(a.z, b.z);
        out.w = __fadd_rn(a.w, b.w);
        ((float4*)dst)[at / 4] = out;
    } else {
        // Lane-strided fallback keeps each scalar transaction coalesced.
        unsigned long long first = (unsigned long long)blockIdx.x * blockDim.x * 4;
        #pragma unroll
        for (int q = 0; q != 4; ++q) {
            unsigned long long i = first + threadIdx.x + (unsigned long long)q * blockDim.x;
            if (i < n) dst[i] = __fadd_rn(dst[i], src[i]);
        }
    }
}

extern "C" __global__ void glue_add_arrays(GLUE_TABLE_PARAM table) {
    const unsigned long long* row = table.rows[blockIdx.y];
    glue_add_chunk((const float*)row[0], (float*)row[1], row[2]);
}

extern "C" __global__ void glue_add(
    const float* src, float* dst, unsigned long long n) {
    glue_add_chunk(src, dst, n);
}

__device__ __forceinline__ float glue_theta_value(
    const float* thb, float thp, unsigned long long at, int ncol, int full) {
    return __fadd_rn(thb[full ? at : at / ncol], thp);
}

extern "C" __global__ void glue_total_theta(
    const float* thb, const float* thp, float* dst,
    unsigned long long n, int ncol, int full) {
    unsigned long long at = ((unsigned long long)blockIdx.x * blockDim.x
                             + threadIdx.x) * 4;
    if (glue_aligned(thp) && glue_aligned(dst)) {
        if (at >= n) return;
        if (at + 3 >= n) {
            for (int q = 0; q != 4 && at + q < n; ++q)
                dst[at + q] = glue_theta_value(thb, thp[at + q], at + q, ncol, full);
            return;
        }
        float4 a = ((const float4*)thp)[at / 4];
        float4 out;
        if (full && glue_aligned(thb)) {
            float4 b = ((const float4*)thb)[at / 4];
            out.x = __fadd_rn(b.x, a.x);
            out.y = __fadd_rn(b.y, a.y);
            out.z = __fadd_rn(b.z, a.z);
            out.w = __fadd_rn(b.w, a.w);
        } else {
            out.x = glue_theta_value(thb, a.x, at, ncol, full);
            out.y = glue_theta_value(thb, a.y, at + 1, ncol, full);
            out.z = glue_theta_value(thb, a.z, at + 2, ncol, full);
            out.w = glue_theta_value(thb, a.w, at + 3, ncol, full);
        }
        ((float4*)dst)[at / 4] = out;
    } else {
        unsigned long long first = (unsigned long long)blockIdx.x * blockDim.x * 4;
        #pragma unroll
        for (int q = 0; q != 4; ++q) {
            unsigned long long i = first + threadIdx.x + (unsigned long long)q * blockDim.x;
            if (i < n) dst[i] = glue_theta_value(thb, thp[i], i, ncol, full);
        }
    }
}

__device__ __forceinline__ float glue_theta_rate(
    float rth, unsigned long long at, const float* mub, const float* mup,
    const float* c1, const float* c2, const float* msft, int ncol, int mapped) {
    int col = at % ncol;
    int k = at / ncol;
    float mu = __fadd_rn(mub[col], mup[col]);
    float coupling = __fadd_rn(__fmul_rn(c1[k], mu), c2[k]);
    float rate = __fdiv_rn(rth, coupling);
    return mapped ? __fmul_rn(rate, msft[col]) : rate;
}

extern "C" __global__ void glue_capture_theta_forcing(
    const float* rth, const float* mub, const float* mup,
    const float* c1, const float* c2, const float* msft, float* dst,
    unsigned long long n, int ncol, int mapped) {
    unsigned long long at = ((unsigned long long)blockIdx.x * blockDim.x
                             + threadIdx.x) * 4;
    if (glue_aligned(rth) && glue_aligned(dst)) {
        if (at >= n) return;
        if (at + 3 >= n) {
            for (int q = 0; q != 4 && at + q < n; ++q)
                dst[at + q] = glue_theta_rate(rth[at + q], at + q, mub, mup, c1, c2, msft, ncol, mapped);
            return;
        }
        float4 a = ((const float4*)rth)[at / 4];
        float4 out;
        out.x = glue_theta_rate(a.x, at, mub, mup, c1, c2, msft, ncol, mapped);
        out.y = glue_theta_rate(a.y, at + 1, mub, mup, c1, c2, msft, ncol, mapped);
        out.z = glue_theta_rate(a.z, at + 2, mub, mup, c1, c2, msft, ncol, mapped);
        out.w = glue_theta_rate(a.w, at + 3, mub, mup, c1, c2, msft, ncol, mapped);
        ((float4*)dst)[at / 4] = out;
    } else {
        unsigned long long first = (unsigned long long)blockIdx.x * blockDim.x * 4;
        #pragma unroll
        for (int q = 0; q != 4; ++q) {
            unsigned long long i = first + threadIdx.x + (unsigned long long)q * blockDim.x;
            if (i < n) dst[i] = glue_theta_rate(rth[i], i, mub, mup, c1, c2, msft, ncol, mapped);
        }
    }
}

// dst += src * scale for each table row (src, dst, n), one row per blockIdx.y:
// the words of bandwidth_glue.py's gpuwm_accumulate_scaled, a rounded
// product and then a rounded sum, for many accumulators in one launch.
extern "C" __global__ void glue_accumulate_scaled_arrays(
    GLUE_TABLE_PARAM table, float scale) {
    const unsigned long long* row = table.rows[blockIdx.y];
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x
                                 + threadIdx.x;
    if (i >= row[2]) return;
    const float* src = (const float*)row[0];
    float* dst = (float*)row[1];
    dst[i] = __fadd_rn(dst[i], __fmul_rn(src[i], scale));
}

// +0 at the outer frame of each row's (planes, ny, nx) word array, rows
// (unused, address, frame elements): ring_frame.py's gpuwm_zero_frame for
// several buffers of one horizontal shape in one launch.  Element r of a
// plane counts the y_lo top rows, the y_hi bottom rows, then each middle
// row's side columns (x_lo first, x_hi last, or all nx when they meet).
// 32-bit indices: the caller launches only when every index fits.
extern "C" __global__ void glue_zero_frames(
    GLUE_TABLE_PARAM table, int nx, int ny, int x_lo, int x_hi, int y_lo,
    int y_hi, int side, int per_plane) {
    const unsigned long long* row = table.rows[blockIdx.y];
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((unsigned long long)j >= row[2]) return;
    const unsigned int plane = j / (unsigned int)per_plane;
    unsigned int r = j - plane * (unsigned int)per_plane;
    const unsigned int w = (unsigned int)nx;
    const unsigned int top = (unsigned int)y_lo * w;
    const unsigned int bottom = (unsigned int)y_hi * w;
    unsigned int y, x;
    if (r < top) {
        y = r / w; x = r - y * w;
    } else if (r - top < bottom) {
        r -= top;
        const unsigned int k = r / w;
        y = (unsigned int)(ny - y_hi) + k; x = r - k * w;
    } else {
        r -= top + bottom;
        const unsigned int k = r / (unsigned int)side;
        const unsigned int c = r - k * (unsigned int)side;
        y = (unsigned int)y_lo + k;
        x = (side == nx || c < (unsigned int)x_lo)
            ? c : (unsigned int)(nx - x_hi) + (c - (unsigned int)x_lo);
    }
    ((unsigned int*)row[1])[plane * (unsigned int)ny * w + y * w + x] = 0u;
}
