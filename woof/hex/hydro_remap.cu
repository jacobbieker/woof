// The hex cell <-> LSM grid exchange on the device (woof.hex.hydro_remap_gpu), the opt-in route of
// woof.hex.hydro_remap.CellGridExchange.  The overlap weights W[g, c] (m2, binary64) are CSR twice: grid rows
// (to_grid) and cell rows (the transpose, each cell's entries in ascending grid order, as scipy's CSC product
// visits them).  Every sum starts at 0.0 and runs over its row in the stored order, each product and sum
// rounded as written (fuse = 0) or as one fma (fuse = 1: a host scipy built with multiply-add contraction
// computes y += a * x as fma(a, x, y)); the quotient by the area is the correctly rounded one.  So each word is
// the host operator's on the host that set ``fuse``.
//
// One thread per output; nothing is reduced across threads, so the order never depends on the launch.

__device__ __forceinline__ double hydro_acc(double s, double w, double x, int fuse) {
    return fuse ? fma(w, x, s) : __dadd_rn(s, __dmul_rn(w, x));
}

// q_g = sum_c W[g, c] x_c / A_g for ``fields`` cell fields (rows ``stride`` words apart): the binary64 means
// and their binary32 cast.
extern "C" __global__ void hydro_to_grid(const int *indptr, const int *index, const double *weight,
                                         const double *area, const float *source, long long stride,
                                         int fields, int rows, int fuse, double *exact, float *out) {
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= (long long)fields * rows) return;
    int k = (int)(t / rows), i = (int)(t - (long long)k * rows);
    const float *x = source + (long long)k * stride;
    double s = 0.0;
    for (int e = indptr[i]; e < indptr[i + 1]; ++e) s = hydro_acc(s, weight[e], (double)x[index[e]], fuse);
    double q = __ddiv_rn(s, area[i]);
    exact[t] = q;
    out[t] = (float)q;
}

// A cell state's share of the grid's change: q_c += sum_g W[g, c] (after_g - before_g) / A_c, in binary64 from
// the binary32 state, cast back in place.  ``prior`` keeps the state as it was, ``exact`` the binary64 result.
extern "C" __global__ void hydro_increment(const int *indptr, const int *index, const double *weight,
                                           const double *area, const float *before, const float *after,
                                           int grid, float *state, long long stride, int fields, int cells,
                                           int fuse, float *prior, double *exact) {
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= (long long)fields * cells) return;
    int k = (int)(t / cells), c = (int)(t - (long long)k * cells);
    const float *b = before + (long long)k * grid;
    const float *a = after + (long long)k * grid;
    double s = 0.0;
    for (int e = indptr[c]; e < indptr[c + 1]; ++e) {
        int g = index[e];
        s = hydro_acc(s, weight[e], __dsub_rn((double)a[g], (double)b[g]), fuse);
    }
    float *q = state + (long long)k * stride + c;
    float old = *q;
    double value = __dadd_rn((double)old, __ddiv_rn(s, area[c]));
    prior[t] = old;
    exact[t] = value;
    *q = (float)value;
}

// A grid field returned to the cells, sum_g W[g, c] x_g / A_c (zero outside the grid), binary64 and its cast.
extern "C" __global__ void hydro_to_cells(const int *indptr, const int *index, const double *weight,
                                          const double *area, const float *source, int cells, int fuse,
                                          double *exact, float *out) {
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= cells) return;
    double s = 0.0;
    for (int e = indptr[c]; e < indptr[c + 1]; ++e) s = hydro_acc(s, weight[e], (double)source[index[e]], fuse);
    double q = __ddiv_rn(s, area[c]);
    exact[c] = q;
    out[c] = (float)q;
}

// Per grid cell, the class with the largest overlap area; ties to the smaller class.  Each class's area is
// summed over the row's entries in the stored order from 0.0, as numpy's add.at over the COO entries does.
extern "C" __global__ void hydro_dominant(const int *indptr, const int *index, const double *weight,
                                          const int *category, int rows, int *out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= rows) return;
    int lo = indptr[i], hi = indptr[i + 1];
    int best = 0;
    double most = -1.0;
    for (int e = lo; e < hi; ++e) {
        int cls = category[index[e]];
        bool seen = false;
        for (int f = lo; f < e; ++f) seen = seen || category[index[f]] == cls;
        if (seen) continue;
        double total = 0.0;
        for (int f = e; f < hi; ++f)
            if (category[index[f]] == cls) total = __dadd_rn(total, weight[f]);
        if (total > most || (total == most && cls < best)) {
            most = total;
            best = cls;
        }
    }
    out[i] = best;
}

// The coupling's water budget (woof.hex.wrf_hydro_hex BUDGET_KEYS), per cell: the soil change after the cast,
// before it, the cast's envelope, the soil water (each sum_k dz_k ... A_c), the returned head and the covered
// runoff (m3), as six rows of ``out`` for one sum each.  A diagnostic: its sums are the device's, not the host's.
extern "C" __global__ void hydro_budget_cells(const double *dz, int layers, const double *area,
                                              const float *prior, const float *state, long long stride,
                                              const double *exact, const double *head, const float *infx,
                                              const float *sold, const double *covered, int cells,
                                              double *out) {
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= cells) return;
    double moved = 0.0, moved_exact = 0.0, envelope = 0.0, water = 0.0;
    for (int k = 0; k < layers; ++k) {
        double old = (double)prior[(long long)k * cells + c];
        double now = (double)state[(long long)k * stride + c];
        double e = exact[(long long)k * cells + c];
        moved += (now - old) * dz[k];
        moved_exact += (e - old) * dz[k];
        envelope += fabs(now - e) * dz[k];
        water += old * dz[k];
    }
    double a = area[c];
    out[c] = moved * a;
    out[cells + c] = moved_exact * a;
    out[2 * cells + c] = envelope * a;
    out[3 * cells + c] = water * a;
    out[4 * cells + c] = head[c] * 1e-3 * a;
    out[5 * cells + c] = ((double)infx[c] + (double)sold[c]) * covered[c] * 1e-3;
}

// ... and per grid cell: the routing's soil change, its head and the runoff it took (as cast, exact, and the
// cast's envelope), five rows of ``out``.
extern "C" __global__ void hydro_budget_grid(const double *dz, int layers, const double *area,
                                             const float *before, const float *after, const float *head,
                                             const float *infx, const float *sold, const double *infx_exact,
                                             const double *sold_exact, int grid, double *out) {
    int g = blockDim.x * blockIdx.x + threadIdx.x;
    if (g >= grid) return;
    double moved = 0.0;
    for (int k = 0; k < layers; ++k)
        moved += ((double)after[(long long)k * grid + g] - (double)before[(long long)k * grid + g]) * dz[k];
    double a = area[g], i32 = (double)infx[g], s32 = (double)sold[g];
    out[g] = moved * a;
    out[grid + g] = (double)head[g] * 1e-3 * a;
    out[2 * grid + g] = (i32 + s32) * 1e-3 * a;
    out[3 * grid + g] = (infx_exact[g] + sold_exact[g]) * 1e-3 * a;
    out[4 * grid + g] = (fabs(i32 - infx_exact[g]) + fabs(s32 - sold_exact[g])) * 1e-3 * a;
}
