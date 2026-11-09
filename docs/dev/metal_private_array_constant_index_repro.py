"""Standalone repro of an Apple Metal shader-compiler bug (no woof code).

A function-local array of more than 32 elements, filled and read by loops of
runtime trip count, reads as 0 at a constant index afterwards.  Observed on
macOS 26.5.2 (25F84), Apple M3 Max, through MLX's mx.fast.metal_kernel:

    float r[64];
    for (int k = 0; k < nz; ++k) r[k] = rho_in[k * n + c];
    float t = 0.0f;
    for (int k = 0; k < nz; ++k) t += r[k];
    q[c] = t;        // right
    o[c] = r[0];     // 0, expected rho_in[c]

Arrays of 32 elements or fewer, and runtime indices (r[nz - 1]), read right;
`volatile` on the array and `#pragma clang loop unroll(disable)` do not help;
an index the compiler cannot fold (r[0 + volatile_zero]) does.  woof works
around it in its CUDA-to-Metal translator (woof.backend.translate,
opaque_constant_index).

Run: python docs/dev/metal_private_array_constant_index_repro.py
"""

import mlx.core as mx
import numpy as np

N, NZ = 6, 12


def run(size, read):
    source = f"""
    uint c = thread_position_in_grid.x;
    if (c >= (uint)n) return;
    float r[{size}];
    volatile int zero = 0;
    for (int k = 0; k < nz; ++k) r[k] = rho_in[k * n + c];
    float t = 0.0f;
    for (int k = 0; k < nz; ++k) t += r[k];
    q[c] = t;
    o[c] = {read};
"""
    kernel = mx.fast.metal_kernel(name=f"repro_{size}_{abs(hash(read))}", input_names=["rho_in", "nz", "n"],
                                  output_names=["q", "o"], source=source)
    rho = mx.array((1.2 + 0.01 * np.arange(NZ * N)).astype(np.float32))
    q, o = kernel(inputs=[rho, NZ, N], grid=(64, 1, 1), threadgroup=(64, 1, 1),
                  output_shapes=[(N,), (N,)], output_dtypes=[mx.float32, mx.float32])
    return np.array(o)


if __name__ == "__main__":
    expected = (1.2 + 0.01 * np.arange(N)).astype(np.float32)
    for size, read in ((32, "r[0]"), (33, "r[0]"), (64, "r[0]"), (64, "r[nz - 1] * 0.0f + r[0 + zero]")):
        got = run(size, read)
        verdict = "ok" if np.allclose(got, expected) else "WRONG"
        print(f"float r[{size}], o = {read:32s} -> {got[:3]}  {verdict}")
