"""Translator and facade constructs the MLX backend once got wrong.

* A kernel-local ``const dim3 blockIdx(...)`` (the ensemble batch adapter's
  virtual block coordinates, woof/ensemble/batch_kernel.py) hides CUDA's
  builtin.  The translator replaced every ``blockIdx`` with the physical
  threadgroup position, so each member past the first indexed past its
  extent and wrote nothing (its words stayed zero).
* ``ndarray.data.mem`` is CuPy's allocation (its ``ptr`` is the first byte
  of the backing, not of the view), and ``reshape`` of a strided view is a
  view where NumPy's layout rules allow it.  The batch adapter admits padded
  member slabs only on both.
* In an ``--ftz=false`` unit ``float`` is ``f32s``, but float vector
  components stay ``float``: ``out.x = f(...)`` did not compile
  (bandwidth_glue under GPUWM_WRF_EXACT=1), and arithmetic on components
  must keep subnormals as CUDA does.
* ``scalbnf`` and ``frexpf`` (the GF gamma normalization) had no Metal
  definition, so every GF unit under GPUWM_WRF_EXACT=1 failed to compile.
  ``frexpf`` works on the bits, so a subnormal operand is normalised as
  CUDA does, not flushed.
* An ``ElementwiseKernel`` output is read-write, as in CuPy.  On a
  non-contiguous view (a ring or interior slice of a 3-D field) the facade
  ran the operation on an empty dense stand-in, so ``if (x < t) x = 0``
  read zeros and wrote zeros everywhere (microphysics_zero_out).
* Metal's compare and fmax/fmin instructions treat a subnormal as zero, so
  under GPUWM_WRF_EXACT=1 ``cp.maximum(qr, 0)`` zeroed a subnormal rain
  mixing ratio that CUDA keeps (RRTM's rain path).  Float32 comparisons
  and min/max take the f32s kernels in that process.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                                reason="needs mlx (Apple silicon)")


@pytest.fixture
def cp():
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    import cupy
    return cupy


_SHADOWED = r'''
extern "C" __global__ void shadowed(float* out, int n) {
    const uint3 physical = blockIdx;
    const dim3 blockIdx(physical.x % 2u, physical.y, physical.z);
    const dim3 gridDim(2u, 1u, 1u);
    out = out + (physical.x / 2u) * n;
    int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at < n) out[at] = (float)(blockIdx.x * 1000u + gridDim.x * 10u + physical.x);
}
'''


def test_local_block_coordinates_shadow_the_builtins(cp):
    from woof.backend.translate import translate
    msl = translate(_SHADOWED, unit="shadowed").msl
    assert "int at = blockIdx.x * __woof_bdim.x" in msl
    out = cp.zeros((3, 8), dtype=cp.float32)
    kernel = cp.RawKernel(_SHADOWED, "shadowed")
    kernel((6,), (4,), (out, np.int32(8)))
    got = cp.asnumpy(out)
    want = np.array([[(b % 2) * 1000 + 20 + b for b in range(m * 2, m * 2 + 2) for _ in range(4)]
                     for m in range(3)], dtype=np.float32)
    assert got.tobytes() == want.tobytes()


def test_data_mem_is_the_allocation_and_strided_reshape_is_a_view(cp):
    backing = cp.zeros((4, 3 + 77 + 5), dtype=cp.uint32)
    view = backing.view(cp.float32)[:, 3:80].reshape((4, 7, 11))
    assert view.data.mem.ptr == backing.data.mem.ptr
    assert view.data.mem is backing.data.mem
    assert view.data.ptr == backing.data.ptr + 12
    assert view.strides == (85 * 4, 44, 4)
    view[...] = 1.5
    host = cp.asnumpy(backing).view(np.float32)
    assert (host[:, 3:80] == 1.5).all() and (cp.asnumpy(backing)[:, :3] == 0).all()
    assert (cp.asnumpy(backing)[:, 80:] == 0).all()
    one = backing[:1, 3:80].reshape((1, 7, 11))
    assert one.data.mem is backing.data.mem
    assert backing[:, 3:80].reshape(-1).data.mem is not backing.data.mem  # a copy, as NumPy


_VECTOR_F32S = r'''
__device__ float halve(float a) { return a * 0.5f; }
extern "C" __global__ void vector_halve(const float* in, float* out) {
    float4 a = ((const float4*)in)[0];
    float4 r;
    r.x = halve(a.x);
    r.y = a.y * 0.5f;
    r.z = a.z;
    r.z *= 0.5f;
    r.w = 0.0f;
    float2 q = make_float2(halve(a.w), a.w);
    r.w += q.x;
    ((float4*)out)[0] = r;
}
'''


def test_float_vector_components_keep_subnormals_in_an_ftz_false_unit(cp):
    tiny = np.float32(np.finfo(np.float32).tiny)  # halves to a subnormal
    module = cp.RawModule(code=_VECTOR_F32S, options=("--ftz=false",))
    src = cp.asarray(np.full(4, tiny, dtype=np.float32))
    out = cp.zeros(4, dtype=cp.float32)
    module.get_function("vector_halve")((1,), (1,), (src, out))
    want = np.full(4, tiny, dtype=np.float32) * np.float32(0.5)
    assert want[0] != 0
    assert cp.asnumpy(out).tobytes() == want.tobytes()


_POINTER_GLOBAL = r'''
__device__ unsigned int* bound_counts;
extern "C" __global__ void bind_counts(unsigned int* counts) {
    if (blockIdx.x == 0 && threadIdx.x == 0) bound_counts = counts;
}
extern "C" __global__ void count(int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&bound_counts[i % 4], 1u);
}
'''


def test_mutable_pointer_global_set_by_one_kernel_and_read_by_another(cp):
    """`__device__ T* p;` (rrtmg_lw's writer proof): Metal binds no buffer
    of pointers, so the global is an address word behind a pointer cell."""
    module = cp.RawModule(code=_POINTER_GLOBAL)
    counts = cp.zeros(4, dtype=cp.uint32)
    module.get_function("bind_counts")((1,), (1,), (counts,))
    module.get_function("count")((1,), (64,), (np.int32(50),))
    assert cp.asnumpy(counts).tolist() == [13, 13, 12, 12]


_REPOINTED = r'''
extern "C" __global__ void repointed(const float* dnw, float* out, int n) {
    __shared__ float cache[4];
    for (unsigned int at = threadIdx.x; at < 4u; at += blockDim.x) cache[at] = dnw[at] * 2.0f;
    __syncthreads();
    dnw = cache;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = dnw[i % 4] + dnw[0];
}
'''


def test_kernel_pointer_parameter_repointed_at_shared_memory(cp):
    """A layout trial stages coefficients in shared memory and points the
    kernel's const parameter at them: a variant pointer on Metal."""
    src = cp.asarray(np.array([1, 2, 3, 4], dtype=np.float32))
    out = cp.zeros(8, dtype=cp.float32)
    cp.RawKernel(_REPOINTED, "repointed")((1,), (8,), (src, out, np.int32(8)))
    assert cp.asnumpy(out).tolist() == [4, 6, 8, 10, 4, 6, 8, 10]


_FREXP_SCALBN = r'''
extern "C" __global__ void frexp_scalbn(const float* in, float* mant, int* expo, float* back, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int e;
    float m = frexpf(in[i], &e);
    mant[i] = m;
    expo[i] = e;
    back[i] = scalbnf(m, e);
}
'''


@pytest.mark.parametrize("options", [(), ("--ftz=false",)], ids=["default", "ftz-false"])
def test_frexpf_and_scalbnf_are_exact_on_normals_and_subnormals(cp, options):
    info = np.finfo(np.float32)
    values = np.array([1.0, -3.75, 0.5, 1e30, -1e-30, info.tiny, info.max,
                       info.smallest_subnormal, -info.smallest_subnormal * 37,
                       info.tiny * np.float32(0.75), 0.0, -0.0, np.inf, -np.inf],
                      dtype=np.float32)
    n = len(values)
    module = cp.RawModule(code=_FREXP_SCALBN, options=options)
    mant = cp.zeros(n, dtype=cp.float32)
    expo = cp.zeros(n, dtype=cp.int32)
    back = cp.zeros(n, dtype=cp.float32)
    module.get_function("frexp_scalbn")((1,), (32,), (cp.asarray(values), mant, expo, back, np.int32(n)))
    want_m, want_e = np.frexp(values)
    assert cp.asnumpy(mant).tobytes() == want_m.astype(np.float32).tobytes()
    finite = np.isfinite(values)
    assert cp.asnumpy(expo)[finite].tolist() == want_e[finite].tolist()
    if options:  # ftz=false keeps the subnormal results scalbnf rebuilds
        assert cp.asnumpy(back).tobytes() == values.tobytes()


def test_elementwise_output_on_a_strided_view_is_read_write(cp):
    zero_below = cp.ElementwiseKernel("float32 t", "float32 x",
                                      "if (x < t) x = 0.0f;", "zero_below_view")
    host = np.arange(-30, 30, dtype=np.float32).reshape(3, 4, 5) / 7
    field = cp.asarray(host)
    zero_below(np.float32(0.5), field[:, 1:3, 1:4])   # interior: not contiguous
    zero_below(np.float32(0.5), field[:, 0, :])        # one ring row
    want = host.copy()
    for view in (want[:, 1:3, 1:4], want[:, 0, :]):
        view[view < np.float32(0.5)] = 0
    assert cp.asnumpy(field).tobytes() == want.tobytes()


@pytest.mark.parametrize("name", ["maximum", "minimum", "fmax", "fmin", "greater",
                                  "greater_equal", "less", "less_equal", "equal",
                                  "not_equal"])
def test_strict_compare_and_minmax_keep_subnormals(cp, monkeypatch, name):
    monkeypatch.setenv("GPUWM_WRF_EXACT", "1")
    info = np.finfo(np.float32)
    x = np.array([info.smallest_subnormal, -info.smallest_subnormal, 3e-39, -0.0,
                  0.0, 2.0, np.nan, -np.inf], dtype=np.float32)
    zero = np.float32(0)
    got = cp.asnumpy(getattr(cp, name)(cp.asarray(x), zero))
    assert got.tobytes() == np.asarray(getattr(np, name)(x, zero)).tobytes()
