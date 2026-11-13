"""Translator and facade constructs the hex MPAS v8.4.2 suite route (R2) needed on Metal.

Each was met running the default R2 suite (convection_permitting: RRTMG,
MYNN, Noah, Grell-Freitas, aerosol-aware Thompson) on a Mac for the first
time; each test is the construct alone, run on the device:

* an initialised ``__constant__`` table the host reads through
  ``get_global`` (RRTMG LW's ``mpas_static_*`` DATA tables): the module had
  no such global;
* a reference member bound to a device object (Noah's shared-tables
  ``NoahThread``: ``const NoahTables& g``) was always ``thread``, and so was
  every pointer read through it;
* a space-generic local declared without an initialiser and given its data
  by ``a.data = p`` (MYNN's ``MynnArray``) or by ``z = (Alias) p`` (GF's
  ``GFP``), and one a plain struct's method makes (Thompson's ``TcWs::ff``);
* a member operator call (``text = "x"``): its object was taken for the
  first parameter;
* a function-scope ``static const`` array of an anonymous union;
* a comparison of pointers into two address spaces (GF's ``gf_memmove``);
* the facade: ``RawModule(path=...)`` over a compile_using_nvrtc image,
  ``ravel(order="F")``, ``int(get_global_var(...))`` and
  ``regsPerMultiprocessor``.
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


def _run(cp, src, kernel, n, *extra):
    out = cp.zeros(n, dtype=cp.float32)
    module = cp.RawModule(code=src)
    module.get_function(kernel)((1,), (n,), (out, *extra))
    return module, cp.asnumpy(out)


def test_initialised_constant_tables_are_host_readable(cp):
    src = r'''
    __constant__ float tab[4] = {1.5f, -2.0f, 3.25f, 0.1f};
    __device__ const int itab[3] = {7, 8, 9};
    __constant__ float filled[2];
    extern "C" __global__ void k(float* out) {
        int i = threadIdx.x; if (i < 4) out[i] = tab[i] + itab[i % 3] + filled[0];
    }'''
    module = cp.RawModule(code=src)
    tab = cp.ndarray((4,), dtype=cp.float32, memptr=module.get_global("tab"))
    itab = cp.ndarray((3,), dtype=cp.int32, memptr=module.get_global("itab"))
    np.testing.assert_array_equal(cp.asnumpy(tab), np.float32([1.5, -2.0, 3.25, 0.1]))
    np.testing.assert_array_equal(cp.asnumpy(itab), [7, 8, 9])
    cp.ndarray((2,), dtype=cp.float32, memptr=module.get_global("filled"))[:] = cp.asarray(
        np.float32([10.0, 0.0]))
    out = cp.zeros(4, dtype=cp.float32)
    module.get_function("k")((1,), (4,), (out,))
    expected = np.float32([1.5, -2, 3.25, 0.1]) + np.float32([7, 8, 9, 7]) + np.float32(10)
    np.testing.assert_array_equal(cp.asnumpy(out), expected)


def test_a_reference_member_bound_to_device_data_reads_it(cp):
    src = r'''
    struct Tables { float scale; char name[8]; };
    struct Thread { const Tables& g; int error;
        __device__ explicit Thread(const Tables& t) : g(t), error(0) {} };
    __device__ float first_letter(const char* text) { return (float)text[0]; }
    __device__ float use(Thread& t) { const char* name = t.g.name; return t.g.scale * first_letter(name); }
    extern "C" __global__ void k(float* out, const Tables* ts) {
        Thread t(*ts); int i = threadIdx.x; if (i < 2) out[i] = use(t) + i;
    }'''
    raw = np.zeros(12, np.uint8)
    raw[:4] = np.frombuffer(np.float32(2.0).tobytes(), np.uint8)
    raw[4] = ord("A")
    tables = cp.asarray(raw)
    _, out = _run(cp, src, "k", 2, tables)
    np.testing.assert_array_equal(out, np.float32([130.0, 131.0]))


GENERIC = r'''
template<class T> struct Col { T* data; int n;
    __device__ Col() = default;
    __device__ Col(T* q) : data(q), n(0) {}
    __device__ explicit Col(void* q) : data((T*)q), n(0) {}
    __device__ T& operator()(int i) const { return data[i]; } };
struct Ws { float* f; __device__ Col<float> ff(int i) const { return Col<float>(f + i * 4); } };
typedef Col<float> ColF;
__device__ float total(Col<float> c) { float s = 0; for (int i = 0; i < c.n; i++) s += c(i); return s; }
'''


def test_space_generic_locals_without_initialisers(cp):
    src = GENERIC + r'''
    extern "C" __global__ void k(float* out, float* buf) {
        float local[4] = {1, 2, 3, 4};
        Col<float> t; t.data = local; t.n = 4;          // thread data
        Col<float> d; d.data = buf; d.n = 4;            // device data, given by assignment
        ColF z; z = (ColF) (void*) buf;  z.n = 4;       // through an alias and a cast
        ColF y = ColF(buf + 4); y.n = 4;                // a functional cast of an offset pointer
        ColF w = (ColF) y;                              // an identity copy
        Ws ws{buf};
        Col<float> m = ws.ff(1); m.n = 4;               // made by a plain struct's method
        if (threadIdx.x == 0) {
            out[0] = total(t); out[1] = total(d); out[2] = total(z); out[3] = total(w);
            out[4] = total(m);
        }
    }'''
    buf = cp.asarray(np.arange(8, dtype=np.float32))
    _, out = _run(cp, src, "k", 5, buf)
    np.testing.assert_array_equal(out, np.float32([10, 6, 6, 22, 22]))


def test_a_member_operator_call_maps_its_parameters_past_its_object(cp):
    src = r'''
    struct Text { const char* text = "";
        __device__ Text& operator=(const char* s) { text = s; return *this; } };
    extern "C" __global__ void k(float* out) {
        Text t; t = "b";
        if (threadIdx.x == 0) out[0] = (float)t.text[0];
    }'''
    _, out = _run(cp, src, "k", 1)
    assert out[0] == ord("b")


def test_a_function_scope_static_union_table(cp):
    src = r'''
    __device__ float pick(int i) {
        static const union { float f; unsigned int u; } table[] = {{1.5f}, {2.5f}, {4.0f}};
        return table[i].f;
    }
    extern "C" __global__ void k(float* out) { int i = threadIdx.x; if (i < 3) out[i] = pick(i); }'''
    _, out = _run(cp, src, "k", 3)
    np.testing.assert_array_equal(out, np.float32([1.5, 2.5, 4.0]))


def test_pointers_into_two_address_spaces_compare(cp):
    src = r'''
    __constant__ float source[3] = {3.f, 4.f, 5.f};
    __device__ void move(void* dst, const void* src, unsigned long count) {
        char* d = (char*)dst; const char* s = (const char*)src;
        if (d < s) { for (unsigned long i = 0; i < count; i++) d[i] = s[i]; }
        else { for (unsigned long i = count; i > 0; i--) d[i - 1] = s[i - 1]; }
    }
    extern "C" __global__ void k(float* out) { if (threadIdx.x == 0) move(out, source, 3 * sizeof(float)); }'''
    _, out = _run(cp, src, "k", 3)
    np.testing.assert_array_equal(out, np.float32([3, 4, 5]))


def test_facade_answers_the_direct_compilers(cp, tmp_path):
    from cupy.cuda import compiler
    src = r'extern "C" __global__ void k(float* out) { out[threadIdx.x] = 2.0f * threadIdx.x; }'
    image, _ = compiler.compile_using_nvrtc(src, options=("--std=c++17",), arch="120", filename="k.cu")
    path = tmp_path / "k.cubin"
    path.write_bytes(image)
    module = cp.RawModule(path=str(path))
    out = cp.zeros(4, dtype=cp.float32)
    module.get_function("k")((1,), (4,), (out,))
    np.testing.assert_array_equal(cp.asnumpy(out), np.float32([0, 2, 4, 6]))
    a = cp.asarray(np.arange(6, dtype=np.float32).reshape(2, 3))
    np.testing.assert_array_equal(cp.asnumpy(cp.ravel(a, order="F")), np.arange(6).reshape(2, 3).ravel("F"))
    counter = cp.RawModule(code=r'''
        __device__ unsigned int next_column;
        extern "C" __global__ void k(float* out) { out[0] = (float)atomicAdd(&next_column, 1u); }''')
    address = int(counter.get_global_var("next_column"))
    cp.cuda.runtime.memsetAsync(address, 0, 4, 0)
    assert int(cp.cuda.runtime.getDeviceProperties(0)["regsPerMultiprocessor"]) > 0
