"""Constructs 1.1.6's units brought to the Metal translator, on any GPU.

Each kernel runs on the active backend (natively on CUDA, translated on
MLX) and is held to a Python transcription of the same control flow.

* Fortran's crossing gotos (deng, kf_old: ``IF (..) GOTO 255`` /
  ``175 ...`` / ``GOTO 255`` / ``255 ...`` / ``GOTO 175``), a backward goto
  from inside a loop (ntu, gbmpbl), a break or continue among a goto's
  statements (ssib) and stacked labels (``L400: L401:``, ssib and
  gfdleta_co2_init) were refused; they are a dispatch loop
  (``switch`` on the label) now.
* A goto from loops in an if's then arm to a label in its else arm
  (Noah-MP's generated dynamic-VIC infiltration, label_1003) entered a
  block from outside; it is a re-entry loop that routes each if to the arm
  holding the label.
* A local array of pointers filled in a loop and handed to a device
  function (nwp_windows' ``float* out[12]``) was declared ``thread``.
* A class template's ``operator()`` returning a reference into a buffer
  (deng's ``DengArray<float>``) bound to a ``float&`` parameter only
  through a ``thread`` overload.
* A module global's initialiser literal was respelled inside the
  declaration the global's buffer replaces (gfdleta, cam3_emissivity).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]

N = 160


@pytest.fixture
def cp():
    import cupy
    return cupy


def _launch(cp, src, name, *args):
    kernel = cp.RawKernel(src, name)
    kernel(((N + 63) // 64,), (64,), args)


_CROSSING = r'''
extern "C" __global__ void crossing(const int* in, int* out, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    int x = in[c], count = 0, acc = 0;
    if (x > 50) goto L255;
L175:
    count = count + 1;
    if (x % 3 == 0) goto L176;
    acc += 7;
    goto L255;
L176:
    acc += x;
    for (int k = 0; k < 4; ++k) {
        if (acc > 100 + k) goto L265;
        acc += k;
    }
L255:
    acc += 1000;
    if (count < 3 && x < 80) { x = x + 5; goto L175; }
L265:
    out[c] = acc * 16 + count;
}
'''


def _crossing(x):
    count = acc = 0
    label = "L255" if x > 50 else "L175"
    while True:
        if label == "L175":
            count += 1
            if x % 3 == 0:
                label = "L176"
                continue
            acc += 7
            label = "L255"
            continue
        if label == "L176":
            acc += x
            label = "L255"
            for k in range(4):
                if acc > 100 + k:
                    label = "L265"
                    break
                acc += k
            continue
        if label == "L255":
            acc += 1000
            if count < 3 and x < 80:
                x += 5
                label = "L175"
                continue
            label = "L265"
            continue
        return acc * 16 + count


def test_crossing_gotos(cp):
    x = np.arange(N, dtype=np.int32) % 97
    out = cp.zeros(N, dtype=cp.int32)
    _launch(cp, _CROSSING, "crossing", cp.asarray(x), out, np.int32(N))
    assert cp.asnumpy(out).tolist() == [_crossing(int(v)) for v in x]


_STACKED = r'''
extern "C" __global__ void stacked(const int* in, int* out, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    int x = in[c], acc = 0;
    for (int it = 0; it < 6; ++it) {
        if (x & 1) goto A;
    B:
        acc += 3;
        if (acc > 40) break;
        if (x & 2) { x >>= 1; continue; }
        goto C;
    A:
    D:
        acc += 5;
        x += 1;
        if (x % 5 == 0) goto B;
    C:
        acc += it;
        if (acc % 7 == 0) goto D;
        x = x * 3 + 1;
    }
    out[c] = acc;
}
'''


def _stacked(x):
    acc = 0
    for it in range(6):
        label = "A" if x & 1 else "B"
        leave = False
        while label:
            if label == "B":
                acc += 3
                if acc > 40:
                    leave = True
                    break
                if x & 2:
                    x >>= 1
                    label = None
                    break
                label = "C"
            elif label == "A":
                acc += 5
                x += 1
                label = "B" if x % 5 == 0 else "C"
            elif label == "C":
                acc += it
                if acc % 7 == 0:
                    label = "A"
                    continue
                x = x * 3 + 1
                label = None
        if leave:
            break
    return acc


def test_stacked_labels_and_a_break_and_continue_among_crossing_gotos(cp):
    x = (np.arange(N, dtype=np.int32) * 7) % 53
    out = cp.zeros(N, dtype=cp.int32)
    _launch(cp, _STACKED, "stacked", cp.asarray(x), out, np.int32(N))
    assert cp.asnumpy(out).tolist() == [_stacked(int(v)) for v in x]


_RETRY = r'''
extern "C" __global__ void retry(const int* in, int* out, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    int x = in[c], tries = 0, acc = 0;
R:
    tries++;
    for (int k = 0; k < 5; ++k) {
        acc += x + k;
        if (acc % 11 == 0 && tries < 4) { x += 1; goto R; }
    }
    out[c] = acc * 8 + tries;
}
'''


def _retry(x):
    tries = acc = 0
    while True:
        tries += 1
        again = False
        for k in range(5):
            acc += x + k
            if acc % 11 == 0 and tries < 4:
                x += 1
                again = True
                break
        if not again:
            return acc * 8 + tries


def test_backward_goto_from_inside_a_loop(cp):
    x = np.arange(N, dtype=np.int32) % 41
    out = cp.zeros(N, dtype=cp.int32)
    _launch(cp, _RETRY, "retry", cp.asarray(x), out, np.int32(N))
    assert cp.asnumpy(out).tolist() == [_retry(int(v)) for v in x]


_TABLES = r'''
__device__ void fill(float* const* out, int c, float v) {
    for (int f = 0; f < 3; ++f)
        if (out[f]) out[f][c] = v + (float)f;
}
extern "C" __global__ void tables(float* storage, float* extra, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    float* out[3];
    for (int f = 0; f < 2; ++f) out[f] = storage + (size_t)f * n;
    out[2] = (float*)0;
    fill(out, c, (float)c);
    float* more[3] = {extra, (float*)0, storage + (size_t)2 * n};
    fill(more, c, 100.0f);
}
'''


def test_a_local_array_of_device_pointers_reaches_a_device_function(cp):
    storage = cp.zeros(3 * N, dtype=cp.float32)
    extra = cp.zeros(N, dtype=cp.float32)
    _launch(cp, _TABLES, "tables", storage, extra, np.int32(N))
    c = np.arange(N, dtype=np.float32)
    want = np.concatenate([c, c + 1.0, np.full(N, 102.0, np.float32)])
    assert cp.asnumpy(storage).tobytes() == want.astype(np.float32).tobytes()
    assert cp.asnumpy(extra).tobytes() == np.full(N, 100.0, np.float32).tobytes()


_REFS = r'''
template<class T> struct Arr {
    T* data; int stride;
    __device__ T& operator()(int i) const { return data[i * stride]; }
};
__device__ void bump(float& v, float by) { v = v + by; }
extern "C" __global__ void refs(float* d, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    Arr<float> a{d + c, n};
    float t = 1.0f;
    bump(a(0), 2.0f);
    bump(t, 3.0f);
    a(1) = a(1) + t;
}
'''


def test_a_class_template_reference_into_a_buffer_binds_a_reference_parameter(cp):
    d = cp.asarray(np.arange(2 * N, dtype=np.float32))
    _launch(cp, _REFS, "refs", d, np.int32(N))
    base = np.arange(2 * N, dtype=np.float32)
    want = np.concatenate([base[:N] + 2.0, base[N:] + 4.0])
    assert cp.asnumpy(d).tobytes() == want.tobytes()


_GLOBAL = r'''
__device__ double scale = 10.;
extern "C" __global__ void scaled(float* out, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    out[c] = (float)(scale * (double)c);
}
'''


def test_a_module_global_with_a_binary64_literal_initialiser(cp):
    out = cp.zeros(N, dtype=cp.float32)
    _launch(cp, _GLOBAL, "scaled", out, np.int32(N))
    assert cp.asnumpy(out).tobytes() == (np.arange(N, dtype=np.float32) * 10).tobytes()


_SWITCHED = r'''
extern "C" __global__ void switched(const int* in, int* out, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    int x = in[c], acc = 0;
    for (int it = 0; it < 5; ++it) {
        if (x & 1) goto A;
    B:
        switch (x % 3) {
        case 0: acc += 2; continue;
        case 1:
            for (int k = 0; k < 3; ++k) { if (k == x % 2) continue; acc += k; }
            break;
        default: acc += 7;
        }
        x += 3;
        goto C;
    A:
        acc += 5;
        x += 1;
        if (x % 4 == 0) goto B;
    C:
        acc += it;
        if (acc % 9 == 0) goto A;
        x = x * 5 + 1;
    }
    out[c] = acc;
}
'''


def _switched(x):
    acc = 0
    for it in range(5):
        label = "A" if x & 1 else "B"
        while label:
            if label == "B":
                r = x % 3
                if r == 0:
                    acc += 2
                    label = None
                    break
                if r == 1:
                    for k in range(3):
                        if k == x % 2:
                            continue
                        acc += k
                else:
                    acc += 7
                x += 3
                label = "C"
            elif label == "A":
                acc += 5
                x += 1
                label = "B" if x % 4 == 0 else "C"
            else:
                acc += it
                if acc % 9 == 0:
                    label = "A"
                    continue
                x = x * 5 + 1
                label = None
    return acc


def test_a_switch_among_dispatched_gotos_keeps_its_breaks_and_continues(cp):
    x = (np.arange(N, dtype=np.int32) * 5) % 47
    out = cp.zeros(N, dtype=cp.int32)
    _launch(cp, _SWITCHED, "switched", cp.asarray(x), out, np.int32(N))
    assert cp.asnumpy(out).tolist() == [_switched(int(v)) for v in x]


_FORWARDED = r'''
__device__ void put(float* const* out, int c, float v) { out[1][c] = v; }
__device__ void relay(float* const* out, int c) { put(out, c, (float)c * 2.0f); }
__device__ void relay2(float* const* out, int c) { relay(out, c); }
extern "C" __global__ void forwarded(float* a, float* b, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    float* out[2] = {a, b};
    relay2(out, c);
}
'''


def test_a_local_pointer_array_passed_on_through_two_device_functions(cp):
    a, b = cp.zeros(N, dtype=cp.float32), cp.zeros(N, dtype=cp.float32)
    _launch(cp, _FORWARDED, "forwarded", a, b, np.int32(N))
    assert cp.asnumpy(b).tobytes() == (np.arange(N, dtype=np.float32) * 2).tobytes()
    assert not cp.asnumpy(a).any()


_ENTER = r'''
extern "C" __global__ void enter(const int* in, int* out, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    int x = in[c], acc = 0, y = 0;
    if (x > 40) {
        acc = 1;
        goto done;
    } else {
        if (x % 2 == 0) {
            for (int k = 0; k < 6; ++k) {
                y += k;
                if (y > x) { goto label_1003; }
            }
        } else {
            acc += 3;
            for (int k = 0; k < 4; ++k) {
                y += 2;
                if (y >= x) goto label_1003;
            }
            y = -1;
            label_1003: ;
            acc += 100 + y;
        }
    }
    acc += 7;
done:
    out[c] = acc * 1000 + y;
}
'''


def _enter(x):
    acc = y = 0
    if x > 40:
        return 1 * 1000 + y
    entered = False
    if x % 2 == 0:
        for k in range(6):
            y += k
            if y > x:
                entered = True
                break
        if not entered:
            return (acc + 7) * 1000 + y
    else:
        acc += 3
        for k in range(4):
            y += 2
            if y >= x:
                entered = True
                break
        if not entered:
            y = -1
    acc += 100 + y
    return (acc + 7) * 1000 + y


def test_a_goto_into_the_else_arm_from_loops_in_the_then_arm(cp):
    x = (np.arange(N, dtype=np.int32) * 3) % 53
    out = cp.zeros(N, dtype=cp.int32)
    _launch(cp, _ENTER, "enter", cp.asarray(x), out, np.int32(N))
    assert cp.asnumpy(out).tolist() == [_enter(int(v)) for v in x]


_ENTER_LAST = r'''
extern "C" __global__ void enter_last(const int* in, int* out, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    int x = in[c], acc = 0, y = 0;
    if (x % 3 == 0) {
        for (int k = 0; k < 5; ++k) { y += k + 1; if (y > x) goto L; }
        acc = 50;
    } else {
        for (int k = 0; k < 4; ++k) { y += 3; if (y >= x) goto L; }
    L:
        acc += 10 + y;
    }
    out[c] = acc * 1000 + y;
}
'''


def _enter_last(x):
    acc = y = 0
    if x % 3 == 0:
        for k in range(5):
            y += k + 1
            if y > x:
                return (acc + 10 + y) * 1000 + y
        acc = 50
        return acc * 1000 + y
    for k in range(4):
        y += 3
        if y >= x:
            break
    return (acc + 10 + y) * 1000 + y


def test_a_goto_into_a_label_right_after_the_loop_that_jumps(cp):
    """The loop holding a jump is the last statement before the label: its
    `if (flag) continue;` must stay inside the skip guard (else the
    re-entry loop never ends)."""
    x = (np.arange(N, dtype=np.int32) * 7) % 31
    out = cp.zeros(N, dtype=cp.int32)
    _launch(cp, _ENTER_LAST, "enter_last", cp.asarray(x), out, np.int32(N))
    assert cp.asnumpy(out).tolist() == [_enter_last(int(v)) for v in x]


_DIRECT = r'''
extern "C" __global__ void direct(const int* in, int* out, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    int x = in[c], acc = 0;
    for (int k = 0; k < 6; ++k) {
        if ((x + k) % 3 == 0) goto L;
        acc += 2;
        if (k % 2) {
            acc += 5;
        }
        goto L;
        acc += 1000;
    L:
        acc += k * x;
    }
    out[c] = acc;
}
'''


def test_an_unconditional_goto_that_is_its_blocks_statement_inside_a_loop(cp):
    """The wrapper's flag is the loop body's: no check after the loop (it
    named the flag out of its scope; gfdleta_co2_init, ssib)."""
    x = np.arange(N, dtype=np.int32) % 29
    want = []
    for v in x:
        acc = 0
        for k in range(6):
            if (int(v) + k) % 3:
                acc += 2 + (5 if k % 2 else 0)
            acc += k * int(v)
        want.append(acc)
    out = cp.zeros(N, dtype=cp.int32)
    _launch(cp, _DIRECT, "direct", cp.asarray(x), out, np.int32(N))
    assert cp.asnumpy(out).tolist() == want
