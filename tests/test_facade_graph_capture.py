"""A captured CUDA graph replays its launches, on every GPU backend.

woof captures a step once and launches the graph every step.  The MLX
facade used to run the work at capture and replay nothing, so a captured
step ran once and the forecast stood still.  It records translated kernel
launches (arrays by reference, by-value arguments by value), host functions
and event records, and replays them in order; anything else it cannot
replay refuses the capture by name.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]

_SOURCE = r'''
extern "C" __global__ void bump(float* x, float by, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 1.0009765625f + by;
}
extern "C" __global__ void copy_scaled(const float* x, float* y, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * s;
}
'''
N = 300


@pytest.fixture
def cp():
    import cupy
    return cupy


def _functions(cp):
    module = cp.RawModule(code=_SOURCE)
    return module.get_function("bump"), module.get_function("copy_scaled")


def _step(bump, copy_scaled, x, y, by):
    grid, block = ((N + 127) // 128,), (128,)
    bump(grid, block, (x, np.float32(by), np.int32(N)))
    copy_scaled(grid, block, (x, y, np.float32(0.5), np.int32(N)))


def _capture(cp, body):
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        try:
            body()
        except BaseException:
            try:
                stream.end_capture()
            except Exception:       # the body's error is the one to report
                pass
            raise
        graph = stream.end_capture()
    return stream, graph


def test_a_replayed_graph_matches_the_launches_run_directly(cp):
    bump, copy_scaled = _functions(cp)
    start = np.linspace(-3.0, 3.0, N, dtype=np.float32)
    direct_x, direct_y = cp.asarray(start), cp.zeros(N, dtype=cp.float32)
    for _ in range(7):
        _step(bump, copy_scaled, direct_x, direct_y, 0.25)
    x, y = cp.asarray(start), cp.zeros(N, dtype=cp.float32)
    cp.cuda.runtime.deviceSynchronize()
    stream, graph = _capture(cp, lambda: _step(bump, copy_scaled, x, y, 0.25))
    # capture records the work; it does not run it
    assert cp.asnumpy(x).tobytes() == start.tobytes()
    for _ in range(7):
        with stream:
            graph.launch(stream)
    stream.synchronize()
    assert cp.asnumpy(x).tobytes() == cp.asnumpy(direct_x).tobytes()
    assert cp.asnumpy(y).tobytes() == cp.asnumpy(direct_y).tobytes()


def test_by_value_arguments_are_fixed_at_capture(cp):
    bump, copy_scaled = _functions(cp)
    x, y = cp.zeros(N, dtype=cp.float32), cp.zeros(N, dtype=cp.float32)
    by = np.zeros(1, dtype=np.float32)
    by[0] = 1.0
    stream, graph = _capture(cp, lambda: _step(bump, copy_scaled, x, y, by[0]))
    by[0] = 100.0
    with stream:
        graph.launch(stream)
        graph.launch(stream)
    stream.synchronize()
    want = np.float32(np.float32(0.0) * np.float32(1.0009765625) + np.float32(1.0))
    want = np.float32(want * np.float32(1.0009765625) + np.float32(1.0))
    assert cp.asnumpy(x).tobytes() == np.full(N, want, np.float32).tobytes()


def _mlx(cp):
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("the facade's refusals: WOOF_BACKEND=mlx")


def test_host_functions_and_launches_replay_in_order(cp):
    _mlx(cp)
    bump, copy_scaled = _functions(cp)
    x, y = cp.zeros(N, dtype=cp.float32), cp.zeros(N, dtype=cp.float32)
    seen = []
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        assert stream.is_capturing()
        _step(bump, copy_scaled, x, y, 2.0)
        stream.launch_host_func(lambda arg: seen.append(float(cp.asnumpy(x)[0])), None)
        graph = stream.end_capture()
    assert seen == [] and not stream.is_capturing()
    graph.launch()
    graph.launch()
    assert seen == [2.0, np.float32(np.float32(2.0) * np.float32(1.0009765625) + np.float32(2.0))]


@pytest.mark.parametrize("operation", ["arithmetic", "write", "read", "synchronize"])
def test_an_operation_the_facade_cannot_replay_refuses_the_capture(cp, operation):
    _mlx(cp)
    from woof.backend.facade._capture import GraphCaptureRefused
    bump, copy_scaled = _functions(cp)
    x, y = cp.zeros(N, dtype=cp.float32), cp.zeros(N, dtype=cp.float32)

    def body():
        _step(bump, copy_scaled, x, y, 1.0)
        if operation == "arithmetic":
            y + 1.0
        elif operation == "write":
            y[...] = 3.0
        elif operation == "read":
            cp.asnumpy(y)
        else:
            cp.cuda.runtime.deviceSynchronize()

    with pytest.raises(GraphCaptureRefused, match="cannot be replayed"):
        _capture(cp, body)
    # the refusal ends the capture: the next launch runs
    _step(bump, copy_scaled, x, y, 1.0)
    assert cp.asnumpy(x)[0] != 0.0


def test_a_capture_whose_body_raises_does_not_swallow_later_launches(cp):
    _mlx(cp)
    bump, copy_scaled = _functions(cp)
    x, y = cp.zeros(N, dtype=cp.float32), cp.zeros(N, dtype=cp.float32)
    stream = cp.cuda.Stream(non_blocking=True)
    with pytest.raises(ZeroDivisionError):
        with stream:
            stream.begin_capture()
            _step(bump, copy_scaled, x, y, 1.0)
            1 / 0
    # outside the capturing stream: the launch runs, and a new capture works
    _step(bump, copy_scaled, x, y, 1.0)
    assert cp.asnumpy(x)[0] == 1.0
    stream2, graph = _capture(cp, lambda: _step(bump, copy_scaled, x, y, 1.0))
    with stream2:
        graph.launch(stream2)
    stream2.synchronize()
    assert cp.asnumpy(x)[0] == np.float32(np.float32(1.0) * np.float32(1.0009765625) + np.float32(1.0))
