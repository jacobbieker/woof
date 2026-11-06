"""CuPy surface the hex lane reaches through woof's facade on Metal.

* The current stream is per thread, as CuPy keeps it: the hex device upload
  builds fields on worker threads, each inside its own ``with Stream()``,
  and the caller's stream must be the one it entered when they finish.
* ``cuOccupancyMaxActiveBlocksPerMultiprocessor`` sizes column tiles
  (Thompson's start-frame radar, TEMPO, MYNN).
* ``from cupy.cuda.function import Module``, as the hex init kernels and
  the direct-NVRTC units import it.
"""

import importlib.util
import threading

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


def test_a_worker_threads_stream_does_not_leak_into_its_caller(cp):
    seen = {}
    entered = threading.Barrier(2)

    def worker():
        with cp.cuda.Stream(non_blocking=True) as mine:
            entered.wait(timeout=10)
            seen["worker"] = cp.cuda.get_current_stream().ptr == mine.ptr
            entered.wait(timeout=10)

    with cp.cuda.Stream(non_blocking=True) as caller:
        thread = threading.Thread(target=worker)
        thread.start()
        entered.wait(timeout=10)
        assert cp.cuda.get_current_stream().ptr == caller.ptr
        entered.wait(timeout=10)
        thread.join()
        assert cp.cuda.get_current_stream().ptr == caller.ptr
    assert seen == {"worker": True}
    assert cp.cuda.get_current_stream().ptr == cp.cuda.Stream.null.ptr


def test_nested_streams_restore_in_order(cp):
    outer, inner = cp.cuda.Stream(), cp.cuda.Stream()
    with outer:
        with inner:
            assert cp.cuda.get_current_stream() is inner
        assert cp.cuda.get_current_stream() is outer


def test_occupancy_fills_one_cores_thread_budget(cp):
    from cupy.cuda import driver, runtime
    per_core = runtime.deviceGetAttribute(runtime.cudaDevAttrMaxThreadsPerMultiProcessor, 0)
    assert driver.occupancyMaxActiveBlocksPerMultiprocessor(0, 32, 0) == per_core // 32
    assert driver.occupancyMaxActiveBlocksPerMultiprocessor(0, 4096, 0) == 1


def test_the_function_module_imports_by_its_cupy_name(cp):
    from cupy.cuda.function import Module
    assert Module is cp.cuda.function.Module
