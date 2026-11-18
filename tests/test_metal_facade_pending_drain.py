"""The facade's pending-storage drain keeps every storage appended while it runs.

Storages join ``_array.PENDING`` from any thread; a launch drains the list
(``take_pending``).  A drain that copied the list and then cleared it lost any
storage appended in between, so its value was not encoded before the next
in-place launch.  Pure Python: no MLX needed.
"""

import threading
from types import SimpleNamespace

from woof.backend.facade import _array


def test_take_pending_returns_values_in_order_and_empties():
    saved = _array.PENDING[:]
    _array.PENDING.clear()
    try:
        _array.PENDING.extend(SimpleNamespace(arr=i) for i in range(5))
        assert _array.take_pending() == [0, 1, 2, 3, 4]
        assert _array.PENDING == [] and _array.take_pending() == []
    finally:
        _array.PENDING[:] = saved


def test_concurrent_appends_and_drains_lose_nothing():
    saved = _array.PENDING[:]
    _array.PENDING.clear()
    per_thread, writers = 20000, 4
    drained, stop = [], threading.Event()

    def write(base):
        for i in range(per_thread):
            _array.PENDING.append(SimpleNamespace(arr=base + i))

    def drain():
        while not stop.is_set():
            drained.extend(_array.take_pending())

    try:
        drainers = [threading.Thread(target=drain) for _ in range(2)]
        threads = [threading.Thread(target=write, args=(k * per_thread,)) for k in range(writers)]
        for t in drainers + threads:
            t.start()
        for t in threads:
            t.join()
        stop.set()
        for t in drainers:
            t.join()
        drained.extend(_array.take_pending())
        assert sorted(drained) == list(range(writers * per_thread))
    finally:
        _array.PENDING[:] = saved


class _AppendsMidDrain(list):
    """A PENDING to which another thread appends one more storage after the
    drain has read the list and before it removes what it read."""

    def __init__(self, items, late):
        super().__init__(items)
        self.late = late

    def _arrive(self):
        if self.late is not None:
            late, self.late = self.late, None
            self.append(late)

    def clear(self):
        self._arrive()
        super().clear()

    def __delitem__(self, key):
        self._arrive()
        super().__delitem__(key)


def test_a_storage_appended_mid_drain_stays_pending(monkeypatch):
    late = SimpleNamespace(arr="late")
    pending = _AppendsMidDrain([SimpleNamespace(arr=i) for i in range(3)], late)
    monkeypatch.setattr(_array, "PENDING", pending)
    assert _array.take_pending() == [0, 1, 2]
    assert list(pending) == [late]          # (a copy-then-clear drain lost it)
