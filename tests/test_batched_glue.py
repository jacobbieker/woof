"""The one-launch row tables (bandwidth_glue.cu glue_accumulate_scaled_arrays,
glue_zero_frames) store the words of the per-buffer launches they replace:
``accumulate_scaled`` per accumulator and ``zero_frame`` per buffer."""
from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import _numerics_child


def _poisoned(rng, shape):
    words = rng.integers(0, 2**32, int(np.prod(shape)), dtype=np.uint32)
    return cp.asarray(words.reshape(shape)).view(cp.float32)


def _words(a):
    return a.view(cp.uint32).get()


def test_accumulate_scaled_rows_are_the_per_row_words():
    from woof.core.bandwidth_glue import accumulate_scaled, accumulate_scaled_rows
    rng = np.random.default_rng(29)
    for scale in (np.float32(24.0), np.float32(7.5), np.float32(1e-3)):
        rows = []
        for n in (1, 3, 127, 129, 4000, 120 * 150):
            src = (rng.standard_normal(n) * 300).astype(np.float32)
            dst = (rng.standard_normal(n) * 1e6).astype(np.float32)
            src[:3] = np.float32(1e-39)              # subnormal products and sums
            dst[:2] = np.float32(-0.0)
            rows.append((dst, src))
        shared = (rng.standard_normal(500) * 50).astype(np.float32)
        rows += [((rng.standard_normal(500) * 1e3).astype(np.float32), shared)] * 2
        want = [cp.asarray(d) for d, _s in rows]
        got = [cp.asarray(d) for d, _s in rows]
        srcs = [cp.asarray(s) for _d, s in rows]
        srcs[-1] = srcs[-2]                          # two accumulators, one source
        for d, s in zip(want, srcs):
            assert accumulate_scaled(d, s, scale)
        assert accumulate_scaled_rows(list(zip(got, srcs)), scale)
        for g, w in zip(got, want):
            assert np.array_equal(_words(g), _words(w))
        alias = cp.asarray(rows[3][0])                # dst += dst * scale
        expect = alias.copy()
        assert accumulate_scaled(expect, expect, scale)
        assert accumulate_scaled_rows([(alias, alias)], scale)
        assert np.array_equal(_words(alias), _words(expect))


def test_accumulate_scaled_rows_leave_unsupported_rows_to_the_caller():
    from woof.core.bandwidth_glue import accumulate_scaled_rows
    a = cp.ones((4, 6), dtype=cp.float32)
    b = cp.ones((4, 6), dtype=cp.float32)
    flat = cp.ones(30, dtype=cp.float32)
    two = np.float32(2)
    assert not accumulate_scaled_rows([], two)
    assert not accumulate_scaled_rows([(a, cp.ones((4, 6), dtype=cp.float64))], two)
    assert not accumulate_scaled_rows([(a, cp.ones((4, 3), dtype=cp.float32))], two)
    assert not accumulate_scaled_rows([(a[:, ::2], b[:, ::2])], two)
    assert not accumulate_scaled_rows([(flat[1:], flat[:-1])], two)   # partial overlap
    c = cp.ones((4, 6), dtype=cp.float32)
    assert not accumulate_scaled_rows([(a, b), (b, a)], two)          # a later row reads a dst
    assert not accumulate_scaled_rows([(a, b), (c, a)], two)          # one way only
    assert not accumulate_scaled_rows([(a, b), (a, c)], two)          # one dst twice
    assert not accumulate_scaled_rows([(a, b)] * 65, two)
    assert all(bool((x == 1).all()) for x in (a, b, c, flat))


def test_accumulate_scaled_rows_fill_the_whole_table():
    from woof.core.bandwidth_glue import accumulate_scaled, accumulate_scaled_rows
    rng = np.random.default_rng(64)
    dsts = [cp.asarray((rng.standard_normal(97 + 13 * r) * 1e4).astype(np.float32)) for r in range(64)]
    srcs = [cp.asarray((rng.standard_normal(d.size) * 10).astype(np.float32)) for d in dsts]
    want = [d.copy() for d in dsts]
    for d, s in zip(want, srcs):
        assert accumulate_scaled(d, s, np.float32(24))
    assert accumulate_scaled_rows(list(zip(dsts, srcs)), np.float32(24))
    for g, w in zip(dsts, want):
        assert np.array_equal(_words(g), _words(w))


@pytest.mark.parametrize("batched", ["1", "0"])
def test_surface_energy_accumulators_are_the_per_row_words(batched, monkeypatch):
    from woof.core import physics
    monkeypatch.setenv("WOOF_BATCHED_GLUE", batched)
    monkeypatch.setattr(physics, "WRF_EXACT", False)
    rng = np.random.default_rng(3)
    names = dict(physics.SURFACE_ENERGY_ACCUMULATORS)
    host = {k: (rng.standard_normal((6, 7)) * 300).astype(np.float32)
            for k in list(names) + list(names.values())}
    cfg = SimpleNamespace(diag_print=0, sf_surface_physics=2, dt=24.0)
    driver = SimpleNamespace(fields={k: cp.asarray(v) for k, v in host.items()})
    physics.PhysicsDriver.accumulate_surface_energy(driver, cfg)
    for name, source in names.items():
        want = cp.asarray(host[name])
        want += cp.asarray(host[source]) * np.float32(24.0)
        assert np.array_equal(_words(driver.fields[name]), _words(want))


@pytest.mark.parametrize("planes", [(), (1,), (3,), (2, 3)])
@pytest.mark.parametrize("ny,nx", [(9, 11), (5, 4), (1, 7), (12, 3)])
def test_zero_frames_are_the_per_buffer_words(planes, ny, nx):
    from woof.core.ring_frame import zero_frame, zero_frames
    rng = np.random.default_rng(ny * 100 + nx + len(planes))
    widths = [(1, 1, 1, 1), (3, 3, 3, 3), (1, 2, 0, 0), (0, 0, 2, 1), (0, 0, 0, 0),
              (2, 1, 1, 2), (6, 6, 1, 1), (1, 1, 5, 7), (-1, 9, 0, 40)]
    for x_lo, x_hi, y_lo, y_hi in widths:
        shapes = [planes + (ny, nx), (4,) + planes + (ny, nx), (ny, nx), (0, ny, nx)]
        got = [_poisoned(rng, s) for s in shapes]
        want = [g.copy() for g in got]
        for w in want:
            if w.size:
                assert zero_frame(w, x_lo, x_hi, y_lo, y_hi)
        assert zero_frames(got, x_lo, x_hi, y_lo, y_hi)
        for g, w in zip(got, want):
            assert np.array_equal(_words(g), _words(w)), (x_lo, x_hi, y_lo, y_hi, g.shape)


def test_zero_frames_leave_unsupported_buffers_to_the_caller():
    from woof.core.ring_frame import zero_frames
    a = cp.ones((3, 5, 6), dtype=cp.float32)
    for bufs in ([], [a, cp.ones((3, 5, 7), dtype=cp.float32)],
                 [a, cp.ones((3, 5, 12), dtype=cp.float32)[:, :, ::2]], [None, a],
                 [a, cp.ones((3, 5, 6), dtype=cp.float64)], [a, cp.ones(6, dtype=cp.float32)],
                 [a] * 65):
        assert not zero_frames(bufs, 1, 1, 1, 1)
    assert bool((a == 1).all())
    assert zero_frames([a, a[1]], 1, 1, 1, 1)          # overlapping views: every store is +0
    assert float(a[:, 0].sum()) == 0 and float(a[:, 1:-1, 1:-1].sum()) == 3 * 3 * 4


@pytest.mark.parametrize("bdy", ["open_x", "open_y", "both", "periodic"])
def test_open_strip_rows_are_the_per_buffer_words(bdy, monkeypatch):
    from woof.core import dycore
    rng = np.random.default_rng(7)
    nz, ny, nx = 4, 10, 12
    cfg = SimpleNamespace(nx=nx, ny=ny)
    monkeypatch.setattr(dycore, "_boundary_x", lambda c: bdy in ("open_x", "both"))
    monkeypatch.setattr(dycore, "_boundary_y", lambda c: bdy in ("open_y", "both"))
    for batched in ("1", "0"):
        monkeypatch.setenv("WOOF_BATCHED_GLUE", batched)
        got = ([_poisoned(rng, (nz, ny, nx)) for _ in range(11)] + [_poisoned(rng, (nz + 1, ny, nx))]
               + [_poisoned(rng, (nz, ny, nx + 1))] * (batched == "0"))
        want = [g.copy() for g in got]
        for w in want:
            dycore._zero_open_strips(w, cfg, 1)
        dycore._zero_open_strips_rows(got, cfg, 1)
        for g, w in zip(got, want):
            assert np.array_equal(_words(g), _words(w))
    mixed = [_poisoned(rng, (nz, ny, nx)), _poisoned(rng, (nz, ny, nx + 1))]   # zero_frames refuses
    want = [m.copy() for m in mixed]
    for w in want:
        dycore._zero_open_strips(w, cfg, 1)
    dycore._zero_open_strips_rows(mixed, cfg, 1)
    for g, w in zip(mixed, want):
        assert np.array_equal(_words(g), _words(w))


@pytest.mark.gpu
@pytest.mark.parametrize(*_numerics_child.MODES)
def test_identity_holds_under_each_numerics(numerics, exact):
    """The battery runs strict; production runs fast.  Each mode in its own
    child process (tests/_numerics_child.py)."""
    _numerics_child.run(__file__, ["test_accumulate_scaled_rows_are_the_per_row_words",
                                   "test_accumulate_scaled_rows_fill_the_whole_table",
                                   "test_surface_energy_accumulators_are_the_per_row_words",
                                   "test_zero_frames_are_the_per_buffer_words",
                                   "test_open_strip_rows_are_the_per_buffer_words"],
                        numerics, exact)
