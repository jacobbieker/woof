"""WRFDA 4D-Var's outer loops (woof.da.var.outer4d, ``analyse(var4d=..., max_ext_its=...)``; PR 4D.4).

Breakage this prevents:

- an outer loop that does not follow ``da_solve``: ``cvt`` not the sum of
  every loop's ``xhat``, ``Jb`` measured from the previous analysis rather
  than the first guess (loop 2's starting ``J`` must be the nonlinear ``J``
  at loop 1's analysis), an increment added twice or not carried, the
  rerun not started from the analysis, or the TL model not relinearised
  about the rerun (a toy with the identity TL model, and the real window);
- an outer loop run without a nonlinear model, a rerun that is not a
  :class:`~woof.da.var.outer4d.Trajectory`, a TL model that cannot be
  relinearised, or a rerun whose slots are off the window, accepted;
- WRFDA's ``var4d_bin``/``time_step`` rules not enforced, or the window
  model's slot spacing not matching them;
- the window model run on Metal (no binary64 there);
- on P3a's real window (dry, periodic, binary64): two outer loops not
  lowering ``J``, the relinearised window not its own transpose, or its TL
  not the finite-difference limit of the nonlinear rerun (the L2-style
  OSSE, ``tests/_var4d_osse.py``; host backend here, ``cuda`` on pb with
  ``WOOF_TEST_VAR4D_BACKEND=cuda``).

Against WRFDA itself: the 4D.4 L1 gate, lane L1 with ``max_ext_its`` 1 or 2
(``WOOF_WRFDA471_4DVAR_L1_DIR``, CUDA; tests/_var4d_l1.py).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from woof.da.var import fourd as f4
from woof.da.var import outer4d as o4
from woof.da.var import window4d as w4
from woof.da.var.cost import VarCost
from woof.da.var.minimise import minimise

ROOT = Path(__file__).resolve().parents[1]
NX, NY, NZ = 4, 3, 2
SIGMA_B, SIGMA_O = 0.7, 0.3


# ------------------------------------------------------------ a toy with WRFDA's loop structure


class ToyU:
    precision = "reference"
    dtype = np.float64
    nx, ny, nz = NX, NY, NZ
    cv_mz = (NZ, 0, 0, 0, 0)
    cv_size = NX * NY * NZ

    def apply(self, cv):
        return {"xa": {"t": SIGMA_B * np.asarray(cv, dtype=np.float64).reshape(NX, NY, NZ)}}

    def adjoint(self, xa):
        return {"cv": SIGMA_B * np.asarray(xa["t"], dtype=np.float64).ravel()}


IDX = np.array([0, 3, 7, 11, 13, 18, 22])


def h(s):
    """The nonlinear observation operator: ``s + 0.05 s^2`` at ``IDX``."""
    s = s.ravel()[IDX]
    return s + 0.05 * s * s


class ToyTerm:
    """``H'_n M'`` over the slots, ``M`` the TL model (the identity here), ``H'_n`` about slot ``n``'s state."""

    def __init__(self, model, slots, d):
        self.model, self.slots = model, slots
        self.d = np.asarray(d, dtype=np.float64)
        self.err = np.full(self.d.size, SIGMA_O)
        self.mask = np.ones(self.d.size, bool)

    def jo_split(self):
        return [("sound", [("t", np.arange(self.d.size))])]

    def tl(self, x):
        gs = self.model.tl({"t_2": np.asarray(x["t"], dtype=np.float64)}, len(self.slots))
        return np.concatenate([(1.0 + 0.1 * s.ravel()[IDX]) * g["t_2"].ravel()[IDX] for g, s in zip(gs, self.slots)])

    def ad(self, y):
        m = IDX.size
        forcings = []
        for n, s in enumerate(self.slots):
            flat = np.zeros(NX * NY * NZ)
            np.add.at(flat, IDX, (1.0 + 0.1 * s.ravel()[IDX]) * y[n * m:(n + 1) * m])
            forcings.append({"t_2": flat.reshape(NX, NY, NZ)})
        return {"t": self.model.ad(forcings)["t_2"]}


class SpyIdentity(f4.IdentityModel):
    def __init__(self):
        self.relinearised = []

    def relinearise(self, trajectory):
        self.relinearised.append(trajectory)


def _toy(nslots=3, model=None):
    """A persistence nonlinear model (every slot is the start) with the identity TL: the pair WRFPLUS's
    ``enable_identity`` makes when the trajectory does not move."""
    rng = np.random.default_rng(4)
    bg = rng.normal(0.0, 1.0, (NX, NY, NZ))
    truth = bg + rng.normal(0.0, 1.5, (NX, NY, NZ))
    y = np.concatenate([h(truth)] * nslots)
    model = model or SpyIdentity()
    calls = []

    def run(x0):
        return o4.Trajectory(slots=[x0] * nslots)

    def nl_model(it, x0):
        calls.append((it, x0.copy()))
        return run(x0)

    def jo_nl(x0):
        d = y - np.concatenate([h(x0)] * nslots)
        return 0.5 * float(np.sum((d / SIGMA_O) ** 2)), d

    def setup(it, trajectory):
        _, d = jo_nl(trajectory.slots[0])
        return VarCost(ToyU(), ToyTerm(model, trajectory.slots, d)), trajectory.slots[0]

    def update(it, x0, xhat):
        return x0 + SIGMA_B * np.asarray(xhat).reshape(NX, NY, NZ)

    return dict(bg=bg, run=run, nl_model=nl_model, setup=setup, update=update, jo_nl=jo_nl, calls=calls,
                model=model)


def test_two_outer_loops_follow_da_solve():
    t = _toy()
    results, cvt, final, x_last = o4.outer_loops(t["setup"], t["update"], t["run"](t["bg"]), max_ext_its=2,
                                                 nl_model=t["nl_model"], model=t["model"], ntmax=[30, 20], eps=1e-8)
    x1, x2 = (np.asarray(r.xhat) for r in results)
    # cvt is every loop's xhat summed (da_solve.inc line 995)
    np.testing.assert_array_equal(cvt, x1 + x2)
    # the rerun starts from loop 1's analysis, the model is relinearised about it, loop 2 is about it
    a1 = t["bg"] + SIGMA_B * x1.reshape(NX, NY, NZ)
    assert [c[0] for c in t["calls"]] == [2]
    np.testing.assert_array_equal(t["calls"][0][1], a1)
    assert len(t["model"].relinearised) == 1 and t["model"].relinearised[0].slots[0] is not None
    np.testing.assert_array_equal(x_last, a1)
    # each loop's xa is its own xhat's, added to the state the loop was about (da_solve.inc line 1032)
    np.testing.assert_allclose(final, t["bg"] + SIGMA_B * (x1 + x2).reshape(NX, NY, NZ), rtol=0, atol=1e-14)
    # Jb from the FIRST guess: loop 2 starts at 0.5 |cvt_1|^2 + Jo at loop 1's analysis (nonlinear)
    jo1, _ = t["jo_nl"](a1)
    assert results[1].j[0] == pytest.approx(0.5 * float(x1 @ x1) + jo1, rel=1e-13)
    jo2, _ = t["jo_nl"](final)
    j_nl = [t["jo_nl"](t["bg"])[0], 0.5 * float(x1 @ x1) + jo1, 0.5 * float(cvt @ cvt) + jo2]
    assert j_nl[1] < 0.5 * j_nl[0] and j_nl[2] < j_nl[1]
    assert results[1].j_final <= results[1].j[0]
    assert results[0].iterations <= 30 and results[1].iterations <= 20
    assert [r.it for r in results] == [1, 2]


def test_one_loop_is_minimise_word_for_word():
    t = _toy(model=f4.IdentityModel())
    first = t["run"](t["bg"])
    results, cvt, _, _ = o4.outer_loops(t["setup"], t["update"], first, max_ext_its=1, ntmax=10, eps=1e-3)
    cost, _ = t["setup"](1, first)
    ref, cvt_ref = minimise(cost, max_ext_its=1, ntmax=10, eps=1e-3)
    assert results[0].lines == ref[0].lines
    np.testing.assert_array_equal(cvt, cvt_ref)


def test_outer_loop_refusals():
    t = _toy()
    first = t["run"](t["bg"])
    with pytest.raises(NotImplementedError, match="needs nl_model"):
        o4.outer_loops(t["setup"], t["update"], first, max_ext_its=2, model=t["model"])
    with pytest.raises(ValueError, match="max_ext_its"):
        o4.outer_loops(t["setup"], t["update"], first, max_ext_its=0)
    with pytest.raises(TypeError, match="not a Trajectory"):
        o4.outer_loops(t["setup"], t["update"], first, max_ext_its=2, nl_model=lambda it, x: [x] * 3,
                       model=t["model"], ntmax=2)
    with pytest.raises(TypeError, match="no relinearise"):
        o4.outer_loops(t["setup"], t["update"], first, max_ext_its=2, nl_model=t["nl_model"], model=object(),
                       ntmax=2)

    def shrinking(it, trajectory):
        cost, ctx = t["setup"](it, trajectory)
        if it > 1:
            cost.cv_size -= 1
        return cost, ctx

    with pytest.raises(ValueError, match="outer loop 2: the control vector"):
        o4.outer_loops(shrinking, t["update"], first, max_ext_its=2, nl_model=t["nl_model"], model=t["model"],
                       ntmax=2)


# ------------------------------------------------------------ the window model's relinearisation


def _fake_window(**kw):
    from test_var_window4d import FakeWindow

    return FakeWindow(3, 2, 2, 4, **kw)


def test_window_model_relinearises_about_the_rerun():
    from test_var_window4d import HALO, _tl_state

    win = _fake_window()
    model = w4.WindowModel(win, 2, to_device=np.asarray, to_host=np.asarray)
    states = [{n: v + 10.0 * k for n, v in win._start.items()} for k in range(5)]
    end = model.relinearise(o4.Trajectory(slots=[], states=states))
    assert win._external and win._checkpoints[3]["u"][0, 0, 0] == 31.0 and end is states[4]
    # the base check reads the stored end for the last slot
    grids = [(None, {g: w4.from_window(states[2 * n][w]) for g, w in w4._BASE.items()}) for n in range(3)]
    assert max(model.check_base(grids).values()) == 0.0
    # still its own transpose after the switch
    rng = np.random.default_rng(1)
    g0, fs = _tl_state(rng, 3, 2, 2), [_tl_state(rng, 3, 2, 2) for _ in range(3)]
    from woof.da.var import transfer4d as t4

    dot = lambda a, b: sum(float(np.sum(a[k] * b[k])) for k in t4.TL_KEYS)  # noqa: E731
    assert sum(dot(g, f) for g, f in zip(model.tl(g0, 3), fs)) == pytest.approx(dot(g0, model.ad(fs)), rel=1e-12)
    with pytest.raises(ValueError, match="per-step states"):
        model.relinearise(o4.Trajectory(slots=[]))
    # a specified window under halo_start="wrfplus": the rerun's state 0 must carry the reset halo
    win = _fake_window(halo=True)
    model = w4.WindowModel(win, 2, to_device=np.asarray, to_host=np.asarray)
    bad = [{n: v + k for n, v in win._start.items()} for k in range(5)]
    bad[0] = dict(bad[0], **{HALO: bad[0][HALO] + 1.0})
    with pytest.raises(ValueError, match="halo"):
        model.relinearise(o4.Trajectory(slots=[], states=bad))
    assert not getattr(win, "_external", False)   # refused before the base moved
    f4.IdentityModel().relinearise(o4.Trajectory(slots=[]))   # nothing to relinearise


def test_window_model_is_refused_on_metal(monkeypatch):
    from woof import backend

    win = _fake_window()
    monkeypatch.setattr(backend, "active", lambda: "mlx")
    with pytest.raises(NotImplementedError, match="Metal"):
        w4.WindowModel(win, 2)
    w4.WindowModel(win, 2, to_device=np.asarray, to_host=np.asarray)   # host arrays: no device, no refusal
    monkeypatch.setattr(backend, "active", lambda: "host")
    w4._refuse_metal()


@pytest.mark.skipif(os.environ.get("WOOF_BACKEND", "").lower() not in ("mlx", "metal"),
                    reason="the Metal arm (WOOF_BACKEND=mlx)")
def test_window_model_on_the_metal_backend_is_refused():
    from woof import backend

    assert backend.active() == "mlx"
    with pytest.raises(NotImplementedError, match="binary64"):
        w4.WindowModel(_fake_window(), 2)


# ------------------------------------------------------------ WRFDA's window rules


def test_check_var4d_bins():
    w = ("2008-02-05_12:00:00", "2008-02-05_18:00:00")
    assert f4.check_var4d_bins(*w, 7, var4d_bin=3600, time_step=360) == {
        "var4d_bin": 3600, "num_fgat_time": 7, "steps_per_bin": 10}
    assert f4.check_var4d_bins(*w, 7) == {"var4d_bin": 3600, "num_fgat_time": 7, "steps_per_bin": None}
    # var4d_bin below time_step becomes time_step (WRFDA's nl_set_var4d_bin)
    assert f4.check_var4d_bins(*w, 61, var4d_bin=60, time_step=360)["var4d_bin"] == 360
    # the window in whole minutes, as da_get_julian_time reads it
    assert f4.check_var4d_bins("2008-02-05_12:00:30", "2008-02-05_18:00:00", 7)["var4d_bin"] == 3600
    for kw, match in ((dict(var4d_bin=5000), "evenly divisible by var4d_bin"),
                      (dict(var4d_bin=3600, time_step=7), "var4d_bin must be evenly divisible by time_step"),
                      (dict(var4d_bin=1800), "makes 13 slots"),
                      (dict(time_step=0), "time_step"),
                      (dict(var4d_bin=3600.5), "whole number of seconds")):
        with pytest.raises(ValueError, match=match):
            f4.check_var4d_bins(*w, 7, **kw)
    with pytest.raises(ValueError, match="give var4d_bin"):
        f4.check_var4d_bins("2008-02-05_12:00:00", "2008-02-05_12:10:00", 8)   # 600 s in 7 bins
    with pytest.raises(ValueError, match="empty"):
        f4.check_var4d_bins(w[1], w[0], 7)


def test_analyse_refuses_outer_loop_setups_before_reading_anything(tmp_path):
    """The outer-loop and window refusals fire on the arguments and the files'
    headers: the first guesses hold no fields and ``be.dat`` does not exist."""
    from test_var_fourd import _wrf_file

    from woof.da.var.analysis import analyse

    fgs = [_wrf_file(tmp_path / f"fg0{n}", f"2008-02-05_{11 + n}:00:00") for n in range(1, 8)]
    obs = []
    for n in range(1, 8):
        (tmp_path / f"ob0{n}.ascii").write_text(f"slot {n}\n")
        obs.append(tmp_path / f"ob0{n}.ascii")
    be = tmp_path / "missing-be.dat"
    kw = dict(fgat=fgs, time_window=("2008-02-05_12:00:00", "2008-02-05_18:00:00"))
    with pytest.raises(NotImplementedError, match="max_ext_its = 2 needs var4d.nl_model"):
        analyse(be, obs, fgs[0], max_ext_its=2, var4d=f4.Var4D(), **kw)
    with pytest.raises(ValueError, match="max_ext_its must be >= 1"):
        analyse(be, obs, fgs[0], max_ext_its=0, var4d=f4.Var4D(), **kw)

    class Frozen:
        def tl(self, g0, n):
            return [g0] * n

        def ad(self, fs):
            return fs[0]

    with pytest.raises(TypeError, match="Frozen has no relinearise"):
        analyse(be, obs, fgs[0], max_ext_its=2, var4d=f4.Var4D(model=Frozen(), nl_model=print), **kw)
    for v4, match in ((f4.Var4D(var4d_bin=5000), "evenly divisible by var4d_bin"),
                      (f4.Var4D(var4d_bin=3600, time_step=7), "divisible by time_step"),
                      (f4.Var4D(var4d_bin=1800), "makes 13 slots")):
        with pytest.raises(ValueError, match=match):
            analyse(be, obs, fgs[0], var4d=v4, **kw)
    spaced = Frozen()
    spaced.sps = 12
    with pytest.raises(ValueError, match="every 12 steps, but var4d_bin / time_step = 3600 / 360 = 10"):
        analyse(be, obs, fgs[0], var4d=f4.Var4D(model=spaced, var4d_bin=3600, time_step=360), **kw)


# ------------------------------------------------------------ analyse's outer loop with WRFDA's pieces faked


def _fake_pieces(monkeypatch, nslots=3):
    """``_analyse_var4d`` with its file, ``U``, transfer and observation pieces replaced by the toy's: a
    first guess is ``{"t": field}``, the observations are ``h`` of each slot's state, ``U`` is ``ToyU``."""
    from woof.da.var import analysis as A
    from woof.da.var import cost as C
    from woof.da.var import fgat as fm
    from woof.da.var import obs_wrfda as ow

    rng = np.random.default_rng(9)
    bg = rng.normal(0.0, 1.0, (NX, NY, NZ))
    y = np.concatenate([h(bg + rng.normal(0.0, 1.5, (NX, NY, NZ)))] * nslots)

    def fg(t, stamp):
        return A.FirstGuess(path=Path(f"fg_{stamp}"), fields={"t": t}, attrs={}, times=stamp)

    def wrftoxb(f, landuse_table=None):
        return {k: f.fields["t"] for k in A._XBU}, {"start": f.times}

    def term(iv, states, base, model, press, ph):
        slots = [x["t"] for x, _ in states]
        return ToyTerm(model, slots, y - np.concatenate([h(s) for s in slots]))

    monkeypatch.setattr(A, "wrftoxb", wrftoxb)
    monkeypatch.setattr(A, "xa_from_cv", lambda b, xbu, cv: {"t": SIGMA_B * cv.reshape(NX, NY, NZ)})
    monkeypatch.setattr(A, "xatowrf", lambda grid, xb, xa, check_rh=0, var4d=False: {"T": xb["t"] + xa["t"]})
    monkeypatch.setattr(A, "_analysis_state", lambda start, wrf: fg(wrf["T"], start.times))
    monkeypatch.setattr(ow, "LambertWRFDA", type("P", (), {"from_attributes": staticmethod(lambda a: "proj")}))
    monkeypatch.setattr(fm, "read_obs_ascii_fgat", lambda paths, proj, **w: "obs")
    monkeypatch.setattr(fm, "innovations_fgat", lambda wobs, xbs: "iv")
    monkeypatch.setattr(fm, "first_guess_time", lambda f: f.times)
    monkeypatch.setattr(fm, "check_fgat", lambda n, times, **w: None)
    monkeypatch.setattr(f4, "FourDObsTerm", term)
    monkeypatch.setattr(C.VarCost, "reference", staticmethod(lambda b, xbu, terms: VarCost(ToyU(), terms[0])))
    stamps = [f"2008-02-05_{12 + n}:00:00" for n in range(nslots)]
    return A, bg, y, fg, stamps


def test_analyse_runs_two_outer_loops_through_its_pieces(monkeypatch):
    """``analyse``'s 4D-Var outer loop (``_analyse_var4d``) on the toy: loop 2
    reruns from loop 1's analysis, the base check sees the rerun's slots, the
    result is the last loop's increment about the rerun's start, the fit and
    the timings are reported."""
    import time

    A, bg, y, fg, stamps = _fake_pieces(monkeypatch)
    calls, checked = [], []

    class Spy(SpyIdentity):
        def check_base(self, states, rtol):
            checked.append(([x["t"] for x, _ in states], rtol))

    def nl_model(it, state):
        calls.append((it, state))
        return o4.Trajectory(slots=[fg(state.fields["t"], s) for s in stamps])

    model = Spy()
    first = [fg(bg, s) for s in stamps]
    xb, grid = A.wrftoxb(first[0])

    def run(nl=nl_model, **kw):
        return A._analyse_var4d(None, first[0], xb, grid, ["o"] * 3, first, "proj", {"time_window_min": stamps[0],
                                "time_window_max": stamps[-1]}, f4.Var4D(model=model, nl_model=nl, base_rtol=1e-6),
                                max_ext_its=2, ntmax=30, eps=1e-8, check_rh=0, landuse_table=None, write=None,
                                timings={}, t0=time.perf_counter(), **kw)

    r = run(report_fit=True)
    x1, x2 = (np.asarray(c.xhat) for c in r.cg)
    a1 = bg + SIGMA_B * x1.reshape(NX, NY, NZ)
    assert [c[0] for c in calls] == [2] and len(model.relinearised) == 1
    np.testing.assert_array_equal(calls[0][1].fields["t"], a1)
    assert len(checked) == 1 and checked[0][1] == 1e-6
    assert all(np.array_equal(s, a1) for s in checked[0][0])
    np.testing.assert_array_equal(r.cvt, x1 + x2)
    np.testing.assert_array_equal(r.xb["t"], a1)
    assert r.grid == {"start": stamps[0]}
    np.testing.assert_array_equal(r.xa["t"], SIGMA_B * x2.reshape(NX, NY, NZ))
    np.testing.assert_array_equal(r.wrf["T"], a1 + SIGMA_B * x2.reshape(NX, NY, NZ))
    # Jb from the first guess: loop 2 starts at 0.5 |xhat1|^2 + Jo at loop 1's analysis
    jo1 = 0.5 * float(np.sum(((y - np.concatenate([h(a1)] * 3)) / SIGMA_O) ** 2))
    assert r.cg[1].j[0] == pytest.approx(0.5 * float(x1 @ x1) + jo1, rel=1e-13)
    assert {"setup_s", "minimise_s", "transfer_s", "outer_setup_s", "nl_model_s"} <= set(r.timings)
    assert r.fit["conventional"]["n"] == y.size and r.fit["conventional"]["oma_rms"] < r.fit["conventional"]["omb_rms"]
    assert r.nobs == y.size and run().fit == {}
    # a rerun with the wrong slot count, or on another grid, is refused in loop 2
    with pytest.raises(ValueError, match="outer loop 2: the rerun has 2 slot states, the window 3"):
        run(nl=lambda it, s: o4.Trajectory(slots=[fg(s.fields["t"], t) for t in stamps[:2]]))
    with pytest.raises(ValueError, match="outer loop 2: slot 1"):
        run(nl=lambda it, s: o4.Trajectory(slots=[fg(np.zeros((NX + 1, NY, NZ)), t) for t in stamps]))


def test_analyse_carries_the_boundary_increment_to_the_next_loop(monkeypatch):
    """A TL model with boundary tables (``WindowModel(lbc="wrfda")``): after
    loop 1, ``analyse`` asks it for the table perturbation of loop 1's
    ``xhat`` (transferred about loop 1's start), adds it to the model's
    tables and hands it to the rerun on the analysis state
    (da_transfer_xatowrf.inc:665-727); the last loop makes none."""
    import time

    from woof.da.var import transfer4d as t4

    A, bg, y, fg, stamps = _fake_pieces(monkeypatch)
    def toy_xa(b, xbu, cv):
        return {k: SIGMA_B * cv.reshape(NX, NY, NZ) for k in ("t", "u", "v", "q", "psfc")}

    monkeypatch.setattr(A, "xa_from_cv", toy_xa)
    monkeypatch.setattr(t4, "xa_to_tl", lambda x0, xb0, grid0, press, ph: {"t_2": 2.0 * x0["t"], "start": grid0})
    asked, added, seen = [], [], []

    class Tables(SpyIdentity):
        def boundary_increment(self, g0):
            asked.append(g0)
            return {"lbc_theta_west": g0["t_2"][:1]}

        def add_boundary_increment(self, inc):
            added.append(inc)

    def nl_model(it, state):
        seen.append(state)
        return o4.Trajectory(slots=[fg(state.fields["t"], s) for s in stamps])

    first = [fg(bg, s) for s in stamps]
    xb, grid = A.wrftoxb(first[0])
    r = A._analyse_var4d(None, first[0], xb, grid, ["o"] * 3, first, "proj",
                         {"time_window_min": stamps[0], "time_window_max": stamps[-1]},
                         f4.Var4D(model=Tables(), nl_model=nl_model), max_ext_its=2, ntmax=30, eps=1e-8, check_rh=0,
                         landuse_table=None, write=None, timings={}, t0=time.perf_counter())
    x1 = np.asarray(r.cg[0].xhat)
    assert len(asked) == 1 and asked[0]["start"] == {"start": stamps[0]}
    np.testing.assert_array_equal(asked[0]["t_2"], 2.0 * SIGMA_B * x1.reshape(NX, NY, NZ))
    assert len(added) == 1 and added[0] is seen[0].lbc_increment
    np.testing.assert_array_equal(seen[0].lbc_increment["lbc_theta_west"], asked[0]["t_2"][:1])


# ------------------------------------------------------------ the tutorial (identity TL, persistence NL)


def _relabel(fg, stamp):
    from woof.da.var.analysis import FirstGuess

    return FirstGuess(path=fg.path, fields=fg.fields, attrs=fg.attrs, times=stamp)


def test_two_outer_loops_on_the_tutorial_with_a_persistence_model():
    """``analyse(var4d=..., max_ext_its=2)`` end to end on the FGAT lane's files
    (``WOOF_WRFDA471_FGAT_DIR``): the 12Z state persisted over the window is
    the trajectory and the identity the TL model (a consistent pair), so
    loop 2 starts below loop 1 and the rerun starts from loop 1's analysis.
    No WRFDA comparison: that is the L1 lane's."""
    d = os.environ.get("WOOF_WRFDA471_FGAT_DIR")
    if not d:
        pytest.skip("WOOF_WRFDA471_FGAT_DIR is not set")
    from woof.da.var import fgat as fm
    from woof.da.var.analysis import WRF_FIELDS, analyse, read_first_guess

    d = Path(d)
    window = ("2008-02-05_12:00:00", "2008-02-05_18:00:00")
    times = [f"2008-02-05_{h}:00:00" for h in range(12, 19)]
    fg = read_first_guess(d / "fg01")
    obs = [d / f"ob0{n}.ascii" for n in range(1, 8)]
    seen = []

    def persistence(it, state):
        seen.append((it, state))
        return o4.Trajectory(slots=[_relabel(state, t) for t in times])

    lines = []
    r = analyse(d / "be.dat", obs, fg, fgat=[_relabel(fg, t) for t in times], time_window=window, ntmax=4,
                max_ext_its=2, var4d=f4.Var4D(nl_model=persistence, var4d_bin=3600, time_step=360),
                write=lines.append)
    assert [s[0] for s in seen] == [2] and len(r.cg) == 2
    assert sum(ln.startswith("Starting outer iteration :   2") for ln in lines) == 1
    j1, j2 = r.cg
    assert j2.j[0] < j1.j[0] and j2.j_final < j2.j[0]
    # the rerun started from loop 1's analysis: every WRF field differs from the first guess's somewhere
    state = seen[0][1]
    assert state.times == fg.times and any(np.any(state.fields[k] != fg.fields[k]) for k in WRF_FIELDS)
    np.testing.assert_array_equal(state.fields["P"], fg.fields["P"])   # var4d: no pressure increment
    # the final analysis is loop 2's increment on top of loop 1's analysis
    assert np.any(r.wrf["T"] != state.fields["T"])
    np.testing.assert_array_equal(r.grid["t_2"], state.fields["T"])
    assert r.timings["nl_model_s"] >= 0.0 and r.nobs > 0
    # a rerun off the slot times is refused
    with pytest.raises(ValueError, match="outer loop 2: the rerun's slots"):
        analyse(d / "be.dat", obs, fg, fgat=[_relabel(fg, t) for t in times], time_window=window, ntmax=1,
                max_ext_its=2, var4d=f4.Var4D(nl_model=lambda it, s: o4.Trajectory(
                    slots=[_relabel(s, times[0])] * 7)))
    assert fm.first_guess_time(state) == times[0]


def test_l1_two_outer_loops_against_wrfda(tmp_path):
    """The 4D.4 L1 gate (plan section 4): ``analyse(var4d=..., max_ext_its=N)``
    against WRFDA 4.7.1 + WRFPLUS 4.7.1 on lane L1 (adiabatic, ``diff_opt 0``, WRF's
    default dynamics ``damp_opt 3``, ``emdiv 0.01``, ``moist_adv_opt 1``, ``mp_zero_out 0``;
    ``var4d_lbc`` off).  The 4D.4 lane (``L1_NARROW=1``: ``damp_opt``, ``emdiv`` and
    ``moist_adv_opt`` 0) passes the same tolerances.

    ``WOOF_WRFDA471_4DVAR_L1_DIR`` is ``tools/wrfda_oracle/run_4dvar.sh l1``'s run
    directory with ``MAX_EXT_ITS`` 1 or 2 (``namelist.input``, ``fg``,
    ``wrfbdy_d01``, ``be.dat``, ``ob01.ascii`` .. ``ob07.ascii``, ``wrfda.out``,
    ``wrfvar_output`` and, for two loops, ``wrfvar_output_01``).  CUDA only
    (``WOOF_TEST_VAR4D_BACKEND=cuda``): tests/_var4d_l1.py runs P3a's binary64
    window as both models.  WRFPLUS's trajectory is not readable from this
    WRFDA build, so woof runs its own: the gate is tolerance-level, not word
    for word (tests/_var4d_l1.py's docstring), and checks per loop the
    iteration count, ``J0``, the final ``J`` and the gradient norms, the
    increments, and the window's dot product with the boundary tables."""
    d = os.environ.get("WOOF_WRFDA471_4DVAR_L1_DIR")
    if not d:
        pytest.skip("WOOF_WRFDA471_4DVAR_L1_DIR is not set")
    if os.environ.get("WOOF_TEST_VAR4D_BACKEND") != "cuda":
        pytest.skip("the L1 gate runs the lane's window on CUDA (WOOF_TEST_VAR4D_BACKEND=cuda)")
    d = Path(d)
    m = re.search(r"max_ext_its\s*=\s*(\d+)", (d / "namelist.input").read_text(), re.I)
    loops = int(m.group(1)) if m else 1
    env = dict(os.environ, WOOF_BACKEND="cuda", PYTHONPATH=str(ROOT), **STRICT)
    done = subprocess.run([sys.executable, str(ROOT / "tests" / "_var4d_l1.py"), str(d), str(tmp_path),
                           str(loops)], capture_output=True, text=True, env=env, timeout=7200)
    assert done.returncode == 0, done.stderr[-4000:]
    out = json.loads([ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1][7:])
    assert out["dot"]["rel"] < 1e-10
    woof, wrfda = out["woof_cg"], out["wrfda_cg"]
    for it in range(1, loops + 1):
        w = [r for r in woof if r[0] == it]
        f = [r for r in wrfda if r[0] == it]
        assert w and f, f"loop {it}: no minimize_cg lines"
        # J0 and |grad J0| (the innovations through woof's trajectory, L1_J0_RTOL), the final J
        assert w[0][2] == pytest.approx(f[0][2], rel=L1_J0_RTOL[it - 1])
        assert w[0][3] == pytest.approx(f[0][3], rel=L1_G0_RTOL[it - 1])
        assert w[-1][2] == pytest.approx(f[-1][2], rel=L1_JF_RTOL[it - 1])
        assert abs(len(w) - len(f)) <= L1_ITER_SLACK
    for name, fields in out["increments"].items():
        for k in ("U", "V", "T", "QVAPOR", "MU", "PH"):
            assert fields[k]["corr"] > L1_INC_CORR, (name, k, fields[k])
            assert abs(fields[k]["rms_ratio"] - 1.0) < 0.01, (name, k, fields[k])


#: L1 tolerances (two models from one start, not word-exact: the test's docstring),
#: from the 2026-11-27 pb runs, whose agreement docs/var.md records: loop 1 J0 4.2e-4,
#: |grad J0| 1.8e-3, final J 1.1e-3; loop 2 J0 5.7e-6, |grad J0| 9.7e-3, final J 6.8e-6;
#: 35 iterations in each loop on both sides; increments correlate to >= 0.99947.
L1_J0_RTOL = (1e-3, 1e-4)
L1_G0_RTOL = (5e-3, 2e-2)
L1_JF_RTOL = (3e-3, 1e-4)
L1_ITER_SLACK = 2
L1_INC_CORR = 0.999


# ------------------------------------------------------------ the L2-style OSSE on the real window

STRICT = dict(GPUWM_WRF_EXACT="1", WOOF_WRF_EXACT_DIAGNOSTICS="1", WOOF_WRF_EXACT_ADVECTION="1",
              GPUWM_WRF_EXACT_DIFFUSION="1", GPUWM_WRF_EXACT_BIGSTEP="1")


@pytest.mark.skipif(os.environ.get("WOOF_TEST_VAR4D_BACKEND", "host") == "host"
                    and not any(shutil.which(c) for c in ("clang++", "g++", "c++")), reason="no C++ compiler")
def test_osse_two_outer_loops_on_the_real_window(tmp_path):
    """The dry periodic OSSE (``tests/_var4d_osse.py``): two outer loops with
    the nonlinear rerun and the window relinearised about it."""
    backend = os.environ.get("WOOF_TEST_VAR4D_BACKEND", "host")
    env = dict(os.environ, WOOF_BACKEND=backend, PYTHONPATH=str(ROOT), WOOF_HOST_CACHE=str(tmp_path / "hostcache"),
               WOOF_HOST_THREADS="4", **STRICT)
    done = subprocess.run([sys.executable, str(ROOT / "tests" / "_var4d_osse.py")], capture_output=True, text=True,
                          env=env, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    out = json.loads([ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1][7:])
    (j0_1, jf_1, it_1), (j0_2, jf_2, it_2) = out["j"]
    jb, ja1, ja2 = out["j_nl"]
    assert out["calls"] == [2]
    # loop 1 starts at the background's J; loop 2 at the nonlinear J of loop 1's analysis (Jb from the first guess)
    assert j0_1 == pytest.approx(jb, rel=1e-12)
    assert j0_2 == pytest.approx(ja1, rel=1e-12)
    # two outer loops lower J: the nonlinear J after each loop, and each loop's own minimisation
    assert ja1 < 0.2 * jb and ja2 < ja1
    assert jf_1 < j0_1 and jf_2 < j0_2 and it_1 >= 1 and it_2 >= 1
    # the increments accumulate (cvt = xhat1 + xhat2; the start is the background plus both)
    assert out["cvt_sum"] and out["acc"] < 1e-12
    # the gradient is consistent: WRFDA's Taylor test, and the relinearised window is its transpose
    for digits, best in out["taylor"]:
        assert digits >= 4 and best == pytest.approx(1.0, abs=1e-3)
    for lhs, rhs in out["dots"]:
        assert abs(lhs) > 1.0 and lhs == pytest.approx(rhs, rel=1e-12)
    # TL against finite differences of the nonlinear rerun about loop 1's analysis: first order in eps
    fd_big, fd_small = out["fd"]
    assert fd_small < 1e-5 and fd_small < fd_big / 50.0
