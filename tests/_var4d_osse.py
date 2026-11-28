"""The 4D-Var outer-loop OSSE (L2-style) on P3a's real window, run as a child
process by tests/test_var_outer4d.py (the backend is the child's
``WOOF_BACKEND``: host on the Mac, CUDA on pb).

A dry periodic ridge case, binary64: the window is 8 steps, three slots 4
steps apart.  The control is the potential-temperature perturbation at the
window's start (``U``: 0.5 K per control unit, through
:class:`~woof.da.var.cost.VarCost`); the observations are ``theta'`` at 40
points in each slot, taken from a nonlinear run of the truth (the
background plus a 3 K bubble).  ``H`` is a point selection, so every
nonlinearity is the model's.  :func:`woof.da.var.outer4d.outer_loops` runs
two outer loops: the nonlinear rerun from loop 1's analysis
(``Window.forward(x0)``), its per-step states as a
:class:`~woof.da.var.outer4d.Trajectory`, ``WindowModel.relinearise``, the
innovations ``y - H(M(x))`` per slot, CG.

Prints ``RESULT {json}``: each loop's ``J`` (start, final) and iteration
count, the nonlinear ``J`` at the background and after each loop, the
gradient (Taylor) test per loop, the dot product of the relinearised
window, TL against finite differences of the nonlinear run, and the
increment's accumulation check.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np

import woof  # noqa: F401  (installs the backend's cupy)
import cupy as cp
from tilestream import harness as H

if cp.__name__ != "cupy" or getattr(cp, "__woof_facade__", False):
    import woof.core.resident_admission as RA

    RA.device_free_bytes = lambda: None

from woof.core import dycore as D  # noqa: E402
from woof.da.var import transfer4d as t4  # noqa: E402
from woof.da.var import window4d as w4  # noqa: E402
from woof.da.var.cost import VarCost, check_gradient  # noqa: E402
from woof.da.var.outer4d import Trajectory, outer_loops  # noqa: E402
from woof.grad.wrfplus import window as W  # noqa: E402

NX, NY, NZ = 16, 12, 10
SPS, NSLOTS = 4, 3
SIGMA_B, SIGMA_O = 0.5, 0.2


def _case():
    cfg = H.make_config(NX, NY, NZ, h_sca_adv_order=5, terrain_opt=1)
    jj, ii = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    ridge = 400.0 * np.exp(-(((ii - NX / 2) / (NX / 5)) ** 2 + ((jj - NY / 2) / (NY / 4)) ** 2))
    geo = dataclasses.replace(H.neutral_geography(cfg), terrain=ridge)
    state = H.make_state(cfg, 1, geography=geo)
    D.step(state, cfg)
    return W.Window(state, cfg, SPS * (NSLOTS - 1))


class ToyU:
    """``U``: the control is ``theta'`` at the start, ``SIGMA_B`` K per unit (``xa['t']``)."""

    precision = "reference"
    dtype = np.float64
    nx, ny, nz = NX, NY, NZ
    cv_mz = (NZ, 0, 0, 0, 0)
    cv_size = NX * NY * NZ

    def apply(self, cv):
        return {"xa": {"t": SIGMA_B * np.asarray(cv, dtype=np.float64).reshape(NX, NY, NZ)}}

    def adjoint(self, xa):
        return {"cv": SIGMA_B * np.asarray(xa["t"], dtype=np.float64).ravel()}


def _shapes():
    return {"u_2": (NX + 1, NY, NZ), "v_2": (NX, NY + 1, NZ), "w_2": (NX, NY, NZ + 1), "ph_2": (NX, NY, NZ + 1),
            "t_2": (NX, NY, NZ), "qv": (NX, NY, NZ), "p": (NX, NY, NZ), "mu_2": (NX, NY)}


class PointTerm:
    """``H' M'``: ``g_t`` at the start through the window model, ``theta'`` at ``idx`` in every slot."""

    def __init__(self, model, idx, d):
        self.model, self.idx = model, idx
        self.m = idx.size
        self.d = np.asarray(d, dtype=np.float64)
        self.err = np.full(self.d.size, SIGMA_O)
        self.mask = np.ones(self.d.size, bool)

    def jo_split(self):
        return [("sound", [("t", np.arange(self.d.size))])]

    def tl(self, x):
        g0 = {k: np.zeros(s) for k, s in _shapes().items()}
        g0["t_2"] = np.asarray(x["t"], dtype=np.float64)
        gs = self.model.tl(g0, NSLOTS)
        return np.concatenate([g["t_2"].ravel()[self.idx] for g in gs])

    def ad(self, y):
        y = np.asarray(y, dtype=np.float64)
        forcings = []
        for n in range(NSLOTS):
            f = {k: np.zeros(s) for k, s in _shapes().items()}
            flat = np.zeros(NX * NY * NZ)
            np.add.at(flat, self.idx, y[n * self.m:(n + 1) * self.m])
            f["t_2"] = flat.reshape(NX, NY, NZ)
            forcings.append(f)
        a = self.model.ad(forcings)
        return {"t": a["t_2"]}


def main() -> None:
    win = _case()
    start = {n: cp.array(win._start[n]) for n in win.inputs}

    def run(x0):
        """The nonlinear run from ``x0``: every step's state (the window's checkpoints) and the end."""
        end = win.forward(x0)
        return [win._checkpoints[k] for k in range(win.nsteps)] + [end]

    def theta(states, n):
        return w4.from_window(cp.asnumpy(states[n * SPS]["thp"])).ravel()

    def perturbed(x0, t):
        x = {k: cp.array(v) for k, v in x0.items()}
        x["thp"] = x["thp"] + cp.asarray(w4.to_window(t))
        return x

    rng = np.random.default_rng(11)
    ii, jj, kk = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing="ij")
    bubble = 3.0 * np.exp(-(((ii - 5) / 2.5) ** 2 + ((jj - 6) / 2.5) ** 2 + ((kk - 3) / 2.0) ** 2))
    idx = np.sort(rng.choice(NX * NY * NZ, size=40, replace=False))
    truth = run(perturbed(start, bubble))
    y_obs = np.concatenate([theta(truth, n)[idx] for n in range(NSLOTS)])

    model = w4.WindowModel(win, SPS)

    def jo_nl(states):
        d = y_obs - np.concatenate([theta(states, n)[idx] for n in range(NSLOTS)])
        return 0.5 * float(np.sum((d / SIGMA_O) ** 2)), d

    taylor, dots, starts = [], [], []

    def setup(it, trajectory):
        x0, states = trajectory.slots
        if it == 1:
            model.relinearise(trajectory)   # loop 1's base: the background's run, as an external trajectory
        _, d = jo_nl(states)
        cost = VarCost(ToyU(), PointTerm(model, idx, d))
        taylor.append(check_gradient(cost, rng.uniform(-0.5, 0.5, cost.cv_size), pdx=1e-6, itertest=6))
        g0 = {k: rng.standard_normal(s) for k, s in _shapes().items()}
        fs = [{k: rng.standard_normal(s) for k, s in _shapes().items()} for _ in range(NSLOTS)]
        lhs = sum(float(np.sum(g[k] * f[k])) for g, f in zip(model.tl(g0, NSLOTS), fs) for k in t4.TL_KEYS)
        a = model.ad(fs)
        rhs = sum(float(np.sum(g0[k] * a[k])) for k in t4.TL_KEYS)
        dots.append((lhs, rhs))
        starts.append(x0)
        return cost, x0

    def update(it, x0, xhat):
        return perturbed(x0, SIGMA_B * np.asarray(xhat).reshape(NX, NY, NZ))

    calls = []

    def nl_model(it, x0):
        """The rerun from the updated analysis: its slots (here the start and every step) and its states."""
        calls.append(it)
        states = run(x0)
        return Trajectory(slots=(x0, states), states=states)

    background = run(start)
    first = Trajectory(slots=(start, background), states=background)
    results, cvt, final, _ = outer_loops(setup, update, first, max_ext_its=2, nl_model=nl_model,
                                         model=model, ntmax=40, eps=1e-4)
    cvt = np.asarray(cvt)
    jo_b, _ = jo_nl(background)
    a1 = update(1, start, results[0].xhat)
    jo_1, _ = jo_nl(run(a1))
    jo_2, _ = jo_nl(run(final))
    cv1 = np.asarray(results[0].xhat)
    j_nl = [jo_b, 0.5 * float(cv1 @ cv1) + jo_1, 0.5 * float(cvt @ cvt) + jo_2]
    # the increments accumulate: the final start is the background plus U(xhat1) + U(xhat2)
    acc = float(np.max(np.abs(w4.from_window(cp.asnumpy(final["thp"] - start["thp"]))
                              - SIGMA_B * (cv1 + np.asarray(results[1].xhat)).reshape(NX, NY, NZ))))
    cvt_sum = bool(np.array_equal(cvt, cv1 + np.asarray(results[1].xhat)))
    # TL against finite differences of the nonlinear run, about loop 2's base
    dt = rng.standard_normal((NX, NY, NZ))
    g0 = {k: np.zeros(s) for k, s in _shapes().items()}
    g0["t_2"] = dt
    base = run(a1)
    model.relinearise(Trajectory(slots=(a1, base), states=base))
    tl = [g["t_2"] for g in model.tl(g0, NSLOTS)]
    fd = []
    for eps in (1e-3, 1e-5):
        pert = run(perturbed(a1, eps * dt))
        err = max(float(np.max(np.abs((theta(pert, n) - theta(base, n)) / eps - tl[n].ravel())))
                  / float(np.max(np.abs(tl[n]))) for n in range(1, NSLOTS))
        fd.append(err)
    print("RESULT " + json.dumps(dict(
        j=[[r.j[0], r.j_final, r.iterations] for r in results], j_nl=j_nl, calls=calls,
        taylor=[[t["digits"], t["best"]] for t in taylor], dots=dots, fd=fd, acc=acc, cvt_sum=cvt_sum)))


if __name__ == "__main__":
    main()
