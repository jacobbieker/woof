"""P3a's window as the 4D-Var TL model (woof.da.var.window4d, PR 4D.3).

Breakage this prevents:

- a transpose or name slip between WRFDA's ``g_*`` (Fortran ``(i, j, k)``)
  and the window's ``(k, j, i)`` STEP_INPUTS, or a slot read at the wrong
  step: ``sum_n <TL(g0)_n, f_n> == <g0, AD(f)>`` with a fake linear window
  that couples the pressure and mass carriers, and with the real window
  (host backend, binary64);
- a start that is not WRFDA's (``g_p`` into ``p`` and ``p_pert``, ``g_mu``
  into ``mup`` and ``muts``): slot 0 reads back ``g0``;
- :class:`~woof.da.var.fourd.FourDObsTerm` over the window not being its
  own transpose;
- a window whose length does not fit the slots, or a moisture option the dry
  window cannot honour, running anyway.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from woof.da.var import fourd as f4
from woof.da.var import transfer4d as t4
from woof.da.var import window4d as w4
from woof.grad.wrfplus.model import STEP_INPUTS

ROOT = Path(__file__).resolve().parents[1]


class FakeWindow:
    """A linear stand-in for :class:`woof.grad.wrfplus.window.Window`
    (numpy, its names and shapes): each step rolls every field along ``i``
    and feeds ``p_pert`` into ``p`` and ``thp`` and ``muts`` into ``mup``."""

    def __init__(self, nx, ny, nz, nsteps):
        mass, two = (nz, ny, nx), (ny, nx)
        shapes = {"u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx), "w": (nz + 1, ny, nx), "php": (nz + 1, ny, nx),
                  "thp": mass, "p": mass, "al": mass, "alt": mass, "p_pert": mass, "mup": two, "muts": two}
        self._start = {n: np.ones(shapes[n]) for n in STEP_INPUTS}
        self.nsteps = nsteps
        self._checkpoints = None

    def forward(self):
        self._checkpoints = {0: True}

    @staticmethod
    def _step(d):
        d = {n: v + 0.1 * np.roll(v, 1, axis=-1) for n, v in d.items()}
        d["p"] = d["p"] + 0.3 * d["p_pert"]
        d["mup"] = d["mup"] + 0.2 * d["muts"]
        d["thp"] = d["thp"] + 0.05 * d["p_pert"]
        return d

    @staticmethod
    def _step_ad(a):
        a = dict(a)
        a["p_pert"] = a["p_pert"] + 0.05 * a["thp"]
        a["muts"] = a["muts"] + 0.2 * a["mup"]
        a["p_pert"] = a["p_pert"] + 0.3 * a["p"]
        return {n: v + 0.1 * np.roll(v, -1, axis=-1) for n, v in a.items()}

    def tl(self, dx0, save_at=(), *, partial=False):
        assert partial and set(dx0) <= set(STEP_INPUTS)
        dx = {n: np.array(dx0[n]) if n in dx0 else np.zeros_like(v) for n, v in self._start.items()}
        out = {0: dx} if 0 in save_at else {}
        for k in range(self.nsteps):
            dx = self._step(dx)
            if k + 1 in save_at or k + 1 == self.nsteps:
                out[k + 1] = dx
        return out

    def ad(self, ax_final, forcing):
        assert ax_final is None and self._checkpoints is not None
        for f in forcing.values():
            assert set(f) == set(STEP_INPUTS)
            assert all(f[n].shape == self._start[n].shape for n in STEP_INPUTS)
        ax = {n: np.array(v) for n, v in forcing.get(self.nsteps, {}).items()} or \
            {n: np.zeros_like(v) for n, v in self._start.items()}
        for k in reversed(range(self.nsteps)):
            ax = self._step_ad(ax)
            if k in forcing:
                ax = {n: ax[n] + forcing[k][n] for n in ax}
        return ax


def _fake(nx=5, ny=4, nz=3, sps=2, nslots=3):
    win = FakeWindow(nx, ny, nz, sps * (nslots - 1))
    return w4.WindowModel(win, sps, to_device=np.asarray, to_host=np.asarray), (nx, ny, nz)


def _tl_state(rng, nx, ny, nz):
    shapes = {"u_2": (nx + 1, ny, nz), "v_2": (nx, ny + 1, nz), "w_2": (nx, ny, nz + 1),
              "ph_2": (nx, ny, nz + 1), "t_2": (nx, ny, nz), "qv": (nx, ny, nz), "p": (nx, ny, nz),
              "mu_2": (nx, ny)}
    return {k: rng.normal(size=s) for k, s in shapes.items()}


def _dot(a, b):
    return sum(float(np.sum(a[k] * b[k])) for k in t4.TL_KEYS)


def test_fake_window_dot_product():
    model, dims = _fake()
    rng = np.random.default_rng(0)
    g0 = _tl_state(rng, *dims)
    fs = [_tl_state(rng, *dims) for _ in range(3)]
    gs = model.tl(g0, 3)
    lhs = sum(_dot(g, f) for g, f in zip(gs, fs))
    rhs = _dot(g0, model.ad(fs))
    assert abs(lhs) > 1.0
    assert lhs == pytest.approx(rhs, rel=1e-12)
    # the slots differ (the window moved them) and p felt p_pert's start
    assert not np.array_equal(gs[1]["p"], gs[2]["p"])


def test_slot_zero_is_wrfda_start_and_qv_persists():
    model, dims = _fake()
    g0 = _tl_state(np.random.default_rng(1), *dims)
    gs = model.tl(g0, 3)
    for k in t4.TL_KEYS:
        np.testing.assert_array_equal(gs[0][k], g0[k])
    for g in gs:
        np.testing.assert_array_equal(g["qv"], g0["qv"])
    # one step: g_p reads back from p_pert (rolled), and thp felt p_pert's
    # start (g_p), so the start copied g_p into p_pert
    r = lambda a: a + 0.1 * np.roll(a, 1, axis=-1)
    p0, t0 = w4.to_window(g0["p"]), w4.to_window(g0["t_2"])
    model.window.nsteps, model.sps = 1, 1
    g1 = model.tl(g0, 2)[1]
    np.testing.assert_allclose(g1["p"], w4.from_window(r(p0)), rtol=1e-14)
    np.testing.assert_allclose(g1["t_2"], w4.from_window(r(t0) + 0.05 * r(p0)), rtol=1e-14)

def test_refusals():
    model, dims = _fake()
    g0 = _tl_state(np.random.default_rng(2), *dims)
    with pytest.raises(ValueError, match="need a window of 2 steps"):
        model.tl(g0, 2)
    with pytest.raises(ValueError, match="lacks"):
        model.tl({k: v for k, v in g0.items() if k != "qv"}, 3)
    with pytest.raises(NotImplementedError, match="dry"):
        w4.WindowModel(model.window, 2, moisture="advect", to_device=np.asarray, to_host=np.asarray)
    with pytest.raises(ValueError, match="steps_per_slot"):
        w4.WindowModel(model.window, 0, to_device=np.asarray, to_host=np.asarray)


@pytest.mark.parametrize("press,ph", [("wrfda", "zero"), ("dnw", "hydrostatic")])
def test_fourd_term_over_the_window_is_its_transpose(press, ph):
    from test_var_fourd import _case, _x

    iv, slots, start = _case()
    nx, ny, nz = start[0]["t"].shape
    win = FakeWindow(nx, ny, nz, 3)
    model = w4.WindowModel(win, 3, to_device=np.asarray, to_host=np.asarray)
    term = f4.FourDObsTerm(iv, slots, start, model=model, press=press, ph=ph)
    rng = np.random.default_rng(3)
    x = _x(rng, start[0]["t"].shape)
    y = rng.normal(size=term.size)
    lhs = float(np.dot(term.tl(x), y))
    ax = term.ad(y)
    rhs = sum(float(np.sum(x[k] * ax[k])) for k in ("u", "v", "t", "q", "psfc"))
    assert abs(lhs) > 1.0
    assert lhs == pytest.approx(rhs, rel=1e-12)
    ident = f4.FourDObsTerm(iv, slots, start, press=press, ph=ph)
    assert not np.array_equal(term.tl(x), ident.tl(x))


STRICT = dict(GPUWM_WRF_EXACT="1", WOOF_WRF_EXACT_DIAGNOSTICS="1", WOOF_WRF_EXACT_ADVECTION="1",
              GPUWM_WRF_EXACT_DIFFUSION="1", GPUWM_WRF_EXACT_BIGSTEP="1")


@pytest.mark.skipif(not any(shutil.which(c) for c in ("clang++", "g++", "c++")), reason="no C++ compiler")
def test_real_window_dot_product(tmp_path):
    """The real window (host backend, binary64, a warm ridge case): three
    slots two steps apart, ``sum_n <TL(g0)_n, f_n> == <g0, AD(f)>``."""
    env = dict(os.environ, WOOF_BACKEND="host", PYTHONPATH=str(ROOT),
               WOOF_HOST_CACHE=str(tmp_path / "hostcache"), WOOF_HOST_THREADS="4", **STRICT)
    script = """
import dataclasses, json, numpy as np, woof
import cupy as cp
from tilestream import harness as H
if cp.__name__ != 'cupy' or getattr(cp, '__woof_facade__', False):
    import woof.core.resident_admission as RA
    RA.device_free_bytes = lambda: None
from woof.core import dycore as D
from woof.grad.wrfplus import window as W
from woof.da.var import transfer4d as t4, window4d as w4
nx, ny, nz = 16, 12, 10
cfg = H.make_config(nx, ny, nz, h_sca_adv_order=5, terrain_opt=1)
jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
ridge = 400.0 * np.exp(-(((ii - nx / 2) / (nx / 5)) ** 2 + ((jj - ny / 2) / (ny / 4)) ** 2))
geo = dataclasses.replace(H.neutral_geography(cfg), terrain=ridge)
state = H.make_state(cfg, 1, geography=geo)
D.step(state, cfg)
win = W.Window(state, cfg, 4)
model = w4.WindowModel(win, 2)
rng = np.random.default_rng(5)
names = {"u_2": "u", "v_2": "v", "w_2": "w", "t_2": "thp", "mu_2": "mup", "ph_2": "php", "p": "p_pert", "qv": "thp"}
def rnd():
    out = {}
    for k, n in names.items():
        base = cp.asnumpy(win._start[n])
        out[k] = rng.standard_normal(base.shape[::-1]) * float(np.abs(base).max() * 1e-3 + 1e-12)
    return out
g0, fs = rnd(), [rnd() for _ in range(3)]
gs = model.tl(g0, 3)
dot = lambda a, b: sum(float(np.sum(a[k] * b[k])) for k in t4.TL_KEYS)
lhs = sum(dot(g, f) for g, f in zip(gs, fs))
rhs = dot(g0, model.ad(fs))
same = all(np.array_equal(gs[0][k], g0[k]) for k in t4.TL_KEYS)
print('RESULT ' + json.dumps(dict(lhs=lhs, rhs=rhs, same=same)))
"""
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    out = json.loads([ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1][7:])
    assert out["same"]
    assert abs(out["lhs"]) > 0.0
    assert out["lhs"] == pytest.approx(out["rhs"], rel=1e-12)
