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
- a window whose length does not fit the slots, fewer than 2 slots, a
  binary32 window (its adjoint is a transpose to only ~1e-6), or a moisture
  option the dry window cannot honour, running anyway;
- the optional base check (the window's base against the slots' first
  guesses) passing a mismatch or comparing the wrong state;
- a checkpoint policy that changes the gradient, keeps the wrong states, or
  reruns an outer loop's base from the window's first start instead of the
  loop's (``Trajectory.start``).
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
from woof.grad.wrfplus.model import SPECIES, STEP_INPUTS

from woof.grad.wrfplus import model as _M  # noqa: E402

HALO = getattr(_M, "HALO", "mu_halo")   # model.HALO on a specified domain (#743)


@pytest.fixture(autouse=True)
def _specified_api(monkeypatch):
    """The fake's halo is model.HALO, and ``wrfda_tl_start(..., specified=)``
    is #743's (a stand-in with its semantics where main's lacks it)."""
    import inspect

    from woof.grad.wrfplus import window as W

    monkeypatch.setattr(w4, "_halo_name", lambda: HALO)
    if "specified" not in inspect.signature(W.wrfda_tl_start).parameters:
        base, base_ad = W.wrfda_tl_start, W.wrfda_tl_start_ad

        def start(g, *, specified=False):
            out = base(g)
            return {**out, HALO: g["mup"]} if specified else out

        def start_ad(adj, *, specified=False):
            out = base_ad(adj)
            if specified:
                out["mup"] = out["mup"] + adj[HALO]
            return out

        monkeypatch.setattr(W, "wrfda_tl_start", start)
        monkeypatch.setattr(W, "wrfda_tl_start_ad", start_ad)

ROOT = Path(__file__).resolve().parents[1]


class FakeWindow:
    """A linear stand-in for :class:`woof.grad.wrfplus.window.Window`
    (numpy, its names and shapes): each step rolls every field along ``i``
    and feeds ``p_pert`` into ``p`` and ``thp`` and ``muts`` into ``mup``.
    With ``lbc`` it is a specified domain's: boundary tables (``params``,
    width 2, coupled-field shapes) that add into ``thp``'s first column
    each step, and the constants :func:`window4d.lbc_tl` reads."""

    def __init__(self, nx, ny, nz, nsteps, dtype=np.float64, every=1, halo=False, moist=False, lbc=False):
        mass, two = (nz, ny, nx), (ny, nx)
        shapes = {"u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx), "w": (nz + 1, ny, nx), "php": (nz + 1, ny, nx),
                  "thp": mass, "p": mass, "al": mass, "alt": mass, "p_pert": mass, "mup": two, "muts": two,
                  HALO: two, "qv": mass, "qc": mass, "qr": mass}
        names = STEP_INPUTS + (SPECIES if moist else ()) + ((HALO,) if halo else ())
        if halo or moist:   # Window.inputs: a moist state's species, a specified domain's halo
            self.inputs = names
        self._start = {n: np.ones(shapes[n], dtype=dtype) for n in names}
        self._names = names
        self.params = ()
        if lbc:
            self.inputs = names
            rng = np.random.default_rng(11)
            for n in names:   # a base with structure (the coupling's products)
                self._start[n] = 1.0 + 0.1 * rng.standard_normal(shapes[n])
            self._start.update(mub2d=9.0 + rng.random(two), msfu=1.0 + 0.1 * rng.random((ny, nx + 1)),
                               msfv=1.0 + 0.1 * rng.random((ny + 1, nx)))
            fields = {"u": "u", "v": "v", "theta": "thp", "phi": "php", "mu": "mup", **({"qv": "qv"} if moist else {})}
            params = []
            for f, n in fields.items():
                full = np.zeros((1,) + two if f == "mu" else shapes[n])
                for side in w4._SIDES:
                    t = np.ascontiguousarray(w4._side(full, side, 2))
                    self._start[f"lbc_{f}_{side}"] = t + 50.0
                    self._start[f"lbc_{f}_{side}_t"] = t - 0.01
                    params += [f"lbc_{f}_{side}", f"lbc_{f}_{side}_t"]
            self.params = tuple(params)
            self.lbc_seconds = 600.0
        self.nsteps = nsteps
        self.every = every
        self._checkpoints = None
        self.forwards = 0

    def forward(self, x0=None, *, trajectory=None):
        """The base: the start plus ``k`` in every field at state ``k`` (or
        ``trajectory``'s states, as ``Window.forward(trajectory=)``)."""
        if trajectory is not None:
            return self.use_trajectory(list(trajectory))
        self.forwards += 1
        self._external = False
        x0 = self._start if x0 is None else x0
        self._checkpoints = {k: {n: x0[n] + k for n in self._names}
                             for k in range(0, self.nsteps, self.every)}
        return {n: x0[n] + self.nsteps for n in self._names}

    def use_trajectory(self, states):
        """``forward(trajectory=states)`` as #746 keeps it: every state but
        the last, ``_external`` set; returns the last."""
        self._checkpoints = {k: st for k, st in enumerate(states[:self.nsteps])}
        self._external = True
        return states[self.nsteps]

    @staticmethod
    def _step(d):
        d = {n: v + 0.1 * np.roll(v, 1, axis=-1) for n, v in d.items()}
        d["p"] = d["p"] + 0.3 * d["p_pert"]
        d["mup"] = d["mup"] + 0.2 * d["muts"]
        d["thp"] = d["thp"] + 0.05 * d["p_pert"]
        if "qv" in d:   # qv feels theta, and qc carries qv
            d["qv"] = d["qv"] + 0.07 * d["thp"]
            d["qc"] = d["qc"] + 0.4 * d["qv"]
        if HALO in d:
            d[HALO] = d[HALO] + 0.5 * d["mup"]
        return d

    @staticmethod
    def _step_ad(a):
        a = dict(a)
        if HALO in a:
            a["mup"] = a["mup"] + 0.5 * a[HALO]
        if "qv" in a:
            a["qv"] = a["qv"] + 0.4 * a["qc"]
            a["thp"] = a["thp"] + 0.07 * a["qv"]
        a["p_pert"] = a["p_pert"] + 0.05 * a["thp"]
        a["muts"] = a["muts"] + 0.2 * a["mup"]
        a["p_pert"] = a["p_pert"] + 0.3 * a["p"]
        return {n: v + 0.1 * np.roll(v, -1, axis=-1) for n, v in a.items()}

    def _push(self, dx, tables):
        """The tables' share of a step: table ``j`` adds ``0.01 (j + 1) sum``
        into thp's first column."""
        for j, n in enumerate(self.params):
            if n in tables:
                dx["thp"][..., 0] += 0.01 * (j + 1) * float(np.sum(tables[n]))

    def tl(self, dx0, save_at=(), *, partial=False):
        assert partial and set(dx0) <= set(self._names) | set(self.params)
        for n in set(dx0) & set(self.params):
            assert dx0[n].shape == self._start[n].shape, n
        self.last_dx0 = dx0
        dx = {n: np.array(dx0[n]) if n in dx0 else np.zeros_like(self._start[n]) for n in self._names}
        out = {0: dx} if 0 in save_at else {}
        for k in range(self.nsteps):
            dx = self._step(dx)
            self._push(dx, dx0)
            if k + 1 in save_at or k + 1 == self.nsteps:
                out[k + 1] = dx
        return out

    def ad(self, ax_final, forcing):
        assert ax_final is None and self._checkpoints is not None
        for f in forcing.values():
            assert set(f) == set(self._names)
            assert all(f[n].shape == self._start[n].shape for n in self._names)
        ax = {n: np.array(v) for n, v in forcing.get(self.nsteps, {}).items()} or \
            {n: np.zeros_like(self._start[n]) for n in self._names}
        at = {n: np.zeros_like(self._start[n]) for n in self.params}
        for k in reversed(range(self.nsteps)):
            s = float(np.sum(ax["thp"][..., 0]))
            for j, n in enumerate(self.params):
                at[n] += 0.01 * (j + 1) * s
            ax = self._step_ad(ax)
            if k in forcing:
                ax = {n: ax[n] + forcing[k][n] for n in ax}
        return {**ax, **at}


def _fake(nx=5, ny=4, nz=3, sps=2, nslots=3, halo=False, moist=False, **kw):
    win = FakeWindow(nx, ny, nz, sps * (nslots - 1), halo=halo, moist=moist)
    return w4.WindowModel(win, sps, to_device=np.asarray, to_host=np.asarray, **kw), (nx, ny, nz)


def _tl_state(rng, nx, ny, nz):
    shapes = {"u_2": (nx + 1, ny, nz), "v_2": (nx, ny + 1, nz), "w_2": (nx, ny, nz + 1),
              "ph_2": (nx, ny, nz + 1), "t_2": (nx, ny, nz), "qv": (nx, ny, nz), "p": (nx, ny, nz),
              "mu_2": (nx, ny)}
    return {k: rng.normal(size=s) for k, s in shapes.items()}


def _dot(a, b):
    return sum(float(np.sum(a[k] * b[k])) for k in t4.TL_KEYS)


@pytest.mark.parametrize("halo,halo_start,moist", [(False, "wrfplus", False), (True, "wrfplus", False),
                                                   (True, "carried", False), (False, "wrfplus", True),
                                                   (True, "wrfplus", True)])
def test_fake_window_dot_product(halo, halo_start, moist):
    """With ``halo`` the window is a specified domain's: its inputs add the
    carried mu halo (fed from ``mup``), which the adjoint must force too,
    and under ``wrfplus`` the TL start seeds it from ``g_mu``.  With
    ``moist`` the window carries qv, qc and qr: g_qv starts the window's qv
    and reads back from it (``moisture="advect"``, the default there)."""
    model, dims = _fake(halo=halo, halo_start=halo_start, moist=moist)
    assert model.moisture == ("advect" if moist else "persist")
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
    if moist:   # g_qv moved with the window, and felt theta
        assert not np.array_equal(gs[1]["qv"], g0["qv"])


def _lbc_case(rng, nx=7, ny=6, nz=4, qv=True):
    two, mass = (ny, nx), (nz, ny, nx)
    base = {"mup": rng.standard_normal(two) * 50.0, "u": rng.standard_normal((nz, ny, nx + 1)),
            "v": rng.standard_normal((nz, ny + 1, nx)), "thp": rng.standard_normal(mass),
            "php": rng.standard_normal((nz + 1, ny, nx)), "qv": rng.random(mass) * 1e-2}
    consts = {"mub2d": 9.0e4 + 1e3 * rng.random(two), "msfu": 1.0 + 0.1 * rng.random((ny, nx + 1)),
              "msfv": 1.0 + 0.1 * rng.random((ny + 1, nx))}
    g = {k: rng.standard_normal(v.shape) for k, v in base.items()}
    fields = ("u", "v", "theta", "phi", "mu") + (("qv",) if qv else ())
    return base, consts, g, fields


@pytest.mark.parametrize("qv", [False, True])
def test_lbc_tl_ad_dot_product(qv):
    """lbc_ad is lbc_tl's transpose: <TL(g), a> == <g, AD(a)> (WRFDA's
    da_transfer_wrftl_lbc_t0 and its adjoint, da_transfer_wrftl_lbc_t0_adj)."""
    rng = np.random.default_rng(21)
    base, consts, g, fields = _lbc_case(rng, qv=qv)
    t = w4.lbc_tl(g, base, consts, fields, 3, 21600.0)
    assert len(t) == 8 * len(fields)
    for n, v in t.items():   # the tendency takes the value to zero over the interval
        if not n.endswith("_t"):
            np.testing.assert_allclose(t[n + "_t"] * 21600.0, -v, rtol=1e-15)
    a = {n: rng.standard_normal(v.shape) for n, v in t.items()}
    ga = w4.lbc_ad(a, base, consts, fields, 3, 21600.0)
    assert set(ga) == {"mup", "u", "v", "thp", "php"} | ({"qv"} if qv else set())
    lhs = sum(float(np.sum(t[n] * a[n])) for n in t)
    rhs = sum(float(np.sum(g[k] * ga[k])) for k in ga)
    assert abs(lhs) > 1.0
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_lbc_tl_is_the_table_coupling_linearised():
    """lbc_tl's value tables are the derivative of woof's own table coupling
    (lateral_bc's domain_boundary_snapshot, non-hybrid: c1 = 1, c2 = 0) cut
    as extract_lateral_side cuts it: the coupling is bilinear, so the
    central difference is exact to rounding."""
    import types

    from woof.ingest import lateral_bc as LB

    rng = np.random.default_rng(22)
    base, consts, g, fields = _lbc_case(rng)
    nz = base["thp"].shape[0]

    def tables(sign):
        f = {k: base[k] + sign * g[k] for k in base}
        mu = consts["mub2d"] + f["mup"]
        st = types.SimpleNamespace(
            total_mu=lambda: mu, c1h=np.ones(nz), c2h=np.zeros(nz), c1f=np.ones(nz + 1), c2f=np.zeros(nz + 1),
            u=f["u"], v=f["v"], has_msf=True, msfu=consts["msfu"], msfv=consts["msfv"], use_theta_m=0,
            total_theta=lambda: f["thp"] + 300.0, php=f["php"], mup=f["mup"], qv=f["qv"],
            _external_scalar_boundary_fields=("qv",))
        snap = dict(LB._coupled_device_field_items(st))
        return {f"lbc_{n}_{side}": np.asarray(v, dtype=np.float64)
                for side in w4._SIDES for n, v in LB.extract_lateral_side(snap, side, 3).items()}

    plus, minus = tables(1.0), tables(-1.0)
    t = w4.lbc_tl(g, base, consts, fields, 3, 21600.0)
    for n in plus:
        fd = 0.5 * (plus[n] - minus[n])
        scale = float(np.max(np.abs(plus[n]))) or 1.0
        assert float(np.max(np.abs(t[n] - fd))) <= 1e-12 * scale, n


@pytest.mark.parametrize("moist,halo", [(False, False), (False, True), (True, True)])
def test_fake_window_lbc_dot_product(moist, halo):
    """A specified window (boundary tables in ``params``): ``lbc="wrfda"``
    (the default there) passes lbc_tl's tables in the TL start and adds their
    adjoint, and the whole map stays a transpose; ``fixed`` passes none."""
    nx, ny, nz, sps = 5, 4, 3, 2
    win = FakeWindow(nx, ny, nz, 2 * sps, halo=halo, moist=moist, lbc=True)
    model = w4.WindowModel(win, sps, to_device=np.asarray, to_host=np.asarray)
    assert model.lbc == "wrfda"
    rng = np.random.default_rng(23)
    g0 = _tl_state(rng, nx, ny, nz)
    fs = [_tl_state(rng, nx, ny, nz) for _ in range(3)]
    gs = model.tl(g0, 3)
    assert set(win.params) <= set(win.last_dx0)
    inc = model.boundary_increment(g0)
    assert set(inc) == set(win.params)
    for n, v in inc.items():
        np.testing.assert_array_equal(win.last_dx0[n], v)
    lhs = sum(_dot(g, f) for g, f in zip(gs, fs))
    rhs = _dot(g0, model.ad(fs))
    assert abs(lhs) > 1.0
    assert lhs == pytest.approx(rhs, rel=1e-12)
    fixed = w4.WindowModel(win, sps, to_device=np.asarray, to_host=np.asarray, lbc="fixed")
    gf = fixed.tl(g0, 3)
    assert not set(win.params) & set(win.last_dx0) and fixed.boundary_increment(g0) == {}
    assert not np.array_equal(gf[2]["t_2"], gs[2]["t_2"])
    assert _dot(g0, fixed.ad(fs)) == pytest.approx(sum(_dot(g, f) for g, f in zip(gf, fs)), rel=1e-12)
    # the outer loop's update: the increment lands on the window's tables
    before = {n: win._start[n].copy() for n in win.params}
    model.add_boundary_increment(inc)
    for n in win.params:
        np.testing.assert_allclose(win._start[n], before[n] + inc[n], rtol=1e-15)
    with pytest.raises(ValueError, match="no boundary table"):
        model.add_boundary_increment({"lbc_w_west": inc["lbc_u_west"]})


def test_lbc_refusals():
    model, dims = _fake()
    assert model.lbc is None and model.boundary_increment(_tl_state(np.random.default_rng(0), *dims)) == {}
    with pytest.raises(ValueError, match="no boundary tables"):
        _fake(lbc="wrfda")
    win = FakeWindow(5, 4, 3, 4, lbc=True)
    with pytest.raises(ValueError, match="lbc must be one of"):
        w4.WindowModel(win, 2, to_device=np.asarray, to_host=np.asarray, lbc="var4d_lbc")


def test_moist_window_advects_qv_and_starts_qc_qr_at_zero():
    model, dims = _fake(moist=True)
    g0 = _tl_state(np.random.default_rng(3), *dims)
    gs = model.tl(g0, 3)
    np.testing.assert_array_equal(gs[0]["qv"], g0["qv"])
    dx0 = model.window.last_dx0
    assert "qv" in dx0 and "qc" not in dx0 and "qr" not in dx0
    # one step: qv rolled, plus theta's share
    r = lambda a: a + 0.1 * np.roll(a, 1, axis=-1)
    model.window.nsteps, model.sps = 1, 1
    g1 = model.tl(g0, 2)[1]
    p0, t0, q0 = (w4.to_window(g0[k]) for k in ("p", "t_2", "qv"))
    th1 = r(t0) + 0.05 * r(p0)
    np.testing.assert_allclose(g1["qv"], w4.from_window(r(q0) + 0.07 * th1), rtol=1e-14)


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
    with pytest.raises(ValueError, match="at least 2 slots"):
        model.tl(g0, 1)
    with pytest.raises(ValueError, match="at least 2 slots"):
        model.ad([g0])
    with pytest.raises(TypeError, match="binary64 window"):
        w4.WindowModel(FakeWindow(*dims, 4, dtype=np.float32), 2, to_device=np.asarray, to_host=np.asarray)
    with pytest.raises(ValueError, match="lacks"):
        model.tl({k: v for k, v in g0.items() if k != "qv"}, 3)
    with pytest.raises(NotImplementedError, match="dry"):
        w4.WindowModel(model.window, 2, moisture="advect", to_device=np.asarray, to_host=np.asarray)
    with pytest.raises(ValueError, match="moist window"):
        w4.WindowModel(FakeWindow(*dims, 4, moist=True), 2, moisture="persist", to_device=np.asarray,
                       to_host=np.asarray)
    with pytest.raises(ValueError, match="moisture must be one of"):
        w4.WindowModel(model.window, 2, moisture="hold", to_device=np.asarray, to_host=np.asarray)
    with pytest.raises(ValueError, match="steps_per_slot"):
        w4.WindowModel(model.window, 0, to_device=np.asarray, to_host=np.asarray)


def _grids(nx, ny, nz, nslots, sps):
    """Each slot's ``grid`` equal to the fake window's base there."""
    shapes = {k: v.shape for k, v in _tl_state(np.random.default_rng(0), nx, ny, nz).items()}
    return [(None, {g: np.full(shapes[g], 1.0 + n * sps) for g in w4._BASE}) for n in range(nslots)]


def test_base_check():
    """Off by default; with ``base=`` the window's base at each slot is the
    slot's ``grid`` or the model is refused, by slot and name."""
    nx, ny, nz = 5, 4, 3
    mk = lambda **kw: w4.WindowModel(FakeWindow(nx, ny, nz, 4, every=kw.pop("every", 1)), 2,
                                      to_device=np.asarray, to_host=np.asarray, **kw)
    assert mk().base_diff is None
    slots = _grids(nx, ny, nz, 3, 2)
    m = mk(base=slots)
    assert len(m.base_diff) == 3 * len(w4._BASE) and max(m.base_diff.values()) == 0.0
    assert mk(base=slots, every=2).base_diff == m.base_diff
    slots[1][1]["t_2"] = slots[1][1]["t_2"] * 1.01
    with pytest.raises(ValueError, match="two trajectories.*slot 1 t_2"):
        mk(base=slots)
    assert mk(base=slots, base_rtol=0.02).base_diff[(1, "t_2")] == pytest.approx(0.01 / 1.01)
    with pytest.raises(ValueError, match="checkpoint_every=3"):
        mk(base=_grids(nx, ny, nz, 3, 2), every=3)


def test_halo_start():
    """``wrfplus`` (default): the primal halo starts as ``mup`` (WRFPLUS's
    start_domain resets it to the edge row), the base reruns, and the TL
    halo starts as ``g_mu``; ``carried`` keeps the window's halo and starts
    the TL halo at zero.  A periodic window has no halo to start."""
    nx, ny, nz = 5, 4, 3
    g0 = _tl_state(np.random.default_rng(4), nx, ny, nz)
    for mode in ("wrfplus", "carried"):
        win = FakeWindow(nx, ny, nz, 4, halo=True)
        win._start[HALO][...] = 7.0          # a carried halo, not mup
        win.forward()
        model = w4.WindowModel(win, 2, to_device=np.asarray, to_host=np.asarray, halo_start=mode)
        model.tl(g0, 3)
        if mode == "wrfplus":
            np.testing.assert_array_equal(win._start[HALO], win._start["mup"])
            assert win.forwards == 2 and win._checkpoints[0][HALO][0, 0] == 1.0
            np.testing.assert_array_equal(win.last_dx0[HALO], w4.to_window(g0["mu_2"]))
        else:
            assert (win._start[HALO] == 7.0).all() and win.forwards == 1
            assert HALO not in win.last_dx0
    model, _ = _fake()
    assert model.halo is None
    with pytest.raises(ValueError, match="halo_start"):
        _fake(halo_start="edge")


def test_halo_start_on_an_external_trajectory():
    nx, ny, nz = 5, 4, 3
    win = FakeWindow(nx, ny, nz, 4, halo=True)
    states = [{n: v + 1.0 * k for n, v in win._start.items()} for k in range(5)]
    states[0][HALO] = states[0][HALO] + 3.0
    win.use_trajectory(states)
    with pytest.raises(ValueError, match="carries a mu halo that is not mup"):
        w4.WindowModel(win, 2, to_device=np.asarray, to_host=np.asarray)
    w4.WindowModel(win, 2, to_device=np.asarray, to_host=np.asarray, halo_start="carried")
    states[0][HALO] = states[0]["mup"].copy()
    w4.WindowModel(win, 2, to_device=np.asarray, to_host=np.asarray)


def test_base_check_reads_an_external_trajectory():
    """On ``forward(trajectory=...)`` (#746) the check reads the given
    states and never reruns ``forward`` (which would replace them); the
    last slot needs ``end``."""
    nx, ny, nz = 5, 4, 3
    win = FakeWindow(nx, ny, nz, 4)
    model = w4.WindowModel(win, 2, to_device=np.asarray, to_host=np.asarray)
    states = [{n: v + 10.0 * k for n, v in win._start.items()} for k in range(5)]
    end = win.use_trajectory(states)
    slots = [(None, {g: np.full(sh, 1.0 + 20.0 * n) for g, sh in
                     ((g, a.shape) for g, a in _grids(nx, ny, nz, 1, 0)[0][1].items())}) for n in range(3)]
    n0 = win.forwards
    with pytest.raises(ValueError, match="pass end="):
        model.check_base(slots)
    assert max(model.check_base(slots, end=end).values()) == 0.0
    assert win.forwards == n0 and win._checkpoints[1]["u"][0, 0, 0] == 11.0
    with pytest.raises(ValueError, match="slot 2"):
        model.check_base(slots, end=states[3])
    assert w4.WindowModel(win, 2, to_device=np.asarray, to_host=np.asarray, base=slots,
                          base_end=end).base_diff[(2, "u_2")] == 0.0


def test_checkpoint_policy():
    """``all`` keeps every step's state, ``slot`` one per slot, an int ``c``
    one every ``c`` steps; an already-forwarded window is re-kept from its
    own state 0; the TL and AD do not change; refused values are named."""
    nx, ny, nz = 5, 4, 3
    rng = np.random.default_rng(9)
    g0 = _tl_state(rng, nx, ny, nz)
    fs = [_tl_state(rng, nx, ny, nz) for _ in range(3)]
    ref = None
    for policy, every, kept in (("all", 1, [0, 1, 2, 3]), ("slot", 2, [0, 2]), (2, 2, [0, 2]), (None, 1, None)):
        win = FakeWindow(nx, ny, nz, 4)
        x0 = {n: v + 5.0 for n, v in win._start.items()}
        win.forward(x0)
        model = w4.WindowModel(win, 2, to_device=np.asarray, to_host=np.asarray, checkpoint=policy)
        assert win.every == every and model.checkpoint == policy
        assert win.forwards == (2 if every != 1 else 1)
        assert win._checkpoints[0]["u"][0, 0, 0] == 6.0   # re-kept from x0, not from _start
        if kept is not None:
            assert sorted(win._checkpoints) == kept
        rep = model.checkpoint_report()
        assert rep["every"] == every and rep["stored_states"] == len(win._checkpoints)
        assert rep["stored_bytes"] == rep["stored_states"] * rep["state_bytes"] > 0
        assert rep["peak_states"] == rep["stored_states"] + every - 1
        assert rep["ad_rerun_steps"] == 4 - rep["stored_states"]
        out = (model.tl(g0, 3), model.ad(fs))
        if ref is None:
            ref = out
        for a, b in zip(out[0] + [out[1]], ref[0] + [ref[1]]):
            for k in a:
                np.testing.assert_array_equal(a[k], b[k])
    for bad, msg in ((3, "divide steps_per_slot=2"), (0, ">= 1"), ("sqrt", "one of"), (True, "one of"),
                     (1.0, "one of")):
        with pytest.raises(ValueError, match=msg):
            w4.WindowModel(FakeWindow(nx, ny, nz, 4), 2, to_device=np.asarray, to_host=np.asarray,
                           checkpoint=bad)


def test_relinearise_from_start():
    """``Trajectory.start``: the window reruns its own base from the loop's
    start under the policy (not every state held, not the first loop's
    start); ``states`` holds every state; one of the two, not both; the
    base check then reruns from that start too."""
    from woof.da.var.outer4d import Trajectory

    nx, ny, nz = 5, 4, 3
    win = FakeWindow(nx, ny, nz, 4)
    model = w4.WindowModel(win, 2, to_device=np.asarray, to_host=np.asarray, checkpoint="slot")
    start = {n: v + 20.0 for n, v in win._start.items()}
    end = model.relinearise(Trajectory(slots=[], start=start))
    assert not win._external and sorted(win._checkpoints) == [0, 2]
    assert win._checkpoints[2]["u"][0, 0, 0] == 23.0 and end["u"][0, 0, 0] == 25.0
    rep = model.checkpoint_report()
    assert rep["ad_rerun_steps"] == 2 and not rep["external"]
    slots = [(None, {g: np.full(a.shape, 21.0 + 2 * n) for g, a in _grids(nx, ny, nz, 1, 0)[0][1].items()})
             for n in range(3)]
    assert max(model.check_base(slots).values()) == 0.0
    assert win._checkpoints[0]["u"][0, 0, 0] == 21.0
    states = [{n: v + 1.0 * k for n, v in win._start.items()} for k in range(5)]
    model.relinearise(Trajectory(slots=[], states=states))
    rep = model.checkpoint_report()
    assert rep["external"] and rep["every"] == 1 and rep["stored_states"] == 4 and rep["ad_rerun_steps"] == 0
    with pytest.raises(ValueError, match="pass one"):
        model.relinearise(Trajectory(slots=[], states=states, start=start))
    with pytest.raises(ValueError, match="Trajectory.start"):
        model.relinearise(Trajectory(slots=[]))
    hwin = FakeWindow(nx, ny, nz, 4, halo=True)
    hmodel = w4.WindowModel(hwin, 2, to_device=np.asarray, to_host=np.asarray)
    bad = {n: v.copy() for n, v in hwin._start.items()}
    bad[HALO] = bad[HALO] + 3.0
    with pytest.raises(ValueError, match="carries a mu halo that is not mup"):
        hmodel.relinearise(Trajectory(slots=[], start=bad))


def test_halo_start_keeps_a_forwarded_start():
    """A window already forwarded from a state other than its first (a
    relinearised start) keeps that state under ``halo_start="wrfplus"``:
    the base reruns from it, halo reset to mup, not from the window's first."""
    nx, ny, nz = 5, 4, 3
    win = FakeWindow(nx, ny, nz, 4, halo=True)
    x0 = {n: v + 5.0 for n, v in win._start.items()}
    x0[HALO] = x0[HALO] + 3.0
    win.forward(x0)
    w4.WindowModel(win, 2, to_device=np.asarray, to_host=np.asarray, checkpoint="slot")
    ck = win._checkpoints[0]
    assert ck["u"][0, 0, 0] == 6.0 and np.array_equal(ck[HALO], ck["mup"])


def test_relinearise_from_start_moves_the_table_base():
    """After ``relinearise(start=)`` the boundary tables' TL map couples about
    the new state 0, and the rerun reads tables ``add_boundary_increment`` changed."""
    from woof.da.var.outer4d import Trajectory

    nx, ny, nz = 7, 6, 4
    win = FakeWindow(nx, ny, nz, 4, lbc=True, halo=True, moist=True)
    model = w4.WindowModel(win, 2, to_device=np.asarray, to_host=np.asarray, checkpoint="slot")
    before = model._lbc_args()[0]["u"].copy()
    model.add_boundary_increment({"lbc_u_west": np.ones_like(win._start["lbc_u_west"])})
    tables = win._start["lbc_u_west"].copy()
    start = {n: win._checkpoints[0][n] + 2.0 for n in win._names}
    start[HALO] = start["mup"]
    model.relinearise(Trajectory(slots=[], start=start))
    np.testing.assert_array_equal(model._lbc_args()[0]["u"], before + 2.0)
    np.testing.assert_array_equal(win._start["lbc_u_west"], tables)


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
@pytest.mark.parametrize("moist", [False, True])
def test_real_window_dot_product(tmp_path, moist):
    """The real window (host backend, binary64, a warm ridge case): three
    slots two steps apart, ``sum_n <TL(g0)_n, f_n> == <g0, AD(f)>``.  Moist:
    qv, qc and qr (mp_physics 0), g_qv advected by the window."""
    env = dict(os.environ, WOOF_BACKEND="host", PYTHONPATH=str(ROOT), WOOF_TEST_MOIST=str(int(moist)),
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
import os
moist = os.environ["WOOF_TEST_MOIST"] == "1"
cfg = H.make_config(nx, ny, nz, h_sca_adv_order=5, terrain_opt=1,
                    **(dict(moist=True, moist_adv_opt=0) if moist else {}))
jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
ridge = 400.0 * np.exp(-(((ii - nx / 2) / (nx / 5)) ** 2 + ((jj - ny / 2) / (ny / 4)) ** 2))
geo = dataclasses.replace(H.neutral_geography(cfg), terrain=ridge)
state = H.make_state(cfg, 1, geography=geo)
D.step(state, cfg)
win = W.Window(state, cfg, 4)
model = w4.WindowModel(win, 2)
rng = np.random.default_rng(5)
names = {"u_2": "u", "v_2": "v", "w_2": "w", "t_2": "thp", "mu_2": "mup", "ph_2": "php", "p": "p_pert",
         "qv": "qv" if moist else "thp"}
assert model.moisture == ("advect" if moist else "persist")
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
moved = (not np.array_equal(gs[2]["qv"], g0["qv"])) if moist else True
print('RESULT ' + json.dumps(dict(lhs=lhs, rhs=rhs, same=same, moved=moved)))
"""
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    out = json.loads([ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1][7:])
    assert out["same"] and out["moved"]
    assert abs(out["lhs"]) > 0.0
    assert out["lhs"] == pytest.approx(out["rhs"], rel=1e-12)


@pytest.mark.skipif(not any(shutil.which(c) for c in ("clang++", "g++", "c++")), reason="no C++ compiler")
def test_real_window_checkpoint_policy_is_word_exact(tmp_path):
    """The real window (host backend, binary64, moist ridge case): the TL
    slots and the AD under ``slot`` equal those under ``all`` to the word,
    and a relinearisation from ``Trajectory.start`` equals one from the
    same run's ``Trajectory.states``."""
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
from woof.da.var import window4d as w4
from woof.da.var.outer4d import Trajectory
nx, ny, nz = 16, 12, 10
cfg = H.make_config(nx, ny, nz, h_sca_adv_order=5, terrain_opt=1, moist=True, moist_adv_opt=0)
jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
ridge = 400.0 * np.exp(-(((ii - nx / 2) / (nx / 5)) ** 2 + ((jj - ny / 2) / (ny / 4)) ** 2))
geo = dataclasses.replace(H.neutral_geography(cfg), terrain=ridge)
state = H.make_state(cfg, 1, geography=geo)
D.step(state, cfg)
rng = np.random.default_rng(5)
names = {"u_2": "u", "v_2": "v", "w_2": "w", "t_2": "thp", "mu_2": "mup", "ph_2": "php", "p": "p_pert",
         "qv": "qv"}
win = W.Window(state, cfg, 4)
def rnd():
    out = {}
    for k, n in names.items():
        base = cp.asnumpy(win._start[n])
        out[k] = rng.standard_normal(base.shape[::-1]) * float(np.abs(base).max() * 1e-3 + 1e-12)
    return out
g0, fs = rnd(), [rnd() for _ in range(3)]
def run(model):
    return model.tl(g0, 3), model.ad(fs)
def same(a, b):
    ta, aa = a
    tb, ab = b
    return all(np.array_equal(x[k], y[k]) for x, y in zip(ta + [aa], tb + [ab]) for k in x)
out = {}
ref = run(w4.WindowModel(win, 2, checkpoint="all"))
slot = w4.WindowModel(win, 2, checkpoint="slot")
out["slot_kept"] = sorted(win._checkpoints)
out["slot_same"] = same(run(slot), ref)
# an outer loop's rerun from a moved start: start= against states=
x0 = {n: win._start[n].copy() for n in win.inputs}
x0["thp"] += 0.3
x0["muts"][...] = win._start["mub2d"] + x0["mup"]
win.every = 1
end = win.forward(x0)
states = [win._checkpoints[k] for k in range(4)] + [end]
ext = w4.WindowModel(win, 2, checkpoint="slot")
ext.relinearise(Trajectory(slots=[], states=states))
a = run(ext)
ext.relinearise(Trajectory(slots=[], start=x0))
out["start_kept"] = sorted(win._checkpoints)
out["start_same"] = same(run(ext), a)
out["moved"] = not same(a, ref)
print('RESULT ' + json.dumps(out))
"""
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    out = json.loads([ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1][7:])
    assert out["slot_kept"] == [0, 2] and out["start_kept"] == [0, 2]
    assert out["slot_same"] and out["start_same"] and out["moved"]
