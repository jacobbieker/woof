"""WRFDA 4D-Var's observation term (woof.da.var.fourd, PR 4D.2).

Synthetic, no oracle data:

- :class:`FourDObsTerm`'s adjoint is its transpose (identity model, two
  slots with different first guesses, upper-air and surface types);
- with every slot's state the start's, the term is FGAT's ``H'`` applied
  to WRFDA's round trip through the TL state (``xatowrftl`` then
  ``wrftltoxa``, ``rho`` from ``xtoxa``), slot by slot;
- the Jo groups run over all slots in ``iv`` order;
- the identity model's adjoint is the sum of the forcings;
- ``analyse(var4d=...)`` refuses what it does not port before reading anything;
- on the tutorial first guess (``WOOF_WRFDA471_DIR``), ``xatowrf``'s
  ``var4d`` branch leaves ``P`` unchanged.

Against WRFDA itself (the 4D.0 identity lane, L-id) once that oracle exists.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import _var_wrfda_synthetic as syn
from woof.da.var import fgat as fm
from woof.da.var import fourd as f4
from woof.da.var import innov_wrfda as iw
from woof.da.var import obs_wrfda as ow
from woof.da.var import transfer4d as t4
from woof.da.var.analysis import KAPPA, T0

M = ow.MISSING_R
P0 = 100000.0


def _state(xb):
    """``(xb, grid)`` for the transfers from a synthetic background: an eta
    column and a WRF state consistent enough for linearisation tests."""
    nx, ny, nz = xb["t"].shape
    znw = np.linspace(1.0, 0.0, nz + 1) ** 1.3
    znu = 0.5 * (znw[:-1] + znw[1:])
    dnw = np.diff(znw)
    dn = np.concatenate([[0.0], np.diff(znu)])
    mub = xb["psfc"] - xb["ptop"] - 200.0
    mu_2 = np.full((nx, ny), 200.0)
    qv = xb["q"] / (1.0 - xb["q"])
    xb = dict(xb, dn=dn, dnw=dnw, psac=mub + mu_2, znu=znu, znw=znw)
    grid = {"qv": qv, "mu_2": mu_2, "mub": mub, "t_2": xb["t"] * (P0 / xb["p"]) ** KAPPA - T0, "dnw": dnw,
            "base_pres": P0, "p_top": xb["ptop"], "hypsometric_opt": 2,
            "c1h": np.ones(nz), "c2h": np.zeros(nz), "c3h": znu, "c4h": np.zeros(nz),
            "c1f": np.ones(nz + 1), "c2f": np.zeros(nz + 1), "c3f": znw, "c4f": np.zeros(nz + 1)}
    return xb, grid


_SFC = {"u": [(2.0, 0, 1.0)], "v": [(1.0, 0, 1.0)], "t": [(285.0, 0, 2.0)],
        "p": [(99000.0, 0, 100.0)], "q": [(0.005, 0, 0.001)]}


def _slot_obs(xb, shift):
    reports = []
    for i, j, dx, dy in [(2, 2, 0.3, 0.6), (3 + shift, 2, 0.9, 0.1), (6, 5 - shift, 0.2, 0.7)]:
        col = iw._hcol(xb["p"], i - 1, j - 1, dx, dy, 1.0 - dx, 1.0 - dy)
        ps = [0.5 * (col[k] + col[k + 1]) for k in (0, 3, 17, 30)]
        reports.append(dict(i=i, j=j, dx=dx, dy=dy, h=[M] * len(ps), p=ps,
                            fields={v: [(0.0, 0, 1.0)] * len(ps) for v in ("u", "v", "t", "q")}))
    hcol = iw._hcol(xb["h"], 1, 1, 0.5, 0.5, 0.5, 0.5)
    return syn.obs({"sound": syn.obs_type("sound", reports),
                    "synop": syn.obs_type("synop", [dict(i=2 + shift, j=2, dx=0.5, dy=0.5, h=[hcol[0] + 30.0],
                                                         fields=_SFC)])})


def _case(same=False):
    xb1 = syn.background()
    xb2 = xb1 if same else dict(xb1, t=xb1["t"] + 1.5, p=xb1["p"] * 1.01, q=xb1["q"] * 1.1,
                                rho=xb1["rho"] * 1.01, psfc=xb1["psfc"] * 1.01)
    s1, s2 = _state(xb1), _state(xb2)
    obs = fm.concat([_slot_obs(xb1, 0), _slot_obs(xb2, 1)])
    opts = iw.Options(check_max_iv=False)
    iv = fm.innovations_fgat(obs, [s1[0], s2[0]], options=opts)
    return iv, [s1, s2], s1


def _x(rng, shape):
    nx, ny, nz = shape
    return {"u": rng.normal(0, 2, shape), "v": rng.normal(0, 2, shape), "t": rng.normal(0, 1, shape),
            "q": rng.normal(0, 1e-4, shape), "psfc": rng.normal(0, 100, (nx, ny)),
            "p": rng.normal(0, 50, shape), "rho": rng.normal(0, 1e-3, shape)}


@pytest.mark.parametrize("press,ph", [("wrfda", "zero"), ("dnw", "hydrostatic")])
def test_adjoint_is_the_transpose(press, ph):
    iv, slots, start = _case()
    term = f4.FourDObsTerm(iv, slots, start, press=press, ph=ph)
    assert term.nslots == 2 and term.size == term.d.size == term.err.size == term.mask.size
    assert term.mask.sum() > 20
    rng = np.random.default_rng(0)
    x = _x(rng, start[0]["t"].shape)
    y = rng.normal(size=term.size)
    lhs = float(np.dot(term.tl(x), y))
    ax = term.ad(y)
    assert not np.any(ax["p"]) and not np.any(ax["rho"])
    rhs = sum(float(np.sum(x[k] * ax[k])) for k in ("u", "v", "t", "q", "psfc"))
    assert abs(lhs) > 1.0
    assert lhs == pytest.approx(rhs, rel=1e-12)
    # each slot's block responds to the increment, and the slots differ
    y1, y2 = (term.tl(x)[term.offsets[n]:term.offsets[n + 1]] for n in range(2))
    assert np.any(y1) and np.any(y2) and not np.array_equal(y1[:y2.size], y2[:y1.size])


def test_tl_ignores_p_and_rho():
    iv, slots, start = _case()
    term = f4.FourDObsTerm(iv, slots, start)
    rng = np.random.default_rng(1)
    x = _x(rng, start[0]["t"].shape)
    np.testing.assert_array_equal(term.tl(x), term.tl(dict(x, p=0.0 * x["p"], rho=0.0 * x["rho"])))


def test_one_state_is_fgat_h_after_the_tl_round_trip():
    """All slots at the start's state, identity model: slot n's ``y`` is FGAT's
    ``H'`` (about that state) of ``wrftltoxa(xatowrftl(x))`` plus ``xtoxa``'s ``rho``."""
    from woof.da.var.obsop_wrfda import FlatObsTerm

    iv, slots, start = _case(same=True)
    term = f4.FourDObsTerm(iv, slots, start)
    rng = np.random.default_rng(2)
    x = _x(rng, start[0]["t"].shape)
    xb, grid = start
    xa = t4.tl_to_xa(t4.xa_to_tl({k: x[k] for k in ("u", "v", "t", "q", "psfc")}, xb, grid), xb, grid)
    xa["rho"] = xb["rho"] * (xa["p"] / xb["p"] - xa["t"] / xb["t"])
    y = term.tl(x)
    for n in range(2):
        want = FlatObsTerm(fm.concat([fm.slot(iv, n)], kte=xb["t"].shape[2]), xb).tl(xa)
        np.testing.assert_array_equal(y[term.offsets[n]:term.offsets[n + 1]], want)


def test_jo_groups_run_over_all_slots_in_iv_order():
    iv, slots, start = _case()
    term = f4.FourDObsTerm(iv, slots, start)
    kinds = [k for k, _ in term.jo_split()]
    assert kinds == [k for k in iw.JO_ORDER if k in ("sound", "synop")]
    for kind, variables in term.jo_split():
        assert [v for v, _ in variables] == list(iw.JO_FIELDS[kind])
        for var, idx in variables:
            np.testing.assert_array_equal(term.d[idx], iv[kind].fields[var].inv)
    every = np.sort(np.concatenate([idx for _, vs in term.jo_split() for _, idx in vs]))
    np.testing.assert_array_equal(every, np.arange(term.size))


def test_identity_model():
    m = f4.IdentityModel()
    g0 = {"t_2": np.arange(3.0), "mu_2": np.ones(2)}
    gs = m.tl(g0, 3)
    assert len(gs) == 3 and all(np.array_equal(g["t_2"], g0["t_2"]) for g in gs)
    gs[0]["t_2"][0] = 99.0
    assert g0["t_2"][0] == 0.0  # each slot is its own copy
    a = m.ad([{"t_2": np.full(3, 1.0)}, {"t_2": np.full(3, 2.0)}, {"t_2": np.full(3, 4.0), "mu_2": np.ones(2)}])
    np.testing.assert_array_equal(a["t_2"], np.full(3, 7.0))
    np.testing.assert_array_equal(a["mu_2"], np.ones(2))


def test_term_refusals():
    iv, slots, start = _case()
    with pytest.raises(ValueError, match="2 observation slots but 1"):
        f4.FourDObsTerm(iv, slots[:1], start)
    with pytest.raises(ValueError, match="plocal is None"):
        f4.FourDObsTerm(fm.slot(iv, 0), slots, start)
    with pytest.raises(ValueError, match="press"):
        f4.FourDObsTerm(iv, slots, start, press="dn")


def _wrf_file(path: Path, stamp: str) -> Path:
    import netCDF4

    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("Time", None)
        nc.createDimension("DateStrLen", 19)
        nc.START_DATE = stamp
        t = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[0] = np.array(list(stamp), dtype="S1")
    return path


def test_analyse_refuses_var4d_setups_before_reading_anything(tmp_path):
    from woof.da.var.analysis import analyse

    fgs = [_wrf_file(tmp_path / f"fg0{n}", f"2008-02-05_{11 + n}:00:00") for n in range(1, 8)]
    obs = []
    for n in range(1, 8):
        (tmp_path / f"ob0{n}.ascii").write_text(f"slot {n}\n")
        obs.append(tmp_path / f"ob0{n}.ascii")
    be = tmp_path / "missing-be.dat"
    window = ("2008-02-05_12:00:00", "2008-02-05_18:00:00")
    with pytest.raises(ValueError, match="needs the trajectory"):
        analyse(be, obs[0], fgs[0], var4d=f4.Var4D())
    with pytest.raises(ValueError, match="not at the window's start"):
        analyse(be, obs, fgs[3], fgat=fgs, time_window=window, var4d=f4.Var4D())
    with pytest.raises(NotImplementedError, match="max_ext_its"):
        analyse(be, obs, fgs[0], fgat=fgs, time_window=window, max_ext_its=2, var4d=f4.Var4D())
    with pytest.raises(NotImplementedError, match="hybrid"):
        analyse(be, obs, fgs[0], fgat=fgs, time_window=window, ensemble=object(), var4d=f4.Var4D())


def test_tutorial_var4d_analysis_has_no_pressure_increment():
    d = os.environ.get("WOOF_WRFDA471_DIR")
    if not d or not (Path(d) / "fg").exists():
        pytest.skip("WOOF_WRFDA471_DIR with fg is not set")
    from woof.da.var.analysis import read_first_guess, wrftoxb, xatowrf

    xb, grid = wrftoxb(read_first_guess(Path(d) / "fg"))
    rng = np.random.default_rng(3)
    shape = xb["t"].shape
    xa = {"u": rng.normal(0, 1, shape), "v": rng.normal(0, 1, shape), "t": rng.normal(0, 0.5, shape),
          "q": np.zeros(shape), "psfc": rng.normal(0, 50, shape[:2])}
    w4 = xatowrf(grid, xb, xa, var4d=True)
    w3 = xatowrf(grid, xb, xa)
    np.testing.assert_array_equal(w4["P"], grid["p"])
    assert not np.any(w4["xa_p"]) and np.any(w3["xa_p"])
    for k in ("U", "V", "MU", "PSFC"):  # what does not read xa%p is the 3D-Var transfer's
        np.testing.assert_array_equal(w4[k], w3[k])


def test_var4d_door_runs_on_the_tutorial_trajectory():
    """``analyse(var4d=Var4D())`` end to end on the FGAT lane's files, ``fg01`` as the
    analysis (the identity model, so no WRFDA comparison here: that is L-id's)."""
    d = os.environ.get("WOOF_WRFDA471_FGAT_DIR")
    if not d:
        pytest.skip("WOOF_WRFDA471_FGAT_DIR is not set")
    from woof.da.var.analysis import analyse

    d = Path(d)
    fgs = [d / f"fg0{n}" for n in range(1, 8)]
    obs = [d / f"ob0{n}.ascii" for n in range(1, 8)]
    r = analyse(d / "be.dat", obs, fgs[0], fgat=fgs, var4d=f4.Var4D(), ntmax=5,
                time_window=("2008-02-05_12:00:00", "2008-02-05_18:00:00"))
    j = r.cg[0].j
    assert len(j) >= 2 and r.cg[0].j_final < 0.95 * j[0]
    np.testing.assert_array_equal(r.wrf["P"], r.grid["p"])
    assert np.isfinite(r.wrf["T"]).all() and np.any(r.wrf["T"] != r.grid["t_2"])
