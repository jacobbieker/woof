"""Strict mode's no-mixing relaxation fold against compiled WRF v4.7.1.

``tests/data/wrf471_bigstep/rk-relax-fold.npz`` holds WRF v4.7.1
``rk_addtend_dry`` (unchanged extract, tools/bigstep_wrf471_oracle,
``rk_pack.py --relax-fold``) run as solve_em does on a specified domain
with km_opt = 0 and no physics:

  * the held ``*_tendf`` are zero, so the stage-1 relaxation in
    ``u_save``/``v_save``/``w_save``/``t_save`` is the whole held tendency;
  * rk_step 1 folds it (``ru_tendf + u_save*msfuy``) and every stage adds
    ``ru_tendf/msfuy`` to a fresh stage tendency, stages 1-3 in sequence
    on the same held arrays.

woof replays the same words through its strict production path:
``prepare``-zeroed held slots, :func:`woof.core.dycore.fold_lateral_relaxation`
(the relaxation rows themselves are the WRF save arrays; their own words
are bound to WRFPLUS's relax_bdytend in tests/test_grad_wrfplus_lbc.py),
then :func:`woof.core.dycore.add_fixed_dry_tendencies` under a no-mixing
config on every stage.  Before the held-slot fix the add returned early
and every relaxation-ring word of the stage tendency missed the fold.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify.bigstep_rk_oracle import RK_ORACLE_DIR, _crop3, _put3

FIXTURE = RK_ORACLE_DIR / "rk-relax-fold.npz"
SHAPES = lambda nz, ny, nx: ((nz, ny, nx + 1), (nz, ny + 1, nx), (nz + 1, ny, nx),
                             (nz + 1, ny, nx), (nz, ny, nx))
TENDS = ("ru_t", "rv_t", "rw_t", "rph_t", "rth_t")
SLOTS = ("smag_ru", "smag_rv", "smag_rw", None, "smag_rth")
SAVES = {"u": 0, "v": 1, "w": 2, "theta": 4}


def _load():
    with np.load(FIXTURE) as src:
        arrays = {k: src[k] for k in src.files}
    return arrays, json.loads(FIXTURE.with_suffix(".json").read_text())


def test_relax_fold_fixture_provenance():
    arrays, meta = _load()
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == meta["fixture_sha256"]
    state = RK_ORACLE_DIR / "state-real.npz"
    assert meta["state_sha256"] == hashlib.sha256(state.read_bytes()).hexdigest()
    assert [c["name"] for c in meta["cases"]] == ["nomix-relax",
                                                 "nomix-relax-map-extremes"]
    for case in meta["cases"]:
        pre = f"relax{case['id']}_"
        for step in (1, 2, 3):
            want, given = arrays[f"{pre}a{step}_wrf"], arrays[f"{pre}a{step}"]
            assert want.dtype == np.float32 and want.shape == given.shape
            # WRF moved the ring: the case is not empty.
            assert (want.view(np.uint32) != given.view(np.uint32)).sum() > 1000


def _replay(arrays, case, rows):
    import cupy as cp

    from woof.core import dycore
    from woof.core.state import DomainState

    nx, ny, nz = (int(arrays[k]) for k in ("nx", "ny", "nz"))
    pre = f"relax{case['id']}_"
    maps, saves = arrays[pre + "maps"], arrays[pre + "saves"]
    cfg = SimpleNamespace(km_opt=0, diff_opt=0, diff_6th_opt=0, obs_nudge_opt=0,
                          specified=True, nested=False)
    assert not dycore.wrf_mixing_package_active(cfg)
    state = DomainState.__new__(DomainState)
    state.c1h = cp.asarray(arrays["c1"][:nz])
    state.c2h = cp.asarray(arrays["c2"][:nz])
    state.c1f = cp.zeros(nz + 1, cp.float32)
    state.c2f = cp.zeros(nz + 1, cp.float32)
    state.qv = None
    state.p = cp.empty((nz, ny, nx), cp.float32)
    state.has_msf = True
    state.msft = cp.asarray(maps[1, 4:4 + ny, 4:4 + nx])
    state.msfu = cp.asarray(maps[3, 4:4 + ny, 4:5 + nx])
    state.msfv = cp.asarray(maps[4, 4:5 + ny, 4:4 + nx])
    state.msfv_inv = cp.asarray(maps[5, 4:5 + ny, 4:4 + nx])
    state.mub2d = cp.asarray(arrays[pre + "mut"][4:4 + ny, 4:4 + nx])
    state.mup = cp.zeros((ny, nx), cp.float32)
    state.h_diabatic = cp.asarray(_crop3(saves[5], nz, ny, nx))
    state._scratch = {}
    shapes = SHAPES(nz, ny, nx)
    for (nk, nj, ni), name, slot in zip(shapes, TENDS, SLOTS):
        setattr(state, name, cp.zeros((nk, nj, ni), cp.float32))
        if slot is not None:
            # prepare_fixed_tendencies' strict zeroing; nothing fills them.
            state._scratch[slot] = cp.zeros((nk, nj, ni), cp.float32)
    state.u0, state.v0 = state.ru_t, state.rv_t
    state.w0, state.thp0 = state.rw_t, state.rth_t
    rows.update({name: cp.asarray(_crop3(saves[i], *shapes[i]))
                 for name, i in SAVES.items()})
    out = {}
    for step in (1, 2, 3):
        a = arrays[f"{pre}a{step}"].copy()
        for i, ((nk, nj, ni), name) in enumerate(zip(shapes, TENDS)):
            getattr(state, name)[...] = cp.asarray(_crop3(a[i], nk, nj, ni))
        if step == 1:
            dycore.fold_lateral_relaxation(state, cfg)
        dycore.add_fixed_dry_tendencies(state, cfg)
        dycore.add_h_diabatic_tendency(state)
        for i, ((nk, nj, ni), name) in enumerate(zip(shapes, TENDS)):
            _put3(a[i], cp.asnumpy(getattr(state, name)), nk, nj, ni)
        out[step] = a
    return out


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("case_id", (0, 1))
def test_strict_no_mixing_fold_matches_compiled_wrf471(case_id, monkeypatch):
    from woof.core import dycore
    from woof.core.fp32_ulp import fp32_ulp_distance

    arrays, meta = _load()
    case = meta["cases"][case_id]
    monkeypatch.setattr(dycore, "WRF_EXACT", True)
    rows = {}
    # The relaxation rows are WRF's own save arrays.
    monkeypatch.setattr(dycore, "capture_folded_relaxation",
                        lambda state, cfg: rows)
    got = _replay(arrays, case, rows)
    worst = {}
    for step in (1, 2, 3):
        want = arrays[f"relax{case_id}_a{step}_wrf"]
        for i, field in enumerate(("u", "v", "w", "ph", "t")):
            d = fp32_ulp_distance(got[step][i], want[i])
            worst[(step, field)] = (int(d.max(initial=0)), int((d > 0).sum()))
    # Word-exact except v: WRF multiplies the held v row by MF_VX_INV, the
    # file's own inverse, and woof's division by msfv rounds 1 ulp apart
    # where that inverse is not 1/msfv's float (the same v words
    # tests/test_bigstep_rk_wrf471_parity.py pins; measured 108/174/305
    # words on stages 1-3 of the real map, none on map-extremes, where
    # the inverse is exact).
    bad = {k: v for k, v in worst.items() if v[0] > (1 if k[1] == "v" else 0)}
    assert not bad, bad
