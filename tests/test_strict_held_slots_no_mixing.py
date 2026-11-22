"""Strict mode adds its held dry slots even when no mixing package runs.

Strict mode (``GPUWM_WRF_EXACT=1``) folds two terms into the held
``smag_r*`` slots that WRF's ``rk_addtend_dry`` adds on every RK stage
whatever the mixing options (module_em.F):

  * the stage-1 lateral relaxation of a specified or nested domain
    (``relax_bdy_dry`` -> ``u_save``/``v_save``/``t_save``/``w_save``,
    :func:`woof.core.dycore.fold_lateral_relaxation`);
  * the physics tendencies (``update_phy_ten`` -> ``*_tendf``,
    :func:`woof.core.dycore._fold_physics_wrf`).

``add_fixed_dry_tendencies`` used to return early when neither a mixing
package nor diff6 nor observation nudging ran, so a strict dry run with
no mixing dropped both: its relaxation zone had no u/v/theta relaxation
and its physics never reached the state.  These tests fail on that code.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from conftest import requires_gpu

NX, NY, NZ = 24, 20, 10
RELAX = 4


STRICT = os.environ.get("GPUWM_WRF_EXACT") == "1"

pytestmark = [requires_gpu, pytest.mark.skipif(
    not STRICT, reason="the held-slot fold is strict mode's (GPUWM_WRF_EXACT=1)")]


def _specified_case():
    import cupy as cp

    import woof.ingest.lateral_bc as lbc
    from tilestream import harness as H

    cfg = H.make_config(NX, NY, NZ, h_sca_adv_order=5, terrain_opt=1,
                        specified=True, spec_zone=1, relax_zone=RELAX,
                        spec_bdy_width=RELAX + 1)
    jj, ii = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    ridge = 400 * np.exp(-(((ii - NX / 2) / (NX / 5)) ** 2
                           + ((jj - NY / 2) / (NY / 4)) ** 2))
    geo = dataclasses.replace(H.neutral_geography(cfg), terrain=ridge)
    state = H.make_state(cfg, 1, geography=geo)
    snaps = []
    for n in range(2):
        forcing = H.make_state(cfg, 1, geography=geo)
        forcing.u += cp.float32(2.0 + n)
        forcing.v += cp.float32(1.0 + n)
        forcing.thp += cp.float32(1.0 + n)
        snaps.append(dict(lbc.domain_boundary_snapshot(forcing)))
    tables = lbc.build_lateral_boundaries(
        snaps, [0.0, 3600.0], spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    lbc.attach_lateral_boundaries(state, tables)
    return cfg, state


def _relax_ring(a):
    """The relaxation zone's points (spec zone excluded) of a C-grid field."""
    ny, nx = a.shape[-2:]
    inner = np.zeros((ny, nx), bool)
    inner[1:ny - 1, 1:nx - 1] = True
    core = np.zeros((ny, nx), bool)
    core[RELAX:ny - RELAX, RELAX:nx - RELAX] = True
    return a[..., inner & ~core]


def _run(cfg, state, nsteps=2):
    import cupy as cp

    from woof.core import dycore
    for _ in range(nsteps):
        dycore.step(state, cfg)
    return {n: cp.asnumpy(getattr(state, n)).copy()
            for n in ("u", "v", "thp")}


def test_specified_no_mixing_relaxes_the_relaxation_zone(monkeypatch):
    from woof.config import wrf_mixing_package_active
    from woof.core import dycore

    cfg, state = _specified_case()
    assert not wrf_mixing_package_active(cfg) and cfg.diff_6th_opt <= 0
    assert dycore._wrf_relaxation_fold(cfg)
    folded = _run(cfg, state)

    monkeypatch.setattr(dycore, "fold_lateral_relaxation",
                        lambda state, cfg: None)
    cfg, state = _specified_case()
    unrelaxed = _run(cfg, state)

    for name in ("u", "v", "thp"):
        a, b = _relax_ring(folded[name]), _relax_ring(unrelaxed[name])
        assert (a.view(np.uint32) != b.view(np.uint32)).sum() > a.size // 4, (
            f"{name}: the folded relaxation never reached the state")
        # The boundary tables raise u, v and theta, so the relaxation pulls
        # the zone up toward them.
        assert float(a.mean()) > float(b.mean()), name


def test_no_mixing_physics_reaches_the_state(monkeypatch):
    import cupy as cp

    from tilestream import harness as H
    from woof.config import wrf_mixing_package_active
    from woof.core import dycore
    from woof.core.physics import PhysicsTendencies

    heating = 1.0e-3  # coupled theta tendency, Pa K s-1

    class _Physics:
        def __init__(self, rate):
            self.rate = rate

        def compute(self, state, cfg):
            tend = PhysicsTendencies.zeros(state)
            mu = state.mub2d + state.mup
            tend.rtheta += cp.float32(self.rate) * mu[None]
            return tend

    monkeypatch.setattr(dycore, "physics_enabled", lambda cfg: True)
    out = {}
    for rate in (0.0, heating):
        cfg = H.make_config(NX, NY, NZ, h_sca_adv_order=5)
        assert not wrf_mixing_package_active(cfg) and cfg.diff_6th_opt <= 0
        state = H.make_state(cfg, 1)
        state.physics = _Physics(rate)
        out[rate] = _run(cfg, state, nsteps=1)["thp"]
    gain = out[heating] - out[0.0]
    expected = heating * H.make_config(NX, NY, NZ).dt
    # One step of a uniform heating: every column warms by about rate*dt;
    # the acoustic response and float32 theta (ulp ~3e-5 K) blur points.
    assert (gain > 0.5 * expected).all(), (float(gain.min()), expected)
    assert abs(float(gain.mean()) - expected) < 0.1 * expected, (
        float(gain.mean()), expected)
