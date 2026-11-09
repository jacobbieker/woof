"""Partial-depth nest with map factors: a real projection's msft/msfu/msfv.

woof/core/partial_nest_run.py ("Map factors"), kernels/nest_top.cu's
``*_msf`` entry points (their words against acoustic.cu's msf kernels are
tests/test_partial_nest_dyn.py's) and kernels/partial_nest.cu's
``nest_top_budget_residual_msf``.  The gates, on the D3 geometry with a
Lambert-like factor (1 at the centre, about 1.03 at the corners) on every
grid:

* the interface budget: each cell's flux weighted by 1/msft (a cell's column
  mass changes at ww*dx*dy/msft), the child's footprint sum is the parent's;
* a resting atmosphere stays at rest, its area-weighted dry mass
  (sum mu/msft^2) unchanged;
* a uniform qv stays uniform through the interface (the scalar face flux
  carries the msft rk_update_scalar multiplies a tendency by);
* a real column's truncation (hypsometric_opt = 2) diagnoses the parent's
  own column below the interface;
* the child carries km_opt = 4 with diff_opt = 1 (WRF's real-data
  horizontal Smagorinsky on coordinate surfaces) and refuses diff_opt = 2.
"""

import dataclasses

import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = pytest.mark.gpu


def _skip_strict():
    from woof import wrf_exact
    if wrf_exact.ENABLED:
        pytest.skip("partial-depth nests run in the default build only")


def _lambert(x, y):
    """A smooth Lambert-like map factor over the D3 parent (180 x 48 km)."""
    return 1.0 + 0.03 * ((x - 90e3) ** 2 + (y - 24e3) ** 2) / 90e3 ** 2


def _offset(x, y):
    """A factor well away from 1 over the band (1.25 to 1.30), so a missing
    msft in a flux shows above rounding."""
    return 1.25 + 0.05 * ((x - 90e3) ** 2 + (y - 24e3) ** 2) / 90e3 ** 2


def _area_mass(state) -> float:
    import cupy as cp
    m = cp.asnumpy(state.msft).astype(np.float64)
    mu = cp.asnumpy(state.mub2d).astype(np.float64) + cp.asnumpy(state.mup).astype(np.float64)
    return float((mu / (m * m)).sum())


@requires_gpu
def test_map_factor_interface_budget_matches_the_parent():
    _skip_strict()
    import cupy as cp
    from woof.core import dycore
    from woof.core.clock import build_schedule
    from woof.core.dycore import stage_fluxes
    from woof.core.partial_nest_run import two_domain_experiment
    from woof.verify.partial_nest_envelope import GEOMETRY, build
    nest, _, _ = build("bubble", map_factor=_lambert)
    assert nest.child.has_msf and nest.parent.has_msf and nest.child.rotational
    exp = two_domain_experiment(nest.pcfg, nest.ccfg, nest.band,
                                run_seconds=GEOMETRY["dt"], time_step=int(GEOMETRY["dt"]))
    clocks = build_schedule(exp).clock.clocks()
    pc, cc = clocks[1], clocks[2]
    nest.snapshot(nest.a)
    pc.prepare_step()
    dycore.step(nest.parent, nest.pcfg)
    pc.advance()
    nest.force(cc, pc)
    k, band = nest.k, nest.band
    r = band.parent_grid_ratio
    _, _, ww = stage_fluxes(nest.parent, nest.pcfg)
    j0, i0 = band.j_parent_start - 1, band.i_parent_start - 1
    box = (slice(j0, j0 + band.ny // r), slice(i0, i0 + band.nx // r))
    pw = cp.asnumpy(ww[k]).astype(np.float64)[box]
    pm = cp.asnumpy(nest.parent.msft).astype(np.float64)[box]
    cm = cp.asnumpy(nest.child.msft).astype(np.float64)
    parent_sum = r * r * (pw / pm).sum()
    child = cp.asnumpy(nest.b.om_k).astype(np.float64)
    scale = np.abs(pw).max() * band.nx * band.ny
    assert nest.max_abs_mass_residual > 0.0
    assert abs((child / cm).sum() - parent_sum) < 1e-6 * scale
    # The unweighted sum is not the budget: the weights matter at this size.
    assert abs(child.sum() - r * r * pw.sum()) > 1e-6 * scale


@requires_gpu
def test_map_factor_rest_stays_at_rest():
    _skip_strict()
    from woof.core.partial_nest_run import run
    from woof.verify.partial_nest_envelope import GEOMETRY, build
    import cupy as cp
    nest, _, _ = build("rest", map_factor=_lambert)
    m_child, m_parent = _area_mass(nest.child), _area_mass(nest.parent)
    run(nest, run_seconds=240.0, time_step=int(GEOMETRY["dt"]))
    for name in ("u", "v", "w", "thp", "php", "mup"):
        assert np.isfinite(cp.asnumpy(getattr(nest.child, name))).all(), name
    assert float(cp.max(cp.abs(nest.child.w))) < 2e-3
    k, fr = nest.k, nest.frame_width
    assert float(cp.max(cp.abs(nest.child.w[k, fr:-fr, fr:-fr]))) < 1e-3
    assert abs(_area_mass(nest.child) - m_child) / m_child < 1e-7
    assert abs(_area_mass(nest.parent) - m_parent) / m_parent < 1e-7


@requires_gpu
def test_map_factor_uniform_vapour_stays_uniform(monkeypatch):
    _skip_strict()
    import cupy as cp
    from woof.core.partial_nest_run import run
    from woof.verify import partial_nest_envelope as E
    case = dataclasses.replace(E.CASES["rest"], name="moist_rest_msf",
                               qv=lambda z: 0.004 + 0.0 * np.asarray(z),
                               physics=dict(moist=True, mp_physics=0, moist_adv_opt=1))
    monkeypatch.setitem(E.CASES, "moist_rest_msf", case)
    nests, _, _ = E.build_chain("moist_rest_msf", map_factor=_offset)
    nest = nests[0]
    assert nest.child.has_msf and "qv" in nest.species
    nest.parent.w[nest.k] += np.float32(0.05)
    run(nest, run_seconds=360.0, time_step=12)
    q = cp.asnumpy(nest.child.qv[:, 6:-6, 6:-6]).astype(np.float64)
    # MEASURED (RTX A6000): 4.0e-9 as carried; 4.3e-8 with the face flux
    # applied without its msft; 2.2e-7 with no interface flux at all.
    assert float(np.abs(q - 0.004).max()) < 1.5e-8



@requires_gpu
def test_a_real_columns_truncation_diagnoses_the_parents_column():
    """A real-data column (hypsometric_opt = 2, map factors, Coriolis) cut at
    its interface diagnoses, below it, the parent's own p, alt and al: the
    log-pressure EOS reads the full-level drops dc3f/dc4f, which the
    truncated coordinate moves (c4' = c4 + c3*(p_i - p_top) + p_top - p_i;
    MEASURED: left at the parent's, a GFS 9 km -> 3 km child's p was off by
    up to 37 kPa).  The flags the dycore keys its msf and Coriolis paths on
    come along."""
    _skip_strict()
    import cupy as cp
    from woof.config import RunConfig
    from woof.core.diagnostics import update_diagnostics
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.partial_nest_dyn import truncate_state
    from woof.core.state import init_theta_perturbation
    cfg = RunConfig(nx=20, ny=12, nz=30, dx=3000.0, dy=3000.0, ztop=16000.0,
                    dt=12.0, run_seconds=0.0, hybrid_opt=2, etac=0.45,
                    hypsometric_opt=2)
    coord = make_vertical_coord(cfg.nz, hybrid_opt=2, etac=cfg.etac)
    base = make_base_state(coord, lambda z: 290.0 + 0.004 * np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    rng = np.random.default_rng(3)
    state = init_theta_perturbation(
        cfg, coord, base,
        lambda x, z: (0.5 * np.sin(np.asarray(z)[:, None, None] / 2000.0)
                      + 0.0 * np.asarray(x)[None, None, :]))
    state.mup[...] = cp.asarray(200.0 * rng.standard_normal((cfg.ny, cfg.nx)),
                                dtype=cp.float32)
    m = 1.0 + 0.05 * rng.random((cfg.ny, cfg.nx))
    state.set_map_coriolis(msft=m, msfu=np.pad(m, ((0, 0), (0, 1)), mode="edge"),
                           msfv=np.pad(m, ((0, 1), (0, 0)), mode="edge"),
                           f=np.full((cfg.ny, cfg.nx), 1e-4))
    update_diagnostics(state, 2)
    k = int(np.flatnonzero(np.asarray(coord.c3f) == 0.0)[0]) + 2
    child, ccfg = truncate_state(state, cfg, k, coord=coord)
    assert child.has_msf and child.rotational and float(child.p_top) > float(state.p_top)
    update_diagnostics(child, 2)
    scale = {"p": "p", "alt": "alt", "al": "alt"}       # al' against alpha
    for name in ("p", "alt", "al"):
        got = cp.asnumpy(getattr(child, name)).astype(np.float64)
        want = cp.asnumpy(getattr(state, name)).astype(np.float64)[:k]
        ref = np.abs(cp.asnumpy(getattr(state, scale[name]))).max()
        rel = np.abs(got - want).max() / ref
        assert rel < 1e-5, (name, rel)


@requires_gpu
def test_the_child_carries_horizontal_smagorinsky_on_coordinate_surfaces():
    _skip_strict()
    from woof.config import RunConfig
    from woof.core.partial_nest import PartialNestRefused
    from woof.core.partial_nest_run import refuse_child
    from woof.core.state import DomainState
    cfg = RunConfig(nx=12, ny=12, nz=10, dx=1000.0, dy=1000.0, ztop=8000.0,
                    dt=6.0, run_seconds=0.0, hybrid_opt=2, nested=True,
                    specified=False, km_opt=4, diff_opt=1, emdiv=0.0, damp_opt=0)
    state = DomainState(cfg)
    refuse_child(cfg, state)
    with pytest.raises(PartialNestRefused, match="km_opt mixing"):
        refuse_child(dataclasses.replace(cfg, diff_opt=2), state)
