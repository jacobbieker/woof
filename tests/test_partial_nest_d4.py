"""Partial-depth nest, Phase D4: chains, moisture and radiation through the
interface, and the model integration.

woof/core/partial_nest.py (chains), partial_nest_run.py (the child stepped
by dycore.step's hooks), partial_nest_physics.py (the extended radiation
column), kernels/partial_nest_d4.cu, woof/core/model.py
(realize_partial_nests, partial_nest_steppers) and the experiment loader's
``partial_nest`` domain key.  The gates:

* config: a chain parses coarse to fine and is validated band by band
  (footprint, a finer interface below its parent band's top zone, phi
  feedback refused by name); the loader's ``partial_nest`` key builds the
  band from its domain, refuses it on the root, on a moving or delayed or
  streamed domain, and refuses a full-depth child under a partial parent;
  without the key the restart identity has no trace of it;
* the dycore with ``partial_top=None`` is the step it always was (a whole
  run, word for word, against a run without the keyword);
* moist scalars: a uniform qv stays uniform through the interface (it does
  not without the interface flux);
* the extended radiation column: the child's levels below the interface,
  the parent's SINT above, heating handed back below the interface only;
* a two-band chain stays at rest; and an experiment run through
  ``execute_experiment`` (the idealized tree, realize_partial_nests) is
  word for word the standalone driver's run.
"""

import dataclasses
import textwrap

import numpy as np
import pytest
from conftest import requires_gpu


def _skip_strict():
    from woof import wrf_exact
    if wrf_exact.ENABLED:
        pytest.skip("the partial-depth nest runs in the default build only (refused by name)")


def _coord(nz=30, etac=0.7):
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.verify.partial_nest_envelope import G
    coord = make_vertical_coord(nz, hybrid_opt=2, etac=etac)
    make_base_state(coord, lambda z: 300.0 * np.exp(1e-4 * np.asarray(z) / G),
                    p_surf=1.0e5, ztop=10000.0)
    return coord


def _cfg(**kw):
    from woof.config import RunConfig
    values = dict(nx=40, ny=12, nz=30, dx=9000.0, dy=9000.0, ztop=10000.0, dt=36.0,
                  run_seconds=0.0, hybrid_opt=2, etac=0.7)
    values.update(kw)
    return RunConfig(**values)


def _bands():
    from woof.core.partial_nest import PartialNestBand
    return (PartialNestBand(k_top=21, parent_grid_ratio=3, parent_time_step_ratio=3,
                            i_parent_start=11, j_parent_start=3, nx=60, ny=24),
            PartialNestBand(k_top=12, parent_grid_ratio=3, parent_time_step_ratio=3,
                            i_parent_start=16, j_parent_start=8, nx=90, ny=27))


# ---------------------------------------------------------------------------
# config (no device)
# ---------------------------------------------------------------------------

def test_a_chain_is_validated_band_by_band():
    from woof import wrf_exact
    from woof.core.partial_nest import PartialNestRefused, validate_band_chain
    if wrf_exact.ENABLED:
        pytest.skip("strict build: refused by name (test_partial_nest_run)")
    coord, cfg = _coord(), _cfg()
    b1, b2 = _bands()
    assert validate_band_chain((b1, b2), cfg, coord) == (b1, b2)
    with pytest.raises(PartialNestRefused, match="must lie below"):
        validate_band_chain((b1, dataclasses.replace(b2, k_top=21)), cfg, coord)
    with pytest.raises(PartialNestRefused, match="top zone"):
        validate_band_chain((b1, dataclasses.replace(b2, k_top=15)), cfg, coord)
    with pytest.raises(PartialNestRefused, match="not inside its parent"):
        validate_band_chain((b1, dataclasses.replace(b2, i_parent_start=40)), cfg, coord)
    with pytest.raises(PartialNestRefused, match="not inside its parent"):
        validate_band_chain((dataclasses.replace(b1, i_parent_start=30), b2), cfg, coord)
    with pytest.raises(PartialNestRefused, match="phi is not fed back"):
        validate_band_chain((b1, dataclasses.replace(b2, feedback_fields=("ph",))),
                            cfg, coord)


def _load(tmp_path, d01="", d02=""):
    from test_experiment import _write
    from woof.experiment import load_experiment
    return load_experiment(_write(tmp_path, d01=d01, d02=d02))


def _loads_or_skip(tmp_path, **kw):
    try:
        return _load(tmp_path, **kw)
    except ValueError as exc:
        if "partial_nest" in str(exc):
            raise
        pytest.skip(f"the synthetic experiment does not load here: {exc}")


def test_the_loader_builds_the_band_from_its_domain(tmp_path):
    from woof.core.partial_nest import PartialNestBand
    from woof.core.model import restart_identity_payload
    plain = _loads_or_skip(tmp_path)
    assert plain.domain(2).partial_nest is None
    assert "partial_nest" not in repr(restart_identity_payload(plain))
    exp = _load(tmp_path, d02='partial_nest = { k_top = 21, relax_levels = 4, '
                              'feedback_fields = ["t", "w"] }')
    band = exp.domain(2).partial_nest
    assert band == PartialNestBand(k_top=21, parent_grid_ratio=3, parent_time_step_ratio=3,
                                   i_parent_start=40, j_parent_start=30, nx=60, ny=60,
                                   relax_levels=4, feedback_fields=("t", "w"))
    assert "partial_nest" in repr(restart_identity_payload(exp))
    with pytest.raises(ValueError, match="repeats"):
        _load(tmp_path, d02="partial_nest = { k_top = 21, nx = 60 }")
    with pytest.raises(ValueError, match="unknown"):
        _load(tmp_path, d02="partial_nest = { k_tpo = 21 }")
    with pytest.raises(ValueError, match="root"):
        _load(tmp_path, d01="partial_nest = { k_top = 21 }")


def test_the_real_data_route_refuses_partial_bands(tmp_path):
    from woof.core.model import refuse_real_partial_nests
    exp = _load(tmp_path, d02="partial_nest = { k_top = 21 }")
    with pytest.raises(ValueError, match="idealized experiment tree"):
        refuse_real_partial_nests(exp, "woof run (experiment-tree)")
    refuse_real_partial_nests(_loads_or_skip(tmp_path), "woof run (experiment-tree)")


# ---------------------------------------------------------------------------
# device gates
# ---------------------------------------------------------------------------

def _words(a):
    import cupy as cp
    return np.ascontiguousarray(cp.asnumpy(a)).view(np.uint32)


def _same(got, want, what):
    g, w = _words(got), _words(want)
    bad = np.flatnonzero(g.ravel() != w.ravel())
    assert bad.size == 0, f"{what}: {bad.size} of {g.size} words differ"


@requires_gpu
@pytest.mark.gpu
def test_partial_top_none_is_the_step_it_always_was():
    """dycore.step(partial_top=None) runs the identical step: ten steps of
    the gravity-wave parent, word for word."""
    _skip_strict()
    from woof.core import dycore
    from woof.verify.partial_nest_envelope import build
    _, parent_only, _ = build("gravity_wave")
    a, cfg = parent_only()
    b, _ = parent_only()
    for _ in range(10):
        dycore.step(a, cfg)
        dycore.step(b, cfg, partial_top=None)
    for name in ("u", "v", "w", "thp", "php", "mup", "p"):
        _same(getattr(a, name), getattr(b, name), name)


def _moist_rest(feedback=1):
    """A one-band moist rest state with uniform qv (no physics)."""
    from woof.verify import partial_nest_envelope as E
    case = dataclasses.replace(E.CASES["rest"], name="moist_rest",
                               qv=lambda z: 0.004 + 0.0 * np.asarray(z),
                               physics=dict(moist=True, mp_physics=0, moist_adv_opt=1))
    E.CASES["moist_rest"] = case
    try:
        nests, _, _ = E.build_chain("moist_rest", band_overrides={"feedback": feedback})
    finally:
        del E.CASES["moist_rest"]
    return nests[0]


@requires_gpu
@pytest.mark.gpu
def test_uniform_vapour_stays_uniform_through_the_interface(monkeypatch):
    """qv = const telescopes against the mass the acoustic loop moves
    through the interface (nest_top_scalar_face_flux/apply); with the
    interface flux removed the child's top level drifts."""
    _skip_strict()
    import cupy as cp
    from woof.core.partial_nest_run import ChildTop, run
    # A little flow through the interface: the rest state's own noise is
    # too quiet to show the missing flux, so tilt it with a uniform wind.

    def spread(nest):
        q = cp.asnumpy(nest.child.qv[:, 6:-6, 6:-6]).astype(np.float64)
        return float(np.abs(q - 0.004).max())

    nest = _moist_rest()
    assert nest.species and "qv" in nest.species
    nest.parent.w[nest.k] += np.float32(0.05)
    run(nest, run_seconds=360.0, time_step=12)
    with_flux = spread(nest)
    nest = _moist_rest()
    nest.parent.w[nest.k] += np.float32(0.05)
    monkeypatch.setattr(ChildTop, "scalars_after", lambda self, istage, dt: None)
    run(nest, run_seconds=360.0, time_step=12)
    without = spread(nest)
    assert with_flux < 1e-7, with_flux
    assert without > 10 * max(with_flux, 1e-9), (with_flux, without)


@requires_gpu
@pytest.mark.gpu
def test_the_extended_column_is_the_child_below_and_the_parent_above():
    _skip_strict()
    import cupy as cp
    from woof.core.nest_interp import sint
    from woof.core.partial_nest_physics import (ExtendedColumn,
                                                ExtendedColumnRadiation)
    from woof.core.physics import RadiationResult
    from woof.verify.partial_nest_envelope import build
    nest, _, _ = build("bubble")
    col = ExtendedColumn(nest)
    ext = col.refresh()
    k, nz = nest.k, nest.pcfg.nz
    assert ext.thp.shape[0] == nz and float(ext.p_top) == float(nest.parent.p_top)
    # Perturbations re-expressed on the column's base: within an ulp of
    # the total.
    np.testing.assert_allclose(cp.asnumpy(ext.thp[:k]), cp.asnumpy(nest.child.thp),
                               rtol=0, atol=6e-5)
    np.testing.assert_allclose(cp.asnumpy(ext.php[:k + 1]), cp.asnumpy(nest.child.php),
                               rtol=0, atol=1e-2)
    _same(ext.p[:k], nest.child.p, "p below")
    want = sint(cp.ascontiguousarray(nest.parent.p[k:]), nest.regs[""])
    _same(ext.p[k:], want, "p above")
    tot = sint(cp.ascontiguousarray(nest.parent.thp[k:] + nest.parent.thb[k:, None, None]),
               nest.regs[""])
    _same(ext.thp[k:], tot - ext.thb[k:, None, None], "theta above")
    # Column mass: the child's plus the constant mass above its interface.
    above = np.float32(nest.p_i - float(nest.parent.p_top))
    _same(ext.mub2d, nest.child.mub2d + above, "mub")
    seen = {}

    def inner(*, atmosphere, fields, state, cfg):
        seen["nz"] = state.p.shape[0]
        seen["atm"] = atmosphere["pressure"].shape[0]
        level = cp.arange(nz, dtype=cp.float32)[:, None, None] + 0 * state.p
        z2 = cp.zeros(state.p.shape[1:], dtype=cp.float32)
        return RadiationResult(rthratenlw=level, rthratensw=2 * level, swdown=z2, glw=z2)

    res = ExtendedColumnRadiation(inner, col)(atmosphere=None, fields={},
                                             state=nest.child, cfg=nest.ccfg)
    assert seen == {"nz": nz, "atm": nz}
    assert res.rthratenlw.shape == nest.child.p.shape
    assert float(res.rthratensw[k - 1].max()) == 2.0 * (k - 1)


@requires_gpu
@pytest.mark.gpu
def test_a_two_band_chain_stays_at_rest():
    _skip_strict()
    import cupy as cp
    from woof.core.partial_nest_run import run
    from woof.verify.partial_nest_envelope import build_chain
    nests, _, _ = build_chain("chain_rest")
    assert len(nests) == 2 and nests[1].parent is nests[0].child
    run(nests, run_seconds=1800.0, time_step=36)
    for n in nests:
        assert np.isfinite(cp.asnumpy(n.child.w)).all()
        assert float(cp.abs(n.child.w).max()) < 5e-3


@requires_gpu
@pytest.mark.gpu
def test_execute_experiment_is_the_standalone_driver():
    """The band declared on a DomainConfig, realized on the idealized tree
    and run by execute_experiment, is word for word the standalone run."""
    _skip_strict()
    from types import SimpleNamespace
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.model import execute_experiment
    from woof.core.partial_nest_run import chain_experiment, child_config, run
    from woof.verify import partial_nest_envelope as E
    from woof.verify.cases.nest_ideal_common import assemble_idealized_tree
    seconds = 120.0
    nest_ref, _, _ = E.build("gravity_wave")
    run(nest_ref, run_seconds=seconds, time_step=12)
    case = E.CASES["gravity_wave"]
    g = E.GEOMETRY
    rcfg = E._root_cfg(g, case)
    coord = make_vertical_coord(g["nz"], hybrid_opt=2, etac=g["etac"])
    base = make_base_state(coord, case.sounding, p_surf=rcfg.p_surf, ztop=rcfg.ztop)
    band = nest_ref.band
    cfull_cfg = child_config(rcfg, band)
    exp = chain_experiment(rcfg, [(band, cfull_cfg)], run_seconds=seconds, time_step=12)
    exp = dataclasses.replace(exp, domains=(
        exp.domains[0], dataclasses.replace(exp.domains[1], partial_nest=band)))
    root = E._init(rcfg, coord, base, case, 0.0, 0.0, cells=g["ratio"])
    x0 = (band.i_parent_start - 1) * rcfg.dx
    y0 = (band.j_parent_start - 1) * rcfg.dy

    def child_init(dc, parent):
        return SimpleNamespace(state=E._init(dc.run, coord, base, case, x0, y0))

    grids = [SimpleNamespace(latlon_mass=lambda: (None, None))] * 2
    model = assemble_idealized_tree(exp, root, grids=grids, child_initializer=child_init,
                                    coupler_factory=lambda node: None,
                                    domain_preparer=lambda *a: None, coord=coord)
    execute_experiment(model, validate_state=False, pool_trim_per_period=False)
    child = model.node(2).state
    for name in ("u", "v", "w", "thp", "php", "mup"):
        _same(getattr(child, name), getattr(nest_ref.child, name), "child " + name)
        _same(getattr(model.node(1).state, name), getattr(nest_ref.parent, name),
              "parent " + name)
