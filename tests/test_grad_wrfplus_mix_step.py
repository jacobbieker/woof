"""The step's TL/AD with WRF's diff_opt 1 / km_opt 4 horizontal mixing
(P3a item 5, steps (d) and (e); woof.grad.wrfplus.model with
diffusion.py): the deformation, smag2d_km and the u, v, w and theta rows
of the time-t fields into the held tendencies at the step start, their
map-factor coupling (on a specified domain after the folded relaxation)
and their add on every stage; on a moist state the qv, qc and qr rows
after rk_step 1's acoustic loop (with its muts) into the species' held
sources, which qv's boundary capture starts from.  Periodic and
specified, dry and moist.

Breakage this prevents:
- a mixing step that drifts from dycore.step's strict words (a row's
  field, mass or coefficient, the coupling, the fold's order, the cold
  diagnosis before the mixing);
- an adjoint that drops the mixing's share of the winds, theta, the
  species, the geopotential or the mass (the time-t or the acoustic one);
- a TL that is not the derivative of the mixing step;
- a mixing option the binding does not cover that is not refused.
"""
import pytest

from test_grad_wrfplus_model import BACKENDS, NO_MLX
from test_grad_wrfplus_model import _run as _run_periodic
from test_grad_wrfplus_spec_step import _run as _run_specified

MIX = dict(diff_opt=1, km_opt=4)
# periodic (over, steps, warm-up): a warm step, one with map factors,
# rotation, the hybrid coordinate and w_damp, and a cold chain of three;
# moist: a warm mapped step and a cold chain of two
PERIODIC = [(dict(MIX), 1, 2),
            (dict(MIX, _msf=1, _rot=1, hybrid_opt=2, w_damping=1), 1, 2),
            (dict(MIX), 3, 0),
            (dict(MIX, _moist=1, _msf=1, hybrid_opt=2), 1, 2),
            (dict(MIX, _moist=1), 2, 0)]
# specified (warm-up, steps, map factors, halo shift, moist): a warm step
# and a cold mapped chain with the halo off the ring; moist with qv's
# tables (1): a warm step and a cold mapped chain of two; moist without
# them (2, qv's held source is its mixing row): a warm step
SPECIFIED = [(2, 1, 0, 0.0, 0), (0, 2, 1, 3.0, 0), (2, 1, 0, 0.0, 1), (0, 2, 1, 3.0, 1),
             (2, 1, 0, 0.0, 2)]
NMOIST = (sum(1 for o, _, _ in PERIODIC if o.get("_moist"))
          + sum(1 for c in SPECIFIED if c[4]))

_PERIODIC_LOOP = f'''
def cases():
    for i, (over, nstep, warm) in enumerate({PERIODIC!r}):
        cfg, state = build(over, warm)
        geoms = M.step_geometries(state, cfg, nstep)
        inputs(geoms)
        yield f"periodic{{i}}", cfg, state, geoms, IN, (), PROG
'''
_SPECIFIED_LOOP = f'''
def cases():
    for i, (warm, nstep, msf, shift, moist) in enumerate({SPECIFIED!r}):
        cfg, state = build(warm, msf, shift, moist=int(moist > 0),
                           drop=("qv",) if moist == 2 else (), diff_opt=1, km_opt=4)
        geoms = M.step_geometries(state, cfg, nstep)
        IN, PAR = M.step_inputs(geoms[0]), M.step_parameters(geoms[0])
        yield (f"specified{{i}}", cfg, state, geoms, IN, PAR,
               [n in M.PROGNOSTICS for n in IN])
'''


def _both(backend, body, tmp_path):
    """``body`` over the periodic and the specified cases: each yields
    (key, cfg, state, geoms, inputs, parameters, identity mask)."""
    out = _run_periodic(backend, _PERIODIC_LOOP + body, tmp_path / "periodic")
    out.update(_run_specified(backend, _SPECIFIED_LOOP + body, tmp_path / "specified"))
    return out


_WORDS = '''
for key, cfg, state, geoms, IN, PAR, PROG in cases():
    assert all(g.mix is not None and g.held == ("u", "v", "w", "theta") for g in geoms)
    x = M.run_state(state, cfg, np.float32)
    nomix = copy(x)
    dx = zeros(x)
    for g in geoms:
        M.step_tl(x, dx, g)
        M.step_tl(nomix, zeros(nomix), dataclasses.replace(
            g, mix=None, held=g.fold if g.specified else (),
            couple=(g.fold if g.specified else ()) if g.couple else (),
            stages=tuple(dataclasses.replace(s, mix_q=()) for s in g.stages)))
        D.step(state, cfg)
    y = M.run_state(state, cfg, np.float32)
    words = lambda a: cp.asnumpy(a).view(np.uint32)
    for n in IN + ("w0", "u_pp", "mu_pp", "ww_ref") + (("ru_m", "ww_m") if cfg.moist else ()):
        out[f"{key}/{n}"] = int((words(x[n]) != words(y[n])).sum())
    # the mixing must move words, the moist rows each species'
    for n in ("u", "v", "w", "thp") + (M.SPECIES if cfg.moist else ()):
        out[f"nomix/{key}/{n}"] = int((words(nomix[n]) != words(y[n])).sum())
'''


@pytest.mark.parametrize("backend", BACKENDS)
def test_mixing_step_reproduces_strict_primal_words(backend, tmp_path):
    """The binary32 base is dycore.step's strict step with diff_opt 1 /
    km_opt 4 word for word, on every input and the scratch the next step
    reads; the same step without the mixing moves u, v, w and theta, and
    on a moist state each species."""
    out = _both(backend, _WORDS, tmp_path)
    nomix = {k: out.pop(k) for k in list(out) if k.startswith("nomix/")}
    ncase = len(PERIODIC) + len(SPECIFIED)
    assert len(nomix) == 4 * ncase + 3 * NMOIST and all(nomix.values()), nomix
    assert len(out) == len(PERIODIC) * (11 + 4) + len(SPECIFIED) * (12 + 4) + 5 * NMOIST
    bad = {k: v for k, v in out.items() if v}
    assert not bad, bad


_DOT_PERIODIC = '''
for dtn in DTYPES:
    for key, cfg, state, geoms, IN, PAR, PROG in cases():
        x0 = M.run_state(state, cfg, np.dtype(dtn))
        p, q = pert(x0, 10), pert(x0, 20)
        def tl(dx):
            return [y - d if k else y for y, d, k in zip(run_tl(x0, dx, geoms)[1], dx, PROG)]
        def ad(dy):
            return [y - d if k else y for y, d, k in zip(run_ad(x0, dy, geoms), dy, PROG)]
        out[f"{dtn}/{key}"] = K.dot_product(tl, ad, p, q, terms=dtn == "float32")
'''
_DOT_SPECIFIED = '''
for dtn in DTYPES:
    for key, cfg, state, geoms, IN, PAR, PROG in cases():
        x0 = M.run_state(state, cfg, np.dtype(dtn))
        ALL = IN + PAR
        p, q = pert(x0, ALL, 10), pert(x0, IN, 20)
        def tl(dx):
            d = run_tl(x0, ALL, dx, geoms)[1]
            return [d[n] - a if k else d[n] for n, a, k in zip(IN, dx, PROG)]
        def ad(dy):
            a = run_ad(x0, ALL, IN, dy, geoms)
            return ([a[n] - b if k else a[n] for n, b, k in zip(IN, dy, PROG)]
                    + [a[n] for n in PAR])
        out[f"{dtn}/{key}"] = K.dot_product(tl, ad, p, q, terms=dtn == "float32")
'''


@pytest.mark.parametrize("backend", BACKENDS)
def test_mixing_step_dot_product(backend, tmp_path):
    """<(T - I) dx, dy> == <dx, (T - I)^T dy> over the step's inputs (and
    on a specified domain the boundary tables): 1e-12 in binary64, 1e-5
    (terms-scaled) in binary32."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    head = f"DTYPES = {dtypes!r}\n"
    out = _run_periodic(backend, _PERIODIC_LOOP + head + _DOT_PERIODIC, tmp_path / "p")
    out.update(_run_specified(backend, _SPECIFIED_LOOP + head + _DOT_SPECIFIED,
                              tmp_path / "s"))
    assert len(out) == (len(PERIODIC) + len(SPECIFIED)) * len(dtypes)
    for key, rel in out.items():
        limit = 1e-12 if key.startswith("float64") else 1e-5
        assert rel < limit, (key, rel)


_BASE = '''
for dtn in DTYPES:
    for key, cfg, state, geoms, IN, PAR, PROG in cases():
        x0 = M.run_state(state, cfg, np.dtype(dtn))
        ta, tb = [[] for _ in geoms], [[] for _ in geoms]
        xa, d = copy(x0), zeros(x0)
        for n in IN:
            d[n][...] = cp.asarray(np.random.default_rng(40).standard_normal(
                x0[n].shape)).astype(x0[n].dtype) * float(cp.abs(x0[n]).max() * 1e-3 + 1e-12)
        for g, tape in zip(geoms, ta):
            M.step_tl(xa, d, g, tape)
        xb, scr = copy(x0), None
        for g, tape in zip(geoms, tb):
            scr = M.step_base(xb, g, tape, scr)
        words = lambda a: cp.asnumpy(a).view(np.uint64 if dtn == "float64" else np.uint32)
        bad = sum(int((words(xa[n]) != words(xb[n])).sum()) for n in xa)
        for sa, sb in zip(ta, tb):
            assert len(sa) == len(sb)
            for da, db in zip(sa, sb):
                assert sorted(da) == sorted(db)
                bad += sum(int((words(da[n]) != words(db[n])).sum()) for n in da)
        out[f"{dtn}/{key}"] = bad
'''


@pytest.mark.parametrize("backend", BACKENDS)
def test_mixing_base_only_step_is_the_joint_base(backend, tmp_path):
    """step_base advances the mixing step's base to the joint TL build's
    words and tapes the same snapshots."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _both(backend, f"DTYPES = {dtypes!r}\n" + _BASE, tmp_path)
    assert len(out) == (len(PERIODIC) + len(SPECIFIED)) * len(dtypes)
    bad = {k: v for k, v in out.items() if v}
    assert not bad, bad


_FD_PERIODIC = '''
for key, cfg, state, geoms, IN, PAR, PROG in cases():
    x0 = M.run_state(state, cfg, np.float64)
    p = pert(x0, 30)
    def model(xs):
        t = copy(x0)
        for n, a in zip(IN, xs):
            t[n][...] = a
        x = run_tl(t, [cp.zeros_like(t[n]) for n in IN], geoms)[0]
        return [x[n] for n in IN]
    out[key] = K.tl_convergence(model, lambda dx: run_tl(x0, dx, geoms)[1],
                                [x0[n] for n in IN], p, [1e-1, 1e-3, 1e-5])
'''
_FD_SPECIFIED = '''
for key, cfg, state, geoms, IN, PAR, PROG in cases():
    ALL = IN + PAR
    x0 = M.run_state(state, cfg, np.float64)
    p = pert(x0, ALL, 30)
    def model(xs):
        t = copy(x0)
        for n, a in zip(ALL, xs):
            t[n][...] = a
        x = run_tl(t, ALL, [cp.zeros_like(t[n]) for n in ALL], geoms)[0]
        return [x[n] for n in IN]
    def tl(dx):
        d = run_tl(x0, ALL, dx, geoms)[1]
        return [d[n] for n in IN]
    out[key] = K.tl_convergence(model, tl, [x0[n] for n in ALL], p, [1e-1, 1e-3, 1e-5])
'''


@pytest.mark.parametrize("backend", NO_MLX)
def test_mixing_step_tl_matches_finite_differences(backend, tmp_path):
    """||NL(x + a dx) - NL(x)|| / ||a TL dx|| -> 1 in binary64."""
    out = _run_periodic(backend, _PERIODIC_LOOP + _FD_PERIODIC, tmp_path / "p")
    out.update(_run_specified(backend, _SPECIFIED_LOOP + _FD_SPECIFIED, tmp_path / "s"))
    assert len(out) == len(PERIODIC) + len(SPECIFIED)
    for key, rows in out.items():
        misfits = [abs(r - 1.0) for _, r in rows]
        assert min(misfits) < 1e-6, (key, rows)
        assert misfits[0] < 1e-2, (key, rows)


@pytest.mark.parametrize("backend", NO_MLX[:1])
def test_mixing_step_refuses_what_it_does_not_cover(backend, tmp_path):
    """Another km_opt under diff_opt 1, the TC horizontal length, m_opt,
    passive tracers and a resumed state without WRF's t_init are refused
    by name; diff_opt 1 / km_opt 4 is accepted, dry or moist, and so is
    diff_opt 2 / km_opt 4 (dry)."""
    out = _run_periodic(backend, '''
cases = {"ok": dict(MIX), "diff2": dict(MIX, diff_opt=2), "km3": dict(MIX, km_opt=3),
         "moist": dict(MIX, moist=True, moist_adv_opt=0),
         "tc": dict(MIX, tc_horizontal_length_m=500.0), "mopt": dict(MIX, m_opt=1),
         "tracer": dict(MIX, tracer_opt=2), "resumed": dict(MIX), "carried": dict(MIX)}
for key, over in cases.items():
    try:
        cfg = H.make_config(8, 6, 6, h_sca_adv_order=5, terrain_opt=1, **over)
        st = H.make_state(cfg, 1, geography=H.neutral_geography(cfg))
        if key in ("resumed", "carried"):   # a restart, with or without WRF's t_init
            D.initialize_coordinate_reference(st, cfg)
            assert "diff1_theta_initial" in st._scratch
            if key == "resumed":
                del st._scratch["diff1_theta_initial"]
            st.elapsed_seconds = 60.0
        g = M.step_geometry(st, cfg)
        out[key] = f"accepted mix={g.mix is not None}"
    except (NotImplementedError, ValueError) as e:
        out[key] = f"{type(e).__name__}: {e}"
'''.replace("MIX", repr(MIX)), tmp_path)
    assert out["ok"] == out["carried"] == "accepted mix=True", out
    # diff_opt 2 / km_opt 4 has its own binding (test_grad_wrfplus_mix2_step)
    assert out["diff2"] == "accepted mix=True", out
    assert "mixing package other than diff_opt 1 / km_opt 4" in out["km3"], out
    assert out["moist"] == "accepted mix=True", out
    assert "tc_horizontal_length_m" in out["tc"], out
    assert "m_opt" in out["mopt"], out
    assert "passive tracers" in out["tracer"], out
    assert "without diff1_theta_initial" in out["resumed"], out
