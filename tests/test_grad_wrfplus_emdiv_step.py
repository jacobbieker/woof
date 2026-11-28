"""The step's TL/AD with WRF's external-mode filter, emdiv > 0 (P3a
window request 1; woof.grad.wrfplus.model): advance_uv's MUDF_XY =
-emdiv*dx*(MUDF(i) - MUDF(i-1))/msfuy from the previous substep's mudf,
which advance_mu_t writes and small_step_prep zeroes on rk_step 1 only, so
stages 2 and 3 start from the previous stage's last mudf.  Periodic and
specified, dry and moist, with map factors and with the diff_opt 1 /
km_opt 4 mixing.

Breakage this prevents:
- a filtered step that drifts from dycore.step's strict words (the scale
  rounded otherwise than acoustic._uv_emdiv_args, mudf not carried
  between stages, or reset on a stage other than the first);
- an adjoint that drops the mudf chain (advance_uv's read of the previous
  substep's mudf, advance_mu_t's write, the rk_step 1 reset);
- a TL that is not the derivative of the filtered step.

advance_uv's own emdiv branch is checked against WRFPLUS g_/a_advance_uv
in test_grad_wrfplus_advance_uv.py, advance_mu_t's mudf in
test_grad_wrfplus_advance_mu_t.py, and the substep chain carrying mudf
against WRFPLUS in test_grad_wrfplus_emdiv_chain.py.
"""
import pytest

from test_grad_wrfplus_model import BACKENDS, NO_MLX
from test_grad_wrfplus_model import _run as _run_periodic
from test_grad_wrfplus_spec_step import _run as _run_specified

EM = dict(emdiv=0.01)
# periodic (over, steps, warm-up): a warm step, one with map factors,
# rotation, the hybrid coordinate, w_damp and six substeps, one with the
# mixing, a cold chain of three and a moist cold chain of two
PERIODIC = [(dict(EM), 1, 2),
            (dict(EM, _msf=1, _rot=1, hybrid_opt=2, w_damping=1, time_step_sound=6), 1, 2),
            (dict(EM, diff_opt=1, km_opt=4), 1, 2),
            (dict(EM), 3, 0),
            (dict(EM, _moist=1, _msf=1), 2, 0)]
# specified (warm-up, steps, map factors, halo shift, moist): a warm step,
# a cold mapped chain with the halo off the ring and a moist warm step
SPECIFIED = [(2, 1, 0, 0.0, 0), (0, 2, 1, 3.0, 0), (2, 1, 0, 0.0, 1)]
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
        cfg, state = build(warm, msf, shift, moist=moist, **{EM!r})
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
def off(g):
    # the same step without the filter: no MUDF_XY, no mudf written
    def sub(t):
        return (dataclasses.replace(t[0], use_emdiv=0, xscale=0.0, yscale=0.0),
                dataclasses.replace(t[1], write_mudf=0), t[2])
    return dataclasses.replace(g, emdiv=False, stages=tuple(
        dataclasses.replace(s, first=sub(s.first), later=sub(s.later)) for s in g.stages))
for key, cfg, state, geoms, IN, PAR, PROG in cases():
    assert all(g.emdiv and all(s.first[0].use_emdiv and s.later[1].write_mudf
                               for s in g.stages) for g in geoms)
    x = M.run_state(state, cfg, np.float32)
    plain = copy(x)
    dx = zeros(x)
    for g in geoms:
        M.step_tl(x, dx, g)
        M.step_tl(plain, zeros(plain), off(g))
        D.step(state, cfg)
    y = M.run_state(state, cfg, np.float32)
    words = lambda a: cp.asnumpy(a).view(np.uint32)
    for n in IN + ("w0", "u_pp", "mu_pp", "ww_ref") + (("ru_m", "ww_m") if cfg.moist else ()):
        out[f"{key}/{n}"] = int((words(x[n]) != words(y[n])).sum())
    # the step's last mudf is the primal's acoustic_mudf scratch
    out[f"{key}/mudf"] = int((words(x["mudf"]) != words(
        state.scratch(state.mup.shape, "acoustic_mudf"))).sum())
    # the filter must move the winds and the mass
    for n in ("u", "v", "mup"):
        out[f"plain/{key}/{n}"] = int((words(plain[n]) != words(y[n])).sum())
'''


@pytest.mark.parametrize("backend", BACKENDS)
def test_emdiv_step_reproduces_strict_primal_words(backend, tmp_path):
    """The binary32 base is dycore.step's strict step with emdiv 0.01
    word for word, on every input, the scratch the next step reads and
    the last mudf; the same step without the filter moves u, v and mu."""
    out = _both(backend, _WORDS, tmp_path)
    plain = {k: out.pop(k) for k in list(out) if k.startswith("plain/")}
    ncase = len(PERIODIC) + len(SPECIFIED)
    assert len(plain) == 3 * ncase and all(plain.values()), plain
    assert (len(out) == len(PERIODIC) * (11 + 5) + len(SPECIFIED) * (12 + 5)
            + 5 * NMOIST)
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
def test_emdiv_step_dot_product(backend, tmp_path):
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


_CONSUMES = '''
for key, cfg, state, geoms, IN, PAR, PROG in cases():
    x0 = M.run_state(state, cfg, np.float64)
    dy = pert(x0, 50)
    tapes = [[] for _ in geoms]
    x, _ = run_tl(x0, [cp.zeros_like(x0[n]) for n in IN], geoms, tapes)
    res = []
    for seed in (0.0, 1.0):
        a = zeros(x0)
        for n, v in zip(IN, dy):
            a[n][...] = v
        a["mudf"][...] = seed
        xs, ts = x, [list(t) for t in tapes]
        for k in reversed(range(len(geoms))):
            xs = M.step_ad(a, xs, ts[k], geoms[k])
        res.append(a)
    a0, a1 = res
    out[key] = [float(cp.abs(a0["mudf"]).max()), float(cp.abs(a1["mudf"]).max()),
                max(float(cp.abs(a1[n] - a0[n]).max()) for n in IN)]
'''


@pytest.mark.parametrize("backend", NO_MLX[:1])
def test_emdiv_step_adjoint_consumes_mudf(backend, tmp_path):
    """mudf is the step's internal carrier: the rk_step 1 reset consumes
    its adjoint, so none is left on the step's input mudf, and an adjoint
    seeded on the last substep's mudf reaches the inputs through
    advance_mu_t's write (mudf = dmdt + mu_tend)."""
    out = _run_periodic(backend, _PERIODIC_LOOP + _CONSUMES, tmp_path)
    assert len(out) == len(PERIODIC)
    for key, (left0, left1, moved) in out.items():
        assert left0 == left1 == 0.0, (key, out[key])
        assert moved > 0.0, (key, out[key])


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
def test_emdiv_base_only_step_is_the_joint_base(backend, tmp_path):
    """step_base advances the filtered step's base to the joint TL
    build's words and tapes the same snapshots."""
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
def test_emdiv_step_tl_matches_finite_differences(backend, tmp_path):
    """||NL(x + a dx) - NL(x)|| / ||a TL dx|| -> 1 in binary64."""
    out = _run_periodic(backend, _PERIODIC_LOOP + _FD_PERIODIC, tmp_path / "p")
    out.update(_run_specified(backend, _SPECIFIED_LOOP + _FD_SPECIFIED, tmp_path / "s"))
    assert len(out) == len(PERIODIC) + len(SPECIFIED)
    for key, rows in out.items():
        misfits = [abs(r - 1.0) for _, r in rows]
        assert min(misfits) < 1e-6, (key, rows)
        assert misfits[0] < 1e-2, (key, rows)


@pytest.mark.parametrize("backend", NO_MLX[:1])
def test_emdiv_step_is_no_longer_refused(backend, tmp_path):
    """emdiv > 0 is accepted (WRF's default 0.01 and a stronger filter),
    the scale is acoustic._uv_emdiv_args' REAL -emdiv*dx, and emdiv 0
    still builds the unfiltered step."""
    out = _run_periodic(backend, '''
import woof.core.acoustic as A
for key, em in (("off", 0.0), ("wrf", 0.01), ("strong", 0.1)):
    cfg = H.make_config(8, 6, 6, h_sca_adv_order=5, terrain_opt=1, emdiv=em)
    st = H.make_state(cfg, 1, geography=H.neutral_geography(cfg))
    g = M.step_geometry(st, cfg)
    uv, mt = g.stages[0].later[0], g.stages[0].later[1]
    want = A._uv_emdiv_args(st, cfg, cp.zeros_like(st.mup)) if em else ()
    out[key] = [bool(g.emdiv), uv.use_emdiv, mt.write_mudf,
                bool(not em or (np.float32(uv.xscale) == want[3]
                                and np.float32(uv.yscale) == want[4]))]
''', tmp_path)
    assert out["off"] == [False, 0, 0, True], out
    assert out["wrf"] == out["strong"] == [True, 1, 1, True], out
