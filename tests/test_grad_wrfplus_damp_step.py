"""The step's TL/AD with damp_opt 3, WRF's implicit Rayleigh w damper in
advance_w (P3a window option 2; woof.grad.wrfplus.model with the damper
in acoustic.advance_w_tl/_ad).  Periodic and specified, dry and moist,
with map factors, rotation and the hybrid coordinate.

Breakage this prevents:
- a damped step that drifts from dycore.step's strict words (the
  layer's heights, the coefficient's float32 rounding, the hybrid mass
  of the damp target, the damper's place between the back substitution
  and the phi'' update);
- an adjoint that drops the damper's share of w, the geopotential
  (heights), the column mass or the reference w (state.w);
- a TL that is not the derivative of the damped step;
- the fork's noaa_wrf39 upper-wind limiter (on only with damp_opt 3)
  run under a step adjoint that does not linearise it.
"""
import pytest

from test_grad_wrfplus_model import BACKENDS, NO_MLX
from test_grad_wrfplus_model import _run as _run_periodic
from test_grad_wrfplus_spec_step import _run as _run_specified

# harness ztop is 8 km: the top 3 km damped, at WRF's default dampcoef
DAMP = dict(damp_opt=3, zdamp=3000.0, dampcoef=0.2)
# periodic (over, steps, warm-up): a warm step; one with map factors,
# rotation, the hybrid coordinate and w_damp; a cold chain of three; a
# warm moist mapped step
PERIODIC = [(dict(DAMP), 1, 2),
            (dict(DAMP, _msf=1, _rot=1, hybrid_opt=2, w_damping=1), 1, 2),
            (dict(DAMP), 3, 0),
            (dict(DAMP, _moist=1, _msf=1), 1, 2)]
# specified (warm-up, steps, map factors, halo shift, moist): a warm step
# and a cold mapped moist chain of two with the halo off the ring
SPECIFIED = [(2, 1, 0, 0.0, 0), (0, 2, 1, 3.0, 1)]
NMOIST = (sum(1 for o, _, _ in PERIODIC if o.get("_moist"))
          + sum(1 for c in SPECIFIED if c[4]))

_HELPERS = '''
def damped(geoms):
    return all(t[2].dampmag > 0 and t[2].zdamp == 3000.0
               for g in geoms for s in g.stages for t in (s.first, s.later))
def undamped(g):
    w0 = lambda t: t[:2] + (dataclasses.replace(t[2], dampmag=0.0),)
    return dataclasses.replace(g, stages=tuple(
        dataclasses.replace(s, first=w0(s.first), later=w0(s.later)) for s in g.stages))
'''
_PERIODIC_LOOP = _HELPERS + f'''
def cases():
    for i, (over, nstep, warm) in enumerate({PERIODIC!r}):
        cfg, state = build(over, warm)
        geoms = M.step_geometries(state, cfg, nstep)
        inputs(geoms)
        assert damped(geoms)
        yield f"periodic{{i}}", cfg, state, geoms, IN, (), PROG
'''
_SPECIFIED_LOOP = _HELPERS + f'''
def cases():
    for i, (warm, nstep, msf, shift, moist) in enumerate({SPECIFIED!r}):
        cfg, state = build(warm, msf, shift, moist=moist, **{DAMP!r})
        geoms = M.step_geometries(state, cfg, nstep)
        assert damped(geoms)
        IN, PAR = M.step_inputs(geoms[0]), M.step_parameters(geoms[0])
        yield (f"specified{{i}}", cfg, state, geoms, IN, PAR,
               [n in M.PROGNOSTICS for n in IN])
'''


def _both(backend, body, tmp_path):
    out = _run_periodic(backend, _PERIODIC_LOOP + body, tmp_path / "periodic")
    out.update(_run_specified(backend, _SPECIFIED_LOOP + body, tmp_path / "specified"))
    return out


_WORDS = '''
for key, cfg, state, geoms, IN, PAR, PROG in cases():
    x = M.run_state(state, cfg, np.float32)
    nodamp = copy(x)
    dx = zeros(x)
    for g in geoms:
        M.step_tl(x, dx, g)
        M.step_tl(nodamp, zeros(nodamp), undamped(g))
        D.step(state, cfg)
    y = M.run_state(state, cfg, np.float32)
    words = lambda a: cp.asnumpy(a).view(np.uint32)
    for n in IN + ("w0", "u_pp", "mu_pp", "ww_ref") + (("ru_m", "ww_m") if cfg.moist else ()):
        out[f"{key}/{n}"] = int((words(x[n]) != words(y[n])).sum())
    # the damper must move words: w directly, the geopotential through it
    for n in ("w", "php"):
        out[f"nodamp/{key}/{n}"] = int((words(nodamp[n]) != words(y[n])).sum())
'''


@pytest.mark.parametrize("backend", BACKENDS)
def test_damped_step_reproduces_strict_primal_words(backend, tmp_path):
    """The binary32 base is dycore.step's strict step with damp_opt 3 word
    for word, on every input and the scratch the next step reads; the
    same step with the damper off moves w and the geopotential."""
    out = _both(backend, _WORDS, tmp_path)
    nodamp = {k: out.pop(k) for k in list(out) if k.startswith("nodamp/")}
    assert len(nodamp) == 2 * (len(PERIODIC) + len(SPECIFIED)) and all(nodamp.values()), nodamp
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
def test_damped_step_dot_product(backend, tmp_path):
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
def test_damped_step_tl_matches_finite_differences(backend, tmp_path):
    """||NL(x + a dx) - NL(x)|| / ||a TL dx|| -> 1 in binary64."""
    out = _run_periodic(backend, _PERIODIC_LOOP + _FD_PERIODIC, tmp_path / "p")
    out.update(_run_specified(backend, _SPECIFIED_LOOP + _FD_SPECIFIED, tmp_path / "s"))
    assert len(out) == len(PERIODIC) + len(SPECIFIED)
    for key, rows in out.items():
        misfits = [abs(r - 1.0) for _, r in rows]
        assert min(misfits) < 1e-6, (key, rows)
        assert misfits[0] < 1e-2, (key, rows)


@pytest.mark.parametrize("backend", NO_MLX[:1])
def test_damped_step_refuses_what_it_does_not_cover(backend, tmp_path):
    """damp_opt 3 is accepted (its dampmag on every substep); with the
    fork's noaa_wrf39 upper-wind limiter it is refused by name."""
    out = _run_periodic(backend, '''
cases = {"ok": dict(DAMP), "off": dict(DAMP, damp_opt=0),
         "limiter": dict(DAMP, upper_wind_limiter_form="noaa_wrf39")}
for key, over in cases.items():
    try:
        cfg = H.make_config(8, 6, 6, h_sca_adv_order=5, terrain_opt=1, **over)
        st = H.make_state(cfg, 1, geography=H.neutral_geography(cfg))
        g = M.step_geometry(st, cfg)
        out[key] = "accepted " + " ".join(sorted({repr(t[2].dampmag) for s in g.stages
                                                  for t in (s.first, s.later)}))
    except (NotImplementedError, ValueError) as e:
        out[key] = f"{type(e).__name__}: {e}"
'''.replace("DAMP", repr(DAMP)), tmp_path)
    assert out["ok"].startswith("accepted ") and "0.0" not in out["ok"].split(), out
    assert out["off"] == "accepted 0.0", out
    assert "damp_opt 3 with upper_wind_limiter_form noaa_wrf39" in out["limiter"], out
