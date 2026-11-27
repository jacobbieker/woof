"""The step's TL/AD with WRF's diff_opt 2 mixing (P3a item 2b-3;
woof.grad.wrfplus.model with diffusion.py): at the step start the
deformation, the coefficients (km_opt 4: smag2d_km with its slope
limiter; km_opt 3: calculate_N2 and smag_km), the u, v, w and theta
horizontal_diffusion_2 rows of the time-t fields into the held
tendencies with their open strips cleared, vertical_diffusion_2's rows
with the surface terms of the isfflx case (the cd0 stress and the
prescribed heat flux), the coupling to the map factors (on a specified
domain after the folded relaxation) and the held add on every stage.
Periodic (with WRF's cold seam slope, diff_metrics_seam_slope false, and
without) and specified, dry.

Breakage this prevents:
- a diff_opt 2 step that drifts from dycore.step's strict words (a
  coefficient, a row's field, K or surface term, the strip clearing, the
  coupling, the fold's order, the seam flag);
- an adjoint that drops the mixing's share of the winds, theta or the
  geopotential;
- a TL that is not the derivative of the mixing step;
- a diff_opt 2 option the binding does not cover that is not refused.
"""
import pytest

from test_grad_wrfplus_model import BACKENDS, NO_MLX
from test_grad_wrfplus_model import _run as _run_periodic
from test_grad_wrfplus_spec_step import _run as _run_specified

K4 = dict(diff_opt=2, km_opt=4)
K3 = dict(diff_opt=2, km_opt=3)
# periodic (over, steps, warm-up): km_opt 4 warm (isfflx 1: no surface
# term) and mapped with the hybrid coordinate, isfflx 0 (the cd0 stress and
# the heat flux) and WRF's cold seam slope; km_opt 3 warm with isfflx 2
# (the heat flux alone), a cold chain of two with the isotropic length and
# the cold seam
SURF = dict(isfflx=0, tke_drag_coefficient=2e-3, tke_heat_flux=0.05)
PERIODIC = [(dict(K4), 1, 2),
            (dict(K4, _msf=1, hybrid_opt=2, diff_metrics_seam_slope=False, **SURF), 1, 2),
            (dict(K3, isfflx=2, tke_heat_flux=0.05), 1, 2),
            (dict(K3, mix_isotropic=1, diff_metrics_seam_slope=False, **SURF), 2, 0)]
# specified (warm-up, steps, map factors, overrides)
SPECIFIED = [(2, 1, 0, dict(K4, **SURF)), (0, 2, 1, dict(K3, **SURF))]

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
    for i, (warm, nstep, msf, over) in enumerate({SPECIFIED!r}):
        cfg, state = build(warm, msf, 0.0, **over)
        geoms = M.step_geometries(state, cfg, nstep)
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
    assert all(g.mix is not None and g.mix.diff2 and g.held == ("u", "v", "w", "theta")
               for g in geoms)
    x = M.run_state(state, cfg, np.float32)
    nomix = copy(x)
    dx = zeros(x)
    for g in geoms:
        M.step_tl(x, dx, g)
        M.step_tl(nomix, zeros(nomix), dataclasses.replace(
            g, mix=None, held=g.fold if g.specified else (),
            couple=(g.fold if g.specified else ()) if g.couple else ()))
        D.step(state, cfg)
    y = M.run_state(state, cfg, np.float32)
    words = lambda a: cp.asnumpy(a).view(np.uint32)
    for n in IN + ("w0", "u_pp", "mu_pp", "ww_ref"):
        out[f"{key}/{n}"] = int((words(x[n]) != words(y[n])).sum())
    # the mixing must move words
    for n in ("u", "v", "w", "thp"):
        out[f"nomix/{key}/{n}"] = int((words(nomix[n]) != words(y[n])).sum())
'''


@pytest.mark.parametrize("backend", NO_MLX)
def test_mixing_step_reproduces_strict_primal_words(backend, tmp_path):
    """The binary32 base is dycore.step's strict step with diff_opt 2
    (km_opt 3 and 4) word for word, on every input and the scratch the
    next step reads; the same step without the mixing moves u, v, w and
    theta.  Host and CUDA; Metal is not checked here."""
    out = _both(backend, _WORDS, tmp_path)
    nomix = {k: out.pop(k) for k in list(out) if k.startswith("nomix/")}
    assert len(nomix) == 4 * (len(PERIODIC) + len(SPECIFIED)) and all(nomix.values()), nomix
    assert len(out) == len(PERIODIC) * (11 + 4) + len(SPECIFIED) * (12 + 4)
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
    """step_base advances the diff_opt 2 step's base to the joint TL
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
def test_mixing_step_tl_matches_finite_differences(backend, tmp_path):
    """||NL(x + a dx) - NL(x)|| / ||a TL dx|| -> 1 in binary64.  The
    cases run in the corrected configuration's step (the strict build),
    and the seam cases take diff_metrics_seam_slope false, WRF's exact
    arm, whose cold zero is a constant and so differentiable."""
    out = _run_periodic(backend, _PERIODIC_LOOP + _FD_PERIODIC, tmp_path / "p")
    out.update(_run_specified(backend, _SPECIFIED_LOOP + _FD_SPECIFIED, tmp_path / "s"))
    assert len(out) == len(PERIODIC) + len(SPECIFIED)
    for key, rows in out.items():
        misfits = [abs(r - 1.0) for _, r in rows]
        assert min(misfits) < 1e-6, (key, rows)
        assert misfits[0] < 1e-2, (key, rows)


@pytest.mark.parametrize("backend", NO_MLX[:1])
def test_mixing_step_refuses_what_it_does_not_cover(backend, tmp_path):
    """Under diff_opt 2: km_opt 1, 2 and 5, a moist state, a PBL scheme,
    isfflx 3, the TC horizontal length, m_opt and passive tracers are
    refused by name; km_opt 3 and 4 are accepted."""
    out = _run_periodic(backend, '''
cases = {"km4": dict(K4), "km3": dict(K3), "km1": dict(K4, km_opt=1, khdif=100.0, kvdif=1.0),
         "km2": dict(K4, km_opt=2), "km5": dict(K4, km_opt=5),
         "moist": dict(K4, moist=True, moist_adv_opt=0), "isfflx": dict(K4, isfflx=3),
         "tc": dict(K4, tc_horizontal_length_m=500.0), "mopt": dict(K4, m_opt=1),
         "tracer": dict(K4, tracer_opt=2)}
for key, over in cases.items():
    try:
        cfg = H.make_config(8, 6, 6, h_sca_adv_order=5, terrain_opt=1, **over)
        st = H.make_state(cfg, 1, geography=H.neutral_geography(cfg))
        g = M.step_geometry(st, cfg)
        out[key] = f"accepted mix={g.mix is not None}"
    except Exception as e:
        out[key] = f"{type(e).__name__}: {e}"
'''.replace("K4", repr(K4)).replace("K3", repr(K3)), tmp_path)
    assert out["km4"] == out["km3"] == "accepted mix=True", out
    for key in ("km1", "km2", "km5"):
        assert "mixing package other than" in out[key], out
        assert "diff_opt 2 with km_opt 1, 2 or 5" in out[key], out
    assert "moist diff_opt 2" in out["moist"], out
    assert "isfflx" in out["isfflx"], out
    assert "tc_horizontal_length_m" in out["tc"], out
    assert "m_opt" in out["mopt"], out
    assert "passive tracers" in out["tracer"], out


def test_refuses_the_other_diffusion_arithmetic(tmp_path):
    """GPUWM_WRF_EXACT_DIFFUSION=0 (the strict build without the
    EXACT_C_DIFFUSION statements the TL/AD port) is refused by name."""
    import os
    import subprocess
    import sys
    from test_grad_wrfplus_model import ROOT, STRICT
    env = dict(os.environ, WOOF_BACKEND="host", PYTHONPATH=str(ROOT),
               WOOF_HOST_CACHE=str(tmp_path / "hc"), **STRICT)
    env["GPUWM_WRF_EXACT_DIFFUSION"] = "0"
    script = (
        "from tilestream import harness as H\n"
        "import woof.core.resident_admission as RA\n"
        "RA.device_free_bytes = lambda: None   # the host facade reports no card\n"
        "from woof.grad.wrfplus import model as M\n"
        f"cfg = H.make_config(8, 6, 6, h_sca_adv_order=5, terrain_opt=1, **{K4!r})\n"
        "st = H.make_state(cfg, 1, geography=H.neutral_geography(cfg))\n"
        "try:\n"
        "    M.step_geometry(st, cfg)\n"
        "    print('accepted')\n"
        "except NotImplementedError as e:\n"
        "    print(e)\n")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    assert "GPUWM_WRF_EXACT_DIFFUSION=0" in done.stdout, done.stdout
