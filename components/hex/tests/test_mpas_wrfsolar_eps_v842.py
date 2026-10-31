"""WRF-Solar EPS on the MPAS-A v8.4.2 suite route (&wrfsolar_eps) and
MAD-WRF's refusal there (woof/hex/mpasphys/wrfsolar_eps_v842.py).

* the option resolves off by default, on with a scheme switch, and is
  refused where its seat does not run, by name for MYNN/Noah hooks, and on
  the RRFS fork route; MAD-WRF's keys are refused by name everywhere;
* the patterns follow WRF's cadence (first step, then every stepsp), are
  bounded by the table's cutoff, and a restored provider continues word for
  word;
* each seat applies WRF's own statements (woof.core.wrfsolar_eps, held to
  compiled WRF) to the MPAS operands: the FARMS service's copies, the Deng
  service's copies, the Thompson driver's fields (halo cells untouched) and
  the cloudiness copies.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys import resolve_namelist
from woof.hex.mpasphys import wrfsolar_eps_v842 as mpeps


def _namelist(eps=None, physics=None, backends=None):
    lines = ["&physics", " config_physics_suite='convection_permitting'"]
    lines += [f" {k}={v!r}" for k, v in (physics or {}).items()]
    lines += ["/"]
    if backends:
        lines += ["&hex_mpasphys"] + [f" {k}={v!r}" for k, v in backends.items()] + ["/"]
    if eps is not None:
        lines += ["&wrfsolar_eps"]
        for k, v in eps.items():
            v = (".true." if v else ".false.") if isinstance(v, bool) else repr(v)
            lines.append(f" {k}={v}")
        lines += ["/"]
    return "\n".join(lines) + "\n"


THOM = dict(config_multi_perturb=True, config_pert_thom=True, config_pert_thom_qv=1.0)


def test_off_by_default_and_on_with_a_switch():
    assert resolve_namelist(_namelist(), default_backend="hex-cuda").wrfsolar_eps is None
    config = resolve_namelist(_namelist(THOM), default_backend="hex-cuda")
    o = config.wrfsolar_eps
    assert o.switches == ("pert_thom",) and o.get("pert_thom_qv") == 1.0
    assert config.receipt()["wrfsolar_eps"]["switches"] == ["pert_thom"]
    assert "config_multi_perturb" not in config.options


@pytest.mark.parametrize("eps, backends, physics, match", [
    (dict(config_pert_thom=True), None, None, "without config_multi_perturb"),
    (dict(config_multi_perturb=True), None, None, "perturbs nothing"),
    (dict(config_multi_perturb=True, config_pert_mynn=True), None, None, "MYNN and Noah"),
    (dict(config_multi_perturb=True, config_pert_thom=True, config_spdtt=1.0), None, None, "has no option"),
    (dict(config_multi_perturb=True, config_pert_farms=True), None, None, "FARMS service"),
    (dict(config_multi_perturb=True, config_pert_deng=True), None, None, "Deng service"),
    (THOM, None, dict(config_microp_scheme="mp_wsm6"), "mp_thompson"),
    (dict(config_multi_perturb=True, config_pert_cld3=True), None,
     dict(config_radt_cld_scheme="cld_fraction"), "cld_fraction_thompson"),
    (THOM, dict(config_microp_backend="native-cpu"), None, "hex-cuda"),
])
def test_refusals_name_the_breakage(eps, backends, physics, match):
    with pytest.raises(ValueError, match=match):
        resolve_namelist(_namelist(eps, physics, backends), default_backend="hex-cuda")


@pytest.mark.parametrize("group, key", [("physics", "config_madwrf_opt"),
                                        ("wrfsolar_eps", "config_madwrf_cldinit"),
                                        ("physics", "madwrf_opt")])
def test_madwrf_is_refused_by_name(group, key):
    text = _namelist() + f"&{group}\n {key}=1\n/\n" if group != "physics" else \
        _namelist(physics={key: 1})
    with pytest.raises(ValueError, match="MAD-WRF is not ported to the MPAS route"):
        resolve_namelist(text, default_backend="hex-cuda")


def test_rrfs_fork_route_refuses_both():
    from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist as resolve_rrfs
    base = "&physics\n config_physics_suite='hrrrv5'\n/\n"
    with pytest.raises(ValueError, match="not available on the RRFS fork route"):
        resolve_rrfs(base + "&wrfsolar_eps\n config_multi_perturb=.true.\n config_pert_thom=.true.\n/\n")
    with pytest.raises(ValueError, match="MAD-WRF"):
        resolve_rrfs(base + "&wrfsolar_eps\n config_madwrf_opt=1\n/\n")


def test_door_override_lands_in_the_group():
    from woof.hex.mpasphys.forecast_v842 import native_route
    arguments = SimpleNamespace(physics_namelist=None, physics_suite="convection_permitting",
                                physics=["config_multi_perturb=true", "config_pert_thom=true",
                                         "config_pert_thom_qc=0.5"])
    config, text = native_route(arguments)
    assert config.wrfsolar_eps.get("pert_thom_qc") == 0.5
    assert "&wrfsolar_eps" in text


# ---------------------------------------------------------------------------
# on a device
# ---------------------------------------------------------------------------

def _resource(cp, degree=12, n=640):
    """Equal-area Fibonacci points (an independent quadrature, not a mesh)."""
    from woof.hex.stochastic import _kernel
    lat = cp.arcsin(1 - 2 * (cp.arange(n, dtype=cp.float64) + 0.5) / n)
    lon = cp.arange(n, dtype=cp.float64) * (math.pi * (3 - math.sqrt(5)))
    modes = (degree + 1) ** 2 - 1
    b = cp.empty((n, modes), cp.float32)
    _kernel("harmonics")((((n * (degree + 1)) + 127) // 128,), (128,),
                         (lat, lon, b, np.int32(n), np.int32(degree)))
    return SimpleNamespace(modes=modes, radius=6371220., basis=b,
                           ell=cp.floor(cp.sqrt(cp.arange(1, modes + 1, dtype=cp.float64))))


def _options(**switches):
    values = {name: default for (name, default) in mpeps.OPTIONS.values()}
    values.pop("multi_perturb"); values.pop("degree")
    values.update(spdt=1.0, **switches)
    for s in ("pert_farms", "pert_deng", "pert_thom", "pert_cld3"):
        if values.get(s):
            for f in mpeps.eps.FACTORS[s]:
                values[f] = 1.0
    return mpeps.EpsOptions(True, values, 12)


@pytest.mark.gpu
def test_patterns_follow_wrfs_cadence_and_restore_word_for_word():
    cp = pytest.importorskip("cupy")
    r = _resource(cp)
    opts = _options(pert_thom=True, pert_farms=True)
    a = mpeps.MpasMultiPerturb(opts, None, dt=20.0, n_solve=600, resource=r)
    assert a.stepsp == 3
    assert set(a.names) == {"ALBEDO", "AOD", "ANGSTROM", "ASSYMFAC", "QVAPOR", "QCLOUD",
                            "QICE", "QSNOW", "NI"}
    moved = [a.update(step, (step - 1) * 20.0) for step in range(1, 7)]
    assert moved == [True, False, True, False, False, True]
    table = mpeps.eps.load_stochpert_table("")
    for name, v in a.values.items():
        v = cp.asnumpy(v)
        row = table[name]
        assert np.abs(v).max() > 0 and np.abs(v).max() <= np.float32(row.cutoff) * np.float32(row.std)
    b = mpeps.MpasMultiPerturb(opts, None, dt=20.0, n_solve=600, resource=r)
    b.restore_restart_state(a.restart_state())
    for step in (7, 8, 9):
        a.update(step, (step - 1) * 20.0)
        b.update(step, (step - 1) * 20.0)
    for name in a.values:
        assert np.array_equal(cp.asnumpy(a.values[name]).view(np.uint32),
                              cp.asnumpy(b.values[name]).view(np.uint32))


@pytest.mark.gpu
def test_the_seats_apply_wrfs_statements():
    cp = pytest.importorskip("cupy")
    from woof.core import wrfsolar_eps as eps
    r = _resource(cp)
    opts = _options(pert_thom=True, pert_farms=True, pert_deng=True, pert_cld3=True)
    p = mpeps.MpasMultiPerturb(opts, None, dt=20.0, n_solve=600, resource=r)
    p.update(1, 0.0)
    session = SimpleNamespace(wrfsolar_eps=p)
    rng = np.random.default_rng(4)
    nl, nc, ncols = 12, 600, 640                 # 40 halo columns beyond nCellsSolve
    q = lambda n: cp.asarray(rng.uniform(0, 1e-3, (nl, n)).astype(np.float32))
    # Thompson: the driver's fields, solve cells perturbed, halo cells untouched
    fields = {n: q(ncols) for n in ("qv", "qc", "qi", "qs", "ni")}
    before = {n: cp.asnumpy(a).copy() for n, a in fields.items()}
    add, remove = mpeps.thompson_pair(session)
    saved = add(fields)
    want = {n: cp.asarray(before[n][:, :nc].copy()) for n in fields}
    pats = {k: eps.Pattern(v.values.reshape(1, 1, -1)) for k, v in p.level_patterns().items()}
    eps.thompson_add(pats, opts.factors("pert_thom"), **want)
    for n in fields:
        got = cp.asnumpy(fields[n])
        assert np.array_equal(got[:, :nc].view(np.uint32), cp.asnumpy(want[n]).view(np.uint32))
        assert np.array_equal(got[:, nc:], before[n][:, nc:])
    remove(fields, saved)
    # FARMS: the copies
    copies = dict(albedo=cp.full(nc, 0.3, cp.float32), aod=cp.zeros(nc, cp.float32),
                  angexp=cp.zeros(nc, cp.float32), aerasy=cp.zeros(nc, cp.float32),
                  qv=q(nc), qc=q(nc), qs=q(nc))
    ref = {k: v.copy() for k, v in copies.items()}
    mpeps.farms_inputs(session, **copies)
    eps.farms_add(p.level_patterns(), opts.factors("pert_farms"), **ref)
    for k in copies:
        assert np.array_equal(cp.asnumpy(copies[k]).view(np.uint32), cp.asnumpy(ref[k]).view(np.uint32)), k
    assert not np.array_equal(cp.asnumpy(copies["albedo"]), np.full(nc, 0.3, np.float32))
    # Deng: qv, qc and w copies ([level, 1, cell]; w on nl+1 levels)
    d = dict(qv=q(nc).reshape(nl, 1, nc), qc=q(nc).reshape(nl, 1, nc),
             w=cp.asarray(rng.normal(0, 1, (nl + 1, 1, nc)).astype(np.float32)))
    dref = {k: v.copy() for k, v in d.items()}
    mpeps.deng_inputs(session, **d)
    eps.deng_add(p.level_patterns(), opts.factors("pert_deng"), th=cp.zeros_like(dref["qv"]), **dref)
    for k in d:
        assert np.array_equal(cp.asnumpy(d[k]).view(np.uint32), cp.asnumpy(dref[k]).view(np.uint32)), k
    # cloudiness: [cell, level] copies of t and qv
    prof = {"t": cp.asarray(rng.uniform(200, 300, (nc, nl)).astype(np.float32)),
            "qv": cp.ascontiguousarray(q(nc).T)}
    out = mpeps.cld3_perturb(session, prof)
    tt, qq = eps.cld3_inputs(prof["t"], prof["qv"], p.column_grid("TH", nl),
                             p.column_grid("QVAPOR", nl), 1.0, 1.0)
    assert np.array_equal(cp.asnumpy(out["t"]), cp.asnumpy(tt))
    assert np.array_equal(cp.asnumpy(out["qv"]), cp.asnumpy(qq))
