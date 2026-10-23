"""The fire-atmosphere driver, host and device, against WRF v4.7.1: max ULP 0.

The fixture is WRF's whole fire driver -- ``fire_driver_em_init`` and
``fire_driver_em_step`` (``phys/module_fr_fire_driver_wrf.F``) with
everything they call, byte-unmodified and pinned by sha256 -- compiled at
-O0 with REAL(4) by ``tools/fire_coupling_wrf471_oracle/build.sh`` and run
on a stub ``TYPE(domain)`` by its ``run_fire_coupling.F90``
(PROVENANCE.md beside the fixtures).  The cases reach every arm the port
carries:

* ``ideal`` -- ideal coordinates, uniform fuel (fire_fuel_read = 0), a
  walked ignition line and a later point, sheared wind turning with time,
  sloping terrain and varying roughness;
* ``real`` -- longitude/latitude ignitions (unit_fxlong from cen_lat), an
  ignition counted out by its zero radius, a fuel map with no-fuel and
  timber cells (fire_fuel_read = -1), fire_atm_feedback 0.7,
  fire_wind_height 10 m, other extinction depths and crown height, a 5 x 4
  refinement;
* ``edge`` -- a model top below fire_wind_height (the top layer's wind),
  roughness above it (zero wind), fuel by altitude (fire_fuel_read = 1),
  fire_advection = 0, no reinitialization;
* ``tendency`` -- fire_tendency alone, with canopy fluxes on both sides of
  the crown height.

:class:`woof.fire.coupling.FireRuntime` replays each case step by step on
the host (NumPy) and on the device, and every dumped field at every dumped
step is compared by raw float32 word with WRF's, and the device with the
host.  Nothing is held out.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "data" / "oracles" / "fire_coupling"
HARNESS = ROOT / "tools" / "fire_coupling_wrf471_oracle"
CASES = ("ideal", "real", "edge")
F = np.float32

#: The per-step outputs: fixture name -> state attribute (fire mesh unless
#: noted) or the runtime's UAH/VAH planes.
STEP_FIELDS = {
    "uf": "fire_uf", "vf": "fire_vf", "lfn": "fire_lfn", "tign": "fire_tign",
    "ros": "fire_ros", "fuel_frac": "fire_fuel_frac",
    "fire_area": "fire_area_frac", "burnt_area_dt": "fire_burnt",
    "fgrnhfx": "fire_fgrnhfx", "fgrnqfx": "fire_fgrnqfx",
    "grnhfx": "fire_grnhfx", "grnqfx": "fire_grnqfx",
    "canhfx": "fire_canhfx", "canqfx": "fire_canqfx",
    "avg_fuel_frac": "fire_avg_fuel_frac", "rthfrten": "fire_rthfrten",
    "rqvfrten": "fire_rqvfrten", "uah": "uah", "vah": "vah",
}
INIT_FIELDS = ("fuel_time", "fgip", "r_0", "bbb", "phiwc", "betafl",
               "fmc_g")
#: Outputs that cannot be bit-exact against WRF, with the reason in
#: PROVENANCE.md.  There are none.
HELD_OUT: frozenset = frozenset()


def _case(name):
    from woof.verify.urban_oracle import load_case
    from woof.verify.wrf471_fixtures import require_fixture_dir

    return load_case(require_fixture_dir(FIXTURES, "fire coupling") / name)


def _scalar(case, key):
    return np.asarray(case[key]).reshape(()).item()


def _t2(a):
    """Fortran (i, j) -> woof (j, i)."""
    return np.ascontiguousarray(np.asarray(a, dtype=F).T)


def _t3(a):
    """Fortran (i, k, j) -> woof (k, j, i)."""
    return np.ascontiguousarray(np.transpose(np.asarray(a, dtype=F),
                                             (1, 2, 0)))


def _words(a):
    return np.ascontiguousarray(np.asarray(a, dtype=F)).view(np.uint32)


def _cfg(case):
    ints = [int(v) for v in np.asarray(case["cfg_ints"]).ravel()]
    reals = [float(v) for v in np.asarray(case["cfg_reals"]).ravel()]
    (num, fuel_read, fuel_cat, advection, reinit, reinit_iter, band,
     visc_ngp) = ints
    (fwh, ext_grnd, ext_crwn, crwn_hgt, feedback, visc, visc_bg, visc_band,
     ext_up, slope, cen_lat) = reals
    ig = np.asarray(case["cfg_ignitions"], dtype=F)       # (12, n)
    records = tuple(tuple(float(v) for v in ig[:, k])
                    for k in range(min(num, 5, ig.shape[1])))
    return SimpleNamespace(
        ifire=2, sr_x=_scalar(case, "ir"), sr_y=_scalar(case, "jr"),
        nx=_scalar(case, "nx"), ny=_scalar(case, "ny"),
        nz=_scalar(case, "nz"), dx=_scalar(case, "dx"),
        dy=_scalar(case, "dx"), dt=_scalar(case, "dt"),
        fire_fuel_read=fuel_read, fire_fuel_cat=fuel_cat,
        fire_fuel_table="namelist", fire_wind_height=fwh,
        fire_ext_grnd=ext_grnd, fire_ext_crwn=ext_crwn,
        fire_crwn_hgt=crwn_hgt, fire_atm_feedback=feedback,
        fire_ignitions=records, fire_advection=advection,
        fire_lsm_reinit=bool(reinit), fire_lsm_reinit_iter=reinit_iter,
        fire_lsm_band_ngp=band, fire_viscosity=visc,
        fire_viscosity_bg=visc_bg, fire_viscosity_band=visc_band,
        fire_viscosity_ngp=visc_ngp, fire_lfn_ext_up=ext_up,
        fire_slope_factor=slope, fire_smoke_factor="fire_tracer_smoke",
        fire_tracer_smoke=0.02, sfc_lu_index=28), cen_lat


class _State(SimpleNamespace):
    def total_mu(self):
        return self.mut


def _replay(name, xp, wind):
    """``{(step, field): host array}`` from FireRuntime, and the oracle."""
    from woof.fire import coupling
    from woof.static import fuel as fuel_statics

    case = _case(name)
    cfg, cen_lat = _cfg(case)
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    ir, jr = cfg.sr_x, cfg.sr_y
    zsf = _t2(case["in_zsf"])
    if cfg.fire_fuel_read == 0:
        nfuel = np.full(zsf.shape, F(cfg.fire_fuel_cat))
    elif cfg.fire_fuel_read == 1:
        nfuel = coupling.nfuel_by_altitude(zsf)
    else:
        nfuel = _t2(case["in_nfuel_cat"])
    plan = coupling.convert_ignitions(cfg.fire_ignitions, cen_lat=cen_lat)
    coords = None
    if plan.longlat:
        coords = (fuel_statics.refine_zsf(_t2(case["in_xlong"]), ir, jr),
                  fuel_statics.refine_zsf(_t2(case["in_xlat"]), ir, jr))
    runtime = coupling.FireRuntime(
        cfg, nx=nx, ny=ny, nz=nz, dx=cfg.dx, dy=cfg.dy, nfuel_cat=nfuel,
        dzdxf=_t2(case["in_dzdxf"]), dzdyf=_t2(case["in_dzdyf"]),
        coords=coords, cen_lat=cen_lat, xp=xp)

    def dev(a):
        return xp.asarray(np.ascontiguousarray(a, dtype=F))

    state = _State(
        phb=dev(_t3(case["in_phb"])), php=dev(_t3(case["in_ph"])),
        ht=dev(_t2(case["in_ht"])), mut=dev(_t2(case["in_mut"])),
        c1h=dev(np.asarray(case["in_c1h"])), c2h=dev(np.asarray(case["in_c2h"])),
        has_msf=False)
    zeros = lambda *shape: xp.zeros(shape, dtype=F)  # noqa: E731
    coupling.allocate_state(state, cfg, zeros)
    from woof.fire import levelset as ls
    init = ls.initial_state(runtime.mesh, time=0.0, xp=xp)
    for attr, value in (("fire_lfn", init.lfn_m), ("fire_tign", init.tign_m),
                        ("fire_fuel_frac", init.fuel_frac)):
        getattr(state, attr)[...] = value
    rho = dev(_t3(case["in_rho"]))
    z0 = dev(_t2(case["in_z0"]))
    host = (lambda a: a) if xp is np else (lambda a: a.get())
    out, want = {}, {}
    p = runtime.params
    for key in INIT_FIELDS:
        out[(0, key)] = host(getattr(p, key)).copy()
        want[(0, key)] = _t2(case["init_" + key])
    out[(0, "nfuel_cat")] = nfuel
    want[(0, "nfuel_cat")] = _t2(case["init_nfuel_cat"])
    out[(0, "lfn")] = host(state.fire_lfn)[1:-1, 1:-1].copy()
    want[(0, "lfn")] = _t2(case["init_lfn"])
    out[(0, "tign")] = host(state.fire_tign)[1:-1, 1:-1].copy()
    want[(0, "tign")] = _t2(case["init_tign"])
    if coords is not None:
        out[(0, "fxlong")], out[(0, "fxlat")] = coords
        want[(0, "fxlong")] = _t2(case["init_fxlong"])
        want[(0, "fxlat")] = _t2(case["init_fxlat"])
    dumps = [int(s) for s in np.asarray(case["dumps"]).ravel()]
    nsteps = _scalar(case, "nsteps")
    dt = F(_scalar(case, "dt"))
    # u and v at every step follow run_fire_coupling.F90's set_wind; the
    # fixtures carry them at the dumps, which is where they are checked.
    for istep in range(1, nsteps + 1):
        u, v = wind(istep, nx, ny, nz)
        state.u, state.v = dev(u), dev(v)
        runtime.step(state, istep, dt, z0=z0, rho=rho)
        if istep not in dumps:
            continue
        pre = f"s{istep}_"
        np.testing.assert_array_equal(_words(u), _words(_t3(case[pre + "in_u"])))
        np.testing.assert_array_equal(_words(v), _words(_t3(case[pre + "in_v"])))
        for key, attr in STEP_FIELDS.items():
            if key == "uah":
                value = host(runtime.ua)[1:, :]
            elif key == "vah":
                value = host(runtime.va)[:, 1:]
            else:
                value = host(getattr(state, attr))
                if attr in ("fire_lfn", "fire_tign"):
                    value = value[1:-1, 1:-1]
            raw = np.asarray(case[pre + key], dtype=F)
            want[(istep, key)] = _t3(raw) if raw.ndim == 3 else _t2(raw)
            out[(istep, key)] = np.array(value, dtype=F)
    return out, want


def _wind_for(name):
    """run_fire_coupling.F90's set_wind for one case, REAL(4), woof layout.

    u0 and v0 are read from the harness's run_case calls, so the replay
    and the oracle cannot drift apart silently (the fixtures also carry the
    winds at every dump, which the replay checks).
    """
    from woof.fire.wrf_sfire import r4

    text = (HARNESS / "run_fire_coupling.F90").read_text()
    calls = re.findall(r"call run_case\(grid, cf, \d+, \(/[^/]*/\), "
                       r"([-\d.]+), ([-\d.]+)\)", text)
    u0, v0 = (r4(c) for c in dict(zip(CASES, calls))[name])

    def wind(istep, nx, ny, nz):
        s = r4("0.02") * F(istep)
        k = np.arange(1, nz + 1, dtype=F)[:, None, None]
        j = np.arange(1, ny + 1, dtype=F)[None, :, None]
        i = np.arange(1, nx + 1, dtype=F)[None, None, :]
        u = u0 + r4("0.35") * k + r4("0.04") * j - s
        v = v0 - r4("0.15") * k + r4("0.03") * i + s
        return (np.broadcast_to(u, (nz, ny, nx + 1)).astype(F),
                np.broadcast_to(v, (nz, ny + 1, nx)).astype(F))
    return wind


def _run(name, xp):
    return _replay(name, xp, _wind_for(name))


def _compare(out, want, label):
    bad = []
    for key, expected in want.items():
        if key[1] in HELD_OUT:
            continue
        got = out[key]
        assert got.shape == expected.shape, (label, key, got.shape,
                                             expected.shape)
        diff = int(np.count_nonzero(_words(got) != _words(expected)))
        if diff:
            bad.append(f"{key}: {diff} words differ")
    assert not bad, f"{label}: " + "; ".join(bad)


@pytest.mark.parametrize("name", CASES)
def test_host_reference_matches_wrf(name):
    out, want = _run(name, np)
    _compare(out, want, f"host {name}")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", CASES)
def test_device_matches_wrf_and_host(name):
    import cupy as cp

    out, want = _run(name, cp)
    _compare(out, want, f"device {name}")
    host, _ = _run(name, np)
    _compare(out, host, f"device vs host {name}")


def _tendency_inputs():
    case = _case("tendency")
    ext_grnd, ext_crwn, crwn_hgt = (float(v) for v in
                                    np.asarray(case["cfg_reals"]).ravel())
    phi = _t3(case["in_phb"]) + _t3(case["in_ph"])
    np.testing.assert_array_equal(
        _words(phi / F(9.81)), _words(_t3(case["in_z_at_w"])))
    args = (_t2(case["in_grnhfx"]), _t2(case["in_grnqfx"]),
            _t2(case["in_canhfx"]), _t2(case["in_canqfx"]),
            _t2(case["in_ht"]), phi, _t2(case["in_mut"]),
            np.asarray(case["in_c1h"], dtype=F),
            np.asarray(case["in_c2h"], dtype=F), _t3(case["in_rho"]))
    return case, args, dict(alfg=ext_grnd, alfc=ext_crwn, z1can=crwn_hgt)


def test_fire_tendency_host_matches_wrf():
    from woof.fire.coupling import fire_tendency_np

    case, args, kw = _tendency_inputs()
    rth, rqv, slow = fire_tendency_np(*args, **kw)
    np.testing.assert_array_equal(_words(rth), _words(_t3(case["rthfrten"])))
    np.testing.assert_array_equal(_words(rqv), _words(_t3(case["rqvfrten"])))
    np.testing.assert_array_equal(_words(slow), _words(rth))
    assert np.count_nonzero(rth) > 0 and np.count_nonzero(rqv) > 0


@pytest.mark.gpu
@requires_gpu
def test_fire_tendency_device_matches_wrf():
    import cupy as cp

    from woof.fire.coupling import fire_tendency_gpu, fire_tendency_np

    case, args, kw = _tendency_inputs()
    nz, ny, nx = args[-1].shape
    rth, rqv, slow = (cp.full((nz, ny, nx), -7.0, dtype=F) for _ in range(3))
    msft = np.linspace(0.9, 1.1, ny * nx, dtype=F).reshape(ny, nx)
    fire_tendency_gpu(*(cp.asarray(a) for a in args), **kw,
                      msft=cp.asarray(msft), rth=rth, rqv=rqv, rth_slow=slow)
    np.testing.assert_array_equal(_words(rth.get()),
                                  _words(_t3(case["rthfrten"])))
    np.testing.assert_array_equal(_words(rqv.get()),
                                  _words(_t3(case["rqvfrten"])))
    _, _, host_slow = fire_tendency_np(*args, **kw, msft=msft)
    np.testing.assert_array_equal(_words(slow.get()), _words(host_slow))


def test_fixture_manifest_hashes():
    """Every fixture file is the one the oracle wrote (FIXTURES.sha256)."""
    lines = (FIXTURES / "FIXTURES.sha256").read_text().splitlines()
    assert lines
    for line in lines:
        digest, rel = line.split(maxsplit=1)
        data = (FIXTURES / rel.strip().lstrip("./")).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, rel


def test_harness_hashes():
    """The harness that wrote the fixtures is the one in the tree."""
    for line in (FIXTURES / "HARNESS.sha256").read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        data = (HARNESS / rel.strip()).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, rel
