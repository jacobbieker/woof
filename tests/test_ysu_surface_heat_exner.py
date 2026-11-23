"""The opt-in YSU surface heat flux on theta (RunConfig.ysu_surface_heat_exner).

THE WRF QUIRK.  WRF 4.7.1's YSU (``phys/physics_mmm/bl_ysu.F90``) works in
potential temperature but takes the surface sensible heat flux as if it were
a theta flux: ``sflux = hfx/rho/cp`` (:613), ``hgamt = gamfac*hfx/cp``
(:687, :891) and the lowest row of the heat solve
``f1(1) = thx(1) - 300 + hfx/cont/del(1)*dt2`` (:1045, ``cont = cp/g``).  The
theta tendency is then turned into the temperature the driver hands back by
multiplying with pi (:1103), so the column receives ``HFX*pi(1)`` rather than
``HFX``: the surface heating is short by ``1 - pi``, about 0 at 1000 hPa, 4.5%
at 850 hPa and 14% at 600 hPa (the Tibetan and Andean plateaus, the Rockies).
The surface layer's own budget (and Noah's) books the full HFX, so the
energy is lost between the two schemes.

THE OPT-IN.  ``ysu_surface_heat_exner = true`` hands the unchanged YSU kernel
(either generation) the theta flux ``HFX/pi(1)`` (w'theta' = w'T'/pi at the
lowest level), so every use of HFX in the scheme sees a theta flux and the
column's temperature receives HFX.  No kernel source changes; off (the
default) the launch is the caller's hfx, WRF's word for word.

WHAT THIS FILE PINS.
* Configuration: off by default, YSU only, not with BEP/BEM.
* The default launch: no key is the historical launch, byte for byte, and
  the opt-in only replaces the hfx the kernel reads.
* The corrected path on the GPU, by the column heat budget: YSU's implicit
  diffusion, countergradient and entrainment terms are all in flux form with
  a closed top, so ``sum_k cp*dtheta_k*delp_k/g`` is exactly the surface
  theta flux.  WRF's arithmetic gives HFX there (a temperature flux booked as
  theta); the corrected arm gives HFX/pi(1), whose temperature at the lowest
  level, pi(1)*theta, is HFX.
* The packed ensemble route applies the same correction as launch_ysu.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu


def _base(**kw):
    from woof.config import RunConfig
    return RunConfig(nx=8, ny=8, nz=12, dx=3000.0, dy=3000.0, ztop=12000.0,
                     dt=10.0, run_seconds=60.0, **kw)


def test_the_key_follows_the_mode_and_is_read_by_ysu_only():
    from woof import correct_mode
    from woof.config import ConfigurationRefused, RunConfig, validate_run_config
    from woof.quirk_corrections import run_reads
    field = RunConfig.__dataclass_fields__["ysu_surface_heat_exner"]
    # correct-mode finding FQ2: on in a default process, off in exact mode
    assert field.default is correct_mode.CORRECT_KEYS_DEFAULT and field.kw_only
    assert correct_mode.resolve("exact").settings["ysu_surface_heat_exner"] is False
    assert run_reads(validate_run_config(_base(bl_pbl_physics=1, sf_sfclay_physics=1,
                                               ysu_surface_heat_exner=True)), "ysu_surface_heat_exner")
    assert not run_reads(validate_run_config(_base(bl_pbl_physics=5, sf_sfclay_physics=5,
                                                   ysu_surface_heat_exner=True)), "ysu_surface_heat_exner")
    assert run_reads(validate_run_config(_base(bl_pbl_physics=1, sf_sfclay_physics=1,
                                               ysu_version="wrf_461",
                                               ysu_surface_heat_exner=True)), "ysu_surface_heat_exner")
    # with BEP/BEM YSU takes the urban heat sources: not read, and listed as uncorrected
    bep = validate_run_config(_base(bl_pbl_physics=1, sf_sfclay_physics=1,
                                    sf_surface_physics=2, sf_urban_physics=2,
                                    ysu_surface_heat_exner=True))
    assert not run_reads(bep, "ysu_surface_heat_exner")
    assert "FQ2" in correct_mode.uncorrected(bep)
    with pytest.raises(ConfigurationRefused, match="true or false"):
        validate_run_config(_base(bl_pbl_physics=1, sf_sfclay_physics=1,
                                  ysu_surface_heat_exner=1))


def _plateau_columns(nz=40, ny=1, nx=4):
    """Convective columns on a 700 hPa surface (pi(1) ~ 0.90)."""
    ztop = 9000.0
    z_ifc = np.linspace(0.0, ztop, nz + 1)
    z = 0.5 * (z_ifc[:-1] + z_ifc[1:])
    p_ifc = 70000.0 * np.exp(-z_ifc / 8000.0)
    p = 0.5 * (p_ifc[:-1] + p_ifc[1:])
    exner = (p / 100000.0) ** (287.0 / 1004.5)
    shp = (nz, ny, nx)
    theta = np.where(z < 1200.0, 312.0, 312.0 + 0.004 * (z - 1200.0))
    f3 = lambda a: np.ascontiguousarray(np.broadcast_to(np.reshape(a, (-1, 1, 1)), shp),
                                        np.float32)
    f2 = lambda value: np.full((ny, nx), value, np.float32)
    u = 4.0 + 0.001 * z
    hfx = np.array([[50.0, 150.0, 300.0, 450.0]], np.float32)
    return dict(
        u=f3(u), v=f3(np.zeros(nz)), theta=f3(theta),
        qv=f3(0.004 * np.exp(-z / 2500.0)), qc=f3(np.zeros(nz)),
        qi=f3(np.zeros(nz)), p=f3(p),
        p_interface=np.ascontiguousarray(
            np.broadcast_to(p_ifc[:, None, None], (nz + 1, ny, nx)), np.float32),
        exner=f3(exner), dz=f3(np.diff(z_ifc)),
        psfc=f2(p_ifc[0]), znt=f2(0.10), ust=f2(0.45), hfx=hfx,
        qfx=f2(0.0), wspd=f2(float(u[0])), br=f2(-0.3),
        psim=f2(5.0), psih=f2(7.0), xland=f2(1.0),
        u10=f2(float(u[0])), v10=f2(0.0)), exner[0]


def _run(arrays, **kw):
    import cupy as cp
    from woof.core.ysu import launch_ysu
    dev = {name: cp.asarray(value) for name, value in arrays.items()}
    out = launch_ysu(dev["u"], dev["v"], dev["theta"], dev["qv"], dev["qc"],
                     dev["qi"], dev["p"], dev["p_interface"], dev["exner"],
                     dev["dz"], psfc=dev["psfc"], znt=dev["znt"], ust=dev["ust"],
                     hfx=dev["hfx"], qfx=dev["qfx"], wspd=dev["wspd"],
                     br=dev["br"], psim=dev["psim"], psih=dev["psih"],
                     xland=dev["xland"], u10=dev["u10"], v10=dev["v10"],
                     dt=60.0, ysu_topdown_pblmix=0, **kw)
    return {name: cp.asnumpy(value) for name, value in out.items()}


@pytest.mark.gpu
@requires_gpu
def test_the_opt_in_changes_only_the_surface_heat_flux_input():
    """Off, the launch is the caller's hfx (the WRF-oracle YSU tests pin
    that launch word for word); on, it is exactly the default launch fed
    the theta flux hfx/pi(1), so nothing else in the program moves."""
    arrays, _ = _plateau_columns()
    default = _run(arrays)
    named_off = _run(arrays, surface_heat_exner=False)
    corrected = _run(arrays, surface_heat_exner=True)
    theta_flux = dict(arrays, hfx=np.ascontiguousarray(
        (arrays["hfx"] / arrays["exner"][0]).astype(np.float32)))
    fed = _run(theta_flux)
    for name, value in default.items():
        assert value.tobytes() == named_off[name].tobytes(), name
        assert corrected[name].tobytes() == fed[name].tobytes(), name
    assert default["dtheta"].tobytes() != corrected["dtheta"].tobytes()


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("version", ("wrf_480", "wrf_461"))
def test_the_corrected_column_receives_the_surface_heat_flux(version):
    arrays, pi1 = _plateau_columns()
    assert 0.88 < pi1 < 0.92
    default = _run(arrays, ysu_version=version)
    corrected = _run(arrays, ysu_version=version, surface_heat_exner=True)
    g, cp_air = 9.81, 1004.5
    delp = (arrays["p_interface"][:-1] - arrays["p_interface"][1:]).astype(np.float64)
    hfx = arrays["hfx"].astype(np.float64)

    def theta_flux(out):
        return np.sum(cp_air * out["dtheta"].astype(np.float64) * delp / g, axis=0)

    # WRF's arithmetic books HFX as a theta flux; in temperature the lowest
    # level gets pi(1)*HFX: the quirk the opt-in corrects.
    np.testing.assert_allclose(theta_flux(default), hfx, rtol=2e-3)
    # The corrected arm's theta flux is HFX/pi(1): HFX in temperature.
    np.testing.assert_allclose(theta_flux(corrected) * pi1, hfx, rtol=2e-3)
    # The PBL is at least as deep and warms more (more buoyancy flux).
    assert np.all(corrected["hpbl"] >= default["hpbl"] - 1e-3)
    assert np.all(theta_flux(corrected) > theta_flux(default))
    assert np.all(np.isfinite(corrected["dtheta"]))


@pytest.mark.gpu
@requires_gpu
def test_the_packed_route_applies_the_same_correction():
    import cupy as cp
    from woof.ensemble.batch_physics import prepare_ysu_column_batch
    arrays, _ = _plateau_columns()
    nz = arrays["theta"].shape[0]
    inputs = {name: cp.asarray(value) for name, value in arrays.items()}
    inputs["rthraten"] = cp.zeros((nz, 1, 4), cp.float32)
    want = _run(arrays, surface_heat_exner=True)
    hfx_before = cp.asnumpy(inputs["hfx"]).copy()
    batch = prepare_ysu_column_batch(inputs, members=1, ny=1, nx=4, dt=60.0,
                                     topdown=0, available_bytes=64 * 1024 * 1024,
                                     surface_heat_exner=True)
    got = batch()
    # The caller's hfx is never written.
    assert cp.asnumpy(inputs["hfx"]).tobytes() == hfx_before.tobytes()
    for name, value in want.items():
        assert cp.asnumpy(got[name]).tobytes() == value.tobytes(), name
