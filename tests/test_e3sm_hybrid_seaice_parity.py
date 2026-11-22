"""The MPAS v8.4.2 sea ice under E3SM v3.2.0's coupler, word for word.

"E3SM v3.2.0 coupler with MPAS v8.4.2 components, not an E3SM configuration"
(:mod:`woof.hex.coupled.hybrid_seaice`). The reference is
``tools/e3sm_coupler_oracle/run_hybrid_seaice.F90``: E3SM v3.2.0's
``ice_init_mct`` and ``ice_run_mct`` sequence around MPAS v8.4.2's time step
with the column package and the velocity solver (every module verbatim,
RKIND 8, the stand-alone build's column constants) and E3SM's
``ice_import_mct`` and ``ice_export_mct`` (cut verbatim) compiled against
v8.4.2's modules. The outputs are committed
(tests/data/oracles/e3sm_hybrid_seaice, PROVENANCE.md).

THE BREAKAGES THESE PREVENT
---------------------------
1. A hybrid that agrees with the oracle only to a tolerance, or only under
   strict numerics: i2x after the init and every window, and every array,
   halo cells included, in fast and strict child processes too.
2. A case generator whose bytes drift. The input digests are pinned.
3. A run loop that drifts from E3SM's: the fluxes re-initialised per step
   instead of per window (v8.4.2's finalize), not re-initialised at all,
   no post-coupling rotation, no first-step shortwave init, or E3SM's
   build's constants in the cap where v8.4.2's column package's belong. Each
   mutation must mismatch.
4. A divergence half-ported: each one E3SM's loop has and v8.4.2's lacks is
   listed, and each refused one is refused by name.
5. The default changed: nothing on the default path builds the hybrid, and
   the cap's constants default to E3SM's build's.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pytest

from _numerics_child import MODES, run as run_child
from conftest import requires_gpu
from woof.hex.coupled import hybrid_seaice as hs
from woof.hex.coupled import ice_coupling_e3sm as icc
from woof.hex.errors import ConfigurationRefusal
from woof.verify import e3sm_hybrid_seaice_oracle as oracle

ROOT = Path(__file__).resolve().parents[1]


def _case(name):
    return next(c for c in oracle.cases() if c["name"] == name)


@contextlib.contextmanager
def _glibc239_opt_in():
    """The dEdd orbit's opt-in glibc 2.39 binary64 transcriptions (LGPL, source checkout only)."""
    from woof.core import glibc239_gpu
    from woof.hex.mpas_seaice import column_radiation_v842 as crad
    from woof.hex.mpas_seaice import column_therm_v842 as ct
    if not glibc239_gpu.header_path().is_file():
        pytest.skip("the opt-in glibc 2.39 transcription is only in a source checkout")
    before = os.environ.get("WOOF_GLIBC239_LIBM64")
    os.environ["WOOF_GLIBC239_LIBM64"] = "1"
    caches = [ct._module_on, crad._module_on]
    for cache in caches:
        cache.cache_clear()
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("WOOF_GLIBC239_LIBM64", None)
        else:
            os.environ["WOOF_GLIBC239_LIBM64"] = before
        for cache in caches:
            cache.cache_clear()


@pytest.fixture
def glibc239():
    with _glibc239_opt_in():
        yield


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oracle.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {case["name"]: oracle.input_digest(case) for case in oracle.cases()}


def test_fixture_covers_the_branches():
    ref = oracle.load_fixture()
    cases = {c["name"]: c for c in oracle.cases()}
    assert set(ref) == set(cases)
    col = {n: i for i, n in enumerate(oracle.I2X_FIELDS)}
    for name, case in cases.items():
        h = case["hybrid"]
        windows = [ref[name][f"i2x/{w}"] for w in range(h["windows"] + 1)]
        # every window moves the ice
        assert all((a[:, col["Si_t"]] != b[:, col["Si_t"]]).any() for a, b in zip(windows, windows[1:]))
        # no icebergs: the two berg fields sent as 0
        assert all((w[:, col["PFioi_bergw"]] == 0).all() for w in windows)
        assert ref[name]["oracle/trig"].shape == (case["dims"]["nCells"], 8)
        # frazilMassAdjust set by each window's import (free surface)
        assert (ref[name]["ocean_coupling/frazilMassAdjust"] != 0).any()
    steps = {c["hybrid"]["steps_per_window"] for c in cases.values()}
    assert steps == {1, 2}
    thermo = {c["config"]["config_thermodynamics_type"] for c in cases.values()}
    assert thermo == {"BL99", "mushy"}
    assert {c["config"]["config_rotate_cartesian_grid"] for c in cases.values()} == {True, False}
    assert {c["config"]["config_shortwave_type"] for c in cases.values()} == {"dEdd", "ccsm3"}


@requires_gpu
def test_gpu_matches_the_hybrid_oracle_word_for_word():
    with _glibc239_opt_in():  # not a fixture: the numerics children call the test bare
        report = oracle.measure()
    bad = {case: {k: v for k, v in r.items() if v} for case, r in report.items()}
    assert not any(bad.values()), bad
    assert all(len(r) > 300 for r in report.values())


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_hybrid_oracle_word_for_word"], numerics, exact)


def _mismatches(name, *, mutate=None):
    case = _case(name)
    want = oracle.load_fixture()[name]
    got = oracle.run_gpu(case, trig=want["oracle/trig"], mutate=mutate)
    return sum(oracle.word_mismatches(got[k], want[k]) for k in want if k in got)


@requires_gpu
def test_a_per_step_finalize_is_caught(glibc239, monkeypatch):
    # v8.4.2's seaice_timestep_finalize after each step: the fluxes re-initialised per step
    from woof.hex.mpas_seaice import timestep_v842 as ts
    advance = ts.advance
    monkeypatch.setattr(ts, "advance", lambda **kw: advance(**{**kw, "finalize_last": True}))
    assert _mismatches("hybrid_bl99_two_steps_gregorian") > 0


@requires_gpu
def test_no_flux_reinitialisation_is_caught(glibc239, monkeypatch):
    from woof.hex.mpas_seaice import column_step_v842 as cstep
    monkeypatch.setattr(cstep, "reinitialize_fluxes", lambda pools, configs: None)
    assert _mismatches("hybrid_bl99_dedd") > 0


@requires_gpu
@pytest.mark.parametrize("routine", ["post_atmospheric_coupling", "post_oceanic_coupling"])
def test_a_missing_post_coupling_is_caught(glibc239, monkeypatch, routine):
    from woof.hex.mpas_seaice import post_coupling_v842 as post
    monkeypatch.setattr(post, routine, lambda **kw: None)
    assert _mismatches("hybrid_bl99_dedd") > 0


@requires_gpu
def test_a_missing_first_step_shortwave_init_is_caught(glibc239):
    def skip(ice):
        ice.first = False
    assert _mismatches("hybrid_bl99_dedd", mutate=skip) > 0


@requires_gpu
def test_e3sm_build_constants_in_the_cap_are_caught(glibc239):
    # shr_const's cp_ice, cp_ocn and latent heat (E3SM's CCSMCOUPLED build) in the frazil mass
    def e3sm(ice):
        ice.constants = icc.e3sm_constants()
    assert _mismatches("hybrid_mushy_level_ponds", mutate=e3sm) > 0


def test_the_cap_constants_default_to_e3sm_builds():
    assert icc.e3sm_constants().cp_ice == icc.CP_ICE == 2.11727e3
    assert hs.V842_CONSTANTS.cp_ice == 2106.0 and hs.V842_CONSTANTS.lfresh == 2.835e6 - 2.501e6
    assert hs.V842_CONSTANTS.latice == hs.V842_CONSTANTS.lfresh


def test_every_divergence_is_listed_with_a_disposition():
    names = [d.name for d in hs.DIVERGENCES]
    assert len(names) == len(set(names))
    assert {d.disposition for d in hs.DIVERGENCES} == {"ported", "refused", "omitted"}
    for d in hs.DIVERGENCES:
        assert d.where and d.reason
        if d.disposition == "refused":
            assert d.knob, d.name
    assert "not an E3SM configuration" in hs.HYBRID_NAME
    assert hs.divergence("the column built with CCSMCOUPLED (v8.4.2's ESM build)").disposition == "omitted"
    from woof.hex.coupled import registry
    name, _, components = registry.hybrids()["e3sm-v3.2.0-coupler+mpas-v8.4.2"]
    assert name == hs.HYBRID_NAME and set(components) == {"ocean", "seaice"}
    assert registry.hybrid_modules("e3sm-v3.2.0-coupler+mpas-v8.4.2")["seaice"] == hs.__name__


_BASE = {"config_dt": 1800.0, "config_use_column_package": True}


@pytest.mark.parametrize("change, kwargs, match", [
    ({"config_use_column_package": False}, {}, "column physics"),
    ({"config_column_physics_type": "icepack"}, {}, "Icepack"),
    ({"config_use_forcing": True}, {}, "forcing"),
    ({"config_use_prescribed_ice": True}, {}, "prescribed"),
    ({"config_use_data_icebergs": True}, {}, "berg_fluxes"),
    ({"config_use_floe_size_distribution": True}, {}, "floe_size_distribution"),
    ({"config_use_column_waves": True}, {}, "wave_coupling"),
    ({}, {"wav_ice_coup": "twoway"}, "wave_coupling"),
    ({"config_scale_dib_by_removed_ice_runoff": True}, {}, "dib"),
    ({"config_couple_biogeochemistry_fields": True}, {}, "DOCPoolFractions"),
    ({"config_use_zaerosols": True}, {}, "DryDustFlux"),
    ({"config_use_aerosols": True}, {}, "aerosols"),
    ({"config_use_modal_aerosols": True}, {}, "modal"),
    ({"config_use_ocean_mixed_layer": True}, {}, "mixed layer"),
    ({"config_ocean_surface_type": "mixed_layer"}, {}, "slab"),
    ({"config_do_restart": True}, {}, "initial"),
    ({}, {"ice_cpl_dt": 2700}, "multiple"),
    ({}, {"ice_cpl_dt": 900}, "multiple"),
])
def test_the_divergences_are_refused_by_name(change, kwargs, match):
    with pytest.raises(ConfigurationRefusal, match=match) as info:
        hs.check({**_BASE, **change}, **{"ice_cpl_dt": 1800, **kwargs})
    assert hs.HYBRID_NAME in str(info.value)


def test_the_accepted_configurations_pass():
    hs.check(_BASE, ice_cpl_dt=1800)
    hs.check(_BASE, ice_cpl_dt=3600)
    for case in oracle.cases():
        hs.check(case["config"], ice_cpl_dt=case["hybrid"]["ice_cpl_dt"])


def test_the_default_path_does_not_build_the_hybrid():
    hits = [p for p in (ROOT / "woof").rglob("*.py")
            if "HybridSeaIce(" in p.read_text(encoding="utf-8") and p.name != "hybrid_seaice.py"
            and "verify" not in p.parts]
    assert hits == []
    import inspect

    from woof.hex.mpas_seaice import timestep_v842 as ts
    sig = inspect.signature(ts.advance)
    assert sig.parameters["finalize_last"].default is True and sig.parameters["after_clock"].default is None



def test_the_basis_configs_come_from_the_api_or_are_refused_by_name():
    from woof.hex.errors import ConfigurationRefusal
    sphere = hs.basis_configs({}, 6.371229e6)
    assert sphere["mesh:on_a_sphere"] is True and sphere["mesh:sphere_radius"] == 6.371229e6
    plane = hs.basis_configs({}, 0.0)
    assert plane["mesh:on_a_sphere"] is False and plane["mesh:sphere_radius"] == 0.0
    # a configured value wins over the argument
    assert hs.basis_configs({"mesh:sphere_radius": 2.0}, 5.0)["mesh:sphere_radius"] == 2.0
    with pytest.raises(ConfigurationRefusal, match="sphere_radius"):
        hs.basis_configs({}, None)
    with pytest.raises(ConfigurationRefusal, match="sphere_radius"):
        hs.basis_configs({"mesh:on_a_sphere": True}, 0.0)
    with pytest.raises(ConfigurationRefusal, match="is_periodic"):
        hs.basis_configs({}, 0.0, ("periodic_local_coords",))
    with pytest.raises(ConfigurationRefusal, match="y_period"):
        hs.basis_configs({"is_periodic": True, "x_period": 4.0}, 0.0, ("periodic_local_coords",))
    periodic = hs.basis_configs({"is_periodic": True, "x_period": 4.0, "y_period": 0.0}, 0.0,
                                ("periodic_local_coords",))
    assert (periodic["mesh:is_periodic"], periodic["mesh:x_period"], periodic["mesh:y_period"]) == (True, 4.0, 0.0)
    import inspect
    assert inspect.signature(hs.HybridSeaIce).parameters["basis"].default == "compute"
