"""The MPAS v8.4.2 ocean under E3SM v3.2.0's coupler, word for word.

"E3SM v3.2.0 coupler with MPAS v8.4.2 components, not an E3SM configuration"
(:mod:`woof.hex.coupled.hybrid_ocean`). The reference is
``tools/e3sm_coupler_oracle/run_hybrid_ocean.F90``: E3SM v3.2.0's
``ocn_init_mct`` and ``ocn_run_mct`` sequence around MPAS v8.4.2's forward
mode (every module verbatim, RKIND 8) and E3SM's ``ocn_import_mct`` and
``ocn_export_mct`` (cut verbatim) compiled against v8.4.2's modules, with
E3SM's build's constants or v8.4.2's own. The outputs are committed
(tests/data/oracles/e3sm_hybrid_ocean, PROVENANCE.md).

THE BREAKAGES THESE PREVENT
---------------------------
1. A hybrid that agrees with the oracle only to a tolerance, or only under
   strict numerics: o2x after the init and every window, and every array,
   halo cells included, in fast and strict child processes too.
2. A case generator whose bytes drift. The input digests are pinned.
3. A run loop that drifts from E3SM's: the means not reset per window, the
   clock not moved to its first alarm, T0_Kelvin or gravity from the wrong
   build. Each mutation must mismatch.
4. A divergence half-ported: each one E3SM's loop has and v8.4.2's lacks is
   listed, and each refused one is refused by name.
5. The default changed: nothing on the default path builds the hybrid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _numerics_child import MODES, run as run_child
from conftest import requires_gpu
from woof.hex.coupled import hybrid_ocean as ho
from woof.hex.errors import ConfigurationRefusal
from woof.verify import e3sm_hybrid_ocean_oracle as oracle

ROOT = Path(__file__).resolve().parents[1]


def _case(name):
    return next(c for c in oracle.cases() if c["name"] == name)


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oracle.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {case["name"]: oracle.input_digest(case) for case in oracle.cases()}


def test_fixture_covers_the_branches():
    ref = oracle.load_fixture()
    cases = {c["name"]: c for c in oracle.cases()}
    assert set(ref) == set(cases)
    col = {n: i for i, n in enumerate(oracle.O2X_FIELDS)}
    for name, case in cases.items():
        windows = [ref[name][f"o2x/{w}"] for w in range(case["hybrid"]["windows"] + 1)]
        # every window moves the ocean
        assert all((a[:, col["So_t"]] != b[:, col["So_t"]]).any() for a, b in zip(windows, windows[1:]))
        # Faoo_h2otemp: v8.4.2 has no source, sent as 0
        assert all((w[:, col["Faoo_h2otemp"]] == 0).all() for w in windows)
        # the sea-ice pressure cap with v8.4.2's framework gravity (9.80616_RKIND)
        cap = (case["config"]["config_density0"] * ho.MPAS_GRAVITY) * 5.0
        assert (ref[name]["forcing/seaIcePressure"] == cap).any()
    assert (ref["hybrid_frazil"]["o2x/1"][:, col["Fioo_q"]] != 0).any()
    assert (ref["hybrid_land_ice_coupled"]["o2x/1"][:, col["So_blt"]] != 0).any()
    assert cases["hybrid_frazil_land_ice_mpas_constants"]["hybrid"]["constants"] == "mpas"


@requires_gpu
def test_gpu_matches_the_hybrid_oracle_word_for_word():
    report = oracle.measure()
    bad = {case: {k: v for k, v in r.items() if v} for case, r in report.items()}
    assert not any(bad.values()), bad
    assert all(len(r) > 150 for r in report.values())


@requires_gpu
@pytest.mark.parametrize(*MODES)
def test_word_exact_in_fast_and_strict_children(numerics, exact):
    run_child(__file__, ["test_gpu_matches_the_hybrid_oracle_word_for_word"], numerics, exact)


def _mismatches(name, *, mutate=None):
    case = _case(name)
    got, want = oracle.run_gpu(case, mutate=mutate), oracle.load_fixture()[name]
    return sum(oracle.word_mismatches(got[k], want[k]) for k in want)


@requires_gpu
@pytest.mark.parametrize("attr, value, case", [
    # the means not reset each window
    ("time_average_init", lambda forcing, configs: None, "hybrid_forward"),
    # v8.4.2's standalone T0_Kelvin under E3SM's constants
    ("SHR_T0_KELVIN", 273.16, "hybrid_forward"),
    # E3SM v3.2.0's float32-rounded gravity in the sea-ice pressure cap
    ("MPAS_GRAVITY", float(np.float32(9.80616)), "hybrid_frazil"),
])
def test_a_mutated_run_loop_is_caught(monkeypatch, attr, value, case):
    monkeypatch.setattr(ho, attr, value)
    assert _mismatches(case) > 0


@requires_gpu
def test_the_clock_lag_at_start_is_caught():
    # the clock not moved to its first coupling alarm (daysSinceStartOfSim, xtime)
    def lag(ocean):
        ocean.clock.now -= ocean.n_steps * ocean.clock.step
    assert _mismatches("hybrid_forward", mutate=lag) > 0


@requires_gpu
def test_a_nonzero_iceberg_flux_is_refused():
    case = _case("hybrid_forward")
    x2o = case["hybrid"]["x2o"].copy()
    x2o[0, 0, oracle.X2O_FIELDS.index("PFioi_bergw")] = 1.0e-9
    case["hybrid"]["x2o"] = x2o
    with pytest.raises(ConfigurationRefusal, match="iceberg"):
        oracle.run_gpu(case)


def test_every_divergence_is_listed_with_a_disposition():
    names = [d.name for d in ho.DIVERGENCES]
    assert len(names) == len(set(names))
    assert {d.disposition for d in ho.DIVERGENCES} <= {"ported", "refused", "omitted"}
    for d in ho.DIVERGENCES:
        assert d.where and d.reason
        if d.disposition == "refused":
            assert d.knob, d.name
    assert "not an E3SM configuration" in ho.HYBRID_NAME
    from woof.hex.coupled import registry
    name, module, components = registry.hybrids()["e3sm-v3.2.0-coupler+mpas-v8.4.2"]
    assert (name, module) == (ho.HYBRID_NAME, ho.__name__) and "ocean" in components


_BASE = {"config_density0": 1026.0, "config_dt": "00:02:00", "config_land_ice_flux_mode": "off"}


@pytest.mark.parametrize("change, kwargs, match", [
    ({"config_use_precip_scaling": True}, {}, "precipitation scaling"),
    ({"config_scale_dismf_by_removed_ice_runoff": True}, {}, "dismf"),
    ({"config_remove_ais_river_runoff": True}, {}, "removedRiverRunoffFlux"),
    ({"config_remove_ais_ice_runoff": True}, {}, "removedIceRunoffFlux"),
    ({"config_cvmix_kpp_use_theory_wave": True}, {}, "windSpeed10m"),
    ({"config_eke_correction_enable": True}, {}, "windSpeed10m"),
    ({"config_momentum_use_active_wave": True}, {}, "wave"),
    ({"config_submesoscale_enable": True}, {}, "submesoscale"),
    ({"config_use_ecosysTracers": True}, {}, "ecosystem"),
    ({"config_use_CFCTracers": True}, {}, "CFC"),
    ({}, {"wav_ocn_coup": "twoway"}, "wave"),
    ({"config_land_ice_flux_mode": "standalone"}, {}, "standalone"),
    ({"config_land_ice_flux_mode": "data"}, {}, "data"),
    ({"config_glc_thermal_forcing_coupling_mode": "3d"}, {}, "avgThermalForcingAtZLevels"),
    ({}, {"glc_nzoc": 4}, "avgThermalForcingAtZLevels"),
    ({"config_open_ocean_freezing_temperature_reference_pressure": -1.0}, {}, "reference pressure"),
    ({"config_density0": 1027.0}, {}, "1026"),
])
def test_the_divergences_are_refused_by_name(change, kwargs, match):
    with pytest.raises(ConfigurationRefusal, match=match) as info:
        ho.check({**_BASE, **change}, ocn_cpl_dt=1800, **kwargs)
    assert ho.HYBRID_NAME in str(info.value)


def test_the_accepted_configurations_pass():
    ho.check(_BASE, ocn_cpl_dt=1800)
    ho.check({**_BASE, "config_density0": 1027.0}, ocn_cpl_dt=1800, constants="mpas")
    for mode in ("pressure_only", "coupled"):
        ho.check({**_BASE, "config_land_ice_flux_mode": mode}, ocn_cpl_dt=1800)
    with pytest.raises((ConfigurationRefusal, ValueError)):
        ho.check(_BASE, ocn_cpl_dt=1801)
    with pytest.raises(ValueError):
        ho.check(_BASE, ocn_cpl_dt=1800, constants="cesm")


def test_the_default_path_does_not_build_the_hybrid():
    hits = [p for p in (ROOT / "woof").rglob("*.py")
            if "HybridOcean(" in p.read_text(encoding="utf-8") and p.name != "hybrid_ocean.py"
            and "verify" not in p.parts]
    assert hits == []
