"""The experimental sub-anchor timestep lane: opt-in, labelled, still gated.

The anchor table (:data:`woof.hex.dt_admission.ADMITTED_TIMESTEPS`) bottoms
out at 5 s, so a 50-100 m mesh -- Courant limit 0.3-0.6 s -- has no anchored
timestep at all.  The ruling: an explicit opt-in EXPERIMENTAL lane
(``woof hex forecast --experimental-dt`` or ``WOOF_HEX_EXPERIMENTAL_DT=1``)
that admits a timestep below the smallest anchor on the host-derivable
checks only -- Courant, cadence divisibility, RK shape, binary64 clock
closure -- labels everything it touches ``experimental-unanchored``, and
never writes a table row.

WOOF 1.1.0 retired the anchor gate (2026-10-10): no timestep is refused for
lacking an anchor, so the lane no longer OPENS anything -- it labels.  What
is pinned here, gate by gate (config, mesh-point choice, bind, door,
driver):

* without the flag a sub-anchor timestep runs unlabelled, as any timestep
  does since the gate's retirement;
* with the flag, 0.5 s and 0.25 s on synthetic 100 m and 50 m meshes are
  labelled ``experimental-unanchored``, and the mesh-point choice is
  exactly what ``largest_courant_divisor_dt`` returns;
* a timestep that fails stability, divisibility, clock closure or the
  millisecond quantum is refused by the lane, and the lane never applies at
  or above the smallest anchor;
* the label reaches every receipt and the history file.

Everything here runs on a CPU-only box.
"""

from __future__ import annotations

import argparse
import importlib
import types
from pathlib import Path

import numpy as np
import pytest

from woof.hex import dt_admission, mesh_point
from woof.hex import forecast_door as door
from woof.hex.config_v841 import V841MpasColumnPhysicsSmagorinskyGwdoConfig
from woof.hex.dt_admission import (
    ADMITTED_TIMESTEPS,
    EXPERIMENTAL_TIMESTEP_EVIDENCE,
    DtAdmissionError,
)
from woof.hex.errors import ConfigurationRefusal
from woof.hex.timestep_admission import (
    TimestepAdmissionError,
    admit_timestep,
    edge_length_authority,
    recommended_dt_seconds,
)

#: The ratio of min(dcEdge) to nominal spacing on the generator's graded
#: meshes (the "0.8484 x dx tail"), so a synthetic "100 m" mesh has a
#: min(dcEdge) of 84.84 m and a "50 m" one 42.42 m.
TAIL = 0.8484
MIN_DC_100M = TAIL * 100.0
MIN_DC_50M = TAIL * 50.0

_TABLE_KEYS = frozenset(ADMITTED_TIMESTEPS)


def _off_config(dt: float) -> V841MpasColumnPhysicsSmagorinskyGwdoConfig:
    return V841MpasColumnPhysicsSmagorinskyGwdoConfig(
        config_dt=dt,
        config_bldt_seconds=dt,
        config_cudt_seconds=None,
        config_convection_scheme="off",
    )


@pytest.fixture(autouse=True)
def _table_is_never_written():
    """Every test here ends with the anchor table exactly as it began."""

    yield
    assert frozenset(dt_admission.ADMITTED_TIMESTEPS) == _TABLE_KEYS
    assert dt_admission.active_experimental_lane() is None
    assert all(
        anchor.timestep_evidence is None
        for anchor in dt_admission.ADMITTED_TIMESTEPS.values()
    )


# ---------------------------------------------------------------------------
# dt_admission: the lane itself
# ---------------------------------------------------------------------------
def test_the_table_still_bottoms_out_at_five_seconds_and_says_diverges():
    assert dt_admission.smallest_anchored_dt() == 5.0
    fives = [a for a in ADMITTED_TIMESTEPS.values() if a.dt_seconds == 5.0]
    assert fives and all(a.physics_health_verdict == "DIVERGES" for a in fives)
    assert all("timestep_evidence" not in a.as_dict() for a in ADMITTED_TIMESTEPS.values())


@pytest.mark.parametrize(
    "dt,minimum",
    [(0.5, MIN_DC_100M), (0.25, MIN_DC_50M)],
)
def test_sub_anchor_timesteps_are_admitted_on_synthetic_fine_meshes(dt, minimum):
    anchor = dt_admission.experimental_dt_anchor(
        dt, cumulus_scheme=None, minimum_dc_edge_m=minimum
    )
    record = anchor.as_dict()
    assert record["timestep_evidence"] == EXPERIMENTAL_TIMESTEP_EVIDENCE
    assert "DIVERGES SEVERELY" in record["timestep_evidence_reason"]
    assert anchor.admitted_on == dt_admission.EXPERIMENTAL_ADMITTED_ON
    assert anchor.integration_anchor.startswith("NOT MEASURED")
    assert anchor.physics_health_verdict == "NOT"
    assert anchor.native_reference is None and anchor.experimental


@pytest.mark.parametrize("minimum", [MIN_DC_100M, MIN_DC_50M, 100.0, 50.0])
def test_the_lane_admits_exactly_what_largest_admissible_dt_returns(minimum):
    """The energy planner uses largest_courant_divisor_dt; the door must agree."""

    dt = dt_admission.largest_courant_divisor_dt(minimum)["largest_admissible_dt_seconds"]
    assert dt is not None and dt < 5.0
    anchor = dt_admission.experimental_dt_anchor(
        dt, cumulus_scheme=None, minimum_dc_edge_m=minimum
    )
    assert anchor.experimental


def test_a_timestep_failing_courant_is_refused_even_in_the_lane():
    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.experimental_dt_anchor(
            0.5, cumulus_scheme=None, minimum_dc_edge_m=MIN_DC_50M
        )
    assert "Courant" in str(caught.value)


@pytest.mark.parametrize(
    "dt,needle",
    [(0.35, "radiation_seconds=600"), (0.1, "clock closure fails"),
     (0.7, "radiation_seconds=600")],
)
def test_a_timestep_failing_divisibility_or_clock_is_refused_even_in_the_lane(dt, needle):
    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.experimental_dt_anchor(
            dt, cumulus_scheme=None, minimum_dc_edge_m=10_000.0
        )
    assert needle in str(caught.value)


def test_a_held_cadence_that_is_not_whole_steps_is_refused_in_the_lane():
    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.experimental_dt_anchor(
            0.5, cumulus_scheme=None, surface_pbl_seconds=0.75,
            minimum_dc_edge_m=MIN_DC_100M,
        )
    assert "surface_pbl_seconds" in str(caught.value)


@pytest.mark.parametrize("dt", [5.0, 6.0, 7.5, 60.0])
def test_the_lane_never_opens_at_or_above_the_smallest_anchor(dt):
    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.experimental_dt_anchor(
            dt, cumulus_scheme=None, minimum_dc_edge_m=1.0e6
        )
    assert "not below the smallest anchored" in str(caught.value) or (
        "holds a registered anchor" in str(caught.value)
    )


def test_the_lane_refuses_without_a_mesh_unless_courant_is_deferred_by_name():
    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.experimental_dt_anchor(0.5, cumulus_scheme=None, minimum_dc_edge_m=None)
    assert "Courant rule stays enforced" in str(caught.value)
    deferred = dt_admission.experimental_dt_anchor(
        0.5, cumulus_scheme=None, minimum_dc_edge_m=None,
        courant_deferred_to="bind_mesh",
    )
    assert '"deferred_to": "bind_mesh"' in deferred.schedule_receipt


def test_the_anchor_gate_is_retired_and_the_lane_only_labels():
    """WOOF 1.1.0 retired ``require_dt_anchor``: a sub-anchor timestep is
    recorded as unmatched evidence, not refused, and the lane's record is a
    label on that run -- never a table row."""

    assert not hasattr(dt_admission, "require_dt_anchor")
    for dt in (0.5, 0.25):
        record = dt_admission.anchor_record(
            dt, radiation_seconds=600.0, surface_pbl_seconds=dt,
            cumulus_seconds=None, cumulus_scheme=None,
        )
        assert record["matched"] is False
    anchor = dt_admission.experimental_dt_anchor(
        0.5, cumulus_scheme=None, minimum_dc_edge_m=MIN_DC_100M
    )
    assert anchor.experimental
    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.experimental_dt_anchor(
            7.5, cumulus_scheme=None, minimum_dc_edge_m=1.0e6
        )
    assert "not below the smallest anchored" in str(caught.value)





def test_the_config_admits_a_sub_anchor_timestep_on_its_cadences():
    """The config's gate is cadence integrality and clock closure (WOOF 1.1.0),
    in or out of a lane scope; the scope changes no admission."""

    _off_config(0.5).validate()
    with dt_admission.experimental_lane(0.5, minimum_dc_edge_m=MIN_DC_100M):
        _off_config(0.5).validate()
        _off_config(0.25).validate()


def test_a_scope_cannot_hold_a_timestep_the_lane_refuses_and_does_not_nest():
    with pytest.raises(DtAdmissionError):
        dt_admission.experimental_lane(0.5, minimum_dc_edge_m=MIN_DC_50M)
    with pytest.raises(DtAdmissionError):
        dt_admission.experimental_lane(10.0, minimum_dc_edge_m=1.0e6)
    # The scope is checked for the run's OWN cadence, not a default one.
    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.experimental_lane(
            0.5, minimum_dc_edge_m=MIN_DC_100M, surface_pbl_seconds=0.75
        )
    assert "surface_pbl_seconds" in str(caught.value)
    with dt_admission.experimental_lane(0.25, minimum_dc_edge_m=MIN_DC_50M):
        with pytest.raises(DtAdmissionError) as caught:
            with dt_admission.experimental_lane(0.25, minimum_dc_edge_m=MIN_DC_50M):
                pass
        assert "already open" in str(caught.value)
        assert dt_admission.active_experimental_lane() is not None


def test_the_anchor_verifier_never_certifies_an_experimental_record():
    import importlib.util
    import sys

    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(
        "_test_experimental_mint_tool", tools / "mint_dt_anchor.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        anchor = dt_admission.experimental_dt_anchor(
            0.5, cumulus_scheme=None, minimum_dc_edge_m=MIN_DC_100M
        )
        verdict = module.verify_anchor(anchor)
    finally:
        sys.modules.pop(spec.name, None)
    assert verdict["certified"] is False


# ---------------------------------------------------------------------------
# timestep_admission: the recommendation survives sub-second limits
# ---------------------------------------------------------------------------
def test_the_recommendation_is_representable_below_one_second():
    assert recommended_dt_seconds(103.67) == 103.6
    assert recommended_dt_seconds(6.2586) == 6.2
    assert recommended_dt_seconds(0.61085) == 0.61
    assert recommended_dt_seconds(0.305424) == 0.3
    assert recommended_dt_seconds(0.0723) == 0.072
    assert recommended_dt_seconds(0.0099) > 0.0
    # Rounded DOWN, never above the maximum, even a hair under a whole step.
    for maximum in (0.36 - 1.0e-14, 0.36, 0.72 - 1.0e-15, 1.0 - 1.0e-15, 0.1, 0.30000000000000004):
        assert 0.0 < recommended_dt_seconds(maximum) <= maximum


def test_the_search_reaches_meshes_finer_than_the_old_divisor_cutoff():
    """Below ~35 m the Courant floor's count passes 2400; the search follows it."""

    result = dt_admission.largest_courant_divisor_dt(TAIL * 25.0)
    assert result["largest_admissible_dt_seconds"] == 0.125
    dt_admission.experimental_dt_anchor(
        0.125, cumulus_scheme=None, minimum_dc_edge_m=TAIL * 25.0
    )
    # On the registered graded rows' edges the search answers an exact
    # divisor under the bind's stability limit (the retired 0.9 x 125 m/s
    # policy declared 75 s and 100 s there; the rows keep their own dt).
    for minimum in (13_311.8, 14_398.0):
        found = dt_admission.largest_courant_divisor_dt(minimum)
        dt = found["largest_admissible_dt_seconds"]
        assert dt <= found["courant_limit_seconds"]
        assert found["radiation_cadence_seconds"] % dt == 0.0


def test_the_courant_refusal_prints_a_sub_second_recommendation():
    authority = edge_length_authority(np.array([MIN_DC_50M, 60.0]))
    with pytest.raises(TimestepAdmissionError) as caught:
        admit_timestep(0.5, authority)
    # 0.3937 s maximum: rounded down at its second significant figure.
    assert "dt <= 0.39 s" in str(caught.value)
    coarse = edge_length_authority(np.array([869.25, 900.0]))
    with pytest.raises(TimestepAdmissionError) as caught:
        admit_timestep(20.0, coarse)
    assert "dt <= 8.0 s" in str(caught.value)


# ---------------------------------------------------------------------------
# mesh_point.choose_timestep
# ---------------------------------------------------------------------------
def test_choose_timestep_without_the_flag_takes_the_mesh_rule_unlabelled():
    """Without the flag a 100 m mesh takes MPAS-A's 6 s/km rule (WOOF 1.1.0),
    600/1024 s, with no label: nothing is refused for lacking an anchor."""

    chosen = mesh_point.choose_timestep(MIN_DC_100M, fine_dx_m=100.0)
    assert chosen["dt_seconds"] == 600.0 / 1024
    assert "timestep_evidence" not in chosen


@pytest.mark.parametrize(
    "fine_dx_m,expected", [(100.0, 0.75), (50.0, 0.375)]
)
def test_choose_timestep_with_the_flag_picks_largest_courant_divisor_dt(fine_dx_m, expected):
    minimum = TAIL * fine_dx_m
    chosen = mesh_point.choose_timestep(minimum, fine_dx_m=fine_dx_m, experimental=True)
    assert chosen["dt_seconds"] == expected
    assert chosen["dt_seconds"] == dt_admission.largest_courant_divisor_dt(minimum)[
        "largest_admissible_dt_seconds"
    ]
    assert chosen["timestep_evidence"] == EXPERIMENTAL_TIMESTEP_EVIDENCE
    assert chosen["cumulus_scheme"] is None
    assert chosen["dt_seconds"] <= chosen["courant_limit_seconds"]


def test_choose_timestep_with_the_flag_changes_nothing_where_an_anchor_fits():
    plain = mesh_point.choose_timestep(869.25, fine_dx_m=937.5)
    flagged = mesh_point.choose_timestep(869.25, fine_dx_m=937.5, experimental=True)
    assert plain == flagged and "timestep_evidence" not in flagged


# ---------------------------------------------------------------------------
# the forecast door
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _door_seams(monkeypatch):
    from woof.hex.device_admission import REFERENCE_CARD

    monkeypatch.setattr(door, "read_device_compute", lambda: (12, 0))
    monkeypatch.setattr(door, "read_card_profile", lambda: REFERENCE_CARD)
    monkeypatch.setattr(door, "seam_source_problem", lambda checkout: None, raising=False)
    # WOOF 1.1.0's door seams, supplied as tests/test_forecast_door.py does.
    monkeypatch.setattr(door, "seam_pin_problem", lambda checkout: None, raising=False)
    monkeypatch.setattr(door, "read_nvrtc_targets", lambda: (75, 80, 86, 89, 90, 100, 120, 121),
                        raising=False)
    monkeypatch.delenv(dt_admission.EXPERIMENTAL_DT_ENV, raising=False)


def _registry() -> dict[str, door.MeshRow]:
    return {
        "x1.40962": door.MeshRow("x1.40962", 40_962, 120.0, nominal_dx_m=120_000.0),
        "fine-100m": door.MeshRow("fine-100m", 20_000, 0.5, nominal_dx_m=100.0),
        "fine-50m": door.MeshRow("fine-50m", 20_000, 0.25, nominal_dx_m=50.0),
        "fine-odd": door.MeshRow("fine-odd", 20_000, 0.35, nominal_dx_m=100.0),
        "fine-7s": door.MeshRow("fine-7s", 20_000, 7.5, nominal_dx_m=1_000.0),
    }


def _namespace(tmp_path: Path, mesh: str, *extra: str) -> argparse.Namespace:
    grid, static, init = (tmp_path / name for name in ("g.nc", "s.nc", "i.nc"))
    for path in (grid, static, init):
        path.write_bytes(b"not really netcdf")
    checkout = tmp_path / "woof"
    checkout.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    return parser.parse_args(
        [
            "--mesh", mesh, "--grid", str(grid), "--static", str(static),
            "--init", str(init), "--init-source", "synthetic",
            "--hours", "0.5", "--history-every-minutes", "15",
            "--out", str(tmp_path / "out"), "--gpuwm-checkout", str(checkout),
            *extra,
        ]
    )


def _refusal(arguments: argparse.Namespace) -> str:
    with pytest.raises(door.ForecastDoorRefusal) as caught:
        door.resolve_request(arguments, registry=_registry())
    return str(caught.value)


@pytest.mark.parametrize("mesh,dt", [("fine-100m", 0.5), ("fine-50m", 0.25)])
def test_without_the_flag_the_door_runs_a_sub_anchor_dt_unlabelled(tmp_path, mesh, dt):
    request = door.resolve_request(
        _namespace(tmp_path, mesh, "--dt", repr(dt)), registry=_registry()
    )
    assert request.dt_seconds == dt
    assert not request.experimental_dt and not request.timestep_experimental
    assert "--experimental-dt" not in door.build_driver_argv(request)


@pytest.mark.parametrize("mesh,dt,steps", [("fine-100m", 0.5, 3600), ("fine-50m", 0.25, 7200)])
def test_the_door_admits_with_the_flag_and_labels_everything(tmp_path, mesh, dt, steps):
    request = door.resolve_request(
        _namespace(tmp_path, mesh, "--experimental-dt", "--dt", repr(dt)),
        registry=_registry(),
    )
    assert request.experimental_dt and request.timestep_experimental
    assert request.dt_seconds == dt and request.steps == steps
    assert request.capture_count == 3
    argv = door.build_driver_argv(request)
    assert "--experimental-dt" in argv
    receipt = door.build_receipt(
        request=request, admission=None, bind_receipt=None,
        driver_receipt=None, history=[], driver_argv=argv, seconds=0.0,
        status="test",
    )
    assert receipt["timestep_evidence"] == EXPERIMENTAL_TIMESTEP_EVIDENCE
    assert "DIVERGES SEVERELY" in receipt["timestep_evidence_reason"]
    warning = door.experimental_dt_warning(request)
    assert warning.startswith("WARNING EXPERIMENTAL TIMESTEP") and f"dt={dt:g} s" in warning


def test_the_door_admission_record_says_courant_is_the_binds(tmp_path):
    record = door.admit_timestep(
        "fine-100m", 0.5, nominal_dx_m=100.0, experimental=True, radiation_seconds=600.0,
    )
    assert record["timestep_evidence"] == EXPERIMENTAL_TIMESTEP_EVIDENCE
    assert record["anchor"]["matched"] is False
    plain = door.admit_timestep("fine-100m", 0.5, nominal_dx_m=100.0, radiation_seconds=600.0)
    assert "timestep_evidence" not in plain


def test_the_environment_switch_opens_the_lane_and_refuses_ambiguity(tmp_path, monkeypatch):
    monkeypatch.setenv(dt_admission.EXPERIMENTAL_DT_ENV, "1")
    request = door.resolve_request(
        _namespace(tmp_path, "fine-100m", "--dt", "0.5"), registry=_registry())
    assert request.experimental_dt and request.timestep_experimental
    monkeypatch.setenv(dt_admission.EXPERIMENTAL_DT_ENV, "0")
    for sub in ("zero", "yes", "pre"):
        (tmp_path / sub).mkdir()
    plain = door.resolve_request(
        _namespace(tmp_path / "zero", "fine-100m", "--dt", "0.5"), registry=_registry())
    assert not plain.experimental_dt
    monkeypatch.setenv(dt_admission.EXPERIMENTAL_DT_ENV, "yes")
    assert "is neither 1" in _refusal(_namespace(tmp_path / "yes", "fine-100m", "--dt", "0.5"))
    # Preflight reports the ambiguous switch BESIDE the other problems.
    arguments = _namespace(tmp_path / "pre", "fine-100m", "--preflight", "--dt", "0.5")
    arguments.init = tmp_path / "absent.init.nc"
    request = door.resolve_request(arguments, registry=_registry())
    joined = " ".join(request.input_problems)
    assert "is neither 1" in joined and "absent.init.nc" in joined
    assert not request.experimental_dt


def test_the_flag_does_not_admit_a_timestep_that_fails_divisibility(tmp_path):
    message = _refusal(_namespace(tmp_path, "fine-odd", "--experimental-dt", "--dt", "0.35"))
    assert "dt=0.35 s" in message
    assert "not a positive integer multiple" in message
    # The lane refuses it by the same rung when it is the one asked.
    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.experimental_dt_anchor(
            0.35, cumulus_scheme=None, minimum_dc_edge_m=None,
            courant_deferred_to="bind_mesh",
        )
    assert "radiation_seconds=600" in str(caught.value)


def test_the_flag_labels_nothing_at_or_above_five_seconds(tmp_path):
    """Above the smallest anchor the lane does not apply: an unanchored 7.5 s
    runs (the gate is retired) and carries no experimental label."""

    request = door.resolve_request(
        _namespace(tmp_path, "fine-7s", "--experimental-dt", "--dt", "7.5"),
        registry=_registry(),
    )
    assert request.experimental_dt and not request.timestep_experimental
    receipt = door.build_receipt(
        request=request, admission=None, bind_receipt=None, driver_receipt=None,
        history=[], driver_argv=[], seconds=0.0, status="test",
    )
    assert "timestep_evidence" not in receipt


def test_an_ordinary_run_carries_no_label_and_no_flag(tmp_path):
    flagged = _namespace(tmp_path, "x1.40962", "--experimental-dt")
    flagged.history_every_minutes = 30
    request = door.resolve_request(flagged, registry=_registry())
    assert request.experimental_dt and not request.timestep_experimental
    ordinary = _namespace(tmp_path, "x1.40962")
    ordinary.history_every_minutes = 30
    plain = door.resolve_request(ordinary, registry=_registry())
    assert "--experimental-dt" not in door.build_driver_argv(plain)
    receipt = door.build_receipt(
        request=plain, admission=None, bind_receipt=None, driver_receipt=None,
        history=[], driver_argv=[], seconds=0.0, status="test",
    )
    assert "timestep_evidence" not in receipt


def test_the_schedule_handles_a_sub_second_timestep():
    assert door._schedule(0.25, 15, 600.0 / 1024, "m") == (1536, 2)
    with pytest.raises(door.ForecastDoorRefusal) as caught:
        door._schedule(1.0, 1, 0.35, "m")
    assert "0.35 s" in str(caught.value)


def test_the_forecast_parser_carries_the_flag():
    from woof.hex.cli import build_parser

    arguments = build_parser().parse_args(
        ["forecast", "--mesh", "m", "--experimental-dt"]
    )
    assert arguments.experimental_dt is True
    assert build_parser().parse_args(["forecast"]).experimental_dt is False


# ---------------------------------------------------------------------------
# bind_mesh
# ---------------------------------------------------------------------------
class _Reached(Exception):
    """The bind got past every timestep decision."""


def _binding_module(monkeypatch, *, dt: float, minimum: float, nominal: float):
    binding = door._load_module(
        "mpas_mesh_binding_experimental", door.DRIVERS_DIR / "mpas_mesh_binding.py"
    )
    row = binding.MeshBinding(
        name="synthetic-fine", n_cells=10, n_edges=30, n_levels=55,
        n_interfaces=56, n_soil_levels=4, nominal_dx_m=nominal, dt_seconds=dt,
        grid_bytes=1, grid_sha256="0" * 64, static_bytes=1, static_sha256="0" * 64,
    )
    monkeypatch.setattr(binding, "MESH_BINDINGS", {**binding.MESH_BINDINGS, row.name: row})
    monkeypatch.setattr(
        binding, "_require_file",
        lambda role, path, want_bytes, want_sha, mesh: {"path": str(path)},
    )
    monkeypatch.setattr(
        binding, "_inspect_grid",
        lambda path, row: {"nCells": 10, "nEdges": 30, "nominalMinDc_f32": nominal},
    )
    monkeypatch.setattr(binding, "admit_regional_row", lambda row, observed: {"regional": False})
    monkeypatch.setattr(binding, "_grid_cell_spacing", lambda path, len_disp_m: None)
    monkeypatch.setattr(binding, "_inspect_static", lambda path, row, observed: {})
    monkeypatch.setattr(
        binding, "_static_edge_authority",
        lambda path: edge_length_authority(np.array([minimum, 2.0 * minimum])),
    )
    passing = types.SimpleNamespace(
        as_dict=lambda: {}, minimum_ratio=1.0, minimum_ratio_edge=0,
        minimum_ratio_dv_edge_m=1.0, amplification=1.0,
        policy=types.SimpleNamespace(minimum_dv_over_dc=0.1, apvm_upwinding=0.5),
    )
    monkeypatch.setattr(binding, "_static_dual_edges", lambda path: (None, None, None))
    monkeypatch.setattr(binding, "admit_dual_edges", lambda *a, **k: passing)
    monkeypatch.setattr(binding, "_grid_cell_coordination", lambda path: None)
    monkeypatch.setattr(binding, "admit_cell_coordination", lambda *a, **k: passing)

    def _stop(proof, authority):
        raise _Reached

    monkeypatch.setattr(binding, "_fingerprint_with_authority", _stop)
    return binding


def _bind(binding, tmp_path, **kwargs):
    forecast = types.SimpleNamespace(TIMESTEP_EVIDENCE_DECISION="stale")
    lines: list[str] = []
    binding.bind_mesh(
        object(), "synthetic-fine", grid=tmp_path / "g.nc", static=tmp_path / "s.nc",
        forecast=forecast, verify_frozen_sources=False, log=lines.append, **kwargs,
    )
    return forecast, lines


@pytest.mark.parametrize(
    "dt,minimum,nominal", [(0.5, MIN_DC_100M, 100.0), (0.25, MIN_DC_50M, 50.0)]
)
def test_bind_labels_only_with_the_flag(
    tmp_path, monkeypatch, dt, minimum, nominal
):
    """Without the flag the bind admits a sub-anchor timestep unlabelled
    (WOOF 1.1.0 retired the anchor gate); with it, it warns and labels."""

    binding = _binding_module(monkeypatch, dt=dt, minimum=minimum, nominal=nominal)
    with pytest.raises(_Reached):
        forecast, lines = _bind(binding, tmp_path, dt_seconds=dt, radiation_seconds=600.0)
    plain_lines: list[str] = []
    with pytest.raises(_Reached):
        binding.bind_mesh(
            object(), "synthetic-fine", grid=tmp_path / "g.nc",
            static=tmp_path / "s.nc", forecast=types.SimpleNamespace(),
            verify_frozen_sources=False, dt_seconds=dt, radiation_seconds=600.0,
            log=plain_lines.append,
        )
    assert not any("EXPERIMENTAL-UNANCHORED" in line for line in plain_lines)

    forecast = types.SimpleNamespace(TIMESTEP_EVIDENCE_DECISION={"stale": True})
    lines: list[str] = []
    with pytest.raises(_Reached):
        binding.bind_mesh(
            object(), "synthetic-fine", grid=tmp_path / "g.nc",
            static=tmp_path / "s.nc", forecast=forecast,
            verify_frozen_sources=False, experimental_dt=True, log=lines.append,
            dt_seconds=dt, radiation_seconds=600.0,
        )
    assert any("WARNING" in line and "EXPERIMENTAL-UNANCHORED" in line for line in lines)
    assert any(f"dt={dt:.9g} s" in line for line in lines)
    # The bind stopped (the stub) before succeeding, so nothing was handed
    # to the driver -- and the stale decision was cleared at entry.
    assert forecast.TIMESTEP_EVIDENCE_DECISION is None


def test_bind_publishes_its_decision_only_through_the_success_helper(monkeypatch):
    binding = _binding_module(monkeypatch, dt=0.5, minimum=MIN_DC_100M, nominal=100.0)
    forecast = types.SimpleNamespace()
    decision = {"timestep_evidence": EXPERIMENTAL_TIMESTEP_EVIDENCE, "dt_seconds": 0.5}
    binding._publish_timestep_evidence(forecast, decision)
    assert forecast.TIMESTEP_EVIDENCE_DECISION == decision
    assert forecast.TIMESTEP_EVIDENCE_DECISION is not decision
    binding._publish_timestep_evidence(forecast, None)
    assert forecast.TIMESTEP_EVIDENCE_DECISION is None
    assert binding._timestep_evidence_fields(None) == {}
    assert binding._timestep_evidence_fields(decision) == (
        dt_admission.experimental_evidence_fields()
    )


def test_bind_refuses_a_courant_failure_even_with_the_flag(tmp_path, monkeypatch):
    binding = _binding_module(monkeypatch, dt=0.5, minimum=MIN_DC_50M, nominal=50.0)
    with pytest.raises(binding.MeshBindingMismatch) as caught:
        _bind(binding, tmp_path, experimental_dt=True, dt_seconds=0.5, radiation_seconds=600.0)
    assert "unstable mesh/timestep combination" in str(caught.value)


def test_bind_resets_a_stale_decision_on_an_ordinary_bind(tmp_path, monkeypatch):
    binding = _binding_module(monkeypatch, dt=5.0, minimum=5_000.0, nominal=1_000.0)
    forecast = types.SimpleNamespace(TIMESTEP_EVIDENCE_DECISION={"stale": True})
    with pytest.raises(_Reached):
        binding.bind_mesh(
            object(), "synthetic-fine", grid=tmp_path / "g.nc",
            static=tmp_path / "s.nc", forecast=forecast,
            verify_frozen_sources=False, experimental_dt=True, log=lambda _: None,
            dt_seconds=5.0, radiation_seconds=600.0,
        )
    assert forecast.TIMESTEP_EVIDENCE_DECISION is None


# ---------------------------------------------------------------------------
# the forecast driver
# ---------------------------------------------------------------------------
@pytest.fixture()
def driver():
    return importlib.import_module("woof.hex.drivers.run_cuda_v841_forecast")


def _decision(dt: float, minimum: float) -> dict:
    return {
        "timestep_evidence": EXPERIMENTAL_TIMESTEP_EVIDENCE,
        "timestep_evidence_reason": dt_admission.EXPERIMENTAL_DT_REASON,
        "dt_seconds": dt,
        "minimum_dc_edge_m": minimum,
        "smallest_anchored_dt_seconds": 5.0,
    }


def test_the_driver_takes_one_decision_from_the_bind(driver, monkeypatch):
    monkeypatch.setattr(driver, "DT_SECONDS", 0.5)
    monkeypatch.setattr(driver, "TIMESTEP_EVIDENCE_DECISION", _decision(0.5, MIN_DC_100M))
    assert driver.timestep_evidence_decision(True)["dt_seconds"] == 0.5
    with pytest.raises(ConfigurationRefusal) as caught:
        driver.timestep_evidence_decision(False)
    assert "one decision, one source" in str(caught.value)

    monkeypatch.setattr(driver, "TIMESTEP_EVIDENCE_DECISION", None)
    with pytest.raises(ConfigurationRefusal) as caught:
        driver.timestep_evidence_decision(True)
    assert "min(dcEdge)" in str(caught.value)
    assert driver.timestep_evidence_decision(False) is None

    monkeypatch.setattr(driver, "DT_SECONDS", 120.0)
    assert driver.timestep_evidence_decision(True) is None


def test_the_driver_schedule_handles_a_sub_second_timestep(driver, monkeypatch):
    monkeypatch.setattr(driver, "DT_SECONDS", 0.25)
    schedule = driver.build_schedule(
        hours=0.5, history_every_minutes=15, start_text="2026-10-10_00:00:00"
    )
    assert schedule["steps"] == 7200 and schedule["history_stride_steps"] == 3600
    assert schedule["capture_steps"] == [0, 3600, 7200]
    monkeypatch.setattr(driver, "DT_SECONDS", 120.0)
    with pytest.raises(ValueError) as caught:
        driver.build_schedule(hours=1.0, history_every_minutes=3, start_text="2026-10-10_00:00:00")
    assert "120 s steps" in str(caught.value)


def test_the_history_file_carries_the_label_and_its_new_digest(driver, tmp_path):
    from netCDF4 import Dataset

    path = tmp_path / "cuda-history.test.nc"
    with Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.setncattr("schema", "test")
    record = driver.stamp_timestep_evidence(path, _decision(0.5, MIN_DC_100M))
    with Dataset(path) as dataset:
        assert dataset.getncattr("timestep_evidence") == EXPERIMENTAL_TIMESTEP_EVIDENCE
        assert "DIVERGES SEVERELY" in dataset.getncattr("timestep_evidence_reason")
        assert dataset.getncattr("timestep_dt_seconds") == "0.5"
    assert record["sha256"] == driver.proof.sha256_file(path)
    assert record["bytes"] == path.stat().st_size


def test_the_driver_parser_carries_the_flag(driver):
    args = driver.parse_args(
        ["--init", "i.nc", "--init-source", "synthetic", "--hours", "1",
         "--history-every-minutes", "30",
         "--preflight-only", "--experimental-dt"]
    )
    assert args.experimental_dt is True


@pytest.mark.parametrize("dt,minimum", [(600.0 / 1024, MIN_DC_100M), (600.0 / 2048, MIN_DC_50M)])
def test_a_timestep_off_the_millisecond_quantum_is_refused(dt, minimum):
    """600/1024 s passes Courant, radiation divisibility and clock closure, but
    the regional class key and the mint tooling quantise to whole
    milliseconds: it could never be minted, so the lane refuses it and the
    search skips it."""

    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.experimental_dt_anchor(
            dt, cumulus_scheme=None, minimum_dc_edge_m=minimum
        )
    assert "whole number of milliseconds" in str(caught.value)
    search = dt_admission.largest_courant_divisor_dt(minimum)
    assert search["rejected_for_millisecond_quantum"]
    assert not any(dt_admission.whole_milliseconds(x)
                   for x in search["rejected_for_millisecond_quantum"])
    assert dt_admission.whole_milliseconds(search["largest_admissible_dt_seconds"])
