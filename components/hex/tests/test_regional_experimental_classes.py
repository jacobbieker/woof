"""The EXPERIMENTAL regional-class lane and the local contract-deck receipt.

CPU-only.  What these tests hold:

* an unminted FINE class (``graded-117m-dt0.5-z7``) is refused with the lane
  closed and admitted with it open, labelled ``"experimental-unminted"``;
* the lane never relaxes the geometry half (a contract deck is still
  required), the Courant ceiling, or the "fine" scope, and never adds a row
  to ``ADMITTED_CLASSES``;
* with the lane closed every decision is the one the gate made before;
* the local receipt tool measures a synthetic cull, stamps a deck receipt,
  and that stamped receipt passes the gate only with the lane open.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import regional_contract_receipt as rcr
from woof.hex.cuda_backend import regional_admission as ra
from woof.hex.mesh import regional_boundary_mask_digest

KERNEL_SET = ra.kernel_set_sha256()
EARTH = 6_371_229.0


@pytest.fixture(autouse=True)
def _lane_closed_by_default(monkeypatch):
    monkeypatch.delenv(ra.EXPERIMENTAL_DT_ENVIRONMENT, raising=False)
    monkeypatch.delenv(ra.CONTRACT_LEDGER_ENVIRONMENT, raising=False)


def _receipt(digest: str, n_cells: int, class_id: str, **extra) -> dict:
    document = {
        "instrument": "run_cuda_regional_contract",
        "all_decks_bitwise": True,
        "all_kernels_covered": True,
        "all_controls_have_teeth": True,
        "dual_run_identical": True,
        "decks_selected": False,
        "bdy_mask_sha256": digest,
        "n_cells": n_cells,
        "boundary_zone_width": 7,
        "kernel_set_sha256": KERNEL_SET,
        "class_id": class_id,
        "card": "synthetic",
        "date_utc": "2026-10-10T00:00:00Z",
    }
    document.update(extra)
    return document


def _ledger(tmp_path: Path, *documents: dict) -> Path:
    ledger = tmp_path / "ledger"
    ledger.mkdir(exist_ok=True)
    for index, document in enumerate(documents):
        (ledger / f"r{index}.json").write_text(json.dumps(document))
    return ledger


FINE = dict(boundary_zone_width=7, n_vert_levels=55, finest_edge_m=117.2, dt_seconds=0.5)


# ---------------------------------------------------------------------------
# the class-id grammar
# ---------------------------------------------------------------------------


def test_the_grammar_reproduces_every_measured_minted_row_id():
    for class_id, row in ra.ADMITTED_CLASSES.items():
        if row.key.finest_edge_measured:
            assert ra.class_id_for_key(row.key) == class_id


def test_the_grammar_names_the_fine_classes():
    key = ra.RegionalClassKey.build(kernel_set=KERNEL_SET, **FINE)
    assert ra.class_id_for_key(key) == "graded-117m-dt0.5-z7"
    finer = ra.RegionalClassKey.build(
        boundary_zone_width=7, n_vert_levels=55, finest_edge_m=58.6,
        dt_seconds=0.25, kernel_set=KERNEL_SET,
    )
    assert ra.class_id_for_key(finer) == "graded-59m-dt0.25-z7"
    other_column = ra.RegionalClassKey.build(
        boundary_zone_width=7, n_vert_levels=80, finest_edge_m=58.6,
        dt_seconds=0.25, kernel_set=KERNEL_SET,
    )
    assert ra.class_id_for_key(other_column) == "graded-59m-dt0.25-z7-l80"


# ---------------------------------------------------------------------------
# the gate: closed, open, and what stays enforced when open
# ---------------------------------------------------------------------------


def test_an_unminted_fine_class_is_refused_without_the_flag(tmp_path):
    digest = "ab" * 32
    ledger = _ledger(tmp_path, _receipt(digest, 400, "graded-117m-dt0.5-z7"))
    with pytest.raises(ra.RegionalAdmissionRefusal) as error:
        ra.require_regional_anchor(
            None, bdy_mask_sha256=digest, n_cells=400,
            contract_directories=[ledger], **FINE,
        )
    assert "holds no forecast mint" in str(error.value)
    # An explicit False is closed even with the environment open.
    with pytest.raises(ra.RegionalAdmissionRefusal):
        ra.require_regional_anchor(
            None, bdy_mask_sha256=digest, n_cells=400,
            contract_directories=[ledger], experimental=False, **FINE,
        )


def test_the_environment_opens_the_lane_only_on_exactly_one(monkeypatch, tmp_path):
    digest = "ab" * 32
    ledger = _ledger(tmp_path, _receipt(digest, 400, "graded-117m-dt0.5-z7"))
    for value in ("true", "yes", "0", ""):
        monkeypatch.setenv(ra.EXPERIMENTAL_DT_ENVIRONMENT, value)
        assert not ra.experimental_lane_requested()
        with pytest.raises(ra.RegionalAdmissionRefusal):
            ra.require_regional_anchor(
                None, bdy_mask_sha256=digest, n_cells=400,
                contract_directories=[ledger], **FINE,
            )
    monkeypatch.setenv(ra.EXPERIMENTAL_DT_ENVIRONMENT, "1")
    assert ra.experimental_lane_requested()
    anchor = ra.require_regional_anchor(
        None, bdy_mask_sha256=digest, n_cells=400,
        contract_directories=[ledger], **FINE,
    )
    assert anchor.class_evidence == ra.CLASS_EVIDENCE_EXPERIMENTAL


def test_an_unminted_fine_class_is_admitted_with_the_flag_and_labelled(tmp_path):
    before = dict(ra.ADMITTED_CLASSES)
    digest = "ab" * 32
    ledger = _ledger(tmp_path, _receipt(digest, 400, "graded-117m-dt0.5-z7"))
    anchor = ra.require_regional_anchor(
        None, bdy_mask_sha256=digest, n_cells=400,
        contract_directories=[ledger], experimental=True, **FINE,
    )
    assert anchor.class_id == "graded-117m-dt0.5-z7"
    assert anchor.experimental
    record = anchor.as_dict()
    assert record["class_evidence"] == "experimental-unminted"
    assert record["contract_route"] == "presented"
    assert record["forecast_anchor"] == ""
    assert "EXPERIMENTAL-UNMINTED" in record["basis"]
    assert record["courant_limit_seconds"] == pytest.approx(117.2 * 0.9 / 125.0)
    # Never a row of the shipped table.
    assert dict(ra.ADMITTED_CLASSES) == before
    assert ra.admitted_class("graded-117m-dt0.5-z7") is None


def test_the_flag_never_waives_the_contract_deck():
    with pytest.raises(ra.RegionalAdmissionRefusal) as error:
        ra.require_regional_anchor(
            None, bdy_mask_sha256="cd" * 32, n_cells=400,
            experimental=True, **FINE,
        )
    message = str(error.value)
    assert "no contract deck has been run" in message
    assert "graded-117m-dt0.5-z7" in message
    assert "woof.hex.regional_contract_receipt" in message
    assert "IS minted" not in message
    assert "holds NO forecast mint" in message


def test_the_flag_never_relaxes_the_courant_ceiling(tmp_path):
    digest = "ab" * 32
    ledger = _ledger(tmp_path, _receipt(digest, 400, "graded-117m-dt1-z7"))
    with pytest.raises(ra.RegionalAdmissionRefusal) as error:
        ra.require_regional_anchor(
            None, bdy_mask_sha256=digest, n_cells=400,
            contract_directories=[ledger], experimental=True,
            boundary_zone_width=7, n_vert_levels=55, finest_edge_m=117.2,
            dt_seconds=1.0,
        )
    assert "Courant ceiling" in str(error.value)


def test_the_flag_does_not_admit_a_coarse_unminted_class():
    with pytest.raises(ra.RegionalAdmissionRefusal) as error:
        ra.require_regional_anchor(
            None, bdy_mask_sha256="00" * 32, n_cells=2_971,
            boundary_zone_width=7, n_vert_levels=55,
            finest_edge_m=1234.5, dt_seconds=7.0, experimental=True,
        )
    message = str(error.value)
    assert "holds no forecast mint" in message
    assert "not a FINE class" in message


def test_a_receipt_claiming_another_unminted_class_is_refused(tmp_path):
    digest = "ab" * 32
    ledger = _ledger(tmp_path, _receipt(digest, 400, "graded-117m-dt0.25-z7"))
    with pytest.raises(ra.RegionalAdmissionRefusal) as error:
        ra.require_regional_anchor(
            None, bdy_mask_sha256=digest, n_cells=400,
            contract_directories=[ledger], experimental=True, **FINE,
        )
    assert "two classifiers disagreeing" in str(error.value)


def test_the_matching_stamp_is_preferred_among_receipts_for_one_geometry(tmp_path):
    digest = "ab" * 32
    ledger = _ledger(
        tmp_path,
        _receipt(digest, 400, "graded-117m-dt0.25-z7"),
        _receipt(digest, 400, "graded-117m-dt0.5-z7"),
    )
    anchor = ra.require_regional_anchor(
        None, bdy_mask_sha256=digest, n_cells=400,
        contract_directories=[ledger], experimental=True, **FINE,
    )
    assert anchor.contract_receipt.endswith("r1.json")


def test_the_flag_never_waives_the_cell_count(tmp_path):
    digest = "ab" * 32
    ledger = _ledger(tmp_path, _receipt(digest, 400, "graded-117m-dt0.5-z7"))
    with pytest.raises(ra.RegionalAdmissionRefusal):
        ra.require_regional_anchor(
            None, bdy_mask_sha256=digest, n_cells=401,
            contract_directories=[ledger], experimental=True, **FINE,
        )


def _shipped_869():
    for contract in ra.SHIPPED_CONTRACTS.values():
        if contract.class_id == "graded-869m-dt5-z7":
            return contract
    pytest.skip("no shipped contract of the 869 m class in this tree")


def test_a_minted_key_takes_the_minted_route_with_the_flag_open():
    contract = _shipped_869()
    klass = ra.ADMITTED_CLASSES[contract.class_id]
    anchor = ra.require_regional_anchor(
        contract.mesh_row, bdy_mask_sha256=contract.bdy_mask_sha256,
        n_cells=contract.n_cells,
        boundary_zone_width=klass.key.boundary_zone_width,
        n_vert_levels=klass.key.n_vert_levels,
        finest_edge_m=klass.key.finest_edge_mm / 1000.0,
        dt_seconds=5.0, experimental=True,
    )
    assert anchor.class_id == "graded-869m-dt5-z7"
    assert anchor.class_evidence == ra.CLASS_EVIDENCE_MINTED
    assert anchor.courant_limit_seconds is None


def test_a_shipped_deck_carries_a_sub_5s_experimental_run():
    contract = _shipped_869()
    klass = ra.ADMITTED_CLASSES[contract.class_id]
    kwargs = dict(
        bdy_mask_sha256=contract.bdy_mask_sha256, n_cells=contract.n_cells,
        boundary_zone_width=klass.key.boundary_zone_width,
        n_vert_levels=klass.key.n_vert_levels,
        finest_edge_m=klass.key.finest_edge_mm / 1000.0, dt_seconds=2.0,
    )
    with pytest.raises(ra.RegionalAdmissionRefusal):
        ra.require_regional_anchor(contract.mesh_row, **kwargs)
    anchor = ra.require_regional_anchor(
        contract.mesh_row, experimental=True, **kwargs
    )
    assert anchor.class_id == "graded-869m-dt2-z7"
    assert anchor.class_evidence == ra.CLASS_EVIDENCE_EXPERIMENTAL
    assert anchor.contract_class_claim == "graded-869m-dt5-z7"
    assert anchor.contract_route == "shipped"


def test_the_global_door_site_is_untouched_by_the_flag():
    """Without a measured key the lane cannot apply; refusals are as before."""

    with pytest.raises(ra.RegionalAdmissionRefusal) as error:
        ra.require_regional_anchor(
            None, bdy_mask_sha256="00" * 32, n_cells=2_971, experimental=True,
        )
    assert "no contract deck has been run" in str(error.value)


# ---------------------------------------------------------------------------
# the local receipt on a synthetic cull
# ---------------------------------------------------------------------------


def _write_synthetic_cull(directory: Path, *, finest_m: float = 117.2) -> dict[str, Path]:
    """A tiny regional cull: 30 cells, a 7-ring zone, unit-sphere grid."""

    import netCDF4

    n_cells, n_edges, n_vertices = 30, 90, 60
    rng = np.random.default_rng(7)
    cell_mask = np.concatenate([np.zeros(16), np.repeat(np.arange(1, 8), 2)]).astype("i4")
    edge_mask = rng.integers(0, 8, n_edges).astype("i4")
    vertex_mask = rng.integers(0, 8, n_vertices).astype("i4")
    dc_edge_m = np.linspace(finest_m, finest_m * 3.0, n_edges)
    paths = {
        "grid": directory / "cull.grid.nc",
        "static": directory / "cull.static.nc",
        "init": directory / "cull.init.nc",
    }
    for role in ("grid", "static"):
        with netCDF4.Dataset(paths[role], "w") as dataset:
            dataset.createDimension("nCells", n_cells)
            dataset.createDimension("nEdges", n_edges)
            dataset.createDimension("nVertices", n_vertices)
            radius = 1.0 if role == "grid" else EARTH
            dataset.sphere_radius = radius
            dc = dataset.createVariable("dcEdge", "f8", ("nEdges",))
            dc[:] = dc_edge_m * (radius / EARTH)
            if role == "grid":
                for name, dim, values in (
                    ("bdyMaskCell", "nCells", cell_mask),
                    ("bdyMaskEdge", "nEdges", edge_mask),
                    ("bdyMaskVertex", "nVertices", vertex_mask),
                ):
                    variable = dataset.createVariable(name, "i4", (dim,))
                    variable[:] = values
    with netCDF4.Dataset(paths["init"], "w") as dataset:
        dataset.createDimension("nCells", n_cells)
        dataset.createDimension("nVertLevels", 55)
    paths["digest"] = regional_boundary_mask_digest(
        {"bdyMaskCell": cell_mask, "bdyMaskEdge": edge_mask, "bdyMaskVertex": vertex_mask}
    )  # type: ignore[assignment]
    return paths


def _fake_deck_runner(digest: str, *, verdict: bool = True, calls: list | None = None):
    def runner(*, grid, init, lbc_dir, out, class_id, mesh_row, start_time):
        if calls is not None:
            calls.append(class_id)
        Path(out).write_text(json.dumps(_receipt(
            digest, 30, class_id, all_decks_bitwise=verdict,
        )))
        return 0 if verdict else 1
    return runner


def test_the_geometry_is_measured_off_the_synthetic_cull(tmp_path):
    paths = _write_synthetic_cull(tmp_path)
    geometry = rcr.measure_cull_geometry(
        paths["grid"], static=paths["static"], init=paths["init"]
    )
    assert geometry.bdy_mask_sha256 == paths["digest"]
    assert geometry.n_cells == 30
    assert geometry.boundary_zone_width == 7
    assert geometry.finest_edge_m == pytest.approx(117.2)
    assert geometry.n_vert_levels == 55
    # Without a static file the unit-sphere grid is scaled to metres.
    bare = rcr.measure_cull_geometry(paths["grid"])
    assert bare.finest_edge_m == pytest.approx(117.2)
    assert rcr.grid_is_regional(paths["grid"])
    assert not rcr.grid_is_regional(paths["static"])


def test_a_local_receipt_is_stamped_experimental_and_passes_the_gate(tmp_path):
    paths = _write_synthetic_cull(tmp_path)
    ledger = tmp_path / "ledger"
    calls: list[str] = []
    summary = rcr.generate_local_contract_receipt(
        paths["grid"], out=ledger / "cull.json", dt_seconds=0.5,
        static=paths["static"], init=paths["init"], lbc_dir=tmp_path,
        experimental=True,
        deck_runner=_fake_deck_runner(paths["digest"], calls=calls),
    )
    assert calls == ["graded-117m-dt0.5-z7"]
    assert summary["class_id"] == "graded-117m-dt0.5-z7"
    assert summary["class_evidence"] == "experimental-unminted"
    assert summary["deck_passed"] is True
    stamped = json.loads((ledger / "cull.json").read_text())
    assert stamped["class_evidence"] == "experimental-unminted"
    assert stamped["mint_pair"] is None
    assert stamped["instrument"] == "run_cuda_regional_contract"
    assert stamped["stamp"]["schema"] == rcr.SCHEMA
    assert len(stamped["stamp"]["deck_receipt_sha256"]) == 64
    assert not list(ledger.glob("*.partial"))

    closed = rcr.regional_preflight(
        paths["grid"], dt_seconds=0.5, static=paths["static"],
        init=paths["init"], contract_directories=[ledger],
    )
    assert closed["admitted"] is False
    assert "holds no forecast mint" in closed["refusal"]
    assert closed["measured_class_id"] == "graded-117m-dt0.5-z7"

    opened = rcr.regional_preflight(
        paths["grid"], dt_seconds=0.5, static=paths["static"],
        init=paths["init"], contract_directories=[ledger], experimental=True,
    )
    assert opened["admitted"] is True
    assert opened["class_id"] == "graded-117m-dt0.5-z7"
    assert opened["class_evidence"] == "experimental-unminted"
    assert opened["contract_route"] == "presented"
    assert opened["anchor"]["class_evidence"] == "experimental-unminted"


def test_receipt_generation_refuses_an_unminted_class_without_the_flag(tmp_path):
    paths = _write_synthetic_cull(tmp_path)
    calls: list[str] = []
    with pytest.raises(rcr.RegionalReceiptRefusal) as error:
        rcr.generate_local_contract_receipt(
            paths["grid"], out=tmp_path / "ledger" / "cull.json",
            dt_seconds=0.5, static=paths["static"], init=paths["init"],
            lbc_dir=tmp_path,
            deck_runner=_fake_deck_runner(paths["digest"], calls=calls),
        )
    assert "--experimental-dt" in str(error.value)
    # The class is resolved before any card time is spent.
    assert calls == []
    assert not (tmp_path / "ledger" / "cull.json").exists()


def test_receipt_generation_refuses_a_courant_violation(tmp_path):
    paths = _write_synthetic_cull(tmp_path)
    with pytest.raises(rcr.RegionalReceiptRefusal) as error:
        rcr.generate_local_contract_receipt(
            paths["grid"], out=tmp_path / "cull.json", dt_seconds=1.0,
            static=paths["static"], init=paths["init"], lbc_dir=tmp_path,
            experimental=True, deck_runner=_fake_deck_runner(paths["digest"]),
        )
    assert "Courant ceiling" in str(error.value)


def test_a_deck_from_other_rings_is_never_stamped(tmp_path):
    paths = _write_synthetic_cull(tmp_path)
    deck = tmp_path / "deck.json"
    deck.write_text(json.dumps(_receipt("ef" * 32, 30, "")))
    with pytest.raises(rcr.RegionalReceiptRefusal) as error:
        rcr.generate_local_contract_receipt(
            paths["grid"], out=tmp_path / "cull.json", dt_seconds=0.5,
            static=paths["static"], deck_receipt=deck, experimental=True,
        )
    assert "other rings" in str(error.value)


def test_a_failing_deck_is_stamped_but_still_refused(tmp_path):
    paths = _write_synthetic_cull(tmp_path)
    ledger = tmp_path / "ledger"
    summary = rcr.generate_local_contract_receipt(
        paths["grid"], out=ledger / "cull.json", dt_seconds=0.5,
        static=paths["static"], init=paths["init"], lbc_dir=tmp_path,
        experimental=True,
        deck_runner=_fake_deck_runner(paths["digest"], verdict=False),
    )
    assert summary["deck_passed"] is False
    stamped = json.loads((ledger / "cull.json").read_text())
    assert stamped["all_decks_bitwise"] is False
    opened = rcr.regional_preflight(
        paths["grid"], dt_seconds=0.5, static=paths["static"],
        init=paths["init"], contract_directories=[ledger], experimental=True,
    )
    assert opened["admitted"] is False
    assert "8 decks bitwise" in opened["refusal"]


def test_the_module_cli_stamps_an_existing_deck_receipt(tmp_path, capsys):
    paths = _write_synthetic_cull(tmp_path)
    deck = tmp_path / "deck.json"
    deck.write_text(json.dumps(_receipt(paths["digest"], 30, "")))
    out = tmp_path / "ledger" / "cull.json"
    argv = [
        "--grid", str(paths["grid"]), "--static", str(paths["static"]),
        "--deck-receipt", str(deck), "--dt-seconds", "0.5", "--out", str(out),
    ]
    assert rcr.main(argv) == 1
    assert "REFUSED" in capsys.readouterr().err
    assert rcr.main([*argv, "--experimental-dt"]) == 0
    assert json.loads(out.read_text())["class_evidence"] == "experimental-unminted"
    preflight = [
        "--grid", str(paths["grid"]), "--static", str(paths["static"]),
        "--dt-seconds", "0.5", "--preflight-only",
    ]
    assert rcr.main(preflight) == 1


def test_the_forecast_door_reports_the_regional_verdict(tmp_path, monkeypatch):
    from woof.hex import forecast_door

    paths = _write_synthetic_cull(tmp_path)
    ledger = tmp_path / "ledger"
    rcr.generate_local_contract_receipt(
        paths["grid"], out=ledger / "cull.json", dt_seconds=0.5,
        static=paths["static"], init=paths["init"], lbc_dir=tmp_path,
        experimental=True, deck_runner=_fake_deck_runner(paths["digest"]),
    )
    monkeypatch.setenv(ra.CONTRACT_LEDGER_ENVIRONMENT, str(ledger))
    request = SimpleNamespace(
        grid=paths["grid"], static=paths["static"], init=paths["init"],
        dt_seconds=0.5,
    )
    closed = forecast_door._regional_admission_block(request)
    assert closed is not None and closed["admitted"] is False
    request.experimental_dt = True
    opened = forecast_door._regional_admission_block(request)
    assert opened["admitted"] is True
    assert opened["class_evidence"] == "experimental-unminted"
    del request.experimental_dt
    monkeypatch.setenv(ra.EXPERIMENTAL_DT_ENVIRONMENT, "1")
    assert forecast_door._regional_admission_block(request)["admitted"] is True
    # A global grid is not asked.
    request.grid = paths["static"]
    assert forecast_door._regional_admission_block(request) is None


# ---------------------------------------------------------------------------
# review follow-ups: claims, kernel sets, receipt choice, columns
# ---------------------------------------------------------------------------


def test_a_minted_claim_that_differs_beyond_the_timestep_is_refused(tmp_path):
    digest = "ab" * 32
    ledger = _ledger(tmp_path, _receipt(digest, 400, "graded-869m-dt5-z7"))
    with pytest.raises(ra.RegionalAdmissionRefusal) as error:
        ra.require_regional_anchor(
            None, bdy_mask_sha256=digest, n_cells=400,
            contract_directories=[ledger], experimental=True, **FINE,
        )
    message = str(error.value)
    assert "besides the timestep" in message
    assert "finest_edge_mm" in message


def test_a_shipped_deck_lapses_on_the_experimental_lane_when_kernels_move():
    contract = _shipped_869()
    klass = ra.ADMITTED_CLASSES[contract.class_id]
    with pytest.raises(ra.RegionalAdmissionRefusal) as error:
        ra.require_regional_anchor(
            contract.mesh_row, bdy_mask_sha256=contract.bdy_mask_sha256,
            n_cells=contract.n_cells,
            boundary_zone_width=klass.key.boundary_zone_width,
            n_vert_levels=klass.key.n_vert_levels,
            finest_edge_m=klass.key.finest_edge_mm / 1000.0, dt_seconds=2.0,
            kernel_set="ff" * 32, experimental=True,
        )
    assert "kernel_set_sha256" in str(error.value)


def test_the_usable_receipt_is_chosen_with_the_lane_closed(tmp_path):
    """An experimental stamp sorting first never shadows a minted receipt."""

    digest = "ab" * 32
    ledger = _ledger(
        tmp_path,
        _receipt(digest, 400, "graded-869m-dt2-z7", class_evidence="experimental-unminted"),
        _receipt(digest, 400, "graded-869m-dt5-z7"),
    )
    anchor = ra.require_regional_anchor(
        None, bdy_mask_sha256=digest, n_cells=400,
        contract_directories=[ledger], boundary_zone_width=7,
        n_vert_levels=55, finest_edge_m=869.251, dt_seconds=5.0,
    )
    assert anchor.class_id == "graded-869m-dt5-z7"
    assert anchor.class_evidence == ra.CLASS_EVIDENCE_MINTED
    assert anchor.contract_receipt.endswith("r1.json")


def test_a_deck_receipt_supplies_its_own_column_count(tmp_path):
    paths = _write_synthetic_cull(tmp_path)
    deck = tmp_path / "deck.json"
    deck.write_text(json.dumps(_receipt(
        paths["digest"], 30, "", mesh={"n_vert_levels": 80},
    )))
    summary = rcr.generate_local_contract_receipt(
        paths["grid"], out=tmp_path / "cull.json", dt_seconds=0.5,
        static=paths["static"], deck_receipt=deck, experimental=True,
    )
    assert summary["class_id"] == "graded-117m-dt0.5-z7-l80"
    # And a deck run on another column than the init declares is refused.
    with pytest.raises(rcr.RegionalReceiptRefusal) as error:
        rcr.generate_local_contract_receipt(
            paths["grid"], out=tmp_path / "cull2.json", dt_seconds=0.5,
            static=paths["static"], init=paths["init"], deck_receipt=deck,
            experimental=True,
        )
    assert "80 levels" in str(error.value)
