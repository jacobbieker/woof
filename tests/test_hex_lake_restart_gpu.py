"""A hex restart restores the lake state from the payload; it never re-fetches it.

``woof.hex.cuda_arwen_physics_v841`` rebuilds its seam on a restart and on a
rollback, then overwrites the seam from the payload.  Building that seam used to
bind the observed lake state again (a network read for state the restore throws
away).  Seams built inside ``restoring_from_checkpoint`` read nothing, and a
restarted run continues bitwise with an unbroken one.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import mpas_column_physics_cases as cases  # noqa: E402
import test_hex_surface_inputs_gpu as base  # noqa: E402
import woof.ingest.lake_state as lake_state  # noqa: E402

DT = cases.DT


def _seam(monkeypatch, calls, *, restoring=False):
    water = np.array([281.25, 279.5, 283.0])

    def load(source, valid_time, lat, lon, **kwargs):
        calls.append(source)
        return (dict(water_temperature=water[:len(lat)]),
                dict(schema="gpuwm.lake-init.v1", status="bound", source=source))

    monkeypatch.setattr(lake_state, "load_state", load)
    physics = dict(base._physics(), lake_state_source="era5-land")
    kwargs = base._seal(base._mapping(physics, base._inputs())).arwen_kwargs()
    from woof.core import mpas_column_suite as suite

    if restoring:
        with lake_state.restoring_from_checkpoint():
            return suite.run_mpas_column_physics(**kwargs)
    return suite.run_mpas_column_physics(**kwargs)


def _step(seam, fields, species, theta):
    seam.run_phase1(
        dt=DT, u=fields["u"], v=fields["v"], theta=theta,
        pressure=fields["pressure"], pressure_interface=fields["pressure_interface"],
        z_interface=fields["z_interface"], w=fields["w"], rho_dry=fields["rho_dry"],
        species=species)
    seam.run_phase2(
        theta=theta, pressure=fields["pressure"], z_interface=fields["z_interface"],
        w=fields["w"], rho_dry=fields["rho_dry"], species=species)


def _words(seam, theta):
    from woof.core.lake_schema import LAKE_OUTPUT_LAYOUT

    names = ("lake_columns", "sst", "tsk", "xice",
             *(name for name, _ in LAKE_OUTPUT_LAYOUT))
    return {"theta": base._host(theta),
            **{name: base._host(seam._driver.fields[name]) for name in names}}


def test_a_restart_binds_no_lake_state_and_continues_bitwise(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOF_LAKE_RECEIPT_DIR", str(tmp_path))
    calls = []
    reference_seam = _seam(monkeypatch, calls)
    assert calls == ["era5-land"], "the unbroken run binds the source once"
    fields, species = cases.profile(cp, reference_seam.plan)
    theta = fields["theta"].copy()
    reference = []
    for _ in range(4):
        _step(reference_seam, fields, species, theta)
        reference.append(({n: base._host(v) for n, v in species.items()},
                          _words(reference_seam, theta)))

    first = _seam(monkeypatch, calls)
    cases.run(cp, first, 2)
    payload = first.export_state()
    fetches = len(calls)
    second = _seam(monkeypatch, calls, restoring=True)
    assert len(calls) == fetches, "building a seam to restore into re-fetched the lake state"
    assert second._driver.lake_initialization_receipt["status"] == "restored_from_checkpoint"
    second.restore_state(payload)
    fields, _ = cases.profile(cp, second.plan)
    theta = cp.asarray(reference[1][1]["theta"])
    species = {n: cp.asarray(reference[1][0][n]) for n in second.plan.species}
    for step in (2, 3):
        _step(second, fields, species, theta)
        for name, value in _words(second, theta).items():
            np.testing.assert_array_equal(
                value.view(np.uint32), reference[step][1][name].view(np.uint32),
                err_msg=f"step {step} {name}")


def test_an_ordinary_build_still_binds_the_source(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOF_LAKE_RECEIPT_DIR", str(tmp_path))
    calls = []
    ordinary = _seam(monkeypatch, calls)
    restored = _seam(monkeypatch, calls, restoring=True)
    assert calls == ["era5-land"]
    assert ordinary._driver.lake_initialization_receipt["status"] == "bound"
    assert restored._driver.lake_initialization_receipt["status"] == "restored_from_checkpoint"
