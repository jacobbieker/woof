"""CCPP TKE-EDMF on the hex generic column seam IS the WRF path's column.

Hex parity item 1.9.  ``bl_pbl_physics = 901`` reaches a hex forecast through
the engine's generic column seam (``run_mpas_column_physics``), which builds
the WRF path's own ``PhysicsDriver`` and so calls the same
``PhysicsDriver._run_satmedmf`` and the same pinned CCPP column
(``woof.core.satmedmf.run_device``).  This file proves, at both the every-step
cadence and a held PBL cadence:

* the seam returns the PBL's momentum rates at the every-step cadence too
  (it used to spell that cadence bldt = 0, where the driver keeps no raw
  momentum rates, and handed back zero du/dv).
* E1: every CCPP call the seam makes is replayed one column at a time through
  the WRF column kernel on the same input bytes, and each column agrees bit
  for bit; the TKE rate the seam hands its caller is the WRF driver's
  conversion of that call (``recover_physics_rates``), on call steps and on
  the held steps between them.
* E4: the coupled TKE (and scalar) rate bank is on the restart manifest
  (``driver/pbl_tendencies/extra_scalars/*``); a restore continues bit for
  bit, and at a held cadence (restored in the middle of a hold) zeroing the
  bank changes the next step, so the bank is load-bearing.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import mpas_column_physics_cases as cases  # noqa: E402
from woof.core import mpas_column_suite as suite  # noqa: E402
from woof.core import satmedmf  # noqa: E402

NAME = "pbl.ccpp-tke-edmf"
NZ, NCOL, DT = cases.NZ, cases.NCOL, cases.DT
HELD = 2.0 * DT
BANK = "driver/pbl_tendencies/extra_scalars/"
RATES = ("du", "dv", "dtheta", "dqv", "dqc", "dqr", "dqi", "dqs", "dqg",
         "h_diabatic")


def _physics():
    return cases.selections()[NAME]


def _build(pbl_seconds):
    physics = _physics()
    plan = suite.column_physics_plan(
        physics, n_levels=NZ, dt=DT, radiation_seconds=120.0,
        surface_pbl_seconds=pbl_seconds)
    rng = np.random.default_rng(2)
    return suite.run_mpas_column_physics(
        physics=physics, n_levels=NZ, n_columns=NCOL, dt=DT,
        radiation_seconds=120.0, surface_pbl_seconds=pbl_seconds,
        start_time=cases.START,
        latitude_deg=np.full(NCOL, 38.0), longitude_deg=np.full(NCOL, -100.0),
        terrain_height_m=rng.uniform(0.0, 50.0, NCOL).astype(np.float32),
        z_interface_nominal_m=np.linspace(0.0, cases.ZTOP, NZ + 1),
        p_top_pa=cases.P_TOP, dx_m=12000.0,
        cell_area_m2=np.full(NCOL, 1.44e8),
        surface=cases.surface(plan), statics=cases.statics(plan))


class _Caller:
    """The dycore's side of the seam: carries theta and species across
    steps and applies the TKE rate it is handed (a forward step, clipped at
    zero), so the TKE the column sees evolves."""

    def __init__(self, seam):
        self.fields, self.species = cases.profile(cp, seam.plan)
        z = np.linspace(0.0, cases.ZTOP, NZ + 1)
        z_mid = 0.5 * (z[:-1] + z[1:])
        tke = 0.05 + 0.6 * np.exp(-z_mid / 1200.0)
        self.species["tke_edmf"] = cp.asarray(np.ascontiguousarray(
            np.repeat(tke[:, None], NCOL, 1), dtype=np.float32))
        self.theta = self.fields["theta"].copy()

    def snapshot(self):
        return (self.theta.copy(),
                {name: value.copy() for name, value in self.species.items()})

    def restore(self, snap):
        self.theta = snap[0].copy()
        self.species = {name: value.copy() for name, value in snap[1].items()}

    def step(self, seam):
        f = self.fields
        result = seam.run_phase1(
            dt=DT, u=f["u"], v=f["v"], theta=self.theta,
            pressure=f["pressure"], pressure_interface=f["pressure_interface"],
            z_interface=f["z_interface"], w=f["w"], rho_dry=f["rho_dry"],
            species=self.species)
        frame = {name: cp.asnumpy(getattr(result, name)).copy()
                 for name in RATES}
        frame["scalar_rates"] = {name: cp.asnumpy(value).copy()
                                 for name, value in result.scalar_rates.items()}
        frame["pbl_ran"] = bool(result.surface_pbl_ran)
        tke = self.species["tke_edmf"]
        rate = result.scalar_rates["tke_edmf"]
        cp.maximum(tke + cp.float32(DT) * rate, cp.float32(0.0), out=tke)
        seam.run_phase2(
            theta=self.theta, pressure=f["pressure"],
            z_interface=f["z_interface"], w=f["w"], rho_dry=f["rho_dry"],
            species=self.species)
        frame["theta"] = cp.asnumpy(self.theta).copy()
        frame["species"] = {name: cp.asnumpy(value).copy()
                            for name, value in self.species.items()}
        return frame


@pytest.fixture
def recorded(monkeypatch):
    """Every CCPP column call: (host inputs, host outputs, dt)."""
    calls = []
    original = satmedmf.run_device

    def recording(inputs, *, dt=60.0, strict=None):
        out = original(inputs, dt=dt, strict=strict)
        calls.append(({k: cp.asnumpy(v).copy() for k, v in inputs.items()},
                      {k: cp.asnumpy(v).copy() for k, v in out.items()},
                      float(dt)))
        return out

    monkeypatch.setattr(satmedmf, "run_device", recording)
    return calls


@pytest.mark.parametrize("pbl_seconds", [DT, HELD], ids=["every-step", "held"])
def test_each_column_is_the_wrf_column_kernel_on_the_same_bytes(
        recorded, pbl_seconds):
    seam = _build(pbl_seconds)
    caller = _Caller(seam)
    steps = 5
    latest = []
    for _ in range(steps):
        frame = caller.step(seam)
        latest.append((len(recorded), frame))
    # The replays below call run_device too; judge the seam's calls only.
    calls = list(recorded)
    assert seam.call_counts["ysu"] == len(calls)
    assert len(calls) == (steps if pbl_seconds == DT else 3)
    assert [f["pbl_ran"] for _, f in latest] == (
        [True] * steps if pbl_seconds == DT
        else [True, True, False, True, False])
    for inputs, out, dt in calls:
        # WRF's dtbl: the model step, or the held interval.
        assert dt == pbl_seconds
        for column in range(NCOL):
            alone = satmedmf.run_columns(
                {k: np.asfortranarray(v[column:column + 1])
                 for k, v in inputs.items()}, dt=dt)
            for key, value in out.items():
                np.testing.assert_array_equal(
                    alone[key], value[column:column + 1],
                    f"column {column} {key}")
    for count, frame in latest:
        inputs, out, dt = calls[count - 1]
        device_in = {k: cp.asarray(v, order="F") for k, v in inputs.items()}
        device_out = {k: cp.asarray(v, order="F") for k, v in out.items()}
        rates = satmedmf.recover_physics_rates(
            device_in, device_out, (NZ, 1, NCOL), dt,
            scalar_names=satmedmf.column_scalar_names(seam._state))
        # The seam's column state has chm == 1, so the coupled bank IS the
        # raw rate; on a held step it is the last call's rate.
        np.testing.assert_array_equal(
            frame["scalar_rates"]["tke_edmf"],
            cp.asnumpy(rates["dtke_edmf"]).reshape(NZ, NCOL))
        for name in ("du", "dv"):
            np.testing.assert_array_equal(
                frame[name], cp.asnumpy(rates[name]).reshape(NZ, NCOL),
                name)
        assert np.any(frame["scalar_rates"]["tke_edmf"] != 0.0)


@pytest.mark.parametrize("pbl_seconds", [DT, HELD], ids=["every-step", "held"])
def test_rate_bank_restarts_bit_for_bit(pbl_seconds):
    reference = _build(pbl_seconds)
    caller = _Caller(reference)
    frames = [caller.step(reference) for _ in range(6)]

    first = _build(pbl_seconds)
    resumed = _Caller(first)
    for _ in range(2):
        resumed.step(first)
    payload = first.export_state()
    snap = resumed.snapshot()
    held = sorted(key[len(BANK):] for key in payload["arrays"]
                  if key.startswith(BANK))
    assert held == sorted(("tke_edmf",)
                          + satmedmf.column_scalar_names(first._state))
    assert np.any(payload["arrays"][BANK + "tke_edmf"] != 0.0)

    second = _build(pbl_seconds)
    second.restore_state(payload)
    resumed.restore(snap)
    for step in range(2, 6):
        frame = resumed.step(second)
        assert frame["pbl_ran"] == frames[step]["pbl_ran"]
        for name in RATES:
            np.testing.assert_array_equal(
                frame[name], frames[step][name], f"step {step} {name}")
        for name, value in frames[step]["scalar_rates"].items():
            np.testing.assert_array_equal(
                frame["scalar_rates"][name], value, f"step {step} {name}")
        np.testing.assert_array_equal(frame["theta"], frames[step]["theta"])
        for name, value in frames[step]["species"].items():
            np.testing.assert_array_equal(
                frame["species"][name], value, f"step {step} {name}")

    if pbl_seconds == DT:
        return
    # The bank is load-bearing: the step after the restart is a held step,
    # and a payload without the held TKE rate resumes differently.
    assert not frames[2]["pbl_ran"]
    dropped = dict(payload, arrays=dict(payload["arrays"]))
    dropped["arrays"][BANK + "tke_edmf"] = np.zeros_like(
        payload["arrays"][BANK + "tke_edmf"])
    third = _build(pbl_seconds)
    third.restore_state(dropped)
    resumed.restore(snap)
    frame = resumed.step(third)
    assert not np.array_equal(frame["scalar_rates"]["tke_edmf"],
                              frames[2]["scalar_rates"]["tke_edmf"])
