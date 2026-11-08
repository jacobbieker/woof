"""MICM on hex grid points (woof.hex.chem_micm_hex): CheMPAS-A's conversions around WOOF's MICM port.

The hex step must be, word for word, the per-point solve of woof.chem_micm (itself graded word-exact
against the pinned MICM oracle) on CheMPAS-A's conversions -- concentration max(q, 0) rho_dry / M, air
density P / (R T), photolysis j_max max(0, cos SZA) -- for every level of every cell, independent of the
batch; the chemistry interval accumulates whole steps; abort restores the accumulator.
"""
import datetime
import json
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from woof.chem_micm import solve  # noqa: E402
from woof.hex import chem_micm_hex  # noqa: E402

MECHANISM = Path(__file__).resolve().parent / "data" / "micm_oracle" / "chempas_a_chapman_nox.yaml"
START = datetime.datetime(2021, 6, 21, 10, 0)
NZ, NC, DT = 6, 9, 60.0
PHOTO = {"jO2": {"j_max": 1e-11}, "jO3_O": {"j_max": 4e-4}, "jO3_O1D": {"j_max": 3e-5},
         "jNO2": {"j_max": 8e-3}}


def _runtime(tmp_path, **block):
    path = tmp_path / "chemistry.json"
    raw = {"schema": "gpuwm-hex.chemistry/v1",
           "micm": {"micm_file": str(MECHANISM), "photolysis": PHOTO,
                    "initial_mass_mixing_ratio": {"O2": 0.231, "O3": 1e-6, "NO": 2e-9, "NO2": 5e-9}, **block}}
    path.write_text(json.dumps(raw))
    rt = chem_micm_hex.HexMicmChemistry(path, raw, "0" * 64, nlev=NZ, ncells=NC, offset=0, start_time=START,
                                        dt=DT, lat_deg=np.linspace(-60, 70, NC), lon_deg=np.linspace(-170, 170, NC))
    scalars = cp.zeros((len(rt.rows), NZ, NC), cp.float32)
    chem_micm_hex.seed(rt, scalars)
    return rt, scalars


def _columns():
    rng = np.random.default_rng(5)
    z = np.linspace(0, 12000, NZ + 1)[:, None] * np.ones(NC)
    p = 101325 * np.exp(-0.5 * (z[1:] + z[:-1]) / 8000) * (1 + 0.01 * rng.standard_normal((NZ, NC)))
    T = 290 - 6.5e-3 * 0.5 * (z[1:] + z[:-1]) + rng.standard_normal((NZ, NC))
    rho = p / (287.0 * T)
    f = lambda a: cp.asarray(a, cp.float32)
    return {"temperature": f(T), "pressure": f(p), "rho_dry": f(rho), "z_interface": f(z)}


def _reference(rt, scalars, columns, elapsed, chem_dt):
    rho = columns["rho_dry"].astype(cp.float64).reshape(-1)
    T = columns["temperature"].astype(cp.float64).reshape(-1)
    P = columns["pressure"].astype(cp.float64).reshape(-1)
    M = cp.asarray([r.molar_mass for r in rt.rows])
    q = cp.maximum(scalars.astype(cp.float64).reshape(len(rt.rows), -1), 0.0)
    conc = cp.ascontiguousarray((q * rho / M[:, None]).T)
    when = START + datetime.timedelta(seconds=elapsed - chem_dt)
    hour = when.hour + when.minute / 60 + when.second / 3600
    cosz = cp.maximum(chem_micm_hex.cos_sza(rt.lat, rt.lon, when.timetuple().tm_yday, hour), 0.0)
    prm = cp.stack([PHOTO[label.split(".", 1)[1]]["j_max"] * cp.broadcast_to(cosz, (NZ, NC)).reshape(-1)
                    for label in rt.mech.parameters], axis=1)
    out, status, _ = solve(rt.mech, conc, temperature=T, pressure=P, air_density=P / (8.314462618 * T),
                           params=prm, time_step=chem_dt)
    assert int(cp.count_nonzero(status != 2)) == 0
    return (out.T * M[:, None] / rho).reshape(len(rt.rows), NZ, NC).astype(cp.float32)


def test_hex_step_is_the_per_point_solve(tmp_path):
    rt, scalars = _runtime(tmp_path)
    cols = _columns()
    for step in range(1, 4):
        expected = _reference(rt, scalars, cols, step * DT, DT)
        rt.step(scalars, cols, dt=DT, elapsed_seconds=step * DT)
        rt.commit()
        assert cp.asnumpy(scalars).tobytes() == cp.asnumpy(expected).tobytes()
    assert float(cp.asnumpy(scalars[rt.mech.species.index("O")]).max()) > 0, "no photolysis happened"


def test_points_are_independent_of_the_batch(tmp_path):
    rt, scalars = _runtime(tmp_path)
    cols = _columns()
    rt.step(scalars, cols, dt=DT, elapsed_seconds=DT)
    half, hs = _runtime(tmp_path)
    half.ncells = NC  # same geometry; solve a copy of the columns with one cell perturbed elsewhere
    cols2 = {k: v.copy() for k, v in cols.items()}
    cols2["temperature"][:, 0] += 5.0
    half.step(hs, cols2, dt=DT, elapsed_seconds=DT)
    assert cp.asnumpy(scalars[:, :, 1:]).tobytes() == cp.asnumpy(hs[:, :, 1:]).tobytes()


def test_interval_accumulates_and_abort_restores(tmp_path):
    rt, scalars = _runtime(tmp_path, chemistry_interval=3 * DT)
    cols = _columns()
    before = scalars.copy()
    rt.step(scalars, cols, dt=DT, elapsed_seconds=DT)
    rt.commit()
    rt.step(scalars, cols, dt=DT, elapsed_seconds=2 * DT)
    rt.abort()
    rt.step(scalars, cols, dt=DT, elapsed_seconds=2 * DT)
    rt.commit()
    assert cp.asnumpy(scalars).tobytes() == cp.asnumpy(before).tobytes()
    expected = _reference(rt, scalars, cols, 3 * DT, 3 * DT)
    rt.step(scalars, cols, dt=DT, elapsed_seconds=3 * DT)
    rt.commit()
    assert cp.asnumpy(scalars).tobytes() == cp.asnumpy(expected).tobytes()
