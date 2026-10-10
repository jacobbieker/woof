"""The column-batch seam with its PBL slot off (``pbl_scheme="off"``).

Runs the REAL chain on synthetic moist columns (legacy RRTMG, revised-MO
surface layer, Noah-MP, WSM6; no cumulus scheme) and gates what PBL-off
promises: YSU is never called, the surface layer and Noah-MP still run on
the surface/PBL cadence and publish their fluxes, the PBL momentum rates
stay exactly zero, ``surface_pbl_ran`` reports the surface call, every
published rate is finite, and the seam identity names the slot so a restart
across it refuses.
"""

import datetime

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from woof.core import mpas_column_batch as mcb  # noqa: E402

NZ, NCOL = 24, 8
DT = 20.0
START = datetime.datetime(2021, 6, 1, 18, 0)
STEPS = 3


def _profile(seed=0):
    rng = np.random.default_rng(seed)
    z_iface = np.linspace(0.0, 16000.0, NZ + 1)
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    p_iface = 101325.0 * np.exp(-z_iface / 8000.0)
    p_mid = 101325.0 * np.exp(-z_mid / 8000.0)
    theta = 300.0 + 25.0 * z_mid / 16000.0
    exner = (p_mid / 1.0e5) ** (287.0 / 1004.0)
    rho_dry = p_mid / (287.0 * theta * exner)
    qv = 0.012 * np.exp(-z_mid / 3000.0)

    def cols(profile, jitter=1.0e-3):
        base = np.repeat(profile[:, None], NCOL, axis=1)
        base *= 1.0 + jitter * rng.standard_normal(base.shape)
        return np.ascontiguousarray(base, dtype=np.float32)

    fields = {
        "u": cols(np.full(NZ, 6.0)),
        "v": cols(np.full(NZ, -2.0)),
        "theta": cols(theta),
        "pressure": cols(p_mid, jitter=0.0),
        "pressure_interface": cols(p_iface, jitter=0.0),
        "z_interface": np.ascontiguousarray(
            np.repeat(z_iface[:, None], NCOL, axis=1), dtype=np.float32),
        "w": np.zeros((NZ + 1, NCOL), dtype=np.float32),
        "rho_dry": cols(rho_dry, jitter=0.0),
        "qv": cols(qv),
        "qc": np.zeros((NZ, NCOL), dtype=np.float32),
        "qr": np.zeros((NZ, NCOL), dtype=np.float32),
        "qi": np.zeros((NZ, NCOL), dtype=np.float32),
        "qs": np.zeros((NZ, NCOL), dtype=np.float32),
        "qg": np.zeros((NZ, NCOL), dtype=np.float32),
    }
    return {name: cp.asarray(value) for name, value in fields.items()}


def _seam(pbl_scheme):
    return mcb.run_mpas_column_batch(
        n_levels=NZ, n_columns=NCOL, dt=DT,
        radiation_seconds=600.0, surface_pbl_seconds=DT,
        cumulus_seconds=None, cumulus_scheme=None,
        start_time=START,
        latitude_deg=np.full(NCOL, 35.0),
        longitude_deg=np.full(NCOL, -97.0),
        terrain_height_m=np.zeros(NCOL),
        z_interface_nominal_m=np.linspace(0.0, 16000.0, NZ + 1),
        p_top_pa=float(101325.0 * np.exp(-2.0)), dx_m=100.0,
        pbl_scheme=pbl_scheme)


def _run(pbl_scheme):
    seam = _seam(pbl_scheme)
    inputs = _profile()
    state2 = {name: inputs[name].copy()
              for name in ("theta", "qv", "qc", "qr", "qi", "qs", "qg",
                           "pressure", "rho_dry", "z_interface")}
    results = []
    for _ in range(STEPS):
        result = seam.run_phase1(dt=DT, **inputs)
        results.append({
            "flags": (result.radiation_ran, result.surface_pbl_ran),
            "arrays": {name: cp.asnumpy(getattr(result, name)).copy()
                       for name in mcb._OUTPUT_BUFFERS},
        })
        seam.run_phase2(**state2)
    return seam, results


@pytest.fixture(scope="module")
def pbl_off():
    return _run("off")


def test_pbl_off_never_calls_ysu_and_still_runs_the_surface(pbl_off):
    seam, results = pbl_off
    counts = seam.call_counts
    assert counts["ysu"] == 0
    assert counts["sfclay"] == STEPS
    assert counts["noah"] == STEPS
    assert seam._driver.last_ysu is None
    assert all(step["flags"][1] for step in results), "surface_pbl_ran"


def test_pbl_off_publishes_zero_pbl_momentum_and_finite_rates(pbl_off):
    _, results = pbl_off
    for step in results:
        for name, array in step["arrays"].items():
            assert np.all(np.isfinite(array)), name
        assert not np.any(step["arrays"]["du"])
        assert not np.any(step["arrays"]["dv"])
    # Radiation still heats or cools the column: the run is not inert.
    assert np.any(results[0]["arrays"]["dtheta"])


def test_pbl_off_surface_fluxes_are_published(pbl_off):
    seam, _ = pbl_off
    fields = seam._driver.fields
    hfx = cp.asnumpy(fields["hfx"])
    assert np.all(np.isfinite(hfx))
    assert np.any(hfx != 0.0)


def test_pbl_off_names_itself_in_the_identity(pbl_off):
    seam, _ = pbl_off
    assert seam._identity["pbl_scheme"] == "off"
    ysu = _seam("ysu")
    assert "pbl_scheme" not in ysu._identity
