"""Device check: the hex physics seam exports SWDDNI/SWDDIF/COSZR.

The hex history's ``swddni``/``swddif``/``coszr`` are selected by the
adapter (``woof.hex.cuda_arwen_physics_v841._ARWEN_EXPORT_KEYS``) from the
column-batch seam's ``export_state()`` under ``diag/``.  This runs the real
seam (legacy RRTMG shortwave, the hex physics set) on a few daytime columns
and checks those keys are present once radiation has run, with physical
values: the adapter would otherwise publish nothing, silently, because the
fields are optional.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from woof.core import mpas_column_batch as mcb  # noqa: E402
from woof.hex import cuda_arwen_physics_v841 as adapter  # noqa: E402

NZ, NCOL = 20, 4
DT = 120.0
# 18 UTC at 97 W is local early afternoon in June: the sun is up.
START = datetime.datetime(2021, 6, 1, 18, 0)


def _inputs():
    z_iface = np.linspace(0.0, 16000.0, NZ + 1)
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    p_iface = 101325.0 * np.exp(-z_iface / 8000.0)
    p_mid = 101325.0 * np.exp(-z_mid / 8000.0)
    theta = 300.0 + 25.0 * z_mid / 16000.0
    temp = theta * (p_mid / 1.0e5) ** (287.0 / 1004.0)

    def cols(profile):
        return cp.asarray(np.ascontiguousarray(
            np.repeat(np.asarray(profile)[:, None], NCOL, axis=1),
            dtype=np.float32))

    zeros = np.zeros(NZ)
    return {
        "u": cols(np.full(NZ, 5.0)), "v": cols(np.full(NZ, -3.0)),
        "theta": cols(theta), "pressure": cols(p_mid),
        "pressure_interface": cols(p_iface), "z_interface": cols(z_iface),
        "w": cols(np.zeros(NZ + 1)), "rho_dry": cols(p_mid / (287.0 * temp)),
        "qv": cols(0.010 * np.exp(-z_mid / 3000.0)), "qc": cols(zeros),
        "qr": cols(zeros), "qi": cols(zeros), "qs": cols(zeros),
        "qg": cols(zeros),
    }


def test_the_seam_exports_the_radiation_history_buffers():
    free, _ = cp.cuda.runtime.memGetInfo()
    if free < 2 * 1024**3:
        pytest.skip("the shared card has under 2 GiB free")
    seam = mcb.run_mpas_column_batch(
        n_levels=NZ, n_columns=NCOL, dt=DT,
        radiation_seconds=600.0, surface_pbl_seconds=120.0,
        cumulus_seconds=600.0, cumulus_scheme="kf", start_time=START,
        latitude_deg=np.full(NCOL, 35.0),
        longitude_deg=np.full(NCOL, -97.0),
        terrain_height_m=np.zeros(NCOL),
        z_interface_nominal_m=np.linspace(0.0, 16000.0, NZ + 1),
        p_top_pa=float(101325.0 * np.exp(-2.0)), dx_m=15000.0)
    inputs = _inputs()
    state2 = {name: inputs[name].copy()
              for name in ("theta", "qv", "qc", "qr", "qi", "qs", "qg",
                           "pressure", "rho_dry", "z_interface")}
    result = seam.run_phase1(dt=DT, **inputs)
    assert result.radiation_ran
    seam.run_phase2(**state2)
    arrays = seam.export_state()["arrays"]
    values = {}
    for name in ("swddni", "swddif", "coszr"):
        key = adapter._ARWEN_EXPORT_KEYS[name]
        assert key in arrays, f"{key} missing from the seam export"
        value = np.asarray(arrays[key])
        assert value.dtype == np.float32
        # The adapter accepts (1, ncol); anything else would refuse a frame.
        assert value.shape == (1, NCOL)
        values[name] = value[0]
    assert np.all((values["coszr"] > 0.0) & (values["coszr"] <= 1.0))
    assert np.all(values["swddni"] > 0.0)
    assert np.all(values["swddif"] > 0.0)
    swdown = np.asarray(arrays["fields/swdown"]).reshape(-1)
    # Direct horizontal plus diffuse is the surface downward shortwave.
    np.testing.assert_allclose(
        values["swddni"] * values["coszr"] + values["swddif"], swdown,
        rtol=2e-3)
