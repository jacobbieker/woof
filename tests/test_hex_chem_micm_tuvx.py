"""MICM on hex with TUV-x photolysis (woof.hex.chem_micm_hex, ``"photolysis": "tuvx"``).

The switch is opt-in: the default block keeps CheMPAS-A's ``j_max * max(0, cos SZA)`` (tests/
test_hex_chem_micm_gpu.py, unchanged).  With ``"photolysis": "tuvx"`` each MICM photolysis rate names a reaction of
a TUV-x configuration; the rate parameters a step hands MICM must be, word for word, woof.chem_tuvx's rates on the
step's columns (completed above the model top, chem_micm_hex.tuvx_columns) at the layer mid-points, whatever the
column batching.  The TUV-x port itself is graded against the oracle in tests/test_chem_tuvx_*.py.
"""
import datetime
import json
import os
from pathlib import Path

import numpy as np
import pytest

from woof import chem_tuvx, chem_tuvx_data
from woof.hex import chem_micm_hex

ROOT = Path(__file__).resolve().parents[1]
MECHANISM = ROOT / "tests" / "data" / "micm_oracle" / "chempas_a_chapman_nox.yaml"
TUVX_CONFIG = ROOT / "tools" / "tuvx_oracle" / "photolysis.json"
REACTIONS = {"jO2": "O2+hv->O+O", "jO3_O": "O3+hv->O2+O(3P)", "jO3_O1D": "O3+hv->O2+O(1D)",
             "jNO2": "NO2+hv->NO+O(3P)"}
START = datetime.datetime(2021, 6, 21, 10, 0)
NZ, NC, DT = 6, 9, 60.0


def _raw(**block):
    micm = {"micm_file": str(MECHANISM), "photolysis": "tuvx",
            "tuvx": {"config": str(TUVX_CONFIG), "reactions": REACTIONS},
            "initial_mass_mixing_ratio": {"O2": 0.231, "O3": 1e-6, "NO": 2e-9, "NO2": 5e-9}}
    micm.update(block)
    return {"schema": "gpuwm-hex.chemistry/v1", "micm": micm}


def _write(tmp_path, raw):
    path = tmp_path / "chemistry.json"
    path.write_text(json.dumps(raw))
    return path


def test_the_switch_maps_each_rate_to_a_tuvx_reaction(tmp_path):
    raw = _raw()
    _mech, _rows, settings = chem_micm_hex.load(_write(tmp_path, raw), raw)
    assert settings["photolysis"] == {f"PHOTO.{k}": v for k, v in REACTIONS.items()}
    assert settings["tuvx"]["config"] == str(TUVX_CONFIG.resolve()) and settings["tuvx"]["data"] is None
    meta = json.loads(chem_micm_hex.history_metadata(_write(tmp_path, raw), raw, "0" * 64))
    assert meta["photolysis"].startswith("TUV-x v0.17.0")


@pytest.mark.parametrize("change,match", [
    ({"tuvx": {"config": str(TUVX_CONFIG), "reactions": {**REACTIONS, "jNO2": "HNO3+hv->OH+NO2"}}}, "not reactions"),
    ({"tuvx": {"config": str(TUVX_CONFIG), "reactions": {k: v for k, v in REACTIONS.items() if k != "jO2"}}},
     "jO2"),
    ({"tuvx": {"reactions": REACTIONS}}, "config and reactions"),
    ({"photolysis": {k: {"j_max": 1e-3} for k in REACTIONS}}, "only with"),
])
def test_bad_switches_are_refused(tmp_path, change, match):
    raw = _raw(**change)
    with pytest.raises(ValueError, match=match):
        chem_micm_hex.load(_write(tmp_path, raw), raw)


def test_the_columns_reach_120_km():
    z = np.linspace(0.0, 30000.0, NZ + 1)[:, None] * np.ones(3)
    z[:, 1] += np.linspace(500.0, 0.0, NZ + 1)                     # terrain under a flat top
    t = np.full((NZ, 3), 250.0)
    p = np.full((NZ, 3), 5.0e4)
    o3 = np.full((NZ, 3), 1.0e-5)
    edges, tedge, air, o2, o3l = chem_micm_hex.tuvx_columns(z, t, p, o3)
    assert edges.shape == tedge.shape and air.shape == o2.shape == o3l.shape == (3, edges.shape[1] - 1)
    assert np.all(np.diff(edges, axis=1) > 0) and np.allclose(edges[:, -1], 120.0)
    assert np.all(np.diff(edges[:, NZ:], axis=1) <= chem_micm_hex.TUVX_EXTENSION_KM + 1e-9)
    assert np.array_equal(edges[:, :NZ + 1], z.T / 1000.0)
    assert np.array_equal(o2, chem_micm_hex.O2_FRACTION * air)


def _data_root():
    root = os.environ.get(chem_tuvx_data.ENV_ROOT) or chem_tuvx_data.cache_root()
    if not all((Path(root) / f).is_file() for f in chem_tuvx_data.FILES):
        return None
    return str(root)


def test_a_step_hands_micm_the_ports_rates(tmp_path, monkeypatch):
    cp = pytest.importorskip("cupy")
    root = _data_root()
    if root is None:
        pytest.skip("the pinned TUV-x data are not cached here (woof.chem_tuvx_data.ensure() fetches them)")
    raw = _raw(tuvx={"config": str(TUVX_CONFIG), "reactions": REACTIONS, "data": root})
    path = _write(tmp_path, raw)
    rt = chem_micm_hex.HexMicmChemistry(path, raw, "0" * 64, nlev=NZ, ncells=NC, offset=0, start_time=START,
                                        dt=DT, lat_deg=np.linspace(-60, 70, NC), lon_deg=np.linspace(-170, 170, NC))
    rng = np.random.default_rng(5)
    z = np.linspace(0, 20000, NZ + 1)[:, None] * np.ones(NC)
    p = 101325 * np.exp(-0.5 * (z[1:] + z[:-1]) / 8000) * (1 + 0.01 * rng.standard_normal((NZ, NC)))
    T = 290 - 6.5e-3 * 0.5 * (z[1:] + z[:-1]) + rng.standard_normal((NZ, NC))
    cols = {"temperature": cp.asarray(T), "pressure": cp.asarray(p), "rho_dry": cp.asarray(p / (287.0 * T)),
            "z_interface": cp.asarray(z)}
    conc = cp.asarray(rng.uniform(1e-6, 1e-4, (NZ * NC, len(rt.rows))))
    monkeypatch.setattr(chem_micm_hex, "TUVX_BATCH", 4)             # batches of 4, 4 and 1 columns
    prm = cp.asnumpy(rt._params(DT, DT, columns=cols, conc=conc))
    # the reference: one batch of every column
    when = START
    cosz = cp.asnumpy(chem_micm_hex.cos_sza(rt.lat, rt.lon, when.timetuple().tm_yday, 10.0))
    sza = np.degrees(np.arccos(np.clip(cosz, -1.0, 1.0)))
    o3 = cp.asnumpy(conc)[:, rt.mech.species.index("O3")].reshape(NZ, NC)
    inputs = chem_micm_hex.tuvx_columns(z, T, p, o3)
    core = chem_tuvx.Tuvx(json.loads(TUVX_CONFIG.read_text()), root=root)
    rates = core.run(*inputs, sza, chem_tuvx.earth_sun_distance(2021, 6, 21, 10.0))
    for i, label in enumerate(rt.mech.parameters):
        if not label.startswith("PHOTO."):
            continue
        edge = rates[REACTIONS[label.split(".", 1)[1]]]
        want = (0.5 * (edge[:, :NZ] + edge[:, 1:NZ + 1])).T.reshape(-1)
        assert prm[:, i].tobytes() == want.tobytes(), label
    lit = cosz > 0.1
    j_no2 = prm[:, rt.mech.parameters.index("PHOTO.jNO2")].reshape(NZ, NC)
    assert np.all(j_no2[:, lit] > 1e-3) and np.all(j_no2[:, cosz < -0.3] < 1e-8)
