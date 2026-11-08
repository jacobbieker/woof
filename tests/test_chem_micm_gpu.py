"""WOOF's MICM port, word for word against the pinned MICM CPU oracle (tools/micm_oracle).

Each fixture is one mechanism and step length: MUSICA's Chapman, CheMPAS-A's Chapman-NOx and a WOOF
mechanism exercising every carried reaction type (Troe, Ea, third bodies, constant species, fractional
yields, repeated reactants, emission, first-order loss, user-defined).  Concentrations, solver state and
step counts must equal the oracle's for every grid point.  With WOOF_MICM_ORACLE pointing at a built
harness the oracle is also run live on fresh cases.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from woof.chem_micm import load_mechanism, solve  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "micm_oracle"
FIXTURES = json.loads((DATA / "fixtures.json").read_text())["fixtures"]


def _hex(rows):
    return np.array([[float.fromhex(x) for x in row] for row in rows]) if rows and isinstance(rows[0], list) \
        else np.array([float.fromhex(x) for x in rows])


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f"{f['mechanism']}-dt{f['time_step']:g}")
def test_word_exact_against_micm(fixture):
    mech = load_mechanism(DATA / fixture["mechanism"])
    conc = _hex(fixture["concentrations"])
    params = _hex(fixture["parameters"]) if mech.parameters else None
    y, status, steps = solve(mech, conc, temperature=_hex(fixture["temperature"]),
                             pressure=_hex(fixture["pressure"]), air_density=_hex(fixture["air_density"]),
                             params=params, time_step=fixture["time_step"])
    y, status, steps = y.get(), status.get(), steps.get()
    expected = _hex(fixture["result"])
    assert status.tolist() == fixture["state"]
    assert steps.tolist() == fixture["steps"]
    for c in range(len(expected)):
        assert y[c].tobytes() == expected[c].tobytes(), (c, [v.hex() for v in y[c]],
                                                          [v.hex() for v in expected[c]])


def test_grid_points_are_independent():
    """MICM's own multi-cell solve shares its step size across cells; here a point's answer is its own."""
    fixture = next(f for f in FIXTURES if f["mechanism"] == "woof_types.json" and f["time_step"] == 600.0)
    mech = load_mechanism(DATA / fixture["mechanism"])
    conc = _hex(fixture["concentrations"])
    args = dict(temperature=_hex(fixture["temperature"]), pressure=_hex(fixture["pressure"]),
                air_density=_hex(fixture["air_density"]), params=_hex(fixture["parameters"]), time_step=600.0)
    whole = solve(mech, conc, **args)[0].get()
    half = slice(0, len(conc) // 2)
    part = solve(mech, conc[half], **{k: (v[half] if isinstance(v, np.ndarray) else v)
                                     for k, v in args.items()})[0].get()
    assert part.tobytes() == whole[half].tobytes()


@pytest.mark.skipif(not os.environ.get("WOOF_MICM_ORACLE"), reason="no built MICM oracle (WOOF_MICM_ORACLE)")
def test_live_oracle_fresh_cases():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "micm_oracle"))
    import generate
    for k, name in enumerate(generate.MECHANISMS):
        mech = load_mechanism(DATA / name)
        T, P, rho, conc, prm = generate.cases(mech, 77 + k)
        _o, _p, results = generate.run(os.environ["WOOF_MICM_ORACLE"], DATA / name, 900.0, mech,
                                       T, P, rho, conc, prm)
        y, status, steps = solve(mech, conc, temperature=T, pressure=P, air_density=rho,
                                 params=prm if mech.parameters else None, time_step=900.0)
        y = y.get()
        for c, r in enumerate(results):
            assert y[c].tobytes() == np.array([float.fromhex(r["conc"][s]) for s in mech.species]).tobytes()
