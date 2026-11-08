"""Write the MICM oracle fixtures tests/data/micm_oracle/fixtures.json from the built harness.

For each committed mechanism and step length: deterministic random conditions, concentrations and rate
parameters, the harness's species and parameter order, and its result words (hex), state and step count.
See build.sh for the oracle.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data" / "micm_oracle"
MECHANISMS = ("musica_chapman.json", "chempas_a_chapman_nox.yaml", "woof_types.json")
STEPS = (1.0, 600.0, 86400.0)
NCASE = 24


def cases(mech, seed):
    rng = np.random.default_rng(seed)
    T = rng.uniform(200, 310, NCASE)
    P = rng.uniform(500, 101325, NCASE)
    rho = P / (8.314462618 * T)
    conc = rng.uniform(0, 1, (NCASE, mech.size)) * 10.0 ** rng.uniform(-14, -2, (NCASE, mech.size))
    prm = 10.0 ** rng.uniform(-9, -3, (NCASE, len(mech.parameters)))
    return T, P, rho, conc, prm


def run(harness, path, dt, mech, T, P, rho, conc, prm):
    lines = [f"config {path}", f"dt {dt!r}"]
    for c in range(NCASE):
        lines += ["case", f"T {float(T[c])!r}", f"P {float(P[c])!r}", f"rho {float(rho[c])!r}"]
        lines += [f"conc {s} {float(conc[c, i])!r}" for i, s in enumerate(mech.species)]
        lines += [f"rate {p} {float(prm[c, i])!r}" for i, p in enumerate(mech.parameters)]
        lines += ["end"]
    with tempfile.NamedTemporaryFile("w", suffix=".in") as f:
        f.write("\n".join(lines) + "\n")
        f.flush()
        out = subprocess.run([harness, f.name], capture_output=True, text=True, check=True).stdout
    order, porder, results = {}, {}, []
    for line in out.splitlines():
        w = line.split()
        if w[0] == "order":
            order[w[1]] = int(w[2])
        elif w[0] == "rate_order":
            porder[w[1]] = int(w[2])
        elif w[0] == "case":
            results.append({"state": int(w[3]), "steps": int(w[5]), "conc": {}})
        elif w[0] == "conc":
            results[-1]["conc"][w[1]] = w[2]
    return order, porder, results


def main(harness):
    sys.path.insert(0, str(ROOT))
    from woof.chem_micm import load_mechanism
    fixtures = []
    for k, name in enumerate(MECHANISMS):
        path = DATA / name
        mech = load_mechanism(path)
        for j, dt in enumerate(STEPS):
            T, P, rho, conc, prm = cases(mech, 1000 * k + j)
            order, porder, results = run(harness, path, dt, mech, T, P, rho, conc, prm)
            fixtures.append({
                "mechanism": name, "time_step": dt,
                "species_order": sorted(order, key=order.get),
                "parameter_order": sorted(porder, key=porder.get),
                "temperature": [x.hex() for x in T.tolist()], "pressure": [x.hex() for x in P.tolist()],
                "air_density": [x.hex() for x in rho.tolist()],
                "concentrations": [[x.hex() for x in row] for row in conc.tolist()],
                "parameters": [[x.hex() for x in row] for row in prm.tolist()],
                "state": [r["state"] for r in results], "steps": [r["steps"] for r in results],
                "result": [[r["conc"][s] for s in sorted(order, key=order.get)] for r in results]})
    out = {"oracle": "MUSICA v0.17.2 (875ef733), MICM e15898ce, RosenbrockStandardOrder, one grid cell; "
                     "x86-64 glibc host, no FMA contraction (tools/micm_oracle/build.sh)",
           "fixtures": fixtures}
    (DATA / "fixtures.json").write_text(json.dumps(out, indent=0) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
