"""Write the TUV-x oracle fixtures tests/data/tuvx_oracle/fixtures.json from the built harness.

Two column layouts: TUV 5.4's 1 km grid to 120 km, and a model-like stretched grid (a terrain
surface, forty layers to 30 km, then coarser layers to 120 km).  Each case perturbs a standard
atmosphere (temperature, air scale, ozone amount and height) and draws the solar zenith angle and
Earth-Sun distance; inputs and results are binary64 words in hex.  See build.sh for the oracle.

    pixi run -e default python tools/tuvx_oracle/generate.py <work-dir>
"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1].parent
OUT = ROOT / "tests" / "data" / "tuvx_oracle" / "fixtures.json"
NCASE = 8
#: US Standard Atmosphere 1976 temperature break points [km, K], extended to 120 km
_TZ = [0, 11, 20, 32, 47, 51, 71, 86, 100, 120]
_TK = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.87, 195.08, 360.0]


def layouts(rng):
    uniform = np.linspace(0.0, 120.0, 121)
    yield "tuv54_1km", [uniform] * NCASE
    grids = []
    for _ in range(NCASE):
        z0 = rng.uniform(0.0, 3.0)
        model = z0 + (30.0 - z0) * (np.linspace(0.0, 1.0, 41) ** 1.6)
        grids.append(np.concatenate([model, np.arange(32.0, 60.0, 2.0), np.arange(60.0, 120.1, 5.0)]))
    yield "model_stretched", grids


def column(rng, z):
    t = np.interp(z, _TZ, _TK) + rng.uniform(-10.0, 10.0) * np.exp(-z / 20.0)
    air = 2.547e19 * rng.uniform(0.9, 1.1) * np.exp(-z / rng.uniform(6.8, 7.6))
    peak, width, amount = rng.uniform(20.0, 27.0), rng.uniform(5.0, 9.0), rng.uniform(2.0e12, 7.0e12)
    o3 = amount * np.exp(-((z - peak) / width) ** 2) + 5.0e11 * np.exp(-z / 4.0)

    def layers(edge):
        return 0.5 * (edge[1:] + edge[:-1]) * np.diff(z) * 1.0e5

    return {"temperature": t, "air": air, "air_layer": layers(air), "O2": 0.2095 * air,
            "O2_layer": layers(0.2095 * air), "O3": o3, "O3_layer": layers(o3)}


def words(values):
    return [float(v).hex() for v in np.asarray(values, dtype=np.float64)]


def run(harness, work, cases):
    nlev = len(cases[0]["edges"]) - 1
    lines = ["config woof_photolysis.json", f"nlev {nlev}"]
    text = lambda v: " ".join(repr(float(x)) for x in v)  # noqa: E731
    for c in cases:
        lines += ["case", f"sza {c['sza']!r}", f"esd {c['esd']!r}", "edges " + text(c["edges"]),
                  "temperature " + text(c["temperature"]),
                  "air " + text(np.r_[c["air"], c["air_layer"]]),
                  "O2 " + text(np.r_[c["O2"], c["O2_layer"]]),
                  "O3 " + text(np.r_[c["O3"], c["O3_layer"]]), "end"]
    out = subprocess.run([str(harness)], input="\n".join(lines) + "\n", capture_output=True, text=True,
                         check=True, cwd=work / "tuv-x").stdout
    labels, results = {}, []
    for line in out.splitlines():
        w = line.split()
        if w[0] == "label":
            labels[int(w[1])] = " ".join(w[2:])
        elif w[0] == "case":
            results.append({})
        elif w[0] == "j":
            j, i = int(w[1]), int(w[2])
            value = np.frombuffer(bytes.fromhex(w[3]), dtype=">f8")[0]
            results[-1].setdefault(labels[j], {})[i] = float(value).hex()
    return [labels[k] for k in sorted(labels)], [
        {name: [r[name][i] for i in sorted(r[name])] for name in r} for r in results]


def main(work):
    work = Path(work).resolve()
    rng = np.random.default_rng(20261110)
    fixtures = []
    for name, grids in layouts(rng):
        cases = []
        for z in grids:
            c = column(rng, z)
            c.update(edges=z, sza=float(rng.uniform(0.0, 89.0)), esd=float(rng.uniform(0.983, 1.017)))
            cases.append(c)
        # One harness per layout: every case shares the layout's level count.
        labels, results = run(work / "harness", work, cases)
        fixtures.append({
            "layout": name, "reactions": labels,
            "cases": [{"solar_zenith_angle_deg": c["sza"].hex(), "earth_sun_distance_au": c["esd"].hex(),
                       "height_edges_km": words(c["edges"]),
                       **{key: words(c[key]) for key in ("temperature", "air", "air_layer", "O2", "O2_layer",
                                                         "O3", "O3_layer")},
                       "photolysis_rate_constants_s-1": r} for c, r in zip(cases, results)]})
    out = {"oracle": "NCAR TUV-x v0.17.0 (ebec1cb4) with a clean-room musica_assert, host-column harness over "
                     "tools/tuvx_oracle/photolysis.json (TUV 5.4 cut to O2, O3 -> O(1D), O3 -> O(3P) and NO2); "
                     "x86-64 glibc host, gfortran -O2 -ffp-contract=off (tools/tuvx_oracle/build.sh)",
           "units": {"temperature": "K", "air, O2, O3": "molecule cm-3 at the edges",
                     "*_layer": "molecule cm-2 per layer", "photolysis_rate_constants_s-1": "s-1 at the edges"},
           "fixtures": fixtures}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=0) + "\n")
    print(OUT)


if __name__ == "__main__":
    main(sys.argv[1])
