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
NTWILIGHT = 4
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
    # Appended after the layouts above (their draws, and so their words, are unchanged): the sun at and below
    # the horizon, where the direct beam grazes or misses layers (spherical geometry nid < 0, screening height).
    yield "twilight", [uniform] * NTWILIGHT


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
    proc = subprocess.run([str(harness)], input="\n".join(lines) + "\n", capture_output=True, text=True,
                          cwd=work / "tuv-x")
    if proc.returncode:
        sys.exit(f"harness failed with status {proc.returncode}:\n{proc.stderr}")
    out = proc.stdout
    labels, results, config = {}, [], {"wedge": {}, "etf": {}, "esd": {}}
    tables = []
    word = lambda h: float(np.frombuffer(bytes.fromhex(h), dtype=">f8")[0]).hex()  # noqa: E731
    for line in out.splitlines():
        w = line.split()
        if w[0] == "label":
            labels[int(w[1])] = " ".join(w[2:])
        elif w[0] in config:
            config[w[0]][int(w[1])] = word(w[2])
        elif w[0] in INTERMEDIATES:
            # (tag, reaction or 0) -> {(height edge, wavelength bin): word}, 1-based
            tables[-1].setdefault((w[0], int(w[1])), {})[int(w[2]), int(w[3])] = word(w[4])
        elif w[0] == "case":
            results.append({})
            tables.append({})
        elif w[0] == "j":
            j, i = int(w[1]), int(w[2])
            value = np.frombuffer(bytes.fromhex(w[3]), dtype=">f8")[0]
            results[-1].setdefault(labels[j], {})[i] = float(value).hex()
    configuration = {"wavelength_edges_nm": [config["wedge"][i] for i in sorted(config["wedge"])],
                     "extraterrestrial_flux_photon_cm-2_s-1": [config["etf"][i] for i in sorted(config["etf"])],
                     "earth_sun_distance_au": [config["esd"][i] for i in sorted(config["esd"])]}
    names = [labels[k] for k in sorted(labels)]
    return names, [
        {name: [r[name][i] for i in sorted(r[name])] for name in r} for r in results], configuration, [
        intermediates(t, names) for t in tables]


#: per-case intermediate tables the harness writes: cross section, quantum yield, actinic flux parts, then the
#: radiative-transfer intermediates (see harness.F90)
INTERMEDIATES = ("xs", "qy", "fdr", "fup", "fdn", "nid", "dsdh", "avc", "asc", "ovc", "osc", "lsx", "rod", "aod",
                 "ass", "ag")

#: cases per layout whose intermediates are kept, and their height edges (five, both ends included)
INTERMEDIATE_CASES = 2


def _keep(nrow):
    return sorted({1, nrow // 4, nrow // 2, (3 * nrow) // 4, nrow})


def intermediates(table, names):
    """{"height_edges": [...], "cross_section": {reaction: [[word per bin] per kept edge]}, ...}, 0-based edges.

    Layer tables keep their own five rows ("layers", 0-based from the surface); the geometry keeps every nid and
    the dsdh rows of the kept edge indices (counted from the top, as TUV-x does); columns are kept whole."""
    any_table = table["xs", 1]
    nedge = max(i for i, _ in any_table)
    nbin = max(k for _, k in any_table)
    keep = _keep(nedge)
    rows = lambda t, kept=keep: [[t[i, k] for k in range(1, nbin + 1)] for i in kept]  # noqa: E731
    nlayer = nedge - 1
    layers = _keep(nlayer)
    column = lambda t: [t[i, 1] for i in range(1, max(i for i, _ in t) + 1)]  # noqa: E731
    o2 = names.index("O2+hv->O+O") + 1
    rt = {"layers": [i - 1 for i in layers],
          "nid": [int(float.fromhex(w)) for w in column(table["nid", 0])],
          "dsdh": [[table["dsdh", 0][i, k] for k in range(1, nlayer + 1)] for i in keep],
          "air_vertical_column": column(table["avc", 0]), "air_slant_column": column(table["asc", 0]),
          "o2_vertical_column": column(table["ovc", 0]), "o2_slant_column": column(table["osc", 0]),
          "o2_band_cross_section_cm2": rows(table["lsx", o2]),
          "radiator_optical_depth": {name: rows(table["rod", r], layers)
                                     for r, name in enumerate(("air", "O2", "O3"), start=1)},
          "accumulated_optical_depth": rows(table["aod", 0], layers),
          "accumulated_single_scattering_albedo": rows(table["ass", 0], layers),
          "accumulated_asymmetry_factor": rows(table["ag", 0], layers)}
    return {"height_edges": [i - 1 for i in keep], "radiative_transfer": rt,
            "cross_section_cm2": {names[j - 1]: rows(table["xs", j]) for j in range(1, len(names) + 1)},
            "quantum_yield": {names[j - 1]: rows(table["qy", j]) for j in range(1, len(names) + 1)},
            "actinic_flux_direct": rows(table["fdr", 0]), "actinic_flux_diffuse_up": rows(table["fup", 0]),
            "actinic_flux_diffuse_down": rows(table["fdn", 0])}


def main(work):
    work = Path(work).resolve()
    rng = np.random.default_rng(20261110)
    fixtures, configuration = [], None
    for name, grids in layouts(rng):
        cases = []
        for z in grids:
            c = column(rng, z)
            low, high = (89.0, 96.0) if name == "twilight" else (0.0, 89.0)
            c.update(edges=z, sza=float(rng.uniform(low, high)), esd=float(rng.uniform(0.983, 1.017)))
            cases.append(c)
        # One harness per layout: every case shares the layout's level count.
        labels, results, layout_configuration, tables = run(work / "harness", work, cases)
        # The configuration-time pieces do not depend on the column: every layout must report the same words.
        if configuration is not None and layout_configuration != configuration:
            sys.exit("the configuration-time words differ between column layouts")
        configuration = layout_configuration
        fixtures.append({
            "layout": name, "reactions": labels,
            "cases": [{"solar_zenith_angle_deg": c["sza"].hex(), "earth_sun_distance_au": c["esd"].hex(),
                       "height_edges_km": words(c["edges"]),
                       **{key: words(c[key]) for key in ("temperature", "air", "air_layer", "O2", "O2_layer",
                                                         "O3", "O3_layer")},
                       "photolysis_rate_constants_s-1": r,
                       **({"intermediates": t} if n < INTERMEDIATE_CASES else {})}
                      for n, (c, r, t) in enumerate(zip(cases, results, tables))]})
    out = {"oracle": "NCAR TUV-x v0.17.0 (ebec1cb4) with a clean-room musica_assert, host-column harness over "
                     "tools/tuvx_oracle/photolysis.json (TUV 5.4 cut to O2, O3 -> O(1D), O3 -> O(3P) and NO2); "
                     "x86-64 glibc host, gfortran -O2 -ffp-contract=off (tools/tuvx_oracle/build.sh)",
           "units": {"temperature": "K", "air, O2, O3": "molecule cm-3 at the edges",
                     "*_layer": "molecule cm-2 per layer", "photolysis_rate_constants_s-1": "s-1 at the edges",
                     "configuration": "the wavelength grid's edges (nm) and the extraterrestrial flux on its bins "
                                      "(photon cm-2 s-1), as the built core holds them; the Earth-Sun "
                                      "distance profile (AU) on the time grid's edges (2002-03-21, 12 and "
                                      "14 UT, photolysis.json)",
                     "intermediates": "first two cases of each layout, five height edges (0-based): cross "
                                      "sections cm2, quantum yields, actinic flux parts per bin; "
                                      "radiative_transfer: spherical geometry (nid per edge and dsdh rows "
                                      "of the kept edges, both counted from the top), air and O2 vertical "
                                      "(per layer) and slant (per edge) columns molecule cm-2, the O2 "
                                      "reaction's cross section with the Lyman-alpha / Schumann-Runge bands "
                                      "applied, and per kept layer (0-based from the surface) each "
                                      "radiator's optical depth and the accumulated optical depth, "
                                      "single-scattering albedo and asymmetry factor"},
           "configuration": configuration, "fixtures": fixtures}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=0) + "\n")
    print(OUT)


if __name__ == "__main__":
    main(sys.argv[1])
