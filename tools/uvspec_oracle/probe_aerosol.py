"""Probe uvspec (black box) for its aerosol-species humidity handling.

    pixi run -e mlx python -m tools.uvspec_oracle.probe_aerosol

Writes tests/data/radtran/uvspec_aerosol_probes.json:

* ``rh_law``: a 300-level atmosphere (temperature 185-335 K, random
  humidity) given to uvspec with ``output_user zout T n_h2o rh``; the
  record holds each level's temperature, the H2O density woof wrote and
  uvspec's printed relative humidity.  woof's saturation law
  (woof.radtran.aerosol.ES_*) was fitted to these.
* ``profiles``: level relative humidities chosen by woof, given to uvspec
  with a water-soluble (waso) ``aerosol_species_file`` on the same levels and
  ``verbose``; the record holds uvspec's printed layer humidity and the
  humidity node it selected for each layer ("search for nearest relative
  humidity point").  These fix the layer-mean rule and the node rule
  (nearest node; a layer whose nearest node is the dry one keeps the node
  of the layer above).

Only uvspec's printed outputs are stored, with the inputs woof generated.
uvspec runs as a separate process; no libRadtran source is read.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from tools.uvspec_oracle.oracle import ROOT, _uvspec_cmd

OUT = ROOT / "tests" / "data" / "radtran" / "uvspec_aerosol_probes.json"
KB = 1.380649e-23


def _run(text, files):
    from woof.radtran import data
    with tempfile.TemporaryDirectory(prefix="wpr", dir="/tmp" if os.path.isdir("/tmp") else None) as td:
        d = Path(td)
        (d / "D").symlink_to(data.data_dir())
        for k, v in files.items():
            (d / k).write_text(v)
        p = subprocess.run(_uvspec_cmd(), input=text, capture_output=True, text=True, cwd=td)
        if p.returncode:
            raise RuntimeError(p.stderr[-2000:])
        return p.stdout, p.stderr


def _atm(z, p, t, h2o):
    n = p * 100.0 / (KB * t) * 1e-6
    rows = [" ".join(repr(float(v)) for v in (z[k], p[k], t[k], n[k], 0.0, 0.209 * n[k], h2o[k], 4e-4 * n[k], 0.0))
            for k in range(z.size)]
    return "\n".join(rows) + "\n"


def _input(library, extra=""):
    return (f"data_files_path D\natmosphere_file atm.dat\nsource solar\nmol_abs_param reptran coarse\n"
            f"wavelength 500 500.2\nrte_solver disort\nnumber_of_streams 4\nsza 30\naerosol_default\n"
            f"aerosol_species_file aer.dat waso\naerosol_species_library {library}/\nzout sur\n{extra}")


def _node_table(err):
    blk = err.split("search for nearest relative humidity point in aerosol data, iv = 0 ")[1].split("...")[0]
    rows = []
    for line in blk.splitlines():
        s = line.split()
        if len(s) == 3 and s[0].isdigit():
            rows.append((int(s[0]), float(s[1]), float(s[2])))
    return rows


def rh_law(library):
    nz = 300
    z = np.linspace(59.8, 0.0, nz)
    t = 335.0 - 2.5 * z
    p = 1000.0 * np.exp(-z / 8.0)
    rh = np.random.default_rng(2).uniform(5.0, 99.0, nz)
    es = 6.112 * np.exp(17.62 * (t - 273.15) / (t - 273.15 + 243.12))      # only to place the samples
    h2o = rh / 100.0 * es * 100.0 / (KB * t) * 1e-6
    zo = " ".join(str(float(v)) for v in z[::-1])
    out, _ = _run(_input(library, f"zout {zo}\noutput_user zout T n_h2o rh\n").replace("zout sur\n", ""),
                  {"atm.dat": _atm(z, p, t, h2o), "aer.dat": "".join(f"{float(v)!r} 1e-6\n" for v in z)})
    rows = np.array([[float(x) for x in l.split()] for l in out.splitlines() if l.strip()])
    idx = [int(np.argmin(np.abs(z - q))) for q in rows[:, 0]]
    return {"t": t[idx].tolist(), "n_h2o": h2o[idx].tolist(), "uvspec_rh": rows[:, 3].tolist()}


def profile(library, level_rh):
    from woof.radtran import aerosol
    rh = np.asarray(level_rh, dtype=np.float64)
    nz = rh.size
    z = np.arange(nz)[::-1] * 0.5
    t = 300.0 - 0.2 * np.arange(nz)[::-1]
    p = 1000.0 * np.exp(-z / 8.0)
    h2o = rh / 100.0 * aerosol.saturation_pressure_hpa(t) * 100.0 / (KB * t) * 1e-6
    _, err = _run(_input(library, "verbose\n"),
                  {"atm.dat": _atm(z, p, t, h2o), "aer.dat": "".join(f"{float(v)!r} 1e-6\n" for v in z)})
    tab = _node_table(err)
    return {"level_t": t.tolist(), "level_n_h2o": h2o.tolist(),
            "uvspec_layer_rh": [r for _, r, _ in tab], "uvspec_node": [n for _, _, n in tab]}


def _blocks(targets):
    out = []
    for v in targets:
        out += [v, v]
    return out


def main():
    from tools.uvspec_oracle import opac_library
    lib = opac_library.library_root() / "woof"
    rng = np.random.default_rng(1)
    profiles = {
        "ascending": _blocks([0.5, 5, 10, 20, 24.9, 25.1, 30, 40, 49.9, 59.9, 60.1, 69.9, 74.9, 75.1, 84.9,
                              85.1, 92.4, 92.6, 96.4, 96.6, 98.4, 98.6, 99.4, 99.6, 100.0, 102.0, 110.0]),
        "descending": _blocks([110.0, 102.0, 99.6, 99.4, 98.6, 98.4, 96.6, 96.4, 92.6, 92.4, 85.1, 84.9, 75.1,
                               74.9, 69.9, 60.1, 59.9, 49.9, 40, 30, 25.1, 24.9, 20, 10, 5, 0.5]),
        "dry_wet_dry": _blocks([5, 60, 5]),
        "dry_25_dry": _blocks([5, 24, 5, 26, 5]),
        "jumps": _blocks([99, 72, 99, 52, 99, 20, 0.1]),
        "random": rng.uniform(0.0, 99.0, 400).tolist(),
        "random_dry": rng.uniform(0.0, 40.0, 200).tolist(),
    }
    rec = {"tool": "uvspec (black box), tools/uvspec_oracle/probe_aerosol.py",
           "library_species": "waso (woof OPAC library; node list 0 50 70 80 90 95 98 99)",
           "rh_law": rh_law(lib),
           "profiles": {k: dict(level_rh=list(map(float, v)), **profile(lib, v)) for k, v in profiles.items()}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
