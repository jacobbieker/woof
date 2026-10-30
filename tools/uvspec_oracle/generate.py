"""Write the uvspec oracle fixtures for woof.radtran.

    pixi run -e mlx python -m tools.uvspec_oracle.generate [case ...]

For every case in tools/uvspec_oracle/cases.py: build woof's inputs, run
uvspec (pixi ``libradtran`` environment, black box) on exactly those inputs,
and write tests/data/oracles/uvspec/<case>.json with the options, a digest
of the column/cloud inputs (rebuilt by the tests from the user's libRadtran
data, never stored here), uvspec's input text and its parsed output.
tests/data/oracles/uvspec/receipt.json records the uvspec version, the
conda build, the platform and the sha256 of every libRadtran data file
woof and uvspec read.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from tools.uvspec_oracle import cases, oracle

OUT = oracle.ROOT / "tests" / "data" / "oracles" / "uvspec"

DATA_FILES = (
    "atmmod/afglus.dat", "atmmod/afglt.dat", "atmmod/afglms.dat",
    "atmmod/afglus_ch4_vmr.dat", "atmmod/afglus_n2o_vmr.dat", "atmmod/afglus_co_vmr.dat",
    "atmmod/afglus_n2_vmr.dat",
    "correlated_k/reptran/reptran_solar_coarse.cdf",
    *(f"correlated_k/reptran/reptran_solar_coarse.lookup.{s}.cdf"
      for s in ("H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2", "N2")),
    "crs/crs_o3_all.dat", "crs/crs_no2_gom.dat", "crs/crs_o4_greenblatt.dat",
    "wc/wc.ext", "wc/wc.ssa", "wc/wc.asy",
    "ic/fu96/fu96.ext", "ic/fu96/fu96.ssa", "ic/fu96/fu96.asy",
)


def _arr(a):
    return None if a is None else np.asarray(a).tolist()


def inputs_digest(col, cl) -> str:
    """sha256 of the exact solver inputs (column arrays and cloud layers).

    Fixtures store this digest, not the arrays: AFGL profiles are libRadtran
    data and stay in the user's installation; tests rebuild the inputs from
    tools/uvspec_oracle/cases.py and check the digest.
    """
    h = hashlib.sha256()
    for a in (col.z, col.p, col.t, col.n_air, *(col.dens[k] for k in sorted(col.dens))):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    if cl is not None:
        for k, v in sorted(dataclasses.asdict(cl).items()):
            if v is not None:
                h.update(k.encode())
                h.update(np.ascontiguousarray(v, dtype=np.float64).tobytes())
    return h.hexdigest()


def fixture(name):
    opts, col, cl = cases.build(name)
    out = oracle.run(opts, col, cl)
    return {
        "case": name,
        "options": dataclasses.asdict(opts),
        "inputs_sha256": inputs_digest(col, cl),
        "uvspec_input": out.pop("input"),
        "uvspec": {k: _arr(v) for k, v in out.items()},
    }


def main(argv):
    names = argv or list(cases.CASES)
    OUT.mkdir(parents=True, exist_ok=True)
    for n in names:
        rec = fixture(n)
        (OUT / f"{n}.json").write_text(json.dumps(rec, indent=1))
        print("wrote", n)
    r = oracle.receipt(DATA_FILES)
    r["cases"] = sorted(p.stem for p in OUT.glob("*.json") if p.stem != "receipt")
    (OUT / "receipt.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
