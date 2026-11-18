"""Pack the RRTMG-K column oracle's inputs and WRF outputs into the fixture.

usage: python make_fixture.py COLUMNS_DIR BUILD_DIR WRF_DIR FIXTURE_DIR

COLUMNS_DIR is make_columns.py's output, BUILD_DIR build.sh's (its
stock/build-receipt.txt), WRF_DIR the rad_k_column_oracle runs
(out_<regime>.bin).  FIXTURE_DIR receives:

* columns_<grid>.npz, regimes.json  the exact float32 inputs
* wrf471.npz   "<regime>/<field>" -> (ncol, nk), (ncol,) or scalar float32:
  every graded field (compare.PAIRS) and the driver inputs the engine arm
  is fed (t8w, cldfra, o3rad, coszen, solcon)
* provenance.json  the build receipt and the fixture's own hashes.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_woof import _compare, record  # noqa: E402

PAIRS = _compare().PAIRS

#: run_woof.run_engine's inputs from the record, beside the graded fields.
ENGINE_INPUTS = ("t8w", "cldfra", "o3rad", "coszen", "solcon", "cdm2",
                 "extliq1", "ssaliq1", "asyliq1")


def main():
    cols, build, wrf, dst = (Path(a) for a in sys.argv[1:5])
    dst.mkdir(parents=True, exist_ok=True)
    meta = json.loads((cols / "regimes.json").read_text())
    for name in ["regimes.json"] + sorted(p.name for p in cols.glob("columns_*.npz")):
        shutil.copyfile(cols / name, dst / name)
    fields = sorted(set(PAIRS["engine"]) | set(PAIRS["adapter"]) | set(ENGINE_INPUTS))
    out = {}
    for name in meta["regimes"]:
        rec = record(wrf, name, meta["ncol"])
        for field in fields:
            if field in rec:
                out[f"{name}/{field}"] = np.ascontiguousarray(rec[field], np.float32)
    np.savez_compressed(dst / "wrf471.npz", **out)
    prov = {"build_receipt": (build / "stock" / "build-receipt.txt").read_text().splitlines()}
    prov["fixture_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(dst.glob("*.np*")) + [dst / "regimes.json"]}
    (dst / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(f"fixture: {len(out)} arrays -> {dst}")


if __name__ == "__main__":
    main()
