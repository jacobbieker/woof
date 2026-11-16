"""Bitwise comparison of WOOF's RRTMG-K outputs with the WRF 4.7.1 oracle.

A field matches when every float32 bit pattern is identical; the ULP
distance is measured on the ordered integer line of float32 bit patterns
(tools/rrtmg_legacy_column_oracle/compare.py).

usage: python compare.py COLUMNS_DIR WRF_DIR WOOF_DIR ARM EXACT > summary.json
  WRF_DIR/out_<regime>.bin, WOOF_DIR/woof_<ARM>_<EXACT>_<regime>.npz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import importlib.util

import numpy as np


def _legacy_compare():
    """tools/rrtmg_legacy_column_oracle/compare.py under its own name (this
    module is also ``compare`` on its directory's path)."""
    name = "rrtmg_legacy_column_oracle_compare"
    if name not in sys.modules:
        path = (Path(__file__).resolve().parent.parent
                / "rrtmg_legacy_column_oracle" / "compare.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


diff = _legacy_compare().diff
read_records = _legacy_compare().read_records
to_cols = _legacy_compare().to_cols

#: WRF record -> WOOF output.  rthratensw_wrapper is rad_rrtmg_driver's own
#: swhr/86400; rthratensw is the driver's published RTHRATEN - RTHRATENLW.
FLUXES = ("lwupt", "lwuptc", "lwdnt", "lwdntc", "lwupb", "lwupbc", "lwdnb",
          "lwdnbc", "swupt", "swuptc", "swdnt", "swdntc", "swupb", "swupbc",
          "swdnb", "swdnbc")
PAIRS = {"engine": {"rthratenlw": "rthratenlw",
                    "rthratensw_wrapper": "rthratensw_wrapper",
                    "rthratensw": "rthratensw", "rthraten": "rthraten",
                    "glw": "glw", "olr": "olr", "gsw": "gsw",
                    "swdown": "swdown", "re_qc": "re_qc", "re_qi": "re_qi",
                    "re_qs": "re_qs", **{k: k for k in FLUXES}},
         "adapter": {"rthratenlw": "rthratenlw", "rthratensw": "rthratensw",
                     "rthraten": "rthraten", "glw": "glw", "olr": "olr",
                     "gsw": "gsw", "swdown": "swdown", "coszen": "coszen",
                     "cldfra": "cldfra", "o3rad": "o3rad",
                     **{k: k for k in FLUXES}}}


def main():
    cols_dir, wrf_dir, woof_dir, arm, exact = (
        Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]),
        sys.argv[4], sys.argv[5])
    meta = json.loads((cols_dir / "regimes.json").read_text())
    ncol = meta["ncol"]
    summary = {"arm": arm, "exact": exact, "regimes": {}}
    total = differ = 0
    for name in meta["regimes"]:
        wpath = wrf_dir / f"out_{name}.bin"
        opath = woof_dir / f"woof_{arm}_{exact}_{name}.npz"
        if not (wpath.exists() and opath.exists()):
            continue
        rec = read_records(wpath)
        out = dict(np.load(opath))
        rows = {}
        for wname, oname in PAIRS[arm].items():
            if wname not in rec or oname not in out:
                continue
            d = diff(to_cols(rec[wname], ncol), out[oname])
            rows[wname] = d
            if "n" in d:
                total += d["n"]
                differ += d["differ"]
        summary["regimes"][name] = rows
    summary["total"] = total
    summary["differ"] = differ
    json.dump(summary, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
