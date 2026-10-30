"""WOOF's radiation-cloud sequence against the sliced WRF oracle.

The sequence is the one both RRTMG adapters run (woof/core/rrtmg_legacy.py
``chunk_inputs`` and woof/core/rrtmgp.py ``__call__``): the ICLOUD_CU = 1
sum, cal_cldfra1 / cal_cldfra2, the ICLOUD_CU = 2 merge inside ICLOUD = 1,
then Deng's -- each step the adapters' own function
(woof.core.cumulus_radiation and woof.core.rrtmgp's cloud fractions).

    python compare_woof.py CASES.bin ORACLE.out
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

try:
    from . import rad_cloud_cases as R
except ImportError:                       # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rad_cloud_cases as R  # noqa: E402


def woof_cloud(case):
    """(cldfra, qc, qi) as the adapters form them, on (ncol, nz) columns."""
    import cupy as cp
    from woof.core import cumulus_radiation as curad
    from woof.core import rrtmgp

    dev = {k: cp.asarray(case[k]) for k in R.FIELDS}
    qc, qi = dev["qc"], dev["qi"]
    f_qc, f_qi, f_qs = bool(case["f_qc"]), bool(case["f_qi"]), bool(case["f_qs"])
    if case["icloud_cu"] == 1:
        qc, qi = curad.add_convective_condensate(
            qc, qi, dev["qc_cu"], dev["qi_cu"], f_qc=f_qc, f_qi=f_qi)
    if case["icloud"] == 1:
        cldfra = rrtmgp.cal_cldfra1(dev["qv"], qc, qi, dev["qs"], dev["t"],
                                    dev["p"], f_qc=True, f_qi=f_qi,
                                    f_qs=f_qs, mp_physics=case["mp"])
        if case["icloud_cu"] == 2:
            cldfra, qc, qi = curad.merge_kf_cloud(
                cldfra, qc, qi, cldfra_dp=dev["cldfra_dp"],
                cldfra_sh=dev["cldfra_sh"], qc_cu=dev["qc_cu"],
                qi_cu=dev["qi_cu"])
    else:
        cldfra = rrtmgp.cal_cldfra2(qc, qi, f_qc=f_qc, f_qi=f_qi)
    if case["shcu"] == 5:
        cldfra, qc = curad.merge_deng_cloud(
            cldfra, qc, cldfra_sh=dev["cldfra_sh"], cw_rad=dev["cw_rad"])
    return {"cldfra": cp.asnumpy(cldfra), "qc": cp.asnumpy(qc),
            "qi": cp.asnumpy(qi)}


def run(cases_path, oracle_path):
    from woof.core.fp32_ulp import monotone_fp32_key
    cases = R.read_input(cases_path)
    outs = R.read_output(oracle_path, cases)
    rows = []
    for n, (case, ref) in enumerate(zip(cases, outs)):
        mine = woof_cloud(case)
        for name in ("cldfra", "qc", "qi"):
            a = np.asarray(mine[name], np.float32).reshape(-1)
            b = np.asarray(ref[name], np.float32).reshape(-1)
            ulp = np.abs(monotone_fp32_key(a) - monotone_fp32_key(b))
            rows.append({"case": n, "field": name, "words": int(a.size),
                         "icloud": case["icloud"],
                         "icloud_cu": case["icloud_cu"], "shcu": case["shcu"],
                         "mp": case["mp"], "max_ulp": int(ulp.max()),
                         "bit_unequal": int((a.view(np.uint32)
                                             != b.view(np.uint32)).sum())})
    return rows


if __name__ == "__main__":
    rows = run(sys.argv[1], sys.argv[2])
    bad = [r for r in rows if r["bit_unequal"]]
    print(json.dumps({"words": sum(r["words"] for r in rows),
                      "bit_unequal": sum(r["bit_unequal"] for r in rows),
                      "max_ulp": max(r["max_ulp"] for r in rows)}))
    for r in bad[:20]:
        print(r)
