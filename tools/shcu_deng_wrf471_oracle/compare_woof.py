"""Grade WOOF's Deng shallow cumulus against the WRF v4.7.1 oracle.

Runs WOOF's real device path -- :func:`woof.core.shcu_deng.launch_deng` with
a :class:`woof.core.shcu_deng.DengState` carried step to step exactly as the
Fortran driver carries the grid -- on the float32 words the Fortran read,
and compares every output word of every step.

    python compare_woof.py INPUT.bin ORACLE.out [REPORT.json]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

try:
    from . import deng_columns as D
except ImportError:                       # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import deng_columns as D  # noqa: E402

#: Every word the oracle writes, WOOF's name for it.
GRADED_3D = D.OUT_3D
GRADED_2D = D.OUT_2D
GRADED_INT = D.OUT_INT


def ulp_distance(a, b):
    from woof.core.fp32_ulp import monotone_fp32_key
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return np.abs(monotone_fp32_key(a) - monotone_fp32_key(b))


def run(input_path, oracle_path):
    """Per (case, step, field): max ULP, words off, bitwise-unequal words."""
    import cupy as cp
    from woof.core.shcu_deng import DengState, launch_deng

    cases = D.read_input(input_path)
    oracle = D.read_output(oracle_path)
    if len(cases) != len(oracle):
        raise SystemExit("case count mismatch")
    rows = []
    for ci, (c, o) in enumerate(zip(cases, oracle)):
        nz, ncol = c["nz"], c["ncol"]
        shape3 = (nz, 1, ncol)
        st = c["state"]

        def dev(a, shape):
            return cp.asarray(np.ascontiguousarray(
                np.asarray(a, dtype=np.float32).reshape(shape)))

        state = DengState.cold(nz, 1, ncol)
        for k in ("w0avg", "tkeavg", "cldareaa", "cldareab", "cldliqa",
                  "cldliqb", "wub"):
            getattr(state, k)[...] = dev(st[k], shape3)
        for k in ("rainsh", "rainshv", "rainshvb", "pblmax", "capesave",
                  "xtime1", "radsave", "clddpthb", "cldtopb", "pblhavg"):
            getattr(state, k)[...] = dev(st[k], (1, ncol))
        for k in D.STATE_INT:
            getattr(state, k)[...] = cp.asarray(
                np.asarray(st[k], dtype=np.int32).reshape(1, ncol))
        state.ainckfsa[...] = dev(st["ainckfsa"], (D.NSH, 1, ncol))
        mavail = dev(st["mavail"], (1, ncol))
        pratesh = dev(st["pratesh"], (1, ncol))
        ht = dev(c["ht"], (1, ncol))
        xlong = dev(c["xlong"], (1, ncol))
        dnw = dev(c["dnw"], (nz,))
        for si, (s, ref) in enumerate(zip(c["steps"], o["steps"])):
            tke = s["qke"] if c["bl_pbl"] == 5 else s["tke_pbl"]
            out = launch_deng(
                u=dev(s["u"], shape3), v=dev(s["v"], shape3),
                t=dev(s["t"], shape3), qv=dev(s["qv"], shape3),
                qc=dev(s["qc"], shape3), qr=dev(s["qr"], shape3),
                p=dev(s["p"], shape3), pii=dev(s["pi"], shape3),
                rho=dev(s["rho"], shape3), dz8w=dev(s["dz8w"], shape3),
                tke=dev(tke, shape3), exch_h=dev(s["exch_h"], shape3),
                el_pbl=dev(s["el_pbl"], shape3),
                rthratenlw=dev(s["rthratenlw"], shape3),
                rthratensw=dev(s["rthratensw"], shape3),
                z_at_w=dev(s["z_at_w"], (nz + 1, 1, ncol)),
                w=dev(s["w"], (nz + 1, 1, ncol)),
                pblh=dev(s["pblh"], (1, ncol)), ht=ht, xlong=xlong, dnw=dnw,
                mavail=mavail, state=state, dt=float(c["dt"]),
                dx=float(c["dx"]), xtime=float(s["xtime"]),
                gmt=float(c["gmt"]), adaptive=c["adapt"],
                bl_pbl_physics=c["bl_pbl"], pratesh=pratesh,
                raise_on_status=False)
            status = cp.asnumpy(out["status"]).reshape(-1)
            mine = {k: cp.asnumpy(out[k]).reshape(nz, ncol)
                    for k in ("rthshten", "rqvshten", "rqcshten", "rqrshten",
                              "rushten", "rvshten", "rdcashten", "rqcdcshten")}
            for k in ("cldfra_sh", "ca_rad", "cw_rad", "w0avg", "tkeavg",
                      "cldareaa", "cldareab", "cldliqa", "cldliqb", "wub"):
                mine[k] = cp.asnumpy(getattr(state, k)).reshape(nz, ncol)
            for k in GRADED_2D:
                src = mavail if k == "mavail" else (
                    pratesh if k == "pratesh" else getattr(state, k))
                mine[k] = cp.asnumpy(src).reshape(ncol)
            for k in GRADED_INT:
                mine[k] = cp.asnumpy(getattr(state, k)).reshape(ncol)
            mine["ainckfsa"] = cp.asnumpy(state.ainckfsa).reshape(D.NSH, ncol)
            for k in GRADED_3D + GRADED_2D + ("ainckfsa",):
                a = np.asarray(mine[k], dtype=np.float32)
                b = np.asarray(ref[k], dtype=np.float32)
                ulp = ulp_distance(a, b)
                bits = (a.view(np.uint32) != b.view(np.uint32)).sum()
                rows.append({"case": ci, "step": si, "field": k,
                             "words": int(a.size), "max_ulp": int(ulp.max()),
                             "off": int((ulp > 0).sum()),
                             "bit_unequal": int(bits),
                             "status_columns": int((status != 0).sum())})
            for k in GRADED_INT:
                a = np.asarray(mine[k]).astype(np.int64)
                b = np.asarray(ref[k]).astype(np.int64)
                rows.append({"case": ci, "step": si, "field": k,
                             "words": int(a.size),
                             "max_ulp": int(np.abs(a - b).max()),
                             "off": int((a != b).sum()),
                             "bit_unequal": int((a != b).sum()),
                             "status_columns": int((status != 0).sum())})
    return rows


def summarize(rows):
    words = sum(r["words"] for r in rows)
    off = sum(r["off"] for r in rows)
    bits = sum(r["bit_unequal"] for r in rows)
    worst = max(rows, key=lambda r: r["max_ulp"])
    return {"words": words, "off": off, "bit_unequal": bits,
            "max_ulp": worst["max_ulp"], "worst": worst,
            "status_columns": sum(r["status_columns"] for r in rows
                                  if r["field"] == "rthshten")}


if __name__ == "__main__":
    rows = run(sys.argv[1], sys.argv[2])
    summary = summarize(rows)
    print(json.dumps(summary, indent=1))
    bad = [r for r in rows if r["off"] or r["bit_unequal"]]
    for r in bad[:40]:
        print(r)
    if len(sys.argv) > 3:
        with open(sys.argv[3], "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "rows": rows}, fh, indent=0)
