#!/usr/bin/env python3
"""Deterministic synthetic per-step inputs for the prec_acc_dt / acc_phy_tend oracle (no data from any case).

Writes ``inputs.bin``: little endian header int32 (nc, nl, ns, cu_used), float32 (dt, prec_acc_dt minutes), int32
history_interval minutes; then per step s = 1..ns (C order, float32): raincv[nc], rainncv[nc], snowncv[nc], t2[nc], and
twelve (nl, nc) rates in the order h_diabatic, qv_diabatic, rthcuten, rqvcuten, rucuten, rvcuten, rthblten, rqvblten,
rublten, rvblten, rthratenlw, rthratensw.

    make_inputs.py OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np

NC, NL, NS = 12, 7, 11
DT, PREC_MIN, HIST_MIN, CU_USED = 60.0, 3.0, 4, 1


def main(out):
    rng = np.random.default_rng(20261113)
    f = np.float32
    with open(Path(out) / "inputs.bin", "wb") as h:
        h.write(np.array([NC, NL, NS, CU_USED], "<i4").tobytes())
        h.write(np.array([DT, PREC_MIN], "<f4").tobytes())
        h.write(np.array([HIST_MIN], "<i4").tobytes())
        for _ in range(NS):
            # negative amounts exercise WRF's MAX(., 0); t2 straddles 273.15 for the frozen-convective-rain branch
            for lo, hi in ((-0.02, 0.5), (-0.02, 1.2), (0.0, 0.4)):
                h.write(rng.uniform(lo, hi, NC).astype(f).tobytes())
            h.write(rng.uniform(268.0, 278.0, NC).astype(f).tobytes())
            for scale in (1e-3, 1e-6, 1e-4, 1e-7, 1e-3, 1e-3, 2e-3, 1e-6, 1e-3, 1e-3, 3e-5, 4e-5):
                h.write((rng.standard_normal((NL, NC)) * scale).astype(f).tobytes())


if __name__ == "__main__":
    main(sys.argv[1])
