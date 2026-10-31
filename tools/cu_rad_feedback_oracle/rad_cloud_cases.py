"""Cases for the radiation-cloud oracle (run_rad_cloud.F90) and their reader.

Every combination the radiation driver distinguishes: ICLOUD 1 and 2,
ICLOUD_CU 0, 1 (Grell) and 2 (Kain-Fritsch), shcu_physics 0 and 5 (Deng),
over the three cal_cldfra1 moisture arms woof admits (qc+qi+qs, qc+qi
without snow -- P3's and, at mp 5, Ferrier-Aligo's -- and qc alone).  The
columns span warm to cirrus-cold layers, sub- to supersaturated vapour,
condensate below and above QCLDMIN, and convective fields that are zero,
small and large (CLDFRA_DP + CLDFRA_SH above 1, so the MIN(1, ...) clamp
runs).

    python rad_cloud_cases.py OUT.bin
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

F = np.float32
I = np.int32
NCOL, NZ = 48, 24
FIELDS = ("qv", "qc", "qi", "qs", "t", "p", "qc_cu", "qi_cu", "cldfra_dp",
          "cldfra_sh", "cw_rad")
#: (mp_physics, f_qc, f_qi, f_qs)
ARMS = ((8, 1, 1, 1), (50, 1, 1, 0), (5, 1, 1, 0), (1, 1, 0, 0))


def _columns(rng):
    nc, nz = NCOL, NZ
    p = np.linspace(100000.0, 15000.0, nz)[None, :] * rng.uniform(
        0.8, 1.02, (nc, 1))
    t = (rng.uniform(280.0, 305.0, (nc, 1))
         - rng.uniform(5.0, 8.0, (nc, 1)) * np.linspace(0.0, 12.0, nz)[None, :])
    t = np.maximum(t, 195.0)
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    qsat = 0.622 * es / (p - es)
    qv = qsat * rng.uniform(0.2, 1.08, (nc, nz))

    def condensate(scale):
        x = rng.uniform(0.0, scale, (nc, nz)) * (rng.random((nc, nz)) < 0.5)
        tiny = rng.random((nc, nz)) < 0.08
        return np.where(tiny, rng.uniform(0.0, 2.0e-12, (nc, nz)), x)

    qc = condensate(1.0e-3)
    qi = condensate(3.0e-4)
    qs = condensate(5.0e-4)
    qc_cu = condensate(8.0e-4)
    qi_cu = condensate(4.0e-4)
    dp = rng.uniform(0.0, 0.6, (nc, nz)) * (rng.random((nc, nz)) < 0.4)
    sh = rng.uniform(0.0, 1.0, (nc, nz)) * (rng.random((nc, nz)) < 0.5)
    cw = condensate(6.0e-4)
    return dict(qv=qv, qc=qc, qi=qi, qs=qs, t=t, p=p, qc_cu=qc_cu,
                qi_cu=qi_cu, cldfra_dp=dp, cldfra_sh=sh, cw_rad=cw)


def build_cases(seed: int = 20261030) -> list[dict]:
    rng = np.random.default_rng(seed)
    cases = []
    for icloud in (1, 2):
        for icloud_cu in (0, 1, 2):
            for shcu in (0, 5):
                for mp, fqc, fqi, fqs in ARMS:
                    cols = _columns(rng)
                    cases.append(dict(icloud=icloud, icloud_cu=icloud_cu,
                                      shcu=shcu, mp=mp, f_qc=fqc, f_qi=fqi,
                                      f_qs=fqs, **{k: np.asarray(v, F)
                                                   for k, v in cols.items()}))
    return cases


def write_input(path, cases) -> None:
    out = [b"RCLDIN1 ", np.asarray([len(cases)], I).tobytes()]
    for c in cases:
        out.append(np.asarray([NCOL, NZ, c["icloud"], c["icloud_cu"],
                               c["shcu"], c["mp"], c["f_qc"], c["f_qi"],
                               c["f_qs"]], I).tobytes())
        # Fortran (ncol, nz, 1) is ncol fastest: write the (nz, ncol) C array.
        for k in FIELDS:
            out.append(np.ascontiguousarray(c[k].T, F).tobytes())
    Path(path).write_bytes(b"".join(out))


def _read_bytes(path) -> bytes:
    data = Path(path).read_bytes()
    if str(path).endswith(".gz"):
        import gzip
        data = gzip.decompress(data)
    return data


def read_input(path) -> list[dict]:
    data = _read_bytes(path)
    assert data[:8] == b"RCLDIN1 "
    pos = 8
    ncase = int(np.frombuffer(data, I, 1, pos)[0])
    pos += 4
    cases = []
    for _ in range(ncase):
        hdr = np.frombuffer(data, I, 9, pos)
        pos += 36
        nc, nz = int(hdr[0]), int(hdr[1])
        c = dict(icloud=int(hdr[2]), icloud_cu=int(hdr[3]), shcu=int(hdr[4]),
                 mp=int(hdr[5]), f_qc=int(hdr[6]), f_qi=int(hdr[7]),
                 f_qs=int(hdr[8]))
        for k in FIELDS:
            c[k] = np.frombuffer(data, F, nz * nc, pos).reshape(nz, nc).T.copy()
            pos += 4 * nz * nc
        cases.append(c)
    assert pos == len(data)
    return cases


def read_output(path, cases) -> list[dict]:
    data = _read_bytes(path)
    assert data[:8] == b"RCLDOU1 "
    pos = 12
    outs = []
    for _c in cases:
        o = {}
        for k in ("cldfra", "qc", "qi"):
            o[k] = np.frombuffer(data, F, NZ * NCOL, pos).reshape(
                NZ, NCOL).T.copy()
            pos += 4 * NZ * NCOL
        outs.append(o)
    assert pos == len(data)
    return outs


if __name__ == "__main__":
    write_input(sys.argv[1], build_cases())
