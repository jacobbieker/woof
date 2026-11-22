"""Desroziers diagnostics (plan gate G4a) against WRFDA 4.7.1's own ``da_tune_obs_desroziers``.

``WOOF_WRFDA471_DESROZIERS_DIR`` is one directory with the files below, or the oracle's run directory
(on pb ``/mnt/NVME2/woof-work/scratch/wrfda/run``: ``des-unpert/`` has ``be.dat``, ``ob.ascii``, ``fg``,
``unpert_obs.0000``, ``gts_omb_oma_01.0000``; ``des-pert/`` has ``rand_obs_error``; ``des-tool/`` has ``tool.out``).
It holds the tutorial's ``be.dat``, ``ob.ascii`` and ``fg`` and the files of
WRFDA's randomised-trace workflow (``tools/wrfda_oracle/run_tutorial.sh`` twice, with
``WRFVAR1='write_unpert_obs=.true.,'`` and the second with ``WRFVAR5='omb_add_noise=.true.,'``, then the tool
with ``fort.45..49`` = ``rand_obs_error``, ``pert_obs.0000``, ``unpert_obs.0000``, the ``jo`` file and the
unperturbed ``wrfda.out`` each ending in a ``*****`` line): ``rand_obs_error``, ``unpert_obs.0000``,
``gts_omb_oma_01.0000`` (the unperturbed run's O, O-B and O-A) and ``tool.out`` (the tool's printed table).

Asserted: the table (``Jo`` expected and actual, the error factors, the totals, the report counts) is the tool's
to 3e-5 absolute plus 1e-6 relative (the tool prints five decimals and sums in binary32; measured at most 2e-5);
``H'(xhat)`` is WRFDA's ``unpert_obs`` to the file's seven digits; and the classic ``<(O-A)(O-B)>``,
``<(A-B)(O-B)>`` statistics equal those computed from WRFDA's own O-B and O-A.
"""

from __future__ import annotations

import math
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

NEED = ("be.dat", "ob.ascii", "fg", "rand_obs_error", "unpert_obs.0000", "gts_omb_oma_01.0000", "tool.out")
GTS_FIELDS = {"synop": 5, "metar": 5, "ships": 5, "buoy": 5, "sonde_sfc": 5, "geoamv": 2, "gpspw": 1, "sound": 4,
              "airep": 4, "pilot": 2, "satem": 1, "qscat": 2, "profiler": 2, "gpsref": 1}


_SUB = ("", "des-unpert", "des-pert", "des-tool")


def _find(d: Path, name: str) -> Path:
    for sub in _SUB:
        if (d / sub / name).is_file():
            return d / sub / name
    raise FileNotFoundError(name)


def _dir() -> Path:
    d = os.environ.get("WOOF_WRFDA471_DESROZIERS_DIR")
    if d:
        try:
            for n in NEED:
                _find(Path(d), n)
        except FileNotFoundError:
            d = None
    if not d:
        pytest.skip("WOOF_WRFDA471_DESROZIERS_DIR with the tutorial and the randomised-trace files is not set")
    return Path(d)


@lru_cache(maxsize=1)
def _run(d: str):
    from woof.da.var import desroziers as D

    p = Path(d)
    return D.run(_find(p, "be.dat"), _find(p, "ob.ascii"), _find(p, "fg"), _find(p, "rand_obs_error"),
                 check_state=True)


def test_noise_and_obs_space_readers_on_a_toy_file(tmp_path):
    from woof.da.var import desroziers as D

    f = tmp_path / "rand"
    f.write_text("               synop       1\n       1\n"
                 + "       1       1" + "".join(f"{x:15.7E}" for x in (1.5, 0.25, -888888.0, -888888.0, 2.0, -1.0,
                                                                         3.0, 0.5, 4.0, 2.0)) + "\n")
    err, z = D.read_rand_obs_error(f)["synop"]
    assert err.tolist() == [[1.5, -888888.0, 2.0, 3.0, 4.0]] and z.shape == (1, 5)
    assert z[0, 0] == 0.25 and np.isnan(z[0, 1]) and z[0, 2] == -1.0 and z[0, 4] == 2.0
    g = tmp_path / "y"
    g.write_text("               synop       1\n       1\n"
                 + "       1       1" + "".join(f"{x:15.7E}" for x in (1.0, -888888.0, 3.0, 4.0, 5.0, -999999.9, -999999.9)) + "\n")
    y = D.read_obs_space_file(g)["synop"]
    assert y.shape == (1, 5) and np.isnan(y[0, 1]) and y[0, 4] == 5.0


def test_table_is_da_tune_obs_desroziers():
    d = _dir()
    from woof.da.var import desroziers as D

    res, _, _, _, _ = _run(str(d))
    mine = D.format_report(res).split("\n")
    tool = _find(d, "tool.out").read_text().split("\n")

    def rows(lines):
        out = {}
        for s in lines:
            m = re.match(r"^ (\S+ ?) obs, (Jo \(expected\)=|Jo \(actual\)  =|Error Factor =)\s*(\d+)\s+(\d+)(.*)$", s)
            if m:
                out[(m.group(1).strip(), m.group(2))] = (int(m.group(3)), int(m.group(4)),
                                                         [float(x) for x in m.group(5).split()])
        return out

    a, b = rows(mine), rows(tool)
    assert a.keys() == b.keys() and len(a) == 3 * 14
    assert len(a) > 0
    worst = 0.0
    for k in a:
        assert a[k][:2] == b[k][:2], k
        assert len(a[k][2]) == len(b[k][2]) == 5, k
        for x, y in zip(a[k][2], b[k][2]):
            worst = max(worst, abs(x - y))
            assert abs(x - y) <= 3e-5 + 1e-6 * abs(y), (k, x, y)

    def total(lines, key):
        return next(float(s.split("=")[1]) for s in lines if s.strip().startswith(key))

    for key in ("Total number of obs.", "Total J (actual)", "Total J (expected)", "Total Jo(actual)",
                "Total Jo(expected)", "Total Jo factor", "Total Jb(actual)", "Total Jb(expected)"):
        x, y = total(mine, key), total(tool, key)
        assert abs(x - y) <= 3e-5 + 1e-6 * abs(y), (key, x, y)
    print(f"Desroziers table vs da_tune_obs_desroziers: worst |difference| {worst:.1e} over {sum(len(v[2]) for v in a.values())} numbers")


def test_obs_space_increment_is_wrfdas_unpert_obs():
    d = _dir()
    from woof.da.var import desroziers as D

    _, _, term, y, _ = _run(str(d))
    theirs = D.read_obs_space_file(_find(d, "unpert_obs.0000"))
    worst = 0.0
    for n, v, a, b in term.slots:
        if b == a:
            continue
        t = theirs[n][:, D.FILE_FIELDS[n].index(v)]
        ok = np.isfinite(t)
        assert ok.sum() == term.mask[a:b].sum(), (n, v)
        assert np.array_equal(ok, term.mask[a:b]), (n, v)
        if not ok.any():
            continue
        scale = max(float(np.max(np.abs(t[ok]))), 1e-30)
        worst = max(worst, float(np.max(np.abs(y[a:b][ok] - t[ok]))) / scale)
    print(f"H'(xhat) vs unpert_obs: worst |difference| / max|y| = {worst:.1e}")
    assert worst < 2e-6


def _gts(path: Path) -> dict:
    """``{type: {k: (omb, oma, err, qc)}}`` from ``gts_omb_oma``: the last ``5 nf`` numbers of each level line."""
    out = {}
    lines = path.read_text().split("\n")
    i = 0
    while i < len(lines):
        m = re.match(r"^\s*([a-z_]+)\s+(\d+)\s*$", lines[i])
        if not m or m.group(1) not in GTS_FIELDS:
            i += 1
            continue
        name, n, nf = m.group(1), int(m.group(2)), GTS_FIELDS[m.group(1)]
        i += 1
        rows = []
        for _ in range(n):
            lev = int(lines[i].split()[0])
            i += 1
            for k in range(lev):
                tok = lines[i].split()
                t = tok[len(tok) - 5 * nf - (name == "airep"):][:5 * nf]  # airep ends with its level's zk
                rows.append([float(x) for x in t])
                i += 1
        a = np.asarray(rows).reshape(len(rows), nf, 5)
        out[name] = a  # (ob, inv, qc, error, re)
    return out


def test_classic_statistics_are_those_of_wrfdas_omb_oma():
    d = _dir()
    from woof.da.var import desroziers as D

    _, cl, term, _, _ = _run(str(d))
    assert len(cl) > 0
    gts = _gts(_find(d, "gts_omb_oma_01.0000"))
    assert {n for n, _ in cl} == {n for n in gts if len(gts[n]) and (gts[n][:, :, 2] >= 0).any()}
    worst = 0.0
    for (n, v), row in cl.items():
        k = D.FILE_FIELDS[n].index(v) if n != "airep" else ("u", "v", "t", "q").index(v)
        a = gts[n][:, k, :]
        m = a[:, 2] >= 0
        assert m.sum() == row["n"], (n, v)
        omb, oma, err = a[m, 1], a[m, 4], a[m, 3]
        ref = {"omb2": np.mean(omb * omb), "sigma_o2": np.mean(err * err), "so2_est": np.mean(oma * omb),
               "hbht_est": np.mean((omb - oma) * omb)}
        for key, val in ref.items():
            # the file prints seven decimals: a quadratic mean of values near 1e-3 (q) is good to 2e-7 * rms
            tol = 2e-5 * abs(val) + 2e-7 * math.sqrt(abs(val))
            worst = max(worst, abs(row[key] - val) / tol)
            assert abs(row[key] - val) <= tol, (n, v, key, row[key], val)
    print(f"classic Desroziers statistics vs WRFDA's gts_omb_oma: worst difference {worst:.2f} of the tolerance over {len(cl)} (type, variable) rows")
