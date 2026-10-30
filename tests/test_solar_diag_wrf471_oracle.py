"""solar_diagnostics = 1 (WRF-Solar module_diag_solar.F) against compiled WRF
v4.7.1, word for word.

tests/data/oracles/solar_diag is tools/solar_diag_wrf471_oracle's record:
the unmodified solar_diag at gfortran -O0 -ffp-contract=off against glibc
2.43's libm, over WRF-shaped columns and MPAS-shaped ones (the inputs
woof/hex/mpasphys/solar_diagnostics_v842.py forms from MPAS fields).  Every
one of the 26 outputs of every column must be the same word; an output WRF
leaves unassigned must keep the word it held.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from woof.core.solar_diagnostics import NOUT, SOLAR_OUTPUTS, run_solar_diag

DATA = Path(__file__).parent / "data" / "oracles" / "solar_diag"


def _read():
    receipt = (DATA / "receipt.txt").read_text()
    for name in ("columns.bin", "columns_meta.json", "mpas_zgrid.bin", "solar_diag.out.bin"):
        digest = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        assert f"{digest}  {name}" in receipt, name
    raw = np.fromfile(DATA / "columns.bin", dtype=np.uint8)
    out = np.fromfile(DATA / "solar_diag.out.bin", dtype="<f4").reshape(-1, NOUT)
    meta = json.loads((DATA / "columns_meta.json").read_text())
    (ncol,) = raw[:4].view("<i4")
    pos = 4
    cols = []
    for c in range(ncol):
        nz = int(raw[pos:pos + 4].view("<i4")[0])
        flags = tuple(int(f) for f in raw[pos + 4:pos + 32].view("<i4"))
        dt, coszen, swdnb, swdnt, prior = raw[pos + 32:pos + 52].view("<f4")
        pos += 52
        arrays = raw[pos:pos + 48 * nz].view("<f4").reshape(12, nz)
        pos += 48 * nz
        ph = raw[pos:pos + 4 * (nz + 1)].view("<f4")
        pos += 4 * (nz + 1)
        phb = raw[pos:pos + 4 * (nz + 1)].view("<f4")
        pos += 4 * (nz + 1)
        cols.append(dict(nz=nz, flags=flags, dt=dt, coszen=coszen, swdnb=swdnb,
                         swdnt=swdnt, prior=prior, arrays=arrays, ph=ph, phb=phb,
                         want=out[c], meta=meta[c]))
    assert pos == raw.size and len(cols) == out.shape[0] == len(meta)
    zg = np.fromfile(DATA / "mpas_zgrid.bin", dtype="<f4")
    zpos = 0
    for c in cols:
        if c["meta"]["mpas"]:
            c["zgrid"] = zg[zpos:zpos + c["nz"] + 1]
            zpos += c["nz"] + 1
    assert zpos == zg.size
    return cols


@pytest.fixture(scope="module")
def columns():
    return _read()


def _flags(f):
    from woof.core.solar_diagnostics import species_flags
    return species_flags(qv=f[0], qc=f[1], qi=f[2], qs=f[3], has_reqc=f[4],
                         has_reqi=f[5], has_reqs=f[6])


def _run(sub):
    nz = sub[0]["nz"]
    ncol = len(sub)
    stack = lambda j: cp.asarray(np.ascontiguousarray(
        np.stack([c["arrays"][j] for c in sub], axis=1)))
    (rho, dz, qv, qc, qi, qs, qc_tot, qi_tot, rec, rei, res, cf) = (
        stack(j) for j in range(12))
    ph = cp.asarray(np.ascontiguousarray(np.stack([c["ph"] for c in sub], axis=1)))
    phb = cp.asarray(np.ascontiguousarray(np.stack([c["phb"] for c in sub], axis=1)))
    if sub[0]["meta"]["mpas"]:
        # the MPAS adapter's own inputs: zgrid, MPAS's moist rho_p and dz_p
        from woof.hex.mpasphys.solar_diagnostics_v842 import mpas_solar_inputs
        zgrid = cp.asarray(np.ascontiguousarray(np.stack([c["zgrid"] for c in sub], axis=1)))
        ph2, phb2, rho, dz = mpas_solar_inputs(cp, zgrid=zgrid, rho_p=rho, dz_p=dz)
        for a, b in ((ph2, ph), (phb2, phb)):
            assert np.array_equal(cp.asnumpy(a).view(np.uint32), cp.asnumpy(b).view(np.uint32))
        ph, phb = ph2, phb2
    surf = lambda k: cp.asarray(np.array([c[k] for c in sub], np.float32))
    out = cp.asarray(np.ascontiguousarray(np.broadcast_to(
        np.array([c["prior"] for c in sub], np.float32), (NOUT, ncol))))
    dts = {float(c["dt"]) for c in sub}
    assert len(dts) == 1
    run_solar_diag(nz=nz, ncol=ncol, dt=dts.pop(), flags=_flags(sub[0]["flags"]),
                   rho=rho, dz8w=dz, ph=ph, phb=phb, cldfra=cf,
                   coszen=surf("coszen"), swdnb=surf("swdnb"), swdnt=surf("swdnt"),
                   qv=qv, qc=qc, qi=qi, qs=qs, qc_tot=qc_tot, qi_tot=qi_tot,
                   re_cloud=rec, re_ice=rei, re_snow=res, out=out,
                   work=cp.empty((nz, ncol), cp.float32))
    return cp.asnumpy(out).T


def test_fixture_reaches_the_arms(columns):
    want = np.stack([c["want"] for c in columns])
    names = [n for _, n, _, _ in SOLAR_OUTPUTS]
    col = {n: want[:, i] for i, n in enumerate(names)}
    assert (col["CLRNIDX"] == -999).any() and (col["CLRNIDX"] > 0).any()
    assert (col["SZA"] >= 90).any() and (col["SZA"] < 90).any()
    for n in ("CBASEHT", "CTOPHT", "CBASEHT_TOT", "CTOPHT_TOT"):
        assert (col[n] == -999).any() and (col[n] > 0).any(), n
    assert (col["TAU_QI"] > 0).any() and (col["TAU_QC"] > 0).any()
    assert (col["RE_QC"] == 0).any()          # negative / zero path
    prior = np.array([c["prior"] for c in columns])
    assert ((col["RE_QC"] == prior) & (prior != 0)).any()   # left unassigned
    assert any(c["meta"]["mpas"] for c in columns)
    assert not np.array_equal(col["LWP"], col["LWP_TOT"])


@pytest.mark.parametrize("exact", ["0", "1"])
@pytest.mark.parametrize("shape", ["wrf", "mpas"])
def test_solar_diag_every_word(columns, monkeypatch, exact, shape):
    monkeypatch.setenv("GPUWM_WRF_EXACT", exact)
    groups = {}
    for c in columns:
        if c["meta"]["mpas"] != (shape == "mpas"):
            continue
        groups.setdefault((c["nz"], c["flags"], float(c["dt"])), []).append(c)
    total = 0
    for key, sub in groups.items():
        got = _run(sub)
        want = np.stack([c["want"] for c in sub])
        bad = np.argwhere(got.view(np.uint32) != want.view(np.uint32))
        assert bad.size == 0, (key, [(SOLAR_OUTPUTS[j][1], got[i, j], want[i, j])
                                     for i, j in bad[:6]])
        total += want.size
    assert total == NOUT * sum(c["meta"]["mpas"] == (shape == "mpas") for c in columns)
