"""FARMS (WRF swint_opt = 2) against compiled WRF v4.7.1, word for word.

The fixture tests/data/oracles/farms/farms_wrf471.npz is
tools/farms_wrf471_oracle's: WRF v4.7.1 phys/module_ra_farms.F compiled
unmodified (gfortran -O0 -ffp-contract=off), one Farms_driver call per
scenario, its binary32 libm bound to woof's word-exact ports; the reference
is byte-identical built on macOS arm64 and Linux x86-64 (provenance.json).
Every output word of woof/core/kernels/farms.cu must equal it (max ULP 0).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data" / "oracles" / "farms"
OUTPUTS = ("swdown2", "swddir2", "swddni2", "swddif2", "swdownc2",
           "swddnic2")


@pytest.fixture(scope="module")
def oracle():
    path = DATA / "farms_wrf471.npz"
    provenance = json.loads((DATA / "provenance.json").read_text())
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    assert got == provenance["npz_sha256"], "FARMS fixture differs from its provenance"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}, provenance


def _scenarios(data):
    return [str(s) for s in data["scenarios"]]


def test_provenance_names_unmodified_wrf_and_the_flags(oracle):
    _, provenance = oracle
    hosts = []
    for build in provenance["builds"]:
        text = "\n".join(build["receipt"])
        assert "f52c197ed39d12e087d02c50f412d90d418f6186" in text
        assert ("1b9e060d71dd7218c39e5a5724a7886307507a7cb85d7db33cd3ed52c0034a60"
                in text), "module_ra_farms.F is not the pinned v4.7.1 file"
        assert "-O0 -ffp-contract=off" in text
        assert "acosf cosf expf log10f powf sinf sinhf" in text
        hosts.append(next(l for l in build["receipt"] if l.startswith("host:")))
    assert any("Darwin" in h for h in hosts) and any("Linux" in h for h in hosts)


def test_fixture_reaches_the_sky_classes(oracle):
    data, _ = oracle
    for scen in _scenarios(data):
        cz = data[f"{scen}/coszen_loc"]
        qc, qi, qs = (data[f"{scen}/{n}"][:-1] for n in ("qc", "qi", "qs"))
        aod = data[f"{scen}/aod5502d"]
        out = data[f"{scen}/out/swdown2"]
        day = cz > 0
        assert (~day).sum() > 50 and (out[~day] == 0).all()
        liquid = (qc[:-1] > 0).any(axis=0)
        ice = (qi[:-1] > 0).any(axis=0)
        snow = (qs[:-1] > 0).any(axis=0)
        for mask in (~liquid & ~ice & ~snow, liquid & ~ice, ice & ~liquid,
                     snow, liquid & ice):
            assert (mask & day).sum() > 20
        assert ((aod > 0) & day).sum() > 200 and ((aod == 0) & day).sum() > 200
        assert ((cz > 0) & (cz < 0.05)).sum() > 30 and (cz > 0.9).sum() > 30
        # the cloud-fraction blend: GHI strictly between all-sky and clear
        assert (day & (out != data[f"{scen}/out/swdownc2"])).sum() > 500


@pytest.mark.parametrize("scen", ["req1_summer", "req0_jan", "req1_dec",
                                  "req0_feb"])
def test_kernel_matches_wrf471_word_for_word(oracle, scen):
    import cupy as cp
    from woof.core.farms import farms_columns, wrf_juday

    data, _ = oracle
    g = lambda n: data[f"{scen}/{n}"]
    nz = g("rho").shape[0] - 1   # the fixture carries kme = nz + 1 rows
    dev3 = lambda n: cp.asarray(np.ascontiguousarray(g(n)[:nz]))
    dev2 = lambda n: cp.asarray(g(n))
    rc, ri, rs, _aer = (int(v) for v in g("scalars"))
    out = farms_columns(
        p8w_sfc=cp.asarray(np.ascontiguousarray(g("p8w")[0])),
        albedo=dev2("albedo"), coszen=dev2("coszen_loc"),
        aerssa=dev2("aerssa2d"), aerasy=dev2("aerasy2d"),
        aod550=dev2("aod5502d"), angexp=dev2("angexp2d"),
        rho=dev3("rho"), dz8w=dev3("dz8w"), qv=dev3("qv"), qc=dev3("qc"),
        qi=dev3("qi"), qs=dev3("qs"), re_cloud=dev3("re_cloud"),
        re_ice=dev3("re_ice"), re_snow=dev3("re_snow"), cldfra=dev3("cldfra"),
        has_req=(rc, ri, rs), juday=wrf_juday(g("julian")[0]))
    bad = {}
    for name in OUTPUTS:
        got = cp.asnumpy(out[name]).view(np.uint32)
        want = g(f"out/{name}").view(np.uint32)
        if (got != want).any():
            i = np.flatnonzero(got != want)
            bad[name] = (i.size, int(i[0]))
    assert not bad, f"{scen}: words differing from WRF (count, first column): {bad}"
    # Farms_driver never writes its intent(inout) aerosol rows
    for name in ("aerssa2d", "aerasy2d", "aod5502d", "angexp2d"):
        assert (g(f"out/{name}").view(np.uint32)
                == g(name).view(np.uint32)).all()
