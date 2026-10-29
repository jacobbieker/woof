"""RRTMG shortwave per-band and per-g-point output (rrtmg_spectral_output).

The option adds separate launches over the spectral buffers the broadband
sum reads (kernels/rrtmg_sw.cu rsw_spc_band_accum_b / rsw_spc_gpt_b).  So:

* the broadband outputs are byte-identical with the option on and off;
* a g-point output is zincflx(iw) * flux, one rounded product;
* a band output is its g-points summed in ascending iw, each sum rounded:
  WRF's own (inactive) per-band accumulation in spcvmc_sw.  It is checked
  here against the g-point output summed the same way in binary32.
The words against compiled WRF are held by the spectral oracle.
"""

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]

BROADBAND = ("swuflx", "swdflx", "swuflxc", "swdflxc", "swdkdir", "swdkdirc",
             "sibvisdir", "sibnirdir", "swhr", "swhrc")


def _run(spectral):
    import test_rrtmg_sw_cuda as T
    engine = T.cuda()
    engine.spectral_output = spectral
    try:
        groups = T._deck_groups()
        return [T._run_batched(cs, T._group_inputs(cs)) for cs in groups.values()]
    finally:
        engine.spectral_output = "none"


@pytest.fixture(scope="module")
def runs():
    return {mode: _run(mode) for mode in ("none", "bands", "gpoints")}


def test_broadband_words_do_not_depend_on_the_option(runs):
    for mode in ("bands", "gpoints"):
        for base, other in zip(runs["none"], runs[mode]):
            for key in BROADBAND:
                assert np.array_equal(base[key].view(np.uint32), other[key].view(np.uint32)), (mode, key)


def test_bands_are_their_gpoints_summed_in_ascending_order(runs):
    from woof.core import rrtmg_sw as sw
    ngb = _ngb()
    for bands, gpts in zip(runs["bands"], runs["gpoints"]):
        band = bands["sw_band"]
        gpt = gpts["sw_gpt"]
        assert band.shape[-1] == sw.NBNDSW and gpt.shape[-1] == sw.NGPTSW
        expect = np.zeros(band.shape, np.float32)
        for iw in range(sw.NGPTSW):
            b = int(ngb[iw]) - 16
            expect[..., b] = (expect[..., b] + gpt[..., iw]).astype(np.float32)
        assert np.array_equal(band.view(np.uint32), expect.view(np.uint32))


def test_bands_add_up_to_the_broadband_flux(runs):
    from woof.core.rrtmg_sw import SPECTRAL_QUANTITIES
    names = dict(zip(SPECTRAL_QUANTITIES, ("swuflx", "swdflx", "swuflxc", "swdflxc", "swdkdir", "swdkdirc")))
    for out in runs["bands"]:
        for q, key in enumerate(SPECTRAL_QUANTITIES):
            total = out["sw_band"][q].astype(np.float64).sum(axis=-1)
            np.testing.assert_allclose(total, out[names[key]], rtol=2e-6, atol=1e-4)


def _ngb():
    import test_rrtmg_sw_cuda as T
    return T.cp.asnumpy(T.cuda().ngb_gpu)


def _oracle_runs():
    """Our band and g-point slabs for every day case the spectral oracle
    recorded, keyed by case."""
    import test_rrtmg_sw_cuda as T
    out = {}
    engine = T.cuda()
    for mode, key in (("bands", "sw_band"), ("gpoints", "sw_gpt")):
        engine.spectral_output = mode
        try:
            for cs in T._deck_groups().values():
                result = T._run_batched(cs, T._group_inputs(cs))
                for i, case in enumerate(cs):
                    out.setdefault(case, {})[key] = result[key][:, i]
        finally:
            engine.spectral_output = "none"
    return out


def test_band_and_gpoint_words_equal_compiled_wrf():
    """Every band and g-point word against WRF v4.6.1's spcvmc_sw with its
    inactive per-band accumulation made live (tools/rrtmg_wrf461_oracle/
    spectral_patch.py; tests/data/oracles/rrtmg_spectral)."""
    from pathlib import Path
    path = Path(__file__).resolve().parent / "data" / "oracles" / "rrtmg_spectral" / "sw_spectral.npz"
    oracle = np.load(path)
    cases = sorted({k.split("/")[0] for k in oracle.files})
    ours = _oracle_runs()
    compared = 0
    for case in cases:
        if case not in ours:
            continue
        for name, key in (("band", "sw_band"), ("gpt", "sw_gpt")):
            want = np.moveaxis(oracle[f"{case}/{name}"], -1, 0)      # (6, n1, width)
            got = ours[case][key]
            assert got.shape == want.shape, (case, name, got.shape, want.shape)
            bad = got.view(np.uint32) != want.view(np.uint32)
            assert not bad.any(), (case, name, int(bad.sum()), np.argwhere(bad)[:4].tolist())
        compared += 1
    assert compared >= 20, f"only {compared} oracle cases matched a day case"


# ---------------------------------------------------------------------------
# Longwave: per-band fluxes (rrtmg_lw_zbatched.cu rlw_rtrn_band_rows)
# against WRF v4.6.1 rtrnmc's band terms times fluxfac, on the synthetic
# oracle columns (tests/data/oracles/rrtmg_spectral/lw_spectral.npz).
# ---------------------------------------------------------------------------

_LW_COL = ("play", "plev", "tlay", "tlev", "tsfc", "h2ovmr", "o3vmr", "co2vmr", "ch4vmr",
           "n2ovmr", "o2vmr", "cfc11vmr", "cfc12vmr", "cfc22vmr", "ccl4vmr", "emis",
           "reicmcl", "relqmcl", "resnmcl", "tauaer")
_LW_MCICA = ("cldfmcl", "taucmcl", "ciwpmcl", "clwpmcl", "cswpmcl")


def _lw_oracle():
    from pathlib import Path
    path = Path(__file__).resolve().parent / "data" / "oracles" / "rrtmg_spectral" / "lw_spectral.npz"
    data = np.load(path)
    cases = {}
    for key in data.files:
        case, rest = key.split("/", 1)
        cases.setdefault(case, {})[rest] = data[key]
    return cases


def _lw_run(cases, names, spectral):
    from woof.core import rrtmg_lw
    from woof.core.rrtmg_legacy import _lw_coeffs
    fx0 = cases[names[0]]
    ins = {k: np.concatenate([np.asarray(cases[n][f"in/{k}"]) for n in names], axis=0) for k in _LW_COL}
    for k in _LW_MCICA:
        ins[k] = np.concatenate([np.asarray(cases[n][f"in/{k}"]) for n in names], axis=1)
    out = rrtmg_lw.gpu_rrtmg_lw_batched_device(
        len(names), int(fx0["meta/nlayers"]), int(fx0["in/icld"]), ins["play"], ins["plev"],
        ins["tlay"], ins["tlev"], ins["tsfc"], ins["h2ovmr"], ins["o3vmr"], ins["co2vmr"],
        ins["ch4vmr"], ins["n2ovmr"], ins["o2vmr"], ins["cfc11vmr"], ins["cfc12vmr"],
        ins["cfc22vmr"], ins["ccl4vmr"], ins["emis"], int(fx0["in/inflglw"]),
        int(fx0["in/iceflglw"]), int(fx0["in/liqflglw"]), ins["cldfmcl"], ins["taucmcl"],
        ins["ciwpmcl"], ins["clwpmcl"], ins["cswpmcl"], ins["reicmcl"], ins["relqmcl"],
        ins["resnmcl"], ins["tauaer"], _lw_coeffs(), spectral_output=spectral)
    import cupy as cp
    return {k: cp.asnumpy(v) for k, v in out.items()}


def test_lw_band_words_equal_compiled_wrf_and_broadband_is_unchanged():
    cases = _lw_oracle()
    groups = {}
    for name, fx in sorted(cases.items()):
        key = (int(fx["meta/nlayers"]), int(fx["in/icld"]), int(fx["in/inflglw"]),
               int(fx["in/iceflglw"]), int(fx["in/liqflglw"]))
        groups.setdefault(key, []).append(name)
    compared = 0
    for names in groups.values():
        plain = _lw_run(cases, names, "none")
        bands = _lw_run(cases, names, "bands")
        for key in ("uflx", "dflx", "uflxc", "dflxc"):
            assert np.array_equal(plain[key].view(np.uint32), bands[key].view(np.uint32)), key
            want = np.concatenate([cases[n][f"out/{key}"] for n in names], axis=0)
            assert np.array_equal(bands[key].view(np.uint32), np.asarray(want, np.float32).view(np.uint32)), key
        for i, name in enumerate(names):
            want = np.moveaxis(cases[name]["out/lw_band"], -1, 0)        # (4, nl+1, 16)
            got = bands["lw_band"][:, i]
            bad = got.view(np.uint32) != np.asarray(want, np.float32).view(np.uint32)
            assert not bad.any(), (name, int(bad.sum()), np.argwhere(bad)[:4].tolist())
            compared += 1
    assert compared == len(cases)
