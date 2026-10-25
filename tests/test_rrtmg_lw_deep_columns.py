"""Legacy RRTMG LW past 128 layers: the per-run ``RLW_MAXLAY`` tier.

``rlw_rtrn_march`` keeps four per-thread float columns of ``RLW_MAXLAY``
entries; until 2026-10 that bound was a fixed 128 and the adapter refused
any deeper radiation column.  It is now compiled per run at
``woof.physics_vertical_contract.legacy_longwave_layer_tier`` of the run's
layer count (WRF's model layers plus its 4 hPa buffer layers above
``p_top``), so a 256-level model runs at any model top.  These gates run
the actual device chain on columns built by WRF's own wrapper statements
(:func:`woof.core.rrtmg_legacy_prep.lwrad_prep`) from the captured WRF
4.7.1 native column in ``tests/data/wrf471_lw_extension`` stretched to the
requested depth, and hold it to the NumPy port bitwise (max_ulp 0, the
discipline every LW gate uses).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from woof.core import rrtmg_legacy_prep as prep
from woof.physics_vertical_contract import legacy_longwave_layer_tier

F = np.float32
NATIVE = (Path(__file__).parent / "data" / "wrf471_lw_extension"
          / "native_profile_50hpa.npz")
OUT_KEYS = ("uflx", "dflx", "hr", "uflxc", "dflxc", "hrc")


def deep_column(nz, p_top, *, shift=0.0, cloud=True):
    """``lwrad_prep``'s dummies for one ``nz``-level column topped at
    ``p_top`` [Pa]: the captured WRF column's profiles interpolated in
    log-pressure onto ``nz`` layers (log-uniform interfaces from its
    surface pressure to ``p_top``), a stratosphere warming to 265 K at
    1 hPa above its 50 hPa top, and (``cloud``) a liquid deck at
    850-700 hPa plus an ice layer at 300-200 hPa."""
    with np.load(NATIVE) as data:
        native = {name.removeprefix("input__"): data[name][()]
                  if data[name].ndim == 0 else data[name].copy()
                  for name in data.files if name.startswith("input__")}
    lnp_w = np.log(native["p8w"].astype(np.float64))
    psfc = float(native["p8w"][0])
    p8w = np.exp(np.linspace(np.log(psfc), np.log(p_top), nz + 1))
    p3d = 0.5 * (p8w[:-1] + p8w[1:])

    def profile(p, values, lnp, top_value):
        # np.interp needs increasing abscissae: interpolate in -ln p.
        inside = np.interp(-np.log(p), -lnp, values)
        above = np.log(p) < lnp[-1]
        frac = np.clip((lnp[-1] - np.log(p)) / (lnp[-1] - np.log(100.0)),
                       0.0, 1.0)
        return np.where(above, values[-1] + frac * (top_value - values[-1]),
                        inside)

    t8w = profile(p8w, native["t8w"].astype(np.float64), lnp_w, 265.0) + shift
    lnp_m = np.log(native["p3d"].astype(np.float64))
    t3d = profile(p3d, native["t3d"].astype(np.float64), lnp_m, 265.0) + shift
    lnq = profile(p3d, np.log(native["qv3d"].astype(np.float64)), lnp_m,
                  np.log(3.0e-6))
    o3 = profile(p3d, native["o33d"].astype(np.float64), lnp_m, 4.0e-6)
    tv = t3d * (1.0 + 0.608 * np.exp(lnq))
    dz8w = 287.04 * tv / 9.81 * np.log(p8w[:-1] / p8w[1:])
    zeros = np.zeros(nz)
    qc, qi, qs, cf = zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()
    if cloud:
        deck = (p3d <= 85000.0) & (p3d >= 70000.0)
        cf[deck], qc[deck] = 0.8, 2.0e-4
        cirrus = (p3d <= 30000.0) & (p3d >= 20000.0)
        cf[cirrus], qi[cirrus], qs[cirrus] = 0.6, 3.0e-5, 1.0e-5
    f = lambda a: np.asarray(a, dtype=F)  # noqa: E731
    column = dict(native)
    column.update(
        p3d=f(p3d), p8w=f(p8w), t3d=f(t3d), t8w=f(t8w), dz8w=f(dz8w),
        qv3d=f(np.exp(lnq)), qc3d=f(qc), qr3d=f(zeros), qi3d=f(qi),
        qs3d=f(qs), qg3d=f(zeros), cldfra3d=f(cf), o33d=f(o3),
        re_cloud=f(np.full(nz, 1.0e-5)), re_ice=f(np.full(nz, 3.0e-5)),
        re_snow=f(np.full(nz, 1.0e-4)),
        nlayers=prep.compute_lw_nlayers(nz + 1, p_top))
    return column


#: The batched engine's arguments, in its order (McICA slabs
#: (NGPTLW, ncol, nlay), every other array ncol-major).
ENGINE_KEYS = ("ncol", "nlay", "icld", "play", "plev", "tlay", "tlev",
               "tsfc", "h2ovmr", "o3vmr", "co2vmr", "ch4vmr", "n2ovmr",
               "o2vmr", "cfc11vmr", "cfc12vmr", "cfc22vmr", "ccl4vmr",
               "emis", "inflglw", "iceflglw", "liqflglw", "cldfmcl",
               "taucmcl", "ciwpmcl", "clwpmcl", "cswpmcl", "reicmcl",
               "relqmcl", "resnmcl", "tauaer")
MCICA_KEYS = ("cldfmcl", "taucmcl", "ciwpmcl", "clwpmcl", "cswpmcl")


def engine_inputs(columns):
    """WRF's wrapper (``lwrad_prep_batch``, bitwise per column with the
    scalar ``lwrad_prep``) over ``columns``: the engine's argument dict."""
    batch = {}
    for name, value in columns[0].items():
        if isinstance(value, np.ndarray) and value.ndim:
            batch[name] = np.stack([c[name] for c in columns])
        elif name in ("tsk", "emiss", "xland", "xice", "snow", "xlat"):
            batch[name] = np.asarray([c[name] for c in columns], F)
        else:
            batch[name] = value
    out = prep.lwrad_prep_batch(**batch)
    return {k: out[k] for k in ENGINE_KEYS}


def _numpy_reference(ins, C):
    from woof.core import rrtmg_lw as lw
    return lw.rrtmg_lw(
        ins["ncol"], ins["nlay"], ins["icld"], ins["play"], ins["plev"],
        ins["tlay"], ins["tlev"], ins["tsfc"], ins["h2ovmr"], ins["o3vmr"],
        ins["co2vmr"], ins["ch4vmr"], ins["n2ovmr"], ins["o2vmr"],
        ins["cfc11vmr"], ins["cfc12vmr"], ins["cfc22vmr"], ins["ccl4vmr"],
        ins["emis"], ins["inflglw"], ins["iceflglw"], ins["liqflglw"],
        ins["cldfmcl"], ins["taucmcl"], ins["ciwpmcl"], ins["clwpmcl"],
        ins["cswpmcl"], ins["reicmcl"], ins["relqmcl"], ins["resnmcl"],
        ins["tauaer"], C)


def _assert_words(got, want, what):
    got = np.ascontiguousarray(np.asarray(got, F))
    want = np.ascontiguousarray(np.asarray(want, F))
    assert got.shape == want.shape, what
    bad = got.view(np.uint32) != want.view(np.uint32)
    assert not bad.any(), (
        f"{what}: {int(bad.sum())}/{bad.size} words differ, first at "
        f"{np.argwhere(bad)[0].tolist()}")


@pytest.mark.parametrize(("nz", "p_top", "nlayers"),
                         ((256, 100.0, 257), (150, 5000.0, 163),
                          (256, 5000.0, 269)))
def test_deep_columns_build_wrf_buffer_layers(nz, p_top, nlayers):
    """CPU leg: WRF's count (kme + nint(p_top / 4 hPa) - 1) and grid hold
    at depth.  At the 1 hPa top WRF's rounded extension is zero, and
    upstream 1.1.0's compute_lw_nlayers keeps one clear buffer layer there
    (max(1, n)), so a 256-level column is 257 longwave layers."""
    column = deep_column(nz, p_top)
    assert column["nlayers"] == nlayers
    out = prep.lwrad_prep(**column)
    assert out["nlay"] == nlayers and out["plev"].shape == (nlayers + 1,)
    assert out["plev"][-1] == 0.0
    assert np.all(np.diff(out["plev"]) < 0)
    assert np.all(out["play"] > 0) and np.isfinite(out["tlay"]).all()
    # Model interfaces below the top stay the model's own (hPa).
    np.testing.assert_array_equal(
        out["plev"][:nz], (column["p8w"][:nz] / F(100)).astype(F))


@pytest.fixture(scope="module")
def C():
    from woof.core.rrtmg_legacy import _lw_coeffs
    return _lw_coeffs()


@pytest.mark.gpu
@pytest.mark.parametrize("maxlay", (128, 192, 256, 320))
def test_compiled_tier_frames_are_priced_per_layer(maxlay):
    """rlw_rtrn_march (and its coalesced twin) at RLW_MAXLAY = maxlay is
    the unit's widest frame and never exceeds what preflight charges for
    that tier (16 B per layer; measured exactly so on sm_86 at NVRTC 12.9
    and 13.4).  Under-pricing is the direction that hurts."""
    pytest.importorskip("cupy")
    from woof.core import preflight as pf
    from woof.core import rrtmg_lw as lw
    frames = lw.gpu_local_frame_bytes(maxlay)
    priced = pf.rrtmg_lw_legacy_chain_frame_bytes(maxlay)
    assert priced == 16 * maxlay
    # The inline march (upstream 1.1.0) carries the same four per-thread
    # columns as the coalesced march it fuses taut/odcld/efclfrac into.
    marches = ("rlw_rtrn_march", "rlw_rtrn_march_inline")
    assert max(frames.values()) == frames["rlw_rtrn_march"] <= priced
    assert frames.get("rlw_rtrn_march_inline", 0) <= priced
    assert all(v <= 512 for k, v in frames.items() if k not in marches)


@pytest.mark.gpu
@pytest.mark.parametrize(("nz", "p_top"), ((256, 100.0), (150, 5000.0)))
def test_deep_column_device_chain_matches_numpy_bitwise(C, nz, p_top):
    """The 256-level / 1 hPa column (257 radiation layers, tier 320) and a
    150-level / 50 hPa column (163 layers, tier 192): the single-column and
    batched device chains equal the NumPy port word for word, twice."""
    cp = pytest.importorskip("cupy")
    from woof.core import rrtmg_lw as lw
    columns = [deep_column(nz, p_top),
               deep_column(nz, p_top, shift=-3.0, cloud=False),
               deep_column(nz, p_top, shift=2.0)]
    ins = engine_inputs(columns)
    assert ins["nlay"] > 128
    assert legacy_longwave_layer_tier(ins["nlay"]) in (192, 256, 320)
    want = _numpy_reference(ins, C)
    for key in OUT_KEYS:
        assert np.isfinite(want[key]).all(), key
    first = dict(ins, ncol=1)
    for k, v in ins.items():
        if isinstance(v, np.ndarray):
            first[k] = v[:, :1] if k in MCICA_KEYS else v[:1]
    for run in range(2):
        single = lw.gpu_rrtmg_lw(**first, C=C)
        for key in OUT_KEYS:
            _assert_words(single[key], want[key][:1], f"single {key} #{run}")
        for chunk in (None, 2):
            batched = lw.gpu_rrtmg_lw_batched(**ins, C=C,
                                              column_chunk=chunk)
            for key in OUT_KEYS:
                _assert_words(batched[key], want[key],
                              f"batched[{chunk}] {key} #{run}")
            # Device-resident inputs take the same path to the same words.
            dev = {k: cp.asarray(v) if isinstance(v, np.ndarray) else v
                   for k, v in ins.items()}
            resident = lw.lw_batched_to_host(lw.gpu_rrtmg_lw_batched_device(
                **dev, C=C, column_chunk=chunk))
            for key in OUT_KEYS:
                _assert_words(resident[key], want[key],
                              f"resident[{chunk}] {key} #{run}")


@pytest.mark.gpu
def test_tier_is_address_space_only_at_128_layers(C, monkeypatch):
    """A column of at most 128 layers runs the shipped unit; compiling the
    same column into a deeper tier changes no output word (the bound sizes
    the per-thread arrays and nothing else)."""
    pytest.importorskip("cupy")
    from woof.core import rrtmg_lw as lw
    ins = engine_inputs([deep_column(115, 5000.0),
                         deep_column(115, 5000.0, shift=1.5, cloud=False)])
    assert ins["nlay"] == 128 and legacy_longwave_layer_tier(128) == 128
    shipped = lw.gpu_rrtmg_lw_batched(**ins, C=C)
    monkeypatch.setattr(lw, "rlw_maxlay_tier", lambda n: 192)
    deeper = lw.gpu_rrtmg_lw_batched(**ins, C=C)
    for key in shipped:
        _assert_words(deeper[key], shipped[key], key)
