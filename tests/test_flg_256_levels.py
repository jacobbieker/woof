"""WRF Fu-Liou-Gu (ra_lw_physics = ra_sw_physics = 7) on 256-layer columns.

WRF sizes FLG's level scratch by PARA_FILE's ``nvx = 100``. The generated
CUDA keeps that bound as ``FLG_NVX`` with the stock 100 as its ``#ifndef``
default, so a column of at most 100 layers compiles stock WRF's bound and a
101 to 256 layer column adds ``-D FLG_NVX=256`` (woof.core.flg_levels).
Claims:

1. The ladder, the contract bound, the config refusal, the registry row and
   every kernel source's own default agree, and the patch that defines the
   deep oracle is the one recorded (no GPU).
2. At nz in {32, 100} the column takes the stock tier, and forcing the 256
   tier reproduces every output word (raw bits); the component oracles at
   the 256 tier still reproduce their pristine-WRF fixture words.
3. Strict (GPUWM_WRF_EXACT=1): a 256-layer column matches WRF's
   module_ra_flg.F with only nvx raised to 256 (tools/flg_wrf461_oracle/
   nvx256.patch, built by build_nvx256.sh) word for word.
4. Any mode: the 256-layer column is finite, its layer heating closes the
   flux budget, it stays near the oracle, and with WOOF_FLG256_XBACKEND_DIR
   set each backend's outputs are written there and compared with the other
   backend's file when present.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "woof/core/kernels"
ORACLE = ROOT / "tools/flg_wrf461_oracle"
FIXTURE = Path(__file__).parent / "fixtures/flg/nvx256-columns.npz"
DEEP, SHIPPED = 256, 100
DEFAULT = "#ifndef FLG_NVX\n#define FLG_NVX 100\n#endif\n"
SOURCES = ("flg.cu", "flg_column_generated.cuh", "flg_gases_generated.cuh",
           "flg_streams_generated.cuh", "flg_workspace_generated.cuh")


def test_the_tier_ladder_contract_and_kernel_defaults_agree():
    from woof.core import flg_levels as lv
    from woof.physics_vertical_contract import FLG_VERTICAL_LEVEL_BOUNDS

    assert (lv.SHIPPED_NVX, lv.DEEP_NVX) == (SHIPPED, DEEP)
    assert FLG_VERTICAL_LEVEL_BOUNDS == lv.VERTICAL_LEVEL_BOUNDS == (4, DEEP)
    assert [lv.nvx_tier(n) for n in (4, 32, 100, 101, 256)] == [100, 100, 100, 256, 256]
    with pytest.raises(ValueError):
        lv.nvx_tier(257)
    assert lv.nvx_defines(100) == () and lv.nvx_defines(101) == (("FLG_NVX", 256),)
    with pytest.raises(ValueError):
        lv.tier_defines(128)
    for name in SOURCES:
        text = (KERNELS / name).read_text()
        assert text.count(DEFAULT) == 1, name
        # No level extent is a literal any more; aerosol strides included.
        assert not re.search(r"(?<!offsets)\[\(?10[01]\)?\]|\(1\*\(100\)\)|\(100\)\*\(18\)", text), name
    workspace = (KERNELS / "flg_workspace_generated.cuh").read_text()
    # The forecast column's per-level storage is the FLG_NV1 workspace; the
    # tier reaches only the naero>=1 aerosol code stock RAD_FLG never runs.
    uses = [line for line in workspace.splitlines() if "FLG_NVX" in line and not line.startswith("#")]
    assert uses and all("tae[" in line or "aprofs[" in line for line in uses)
    assert re.search(r"#define FLG_WORKSPACE_WORDS \(300\*FLG_NV1\+1\)", workspace)


def test_the_registry_row_is_the_contract():
    from woof.physics_vertical_contract import FLG_VERTICAL_LEVEL_BOUNDS

    registry = json.loads((ROOT / "woof/physics_registry_v2.json").read_text())
    row = registry["components"]["radiation"]["options"]["fu-liou-gu"]
    contract = row["extensions"]["column_contract"]
    assert contract["layers"] == list(FLG_VERTICAL_LEVEL_BOUNDS)
    assert contract["nvx_tiers"] == [SHIPPED, DEEP]


def test_the_config_admits_256_layers_and_refuses_257_and_3():
    from woof.config import ConfigurationRefused, RunConfig, validate_run_config

    def cfg(nz):
        return RunConfig(nx=8, ny=8, nz=nz, dx=1000.0, dy=1000.0,
                         ztop=30000.0, dt=1.0, run_seconds=10.0,
                         ra_physics=0, ra_lw_physics=7, ra_sw_physics=7)

    validate_run_config(cfg(SHIPPED + 1))
    validate_run_config(cfg(DEEP))
    with pytest.raises(ConfigurationRefused, match="WRF FLG needs 4 to 256"):
        validate_run_config(cfg(DEEP + 1))
    with pytest.raises(ConfigurationRefused, match="FLG|nz must"):
        validate_run_config(cfg(3))


@pytest.mark.parametrize("nz", [SHIPPED, DEEP])
def test_admission_prices_the_workspace_of_the_column_it_launches(nz):
    from woof.core.flg_radiation import flg_radiation_vram_bytes

    # Per column the arena is 300 x (nz + 1) + 1 words; admission's per-column
    # words add packing and carriers that are also affine in nz.
    per_column = (flg_radiation_vram_bytes(64, nz) - flg_radiation_vram_bytes(32, nz)) // 32
    assert per_column >= 4 * (300 * (nz + 1) + 1)
    assert per_column == 4 * (300 * (nz + 1) + 1 + (12 * nz + 25) + 40 + (11 * nz + 8) + 1
                              + 4 * nz + 17 + 36 * (nz + 1))


def test_the_deep_oracle_patch_is_the_recorded_one():
    patch = (ORACLE / "nvx256.patch").read_bytes()
    digest = hashlib.sha256(patch).hexdigest()
    removed = [l for l in patch.decode().splitlines() if l.startswith("-") and not l.startswith("---")]
    added = [l for l in patch.decode().splitlines() if l.startswith("+") and not l.startswith("+++")]
    assert removed == ["-   &                      nvx    = 100,      &"]
    assert added == ["+   &                      nvx    = 256,      &"]
    assert f"patch_sha={digest}" in (ORACLE / "build_nvx256.sh").read_text()
    provenance = (ROOT / "woof/data/wrf_radiation/flg/PROVENANCE.md").read_text()
    assert digest in provenance and patch.decode() in provenance
    receipt = json.loads(FIXTURE.with_suffix(".json").read_text())
    assert receipt["patch_sha256"] == digest
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == receipt["fixture_sha256"]
    assert receipt["layers"] == DEEP and receipt["patch_inert_words"] == {"24": 1632, "100": 6648}


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

def _deck(nz):
    from tools.flg_wrf461_oracle.make_nvx256_fixture import deck
    return deck(nz)


def _run(cp, fields, **kw):
    from woof.core.flg import FLGColumn

    n, nv = fields["ice"].shape
    column = FLGColumn(n, nv, **kw)
    out = column(**{k: cp.ascontiguousarray(cp.asarray(v, dtype=cp.float32)) for k, v in fields.items()})
    column.check_errors()
    return column, {k: np.ascontiguousarray(cp.asnumpy(v)).copy() for k, v in out.items()}


@pytest.mark.gpu
@pytest.mark.parametrize("nz", [32, SHIPPED])
def test_the_256_tier_reproduces_the_stock_tier(nz):
    import cupy as cp

    fields = _deck(nz)
    column, shipped = _run(cp, fields)
    assert column.nvx == SHIPPED, "nz <= 100 must take the stock nvx tier"
    assert np.any(shipped["dts"] != 0) and np.any(shipped["dtir"] != 0)
    deep_column, deep = _run(cp, fields, nvx=DEEP)
    assert deep_column.nvx == DEEP
    for name in shipped:
        np.testing.assert_array_equal(deep[name].view(np.uint32), shipped[name].view(np.uint32),
                                      err_msg=f"FLG_NVX 100 -> 256 at nz={nz}: {name}")


@pytest.mark.gpu
def test_component_oracles_at_the_256_tier_keep_pristine_wrf_words(monkeypatch):
    import cupy as cp

    from woof.core import flg

    monkeypatch.setattr(flg, "nvx_defines", lambda layers: (("FLG_NVX", DEEP),))
    root = Path(__file__).parent / "fixtures/flg"
    with np.load(root / "gas_helpers.npz") as f:
        got = cp.asnumpy(flg.continuum_depth(f["pressure"], f["temperature"], f["humidity"]))
        np.testing.assert_array_equal(got.view(np.uint32), f["continuum"].view(np.uint32))
    with np.load(root / "gas_driver.npz") as f:
        out = flg.gas_optical_depth(*[f[k] for k in ("pressure", "temperature", "humidity",
                                                     "ozone", "gas_values", "switches")])
        for key, value in out.items():
            np.testing.assert_array_equal(cp.asnumpy(value).view(np.uint32), f[key].view(np.uint32), err_msg=key)
    with np.load(root / "streams.npz") as f:
        keys = ("bands", "albedo", "coszen", "f0", "emissivity", "bf", "bs", "moments", "scattering", "depth")
        out = flg.stream_fluxes(*[f[k] for k in keys])
        for key, value in out.items():
            np.testing.assert_array_equal(cp.asnumpy(value).view(np.uint32), f[key].view(np.uint32), err_msg=key)


def _oracle():
    with np.load(FIXTURE) as f:
        fields = {k[3:]: f[k] for k in f.files if k.startswith("in_")}
        reference = {k[4:]: f[k] for k in f.files if k.startswith("out_")}
    return fields, reference


@pytest.mark.gpu
def test_a_256_layer_column_matches_the_patched_nvx_oracle_word_for_word():
    import cupy as cp

    from woof import wrf_exact

    if not wrf_exact.ENABLED:
        pytest.skip("word comparison with the Fortran oracle requires GPUWM_WRF_EXACT=1")
    fields, reference = _oracle()
    column, out = _run(cp, fields)
    assert column.nvx == DEEP
    assert sum(v.size for v in reference.values()) == 16944
    for name, expected in reference.items():
        misses = np.argwhere(out[name].view(np.uint32) != expected.view(np.uint32))
        assert not len(misses), f"{name}: {len(misses)} words differ; first {misses[:1].tolist()}"


HEATING = ("dts", "dtir", "dt")


def _watts(result, dp):
    """Fluxes as they are; heating (W/m2 per hPa) times its layer's hPa, the
    layer's absorbed W/m2. Near a 1 hPa top a layer is ~0.03 hPa thick, so
    a per-hPa limit would ask 30x more of those layers than of the rest."""
    return {k: v.astype(np.float64) * (dp if k in HEATING else 1.0) for k, v in result.items()}


def _check_xbackend(label, result, dp):
    directory = os.environ.get("WOOF_FLG256_XBACKEND_DIR")
    if not directory:
        return
    from woof import backend

    os.makedirs(directory, exist_ok=True)
    from woof import wrf_exact

    mode = "strict" if wrf_exact.ENABLED else "default"
    mine = f"{label}.{mode}.{backend.active()}.npz"
    np.savez(os.path.join(directory, mine), **result)
    for other in sorted(n for n in os.listdir(directory)
                        if n.startswith(f"{label}.{mode}.") and n.endswith(".npz") and n != mine):
        with np.load(os.path.join(directory, other)) as theirs:
            mine_w, theirs_w = _watts(result, dp), _watts({k: theirs[k] for k in result}, dp)
            for name in sorted(result):
                err = float(np.abs(mine_w[name] - theirs_w[name]).max())
                limit = 0.5  # W/m2
                assert err < limit, f"{label}/{name} vs {other}: {err:.3e}"


@pytest.mark.gpu
def test_a_256_layer_column_is_finite_conserves_energy_and_tracks_the_oracle():
    import cupy as cp

    fields, reference = _oracle()
    column, out = _run(cp, fields)
    assert column.nvx == DEEP
    for name, value in out.items():
        assert np.isfinite(value).all(), name
    mu = fields["coszen"].astype(np.float64)
    day = mu > 0
    # Night: no solar flux anywhere; day: incoming at the top is S0 * mu.
    for name in ("fds", "fus", "direct", "diffuse", "dts"):
        assert not out[name][~day].any(), name
    # (the band solar weights sum to 1 within 5e-5)
    np.testing.assert_allclose(out["fds"][day, 0], 1361.0 * mu[day], rtol=1e-4)
    # Energy closure: each heating rate is a net flux divergence per hPa,
    # so the column sum telescopes to net top minus net surface flux.
    dp = np.diff(fields["pressure"].astype(np.float64), axis=1)
    for heat, down, up in (("dts", "fds", "fus"), ("dtir", "fdir", "fuir")):
        net = out[down].astype(np.float64) - out[up]
        absorbed = (out[heat] * dp).sum(axis=1)
        np.testing.assert_allclose(absorbed, net[:, 0] - net[:, -1], rtol=0, atol=0.05,
                                   err_msg=heat)
    np.testing.assert_array_equal(out["dt"], out["dts"] + out["dtir"])
    # Physically bounded: reflected and emitted top fluxes, surface IR.
    assert (out["fuir"][:, 0] > 100).all() and (out["fuir"][:, 0] < 350).all()
    assert (out["fus"][day, 0] < 1361.0 * mu[day]).all()
    assert (out["fdir"][:, -1] > 200).all()
    # Default numerics stay near the strict Fortran oracle (a strict run is
    # held to words above): 0.5 W/m2 in every flux and layer absorption.
    got, want = _watts(out, dp), _watts(reference, dp)
    for name in reference:
        np.testing.assert_allclose(got[name], want[name], rtol=0, atol=0.5, err_msg=name)
    _check_xbackend("flg256", out, dp)


@pytest.mark.gpu
@pytest.mark.parametrize("layers", [SHIPPED, DEEP])
def test_the_deep_tier_frame_is_the_priced_frame(layers):
    import cupy as cp  # noqa: F401  (GPU test; the CPU tests above stay unmarked)

    from woof import backend
    from woof.core import preflight as pf
    from woof.core.flg import FLGColumn
    from woof.core.kernels import kernel_local_size_bytes

    if backend.active() != "cuda":
        pytest.skip("driver local frames are a CUDA attribute")
    column = FLGColumn(1, layers)
    # The workspace, not the frame, grows with the column: 300 x (nz+1) + 1 words.
    assert column.workspace.shape[1] == 300 * (layers + 1) + 1
    assert kernel_local_size_bytes(column.kernel) <= pf.KERNEL_MAX_LOCAL_SIZE_BYTES["flg_column"]
