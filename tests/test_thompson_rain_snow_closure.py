"""Thompson's rain-snow collection closure (quirk pass, correct-mode FQ5).

THE QUIRK
---------
WRF 4.7.1's Thompson (module_mp_thompson.F) limits the rain and snow
losses of rain-snow collection (prr_rcs, prs_rcs) when they would exhaust
either species, and re-pairs the transfer only above 0 C wet bulb
(:2946-2950).  Below it the graupel gain prg_rcs keeps its pre-limiter
value, so the exchange creates graupel (and its freezing heat) from rain
and snow that ran out.  RunConfig.thompson_rain_snow_closure sets
prg_rcs = -(prr_rcs + prs_rcs) after the limiters.

WHAT THIS FILE HOLDS
--------------------
* The key is correct-mode finding FQ5: on by default, off in exact mode,
  read only under mp_physics = 8, 28 or 38.
* Exact mode is WRF's program word for word: every Thompson unit carries
  the correction only inside ``#ifdef THOMPSON_RAIN_SNOW_CLOSURE`` blocks,
  and each unit with those blocks removed is byte-identical to the source
  frozen before the correction (the digests below are the files at
  d08b5f729, the 1.1.7 sync).
* Every launcher compiles the define exactly when the key reaches the run.
* On the device, below 0 C with rain nearly exhausted, the corrected
  exchange makes no water: what graupel gains, rain and snow lose.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from woof import correct_mode
from woof.config import RunConfig, validate_run_config

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "woof" / "core" / "kernels"
KEY = "thompson_rain_snow_closure"

#: Each unit: (closure blocks, sha256 of the unit with them removed, which
#: is the unit's source before the correction).
UNITS = {
    "thompson.cu": (3, "cad801b72ab59d54c28e7903b125cbced4d8ae7211e1ff28427ffaee0dcd9b43"),
    "thompson_aerosol_cold.cu": (2, "ac6af93c55763d6fc1b42697ad5b93be447547e931a2010e6ba2ba4ed90d4743"),
    "thompson_aerosol_warm.cu": (2, "c3e55b92419e88dbeccb77fdceac77f5e0db5497e8bcc4b381c06de0812efe16"),
    "thompson_classic.cu": (1, "efc122d5dcdc19763584ed856dc7800b20621fd1b7dd80defefe82583b6152bd"),
    "thompson_fork.cu": (1, "c7313ce62d863bdeb9478b7a503c8faffae6d17c8a20cbb1b6d458c216e4c947"),
    "thompson_column_generated.cuh": (1, "f07dd881fbec30952ba3d17b16012693e4ea689f50dcefde350c0222473fb552"),
}


def _split_blocks(text):
    """(block bodies, text with the blocks removed).  A block ends at the
    #endif that closes its #ifdef, so it may hold its own #if arms."""
    bodies, kept, current, depth = [], [], None, 0
    for line in text.splitlines(keepends=True):
        word = line.strip()
        if current is None:
            if word == "#ifdef THOMPSON_RAIN_SNOW_CLOSURE":
                current, depth = [], 1
            else:
                kept.append(line)
            continue
        if word.startswith("#if"):
            depth += 1
        elif word.startswith("#endif"):
            depth -= 1
            if depth == 0:
                bodies.append("".join(current))
                current = None
                continue
        current.append(line)
    assert current is None, "unterminated closure block"
    return bodies, "".join(kept)


def _run(**kw):
    return validate_run_config(RunConfig(
        nx=8, ny=8, nz=12, dx=3000.0, dy=3000.0, ztop=12000.0, dt=10.0,
        run_seconds=60.0, moist=True, **kw))


def test_the_key_is_correct_mode_finding_fq5():
    assert correct_mode.FINDING_BY_KEY[KEY] == "FQ5"
    assert correct_mode.KEY_SCOPE[KEY] == ("mp_physics", (8, 28, 38))
    assert RunConfig.__dataclass_fields__[KEY].default is correct_mode.CORRECT_KEYS_DEFAULT
    assert correct_mode.resolve("exact").settings[KEY] is False
    assert correct_mode.resolve("correct").settings[KEY] is True


@pytest.mark.parametrize("mp", [8, 28, 38])
def test_the_launchers_read_it_where_thompson_runs(mp):
    from woof.core.thompson import rain_snow_closure_due
    assert rain_snow_closure_due(_run(mp_physics=mp)) is correct_mode.CORRECT_KEYS_DEFAULT
    assert rain_snow_closure_due(_run(mp_physics=mp, **{KEY: True})) is True
    # one key turns the one correction off
    assert rain_snow_closure_due(_run(mp_physics=mp, **{KEY: False})) is False
    exact = _run(mp_physics=mp, **correct_mode.resolve("exact").settings)
    assert rain_snow_closure_due(exact) is False


@pytest.mark.parametrize("mp", [6, 10, 29])
def test_it_is_inert_where_thompson_does_not_run(mp):
    """RCON (29) shares no rain-snow closure; WSM6 and Morrison never read it."""
    from woof.core.thompson import rain_snow_closure_due
    assert rain_snow_closure_due(_run(mp_physics=mp, **{KEY: True})) is False


@pytest.mark.parametrize("unit", sorted(UNITS))
def test_without_the_define_every_unit_is_the_program_before_the_correction(unit):
    text = (KERNELS / unit).read_text(encoding="utf-8")
    blocks, digest = UNITS[unit]
    found, stripped = _split_blocks(text)
    assert len(found) == blocks, unit
    for body in found:
        # A pure addition: no arm at the block's own level that exact mode
        # would compile (a nested #if/#else wholly inside the block is
        # skipped with it).
        depth = 0
        for line in body.splitlines():
            word = line.strip()
            if word.startswith("#if"):
                depth += 1
            elif word.startswith("#endif"):
                depth -= 1
            elif word.startswith(("#else", "#elif")):
                assert depth > 0, unit
        assert "rain_snow" in body or "prg_rcs" in body, unit
    assert hashlib.sha256(stripped.encode("utf-8")).hexdigest() == digest, unit


def test_the_default_compile_never_sets_the_define():
    from woof.core import kernels
    assert "THOMPSON_RAIN_SNOW_CLOSURE" not in repr(getattr(kernels, "CUDA_DEFINES", ""))
    from woof.core.thompson import RAIN_SNOW_CLOSURE_DEFINES
    assert RAIN_SNOW_CLOSURE_DEFINES == (("THOMPSON_RAIN_SNOW_CLOSURE", 1),)


@pytest.mark.parametrize("unit", sorted(UNITS))
def test_every_closure_block_floors_the_graupel_gain_at_zero(unit):
    """Below freezing the snow term can be a net gain; when the rain limiter
    shrinks rain's loss under it, -(prr+prs) < 0 would make graupel a sink
    clipped at qg = 0 (water created as snow).  Every block floors the gain
    at zero and has snow take only what rain lost."""
    bodies, _ = _split_blocks((KERNELS / unit).read_text(encoding="utf-8"))
    assert bodies
    for body in bodies:
        assert re.search(r"<\s*0\.(0f?|f|)\)", body), (unit, body)
        compact = re.sub(r"\s+", "", body)
        assert re.search(
            r"(prs_rcs|rain_snow_snow_rate|rain_snow_category_rate)"
            r"(\[[^;]*\])?=\(?-(L\.)?(prr_rcs|rain_snow_rain_rate)",
            compact), (unit, body)
        assert re.search(r"(prg_rcs|rain_snow_graupel_rate)(\[[^;]*\])?=0\.(0f?|f|);",
                         compact), (unit, body)


def test_the_generated_column_block_is_the_translator_s():
    import sys
    sys.path.insert(0, str(ROOT / "tools" / "thompson_generation_oracle"))
    try:
        import translate
    finally:
        sys.path.pop(0)
    text = (KERNELS / "thompson_column_generated.cuh").read_text(encoding="utf-8")
    assert "// woof RunConfig.thompson_rain_snow_closure (FQ5)\n" in text
    assert translate.RAIN_SNOW_CLOSURE.startswith("if (temp(k).lt.T_0) prg_rcs(k)")


@pytest.mark.parametrize("closure", [False, True])
def test_thompson_py_compiles_the_define_only_when_asked(monkeypatch, closure):
    from woof.core import kernels, thompson
    calls = []
    monkeypatch.setattr(thompson, "get_kernel",
                        lambda module, func: calls.append((module, func, ())))
    monkeypatch.setattr(kernels, "get_kernel_int_defines",
                        lambda module, func, defines: calls.append((module, func, defines)))
    thompson._closure_kernel("thompson_frozen_vapor_network", closure)
    assert calls == [("thompson", "thompson_frozen_vapor_network",
                      thompson.RAIN_SNOW_CLOSURE_DEFINES if closure else ())]


# --- the device: the corrected exchange makes no water --------------------

def _cold_column(cp, closure: bool):
    """A cold (260-263 K) column holding snow and a trace of rain, so the
    rain limiter binds in rain-snow collection; returns the fields after
    one default-Thompson call.  WRF's arithmetic makes 3.5 times the rain
    there as graupel; the corrected exchange makes none."""
    from woof.core import constants
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.physics import initialize_physics
    from woof.core.state import init_theta_perturbation

    cfg = RunConfig(nx=16, ny=12, nz=4, dx=1000., dy=1000., ztop=4000.,
                    dt=60., run_seconds=120., time_step_sound=4,
                    moist=True, mp_physics=8, **{KEY: closure})
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_theta_perturbation(
        cfg, coord, base, lambda x, z: np.zeros((cfg.nz, 1, 1)))
    p = np.asarray([70000., 68000., 66000., 64000.], np.float32)
    t = np.asarray([263., 262., 261., 260.], np.float32)
    theta = t * (1.0e5 / p) ** np.float32(constants.RD / constants.CP)
    state.thb.fill(0.)
    state.thp[...] = cp.asarray(theta)[:, None, None]
    state.p[...] = cp.asarray(p)[:, None, None]
    state.php.fill(0.)
    state.w.fill(0.)
    z = np.arange(cfg.nz + 1, dtype=np.float32) * np.float32(1000.)
    state.phb[...] = cp.asarray(z * np.float32(constants.G)).reshape(state.phb.shape)
    values = {"qv": 1.0e-3, "qc": 0.0, "qr": 1.0e-5, "qi": 0.0, "qs": 5.0e-3,
              "qg": 0.0, "nr": 2.0e3, "ni": 0.0}
    for name, value in values.items():
        getattr(state, name)[...] = np.float32(value)
    initialize_physics(state, cfg, landmask=1.)
    from woof.core import microphysics
    result = microphysics._apply_thompson(state, cfg, cfg.dt)
    cp.cuda.Stream.null.synchronize()
    out = {n: cp.asnumpy(getattr(state, n))[:, 0, 0].astype(np.float64)
           for n in ("qv", "qc", "qr", "qi", "qs", "qg")}
    out["precip_mm"] = sum(float(cp.asnumpy(getattr(result, f))[0, 0])
                           for f in ("rainncv", "snowncv", "graupelncv"))
    return out


@pytest.mark.gpu
def test_the_corrected_cold_exchange_makes_no_graupel_from_nothing():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("device verification needs CUDA")
    stock = _cold_column(cp, False)
    fixed = _cold_column(cp, True)
    # Only graupel's source differs: what the stock exchange made from
    # nothing is graupel the corrected one does not hold.
    stock_total = sum(stock[n] for n in ("qv", "qc", "qr", "qi", "qs", "qg"))
    fixed_total = sum(fixed[n] for n in ("qv", "qc", "qr", "qi", "qs", "qg"))
    assert (stock["qg"] >= fixed["qg"]).all()
    assert (stock["qg"] > fixed["qg"]).any(), "the rain limiter never bound"
    assert stock["precip_mm"] >= fixed["precip_mm"]
    # The stock column holds more water than the corrected one: the
    # graupel it made from rain and snow that ran out.
    assert float((stock_total - fixed_total).sum()) > 0.0
