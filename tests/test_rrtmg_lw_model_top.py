"""RRTMG longwave model-top correction (quirk pass, correct-mode FQ6).

THE QUIRK
---------
WRF 4.7.1's RRTMG longwave wrapper (module_ra_rrtmg_lw.F:12375-12380)
extends the column above the model top with a standard profile::

    do L=kte+1,nlayers+1
      tlev(L) = varint(L) + (tlev(kte) - varint(kte))
      tlay(L-1) = 0.5*(tlev(L)+tlev(L-1))

The loop starts at kte+1, so it overwrites tlev(kte+1) (the model-top
interface temperature from t8w) and tlay(kte) (the top model layer from
t3d), and anchors at the interface below the top.  RunConfig.rrtmg_lw_model_top
starts it at kte+2 and anchors at tlev(kte+1).

WHAT THIS FILE HOLDS
--------------------
* The key is correct-mode finding FQ6: on by default, off in exact mode,
  read only by the legacy RRTMG longwave (ra_lw_physics = 4, or the
  ra_physics = 4 aggregate, with ra_rrtmg_variant = "rrtmg_legacy").
* Without the define the kernel source is the program before the correction
  (the digest below is the file at 51cad4fdf).
* The Python mirror (``model_top=True``) keeps tlev(kte+1) and tlay(kte) and
  anchors the rest of the profile at tlev(kte+1); False is unchanged.
* On the device, the kernel compiled with RRTMG_LW_MODEL_TOP matches the
  mirror bitwise, and without it matches WRF's arithmetic.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from woof import correct_mode
from woof.config import RunConfig, validate_run_config
from woof.core import rrtmg_legacy_prep as ref
from woof.quirk_corrections import run_reads

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "woof" / "core" / "kernels" / "rrtmg_legacy_prep.cu"
KEY = "rrtmg_lw_model_top"

#: sha256 of rrtmg_legacy_prep.cu at 51cad4fdf, before the correction.
SOURCE_BEFORE = "bb6fb8f5194954d6208f66069b3259b02881d2220f22ec4437b66c9544ab7563"
_ARM = re.compile(r"#ifdef RRTMG_LW_MODEL_TOP\n.*?#else\n(.*?)#endif\n", re.S)

_LEGACY = dict(ra_lw_physics=4, ra_sw_physics=4, ra_rrtmg_variant="rrtmg_legacy")


def _run(**kw):
    return validate_run_config(RunConfig(
        nx=8, ny=8, nz=12, dx=3000.0, dy=3000.0, ztop=12000.0, dt=10.0,
        run_seconds=60.0, **kw))


# --- the key ---------------------------------------------------------------

def test_the_key_is_correct_mode_finding_fq6():
    assert correct_mode.FINDING_BY_KEY[KEY] == "FQ6"
    assert correct_mode.KEY_SCOPE[KEY] == ("ra_lw_physics", (4,))
    assert correct_mode.KEY_GENERATIONS[KEY] == (
        "ra_rrtmg_variant", "rte-rrtmgp", ("rrtmg_legacy",))
    assert RunConfig.__dataclass_fields__[KEY].default is correct_mode.CORRECT_KEYS_DEFAULT
    assert RunConfig.__dataclass_fields__[KEY].kw_only
    assert correct_mode.resolve("exact").settings[KEY] is False
    assert correct_mode.resolve("correct").settings[KEY] is True


def test_it_must_be_a_bool():
    from woof.config import ConfigurationRefused
    with pytest.raises(ConfigurationRefused, match=KEY):
        _run(**_LEGACY, **{KEY: 1})


@pytest.mark.parametrize("spelling", [
    _LEGACY,
    dict(ra_physics=4, ra_rrtmg_variant="rrtmg_legacy"),
    dict(ra_lw_physics=4, ra_sw_physics=4, ra_physics=4, ra_rrtmg_variant="rrtmg_legacy"),
])
def test_run_reads_on_off_and_exact(spelling):
    assert run_reads(_run(**spelling), KEY) is correct_mode.CORRECT_KEYS_DEFAULT
    assert run_reads(_run(**spelling, **{KEY: True}), KEY) is True
    assert run_reads(_run(**spelling, **{KEY: False}), KEY) is False
    exact = _run(**spelling, **correct_mode.resolve("exact").settings)
    assert run_reads(exact, KEY) is False


@pytest.mark.parametrize("other", [
    dict(ra_physics=4),                                   # RTE+RRTMGP
    dict(ra_lw_physics=4, ra_sw_physics=4),               # RTE+RRTMGP
    dict(ra_physics=0),
    dict(ra_lw_physics=1, ra_sw_physics=1, ra_rrtmg_variant="rrtmg_legacy"),
    dict(ra_physics=90, ra_rrtmg_variant="rrtmg_legacy"),
])
def test_it_is_inert_where_the_legacy_rrtmg_longwave_does_not_run(other):
    assert run_reads(_run(**other, **{KEY: True}), KEY) is False


def test_the_umbrella_covers_it():
    from woof import quirk_corrections as qc
    assert _run(**_LEGACY, woof_quirk_corrections="all").rrtmg_lw_model_top is True
    assert _run(**_LEGACY, woof_quirk_corrections="none").rrtmg_lw_model_top is False
    assert qc.run_scheme_active(KEY, _run(**_LEGACY))
    assert not qc.run_scheme_active(KEY, _run(ra_physics=4))
    from woof.config import ConfigurationRefused
    with pytest.raises(ConfigurationRefused, match=KEY):
        qc.apply_run_corrections({qc.UMBRELLA: "all", **_LEGACY, KEY: False})


def test_the_run_receipt_names_the_finding_only_where_it_is_read():
    legacy = correct_mode.in_effect(_run(**_LEGACY))
    assert "FQ6" in legacy["findings"] and KEY in legacy["keys"]
    rte = correct_mode.in_effect(_run(ra_physics=4))
    assert "FQ6" not in rte["findings"]
    # RTE+RRTMGP carries no such finding: nothing is uncorrected there
    assert "FQ6" not in correct_mode.uncorrected(_run(ra_physics=4))
    assert "FQ6" not in correct_mode.uncorrected(_run(**_LEGACY))


# --- the kernel source -----------------------------------------------------

def test_without_the_define_the_kernel_is_the_program_before_the_correction():
    text = KERNEL.read_text(encoding="utf-8")
    assert len(_ARM.findall(text)) == 1
    arm = text[text.index("#ifdef RRTMG_LW_MODEL_TOP"):text.index("#else\n", text.index("#ifdef RRTMG_LW_MODEL_TOP"))]
    assert "tl[e+nk]" in arm and "k=nk+1" in arm
    stripped = _ARM.sub(lambda m: m.group(1), text)
    assert hashlib.sha256(stripped.encode("utf-8")).hexdigest() == SOURCE_BEFORE
    # the other prep kernel has no such loop
    assert "MODEL_TOP" not in (KERNEL.parent / "rrtmgk_prep.cu").read_text(encoding="utf-8")


# --- the Python mirror -----------------------------------------------------

def _columns(ncol=3, nz=20, seed=4411):
    rng = np.random.default_rng(seed)
    pw = np.broadcast_to(np.linspace(101300, 5000, nz + 1, dtype=np.float32),
                         (ncol, nz + 1)).copy()
    p = ((pw[:, :-1] + pw[:, 1:]) * np.float32(.5)).astype(np.float32)
    # a model-top temperature and a layer temperature the standard profile
    # would never give: the overwrite is then visible
    kw = dict(p3d=p, p8w=pw,
              t3d=rng.uniform(200, 290, (ncol, nz)).astype(np.float32),
              t8w=rng.uniform(200, 290, (ncol, nz + 1)).astype(np.float32),
              dz8w=rng.uniform(200, 1000, (ncol, nz)).astype(np.float32))
    for name in ("qv3d", "qc3d", "qr3d", "qi3d", "qs3d", "qg3d"):
        kw[name] = rng.uniform(0, .001, (ncol, nz)).astype(np.float32)
    kw["cldfra3d"] = rng.uniform(0, 1, (ncol, nz)).astype(np.float32)
    kw["o33d"] = rng.uniform(1.e-8, 1.e-6, (ncol, nz)).astype(np.float32)
    for name, lo, hi in (("re_cloud", 2.5e-6, 30.e-6), ("re_ice", 5.e-6, 150.e-6),
                         ("re_snow", 10.e-6, 200.e-6)):
        kw[name] = rng.uniform(lo, hi, (ncol, nz)).astype(np.float32)
    for name, value in (("tsk", 290), ("xice", .2), ("snow", 30), ("xlat", 35),
                        ("emiss", .98)):
        kw[name] = np.full(ncol, value, np.float32)
    kw["xland"] = np.resize(np.array([1.5, 1, 2], np.float32), ncol)
    kw.update(icloud=1, warm_rain=False, cldovrlp=2, idcor=0, o3input=0,
              has_reqc=1, has_reqi=1, has_reqs=1, yr=2026, julian=183.75,
              nlayers=ref.compute_lw_nlayers(nz + 1, 50.0))
    return kw


@pytest.fixture(autouse=True)
def host_oracle(monkeypatch):
    monkeypatch.setattr(ref, "_PERFWAVE_DEVICE_XP", None)


def _bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def test_the_mirror_keeps_the_model_top_and_anchors_there():
    kw = _columns()
    kte = kw["p3d"].shape[1]
    nlayers = kw["nlayers"]
    assert nlayers > kte
    wrf = ref.lwrad_prep_batch(**kw)
    fixed = ref.lwrad_prep_batch(**kw, model_top=True)
    assert np.array_equal(_bits(ref.lwrad_prep_batch(**kw, model_top=False)["tlev"]),
                          _bits(wrf["tlev"]))
    t8w, t3d = kw["t8w"], kw["t3d"]
    # WRF overwrites the top interface and top layer ...
    assert not np.array_equal(_bits(wrf["tlev"][:, kte]), _bits(t8w[:, kte]))
    assert not np.array_equal(_bits(wrf["tlay"][:, kte - 1]), _bits(t3d[:, kte - 1]))
    # ... the correction keeps both
    assert np.array_equal(_bits(fixed["tlev"][:, kte]), _bits(t8w[:, kte]))
    assert np.array_equal(_bits(fixed["tlay"][:, kte - 1]), _bits(t3d[:, kte - 1]))
    # below the top nothing moves
    assert np.array_equal(_bits(fixed["tlev"][:, :kte]), _bits(wrf["tlev"][:, :kte]))
    assert np.array_equal(_bits(fixed["tlay"][:, :kte - 1]), _bits(wrf["tlay"][:, :kte - 1]))
    # above it: the standard profile anchored at tlev(kte+1)
    F = np.float32
    varint = ref._varint_b(np, fixed["plev"])
    anchor = (fixed["tlev"][:, kte] - varint[:, kte]).astype(F)
    want = (varint[:, kte + 1:nlayers + 1] + anchor[:, None]).astype(F)
    assert np.array_equal(_bits(fixed["tlev"][:, kte + 1:]), _bits(want))
    tlev = fixed["tlev"]
    want_lay = (F(.5) * (tlev[:, kte:nlayers + 1][:, 1:] + tlev[:, kte:nlayers]
                         ).astype(F)).astype(F)
    assert np.array_equal(_bits(fixed["tlay"][:, kte:]), _bits(want_lay))
    # nothing but the temperatures moves
    for name in ("play", "plev", "h2ovmr", "o3vmr", "co2vmr"):
        assert np.array_equal(_bits(fixed[name]), _bits(wrf[name])), name


def test_the_scalar_and_the_batch_mirror_agree_bitwise():
    kw = _columns(ncol=2)
    for model_top in (False, True):
        batch = ref.lwrad_prep_batch(**kw, model_top=model_top)
        for c in range(2):
            col = {k: (v[c] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == 2 else v)
                   for k, v in kw.items()}
            one = ref.lwrad_prep(**col, model_top=model_top)
            for name in ("tlev", "tlay", "play", "plev"):
                assert np.array_equal(_bits(one[name]), _bits(batch[name][c])), (model_top, c, name)


def test_a_column_without_buffer_layers_keeps_its_top_too():
    """nlayers == kte (p_top < 2 hPa): WRF still overwrites tlev(kte+1); the
    correction has no extension to anchor and keeps the model values."""
    kw = _columns(nz=10)
    kw["nlayers"] = kw["p3d"].shape[1]
    wrf = ref.lwrad_prep_batch(**kw)
    fixed = ref.lwrad_prep_batch(**kw, model_top=True)
    kte = kw["nlayers"]
    assert np.array_equal(_bits(fixed["tlev"][:, kte]), _bits(kw["t8w"][:, kte]))
    assert np.array_equal(_bits(fixed["tlay"][:, kte - 1]), _bits(kw["t3d"][:, kte - 1]))
    assert np.array_equal(_bits(fixed["tlev"][:, :kte]), _bits(wrf["tlev"][:, :kte]))


# --- the device ------------------------------------------------------------

def test_the_define_is_compiled_only_when_asked(monkeypatch):
    pytest.importorskip("cupy")
    from woof import nvrtc_ptx_cache
    from woof.core import rrtmg_legacy_device as dev
    seen = []

    def fake(source, options, arch, filename, *a, **k):
        seen.append(tuple(options))
        raise RuntimeError("captured")

    monkeypatch.setattr(nvrtc_ptx_cache, "compile_using_nvrtc", fake)
    dev._gpu_module.cache_clear()
    try:
        for model_top in (False, True):
            with pytest.raises(RuntimeError, match="captured"):
                dev._gpu_module(0, "wrf_461", "meters", model_top)
    finally:
        dev._gpu_module.cache_clear()
    assert "-DRRTMG_LW_MODEL_TOP=1" not in seen[0]
    assert "-DRRTMG_LW_MODEL_TOP=1" in seen[1]


@pytest.mark.gpu
@pytest.mark.parametrize("nz", [20, 11])
def test_the_kernel_with_the_define_matches_the_mirror_bitwise(nz):
    cp = pytest.importorskip("cupy")
    from woof.core import rrtmg_legacy_device as dev
    kw = _columns(nz=nz)
    dkw = {k: cp.asarray(v) if isinstance(v, np.ndarray) and v.ndim else v
           for k, v in kw.items()}
    for model_top in (False, True):
        want = ref.lwrad_prep_batch(**kw, model_top=model_top)
        got = dev.lwrad_prep_batch_device(**dkw, model_top=model_top)
        for name in ("tlev", "tlay", "play", "plev"):
            assert np.array_equal(_bits(cp.asnumpy(got[name])), _bits(want[name])), (
                model_top, name)
    # no argument is the existing behaviour, WRF's arithmetic
    default = dev.lwrad_prep_batch_device(**dkw)
    wrf = ref.lwrad_prep_batch(**kw)
    for name in ("tlev", "tlay"):
        assert np.array_equal(_bits(cp.asnumpy(default[name])), _bits(wrf[name])), name
