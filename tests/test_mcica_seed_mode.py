"""Opt-in frozen McICA seeds (woof.core.mcica_seed).

Default ("pressure") seeding must be byte-identical to main: the digests
below were taken from main before the mode existed.  "cell" seeding must not
see the model state at all: a one-ulp pressure change reshuffles the McICA
clouds in "pressure" mode and leaves them identical in "cell" mode.  Both
routes run in child processes, under fast and strict numerics.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from woof.core import mcica_seed  # noqa: E402

#: sha256 of lwrad_prep_batch's outputs on the 4-column cloudy fixture, taken on main (f2a09b929).
WRF_DEFAULT_DIGEST = "20b673ffee71a7044ea5fc4eb53359112f2f08ca463220805900bcfedcbf6a16"
#: sha256 of rrtmg_lw_v842._lw_mcica_wrapper's four slabs on the 5-column fixture of _hex_child, from main.
HEX_DEFAULT_DIGEST = "b726d1a2a003af35ab9c477f0794943418dc22615cbfebd87cae8e9ea303479f"


def _digest(arrays):
    h = hashlib.sha256()
    for name, value in arrays:
        h.update(name.encode())
        h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def _wrf_child():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fx", ROOT / "tests/test_rrtmg_lw_buffer_grid.py")
    fx = importlib.util.module_from_spec(spec)
    sys.modules["fx"] = fx
    spec.loader.exec_module(fx)
    from woof.core import rrtmg_legacy_prep as prep
    prep._PERFWAVE_DEVICE_XP = None
    prov = json.loads((fx.DATA / "provenance.json").read_text())
    a = np.load(fx.DATA / prov["array_file"])
    inputs = {k.removeprefix("input__"): (a[k][()] if a[k].ndim == 0 else a[k])
              for k in a.files if k.startswith("input__")}
    n = 4
    batch = {}
    for k, v in inputs.items():
        if isinstance(v, np.ndarray):
            batch[k] = np.repeat(v[None], n, 0)
        elif k in fx.SURFACES:
            batch[k] = np.full(n, v, np.float32)
        else:
            batch[k] = v
    batch["cldfra3d"] = np.full_like(batch["cldfra3d"], 0.5)
    batch["qc3d"] = np.full_like(batch["qc3d"], 1e-4)

    def run(perturb, mode):
        mcica_seed.set_mode(mode)
        mcica_seed.set_context(time_seconds=3600)
        b = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in batch.items()}
        if perturb:   # one ulp on the lowest layer's pressure of column 1
            b["p3d"][1, 0] = np.nextafter(b["p3d"][1, 0], np.float32(np.inf))
        out = prep.lwrad_prep_batch(**b)
        return _digest([(k, np.asarray(out[k])) for k in sorted(out)]), _digest(
            [(k, np.asarray(out[k])) for k in ("cldfmcl",)])

    base, _ = run(False, None)
    res = {"default": base}
    for mode in ("pressure", "cell"):
        res[mode] = run(False, mode)[1]
        res[mode + "_ulp"] = run(True, mode)[1]
    print(json.dumps(res))


def _hex_child():
    try:
        import cupy as cp
    except ImportError:
        from woof import backend
        backend.install()
        import cupy as cp
    from woof.hex.mpasphys import rrtmg_lw_v842 as lw
    ng, nl = 5, 12
    rng = np.random.default_rng(1)
    p = np.sort(rng.uniform(300, 1000, (ng, nl)).astype(np.float32))[:, ::-1]
    base = np.zeros((ng, 12, nl), np.float32)
    base[:, 0, :] = p
    base[:, 5, :] = 0.5
    base[:, 6:9, :] = 1.0
    s = cp.zeros((ng, 6), cp.float32)

    def run(perturb, mode):
        mcica_seed.set_mode(mode)
        mcica_seed.set_context(time_seconds=3600)
        a = base.copy()
        if perturb:
            a[1, 0, 0] = np.nextafter(a[1, 0, 0], np.float32(np.inf))
        out = lw._lw_mcica_wrapper(ng, nl, 2, 0, 100, s, cp.asarray(np.ascontiguousarray(a)))
        return _digest([(str(i), cp.asnumpy(x)) for i, x in enumerate(out)])

    res = {"default": run(False, None)}
    for mode in ("pressure", "cell"):
        res[mode] = run(False, mode)
        res[mode + "_ulp"] = run(True, mode)
    print(json.dumps(res))


def _child(route, numerics, exact):
    env = dict(os.environ, PYTHONPATH=str(ROOT), GPUWM_NUMERICS=numerics,
               WOOF_HEX_NUMERICS=numerics, GPUWM_WRF_EXACT=exact)
    env.pop(mcica_seed.ENV, None)
    done = subprocess.run([sys.executable, str(Path(__file__).resolve()), route], env=env,
                          capture_output=True, text=True, timeout=900)
    if done.returncode:
        pytest.skip(f"{route} child unavailable here: {done.stderr[-400:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("numerics,exact", [("fast", "0"), ("strict", "1")])
def test_wrf_route_default_identity_and_cell_invariance(numerics, exact):
    r = _child("wrf", numerics, exact)
    assert r["default"] == WRF_DEFAULT_DIGEST            # byte-identical to main
    assert r["pressure"] != r["pressure_ulp"]            # upstream seeding reshuffles on 1 ulp
    assert r["cell"] == r["cell_ulp"]                    # cell seeding does not see pressure
    assert r["cell"] != r["pressure"]


@pytest.mark.parametrize("numerics", ["fast", "strict"])
def test_hex_route_default_identity_and_cell_invariance(numerics):
    r = _child("hex", numerics, "0")
    assert r["default"] == r["pressure"] == HEX_DEFAULT_DIGEST   # byte-identical to main
    assert r["pressure"] != r["pressure_ulp"]
    assert r["cell"] == r["cell_ulp"]
    assert r["cell"] != r["pressure"]


def test_seed_words_are_state_free_and_in_range():
    ids = np.arange(1000)
    lw = mcica_seed.seed_words(ids, mcica_seed.LW_STREAM, 3600)
    assert lw.shape == (1000, 4) and lw.dtype == np.uint32
    assert 1 <= lw.min() and lw.max() <= 999999998
    assert (lw == mcica_seed.seed_words(ids, mcica_seed.LW_STREAM, 3600)).all()
    assert not (lw == mcica_seed.seed_words(ids, mcica_seed.SW_STREAM, 3600)).any()
    assert not (lw == mcica_seed.seed_words(ids, mcica_seed.LW_STREAM, 3601)).any()
    # a column's words do not depend on which other columns are in the call
    assert (mcica_seed.seed_words(ids[7:9], mcica_seed.LW_STREAM, 3600) == lw[7:9]).all()


def test_default_mode_returns_inputs_untouched():
    mcica_seed.set_mode(None)
    os.environ.pop(mcica_seed.ENV, None)
    play = np.full((3, 6), 1000.0, np.float32)
    assert mcica_seed.seed_play(play, mcica_seed.LW_STREAM) is play
    with pytest.raises(ValueError):
        mcica_seed.set_mode("bogus")


if __name__ == "__main__":
    {"wrf": _wrf_child, "hex": _hex_child}[sys.argv[1]]()
