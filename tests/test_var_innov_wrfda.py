"""woof.da.var.innov_wrfda: WRFDA's innovation step (plan V1.2/V1.3, gate O5).

The synthetic tests run everywhere. The tutorial gate needs
``WOOF_WRFDA471_DIR`` with ``ob.ascii``, ``woof_xb.bin`` and ``woof_iv.bin``
(the oracle build's dumps; see ``test_var_be_dat.py``).

O5: per type and field the qc flags equal the oracle's everywhere (so the
accepted sets are equal) and ``inv`` agrees to 1e-10 relative. The words
are not all identical: the observation positions and rotated winds carry the
libm ulps documented in ``test_var_obs_wrfda.py`` (crmath against glibc), and
the surface correction's ``**``/``exp`` add their own, and so do gpsref's
``exp`` of the interpolated ``log N`` and satem's thickness ``log`` (at
most 2 ulps per call, giving 2.7e-11 and 5.1e-12 relative in ``inv``).
With the oracle's own ``dx``/``dy`` injected, the upper-air types and
gpspw agree word for word.
"""

from __future__ import annotations

import copy
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import _var_wrfda_synthetic as syn
from woof.da.var import innov_wrfda as iw
from woof.da.var import obs_wrfda as ow
from woof.da.var.oracle import read_dump

MANIFEST = Path(__file__).parent / "data" / "wrfda471" / "manifest.json"
M = ow.MISSING_R


# --------------------------------------------------------------------------- synthetic
def test_to_zk_pressure_and_height():
    v = np.array([1000.0, 900.0, 800.0])
    assert iw.to_zk(950.0, v, True) == 1.5
    assert iw.to_zk(800.0, v, True) == 2.0 + (900.0 - 800.0) / (900.0 - 800.0) - 0.0
    assert iw.to_zk(1001.0, v, True) == M and iw.to_zk(799.0, v, True) == M
    h = np.array([10.0, 110.0, 310.0])
    assert iw.to_zk(60.0, h, False) == 1.0 + (10.0 - 60.0) / (10.0 - 110.0)
    assert iw.to_zk(5.0, h, False) == M


def test_convert_zk_clips_and_flags_missing():
    k, dz, dzm = iw.convert_zk(np.array([M, 1.25, 40.0, 0.5, -3.0]), kte=40)
    assert k.tolist() == [0, 1, 39, 1, 0]
    assert dz.tolist() == [0.0, 0.25, 1.0, -0.5, 0.0]
    assert dzm.tolist() == [0.0, 0.75, 0.0, 1.5, 0.0]


def _sound(xb, p_levels, t_obs, terr=1.0):
    return syn.obs_type("sound", [dict(
        i=3, j=4, dx=0.25, dy=0.5, h=[M] * len(p_levels), p=p_levels,
        fields={"u": [(5.0, 0, 2.0)] * len(p_levels), "v": [(-2.0, 0, 2.0)] * len(p_levels),
                "t": [(x, 0, terr) for x in t_obs], "q": [(0.005, 0, 0.001)] * len(p_levels)})])


def test_sound_innovation_is_ob_minus_interpolated_background():
    xb = syn.background()
    col = iw._hcol(xb["p"], 2, 3, 0.25, 0.5, 0.75, 0.5)
    p_ob = 0.5 * (col[4] + col[5])
    tcol = iw._hcol(xb["t"], 2, 3, 0.25, 0.5, 0.75, 0.5)
    iv = iw.innovations(syn.obs({"sound": _sound(xb, [p_ob, 10.0], [280.0, 250.0], terr=3.0)}), xb)
    s = iv["sound"]
    zk = s.extra["zk"]
    assert 5.0 < zk[0] < 6.0 and zk[1] == M
    dz = zk[0] - 5.0
    model = (1.0 - dz) * tcol[4] + dz * tcol[5]
    assert s.fields["t"].inv[0] == pytest.approx(280.0 - model, rel=1e-12)
    # the level above the model top is missing data with a zero innovation
    assert s.fields["t"].qc[1] == ow.MISSING_DATA and s.fields["t"].inv[1] == 0.0


def test_check_max_iv_rejects_and_q_follows_t():
    xb = syn.background()
    col = iw._hcol(xb["p"], 2, 3, 0.25, 0.5, 0.75, 0.5)
    p_ob = 0.5 * (col[4] + col[5])
    iv = iw.innovations(syn.obs({"sound": _sound(xb, [p_ob], [400.0], terr=1.0)}), xb)
    f = iv["sound"].fields
    assert f["t"].qc[0] == iw.FAILS_ERROR_MAX and f["t"].inv[0] == 0.0
    assert f["q"].qc[0] == iw.FAILS_ERROR_MAX and f["q"].inv[0] == 0.0
    assert f["u"].qc[0] == 0
    off = iw.innovations(syn.obs({"sound": _sound(xb, [p_ob], [400.0])}), xb,
                         options=iw.Options(check_max_iv=False))
    assert off["sound"].fields["t"].qc[0] == 0


def _synop(h, p=M, t=M, u=3.0):
    return syn.obs_type("synop", [dict(
        i=2, j=2, dx=0.5, dy=0.5, h=[h],
        fields={"u": [(u, 0, 1.1)], "v": [(1.0, 0, 1.1)], "t": [(t, 0 if t != M else -88, 2.0)],
                "p": [(p, 0 if p != M else -88, 100.0)], "q": [(M, -88, 0.002)]})])


def test_surface_station_below_the_model_surface_is_corrected():
    xb = syn.background()
    hcol = iw._hcol(xb["h"], 1, 1, 0.5, 0.5, 0.5, 0.5)
    ho = hcol[0] - 50.0
    iv = iw.innovations(syn.obs({"synop": _synop(ho, p=99000.0, t=285.0)}), xb)
    s = iv["synop"]
    assert s.extra["zk"][0] == 1.0 + 1.0e-6
    assert s.fields["p"].qc[0] == iw.SURFACE_CORRECTION
    assert s.fields["t"].qc[0] == iw.SURFACE_CORRECTION
    # the reduced pressure is lower than the station's (station is lower)
    geo = s.extra["geometry"]
    model_p = iw.interp_lin_3d(xb["p"], geo)[0]
    assert s.fields["p"].inv[0] + model_p < 99000.0


def test_surface_station_far_above_the_model_is_dropped():
    xb = syn.background()
    hcol = iw._hcol(xb["h"], 1, 1, 0.5, 0.5, 0.5, 0.5)
    iv = iw.innovations(syn.obs({"synop": _synop(hcol[0] + 500.0, t=280.0)}), xb)
    s = iv["synop"]
    assert all(s.fields[v].qc[0] == ow.MISSING_DATA for v in ("u", "v", "t", "p", "q"))
    assert s.fields["u"].inv[0] == 0.0


def test_jo_is_half_the_weighted_sum_of_accepted_innovations():
    xb = syn.background()
    col = iw._hcol(xb["p"], 2, 3, 0.25, 0.5, 0.75, 0.5)
    iv = iw.innovations(syn.obs({"sound": _sound(xb, [0.5 * (col[4] + col[5]), 10.0], [281.0, 250.0])}), xb)
    j = iw.jo(iv)["sound"]
    f = iv["sound"].fields
    want = 0.5 * sum(float(f[v].inv[0] ** 2 / f[v].error[0] ** 2) for v in ("u", "v", "t", "q"))
    assert j["_"] == pytest.approx(want, rel=1e-14)
    assert iw.jo_total({"sound": j}) == j["_"]


# --------------------------------------------------------------------------- tutorial (O5)
@lru_cache(maxsize=1)
def _tutorial():
    d = Path(os.environ.get("WOOF_WRFDA471_DIR", ""))
    need = ("ob.ascii", "woof_xb.bin", "woof_iv.bin")
    if not os.environ.get("WOOF_WRFDA471_DIR") or not all((d / n).is_file() for n in need):
        pytest.skip("WOOF_WRFDA471_DIR with ob.ascii, woof_xb.bin and woof_iv.bin not available")
    man = json.loads(MANIFEST.read_text())["obs"]
    proj = ow.LambertWRFDA.from_attributes(man["first_guess_attributes"])
    obs = ow.read_obs_ascii(d / "ob.ascii", proj, **man["window"])
    xb = read_dump(d / "woof_xb.bin")
    dump = read_dump(d / "woof_iv.bin")
    return obs, xb, dump, iw.innovations(obs, xb)


def _tag(name):
    return "gpsrf" if name == "gpsref" else name


#: O5's inv tolerance (relative to the oracle's |inv|).
INV_RTOL = 1e-10


@pytest.mark.parametrize("name", [n for n in ow.TYPES if n in iw.IMPLEMENTED])
def test_o5_qc_and_innovations_against_the_oracle(name):
    obs, xb, dump, iv = _tutorial()
    t = iv[name]
    if f"{_tag(name)}%levels" not in dump:
        assert t.nlocal == 0
        return
    for v, f in t.fields.items():
        inv, qc = dump[f"{name}%{v}%inv"], dump[f"{name}%{v}%qc"]
        assert np.array_equal(f.qc, qc), (name, v, np.flatnonzero(f.qc != qc)[:5])
        assert (f.qc >= 0).sum() == (qc >= 0).sum()
        assert np.array_equal(f.error, dump[f"{name}%{v}%error"]) or v == "q"
        rel = np.abs(f.inv - inv) / np.maximum(np.abs(inv), 1e-300)
        rel[(inv == 0.0) & (f.inv == 0.0)] = 0.0
        assert rel.max(initial=0.0) <= INV_RTOL, (name, v, rel.max())


@pytest.mark.parametrize("name", ["sound", "pilot", "profiler", "airep", "geoamv", "buoy", "gpspw"])
def test_o5_word_for_word_with_the_oracles_positions(name):
    """With the oracle's own ``dx``/``dy`` the innovation arithmetic itself is
    WRFDA's word for word (apart from the rotated winds' libm ulps)."""
    obs, xb, dump, _ = _tutorial()
    obs = copy.deepcopy(obs)
    t = obs[name]
    for k in ("dx", "dy", "dxm", "dym"):
        t.info[k] = dump[f"{_tag(name)}%{k}"][0].copy()
    iv = iw.innovations(obs, xb, use=[name])
    for v, f in iv[name].fields.items():
        if v in ("u", "v"):
            continue  # da_ffdduv's sin/cos ulps (test_var_obs_wrfda.py)
        assert np.array_equal(f.inv, dump[f"{name}%{v}%inv"]), (name, v)


def _dump_iv(iv, dump):
    out = copy.deepcopy(iv)
    for name, t in out.types.items():
        if t.nlocal == 0 or name not in iw.JO_FIELDS:
            continue
        for v, f in t.fields.items():
            if f"{name}%{v}%inv" in dump:
                f.inv[:] = dump[f"{name}%{v}%inv"]
                f.qc[:] = dump[f"{name}%{v}%qc"]
                f.error[:] = dump[f"{name}%{v}%error"]
    return out


def test_jo_at_the_background_per_type_against_the_oracles_innovations():
    """Jo at x_b per type, ours against the same sums over the oracle's own
    (inv, qc, error): like for like (the tutorial's ``jo`` file is the
    analysis Jo, not Jo at x_b)."""
    obs, xb, dump, iv = _tutorial()
    ours, ref = iw.jo(iv), iw.jo(_dump_iv(iv, dump))
    for name in ours:
        if name in iw.IMPLEMENTED:
            assert ours[name]["_"] == pytest.approx(ref[name]["_"], rel=1e-11), name


#: WRFDA 4.7.1's tutorial ``J_0`` (``Initial cost function``, Jo at x_b).
WRFDA_J0 = 29710.45778742540


def test_total_jo_at_the_background_is_wrfdas_initial_cost():
    obs, xb, dump, iv = _tutorial()
    assert iw.jo_total(iw.jo(iv)) == pytest.approx(WRFDA_J0, rel=1e-13)
