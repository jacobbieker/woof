"""WRF-Solar EPS (multi_perturb) and MAD-WRF keys: config, namelist import,
refusals and the restart digest (woof/config.py, woof/namelist_stochastic.py,
woof/namelist_import.py, woof/io/restart.py)."""
from __future__ import annotations

import dataclasses
import tomllib

import pytest

from woof.config import (MULTI_PERTURB_RUN_DEFAULTS, RunConfig,
                         multi_perturb_refusal)
from woof.namelist_import import NamelistRefusal, import_namelists
from test_namelist_import import INPUT_TEXT, WPS_TEXT, _pair


def _cfg(**kw):
    base = dict(nx=40, ny=40, nz=30, dx=3000.0, dy=3000.0, dt=18.0, moist=True,
                ztop=20000.0, run_seconds=3600.0,
                mp_physics=8)
    base.update(kw)
    return RunConfig(**base)


def _import(tmp_path, inp=INPUT_TEXT):
    return import_namelists(*_pair(tmp_path, wps=WPS_TEXT, inp=inp))


EPS = (" multi_perturb = 1, 1,\n pert_thom = .true., .true.,\n"
       " pert_thom_qv = 1.0, 1.0,\n pert_thom_qc = 0.5, 0.5,\n spdt = 15., 15.,\n nens = 3,\n")


# ---- RunConfig -------------------------------------------------------------

def test_the_keys_are_appended_off_and_keyword_only():
    fields = {f.name: f for f in dataclasses.fields(RunConfig)}
    for key, default in MULTI_PERTURB_RUN_DEFAULTS.items():
        assert fields[key].default == default and fields[key].kw_only, key
    assert multi_perturb_refusal(_cfg()) is None


@pytest.mark.parametrize("kw, words", [
    (dict(pert_thom=True), "without multi_perturb = 1 does nothing"),
    (dict(multi_perturb=1), "every pert_<scheme> switch off"),
    # 1.1.6: the WRF engine's FARMS, Deng and cal_cldfra3 are upstream's.
    (dict(multi_perturb=1, pert_farms=True), "carry no WRF-Solar EPS hook"),
    (dict(multi_perturb=1, pert_deng=True), "carry no WRF-Solar EPS hook"),
    (dict(multi_perturb=1, pert_thom=True, mp_physics=6), "mp_physics 28 and 8"),
    (dict(multi_perturb=1, pert_cld3=True), "carry no WRF-Solar EPS hook"),
    (dict(multi_perturb=1, pert_thom=True, use_adaptive_time_step=True), "adpative time step"),
    (dict(multi_perturb=1, pert_thom=True, num_pert_3d=16), "num_pert_3d=16"),
    (dict(multi_perturb=2), "must be 0 or 1"),
])
def test_eps_refusals_name_the_breakage(kw, words):
    assert words in multi_perturb_refusal(_cfg(**kw))


def test_eps_admits_a_wrf_solar_thompson_run():
    assert multi_perturb_refusal(_cfg(multi_perturb=1, pert_thom=True,
                                      pert_thom_qv=1.0)) is None


# ---- the restart digest ------------------------------------------------------

def test_restart_digest_drops_the_defaults_and_keeps_a_moved_key():
    from woof.io.restart import _drop_default_off_run_keys
    values = dict(MULTI_PERTURB_RUN_DEFAULTS)
    _drop_default_off_run_keys(values)
    assert not set(values) & set(MULTI_PERTURB_RUN_DEFAULTS)
    values = {**MULTI_PERTURB_RUN_DEFAULTS, "multi_perturb": 1}
    _drop_default_off_run_keys(values)
    assert values == {"multi_perturb": 1}


# ---- namelist import ------------------------------------------------------------

def test_a_wrf_solar_eps_namelist_imports_to_run_keys(tmp_path):
    inp = INPUT_TEXT.replace("mp_physics = 10, 10", "mp_physics = 8, 8") + "&stoch\n" + EPS + "/\n"
    text, _ = _import(tmp_path, inp)
    raw = tomllib.loads(text)
    shared = raw["shared"]
    assert shared["multi_perturb"] == 1 and shared["pert_thom"] is True
    assert shared["pert_thom_qv"] == 1.0 and shared["pert_thom_qc"] == 0.5
    assert shared["spdt"] == 15.0 and shared["nens"] == 3
    assert "pert_farms" not in shared and "ensemble" not in raw


def test_eps_beside_spp_is_refused(tmp_path):
    inp = INPUT_TEXT + "&stoch\n" + EPS + " sppt = 1, 1,\n/\n"
    with pytest.raises(NamelistRefusal, match="ordinary forecast"):
        _import(tmp_path, inp)


def test_a_moved_forcing_band_is_refused(tmp_path):
    inp = (INPUT_TEXT.replace("mp_physics = 10, 10", "mp_physics = 8, 8")
           + "&stoch\n" + EPS + " kmaxforct = 20,\n/\n")
    with pytest.raises(NamelistRefusal, match="forcing band"):
        _import(tmp_path, inp)


def test_a_stochpert_table_beside_the_namelist_is_named(tmp_path):
    inp = INPUT_TEXT.replace("mp_physics = 10, 10", "mp_physics = 8, 8") + "&stoch\n" + EPS + "/\n"
    from woof.core.wrfsolar_eps import WRF_STOCHPERT_TBL
    wps, inp_path = _pair(tmp_path, wps=WPS_TEXT, inp=inp)
    (inp_path.parent / "STOCHPERT.TBL").write_text(WRF_STOCHPERT_TBL)
    text, _ = import_namelists(wps, inp_path)
    assert tomllib.loads(text)["shared"]["stochpert_tbl"].endswith("STOCHPERT.TBL")


def test_stochpert_table_parse_matches_wrf_rows():
    from woof.core.wrfsolar_eps import WRF_STOCHPERT_TBL, parse_stochpert_table
    rows = parse_stochpert_table(WRF_STOCHPERT_TBL)
    assert list(rows) == ["ALBEDO", "AOD", "ANGSTROM", "ASSYMFAC", "QVAPOR", "QCLOUD",
                          "QICE", "QSNOW", "NI", "TH", "TKE", "SMOIS", "TSOIL", "W"]
    assert rows["TH"].std == pytest.approx(0.001) and rows["W"].vertstruc == 1
    assert rows["ALBEDO"].timescale == 86400.0 and rows["AOD"].vertstruc == 0
    assert list(parse_stochpert_table(WRF_STOCHPERT_TBL, num_pert_3d=6)) == [
        "ALBEDO", "AOD", "ANGSTROM", "ASSYMFAC", "QVAPOR"]
