"""hydro.namelist mapped to the WRF-Hydro opt-in, refusing unported options by name.

The fixture is WRF 4.7.1's own hydro/template/HYDRO/hydro.namelist (copied
under the opt-in, sha256 72d1017f...).  Every test skips, naming the opt-in
rule, unless ``WOOF_ENABLE_WRF_HYDRO=1``.
"""

from __future__ import annotations

import hashlib

import pytest

from woof import optin

REASON = optin.unavailable_reason("wrf_hydro")
pytestmark = pytest.mark.skipif(REASON is not None, reason=f"opt-in rule: {REASON}")

TEMPLATE = optin.optional_root() / "wrf_hydro" / "oracle" / "fixtures" / "namelist" / "hydro.namelist.wrf471"


def _nl():
    return optin.load("wrf_hydro").namelist


def _text():
    return TEMPLATE.read_text(encoding="utf-8")


def _drop_nudging(text):
    return text[:text.index("&NUDGING_nlist")]


def test_template_is_wrfs():
    digest = hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
    assert digest == "72d1017ff4fc486bfb6e3fa1f3d2205955662428cbd1eb0bfe0a2030bb8e19dd"


def test_every_template_key_is_one_wrf_hydro_reads():
    nl = _nl()
    groups = nl.parse(_text())
    assert set(groups["hydro_nlist"]) <= nl.HYDRO_NLIST_KEYS
    assert set(groups["reservoir_nlist"]) <= nl.RESERVOIR_NLIST_KEYS
    assert groups["hydro_nlist"]["zsoil8"] == [-0.10, -0.40, -1.00, -2.00]


def test_nudging_group_is_refused_by_name():
    nl = _nl()
    with pytest.raises(nl.UnportedOption, match="NUDGING_nlist"):
        nl.map_hydro_nlist(nl.parse(_text()))


def test_template_with_the_diffusive_wave_maps():
    nl = _nl()
    cfg = nl.map_hydro_nlist(nl.parse(_drop_nudging(_text())))
    assert cfg.channel and cfg.channel_option == 3
    assert cfg.lake_option == 1


def test_template_with_muskingum_cunge_maps():
    nl = _nl()
    cfg = nl.map_hydro_nlist(nl.parse(_drop_nudging(_text()).replace(
        "channel_option = 3", "channel_option = 2")))
    assert (cfg.nsoil, cfg.zsoil, cfg.dxrt, cfg.aggfactrt) == (4, (-0.1, -0.4, -1.0, -2.0), 250.0, 4)
    assert (cfg.dtrt_ter, cfg.dtrt_ch) == (10.0, 10.0)
    assert cfg.subsurface and cfg.overland and cfg.channel
    assert (cfg.rt_option, cfg.channel_option, cfg.gwbaseswcrt) == (1, 2, 1)
    assert cfg.io["geo_finegrid_flnm"] == "./DOMAIN/Fulldom_hires.nc"


@pytest.mark.parametrize("edit,word", [
    (("GWBASESWCRT = 1", "GWBASESWCRT = 4"), "UDMP bucket"),
    (("UDMP_OPT = 0", "UDMP_OPT = 1"), "UDMAP|UDMP"),
    (("bucket_loss = 0", "bucket_loss = 1"), "bucket loss"),
    (("sys_cpl = 1", "sys_cpl = 2"), "sys_cpl"),
])
def test_unported_options_are_refused_by_name(edit, word):
    nl = _nl()
    text = _drop_nudging(_text()).replace("channel_option = 3", "channel_option = 2")
    assert edit[0] in text
    with pytest.raises(nl.UnportedOption, match=word):
        nl.map_hydro_nlist(nl.parse(text.replace(*edit)))


def test_unknown_key_is_refused():
    nl = _nl()
    text = _drop_nudging(_text()).replace("channel_option = 3", "channel_option = 2")
    with pytest.raises(ValueError, match="does not read"):
        nl.map_hydro_nlist(nl.parse(text.replace("UDMP_OPT = 0", "UDMP_OPT = 0\nSPEED_HACK = 1")))


def _lakes(text, option):
    return text.replace("lake_option = 1", option)


@pytest.mark.parametrize("option,route_lake,expected", [
    ("lake_option = 1", True, 1), ("lake_option = 2", True, 2), ("lake_option = 0", True, 0),
    ("", True, 1), ("", False, 0), ("lake_option = 3", True, 1)])
def test_lake_option_resolves_as_read_rt_nlst_does(option, route_lake, expected):
    # config.F90 :762-778: 3 without a DA type falls back to 1; unset is 1
    # with a route_lake_f and 0 without one; 0 clears route_lake_f and outlake.
    nl = _nl()
    text = _lakes(_drop_nudging(_text()), option)
    if not route_lake:
        text = text.replace('route_lake_f = "./DOMAIN/LAKEPARM.nc"', "")
    cfg = nl.map_hydro_nlist(nl.parse(text))
    assert cfg.lake_option == expected
    if expected == 0:
        assert "route_lake_f" not in cfg.io and "outlake" not in cfg.io


def test_the_persistence_and_rfc_reservoirs_are_refused_by_name():
    nl = _nl()
    text = _lakes(_drop_nudging(_text()), "lake_option = 3")
    for key in ("reservoir_persistence_usgs", "reservoir_persistence_usace", "reservoir_rfc_forecasts"):
        edited = text.replace(f"{key} = .FALSE.", f"{key} = .TRUE.")
        assert edited != text
        with pytest.raises(nl.UnportedOption, match=key):
            nl.map_hydro_nlist(nl.parse(edited))
    # without lake_option = 3, WRF never reads &reservoir_nlist
    cfg = nl.map_hydro_nlist(nl.parse(text.replace("lake_option = 3", "lake_option = 1").replace(
        "reservoir_persistence_usgs = .FALSE.", "reservoir_persistence_usgs = .TRUE.")))
    assert cfg.lake_option == 1


def test_a_reservoir_key_in_hydro_nlist_is_unknown_and_lake_option_4_is_invalid():
    nl = _nl()
    text = _drop_nudging(_text())
    with pytest.raises(ValueError, match="does not read"):
        nl.map_hydro_nlist(nl.parse(text.replace("UDMP_OPT = 0", "UDMP_OPT = 0\nreservoir_type_specified = .TRUE.")))
    with pytest.raises(ValueError, match="lake_option = 4"):
        nl.map_hydro_nlist(nl.parse(_lakes(text, "lake_option = 4")))


def test_the_2d_groundwater_model_and_its_channel_exchange_map():
    nl = _nl()
    text = _drop_nudging(_text()).replace("GWBASESWCRT = 1", "GWBASESWCRT = 3")
    cfg = nl.map_hydro_nlist(nl.parse(text.replace(
        "UDMP_OPT = 0", "UDMP_OPT = 0\ngwChanCondSw = 1\ngwChanCondConstIn = 0.0001\n"
        "gwChanCondConstOut = 0.0002\ngwsoilcpl = 1\nGwPreCycles = 3")))
    assert (cfg.gwbaseswcrt, cfg.gw_chan_cond_sw, cfg.gwsoilcpl, cfg.gw_pre_cycles) == (3, 1, 1, 3)
    assert (cfg.gw_chan_cond_const_in, cfg.gw_chan_cond_const_out) == (0.0001, 0.0002)


def test_rt_option_2_maps_and_3_is_invalid():
    nl = _nl()
    text = _drop_nudging(_text())
    assert nl.map_hydro_nlist(nl.parse(text.replace("rt_option = 1", "rt_option = 2"))).rt_option == 2
    with pytest.raises(ValueError, match="rt_option = 3"):
        nl.map_hydro_nlist(nl.parse(text.replace("rt_option = 1", "rt_option = 3")))
