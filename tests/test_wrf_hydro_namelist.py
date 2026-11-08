"""hydro.namelist mapped to the WRF-Hydro opt-in, refusing unported options by name.

The fixture is WRF 4.6.1's own hydro/template/HYDRO/hydro.namelist (copied
under the opt-in, sha256 7c306e1b...).  Every test skips, naming the opt-in
rule, unless ``WOOF_ENABLE_WRF_HYDRO=1``.
"""

from __future__ import annotations

import hashlib

import pytest

from woof import optin

REASON = optin.unavailable_reason("wrf_hydro")
pytestmark = pytest.mark.skipif(REASON is not None, reason=f"opt-in rule: {REASON}")

TEMPLATE = optin.optional_root() / "wrf_hydro" / "oracle" / "fixtures" / "namelist" / "hydro.namelist.wrf461"


def _nl():
    return optin.load("wrf_hydro").namelist


def _text():
    return TEMPLATE.read_text(encoding="utf-8")


def _drop_nudging(text):
    return text[:text.index("&NUDGING_nlist")]


def test_template_is_wrfs():
    digest = hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
    assert digest == "7c306e1bfb20561f72ebecfd60a593dabf1a6bda7ab418876f94a82256efe977"


def test_every_template_key_is_one_wrf_hydro_reads():
    nl = _nl()
    groups = nl.parse(_text())
    assert set(groups["hydro_nlist"]) <= nl.HYDRO_NLIST_KEYS
    assert groups["hydro_nlist"]["zsoil8"] == [-0.10, -0.40, -1.00, -2.00]


def test_nudging_group_is_refused_by_name():
    nl = _nl()
    with pytest.raises(nl.UnportedOption, match="NUDGING_nlist"):
        nl.map_hydro_nlist(nl.parse(_text()))


def test_template_channel_option_3_is_refused_by_name():
    nl = _nl()
    with pytest.raises(nl.UnportedOption, match="channel_option = 3"):
        nl.map_hydro_nlist(nl.parse(_drop_nudging(_text())))


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
    (("rt_option = 1", "rt_option = 2"), "rt_option = 2"),
    (("GWBASESWCRT = 1", "GWBASESWCRT = 3"), "gw2d"),
    (("GWBASESWCRT = 1", "GWBASESWCRT = 4"), "UDMP bucket"),
    (("UDMP_OPT = 0", "UDMP_OPT = 1"), "UDMAP|UDMP"),
    (("reservoir_persistence_usgs = .FALSE.", "reservoir_persistence_usgs = .TRUE."), "reservoirs"),
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
