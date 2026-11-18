"""Every WRF radiation selector value this line does not run refuses by name.

Named breakage: an unported ra_lw_physics/ra_sw_physics value used to refuse
with only "no ratified woof mapping", which names no breakage (the refusal
law).  Each value now carries the reason it is not run -- the forecast would
otherwise carry a radiation label for a scheme that never computed it.
"""
import pytest

from woof import config as config_mod
from woof.namelist_import import import_namelists

from test_namelist_import import INPUT_TEXT, _pair

#: WRF v4.6.1/v4.7.1 Registry.EM_COMMON radiation selector values.
WRF_RA_LW = (0, 1, 3, 4, 5, 7, 14, 24, 31, 99)
WRF_RA_SW = (0, 1, 2, 3, 4, 5, 7, 14, 24, 99)


@pytest.mark.parametrize("key,values,accepted", [
    ("ra_lw_physics", WRF_RA_LW, config_mod.RA_LW_PHYSICS_ACCEPTED),
    ("ra_sw_physics", WRF_RA_SW, config_mod.RA_SW_PHYSICS_ACCEPTED),
])
def test_every_wrf_value_is_run_or_refused_with_a_reason(key, values, accepted):
    for value in values:
        reason = config_mod.radiation_unported_reason(key, value)
        if value in accepted:
            assert reason == "", (key, value)
        else:
            assert len(reason) > 40, (key, value)
    assert set(config_mod.RA_UNPORTED_REASONS[key]) == set(values) - set(accepted)


# Fu-Liou-Gu (7), Held-Suarez (31), GFDL (99), old Goddard SW (2), CAM SW
# (3) and RRTMG-K (14, as the 14/14 pair) are ported now; fast RRTMG (24) is
# the WRF value still refused, on either stream, and RRTMG-K refuses any
# split pair because WRF computes both spectra in its longwave call.
@pytest.mark.parametrize("lw,sw,needle", [
    (24, 4, "fast RRTMG longwave"), (14, 4, "only as the pair"),
    (4, 24, "fast RRTMG shortwave"), (4, 14, "only as the pair"),
])
def test_the_importer_names_the_reason(tmp_path, lw, sw, needle):
    # mp 8 (Thompson): the fixture's mp 55 is refused by its own name on
    # lines that retire the ISHMAEL-as-Morrison import.
    inp = (INPUT_TEXT.replace("mp_physics = 55, 55", "mp_physics = 8, 8")
           .replace("ra_lw_physics = 4, 4", f"ra_lw_physics = {lw}, {lw}")
           .replace("ra_sw_physics = 4, 4", f"ra_sw_physics = {sw}, {sw}"))
    with pytest.raises(ValueError, match=needle):
        import_namelists(*_pair(tmp_path, inp=inp))


def test_new_goddard_is_named_in_the_config_menu_message():
    with pytest.raises(ValueError, match=r"5 \(WRF New Goddard\).*7 \(WRF Fu-Liou-Gu\)"):
        config_mod.validate_run_config(_cfg(ra_lw_physics=24, ra_sw_physics=24))


def _cfg(**kw):
    from test_checkpoint_scheme_identity_gate import _cfg as base_cfg
    return base_cfg(**kw)


def test_aer_opt_1_selects_the_legacy_rrtmg_spectrum_like_2_and_3():
    """aer_opt 1 runs only on legacy RRTMG shortwave, so an ordinary namelist
    selecting it takes that spectrum by default instead of refusing and
    asking for --rrtmg-variant (aer_opt 2 and 3 already did)."""
    from woof.physics_compat import RRTMG_VARIANT_LEGACY
    from woof.namelist_import import _default_rrtmg_variant_for_namelist
    for value in (1, 2, 3):
        inp = {"physics": {"aer_opt": [value], "bl_pbl_physics": [1]}}
        assert _default_rrtmg_variant_for_namelist(inp) == RRTMG_VARIANT_LEGACY


def test_rrtmgk_pair_imports_and_refuses_what_it_does_not_port():
    """14/14 is admitted; what rad_rrtmg_driver is not ported for refuses
    by name (rrtmg_spectral_output, aer_opt, ghg_input, icloud, no snow)."""
    config_mod.validate_run_config(_cfg(moist=True, mp_physics=8,
                                        ra_lw_physics=14, ra_sw_physics=14))
    for kw, needle in (
            ({"rrtmg_spectral_output": "bands"}, "rrtmg_spectral_output"),
            ({"aer_opt": 1}, "aer_opt=1"),
            ({"ghg_input": 1}, "ghg_input=1"),
            ({"icloud": 2}, "icloud=2"),
            ({"mp_physics": 1}, "carries QS"),
            ({"ra_sw_physics": 4}, "only as the pair")):
        cfg = dict(moist=True, mp_physics=8, ra_lw_physics=14,
                   ra_sw_physics=14)
        cfg.update(kw)
        with pytest.raises(ValueError, match=needle):
            config_mod.validate_run_config(_cfg(**cfg))


def test_the_importer_maps_rrtmgk(tmp_path):
    """A WRF 14/14 namelist imports as itself, with no radiation substitution."""
    from woof.experiment import load_experiment
    inp = (INPUT_TEXT.replace("mp_physics = 55, 55", "mp_physics = 8, 8")
           .replace("ra_lw_physics = 4, 4", "ra_lw_physics = 14, 14")
           .replace("ra_sw_physics = 4, 4", "ra_sw_physics = 14, 14"))
    toml_text, report = import_namelists(*_pair(tmp_path, inp=inp), name="rrtmgk")
    output = tmp_path / "rrtmgk.toml"
    output.write_text(toml_text)
    for dc in load_experiment(output).domains:
        assert (dc.run.ra_lw_physics, dc.run.ra_sw_physics) == (14, 14)
    assert all("ra_" not in substitution.key
               for substitution in report.substitutions)
