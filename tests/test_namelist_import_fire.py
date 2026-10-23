"""WRF v4.7.1's &fire group through the namelist importer.

The coupled fire (woof/fire/coupling.py) reads WRF's own &fire keys,
&domains sr_x/sr_y and &dynamics tracer_opt = 3; the importer translates
them, drops what reaches nothing with a reason, and refuses by name what
woof does not run.  A namelist whose ifire is 0 everywhere imports to the
same TOML as one with no &fire at all.
"""
from __future__ import annotations

import tomllib

import pytest

from woof.experiment import build_experiment
from woof.namelist_import import NamelistRefusal, import_namelists

#: The synthetic two-domain pair of tests/test_namelist_import.py, copied
#: here because that module imports a verify bundle this one does not need.
WPS_TEXT = """\
&share
 wrf_core = 'ARW',
 max_dom = 2,
 start_date = '1999-05-03_12:00:00', '1999-05-03_12:00:00',
 end_date   = '1999-05-03_18:00:00', '1999-05-03_18:00:00',
 interval_seconds = 21600,
 io_form_geogrid = 2,
/
&geogrid
 parent_id         = 1, 1,
 parent_grid_ratio = 1, 3,
 i_parent_start    = 1, 40,
 j_parent_start    = 1, 30,
 e_we              = 101, 61,
 e_sn              = 81, 61,
 geog_data_res     = 'default', 'default',
 dx = 12000,
 dy = 12000,
 map_proj = 'lambert',
 ref_lat   = 39.7,
 ref_lon   = -83.9,
 truelat1  = 30.0,
 truelat2  = 60.0,
 stand_lon = -83.9,
 geog_data_path = '/geog',
/
&ungrib
 out_format = 'WPS',
 prefix = 'ERA5',
/
&metgrid
 fg_name = 'ERA5',
/
"""

INPUT_TEXT = """\
&time_control
 run_hours = 6,
 start_year = 1999, 1999,
 start_month = 05, 05,
 start_day = 03, 03,
 start_hour = 12, 12,
 end_year = 1999, 1999,
 end_month = 05, 05,
 end_day = 03, 03,
 end_hour = 18, 18,
 interval_seconds = 21600,
 input_from_file = .true., .true.,
 history_interval = 60, 15,
 restart = .false.,
 restart_interval = 60,
/
&domains
 time_step = 60,
 max_dom = 2,
 e_we = 101, 61,
 e_sn = 81, 61,
 e_vert = 9, 9,
 eta_levels = 1.0, 0.9, 0.8, 0.7, 0.6,
              0.5, 0.4, 0.2, 0.0,
 p_top_requested = 5000,
 dx = 12000.0, 4000.0,
 dy = 12000.0, 4000.0,
 grid_id = 1, 2,
 parent_id = 0, 1,
 i_parent_start = 1, 40,
 j_parent_start = 1, 30,
 parent_grid_ratio = 1, 3,
 parent_time_step_ratio = 1, 3,
 feedback = 0,
 smooth_option = 0,
/
&physics
 mp_physics = 55, 55,
 ra_lw_physics = 4, 4,
 ra_sw_physics = 4, 4,
 radt = 12, 3,
 sf_sfclay_physics = 91, 91,
 sf_surface_physics = 2, 2,
 bl_pbl_physics = 11, 11,
 bldt = 0, 0,
 cu_physics = 1, 0,
 cudt = 5, 0,
/
&dynamics
 hybrid_opt = 2,
 etac = 0.2,
 w_damping = 1,
 epssm = 0.5,
 diff_opt = 2, 2,
 km_opt = 4, 4,
 mix_full_fields = .true., .true.,
 diff_6th_opt = 2, 2,
 diff_6th_factor = 0.12, 0.10,
 diff_6th_slopeopt = 1, 1,
 base_temp = 2.90D2,
 damp_opt = 3,
 zdamp = 2*5000.,
 dampcoef = 0.2, 0.2,
 khdif = 0, 0,
 kvdif = 0, 0,
 non_hydrostatic = 2*.true.,
 use_theta_m = 0,
 moist_adv_opt = 1, 1,
/
&bdy_control
 spec_bdy_width = 5,
 spec_zone = 1,
 relax_zone = 4,
 specified = .true., .false.,
 nested = .false., .true.,
/
"""


def _pair(tmp_path, wps=WPS_TEXT, inp=INPUT_TEXT):
    wps_path = tmp_path / "namelist.wps"
    inp_path = tmp_path / "namelist.input"
    wps_path.write_text(wps)
    inp_path.write_text(inp)
    return wps_path, inp_path


FIRE = """\
&fire
 ifire = 2, 2,
 fire_fuel_read = 0, 0,
 fire_fuel_cat = 3, 3,
 fire_num_ignitions = 2, 2,
 fire_ignition_start_x1 = 1000., 1000.,
 fire_ignition_start_y1 = 800., 800.,
 fire_ignition_end_x1 = 1000., 1000.,
 fire_ignition_end_y1 = 1400., 1400.,
 fire_ignition_radius1 = 25., 25.,
 fire_ignition_ros1 = 0.5, 0.5,
 fire_ignition_start_time1 = 30., 30.,
 fire_ignition_end_time1 = 40., 40.,
 fire_ignition_start_x2 = 2000., 2000.,
 fire_ignition_start_y2 = 900., 900.,
 fire_ignition_radius2 = 20., 20.,
 fire_ignition_start_time2 = 60., 60.,
 fire_wind_height = 1., 1.,
 fire_ext_grnd = 30., 30.,
 fire_ext_crwn = 40., 40.,
 fire_print_msg = 1,
 fire_boundary_guard = 8,
/
"""
SR = " sr_x = 4, 4,\n sr_y = 4, 4,\n"


def _input(fire=FIRE, domains=SR, dynamics="", replace=()):
    text = INPUT_TEXT
    for old, new in replace:
        assert old in text, old
        text = text.replace(old, new)
    if domains:
        text = text.replace("&domains\n", "&domains\n" + domains)
    if dynamics:
        text = text.replace("&dynamics\n", "&dynamics\n" + dynamics)
    return text + fire


def _import(tmp_path, inp, **kwargs):
    return import_namelists(*_pair(tmp_path, inp=inp), **kwargs)


def _runs(text):
    exp = build_experiment(tomllib.loads(text), source="test")
    return [dc.run for dc in exp.domains]


def test_fire_imports_onto_every_domain(tmp_path):
    text, report = _import(tmp_path, _input())
    shared = tomllib.loads(text)["shared"]
    assert shared["ifire"] == 2
    assert shared["sr_x"] == shared["sr_y"] == 4
    assert shared["fire_wind_height"] == 1.0
    lines = shared["fire_ignitions"]
    assert [line["radius"] for line in lines] == [25.0, 20.0]
    assert lines[1]["end_x"] == 0.0              # verbatim; WRF expands
    for run in _runs(text):
        assert run.ifire == 2 and (run.sr_x, run.sr_y) == (4, 4)
        assert run.fire_fuel_read == 0 and run.fire_fuel_cat == 3
        assert (run.fire_ext_grnd, run.fire_ext_crwn) == (30.0, 40.0)
        assert len(run.fire_ignitions) == 2
        assert run.fire_ignitions[0][0] == 1000.0       # start_x
        assert run.fire_ignitions[0][8] == 25.0         # radius
    dropped = {(d.section, d.key) for d in report.dropped}
    assert ("fire", "fire_print_msg") in dropped
    assert ("fire", "fire_boundary_guard") in dropped


def test_a_differing_column_goes_per_domain(tmp_path):
    fire = FIRE.replace(" ifire = 2, 2,", " ifire = 0, 2,")
    text, _ = _import(tmp_path, _input(fire=fire))
    document = tomllib.loads(text)
    assert "ifire" not in document["shared"]
    assert document["domain"][1]["ifire"] == 2
    root, nest = _runs(text)
    assert root.ifire == 0 and nest.ifire == 2


def test_fortran_tail_keeps_the_registry_default(tmp_path):
    """``ifire = 2,`` on two domains burns on the root only, as wrf.exe."""
    fire = FIRE.replace(" ifire = 2, 2,", " ifire = 2,")
    text, _ = _import(tmp_path, _input(fire=fire))
    root, nest = _runs(text)
    assert root.ifire == 2 and nest.ifire == 0


def test_tracer_opt_3_is_the_smoke_tracer_with_wrfs_factor(tmp_path):
    text, _ = _import(tmp_path, _input(dynamics=" tracer_opt = 3, 3,\n"))
    shared = tomllib.loads(text)["shared"]
    assert shared["smoke_tracer"] is True
    assert shared["fire_smoke_factor"] == "fire_tracer_smoke"
    for run in _runs(text):
        assert run.smoke_tracer and run.fire_smoke_factor == \
            "fire_tracer_smoke"


@pytest.mark.parametrize("change,key", [
    ((" ifire = 2, 2,", " ifire = 1, 1,"), "ifire"),
    ((" fire_print_msg = 1,", " fire_print_msg = 1,\n fire_sfc_flx = 1,"),
     "fire_sfc_flx"),
    ((" fire_print_msg = 1,", " fire_print_msg = 1,\n fire_smk_scheme = 1,"),
     "fire_smk_scheme"),
    ((" fire_print_msg = 1,", " fire_print_msg = 1,\n fmoist_run = .true.,"),
     "fmoist_run"),
    ((" fire_print_msg = 1,",
      " fire_print_msg = 1,\n fire_is_real_perim = .true.,"),
     "fire_is_real_perim"),
    ((" fire_print_msg = 1,",
      " fire_print_msg = 1,\n fire_lsm_zcoupling = .true.,"),
     "fire_lsm_zcoupling"),
    ((" fire_print_msg = 1,", " fire_print_msg = 1,\n fire_upwinding = 8,"),
     "fire_upwinding"),
    ((" fire_print_msg = 1,", " fire_print_msg = 1,\n fire_fmc_read = 0,"),
     "fire_fmc_read"),
    ((" fire_print_msg = 1,", " fire_print_msg = 1,\n fire_test_steps = 4,"),
     "fire_test_steps"),
    ((" fire_print_msg = 1,",
      " fire_print_msg = 1,\n fire_const_time = 100.,"), "fire_const_time"),
    ((" fire_print_msg = 1,",
      " fire_print_msg = 1,\n fs_firebrand_gen_lim = 10,"),
     "fs_firebrand_gen_lim"),
    ((" fire_print_msg = 1,",
      " fire_print_msg = 1,\n fire_mountain_type = 1,"),
     "fire_mountain_type"),
    ((" fire_print_msg = 1,", " fire_print_msg = 1,\n fire_frobnicate = 1,"),
     "fire_frobnicate"),
])
def test_what_woof_does_not_run_is_refused_by_name(tmp_path, change, key):
    fire = FIRE.replace(*change)
    with pytest.raises(ValueError, match=key):
        _import(tmp_path, _input(fire=fire))


def test_other_tracer_packages_are_refused(tmp_path):
    # Any other package is the unmapped key it always was.
    with pytest.raises(NamelistRefusal, match=r"unmapped key.*tracer_opt"):
        _import(tmp_path, _input(dynamics=" tracer_opt = 2, 2,\n"))
    with pytest.raises(NamelistRefusal, match="tracer_opt"):
        _import(tmp_path, _input(dynamics=" tracer_opt = 3, 0,\n"))


def test_fire_off_imports_as_no_fire_at_all(tmp_path):
    reference, _ = _import(tmp_path, INPUT_TEXT)
    off = FIRE.replace(" ifire = 2, 2,", " ifire = 0, 0,").replace(
        " fire_print_msg = 1,", " fire_print_msg = 1,\n fire_sfc_flx = 1,")
    text, report = _import(tmp_path, _input(fire=off))
    assert text == reference
    dropped = {(d.section, d.key) for d in report.dropped}
    for key in ("ifire", "fire_wind_height", "fire_ignition_radius1",
                "fire_num_ignitions", "fire_sfc_flx"):
        assert ("fire", key) in dropped, key
    assert ("domains", "sr_x") in dropped
    # tracer_opt = 0 likewise changes nothing.
    text, _ = _import(tmp_path, _input(fire=off,
                                       dynamics=" tracer_opt = 0, 0,\n"))
    assert text == reference
