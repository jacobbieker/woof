"""Host checks of hex analysis nudging: WRF controls, named refusals, the analysis clock."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import nudging
from woof.spectral_nudging.config import NudgingConfig


def _grid(**changes):
    row = dict(enabled=True, mode="grid", convention="wrf", guv=3e-4, gt=3e-4, gq=3e-4,
               gfdda_interval_m=60.0, gfdda_end_h=1.0)
    row.update(changes)
    return NudgingConfig(**row)


def test_wrf_namelist_and_toml_give_the_same_controls(tmp_path):
    (tmp_path / "namelist.input").write_text(
        "&fdda\n grid_fdda = 1, gfdda_interval_m = 60, gfdda_end_h = 6,\n"
        " guv = 0.0003, gt = 0.0003, gq = 0.0001,\n if_no_pbl_nudging_uv = 1, if_zfac_t = 1, k_zfac_t = 12,\n"
        " dk_zfac_t = 2, fgdt = 2, fgdtzero = 1, if_ramping = 1, dtramp_min = 60,\n/\n")
    (tmp_path / "n.toml").write_text(
        "[nudging]\nmode = 'grid'\nguv = 0.0003\ngt = 0.0003\ngq = 0.0001\ngfdda_interval_m = 60.0\n"
        "gfdda_end_h = 6.0\nif_no_pbl_nudging_uv = 1\nif_no_pbl_nudging_t = 0\nif_no_pbl_nudging_q = 0\n"
        "if_zfac_t = 1\nk_zfac_t = 12\ndk_zfac_t = 2\nfgdt = 2.0\nfgdtzero = 1\nif_ramping = 1\ndtramp_min = 60.0\n")
    a = nudging.load_config(tmp_path / "namelist.input")
    b = nudging.load_config(tmp_path / "n.toml")
    for key in ("mode", "convention", "guv", "gt", "gq", "if_no_pbl_nudging_uv", "if_zfac_t", "k_zfac_t",
                "dk_zfac_t", "fgdt", "fgdtzero", "if_ramping", "dtramp_min", "gfdda_interval_m", "gfdda_end_h"):
        assert getattr(a, key) == getattr(b, key), key
    assert a.cadence_steps(30.0) == 4


def test_spectral_namelist_reads_wavenumbers(tmp_path):
    (tmp_path / "namelist.input").write_text(
        "&fdda\n grid_fdda = 2, gfdda_interval_m = 360, gfdda_end_h = 24, xwavenum = 4, ywavenum = 3,\n"
        " guv = 0.0003, gt = 0.0003, gph = 0.0,\n/\n")
    config = nudging.load_config(tmp_path / "namelist.input")
    assert (config.mode, config.cutoff_x, config.cutoff_y) == ("spectral", 4, 3)
    assert not nudging.refusals(config, limited_area=True)


def test_off_namelist_is_off(tmp_path):
    (tmp_path / "namelist.input").write_text("&fdda\n grid_fdda = 0,\n/\n")
    assert nudging.load_config(tmp_path / "namelist.input") is None


@pytest.mark.parametrize("text,code", [
    ("&fdda\n grid_fdda = 2, gfdda_interval_m = 60, gfdda_end_h = 1, guv = 3e-4, gph = 3e-4,\n/\n", "HEX_NUDGING_GPH"),
    ("&fdda\n grid_fdda = 1, gfdda_interval_m = 60, gfdda_end_h = 1,\n/\n", "HEX_NUDGING_NO_FIELD"),
])
def test_named_refusals(tmp_path, text, code):
    (tmp_path / "namelist.input").write_text(text)
    config = nudging.load_config(tmp_path / "namelist.input")
    assert code in [problem.code for problem in nudging.refusals(config, limited_area=False)]


def test_regional_convention_and_nest_domains_are_refused(tmp_path):
    config = NudgingConfig(enabled=True, mode="spectral", convention="regional")
    assert "HEX_NUDGING_CONVENTION" in [p.code for p in nudging.refusals(config, limited_area=True)]
    (tmp_path / "namelist.input").write_text(
        "&fdda\n grid_fdda = 1, 1, gfdda_interval_m = 60, 60, gfdda_end_h = 1, 1, guv = 3e-4, 3e-4,\n/\n")
    with pytest.raises(nudging.HexNudgingRefusal, match="HEX_NUDGING_DOMAIN"):
        nudging.load_config(tmp_path / "namelist.input")


def _frames(*offsets):
    return [nudging.AnalysisFrame(Path(f"a{i}.nc"), "t", float(o), "") for i, o in enumerate(offsets)]


def test_analysis_clock():
    config = _grid()
    nudging.check_frames(config, _frames(0, 3600), run_seconds=3600.0, dt=30.0)
    with pytest.raises(nudging.HexNudgingRefusal, match="HEX_NUDGING_ANALYSIS_CLOCK"):
        nudging.check_frames(config, _frames(0, 1800), run_seconds=1800.0, dt=30.0)
    with pytest.raises(nudging.HexNudgingRefusal, match="HEX_NUDGING_ANALYSES_SHORT"):
        nudging.check_frames(_grid(gfdda_end_h=2.0), _frames(0, 3600), run_seconds=7200.0, dt=30.0)
    # gfdda_end_h before the run's end: analyses need cover only the window.
    nudging.check_frames(_grid(gfdda_end_h=1.0), _frames(0, 3600), run_seconds=7200.0, dt=30.0)
    with pytest.raises(nudging.HexNudgingRefusal, match="HEX_NUDGING_INTERVAL_DT"):
        nudging.check_frames(config, _frames(0, 3600), run_seconds=3600.0, dt=7.0)


def _mesh(limited, lat_span=0.3, lon_span=0.4, n=400):
    rng = np.random.default_rng(0)
    lat = rng.uniform(-lat_span / 2, lat_span / 2, n) + 0.6
    lon = rng.uniform(-lon_span / 2, lon_span / 2, n) - 1.9
    arrays = {"latCell": lat, "lonCell": lon, "areaCell": np.ones(n)}
    if limited:
        mask = np.zeros(n, dtype=np.int32)
        mask[:20] = 7
        arrays["bdyMaskCell"] = mask
    return SimpleNamespace(n_cells=n, attrs={"sphere_radius": 6371229.0}, arrays=arrays)


def test_spectral_degree_follows_the_kept_wavelength():
    config = NudgingConfig(enabled=True, mode="spectral", convention="wrf", cutoff_x=3, cutoff_y=2)
    assert nudging.spectral_degree(config, _mesh(False)) == 4      # global: max(xw, 2 yw)
    degree = nudging.spectral_degree(config, _mesh(True, lat_span=0.6, lon_span=0.8))
    # interior extent ~ 0.8 rad * cos(0.6) zonally: 2 pi * 3 / (0.8 cos 0.6) ~ 28
    assert 25 <= degree <= 32
    with pytest.raises(nudging.HexNudgingRefusal, match="HEX_SPECTRAL_NUDGING_DEGREE"):
        nudging.HarmonicFilter(_mesh(True, lat_span=0.03, lon_span=0.04),
                               nudging.spectral_degree(config, _mesh(True, lat_span=0.03, lon_span=0.04)))


def test_specified_zone_and_garbage_column_are_never_forced():
    mesh = _mesh(True, n=10)
    mesh.arrays["bdyMaskCell"] = np.array([0, 1, 5, 6, 7, 0, 0, 3, 7, 0])
    mask = nudging.nudged_cell_mask(mesh, 11)
    assert mask.tolist() == [1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0]
