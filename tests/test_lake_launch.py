"""``woof go --lakes``: one opt-in for the lake model and its initial state."""
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from woof.lake_launch import MODES, lake_settings, prepare_launch


def test_clm_binds_every_domain_and_keeps_the_original(tmp_path):
    original = tmp_path / "forecast.toml"
    original.write_text('[experiment]\nname="demo"\n[shared]\nlake_state_path="x.nc"\nmp_physics=8\n'
                        '[[domain]]\ngrid_id=1\n[[domain]]\ngrid_id=2\nsf_lake_physics=0\nuse_lakedepth=0\n'
                        '[fetch]\nout="relative/forcing"\n')
    before = original.read_bytes()
    args = SimpleNamespace(config=original, lakes="clm")
    prepare_launch(args)
    raw = tomllib.loads(Path(args.config).read_text())
    assert original.read_bytes() == before and Path(args.config).parent == original.parent
    assert Path(args.config).name.startswith("forecast.lakes-clm-")
    assert raw["fetch"]["out"] == "relative/forcing" and raw["shared"]["mp_physics"] == 8
    assert {k: raw["shared"][k] for k in MODES["clm"]} == MODES["clm"]
    assert raw["shared"]["lake_state_source"] == "auto+era5" and "lake_state_path" not in raw["shared"]
    assert all(not set(d) & {"sf_lake_physics", "use_lakedepth"} for d in raw["domain"])
    # The same request names the same file; a second launch reuses it.
    again = SimpleNamespace(config=original, lakes="clm")
    prepare_launch(again)
    assert again.config == args.config


@pytest.mark.parametrize("mode", ["ic", "off"])
def test_ic_and_off_leave_the_lake_model_off(mode):
    raw = lake_settings({"shared": {"sf_lake_physics": 1}}, mode)
    assert raw["shared"]["sf_lake_physics"] == 0
    assert raw["shared"]["lake_state_source"] == ("auto" if mode == "off" else "auto+era5")


def test_absent_switch_reads_nothing_and_hex_is_refused_by_name():
    args = SimpleNamespace(config=Path("absent.toml"))
    prepare_launch(args)
    assert args.config == Path("absent.toml")
    with pytest.raises(ValueError, match="engine = 'hex'"):
        lake_settings({"model": {"engine": "hex"}}, "clm")


def test_every_mode_is_a_valid_run_configuration():
    from woof.config import validate_run_config
    from test_ruc_mosaic_options import _config
    for mode, settings in MODES.items():
        validate_run_config(_config(**settings))
