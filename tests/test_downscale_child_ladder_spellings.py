"""``woof downscale --child-levels``: the explicit and band-biased spellings.

``N,STRETCH`` is pinned by ``tests/test_downscale_child_levels.py``; this
file pins ``eta:LIST|FILE`` and ``bands:NZ,...`` and that both reach the
plan through the one shared resolution function.
"""

from __future__ import annotations

import json

import pytest

from woof.downscale import (
    CHILD_LEVELS_SPELLINGS,
    _parse_child_levels,
    child_levels_kind,
    child_levels_need_column,
)
from woof.offline_child import OfflineChildContractError

LADDER = (1.0, 0.99, 0.97, 0.94, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0)
COLUMN = {"p_top_pa": 5000.0, "hybrid_opt": 2, "etac": 0.2,
          "surface_elevation_m": 0.0}


def test_the_three_spellings_are_told_apart():
    assert child_levels_kind(None) is None
    assert child_levels_kind("96,2.5") == "stretch"
    assert child_levels_kind("eta:1,0.5,0") == "eta"
    assert child_levels_kind("bands:60,0:2000:30:agl") == "bands"
    assert child_levels_need_column("bands:60") is True
    assert child_levels_need_column("eta:1,0") is False
    assert len(CHILD_LEVELS_SPELLINGS) == 3


def test_an_explicit_comma_ladder_is_taken_verbatim():
    spec = "eta:" + ",".join(repr(v) for v in LADDER)
    assert _parse_child_levels(spec) == LADDER


def test_an_explicit_ladder_file_is_read_like_eta_levels(tmp_path):
    from woof.vertical_bands import write_eta_file

    path = write_eta_file(tmp_path / "child.eta", LADDER,
                          header=("a hand-written ladder",))
    assert _parse_child_levels(f"eta:{path}") == LADDER


@pytest.mark.parametrize("ladder", [
    (1.0, 0.5, 0.0),                       # too few levels
    (1.0, 0.9, 0.95, 0.5, 0.2, 0.0),       # not decreasing
    (0.99, 0.9, 0.7, 0.5, 0.2, 0.0),       # does not start at 1
])
def test_a_malformed_explicit_ladder_is_refused(ladder):
    spec = "eta:" + ",".join(repr(v) for v in ladder)
    with pytest.raises(OfflineChildContractError, match="--child-levels"):
        _parse_child_levels(spec)


def test_a_band_ladder_holds_its_bands_under_the_parents_top():
    from woof.vertical_bands import dz_profile, levels_in_band, parse_band

    spec = "bands:80,0:2000:30:agl,dzbot=20"
    eta = _parse_child_levels(spec, column=COLUMN)
    assert len(eta) == 81 and eta[0] == 1.0 and eta[-1] == 0.0
    assert all(b < a for a, b in zip(eta, eta[1:]))
    import numpy as np

    profile = dz_profile(np.asarray(eta), p_top_pa=5000.0)
    assert levels_in_band(profile, parse_band("0:2000:30:agl")) >= 30


def test_an_agl_band_stands_on_the_parents_ground_unless_told_otherwise():
    spec = "bands:80,0:2000:30:agl,dzbot=20"
    low = _parse_child_levels(spec, column=COLUMN)
    high = _parse_child_levels(spec, column={**COLUMN,
                                             "surface_elevation_m": 2500.0})
    explicit = _parse_child_levels(spec + ",ground=0", column={
        **COLUMN, "surface_elevation_m": 2500.0})
    assert low != high
    assert explicit == low


def test_a_band_ladder_without_the_parents_top_is_refused_by_name():
    with pytest.raises(OfflineChildContractError, match="model top"):
        _parse_child_levels("bands:80,0:3000:45:agl")
    # ... but its spelling alone can be checked before the parent is read.
    assert _parse_child_levels("bands:80,0:3000:45:agl",
                               syntax_only=True) is None


@pytest.mark.parametrize("spec, words", [
    ("bands:", "expected"),
    ("bands:x,0:3000:45", "level count"),
    ("bands:80,0:3000", "LO:HI:N"),
    ("bands:80,0:3000:45,colour=red", "unknown key"),
    ("bands:80,dzbot=1,dzbot=2", "twice"),
    ("bands:80,reference=mars", "reference"),
    ("bands:80,dzbot=nan", "finite"),
])
def test_a_malformed_band_spec_is_refused_at_syntax(spec, words):
    with pytest.raises(OfflineChildContractError, match=words):
        _parse_child_levels(spec, syntax_only=True)


def test_an_unbuildable_band_ladder_is_refused_with_the_generators_words():
    with pytest.raises(OfflineChildContractError, match="cannot reach"):
        _parse_child_levels("bands:40,0:3000:30:agl", column=COLUMN)


def _plan(capsys):
    out = capsys.readouterr().out
    return json.loads(out[out.index("{"):])


@pytest.mark.parametrize("spelling", ["eta", "bands"])
def test_both_spellings_reach_the_plan_on_the_config_route(
        tmp_path, capsys, spelling):
    """The plan is priced on the spelled ladder, through the shared door."""
    from woof.cli import main as cli_main
    from test_offline_child_tiles import _child_toml, _parent_archive

    namelist = _parent_archive(tmp_path)
    if spelling == "eta":
        spec = "eta:" + ",".join(repr(v) for v in LADDER)
        nz = len(LADDER) - 1
    else:
        # The fixture archive's model top is 100 hPa (P_TOP 10000 Pa).
        spec = "bands:60,0:2000:20:agl,dzbot=40"
        nz = 60
    plain = _child_toml(tmp_path / "plain")
    assert cli_main([
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--accept-parent-cadence", "--child-config", str(plain),
        "--child-levels", spec,
        "--out", str(tmp_path / "child-run"), "--dry-run"]) == 0
    plan = _plan(capsys)
    assert plan["child_levels_override"] == spec
    assert plan["effective_nz"] == nz
    assert plan["child_grid"]["nz"] == nz
    eta = plan["effective_eta_levels"]
    assert eta[0] == 1.0 and eta[-1] == 0.0 and len(eta) == nz + 1
    if spelling == "eta":
        assert tuple(eta) == LADDER
