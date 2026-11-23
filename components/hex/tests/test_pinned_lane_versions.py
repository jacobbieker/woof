"""Pinned oracle lanes name their scheme versions (woof.hex.mpasphys.version_defaults).

Upstream WOOF 1.1.7 made the newest MMM-physics (550b5b4), MYNN (v6.1) and TEMPO (v3.2.5) the hex defaults in
correct mode.  A lane that compares hex with pinned MPAS-A v8.4.2 or RRFS 44c0849 Fortran therefore names
the pin.  THE BREAKAGE THESE PREVENT: a lane that leaned on a default would quietly compare the newer scheme
with the pinned Fortran once the default moved.  Inside a pinned lane, a version taken from a default is
refused by name.
"""
from dataclasses import dataclass

import pytest

from woof.hex.mpasphys import gwdo_version as gwdo
from woof.hex.mpasphys import mmm_version as mmm
from woof.hex.mpasphys import ysu_v842 as ysu
from woof.hex.mpasphys.driver_v842 import resolve_namelist as resolve_v842
from woof.hex.mpasphys.rrfs import tempo_v303
from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist as resolve_rrfs
from woof.hex.mpasphys.version_defaults import (ENVIRONMENT, SELECTORS, DefaultedVersion, defaulted, pinned_lane,
                                                strict)

RRFS = "mpasphys_rrfs_44c0849"
HRRRV5 = '&physics config_physics_suite="hrrrv5" /'
MESOSCALE = '&physics config_physics_suite="mesoscale_reference" /'


def _v842(mode="correct", text=MESOSCALE):
    return resolve_v842(text, default_backend="hex-cuda", physics_mode=mode)


def _rrfs(mode="correct", extra=""):
    return resolve_rrfs(HRRRV5 + extra, table_key=RRFS, default_backend="hex-cuda", physics_mode=mode)


def test_a_default_is_returned_outside_a_lane_and_refused_inside(monkeypatch):
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    assert not strict() and defaulted("mmm_version", "550b5b4", where="here") == "550b5b4"
    with pinned_lane():
        with pinned_lane():
            assert strict()
        assert strict()
        with pytest.raises(DefaultedVersion, match=r"here: mmm_version was not named, so it took the default '550b5b4'"):
            defaulted("mmm_version", "550b5b4", where="here")
    assert not strict()
    monkeypatch.setenv(ENVIRONMENT, "1")
    assert strict()


@pytest.mark.parametrize("mode", ["correct", "exact"])
def test_mmm_generation_on_the_v842_route(mode):
    # YSU and the revised MM5 layer run on hex-cuda: the generation is the mode's default unless named.
    assert mmm.configure(_v842(mode), {}).mmm["mmm_version"] == mmm.default_version(mode)
    rrfs = _rrfs(mode)   # resolved outside the lane: its TEMPO and MYNN are other tests' business
    with pinned_lane():
        with pytest.raises(DefaultedVersion, match="mmm_version"):
            mmm.configure(_v842(mode), {})
        assert mmm.configure(_v842(mode), {"mmm_version": mmm.PINNED}).mmm["mmm_version"] == "a4baf7f"
        # No YSU/MM5 slot runs, or it runs native-cpu: the run is the pin and nothing was defaulted.
        assert mmm.configure(rrfs, {}).mmm["mmm_version"] == "a4baf7f"
        native = _v842(mode, MESOSCALE + "\n&hex_mpasphys config_pbl_backend='native-cpu' "
                                         "config_sfclayer_backend='native-cpu' /")
        assert mmm.configure(native, {}).mmm["mmm_version"] == "a4baf7f"


def test_scheme_module_arguments_and_slot_calls_name_the_generation():
    @dataclass
    class Call:
        mmm: object = None

    header = ysu.shared_device_math_header()
    assert ysu.cuda_source(header) == ysu.cuda_source(header, "a4baf7f") != ysu.cuda_source(header, "550b5b4")
    assert mmm.of(Call()) is mmm.PINNED_OPTIONS
    with pinned_lane():
        with pytest.raises(DefaultedVersion, match="ysu_v842.cuda_source"):
            ysu.cuda_source(header)
        with pytest.raises(DefaultedVersion, match="SlotCall built without an MMM choice"):
            mmm.of(Call())
        ysu.cuda_source(header, mmm.PINNED)
        assert mmm.of(Call(mmm.PINNED_OPTIONS)) is mmm.PINNED_OPTIONS


def test_drag_generation_only_where_the_suite_runs_the_ysu_drag():
    assert gwdo.configure(_v842(), {}).gwdo is None   # v842, the default in every mode
    rrfs = _rrfs()
    with pinned_lane():
        with pytest.raises(DefaultedVersion, match="gwdo_version"):
            gwdo.configure(_v842(), {})
        assert gwdo.configure(_v842(), {"gwdo_version": "v842"}).selectors["gwdo"] == gwdo.YSU
        # The RRFS suite's GSL drag is not a generation the version maps: nothing is defaulted.
        assert gwdo.configure(rrfs, {}).selectors["gwdo"] == "bl_ugwp_gwdo"


def test_tempo_and_mynn_versions_on_the_rrfs_route():
    assert _rrfs("correct").options["config_tempo_version"] == "v3.2.5"
    assert _rrfs("exact").options["config_tempo_version"] == "v3.0.3"
    assert tempo_v303.tempo_version({}) == tempo_v303.DEFAULT_TEMPO_VERSION
    named = "\n&physics_mp_tempo config_tempo_version='v3.0.3' /\n&hex_mpasphys config_mynn_version='pinned' /"
    with pinned_lane():
        for mode in ("correct", "exact"):
            with pytest.raises(DefaultedVersion, match="config_tempo_version"):
                _rrfs(mode)
            with pytest.raises(DefaultedVersion, match="config_mynn_version"):
                _rrfs(mode, "\n&physics_mp_tempo config_tempo_version='v3.0.3' /")
            config = _rrfs(mode, named)
            assert (config.options["config_tempo_version"], config.mynn_version) == ("v3.0.3", "pinned")
        with pytest.raises(DefaultedVersion, match="tempo_v303.tempo_version"):
            tempo_v303.tempo_version({})
        assert tempo_v303.tempo_version({"config_tempo_version": "v3.0.3"}) == "v3.0.3"


def test_proofs_name_every_selector_a_pinned_lane_must_name(tmp_path):
    # verify_mpas.proof_referee_versions is where every forecast proof names its pins; a selector added to
    # version_defaults.SELECTORS without a proof pin fails here.
    from woof.hex.verify_mpas import proof_referee_versions

    namelist = tmp_path / "namelist.atmosphere"
    named = set()
    for text in (HRRRV5, MESOSCALE):
        namelist.write_text(text + "\n")
        run, _ = proof_referee_versions(["woof hex", "forecast", "--physics-backend", "mpasphys"], namelist)
        named |= {part.split("=", 1)[0] for part in run[4:] if part != "--physics"}
    assert named == set(SELECTORS)
