"""Hex partial-depth band (Phase D5): configuration, refusals, truncation.

No device: the band's dataclass and namelist group, every refusal by name,
the dycore loader's handling of ``&woof_partial_nest`` (refused by name on
every route but the band's; a namelist without it maps exactly as before),
the interface-flatness check, the truncation of a full-depth host object,
and the counterclockwise normalization of ``cellsOnVertex``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from woof.hex.config import ConfigurationRefusal, dry_dycore_config_from_namelist
from woof.hex.partial_nest import (HexPartialNestBand, PartialNestRefused, band_from_namelist,
                                   configs_from_namelist, interface_height,
                                   refuse_unsupported_config, truncate)

DYCORE = """
&nhyd_model
    config_dt = 12.0
    config_monotonic = false
    config_positive_definite = false
    config_dynamics_split_steps = 3
    config_split_dynamics_transport = true
/
"""

BAND = """
&woof_partial_nest
    config_partial_nest_interface_level = 18
    config_partial_nest_time_step_ratio = 2
    config_partial_nest_relax_levels = 4
    config_partial_nest_feedback_fields = 'theta'
/
"""


def test_band_defaults_are_the_d4_lessons():
    band = HexPartialNestBand(interface_level=18, parent_time_step_ratio=2)
    assert band.face_scheme == "second_order"
    assert band.feedback_fields == ("theta",)
    assert band.mass_correction


@pytest.mark.parametrize("field,word", [("w", "delayed-feedback"), ("rho_zz", "coordinate"),
                                        ("zgrid", "vertical coordinate"), ("u", "short waves")])
def test_feedback_of_the_coordinate_w_and_u_is_refused_by_name(field, word):
    with pytest.raises(PartialNestRefused, match=word):
        HexPartialNestBand(interface_level=18, parent_time_step_ratio=2, feedback_fields=(field,))


@pytest.mark.parametrize("kwargs,word", [
    (dict(interface_level=3), "at least four layers"),
    (dict(relax_levels=5), "ring strengths"),
    (dict(interface_level=5, relax_levels=4), "does not fit"),
    (dict(face_scheme="fourth"), "face_scheme"),
    (dict(parent_time_step_ratio=0), "parent_time_step_ratio"),
])
def test_band_settings_outside_the_formulation_are_refused(kwargs, word):
    values = dict(interface_level=18, parent_time_step_ratio=2)
    values.update(kwargs)
    with pytest.raises(PartialNestRefused, match=word):
        HexPartialNestBand(**values)


def test_namelist_group_reads_the_band_and_refuses_unknown_knobs():
    assert band_from_namelist(DYCORE) is None
    band = band_from_namelist(DYCORE + BAND)
    assert (band.interface_level, band.parent_time_step_ratio, band.relax_levels) == (18, 2, 4)
    with pytest.raises(PartialNestRefused, match="unknown knob"):
        band_from_namelist(BAND.replace("relax_levels", "relax_rings"))


def test_without_the_group_the_dycore_config_is_unchanged():
    config, band = configs_from_namelist(DYCORE)
    assert band is None
    assert config == dry_dycore_config_from_namelist(DYCORE, source_release="v8.4.1")


def test_the_dycore_loader_refuses_the_group_on_every_other_route():
    with pytest.raises(ConfigurationRefusal, match="woof_partial_nest"):
        dry_dycore_config_from_namelist(DYCORE + BAND, source_release="v8.4.1")
    config, band = configs_from_namelist(DYCORE + BAND)
    assert band.interface_level == 18
    assert config == dry_dycore_config_from_namelist(DYCORE, source_release="v8.4.1")


@dataclass
class _Config:
    source_release: str = "v8.4.1"
    config_dynamics_split_steps: int = 3
    config_split_dynamics_transport: bool = True
    config_monotonic: bool = False
    config_positive_definite: bool = False
    config_apply_lbcs: bool = True
    config_physics_suite: str = "none"
    config_moist_physics: bool = False
    config_rayleigh_damp_u: bool = False
    config_mpas_cam_coef: float = 0.0
    config_les_model: str = "none"
    config_horiz_mixing: str = "off"


@pytest.mark.parametrize("change,word", [
    (dict(config_physics_suite="mesoscale_reference"), "extended column"),
    (dict(config_moist_physics=True), "config_moist_physics"),
    (dict(config_rayleigh_damp_u=True), "config_rayleigh_damp_u"),
    (dict(config_mpas_cam_coef=0.2), "config_mpas_cam_coef"),
    (dict(config_monotonic=True), "config_monotonic"),
    (dict(config_dynamics_split_steps=1), "split-three"),
    (dict(config_apply_lbcs=False), "config_apply_lbcs"),
])
def test_dycore_settings_the_band_cannot_carry_are_refused_by_name(change, word):
    refuse_unsupported_config(_Config(), role="child")
    config = _Config(**change)
    with pytest.raises(PartialNestRefused, match=word):
        refuse_unsupported_config(config, role="child")


def test_interface_must_be_a_flat_interior_face():
    z = np.tile(np.linspace(0.0, 10_000.0, 31)[:, None], (1, 5))
    assert interface_height(z, 18) == 6000.0
    z[2, 0] += 1.0
    with pytest.raises(PartialNestRefused, match="not flat"):
        interface_height(z, 2)
    with pytest.raises(PartialNestRefused, match="interior face"):
        interface_height(z, 30)


def test_truncation_keeps_the_layers_and_faces_below_the_interface():
    @dataclass
    class Host:
        rho: np.ndarray
        rho_w: np.ndarray
        zb: np.ndarray
        flat: np.ndarray
        time_seconds: float = 0.0

    nlev, k = 30, 18
    host = Host(np.zeros((nlev, 7)), np.zeros((nlev + 1, 7)), np.zeros((3, nlev, 2)),
                np.zeros(4))
    cut = truncate(host, nlev, k)
    assert cut.rho.shape == (k, 7) and cut.rho_w.shape == (k + 1, 7)
    assert cut.zb.shape == (3, k, 2) and cut.flat.shape == (4,)


def test_clockwise_triangles_are_reordered_counterclockwise():
    from woof.hex.partial_nest_coupling import counterclockwise_cells_on_vertex
    x = np.array([[1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [1.0, 0.0, 0.1]])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    cov = np.array([[0, 1, 2], [0, 2, 1], [0, 1, 3]])
    out = counterclockwise_cells_on_vertex(cov, x)
    assert out.tolist() == [[0, 1, 2], [0, 1, 2], [0, 1, 3]]
