"""WRF's inflow perturbation on a regional hex mesh's relax rings (matrix 5.7).

``woof.hex.inflow_perturbation`` adds WRF's seeding (woof.core.
inflow_perturbation: keyed draws, Eckert amplitude, PBL depth, 100 s holds,
8-cell blocks) to the rtheta_m relaxation target of the inflow faces'
relax-zone cells.  On a synthetic regional patch (a lat/lon block of cells
whose rings count inward from the edge, as bdyMaskCell does) this file holds:

* the faces, along-face blocks and inward signs of the mesh geometry;
* the increment, cell for cell, to WRF's own formulas on the same inputs;
* the hold, the face selection (inflow vs outflow), the untouched specified
  zone and interior, and the zero-depth (no PBL yet) step;
* the door: off by default, flags refused without the switch.
"""
from __future__ import annotations

from types import SimpleNamespace
import math

import numpy as np
import pytest

from woof.core import inflow_perturbation as wrf
from woof.hex import inflow_perturbation as hexip

NX = NY = 40
NLEV = 10
DT = 20.0
DX = 3000.0
R = 6371229.0


def _patch():
    """A lat/lon block near (35N, 100W) with MPAS-style connectivity and rings."""
    j, i = np.mgrid[0:NY, 0:NX]
    lat = np.radians(35.0) + (j - NY / 2) * DX / R
    lon = np.radians(-100.0) + (i - NX / 2) * DX / (R * math.cos(math.radians(35.0)))
    edge_distance = np.minimum.reduce([i, j, NX - 1 - i, NY - 1 - j])
    ring = np.where(edge_distance < 7, 7 - edge_distance, 0)
    index = (j * NX + i)
    edges = [(index[a, b], index[a, b + 1]) for a in range(NY) for b in range(NX - 1)]
    edges += [(index[a, b], index[a + 1, b]) for a in range(NY - 1) for b in range(NX)]
    coe = np.asarray(edges, dtype=np.int64)
    ring = ring.reshape(-1)
    edge_ring = np.maximum(ring[coe[:, 0]], ring[coe[:, 1]])
    mesh = SimpleNamespace(arrays={"latCell": lat.reshape(-1), "lonCell": lon.reshape(-1),
                                   "cellsOnEdge": coe, "dcEdge": np.full(len(coe), DX)},
                           attrs={"sphere_radius": R})
    return mesh, ring, edge_ring, i.reshape(-1), j.reshape(-1)


def test_geometry_faces_blocks_and_inward_signs():
    mesh, ring, edge_ring, i, j = _patch()
    g = hexip.face_geometry(mesh, ring, edge_ring, n_relax_zone=5)
    relax = (ring >= 1) & (ring <= 5)
    assert sum(g[f]["cells"].size for f in hexip.FACES) == int(relax.sum())
    west = g["west"]["cells"]
    assert np.all(i[west] < NX / 2) and np.all(np.abs(i[west] - NX / 2) >= np.abs(j[west] - NY / 2) - 1)
    # Each WRF relax row of one column shares a block: the block follows j on the west face.
    blocks = g["west"]["blocks"]
    for row in np.unique(j[west]):
        assert len(set(blocks[j[west] == row])) == 1
    assert g["west"]["n_blocks"] == pytest.approx(math.ceil(NY / wrf.BLOCK_CELLS), abs=1)
    # Inward sign: a west-face edge from an outer ring (c1) to an inner one (c2) is +1.
    coe = mesh.arrays["cellsOnEdge"]
    for e, s in zip(g["west"]["edges"], g["west"]["signs"]):
        a, b = coe[e]
        assert s == (1.0 if ring[b] < ring[a] else -1.0)


class _Runtime:
    def __init__(self, cp, u_east, *, time=0.0):
        mesh, ring, edge_ring, _, _ = _patch()
        n = ring.size
        self.padded = SimpleNamespace(base=mesh)
        self.masks_host = SimpleNamespace(bdy_mask_cell=ring, bdy_mask_edge=edge_ring)
        self.nlev, self.outer_dt = NLEV, DT
        z = np.linspace(0.0, 10000.0, NLEV + 1)[:, None] * np.ones((1, n + 1))
        self.atmosphere = SimpleNamespace(vertical=SimpleNamespace(zgrid=cp.asarray(z, dtype=cp.float32)),
                                          state=SimpleNamespace(time_seconds=time))
        coe = mesh.arrays["cellsOnEdge"]
        lon = mesh.arrays["lonCell"]
        # A uniform eastward wind: an edge's normal component (c1 -> c2) is
        # u_east when it points east, 0 when it points north.
        east = (lon[coe[:, 1]] > lon[coe[:, 0]]).astype(np.float32)
        self.u = cp.asarray(np.repeat((u_east * east)[None, :], NLEV, 0))
        rng = np.random.default_rng(3)
        self.theta = cp.asarray(rng.uniform(290, 310, (NLEV, n + 1)).astype(np.float32))
        self.rt = cp.asarray(rng.uniform(250, 350, (NLEV, n + 1)).astype(np.float32))

    def driving_values(self, name, rk_step):
        return {"u": self.u, "theta": self.theta, "rtheta_m": self.rt}[name]


def _gpu():
    return pytest.importorskip("cupy")


@pytest.mark.parametrize("mode, face", [("inflow", "west"), ("outflow", "east")])
def test_increment_is_wrfs_formulas_on_the_relax_rings(mode, face):
    cp = _gpu()
    runtime = _Runtime(cp, 6.0, time=7 * 60.0)
    pblh = 2600.0
    p = hexip.HexInflowPerturbation(runtime, seed=11, amplitude_scale=1.0, faces=mode,
                                    pblh_source=lambda: pblh)
    got = cp.asnumpy(p.target(runtime.rt, 1))
    rt, theta = cp.asnumpy(runtime.rt), cp.asnumpy(runtime.theta)
    g = p.geometry[face]
    z_half = 0.5 * (np.linspace(0, 10000, NLEV + 1)[:-1] + np.linspace(0, 10000, NLEV + 1)[1:])
    levels = wrf.perturbed_level_count(z_half, pblh)
    assert levels == 3
    refresh = wrf.refresh_index(int(7 * 60 / DT), 1, DT)
    inward = p._inward_means(levels)
    assert inward["west"] == pytest.approx(6.0) and inward["east"] == pytest.approx(-6.0)
    theta_max = wrf.face_amplitude(inward[face], 1.0)
    draw = wrf.draw_unit_pattern(11, hexip.GRID_ID, face, refresh, NLEV, g["n_blocks"])
    delta = (theta_max * draw[:, g["blocks"]]).astype(np.float32)
    delta[levels:] = 0
    cells = g["cells"]
    want = rt.copy()
    want[:, cells] = rt[:, cells] + (rt[:, cells] / theta[:, cells]) * delta
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
    touched = np.any(got != rt, axis=0)
    assert set(np.nonzero(touched)[0]) <= set(cells.tolist()) and touched[cells].any()
    ring = runtime.masks_host.bdy_mask_cell
    assert not np.any(touched[:ring.size][(ring == 0) | (ring > 5)])


def test_hold_redraw_and_zero_depth():
    cp = _gpu()
    runtime = _Runtime(cp, 6.0)
    depth = {"pblh": 0.0}
    p = hexip.HexInflowPerturbation(runtime, seed=1, pblh_source=lambda: depth["pblh"])
    assert p.target(runtime.rt, 1) is runtime.rt           # no PBL depth yet: untouched
    depth["pblh"] = 3000.0
    frames = []
    for seconds in (20.0, 80.0, 100.0, 120.0):
        runtime.atmosphere.state.time_seconds = seconds
        frames.append(cp.asnumpy(p.target(runtime.rt, 1)))
    assert np.array_equal(frames[0], frames[1])              # held for 100 s
    assert not np.array_equal(frames[1], frames[2])          # redrawn at the next hold
    assert np.array_equal(frames[2], frames[3])
    runtime.atmosphere.state.time_seconds = 20.0
    with pytest.raises(ValueError, match="went backward"):
        p.target(runtime.rt, 1)


def test_off_by_default_and_needs_a_pbl():
    assert hexip.attach(SimpleNamespace(), None, None) is None
    with pytest.raises(ValueError, match="seeds a regional mesh"):
        hexip.attach(SimpleNamespace(), None, {"seed": 0})
    with pytest.raises(ValueError, match="requires"):
        hexip.HexInflowPerturbation(SimpleNamespace(), pblh_source=None)


def test_door_flags():
    from woof.hex.forecast_door import _resolve_inflow_perturbation
    off = SimpleNamespace(inflow_perturbation=False, inflow_perturbation_seed=None,
                          inflow_perturbation_amplitude_scale=None, inflow_perturbation_faces=None)
    assert _resolve_inflow_perturbation(off) is None
    with pytest.raises(Exception, match="read only with --inflow-perturbation"):
        _resolve_inflow_perturbation(SimpleNamespace(**{**vars(off), "inflow_perturbation_seed": 3}))
    on = SimpleNamespace(**{**vars(off), "inflow_perturbation": True, "inflow_perturbation_faces": "outflow"})
    assert _resolve_inflow_perturbation(on) == {"seed": 0, "amplitude_scale": 1.0, "faces": "outflow"}
