"""Regional hex RADM2 / RADM2SORG with ``have_bcs_chem = 0``: WRF's default boundary profile at the rings.

WRF (flow_dep_bdy_chem) puts bdy_chem_value / bdy_chem_value_sorgam in the specified zone's INFLOW cells and
copies the inner neighbour into OUTFLOW cells.  Hex drives chemistry rows through MPAS's scalar driving pool
whatever the flow's sign, so the user's decision (2026-11-12) is that the profile drives the whole specified
zone (and nudges the relaxation zone), outflow cells included.  These checks hold the reader to that:

* the pool gains every row's ``lbc_<row>`` for radm2 / radm2sorg without a boundary source, and nothing for a
  set without the profile (cams_aq still needs have_bcs_chem = 1);
* each row's boundary field is the WRF path's own hook (chem_radm2.boundary_inflow,
  chem_sorgam.boundary_inflow) at the mesh's mass-level heights with alt and convfac from the file's state,
  word for word, and the file's met words pass through untouched; no data-store fill is asked for.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import chem_wrf

NZ, NCOL = 12, 6


def _raw(name, **switches):
    return {"schema": "gpuwm-hex.chemistry/v1", "wrf_chem": {"chem_sets": name, **switches}}


def test_the_profile_drives_every_row_only_without_a_boundary_source():
    for name in ("radm2", "radm2sorg"):
        raw = _raw(name)
        assert chem_wrf.profile_boundary(raw)
        rows = chem_wrf.load_table(raw)[1].transported
        assert chem_wrf.lbc_scalar_names(raw) == tuple("lbc_" + row.name for row in rows)
    assert not chem_wrf.profile_boundary(_raw("cams_aq"))
    assert chem_wrf.lbc_scalar_names(_raw("cams_aq")) == ()


class _Runtime:
    coupling = "none"

    def __init__(self, raw):
        self.raw = raw
        self.set_name, self.table, _ = chem_wrf.load_table(raw)
        self.rows = self.table.transported


def _met(rng):
    rho = np.sort(rng.uniform(0.3, 1.2, (NCOL, NZ)).astype(np.float32), axis=1)[:, ::-1]
    return {"lbc_rho": np.ascontiguousarray(rho),
            "lbc_theta": np.sort(rng.uniform(285.0, 340.0, (NCOL, NZ)), axis=1).astype(np.float32),
            "lbc_qv": rng.uniform(1e-4, 1.5e-2, (NCOL, NZ)).astype(np.float32),
            "lbc_u": rng.standard_normal((3 * NCOL, NZ)).astype(np.float32)}


@pytest.mark.parametrize("name", ["radm2", "radm2sorg"])
def test_the_reader_fills_the_rows_with_the_wrf_paths_own_profile(name, monkeypatch):
    import woof.chem_source_init as init
    from woof.core import chem_radm2, chem_sorgam
    from woof.hex.lbc import LbcFile

    monkeypatch.setattr(init, "fill_boundary_sources",
                        lambda *a, **k: pytest.fail("the profile asked the data store"))
    rng = np.random.default_rng(11)
    met = _met(rng)
    valid = datetime(2025, 5, 31, 1)

    def base(path, *, extra_scalars=()):
        return LbcFile(Path(path), valid, dict(met))

    terrain = rng.uniform(0.0, 1500.0, NCOL + 1).astype(np.float32)
    zgrid = (terrain[None, :] + np.linspace(0.0, 18000.0, NZ + 1, dtype=np.float32)[:, None]).astype(np.float32)
    raw = _raw(name)
    runtime = _Runtime(raw)
    lat = np.linspace(30.0, 35.0, NCOL + 1)
    reader = chem_wrf.BoundaryReader(runtime, latitude=lat, longitude=-lat, base=base, heights=zgrid)
    admitted = reader("lbc.nc", extra_scalars=reader.names)
    for key, value in met.items():
        assert admitted.fields[key] is value
    # The WRF path's hooks on the same inputs.
    z = (np.float32(0.5) * (zgrid[:-1] + zgrid[1:]))[:, :NCOL]
    p = chem_wrf.lbc_pressure(met)
    t = met["lbc_theta"].astype(np.float64).T * (p / 1.0e5) ** (287.0 / (3.5 * 287.0))
    alt = (1.0 / met["lbc_rho"].astype(np.float64).T).astype(np.float32)
    convfac = (p / 8.314510 / t).astype(np.float32)
    expect = dict(chem_radm2.boundary_inflow(runtime.table, z, alt, convfac))
    if name == "radm2sorg":
        expect.update(chem_sorgam.boundary_inflow(runtime.table, z, alt, convfac))
    assert set(expect) == {row.name for row in runtime.rows}
    for field, row in zip(reader.names, runtime.rows):
        got = admitted.fields[field]
        assert got.shape == (NCOL, NZ) and got.dtype == np.float32
        assert got.tobytes() == np.ascontiguousarray(expect[row.name].T, np.float32).tobytes(), row.name
    assert "outflow" in reader.receipt()["values"]


def test_a_profile_reader_needs_the_mesh_heights():
    with pytest.raises(ValueError, match="zgrid"):
        chem_wrf.BoundaryReader(_Runtime(_raw("radm2")), latitude=np.zeros(3), longitude=np.zeros(3))


def test_an_init_2d_static_named_like_a_row_leaves_the_rows_to_the_chemistry(tmp_path):
    """An MPAS init carries the GWDO statistic ``ol2`` (nCells), RADM2 has an ``ol2`` gas: the static is not a
    species plane, so the rows still start from the chemistry's own cold start instead of being refused as a
    partial carried row."""
    import json
    from types import SimpleNamespace
    from woof.hex.chem_transport import configure
    from woof.hex.drivers import run_cuda_v841_forecast as forecast
    from woof.hex.species_row import WSM6_SPECIES_ROW
    path = tmp_path / "chemistry.json"
    path.write_text(json.dumps(_raw("radm2")))
    row = configure(path)
    atmospheric = WSM6_SPECIES_ROW.names()
    names = (*atmospheric, *row.appended_scalar_names)
    assert "ol2" in names
    plane = SimpleNamespace(dimensions=("Time", "nCells", "nVertLevels"))
    variables = {name: plane for name in atmospheric}
    variables["ol2"] = SimpleNamespace(dimensions=("nCells",))
    assert forecast.select_initial_scalar_names(variables, names) == atmospheric
    variables["ol2"] = plane
    with pytest.raises(ValueError, match="partial model species row"):
        forecast.select_initial_scalar_names(variables, names)
