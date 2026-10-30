"""The woof.radtran spectral diagnostic on the MPAS suite route (woof.hex.mpasphys.radtran_v842).

* the WOOF-only &radtran group resolves to a woof.radtran Diagnostic, is off by
  default and refused where it cannot run;
* the Registry pools, default streams and stream definitions carry the rt_*
  fields only when it is set (every other row is the pinned Registry's);
* RadtranOutput fills the fields from a session's pools with the same words
  woof.radtran.model.compute gives on the same columns (host build), and the
  interface pressures are the documented interpolation.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys import resolve_namelist
from woof.hex.mpasphys import radtran_v842 as radtran

NZ = 20


def _namelist(waves="500 760.5", **extra):
    lines = ["&physics", " config_physics_suite='mesoscale_reference'", "/"]
    if waves is not None:
        lines += ["&radtran", f" config_radtran_wavelengths={waves!r}"]
        lines += [f" {k}={v!r}" if not isinstance(v, bool) else f" {k}={'.true.' if v else '.false.'}"
                  for k, v in extra.items()]
        lines += ["/"]
    return "\n".join(lines) + "\n"


def test_group_is_off_by_default_and_resolves():
    assert resolve_namelist(_namelist(None), default_backend="hex-cuda").radtran is None
    config = resolve_namelist(_namelist(config_radtran_umu="0.6 -0.8", config_radtran_phi="0 90",
                                        config_radtran_streams=8, config_radtran_pseudospherical=True),
                              default_backend="hex-cuda")
    d = config.radtran
    assert d.wavelengths == (500.0, 760.5) and d.umu == (0.6, -0.8) and d.phi == (0.0, 90.0)
    assert d.streams == 8 and d.pseudospherical
    assert config.receipt()["radtran"]["wavelengths"] == [500.0, 760.5]
    assert "radtran" not in resolve_namelist(_namelist(None), default_backend="hex-cuda").receipt()
    assert radtran.packages(config) == {radtran.PACKAGE}
    assert not any(k.startswith("config_radtran") for k in config.options)


@pytest.mark.parametrize("text,match", [
    (_namelist(None) + "&radtran\n config_radtran_umu='0.5'\n/\n", "need config_radtran_wavelengths"),
    (_namelist("500", config_radtran_colour=1), "has no option"),
    (_namelist("50"), "REPTRAN's solar range"),
    (_namelist("500", config_radtran_solver="twostr", config_radtran_umu="0.5"), "radtran_umu needs"),
    (_namelist("500 abc"), "quoted list of numbers"),
    (_namelist("500", config_radtran_streams=7), "radtran_streams"),
])
def test_group_is_refused_where_it_cannot_run(text, match):
    with pytest.raises(ValueError, match=match):
        resolve_namelist(text, default_backend="hex-cuda")


def test_rrfs_fork_route_refuses_the_group():
    from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist as resolve_rrfs
    text = "&physics\n config_physics_suite='hrrrv5'\n/\n&radtran\n config_radtran_wavelengths='500'\n/\n"
    with pytest.raises(ValueError, match="not available on the RRFS fork route"):
        resolve_rrfs(text)


def test_door_override_lands_in_the_radtran_group():
    from woof.hex.mpasphys.forecast_v842 import native_route
    arguments = SimpleNamespace(physics_namelist=None, physics_suite="mesoscale_reference",
                                physics=["config_radtran_wavelengths=550"])
    config, text = native_route(arguments)
    assert config.radtran is not None and config.radtran.wavelengths == (550.0,)
    assert "&radtran\nconfig_radtran_wavelengths=" in text


def _layout(text):
    from woof.hex.mpasphys.framework_backend_v842 import registry_source
    from woof.hex.mpasphys.pool_v842 import RegistryLayout
    config = resolve_namelist(text, default_backend="hex-cuda")
    dims = dict(nCells=10, nEdges=30, nVertices=20, nVertLevels=NZ, maxEdges=6, maxEdges2=12,
                vertexDegree=3, nSoilLevels=4, nOznLevels=59, nAerLevels=1, nMonths=12)
    return RegistryLayout(registry_source(), config, text, dims)


@pytest.fixture(scope="module")
def layouts():
    return {"off": _layout(_namelist(None)), "irr": _layout(_namelist()),
            "rad": _layout(_namelist(config_radtran_umu="0.6 -0.8", config_radtran_phi="0 90"))}


def test_pools_allocate_the_fields_only_with_the_group(layouts):
    rows = {k: {r["name"]: r for r in layout.fields if r["pool"] == "diag_physics"} for k, layout in layouts.items()}
    assert not set(radtran.FIELDS) & set(rows["off"])
    assert radtran.PACKAGE not in layouts["off"].packages
    assert {radtran.W_DIM, radtran.D_DIM}.isdisjoint(layouts["off"].dimensions)
    assert set(radtran.FIELDS) & set(rows["irr"]) == {*radtran.FLUX_FIELDS, *radtran.WAVELENGTH_FIELDS}
    assert set(radtran.FIELDS) <= set(rows["rad"])
    plain = list(layouts["off"].fields)
    assert [r for r in layouts["rad"].fields if r["name"] not in radtran.FIELDS] == plain
    assert rows["rad"]["rt_edir"]["dimensions"] == [radtran.W_DIM, "nCells"]
    assert rows["rad"]["rt_edir"]["shape"] == [2, 11]
    assert rows["rad"]["rt_rad"]["shape"] == [2, 4, 11]
    assert rows["rad"]["rt_umu"]["shape"] == [4]
    assert layouts["rad"].dimensions[radtran.W_DIM] == 2 and layouts["rad"].dimensions[radtran.D_DIM] == 4


def test_default_streams_name_the_fields_in_diagnostics(layouts, tmp_path):
    from woof.hex.mpasphys.forecast_v842 import read_streams, write_default_streams
    from woof.hex.mpasphys.streams_output_v842 import output_inventory
    from woof.hex.streams_v842 import _all_packages, field_definition
    listed = {}
    for key, layout in layouts.items():
        d = tmp_path / key
        write_default_streams(layout, d / "streams.atmosphere")
        listed[key] = (d / "stream_list.atmosphere.diagnostics").read_text().split()
        streams = read_streams(d / "streams.atmosphere", layout)
        diagnostics = next(s for s in streams if s.name == "diagnostics")
        produced = {r["name"] for r in layout.fields} & set(radtran.FIELDS)
        assert produced <= set(diagnostics.fields)
        assert not produced & set(output_inventory(streams, layout)["diagnostics"])
    assert not set(radtran.FIELDS) & set(listed["off"])
    assert set(listed["rad"]) - set(listed["off"]) == set(radtran.FIELDS)
    row = field_definition("rt_rad")
    assert row["dimensions"] == [radtran.W_DIM, radtran.D_DIM, "nCells", "Time"]
    assert row["attrs"]["units"] == "W m^{-2} nm^{-1} sr^{-1}" and row["gates"] == [radtran.PACKAGE]
    assert radtran.PACKAGE in _all_packages()


def test_interface_pressure_is_the_documented_interpolation():
    z = np.linspace(0.0, 20000.0, NZ + 1)[:, None] * np.ones((1, 3))
    zm = 0.5 * (z[:-1] + z[1:])
    t = 288.0 - 0.0065 * zm
    p = 1e5 * np.exp(-zm / 8000.0)
    pw = radtran.interface_pressure(z, p, t, np.zeros_like(p))
    assert np.allclose(pw[1:-1], 1e5 * np.exp(-z[1:-1] / 8000.0), rtol=1e-12)
    assert np.all(np.diff(pw, axis=0) < 0) and pw[0, 0] > p[0, 0] and pw[-1, 0] < p[-1, 0]


class _Session:
    """A session's pool reads/writes over host arrays (the non-resident route)."""

    def __init__(self, layout, config, arrays, dtypes):
        self.layout, self.config, self.arrays = layout, config, arrays
        self.native = SimpleNamespace(_fields={("diag_physics", k): (v, None, 1) for k, v in dtypes.items()})
        self.written = {}

    def field(self, pool, name, *, level=1):
        if (pool, name) not in self.arrays:
            raise KeyError(name)
        return self.arrays[pool, name]

    def put(self, pool, name, value, *, level=1):
        self.written[name] = np.array(value)


def _columns(ncell=6):
    z = np.linspace(0.0, 22000.0, NZ + 1)[:, None] + np.linspace(0.0, 800.0, ncell)[None, :]
    zm = 0.5 * (z[:-1] + z[1:])
    t = np.maximum(288.0 - 0.0065 * zm, 216.0)
    p = 1.0e5 * np.exp(-zm / 7800.0)
    exner = (p / 1e5) ** (287.04 / 1004.5)
    qv = 0.01 * np.exp(-zm / 2000.0)
    qc = np.zeros_like(qv)
    qc[2:5, 2] = 3e-4
    rho = p / (287.04 * t)
    return dict(zgrid=z, pressure=p, theta=t / exner, exner=exner, qv=qv, qc=qc, qi=np.zeros_like(qv), rho=rho)


def test_output_fills_the_fields_with_the_librarys_words(layouts):
    pytest.importorskip("cupy")
    from woof.radtran import data, model
    try:
        data.data_dir()
    except data.LibradtranDataMissing as exc:
        pytest.skip(str(exc))
    layout = layouts["rad"]
    config = resolve_namelist(_namelist(config_radtran_umu="0.6 -0.8", config_radtran_phi="0 90"),
                              default_backend="hex-cuda")
    c = _columns()
    n = c["pressure"].shape[1]
    nscal = max(layout.indices["state", k] for k in ("index_qv", "index_qc", "index_qi"))
    scalars = np.zeros((nscal, NZ, n))
    for key, name in (("index_qv", "qv"), ("index_qc", "qc"), ("index_qi", "qi")):
        scalars[layout.indices["state", key] - 1] = c[name]
    lat = np.radians(np.linspace(20.0, 50.0, n))
    lon = np.radians(np.linspace(-120.0, -70.0, n))
    arrays = {("mesh", "zgrid"): c["zgrid"], ("diag", "pressure"): c["pressure"],
              ("diag", "theta"): c["theta"], ("diag", "exner"): c["exner"], ("diag", "rho"): c["rho"],
              ("state", "scalars"): scalars, ("mesh", "latCell"): lat, ("mesh", "lonCell"): lon,
              ("diag_physics", "sfc_albedo"): np.full(n, 0.15)}
    session = _Session(layout, config, arrays, {k: np.float64 for k in radtran.FIELDS})
    out = radtran.RadtranOutput(session)
    when = datetime(2025, 6, 21, 18, 0, 0)
    out.compute(when)
    w = session.written
    assert w["rt_edir"].shape == (2, n) and w["rt_rad"].shape == (2, 4, n)
    assert np.array_equal(w["rt_wavelength"], [500.0, 760.5])
    assert np.array_equal(w["rt_umu"], [0.6, 0.6, -0.8, -0.8]) and np.array_equal(w["rt_phi"], [0, 90, 0, 90])
    t = c["theta"] * c["exner"]
    ref = model.compute(config.radtran, z_w_m=c["zgrid"].T,
                        p_w_pa=radtran.interface_pressure(c["zgrid"], c["pressure"], t, c["qv"]).T,
                        t_mass=t.T, qv_mass=c["qv"].T, umu0=radtran.solar_cosine(when, np.degrees(lat), np.degrees(lon)),
                        albedo=np.full(n, 0.15), qc_mass=c["qc"].T, qi_mass=c["qi"].T, rho_dry=c["rho"].T,
                        day_of_year=172, backend="host")
    for name in radtran.FLUX_FIELDS:
        assert np.array_equal(w[name], ref[name[3:]]), name
    assert np.array_equal(w["rt_rad"], ref["rad"])
    assert np.all(np.delete(w["rt_edir"][0], 2) > 0.1)       # clear columns, all in daylight
    assert w["rt_edir"][0, 2] < 0.5 * w["rt_edir"][0, 1]      # the cloudy column
