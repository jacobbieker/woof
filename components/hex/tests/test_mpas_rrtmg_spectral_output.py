"""rrtmg_spectral_output on the MPAS suite route (woof.hex.mpasphys.radiation_spectral_v842).

The MPAS RRTMG slots (shortwave_slot, radiation_lw_slot) run on a few synthetic
columns, day and night, with &radiation config_rrtmg_spectral_output off, 'bands'
and 'gpoints':

* every word the slots wrote before the option existed is the same with it on;
* the [band, cell] fields have their Registry shapes, keep the halo column's
  words, add up to the broadband boundary fluxes, and are zero in night
  shortwave columns;
* the option is refused where no GPU RRTMG slot would produce it, and the
  Registry pools, default streams and stream writer carry the fields only
  when it is on.

The band words themselves are the engine kernels', held to compiled WRF by
tests/test_rrtmg_spectral_output.py.  Tables: the packaged RRTMG data
(woof.core.rrtmg_sw.load_sw_tables, woof.ingest.rrtmg_coeffs), the same
files MPAS-A reads.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys import resolve_namelist
from woof.hex.mpasphys import radiation_spectral_v842 as spectral

NZ, NC = 24, 8
HALO = np.float32(-987.25)
#: Longitudes across the globe at 12 UTC: about half the columns are in daylight.
LON_DEG = np.linspace(-170.0, 170.0, NC)
LAT_DEG = np.linspace(-50.0, 60.0, NC)


def _namelist(mode="bands", **physics):
    lines = ["&physics", " config_physics_suite='mesoscale_reference'"]
    lines += [f" {k}={v!r}" for k, v in physics.items()]
    lines += ["/"]
    if mode is not None:
        lines += ["&radiation", f" config_rrtmg_spectral_output={mode!r}", "/"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# namelist option
# ---------------------------------------------------------------------------
def test_option_defaults_to_none_and_resolves_on_the_gpu_route():
    assert resolve_namelist(_namelist(None), default_backend="hex-cuda").spectral_output == "none"
    for mode in ("none", "bands", "gpoints"):
        config = resolve_namelist(_namelist(mode), default_backend="hex-cuda")
        assert config.spectral_output == mode
        assert config.receipt().get("rrtmg_spectral_output") == (None if mode == "none" else mode)
        # Not a Registry config: it never joins the options pushed to the native config pool.
        assert spectral.OPTION not in config.options
    assert spectral.packages(resolve_namelist(_namelist("gpoints"), default_backend="hex-cuda")) == {
        "rrtmg_spectral_sw_gpoints", "rrtmg_spectral_lw_bands"}


@pytest.mark.parametrize("text,match", [
    (_namelist("spectral"), "must be 'none', 'bands' or 'gpoints'"),
    (_namelist(None) + "&radiation\n config_rrtmg_bands=1\n/\n", "has no option"),
    (_namelist("bands", config_radt_lw_scheme="off", config_radt_sw_scheme="off"),
     "needs RRTMG radiation"),
    (_namelist("bands", config_radt_lw_scheme="cam_lw", config_radt_sw_scheme="cam_sw"),
     "needs RRTMG radiation"),
    (_namelist("bands") + "&hex_mpasphys\n config_radt_sw_backend='native-cpu'\n/\n",
     "needs radiation_sw on hex-cuda"),
])
def test_option_is_refused_where_nothing_produces_it(text, match):
    with pytest.raises(ValueError, match=match):
        resolve_namelist(text, default_backend="hex-cuda")


def test_rrfs_fork_route_refuses_the_option():
    from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist as resolve_rrfs
    text = "&physics\n config_physics_suite='hrrrv5'\n/\n&radiation\n config_rrtmg_spectral_output='bands'\n/\n"
    with pytest.raises(ValueError, match="not available on the RRFS fork route"):
        resolve_rrfs(text)


def test_door_override_lands_in_the_radiation_group():
    from woof.hex.mpasphys.forecast_v842 import native_route
    arguments = SimpleNamespace(physics_namelist=None, physics_suite="mesoscale_reference",
                                physics=["config_rrtmg_spectral_output=gpoints"])
    config, text = native_route(arguments)
    assert config.spectral_output == "gpoints"
    assert "&radiation\nconfig_rrtmg_spectral_output='gpoints'" in text


def test_one_rrtmg_side_carries_only_its_own_fields():
    config = resolve_namelist(_namelist("bands", config_radt_sw_scheme="off"), default_backend="hex-cuda")
    assert spectral.sides(config) == ("lw",)
    assert spectral.packages(config) == {"rrtmg_spectral_lw_bands"}


# ---------------------------------------------------------------------------
# Registry pools, default streams and the stream writer's definitions
# ---------------------------------------------------------------------------
def _layout(mode):
    from woof.hex.mpasphys.framework_backend_v842 import registry_source
    from woof.hex.mpasphys.pool_v842 import RegistryLayout
    text = _namelist(mode)
    config = resolve_namelist(text, default_backend="hex-cuda")
    dims = dict(nCells=10, nEdges=30, nVertices=20, nVertLevels=NZ, maxEdges=6, maxEdges2=12,
                vertexDegree=3, nSoilLevels=4, nOznLevels=59, nAerLevels=1, nMonths=12)
    return RegistryLayout(registry_source(), config, text, dims)


@pytest.fixture(scope="module")
def layouts():
    return {mode: _layout(mode) for mode in ("none", "bands", "gpoints")}


def test_pools_allocate_the_fields_only_with_the_option(layouts):
    rows = {mode: {r["name"]: r for r in layout.fields if r["pool"] == "diag_physics"}
            for mode, layout in layouts.items()}
    names = {mode: set(spectral.FIELDS) & set(r) for mode, r in rows.items()}
    assert names["none"] == set()
    assert not {"nSWBands", "nSWGpoints", "nLWBands"} & set(layouts["none"].dimensions)
    assert not set(spectral.PACKAGES.values()) & layouts["none"].packages
    assert names["bands"] == {n for n in spectral.FIELDS if not n.endswith("_gpt")}
    assert names["gpoints"] == {n for n in spectral.FIELDS if n.endswith("_gpt") or n.startswith("lw")}
    # Everything else is the pinned Registry's, row for row.
    plain = [r for r in layouts["none"].fields]
    assert [r for r in layouts["bands"].fields if r["name"] not in spectral.FIELDS] == plain
    for mode, count, dim in (("bands", 14, "nSWBands"), ("gpoints", 112, "nSWGpoints")):
        row = rows[mode]["swdnb" + ("_band" if mode == "bands" else "_gpt")]
        assert row["dimensions"] == [dim, "nCells"] and row["shape"] == [count, 11]
        assert layouts[mode].dimensions[dim] == count
        lw = rows[mode]["lwupt_band"]
        assert lw["dimensions"] == ["nLWBands", "nCells"] and lw["shape"] == [16, 11]


def test_default_streams_name_the_fields_in_diagnostics(layouts, tmp_path):
    from woof.hex.mpasphys.forecast_v842 import read_streams, write_default_streams
    from woof.hex.mpasphys.streams_output_v842 import output_inventory
    listed = {}
    for mode, layout in layouts.items():
        directory = tmp_path / mode
        write_default_streams(layout, directory / "streams.atmosphere")
        listed[mode] = (directory / "stream_list.atmosphere.diagnostics").read_text().split()
        streams = read_streams(directory / "streams.atmosphere", layout)
        diagnostics = next(s for s in streams if s.name == "diagnostics")
        produced = {r["name"] for r in layout.fields} & set(spectral.FIELDS)
        assert produced <= set(diagnostics.fields)
        # diag_physics words have a computed authority: none is reported missing.
        assert not produced & set(output_inventory(streams, layout)["diagnostics"])
    assert not set(spectral.FIELDS) & set(listed["none"])
    assert listed["bands"][:len(listed["none"])] == listed["none"]
    assert set(listed["bands"]) - set(listed["none"]) == {n for n in spectral.FIELDS if not n.endswith("_gpt")}
    assert set(listed["gpoints"]) - set(listed["none"]) == {
        n for n in spectral.FIELDS if n.endswith("_gpt") or n.startswith("lw")}


def test_stream_definitions_carry_the_band_edges():
    from woof.core.rrtmg_sw import band_wavenumbers, gpoint_bands
    from woof.hex.streams_v842 import Stream, field_definition, _all_packages
    from woof.io.wrf_output_schema import RRTMG_LW_WAVENUMBERS
    sw = field_definition("swdnb_band")
    assert sw["dimensions"] == ["nSWBands", "nCells", "Time"] and sw["gates"] == ["rrtmg_spectral_sw_bands"]
    assert sw["attrs"]["units"] == "W m^{-2}" and sw["attrs"]["wavenumber_units"] == "cm-1"
    assert sw["attrs"]["wavenumber_low"]["values"] == list(band_wavenumbers()[0])
    assert sw["attrs"]["wavenumber_high"]["values"] == list(band_wavenumbers()[1])
    gpt = field_definition("swupt_gpt")
    assert gpt["dimensions"][0] == "nSWGpoints"
    assert gpt["attrs"]["band_of_gpoint"]["values"] == list(gpoint_bands())
    lw = field_definition("lwdnb_band")
    assert lw["attrs"]["wavenumber_low"]["values"] == [a for a, _ in RRTMG_LW_WAVENUMBERS]
    assert set(spectral.PACKAGES.values()) <= _all_packages()
    # A deck naming them writes them only in a run whose package is on.
    stream = Stream("spectral", {"type": "output"}, (("var", "swdnb_band", ""), ("var", "lwupt_band", "")))
    assert stream.fields(()) == ()
    assert stream.fields({"rrtmg_spectral_lw_bands"}) == ("lwupt_band",)


def test_history_writer_names_the_axes_and_band_edges(monkeypatch, tmp_path):
    """write_stream's request to the Rust NetCDF writer: the spectral axis, its size and the
    band-edge attributes (the bridge itself is exercised wherever it is installed)."""
    from datetime import datetime
    from woof.hex import restart_v842
    from woof.hex.output import HistoryField
    from woof.hex.streams_v842 import Stream
    seen = {}

    def bridge(request, directory):
        seen.update(request)
        return {"schema": "mpas-native-stream/v1", "ordered_metadata": True}
    monkeypatch.setattr(restart_v842, "_call", bridge)
    stream = Stream("spectral", {"type": "output", "filename_template": "s.nc", "output_interval": "1:00:00"},
                    (("var", "xtime", ""), ("var", "swdnb_gpt", ""), ("var", "lwupt_band", "")))
    rng = np.random.default_rng(0)
    fields = {"swdnb_gpt": HistoryField(rng.random((112, 5), dtype=np.float32), ("nSWGpoints", "nCells")),
              "lwupt_band": HistoryField(rng.random((16, 5), dtype=np.float32), ("nLWBands", "nCells"))}
    mesh = SimpleNamespace(dimensions={"nCells": 5})
    when = datetime(2026, 6, 21, 12)
    try:
        restart_v842.write_stream(tmp_path / "s.nc", stream, mesh, fields, when, initial_time=when,
                                  active_packages={"rrtmg_spectral_sw_gpoints", "rrtmg_spectral_lw_bands"})
    except (FileNotFoundError, OSError):
        pass  # the staged file is never written by the stand-in bridge
    variables = {v["name"]: v for v in seen["variables"]}
    assert seen["dimensions"]["nSWGpoints"] == 112 and seen["dimensions"]["nLWBands"] == 16
    assert variables["swdnb_gpt"]["dimensions"] == ("Time", "nCells", "nSWGpoints")
    assert variables["lwupt_band"]["dimensions"] == ("Time", "nCells", "nLWBands")
    attrs = variables["lwupt_band"]["attrs"]
    assert attrs["wavenumber_low"] == {"dtype": "f4", "values": [a for a, _ in __import__(
        "woof.io.wrf_output_schema", fromlist=["x"]).RRTMG_LW_WAVENUMBERS]}
    assert variables["swdnb_gpt"]["attrs"]["band_of_gpoint"]["dtype"] == "i4"


def test_history_file_carries_the_axes_and_band_edges(tmp_path):
    """The same write through the real Rust NetCDF writer, where rw_mpas_stream is installed."""
    from datetime import datetime
    from woof.core.rrtmg_sw import band_wavenumbers, gpoint_bands
    from woof.hex import restart_v842
    from woof.hex.engines import EngineRefusal
    from woof.hex.output import HistoryField
    from woof.hex.streams_v842 import Stream
    try:
        restart_v842._bridge()
    except EngineRefusal:
        pytest.skip("rw_mpas_stream is not installed here")
    netCDF4 = pytest.importorskip("netCDF4")
    stream = Stream("spectral", {"type": "output", "filename_template": "s.nc", "output_interval": "1:00:00",
                                 "io_type": "netcdf4"},
                    (("var", "xtime", ""), ("var", "swdnb_gpt", ""), ("var", "swupt_band", ""),
                     ("var", "lwupt_band", "")))
    rng = np.random.default_rng(0)
    values = {"swdnb_gpt": (rng.random((112, 5), dtype=np.float32), "nSWGpoints"),
              "swupt_band": (rng.random((14, 5), dtype=np.float32), "nSWBands"),
              "lwupt_band": (rng.random((16, 5), dtype=np.float32), "nLWBands")}
    fields = {name: HistoryField(value, (dim, "nCells")) for name, (value, dim) in values.items()}
    when = datetime(2026, 6, 21, 12)
    path = restart_v842.write_stream(
        tmp_path / "s.nc", stream, SimpleNamespace(dimensions={"nCells": 5}), fields, when, initial_time=when,
        active_packages=set(spectral.PACKAGES.values()), scratch=tmp_path / "scratch")
    with netCDF4.Dataset(path) as ds:
        for name, (value, dim) in values.items():
            var = ds.variables[name]
            assert var.dimensions == ("Time", "nCells", dim)
            assert np.array_equal(var[0].T.view(np.uint32), value.view(np.uint32))
            assert var.units == "W m^{-2}" and var.wavenumber_units == "cm-1"
        assert ds.dimensions["nSWGpoints"].size == 112 and ds.dimensions["nLWBands"].size == 16
        np.testing.assert_array_equal(ds.variables["swupt_band"].wavenumber_low,
                                      np.asarray(band_wavenumbers()[0], np.float32))
        np.testing.assert_array_equal(ds.variables["swdnb_gpt"].band_of_gpoint, gpoint_bands())
        assert ds.variables["lwupt_band"].wavenumber_high[-1] == np.float32(3250.0)


# ---------------------------------------------------------------------------
# the slots on the device
# ---------------------------------------------------------------------------
def _sounding(cp):
    """Level-major (level, cell) MPAS radiation inputs with one halo column."""
    rng = np.random.default_rng(7)
    z = np.linspace(0.0, 20000.0, NZ + 1)
    p8w = 101000.0 * np.exp(-z / 7800.0)
    zm = 0.5 * (z[1:] + z[:-1])
    p = 101000.0 * np.exp(-zm / 7800.0)
    t8w = np.maximum(300.0 - 6.5e-3 * z, 210.0)
    t = np.maximum(300.0 - 6.5e-3 * zm, 210.0)
    qv = 0.014 * np.exp(-zm / 2500.0)
    def cols(profile, jitter=0.0):
        out = np.repeat(np.asarray(profile, np.float64)[:, None], NC + 1, axis=1)
        out *= 1.0 + jitter * rng.standard_normal(out.shape)
        out[:, NC] = HALO
        return cp.asarray(out.astype(np.float32))
    cloud = np.zeros(NZ)
    cloud[4:8], cloud[14:16] = 0.8, 0.5
    qc = np.where(cloud > 0, 2e-4, 0.0)
    qi = np.where(np.arange(NZ) >= 14, 1e-5, 0.0) * (cloud > 0)
    return dict(p=cols(p), p8w=cols(p8w), t=cols(t, 0.002), t8w=cols(t8w, 0.002),
                dz=cols(np.diff(z)), pi=cols((p / 1.0e5) ** (287.0 / 1004.5)),
                qv=cols(qv, 0.05), qc=cols(qc), qi=cols(qi), qs=cols(np.zeros(NZ)),
                cldfrac=cols(cloud), zeros=cols(np.zeros(NZ)))


def _surface(cp, value):
    out = np.full(NC + 1, value, np.float32)
    out[NC] = HALO
    return cp.asarray(out)


def _rows(cp, values):
    out = np.append(np.asarray(values, np.float32), HALO)
    return cp.asarray(out)


def _sw_call(cp, engine, mode):
    from woof.hex.mpasphys.rrtmg_sw_v842 import SCALAR_OUTPUTS
    s = _sounding(cp)
    configs = dict(config_radt_cld_overlap="maximum_random", config_radt_cld_dcorrlen="constant",
                   config_o3climatology=False)
    controls = engine.driver_controls(configs, xtime_s=0.0, dt_radtsw=1800.0, curr_julday=172.0,
                                      degrad=np.float32(np.pi / 180.0), dpd=np.float32(360.0 / 365.0))
    controls.update(gmt=np.float32(12.0), julday=172, icloud=1, has_reqc=0, has_reqi=0, has_reqs=0)
    sounding = dict(p3d=s["p"], t3d=s["t"], dz8w=s["dz"], pi3d=s["pi"], qv3d=s["qv"], qc3d=s["qc"],
                    qi3d=s["qi"], qs3d=s["qs"], cldfra3d=s["cldfrac"], re_cloud=s["zeros"],
                    re_ice=s["zeros"], re_snow=s["zeros"], p8w=s["p8w"], t8w=s["t8w"], controls=controls)
    diag = {n: _surface(cp, HALO) for n in SCALAR_OUTPUTS}
    o3 = cp.full((3, NC + 1), 1.0e-7, dtype=cp.float32)
    diag.update(sfc_albedo=_surface(cp, 0.18), o3clim=o3)
    widths = {"_band": 14, "_gpt": 112}
    for name in spectral.slot_fields("sw", mode):
        diag[name] = cp.full((widths["_" + name.rsplit("_", 1)[1]], NC + 1), HALO, dtype=cp.float32)
    pools = dict(nCellsSolve=NC, rrtmg_sw_engine=engine, radiation_sw_input=sounding,
                 mesh=dict(latCell=_rows(cp, np.radians(LAT_DEG)), lonCell=_rows(cp, np.radians(LON_DEG))),
                 sfc_input=dict(skintemp=_surface(cp, 295.0), snow=_surface(cp, 0.0), xice=_surface(cp, 0.0),
                                xland=_surface(cp, 1.0)),
                 diag_physics=diag, tend_physics=dict(rthratensw=cp.full((NZ, NC + 1), HALO, dtype=cp.float32)),
                 atm_input=dict(pin=cp.asarray(np.array([10.0, 500.0, 1000.0], np.float32))))
    options = dict(config_microp_scheme="mp_thompson", config_microp_re=False)
    if mode != "none":
        options[spectral.OPTION] = mode
    return SimpleNamespace(pools=pools, options=options)


def _lw_call(cp, tables, mode):
    s = _sounding(cp)
    prepared = dict(pres_hyd=s["p"], pres2_hyd=s["p8w"], t=s["t"], t2=s["t8w"], dz=s["dz"], pi=s["pi"],
                    xlat=_rows(cp, LAT_DEG))
    radshared = {name: cp.ascontiguousarray(s[key].T) for name, key in
                 (("qvrad", "qv"), ("qcrad", "qc"), ("qirad", "qi"), ("qsrad", "qs"))}
    from woof.hex.mpasphys.rrtmg_lw_v842 import FLUX_FIELDS
    diag = {("olrtoa" if n == "olr" else n): _surface(cp, HALO) for n in FLUX_FIELDS}
    diag.update(cldfrac=s["cldfrac"], re_cloud=s["zeros"], re_ice=s["zeros"], re_snow=s["zeros"],
                sfc_emiss=_surface(cp, 0.98), o3clim=cp.full((3, NC + 1), 1.0e-7, dtype=cp.float32),
                o3vmr=cp.full((NZ, NC + 1), HALO, dtype=cp.float32),
                **{n: cp.full((NZ, NC + 1), HALO, dtype=cp.float32) for n in ("rre_cloud", "rre_ice", "rre_snow")})
    for name in spectral.slot_fields("lw", mode):
        diag[name] = cp.full((16, NC + 1), HALO, dtype=cp.float32)
    pools = {"nCellsSolve": NC, "prepared": prepared, "radshared": radshared, "diag_physics": diag,
             "sfc_input": dict(skintemp=_surface(cp, 295.0), xland=_surface(cp, 1.0), xice=_surface(cp, 0.0),
                               snow=_surface(cp, 0.0)),
             "atm_input": dict(pin=cp.asarray(np.array([10.0, 500.0, 1000.0], np.float32))),
             "tend_physics": dict(rthratenlw=cp.full((NZ, NC + 1), HALO, dtype=cp.float32)),
             "rrtmg_lw_tables": tables, "retain_lw_profiles": False, "icloud": 1, "julday": 172,
             "radius_flags": (0, 0, 0)}
    options = dict(config_radt_cld_overlap="maximum_random", config_radt_cld_dcorrlen="constant",
                   config_o3climatology=False, config_microp_re=False, config_microp_scheme="mp_thompson",
                   config_radt_lw_scheme="rrtmg_lw", config_radt_sw_scheme="rrtmg_sw")
    if mode != "none":
        options[spectral.OPTION] = mode
    return SimpleNamespace(pools=pools, options=options)


def _lw_tables(cp):
    """The slot's table mapping: MPAS's own device initialization (initialize_device) from the
    packaged RRTMG_LW_DATA records.  Under MLX that initializer's device-constant table views
    (module globals) are not exposed, so the same solver mapping is taken from the WRF engine's
    upload of the same data (rrtmg_lw._lw_dev_consts, the keys solve_device reads); the claims
    here (option on/off identity, bands summing to broadband) do not depend on which."""
    import woof.backend
    from woof.hex.mpasphys.rrtmg_lw_v842 import initialize_device
    from woof.ingest.rrtmg_coeffs import load_rrtmg_lw_coefficients
    if woof.backend.active() == "mlx":
        from woof.core import rrtmg_lw
        from woof.core.rrtmg_legacy import _lw_coeffs
        return dict(rrtmg_lw._lw_dev_consts(cp, _lw_coeffs()), ready=cp.cuda.Event(disable_timing=True))
    loaded = load_rrtmg_lw_coefficients()
    raw = {module: {name: cp.asarray(np.ravel(np.asarray(value, np.float32), order="F"))
                    for name, value in arrays.items()}
           for module, arrays in loaded.items() if module.startswith("rrlw_kg")}
    return initialize_device(raw)


def _host(pools):
    return {f"{pool}/{name}": value.get() for pool in ("diag_physics", "tend_physics")
            for name, value in pools[pool].items()}


@pytest.fixture(scope="module")
def runs():
    cp = pytest.importorskip("cupy")
    from woof.core.rrtmg_sw import load_sw_tables
    from woof.hex.mpasphys.rrtmg_lw_v842 import radiation_lw_slot
    from woof.hex.mpasphys.rrtmg_sw_v842 import MPASShortwave, shortwave_slot
    engine = MPASShortwave(tables=load_sw_tables())
    tables = _lw_tables(cp)
    out = {}
    for mode in ("none", "bands", "gpoints"):
        sw = _sw_call(cp, engine, mode)
        shortwave_slot(sw)
        lw = _lw_call(cp, tables, mode)
        radiation_lw_slot(lw)
        out[mode] = dict(sw=_host(sw.pools), lw=_host(lw.pools))
    return out


def test_every_existing_word_is_unchanged(runs):
    for side in ("sw", "lw"):
        base = runs["none"][side]
        for mode in ("bands", "gpoints"):
            other = runs[mode][side]
            assert set(base) <= set(other)
            for key, value in base.items():
                assert np.array_equal(value.view(np.uint32), other[key].view(np.uint32)), (mode, key)


def test_some_columns_are_day_and_some_night(runs):
    coszr = runs["none"]["sw"]["diag_physics/coszr"][:NC]
    assert (coszr > 0).sum() >= 2 and (coszr <= 0).sum() >= 2


def test_fields_have_their_shapes_and_keep_the_halo(runs):
    for mode, width in (("bands", 14), ("gpoints", 112)):
        for side, count in (("sw", width), ("lw", 16)):
            for name in spectral.slot_fields(side, mode):
                value = runs[mode][side][f"diag_physics/{name}"]
                assert value.shape == (count, NC + 1), name
                assert (value[:, NC] == HALO).all(), name
                assert np.isfinite(value[:, :NC]).all() and (value[:, :NC] >= 0).all(), name


def test_night_shortwave_columns_are_zero(runs):
    night = runs["none"]["sw"]["diag_physics/coszr"][:NC] <= 0
    for mode in ("bands", "gpoints"):
        for name in spectral.slot_fields("sw", mode):
            value = runs[mode]["sw"][f"diag_physics/{name}"][:, :NC]
            assert (value[:, night] == 0).all(), name
            assert (value[:, ~night].sum(axis=0) > 0).any() or name.startswith("swup"), name


def test_bands_add_up_to_the_broadband_boundary_fluxes(runs):
    for mode in ("bands", "gpoints"):
        for side in ("sw", "lw"):
            got = runs[mode][side]
            for name in spectral.slot_fields(side, mode):
                base = name.rsplit("_", 1)[0]
                if f"diag_physics/{base}" not in got:
                    continue  # swddirc: no MPAS broadband field
                total = got[f"diag_physics/{name}"][:, :NC].astype(np.float64).sum(axis=0)
                np.testing.assert_allclose(total, got[f"diag_physics/{base}"][:NC], rtol=2e-5, atol=1e-3,
                                           err_msg=f"{mode} {name}")


def test_gpoints_sum_to_their_bands(runs):
    from woof.core.rrtmg_sw import gpoint_bands
    owner = np.asarray(gpoint_bands())
    for base in spectral.slot_fields("sw", "bands"):
        bands = runs["bands"]["sw"][f"diag_physics/{base}"][:, :NC].astype(np.float64)
        gpts = runs["gpoints"]["sw"][f"diag_physics/{base[:-5]}_gpt"][:, :NC].astype(np.float64)
        summed = np.stack([gpts[owner == b].sum(axis=0) for b in range(14)])
        np.testing.assert_allclose(summed, bands, rtol=2e-5, atol=1e-3, err_msg=base)
