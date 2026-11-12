"""WRF's p_lev / z_lev diagnostics on the MPAS suite route (woof.hex.mpasdiag.plevel_diag_v842).

&diagnostics config_p_lev_diags / config_z_lev_diags with the levels in config_press_levels / config_z_levels write
WRF's ``pld`` / ``zld`` column interpolation as ``diag`` fields [nPLevels|nZLevels, nCells].  Here:

* the options resolve (off by default), refuse what they cannot honour by name and are refused on the RRFS fork
  route; the Registry and the default streams carry the dimensions and fields only with them, leaving every other
  row untouched (the off state keeps its registry and stream rows);
* per mesh column the interpolation equals WRF v4.7.1's byte-unmodified module_diag_pld.F and module_diag_zld.F,
  compiled and run on the committed synthetic columns (tools/plevel_wrf471_oracle, data in tests/plevel_oracle),
  for both extrap_below_grnd values, with the carried search state, the missing values, the above-ground
  levels and the extrapolation below ground all in the graded set;
* the route's module publishes exactly the fields a due stream carries, from the pools' words.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasdiag import plevel_diag_v842 as pl
from woof.hex.mpasphys import resolve_namelist

ORACLE = Path(__file__).resolve().parent / "plevel_oracle"
F32 = np.float32


def _namelist(**diagnostics):
    lines = ["&physics", " config_physics_suite='convection_permitting'", "/"]
    if diagnostics:
        lines.append("&diagnostics")
        lines += [f" {key}={value}" for key, value in diagnostics.items()]
        lines.append("/")
    return "\n".join(lines) + "\n"


ON = dict(config_p_lev_diags=".true.", config_press_levels="'85000,50000,20000'",
          config_z_lev_diags=".true.", config_z_levels="'-10,500,3000'")


def _resolve(**diagnostics):
    return resolve_namelist(_namelist(**diagnostics), default_backend="hex-cuda")


def test_options_default_off_and_resolve():
    off = _resolve()
    assert off.plevel_diagnostics is None and "plevel_diagnostics" not in off.receipt()
    assert pl.packages(off) == set()
    config = _resolve(**ON)
    levels = config.plevel_diagnostics
    assert levels.press_levels == (85000.0, 50000.0, 20000.0) and levels.z_levels == (-10.0, 500.0, 3000.0)
    assert (levels.extrap_below_grnd, levels.p_missing, levels.z_missing) == (1, -999.0, -999.0)
    assert pl.packages(config) == {pl.P_PACKAGE, pl.Z_PACKAGE}
    assert config.receipt()["plevel_diagnostics"]["press_levels"] == [85000.0, 50000.0, 20000.0]
    only_p = _resolve(config_p_lev_diags=".true.", config_press_levels="'70000'", config_extrap_below_grnd=2,
                      config_p_lev_missing=-9999.)
    assert pl.packages(only_p) == {pl.P_PACKAGE} and only_p.plevel_diagnostics.z_levels == ()
    assert only_p.plevel_diagnostics.extrap_below_grnd == 2 and only_p.plevel_diagnostics.p_missing == -9999.0
    # the group's other WOOF options still live beside them, and an unknown one is still refused
    both = _resolve(config_nwp_diagnostics=".true.", **ON)
    assert both.nwp_diagnostics and both.plevel_diagnostics is not None
    with pytest.raises(ValueError, match="has no option"):
        _resolve(config_p_level_diags=".true.")


@pytest.mark.parametrize("diagnostics,match", [
    (dict(config_p_lev_diags=".true."), "needs config_press_levels"),
    (dict(config_z_lev_diags=".true."), "needs config_z_levels"),
    (dict(config_p_lev_diags=".true.", config_press_levels="'100000,-5'"), "must be positive"),
    (dict(config_p_lev_diags=".true.", config_press_levels="'abc'"), "not a list of numbers"),
    (dict(config_p_lev_diags=".true.", config_press_levels="'85000'", config_extrap_below_grnd=3),
     "config_extrap_below_grnd is 1"),
    (dict(config_p_lev_diags=".true.", config_press_levels="'85000'", config_use_tot_or_hyd_p=2),
     "hydrostatic pressure"),
    (dict(config_press_levels="'85000'"), "read by nothing"),
    (dict(config_z_levels="'100'"), "read by nothing"),
    (dict(config_z_lev_missing=-5.), "read by nothing"),
])
def test_refusals_are_named(diagnostics, match):
    with pytest.raises(ValueError, match=match):
        _resolve(**diagnostics)


def test_rrfs_fork_route_refuses_the_options():
    from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist as resolve_rrfs
    text = ("&physics\n config_physics_suite='hrrrv5'\n/\n&diagnostics\n config_p_lev_diags=.true.\n"
            " config_press_levels='85000'\n/\n")
    with pytest.raises(ValueError, match="not available on the RRFS fork route"):
        resolve_rrfs(text)


def _layout(**diagnostics):
    from woof.hex.mpasphys.framework_backend_v842 import registry_source
    from woof.hex.mpasphys.pool_v842 import RegistryLayout
    text = _namelist(**diagnostics)
    config = resolve_namelist(text, default_backend="hex-cuda")
    dims = dict(nCells=10, nEdges=30, nVertices=20, nVertLevels=12, maxEdges=6, maxEdges2=12,
                vertexDegree=3, nSoilLevels=4, nOznLevels=59, nAerLevels=1, nMonths=12)
    return RegistryLayout(registry_source(), config, text, dims)


def test_pools_and_streams_only_with_the_options(tmp_path):
    from woof.hex.mpasphys.forecast_v842 import read_streams, write_default_streams
    from woof.hex.streams_v842 import field_definition, _all_packages
    off, on = _layout(), _layout(**ON)
    rows = lambda layout: {r["name"]: r for r in layout.fields if r["pool"] == "diag"}
    assert not set(pl.NAMES) & set(rows(off)) and "nPLevels" not in off.dimensions
    assert set(pl.NAMES) <= set(rows(on))
    assert (on.dimensions["nPLevels"], on.dimensions["nZLevels"]) == (3, 3)
    assert [r for r in on.fields if r["name"] not in pl.NAMES] == list(off.fields)
    assert pl.P_PACKAGE in on.packages and pl.Z_PACKAGE in on.packages and not {pl.P_PACKAGE} & off.packages
    assert rows(on)["t_pl"]["dimensions"] == ["nPLevels", "nCells"] and rows(on)["t_pl"]["shape"] == [3, 11]
    assert rows(on)["p_zl"]["dimensions"] == ["nZLevels", "nCells"] and rows(on)["z_zl"]["shape"] == [3]
    # one diagnostic alone adds only its own fields and its own dimension
    only_z = _layout(config_z_lev_diags=".true.", config_z_levels="'100,200'")
    assert "nPLevels" not in only_z.dimensions and only_z.dimensions["nZLevels"] == 2
    assert {n for n in pl.NAMES if n in rows(only_z)} == set(pl.Z_FIELDS_2D) | {"z_zl"}
    streams = {}
    for label, layout in (("off", off), ("on", on)):
        write_default_streams(layout, tmp_path / f"streams.{label}")
        streams[label] = {s.name: s for s in read_streams(tmp_path / f"streams.{label}", layout)}
    assert set(pl.NAMES) <= set(streams["on"]["diagnostics"].fields)
    assert not set(pl.NAMES) & set(streams["off"]["diagnostics"].fields)
    assert {n: s.fields for n, s in streams["on"].items() if n != "diagnostics"} == \
           {n: s.fields for n, s in streams["off"].items() if n != "diagnostics"}
    assert [f for f in streams["on"]["diagnostics"].fields if f not in pl.NAMES] == \
           list(streams["off"]["diagnostics"].fields)
    d = field_definition("t_pl")
    assert d["attrs"]["units"] == "K" and d["attrs"]["wrf_name"] == "T_PL" and d["dimensions"] == ["nPLevels", "nCells", "Time"]
    assert {pl.P_PACKAGE, pl.Z_PACKAGE} <= _all_packages()


# ---------------------------------------------------------------------------
# E1: WRF v4.7.1's pld / zld, byte unmodified, on the committed columns
# ---------------------------------------------------------------------------
def _load(extrap):
    with (ORACLE / f"columns_extrap{extrap}.bin").open("rb") as f:
        nc, nz, e, npl, nzl = (int(x) for x in np.fromfile(f, "<i4", 5))
        plev = np.fromfile(f, "<f4", npl)
        zlev = np.fromfile(f, "<f4", nzl)
        mass = [np.fromfile(f, "<f4", nz * nc).reshape(nz, nc) for _ in range(6)]
        zgrid = np.fromfile(f, "<f4", (nz + 1) * nc).reshape(nz + 1, nc)
        pw = np.fromfile(f, "<f4", (nz + 1) * nc).reshape(nz + 1, nc)
    out = np.fromfile(ORACLE / f"oracle_extrap{extrap}.bin", "<f4")
    wrf_p = out[:8 * npl * nc].reshape(8, npl, nc)
    wrf_z = out[8 * npl * nc:].reshape(9, nzl, nc)
    return dict(nc=nc, nz=nz, extrap=e, plev=plev, zlev=zlev, mass=mass, zgrid=zgrid, pw=pw, wrf_p=wrf_p, wrf_z=wrf_z)


def _common(case):
    u, v, theta, qv, pp, pb = case["mass"]
    zgrid = case["zgrid"]
    return dict(u=u, v=v, t=theta - F32(300), qv=qv, zp=np.zeros_like(zgrid), zb=zgrid * F32(9.81), pp=pp, pb=pb)


def _agree(ours, theirs, label):
    """Same missing mask, and the same words to the libm's few ulps.

    The oracle was recorded with macOS's libm (log, exp, powf); numpy on another machine rounds them differently
    in the last place (Linux: up to 5e-6 relative in the humidity ratios, which divide by a difference of the
    saturation pressures).  The below-ground height ``ght`` subtracts nearly equal terms, so it gets 2e-4; every
    word that is a pure blend (u, v, s, and the interface search) matches exactly on both machines.
    """
    gone = theirs == F32(-999)
    assert np.array_equal(ours == F32(-999), gone), f"{label}: the missing mask differs"
    rtol = 2e-4 if label.startswith("ght") else 1e-5
    np.testing.assert_allclose(ours[~gone], theirs[~gone], rtol=rtol, atol=0, err_msg=label)
    return int((~gone).sum())


@pytest.mark.parametrize("extrap", [1, 2])
def test_pressure_levels_equal_wrf_pld(extrap):
    case = _load(extrap)
    assert case["extrap"] == extrap
    result = pl.pressure_levels(np, pw=case["pw"], levels=tuple(case["plev"]), extrap_below_grnd=extrap,
                                **_common(case))
    graded = 0
    for index, name in enumerate(("u", "v", "t", "rh", "ght", "s", "td", "q")):
        graded += _agree(result[name], case["wrf_p"][index], f"{name}_pl extrap={extrap}")
    below = case["wrf_p"][2] != F32(-999)
    assert graded > 2000
    if extrap == 2:
        # the extrapolated words exist: levels deeper than the lowest mass level are not missing
        assert below.sum() > (_load(1)["wrf_p"][2] != F32(-999)).sum()


@pytest.mark.parametrize("extrap", [1, 2])
def test_height_levels_equal_wrf_zld(extrap):
    case = _load(extrap)
    result = pl.height_levels(np, ht=case["zgrid"][0], levels=tuple(case["zlev"]), **_common(case))
    graded = 0
    for index, name in enumerate(("u", "v", "t", "rh", "ght", "s", "td", "q", "p")):
        graded += _agree(result[name], case["wrf_z"][index], f"{name}_zl extrap={extrap}")
    assert graded > 1500
    # above-ground levels are measured from each column's own terrain
    agl = case["wrf_z"][4][list(case["zlev"]).index(-100.0)]
    found = agl != F32(-999)
    assert found.any()
    np.testing.assert_allclose(agl[found], case["zgrid"][0][found] + 100.0, rtol=2e-7)


def test_the_search_state_carried_between_levels_is_graded():
    # the committed level lists are deliberately not sorted: a level above the last found layer, then a repeat
    # of a lower one, which WRF's carried ke_h cannot return to (missing).  Pin that the data holds the case.
    case = _load(1)
    levels = list(case["plev"])
    assert levels.index(85000.0) < len(levels) - 1 and levels[7] == 85000.0
    assert not (case["wrf_p"][2][7] != F32(-999)).any()
    assert (case["wrf_p"][2][2] != F32(-999)).any()


# ---------------------------------------------------------------------------
# the route's module: what it publishes, from which pools
# ---------------------------------------------------------------------------
class _Utils:
    def __init__(self, written):
        self.written = set(written)

    def field_will_be_written(self, name):
        return name in self.written


def _owner(case, layout_fields):
    nc, nz = case["nc"], case["nz"]
    u, v, theta, qv, pp, pb = case["mass"]
    pad = lambda a: np.concatenate([a, a[..., :1]], axis=-1)
    fields = {("mesh", "zgrid"): pad(case["zgrid"]), ("diag", "uReconstructZonal"): pad(u),
              ("diag", "uReconstructMeridional"): pad(v), ("diag", "theta"): pad(theta),
              ("diag", "pressure_p"): pad(pp), ("diag", "pressure_base"): pad(pb),
              ("state", "scalars"): pad(qv)[None]}
    buffers = {}
    shapes = {row["name"]: row["shape"] for row in layout_fields}

    def buffer(pool, name):
        return buffers.setdefault((pool, name), np.zeros(shapes[name], dtype=np.float32))

    session = SimpleNamespace(
        layout=SimpleNamespace(dimensions=dict(nCellsSolve=nc, nVertLevels=nz), indices={("state", "index_qv"): 1},
                               fields=[dict(pool="diag", name=name) for name in shapes]),
        config=None)
    return SimpleNamespace(cp=np, session=session, authority=lambda pool, name: fields[pool, name], buffer=buffer,
                           buffers=buffers)


def test_the_module_publishes_only_what_a_due_stream_carries():
    case = _load(2)
    config = pl.LevelConfig(press_levels=tuple(float(x) for x in case["plev"]),
                            z_levels=tuple(float(x) for x in case["zlev"]), extrap_below_grnd=2)
    shapes = [dict(name=name, shape=[len(config.press_levels) if name.endswith("_pl") or name == "p_pl" else
                                       len(config.z_levels)] + ([] if name in ("p_pl", "z_zl") else [case["nc"] + 1]))
              for name in pl.NAMES]
    owner = _owner(case, shapes)
    module = pl.LevelDiagnosticsModule(owner, config)
    utils = _Utils({"t_pl", "ght_pl", "p_pl", "p_zl", "z_zl"})
    module.setup(utils)
    module.compute()
    assert set(name for _, name in owner.buffers) == {"t_pl", "ght_pl", "p_pl", "p_zl", "z_zl"}
    nc = case["nc"]
    expected = pl.pressure_levels(np, pw=pl.interface_pressure(np, case["mass"][4] + case["mass"][5], case["zgrid"]),
                                  levels=config.press_levels, extrap_below_grnd=2, **_common(case))
    got_t = owner.buffers["diag", "t_pl"]
    assert np.array_equal(got_t[:, :nc], expected["t"]) and np.array_equal(owner.buffers["diag", "ght_pl"][:, :nc], expected["ght"])
    assert np.array_equal(owner.buffers["diag", "p_pl"], np.asarray(config.press_levels, np.float32))
    assert np.array_equal(owner.buffers["diag", "z_zl"], np.asarray(config.z_levels, np.float32))
    z = pl.height_levels(np, ht=case["zgrid"][0], levels=config.z_levels, **_common(case))
    assert np.array_equal(owner.buffers["diag", "p_zl"][:, :nc], z["p"])


def test_a_run_without_a_due_field_does_nothing():
    case = _load(1)
    config = pl.LevelConfig(press_levels=(85000.0,))
    owner = _owner(case, [dict(name="t_pl", shape=[1, case["nc"] + 1])])
    module = pl.LevelDiagnosticsModule(owner, config)
    module.setup(_Utils(set()))
    module.compute()
    assert not owner.buffers


def test_the_off_state_registry_streams_and_receipt_are_unchanged():
    """Digests recorded on origin/main (895eb6712), before these options existed."""
    import hashlib
    import json
    import tempfile
    from woof.hex.mpasphys.forecast_v842 import write_default_streams
    layout = _layout()
    digest = hashlib.sha256(json.dumps([layout.fields, sorted(layout.packages), sorted(layout.dimensions.items())],
                                       sort_keys=True, default=str).encode()).hexdigest()[:16]
    assert digest == "91afd1d2f4968e3f"
    with tempfile.TemporaryDirectory() as directory:
        write_default_streams(layout, Path(directory) / "streams")
        assert hashlib.sha256((Path(directory) / "streams").read_bytes()).hexdigest()[:16] == "b868bf60f2a107f7"
    receipt = _resolve().receipt()
    assert hashlib.sha256(json.dumps(receipt, sort_keys=True, default=str).encode()).hexdigest()[:16] == "bb186b77c0898a2c"
