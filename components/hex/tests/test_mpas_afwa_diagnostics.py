"""WRF's AFWA diagnostics on the MPAS suite route (woof.hex.mpasdiag.afwa_diag_v842).

&diagnostics config_afwa_diag_opt and its sub-options (therm, ptype, vil, radar, vis, buoy, severe) add WRF's AFWA fields as
``diag`` fields.  Here:

* the options resolve (off by default; any sub-option enables the base, as native WRF), icing is refused by name (no
  scheme of the route is one WRF's icing accepts), radar/VIL/vis by microphysics, severe with nwp_diagnostics, and every
  option on the RRFS fork route; the Registry, default streams and driver receipt carry the fields only with the
  options (the off state is pinned by digest);
* per mesh column, every field equals what the WRF route's own ``woof.core.afwa_diagnostics.update`` (the kernels that
  tests/test_native_afwa*.py and tools/diag_wrf471_oracle replay against WRF 4.7.1's Fortran) writes on a structured grid of
  the same columns, word for word;
* the same against WRF 4.7.1's Fortran (components/hex/tools/afwa_wrf471_oracle, gfortran), severe_wx_diagnostics
  and the driver's gust/shear blocks over three steps included;
* severe follows WRF's F_QNG guard (only gust and shear without a graupel-number carrier), reads the step's MPAS
  updraft helicity, and its maxima and tornado state reset after a frame and ride a checkpoint;
* ptype accumulates every step and rides a checkpoint, the t = 0 frame computes the instantaneous packages on the
  initial state, and a run whose microphysics lacks WRF's required hydrometeors is refused.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasdiag import afwa_diag_v842 as afwa
from woof.hex.mpasphys import resolve_namelist

NL, NC = 30, 24   # = make_columns.NL, NC


def _namelist(extra=None, physics=()):
    lines = ["&physics", " config_physics_suite='convection_permitting'", *physics, "/"]
    if extra is not None:
        lines += ["&diagnostics", *[f" {k}={v}" for k, v in extra.items()], "/"]
    return "\n".join(lines) + "\n"


def _resolve(extra=None, physics=()):
    return resolve_namelist(_namelist(extra, physics), default_backend="hex-cuda")


ALL = {f"config_afwa_{s}_opt": ".true." for s in ("diag", "therm", "ptype", "vil", "radar", "vis", "buoy", "severe")}


def test_options_default_off_and_resolve():
    assert _resolve().afwa_diagnostics is None and "afwa_diagnostics" not in _resolve().receipt()
    assert _resolve({"config_afwa_diag_opt": ".false."}).afwa_diagnostics is None
    config = _resolve({"config_afwa_therm_opt": ".true."})
    # native WRF: any sub-option enables the base package
    assert config.afwa_diagnostics.options == ("afwa_diag_opt", "afwa_therm_opt")
    assert afwa.packages(config) == {"afwa_diag_opt", "afwa_therm_opt"}
    assert config.receipt()["afwa_diagnostics"] == dict(options=["afwa_diag_opt", "afwa_therm_opt"],
                                                       ptype_ccn_tmp=264.15, ptype_tot_melt=50.0)
    every = _resolve({**ALL, "config_afwa_ptype_ccn_tmp": 260.0, "config_afwa_ptype_tot_melt": 55})
    assert every.afwa_diagnostics.options == afwa.SELECTORS
    assert (every.afwa_diagnostics.ptype_ccn_tmp, every.afwa_diagnostics.ptype_tot_melt) == (260.0, 55.0)
    for key in afwa.OPTIONS:
        assert key not in every.options
    mixed = _resolve({"config_nwp_diagnostics": ".true.", "config_afwa_diag_opt": ".true."})
    assert mixed.nwp_diagnostics and mixed.afwa_diagnostics is not None


@pytest.mark.parametrize("extra, message", [
    ({"config_afwa_icing_opt": ".true."}, "icing_opt is not available.*QNC"),
    ({"config_afwa_icing_opt": ".true."}, "WDM6 \\(mp_physics 16\\) and Morrison two-moment with aerosols \\(40\\)"),
    ({"config_afwa_severe_opt": ".true.", "config_nwp_diagnostics": ".true."}, "WSPD10MAX, W_UP_MAX, W_DN_MAX are one array"),
    ({"config_afwa_diag_opt": 1}, "logical"),
    ({"config_afwa_ptype_ccn_tmp": 260.0}, "read by nothing"),
    ({"config_afwa_diag_opt": ".true.", "config_afwa_ptype_tot_melt": 40.0}, "read by nothing"),
    ({"config_afwa_diag_opt": ".true.", "config_afwa_ptype_opt": ".true.", "config_afwa_ptype_ccn_tmp": ".true."}, "is real"),
])
def test_inconsistent_or_unavailable_options_are_refused(extra, message):
    with pytest.raises(ValueError, match=message):
        _resolve(extra)


def test_rrfs_fork_route_refuses_the_options():
    from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist as resolve_rrfs
    text = "&physics\n config_physics_suite='hrrrv5'\n/\n&diagnostics\n config_afwa_diag_opt=.true.\n/\n"
    with pytest.raises(ValueError, match="not available on the RRFS fork route"):
        resolve_rrfs(text)


def _layout(extra):
    from woof.hex.mpasphys.framework_backend_v842 import registry_source
    from woof.hex.mpasphys.pool_v842 import RegistryLayout
    text = _namelist(extra)
    config = resolve_namelist(text, default_backend="hex-cuda")
    dims = dict(nCells=10, nEdges=30, nVertices=20, nVertLevels=12, maxEdges=6, maxEdges2=12,
                vertexDegree=3, nSoilLevels=4, nOznLevels=59, nAerLevels=1, nMonths=12)
    return RegistryLayout(registry_source(), config, text, dims)


def test_pools_and_streams_only_with_the_options(tmp_path):
    from woof.hex.mpasphys.forecast_v842 import read_streams, write_default_streams
    from woof.hex.streams_v842 import field_definition, _all_packages
    off, on, base = _layout(None), _layout(ALL), _layout({"config_afwa_diag_opt": ".true."})
    rows = lambda layout: {r["name"]: r for r in layout.fields if r["pool"] == "diag"}
    assert not set(afwa.NAMES) & set(rows(off))
    assert set(afwa.NAMES) <= set(rows(on))
    assert set(afwa.BASE_NAMES) <= set(rows(base)) and not set(afwa.NAMES) - set(afwa.BASE_NAMES) & set(rows(base))
    assert [r for r in on.fields if r["name"] not in afwa.NAMES] == list(off.fields)
    assert set(afwa.SELECTORS) <= on.packages and not set(afwa.SELECTORS) & off.packages
    for name in afwa.NAMES:
        assert rows(on)[name]["dimensions"] == ["nCells"] and rows(on)[name]["shape"] == [11]
    streams = {}
    for label, layout in (("off", off), ("on", on)):
        write_default_streams(layout, tmp_path / f"streams.{label}")
        streams[label] = {s.name: s for s in read_streams(tmp_path / f"streams.{label}", layout)}
    assert set(afwa.NAMES) <= set(streams["on"]["diagnostics"].fields)
    assert not set(afwa.NAMES) & set(streams["off"]["diagnostics"].fields)
    assert {n: s.fields for n, s in streams["on"].items() if n != "diagnostics"} == \
           {n: s.fields for n, s in streams["off"].items() if n != "diagnostics"}
    assert [f for f in streams["on"]["diagnostics"].fields if f not in afwa.NAMES] == \
           list(streams["off"]["diagnostics"].fields)
    d = field_definition("afwa_cape")
    assert d["attrs"]["units"] == "J kg-1" and d["attrs"]["wrf_name"] == "AFWA_CAPE"
    assert d["gates"] == ["afwa_buoy_opt;afwa_severe_opt"]
    # one WRF array with nwp_diagnostics: one definition, either package carries it
    assert field_definition("wspd10max")["gates"] == ["nwp_diagnostics;afwa_severe_opt"]
    severe = _layout({"config_afwa_severe_opt": ".true."})
    assert set(afwa.SEVERE_NAMES) <= set(rows(severe)) and not set(afwa.PTYPE_NAMES) & set(rows(severe))
    write_default_streams(severe, tmp_path / "streams.severe")
    fields = {s.name: s for s in read_streams(tmp_path / "streams.severe", severe)}["diagnostics"].fields
    assert set(afwa.SEVERE_NAMES) <= set(fields) and "up_heli_max" not in fields and "w_mean" not in fields
    assert set(afwa.SELECTORS) <= _all_packages()


def test_the_field_rows_are_the_wrf_routes_rows():
    from woof.core.output_diagnostics import HISTORY as OUTPUT
    from woof.core.afwa_diagnostics import HISTORY
    assert set(afwa.NAMES) <= set(OUTPUT)
    for name in afwa.NAMES:
        # (the route's output table carries WSPD10MAX with the wind-gust package's text; the AFWA table's is WRF's)
        wrf, units, text = HISTORY[name]
        a = afwa.FIELDS[name]["attrs"]
        assert (a["wrf_name"], a["units"], a["long_name"]) == (wrf, units, text)


def test_the_off_state_registry_streams_and_receipt_are_unchanged():
    """Digests recorded on origin/main (fe554cdb2), before these options existed (the p_lev / z_lev test's values)."""
    import hashlib
    import json
    import tempfile
    from pathlib import Path
    from woof.hex.mpasphys.forecast_v842 import write_default_streams
    layout = _layout(None)
    digest = hashlib.sha256(json.dumps([layout.fields, sorted(layout.packages), sorted(layout.dimensions.items())],
                                       sort_keys=True, default=str).encode()).hexdigest()[:16]
    assert digest == "91afd1d2f4968e3f"
    with tempfile.TemporaryDirectory() as directory:
        write_default_streams(layout, Path(directory) / "streams")
        assert hashlib.sha256((Path(directory) / "streams").read_bytes()).hexdigest()[:16] == "b868bf60f2a107f7"
    assert hashlib.sha256(json.dumps(_resolve().receipt(), sort_keys=True, default=str).encode()).hexdigest()[:16] == "bb186b77c0898a2c"


# ---------------------------------------------------------------------------
# synthetic mesh columns, as pools
# ---------------------------------------------------------------------------
def _load_columns_module():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "tools" / "afwa_wrf471_oracle" / "make_columns.py"
    spec = importlib.util.spec_from_file_location("afwa_make_columns", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAKE = _load_columns_module()
columns = MAKE.columns


def _ring(nc=NC, periodic=True):
    """cellsOnCell (maxEdges = 2, nCells + 1) and nEdgesOnCell of cells in a row: WRF's j = 1 row of a ny = 1 grid.
    Without ``periodic`` the ends have one neighbour (the other slot is MPAS's nCells + 1 sentinel)."""
    cells = np.zeros((2, nc + 1), np.int32)
    index = np.arange(nc)
    cells[0, :nc], cells[1, :nc] = (index - 1) % nc + 1, (index + 1) % nc + 1
    if not periodic:
        cells[0, 0], cells[1, nc - 1] = nc + 1, nc + 1
    return cells, np.full(nc + 1, 2, np.int32)


def _owner(cp, mass, surface, scalars=("qv", "qc", "qi", "qr", "qs", "qg"), nl=NL, nc=NC, w=None, periodic=True):
    fields, buffers = {}, {}
    pad = lambda a: cp.asarray(np.concatenate([a, a[..., :1]], axis=-1))
    cells, edges = _ring(nc, periodic)
    fields["mesh", "cellsOnCell"], fields["mesh", "nEdgesOnCell"] = cp.asarray(cells), cp.asarray(edges)
    fields["state", "w"] = pad(np.zeros((nl + 1, nc), np.float32) if w is None else w)
    for name in ("zgrid", "zz"):
        fields["mesh", name] = pad(mass[name])
    fields["state", "rho_zz"] = pad(mass["rho_zz"])
    for name in ("theta", "exner", "pressure_base", "pressure_p"):
        fields["diag", name] = pad(mass[name])
    fields["diag", "uReconstructZonal"] = pad(mass["u"])
    fields["diag", "uReconstructMeridional"] = pad(mass["v"])
    fields["diag", "surface_pressure"] = pad(surface["surface_pressure"])
    layout = [dict(pool="diag", pool_name="surface_pressure")]
    for name in ("t2m", "q2", "u10", "v10", "swdnb", "raincv", "rainncv", "rainc", "rainnc"):
        fields["diag_physics", name] = pad(surface[name])
        layout.append(dict(pool="diag_physics", pool_name=name))
    stack = cp.asarray(np.stack([pad(mass[n]) if n in mass else pad(np.full((nl, nc), 1e5, np.float32))
                                 for n in scalars]))
    fields["state", "scalars"] = stack
    indices = {("state", "index_" + n): i + 1 for i, n in enumerate(scalars)}
    for n in ("qv", "qc", "qi", "qr", "qs", "qg"):
        indices.setdefault(("state", "index_" + n), -1)
    session = SimpleNamespace(layout=SimpleNamespace(dimensions=dict(nCellsSolve=nc, nVertLevels=nl), fields=layout,
                                                     indices=indices))

    def buffer(pool, name):
        return buffers.setdefault((pool, name), cp.zeros(nc + 1, dtype=cp.float32))

    start = dt.datetime(2026, 1, 1)
    return SimpleNamespace(cp=cp, session=session, authority=lambda pool, name: fields[pool, name], buffer=buffer,
                           fields=fields, buffers=buffers, writer=SimpleNamespace(start=start), start=start,
                           dycore=SimpleNamespace(config=SimpleNamespace(config_dt=12.0)), halo=None)


class _Utils:
    def __init__(self, written=()):
        self.written = set(written)

    def stream_inclusion_count(self, name, direction=None, **kw):
        return 1

    def field_will_be_written(self, name):
        return name in self.written


@pytest.fixture
def cp():
    return pytest.importorskip("cupy")


def _module(cp, options=afwa.SELECTORS, seed=11, scalars=("qv", "qc", "qi", "qr", "qs", "qg"), **config):
    mass, surface = columns(seed)
    w, uh = MAKE.severe_columns()
    owner = _owner(cp, mass, surface, scalars, w=w)
    module = afwa.AfwaModule(owner, afwa.AfwaConfig(tuple(options), **config))
    utils = _Utils(afwa.NAMES)
    module.setup(utils)
    module.helicity = lambda: cp.asarray(uh)
    return module, owner, utils, mass, surface


# ---------------------------------------------------------------------------
# the WRF route's own afwa_diagnostics.update on the same columns, as a structured (nz, 1, nc) grid
# ---------------------------------------------------------------------------
def _wrf_route(cp, mass, surface, options, ccn=264.15, melt=50.0, mp_physics=8, steps=((0.0, 12.0),)):
    """The WRF route on the columns; ``steps`` are (elapsed_seconds, output_interval_s) per update (the route derives
    its history reset and output step from them), the scratch after the last is returned."""
    from woof.core import afwa_diagnostics as wrf
    f = np.float32
    nl, nc = NL, NC
    w, uh = MAKE.severe_columns()
    cfg = SimpleNamespace(
        nz=nl, ny=1, nx=nc, mp_physics=mp_physics, moist=True, dt=12.0, output_interval_s=3600.0, afwa_bad_data_check=0,
        afwa_diag_opt=1, afwa_icing_opt=0, afwa_ptype_ccn_tmp=ccn, afwa_ptype_tot_melt=melt,
        specified=False, nested=False, open_x=False, open_y=False,
        **{s: int(s in options) for s in afwa.SELECTORS if s != "afwa_diag_opt"})
    scratch = {name: cp.zeros(shape, dtype=cp.float32) for name, shape in wrf.scratch_shapes(cfg).items()}
    g = lambda a, shape: cp.asarray(np.ascontiguousarray(a.reshape(shape)))
    cube = lambda a: g(a, (nl, 1, nc))
    plane = lambda a: g(a, (1, nc))
    zgrid = mass["zgrid"]
    alt = f(1) / (mass["rho_zz"] * mass["zz"])
    physics = SimpleNamespace(
        cu_raincv=plane(surface["raincv"]), rainc=plane(surface["rainc"]),
        microphysics=SimpleNamespace(rainncv=plane(surface["rainncv"]), rainnc=plane(surface["rainnc"])),
        fields=dict(psfc=plane(surface["surface_pressure"]), t2=plane(surface["t2m"]), q2=plane(surface["q2"]),
                    u10=plane(surface["u10"]), v10=plane(surface["v10"]), swdown=plane(surface["swdnb"])))
    state = SimpleNamespace(
        elapsed_seconds=3588.0, domain_start_offset=0.0, qv=cube(mass["qv"]), qr=cube(mass["qr"]), qs=cube(mass["qs"]),
        qg=cube(mass["qg"]), qc=cube(mass["qc"]), qi=cube(mass["qi"]), nc=cube(mass["qc"]), p=cube(mass["p"]),
        w=g(w, (nl + 1, 1, nc)),
        alt=cube(alt), php=g(np.zeros((nl + 1, nc), f), (nl + 1, 1, nc)), phb=g((zgrid * f(9.81)).astype(f), (nl + 1, 1, nc)),
        ht=plane(zgrid[0]), physics=physics,
        scratch=lambda shape, name: scratch[name], existing_scratch=lambda name: scratch[name])
    atmosphere = dict(temperature=cube(mass["T"]), z_interface=g(zgrid, (nl + 1, 1, nc)),
                      u=cube(mass["u"]), v=cube(mass["v"]))
    if "uh_diag_col" in scratch:
        scratch["uh_diag_col"][...] = plane(uh)
    every = []
    for elapsed, interval in steps:
        state.elapsed_seconds, cfg.output_interval_s = elapsed, interval
        wrf.capture_atmosphere(state, cfg, atmosphere)
        wrf.prepare_microphysics(state, cfg)
        wrf.update(state, cfg)
        every.append({name: cp.asnumpy(scratch[name]).reshape(-1)[:nc].copy() for name in (*afwa.NAMES, *wrf.SEVERE_STATE)
                      if name in scratch})
    return every[-1] if len(steps) == 1 else every


@pytest.mark.parametrize("options", [afwa.SELECTORS, ("afwa_diag_opt", "afwa_ptype_opt"), ("afwa_diag_opt", "afwa_buoy_opt",
                                                                                          "afwa_therm_opt")])
def test_every_field_equals_the_wrf_routes_kernels_on_the_same_columns(cp, options):
    module, owner, utils, mass, surface = _module(cp, options, ptype_ccn_tmp=263.15, ptype_tot_melt=45.0)
    expected = _wrf_route(cp, mass, surface, options, ccn=263.15, melt=45.0)
    module.begin(1, owner.start + dt.timedelta(seconds=12))
    module.update()
    module.compute()
    produced = {n for s in options for n in afwa.TABLES[s]}
    if "afwa_severe_opt" in options:
        # neither side carries graupel number (mp_physics 8): WRF's F_QNG guard leaves only the gust and the shear
        assert not module.severe_column and (expected["wspd10max"] > 0).all() and (expected["afwa_llws"] > 0).all()
        assert not expected["afwa_hail"].any() and not expected["w_up_max"].any()
    assert produced <= set(expected)
    for name in sorted(produced):
        got = cp.asnumpy(owner.buffers["diag", name])[:NC]
        want = expected[name]
        both_nan = np.isnan(got) & np.isnan(want)
        np.testing.assert_array_equal(got.view(np.uint32)[~both_nan], want.view(np.uint32)[~both_nan], err_msg=name)
        assert np.array_equal(np.isnan(got), np.isnan(want)), name
    # the columns exercise the branches: precipitation of every type, an echo, CAPE, a heat index and a visibility
    if "afwa_ptype_opt" in options:
        for name in afwa.PTYPE_NAMES:
            assert (np.nan_to_num(expected[name]) > 0).any(), name
    if "afwa_buoy_opt" in options:
        assert (expected["afwa_cape"] > 0).any() and (expected["afwa_cin"] != 0).any()
    if "afwa_radar_opt" in options:
        assert (expected["afwa_refd"] > 0).any() and (expected["afwa_echotop"] > 0).any()
    if "afwa_vis_opt" in options:
        assert (expected["afwa_vis"] < 999999.0).any() and (expected["afwa_vis"] > 0).all()
    if "afwa_therm_opt" in options:
        assert (expected["afwa_heatidx"] > 295).any() and (expected["afwa_wchill"] < 273).any()


# ---------------------------------------------------------------------------
# WRF 4.7.1's own Fortran (components/hex/tools/afwa_wrf471_oracle) on the same columns
# ---------------------------------------------------------------------------
ORACLE = Path(__file__).resolve().parent / "afwa_oracle"
#: out.bin of run_afwa.F90, in order
ORACLE_FIELDS = ("afwa_mslp", "afwa_pwat", "afwa_totprecip", "precip", "afwa_heatidx", "afwa_wchill", "afwa_fits",
                 "afwa_rain", "afwa_snow", "afwa_ice", "afwa_fzra", "afwa_snowfall", "afwa_vil", "afwa_radarvil",
                 "afwa_refd", "afwa_refd_com", "afwa_echotop", "afwa_vis", "afwa_vis_dust", "afwa_vis_alpha",
                 "afwa_cape", "afwa_cin", "afwa_zlfc", "afwa_plfc", "afwa_lidx", "afwa_cape_mu", "afwa_cin_mu")
#: relative tolerance per field: the kernels and the Fortran differ only in expf/powf/log10f/tanhf (the device's libm
#: against the build machine's); the parcel integrals amplify a few ulp
RTOL = {"afwa_cape": 2e-3, "afwa_cin": 2e-3, "afwa_cape_mu": 2e-3, "afwa_cin_mu": 2e-3}
DEFAULT_RTOL = 1e-4
#: THE FINDING.  WRF's precip_type_diagnostics writes ``IF (rh .gt. 80 .and. saturation .eqv. .false.)``, which Fortran
#: parses as ``(rh .gt. 80 .and. saturation) .eqv. .false.`` (.eqv. binds weakest): the cloud top is re-assigned at every
#: level that is not (saturated and RH > 80).  woof/core/kernels/afwa_diagnostics.cu afwa_ptype (the WRF route's kernel,
#: shared by the hex module) tests ``rh > 80 && !saturation``, the intent.  The two agree on every column of the oracle but
#: these, where a column has a moist layer below another moist layer with a dry gap between them.
PTYPE_DIVERGES_AT = (15,)


def _columns_from_file():
    """The oracle's input columns as the pools hold them: T = theta * 1 and p = pressure_base + 0, so the module sees the
    Fortran's words exactly (numpy's exp/power differ between hosts: the committed file, not a regeneration, is the input)."""
    raw = (ORACLE / "columns.bin").read_bytes()
    nc, nl = (int(v) for v in np.frombuffer(raw, "<i4", 2))
    ccn, melt = (float(v) for v in np.frombuffer(raw, "<f4", 2, 8))
    body = np.frombuffer(raw, "<f4", offset=16)
    take = iter(np.split(body, np.cumsum([(nl + 1) * nc] + [nl * nc] * 11 + [nc] * 10)[:-1]))
    zgrid = next(take).reshape(nl + 1, nc)
    names = ("rho_zz", "T", "p", "qv", "qc", "qi", "qr", "qs", "qg", "u", "v")
    mass = {n: next(take).reshape(nl, nc) for n in names}
    mass.update(zgrid=zgrid, zz=np.ones((nl, nc), np.float32), theta=mass["T"], exner=np.ones((nl, nc), np.float32),
                pressure_base=mass["p"], pressure_p=np.zeros((nl, nc), np.float32))
    surface = {n: next(take) for n in ("surface_pressure", "t2m", "q2", "u10", "v10", "swdnb", "raincv", "rainncv",
                                       "rainc", "rainnc")}
    return nc, nl, ccn, melt, mass, surface


def test_fields_equal_wrf_fortran_on_the_same_columns(cp):
    nc, nl, ccn, melt, mass, surface = _columns_from_file()
    assert (nc, nl) == (NC, NL) and (ccn, melt) == (np.float32(MAKE.CCN), np.float32(MAKE.MELT))
    # the generator reproduces the committed columns (to the host's numpy exp/power round-off)
    regenerated, _ = MAKE.columns()
    np.testing.assert_allclose(regenerated["T"], mass["T"], rtol=1e-5)
    oracle = np.fromfile(ORACLE / "oracle.bin", "<f4").reshape(len(ORACLE_FIELDS), nc)
    owner = _owner(cp, mass, surface)
    module = afwa.AfwaModule(owner, afwa.AfwaConfig(afwa.SELECTORS, ptype_ccn_tmp=ccn, ptype_tot_melt=melt))
    module.setup(_Utils(afwa.NAMES))
    module.begin(1, owner.start + dt.timedelta(seconds=12))
    module.update()
    module.compute()
    keep = np.setdiff1d(np.arange(nc), PTYPE_DIVERGES_AT)
    for index, name in enumerate(ORACLE_FIELDS):
        if name == "precip":
            np.testing.assert_array_equal(cp.asnumpy(module.w["afwa_precip"])[:nc].view(np.uint32), oracle[index].view(np.uint32))
            continue
        got, want = cp.asnumpy(owner.buffers["diag", name])[:nc], oracle[index]
        if name in afwa.PTYPE_NAMES:
            got, want = got[keep], want[keep]
        assert np.array_equal(np.isnan(got), np.isnan(want)), name
        ok = np.isfinite(want)
        np.testing.assert_allclose(got[ok], want[ok], rtol=RTOL.get(name, DEFAULT_RTOL), atol=0.0, err_msg=name)
    # the columns exercise every branch of the Fortran: every precipitation type, echo, CAPE, CIN, fog and haze
    for name in ("afwa_rain", "afwa_snow", "afwa_ice", "afwa_fzra", "afwa_cape", "afwa_refd", "afwa_echotop",
                 "afwa_vil", "afwa_heatidx"):
        assert (oracle[ORACLE_FIELDS.index(name)][keep] > 0).any(), name
    cin = oracle[ORACLE_FIELDS.index("afwa_cin")]
    assert ((cin < 0) & (cin > -1e30)).any()
    alpha = oracle[ORACLE_FIELDS.index("afwa_vis_alpha")]
    assert (alpha == np.float32(3.6)).any() and (alpha < np.float32(3.6)).any()      # hydrometeor- and haze-dominated


def test_the_precip_type_divergence_from_wrf_fortran_is_pinned(cp):
    """If this fails the shared kernel was changed to WRF's actual operator precedence: drop PTYPE_DIVERGES_AT."""
    nc, nl, ccn, melt, mass, surface = _columns_from_file()
    oracle = np.fromfile(ORACLE / "oracle.bin", "<f4").reshape(len(ORACLE_FIELDS), nc)
    owner = _owner(cp, mass, surface)
    module = afwa.AfwaModule(owner, afwa.AfwaConfig(afwa.SELECTORS, ptype_ccn_tmp=ccn, ptype_tot_melt=melt))
    module.setup(_Utils(afwa.NAMES))
    module.begin(1, owner.start + dt.timedelta(seconds=12))
    module.update()
    differ = set()
    for name in afwa.PTYPE_NAMES:
        got, want = cp.asnumpy(module.w[name])[:nc], oracle[ORACLE_FIELDS.index(name)]
        differ |= {int(i) for i in np.where(~np.isclose(got, want, rtol=1e-4, atol=0.0))[0]}
    assert differ == set(PTYPE_DIVERGES_AT)


# ---------------------------------------------------------------------------
# the per-step fold, the due frame, checkpoint, refusals, the provider
# ---------------------------------------------------------------------------
def test_ptype_accumulates_every_step_and_the_t0_frame_computes_the_instantaneous_packages(cp):
    module, owner, utils, mass, surface = _module(cp, afwa.SELECTORS)
    module.begin(0, owner.start)
    module.update()                                    # no step has run: nothing is folded
    module.compute()                                   # but the t = 0 frame has its instantaneous words
    assert not any(cp.asnumpy(owner.buffers["diag", n]).any() for n in afwa.PTYPE_NAMES)
    assert cp.asnumpy(owner.buffers["diag", "afwa_mslp"])[:NC].min() > 90000 and cp.asnumpy(owner.buffers["diag", "afwa_pwat"]).any()
    module.begin(1, owner.start + dt.timedelta(seconds=12))
    module.update()
    once = {n: cp.asnumpy(module.w[n]).copy() for n in afwa.PTYPE_NAMES}
    assert any(v.any() for v in once.values())
    module.begin(2, owner.start + dt.timedelta(seconds=24))
    module.update()
    for n, v in once.items():
        np.testing.assert_allclose(cp.asnumpy(module.w[n]), 2 * v, rtol=1e-6, atol=0.0, err_msg=n)
    module.compute()
    np.testing.assert_array_equal(cp.asnumpy(owner.buffers["diag", "afwa_rain"])[:NC], cp.asnumpy(module.w["afwa_rain"]))


#: every package but severe (whose state the severe tests check)
NO_SEVERE = tuple(s for s in afwa.SELECTORS if s != "afwa_severe_opt")


def test_a_checkpoint_resumes_the_ptype_accumulators(cp):
    module, owner, utils, mass, surface = _module(cp, NO_SEVERE)
    for step in (1, 2, 3):
        module.begin(step, owner.start + dt.timedelta(seconds=12 * step))
        module.update()
    state = module.state()
    assert set(state) == {afwa.STATE_PREFIX + n for n in afwa.PTYPE_NAMES}
    fresh, f_owner, _, _, _ = _module(cp, NO_SEVERE)
    fresh.resume(state)
    fresh.begin(4, f_owner.start + dt.timedelta(seconds=48))
    fresh.update()
    module.begin(4, owner.start + dt.timedelta(seconds=48))
    module.update()
    for n in afwa.PTYPE_NAMES:
        np.testing.assert_array_equal(cp.asnumpy(fresh.w[n]).view(np.uint32), cp.asnumpy(module.w[n]).view(np.uint32))


def test_radar_vil_and_vis_need_the_hydrometeors_wrf_reads(cp):
    for options, scalars, missing in (
            (("afwa_diag_opt", "afwa_radar_opt"), ("qv", "qc", "qr"), "qs, qg"),
            (("afwa_diag_opt", "afwa_vil_opt"), ("qv", "qr", "qs"), "qg"),
            (("afwa_diag_opt", "afwa_vis_opt"), ("qv", "qr", "qs", "qg"), "qc, qi")):
        mass, surface = columns()
        owner = _owner(cp, mass, surface, scalars)
        with pytest.raises(ValueError, match=f"needs the {missing} carrier"):
            afwa.AfwaModule(owner, afwa.AfwaConfig(options)).setup(_Utils(afwa.NAMES))
    # the thermal, base and buoyancy packages need only moisture
    owner = _owner(cp, *columns(), ("qv", "qc"))
    afwa.AfwaModule(owner, afwa.AfwaConfig(("afwa_diag_opt", "afwa_therm_opt", "afwa_buoy_opt"))).setup(_Utils(afwa.NAMES))


def test_only_due_carried_fields_are_published(cp):
    module, owner, utils, mass, surface = _module(cp, afwa.SELECTORS)
    utils.written = {"afwa_cape", "afwa_mslp"}
    module.begin(1, owner.start + dt.timedelta(seconds=12))
    module.update()
    module.compute()
    assert set(name for (pool, name) in owner.buffers) == {"afwa_cape", "afwa_mslp"}
    idle, o2, u2, _, _ = _module(cp, afwa.SELECTORS)
    u2.written = set()
    idle.compute()
    assert not o2.buffers


def test_resident_provider_folds_publishes_and_carries_the_module(cp):
    from woof.hex.mpasdiag.convective_v842 import MAXIMA
    from woof.hex.mpasphys.resident_output_diagnostics_v842 import PRODUCED, ResidentOutputDiagnostics as R
    module, owner, utils, mass, surface = _module(cp, NO_SEVERE)
    provider = R.__new__(R)
    provider.cp, provider.afwa, provider.accdiag, provider.nwp, provider.tslist = cp, module, None, None, None
    provider.buffers, provider._event_views, provider._resume, provider.updates, provider.acoustic = {}, None, None, 0, {}
    provider.convective_maxima = MAXIMA
    calls = []
    provider.convective = SimpleNamespace(update=lambda: calls.append("convective"), reset=lambda: calls.append("reset"))
    assert R.produced is PRODUCED and not set(afwa.NAMES) & set(PRODUCED)
    module.begin(1, owner.start + dt.timedelta(seconds=12))
    provider._update_convective()
    assert calls == ["convective"] and any(cp.asnumpy(module.w[n]).any() for n in afwa.PTYPE_NAMES)
    state = provider.accumulator_state()
    assert {afwa.STATE_PREFIX + n for n in afwa.PTYPE_NAMES} <= set(state)
    provider.resume_accumulators(state)
    assert set(provider._resume) == {afwa.STATE_PREFIX + n for n in afwa.PTYPE_NAMES}
    provider.after_snapshot()           # ptype accumulates for the whole run: a write resets nothing
    assert any(cp.asnumpy(module.w[n]).any() for n in afwa.PTYPE_NAMES)


# ---------------------------------------------------------------------------
# severe: WRF's F_QNG guard, the step's helicity, the reset, the WRF route's kernels and WRF's Fortran
# ---------------------------------------------------------------------------
WITH_NG = ("qv", "qc", "qi", "qr", "qs", "qg", "ng")
SEVERE_CHECKED = (*afwa.SEVERE_NAMES, "afwa_midrh_min", "afwa_midrh_min_old", "afwa_tornado_mask", "afwa_tornado_dur")


def _severe_steps(module, owner, utils):
    """The three steps of run_severe.F90: (reset, buoyancy), (neither), a frame then (reset, buoyancy)."""
    after = []
    for step in (1, 2, 3):
        utils.written = set(afwa.NAMES) if step != 2 else set()
        if step == 3:
            module.reset()                      # the frame written after step 2
        module.begin(step, owner.start + dt.timedelta(seconds=12 * step))
        module.update()
        after.append({n: module.cp.asnumpy(module.w[n]).copy() for n in SEVERE_CHECKED})
    return after


def test_severe_equals_the_wrf_routes_kernels_over_three_steps(cp):
    options = ("afwa_diag_opt", "afwa_severe_opt")
    module, owner, utils, mass, surface = _module(cp, options, scalars=WITH_NG)
    assert module.severe_column
    got = _severe_steps(module, owner, utils)
    # the WRF route, mp_physics 10 (Morrison: QNG carried), its periodic row = the module's ring of neighbours;
    # (elapsed, interval) give the route's reset/output decisions: (yes, yes), (no, no), (yes, yes)
    want = _wrf_route(cp, mass, surface, options, mp_physics=10, steps=((0.0, 12.0), (24.0, 1e6), (36.0, 12.0)))
    for step, (g, e) in enumerate(zip(got, want), 1):
        for name in SEVERE_CHECKED:
            np.testing.assert_array_equal(g[name].view(np.uint32), e[name].view(np.uint32), err_msg=f"{name} step {step}")
    last = got[-1]
    assert (last["afwa_hail"] > 0).any() and (last["afwa_tornado"] > 0).any() and (last["afwa_tornado_dur"] > 12).any()
    assert (last["tcoli_max"] > 0).any() and (last["grpl_flx_max"] > 0).any() and (last["w_dn_max"] < 0).any()


def test_severe_without_graupel_number_moves_only_gust_and_shear(cp):
    """WRF calls severe_wx_diagnostics only with F_QNG; no microphysics of this route carries ng."""
    module, owner, utils, mass, surface = _module(cp, ("afwa_diag_opt", "afwa_severe_opt"))
    module.helicity = lambda: pytest.fail("the helicity is read only by the guarded column part")
    after = _severe_steps(module, owner, utils)[-1]
    assert (after["wspd10max"] > 0).all() and (after["afwa_llws"] > 0).all()
    for name in set(SEVERE_CHECKED) - {"wspd10max", "afwa_llws", "afwa_midrh_min", "afwa_midrh_min_old"}:
        assert not after[name].any(), name
    assert (after["afwa_midrh_min"] == np.float32(999.0)).all()     # reset, never folded: WRF's words too


SEVERE_ORACLE_FIELDS = ("wspd10max", "afwa_llws", "w_up_max", "w_dn_max", "tcoli_max", "grpl_flx_max", "afwa_hail",
                        "afwa_tornado", "afwa_cape", "afwa_cin", "afwa_zlfc", "afwa_plfc", "afwa_lidx", "afwa_midrh_min",
                        "afwa_midrh_min_old", "afwa_tornado_mask", "afwa_tornado_dur")
SEVERE_RTOL = {"afwa_cape": 2e-3, "afwa_cin": 2e-3, "afwa_tornado": 1e-3}


def test_severe_equals_wrf_fortran_over_three_steps(cp):
    """run_severe.F90: severe_wx_diagnostics and the driver's gust/shear blocks, gfortran, on the committed columns.
    The Fortran's row has zero halos at both ends: the module's neighbours are the row without wrap."""
    nc, nl, ccn, melt, mass, surface = _columns_from_file()
    raw = np.fromfile(ORACLE / "severe_columns.bin", "<f4")
    step_dt, w, uh = raw[0], raw[1:1 + (nl + 1) * nc].reshape(nl + 1, nc), raw[1 + (nl + 1) * nc:]
    assert step_dt == np.float32(MAKE.SEVERE_DT) and uh.size == nc
    regenerated_w, regenerated_uh = MAKE.severe_columns()
    np.testing.assert_allclose(regenerated_w, w, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(regenerated_uh, uh)
    oracle = np.fromfile(ORACLE / "severe.bin", "<f4").reshape(3, len(SEVERE_ORACLE_FIELDS), nc)
    owner = _owner(cp, mass, surface, WITH_NG, w=w, periodic=False)
    module = afwa.AfwaModule(owner, afwa.AfwaConfig(("afwa_diag_opt", "afwa_severe_opt")))
    utils = _Utils(afwa.NAMES)
    module.setup(utils)
    module.helicity = lambda: cp.asarray(uh)
    got = _severe_steps(module, owner, utils)
    for step in range(3):
        for index, name in enumerate(SEVERE_ORACLE_FIELDS):
            g, want = got[step][name], oracle[step, index]
            np.testing.assert_allclose(g, want, rtol=SEVERE_RTOL.get(name, 1e-4), atol=1e-5,
                                       err_msg=f"{name} step {step + 1}")
    # the steps exercise the branches: blended gust, hail, a tornado whose duration spread to neighbours, the
    # closed window's mid-level RH minimum
    assert (oracle[2, SEVERE_ORACLE_FIELDS.index("afwa_tornado_dur")] == np.float32(36)).sum() > \
        (oracle[2, SEVERE_ORACLE_FIELDS.index("afwa_tornado_mask")] > 0).sum()
    assert (oracle[2, SEVERE_ORACLE_FIELDS.index("afwa_hail")] > 0).any()
    assert (oracle[2, SEVERE_ORACLE_FIELDS.index("afwa_midrh_min_old")] > 0).all()


def test_the_step_helicity_is_the_one_updraft_helicity_max_folds(cp):
    """helicity() runs MPAS's update_diag on a zeroed row: its values over the steps fold to updraft_helicity_max."""
    from woof.hex.mpasdiag.convective_v842 import ConvectiveDiagnostics, FIELDS as CONVECTIVE
    rng = np.random.default_rng(3)
    nc, nl, nv, degree = 9, 6, 18, 3
    f = np.float32

    def arrays():
        z = np.cumsum(rng.uniform(300, 1200, (nc, nl + 1)), axis=1).astype(f)
        z[:, 0] = 0
        return dict(zgrid=cp.asarray(z), w=cp.asarray(rng.uniform(-3, 25, (nc, nl + 1)).astype(f)),
                    areaCell=cp.asarray(rng.uniform(1e6, 2e6, nc).astype(f)),
                    cellsOnVertex=cp.asarray(rng.integers(1, nc + 1, (nv, degree)).astype(np.int32)),
                    kiteAreasOnVertex=cp.asarray(rng.uniform(1e5, 4e5, (nv, degree)).astype(f)),
                    vorticity=cp.asarray(rng.uniform(-0.01, 0.03, (nv, nl)).astype(f)))
    steps = [arrays() for _ in range(3)]
    reference = cp.zeros((len(CONVECTIVE), nc), dtype=cp.float32)
    folded = ConvectiveDiagnostics(reference, nc, nl, ("updraft_helicity_max",))
    convective = SimpleNamespace(n_alloc=nc, driver=SimpleNamespace(vertex_order=None))
    module = afwa.AfwaModule(SimpleNamespace(cp=cp, convective=convective), afwa.AfwaConfig())
    module.cp, module.nc, module.nl, module._uh = cp, nc, nl, None
    values = []
    for step in steps:
        convective._arrays = lambda names, step=step: {n: step[n] for n in names}
        values.append(cp.asnumpy(module.helicity()))
        folded.update(step)
    assert (np.stack(values) > 0).any()
    np.testing.assert_array_equal(np.max(np.stack(values), axis=0), cp.asnumpy(reference[CONVECTIVE.index("updraft_helicity_max")]))


def test_a_checkpoint_resumes_the_severe_state(cp):
    options = ("afwa_diag_opt", "afwa_severe_opt")
    module, owner, utils, mass, surface = _module(cp, options, scalars=WITH_NG)
    utils.written = set()
    for step in (1, 2):
        module.begin(step, owner.start + dt.timedelta(seconds=12 * step))
        module.update()
    utils.written = set(afwa.NAMES)
    module.reset()
    state = module.state()
    assert {afwa.STATE_PREFIX + n for n in (*afwa.SEVERE_NAMES, *afwa.SEVERE_STATE)} | {afwa.SEVERE_RESET_KEY} == set(state)
    fresh, f_owner, f_utils, _, _ = _module(cp, options, scalars=WITH_NG)
    fresh._reset = False
    fresh.resume(state)
    for m, o in ((module, owner), (fresh, f_owner)):
        m.begin(3, o.start + dt.timedelta(seconds=36))
        m.update()
    for n in SEVERE_CHECKED:
        np.testing.assert_array_equal(cp.asnumpy(fresh.w[n]).view(np.uint32), cp.asnumpy(module.w[n]).view(np.uint32), n)


def test_severe_is_refused_on_a_multi_rank_mesh(cp):
    mass, surface = columns()
    owner = _owner(cp, mass, surface)
    owner.halo = object()
    with pytest.raises(ValueError, match="multi-rank"):
        afwa.AfwaModule(owner, afwa.AfwaConfig(("afwa_diag_opt", "afwa_severe_opt"))).setup(_Utils(afwa.NAMES))


@pytest.mark.parametrize("scheme", afwa.HYDROMETEOR_SCHEMES)
def test_radar_vil_and_vis_resolve_for_every_scheme_that_carries_the_hydrometeors(scheme):
    every = {**ALL, "config_afwa_severe_opt": ".false."}
    physics = (f" config_microp_scheme='{scheme}'",)
    config = _resolve(every, physics)
    assert {"afwa_radar_opt", "afwa_vil_opt", "afwa_vis_opt"} <= set(config.afwa_diagnostics.options)
    from woof.hex.mpasphys.framework_backend_v842 import registry_source
    from woof.hex.mpasphys.pool_v842 import RegistryLayout
    dims = dict(nCells=10, nEdges=30, nVertices=20, nVertLevels=12, maxEdges=6, maxEdges2=12,
                vertexDegree=3, nSoilLevels=4, nOznLevels=59, nAerLevels=1, nMonths=12)
    layout = RegistryLayout(registry_source(), config, _namelist(every, physics), dims)
    for name in ("qv", "qc", "qi", "qr", "qs", "qg"):
        assert int(layout.indices["state", "index_" + name]) >= 1, (scheme, name)
    assert int(layout.indices.get(("state", "index_ng"), -1)) < 1          # WRF's severe guard stays off


@pytest.mark.parametrize("option", ["radar", "vil", "vis"])
@pytest.mark.parametrize("scheme", ["mp_kessler", "off"])
def test_radar_vil_and_vis_are_refused_for_microphysics_without_the_hydrometeors(option, scheme):
    with pytest.raises(ValueError, match=f"config_afwa_{option}_opt needs the .* mp_wsm6, mp_thompson, "
                                         f"mp_thompson_aerosols carry them.*'{scheme}'"):
        _resolve({f"config_afwa_{option}_opt": ".true."}, (f" config_microp_scheme='{scheme}'",))
