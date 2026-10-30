"""WRF-Solar diagnostics on the MPAS suite route (woof.hex.mpasphys.solar_diagnostics_v842).

&diagnostics config_solar_diagnostics = .true. adds WRF v4.7.1 module_diag_solar.F's
26 fields to diag_physics and the diagnostics stream, computed after every MPAS step
by the routine held to compiled WRF (tests/test_solar_diag_wrf471_oracle.py, whose
MPAS-shaped columns are formed exactly as this adapter forms them).  Here:

* the option resolves, is refused where its inputs do not exist (no GPU RRTMG
  shortwave, the RRFS fork route), and reaches the &diagnostics group from the door;
* the Registry pools and default diagnostics stream carry the 26 fields only with it;
* a checkpoint leaves them out (WRF gives them no restart I/O);
* the adapter's step, on the oracle's MPAS-shaped columns behind stand-in MPAS pools,
  writes every field word WRF writes, and GHI_ACCUM keeps summing.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys import resolve_namelist
from woof.hex.mpasphys import solar_diagnostics_v842 as solar

ORACLE = Path(__file__).resolve().parents[3] / "tests" / "data" / "oracles" / "solar_diag"


def _namelist(on=True, **physics):
    lines = ["&physics", " config_physics_suite='convection_permitting'"]
    lines += [f" {k}={v!r}" for k, v in physics.items()]
    lines += ["/"]
    if on is not None:
        lines += ["&diagnostics", f" config_solar_diagnostics={'.true.' if on else '.false.'}", "/"]
    return "\n".join(lines) + "\n"


def test_option_defaults_off_and_resolves_on_the_gpu_route():
    assert not resolve_namelist(_namelist(None), default_backend="hex-cuda").solar_diagnostics
    assert not resolve_namelist(_namelist(False), default_backend="hex-cuda").solar_diagnostics
    config = resolve_namelist(_namelist(True), default_backend="hex-cuda")
    assert config.solar_diagnostics and config.receipt()["solar_diagnostics"] is True
    assert solar.OPTION not in config.options
    assert solar.packages(config) == {solar.PACKAGE}


@pytest.mark.parametrize("text,match", [
    (_namelist(None) + "&diagnostics\n config_solar_diags=1\n/\n", "has no option"),
    (_namelist(True, config_radt_sw_scheme="off"), "needs the RRTMG shortwave"),
    (_namelist(True) + "&hex_mpasphys\n config_radt_sw_backend='native-cpu'\n/\n", "on hex-cuda"),
])
def test_option_is_refused_where_its_inputs_do_not_exist(text, match):
    with pytest.raises(ValueError, match=match):
        resolve_namelist(text, default_backend="hex-cuda")


def test_rrfs_fork_route_refuses_the_option():
    from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist as resolve_rrfs
    text = "&physics\n config_physics_suite='hrrrv5'\n/\n&diagnostics\n config_solar_diagnostics=.true.\n/\n"
    with pytest.raises(ValueError, match="not available on the RRFS fork route"):
        resolve_rrfs(text)


def test_door_override_lands_in_the_diagnostics_group():
    from woof.hex.mpasphys.forecast_v842 import native_route
    arguments = SimpleNamespace(physics_namelist=None, physics_suite="convection_permitting",
                                physics=["config_solar_diagnostics=true"])
    config, text = native_route(arguments)
    assert config.solar_diagnostics
    assert "&diagnostics" in text


def _layout(on):
    from woof.hex.mpasphys.framework_backend_v842 import registry_source
    from woof.hex.mpasphys.pool_v842 import RegistryLayout
    text = _namelist(on)
    config = resolve_namelist(text, default_backend="hex-cuda")
    dims = dict(nCells=10, nEdges=30, nVertices=20, nVertLevels=12, maxEdges=6, maxEdges2=12,
                vertexDegree=3, nSoilLevels=4, nOznLevels=59, nAerLevels=1, nMonths=12)
    return RegistryLayout(registry_source(), config, text, dims)


def test_pools_streams_and_checkpoints(tmp_path):
    from woof.hex.mpasphys.forecast_v842 import read_streams, write_default_streams
    from woof.hex.mpasphys.checkpoint_v842 import CheckpointRoute
    from woof.hex.mpasphys.moving_v842 import carried_rows
    from woof.hex.streams_v842 import field_definition, _all_packages
    off, on = _layout(False), _layout(True)
    rows = lambda layout: {r["name"]: r for r in layout.fields if r["pool"] == "diag_physics"}
    assert not set(solar.FIELDS) & set(rows(off))
    assert set(solar.FIELDS) <= set(rows(on))
    assert [r for r in on.fields if r["name"] not in solar.FIELDS] == list(off.fields)
    for name in solar.FIELDS:
        assert rows(on)[name]["dimensions"] == ["nCells"] and rows(on)[name]["shape"] == [11]
    write_default_streams(on, tmp_path / "streams.atmosphere")
    streams = read_streams(tmp_path / "streams.atmosphere", on)
    diagnostics = next(s for s in streams if s.name == "diagnostics")
    assert set(solar.FIELDS) <= set(diagnostics.fields)
    d = field_definition("ghi_accum")
    assert d["attrs"]["units"] == "J m-2" and d["attrs"]["long_name"] == "ACCUMULATED GHI"
    assert solar.PACKAGE in _all_packages()
    # a move carries them (diag rows with Time); a checkpoint does not
    moved = {r["name"] for r in carried_rows(on)}
    assert set(solar.FIELDS) <= moved
    route = CheckpointRoute.__new__(CheckpointRoute)
    kept = {r["name"] for r in CheckpointRoute.carried_rows(route, on)}
    assert not set(solar.FIELDS) & kept


# ---------------------------------------------------------------------------
# the step, on the oracle's MPAS-shaped columns behind stand-in pools
# ---------------------------------------------------------------------------
def _oracle_mpas_columns():
    import sys
    sys.path.insert(0, str(ORACLE.parents[2]))
    from test_solar_diag_wrf471_oracle import _read
    return [c for c in _read() if c["meta"]["mpas"]]


def test_step_writes_wrfs_words_on_mpas_fields(monkeypatch):
    cp = pytest.importorskip("cupy")
    from woof.core.solar_diagnostics import NOUT, species_flags
    from woof.hex import cuda_physics_prep_v841 as prep
    cols = _oracle_mpas_columns()
    # one group the adapter can run as one mesh: same nz, all species, radii, one dt
    key = lambda c: (c["nz"], c["flags"], float(c["dt"]))
    groups = {}
    for c in cols:
        groups.setdefault(key(c), []).append(c)
    sub = max((g for k, g in groups.items() if k[1] == (1, 1, 1, 1, 1, 1, 1)), key=len)
    nz, nc = sub[0]["nz"], len(sub)
    stack = lambda j: np.ascontiguousarray(np.stack([c["arrays"][j] for c in sub], axis=1))
    rho, dz, qv, qc, qi, qs, qc_tot, qi_tot, rec, rei, res, cf = (stack(j) for j in range(12))
    zgrid = np.ascontiguousarray(np.stack([c["zgrid"] for c in sub], axis=1))
    surf = lambda k: np.array([c[k] for c in sub], np.float32)
    D = lambda a: cp.asarray(a)
    fields = {("mesh", "zgrid"): D(zgrid), ("diag_physics", "cldfrac"): D(cf),
              ("diag_physics", "coszr"): D(surf("coszen")), ("diag_physics", "swdnb"): D(surf("swdnb")),
              ("diag_physics", "swdnt"): D(surf("swdnt")), ("diag_physics", "re_cloud"): D(rec),
              ("diag_physics", "re_ice"): D(rei), ("diag_physics", "re_snow"): D(res)}
    for name in solar.NAMES:
        fields["diag_physics", name] = cp.zeros(nc, cp.float32)
    prepared = {"qcrad_p": D(qc_tot), "qirad_p": D(qi_tot)}
    session = SimpleNamespace(
        layout=SimpleNamespace(dimensions={"nVertLevels": nz, "nCellsSolve": nc}),
        config=SimpleNamespace(execution_options={"config_microp_re": True,
                                                  "config_microp_scheme": "mp_thompson"}),
        field=lambda pool, name: fields[pool, name],
        device_pools=SimpleNamespace(prepared=lambda name: prepared[name]))
    names = ("qv", "qc", "qr", "qi", "qs", "qg")
    scalars = cp.zeros((len(names), nz, nc), cp.float32)
    for row, value in zip((0, 1, 3, 4), (qv, qc, qi, qs)):
        scalars[row] = D(value)
    coupling = SimpleNamespace(names=names, geometry=None, physics_cache=None)
    dycore = SimpleNamespace(cp=cp, atmosphere=SimpleNamespace(state=SimpleNamespace(scalars=scalars)))
    monkeypatch.setattr(prep, "prepare_mpas_to_phys_cuda_v841",
                        lambda *a, **k: SimpleNamespace(rho_p=D(rho), dz_p=D(dz)))
    diag = solar.MpasSolarDiagnostics(session, coupling, dycore)
    diag.out[2] = D(surf("prior"))                     # the oracle's GHI_ACCUM prior
    diag.out[12:22] = D(np.broadcast_to(surf("prior"), (10, nc)))
    diag.step(dt=float(sub[0]["dt"]))
    want = np.stack([c["want"] for c in sub])
    for i, name in enumerate(solar.NAMES):
        got = cp.asnumpy(fields["diag_physics", name])
        np.testing.assert_array_equal(got.view(np.uint32), want[:, i].view(np.uint32), err_msg=name)
    # the accumulator keeps summing on the next step
    before = cp.asnumpy(fields["diag_physics", "ghi_accum"]).copy()
    diag.step(dt=float(sub[0]["dt"]))
    after = cp.asnumpy(fields["diag_physics", "ghi_accum"])
    np.testing.assert_array_equal(
        after, (before + (surf("swdnb") * np.float32(sub[0]["dt"])).astype(np.float32)).astype(np.float32))
