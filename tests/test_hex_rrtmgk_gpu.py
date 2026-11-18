"""WRF RRTMG-K (ra_lw_physics = ra_sw_physics = 14) on hex's generic column seam.

Hex's R1 route runs WRF's own ``PhysicsDriver`` on its cell columns through
``woof.core.mpas_column_suite.run_mpas_column_physics``; the 14/14 pair
reaches it through ``woof.core.radiation_composition.make_radiation`` like
every other radiation row.  These tests hold the seam to the WRF route:

* the seam builds ``woof.core.rrtmgk.RRTMGKRadiation`` (not another
  selector) and, at every radiation call, its result is bit for bit the one
  a fresh adapter built the way the WRF route builds it computes on the
  same atmosphere, fields and state -- the K columns themselves are held
  to compiled WRF 4.7.1 in tests/test_rrtmg_k_column_oracle.py;
* the columns are independent: on cells scattered over a spherical cap
  (the column set of a variable-resolution mesh, in no particular order),
  permuting the cells permutes every radiation output bit for bit, and a
  column chunk smaller than the cell count changes no bit;
* K is the scheme that ran: the seam's radiative heating rates differ from legacy
  RRTMG's on the same columns by more than rounding.

The cap sits near 56N at local midday, so every column is lit and the
shortwave gather/scatter carries real fluxes.

The one-hour forecast on a variable-resolution mesh is
tests/test_hex_varres_gpu.py with
``WOOF_HEX_VARRES_PHYSICS=mp_physics=6,ra_lw_physics=14,ra_sw_physics=14``.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import mpas_column_physics_cases as cases  # noqa: E402
from woof.core import mpas_column_suite as suite  # noqa: E402
from woof.core.rrtmgk import RRTMGKRadiation  # noqa: E402

NZ, DT = cases.NZ, cases.DT
NCOL = 24
#: Local midday on the cap (3W): every column lit, coszen ~0.5.
START = cases.START.replace(hour=12)


def _physics():
    physics = dict(cases.selections()["radiation.wrf-rrtmg-k"])
    assert (physics["ra_lw_physics"], physics["ra_sw_physics"]) == (14, 14)
    return physics


def _cap(n, radius_m=900.0e3, lat0=0.98, lon0=-0.06):
    """``n`` points spread over a spherical cap, in degrees."""
    z = 1 - (1 - math.cos(radius_m / 6371.0e3)) * (np.arange(n) + 0.5) / n
    phi = np.arange(n) * (math.pi * (3 - math.sqrt(5)))
    r = np.sqrt(1 - z * z)
    local = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)
    t, p = math.pi / 2 - lat0, lon0
    ry = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0],
                   [-math.sin(t), 0, math.cos(t)]])
    rz = np.array([[math.cos(p), -math.sin(p), 0],
                   [math.sin(p), math.cos(p), 0], [0, 0, 1]])
    xyz = local @ (rz @ ry).T
    return (np.degrees(np.arcsin(np.clip(xyz[:, 2], -1, 1))),
            np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])))


def _build(physics, order=None, column_chunk=None):
    """The generic seam on cap cells, the cells taken in ``order``."""
    order = np.arange(NCOL) if order is None else np.asarray(order)
    plan = suite.column_physics_plan(
        physics, n_levels=NZ, dt=DT, radiation_seconds=120.0,
        surface_pbl_seconds=DT)
    lat, lon = _cap(NCOL)
    terrain = np.random.default_rng(2).uniform(0.0, 50.0, NCOL)
    area = np.linspace(9.0e6, 4.0e8, NCOL)       # 3 km to 20 km cells
    surf = {key: (value[..., order] if isinstance(value, np.ndarray) else value)
            for key, value in cases.surface(plan, NCOL).items()}
    seam = suite.run_mpas_column_physics(
        physics=physics, n_levels=NZ, n_columns=NCOL, dt=DT,
        radiation_seconds=120.0, surface_pbl_seconds=DT,
        start_time=START, latitude_deg=lat[order],
        longitude_deg=lon[order],
        terrain_height_m=terrain[order].astype(np.float32),
        z_interface_nominal_m=np.linspace(0.0, cases.ZTOP, NZ + 1),
        p_top_pa=cases.P_TOP, dx_m=3000.0,
        cell_area_m2=area[order], surface=surf,
        statics={k: v[order] for k, v in cases.statics(plan, NCOL).items()})
    if column_chunk is not None:
        seam._driver.radiation_callable.column_chunk = int(column_chunk)
    return seam


def _spy(monkeypatch, seam, also=None):
    """Record every radiation result the seam's driver receives.

    The class's ``__call__`` is wrapped, so the driver keeps the adapter
    object itself (its ozone carrier, diagnostics and attributes) and runs
    the production path; only calls on this seam's adapter are recorded.
    """
    calls = []
    target = seam._driver.radiation_callable
    cls = type(target)
    inner = cls.__call__

    def spy(self, **kwargs):
        if self is not target:
            return inner(self, **kwargs)
        reference = also(**kwargs) if also is not None else None
        result = inner(self, **kwargs)
        calls.append((_host(result), None if reference is None
                      else _host(reference)))
        return result
    monkeypatch.setattr(cls, "__call__", spy)
    return calls


def _leaf(value):
    if isinstance(value, cp.ndarray):
        return cp.asnumpy(value).copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, (bool, int, float, str, np.generic)):
        return np.asarray(value)
    return None


def _host(result):
    """Every array or scalar of a ``RadiationResult``, on the host."""
    out = {}
    for field in dataclasses.fields(result):
        value = getattr(result, field.name)
        items = value.items() if isinstance(value, dict) else [(None, value)]
        for key, item in items:
            leaf = _leaf(item)
            if leaf is not None:
                out[field.name if key is None else f"{field.name}.{key}"] = leaf
            elif item is not None:
                raise TypeError(f"unrecorded {field.name}.{key}: {type(item)}")
    return out


def _run(seam, order=None, steps=3):
    order = np.arange(NCOL) if order is None else np.asarray(order)
    fields, species = cases.profile(cp, seam.plan, NCOL)
    fields = {k: cp.ascontiguousarray(v[:, order]) for k, v in fields.items()}
    species = {k: cp.ascontiguousarray(v[:, order]) for k, v in species.items()}
    theta = fields["theta"].copy()
    dtheta = []
    for _ in range(steps):
        result = seam.run_phase1(
            dt=DT, u=fields["u"], v=fields["v"], theta=theta,
            pressure=fields["pressure"],
            pressure_interface=fields["pressure_interface"],
            z_interface=fields["z_interface"], w=fields["w"],
            rho_dry=fields["rho_dry"], species=species)
        dtheta.append(cp.asnumpy(result.dtheta).copy())
        seam.run_phase2(
            theta=theta, pressure=fields["pressure"],
            z_interface=fields["z_interface"], w=fields["w"],
            rho_dry=fields["rho_dry"], species=species)
    return dtheta


def test_the_seam_radiates_through_the_wrf_route_adapter(monkeypatch):
    seam = _build(_physics())
    adapter = seam._driver.radiation_callable
    assert type(adapter) is RRTMGKRadiation
    # make_radiation's arguments, from the run's config and the seam's
    # inputs, not read back off the adapter under test.
    cfg = seam._cfg
    assert (cfg.o3input, cfg.icloud) == (2, 1)    # WRF's defaults
    lat, lon = _cap(NCOL)
    fresh = RRTMGKRadiation(
        START, lat.reshape(1, -1), lon.reshape(1, -1),
        o3input=cfg.o3input, icloud=cfg.icloud, column_chunk=None,
        p_top=cases.P_TOP)
    calls = _spy(monkeypatch, seam, also=fresh)
    dtheta = _run(seam)
    assert calls, "the seam never radiated"
    assert all(np.isfinite(d).all() for d in dtheta)
    for step, (got, want) in enumerate(calls):
        assert set(got) == set(want)
        for name in got:
            np.testing.assert_array_equal(got[name], want[name],
                                          f"call {step} {name}")
        assert np.isfinite(got["olr"]).all() and (got["olr"] > 50).all()
        assert np.isfinite(got["glw"]).all() and (got["glw"] > 50).all()
        assert (got["coszen"] > 0.3).all() and (got["swdown"] > 50).all()


def test_columns_are_independent_of_cell_order_and_chunking(monkeypatch):
    physics = _physics()
    base = _build(physics)
    base_calls = _spy(monkeypatch, base)
    _run(base)
    assert base_calls, "the seam never radiated"
    order = np.random.default_rng(7).permutation(NCOL)
    for label, seam, cells in (
            ("permuted", _build(physics, order=order), order),
            ("chunked", _build(physics, column_chunk=7), None)):
        with monkeypatch.context() as patch:
            calls = _spy(patch, seam)
            _run(seam, order=cells)
        idx = np.arange(NCOL) if cells is None else cells
        assert len(calls) == len(base_calls), label
        for step, ((got, _), (want, _)) in enumerate(zip(calls, base_calls)):
            assert set(got) == set(want), label
            for name in want:
                np.testing.assert_array_equal(
                    got[name], want[name][..., idx],
                    f"{label} call {step} {name}")


def test_rrtmgk_heating_differs_from_legacy_rrtmg(monkeypatch):
    heating = {}
    for label, physics in (
            ("k", _physics()),
            ("legacy", cases.selections()["radiation.rrtmg-legacy"])):
        seam = _build(physics)
        with monkeypatch.context() as patch:
            calls = _spy(patch, seam)
            _run(seam, steps=1)
        assert calls, label
        heating[label] = calls[0][0]
    for name in ("rthratenlw", "rthratensw"):
        k, legacy = heating["k"][name], heating["legacy"][name]
        scale = float(np.abs(legacy).max())
        assert scale > 0, name
        # More than rounding: K's gas optics and cloud path, not ULPs.
        assert float(np.abs(k - legacy).max()) > 1e-3 * scale, name


def test_ghg_input_reaches_the_seam_and_takes_the_camtr_gases(monkeypatch):
    """ghg_input = 1 on the seam (beyond WRF, woof.core.rrtmgk): the adapter
    radiates the CAMtr gases at each call's valid time, bit for bit the
    WRF-route adapter built with ghg_input = 1; unset, no gases are read."""
    from woof.core import rrtmgk
    from woof.core.camtr_gases import camtr_trace_gases
    reads = []
    inner = rrtmgk.camtr_gases

    def camtr(yr, julian):
        gases = inner(yr, julian)
        reads.append((yr, julian, gases))
        return gases
    monkeypatch.setattr(rrtmgk, "camtr_gases", camtr)

    physics = dict(_physics(), ghg_input=1)
    seam = _build(physics)
    adapter = seam._driver.radiation_callable
    assert seam._cfg.ghg_input == 1 and adapter.ghg_input == 1
    lat, lon = _cap(NCOL)
    fresh = RRTMGKRadiation(
        START, lat.reshape(1, -1), lon.reshape(1, -1), o3input=2, icloud=1,
        column_chunk=None, p_top=cases.P_TOP, ghg_input=1)
    calls = _spy(monkeypatch, seam, also=fresh)
    _run(seam, steps=1)
    assert calls and reads
    for yr, julian, gases in reads:
        assert yr == START.year
        table = camtr_trace_gases(yr, julian)
        assert gases == {k: np.float32(table[k]) for k in rrtmgk.GHG_INPUT_GASES}
    k_ghg = calls[0][0]
    for name in k_ghg:
        np.testing.assert_array_equal(k_ghg[name], calls[0][1][name], name)

    reads.clear()
    base = _build(_physics())
    assert base._driver.radiation_callable.ghg_input == 0
    with monkeypatch.context() as patch:
        base_calls = _spy(patch, base)
        _run(base, steps=1)
    assert base_calls and not reads
    assert not np.array_equal(k_ghg["rthratenlw"], base_calls[0][0]["rthratenlw"])
