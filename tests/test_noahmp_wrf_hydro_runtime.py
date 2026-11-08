"""Noah-MP's WRF_HYDRO exchange through the real driver (noahmp_lsm_step).

module_sf_noahmpdrv.F under -DWRF_HYDRO reads sfcheadrt (the ponded head
WRF-Hydro returns, mm) and writes INFXSRT = RUNSF, SOLDRAIN = MAX(RUNSB, 0)
and QTILEDRAIN = QTLDRN on every land column (:1211-1218).  The leaf
arithmetic is graded against WRF elsewhere (WATER:
tests/test_noahmp_water_wrf_hydro.py; glacier:
tests/test_noahmp_glacier_wrf_hydro.py); here the driver wiring is checked
on a small domain with land, water and a glacier column: a zero head leaves
the forecast word-identical and the exchange planes equal the runoff the
driver carries; a non-zero head moves land and glacier runoff; water columns
are never written; unported pieces are refused by name.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = pytest.mark.gpu

cp = pytest.importorskip("cupy")


def _selected_water_on_metal():
    from woof import backend
    return backend.active() == "mlx"


#: The selected-options water unit (noahmp_water_options.cuh), which carries
#: the WRF_HYDRO head, does not translate to Metal on main: woof's translator
#: refuses the goto into a block in its generated dynamic-VIC code ("goto
#: 'label_1003' from outside the label's block"), as it does for every
#: selected water option.  The default water unit is not used for WRF_HYDRO
#: so that its PTX, digests and recorded frames stay as they are.
METAL_SKIP = pytest.mark.skipif("_selected_water_on_metal()",
                                reason="selected Noah-MP water unit does not translate to Metal")


def _domain(**options):
    from test_noahmp_runtime import _GRASSLAND, _ICE, _build

    vegetation = np.full((4, 6), _GRASSLAND)
    vegetation[1, 2] = _ICE
    return _build(nx=6, ny=4, vegtyp=vegetation, **options)


def _planes(shape, head):
    sentinel = cp.float32(-7.0)
    return {"sfcheadrt": cp.full(shape, cp.float32(head), cp.float32),
            "infxsrt": cp.full(shape, sentinel, cp.float32),
            "soldrain": cp.full(shape, sentinel, cp.float32),
            "qtiledrain": cp.full(shape, sentinel, cp.float32)}


def _run(monkeypatch, planes, **options):
    import woof.core.physics as physics
    from woof.core.dycore import step

    real = physics.noahmp_lsm_step

    def hydro_step(*args, **kwargs):
        if planes is not None:
            kwargs["wrf_hydro"] = planes
        return real(*args, **kwargs)

    monkeypatch.setattr(physics, "noahmp_lsm_step", hydro_step)
    state, cfg, driver = _domain(**options)
    step(state, cfg)
    monkeypatch.setattr(physics, "noahmp_lsm_step", real)
    return driver


@requires_gpu
@METAL_SKIP
def test_a_zero_head_is_the_plain_forecast_and_fills_the_exchange(monkeypatch):
    plain = _run(monkeypatch, None)
    planes = _planes(plain.fields["tsk"].shape, 0.0)
    hydro = _run(monkeypatch, planes)
    for name in ("smois", "sh2o", "tslb", "sfcrunoff", "udrunoff", "tsk", "hfx", "qfx",
                 "snow", "runsfxy", "runsbxy"):
        a, b = plain.fields[name].get(), hydro.fields[name].get()
        assert (a.view(np.uint32) == b.view(np.uint32)).all(), name
    land = hydro.fields["xland"].get() < 1.5
    runsf = hydro.fields["runsfxy"].get()
    runsb = hydro.fields["runsbxy"].get()
    assert (planes["infxsrt"].get()[land] == runsf[land]).all()
    assert (planes["soldrain"].get()[land] == np.where(runsb > 0, runsb, 0)[land]).all()
    assert (planes["qtiledrain"].get()[land] == 0).all()
    assert (planes["infxsrt"].get()[~land] == -7.0).all()      # water columns untouched


@requires_gpu
@METAL_SKIP
def test_a_ponded_head_moves_land_and_glacier_runoff(monkeypatch):
    plain = _run(monkeypatch, None)
    planes = _planes(plain.fields["tsk"].shape, 6.0)
    hydro = _run(monkeypatch, planes)
    land = hydro.fields["xland"].get() < 1.5
    moved = hydro.fields["sfcrunoff"].get() != plain.fields["sfcrunoff"].get()
    moved |= hydro.fields["smois"].get()[0] != plain.fields["smois"].get()[0]
    assert moved[land].all()
    assert not moved[~land].any()
    # the glacier column: the head leaves as surface runoff within the step
    assert hydro.fields["runsfxy"].get()[1, 2] > plain.fields["runsfxy"].get()[1, 2]
    assert planes["infxsrt"].get()[1, 2] == hydro.fields["runsfxy"].get()[1, 2]


#: noahmp_output=3 routes the step through the resident device-surface
#: layout, whose glacier columns run noahmp_glacier_driver_resident.
RESIDENT = dict(noahmp_output=3, noahmp_acc_dt=0.4)


@requires_gpu
@METAL_SKIP
def test_the_resident_glacier_driver_carries_the_head_and_fills_the_exchange(monkeypatch):
    plain = _run(monkeypatch, None, **RESIDENT)
    zero = _planes(plain.fields["tsk"].shape, 0.0)
    hydro = _run(monkeypatch, zero, **RESIDENT)
    for name in ("smois", "sh2o", "tslb", "sfcrunoff", "udrunoff", "tsk", "hfx", "qfx",
                 "snow", "runsfxy", "runsbxy"):
        a, b = plain.fields[name].get(), hydro.fields[name].get()
        assert (a.view(np.uint32) == b.view(np.uint32)).all(), name
    runsf = hydro.fields["runsfxy"].get()[1, 2]
    runsb = hydro.fields["runsbxy"].get()[1, 2]
    assert zero["infxsrt"].get()[1, 2] == runsf
    assert zero["soldrain"].get()[1, 2] == (runsb if runsb > 0 else 0)
    assert zero["qtiledrain"].get()[1, 2] == 0

    ponded = _planes(plain.fields["tsk"].shape, 6.0)
    resident = _run(monkeypatch, ponded, **RESIDENT)
    assert resident.fields["runsfxy"].get()[1, 2] > runsf
    assert ponded["infxsrt"].get()[1, 2] == resident.fields["runsfxy"].get()[1, 2]
    # the host-marshalled glacier path with the same head agrees word for word
    host_planes = _planes(plain.fields["tsk"].shape, 6.0)
    host = _run(monkeypatch, host_planes)
    for name in ("runsfxy", "runsbxy", "snow", "tslb", "tsk"):
        a = resident.fields[name].get()[..., 1, 2]
        b = host.fields[name].get()[..., 1, 2]
        assert (np.asarray(a).view(np.uint32) == np.asarray(b).view(np.uint32)).all(), name
    for name in ("infxsrt", "soldrain", "qtiledrain"):
        assert ponded[name].get()[1, 2].view(np.uint32) == host_planes[name].get()[1, 2].view(np.uint32), name


def test_unported_wrf_hydro_pieces_are_refused_by_name():
    from woof.core.noahmp_runtime import _check_wrf_hydro

    fields = {"tsk": cp.zeros((2, 3), cp.float32)}
    with pytest.raises(NotImplementedError, match="ZWATBLE2D"):
        _check_wrf_hydro(fields, _planes((2, 3), 0.0), opt_tdrn=2)
    with pytest.raises(ValueError, match="exactly"):
        _check_wrf_hydro(fields, {"sfcheadrt": cp.zeros((2, 3), cp.float32)}, opt_tdrn=0)
