"""The Fitch drag on columns of unequal area (``fitch_column_cells``).

hex's Voronoi columns each have their own ``areaCell``; WRF's Fitch scheme
divides the rotor thrust by the cell volume ``dx*dx*dz``.  The cells kernel
is WRF's ``fitch_column`` with each column's area in ``dx*dx``'s place
(``wfdensity = float32(1)/float32(area)``).  Held here:

* WRF v4.7.1's own Fortran (``tools/windfarm_fitch_wrf471_oracle``) on its
  eleven columns, with every column's area ``float32(dx)*float32(dx)``: the
  cells kernel and its CPU twin reproduce every word of DU, DV, QKE and
  POWER (max ULP 0).
* The WRF-grid kernel, column by column, on random columns of unequal
  area: each column equals ``fitch_column`` run with that column's ``dx``
  (area = dx*dx), word for word, the diagnostics included.
* GPU against the binary32 CPU twin on the same random columns: max ULP 0.

NO MPAS ORACLE EXISTS: MPAS-A has no wind-farm scheme, so the hex variant
is held to WRF's Fortran through the dx*dx identity and to its CPU twin, and
nothing else.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import requires_gpu
from tests.test_windfarm_fitch_wrf471_parity import (CASES, FIELDS,
                                                     oracle_case, words)

F = np.float32
ALL = FIELDS + ("thrust", "overlap")


def _flat(o, area):
    """The oracle case on one row of columns, each with its own area."""
    nz, ny, nx = o["u"].shape
    assert ny == 1
    return dict(o, cell_area=np.asarray(area, dtype=F).reshape(nx))


def run_cells_gpu(o, area):
    import cupy as cp
    from woof.core.windfarm_fitch import launch_fitch

    du, dv, qke = (cp.asarray(o[f"{n}_in"]) for n in ("du", "dv", "qke"))
    diagnostics = {}
    power = launch_fitch(cp.asarray(o["u"]), cp.asarray(o["v"]),
                         cp.asarray(o["z_w"]), o["layout"], du, dv,
                         cell_area=cp.asarray(area), qke=qke, dt=o["dt"],
                         tke_factor=o["factor"], rho=cp.asarray(o["rho"]),
                         diagnostics=diagnostics)
    return {"du": cp.asnumpy(du), "dv": cp.asnumpy(dv), "qke": cp.asnumpy(qke),
            "power": cp.asnumpy(power),
            "thrust": cp.asnumpy(diagnostics["thrust"]),
            "overlap": cp.asnumpy(diagnostics["overlap"])}


def run_cells_cpu(o, area):
    from woof.verify.windfarm_fitch_ref import f32_fitch_column

    du, dv, qke = (o[f"{n}_in"].copy() for n in ("du", "dv", "qke"))
    power, thrust, overlap = f32_fitch_column(
        o["u"], o["v"], o["z_w"], o["layout"], du, dv, qke, cell_area=area,
        dt=o["dt"], tke_factor=o["factor"], rho=o["rho"])
    return {"du": du, "dv": dv, "qke": qke, "power": power, "thrust": thrust,
            "overlap": overlap}


def run_wrf_gpu(o, dx):
    from tests.test_windfarm_fitch_wrf471_parity import run_gpu
    return run_gpu(dict(o, dx=float(dx)))


def random_case(seed=7, nx=24, nz=40):
    """Columns with several turbines each, rotors over many levels, winds
    across cut-in, rated and cut-out, and unequal areas."""
    from woof.core.windfarm_fitch import FitchLayout

    rng = np.random.default_rng(seed)
    dz = rng.uniform(15.0, 60.0, size=(nz, 1, nx)).astype(F)
    z_w = np.concatenate([rng.uniform(0, 300, (1, 1, nx)).astype(F),
                          np.zeros((nz, 1, nx), F)])
    for k in range(nz):
        z_w[k + 1] = z_w[k] + dz[k]
    u = rng.normal(9.0, 7.0, (nz, 1, nx)).astype(F)
    v = rng.normal(-2.0, 6.0, (nz, 1, nx)).astype(F)
    speeds = np.arange(3.0, 25.5, 1.0).astype(F)
    ct = np.clip(0.9 - 0.035 * (speeds - 3.0), 0.05, None).astype(F)
    pw = np.clip(3450.0 * ((speeds - 3.0) / 9.0) ** 3, 0, 3450.0).astype(F)
    cells, counts, hubs, dias, curves = [], [], [], [], []
    for c in range(nx):
        if c % 5 == 4:
            continue            # an empty column
        for g in range(1 + c % 3):
            cells.append(c)
            counts.append(1 + (c + g) % 3)
            dia = F(rng.choice([117.0, 136.0, 90.0]))
            dias.append(dia)
            hubs.append(F(dia / 2 + rng.uniform(5.0, 80.0)))
            curves.append(g % 2)
    layout = FitchLayout(
        ny=1, nx=nx, cell=cells, count=counts, curve=curves,
        hub_height_m=hubs, rotor_diameter_m=dias,
        speeds_ms=np.stack([speeds, speeds + F(0.5)]),
        thrust_coefficient=np.stack([ct, ct * F(0.97)]),
        power_kw=np.stack([pw, pw * F(0.9)]),
        nval=[speeds.size, speeds.size], standing_thrust=[0.0, 0.05])
    dx = rng.uniform(800.0, 4000.0, size=nx).astype(F)
    o = {"layout": layout, "z_w": z_w, "u": u, "v": v, "dt": 4.0,
         "factor": 0.25, "rho": np.full_like(u, F(1.17)),
         "du_in": rng.normal(0, 1e-3, u.shape).astype(F),
         "dv_in": rng.normal(0, 1e-3, u.shape).astype(F),
         "qke_in": rng.uniform(0.0, 2.0, u.shape).astype(F)}
    return o, dx


@pytest.fixture(scope="module", params=CASES)
def case(request):
    return request.param, oracle_case(request.param)


# ---------------------------------------------------------------------------
# no device


def test_cell_wfdensity_is_wrf_word_when_the_area_is_dx_squared():
    from woof.core.windfarm_fitch import cell_wfdensity

    for dx in (3000.0, 937.5, 1234.567, 27000.0):
        area = F(dx) * F(dx)
        assert cell_wfdensity(np.array([area]))[0].view(np.uint32) == \
            F(F(1.0) / (F(dx) * F(dx))).view(np.uint32)
    with pytest.raises(ValueError):
        cell_wfdensity(np.array([0.0]))


def test_the_cpu_twin_reproduces_wrf_with_area_as_dx_squared(case):
    name, o = case
    nx = o["u"].shape[2]
    area = np.full(nx, F(o["dx"]) * F(o["dx"]), F)
    cpu = run_cells_cpu(o, area)
    for field in FIELDS:
        table = words(cpu[field], o[field])
        assert table["max_ulp"] == 0, (name, field, table)


def test_the_cpu_twin_is_the_wrf_grid_mirror_column_by_column():
    """Unequal areas: each column is the WRF-grid mirror at its own dx."""
    from tests.test_windfarm_fitch_wrf471_parity import run_cpu

    o, dx = random_case()
    area = dx * dx
    cells = run_cells_cpu(o, area)
    for c in range(dx.size):
        wrf = run_cpu(dict(o, dx=float(dx[c])))
        for field in ALL:
            got = cells[field][..., c]
            want = wrf[field][..., c]
            assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), (
                c, field)


def test_the_cells_unit_is_fitch_column_with_one_line_moved():
    """The cells kernel's body is fitch_column's, statement for statement:
    only where wfdensity comes from and the column count differ."""
    from pathlib import Path
    import re

    kdir = Path(__file__).resolve().parents[1] / "woof" / "core" / "kernels"

    def body(text, name):
        start = text.index(f"void {name}(")
        start = text.index("{", text.index(")", start))
        return text[start:]

    def norm(text):
        return [line.strip() for line in text.splitlines()
                if line.strip() and not line.strip().startswith("//")]

    wrf = norm(body((kdir / "windfarm_fitch.cu").read_text(), "fitch_column"))
    cells = norm(body((kdir / "windfarm_fitch_cells.cu").read_text(),
                      "fitch_column_cells"))
    extra = [line for line in cells if line not in wrf]
    assert set(extra) <= {
        "size_t ncol = (size_t)ncol_in;",
        "if (offsets[col] == offsets[col + 1]) {",
        "power[col] = pw;", "if (thrust) thrust[col] = th;", "return;", "}",
        "real wfdensity = wfdensity_col[col];"}, extra
    missing = [line for line in wrf if line not in cells]
    assert all(re.search(r"ncol = \(size_t\)ny \* nx", line)
               for line in missing), missing


# ---------------------------------------------------------------------------
# the device


@requires_gpu
@pytest.mark.gpu
def test_the_cells_kernel_reproduces_wrf_with_area_as_dx_squared(case):
    name, o = case
    nx = o["u"].shape[2]
    area = np.full(nx, F(o["dx"]) * F(o["dx"]), F)
    gpu = run_cells_gpu(o, area)
    for field in FIELDS:
        table = words(gpu[field], o[field])
        assert table["max_ulp"] == 0, (name, field, table)
    wrf = run_wrf_gpu(o, o["dx"])
    for field in ALL:
        assert words(gpu[field], wrf[field])["max_ulp"] == 0, (name, field)


@requires_gpu
@pytest.mark.gpu
def test_cells_gpu_and_cpu_agree_on_every_word_with_unequal_areas():
    o, dx = random_case()
    area = dx * dx * F(1.0001)    # areas that are no float32 square
    gpu, cpu = run_cells_gpu(o, area), run_cells_cpu(o, area)
    for field in ALL:
        table = words(gpu[field], cpu[field])
        assert table["max_ulp"] == 0, (field, table)
    assert np.count_nonzero(gpu["power"]) > 5
    assert np.any(gpu["du"] != o["du_in"])


@requires_gpu
@pytest.mark.gpu
def test_each_column_is_the_wrf_kernel_at_its_own_dx():
    o, dx = random_case(seed=11)
    gpu = run_cells_gpu(o, dx * dx)
    for c in range(dx.size):
        wrf = run_wrf_gpu(o, dx[c])
        for field in ALL:
            got, want = gpu[field][..., c], wrf[field][..., c]
            assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), (
                c, field)


@requires_gpu
@pytest.mark.gpu
def test_the_dx_path_is_unchanged_and_the_two_are_exclusive():
    import cupy as cp
    from woof.core.windfarm_fitch import launch_fitch

    o, dx = random_case(seed=3)
    z = cp.zeros_like(cp.asarray(o["u"]))
    with pytest.raises(ValueError, match="not both"):
        launch_fitch(z, z, cp.asarray(o["z_w"]), o["layout"], z.copy(),
                     z.copy(), dx=1000.0, cell_area=cp.asarray(dx * dx))
    with pytest.raises(ValueError, match="dx is required"):
        launch_fitch(z, z, cp.asarray(o["z_w"]), o["layout"], z.copy(),
                     z.copy())


@requires_gpu
@pytest.mark.gpu
def test_the_cells_unit_compiles_within_the_priced_frame():
    import cupy as cp

    from woof.core import windfarm_fitch as wf
    from woof.core.preflight import CHAINED_TRANSLATION_UNIT_FRAMES

    module = wf._cells_module(cp.cuda.Device().id)
    frame = int(cp.cuda.driver.funcGetAttribute(
        cp.cuda.driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        module.get_function("fitch_column_cells").ptr))
    priced = CHAINED_TRANSLATION_UNIT_FRAMES["windfarm_fitch_cells_composed"]
    assert frame <= priced.max_local_size_bytes
