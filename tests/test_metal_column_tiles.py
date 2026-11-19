"""The Metal column tile (physics_inventory.metal_column_tile_columns) moves
no word in MRF, New SAS (the column_cumulus table route) or the registry
column-kernel cumulus route (BMJ), and each launcher's tile rule.  EEPS
takes the same Metal tile (2026-11-19: one launch measured 11.7 vs 14.4 ms
at 49 x 228 x 276 on an M3 Max, interleaved; its words are tile-invariant
by tests/test_eeps_wrf461_parity.py's split and partial-tile tests).

On Metal those launchers put a whole domain in one launch under a byte cap
(YSU's #602 rule); CUDA keeps its occupancy-bounded tiles.  The tile only
partitions independent columns: one block per launch, a few blocks, the
backend's own rule and the whole domain give the same words, with every
float workspace poisoned with NaN, under fast, strict and WRF-exact
numerics (tests/_numerics_child; nothing imports a launcher at module
scope, since that import fixes the numerics mode).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import _numerics_child

NZ, NY, NX = 31, 21, 23           # 483 columns: partial last tiles


def _poisoned_empty(cp):
    real = cp.empty

    def empty(shape, dtype=float, *a, **kw):
        arr = real(shape, dtype, *a, **kw)
        if np.dtype(dtype).kind == "f":
            arr.fill(np.nan)
        return arr
    return real, empty


def _digest(out):
    import cupy as cp
    h = hashlib.sha256()
    for name in sorted(out):
        if out[name] is not None:
            h.update(name.encode())
            h.update(cp.asnumpy(out[name]).tobytes())
    return h.hexdigest()


def _profile(rng):
    nz, ny, nx = NZ, NY, NX
    psf = 1.0e5 + rng.uniform(-2e3, 2e3, (ny, nx))
    pif = np.empty((nz + 1, ny, nx))
    for k in range(nz + 1):
        pif[k] = psf * (1.0 - 0.9 * k / nz)
    p = 0.5 * (pif[:-1] + pif[1:])
    pi = (p / 1.0e5) ** 0.2857
    z = np.empty((nz, ny, nx))
    t = np.empty((nz, ny, nx))
    for k in range(nz):
        z[k] = 40.0 + 420.0 * k
        t[k] = 300.0 - 6.5e-3 * z[k] + rng.uniform(-1.5, 1.5, (ny, nx))
    qv = np.clip(0.016 * np.exp(-z / 2500.0) * rng.uniform(0.7, 1.0, (nz, ny, nx)),
                 1e-6, None)
    rho = p / (287.0 * t * (1.0 + 0.61 * qv))
    return dict(psfc=psf, pif=pif, p=p, pi=pi, z=z, t=t, qv=qv, rho=rho,
                dz=np.full((nz, ny, nx), 420.0))


def _f32(cp, a):
    return cp.asarray(np.ascontiguousarray(a, dtype=np.float32))


def _runs(cp, launch, rules, patch):
    """Digest ``launch()`` under each tile rule, workspaces poisoned."""
    real, poisoned = _poisoned_empty(cp)
    runs = {}
    for name, rule in rules.items():
        cp.empty = poisoned
        undo = patch(rule)
        try:
            runs[name] = launch()
        finally:
            cp.empty = real
            undo()
    return runs


def _assert_same(runs):
    whole = runs["whole domain"]
    for name, digest in runs.items():
        assert digest == whole, f"the {name} tile changed the result"


@pytest.mark.gpu
def test_mrf_tile_keeps_every_word():
    cp = pytest.importorskip("cupy")
    from woof.core import mrf as M

    rng = np.random.default_rng(7)
    a = _profile(rng)
    col = {k: _f32(cp, v) for k, v in dict(
        u=rng.uniform(-10, 10, (NZ, NY, NX)), v=rng.uniform(-10, 10, (NZ, NY, NX)),
        t=a["t"], qv=a["qv"], qc=rng.uniform(0, 3e-5, (NZ, NY, NX)),
        qi=rng.uniform(0, 2e-5, (NZ, NY, NX)), p=a["p"], pi=a["pi"],
        dz8w=a["dz"], z=a["z"]).items()}
    surf = {k: _f32(cp, v) for k, v in dict(
        psfc=a["psfc"], psim=rng.uniform(0.4, 4, (NY, NX)),
        psih=rng.uniform(0.4, 4, (NY, NX)), xland=rng.uniform(1, 2, (NY, NX)),
        hfx=rng.uniform(-30, 250, (NY, NX)), qfx=rng.uniform(-1e-5, 2e-4, (NY, NX)),
        gz1oz0=rng.uniform(3, 9, (NY, NX)), wspd=rng.uniform(0.7, 15, (NY, NX)),
        br=rng.uniform(-2, 0.6, (NY, NX))).items()}
    ust = rng.uniform(0.05, 0.85, (NY, NX))
    from woof.core.physics_inventory import metal_column_tile_columns
    original = M.mrf_tile_columns
    rules = {"one block": lambda fn, n, nz=None: M.MRF_BLOCK,
             "three blocks": lambda fn, n, nz=None: 3 * M.MRF_BLOCK,
             "metal rule, one core": lambda fn, n, nz=None: metal_column_tile_columns(
                 1, n, M.MRF_BLOCK, block_bytes=M.mrf_workspace_floats(nz, M.MRF_BLOCK) * 4),
             "backend rule": original,
             "whole domain": lambda fn, n, nz=None: n}

    def patch(rule):
        M.mrf_tile_columns = rule
        return lambda: setattr(M, "mrf_tile_columns", original)

    def launch():
        out = M.launch_mrf(**col, **surf, ust=_f32(cp, ust), dt=30.0, flag_qi=True)
        assert np.isfinite(cp.asnumpy(out["rthblten"])).all()
        return _digest(out)

    _assert_same(_runs(cp, launch, rules, patch))


@pytest.mark.gpu
def test_new_sas_tile_keeps_every_word():
    cp = pytest.importorskip("cupy")
    from woof.core import column_cumulus as C

    scheme = C.COLUMN_CUMULUS_SCHEMES[96]
    rng = np.random.default_rng(13)
    a = _profile(rng)
    ncol = NY * NX

    def lev(x, levels=NZ):
        return _f32(cp, np.asarray(x).reshape(levels, ncol))

    shape3 = (NZ, NY, NX)
    mass = {"t": lev(a["t"]), "qv": lev(a["qv"]),
            "qc": lev(rng.uniform(0, 3e-5, shape3)), "qi": lev(rng.uniform(0, 2e-5, shape3)),
            "p": lev(a["p"]), "pi": lev(a["pi"]), "rho": lev(a["rho"]),
            "dz8w": lev(a["dz"]), "u": lev(rng.uniform(-10, 10, shape3)),
            "v": lev(rng.uniform(-10, 10, shape3))}
    iface = {"p8w": lev(a["pif"], NZ + 1),
             "w": lev(rng.uniform(-0.5, 1.5, (NZ + 1, NY, NX)), NZ + 1)}
    sc_host = np.empty((ncol, len(C.SC_FIELDS)), np.float32)
    for name, value in (("xland", rng.uniform(1, 2, ncol)), ("hpbl", rng.uniform(200, 2000, ncol)),
                        ("hfx", rng.uniform(-30, 300, ncol)), ("qfx", rng.uniform(0, 2e-4, ncol)),
                        ("dx", np.full(ncol, 9000.0)), ("dt", np.full(ncol, 45.0)),
                        ("pgcon", np.full(ncol, 0.55))):
        sc_host[:, C.SC_FIELDS.index(name)] = value
    ic_host = np.empty((ncol, len(C.IC_FIELDS)), np.int32)
    for name, value in (("stepcu", 1), ("ncloud", 2), ("dx_factor", 0)):
        ic_host[:, C.IC_FIELDS.index(name)] = value
    sc, ic = cp.asarray(sc_host), cp.asarray(ic_host)
    original = C.tile_columns
    rules = {"one block": lambda n, sms, **kw: C.BLOCK,
             "three blocks": lambda n, sms, **kw: 3 * C.BLOCK,
             "metal rule, one core": lambda n, sms, **kw: original(n, 1, metal=True, **{
                 k: v for k, v in kw.items() if k != "metal"}),
             "backend rule": original,
             "whole domain": lambda n, sms, **kw: n}

    def patch(rule):
        C.tile_columns = rule
        return lambda: setattr(C, "tile_columns", original)

    def launch():
        out = {n: cp.zeros((NZ, ncol), dtype=cp.float32) for n in C.LEVEL_OUTPUTS}
        sfc = cp.zeros((len(C.SURFACE_OUTPUTS), ncol), dtype=cp.float32)
        C.launch_columns(scheme, mass=mass, iface=iface, sc=sc, ic=ic, outputs=out,
                         sfc=sfc, nz=NZ, ncol=ncol, row_width=NX)
        assert np.isfinite(cp.asnumpy(out["rthcuten"])).all()
        return _digest({**out, "sfc": sfc})

    _assert_same(_runs(cp, launch, rules, patch))


@pytest.mark.gpu
def test_registry_cumulus_tile_keeps_every_word():
    """BMJ (cu_physics = 2) on the registry column-kernel route, every
    column of its committed WRF capture (arm 2, the feedback arm)."""
    cp = pytest.importorskip("cupy")
    import gzip
    import shutil
    import tempfile
    from pathlib import Path

    from test_bmj_wrf461_parity import DATA, _spec, io
    from woof.core import column_cumulus as C

    spec = _spec()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for gz in DATA.glob("*.bin.gz"):
            with gzip.open(gz, "rb") as src, open(tmp / gz.name[:-3], "wb") as dst:
                shutil.copyfileobj(src, dst)
        ncol, nz = io.read_cases(tmp / "cases.bin")
        arm = max(io.ARMS)
        fb, dt, stepcu = io.ARMS[arm]
        st = io.read_arm(tmp / f"stock-arm{arm}.bin", ncol, nz)[0]
    tables = C.column_tables(spec)
    original = C.registry_tile_columns
    rules = {"one block": lambda fn, n, block, **kw: block,
             "three blocks": lambda fn, n, block, **kw: 3 * block,
             "backend rule": original,
             "whole domain": lambda fn, n, block, **kw: n}

    def patch(rule):
        C.registry_tile_columns = rule
        return lambda: setattr(C, "registry_tile_columns", original)

    def launch():
        cld = cp.asarray(st["in_cldefi"])
        out = C.run_columns(
            spec, level={k: st["in_" + k] for k in ("t", "qv", "pmid", "pi", "rho", "dz8w")},
            interface={"pint": st["in_pint"]}, column={"xland": st["in_xland"]},
            carried={"cldefi": cld}, dt=dt, stepcu=stepcu, rad_feedback=fb,
            tables=tables)
        return _digest({**out, "cldefi": cld})

    _assert_same(_runs(cp, launch, rules, patch))


@pytest.mark.gpu
def test_eeps_tile_keeps_every_word():
    cp = pytest.importorskip("cupy")
    from woof.core import eeps_pbl as E
    from woof.core.physics_inventory import (eeps_workspace_floats,
                                             metal_column_tile_columns)

    rng = np.random.default_rng(11)
    a = _profile(rng)
    shape = (NZ, NY, NX)
    t = a["t"]
    col = {k: _f32(cp, v) for k, v in dict(
        u=rng.uniform(-10, 10, shape), v=rng.uniform(-10, 10, shape),
        t=t, qv=a["qv"], qc=rng.uniform(0, 3e-5, shape),
        qi=rng.uniform(0, 2e-5, shape), qr=rng.uniform(0, 1e-5, shape),
        qs=rng.uniform(0, 1e-5, shape), qg=rng.uniform(0, 1e-6, shape),
        p=a["p"], pi=a["pi"], rho=a["rho"],
        rthraten=rng.uniform(-2e-5, 1e-5, shape), dz8w=a["dz"]).items()}
    surf = {k: _f32(cp, v) for k, v in dict(
        psfc=a["psfc"], qsfc=rng.uniform(0.008, 0.018, (NY, NX)),
        tsk=t[0] + rng.uniform(-3, 6, (NY, NX)),
        ust=rng.uniform(0.05, 0.85, (NY, NX)),
        rmol=rng.uniform(-0.05, 0.02, (NY, NX)),
        wspd=rng.uniform(0.7, 15, (NY, NX)),
        hfx=rng.uniform(-30, 250, (NY, NX)),
        qfx=rng.uniform(-1e-5, 2e-4, (NY, NX))).items()}
    original = E.tile_columns
    rules = {"one block": lambda fn, n, nz=None: E._BLOCK,
             "three blocks": lambda fn, n, nz=None: 3 * E._BLOCK,
             "metal rule, one core": lambda fn, n, nz=None: metal_column_tile_columns(
                 1, n, E._BLOCK, block_bytes=eeps_workspace_floats(nz, E._BLOCK) * 4),
             "backend rule": original,
             "whole domain": lambda fn, n, nz=None: n}

    def patch(rule):
        E.tile_columns = rule
        return lambda: setattr(E, "tile_columns", original)

    def launch():
        carried = {k: cp.zeros(shape, dtype=cp.float32) for k in E.CARRIED}
        out = E.eeps_step(col, surf, carried, dt=30.0, first_step=True)
        assert np.isfinite(cp.asnumpy(out["rthblten"])).all()
        return _digest({**out, **carried})

    _assert_same(_runs(cp, launch, rules, patch))


@pytest.mark.gpu
@pytest.mark.parametrize(*_numerics_child.MODES)
def test_column_tiles_keep_every_word_in_every_numerics_mode(numerics, exact):
    _numerics_child.run(__file__, [
        "test_mrf_tile_keeps_every_word",
        "test_new_sas_tile_keeps_every_word",
        "test_registry_cumulus_tile_keeps_every_word",
        "test_eeps_tile_keeps_every_word"], numerics, exact)


def test_metal_tiles_are_one_launch_and_cuda_keeps_its_rules(monkeypatch):
    """On Metal each launcher takes metal_column_tile_columns (a 228 x 276
    domain in one launch on 40 cores, under the byte cap), the variable
    overrides it, and CUDA keeps its occupancy-bounded tiles."""
    cp = pytest.importorskip("cupy")
    import woof.backend as wb
    from woof.core import column_cumulus as C
    from woof.core import eeps_pbl as E
    from woof.core import mrf as M
    from cupy_backends.cuda.api import driver
    from woof.core import physics_inventory as P

    class _Device:
        attributes = {"MultiProcessorCount": 40}

    class _Fn:
        kernel = type("K", (), {"ptr": 0})()

    monkeypatch.setattr(cp.cuda, "Device", _Device)
    monkeypatch.delenv("WOOF_METAL_COLUMN_TILE_BLOCKS", raising=False)
    monkeypatch.setattr(wb, "active", lambda: "mlx")
    ncol = 228 * 276
    nsas = C.COLUMN_CUMULUS_SCHEMES[96]
    assert M.mrf_tile_columns(_Fn(), ncol, 49) == ncol
    # EEPS takes the Metal tile too: one launch at 49 levels.
    monkeypatch.setattr(driver, "occupancyMaxActiveBlocksPerMultiprocessor",
                        lambda *a: 1, raising=False)
    assert E.tile_columns(_Fn(), ncol, 49) == ncol
    assert E.tile_columns(_Fn(), 10**6) == 40 * 2048
    # New SAS's 50 slots hit the byte cap: whole blocks under it.
    capped_sas = C.tile_columns(ncol, 40, scheme=nsas, nz=49)
    assert capped_sas % C.BLOCK == 0 and capped_sas < ncol
    assert C.workspace_floats(nsas, 49, capped_sas) * 4 <= P.YSU_TILE_BYTES_CAP_METAL
    assert C.workspace_floats(nsas, 49, capped_sas + C.BLOCK) * 4 > P.YSU_TILE_BYTES_CAP_METAL
    assert C.registry_tile_columns(_Fn(), ncol, 128, block_bytes=4 * 30 * 49 * 128) == ncol
    # 2,048 columns per core whatever the block.
    assert M.mrf_tile_columns(_Fn(), 10**6) == 40 * 2048
    assert C.registry_tile_columns(_Fn(), 10**6, 128) == 40 * 2048
    # The byte cap holds a deep tile to whole blocks under it.
    capped = M.mrf_tile_columns(_Fn(), 10**6, 256)
    assert capped % P.MRF_BLOCK == 0
    assert P.mrf_workspace_floats(256, capped) * 4 <= P.YSU_TILE_BYTES_CAP_METAL
    assert P.mrf_workspace_floats(256, capped + P.MRF_BLOCK) * 4 > P.YSU_TILE_BYTES_CAP_METAL
    # A workspace-free kernel (zero bytes per block) sets no byte cap.
    assert C.registry_tile_columns(_Fn(), ncol, 128, block_bytes=0) == ncol
    monkeypatch.setenv("WOOF_METAL_COLUMN_TILE_BLOCKS", "4")
    assert M.mrf_tile_columns(_Fn(), ncol, 49) == 40 * 4 * 32
    assert C.tile_columns(ncol, 40, scheme=nsas, nz=49) == 40 * 2 * C.BLOCK
    for bad in ("0", "many"):
        monkeypatch.setenv("WOOF_METAL_COLUMN_TILE_BLOCKS", bad)
        with pytest.raises(ValueError, match="WOOF_METAL_COLUMN_TILE_BLOCKS"):
            M.mrf_tile_columns(_Fn(), ncol, 49)
    monkeypatch.delenv("WOOF_METAL_COLUMN_TILE_BLOCKS")
    monkeypatch.setattr(wb, "active", lambda: "cuda")
    assert C.tile_columns(ncol, 40, scheme=nsas, nz=49) == 40 * C.TILE_BLOCKS_PER_SM * C.BLOCK
    assert M.mrf_tile_columns(_Fn(), ncol) <= 40 * P.MRF_TILE_BLOCKS_PER_SM * P.MRF_BLOCK
    assert E.tile_columns(_Fn(), ncol, 49) == 40 * 1 * P.EEPS_BLOCK


def test_preflight_prices_the_metal_tiles():
    """A Metal profile prices the one-launch tile each launcher takes; a
    CUDA profile keeps its own pricing."""
    from types import SimpleNamespace

    from woof.core import physics_inventory as P
    from woof.core import preflight as F

    def exp(**run):
        base = dict(nx=276, ny=228, nz=49, bl_pbl_physics=0, cu_physics=0)
        base.update(run)
        return SimpleNamespace(domains=[SimpleNamespace(run=SimpleNamespace(**base))])

    cuda = SimpleNamespace(multiprocessor_count=40, backend="cuda")
    metal = SimpleNamespace(multiprocessor_count=40, backend="mlx")
    ncol = 276 * 228
    mrf = exp(bl_pbl_physics=99)
    assert F.mrf_column_workspace_bytes(mrf, profile=cuda) == (
        P.mrf_workspace_floats(49, 40 * P.MRF_TILE_BLOCKS_PER_SM * P.MRF_BLOCK) * 4)
    assert F.mrf_column_workspace_bytes(mrf, profile=metal) == P.mrf_workspace_floats(49, ncol) * 4
    eeps = exp(bl_pbl_physics=F.EEPS_PBL_SCHEME)
    assert F.eeps_column_workspace_bytes(eeps, profile=cuda) == (
        P.eeps_workspace_floats(49, 40 * P.EEPS_TILE_BLOCKS_PER_SM * P.EEPS_BLOCK) * 4)
    assert F.eeps_column_workspace_bytes(eeps, profile=metal) == P.eeps_workspace_floats(49, ncol) * 4
    from woof.core import column_cumulus as C
    nsas = exp(cu_physics=96)
    assert F.column_cumulus_workspace_bytes(nsas, profile=cuda) == C.workspace_bytes(96, 49, ncol, 40)
    assert F.column_cumulus_workspace_bytes(nsas, profile=metal) == (
        C.workspace_bytes(96, 49, ncol, 40, metal=True))
    assert (C.workspace_bytes(96, 49, ncol, 40, metal=True)
            > C.workspace_bytes(96, 49, ncol, 40))
    spec = C.column_kernel_spec(2)
    bmj = exp(cu_physics=2)
    assert (F.column_kernel_cumulus_workspace_bytes(bmj, profile=metal)
            > F.column_kernel_cumulus_workspace_bytes(bmj, profile=cuda))
    assert F.column_kernel_cumulus_workspace_bytes(bmj, profile=metal) >= (
        4 * spec.workspace_slots * 49 * ncol)
