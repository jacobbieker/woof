"""radtran_disort.cu's host build reproduces the committed output words.

tests/data/radtran/kernel_words.npz holds fixed binary64 solver inputs and
the words the host build produced (tools/uvspec_oracle/kernel_words.py).
The device builds are held to the same words in
tests/test_radtran_kernel_words_gpu.py, so CUDA and Metal agree word for
word with each other through these fixtures.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

WORDS = Path(__file__).parent / "data" / "radtran" / "kernel_words.npz"


def _have_cxx():
    return any(shutil.which(c) for c in ("clang++", "g++", "c++"))


@pytest.mark.skipif(not _have_cxx(), reason="needs a host C++ compiler")
def test_host_build_words():
    from tools.uvspec_oracle import kernel_words as kw
    d = np.load(WORDS)
    inp = {k[3:]: d[k] for k in d.files if k.startswith("in_")}
    for label, nstr, flags, angles in kw.RUNS:
        f, r = kw.run(inp, label, nstr, flags, angles, "host")
        assert np.array_equal(f.view(np.uint64), d[f"{label}_flux"].view(np.uint64)), label
        assert np.array_equal(r.view(np.uint64), d[f"{label}_rad"].view(np.uint64)), label


def test_work_size_matches_kernel():
    from woof.radtran import _host, solver
    if not _have_cxx():
        pytest.skip("needs a host C++ compiler")
    lib = _host.library()
    for nlyr in (1, 7, 60):
        for nstr in (2, 4, 8, 16, 32):
            assert lib.radtran_work_size_host(nlyr, nstr // 2, nstr) == solver.work_size(nlyr, nstr)


def test_hu_stamnes_rows_match_np_interp_words():
    """clouds._hu_prop evaluates every row at once; the words must be
    np.interp's, per row, over the whole table (this numpy's multiply-add)."""
    from woof.radtran import clouds
    rng = np.random.default_rng(7)
    lam = np.sort(rng.uniform(0.25, 4.0, 40))
    blocks = [{"r": (2.5, 12.5), "rows": np.column_stack((lam, rng.uniform(-3, 3, (40, 3))))},
              {"r": (12.5, 60.0), "rows": np.column_stack((lam, rng.uniform(-3, 3, (40, 3))))}]
    reff = np.where(rng.random(5000) < 0.7, 10.0, rng.uniform(2.5, 60.0, 5000))
    reff[:4] = (2.5, 12.5, 12.5000001, 60.0)

    def per_row(lam_um):
        out = np.zeros(reff.shape)
        for k, b in enumerate(blocks):
            lo, hi = b["r"]
            sel = (reff >= lo) & (reff <= hi) if k == 0 else (reff > lo) & (reff <= hi)
            rows, r = b["rows"], reff[sel]
            vals = rows[:, 1][None, :] * r[:, None] ** rows[:, 2][None, :] + rows[:, 3][None, :]
            out[sel] = np.array([np.interp(lam_um, rows[:, 0], v) for v in vals])
        return out

    for lam_um in (*rng.uniform(0.3, 3.9, 25), lam[0], lam[17], lam[-1], 0.1, 5.0):
        got = clouds._hu_prop(blocks, float(lam_um), reff)
        assert np.array_equal(got.view(np.uint64), per_row(float(lam_um)).view(np.uint64)), lam_um


@pytest.mark.skipif(not _have_cxx(), reason="needs a host C++ compiler")
def test_layer_optics_column_blocks_keep_the_words(monkeypatch):
    """driver._layer_optics_columns (column blocks on host threads) gives the
    single call's words, radiances and the IMS peaks included."""
    from woof.radtran import data, driver, model
    try:
        data.data_dir()
    except data.LibradtranDataMissing as exc:
        pytest.skip(str(exc))
    rng = np.random.default_rng(3)
    ncol, nz = 300, 20
    zw = rng.uniform(0, 2500, ncol)[:, None] + np.linspace(0, 20000, nz + 1)[None, :]
    pw = 101325.0 * np.exp(-zw / 7600.0)
    zm = 0.5 * (zw[:, 1:] + zw[:, :-1])
    t = 288.0 - 6.5e-3 * np.minimum(zm, 11000) + rng.normal(0, 1, (ncol, nz))
    kw = dict(z_w_m=zw, p_w_pa=pw, t_mass=t, qv_mass=0.01 * np.exp(-zm / 2500.0),
              umu0=rng.uniform(-0.2, 1.0, ncol), albedo=rng.uniform(0.05, 0.4, ncol),
              qc_mass=np.where((zm > 1000) & (zm < 3000) & (rng.random((ncol, nz)) < 0.3), 2e-4, 0.0),
              qi_mass=np.where((zm > 6000) & (zm < 10000) & (rng.random((ncol, nz)) < 0.3), 5e-5, 0.0),
              rho_dry=pw[:, :-1] / (287.0 * t), re_cloud_um=rng.uniform(4, 20, (ncol, nz)),
              re_ice_um=rng.uniform(10, 40, (ncol, nz)), day_of_year=172, backend="host")
    diag = model.Diagnostic(wavelengths=(450.0, 760.5), umu=(0.6, -0.8), phi=(0.0, 90.0))
    ref = model.compute(diag, **kw)
    monkeypatch.setattr(driver, "OPTICS_CHUNK", 64)
    monkeypatch.setattr(driver.os, "cpu_count", lambda: 2)     # blocks on any host
    got = model.compute(diag, **kw)
    assert sorted(got) == sorted(ref)
    for k in ref:
        a, b = np.asarray(ref[k], dtype=np.float64), np.asarray(got[k], dtype=np.float64)
        assert np.array_equal(a.view(np.uint64), b.view(np.uint64)), k
