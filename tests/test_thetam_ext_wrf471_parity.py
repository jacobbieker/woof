"""use_theta_m = 1 beyond Kessler, against compiled WRF v4.7.1 word for word.

tests/data/wrf471_thetam_ext is tools/bigstep_wrf471_oracle/
thetam_ext_build.sh's record: WRF v4.7.1's phy_prep and phy_prep_part2 and
vertical_diffusion_2's hflux block, extracted unchanged, gfortran -O0
-ffp-contract=off against glibc 2.43 (ubuntu:26.04 on pb), on the real WRF
patch of tests/data/wrf471_bigstep/state-real.npz (12 x 10 x 49).

* phy_prep_part2's RTHFTEN block (kernels/theta_m_cumulus.cu): the coupled
  pair decoupled by c1h*mut + c2h and converted (strict Grell-Freitas's
  route), and the conversion alone (every other route).
* vertical_diffusion_2's theta_m hflux (kernels/smag2d_theta_m.cu) under
  isfflx = 0, 1 and 2, with the HFX it writes.
* The microphysics window (woof.core.theta_m.microphysics_window) around
  stand-in schemes that finish as woof's adapters do -- the shared
  moist_physics_finish (Kessler, Lin, the WSMs, WDM6, Morrison, MY, NSSL,
  Thompson-aerosol, P3, FSBM, TEMPO, Ferrier) and classic Thompson's two
  fused dry finishes followed by finish_microphysics -- graded against the
  FIRST record's moist_physics_prep_em/moist_physics_finish_em
  (tests/data/wrf471_thetam), unclamped and clamped.
"""
from __future__ import annotations

import hashlib
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify import thetam_ext_oracle as oracle
from woof.verify import thetam_oracle

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "bigstep_wrf471_oracle"


def _module(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _receipt() -> str:
    return (oracle.ORACLE_DIR / "receipt.txt").read_text()


def test_record_is_the_pinned_wrf_source_on_glibc_243():
    receipt = _receipt()
    assert "commit f52c197ed39d12e087d02c50f412d90d418f6186" in receipt
    assert "GLIBC 2.43" in receipt
    assert "-O0 -ffp-contract=off" in receipt
    for relative, digest in _module("thetam_ext_build").SOURCES.items():
        assert f"{digest}  {relative}" in receipt, relative
    for name in ("thetam_ext_build.py", "thetam_ext_cases.py",
                 "thetam_cases.py", "thetam_ext_run.F90",
                 "thetam_ext_services.F90", "thetam_ext_build.sh"):
        digest = hashlib.sha256((TOOLS / name).read_bytes()).hexdigest()
        assert f"{digest}  tools/bigstep_wrf471_oracle/{name}" in receipt, name
    outputs = hashlib.sha256(
        (oracle.ORACLE_DIR / "thetam_ext_outputs.bin").read_bytes()).hexdigest()
    assert f"{outputs}  thetam_ext_outputs.bin" in receipt


def test_the_inputs_are_the_record_inputs_and_the_real_patch():
    cases = _module("thetam_ext_cases")
    inputs = oracle.load_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        cases.write(inputs, Path(tmp))
        digest = hashlib.sha256(
            (Path(tmp) / "thetam_ext_inputs.bin").read_bytes()).hexdigest()
    assert f"{digest}  thetam_ext_inputs.bin" in _receipt()
    rebuilt = cases.build(ROOT / "tests" / "data" / "wrf471_bigstep" / "state-real.npz")
    assert sorted(rebuilt) == sorted(inputs)
    for name, value in rebuilt.items():
        assert np.array_equal(value.view(np.uint32), inputs[name].view(np.uint32)), name


def test_the_record_moves_every_graded_word():
    """Each graded statement changes what it is handed: the conversion is
    not the identity, every isfflx arm moves the surface tendency, and the
    two RTHFTEN routes are distinct."""
    inputs = oracle.load_inputs()
    record = oracle.load_outputs(inputs=inputs)
    for case in ("coupled", "decoupled"):
        assert not np.array_equal(record[f"part2_{case}_rthften"], inputs["rthften"])
    assert not np.array_equal(record["part2_coupled_rthften"],
                              record["part2_decoupled_rthften"])
    for c in (0, 1, 2):
        rt = record[f"hflux_isfflx{c}_rt_tendf"]
        assert (rt[0] != inputs["t_tendf"][0]).mean() > 0.9
        assert np.array_equal(rt[1:], inputs["t_tendf"][1:])


@pytest.mark.gpu
@requires_gpu
def test_every_rthften_and_hflux_word_is_compiled_wrfs():
    inputs = oracle.load_inputs()
    record = oracle.load_outputs(inputs=inputs)
    rows = oracle.compare(oracle.port_outputs(inputs), record)
    differ = {name: row["differ"] for name, row in rows.items() if row["differ"]}
    assert differ == {}
    # The graded words: both RTHFTEN routes (and RQVFTEN's decoupling) whole,
    # hflux's surface plane and HFX for each isfflx.
    nz, ny, nx = inputs["t_m"].shape
    graded = 4 * nz * ny * nx + 3 * 2 * ny * nx
    assert graded == 24240
    assert sum(row["words"] for row in rows.values()) >= graded


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("finish", oracle.WINDOW_FINISHES)
@pytest.mark.parametrize("case", ("a", "b"))
def test_the_microphysics_window_is_moist_physics_finish_em(finish, case):
    inputs = thetam_oracle.load_inputs()
    record = thetam_oracle.load_outputs(inputs=inputs)
    out = oracle.window_outputs(inputs, record, finish, case)
    for name, value in out.items():
        a = np.ascontiguousarray(value, np.float32).view(np.uint32)
        b = np.ascontiguousarray(record[name], np.float32).view(np.uint32)
        assert a.shape == b.shape, name
        assert int((a != b).sum()) == 0, name
