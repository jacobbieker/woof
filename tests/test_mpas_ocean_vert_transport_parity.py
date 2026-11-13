"""MPAS-Ocean's vertical (ALE) transport velocity, word for word against v8.4.2, at RKIND 4 and 8.

The reference is MPAS-Model v8.4.2's own ``ocn_vert_transport_velocity_top``
(cut verbatim out of ``mpas_ocn_diagnostics.F``) with ``ocn_ALE_thickness``
(``mpas_ocn_thick_ale.F``, verbatim), compiled with gfortran -O0
-ffp-contract=off at MPAS's default single precision and at its double
precision, driven by ``tools/mpas_ocean_oracle/run_vert_transport.F90``. The
outputs are committed (tests/data/oracles/mpas_ocean_vert_transport,
PROVENANCE.md). When ``GFORTRAN`` (or gfortran on PATH) and
``WOOF_MPAS_V842_SOURCE`` are set, the fixture itself is rebuilt and compared.

THE BREAKAGES THESE PREVENT
---------------------------
1. A port that agrees with MPAS only to a tolerance, in either precision.
2. A case generator whose bytes drift. The input digests are pinned.
3. A fixture that misses a branch: z-star over the whole column or part of it,
   the min/max thickness limits, impermeable interfaces.
4. A gate that cannot fail: a mutated evaluation order or a skipped limit must
   produce mismatches.
5. The thickness filter silently skipped: refused by name.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.hex.errors import ConfigurationRefusal
from woof.verify import mpas_ocean_vert_transport_oracle as oo

ROOT = Path(__file__).resolve().parents[1]
BOTH = ("r4", "r8")
W = "diagnostics/vertAleTransportTop"


def test_case_inputs_are_the_recorded_bytes():
    recorded = dict(reversed(line.split()) for line in
                    (oo.FIXTURE_DIR / "inputs.sha256").read_text(encoding="utf-8").splitlines())
    assert recorded == {f"{case['name']}/{p}": oo.input_digest(case) for p in BOTH for case in oo.cases(p)}


@pytest.mark.parametrize("precision", BOTH)
def test_fixture_covers_the_branches(precision):
    fixture = oo.load_fixture(precision)
    cases = {case["name"]: case for case in oo.cases(precision)}
    assert set(fixture) == set(cases)
    for name, out in fixture.items():
        case = cases[name]
        w = out[W]
        nc = case["dims"]["nCells"]
        if case["config"]["config_vert_coord_movement"] == "impermeable_interfaces":
            assert (w == 0).all()
            continue
        mlc = case["pools"]["mesh"]["maxLevelCell"][:nc]
        assert (w[:nc, 0] == 0).all() and (w[np.arange(nc), mlc] == 0).all()
        assert np.count_nonzero(w[:nc]) > nc and np.isfinite(w).all()


def test_the_thickness_filter_is_refused_by_name():
    from woof.hex.mpas_ocean import vert_transport_v842 as vt
    vt.check_config(dict(oo.BASE_CONFIG))
    with pytest.raises(ConfigurationRefusal, match="config_use_freq_filtered_thickness"):
        vt.check_config(dict(oo.BASE_CONFIG, config_use_freq_filtered_thickness=True))


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
def test_gpu_matches_the_v842_oracle_word_for_word(precision):
    report = oo.measure(precision)
    bad = {case: {k: v for k, v in fields.items() if v} for case, fields in report.items()}
    assert not any(bad.values()), bad


@requires_gpu
@pytest.mark.parametrize("precision", BOTH)
@pytest.mark.parametrize("original, mutated, case_name", [
    # the min/max thickness limits skipped
    ("    if (!minMax) return;", "    return;", "zstar_min_max"),
    # z-star's share as a product with the reciprocal
    ("sshAle[k] = D(sshAle[k], thicknessSum);", "sshAle[k] = M(sshAle[k], D(C(1.0), thicknessSum));", "zstar"),
    # the thickness flux with the velocity and edge length taken first
    ("M(M(M(M(layerThicknessEdge[iEdge * nl + k], normalVelocity[iEdge * nl + k]),\n"
     "                                  dvEdge[iEdge]), sign), invAreaCell)",
     "M(M(M(layerThicknessEdge[iEdge * nl + k], M(normalVelocity[iEdge * nl + k],\n"
     "                                  dvEdge[iEdge])), sign), invAreaCell)", "zstar"),
    # the transport's two sources subtracted together
    ("w[k] = S(S(w[k + 1], divHu[iCell * nl + k]),\n"
     "                 D(S(aleThickness[iCell * nl + k], oldLayerThickness[iCell * nl + k]), dt));",
     "w[k] = S(w[k + 1], A(divHu[iCell * nl + k],\n"
     "                 D(S(aleThickness[iCell * nl + k], oldLayerThickness[iCell * nl + k]), dt)));",
     "fixed_bottom_weights"),
])
def test_a_mutated_port_is_caught(monkeypatch, precision, original, mutated, case_name):
    from woof.hex.mpas_ocean import vert_transport_v842
    real = vert_transport_v842.module_source
    assert real(precision).count(original) == 1
    monkeypatch.setattr(vert_transport_v842, "module_source", lambda p="r4": real(p).replace(original, mutated))
    vert_transport_v842._module_on.cache_clear()
    try:
        case = next(c for c in oo.cases(precision) if c["name"] == case_name)
        got = oo.run_gpu(case)
        assert oo.word_mismatches(got[W], oo.load_fixture(precision)[case_name][W]) > 0
    finally:
        vert_transport_v842._module_on.cache_clear()


def _oracle_toolchain() -> tuple[str, Path] | None:
    source = os.environ.get("WOOF_MPAS_V842_SOURCE")
    gfortran = os.environ.get("GFORTRAN") or shutil.which("gfortran")
    if not source or not gfortran or not Path(source, "src", "core_ocean").is_dir():
        return None
    return gfortran, Path(source)


@pytest.mark.skipif(_oracle_toolchain() is None,
                    reason="the MPAS-Ocean oracle build needs gfortran (pixi -e mpas-oracle) and "
                           "WOOF_MPAS_V842_SOURCE naming an MPAS-Model v8.4.2 tree")
def test_the_oracle_rebuilt_here_writes_the_fixture(tmp_path):
    gfortran, source = _oracle_toolchain()
    subprocess.run(["bash", str(ROOT / "tools" / "mpas_ocean_oracle" / "build.sh"), str(source),
                    str(tmp_path / "build")], check=True, env={**os.environ, "GFORTRAN": gfortran})
    subprocess.run([sys.executable, str(ROOT / "tools" / "mpas_ocean_oracle" / "make_fixture.py"),
                    str(tmp_path / "build"), "--piece", "vert_transport", "--out", str(tmp_path / "fixture")],
                   check=True)
    for precision in BOTH:
        fresh = np.load(tmp_path / "fixture" / f"oracle-{precision}.npz")
        pinned = np.load(oo.FIXTURE_DIR / f"oracle-{precision}.npz")
        assert sorted(fresh.files) == sorted(pinned.files)
        for key in pinned.files:
            assert oo.word_mismatches(fresh[key], pinned[key]) == 0, (precision, key)
