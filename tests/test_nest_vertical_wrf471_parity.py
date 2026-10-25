"""WRF v4.7.1 vertical grid nesting: the CPU twin and the GPU kernels
against WRF's own Fortran.

The fixture is built by ``tools/vert_nest_wrf471_oracle/build.sh`` from
WRF's own statements, byte-unmodified and pinned by sha256:
``share/interp_fcn.F`` (``vert_interp_vert_nesting``, ``_w``,
``inter_wrf_copy``, ``coeff_mon_wrf_copy``), ``dyn_em/nest_init_utils.F``
(``compute_vcoord_1d_coeffs`` and the dn..cfn1 lines of
``vert_cor_vertical_nesting_arbitrary``) and
``external/RSL_LITE/force_domain_em_part2.F`` (the reference heights and
the rebalance).  Four cases: a parent of 40 levels refined to 79 under
``hypsometric_opt`` 2 and 1, a parent of 60 coarsened to 30, and two equal
ladders.

Both ports must equal every recorded word: max ULP 0, raw words equal
(so a signed zero is a zero of the same sign).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parent / "data" / "oracles" / "vert_nest"
TOOLS = Path(__file__).resolve().parents[1] / "tools" / "vert_nest_wrf471_oracle"
CASES = ("refine_h2", "refine_h1", "coarsen_h2", "same_h2")
#: The rebalanced fields live on the intermediate grid's mass columns
#: (WRF loops i < ide, j < jde); the interpolated u/w on every column.
NY, NX = 4, 5


def _case(name):
    from woof.verify.urban_oracle import load_case
    return load_case(ROOT / name)


def _plan(o):
    from woof.core.nest_vertical import VerticalNestPlan

    nzp = o["p_c3h"].size - 1
    nzn = o["n_c3h"].size - 1

    def ladder(prefix, n):
        out = {k: o[prefix + k][:n] for k in
               ("c1h", "c2h", "c3h", "c4h", "dnw", "rdnw", "rdn")}
        out.update({k: o[prefix + k][:n + 1] for k in
                    ("c1f", "c2f", "c3f", "c4f")})
        return out

    parent = ladder("p_", nzp)
    parent["cf"] = tuple(o["p_cf"])
    base = o["n_base"]
    plan = VerticalNestPlan.from_coefficients(
        parent=parent, child=ladder("n_", nzn), p_top=base[0],
        hypsometric_opt=int(o["hypsometric_opt"]), base_temp=base[2],
        base={"tiso": base[1], "tlp": base[3], "p00": base[4],
              "p_strat": base[5], "tlp_strat": base[6]})
    return plan, nzp, nzn


def _kji(array, levels):
    """WRF (i, k, j) -> woof (k, j, i), the first ``levels`` levels."""
    return np.ascontiguousarray(np.transpose(array[:, :levels, :], (1, 2, 0)))


def _words(array):
    return np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)


def _same(port, ref):
    return np.array_equal(_words(port), _words(ref))


def test_the_fixture_is_the_one_the_harness_wrote():
    lines = (ROOT / "FIXTURES.sha256").read_text().splitlines()
    assert lines
    for line in lines:
        digest, name = line.split(None, 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest


def test_the_extracted_blocks_are_pinned_to_the_pinned_sources():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "vert_nest_extract", TOOLS / "extract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pinned = {line.split()[1]: line.split()[0]
              for line in (TOOLS / "SOURCES.sha256").read_text().splitlines()
              if line and not line.startswith("#")}
    for path, digest in module.FILES.items():
        assert pinned[path] == digest
    assert (ROOT / "SOURCES.sha256").read_text().split() == [
        token for line in (TOOLS / "SOURCES.sha256").read_text().splitlines()
        if line and not line.startswith("#") for token in line.split()]


@pytest.mark.parametrize("name", CASES)
def test_the_reference_heights_are_wrfs(name):
    o = _case(name)
    plan, _, _ = _plan(o)
    for key in ("alt_w_c", "alt_u_c", "alt_w_n", "alt_u_n"):
        assert _same(getattr(plan, key), o[key]), key


@pytest.mark.parametrize("name", CASES)
def test_the_cpu_twin_is_wrfs_word_for_word(name):
    o = _case(name)
    plan, nzp, nzn = _plan(o)
    assert _same(plan.interp_cpu(_kji(o["u_in"], nzp)), _kji(o["u_out"], nzn))
    assert _same(plan.interp_cpu(_kji(o["w_in"], nzp + 1)),
                 _kji(o["w_out"], nzn + 1))
    t = plan.interp_cpu(_kji(o["t_in"], nzp))
    qv = plan.interp_cpu(_kji(o["qv_in"], nzp))
    ph = plan.interp_cpu(_kji(o["ph_in"], nzp + 1))
    scratch = {}
    t, qv, ph = plan.rebalance_cpu(t, qv, ph, o["mub"].T, o["mu_2"].T,
                                   o["ht"].T, scratch=scratch)
    cut = (slice(None), slice(0, NY), slice(0, NX))
    for port, ref, levels in (
            (t, "t_out", nzn), (qv, "qv_out", nzn), (ph, "ph_out", nzn + 1),
            (scratch["pb"], "pb_out", nzn), (scratch["alb"], "alb_out", nzn),
            (scratch["phb"], "phb_out", nzn + 1)):
        assert _same(port[cut], _kji(o[ref], levels)[cut]), ref


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("name", CASES)
def test_the_gpu_kernels_are_wrfs_word_for_word(name):
    import cupy as cp

    o = _case(name)
    plan, nzp, nzn = _plan(o)
    u = plan.interp(cp.asarray(_kji(o["u_in"], nzp)))
    w = plan.interp(cp.asarray(_kji(o["w_in"], nzp + 1)))
    assert _same(cp.asnumpy(u), _kji(o["u_out"], nzn))
    assert _same(cp.asnumpy(w), _kji(o["w_out"], nzn + 1))
    t = plan.interp(cp.asarray(_kji(o["t_in"], nzp)))
    qv = plan.interp(cp.asarray(_kji(o["qv_in"], nzp)))
    ph = plan.interp(cp.asarray(_kji(o["ph_in"], nzp + 1)))
    scratch = {}
    plan.rebalance(t, qv, ph, cp.asarray(o["mub"].T), cp.asarray(o["mu_2"].T),
                   cp.asarray(o["ht"].T), scratch=scratch)
    cut = (slice(None), slice(0, NY), slice(0, NX))
    for port, ref, levels in (
            (t, "t_out", nzn), (qv, "qv_out", nzn), (ph, "ph_out", nzn + 1),
            (scratch["pb"], "pb_out", nzn), (scratch["alb"], "alb_out", nzn),
            (scratch["phb"], "phb_out", nzn + 1)):
        assert _same(cp.asnumpy(port)[cut], _kji(o[ref], levels)[cut]), ref


@requires_gpu
@pytest.mark.gpu
def test_the_gpu_and_cpu_agree_on_a_wide_random_slab():
    """Beyond the fixture's 30 columns: 3,000 random columns, every
    kernel against the twin, raw words."""
    import cupy as cp

    o = _case("refine_h2")
    plan, nzp, nzn = _plan(o)
    rng = np.random.default_rng(7)
    base_t = _kji(o["t_in"], nzp)[:, :1, :1]
    t_in = (base_t * (1 + 1e-3 * rng.standard_normal((nzp, 50, 60)))).astype(
        np.float32)
    qv_in = np.abs(_kji(o["qv_in"], nzp)[:, :1, :1] * (
        1 + 0.1 * rng.standard_normal((nzp, 50, 60)))).astype(np.float32)
    ph_in = (_kji(o["ph_in"], nzp + 1)[:, :1, :1] * (
        1 + 0.1 * rng.standard_normal((nzp + 1, 50, 60)))).astype(np.float32)
    mub = (o["mub"][0, 0] + 2000 * rng.standard_normal((50, 60))).astype(
        np.float32)
    mu = (100 * rng.standard_normal((50, 60))).astype(np.float32)
    ht = np.abs(1000 * rng.standard_normal((50, 60))).astype(np.float32)
    cpu = plan.rebalance_cpu(plan.interp_cpu(t_in), plan.interp_cpu(qv_in),
                             plan.interp_cpu(ph_in), mub, mu, ht)
    t = plan.interp(cp.asarray(t_in))
    qv = plan.interp(cp.asarray(qv_in))
    ph = plan.interp(cp.asarray(ph_in))
    plan.rebalance(t, qv, ph, cp.asarray(mub), cp.asarray(mu), cp.asarray(ht))
    for port, ref in zip((t, qv, ph), cpu):
        assert _same(cp.asnumpy(port), ref)


@requires_gpu
@pytest.mark.gpu
def test_the_composed_unit_compiles_within_the_priced_frame():
    import cupy as cp

    from woof.core import nest_vertical as nv
    from woof.core.preflight import CHAINED_TRANSLATION_UNIT_FRAMES

    module = nv._module(cp.cuda.Device().id)
    priced = CHAINED_TRANSLATION_UNIT_FRAMES["nest_vertical_composed"]
    for name in ("nest_vert_interp", "nest_vert_rebalance"):
        frame = int(cp.cuda.driver.funcGetAttribute(
            cp.cuda.driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
            module.get_function(name).ptr))
        assert frame <= priced.max_local_size_bytes, name


def test_the_composed_source_is_the_pinned_kernel():
    from woof.core import nest_vertical as nv

    source = nv.module_source()
    kernel = (Path(nv.__file__).parent / "kernels" / "nest_vertical.cu"
              ).read_text(encoding="utf-8")
    assert source.endswith(kernel)
    assert "-fmad=false" in nv.MODULE_OPTIONS
    assert "--ftz=false" in nv.MODULE_OPTIONS
